"""
FTS query translator for converting between different full-text search syntaxes.

This module handles the translation between SQLite FTS5 queries and
PostgreSQL tsquery syntax, ensuring that search functionality works
consistently across different database backends.
"""

import re
from typing import Optional

from loguru import logger

MAX_FTS_QUERY_LENGTH = 8192
SQLITE_FTS_OPERATOR_TOKENS = {'AND', 'OR', 'NEAR', 'NOT'}
SQLITE_FTS_SAFE_TOKEN_RE = re.compile(r'^[A-Za-z0-9_]+\*?$')
SQLITE_FTS_COLUMN_TOKEN_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*):(.*)$')


def _is_ascii_identifier_start(char: str) -> bool:
    return char == '_' or 'A' <= char <= 'Z' or 'a' <= char <= 'z'


def _is_ascii_identifier_char(char: str) -> bool:
    return _is_ascii_identifier_start(char) or '0' <= char <= '9'


def _quoted_sqlite_token_start(query: str, start: int) -> int:
    """Return the quote index for phrase tokens, or -1 for plain tokens."""
    index = start + 1 if query[start] == '-' else start
    if index >= len(query):
        return -1
    if query[index] == '"':
        return index
    if not _is_ascii_identifier_start(query[index]):
        return -1

    index += 1
    while index < len(query) and _is_ascii_identifier_char(query[index]):
        index += 1

    if index + 1 < len(query) and query[index] == ':' and query[index + 1] == '"':
        return index + 1
    return -1


def _quoted_sqlite_token_end(query: str, quote_index: int) -> int:
    """Return the end-exclusive phrase token index, honoring doubled quotes."""
    index = quote_index + 1
    while index < len(query):
        if query[index] != '"':
            index += 1
            continue
        if index + 1 < len(query) and query[index + 1] == '"':
            index += 2
            continue
        return index + 1
    return -1


def _tokenize_sqlite_fts_query(query: str) -> list[str]:
    """Tokenize SQLite FTS5 input linearly without regex backtracking."""
    parts: list[str] = []
    index = 0
    query_length = len(query)

    while index < query_length:
        while index < query_length and query[index].isspace():
            index += 1
        if index >= query_length:
            break

        quote_index = _quoted_sqlite_token_start(query, index)
        if quote_index != -1:
            token_end = _quoted_sqlite_token_end(query, quote_index)
            if token_end != -1:
                parts.append(query[index:token_end])
                index = token_end
                continue

        token_start = index
        while index < query_length and not query[index].isspace():
            index += 1
        parts.append(query[token_start:index])

    return parts


class FTSQueryTranslator:
    """
    Translates full-text search queries between different database syntaxes.

    Supports conversion between:
    - SQLite FTS5 MATCH syntax
    - PostgreSQL to_tsquery syntax
    """

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Normalize and bound query length before regex processing."""
        if not query:
            return ''
        normalized = query.strip()
        if not normalized:
            return ''
        if len(normalized) > MAX_FTS_QUERY_LENGTH:
            logger.warning(
                'FTS query length {} exceeds max {}; truncating',
                len(normalized),
                MAX_FTS_QUERY_LENGTH,
            )
            normalized = normalized[:MAX_FTS_QUERY_LENGTH]
        return normalized

    @staticmethod
    def sqlite_to_postgres(fts5_query: str, language: str = 'english') -> str:
        """
        Convert SQLite FTS5 query to PostgreSQL tsquery format.

        FTS5 syntax -> PostgreSQL syntax:
        - word1 word2 -> word1 & word2 (AND)
        - word1 OR word2 -> word1 | word2 (OR)
        - "exact phrase" -> 'exact phrase' (phrase)
        - word* -> word:* (prefix)
        - -word -> !word (NOT)
        - word1 NEAR word2 -> word1 <-> word2 (proximity)

        Args:
            fts5_query: SQLite FTS5 query string
            language: PostgreSQL text search configuration

        Returns:
            PostgreSQL tsquery string
        """
        query = FTSQueryTranslator._sanitize_query(fts5_query)
        if not query:
            return ''

        # Handle quoted phrases first
        # "exact phrase" -> 'exact phrase'
        phrases = []
        phrase_pattern = r'"([^"]*)"'

        def save_phrase(match):
            phrase = match.group(1)
            placeholder = f"__PHRASE_{len(phrases)}__"
            phrases.append(phrase.replace(' ', ' <-> '))
            return placeholder

        query = re.sub(phrase_pattern, save_phrase, query)

        # Handle wildcards
        # token* -> token:*
        # Support hyphens and underscores in tokens
        query = re.sub(r'([A-Za-z0-9_\-]+)\*', r'\1:*', query)

        # Handle NOT operator
        # -token -> !token
        query = re.sub(r'\s+-([A-Za-z0-9_\-]+)', r' !\1', query)
        query = re.sub(r'^-([A-Za-z0-9_\-]+)', r'!\1', query)

        # Handle NEAR operator
        # token1 NEAR token2 -> token1 <-> token2
        query = re.sub(r'([A-Za-z0-9_\-]+)\s+NEAR\s+([A-Za-z0-9_\-]+)', r'\1 <-> \2', query, flags=re.IGNORECASE)

        # Handle OR operator (already uses |)
        query = re.sub(r'\s+OR\s+', ' | ', query, flags=re.IGNORECASE)

        # Handle AND operator (implicit in FTS5, explicit in PostgreSQL)
        # Replace spaces between words with &
        # But don't replace spaces that are part of operators
        words = query.split()
        result_parts = []

        skip_next = False
        for i, word in enumerate(words):
            if skip_next:
                skip_next = False
                continue

            if word in ('|', '&', '<->', '!') or i < len(words) - 1 and words[i + 1] in ('|', '<->'):
                result_parts.append(word)
            elif i < len(words) - 1 and words[i + 1] not in ('|', '&', '<->', '!'):
                result_parts.append(word + ' &')
            else:
                result_parts.append(word)

        query = ' '.join(result_parts)

        # Restore phrases
        for i, phrase in enumerate(phrases):
            query = query.replace(f"__PHRASE_{i}__", f"({phrase})")

        # Clean up extra spaces and operators
        query = re.sub(r'\s+', ' ', query)
        query = re.sub(r'&\s+&', '&', query)
        query = re.sub(r'\|\s+\|', '|', query)
        query = query.strip(' &|')

        return query

    @staticmethod
    def postgres_to_sqlite(tsquery: str) -> str:
        """
        Convert PostgreSQL tsquery to SQLite FTS5 format.

        PostgreSQL syntax -> FTS5 syntax:
        - word1 & word2 -> word1 word2 (AND is implicit)
        - word1 | word2 -> word1 OR word2
        - !word -> -word (NOT)
        - word:* -> word* (prefix)
        - word1 <-> word2 -> word1 NEAR word2 (proximity)

        Args:
            tsquery: PostgreSQL tsquery string

        Returns:
            SQLite FTS5 query string
        """
        query = FTSQueryTranslator._sanitize_query(tsquery)
        if not query:
            return ''

        # Handle prefix searches
        # word:* -> word*
        query = re.sub(r'(\w+):\*', r'\1*', query)

        # Handle NOT operator
        # !word -> -word
        query = re.sub(r'!(\w+)', r'-\1', query)

        # Handle proximity operator
        # word1 <-> word2 -> word1 NEAR word2
        query = re.sub(r'(\w+)\s*<->\s*(\w+)', r'\1 NEAR \2', query)

        # Handle OR operator
        # | -> OR
        query = re.sub(r'\s*\|\s*', ' OR ', query)

        # Handle AND operator (remove it, as it's implicit in FTS5)
        # & -> space
        query = re.sub(r'\s*&\s*', ' ', query)

        # Handle parentheses (keep them for grouping)
        # Clean up extra spaces
        query = re.sub(r'\s+', ' ', query)
        query = query.strip()

        return query

    @staticmethod
    def _quote_sqlite_fts_token(token: str) -> str:
        """Quote FTS5 tokens that contain punctuation with query semantics."""
        if not token or SQLITE_FTS_SAFE_TOKEN_RE.fullmatch(token):
            return token

        suffix = '*' if token.endswith('*') and len(token) > 1 else ''
        token_body = token[:-1] if suffix else token
        escaped = token_body.replace('"', '""')
        return f'"{escaped}"{suffix}'

    @staticmethod
    def _normalize_sqlite_query(
        query: str,
        column_aliases: Optional[dict[str, str]] = None,
    ) -> str:
        """Protect plain text SQLite FTS5 searches from punctuation parsing."""
        parts = _tokenize_sqlite_fts_query(query)
        normalized_parts = []
        last_allows_not = False
        has_operand = False

        for part in parts:
            if part in SQLITE_FTS_OPERATOR_TOKENS:
                normalized_parts.append(part)
                last_allows_not = part != 'NOT' and has_operand
                continue

            if part.startswith('"') and part.endswith('"'):
                normalized_parts.append(part)
                last_allows_not = True
                has_operand = True
                continue

            if part.startswith('-') and len(part) > 1:
                body = part[1:]
                column_match = SQLITE_FTS_COLUMN_TOKEN_RE.fullmatch(body)
                if column_match:
                    column, value = column_match.groups()
                    if column_aliases:
                        column = column_aliases.get(column.lower())
                    if column:
                        operand = (
                            f'{column}:{value}'
                            if value.startswith('"') and value.endswith('"')
                            else f'{column}:{FTSQueryTranslator._quote_sqlite_fts_token(value)}'
                        )
                    else:
                        operand = (
                            value
                            if value.startswith('"') and value.endswith('"')
                            else FTSQueryTranslator._quote_sqlite_fts_token(value)
                        )
                elif body.startswith('"') and body.endswith('"'):
                    operand = body
                elif last_allows_not:
                    operand = FTSQueryTranslator._quote_sqlite_fts_token(body)
                else:
                    operand = FTSQueryTranslator._quote_sqlite_fts_token(part)
                if last_allows_not:
                    normalized_parts.extend(['NOT', operand])
                else:
                    normalized_parts.append(operand)
                last_allows_not = True
                has_operand = True
                continue

            column_match = SQLITE_FTS_COLUMN_TOKEN_RE.fullmatch(part)
            if column_match:
                column, value = column_match.groups()
                if column_aliases:
                    column = column_aliases.get(column.lower())
                    if not column:
                        normalized_parts.append(
                            value
                            if value.startswith('"') and value.endswith('"')
                            else FTSQueryTranslator._quote_sqlite_fts_token(value)
                        )
                        last_allows_not = True
                        has_operand = True
                        continue
                normalized_parts.append(
                    f'{column}:{value}'
                    if value.startswith('"') and value.endswith('"')
                    else f'{column}:{FTSQueryTranslator._quote_sqlite_fts_token(value)}'
                )
                last_allows_not = True
                has_operand = True
                continue

            normalized_parts.append(FTSQueryTranslator._quote_sqlite_fts_token(part))
            last_allows_not = True
            has_operand = True

        return ' '.join(normalized_parts)

    @staticmethod
    def normalize_query(
        query: str,
        backend: str,
        sqlite_column_aliases: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Normalize a query for a specific backend.

        This function attempts to detect the query format and convert
        it to the appropriate backend syntax.

        Args:
            query: Search query in unknown format
            backend: Target backend ('sqlite' or 'postgresql')
            sqlite_column_aliases: Optional SQLite FTS5 column name aliases

        Returns:
            Normalized query for the target backend
        """
        query = FTSQueryTranslator._sanitize_query(query)
        if not query:
            return ''

        # Try to detect query format
        is_postgres = any(op in query for op in ['&', '|', '<->', ':*', '!'])
        is_sqlite = any(op in query for op in ['NEAR', 'OR']) or query.startswith('-')

        if backend == 'postgresql':
            if is_sqlite and not is_postgres:
                return FTSQueryTranslator.sqlite_to_postgres(query)
            elif not is_postgres:
                # Assume simple word query, convert to PostgreSQL
                words = query.split()
                return ' & '.join(words)
            return query

        elif backend == 'sqlite':
            if is_postgres and not is_sqlite:
                query = FTSQueryTranslator.postgres_to_sqlite(query)
            return FTSQueryTranslator._normalize_sqlite_query(query, sqlite_column_aliases)

        return query

    @staticmethod
    def extract_search_terms(query: str) -> list[str]:
        """
        Extract individual search terms from a query.

        Args:
            query: FTS query in any format

        Returns:
            List of search terms
        """
        query = FTSQueryTranslator._sanitize_query(query)
        if not query:
            return []

        # Remove operators
        cleaned = re.sub(r'[&|!<>\-*:()]', ' ', query)
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Extract quoted phrases
        phrases = re.findall(r'"([^"]*)"', query)

        # Extract individual words
        words = cleaned.split()

        # Combine and deduplicate
        terms = list(set(words + phrases))
        return [term for term in terms if term]


class FTSRankNormalizer:
    """
    Normalizes ranking scores between different FTS implementations.

    SQLite FTS5 uses BM25 ranking, while PostgreSQL uses ts_rank.
    This class provides methods to normalize scores for consistent
    ranking across backends.
    """

    @staticmethod
    def normalize_score(
        score: float,
        backend: str,
        max_score: Optional[float] = None
    ) -> float:
        """
        Normalize a ranking score to [0, 1] range.

        Args:
            score: Raw ranking score
            backend: Backend that generated the score
            max_score: Maximum score in result set (for normalization)

        Returns:
            Normalized score between 0 and 1
        """
        if backend == 'sqlite':
            # SQLite FTS5 BM25 scores are typically negative
            # More negative = better match
            if score >= 0:
                return 0.0

            # Convert negative score to positive
            # and normalize
            positive_score = abs(score)

            if max_score:
                max_positive = abs(max_score) if max_score < 0 else 1.0
                return min(1.0, positive_score / max_positive)
            else:
                # Use empirical range for BM25
                return min(1.0, positive_score / 10.0)

        elif backend == 'postgresql':
            # PostgreSQL ts_rank returns positive scores
            # Higher = better match
            if score <= 0:
                return 0.0

            if max_score and max_score > 0:
                return min(1.0, score / max_score)
            else:
                # ts_rank typically returns values < 1
                return min(1.0, score)

        else:
            # Unknown backend, assume linear scale
            if max_score and max_score != 0:
                return abs(score / max_score)
            return min(1.0, abs(score))

    @staticmethod
    def compare_rankings(
        results_a: list[tuple[str, float]],
        results_b: list[tuple[str, float]],
        backend_a: str,
        backend_b: str
    ) -> float:
        """
        Compare ranking consistency between two backends.

        Args:
            results_a: List of (doc_id, score) from backend A
            results_b: List of (doc_id, score) from backend B
            backend_a: Name of backend A
            backend_b: Name of backend B

        Returns:
            Similarity score between 0 and 1
        """
        # Create rank dictionaries
        rank_a = {doc_id: i for i, (doc_id, _) in enumerate(results_a)}
        rank_b = {doc_id: i for i, (doc_id, _) in enumerate(results_b)}

        # Find common documents
        common_docs = set(rank_a.keys()) & set(rank_b.keys())

        if not common_docs:
            return 0.0

        # Calculate Spearman's rank correlation
        n = len(common_docs)
        rank_diff_squared = sum(
            (rank_a[doc] - rank_b[doc]) ** 2
            for doc in common_docs
        )

        # Spearman's rho
        rho = 1 - (6 * rank_diff_squared) / (n * (n**2 - 1))

        # Convert to [0, 1] where 1 = perfect agreement
        return (rho + 1) / 2


# Example usage and testing
if __name__ == "__main__":
    # Test query translations
    translator = FTSQueryTranslator()

    test_queries = [
        ('python programming', 'postgresql'),
        ('"exact phrase" near tutorial', 'postgresql'),
        ('python* OR java*', 'postgresql'),
        ('-deprecated python', 'postgresql'),
        ('python & java | ruby', 'sqlite'),
        ('!old <-> version:*', 'sqlite'),
    ]

    print("Query Translation Tests:")
    print("-" * 50)

    for query, target in test_queries:
        result = translator.normalize_query(query, target)
        print(f"Original: {query}")
        print(f"Target: {target}")
        print(f"Result: {result}")
        print()

    # Test rank normalization
    normalizer = FTSRankNormalizer()

    sqlite_scores = [(-5.2, 'doc1'), (-3.8, 'doc2'), (-7.1, 'doc3')]
    pg_scores = [(0.8, 'doc1'), (0.6, 'doc2'), (0.9, 'doc3')]

    print("Rank Normalization Tests:")
    print("-" * 50)

    for score, doc_id in sqlite_scores:
        normalized = normalizer.normalize_score(score, 'sqlite', max_score=-7.1)
        print(f"SQLite {doc_id}: {score} -> {normalized:.2f}")

    print()

    for score, doc_id in pg_scores:
        normalized = normalizer.normalize_score(score, 'postgresql', max_score=0.9)
        print(f"PostgreSQL {doc_id}: {score} -> {normalized:.2f}")
