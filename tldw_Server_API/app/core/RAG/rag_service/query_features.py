# query_features.py
"""
Advanced query features for the RAG service.

This module provides query understanding, intent detection, query rewriting,
and other advanced query processing capabilities.
"""

import contextlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, cast

import nltk
from loguru import logger
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

from tldw_Server_API.app.core.testing import (
    env_flag_enabled,
    is_explicit_pytest_runtime,
    is_test_mode,
)


# --- NLTK downloads guarded by timeout ---
def _download_with_timeout(resource: str, timeout_s: int = 60) -> bool:
    """Attempt to download an NLTK resource with a timeout.

    Returns True on success, False on timeout/failure. Uses a daemon thread so
    it never blocks process shutdown if the network hangs.
    """
    import queue
    import threading

    q: queue.Queue[bool] = queue.Queue(maxsize=1)

    def _runner():
        ok = False
        try:
            ok = nltk.download(resource, quiet=True)
        except Exception:  # pragma: no cover - defensive
            logger.warning("NLTK download error; continuing without resource")
            ok = False
        with contextlib.suppress(Exception):
            q.put_nowait(ok)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        logger.warning(f"NLTK download for '{resource}' timed out after {timeout_s}s; proceeding without it")
        return False
    try:
        return bool(q.get_nowait())
    except Exception:
        return False

# Ensure NLTK resources are present, but avoid hangs by enforcing timeouts.
# In tests or when explicitly disabled, skip downloads and degrade gracefully.
_TEST_MODE = is_test_mode()
_DISABLE_NLTK_DOWNLOADS = env_flag_enabled("DISABLE_NLTK_DOWNLOADS")
_RUNNING_PYTEST = is_explicit_pytest_runtime()
_FORCE_ALLOW_NLTK = env_flag_enabled("ALLOW_NLTK_DOWNLOADS")
_ALLOW_NLTK_DOWNLOADS = _FORCE_ALLOW_NLTK or not (_TEST_MODE or _DISABLE_NLTK_DOWNLOADS or _RUNNING_PYTEST)

def _ensure_resource(path: str, resource_key: str) -> bool:
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        if _ALLOW_NLTK_DOWNLOADS:
            ok = _download_with_timeout(resource_key, timeout_s=60)
            if not ok:
                logger.warning(f"NLTK resource '{resource_key}' unavailable; continuing without it")
            return ok
        else:
            logger.info(
                f"Skipping NLTK download for '{resource_key}' (TEST_MODE={_TEST_MODE}, DISABLE_NLTK_DOWNLOADS={_DISABLE_NLTK_DOWNLOADS}, PYTEST={_RUNNING_PYTEST}); set ALLOW_NLTK_DOWNLOADS=1 to override"
            )
            return False

_HAS_PUNKT = _ensure_resource('tokenizers/punkt', 'punkt')
_HAS_WORDNET = _ensure_resource('corpora/wordnet', 'wordnet')
_HAS_STOPWORDS = _ensure_resource('corpora/stopwords', 'stopwords')


class QueryIntent(Enum):
    """Types of query intent."""
    FACTUAL = "factual"          # Looking for specific facts
    EXPLORATORY = "exploratory"   # Broad exploration of topic
    COMPARATIVE = "comparative"   # Comparing multiple things
    PROCEDURAL = "procedural"     # How-to questions
    DEFINITIONAL = "definitional" # What is X questions
    CAUSAL = "causal"            # Why/cause questions
    TEMPORAL = "temporal"         # When/timeline questions
    ANALYTICAL = "analytical"     # Analysis/evaluation questions


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"      # Single concept
    MODERATE = "moderate"  # 2-3 concepts
    COMPLEX = "complex"    # Multiple interrelated concepts


@dataclass
class QueryAnalysis:
    """Analysis results for a query."""
    original_query: str
    cleaned_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    key_terms: list[str]
    entities: list[str]
    temporal_refs: list[str]
    question_type: Optional[str] = None
    domain: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryRewrite:
    """A rewritten version of a query."""
    rewritten_query: str
    rewrite_type: str
    confidence: float
    explanation: str


class QueryAnalyzer:
    """Analyzes queries to understand intent and structure."""

    def __init__(self):
        """Initialize query analyzer."""
        # Stopwords may be unavailable if download failed; fall back gracefully
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not available; using minimal fallback set")
            self.stop_words = {
                'the', 'is', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'with',
                'it', 'this', 'that', 'as', 'by', 'at', 'from', 'are', 'be'
            }
        self.intent_patterns = self._compile_intent_patterns()
        self.domain_keywords = self._load_domain_keywords()

    @staticmethod
    def _safe_word_tokenize(text: str) -> list[str]:
        """Tokenize text, falling back if punkt is unavailable."""
        try:
            return cast(list[str], word_tokenize(text))
        except LookupError:
            import re
            # Simple fallback: split into words and punctuation
            return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def _compile_intent_patterns(self) -> dict[QueryIntent, list[re.Pattern]]:
        """Compile regex patterns for intent detection."""
        patterns = {
            QueryIntent.FACTUAL: [
                re.compile(r'^(what|who|where|which)\s+', re.I),
                re.compile(r'\b(fact|detail|information|data)\b', re.I)
            ],
            QueryIntent.EXPLORATORY: [
                re.compile(r'^tell me about\s+', re.I),
                re.compile(r'\b(explore|overview|explain|describe)\b', re.I)
            ],
            QueryIntent.COMPARATIVE: [
                re.compile(r'\b(compare|versus|vs|difference|similar|better|worse)\b', re.I),
                re.compile(r'\b(than|between|among)\b', re.I)
            ],
            QueryIntent.PROCEDURAL: [
                re.compile(r'^how (to|do|can)\s+', re.I),
                re.compile(r'\b(step|procedure|method|process|guide)\b', re.I)
            ],
            QueryIntent.DEFINITIONAL: [
                re.compile(r'^what (is|are)\s+', re.I),
                re.compile(r'\b(define|definition|meaning|means)\b', re.I)
            ],
            QueryIntent.CAUSAL: [
                re.compile(r'^why\s+', re.I),
                re.compile(r'\b(cause|reason|because|result|effect|lead to)\b', re.I)
            ],
            QueryIntent.TEMPORAL: [
                re.compile(r'^when\s+', re.I),
                re.compile(r'\b(timeline|history|date|year|period|era)\b', re.I),
                re.compile(r'\b(before|after|during|since|until)\b', re.I)
            ],
            QueryIntent.ANALYTICAL: [
                re.compile(r'\b(analyze|evaluate|assess|critique|review)\b', re.I),
                re.compile(r'\b(impact|significance|implications|consequences)\b', re.I)
            ]
        }
        return patterns

    def _load_domain_keywords(self) -> dict[str, set[str]]:
        """Load domain-specific keywords."""
        return {
            "technology": {"software", "hardware", "code", "programming", "algorithm",
                          "database", "network", "system", "application", "framework"},
            "science": {"research", "experiment", "hypothesis", "theory", "data",
                       "analysis", "study", "observation", "evidence", "methodology"},
            "business": {"market", "strategy", "revenue", "customer", "product",
                        "service", "growth", "investment", "profit", "company"},
            "medical": {"patient", "treatment", "diagnosis", "symptom", "disease",
                       "medication", "therapy", "clinical", "health", "medical"},
            "legal": {"law", "regulation", "contract", "legal", "court",
                     "rights", "obligation", "liability", "compliance", "jurisdiction"}
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a query to understand its intent and structure.

        Args:
            query: The query to analyze

        Returns:
            QueryAnalysis object with analysis results
        """
        # Clean query
        cleaned_query = self._clean_query(query)

        # Detect intent
        intent = self._detect_intent(query)

        # Assess complexity
        complexity = self._assess_complexity(cleaned_query)

        # Extract key terms
        key_terms = self._extract_key_terms(cleaned_query)

        # Extract entities
        entities = self._extract_entities(cleaned_query)

        # Extract temporal references
        temporal_refs = self._extract_temporal_references(cleaned_query)

        # Detect question type
        question_type = self._detect_question_type(query)

        # Detect domain
        domain = self._detect_domain(cleaned_query)

        return QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned_query,
            intent=intent,
            complexity=complexity,
            key_terms=key_terms,
            entities=entities,
            temporal_refs=temporal_refs,
            question_type=question_type,
            domain=domain,
            metadata={
                "word_count": len(cleaned_query.split()),
                "has_question_mark": '?' in query,
                "has_quotes": '"' in query or "'" in query
            }
        )

    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        # Remove extra whitespace
        cleaned = ' '.join(query.split())

        # Normalize punctuation
        cleaned = re.sub(r'\s+([?.!,])', r'\1', cleaned)

        return cleaned

    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent."""
        query_lower = query.lower()

        # Check each intent pattern
        intent_scores: defaultdict[QueryIntent, int] = defaultdict(int)

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    intent_scores[intent] += 1

        # Return intent with highest score
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]

        # Default to factual
        return QueryIntent.FACTUAL

    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity."""
        # Count concepts (approximated by noun phrases)
        words = self._safe_word_tokenize(query.lower())

        # Remove stop words
        content_words = [w for w in words if w not in self.stop_words and w.isalnum()]

        # Count logical operators
        logical_ops = sum(1 for w in words if w in ['and', 'or', 'but', 'however', 'although'])

        # Assess based on content words and operators
        if len(content_words) <= 3 and logical_ops == 0:
            return QueryComplexity.SIMPLE
        elif len(content_words) <= 6 or logical_ops <= 1:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX

    def _extract_key_terms(self, query: str) -> list[str]:
        """Extract key terms from query."""
        words = self._safe_word_tokenize(query.lower())

        # Filter stop words and punctuation
        key_terms = [
            w for w in words
            if w not in self.stop_words and w.isalnum() and len(w) > 2
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms

    def _extract_entities(self, query: str) -> list[str]:
        """Extract named entities from query."""
        entities = []

        # Simple pattern-based entity extraction
        # Look for capitalized words (not at sentence start)
        words = query.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and word.isalpha():
                entities.append(word)

        # Look for quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)

        return entities

    def _extract_temporal_references(self, query: str) -> list[str]:
        """Extract temporal references from query."""
        temporal_refs = []

        # Year patterns
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        temporal_refs.extend(years)

        # Month patterns
        months = re.findall(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',
            query, re.I
        )
        temporal_refs.extend(months)

        # Relative time patterns
        relative = re.findall(
            r'\b(yesterday|today|tomorrow|last\s+\w+|next\s+\w+|recent|recently)\b',
            query, re.I
        )
        temporal_refs.extend(relative)

        return temporal_refs

    def _detect_question_type(self, query: str) -> Optional[str]:
        """Detect the type of question."""
        query_lower = query.lower().strip()

        if query_lower.startswith('what'):
            return 'what'
        elif query_lower.startswith('why'):
            return 'why'
        elif query_lower.startswith('how'):
            return 'how'
        elif query_lower.startswith('when'):
            return 'when'
        elif query_lower.startswith('where'):
            return 'where'
        elif query_lower.startswith('who'):
            return 'who'
        elif query_lower.startswith('which'):
            return 'which'
        elif query_lower.startswith(('is', 'are', 'do', 'does', 'can', 'could', 'should', 'would')):
            return 'yes/no'

        return None

    def _detect_domain(self, query: str) -> Optional[str]:
        """Detect the domain of the query."""
        query_lower = query.lower()

        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]

        return None


class QueryRewriter:
    """Rewrites queries for better retrieval."""

    def __init__(self):
        """Initialize query rewriter."""
        self.analyzer = QueryAnalyzer()

    def rewrite_query(
        self,
        query: str,
        strategies: Optional[list[str]] = None,
        failed_docs: Optional[list[Any]] = None,
        failure_reason: Optional[str] = None,
    ) -> list[QueryRewrite]:
        """
        Rewrite query using various strategies.

        Args:
            query: Original query
            strategies: List of strategies to use
            failed_docs: Optional list of documents that failed grading (for improve_for_retrieval)
            failure_reason: Optional reason for low relevance (for improve_for_retrieval)

        Returns:
            List of query rewrites
        """
        if strategies is None:
            strategies = ["synonym", "decompose", "generalize", "specify"]

        rewrites: list[QueryRewrite] = []

        # Analyze original query
        analysis = self.analyzer.analyze_query(query)

        for strategy in strategies:
            if strategy == "synonym":
                rewrites.extend(self._rewrite_with_synonyms(query, analysis))
            elif strategy == "decompose":
                rewrites.extend(self._decompose_query(query, analysis))
            elif strategy == "generalize":
                rewrites.extend(self._generalize_query(query, analysis))
            elif strategy == "specify":
                rewrites.extend(self._specify_query(query, analysis))
            elif strategy == "clarify":
                rewrites.extend(self._clarify_query(query, analysis))
            elif strategy == "improve_for_retrieval":
                rewrites.extend(self._improve_for_retrieval(query, analysis, failed_docs, failure_reason))

        return rewrites

    def _rewrite_with_synonyms(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> list[QueryRewrite]:
        """Rewrite query using synonyms."""
        rewrites: list[QueryRewrite] = []

        for term in analysis.key_terms:
            synonyms = self._get_synonyms(term)

            for synonym in synonyms[:2]:  # Limit to 2 synonyms per term
                rewritten = query.replace(term, synonym)
                if rewritten != query:
                    rewrites.append(QueryRewrite(
                        rewritten_query=rewritten,
                        rewrite_type="synonym",
                        confidence=0.8,
                        explanation=f"Replaced '{term}' with synonym '{synonym}'"
                    ))

        return rewrites

    def _decompose_query(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> list[QueryRewrite]:
        """Decompose complex query into simpler parts."""
        rewrites: list[QueryRewrite] = []

        if analysis.complexity != QueryComplexity.COMPLEX:
            return rewrites

        # Split on conjunctions
        parts = re.split(r'\b(?:and|or|but)\b', query, flags=re.I)

        if len(parts) > 1:
            for i, part in enumerate(parts):
                part = part.strip()
                if len(part) > 10:  # Meaningful part
                    rewrites.append(QueryRewrite(
                        rewritten_query=part,
                        rewrite_type="decompose",
                        confidence=0.7,
                        explanation=f"Part {i+1} of decomposed query"
                    ))

        return rewrites

    def _generalize_query(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> list[QueryRewrite]:
        """Generalize specific query."""
        rewrites = []

        # Remove specific entities
        generalized = query
        for entity in analysis.entities:
            generalized = generalized.replace(entity, "[entity]")

        # Remove temporal references
        for temporal in analysis.temporal_refs:
            generalized = generalized.replace(temporal, "[time]")

        if generalized != query:
            rewrites.append(QueryRewrite(
                rewritten_query=generalized,
                rewrite_type="generalize",
                confidence=0.6,
                explanation="Removed specific entities and temporal references"
            ))

        return rewrites

    def _specify_query(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> list[QueryRewrite]:
        """Add specificity to vague query."""
        rewrites = []

        # Add domain context if detected
        if analysis.domain:
            specified = f"{query} in {analysis.domain} context"
            rewrites.append(QueryRewrite(
                rewritten_query=specified,
                rewrite_type="specify",
                confidence=0.7,
                explanation=f"Added domain context: {analysis.domain}"
            ))

        # Add intent-based specifications
        if analysis.intent == QueryIntent.PROCEDURAL:
            specified = f"{query} step by step"
            rewrites.append(QueryRewrite(
                rewritten_query=specified,
                rewrite_type="specify",
                confidence=0.8,
                explanation="Added procedural specification"
            ))

        return rewrites

    def _clarify_query(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> list[QueryRewrite]:
        """Clarify ambiguous query."""
        rewrites = []

        # Add question words if missing
        if not analysis.question_type and not query.endswith('?'):
            if analysis.intent == QueryIntent.DEFINITIONAL:
                clarified = f"What is {query}?"
            elif analysis.intent == QueryIntent.PROCEDURAL:
                clarified = f"How to {query}?"
            elif analysis.intent == QueryIntent.CAUSAL:
                clarified = f"Why {query}?"
            else:
                clarified = f"What about {query}?"

            rewrites.append(QueryRewrite(
                rewritten_query=clarified,
                rewrite_type="clarify",
                confidence=0.6,
                explanation="Added question structure for clarity"
            ))

        return rewrites

    def _get_synonyms(self, word: str) -> list[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()

        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)

        return list(synonyms)

    # ========== SELF-CORRECTING RAG: Query Rewriting Loop (Stage 2) ==========

    def _improve_for_retrieval(
        self,
        query: str,
        analysis: QueryAnalysis,
        failed_docs: Optional[list[Any]] = None,
        failure_reason: Optional[str] = None,
    ) -> list[QueryRewrite]:
        """
        Rewrite query to improve retrieval when document grading shows low relevance.

        This strategy analyzes the failure reason and documents to reformulate the query
        for better results. Part of Self-Correcting RAG Stage 2.

        Args:
            query: Original query
            analysis: QueryAnalysis from analyzer
            failed_docs: Documents that failed grading (may contain partial useful info)
            failure_reason: Reason for low relevance (e.g., "avg_relevance_0.2")

        Returns:
            List of query rewrites designed to improve retrieval
        """
        rewrites = []

        # Strategy 1: Remove modifiers that may be too restrictive
        simplified = self._remove_modifiers(query, analysis)
        if simplified and simplified != query:
            rewrites.append(QueryRewrite(
                rewritten_query=simplified,
                rewrite_type="improve_for_retrieval",
                confidence=0.75,
                explanation="Removed restrictive modifiers to broaden search"
            ))

        # Strategy 2: Add focus terms based on domain/intent
        focused = self._add_focus_terms(query, analysis)
        if focused and focused != query:
            rewrites.append(QueryRewrite(
                rewritten_query=focused,
                rewrite_type="improve_for_retrieval",
                confidence=0.7,
                explanation="Added domain-specific focus terms"
            ))

        # Strategy 3: Extract entities from failed docs and use them
        if failed_docs:
            entity_based = self._extract_and_use_entities(query, analysis, failed_docs)
            if entity_based and entity_based != query:
                rewrites.append(QueryRewrite(
                    rewritten_query=entity_based,
                    rewrite_type="improve_for_retrieval",
                    confidence=0.65,
                    explanation="Incorporated relevant entities from partial results"
                ))

        # Strategy 4: Convert to a more specific question form
        question_form = self._convert_to_specific_question(query, analysis)
        if question_form and question_form != query:
            rewrites.append(QueryRewrite(
                rewritten_query=question_form,
                rewrite_type="improve_for_retrieval",
                confidence=0.6,
                explanation="Converted to more specific question form"
            ))

        # Strategy 5: Expand with related concepts
        expanded = self._expand_with_related_concepts(query, analysis)
        if expanded and expanded != query:
            rewrites.append(QueryRewrite(
                rewritten_query=expanded,
                rewrite_type="improve_for_retrieval",
                confidence=0.55,
                explanation="Expanded with related concepts"
            ))

        if not rewrites:
            fallback = self._keyword_fallback_for_retrieval(query, analysis)
            if fallback and fallback.lower() != query.lower():
                rewrites.append(QueryRewrite(
                    rewritten_query=fallback,
                    rewrite_type="improve_for_retrieval",
                    confidence=0.5,
                    explanation="Generated keyword-focused fallback rewrite"
                ))

        return rewrites

    def _remove_modifiers(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> Optional[str]:
        """Remove restrictive modifiers that may be too specific."""
        # Common restrictive modifiers that can narrow search too much
        restrictive_patterns = [
            r'\b(exactly|precisely|specifically|only|just|merely)\b',
            r'\b(latest|newest|most recent|current)\b',
            r'\b(best|top|greatest|finest)\b',
            r'\b(in \d{4}|during \d{4}|from \d{4})\b',  # Year specifications
        ]

        modified = query
        for pattern in restrictive_patterns:
            modified = re.sub(pattern, '', modified, flags=re.I)

        # Clean up extra whitespace
        modified = ' '.join(modified.split()).strip()

        if len(modified) < 5:  # Don't return if too short
            return None

        return modified if modified != query else None

    def _add_focus_terms(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> Optional[str]:
        """Add domain-specific focus terms to improve retrieval."""
        # Domain-specific focus terms to add
        domain_focus = {
            "technology": ["software", "development", "programming"],
            "science": ["research", "study", "analysis"],
            "business": ["company", "market", "industry"],
            "medical": ["health", "treatment", "clinical"],
            "legal": ["law", "regulation", "compliance"],
        }

        if not analysis.domain:
            return None

        focus_terms = domain_focus.get(analysis.domain, [])
        if not focus_terms:
            return None

        # Add the first focus term that's not already in the query
        query_lower = query.lower()
        for term in focus_terms:
            if term not in query_lower:
                return f"{query} {term}"

        return None

    def _extract_and_use_entities(
        self,
        query: str,
        analysis: QueryAnalysis,
        failed_docs: list[Any],
    ) -> Optional[str]:
        """Extract potentially useful entities from failed docs and incorporate them."""
        if not failed_docs:
            return None

        # Extract text from failed docs
        all_text = []
        for doc in failed_docs[:5]:  # Limit to top 5
            content = getattr(doc, 'content', '')
            if content:
                all_text.append(content[:500])  # First 500 chars

        if not all_text:
            return None

        combined = ' '.join(all_text)

        # Find capitalized words (potential entities) not in original query
        entity_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
        found_entities = entity_pattern.findall(combined)

        # Filter out common words and those already in query
        query_lower = query.lower()
        common_words = {'the', 'this', 'that', 'these', 'those', 'there', 'here', 'where', 'when', 'what', 'which', 'who', 'how', 'why'}
        useful_entities = [
            e for e in found_entities
            if e.lower() not in query_lower
            and e.lower() not in common_words
            and len(e) > 2
        ]

        if not useful_entities:
            return None

        # Take most frequent entity
        from collections import Counter
        entity_counts = Counter(useful_entities)
        top_entity = entity_counts.most_common(1)[0][0]

        return f"{query} {top_entity}"

    def _convert_to_specific_question(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> Optional[str]:
        """Convert query to a more specific question form for better retrieval."""
        # Skip if already a question
        if analysis.question_type or query.strip().endswith('?'):
            return None

        # Map intent values to question forms (keyed by value string to
        # remain robust if the module is reloaded and enum identity changes).
        intent_to_question = {
            "factual": f"What are the key facts about {query}?",
            "definitional": f"What is the definition and meaning of {query}?",
            "causal": f"What causes {query} and why does it happen?",
            "procedural": f"What are the steps to {query}?",
            "analytical": f"What is the analysis and evaluation of {query}?",
            "comparative": f"How does {query} compare to alternatives?",
            "temporal": f"What is the timeline and history of {query}?",
            "exploratory": f"What are the main aspects of {query}?",
        }

        intent_val = analysis.intent.value if isinstance(analysis.intent, Enum) else analysis.intent
        return intent_to_question.get(intent_val)

    def _expand_with_related_concepts(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> Optional[str]:
        """Expand query with semantically related concepts."""
        # Get hypernyms (broader concepts) for key terms
        expanded_terms = []
        for term in analysis.key_terms[:3]:  # Limit to first 3 key terms
            try:
                synsets = wordnet.synsets(term)
                for syn in synsets[:1]:  # First synset
                    hypernyms = syn.hypernyms()
                    for hyp in hypernyms[:1]:  # First hypernym
                        lemma = hyp.lemmas()[0].name().replace('_', ' ')
                        if lemma.lower() not in query.lower():
                            expanded_terms.append(lemma)
                            break
            except LookupError as exc:
                logger.debug(
                    "NLTK WordNet unavailable; skipping related concept expansion for '{}': {}",
                    term,
                    exc,
                )
                continue

        if not expanded_terms:
            return None

        # Add the first expanded term
        return f"{query} ({expanded_terms[0]})"

    def _keyword_fallback_for_retrieval(
        self,
        query: str,
        analysis: QueryAnalysis,
    ) -> Optional[str]:
        """Create a deterministic retrieval rewrite when optional NLP data is unavailable."""
        question_terms = {
            "what",
            "who",
            "where",
            "which",
            "how",
            "why",
            "when",
            "do",
            "does",
            "did",
            "can",
            "could",
            "should",
            "would",
        }
        terms = [term for term in analysis.key_terms if term not in question_terms]
        if len(terms) >= 2:
            fallback = " ".join(terms)
            if len(fallback) >= 5:
                return fallback

        cleaned = re.sub(
            r"^\s*(what|who|where|which|how|why|when)\s+"
            r"(is|are|was|were|do|does|did|can|could|should|would)\s+",
            "",
            query,
            flags=re.I,
        ).strip(" ?")
        if len(cleaned) >= 5:
            return cleaned
        return None


class QueryRouter:
    """Routes queries to appropriate retrieval strategies."""

    def __init__(self):
        """Initialize query router."""
        self.analyzer = QueryAnalyzer()
        self.routing_rules = self._define_routing_rules()

    def _define_routing_rules(self) -> dict[QueryIntent, dict[str, Any]]:
        """Define routing rules based on query intent."""
        return {
            QueryIntent.FACTUAL: {
                "retrieval_strategy": "precise",
                "reranking": True,
                "top_k": 5,
                "use_keywords": True
            },
            QueryIntent.EXPLORATORY: {
                "retrieval_strategy": "broad",
                "reranking": False,
                "top_k": 20,
                "use_semantic": True
            },
            QueryIntent.COMPARATIVE: {
                "retrieval_strategy": "multi_doc",
                "reranking": True,
                "top_k": 10,
                "group_by_source": True
            },
            QueryIntent.PROCEDURAL: {
                "retrieval_strategy": "sequential",
                "reranking": True,
                "top_k": 10,
                "preserve_order": True
            },
            QueryIntent.ANALYTICAL: {
                "retrieval_strategy": "comprehensive",
                "reranking": True,
                "top_k": 15,
                "include_context": True
            }
        }

    def route_query(self, query: str) -> dict[str, Any]:
        """
        Route query to appropriate retrieval strategy.

        Args:
            query: The query to route

        Returns:
            Routing configuration
        """
        # Analyze query
        analysis = self.analyzer.analyze_query(query)

        # Get base routing from intent
        routing = self.routing_rules.get(
            analysis.intent,
            self.routing_rules[QueryIntent.FACTUAL]
        ).copy()

        # Adjust based on complexity
        if analysis.complexity == QueryComplexity.COMPLEX:
            routing["top_k"] = min(routing["top_k"] * 2, 30)
            routing["use_query_expansion"] = True
        elif analysis.complexity == QueryComplexity.SIMPLE:
            routing["top_k"] = max(routing["top_k"] // 2, 3)

        # Add metadata
        routing["query_analysis"] = {
            "intent": analysis.intent.value,
            "complexity": analysis.complexity.value,
            "domain": analysis.domain,
            "key_terms": analysis.key_terms
        }

        logger.debug(f"Routed query with intent {analysis.intent.value} to {routing['retrieval_strategy']}")

        return routing


# Pipeline integration functions

async def analyze_query(context: Any, **kwargs) -> Any:
    """Analyze query for pipeline context."""
    analyzer = QueryAnalyzer()

    analysis = analyzer.analyze_query(context.query)

    # Store analysis in context
    context.metadata["query_analysis"] = {
        "intent": analysis.intent.value,
        "complexity": analysis.complexity.value,
        "key_terms": analysis.key_terms,
        "entities": analysis.entities,
        "domain": analysis.domain,
        "question_type": analysis.question_type
    }

    # Adjust pipeline based on analysis
    if analysis.complexity == QueryComplexity.COMPLEX:
        context.config["top_k"] = min(context.config.get("top_k", 10) * 2, 30)

    if analysis.intent == QueryIntent.COMPARATIVE:
        context.config["group_by_source"] = True

    logger.info(f"Query analysis: intent={analysis.intent.value}, complexity={analysis.complexity.value}")

    return context


async def rewrite_query(context: Any, **kwargs) -> Any:
    """Rewrite query for better retrieval."""
    config = context.config.get("query_rewriting", {})

    if not config.get("enabled", True):
        return context

    rewriter = QueryRewriter()

    strategies = config.get("strategies", ["synonym", "decompose"])
    rewrites = rewriter.rewrite_query(context.query, strategies)

    # Store rewrites
    context.metadata["query_rewrites"] = [
        {
            "query": r.rewritten_query,
            "type": r.rewrite_type,
            "confidence": r.confidence
        }
        for r in rewrites
    ]

    # Use best rewrite if confidence is high
    if rewrites:
        best_rewrite = max(rewrites, key=lambda r: r.confidence)
        if best_rewrite.confidence > config.get("min_confidence", 0.7):
            context.metadata["original_query"] = context.query
            context.query = best_rewrite.rewritten_query
            try:
                _qh = __import__("hashlib").md5((best_rewrite.rewritten_query or "").encode("utf-8")).hexdigest()[:8]
                logger.info(f"Rewrote query using {best_rewrite.rewrite_type}: hash={_qh}")
            except Exception:
                logger.info(f"Rewrote query using {best_rewrite.rewrite_type}")

    return context


async def route_query(context: Any, **kwargs) -> Any:
    """Route query to appropriate retrieval strategy."""
    router = QueryRouter()

    routing = router.route_query(context.query)

    # Apply routing configuration
    context.config.update(routing)

    # Store routing decision
    context.metadata["query_routing"] = routing

    logger.info(f"Routed query to {routing['retrieval_strategy']} strategy")

    return context
