import pytest


def test_sqlite_to_postgres_handles_hyphens_and_near():

    from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator

    # hyphenated word with wildcard
    q1 = "state-of-the-art*"
    out1 = FTSQueryTranslator.sqlite_to_postgres(q1)
    # Ensure wildcard is converted and hyphens preserved
    assert ":*" in out1
    assert "state-of-the-art" in out1

    # NEAR operator with hyphens
    q2 = "alpha-beta NEAR gamma-delta"
    out2 = FTSQueryTranslator.sqlite_to_postgres(q2)
    assert "<->" in out2
    assert "alpha-beta" in out2 and "gamma-delta" in out2


def test_sqlite_to_postgres_handles_quotes_and_parentheses():

    from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator

    q = '"exact phrase" (bonus)'
    out = FTSQueryTranslator.sqlite_to_postgres(q)
    # Phrase converted to parentheses and ANDs for spaces
    assert "(" in out and ")" in out
    assert "<->" in out or "&" in out


def test_fts_query_builder_hyphen_and_unicode():

    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MediaDBRetriever

    r = MediaDBRetriever(db_path="/tmp/test.db")  # path used for constructing object only  # nosec B108

    q1 = "state-of-the-art models"
    built1 = r._build_fts_query(q1)
    # Should be quoted phrase
    assert built1.startswith('"') and built1.endswith('"')

    q2 = "naïve café"
    built2 = r._build_fts_query(q2)
    assert built2.startswith('"') and built2.endswith('"')


def test_fts_query_translation_truncates_long_input():

    from tldw_Server_API.app.core.DB_Management.backends.fts_translator import (
        FTSQueryTranslator,
        MAX_FTS_QUERY_LENGTH,
    )

    long_query = "a" * (MAX_FTS_QUERY_LENGTH + 50)
    out = FTSQueryTranslator.sqlite_to_postgres(long_query)
    assert len(out) == MAX_FTS_QUERY_LENGTH


def test_postgres_to_sqlite_truncates_long_input():

    from tldw_Server_API.app.core.DB_Management.backends.fts_translator import (
        FTSQueryTranslator,
        MAX_FTS_QUERY_LENGTH,
    )

    long_query = "a" * (MAX_FTS_QUERY_LENGTH + 50)
    out = FTSQueryTranslator.postgres_to_sqlite(long_query)
    assert len(out) == MAX_FTS_QUERY_LENGTH


def test_sqlite_normalization_quotes_plain_tokens_with_punctuation():
    """Plain punctuation terms are quoted without destroying column filters."""
    from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator

    out = FTSQueryTranslator.normalize_query(
        "codex-flashcards-ux front:state-of-the-art",
        "sqlite",
    )

    assert out == '"codex-flashcards-ux" front:"state-of-the-art"'

    aliased = FTSQueryTranslator.normalize_query(
        "codex-flashcards-ux front:state-of-the-art",
        "sqlite",
        sqlite_column_aliases={"front": "front_search"},
    )

    assert aliased == '"codex-flashcards-ux" front_search:"state-of-the-art"'


def test_sqlite_normalization_resolves_aliases_case_insensitively_and_quotes_unknown_scopes():
    from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator

    out = FTSQueryTranslator.normalize_query(
        "Front:alpha extra:beta OR -Back:old-style OR -extra:legacy",
        "sqlite",
        sqlite_column_aliases={"front": "front_search", "back": "back_search"},
    )

    assert out == 'front_search:alpha beta OR NOT back_search:"old-style" OR NOT legacy'


def test_sqlite_normalization_preserves_quoted_column_phrases_and_quotes_negative_terms():
    """Quoted phrases and allowed negative punctuation terms stay valid FTS5."""
    from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator

    out = FTSQueryTranslator.normalize_query(
        'front:"two words" front:"already-quoted" front:"state of the art" '
        '-state-of-the-art -back:old-style -"two words" -front:"old style"',
        "sqlite",
    )

    assert (
        out == 'front:"two words" front:"already-quoted" front:"state of the art" '
        'NOT "state-of-the-art" NOT back:"old-style" NOT "two words" NOT front:"old style"'
    )

    assert FTSQueryTranslator.normalize_query("-state-of-the-art", "sqlite") == '"-state-of-the-art"'


def test_sqlite_normalization_handles_escaped_phrase_quotes_and_case_sensitive_operators():
    """Doubled phrase quotes stay tokenized, while lowercase operator words stay terms."""
    from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator

    out = FTSQueryTranslator.normalize_query(
        'front:"said ""hello""" alpha and beta OR gamma',
        "sqlite",
    )

    assert out == 'front:"said ""hello""" alpha and beta OR gamma'


def test_sqlite_normalization_allows_negative_terms_after_binary_operators():
    """Negative punctuation terms after AND/OR are rewritten to SQLite FTS5 NOT operands."""
    from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator

    assert (
        FTSQueryTranslator.normalize_query("alpha AND -state-of-the-art OR -front:old-style", "sqlite")
        == 'alpha AND NOT "state-of-the-art" OR NOT front:"old-style"'
    )


def test_sqlite_normalization_does_not_depend_on_regex_tokenizer(monkeypatch):
    """SQLite normalization tokenizes user input without a backtracking regex."""
    from tldw_Server_API.app.core.DB_Management.backends import fts_translator
    from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator

    class ExplodingTokenizer:
        def findall(self, query):
            raise AssertionError("regex tokenizer should not be used")

    monkeypatch.setattr(fts_translator, "SQLITE_FTS_TOKEN_RE", ExplodingTokenizer(), raising=False)

    assert (
        FTSQueryTranslator.normalize_query('front:"said ""hello""" alpha OR -state-of-the-art', "sqlite")
        == 'front:"said ""hello""" alpha OR NOT "state-of-the-art"'
    )
