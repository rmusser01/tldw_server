"""Converge genuinely different published and dev PostgreSQL V83 histories."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import build_dictionary_rls_sql
from tldw_Server_API.app.core.DB_Management.chacha.dictionary_schema import postgres_dictionary_schema_statements
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, SchemaError

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

# Exact historical unowned catalog DDL, shared with test_dictionary_postgres_ownership.
LEGACY_CATALOG_DDL = (
    "\n                        CREATE TABLE IF NOT EXISTS chat_dictionaries (\n                            id SERIAL PRIMARY KEY,\n                            name TEXT NOT NULL UNIQUE,\n                            description TEXT,\n                            category TEXT,\n                            tags TEXT,\n                            included_dictionary_ids TEXT,\n                            is_active BOOLEAN DEFAULT TRUE,\n                            default_token_budget INTEGER NULL,\n                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                            usage_count INTEGER DEFAULT 0,\n                            last_used_at TIMESTAMP NULL,\n                            version INTEGER DEFAULT 1,\n                            deleted BOOLEAN DEFAULT FALSE\n                        )\n                        ",
    "\n                        CREATE TABLE IF NOT EXISTS dictionary_entries (\n                            id SERIAL PRIMARY KEY,\n                            dictionary_id INTEGER NOT NULL,\n                            key TEXT NOT NULL,\n                            content TEXT NOT NULL,\n                            is_regex BOOLEAN DEFAULT FALSE,\n                            probability REAL DEFAULT 1.0,\n                            max_replacements INTEGER DEFAULT 1,\n                            group_name TEXT,\n                            timed_effects TEXT DEFAULT '{}',\n                            enabled BOOLEAN DEFAULT TRUE,\n                            case_sensitive BOOLEAN DEFAULT TRUE,\n                            sort_order INTEGER,\n                            usage_count INTEGER DEFAULT 0,\n                            last_used_at TIMESTAMP NULL,\n                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE\n                        )\n                        ",
    "\n                        CREATE TABLE IF NOT EXISTS dictionary_transform_activity (\n                            id SERIAL PRIMARY KEY,\n                            dictionary_id INTEGER NOT NULL,\n                            chat_id TEXT NULL,\n                            entries_used TEXT NOT NULL,\n                            replacements INTEGER DEFAULT 0,\n                            iterations INTEGER DEFAULT 0,\n                            token_budget_used INTEGER NULL,\n                            original_text_preview TEXT NOT NULL,\n                            processed_text_preview TEXT NOT NULL,\n                            processing_time_ms REAL NULL,\n                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE\n                        )\n                        ",
    "\n                        CREATE TABLE IF NOT EXISTS dictionary_version_history (\n                            id SERIAL PRIMARY KEY,\n                            dictionary_id INTEGER NOT NULL,\n                            revision INTEGER NOT NULL,\n                            source_dictionary_version INTEGER NULL,\n                            change_type TEXT NOT NULL,\n                            summary TEXT NULL,\n                            entry_count INTEGER DEFAULT 0,\n                            dictionary_snapshot TEXT NOT NULL,\n                            entries_snapshot TEXT NOT NULL,\n                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n                            UNIQUE(dictionary_id, revision),\n                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE\n                        )\n                        ",
)


class ProvenanceV82(CharactersRAGDB):
    """Registered shared provenance-era migration frontier, before either V83."""

    _POSTGRES_SCHEMA_VERSION = 82


def seed_history(backend, history):
    """Apply actual historical DDL, not a version stamp on today's schema."""
    legacy = ProvenanceV82(":memory:", client_id="1", backend=backend)
    note_id = legacy.add_note("Keep this note", "Original owned content")
    with legacy.transaction() as conn:
        for statement in LEGACY_CATALOG_DDL:
            conn.execute(statement)
        row = conn.execute(
            "INSERT INTO chat_dictionaries(name) VALUES (?) RETURNING id", ("Unowned legacy",)
        ).fetchone()
        dictionary_id = row["id"]
        conn.execute(
            "INSERT INTO dictionary_entries(dictionary_id,key,content) VALUES (?,?,?)",
            (dictionary_id, "PRIVATE", "Never adopt this row"),
        )
        if history == "published83":
            for statement in (*postgres_dictionary_schema_statements(), *build_dictionary_rls_sql()):
                conn.execute(statement)
            legacy._verify_dictionary_schema_postgres(conn)
        else:
            conn.execute("ALTER TABLE native_chat_quota_intents ADD COLUMN quota_owner_json TEXT")
        legacy._set_schema_version_postgres(conn, 83)
        if history in ("dev84", "dev85"):
            legacy._migrate_from_v83_to_v84_postgres(conn)
        if history == "dev85":
            legacy._migrate_from_v84_to_v85_postgres(conn)
        columns = {item["name"] for item in backend.get_table_info("native_chat_quota_intents", connection=conn)}
        assert ("quota_owner_json" in columns) == (history != "published83")  # nosec B101 - verify distinct historical fixture shapes.
        dictionary_columns = {item["name"] for item in backend.get_table_info("chat_dictionaries", connection=conn)}
        assert ("client_id" in dictionary_columns) == (history == "published83")  # nosec B101 - verify distinct historical fixture shapes.
    legacy.close_connection()
    return note_id, dictionary_id


@pytest.mark.parametrize("history", ["published83", "dev83", "dev84", "dev85", "fresh"])
def test_upgrade_converges_quota_and_dictionary_authority(pg_restricted_backend, history):
    """Reopen twice under the actual non-bypass owner with both contracts intact."""
    note_id, dictionary_id = seed_history(pg_restricted_backend, history) if history != "fresh" else (None, None)
    for _ in range(2):
        db = CharactersRAGDB(":memory:", client_id="1", backend=pg_restricted_backend)
        try:
            with db.transaction() as conn:
                columns = {
                    item["name"]
                    for item in pg_restricted_backend.get_table_info("native_chat_quota_intents", connection=conn)
                }
                assert "quota_owner_json" in columns  # nosec B101 - real upgrade outcome.
                db._verify_dictionary_schema_postgres(conn)
                assert db._get_schema_version_postgres(conn) == 86  # nosec B101 - combined contract version.
                if dictionary_id is not None:
                    assert (
                        conn.execute("SELECT id FROM chat_dictionaries WHERE id = ?", (dictionary_id,)).fetchone()
                        is None
                    )  # nosec B101 - preserve quarantine.
            if note_id is not None:
                assert db.get_note_by_id(note_id)["content"] == "Original owned content"  # nosec B101 - preserve owned data.
        finally:
            db.close_connection()


@pytest.mark.parametrize("history", ["published83", "dev85"])
def test_convergence_failure_rolls_back_ddl_and_version(pg_restricted_backend, monkeypatch, history):
    """A failure after DDL leaves the old frontier and schema recoverable."""
    note_id, _ = seed_history(pg_restricted_backend, history)
    verify = CharactersRAGDB._verify_native_quota_owner_schema_postgres

    def fail_after_ddl(self, conn):
        verify(self, conn)
        raise SchemaError("convergence verification fault")

    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_verify_native_quota_owner_schema_postgres", fail_after_ddl)
        with pytest.raises(CharactersRAGDBError, match="convergence verification fault"):
            CharactersRAGDB(":memory:", client_id="1", backend=pg_restricted_backend)
    with pg_restricted_backend.transaction() as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s", (CharactersRAGDB._SCHEMA_NAME,)
        ).fetchone()["version"]
        assert version == (83 if history == "published83" else 85)  # nosec B101 - transaction rollback.
        columns = {
            row["name"] for row in pg_restricted_backend.get_table_info("native_chat_quota_intents", connection=conn)
        }
        assert ("quota_owner_json" in columns) == (history != "published83")  # nosec B101 - DDL rollback.
    recovered = CharactersRAGDB(":memory:", client_id="1", backend=pg_restricted_backend)
    try:
        assert recovered.get_note_by_id(note_id)["content"] == "Original owned content"  # nosec B101 - preserve data through failure.
    finally:
        recovered.close_connection()


@pytest.mark.parametrize("incompatible", ["integer", "not-null"])
def test_convergence_rejects_incompatible_quota_descriptor(pg_restricted_backend, incompatible):
    """Existing incompatible descriptors must never publish a ready V86."""
    seed_history(pg_restricted_backend, "dev85")
    with pg_restricted_backend.transaction() as conn:
        if incompatible == "integer":
            conn.execute("ALTER TABLE native_chat_quota_intents ALTER COLUMN quota_owner_json TYPE INTEGER USING NULL")
        else:
            conn.execute("ALTER TABLE native_chat_quota_intents ALTER COLUMN quota_owner_json SET NOT NULL")
    with pytest.raises(CharactersRAGDBError, match="descriptor schema is missing or invalid"):
        CharactersRAGDB(":memory:", client_id="1", backend=pg_restricted_backend)
    with pg_restricted_backend.transaction() as conn:
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = %s", (CharactersRAGDB._SCHEMA_NAME,)
            ).fetchone()["version"]
            == 85
        )  # nosec B101 - reject version publication.


def test_convergence_preserves_owned_catalog_and_foreign_denial(pg_restricted_backend):
    """A published owned catalog keeps its entries without gaining foreign access."""
    from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService

    seed_history(pg_restricted_backend, "published83")

    class PublishedV83(CharactersRAGDB):
        _POSTGRES_SCHEMA_VERSION = 83

    old = PublishedV83(":memory:", client_id="1", backend=pg_restricted_backend)
    service = ChatDictionaryService(old)
    dictionary_id = service.create_dictionary("Owned published catalog")
    entry_id = service.add_entry(dictionary_id, "OWNER_TOKEN", "OWNER_CONTENT")
    old.close_connection()
    alice = CharactersRAGDB(":memory:", client_id="1", backend=pg_restricted_backend)
    bob = CharactersRAGDB(":memory:", client_id="2", backend=pg_restricted_backend)
    try:
        own, foreign = ChatDictionaryService(alice), ChatDictionaryService(bob)
        assert own.get_entry(entry_id)["replacement"] == "OWNER_CONTENT"  # nosec B101 - owned data preservation.
        assert foreign.get_dictionary(dictionary_id) is None  # nosec B101 - catalog isolation.
        assert foreign.get_entry(entry_id) is None  # nosec B101 - entry isolation.
    finally:
        alice.close_connection()
        bob.close_connection()
