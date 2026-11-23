# TASK-13950.22 PR H: the schema bootstrap moves into chacha/schema_bootstrap.py. Names: `split.py select init` plus `_initialize_schema_postgres`, whose class forwarder stays (the only entry point, so the 35 `_initialize_schema` patches stay live).
# The target already holds `postgres_schema_migration`. The `functions` mode only creates a module, so the apply is: move schema_bootstrap.py aside, run `split.py functions`, then paste `postgres_schema_migration` and `_POSTGRES_SCHEMA_BOOTSTRAP_LOCK_TIMEOUT = "30s"` (the class attribute moves, no reader is left) back by hand. `verify` compares only the moved defs.
# R18 was a separate commit before this one (7e0de4f9e7), so this move is verbatim; run verify and diffpairs with --base 7e0de4f9e7.
# R29: `_runtime_schema_version` becomes the public `runtime_schema_version` (class default None) and the moved body writes `ctx.db.runtime_schema_version`; `supports_notes_moodboard_studio_v61` holds the one string probe of it, a hand edit.
# `_initialize_schema` is structural only because the tool rewrote its `self._initialize_schema_postgres()` to a module call; that line is reverted by hand, so the class's shim stays the entry point and a patch of it stays live.
# Post-apply step (ruff's isort order): `ruff check --select I --fix` on ChaChaNotes_DB.py, chacha/schema_bootstrap.py and every edited .py file; then `ruff check --select F401 --fix` on ChaChaNotes_DB.py for the imports it no longer uses.
MANIFEST = {
    "task": "T9",
    "mode": "functions",
    "source": "tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py",
    "target": "tldw_Server_API/app/core/DB_Management/chacha/schema_bootstrap.py",
    "alias": "schema_bootstrap",
    "doc": "PostgreSQL bootstrap coordination and the schema bootstrap moved out of CharactersRAGDB (TASK-13950.22).",
    "store_class": None,
    "store_attr": None,
    "names": [
        "_initialize_schema_sqlite_legacy",
        "_postgres_schema_is_current",
        "_initialize_schema_postgres",
    ],
    "shims": ["_initialize_schema_postgres"],
    "rewrites": {
        "_runtime_schema_version": "ctx.db.runtime_schema_version",
        "_normalize_postgres_catalog_expression": "ctx.normalize_postgres_catalog_expression",
        "_ensure_conversation_settings_table": "ctx.db._ensure_conversation_settings_table",
        "_ensure_persona_live_voice_session_summaries_table": "ctx.db._ensure_persona_live_voice_session_summaries_table",
        "_POSTGRES_SCHEMA_BOOTSTRAP_LOCK_TIMEOUT": "_POSTGRES_SCHEMA_BOOTSTRAP_LOCK_TIMEOUT"
    },
    "renames": {"_runtime_schema_version": "runtime_schema_version"},
    "structural": ["supports_notes_moodboard_studio_v61", "_initialize_schema"],
}
