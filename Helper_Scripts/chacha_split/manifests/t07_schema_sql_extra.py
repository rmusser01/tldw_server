# TASK-13950.22 PR F: four class constants the PostgreSQL helpers read that t05 did not select (A.3: "another constant -> schema_sql").
# Run `split.py constants` on this file first (it appends to chacha/schema_sql.py), then `split.py functions` on t07_postgres_schema.py.
# Post-constants hand fix: the constants run leaves a second schema_sql import block in the class body; `ruff --select I` merges it and keeps both F401 noqa comments, so delete the duplicate comment.
# The hand edits before apply: ChaChaContext.normalize_postgres_catalog_expression (a pure static helper, A.3) and the class forwarder `_normalize_postgres_catalog_expression` (structural in t07_postgres_schema.py).
MANIFEST = {
    "task": "T7",
    "mode": "constants",
    "source": "tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py",
    "target": "tldw_Server_API/app/core/DB_Management/chacha/schema_sql.py",
    "alias": "schema_sql",
    "store_class": None,
    "store_attr": None,
    "names": [],
    "shims": [],
    "constants": [
        "_NOTES_MOODBOARD_STUDIO_V61_MIGRATION_ID",
        "_NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SECONDS",
        "_NOTES_MOODBOARD_STUDIO_V61_POSTGRES_LOCK_TIMEOUT",
        "_NOTES_MOODBOARD_STUDIO_V61_POSTGRES_STATEMENT_TIMEOUT",
    ],
    "rewrites": {},
    "renames": {},
    "structural": [],
}
