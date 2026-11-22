# TASK-13950.22 PR E: two class constants the SQLite helpers read that t05 did not select (A.3: "another constant -> schema_sql").
# Run `split.py constants` on this file first (it appends to chacha/schema_sql.py), then `split.py functions` on t06_*.py.
MANIFEST = {
    "task": "T6",
    "mode": "constants",
    "source": "tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py",
    "target": "tldw_Server_API/app/core/DB_Management/chacha/schema_sql.py",
    "alias": "schema_sql",
    "store_class": None,
    "store_attr": None,
    "names": [],
    "shims": [],
    "constants": [
        "_LOCAL_UNBOUND_TASK_DATASET_ID",
        "_NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SIZE",
    ],
    "rewrites": {},
    "renames": {},
    "structural": [],
}
