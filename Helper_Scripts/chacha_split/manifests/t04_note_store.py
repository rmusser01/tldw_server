# TASK-13950.22 PR C2: the note store receives the context. Run `split.py store` on this and t04_keyword_store.py together.
MANIFEST = {
    "task": "T4",
    "mode": "store",
    "source": "tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py",
    "target": "tldw_Server_API/app/core/DB_Management/chacha/note_store.py",
    "alias": None,
    "store_class": "NoteStore",
    "store_attr": "note_store",
    "names": ["_invalidate_note_clipper_sidecars", "_delete_note_clipper_sidecars"],
    "shims": [],
    "constants": [],
    "rewrites": {},
    "renames": {"_supports_notes_moodboard_studio_v61": "supports_notes_moodboard_studio_v61"},
    "structural": [],
}
