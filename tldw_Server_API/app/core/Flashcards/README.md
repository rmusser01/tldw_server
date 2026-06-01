# Flashcards

The Flashcards module supports study-card import/export and scheduling helpers,
including Anki APKG packaging, structured Q&A import, asset references, SM-2 and
FSRS schedulers, and study assistant utilities. API CRUD lives in the
flashcards endpoint and DB adapters; this package holds portable card logic.

## Start Here

- APKG import/export: `apkg_importer.py` and `apkg_exporter.py`.
- Scheduling: `scheduler_sm2.py` and `scheduler_fsrs.py`.
- Structured imports and study helpers: `structured_qa_import.py`,
  `study_assistant.py`, and `asset_refs.py`.
- API endpoint and schemas: `app/api/v1/endpoints/flashcards.py`,
  `app/api/v1/schemas/flashcards.py`, and `app/api/v1/schemas/study_packs.py`.
- Tests: `tests/Flashcards/` and `tests/StudyPacks/`.

## Responsibilities

- Convert flashcard rows into Anki-compatible `.apkg` collections.
- Parse supported `.apkg` exports back into server-side cards/decks.
- Normalize structured question/answer imports.
- Compute review scheduling metadata for SM-2 and FSRS-style flows.
- Resolve card asset references without coupling card logic to storage internals.

## Module Map

- `apkg_exporter.py` creates Anki collection SQLite files and ZIP packages.
- `apkg_importer.py` reads Anki packages and normalizes notes/cards.
- `structured_qa_import.py` validates structured text/JSON card imports.
- `scheduler_sm2.py` and `scheduler_fsrs.py` compute review state transitions.
- `study_assistant.py` contains service helpers for study sessions.
- `asset_refs.py` validates and normalizes media references on cards.

## How It Connects

- The flashcards endpoint owns HTTP shape, AuthNZ dependencies, and DB access.
- StudyPacks can generate flashcards and enqueue background study-pack jobs.
- File/asset storage is handled by adjacent storage and DB modules, not by the
  APKG helpers directly.

## Extension Points

- Add import formats in `structured_qa_import.py` or a new focused parser module.
- Add scheduler variants behind explicit scheduler names and deterministic tests.
- Add APKG media support by extending `asset_refs.py` and the APKG pack/unpack
  paths together.

## Testing

- APKG package behavior: `tests/Flashcards/test_apkg_exporter.py` and
  `tests/Flashcards/test_apkg_importer.py`.
- Endpoint/DB behavior: `tests/Flashcards/test_flashcards_endpoint_integration.py`
  and `tests/Flashcards/test_flashcards_db_assets.py`.
- Scheduling and imports: `tests/Flashcards/test_scheduler_fsrs.py`,
  `tests/Flashcards/test_flashcards_scheduler_schema.py`, and
  `tests/Flashcards/test_structured_qa_import.py`.

## Gotchas

- APKG generation builds a SQLite collection inside a ZIP archive; keep file
  handles and temporary paths scoped tightly.
- Large decks can use substantial memory. Add streaming/temp-file coverage before
  expanding media-heavy export paths.
