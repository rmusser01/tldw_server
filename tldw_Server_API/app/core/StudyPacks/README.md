# StudyPacks

StudyPacks generates flashcard study packs from notes, media, and chat messages. The package resolves source evidence, builds LLM generation prompts, validates and repairs generated JSON, persists flashcards through ChaChaNotes, records provenance, and queues background generation jobs.

## Start Here

- `source_resolver.py` resolves note, media, and message sources into `StudySourceBundle` inputs.
- `generation_service.py` handles provider selection, prompt construction, LLM output validation, JSON repair, and flashcard persistence.
- `jobs.py` defines job payload and enqueue helpers for background generation.
- Related API surface: `app/api/v1/endpoints/flashcards.py`.
- Related schemas: `app/api/v1/schemas/study_packs.py`.
- Related tests: `tests/StudyPacks/`.

## Responsibilities

- Resolve source text and metadata from notes, media transcripts/chunks, and chat messages.
- Build constrained prompts that include evidence and requested generation options.
- Call the configured LLM provider and validate generated flashcard payloads.
- Attempt bounded JSON repair when generated output is malformed.
- Persist generated flashcards and link them back to source provenance.
- Queue and track background study-pack generation work.

## Module Map

- `source_resolver.py` - note/media/message source resolution and evidence bundling.
- `generation_service.py` - generation orchestration and persistence.
- `provenance.py` - provenance records for generated study content.
- `jobs.py` - background job payloads and enqueue helpers.
- `types.py` - shared dataclasses and type definitions.

## How It Connects

- `app/api/v1/endpoints/flashcards.py` exposes study-pack job creation, status, and generated flashcard integration.
- `app/services/study_pack_jobs_worker.py` runs queued study-pack generation jobs.
- ChaChaNotes provides notes, messages, flashcard persistence, and source metadata.
- Media DB and transcript/chunk helpers provide media evidence when media sources are selected.
- LLM provider configuration is resolved by the generation service.

## Extension Points

- For a new source type, extend `source_resolver.py`, `types.py`, `study_packs.py`, and source-resolution tests.
- For prompt or output schema changes, update `generation_service.py` and tests covering validation and repair.
- For provenance changes, inspect `provenance.py` and storage tests before changing endpoint responses.
- For background job behavior, update `jobs.py`, `study_pack_jobs_worker.py`, and job-worker tests together.

## Testing

- `tests/StudyPacks/test_source_resolver.py`
- `tests/StudyPacks/test_source_resolver_db_error_fallback.py`
- `tests/StudyPacks/test_generation_service.py`
- `tests/StudyPacks/test_provenance.py`
- `tests/StudyPacks/test_study_pack_jobs_worker.py`
- `tests/StudyPacks/test_study_pack_endpoints_api.py`
- `tests/StudyPacks/test_study_pack_storage.py`
- `tests/StudyPacks/test_study_pack_schemas.py`
- Related flashcard integration tests live under `tests/Flashcards/`.

## Gotchas

- Generated study packs persist as flashcards; there is not a separate study-pack content database in this package.
- Source resolution deliberately falls back around some DB chunk lookup failures, so tests should distinguish missing evidence from recoverable lookup errors.
- LLM output repair is bounded; schema drift should be fixed in prompts/schemas instead of relying on repeated repair attempts.
