# Writing

The Writing module contains backend helpers for the Writing Playground, manuscript
workflows, manuscript analysis, tokenization support, word-cloud jobs, and note
title generation. Most HTTP behavior lives in the writing endpoints, while this
core package keeps reusable analysis and title-generation logic close to the
ChaChaNotes/Manuscript DB helpers that persist drafts and manuscripts.

## Start Here

- Core helpers: `manuscript_analysis.py` and `note_title.py`.
- API endpoints: `app/api/v1/endpoints/writing.py` for Writing Playground
  sessions/templates/themes/snapshots/tokenizers/word clouds, and
  `app/api/v1/endpoints/writing_manuscripts.py` for manuscript projects,
  structure, characters, world info, plot lines, citations, research, and
  analysis.
- Schemas: `app/api/v1/schemas/writing_schemas.py` and
  `app/api/v1/schemas/writing_manuscript_schemas.py`.
- Persistence: Writing sessions/templates/themes/word clouds are methods on
  `app/core/DB_Management/ChaChaNotes_DB.py`; manuscript CRUD is mediated by
  `app/core/DB_Management/ManuscriptDB.py`.
- Tests: `tests/Writing/`, `tests/ChaChaNotesDB/test_writing_playground_db.py`,
  and note-title coverage in `tests/Notes/`.

## Responsibilities

- Run structured manuscript analysis prompts for pacing, plot holes, and
  consistency, then normalize provider responses into JSON-ish dictionaries.
- Generate note titles through a heuristic path, with an optional LLM-backed path
  behind `NOTES_TITLE_LLM_ENABLED`.
- Support endpoint-layer writing operations by documenting the core DB,
  tokenizer, rate-limit, and LLM dependencies used by those routes.
- Keep error surfaces sanitized: endpoint helpers map DB and provider failures to
  HTTP errors without leaking private paths or backend exception details.

## Module Map

- `manuscript_analysis.py` builds literary analysis prompts, calls the Chat
  service, strips Markdown fences, extracts content from provider response
  shapes, and parses JSON output.
- `note_title.py` extracts a title from free-form note content, truncates safely,
  and optionally asks an LLM adapter for a concise title.
- `__init__.py` intentionally exports no broad facade; call specific helpers
  directly.

## How It Connects

- The writing endpoints use `get_chacha_db_for_user`, AuthNZ dependencies, and
  `rbac_rate_limit("writing.*")` scopes to enforce per-user access and rate
  controls.
- Tokenizer operations are delegated to shared `LLM_Calls.tokenizer_resolver`
  helpers, including provider-native token counting when available.
- Manuscript analysis calls `Chat.chat_service.perform_chat_api_call_async`, so
  provider selection and model validation follow the normal Chat/LLM path.
- Writing Playground data is stored in the per-user ChaChaNotes DB; manuscript
  structures use the manuscript DB helper layered over the same per-user DB.

## Extension Points

- Add new analysis types by defining a prompt and wrapper in
  `manuscript_analysis.py`, then expose it through
  `writing_manuscripts.py` and `writing_manuscript_schemas.py`.
- Add title strategies by extending `TitleStrategy` and keeping heuristic
  fallback behavior deterministic.
- Add Writing Playground entities through the ChaChaNotes DB adapter first, then
  wire schemas and endpoint handlers around those adapter methods.

## Testing

- Endpoint and DB flows: `tests/Writing/test_writing_endpoint_integration.py`,
  `tests/Writing/test_manuscript_endpoint_integration.py`,
  `tests/Writing/test_manuscript_phase2_integration.py`, and
  `tests/ChaChaNotesDB/test_writing_playground_db.py`.
- Analysis service behavior: `tests/Writing/test_manuscript_analysis_service.py`
  and `tests/Writing/test_manuscript_analysis_integration.py`.
- Error mapping and sanitizer coverage: `tests/Writing/test_writing_error_mapping.py`.
- Tokenizer metadata and fallback behavior: `tests/Writing/test_tokenizer_resolver_unit.py`
  and `tests/Writing/test_writing_tokenizer_sanitizers.py`.

## Gotchas

- `manuscript_analysis.py` truncates text and context before sending it to the
  LLM. Preserve those bounds or add explicit tests when changing prompt sizes.
- `note_title.py` keeps LLM generation best-effort and synchronous; failures must
  fall back to deterministic heuristic titles.
- Writing endpoints have many RBAC scope strings. When adding routes, keep scope
  names aligned with `PrivilegeMaps` and AuthNZ permission tests.
