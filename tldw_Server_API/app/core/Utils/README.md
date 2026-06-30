# Utils

The Utils package holds shared helpers that are safe to import from endpoints,
services, and core modules. It covers filesystem path helpers, prompt loading,
metadata normalization, image validation, tokenizer helpers, pydantic
compatibility, CPU-bound execution, and optional system checks.

## Start Here

- General helpers: `Utils.py`, `common.py`, `path_utils.py`, and
  `metadata_utils.py`.
- Prompt and schema compatibility: `prompt_loader.py` and
  `pydantic_compat.py`.
- Runtime helpers: `cpu_bound_handler.py`, `executor_registry.py`,
  `tokenizer.py`, and `torch_import_guard.py`.
- Media helpers: `chunked_image_processor.py` and `image_validation.py`.
- Tests: `tests/Utils/` plus module-specific callers across media, TTS, config,
  and prompt tests.

## Responsibilities

- Provide low-level helpers without importing heavyweight app subsystems at
  module import time.
- Normalize metadata, safe filenames, paths, and project-directory lookups used
  by ingestion and API handlers.
- Centralize compatibility shims that keep Pydantic v1/v2 call sites readable.
- Offer bounded CPU/offload helpers for work that should not block the event
  loop.

## Module Map

- `Utils.py` contains legacy general-purpose helpers and project path helpers.
- `metadata_utils.py` normalizes safe metadata dictionaries for media APIs.
- `prompt_loader.py` loads prompt resources from configured namespaces.
- `pydantic_compat.py` wraps model dump/validation compatibility.
- `executor_registry.py` and `cpu_bound_handler.py` coordinate executor reuse.
- `torch_import_guard.py` keeps optional ML imports explicit and diagnosable.

## How It Connects

- Web scraping, setup, user profiles, media endpoints, TTS, and prompt flows all
  import these helpers directly.
- This package should stay dependency-light because it is imported from many
  startup paths.

## Extension Points

- Add helpers here only when they are reused by multiple modules or remove a
  real circular dependency.
- Prefer narrowly named modules over growing `Utils.py`.

## Testing

- Utility contract tests live under `tests/Utils/`.
- Caller tests often cover helper behavior through higher-level flows, especially
  media metadata, prompt loading, Docker/profile config, and pagination tests.

## Gotchas

- Avoid adding provider SDKs, database imports, or FastAPI dependencies here.
  Those imports can slow startup or create circular dependencies.
- Keep filesystem helpers explicit about whether they return project-root,
  config-root, user-db, or caller-provided paths.
