# config_sections

`config_sections` is a support package for typed configuration section loading. It is not a user-facing feature module; it breaks the larger configuration loader into small section-specific dataclasses and helpers used by `app/core/config.py` and adjacent policy code.

## Start Here

- `__init__.py` exports `ConfigSections`, `load_config_sections`, and section-specific loader helpers.
- `types.py` defines shared configuration parser protocols.
- Section files such as `auth.py`, `database.py`, `server.py`, `stt.py`, and `providers.py` contain typed dataclasses and loaders for specific config areas.
- Related runtime entry point: `app/core/config.py`.
- Related tests: `tests/Config/test_config_sections_typed_loaders.py`.

## Responsibilities

- Load individual config sections into typed dataclasses instead of raw dictionaries.
- Keep section parsing for auth, audio, chat, chunking, database, embeddings, jobs, logging, moderation, providers, RAG, server, and STT separated by concern.
- Provide an aggregate `ConfigSections` object for callers that need multiple typed sections.
- Support config/environment precedence behavior through the parent configuration loader.
- Give STT policy code typed access to vNext STT config flags.

## Module Map

- `__init__.py` - aggregate exports and top-level loader helpers.
- `types.py` - shared parser type definitions.
- `audio.py`, `auth.py`, `chat.py`, `chunking.py`, `database.py`, `embeddings.py`, `jobs.py`, `logging.py`, `moderation.py`, `providers.py`, `rag.py`, `server.py`, and `stt.py` - section dataclasses and loaders.

## How It Connects

- `app/core/config.py` imports the aggregate loaders and exposes the effective application configuration.
- `app/core/Ingestion_Media_Processing/Audio/stt_policy.py` reads typed STT configuration from this package.
- Config endpoint and effective-config tests cover behavior that flows through these typed loaders.
- Documentation for the modularization work lives under `Docs/superpowers/plans/2026-03-02-config-modularization-*`.

## Extension Points

- For a new config section, add a section module, export it from `__init__.py`, and add typed-loader tests.
- For new keys in an existing section, update the section dataclass/loader and the relevant config precedence tests.
- For STT config flags, inspect `stt.py` and `stt_policy.py` together.

## Testing

- `tests/Config/test_config_sections_typed_loaders.py`
- `tests/Audio/test_stt_vnext_config_flags.py`
- `tests/Config/test_effective_config_api.py`

## Gotchas

- This package should stay lightweight. Avoid importing provider, database, or service modules from section loaders.
- Environment/config precedence is handled with the parent config loader, so new section logic should be tested through the same path callers use.
