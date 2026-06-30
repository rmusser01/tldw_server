# Setup

The Setup module owns first-run setup state, config-file updates, provider
validation, local audio resource readiness, and install-plan previews. It keeps
setup mutations explicit and sanitized so `/setup` and admin setup endpoints can
guide a new deployment without leaking secrets or silently rewriting unrelated
configuration.

## Start Here

- Config management: `setup_manager.py`.
- First-run state: `first_run_state.py` and `first_run_models.py`.
- Readiness and install planning: `readiness_service.py`, `readiness_store.py`,
  `readiness_profiles.py`, `install_manager.py`, and `install_schema.py`.
- Audio setup packs: `audio_bundle_catalog.py`, `audio_profile_service.py`,
  `audio_pack_service.py`, and `audio_readiness_store.py`.
- API endpoint and schemas: `app/api/v1/endpoints/setup.py` and
  `app/api/v1/schemas/setup_schemas.py`.
- Tests: `tests/Setup/`.

## Responsibilities

- Read, validate, mask, preview, and update `Config_Files/config.txt`.
- Track first-run progress and setup readiness overlays.
- Build install plans for audio/STT/TTS/embedding resources without performing
  unexpected writes during preview paths.
- Validate provider keys/settings and support the first-chat readiness check.
- Coordinate remote setup access policy with Security setup guards.

## Module Map

- `setup_manager.py` centralizes config section metadata and safe writes.
- `provider_catalog.py` and `provider_validation.py` describe and validate
  provider setup inputs.
- `install_manager.py` builds and executes controlled install plans.
- `readiness_*` modules store and compute setup readiness state.
- `audio_*` modules rank, package, and track local audio resource bundles.
- `first_chat_verifier.py` checks whether configured chat providers can respond.

## How It Connects

- Security middlewares guard remote setup access and Setup UI CSP.
- AuthNZ setup dependencies decide when setup routes are public, admin-only, or
  disabled.
- Config section loaders in `config_sections/` consume settings written here.

## Architecture Notes

### Core Flow

- Setup endpoints call setup access guards and AuthNZ setup dependencies before reading or mutating first-run state, readiness overlays, provider settings, or install plans.
- `setup_manager.py` reads, masks, previews, and writes `Config_Files/config.txt`, then clears config caches after controlled writes.
- Readiness preview paths build install/config plans without side effects; provision paths run only after explicit confirmation.
- Audio setup services rank bundles and track local audio pack readiness separately from provider key validation.

### State And Data

- Setup state spans `Config_Files/config.txt`, first-run state JSON, readiness stores, readiness profiles, install-plan schemas, and audio readiness stores.
- Config section metadata controls which values are visible, masked, inserted, or validated.
- Install plans describe actions before execution so the API can show users what will change.

### Security And Operations

- Mutating setup routes are local-only unless remote setup access is explicitly enabled and the caller passes admin checks.
- Proxy headers are trusted only through the configured setup access guard path.
- Provider keys and config values must stay masked in responses and logs.
- Preview endpoints must remain side-effect-free; tests should fail if previews write files or start installs.

### Extension Checklist

- New config field: update setup metadata, schema, masking/default tests, and config section readers.
- New installer: update install schema, preview/provision logic, and readiness tests.
- New remote setup behavior: update Security setup access guard tests and Setup endpoint tests together.

## Extension Points

- Add new setup fields by updating config metadata in `setup_manager.py`, schema
  fields in `setup_schemas.py`, and tests for masking/defaults.
- Add resource installers through `install_schema.py` and `install_manager.py`;
  preview behavior should stay side-effect-free.

## Testing

- Config masking and insertion: `tests/Setup/test_setup_manager_masking.py` and
  `tests/Setup/test_setup_manager_provider_field_insertion.py`.
- Readiness APIs and stores: `tests/Setup/test_setup_readiness_api.py`,
  `tests/Setup/test_setup_readiness_preview.py`, and
  `tests/Setup/test_setup_readiness_store.py`.
- Audio bundle/profile/pack flows: `tests/Setup/test_audio_bundle_catalog.py`,
  `tests/Setup/test_audio_profile_service.py`, and
  `tests/Setup/test_audio_pack_service.py`.

## Gotchas

- Preview endpoints must not write config or resource files.
- Never log provider keys or unmasked config values.
- Remote setup access and CSP behavior are security-sensitive; update
  `tests/Security/` when changing those hooks.
