# UserProfiles

UserProfiles assembles readable and editable user profile state from AuthNZ identity, memberships, security settings, quotas, catalog-controlled fields, and user/org/team overrides. The package backs profile read/update endpoints while keeping editable keys and effective profile layering explicit.

## Start Here

- `service.py` builds user profile responses from AuthNZ, membership, quota, and override sources.
- `update_service.py` validates and applies catalog-controlled profile updates.
- `overrides_repo.py` persists user/org/team profile overrides.
- `user_profile_catalog.py` loads and validates the editable profile catalog.
- Related API surface: `app/api/v1/endpoints/users.py`.
- Related tests: `tests/UserProfile/`.

## Responsibilities

- Build profile identity, account state, memberships, security summary, quota summary, raw overrides, and effective layered profile data.
- Load the profile catalog that defines editable keys, value types, defaults, and role/permission constraints.
- Validate profile update requests against the catalog and caller permissions.
- Persist overrides at supported scopes and compute effective profile layers.
- Surface update/read errors through profile-specific error mapping utilities.
- Support adjacent quota and personalization code that reads effective profile values.

## Module Map

- `service.py` - profile read service and effective profile assembly.
- `update_service.py` - profile update validation and persistence orchestration.
- `overrides_repo.py` - override storage repository.
- `user_profile_catalog.py` - catalog loading and validation.

## How It Connects

- `app/api/v1/endpoints/users.py` exposes profile read, update, admin, bulk, audit, and legacy compatibility routes.
- `app/api/v1/schemas/user_profile_schemas.py` defines profile request and response models.
- `app/api/v1/utils/profile_errors.py` maps profile service errors to API responses.
- AuthNZ repositories provide user, membership, permission, and quota context.
- `app/core/Usage/audio_quota.py` and prompt-studio quota configuration read profile or quota-related state.

## Architecture Notes

### Core Flow

- Profile reads enter through `users.py`, resolve the authenticated principal,
  and call `UserProfileService` to assemble identity, membership, security,
  quota, raw override, and effective config sections.
- Profile updates go through `UserProfileUpdateService`, which loads the
  editable catalog, validates requested keys and values, checks actor roles
  against catalog edit rules, and writes user/org/team overrides through
  `overrides_repo.py`.
- API schemas keep raw overrides, effective config, catalog entries, bulk
  updates, and structured profile errors explicit for clients.

### State And Security

- Override state is layered from defaults, user overrides, org overrides, team
  overrides, and AuthNZ-derived facts. Contributors must preserve deterministic
  precedence when adding scopes or keys.
- `user_profile_catalog.py` is the authority for editable keys, value types,
  default values, and role constraints. Endpoint payloads must not create
  arbitrary profile keys.
- Profile responses include security, BYOK, quota, and membership summaries;
  logs and error responses must stay sanitized.

### Extension Checklist

- New editable key: update the catalog loader/input data, `update_service.py`
  validation, response schemas when needed, and `tests/UserProfile/`.
- New override scope: add repository support, effective-layer assembly, role
  validation, and migration/test coverage for both SQLite and Postgres paths.
- New profile endpoint behavior: update `users.py`, `user_profile_schemas.py`,
  `profile_errors.py`, and API tests together.

## Extension Points

- For a new editable profile field, update the catalog, `user_profile_catalog.py`, `update_service.py`, and profile update tests.
- For a new profile layer or scope, update `overrides_repo.py`, effective-layer assembly in `service.py`, and effective-layer tests.
- For admin/bulk behavior, inspect `users.py`, `user_profile_schemas.py`, and the `tests/UserProfile/` API coverage.
- For quota or security profile data, keep AuthNZ repository boundaries explicit in `service.py`.

## Testing

- `tests/UserProfile/`
- `tests/AuthNZ/unit/test_user_profile_update_service_backend_selection.py`
- `tests/Admin/test_admin_service_log_sanitizers.py`
- `tests/Audio/test_audio_quota_unit.py`

## Gotchas

- The catalog controls which keys are editable and by whom; do not accept arbitrary profile keys from requests.
- Effective profile values may combine defaults, user overrides, team/org overrides, and AuthNZ-derived state.
- Profile responses contain security and quota-adjacent data, so logs and error messages should stay sanitized.
