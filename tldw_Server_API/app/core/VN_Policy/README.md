# VN_Policy

VN_Policy owns visual novel policy profile and safety-definition behavior. It provides the service layer for creating, reading, updating, deleting, and evaluating VN policy data that is consumed by VN play and script authoring flows.

## Start Here

- `service.py` contains the VN policy service and safety-definition evaluation logic.
- Related API surface: `app/api/v1/endpoints/vn_policy.py`.
- Related schemas: `app/api/v1/schemas/vn_policy_schemas.py`.
- Related tests: `tests/VN_Policy/`.

## Responsibilities

- Manage VN policy profiles and safety definitions through the service layer.
- Evaluate character safety definitions and return deterministic policy results.
- Coordinate policy persistence through `VNPolicy_DB`.
- Provide policy snapshots for VN play and script flows that need stable runtime behavior.
- Surface domain errors to the VN policy API.

## Module Map

- `service.py` - policy profile/safety definition service and evaluation helpers.

## How It Connects

- `app/api/v1/endpoints/vn_policy.py` exposes policy and safety-definition routes.
- `app/core/DB_Management/VNPolicy_DB.py` stores policy profiles and safety definitions.
- `app/core/VN_Play/` consumes policy profiles during runtime setup and generated turns.
- `app/core/VN_Scripts/` uses policy data during authoring and validation flows.

## Architecture Notes

### Core Flow

- VN policy endpoints build `VNPolicyService` with the current user and `VNPolicyProfileStore`, then delegate profile CRUD and evaluation.
- `service.py` resolves built-in or stored policy/generation profiles, validates definitions, and evaluates character safety metadata into a deterministic decision and reason list.
- VN Play and VN Scripts consume policy profile ids, profile definitions, and snapshots so runtime and published-script behavior stays stable.

### State And Data

- `VNPolicy_DB.py` owns profile persistence, versioning, disabled rows, and user/global visibility rules.
- API schemas define the shared shape for policy profiles, generation profiles, evaluation requests, and evaluation responses.
- Snapshot consumers depend on profile id, definition, and version fields staying coordinated across service, DB, and schemas.

### Security And Operations

- Profile mutation is admin-only in the endpoint; normal users can evaluate and read usable profiles.
- Evaluation should remain deterministic and free of provider calls so VN authoring and playback can safely reuse results.
- Schema changes must be coordinated with VN Play and VN Scripts because both modules validate policy-dependent manifests and turns.

### Extension Checklist

- New policy field: update schemas, service validation, `VNPolicy_DB.py`, API tests, and VN consumer tests.
- New evaluation rule: update `service.py`, service tests, and script/playback policy validation paths.
- New profile visibility behavior: update endpoint authorization, DB store tests, and profile list/read tests.

## Extension Points

- For new policy fields, update `service.py`, `vn_policy_schemas.py`, `VNPolicy_DB.py`, and policy API tests.
- For safety evaluation changes, start with the evaluation helpers in `service.py` and `tests/VN_Policy/test_vn_policy_service.py`.
- For consumer behavior, inspect VN play and VN script tests before changing policy response shape.

## Testing

- `tests/VN_Policy/test_vn_policy_service.py`
- `tests/VN_Policy/test_vn_policy_api.py`
- `tests/VN_Policy/test_vn_policy_db.py`

## Gotchas

- Policy evaluation should remain deterministic because VN play and authoring flows can snapshot and reuse policy results.
- Endpoint, DB, and runtime consumers all depend on the same policy shapes, so schema changes need coordinated tests.
