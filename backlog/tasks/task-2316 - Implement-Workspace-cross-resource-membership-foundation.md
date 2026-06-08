---
id: TASK-2316
title: Implement Workspace cross-resource membership foundation
status: In Progress
labels:
- workspaces
- project-workspace
- membership
- implementation
priority: high
references:
- https://github.com/rmusser01/tldw_server/issues/1990
- https://github.com/rmusser01/tldw_server/issues/1984
- Docs/superpowers/specs/2026-06-07-workspace-cross-resource-membership-design.md
documentation:
- Docs/superpowers/specs/2026-06-07-workspace-cross-resource-membership-design.md
- Docs/superpowers/plans/2026-06-07-workspace-cross-resource-membership-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/tests/ChaChaNotesDB/test_workspace_resource_memberships_db.py
- tldw_Server_API/app/core/Workspaces/membership_models.py
- tldw_Server_API/app/api/v1/schemas/workspace_schemas.py
- tldw_Server_API/app/core/Workspaces/membership_adapters.py
- tldw_Server_API/app/core/Workspaces/membership_service.py
- tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py
- tldw_Server_API/app/api/v1/endpoints/workspaces.py
- tldw_Server_API/app/api/v1/endpoints/workspace_memberships.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/api/v1/router_groups/minimal.py
- tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved first server-backed Workspace cross-resource membership slice. Scope includes ChaChaNotes persistence, fail-closed resource adapters, Workspace Core service, API schemas/endpoints, explicit backfill helper, context summary, tests, docs, and verification. Preserve the boundary that generic membership is association, not ownership transfer, global filtering, or MCP permission/path trust.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ChaChaNotes persists `workspace_resource_memberships` with SQLite/PostgreSQL schema support, idempotent create, conflict handling, soft-delete, restore, deterministic workspace listing, and reverse resource lookup.
- [x] #2 Workspace membership models, schemas, adapters, service, and API endpoints implement the approved first slice for `workspace_note`, `media`, `workspace_source`, `workspace_artifact`, and `chat`.
- [ ] #3 Backfill helper is explicit and idempotent; Workspace context exposes compact membership totals without making membership a global Library/Notes/search filter.
- [ ] #4 MCP permission preview/path admission remains driven by MCP policy/root bindings, not generic membership.
- [ ] #5 Focused tests and Bandit verification are recorded; known skips or unrelated failures are documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-07-workspace-cross-resource-membership-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Execution started with subagent-driven Task 1: persistence and DB contract.

Task 1 completed and reviewed:
- `f215366949` added `workspace_resource_memberships` persistence and focused DB tests.
- `227a54fa83` fixed `limit + 1` pagination behavior and added missing delete/backend-error coverage.
- `924aafeb0a` made restore/delete state transitions race-idempotent with state-predicated updates and rowcount handling.
- Spec compliance review approved the DB contract after follow-up.
- Code quality review approved the concurrency fix and found no remaining Task 1 issues.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_resource_memberships_db.py -q` passed with `21 passed, 6 warnings`.
- Verification: `git diff --check 243e0b63c9..924aafeb0a66919201b60b4c19414a399f2b81bf` passed.
- Worker Bandit touched-scope scan reported zero findings for `ChaChaNotes_DB.py`.

Task 2 completed by Worker 2:
- Added `membership_models.py` constants, dataclasses, and base64-url JSON cursor helpers for workspace and reverse resource membership pagination.
- Added Workspace membership request/response/list/context summary schemas with Literal validation and 16 KiB provenance/metadata JSON bounds.
- Added focused model/schema tests in `tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py`.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q` passed with `21 passed, 6 warnings`.
- Verification: `git diff --check` passed.
- Bandit touched-scope scan reported zero findings for `membership_models.py` and `workspace_schemas.py`.
Task 2 follow-up review fixes:
- Rejected non-finite floats in membership provenance/metadata by making schema JSON sizing use strict JSON serialization (`allow_nan=False`).
- Added a 2048-character encoded cursor cap before base64 decode and exact integer cursor-version validation for both cursor shapes.
- Added regression coverage for NaN/Infinity metadata/provenance, oversized cursors, and boolean/float cursor versions.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q` passed with `33 passed, 6 warnings`.
- Verification: `git diff --check` passed.
- Bandit touched-scope scan reported zero findings for `membership_models.py` and `workspace_schemas.py`.
Task 2 follow-up self-review update:
- Added an oversized-cursor regression that monkeypatches base64 decode, proving the length guard runs before decode.
- Final focused verification after that addition: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q` passed with `34 passed, 6 warnings`.
- Spec compliance review approved Task 2 after checking constants, dataclasses, cursor helpers, schema literals, bounded JSON validation, response/list/context summaries, and no endpoint/service wiring.
- Code quality review approved Task 2 after the hardening follow-up and found no remaining model/schema issues.

Task 3 completed by Worker 3:
- Added fail-closed Workspace membership adapters for `workspace_note`, `workspace_source`, `workspace_artifact`, `media`, and `chat`.
- Added `WorkspaceMembershipService` with link/get/list/reverse-list/unlink/summary orchestration, archived write rejection, restore-after-soft-delete, cursor payloads, and unresolved per-row summaries for adapter read failures.
- Added focused adapter/service tests to `tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py`.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q` passed with `48 passed, 6 warnings`.
- Verification: `git diff --check` passed.
- Bandit touched-scope scan reported zero findings for `membership_adapters.py` and `membership_service.py`.
Task 3 spec-review coverage follow-up:
- Added missing-workspace service coverage for `workspace_not_found` / 404.
- Added unlink coverage for successful soft-delete hook invocation and missing/already-deleted no-op behavior.
- Added reverse lookup fail-closed coverage for unsupported resource types.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q` passed with `52 passed, 6 warnings`.
Task 3 code-quality follow-up:
- First attempted to gate `on_link` by active-row retries, then replaced that with the final reserved-hook design below after code-quality re-review exposed race and retry ambiguity.
- Made generic summary adapter failures return a safe unresolved message instead of raw exception text.
- Changed direct `get_membership` to read existing memberships before resource resolution so unavailable backing services produce unresolved summaries instead of hard failures.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q` passed with `55 passed, 6 warnings`.
- Bandit touched-scope scan reported zero findings for `membership_service.py`.
Task 3 on-link race follow-up:
- Reserved `on_link` as a future transition-aware adapter hook and stopped invoking it from first-slice `link_membership`.
- This avoids duplicate side effects under insert/restore races and avoids non-retryable post-commit hook failures until the DB contract or an outbox can report durable transitions.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q` passed with `55 passed, 6 warnings`.
- Bandit touched-scope scan reported zero findings for `membership_adapters.py` and `membership_service.py`.
- Spec compliance review approved Task 3 after coverage follow-up.
- Code quality review approved Task 3 after the reserved `on_link` correction and found no remaining adapter/service issues.

Task 4 completed by Worker 4:
- Added Workspace membership API routes under `/api/v1/workspaces/{workspace_id}/memberships` for list, create, get, and soft-delete.
- Added reverse resource lookup router under `/api/v1/workspace-memberships/resources/{resource_type}/{resource_id}` with the resource-scoped response shape.
- Registered the reverse router in the content and minimal router groups using the Workspace route gate.
- Added focused FastAPI endpoint tests for create, idempotent duplicate, conflicting duplicate, archived write rejection, filtered list with `resolve=false`, get/missing get, soft-delete hiding, relink restore, reverse lookup, unsupported resource type, and missing media DB.
- TDD red run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py -q` failed with `11 failed` because the membership routes returned `404`.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py -q` passed with `11 passed, 6 warnings`.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py -q` passed with `55 passed, 6 warnings`.
- Verification: `git diff --check` passed.
- Bandit touched-scope scan reported zero findings for `workspaces.py`, `workspace_memberships.py`, and `workspace_schemas.py` (`/tmp/bandit_workspace_membership_api_task2316.json`).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
