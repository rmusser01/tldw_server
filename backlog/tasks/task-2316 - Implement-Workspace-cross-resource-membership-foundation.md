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
- tldw_Server_API/tests/Workspaces/test_workspace_context_membership_summary.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved first server-backed Workspace cross-resource membership slice. Scope includes ChaChaNotes persistence, fail-closed resource adapters, Workspace Core service, API schemas/endpoints, explicit backfill helper, context summary, tests, docs, and verification. Preserve the boundary that generic membership is association, not ownership transfer, global filtering, or MCP permission/path trust.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ChaChaNotes persists `workspace_resource_memberships` with SQLite/PostgreSQL schema support, idempotent create, conflict handling, soft-delete, restore, deterministic workspace listing, and reverse resource lookup.
- [x] #2 Workspace membership models, schemas, adapters, service, and API endpoints implement the approved first slice for `workspace_note`, `media`, `workspace_source`, `workspace_artifact`, and `chat`.
- [x] #3 Backfill helper is explicit and idempotent; Workspace context exposes compact membership totals without making membership a global Library/Notes/search filter.
- [x] #4 MCP permission preview/path admission remains driven by MCP policy/root bindings, not generic membership.
- [x] #5 Focused tests and Bandit verification are recorded; known skips or unrelated failures are documented.
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
Task 4 spec-review fix:
- Fixed unsupported POST resource types reaching FastAPI/Pydantic literal validation before the service by changing `WorkspaceMembershipCreateRequest.resource_type` to a raw non-empty string while keeping response schemas typed.
- Added POST regression coverage for `resource_type="note"` returning `400` with `detail.code == "unsupported_resource_type"`.
- Updated the older schema unit expectation so role/transfer-policy literals remain schema-owned while create `resource_type` validation is service-owned.
- TDD red run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py::test_post_unsupported_resource_type_returns_stable_400_code -q --tb=short` failed with `assert 422 == 400`.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py -q` passed with `12 passed, 6 warnings`.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py -q` passed with `66 passed, 6 warnings`.
- Verification: `git diff --check` passed.
- Bandit touched-scope scan reported zero findings for `workspace_schemas.py` (`/tmp/bandit_workspace_membership_schema_review_fix.json`).
Task 4 reviews after follow-up:
- Spec compliance re-review approved the API routes after the unsupported POST resource-type fix; required prefixes, filters, `resolve=false`, soft-delete/relink, reverse lookup, archived write, and missing media DB behavior remain compliant.
- Code quality review approved the endpoint implementation and focused tests; no route registration, dependency usage, response model, or error mapping issues were found.

Task 5 completed by Codex:
- Added explicit `WorkspaceMembershipService.backfill_workspace_memberships(...)` that reads workspace sources, artifacts, notes, and workspace-scoped conversations through existing helpers, then links candidates via `link_membership(resolve=False)` with stable `workspace_backfill` provenance.
- Backfill creates `workspace_source`, optional `media`, `workspace_artifact`, `workspace_note`, and `chat` memberships; repeated runs count active duplicates as existing and do not create duplicate rows.
- Backfill records bounded diagnostics capped at 25 entries with only `resource_type`, `resource_id`, `code`, and safe `message`, continuing without deleting or rewriting sub-resources.
- Added compact `memberships` totals to `WorkspaceContextResponse` and `GET /api/v1/workspaces/{workspace_id}/context`; membership summary failures return an empty summary plus `partial_errors[{scope: "memberships", code: "membership_summary_unavailable"}]`.
- Added focused tests for backfill creation, idempotency, bounded unresolved diagnostics, compact context totals without item lists, context summary fallback, and MCP capability/trust separation.
- TDD red run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_context_membership_summary.py -q` failed with 6 expected failures because backfill returned `not_implemented` and context lacked `memberships`.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_context_membership_summary.py -q` passed with `6 passed, 6 warnings`.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_context_membership_summary.py tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py -q` passed with `72 passed, 6 warnings`.
- Verification: `git diff --check` passed.
- Bandit touched runtime scan reported zero findings in `/tmp/bandit_workspace_membership_task5.json`.
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
