---
id: TASK-15638
title: Persist the Research Workspace origin of native Runs
status: Done
labels:
- research
- workspace
- api
references:
- Docs/ADR/102-research-workspace-run-origin.md
- Docs/Design/research-workspace-run-origin.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Chatbook recover an authenticated Server run and its exact workspace origin after restart or a lost launch response.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace launch checks the authenticated account, workspace, and captured source and output versions.
- [x] #2 Repeated and concurrent requests recover one native run and reject mismatched or deleted operations.
- [x] #3 The run detail provides a verified workspace origin for return while the existing Runs and Chat paths preserve their behavior.
- [x] #4 Focused API, persistence, dispatch, and OpenAPI checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: yes. ADR path: Docs/ADR/102-research-workspace-run-origin.md. The dedicated Workspace API and durable native Runs receipt define the authenticated launch and return boundary.
The Workspace launch stores an owner-scoped receipt with the exact native run and its first phase dispatch intent. Native detail and bundle reads project the launch to that account. Concurrent replay, interruption before and after Jobs admission, deleted operation replay, and changed references are covered. The focused Server run passed 144 tests. The PostgreSQL-backed existing Chat ownership test passed separately against a disposable database. OpenAPI fingerprint and path union checks, Ruff, and Bandit on changed production paths passed. A localhost Chatbook smoke test dropped the successful launch response, recovered one run, and saved a verified draft Report using a stored bundle fixture.
The rebased CI run records the launch receipt as SQLite-only and verifies owner-scoped lookup across the supported deployment modes. The tenant isolation checks and focused Workspace Study and Runs regressions passed locally.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Authenticated Research Workspace launches can be recovered and verified from native Runs after a restart without re-launching or delivering a Chat message.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
