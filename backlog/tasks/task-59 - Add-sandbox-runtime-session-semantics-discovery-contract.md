---
id: TASK-59
title: Add sandbox runtime session semantics discovery contract
status: Done
assignee: []
created_date: '2026-05-05 03:15'
labels:
  - sandbox
  - runtime-discovery
  - phase-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a narrow Phase 4 sandbox runtime parity slice that exposes stable session semantics metadata in runtime discovery. This should let clients distinguish real warm reuse from scaffold or host-local session support without runtime-specific notes parsing, while avoiding behavior changes to session execution or repair.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime discovery exposes a structured session contract for every RuntimeType without changing session execution behavior.
- [x] #2 Session contract metadata distinguishes support state, reuse model, health check expectation, and recovery/repair posture for Docker, Firecracker, Lima, vz_linux, vz_macos, seatbelt, and worktree.
- [x] #3 Focused tests verify all runtimes include session contract metadata and the current vz_linux warm VM reuse/health semantics remain distinct from scaffold and host-local runtimes.
- [x] #4 Sandbox runtime inventory documentation is updated to replace or narrow the current session-semantics gap.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a narrow discovery-only `session_contract` metadata layer for every sandbox runtime. This keeps runtime behavior unchanged while making workspace-only, warm VM, and scaffolded session semantics machine-readable.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py tldw_Server_API/tests/Docs/test_sandbox_public_docs_contract.py -q` passed with 30 tests.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Sandbox/runtime_capabilities.py tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py -f json -o /tmp/bandit_sandbox_session_contract.json` completed with no findings.
- Verification: `git diff --check` passed.
- Known limitation: the broader TestClient feature-discovery suite timed out in unrelated full-app lifespan teardown while background workers/schedulers were active. The branch adds direct `SandboxRuntimesResponse` validation of `SandboxService.feature_discovery()` to cover the endpoint response schema without starting full app lifespan.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exposed static sandbox runtime session semantics through discovery using a complete metadata map, schema model, and service wiring. Updated sandbox runtime inventory and public API docs so clients can distinguish workspace-only sessions, scaffolded session shapes, and `vz_linux` warm VM reuse/recovery posture without parsing runtime notes.
<!-- SECTION:FINAL_SUMMARY:END -->
