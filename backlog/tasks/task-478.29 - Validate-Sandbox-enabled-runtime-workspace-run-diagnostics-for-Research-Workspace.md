---
id: TASK-478.29
title: Validate Sandbox enabled-runtime workspace run diagnostics for Research Workspace
status: Done
labels:
- research-workspace
- sandbox
- uat
- workspace-model
priority: High
milestone: Research Workspace UAT Remediation
ordinal: 29
parent_task_id: TASK-478
references:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md
- TASK-478.23
- TASK-478.24
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the remaining RW-UAT-023 Partial gap in an environment where the sandbox route/runtime is enabled. Validate that a real workspace-linked sandbox run can be created for or from the active Research Workspace ID and appears in the workspace-scoped diagnostics list. Preserve sandbox ownership of run execution and diagnostics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validation environment has sandbox route/runtime enabled, or the task records the exact blocker without changing RW-UAT-023 to Pass.
- [x] #2 Live fixture creates a real sandbox run with the active Research Workspace ID in workspace context.
- [x] #3 Workspace-scoped sandbox diagnostics returns the created run and exposes truthful admission/runtime state.
- [x] #4 Research Workspace opens sandbox-owned diagnostics without owning sandbox execution state.
- [x] #5 RW-UAT-023 is updated only as far as live backend + WebUI + CDP evidence supports.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `sandbox_execution_disabled` admission handling so route-enabled environments with `SANDBOX_ENABLE_EXECUTION=0` report blocked admission instead of implying sandbox workspace actions may run.
- Added backend regression coverage for execution-disabled admission while preserving the existing route-disabled and runtime-unavailable diagnostics behavior.
- Added strict-mode controls to the Research Workspace real-backend sandbox E2E: `TLDW_E2E_REQUIRE_SANDBOX_WORKSPACE_RUN=1` fails instead of skipping when run creation is unavailable, and `TLDW_E2E_EXPECT_SANDBOX_RUN_PHASE` asserts fixture phase when set.
- Live startup under the normal sandbox failed to bind localhost with `[Errno 1] ... operation not permitted`; rerunning the backend outside the sandbox allowed live validation.
- Normal live startup rejects `TLDW_TEST_MODE=1` outside pytest, so enabled-route validation used `/private/tmp/tldw_task47829_config.txt` with `sandbox` added to `[API-Routes].enable`.
- Docker CLI is installed, but the host Docker daemon was unavailable (`Cannot connect to the Docker daemon at unix:///Users/macbook-dev/.docker/run/docker.sock`). The enabled-runtime proof therefore used `SANDBOX_ENABLE_EXECUTION=1`, `TLDW_SANDBOX_DOCKER_AVAILABLE=1`, and `TLDW_SANDBOX_DOCKER_FAKE_EXEC=1`; this is a fake-execution fixture, not a real Docker daemon proof.
- Direct enabled-route API evidence on backend `127.0.0.1:18031`: `POST /api/v1/sandbox/runs` created run `a309eec3-2f7d-413c-86c9-0a096408a984` for workspace `task47829-direct-workspace`; diagnostics returned `runtime.state=available`, `admission.state=available`, and `runs.total=1`.
- WebUI/CDP evidence on backend `127.0.0.1:18031` and WebUI `127.0.0.1:18090`: focused Playwright created run `75582dd2-11f3-455a-bb44-52792f480f32` for active Research Workspace `6c6dd021-48b5-474a-ba1b-4ffe70c07f0b`; diagnostics returned the run with `phase=completed`, `workspace_group_id=research-workspace`, and `message=Docker fake execution`.
- Strict route-disabled evidence on default backend `127.0.0.1:18030`: focused Playwright failed closed with `POST /api/v1/sandbox/runs returned HTTP 404: {"detail":"Not Found"}`.
- Non-strict route-disabled compatibility evidence on default backend `127.0.0.1:18032`: the same focused Playwright test skipped when `TLDW_E2E_REQUIRE_SANDBOX_WORKSPACE_RUN` was unset.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-478.29 closes RW-UAT-023 as fixture-backed Pass for the Research Workspace to Sandbox handoff. The backend now reports execution-disabled sandbox admission truthfully, the real-backend E2E can fail closed when sandbox run creation is required, and live WebUI/CDP validation proved that an active Research Workspace ID can create a sandbox-owned run and reopen it through sandbox-owned workspace diagnostics. The only remaining caveat is runtime fidelity: local Docker daemon access was unavailable, so the enabled-route proof used the existing fake Docker execution fixture rather than a real Docker container.

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
