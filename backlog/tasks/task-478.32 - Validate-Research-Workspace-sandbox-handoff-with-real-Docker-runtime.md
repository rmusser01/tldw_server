---
id: TASK-478.32
title: Validate Research Workspace sandbox handoff with real Docker runtime
status: Done
labels:
- research-workspace
- uat
- sandbox
- workspaces
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
references:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md
- TASK-478.29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up the fixture-backed Sandbox Pass row by validating Research Workspace sandbox handoff against a real Docker-backed sandbox runtime. TASK-478.29 proved the route-enabled contract with fake Docker execution because the host Docker daemon was unavailable; this task should repeat the strict live backend + WebUI/CDP path with a real Docker daemon so release/runtime fidelity is not inferred from the fake-execution fixture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A live backend is started with sandbox routes and real Docker execution enabled, without `TLDW_SANDBOX_DOCKER_FAKE_EXEC=1`.
- [x] #2 A canonical Research Workspace ID can create or observe a workspace-linked sandbox run through sandbox-owned APIs/surfaces.
- [x] #3 Research Workspace sandbox diagnostics show the active workspace ID, runtime/admission state, and the real run without duplicating sandbox execution state inside Research Workspace.
- [x] #4 Failure states distinguish Docker unavailable, sandbox admission blocked, route disabled, and run failed/triaged states with actionable evidence.
- [x] #5 `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md` is updated only as far as live evidence supports, preserving the no `/workspace-playground` alias/redirect rule.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Start from the strict TASK-478.29 live Playwright/backend fixture and remove the fake-execution path.
- Use a live backend, WebUI, and CDP/Playwright walkthrough; capture exact API responses, run IDs, diagnostics state, and screenshot paths.
- If the current host still lacks a reachable Docker daemon, record the blocker and leave the matrix wording at fixture-backed Pass/watch risk rather than overclaiming real-runtime validation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated the Research Workspace sandbox handoff against a live backend, WebUI, CDP/Playwright, and real Docker daemon with fake execution disabled. During validation, real Docker runs exposed that the Docker runner was copying inline/session files into a stopped read-only container, which fails before the `/workspace` tmpfs exists. Fixed the runner by mounting a read-only staged input directory and copying it into the hardened `/workspace` tmpfs at container start, preserving Sandbox ownership of execution state.

Evidence:
- Real Docker lifecycle: `test_docker_runner_integration.py::test_full_lifecycle` passed with `SANDBOX_ENABLE_EXECUTION=1` and `TLDW_SANDBOX_DOCKER_FAKE_EXEC=0`.
- Strict Research Workspace Playwright: `shows workspace-linked sandbox run in diagnostics when sandbox run API is available` passed against backend `127.0.0.1:18041` and WebUI `127.0.0.1:18042`: `1 passed (31.8s)`.
- CDP probe captured workspace `420d15bb-aaae-4f02-ab0d-35376732fb0a`, real Docker run `d9af7f15-2ed0-44e7-acf9-5295fb633bce`, `phase=completed`, `exit_code=0`, `runtime.state=available`, `admission.state=available`, and screenshot `/private/tmp/task47832-real-docker-sandbox-diagnostics-full.png`.
- UAT matrix RW-UAT-020 and RW-UAT-023 now record the real-Docker evidence while preserving no `/workspace-playground` aliases or redirects.
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
