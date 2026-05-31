---
id: TASK-478.32
title: Validate Research Workspace sandbox handoff with real Docker runtime
status: To Do
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
- [ ] #1 A live backend is started with sandbox routes and real Docker execution enabled, without `TLDW_SANDBOX_DOCKER_FAKE_EXEC=1`.
- [ ] #2 A canonical Research Workspace ID can create or observe a workspace-linked sandbox run through sandbox-owned APIs/surfaces.
- [ ] #3 Research Workspace sandbox diagnostics show the active workspace ID, runtime/admission state, and the real run without duplicating sandbox execution state inside Research Workspace.
- [ ] #4 Failure states distinguish Docker unavailable, sandbox admission blocked, route disabled, and run failed/triaged states with actionable evidence.
- [ ] #5 `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md` is updated only as far as live evidence supports, preserving the no `/workspace-playground` alias/redirect rule.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Start from the strict TASK-478.29 live Playwright/backend fixture and remove the fake-execution path.
- Use a live backend, WebUI, and CDP/Playwright walkthrough; capture exact API responses, run IDs, diagnostics state, and screenshot paths.
- If the current host still lacks a reachable Docker daemon, record the blocker and leave the matrix wording at fixture-backed Pass/watch risk rather than overclaiming real-runtime validation.
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
