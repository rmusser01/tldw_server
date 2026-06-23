---
id: TASK-2423
title: Harden Research Workspace capability readiness
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the current Research_Workspace capability module review findings: fail closed for real Slides DB unavailability, decouple core health collection from API endpoint functions, bound/concurrently run health probes, stop placeholder sync/share from forcing degraded overall status, tighten capability response typing, and correct README function naming.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-06-23-research-workspace-capability-readiness-hardening.md

Implemented review fixes:
- Slides DB lookup returning None now reports unavailable, so slides_generation blocks instead of warning/failing open.
- Core capability collection now uses injectable ResearchWorkspaceHealthCollectors and concurrent bounded probes instead of importing API endpoint health functions.
- The sync_share placeholder remains in the response but no longer degrades the top-level status by itself.
- Capability response mapping keys are typed as ResearchWorkspaceCapabilityId.
- README function naming now matches collect_research_workspace_capabilities().

Verification:
- source .venv/bin/activate && python -m pytest --confcutdir=tldw_Server_API/tests/Research_Workspace tldw_Server_API/tests/Research_Workspace -q => 20 passed, 2 warnings in 95.67s.
- source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Research_Workspace tldw_Server_API/app/api/v1/schemas/research_workspace_capabilities.py -f json -o /tmp/bandit_research_workspace_capability_readiness.json => 0 findings.
- git diff --check => passed.

Note: the official Backlog CLI hung in this environment and the fallback CLI did not support adding acceptance criteria after task creation, so the task acceptance criteria section remains empty.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Research Workspace capability readiness by failing closed for unavailable Slides DB access, replacing endpoint health imports with injectable concurrent timeout-bounded core probes, preventing the sync/share placeholder from degrading otherwise-ready capabilities, tightening schema key typing, and correcting README naming. Focused Research Workspace tests, Bandit, and diff checks passed.
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
