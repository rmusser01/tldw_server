---
id: TASK-9
title: Phase 2.1 worker lifecycle cleanup A
status: Done
assignee: []
created_date: '2026-05-03 18:51'
updated_date: '2026-05-03 21:35'
labels:
  - phase-2
  - issue-1116
  - lifecycle
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
documentation:
  - Docs/superpowers/specs/2026-05-03-phase2-followup-stack-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
First conservative Phase 2.1 follow-up tranche for #1116. Prove the selected lifecycle worker has a single registry shutdown owner, then remove one duplicate legacy direct-stop path only if covered by focused tests. Keep startup order, enablement flags, app-state inventory semantics, and shutdown behavior stable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A focused lifecycle ownership test proves the targeted worker is registry-owned in the expected shutdown phase.
- [x] #2 The targeted worker no longer has an unguarded duplicate legacy direct-stop path.
- [x] #3 Startup/shutdown behavior and app-state inventory semantics are preserved.
- [x] #4 Focused lifecycle/startup/shutdown tests, Bandit touched-source scope, and git diff --check pass.
- [x] #5 PR #1241 review comments are resolved: stopped-handle filtering has a single helper path and the new async regression test has an intent docstring.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused coverage proving chatbooks_cleanup is registry-owned in BACKGROUND_WORKER_SHUTDOWN and appears exactly once in app-state worker inventory.
2. Add focused red coverage proving shutdown_pre_worker_cleanup drops legacy chatbooks_cleanup task/stop-event handles after WorkerRegistry has already stopped chatbooks_cleanup.
3. Patch only the shutdown pre-worker handoff to suppress those legacy chatbooks_cleanup handles when stopped_background_worker_names contains chatbooks_cleanup, preserving existing behavior for unstopped handles and other workers.
4. Rerun focused startup/shutdown tests, related lifecycle shutdown tests, Bandit on touched source, and git diff --check before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification completed:
- RED: test_shutdown_pre_worker_cleanup_drops_registry_stopped_chatbooks_handles failed before implementation because chatbooks_cleanup handles still passed through.
- GREEN: startup chatbooks focused 4 passed; shutdown pre-worker chatbooks focused 4 passed.
- Full/adjacent: startup_cleanup_workers 15 passed; shutdown_pre_worker_cleanup 12 passed; lifecycle_workers 14 passed; lifespan_shutdown_sequence 1 passed; shutdown_coordinated_legacy_components 6 passed; shutdown_transition_handoff 4 passed; main_lifecycle_contract 55 passed.
- Bandit touched source: 0 findings in /tmp/bandit_phase2_1_lifecycle_cleanup_a.json.
- git diff --check passed.

PR opened: https://github.com/rmusser01/tldw_server/pull/1241

Review follow-up started for PR #1241: Qodo requested an intent docstring on the new async regression test and centralization of duplicated chatbooks stopped-handle suppression between normal and fallback paths.

Review follow-up verification completed: RED helper test failed with AttributeError before implementation; GREEN helper test passed after adding _filtered_pre_worker_handles; focused Services tests passed with 28 passed; Bandit touched source reported 0 findings in /tmp/bandit_pr1241_review_fixes.json; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added focused worker ownership coverage for chatbooks_cleanup and a red/green shutdown pre-worker regression proving registry-stopped chatbooks handles are removed from the legacy direct-stop handoff. The production change is limited to reusing the existing background-stopped suppression helper for chatbooks task and stop-event handles before invoking and returning pre-worker cleanup state.

Review follow-up centralized pre-worker stopped-handle filtering in _filtered_pre_worker_handles and added the requested intent docstring to the chatbooks regression test.
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
