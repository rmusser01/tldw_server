---
id: TASK-433
title: Implement VZ stale socket lifecycle drill
status: To Do
labels:
- sandbox
- vz_linux
- lifecycle
- hardening
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next sandbox lifecycle slice from the merged drill-gaps design: a bounded stale-socket operator/check path that proves stale helper socket recovery is safe, diagnosable, and refuses symlinks/non-socket/user-controlled paths. Keep it host-independent first, with docs and focused tests; no workflow trigger expansion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Host-independent tests cover stale Unix socket recovery under a private runtime directory.
- [ ] #2 Tests cover fail-closed behavior for symlinks, non-socket files, directories, and unsafe parent paths.
- [ ] #3 Implementation reuses existing vz-helperctl/helper socket-safety primitives instead of adding a parallel cleanup path.
- [ ] #4 Operator docs or drill notes explain how to run the stale socket check manually and what evidence to capture.
- [ ] #5 Prepared-host evidence tracker references the new stale socket drill/check without expanding PR/push/scheduled destructive triggers.
- [ ] #6 Focused verification, diff hygiene, and Bandit for touched Python scope are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started from merged design PR #1850 / `Docs/superpowers/specs/2026-05-18-vz-linux-lifecycle-drill-gaps-design.md`.

Plan: `Docs/superpowers/plans/2026-05-19-vz-stale-socket-lifecycle-drill.md`.

Baseline verification before implementation:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q` passed: 134 passed, 1 skipped.
- `swift test` initially failed under the Codex filesystem sandbox because SwiftPM/Clang could not write `~/.cache/clang/ModuleCache`.
- `swift test` passed after host permission for SwiftPM cache access: 88 tests passed.

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
