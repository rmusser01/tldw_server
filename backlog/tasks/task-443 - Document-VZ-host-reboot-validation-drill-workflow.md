---
id: TASK-443
title: Document VZ host reboot validation drill workflow
status: Done
documentation:
- Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md
modified_files:
- tools/macos-vz-helper/README.md
- Docs/Sandbox/macos-runtime-operator-notes.md
- Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
- backlog/tasks/task-438 - Design-VZ-helper-host-reboot-validation-procedure.md
- Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md
- backlog/tasks/task-443 - Document-VZ-host-reboot-validation-drill-workflow.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 5 from Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md: document the host reboot drill operator workflow, CI policy boundaries, durable evidence requirements, restored-helper smoke targeting, diagnostics/dry-run repair expectations, and close out parent tracking with verification results. Scope is docs/backlog plus final verification only; no new helper behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 README documents the durable evidence directory requirement and direct/launchd pre -> reboot -> post command sequence.
- [x] #2 Operator notes explain restored-helper smoke targeting and keep diagnostics/dry-run repair separate and operator-reviewed.
- [x] #3 Host-gated CI policy states scheduled/nightly CI must not reboot hosts and records skip/blocking behavior.
- [x] #4 Task plan and Backlog records include verification results and the prepared-host reboot skip reason.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Documented the host reboot validation drill as an operator workflow in the helper README, macOS runtime operator notes, and host-gated CI acceptance policy.

Key requirements captured:
- evidence directories must be durable across reboot, private, and outside `/tmp`, `$TMPDIR`, or other volatile roots
- direct helper mode uses the managed helper socket defaults
- launchd helper mode requires explicit `--label` and `--plist-output` in both `pre` and `post`
- the operator sequence is pre -> manual reboot -> post
- `post --run-smoke` targets the restored helper socket through the host smoke path and must not start a new helper process
- diagnostics and dry-run reconciliation repair are separate operator-reviewed steps
- scheduled/nightly CI must not reboot hosts
- scheduled CI skips reboot validation, while a manual prepared-host drill is blocking only when explicitly invoked
- explicit drill failures include missing/unsafe/volatile evidence directories, pre/post metadata mismatch, helper ping/protocol failure, and post-smoke failure when requested

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q` passed: 182 passed, 6 skipped, 2 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tools/macos-vz-helper/scripts/vz-helperctl.py` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tools/macos-vz-helper/scripts/vz-helperctl.py -f json -o /tmp/bandit_vz_host_reboot_drill.json` passed; JSON contains `results=[]` and `errors=[]`.
- `git diff --check` passed.

Prepared-host validation was not run because Task 5 must not perform a real host reboot. It remains manual or explicitly operator-triggered only.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Task 5 docs/task closeout for the VZ helper host reboot validation drill. The documented workflow now covers durable evidence, direct and launchd command sequences, restored-helper smoke targeting, diagnostics/repair separation, CI reboot prohibition, expected skips, and explicit blocking failure modes.

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
