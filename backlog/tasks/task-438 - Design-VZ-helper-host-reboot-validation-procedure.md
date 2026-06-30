---
id: TASK-438
title: Design VZ helper host reboot validation procedure
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-19 04:59'
labels:
  - sandbox
  - vz-linux
  - operator-workflow
  - host-reboot
dependencies: []
references:
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
  - tools/macos-vz-helper/scripts/vz-helperctl.py
documentation:
  - Docs/superpowers/specs/2026-05-19-vz-helper-host-reboot-validation-design.md
  - Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the next VZ Linux operator recovery slice after the merged launchd drill: a host-reboot validation procedure that proves reboot recovery through explicit operator steps, diagnostics, dry-run repair, helper readiness, and real smoke without adding automated reboot behavior to normal CI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A focused design spec defines host-reboot validation scope, sequence, evidence, and non-goals.
- [x] #2 The design keeps host reboot manual/operator-owned and does not add automatic reboot to scheduled CI or application startup.
- [x] #3 The design reuses existing helperctl launchd/start/status/smoke, diagnostics, and dry-run repair surfaces instead of adding a parallel recovery path.
- [x] #4 The design defines safe evidence capture before and after reboot, including logs, helper generation, stale session-control expectations, and cleanup boundaries.
- [x] #5 An implementation plan is written after risk review with exact docs/tests/commands and host-gated validation expectations.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created the host reboot validation design spec at Docs/superpowers/specs/2026-05-19-vz-helper-host-reboot-validation-design.md. Risk review patched the design before planning: evidence must use a durable non-/tmp private directory, post-reboot smoke must target the restored helper socket instead of starting a second helper, launchd mode must use explicit ownership, and stale pre-reboot pid files must not be trusted after reboot.

Created the execution-ready implementation plan at Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md. The plan scopes the future implementation to host-reboot-drill pre/post evidence, PingState helper details, portable helperctl tests, restored-helper smoke targeting, docs, Bandit, and optional prepared-host validation.

Verification: git diff --check passed for the worktree after adding the task/spec/plan. Bandit skipped because this task only adds documentation and Backlog metadata; the implementation plan requires Bandit for future Python changes.

Task 5 documentation closeout recorded the operator host reboot validation drill in the helper README, macOS operator notes, and host-gated CI policy. The docs now require durable private evidence outside `/tmp`/volatile roots, document the exact pre -> manual reboot -> post sequence for direct and launchd helper modes, require explicit launchd `--label` and `--plist-output` in both phases, clarify that post-reboot smoke targets the restored helper socket through the host smoke path, keep diagnostics and dry-run repair operator-reviewed and separate, and prohibit scheduled/nightly CI from rebooting hosts.

Task 5 touched files:
- tools/macos-vz-helper/README.md
- Docs/Sandbox/macos-runtime-operator-notes.md
- Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
- Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md
- backlog/tasks/task-438 - Design-VZ-helper-host-reboot-validation-procedure.md
- backlog/tasks/task-443 - Document-VZ-host-reboot-validation-drill-workflow.md

Task 5 verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q` passed: 182 passed, 6 skipped, 2 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tools/macos-vz-helper/scripts/vz-helperctl.py` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tools/macos-vz-helper/scripts/vz-helperctl.py -f json -o /tmp/bandit_vz_host_reboot_drill.json` passed; JSON contains `results=[]` and `errors=[]`.
- `git diff --check` passed.

Prepared-host reboot validation was not run in Task 5 because it is disruptive and requires an operator reboot. Scheduled/nightly CI is expected to skip host reboot validation. A manual prepared-host drill is blocking only when explicitly invoked; blocking failures include missing, unsafe, or volatile evidence directories; pre/post metadata mismatch; helper ping/protocol failure; and post-smoke failure when `--run-smoke` is requested.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Designed the next VZ Linux host-reboot validation slice and wrote an implementation plan. The design keeps reboot manual/operator-owned, uses durable bounded evidence, reuses existing helperctl/status/launchd/smoke/diagnostics/dry-run repair surfaces, and avoids hidden reboot, repair, or launchd mutation. The plan is ready for a future implementation PR.

Task 5 completed the operator documentation closeout for the implemented drill workflow and recorded verification/skip policy in TASK-443.
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
