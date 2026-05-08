---
id: TASK-121
title: Refresh sandbox inventory after host-local warnings UI
status: Done
assignee: []
created_date: '2026-05-08 02:52'
updated_date: '2026-05-08 03:05'
labels:
  - sandbox
  - docs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1336'
documentation:
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
  - Docs/superpowers/plans/2026-05-06-sandbox-host-local-warnings-ui.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile the sandbox runtime capability inventory after PR #1336 delivered the admin Monitoring UI surface for host-local runtime warnings. Keep this narrow: update stale current-gap wording so the inventory no longer claims seatbelt/worktree warnings still need future UI/operator dashboard propagation, and add a focused guard test so the stale wording does not return.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capability inventory no longer lists host-local warning UI/operator dashboard propagation as an open gap after PR #1336.
- [x] #2 Inventory still preserves explicit weaker-isolation guidance for seatbelt and worktree.
- [x] #3 Focused docs/capability guard test fails before the wording update and passes after it.
- [x] #4 Verification records the focused test, diff check, and Bandit result/rationale for docs/test-only scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
PR #1373 review fix pass: remove local absolute paths from the task record, scope the stale-gap guard to the Current Gaps section, add the missing test docstring, make the host-local warning assertion match inventory text case-insensitively, rerun focused sandbox checks plus hygiene, then reply to and resolve review threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py::test_inventory_no_longer_lists_host_local_warning_ui_as_missing -q failed because the inventory still contained `future UI/operator dashboards`.

GREEN: Updated `Docs/Sandbox/sandbox-runtime-capability-inventory.md` to say host-local isolation warnings now flow through public discovery, cross-runtime admin diagnostics, and the admin Monitoring page `Sandbox Runtime Isolation` card. Removed the stale current-gap row while keeping seatbelt/worktree weaker-isolation and not-untrusted-eligible guidance.

Verification: focused RED/GREEN guard passed after the doc update; full python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q passed with 5 tests; py_compile passed for the touched test; Ruff passed for the touched test; Bandit on the touched test with pytest assert-use noise excluded (-s B101) reported zero findings; git diff --check passed.

Reopened for PR #1373 review comments. Gemini and Qodo findings were valid against the reviewed code/task text: task notes contained local absolute paths; the stale phrase assertion scanned the whole document; the new guard test had no docstring; and the host-local phrase check was case-sensitive against capitalized inventory text.

PR #1373 review fix validation: focused host-local inventory guard passed; full python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q passed with 5 tests; py_compile passed; Ruff passed; Bandit on the touched test with -s B101 reported zero findings; git diff --check passed. Review fixes removed local absolute paths from task verification text, scoped the stale-gap assertion to Current Gaps, added the test docstring, and made the host-local warning assertion case-insensitive within Recovery And Diagnostics Support.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refreshed the sandbox runtime capability inventory after PR #1336. The inventory now records that host-local isolation warnings are surfaced through discovery, admin runtime diagnostics, and the admin Monitoring `Sandbox Runtime Isolation` card, and no longer lists UI/operator dashboard propagation as an open gap. Added and review-hardened a focused guard test so stale gap wording does not return and the host-local warning coverage is checked in the intended section.

PR #1373 review comments addressed: task verification commands are now portable, and the sandbox inventory guard test now documents its intent while checking the relevant markdown sections instead of relying on whole-document substring matches.
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
