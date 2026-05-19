---
id: TASK-442
title: Add VZ host reboot drill CLI wiring
status: Done
documentation:
- Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md
modified_files:
- tools/macos-vz-helper/scripts/vz-helperctl.py
- tools/macos-vz-helper/Tests/test_vz_helperctl.py
- Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md
- backlog/tasks/task-442 - Add-VZ-host-reboot-drill-CLI-wiring.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 4 from Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md: add `host-reboot-drill {pre,post}` CLI wiring, JSON output, launchd-mode validation, and post-reboot restored-helper smoke targeting via `run_vz_linux_host_smoke(...)`. Scope excludes operator documentation and final task closeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] `host-reboot-drill pre --json` emits parseable `_print_results` JSON.
- [x] `host-reboot-drill post --json` emits parseable `_print_results` JSON.
- [x] `post --run-smoke` appends `vz_linux_smoke` from `run_vz_linux_host_smoke(...)` using the restored helper socket.
- [x] `post --run-smoke` does not call `smoke_helper`.
- [x] Host reboot launchd mode requires explicit `--label` and `--plist-output` and fails clearly when missing.
- [x] CLI exit code is zero only when all named results are ok.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added focused CLI tests first and verified they failed because `host-reboot-drill` was not a parser choice.
- Added `host-reboot-drill {pre,post}` parser wiring with shared Task 4 arguments.
- Routed `pre` to `run_host_reboot_pre(...)` and `post` to `run_host_reboot_post(...)`.
- Routed `post --run-smoke` through `run_vz_linux_host_smoke(...)` with the restored helper socket and appended a `vz_linux_smoke` named result.
- JSON mode redirects incidental stdout from delegated helpers and captured subprocess runners before printing `_print_results` JSON.
- Launchd mode now rejects missing host-reboot metadata with `host_reboot_launchd_metadata_missing` instead of using unrelated defaults.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 4 code-quality review findings fixed only; Task 5 docs/final closeout intentionally not implemented.

Review fixes:
- `host-reboot-drill pre/post --dry-run` now returns named dry-run results without creating evidence directories or writing pre/post manifests.
- Host reboot CLI parser now keeps `--create-evidence-dir` pre-only and `--run-smoke`/`--python` post-only.
- Added subprocess JSON coverage for `host-reboot-drill pre --json --dry-run`; helperctl now suppresses import-time stdout from the backend helper-client import path so JSON stdout remains parseable.
- Post validation now compares `helper_mode`, `launchd_label`, `launchd_plist_path`, `socket_path`, `helper_path`, and `bundle_path` against the pre manifest before post evidence is written or smoke is considered.
- `post --run-smoke` is gated on successful post validation and appends `host_reboot_smoke_skipped` when post validation fails; smoke failure still drives a nonzero CLI exit.

Verification:
- Red run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -k "host_reboot_drill_cli or host_reboot_post_runs_smoke or host_reboot_pre_dry_run or host_reboot_post_dry_run or metadata_mismatch" -q` failed with the expected seven review-finding failures.
- Focused green: same selector passed with `12 passed, 174 deselected`.
- Full helperctl file: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py` passed with `180 passed, 6 skipped`.
- Bandit: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tools/macos-vz-helper/scripts/vz-helperctl.py -f json -o /tmp/bandit_task442_task4.json` completed with zero findings.
- Compile: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m py_compile tools/macos-vz-helper/scripts/vz-helperctl.py` passed.
- Whitespace: `git diff --check` passed.

Known skips: full helperctl test file reported six existing platform/socket dependent skips; no Task 4 blocker.
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
