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

Prior Task 4 review fixes:
- `host-reboot-drill pre/post --dry-run` returns named dry-run results without creating evidence directories or writing pre/post manifests.
- Host reboot CLI parser keeps `--create-evidence-dir` pre-only and `--run-smoke`/`--python` post-only.
- Subprocess JSON coverage for `host-reboot-drill pre --json --dry-run` keeps JSON stdout parseable despite helper-client import-time stdout.
- Post validation compares `helper_mode`, `launchd_label`, `launchd_plist_path`, `socket_path`, `helper_path`, and `bundle_path` against the pre manifest before post evidence is written or smoke is considered.
- `post --run-smoke` is gated on successful post validation and appends `host_reboot_smoke_skipped` when post validation fails; smoke failure still drives a nonzero CLI exit.

Additional review fixes in this commit:
- Host reboot metadata path fields (`bundle_path`, `helper_path`, `socket_path`, `launchd_plist_path`) are now canonicalized with `expanduser().resolve(strict=False)` before pre/post manifests are written or compared; omitted launchd plist remains the explicit empty string.
- Post validation now rejects any loaded `host-reboot-pre.json` whose `phase` is not `pre` with `host_reboot_pre_manifest_invalid`.
- Added regressions for equivalent relative/absolute metadata paths and for rejecting `phase: post` pre manifests.

Verification:
- Prior red run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -k "host_reboot_drill_cli or host_reboot_post_runs_smoke or host_reboot_pre_dry_run or host_reboot_post_dry_run or metadata_mismatch" -q` failed with the expected seven review-finding failures.
- Prior focused green: same selector passed with `12 passed, 174 deselected`.
- Current red run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -k "post_phase or equivalent_relative_paths" -q` failed with the expected two review-finding failures on current behavior.
- Current focused green: same selector passed with `2 passed, 186 deselected`.
- Full helperctl file: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py` passed with `182 passed, 6 skipped`.
- Bandit: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tools/macos-vz-helper/scripts/vz-helperctl.py -f json -o /tmp/bandit_task442_task4_final.json` completed; JSON check reported `results=0, errors=0`.
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
