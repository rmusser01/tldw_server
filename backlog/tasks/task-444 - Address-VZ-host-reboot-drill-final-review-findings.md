---
id: TASK-444
title: Address VZ host reboot drill final review findings
status: Done
labels:
- sandbox
- vz-linux
- review-fix
- host-reboot
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix final review findings on the host-reboot-drill branch: add stronger pre/post lifecycle readiness checks for direct and launchd modes, add a reboot marker so post does not pass without an actual reboot when marker evidence is available, update tests/docs/tasks, and rerun verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pre phase records lifecycle readiness, host boot marker, and bundle dry-run validation evidence.
- [x] #2 Post phase checks lifecycle readiness for the selected helper mode and fails if the host boot marker is missing or unchanged.
- [x] #3 Launchd mode readiness checks explicit launchd status using the provided label/plist metadata.
- [x] #4 Regression tests cover lifecycle readiness failure, launchd status checking, and no-reboot marker rejection.
- [x] #5 Docs/task records explain the stronger blocking behavior and scheduled CI remains non-rebooting.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Final review found that pre/post relied too heavily on helper ping and metadata. The fix adds non-mutating lifecycle readiness collection for direct and launchd modes, records bounded readiness results into evidence manifests, runs bundle validation through the existing host smoke dry-run path, records a host boot marker, and makes post validation fail with `host_reboot_not_detected` when the marker is unchanged.

The host reboot drill still remains explicit operator workflow only. No workflow files or scheduled reboot automation were added.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q` passed: 185 passed, 6 skipped, 2 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tools/macos-vz-helper/scripts/vz-helperctl.py` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tools/macos-vz-helper/scripts/vz-helperctl.py -f json -o /tmp/bandit_vz_host_reboot_drill.json` passed with `results=[]` and `errors=[]`.
- `git diff --check` passed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed final review findings for the VZ host reboot drill. The drill now proves a stronger lifecycle boundary: pre records readiness/bundle/boot-marker evidence, post verifies selected-mode readiness and boot-marker drift before optional restored-socket smoke, and docs state unchanged/missing boot marker as a blocking failure.

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
