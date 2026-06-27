---
id: TASK-2373
title: 'Task 4: Add CATS command builder and summary JSON'
status: Done
references:
- TASK-2371
modified_files:
- Helper_Scripts/cats_fuzz/cats_cli.py
- Helper_Scripts/cats_fuzz/summary.py
- tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py
- tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 4 for the CATS API fuzzing harness: add CATS command construction helpers, exit classification, safe subprocess runner, run summary dataclass, secret-masking summary writer, focused unit tests, verification, and commit with the requested message.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- TDD red run: focused pytest failed during collection because `Helper_Scripts.cats_fuzz.cats_cli` and `Helper_Scripts.cats_fuzz.summary` did not exist.
- Verification: `python -m pytest tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py -q` passed with 8 tests.
- Formatting: `python -m black --check Helper_Scripts/cats_fuzz/cats_cli.py Helper_Scripts/cats_fuzz/summary.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py` passed.
- Security: `python -m bandit -r Helper_Scripts/cats_fuzz/cats_cli.py Helper_Scripts/cats_fuzz/summary.py -f json -o /tmp/bandit_cats_fuzz_cli_summary.json` passed with no findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Added CATS run/validate/stats command builders, exit classification, and a captured-output subprocess runner.
- Added CATS run summary JSON writing with command masking at the persistence boundary so raw `X-API-KEY` and `Authorization` header values are not written.
- Added focused unit coverage for command construction, exit classification, summary shape, and API-key masking regression behavior.
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
