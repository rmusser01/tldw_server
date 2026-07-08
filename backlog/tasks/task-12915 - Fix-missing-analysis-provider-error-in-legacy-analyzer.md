---
id: TASK-12915
title: Fix missing analysis provider error in legacy analyzer
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-08 02:35'
labels:
  - bug
  - media
  - analysis
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent analysis/summarization requests with a missing api_name from surfacing `Error calling API None: 'NoneType' object has no attribute 'lower'`. Add focused regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shared analyzer guard normalizes api_name and rejects None, blank, and 'none' before dispatch. Regression covers all missing-provider aliases.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the missing analysis provider path so it returns `Error: Analysis API provider is required.` instead of leaking `Error calling API None: 'NoneType' object has no attribute 'lower'`. Verification: regression failed before the fix, then `python -m pytest tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py tldw_Server_API/tests/LLM_Calls/test_local_summarization_config.py -q` passed with 8 tests. Bandit on the touched backend file reported 0 findings.
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
