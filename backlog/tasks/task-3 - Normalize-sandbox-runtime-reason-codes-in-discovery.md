---
id: TASK-3
title: Normalize sandbox runtime reason codes in discovery
status: Done
assignee: []
created_date: '2026-05-03 17:58'
updated_date: '2026-05-03 18:03'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an additive normalized reason-code layer to sandbox runtime discovery while preserving raw preflight reasons.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Discovery includes normalized_reasons for runtimes with raw reasons.
- [x] #2 Raw reasons remain unchanged for operator diagnostics and compatibility.
- [x] #3 Reason-code vocabulary is centralized in runtime_capabilities.py and exposed through the API schema.
- [x] #4 Sandbox runtime inventory documents the additive normalized reason-code contract.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented additive normalized_reasons on runtime discovery while preserving raw reasons. Verification: runtime inventory contract tests passed; runtime capabilities policy tests passed; py_compile passed; Bandit touched Python scan had 0 findings; git diff --check passed. Known skip/caveat: selected TestClient-based feature_discovery_flags group timed out in existing app shutdown/background Jobs teardown after first selected test passed; stack was TestClient/Jobs teardown, not normalized reason code logic.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added centralized sandbox runtime reason-code normalization and exposed normalized_reasons in discovery/schema/docs.
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
