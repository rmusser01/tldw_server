---
id: TASK-15559
title: Stub the tests in live_network_baseline.txt by seam and shrink the baseline
  to empty
status: To Do
labels:
- tests
- offline
dependencies:
- TASK-15486
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15486 (PR #3494) landed the offline blocker with tldw_Server_API/tests/live_network_baseline.txt listing the tests that still reach a real host (387 surfaced by the first full CI run, 44046470339). Stub each by its seam (tiktoken encoding download, HuggingFace tokenizer download, provider DNS via Utils.egress.resolve_host_ips / offline_network.stub_host_resolution, example.com downloads, the Slides standalone provider, the embeddings orchestrator, and the rest grouped in the notes of TASK-15486), delete its line, and lower CEILING in tests/lint/test_live_network_baseline_ratchet.py, until the file is empty.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every node id in tldw_Server_API/tests/live_network_baseline.txt is stubbed at its seam (or marked live_network where the live call is the assertion) and its line deleted
- [ ] #2 The baseline file holds no node ids and CEILING is 0
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
