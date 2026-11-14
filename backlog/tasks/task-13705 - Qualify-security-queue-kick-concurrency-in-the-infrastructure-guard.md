---
id: TASK-13705
title: Qualify security queue-kick concurrency in the infrastructure guard
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final PR #3229 broader matrix failed one literal-boolean infrastructure guard after latest dev adopted a guarded queue-kick exception. Preserve cancellation of superseded PR runs and noncancelling queue kicks; align the stale guard with the exact accepted ADR-063 workflow contract, without changing production workflow behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce the exact infrastructure guard failure before repair
- [x] #2 Only the security workflow may use the exact queue-kick cancellation exception; other required workflows still require true
- [x] #3 Connected infrastructure/merge-queue contracts, lint, Bandit and independent review qualify the guard without changing workflow source
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Failing head f2b9bc84a7567635527f3591da325ea531756867; run38707188911 job116335357295:1failed483passed, test_workflow_concurrency_policy.py::test_required_workflow_cancels_superseded_runs[security-required.yml]. Root: latest dev explicitly excludes queue_kick=true from cancellation; ordinary PR inputs remain absent and cancel. Governing ADR063; no new ADR needed. Task-ID collision exception for separate13639–13642 records remains pending user approval.
Red1failed20passed matches actual CI failure. Narrow guard permits only literalTrue or the exact accepted security queue-kick exception; production workflows and all identity/group guards unchanged. Final363connected contracts pass; actual484case platform shard passes34.63s on projectPython3.12;5unsafe mutations rejected/2safe forms accepted; Ruff/Blackpass; Bandit5parent-identicalB101/0scanerrors. Independent source reviewerBclear; final exact committed reviews and new-head GitHubCI remain PR landing conditions. Separate13639–13642 renumbering exception is still pending user approval; affected records untouched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aligned the stale infrastructure cancellation guard with accepted ADR063 queue-kick serialization while preserving superseded PR cancellation. No production workflow or frontend source changes. Reproduced red, then qualified363connected contracts,484platform cases, mutation controls, lint/format and parent-identical security reports. Exact-head CI remains the PR merge gate.
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
