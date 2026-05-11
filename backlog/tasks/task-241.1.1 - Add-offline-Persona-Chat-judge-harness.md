---
id: TASK-241.1.1
title: Add offline Persona Chat judge harness
status: Done
assignee: []
created_date: '2026-05-11 05:48'
updated_date: '2026-05-11 15:49'
labels:
  - persona
  - chat
  - evaluations
  - stage-2
  - judge
  - harness
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1572'
  - 'https://github.com/rmusser01/tldw_server/issues/1566'
documentation:
  - Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md
  - tldw_Server_API/tests/fixtures/persona_chat_judge_contract_cases.json
parent_task_id: TASK-241.1
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build the next #1566 slice from #1572: a deterministic offline Persona Chat judge harness that replays contract fixture candidate outputs and produces a bounded calibration report without provider calls, runtime Persona Chat changes, or production gating.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Harness loads/accepts V1 contract cases and candidate judge outputs without model/provider calls.
- [x] #2 Report includes case counts, verdict agreement, expected/actual flag agreement, score schema validation, and bounded mismatch details.
- [x] #3 Tests cover matching outputs, verdict mismatch, flag mismatch, invalid labels, and invalid score schema.
- [x] #4 Implementation has module/function docstrings and type hints.
- [x] #5 No runtime Persona Chat path, Jobs worker, API endpoint, or WebUI behavior changes.
- [x] #6 Focused pytest, Bandit on touched Python, and diff hygiene are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write a plan document for the offline Persona Chat judge harness.
2. Add failing tests for report generation and mismatch validation.
3. Implement the minimal core harness helper under app/core/Evaluations.
4. Update contract docs with the offline harness boundary.
5. Run focused pytest, Bandit, placeholder scan, and diff hygiene.
6. Update Backlog, commit, push, and open a PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created after PR #1569 merged. #1572 tracks the executable offline harness slice; #1566 remains the parent optional judge evaluation tracker.

Implemented offline Persona Chat judge harness under app/core/Evaluations with pure dataclass report generation, strict candidate envelope validation, and bounded mismatch reporting. Verification on 2026-05-11: focused harness pytest passed 5 tests; combined judge harness/contract pytest passed 15 tests; Bandit reported zero findings for touched Python; placeholder scan found no matches; git diff --check passed.

No known skips or blockers for this slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a deterministic offline Persona Chat judge harness for issue #1572. The helper compares already-produced candidate judge outputs with the V1 contract fixture, validates verdict/flag/score envelopes, returns bounded agreement and mismatch reports, and avoids provider calls, persistence, Jobs, endpoints, WebUI changes, or runtime Persona Chat gating. Documentation now records that boundary and tests cover agreement, mismatch, invalid labels, and invalid score schema.
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
