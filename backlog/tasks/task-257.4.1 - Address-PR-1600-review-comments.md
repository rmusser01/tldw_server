---
id: TASK-257.4.1
title: Address PR 1600 review comments
status: In Progress
assignee: []
created_date: '2026-05-12 04:47'
updated_date: '2026-05-12 04:54'
labels:
  - persona-chat
  - evaluations
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1600'
  - 'https://github.com/rmusser01/tldw_server/issues/1598'
parent_task_id: TASK-257.4
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address all actionable review comments on PR #1600 for the Persona Chat judge trace-safe execution artifact slice. Verify each finding against current code, fix only still-valid issues, preserve the offline/no-runtime-gating boundary, and keep changes focused.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Artifact serialization emits structured bounded calibration warning keys instead of free-form warning sentences.
- [x] #2 Artifact serialization defensively sanitizes public input_case_ids, dimension_keys, and prediction result fields at the final to_dict boundary.
- [x] #3 Unsafe case-id redaction collisions do not emit duplicate predictions and do not crash artifact generation.
- [x] #4 Artifact input counts report actual input row count rather than unique sanitized case-id count.
- [x] #5 Represented dimension keys are deterministic and stable across prediction/failure/calibration sources.
- [ ] #6 Focused tests, Bandit on touched Python scope, git diff hygiene, and PR replies/resolution are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regression tests for Qodo/CodeRabbit/Gemini findings. 2. Patch artifact serialization and duplicate handling minimally. 3. Update docs/task notes if the JSON shape changes. 4. Run focused verification, commit, push, and resolve/reply to review threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified PR #1600 review surfaces and addressed still-valid findings: artifact serialization now re-sanitizes public IDs/results, emits structured calibration warning keys, counts actual input rows separately from deduplicated IDs, detects redaction collisions using sanitized prediction keys, and sorts represented dimension keys.

Validation so far: focused execution tests passed, broader Persona Chat judge suite passed with 55 tests, py_compile passed, Bandit on touched Python scope passed with 0 findings, and git diff --check passed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
