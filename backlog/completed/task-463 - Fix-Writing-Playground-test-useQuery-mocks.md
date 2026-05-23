---
id: TASK-463
title: Fix Writing Playground test useQuery mocks
status: Done
labels:
- bugfix
- webui
- writing-playground
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address PR feedback that Writing Playground test useQuery mocks ignore enabled and collapse array query keys, causing disabled queries to appear populated and distinct query keys to share results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Writing Playground test useQuery mocks return no data when enabled is false.
- [x] #2 Mock query data lookup keys full queryKey values by stable serialization where a map-backed mock is used.
- [x] #3 Regression coverage or focused test evidence verifies the updated mocks still pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause:
- The WritingPlayground test-local `useQuery` mocks returned seeded data even when `enabled: false`.
- The mocks collapsed array query keys to their first element, so `["writing-capabilities"]` and requested capability variants could share data.

Changes:
- Updated the phase1 baseline mock to serialize the full query key for map lookups and to return empty query state for disabled queries.
- Added regression tests for disabled query behavior and full array query-key separation.
- Updated the inspector-tabs mock to use the same full query-key serialization and disabled-query behavior.

Verification:
- Red evidence: `bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx --maxWorkers=1 --no-file-parallelism` failed on the two new regression tests before the mock fix.
- Green evidence: `bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx src/components/Option/WritingPlayground/__tests__/WritingPlayground.inspector-tabs.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 2 files / 33 tests.
- Broader focused evidence: `bunx vitest run` for the WritingPlayground focused suite passed with 11 files / 104 tests.
- Mechanical checks: `git diff --check` passed over touched paths; ASCII scan over touched paths had no matches.

Bandit:
- Touched files are TS/TSX frontend tests and a Backlog task record; Bandit is not applicable.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the WritingPlayground test `useQuery` mocks so disabled queries return empty state and full array query keys resolve independently. Added regression coverage in the phase1 baseline test and updated the related inspector-tabs mock with the same semantics.

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
