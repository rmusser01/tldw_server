---
id: TASK-460
title: Implement document-first Writing Playground revision workflow
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-23 02:47'
labels:
  - implementation
  - webui
  - extension
  - writing-playground
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved document-first Writing Playground revision workflow from TASK-443 using the reviewed plan in Docs/superpowers/plans/2026-05-22-writing-playground-document-first-revisions-implementation-plan.md. Planning task: TASK-458.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Task 1-9 from the implementation plan are completed using test-first execution.
- [ ] #2 Writing Playground proposed-edit workflow supports reviewable proposals, safe apply/reject/copy/regenerate states, workflow presets, and status counts.
- [ ] #3 Revision state is schema-versioned in existing session payload persistence without overwriting pending prompt/settings changes.
- [ ] #4 WebUI and extension route parity are preserved through shared UI implementation and focused tests.
- [ ] #5 Verification results, Bandit skip/results, and known blockers are recorded before final handoff.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-22-writing-playground-document-first-revisions-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 completed.

Files touched:
- apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-types.ts
- apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-utils.ts
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts

Commits:
- 24aa3d3e4 feat: add writing revision utilities
- 9c5ac7b2b fix: no-op missing writing revision replacement
- df40a2ac2 fix: harden writing revision apply utilities
- 3d7a691c7 fix: preserve paragraph boundary revision targets

TDD evidence:
- Initial red: focused Vitest failed with missing ../writing-revision-utils before implementation.
- Follow-up reds: missing replacement returned conflict, malformed insert replaced text, leading blank paragraph returned inverted range, and paragraph boundary cursor targeted the following paragraph.
- Final green: bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts passed with 16 tests.

Review evidence:
- Spec compliance review initially found missing replacement should return noop; fixed in 9c5ac7b2b. Re-review approved.
- Code quality review found insert fallthrough and paragraph edge cases; fixed in df40a2ac2 and 3d7a691c7. Re-review approved with no Critical or Important issues.

Bandit:
- Worker reported touched-scope Bandit produced no findings but parse errors because files are TypeScript; final Bandit decision will be recorded at final verification.

Task 2 completed.

Files touched:
- apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-presets.ts
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-presets.test.ts

Commit:
- 75f475ed7 feat: add writing revision presets

TDD evidence:
- Initial red: focused Vitest failed because ../writing-revision-presets did not exist.
- Final green: bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-presets.test.ts passed with 4 tests.

Review evidence:
- Spec compliance review approved with no issues.
- Code quality review approved with no Critical or Important issues. Minor readonly-array hardening was noted as non-blocking.

Bandit:
- Worker reported touched-scope Bandit produced no findings but parse errors because files are TypeScript; final Bandit decision will be recorded at final verification.

Task 3 completed.

Files touched:
- apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-prompt-utils.ts
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-prompt-utils.test.ts

Commit:
- 4492a4960 feat: validate writing revision proposals

TDD evidence:
- Initial red: focused Vitest failed because ../writing-revision-prompt-utils did not exist.
- Final green: bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-prompt-utils.test.ts passed with 10 tests.

Review evidence:
- Spec compliance review approved with no issues.
- Code quality review approved with no Critical or Important issues. Minor notes: exported input types may help later integration, and invalid advisory notes-only schemas could be tightened later.

Bandit:
- Worker reported touched-scope Bandit produced no findings but parse errors because files are TypeScript; final Bandit decision will be recorded at final verification.

Task 4 completed.

Files touched:
- apps/packages/ui/src/components/Option/WritingPlayground/hooks/utils.ts
- apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingSessionManagement.ts
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts

Commits:
- b86252858 feat: persist writing revision state
- c3c2339d fix: preserve pending writing revision payloads
- a780eb11 fix: keep pending writing revision saves dirty
- 16e9c6f6 fix: keep pending writing preset saves dirty

TDD evidence:
- Initial red: focused Vitest failed with missing revision helper failures.
- Follow-up reds: malformed target offsets accepted, missing pending-aware helper, revision-only pending save considered clearable, and preset-id-only pending save considered clearable.
- Final green: bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts passed with 351 files / 1063 tests.

Review evidence:
- Spec compliance review approved with no issues.
- Code quality review found stale pending payload overwrite, malformed target offset acceptance, revision-only dirty clearing, and preset-id-only dirty clearing. Fixes landed in c3c2339d, a780eb11, and 16e9c6f6. Final re-review found no Critical or Important issues.

Bandit:
- Workers reported touched-scope Bandit produced no findings but parse errors because files are TypeScript; final Bandit decision will be recorded at final verification.
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
