---
id: TASK-460
title: Implement document-first Writing Playground revision workflow
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-23 02:47
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
- [x] #1 Task 1-9 from the implementation plan are completed using test-first execution.
- [x] #2 Writing Playground proposed-edit workflow supports reviewable proposals, safe apply/reject/copy/regenerate states, workflow presets, and status counts.
- [x] #3 Revision state is schema-versioned in existing session payload persistence without overwriting pending prompt/settings changes.
- [x] #4 WebUI and extension route parity are preserved through shared UI implementation and focused tests.
- [x] #5 Verification results, Bandit skip/results, and known blockers are recorded before final handoff.
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

Task 5 completed.

Files touched:
- apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingRevisions.ts
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx

Commits:
- 2b977be87 feat: manage writing revision queue state
- 40e17037 fix: guard stale writing revision regeneration

TDD evidence:
- Initial red: focused Vitest failed because hooks/useWritingRevisions.ts did not exist.
- Follow-up reds: stale regeneration appended after active session switch, and regeneration appended after source proposal was rejected while async replacement was pending.
- Final green: bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx passed with 1 test file / 10 tests.

Review evidence:
- Spec compliance review approved the initial Task 5 hook implementation.
- Code quality review found async regeneration could append/persist stale replacements after session/source changes. Fixed in 40e17037. Re-review approved with no Critical or Important issues.

Bandit:
- Workers reported touched-scope Bandit produced no findings but parse errors because files are TypeScript; final Bandit decision will be recorded at final verification.

Task 6 completed.

Files touched:
- apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingRevisionDiff.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingRevisionQueue.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx

Commits:
- 97c6561a feat: add writing revision queue UI
- 93bb0f6a fix: add icons to writing revision actions
- c9af6d91 fix: reset writing revision confirmation on target change

TDD evidence:
- Initial red: focused Vitest failed because WritingActionBar and WritingRevisionQueue did not exist.
- Follow-up reds/review regressions: spec review found label-only action buttons; code quality review found broad-target confirmation survived target changes.
- Final green: bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx passed with 2 test files / 11 tests.

Review evidence:
- Spec compliance review initially found missing icon+label buttons. Fixed in 93bb0f6a; re-review approved.
- Code quality review found stale confirmation state when target changes. Fixed in c9af6d91; re-review approved with no Critical or Important issues.

Bandit:
- Touched files are TypeScript/TSX frontend code; final Bandit decision will be recorded at final verification.

Task 7 completed.

Files touched:
- apps/packages/ui/src/components/Option/WritingPlayground/index.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingRevisionQueue.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx

Commits:
- 84523191 feat: wire writing revision proposals
- 665967dd fix: honor saved writing revision preset
- ecd44d6 test: assert writing revision regeneration metadata
- a99a715e fix: harden writing revision generation state

TDD evidence:
- Initial red: integration tests failed because the Writing Playground did not render the revision action bar/queue or create proposed-edit messages.
- Follow-up reds: saved preset not honored, regenerated proposal metadata unverified, stale action-bar target after selection changes, topbar exposed Stop for non-streaming revision request, rich-editor selection did not refresh action-bar target, and pending revision requests allowed session switching.
- Final green: cd apps/packages/ui && bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx --maxWorkers=1 --no-file-parallelism passed with 1 test file / 18 tests.
- Final green: cd apps/packages/ui && bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx src/components/Option/WritingPlayground/__tests__/WritingTipTapEditor.external-sync.test.tsx --maxWorkers=1 --no-file-parallelism passed with 4 test files / 23 tests.

Review evidence:
- Spec compliance review initially found missing coverage for regenerated proposal metadata and rich editor apply non-mutation. Fixed in ecd44d6; re-review approved.
- Code quality review found stale selection gating, fake Stop semantics for revision requests, missing rich-editor selection updates, and session-switch race while revision generation was pending. Fixed in a99a715e. Final re-review reported no remaining Critical or Important issues.

Bandit:
- Touched files are TypeScript/TSX frontend code; final Bandit decision will be recorded at final verification.

Task 8 completed.

Files touched:
- apps/packages/ui/src/components/Option/WritingPlayground/index.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx
- apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts
- apps/extension/tests/e2e/writing-playground-mode-parity.spec.ts

Commit:
- 2a20b181c test: cover writing revision parity

TDD evidence:
- Initial red: route parity guard failed on missing writing-revision-pending-count in the shared WritingPlayground source.
- Initial red: WritingPlayground.phase1-baseline.test.tsx failed on missing writing-status-word-count and writing-revision-pending-count.
- Final green: cd apps/packages/ui && bunx vitest run src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts src/components/Option/WritingPlayground/__tests__/writing-revision-presets.test.ts src/components/Option/WritingPlayground/__tests__/writing-revision-prompt-utils.test.ts src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx --maxWorkers=1 --no-file-parallelism passed with 8 test files / 84 tests.
- Final green: cd apps/tldw-frontend && bunx vitest run extension/__tests__/writing-playground-route-parity.guard.test.ts --maxWorkers=1 --no-file-parallelism passed with 1 test.
- Post-review green after i18n fallback fix: WritingPlayground.phase1-baseline.test.tsx passed with 19 tests, route parity guard passed with 1 test, and git diff --check passed.

Extension smoke:
- Attempted cd apps/extension && bunx playwright test tests/e2e/writing-playground-mode-parity.spec.ts.
- The command entered WXT production build, emitted existing duplicate-import warnings for MediaNavigationFormat and estimateStorageCost, then stalled in wxt build. The stuck process was killed; command reported build:chrome:prod exited with code 1. Route parity and component coverage are the current extension evidence for Task 8.

Review evidence:
- Spec compliance review approved Task 8 with no findings.
- Code quality review found no Critical or Important issues and one Minor i18n fallback consistency issue. Fixed by routing new status labels through t(...) with formatted fallbacks. Re-review approved with no Critical or Important issues.

Bandit:
- Touched files are TypeScript/TSX frontend code; final Bandit decision will be recorded at final verification.

Task 9 final verification and review completed.

Commit:
- 6fe6e83f0 fix: harden writing revision final review issues

Files touched:
- apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingRevisions.ts
- apps/packages/ui/src/components/Option/WritingPlayground/index.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/writing-editor-adapter.ts
- apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-utils.ts
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingTipTapEditor.external-sync.test.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-editor-adapter.test.ts
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts

Final review fixes:
- Continue now resolves to a cursor insert before selection targeting, so selected text does not create a non-zero insert target.
- Continue no longer uses broad replace-target confirmation for insert operations.
- Revision payload refreshes compare revision signatures, so same-payload autosave echoes do not discard in-flight regenerated proposals.
- Revision proposed-edit requests preserve actual advanced extra_body generation controls while prompt/debug summaries remain sanitized.
- TipTap selection mapping now uses the same plain-text serialization as rich editor persistence, including paragraph boundaries and sceneBreak serialization.

TDD/regression evidence:
- Added red regressions for Continue-with-selection, Continue broad confirmation, same-payload regeneration echo, TipTap paragraph-boundary selection, TipTap sceneBreak selection, and advanced extra_body parity.
- Final green: cd apps/packages/ui && bunx vitest run src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts src/components/Option/WritingPlayground/__tests__/writing-revision-presets.test.ts src/components/Option/WritingPlayground/__tests__/writing-revision-prompt-utils.test.ts src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx src/components/Option/WritingPlayground/__tests__/writing-editor-adapter.test.ts src/components/Option/WritingPlayground/__tests__/WritingTipTapEditor.external-sync.test.tsx src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx --maxWorkers=1 --no-file-parallelism passed with 10 files / 92 tests.
- Final green: cd apps/tldw-frontend/extension && bunx vitest run __tests__/writing-playground-route-parity.guard.test.ts --maxWorkers=1 --no-file-parallelism passed with 1 file / 1 test.
- Mechanical checks: git diff --check over touched paths passed; ASCII scan over touched paths had no matches.

Review evidence:
- Final targeted review initially found four Important issues: Continue selected-text targeting/confirmation, regeneration dropped on same-payload save echo, TipTap plain-text offset mismatch after rich paragraph boundaries, and advanced extra_body token-key sanitization dropping valid controls.
- Fix re-review found one remaining Important sceneBreak/custom atom mapping gap.
- Final re-review found no Critical or Important findings for the remaining TipTap mapping issue.

Bandit:
- Touched code for the final fixes is TypeScript/TSX frontend code; Bandit is a Python analyzer and is not applicable to these paths. Earlier TypeScript touched-scope Bandit attempts produced no findings but parse errors, so final security validation for this slice is recorded as not applicable.

Known blocker:
- Extension Playwright smoke remains blocked by the existing WXT production build stall documented during Task 8. Focused shared UI coverage and route parity guard are the verified extension evidence for this task.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
