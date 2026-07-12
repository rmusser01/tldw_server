---
id: TASK-12949
title: 'Rebase PR #2633 and address review feedback'
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-12 18:37'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2633'
  - 'https://github.com/rmusser01/tldw_server/issues/2605'
  - Docs/superpowers/specs/2026-07-12-pr-2633-review-rebase-design.md
  - >-
    Docs/superpowers/plans/2026-07-12-pr-2633-review-rebase-implementation-plan.md
documentation:
  - Docs/superpowers/specs/2026-07-12-pr-2633-review-rebase-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase the Research Workspace UAT and artifact verification PR onto current dev, resolve conflicts without regressing current behavior, address every still-valid review comment, verify touched frontend/backend scope, and update the existing PR branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR branch is rebased onto the audited dev commit with conflicts resolved.
- [ ] #2 Every unresolved inline and outside-diff review finding is verified and either fixed or answered with technical rationale.
- [ ] #3 Valid stored ACP configuration takes precedence; runtime single-user API key is only a fallback for missing, blank, or placeholder stored keys, and multi-user auth remains isolated.
- [ ] #4 Focused frontend and backend regression tests pass.
- [ ] #5 Bandit reports no new findings in touched Python code and git diff --check passes.
- [ ] #6 Updated branch is force-pushed with lease and PR review/check status is re-inspected.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-12-pr-2633-review-rebase-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task tracking: after the rebase, current dev already owned TASK-12148 for first-run MCP tool packs. This repair work was migrated to TASK-12949 and the colliding repair record was removed.

Rebase evidence:
- Original PR head: 07292d91aa046f60902d0a81cd0ab354ed991871.
- Audited dev/base: 5a309be86b043f5a67b65324a81819f59aa860fc.
- Rebase completed; merge-base equals the audited dev SHA.
- Conflict reconciliation retained current dev behavior plus claim-verification gates; follow-up commit 9c90471c43 fixed the TypeScript conflict artifact.

Review ledger (reply/resolution status remains pending until the guarded push):
1. satisfied_by_dev — ACP runtime fallback is already string|null and preserves auth metadata.
2. fixed — E2E auth evidence now checks exact configured/legacy key values and removes the stale session key.
3. satisfied_by_dev — activateMenuItem only suppresses nextjs-portal errors and rethrows all others.
4. rejected_with_rationale — runtime-first ACP precedence conflicts with current dev TASK-12905; valid normalized stored single-user keys win, runtime is fallback only.
5. fixed — repeated grounded claim-verification payloads now use createGroundedClaimVerification.
6. fixed — usable markdown slide fallback may complete without presentationId; API presentations remain validated before return.
7. satisfied_by_dev + coverage — explicit multi-user isolation test added; runtime single-user override cannot leak into bearer auth.
8. satisfied_by_dev — duplicate catch-all review finding is covered by the current narrow nextjs-portal exception handling.
9. fixed — E2E evidence no longer reads the obsolete tldwRuntimeSessionSingleUserApiKey key.
10. fixed — duplicate TASK-12142 final-summary end marker removed.
11. fixed — duplicate slides TASK-12143 final-summary end marker removed.
12. fixed — empty normalized flashcard output returns direct 422 before verifier invocation.
13. fixed — Research Workspace artifact media_ids capped at 50 with schema coverage.
14. fixed — no_claims unit results retain truncation metadata and add the reason.
15. fixed — verified units with text/claim truncation are downgraded to needs_revision.
16. fixed — Claims verifier provider/model keys added to the environment allowlist with env-over-config coverage.
17. fixed — MISLEADING maps to needs_revision in the status table test.
18. fixed — flashcard verifier monkeypatches are strict; affected legacy generation tests use a shared grounded verifier fixture.
19. fixed — quiz verifier monkeypatches are strict; missing exception import and current call-signature compatibility repaired.
20. fixed — slides verifier monkeypatches are strict; production now imports get_document_version so the strict patch target exists.
21. fixed — quiz normalization retains each normalized question with its original source entry, preventing citation/media index shifts.

Additional reconciliation fixes found by full verification:
- Removed the obsolete endpoint-local test-mode flashcard builder that shadowed the plan-aware core implementation.
- Removed a duplicate producerMetadata TypeScript field left by conflict resolution.

Independent final review:
- Fixed API-backed slides accepting valid content without a persisted presentation id; fallback markdown remains allowed without an id.
- Added the missing audio_overview cap aliases so Research Workspace audio uses the intended 60-unit, 10-claim, 6,000-character profile.
- Focused reviewer re-review confirmed both findings resolved with no remaining issue in those areas.

Verification:
- Frontend ACP + StudioPane: 113 passed.
- Frontend E2E auth: 4 passed.
- Playwright discovery: 12 Research Workspace tests listed.
- Backend focused suite: 402 passed.
- ESLint changed TS/TSX scope: 0 errors; 101 existing warnings in the rebased PR scope.
- Bandit changed Python implementation scope: 0 findings, final report /tmp/bandit_task12949.json.
- git diff --check: clean.
- TypeScript touched-file error fixed. Full typecheck still reports one unrelated baseline error in QuickIngestWizardModal.tsx:1813 (overflowY inferred as string); that file is unchanged from the audited dev base.
- Final reviewer follow-up suites: 78 affected StudioPane tests and 19 Claims unit/property tests passed.

Known merge gate: the human requester must provide their own Change summary explaining what changed and why before this AI-materially-authored PR is merge-ready.
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
