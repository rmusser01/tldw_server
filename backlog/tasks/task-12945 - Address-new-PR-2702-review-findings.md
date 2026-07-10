---
id: TASK-12945
title: Address new PR 2702 review findings
status: In Progress
labels:
- review
- release
- frontend
- ci
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate and address the new CodeRabbit/Gemini review findings on PR #2702, add regression coverage for confirmed issues, resolve stale comments with evidence, and re-audit the PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every new unresolved PR #2702 review thread is either fixed with verification or answered with evidence that the current head already satisfies it
- [ ] #2 Confirmed frontend state, accessibility, i18n, preflight, and recovery-action issues have regression coverage
- [ ] #3 Focused tests and relevant type/workflow checks pass
- [ ] #4 PR review threads are replied to and resolved, with a final audit showing no unresolved actionable comments
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Validated 12 unresolved inline CodeRabbit threads plus the outside-diff queued-send finding. Implemented stable publish concurrency, 2xx/DOM assertion hardening, document-processing live status, localized OpenWebUI scope summaries, manual-scope preservation, per-job OpenWebUI refresh classification backed by persisted async job source metadata, explicit ready/pending counts, capability gating, selected-mode-only block reasons, no-op recovery removal, and pre-validation of queued documents before composer state mutation. Populated the reviewed planning task's acceptance criteria. Verification: 52 focused Vitest tests passed; apps/tldw-frontend typecheck passed; extension compile passed; Playwright discovered the changed extension E2E test; 7 Chatbooks backend tests passed; git diff --check passed. CodeRabbit's docstring warning was audited with AST over all changed Python definitions and found no changed function without a docstring.
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
