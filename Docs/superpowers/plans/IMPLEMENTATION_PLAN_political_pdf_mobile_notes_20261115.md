# TASK-15505.5 — Mobile reading and Notes capture

Design: [approved addendum](../../Design/POLITICAL_PDF_VIP_FIXES_2026_11_15.md). ADR assessment: ADR-007/065 apply; no new persistence, API or authority policy.

## Stage 1: Regression coverage
**Goal**: Reproduce expanded mobile controls, undisclosed draft capture and empty Contents dead end.
**Success Criteria**: Behavioral tests fail for those missing affordances.
**Tests**: ResearchWorkspace responsive, ChatPane Stage 3, TableOfContentsTab.
**Status**: Complete — RED5; saved-notice follow-up RED1.

## Stage 2: Existing component projection
**Goal**: Default mobile controls closed, reveal captured draft and offer empty Contents hide action.
**Success Criteria**: Existing desktop, canonical Save, citation/provenance and populated outline behavior remain intact.
**Tests**: Focused red-to-green suites, warning/empty semantic tokens.
**Status**: Complete — 165 tests across nine suites pass.

## Stage 3: Verification and independent review
**Goal**: Verify types/lint and phone/keyboard usability through parent acceptance.
**Success Criteria**: Focused suites green, no new lint/type errors, positive reading height on 390×844 and usable draft focus.
**Tests**: Full affected suites; parent browser acceptance. Bandit does not apply to TypeScript-only scope.
**Status**: In Progress — typecheck passes; lint has no new findings versus HEAD; detector clean; parent browser/independent review pending.
