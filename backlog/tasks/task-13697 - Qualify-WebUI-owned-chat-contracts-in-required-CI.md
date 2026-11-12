---
id: TASK-13697
title: Qualify WebUI-owned chat contracts in required CI
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair final PR 3229 WebUI fixture drift without weakening owner, greeting, credential privacy, failure reporter, or timeout contracts; ensure WebUI-owned unit failures block frontend-required.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All seven reported WebUI files pass using real state and original deadlines
- [x] #2 Required frontend gate runs WebUI-owned tests and rejects failures
- [x] #3 Connected contracts, full owned WebUI qualification, independent review and normal hooks pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #3229. ADR assessment: existing CI required-gate policy and shared chat ownership rules govern; fixture migration and enforcing existing WebUI test coverage introduce no new durable architectural decision. Stages: reproduce and classify; repair fixture contracts and scope accessible setup queries; close required CI ownership gap and qualify; final review and publication.
Reproduced 14 failing assertions across six files locally (36 cases total; setup passed locally but original CI first case was 5198ms against 5000ms). Root causes: retired greeting transcript side effects and untracked fixture seed; incomplete saved scope/assistant metadata; missing Zustand getState in mocks; credentialed discovery now authenticated; reporter stdout and mobile panels wording drift. Real preview is rendered, tracked mode/saved metadata declared, expected hashed credential scope+signal retained, actual store restored, public noAuth header/cookie privacy tested separately. Profiling delegated actual getComputedStyle: 62 calls and about 2 seconds before Save; now scope original accessible role queries to labeled field/button container, no timeout changes. Existing full WebUI coverage now blocks failures, coverage percentages advisory, same 15m/17m limits, two workers. Original account/clear transcript negatives retained alongside preview and empty-history checks after independent reviewer feedback. Node26/default16-worker full diagnostic excluded; rerunning exact CI Node20/two-worker coverage.
Final exact CI-runtime command Node20/two-worker full WebUI coverage passes192files/2371cases, existing4files/14cases skipped,325.82s within unchanged15m. All7 reported files pass. Connected contracts126pass; final gate subset24pass. Seven TS lint0errors/0warnings, Black/Ruff/actionlint pass; normal hooks pass. Touched test Bandit50baseline=50current B101,0errors. Both independent reviewers clear restored transcript+preview privacy oracles. All98 production hashes unchanged; protected9conversations/28rows deepEqual immutable baseline. Separate advisory shared-UI report incomplete15m with32 Skills Manager fixture failures; inspected source/dependencies match dev, causal attribution unqualified, no shared-UI full pass claimed. Handoff: Docs/Reviews/CHAT_FINAL_RECOVERY_SETUP_HANDOFF_13673_13680_2026_10_14.md. Final committed-head review/CI remain landing gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Repaired WebUI fixtures against current preview, ownership, scope and authentication contracts, scoped accessible setup queries and made existing full WebUI test failures block frontend-required with unchanged deadlines. Full owned coverage2371pass, connected contracts126pass, lint/security/hooks clear; production and protected conversations unchanged. Final published-head CI precedes merge.
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
