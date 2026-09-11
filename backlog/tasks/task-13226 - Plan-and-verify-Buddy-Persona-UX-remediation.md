---
id: TASK-13226
title: Plan and verify Buddy Persona UX remediation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 22:15'
updated_date: '2026-09-09 03:43'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the approved Buddy and Persona usability findings in tldw_server and its shared WebUI/extension, preserving independent artwork, explicit attachment scope, workspace defaults, and work continuity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every approved finding maps to an implemented change or verified existing behavior on latest dev.
- [x] #2 Focused automated tests, rendered desktop and compact walkthroughs, security checks, and a final review verify the combined behavior.
- [x] #3 Design, ownership decision, user documentation, and implementation evidence describe the final behavior.
- [x] #4 New workspace creation tests are assigned to all corresponding CI shard variants and the shard coverage guard passes.
- [x] #5 Verified PR review findings are fixed or explained with source evidence, and affected behavior has focused regression coverage.
- [x] #6 The current frontend shard failures are reproduced and repaired without weakening assertions, and focused checks plus review verify the fixes before merge.
- [x] #7 The newly reported diagnostics-card cap test verifies asynchronous publication and the eight-card bound deterministically; affected tests and independent review pass.
- [x] #8 The newly exposed cockpit mock import and assistant reload synchronization failures are corrected without changing product behavior or weakening assertions; focused verification and independent review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md. Reason: existing frontend test harness corrections only. Follow Task3 in Docs/superpowers/plans/2026-09-09-buddy-pr-review-and-merge.md; preserve Buddy scope and do not change Flashcards features.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed Buddy/Persona remediation for tldw_server and the shared WebUI/extension on isolated origin/dev base6cd2745f69. No Flashcards changes. All reviewed findings map to implemented behavior or verified latest-dev equivalents in Docs/superpowers/plans/2026-09-08-buddy-persona-ux-remediation.md; existing workspace defaults and Ant Design status contrast were reused. Final shared UI437 tests/26 files and WebUI app-layout/networking40 tests passed, plus backend foundation34 tests including real PostgreSQL18 and targeted authenticated Chat/queue/ledger regressions. Rendered management and drawer mobile/theme matrices, exact target reply, real Next navigation preserving draft/selection/speech controls, Cancel/Escape/X focus and discard, one visible Buddy, and38 status contrast checks passed. Python security/static checks and final diff whitespace checks passed. Frontend typecheck retains81 unchanged errors outside modified files; no changed-file errors. ADR-005, design/plan, user guide/navigation, API/Operations and evidence lessons are updated. Limitations: fixture browser APIs and controlled backend provider responses; no real audio/model provider, packaged extension or full suite. Accepted Buddy work is process-owned with documented restart/no-replay and principal-affinity limits; legacy Persona Live remains connection-owned. No commit, PR, merge or publication performed.

PR follow-up: published PR #2933 against dev, then reproduced and fixed the shard coverage guard failure by assigning test_workspace_assistant_creation.py to all five workspace CI matrix variants. Guard now reports zero newly uncovered files; diff whitespace check passed. Fresh pre-PR reruns passed 437 UI tests and 57 backend tests (one PostgreSQL environment skip). No merge performed.

Completed all 13 PR comment dispositions in Docs/Reviews/BUDDY_PERSONA_PR_FOLLOWUP_2026_09_08.md under existing ADR-005. Fixed foreign workspace disclosure, asynchronous Apply retarget races, overly broad validation mapping, endpoint/core placement, docs and HTTP/auth artwork coverage; replaced timing sleep with actual worker settlement. Canonical selection and live-session counterexamples have regression coverage; integration test relocation explicitly declined. Independent review identified the async race and confirms the final fix has no remaining Important/Critical findings. Follow-up UI109 passed; final race modal/host24 passed. All51 runnable backend cases pass across the50-pass run and corrected TestClient assertion rerun; one PostgreSQL environment skip. Bandit0,12-file Ruff/format pass; existing Chat/TSX formatting debt documented. Published-docs33 and CommandPalette26 pass; OpenAPI/types fingerprint and shard registration checks pass. Remaining older-head frontend CI failures and source-comparison limits are documented without claiming pristine-dev reproduction. No full suite, real provider/audio, packaged extension or merge.

September 9 CI repair complete under existing ADR-005; latest dev6cd2745f69 required no rebase changes. Reproduced all remaining frontend shard failures and repaired6existing test files with current router/provider/MCP/service-prompt/remediation mocks, documented Research publication/failure ordering and deterministic polling, and distinct route-only locale expectations. No production/dependency changes or weakened assertions. Final combined58tests pass12.86s; fresh backend60passed1PostgreSQL-environment skip. Pinned ESLint0errors, no introduced warnings; no formatter changes intersect edited ranges. Independent review approved with no findings. Docs/Reviews/BUDDY_PERSONA_PR_FOLLOWUP_2026_09_08.md records contracts, evidence and environment limits. Bandit N/A for this test-only TypeScript delta; previous backend Bandit evidence unchanged. Remote CI and merge remain integration steps.

Final shard7 correction: deterministic deferred diagnostics response reproduces the premature DOM query. The test now verifies loading, waits for actual response publication and asserts exactly8of120cards plus summary; success and forbidden-response coverage are unchanged. Only one test file changed. Final3test file and70related ChatPane tests pass; no new pinned ESLint/formatter findings, diffcheckclean. Independent review approved with no findings. ExistingADR005unchanged; no production/runtime change. Evidence appended to Docs/Reviews/BUDDY_PERSONA_PR_FOLLOWUP_2026_09_08.md.

Shard 5 repair complete: the cockpit partial mock preserves real service exports; the ReviewTab test deterministically covers the loading-guarded early click and a separately accepted manual reload, retaining Retry until completion. No product behavior or assertion was weakened. Focused runs passed 53 affected tests, 72 including the direct title consumer, and 42 cockpit tests after formatting (overlapping counts). Pinned ESLint/formatter delta and whitespace checks introduce no findings. Independent review approved both repairs. Existing ADR-005 applies; no new ADR or Python security scan is needed for this TypeScript test-only delta. Evidence is recorded in Docs/Reviews/BUDDY_PERSONA_PR_FOLLOWUP_2026_09_08.md. Final published CI and merge remain root integration steps.
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
