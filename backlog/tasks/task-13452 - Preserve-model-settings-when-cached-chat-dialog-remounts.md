---
id: TASK-13452
title: Preserve model settings when cached chat dialog remounts
status: In Progress
assignee: []
created_date: '2026-10-05 07:36'
updated_date: '2026-10-05 07:53'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/3193'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Staging acceptance reproduced new-chat settings blank on cached dialog remount after Save; uncapped outgoing request was blocked by a client safety guard. Shared CurrentChatModelSettings initializes only systemPrompt when its query is cached. Cover real token/provider fields across Save/remount and fix shared initialization without changing defaults, provider identities, or resetting active edits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cached remount displays the current token limit and provider alongside existing prompt settings.
- [x] #2 Saving a cached remount preserves token limit and provider routing; clearing numeric input still works.
- [x] #3 Focused real-component regressions and adjacent scoped-settings tests pass; record upstream PR and hosted backport.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Reproduce with real ModelBasicsTab and shared QueryClient; initialize all current model fields from existing state and cached config; run focused red/green plus adjacent tests and review before PR. ADR not required: restore existing per-model settings contract, no new architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED: real ModelBasicsTab + shared QueryClient Save/remount returns blank token field although store numPredict=48. Minimal two-line production fix reuses buildBaseValues(modelConfig) for cached Form initialization. GREEN: 51 tests across real dialog, numeric normalization, scoped settings, and provider selection passed. Independent review Godel: no actionable findings across four callers/owner invalidation; no new architecture (ADR not required). Adding real numeric clear/Save/remount coverage before commit. Python3.12 Bandit frontend-only scan has zero executable Python scope.

Correction: Bandit cannot run in the shared Python3.12 venv (No module named bandit), and no standalone bandit executable is installed. This PR changes only TypeScript/React plus task Markdown; no Python executable source. Independent code/security-owner review completed; no claim of a successful Bandit scan.

Final focused run:51/51 passed including real Save/remount/save plus explicit numeric clear/save/remount (four real Antd dialogs; test uses a15s allowance, completed8.2s). Diffcheck clean. Hosted production-only patch023 and strict source-parity test prepared underTASK14.4.74. Shared production SHA2567a05bb6a3dc104728954d285f2c54f42c177ef34cde6e34b2a374e759c71ca32. Change-summary requirement waived by owner in current thread. Actual staging deployment remains pending; this task does not claim betaGO.

Upstream PR3193 opened targeting latestdev49cec71190. Private PR119 headc1f8efb contains exact production-only backport and111 strict focused checks; all1110 hostedfrontend checks pass. Qodo externalreview requested and is running. PublicCI queued; merge pendingCI/review. Local25GB-start/12GB-floor WebUI build started independently, no production target.

Qodo review5990315949 posted valid configured-default clear case: undefined cleared temperature falls back to cached0.7 on remount and Save. Reproducing explicitclear with cachedgetAllModelSettings first, splitting clearing into its own focused scenario per second Qodo comment. The minimum correct refinement is current-state-only initialization on cached remount; fresh query still supplies initial configuration as before. No extra cleared-state registry or refetch needed. Private119 merged but its currentlybuildingc1f8 image will NOTbe deployed while this finding is open.

Qodo bug verifiedRED: configuredtemperature0.7 revivedafterexplicitclear. GREEN refinement: initialValues=buildBaseValues() reads current state only on cached remount; freshquery still applies fetched defaults.52/52 focused tests pass, including separate realclear/remount/Save regression. Second Qodo test-separation issue also addressed. Production change reduced to ONE original line, no cleared-value registry/refetch/newdependency. Private source parity now66c6352239fe53fa0db72f2912ec8d6d7f2f90ef3e5d3dd2927180eee85aba4d. Numeric clearer no longer needs15s combinedtest allowance; independentcases each<5s.
Review refinement: independently verify Cancel/reopen against the original dev baseline before accepting a requested defaults-on-open store mutation. Such a mutation changes request settings even when users cancel. Existing saved-value and explicit-clearing regressions passed 52 focused tests; final rerun and PR replies pending. Task edits now use repository backlog-py and canonical normalization as required by current AGENTS.md.
Final refinement review found no new regression against dev49cec711. Cancel/reopen default display gap is pre-existing; rejected default store seeding because Cancel must not commit request values or introduce scope/owner races. Fresh final verification: 42 real-dialog/normalization/scoped tests plus10 provider-selection tests passed (52 total); hosted Python3.12 compatibility/overlay66 and all1110 hostedfrontend tests passed. Task normalize --check and git diff --check clean. Existing environment warnings (JSDOM CSS/localStorage, shared pytest cleanup) do not fail tests. Addressing both Qodo inline findings with current-state-only initialization and independent cached-default clearing test. Live corrected build/deployment remains pending local headroom; no betaGO claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
