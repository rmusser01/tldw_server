---
id: TASK-13452
title: Preserve model settings when cached chat dialog remounts
status: In Progress
assignee: []
created_date: '2026-10-05 07:36'
updated_date: '2026-10-05 07:43'
labels: []
dependencies: []
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
- [ ] #3 Focused real-component regressions and adjacent scoped-settings tests pass; record upstream PR and hosted backport.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Reproduce with real ModelBasicsTab and shared QueryClient; initialize all current model fields from existing state and cached config; run focused red/green plus adjacent tests and review before PR. ADR not required: restore existing per-model settings contract, no new architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: real ModelBasicsTab + shared QueryClient Save/remount returns blank token field although store numPredict=48. Minimal two-line production fix reuses buildBaseValues(modelConfig) for cached Form initialization. GREEN: 51 tests across real dialog, numeric normalization, scoped settings, and provider selection passed. Independent review Godel: no actionable findings across four callers/owner invalidation; no new architecture (ADR not required). Adding real numeric clear/Save/remount coverage before commit. Python3.12 Bandit frontend-only scan has zero executable Python scope.

Correction: Bandit cannot run in the shared Python3.12 venv (No module named bandit), and no standalone bandit executable is installed. This PR changes only TypeScript/React plus task Markdown; no Python executable source. Independent code/security-owner review completed; no claim of a successful Bandit scan.

Final focused run:51/51 passed including real Save/remount/save plus explicit numeric clear/save/remount (four real Antd dialogs; test uses a15s allowance, completed8.2s). Diffcheck clean. Hosted production-only patch023 and strict source-parity test prepared underTASK14.4.74. Shared production SHA2567a05bb6a3dc104728954d285f2c54f42c177ef34cde6e34b2a374e759c71ca32. Change-summary requirement waived by owner in current thread. Actual staging deployment remains pending; this task does not claim betaGO.
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
