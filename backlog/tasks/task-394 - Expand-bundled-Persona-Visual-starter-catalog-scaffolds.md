---
id: TASK-394
title: Expand bundled Persona Visual starter catalog scaffolds
status: Done
assignee: []
created_date: '2026-05-16 00:36'
updated_date: '2026-05-16 02:23'
labels:
  - persona
  - buddy
  - visuals
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1732'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-14-persona-buddy-default-catalog-state-catalog-extension-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement issue #1732 by adding nine server-owned Persona Visual starter catalog scaffold fixtures. Preserve copy-to-user-owned-inactive-draft behavior, avoid runtime renderer expansion, and make clear this slice does not ship finished buddy artwork or completed animation packs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Starter catalog lists all nine approved default starter IDs in stable order.
- [x] #2 Each starter manifest includes required built-in states and validates under the existing sprite_frames V1 contract.
- [x] #3 Copying every starter creates an inactive user-owned draft with remapped asset IDs and no fixture asset-key leakage.
- [x] #4 Existing research-buddy-starter compatibility remains intentional and tested.
- [x] #5 Relevant Persona Visual docs describe the nine-default catalog and non-goals.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing starter catalog fixture/service/tests. 2. Add reusable fixture helpers and nine starter definitions while preserving research-buddy-starter compatibility. 3. Expand focused unit coverage for listing, manifest validity, and copy behavior across all starters. 4. Update docs/tracker notes and run focused pytest, Bandit on touched backend scope, and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created after PR #1725 merged and issue #1695 closed. Issue #1732 tracks this next Persona/Buddy visual catalog slice under epic #1510.

Supersession note from TASK-419: this task recorded the earlier nine-scaffold
catalog expansion. Later basic-tier asset production replaced that basic set
with the six Codex Buddy defaults and the current twelve-ID starter catalog.
Keep this task as historical evidence for the scaffold expansion, not as the
current basic-tier source of truth.

Implementation started in worktree .worktrees/persona-visual-nine-starters-1732. Baseline focused starter catalog pytest passed before edits: 9 tests.

Implemented the initial bundled Persona Visual starter fixture set with stable IDs, required sprite_frames states, custom-state examples, an atlas-backed starter, legacy research-buddy-starter alias support, API/test expectation updates, and docs for the then-current default catalog.

Verification: focused starter catalog pytest passed with 20 tests; broader persona visual slice passed with 84 tests across starter catalog, visual API, and visual service; git diff --check passed; Bandit JSON report for touched Persona backend starter modules reported zero findings.

Post-docstring verification refreshed: persona visual starter catalog, visual API, and visual service pytest passed with 84 tests; git diff --check passed; Bandit JSON report still has zero findings.

Reopened for wording correction after review: clarify that this PR adds backend catalog scaffold fixtures, not finished animated buddy art or completed default animation packs.

Clarified fixture descriptions, tags, tests, docs, and task language so the PR represents backend catalog scaffolds only. The real default buddy art/animation pipeline remains future work.

Wording-correction verification: focused starter catalog/API pytest passed with 23 tests; git diff --check passed; Bandit JSON report for touched Persona starter backend modules reported zero findings.

PR #1734 review fixes: wrapped the long fixture/test lines identified by Qodo and changed multi-custom-state scaffold generation so each custom state gets a distinct deterministic variant asset key. Added a regression test for action-guide and elaborate-persona custom-state asset separation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the initial bundled Persona Visual starter catalog scaffold fixtures across basic, intermediate, and intricate tiers. Added deterministic fixture PNG generation, custom-state and atlas metadata examples, legacy research-buddy-starter alias handling, updated API/service coverage, and documented that these are backend scaffolds rather than finished buddy art or completed animation packs. PR review fixes also wrap newly touched long Python lines and ensure multi-custom-state scaffolds use distinct variant fixture assets. Verification: focused starter catalog/API pytest passed with 25 tests after review fixes; prior broader persona visual slice passed with 84 tests; git diff --check passed; Bandit on touched Persona starter backend modules reported zero findings. Known skips/blockers: real default buddy art and neutral-pose-to-animation asset creation remain future work.
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
