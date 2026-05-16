---
id: TASK-394
title: Expand bundled Persona Visual starter catalog to nine defaults
status: Done
assignee: []
created_date: '2026-05-16 00:36'
updated_date: '2026-05-16 01:25'
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
Implement issue #1732 by expanding the server-owned Persona Visual starter fixture catalog from the single Research Buddy starter to the approved nine default buddy starter packs. Preserve copy-to-user-owned-inactive-draft behavior and avoid runtime renderer expansion.
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

Implementation started in worktree .worktrees/persona-visual-nine-starters-1732. Baseline focused starter catalog pytest passed before edits: 9 tests.

Implemented nine bundled Persona Visual starter fixture packs with stable IDs, required sprite_frames states, custom-state examples, an atlas-backed starter, legacy research-buddy-starter alias support, API/test expectation updates, and docs for the nine-default catalog.

Verification: focused starter catalog pytest passed with 20 tests; broader persona visual slice passed with 84 tests across starter catalog, visual API, and visual service; git diff --check passed; Bandit JSON report for touched Persona backend starter modules reported zero findings.

Post-docstring verification refreshed: persona visual starter catalog, visual API, and visual service pytest passed with 84 tests; git diff --check passed; Bandit JSON report still has zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Expanded the bundled Persona Visual starter catalog to nine default sprite_frames packs across basic, intermediate, and intricate tiers. Added deterministic fixture PNG generation, custom-state and atlas examples, legacy research-buddy-starter alias handling, updated API/service coverage, and documented the nine-default catalog and non-goals. Verification: pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py tldw_Server_API/tests/Persona/test_persona_visual_service.py -q passed with 84 tests; git diff --check passed; Bandit on touched Persona starter backend modules reported zero findings. Known skips/blockers: none.
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
