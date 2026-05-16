---
id: TASK-415
title: Produce basic tier Buddy default animation packs
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-16 22:33
labels:
- persona
- buddy
- visuals
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/1807
- https://github.com/rmusser01/tldw_server/issues/1803
- https://github.com/rmusser01/tldw_server/issues/1787
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Produce and validate the three bundled basic tier Buddy defaults as production-ready visual packs: research-buddy-basic, migu-marker-basic, and minimal-helper-basic.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 research-buddy-basic has reviewed production assets and passes manifest validation.
- [x] #2 migu-marker-basic has reviewed production assets and passes manifest validation.
- [x] #3 minimal-helper-basic has reviewed production assets and passes manifest validation.
- [x] #4 Each basic pack imports as an inactive draft before activation.
- [x] #5 Each basic pack renders through sprite-frame manifest coverage for all required states.
- [x] #6 Visual review evidence is stored at Docs/Code_Documentation/assets/persona-basic-buddy-defaults-review.png.
- [x] #7 Catalog status is art_ready only for the three basic packs touched in this slice.
- [x] #8 Documentation states the nine defaults are bundled and additional packs are optional.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-16-basic-buddy-default-assets-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -> 64 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_starter_fixtures.py -> passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_starter_fixtures.py -f json -o /tmp/bandit_basic_buddy_defaults.json -> 0 findings.
- git diff --check -> passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the basic tier Buddy default asset slice and follow-up recreation walkthrough. The three basic bundled starters now expose art_ready production metadata, deterministic transparent 96x96 neutral/preview assets, two-frame required-state loops for idle, listening, thinking, speaking, and error, and design-specific neutral-pose/state-delta guidance for recreating each default. Intermediate and intricate starters remain scaffolded for later tracking issues. Verification passed: focused pytest, py_compile, Bandit, and git diff --check.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Follow-up PR update: added design-specific recreation guidance for research-buddy-basic, migu-marker-basic, and minimal-helper-basic in starter production metadata and docs. Verification: focused recreation-guidance pytest failed before metadata updates, then passed; full starter catalog pytest -> 67 passed, 5 warnings; py_compile passed; Bandit -> 0 findings; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
