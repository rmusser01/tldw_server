---
id: TASK-415
title: Produce basic tier Buddy default animation packs
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-16 20:32
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
modified_files:
- backlog/tasks/task-415 - Produce-basic-tier-Buddy-default-animation-packs.md
- Docs/superpowers/plans/2026-05-16-basic-buddy-default-assets-plan.md
- Docs/Code_Documentation/Persona_Visual_Packs.md
- tldw_Server_API/app/api/v1/endpoints/persona.py
- tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py
- tldw_Server_API/app/core/Persona/visual_portability/preview.py
- tldw_Server_API/app/core/Persona/visual_portability/importer.py
- tldw_Server_API/app/core/Persona/visual_service.py
- tldw_Server_API/tests/Persona/test_persona_visual_portability.py
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/source/search-lens-basic-3x4-source.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/source/search-lens-basic-3x4-transparent.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/frames-v2/00_neutral_anchor.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/frames-v2/01_idle_a.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/frames-v2/02_idle_b.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/frames-v2/03_listening_a.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/frames-v2/04_listening_b.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/frames-v2/05_thinking_a.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/frames-v2/06_thinking_b.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/frames-v2/07_speaking_a.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/frames-v2/08_speaking_b.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/frames-v2/09_success.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/frames-v2/10_error_a.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/frames-v2/11_error_b.png
- Docs/Code_Documentation/assets/buddy-defaults/search-lens-basic/review/search-lens-basic-3x4-processed-review-v2.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/source/index-card-basic-3x4-source.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/source/index-card-basic-3x4-transparent.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/frames-v2/00_neutral_anchor.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/frames-v2/01_idle_a.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/frames-v2/02_idle_b.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/frames-v2/03_listening_a.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/frames-v2/04_listening_b.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/frames-v2/05_thinking_a.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/frames-v2/06_thinking_b.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/frames-v2/07_speaking_a.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/frames-v2/08_speaking_b.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/frames-v2/09_success.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/frames-v2/10_error_a.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/frames-v2/11_error_b.png
- Docs/Code_Documentation/assets/buddy-defaults/index-card-basic/review/index-card-basic-3x4-processed-review-v2.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/source/archive-cube-basic-3x4-source.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/source/archive-cube-basic-3x4-transparent.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/frames-v1/00_neutral_anchor.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/frames-v1/01_idle_a.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/frames-v1/02_idle_b.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/frames-v1/03_listening_a.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/frames-v1/04_listening_b.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/frames-v1/05_thinking_a.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/frames-v1/06_thinking_b.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/frames-v1/07_speaking_a.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/frames-v1/08_speaking_b.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/frames-v1/09_success.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/frames-v1/10_error_a.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/frames-v1/11_error_b.png
- Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/review/archive-cube-basic-3x4-processed-review-v1.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/source/paperclip-basic-3x4-source.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/source/paperclip-basic-3x4-transparent.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/frames-v1/00_neutral_anchor.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/frames-v1/01_idle_a.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/frames-v1/02_idle_b.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/frames-v1/03_listening_a.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/frames-v1/04_listening_b.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/frames-v1/05_thinking_a.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/frames-v1/06_thinking_b.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/frames-v1/07_speaking_a.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/frames-v1/08_speaking_b.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/frames-v1/09_success.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/frames-v1/10_error_a.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/frames-v1/11_error_b.png
- Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/review/paperclip-basic-3x4-processed-review-v1.png
- pyproject.toml
- tldw_Server_API/app/core/Persona/visual_starter_fixtures.py
- Docs/Code_Documentation/assets/buddy-defaults/terminal-tile-basic/source/terminal-tile-basic-neutral-source-v1.png
- Docs/Code_Documentation/assets/buddy-defaults/terminal-tile-basic/source/terminal-tile-basic-neutral-transparent-v1.png
- Docs/Code_Documentation/assets/buddy-defaults/terminal-tile-basic/source/terminal-tile-basic-3x4-source-v1.png
- Docs/Code_Documentation/assets/buddy-defaults/terminal-tile-basic/source/terminal-tile-basic-3x4-transparent-v1.png
- Docs/Code_Documentation/assets/buddy-defaults/terminal-tile-basic/review/terminal-tile-basic-3x4-processed-review-v1.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/terminal-tile-basic/frames
- Docs/Code_Documentation/assets/buddy-defaults/migu-marker-basic/source/migu-marker-basic-neutral-source-v1.png
- Docs/Code_Documentation/assets/buddy-defaults/migu-marker-basic/source/migu-marker-basic-neutral-transparent-v1.png
- Docs/Code_Documentation/assets/buddy-defaults/migu-marker-basic/source/migu-marker-basic-3x4-source-v1.png
- Docs/Code_Documentation/assets/buddy-defaults/migu-marker-basic/source/migu-marker-basic-3x4-transparent-v1.png
- Docs/Code_Documentation/assets/buddy-defaults/migu-marker-basic/frames-v1
- Docs/Code_Documentation/assets/buddy-defaults/migu-marker-basic/review/migu-marker-basic-3x4-processed-review-v1.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/migu-marker-basic/frames
- tldw_Server_API/app/core/Persona/assets/starter_packs/search-lens-basic/frames/00_neutral_anchor.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/search-lens-basic/frames/01_idle_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/search-lens-basic/frames/02_idle_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/search-lens-basic/frames/03_listening_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/search-lens-basic/frames/04_listening_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/search-lens-basic/frames/05_thinking_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/search-lens-basic/frames/06_thinking_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/search-lens-basic/frames/07_speaking_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/search-lens-basic/frames/08_speaking_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/search-lens-basic/frames/09_success.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/search-lens-basic/frames/10_error_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/search-lens-basic/frames/11_error_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/index-card-basic/frames/00_neutral_anchor.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/index-card-basic/frames/01_idle_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/index-card-basic/frames/02_idle_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/index-card-basic/frames/03_listening_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/index-card-basic/frames/04_listening_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/index-card-basic/frames/05_thinking_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/index-card-basic/frames/06_thinking_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/index-card-basic/frames/07_speaking_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/index-card-basic/frames/08_speaking_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/index-card-basic/frames/09_success.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/index-card-basic/frames/10_error_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/index-card-basic/frames/11_error_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/archive-cube-basic/frames/00_neutral_anchor.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/archive-cube-basic/frames/01_idle_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/archive-cube-basic/frames/02_idle_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/archive-cube-basic/frames/03_listening_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/archive-cube-basic/frames/04_listening_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/archive-cube-basic/frames/05_thinking_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/archive-cube-basic/frames/06_thinking_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/archive-cube-basic/frames/07_speaking_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/archive-cube-basic/frames/08_speaking_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/archive-cube-basic/frames/09_success.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/archive-cube-basic/frames/10_error_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/archive-cube-basic/frames/11_error_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/paperclip-basic/frames/00_neutral_anchor.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/paperclip-basic/frames/01_idle_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/paperclip-basic/frames/02_idle_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/paperclip-basic/frames/03_listening_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/paperclip-basic/frames/04_listening_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/paperclip-basic/frames/05_thinking_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/paperclip-basic/frames/06_thinking_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/paperclip-basic/frames/07_speaking_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/paperclip-basic/frames/08_speaking_b.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/paperclip-basic/frames/09_success.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/paperclip-basic/frames/10_error_a.png
- tldw_Server_API/app/core/Persona/assets/starter_packs/paperclip-basic/frames/11_error_b.png
- tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py
- tldw_Server_API/tests/Persona/test_persona_visual_service.py
- tldw_Server_API/tests/Persona/test_persona_visuals_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Produce and validate the six bundled basic tier Buddy defaults as production-ready visual packs: search-lens-basic, index-card-basic, archive-cube-basic, paperclip-basic, terminal-tile-basic, and migu-marker-basic. Work proceeds one buddy at a time with user approval checkpoints for source sheet, processed frames, and catalog integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 search-lens-basic has an approved 3x4 source sheet and reviewed v2 processed frame packet.
- [x] #2 search-lens-basic is wired as an art-ready bundled starter with package-backed PNG resources.
- [x] #3 search-lens-basic imports as an inactive draft before activation.
- [x] #4 search-lens-basic renders through sprite-frame manifest coverage for all required states plus reaction.success.
- [x] #5 index-card-basic has reviewed production assets and passes manifest validation.
- [x] #6 archive-cube-basic has reviewed production assets and passes manifest validation.
- [x] #7 paperclip-basic has reviewed production assets and passes manifest validation.
- [x] #8 terminal-tile-basic has reviewed production assets and passes manifest validation.
- [x] #9 migu-marker-basic is reviewed against the six-basic-pack direction and passes manifest validation.
- [x] #10 Basic-tier documentation and visual review evidence reflect the final six bundled basic defaults.
- [x] #11 Codex/Petdex pet import preview and commit accept `pet.json` plus spritesheet `.zip` packages as draft Persona Visual packs.
- [x] #12 Basic-tier production planning is pivoted to Codex-pet-compatible 8x9 atlas packets, with `moving_right` and `moving_left` as movement states.
- [x] #13 Buddy creation docs use hatch-pet as the reference workflow for a tldw-native Simple Buddy Creator with simple draft-pack and full Codex-compatible modes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-16-basic-buddy-default-assets-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Search Lens checkpoint:
- Approved source sheet processed with fixed global crop/scale into 12 transparent 96x96 frames.
- Runtime starter now uses package-backed PNG resources for search-lens-basic.
- search-lens-basic is the default starter and includes the five required states plus reaction.success.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -> 67 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q -> 63 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python package resource/image validation script -> validated search-lens-basic package resources and manifest.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_starter_fixtures.py -> passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_starter_fixtures.py -f json -o /tmp/bandit_search_lens_basic.json -> 0 findings.
- git diff --check -> passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the six-pack basic Buddy default slice. The catalog now exposes Search Lens, Index Card, Archive Cube, Paperclip, Terminal Tile, and Migu Marker as art-ready starter packs with package-backed 96x96 frame resources, required-state loops, neutral/preview assets, inactive draft copy semantics, and reaction.success coverage. Docs now include the final six review packets plus Simple Buddy/Codex-compatible creation guidance.
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

## Checkpoint Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Current checkpoint wires the approved Search Lens Buddy v2 frame packet into the starter catalog. Remaining work is the next approved basic buddies: index card, archive cube, paperclip, terminal tile, then Migu review against the six-basic-pack direction.
Index Card checkpoint:
- Approved source sheet processed into 12 transparent 96x96 frames with one global scale and component-mask extraction because the generated sheet had irregular gutters.
- Runtime starter catalog now includes index-card-basic with package-backed PNG resources, required state loops, and reaction.success.
- Documentation now records the Index Card review packet and recreation walkthrough.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py::test_basic_starter_packs_use_reviewed_multi_frame_state_assets tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py::test_basic_starter_packs_expose_design_specific_recreation_guidance -q -> 8 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -> 70 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q -> 63 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python package resource/image validation script -> validated search-lens-basic and index-card-basic package resources and manifests.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_starter_fixtures.py -> passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_starter_fixtures.py -f json -o /tmp/bandit_basic_buddy_defaults.json -> 0 findings.
- git diff --check -> passed.

PR review follow-up:
- Removed the duplicate task heading by renaming this section to `Checkpoint Notes`.
- Simplified ZIP member-name handling to use `ZipInfo.filename` for Codex/native import preview member maps.
- Added Codex pet loader diagnostic logs for archive load, member validation, manifest resolution, sprite dimensions, manifest validation, and successful load.
- Hardened Codex pet import commit so asset/manifest failures clean up the newly created draft pack and assets before replacement handling.
- Hardened no-target Codex pet preview choices so imports without a source persona require `select_existing_persona`.
- Left the Codex pet double-open review item unchanged as a non-blocking performance tradeoff: validation and content loading stay separate, and archive/member sizes are already bounded.

PR review follow-up verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q -> 20 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py -q -> 14 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -> 76 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q -> 64 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/tests/Persona/test_persona_visual_portability.py -> passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py -f json -o /tmp/bandit_basic_buddy_pr_review.json -> 0 findings, 0 errors.
- git diff --check -> passed.
Archive Cube processed-frame review checkpoint:
- User approved the Archive Cube 3x4 source sheet.
- Source was copied into Docs/Code_Documentation/assets/buddy-defaults/archive-cube-basic/source/ and chroma-keyed to alpha.
- Processed v1 frame packet uses component-mask extraction with one global scale and nearest-component accent assignment.
- Pending user review before runtime catalog/package wiring.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python Archive Cube frame validation script -> validated 12 transparent 96x96 frames plus 384x512 review sheet.
- git diff --check -> passed.
Archive Cube catalog wiring checkpoint:
- Runtime starter catalog now includes archive-cube-basic in stable order after index-card-basic.
- archive-cube-basic uses the approved v1 package-backed PNG frame packet, required state loops, and reaction.success.
- Documentation now records the Archive Cube review packet and recreation walkthrough.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py::test_basic_starter_packs_use_reviewed_multi_frame_state_assets tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py::test_basic_starter_packs_expose_design_specific_recreation_guidance -q -> 10 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -> 73 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q -> 63 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python package resource/image validation script -> validated search-lens-basic, index-card-basic, and archive-cube-basic package resources and manifests.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_starter_fixtures.py -> passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_starter_fixtures.py -f json -o /tmp/bandit_basic_buddy_defaults.json -> 0 findings.
- git diff --check -> passed.
Paperclip processed-frame review checkpoint:
- User approved the Paperclip 3x4 source sheet.
- Source was copied into Docs/Code_Documentation/assets/buddy-defaults/paperclip-basic/source/.
- The generated sheet used a magenta cell background plus white grid lines, so processing used explicit magenta keying and cell-by-cell extraction to keep the grid out of the sprite content.
- Processed v1 frame packet uses one global scale across all 12 frames.
- Runtime starter catalog now includes paperclip-basic in stable order after archive-cube-basic.
- paperclip-basic uses the approved v1 package-backed PNG frame packet, required state loops, and reaction.success.
- Documentation now records the Paperclip review packet and recreation walkthrough.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python Paperclip frame validation script -> validated 12 transparent 96x96 frames plus 384x512 review sheet.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -k 'basic_starter_packs_use_reviewed_multi_frame_state_assets or basic_starter_packs_expose_design_specific_recreation_guidance' -> 12 passed, 64 deselected, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -> 76 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q -> 63 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python package resource/image validation script -> validated search-lens-basic, index-card-basic, archive-cube-basic, and paperclip-basic package resources and manifests.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_starter_fixtures.py -> passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_starter_fixtures.py -f json -o /tmp/bandit_basic_buddy_defaults.json -> 0 findings.
- git diff --check -> passed.
Codex pet compatibility checkpoint:
- Import preview now accepts Codex/Petdex `.zip` packages containing `pet.json` or `petjson.json` plus a 1536x1872 PNG/WebP spritesheet.
- Preview translates the Codex 8x9 atlas into the existing `sprite_frames` Persona Visual manifest shape with one `sprite_sheet` / `animation_atlas` asset.
- Commit imports the atlas through the existing Persona Visual storage service and leaves the pack as an inactive draft.
- Codex `running-right` and `running-left` rows map to tldw custom movement states `moving_right` and `moving_left`.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q -k 'codex_pet or petdex' -> 3 passed, 15 deselected, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q -k 'codex_pet_zip' -> 1 passed, 63 deselected, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/core/Persona/visual_starter_fixtures.py -> passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q -> 18 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q -> 64 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -> 76 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_starter_fixtures.py tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/api/v1/endpoints/persona.py -f json -o /tmp/bandit_basic_buddy_defaults.json -> 0 findings.
- git diff --check -> passed.
Hatch-pet reference checkpoint:
- Persona Visual docs now treat hatch-pet as the reference workflow for a user-facing tldw Simple Buddy Creator, not as a tldw runtime dependency.
- The documented creator flow has two modes: simple tldw draft-pack mode for name/description/reference/style plus core states, and full Codex-compatible mode for the nine-row `pet.json` plus `spritesheet.webp` atlas.
- The documented Buddy process now uses tldw surfaces for Persona Garden review, Persona Visual draft storage, import-preview diagnostics, optional library reuse, MCP-triggerable custom states, and explicit activation.
- The docs carry over hatch visual QA blockers such as identity drift, clipping, slot overlap, copied guides, nontransparent backgrounds, size popping, wrong facing direction, inert idle loops, and detached effects.
Terminal Tile checkpoint:
- User approved the Terminal Tile neutral anchor and 3x4 simple-state source sheet.
- Source was copied into Docs/Code_Documentation/assets/buddy-defaults/terminal-tile-basic/source/ and chroma-keyed to alpha.
- Processed v1 frame packet uses one stable global scale across all 12 frames, with tiny detached extraction specks removed before review.
- Runtime starter catalog now includes terminal-tile-basic in stable order after paperclip-basic, replacing the old minimal-helper-basic basic slot.
- terminal-tile-basic uses the approved v1 package-backed PNG frame packet, required state loops, and reaction.success.
- Documentation now records the Terminal Tile review packet and recreation walkthrough.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -k 'lists_bundled_packs or production_guidance' -> failed before implementation, confirming terminal-tile-basic was not yet the sixth basic pack.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -> 76 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q -> 18 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q -> 64 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_starter_fixtures.py tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/api/v1/endpoints/persona.py -> passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python package resource/image validation script -> validated search-lens-basic, index-card-basic, archive-cube-basic, paperclip-basic, and terminal-tile-basic package resources and manifests.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_starter_fixtures.py tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/api/v1/endpoints/persona.py -f json -o /tmp/bandit_basic_buddy_defaults.json -> 0 findings, 0 errors.
- git diff --check -> passed.
Migu Marker checkpoint:
- User approved the Migu neutral anchor and revised 3x4 simple-state source sheet with earpiece headset, mic, and black center shirt split.
- Source was copied into Docs/Code_Documentation/assets/buddy-defaults/migu-marker-basic/source/ and chroma-keyed to alpha.
- Processed v1 frame packet uses one stable global scale across all 12 frames, with detached underline artifacts removed before review.
- Runtime starter catalog now uses migu-marker-basic as the sixth reviewed basic starter after terminal-tile-basic.
- migu-marker-basic uses the approved v1 package-backed PNG frame packet, required state loops, and reaction.success.
- Documentation now records the Migu review packet and recreation walkthrough.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -k 'basic_starter_packs_use_reviewed_multi_frame_state_assets or copy_every_default_scaffold' -> 18 passed, 58 deselected, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -> 76 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q -> 18 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q -> 64 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_starter_fixtures.py tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/api/v1/endpoints/persona.py -> passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python package resource/image validation script -> validated search-lens-basic, index-card-basic, archive-cube-basic, paperclip-basic, terminal-tile-basic, and migu-marker-basic package resources and manifests.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_starter_fixtures.py tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/api/v1/endpoints/persona.py -f json -o /tmp/bandit_basic_buddy_defaults.json -> 0 findings, 0 errors.
PR review follow-up checkpoint:
- Verified Qodo's nine-pack comment against the current staged effort. It is not a code change for this basic-tier PR: the nine-pack epic remains split across basic, intermediate, and intricate asset-production issues, and this task/PR intentionally completes the basic tier first.
- Wrapped the long `_png_chunk()` return expression back to PEP 8-friendly multiline form.
- Updated duplicate-pack copying so Persona Visual duplicate-to-persona preserves all source-pack asset rows, including non-manifest `preview`, `still_pose`, and generated-candidate assets, while still remapping only manifest references.
- Added regression coverage that duplicate-to-persona preserves unreferenced non-manifest assets.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py::test_duplicate_pack_to_persona_preserves_all_pack_assets_and_remaps_manifest -q -> failed before implementation with 2 copied assets instead of 4, then passed after the fix.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py -q -> 14 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q -> 76 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q -> 18 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q -> 64 passed, 5 warnings.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/Persona/visual_service.py tldw_Server_API/app/core/Persona/visual_starter_fixtures.py tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/api/v1/endpoints/persona.py -> passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python package resource/image validation script -> validated search-lens-basic, index-card-basic, archive-cube-basic, paperclip-basic, terminal-tile-basic, and migu-marker-basic package resources and manifests.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_service.py tldw_Server_API/app/core/Persona/visual_starter_fixtures.py tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/api/v1/endpoints/persona.py -f json -o /tmp/bandit_basic_buddy_defaults.json -> 0 findings, 0 errors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
