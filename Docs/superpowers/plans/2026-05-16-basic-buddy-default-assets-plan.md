# Basic Buddy Default Assets Implementation Plan

## Stage 1: Lock Basic Tier Readiness Contract
**Goal**: Update focused tests so approved basic bundled Buddy packs expose art-ready metadata and each newly accepted default is added only after review.
**Success Criteria**: Tests assert approved basic packs expose reviewed production metadata, multi-frame required-state loops, neutral anchor/preview assets, and inactive draft copy semantics.
**Tests**: `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q`
**Status**: Complete

## Stage 2: Replace Basic Scaffold Art
**Goal**: Replace the basic tier's placeholder fixtures one approved Buddy at a time with Codex-pet-compatible atlas packets derived from one neutral-anchor identity per pack.
**Success Criteria**: `search-lens-basic`, `index-card-basic`, `archive-cube-basic`, `paperclip-basic`, `terminal-tile-basic`, and `migu-marker-basic` each provide reviewed required-state loops plus neutral/preview assets before the six-pack basic tier is complete. The six basic defaults are the current basic tier. The merged 96x96 frame packets are accepted tldw runtime assets for the basic slice, while the final cross-app/interchange target is the Codex/Petdex-compatible 8x9 atlas contract when a default or user-created Buddy needs Codex Buddy parity. The creation process follows the hatch-style bar retuned for tldw: canonical neutral anchor, state rows/frames, deterministic assembly, contact-sheet review, motion-preview review when animated, inactive draft import, and explicit activation. The already approved 3x4 packets remain review/concept evidence for any later atlas upgrade instead of defining a separate lower-tier contract.
**Tests**:
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q`
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q -k 'codex_pet'`
**Status**: Complete

## Stage 3: Documentation And Review Evidence
**Goal**: Document approved basic defaults, Codex-pet import compatibility, and review evidence for each accepted source sheet and processed frame packet.
**Success Criteria**: Persona Visual docs distinguish bundled basic art-ready defaults from unfinished starters, preserve draft-first activation semantics, define the `moving_right` / `moving_left` movement-state translation for Codex pet running rows, and record hatch-pet as the reference workflow for a tldw-native Buddy creation flow whose simple UX path still converges on the same Persona Visual draft/review contract and can produce or import full Codex-compatible packets.
**Tests**: `git diff --check`
**Status**: Complete

## Stage 4: Verification And Tracker Update
**Goal**: Run focused backend verification and record results in Backlog/GitHub.
**Success Criteria**: Focused pytest, py_compile, Bandit for touched Python, and whitespace checks pass or have documented skips.
**Tests**:
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q`
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q`
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q`
- `python -m py_compile tldw_Server_API/app/core/Persona/visual_starter_fixtures.py`
- `python -m py_compile tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/api/v1/endpoints/persona.py`
- `python -m bandit -r tldw_Server_API/app/core/Persona/visual_starter_fixtures.py tldw_Server_API/app/core/Persona/visual_portability/codex_pet.py tldw_Server_API/app/core/Persona/visual_portability/preview.py tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/api/v1/endpoints/persona.py -f json -o /tmp/bandit_basic_buddy_defaults.json`
- `git diff --check`
**Status**: Complete
