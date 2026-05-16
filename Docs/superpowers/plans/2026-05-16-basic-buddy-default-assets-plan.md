# Basic Buddy Default Assets Implementation Plan

## Stage 1: Lock Basic Tier Readiness Contract
**Goal**: Update focused tests so the three basic bundled Buddy packs are the only starter packs expected to be art-ready in this slice.
**Success Criteria**: Tests assert basic packs expose reviewed production metadata, multi-frame required-state loops, neutral anchor/preview assets, and inactive draft copy semantics.
**Tests**: `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q`
**Status**: Complete

## Stage 2: Replace Basic Scaffold Art
**Goal**: Replace the basic tier's 4x4 solid-color fixtures with deterministic transparent PNG frames generated from one neutral-anchor identity per pack.
**Success Criteria**: `research-buddy-basic`, `migu-marker-basic`, and `minimal-helper-basic` each provide `idle`, `listening`, `thinking`, `speaking`, and `error` two-frame loops plus neutral/preview assets.
**Tests**: `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q`
**Status**: Complete

## Stage 3: Documentation And Review Evidence
**Goal**: Document that the basic defaults are bundled by default and art-ready, while additional packs remain optional.
**Success Criteria**: Persona Visual docs distinguish bundled basic art-ready defaults from intermediate/intricate scaffolds and preserve draft-first activation semantics.
**Tests**: `git diff --check`
**Status**: Complete

## Stage 4: Verification And Tracker Update
**Goal**: Run focused backend verification and record results in Backlog/GitHub.
**Success Criteria**: Focused pytest, py_compile, Bandit for touched Python, and whitespace checks pass or have documented skips.
**Tests**:
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q`
- `python -m py_compile tldw_Server_API/app/core/Persona/visual_starter_fixtures.py`
- `python -m bandit -r tldw_Server_API/app/core/Persona/visual_starter_fixtures.py -f json -o /tmp/bandit_basic_buddy_defaults.json`
- `git diff --check`
**Status**: Complete
