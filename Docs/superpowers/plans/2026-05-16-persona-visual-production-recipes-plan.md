# Persona Visual Starter Production Recipes Plan

## Stage 1: Contract Tests
**Goal**: Define the backend/API recipe contract from the caller perspective.
**Success Criteria**: Focused tests require every starter to expose a bounded production recipe that distinguishes neutral anchors, static sheets, animation outputs, and review checks.
**Tests**: `test_persona_visual_starter_catalog.py` and `test_persona_visuals_api.py` fail before implementation.
**Status**: Complete

## Stage 2: Fixture and Service Metadata
**Goal**: Add deterministic recipe metadata to immutable starter fixtures and service summaries/details without changing copy-to-draft behavior.
**Success Criteria**: All nine starters carry tier-appropriate recipes; malformed recipe values fail fixture validation.
**Tests**: Focused starter catalog tests pass.
**Status**: Complete

## Stage 3: API Schema and Documentation
**Goal**: Expose recipes through the typed Persona starter catalog response and explain the authored-asset handoff in docs.
**Success Criteria**: API tests see recipe fields; docs clarify recipes are production handoff metadata, not finished animation packs.
**Tests**: Focused Persona Visual API tests pass.
**Status**: Complete

## Stage 4: Verification and PR
**Goal**: Validate, commit, push, and open a reviewable PR against `dev`.
**Success Criteria**: Focused tests, `git diff --check`, and Bandit on touched backend code pass or have documented skips.
**Tests**: Focused pytest suite, Bandit, diff checks.
**Status**: Complete
