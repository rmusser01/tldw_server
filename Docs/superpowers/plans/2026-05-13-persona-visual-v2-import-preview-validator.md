# Persona Visual V2 Import Preview Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backend renderer-specific import-preview validation seam for Manifest V2 Persona Visual packs without enabling archive parsing, commit, or runtime renderer support.

**Architecture:** Keep the existing V1 portability preview path unchanged. Add a pure, fixture-driven validator module that takes already-normalized manifest/assets metadata, resolves the renderer capability registry, and returns structured preview diagnostics for blocked or unsupported V2 renderers. This gives later archive-parser and UI slices a stable contract without writing assets or activating packs.

**Tech Stack:** Python dataclasses, existing Persona visual renderer capability registry, pytest, Bandit.

---

## Stage 1: Fixture Preview Contract
**Goal**: Define the minimal import-preview result contract for renderer-specific Manifest V2 validation.
**Success Criteria**: Tests can call the validator directly with fixture manifests/assets and receive deterministic statuses, blockers, warnings, normalized role categories, and activation eligibility.
**Tests**: Focused pytest coverage in `tldw_Server_API/tests/Persona/test_persona_visual_import_preview_validators.py`.
**Status**: Complete

### Task 1: Validator Interface and Result Model
**Files:**
- Create: `tldw_Server_API/app/core/Persona/visual_import_preview_validators.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_import_preview_validators.py`

- [x] Write failing tests for known disabled `live2d`, unknown renderer, missing required fallback/source categories, and non-activatable preview diagnostics.
- [x] Run focused tests and verify they fail because the interface does not exist.
- [x] Add dataclasses for preview assets and preview results.
- [x] Add `preview_renderer_import()` that uses the renderer capability registry and returns deterministic diagnostics without parsing archives or writing assets.
- [x] Re-run focused tests and verify green.

## Stage 2: V1 Boundary and Docs
**Goal**: Preserve existing V1 import behavior while documenting the new fixture-only V2 seam.
**Success Criteria**: Existing portability preview tests still pass; docs state this is not archive parsing, runtime renderer support, or activation support.
**Tests**: Existing persona visual portability tests plus focused validator tests.
**Status**: Complete

### Task 2: Boundary Verification and Documentation
**Files:**
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
- Modify: `Docs/Design/2026-05-13-persona-visual-manifest-v2-contract.md`
- Modify: `backlog/tasks/task-317 - Implement-Persona-Visual-Manifest-V2-import-preview-validator-interface.md` via Backlog tooling

- [x] Update docs with the fixture-only validator boundary.
- [x] Run focused validator and portability tests.
- [x] Run `git diff --check`.
- [x] Run Bandit on touched backend files.
- [x] Update TASK-317 with verification and final summary.
