# Persona Visual V2 Capability Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first Manifest V2 renderer capability foundation for Persona/Buddy visual packs without enabling a new runtime renderer.

**Architecture:** Extend the existing backend renderer capability registry as the single source of truth, then serialize additive metadata through the existing `/api/v1/persona/visual-renderers` endpoint. Frontend types and docs mirror the backend contract; `sprite_frames` remains the only activatable Buddy renderer.

**Tech Stack:** Python dataclasses and Pydantic schemas, FastAPI endpoint mapping, TypeScript shared UI types, pytest, Vitest type-adjacent coverage where useful.

---

## Stage 1: Backend Capability Contract
**Goal**: Add additive Manifest V2 fields to the renderer capability registry and API response while preserving existing response fields.
**Success Criteria**: `sprite_frames` still validates and activates; future/non-sprite renderers are represented as non-activatable capability metadata only.
**Tests**: Focused pytest coverage in `tldw_Server_API/tests/Persona/test_persona_visuals_core.py` and `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`.
**Status**: Complete

### Task 1: Registry and API Schema
**Files:**
- Modify: `tldw_Server_API/app/core/Persona/visual_renderer_capabilities.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`

- [x] Write failing tests for additive renderer capability metadata and explicit disabled future renderer state.
- [x] Run the focused pytest tests and verify they fail because fields/renderers are missing.
- [x] Add dataclass fields for contract versions, roles, limits, setup state, setup blockers, and static-fallback/license requirements.
- [x] Populate `sprite_frames` metadata and an explicit disabled `live2d` Manifest V2 capability.
- [x] Serialize the new fields from the API response without removing existing fields.
- [x] Re-run focused pytest tests and verify green.

## Stage 2: Frontend Type Contract
**Goal**: Keep shared UI types aligned with the backend capability response.
**Success Criteria**: TypeScript consumers can read new optional fields without treating disabled renderers as activatable packs.
**Tests**: Focused frontend tests if existing coverage references renderer capabilities; otherwise TypeScript type update only plus backend contract tests.
**Status**: Complete

### Task 2: Shared UI Types
**Files:**
- Modify: `apps/packages/ui/src/types/persona-visuals.ts`

- [x] Add optional capability metadata fields to `PersonaVisualRendererCapability`.
- [x] Keep current required fields unchanged.
- [x] Run focused frontend type/test coverage if available.

## Stage 3: Documentation and Tracker
**Goal**: Document the implemented capability foundation and update work tracking.
**Success Criteria**: Docs explain the new fields and clearly state that Live2D/non-sprite renderers are not runtime-enabled by this slice.
**Tests**: `git diff --check`; Bandit on touched backend code; focused pytest; frontend verification as applicable.
**Status**: Complete

### Task 3: Docs and Finalization
**Files:**
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
- Modify: `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
- Modify: `backlog/tasks/task-316 - Implement-Persona-Visual-Manifest-V2-capability-foundation.md` via Backlog tooling

- [x] Update docs with capability field semantics and disabled/future renderer boundary.
- [x] Update TASK-316 notes and final verification through Backlog tooling.
- [x] Run verification: focused pytest, relevant frontend tests if touched beyond types, `git diff --check`, and Bandit on touched backend files.
- [x] Prepare the branch package for the PR tied to issue #1628.
