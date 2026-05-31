# Persona Visual Starter Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bundled Persona Visual starter packs that can be listed by the backend and copied into user-owned inactive draft packs for a selected persona.

**Architecture:** Keep bundled starter packs as immutable server fixtures, not shared mutable pack rows. A new backend service validates fixture manifests/assets and creates normal user-owned draft pack plus asset rows through the existing `PersonaVisualService` storage and validation paths. REST endpoints expose catalog list/detail and copy-to-draft actions beside the existing Persona Visual pack API.

**Tech Stack:** FastAPI, Pydantic, SQLite-backed `CharactersRAGDB`, existing Persona Visual service/storage, pytest.

---

## File Structure

- Create: `tldw_Server_API/app/core/Persona/visual_starter_catalog.py`
  - Loads bundled fixture definitions, validates catalog metadata, validates sprite-frame manifests against copied assets, and copies one starter pack to a target persona as a normal draft.
- Create: `tldw_Server_API/app/core/Persona/visual_starter_fixtures.py`
  - Contains one small built-in sprite_frames starter fixture with embedded PNG bytes and manifest builder metadata.
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
  - Adds catalog response and copy request/response schemas.
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
  - Adds service dependency and REST endpoints under `/visual-starter-packs`.
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
  - Documents that bundled starters are copied into user-owned draft storage and are never activated automatically.
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py`
  - Unit tests for service behavior, validation, and malformed fixture rejection.
- Test: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`
  - API tests for list and copy endpoints.

---

## Stage 1: Service Contract And Fixture Validation

**Goal:** Introduce a fixture-backed catalog service without API wiring.

**Success Criteria:** The service lists the bundled starter pack, rejects malformed fixture data, and copies a fixture into a user-owned draft using existing asset storage and manifest validation.

**Tests:**
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py -q`

- [x] Write failing tests in `test_persona_visual_starter_catalog.py` for listing the default fixture and copying it to a target persona.
- [x] Run the focused test and confirm it fails because `visual_starter_catalog` is missing.
- [x] Create `visual_starter_fixtures.py` with a tiny embedded PNG starter fixture and manifest builder metadata.
- [x] Create `visual_starter_catalog.py` with a documented service, stable error codes, list/get/copy methods, and explicit malformed-fixture validation.
- [x] Run the focused service tests and keep them passing.

## Stage 2: REST Schemas And Endpoints

**Goal:** Expose catalog list/detail and copy-to-draft actions through the existing Persona API.

**Success Criteria:** Authenticated users can list bundled starters and copy a starter into one of their personas as a draft; cross-user persona targets and unknown starter IDs fail with stable HTTP errors.

**Tests:**
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q`

- [x] Add Pydantic schemas for starter pack summaries, detail responses, and copy requests.
- [x] Add `get_persona_visual_starter_catalog_service()` in `endpoints/persona.py`.
- [x] Add `GET /api/v1/persona/visual-starter-packs`.
- [x] Add `GET /api/v1/persona/visual-starter-packs/{starter_pack_id}`.
- [x] Add `POST /api/v1/persona/visual-starter-packs/{starter_pack_id}/copy`.
- [x] Add API tests proving copy creates a draft and preserves the target persona's active pack.

## Stage 3: Docs, Tracker, And Verification

**Goal:** Document semantics and package the PR-ready branch.

**Success Criteria:** Docs explain copy semantics, Backlog acceptance criteria are checked, and focused tests plus Bandit pass for touched Python scope.

**Tests:**
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q`
- `python -m py_compile tldw_Server_API/app/core/Persona/visual_starter_catalog.py tldw_Server_API/app/core/Persona/visual_starter_fixtures.py`
- `python -m bandit -r tldw_Server_API/app/core/Persona/visual_starter_catalog.py tldw_Server_API/app/core/Persona/visual_starter_fixtures.py tldw_Server_API/app/api/v1/endpoints/persona.py -f json -o /tmp/bandit_persona_visual_starter_catalog.json`
- `git diff --check`

- [x] Update `Docs/Code_Documentation/Persona_Visual_Packs.md`.
- [x] Update Backlog task notes and check completed acceptance criteria.
- [x] Run focused pytest, py_compile, Bandit, and diff whitespace checks.
- [x] Commit the implementation with issue `#1694` references.
- [x] Open a draft PR against `dev` and link `#1694` plus epic `#1510`.

---

## Boundaries

- No Live2D runtime adapter.
- No marketplace/shared public library.
- No external MCP provider execution.
- No VN/CYOA behavior.
- No auto-activation.
- No global mutable pack rows or reference-backed starter rows; starters are copied into user-owned draft storage.
