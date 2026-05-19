# Persona Visual Import Conflict Choices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit Persona/Buddy visual-pack import conflict choices while preserving review-before-commit and separate activation.

**Architecture:** Extend the existing job-backed Persona Visual portability pipeline. Preview will report target-persona pack conflicts and allowed choices; commit will require an explicit target choice when conflicts are present and will only create or replace draft-like packs, never active packs.

**Tech Stack:** FastAPI/Pydantic, ChaChaNotes SQLite persona store, Persona Visual portability Jobs, React/AntD Persona Garden, Vitest, pytest.

---

### Task 1: Backend Preview Conflict Contract

**Files:**
- Modify: `tldw_Server_API/app/core/Persona/visual_portability/preview.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_jobs_worker.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_portability.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py`

- [x] Add a failing previewer test where the target persona already has an active pack and a draft pack with the same incoming pack title; expect conflict entries, allowed choices, and required commit choice metadata.
- [x] Run the focused pytest test and confirm it fails because preview conflicts are empty.
- [x] Implement preview conflict building from target-persona pack summaries supplied by the worker, including stable `conflict_id`, `type`, `severity`, `message`, `pack_id`, `pack_title`, `pack_status`, and `allowed_choices`.
- [x] Update the preview worker to load same-user target persona packs and pass them into the previewer.
- [x] Run the focused backend preview/worker tests and confirm they pass.

### Task 2: Backend Commit Choice Enforcement

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_jobs.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_portability/importer.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_portability.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_portability_worker.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_jobs.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`

- [x] Add failing tests proving a commit with preview conflicts is rejected unless the request includes an explicit conflict choice.
- [x] Add failing tests for `replace_draft` with a target draft pack id and title override.
- [x] Extend commit request/payload schema with `target_mode: create_new | replace_draft`, optional `title`, and optional `target_pack_id`.
- [x] Enforce that `replace_draft` can only target a same-user pack on the target persona with a non-active status; reject active packs and missing packs.
- [x] Keep imported packs draft-only and leave active packs unchanged.
- [x] Run focused backend tests and confirm they pass.

### Task 3: Persona Garden Conflict UX

**Files:**
- Modify: `apps/packages/ui/src/types/persona-visuals.ts`
- Modify: `apps/packages/ui/src/services/persona-visuals.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Test: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [x] Add a Vitest case where preview conflicts show actionable import choices and disable commit until the user chooses a conflict policy.
- [x] Add a Vitest case where choosing `replace draft` posts `target_mode`, `target_pack_id`, and optional title to the commit endpoint.
- [x] Add typed conflict/choice metadata to the frontend Persona Visual models.
- [x] Update the import panel to show conflicts as Persona/Buddy visual-pack choices, with `create new draft` and safe `replace draft` controls.
- [x] Keep copy clear that import commit creates or replaces a draft and activation remains separate.
- [x] Run the focused VisualPackEditor Vitest suite and confirm it passes.

### Task 4: Docs And Verification

**Files:**
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
- Modify: `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
- Modify: `backlog/tasks/task-208 - Implement-Persona-visual-pack-import-conflict-choices.md`

- [x] Update code docs and PRD portability notes with the V1 conflict choices and non-goals.
- [ ] Run `git diff --check`.
- [ ] Run focused pytest suites.
- [ ] Run focused Vitest suite.
- [ ] Run Bandit on touched backend production files.
- [ ] Update TASK-208 acceptance criteria, DoD, verification notes, and final summary.
