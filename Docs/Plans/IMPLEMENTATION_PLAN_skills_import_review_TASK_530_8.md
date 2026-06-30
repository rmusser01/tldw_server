# Skills Import Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only Skills import review flow so text and file imports expose parsed metadata, validation errors, and conflicts before mutating user skills.

**Architecture:** Reuse the existing Skills parser and service validation paths to build a non-mutating preview contract. The API adds preview endpoints beside the existing import endpoints, and the WebUI changes the current import modal/file upload handlers into a review-first workflow that only calls mutating imports after explicit confirmation.

**Tech Stack:** FastAPI, Pydantic, existing `SkillsService`, React, Ant Design, TanStack Query, Vitest, pytest.

---

## Stage 1: Backend Preview Contract

**Goal:** Lock the preview API behavior with failing tests before service changes.

**Success Criteria:** Tests prove preview parses valid imports, reports conflicts without writing files, returns validation errors for invalid content, and handles uploaded files without mutation.

**Tests:**
- `tldw_Server_API/tests/Skills/unit/test_skills_service.py`
- `tldw_Server_API/tests/Skills/integration/test_skills_api.py`

**Status:** Complete

- [x] Add a service unit test for `preview_import_skill` returning parsed metadata with `valid=True`, `conflict=False`, and no created skill directory.
- [x] Add a service unit test for a conflicting existing skill returning `conflict=True`, `can_overwrite=True`, and the existing version without deleting or rewriting the skill.
- [x] Add an integration test for `POST /api/v1/skills/import/preview` returning parsed metadata and conflict state.
- [x] Add an integration test for invalid preview content returning a non-mutating validation response.
- [x] Add an integration test for `POST /api/v1/skills/import/file/preview` returning metadata for a `.md` upload.
- [x] Run the targeted backend tests and confirm they fail because the preview contract does not exist yet.

## Stage 2: Backend Preview Implementation

**Goal:** Implement the minimal read-only service and endpoint paths needed by Stage 1.

**Success Criteria:** Preview endpoints share import parsing and validation semantics, never write skills, and existing import behavior remains unchanged.

**Tests:**
- Targeted unit and integration tests from Stage 1.

**Status:** Complete

- [x] Add Pydantic models for import preview responses in `skills_schemas.py`.
- [x] Add `SkillsService.preview_import_skill(...)` that parses content, normalizes names/supporting files, checks registry conflict state, and returns metadata without calling `create_skill` or `delete_skill`.
- [x] Add `SkillsService.preview_import_from_zip(...)` or an equivalent helper that reuses safe zip extraction and calls the preview method.
- [x] Add `POST /api/v1/skills/import/preview` and `POST /api/v1/skills/import/file/preview` endpoints in `skills.py`.
- [x] Keep malicious zip/path traversal failures as HTTP errors while ordinary content/name validation is returned as review data.
- [x] Run targeted backend tests and confirm they pass.

## Stage 3: Frontend API and UI Contract

**Goal:** Lock the review-first UI behavior with failing tests before changing the manager implementation.

**Success Criteria:** Tests prove text and file imports preview first, display validation/conflict review state, and only mutate after explicit confirmation.

**Tests:**
- `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`

**Status:** Complete

- [x] Extend mocked `tldwClient` with `previewSkillImport` and `previewSkillImportFile`.
- [x] Update the text import test to expect the first primary action to call preview, not `importSkill`.
- [x] Add a conflict text import test that shows the overwrite control only after a conflict preview and sends `overwrite: true` on final import.
- [x] Update the file import test to expect preview first and final import only after confirmation.
- [x] Run the targeted Vitest file and confirm the new expectations fail against current direct-import behavior.

## Stage 4: Frontend Review Implementation

**Goal:** Implement the smallest UI change that makes import review clear and reliable without redesigning unrelated Skills surfaces.

**Success Criteria:** The import modal presents parsed name/description/context/tools/conflict/validation results, disables final import until preview is valid, and preserves existing success actions.

**Tests:**
- Targeted Vitest file from Stage 3.

**Status:** Complete

- [x] Add `SkillImportPreviewResponse` and client methods in the UI type/API layer.
- [x] Add import review state to `Manager.tsx` for text and file flows.
- [x] Change the text modal primary action from direct import to preview when no valid review exists.
- [x] Show review details, validation errors, and conflict copy inside the modal using existing Ant Design components.
- [x] Add a final import confirmation path that sends overwrite only when the user explicitly enables it after conflict review.
- [x] Change file upload to preview first, then open the same review/confirmation path before calling `importSkillFile`.
- [x] Run targeted Vitest tests and confirm they pass.

## Stage 5: Verification, Backlog, and PR

**Goal:** Finish the branch with focused verification, task updates, and a PR against `dev`.

**Success Criteria:** Backend/frontend targeted tests pass, Bandit has no new findings in touched backend code, Backlog task is current, and the PR contains only related files.

**Tests:**
- `python -m pytest tldw_Server_API/tests/Skills/unit/test_skills_service.py tldw_Server_API/tests/Skills/integration/test_skills_api.py -q`
- `bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot`
- `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/schemas/skills_schemas.py tldw_Server_API/app/core/Skills -f json -o /tmp/bandit_skills_import_review_TASK_530_8.json`

**Status:** In Progress

- [x] Run all targeted verification commands from this plan.
- [x] Inspect `git diff` and remove unrelated/generated artifacts.
- [x] Update Backlog `TASK-530.8` with touched files and verification results.
- [x] Commit with a `TASK-530.8` message.
- [x] Push `codex/skills-import-review` and open a PR against `dev`.
