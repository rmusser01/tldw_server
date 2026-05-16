# Prototype Workspace Risk Gate 4 Contract Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the prototype workspace API contract, error semantics, docs, and frontend fixtures needed by Risk Gates 5 through 8.

**Architecture:** Keep the contract narrow and prototype-specific. Add typed prototype error detail schemas, use centralized helpers in the prototype endpoint paths, document response models through OpenAPI `responses`, and update the existing contract matrix plus fixture artifact so frontend work consumes stable categories instead of human-only detail strings.

**Tech Stack:** FastAPI, Pydantic v2, pytest, existing prototype workspace repositories/services, Markdown API docs, WebUI E2E fixture JSON.

---

## Stage 1: Contract Inventory And Baseline

**Goal:** Confirm the current backend behavior and record the clean baseline before changing API semantics.

**Success Criteria:** Focused prototype tests pass from the Risk Gate 4 worktree, and TASK-399 records the worktree, baseline, and plan path.

**Tests:** `python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q`

**Status:** Complete

- [x] Create isolated worktree `.worktrees/prototype-risk-gate-4-contract-freeze` from `origin/dev`.
- [x] Create Backlog task TASK-399 linked to GitHub issue #1456.
- [x] Run the focused PrototypeWorkspaces suite and record the passing baseline.

## Stage 2: Error Contract Schema And Endpoint Helpers

**Goal:** Freeze machine-readable prototype error details without broad API cleanup.

**Success Criteria:** Prototype-only error responses have stable `category`, `message`, `frontend_state`, and `retryable` fields while preserving appropriate HTTP statuses.

**Tests:** Add failing endpoint tests that assert structured details for inactive session, archived workspace, missing workspace, unauthorized owner access, bootstrap failure, preview unavailable, stale promotion, conflict, invalid password, and password-required paths.

**Status:** Complete

- [x] Add `PrototypeErrorDetail` and `PrototypeErrorResponse` to `tldw_Server_API/app/api/v1/schemas/prototype_workspace_schemas.py`.
- [x] Add a local prototype error helper in `tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py`.
- [x] Convert expected prototype endpoint failures to the structured detail helper.
- [x] Convert the prototype share-link exchange path in `tldw_Server_API/app/api/v1/endpoints/sharing.py` to the same structured detail shape for prototype-specific failures.
- [x] Keep validation errors and unrelated sharing endpoints on existing FastAPI behavior.

## Stage 3: OpenAPI Response Contract

**Goal:** Make generated OpenAPI describe the frozen prototype error contract.

**Success Criteria:** Prototype owner, collaborator, promotion, preview-renewal, and public exchange routes include documented error responses that reference `PrototypeErrorResponse`.

**Tests:** Add an OpenAPI regression test that inspects `app.openapi()` for prototype route response models and key category examples.

**Status:** Complete

- [x] Define reusable prototype error response metadata near the endpoint helpers.
- [x] Add `responses` metadata to prototype workspace endpoint decorators.
- [x] Add `responses` metadata to `POST /api/v1/sharing/public/{token}/prototype-session`.
- [x] Verify no unrelated route metadata changes are introduced.

## Stage 4: Contract Matrix, Lifecycle Examples, And Fixtures

**Goal:** Freeze the human-facing contract artifacts that frontend implementation will build against.

**Success Criteria:** The contract matrix has no Risk Gate 4 open decisions, the WebUI fixture matches backend error details, and API docs include lifecycle, configuration, migration, and rollback notes.

**Tests:** Add lightweight fixture/docs consistency checks where practical, plus `git diff --check`.

**Status:** Complete

- [x] Update `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md` from draft to frozen Risk Gate 4 status.
- [x] Finalize every matrix state with HTTP status, stable category, frontend bucket, retryability, and handling.
- [x] Update `apps/tldw-frontend/e2e/fixtures/prototype-workspaces/contract-states.json` mock responses to the structured error detail shape.
- [x] Expand `Docs/API-related/Prototype_Workspaces_API.md` with lifecycle examples, configuration requirements, and migration/rollback notes.

## Stage 5: Verification And Closeout

**Goal:** Prove Risk Gate 4 is ready for review and update the trackers.

**Success Criteria:** Focused backend tests pass, fixture/docs checks pass, Bandit is clean for touched backend paths, TASK-399 is updated, and a PR is opened against `dev` linked to issue #1456.

**Tests:** `python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q`, targeted sharing/prototype contract tests, Bandit on touched backend files, and `git diff --check`.

**Status:** Complete

- [x] Run focused backend tests.
- [x] Run Bandit on touched backend endpoint/schema paths.
- [x] Run `git diff --check`.
- [x] Update TASK-399 acceptance criteria and notes.
- [x] Open a PR linked to GitHub issue #1456.
