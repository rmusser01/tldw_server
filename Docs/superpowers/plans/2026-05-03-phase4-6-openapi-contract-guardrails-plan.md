# Phase 4.6 OpenAPI Contract Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add narrow backend OpenAPI guardrails now that the Phase 3 response-envelope and pagination helpers have landed.

**Architecture:** Treat the generated FastAPI `app.openapi()` document as the backend contract source of truth. This tranche does not migrate endpoints, change payloads, or enable strict frontend OpenAPI drift mode; it adds tests plus a narrow verifier parser fix needed to keep the existing OpenAPI guard runnable.

**Tech Stack:** FastAPI OpenAPI generation, pytest, existing `apps/extension` OpenAPI verifier.

---

### Stage 1: Backend OpenAPI Contract Guardrails

**Goal:** Lock the Phase 3 boundary where shared response-envelope helpers are available but default v1 endpoints have not opted into envelope payloads.

**Success Criteria:** Provider-compatible routes and no-content routes do not expose the shared `ResponseEnvelope` shape in OpenAPI, and canonical auth security scheme names remain stable.

**Tests:** `python -m pytest tldw_Server_API/tests/Utils/test_openapi_phase4_contract.py -q`

**Status:** Complete

- [x] Add `tldw_Server_API/tests/Utils/test_openapi_phase4_contract.py`.
- [x] Generate OpenAPI from `tldw_Server_API.app.main.app` under synthetic single-user auth.
- [x] Assert `ResponseEnvelope` is not exposed as a component until a route opts in.
- [x] Assert OpenAI-compatible chat/audio/embeddings routes are not wrapped in envelope schemas.
- [x] Assert `204` operations do not declare JSON bodies.
- [x] Assert canonical auth security scheme names remain present.

### Stage 2: Existing OpenAPI Guard Verification

**Goal:** Confirm the existing frontend/shared OpenAPI verifier remains compatible with generated backend OpenAPI after PR #1215.

**Success Criteria:** Existing backend contract tests and the JS OpenAPI verifier run without regressions or report only known reviewed exceptions.

**Tests:** `python -m pytest tldw_Server_API/tests/Utils/test_openapi_phase4_contract.py tldw_Server_API/tests/Utils/test_pagination_openapi_contract.py tldw_Server_API/tests/Utils/test_response_envelope.py -q`; `bun run verify:openapi` from `apps/packages/ui` when dependencies are available.

**Status:** Complete

- [x] Run the focused backend tests.
- [x] Run the existing pagination/envelope contract tests.
- [x] Run the existing frontend OpenAPI verifier or document why it cannot run locally.

### Stage 3: Hygiene And Tracker Update

**Goal:** Leave the branch reviewable and keep issue #1116 current.

**Success Criteria:** `git diff --check` passes, status is understood, and issue #1116 is updated with current Phase 4.6 status if a commit is made.

**Tests:** `git diff --check`; `git status --short --branch`

**Status:** In Progress

- [x] Run whitespace/status hygiene.
- [ ] Commit the tranche if verification is clean.
- [ ] Update issue #1116 with the Phase 4.6 guardrail status.
