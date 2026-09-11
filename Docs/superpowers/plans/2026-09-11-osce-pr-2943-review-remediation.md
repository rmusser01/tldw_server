# OSCE PR 2943 Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every actionable Qodo finding and CI regression on PR #2943, document evidence for non-actionable findings, and merge the verified branch into `dev`.

**Architecture:** Preserve the existing OSCE service and persistence boundaries while tightening API contracts, source compatibility, optimistic-concurrency behavior, and completed-attempt immutability. Keep legacy quiz consumers safe by making activity selection explicit at the API/client boundary, and keep frontend exports bounded without adding dependencies.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL, pytest, Next.js/React, TypeScript, TanStack Query, Vitest, GitHub Actions.

**Spec:** `Docs/superpowers/specs/2026-09-10-osce-scenario-practice-design.md`

## Global Constraints

- Keep OSCE marking guides hidden until self-assessment and retain no-score semantics.
- Preserve tenant scoping and optimistic concurrency on all station and attempt writes.
- Do not add OSCE support to MCP in this release; its advertised schema must match that limitation.
- Keep WebUI and extension behavior aligned through the shared UI package.
- Add no new runtime dependency.

---

### Task 1: Reproduce And Fix Correctness Findings

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/quizzes.py`
- Modify: `tldw_Server_API/app/core/Quizzes/osce_generator.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tools/quiz_tools.py`
- Modify: `apps/packages/ui/src/components/Quiz/osce/osceDraftStore.ts`
- Modify: `apps/packages/ui/src/components/Quiz/osce/OscePracticePanel.tsx`
- Test: existing OSCE backend, MCP, draft-store, and practice-panel test modules

**Interfaces:**
- Consumes: shared quiz source resolver types and OSCE attempt `version` tokens.
- Produces: complete source-type acceptance, truthful MCP schema, conflict-safe drafts, and immutable completed-attempt controls.

- [x] Add regression tests for every high-severity Qodo scenario and run each test to confirm the current failure.
- [x] Extend OSCE citation/source validation to the source types already accepted by shared quiz generation.
- [x] Remove unsupported OSCE activity advertising from the MCP quiz tool schema.
- [x] Preserve dirty local text across conflict refetches until it is explicitly saved or discarded.
- [x] Disable notes and self-assessment editing after attempt completion and discard stale completed drafts.
- [x] Run the focused backend and frontend tests until green.

### Task 2: Restore Client Compatibility And Frontend Test Isolation

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/quizzes.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `apps/packages/ui/src/services/quizzes.ts`
- Modify: `apps/packages/ui/src/components/Quiz/tabs/ManageTab.tsx`
- Modify: affected quiz tests and fixtures

**Interfaces:**
- Consumes: quiz `activity_type` discriminator.
- Produces: a legacy-safe default list contract plus an explicit all-activities request for current shared clients.

- [x] Add API and client tests proving legacy callers receive question quizzes while the shared UI explicitly requests all supported activities.
- [x] Add the activity filter to the list endpoint and persistence query using parameterized DB helpers.
- [x] Update the shared WebUI/extension service to request all activity types where OSCE cards are supported.
- [x] Remove accidental QueryClient requirements from legacy component test paths or provide the established shared test wrapper.
- [x] Stabilize the definitive station-rejection retry test around observable settled state.
- [x] Run all affected quiz component suites with the CI timeout settings.

### Task 3: Address Contract And Quality Findings

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/quizzes_osce.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/quizzes_osce.py`
- Modify: `tldw_Server_API/app/core/exceptions.py`
- Modify: `tldw_Server_API/app/core/Quizzes/osce_practice.py`
- Modify: `.github/workflows/ci.yml`
- Modify: OSCE tests and fixtures identified by Qodo

**Interfaces:**
- Consumes: central exception conventions and public Pydantic schema module.
- Produces: reusable endpoint contracts, project-standard exceptions/docstrings/typing/markers, bounded export requests, and complete CI shard coverage.

- [x] Move public request/page contracts to the OSCE schema module and add concise module/class documentation.
- [x] Move OSCE domain exceptions to the central exception module and update imports/tests.
- [x] Add missing return annotations and pytest markers in touched OSCE test support.
- [x] Bound OSCE detail fetch concurrency in JSON export and test that the limit is respected.
- [x] Add both OSCE database test modules to the appropriate backend CI shard.
- [x] Validate the PostgreSQL fixture claim; change it only if it bypasses the canonical isolated environment.
- [x] Compare endpoint orchestration with three neighboring quiz endpoints and refactor only if the finding identifies a real ownership violation.

### Task 4: Regenerate Contracts And Verify The Pull Request

**Files:**
- Modify: `apps/tldw-frontend/lib/api/openapi.fingerprint.json`
- Modify: `backlog/tasks/task-12102.3.5.7 - OSCE-PR-2943-review-and-CI-remediation.md`
- Remove: `Docs/superpowers/plans/2026-09-11-osce-pr-2943-review-remediation.md` after all stages pass

**Interfaces:**
- Consumes: final FastAPI OpenAPI document and touched-file inventory.
- Produces: synchronized contract fingerprint, review evidence, and merge-ready PR state.

- [x] Regenerate the checked-in OpenAPI fingerprint using the repository command and verify the contract gate.
- [x] Run focused backend/frontend tests, TypeScript checks, `git diff --check`, and Bandit on touched Python paths.
- [x] Review the complete diff for regressions and update the Backlog task with commands and results.
- [ ] Commit and push the remediation, then reply to and resolve every Qodo thread with specific evidence.
- [ ] Wait for required checks, fix any branch-caused failure, and merge PR #2943 only when all merge gates pass.
