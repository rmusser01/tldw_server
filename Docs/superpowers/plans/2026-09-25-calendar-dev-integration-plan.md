# Calendar Dev Integration Plan

Backlog task: TASK-13356
Source: `codex/calendar-verification-followup` at `964beb522e`
Target: `origin/dev`

## Stage 1: Inventory
**Goal**: Identify the Calendar-only source commits and current dev contracts.
**Success Criteria**: No unrelated source commits or files are planned for the PR.
**Tests**: Inspect commit range, file list, and merge base.
**Status**: Complete

## Stage 2: Port
**Goal**: Apply the Calendar module and required integration changes to a fresh dev branch.
**Success Criteria**: Calendar API, jobs, WebUI, docs, and tests are present without unrelated historical changes.
**Tests**: Import and route checks; `git diff --check`.
**Status**: Complete

## Stage 3: Verify
**Goal**: Validate Calendar behavior and security on the current dev baseline.
**Success Criteria**: Focused backend/frontend tests pass, Bandit has no new findings, and any test limits are recorded.
**Tests**: Calendar pytest suite, Calendar Vitest suite, route build/smoke, Bandit on touched Python.
**Status**: Complete

## Stage 4: Publish
**Goal**: Push a reviewable Calendar-only PR against `dev`.
**Success Criteria**: PR targets `dev`, contains verification and manual-smoke limits, and is attached to the task.
**Tests**: Inspect PR base/head and changed-file summary.
**Status**: Complete

Draft PR: https://github.com/rmusser01/tldw_server/pull/3019 (`dev` <- `codex/calendar-dev-pr`)

## Verification Notes

- Calendar backend, AuthNZ seed/startup, and lifecycle catalog: 142 passed, 2 unrelated stale migration-version assertions deselected.
- Calendar frontend: 30 passed; web navigation locale resolution, first-calendar creation, and delete confirmation covered.
- Frontend TypeScript, targeted ESLint, and Bandit: passed (Bandit: 0 findings).
- Next.js `/calendar` route compiled and returned HTTP 200 in local smoke test.
- Broader router contract tests remain limited by unrelated existing optional-router imports (`media.ingest_jobs` and prompt-studio websocket); Calendar registration tests pass.
- A bounded CalDAV REPORT cannot prove remote deletion from omission, so inferred tombstoning is deferred until complete delta-sync support.
- External CalDAV provider smoke was not run; see `Docs/Development/Calendar_CalDAV_Smoke_Test.md`.
