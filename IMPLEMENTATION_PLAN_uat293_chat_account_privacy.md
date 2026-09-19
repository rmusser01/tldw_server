# UAT293 — Chat account privacy

Task: TASK-13260.230. Native frozen-matrix evidence confirms that Back restores a prior account's Chat transcript and that an unowned draft survives reload. Backend foreign reads correctly deny access.

## Design

Clear private live Chat state synchronously when the existing auth/config/storage lifecycle reports a different authority, including when Chat is unmounted. Keep model preferences. Give composer drafts an account/server/auth-source/organization namespace, reject legacy unowned drafts, and cancel stale hydration and persistence at an account boundary. Keep owned reload recovery and same-principal token refresh. Clear visible cached messages when a canonical saved-chat read is denied or missing.

## Stage 1: Causal regressions
**Goal**: Reproduce actual boundary and persistence failures without manually clearing stores.
**Success Criteria**: Failing tests for cross-account live state and draft recovery, plus positive owned recovery/refresh controls.
**Tests**: Auth/config/storage lifecycle, reciprocal switches, delayed reads and saves, canonical denied restore.
**Status**: Complete

## Stage 2: Minimal repair and focused validation
**Goal**: Implement account-bound state and draft cleanup using existing auth and storage conventions.
**Success Criteria**: Focused regressions pass; independent review addressed; no new touched-scope lint/type/security findings.
**Tests**: Composer, persistence, account boundary, session restoration, and server chat loader suites. Bandit N/A if changes remain TypeScript-only.
**Status**: Complete

Validation: 231 tests in 11 files pass (`/tmp/uat293-final-focused-tests.txt`). ESLint: zero errors and the same ten warnings as HEAD. Frontend typecheck: 93 identical pre-existing diagnostics, no added/removed errors. Independent bounded review of draft ownership, persistence, composer integration and boundary observation found no actionable production privacy issue; self-review covered saved-chat and session-save changes. Bandit is not applicable to this TypeScript-only scope. Generated browser artifacts remain ignored.

## Stage 3: Native acceptance and tracker
**Goal**: Verify reciprocal account switches in SQLite and real PostgreSQL.
**Success Criteria**: Back/Forward/reload and active tabs show no prior-account transcript/title/draft, own recovery works, backend denials stay intact. Commit evidence summaries; keep generated captures ignored.
**Tests**: Native browser workflows using normal login/logout and official PostgreSQL fixtures.
**Status**: In Progress

SQLite targeted acceptance uses the changed frontend with the retained isolated SQLite multi-user backend. Alice/Bob owned transcript/draft and reload controls pass. Reciprocal logout, Back, Forward and reload pass in both Chat tabs. Matching-workspace reciprocal draft recovery passes at05:28:17UTC. A separate early-login workspace-initialization defect is tracked as UAT300/TASK13260.237; its org-null draft remains correctly separate from org2. An additional organization-isolation regression passes with all six existing owner tests. Real PostgreSQL acceptance is still pending; UAT293 remains open. The first multi-tab automation process exited before acceptance; its incomplete probe is explicitly excluded.
