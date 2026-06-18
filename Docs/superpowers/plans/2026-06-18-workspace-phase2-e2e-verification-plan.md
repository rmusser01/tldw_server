# Workspace Phase 2 E2E Verification Plan

Task: `TASK-2384`

GitHub: [#1995](https://github.com/rmusser01/tldw_server/issues/1995)

## Stage 1: Baseline Reconciliation
**Goal**: Reconcile the current `dev` implementation against the Phase 2 issue requirements and existing Workspace evidence.
**Success Criteria**: Existing docs, tests, and prior UAT evidence are mapped to #1995 scope; baseline failures are identified with root cause.
**Tests**:
- `python -m pytest tldw_Server_API/tests/Workspaces -q`
**Status**: Complete

## Stage 2: Contract Repair
**Goal**: Fix stale or missing verification contracts exposed by the baseline run without broadening product behavior.
**Success Criteria**: Eligibility/resource membership tests reflect the current supported adapter set and still fail closed for future resource types.
**Tests**:
- `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_eligibility.py -q`
- `python -m pytest tldw_Server_API/tests/Workspaces -q`
**Status**: Complete

## Stage 3: Scenario Matrix
**Goal**: Create the full Phase 2 single-user workspace loop matrix required by #1995.
**Success Criteria**: Matrix covers workspace lifecycle/import, notes, media/sources, artifacts, chats, prompts, workflows, watchlists, ACP sessions, sandbox sessions, active-context gates, global visibility, cross-workspace recovery, runtime bindings, archived workspace behavior, API contracts, and frontend contracts.
**Tests**:
- Documentation review against #1995 acceptance criteria.
**Status**: Complete

## Stage 4: Focused Verification
**Goal**: Run backend, frontend/client, and browser/manual evidence checks that map to the matrix.
**Success Criteria**: Focused backend suites pass, focused frontend Workspace contract suites pass or documented blockers are filed, and visible single-user loop evidence is captured.
**Tests**:
- Backend Workspace pytest suite.
- Focused frontend Vitest Workspace suites.
- Focused Playwright Workspace manager/research Workspace suites where local services are available.
**Status**: Complete

## Stage 5: Release Evidence Closeout
**Goal**: Update tracking records and post final evidence back to the Workspace epic.
**Success Criteria**: Backlog task includes verification results and limitations, #1995 has a PR/evidence comment, and #1984 receives the final evidence summary required by #1995.
**Tests**:
- Bandit on touched backend paths, or document non-code/touched-test-only scope.
**Status**: In Progress
