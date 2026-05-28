# Research Workspace ACP Canonical Bridge Design

## Goal

Close `TASK-478.22` by proving that Research Workspace can hand off its active canonical workspace ID into ACP-owned project, task, run, and diagnostic surfaces without duplicating ACP run state inside Research Workspace.

## Context

The canonical workspace contract already defines ACP as the owner of execution roots, projects, tasks, runs, reviews, and completion artifacts. Research Workspace owns source truth and passes `canonical_workspace_id` with `canonical_workspace_source: research_workspace` into `/api/v1/agent-orchestration/workspaces/canonical-bridge`.

Existing WebUI behavior already provides:

- `WorkspaceAgentTaskHandoffModal` for creating ACP-backed tasks from Research Workspace.
- `WorkspaceACPHistoryModal` for showing recent runs linked to the current Research Workspace.
- `AgentTasksPage` for ACP-owned task management with a workspace query filter.
- `/acp-playground?session=...&view=diagnostics` deep links for ACP session diagnostics.

The remaining gap is that ACP project listing is still fetched globally and filtered client-side. That makes live validation weaker, scales poorly for power users, and does not give the API a first-class way to answer "show me ACP work for this canonical Research Workspace."

## Design

### API

Extend `GET /api/v1/agent-orchestration/projects` with optional query parameters:

- `canonical_workspace_id`: canonical product workspace ID, for example `workspace-alpha`.
- `canonical_workspace_source`: canonical source label, defaulting to `research_workspace`.

The endpoint remains backward-compatible with existing `workspace_id` and `unbound` filters. When canonical filters are present, ACP filters the project list after applying the existing workspace scope:

1. Include projects whose bound ACP workspace has a `canonical_workspace` link matching the requested ID and source.
2. Include legacy or transitional projects whose project metadata has the same canonical ID and source.
3. Normalize legacy source labels internally, but emit the active `research_workspace` source label.

This keeps ownership in ACP while letting Research Workspace ask for the ACP view of one canonical workspace.

### WebUI

Update `WorkspaceACPHistoryModal` to request:

`/api/v1/agent-orchestration/projects?canonical_workspace_id=<id>&canonical_workspace_source=research_workspace`

The modal still keeps its client-side canonical ID guard before requesting tasks. That is a defensive check against older servers, proxy caches, or malformed rows.

Update `AgentTasksPage` to use the same server-side canonical filter when the route includes `workspace`, `workspace_id`, or `canonical_workspace_id`. It should still filter client-side for safety and retain the current URL/query behavior.

### Empty And Error States

No new Research Workspace trust banner or duplicated ACP status model is added. ACP-owned surfaces remain responsible for ACP state:

- Research Workspace history modal: no workspace selected, no linked runs, unsupported endpoint, backend unavailable, task-detail error.
- Agent Tasks: no linked ACP execution workspace, unsupported orchestration endpoints, ACP readiness issues, existing workspace-scoped projects.

### Live Validation

Use a live backend and WebUI with Playwright/CDP. The minimum accepted evidence is:

- Load `#/research-workspace`.
- Open Workspace settings and then `ACP run history`.
- Observe a request to `/api/v1/agent-orchestration/projects` carrying the active canonical workspace ID.
- Observe one truthful terminal state in the modal, such as explicit empty history or unavailable ACP service.
- Confirm no `/workspace-playground` route, redirect, alias, or active label is reintroduced.

`RW-UAT-022` should remain `Partial` unless the live run creates or finds a real workspace-scoped ACP run and opens its diagnostics successfully.

## Non-Goals

- Do not move ACP project/task/run ownership into Research Workspace.
- Do not add redirects or aliases for `/workspace-playground`.
- Do not redesign the Research Workspace header, status bars, or onboarding.
- Do not build sandbox diagnostics in this slice. Sandbox validation remains a separate gate.

## Risks And Mitigations

- Older data may only have project metadata and no bound ACP workspace link. Mitigate by matching both workspace canonical links and project metadata.
- Filtering entirely in SQL would be more efficient but larger. Mitigate by adding endpoint-level filtering first because current project lists are already fetched and enriched there.
- Query-param filtering may hide malformed source labels. Mitigate by defaulting missing source metadata to `research_workspace` when the canonical ID matches.

## Verification

- Backend unit tests for canonical project filters.
- Frontend unit tests for Research Workspace ACP history and Agent Tasks request URLs.
- Focused Playwright live backend/WebUI validation.
- Bandit on touched backend Python.
- Guard search for `workspace-playground`, `workspace_playground`, and `Workspace Playground` in active code/tests/docs touched by this slice.
