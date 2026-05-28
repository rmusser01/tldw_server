# Research Workspace Sandbox Handoff Design

## Goal

Close `TASK-478.23` by validating that Research Workspace can hand off its
canonical workspace ID into sandbox-owned diagnostics and admission state without
duplicating sandbox runtime or run lifecycle state in Research Workspace.

## Current State

The canonical workspace contract already defines the boundary:

- Workspaces owns canonical workspace identity, source membership, and status.
- Sandbox owns admission policy, session and run lifecycle, runtime isolation,
  and diagnostic envelopes.
- Research Workspace passes canonical workspace context into agent, tool, and
  sandbox handoffs.

The backend already persists `workspace_id`, `workspace_group_id`, and
`scope_snapshot_id` on sandbox sessions and runs. `SandboxStore.list_runs()` and
`count_runs()` already support those filters for in-memory, SQLite, and
PostgreSQL stores. The public admin runs endpoint returns workspace fields but
does not expose those filters. Research Workspace currently shows generic
sandbox capability remediation and links to runtime config, but it has no
workspace-scoped sandbox diagnostics surface.

## Design Review Adjustments

The initial design is directionally right, but three issues need to be locked
before implementation:

1. **Avoid moving sandbox state into Research Workspace.** The new UI must be a
   read-only diagnostic consumer. It may display sandbox-owned readiness,
   admission, and recent run summaries, but it must not store or synthesize
   sandbox run state in Research Workspace.
2. **Avoid another global banner.** The handoff belongs in the existing
   workspace capability/remediation flow or workspace settings/details path.
   The UI should be a compact inline action and detail panel, not a persistent
   trust bar or page-wide alert.
3. **Avoid admin-only diagnostics for normal users.** Runtime internals can stay
   admin-only, but the workspace handoff needs a user-safe endpoint that can
   answer "can this workspace use sandboxed actions, and has it produced any
   workspace-scoped runs?" without leaking unrelated users, host paths, or raw
   runtime details.
4. **Avoid route-policy coupling to sandbox admin.** The workspace diagnostics
   endpoint is user-facing and must remain registered even when the broader
   sandbox admin/ops router is disabled by route policy. Keep the admin sandbox
   router gated, but register this narrow read-only endpoint through a stable
   user-facing router spec.

## API

Add a sandbox-owned endpoint:

`GET /api/v1/sandbox/workspaces/{workspace_id}/diagnostics?source_label=research_workspace&limit=10`

The endpoint returns a single workspace diagnostics envelope:

```json
{
  "workspace_id": "workspace-alpha",
  "source_label": "research_workspace",
  "runtime": {
    "state": "not_configured",
    "reason_code": "sandbox_no_runtimes_discovered",
    "message": "No sandbox runtimes are available for workspace actions.",
    "management_surface": "sandbox_settings"
  },
  "admission": {
    "state": "blocked",
    "reason_code": "sandbox_not_configured",
    "message": "Enable a sandbox runtime before sandboxed workspace actions can run."
  },
  "runs": {
    "total": 0,
    "limit": 10,
    "has_more": false,
    "items": []
  },
  "links": {
    "runtime_config": "/admin/runtime-config",
    "admin_runs": "/admin/monitoring?focus=sandbox&workspace_id=workspace-alpha"
  }
}
```

The endpoint is sandbox-owned and read-only. It uses the authenticated user
context and the sandbox store filters. It should only return runs associated
with the requested canonical workspace ID and the requesting user unless the
caller is admin and an explicit admin mode is added later. The route should be
implemented as a narrow diagnostics router and share the same sandbox service
singleton as the admin sandbox router, so memory-backed stores still show runs
created through the admin/ops APIs when those APIs are enabled.

`source_label` is a contract label, not a routing alias. The active label is
`research_workspace`. The endpoint may default missing labels to
`research_workspace`, but it must not emit `workspace_playground` or treat old
route labels as current API metadata.

The endpoint should distinguish:

- `not_configured`: no runtime configured or discovered.
- `unavailable`: runtime discovery failed or all runtimes are unavailable.
- `blocked`: policy/admission denies sandboxed actions.
- `available`: sandboxed actions may run.
- `unknown`: sandbox status cannot be checked.

The first slice does not need to execute a sandbox run. It must prove the
diagnostic/admission handoff and workspace-scoped run filtering.

Also extend the existing admin runs endpoint with filters already supported by
the store:

- `workspace_id`
- `workspace_group_id`
- `scope_snapshot_id`

This keeps admin diagnostics useful and avoids duplicating filtering logic.

## WebUI

Research Workspace should expose one contextual action from existing workspace
service remediation/settings:

- Label: `View Sandbox Diagnostics`
- Location: existing workspace settings or capability remediation surface.
- Behavior: opens a compact details panel/drawer inline with the workspace
  workflow and calls the sandbox-owned diagnostics endpoint with the active
  canonical workspace ID and `source_label=research_workspace`.

The panel should show:

- Runtime/admission state with user-safe copy.
- Recent workspace-scoped sandbox runs if present.
- Empty state: `No sandbox runs are linked to this workspace yet.`
- Forbidden/unavailable states without raw endpoint dumps.
- A runtime config/admin link only when exposed by the response or existing
  management surface.

The UI must not add a new persistent banner, must not introduce
`/workspace-playground`, and must not emit active `workspace_playground` labels.

## UX Copy

Use concise operational copy:

- Not configured: `Sandbox is not configured for workspace actions. Enable a runtime before agents or tools can run isolated work.`
- Runtime unavailable: `Sandbox runtime discovery failed. Workspace actions that require isolation are blocked until a runtime is healthy.`
- Admission denied: `Sandbox admission denied this workspace. Review the policy before starting isolated actions.`
- Empty runs: `No sandbox runs are linked to this workspace yet.`
- Loading: `Loading sandbox diagnostics for this workspace.`
- Forbidden: `You do not have permission to view sandbox diagnostics for this workspace.`
- Backend unavailable: `Sandbox diagnostics are unavailable right now. Workspace sources and chat are unaffected.`

## Non-Goals

- Do not execute sandbox runs from Research Workspace in this slice.
- Do not duplicate sandbox run lifecycle state in Research Workspace stores.
- Do not redesign Research Workspace layout, header, or onboarding.
- Do not add a trust bar or new global banner.
- Do not add redirects, aliases, or active compatibility routes for
  `/workspace-playground`.

## Verification

- Backend tests for workspace diagnostics response states and workspace-scoped
  run filters.
- Admin runs tests for `workspace_id`, `workspace_group_id`, and
  `scope_snapshot_id` query filters.
- Frontend tests for the Research Workspace sandbox diagnostics action, request
  path, loading, empty, denied, and unavailable states.
- Live backend and WebUI validation via CDP/Playwright:
  - load `/research-workspace`;
  - open the workspace sandbox diagnostics action;
  - observe the request contains the active canonical workspace ID and
    `source_label=research_workspace`;
  - observe a truthful terminal state;
  - confirm no `/workspace-playground` redirect, alias, or active label.
- Bandit on touched backend Python.
- `RW-UAT-023` update only as far as live evidence supports.
