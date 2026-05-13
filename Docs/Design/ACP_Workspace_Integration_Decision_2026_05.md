# ACP Workspace Integration Decision - May 2026

## Parent Issues

- ACP maturity parent: https://github.com/rmusser01/tldw_server/issues/1532
- ACP workspace integration: https://github.com/rmusser01/tldw_server/issues/1540
- Canonical workspace dependency: https://github.com/rmusser01/tldw_server/issues/1526

## Decision

ACP projects, tasks, runs, reviews, diagnostics, and execution artifacts attach
to the canonical workspace model instead of defining a second product workspace.

For the first implementation slice:

- `WorkspacePlayground` remains the canonical user-facing workspace shell.
- `/api/v1/workspaces/{workspace_id}` remains the canonical server workspace
  record for product membership: sources, selected source state, generated
  artifacts, notes, and workspace-level settings.
- `/api/v1/agent-orchestration/workspaces/{id}` remains the ACP execution
  binding for filesystem root, health, MCP server config, and per-session env.
- The ACP execution binding must carry a reference to the canonical workspace
  when it is created from, displayed in, or used by WorkspacePlayground.

This preserves the existing ACP runtime safety model while preventing ACP from
becoming a parallel workspace product.

## Existing Anchors

### Canonical Workspace

`Docs/Design/Workspace_Canonical_Model_Decision_2026_05.md` names
`WorkspacePlayground` as the canonical first-slice shell. The browser-local
Zustand store in `apps/packages/ui/src/store/workspace.ts` remains the
responsive cache and offline-friendly UI state.

The backend canonical workspace route family is the repository's mixed-case
`tldw_Server_API/app/api/v1/endpoints/workspaces.py` package path. Those
endpoints use string workspace IDs and own workspace sources, selected source
order, notes, and artifacts.

### ACP Execution Workspace

`tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py` owns ACP
execution workspaces under `/api/v1/agent-orchestration/workspaces`. These rows
use integer IDs and hold:

- `root_path`
- `workspace_type`
- `parent_workspace_id`
- plaintext `env_vars`
- health status
- workspace MCP server config
- arbitrary metadata

`AgentProject.workspace_id` binds an ACP project to this execution workspace.
Dispatch then resolves `cwd` inside the execution workspace root, injects
enabled workspace MCP servers, and forwards workspace env into ACP sessions.

### Current Mismatch

The canonical workspace ID is a string. The ACP orchestration workspace ID is an
integer. Existing UI paths also mix them:

- `ACPSessionCreateModal` reads the current `WorkspacePlayground` `workspaceId`
  and sends it to ACP session creation as `workspace_id`.
- `AgentTasks` currently talks to `/api/v1/agent-orchestration/projects` and
  does not yet present canonical workspace membership as a first-class filter.
- ACP projects can already bind to an ACP execution workspace through
  `ProjectCreateRequest.workspace_id`.

The bridge must make this explicit rather than relying on matching names or
overloading one ID field.

## Data Relationship

### Canonical Workspace Owns Product Context

The canonical workspace is the product context users understand. It owns:

- source membership and selected-source state;
- workspace notes;
- generated work-product artifacts;
- saved workspace metadata and local persistence state;
- route handoffs from WorkspacePlayground, ChatWorkspace, DocumentWorkspace,
  Chatbooks, and extension capture.

### ACP Execution Workspace Owns Runtime Context

The ACP execution workspace is the runtime context needed to safely run an
agent. It owns:

- allowlisted filesystem root;
- health and git metadata;
- workspace-specific MCP server definitions;
- per-session env values;
- sandbox/runtime compatibility state.

### Bridge Contract

The first implementation should add an explicit bridge from ACP execution
workspace to canonical workspace:

```json
{
  "acp_workspace_id": 42,
  "canonical_workspace_id": "workspace-alpha",
  "canonical_workspace_source": "workspace_playground",
  "link_status": "linked"
}
```

Initial storage can use `ACPWorkspace.metadata.canonical_workspace_id` to avoid a
large migration. A later implementation can promote this to a dedicated indexed
column if filtering or reporting requires it.

Agent projects and runs should inherit the canonical workspace reference through
their bound ACP execution workspace. If a task is created without an ACP
execution workspace, it may still carry canonical workspace metadata, but it
cannot claim workspace root, MCP injection, or env execution guarantees until a
valid ACP execution workspace exists.

## User Flows

### WorkspacePlayground: Run Agent Task From Workspace

WorkspacePlayground is the preferred entry point for workspace-scoped ACP work.
A user should be able to start from the current workspace, choose a configured
agent/task template, and create an ACP project/task bound to the workspace.

Expected first-slice behavior:

- show the current workspace as the product context;
- create or select a linked ACP execution workspace for root/env/MCP behavior;
- create an AgentProject that references that execution workspace;
- create AgentTasks under that project;
- show run/review/diagnostic links back into Agent Tasks or ACP session detail.

### Agent Tasks: Manage Execution Inside A Workspace

Agent Tasks remains the detailed project/task/run/review surface. It should
support:

- filtering projects by canonical workspace;
- showing the canonical workspace name and link for each project;
- creating a project from a selected canonical workspace;
- surfacing execution-workspace setup gaps before dispatch.

Agent Tasks should not become a separate workspace browser. It should reuse the
canonical workspace identity and link back to WorkspacePlayground.

### ACP Playground: Direct Session Diagnostics

ACP Playground remains the direct session experimentation and diagnostics
surface. It may use the active WorkspacePlayground context as a convenience,
but it should not be the main workspace task workflow.

Direct sessions can record the canonical workspace ID as session metadata when
provided, but durable product work should become an AgentProject/AgentTask.

## Permissions, Ownership, And Safety

Workspace-scoped ACP behavior has two gates:

1. Canonical workspace ownership or access level.
2. ACP execution workspace root allowlist and runtime health.

The first gate decides whether the user may attach work to a workspace. The
second gate decides whether a downstream agent may run with filesystem, MCP,
and env access. Both must pass before a run is presented as workspace-scoped.

Existing `TokenScopeGuard` coverage remains authoritative for route-level API
access. Multi-user behavior must keep user ownership on canonical workspace
rows, ACP execution workspace rows, projects, tasks, runs, and session detail.

Workspace `env_vars` are plaintext operational configuration. UI and docs must
not present them as a secret store.

## Retention, Redaction, And Support Views

ACP retention and redaction rules already cover session detail, events,
artifacts, diagnostics, and audit views. Workspace integration should consume
those views rather than reimplement redaction.

The workspace handoff should stay coordinated with the retention/export/delete
policy tracked in #1512 and the support/audit redaction work tracked in #1513.
Until those issues are fully closed, this integration must not introduce a
separate workspace-level ACP retention pipeline or bypass the existing ACP
support-safe view controls.

Workspace-level history should show safe previews and link to the authenticated
ACP detail routes. Support-safe views should prefer the existing `redacted=true`
ACP route behavior and sanitized diagnostic/audit metadata.

Generated work products promoted into canonical workspace artifacts must follow
the future artifact contract in #1525. Until that lands, ACP run artifacts stay
linked as execution outputs, not polished workspace work products.

## MCP, Trusted Roots, And Environment Flow

Workspace MCP server configuration for ACP runs remains owned by the ACP
execution workspace in the first slice. Canonical workspace sources and selected
source state can guide task prompts and templates, but they should not silently
become MCP tools or environment variables.

Trusted roots are selected from the ACP execution workspace, not from canonical
workspace membership. The linked ACP execution workspace `root_path` is the
trusted root only after it is absolute and passes the existing
`ACP-WORKSPACE.allowed_base_paths` / `ACP_WORKSPACE_ALLOWED_BASE_PATHS`
allowlist validation. A canonical workspace ID by itself never widens the
filesystem trust boundary.

Projects, tasks, and runs inherit trusted-root behavior through their bound ACP
execution workspace. If a canonical workspace does not have a linked ACP
execution workspace, no workspace root, MCP server config, or env values flow
into the ACP run.

Dispatch behavior remains:

- resolve `cwd` inside the ACP execution workspace root;
- convert enabled workspace MCP servers into `mcpServers`;
- forward ACP execution workspace `env_vars` as per-session env;
- merge sandbox env through the existing sandbox path when sandbox mode is used.
- reject absolute or escaping `cwd` values outside the validated trusted root.

MCP Hub path-scope enforcement should consume the same resolved trusted root
when ACP context is present. Existing `McpHubWorkspaceRootResolver` and path
scope behavior are the baseline: unresolved or ambiguous workspace roots must
fail closed or require approval according to the active path-scope policy.

Future work may map canonical workspace source sets to MCP workspace sets, but
that should be a separate MCP Hub integration slice.

## API And UI Touchpoints

### Backend

- `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
  - Canonical workspace CRUD, sources, artifacts, notes, and selected source
    state.
- `tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py`
  - ACP execution workspaces, projects, tasks, runs, review, workspace health,
    MCP servers, and dispatch.
- `tldw_Server_API/app/core/Agent_Orchestration/models.py`
  - `ACPWorkspace`, `AgentProject`, `AgentTask`, and `AgentRun` contracts.

### Frontend

- `apps/packages/ui/src/store/workspace.ts`
  - Canonical browser workspace state and local persistence.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/`
  - Preferred workspace-scoped ACP entry point.
- `apps/packages/ui/src/components/Option/AgentTasks/`
  - Project/task/run/review management and diagnostics.
- `apps/packages/ui/src/components/Option/ACPPlayground/`
  - Direct session experimentation and troubleshooting.

## Implementation Slices

The implementation is split into three explicit delivery tracks:

- Backend model/API: Slice 1.
- Frontend navigation/UI: Slices 2, 3, and 4.
- Verification/testing: Slice 5.

### Slice 1: Backend Bridge Contract

Goal: make the canonical workspace to ACP execution workspace link explicit.

Scope:

- Add helper logic to find or create an ACP execution workspace linked to a
  canonical workspace.
- Store `canonical_workspace_id` in ACP workspace metadata initially.
- Include canonical workspace metadata in project/task detail responses.
- Add backend tests for ownership, missing workspace, missing allowlist,
  trusted-root inheritance, cwd containment, and duplicate bridge prevention.

### Slice 2: WorkspacePlayground Handoff

Goal: let users start an agent task from the canonical workspace shell.

Scope:

- Add a WorkspacePlayground action that opens a create-agent-task flow.
- Create or select the linked ACP execution workspace.
- Create the AgentProject/AgentTask with workspace context.
- Link back to Agent Tasks and ACP session detail after dispatch.

### Slice 3: Agent Tasks Workspace Filter

Goal: make Agent Tasks feel workspace-native without replacing it.

Scope:

- Filter projects/tasks by canonical workspace.
- Show workspace badges and links.
- Surface root/env/MCP setup gaps before dispatch.
- Preserve the all-projects view for operators.

### Slice 4: Workspace History And Diagnostics

Goal: make run history discoverable from workspace views.

Scope:

- Show recent ACP runs for the current canonical workspace.
- Link to ACP detail/events/artifacts/diagnostics/audit routes.
- Use redacted previews where support-safe views are required.

### Slice 5: Verification, Testing, And Closeout

Goal: prove the integration behavior before closing #1540 and split any
remaining implementation work.

Scope:

- Run backend tests for bridge creation, ownership, allowlist failures,
  trusted-root inheritance, and cwd escape rejection.
- Run frontend tests for WorkspacePlayground handoff, Agent Tasks filtering,
  setup-gap states, and ACP detail links.
- Add or update docs checks that preserve the canonical workspace versus ACP
  execution workspace distinction.
- Run Bandit on touched backend paths when implementation code changes.
- Update #1540 with the bridge decision and implementation issue list.
- Keep #1538 blocked on #1525 for polished artifact promotion.
- Keep #1537 downstream of stable workspace, artifact, retention, redaction,
  compatibility, and admin/deployment signals.

## Non-Goals

- No full route consolidation of ChatWorkspace or DocumentWorkspace.
- No ACP-only workspace browser.
- No public agent marketplace or installer.
- No new secret-management system for workspace env vars.
- No promotion of ACP run artifacts into canonical work products before #1525.

## Open Questions

- Should the canonical workspace to ACP execution workspace link be promoted
  from metadata to a dedicated column before team-scale filtering?
- Should WorkspacePlayground create execution workspaces automatically, or
  require explicit user confirmation when a filesystem root/env/MCP config is
  needed?
- Which artifact states from #1525 are required before ACP-generated outputs can
  appear as accepted workspace artifacts?

## Acceptance Mapping For #1540

- Workspace-to-ACP data relationship: covered by the data relationship and
  bridge contract sections.
- UI/API touchpoints: covered by backend/frontend touchpoints and user flows.
- Permission and retention implications: covered by permissions, safety,
  retention, redaction, trusted-root, MCP, and env sections.
- Implementation split: covered by the five implementation slices.
