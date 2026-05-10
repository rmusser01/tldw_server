# MCP Hub Workflow-First Control Panel Design

Date: 2026-05-10
Status: Approved for planning
Owner: Codex brainstorming session
Backlog: TASK-211

## Summary

Redesign the MCP Hub control panel from an object-centric tab list into a workflow-first hub for WebUI and extension users.

The current shared page in [McpHubPage.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx) exposes 11 peer tabs: Tool Catalog, Servers & Credentials, Profiles, Assignments, Approvals, Path Scopes, Capability Mappings, Workspace Sets, Shared Workspaces, Governance Packs, and Audit. Those are accurate backend concepts, but they ask users to already understand the MCP control plane before they can complete a basic task.

The approved direction is the heavier workflow-first redesign. The new page should organize the same underlying capabilities around the jobs users are trying to complete:

- Setup
- Access
- Workspaces
- Governance
- Audit

The first implementation program should preserve existing child components and service contracts wherever possible. It should change the information architecture, routing state, workflow framing, readiness summaries, and first-use guidance before attempting deeper form rewrites.

For PR sizing, split that program into at least two slices: Stage 1 ships the workflow shell, URL state, child-view grouping, audit drilldown mapping, and Setup-first default; Stage 2 adds readiness/status summaries and richer first-use empty-state guidance. Stage 1 may include static workflow descriptions or lightweight placeholders, but it should not require all readiness aggregation to land in the same PR.

## Problem

MCP Hub has grown into a capable control panel, but its navigation exposes storage and policy objects as equal choices. That creates avoidable cognitive load for both first-time users and admins.

Current pain points:

- P1: Navigation overload. Eleven peer tabs make the page feel like an admin database browser rather than a control panel.
- P1: First-use dead end. The explainer says to start with Tool Catalog, but an empty catalog tells users to add a server elsewhere. The primary first-use action should be Add Managed Server.
- P1: Cross-tab dependency blindness. Profiles depend on tools, assignments depend on profiles, workspace policies depend on scopes and workspace sets, and audit findings point back into all of them. The UI does not surface dependency readiness.
- P2: Extension width risk. The flat tab row is fragile in extension/options layouts and becomes harder to scan as the surface grows.
- P2: Object labels are too internal. Terms like Capability Mappings, Path Scopes, and Workspace Sets are correct, but they need workflow framing so users know when to use them.
- P2: Remediation context is weak. Audit can open affected objects, but the destination views do not strongly carry why the user was sent there.

From a senior UX/HCI perspective, the core issue is a mismatch between the system model and the user's task model. The backend model is object based; the user model is workflow based: connect tools, grant access, scope local workspaces, govern risk, and fix audit findings.

## Goals

- Replace the single 11-tab peer navigation with workflow-level navigation.
- Preserve existing MCP Hub functionality in the first implementation slice.
- Make first-time setup legible without requiring prior MCP Hub knowledge.
- Make cross-object dependencies visible through readiness/status summaries.
- Keep expert object-level editors available under their workflow groups.
- Support WebUI and extension through the shared UI package.
- Preserve audit drilldown behavior.
- Preserve deep links and provide a compatibility path for old tab-key references.
- Avoid backend changes in the first slice unless frontend aggregation is demonstrably insufficient.
- Add focused test coverage for workflow navigation, deep links, audit drilldown, and extension parity.

## Non-Goals

- No new MCP Hub backend capabilities in the first workflow shell slice.
- No replacement of the underlying permission, assignment, workspace, approval, governance pack, or audit service contracts.
- No removal of existing object editors.
- No live migration of stored MCP Hub policies.
- No extension-only UI fork.
- No visual redesign of every form in the first slice.
- No automatic policy mutation from audit findings beyond the inline remediation actions that already exist.
- No new backend summary endpoint unless the implementation proves that client-side aggregation is too slow, inconsistent, or too duplicative.

## Current Repo Foundation

### Shared route and page

- [apps/tldw-frontend/pages/mcp-hub.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/pages/mcp-hub.tsx) dynamically imports the shared route.
- [apps/packages/ui/src/routes/option-mcp-hub.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/routes/option-mcp-hub.tsx) renders `McpHubPage` inside the option layout.
- [apps/tldw-frontend/extension/routes/option-mcp-hub.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/extension/routes/option-mcp-hub.tsx) renders the same shared `McpHubPage` through the extension route.
- [apps/packages/ui/src/routes/option-settings-mcp-hub.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/routes/option-settings-mcp-hub.tsx) redirects `/settings/mcp-hub` to `/mcp-hub`.

### Current MCP Hub component set

The current components already map cleanly into workflow groups:

- [ToolCatalogsTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx)
- [ExternalServersTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx)
- [PermissionProfilesTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/PermissionProfilesTab.tsx)
- [PolicyAssignmentsTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/PolicyAssignmentsTab.tsx)
- [PathScopesTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/PathScopesTab.tsx)
- [WorkspaceSetsTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/WorkspaceSetsTab.tsx)
- [SharedWorkspacesTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/SharedWorkspacesTab.tsx)
- [ApprovalPoliciesTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/ApprovalPoliciesTab.tsx)
- [GovernancePacksTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/GovernancePacksTab.tsx)
- [CapabilityMappingsTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/CapabilityMappingsTab.tsx)
- [GovernanceAuditTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/GovernanceAuditTab.tsx)

### Service contract

[mcp-hub.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/tldw/mcp-hub.ts) already exposes typed client calls for:

- tool registry summary
- external servers and credential slots
- permission profiles
- policy assignments and overrides
- approval policies
- path scopes
- workspace sets
- shared workspaces
- capability mappings
- governance packs
- governance audit findings
- effective policy and external access summaries

These endpoints are enough for the first workflow shell and readiness summaries.

## Proposed Information Architecture

Use five top-level workflows:

| Workflow | User job | Existing child views |
| --- | --- | --- |
| Setup | Connect MCP servers, configure credentials, verify available tools | Tool Catalog, Servers & Credentials |
| Access | Define who can use tools and bind access to personas, groups, or defaults | Profiles, Assignments |
| Workspaces | Define trusted local path and workspace boundaries | Path Scopes, Workspace Sets, Shared Workspaces |
| Governance | Manage approvals, portable packs, and capability adapters | Approvals, Governance Packs, Capability Mappings |
| Audit | Find and remediate broken or risky MCP Hub configuration | Audit Findings |

The workflow names should be the main navigation labels. Existing object names remain visible as child views inside each workflow. This keeps expert precision without forcing every user through the object model first.

## Navigation And State Contract

`McpHubPage` should own three related pieces of state:

- `workflow`
- `view`
- `drillTarget`

Suggested type names:

```ts
type McpHubWorkflowKey =
  | "setup"
  | "access"
  | "workspaces"
  | "governance"
  | "audit"

type McpHubViewKey =
  | "tool-catalogs"
  | "credentials"
  | "profiles"
  | "assignments"
  | "path-scopes"
  | "workspace-sets"
  | "shared-workspaces"
  | "approvals"
  | "governance-packs"
  | "capability-mappings"
  | "audit"
```

Workflow/view mapping:

```ts
const MCP_HUB_WORKFLOWS = {
  setup: ["credentials", "tool-catalogs"],
  access: ["profiles", "assignments"],
  workspaces: ["path-scopes", "workspace-sets", "shared-workspaces"],
  governance: ["approvals", "governance-packs", "capability-mappings"],
  audit: ["audit"]
}
```

Recommended defaults:

- Default workflow: `setup`
- Default Setup view: `credentials`
- Default Access view: `profiles`
- Default Workspaces view: `path-scopes`
- Default Governance view: `approvals`
- Default Audit view: `audit`

This intentionally changes the first-use default from Tool Catalog to Servers & Credentials. Tool Catalog stays available in Setup, but users without connected servers should first see the action that can create useful catalog data.

### URL state

Support query parameters:

```text
/mcp-hub?workflow=setup&view=credentials
/mcp-hub?workflow=access&view=assignments
/mcp-hub?workflow=workspaces&view=workspace-sets
/mcp-hub?workflow=governance&view=governance-packs
/mcp-hub?workflow=audit&view=audit
```

Rules:

- If `workflow` is missing, default to `setup`.
- If `view` is missing, use the workflow's default view.
- If `view` belongs to a different workflow, prefer the view and derive its workflow.
- If either query value is invalid, fall back to `setup/credentials`.
- Query updates should use replace behavior for internal tab switching where possible, so navigation does not pollute browser history.
- `/settings/mcp-hub` should continue to reach the same workflow shell through existing route behavior.

### Legacy tab-key compatibility

Audit findings and older tests already use tab/view keys such as `assignments`, `credentials`, and `workspace-sets`. Keep those keys as the canonical child view keys. The redesign should wrap them in workflows, not rename them.

This compatibility requirement applies to internal view keys, audit `navigate_to.tab` payloads, and existing test/page-object terminology. The current MCP Hub page does not expose a documented `?tab=` URL contract. If implementation discovers real `/mcp-hub?tab=...` usage in code or tests, map it through the same view-to-workflow helper as a compatibility convenience; otherwise `workflow` and `view` are the new URL parameters.

## Layout Design

The new layout should have two navigation levels:

1. Top-level workflow navigation: Setup, Access, Workspaces, Governance, Audit.
2. Child view navigation inside the selected workflow.

The layout should be compact and operational:

- no marketing-style hero
- no decorative cards
- no nested cards for page sections
- restrained status indicators
- dense enough for admin use
- responsive enough for extension widths

Recommended structure:

```text
MCP Hub
Manage external tool servers, access policy, workspace trust boundaries, and audit findings.

[Setup] [Access] [Workspaces] [Governance] [Audit]

Workflow status strip:
  counts, warnings, next action

Child views:
  [Servers & Credentials] [Tool Catalog]

Selected child view content:
  existing component
```

The status strip should be a short operational summary, not a tutorial block. It should answer:

- What exists?
- What is missing?
- What should I do next?

The existing dismissible explainer can remain, but it should be rewritten to reflect the workflow model and should not be the only source of first-use guidance.

## Readiness Summary Model

The first implementation should derive summary state on the frontend from existing endpoints. A later backend summary endpoint can be added only if the frontend implementation becomes too slow, inconsistent, or duplicated across components.

### Setup readiness

Inputs:

- `getToolRegistrySummary`
- `listExternalServers`

Summary fields:

- managed server count
- legacy server count
- executable managed server count
- servers with missing secrets or slot secrets
- servers with invalid or blocked auth templates
- registered tool count
- high/medium/low risk tool counts

Next action examples:

- If no managed server exists: Add Managed Server.
- If managed servers exist but secrets are missing: Configure Credentials.
- If servers are executable but no tools are registered: Review Server Runtime or Tool Catalog.
- If tools exist: Review Tool Catalog.

### Access readiness

Inputs:

- `listPermissionProfiles`
- `listPolicyAssignments`
- `listExternalServers`
- `listProfileCredentialBindings` for profiles included in the visible readiness check
- optionally `getEffectivePolicy` for the current sample preview

Summary fields:

- active profile count
- active assignment count
- assignments with no profile
- profiles with no credential binding where external servers require one, computed only after binding rows are fetched for the profiles being summarized
- assignments with overrides

Next action examples:

- If no profiles exist: Create Profile.
- If profiles exist but no assignments exist: Assign Profile.
- If assignments exist with overrides: Review Overrides.

### Workspaces readiness

Inputs:

- `listPathScopeObjects`
- `listWorkspaceSetObjects`
- `listSharedWorkspaces`

Summary fields:

- path scope count
- workspace set count
- shared workspace count
- multi-root readiness warnings
- unresolved workspace IDs
- overlapping shared roots

Next action examples:

- If no path scopes exist: Create Path Scope.
- If multi-root warnings exist: Review Workspace Warnings.
- If shared-scope policies are needed and no shared workspaces exist: Add Shared Workspace.

### Governance readiness

Inputs:

- `listApprovalPolicies`
- `listGovernancePacks`
- `listCapabilityAdapterMappings`

Summary fields:

- active approval policies
- installed governance packs
- active governance pack installs
- capability mappings
- unverified or failed source verification states where available

Next action examples:

- If no approval policy exists: Create Approval Policy.
- If no governance pack exists: Preview Governance Pack.
- If portable capability behavior is needed: Add Capability Mapping.

### Audit readiness

Inputs:

- `listGovernanceAuditFindings`

Summary fields:

- total findings
- error count
- warning count
- top related object count
- available safe inline remediation count, if derivable through existing audit helpers

Next action examples:

- If errors exist: Review Errors.
- If only warnings exist: Review Warnings.
- If no findings exist: Export Clean Audit or continue monitoring.

## First-Use Behavior

First-time users should land in Setup with Servers & Credentials selected.

Empty state requirements:

- If no managed servers exist, show Add Managed Server as the primary action.
- Tool Catalog should not be the first empty surface for a new user.
- If only legacy servers exist, explain that they are read-only inventory until imported into MCP Hub and offer Import where the current component already supports it.
- If servers exist but are not executable, the status strip should identify missing secrets or auth template issues before the user reaches Audit.

The copy should be direct and operational. Avoid explaining MCP generally on every screen. Explain the specific missing prerequisite and the next action.

## Audit Drilldown Behavior

The existing audit flow passes `navigate_to` targets into `McpHubPage`, then switches to the matching tab and stores `drillTarget`. Preserve that behavior through workflow mapping.

Mapping:

| Audit target tab | Workflow | Child view |
| --- | --- | --- |
| `credentials` | setup | credentials |
| `profiles` | access | profiles |
| `assignments` | access | assignments |
| `path-scopes` | workspaces | path-scopes |
| `workspace-sets` | workspaces | workspace-sets |
| `shared-workspaces` | workspaces | shared-workspaces |
| `approvals` | governance | approvals |
| `governance-packs` | governance | governance-packs |
| `capability-mappings` | governance | capability-mappings |
| `audit` | audit | audit |

Stage 1 requirements:

- `GovernanceAuditTab` continues to call `onOpen(target)`.
- `McpHubPage` resolves `target.tab` to workflow and view.
- The target child component receives the same `drillTarget` shape it expects today.
- If the destination object cannot be found, the child component should keep its existing behavior and the page-level shell should not throw.

Stage 3 can add richer audit-origin context after the shell is stable. That later context may show messages such as "Opened from audit: missing credential slot on Docs Managed." Stage 1 only needs to preserve the existing open-and-focus behavior through the new workflow/view mapping.

## Component Architecture

Stage 1 shell components:

- `McpHubPage`
  - owns URL-backed workflow/view state
  - owns drill target state
  - renders workflow navigation
  - renders child view navigation
  - renders selected child component

- `mcpHubWorkflowConfig.ts`
  - exports workflow metadata
  - exports view-to-workflow mapping
  - exports default view per workflow
  - exports validation helpers for query parameters

- `McpHubWorkflowNav`
  - renders top-level workflow nav
  - exposes selected workflow through ARIA-compatible tabs or segmented controls
  - remains responsive in extension widths

- `McpHubViewNav`
  - renders child views for the active workflow
  - reuses existing view labels

Stage 2 readiness components:

- `useMcpHubReadiness`
  - aggregates existing service calls
  - returns per-workflow summary data
  - exposes loading and error states per workflow
  - avoids blocking child view rendering on summary fetch failures

- `McpHubWorkflowStatus`
  - renders counts, warnings, and next action for the active workflow
  - links or buttons navigate to the relevant child view

The Stage 1 shell should avoid modifying the internals of large child components unless required for drilldown or Setup-first action wiring. Stage 2 introduces readiness aggregation and workflow status once the shell and URL contract are stable.

## Error Handling

Readiness summary failures should not make the page unusable.

Rules:

- Child views remain the source of truth for detailed errors.
- Summary fetch failures render a compact warning in the workflow status strip.
- If one summary endpoint fails, other summaries should still render where available.
- Invalid URL query values fall back to `setup/credentials`.
- Audit drilldown to an unknown view falls back to Audit and records no drill target.
- Summary actions should be disabled or hidden only when their destination is not available.

## Accessibility And Responsive Requirements

- Top-level workflow navigation must be keyboard reachable.
- Child navigation must expose selected state.
- Workflow and child view labels must remain readable at extension option widths.
- Status strip actions must have visible text labels.
- Do not rely on color alone for readiness severity.
- Audit drilldown context should be announced as visible text, not only as color or icon state.
- Existing `data-testid` hooks should either remain stable or receive compatibility replacements for E2E tests.

## Implementation Sequence

### Stage 1: Workflow shell and routing

Goal: Replace the flat tab row with workflow and child view navigation while keeping current child components.

Deliverables:

- workflow config module
- query parsing and URL sync
- top-level workflow nav
- child view nav
- child component rendering by view key
- old audit drilldown mapped through workflow/view
- Setup-first default route

Primary files:

- [McpHubPage.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx)
- new workflow config/component files under [MCPHub](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub)
- [mcp-hub.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/tldw/mcp-hub.ts) for type reuse only

### Stage 2: Readiness summaries and first-use flow

Goal: Add per-workflow status and next actions using existing endpoints.

Deliverables:

- `useMcpHubReadiness`
- status strip for active workflow
- first-use Setup behavior emphasizing Add Managed Server
- empty-state copy alignment for Tool Catalog and Setup
- summary error handling

### Stage 3: Audit context and governance hardening

Goal: Improve cross-workflow remediation without rewriting every object editor.

Deliverables:

- audit-origin context message beyond the Stage 1 open-and-focus drilldown
- drilldown tests for each workflow group
- status summary links from Audit to affected workflow/view
- optional grouping of audit filters around workflow impact

### Stage 4: Form-level UX follow-ups

Goal: After the shell proves useful, simplify the largest editors.

Candidate follow-ups:

- split External Servers into server identity, credential slots, auth template, and secret configuration panels
- make Profiles and Assignments use clearer progressive sections
- make Workspaces use a trust-boundary language layer above path/workspace objects
- make Governance Packs and Capability Mappings less JSON-first where possible

These are intentionally outside the first shell slice.

## Testing Plan

### Unit and component tests

Add or update focused tests under [apps/packages/ui/src/components/Option/MCPHub/__tests__](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/MCPHub/__tests__):

- workflow config maps every current view key to exactly one workflow
- default route lands on Setup / Servers & Credentials
- invalid query parameters fall back to Setup / Servers & Credentials
- selecting a workflow updates selected workflow and default child view
- selecting a child view updates selected view
- audit `Open` maps a finding to the expected workflow and child view
- explainer dismissal behavior still works
- readiness summary failure does not hide the selected child view

### Extension route parity

Update existing extension route tests so `/mcp-hub` still renders the shared workflow shell.

Required checks:

- extension route still imports `OptionMcpHub`
- `/mcp-hub` renders the shared `McpHubPage`
- `/settings/mcp-hub` continues to resolve to the shared hub route
- route parity tests do not require extension-only MCP Hub behavior

### E2E

Update [mcp-hub.spec.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts):

- page loads with MCP Hub heading and workflow navigation
- each workflow can be selected
- representative child views load and fire expected API calls:
  - Setup / Tool Catalog -> `GET /api/v1/mcp/hub/tool-registry`
  - Setup / Servers & Credentials -> `GET /api/v1/mcp/hub/external-servers`
  - Access / Profiles -> `GET /api/v1/mcp/hub/permission-profiles`
  - Access / Assignments -> `GET /api/v1/mcp/hub/policy-assignments`
  - Audit -> `GET /api/v1/mcp/hub/governance-audit`
- audit drilldown opens the mapped workflow and child view

### Verification commands

Expected focused verification after implementation:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx ../packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx
bunx vitest run ../packages/ui/src/routes/__tests__/mcp-hub-route.test.tsx ../../apps/tldw-frontend/__tests__/extension/route-registry.mcp-hub.test.tsx
npx playwright test e2e/workflows/tier-2-features/mcp-hub.spec.ts --reporter=line
```

Run `bun run verify:openapi` only if endpoint paths, generated clients, or OpenAPI-guarded service calls change.

For this design-only task, Bandit is not applicable because no Python code is touched.

## Risks

### User muscle memory

Existing users may expect the flat tab row. Mitigation: keep child view labels unchanged and support deep links by view key.

### Query state complexity

Workflow and child view state can drift if URL parsing is ad hoc. Mitigation: centralize mapping and validation in one config/helper module.

### Summary fetch overhead

Readiness summaries may trigger several endpoints on initial load. Mitigation: fetch per active workflow first or use lazy summary loading if the eager approach is too heavy.

### Backend summary temptation

A backend summary endpoint might be useful later, but adding it too early risks coupling the first UX slice to a broader API change. Mitigation: start frontend-first and promote a backend summary only after measuring duplicate work or latency.

### Scope creep into form rewrites

The largest current tabs are complex enough to absorb a full PR each. Mitigation: first implementation keeps child components mostly intact and limits form rewrites to follow-up tasks.

## Open Implementation Decisions

These decisions should be made during implementation planning, not in this design:

- Whether workflow navigation should use Ant Design `Tabs`, segmented controls, or a small custom responsive nav.
- Whether readiness summaries are eager for all workflows or lazy per active workflow.
- Whether URL updates use router replace on every view change or only on top-level workflow changes.
- Whether audit-origin context is page-level only or passed into child components for more specific focus messaging.

## Decision

Proceed with the workflow-first MCP Hub redesign as the implementation-ready direction. The first PR should focus on the workflow shell, route state, existing child view grouping, first-use Setup default, and test coverage. Deeper form simplification should be split into follow-up work after the shell lands.
