# MCP Hub Setup And Recovery UX Design

Date: 2026-06-26
Status: Approved for spec review
Owner: Codex brainstorming session
Backlog: TASK-223.2

## Summary

Improve MCP Hub setup and recovery around one measurable outcome: a user can
add one MCP server, understand whether it is usable, and recover from common
failure states without leaving the hub or reading raw configuration docs first.

This design focuses on the current MCP Hub experience inside `tldw_server`. It
does not create the standalone `mcp_unified` package or gateway. The current
standalone extraction work explicitly keeps the first stage inside
`tldw_Server_API`, so standalone package readiness belongs to a separate plan.

## Review Findings Addressed

- New users currently meet expert controls first: server ID, transport, owner
  scope, raw Config JSON, auth templates, credential slots, and legacy fallback
  state.
- Saved or imported servers can leave Tool Catalog in empty or stale states
  without a local recovery action.
- The MCP Hub status summary can look like readiness but is mostly static
  navigation.
- No-auth local stdio servers can look incomplete because the UI emphasizes
  "No auth template", "no secret", or "Legacy Secret Fallback".
- Experienced users need faster setup, explicit diagnostics, predictable
  refresh controls, and trustworthy error details.
- MCP Hub diagnostics need enough deployment context to diagnose quickstart
  versus advanced API-origin split-brain failures.
- Local walkthrough and E2E setup need clearer isolation expectations so setup
  tests do not unexpectedly touch repository runtime databases.
- The design-system scan found literal `rgba(0,0,0,0.45)` color tokens in
  `ExternalServersTab.tsx` that should be moved to the design token system.

## Goals

- Make first-run MCP server setup task-led instead of raw-config-led.
- Preserve expert/manual configuration paths from the first setup screen.
- Separate safe preflight validation from explicit runtime discovery.
- Provide consistent readiness state across Setup, server rows, status summary,
  and Tool Catalog.
- Put recovery actions where users encounter the problem.
- Treat no-auth stdio as intentionally complete when credentials are not
  required.
- Give experienced users sanitized diagnostics, operation timestamps, reason
  codes, and Audit handoff without exposing secrets.
- Expose effective deployment mode, frontend API origin, health endpoint, and
  setup-isolation guidance where they help diagnose setup failures.
- Keep the first implementation as a reviewable PR-sized remediation slice.

## Non-Goals

- Do not create the standalone `mcp_unified` package or gateway.
- Do not redesign every MCP Hub tab.
- Do not replace policy, credential, governance, or audit contracts.
- Do not fully solve chat/tool eligibility consistency in this slice.
- Do not add a broad third-party template marketplace.
- Do not silently launch local commands or contact remote servers during static
  preflight.

## Current Repo Anchors

- `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx` renders the MCP
  Hub shell and current status cards.
- `apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx` owns
  managed server creation, import, edit, credentials, and server list state.
- `apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx` renders
  registry-backed tool metadata and current empty states.
- `apps/packages/ui/src/components/Option/MCPHub/PolicyDocumentEditor.tsx`
  already provides a good guided/advanced pattern for policy editing.
- `Docs/superpowers/specs/2026-05-10-mcp-hub-walkthrough-remediation-design.md`
  defines the earlier MCP Hub PR 1/PR 2 walkthrough remediation split.
- `Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md`
  defines the separate standalone extraction direction.

## UX Architecture

Keep the existing workflow shell, but make Setup the default path to first
success. The first-run/add-server entry point should offer four choices:

- Generic local stdio server.
- Generic HTTP/SSE server.
- Import existing config.
- Advanced/manual configuration.

The guided paths use structured fields before exposing raw JSON:

- Stdio: command, args, env vars, working directory, owner scope, and optional
  display name.
- HTTP/SSE: URL, headers/auth choice, owner scope, and optional display name.
- Import: pasted or uploaded config, with preview and validation.
- Advanced/manual: raw Config JSON and existing expert controls.

The primary action is `Save and discover tools`. It clearly communicates that
the next step may contact a remote endpoint or start a local stdio process. An
expert secondary action, `Save without discovery`, preserves low-friction manual
workflows.

After save/import/discovery, the user lands on a result panel, not a passive tab.
The panel reports what happened, what was checked, what is usable now, and the
next valid action.

## Readiness Model

The UI should not infer readiness separately in each component. Introduce one
shared normalized readiness model. It may be backend-owned or frontend-normalized
from existing APIs in the first pass, but Setup, server rows, status summary, and
Tool Catalog must consume the same object.

Use a small set of user-facing display states:

- `needs_setup`
- `checking`
- `ready`
- `needs_attention`
- `no_tools`
- `stale`

Preserve detailed machine-readable reasons under the display state:

- `not_configured`
- `preflight_failed`
- `auth_missing`
- `runtime_unavailable`
- `unreachable`
- `discovery_failed`
- `no_tools_returned`
- `config_changed`
- `catalog_expired`
- `partial_capability`

Use `primaryReasonCode` plus `reasonCodes[]` because servers can have multiple
problems. For example, a server can have a stale prior catalog and a new missing
credential.

Suggested minimal shape:

```ts
type McpReadinessAction =
  | "add_server"
  | "edit_config"
  | "open_credentials"
  | "refresh_discovery"
  | "validate"
  | "view_details"
  | "open_tool_catalog"
  | "open_audit";

type McpDisplayState =
  | "needs_setup"
  | "checking"
  | "ready"
  | "needs_attention"
  | "no_tools"
  | "stale";

type McpReasonCode =
  | "not_configured"
  | "preflight_failed"
  | "auth_missing"
  | "runtime_unavailable"
  | "unreachable"
  | "discovery_failed"
  | "no_tools_returned"
  | "config_changed"
  | "catalog_expired"
  | "partial_capability";

type McpOperationState = {
  operationId: string;
  operation: "preflight" | "discovery";
  startedAt: string;
  initiatedBy: "user" | "system";
  timeoutMs?: number;
};

type McpCredentialState =
  | "not_required"
  | "required_missing"
  | "configured"
  | "legacy_fallback"
  | "unknown";

type McpServerReadiness = {
  serverId: string;
  displayName?: string;
  transport?: "stdio" | "http" | "sse";
  displayState: McpDisplayState;
  primaryReasonCode?: McpReasonCode;
  reasonCodes: McpReasonCode[];
  credentialState: McpCredentialState;
  toolCount: number;
  lastValidationAt?: string;
  lastDiscoveryAt?: string;
  lastSuccessfulDiscoveryAt?: string;
  inProgressOperation?: McpOperationState;
  message: string;
  allowedActions: McpReadinessAction[];
};

type McpHubReadiness = {
  displayState: McpDisplayState;
  primaryReasonCode?: McpReasonCode;
  reasonCodes: McpReasonCode[];
  serverCount: number;
  readyServerCount: number;
  toolCount: number;
  allowedActions: McpReadinessAction[];
};
```

`McpHubReadiness` handles the global first-run state where no `serverId` exists.
`McpServerReadiness` is per-server only. `credentialState` is separate from
problem reason codes so no-auth stdio can render as an intentional
`not_required` state instead of masquerading as a missing credential.

## Preflight Versus Discovery

Separate two operations:

### Preflight

Preflight is static validation. It can run before save. It may check:

- required fields;
- JSON shape;
- URL syntax;
- transport-specific config;
- owner scope;
- safe command/path facts for stdio;
- auth/secret binding completeness.

Preflight must not launch arbitrary stdio commands or contact remote servers.
For stdio, safe command facts mean static checks such as missing command,
obvious invalid path, or executable-on-PATH hint. They must not invoke a shell,
expand user-controlled command strings, or execute the target process.

### Discovery Refresh

Discovery refresh is an explicit bounded runtime operation. It can start a local
stdio process or contact HTTP/SSE endpoints only after save/import or direct user
action. It must:

- require existing MCP Hub mutation/admin permissions;
- respect RBAC and owner scope;
- respect existing outbound/runtime safety policy for configured external
  servers;
- use timeouts;
- serialize per server;
- expose operation metadata for UI polling/recovery;
- produce sanitized audit/log records;
- redact secrets from errors and diagnostics;
- update registry/catalog summary and readiness together.

The default create action should be `Save and discover tools`, with a secondary
`Save without discovery` path for expert/manual workflows.

## Recovery Actions

Recovery should appear where the user hits the problem.

Setup result panel:

- `ready`: link to Tool Catalog and any safe downstream use location.
- `no_tools`: explain that the server responded but exposed no tools; offer
  details and refresh.
- `auth_missing`: offer credentials.
- `runtime_unavailable`: offer edit config.
- `unreachable`: offer edit config and refresh.
- `stale`: offer refresh discovery.

Server rows:

- Validate.
- Refresh tools.
- Edit config.
- Credentials.
- Details.

Tool Catalog:

- Empty with no servers: add server.
- Empty with configured server and no discovery: refresh discovery.
- Stale catalog: refresh discovery.
- Auth/runtime failure: open credentials or config.
- Permissions-gated catalog: explain permission boundary and link to access
  controls only when allowed.

Status summary:

- Prefer data-backed readiness cards.
- If data-backed cards are too large for this slice, demote static cards to
  navigation and keep real readiness in Setup/server rows/Tool Catalog.

## No-Auth Stdio Copy

No-auth local stdio servers should render as intentionally complete wherever the
state appears:

- Setup result.
- Server row.
- Credential summary.
- Details drawer.

Use user-facing copy such as `No credentials required` and secondary detail such
as `Local stdio server`. Expert diagnostics can still show raw auth-template and
secret-binding facts, but those facts should not read as missing setup unless
the selected server actually requires credentials.

`Legacy Secret Fallback` should appear only when the selected server actually
uses the transitional server-level secret flow.

## Diagnostics And Trust

Add a diagnostics/details drawer for experienced users. It should show:

- display state and reason codes;
- credential state;
- transport and sanitized endpoint/command facts;
- tool count;
- last validation time;
- last discovery time;
- last successful discovery time;
- current operation metadata;
- last sanitized error category/message;
- links to the Audit tab for full event history.

Do not duplicate the full Audit tab inside the drawer.

Never show secret values, auth headers, tokens, env var values, or raw config
containing credentials. Redaction should be layered:

- field-name redaction for common secret keys;
- known auth/header/env locations;
- URL query string redaction;
- suspicious value-pattern redaction.

Validation, save, discovery, credential repair, and refresh actions should
produce sanitized audit records and never log secrets.

Add a small Hub environment diagnostics area, either inside the diagnostics
drawer or near Setup, with:

- effective deployment mode;
- frontend API origin;
- health endpoint URL and latest result;
- detected quickstart/advanced origin mismatch when known;
- setup isolation guidance for local walkthroughs and E2E runs.

This area must not expose API keys, JWTs, cookies, or secret-bearing headers.

## Phasing

### Phase 1: Discovery Checkpoint And Minimal Readiness Slice

Start with an implementation discovery checkpoint:

- confirm what existing APIs expose for managed servers and catalog summary;
- decide whether first-pass readiness can be frontend-normalized;
- add the smallest missing backend readiness/refresh contract only if existing
  APIs cannot support the slice.

Then implement a minimal vertical slice:

- readiness mapper;
- server row rendering;
- Tool Catalog empty/stale recovery state;
- no-auth stdio copy in the touched surfaces.

Acceptance:

- the same server has the same readiness state in server rows and Tool Catalog;
- stale/empty Tool Catalog states offer valid actions;
- no-auth stdio does not render as missing credentials;
- mapper tests cover display state, primary reason, secondary reasons, and
  allowed actions;
- credential state is derived explicitly, including `not_required` and
  `legacy_fallback`.

### Phase 2: First-Success Setup Flow

Add the guided add-server flow:

- generic stdio;
- generic HTTP/SSE;
- import config;
- advanced/manual.

Add preflight feedback and the save/discover result panel.

Acceptance:

- a new user can create a server through structured fields;
- preflight issues appear before save;
- `Save and discover tools` shows discovery result state after save/import;
- `Save without discovery` remains available for experienced users;
- mobile layout has no horizontal overflow, clipped controls, or hidden primary
  actions.

### Phase 3: Diagnostics And Audit Handoff

Add sanitized diagnostics:

- details drawer;
- operation metadata;
- timestamps;
- reason codes;
- credential state;
- deployment mode, API origin, health endpoint, and setup isolation guidance;
- redacted last error;
- Audit tab link.

Acceptance:

- experienced users can diagnose auth/runtime/discovery failures without
  inspecting browser/API state;
- users can diagnose quickstart versus advanced API-origin split-brain from MCP
  Hub or shared diagnostics;
- local walkthrough/E2E setup isolation expectations are documented or linked;
- redaction tests cover headers, env vars, URL query strings, args, and raw JSON;
- RBAC/audit behavior is covered for validation and discovery actions.

### Phase 4: Status, Polish, And Verification

Complete product polish:

- make status cards data-backed or demote them to navigation;
- remove misleading auth labels in remaining surfaces;
- fix literal color tokens in `ExternalServersTab.tsx`;
- add focused browser smoke coverage for first-run setup;
- add mobile/responsive checks.

Acceptance:

- status summary no longer implies false readiness;
- no-auth copy is consistent across setup, server row, credentials, and details;
- Tool Catalog recovery is available from empty and stale states;
- first-run setup browser smoke passes;
- mobile screenshot/visual QA shows no overlap, hidden actions, or horizontal
  overflow;
- touched backend code, if any, has Bandit run recorded in the implementation
  task.

## Testing Strategy

- Unit-test the readiness mapper with each display state and reason combination.
- Unit-test credential-state derivation for not-required, required-missing,
  configured, and legacy-fallback cases.
- Component-test server rows and Tool Catalog recovery actions.
- Add API tests for discovery refresh and readiness response if backend contract
  changes are required.
- Add redaction tests for diagnostics and audit/log payloads.
- Add one browser smoke path for guided first-run server setup.
- Add responsive/mobile checks for guided setup and diagnostics drawer.

## Risks And Mitigations

- Backend summaries may not expose enough detail for readiness. Mitigation:
  include an implementation discovery checkpoint before broader UI work.
- Discovery refresh can start local processes or contact remote endpoints.
  Mitigation: use explicit copy, `Save and discover tools`, timeouts, audit, and
  per-server locking.
- Guided UI can slow experienced users. Mitigation: keep advanced/manual visible
  from the first choice screen and keep `Save without discovery`.
- Status cards can grow into a large backend project. Mitigation: data-back them
  only if cheap; otherwise demote to navigation.
- Chat handoff can cause scope creep. Mitigation: only link to downstream use
  locations when current readiness data supports the claim.
- Deployment split-brain can look like an MCP failure. Mitigation: expose
  effective deployment mode, API origin, and health endpoint diagnostics in the
  setup/recovery surface.
- Local walkthroughs can touch unintended runtime databases. Mitigation:
  document or verify setup isolation expectations in the implementation slice.

## Open Questions For Implementation Planning

- Can current management/catalog APIs support the first readiness mapper without
  backend changes?
- What exact permission name should guard discovery refresh and preflight?
- Should discovery operation state be persisted, in-memory, or derived from
  current request lifecycle for the first slice?
- What catalog age threshold, if any, should mark `catalog_expired`?
- Which toy MCP server path should the browser smoke use?
