# MCP Hub Walkthrough Remediation Design

Date: 2026-05-10
Status: Approved for planning
Owner: Codex brainstorming session
Backlog: TASK-223

## Summary

Address the issues found during the toy MCP server walkthrough of the workflow-first MCP Hub and chat MCP controls.

The walkthrough proved that a managed stdio server can be saved in MCP Hub and can execute through MCP Unified after backend restart. It also exposed several broken or confusing transitions:

- Newly saved managed external servers do not become discoverable without backend restart.
- The apparent MCP refresh tool, `external.tools.refresh`, fails pre-execution validation because it is write-classified but does not override `validate_tool_arguments`.
- Chat can show `MCP Auto` with enabled tools while the raw request preview omits `tools` and `tool_choice`.
- The WebUI readiness gate blocks degraded but usable health behind a generic retry state.
- No-auth stdio servers look like they have missing auth because the setup UI emphasizes "No auth template", "no secret", and "Legacy Secret Fallback".
- Tool Catalog empty and stale states do not provide a reliable recovery path.
- Development networking can split between quickstart and advanced assumptions, producing silent API 404s.
- Local walkthrough setup can still touch repo runtime databases even when an alternate AuthNZ database is configured.

Implement the remediation as two PR-sized phases:

1. PR 1 fixes the end-to-end blockers: live external discovery refresh, refresh validation, chat payload consistency, and degraded readiness behavior.
2. PR 2 polishes setup comprehension: no-auth status copy, catalog guidance, deployment diagnostics, and setup isolation guidance or tests.

## Goals

- Make managed external server setup usable without restarting the backend.
- Make MCP Hub refresh and discovery errors explicit and recoverable.
- Ensure chat MCP controls, raw request preview, and actual request construction agree.
- Let degraded but usable API health enter the app with diagnostics instead of blocking the full UI.
- Make no-auth local stdio server setup look intentionally complete, not half-configured.
- Add enough diagnostics to identify WebUI deployment-mode and API-origin mismatch.
- Preserve the existing workflow-first MCP Hub shell and shared WebUI/extension route structure.

## Non-Goals

- Do not redesign every MCP Hub form.
- Do not replace MCP Hub policy, credential, or governance contracts.
- Do not move server lifecycle management into chat.
- Do not introduce server-side personal chat tool preferences.
- Do not build a new MCP transport stack.
- Do not make runtime refresh silently bypass MCP Hub mutation permissions.

## Current Repo Anchors

- `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx` renders the workflow-first MCP Hub shell.
- `apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx` creates, updates, imports, and displays managed external servers and credentials.
- `apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx` renders registry-backed tool metadata.
- `apps/packages/ui/src/hooks/useMcpTools.tsx` fetches MCP health, tools, catalogs, modules, and chat tool filter state.
- `apps/packages/ui/src/components/Common/McpToolSelector.tsx` renders shared MCP chat tool availability controls.
- `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundRawPreview.ts` builds raw chat request previews.
- `apps/packages/ui/src/models/index.ts` builds `ChatTldw` instances for actual chat requests.
- `apps/tldw-frontend/components/networking/ServerReadinessGate.tsx` blocks app entry until `/api/v1/health` reports healthy.
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/external_federation_module.py` exposes external federation tools and uses `ExternalServerManager`.
- `tldw_Server_API/app/core/MCP_unified/external_servers/manager.py` owns external server adapters and virtual tool discovery.
- `tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py` owns MCP Hub management endpoints.

## PR 1: Live Discovery And Chat Payload Correctness

### Backend Runtime Refresh Contract

Add a small MCP Hub management boundary for runtime discovery. The UI should not rely on calling MCP tools as an admin API.

Recommended endpoint:

```text
POST /api/v1/mcp/hub/external-servers/{server_id}/refresh-discovery
POST /api/v1/mcp/hub/external-servers/refresh-discovery
```

The single-server endpoint refreshes one server. The collection endpoint refreshes all managed runtime servers.

Both endpoints should:

- require the same mutation/admin permission used by create/update/delete external server endpoints;
- reach the live external federation runtime if it is initialized;
- reconcile managed server definitions from the configured server loader;
- add adapters for new enabled servers;
- update adapters for changed enabled servers;
- remove adapters and virtual tools for disabled or deleted servers;
- refresh tool discovery for the requested target;
- invalidate MCP capability and tool caches;
- return a structured result.

Suggested response shape:

```json
{
  "server_id": "toy-mcp",
  "refreshed_servers": 1,
  "total_servers": 1,
  "virtual_tools": 2,
  "errors": {},
  "requires_restart": false,
  "message": "Discovery refreshed"
}
```

If the live MCP runtime cannot be reached, return a non-2xx response only when the operation cannot be attempted. If the server was saved successfully but runtime refresh cannot run, the frontend should still preserve the saved server state and show retry or restart guidance.

### External Federation Validation

Add `ExternalFederationModule.validate_tool_arguments()`.

Rules:

- `external.tools.refresh` accepts no arguments or an optional non-empty string `server_id`.
- `external.servers.list` accepts no arguments.
- `ext.*` calls should not be validated as management refresh calls; upstream tool argument validation remains external-server specific.
- Unknown management tool names should fail closed.

This fixes the current pre-exec failure:

```text
Write-capable tool requires module.validate_tool_arguments override
```

### Frontend Setup Flow

After create, update, or import in `ExternalServersTab`:

1. Save the managed server.
2. Call refresh discovery for that server.
3. Reload external server list.
4. Refresh tool registry summary.
5. Invalidate or refetch chat MCP tool queries.
6. Show one of:
   - `Server saved and tools refreshed.`
   - `Server saved, but tool discovery failed.`
   - `Server saved. Restart or retry discovery after MCP runtime is available.`

The save operation and refresh operation should be represented separately in state. A refresh failure must not imply that the server failed to save.

### Tool Catalog Refresh

`ToolCatalogsTab` should add an explicit refresh action.

The action should:

- call the refresh-all discovery endpoint;
- reload the tool registry summary afterward;
- report refresh errors inline;
- preserve the existing catalog grouping and risk tags.

This is a recovery affordance, not a replacement for first-time server setup.

### Chat Effective Tool Decision

Create one shared helper for the effective chat MCP request decision, used by both raw preview and actual request construction.

Inputs:

- selected tool choice: `none`, `auto`, or `required`;
- normalized chat tools from `useMcpTools`;
- MCP health state;
- selected model id;
- model capability metadata;
- provider capability metadata if already available.

Outputs:

- `sendTools: boolean`;
- `effectiveTools?: ChatToolRecord[]`;
- `effectiveToolChoice?: "auto" | "required"`;
- `omissionReason?: string`;
- `modelSupportsTools: boolean`;
- `toolCount: number`.

Rules:

- `none` omits `tools` and `tool_choice`.
- `auto` and `required` send tools only when the model is tool-capable, MCP is not unavailable or unhealthy, and at least one `chatTool` exists.
- `required` downgrades to omission when no tool can be sent; the UI must show why.
- Raw preview and `pageAssistModel` must call the same helper or the same lower-level function so they cannot drift.

The chat UI should show a warning when the selected MCP state is not the effective request state, for example:

- `Tools omitted because selected model is not marked tool-capable.`
- `Tools omitted because MCP is offline.`
- `Tools omitted because no executable tools are enabled.`

### Readiness Gate

Adjust `ServerReadinessGate` to distinguish transport failure, unhealthy health, and degraded health.

Rules:

- Healthy `status: ok` or `status: healthy`: enter app normally.
- Degraded `status: degraded`, including HTTP 206: enter app and surface a non-blocking diagnostic warning.
- Unhealthy HTTP 503 or `status: unhealthy`: keep retry behavior.
- Network failure: keep retry behavior.
- Timeout behavior remains permissive enough for local development, but should explain the last known health state if available.

The gate should parse health body even when HTTP status is 206.

## PR 2: Setup Polish And Diagnostics

PR 2 depends on PR 1 because copy and diagnostics should sit on a working runtime refresh path.

### No-Auth Server State

Treat no-auth local stdio servers as a first-class valid state.

A managed server should show a neutral or success state such as `No credentials required` when:

- transport is `stdio`;
- config auth mode is absent, empty, or `none`;
- no credential slots are defined;
- no auth template is present or required.

In that state:

- do not show `No auth template` as a warning;
- do not show `no secret` as a missing requirement;
- do not show `Legacy Secret Fallback` unless server-level secrets are actually configured or explicitly selected.

### Legacy Secret Fallback

Only show the transitional server-level secret UI when the selected server needs or already uses server-level secret storage.

Recommended display logic:

- Show credential slots and slot secrets when slots exist.
- Show no-auth status when no credentials are required.
- Show legacy secret fallback only when the server config or stored state indicates server-level auth is relevant.

### Tool Catalog Empty And Stale States

Improve the catalog empty state:

- If no managed server exists: primary action `Add server`.
- If managed servers exist but no tools are registered: primary action `Refresh discovery`.
- If refresh fails: show server-specific errors.
- If tools exist but are not chat-executable: route the user toward Access or policy state instead of implying setup is incomplete.

The copy should separate:

- setup absence: no server exists;
- runtime absence: server exists but discovery did not produce tools;
- permission absence: tools exist but cannot be executed by the current context.

### Deployment Diagnostics

Add a compact diagnostics surface, preferably in MCP Hub or the shared networking diagnostics area.

It should expose:

- effective deployment mode;
- resolved API origin;
- health URL;
- whether calls are relative/proxied or direct;
- last health status and status code.

This is intended for quickstart versus advanced split-brain debugging. It should be hidden or compact by default so the admin surface stays dense.

### Setup Isolation Notes

Document or test the expected database isolation behavior for local MCP Hub walkthroughs.

The observed issue was that an alternate AuthNZ `DATABASE_URL` still allowed other runtime subsystems to touch repo-local databases. The PR 2 scope should not necessarily rework all database configuration, but it should make local walkthrough setup explicit enough that E2E and contributor runs avoid accidental repo DB churn where practical.

## Error Handling

Use separate save and refresh outcomes:

- Save failed: keep the form open and show the save error.
- Save succeeded, refresh succeeded: close form, reload lists, show success.
- Save succeeded, refresh failed: close or preserve form according to existing UX, but show retry with error detail.
- Runtime unavailable: show restart or retry guidance.
- Partial refresh: show counts plus errors, and keep successfully discovered tools.

Do not hide backend error strings that are already sanitized. Avoid logging secrets or config payloads in UI messages.

## Testing Strategy

### PR 1 Backend Tests

- Managed external server create/update followed by runtime refresh works without backend restart.
- Runtime refresh reconciles newly added, updated, disabled, and deleted managed servers.
- `external.tools.refresh` accepts valid arguments and rejects invalid `server_id`.
- `external.tools.refresh` passes protocol pre-exec validation.
- Refresh endpoint enforces MCP Hub mutation permissions.
- Refresh endpoint returns structured runtime-unavailable failure.

### PR 1 Frontend Tests

- `ExternalServersTab` reports save plus refresh success.
- `ExternalServersTab` reports save success plus refresh failure separately.
- `ToolCatalogsTab` refresh action reloads registry summary.
- Raw request preview and `pageAssistModel` include identical `tools` and `tool_choice` when MCP Auto is sendable.
- Raw request preview and `pageAssistModel` omit tools with the same omission reason when MCP tools are unavailable.
- `ServerReadinessGate` enters the app on degraded HTTP 206 health with diagnostics.

### PR 2 Frontend Tests

- No-auth stdio server displays `No credentials required`.
- No-auth stdio server does not show missing secret or missing auth-template warning copy.
- Legacy Secret Fallback appears only for server-level secret cases.
- Tool Catalog empty state chooses Add server, Refresh discovery, or policy guidance based on available data.
- Deployment diagnostics render quickstart and advanced effective values.

### E2E Smoke

Add or extend an MCP Hub walkthrough smoke with a toy stdio MCP server:

1. Create toy server.
2. Add it through MCP Hub.
3. Refresh discovery without backend restart.
4. Confirm toy tools appear in Tool Catalog.
5. Open chat, set MCP Auto, and confirm raw preview includes the toy tool definitions.

The E2E should avoid real external network dependencies and should clean up temporary files and runtime processes.

## Verification

Run focused verification for each PR:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_protocol_external_federation.py
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py
python -m bandit -r tldw_Server_API/app/core/MCP_unified tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py -f json -o /tmp/bandit_mcp_hub_walkthrough.json
```

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx ../packages/ui/src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx
bunx vitest run ../packages/ui/src/components/Option/Playground --run
npx playwright test e2e/workflows/tier-2-features/mcp-hub.spec.ts --reporter=line
```

Exact test paths may be adjusted during implementation based on the touched files and existing test organization.

## Acceptance Criteria

- PR 1 removes the backend restart requirement for normal managed external server discovery.
- PR 1 makes `external.tools.refresh` callable for valid refresh requests.
- PR 1 makes chat UI, raw preview, and actual request payload agree about MCP tools.
- PR 1 handles degraded health without blocking app entry.
- PR 2 makes no-auth local stdio setup read as intentionally configured.
- PR 2 makes Tool Catalog recovery actions clear.
- PR 2 exposes enough deployment diagnostics to identify quickstart/advanced API mismatch.
- Both PRs include focused tests and verification notes.

## Open Questions For Implementation

- Whether the refresh-all endpoint should live under `/external-servers/refresh-discovery` or a broader `/runtime/refresh-discovery` path.
- Whether the runtime manager should expose one `reconcile_servers()` method or combine reconciliation into `refresh_discovery()`.
- Whether deployment diagnostics belong directly in MCP Hub, a shared networking panel, or both.

These questions should be answered in the implementation plan, but they do not change the two-PR design direction.
