# MCP Unified User Guide

This guide is for users and operators working with the package-local MCP
Unified standalone gateway boundary. It focuses on profiles, external servers,
credential grants, configuration snapshots, and remote runtime commands.

The package is currently internal/experimental and distributed with the
`tldw-server` source tree. It is not a separately published standalone package.

## 1. Install The Package Boundary

From the repository root:

```bash
python -m pip install -e "mcp_unified[gateway]"
```

For development and artifact checks:

```bash
python -m pip install -e "mcp_unified[dev]"
```

Confirm the CLI is available:

```bash
mcp-unified-gateway package-info
```

## 2. Choose A Store

The standalone gateway supports a transient memory store and a persistent SQLite
store. Use SQLite for real profile and external server management.

Example `gateway.json`:

```json
{
  "store": {
    "kind": "sqlite",
    "sqlite_path": "./mcp-gateway.db"
  },
  "default_preset_id": "project-researcher"
}
```

Validate it:

```bash
mcp-unified-gateway validate-config ./gateway.json
```

You can also set a default config path:

```bash
export MCP_UNIFIED_GATEWAY_CONFIG=./gateway.json
```

## 3. Work With Profiles

Profiles define which tools, capabilities, external servers, credentials, and
approval behavior are available to a client mode.

List bundled presets:

```bash
mcp-unified-gateway list-presets
```

Inspect a preset:

```bash
mcp-unified-gateway show-preset project-researcher
```

Duplicate a preset into your persistent store:

```bash
mcp-unified-gateway duplicate-preset project-researcher \
  --profile-id researcher \
  --name "Project Researcher" \
  --config ./gateway.json
```

Make that profile the gateway default:

```bash
mcp-unified-gateway set-default-profile researcher \
  --config ./gateway.json
```

Show the current default:

```bash
mcp-unified-gateway get-default-profile \
  --config ./gateway.json
```

### Profile Tooling Discovery

`list-presets` includes compact tooling discovery metadata for role presets.
Direct categories describe tools the profile can expose immediately, subject to
the profile policy and any assignment constraints. Deferred categories describe
recommended unavailable tools or server-backed capabilities that may be useful
for the role but are not executable until explicitly registered, granted, and
allowed by policy.

Profiles can expose progressive disclosure bridge tools such as
`tool_categories.list`, `tool_search`, `tool_describe`, and
`profile.tools.list`. These help clients inspect available direct tools first,
then discover recommended next-step tools without expanding the executable tool
surface by default.

Filesystem-capable presets expose portable workspace-bounded helpers for common
read workflows: `fs.stat` for metadata, `fs.glob` for cross-platform path
matching, and `fs.grep` for UTF-8 text search. These helpers do not invoke a
host shell and remain subject to the active profile policy and workspace path
scope. `fs.grep` uses literal matching by default; regex matching requires the
filesystem module `grep_allow_regex` setting. Grep scans are also bounded by
per-file, total-byte, total-file, and walk-entry limits.

Recommendation catalog patches only change discovery metadata. They do not grant
execution authority, start external servers, create credential grants, or bypass
profile policy and approval requirements.

For native browser inspection, prefer the Chrome DevTools Protocol path first.
The stable exact target for that path is
`ChromeDevTools/chrome-devtools-mcp`.

### Native CDP Browser Inspection

The `tldw-server` MCP host can expose optional read-only browser inspection
tools through Chrome DevTools Protocol. This path is intended for Frontend
Engineer, QA Engineer, and SDET profiles that need to inspect a running app
without granting browser interaction or arbitrary JavaScript execution.

Enable the module by setting a local CDP debugger URL:

```bash
export MCP_BROWSER_CDP_URL=http://127.0.0.1:9222
```

You can also enable the module explicitly:

```bash
export MCP_ENABLE_BROWSER_CDP_MODULE=true
```

If `MCP_BROWSER_CDP_URL` is set, the module auto-registers unless explicitly
disabled:

```bash
export MCP_ENABLE_BROWSER_CDP_MODULE=false
```

By default, the CDP endpoint must use a literal loopback host such as
`localhost`, `127.0.0.1`, or `::1`. Non-loopback CDP endpoints require an
operator-owned override:

```bash
export MCP_BROWSER_CDP_ALLOW_NON_LOOPBACK=true
```

Available native read tools:

- `browser.status` - report configured/reachable CDP availability.
- `browser.pages.list` - list inspectable page targets.
- `browser.snapshot` - return a bounded accessibility-tree snapshot.
- `browser.page_state` - return fixed URL/title/readiness/viewport state.
- `browser.screenshot` - capture a bounded base64 screenshot payload.
- `browser.console` - observe console/log events during a bounded window.
- `browser.network` - observe network events during a bounded window.

These tools do not accept CDP URLs, navigation targets, selectors, custom
scripts, expressions, clicks, typing, reloads, focus changes, or storage
mutation arguments. Screenshot, snapshot, console, and network reads are bounded
by module settings such as `MCP_BROWSER_CDP_MAX_SNAPSHOT_NODES`,
`MCP_BROWSER_CDP_MAX_EVENTS`, `MCP_BROWSER_CDP_OBSERVATION_WINDOW_MS`, and
`MCP_BROWSER_CDP_SCREENSHOT_MAX_BYTES`.

## 4. Register External Servers

External servers describe upstream MCP servers that the gateway can manage and
expose through profile policy.

Create `search-server.json`:

```json
{
  "id": "search",
  "name": "Search MCP Server",
  "transport": "stdio",
  "command": ["python", "-m", "search_mcp_server"],
  "env_allowlist": ["PATH", "SEARCH_API_ENDPOINT"],
  "credential_slots": ["search_api_key"],
  "enabled": true,
  "auto_start": false
}
```

Add it to the registry:

```bash
mcp-unified-gateway create-external-server \
  --server-file ./search-server.json \
  --config ./gateway.json
```

List registered servers:

```bash
mcp-unified-gateway list-external-servers \
  --config ./gateway.json
```

The registry alone does not grant execution authority. A profile must also
allow the server/tool path, and any required credentials must be granted.

## 5. Add Credential Grants

Credential grants are metadata that bind a profile, external server, credential
slot, and broker reference. Do not put secret values in a grant file.

Create `researcher-search-grant.json`:

```json
{
  "id": "researcher-search-api-key",
  "profile_id": "researcher",
  "external_server_id": "search",
  "broker_id": "env",
  "credential_slot": "search_api_key",
  "scopes": ["search:read"],
  "enabled": true
}
```

Create the grant:

```bash
mcp-unified-gateway create-credential-grant \
  --grant-file ./researcher-search-grant.json \
  --config ./gateway.json
```

List grants:

```bash
mcp-unified-gateway list-credential-grants \
  --profile-id researcher \
  --config ./gateway.json
```

## 6. Export And Import Configuration Snapshots

Use snapshots to move gateway configuration between environments or review a
pending import before applying it.

Export:

```bash
mcp-unified-gateway export-config \
  --output ./gateway-snapshot.json \
  --config ./gateway.json
```

Validate an import without writing:

```bash
mcp-unified-gateway import-config \
  --snapshot-file ./gateway-snapshot.json \
  --dry-run \
  --config ./gateway.json
```

Apply the import:

```bash
mcp-unified-gateway import-config \
  --snapshot-file ./gateway-snapshot.json \
  --config ./gateway.json
```

Snapshots preserve profile, assignment, external server, and credential grant
metadata. They do not store secret values.

## 7. Use Remote Runtime Commands

Local store commands mutate the configured store. Remote runtime commands call
admin endpoints on a running gateway process.

Set the mounted gateway URL:

```bash
export MCP_UNIFIED_GATEWAY_URL=http://127.0.0.1:8000/mcp
```

If admin auth is enabled, set the secret in the environment:

```bash
export MCP_UNIFIED_GATEWAY_ADMIN_KEY=replace-with-admin-key
```

List runtime state:

```bash
mcp-unified-gateway runtime-list
```

Start, stop, restart, or refresh an external server:

```bash
mcp-unified-gateway runtime-start search
mcp-unified-gateway runtime-stop search
mcp-unified-gateway runtime-restart search
mcp-unified-gateway runtime-refresh search
```

Refresh or reconcile all servers:

```bash
mcp-unified-gateway runtime-refresh
mcp-unified-gateway runtime-reconcile
```

Run configured install/update flows:

```bash
mcp-unified-gateway runtime-install search
mcp-unified-gateway runtime-update search
```

The CLI reads the admin key from `MCP_UNIFIED_GATEWAY_ADMIN_KEY`; it intentionally
does not accept the secret value as a command-line argument.

## 8. Troubleshooting

`--config is required`

: Pass `--config ./gateway.json` or set `MCP_UNIFIED_GATEWAY_CONFIG`.

`Credential grant management requires a persistent gateway store`

: Switch from memory storage to SQLite.

`Unable to reach gateway`

: Confirm `MCP_UNIFIED_GATEWAY_URL` includes the mounted base path, such as
  `http://127.0.0.1:8000/mcp`.

`external_server_not_found`

: Check the server id with `list-external-servers` for local store state or
  `runtime-list` for running gateway state.

Secrets appear in a config file

: Remove them. Store only broker ids, credential slot names, scopes, and binding
  metadata in gateway configuration.
