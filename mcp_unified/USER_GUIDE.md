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

### Safe File Read, Patch, And Write Tools

Use `fs.read` as the canonical file-inspection tool. It returns bounded UTF-8
content plus file size, newline style, SHA-256 when available, truncation state,
and a short-lived read receipt for complete hashed reads when the filesystem
module has a stable `read_receipt_secret` configured.

For existing-file edits, prefer `fs.patch` over whole-file replacement. It
accepts unified diff text, derives affected paths before execution for path
policy checks, validates context in memory, and only writes after preimage
checks pass. For whole-file creation or deliberate replacement, use `fs.write`.
`fs.write` `mode="create"` fails if the file already exists. `mode="replace"`
requires either `expected_sha256` or a valid `read_receipt` from `fs.read`.

This read-before-mutate flow protects against stale edits: if a file changes
after the model read it, the expected hash or receipt no longer matches and the
write is rejected instead of silently overwriting newer content.

Example action-aware path grants:

```json
{
  "path_scope_mode": "workspace_root",
  "path_grants": [
    {"path": "docs", "actions": ["read", "edit", "write"]},
    {"path": "docs/private", "actions": ["edit", "write"], "effect": "deny"},
    {"path": "downloads", "actions": ["read"]}
  ]
}
```

Actions do not imply each other. A profile with `read` can inspect files but not
edit them. A profile with `edit` can use `fs.patch` for existing files. A
profile with `write` can use `fs.write` and patch-created files when policy also
allows creation. Deny grants take precedence over broader allow grants, so a
private subtree can remain read-only or blocked under a writable parent.

Denials and permission-decision metadata should be safe to show to operators:
reason code, requested action, workspace-relative path, grant outcome, grant
source, and redaction status. They should not include raw file content, read
receipts, raw diffs, or absolute host paths.

`fs.read_text` and `fs.write_text` remain compatibility tools for older clients.
New profiles and front-ends should prefer `fs.read`, `fs.patch`, and `fs.write`.

Recommendation catalog patches only change discovery metadata. They do not grant
execution authority, start external servers, create credential grants, or bypass
profile policy and approval requirements.

### Git Inspection Tools

The `tldw-server` MCP host can expose optional native Git inspection tools when
the operator enables them with `MCP_ENABLE_GIT_MODULE=true`. These tools are for
reading the active workspace repository only; users cannot point them at another
repository path, and they do not check out branches, stage files, commit, merge,
rebase, stash, reset, clean, push, pull, run arbitrary Git commands, or invoke a
shell.

When enabled by the host, the bundled Architect, Merge Conflict Resolver, Code
Reviewer, DevOps Engineer, Backend Engineer, Frontend Engineer, QA Engineer, and
SDET profiles receive the Git inspection tools by default. Product Owner and
Documentation Writer do not receive Git tools by default.

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
`MCP_BROWSER_CDP_MAX_EVENTS`, `MCP_BROWSER_CDP_OBSERVATION_WINDOW_MS`,
`MCP_BROWSER_CDP_MAX_OBSERVATION_WINDOW_MS`, and
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

## 8. Inspect Tool-Use Reporting

Tool-use reporting is an optional metadata-only event stream for understanding
how profiles, modes, models, and tool prompt ids are used. It is intended for
operational review, prompt iteration, and later evaluator-labeled task outcomes.

Reporting is disabled by default. Enable it in the gateway config with a
persistent SQLite event store:

```json
{
  "store": {
    "kind": "sqlite",
    "sqlite_path": "./mcp-gateway.db"
  },
  "default_preset_id": "project-researcher",
  "tool_use_reporting": {
    "enabled": true,
    "store": {
      "kind": "sqlite",
      "sqlite_path": "./mcp-tool-events.db"
    },
    "write_timeout_seconds": 2.0,
    "retention_max_age_days": 30,
    "retention_max_events": 100000,
    "export_default_limit": 1000,
    "report_default_window": "24h"
  }
}
```

The report/export/cleanup CLI requires `tool_use_reporting.enabled=true` and
`tool_use_reporting.store.kind=sqlite`. A memory reporting store is only useful
for embedders and tests because it is process-local.

Build a profile-level report:

```bash
mcp-unified-gateway tool-events report --group-by profile \
  --config ./gateway.json
```

Common grouping dimensions are `profile`, `tool_prompt`, `model`, and `tool`.
Reports include `events_scanned`, `event_limit`, `truncated`, call counts, tool
call success rate, top sanitized reason codes, and p50/p95 duration values.

Export recent events as JSON Lines:

```bash
mcp-unified-gateway tool-events export --format jsonl --since 7d \
  --config ./gateway.json
```

Clean up events using explicit retention limits:

```bash
mcp-unified-gateway tool-events cleanup --max-age-days 30 --max-events 100000 \
  --config ./gateway.json
```

If `--max-age-days` or `--max-events` is omitted, cleanup falls back to
`tool_use_reporting.retention_max_age_days` and
`tool_use_reporting.retention_max_events`.

### What Reporting Captures

Each event is one attempted MCP tool call. The stored metadata can include:

- Runtime surface, execution origin, status, sanitized reason code, and duration.
- Requested and effective tool names, module id, category, source kind, and
  read/write flags.
- Profile id, mode id, model id, tool prompt id, prompt version, prompt variant,
  and action family.
- Grant, approval, installation, runtime availability, path-filter, truncation,
  idempotency replay, and nested-call indicators.
- UTC timestamp and integer epoch microseconds for stable ordering.

This lets operators compare, for example, whether one profile mode has a higher
tool-call success rate, whether a model is repeatedly denied a tool, or whether
a new tool prompt version changes latency or reason-code distribution.

### What Reporting Does Not Capture

The metadata-only recorder does not capture tool arguments, tool result payloads,
secret values, raw exception text, conversation messages, files, screenshots, or
browser/page contents. The `capture_ref` field is only a future-safe reference
slot; this slice does not create or store raw captures.

### Privacy, Retention, And Evaluations

Store event databases alongside other operator-controlled gateway state and
apply retention with `tool-events cleanup`. Short windows are usually enough for
prompt and profile iteration. Exported event files should be treated as
operational telemetry, reviewed before sharing, and deleted when no longer
needed.

Tool-use reporting complements operational metrics and traces. Metrics answer
"how much" and traces explain a specific request path; reporting gives a bounded
dimensioned event table for comparing profiles, models, modes, tools, and tool
prompt ids. Future evaluator-labeled task outcomes should join to these
metadata dimensions instead of requiring argument or payload capture.

## 9. Troubleshooting

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
