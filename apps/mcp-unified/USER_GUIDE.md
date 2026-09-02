# MCP Unified User Guide

This guide is for users and operators working with the package-local MCP
Unified standalone gateway boundary. It focuses on profiles, external servers,
credential grants, configuration snapshots, and remote runtime commands.

The package status is `public-alpha`, and its publishing status is `published`.
Released versions are published on PyPI; repository versions remain release
candidates until their protected publish succeeds. The former
internal/experimental phase applies only to earlier
releases. The package CLI does not launch a supported end-user gateway server;
remote runtime commands require an already running package-local gateway
mounted by a host application.

## Publishing Readiness

The standalone package has publishing status `published` and package status
`public-alpha`. New users can install released versions from PyPI; developers
testing an unpublished repository version should install from the repository.

Run the full internal release candidate gate from the repository root:

```bash
make mcp-unified-rc
```

The RC installs the exact official Tier 1 Python SDK pin `mcp==2.0.0`
separately with the wheel and sdist, then verifies automatic `2026-07-28`
stdio negotiation, tool discovery, and one tool call. The pin is the official
Python SDK
[`v2.0.0`](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.0.0)
release at tag commit `6f69a37`. These are the package's explicit official-SDK
stdio scenarios. The URL-oriented official conformance server harness is not
applicable to this stdio-only strict surface, so the evidence does not claim
full transport or modern HTTP conformance.

Generate a TestPyPI publish plan without uploading artifacts:

```bash
make mcp-unified-publish-dry-run
```

Live upload is intentionally gated. The workflow confirmation must be set to
`MCP_UNIFIED_PUBLISH`, and the RC helper refuses execution unless
`MCP_UNIFIED_ALLOW_PUBLISH=1` is present in the environment. Production PyPI
uploads use the repository's configured trusted publishing environment rather
than a long-lived PyPI token.

## 1. Install The Package Boundary

From PyPI:

```bash
python -m pip install "mcp-unified[gateway]"
```

Downstream applications should use a compatible-minor pin:

```bash
python -m pip install "mcp-unified[gateway]~=0.3.0"
```

From the repository root, when testing unpublished changes:

```bash
python -m pip install -e "apps/mcp-unified[gateway]"
```

For development and artifact checks from the repository, install both the
gateway runtime and development tools:

```bash
python -m pip install -e "apps/mcp-unified[gateway,dev]"
```

For development extras from PyPI:

```bash
python -m pip install "mcp-unified[gateway,dev]"
```

Confirm the CLI is available:

```bash
mcp-unified-gateway package-info
```

When a host application mounts the package gateway, use `GET /mcp/status` for
package-local readiness. The response reports static package boundary metadata,
profile/external registry store persistence, default profile state, admin-auth
configured state, external server counts, warnings, and next actions. For the
embedded TLDW Server product, use `/api/v1/mcp/status` instead.

For a quick standalone JSON-RPC smoke check, confirm the smoke client is also
available and run the deterministic in-process fixture:

```bash
mcp-unified-smoke --help
mcp-unified-smoke inprocess --json-report -
```

### Embed The Strict Stdio Server

The public `mcp_unified.gateway` API supports these exact protocol profiles:

| Revision | Lifecycle | Batch requests |
| --- | --- | --- |
| `2026-07-28` | Per-request `_meta`; no initialize session | Rejected |
| `2025-11-25` | `initialize`, then operations | Rejected |
| `2025-06-18` | `initialize`, then operations | Rejected |
| `2025-03-26` | Standalone `initialize`, then operations | Accepted only after initialization |
| `2024-11-05` | `initialize`, then operations | Rejected |

Strict stdio owns negotiation, validation, profile projection, pagination,
cancellation, limits, and newline-delimited binary framing. Existing
HTTP/WebSocket routes retain their compatibility contracts; this package does
not claim modern MCP conformance for HTTP.

Pass a `GatewayCoreRuntime` implementation to the public entrypoint. Inject
caller-owned asynchronous binary streams for an embedded transport, or omit
them to use process stdin/stdout:

```python
import asyncio

from mcp_unified.gateway import GatewayLimits, serve_stdio

raise SystemExit(
    asyncio.run(
        serve_stdio(runtime, limits=GatewayLimits(max_in_flight=1))
    )
)
```

The runtime and host application own catalog contents, authorization, policy,
audit, local files and databases, application errors, and privacy decisions.
The protocol layer neither exposes nor duplicates application-local data, and
self-reported client identity is never an authorization input.

The exact `GatewayLimits` defaults are:

| Limit | Default | Limit | Default |
| --- | ---: | --- | ---: |
| `max_input_line_bytes` | 1,048,576 | `max_output_line_bytes` | 1,048,576 |
| `max_result_bytes` | 786,432 | `max_json_depth` | 64 |
| `max_in_flight` | 16 | `default_catalog_page_size` | 50 |
| `max_catalog_page_size` | 100 | `max_catalog_items` | 10,000 |
| `max_batch_items` | 100 | `max_requests_per_minute` | 600 |
| `request_burst` | 32 | `max_schema_bytes` | 262,144 |
| `max_schema_depth` | 32 | `max_schema_subschemas` | 1,024 |
| `max_schema_refs` | 256 | `max_schema_pattern_chars` | 4,096 |
| `max_schema_validation_processes` | 4 | `schema_validation_timeout_seconds` | 5.0 |
| `graceful_shutdown_timeout_seconds` | 5.0 | | |

Schema compilation and instance validation run in disposable bounded child
processes. On native Windows, the preflighted schema and complete validation
instance are briefly stored in an owner-only file in the operating-system
temporary directory so the nested stdio server can launch the child reliably.
The file is never logged, is removed during the same bounded child cleanup,
and is not retained after success, failure, timeout, cancellation, or shutdown.
Applications handling data that must never touch temporary storage should
account for this Windows behavior before enabling strict tool calls.

The modern profile emits private, zero-TTL cache hints:
`{"ttlMs": 0, "cacheScope": "private"}`. Legacy profiles omit those modern
fields. Public errors are typed and bounded; they allowlist stable reason,
kind, and safe limit metadata rather than leaking payloads, paths, credentials,
schemas, exception strings, or private result sizes. The smallest generic
overflow response is exactly 79 bytes including its newline: an output limit
of 79 emits it, while 78 fails closed with no truncated line.

Cancellation propagates to pending asynchronous runtime work. Shutdown uses
the configured 5.0-second grace period and reports residual input, output, or
cleanup work to stderr only. Python cannot forcibly stop a non-returning worker
thread, so hosts must bound synchronous work; clients supervising a stuck child
must escalate from closing streams to process termination and finally a kill.

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

### Know Which Tools Are Available

The standalone gateway does not have one static global tool list. The effective
tool surface depends on the installed backend tools, registered external MCP
servers, profile policy, credentials, workspace/path grants, approvals, and any
session-scoped grants. Use discovery in this order:

1. Start with presets to understand the intended role shape:

   ```bash
   mcp-unified-gateway list-presets
   mcp-unified-gateway show-preset project-researcher
   ```

2. Create or duplicate a profile into the configured store.

3. Preview that stored profile's effective tool surface:

   ```bash
   mcp-unified-gateway preview-profile-tools --profile <profile-id> \
     --config ./gateway.json

   mcp-unified-gateway preview-profile-tools --profile <profile-id> \
     --category filesystem --exclude-denied --config ./gateway.json
   ```

4. For a running gateway, the final model-facing discovery surface is still the
   MCP `tools/list` response for that authenticated session/profile. Profiles
   can also expose bridge tools such as `tool_categories.list`, `tool_search`,
   `tool_describe`, and `profile.tools.list` so clients can discover deferred
   or recommended tools without expanding the direct executable surface.

### Create A Custom Profile

For most operators, duplicating a bundled preset is the safest starting point.
When a preset is not close enough, create a profile JSON document and store it
with `create-profile`.

Minimal read-oriented profile:

```json
{
  "id": "docs-researcher",
  "name": "Docs Researcher",
  "description": "Read-only documentation and workspace research profile.",
  "policy_document": {
    "allowed_tools": [
      "tool_categories.list",
      "tool_search",
      "tool_describe",
      "profile.tools.list",
      "fs.list",
      "fs.read",
      "fs.stat",
      "fs.glob",
      "fs.grep"
    ],
    "capabilities": [
      "filesystem.read"
    ],
    "path_scope_mode": "workspace_root",
    "path_grants": [
      {
        "path": "docs",
        "actions": ["read"]
      },
      {
        "path": "src",
        "actions": ["read"]
      }
    ]
  },
  "metadata": {
    "agent_metadata": {
      "ui_label": "Docs Researcher"
    }
  }
}
```

Save the document as `docs-researcher.json`, then create and inspect it:

```bash
mcp-unified-gateway create-profile --profile-file ./docs-researcher.json \
  --config ./gateway.json

mcp-unified-gateway show-profile docs-researcher --config ./gateway.json
```

Validate the profile's tool surface before assigning it:

```bash
mcp-unified-gateway preview-profile-tools --profile docs-researcher \
  --config ./gateway.json
```

Use `explain-policy` to inspect tool-level allow/ask/deny decisions,
`permission_rules`, and runtime TTL grants for a hypothetical call:

```bash
echo '{"path":"docs/README.md"}' | mcp-unified-gateway explain-policy \
  --profile docs-researcher --tool fs.read --args-stdin --config ./gateway.json
```

`explain-policy` does not execute filesystem tools and should not be treated as
a full validation of authored `policy_document.path_grants`. Validate path
grants with safe runtime tool calls against representative allowed and denied
paths in the intended workspace/session before assigning the profile.

If a tool is missing, first check whether the backend tool is installed or only
recommended, then check `policy_document.allowed_tools`, path grants, external
server registration, credential grants, and approval/session state. Use
`patch-profile` for small policy updates after creation.

### Profile Tooling Discovery

`list-presets` includes compact tooling discovery metadata for role presets.
Direct categories describe installed tools the profile can expose immediately
through `tools/list`, subject to the profile policy and any assignment
constraints. Deferred categories keep installed tools out of the initial model
tool surface while still making them discoverable through the profile discovery
bridge. Recommendation-only tools remain setup hints until an operator installs
or registers the matching backend tool and profile policy grants it.

Profiles can expose progressive disclosure bridge tools such as
`tool_categories.list`, `tool_search`, `tool_describe`, and
`profile.tools.list`. These help clients inspect available direct tools first,
then discover deferred installed tools and recommended next-step tools without
expanding the executable tool surface by default. `profile.tools.list` and
`tool_search` report per-tool availability metadata, including whether a tool is
`direct`, `deferred`, or `recommended_unavailable`, plus category-level counts.
When `tool_call` is present, clients can delegate to an installed profile-visible
tool by id after discovery; recommendation-only tools return `tool_not_enabled`
until their backend becomes available.

Filesystem-capable presets expose portable workspace-bounded helpers for common
read workflows: `fs.stat` for metadata, `fs.glob` for cross-platform path
matching, and `fs.grep` for UTF-8 text search. These helpers do not invoke a
host shell and remain subject to the active profile policy and workspace path
scope. `fs.glob` returns capped matches sorted by newest modification time by
default; pass `sort_by: "path"` when deterministic path ordering is more useful.
Glob does not apply `.gitignore` by default, but callers can pass
`respect_gitignore: true` when they want ignored paths filtered.
`fs.grep` defaults to `output_mode: "files_with_matches"` and can also return
matching line records with `output_mode: "content"` or per-file totals with
`output_mode: "count"`. Use `glob` or `type` to narrow grep scans by file
pattern or common language/file extension aliases. Directory grep scans respect
the workspace root `.gitignore` by default; direct-file grep still works for a
named ignored file when profile and path policy allow that file. `fs.grep` uses
literal matching by default; regex matching requires the filesystem module
`grep_allow_regex` setting. `multiline: true` is available only with regex mode
and `files_with_matches` or `count` output. Grep scans are also bounded by
per-file, total-byte, total-file, and walk-entry limits.

### Safe File Read, Patch, And Write Tools

Use `fs.read` as the canonical file-inspection tool. It returns bounded UTF-8
content plus file size, newline style, SHA-256 when available, truncation state,
and a short-lived read receipt for complete hashed reads when the filesystem
module has a stable `read_receipt_secret` configured.

For Jupyter notebooks, use `notebook.read` instead of `fs.read` when the caller
needs notebook structure rather than raw JSON. It returns notebook metadata,
cell ids, cell types, execution counts, output counts, byte size, SHA-256, and
an optional read receipt. Source is omitted by default. Callers can pass
`include_source=true` for bounded source previews, narrow the response with
`cell_ids`, and tune `max_source_chars` or `max_total_source_chars` within the
module limits.

For existing-file edits, prefer `fs.patch` over whole-file replacement. It
accepts unified diff text, derives affected paths before execution for path
policy checks, validates context in memory, and only writes after preimage
checks pass. For small literal replacements where a unified diff is unnecessary,
`fs.edit` replaces one exact UTF-8 string in an existing file. It rejects missing
or non-unique matches unless `replace_all=true`, rejects overlapping matches,
and it also requires either `expected_sha256` or a valid `read_receipt` from
`fs.read`. For whole-file creation or deliberate replacement, use `fs.write`.
`fs.write` `mode="create"` fails if the file already exists. `mode="replace"`
requires either `expected_sha256` or a valid `read_receipt` from `fs.read`.

For notebook edits, use `notebook.edit_cell` instead of `fs.write`. It performs
one cell-scoped operation by stable cell id: `mode="replace"` updates the target
cell source, `mode="insert"` adds a `code`, `markdown`, or `raw` cell before or
after the target, and `mode="delete"` removes the target cell. Replacing a code
cell clears stored outputs and execution count so stale execution artifacts are
not preserved. Like file edits, notebook edits require either
`expected_sha256` or a valid read receipt from `notebook.read`, and they can use
`dry_run=true` before writing.

This read-before-mutate flow protects against stale edits: if a file changes
after the model read it, the expected hash or receipt no longer matches and the
write is rejected instead of silently overwriting newer content.

For concurrent editing workflows, `fs.lock_acquire` and `fs.lock_release`
provide advisory leases for workspace-relative file paths. A successful acquire
returns a `lease_id` that callers can pass to `fs.edit` and `fs.write` as
`lock_lease_id`, to `notebook.edit_cell` as `lock_lease_id`, or to `fs.patch` as
`lock_lease_id_by_path`. Leases do not replace hashes or read receipts;
mutation tools still run their normal preimage checks. Operators can set
`require_lock_for_mutation=true` on the filesystem module when they want
mutations to fail with `lock_required` unless the caller supplies a matching
active lease.

The packaged lock manager supports `lock_manager_backend` values of `memory`,
`in_memory`, or `sqlite`. The memory and in-memory backends are process-local
and remain the default. The SQLite backend requires
`lock_manager_sqlite_path`, which is operator configuration for the lock store,
not a model or agent filesystem tool path. SQLite can coordinate cooperating
processes that share the same local database file.
It is not a distributed lock across hosts and is not guaranteed on unreliable
network filesystems.
`lock_manager_sqlite_timeout_seconds` controls SQLite wait timeouts.
`lock_manager_cleanup_interval` and `lock_manager_cleanup_limit` bound periodic
expired-lease cleanup. Unsupported configured lock backends fail at module
creation instead of silently falling back to memory.

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

Tool allow lists and path grants are separate checks. A profile must allow
`notebook.read` or `notebook.edit_cell` as tools, and the target path must also
have the matching `read` or `edit` path action. The shorthand mental model is
`NotebookRead(path)` for a read action and `NotebookEdit(path)` for an edit
action; these are policy concepts, not extra executable tool names.

The file-policy action vocabulary is broader than the tools currently shipped.
The executable filesystem actions today are:

- `read`: inspect file content, directory listings, search results, and path
  metadata, including notebook structure through `notebook.read`.
- `edit`: bounded existing-file edits through `fs.patch`, `fs.edit`, or
  `notebook.edit_cell`.
- `write`: deliberate whole-file create or replace through `fs.write`.
- `lock`: acquire or release advisory path locks through `fs.lock_acquire` and
  `fs.lock_release`.

The reserved action names are `delete`, `rename`, `move`, `share`, `export`,
`chmod`, and `admin`. Profiles may author and preview these grants now so policy
intent is explicit, but they do not become executable until a dedicated safe
tool for that operation lands. Do not treat these actions as aliases for
`write`: `share` and `export` are exfiltration-sensitive, `delete`, `rename`,
and `move` are destructive, and `chmod` and `admin` are administrative.

Operators that prefer inherited policy authoring can keep the executable
runtime contract flat by compiling `path_grant_authoring` into `path_grants`.
The supported authoring levels are `org`, `workspace`, `folders`, and `files`;
each rule still uses workspace-relative `prefix` or `path`, explicit `actions`,
and optional `effect`:

```json
{
  "path_grant_authoring": {
    "org": [
      {"prefix": ".", "actions": ["read"]}
    ],
    "workspace": [
      {"prefix": "documents", "actions": ["read", "edit", "write"]}
    ],
    "folders": [
      {"prefix": "documents/private", "actions": ["edit", "write"], "effect": "deny"}
    ],
    "files": [
      {"path": "downloads/report.md", "actions": ["read"]}
    ]
  }
}
```

The compiler emits normalized flat grants and validation diagnostics. Explicit
`path_grants` remain the authoritative runtime form; when both flat and
authored grants are present, flat `path_grants` win. Invalid authored grants are
not treated as legacy allowlists, so malformed authored policy fails closed
rather than widening access.

Denials and permission-decision metadata should be safe to show to operators:
reason code, requested action, workspace-relative path, grant outcome, grant
source, and redaction status. They should not include raw file content, read
receipts, raw diffs, or absolute host paths.

`fs.read_text` and `fs.write_text` remain compatibility tools for older clients.
New profiles and front-ends should prefer `fs.read`, `fs.patch`, `fs.edit`,
`fs.write`, and the lock tools when coordinating edits.

Recommendation catalog patches only change discovery metadata. They do not grant
execution authority, start external servers, create credential grants, or bypass
profile policy and approval requirements.

### Profile Permission Rule Grammar

Profiles can author first-slice Claude-style permission rules in the
`permission_rules` policy field. Rules compile into the same package-owned
`deny`, `ask`, and `allow` decision primitives used by profile explanations and
future hook/runtime integrations.

Examples:

```json
{
  "policy_document": {
    "permission_rules": [
      "Read(/docs/**)",
      {"pattern": "Edit(src/*.py)", "outcome": "ask"},
      {"pattern": "Bash(git *)", "outcome": "allow"},
      {"pattern": "WebFetch(https://*.example.com/docs)", "outcome": "ask"},
      {"pattern": "mcp__github__delete_repo", "outcome": "deny"},
      "Skill(review)",
      {"pattern": "Agent(backend-*)", "outcome": "ask"}
    ]
  }
}
```

Supported subject families in this slice:

- Exact tools, such as `fs.read`.
- Governed command aliases: `Bash(...)`, `Shell(...)`, `PowerShell(...)`, and
  `Monitor(...)`.
- Path-oriented tools: `Read(...)`, `Edit(...)`, `Write(...)`,
  `NotebookEdit(...)`, `Grep(...)`, `Glob(...)`, and `LSP(...)`.
- Domain-oriented tools: `WebFetch(...)` and `WebSearch(...)`.
- External MCP wildcard names, such as `mcp__github__*`.
- `Skill(...)` and `Agent(...)` subjects for future reusable workflow and
  subagent routing.

Command rules match parsed argv tokens rather than raw shell strings. `*`
matches exactly one argv token, and the executable token must be fixed. Broad
command grants such as `Bash(*)` are rejected, and shell control syntax such as
`&&`, `||`, `;`, `|`, redirection, command substitution, or backticks is not
accepted by this parser. These rules authorize only the governed virtual command
surfaces; they do not grant raw host shell execution. Empty string arguments are
valid after the executable, so patterns such as `Bash(git commit -m '')` can
match explicit empty argument values without allowing an empty executable.

Path rules are segment-aware. `*` matches within one path segment, while `**`
is the cross-segment wildcard. For example, `Edit(src/*.py)` matches
`src/app.py` but not `src/pkg/app.py`.

Domain rules normalize URL hosts before matching. URL credentials, ports, and
IPv6 brackets are stripped, so `WebFetch(http://[::1]:8000/docs)` and a subject
such as `http://[::1]:9999/anything` both match the normalized host `::1`.

`permission_rules` do not replace existing runtime checks. A path rule such as
`Read(/docs/**)` does not by itself grant the `fs.read` tool in
`evaluate_profile_tool_decision()`, and tool execution still needs the relevant
profile grants, path grants, credential grants, sandbox/process checks, and
runtime approvals. Runtime integrations for governed shell execution, WebFetch,
WebSearch, LSP diagnostics, hooks, and admin policy simulation are separate
follow-up tasks.

### Explain Policy Decisions And Tool Previews

Use `explain-policy` when you need to understand one effective profile/tool
decision before execution. It returns the final `allow`, `ask`, or `deny`
outcome, reason code, visibility, call state, relevant policy contributors, and
redacted subjects for the hypothetical call.

Use `preview-profile-tools` when you need to review a profile's effective tool
surface. It previews installed tools and profile recommendations, including
whether tools are visible, deferred, denied, or unavailable. Include a
`session_id` when runtime-effective preview should account for session-scoped
approval grants.

Local CLI examples:

```bash
mcp-unified-gateway explain-policy --profile <profile-id> --tool fs.patch \
  --args-json-file ./patch-args.json --config ./gateway.json

mcp-unified-gateway preview-profile-tools --profile <profile-id> \
  --category filesystem --config ./gateway.json
```

Remote CLI example:

```bash
export MCP_UNIFIED_GATEWAY_URL=http://127.0.0.1:8000/mcp
export MCP_UNIFIED_GATEWAY_ADMIN_KEY=replace-with-admin-key

echo '{"path":"src/app.py"}' | mcp-unified-gateway explain-policy \
  --remote --profile <profile-id> --tool fs.read --args-stdin

mcp-unified-gateway preview-profile-tools --remote --profile <profile-id> \
  --category filesystem --session-id "$MCP_SESSION_ID" --exclude-denied
```

Direct admin API examples:

```bash
curl -sS -X POST "$MCP_UNIFIED_GATEWAY_URL/policy/explain" \
  -H "X-MCP-Gateway-Admin-Key: $MCP_UNIFIED_GATEWAY_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"profile_id":"<profile-id>","tool_name":"fs.read","arguments":{"path":"src/app.py"}}'

curl -sS -X POST "$MCP_UNIFIED_GATEWAY_URL/profiles/<profile-id>/tool-preview" \
  -H "X-MCP-Gateway-Admin-Key: $MCP_UNIFIED_GATEWAY_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"category":"filesystem","include_denied":true,"session_id":"session-1"}'
```

Security notes:

- These calls require the gateway admin policy-explain permission remotely.
- Calls are audited when audit storage is configured.
- Responses redact or sanitize sensitive subjects and do not echo raw arguments.
- Prefer `--args-json-file` or `--args-stdin` over inline `--args-json` for
  sensitive arguments so values are not exposed in shell history or process
  listings.

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

### Remote (URL-based) external servers

Hosted MCP servers are registered the same way with a URL-based transport
instead of a command. `streamable_http` targets a Streamable HTTP endpoint
(one URL, JSON or SSE-framed responses, `mcp-session-id` session handling);
`sse` targets the legacy HTTP+SSE pairing (persistent event stream plus a
POST message endpoint).

```json
{
  "id": "linear",
  "name": "Linear MCP",
  "transport": "streamable_http",
  "url": "https://mcp.linear.app/mcp",
  "headers": {"Authorization": "Bearer <token>"},
  "enabled": true
}
```

`headers` are static headers sent on every request (typically authorization).
Per-call brokered credentials merge their `headers` into each tool call;
brokered `env` values have no HTTP equivalent and are ignored. Connection
failures map to distinct reason codes (`auth_required`, `tls_failed`,
`connect_failed`, `request_timeout`, `connection_closed`) so downstream
clients can surface honest readiness states.

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
- Bounded tool-call hook summaries: phase, hook id, hook order, action, status,
  sanitized reason code, and sanitized error type.
- UTC timestamp and integer epoch microseconds for stable ordering.

This lets operators compare, for example, whether one profile mode has a higher
tool-call success rate, whether a model is repeatedly denied a tool, or whether
a new tool prompt version changes latency or reason-code distribution.

### Package-Level Tool-Call Hooks

Embedders can provide ordered pre/post hooks by constructing a
`ConfiguredToolCallHookManager` and passing it through `MCPRuntimeDependencies`
as `tool_call_hook_manager`.

```python
from mcp_unified.tool_hooks import (
    ConfiguredToolCallHookManager,
    ToolHookRegistration,
)

hook_manager = ConfiguredToolCallHookManager(
    [
        ToolHookRegistration(
            hook_id="profile-policy",
            before=check_profile_policy,
            after=record_profile_observation,
            order=10,
        ),
        ToolHookRegistration(
            hook_id="approval-gate",
            before=request_approval_if_needed,
            phases=("pre",),
            order=20,
        ),
    ]
)
```

Pre-hooks run in ascending `order` and then by `hook_id`. The first pre-hook
decision with `deny`, `ask`, or `approval_required` stops evaluation and is
enforced by the protocol. If a pre-hook raises, the protocol fails closed and
the tool is not executed. Post-hooks run after success or failure; individual
post-hook failures are recorded as hook metadata and do not suppress the
original tool result or error.

Hook reporting is metadata-only. Stored events do not include hook messages,
raw callback metadata, tool arguments, result payloads, raw exception messages,
or absolute paths. Gateway JSON/admin configuration for hook registries is a
future surface; this slice exposes the package API for hosts and tests.

### What Reporting Does Not Capture

The metadata-only recorder does not capture tool arguments, tool result payloads,
secret values, raw exception text, hook messages, raw hook metadata,
conversation messages, files, screenshots, or browser/page contents. The
`capture_ref` field is only a future-safe reference slot; this slice does not
create or store raw captures.

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
