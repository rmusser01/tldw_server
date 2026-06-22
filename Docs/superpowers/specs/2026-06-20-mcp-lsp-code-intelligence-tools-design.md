# MCP LSP Code Intelligence Tools Design

## Context

`TASK-2281` tracks LSP-backed MCP tools for code intelligence. The current MCP
surface already has filesystem primitives, path-scoped profile grants, hooks,
lock leases, audit/metrics contracts, CodeGraph read tools, and a reusable MCP
smoke client harness. The missing piece is a live language-server-backed layer
for editor-grade diagnostics and code navigation.

The first slice should be deliberately narrow: Python only, read-first, and
safe to expose through existing MCP profiles. Ruff is the preferred Python
backend for diagnostics, formatting, and code-action previews. `python-lsp-server`
(`pylsp`) is the semantic navigation backend for definitions, references,
symbols, hover, and signature help.

References:

- Ruff editor setup and server command: https://docs.astral.sh/ruff/editors/setup/
- Ruff language server features: https://docs.astral.sh/ruff/editors/features/
- python-lsp-server capabilities: https://github.com/python-lsp/python-lsp-server
- Claude Code tool reference for north-star behavior: https://code.claude.com/docs/en/tools-reference
- Existing filesystem policy design: `Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md`
- Existing MCP smoke harness design: `Docs/superpowers/specs/2026-06-19-mcp-smoke-client-transport-harness-design.md`

## Goals

- Add a Python-first `lsp` MCP module with a stable `lsp.*` tool surface.
- Use a capability router so Ruff and pylsp each handle the operations they are
  good at.
- Keep the first slice read-oriented. Formatting and code actions return
  bounded proposed edits only; applying edits remains the job of `fs.patch` or
  `fs.write`.
- Enforce existing profile grants, path grants, hooks, lock/audit semantics, and
  observability contracts instead of creating a separate policy lane.
- Support optional installation through an `lsp` extra while also discovering
  configured or PATH-provided executables at runtime.
- Make missing or crashed backends degrade capabilities without disabling the
  whole module.

## Non-Goals

- Do not implement multi-language defaults in the first slice.
- Do not support arbitrary model-selected LSP executables or command strings.
- Do not allow LSP tools to apply file edits directly.
- Do not implement multi-root workspaces in the first slice.
- Do not enable or install arbitrary pylsp plugins from MCP tool calls.
- Do not expose raw absolute paths, raw stderr, full file contents, or secrets in
  metrics, audit events, or user-visible errors.
- Do not make the LSP module replace CodeGraph. CodeGraph remains useful for
  indexed/offline repository context; LSP handles live editor-grade lookups.

## Design Principles

1. **Capability routing over backend selection.** Tool callers should ask for
   `lsp.definition` or `lsp.diagnostics`, not choose Ruff or pylsp.
2. **Preview before mutation.** Any LSP operation that can produce edits returns
   a bounded preview. Existing filesystem tools remain the only mutation path.
3. **Policy reuse.** Tool grants determine visibility; path grants determine
   which files a request may inspect; result filtering uses the same effective
   path policy.
4. **Partial availability.** Ruff can be healthy while pylsp is missing, or the
   reverse. `lsp.status` explains the effective capability set.
5. **Single-root first.** Use one active workspace root per session manager
   entry. Multi-root can be added later without changing tool names.
6. **Direct process execution.** Start known executables with argv arrays, never
   through a shell string.
7. **Bounded outputs.** All result lists, snippets, diagnostics, preview edits,
   stderr details, and workspace symbol results have explicit limits.

## Proposed Tool Surface

The first slice exposes these tools:

- `lsp.status`
- `lsp.diagnostics`
- `lsp.document_symbols`
- `lsp.workspace_symbols`
- `lsp.definition`
- `lsp.references`
- `lsp.hover`
- `lsp.signature_help`
- `lsp.format_preview`
- `lsp.code_actions`

Profiles can grant these with `lsp.*`. Per-tool grants may still work through
the existing wildcard policy model, but the default code-oriented presets should
grant the full `lsp.*` category plus path read grants. Operators can narrow that
with explicit per-tool deny rules, for example denying `lsp.format_preview` or
`lsp.code_actions`, and deny still wins over the category allow.

### Common Request Semantics

- File paths are workspace-relative.
- Paths go through the same normalization, traversal rejection, symlink checks,
  and deny-precedence behavior as filesystem tools.
- Every file path argument requires an effective read grant.
- Workspace-wide or index-style tools require an effective read grant for the
  whole active workspace root in the first slice.
- Results that mention additional files are filtered through the current
  request/profile path policy before returning.
- Positions are zero-based LSP UTF-16 `line` and `character` offsets.
- Invalid positions return `invalid_position` with safe context.
- Response limits are explicit in tool input where useful and capped by server
  defaults.

### `lsp.status`

Returns the effective LSP health for the current workspace:

- configured workspace root;
- supported language set for this slice (`python`);
- discovered Ruff and pylsp executable path source, with absolute paths redacted
  or relativized where necessary;
- backend versions when available;
- backend process state and last health check;
- supported and missing capabilities;
- installation hints for missing optional backends;
- detected config source metadata where available, such as Ruff config files.

This tool should not require a file path, but it still requires an `lsp.*` grant.

### `lsp.diagnostics`

Uses Ruff by default. The first slice supports one workspace-relative Python
file per call. Directory, glob, and whole-repository diagnostics are deferred to
a later slice because they need additional performance and policy controls.

Returned diagnostics include:

- severity;
- code/rule id;
- message;
- relative path;
- zero-based range;
- source backend;
- fix availability metadata;
- bounded related information.

Diagnostics do not include full file contents. Any rule docs or snippets are
bounded.

### `lsp.document_symbols`

Uses pylsp. Accepts one workspace-relative Python file and returns a bounded
hierarchical or flattened symbol list:

- symbol name;
- kind;
- range and selection range;
- container name when available;
- relative path.

### `lsp.workspace_symbols`

Uses pylsp. Accepts a search query and returns a bounded list of matching
symbols visible to the current profile. Results outside granted read paths are
filtered out.

### `lsp.definition`

Uses pylsp. Accepts a file and UTF-16 position. Returns bounded target
locations. Targets outside granted paths are omitted, with a filtered count
included when safe.

### `lsp.references`

Uses pylsp. Accepts a file, UTF-16 position, an `include_declaration` flag, and
limits. Returns bounded locations filtered through current path grants.

### `lsp.hover`

Uses pylsp semantic hover by default. Ruff rule documentation may be included
only when the position maps to a Ruff diagnostic or rule. Hover text is bounded
and should not include raw unbounded source file content.

### `lsp.signature_help`

Uses pylsp. Accepts a file and UTF-16 position. Returns active signature,
parameter metadata, and bounded documentation when available.

### `lsp.format_preview`

Uses Ruff formatting. It never writes files. The public MCP response contract is
a canonical unified diff preview plus metadata. Structured LSP text edits are
best-effort supplemental data and must be returned only when the request opts in
to `include_text_edits`. Clients and baseline smoke tests should treat
`unified_diff` as the stable preview payload.

Preview output must include:

- affected relative path list;
- before hash where available;
- backend and formatter version metadata;
- text edit count;
- bounded `unified_diff`;
- optional structured `text_edits` only when requested;
- truncation flag when preview limits are hit.

If a preview would exceed configured path or byte limits, return
`preview_too_large` or a truncated preview that cannot be mistaken for a complete
patch.

### `lsp.code_actions`

Uses Ruff code actions. It never writes files and never executes arbitrary
server commands. It returns available actions and bounded proposed workspace
edits. Applying an action is a later `fs.patch` or `fs.write` operation using
the existing policy, hook, lock, and audit flow.

The first slice must reject opaque `workspace/executeCommand` actions unless
they can be represented as explicit bounded text edits. If Ruff reports only
opaque command-shaped actions for a request, return a deterministic structured
error such as `unsupported_action_shape` rather than silently omitting those
actions.

## Architecture

### Components

`LspModule`
: MCP-facing module that registers tools, validates input, applies profile/path
  policy, routes calls, records audit/metrics, and normalizes responses.

`LspSessionManager`
: Owns per-workspace long-lived backend sessions with idle timeout, startup
  health checks, graceful shutdown, restart-on-failure, and bounded stderr
  capture. The first slice uses one active root per session.

`LspBackend`
: Internal protocol for backend capabilities, status, initialization, request
  handling, shutdown, and health reporting.

`RuffBackend`
: Starts and speaks to `ruff server` for diagnostics, formatting previews, code
  action previews, organize-import previews, and rule-oriented metadata where
  supported.

`PylspBackend`
: Starts and speaks to `pylsp` for document symbols, workspace symbols,
  definitions, references, semantic hover, and signature help.

`LspCapabilityRouter`
: Maps each tool/capability to the preferred backend. It returns structured
  `capability_unavailable` or `backend_missing` errors when a backend cannot
  satisfy a request.

`LspExecutableResolver`
: Discovers executables from the project virtual environment, configured admin
  paths, and PATH. It returns command argv arrays and provenance, not shell
  strings.

### Session Identity And Policy

The process session can be workspace-scoped, but authorization is always
request-scoped. The implementation must not reuse cached result payloads across
profiles without rechecking path grants and result filtering.

The LSP process itself is also a trust boundary. A long-lived server may parse,
index, import-resolve, or cache files outside the single path returned in a tool
response. For the first slice, any workspace-wide or index-style tool such as
`lsp.workspace_symbols` requires an effective read grant for the whole active
workspace root. File-scoped tools may run in the shared workspace session, but
their requested path and returned locations still require request/profile-scoped
grant checks. Future work may add grant-scoped sandboxed views, but this spec
does not claim process-level isolation for partial path grants.

Recommended cache key inputs:

- canonical workspace root;
- backend id (`ruff`, `pylsp`);
- backend executable identity/version;
- relevant backend configuration fingerprint when available.

Profile id should not necessarily start a separate LSP process, but every
request must carry the profile context into input validation and result
filtering.

### Process Management

- Start backends with direct argv arrays.
- Do not accept executable commands from model/tool input.
- Do not use shell wrappers, `npx`, `docker exec`, `devbox run`, or similar
  wrapper unwrapping in the first slice.
- Bound startup and request timeouts.
- Capture stderr only for bounded diagnostic reporting.
- On backend crash, mark that backend unhealthy, close the process, and preserve
  other backend capabilities.
- Idle sessions shut down after a configurable timeout.
- Explicit server shutdown stops all LSP sessions.

### Optional Dependency Model

The package should expose an `lsp` extra that installs supported default Python
backends where practical:

- `ruff`
- `python-lsp-server`

Runtime discovery still works when operators install those tools themselves.
Missing dependencies are represented in `lsp.status` and capability errors.

## Data Flow

1. MCP client calls an `lsp.*` tool.
2. `LspModule` checks that the profile grants the tool, usually via `lsp.*`.
3. The module validates inputs and normalizes workspace-relative paths.
4. For file-scoped calls, the module checks effective read permission for each
   requested path.
5. The capability router selects Ruff or pylsp.
6. `LspSessionManager` starts or reuses the workspace backend session.
7. The backend sends the LSP request and returns raw LSP data.
8. The module normalizes paths, UTF-16 positions, ranges, and backend-specific
   response structures into MCP schemas.
9. Results are filtered through the current request/profile path grants.
10. Response limits are applied.
11. Audit/metrics events are emitted.
12. The normalized result or structured error is returned.

## Error Model

Use structured reason codes and safe details:

- `tool_not_granted`
- `path_denied`
- `invalid_path`
- `invalid_position`
- `backend_missing`
- `backend_unhealthy`
- `backend_timeout`
- `capability_unavailable`
- `response_truncated`
- `preview_too_large`
- `unsupported_action_shape`
- `unsupported_language`
- `workspace_not_supported`
- `config_error`

Backend errors should not include raw absolute paths, raw environment values,
secrets, or unbounded stderr. Where useful, include remediation hints such as
"install the `lsp` extra" or "pylsp is required for definition lookup".

## Policy And Security

- `lsp.*` governs tool visibility and tool-call authorization.
- Existing path grants govern all file reads and all returned file locations.
- Explicit deny still wins over allow/ask.
- Hooks participate through the standard tool-call hook path.
- LSP edit previews do not grant write permission.
- Actual mutation remains behind `fs.patch` or `fs.write`.
- LSP subprocesses are sandboxed by process/runtime constraints where available,
  but sandboxing is defense-in-depth, not a replacement for tool/path policy.
- Model requests cannot choose executables, change server config, enable plugins,
  or execute server commands.
- Operator-managed Python environments and pylsp plugins are trusted runtime
  inputs. The first slice should document this clearly.

## Observability And Audit

Each call should emit safe observability fields aligned with the existing MCP
tool reporting contract:

- tool name;
- backend selected;
- requested capability;
- stable opaque workspace id or hash, never a raw absolute path;
- relative file path when applicable;
- language;
- profile id or profile label where safe;
- elapsed time;
- result counts;
- truncation status;
- missing-capability reason;
- proposed edit generated flag;
- affected path count for previews;
- denial reason when policy blocks a request.

No raw file content, raw absolute paths, secrets, or unbounded backend stderr
should be included.

## Testing Strategy

### Unit Tests

- Capability routing chooses Ruff or pylsp correctly.
- Missing backends produce degraded capability errors.
- Profile grants hide/block `lsp.*` tools correctly.
- Path normalization rejects traversal, absolute paths, denied paths, and
  symlink escapes.
- Returned locations are filtered through profile path grants.
- UTF-16 positions are validated.
- Preview edit limits enforce max paths, max bytes, and truncation markers.
- `lsp.format_preview` returns `text_edits` only when the request opts in.
- `lsp.code_actions` returns `unsupported_action_shape` for opaque command-only
  action results.
- `lsp.workspace_symbols` requires a workspace-root read grant in the first
  slice.
- Error payloads are structured and redacted.

### Fake Backend Tests

Use deterministic fake Ruff/pylsp backends for every tool:

- diagnostics;
- symbols;
- definitions;
- references;
- hover;
- signature help;
- format previews;
- code-action previews;
- backend crash and restart behavior;
- partial availability when only one backend exists.

### Real Backend Tests

Add env-gated tests that run only when `ruff` and/or `pylsp` are available:

- create isolated on-disk Python fixtures;
- verify Ruff diagnostics for a known lint error;
- verify Ruff format preview for unformatted code;
- verify pylsp definition and document symbols for a small module;
- verify path filtering hides denied files;
- verify backend health degrades cleanly when a process exits.

### Smoke/UAT

Extend the MCP smoke harness with optional LSP scenarios:

- standalone MCP server;
- tldw-hosted MCP server;
- in-process, live HTTP, live WebSocket, and stdio where supported;
- strict mode requiring LSP backends;
- non-strict mode that reports skipped capabilities when dependencies are
  missing.

## Rollout

1. Implement the Python-only `lsp` module with fake backend tests and no direct
   file mutation.
2. Add Ruff and pylsp process backends behind runtime discovery.
3. Add profile defaults for code-oriented profiles that should receive `lsp.*`.
4. Add docs for optional installation, server status, and safe use of previews.
5. Add smoke/UAT scenarios for standalone and tldw-hosted MCP.
6. Later, consider apply-edit flows that convert preview edits into explicit
   `fs.patch` requests with approval, locks, hooks, and audit.
7. Later, add TypeScript, Go, Rust, or generic LSP configuration once the Python
   contract is proven.

## Open Questions For Implementation Planning

- Exact max defaults for diagnostics, symbol results, references, hover bytes,
  and preview bytes.
- Where to place the LSP runtime code inside `mcp_unified` so standalone and
  tldw-hosted modes share it without importing tldw-specific internals.
- How much backend config metadata can be retrieved portably from Ruff and pylsp
  without relying on unstable internals.

## Acceptance Criteria For The First Implementation Slice

- `lsp.*` tools are available only when granted by profile policy.
- File-scoped tools require effective read grants and filter returned paths.
- Workspace-wide/index tools require effective workspace-root read grants in the
  first slice.
- Ruff-backed diagnostics and previews work when Ruff is installed.
- pylsp-backed navigation works when pylsp is installed.
- Missing Ruff or pylsp reports clean degraded capabilities.
- Formatting and code actions never mutate files directly.
- `format_preview` exposes `unified_diff` as the stable payload and returns
  `text_edits` only when requested.
- Opaque command-shaped code actions fail with `unsupported_action_shape`.
- Results and errors are bounded and redact absolute paths/secrets.
- Fake backend tests cover every tool and error family.
- Env-gated real backend tests cover Ruff and pylsp happy paths.
- MCP smoke/UAT can exercise LSP tools against standalone and tldw-hosted MCP.
