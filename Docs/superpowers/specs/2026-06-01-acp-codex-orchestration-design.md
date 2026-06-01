# Codex ACP Adapter and App-Server Orchestration Design

Date: 2026-06-01
Status: Draft for spec review
Tracking: TASK-582

## Summary

tldw_server should support Codex through two deliberately separate paths:

1. A first-class external ACP adapter path using
   `zed-industries/codex-acp` as the initial Codex integration.
2. A later Codex app-server backend for deeper orchestration features that ACP
   does not naturally expose.

The generic reusable harness adapter still matters, but it should be the
fallback for agents that do not provide ACP or an app-server-like control plane.
It should not be the first Codex implementation path while `codex-acp` and the
official Codex app-server exist.

The design therefore extends the agent registry and runtime model around a
clear backend taxonomy, explicit setup/certification states, normalized event
records, and workspace-owned session metadata. This keeps tldw_server honest
about what is native ACP, what is an external adapter, what is a deeper Codex
control backend, and what is only a documented candidate.

## Context and Current Repo Shape

The current seeded Codex registry row is conservative. In
`tldw_Server_API/Config_Files/agents.yaml`, Codex is still
`documented_unverified`, `documented_only`, and blocked by
`adapter_required`; its `entrypoint_strategy` is `documented_candidate`, and no
`acp_command` is configured. That is correct for the current shipped evidence,
but it does not represent the available `codex-acp` path.

The Python registry already has a deterministic entrypoint classifier in
`tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py`. It currently
recognizes `native_acp`, `adapter_acp`, `documented_candidate`, and
`custom_template`. It also stores `acp_command`, `acp_args`, `adapter_source`,
`adapter_docs_url`, and `certification_blocker`. This is close to the needed
shape, but `adapter_acp` is too vague. The active canonical term should become
`external_acp_adapter`; any old `adapter_acp` value should be accepted only as a
legacy import/DB compatibility alias and never emitted in current UI, API, or
docs.

The Go runner at `tools/tldw-agent/internal/acp/runner.go` assumes every
configured agent session launches a downstream ACP process and keeps a
`downstream *Conn`. `session/new` builds a `config.AgentConfig` from
`agentEntry.Command`, `agentEntry.Args`, and env, launches it, calls
downstream `initialize`, then forwards ACP `session/new`. That is a good shape
for native ACP and external ACP adapters. It is not the right shape for the
future Codex app-server backend, because app-server has a different JSON-RPC
protocol and richer thread/turn/item semantics.

The Go config model in `tools/tldw-agent/internal/config/config.go` only stores
`RegisteredAgent.Command`, `Args`, `Env`, and `RequiresAPIKey`. It does not yet
carry strategy, adapter, credential, or certification metadata into the runner.

Relevant local references:

- `tldw_Server_API/Config_Files/agents.yaml`
- `tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py`
- `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py`
- `tools/tldw-agent/internal/config/config.go`
- `tools/tldw-agent/internal/acp/runner.go`
- `Docs/superpowers/specs/2026-05-12-acp-downstream-entrypoint-strategy-design.md`
- `Docs/Product/ACP_Agent_Orchestration_PRD.md`
- `Docs/Development/ACP_Compatibility_Matrix.md`

## Prior-Art Review

The projects reviewed point to the same pattern: remote agent orchestration is
not just "run a CLI and scrape output." Mature flows need sessions, approvals,
streaming events, worktree ownership, resumability, and setup transparency.

- `formulahendry/acp-ui` is the strongest ACP-specific comparator. It treats
  ACP agents as configured process or WebSocket transports, includes traffic
  inspection, supports permissions, and lists Codex through
  `@zed-industries/codex-acp`.
- `zed-industries/codex-acp` is a real ACP adapter around Codex CLI. Its README
  says it supports context mentions, images, tool calls with permission
  requests, edit review, TODO lists, slash commands, client MCP servers, and
  ChatGPT/Codex/OpenAI API key auth modes. Its latest listed release during
  review was `0.15.0` on 2026-05-22.
- `slopus/happy` and `The-Vibe-Company/companion` show that remote control
  products need encrypted or authenticated remote access, push/notification
  paths for approvals and errors, and the ability to take control from another
  device.
- `milisp/codexia` and `jamesrochabrun/AgentHub` show that worktree management
  and session grouping are not secondary features. They are part of serious
  multi-agent work.
- `OpenJeDi/codex-control` explicitly points to the official Codex app-server
  protocol and calls out that account/auth state failures can look like app
  failures.
- `friuns2/codex-mobile` reinforces that mobile/remote control introduces
  lifecycle constraints such as background process survival, tunnels, and
  HTTPS/private-network setup.

Official Codex app-server documentation says app-server is the interface Codex
uses for rich clients, including authentication, conversation history,
approvals, and streamed agent events. It supports stdio, WebSocket, Unix socket,
and off transports. WebSocket is experimental and unsupported; non-loopback
listeners can be unauthenticated by default unless WebSocket auth is configured.
The same docs expose first-class `thread/start`, `thread/resume`,
`thread/list`, `turn/start`, `turn/steer`, `turn/interrupt`, `model/list`,
MCP status, config import, filesystem, approvals, and account APIs.

Sources:

- https://github.com/formulahendry/acp-ui
- https://github.com/zed-industries/codex-acp
- https://github.com/slopus/happy
- https://github.com/The-Vibe-Company/companion
- https://github.com/milisp/codexia
- https://github.com/OpenJeDi/codex-control
- https://github.com/friuns2/codex-mobile
- https://github.com/jamesrochabrun/AgentHub
- https://developers.openai.com/codex/app-server
- https://developers.openai.com/codex/config-reference

## Goals

1. Make Codex support honest and actionable by distinguishing external ACP
   adapter support from generic CLI harnessing.
2. Represent adapter-backed agents in the registry without overclaiming native
   ACP support.
3. Keep `agent/list`, setup guides, status cards, and certification helpers
   strategy-aware.
4. Avoid runtime probes that fetch or execute mutable remote packages such as
   `npx @latest` during passive listing.
5. Preserve tldw_server's workspace model: workspaces, worktrees, MCP, ACP,
   sandboxes, approvals, and traceability are part of one agent orchestration
   surface.
6. Create a later path to Codex app-server integration without forcing
   app-server into ACP-shaped abstractions.
7. Leave a reusable generic runner-adapter model for agents that have no ACP
   or app-server bridge.

## Non-Goals

- Do not implement Codex app-server in the first slice.
- Do not implement a custom Codex `exec --json` ACP shim before trying
  `codex-acp`.
- Do not treat `codex mcp-server` as ACP.
- Do not claim Codex is `supported_with_caveats` or `live_e2e_tested` until
  live certification passes.
- Do not expose `adapter_acp` as a current public label after this design is
  implemented.
- Do not install third-party packages during passive registry/status listing.
- Do not redesign the Research Workspace UI in this task.

## Runtime Taxonomy

Use a small set of explicit strategies. These names should be stable product
metadata, not marketing labels.

| Strategy | Meaning | Example |
| --- | --- | --- |
| `native_acp` | The downstream process speaks ACP directly over stdio. | `opencode acp`, `goose acp`, `hermes acp --accept-hooks` |
| `external_acp_adapter` | A separate installed adapter speaks ACP and controls the target agent. | `codex-acp` controlling Codex CLI |
| `codex_app_server` | tldw connects to `codex app-server` over its JSON-RPC protocol and maps events into tldw's orchestration model. | Future deep Codex backend |
| `runner_adapter` | tldw owns an adapter around a non-ACP/non-app-server CLI, SDK, or API. | Future fallback for unsupported tools |
| `documented_candidate` | Setup exists, but no compatible runtime entrypoint has been selected or certified. | Codex before `codex-acp`, Aider until an entrypoint exists |
| `custom_template` | Operator must provide a concrete command, env, policy, and evidence. | Custom profiles |

`native_acp` and `external_acp_adapter` can reuse the existing ACP downstream
runner shape. `codex_app_server` and `runner_adapter` need a broader backend
interface because they do not necessarily speak ACP downstream.

## Registry and Config Model

The canonical seeded Codex row should move from documented candidate to an
external adapter candidate only when the product is ready to expose setup and
health semantics for the adapter.

Proposed active YAML shape:

```yaml
- type: codex
  name: OpenAI Codex CLI
  description: "OpenAI's coding agent via the Codex ACP adapter"
  command: codex
  args: []
  env: {}
  requires_api_key: null
  install_instructions:
    - "npm install -g @openai/codex"
    - "Install zed-industries/codex-acp from a pinned release, then ensure codex-acp is on PATH"
  docs_url: "https://github.com/openai/codex"
  support_state: experimental
  verification_level: documented_only
  compatibility_notes: "Codex can be reached through zed-industries/codex-acp, but this tldw profile remains experimental until live ACP certification passes."
  compatibility_docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md"
  entrypoint_strategy: external_acp_adapter
  acp_command: codex-acp
  acp_args: []
  adapter_source: zed-industries/codex-acp
  adapter_docs_url: "https://github.com/zed-industries/codex-acp"
  adapter_package: "@zed-industries/codex-acp"
  adapter_version_policy: pinned_release_required
  credential_policy: delegated_to_adapter
  certification_blocker: live_certification_required
```

Important details:

- `command: codex` remains the user-facing/downstream agent command.
- `acp_command: codex-acp` is the command the ACP runner launches.
- `requires_api_key` should not be hard-coded to `OPENAI_API_KEY` for Codex
  adapter readiness. `codex-acp` can use ChatGPT subscription, `CODEX_API_KEY`,
  or `OPENAI_API_KEY`; setup/status should present this as delegated adapter or
  Codex auth state, not a single env-var requirement.
- `npx @latest` can appear in exploratory docs, but the runtime registry should
  prefer an installed binary or a pinned adapter version. Passive health/listing
  must not launch `npx @latest`.
- `support_state: experimental` means the product knows a plausible adapter
  path. It is not the same as live support.
- `verification_level` should move to `stub_smoke_tested` or
  `live_e2e_tested` only after evidence exists.

Schema changes needed later:

- Add canonical `external_acp_adapter` to the Python strategy literal and
  emitted API/schema types.
- Read old `adapter_acp` as an internal legacy import value only. Store and
  emit `external_acp_adapter`.
- Extend dynamic registry storage with structured adapter fields if a flat
  column set becomes too cramped:
  - `adapter_source`
  - `adapter_docs_url`
  - `adapter_package`
  - `adapter_version`
  - `adapter_version_policy`
  - `adapter_install_source`
  - `credential_policy`
  - `runtime_backend`
- Extend Go `RegisteredAgent` with strategy, ACP command, adapter fields, and
  credential policy so runner behavior is not inferred from the display
  command.

## Readiness and Health Semantics

Do not collapse readiness into `is_configured`.

Expose separate signals:

- `binary_found`: the display/downstream command exists, for example `codex`.
- `adapter_found`: the ACP adapter command exists, for example `codex-acp`.
- `credential_state`: `ready`, `missing`, `delegated`, or `unknown`.
- `entrypoint_strategy`: one of the canonical strategy names.
- `probe_state`: `ready_to_probe`, `blocked`, `custom_template`,
  `documented_only`, or `unsupported_backend`.
- `primary_blocker`: one stable blocker code.
- `blockers`: all known stable blocker codes.
- `support_state`: release support posture.
- `verification_level`: evidence level.
- `last_certified_at`: timestamp when a live certification matrix passed.
- `last_certification_artifact`: path or URL to the evidence record when
  available.

For `external_acp_adapter`, passive readiness checks may use `shutil.which` or
the Go equivalent for both `command` and `acp_command`. They must not start the
agent, contact OpenAI, or install packages.

A bounded active health check can launch `codex-acp`, send ACP `initialize`,
then terminate. That check belongs in an explicit certification or health action,
not passive `agent/list`.

## Runner Architecture

### First slice: external ACP adapter

The existing Go runner can handle `codex-acp` if the agent entry resolves to
the adapter command:

```text
tldw WebUI/API
  -> tldw-agent-acp
  -> tools/tldw-agent ACP runner
  -> codex-acp over ACP stdio
  -> Codex CLI / Codex auth / Codex tools
```

The runner should launch `acp_command + acp_args` for `native_acp` and
`external_acp_adapter`. It should not launch `command + args` unless that is
explicitly the ACP entrypoint.

The session record should capture:

- `agent_type`
- `runtime_backend: acp_downstream`
- `entrypoint_strategy`
- `command` and `acp_command`
- `adapter_source` and adapter version when available
- `cwd`
- workspace id, if launched from a workspace
- worktree id/path/branch/owner, if applicable
- sandbox policy snapshot
- MCP policy snapshot
- approval policy snapshot

### Later slice: Codex app-server backend

Codex app-server should be modeled as a separate backend, not as a fake ACP
downstream agent.

```text
tldw WebUI/API
  -> Agent orchestration service
  -> Codex app-server client
  -> codex app-server over stdio, Unix socket, or authenticated WebSocket
  -> Codex thread/turn/item APIs
```

This backend should map app-server primitives into tldw's normalized session
model:

- App-server `thread` -> tldw agent session.
- App-server `turn` -> tldw run/interaction.
- App-server `item` -> tldw event/artifact/tool/action record.
- App-server approval requests -> tldw approval requests.
- App-server `thread/list`, `thread/read`, and `thread/resume` -> tldw resume
  and session continuity.

The Codex app-server backend should preserve app-server features that ACP does
not naturally cover:

- model/provider listing
- account/auth state
- rate-limit state
- ChatGPT login/device code flow
- app/plugin/MCP server status
- config import from external agents
- thread archive/fork/rollback/compact
- filesystem APIs where policy allows
- detailed command and file-change approvals

## Normalized Event Model

tldw needs one event envelope for UI, audit, diagnostics, and future backends.
The envelope should be broader than ACP but mappable from ACP.

```json
{
  "event_id": "evt_...",
  "session_id": "sess_...",
  "turn_id": "turn_...",
  "item_id": "item_...",
  "backend": "acp_downstream | codex_app_server | runner_adapter",
  "agent_type": "codex",
  "workspace_id": "ws_...",
  "worktree_id": "wt_...",
  "kind": "turn.started",
  "status": "in_progress",
  "payload": {},
  "created_at": "2026-06-01T00:00:00Z"
}
```

Core event kinds:

- `session.created`
- `session.resumed`
- `session.closed`
- `turn.started`
- `turn.steered`
- `turn.interrupted`
- `turn.completed`
- `item.started`
- `item.delta`
- `item.completed`
- `tool.requested`
- `tool.started`
- `tool.output_delta`
- `tool.completed`
- `approval.requested`
- `approval.resolved`
- `file_change.proposed`
- `file_change.applied`
- `file_change.declined`
- `model.changed`
- `auth.changed`
- `mcp.status_changed`
- `runtime.error`

ACP notifications and downstream requests can be translated into this envelope.
Codex app-server events map more directly because app-server already has
thread/turn/item semantics.

## Approvals, Security, and Remote Control

Approval handling must be centralized. ACP permission requests and Codex
app-server approvals should share the same tldw approval service and audit
model.

Approval request records should include:

- `session_id`
- `turn_id`
- `item_id` when available
- `backend`
- `agent_type`
- `workspace_id`
- `worktree_id`
- `approval_kind`: command, file change, network, MCP tool, dynamic tool,
  user input, other
- proposed command/path/network destination/tool name
- policy snapshot fingerprint
- available decisions
- final decision
- resolver identity
- created/resolved timestamps

For Codex app-server, map command approvals, network approvals, file-change
approvals, MCP tool approvals, and `tool/requestUserInput` into this same model.
For ACP, map `session/request_permission` and any downstream permission shape
into the same model.

Remote transport policy:

- Prefer stdio or Unix socket for local integrations.
- Treat WebSocket as experimental for Codex app-server until the upstream docs
  mark it stable.
- Never expose non-loopback WebSocket without explicit auth configuration.
- Prefer token files or verifier hashes over raw command-line bearer tokens.
- Redact tokens, API keys, full auth URLs, and sensitive command env in logs.
- Show clear diagnostics for auth/account failures because they can look like
  agent failures.

## Workspaces, Worktrees, MCP, and Sandboxes

The workspace model is not only document storage. It also owns agent/tool
execution context. A useful agent session must know which workspace, worktree,
MCP server set, and sandbox policy it belongs to.

Session creation should therefore require or derive:

- workspace id
- repo root
- cwd
- worktree path
- branch name
- owner/source of the worktree
- MCP profile or server set
- sandbox runtime/profile
- network policy
- approval policy
- agent runtime backend

This is what lets Research Workspace, Shared Workspaces, MCP Hub, ACP, and
sandbox diagnostics converge instead of becoming disconnected product surfaces.

## Generic Runner Adapter Model

The reusable harness adapter remains useful for agents without ACP or an
app-server-like protocol. It should be designed after the external ACP adapter
slice so it can reuse the same registry/status/session/event/approval model.

Runner adapters should have a small interface:

```text
Probe(ctx) -> RuntimeProbe
StartSession(ctx, SessionRequest) -> SessionHandle
SendTurn(ctx, SessionHandle, TurnRequest) -> TurnHandle
StreamEvents(ctx, SessionHandle) -> EventStream
RequestInterrupt(ctx, TurnHandle) -> InterruptResult
CloseSession(ctx, SessionHandle) -> CloseResult
```

This interface should not promise full parity with ACP or Codex app-server.
Each adapter reports capabilities:

- supports resume
- supports steering
- supports interrupt
- supports approval requests
- supports file-change previews
- supports MCP pass-through
- supports model selection
- supports worktree ownership
- supports artifact export

Adapters that only offer line-oriented CLI output can still integrate, but the
UI should display reduced capability states rather than pretending they support
rich event streams.

## API and UI Implications

The API should expose a singular agent runtime status envelope for registry and
session surfaces. It should avoid spreading separate status meanings across
unrelated endpoints.

Suggested status fields:

```json
{
  "agent_type": "codex",
  "display_name": "OpenAI Codex CLI",
  "runtime_backend": "acp_downstream",
  "entrypoint_strategy": "external_acp_adapter",
  "support_state": "experimental",
  "verification_level": "documented_only",
  "readiness": {
    "state": "blocked",
    "binary_found": true,
    "adapter_found": false,
    "credential_state": "delegated",
    "primary_blocker": "adapter_missing",
    "blockers": ["adapter_missing", "live_certification_required"]
  },
  "adapter": {
    "source": "zed-industries/codex-acp",
    "docs_url": "https://github.com/zed-industries/codex-acp",
    "package": "@zed-industries/codex-acp",
    "version_policy": "pinned_release_required"
  },
  "certification": {
    "last_certified_at": null,
    "evidence_url": "/docs-static/Development/ACP_Compatibility_Matrix.md"
  }
}
```

UI language should be concrete:

- "Codex via codex-acp adapter"
- "Adapter not installed"
- "Codex auth is handled by the adapter or Codex account state"
- "Live certification has not passed on this host"
- "Run health check" for active probe actions

Avoid:

- "Codex supported" before certification
- "Configured" as the only status
- "ACP adapter" without naming the adapter
- "OpenAI API key missing" when Codex may use ChatGPT or Codex-specific auth

## Staged Roadmap

### Stage 0: Spec and review

Write this spec, review it, and get user approval before implementation
planning.

Success criteria:

- Spec captures external ACP adapter, app-server, and runner-adapter boundaries.
- Spec cites verified repo context and reviewed prior art.
- Spec review loop passes.

### Stage 1: Registry and status model

Add canonical `external_acp_adapter` strategy support across Python registry,
DB persistence, API schemas, frontend types, readiness utilities, and docs.

Success criteria:

- Existing `adapter_acp` values are imported as `external_acp_adapter` but never
  emitted.
- `agent/list` and setup/status surfaces separate display binary, adapter
  command, credential state, support state, and certification evidence.
- Passive listing does not launch agents or install packages.

### Stage 2: Codex ACP adapter profile and certification

Add Codex's adapter-backed profile using `codex-acp`, then certify it through
the existing ACP runner path.

Success criteria:

- The Codex row identifies `zed-industries/codex-acp`.
- Runtime launches `acp_command` for adapter-backed profiles.
- Setup docs require a pinned adapter release or installed binary.
- Live certification evidence records initialize, session create, prompt,
  permission/approval behavior if reachable, cancel, close, and failure modes.

### Stage 3: Workspace-aware session diagnostics

Thread adapter-backed Codex sessions through workspace/worktree/MCP/sandbox
diagnostics.

Success criteria:

- Session records show workspace id, worktree path/branch, sandbox profile, MCP
  profile, adapter source, and certification state.
- Research Workspace and Shared Workspaces can link to the same session history
  and diagnostics without inventing separate agent concepts.

### Stage 4: Codex app-server backend design and implementation

Design and implement a separate Codex app-server backend using the official
thread/turn/item protocol.

Success criteria:

- Supports initialize/initialized handshake.
- Supports local stdio first; Unix socket and authenticated WebSocket follow.
- Maps thread/turn/item events into the normalized tldw event envelope.
- Supports account state, model list, MCP status, approval flows, resume/list,
  and interrupt.
- Does not pretend app-server is ACP.

### Stage 5: Generic runner adapter fallback

Implement the reusable fallback adapter interface for non-ACP agents.

Success criteria:

- Adapters report capability gaps explicitly.
- Reduced-capability agents do not appear equivalent to ACP/app-server agents.
- The same event, approval, diagnostics, and workspace ownership model is reused.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| `codex-acp` version drift breaks behavior. | Pin a release in setup/certification docs; store adapter version in evidence. |
| Runtime accidentally invokes `npx @latest`. | Runtime registry should use installed commands or pinned package references only. |
| Codex auth modes are misrepresented. | Use `credential_policy: delegated_to_adapter`; app-server can later expose real account state. |
| App-server is forced into ACP abstractions. | Treat it as `codex_app_server` backend with its own client and event mapping. |
| Remote control exposes unauthenticated agent access. | Require auth for non-loopback WebSocket; prefer stdio/Unix socket. |
| UI overclaims support. | Separate support state, verification level, passive readiness, and active certification. |
| Worktree/session ownership becomes ambiguous. | Persist workspace, repo, worktree, branch, backend, and policy snapshots on session create. |
| Generic runner adapter becomes a brittle CLI scraper. | Make it capability-reported fallback only, not the primary Codex path. |

## First Implementation Slice Acceptance Criteria

The first implementation slice should be considered complete when:

- `external_acp_adapter` is the canonical active strategy in backend, frontend,
  docs, and seeded config.
- Any legacy `adapter_acp` value is read internally as
  `external_acp_adapter`, but not emitted.
- The Codex registry row names `zed-industries/codex-acp` and remains
  experimental until live certification passes.
- Passive registry/listing reports adapter readiness without launching or
  installing packages.
- The runner launches `acp_command + acp_args` for `native_acp` and
  `external_acp_adapter` strategies.
- Tests cover strategy coercion, readiness classification, API serialization,
  Go config parsing, and no-network/no-`npx @latest` passive listing behavior.
- Documentation explains Codex ACP adapter setup, credential delegation, and
  certification state.

## Open Questions for Implementation Planning

1. Which exact `codex-acp` version should the first certification pin?
2. Should the initial setup guide recommend release binary installation first,
   npm package installation first, or both?
3. Should app-server live under ACP navigation as an agent backend, or under a
   broader Agent Orchestration surface with ACP as one backend?
4. How much of Codex app-server's account/login surface should be exposed in
   tldw_server versus documented as external Codex setup?
5. Which existing session/event tables should own the normalized event envelope,
   and what retention/redaction policy should apply?
