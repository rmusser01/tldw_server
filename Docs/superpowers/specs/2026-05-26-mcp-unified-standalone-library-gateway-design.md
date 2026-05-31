# MCP Unified Standalone Library And Gateway Design

Date: 2026-05-26
Status: Draft for spec review
Backlog: TASK-480

## Summary

MCP Unified should be modularized through a strangler extraction: first into a reusable MCP runtime library, then into a standalone gateway product built on that library.

The library phase must preserve `tldw_server` behavior through compatibility shims and host adapters while creating a clean package API that has no dependency on `tldw_Server_API`. The gateway phase then uses that package to provide a ready-to-run MCP server with profiles, policy enforcement, external MCP server lifecycle management, and transport options for HTTP, WebSocket, and stdio clients.

The initial profile system should focus on enforceable governance. A profile is the action boundary a front-end or agent client selects. Full agent-role metadata such as prompts, model defaults, memory settings, and UI labels may be stored as extension metadata, but Phase A enforcement covers tool access, capabilities, path scopes, approval behavior, credential grants, external MCP access, and audit provenance.

## Goals

- Extract a reusable MCP runtime package without breaking existing `tldw_server` routes, imports, env vars, JSON-RPC behavior, MCP Hub policy behavior, or tests.
- Make the package usable by other Python applications as an embeddable runtime.
- Provide clean adapter interfaces for auth, RBAC, rate limits, metrics, audit, path scopes, credential brokering, profile resolution, approval evaluation, and module config loading.
- Add optional built-in SQLite-backed stores for standalone use while allowing hosts to replace persistence.
- Support external MCP server registry, lifecycle, health checks, reconnects, discovery refresh, and policy-gated calls.
- Support a standalone gateway after the package seam is proven.
- Treat role-like front-end modes as profile presets, not hardcoded runtime roles.

## Non-Goals

- No one-shot rewrite of all MCP modules.
- No immediate migration of every `tldw_server` domain tool into the standalone package.
- No split-brain governance in `tldw_server`; MCP Hub remains the authoritative policy and credential store there.
- No package-manager or marketplace feature for installing third-party MCP servers in the first library phase.
- No full agent platform in Phase A. Prompt/persona/model/memory role metadata is extension data for later front-end and gateway use.
- No fail-open execution behavior when policy, credential, approval, or path-scope dependencies are unavailable.

## Phase Boundaries And Acceptance Criteria

The work should be explicitly split into a library phase and a gateway phase.

### Phase A: Embeddable Runtime Library

Acceptance criteria:

- A new package boundary exposes the runtime core, profile schema/resolver interfaces, external server manager, and transport adapters.
- The package has no imports from `tldw_Server_API`.
- Existing `tldw_Server_API.app.core.MCP_unified` imports still work through shims.
- Existing `/api/v1/mcp/*` routes, JSON-RPC payloads, auth behavior, env vars, MCP Hub policy behavior, approvals, path scopes, and credential brokering remain compatible in `tldw_server`.
- Existing clients that do not select a profile continue to resolve through the current default MCP Hub/AuthNZ/RBAC behavior in `tldw_server`; extraction must not unexpectedly deny or grant tools for no-profile legacy requests.
- A minimal external FastAPI app can mount the package router with fake adapters and successfully call status, tools/list, and tools/call for a stub tool.
- Runtime-neutral Stage 2 work is split into explicit sub-stages. Profile schema/store/resolver primitives are not treated as execution enforcement until structured profile/effective-policy resolution, reason codes, audit provenance, and host compatibility tests exist.
- Profile and effective-policy resolution returns structured outcomes with machine-readable reason codes and provenance; execution code must not infer deny reasons from `None` alone.
- The optional built-in SQLite storage provides separate `ProfileStore`, `ExternalRegistryStore`, and `AuditStore` interfaces for profiles/assignments/approval policies/credential grants, external server definitions, provenance, and audit events.
- File/YAML data is limited to preset loading, seed/import/export, or read-only configuration, not concurrent policy persistence.
- External-server stdio is supported in Phase A only as a managed upstream MCP server transport; the client-facing gateway stdio MCP server entrypoint belongs to Phase B.
- The core runtime is instance-scoped. `tldw_server` may keep a singleton compatibility shim, but the package API must allow multiple isolated runtime instances in one process.
- The package declares its dependency extras and license/publishing decision before release.

### Phase B: Standalone Gateway

Acceptance criteria:

- A runnable gateway entrypoint exposes the same runtime through FastAPI HTTP/WebSocket.
- A client-facing stdio MCP server entrypoint exposes the same profile and policy semantics to local agent clients.
- The gateway can load built-in profile presets, duplicate them into user profiles, and resolve effective policy from its local SQLite store.
- The gateway can manage configured external MCP servers, including stdio process lifecycle, health, reconnects, discovery refresh, and circuit breaker state.
- Gateway execution uses the same policy, approval, credential, path-scope, audit, and metrics interfaces as embedded hosts.

Phase B must not begin until Phase A proves the import boundary and `tldw_server` compatibility.

## Current Repo Foundation

MCP Unified already contains a production-ready runtime surface:

- JSON-RPC protocol and request handling in `tldw_Server_API/app/core/MCP_unified/protocol.py`
- WebSocket and HTTP server behavior in `server.py` plus `mcp_unified_endpoint.py`
- Module lifecycle and registry in `modules/base.py` and `modules/registry.py`
- External MCP federation under `external_servers/`
- Command runtime under `command_runtime/`
- Governance packs under `governance_packs/`
- MCP Hub policy, approval, external access, path scope, and credential broker services under `tldw_Server_API/app/services/`
- A large existing MCP test set under `tldw_Server_API/app/core/MCP_unified/tests` and `tldw_Server_API/tests/MCP_unified`

The feasibility risk is coupling. Several runtime files import `tldw_Server_API` directly for AuthNZ, DB paths, telemetry, feature flags, circuit breakers, app lifecycle, MCP Hub policy services, managed secret brokers, and domain modules. The extraction must start by converting those direct dependencies into host adapter seams.

## Recommended Approach

Use a strangler extraction.

1. Add adapter interfaces in the current MCP tree.
2. Convert current direct dependencies to default `tldw_server` adapters.
3. Move runtime-neutral code into the standalone package.
4. Keep `tldw_Server_API.app.core.MCP_unified` as a compatibility shim.
5. Build the standalone gateway on the clean package API.

This preserves strict compatibility for the host app while allowing the package API to be clean and forward-looking.

## Packaging, Licensing, And Dependency Budget

The standalone package cannot inherit the full root dependency surface. It should define a small core install plus optional extras.

Core package dependencies should be limited to runtime essentials such as:

- Python `>=3.10`
- Pydantic models/settings if needed for schema validation
- lightweight async utilities from the standard library where possible

Suggested extras:

- `fastapi`: FastAPI router, WebSocket server support, HTTP test client helpers
- `sqlite`: bundled SQLite stores and migrations
- `federation`: external MCP transports and lifecycle dependencies
- `gateway`: all dependencies needed for the standalone HTTP/WebSocket and stdio gateway
- `dev`: tests, linting, type checking, import-boundary checks

The repository has conflicting license signals in local guidance and package metadata. Before publishing or encouraging third-party embedding, the implementation plan must verify the canonical repo license from authoritative project files and maintainer intent, then make an explicit standalone-package licensing decision:

- keep the standalone package under the verified repo license and document implications for downstream users, or
- obtain the needed project approval for a different library license such as LGPL or a permissive license

Until that decision is made, the package should be treated as an in-repo/internal package even if its technical boundary is clean.

The project must not market, document, or tag `mcp_unified` as a standalone third-party package until the packaging gate is complete. Early in-repo users may depend on the boundary, but release notes and examples should describe it as internal/experimental until minimal-install, extras, and license metadata are proven.

Packaging acceptance criteria:

- `mcp_unified` can be installed without installing media ingestion, RAG, UI, STT/TTS, or other `tldw_server` optional dependency groups
- extras are documented and tested independently
- import-boundary tests run in a minimal environment
- license metadata is explicit in the package metadata and docs
- a CI or local smoke install verifies `mcp_unified` imports in a clean environment with only declared core dependencies

## Runtime Instance Model

The package runtime should be instance-scoped by design.

`MCPRuntime` or equivalent should own:

- settings
- module/tool registry
- profile resolver
- external server manager
- idempotency manager
- approval, credential, path-scope, audit, metrics, RBAC, and rate-limit adapters
- transport session state

Host applications can construct multiple isolated runtime instances in the same Python process for tests, tenants, workspaces, or separate mounted routers. No standalone package component should require process-global singleton state for normal operation.

For compatibility, `tldw_Server_API.app.core.MCP_unified.get_mcp_server()` may remain as a shim that returns the existing singleton facade. That shim should wrap or delegate to an instance-scoped runtime internally, and new package examples should not teach singleton usage.

## Legacy And No-Profile Compatibility

Existing `tldw_server` clients may call MCP routes without selecting any new profile or front-end mode. Those requests must keep their current semantics during extraction.

In `tldw_server`, a no-profile request resolves policy through the existing host stack:

1. authenticated identity and API-key scopes
2. current MCP Hub default/group/persona policy resolution when policy context metadata is present
3. existing AuthNZ RBAC tool permission checks
4. existing path-scope, approval, and credential broker behavior when those features are enabled

If no MCP Hub policy context is enabled for a legacy request, the package adapter should represent this as a host-managed legacy policy context rather than as an absent standalone profile. The runtime must still apply host RBAC and API-key ceilings exactly as it does today.

Standalone gateway behavior is different: a gateway request without an explicit profile should use a configured default profile. If no default profile exists, discovery may report no executable tools and execution must deny with a clear `profile_required` reason.

Compatibility tests must prove:

- legacy no-profile `tldw_server` requests keep the same allow/deny outcomes as the pre-extraction path
- profile-selected requests apply the new profile resolver path
- no-profile standalone gateway requests deny execution when no default profile is configured
- adding a default gateway profile changes only gateway behavior, not `tldw_server` compatibility behavior

## Package Boundary

Recommended package layout:

```text
mcp_unified/
  runtime/
    protocol.py
    registry.py
    execution.py
    context.py
    idempotency.py
    errors.py
  profiles/
    models.py
    store.py
    resolver.py
    presets.py
    migrations.py
  federation/
    registry.py
    manager.py
    transports/
      base.py
      stdio.py
      websocket.py
  transports/
    fastapi.py
    stdio.py
  interfaces/
    auth.py
    rbac.py
    rate_limit.py
    metrics.py
    audit.py
    credentials.py
    path_scope.py
    approvals.py
    module_config.py
  storage/
    sqlite.py
    migrations.py
```

The standalone package must not import `tldw_Server_API`. This should be enforced by an import-boundary test or CI check.

`tldw_server` should provide host adapters under its own tree, for example:

```text
tldw_Server_API/app/core/MCP_unified/adapters/
  authnz.py
  mcp_hub_policy.py
  mcp_hub_approvals.py
  mcp_hub_credentials.py
  path_scopes.py
  telemetry.py
  module_config.py
```

Storage interfaces should be split by responsibility:

- `ProfileStore`: profile documents, profile schema version, profile lifecycle timestamps, and profile migrations
- `ProfileAssignmentStore`: principal/workspace/default-profile assignments and assignment provenance
- `ApprovalPolicyStore`: reusable approval policies and profile-to-approval-policy bindings
- `CredentialGrantStore`: credential grant metadata and broker binding references, never secret values
- `ExternalRegistryStore`: external server definitions, credential slot metadata, enabled/disabled state, registry migrations
- `AuditStore`: append-only audit/provenance events and query helpers

The bundled SQLite implementation can implement all of these interfaces for the standalone gateway, but host integrations may provide each interface independently. `tldw_server` should adapt MCP Hub and AuthNZ repositories into these interfaces rather than routing all persistence through a generic profile store. If an early package slice provides only `ProfileStore`, it is a profile-document primitive, not the full standalone persistence contract.

## Protocol Compatibility

The package must define its MCP protocol compatibility contract explicitly.

Initial compatibility target:

- JSON-RPC 2.0 request/response behavior matching the current MCP Unified implementation
- protocol version `2024-11-05` unless implementation planning chooses an intentional upgrade
- method support for `initialize`, `ping`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`, `modules/list`, and `modules/health`
- existing `tldw_server` HTTP and WebSocket mapping for `/api/v1/mcp/*`

Protocol compatibility tests must cover:

- initialize response shape and negotiated protocol version
- JSON-RPC error codes and notification behavior
- batch request handling
- tool list and tool call behavior under allowed, denied, approval-required, and unavailable-policy states
- resource and prompt discovery behavior
- HTTP, WebSocket, and later client-facing stdio transport parity for equivalent JSON-RPC messages

Any future protocol-version upgrade must be handled as negotiation or an explicit compatibility break, not as an incidental extraction side effect.

## Runtime Data Flow

```text
client request
  -> transport adapter (FastAPI HTTP/WS or stdio)
  -> auth/session adapter builds RequestContext
  -> runtime protocol validates JSON-RPC
  -> profile resolver returns ProfileResolutionResult
  -> effective-policy resolver computes EffectivePolicyResult
  -> RBAC/API-key ceilings are applied
  -> registry resolves local or external tool
  -> path, credential, approval, and risk hooks run
  -> module or external MCP server executes
  -> audit and metrics hooks record outcome
  -> transport returns MCP response
```

Discovery and execution must be separate authority decisions.

- Discovery may show unavailable or blocked tools with reason metadata when the host allows that visibility.
- Execution must always require profile policy, host RBAC, API-key ceilings, path scope, approval, and credential checks.
- Catalogs and profile filters shape visibility and UX, not execution authority by themselves.

## Profiles And Modes

A front-end mode maps to an enforceable MCP profile. The core profile fields are:

```text
id
name
description
schema_version
preset_id
preset_version
enabled
policy_document
approval_policy
path_scopes
external_server_grants
credential_grants
metadata
provenance
created_at
updated_at
```

`policy_document` should support:

```text
allowed_tools
denied_tools
capabilities
denied_capabilities
tool_patterns
module_patterns
risk_classes
resource_constraints
```

Profiles need schema and preset versioning from the start. Built-in presets can evolve, but duplicated user profiles must record the preset version they were created from. Migrations must be explicit and auditable; user-customized profiles should not be silently rewritten by preset updates.

Write-capable profiles require an explicit workspace/path binding before they are executable. Capabilities such as `workspace.write_scoped`, `docs.write_scoped`, `source.write_scoped`, `repo.write_scoped`, `tests.write_scoped`, or any future mutating capability may appear in a preset template, but enabling that duplicated profile for execution requires one of:

- a concrete `path_scopes` entry bound to a canonical workspace/root
- a host-provided workspace binding in the effective policy provenance
- a gateway-local assignment that supplies the workspace/root scope

If a write-capable profile has no effective workspace/path binding, discovery may show the profile as incomplete, but execution must deny with a machine-readable `workspace_scope_required` reason. Preset safety tests should check that write-capable bundled presets either include this requirement in `resource_constraints` or are explicitly marked as templates requiring assignment-time binding.

## Preset Safety Baseline

Built-in presets are templates. Users duplicate or override them; the built-in definitions are not silently mutable through normal profile editing.

Every bundled preset should start from this safety baseline:

- no broad process execution by default
- no destructive filesystem action by default
- no credential use unless explicitly granted
- no external network capability unless explicitly granted
- write actions are scoped and usually approval-gated
- dangerous git or shell actions require approval or are omitted
- path scope is constrained to an explicit workspace/root when writes are allowed
- high-risk capability grants must include provenance explaining why the preset includes them

Candidate bundled presets:

- `orchestrator`: workflow/task tools, broad read, constrained writes via approval
- `product-owner`: issue, story, planning, and documentation tools; no process execution
- `architect`: code search, docs, diagram generation, read-only filesystem
- `merge-conflict-resolver`: git status/diff/conflict tooling, repo-scoped write, approval for destructive git operations
- `documentation-writer`: docs read/write within workspace, no shell by default
- `project-researcher`: codebase search and read-only filesystem
- `deep-researcher`: web/research tools and citations, external network explicitly granted
- `code-reviewer`: read-only code search, diffs, test result inspection
- `devops-engineer`: deployment/log/infra tools with approval gates for mutating actions
- `backend-engineer`: scoped source writes, tests, package commands behind approval
- `frontend-engineer`: scoped frontend writes, browser/debug tools, build commands behind approval
- `qa-engineer`: browser/app-debug/log/screenshot tools, no broad write
- `sdet`: test authoring and test runner tools, scoped write
- `memory-keeper`: graph/memory tools such as Graphiti, no shell/process by default

Future full role metadata can be stored under extension metadata:

```json
{
  "agent_metadata": {
    "system_prompt": "...",
    "model_defaults": {},
    "memory_defaults": {},
    "ui_label": "Architect"
  }
}
```

Phase A enforcement ignores this metadata except for display and audit context.

## Policy Resolution And Failure Semantics

The runtime should fail closed for execution. Exact semantics:

- unresolved effective policy: deny tool execution
- `ProfileStore` unavailable: deny execution unless the host explicitly enables discovery-only degraded mode
- `ExternalRegistryStore` unavailable: deny external-server execution and mark external tools unavailable; local tools may still follow profile/RBAC policy when their required stores are available
- `AuditStore` unavailable: deny mutating, credential-bearing, external, or approval-required executions unless the host explicitly configures an emergency audit-degraded mode; read-only discovery may continue with an audit-unavailable warning
- approval service unavailable: deny any action that requires approval
- credential broker unavailable: deny secret-bearing or external-credential tool calls
- path-scope resolver unavailable: deny filesystem-affecting tool calls
- unknown high-risk metadata for a mutating tool: deny or require approval based on host policy, default deny for standalone gateway
- external server unavailable: mark that server degraded and deny execution for its tools

Discovery may degrade more softly:

- unavailable `ProfileStore` may return no executable tools plus reason metadata
- unavailable `ExternalRegistryStore` may return local tools and mark external tools unavailable
- unavailable `AuditStore` may return discovery results with audit-degraded warning metadata, but must not silently permit high-risk execution
- external discovery failure degrades only the affected server
- catalog/profile resolution failures should not grant extra execution authority

Resolution must be structured before it is wired into execution. The minimum package contract should include:

```text
ProfileResolutionResult
  status: resolved | profile_required | profile_not_found | profile_disabled | store_unavailable
  profile: MCPProfile | None
  reason_code: str
  provenance: dict
  warnings: list[dict]

EffectivePolicyResult
  status: resolved | denied | approval_required | degraded
  policy: EffectivePolicy | None
  reason_code: str
  provenance: dict
  warnings: list[dict]
```

Allowed `ProfileResolutionResult.reason_code` values should include at least `profile_resolved`, `profile_required`, `profile_not_found`, `profile_disabled`, and `profile_store_unavailable`. Execution code must branch on these status and reason fields, not on whether a profile object is `None`. Discovery may translate the same result into softer visibility metadata, but execution remains fail-closed.

Approval-required is a structured runtime outcome, not a generic permission error. It should include tool name, reason, target context, requested scope, expiry options when applicable, and safe argument summary or hash.

## Security Requirements

Profile policy is an intermediate gate, not the root of trust. It is capped by:

- authenticated identity
- host RBAC
- API key scopes
- credential availability
- path scopes
- host-level disabled modules or tools
- external server health and local transport policy

Credential grants use brokered ephemeral material. Secrets must not be stored in profile documents, logs, metrics, long-lived transport adapter state, or audit payloads.

Path scopes must be canonicalized at enforcement time, not only when saved.

Stdio external servers require:

- static operator configuration
- direct exec APIs only, with no shell
- executable allowlist
- canonical executable path validation at load and spawn time
- bounded canonical cwd
- minimal inherited environment plus explicit allowlisted keys
- resource limits where available
- audit records for spawn validation and execution outcomes

## External MCP Server Management

Phase A should manage registry plus lifecycle:

- configured external MCP servers
- stdio and WebSocket transports
- health checks
- reconnects
- discovery refresh
- per-server circuit breaker state
- namespaced virtual tools
- policy-gated execution

It should not install or update third-party server packages in Phase A.

Process-spawning upstream stdio is not the first external-server slice. Before the package starts external stdio processes, the implementation must already have registry storage, credential broker, audit sink, path/workspace policy, executable allowlist, environment allowlist, and process-policy adapters under test. Earlier external-server slices should be limited to non-spawning registry models, fake transports, health state, namespaced virtual-tool metadata, and policy-gated execution tests.

There are two distinct stdio concerns:

- upstream external-server stdio transport: Phase A, used to launch and manage configured external MCP servers behind the policy gateway
- client-facing gateway stdio transport: Phase B, used by local agent clients to launch the standalone gateway as an MCP server

HTTP/WebSocket compatibility for `tldw_server` is required before the client-facing gateway stdio entrypoint. Upstream external-server stdio can land in Phase A because existing federation already models stdio and WebSocket backends.

External server definitions have host-specific sources of truth:

- In `tldw_server`, MCP Hub remains authoritative for UI-managed external servers, credential slots, and managed secret bindings. File/YAML-backed external config may remain as a compatibility and migration source, but runtime reads should go through one adapter that merges or selects authoritative definitions according to existing MCP Hub precedence.
- In the standalone gateway, external server definitions live in the built-in SQLite store with schema versioning, timestamps, actor metadata when available, enabled/disabled state, and audit events for create/update/delete/secret-slot changes. YAML may seed or import definitions but is not the writable concurrent store.

Registry state must not bypass profile execution policy. Discovering or enabling an external server only makes its virtual tools visible as configured resources. A tool call still needs profile permission, host/gateway RBAC ceilings, external-server grant, credential grant when needed, and path/approval checks.

Tests must prove:

- external registry definitions are versioned and audited in the gateway store
- `tldw_server` still honors MCP Hub external-server precedence
- file/YAML definitions cannot override a managed MCP Hub definition without the host adapter explicitly allowing it
- discovered external tools are not executable until profile and credential grants allow them
- upstream external-server stdio lifecycle tests are separate from client-facing gateway stdio smoke tests

## Audit And Provenance

Audit should be an interface from day one. The package should emit structured events for:

- profile creation, update, duplication, migration, and disable
- effective policy resolution
- tool execution allow, deny, approval-required, approval-granted, and approval-denied
- credential grant resolution and brokered credential use, without secret values
- path-scope decisions
- external server lifecycle events
- discovery refresh outcomes

The built-in sink may be local logging and/or SQLite. Hosts can replace it.

Every deny or approval-required response should include machine-readable reason codes. Every effective policy should carry provenance that explains which profile, assignment, override, preset, or host cap produced the final allow/deny state.

## `tldw_server` Compatibility

Compatibility requirements:

- Existing `tldw_Server_API.app.core.MCP_unified` imports continue working through shims.
- Existing `/api/v1/mcp/*` route paths and payloads remain unchanged.
- Existing env vars remain accepted; new package settings may use cleaner names but must map from existing names in the host adapter.
- MCP Hub remains the authoritative store for profiles, assignments, approvals, credentials, path scopes, external server definitions, and policy resolution in `tldw_server`.
- Existing AuthNZ RBAC and API key scope ceilings remain authoritative.
- Existing tests should be moved or split only when the package boundary makes that useful; behavior coverage must not be reduced.

## Testing Strategy

### Package Tests

- Protocol request/response behavior with fake adapters
- Module registry lifecycle and dynamic discovery
- Profile schema validation and migrations
- Effective policy resolution, provenance, and failure semantics
- Preset safety baseline
- Optional SQLite `ProfileStore`, `ExternalRegistryStore`, and `AuditStore`
- External server discovery refresh, reconnect, circuit breaker, and policy-gated execution
- Stdio spawn validation with no shell and bounded cwd
- FastAPI router mounted in a minimal external app
- Stdio gateway smoke test after stdio transport lands
- Dependency-extra tests that prove core, `fastapi`, `sqlite`, `federation`, and `gateway` installs do not accidentally depend on the full `tldw_server` dependency graph

### Host Compatibility Tests

- Existing `/api/v1/mcp/*` HTTP and WebSocket behavior
- Compatibility imports from `tldw_Server_API.app.core.MCP_unified`
- MCP Hub policy resolver integration
- Approval-required structured response integration
- Credential broker integration
- Path-scope enforcement integration
- Existing chat/tool discovery behavior
- Existing env var compatibility

### Boundary Tests

- The standalone package must not import `tldw_Server_API`.
- Host adapters may import both the standalone package and `tldw_Server_API`.
- Domain modules that remain tldw-specific must not be accidentally packaged as generic runtime modules.
- Two runtime instances in one process must not share registry, profile, idempotency, session, or external-server state unless explicitly configured to share a store.

## Module Ownership Inventory

Before Stage 2 moves code into the standalone package, Stage 1 must produce a module ownership inventory.

Each current MCP module should be classified as one of:

- `runtime-neutral`: safe to move into the package with no `tldw_Server_API` dependency
- `adapter-backed`: reusable only after its tldw dependencies are replaced with explicit host adapters
- `tldw-owned`: remains in `tldw_server` and is exposed to the package runtime as a host-provided module

The inventory should include:

- current file path
- tool names exposed by the module
- dependency class
- data stores touched
- capability/risk metadata needed for profiles
- migration recommendation
- tests that protect current behavior

Stage 2 should move only runtime-neutral modules and infrastructure. Adapter-backed modules require separate design/plan slices. Tldw-owned modules should stay host-owned until there is a clear standalone use case and adapter contract.

## Migration Plan

### Stage 1: In-Place Adapter Seams

Add interfaces in the current MCP tree for auth, RBAC, rate limiting, metrics, profile resolution, approvals, credentials, path scopes, audit, and module config loading. Convert current direct calls to use default tldw adapters.

Also produce the module ownership inventory and add import-boundary scaffolding tests.

Success means existing tests keep passing and behavior is unchanged.

### Stage 2: Runtime-Neutral Package

Stage 2 is not a single move. It should be broken into independently reviewable gates:

- **Stage 2A: package boundary and neutral interfaces.** Create or maintain the `mcp_unified` package, host shim re-exports, import-boundary tests, and minimal dependency surface.
- **Stage 2B: profile schema, presets, and profile-document store primitives.** Keep this limited to data contracts, copy isolation, preset safety, and profile lookup. No execution enforcement yet.
- **Stage 2C: structured profile/effective-policy resolution.** Add result objects, reason codes, provenance, deny-over-allow precedence, workspace-binding requirements for write-capable profiles, and tests for standalone default-profile behavior versus `tldw_server` legacy no-profile behavior.
- **Stage 2D: runtime protocol and execution shell.** Move request/response models, JSON-RPC protocol dispatch, module registry contracts, idempotency helpers, and execution preparation behind fake adapters. Prove two runtime instances do not share registry, profile, idempotency, session, or external-server state.
- **Stage 2E: SQLite stores and migrations.** Add gateway-local SQLite implementations for the split stores: profiles, assignments, approval policies, credential grants, external registry, and audit. File/YAML remains seed/import/export only.
- **Stage 2F: external registry and non-spawning federation shell.** Add external-server registry models, fake transport lifecycle, namespaced virtual-tool metadata, health state, and policy-gated execution tests. Defer real upstream stdio process spawning until registry, credential, audit, path, and process-policy adapters are all present.

Success means each sub-stage has focused package tests that run without importing `tldw_Server_API`, and host compatibility tests prove existing `tldw_server` behavior remains unchanged before the next sub-stage begins.

### Stage 3: Host Adapters And Shims

Wire `tldw_server` through adapters for AuthNZ, MCP Hub policy, approval service, managed secrets, path scopes, telemetry, DB path behavior, and app lifecycle. Keep import shims and endpoint behavior intact.

Success means existing MCP tests and route compatibility tests pass.

### Stage 4: Standalone Gateway

Add package entrypoints for FastAPI HTTP/WebSocket and stdio MCP server process. Add local SQLite stores, preset loading, external MCP lifecycle management, and gateway config commands.

Success means a user can run the gateway without installing the full `tldw_server` app.

## Risks And Mitigations

- Hidden imports back into `tldw_Server_API`: enforce with boundary tests.
- Compatibility shims become permanent clutter: track deprecated shim imports and set removal criteria after host migration.
- Profile store concurrency issues: use SQLite as the supported writeable store; keep YAML/file storage read-only or import/export only.
- Presets grant too much authority: test every preset against the baseline safety rules.
- Gateway scope expands too early: keep Phase B blocked on Phase A acceptance criteria.
- External stdio transports create host risk: require direct exec, allowlists, bounded cwd, minimal env, and audit from the first stdio implementation.
- Discovery/execution confusion: keep discovery visibility separate from execution authority in tests and docs.

## Open Questions For Implementation Planning

- Should the library package live inside this repo initially or as a sibling package/worktree from the start?
- What preset set should ship in the first gateway release versus remain examples?
- Should package settings use Pydantic Settings directly or a smaller dataclass plus host-provided loaders?
- What license will the standalone package use if published outside this repo?

## Recommended First Slice

The first implementation slice should be Stage 1 only:

1. Introduce adapter protocols in the current MCP tree.
2. Convert the highest-risk direct dependencies in `protocol.py`, `server.py`, and `modules/base.py` to use adapters.
3. Keep behavior and imports compatible.
4. Add the module ownership inventory.
5. Add boundary-oriented tests that prepare for extraction, including no `tldw_Server_API` imports from the future package boundary and no shared state between runtime instances.

This creates the seam needed for the package without mixing it with gateway product work.
