# MCP Unified Residual UX Hardening Design

Date: 2026-06-28
Status: Draft for spec review
Backlog: TASK-2372
Builds on: TASK-2393

## Summary

This design closes the remaining Unified MCP UX and trust gaps found after the
completed TASK-2393 remediation. It does not reopen the completed remediation
slice. Instead, it adds a narrower residual hardening pass focused on product
truth, safer first-run defaults, error recovery, status/readiness clarity, and
documentation contracts.

The supported user-facing product remains embedded TLDW MCP at
`/api/v1/mcp`. The `apps/mcp-unified` package is an internal and experimental
package boundary for future standalone gateway work. A runnable standalone
gateway server command remains out of scope for this pass and belongs to the
future Phase B standalone gateway work.

## Goals

- Make the current embedded vs package-local vs future-standalone model
  impossible to misunderstand in docs, client examples, smoke guides, admin
  guides, and package metadata.
- Require explicit opt-in before enabling local filesystem or local process
  MCP capabilities by default.
- Preserve compatibility except for that intentional safer-default change.
- Improve diagnostics for known failure modes without redesigning the MCP
  protocol.
- Make status/readiness useful for both first-time users and returning power
  users.
- Add contract tests so the same UX regressions do not recur.

## Non-Goals

- No `mcp-unified-gateway serve` command.
- No standalone gateway product completion.
- No PyPI/TestPyPI publishing changes.
- No package marketplace, server installer, or third-party registry work.
- No broad MCP Hub UI redesign.
- No broad JSON-RPC protocol redesign.
- No removal of legacy query-auth support in this pass.

## Product Positioning

The documentation and package status should use one consistent model:

- Embedded TLDW MCP is supported today and uses the TLDW Server launch path.
- `apps/mcp-unified` is an internal, experimental package boundary with CLI
  utilities, package-local smoke tests, profile/policy primitives, and gateway
  building blocks.
- The package-local FastAPI gateway app is an embeddable factory, not a
  supported end-user standalone server unless a later Phase B adds and verifies
  a serve command.
- MCP-specific Docker assets are experimental unless a tested supported image
  is explicitly added later.

This wording should appear in the primary hosted docs, client snippets,
package docs, smoke client docs, and admin docs. Docs should not use
"standalone gateway" as a promise unless the same page also states the current
unsupported/experimental state.

## Findings Addressed

This residual pass addresses these latest-review findings:

- Package-local docs still read like a runnable standalone gateway product.
- The gateway CLI can validate/configure/administer, but not launch an
  end-user gateway server.
- Admin configuration docs are not tightly checked against actual `MCPConfig`
  names and host-level AuthNZ boundaries.
- MCP-specific Docker still contains production-sounding language.
- Default module configuration exposes high-risk local file/process capability
  too early.
- WebSocket docs and generated OpenAPI copy still present query-token auth as
  a normal path.
- Known errors still collapse to generic messages in important user flows.
- Package gateway `/status` is too thin for readiness and troubleshooting.
- Smoke/client docs mix package-local and embedded base paths.

## Compatibility Guardrails

Runtime and API changes must be additive except for the deliberate safer-default
change to high-risk modules.

The safer-default behavior change must include:

- a migration note for users who depended on default filesystem or command
  tools
- an opt-in YAML example
- a guarantee that existing explicit operator configuration enabling these
  modules remains honored; only implicit defaults and generated default configs
  change
- tests that explicitly enable those modules when they need them
- status output that explains "available but disabled by default" instead of
  making tools look missing

Error handling must preserve JSON-RPC compatibility:

- JSON-RPC errors keep the current error shape.
- Known structured details may be added under `error.data`.
- HTTP helper endpoints must preserve existing status codes and public error
  body shape where clients may already depend on it. Endpoints that already
  return object-shaped `detail` may add `reason_code` and `next_action` keys.
  Endpoints that currently return string-shaped `detail` should keep that
  string stable and expose recovery metadata through additive response headers
  or another explicitly documented additive channel.
- Initial structured errors should cover only known classes: auth required,
  permission denied, invalid params, unresolved catalog, module unavailable,
  external/upstream unavailable, and unsupported/experimental launch path.

WebSocket query-auth behavior should not be removed in this pass. Docs and
OpenAPI text should mark query-token and query-api-key auth as legacy,
disabled by default, and not recommended. Normal examples should use headers
or subprotocol-based auth.

Env-var docs checks must allow explicitly documented host/AuthNZ-level
exceptions. The test should fail unknown variables only when they are neither
real `MCPConfig` aliases nor listed in a documented exception map.

## Design Areas

### 1. Product Truth And Docs Contracts

Add docs contract tests that scan the hosted MCP docs, client snippets,
package-local docs, smoke docs, admin docs, and package README for consistent
status language. The tests should assert that:

- hosted MCP docs identify `/api/v1/mcp` as the supported path
- package docs identify `apps/mcp-unified` as internal/experimental
- package docs do not claim an end-user serve path exists
- smoke and client-facing examples clearly label embedded vs package-local
  examples
- client-facing examples use embedded `/api/v1/mcp` unless they are explicitly
  labeled package-local, in-process, or host-mounted gateway examples
- docs use the correct package install path: `apps/mcp-unified[...]`
- docs link targets exist

The docs should include a short "Which path should I use?" table:

- "I want to connect MCP clients to TLDW": start TLDW Server and use
  `/api/v1/mcp`.
- "I am testing the future package boundary": install `apps/mcp-unified`.
- "I want a separate standalone gateway process": not supported yet; see the
  Phase B standalone gateway design.

### 2. Safer Default Capability Surface

Disable local filesystem and local process modules in the default module config
unless the operator explicitly opts in. The initial target modules are the
high-risk tiers already described in `module_surface.py`:

- `filesystem`
- `run_command`
- `sandbox`
- `browser_cdp`
- `git`
- other modules classified as `local_files` or `local_process`

The design should prefer configuration-level opt-in over hiding tools at a
later layer. That keeps the mental model simple: high-risk modules are not
enabled until an operator enables them.

Existing explicit operator configuration remains authoritative. If a local
deployment already has `enabled: true` for one of these modules in its module
YAML, that explicit opt-in should continue to enable the module. The behavior
change applies to implicit defaults, checked-in default configs, and generated
starter configs.

Status output should be extended beyond enabled modules. The surface summary
should include:

- `enabled_count`
- `tiers`
- `disabled_available`
- `requires_explicit_opt_in`

Each disabled high-risk module should show an id, risk tier, short
description, and next action such as "enable this module in
Config_Files/mcp_modules.yaml only if this deployment should expose local file
access."

### 3. Runtime And API Trust Improvements

Add small diagnostic improvements where users currently hit generic failures.

HTTP helper endpoints should expose structured recovery metadata for known
failures without breaking existing public error shapes.

Endpoints that already return object-shaped `detail` may extend that object:

```json
{
  "reason_code": "permission_denied",
  "message": "Permission denied for listing MCP modules.",
  "next_action": "Use a token or API key with the required MCP permission."
}
```

Endpoints that currently return string-shaped `detail` should preserve the
string and add equivalent metadata through an additive channel, for example:

```http
HTTP/1.1 500 Internal Server Error
X-MCP-Reason-Code: module_unavailable
X-MCP-Next-Action: Check /api/v1/mcp/status for problem_modules.

{"detail":"Failed to list MCP modules"}
```

JSON-RPC should keep its current error code/message behavior, but known errors
may add structured data:

```json
{
  "code": -32602,
  "message": "Invalid params",
  "data": {
    "reason_code": "invalid_params",
    "next_action": "Check the tool input schema from tools/list."
  }
}
```

Generated endpoint docs should be corrected so WebSocket query auth is not the
normal-path example. Query auth parameters can remain visible, but their
description should say they are legacy and disabled by default unless
`MCP_WS_ALLOW_QUERY_AUTH=true`.

Route-specific copy should be corrected where it references the wrong action,
such as saying "listing tools" while the user is listing modules.

### 4. Package Gateway Readiness Status

The package-local gateway `/status` response should become a best-effort
readiness report. It must not require a fully configured external runtime to
respond.

The response should include redacted fields when available:

- package status and publishing status
- version
- transport base path or mount context if known
- store kind and whether persistence is configured
- default profile id or "none"
- admin auth enabled/disabled
- external server count and unavailable count
- warnings
- next actions

If a field is unavailable because the app is embedded by another host, the
response should say `unknown` or omit the field rather than failing.

Publishing status here is a static support/readiness label from package
metadata, such as `not-published`. It must not imply PyPI integration,
release automation, marketplace readiness, or any publishing workflow change.

### 5. Admin, Docker, And Smoke Cleanup

Admin docs should be checked against actual `MCPConfig` aliases plus an
explicit allowlist for host/AuthNZ variables. Unsupported MCP-like variables
should be removed or marked as host-level examples.

Docker cleanup should quarantine rather than repair:

- remove production-ready language from the MCP-specific Dockerfile comments
  and docs
- label the directory experimental
- ensure primary docs do not present that Dockerfile as a supported launch path
- defer a tested standalone Docker image to Phase B

Smoke and client docs should separate:

- embedded TLDW HTTP/WebSocket examples at `/api/v1/mcp`
- package-local smoke/in-process examples
- remote gateway examples that require an already-running host-mounted gateway

Embedded status examples must use the exact hosted status path, such as
`/api/v1/mcp/status`. Package-local gateway examples may use `/status` only
when the example is explicitly labeled as package-local or host-mounted gateway
behavior. The smoke and client docs should not imply the package CLI starts a
server.

## Data Flow And Mental Model

First-time embedded user:

1. Reads "Which path should I use?"
2. Starts TLDW Server.
3. Uses one documented auth path.
4. Calls status and sees enabled modules plus high-risk modules disabled by
   default.
5. Lists tools and runs a safe read-only diagnostic call.
6. If they need files or commands, follows explicit opt-in docs.

Experienced operator:

1. Uses the operator cheatsheet.
2. Checks `/api/v1/mcp/status` and reads `surface` plus
   `problem_modules`.
3. Enables high-risk modules deliberately through YAML.
4. Uses structured errors and reason codes to diagnose auth, permission,
   catalog, module, or upstream failures.
5. Uses package-local docs only when working on the future standalone package
   boundary.

Package-boundary developer:

1. Installs `apps/mcp-unified[gateway,dev]`.
2. Runs package-info, preset listing, validate-config, and in-process smoke.
3. Understands that remote runtime commands require an already-running mounted
   gateway supplied by a host.
4. Uses package `/status` readiness fields to diagnose store/admin/profile
   configuration.

## Testing Strategy

Add focused tests before implementation where practical:

- docs contract tests for product status language, install paths, base paths,
  status paths, missing links, client snippet path consistency, and absence of
  normal-path query-token examples
- config/module tests proving high-risk modules are disabled by default and
  existing explicit `enabled: true` operator config remains honored
- status tests proving disabled high-risk modules are shown as available and
  requiring opt-in
- HTTP endpoint tests for known recovery metadata that preserves existing
  error body shape
- JSON-RPC protocol tests for structured `error.data` on known failures without
  changing the outer error shape
- package gateway status tests for best-effort readiness fields
- admin env-var docs tests using a real-alias set plus explicit exception map
- Docker/docs contract tests ensuring the MCP-specific Docker path is
  experimental only
- smoke docs tests for correct package path and embedded/package base paths

Verification should include:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py -v
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py -v
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py -v
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -v
python -m bandit -r tldw_Server_API/app/core/MCP_unified tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py -f json -o /tmp/bandit_mcp_residual_ux.json
```

The exact test set may be refined during implementation, but the plan should
cover docs contracts, safer defaults, status, known errors, and package status.

## Sequencing

1. Product truth and docs contract tests.
2. Safer high-risk defaults and migration docs.
3. Status/module-surface visibility for disabled available capabilities.
4. Known-error details and WebSocket auth docs cleanup.
5. Package gateway readiness status.
6. Admin env-var, Docker, and smoke docs cleanup.
7. Focused verification, Bandit touched-scope scan, and Backlog closeout.

## Risks And Mitigations

- Risk: safer defaults break local workflows.
  Mitigation: migration note, opt-in example, explicit test fixtures, and clear
  status next actions.

- Risk: structured errors accidentally change client contracts.
  Mitigation: keep JSON-RPC outer shape stable and add details only under
  `error.data`.

- Risk: docs contracts become brittle.
  Mitigation: assert stable promises and required phrases, not exact prose.

- Risk: env-var docs tests reject legitimate host-level configuration.
  Mitigation: maintain an explicit exception map with comments.

- Risk: package gateway readiness implies product readiness.
  Mitigation: include package status and publishing status in the response and
  keep docs explicit that the gateway is experimental.

## Acceptance Criteria

- Primary docs, client snippets, package docs, smoke docs, and admin docs use
  one consistent embedded/package/future-standalone model.
- No docs imply `mcp-unified-gateway` launches an end-user standalone server.
- High-risk local file/process modules are disabled by default and have an
  explicit opt-in path.
- Existing explicit operator config that enables high-risk modules remains
  honored.
- Status shows disabled high-risk capabilities as available but requiring
  explicit opt-in.
- WebSocket query auth is documented as legacy, disabled by default, and not
  the normal path.
- Known HTTP helper failures expose reason code and next action without
  breaking existing status codes or string/object error body shape.
- JSON-RPC compatibility is preserved while known errors may include
  structured `error.data`.
- Package gateway `/status` returns best-effort readiness without requiring a
  fully initialized runtime.
- Embedded client and smoke examples use `/api/v1/mcp/status`; package-local
  `/status` examples are explicitly labeled package-local or host-mounted.
- Package gateway `/status` treats publishing status as static readiness
  metadata, not as publishing workflow scope.
- Admin/env-var docs are mechanically checked against `MCPConfig` plus an
  explicit host/AuthNZ allowlist.
- MCP-specific Docker assets are clearly experimental and not a supported
  launch path.
- Focused tests and Bandit touched-scope scan pass or document only existing
  unrelated baseline findings.
