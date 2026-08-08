# ADR-001: MCP Unified Multi-Revision Stdio Protocol Boundary

Status: Accepted
Date: 2026-08-08
Related Task: [TASK-13008](../tasks/task-13008%20-%20Design-MCP-Unified-multi-revision-stdio-protocol-compatibility.md)
Supersedes: N/A

## Decision

`mcp-unified` will own a reusable, package-isolated stdio protocol engine for
MCP revisions `2026-07-28`, `2025-11-25`, `2025-06-18`, `2025-03-26`, and
`2024-11-05`.

The engine uses one immutable protocol-profile table. `2026-07-28` requests
are stateless and validated from required per-request metadata; any valid
modern request may be first, and `server/discover` is implemented but not
required as a precondition. Legacy revisions negotiate through standalone
`initialize`. JSON-RPC batching is received only after a standalone
`2025-03-26` initialization and rejected for every other supported revision.

The public package API adds `GatewayCancellationToken`,
`GatewayProtocolConnection`, `GatewayLimits`, `GatewayProtocolProfile`,
`GatewayToolExecutionError`, and `serve_stdio(runtime, ...)`. Existing
`GatewayRuntime` async signatures and dictionary results remain compatible;
resource-template listing is the only optional runtime-method addition.
`GatewayRequestContext` receives optional protocol/client/cancellation fields
with compatibility defaults.

The protocol layer owns envelopes, lifecycle, projection, pagination,
cancellation, limits, JSON Schema validation, cache hints, and safe errors.
Injected runtimes own application tools/content and authorization. Reserved MCP
metadata is authoritative at the gateway boundary and cannot be forged by
runtime results or caller metadata.

Modern compliance is stdio-only in this effort. Existing `tldw_server`
FastAPI/WebSocket behavior stays on its legacy compatibility path. Package
extensions `modules/list` and `modules/health` remain legacy-only aliases and
are not presented as modern MCP core.

The package will directly depend on a JSON Schema 2020-12 validator, prohibit
automatic external `$ref` dereferencing, and bound schema/input/output
complexity. Modern list/read results default to `ttlMs: 0` and
`cacheScope: private`. Tool execution failures use typed `isError: true`
results; raw exceptions and protocol payloads are never returned or logged.

The package must be built, tested as an installed artifact, checked against a
synthetic downstream consumer and an official current stdio client, published,
and verified before `tldw_chatbook` migrates. The downstream application pins
the released pre-1.0 minor series.

## Context

The standalone gateway currently hardcodes `2024-11-05`, advertises
non-standard capability members, allows nullable request IDs, and applies
batching independently of negotiated revision. Its line helper constructs a
new server for each message and is not a long-lived reader/writer/cancellation
runtime. The package nevertheless already has the reusable `GatewayRuntime`,
resource/prompt dispatch, governance, packaging, and release boundaries needed
by downstream applications.

`tldw_chatbook` needs to retire its optional FastMCP standalone server while
preserving its module launch command, application-owned tools/resources/
prompts, local permission behavior, and private Library boundary. Copying a
second protocol implementation into Chatbook would create drift, while
extracting `tldw_server` host modules would import unrelated server ownership
and dependencies.

The current MCP revision is a breaking transition from initialization sessions
to per-request stateless metadata. It also requires result discriminators,
server discovery, deterministic catalogs, private/public cache hints, and
updated schema/error behavior. Supporting it by adding conditionals to the
existing dispatcher without a profile model would make every subsequent
method a compatibility hazard.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Use the official MCP SDK as the runtime | Loses the package's governance/runtime investment and makes downstream policy integration depend on another runtime abstraction. The SDK remains an interoperability oracle. |
| Wrap FastMCP inside `mcp-unified` | Retains the decorator and dependency boundary being removed and does not solve version-specific behavior explicitly. |
| Implement the protocol only in Chatbook | Duplicates versioning, stdio, cancellation, limits, and security machinery and prevents reuse by other package consumers. |
| Support only `2026-07-28` | Breaks the approved legacy client compatibility chain and existing package behavior. |
| Upgrade HTTP/WebSocket simultaneously | Expands scope into headers/auth/routing and creates unnecessary regression risk for current server routes. |
| Permit batches for all revisions because JSON-RPC supports them | Violates revision-specific MCP behavior; only `2025-03-26` in the approved chain requires receiving them. |
| Cache modern client metadata on the stdio connection | Contradicts the stateless current protocol and makes interleaved requests depend on prior traffic. |
| Pass schemas/results through without validation | Can advertise invalid contracts, leak unsafe errors, and permit avoidable resource-exhaustion behavior. |

## Consequences

### Benefits

- Downstream applications get one reusable and tested stdio protocol engine.
- Modern and legacy behavior is reviewable in one explicit compatibility
  table.
- Existing host HTTP behavior is insulated from the modern upgrade.
- Safe defaults bound local subprocess resource use and keep private catalogs
  out of shared caches.
- Installed-artifact consumer tests reduce the risk of an unusable immutable
  PyPI release.

### Accepted trade-offs

- Five protocol revisions materially increase fixture and maintenance cost.
- The strict stdio surface intentionally corrects some permissive legacy
  behavior, including nullable IDs, non-standard capabilities, unsafe raw
  errors, and globally enabled batching.
- Existing FastAPI compatibility behavior and strict stdio behavior are not
  identical until a separately approved HTTP migration occurs.
- Running synchronous application work cannot be forcibly cancelled inside a
  Python thread; cancellation suppresses output and clients retain process
  termination escalation.
- `ttlMs: 0` sacrifices catalog caching in favor of a safe default for
  configuration- and permission-dependent local servers.
- Tasks, MRTR generation, subscriptions, modern HTTP, and remote icon fetching
  remain outside this decision.

## Compliance and Release Boundary

The implementation records the exact dated official schemas/conformance commit
used by tests. Documentation claims only the protocols, methods, and stdio
scenarios actually verified.

The checked-in development metadata currently names package version `0.2.0`,
but release automation must recheck PyPI and select the next available intended
minor. Production artifacts come from the controlled merged-commit workflow,
not a local checkout. Published versions are immutable; defects receive a new
corrective release.

## Links

- [Design specification](../../Docs/superpowers/specs/2026-08-08-mcp-unified-multi-revision-stdio-protocol-design.md)
- [MCP 2026-07-28 versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP 2026-07-28 stdio](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP 2026-07-28 discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP conformance project](https://github.com/modelcontextprotocol/conformance)
- [Standalone library and gateway design](../../Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md)
- [Downstream Chatbook migration design](https://github.com/rmusser01/tldw_chatbook/blob/main/Docs/superpowers/specs/2026-08-08-mcp-unified-chatbook-migration-design.md)
