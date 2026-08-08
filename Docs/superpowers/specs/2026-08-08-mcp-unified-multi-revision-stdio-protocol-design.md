# MCP Unified Multi-Revision Stdio Protocol Design

Date: 2026-08-08
Status: Approved design; implementation planning intentionally deferred
Backlog: TASK-13008
ADR: `backlog/decisions/001-mcp-unified-multi-revision-stdio-protocol.md`
Downstream consumer: `rmusser01/tldw_chatbook` TASK-2512

## 1. Summary

Extend the standalone `mcp-unified` package with a reusable, protocol-correct
stdio server surface that supports the current stateless MCP revision and the
approved legacy compatibility chain:

- `2026-07-28`
- `2025-11-25`
- `2025-06-18`
- `2025-03-26`
- `2024-11-05`

The package owns wire parsing, version profiles, lifecycle, schema validation,
pagination, cancellation, limits, safe error projection, and stdio process
behavior. Host applications continue to own tools, resources, prompts, data,
permissions, and privacy policy.

This is an additive package design. Existing `tldw_server` FastAPI and
WebSocket behavior stays pinned to its existing legacy contract. Modern
`2026-07-28` support is stdio-only in this effort. After the package is
implemented, verified, merged, and published, `tldw_chatbook` can replace its
legacy FastMCP-based standalone server without copying protocol machinery.

## 2. Goals

- Provide a public `serve_stdio(runtime, ...)` entrypoint suitable for an
  embedded consumer.
- Support all five approved MCP revisions through one explicit profile table.
- Keep `2026-07-28` requests genuinely stateless and self-describing.
- Preserve initialization-based legacy sessions and restrict batching to
  `2025-03-26` after standalone initialization.
- Support tools, resources, resource templates, and prompts.
- Keep runtime method signatures and dictionary return values compatible where
  possible.
- Validate protocol envelopes, JSON Schemas, tool inputs, and structured tool
  outputs at the gateway boundary.
- Bound input, output, concurrency, schema complexity, request rate, stderr
  capture, and shutdown work without silently truncating protocol results.
- Give downstream clients deterministic version negotiation and fallback
  behavior.
- Release and verify the package before any downstream dependency migration.

## 3. Non-Goals

- No modern Streamable HTTP, HTTP authorization, header routing, or FastAPI
  protocol upgrade.
- No Tasks extension.
- No MCP Apps or other extensions.
- No server-generated multi-round-trip `input_required` results.
- No subscriptions, list-change notifications, Roots, Sampling, Elicitation,
  or Logging expansion.
- No wrapping the official MCP SDK and no extraction of
  `tldw_Server_API.app.core.MCP_unified` host modules into downstream apps.
- No change to `tldw_server` domain tools, MCP Hub governance, external-server
  lifecycle, or stored data.
- No semantic search or downstream Library implementation in this package.
- No claim of modern HTTP conformance.

## 4. Normative Sources and Snapshot Policy

Implementation is pinned to the dated final specifications and their generated
schemas, not to an undated documentation landing page:

- `2026-07-28` base, versioning, stdio, discovery, tools, resources, and prompts
- `2025-11-25` base and schema
- `2025-06-18` base and schema
- `2025-03-26` base, lifecycle, and schema
- `2024-11-05` base, lifecycle, and schema

The implementation task must record the exact official schema or conformance
repository commit used by fixtures. Network access is not required during the
normal test suite. Any vendored normative fixture must retain its source URL,
commit, checksum, and license notice.

## 5. Protocol Profiles

One immutable `GatewayProtocolProfile` table is the sole source of
version-specific behavior. Dispatch code must not scatter date comparisons.

| Revision | Era | Lifecycle | Batches | Projection highlights |
| --- | --- | --- | --- | --- |
| `2026-07-28` | modern | Per-request `_meta`; no initialize session | rejected | `resultType`; server identity metadata; cache hints; full JSON Schema 2020-12; arbitrary JSON `structuredContent`; missing resource is `-32602` |
| `2025-11-25` | legacy | `initialize` then operation | rejected | icons/titles; JSON Schema 2020-12 default; structured tool output object; optional Tasks not advertised |
| `2025-06-18` | legacy | `initialize` then operation | rejected | titles, resource links, and structured tool output object |
| `2025-03-26` | legacy | standalone `initialize` then operation | receive required after initialization | legacy schemas and batch response rules |
| `2024-11-05` | legacy | `initialize` then operation | rejected by this server | basic tools/resources/prompts projection; no newer descriptor/result fields |

Constants exposed by the package:

- `CURRENT_PROTOCOL_VERSION = "2026-07-28"`
- `PREFERRED_LEGACY_PROTOCOL_VERSION = "2025-11-25"`
- `SUPPORTED_PROTOCOL_VERSIONS` in newest-first order
- `SUPPORTED_MODERN_PROTOCOL_VERSIONS = ("2026-07-28",)`
- `SUPPORTED_LEGACY_PROTOCOL_VERSIONS` in newest-first order

Historical support is a documented compatibility commitment, not a promise to
support every MCP revision forever. Removing a listed revision requires a
separate breaking-change decision and release note.

## 6. Architecture and Ownership

### 6.1 Package-owned protocol layer

`mcp_unified.gateway` owns:

- JSON-RPC envelope parsing and serialization
- MCP request and result validation
- protocol profile selection and result projection
- modern discovery and per-request metadata
- legacy initialization state
- list pagination and cache hints
- request task tracking and cancellation
- stdio framing and process lifecycle
- safe protocol and execution error mapping
- configurable resource limits

### 6.2 Runtime-owned application layer

An injected `GatewayRuntime` owns:

- visible tool/resource/template/prompt catalogs
- tool execution and resource/prompt reads
- authorization, policy, and audit decisions
- application-specific content and errors

The protocol layer never infers policy from self-reported `clientInfo` and does
not interpret application content as instructions.

### 6.3 Existing HTTP/WebSocket isolation

Existing `create_gateway_app`, `create_gateway_router`, and shared FastAPI
dispatch paths stay legacy-compatible. They do not begin accepting modern
per-request requests in this effort. The existing one-message helper behavior
remains available through compatibility wrappers, pinned to the current
legacy profile unless a caller explicitly opts into the new connection API.

`modules/list` and `modules/health` remain package-specific legacy aliases.
They are not advertised or served as modern MCP core methods.

## 7. Public Package API

The additive public surface is:

```python
from mcp_unified.gateway import (
    GatewayCancellationToken,
    GatewayLimits,
    GatewayProtocolConnection,
    GatewayProtocolProfile,
    GatewayRequestContext,
    GatewayRuntime,
    GatewayStdioServer,
    GatewayToolExecutionError,
    serve_stdio,
)
```

### 7.1 `GatewayProtocolConnection`

This object owns transport-scoped mechanics:

- legacy negotiated revision and initialization state
- in-flight request IDs and cancellation state
- serialized writer access
- concurrency/rate accounting
- configured limits

It must not turn modern MCP into a session. A modern request is validated and
dispatched from its own metadata even when another modern request previously
used the same stream. Any valid modern request may be the first request;
`server/discover` is not a prerequisite.

Era mixing by one stdio client is rejected after a valid legacy initialize or
after modern traffic establishes unambiguous use of the stream. That guard
prevents accidental mixed semantics; it is not used to supply missing modern
request metadata.

### 7.2 `GatewayRequestContext`

Keep existing fields and add optional fields with compatibility defaults:

```python
protocol_version: str | None = None
protocol_era: Literal["modern", "legacy"] | None = None
client_info: dict[str, Any] | None = None
client_capabilities: dict[str, Any] = field(default_factory=dict)
cancellation: GatewayCancellationToken | None = None
```

Reserved protocol metadata is constructed authoritatively by the gateway.
Caller/runtime metadata cannot replace the negotiated version, era, request
identity, client capabilities, or transport identity.

### 7.3 `GatewayRuntime`

Existing async signatures and dictionary results remain valid. Add only the
optional method:

```python
async def list_resource_templates(
    self,
    context: GatewayRequestContext,
) -> list[dict[str, Any]]: ...
```

Capability advertisement derives from callable runtime methods. A runtime may
return an empty catalog; emptiness is not an unavailable server.

### 7.4 `serve_stdio`

`serve_stdio(runtime, *, input_stream=None, output_stream=None, limits=...,
metadata=None)` owns the long-lived reader, in-flight tasks, serialized writer,
cancellation, EOF handling, and exit status. Omitted streams use the process's
binary stdin/stdout; injectable streams make the API embeddable and testable
without replacing globals. `GatewayStdioServer` delegates to the same
connection engine. `handle_stdio_line(...)` remains for compatibility and
deterministic unit tests but is not the recommended long-running server API.

## 8. Lifecycle and Version Negotiation

### 8.1 Modern `2026-07-28`

Every request must include:

- `_meta["io.modelcontextprotocol/protocolVersion"]`
- `_meta["io.modelcontextprotocol/clientCapabilities"]`

`clientInfo` is accepted but optional. Required metadata missing or malformed
returns `-32602`. Unsupported modern versions return `-32022` with only safe
`requested` and `supported` fields. The `supported` array names modern
per-request revisions only, preventing a client from retrying a legacy revision
with modern metadata instead of initializing.

The server implements `server/discover`. Discovery advertises only versions
that use the modern per-request wire contract, currently `2026-07-28`. It
returns deterministic capabilities, `resultType: "complete"`, conservative
cache hints, and server identity in reserved result metadata.

Every successful modern result includes `resultType`. Complete application
results use `"complete"`. The runtime does not generate `"input_required"` in
this scope. Server identity is injected into result `_meta` after runtime
execution so application data cannot forge it.

### 8.2 Legacy initialization

`initialize` must be a standalone request and the first lifecycle operation.
If the requested revision is supported, the server echoes it. Otherwise it
offers `2025-11-25`. A client that does not support the offered revision must
disconnect.

`notifications/initialized` is recognized. The server remains tolerant of
legacy clients that issue ordinary client-to-server requests immediately after
the initialize response; it does not incorrectly gate all work on observing
the notification. The notification still controls whether any optional
server-to-client behavior could begin, though this server does not initiate
requests.

Before initialization, only lifecycle-safe operations such as initialize and
ping are accepted. A second initialize or era switch is rejected.

### 8.3 Dual-era client fallback contract

Downstream clients probing stdio classify responses as follows:

1. `DiscoverResult`: modern; choose a mutually supported modern revision.
2. Recognized modern error such as `-32022`: modern; retry a mutually supported
   modern revision or fail. Do not initialize as legacy.
3. Any other error or a bounded timeout: legacy; close the disposable probe
   process, start a fresh process, and initialize.

The fresh process prevents an unknown legacy implementation from retaining
ambiguous pre-initialize state. Probe timeout and shutdown escalation are
configurable.

## 9. JSON-RPC and Batching

- MCP request IDs are strings or integers. `null` and booleans are rejected.
- Notifications omit `id` and never receive a response.
- Empty or whitespace-only stdio lines are ignored.
- Each non-empty line contains exactly one JSON value and no embedded newline.
- A modern or non-`2025-03-26` array request is rejected as invalid request.
- A `2025-03-26` initialize request must not be batched.
- After standalone `2025-03-26` initialization, the server receives batches,
  omits notification responses, emits no line for notification-only batches,
  and returns one response array for mixed/request batches.
- An empty batch is invalid.
- Batch elements use the same validation, policy, limits, and cancellation
  behavior as individual requests.

## 10. Methods, Capabilities, and Pagination

Core method support:

- `ping`
- `tools/list`, `tools/call`
- `resources/list`, `resources/templates/list`, `resources/read`
- `prompts/list`, `prompts/get`

Capabilities are standard MCP objects such as `{"tools": {}}`; the existing
non-standard `{"available": true}` shape is not emitted by the strict stdio
connection. Existing HTTP compatibility wrappers remain unchanged.

List methods are deterministically ordered and paginated. The gateway accepts
an opaque cursor bound to the method, protocol profile, and next offset. A
cursor is size-bounded and cannot be used across methods. The default and
maximum page sizes are explicit `GatewayLimits` values. MCP catalog pages do
not invent a total-count field; exact totals are an application tool contract,
not an MCP list-protocol requirement.

For `2026-07-28`, the gateway injects these conservative hints into
`server/discover`, list results, and resource reads:

```json
{"ttlMs": 0, "cacheScope": "private"}
```

Legacy projections omit modern cache fields. The gateway never claims
`listChanged` or subscriptions unless it implements them.

## 11. Descriptor and Result Projection

Runtime descriptors are normalized once, validated, then projected through
the selected profile.

- Unsupported newer descriptor fields are removed from older revisions.
- Tool and prompt names are validated and bounded before they can enter errors
  or logs.
- Resource URIs are validated and bounded; raw URI values are not logged.
- `2025-06-18` and `2025-11-25` structured tool content remains an object.
- `2026-07-28` structured content may be any JSON value.
- When structured content is returned, a backward-compatible text content
  representation is retained where the profile recommends it.
- If an output schema exists, structured output must conform or the gateway
  returns a safe execution failure; invalid output is never advertised as
  successful.
- Runtime `_meta` may retain validated vendor keys, but reserved MCP keys are
  injected or overwritten by the gateway.
- Legacy result projections omit modern `resultType`, cache fields, and
  per-response server identity metadata.

The server does not fetch tool/resource/prompt icons. It validates metadata
shape and preserves supported safe URI metadata only. Consumers remain
responsible for any rendering policy.

## 12. JSON Schema Validation

The package adds a direct dependency on a maintained validator supporting JSON
Schema 2020-12. Protocol validation cannot depend on an undeclared transitive
dependency.

Validation occurs at these boundaries:

- static descriptor construction/registration helpers
- dynamic descriptor list responses before publication
- `tools/call` arguments
- `structuredContent` when a tool declares `outputSchema`

The validator:

- defaults to the dialect required by the selected profile
- supports JSON Schema 2020-12
- rejects unsupported declared dialects with a safe error
- never automatically dereferences network `$ref` values
- rejects unresolved external references
- bounds schema bytes, depth, total subschemas, reference count, and pattern
  length before validation
- validates within the request input/output byte ceilings

Schema-validation errors return bounded public messages without echoing the
arguments, schema, content, pattern, or exception representation.

## 13. Stdio Execution, Cancellation, and Limits

The long-lived server uses one reader loop, one serialized writer, and a
bounded map of in-flight request tasks. Reading continues while async work is
running so cancellation can be observed.

Default package limits are documented and configurable. The implementation
plan may lower them after characterization but may not remove a category:

- maximum input line: 1 MiB
- maximum serialized output line: 1 MiB
- maximum in-flight requests: 16
- maximum catalog page size: 100
- maximum requests per minute: 600 with bounded burst
- bounded schema complexity and stderr/log message size
- bounded graceful shutdown and client probe timeouts

Crossing a limit produces a safe explicit error when a response is possible.
The gateway never silently truncates a tool result, resource, prompt, or JSON
message into an invalid or misleading success.

`notifications/cancelled` marks the referenced request, cancels cooperative
async work, and suppresses all later response/progress output for that request.
The writer rechecks cancellation immediately before writing to close the race
between result completion and cancellation arrival. Already-running sync work
may finish in its executor, but its output remains suppressed.

EOF closes input, cancels queued/cooperative work, shuts down the writer, and
exits promptly when possible. Python threads cannot be forcibly killed; hosts
must keep sync work bounded, and clients retain close/terminate/kill escalation
for a process that does not exit within its grace period.

## 14. Errors, Logging, and Privacy

The protocol layer distinguishes:

- malformed/protocol requests: JSON-RPC errors
- unknown method/tool/resource: version-correct JSON-RPC errors
- actionable tool execution/business failures: typed tool results with
  `isError: true`
- unexpected internal failures: safe generic internal errors

`GatewayToolExecutionError` carries a safe public message and optional stable
reason code. It does not expose the original exception. Application handlers
must stop converting raw `str(exc)` values into successful `{error: ...}`
payloads.

Version-specific error projection includes:

- modern missing metadata: `-32602`
- modern unsupported version: `-32022` with `requested` and `supported`
- modern missing resource: `-32602`
- legacy missing resource: `-32002` where required for compatibility
- no undefined modern use of reserved `-32020..-32099` codes

Existing legacy `GatewayPolicyDenied -> -32001` behavior may remain in the
legacy FastAPI compatibility path. The strict modern path uses standard errors
where possible; any retained application-defined protocol code must be outside
the JSON-RPC/MCP reserved range and documented. Policy error data is projected
through an allowlist of stable status/reason fields. Provenance and warnings
must be recursively bounded and stripped of payloads, paths, credentials, and
arbitrary exception strings.

Logs contain method category, validated identifier, protocol revision, status,
duration, and byte counts. They never contain raw JSON-RPC payloads, tool
arguments/results, resource URIs/content, prompt content, client-supplied
identity text, credentials, or stderr. Raw external-server stderr exposure is
opt-in and outside this stdio server scope.

Self-reported `clientInfo` is display/debug metadata only. It never keys
authorization or rate limiting. Stdio rate limiting is per process.

## 15. Downstream Consumer Contract

Before publication, upstream tests include a synthetic consumer shaped like
`tldw_chatbook` without importing that repository. It must prove:

- explicit descriptor registries can supply tools/resources/templates/prompts
- a runtime with empty capabilities is valid
- `serve_stdio` can run from an embedded module entrypoint
- `GatewayRequestContext` additions are optional for existing runtimes
- an application can choose serialized tool execution (`max_in_flight=1`)
- typed execution errors, pagination, cancellation, and limits cross the public
  API without private imports

This contract does not expose or duplicate the downstream application's local
Library tools. Application data and privacy boundaries remain downstream-owned.
The corresponding downstream boundary is specified in the
[Chatbook MCP Unified migration design](https://github.com/rmusser01/tldw_chatbook/blob/main/Docs/superpowers/specs/2026-08-08-mcp-unified-chatbook-migration-design.md).

## 16. Verification Matrix

### 16.1 Protocol vectors

Tests cover every revision for:

- initialization/discovery and version mismatch
- capability and descriptor projection
- result projection and server metadata
- missing/invalid/null request IDs
- notifications and notification suppression
- tools/resources/templates/prompts happy paths and errors
- empty catalogs
- pagination and invalid/repeated/cross-method cursors
- structured output and schema validation
- resource-not-found error changes
- modern cache hints and legacy field absence
- reserved metadata authority

### 16.2 Batching

- initialize-in-batch rejection for every revision
- receive batches only after `2025-03-26` initialization
- mixed, request-only, notification-only, empty, malformed-element, and
  oversized batch behavior
- explicit rejection in the other four revisions

### 16.3 Stdio subprocess behavior

- one valid message per stdout line and no non-protocol stdout
- concurrent request reading and serialized output
- cancellation before start, during async work, and at the writer race
- no output after cancellation
- EOF and graceful/forced shutdown behavior
- input/output/in-flight/rate/schema limits
- payload-free diagnostics

### 16.4 Interoperability

- validate emitted/accepted messages against pinned official wire schemas
- run applicable official conformance scenarios that support this stdio scope
- run at least one current Tier 1 official SDK client against the stdio server
- preserve raw deterministic vectors as the five-revision authority

The official conformance server harness is currently URL-oriented. This effort
does not add modern HTTP merely to obtain a broader badge. Documentation states
the exact stdio scenarios and versions verified and does not claim full
transport conformance.

### 16.5 Compatibility and packaging

- existing FastAPI/WebSocket gateway regression suite
- old `handle_stdio_line` compatibility tests
- package import-boundary tests
- Python 3.10 through the supported upper CI version
- wheel and sdist build, metadata, contents, and fresh-environment installs
- base and documented extra installation smokes
- API/signature tests against the installed artifact, not the source tree
- no sensitive fixture data

## 17. Release Sequence

The checked-in `origin/dev` package metadata currently says `0.2.0`; project
history records `0.1.1` as the first published version. This document does not
assume `0.2.0` remains available.

Release order:

1. Implement and verify on a clean branch from current `origin/dev`.
2. Run the package RC and publish dry-run gates.
3. Run the synthetic downstream consumer contract against built wheel and
   sdist artifacts.
4. Recheck PyPI immediately before release. Use `0.2.0` only if it is still the
   next available intended release; otherwise select the next available minor.
5. Merge through the repository's controlled `dev` to `main` release path.
   The existing workflow publishes a package-version change on `main`; no
   artifact is uploaded from a developer's dirty/local checkout.
6. Verify the PyPI artifact metadata, hash, import surface, and stdio smoke in a
   fresh environment.
7. Only then allow `tldw_chatbook` to pin the released compatible minor series
   and begin its migration.

The downstream compatible pin is `~=X.Y.0`, equivalent to `>=X.Y.0,<X.(Y+1).0`
for this pre-1.0 package. This prevents an unreviewed later minor from silently
changing protocol behavior.

If a release defect is found, publish a corrective version under the existing
release policy; never overwrite an artifact. Downstream remains on FastMCP
until the published package passes verification, so no cross-repository
rollback or data migration is required.

## 18. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Modern connection object accidentally becomes a session | Per-request modern context tests; connection retains transport state only |
| Legacy changes leak into existing HTTP routes | Explicit legacy compatibility wrapper and FastAPI/WebSocket regression suite |
| Version logic drifts across methods | One immutable protocol profile table and complete profile-vector tests |
| PyPI release exposes a missing consumer API | Installed-artifact synthetic consumer contract before publication |
| Schema validation enables SSRF or CPU exhaustion | No external `$ref`; bounded schema size/depth/subschemas/patterns |
| Cancellation races emit a late result | Cancellation state rechecked under serialized writer immediately before output |
| Large content deadlocks or corrupts stdout | Bounded line/output, explicit error, continuous stderr draining, no truncation |
| Cache metadata leaks user-specific catalogs | `ttlMs: 0`, `cacheScope: private`, authorization remains independent |
| Old and modern fields are mixed | Projection tests assert required presence and forbidden absence per revision |
| Release automation publishes the wrong artifact | Build from merged commit, immutable artifact handoff, PyPI availability/hash checks |

## 19. Alternatives Considered

### Replace the package with the official SDK

Rejected. It would discard the package's existing runtime, profiles, policy,
audit, lifecycle, and host compatibility work. The official SDK remains an
interoperability oracle, not the application runtime.

### Wrap FastMCP inside `mcp-unified`

Rejected. It would preserve the dependency and decorator/runtime coupling the
downstream migration is intended to remove.

### Support only `2026-07-28`

Rejected. Existing clients and downstream fallback requirements need the
approved legacy chain, and the package already exposes a legacy surface.

### Upgrade the shared HTTP dispatcher to modern behavior

Rejected for this effort. Modern HTTP adds headers, authorization, and
stateless routing concerns beyond the requested stdio prerequisite and risks
breaking existing `tldw_server` routes.

### Keep permissive dictionary pass-through without schema validation

Rejected. It would advertise invalid tools/results as successful, violate the
current schema contract, and leave an avoidable denial-of-service/security
boundary open.

## 20. ADR Check

ADR required: **yes**

ADR path:
`backlog/decisions/001-mcp-unified-multi-revision-stdio-protocol.md`

Reason: the change establishes a public protocol/runtime API, dependency and
release boundary, cross-version service contract, and security/privacy policy.
