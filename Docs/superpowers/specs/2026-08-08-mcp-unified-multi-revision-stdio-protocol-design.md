# MCP Unified Multi-Revision Stdio Protocol Design

Date: 2026-08-08
Status: Approved design; implementation planning intentionally deferred
Backlog: TASK-13008
ADR: `Docs/ADR/033-mcp-unified-stdio-contract-hardening.md`
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
- Keep existing dictionary-returning runtimes structurally compatible while
  widening the strict core runtime to support every JSON value allowed by the
  current protocol.
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
| `2026-07-28` | modern | Per-request `_meta`; no initialize session | rejected | required `resultType`; server identity metadata; cache hints; JSON Schema 2020-12 dialect with object-rooted `inputSchema` and arbitrary-root `outputSchema`; arbitrary JSON `structuredContent`; missing resource is `-32602` |
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

An injected `GatewayCoreRuntime` owns:

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
and defaults remain available through `GatewayStdioServer`; it does not
silently acquire lifecycle state or strict long-lived semantics. The new
`GatewayProtocolStdioServer` and `serve_stdio` entrypoint own the strict
connection engine.

`modules/list` and `modules/health` remain package-specific legacy aliases.
They are not advertised or served as modern MCP core methods.

## 7. Public Package API

The additive public surface is:

```python
from mcp_unified.gateway import (
    GatewayApplicationError,
    GatewayAsyncByteReader,
    GatewayAsyncByteWriter,
    GatewayCancellationToken,
    GatewayCoreRuntime,
    GatewayInvalidApplicationResult,
    GatewayLimits,
    GatewayProtocolConnection,
    GatewayProtocolProfile,
    GatewayProtocolStdioServer,
    GatewayRequestContext,
    GatewayResourceNotFound,
    GatewayResourceTemplateRuntime,
    GatewayResultTooLarge,
    GatewayRuntime,
    GatewayStdioServer,
    GatewayToolExecutionError,
    GatewayJSONValue,
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

The existing `request_id` field is widened from `str` to `str | int`. This is a
source-compatible annotation widening for current callers and is required to
preserve MCP identity: integer `1` and string `"1"` must remain distinct through
runtime dispatch, cancellation, audit correlation, and response echo. Boolean
and null IDs are rejected before context construction.

Keep the other existing fields and add optional fields with compatibility
defaults:

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

### 7.3 Runtime protocols

The package exposes a recursive JSON type alias:

```python
GatewayJSONScalar: TypeAlias = Union[None, bool, int, float, str]
GatewayJSONValue: TypeAlias = Union[
    GatewayJSONScalar,
    list["GatewayJSONValue"],
    dict[str, "GatewayJSONValue"],
]
```

Finite numbers and string-keyed mappings are enforced at runtime. The new
strict server accepts a narrow `GatewayCoreRuntime` containing only the MCP core
tool/resource/prompt methods plus `name` and `version`. Its `call_tool` result is
`GatewayJSONValue`; existing implementations that return `dict[str, Any]`
remain structurally compatible because their return type is narrower.

```python
@runtime_checkable
class GatewayCoreRuntime(Protocol):
    name: str
    version: str

    async def list_tools(
        self, context: GatewayRequestContext
    ) -> list[dict[str, Any]]: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> GatewayJSONValue: ...

    async def list_resources(
        self, context: GatewayRequestContext
    ) -> list[dict[str, Any]]: ...

    async def read_resource(
        self, uri: str, context: GatewayRequestContext
    ) -> dict[str, Any]: ...

    async def list_prompts(
        self, context: GatewayRequestContext
    ) -> list[dict[str, Any]]: ...

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]: ...
```

The existing `GatewayRuntime` remains public and unchanged for legacy gateway
callers, including its package-specific `list_modules` and
`get_modules_health` methods. Strict stdio does not require downstream
consumers to implement those unrelated aliases. Resource-template listing is
additive through a separate runtime extension, not a statically mandatory core
member:

```python
@runtime_checkable
class GatewayResourceTemplateRuntime(Protocol):
    """Optional extension detected only when the method is callable."""

    async def list_resource_templates(
        self,
        context: GatewayRequestContext,
    ) -> list[dict[str, Any]]: ...
```

The gateway uses callable `getattr` detection and validates returned
descriptors. Absence means the optional template-listing capability is not
advertised. A runtime may return an empty catalog; emptiness is not an
unavailable server.

### 7.4 Strict and compatibility stdio surfaces

`GatewayStdioServer` and `handle_stdio_line(...)` retain their historical
independent-message behavior and compatibility defaults. They are not the
recommended long-running server API and do not delegate to a stateful lifecycle
engine.

`GatewayProtocolStdioServer` owns the strict long-lived connection mechanics and
backs this coroutine:

```python
async def serve_stdio(
    runtime: GatewayCoreRuntime,
    *,
    input_stream: GatewayAsyncByteReader | None = None,
    output_stream: GatewayAsyncByteWriter | None = None,
    limits: GatewayLimits = GatewayLimits(),
    metadata: Mapping[str, Any] | None = None,
) -> int: ...
```

The reader supplies `async readline() -> bytes`. The writer supplies
`write(bytes) -> None` and `async drain() -> None`; text streams are rejected at
startup. Omitted streams use process binary stdin/stdout without replacing or
closing the global objects. Injected streams are caller-owned and are flushed
but never closed by the gateway.

The coroutine returns `0` after clean EOF or protocol shutdown and `1` after a
fatal transport/internal server failure. Invalid local construction raises
before reading input. Individual protocol/application errors remain JSON-RPC
responses and do not change the process exit status. If the serving task is
cancelled, it cancels tracked work, performs the bounded graceful-shutdown
sequence, and re-raises `asyncio.CancelledError`. A module entrypoint maps the
returned integer to its process exit code.

### 7.5 Cancellation, limits, and safe application errors

One `GatewayCancellationToken` instance is created per request and passed in
that request's context. Its public thread-safe API is:

- `cancel(reason: str | None = None) -> bool`, returning whether state changed;
- read-only `cancelled: bool` and bounded `reason: str | None`;
- `is_cancelled() -> bool` and `raise_if_cancelled() -> None`; and
- `async wait() -> None`.

Cancellation reasons are diagnostic classifications, not payloads, and are
never logged raw. The same token instance is used by dispatch, runtime work,
and the writer race check.

`GatewayLimits` is a frozen dataclass with these initial public fields and
defaults:

| Field | Type | Default | Accepted range |
| --- | --- | --- | --- |
| `max_input_line_bytes` | `int` | `1_048_576` | `1..16_777_216` |
| `max_output_line_bytes` | `int` | `1_048_576` | `1..16_777_216` |
| `max_result_bytes` | `int` | `786_432` | `1..16_777_216` and no greater than `max_output_line_bytes` |
| `max_json_depth` | `int` | `64` | `1..256` |
| `max_in_flight` | `int` | `16` | `1..1_024` |
| `default_catalog_page_size` | `int` | `50` | `1..1_000` and no greater than `max_catalog_page_size` |
| `max_catalog_page_size` | `int` | `100` | `1..1_000` |
| `max_catalog_items` | `int` | `10_000` | `1..100_000` and no less than `max_catalog_page_size` |
| `max_batch_items` | `int` | `100` | `1..1_000` |
| `max_requests_per_minute` | `int` | `600` | `1..60_000` |
| `request_burst` | `int` | `32` | `1..10_000` and no greater than `max_requests_per_minute` |
| `max_schema_bytes` | `int` | `262_144` | `1..4_194_304` |
| `max_schema_depth` | `int` | `32` | `1..128` |
| `max_schema_subschemas` | `int` | `1_024` | `1..10_000` |
| `max_schema_refs` | `int` | `256` | `1..4_096` |
| `max_schema_pattern_chars` | `int` | `4_096` | `1..65_536` |
| `max_schema_validation_processes` | `int` | `4` | `1..32` |
| `schema_validation_timeout_seconds` | `float` | `10.0` | finite and `(0, 10]` |
| `graceful_shutdown_timeout_seconds` | `float` | `5.0` | finite and `(0, 60]` |

Integer fields reject booleans and values outside their accepted ranges.
The validation timeout default was raised from `1.0` to `5.0` in `0.2.1`
after the protected Windows SDK smoke demonstrated that a clean spawned worker
can need more than one second to import before validation begins; the `10.0`
second upper bound is unchanged.
Cross-field relationships in the table are validated. Construction fails
before serving; values are never silently clamped. `max_json_depth` applies to
decoded requests, runtime results, and metadata; `max_schema_depth` separately
applies to schema structure. `max_result_bytes` is checked before envelope
projection, while `max_output_line_bytes` includes the final JSON-RPC envelope
and delimiter. `max_batch_items` is enforced before creating element tasks, and
`max_catalog_items` is enforced before sorting, hashing, or paginating a runtime
catalog. `default_catalog_page_size` is the fixed initial page size because MCP
list requests do not carry a page-size parameter; every continuation cursor
binds that configured value.

`GatewayApplicationError` is the common safe runtime exception. It exposes
only bounded `public_message`, stable `reason_code`, and stable `kind` fields;
the underlying exception is never attached to protocol data. Public subclasses
are `GatewayToolExecutionError`, `GatewayResourceNotFound`,
`GatewayResultTooLarge`, and `GatewayInvalidApplicationResult`. Section 14
defines their wire projection.

The base constructor is
`GatewayApplicationError(public_message, *, reason_code, kind="application")`.
`public_message` must be a non-empty string of at most 512 Unicode code points;
`reason_code` must match `[a-z][a-z0-9_]{0,63}`; and `kind` is one of
`application`, `tool`, `resource`, or `prompt`. Subclasses fix their applicable
kind and default reason code. `GatewayResourceNotFound` and
`GatewayInvalidApplicationResult` use fixed generic public messages.
`GatewayResultTooLarge` additionally accepts a positive `limit_bytes` and does
not expose the actual private result size. Invalid error construction fails
locally and is projected as a generic internal failure, never as attacker-
controlled error data.

## 8. Lifecycle and Version Negotiation

### 8.1 Modern `2026-07-28`

Every request must include:

- `_meta["io.modelcontextprotocol/protocolVersion"]`
- `_meta["io.modelcontextprotocol/clientCapabilities"]`

`clientInfo` is accepted but optional. Required metadata missing or malformed
returns `-32602`. Unsupported modern versions return `-32022` with only safe
`requested` and `supported` fields. The `supported` array names all five
revisions the server supports, newest first, as required by the current
versioning contract.

The server implements `server/discover`. Its `supportedVersions` likewise
advertises all five revisions. The client filters that array through its own
modern-profile table before retrying on the same process; a legacy entry is
never retried with modern metadata. Selecting any legacy revision requires a
fresh process and `initialize`. Discovery returns deterministic capabilities,
`resultType: "complete"`, conservative cache hints, and server identity in
reserved result metadata.

Every successful modern result includes `resultType`. Complete application
results use `"complete"`. In legacy profiles an absent discriminator means
complete. Any present unknown or unadvertised value is invalid rather than
retained as additive metadata. The runtime and stdio server do not generate
`"input_required"` in this scope. Server identity is injected into result
`_meta` after runtime execution so application data cannot forge it.

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
3. A well-formed non-modern JSON-RPC method/protocol error, or timeout/EOF
   before the peer emits any invalid bytes: legacy candidate; close the
   disposable probe process, start a fresh process, and initialize.
4. Malformed JSON/envelopes, a response-ID mismatch, trailing/extra protocol
   bytes, or EOF after invalid bytes: fail closed. Do not reinterpret a broken
   or hostile peer as legacy.

The fresh process prevents an unknown legacy implementation from retaining
ambiguous pre-initialize state. Probe timeout and shutdown escalation are
configurable.

## 9. JSON-RPC and Batching

- MCP request IDs are strings or integers. `null` and booleans are rejected.
  The in-flight/cancellation map uses typed keys, so integer `1` and string
  `"1"` remain distinct; duplicate active IDs of the same type are rejected.
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

List methods reject duplicate normalized descriptor identities, sort by the
profile-projected stable identity (`name` for tools/prompts and normalized URI
for resources/templates), and paginate only after enforcing
`max_catalog_items`. The gateway accepts an authenticated opaque cursor bound
to the method, protocol profile, next offset, configured
`default_catalog_page_size`, and a fingerprint of the normalized projected
catalog. A cursor is size-bounded, cannot be used
across methods, and fails with a stable invalid-cursor error if the catalog
changes between pages; it never silently skips or duplicates entries. The
default and maximum page sizes are explicit `GatewayLimits` values. MCP catalog
pages do not invent a total-count field; exact totals are an application tool
contract, not an MCP list-protocol requirement.

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
- A current tool may declare an arbitrary-root `outputSchema`. For a legacy
  profile whose tool schema/result grammar cannot represent that root, the
  gateway validates against the declared schema, omits `outputSchema` and
  `structuredContent` from the legacy projection, and retains the deterministic
  JSON text content block. It does not invent an application wrapper.
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

The package adds the direct dependency `jsonschema>=4.23,<5`, whose public
`Draft202012Validator` supports JSON Schema 2020-12. Protocol validation cannot
depend on an undeclared transitive dependency. The compatible-major ceiling is
part of the public package/release decision and is verified from both wheel and
sdist installs.

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
- sends only the bounded schema and instance JSON to a fresh spawned validation
  worker process; at most `max_schema_validation_processes` run concurrently
- terminates, force-kills if necessary, and reaps the validation worker when
  `schema_validation_timeout_seconds` expires, then returns the stable safe
  `schema_validation_timeout` failure
- joins and reaps the worker on every other exit path: successful verdict,
  validation failure, abnormal child exit, request cancellation, and server
  shutdown. The concurrency permit is released only after reaping completes

The parent performs only bounded structural preflight before spawning. Schema
compilation, format/pattern evaluation, and instance validation all occur in
the disposable worker, so a short catastrophic-backtracking regex cannot pin
the long-lived stdio process. The child has no network `$ref` resolver, returns
only a bounded verdict, and is never reused after timeout or abnormal exit.
Queue admission remains bounded by the process limit and request cancellation;
cancelled requests discard the verdict and reap their worker. Shutdown rejects
new validation work, terminates any live validation children with the same
bounded terminate/kill escalation, reaps them, and then releases their permits.

Dialect support does not loosen MCP descriptor shapes. Every tool
`inputSchema` is a valid object schema with an object root. A `2026-07-28`
`outputSchema` may describe any JSON root. Legacy profiles retain their dated
object-root restriction and use the text-only compatibility projection in
section 11 when a current arbitrary-root schema cannot be represented. A
modern tool without an `outputSchema` may return any JSON value as
`structuredContent`.

Schema-validation errors return bounded public messages without echoing the
arguments, schema, content, pattern, or exception representation.

## 13. Stdio Execution, Cancellation, and Limits

The long-lived server uses one reader loop, one serialized writer, and a
bounded map of in-flight request tasks. Reading continues while async work is
running so cancellation can be observed.

Default package limits are the exact `GatewayLimits` fields in section 7.5.
The implementation plan may propose a versioned API change after
characterization but may not silently change a default or remove a category.

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

Omitted process streams use a portable binary adapter. On event loops that do
not support asynchronous pipe registration for process stdin/stdout, including
affected Windows configurations, the adapter performs bounded blocking reads
and writes in dedicated threads while retaining the same cancellation, byte,
and shutdown limits. Supported-platform CI exercises the native POSIX path and
the Windows-compatible fallback; `Operating System :: OS Independent` is not
claimed solely from mocked POSIX pipes.

## 14. Errors, Logging, and Privacy

The protocol layer distinguishes:

- malformed/protocol requests: JSON-RPC errors
- unknown method/tool/resource: version-correct JSON-RPC errors
- actionable tool execution/business failures: typed tool results with
  `isError: true`
- unexpected internal failures: safe generic internal errors

All typed application errors carry only the safe public fields defined in
section 7.5. Application handlers must stop converting raw `str(exc)` values
into successful `{error: ...}` payloads.

Version-specific error projection includes:

- modern missing metadata: `-32602`
- modern unsupported version: `-32022` with `requested` and `supported`
- modern missing resource: `-32602`
- legacy missing resource: `-32002` where required for compatibility
- `GatewayResultTooLarge`: application-defined `-33001` with stable
  `result_too_large` data
- other safe non-tool application failures: application-defined `-33002`
- invalid application results and unexpected failures: generic `-32603`
- no undefined modern use of reserved `-32020..-32099` codes

`GatewayToolExecutionError` is projected as a successful `tools/call` envelope
whose result has `isError: true` and a bounded safe text content block; it does
not include `structuredContent` unless a future explicit error schema is
designed. Its stable classification is carried in result metadata under the
valid package-owned key `io.github.rmusser01.mcp-unified/error`:

```json
{"reasonCode": "not_implemented", "kind": "tool"}
```

The gateway owns and overwrites that key; runtimes cannot forge it. The value is
bounded to the validated `reason_code` and `kind` fields and contains no
application payload. Malformed calls and unknown tool names remain protocol
errors.
`GatewayResourceNotFound` uses the profile-specific missing-resource code.
`GatewayInvalidApplicationResult` is never exposed verbatim and maps to the
generic internal error. Error `data` is an allowlist of stable `reason_code`,
`kind`, and safe limit metadata; no payload, schema, URI, path, SQL, actual
private result size, or exception representation is included.

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
- a runtime with empty catalogs is valid even though core capabilities remain
  protocol-advertised
- `serve_stdio` can run from an embedded module entrypoint
- `GatewayRequestContext` additions are optional for existing runtimes
- integer and string request IDs remain type-distinct inside the runtime
- arbitrary current JSON tool results and legacy text-only projection work
- strict stdio accepts `GatewayCoreRuntime` without legacy module methods
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
- discovery and `-32022` advertise all five revisions while modern retry
  filters to mutually supported modern profiles
- capability and descriptor projection
- required modern `resultType`, legacy absent-means-complete behavior, rejection
  of unknown values, and no generated `input_required`
- missing/invalid/null request IDs
- typed request-ID distinction, duplicate active IDs, and cancellation lookup
- notifications and notification suppression
- tools/resources/templates/prompts happy paths and errors
- empty catalogs
- pagination plus stale/repeated/cross-method cursor rejection and catalog
  fingerprint changes
- object and array/scalar/null structured output plus schema validation
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
- the server never writes server-to-client JSON-RPC requests
- concurrent request reading and serialized output
- cancellation before start, during async work, and at the writer race
- no output after cancellation
- EOF and graceful/forced shutdown behavior
- input/output/result/JSON-depth/in-flight/rate/catalog/page/batch/schema limits
- schema-validation process saturation, timeout, cancellation, abnormal exit,
  normal-completion/abnormal-exit reaping, shutdown with live children, and an
  adversarial catastrophic-backtracking pattern while the server remains
  responsive
- payload-free diagnostics
- strict `GatewayProtocolStdioServer` lifecycle behavior and unchanged
  independent-message `GatewayStdioServer`/`handle_stdio_line` behavior

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
- public signatures/defaults/validation for cancellation tokens, limits,
  application errors, strict stdio streams, ownership, and exit semantics
- optional resource-template extension detection with an unmodified core
  `GatewayRuntime` implementation
- package import-boundary tests
- Python 3.10 through the supported upper CI version
- POSIX native-pipe and Windows-compatible binary-stream execution
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
| Schema validation enables SSRF or CPU exhaustion | Direct bounded `jsonschema` dependency; no external `$ref`; bounded schema structure plus disposable validation processes with exact concurrency/time limits |
| Cancellation races emit a late result | Cancellation state rechecked under serialized writer immediately before output |
| Large content deadlocks or corrupts stdout | Bounded line/output, explicit error, continuous stderr draining, no truncation |
| Cache metadata leaks user-specific catalogs | `ttlMs: 0`, `cacheScope: private`, authorization remains independent |
| Old and modern fields are mixed | Projection tests assert required presence and forbidden absence per revision |
| Public helper silently changes lifecycle semantics | Preserve `GatewayStdioServer`; put strict behavior in `GatewayProtocolStdioServer` |
| Discovery advertises legacy versions that a client retries as modern | Advertise the complete required set, then filter retries through the client's modern profile table |
| A moving catalog skips or duplicates paginated descriptors | Stable identities, duplicate rejection, catalog fingerprints, and stale-cursor failure |
| A Windows event loop cannot register binary stdio pipes | Portable bounded thread-backed adapter and supported-platform CI |
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
`Docs/ADR/033-mcp-unified-stdio-contract-hardening.md` (supersedes ADR-032)

Reason: the change establishes a public protocol/runtime API, dependency and
release boundary, cross-version service contract, and security/privacy policy.
