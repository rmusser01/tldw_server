# ADR-033: MCP Unified Stdio Contract Hardening

**Status:** Accepted
**Date:** 2026-08-08
**Supersedes:** [ADR-032](032-mcp-unified-multi-revision-stdio-protocol.md)
**Decision owner:** MCP Unified protocol design review
**Related task:** `backlog/tasks/task-13008 - Design-MCP-Unified-multi-revision-stdio-protocol-compatibility.md`
**Related spec/plan:** `Docs/superpowers/specs/2026-08-08-mcp-unified-multi-revision-stdio-protocol-design.md`

## Decision

All unchanged lifecycle, revision, discovery, batching, transport-isolation,
release, and application-ownership decisions from ADR-032 remain adopted. This
ADR corrects and makes exact the reusable public contract before implementation.

The strict stdio surface accepts a new narrow `GatewayCoreRuntime`; the existing
`GatewayRuntime` remains unchanged for legacy gateway callers and retains its
package-specific module methods. `GatewayRequestContext.request_id` widens to
`str | int`. The public `GatewayJSONValue` alias and the core `call_tool` return
contract support every finite, recursively bounded JSON value while preserving
structural compatibility for dictionary-returning implementations.

For `2026-07-28`, tool `inputSchema` remains object-rooted but `outputSchema`
and `structuredContent` may use any JSON root. When a legacy profile cannot
represent an arbitrary structured root, the gateway validates the application
value against its declared schema, omits the incompatible legacy schema and
structured field, and emits deterministic JSON text. It never invents an
application wrapper.

The package directly depends on `jsonschema>=4.23,<5`. Immutable limits
separately bound wire lines, decoded JSON depth, application-result bytes,
aggregate catalogs, a fixed default/maximum page size, batch items,
concurrency/rate, schema complexity, and shutdown. Schema compilation and
instance validation run in disposable spawned worker processes under exact
concurrency and wall-time limits, so pathological regex evaluation cannot pin
the long-lived server. Every validation-child exit path is bounded and reaped,
and a concurrency permit is released only after reaping. Pagination rejects
duplicate identities, uses deterministic ordering, and binds an integrity-
protected cursor to method, profile, configured page size, offset, and
normalized catalog fingerprint; a moving catalog invalidates the cursor.

POSIX hosts use the multiprocessing `spawn` worker. Native Windows hosts use
an equivalent fixed-argument Python subprocess because protected official-SDK
stdio execution demonstrated that nested multiprocessing reconstruction could
not complete reliably there. The Windows handoff is an exact-length,
owner-only temporary payload; the child has no shell, stdin, inherited stdio,
or network resolver, emits only the same bounded verdict, and the parent
removes the payload only as part of bounded child cleanup.

`GatewayToolExecutionError` remains a protocol-valid `isError: true` tool
result. Its safe stable classification is carried under the gateway-owned
result metadata key `io.github.rmusser01.mcp-unified/error` with bounded
`reasonCode` and `kind`. Omitted process streams use native asynchronous binary
pipes where supported and a bounded thread-backed binary adapter otherwise, so
the package's platform claim includes Windows-compatible behavior.

## Context

Review of ADR-032 against the dated current specification and the existing
package API found six contradictions: current output schemas were accidentally
kept object-rooted; `request_id: str` collapsed typed identity at the runtime
boundary; strict consumers inherited unrelated module-health methods; generic
JSON/result/catalog/batch bounds were absent; stable tool failure reasons had no
wire location; and cursor/platform behavior was underdefined.

ADR-001 makes accepted ADRs immutable. These corrections therefore supersede
rather than edit ADR-032.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep object-only runtime results | Cannot represent valid current array, scalar, boolean, or null structured outputs and schemas. |
| Reuse `GatewayRuntime` for strict stdio | Forces downstream MCP core consumers to implement package-specific module aliases. |
| Stringify integer request IDs | Collapses integer `1` and string `"1"`, breaking correlation and cancellation identity. |
| Wrap arbitrary JSON into an object for legacy clients | Invents application semantics and can contradict the declared output schema. |
| Leave validator choice or execution budget to implementation | Makes undeclared dependency and denial-of-service decisions after the design gate. |
| Automatically continue a page after catalog movement | Can silently skip or duplicate descriptors. |

## Consequences

The implementation surface is larger but testable and downstream-complete.
Existing gateway runtimes and compatibility stdio helpers remain valid. Current
tools can use the full protocol result space, while legacy clients receive an
honest text-only fallback when necessary. The release must prove typed IDs,
arbitrary JSON roots, all new limits, moving cursors, error metadata, direct
dependency packaging, and both native/fallback binary stdio paths.

## Follow-up

- Create a separate implementation task only after this corrected design gate.
- Publish and independently verify the package artifact before Chatbook
  migration begins.
