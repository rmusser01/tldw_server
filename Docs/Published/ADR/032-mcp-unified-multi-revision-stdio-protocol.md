# ADR-032: MCP Unified Multi-Revision Stdio Protocol

**Status:** Superseded by ADR-033
**Date:** 2026-08-08
**Superseded by:** [ADR-033](033-mcp-unified-stdio-contract-hardening.md)
**Backfilled from:** `backlog/decisions/001-mcp-unified-multi-revision-stdio-protocol.md`
**Decision owner:** MCP Unified protocol design review
**Related task:** `backlog/tasks/task-13008 - Design-MCP-Unified-multi-revision-stdio-protocol-compatibility.md`
**Related spec/plan:** `Docs/superpowers/specs/2026-08-08-mcp-unified-multi-revision-stdio-protocol-design.md`

## Decision

`mcp-unified` will provide a strict, reusable stdio protocol engine for MCP
revisions `2026-07-28`, `2025-11-25`, `2025-06-18`, `2025-03-26`, and
`2024-11-05`. One immutable protocol-profile table controls lifecycle,
projection, errors, and batching; batching is accepted only after a standalone
`2025-03-26` initialization.

The modern server advertises all five revisions in `server/discover` and in an
unsupported-version error. A client may retry the same process only with a
mutually supported modern revision. A legacy selection always starts a fresh
process and uses `initialize`, so a legacy revision is never sent with modern
per-request metadata.

The strict public API adds `GatewayProtocolStdioServer`, `serve_stdio`, explicit
cancellation/limit/application-error primitives, and an optional resource-
template runtime extension. The existing `GatewayStdioServer` and
`handle_stdio_line` retain their historical one-message behavior. Existing
`GatewayRuntime` implementations do not acquire a statically mandatory method.

The package owns envelopes, lifecycle, schema/result validation, pagination,
cancellation, bounds, and safe error projection. An injected runtime owns
catalogs, application execution, authorization, and private data. JSON Schema
2020-12 is supported without automatic network `$ref` resolution, while MCP
tool input and output schemas remain object-rooted where the protocol requires
that shape.

Modern successful results always include `resultType: "complete"`. A missing
legacy `resultType` means complete; unknown values are invalid.
`input_required` and server-to-client JSON-RPC requests are not implemented by
this stdio server.

Modern compliance remains stdio-only. Existing FastAPI/WebSocket and legacy
one-message gateway behavior is unchanged. The package must pass installed-
artifact and downstream-consumer tests and be published before Chatbook
migrates.

## Context

The existing standalone gateway is a useful application/runtime boundary, but
its one-message stdio helper is not a long-lived current-protocol server. It
hard-codes a legacy revision, permits behavior that is no longer version-
agnostic, and lacks the cancellation, output, and schema contracts required by
an embedded downstream consumer.

The first design record for this work was mistakenly placed under
`backlog/decisions/`. ADR-001 establishes `Docs/ADR/` as the canonical location
and forbids changing accepted decisions in place. This ADR therefore
supersedes that record and incorporates the approved compatibility and public-
API corrections.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Replace the package with the official MCP SDK | Discards the existing gateway/runtime boundary; the SDK remains an interoperability oracle. |
| Wrap FastMCP | Retains the dependency and decorator/runtime coupling that downstream migration removes. |
| Change `GatewayStdioServer` to strict long-lived semantics | Silently breaks a public helper whose callers rely on independent one-message handling. |
| Add resource-template listing to `GatewayRuntime` directly | Makes an additive capability statically mandatory for every existing runtime. |
| Advertise only modern revisions during discovery | Contradicts the current discovery/versioning contract, which describes all revisions the server supports. |
| Retry a discovered legacy revision with modern metadata | Mixes incompatible lifecycle eras; legacy negotiation requires a fresh process and `initialize`. |
| Upgrade HTTP/WebSocket in the same effort | Expands scope into transport routing and authorization while risking existing server behavior. |
| Permit batches for all revisions | Violates the approved profile-specific contract; only initialized `2025-03-26` receives batches. |

## Consequences

The package gains a clear compatibility table and an embeddable strict stdio
surface without taking ownership of application data. Existing callers keep
their public one-message helper semantics, and existing runtimes remain
structurally compatible.

Five revisions and two intentionally distinct stdio surfaces increase fixture
and maintenance cost. Strict stdio also intentionally corrects unsafe legacy
behavior such as nullable IDs, raw errors, unbounded lines, globally enabled
batching, and ambiguous result types. Synchronous runtime work cannot be
forcibly stopped inside a Python thread; cancellation suppresses its output and
hosts must keep such work bounded.

## Follow-up

- Implement and verify `TASK-13008` only after the design review gate.
- Publish and independently verify the package artifact before the downstream Chatbook migration.
