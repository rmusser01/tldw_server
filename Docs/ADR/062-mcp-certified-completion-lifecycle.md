# ADR-062: MCP Certified Completion Lifecycle

**Status:** Proposed
**Date:** 2026-10-03
**Backfilled from:** not backfilled
**Decision owner:** MCP adapter workstream requester and reviewers
**Related task:** TASK-2294.3.2
**Related spec/plan:** `Docs/superpowers/specs/2026-07-23-mcp-skills-model-only-runner-design.md`; `Docs/superpowers/plans/2026-09-07-mcp-bounded-model-completion-adapter-implementation-plan.md`

## Decision

Certify only the native-async, single-attempt OpenAI completion path. Capture the provider, model, endpoint, native output-token field, and timeouts from operator configuration. Generate credential headers from an authoritative per-call runtime. Send one non-streaming, one-choice request with no tools, redirects, retries, or fallback.

The managed adapter owns each completion child through termination. Run timeout and cancellation cleanup have separate deadlines. A child that outlives cleanup is abandoned permanently, remains referenced, leaves conservative durable exposure, and makes the adapter unhealthy. Late results cannot publish output or reconcile an ambiguous reservation. Shutdown stops admission and drains owned work. Only an explicit health check may restore readiness after retained work terminates and the full capability contract is revalidated.

## Context

The generic provider stack supports behaviors unsuitable for a bounded MCP model runner, including multiple providers, streaming, retries, and tools. Cancellation of a caller does not prove that a paid provider operation stopped. Task references and durable reservation fences are both necessary: local ownership contains work, while the repository prevents late settlement after abandonment.

ADR-025/026 govern provider routing and outbound egress. ADR-060 fixes credential authority and endpoint provenance; ADR-061 supplies durable admission and settlement. This decision composes those boundaries without exposing a new MCP tool or enabling Skills execution.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Reuse the generic provider adapter | Its optional retries, tools, streaming, and fallbacks broaden certification. |
| Use a synchronous worker with an async wrapper | Caller cancellation cannot prove transport termination. |
| Drop a timed-out child reference | Orphans work and loses shutdown ownership. |
| Accept a late success after cleanup expires | Violates deadline semantics and can race conservative settlement. |
| Classify shared failures by HTTP status alone | Credential and request-local errors could open the global breaker. |

## Consequences

Bounded JSON reads propagate the captured endpoint scope through each byte-stream adapter and every egress decision. The application consumes at most the decoded response cap plus one overflow byte before rejection and retains no oversized JSON envelope. Chunked bodies, forged lengths, and compressed expansion cannot bypass that application boundary. This is not a bound on internal HTTP-library decompression allocations; codec implementations remain part of transport dependency risk.

Output normalization requires exactly one textual choice, rejects tool-call fields and disallowed controls, validates strict UTF-8, and changes only CRLF/CR to LF. Character and byte overflow rejects the whole output, never truncates it. Optional provider counts are trusted only within integer/request accounting bounds; invalid usage retains conservative estimates.

HTTP status, egress-policy denial, and provider content failures are breaker neutral. Shared failures require explicit provenance from the frozen transport or required shared services. Arbitrary exceptions are detached from sanitized port failures; credentials, prompts, completion bodies, and endpoint queries are excluded from operational logging and persistent accounting.

Owned cancellation draining must also contain the event-loop exception channel. On Python 3.14, cancelling `asyncio.shield()` can install a callback that logs a later private child exception even when the owner subsequently retrieves it. Wait for the owned task without that callback, preserve native caller cancellation, and consume every terminal result or exception. Tests capture both the event-loop handler and default asyncio logger, in addition to application logging.

Valid output survives post-success accounting or credential-usage persistence failure. Native cancellation still propagates, and the reservation remains conservative when the outcome is unknown. The adapter stays unavailable when any required capability is missing.

The shared credential runtime keeps its legacy bounded `close()` behavior, but exposes truthful pending-work and shutdown-drain operations. Resolver and usage tasks remain owned from creation until termination, including tasks removed from active maps by an earlier bounded drain. The adapter owns that drain boundary; a returned `close()` alone does not establish health or shutdown completion.

Post-close lifecycle API faults are also contained. An owned supervisor revalidates the public pending-work and async drain contract and requires positive empty-work confirmation before releasing ownership. Faults or premature drain completion keep the adapter unavailable; short delayed rechecks avoid a busy loop, and a single fixed warning bounds observability. This supervision never retries the paid provider operation. Persistent API failure requires recovery of the lifecycle boundary before shutdown can truthfully complete.

Supervisory backoff preserves a monotonic retry deadline across task cancellation. Positive-empty completion also delivers pending ancillary task cancellation inside containment before the supervisor returns normally. Internal cancellation must not become public caller cancellation or disclose its private argument; genuine caller cancellation and shutdown still suppress output. No caller cancellation state is cleared by this boundary.

The public boundary also distinguishes cancellation of a completed owned invocation from cancellation of its caller. After invocation/shutdown guards, only an existing timely validated receipt permits recovery from that owned task's `CancelledError`. Without such a receipt, provider-native cancellation remains native. Genuine caller cancellation always wins, and neither bookkeeping retries nor caller cancellation-state clearing are permitted.

Cancellation and shutdown take precedence over result publication. Otherwise, immutable validated content received before the original monotonic run deadline is cached before bookkeeping awaits. Ancillary bookkeeping or cleanup failure cannot turn that known paid completion into a retriable failure. If ancillary work misses cleanup, it remains retained, health is latched false, and settlement is fenced conservatively; publication uses only the earlier receipt, never the retained child's eventual result. Provider content received after the deadline is never cached or published. False persistence outcomes produce fixed degradation warnings without private context.

The native transport synchronously hands validated content to the invocation-local receipt sink before awaiting client cleanup. The adapter applies its original deadline and cancellation/shutdown guards at that handoff. Close errors or native close self-cancellation invalidate transport readiness without deleting an earlier timely receipt; caller cancellation during successfully drained closure does not itself invalidate certification. The callback is a trusted internal composition detail, not part of the public completion port.

Without a receipt, an already classified provider failure also survives native client-close self-cancellation. Contain that cancellation inside the independent owned close task, latch readiness false, and preserve the sanitized failure and its original domain. The separate owned wait still propagates genuine caller cancellation; provider-native cancellation remains native. Cleanup-private cancellation arguments must not replace a known public failure.

The shared HTTP client's certificate-pinning preflight currently opens a synchronous TLS socket. The bounded MCP transport cannot certify native cancellation when pins apply to its selected endpoint. It rejects that configuration without bypassing operator pins, and rechecks the fresh client's pin state before dispatch to contain later configuration changes. Pins for unrelated hosts do not invalidate the selected path. Supporting selected-endpoint pinning requires a separately certified native-async implementation.

## Follow-Up

Stage 5 composes and verifies the host factory. TASK-2294.3.3 may consume the managed port for the disabled-by-default Skills runner only after those gates pass. Other providers require separate certification, not automatic fallback.
