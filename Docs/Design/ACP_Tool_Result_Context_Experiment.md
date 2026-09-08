# ACP/MCP tool-result context experiment

Task: TASK-13219. Approved in conversation on 2026-09-07 following the Spotify architecture assessment.

## Scope and decision

Extend the existing LLM-driven MCP runner with an opt-in policy at the boundary where successful tool output enters model history. Compare `off`, `excerpt`, and `worker`. Existing agent-driven ACP sessions are outside this experiment. TASK-13218 separately covers agentic RAG compression.

Deterministic excerpts are the cheapest baseline. Worker mode selects server-numbered source segments; it does not author summaries or invent citations. This avoids unreliable model-generated character offsets. The final model can request exact surrounding text through a bounded, run-local `tldw_read_tool_result` tool.

## Contracts

- Default is `off`, with unchanged model history, tool schemas, and event payloads.
- Budget UTF-8 bytes of decoded model-facing text. JSON transport escaping/headers and model tokens are separate measurements. Only oversized successful text results are eligible; errors pass through.
- Store eligible original text in a bounded per-run memory store. References are opaque, run-local, never filesystem paths. If storage capacity is exceeded, preserve the original result and record the reason rather than advertise unavailable evidence.
- Excerpts preserve exact text and character offsets. Worker selection is strict JSON containing unique valid segment IDs. Invalid, empty, oversized, failed, or timed-out selections use deterministic excerpts. Worker input is bounded, with no tools and no conversation history beyond the current question. Cancellation propagates and cancels the worker.
- Deterministic ranking uses at most 128 unique terms from the first 8192 question characters, folds each segment once, and yields/checks cancellation every 64 segments. Successful worker selection skips this ranking. Query terms beyond those limits can affect relevance and require source rereads.
- Follow-up reads recheck the original tool name and arguments through the existing ToolGate. Unknown references disclose no source. Source state is discarded when a run ends, including error and cancellation exits.
- Inject the worker through the existing LLMCaller interface. Adapter configuration must explicitly name the same provider as the main caller; automatic cross-provider fallback is excluded. Injected caller implementations remain responsible for authentication, actual model identity, usage accounting, and egress policy.
- Build worker requests incrementally, accounting for UTF-8 and nested JSON escapes, and stop at the input budget. Encoding yields for cancellation. Requests rejected before complete encoding have unknown total request bytes. A worker cancelling its own request uses deterministic fallback; run/caller cancellation still propagates.
- Preserve raw TOOL_RESULT payloads for existing consumers. Add content-free comparison metadata describing input/output sizes, selection outcome, worker latency and available usage. Optional usage is unknown when absent, never treated as zero cost.
- If optional selection is cancelled after tool execution completes, emit the raw TOOL_RESULT exactly once before propagating cancellation. No selection metadata is claimed for unfinished preparation. Internal source reads do not count as first tools or typed-tool fallbacks in run-first rollout metrics.
- Extend AdapterConfig.protocol_config, without adding public REST fields or database migrations. The repository currently defines the LLMCaller interface but no production concrete implementation; this experiment is available to embedding callers and the comparison harness, not automatically to external coding-agent sessions.

## Evaluation and rollout

Provide a local comparison harness with fixed source fixtures and independently specified expected evidence. Run the real policy and follow-up retrieval for each mode. Its default worker is explicitly a deterministic test double: results verify machinery, not model quality or savings. Allow an explicit caller factory for experiments with a configured real worker. Report actual byte counts, elapsed time, evidence inclusion/recovery, and worker-provided usage; do not fabricate token counts or prices.

Evidence presence requires an expected quote inside an exact source range whose labeled text was actually returned. Wrapper headers and read instructions are excluded. Passthrough is checked against the original source; recovery validates returned read text before crediting success.

Before operational rollout, replay representative completed tasks using real approved main/worker callers. Compare completed-task cost including worker and cached tokens, p50/p95 latency, task success, citation validity, and extra reads. Keep baseline, deterministic, and worker arms separate. Source selection can omit decisive evidence even when every quote is exact.

## Verification

Unit and Unicode property tests cover byte bounds, exact offsets, input/storage limits, selection validation, failure, timeout, and cancellation. Runner and adapter tests cover default parity, selection visibility in subsequent LLM calls, unchanged source events, authorization on reread, isolation, lifecycle cleanup, provider checks, and reserved-name collisions. Run focused existing ACP/MCP regressions, Ruff, and Bandit on touched Python files.
