# PostgreSQL multi-user TestBot timeout — bounded diagnosis

Parent TASK13260. Read-only audit of frozen `8f8774e6` matrix source and retained native evidence. **The native turn failed; the exact upstream reason remains unproven.** No product change, task/tracker mutation, browser action, request, database query, inference or process control was performed.

## Confirmed request sequence

- Native Character creation/entry succeeded. Conversation `3165d63b-f3b1-4629-9d25-b5fd36f5acd6` contains the acknowledged user `2ca9b706-8d0d-406f-b0b9-34b828e58753`, “Hello, who are you?”.
- The browser sent `complete-v2` at **17:28:00.726Z**, using provider `llama`, the advertised Gemma model, `stream:true`, `save_to_db:true`, and character context. No explicit `max_tokens` or stream timeout was in that request body.
- A bounded private-log projection confirms WorldBook initialization completed at **17:28:00.787Z** and HTTPX received **HTTP200 from http://127.0.0.1:9099/v1/chat/completions at17:28:00.831Z**. The full URL was parsed privately; only protocol/loopback host/port/path/status were projected. No headers or upstream content were retained.
- Browser response headers arrived **17:28:01.096Z**; the capture's body-read attempt settled **17:28:46.133Z**, **45.037 seconds later**. The response body is absent from the retained observer output. This is not an empty successful SSE body assertion.
- UI displays **“Your chat timed out.”** The captured message reads before timeout contain only the original user. Ordinary research-run/settings/other scoped reads continued to return200; token-expiry warnings begin later, at17:29. The earlier WorldBook list500 is separately confirmed UAT239 and did not prevent this completion from reaching the provider.

## Source-backed classification

Frozen `apps/packages/ui/src/services/background-proxy.ts:1469–1483` uses a **45,000ms fallback idle budget**. Only `/api/v1/chat/completions` consults `chatStreamIdleTimeoutMs`; `complete-v2` uses general `streamIdleTimeoutMs` or this fallback. The direct stream implementation arms the timer after successful headers (`:1789`) and resets it on **each raw reader chunk**, before line parsing (`:1792–1795`). A silent interval aborts its controller with `Stream timeout: no updates received` (`:1681–1690`), surfaced by the existing chat error mapper.

The active Character hook has its own **60-second** inactivity watchdog (`hooks/chat/useChatActions.ts:2288–2323`); it calls the real stream client without an explicit timeout override. The actual client forwards the option if present (`services/tldw/domains/chat-rag.ts:1361–1383`). Backend `character_chat_sessions.py:284–286,497–559` allows300seconds per provider call/next chunk. These distinct budgets support the frontend45-second idle interpretation; the canonical browser timeout configuration was not read, so the actual saved value is not independently established.

The tracked backend streams normalized provider chunks without a PostgreSQL-specific branch (`character_chat_sessions.py:6395,6794–7015`). Unified heartbeat is opt-in; its live environment value was not inspected. The local adapter reads upstream lines and yields normalized SSE (`providers/local_adapters.py:450–506`); proper `data:` lines retain their payload (`LLM_Calls/sse.py:121–124`). Thus **hidden reasoning content alone is not evidence of an idle period**: raw reasoning bytes should reset the browser timer even before visible final text. Upstream comment/control frames may be dropped by the adapter's documented default; no capture proves that occurred here.

**Most likely interpretation:** a first-byte/stream-delivery stall at or downstream of the local provider exceeded the existing client idle budget. The timing is particularly consistent with no delivered chunk after headers. This is an inference, not a measured absence of every chunk. The current evidence cannot distinguish model prefill/queue delay, provider stream stall, adapter forwarding, or same-origin proxy delivery. It does not establish a PostgreSQL persistence/lock failure, an expired-login cause, an exact prompt mismatch, or an incorrect reasoning parser. Do not close the failed native journey as harmless, increase a timeout to hide it, or claim a new product defect solely from this timing.

## SQLite multi-user comparison

Retained SQLite TestBot uses the **same provider, advertised model, stream/save settings and user prompt**, in conversation `ea6efc19-3395-4640-8757-88bd6f0be6ec`. Request15:58:32.965Z → headers15:58:33.219Z → observer body-read15:58:47.900Z: **14.681 seconds after headers**. Canonical reload has one user and one assistant; the visible final answer is BEEP BOOP. The stored response includes reasoning before its final answer; only existence/length/final-answer metadata, not that content, is projected in the companion JSON.

Relevant adapter/endpoint/frontend source hashes match between the two frozen cells (companion JSON). Character names differ by the declared SQLite/PostgreSQL fixture suffix. The retained request payload does not expose the complete backend-assembled provider prompt, effective token budget, or token arrival timestamps. A shorter successful request does not prove that a45-second failure is database-specific.

## Provider-log provenance limit

The suggested `.tmp/fresh-uat-recovery-20260916/llama-server.log` is only883bytes and last modified **2026-09-16T12:24:20-0700**, before this run. A read-only process-name/start-time listing identifies PID9146 `./llama-server`, startedSep16 12:11:40. Read-only listener/descriptor queries returned exit1 with no rows, including the explicitly identified process; this does **not** establish the provider is absent. It does not bind the stale file to the current9099 traffic. Its contents were not used to invent a completion/throughput result. No valid current provider token-timing receipt is available in this audit.

## Existing issue/task overlap

- Existing **TASK12108**, “Consolidate triplicated characterChatMode and add stream-inactivity watchdog to the live path”, explicitly covers stalled Character streams and shipped-path coverage. Its source-string guard confirms a60s watchdog but cannot prove this actual transport hop. This is the closest existing tracking context; no duplicate task was created.
- **TASK13260.91 / UAT152–153** fixed Settings shortening *request/startup/RAG generation* budgets to10seconds. The documented general stream idle budget remains distinct. This observation is not evidence of that repaired default regression.
- **TASK13260.57 / UAT117** concerns completed reasoning-only output; this request did not have a retained completed response, so that classification is unsupported.
- **UAT239** covers the earlier WorldBook list wrapper500. Provider dispatch200 and completed predispatch initialization separate it from this stalled response.

## Smallest proposed causal regression scope, before any repair

1. Extend the **actual live Character action → real `streamCharacterChatCompletion` → real `bgStream`** harness with controlled `fetch`/`ReadableStream` only. Headers200 plus no bytes must produce the existing timeout once and leave the exact acknowledged user/conversation; no automatic POST replay or false assistant success. A role/reasoning/data chunk just before the deadline must reset the timer, allowing a total response longer than45seconds. Also cover explicit general stream timeout and caller cancellation. Fake timers avoid slow wall-clock tests. Do not rely on the existing source-string watchdog guard or unused extracted mode.
2. If that boundary behaves as designed, use the **actual complete-v2/local adapter streaming boundary** with a deterministic upstream line source, real SQLite/official-PG fixtures, equivalent owned Character context and held first data. Assert first chunk forwarding before terminal completion, cleanup on client disconnect, preserved owner/transaction state and no persistence duplication. This probes an actual backend regression without model nondeterminism or new database provisioning.
3. Only if provider/proxy ambiguity remains, collect **first upstream data timestamp/first downstream byte timestamp and model queue/prefill metadata** during one parent-authorized native attempt. Preserve the original failed turn; do not rerate/retry or modify budgets just to obtain a pass. No such attempt was made by this auditor.

Existing useful fixtures include `tests/Character_Chat/test_complete_v2_streaming_e2e_mock.py`, `test_complete_v2_streaming_with_mock_openai.py`, `test_complete_v2_streaming_unified_flag_monkeypatched.py`, frontend `services/__tests__/background-proxy.test.ts`, and the actual `useChatActions.character.integration.test.tsx`. The current watchdog guard is source-string coverage and insufficient alone.

## Evidence and limits

Companion JSON pins the five original PG receipts, SQLite successful reload receipt, compared frozen source paths and relevant task records with SHA256. The private backend log was read only through bounded timestamp/function/URL metadata; raw log content and credentials are excluded. No current native database state or runtime environment was queried. No tests were run. The audit preserves the failed native outcome and leaves repair/classification to a causal boundary result.
