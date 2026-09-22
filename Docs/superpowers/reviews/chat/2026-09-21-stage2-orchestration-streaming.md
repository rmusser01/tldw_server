# Stage 2 — Core orchestration and streaming (`core/Chat`)

Date: 2026-09-21. Read-only audit.

## Scope

The three hot files: `chat_orchestrator.py` (1,670 LOC / 34 commits), `chat_service.py`
(7,285 / 148) and `streaming_utils.py` (2,541 / 42). Correctness, duplication and efficiency on the
request path. Seed cluster **C8** (error → HTTP status) and the **SSE frame formatting** secondary
cluster are resolved here; seed cluster **C1** (base64 cursor padding) is resolved here because its
one in-module site is a signed-token codec, not a cursor.

Seed clusters **C2, C3, C4, C5, C6, C7, C9, C10** were checked against this module and have no
material sites — see `## Not covered` in stage 3.

## Code Paths Reviewed

- `core/Chat/chat_orchestrator.py:_get_http_status_from_exception (247-274)`,
  `:_get_http_error_text (277-294)`, `:_is_network_exception (296-305)`,
  `:chat_api_call (342-593)` error block at `(550-589)`,
  `:chat_api_call_async (594-703)` error block at `(667-690)`,
  `:_CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS (111-116)`
- `core/Chat/chat_orchestrator.py:_chat_sync_impl (869-1190)` vs `:achat (1379-1670)`
- `core/Chat/chat_service.py:estimate_tokens_from_json (3000-3010)`, `:_sanitize_data_uris (3015-3020)`,
  `:_sanitize_messages_for_token_estimate (3023-3052)`, `:_estimate_tokens_from_messages (3054-3060)`
- `core/Chat/chat_service.py:build_context_and_messages (4009-4518)`, history loop at `(4180-4300)`
- `core/Chat/chat_service.py:write_mandatory_moderation_audit (3513-3544)` vs
  `core/Chat/moderation_pipeline.py:write_mandatory_moderation_audit (140-172)`
- `core/Chat/chat_service.py:perform_chat_api_call_async (2653-2716)`,
  `:_attach_internal_http_hooks (2514-2516)`, `:_get_llm_registry (2532-2534)`
- `core/Chat/prompt_cost_envelope.py:estimate_segment_tokens (95-100)`,
  `:_canonical_json (177-178)`, `:_sanitize_data_uris (181-200)`
- `core/Chat/chat_loop_approval.py:_b64url_encode (16-17)`, `:_b64url_decode (20-22)`,
  `:_canonical_json (25-26)`
- `core/Chat/streaming_utils.py:_SSE_CONTROL_PREFIXES (57)`, `:_SSE_FRAMED_CONTROL_PREFIXES (58)`,
  `:_extract_text_from_upstream_sse (1014-1095)`, `:StreamingResponseHandler (1281-2285)`
- Cross-module, read for the scenario only: `core/LLM_Calls/error_utils.py:126-150`,
  `core/LLM_Calls/error_utils.py:build_sanitized_chat_error (~110-123)`,
  `core/LLM_Calls/providers/base.py:_raise_sanitized_provider_failure (85-108)` and `(158-190)`,
  `core/Local_LLM/http_utils.py:54-77`, `core/LLM_Calls/sse.py:19,41-76`,
  `core/exceptions.py:NetworkError (536-555)`, `:ChatAPIError (794-806)`,
  `:ChatRateLimitError (852-860)`, `:ChatProviderError (862-879)`,
  `core/http_client.py:_TerminalHTTPStatusError (114-119)`,
  `:_terminal_status_network_error (122-125)`, `:raise_for_status (778-783)`,
  `core/DB_Management/chacha/message_store.py:get_message_metadata (1689-1721)`,
  `:get_message_metadata_map (1724-1765)`

## Tests Reviewed

Import-grep reachability, **not** measured coverage. Counts from
`2026-09-21-stage3-test-inventory.txt`.

| Test file | Protects | Downgrades the risk? |
| --- | --- | --- |
| `tests/Chat/unit/test_error_handling.py` | Chat exception → HTTP mapping shapes | **No** for `chat-1`: it exercises exceptions that already carry `.status_code`, which is the attribute branch, never the `NetworkError` message-regex branch. |
| `tests/Chat/unit/test_chat_orchestrator_contract.py` | `chat_api_call` dispatch/normalization | No for `chat-1` — same reason. |
| `tests/Chat/unit/test_chat_service_fallback.py` | provider fallback + `write_mandatory_moderation_audit`; also sets `STREAMS_UNIFIED` | Partially — it imports the `chat_service` copy of the audit helper, not the `moderation_pipeline` copy (`chat-8`). |
| `tests/Chat/unit/test_streaming_utils.py`, `tests/Chat/unit/test_streaming_structured_events.py`, `tests/Streaming/test_chat_completions_sse_unified_flag.py` | SSE frame content and the unified-stream flag | Yes for behavior, **no** for `chat-6`: they assert on the emitted bytes, which is exactly why 33 hand-built frames can drift without failing anything until one of them is wrong. |
| `tests/Chat/unit/test_chat_service_token_estimates.py`, `test_chat_service_queue_estimate.py`, `tests/Chat/unit/test_prompt_cost_envelope.py`, `test_prompt_cost_guardrails.py` | the three token estimators — separately | **No** for `chat-7`: each estimator is tested against itself; nothing asserts the three agree. |
| `tests/Chat/unit/test_chat_history_multi_image.py`, `test_chat_persistence_content.py`, `tests/Chat/unit/test_failed_retry_provider_order.py` | the history loop that contains the N+1 | Yes for correctness, no for `chat-3` — they are correctness tests, not performance tests. |
| `tests/Chat_NEW/unit/test_chat_loop_approval.py` | the HMAC approval token round-trip (`chat-11`) | Yes — a single-implementation round-trip test; it cannot catch cross-copy canonicalization drift. |
| `tests/Chat_NEW/unit/test_chat_sync_wrapper.py`, `tests/Chat/unit/test_chat_workflows.py` | the `chat()`/`achat()` twins (`chat-4`) | Weakly — the wrapper guards are covered; the 320-line bodies are not differentially tested against each other. |

## Validation Commands

```
$ grep -rn 'HTTP\\\\s' --include='*.py' tldw_Server_API/
tldw_Server_API/app/core/Chat/chat_orchestrator.py:268:        match = re.search(r"HTTP\\s+(\\d{3})", str(exc))
tldw_Server_API/app/core/LLM_Calls/error_utils.py:145:        match = re.search(r"HTTP\\s+(\\d{3})", str(exc))
(2 hits)

$ grep -rn 'HTTP\\s' --include='*.py' tldw_Server_API/ | grep -v 'HTTP\\\\s'
tldw_Server_API/app/core/Local_LLM/http_utils.py:72:        match = re.search(r"HTTP\s+(\d{3})", str(exc))
(1 hit)
```

```
$ python3 -c "import re; print(re.search(r'HTTP\\\\s+(\\\\d{3})', 'HTTP 429')); print(re.search(r'HTTP\\s+(\\d{3})', 'HTTP 429'))"
None
<re.Match object; span=(0, 8), match='HTTP 429'>
```

```
$ grep -rn 'f"data: \|"data: \[DONE\]\|f"event: ' tldw_Server_API/app/core/Chat/*.py | wc -l
34
$ grep -rc 'f"data: \|"data: \[DONE\]\|f"event: ' tldw_Server_API/app/core/Chat/*.py | grep -v ':0'
tldw_Server_API/app/core/Chat/chat_service.py:6
tldw_Server_API/app/core/Chat/streaming_utils.py:28
  → 34 raw hits, of which streaming_utils.py:1021 is a docstring, so **33 real frame
    constructions**: chat_service.py 6, streaming_utils.py 27. Full list in
    2026-09-21-stage2-sse-inline-sites.txt.

$ grep -rn 'LLM_Calls.sse\|LLM_Calls import sse\|sse_data\|sse_done' tldw_Server_API/app/core/Chat/ | wc -l
0
$ grep -rln 'LLM_Calls.sse import\|LLM_Calls import sse' --include='*.py' tldw_Server_API/app/ | wc -l
19
  → the canonical helper has 19 non-test adopters app-wide and zero inside core/Chat.
```

```
$ grep -rn 'get_message_metadata\b' --include='*.py' tldw_Server_API/app/core/Chat/
chat_service.py:4188:  metadata = await asyncio.to_thread(chat_db.get_message_metadata, db_msg.get("id"))
chat_service.py:4326:  tail_metadata = {row["id"]: await asyncio.to_thread(chat_db.get_message_metadata, row["id"]) for row in tail_rows}
chat_service.py:4454:  saved_metadata = await asyncio.to_thread(chat_db.get_message_metadata, retry_user_message_id)

$ grep -rn 'get_message_metadata_map' --include='*.py' tldw_Server_API/app/ | grep -v 'def \|__all__\|"get_'
core/Chatbooks/openwebui_hydration.py:1017
core/Chatbooks/chatbook_service.py:4128
core/Chat_Macros/jobs.py:353
api/v1/endpoints/character_chat_sessions.py:1271
  → the batch API exists and has 4 adopters; core/Chat has none.
```

```
$ grep -rn '"=" \* (-len' --include='*.py' tldw_Server_API/app/ | wc -l
22
$ grep -rn '"=" \* (-len' --include='*.py' tldw_Server_API/app/core/Chat/
tldw_Server_API/app/core/Chat/chat_loop_approval.py:21:    padding = "=" * (-len(text) % 4)
  → 1 of the cluster's 22 sites is in this module.
```

## Findings

### FINDING chat-1
```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       core/Chat/chat_orchestrator.py:_get_http_status_from_exception (247-274), broken regex at
               :268  ->  r"HTTP\\s+(\\d{3})"
             core/LLM_Calls/error_utils.py:get_http_status_from_exception (126-150), broken regex at
               :145  ->  r"HTTP\\s+(\\d{3})"        [other reviewer's module; listed for the merge]
             core/Local_LLM/http_utils.py:get_http_status_from_exception (54-77), CORRECT regex at
               :72   ->  r"HTTP\s+(\d{3})"          [other reviewer's module; this is the right copy]
             core/Embeddings/Embeddings_Server/Embeddings_Create.py:_get_http_status_from_exception
               (174-188) — 4th copy, different attribute precedence, no regex branch at all
             Consumers of the broken copies inside the live chat request path:
               core/LLM_Calls/providers/base.py:101 (_raise_sanitized_provider_failure)
               core/LLM_Calls/providers/base.py:178 (error normalization)
               core/Chat/chat_orchestrator.py:551 (chat_api_call error block)
canonical:   api/v1/utils/http_errors.py:145 and core/exceptions.py:1104 both exist and are bypassed by
             all four copies.
destination: promote core/Local_LLM/http_utils.py:54 verbatim into a new cohesive module
             core/Utils/http_status.py whose single responsibility is "extract and classify an HTTP
             status from a heterogeneous exception". FORBIDDEN destinations, per the audit constraints:
             core/Utils/Utils.py and core/http_client.py.
knowledge:   "How do we recover an HTTP status from an exception that did not carry one as an
             attribute." Four copies, three precedences, two regexes, one of which is dead.
scenario:    `core/http_client.py:782-783` raises `NetworkError(f"HTTP {self.status_code}")` with NO
             `status_code=` kwarg, so `exc.status_code` is None (core/exceptions.py:553) and the message
             text is the only carrier. A 429 from a provider therefore arrives as
             `NetworkError("HTTP 429")`. Trace:
               1. `exc.response` -> None. 2. `exc.status_code` -> None. 3. `exc.status` -> None.
               4. `isinstance(exc, NetworkError)` -> True, regex at error_utils.py:145 never matches.
               5. returns None.
             At `core/LLM_Calls/providers/base.py:101-103`, `build_sanitized_chat_error(name,
             status_code=None)` then returns `ChatProviderError(provider=...)`, whose default status is
             **502** (core/exceptions.py:862-879). The client receives **502 Bad Gateway** for what was
             an upstream **429 rate limit** — losing the one signal that tells a client to back off, and
             giving it a status that most retry middleware treats as immediately retryable, which
             amplifies the rate limit instead of relieving it. The same path turns an upstream 401 into
             502 instead of `ChatAuthenticationError` "check your API key" (core/exceptions.py:809-818).
             `core/Local_LLM/http_utils.py:72`, with the correct regex, classifies all three correctly —
             so the three copies return different answers for the same exception.
             SECOND, COMPOUNDING DEFECT, specific to this module: at chat_orchestrator.py:551 the regex
             branch is unreachable even if fixed. `NetworkError` subclasses `Exception`
             (core/exceptions.py:536) and is absent from `_CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS`
             (chat_orchestrator.py:111-116 = AssertionError, AttributeError, ConnectionError,
             ImportError, KeyError, LookupError, OSError, RuntimeError, TimeoutError, TypeError,
             ValueError, UnicodeDecodeError + requests.RequestException + httpx.RequestError +
             httpx.HTTPError). So a `NetworkError` escapes `chat_api_call` entirely unmapped, and the
             `_is_network_exception` -> `ChatProviderError(504)` branch at :576 — which names
             `NetworkError` and `RetryExhaustedError` explicitly — is dead for both of them.
impact:      High. Silent, systematic misclassification of upstream rate-limit and auth failures on the
             shared provider error path, with no test that can fail. The reachability is narrower than
             the raw grep suggests (`core/http_client.py:122 _terminal_status_network_error` DOES set
             `status_code=`, so the mainline terminal-status path uses the attribute branch and is
             unaffected); the message-regex fallback covers `http_client.py:782-783`
             (`raise_for_status` when httpx is absent or `httpx.Request(...)` construction fails) and
             `core/Embeddings/connection_pool.py:204`.
cost-driver: n/a
tests:       tests/Chat/unit/test_error_handling.py, tests/Chat/unit/test_chat_orchestrator_contract.py,
             tests/LLM_Adapters/unit/test_adapter_stream_error_normalization.py — all exercise the
             attribute branch only (import-grep reachability, not measured coverage). No test in the
             repo asserts status extraction from a NetworkError message, which is exactly why a regex
             that never matches survived in two of three copies.
effort:      cheap for the fix (three characters in each of two files); moderate for the consolidation
             (one new module + 4 call-site migrations + a regression test that feeds
             `NetworkError("HTTP 429")` through each entrypoint and asserts 429, not 502).
owner-only:  no for the core/ sites; YES for api/v1/utils/http_errors.py if the canonical helper is
             moved rather than wrapped.
confidence:  confirmed (the dead regex, proven by the Python one-liner above; and the
             NetworkError-not-in-the-except-tuple defect, proven by reading both line ranges);
             probable-risk (the 429 -> 502 user-visible outcome, which depends on reaching
             `http_client.py:782-783` rather than the `_terminal_status_network_error` path).
NOTE:        The LLM_Calls reviewer is covering `error_utils.py` / `http_utils.py` from the other side.
             Merge into ONE cluster finding carrying all five sites above.
```

### FINDING chat-3
```
axis:        efficiency
class:       adoption-gap
severity:    High
sites:       core/Chat/chat_service.py:4188 (inside build_context_and_messages, loop body at 4184-4300)
               `metadata = await asyncio.to_thread(chat_db.get_message_metadata, db_msg.get("id"))`
             secondary, bounded-at-2 and therefore NOT a finding on their own but the same pattern:
               core/Chat/chat_service.py:4326, core/Chat/chat_service.py:4454
canonical:   core/DB_Management/chacha/message_store.py:get_message_metadata_map (1724-1765) — "Fetch
             metadata for multiple messages in a single query", already exported through
             ChaChaNotes_DB.py:44974 and already adopted by four callers:
             core/Chatbooks/openwebui_hydration.py:1017, core/Chatbooks/chatbook_service.py:4128,
             core/Chat_Macros/jobs.py:353, api/v1/endpoints/character_chat_sessions.py:1271.
destination: n/a — the shared helper exists; this is a pure adoption gap. Hoist one
             `get_message_metadata_map([m["id"] for m in raw_hist])` above the loop and index into it.
knowledge:   n/a (efficiency axis)
scenario:    n/a
impact:      High: this is on the hot path of every `/api/v1/chat/completions` request that carries a
             `conversation_id`, i.e. essentially all of them.
cost-driver: One `SELECT ... FROM message_metadata WHERE message_id = ?` **plus one
             `asyncio.to_thread` dispatch** per history message, awaited serially inside the loop, on
             top of the single `get_messages_for_conversation` query at chat_service.py:4180. Scales
             linearly with the history window: `DEFAULT_HISTORY_MESSAGE_LIMIT` = 20
             (chat_service.py:1057-1067) and `_MAX_HISTORY_MESSAGES` = 200
             (chat_service.py:1055), and the window is client-settable per request via
             `history_message_limit` (chat_service.py:4141-4149). So a client asking for a 200-message
             window costs 201 serialized DB round-trips and 200 thread-pool hops before the provider
             call is even issued. The batch helper replaces all of them with one `IN (...)` query.
             The history fetch itself IS bounded — `min(_MAX_HISTORY_MESSAGES, ...)` at :4149 — so
             "unbounded fetch" is NOT a finding here; the N+1 is.
tests:       tests/Chat/unit/test_chat_history_multi_image.py,
             tests/Chat/unit/test_chat_persistence_content.py,
             tests/Chat/unit/test_chat_history_and_streaming.py,
             tests/Chat_NEW/unit/test_chat_history_dedup.py,
             tests/ChaChaNotesDB/test_chacha_message_store.py:278 (covers the batch helper itself,
             including the missing-id case) — import-grep reachability, not measured coverage.
effort:      cheap. The batch helper is already tested, already adopted four times, and the loop body
             only reads `metadata.get("tool_calls")` and `metadata.get("extra")`
             (chat_service.py:4198-4206), both of which the map returns.
owner-only:  no
confidence:  confirmed
```

### FINDING chat-4
```
axis:        duplication
class:       divergent-copies
severity:    High
sites:       core/Chat/chat_orchestrator.py:_chat_sync_impl (869-1190, 322 lines)
             core/Chat/chat_orchestrator.py:achat (1379-1670, 292 lines)
             dispatcher that routes between them: core/Chat/chat_orchestrator.py:chat (1191-1378),
               streaming -> _chat_sync_impl (1243-1261), non-streaming -> _run_achat_sync (769-868)
canonical:   NONE
destination: n/a — see the action below. The right move is DELETION, not extraction.
knowledge:   The entire multimodal turn-assembly policy: slash-command parsing and injection mode,
             chat-dictionary pre/post-gen replacement, image-history handling (`tag_past` vs carry the
             last image forward, de-duplication against the prior user turn), RAG prefix construction
             from `media_content`/`selected_parts`, custom-prompt placement, and the empty-message
             placeholder. All of it is written twice.
scenario:    ONE REAL BEHAVIOURAL DIVERGENCE, not just logging. `achat` wraps the RAG prefix
             construction in a swallow-and-continue:
               achat (chat_orchestrator.py, the `rag_text_prefix` block):
                 try:  rag_text_prefix = "\n\n".join([f"{part.capitalize()}: ..." ...]).strip()
                       if rag_text_prefix: rag_text_prefix += "\n\n---\n\n"
                 except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS:  rag_text_prefix = ""
               _chat_sync_impl has the identical expression with NO try/except.
             `_CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS` (chat_orchestrator.py:84-96) includes
             `AttributeError` and `TypeError`. Input: a caller passes `selected_parts` containing a
             non-string element (an int media-part id, say), so `part.capitalize()` raises
             AttributeError; or `media_content` arrives as a list rather than a dict, so
             `media_content.get` raises AttributeError. Same input, two outcomes: the async path
             silently drops the retrieval context and answers the user **ungrounded**, with no error
             and no log; the sync path raises, is caught by the function's outer
             `except _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS`, and returns
             `ChatProviderError(status_code=500)`. An ungrounded answer that looks successful is the
             worse of the two, and it is the one the "canonical" path produces.
             SECOND DIVERGENCE (lower stakes): `_chat_sync_impl` logs a masked API key when
             `ALLOW_MASKED_KEY_LOG` is truthy; `achat` has no such block. Two answers to "may this
             process log key material".
impact:      High as a maintenance hazard, LOW as a live incident risk — and the reason is the action
             below. Reachability (stage 1): the only importer of `chat_orchestrator.chat` is
             `core/Chat/Workflows.py:34`, whose only importer is
             `tests/Chat/unit/test_chat_workflows.py:7`, and which loads
             `./App_Function_Libraries/Workflows/Workflows.json` — a path that does not exist in this
             repo. `achat` has no non-test importer outside `chat_orchestrator.py` itself. These 614
             lines are dead production code that still take commits.
cost-driver: n/a
tests:       tests/Chat/unit/test_chat_workflows.py, tests/Chat_NEW/unit/test_chat_sync_wrapper.py
             (import-grep reachability). Neither differentially tests the two bodies against each
             other, which is why the RAG try/except could appear in one and not the other.
effort:      cheap if the answer is deletion; expensive if the answer is unification (ADR-025's
             Follow-up section explicitly reserves "sync/async provider call paths are unified under a
             new provider runtime policy" for a separate ADR, so unification is a decision, not a
             refactor).
owner-only:  no
confidence:  confirmed (the twinning and the try/except divergence, both diffed line by line);
             confirmed (the reachability, by exhaustive import-grep).
```

### FINDING chat-6
```
axis:        duplication
class:       adoption-gap
severity:    Medium
sites:       Canonical, bypassed: core/LLM_Calls/sse.py:sse_data (41-43), :sse_done (46-48),
               :is_done_line (74-76), :_SSE_CONTROL_PREFIXES (19), :ensure_sse_line (51-57).
               19 non-test modules app-wide import sse.py. core/Chat imports it ZERO times.
             33 inline frame constructions in core/Chat (full list:
               2026-09-21-stage2-sse-inline-sites.txt):
               core/Chat/chat_service.py — 6: :4892, :4893 (DONE), :5150, :5157, :5159 (DONE), :5908
               core/Chat/streaming_utils.py — 27: :111, :1554, :1590 (event:), :1624, :1701
                 (pre-serialized payload_str), :1784, :1804, :1810, :1815, :1823, :1851, :1871,
                 :1877, :1883, :2044, :2050, :2123, :2128, :2152, :2163, :2225 (event:), :2244,
                 :2269 (event:), :2274 (DONE), :2372, :2480 (DONE), :2493 (DONE)
             Two re-implementations of `sse.is_done_line`:
               core/Chat/chat_service.py:3088  `if stripped.lower() in {"[done]", "data: [done]"}:`
               core/Chat/chat_service.py:6194  `if ln.strip().lower() == "data: [done]":`
             A THIRD, divergent one that is not case-insensitive:
               core/Chat/streaming_utils.py:1056 and :1690  `if payload_str == "[DONE]":`
             A same-named constant with a DIFFERENT VALUE:
               core/LLM_Calls/sse.py:19          _SSE_CONTROL_PREFIXES = ("event:", "id:", "retry:")
               core/Chat/streaming_utils.py:57   _SSE_CONTROL_PREFIXES = (":", "event:")
               core/Chat/streaming_utils.py:58   _SSE_FRAMED_CONTROL_PREFIXES = ("id:", "retry:")
canonical:   core/LLM_Calls/sse.py (already adopted 19x outside this module).
destination: n/a — adoption gap, no new module needed. `sse_data(payload)` is byte-for-byte the
             f-string at 20+ of the sites; `sse_done()` is byte-for-byte all 5 DONE sentinels.
knowledge:   The SSE wire contract that ADR-025 makes binding: "Streaming results are normalized to
             OpenAI-style SSE `data: ...` chunks and terminated with one final `[DONE]`." That contract
             currently has 33 independent implementations inside this module and a 36th in the module
             that owns it. Any future change to the frame shape — a `retry:` hint, `\r\n` framing, a
             size guard, an `id:` for resumable streams — has to be made 35 times here or it is made
             inconsistently.
scenario:    n/a (duplication axis). The nearest concrete divergence, marked as a probable risk rather
             than a defect: `sse.is_done_line` matches `data: [done]` case-insensitively while
             `streaming_utils.py:1056/:1690` require exactly `[DONE]`. A provider emitting a
             lowercase sentinel terminates cleanly through `sse.py` and, through `streaming_utils`,
             falls to the `json.loads` branch, fails to parse, is skipped by
             `except _STREAMING_NONCRITICAL_EXCEPTIONS`, and leaves `saw_done` False — so the stream
             is never marked complete by that path. I did not find a provider in-tree that emits a
             lowercase sentinel, so this is the shape of the hazard, not a demonstrated bug.
impact:      Medium. Not currently broken, but it is 33 copies of a contract an ADR declares binding,
             in the second-most-churned file in the module (streaming_utils.py, 42 commits/12mo).
             `_SSE_CONTROL_PREFIXES` having two different values under one name is the proof the drift
             has already started.
cost-driver: n/a
tests:       tests/Chat/unit/test_streaming_utils.py, test_streaming_structured_events.py,
             tests/Streaming/test_chat_completions_sse_unified_flag.py,
             tests/Streaming/test_chat_doc_stream_unified_flag.py,
             tests/LLM_Adapters/integration/test_adapters_chat_endpoint_midstream_error_all.py
             (13 files import streaming_utils; import-grep reachability, not measured coverage).
             They assert emitted bytes, so a mechanical `sse_data(...)` substitution is verifiable (they assert the emitted bytes).
effort:      cheap-to-moderate. Mechanical for the 20 `sse_data` and 5 `sse_done` sites; the judgement
             calls are the two `event:`-prefixed frames (:1590, :2269) — `sse.py` has
             `ensure_sse_control_line` but no `sse_event(name, payload)`, so add that one function —
             and reconciling the two `_SSE_CONTROL_PREFIXES`, which is a real decision because
             `streaming_utils` deliberately splits framed vs unframed control handling.
owner-only:  no
confidence:  confirmed
```

### FINDING chat-7
```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       THREE data-URI redactors:
               core/Chat/chat_service.py:_sanitize_data_uris (3015-3020) + _DATA_URI_RE (3012)
                 re.compile(r'(data:image[^,]*,)[^"\s]+', re.IGNORECASE), swallows exceptions
               core/Chat/prompt_cost_envelope.py:_sanitize_data_uris (181-200)
                 hand-rolled scanner; also stops the payload at a single quote, does NOT swallow
               api/v1/endpoints/chat.py:_sanitize_json_for_rate_limit (6129-6139)
                 re.compile(r"(\"url\"\s*:\s*\"data:image[^,]*,)[^\"\s]+") — case-SENSITIVE and only
                 matches a payload sitting under a literal `"url":` JSON key
             THREE token estimators over those redactors, with TWO rounding rules:
               core/Chat/chat_service.py:estimate_tokens_from_json (3000-3010)   len // 4   (floor)
               core/Chat/chat_service.py:_estimate_tokens_from_messages (3054-3060) len // 4 (floor)
               core/Chat/prompt_cost_envelope.py:estimate_segment_tokens (95-100)
                                                                        (len + 3) // 4   (ceiling)
             Call sites that consume them:
               api/v1/endpoints/chat.py:3958 and :4073 — rate limiting
               api/v1/endpoints/chat.py:4183 — LimitEnforcer(LLM_TOKENS_MONTH) billing reservation
               core/Chat/chat_service.py:4950 and :6503 — queue admission estimate
               core/Chat/chat_service.py:5748 and :6402 — prompt-token accounting
canonical:   NONE for redaction. NOTE: core/LLM_Calls/tokenizer_resolver.py:929 is the canonical
             tokenizer and none of these use it; that is deliberate here (these are cheap
             rate-limiter heuristics, not billing-accurate counts) and is NOT part of this finding.
destination: create core/Chat/token_estimation.py with one responsibility: "produce the Chat module's
             heuristic prompt-size estimate, including payload redaction" — exporting one redactor and
             one `estimate_tokens(text|messages)` with one rounding rule. Not Utils.py.
knowledge:   Two things that must not diverge and already have: (a) what counts as an inline binary
             payload that must not inflate a token estimate, and (b) how characters round to tokens.
scenario:    n/a (duplication axis) — but the divergence is demonstrable by inspection. A base64 image
             delivered as a bare string content (`"content": "data:image/png;base64,iVBOR..."`) or under
             any key other than `"url"` is redacted by `chat_service._sanitize_data_uris` and by
             `prompt_cost_envelope._sanitize_data_uris`, and NOT by
             `_sanitize_json_for_rate_limit`. So for one request, the rate limiter, the billing
             enforcer and the prompt-cost guardrail can each charge a different number of tokens, and
             the largest of the three is the one that counts megabytes of base64 as prompt text.
impact:      Medium. These estimates gate rate limiting (429s), a monthly billing reservation, and a
             guardrail that can BLOCK a request (prompt_cost_guardrails.py default_action "warn"|"block").
             Three answers to "how big is this prompt" behind three enforcement decisions.
             Also a small, real waste: api/v1/endpoints/chat.py:3958 applies
             `_sanitize_json_for_rate_limit` and then hands the result to `estimate_tokens_from_json`,
             which applies `_sanitize_data_uris` to the same string again — two full regex passes over
             the whole serialized request (`request_json = json.dumps(request_data.model_dump())` at
             api/v1/endpoints/chat.py:3876) where the outer pass is strictly weaker than the inner one
             and therefore contributes nothing.
cost-driver: (for the redundant-pass half) one extra full-string regex scan per estimate call, up to
             3 calls per request; scales with total serialized request size, dominated by inline
             base64 image payloads.
tests:       tests/Chat/unit/test_chat_service_token_estimates.py,
             tests/Chat/unit/test_chat_service_queue_estimate.py,
             tests/Chat/unit/test_prompt_cost_envelope.py,
             tests/Chat/unit/test_prompt_cost_guardrails.py,
             tests/Chat/unit/test_phase3_3_sanitizers.py (import-grep reachability, not measured
             coverage). Each estimator is tested against itself; nothing asserts agreement between
             them, which is the gap that let the ceiling/floor split happen.
effort:      moderate. The consolidation is small, but changing a rounding rule moves rate-limit and
             billing numbers, so it needs a Docs/Design note and a decision on which rule wins
             (the conservative `(len+3)//4` is the defensible one for anything that gates money).
owner-only:  YES for the api/v1/endpoints/chat.py sites (:3958, :4073, :4183, :6129).
confidence:  confirmed
```

### FINDING chat-8
```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       core/Chat/chat_service.py:write_mandatory_moderation_audit (3513-3544)
             core/Chat/moderation_pipeline.py:write_mandatory_moderation_audit (140-172)
             Identical keyword-only signature (audit_service, audit_context, audit_event_type, action,
             result, metadata) -> None. BOTH ARE LIVE:
               chat_service copy called at chat_service.py:3772, :6054, :6082 and imported by
                 api/v1/endpoints/chat.py:212, called at api/v1/endpoints/chat.py:3702
               moderation_pipeline copy called at moderation_pipeline.py:384 and :421
canonical:   NONE — two same-named functions in sibling files of the same package, no owner.
destination: keep exactly one, in core/Chat/moderation_pipeline.py (moderation is its stated
             responsibility); chat_service and the endpoint import it from there.
knowledge:   The mandatory-audit failure contract: swallow nothing, re-raise as
             `MandatoryAuditWriteError`, and log without leaking the moderated payload.
scenario:    The two copies disagree on the last clause. Divergent lines:
               chat_service.py:3537-3543
                 except Exception as exc:
                     logger.error("Mandatory moderation audit write failed for {} ({}) error_type={}",
                                  action, result, type(exc).__name__)
               moderation_pipeline.py:164-171
                 except Exception as exc:
                     logger.error("Mandatory moderation audit write failed for {} ({}): {}",
                                  action, result, exc, exc_info=True)
             The chat_service copy logs only the exception CLASS NAME. The moderation_pipeline copy
             interpolates `str(exc)` and adds a full traceback. Backend exception strings routinely
             embed the failing statement and bound parameter values, and the bound parameters on this
             path are the moderation audit row — which carries the flagged content metadata. So the
             same failure, on two code paths a request may take, produces either a redacted one-line
             error or the moderated payload in the log.
             This is a REGRESSION of a hardening the module claims as done:
             core/Chat/REFACTORING_PLAN.md:22 — "Sensitive Chat logs use metadata summaries instead of
             raw prompt, message, tool, or assistant content."
impact:      Medium. Security-consistency rather than a live breach: it needs an audit-write failure to
             trigger. But it is a moderation audit path, which is precisely where log content is
             sensitive, and one of the two copies already implements the correct behavior — so the fix
             is "delete the worse one", not "design something".
cost-driver: n/a
tests:       tests/Chat/unit/test_chat_service_fallback.py:41,199 imports and exercises the
             **chat_service** copy (the correct one). The moderation_pipeline copy has 1 importing test
             file in total for the whole module (tests/Chat/unit/test_chat_service_content.py) and no
             dedicated test — import-grep reachability, not measured coverage. The untested copy is the
             leaky one.
effort:      cheap. Identical signatures; delete one, re-point three call sites and one import.
owner-only:  YES — api/v1/endpoints/chat.py:212 imports the copy that would move.
confidence:  confirmed (the duplication and the logging divergence); probable-risk (that a real backend
             exception string on this path contains audit-row content — argued from how DB drivers
             format errors, not demonstrated with a captured log line).
```

### FINDING chat-11
```
axis:        duplication
class:       true-duplication
severity:    Low
sites:       core/Chat/chat_loop_approval.py:_b64url_encode (16-17), :_b64url_decode (20-22) — this
               module's single site in seed cluster C1 (22 sites app-wide; the other 21 are outside
               core/Chat). Nearest sibling: core/Prototype_Workspaces/access.py:394-396, which uses
               `("=" * ((4 - len(raw) % 4) % 4))` and `.encode("ascii")` instead of
               `"=" * (-len(text) % 4)` and `.encode("utf-8")` — arithmetically equivalent padding,
               different codec.
             core/Chat/chat_loop_approval.py:_canonical_json (25-26)
               json.dumps(payload, separators=(",",":"), sort_keys=True)
             core/Chat/prompt_cost_envelope.py:_canonical_json (177-178)
               json.dumps(value, sort_keys=True, separators=(",",":"), ensure_ascii=False, default=str)
             services/acp_runtime_policy_service.py:49-50 — a third spelling.
canonical:   NONE.
destination: create core/Utils/token_codec.py owning "encode/decode and canonicalize a compact
             signed or opaque token payload". CRITICAL CONSTRAINT, carried from the C1 cluster brief:
             the 22 sites split into two trust classes — opaque pagination cursors, and signed/crypto
             tokens (core/AuthNZ/api_key_crypto.py:119, api/v1/endpoints/notes.py:878,881, and THIS
             module's `chat_loop_approval`, which HMACs the canonical JSON via `_approval_secret`
             at :32-38). The destination must keep those classes distinct — a cursor codec that grows
             a `verify=False` default is worse than the duplication. Propose the signed-token half
             only; leave cursors alone.
knowledge:   Byte-exact JSON canonicalization under an HMAC. `chat_loop_approval._canonical_json` has
             `ensure_ascii` at its default True (non-ASCII escaped) and no `default=str` (raises on a
             non-serializable value); `prompt_cost_envelope._canonical_json` is the opposite on both.
             If these are ever unified carelessly, every previously minted approval token stops
             verifying, because the signed bytes change for any payload containing a non-ASCII
             character.
scenario:    n/a (duplication axis; no current defect — the mint and verify paths in
             chat_loop_approval.py use the same local copy, so it is self-consistent today).
impact:      Low today, High blast radius if consolidated wrong, which is exactly why it is worth
             writing down before someone does the tidy-up. One site of a 22-site cluster.
cost-driver: n/a
tests:       tests/Chat_NEW/unit/test_chat_loop_approval.py (1 importing file; import-grep
             reachability, not measured coverage). It round-trips mint->verify against the same
             implementation, so it cannot detect a canonicalization change that is applied to both
             halves at once.
effort:      cheap for this module's one site; the cluster-level consolidation belongs to whoever owns
             C1 repo-wide.
owner-only:  no for the core/Chat site; yes for api/v1/endpoints/notes.py if the cluster is taken on.
confidence:  confirmed
```

## Suggested Refactor/Actions

Ordered by value per unit of risk.

1. **`chat-1` — fix the two dead regexes now, consolidate second.** The one-character-class fix at
   `chat_orchestrator.py:268` and `error_utils.py:145` is a three-character edit in each. Ship it with
   a regression test that pushes `NetworkError("HTTP 429")` through
   `core/LLM_Calls/providers/base.py:_raise_sanitized_provider_failure` and asserts 429, not 502.
   Separately, in `chat_orchestrator.py`, either add `NetworkError`/`RetryExhaustedError` to
   `_CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS` (111-116) or delete the `_is_network_exception` branch at
   :576 — as written, the branch names two exception types that cannot reach it. Coordinate with the
   LLM_Calls reviewer; this is one cluster, not two findings. No design doc needed.
2. **`chat-3` — hoist one `get_message_metadata_map` call.** Cheapest measurable win in the module.
   No design doc needed. Small enough for a single PR with a test that counts DB calls.
3. **`chat-4` — propose deleting `Workflows.py` + `chat`/`achat`/`_chat_sync_impl`** (see stage 1
   action 1). If deletion is refused, the RAG try/except divergence must be reconciled in the same PR,
   and unification of the sync/async bodies needs its own ADR per ADR-025's Follow-up section.
4. **`chat-8` — delete one copy of `write_mandatory_moderation_audit`.** Keep the redacting one; keep
   it in `moderation_pipeline.py`. Owner-only because of the endpoint import.
5. **`chat-6` — mechanical `sse_data`/`sse_done` adoption**, plus one new `sse_event(name, payload)` in
   `core/LLM_Calls/sse.py` for the two `event:`-prefixed frames. Do NOT reconcile the two
   `_SSE_CONTROL_PREFIXES` in the same PR — that one is a behavior decision and deserves its own
   change with the streaming tests as the arbiter.
6. **`chat-7` — one `core/Chat/token_estimation.py`.** Needs a short `Docs/Design/` note because
   picking a single rounding rule changes rate-limit and billing numbers. Backlog task, not a
   drive-by.
7. **`chat-11` — record only.** Do not consolidate the approval-token codec as part of a generic
   cursor-helper effort; it is HMAC-signed and the canonicalization is load-bearing.
