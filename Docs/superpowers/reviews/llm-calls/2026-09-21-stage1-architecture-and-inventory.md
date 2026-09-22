# Stage 1 — Architecture survey, inventory, and transport layering

## Scope

Establish the shape of `tldw_Server_API/app/core/LLM_Calls/`: file inventory, 12-month churn,
test reachability, layering position, and the three competing HTTP abstraction layers that the
shared briefing flagged as over-layering rather than duplication. No provider-specific behavior
is analysed here — that is Stage 2.

## Code Paths Reviewed

- `core/LLM_Calls/http_helpers.py:_RetrySession (55-106)` and `:create_session_with_retries (109-121)`
- `core/LLM_Calls/chat_calls.py:_SessionShim (62-106)` and `:create_session_with_retries (108-140)`
- `core/http_client.py:create_client (2667-2742)`, `:fetch (4627-4652)`, `:afetch (4053-4074)`,
  `:_Response.raise_for_status (773-783)`
- `core/LLM_Calls/streaming.py:iter_sse_lines_requests (73-117)`, `:aiter_sse_lines_httpx (120-156)`,
  `:wrap_sync_stream (159-277)`, `:aiter_normalized_sse (280-314)`
- `core/LLM_Calls/providers/base.py:ChatProvider (58-193)` — `achat (144-150)`, `astream (152-154)`,
  `async_chat_is_native (62)`
- `core/LLM_Calls/adapter_registry.py:AuditedCallPolicyTransport (31-37)`,
  `:_OPENAI_STRICT_TRANSPORT (40-44)`, `:ChatProviderRegistry.DEFAULT_ADAPTERS (51-...)`,
  `:get_audited_call_policy_transport (193-204)`
- `core/Chat/chat_service.py:_validate_audited_call_policy_transport (2584-2610)`,
  `:perform_chat_api_call (2624-2650)`, async dispatch (2660-2721)
- `core/Chat/bounded_daemon.py:SYNC_ADAPTER_CALL_POOL (357-359)`
- `core/LLM_Calls/deprecation.py (1-7)`, `core/LLM_Calls/README.md`

## Tests Reviewed

Located by import-grep (`grep -rl "core\.LLM_Calls" tldw_Server_API/tests`), never by path.
Full list in `2026-09-21-stage1-test-inventory.txt`. These counts are **import-grep reachability,
not measured coverage** — the suite was not executed beyond the focused runs below.

| Test area | Files | What it protects | Downgrades risk? |
| --- | --- | --- | --- |
| `tests/LLM_Adapters/` | 55 | Adapter error sanitization, provider behavior snapshots, unsafe-POST no-retry, call-policy | Yes for adapter contract shape |
| `tests/LLM_Calls/` | 46 | Per-provider request/response/streaming contracts, local adapters, mlx | Yes for provider payloads |
| `tests/Chat/` | 25 | Chat-service→adapter dispatch, credential policy | Yes for the dispatch seam |
| `tests/RAG_NEW/`, `tests/Media/` | 16 | Downstream consumers | Indirect only |
| remainder | 50 | Writing, Streaming, Evaluations, Audio, Admin, AuthNZ_Unit, Notes_Graph, … | Indirect only |

Notable per-file reachability gaps (by name-grep against the 192 files):

- `error_utils.py` — referenced by exactly **one** test file (`tests/LLM_Calls/test_llm_streaming_and_security.py:388`),
  and only for `log_http_400_body` / `raise_chat_error_from_http`. It is the error-classification hub
  for all 16 adapters.
- `http_helpers.py` — one test file (`tests/LLM_Adapters/unit/test_provider_unsafe_post_no_retry.py`).
- `local_adapters.py` (2,394 LOC, largest file) — 0 direct module imports but **23** test files
  reference it via the registry and adapter names. Well covered; not a gap.
- `mlx_provider.py` — 4 test files. Covered.
- `tests/LLM_Calls/property/test_anthropic_messages_properties.py` and `test_structured_output_properties.py`
  fail to collect locally on missing `hypothesis`; `hypothesis` IS declared at `pyproject.toml:61`,
  so this is a local-env artefact and **not** a finding.

## Validation Commands

```
$ find tldw_Server_API/app/core/LLM_Calls -name '*.py' | xargs wc -l | sort -rn | head -1
   21841 total          # 64 files; largest providers/local_adapters.py 2394, tokenizer_resolver.py 1743

$ git log --since='12 months ago' --name-only --pretty=format: -- tldw_Server_API/app/core/LLM_Calls \
    | grep '\.py$' | sort | uniq -c | sort -rn | head -5
  30 .../providers/openai_adapter.py
  30 .../providers/google_adapter.py
  29 .../LLM_API_Calls.py            # file no longer exists — prior decomposition
  27 .../providers/qwen_adapter.py
  25 .../Summarization_General_Lib.py

$ grep -rl "core\.LLM_Calls" tldw_Server_API/tests | wc -l
     192

$ rg -n 'import sqlite3|DB_Management|Databases' --glob '*.py' tldw_Server_API/app/core/LLM_Calls
<no output>

$ rg -n 'from tldw_Server_API.app.api' --glob '*.py' tldw_Server_API/app/core/LLM_Calls
<no output>

$ rg -n 'aiter_normalized_sse' --glob '*.py' tldw_Server_API/
tldw_Server_API/app/core/LLM_Calls/streaming.py:280:async def aiter_normalized_sse(

$ rg -n 'async_chat_is_native' --glob '*.py' tldw_Server_API/app/core/LLM_Calls/providers/
tldw_Server_API/app/core/LLM_Calls/providers/base.py:62:    async_chat_is_native: bool = False
tldw_Server_API/app/core/LLM_Calls/providers/base.py:147:        Native async implementations must also set ``async_chat_is_native`` to
# i.e. no concrete adapter sets it True

$ rg -n '\.achat\(' --glob '*.py' tldw_Server_API/app/
tldw_Server_API/app/core/Chat/chat_service.py:2686
tldw_Server_API/app/core/Evaluations/wordbench_runner.py:124
tldw_Server_API/app/core/Streaming/speech_chat_service.py:774
tldw_Server_API/app/core/Workflows/adapters/llm/translate.py:59

$ rg -c 'LLM_Calls' pyproject.toml
26                      # 24 source files on the BLE001 per-file grandfather list (:1095-1118)

$ python -m pytest tldw_Server_API/tests/LLM_Calls/test_llm_streaming_and_security.py \
    tldw_Server_API/tests/LLM_Adapters/unit/test_provider_unsafe_post_no_retry.py -q --no-header
24 passed, 8 warnings in 1.60s

$ python -m pytest tldw_Server_API/tests/LLM_Calls -q --no-header --collect-only | tail -1
542 tests collected, 2 errors in 1.31s     # both errors = missing local `hypothesis`
```

## Findings

### FINDING llm-calls-1 — `create_session_with_retries` returns a different class under pytest than in production

```
axis:        correctness
class:       n/a
severity:    High
sites:       core/LLM_Calls/chat_calls.py:create_session_with_retries (108-140) — the
             `if _os.getenv("PYTEST_CURRENT_TEST")` branch at :125;
             core/LLM_Calls/chat_calls.py:_SessionShim (62-106) — production object;
             core/LLM_Calls/http_helpers.py:_RetrySession (55-106) — test object;
             consumers: providers/cohere_adapter.py:259, providers/moonshot_adapter.py:205,
             providers/moonshot_adapter.py:232, providers/zai_adapter.py:129,
             providers/zai_adapter.py:192, chat_calls.py:227, chat_calls.py:320
canonical:   NONE
destination: n/a — the fix is deletion of the branch, not a new module
knowledge:   "which HTTP object a provider POST travels through". Two objects, chosen by an
             environment variable that only exists in the test harness.
scenario:    Under pytest, `create_session_with_retries()` returns `_RetrySession`, whose
             non-streaming POST is `_hc_fetch(..., client=_hc_create_client())` — a dedicated
             client with its own pool, TLS context and `event_hooks`. In production it returns
             `_SessionShim`, whose non-streaming POST is `fetch(...)` with **no** `client=`
             argument, taking http_client's default transport-adapter path. Every one of the
             seven call sites above is therefore exercised in CI against an object that is not
             the object that runs in production. A regression in `_SessionShim.post` — a changed
             default timeout, a missing header, a pooling change — is invisible to the entire
             suite. `tests/LLM_Adapters/unit/test_provider_unsafe_post_no_retry.py:97-98` has to
             explicitly monkeypatch the name back to `http_helpers.create_session_with_retries`
             to test anything, which is the suite conceding the point.
impact:      High. This is not a coverage gap in a corner; it is the Cohere, Moonshot and Zai
             chat path plus the legacy OpenAI embeddings path, and it makes green CI a weaker
             signal than it appears for those providers.
tests:       tests/LLM_Adapters/unit/test_provider_unsafe_post_no_retry.py (the only file that
             reaches `_RetrySession` deliberately); 21 further test files monkeypatch
             `chat_calls.create_session_with_retries` wholesale and so reach neither object.
             Import-grep reachability, not coverage.
effort:      moderate — deleting the pytest branch will break tests that rely on the legacy
             object's `iter_lines` shape; those need porting to `_SessionShim` first.
owner-only:  no
confidence:  confirmed (the branch and the two divergent objects); confirmed (that no test
             exercises `_SessionShim.post`, since `PYTEST_CURRENT_TEST` is set for every test)
```

### FINDING llm-calls-2 — `create_session_with_retries` takes four retry parameters and discards all four

```
axis:        encapsulation
class:       n/a
severity:    Medium
sites:       core/LLM_Calls/http_helpers.py:create_session_with_retries (109-121) and
             :_RetrySession.__init__ (56-68) — `_ = total, backoff_factor, status_forcelist,
             allowed_methods` then `RetryPolicy(attempts=1)`;
             core/LLM_Calls/chat_calls.py:create_session_with_retries (108-140) and
             :_SessionShim.__init__ (63-77) — same discard;
             callers passing values that are ignored: providers/cohere_adapter.py:259,
             providers/moonshot_adapter.py:205,232, providers/zai_adapter.py:129,192,
             chat_calls.py:227, chat_calls.py:320
canonical:   core/http_client.py:RetryPolicy is the real retry surface
destination: n/a — rename to `create_single_attempt_session()` and drop the four parameters
knowledge:   "does a provider POST retry, and on which statuses". The function name and
             signature say yes-and-configurably; the body says never.
scenario:    A caller writes `create_session_with_retries(total=5, status_forcelist=[429, 502])`
             reasonably expecting five attempts with backoff on 429. It makes exactly one
             attempt and no backoff. The docstring at http_helpers.py:5 explains why
             (no idempotency contract for provider POSTs) — the decision is correct; only its
             API is a lie. Six call sites currently pass `total=1`, which happens to match, so
             the lie is dormant rather than active.
impact:      Medium. No live defect today because every caller happens to pass `total=1`, but
             the signature actively invites a future caller to configure retries that silently
             do not happen — the exact class of bug the comment was written to prevent.
tests:       tests/LLM_Adapters/unit/test_provider_unsafe_post_no_retry.py asserts the
             single-attempt behavior, so the fix is guarded. Import-grep reachability.
effort:      cheap — mechanical rename plus six call-site edits; behavior unchanged.
owner-only:  no
confidence:  confirmed
```

### FINDING llm-calls-3 — three abstraction layers over one HTTP client, disagreeing on egress enforcement

```
axis:        encapsulation
class:       n/a  (over-layering, per the briefing's do-not-seed list — NOT duplication)
severity:    Medium
sites:       Layer 1 core/LLM_Calls/http_helpers.py:_RetrySession (55-106) — POST-only facade,
               `.post(stream=)`, `.close()`, no `get`, no context manager, single attempt,
               owns its own `create_client()`;
             Layer 2 core/LLM_Calls/chat_calls.py:_SessionShim (62-106) — POST-only facade,
               non-stream via `fetch()` (no client), stream by constructing a Layer-1 object
               per call at :87, `.close()` closes only the delegate;
             Layer 3 core/http_client.py:fetch (4627-4652) / create_client (2667-2742) /
               astream_sse — the full surface: `configured_endpoint`, `cert_pinning`,
               `max_response_bytes`, `sensitive_observability`, `RetryPolicy`, `transport`;
             Layer-3 direct users in this module: providers/custom_openai_adapter.py:299,338;
               providers/local_adapters.py:295-298; providers/openai_adapter.py:322;
               providers/openai_embeddings_adapter.py:113; streaming.py:303
canonical:   core/http_client.py (Layer 3) is the designated boundary per ADR-026
destination: n/a — the fix is removing layers, not adding one. **Forbidden**: growing
             `core/http_client.py` (6,600 LOC, itself a cohesion problem — `fetch` at :4627 is
             a dual-mode function with two unrelated signatures discriminated by whether
             `method` is in kwargs).
knowledge:   "what policy an outbound LLM request is subject to". Layers 1 and 2 cannot express
             `configured_endpoint`, `cert_pinning` or `max_response_bytes` at all — the
             parameters do not exist on `.post()`. A caller choosing Layer 1 or 2 silently opts
             out of ADR-030's scoped-endpoint machinery with no diagnostic.
scenario:    A maintainer adding a new commercial provider must pick one of three layers with no
             stated rule. Picking Layer 1 or 2 (as Cohere, Moonshot and Zai did) yields an
             adapter that cannot ever carry a configured-endpoint scope, so it can never be
             admitted to the ADR-030 trusted-origin path without being rewritten onto Layer 3.
impact:      Medium. No current defect — the three Layer-1/2 providers are commercial and do not
             need configured-local scope — but the layering is the reason the strict-transport
             registry at adapter_registry.py:193-204 can only ever return a guarantee for
             `openai`, and it is what FINDING llm-calls-1 rides on.
tests:       tests/LLM_Adapters/unit/test_provider_unsafe_post_no_retry.py;
             tests/LLM_Calls/test_provider_adapter_runtime_boundary.py;
             tests/LLM_Calls/test_provider_timeout_and_role_regressions.py. Import-grep
             reachability, not coverage.
effort:      expensive — needs a design doc; three providers must move from Layer 1/2 to
             Layer 3 and the streaming `iter_lines` contract differs between them.
owner-only:  no (all paths are under `core/`)
confidence:  confirmed (three layers exist and differ in expressible policy);
             probable-risk (the ADR-030 admission consequence — no provider is blocked today)
```

### FINDING llm-calls-4 — 13 adapters override `achat` with an unbounded thread hop that two of four callers bypass

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       Base contract: providers/base.py:async_chat_is_native (62) and
               :achat (144-150) — "The default raises instead of silently running sync work inline";
             Overrides that do exactly that, none of which set the flag:
               providers/openai_adapter.py:403, anthropic_adapter.py:572, google_adapter.py:690,
               groq_adapter.py:198, openrouter_adapter.py:259, mistral_adapter.py:295,
               qwen_adapter.py:320, deepseek_adapter.py:393, huggingface_adapter.py:306,
               cohere_adapter.py:513, bedrock_adapter.py:652, custom_openai_adapter.py:373,
               local_adapters.py:2083 — all `return await asyncio.to_thread(self.chat, request, timeout=timeout)`;
             Gated callers (never reach `achat`): core/Chat/chat_service.py:2683-2689,
               core/Streaming/speech_chat_service.py:770-786;
             Ungated callers (always reach it): core/Evaluations/wordbench_runner.py:124,
               core/Workflows/adapters/llm/translate.py:59
canonical:   core/Chat/bounded_daemon.py:SYNC_ADAPTER_CALL_POOL (357-359), driven by
             `CHAT_SYNC_ADAPTER_MAX_WORKERS`
destination: n/a
knowledge:   "how many blocking provider calls may be in flight at once". Chat and speech-chat
             answer via `SYNC_ADAPTER_CALL_POOL`; the 13 `achat` overrides answer via asyncio's
             default executor, which is not that pool and is not configurable by that env var.
scenario:    `core/Evaluations/wordbench_runner.py:124` awaits `adapter.achat(payload)` per
             evaluation item with no capacity gate. Each call parks one default-executor thread
             for the whole provider round trip — 60 s for `openai`/`groq` defaults, 90 s for
             `openrouter`/`qwen`/`deepseek`, 120 s for `custom_openai`. A WordBench batch wider
             than the default executor (`min(32, cpu+4)` threads) saturates it, and every other
             `asyncio.to_thread` user in the process queues behind provider latency. The
             operator's `CHAT_SYNC_ADAPTER_MAX_WORKERS` setting has no effect on this path.
cost-driver: one blocked default-executor thread per concurrent `achat`, held for the full
             provider round trip; scales with evaluation/workflow batch width, not with request
             size. Ceiling is the default executor's `min(32, cpu_count+4)`.
impact:      Medium. Degrades to queueing rather than failing, and only two call sites are
             ungated — but those two are exactly the batch-shaped ones.
tests:       tests/Chat/unit/test_chat_service_fallback.py exercises both sides of the
             `async_chat_is_native` branch with stub adapters that set the flag True;
             tests/Audio/test_speech_chat_service.py:1120,1442 assert it is False for real
             adapters. No test covers `wordbench_runner.py:124` or `translate.py:59` under
             concurrency. Import-grep reachability, not coverage.
effort:      cheap — route the two ungated callers through `await_bounded_sync_call(...,
             pool=SYNC_ADAPTER_CALL_POOL)` exactly as chat_service.py:2692-2696 does, and
             delete the 13 dead `achat` overrides (or keep one and lift it to `base.py`).
owner-only:  no
confidence:  confirmed (the flag is never set, the overrides exist, the two callers are ungated);
             probable-risk (the saturation scenario — not reproduced under load here)
```

### FINDING llm-calls-5 — `streaming.py:aiter_normalized_sse` is dead code and `aiter_sse_lines_httpx` is test-only

```
axis:        duplication
class:       true-duplication
severity:    Low
sites:       Dead: core/LLM_Calls/streaming.py:aiter_normalized_sse (280-314) — 0 callers
               repo-wide, 0 tests;
             Test-only: core/LLM_Calls/streaming.py:aiter_sse_lines_httpx (120-156) — reached
               only from tests/LLM_Calls/test_llm_streaming_and_security.py:95,103,120,148;
             Live shared helper with one user: :iter_sse_lines_requests (73-117) ←
               providers/moonshot_adapter.py:222 only
canonical:   itself
destination: n/a — delete `aiter_normalized_sse`; see FINDING llm-calls-6 for adopting
             `iter_sse_lines_requests`
knowledge:   `aiter_normalized_sse` is the only helper in the module that pairs SSE
             normalisation with `astream_sse`'s egress enforcement and retry policy, and its
             docstring says so ("Enforces egress policy and retries per PRD defaults"). It has
             no users, so that knowledge is unexercised and will rot.
scenario:    n/a (not a correctness finding)
impact:      Low on its own. Notable because the module's most policy-correct streaming helper
             is the one nobody adopted, while nine adapters hand-rolled the loop without it.
tests:       none for `aiter_normalized_sse`; tests/Streaming/test_streams.py and
             tests/LLM_Calls/test_llm_streaming_and_security.py for the other two.
effort:      cheap — delete 35 lines, or wire `astream` onto it (that is Stage 2's recommendation).
owner-only:  no
confidence:  confirmed
```

## Suggested Refactor/Actions

1. **llm-calls-1 (High)** — delete the `PYTEST_CURRENT_TEST` branch at `chat_calls.py:125-134` so
   tests and production share one object. Needs a design note (`Docs/Design/2026-09-21-llm-session-shim-unification-design.md`)
   because the two objects' streaming `iter_lines` contracts differ and 21 tests monkeypatch the
   factory. Stage it: (a) make `_SessionShim.post(stream=True)` return the same shape as
   `_RetrySession`, (b) flip the branch off behind a temporary env guard, (c) delete
   `http_helpers.py` entirely. Small enough for a Backlog task + implementation plan; no ADR needed
   (no decision reversed — ADR-025 does not name the session facade).
2. **llm-calls-2 (Medium)** — rename to `create_single_attempt_session()` and drop the four discarded
   parameters. Pure mechanical; no design doc. Do it with (1).
3. **llm-calls-3 (Medium)** — record the layer choice as a rule rather than removing layers
   immediately: add one paragraph to `core/LLM_Calls/README.md` stating that new adapters use
   `http_client.fetch`/`stream_response` directly and that the two session facades are frozen for
   the three legacy providers. Then the removal is (1)'s stage (c). Do **not** add a fourth wrapper
   and do **not** grow `http_client.py`.
4. **llm-calls-4 (Medium)** — two-line change at `wordbench_runner.py:124` and
   `translate.py:59` to use `await_bounded_sync_call(..., pool=SYNC_ADAPTER_CALL_POOL)`; then delete
   the 13 `achat` overrides so `base.py:144`'s documented contract is true again. Cheap, no design doc.
5. **llm-calls-5 (Low)** — delete `aiter_normalized_sse`. Opportunistic.
