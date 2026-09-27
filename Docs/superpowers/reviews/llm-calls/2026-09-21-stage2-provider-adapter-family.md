# Stage 2 — The provider adapter family

## Scope

The 16 chat adapters plus 4 embeddings adapters under `core/LLM_Calls/providers/`, treated as one
family. Builds the behavior matrix the audit brief asked for (per-provider retry, error mapping,
streaming-chunk parsing, token accounting) and names which implementation is best, so that
promoting one is cheaper than inventing an abstraction. Checked against ADR-025 (routing,
overrides, OpenAI-compatible SSE contract), ADR-026 (outbound egress), and ADR-030 (configured-local
scope) before asserting anything.

Full matrix: `2026-09-21-stage2-behavior-matrix.txt`.

## Code Paths Reviewed

- Transport: `providers/openai_adapter.py:chat (299-346)` and `:stream (348-398)`;
  `providers/groq_adapter.py:chat (128-151)` and `:stream (153-195)`;
  `providers/custom_openai_adapter.py:chat (…-317)` and `:stream (319-370)`;
  `providers/local_adapters.py (295-298, 474-506, 2083-2087)`;
  `providers/bedrock_adapter.py:600,621`; `providers/deepseek_adapter.py:279,309`;
  `providers/google_adapter.py:592,625`; `providers/huggingface_adapter.py:277,325`;
  `providers/mistral_adapter.py:238,263`; `providers/openrouter_adapter.py:201,227`;
  `providers/qwen_adapter.py:220,246`; `providers/anthropic_adapter.py:403,463`;
  `providers/cohere_adapter.py:259,334`; `providers/moonshot_adapter.py:205,222,232`;
  `providers/zai_adapter.py:129,141,192`
- Feature flags: `providers/{groq:41, openrouter:70, openai:157, bedrock:278, anthropic:81,
  custom_openai:123, openai_embeddings:24, google_embeddings:29, huggingface_embeddings:27}._use_native_http`;
  `providers/{groq:26, anthropic:61, openrouter:25}._prefer_httpx_in_tests`
- Timeout/base-url: `providers/{groq:66, mistral:118, openrouter:95, deepseek:184, qwen:145,
  openai:262, anthropic:113, google:199, huggingface:201}._resolve_timeout`;
  `providers/{groq:46,51, openrouter:75,79, mistral:100,103, openai:245, google:166, qwen:117,
  deepseek:172, bedrock:283, anthropic:96, custom_openai (via _resolve_transport_context),
  openai_embeddings:30, google_embeddings:34, huggingface_embeddings:33}._base_url/_resolve_base_url`
- Canonical config helpers: `adapter_utils.py:resolve_provider_section (108-118)` and
  `_PROVIDER_SECTION_MAP (22-50)`; `provider_config_resolution.py:resolve_provider_endpoint_url (249-263)`
- Strict transport: `adapter_registry.py:AuditedCallPolicyTransport (31-37)`,
  `:_OPENAI_STRICT_TRANSPORT (40-44)`, `:get_audited_call_policy_transport (193-204)`;
  `providers/openai_adapter.py:_strict_call_policy (278-286)`;
  `core/Chat/chat_service.py:_validate_audited_call_policy_transport (2584-2610)`;
  `core/Notes_Graph/suggestion_generation.py:build_provider_call_policy (652-…)` and `:758-768`;
  `core/Notes_Graph/suggestion_capabilities.py:159-183`
- Shared helpers already in place: `payload_utils.py:merge_extra_body (369-377)`,
  `:merge_extra_headers (379-…)`; `sse.py:is_done_line/normalize_provider_line/finalize_stream/sse_done`;
  `streaming.py:wrap_sync_stream (159-277)`; `providers/base.py:raise_if_in_band_provider_error (25-55)`,
  `:_raise_sanitized_provider_failure (85-108)`
- `core/http_client.py:create_client (2667-2742)`, `:_build_ssl_context (2399-2407)`,
  `:_httpx_limits_default (2371-2384)`

## Tests Reviewed

| Test file | What it protects | Downgrades risk? |
| --- | --- | --- |
| `tests/LLM_Calls/test_llm_providers.py` (2,348+ lines) | Per-provider request shape and response mapping; comments at :151-153 explicitly note the split between `create_session_with_retries` providers and `http_client_factory` providers | Partly — it documents the split rather than removing it |
| `tests/LLM_Calls/test_provider_adapter_runtime_boundary.py` | Adapter runtime boundary incl. mlx | Yes |
| `tests/LLM_Calls/test_provider_timeout_and_role_regressions.py` | Timeout resolution regressions for 3 providers | Partly — 3 of 9 `_resolve_timeout` copies |
| `tests/LLM_Calls/test_local_streaming_contract.py`, `test_*_strict_filter.py` (5 files) | Local adapter strict-mode field filtering, streaming contract | Yes for local adapters |
| `tests/LLM_Adapters/unit/test_adapter_error_sanitization.py` | That upstream bodies/URLs do not leak into public errors | Yes |
| `tests/LLM_Adapters/unit/test_provider_behavior_config_snapshot.py`, `test_authoritative_chat_endpoint_snapshot.py` | Snapshot of provider behavior config and endpoint output | Yes for drift detection |
| `tests/LLM_Adapters/unit/test_notes_graph_suggestion_call_policy.py` | Strict call policy for the Notes_Graph consumer | **No** — it sets `adapter.async_chat_is_native = True` at :398 on a stub, and the consumer is non-streaming, so the streaming gap in llm-calls-6 is not covered |
| `tests/LLM_Calls/test_mlx_provider.py`, `test_mlx_provider_integration.py` | MLX runtime | Yes |

Reachability by import-grep, not measured coverage.

## Validation Commands

```
$ rg -n 'http_client_factory\(timeout' --glob '*.py' tldw_Server_API/app/core/LLM_Calls/providers/ | wc -l
      20          # 10 adapters x (chat + stream), one httpx.Client per request

$ rg -n '_strict_call_policy' --glob '*.py' tldw_Server_API/app/
tldw_Server_API/app/core/LLM_Calls/providers/openai_adapter.py:278:    def _strict_call_policy(...)
tldw_Server_API/app/core/LLM_Calls/providers/openai_adapter.py:283
tldw_Server_API/app/core/LLM_Calls/providers/openai_adapter.py:314
# :314 is inside chat(). stream() (348-398) never calls it.

$ rg -n 'native HTTP disabled by configuration' tldw_Server_API/app/core/LLM_Calls/providers/*.py | wc -l
      12          # 6 adapters x (chat + stream); every "off" branch is an unconditional RuntimeError

$ rg -n 'def _resolve_timeout' tldw_Server_API/app/core/LLM_Calls/providers/*.py | wc -l
       9

$ rg -n 'resolve_provider_section' tldw_Server_API/app/core/LLM_Calls/providers/*.py
<no output>          # zero adapters use the canonical provider->section map

$ rg -n 'seen_done = False' tldw_Server_API/app/core/LLM_Calls/providers/*.py | wc -l
       8          # openai, groq, openrouter, qwen, mistral, deepseek, huggingface, bedrock
                  # (+ google:629 variant) vs moonshot:222 which uses the shared helper

$ python - <<'PY'
import ssl, time
ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
N=50; t=time.perf_counter()
for _ in range(N): ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
d=time.perf_counter()-t; print("x%d: %.3fs -> %.2f ms each"%(N,d,d/N*1000))
PY
x50: 0.365s -> 7.29 ms each      # create_client() calls this on every request
```

## Findings

### FINDING llm-calls-6 — the registry promises strict endpoint-scope and timeout enforcement that `OpenAIAdapter.stream()` does not implement

```
axis:        correctness
class:       n/a
severity:    High
sites:       Promise: core/LLM_Calls/adapter_registry.py:_OPENAI_STRICT_TRANSPORT (40-44) —
               `enforces_configured_endpoint_scope=True, enforces_maximum_timeout=True`;
               returned unconditionally by :get_audited_call_policy_transport (193-204) for any
               `openai` adapter, with no streaming/non-streaming distinction;
             Fail-closed gate that trusts it: core/Chat/chat_service.py:_validate_audited_call_policy_transport
               (2584-2610), called at :2638 (sync) and :2667 (async) BEFORE the stream branch;
             Honoured: core/LLM_Calls/providers/openai_adapter.py:chat (314-337) — checks
               `scope.matches(url)`, clamps `resolved_timeout` to
               `strict_policy.maximum_timeout_seconds`, passes `configured_endpoint=scope`;
             NOT honoured: core/LLM_Calls/providers/openai_adapter.py:stream (348-398) — no
               `_strict_call_policy` call, no `scope.matches`, no timeout clamp, and
               `http_client_factory(timeout=resolved_timeout)` at :364 cannot pass
               `configured_endpoint` at all
canonical:   core/LLM_Calls/capability_registry.py:ProviderCallPolicy (45, 96-100)
destination: n/a — either enforce in `stream()` or make
             `get_audited_call_policy_transport` streaming-aware
knowledge:   "what an audited strict call actually guarantees". The registry states it once for
             the adapter; the adapter implements it in one of its two entrypoints.
scenario:    A caller builds `ProviderCallPolicy(required_endpoint_scope=ConfiguredEndpointScope.from_url(
             "https://approved.example/v1"), maximum_timeout_seconds=10, max_transport_attempts=1)`
             — the exact shape `core/Notes_Graph/suggestion_generation.py:669` builds — and issues
             it with `request["stream"] = True` against provider `openai`.
             `_validate_audited_call_policy_transport` at chat_service.py:2596-2606 reads the
             registry, sees all three guarantees True, and returns the policy instead of raising
             `ChatConfigurationError`. Dispatch then takes the streaming branch at
             chat_service.py:2671-2676, which calls `adapter.astream` → `stream()`. `stream()`
             resolves the base URL from `request["base_url"]` / `OPENAI_BASE_URL` with **no**
             `scope.matches(url)` check and runs on the adapter's own 60 s default rather than the
             policy's 10 s. The caller's audit trail records an endpoint-scoped, timeout-bounded
             call; the wire carries an unscoped, 60 s one. The gate is fail-closed by design —
             chat_service.py:2599-2605 exists precisely to reject adapters lacking these
             guarantees — so the incorrect declaration converts a would-be rejection into a
             silent bypass.
impact:      High. It is a security-relevant guarantee (`enforces_configured_endpoint_scope`) that
             a purpose-built fail-closed gate is trusting, and the failure mode is silence rather
             than an error. Contradicts neither ADR-025 (which permits trusted allowlisted
             `base_url` overrides but says nothing about skipping a call policy) nor ADR-030
             (whose "checked synchronous requests, asynchronous requests, and synchronous response
             streams carry the same scope" is precisely what this violates for `openai`).
tests:       tests/LLM_Adapters/unit/test_notes_graph_suggestion_call_policy.py and
             core/Notes_Graph/suggestion_capabilities.py:159-183 cover the non-streaming path only.
             No test issues a strict policy with `stream=True`. Import-grep reachability, not coverage.
effort:      cheap — lift `_strict_call_policy` into `stream()` mirroring chat's :314-337, and
             switch that branch to `self.http_fetcher`/`stream_response` so `configured_endpoint`
             can be passed. Guarded by a new test asserting a scope mismatch raises on the
             streaming path.
owner-only:  no
confidence:  confirmed (the asymmetry, the unconditional declaration, and the fail-closed gate
             that trusts it); probable-risk (exploitation today — the only current strict-policy
             consumer, Notes_Graph, does not stream, so this is latent rather than live)
```

### FINDING llm-calls-7 — `LLM_ADAPTERS_NATIVE_HTTP_*=0` does not select a fallback path; it takes the provider down

```
axis:        correctness
class:       n/a
severity:    Medium
sites:       Flags with a denylist default-ON:
               providers/openai_adapter.py:_use_native_http (157-160),
               providers/groq_adapter.py:41-44, providers/openrouter_adapter.py:70-73,
               providers/bedrock_adapter.py:278-281;
             Same, plus a hard `return True` under pytest:
               providers/anthropic_adapter.py:81-87, providers/custom_openai_adapter.py:123-129;
             The "off" branches, all 12 an unconditional raise:
               openai_adapter.py:348,400; groq_adapter.py:151,195; openrouter_adapter.py:211,256;
               bedrock_adapter.py:592,613; anthropic_adapter.py:416,569;
               custom_openai_adapter.py:317,370;
             Opposite polarity under a near-identical name (genuine opt-in, default OFF, with a
             working alternative path): providers/openai_embeddings_adapter.py:24-28,
               providers/google_embeddings_adapter.py:29-32,
               providers/huggingface_embeddings_adapter.py:27-31
canonical:   NONE
destination: n/a — delete the six chat-adapter flags and their dead else-branches
knowledge:   "what turning this flag off does". For `LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI` it selects
             the legacy `chat_calls` embeddings path. For `LLM_ADAPTERS_NATIVE_HTTP_OPENAI` there
             is no legacy path left — the comment at openai_adapter.py:346 says
             "raise clear error rather than falling back".
scenario:    An operator sets `LLM_ADAPTERS_NATIVE_HTTP_OPENAI=0` to revert to the legacy
             transport after a suspected regression — a reasonable reading of the name, and the
             exact semantics the sibling `LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI` has. Every OpenAI
             chat and stream request then raises
             `RuntimeError("OpenAIAdapter native HTTP disabled by configuration")` from
             openai_adapter.py:348/:400, surfacing as a 500 for the whole provider until the
             variable is unset. Six providers behave this way; three embeddings adapters with
             nearly the same variable name behave the opposite way.
impact:      Medium. Requires an operator action to trigger, but the action is one a careful
             operator would take, the name invites it, and the blast radius is a whole provider.
tests:       tests/LLM_Adapters/unit/test_provider_behavior_config_snapshot.py snapshots provider
             behavior config; no test sets any `LLM_ADAPTERS_NATIVE_HTTP_*` to a falsy value.
             Import-grep reachability, not coverage.
effort:      cheap — delete the flag, the `if`, and the 12 raise lines; the branch has no
             alternative implementation to preserve. Add a one-line note to
             `core/LLM_Calls/README.md` that the embeddings flags are the only real toggles.
owner-only:  no
confidence:  confirmed
```

### FINDING llm-calls-8 — one SSE loop copy-pasted into nine adapters while the shared helper has one user

```
axis:        duplication
class:       adoption-gap
severity:    Medium
sites:       Canonical: core/LLM_Calls/streaming.py:iter_sse_lines_requests (73-117);
             Its only production user: providers/moonshot_adapter.py:222;
             Inline near-identical copies (`seen_done` + decode + `is_done_line` + `normalize_provider_line`
             + `finalize_stream`): providers/openai_adapter.py:368-390,
               providers/groq_adapter.py:169-189, providers/openrouter_adapter.py:230-250,
               providers/qwen_adapter.py:249-269, providers/mistral_adapter.py:267-289,
               providers/deepseek_adapter.py:316-336, providers/huggingface_adapter.py:280-300,
               providers/bedrock_adapter.py:624-646, providers/google_adapter.py:629-684 (variant
               with an extra non-SSE JSON branch);
             Further divergent shapes: providers/zai_adapter.py:141-176 (uses
               `iter_lines(decode_unicode=True)`, no `seen_done`), providers/anthropic_adapter.py:463-563,
               providers/custom_openai_adapter.py:344-364 (breaks on `[DONE]` rather than
               continuing, and uses `_provider_response_has_error` instead of the base helper),
               providers/local_adapters.py:474-506
canonical:   core/LLM_Calls/streaming.py:iter_sse_lines_requests (73-117)
destination: n/a — adopt the existing helper; extend it with a `control_filter`/error-hook
             parameter it already has (`:79`) rather than writing a new one
knowledge:   "how an OpenAI-compatible SSE stream is read, decoded, de-duplicated at `[DONE]`,
             and terminated". ADR-025 makes the single-terminal-`[DONE]` behavior part of the
             provider contract, so this is contract knowledge, not incidental code shape.
scenario:    n/a (duplication finding). Change amplification is already visible: the shared helper
             at streaming.py:100 routes in-band errors through
             `normalize_provider_stream_error` → `RuntimeError`; the nine inline copies route them
             through `self._raise_if_in_band_provider_error` (providers/base.py:110-118 →
             `raise_if_in_band_provider_error`, which builds a *sanitized typed* error and logs a
             provider failure); `custom_openai_adapter.py:352-354` uses a third predicate,
             `_provider_response_has_error`, which sets a flag and emits
             `provider_stream_error_frame` instead of raising. Three answers to "what happens when
             a provider sends an error event mid-stream", one of which produces a different
             public error class than the other two.
impact:      Medium. Every change to the SSE contract — a new provider control frame, a change to
             `[DONE]` handling, a change to in-band error classification — is a 13-site edit today,
             and the in-band-error divergence means the public error a client sees already depends
             on which provider it picked.
tests:       tests/Streaming/test_streams.py and tests/LLM_Calls/test_llm_streaming_and_security.py
             cover `iter_sse_lines_requests` directly; tests/LLM_Calls/test_local_streaming_contract.py
             and the per-provider tests in tests/LLM_Calls/test_llm_providers.py cover the inline
             copies individually. Good coverage makes the migration cheap. Import-grep
             reachability, not coverage.
effort:      cheap-to-moderate per adapter — the helper already accepts `provider`,
             `decode_unicode`, `provider_control_passthru` and `control_filter`; it needs one new
             parameter for the in-band-error callback so `_raise_if_in_band_provider_error` can be
             injected. Migrate one adapter, confirm its existing tests pass, then the rest.
             **Which copy is best**: `streaming.py:iter_sse_lines_requests` — it is the only one
             that handles bytes/str decoding with `errors="replace"`, converts transport errors
             mid-stream into an SSE error frame instead of raising through the generator, and
             honours `STREAM_PROVIDER_CONTROL_PASSTHRU`. Promote it; do not write a new base-class
             method.
owner-only:  no
confidence:  confirmed
```

### FINDING llm-calls-9 — `_resolve_timeout` and `_base_url` hardcode the provider→config-section map that `adapter_utils` already owns

```
axis:        duplication
class:       adoption-gap
severity:    Medium
sites:       Canonical: adapter_utils.py:resolve_provider_section (108-118) over
               :_PROVIDER_SECTION_MAP (22-50) — adopted by core/Chat/chat_service.py:2401,2413,
               core/AuthNZ/byok_testing.py:329, core/RAG/rag_service/hyde.py:260, and
               core/LLM_Calls/Summarization_General_Lib.py:187,252,269;
             `_resolve_timeout` copies with the section string inlined:
               providers/groq_adapter.py:66-79 ("groq_api"), mistral_adapter.py:118-132 ("mistral_api"),
               openrouter_adapter.py:95-109 ("openrouter_api"), deepseek_adapter.py:184-198 ("deepseek_api"),
               qwen_adapter.py:145-159 ("qwen_api"), openai_adapter.py:262-276 ("openai_api"),
               anthropic_adapter.py:113-133, google_adapter.py:199-…, huggingface_adapter.py:201-…;
             `_base_url` / `_resolve_base_url` copies:
               providers/groq_adapter.py:46,51; openrouter_adapter.py:75,79; mistral_adapter.py:100,103;
               openai_adapter.py:245; google_adapter.py:166; qwen_adapter.py:117; deepseek_adapter.py:172;
               bedrock_adapter.py:283; anthropic_adapter.py:96; openai_embeddings_adapter.py:30;
               google_embeddings_adapter.py:34; huggingface_embeddings_adapter.py:33
canonical:   adapter_utils.py:resolve_provider_section (108-118)
destination: n/a — adopt the existing helper and add one sibling,
             `resolve_provider_timeout(request, provider, default)`, in `adapter_utils.py`
             (already scoped to "provider identity/config resolution for adapters", 272 LOC, so it
             stays cohesive — this is NOT `Utils.py` or `http_client.py`)
knowledge:   "which `app_config` section holds a provider's operator settings, and what to do when
             the value is malformed". `_PROVIDER_SECTION_MAP` is the single source of truth and
             has already absorbed renames (`tabbyapi` → `tabby_api`, `local-llm` → `local_llm`,
             `mlx` → `mlx`), so the map changes — and when it does, 21 inlined literals do not.
scenario:    n/a (duplication finding). Change amplification has already materialised twice:
             (a) FINDING llm-calls-6 — `openai_adapter.chat` grew a timeout clamp that
             `openai_adapter.stream` did not, because the resolution logic is per-method-per-adapter
             rather than shared; (b) the error handling has drifted into five mutually
             incompatible shapes for what is the same three lines of dict lookup:
             `groq:66`/`mistral:118` use `isinstance` guards with no `try` (safest);
             `openrouter:95` and `deepseek:184` use named exception tuples with `pass`;
             `qwen:145` uses a bare `except Exception` plus two `logger.debug` calls;
             `openai:262` uses `except (AttributeError, LookupError, TypeError)`;
             `deepseek:187` additionally does `request.get(...)` with no `or {}` guard on
             `request`, relying on its outer `try` to swallow the resulting `AttributeError`.
             A future "clamp all operator timeouts to a global maximum" change must land in nine
             places; missing one silently ignores the clamp for that provider.
impact:      Medium. No live wrong answer today — each copy reads the right section for its own
             provider — but the map and the readers are two sources of truth for one fact, and the
             already-observed drift (five error shapes, one missed clamp) is the cost.
             **Which copy is best**: `groq_adapter.py:66-79` / `mistral_adapter.py:118-132` —
             the `isinstance(cfg, dict)` / `isinstance(section, dict)` guards make the surrounding
             `try/except` unnecessary, so they are the only two that do not swallow programming
             errors. Promote that body, parameterised by `resolve_provider_section(self.name)` and
             `self.capabilities()["default_timeout_seconds"]`.
tests:       tests/LLM_Calls/test_provider_timeout_and_role_regressions.py (3 providers);
             tests/LLM_Calls/test_provider_config_resolution.py;
             tests/LLM_Calls/test_llm_providers.py (per-provider request shape).
             Import-grep reachability, not coverage.
effort:      cheap — nine three-line method bodies collapse to one call each; the existing
             per-provider timeout tests guard the change.
owner-only:  no
confidence:  confirmed
```

### FINDING llm-calls-10 — ten adapters build and discard a fresh `httpx.Client` (and a fresh CA bundle) on every request

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       `with http_client_factory(timeout=...) as client:` — one per chat, one per stream:
               providers/openai_adapter.py:338,364; providers/anthropic_adapter.py:403 (+ stream);
               providers/bedrock_adapter.py:600,621; providers/deepseek_adapter.py:279,309;
               providers/google_adapter.py:592,625; providers/groq_adapter.py:141,166;
               providers/huggingface_adapter.py:277,325; providers/mistral_adapter.py:238,263;
               providers/openrouter_adapter.py:201,227; providers/qwen_adapter.py:220,246
             Factory: core/http_client.py:create_client (2667-2742) — uncached; calls
               :_build_ssl_context (2399-2407) → `ssl.create_default_context()` per call, and
               :_httpx_limits_default (2371-2384) → `max_keepalive_connections=20` on a client
               that is closed immediately
             Contrast, already correct: providers/custom_openai_adapter.py:299,338 and
               providers/openai_embeddings_adapter.py:113 use module-level `fetch`/`stream_response`,
               which reuse http_client's transport adapter
canonical:   core/http_client.py:fetch / stream_response (the shared-transport path)
destination: n/a — switch these ten to the shared `fetch`/`stream_response` path, which is also
             what FINDING llm-calls-6 needs for `configured_endpoint`. **Forbidden**: adding a
             client cache to `core/http_client.py` (6,600 LOC).
knowledge:   n/a
scenario:    n/a (efficiency finding)
cost-driver: Per LLM request: (a) one `ssl.create_default_context()` — measured at **7.29 ms**
             on this machine over 50 iterations, parsing the full system CA bundle; (b) one full
             TCP + TLS handshake, because the client and its 20-slot keepalive pool are torn down
             by the `with` block before a second request can reuse the connection. Scales
             linearly with chat-completion volume, independent of prompt size. On a chat endpoint
             the handshake RTT dominates: for a 300 ms-latency provider, connection setup is paid
             on 100% of requests instead of ~0% with a reused pool.
impact:      Medium. Adds fixed per-request cost and CPU to every commercial-provider call, and it
             is the same change that FINDING llm-calls-6's fix requires, so the two should be done
             together.
tests:       tests/LLM_Calls/test_llm_providers.py monkeypatches `http_client_factory` per
             provider, so the tests already treat it as an injection point and will keep passing
             if it becomes a module-level fetch call with the same patch target.
             Import-grep reachability, not coverage.
effort:      moderate — 20 call sites, but mechanical, and the streaming sites need
             `stream_response` semantics rather than `client.stream`.
owner-only:  no
confidence:  confirmed (the per-request construction and the 7.29 ms measurement);
             assumption (the handshake share of end-to-end latency — not measured against a live
             provider)
```

### NOT A FINDING — per-provider token accounting

The brief hypothesised that token accounting is written once per provider. It is not.
`providers/*` pass the upstream `usage` object through untouched; the only normaliser is
`routing/runtime.py:extract_router_usage (256-268)`, and `anthropic_messages.py:754-770` is a
genuine Anthropic↔OpenAI field translation (`input_tokens`/`output_tokens` vs
`prompt_tokens`/`completion_tokens`), not a duplicate. Classified **justified-divergence**;
dropped per the Axis 1 drop rule.

### NOT A FINDING — `asyncio.to_thread` / `wrap_sync_stream` boilerplate

`astream` is the same two lines in 13 adapters, but it already delegates to one shared helper
(`streaming.py:wrap_sync_stream`), so there is no divergent knowledge. The *capacity* problem it
creates is filed separately as llm-calls-4. Dropped as duplication.

## Suggested Refactor/Actions

1. **llm-calls-6 (High)** — smallest credible fix: lift the `_strict_call_policy` block from
   `openai_adapter.py:314-337` into `stream()`, and move that branch onto `self.http_fetcher` /
   `stream_response` so `configured_endpoint=scope` can be passed. Add a regression test issuing a
   strict policy with `stream=True` and a mismatched `base_url`, asserting it raises rather than
   dialing out. If enforcement in `stream()` is not wanted, the alternative is making
   `get_audited_call_policy_transport` return `None` for streaming requests so the fail-closed gate
   rejects instead of silently passing — either is acceptable, silence is not. Needs a Backlog task;
   no ADR (it implements ADR-030's stated intent rather than changing a decision).
2. **llm-calls-7 (Medium)** — delete the six `LLM_ADAPTERS_NATIVE_HTTP_*` flags and their 12 dead
   raise branches. Document in `core/LLM_Calls/README.md` that only the three
   `LLM_EMBEDDINGS_NATIVE_HTTP_*` flags are real toggles. Cheap; no design doc.
3. **llm-calls-8 (Medium)** — add an in-band-error-callback parameter to
   `streaming.py:iter_sse_lines_requests` and migrate the nine inline copies to it, one adapter per
   commit, leaning on the existing per-provider streaming tests. Settle the three in-band-error
   behaviors on one (the base-class `_raise_if_in_band_provider_error`, which is the only one that
   produces a sanitized typed error). Needs a short design note because it changes which public
   error class two providers emit: `Docs/Design/2026-09-21-llm-sse-loop-consolidation-design.md`.
4. **llm-calls-9 (Medium)** — add `resolve_provider_timeout(request, provider, fallback)` next to
   `resolve_provider_section` in `adapter_utils.py` (promoting the groq/mistral body), then collapse
   the nine `_resolve_timeout` methods and the 21 inlined section literals onto it. Cheap; guarded
   by `test_provider_timeout_and_role_regressions.py`.
5. **llm-calls-10 (Medium)** — do this together with (1): moving the ten adapters onto
   `fetch`/`stream_response` removes the per-request client *and* unblocks `configured_endpoint`.
   One design doc covering (1) + (5) + (3) is proportionate:
   `Docs/Design/2026-09-21-llm-adapter-transport-unification-design.md` plus
   `IMPLEMENTATION_PLAN_llm_adapter_transport.md` with 4 stages (strict-policy parity → shared
   fetch migration → SSE loop adoption → facade deletion).
