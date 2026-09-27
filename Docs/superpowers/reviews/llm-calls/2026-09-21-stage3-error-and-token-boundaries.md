# Stage 3 — Error → HTTP status mapping, tokenizer fallback, scalar coercion

## Scope

The three cross-module clusters whose sites live partly or wholly in `LLM_Calls`:
C8 (error → HTTP status), C6 (tiktoken fallback), C2 (scalar/env coercion). For C8 the Chat and
TTS reviewers cover their own sides; this stage reports the `LLM_Calls` sites in full and
enumerates the sibling copies only so the merged cluster is complete. For C6 the canonical helper
the brief pointed at lives in this module, so the "is it adoptable?" question is answered here.

## Code Paths Reviewed

- `core/LLM_Calls/error_utils.py:build_sanitized_chat_error (94-123)`,
  `:get_http_status_from_exception (126-150)`, `:get_http_error_text (154-189)`,
  `:_redact_sensitive_text (192-203)`, `:is_http_status_error (407-414)`,
  `:is_chunked_encoding_error (417-420)`, and the five in-module consumers at
  `:82, :236, :283, :299, :355`
- Sibling copies: `core/Local_LLM/http_utils.py:get_http_status_from_exception (54-79)`;
  `core/Chat/chat_orchestrator.py:_get_http_status_from_exception (247-273)`;
  `core/Embeddings/Embeddings_Server/Embeddings_Create.py:_get_http_status_from_exception (174-190)`
- `core/exceptions.py:NetworkError (536-555)`, `:ChatAPIError (796-808)`,
  `:ChatAuthenticationError (810-823)`, `:ChatBadRequestError (843-852)`,
  `:ChatRateLimitError (854-863)`, `:ChatProviderError (865-879)`
- Raise sites producing the message shape the dead regex targets:
  `core/http_client.py:_Response.raise_for_status (773-783)`;
  `core/Embeddings/connection_pool.py:204`
- Candidate canonical homes, checked and rejected: `api/v1/utils/http_errors.py:map_db_error_to_http (145-…)`;
  `core/exceptions.py:EmbeddingDomainError.to_http_payload (1104-1115)`
- `core/LLM_Calls/tokenizer_resolver.py:_resolve_tiktoken_encoding_cached (926-936)`,
  `:resolve_tiktoken_encoding (938-942)`, `:_resolve_tiktoken_tokenizer_name (945-951)`,
  `:_TRUE_VALUES (30)`, imports block (1-28)
- The 11 tiktoken fallback sites listed in `2026-09-21-stage2-behavior-matrix.txt`
- Coercion predicates: `core/LLM_Calls/{adapter_registry.py:103-115, cache_intents.py:88-95,
  local_cache_diagnostics.py:99-108, provider_readiness.py:15,55-60,
  extra_body_compat_catalog.py:12,93-101, llamacpp_request_extensions.py:18-28,
  routing/candidate_pool.py:47-54, tokenizer_resolver.py:30,657,
  providers/google_adapter.py:68,82, providers/mistral_adapter.py:33,
  providers/{openai:160, groq:44, openrouter:73, bedrock:281}}`;
  `core/MCP_unified/environment.py:is_truthy (4,8-13)`;
  `core/Setup/readiness_service.py:_truthy (69-72)`

## Tests Reviewed

| Test file | What it protects | Downgrades risk? |
| --- | --- | --- |
| `tests/Local_LLM/test_http_utils.py:105-107` `test_get_http_status_from_network_error_text` | Asserts `get_http_status_from_exception(NetworkError("HTTP 503 from local backend")) == 503` — against the **Local_LLM** copy | **No** for llm-calls-11 — this is the test that exists for one copy and not the other three |
| `tests/LLM_Calls/test_llm_streaming_and_security.py:388-427` | `error_utils.log_http_400_body` and `raise_chat_error_from_http` | Partly — does not reach `get_http_status_from_exception`'s `NetworkError` branch |
| `tests/LLM_Adapters/unit/test_adapter_error_sanitization.py` | That public errors carry no upstream body/URL | Yes for redaction, no for status fidelity |
| `tests/Writing/test_tokenizer_resolver_unit.py`, `test_llm_providers_tokenizer_metadata.py`, `test_writing_endpoint_integration.py` | The 3 tests covering `tokenizer_resolver.py` (1,743 LOC) | Partly — only the Writing-endpoint surface |
| `tests/LLM_Calls/test_llm_providers.py` | Per-provider error mapping end-to-end | Yes for the mapped-status path, no for the extraction-failure path |

No test file in the repo reaches `core/LLM_Calls/error_utils.py:get_http_status_from_exception`
(grep across `tldw_Server_API/tests` for the symbol returns only `tests/Local_LLM/test_http_utils.py`).
Import-grep reachability, not measured coverage.

## Validation Commands

```
$ rg -n 'HTTP\\\\s' --glob '*.py' tldw_Server_API/
tldw_Server_API/app/core/Chat/chat_orchestrator.py:268:        match = re.search(r"HTTP\\s+(\\d{3})", str(exc))
tldw_Server_API/app/core/LLM_Calls/error_utils.py:145:        match = re.search(r"HTTP\\s+(\\d{3})", str(exc))

$ rg -n 'HTTP\\s\+\(' --glob '*.py' tldw_Server_API/
tldw_Server_API/app/core/Local_LLM/http_utils.py:72:        match = re.search(r"HTTP\s+(\d{3})", str(exc))

$ python - <<'PY'
import re
b=re.compile(r"HTTP\\s+(\\d{3})"); g=re.compile(r"HTTP\s+(\d{3})")
print("broken pattern:", repr(b.pattern)); print("good   pattern:", repr(g.pattern))
print("broken.search('HTTP 429') ->", b.search("HTTP 429"))
print("good.search('HTTP 429')   ->", g.search("HTTP 429"))
PY
broken pattern: 'HTTP\\\\s+(\\\\d{3})'
good   pattern: 'HTTP\\s+(\\d{3})'
broken.search('HTTP 429') -> None
good.search('HTTP 429')   -> <re.Match object; span=(0, 8), match='HTTP 429'>

$ PYTHONPATH=. python - <<'PY'      # live A/B against the real functions
from tldw_Server_API.app.core.exceptions import NetworkError
from tldw_Server_API.app.core.LLM_Calls.error_utils import (
    get_http_status_from_exception as llm_calls_copy, build_sanitized_chat_error)
from tldw_Server_API.app.core.Local_LLM.http_utils import (
    get_http_status_from_exception as local_llm_copy)
exc = NetworkError("HTTP 429")
a, b = llm_calls_copy(exc), local_llm_copy(exc)
print("LLM_Calls/error_utils.py:126 ->", a)
print("Local_LLM/http_utils.py:54  ->", b)
e1, e2 = build_sanitized_chat_error("openai", status_code=a), build_sanitized_chat_error("openai", status_code=b)
print("actual   :", type(e1).__name__, e1.status_code)
print("should be:", type(e2).__name__, e2.status_code)
PY
LLM_Calls/error_utils.py:126 -> None
Local_LLM/http_utils.py:54  -> 429
actual   : ChatProviderError 502
should be: ChatRateLimitError 429

$ python -m pytest tldw_Server_API/tests/Local_LLM/test_http_utils.py -q --no-header
4 failed, 11 passed, 6 warnings in 1.03s
# the 4 failures are all `test_wait_for_http_ready_*` (unrelated, pre-existing);
# `test_get_http_status_from_network_error_text` is among the 11 passing.

$ rg -n 'get_http_status_from_exception' tldw_Server_API/tests -l
tldw_Server_API/tests/Local_LLM/test_http_utils.py

$ rg -n 'cl100k_base' --glob '*.py' tldw_Server_API/app/ | wc -l
      24          # 11 distinct encoding_for_model->cl100k_base fallbacks + comments/degenerate sites

$ rg -n 'httpx|aiohttp|requests\.' tldw_Server_API/app/core/LLM_Calls/tokenizer_resolver.py
<no output>       # ADR-026 direct-egress regression (fix 883b6c4dbd / PR #2604) has NOT returned

$ rg -ln 'tokenizer_resolver' --glob '*.py' tldw_Server_API/app/
tldw_Server_API/app/api/v1/endpoints/llm_providers.py
tldw_Server_API/app/api/v1/endpoints/writing.py
tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py
# 1,743 LOC, 3 non-test importers

$ PYTHONPATH=. python - <<'PY'
import tiktoken
print("tiktoken", tiktoken.__version__)
for bad in ["definitely-not-a-model", None, 123]:
    try: tiktoken.encoding_for_model(bad); print(repr(bad), "-> OK")
    except Exception as e: print(repr(bad), "->", type(e).__name__)
PY
tiktoken 0.14.0
'definitely-not-a-model' -> KeyError
None -> AttributeError
123 -> AttributeError
```

## Findings

### FINDING llm-calls-11 — dead regex from a double-escape: an upstream HTTP 429 is returned to clients as a 502

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       BROKEN — core/LLM_Calls/error_utils.py:get_http_status_from_exception (126-150),
               regex at :145 `re.search(r"HTTP\\s+(\\d{3})", str(exc))`;
             BROKEN — core/Chat/chat_orchestrator.py:_get_http_status_from_exception (247-273),
               regex at :268 (Chat reviewer's side; listed for cluster completeness);
             CORRECT — core/Local_LLM/http_utils.py:get_http_status_from_exception (54-79),
               regex at :72;
             NO REGEX BRANCH AT ALL — core/Embeddings/Embeddings_Server/Embeddings_Create.py:
               _get_http_status_from_exception (174-190);
             Consumers of the broken copy: error_utils.py:82,236,283,299,355;
               chat_calls.py:250,377; providers/base.py:101,178;
               providers/deepseek_adapter.py:364; providers/local_adapters.py:128,202;
               providers/google_adapter.py:547; providers/openai_embeddings_adapter.py:126,170;
               providers/qwen_adapter.py:293; providers/mistral_adapter.py:196;
             Producers of the matching message shape: core/http_client.py:783 (and :782)
               `raise NetworkError(f"HTTP {self.status_code}")` — status_code kwarg NOT passed,
               so the attribute is None and the regex is the only remaining extraction path;
               core/Embeddings/connection_pool.py:204, same shape
canonical:   NONE. `api/v1/utils/http_errors.py:map_db_error_to_http (145-…)` handles DB-layer
             exception hierarchies only and takes no transport exceptions;
             `core/exceptions.py:EmbeddingDomainError.to_http_payload (1104-1115)` serialises an
             already-classified domain error. **Neither serves this case** — so the fix is
             promote-one, not adopt-existing.
destination: promote `core/Local_LLM/http_utils.py:54-79` into a new
             `core/Utils/http_status_extraction.py` whose single responsibility is
             "derive an HTTP status from a transport exception of unknown provenance"
             (functions: `get_http_status_from_exception`, `is_http_status_error`,
             `get_http_error_text`, `is_chunked_encoding_error`). NOT `Utils/Utils.py`,
             NOT `http_client.py`.
knowledge:   "how to recover an HTTP status from an exception that carries it in four possible
             places — `exc.response.status_code`, `exc.response.status`, `exc.status_code`,
             `exc.status` — or only in its message text."
scenario:    Executed above. `core/http_client.py:783` raises `NetworkError("HTTP 429")` with
             `status_code=None` (the kwarg is not passed, and `core/exceptions.py:553` therefore
             leaves the attribute None). `error_utils.get_http_status_from_exception` walks
             `exc.response` (absent), `exc.status_code`/`exc.status` (both None), then reaches the
             regex at :145 — which, being double-escaped inside a raw string, searches for a
             literal backslash and never matches — and returns `None`.
             `build_sanitized_chat_error("openai", status_code=None)` then takes the
             `status_code is None` branch and returns `ChatProviderError`, whose default status is
             **502** (`core/exceptions.py:865-879`). The identical input through the Local_LLM copy
             returns 429 and yields `ChatRateLimitError`. Net effect: an upstream rate limit is
             reported to the client as a gateway error. Client-side retry/backoff keyed on 429
             never fires, the `Retry-After` semantics are lost, and rate-limit dashboards under-count
             while 5xx dashboards over-count. Identical logic applies to a 401 (would be
             `ChatAuthenticationError`) and a 400 (`ChatBadRequestError`) — all collapse to 502.
impact:      High. Silent, affects every provider that routes through `http_client`'s
             `raise_for_status`, and it is a *misclassification*, not a crash — so it produces
             wrong client behavior rather than an alert.
tests:       The proof that duplication caused this: `tests/Local_LLM/test_http_utils.py:105-107`
             is exactly the test that would catch it, written against one of four copies. No test
             in the repo reaches the `LLM_Calls` copy's `NetworkError` branch. Import-grep
             reachability, not coverage.
effort:      cheap for the one-character fix (`\\s`→`\s`, `\\d`→`\d` at error_utils.py:145 and
             chat_orchestrator.py:268); moderate for the consolidation, which needs the four copies
             reconciled — they also differ on precedence (`Embeddings_Create.py:175-180` checks
             `exc.status_code` **before** `exc.response`, and never checks `.status`, so an aiohttp
             `ClientResponseError` — which exposes `.status`, not `.status_code` — returns None
             there and the correct status everywhere else) and on failure mode (`chat_orchestrator.py:257`
             returns None on an `int()` failure where the others continue to the next candidate).
owner-only:  no (all four sites are under `core/`)
confidence:  confirmed — executed A/B above, with the live class and status code printed
```

### FINDING llm-calls-12 — the canonical tiktoken helper is in this module and is unusable for the 11 sites that need it

```
axis:        duplication
class:       true-duplication   (NOT adoption-gap — see `canonical:`)
severity:    Medium
sites:       `encoding_for_model(model)` → `get_encoding("cl100k_base")`, 11 sites in 5 exception
             dialects:
               except KeyError — api/v1/endpoints/embeddings_v5_production_enhanced.py:get_tokenizer (1445-1452);
                 core/RAG/rag_service/utils.py:26-31; core/Workflows/adapters/evaluation/eval.py:457-462;
                 core/Workflows/adapters/text/nlp.py:479-484;
               except <named tuple> — api/v1/endpoints/evaluations/evaluations_unified.py:_estimate_tokens_from_texts (501-541);
                 api/v1/endpoints/vector_stores_openai.py:_get_tokenizer (222-226);
                 core/Chunking/strategies/tokens.py:139-146;
                 core/Workflows/adapters/rag/search.py:178-186;
               bare except Exception — core/Chunking/strategies/semantic.py:414-421;
                 core/RAG/rag_service/quick_wins.py:330-334;
                 core/RAG/rag_service/web_fallback.py:63-71;
             degenerate 12th (no `encoding_for_model`, cl100k unconditionally) —
               core/VN_Assets/prompts.py:175
canonical:   core/LLM_Calls/tokenizer_resolver.py:resolve_tiktoken_encoding (938-942) over
             :_resolve_tiktoken_encoding_cached (926-936) — **exists, is importable, is
             `@lru_cache`d, has no side effects, and is still NOT adoptable by these 11 sites**,
             because its contract is the opposite of theirs: on `encoding_for_model` failure it
             raises `TokenizerUnavailable` (:934-936) rather than falling back. That is
             deliberate — it feeds the strict-token-counting machinery at
             tokenizer_resolver.py:1163,1255,1638 (`strict_token_counting_enabled()`), whose whole
             purpose is to refuse an approximate count. **This is why nobody adopted it**, and it
             changes the fix: the module needs a *second*, explicitly-lenient entry point, not an
             adoption campaign.
destination: add `resolve_tiktoken_encoding_or_default(model) -> Any` beside the strict one in
             `core/LLM_Calls/tokenizer_resolver.py` (same module, same responsibility —
             "resolve a tokenizer for a provider/model"), catching `Exception` and returning the
             `cl100k_base` encoding. Callers that must not approximate keep the strict function;
             the 11 lenient sites import the new one.
knowledge:   "what counts as a recoverable tokenizer-resolution failure, and what to fall back to."
             Five different answers today. The set matters: `tiktoken.encoding_for_model` raises
             `KeyError` for an unknown model but `AttributeError` for a non-string model
             (verified above on tiktoken 0.14.0).
scenario:    `core/Workflows/adapters/text/nlp.py:474` reads `model = config.get("model", "gpt-4")`
             from a user-authored workflow step. A step written as `model: null` in YAML yields
             `None` (the `.get` default applies only to a missing key, not an explicit null), so
             `encoding_for_model(None)` raises `AttributeError: 'NoneType' object has no attribute
             'startswith'`. The inner handler catches only `KeyError` and the outer one only
             `ImportError` (:485), so the exception escapes and fails the whole workflow step —
             where the three bare-`except Exception` siblings would have degraded to `cl100k_base`
             and returned a count. Same shape at `core/RAG/rag_service/utils.py:26` for
             `TokenCounter(model=None)` and at `core/Workflows/adapters/evaluation/eval.py:459`.
impact:      Medium. A user-triggerable crash in three of eleven sites, plus an approximate-vs-exact
             token-budget decision that is made inconsistently across RAG, Chunking, Workflows and
             Evaluations — the same document can be truncated to different lengths depending on
             which subsystem measures it.
tests:       `tests/Writing/test_tokenizer_resolver_unit.py` covers the strict helper. No test
             covers a non-string model at any of the 11 fallback sites. Import-grep reachability,
             not coverage.
effort:      cheap for the new lenient helper (6 lines beside an existing `@lru_cache`d twin);
             moderate for the 11-site migration, which crosses four owning modules. Ship the
             helper first, migrate opportunistically.
owner-only:  **yes for 3 of 11 sites** — `api/v1/endpoints/embeddings_v5_production_enhanced.py:1445`,
             `api/v1/endpoints/evaluations/evaluations_unified.py:518`,
             `api/v1/endpoints/vector_stores_openai.py:222` are under `app/api/v1/**`.
             The other 8 and the new helper are not.
confidence:  confirmed (the 11 sites, the 5 dialects, and that the canonical helper does not
             provide the needed behavior — all verified by reading and by the executed tiktoken
             probe)
```

### FINDING llm-calls-13 — `tokenizer_resolver.py` is 1,743 LOC with three non-test importers

```
axis:        encapsulation
class:       n/a
severity:    Low
sites:       core/LLM_Calls/tokenizer_resolver.py (whole file, 1,743 LOC — second-largest in the
             module); non-test importers: api/v1/endpoints/llm_providers.py,
             api/v1/endpoints/writing.py, core/Sharing/shared_workspace_chat_service.py;
             test importers: tests/Writing/{test_tokenizer_resolver_unit.py,
             test_llm_providers_tokenizer_metadata.py, test_writing_endpoint_integration.py}
canonical:   n/a
destination: n/a — this is a placement observation, not a consolidation proposal
knowledge:   n/a
scenario:    n/a
impact:      Low. Nothing is broken; the concern is that 8% of this module's code serves one
             feature (Writing's token metadata) and is tested only from that feature's test
             directory, while carrying provider-endpoint resolution, IDNA/`ipaddress` handling and
             HMAC helpers that look module-general but have no module-general users. It is the
             file the 2026-07-04 audit found making direct outbound `httpx` calls in violation of
             ADR-026 (fixed by 883b6c4dbd / PR #2604) — **re-verified clean today**: no
             `httpx`/`aiohttp`/`requests` import remains, only `urllib.parse` for parsing. Low
             importer count plus a history of egress regressions is worth noting so the next
             reviewer checks it again rather than assuming.
tests:       3 files, all under tests/Writing/. Import-grep reachability, not coverage.
effort:      n/a — recommend no action beyond keeping it on the watch list.
owner-only:  no
confidence:  confirmed (the size, the importer count, and that the direct-egress regression has
             not returned)
```

### FINDING llm-calls-14 — five truthy vocabularies inside one module, under five different function names

```
axis:        duplication
class:       true-duplication
severity:    Medium
sites:       SET A `{"1","true","yes","on"}` —
               extra_body_compat_catalog.py:_TRUE_SET (12) / _coerce_bool (93-101);
               llamacpp_request_extensions.py:_coerce_bool (18-28);
               routing/candidate_pool.py:_to_bool (47-54);
               tokenizer_resolver.py:_TRUE_VALUES (30), used at :657;
             SET B `{"1","true","yes","on","enabled"}` —
               adapter_registry.py:_parse_optional_bool (103-115) (tri-state; also accepts
                 `{"0","false","no","off","disabled"}` as False, anything else as None);
               cache_intents.py:_truthy (88-95);
               local_cache_diagnostics.py:_truthy (99-108);
               provider_readiness.py:_TRUE_VALUES (15) / _truthy (55-60);
             SET C `{"1","true","yes","on","all"}` —
               providers/google_adapter.py:68; providers/mistral_adapter.py:33;
             SET D denylist `not in {"0","false","no","off"}` (default-ON) —
               providers/openai_adapter.py:160; providers/groq_adapter.py:44;
               providers/openrouter_adapter.py:73; providers/bedrock_adapter.py:281;
               providers/google_adapter.py:82;
             SET E `{"1","true","yes","y","on"}` — imported into this module from
               core/MCP_unified/environment.py:_TRUTHY (4) / is_truthy (8-13), used by
               providers/openai_embeddings_adapter.py:28, providers/google_embeddings_adapter.py:32,
               providers/huggingface_embeddings_adapter.py:31;
             External comparator from the shared briefing:
               core/Setup/readiness_service.py:_truthy (69-72) — SET A, rejects "enabled"
canonical:   NONE that is authoritative. `core/MCP_unified/environment.py:is_truthy` is the only
             shared one and is already imported across module boundaries — but it belongs to
             MCP_unified, uses a sixth vocabulary of its own ("y"), and has no non-string handling.
destination: `core/Utils/coercion.py`, single responsibility "normalize scalar and environment
             values to bool/int/float with one documented vocabulary", exporting
             `coerce_bool(value, *, default=False)`, `coerce_optional_bool(value)` (tri-state, for
             adapter_registry's "unspecified" case), `env_bool(name, *, default)`.
             Explicitly **not** `Utils/Utils.py` (1,110 LOC junk drawer) and **not**
             `MCP_unified/environment.py` (wrong owner).
knowledge:   "which spellings of yes an operator may write in config or env." Today that is five
             answers in one module and six across the two modules that share code, under five
             function names (`_coerce_bool`, `_to_bool`, `_truthy`, `_parse_optional_bool`,
             `is_truthy`) — the naming spread is itself evidence that no owner was ever designated.
scenario:    An operator writes two settings in one config file using the same spelling:
             `providers.openai.enabled: enabled` and a llama.cpp runtime flag as `enabled`.
             The first is parsed by `adapter_registry._parse_optional_bool` (SET B) → True, and
             the provider is registered. The second is parsed by
             `llamacpp_request_extensions._coerce_bool` (SET A) → False, and the extension is
             silently off. No warning is emitted at either site. Having seen the first setting
             take effect, the operator has no reason to suspect the second spelling is rejected.
             The same split hits `"all"` (accepted only at google_adapter.py:68 and
             mistral_adapter.py:33) and `"y"` (accepted only via the three embeddings adapters'
             imported `is_truthy`). Non-string inputs diverge too:
             `llamacpp_request_extensions._coerce_bool (28)` ends in `return bool(value)`, so a
             non-empty list is True, where `extra_body_compat_catalog._coerce_bool (101)` and
             `routing/candidate_pool._to_bool (54)` return False for the same input.
impact:      Medium. Each individual miss is a silently-ignored setting rather than a crash, which
             is what makes it expensive — it surfaces as "the flag doesn't work" support load, not
             as an error. This module is 13 of the repo-wide ~91 bool coercers, and it is the one
             with the widest vocabulary spread.
tests:       `tests/LLM_Adapters/unit/test_extra_body_compat_catalog.py`,
             `tests/LLM_Calls/test_llamacpp_request_extensions.py`, and
             `tests/LLM_Calls/test_adapter_registry_defaults.py` each cover their own predicate;
             none compares them. Import-grep reachability, not coverage.
effort:      moderate — the module is a proportionate first adopter (13 sites, all internal, all
             covered), but choosing the one vocabulary is a decision: SET B's `"enabled"` and
             SET C's `"all"` are load-bearing at their sites today, so the consolidated
             `coerce_bool` must either accept the union or the migration must convert those
             configs. That decision needs a design note.
owner-only:  no
confidence:  confirmed (the five sets and five names); probable-risk (the operator scenario — no
             bug report was located, the divergence is inferred from the code)
```

### FINDING llm-calls-15 — `is_http_status_error` is the most complete copy of four and is bypassed by three TTS adapters

```
axis:        duplication
class:       adoption-gap
severity:    Low
sites:       Most complete, public, importable —
               core/LLM_Calls/error_utils.py:is_http_status_error (407-414): recognises both
               `httpx.HTTPStatusError` and `requests.HTTPError`;
             Bypassing copies, httpx-only —
               core/TTS/adapters/openai_adapter.py:_is_http_status_error (52-55);
               core/TTS/adapters/elevenlabs_adapter.py:_is_http_status_error (54-57);
               core/TTS/adapters/qwen3_runtime_remote.py:_is_http_status_error (130-133)
canonical:   core/LLM_Calls/error_utils.py:is_http_status_error (407-414)
destination: same as llm-calls-11 — move it with the rest of the cluster into
             `core/Utils/http_status_extraction.py`, since three TTS adapters importing from
             `core/LLM_Calls/` would itself be a questionable dependency
knowledge:   "which exception classes mean the server answered with an error status" — a set that
             grows whenever a transport library is added or swapped.
scenario:    n/a (duplication finding). Divergence cost: if a TTS adapter is ever moved onto a
             `requests`-based transport — or receives a `requests.HTTPError` re-raised from a
             shared helper — its `_is_http_status_error` returns False and the error is classified
             as generic instead of status-bearing. The LLM_Calls copy already handles that case.
impact:      Low. The three TTS sites are httpx-only today so the gap is latent. Reported for
             cluster completeness; the TTS reviewer owns those three files.
tests:       `tests/LLM_Calls/test_llm_streaming_and_security.py` reaches the LLM_Calls copy
             indirectly via `raise_chat_error_from_http`. Import-grep reachability, not coverage.
effort:      cheap once llm-calls-11's destination module exists; do them together.
owner-only:  no
confidence:  confirmed
```

## Suggested Refactor/Actions

1. **llm-calls-11 (High) — do the one-character fix immediately, separately from the
   consolidation.** `error_utils.py:145` and `chat_orchestrator.py:268`: `r"HTTP\\s+(\\d{3})"` →
   `r"HTTP\s+(\d{3})"`. Add a test mirroring `tests/Local_LLM/test_http_utils.py:105-107` against
   the `LLM_Calls` copy. Then, as a follow-up with a design doc
   (`Docs/Design/2026-09-21-http-status-extraction-consolidation-design.md` + a Backlog task +
   `IMPLEMENTATION_PLAN_http_status_extraction.md`), promote the Local_LLM body into
   `core/Utils/http_status_extraction.py` and delete the other three, reconciling the precedence
   and `.status` divergences noted above. Bandit/`security-required` is unaffected. Base branch:
   `dev` per `CONTRIBUTING.md:86,121` (note `origin/HEAD` resolves to `main`; the fix is
   branch-agnostic).
2. **llm-calls-12 (Medium)** — add `resolve_tiktoken_encoding_or_default()` to
   `tokenizer_resolver.py` beside the strict twin, then migrate the 8 non-owner-only sites. Label
   the 3 `app/api/v1/**` sites owner-only in the task. Fix
   `core/Workflows/adapters/text/nlp.py:481` and `core/RAG/rag_service/utils.py:26` first — they
   are the crash-on-`None` cases. No ADR needed; ADR-025/026/030 are silent on tokenizers.
3. **llm-calls-14 (Medium)** — create `core/Utils/coercion.py` with one documented vocabulary and
   adopt it in the 13 `LLM_Calls` sites as the pilot. Needs a design note to settle whether
   `"enabled"`, `"all"` and `"y"` join the canonical set or are migrated out of config, because
   SET B and SET C sites depend on them today. Do **not** add these functions to `Utils/Utils.py`.
4. **llm-calls-15 (Low)** and **llm-calls-13 (Low)** — fold llm-calls-15 into (1)'s destination
   module. For llm-calls-13, no action; record `tokenizer_resolver.py` on the egress watch list so
   the next audit re-checks it rather than assuming the 883b6c4dbd fix holds.
5. **Cross-cutting, cheap, high leverage** — the repo already has the right enforcement pattern in
   `tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py` (an AST-based import
   ratchet). A sibling lint test asserting that `re.search`/`re.compile` literals inside raw
   strings contain no `\\\\s`/`\\\\d` double-escape would have caught llm-calls-11 at review time
   and costs one test file. Propose as a Backlog task; it fits the six-gate contract in
   `Docs/Development/CI_REQUIRED_GATES.md` under `backend-required` with no new gate.
