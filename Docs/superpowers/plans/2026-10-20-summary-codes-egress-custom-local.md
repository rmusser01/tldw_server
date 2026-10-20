# Summary Error Codes for Egress-Blocked, Unconfigured Custom and Local Providers Implementation Plan (TASK-13865)

Review folded in (2026-10-20). The independent review and dry run found 0 Blocker, 1 Important and 6 Minor, and all of them are folded in:
- I1: the measured list of in-process callers whose behaviour changes with AC2, plus a `tests/Setup` test pinning the first-chat category.
- M1: the setting the chat probe needs.
- M2: the Evaluations change is listed with the owner decisions, and `eval_runner` is in the reader audit.
- M3: the full egress reason mapping.
- M4: the corrected retry cost.
- M5: a transport `NetworkError` (a local server that is down) is now typed as `provider_failure` in this PR, test-first.
- M6: six sweep cases need DNS; a test-hygiene follow-up is filed.

Every RED, GREEN and sweep count below was re-measured after the fold, on origin/dev `e521605198`, with every non-loopback connect and DNS lookup blocked.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Summary failures that read as the wrong code, or as no code at all, get the right `SummaryProviderError` code:
- a provider endpoint the outbound (egress) policy blocks;
- a local server that is down (a transport `NetworkError`, which kobold lets through);
- a custom OpenAI-compatible provider with no base URL;
- a local provider's 401, 403 or 429 reply.

No new summary code is added.

**Architecture:**
- AC1, egress. `EgressPolicyError` joins `_SUMMARY_ADAPTER_EXCEPTIONS` (`core/LLM_Calls/Summarization_General_Lib.py`).
  - `_summary_error_code` maps it the way `chat_service._map_provider_egress_error` already does for chat: `dns_unresolved` becomes `provider_failure` (retried), and every other reason becomes the existing `configuration` code (not retried).
  - A transport `NetworkError` joins it. It falls through to `provider_failure`, so the audio stage retries it and, on the last attempt, stores the transcript with `analysis_error` (review M5).
  - With `raise_on_error=False` the two handlers re-raise either exception itself, so those callers see what they see today.
- AC2, custom endpoints. The two bare `RuntimeError`s for a missing base URL in `custom_openai_adapter.py` (lines 158 and 211) become `ChatConfigurationError`. That is the class the same adapter already raises at line 195 for the same condition when a credential handle is bound, so the summary maps it to `configuration` with no summary change.
- AC3, local 4xx. `_raise_sanitized_local_provider_error` (`local_adapters.py`) keeps the upstream status on its `ChatBadRequestError` as `upstream_status_code`, as `ChatProvider._raise_sanitized_provider_failure` already does.
  - `_summary_error_code` reads it: 401 and 403 become `authentication`, and 429 becomes `rate_limit`.
  - 408 now gets the existing transient rule (`provider_failure`). Every other 4xx, including an unpulled model's 404, stays `request_rejected`.
  - The error's class, status (400) and message do not change, so chat responses do not either (measured below).

**Tech Stack:** Python 3.12, httpx (`Response` and `HTTPStatusError` in tests), pytest, FastAPI `TestClient` (chat probes only).

**Spec:** `backlog/tasks/task-13865 - Summary-errors-an-egress-blocked-endpoint-an-unconfigured-custom-endpoint-and-local-provider-4xx-replies-get-the-wrong-code.md` on `dev`. Background: `Docs/superpowers/plans/2026-10-18-summary-provider-error-codes.md` (TASK-13844, merged as #3279).

## What dev does today

Measured on origin/dev `a6719bf24f` (2026-10-20) in a scratch worktree, then re-checked at `2b925562e5` and, after the review, at `e521605198`. None of the files this plan touches or reads changed between the three. The probe called `analyze(..., raise_on_error=True)` through the real adapters:
- AC1: `CustomOpenAIAdapter`, with its `http_fetcher` and `http_streamer` raising `EgressPolicyError`.
- AC2: `custom-openai-api` and `custom-openai-api-2`, with `resolve_trusted_provider_endpoint` returning `None`.
- AC3: `OllamaAdapter`, `LlamaCppAdapter` and `KoboldAdapter`, with a configured endpoint and an `http_fetcher` that returns the status.
- A local server that is down: the same three adapters through the real `http_client.fetch`, whose httpx client has a `MockTransport` that raises `httpx.ConnectError`. No socket is opened.

The "after" column is the same probe run on a prototype of this plan.

| Failure | Dev, `raise_on_error=True` | Dev, `raise_on_error=False` | After, `raise_on_error=True` | After, `raise_on_error=False` |
|---|---|---|---|---|
| Egress `host_denied` or `origin_mismatch`, chat or stream | `EgressPolicyError` escapes untyped | `EgressPolicyError` escapes | `configuration` | `EgressPolicyError` (unchanged) |
| Egress `dns_unresolved`, chat or stream | `EgressPolicyError` escapes untyped | `EgressPolicyError` escapes | `provider_failure` | `EgressPolicyError` (unchanged) |
| `custom-openai-api`, `-2` or `-7` with no base URL | `provider_failure` | `Error: Provider request failed for '<provider>'.` | `configuration` | the same `Error:` string (unchanged) |
| Local 401 or 403 (ollama, llama.cpp, kobold) | `request_rejected` | `Error:` string | `authentication` | `Error:` string |
| Local 429 | `request_rejected` | `Error:` string | `rate_limit` | `Error:` string |
| Local 408 | `request_rejected` | `Error:` string | `provider_failure` | `Error:` string |
| Local 400 or 404 (an unpulled model) | `request_rejected` | `Error:` string | `request_rejected` | `Error:` string |
| Local 500 | `provider_failure` | `Error:` string | `provider_failure` | `Error:` string |
| kobold server down (connection refused) | `NetworkError: ConnectError` escapes untyped | `NetworkError` escapes | `provider_failure` | `NetworkError` (unchanged) |
| ollama or llama.cpp server down | `provider_failure` | `Error:` string | `provider_failure` | `Error:` string |

How each happens on dev:
- `EgressPolicyError` (`core/exceptions.py:587`) subclasses `Exception` directly. It is in neither `_SUMMARY_ADAPTER_EXCEPTIONS` (`Summarization_General_Lib.py:110`) nor `_SUMMARIZATION_NONCRITICAL_EXCEPTIONS` (`:97-109`).
  - So it passes the adapter handlers (`:355`, `:406`), `_dispatch_to_api` (`:559`) and `analyze` (`:814`).
  - The adapters let it through on purpose: `custom_openai_adapter.py:311` and `:367`, `local_adapters.py:494` and `:553` (`except EgressPolicyError: raise`). It is not one of the OpenAI adapter's caught classes either (`openai_adapter.py:50-59`).
  - The audio worker's generic handler (`services/audio_jobs_worker.py:579`) then fails the job as retryable and quarantinable, with the policy text as its error. The same failure twice quarantines it (`JOBS_QUARANTINE_THRESHOLD`, default 2), so the transcript never reaches `audio_store`.
- A transport failure from `http_client.fetch` is a `NetworkError` (`core/exceptions.py:598`), which also subclasses `Exception` directly.
  - The OpenAI-compatible local path (`local_adapters.py:553-560`) catches `Exception` and maps it to `ChatProviderError`, so ollama, llama.cpp and the others already read `provider_failure`.
  - Kobold's handler (`local_adapters.py:1097`) catches only `_LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS` (`:81-102`), which has no `NetworkError`. So a kobold server that is down escapes `analyze` untyped, and the audio worker loses the transcript the same way as for an egress block.
  - The fix types it at the summary layer, not by widening `_LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS`. That tuple guards 53 handlers in `local_adapters.py`, most of them parameter coercion, and widening it would also change what kobold raises to chat callers.
- Every egress reason code, and the summary code it now maps to (`_summary_error_code`; the same split chat makes in `chat_service._map_provider_egress_error`):
  - `provider_failure` (retried): `dns_unresolved`. That also covers a DNS resolver timeout and resolver-slot exhaustion, because `Security/egress.py:355-490` returns no addresses for them, which reads as "could not be resolved".
  - `configuration` (not retried): `invalid_url`, `unsupported_scheme`, `userinfo_not_allowed`, `port_not_allowed`, `host_denied`, `origin_mismatch`, `address_forbidden`, `address_invalid`, `dns_changed`, `tls_pin_error`, `tls_pin_missing`, `tls_pin_mismatch`, and a reason of `None`. `None` comes from a refused proxy (`http_client.py:1895`, `:1900`) and from the curl transport checks (`:1627-1644`).
- `custom_openai_adapter.py:158` (`_resolve_base`) and `:211` (`_resolve_transport_context`) raise `RuntimeError("<name> requires an explicit [configured] base URL")`. `_summary_error_code` has no branch for it, so it reads `provider_failure`, and the audio worker retries it.
  - Line 211 is the path a summary takes: the request has no credential handle and no `_endpoint_provenance`.
  - Line 158 is reached only without a credential handle and with `_endpoint_provenance` set to `byok` or `request_override`. Only `_resolve_base`'s unit tests reach it.
- `_raise_sanitized_local_provider_error` (`local_adapters.py:193-227`) makes every 4xx a `ChatBadRequestError` without `upstream_status_code`. Its callers are `local_adapters.py:556` (the OpenAI-compatible path: local-llm, llama.cpp, ooba, tabbyapi, vllm, ollama, aphrodite) and `:1098` (kobold).
  - TASK-13844 made `ChatBadRequestError` read `request_rejected`, except for an `upstream_status_code` of 408 or 498.
  - So a local 401, 403, 408 and 429 all read `request_rejected` and are not retried.

### Chat responses (measured)

Measured with `authenticated_client` posting to `/api/v1/chat/completions`, on dev and on the prototype, with `CHAT_ENFORCE_STRICT_MODEL_SELECTION=0` so that the probe's model `m` reaches the adapter (review M1; with strict selection on, both trees answer `400 model_not_available` before the adapter runs). All transports were stubbed. The two columns match row for row (re-measured on `e521605198`):

| Request | Dev | After |
|---|---|---|
| `custom-openai-api` or `custom-openai-api-2`, no endpoint configured, `stream` false or true | 503 `{"detail": {"error_code": "provider_configuration_invalid", "message": "The selected provider configuration is invalid."}}` | the same |
| `llama.cpp`, upstream 401, 403, 404 or 429, `stream: false` | 400 `{"detail": "Invalid request."}` | the same |
| `llama.cpp`, upstream 401, 403, 404 or 429, `stream: true` | 502 `{"detail": {"error_code": "provider_unavailable", "message": "The chat service provider is currently unavailable."}}` | the same |

Why the custom-endpoint row does not change, though the task expected it to:
- `/chat/completions` always binds a credential handle (`api/v1/endpoints/chat.py:4988`, `cleaned_args[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] = credential_handle`).
- So the adapter takes the `credentials is not None` branch (`custom_openai_adapter.py:191-200`). For a numbered custom slot with no endpoint, that branch already raises `ChatConfigurationError("<name> endpoint is not configured.")` at line 195.
- With a handle, line 158 is reached only for the public subclasses (Novita, Poe, Together), and they all have a `default_base_url`, so it never raises. Line 211 is not reached at all.
- The probe's spy on `_resolve_transport_context` recorded `('credentials', 'server_config', None, 'raised ChatConfigurationError: custom-openai-api-2 endpoint is not configured.')` on both trees.

What does change on the chat side is `chat_service.perform_chat_api_call` called without a credential handle (its in-process callers, not the endpoint):
- Before, it raised `RuntimeError: custom-openai-api-2 requires an explicit configured base URL`, for both `stream=False` and `stream=True` (measured).
- After, it raises `ChatConfigurationError`. That is the error those callers already get from this adapter with a handle, and from every local adapter (`<name> endpoint is not configured.`).
- `chat_service._bound_provider_invocation_exception` (`chat_service.py:3489-3515`) turns it into `ChatConfigurationError("The selected provider configuration is invalid.")`. Before, it turned the `RuntimeError` into `sanitized_provider_stream_exception` (`provider_unavailable`, 502).
- Task 2 pins it with a test.

The local rows do not change because the class (`ChatBadRequestError`), status (400) and message stay the same. Of the code that reads `upstream_status_code`, only one reader is in chat: `chat.py:5336-5346` (`_is_auth_401_error`). It runs only for an OpenAI OAuth credential (`_active_provider_uses_openai_oauth`, `chat.py:5328-5334`). `_bound_provider_invocation_exception` rebuilds the error without the attribute.

## Global Constraints

- Branch `fix/summary-codes-egress-custom-local` from `origin/dev`. One PR, to `dev`.
- No new summary code. `_SUMMARY_ERROR_CODES` and `_RETRYABLE_ANALYSIS_CODES` (`{"rate_limit", "provider_failure"}`) stay as they are. So the Evaluations response mapping that TASK-13844 pinned (`tests/Evaluations/unit/test_specialized_provider_auth_failures.py`, `test_every_summary_code_has_a_chosen_evaluations_response`) needs no new row.
- `analyze(raise_on_error=False)` keeps its current behaviour for every failure here. An egress block still raises `EgressPolicyError`, and a kobold server that is down still raises `NetworkError`. A missing custom base URL and a local 4xx still return the legacy `Error:` string.
- `_LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS` is not widened (it would change chat); a `NetworkError` is typed at the summary layer only.
- No upstream text in any error. `SummaryProviderError` carries only the bounded code and provider name. The new local attribute is an int status.
- `/api/v1/chat/completions` responses do not change (measured above). The local adapters' `ChatBadRequestError` keeps its class, its status (400) and its message.
- `ChatConfigurationError` for the missing base URL keeps the old message text (`<name> requires an explicit base URL` and `<name> requires an explicit configured base URL`), so the tests change only the class they expect.
- Every PR's final sweep includes `tldw_Server_API/tests/CI/test_test_quality_ratchet.py` (enforced on dev since #3243).
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`.
- Backlog changes go only through backlog-py, with `--append-notes` and explicit ids. Stage them with `git add -A backlog/tasks`.
- Editing `Docs/API-related/Audio_Jobs_API.md` needs `bash Helper_Scripts/refresh_docs_published.sh` and the `Docs/Published` diff in the same commit.

Shell setup used by every command below (run from the repository root):

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
T=tldw_Server_API/tests
```

## Decisions for the owner

None. Four choices are settled by existing behaviour:
- **Egress maps to the existing `configuration` code, not a new code.** `configuration` is already not retried. Evaluations already answers it 503 `provider_configuration_invalid`, which is what chat answers for the same block (`chat_service._map_provider_egress_error` returns `ChatConfigurationError`). A new code would need a new row in the Evaluations mapping test, the worker set and the docs, and would tell callers nothing that `configuration` does not.
- **`dns_unresolved` is retried as `provider_failure`.** Chat answers it as an unreachable provider (`ChatProviderError` 503), not as a configuration error, and `provider_readiness.py:163` and `Setup/provider_validation.py:70` single it out the same way. With `max_retries` at its default of 3, the retries wait 5, 10 and 20 s: about 35 s in all, plus each attempt's DNS timeout, before the transcript is stored with `provider_failure` (review M4). Since TASK-13847, the last attempt keeps the transcript, so a host that never resolves loses nothing. A local server that is down (`NetworkError`) is treated the same way.
- **Local 4xx keep their chat response.** Typing a local 401 as `ChatAuthenticationError`, and a local 429 as `ChatRateLimitError`, at the adapter would change what chat users see. The chat side of that is filed as a follow-up in Task 4, not decided here.
- **An evaluation against an egress-blocked or unreachable provider now fails instead of scoring 0 (review M2).**
  - What changes: the response-quality and RAG evaluators re-raise `SummaryProviderError`, so the request now fails with 503 `provider_configuration_invalid`, or with 502 `provider_unavailable` for `dns_unresolved` or a server that is down. Before, `except Exception` scored the metric 0.0. `eval_runner`'s async runs record the sample as an error instead of 0.0 metrics (see the reader audit).
  - Why it is already settled: both evaluators re-raised `SummaryProviderError` before TASK-13844 (`response_quality_evaluator.py` 237, 302, 364, 429, 498 and 561; `rag_evaluator.py` 363, 385 and 424). A missing API key and a switched-off adapter already fail the request the same way.
  - No test or doc pins a 0.0 score for these failures. `test_rag_evaluator.py:422` and `test_evaluation_manager.py:535` pin 0.0 for a mocked `RuntimeError` or `Exception`, which this change does not touch.

## ADR assessment

No ADR is needed. The change maps existing failures to existing summary codes and keeps one int on an existing exception. It makes no durable rule for module boundaries, public API shape, persistence, security or provider integration.
- ADR-025 (the adapter registry is the shared error-mapping point) is followed. The local mapper keeps the upstream status the way the shared `_raise_sanitized_provider_failure` does, and the summary reads it in its one mapping function.
- ADR-026 and ADR-030 (egress policy) are followed. Line 18 of ADR-030 asks that policy failures keep a stable `EgressPolicyError.reason_code` rather than collapse into a generic network error. The transport is unchanged, and the summary reads `reason_code` only to choose between `configuration` and `provider_failure`.

## Overlap with TASK-13864 and open PRs

This plan touches these files and no others:
- `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`:
  - one import after line 45;
  - `_SUMMARY_ADAPTER_EXCEPTIONS` (line 110);
  - `_summary_error_code` (149-171);
  - the two `except _SUMMARY_ADAPTER_EXCEPTIONS` handlers in `_summarize_via_adapter` (355-360, 406-411).

  It does not touch `analyze`, `_dispatch_to_api` or any `LLMQuotaExceededError` handling.
- `tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py` (lines 158 and 211)
- `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py` (`_raise_sanitized_local_provider_error`, lines 211-219)
- `Docs/API-related/Audio_Jobs_API.md` (lines 49-50, the two analysis-failure bullets) and its mirror `Docs/Published/API-related/Audio_Jobs_API.md`
- Tests:
  - `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py`
  - `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py` (one row and two new tests)
  - `tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py`
  - `tldw_Server_API/tests/LLM_Adapters/unit/test_authoritative_custom_openai_endpoint_snapshot.py`
  - `tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py`
  - `tldw_Server_API/tests/LLM_Calls/test_local_http_error_mapping.py`
  - `tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py` (one new test, appended)
- `Docs/superpowers/plans/2026-10-20-summary-codes-egress-custom-local.md` (this plan)
- `backlog/tasks/` (TASK-13865, and two follow-up tasks)

It does not touch `Docs/Operations/Usage_Quotas.md`. If TASK-13864 also edits `Summarization_General_Lib.py`, the conflict is textual unless it edits the same functions; keep both sides.

Open PRs that touch the same files (`gh pr list --state open`, measured 2026-10-20):
- #854 (managed vLLM instance lifecycle) touches `local_adapters.py` and `test_chat_service_base_url_override.py`.
- #3084 touches `Usage_Quotas.md` only.

## Reader audit

AC1 changes what a `raise_on_error=True` caller receives for an egress block, or for a kobold server that is down: a `SummaryProviderError` instead of an `EgressPolicyError` or `NetworkError`. AC2 and AC3 change only the code inside a `SummaryProviderError`. Readers were measured on dev with `git grep -n "raise_on_error"` (every `analyze` caller in `app/` that passes it) and `git grep -n "SummaryProviderError"`. The RAG rows follow the TASK-13844 reader audit.

| Reader | Egress block (AC1) | No custom base URL (AC2) | Local 401/403 (AC3) | Local 429 (AC3) |
|---|---|---|---|---|
| `services/audio_jobs_worker.py:541-576` | Dev: generic handler at 579, retried, then quarantined, so the transcript never reaches `audio_store`. After: `configuration`, so the transcript is stored at once with `analysis_error`. For `dns_unresolved`: `provider_failure`, retried. | Dev: `provider_failure`, retried. After: `configuration`, stored at once. | Dev: `request_rejected`, stored at once. After: `authentication`, stored at once. | Dev: `request_rejected`, not retried. After: `rate_limit`, retried after 30 s. |
| `core/Evaluations/response_quality_evaluator.py` (6 calls; the first's handlers are at 237 and 239) and `core/Evaluations/rag_evaluator.py:99` (handlers at 363 and 365) | Dev: `except Exception` swallowed it (the response-quality metric scored 0.0, "Provider evaluation failed."; the RAG evaluator kept partial results). After: re-raised, and `evaluations_unified.py` (1178, 1378, 1574) answers 503 `provider_configuration_invalid`, or 502 `provider_unavailable` for `dns_unresolved`. | Dev: 502 `provider_unavailable`. After: 503 `provider_configuration_invalid`. | Dev: 502 `provider_unavailable`. After: 502 `provider_authentication_failed`. | 502 `provider_unavailable` before and after. |
| `core/RAG/rag_service/hyde.py:148` (handler 161, `_record_embedding_degraded` at 299) | heuristic fallback before and after. Stage code: `provider_unavailable` before; `provider_configuration_invalid` after. | `provider_unavailable` before; `provider_configuration_invalid` after. | `provider_unavailable` before; `invalid_provider_credentials` after. | no change |
| `agentic_chunker.py:757`, `document_grader.py:257`, `quality_graders.py:250` and `:555`, `knowledge_strips.py:420`, `post_generation_verifier.py:287`, `unified_pipeline.py:5219`, `6308`, `7257`, `7741` and `9035` | record the constant `provider_unavailable`, or a heuristic fallback, before and after | no change | no change | no change |
| `core/Evaluations/eval_runner.py` (async runs; the evaluator calls are at 780, 1478 and 1541) | Dev: the evaluators swallowed it, so the sample got 0.0 metric scores. After: `_EVAL_RUNNER_NONCRITICAL_EXCEPTIONS` (71-81) does not hold `SummaryProviderError`, so it reaches `_process_batch`'s `gather(return_exceptions=True)` (1359), which records the sample as `{"error": "Summary provider error: configuration (<provider>)"}`. | The sample's error names `configuration` (was `provider_failure`). | The sample's error names `authentication` (was `request_rejected`). | The sample's error names `rate_limit` (was `request_rejected`). |
| `services/outputs_service.py:669` (handler 673) | logs `str(exc)` and returns `""`. Before, the log line carried the policy text; after, `Summary provider error: configuration (<provider>)`. | logged | logged | logged |

A kobold server that is down (review M5) reads like `dns_unresolved` in every row:
- the audio worker retries it as `provider_failure`, then stores the transcript with `analysis_error`;
- Evaluations answers 502 `provider_unavailable` where the metric scored 0.0;
- HyDE records `provider_unavailable`, as before.

The Evaluations mapping for every code is already pinned by `test_summary_code_maps_to_its_evaluations_response`. The audio worker rows are pinned by the worker tests (Task 1, and the existing `rate_limit` and `authentication` worker tests).

## Review Focus

These are the input classes and failure modes the tests below do not exercise:

1. **In-process callers without a credential handle now get `ChatConfigurationError` for a missing custom base URL, where they got `RuntimeError` (Task 2, review I1).**
   - How it was measured: the review scanned the enclosing handlers of all 96 `chat_api_call` and `perform_chat_api_call` call sites, and probed five callers with custom-openai-api-2 and no endpoint.
   - *Degrade → raise.* These sites catch `RuntimeError` but neither `ChatAPIError` nor `Exception`. They used to return an error payload or `""`, and now let the error through:
     - `Workflows/adapters/llm/llm.py:282` (`llm_with_tools`), `:391` (`llm_compare`), `:483` and `:506` (`llm_critique`);
     - `Workflows/adapters/rag/query.py:151` (`query_rewrite`) and `:383` (`hyde_generate`);
     - `Workflows/adapters/media/documents.py:520`;
     - `Workflows/adapters/content/audio_briefing.py:319` and `:1045`;
     - `Workflows/adapters/integration/messaging.py:958`;
     - `Claims_Extraction/claims_service.py:611` (`_fva_claims_analyze_call`).
   - *Raise → degrade.* `Web_Scraping/WebSearch_APIs.aggregate_results` (its call is at `:1381`) catches `ChatConfigurationError` but not `ChatAPIError`.
     - On dev, `chat_api_call` wrapped the `RuntimeError` as a `ChatAPIError` (500), which escaped.
     - Now the call returns its "Could not create the report due to an error." answer.
   - *Reclassified, and the user sees it.* The Setup first-chat verification reports `configuration_error` ("Provider configuration is incomplete or invalid.") where it reported `provider_error`. Task 2 pins it with a test.
   - Not affected: `Moderation/semantic_matcher.py:149` also catches `RuntimeError`, but it hard-codes `openai`.
   - Each new behaviour is what the same caller already does for a credentialed `ChatConfigurationError` (`custom_openai_adapter.py:195`), or for any other `ChatAPIError`.
   - What is not tested: the Workflows, FVA, audio-briefing, messaging and web-search paths. Only the Setup category and `perform_chat_api_call` are pinned. Task 4's sweep runs every test file that names `custom-openai-api` or the custom adapter.
2. **Egress reason codes that may be transient besides `dns_unresolved` (review M3; the full mapping is under "What dev does today").** Both read `configuration` and are not retried. Chat maps both the same way.
   - `dns_changed` (`Security/egress.py:816-822`): the DNS answer changed between the policy check and the request.
   - `tls_pin_error` (`http_client.py:2495-2525`): it wraps any connect failure or timeout during certificate-pin verification. It applies only to hosts with configured pins.
   - The cost: a DNS-rotation race, or a pinned host that times out, loses that job's analysis, though the transcript is kept.
3. **Streamed local summaries.**
   - What happens: a local adapter's stream turns any upstream failure into an in-band SSE error frame (`local_adapters.py:494-506`). The summary maps that frame to `provider_failure`, whatever the status.
   - Why it is not changed: no `raise_on_error` caller streams. TASK-13844's plan lists the same gap for in-stream SSE errors.
4. **Evaluations against an egress-blocked provider.**
   - What changes: the response-quality and RAG evaluators now fail the request with 503 `provider_configuration_invalid` where they used to score the metric 0.0. A missing API key and a switched-off adapter already fail it the same way.
   - What is tested: the code mapping is pinned. The evaluator path for an egress block is not tested end to end.
5. **A local 401, 403 or 404 that is really a configuration problem.**
   - What happens: a reverse proxy's IP allowlist answering 403 reads `authentication`. A base URL with the wrong path answering 404 reads `request_rejected`. Neither is retried, which is right.
   - What may mislead: the audio jobs doc names only the commoner causes.

---

### Task 1: an egress-blocked endpoint, or a local server that is down, is typed instead of escaping (AC1, review M5)

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`:
  - `:42-45` (import);
  - `:110` (`_SUMMARY_ADAPTER_EXCEPTIONS`);
  - `:149-171` (`_summary_error_code`);
  - `:355-360` and `:406-411` (the two handlers).
- Create: `Docs/superpowers/plans/2026-10-20-summary-codes-egress-custom-local.md` (this plan, committed with Task 1)
- Test: `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py` (imports at 1-25; new tests after `test_a_disabled_native_http_adapter_is_a_configuration_error`, which ends at line 484)
- Test: `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py:1094-1098` (one row), plus two new tests after `test_audio_analyze_failure_a_retry_cannot_fix_is_not_retried`

**Interfaces:**
- Produces: `_SUMMARY_PASSTHROUGH_EXCEPTIONS = (EgressPolicyError, NetworkError)`, which `_SUMMARY_ADAPTER_EXCEPTIONS` now contains.
- Produces: `_summary_error_code(exc, adapter=None) -> str` maps an `EgressPolicyError` to `"configuration"`, or to `"provider_failure"` when `exc.reason_code == "dns_unresolved"`. A `NetworkError` falls through to `"provider_failure"`.
- Produces, in `test_summarization_adapter.py`:
  - the module aliases `custom_openai_mod` (Task 2 uses it) and `local_mod`;
  - the imports `TrustedProviderEndpoint` and `ConfiguredEndpointScope`;
  - the helper `_configured_local_endpoint(monkeypatch) -> None` (Task 3 uses it);
  - the helpers `_custom_openai_egress_blocked(monkeypatch, reason_code: str) -> None` and `_local_server_down(monkeypatch) -> None`.

- [ ] **Step 1: branch and mark the task in progress**

```bash
git fetch origin dev --quiet
git switch -c fix/summary-codes-egress-custom-local origin/dev
cp /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad/plan_fu_summary_codes.md Docs/superpowers/plans/2026-10-20-summary-codes-egress-custom-local.md
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13865 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-10-20-summary-codes-egress-custom-local.md (four tasks: egress and a local server that is down, custom base URL, local 4xx, docs and close-out; independent review folded in: 0 Blocker, 1 Important, 6 Minor). ADR assessment: none needed (the failures map to existing summary codes and one int is kept on ChatBadRequestError; ADR-025 followed, the local mapper keeps the upstream status as the shared _raise_sanitized_provider_failure does; ADR-026/030 followed, the transport keeps EgressPolicyError.reason_code and the summary only reads it)."
```

- [ ] **Step 2: write the failing tests**

In `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py`, extend the imports. The first block becomes:

```python
import asyncio
import inspect
from contextlib import contextmanager
from typing import Any

import httpx
import pytest

import tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter as custom_openai_mod
import tldw_Server_API.app.core.LLM_Calls.providers.local_adapters as local_mod
import tldw_Server_API.app.core.LLM_Calls.providers.moonshot_adapter as moonshot_mod
import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod
```

and the last two import lines become:

```python
from tldw_Server_API.app.core.exceptions import EgressPolicyError, NetworkError
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as sgl
from tldw_Server_API.app.core.LLM_Calls import chat_calls
from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import TrustedProviderEndpoint
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope
```

Insert after `test_a_disabled_native_http_adapter_is_a_configuration_error` (before `test_dispatch_failure_is_typed_for_runtime_callers`).
- The egress tests go through the real `CustomOpenAIAdapter`, which lets `EgressPolicyError` through on purpose (`custom_openai_adapter.py:311`, `:367`).
- The local-server tests go through the real kobold adapter and the real `http_client.fetch`. The shared httpx client is swapped for one whose `MockTransport` raises `httpx.ConnectError`, so no socket is opened.

```python
def _custom_openai_egress_blocked(monkeypatch, reason_code: str) -> None:
    """Give custom-openai-api a configured endpoint whose every request the outbound policy refuses."""
    monkeypatch.setenv("CUSTOM_OPENAI_API_IP", "http://configured-custom:18091/v1")

    def _refuse(**_kwargs):
        raise EgressPolicyError(SECRET, reason_code=reason_code)

    @contextmanager
    def _refuse_stream(**_kwargs):
        raise EgressPolicyError(SECRET, reason_code=reason_code)
        yield  # pragma: no cover

    monkeypatch.setattr(custom_openai_mod.CustomOpenAIAdapter, "http_fetcher", staticmethod(_refuse))
    monkeypatch.setattr(custom_openai_mod.CustomOpenAIAdapter, "http_streamer", staticmethod(_refuse_stream))


@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True], ids=["chat", "stream"])
@pytest.mark.parametrize(
    ("reason_code", "expected_code"),
    [
        ("host_denied", "configuration"),
        ("origin_mismatch", "configuration"),
        ("dns_unresolved", "provider_failure"),
    ],
    ids=["host-denied", "origin-mismatch", "dns-unresolved"],
)
def test_an_egress_blocked_endpoint_gets_a_summary_code(monkeypatch, reason_code, expected_code, streaming):
    _custom_openai_egress_blocked(monkeypatch, reason_code)

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        result = sgl.analyze(
            "custom-openai-api", "hello", None, api_key="key", model_override="m", streaming=streaming, raise_on_error=True
        )
        if streaming:
            list(result)

    assert exc_info.value.code == expected_code
    assert exc_info.value.__cause__ is None
    assert SECRET not in repr(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True], ids=["chat", "stream"])
def test_untyped_callers_still_get_the_egress_policy_error(monkeypatch, streaming):
    _custom_openai_egress_blocked(monkeypatch, "host_denied")

    with pytest.raises(EgressPolicyError) as exc_info:
        result = sgl.analyze("custom-openai-api", "hello", None, api_key="key", model_override="m", streaming=streaming)
        if streaming:
            list(result)

    assert exc_info.value.reason_code == "host_denied"


def _configured_local_endpoint(monkeypatch) -> None:
    """Give every local adapter the configured endpoint http://127.0.0.1:8080."""
    endpoint = TrustedProviderEndpoint(
        base_url="http://127.0.0.1:8080", scope=ConfiguredEndpointScope.from_url("http://127.0.0.1:8080")
    )
    monkeypatch.setattr(local_mod, "resolve_trusted_provider_endpoint", lambda _name: endpoint)


def _local_server_down(monkeypatch) -> None:
    """A configured local endpoint whose server refuses every connection; no socket is opened."""
    _configured_local_endpoint(monkeypatch)

    def _refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    refusing_client = httpx.Client(transport=httpx.MockTransport(_refuse))
    monkeypatch.setattr("tldw_Server_API.app.core.http_client._get_httpx_client", lambda **_kwargs: refusing_client)


@pytest.mark.unit
def test_a_local_server_that_is_down_is_a_provider_failure(monkeypatch):
    _local_server_down(monkeypatch)

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        sgl.analyze("kobold", "hello", None, api_key="key", model_override="m", raise_on_error=True)

    assert exc_info.value.code == "provider_failure"
    assert exc_info.value.__cause__ is None


@pytest.mark.unit
def test_untyped_callers_still_get_the_network_error(monkeypatch):
    _local_server_down(monkeypatch)

    with pytest.raises(NetworkError):
        sgl.analyze("kobold", "hello", None, api_key="key", model_override="m")
```

The two `test_untyped_callers_still_get_*` tests are the AC1 controls for `raise_on_error=False`. They pass on dev and must stay green.

In `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py`, give `test_audio_analyze_failure_a_retry_cannot_fix_is_not_retried` a third row (lines 1094-1098 become):

```python
    [
        (lambda errors: errors.ChatBadRequestError(provider="openai"), "request_rejected"),
        (lambda errors: errors.ChatRateLimitError(provider="openai", quota_exhausted=True), "provider_quota_exhausted"),
        (lambda errors: errors.EgressPolicyError("blocked-host.example", reason_code="host_denied"), "configuration"),
    ],
    ids=["rejected-request", "spent-quota", "egress-blocked"],
```

and insert these two tests after that test, before the `# --- the audio_convert -> audio_transcribe hand-off (TASK-13848) ---` comment. They pin the worker's side of a local server that is down: retried as a provider failure, then stored with the transcript on the last attempt. They do not assert the backoff, which is timing-sensitive (see Task 4, Step 3).

```python
@pytest.mark.unit
async def test_audio_analyze_unreachable_provider_is_retried_as_a_provider_failure(analyze_worker):
    from tldw_Server_API.app.core.exceptions import NetworkError

    worker, jm, adapter = analyze_worker
    adapter.error = NetworkError("ConnectError")
    job_id = _analyze_job(jm, "hello transcript")
    await _run_worker_until(worker, lambda: _settled(jm, job_id))
    job = jm.get_job(job_id) or {}
    assert (job.get("status"), job.get("retry_count"), job.get("error_code")) == ("queued", 1, "provider_failure")
    assert _stored(jm) == []


@pytest.mark.unit
async def test_audio_analyze_unreachable_provider_on_the_last_attempt_keeps_the_transcript(analyze_worker):
    from tldw_Server_API.app.core.exceptions import NetworkError

    worker, jm, adapter = analyze_worker
    adapter.error = NetworkError("ConnectError")
    job_id = _analyze_job(jm, "hello transcript")
    max_retries = _exhaust_retries(jm, job_id)
    await _run_worker_until(worker, lambda: _settled(jm, job_id, retries=max_retries + 1))
    assert _stored(jm), "audio_analyze did not hand over to audio_store"
    payload = _stored(jm)[0]["payload"]
    assert payload["text"] == "hello transcript"
    assert payload["analysis_error"]["code"] == "provider_failure"
    assert "analysis" not in payload
```

- [ ] **Step 3: run them and watch them fail**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q $T/LLM_Calls/test_summarization_adapter.py -k "egress or local_server or network_error"
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q $T/AudioJobs/test_audio_worker_smoke.py -k "retry_cannot_fix or unreachable_provider"
```

Expected (measured on dev `e521605198`, network blocked):
- First command: `7 failed, 3 passed`.
  - All six `test_an_egress_blocked_endpoint_gets_a_summary_code` cases fail with `tldw_Server_API.app.core.exceptions.EgressPolicyError: upstream-secret-body`: the error escaped `analyze` untyped.
  - `test_a_local_server_that_is_down_is_a_provider_failure` fails with `tldw_Server_API.app.core.exceptions.NetworkError: ConnectError`.
  - The three `test_untyped_callers_still_get_*` controls pass.
- Second command: `3 failed, 2 passed`.
  - `[egress-blocked]` fails with `AssertionError: assert 2 == 0` on `retry_count`: the generic handler retried the job. The count depends on how far the retries got in the test's 5 s.
  - `test_audio_analyze_unreachable_provider_is_retried_as_a_provider_failure` fails with `AssertionError: assert ('quarantined', 2, None) == ('queued', 1, 'provider_failure')` in the measured run. The generic handler recorded no code; how far the retries got before the check (and so the status and count) depends on timing.
  - `test_audio_analyze_unreachable_provider_on_the_last_attempt_keeps_the_transcript` fails with `AssertionError: audio_analyze did not hand over to audio_store`.
  - The `[rejected-request]` and `[spent-quota]` rows pass.

- [ ] **Step 4: implement**

In `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`, import the two exceptions after the `custom_openai_providers` import (lines 42-45):

```python
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_provider_name,
    custom_openai_provider_number,
)
from tldw_Server_API.app.core.exceptions import EgressPolicyError, NetworkError
```

Replace line 110:

```python
# Not ChatAPIErrors, and some adapters let them through untouched: an endpoint the egress policy
# blocks, and a transport failure such as a local server that is down (it falls through to
# provider_failure). They are caught only to type them for raise_on_error callers; the other
# callers still get the exception itself.
_SUMMARY_PASSTHROUGH_EXCEPTIONS = (EgressPolicyError, NetworkError)
_SUMMARY_ADAPTER_EXCEPTIONS = (ChatAPIError, *_SUMMARY_PASSTHROUGH_EXCEPTIONS) + _SUMMARIZATION_NONCRITICAL_EXCEPTIONS
```

Add the first branch of `_summary_error_code`, so it reads:

```python
def _summary_error_code(exc: BaseException, adapter: Any = None) -> str:
    """Map chat adapter failures to a bounded, detail-free summary taxonomy."""
    if isinstance(exc, EgressPolicyError):
        # As chat_service._map_provider_egress_error: a host that did not resolve may resolve on a
        # retry; an endpoint the outbound policy blocks stays blocked until the configuration changes.
        return "provider_failure" if exc.reason_code == "dns_unresolved" else "configuration"
    if isinstance(exc, ChatAuthenticationError):
        return "authentication"
    if isinstance(exc, ChatConfigurationError):
        return "configuration"
    if isinstance(exc, ChatRateLimitError):
        return "provider_quota_exhausted" if exc.quota_exhausted is True else "rate_limit"
    if isinstance(exc, ChatBadRequestError):
        # 408 (OpenRouter timeout) and 498 (Groq flex capacity) are transient 4xx replies.
        if getattr(exc, "upstream_status_code", None) in {408, 498}:
            return "provider_failure"
        # The provider refused this request (an unknown model, a prompt over its context);
        # sending it again gets the same answer.
        return "request_rejected"
    if isinstance(exc, ChatAPIError):
        if exc.status_code in {401, 403}:
            return "authentication"
        if exc.status_code == 429:
            return "rate_limit"
    if isinstance(exc, RuntimeError) and _native_http_disabled(adapter):
        return "configuration"
    return "provider_failure"
```

In `_summarize_via_adapter`, the stream generator's handler (lines 355-360) becomes:

```python
            except _SUMMARY_ADAPTER_EXCEPTIONS as exc:
                if raise_on_error:
                    logging.error(f"Adapter streaming failed for {provider}")
                    raise SummaryProviderError(code=_summary_error_code(exc, adapter), provider=provider) from None
                if isinstance(exc, _SUMMARY_PASSTHROUGH_EXCEPTIONS):
                    raise
                logging.error(f"Adapter streaming failed for {provider}")
                yield f"Error: Provider streaming failed for '{provider}'."
```

and the non-streaming handler (lines 406-411) becomes:

```python
    except _SUMMARY_ADAPTER_EXCEPTIONS as exc:
        if raise_on_error:
            logging.error(f"Adapter summarization failed for {provider}")
            raise SummaryProviderError(code=_summary_error_code(exc, adapter), provider=provider) from None
        if isinstance(exc, _SUMMARY_PASSTHROUGH_EXCEPTIONS):
            raise
        logging.error(f"Adapter summarization failed for {provider}")
        return f"Error: Provider request failed for '{provider}'."
```

The bare `raise` re-raises the original exception with its traceback, so a `raise_on_error=False` caller sees exactly what it sees on dev. `_LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS` is not widened (see "What dev does today").

- [ ] **Step 5: run the new tests, then both whole files**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q $T/LLM_Calls/test_summarization_adapter.py -k "egress or local_server or network_error"
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q $T/AudioJobs/test_audio_worker_smoke.py -k "retry_cannot_fix or unreachable_provider"
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q $T/LLM_Calls/test_summarization_adapter.py $T/AudioJobs/test_audio_worker_smoke.py
```

Expected: `10 passed`, then `5 passed`. The two whole files then pass. The only non-passing result is the skip the smoke file already has on dev (`1 skipped`). If `test_audio_analyze_error_reply_is_retried_as_a_provider_failure` fails with `assert 3.9... >= 4`, run it again: it is timing-sensitive on unchanged dev too (see Task 4, Step 3).

- [ ] **Step 6: commit**

```bash
git add Docs/superpowers/plans/2026-10-20-summary-codes-egress-custom-local.md tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py $T/LLM_Calls/test_summarization_adapter.py $T/AudioJobs/test_audio_worker_smoke.py
git add -A backlog/tasks
git commit -F - <<'EOF'
fix(summarization): an egress-blocked endpoint or a local server that is down gets a summary code (TASK-13865)

EgressPolicyError and the transport's NetworkError subclass Exception
directly and were in none of the summary's handlers, so they escaped
analyze() untyped: an endpoint the outbound policy blocks, and a kobold
server that is down (kobold's handler does not catch NetworkError). The
audio analysis stage's generic handler then retried the job and
quarantined it, so the transcript never reached audio_store.

The summary now types both for raise_on_error callers. An egress block is
configuration, which is not retried, and a host that did not resolve is
provider_failure, as chat_service._map_provider_egress_error already does.
A NetworkError is provider_failure, so the stage retries it and then
stores the transcript with analysis_error. raise_on_error=False callers
still get the exception itself.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 2: a custom OpenAI-compatible provider with no base URL is a configuration error (AC2)

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py:158` (`_resolve_base`) and `:211` (`_resolve_transport_context`)
- Test: `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py` (new test after Task 1's `test_untyped_callers_still_get_the_egress_policy_error`)
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py:7` (import), `:140` and `:143` (the class each `pytest.raises` expects)
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_authoritative_custom_openai_endpoint_snapshot.py:7-9` (import), `:95` (the class `pytest.raises` expects)
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py` (new test before `test_dns_unresolved_maps_to_sanitized_reachability_error`, line 141)
- Test: `tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py` (new test appended after `test_first_chat_verifier_maps_model_not_found_to_model_unavailable`, the last test; review I1)

**Interfaces:**
- Consumes: Task 1's `custom_openai_mod` import alias in `test_summarization_adapter.py`.
- Produces: `CustomOpenAIAdapter._resolve_base` and `CustomOpenAIAdapter._resolve_transport_context` raise `ChatConfigurationError(provider=self.name, message=...)` where they raised `RuntimeError`. The messages are unchanged.

The tests that pin `RuntimeError` today, found with `git grep -n "requires an explicit" tldw_Server_API/tests`:
- `test_custom_openai_native_http.py:140` and `:143`, in one test, `test_numbered_custom_openai_adapters_require_explicit_base_url`.
- `test_authoritative_custom_openai_endpoint_snapshot.py:95`, in `test_resolved_numbered_custom_adapter_fails_closed_when_snapshot_has_no_endpoint` (3 adapters × 2 ids = 6 cases).

Both reach line 158. No test pins line 211's error; Task 2's summary test is the first.

- [ ] **Step 1: write the failing tests**

In `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py`, insert after `test_untyped_callers_still_get_the_network_error` (Task 1). This goes through line 211, the path a summary takes:

```python
@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True], ids=["chat", "stream"])
@pytest.mark.parametrize("provider", ["custom-openai-api", "custom-openai-api-2"])
def test_a_custom_provider_without_a_base_url_is_a_configuration_error(monkeypatch, provider, streaming):
    monkeypatch.setattr(custom_openai_mod, "resolve_trusted_provider_endpoint", lambda _name: None)

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        result = sgl.analyze(
            provider, "hello", None, api_key="key", model_override="m", streaming=streaming, raise_on_error=True
        )
        if streaming:
            list(result)

    assert exc_info.value.code == "configuration"
```

In `tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py`, line 7 becomes:

```python
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError, ChatConfigurationError
```

and both `pytest.raises` lines in `test_numbered_custom_openai_adapters_require_explicit_base_url` (140 and 143) become:

```python
    with pytest.raises(ChatConfigurationError, match="requires an explicit base URL"):
```

In `tldw_Server_API/tests/LLM_Adapters/unit/test_authoritative_custom_openai_endpoint_snapshot.py`, add the import after `import pytest`:

```python
import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import (
```

and line 95 becomes:

```python
    with pytest.raises(ChatConfigurationError, match="requires an explicit base URL"):
```

In `tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py`, insert before `test_dns_unresolved_maps_to_sanitized_reachability_error`. This pins the one chat-side change, for an in-process caller with no credential handle. The file already imports `chat_service`, `pytest` and `ChatConfigurationError`.

```python
@pytest.mark.parametrize("stream", [False, True], ids=["chat", "stream"])
def test_a_custom_provider_without_a_base_url_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    """A caller with no credential handle gets the error a credentialed caller already gets."""
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter

    monkeypatch.setattr(custom_openai_adapter, "resolve_trusted_provider_endpoint", lambda _name: None)

    with pytest.raises(ChatConfigurationError):
        result = chat_service.perform_chat_api_call(
            api_provider="custom-openai-api-2",
            messages=[{"role": "user", "content": "hi"}],
            model="model",
            stream=stream,
        )
        if stream:
            list(result)
```

Append to `tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py`. This pins the one user-facing response that changes (review I1). The Setup first-chat verification calls `perform_chat_api_call_async` without a credential handle, so it reaches line 211. The test goes through the real verifier, chat service and adapter, and the adapter's fetcher fails the test if a request were ever sent. The file already imports `pytest`.

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["custom-openai-api", "custom-openai-api-2"])
async def test_first_chat_verifier_reports_a_custom_provider_without_a_base_url_as_configuration(
    monkeypatch, provider
):
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter
    from tldw_Server_API.app.core.Setup import first_chat_verifier

    def _no_request(**_kwargs):
        raise AssertionError("no request is sent without a base URL")

    monkeypatch.setattr(custom_openai_adapter, "resolve_trusted_provider_endpoint", lambda _name: None)
    monkeypatch.setattr(custom_openai_adapter.CustomOpenAIAdapter, "http_fetcher", staticmethod(_no_request))

    result = await first_chat_verifier.verify_first_chat(provider=provider, model="m")

    assert (result.status, result.failure_category) == ("failed", "configuration_error")
```

- [ ] **Step 2: run them and watch them fail**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  $T/LLM_Calls/test_summarization_adapter.py \
  $T/LLM_Adapters/unit/test_custom_openai_native_http.py \
  $T/LLM_Adapters/unit/test_authoritative_custom_openai_endpoint_snapshot.py \
  $T/Chat/unit/test_chat_service_base_url_override.py \
  $T/Setup/test_setup_first_chat_completion.py \
  -k "without_a_base_url or require_explicit_base_url or fails_closed"
```

Expected (measured on dev `e521605198`, network blocked): `15 failed`.
- The 4 summary cases fail with `AssertionError: assert 'provider_failure' == 'configuration'`.
- The 7 adapter cases fail with `RuntimeError: custom-openai-api-<n> requires an explicit base URL` (n is 2, 3 or 99).
- The 2 chat_service cases fail with `RuntimeError: custom-openai-api-2 requires an explicit configured base URL`.
- The 2 Setup cases fail with `AssertionError: assert ('failed', 'provider_error') == ('failed', 'configuration_error')`.

- [ ] **Step 3: implement**

In `tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py` (it already imports `ChatConfigurationError`, line 12), line 158 in `_resolve_base` becomes:

```python
                raise ChatConfigurationError(
                    provider=self.name,
                    message=f"{self.name} requires an explicit base URL",
                )
```

and line 211 in `_resolve_transport_context` becomes:

```python
            raise ChatConfigurationError(
                provider=self.name,
                message=f"{self.name} requires an explicit configured base URL",
            )
```

`_summary_error_code` already maps `ChatConfigurationError` to `configuration`. No summary change is needed.

- [ ] **Step 4: run the new tests, then the whole files**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  $T/LLM_Calls/test_summarization_adapter.py \
  $T/LLM_Adapters/unit/test_custom_openai_native_http.py \
  $T/LLM_Adapters/unit/test_authoritative_custom_openai_endpoint_snapshot.py \
  $T/Chat/unit/test_chat_service_base_url_override.py \
  $T/Setup/test_setup_first_chat_completion.py \
  -k "without_a_base_url or require_explicit_base_url or fails_closed"
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  $T/LLM_Calls/test_summarization_adapter.py \
  $T/LLM_Adapters/unit/test_custom_openai_native_http.py \
  $T/LLM_Adapters/unit/test_authoritative_custom_openai_endpoint_snapshot.py \
  $T/Chat/unit/test_chat_service_base_url_override.py \
  $T/Setup/test_setup_first_chat_completion.py \
  $T/LLM_Adapters/unit/test_custom_openai_runtime_credentials.py
```

Expected: `15 passed`, then the six whole files pass with no failures. `test_custom_openai_runtime_credentials.py` pins the credentialed path, which still raises `ChatConfigurationError("... endpoint is not configured")` at line 195.

- [ ] **Step 5: commit**

```bash
git add tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py $T/LLM_Calls/test_summarization_adapter.py $T/LLM_Adapters/unit/test_custom_openai_native_http.py $T/LLM_Adapters/unit/test_authoritative_custom_openai_endpoint_snapshot.py $T/Chat/unit/test_chat_service_base_url_override.py $T/Setup/test_setup_first_chat_completion.py
git commit -F - <<'EOF'
fix(llm): a custom OpenAI-compatible provider with no base URL is a configuration error (TASK-13865)

Without a credential handle, CustomOpenAIAdapter raised a bare RuntimeError
for a missing base URL (_resolve_base and _resolve_transport_context). A
summary recorded it as provider_failure, and the audio analysis stage
retried it. With a handle, the same adapter already raised
ChatConfigurationError for the same condition.

Both raises are now ChatConfigurationError with the same messages, so the
summary records configuration. /chat/completions always binds a handle and
answers as before (503 provider_configuration_invalid). An in-process
perform_chat_api_call without a handle now gets ChatConfigurationError
instead of RuntimeError. So the Workflows LLM, RAG, media, audio-briefing
and messaging steps, the FVA analyze function, the web-search aggregation
report and the Setup first-chat category now behave as they already do for
a credentialed configuration error; a new Setup test pins the first-chat
category (configuration_error, was provider_error).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 3: a local provider's 401, 403 and 429 are not a rejected request (AC3)

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py:211-219` (the 4xx branch of `_raise_sanitized_local_provider_error`)
- Modify: `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py` (the `ChatBadRequestError` branch of `_summary_error_code`, as Task 1 left it)
- Test: `tldw_Server_API/tests/LLM_Calls/test_local_http_error_mapping.py` (new test before `test_local_openai_raises_provider_error_on_5xx`, line 35)
- Test: `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py` (new test after Task 2's `test_a_custom_provider_without_a_base_url_is_a_configuration_error`)

**Interfaces:**
- Consumes: `_summary_error_code` as Task 1 left it; Task 1's `local_mod` alias and `_configured_local_endpoint(monkeypatch)` helper in `test_summarization_adapter.py`.
- Produces: the `ChatBadRequestError` that `_raise_sanitized_local_provider_error` raises carries `upstream_status_code: int` whenever the upstream status is known. `_summary_error_code` maps a `ChatBadRequestError` whose `upstream_status_code` is 401 or 403 to `"authentication"`, and 429 to `"rate_limit"`.

- [ ] **Step 1: write the failing tests**

In `tldw_Server_API/tests/LLM_Calls/test_local_http_error_mapping.py`, insert before `test_local_openai_raises_provider_error_on_5xx`. The first two assertions pin the unchanged chat contract (class, status, message):

```python
@pytest.mark.unit
@pytest.mark.parametrize("status_code", [401, 403, 404, 429])
def test_local_openai_rejection_keeps_the_upstream_status(status_code):
    error = _make_httpx_error(status_code)

    def _fetch(**_kwargs):
        raise error

    with pytest.raises(ChatBadRequestError) as exc:
        _chat_with_openai_compatible_local_server(
            api_base_url="http://localhost:8000",
            model_name="local-model",
            input_data=[{"role": "user", "content": "ping"}],
            http_fetcher=_fetch,
        )
    assert exc.value.status_code == 400
    assert exc.value.message == "The local model provider rejected the request."
    assert exc.value.upstream_status_code == status_code
```

In `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py` (Task 1 added the imports this needs), insert after `test_a_custom_provider_without_a_base_url_is_a_configuration_error` (Task 2). It goes through the real local adapters, so the whole chain is under test: the HTTP status, `_raise_sanitized_local_provider_error`, then the summary code.
- `ollama` covers the OpenAI-compatible path (`local_adapters.py:556`, shared by local-llm, llama.cpp, ooba, tabbyapi, vllm and aphrodite).
- `kobold` covers its own path (`:1098`).

```python
def _local_upstream_replies(monkeypatch, status: int) -> None:
    """Give every local adapter a configured endpoint whose every request gets ``status``."""
    _configured_local_endpoint(monkeypatch)

    def _reply(**kwargs):
        return httpx.Response(status, request=httpx.Request("POST", kwargs["url"]), json={"error": {"message": SECRET}})

    monkeypatch.setattr(local_mod._LocalAdapterBase, "http_fetcher", staticmethod(_reply))


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["ollama", "kobold"])
@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (401, "authentication"),
        (403, "authentication"),
        (429, "rate_limit"),
        (408, "provider_failure"),
        (404, "request_rejected"),
        (400, "request_rejected"),
        (500, "provider_failure"),
    ],
    ids=["unauthorized", "forbidden", "rate-limit", "timed-out", "unpulled-model", "bad-request", "server-error"],
)
def test_a_local_provider_reply_reaches_the_summary_code(monkeypatch, provider, status, expected_code):
    _local_upstream_replies(monkeypatch, status)

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        sgl.analyze(provider, "hello", None, api_key="key", model_override="m", raise_on_error=True)

    assert exc_info.value.code == expected_code
    assert SECRET not in repr(exc_info.value)
```

The `unpulled-model`, `bad-request` and `server-error` rows are controls: they pass on dev and must stay green.

- [ ] **Step 2: run them and watch them fail**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q $T/LLM_Calls/test_local_http_error_mapping.py -k keeps_the_upstream_status
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q $T/LLM_Calls/test_summarization_adapter.py -k local_provider_reply
```

Expected (measured on dev):
- First command: `4 failed`, each with `AttributeError: 'ChatBadRequestError' object has no attribute 'upstream_status_code'`.
- Second command: `8 failed, 6 passed`.
  - The `unauthorized` and `forbidden` rows fail with `assert 'request_rejected' == 'authentication'`.
  - The `rate-limit` rows fail with `assert 'request_rejected' == 'rate_limit'`.
  - The `timed-out` rows fail with `assert 'request_rejected' == 'provider_failure'`.
  - The six control rows pass.

- [ ] **Step 3: implement**

In `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py`, the 4xx branch of `_raise_sanitized_local_provider_error` (lines 211-219) becomes:

```python
    if isinstance(exc, ChatBadRequestError) or (
        isinstance(status, int) and 400 <= status < 500
    ):
        error = ChatBadRequestError(
            provider=provider,
            message="The local model provider rejected the request.",
        )
        if status is not None:
            # As ChatProvider._raise_sanitized_provider_failure: a summary reads it to tell a
            # local 401, 403 or 429 from a rejected request. Chat responses do not use it.
            error.upstream_status_code = status
        raise_detached_error(error)
```

In `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`, the `ChatBadRequestError` branch of `_summary_error_code` becomes:

```python
    if isinstance(exc, ChatBadRequestError):
        upstream_status = getattr(exc, "upstream_status_code", None)
        # Local adapters make every 4xx a ChatBadRequestError, so their 401, 403 and 429 land here.
        if upstream_status in {401, 403}:
            return "authentication"
        if upstream_status == 429:
            return "rate_limit"
        # 408 (OpenRouter timeout) and 498 (Groq flex capacity) are transient 4xx replies.
        if upstream_status in {408, 498}:
            return "provider_failure"
        # The provider refused this request (an unknown model, a prompt over its context);
        # sending it again gets the same answer.
        return "request_rejected"
```

The new branches reach only the local adapters. `build_sanitized_chat_error` never makes a `ChatBadRequestError` for 401, 403 or 429 (it makes `ChatAuthenticationError` and `ChatRateLimitError`), so no remote provider's code changes.

- [ ] **Step 4: run the new tests, then the files they live in**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q $T/LLM_Calls/test_local_http_error_mapping.py -k keeps_the_upstream_status
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q $T/LLM_Calls/test_summarization_adapter.py -k local_provider_reply
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q $T/LLM_Calls/test_local_http_error_mapping.py $T/LLM_Calls/test_summarization_adapter.py $T/LLM_Calls/test_local_streaming_contract.py $T/LLM_Adapters/unit/test_authoritative_local_adapter_snapshot.py
```

Expected: `4 passed`, then `14 passed`, then the four whole files pass with no failures.

- [ ] **Step 5: commit**

```bash
git add tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py $T/LLM_Calls/test_local_http_error_mapping.py $T/LLM_Calls/test_summarization_adapter.py
git commit -F - <<'EOF'
fix(llm): a local provider's 401, 403 and 429 are not a rejected request (TASK-13865)

_raise_sanitized_local_provider_error makes every 4xx a ChatBadRequestError
and dropped the status. Since TASK-13844 a summary reads that class as
request_rejected, so a local server's 401, 403 and 429 (a rate-limiting
proxy, for one) were not retried and read as a refused request.

The error now keeps the status as upstream_status_code, as
ChatProvider._raise_sanitized_provider_failure does. The summary maps
401/403 to authentication and 429 to rate_limit. 408 gets the existing
transient rule, and an unpulled model's 404 stays request_rejected. The
class, status and message are unchanged, so chat responses are too.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 4: the audio jobs doc, the sweep, the follow-up and close-out

**Files:**
- Modify: `Docs/API-related/Audio_Jobs_API.md:49-50`, plus its mirror `Docs/Published/API-related/Audio_Jobs_API.md` (regenerated)
- Backlog: `TASK-13865` (close), two new follow-up tasks

**Interfaces:**
- Consumes: the summary codes from Tasks 1-3.

- [ ] **Step 1: the audio jobs doc**

In `Docs/API-related/Audio_Jobs_API.md`, line 49 (the "Stored at once, without a retry" bullet), replace this text:

```markdown
A provider switched off with `LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER>=0` shows up as `configuration`. `request_rejected` means the provider refused the request (an unknown model, a transcript too long for the model);
```

with:

```markdown
A provider switched off with `LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER>=0`, a provider endpoint the outbound (egress) policy blocks, and a custom OpenAI-compatible provider with no base URL show up as `configuration`. A local provider's (Ollama, llama.cpp, Kobold and the other local servers) 401 or 403 reply shows up as `authentication`. `request_rejected` means the provider refused the request (an unknown model or one a local server has not pulled, a transcript too long for the model);
```

Line 50 (the "Retried" bullet), replace this text:

```markdown
  - Retried: a rate limit (`rate_limit`) and any other provider failure (`provider_failure`, which includes a provider's transient 408 or 498 reply and an analysis reply that starts with `Error:`).
```

with:

```markdown
  - Retried: a rate limit (`rate_limit`, which includes a local provider's 429 reply) and any other provider failure (`provider_failure`, which includes a provider's transient 408 or 498 reply, a provider host that does not resolve, a provider server that cannot be reached (a local server that is down), and an analysis reply that starts with `Error:`).
```

The rest of both bullets is unchanged.

- [ ] **Step 2: refresh the published mirror and run the docs tests**

```bash
bash Helper_Scripts/refresh_docs_published.sh
git status --short Docs/
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q $T/Docs
```

Expected:
- `git status` shows exactly ` M Docs/API-related/Audio_Jobs_API.md` and ` M Docs/Published/API-related/Audio_Jobs_API.md` (measured on the prototype).
- `$T/Docs`: `212 passed` (measured on the prototype). It checks the published mirror.

- [ ] **Step 3: the regression sweep, the quality ratchet, lint and Bandit**

Run every test file that names a changed symbol or path. The list was made with `git grep -l -E` over `tldw_Server_API/tests` for these patterns:
- `custom_openai_adapter|CustomOpenAIAdapter|local_adapters|_LocalAdapterBase|OllamaAdapter|LlamaCppAdapter|KoboldAdapter`
- `SummaryProviderError|_summary_error_code|Summarization_General_Lib|upstream_status_code|EgressPolicyError|explicit base URL`
- `custom-openai-api`

To those the plan adds the chat error-mapping files from TASK-13844's sweep. It leaves out:
- `tests/http_client/*` and `tests/Security/test_egress.py`: they name `EgressPolicyError` only for the transport, which this plan does not touch.
- `tests/AuthNZ_Postgres/*`: they name `custom-openai-api` only as a BYOK provider, and they need a reachable Postgres. Without one, a run of the sweep that included them hung in their fixture until the 300 s pytest timeout ended the session.
- `conftest.py`.

```bash
FILES=$({
  git grep -l -E "custom_openai_adapter|CustomOpenAIAdapter|local_adapters|_LocalAdapterBase|OllamaAdapter|LlamaCppAdapter|KoboldAdapter|SummaryProviderError|_summary_error_code|Summarization_General_Lib|upstream_status_code|EgressPolicyError|explicit base URL" -- $T
  git grep -l -e "custom-openai-api" -- $T
  printf '%s\n' $T/LLM_Calls/test_http_status_extraction_parity.py $T/LLM_Calls/test_llm_streaming_and_security.py \
    $T/LLM_Adapters/integration/test_adapters_chat_endpoint_error_sse_core_providers.py $T/Audio/test_audio_jobs_worker_helper.py \
    $T/Chat/unit/test_error_handling.py $T/Chat/unit/test_chat_service_fallback.py \
    $T/Chat/unit/test_orchestrator_network_exception_routing.py $T/Chat/unit/test_streaming_utils.py \
    $T/Chat/integration/test_chat_endpoint.py $T/Character_Chat_NEW/unit/test_chat_session_error_mapping.py \
    $T/Admin/test_admin_llm_provider_test_runtime.py $T/Admin/test_admin_llm_providers_service_sanitizers.py
} | sort -u | grep -v -E "^$T/(http_client/|Security/test_egress\.py|AuthNZ_Postgres/)|conftest\.py$")
printf '%s\n' $FILES | wc -l
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -rfE $FILES
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q $T/CI/test_test_quality_ratchet.py $T/lint/test_scope_predicate_ratchet.py
```

Expected (measured):
- `wc -l`: `118` files. The grep also picks up `Setup/test_setup_first_chat_completion.py`, through Task 2's new test.
- The sweep, measured with every non-loopback connect and DNS lookup blocked: `7 failed, 3813 passed, 18 skipped, 1 error`.
  - On unchanged dev `e521605198` the same 118 files give `7 failed, 3774 passed, 18 skipped, 1 error`, also with the network blocked.
  - The difference is exactly the 39 new test cases: 13 in Task 1, 8 in Task 2 (the 7 adapter cases are edited, not new), and 18 in Task 3.
  - The 7 failures and the error are the same on both trees, so they are dev's, not this plan's.
  - Six of the failures need DNS (review M6). These are the two `CustomOpenAIAdapter` and `CustomOpenAIAdapter2` cases of `LLM_Adapters/unit/test_adapter_midstream_error_raises.py::test_adapter_midstream_error_raises`, and the four `custom_openai` cases (400, 401, 429, 500) of `LLM_Adapters/unit/test_adapter_stream_error_normalization.py::test_adapter_stream_normalizes_httpx_errors`.
    - They send their request through the real egress check, which resolves `api.openai.com`. With the lookup blocked they fail with `EgressPolicyError('Host could not be resolved')`.
    - With working DNS they pass (measured by the first draft's run on 117 files, `1 failed, 3803 passed, 18 skipped, 1 error`, and by the review). So an executor whose DNS works should see 6 fewer failures and 6 more passes than the blocked counts above; that combined count was not run, because this plan's runs make no outbound lookups.
    - An offline executor sees these six on dev too. Step 4 files the test-hygiene follow-up.
  - The other failure and the error depend on run order, and both pass alone. The failure is `Prompt_Management/test_service_prompts_api.py::test_service_prompt_catalog_returns_exact_metadata_without_prompt_bodies` (`assert 401 == 200` when it runs inside this sweep). The error is `LLM_Calls/test_summarization_input_extraction.py::test_extractor_never_opens_files` (`extract_text_from_input must not open files`, in setup).
  - `AudioJobs/test_audio_worker_smoke.py::test_audio_analyze_error_reply_is_retried_as_a_provider_failure` is timing-sensitive on dev. It failed 1 of 3 runs of the smoke file on unchanged dev with `assert 3.944014012813568 >= 4`, so a single failure of it is not this plan's.
- The ratchets: `14 passed` (measured on the prototype). The quality ratchet (1 test) scans the working tree against `Helper_Scripts/ci/test_quality_baseline.txt`. The scope-predicate ratchet (13 tests) is unaffected: no touched file is in `scope_predicate_baseline.txt`.

If anything else fails, run the same file on unchanged `origin/dev` before blaming the change.

```bash
$PY -m ruff check tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py $T/LLM_Calls/test_summarization_adapter.py $T/AudioJobs/test_audio_worker_smoke.py $T/LLM_Adapters/unit/test_custom_openai_native_http.py $T/LLM_Adapters/unit/test_authoritative_custom_openai_endpoint_snapshot.py $T/Chat/unit/test_chat_service_base_url_override.py $T/LLM_Calls/test_local_http_error_mapping.py $T/Setup/test_setup_first_chat_completion.py
uvx bandit -ll tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py
git diff --check origin/dev
```

Expected (measured on dev and on the prototype):
- ruff: `Found 7 errors`, the same 7 that dev has in these files: 2 × UP035, 2 × UP006, SIM300 and UP028 in `test_custom_openai_native_http.py`, and UP028 in `test_summarization_adapter.py` (its line number moves from 156 to 162). None are new; the Setup test file has none.
- Bandit: `No issues identified.`
- `git diff --check`: no output.

- [ ] **Step 4: file the two follow-ups**

Two gaps this task leaves alone:
- A chat-side gap, measured while planning (see "Chat responses"): a local provider's 401, 403 and 429 answer `/chat/completions` with 400 `Invalid request.` when not streamed, and with 502 `provider_unavailable` when streamed.
- A test-hygiene gap (review M6): six sweep cases resolve `api.openai.com` through the real egress check, on dev too (Step 3).

Find the highest id across `origin/dev` and the open PRs with the AGENTS.md command, and create the tasks with the next two ids. Run Steps 4 and 5 in the same shell, because Step 5 uses `$FOLLOW_UP` and `$FOLLOW_UP_DNS`.

```bash
git fetch origin dev --quiet
MAX_ID=$({ git ls-tree -r --name-only origin/dev backlog/; gh pr list --state open --limit 500 --json files --jq '.[].files[].path'; } \
  | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1)
FOLLOW_UP="TASK-$((MAX_ID + 1))"
echo "$FOLLOW_UP"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Chat: a local provider's 401, 403 or 429 reply answers 400 Invalid request" \
  --id "$FOLLOW_UP" \
  -d "Found while planning TASK-13865. _raise_sanitized_local_provider_error (core/LLM_Calls/providers/local_adapters.py) makes every upstream 4xx a ChatBadRequestError. So /chat/completions answers a local server's 401, 403 or 429 (llama.cpp, Ollama, Kobold and the other local adapters) with 400 {\"detail\": \"Invalid request.\"}, or with 502 provider_unavailable when streamed (measured on dev 2026-10-20). A remote provider's 401 answers provider_authentication_failed, and its 429 is a rate limit. TASK-13865 kept the upstream status on the error as upstream_status_code for the summary codes only, and left chat responses unchanged." \
  --ac "A local provider's 401 or 403 reply gets the chat response a remote provider's authentication failure gets, streamed and not" \
  --ac "A local provider's 429 reply gets the chat response a remote provider's rate limit gets, streamed and not, and an unpulled model (404) keeps its current response" \
  -l llm,errors,chat
FOLLOW_UP_DNS="TASK-$((MAX_ID + 2))"
echo "$FOLLOW_UP_DNS"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Tests: custom OpenAI adapter error tests resolve api.openai.com through the real egress check" \
  --id "$FOLLOW_UP_DNS" \
  -d "Found by the TASK-13865 plan review. The custom_openai cases of tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_midstream_error_raises.py (CustomOpenAIAdapter, CustomOpenAIAdapter2) and test_adapter_stream_error_normalization.py (custom_openai 400, 401, 429, 500) send their request through the real egress check, which resolves api.openai.com. With DNS blocked they fail with EgressPolicyError('Host could not be resolved') on dev; with DNS they pass, after a live lookup from the test suite." \
  --ac "The six cases pass with every non-loopback connect and DNS lookup blocked, and make no live DNS query" \
  -l tests,llm
git add -A backlog/tasks
```

- [ ] **Step 5: close TASK-13865**

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13865 \
  --append-notes "AC1: EgressPolicyError and the transport's NetworkError join _SUMMARY_ADAPTER_EXCEPTIONS as _SUMMARY_PASSTHROUGH_EXCEPTIONS. _summary_error_code maps an egress block as chat_service._map_provider_egress_error does: dns_unresolved -> provider_failure (retried), every other reason -> configuration (not retried). A NetworkError (a kobold server that is down; review M5) falls through to provider_failure, so the audio stage retries it and then stores the transcript with analysis_error; _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS is not widened. No new summary code, so the Evaluations mapping, _RETRYABLE_ANALYSIS_CODES and the docs list need no new entry. raise_on_error=False callers still get either exception itself (two control tests). Readers: the audio worker keeps the transcript; the response-quality and RAG evaluators now fail the request (503 provider_configuration_invalid, or 502 provider_unavailable) instead of scoring 0, and eval_runner records the sample as an error. RED 10 failed (7 summary, 3 worker), GREEN 10 + 5 passed." \
  --append-notes "AC2: custom_openai_adapter.py _resolve_base and _resolve_transport_context raise ChatConfigurationError (same messages) instead of RuntimeError, as the credentialed branch already did; the summary records configuration. /chat/completions is unchanged (it always binds a credential handle: 503 provider_configuration_invalid before and after, streamed and not, measured); perform_chat_api_call without a handle now raises ChatConfigurationError instead of RuntimeError (new chat_service test). So the Workflows LLM, RAG, media, audio-briefing and messaging steps, the FVA analyze function, the web-search aggregation report and the Setup first-chat category now behave as they already do for a credentialed configuration error (review I1's measured caller list is in the plan); a new Setup test pins the first-chat category (configuration_error, was provider_error). The 7 adapter test cases that pinned RuntimeError now expect ChatConfigurationError. RED 15 failed, GREEN 15 passed." \
  --append-notes "AC3: _raise_sanitized_local_provider_error keeps the upstream status as upstream_status_code (class, status 400 and message unchanged, so chat responses are unchanged: 400 Invalid request non-stream, 502 provider_unavailable stream, measured before and after). _summary_error_code maps a ChatBadRequestError with upstream 401/403 to authentication and 429 to rate_limit; 408 now gets the transient rule; an unpulled model's 404 stays request_rejected. RED 12 failed (8 summary through ollama and kobold, 4 mapper), GREEN 4 + 14 passed." \
  --append-notes "Sweep over the 118 test files that name a changed symbol or custom-openai-api, with every non-loopback connect and DNS lookup blocked: 7 failed, 3813 passed, 18 skipped, 1 error (dev e521605198, same files and blocking: 7 failed, 3774 passed, 18 skipped, 1 error; the same 7 failures and error are dev's: 6 custom_openai adapter cases that resolve api.openai.com, plus test_service_prompts_api and test_summarization_input_extraction, which depend on run order; the 39 new cases are the difference). tests/Docs 212 passed; quality and scope-predicate ratchets passed. Ruff: only the 7 findings dev has in these files. Bandit clean. Docs mirror refreshed. Follow-ups: $FOLLOW_UP (the chat response for a local 401, 403 or 429) and $FOLLOW_UP_DNS (the custom OpenAI adapter tests that resolve api.openai.com)." \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A summary whose provider endpoint the egress policy blocks is now configuration (a host that does not resolve is provider_failure and retried), a local server that is down is provider_failure (retried, then the transcript is kept) instead of escaping untyped, a custom OpenAI-compatible provider with no base URL is configuration, and a local provider's 401/403 is authentication and its 429 rate_limit (an unpulled model stays request_rejected). No new summary code. raise_on_error=False callers and /chat/completions responses are unchanged; an in-process perform_chat_api_call without a credential handle now gets ChatConfigurationError for a missing custom base URL." \
  -s Done
git add -A backlog/tasks
```

The counts in the notes are the measured ones. If Step 3 printed different counts, write those instead.

- [ ] **Step 6: commit**

```bash
git add Docs/API-related/Audio_Jobs_API.md Docs/Published/API-related/Audio_Jobs_API.md
git add -A backlog/tasks
git commit -F - <<'EOF'
docs(audio-jobs): egress-blocked, unconfigured custom and local 4xx analysis codes (TASK-13865)

The audio jobs doc now says that an endpoint the egress policy blocks and a
custom provider with no base URL are stored as configuration, a local 401
or 403 as authentication, and an unpulled model as request_rejected, and
that a local 429, a host that does not resolve and a local server that is
down are retried. Closes TASK-13865 and files two follow-ups: the chat
response to a local 401, 403 or 429, and the custom OpenAI adapter tests
that resolve api.openai.com.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

## Out of scope

- **The chat response to a local 401, 403 or 429.** It stays 400 `Invalid request.` (502 when streamed). Typing those replies at the adapter would change chat responses. That is the follow-up filed in Task 4, Step 4.
- **A new summary code for egress.** `configuration` already says "fix the setup, do not retry"; see "Decisions for the owner".
- **In-stream SSE errors in a streamed summary**, including every local stream failure. They stay `provider_failure`, as in TASK-13844, and no `raise_on_error` caller streams.
- **`EgressPolicyError` raised before the adapter call** (credential binding, registry lookup). None of those steps makes a request today, so none raises it.
