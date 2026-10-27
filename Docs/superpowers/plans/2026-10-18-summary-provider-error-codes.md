# Summary Provider Error Codes Implementation Plan (TASK-13844)

Independent review and dry run (2026-10-18): 0 Blocker, 2 Important, 5 Minor; all folded in.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A summary call that the provider refuses (an unknown model, a prompt over the model's context), an adapter switched off by configuration, and a spent billing quota each get a `SummaryProviderError` code that the audio analysis stage does not retry. A spent quota is told apart from a rate limit.

**Architecture:**
- `_summary_error_code` (`core/LLM_Calls/Summarization_General_Lib.py`) gains two mappings:
  - `ChatBadRequestError` becomes a new code, `request_rejected`, except for the transient 4xx replies 408 and 498, which stay `provider_failure` and are retried (Review I1).
  - A `RuntimeError` from an adapter whose `LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER>` switch is off becomes the existing `configuration` code.
- `ChatRateLimitError` gains one bool, `quota_exhausted`.
  - `build_sanitized_chat_error` sets it from the failed upstream reply by comparing the body's `error.type` and `error.code` with `{"insufficient_quota"}`. No upstream text is kept.
  - The summary maps it to a new code, `provider_quota_exhausted`.
- Neither new code is in the audio worker's `_RETRYABLE_ANALYSIS_CODES`, so both fail the stage at once. Every other reader either ignores the code or allowlists it, and keeps its current response.

**Tech Stack:** Python 3.12, httpx (`MockTransport` in tests), requests, pytest.

**Spec:** `backlog/tasks/task-13844 - Summary-provider-errors-a-provider-rejecting-the-request-reads-as-a-transient-failure.md` on `dev`.

## What dev does today

Measured on origin/dev `e1f7770953` (2026-10-18) in a scratch worktree. The probe called `analyze(..., raise_on_error=True)` through the real `OpenAIAdapter`, with an httpx `MockTransport` returning each reply. The "after" column is the same probe run on a prototype of this plan.

| Upstream reply | Code on dev | Code after this plan |
|---|---|---|
| 404, `model_not_found` | `provider_failure` | `request_rejected` |
| 400, `context_length_exceeded` | `provider_failure` | `request_rejected` |
| 402 | `provider_failure` | `request_rejected` |
| 413 | `provider_failure` | `request_rejected` |
| 408 (OpenRouter timeout; Groq's 498 flex capacity is the same case) | `provider_failure` | `provider_failure` (Review I1) |
| 429, `rate_limit_exceeded` | `rate_limit` | `rate_limit` |
| 429, `insufficient_quota` | `rate_limit` | `provider_quota_exhausted` |
| 500 | `provider_failure` | `provider_failure` |
| `LLM_ADAPTERS_NATIVE_HTTP_OPENAI=0` | `provider_failure` | `configuration` |

How each of these happens on dev:
- `build_sanitized_chat_error` (`error_utils.py:101-130`) turns every 4xx except 401, 403 and 429 into `ChatBadRequestError`. `_summary_error_code` (`Summarization_General_Lib.py:135-148`) has no branch for that class, so it falls through to `provider_failure`.
- The kill switch raises a bare `RuntimeError("<Adapter> native HTTP disabled by configuration")`. There are 12 such raises in 6 adapters: openai, anthropic, bedrock, custom_openai, groq and openrouter. `_summarize_via_adapter` catches it as a non-critical exception, and it too becomes `provider_failure`.
- The other configuration error the task names, `ValueError("Configured endpoint scope mismatch")` (`openai_adapter.py:315`), cannot reach a summary:
  - It is raised only when the request carries a `call_policy` with a `required_endpoint_scope` (`openai_adapter.py:275-283`).
  - The summary request (`Summarization_General_Lib.py:302-314`) never has one. Only Notes_Graph, Flashcards, prompt improvement and character chat send one.
- The upstream body is still there for the error mapping:
  - In `ChatProvider._raise_sanitized_provider_failure` and `ChatProvider.normalize_error` (`providers/base.py:92-115`, `163-200`), `exc.response` is the read httpx or requests response.
  - In `raise_chat_error_from_http` (`error_utils.py:286-333`), the body has already been parsed for log metadata.
  - That covers every adapter that maps HTTP errors through the shared builder. Eleven use `_raise_sanitized_provider_failure` or `normalize_error`: anthropic, bedrock, custom_openai, deepseek, google, groq, huggingface, mistral, openai, openrouter and qwen. Cohere, moonshot and zai use `raise_chat_error_from_http`.
  - The local adapters have their own mapper (`local_adapters.py:193`), which turns every 4xx into `ChatBadRequestError`.
- The audio worker retries only `rate_limit` and `provider_failure` (`audio_jobs_worker.py:23-25`), so all of these replies are retried today. Only the `rate_limit_exceeded` 429, the 500 and the transient 408 and 498 should be. Task 1's worker test shows it on dev: a rejected request ends with `retry_count` 2.

## Global Constraints

- Every PR's final sweep includes `tldw_Server_API/tests/CI/test_test_quality_ratchet.py`: since #3243, dev enforces it, so new tests must not be status-only or tautological.
- Branch `fix/summary-provider-error-codes` from `origin/dev`. One PR, to `dev`.
- The two new summary codes are spelled exactly `request_rejected` and `provider_quota_exhausted`.
- No upstream text in any error. The only new upstream-derived value is one bool, from comparing `error.type` and `error.code` with the allowlist `{"insufficient_quota"}`.
- No adapter raise site changes. The 12 kill-switch raises stay `RuntimeError`; `tests/LLM_Adapters/unit/test_native_http_kill_switch_reachable.py` pins them.
- Chat endpoint responses do not change:
  - `ChatRateLimitError` keeps its class, its status (429) and its message.
  - `chat_service._bound_provider_invocation_exception` (`chat_service.py:3548-3552`) rebuilds a fresh `ChatRateLimitError` without the flag.
  - Nothing outside the summary reads `quota_exhausted` (`git grep quota_exhausted` is empty on dev).
  - `streaming_utils._PROVIDER_STREAM_ERROR_TYPE_CODES` keys on the class name, which is unchanged. This is why the plan adds an attribute, not a subclass.
- `analyze(raise_on_error=False)` keeps its legacy `Error:` strings. The new mappings run only where the raise path builds a `SummaryProviderError`.
- `_RETRYABLE_ANALYSIS_CODES` stays `{"rate_limit", "provider_failure"}`.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`.
- Backlog changes go only through backlog-py, with `--append-notes` and the explicit id `TASK-13844`. Stage them with `git add -A backlog/tasks`.

Shell setup used by every command below (run from the repository root):

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
```

## Decisions for the owner

None. Two follow-ups are possible without a decision now:
- The Evaluations endpoints could give the new codes their own responses. Today both get 502 `provider_unavailable`, like `rate_limit` and `provider_failure` (see the reader audit). AC3 asks only that the readers be checked, so this plan keeps the responses and pins them with a test.
- A 402 could map to `provider_quota_exhausted`. DeepSeek and OpenRouter use 402 for "insufficient balance". This plan maps it to `request_rejected`, which is also not retried; see Out of scope.

## ADR assessment

No ADR is needed. The change adds two values to an internal bounded code set and one bool to an existing exception. It makes no durable rule for module boundaries, public API shape, persistence or provider integration.
- ADR-025 (the adapter registry is the shared error-mapping point) is followed: the quota signal is decided in the shared `build_sanitized_chat_error`, not in an adapter.
- ADR-064 (usage quotas) concerns tldw's own per-user quotas, whose code is `llm_quota_exceeded`. The new name, `provider_quota_exhausted`, keeps the provider account's billing quota apart from it.

## Interaction with the audio analysis plan

`plan_audio_analysis.md` (TASK-13845/13846/13847) changes what the worker does with a code that is not retried. Under it, the transcript goes on to `audio_store` with `analysis_error.code` instead of the job failing. It does not change `_RETRYABLE_ANALYSIS_CODES`. The two plans agree on what matters: `request_rejected` and `provider_quota_exhausted` are not retried, and the code is recorded.

They can land in either order:
- The worker test in Task 1 reads the code from wherever it was recorded: the job's `error_code` today, and the stored payload's `analysis_error.code` after TASK-13847.
- Both plans edit the same paragraph of `Docs/API-related/Audio_Jobs_API.md`. Whichever lands second keeps the two new codes, and the sentence that a provider switched off with `LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER>=0` shows up as `configuration`, in the list of failures that are not retried (Task 3, Step 3). The audio plan carries the same instruction for the case where it lands second (Review I2).
- Both plans add tests after `test_audio_analyze_provider_failure_still_retries` in `test_audio_worker_smoke.py`. A rebase conflict there is textual; keep both sides.

## Reader audit (AC3)

Every reader of `SummaryProviderError.code` on dev was measured with `git grep -n "SummaryProviderError"`, plus the generic `getattr(exc, "code", ...)` readers in RAG. The task's description says about 12 RAG and Evaluations sites record the code as metadata. On dev, none do: they record the constant `provider_unavailable`, or allowlist the code.

| Reader | What it does with the code | `request_rejected` | `provider_quota_exhausted` | kill switch (`configuration`, was `provider_failure`) |
|---|---|---|---|---|
| `services/audio_jobs_worker.py:395-398` | `retryable = code in _RETRYABLE_ANALYSIS_CODES`; the code becomes the job's `error_code` | fails at once (was retried) | fails at once (was retried as `rate_limit`) | fails at once (was retried) |
| `api/v1/endpoints/evaluations/evaluations_unified.py:241-293` (`_eval_provider_http_exception`, whose calls are at 1180, 1380 and 1579; the `except` lines are 1178, 1378 and 1574) (Review M5) | maps the code to an HTTP response | 502 `provider_unavailable` (unchanged) | 502 `provider_unavailable` (unchanged) | 503 `provider_configuration_invalid` (was 502 `provider_unavailable`); the same response as every other configuration error |
| `core/Evaluations/rag_evaluator.py:363, 385, 424`; `core/Evaluations/response_quality_evaluator.py:237, 302, 364, 429, 498, 561` | re-raise; ignore the code | no change | no change | no change |
| `core/RAG/rag_service/agentic_chunker.py:819`, `document_grader.py:320`, `post_generation_verifier.py:485, 725`, `quality_graders.py:322, 626`, `unified_pipeline.py:5268, 7296, 8091, 9108` | record the constant `provider_unavailable` (`exc.code` is read only for `ByokResolutionError`) | no change | no change | no change |
| `core/RAG/rag_service/hyde.py:299` (`_record_embedding_degraded`, reached through the `except Exception` at 161) | maps `configuration` to `provider_configuration_invalid`; anything outside its allowlist becomes `provider_unavailable` | `provider_unavailable` (unchanged) | `provider_unavailable` (unchanged) | `provider_configuration_invalid` (was `provider_unavailable`) |
| `_bounded_provider_failure_code` in `agentic_chunker.py:73`, `agentic_execution.py:71`, `unified_pipeline.py:97` | allowlist without the summary codes; anything else becomes `provider_unavailable` | no change | no change | no change |
| `knowledge_strips.py:491` | `except Exception`: heuristic fallback | no change | no change | no change |
| `services/outputs_service.py` (open PR #3267, TASK-13642: `_llm_reply` calls `analyze(raise_on_error=True)`) | `except Exception`: logs `str(exc)`, which names the code, and returns `""` | logged | logged | logged |

Only two readers change their output, both for the kill switch, and both for the better: the Evaluations response and HyDE's stage metadata now say "configuration". Task 3 pins the Evaluations mapping for every summary code.

## Review Focus

These are the input classes and failure modes that the tests do not exercise:

1. **A streamed 429 `insufficient_quota` over a real network.**
   - httpx raises in `raise_for_status()` before the stream is read, and the `with client.stream(...)` block has closed the response by the time the adapter maps it. So `upstream_quota_exhausted` gets `ResponseNotRead` and returns False, and the reply stays `rate_limit`.
   - The `unread-body` test row pins the helper's side of this. A `MockTransport` reply is already in memory, so an end-to-end stream test cannot show it.
   - No `raise_on_error` summary caller streams. `rag_evaluator.py:482` passes ClaimsEngine's `streaming=False` through.
2. **Transient 4xx replies (Review I1).** `build_sanitized_chat_error` makes every 4xx but 401, 403 and 429 a `ChatBadRequestError`. Two supported providers document a transient one: OpenRouter's 408 "Your request timed out" and Groq's 498 "Flex Tier Capacity Exceeded ... try again later".
   - `_summary_error_code` reads `upstream_status_code`, which `_raise_sanitized_provider_failure` sets (`providers/base.py:111-112`), and keeps 408 and 498 as `provider_failure`, so they are still retried.
   - The `timed-out` row in the OpenAI-adapter test is a control: it passes on dev and stays green.
   - Another provider's transient 4xx that is not 408 or 498 would read `request_rejected`; none is known.
   - Whether OpenRouter's 408 should be retried at all is a product question (OpenRouter suggests lowering `max_tokens` or changing model). This plan keeps today's behaviour for it.
3. **Local providers.** `_raise_sanitized_local_provider_error` (`local_adapters.py:193-227`) maps every 4xx, including 401, 403 and 429, to `ChatBadRequestError`, without `upstream_status_code`. A local server behind a rate-limiting proxy is no longer retried. An unpulled Ollama model (404) is the case this task wants. The follow-up task in Task 3, Step 6 tracks this (Review M1).
4. **Other providers' spent-quota replies.** Gemini's 429 `RESOURCE_EXHAUSTED` and Moonshot's quota type still read as `rate_limit` and are retried. DeepSeek's and OpenRouter's 402 read as `request_rejected` and are not retried.
5. **A very large 429 body from a configured custom endpoint.**
   - `_raise_sanitized_provider_failure` and `normalize_error` did not parse the body before. Now `upstream_quota_exhausted` JSON-parses it on every 429.
   - The non-streamed body is already in memory, and a streamed one is never read (item 1), so the cost is CPU only, and only for a 429.
   - `raise_chat_error_from_http` already parsed every error body on dev.

---

### Task 1: a request the provider rejects, and a switched-off adapter, are not retried (AC1)

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py:28-33` (import), `:111-122` (`_SUMMARY_ERROR_CODES`), `:135-148` (`_summary_error_code`), and the two `raise SummaryProviderError(code=_summary_error_code(exc), ...)` lines (pass the adapter; Review M5)
- Create: `Docs/superpowers/plans/2026-10-18-summary-provider-error-codes.md` (this plan, committed with Task 1; Review M2)
- Test: `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py` (imports at 5-21, taxonomy parametrization at 376-385, new tests after line 413)
- Test: `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py` (new helper and test after line 959)

**Interfaces:**
- Produces: summary code `"request_rejected"`; `_summary_error_code(exc: BaseException, adapter: Any = None) -> str`; `_native_http_disabled(adapter: Any) -> bool`; test helpers `_openai_upstream_replies(status: int, body: dict[str, Any])` (in `test_summarization_adapter.py`) and `_analysis_failure_code(jm, job_id: int) -> str | None` (in `test_audio_worker_smoke.py`).

- [ ] **Step 1: branch and mark the task in progress**

```bash
git fetch origin dev --quiet
git switch -c fix/summary-provider-error-codes origin/dev
mkdir -p Docs/superpowers/plans
cp /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad/plan_summary_error_codes.md Docs/superpowers/plans/2026-10-18-summary-provider-error-codes.md
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13844 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-10-18-summary-provider-error-codes.md (three tasks: request_rejected and the kill switch; provider_quota_exhausted; the reader audit, docs and close-out). ADR assessment: none needed (two codes added to an internal bounded set and one bool on ChatRateLimitError; ADR-025 followed, the quota signal is decided in the shared build_sanitized_chat_error)."
```

- [ ] **Step 2: write the failing tests**

In `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py`, import httpx and `ChatBadRequestError`:

```python
import httpx
import pytest

import tldw_Server_API.app.core.LLM_Calls.providers.moonshot_adapter as moonshot_mod
import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
```

Add one row to the end of `test_chat_error_taxonomy_is_preserved_without_upstream_details`'s parametrization:

```python
        (ChatProviderError(message=SECRET, status_code=429, provider="openai"), "rate_limit"),
        (ChatBadRequestError(message=SECRET, provider="openai"), "request_rejected"),
    ],
)
def test_chat_error_taxonomy_is_preserved_without_upstream_details(
```

Insert these tests after `test_chat_error_taxonomy_is_preserved_without_upstream_details`. They go through the real `OpenAIAdapter`, so the whole chain is under test: the HTTP status, `build_sanitized_chat_error`, then the summary code.

```python
def _openai_upstream_replies(status: int, body: dict[str, Any]):
    """An OpenAI adapter HTTP client factory whose every request gets ``status`` and JSON ``body``."""
    transport = httpx.MockTransport(lambda _request: httpx.Response(status, json=body))
    return lambda *_args, **_kwargs: httpx.Client(transport=transport)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "error", "expected_code"),
    [
        (404, {"type": "invalid_request_error", "code": "model_not_found"}, "request_rejected"),
        (400, {"type": "invalid_request_error", "code": "context_length_exceeded"}, "request_rejected"),
        (413, {"type": "invalid_request_error"}, "request_rejected"),
        (408, {"type": "timeout"}, "provider_failure"),
        (429, {"type": "requests", "code": "rate_limit_exceeded"}, "rate_limit"),
        (500, {"type": "server_error"}, "provider_failure"),
    ],
    ids=["unknown-model", "context-overflow", "too-large", "timed-out", "rate-limit", "server-error"],
)
def test_upstream_reply_reaches_the_summary_code_through_the_openai_adapter(
    monkeypatch, status, error, expected_code
):
    monkeypatch.setattr(
        openai_mod, "http_client_factory", _openai_upstream_replies(status, {"error": {"message": SECRET, **error}})
    )

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        sgl.analyze("openai", "hello", None, api_key="key", model_override="gpt-test", raise_on_error=True)

    assert exc_info.value.code == expected_code
    assert SECRET not in str(exc_info.value)
    assert SECRET not in repr(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True], ids=["chat", "stream"])
def test_a_disabled_native_http_adapter_is_a_configuration_error(monkeypatch, streaming):
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENAI", "0")

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        result = sgl.analyze(
            "openai", "hello", None, api_key="key", model_override="gpt-test", streaming=streaming, raise_on_error=True
        )
        if streaming:
            list(result)

    assert exc_info.value.code == "configuration"
```

The `timed-out` row (Review I1) is a control for the transient 4xx replies: the OpenAI adapter also maps through `_raise_sanitized_provider_failure`, so the error carries `upstream_status_code` 408. It passes on dev and must stay green.

The control for the kill switch already exists: `test_adapter_error_raises_typed_error_without_upstream_body` has a fake adapter with no `_use_native_http` raise `RuntimeError`, and it must stay `provider_failure`.

In `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py`, insert after `test_audio_analyze_provider_failure_still_retries` (it ends at line 959), before the `# --- the audio_convert -> audio_transcribe hand-off (TASK-13848) ---` comment:

```python
def _analysis_failure_code(jm, job_id: int) -> str | None:
    """Where the worker recorded an analysis failure: the job's error_code, or, once a failed
    analysis keeps the transcript (TASK-13847), the stored payload's analysis_error.code."""
    job = jm.get_job(job_id) or {}
    if job.get("error_code"):
        return job["error_code"]
    stored = jm.list_jobs(domain="audio", job_type="audio_store", owner_user_id="7")
    return (stored[0]["payload"].get("analysis_error") or {}).get("code") if stored else None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("make_error", "code"),
    [(lambda errors: errors.ChatBadRequestError(provider="openai"), "request_rejected")],
    ids=["rejected-request"],
)
async def test_audio_analyze_failure_a_retry_cannot_fix_is_not_retried(analyze_worker, make_error, code):
    from tldw_Server_API.app.core import exceptions

    worker, jm, adapter = analyze_worker
    adapter.error = make_error(exceptions)
    job_id = _analyze_job(jm, "hello transcript")
    await _run_worker_until(worker, lambda: _analysis_failure_code(jm, job_id) is not None)
    job = jm.get_job(job_id) or {}
    assert int(job.get("retry_count") or 0) == 0
    assert len(adapter.requests) == 1
    assert _analysis_failure_code(jm, job_id) == code
```

- [ ] **Step 3: run them and watch them fail**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py -k "taxonomy or upstream_reply or disabled_native"
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k "retry_cannot_fix"
```

Expected (measured on dev):
- First command: `6 failed, 9 passed` (Review I1: was `8 passed`).
  - The taxonomy row and the `unknown-model`, `context-overflow` and `too-large` rows fail with `AssertionError: assert 'provider_failure' == 'request_rejected'`.
  - Both kill-switch tests fail with `assert 'provider_failure' == 'configuration'`.
  - The `timed-out`, `rate-limit` and `server-error` controls pass.
- Second command: `1 failed`, with `assert 2 == 0` on `retry_count` in the measured run. The job was retried; the count is 1 or 2, depending on how far the retries got before the worker stopped.

- [ ] **Step 4: implement**

In `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`, import `ChatBadRequestError`:

```python
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatRateLimitError,
)
```

Add the code to the bounded set:

```python
_SUMMARY_ERROR_CODES = frozenset(
    {
        "invalid_provider",
        "adapter_unavailable",
        "missing_model",
        "missing_credentials",
        "authentication",
        "configuration",
        "rate_limit",
        "request_rejected",
        "provider_failure",
    }
)
```

Replace `_summary_error_code` with:

```python
def _native_http_disabled(adapter: Any) -> bool:
    """Whether the adapter's LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER> switch is off.

    An adapter with the switch off raises a bare RuntimeError on every call, before any request.
    """
    # ponytail: reads the adapter's own switch instead of retyping its 12 raise sites; a typed
    # error there is the upgrade if other untyped adapter configuration errors need a code.
    switch = getattr(adapter, "_use_native_http", None)
    return callable(switch) and switch() is False


def _summary_error_code(exc: BaseException, adapter: Any = None) -> str:
    """Map chat adapter failures to a bounded, detail-free summary taxonomy."""
    if isinstance(exc, ChatAuthenticationError):
        return "authentication"
    if isinstance(exc, ChatConfigurationError):
        return "configuration"
    if isinstance(exc, ChatRateLimitError):
        return "rate_limit"
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

In `_summarize_via_adapter`, pass the adapter in both raise paths (Review M5: find them by text, not by line number, because the earlier edits in this step move the lines). The two lines differ only in indentation. The stream generator's closure already sees `adapter`.

```python
                    raise SummaryProviderError(code=_summary_error_code(exc, adapter), provider=provider) from None
```

(the deeper-indented one, in `stream_generator`) and

```python
            raise SummaryProviderError(code=_summary_error_code(exc, adapter), provider=provider) from None
```

(the other, in the non-streaming path).

- [ ] **Step 5: run the new tests, then both whole files**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py -k "taxonomy or upstream_reply or disabled_native"
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k "retry_cannot_fix"
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
```

Expected: `15 passed` (Review I1: was `14`), then `1 passed`, then the two whole files pass. The only non-passing results are the skips the smoke file already has on dev (`test_audio_worker_pipeline_smoke_skip_if_missing`).

- [ ] **Step 6: commit**

```bash
git add Docs/superpowers/plans/2026-10-18-summary-provider-error-codes.md tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
git add -A backlog/tasks
git commit -F - <<'EOF'
fix(summarization): a request the provider rejects, or a switched-off adapter, is not a transient failure (TASK-13844)

_summary_error_code had no branch for ChatBadRequestError, which
build_sanitized_chat_error makes of every 4xx but 401, 403 and 429, so an
unknown model or a prompt over the model's context read as provider_failure and
the audio analysis stage retried it. It is now request_rejected, which no
reader retries. The transient 4xx replies (OpenRouter's 408, Groq's 498) stay
provider_failure, keyed on the error's upstream_status_code.

An adapter whose LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER> switch is off raises a bare
RuntimeError on every call; the summary now reads that switch and records
configuration instead of provider_failure. The adapters' raise sites are
unchanged, so chat responses are too.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 2: a spent billing quota is not a rate limit (AC2)

**Files:**
- Modify: `tldw_Server_API/app/core/exceptions.py:925-933` (`ChatRateLimitError`)
- Modify: `tldw_Server_API/app/core/LLM_Calls/error_utils.py:101-130` (new `upstream_quota_exhausted`; `build_sanitized_chat_error` gains `upstream`), `:324-333` (`raise_chat_error_from_http` passes it)
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/base.py:110` (`_raise_sanitized_provider_failure`) and `:192` (`normalize_error`)
- Modify: `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py` (`_SUMMARY_ERROR_CODES`, the `ChatRateLimitError` branch)
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_error_sanitization.py` (imports at 13-25, new tests before line 156)
- Test: `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py`, `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py` (one row each in Task 1's tests)

**Interfaces:**
- Consumes: Task 1's `_openai_upstream_replies` parametrization and `test_audio_analyze_failure_a_retry_cannot_fix_is_not_retried`.
- Produces: `ChatRateLimitError(message=..., provider=None, quota_exhausted: bool = False)` with attribute `quota_exhausted: bool`; `upstream_quota_exhausted(exc: BaseException | None) -> bool`; `build_sanitized_chat_error(..., upstream: BaseException | None = None)`; summary code `"provider_quota_exhausted"`.

- [ ] **Step 1: write the failing tests**

In `tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_error_sanitization.py`, extend the imports:

```python
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.LLM_Calls.error_utils import (
    build_sanitized_chat_error,
    raise_chat_error_from_http,
)
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
```

Insert before `test_base_generic_error_normalization_never_reflects_exception_text`:

```python
_SPENT_QUOTA_BODY = {
    "error": {"message": "private-quota-body", "type": "insufficient_quota", "code": "insufficient_quota"}
}


def _httpx_429(content: bytes, *, read: bool = True) -> httpx.HTTPStatusError:
    """A 429 as httpx raises it; ``read=False`` leaves the body unread, as a streamed reply does."""
    request = httpx.Request("POST", "https://api.openai.example/v1/chat/completions")
    if read:
        response = httpx.Response(429, request=request, content=content)
    else:
        response = httpx.Response(429, request=request, stream=httpx.ByteStream(content))
    return httpx.HTTPStatusError("private-exception", request=request, response=response)


@pytest.mark.parametrize(
    ("content", "read", "expected"),
    [
        (json.dumps(_SPENT_QUOTA_BODY).encode(), True, True),
        (json.dumps({"error": {"type": "tokens", "code": "rate_limit_exceeded"}}).encode(), True, False),
        (json.dumps({"error": {"code": ["insufficient_quota"]}}).encode(), True, False),
        (json.dumps({"error": "insufficient_quota"}).encode(), True, False),
        (b"insufficient_quota", True, False),
        (json.dumps(_SPENT_QUOTA_BODY).encode(), False, False),
    ],
    ids=["insufficient-quota", "rate-limit", "list-code", "string-error", "not-json", "unread-body"],
)
def test_a_rate_limit_says_whether_the_billing_quota_is_spent(
    content: bytes,
    read: bool,
    expected: bool,
) -> None:
    normalized = _DummyProvider().normalize_error(_httpx_429(content, read=read))

    assert isinstance(normalized, ChatRateLimitError)
    assert normalized.status_code == 429
    assert normalized.quota_exhausted is expected
    assert "private-quota-body" not in repr(vars(normalized))


def test_requests_adapters_keep_the_spent_quota_signal() -> None:
    response = requests.Response()
    response.status_code = 429
    response._content = json.dumps(_SPENT_QUOTA_BODY).encode()
    error = requests.HTTPError("private-exception", response=response)

    with pytest.raises(ChatRateLimitError) as exc_info:
        raise_chat_error_from_http("moonshot", error)

    assert exc_info.value.quota_exhausted is True


def test_a_rate_limit_without_an_upstream_reply_is_not_a_spent_quota() -> None:
    error = build_sanitized_chat_error("openai", status_code=429)

    assert isinstance(error, ChatRateLimitError)
    assert error.quota_exhausted is False
```

In `test_summarization_adapter.py`, add a `spent-quota` row to `test_upstream_reply_reaches_the_summary_code_through_the_openai_adapter`. This row goes through `_raise_sanitized_provider_failure`, the third caller.

```python
        (408, {"type": "timeout"}, "provider_failure"),
        (429, {"type": "requests", "code": "rate_limit_exceeded"}, "rate_limit"),
        (429, {"type": "insufficient_quota", "code": "insufficient_quota"}, "provider_quota_exhausted"),
        (500, {"type": "server_error"}, "provider_failure"),
    ],
    ids=["unknown-model", "context-overflow", "too-large", "timed-out", "rate-limit", "spent-quota", "server-error"],
```

In `test_audio_worker_smoke.py`, give `test_audio_analyze_failure_a_retry_cannot_fix_is_not_retried` its second row. The lambda builds the error at run time, so a missing keyword fails one test, not the module's collection.

```python
    [
        (lambda errors: errors.ChatBadRequestError(provider="openai"), "request_rejected"),
        (lambda errors: errors.ChatRateLimitError(provider="openai", quota_exhausted=True), "provider_quota_exhausted"),
    ],
    ids=["rejected-request", "spent-quota"],
```

- [ ] **Step 2: run them and watch them fail**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_error_sanitization.py -k quota
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py -k upstream_reply
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k retry_cannot_fix
```

Expected (measured on Task 1's tree):
- `8 failed`, each with `AttributeError: 'ChatRateLimitError' object has no attribute 'quota_exhausted'`.
- `1 failed, 6 passed` (Review I1: was `5 passed`), with `assert 'rate_limit' == 'provider_quota_exhausted'`.
- `1 failed, 1 passed`, with `TypeError: ChatRateLimitError.__init__() got an unexpected keyword argument 'quota_exhausted'`.

- [ ] **Step 3: implement**

`tldw_Server_API/app/core/exceptions.py`, replace `ChatRateLimitError`:

```python
class ChatRateLimitError(ChatAPIError):
    """Raised when a chat provider rate-limits a request.

    ``quota_exhausted`` is True when the provider said the account's billing quota is spent
    (OpenAI ``insufficient_quota``), so waiting will not help. It is a bool, never upstream text.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded with the chat provider.",
        provider: str | None = None,
        quota_exhausted: bool = False,
    ) -> None:
        self.quota_exhausted = quota_exhausted is True
        super().__init__(message, status_code=429, provider=provider)
```

`tldw_Server_API/app/core/LLM_Calls/error_utils.py`: add the helper just above `build_sanitized_chat_error`, then give the builder the `upstream` keyword. `_ERROR_UTILS_NONCRITICAL_EXCEPTIONS` (line 33) already covers what `.json()` can raise here:
- `AttributeError` when there is no response;
- `ResponseNotRead` and `StreamClosed`, which are `RuntimeError`s;
- `JSONDecodeError`, which is a `ValueError`.

```python
# Upstream error codes that mean the account's billing quota is spent, not a rate limit.
_SPENT_QUOTA_ERROR_CODES = frozenset({"insufficient_quota"})


def upstream_quota_exhausted(exc: BaseException | None) -> bool:
    """Whether a failed upstream reply says the billing quota is spent (OpenAI ``insufficient_quota``).

    Only the body's ``error.type`` and ``error.code`` are compared with a fixed allowlist, and only
    a bool comes back, so no upstream text is kept. No reply, an unread (streamed) body, or a body
    that is not JSON gives False.
    """

    try:
        body = exc.response.json()
    except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
        return False
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return False
    return any(
        isinstance(error.get(key), str) and error.get(key) in _SPENT_QUOTA_ERROR_CODES
        for key in ("type", "code")
    )


def build_sanitized_chat_error(
    provider: str,
    *,
    status_code: int | None = None,
    auth_statuses: tuple[int, ...] = (401, 403),
    rate_limit_statuses: tuple[int, ...] = (429,),
    bad_request_statuses: tuple[int, ...] = (400, 404, 422),
    treat_other_4xx_as_bad_request: bool = True,
    upstream: BaseException | None = None,
) -> ChatAPIError:
    """Build one typed public error without reflecting untrusted upstream text.

    ``upstream`` is the failed upstream call's exception. It is read only to tell a spent
    billing quota from a rate limit (``upstream_quota_exhausted``).
    """

    safe_provider = _bounded_log_label(provider)
    if status_code in auth_statuses:
        return ChatAuthenticationError(
            provider=safe_provider,
            status_code=status_code or 401,
        )
    if status_code in rate_limit_statuses:
        return ChatRateLimitError(
            provider=safe_provider,
            quota_exhausted=upstream_quota_exhausted(upstream),
        )
```

The rest of `build_sanitized_chat_error` is unchanged. At its end, `raise_chat_error_from_http` passes the exception:

```python
    raise_detached_error(
        build_sanitized_chat_error(
            provider,
            status_code=status_code,
            auth_statuses=auth_statuses,
            rate_limit_statuses=rate_limit_statuses,
            bad_request_statuses=bad_request_statuses,
            treat_other_4xx_as_bad_request=treat_other_4xx_as_bad_request,
            upstream=exc,
        )
    )
```

The requests-based adapters (cohere, moonshot, zai) stream with `requests`, but `raise_chat_error_from_http` already calls `response.json()` for its log metadata at line 302. So the second `.json()` reads the cached body and does no new network read.

`tldw_Server_API/app/core/LLM_Calls/providers/base.py`, line 110 in `_raise_sanitized_provider_failure`:

```python
        error = build_sanitized_chat_error(self.name, status_code=status, upstream=exc)
```

and line 192 in `normalize_error`:

```python
            return build_sanitized_chat_error(self.name, status_code=status, upstream=exc)
```

The detached error keeps only the bool; `raise_detached_error` drops `exc` as cause and context.

`Summarization_General_Lib.py`: add `"provider_quota_exhausted",` to `_SUMMARY_ERROR_CODES`, after `"rate_limit",`. Then change the rate-limit branch of `_summary_error_code`:

```python
    if isinstance(exc, ChatRateLimitError):
        return "provider_quota_exhausted" if exc.quota_exhausted is True else "rate_limit"
```

- [ ] **Step 4: run the new tests, then the files they live in**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_error_sanitization.py tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k retry_cannot_fix
```

Expected: `114 passed` (both files; Review I1: was `113`, measured on the prototype before the `timed-out` row), then `2 passed`.

- [ ] **Step 5: commit**

```bash
git add tldw_Server_API/app/core/exceptions.py tldw_Server_API/app/core/LLM_Calls/error_utils.py tldw_Server_API/app/core/LLM_Calls/providers/base.py tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_error_sanitization.py tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
git commit -F - <<'EOF'
fix(llm): a spent billing quota is not a rate limit (TASK-13844)

OpenAI answers a spent billing quota with a 429 whose error type and code are
insufficient_quota. build_sanitized_chat_error made it a plain
ChatRateLimitError, so a summary read it as rate_limit and the audio analysis
stage retried it.

ChatRateLimitError gains quota_exhausted, a bool that build_sanitized_chat_error
sets by comparing the failed reply's error.type and error.code with
{"insufficient_quota"}; no upstream text is kept. The summary maps it to
provider_quota_exhausted, which is not retried. Chat responses are unchanged:
the class, status and message are the same, chat_service rebuilds the error
without the flag, and nothing else reads it.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 3: the readers, the docs, and close-out (AC3)

**Files:**
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py:23-24` (comment)
- Test: `tldw_Server_API/tests/Evaluations/unit/test_specialized_provider_auth_failures.py` (append)
- Modify: `Docs/API-related/Audio_Jobs_API.md:41`, plus its mirror `Docs/Published/API-related/Audio_Jobs_API.md` (regenerated)
- Backlog: `TASK-13844` (close), one new follow-up task

**Interfaces:**
- Consumes: `_SUMMARY_ERROR_CODES` with both new codes (Tasks 1 and 2); `evaluations_unified._eval_provider_http_exception(exc) -> HTTPException`.

- [ ] **Step 1: pin every summary code's Evaluations response**

This step records AC3's check, not a defect fix. The tests pass on their first run on Task 2's tree; on dev, the set check fails because the two codes do not exist yet. From now on, a new summary code fails `test_every_summary_code_has_a_chosen_evaluations_response` until its Evaluations response is chosen.

Append to `tldw_Server_API/tests/Evaluations/unit/test_specialized_provider_auth_failures.py` (it already imports `eval_endpoint` and `SummaryProviderError`):

```python
# Every summary code and the evaluations response it gets (TASK-13844). A new summary code
# fails the set check until someone chooses its response here.
_EVAL_RESPONSE_FOR_SUMMARY_CODE = {
    "authentication": (502, "provider_authentication_failed"),
    "missing_credentials": (503, "missing_provider_credentials"),
    "configuration": (503, "provider_configuration_invalid"),
    "invalid_provider": (503, "provider_configuration_invalid"),
    "adapter_unavailable": (503, "provider_configuration_invalid"),
    "missing_model": (503, "provider_configuration_invalid"),
    "rate_limit": (502, "provider_unavailable"),
    "provider_quota_exhausted": (502, "provider_unavailable"),
    "request_rejected": (502, "provider_unavailable"),
    "provider_failure": (502, "provider_unavailable"),
}


def test_every_summary_code_has_a_chosen_evaluations_response() -> None:
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as sgl

    assert set(_EVAL_RESPONSE_FOR_SUMMARY_CODE) == sgl._SUMMARY_ERROR_CODES


@pytest.mark.parametrize(("summary_code", "expected"), sorted(_EVAL_RESPONSE_FOR_SUMMARY_CODE.items()))
def test_summary_code_maps_to_its_evaluations_response(summary_code: str, expected: tuple[int, str]) -> None:
    response = eval_endpoint._eval_provider_http_exception(
        SummaryProviderError(code=summary_code, provider="openai")
    )

    assert (response.status_code, response.detail["error_code"]) == expected
```

Run:

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Evaluations/unit/test_specialized_provider_auth_failures.py -k summary_code
```

Expected: `11 passed`.

- [ ] **Step 2: name the new codes where the worker says which codes are not retried**

`tldw_Server_API/app/services/audio_jobs_worker.py:23-25`:

```python
# Analysis failures a retry can fix. The other SummaryProviderError codes (missing model or
# credentials, bad configuration or authentication, an unknown provider, a request the provider
# rejected, a spent provider quota) fail the same way every time.
_RETRYABLE_ANALYSIS_CODES = frozenset({"rate_limit", "provider_failure"})
```

- [ ] **Step 3: the audio jobs doc**

Replace the "Analysis failures" bullet in `Docs/API-related/Audio_Jobs_API.md` (line 41) with:

```markdown
- Analysis failures: these fail the `audio_analyze` stage job at once, with the reason in that job row's `error_code`: `missing_model`, `authentication`, `configuration`, `invalid_provider`, `adapter_unavailable`, `request_rejected`, `provider_quota_exhausted`, `llm_quota_exceeded` or `empty_transcript`. A missing API key shows up as `authentication` or `configuration`. A provider switched off with `LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER>=0` shows up as `configuration` (Review M3). `request_rejected` means the provider refused the request (an unknown model, a transcript too long for the model); `provider_quota_exhausted` means the provider account's billing quota is spent (OpenAI `insufficient_quota`). A rate limit (`rate_limit`) and any other provider failure (`provider_failure`, which includes a provider's transient 408 or 498 reply) are retried. The same failure twice in a row quarantines the job (`JOBS_QUARANTINE_THRESHOLD`, default 2).
```

If TASK-13847 (`plan_audio_analysis.md`) has already rewritten this bullet, keep its wording (Review I2). Add `request_rejected` and `provider_quota_exhausted` to its list of failures that are stored without a retry, with the two meanings above. Also keep, or add, the sentence "A provider switched off with `LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER>=0` shows up as `configuration`."

- [ ] **Step 4: refresh the published mirror**

```bash
bash Helper_Scripts/refresh_docs_published.sh
git status --short Docs/
```

Expected: exactly `Docs/API-related/Audio_Jobs_API.md` and `Docs/Published/API-related/Audio_Jobs_API.md` modified (measured on the prototype).

- [ ] **Step 5: the regression sweep, lint and Bandit**

Run every test file that names a changed symbol: `ChatRateLimitError`, `build_sanitized_chat_error`, `raise_chat_error_from_http`, `normalize_error`, `SummaryProviderError`, the chat error mapping and the audio worker.

```bash
T=tldw_Server_API/tests
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  $T/LLM_Calls/test_summarization_adapter.py $T/LLM_Calls/test_summarization_runtime_credentials.py \
  $T/LLM_Calls/test_provider_adapter_runtime_boundary.py $T/LLM_Calls/test_http_status_extraction_parity.py \
  $T/LLM_Calls/test_llm_streaming_and_security.py $T/LLM_Calls/test_llm_providers.py $T/LLM_Calls/test_mlx_provider.py \
  $T/LLM_Adapters/unit $T/LLM_Adapters/integration/test_adapters_chat_endpoint_error_sse_core_providers.py \
  $T/AudioJobs/test_audio_worker_smoke.py $T/Audio/test_audio_jobs_worker_helper.py \
  $T/Evaluations/unit/test_specialized_provider_auth_failures.py \
  $T/RAG_NEW/unit/test_hyde.py $T/RAG_NEW/unit/test_post_verifier.py $T/RAG_NEW/unit/test_generation_executor.py \
  $T/Chat/unit/test_error_handling.py $T/Chat/unit/test_chat_service_fallback.py $T/Chat/unit/test_chat_processing_unit.py \
  $T/Chat/unit/test_orchestrator_network_exception_routing.py \
  $T/Chat/integration/test_chat_endpoint.py $T/Chat/integration/test_chat_fixes_integration.py \
  $T/Character_Chat_NEW/unit/test_chat_session_error_mapping.py \
  $T/Admin/test_admin_llm_provider_test_runtime.py $T/Admin/test_admin_llm_providers_service_sanitizers.py \
  $T/AuthNZ_Unit/test_byok_validation_boundary.py $T/Flashcards/test_study_assistant_adapter_contract.py \
  $T/Services/test_writing_annotation_review_jobs_worker.py $T/Config/test_config_providers_endpoints.py \
  $T/VN_Play/test_vn_play_generated_outputs.py $T/Web_Scraping/test_phase4_extraction_failures.py \
  $T/Docs
```

Expected, measured on a prototype of all three tasks, then adjusted for the one added row (Review I1):
- Without `$T/Docs`: `2257 passed, 4 skipped` (was 2256). On unchanged dev, the same files give `2226 passed, 4 skipped` (measured). The difference is exactly the 31 new tests: 10 in Task 1, 10 in Task 2, 11 in Task 3.
- `$T/Docs` on its own: `212 passed`. It checks the published mirror.
- So the whole command: `2469 passed, 4 skipped` (was 2468).

Review M4: these test files name a changed symbol too and pin the chat surface that this plan promises not to change. Run them as well:

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  $T/AuthNZ_Unit/test_byok_runtime.py \
  $T/Chat_NEW/integration/test_roleplay_resume_completion.py \
  $T/Chat/integration/test_chat_endpoint_simplified.py $T/Chat/integration/test_chat_public_request_trust_boundary.py \
  $T/Chat/unit/test_chat_service_tool_autoexec.py $T/Chat/unit/test_messages_endpoint_usage.py \
  $T/Chat/unit/test_authoritative_adapter_translation.py $T/Chat/unit/test_streaming_utils.py \
  $T/LLM_Adapters/integration/test_async_adapters_orchestrator.py \
  $T/LLM_Adapters/integration/test_async_adapters_orchestrator_anthropic_native.py \
  $T/LLM_Adapters/integration/test_async_adapters_orchestrator_groq_openrouter_native.py \
  $T/LLM_Adapters/integration/test_adapters_chat_endpoint_anthropic_native.py \
  $T/Usage/test_llm_roots_non_sdk.py
```

Expected: all pass. The reviewer's run of this set plus two `Sync` files (the review does not name them) gave `1030 passed, 3 skipped`; write down the count the files above give, because it will be a little lower without those two.

If anything fails, run the same file on unchanged `origin/dev` before blaming the change.

```bash
$PY -m ruff check tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py tldw_Server_API/app/core/LLM_Calls/error_utils.py tldw_Server_API/app/core/LLM_Calls/providers/base.py tldw_Server_API/app/core/exceptions.py tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_error_sanitization.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py tldw_Server_API/tests/Evaluations/unit/test_specialized_provider_auth_failures.py
uvx bandit -ll tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py tldw_Server_API/app/core/LLM_Calls/error_utils.py tldw_Server_API/app/core/LLM_Calls/providers/base.py tldw_Server_API/app/core/exceptions.py tldw_Server_API/app/services/audio_jobs_worker.py
git diff --check origin/dev
```

Expected:
- ruff: `Found 3 errors` (2 × I001, 1 × UP028). Dev has the same three in these files (measured), so none are new.
- Bandit: `No issues identified.`
- `git diff --check`: no output.

- [ ] **Step 6: file the follow-up**

Planning found two untyped adapter configuration errors outside this task, and review found a third gap (Review M1):
- `EgressPolicyError` is not in `_SUMMARY_ADAPTER_EXCEPTIONS` or `_SUMMARIZATION_NONCRITICAL_EXCEPTIONS`. It escapes `analyze()` untyped whatever `raise_on_error` is; measured on dev with a fake adapter. The audio worker's generic handler then retries it.
- `custom_openai_adapter.py:158` and `:211` raise a bare `RuntimeError` for a missing base URL. Its tests (`test_custom_openai_native_http.py:140`, `test_authoritative_custom_openai_endpoint_snapshot.py:95`) pin it as `RuntimeError`.
- Local adapters: `_raise_sanitized_local_provider_error` (`local_adapters.py:193-227`) maps every 4xx, including 401, 403 and 429, to `ChatBadRequestError` without `upstream_status_code`. A local 429 (a server behind a rate-limiting proxy) becomes `request_rejected` and is no longer retried.

Find the highest id across `origin/dev` and the open PRs with the AGENTS.md command, and create the task with the next one. Run Steps 6 and 7 in the same shell, because Step 7 uses `$FOLLOW_UP`.

```bash
git fetch origin dev --quiet
MAX_ID=$({ git ls-tree -r --name-only origin/dev backlog/; gh pr list --state open --limit 500 --json files --jq '.[].files[].path'; } \
  | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1)
FOLLOW_UP="TASK-$((MAX_ID + 1))"
echo "$FOLLOW_UP"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Summary errors: an egress-blocked or unconfigured custom endpoint is not typed as a configuration error" \
  --id "$FOLLOW_UP" \
  -d "Found while planning TASK-13844. EgressPolicyError (core/exceptions.py) is in neither _SUMMARY_ADAPTER_EXCEPTIONS nor _SUMMARIZATION_NONCRITICAL_EXCEPTIONS (core/LLM_Calls/Summarization_General_Lib.py), so it escapes analyze() untyped whatever raise_on_error is, and the audio jobs worker's generic handler retries it. custom_openai_adapter.py:158 and :211 raise a bare RuntimeError for a missing base URL, which a summary records as provider_failure; tests pin it as RuntimeError. Local adapters (local_adapters.py _raise_sanitized_local_provider_error) map every 4xx, including 401, 403 and 429, to ChatBadRequestError without upstream_status_code, so a local 429 now reads request_rejected and is not retried. TASK-13844 typed only the LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER> kill switch, by reading the adapter's own switch." \
  --ac "An egress-blocked provider endpoint gets a SummaryProviderError code the audio analysis stage does not retry, and analyze(raise_on_error=False) callers keep their current behaviour" \
  --ac "A custom OpenAI-compatible provider with no base URL gets the configuration summary code; any chat response change is stated and tested" \
  --ac "A local provider's 401, 403 and 429 map to authentication and rate_limit summary codes (not request_rejected), and an unpulled model (404) stays request_rejected" \
  -l llm,errors
git add -A backlog/tasks
```

- [ ] **Step 7: close TASK-13844**

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13844 \
  --append-notes "AC1: _summary_error_code maps ChatBadRequestError (every upstream 4xx but 401, 403 and 429, via build_sanitized_chat_error) to the new request_rejected code, and a RuntimeError from an adapter whose LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER> switch is off to configuration. The transient 4xx replies (OpenRouter 408, Groq 498) stay provider_failure, read from the error's upstream_status_code, so they are still retried. The adapters' 12 raise sites are unchanged. The openai_adapter scope-mismatch ValueError cannot reach a summary: the summary request never carries a call_policy. RED 7 failed (6 summary codes, 1 worker retry), GREEN 16 passed (15 in the summarization file's selection, 1 worker test)." \
  --append-notes "AC2: ChatRateLimitError gains quota_exhausted (bool, default False). build_sanitized_chat_error sets it from the failed reply's error.type/error.code against {insufficient_quota}, through _raise_sanitized_provider_failure, normalize_error and raise_chat_error_from_http. The summary maps it to provider_quota_exhausted. Chat responses are unchanged: same class, status and message, chat_service rebuilds the error without the flag, and nothing else reads it. A streamed 429 keeps rate_limit (the body is unread when httpx raises). RED 10 failed, GREEN 114 passed in the two files plus 2 worker tests." \
  --append-notes "AC3: reader audit in the plan. Neither new code is in _RETRYABLE_ANALYSIS_CODES, so both fail the analysis stage at once. Evaluations (_eval_provider_http_exception) answers both with 502 provider_unavailable, as before; the kill switch now gets 503 provider_configuration_invalid, as other configuration errors do. The RAG readers record the constant provider_unavailable or allowlist the code (the description's 12 metadata sites record no summary code on dev); HyDE now says provider_configuration_invalid for the kill switch. New test pins every summary code's Evaluations response. Sweep over every test file naming a changed symbol, plus tests/Docs: 2469 passed, 4 skipped (dev: 2226 passed, 4 skipped without tests/Docs; the 31 new tests are the difference), plus the Review M4 chat-surface files. Ruff: only the 3 findings dev has. Bandit clean. Follow-up: $FOLLOW_UP (EgressPolicyError, the custom_openai base-URL RuntimeError and local-adapter 4xx mapping)." \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A summary the provider refuses (an unknown model, a prompt over the context) is now request_rejected, a spent billing quota (OpenAI insufficient_quota) is provider_quota_exhausted, and an adapter switched off with LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER>=0 is configuration. The audio analysis stage retries none of them. Chat responses are unchanged; Evaluations and HyDE now report the switched-off adapter as a configuration error." \
  -s Done
git add -A backlog/tasks
```

The counts in the notes are the measured ones. If Step 5 printed different counts, write those instead. `$FOLLOW_UP` is set in Step 6, which is why the two steps run in the same shell.

- [ ] **Step 8: commit**

```bash
git add tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/tests/Evaluations/unit/test_specialized_provider_auth_failures.py Docs/API-related/Audio_Jobs_API.md Docs/Published/API-related/Audio_Jobs_API.md
git add -A backlog/tasks
git commit -F - <<'EOF'
docs(audio-jobs): the analysis codes that are not retried; pin each summary code's Evaluations response (TASK-13844)

The audio jobs doc said a provider rejecting the request was retried; it now
lists request_rejected and provider_quota_exhausted with the failures that are
not, and says a switched-off adapter is configuration. A new test pins the
Evaluations response for every summary code, so the next code added has to
choose one. Closes TASK-13844 and files the follow-up for the egress-blocked
and unconfigured custom endpoints and the local adapters' 4xx mapping.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

## Out of scope

- **402 as a spent quota.** DeepSeek and OpenRouter answer an empty balance with 402, which this plan makes `request_rejected`. That code is also not retried. Mapping 402 to `provider_quota_exhausted` would need the upstream status. Only `_raise_sanitized_provider_failure` keeps it (`upstream_status_code`), so `normalize_error` and `raise_chat_error_from_http` would need it too.
- **Typing the kill switch at its raise sites.** Retyping the 12 raises as a `ChatConfigurationError` would change the chat endpoints' response for a switched-off provider: `chat_service` would map it to "configuration invalid" instead of "provider unavailable". It would also break `test_native_http_kill_switch_reachable.py`. The `ponytail:` comment on `_native_http_disabled` records the upgrade.
- **The Evaluations response for the new codes.** It stays 502 `provider_unavailable`; see "Decisions for the owner".
- **In-stream SSE error events in a streamed summary** (`Summarization_General_Lib.py:325-329`). These stay `provider_failure`, and no `raise_on_error` caller streams.
- **`EgressPolicyError` and the custom_openai base-URL `RuntimeError`.** These are the follow-up task from Task 3, Step 6.
