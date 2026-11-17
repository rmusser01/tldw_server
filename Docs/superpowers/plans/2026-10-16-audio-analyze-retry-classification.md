# Audio Analysis Stage Retries Only What a Retry Can Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The audio jobs worker's `audio_analyze` stage fails a job at once when a retry cannot help (bad provider setup, an empty transcript), still retries rate limits and transient provider failures, records the reason in the job's `error_code`, and skips analysis when the job names no provider instead of calling `openai` (TASK-13640).

**Architecture:** One file changes behaviour: `tldw_Server_API/app/services/audio_jobs_worker.py`. The stage catches `SummaryProviderError` around its `analyze` call and re-raises it as a small private `_StageFailure` carrying `failure_code` (the error's `code`) and `retryable` (true only for `rate_limit` and `provider_failure`). The worker's existing failure branch already passes `getattr(failure, "retryable", True)` and `getattr(failure, "failure_code", None)` to `JobManager.fail_job`, so it needs no change. An empty transcript raises `_StageFailure(failure_code="empty_transcript", retryable=False)` before any provider call. A job whose `api_name` is missing, blank or `"None"` records `analysis_skipped: "no api_name"` and moves on to `audio_store`, as `/media/process-audios` does.

**Tech Stack:** Python, asyncio, pytest (the existing `analyze_worker` fixture in `tests/AudioJobs/test_audio_worker_smoke.py`, which runs the real worker loop and the real `analyze()` over a fake provider adapter and a temporary SQLite jobs DB).

**Spec:** `backlog/tasks/task-13640 - Audio-analysis-stage-retries-configuration-errors.md` (no design doc: a bounded bug fix). Its two implementation notes are part of the spec: forward `SummaryProviderError.code` as the job's error code (PR review of TASK-13636/13637), and stop falling back to `openai` for a job without `api_name` (clean-session review of PR #3240).

## Global Constraints

- **Branch:** `fix/audio-analyze-retry-classification` from `origin/dev`; the PR targets `dev`.
- **Retryable codes, verbatim:** `frozenset({"rate_limit", "provider_failure"})`. Every other `SummaryProviderError` code (`invalid_provider`, `adapter_unavailable`, `missing_model`, `missing_credentials`, `authentication`, `configuration`) fails the job without retry.
- **Error codes:** a `SummaryProviderError`'s `code` becomes the job's `error_code`, retried or not; an empty or whitespace-only transcript fails with `error_code` `empty_transcript`.
- **No provider named:** `api_name` missing, blank or `"None"` (any case) skips analysis: the payload gets `analysis_skipped: "no api_name"`, no provider is called, and the job continues to `audio_store`. No default provider is guessed. This is the implementation note's second option and matches `/media/process-audios` and `Audio_Files.py` (`"[Analysis Skipped: No API specified]"`). Its first option, "the configured default LLM provider", is not taken:
  - `core/Chat/chat_target_resolution.config_default_llm_provider()` exists.
  - But the config loader falls back to `openai` (`core/config.py:4183`), and the shipped `config.txt` sets `default_api = openai`, so a configured default cannot be told from the shipped one.
  - Resolving it would call `openai` on a local-only install, which is the failure the note describes.
  - Trade-off: the setup wizard writes `default_api` (`core/Setup/readiness_service.py:175`, `:211`), so wizard users would have had analysis under the first option.
- **`SummaryProviderError` itself is not changed.** Other job workers read `retryable` from exceptions that escape their handlers, with different defaults: `Evaluations/recipe_runs_jobs_worker.py:227` (False), `Evaluations/embeddings_abtest_jobs_worker.py:208` (True), `Jobs/worker_sdk.py:852` (True), `Workflows/engine.py:366` (None). Giving the class a `retryable` attribute would change their retries too.
- **Test first:** the RED run below happens before the source change.
- **Docs mirror:** edit `Docs/API-related/Audio_Jobs_API.md`, then run `bash Helper_Scripts/refresh_docs_published.sh` (bash, not sh: the script uses `BASH_SOURCE`) and commit the `Docs/Published/API-related/Audio_Jobs_API.md` change with it. Without the mirror, `tests/Docs/test_docs_published_refresh.py` fails.
- **Backlog:** `tools/backlog-py` only (`PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit ...`), `--append-notes`, explicit id `TASK-13640`, `git add -A backlog/tasks`.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never `git stash`.
- **Out of scope:**
  - TASK-13641 (the TTS worker acquires from the same queue).
  - With `perform_chunking: false`, `audio_transcribe` hands over straight to `audio_store`, so analysis never runs even when `perform_analysis` is true. This is pre-existing and a follow-up candidate.
  - **A provider rejecting the request is still retried until quarantined.** Any 4xx other than 401, 403 and 429 is mapped to `ChatBadRequestError` (`core/LLM_Calls/error_utils.py:120-125`). That includes an unknown or unpulled model (404) and a transcript too long for the model's context (400). `_summary_error_code` maps it to `provider_failure`, as it does the adapter's configuration `RuntimeError` and `ValueError` (`providers/openai_adapter.py:315`, `:345`) and a reply truncated at the output limit. `SummaryProviderError` is raised `from None`, so the worker cannot tell these apart. Classifying them needs a new summary code (for example `bad_request`), which about 12 RAG and Evaluations sites that record `SummaryProviderError.code` would also see. This is a follow-up candidate. It is not filed in this PR, because backlog ids are allocated locally and other open PRs also add tasks.
  - **A retried failure gets one retry.** With `JOBS_QUARANTINE_THRESHOLD` at its default of 2 (`core/Jobs/manager.py`, the streak keyed on `error_code or error`), a job that fails the same way twice in a row is quarantined. A rate limit therefore gets one retry, after about a second. This is pre-existing. The docs line says so.
  - An `Error:` string that `analyze` returns rather than raises (for example `"Error: Summarization failed unexpectedly."`) still raises `RuntimeError` and is retried, as before.

## Review Focus

1. **`api_name` given as `"None"` or blank.** `Audio_Files.py` writes the string `"None"`, and a client may send `""`. Either is now skipped like a missing name. Before, `""` fell back to `openai`. `"None"` reached `analyze`, which returns `"Error: Analysis API provider is required."`, and the job was retried until quarantined. Tests: `test_audio_analyze_without_api_name_skips_analysis[blank]` and `[None-string]`.
2. **A whitespace-only transcript** (silence transcribed as `" \n "`). It fails with `empty_transcript` without calling the provider. Before, `analyze` sent it to the model, because its own check is `if not text_content`, not a strip. Test: `test_audio_analyze_empty_transcript_fails_without_retry[whitespace]`.
3. **Missing credentials without explicit credentials.** The worker never passes `provider_credentials`, so `analyze`'s own `missing_credentials` check does not run. A missing key surfaces from the adapter as `ChatAuthenticationError` or `ChatConfigurationError`, and both map to codes that fail without retry. Test: `test_audio_analyze_provider_setup_failure_fails_without_retry[ChatAuthenticationError-authentication]` and `[ChatConfigurationError-configuration]`.
4. **Retried failures now carry a code.** `JobManager.fail_job` keys its failure-streak quarantine on `error_code or error`, so the streak moves from the message to the code. For a given code and provider the message (`"Summary provider error: rate_limit (openai)"`) was already constant, so the quarantine point does not move. Tests: `test_audio_analyze_rate_limit_still_retries`, `test_audio_analyze_provider_failure_still_retries`.
5. **An LLM quota refusal still fails at once with `llm_quota_exceeded`.** `LLMQuotaExceededError` is not a `SummaryProviderError`, so the new `except` does not catch it, and the failure branch reads its own `retryable` and `failure_code`. Test: `test_audio_analyze_quota_refusal_fails_the_job_without_retry` (unchanged).

---

### Task 1: The analysis stage classifies its failures

**Files:**
- Modify: `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py` (the `audio_analyze` section at the end of the file)
- Modify: `tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py` (`test_the_audio_jobs_worker_runs_analysis_for_the_job_owner`: its job now names `api_name`, or it would be skipped)
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py` (after `DOMAIN`; the `audio_analyze` branch; one comment in the failure branch)
- Modify: `Docs/API-related/Audio_Jobs_API.md` and its mirror `Docs/Published/API-related/Audio_Jobs_API.md`
- Modify: `backlog/tasks/task-13640*.md` (backlog-py only)
- Add: this plan, `Docs/superpowers/plans/2026-10-16-audio-analyze-retry-classification.md`

**Interfaces:**
- Produces: nothing public. `_StageFailure` and `_RETRYABLE_ANALYSIS_CODES` are private to `audio_jobs_worker.py`.
- Consumes: `SummaryProviderError.code` (`core/LLM_Calls/Summarization_General_Lib.py:125`); `JobManager.fail_job(..., retryable=, error_code=)` (unchanged, `core/Jobs/manager.py:8168`).
- Changes: a job payload that reaches `audio_store` without analysis now carries `analysis_skipped: "no api_name"`. `audio_store` is a placeholder that stores the payload as the job result, and nothing reads `analysis_skipped`.

Set once for every step: `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` and `export TLDW_TEST_NO_DOCKER=1`.

- [ ] **Step 1: Branch**

```bash
git fetch -q origin dev
git switch -c fix/audio-analyze-retry-classification origin/dev
```

Copy this plan to `Docs/superpowers/plans/2026-10-16-audio-analyze-retry-classification.md`.

The plan stays uncommitted until Step 10.

- [ ] **Step 2: Write the failing tests**

Apply exactly this diff (save it to a file and `git apply` it). It renames `test_audio_analyze_fails_the_stage_on_an_error_result`, which asserted the old retry, and changes its assertions.

```diff
diff --git a/tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py b/tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
index 33652b6865..c647a0e521 100644
--- a/tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
+++ b/tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
@@ -809,14 +809,19 @@ async def _run_worker_until(worker, done) -> None:
         await asyncio.wait_for(task, timeout=2.0)


-def _analyze_job(jm, text: str) -> int:
+def _analyze_job(jm, text: str, api_name: str | None = "openai") -> int:
     row = jm.create_job(
         domain="audio", queue="default", job_type="audio_analyze",
-        payload={"text": text, "api_name": "openai", "model": "whisper-1", "perform_analysis": True}, owner_user_id="7",
+        payload={"text": text, "api_name": api_name, "model": "whisper-1", "perform_analysis": True}, owner_user_id="7",
     )
     return int(row["id"])


+def _failed_or_retried(jm, job_id: int) -> bool:
+    job = jm.get_job(job_id) or {}
+    return job.get("status") in {"failed", "quarantined"} or int(job.get("retry_count") or 0) >= 1
+
+
 @pytest.mark.unit
 async def test_audio_analyze_stores_the_model_analysis(analyze_worker):
     worker, jm, adapter = analyze_worker
@@ -833,16 +838,84 @@ async def test_audio_analyze_stores_the_model_analysis(analyze_worker):


 @pytest.mark.unit
-async def test_audio_analyze_fails_the_stage_on_an_error_result(analyze_worker):
+@pytest.mark.parametrize("text", ["", " \n "], ids=["empty", "whitespace"])
+async def test_audio_analyze_empty_transcript_fails_without_retry(analyze_worker, text):
     worker, jm, adapter = analyze_worker
-    job_id = _analyze_job(jm, "")  # analyze() returns "Error: Could not extract text content."
-    await _run_worker_until(worker, lambda: int((jm.get_job(job_id) or {}).get("retry_count") or 0) >= 1)
+    job_id = _analyze_job(jm, text)
+    await _run_worker_until(worker, lambda: _failed_or_retried(jm, job_id))
     job = jm.get_job(job_id) or {}
-    assert int(job.get("retry_count") or 0) >= 1
-    assert "Could not extract text content" in str(job.get("error_message") or job.get("last_error"))
+    assert job.get("status") == "failed"
+    assert int(job.get("retry_count") or 0) == 0
+    assert job.get("error_code") == "empty_transcript"
+    assert adapter.requests == []
     assert jm.list_jobs(domain="audio", job_type="audio_store", owner_user_id="7") == []


+@pytest.mark.unit
+@pytest.mark.parametrize(
+    ("error_name", "code"),
+    [("ChatAuthenticationError", "authentication"), ("ChatConfigurationError", "configuration")],
+)
+async def test_audio_analyze_provider_setup_failure_fails_without_retry(analyze_worker, error_name, code):
+    from tldw_Server_API.app.core import exceptions
+
+    worker, jm, adapter = analyze_worker
+    adapter.error = getattr(exceptions, error_name)(provider="openai")
+    job_id = _analyze_job(jm, "hello transcript")
+    await _run_worker_until(worker, lambda: _failed_or_retried(jm, job_id))
+    job = jm.get_job(job_id) or {}
+    assert job.get("status") == "failed"
+    assert int(job.get("retry_count") or 0) == 0
+    assert job.get("error_code") == code
+
+
+@pytest.mark.unit
+async def test_audio_analyze_missing_model_fails_without_retry(analyze_worker, monkeypatch):
+    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as summarization
+
+    worker, jm, adapter = analyze_worker
+    monkeypatch.setattr(summarization, "load_and_log_configs", lambda: {"openai_api": {}})
+    monkeypatch.delenv("DEFAULT_MODEL_OPENAI", raising=False)
+    job_id = _analyze_job(jm, "hello transcript")
+    await _run_worker_until(worker, lambda: _failed_or_retried(jm, job_id))
+    job = jm.get_job(job_id) or {}
+    assert job.get("status") == "failed"
+    assert int(job.get("retry_count") or 0) == 0
+    assert job.get("error_code") == "missing_model"
+    assert adapter.requests == []
+
+
+@pytest.mark.unit
+async def test_audio_analyze_rate_limit_still_retries(analyze_worker):
+    from tldw_Server_API.app.core.exceptions import ChatRateLimitError
+
+    worker, jm, adapter = analyze_worker
+    adapter.error = ChatRateLimitError(provider="openai")
+    job_id = _analyze_job(jm, "hello transcript")
+    await _run_worker_until(worker, lambda: _failed_or_retried(jm, job_id))
+    job = jm.get_job(job_id) or {}
+    assert int(job.get("retry_count") or 0) >= 1
+    assert job.get("status") != "failed"
+    assert job.get("error_code") == "rate_limit"
+
+
+@pytest.mark.unit
+@pytest.mark.parametrize("api_name", [None, "", "None"], ids=["missing", "blank", "None-string"])
+async def test_audio_analyze_without_api_name_skips_analysis(analyze_worker, api_name):
+    worker, jm, adapter = analyze_worker
+    _analyze_job(jm, "hello transcript", api_name=api_name)
+
+    def stored():
+        return jm.list_jobs(domain="audio", job_type="audio_store", owner_user_id="7")
+
+    await _run_worker_until(worker, stored)
+    assert stored(), "audio_analyze did not hand over to audio_store"
+    payload = stored()[0]["payload"]
+    assert payload.get("analysis_skipped") == "no api_name"
+    assert "analysis" not in payload
+    assert adapter.requests == []
+
+
 @pytest.mark.unit
 async def test_audio_analyze_quota_refusal_fails_the_job_without_retry(analyze_worker):
     from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
@@ -868,4 +941,5 @@ async def test_audio_analyze_provider_failure_still_retries(analyze_worker):
     job = jm.get_job(job_id) or {}
     assert int(job.get("retry_count") or 0) >= 1
     assert job.get("status") != "failed"
+    assert job.get("error_code") == "provider_failure"
     assert len(adapter.requests) >= 1
diff --git a/tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py b/tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py
index 01a626c2b9..f1447f422e 100644
--- a/tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py
+++ b/tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py
@@ -94,7 +94,7 @@ async def test_the_audio_jobs_worker_runs_analysis_for_the_job_owner(monkeypatch
     jm = JobManager()
     jm.create_job(
         domain="audio", queue="default", job_type="audio_analyze",
-        payload={"text": "hello worker", "perform_analysis": True}, owner_user_id="7",
+        payload={"text": "hello worker", "api_name": "openai", "perform_analysis": True}, owner_user_id="7",
     )
     stop = asyncio.Event()
     task = asyncio.create_task(worker.run_audio_jobs_worker(stop))
```

- [ ] **Step 3: Run them to see them fail**

Run: `$PY -m pytest -p no:cacheprovider -p no:randomly -q tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k audio_analyze`

Expected: `10 failed, 2 passed, 10 deselected`. Measured on `origin/dev` `0566130cbe`:
- `test_audio_analyze_empty_transcript_fails_without_retry[empty]`: `AssertionError: assert 'quarantined' == 'failed'`. `analyze` returned `"Error: Could not extract text content."` and the job was retried until quarantined.
- `test_audio_analyze_empty_transcript_fails_without_retry[whitespace]`: `AssertionError: assert 'completed' == 'failed'`. The model was called with `" \n "` and the stage completed.
- `test_audio_analyze_provider_setup_failure_fails_without_retry[ChatAuthenticationError-authentication]`, `[ChatConfigurationError-configuration]` and `test_audio_analyze_missing_model_fails_without_retry`: `AssertionError: assert 'quarantined' == 'failed'`.
- `test_audio_analyze_rate_limit_still_retries`: `AssertionError: assert None == 'rate_limit'`; `test_audio_analyze_provider_failure_still_retries`: `AssertionError: assert None == 'provider_failure'`. Both retried, as they should, but recorded no error code.
- `test_audio_analyze_without_api_name_skips_analysis[missing]` and `[blank]`: `AssertionError: assert None == 'no api_name'`. The stage called `openai` (the fake adapter) and stored its analysis.
- `test_audio_analyze_without_api_name_skips_analysis[None-string]`: `AssertionError: audio_analyze did not hand over to audio_store`. `analyze("None", ...)` returned an `Error:` string and the job was retried.

The two that pass are guards: the stored analysis and the quota refusal, which the base worker already handles. On base, a job that should fail at once reads `quarantined` rather than `failed`: its identical failures trip the Jobs failure-streak quarantine before the test reads it.

- [ ] **Step 4: Apply the source change**

Apply exactly this diff:

```diff
diff --git a/tldw_Server_API/app/services/audio_jobs_worker.py b/tldw_Server_API/app/services/audio_jobs_worker.py
index e23d2f19b3..89e3adcb9f 100644
--- a/tldw_Server_API/app/services/audio_jobs_worker.py
+++ b/tldw_Server_API/app/services/audio_jobs_worker.py
@@ -20,6 +20,18 @@ from tldw_Server_API.app.core.Usage.audio_quota import (
 from tldw_Server_API.app.core.Usage.llm_roots import llm_call_root

 DOMAIN = "audio"
+# Analysis failures a retry can fix. The other SummaryProviderError codes (missing model or
+# credentials, bad configuration or authentication, an unknown provider) fail the same way every time.
+_RETRYABLE_ANALYSIS_CODES = frozenset({"rate_limit", "provider_failure"})
+
+
+class _StageFailure(RuntimeError):
+    """A stage failure carrying its job error code and whether a retry can fix it."""
+
+    def __init__(self, message: str, *, failure_code: str, retryable: bool) -> None:
+        super().__init__(message)
+        self.failure_code = failure_code
+        self.retryable = retryable


 async def _handle_audio_transcribe_stage(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
@@ -345,24 +357,41 @@ async def run_audio_jobs_worker(stop_event: asyncio.Event | None = None) -> None
                     updated_payload["chunks"] = chunks
                     next_type = payload.get("perform_analysis") and "audio_analyze" or "audio_store"
                 elif jtype == "audio_analyze":
-                    from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
+                    from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import (
+                        SummaryProviderError,
+                        analyze,
+                    )
                     text = updated_payload.get("text")
                     if not text:
                         # Fallback to segments
                         segs = updated_payload.get("segments") or []
                         text = " ".join((s.get("Text", "").strip() if isinstance(s, dict) else "") for s in segs).strip()
-                    with llm_call_root(owner):
-                        analysis = await asyncio.to_thread(
-                            analyze,
-                            payload.get("api_name") or "openai",
-                            text,
-                            None,  # jobs carry no prompt: analyze's default system prompt (bulleted notes)
-                            input_is_literal_text=True,
-                            raise_on_error=True,
+                    api_name = str(payload.get("api_name") or "").strip()
+                    if not api_name or api_name.lower() == "none":
+                        # No provider named: skip analysis, as /media/process-audios does.
+                        updated_payload["analysis_skipped"] = "no api_name"
+                    elif not text.strip():
+                        raise _StageFailure(
+                            "empty transcript: nothing to analyze", failure_code="empty_transcript", retryable=False
                         )
-                    if isinstance(analysis, str) and analysis.startswith("Error:"):
-                        raise RuntimeError(analysis)
-                    updated_payload["analysis"] = analysis
+                    else:
+                        try:
+                            with llm_call_root(owner):
+                                analysis = await asyncio.to_thread(
+                                    analyze,
+                                    api_name,
+                                    text,
+                                    None,  # jobs carry no prompt: analyze's default system prompt (bulleted notes)
+                                    input_is_literal_text=True,
+                                    raise_on_error=True,
+                                )
+                        except SummaryProviderError as exc:
+                            raise _StageFailure(
+                                str(exc), failure_code=exc.code, retryable=exc.code in _RETRYABLE_ANALYSIS_CODES
+                            ) from exc
+                        if isinstance(analysis, str) and analysis.startswith("Error:"):
+                            raise RuntimeError(analysis)
+                        updated_payload["analysis"] = analysis
                     next_type = "audio_store"
                 elif jtype == "audio_store":
                     # Placeholder: integration to store into DB or export artifacts
@@ -392,7 +421,7 @@ async def run_audio_jobs_worker(stop_event: asyncio.Event | None = None) -> None
                 else:
                     jm.complete_job(int(job["id"]), result={"payload": updated_payload}, worker_id=worker_id, lease_id=str(job.get("lease_id")), completion_token=str(job.get("lease_id")))
             else:
-                # An error that says it is not retryable (an LLM quota refusal) fails the job now.
+                # An error that says it is not retryable (an LLM quota refusal, a _StageFailure) fails the job now.
                 jm.fail_job(
                     int(job["id"]),
                     error=msg_err,
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `$PY -m pytest -p no:cacheprovider -p no:randomly -q tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k audio_analyze`

Expected: `12 passed, 10 deselected`.

- [ ] **Step 6: Document it and refresh the mirror**

Apply this diff to `Docs/API-related/Audio_Jobs_API.md` only, then mirror it:

```diff
diff --git a/Docs/API-related/Audio_Jobs_API.md b/Docs/API-related/Audio_Jobs_API.md
index 666ff0c621..4c46cbc54f 100644
--- a/Docs/API-related/Audio_Jobs_API.md
+++ b/Docs/API-related/Audio_Jobs_API.md
@@ -13,7 +13,7 @@ Base path: `/api/v1/audio`
     - `model` (string, default `whisper-1`): Transcription model.
     - `perform_chunking` (bool, default true)
     - `perform_analysis` (bool, default false)
-    - `api_name` (string, optional): LLM provider for analysis.
+    - `api_name` (string, optional): LLM provider for analysis. Without it the analysis stage is skipped, and the final payload records `analysis_skipped: "no api_name"`.
   - Response: `{ id, uuid, domain, queue, job_type, status }`
   - Auth: Single-user X-API-KEY or multi-user JWT.

@@ -36,6 +36,7 @@ Base path: `/api/v1/audio`
 ## Worker

 - Service: runs the pipeline stages `audio_download → audio_convert → audio_transcribe → audio_chunk → audio_analyze → audio_store`.
+- Analysis failures: these fail the job at once, with the reason in `error_code`: `missing_model`, `missing_credentials`, `authentication`, `configuration`, `invalid_provider`, `adapter_unavailable`, `llm_quota_exceeded` or `empty_transcript`. Every other provider failure is recorded as `rate_limit` or `provider_failure` and retried, including a provider rejecting the request (an unknown model, a transcript too long for the model). The same failure twice in a row quarantines the job (`JOBS_QUARANTINE_THRESHOLD`, default 2).
 - Env flags:
   - `AUDIO_JOBS_WORKER_ENABLED`: `true|false` (default false)
   - `AUDIO_JOBS_OWNER_STRICT`: enable owner-aware acquisition (default false)
```

```bash
bash Helper_Scripts/refresh_docs_published.sh
git status --short Docs
```

Expected: both `Docs/API-related/Audio_Jobs_API.md` and `Docs/Published/API-related/Audio_Jobs_API.md` are modified, and nothing else under `Docs`.

- [ ] **Step 7: Run every test file that imports the worker, and the docs tests**

```bash
git grep -l -F 'audio_jobs_worker' -- 'tldw_Server_API/tests/*.py'
RUN_JOBS=1 $PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Audio/test_audio_jobs_worker_helper.py \
  tldw_Server_API/tests/AudioJobs \
  tldw_Server_API/tests/Jobs/test_worker_owner_fallback_fails_closed.py \
  tldw_Server_API/tests/Services/test_main_lifecycle_contract.py \
  tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py \
  tldw_Server_API/tests/Services/test_shutdown_audio_jobs_worker.py \
  tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py \
  tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py \
  tldw_Server_API/tests/Docs
```

Expected: the `git grep` lists exactly the 8 files named above, outside `tests/Docs`. The run gives `405 passed, 4 skipped` (measured on the prototype). Without the Step 2 edit to `test_llm_roots_non_sdk.py`, its job (no `api_name`) would be skipped and the test would wait for an analysis that never comes. Without Step 6's mirror, two `tests/Docs/test_docs_published_refresh.py` tests fail.

- [ ] **Step 8: Lint and security**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py
uvx bandit -q -ll tldw_Server_API/app/services/audio_jobs_worker.py
```

Expected: ruff reports only `test_audio_worker_smoke.py:1:1: I001` and `test_audio_worker_smoke.py:224:5: I001`, both already on `origin/dev` (the file's top import block and an import inside an earlier test; this change touches neither). Bandit prints nothing and exits 0.

- [ ] **Step 9: Commit the fix**

```bash
git add tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py Docs/API-related/Audio_Jobs_API.md Docs/Published/API-related/Audio_Jobs_API.md
git commit -m "fix(audio): the analysis stage retries only what a retry can fix; no provider named skips analysis (TASK-13640)

A missing model, missing or rejected credentials, a bad configuration or an
unknown provider made the audio_analyze stage retry until the job was
quarantined, the same as a rate limit; so did an empty transcript. These now
fail the job at once, and the provider error's code (missing_model,
authentication, rate_limit, ...) becomes the job's error_code. Rate limits
and transient provider failures still retry.

A job without api_name used to call openai, which on a local-only install
failed on credentials and retried until quarantined. It now skips analysis
and records analysis_skipped, as /media/process-audios does.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 10: Close the task and commit the plan**

The notes below quote the measured results. If Steps 3, 5, 7 or 8 printed different lines, put the actual ones in the notes.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13640 --append-notes "Fixed in app/services/audio_jobs_worker.py: the audio_analyze stage re-raises SummaryProviderError as a private _StageFailure whose retryable is true only for rate_limit and provider_failure, and whose failure_code (the error's code) becomes the job's error_code; an empty or whitespace-only transcript fails with empty_transcript before any provider call; a job whose api_name is missing, blank or \"None\" records analysis_skipped: \"no api_name\" and continues to audio_store instead of calling openai. That is the implementation note's second option, as /media/process-audios does. The first option is not taken: core/Chat/chat_target_resolution.config_default_llm_provider() exists, but the config loader falls back to openai and the shipped config.txt sets default_api = openai, so resolving it would call openai on a local-only install. SummaryProviderError is unchanged, because other job workers read retryable from escaping exceptions with different defaults. Still retried until quarantined (follow-up candidate): a provider 4xx other than 401/403/429, such as an unknown or unpulled model (404) or a context overflow (400), maps to ChatBadRequestError and then to provider_failure in _summary_error_code; classifying it needs a new summary code that the RAG and Evaluations readers of SummaryProviderError.code would also see. Tests: the analyze tests 12 passed, 10 deselected (before the fix 10 failed, 2 passed, 10 deselected); every test file importing the worker plus tests/Docs 405 passed, 4 skipped. Ruff: only the two I001 in test_audio_worker_smoke.py that origin/dev has. Bandit -ll: no issues. Docs/API-related/Audio_Jobs_API.md documents the skip and the error codes (mirrored). Not here: with perform_chunking false, audio_transcribe goes straight to audio_store, so analysis never runs (pre-existing; follow-up candidate)." --check-ac 1 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "The audio analysis stage fails a job at once on a failure a retry cannot fix (missing model or credentials, bad configuration or authentication, unknown provider, empty transcript) and records the reason in error_code; rate limits and transient provider failures still retry. A job that names no provider skips analysis instead of calling openai." --status Done
git add -A backlog/tasks
git add Docs/superpowers/plans/2026-10-16-audio-analyze-retry-classification.md
git commit -m "docs(audio): plan for the analysis stage's retry classification; close TASK-13640

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: `git show --stat HEAD` lists the task file and the plan only.
