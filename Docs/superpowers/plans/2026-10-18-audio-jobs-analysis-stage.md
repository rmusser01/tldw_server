# Audio Jobs Analysis Stage Implementation Plan (TASK-13845, TASK-13846, TASK-13847)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An audio job that asks for analysis gets one whenever it can, waits sensibly when the provider rate-limits it, and never loses its transcript because analysis failed.

**Architecture:** All three tasks live in `app/services/audio_jobs_worker.py`'s stage hand-offs and its `audio_analyze` branch, plus one optional argument on `JobManager.fail_job`.
- A single helper picks the stage after transcription. Both the in-process worker and the GPU transcription worker use it, so analysis is no longer skipped when chunking is off.
- Analysis failures that a retry can fix are rescheduled with a real backoff, and they never quarantine the job.
- On the last attempt, or when the failure is not retryable, the transcript goes on to `audio_store`, with `analysis_error` recorded.

**Tech Stack:** Python 3.12, asyncio, the Jobs `JobManager` (SQLite and Postgres backends), pytest.

**Spec:** the three backlog tasks' acceptance criteria (on `dev`: `backlog/tasks/task-13845 …`, `task-13846 …`, `task-13847 …`).

**Independent review and dry run (2026-10-18):**
- Found 0 Blocker, 4 Important and 9 Minor; every RED the plan predicted appeared, except one (M3).
- All findings are folded in below, marked "Review I1" to "Review M9".
- Full report: `scratchpad/review_plan_audio_analysis.md`.

## Global Constraints

- Branch from `origin/dev`; the PR targets `dev`. **All four tasks ship in one PR.** On its own, Task 1 would send jobs without chunking into an analysis that can fail and lose the transcript; on `dev`, those jobs skip analysis and keep it. Task 3 is what makes Task 1 safe. (Review M8.)
- `fail_job`'s new argument defaults to today's behaviour. No other caller changes.
- The `audio_store` stage stays a placeholder (TASK-13851 owns the stored result's size). This plan only makes sure the transcript reaches it.
- **Backoffs:** rate limit 30 s base, provider failure 5 s base. `fail_job` doubles the base on each retry: 30, 60 and 120 s for a rate limit; 5, 10 and 20 s for a provider failure.
- **Attempts follow Jobs' own rule.** A job is requeued while `retry_count < max_retries`, checked before the increment. With the default `max_retries=3` that gives 4 attempts, with `retry_count` 0 to 3. The analysis stage stops retrying when `retry_count >= max_retries` and stores the transcript instead. (Review M1: the dry run showed 4 attempts, not 3.)

## Decisions for the owner (recommendation first)

1. **An LLM quota refusal during analysis**, which today fails the job (`llm_quota_exceeded`).
   - **Recommended:** store the transcript, with `analysis_error = {"code": "llm_quota_exceeded", "message": "LLM quota exceeded (<category>)"}` and no retry.
   - Why: no LLM call is made either way, and the refusal is recorded (TASK-13638 AC #1, "records that it was refused"). Design §4b says "a job fails as its handler decides" and "a job worker does not retry it", and neither is broken. The user keeps the transcript they asked for.
   - If taken, `Docs/Operations/Usage_Quotas.md:95`, the operator page TASK-13638 AC #2 names, changes in the same PR. (Review I2: `str(exc)` would store a dict repr, so the message is set explicitly.)
   - Alternative: keep failing the job.
2. **An empty transcript.** Today, with `perform_analysis` and an `api_name`, it fails the job (`empty_transcript`), and the transcript artifact (segments, language, duration) is lost. Without analysis, or without an `api_name`, the same empty transcript is stored.
   - **Recommended:** store it with `analysis_error.code = "empty_transcript"`. That removes a special case and makes all three paths agree; one existing test changes. (Review I4.)
   - Alternative: keep failing it.
3. **Where an analysis failure shows.** Today a setup failure (bad credentials, missing model) marks the `audio_analyze` job `failed`, with its `error_code`, a `job.failed` event and the failures-by-code metric. Storing the transcript instead completes that job.
   - **Recommended:** complete the analyze job with `result={"analysis_error": {...}}`, so `GET /api/v1/jobs/{id}` on it still shows the code, and log a warning. The failed-job event and metric are no longer raised for these cases.
   - Alternative: give up the transcript (today's behaviour) to keep the failed-job signal. (Review I3.)

## Review Focus

1. A rate-limited analysis on its last attempt: the transcript is stored, and the job does not end failed or quarantined.
2. A job that has already reached `audio_store` with an `analysis_error` and is then retried. The idempotency key `audio-stage:<id>:audio_store` must stop a second store stage.
3. GPU worker hand-off: its next stage must follow the same rule as the in-process worker.
4. An analysis reply that starts with `Error:` (analyze's legacy convention): it is treated as a provider failure, retried, then stored with `analysis_error` on the last attempt.
5. `fail_job(quarantine=False)` on both backends, and with `JOBS_QUARANTINE_THRESHOLD` set.

---

### Task 1: the stage after transcription (TASK-13845)

**Files:**
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py`:
  - new `stage_after_transcription`, used in `_handle_audio_transcribe_stage` at line ~98;
  - the `audio_chunk` branch keeps its own expression;
  - the `analysis_skipped` branch logs.
- Modify: `tldw_Server_API/app/services/audio_transcribe_gpu_worker.py:124`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py`: the `api_name` and `perform_analysis` descriptions on `SubmitAudioJobRequest`. `perform_analysis` says "after chunking", which goes stale. (Review M7.)
- Modify: the `_CaptureLogger` helper in `test_audio_worker_smoke.py` gains `warning`. Task 3 adds a `logger.warning`, and a test that swaps in `_CaptureLogger` on a failing-analysis path would otherwise raise `AttributeError` inside the worker. (Review M6.)
- Test: `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py`

- [ ] **Step 1: failing tests**

```python
@pytest.mark.unit
@pytest.mark.parametrize(
    ("chunking", "analysis", "expected"),
    [(False, True, "audio_analyze"), (False, False, "audio_store"), (True, True, "audio_chunk"), (True, False, "audio_chunk")],
)
def test_stage_after_transcription(chunking, analysis, expected):
    from tldw_Server_API.app.services.audio_jobs_worker import stage_after_transcription

    assert stage_after_transcription({"perform_chunking": chunking, "perform_analysis": analysis}) == expected


@pytest.mark.unit
async def test_audio_pipeline_without_chunking_still_analyzes(pipeline_worker):
    worker, jm, stt_calls = pipeline_worker
    jm.create_job(
        domain="audio", queue="default", job_type="audio_convert",
        payload={"local_path": "/tmp/p.mp3", "model": "whisper-1", "perform_chunking": False,
                 "perform_analysis": True, "api_name": None},
        owner_user_id="7",
    )
    await _run_worker_until(worker, lambda: bool(_jobs_of(jm, "audio_store")))
    assert [j["status"] for j in _jobs_of(jm, "audio_analyze")] == ["completed"]
```

  Plus a GPU-worker test in the style of `test_audio_gpu_worker_normalizes_segments_and_text` (line 270). It runs one `audio_transcribe` job with `perform_analysis: True, perform_chunking: False` and asserts that the queued next job is `audio_analyze`.

  Plus a test of the skip log. It captures loguru output, as `test_audio_gpu_worker_sanitizes_worker_error_log_but_preserves_job_error` does, and asserts that the line names the job and `api_name`.
- [ ] **Step 2:** run them. Expected:
  - `ImportError` for `stage_after_transcription`;
  - the pipeline test fails with `[] == ['completed']` (no `audio_analyze` job);
  - the GPU test gets `audio_store`;
  - no skip log.
- [ ] **Step 3: implement**

```python
def stage_after_transcription(payload: dict[str, Any]) -> str:
    """The stage after audio_transcribe: chunking first if asked, then analysis if asked, then store."""
    if payload.get("perform_chunking"):
        return "audio_chunk"
    return "audio_analyze" if payload.get("perform_analysis") else "audio_store"
```

  - Use it in `_handle_audio_transcribe_stage`. The GPU worker imports and uses it in place of its line 124.
  - The `audio_chunk` branch keeps `"audio_analyze" if payload.get("perform_analysis") else "audio_store"` (it already does this).
  - In the `analysis_skipped` branch, add `logger.info("Audio job {}: analysis skipped, no api_name", job["id"])`.
  - Give `SubmitAudioJobRequest.api_name` this description: "Provider for the analysis stage. Without it, analysis is skipped even when perform_analysis is true, and the job records analysis_skipped."
- [ ] **Step 4:** run the new tests and the whole file. Expected: all pass, apart from the pre-existing results recorded on `dev`.
- [ ] **Step 4b: regenerate the API contract.** The OpenAPI fingerprint hashes field descriptions, and Step 3 changes the `api_name` and `perform_analysis` descriptions on `SubmitAudioJobRequest`.
  - Run `PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --fingerprint apps/packages/api-types/openapi.fingerprint.json --paths apps/packages/api-types/paths.d.ts`. Expected: only `openapi.fingerprint.json` changes, with a new `sha256`.
  - Then run `... --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts`. Expected: exit 0.
  - Add the fingerprint file to this commit. (Flagged by the audio pipeline plan's review, M13.)
- [ ] **Step 5:** commit `fix(audio-jobs): analysis runs whether or not chunking is on; a skipped analysis is logged (TASK-13845)`.

### Task 2: `fail_job(quarantine=False)` and rate-limit backoff (TASK-13846)

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/manager.py` (`fail_job`: new keyword `quarantine: bool = True`; both backends' threshold computation)
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py` (`_StageFailure` gains `backoff_seconds` and `quarantine`; the final `fail_job` call passes them)
- Test: `tldw_Server_API/tests/Jobs/` (a `fail_job(quarantine=False)` test, SQLite; plus Postgres where the suite has a backend fixture) and `test_audio_worker_smoke.py`

- [ ] **Step 1: failing tests**
  - **`fail_job`** called twice with the same `error_code` and `quarantine=False` leaves the job `queued` with `retry_count == 2`. The same two calls with the default quarantine it, as today; that is the control.
    - Acquire the job before each `fail_job`, since it only moves a `processing` row. Reset `available_at` between rounds, and use `backoff_seconds >= 1`: in test mode, `backoff_seconds=0` disables quarantine for the control too.
    - SQLite in `tests/Jobs`. Postgres in `tests/Jobs/test_jobs_quarantine_postgres.py`, with its `jobs_pg_dsn` fixture. (Review M4.)
  - **The audio worker:** a rate-limited analysis is rescheduled with `available_at` not NULL and at least 25 s in the future. After two rate limits the job is still not quarantined.
    - Drive it as the existing `test_audio_analyze_rate_limit_still_retries` does, and set `available_at` to now between attempts.
    - End each wait on `stored(jm) or status in {failed, quarantined}`, so a RED run doesn't idle out its 5 s timeout.
- [ ] **Step 2:** run them. Expected:
  - `TypeError` (unknown keyword `quarantine`);
  - `available_at` NULL: in test mode the 1 s default backoff becomes no delay (Review M3);
  - quarantined after the second rate limit.
- [ ] **Step 3: implement**
  - `fail_job(..., quarantine: bool = True)`. Where each backend computes `thresh`, add `if not quarantine: thresh = 10**9`, the device the test-mode path already uses.
    - On SQLite, put it after the test-mode block (`manager.py` ~8863-8868), which resets `thresh` when `JOBS_QUARANTINE_THRESHOLD` is set. (Review M5.)
    - Document the keyword in the docstring: retries still stop at `max_retries`.
  - `_StageFailure.__init__(..., backoff_seconds: int = 1, quarantine: bool = True)`.
  - The analysis branch raises `_StageFailure(..., retryable=True, backoff_seconds=30 if code == "rate_limit" else 5, quarantine=False)` for the retryable codes.
  - The final `jm.fail_job` passes `backoff_seconds=getattr(failure, "backoff_seconds", 1)` and `quarantine=getattr(failure, "quarantine", True)`.
- [ ] **Step 4:** run the new tests, `tests/Jobs` (SQLite; Postgres if reachable), and the audio worker file.
- [ ] **Step 5:** commit `fix(audio-jobs): a rate-limited analysis waits and is not quarantined (TASK-13846)`.

### Task 3: a failed analysis keeps the transcript (TASK-13847)

**Files:** `audio_jobs_worker.py` (the `audio_analyze` branch); `test_audio_worker_smoke.py`.

- [ ] **Step 1: failing tests.** They replace the expectations of the existing setup-failure tests, which today assert `status == "failed"`.
  - **`test_audio_analyze_provider_setup_failure_keeps_the_transcript`**, parametrized over authentication, configuration, missing model and unavailable adapter:
    - the job hands over to `audio_store`;
    - the store payload keeps `text`, has `analysis_error == {"code": <code>, "message": ...}`, and has no `analysis`;
    - the analyze job is `completed` with `retry_count == 0`;
    - with Decision 3, the analyze job's `result["analysis_error"]["code"] == <code>`.
  - **`test_audio_analyze_rate_limited_on_the_last_attempt_keeps_the_transcript`:** set the job's `retry_count` to `max_retries` before the worker runs it. Expect a hand-off to `audio_store` with `analysis_error.code == "rate_limit"`.
  - **With Decision 1:** `test_audio_analyze_quota_refusal_keeps_the_transcript` replaces `test_audio_analyze_quota_refusal_fails_the_job_without_retry`. The message is `"LLM quota exceeded (llm_tokens_month)"`.
  - **With Decision 2:** `test_audio_analyze_empty_transcript_keeps_it` replaces `test_audio_analyze_empty_transcript_fails_without_retry`, with `analysis_error.code == "empty_transcript"`.
  - **An `Error:` reply on the last attempt** is stored with `analysis_error.code == "provider_failure"`, and there is no `analysis` key.
  - End each wait on `stored(jm) or status in {failed, quarantined}`.
- [ ] **Step 2:** run them. Expected: the old behaviour (`failed`, no `audio_store`).
- [ ] **Step 3: implement.** In the `audio_analyze` branch, set the code before the `try`, so the success path cannot reach an unbound `analysis` (Review M2):

```python
                    code: str | None = None
                    message = ""
                    if not api_name or api_name.lower() == "none":
                        updated_payload["analysis_skipped"] = "no api_name"
                        logger.info("Audio job {}: analysis skipped, no api_name", job["id"])
                    elif not text.strip():
                        code, message = "empty_transcript", "empty transcript: nothing to analyze"  # Decision 2
                    else:
                        try:
                            with llm_call_root(owner):
                                analysis = await asyncio.to_thread(
                                    analyze, api_name, text, None, input_is_literal_text=True, raise_on_error=True
                                )
                        except SummaryProviderError as exc:
                            code, message = exc.code, str(exc)
                        except LLMQuotaExceededError as exc:  # Decision 1; from app.core.exceptions
                            code = "llm_quota_exceeded"
                            message = f"LLM quota exceeded ({(exc.detail or {}).get('category', 'llm')})"
                        else:
                            if isinstance(analysis, str) and analysis.startswith("Error:"):
                                code, message = "provider_failure", analysis
                            else:
                                updated_payload["analysis"] = analysis
                    if code is not None:
                        attempts_left = int(job.get("retry_count") or 0) < int(job.get("max_retries") or 0)
                        if code in _RETRYABLE_ANALYSIS_CODES and attempts_left:
                            raise _StageFailure(message, failure_code=code, retryable=True,
                                                backoff_seconds=30 if code == "rate_limit" else 5, quarantine=False)
                        # For good: keep the transcript and record why there is no analysis.
                        updated_payload["analysis_error"] = {"code": code, "message": message}
                        logger.warning("Audio job {}: analysis failed ({}); storing the transcript without it", job["id"], code)
                    next_type = "audio_store"
```

  - Check what attributes `LLMQuotaExceededError` actually carries (category, and `detail` versus attributes) and use those. The existing quota test builds it with `category=...`.
  - **With Decision 3:** the hand-off branch's `jm.complete_job(...)` passes `result={"analysis_error": updated_payload["analysis_error"]}` when the key is present. It passes no result today.
  - **Without Decision 1 or 2:** leave those two cases failing the job as today (a `_StageFailure`, not retryable).
- [ ] **Step 4:** run the audio worker file, `tests/Audio/test_audio_jobs_worker_helper.py` and `tests/Usage/test_llm_roots_non_sdk.py`, which also drives `audio_analyze`. (Review M9.)
- [ ] **Step 5:** commit `fix(audio-jobs): a failed analysis stores the transcript with the failure recorded (TASK-13847)`.

### Task 4: close-out

- [ ] **Docs** (Review I1):
  - **`Docs/API-related/Audio_Jobs_API.md:41`** describes the behaviour this plan replaces: analysis failures fail the stage, `llm_quota_exceeded`, and quarantine after two identical failures. Rewrite it to say:
    - setup failures, and retries that run out, store the transcript with `analysis_error`;
    - retries back off 30 s or 5 s and never quarantine;
    - the outcome of Decisions 1 and 2.
  - On line 38, say that chunking and analysis are each optional.
  - If TASK-13844 (`plan_summary_error_codes.md`) has already landed, keep two things from its wording:
    - `request_rejected` and `provider_quota_exhausted` among the failures that store the transcript without a retry;
    - its sentence that a provider switched off with `LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER>=0` shows up as `configuration`.

    (Its review, I2.)
  - **`Docs/Code_Documentation/Jobs_Module.md`** (~166 and ~497): document `fail_job(quarantine=False)`, and correct the stated quarantine default from 3 to 2, which is the code's.
  - Both pages are mirrored: run `bash Helper_Scripts/refresh_docs_published.sh` and commit the `Docs/Published` diff.
  - **With Decision 1:** `Docs/Operations/Usage_Quotas.md:95` (not mirrored) says an audio analysis job stores its transcript with `analysis_error.code = "llm_quota_exceeded"` and is not retried.
- [ ] **Checks:** run `tests/Docs` and `tldw_Server_API/tests/CI/test_test_quality_ratchet.py`. Since #3243, dev enforces that ratchet: new tests must not be status-only or tautological. Then run bandit `-ll` on `audio_jobs_worker.py`, `audio_transcribe_gpu_worker.py` and `manager.py` (DoD #4). (Review M9.)
- [ ] **Merge** through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset.
- [ ] **Close** each of TASK-13845, 13846 and 13847 with backlog-py: notes with the RED and GREEN counts, ACs, DoD, final summary and status Done.
- [ ] **Commit** `docs(audio-jobs): analysis failures keep the transcript; close TASK-13845/13846/13847`.
- [ ] **PR** to `dev` with the owner's ADR-004 answer recorded, then a clean-session review before merge.

## Out of scope

- The GPU worker's older hand-off order (complete, then create, with no idempotency key). It is noted here; TASK-13850/13851 own the hand-off.
- Reading the provider's `retry-after`. `SummaryProviderError` does not carry it today; the fixed backoff meets AC1 ("a meaningful backoff").
