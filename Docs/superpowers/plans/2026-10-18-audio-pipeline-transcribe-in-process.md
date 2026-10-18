# Audio Jobs Pipeline Transcribes In-Process Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A job submitted to `POST /api/v1/audio/jobs/submit` gets past conversion. The in-process audio worker transcribes it unless a GPU transcription worker is configured, and a stage that cannot queue the next one fails visibly (TASK-13848).

**Architecture:**
- **Routing (`app/services/audio_jobs_worker.py`).** `_next_stage_queue(next_type)` puts every next stage on the worker's own `default` queue. The exception is `audio_transcribe` when the `AUDIO_JOBS_GPU_TRANSCRIBE` flag is set: it then goes to `transcribe`, which the GPU worker (`app/services/audio_transcribe_gpu_worker.py`) reads.
- **Allowlist.** `transcribe` is registered for the `audio` domain in `JobManager.DOMAIN_ALLOWED_QUEUES`.
- **Hand-off order.** The worker queues the next stage before completing the current one. It uses the idempotency key `audio-stage:<current job id>:<next type>`, so a re-run stage reuses what it already queued.
- **A failed hand-off fails the current job** with `error_code` `next_stage_enqueue_failed`, and logs a warning. The failure is permanent only when the queue itself is refused. A quota refusal or a transient error is retried.
- **Fair-share.** Because the running stage is still `processing` when its successor is queued, the per-user fair-share limit (`JOBS_MAX_PER_USER`) would count it against its own hand-off. `create_job` and `admit_job` therefore gain `fair_share_exempt: bool = False`, which skips that limit. The worker passes it for hand-offs: the next stage replaces the running job and adds nothing to the owner's concurrency.
- **Unchanged:** the GPU worker.

**Tech Stack:** Python, asyncio, the Jobs module, pytest.

**Spec:** `backlog/tasks/task-13848 - Audio-jobs-pipeline-stops-after-conversion-the-transcribe-stage-s-queue-is-not-allowed.md`, its three acceptance criteria, plus the note appended from PR #3259: any queue this task routes the pipeline through must be in `tts_jobs_queue()`'s refusal set.

## What was verified (on `origin/dev` `b88a983439`)

- **Every `/audio/jobs` job stops after conversion.** `audio_convert` completes. Creating `audio_transcribe` on `transcribe` then raises `Queue 'transcribe' not allowed for domain 'audio'`. The worker logs `Audio worker loop error`, nothing further runs, and the convert job reads `completed`.
- **Registering the queue alone would not help.** Only the GPU worker container reads `transcribe`. That is `Dockerfiles/Dockerfile.audio_gpu_worker`, a published image, which sets `AUDIO_JOBS_WORKER_ENABLED=false`. The in-process worker, which has a working `audio_transcribe` handler, reads only `default`.
- **Creating the next stage before completing the current one needs the fair-share exemption.** The plan review probed this with `JOBS_MAX_PER_USER=1`, without the exemption:
  - the hand-off was refused by `BadRequestError` (a `ValueError`) and the job failed;
  - on `dev`'s order (complete, then create) the slot was free.
  - Fair-share is on only when `JOBS_MAX_PER_USER` or `JOBS_MAX_PER_ORG` is set (`_fair_share_enabled`), so default deployments never hit it.
- **The existing tests miss all of this.** They create `audio_transcribe` jobs on `default` by hand, and the opt-in pipeline smoke test ends in `assert True`.

## Global Constraints

- **Branch:** `fix/audio-pipeline-transcribe-in-process` from `origin/dev`. The PR targets `dev`.
- **Routing, verbatim:** `_next_stage_queue(next_type)` returns `"transcribe"` only when `next_type == "audio_transcribe"` and `AUDIO_JOBS_GPU_TRANSCRIBE` is one of `true, 1, yes, y, on` (any case). Otherwise it returns `"default"`.
  - This is the owner's routing decision for this task: in-process by default, GPU by opt-in flag.
  - The flag is a boolean, not a queue name. That keeps the queue set fixed, so `tts_jobs_queue()`'s refusal of `default` and `transcribe` (PR #3259) stays complete, as the TASK-13848 note asks.
- **Allowlist:** `"audio": ("tts", "transcribe")`.
- **Hand-off:**
  - `create_job(..., idempotency_key=f"audio-stage:{job['id']}:{next_type}", fair_share_exempt=True)` runs before `complete_job`.
  - On any exception the worker logs a warning. The current job then fails through the existing `_StageFailure` with `failure_code="next_stage_enqueue_failed"` and `retryable=next_queue in jm._get_allowed_queues(DOMAIN)`, so only a refused queue is permanent.
  - The terminal-stage branch (`audio_store`, no next type) is unchanged.
- **`JobManager`:**
  - `admit_job` and `create_job` gain `fair_share_exempt: bool = False`, documented in `admit_job`'s Args.
  - It skips only the fair-share block in `admit_job` (`if owner_user_id and _fair_share_enabled() and not fair_share_exempt:`). The second fair-share check, in `retry_failed_job_admission`, is untouched.
  - No other caller passes it.
- **Line-number baseline:** this changes `Jobs/manager.py`, whose conditional tenant predicates are recorded by line in `Helper_Scripts/ci/scope_predicate_baseline.txt`. Regenerate it with `Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline`, and check that the diff only moves existing entries. Without that, `backend-required` fails, as it did on PR #3259.
- **The GPU worker and the TTS worker are not changed.**
- **Test first:** the RED run happens before the source change.
- **Docs:**
  - `Docs/API-related/Audio_Jobs_API.md` and `Docs/Operations/Env_Vars.md`.
  - Their `Docs/Published` mirrors, via `bash Helper_Scripts/refresh_docs_published.sh`.
- **Plan file:** this plan has had its trailing whitespace stripped, including the blank context lines inside its diffs. `git apply` accepts empty context lines, and the `trailing-whitespace` hook would reject single spaces.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit id `TASK-13848`, `git add -A backlog/tasks`.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never `git stash`.
- **Out of scope:**
  - **`AUDIO_JOBS_GPU_TRANSCRIBE` set where no GPU worker reads the same Jobs database.** Transcription waits on `transcribe`, as documented.
  - **Jobs whose `audio_convert` completed before the deploy.** They have no next stage to resume and must be re-submitted.
  - **The GPU worker's own hand-back** queues its next stage without an idempotency key or the fair-share exemption. The GPU worker is unchanged.
  - **Long in-process transcription can outlive its lease.** Transcription runs without lease renewal, and a long file can exceed `JOBS_LEASE_SECONDS` (120). With more than one audio worker process, another worker may recover the lease mid-run and transcribe again. The idempotency key prevents a duplicate next stage. Other stages already behave this way; this plan only makes transcription an in-process stage by default.
  - **`perform_chunking: false` skips analysis.** That is TASK-13845.

## Review Focus

1. **A stage re-run after it already queued the next one** (its lease lost between `create_job` and `complete_job`) reuses that next stage instead of queuing a second. The Jobs module scopes idempotency keys to (domain, queue, job type, key) and returns the existing row on a repeat (`tests/Jobs/parity/*idempotent_create*`). Test: `test_audio_pipeline_transcribes_in_process_by_default` asserts the key on the queued stage.
2. **A refused queue fails the stage at once, without retry. A transient enqueue error is retried.** Tests:
   - `test_a_failed_stage_hand_off_fails_the_job_visibly` empties `DOMAIN_ALLOWED_QUEUES` with the flag on, and clears `JOBS_ALLOWED_QUEUES[_AUDIO]` from the environment.
   - `test_a_transient_hand_off_failure_is_retried` makes `create_job` raise `RuntimeError` for the hand-off.
3. **A per-user fair-share limit of 1 still lets a job through every stage**, because the running stage does not count against its own hand-off. Test: `test_audio_pipeline_runs_under_a_per_user_job_limit_of_one`. In a mutation run with the exemption switched off, only this test fails.
4. **With the flag on, the in-process worker leaves the transcription job alone.** It waits `queued` on `transcribe`, and no STT call is made. Test: `test_audio_pipeline_routes_transcription_to_the_gpu_queue_when_enabled`.
5. **TTS isolation still holds.** `tts_jobs_queue()` refuses `transcribe`, so registering it for `audio` does not let a TTS job land where the GPU worker reads. Test: `test_tts_jobs_queue_refuses_the_audio_pipeline_queues`, which is in the Step 7 sweep.

---

### Task 1: The pipeline transcribes in-process, and a failed hand-off fails visibly

**Files:**
- Modify: `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py`. Add a `pipeline_worker` fixture and five tests at the end.
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py`. Add `_next_stage_queue` after `_StageFailure`, and rewrite the success branch of the stage hand-off.
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`. Change the `"audio"` allowlist entry, add `fair_share_exempt` to `admit_job` and `create_job`, and add it to `admit_job`'s fair-share condition.
- Modify: `Helper_Scripts/ci/scope_predicate_baseline.txt` (regenerated).
- Modify: `Docs/API-related/Audio_Jobs_API.md`, `Docs/Operations/Env_Vars.md`, and their mirrors `Docs/Published/API-related/Audio_Jobs_API.md` and `Docs/Published/Env_Vars.md`.
- Modify: `backlog/tasks/task-13848*.md` (backlog-py only).
- Add: this plan, `Docs/superpowers/plans/2026-10-18-audio-pipeline-transcribe-in-process.md`.

**Interfaces:**
- Produces: `JobManager.create_job(..., fair_share_exempt: bool = False)` and `JobManager.admit_job(..., fair_share_exempt: bool = False)`.
- Changes:
  - The `audio_transcribe` job goes on `default`, or on `transcribe` with the flag set. Before, it went on `transcribe` and was refused.
  - A failed hand-off now fails the current job.

Set once for every step: `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` and `export TLDW_TEST_NO_DOCKER=1`.

- [ ] **Step 1: Branch**

```bash
git fetch -q origin dev
git switch -c fix/audio-pipeline-transcribe-in-process origin/dev
```

Copy this plan to `Docs/superpowers/plans/2026-10-18-audio-pipeline-transcribe-in-process.md`.

- [ ] **Step 2: Write the failing tests**

Apply exactly this diff with `git apply`:

```diff
diff --git a/tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py b/tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
index 9fb8e91273..1e016c5e23 100644
--- a/tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
+++ b/tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
@@ -957,3 +957,131 @@ async def test_audio_analyze_provider_failure_still_retries(analyze_worker):
     assert job.get("status") != "failed"
     assert job.get("error_code") == "provider_failure"
     assert len(adapter.requests) >= 1
+
+
+# --- the audio_convert -> audio_transcribe hand-off (TASK-13848) ---
+
+
+@pytest.fixture
+def pipeline_worker(monkeypatch, tmp_path):
+    """The audio worker with quota stubs, a fresh jobs DB, and conversion and STT stubbed out."""
+    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs_pipeline.db"))
+    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")
+    monkeypatch.delenv("AUDIO_JOBS_GPU_TRANSCRIBE", raising=False)
+
+    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
+    import tldw_Server_API.app.services.audio_jobs_worker as worker
+    from tldw_Server_API.app.core.Jobs.manager import JobManager
+
+    async def _unlimited(user_id: int):
+        return {"concurrent_jobs": None}
+
+    async def _yes(user_id: int):
+        return True, ""
+
+    async def _noop(user_id: int):
+        return None
+
+    monkeypatch.setattr(worker, "get_limits_for_user", _unlimited)
+    monkeypatch.setattr(worker, "can_start_job", _yes)
+    monkeypatch.setattr(worker, "increment_jobs_started", _noop)
+    monkeypatch.setattr(worker, "finish_job", _noop)
+
+    stt_calls = []
+
+    def _fake_stt(wav_path, model, *args, **kwargs):
+        stt_calls.append(wav_path)
+        return {"text": "hello pipeline", "segments": [{"Text": "hello pipeline"}]}
+
+    monkeypatch.setattr(atlib, "convert_to_wav", lambda path, *a, **k: str(path) + ".wav")
+    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _fake_stt)
+    return worker, JobManager(), stt_calls
+
+
+def _convert_job(jm) -> int:
+    row = jm.create_job(
+        domain="audio", queue="default", job_type="audio_convert",
+        payload={"local_path": "/tmp/pipeline.mp3", "model": "whisper-1", "perform_chunking": False, "perform_analysis": False},
+        owner_user_id="7",
+    )
+    return int(row["id"])
+
+
+def _jobs_of(jm, job_type: str):
+    return jm.list_jobs(domain="audio", job_type=job_type, owner_user_id="7")
+
+
+@pytest.mark.unit
+async def test_audio_pipeline_transcribes_in_process_by_default(pipeline_worker):
+    worker, jm, stt_calls = pipeline_worker
+    convert_id = _convert_job(jm)
+    await _run_worker_until(worker, lambda: any(j.get("status") == "completed" for j in _jobs_of(jm, "audio_store")))
+    transcribe = _jobs_of(jm, "audio_transcribe")
+    assert [j["queue"] for j in transcribe] == ["default"]
+    # The key that lets a re-run stage reuse the next stage it already queued.
+    assert transcribe[0]["idempotency_key"] == f"audio-stage:{convert_id}:audio_transcribe"
+    assert [j["status"] for j in _jobs_of(jm, "audio_store")] == ["completed"]
+    assert len(stt_calls) == 1
+
+
+@pytest.mark.unit
+async def test_audio_pipeline_runs_under_a_per_user_job_limit_of_one(pipeline_worker, monkeypatch):
+    # The running stage must not count against its own hand-off under fair-share.
+    worker, jm, stt_calls = pipeline_worker
+    monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
+    _convert_job(jm)
+    await _run_worker_until(worker, lambda: any(j.get("status") == "completed" for j in _jobs_of(jm, "audio_store")))
+    assert [j["status"] for j in _jobs_of(jm, "audio_store")] == ["completed"]
+    assert len(stt_calls) == 1
+
+
+@pytest.mark.unit
+async def test_audio_pipeline_routes_transcription_to_the_gpu_queue_when_enabled(pipeline_worker, monkeypatch):
+    worker, jm, stt_calls = pipeline_worker
+    monkeypatch.setenv("AUDIO_JOBS_GPU_TRANSCRIBE", "true")
+    convert_id = _convert_job(jm)
+    await _run_worker_until(
+        worker,
+        lambda: bool(_jobs_of(jm, "audio_transcribe")) or (jm.get_job(convert_id) or {}).get("status") == "completed",
+    )
+    assert [(j["queue"], j["status"]) for j in _jobs_of(jm, "audio_transcribe")] == [("transcribe", "queued")]
+    assert stt_calls == []
+
+
+@pytest.mark.unit
+async def test_a_failed_stage_hand_off_fails_the_job_visibly(pipeline_worker, monkeypatch):
+    from tldw_Server_API.app.core.Jobs.manager import JobManager
+
+    worker, jm, stt_calls = pipeline_worker
+    monkeypatch.setenv("AUDIO_JOBS_GPU_TRANSCRIBE", "true")
+    monkeypatch.delenv("JOBS_ALLOWED_QUEUES", raising=False)
+    monkeypatch.delenv("JOBS_ALLOWED_QUEUES_AUDIO", raising=False)
+    monkeypatch.setattr(JobManager, "DOMAIN_ALLOWED_QUEUES", {})  # "transcribe" is now refused
+    convert_id = _convert_job(jm)
+    await _run_worker_until(worker, lambda: (jm.get_job(convert_id) or {}).get("status") in {"completed", "failed"})
+    job = jm.get_job(convert_id) or {}
+    assert job.get("status") == "failed"
+    assert job.get("error_code") == "next_stage_enqueue_failed"
+    assert int(job.get("retry_count") or 0) == 0
+    assert _jobs_of(jm, "audio_transcribe") == []
+
+
+@pytest.mark.unit
+async def test_a_transient_hand_off_failure_is_retried(pipeline_worker, monkeypatch):
+    from tldw_Server_API.app.core.Jobs.manager import JobManager
+
+    worker, jm, stt_calls = pipeline_worker
+    original_create = JobManager.create_job
+
+    def _flaky_create(self, **kwargs):
+        if kwargs.get("job_type") == "audio_transcribe":
+            raise RuntimeError("database is locked")
+        return original_create(self, **kwargs)
+
+    monkeypatch.setattr(JobManager, "create_job", _flaky_create)
+    convert_id = _convert_job(jm)
+    await _run_worker_until(worker, lambda: int((jm.get_job(convert_id) or {}).get("retry_count") or 0) >= 1)
+    job = jm.get_job(convert_id) or {}
+    assert int(job.get("retry_count") or 0) >= 1
+    assert job.get("status") != "completed"
+    assert job.get("error_code") == "next_stage_enqueue_failed"
```

- [ ] **Step 3: Run them to see them fail**

Run: `$PY -m pytest -p no:cacheprovider -p no:randomly -q --tb=line tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k "pipeline or hand_off"`

Expected: `5 failed, 1 skipped, 22 deselected`. The skip is the opt-in `test_audio_worker_pipeline_smoke_skip_if_missing`. Measured on `origin/dev` `b88a983439`:
- `test_audio_pipeline_transcribes_in_process_by_default`: `AssertionError: assert [] == ['default']`. No transcription job is ever created.
- `test_audio_pipeline_runs_under_a_per_user_job_limit_of_one`: `AssertionError: assert [] == ['completed']`. Same cause.
- `test_audio_pipeline_routes_transcription_to_the_gpu_queue_when_enabled`: `AssertionError: assert [] == [('transcribe', 'queued')]`. The queue is refused.
- `test_a_failed_stage_hand_off_fails_the_job_visibly`: `AssertionError: assert 'completed' == 'failed'`.
- `test_a_transient_hand_off_failure_is_retried`: `AssertionError: assert 0 >= 1`. The convert job completes with nothing after it.

- [ ] **Step 4: Apply the source change and regenerate the line baseline**

Apply exactly this diff, then regenerate the baseline:

```diff
diff --git a/tldw_Server_API/app/core/Jobs/manager.py b/tldw_Server_API/app/core/Jobs/manager.py
index 5842e855a3..c1e5061a1c 100644
--- a/tldw_Server_API/app/core/Jobs/manager.py
+++ b/tldw_Server_API/app/core/Jobs/manager.py
@@ -683,8 +683,8 @@ class JobManager:
         "writing": ("writing-review", "writing-ai"),
         "scheduled_tasks": ("scheduled-tasks",),
         "admin_webhooks": ("delivery",),
-        # Long-form TTS jobs, kept off "default", which the audio pipeline worker reads unfiltered.
-        "audio": ("tts",),
+        # Long-form TTS jobs (off "default", read unfiltered by the pipeline); GPU transcription.
+        "audio": ("tts", "transcribe"),
     }

     # --- Shutdown/acquisition gate (process-wide) ---
@@ -3514,6 +3514,7 @@ class JobManager:
         owner_scope_admission: OwnerScopeAdmissionPolicy | None = None,
         expired_lease_policy: ExpiredLeasePolicy = ExpiredLeasePolicy.CONSUME_RETRY,
         quarantine_threshold: int | None = None,
+        fair_share_exempt: bool = False,
     ) -> AdmissionResult:
         """Run the complete admission pipeline and return its typed outcome.

@@ -3532,6 +3533,8 @@ class JobManager:
             owner_scope_admission: Optional transactional owner/job-scope limits.
             expired_lease_policy: Behavior when a processing lease expires.
             quarantine_threshold: Optional consecutive-failure quarantine limit.
+            fair_share_exempt: Skip the per-user fair-share limit, for a next stage queued by
+                a job that is still running and is about to complete (it replaces that job).

         ``create_job`` delegates here and maps only the final typed outcome to
         its legacy dict/exception contract.
@@ -3606,7 +3609,7 @@ class JobManager:
             raise error  # noqa: TRY003

         # Fair-share scheduling: enforce per-user concurrency limits and adjust priority
-        if owner_user_id and _fair_share_enabled():
+        if owner_user_id and _fair_share_enabled() and not fair_share_exempt:
             try:
                 scheduler = _get_fair_share()
                 active_count = self._count_active_jobs_for_user(owner_user_id)
@@ -3972,6 +3975,7 @@ class JobManager:
         owner_scope_admission: OwnerScopeAdmissionPolicy | None = None,
         expired_lease_policy: ExpiredLeasePolicy = ExpiredLeasePolicy.CONSUME_RETRY,
         quarantine_threshold: int | None = None,
+        fair_share_exempt: bool = False,
     ) -> dict[str, Any]:
         """Create a job using the typed admission pipeline and legacy mapping."""

@@ -3992,6 +3996,7 @@ class JobManager:
             owner_scope_admission=owner_scope_admission,
             expired_lease_policy=expired_lease_policy,
             quarantine_threshold=quarantine_threshold,
+            fair_share_exempt=fair_share_exempt,
         )
         return self._map_admission_result(result)

diff --git a/tldw_Server_API/app/services/audio_jobs_worker.py b/tldw_Server_API/app/services/audio_jobs_worker.py
index 89e3adcb9f..f2a150bfc3 100644
--- a/tldw_Server_API/app/services/audio_jobs_worker.py
+++ b/tldw_Server_API/app/services/audio_jobs_worker.py
@@ -34,6 +34,13 @@ class _StageFailure(RuntimeError):
         self.retryable = retryable


+def _next_stage_queue(next_type: str) -> str:
+    """The queue for the next stage: this worker's own ``default``, except that ``audio_transcribe``
+    goes to the GPU transcription worker's ``transcribe`` queue when AUDIO_JOBS_GPU_TRANSCRIBE is set."""
+    gpu = os.getenv("AUDIO_JOBS_GPU_TRANSCRIBE", "false").lower() in {"true", "1", "yes", "y", "on"}
+    return "transcribe" if next_type == "audio_transcribe" and gpu else "default"
+
+
 async def _handle_audio_transcribe_stage(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
     """
     Handle the `audio_transcribe` stage for a single job payload.
@@ -406,21 +413,38 @@ async def run_audio_jobs_worker(stop_event: asyncio.Event | None = None) -> None
                 msg_err = str(e)
                 failure = e

-            if ok:
-                if next_type:
-                    # Create next stage
-                    jm.complete_job(int(job["id"]), worker_id=worker_id, lease_id=str(job.get("lease_id")), completion_token=str(job.get("lease_id")))
+            if ok and next_type:
+                # Queue the next stage before completing this one, so a failed hand-off fails this job
+                # instead of leaving it completed with nothing after it. The key makes a retried stage
+                # reuse the next stage it already queued. The next stage replaces this running job, so
+                # it is exempt from the per-user fair-share limit, which would count this job against it.
+                next_queue = _next_stage_queue(next_type)
+                try:
                     jm.create_job(
                         domain=DOMAIN,
-                        queue=("transcribe" if next_type == "audio_transcribe" else "default"),
+                        queue=next_queue,
                         job_type=next_type,
                         payload=updated_payload,
                         owner_user_id=str(owner),
                         priority=5,
+                        idempotency_key=f"audio-stage:{job['id']}:{next_type}",
+                        fair_share_exempt=True,
+                    )
+                except Exception as e:  # noqa: BLE001 - any failure to queue the next stage fails this one
+                    ok = False
+                    msg_err = f"could not queue {next_type}: {e}"
+                    logger.warning(f"Audio job {job['id']}: {msg_err}")
+                    # A refused queue fails the same way every time; a quota or a transient error may pass on retry.
+                    failure = _StageFailure(
+                        msg_err,
+                        failure_code="next_stage_enqueue_failed",
+                        retryable=next_queue in jm._get_allowed_queues(DOMAIN),
                     )
                 else:
-                    jm.complete_job(int(job["id"]), result={"payload": updated_payload}, worker_id=worker_id, lease_id=str(job.get("lease_id")), completion_token=str(job.get("lease_id")))
-            else:
+                    jm.complete_job(int(job["id"]), worker_id=worker_id, lease_id=str(job.get("lease_id")), completion_token=str(job.get("lease_id")))
+            elif ok:
+                jm.complete_job(int(job["id"]), result={"payload": updated_payload}, worker_id=worker_id, lease_id=str(job.get("lease_id")), completion_token=str(job.get("lease_id")))
+            if not ok:
                 # An error that says it is not retryable (an LLM quota refusal, a _StageFailure) fails the job now.
                 jm.fail_job(
                     int(job["id"]),
```

```bash
$PY Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline
git diff -U0 Helper_Scripts/ci/scope_predicate_baseline.txt
```

Expected:
- The script prints `Recorded 28 conditional scope predicates.`
- The diff changes 7 lines, all `tldw_Server_API/app/core/Jobs/manager.py:<n>` entries. Each moves down by the number of lines this diff adds above it, and none is added or removed.

- [ ] **Step 5: Run the tests to see them pass**

Run the Step 3 command. Expected: `5 passed, 1 skipped, 22 deselected`.

- [ ] **Step 6: Document it and refresh the mirrors**

Apply this diff to the two source docs only:

```diff
diff --git a/Docs/API-related/Audio_Jobs_API.md b/Docs/API-related/Audio_Jobs_API.md
index c22bb52a05..7fd654d58e 100644
--- a/Docs/API-related/Audio_Jobs_API.md
+++ b/Docs/API-related/Audio_Jobs_API.md
@@ -37,10 +37,12 @@ Base path: `/api/v1/audio`

 - Service: runs the pipeline stages `audio_download → audio_convert → audio_transcribe → audio_chunk → audio_analyze → audio_store`.
 - Each stage runs as its own job in the `audio` domain; the submit response carries only the first stage's id.
+- Every stage, transcription included, runs in this worker on the `default` queue, unless `AUDIO_JOBS_GPU_TRANSCRIBE` sends transcription to the GPU worker (below). A stage that cannot queue the next one fails with `error_code` `next_stage_enqueue_failed` instead of completing with nothing after it.
 - Analysis failures: these fail the `audio_analyze` stage job at once, with the reason in that job row's `error_code`: `missing_model`, `authentication`, `configuration`, `invalid_provider`, `adapter_unavailable`, `llm_quota_exceeded` or `empty_transcript`. A missing API key shows up as `authentication` or `configuration`. Every other provider failure is recorded as `rate_limit` or `provider_failure` and retried, including a provider rejecting the request (an unknown model, a transcript too long for the model). The same failure twice in a row quarantines the job (`JOBS_QUARANTINE_THRESHOLD`, default 2).
 - Env flags:
   - `AUDIO_JOBS_WORKER_ENABLED`: `true|false` (default false)
   - `AUDIO_JOBS_OWNER_STRICT`: enable owner-aware acquisition (default false)
+  - `AUDIO_JOBS_GPU_TRANSCRIBE`: `true` queues `audio_transcribe` on the `transcribe` queue for the GPU worker instead of transcribing in this worker (default false). It is read by this worker, so set it where this worker runs, and only when a GPU worker reads the same Jobs database; otherwise transcription waits for a worker that never comes.
   - `JOBS_LEASE_SECONDS`: lease duration (default 120)

 ### GPU Worker (stub)
@@ -48,7 +50,9 @@ Base path: `/api/v1/audio`
 - A GPU-oriented worker stub is provided to process only the `audio_transcribe` stage on GPU nodes.
 - Location: `tldw_Server_API/app/services/audio_transcribe_gpu_worker.py`
 - Container: `Dockerfiles/Dockerfile.audio_gpu_worker`
-- Behavior: acquires `audio` domain jobs and processes only `audio_transcribe`; other stages are re-queued with a short backoff for CPU workers.
+- Behavior: acquires `audio` domain jobs from the `transcribe` queue and processes only `audio_transcribe`; it then queues the next stage on `default` for the CPU audio worker, which still runs every other stage.
+- To use it, set `AUDIO_JOBS_GPU_TRANSCRIBE=true` where the audio jobs worker runs (the API server, or a standalone `python -m tldw_Server_API.app.services.audio_jobs_worker`). The GPU container must share the Jobs database (`JOBS_DB_URL`) and the filesystem behind the converted WAV files. The `transcribe` queue is registered for the `audio` domain.
+- Upgrading: a deployment that made the GPU worker work by setting `JOBS_ALLOWED_QUEUES_AUDIO=transcribe` must now set `AUDIO_JOBS_GPU_TRANSCRIBE=true` as well, or transcription moves in-process; the allowlist setting is no longer needed.

 ## Quotas & Fairness

diff --git a/Docs/Operations/Env_Vars.md b/Docs/Operations/Env_Vars.md
index 318a71abdb..2f8b812983 100644
--- a/Docs/Operations/Env_Vars.md
+++ b/Docs/Operations/Env_Vars.md
@@ -289,6 +289,7 @@ Notes:
 ## Audio Jobs
 - `AUDIO_JOBS_WORKER_ENABLED`: Enable the in-process Audio Jobs worker (`true|false`, default follows route policy for `audio-jobs`). When true, the worker starts at app startup and polls the Jobs backend for the `audio` domain pipeline stages.
 - `AUDIO_JOBS_OWNER_STRICT`: Enable owner-aware acquisition for fairness across users (`true|false`, default `false`). When enabled, the worker preferentially acquires jobs for owners under their concurrent-job caps.
+- `AUDIO_JOBS_GPU_TRANSCRIBE`: Send the `audio_transcribe` stage to the GPU transcription worker's `transcribe` queue instead of transcribing in the audio jobs worker (`true|false`, default `false`). Set it where the audio jobs worker runs, and only when a GPU worker (`Dockerfiles/Dockerfile.audio_gpu_worker`) reads the same Jobs database.
 - `REDIS_URL`: Optional Redis URL used by Resource Governor when `RG_BACKEND=redis`.

 ## Media Ingest Jobs
```

Then refresh the mirrors and check:

```bash
bash Helper_Scripts/refresh_docs_published.sh
git status --short Docs
```

Expected:
- `Docs/API-related/Audio_Jobs_API.md`, `Docs/Operations/Env_Vars.md`, `Docs/Published/API-related/Audio_Jobs_API.md` and `Docs/Published/Env_Vars.md` are modified.
- This plan shows as untracked under `Docs/superpowers/plans/`.
- Nothing else changes under `Docs`.

- [ ] **Step 7: Run every test file that touches the changed code, plus the AudioJobs, Jobs, Docs and lint tests**

```bash
git grep -l -E "audio_jobs_worker|audio_transcribe_gpu_worker|_get_allowed_queues|DOMAIN_ALLOWED_QUEUES|AUDIO_JOBS_GPU_TRANSCRIBE|tts_jobs_queue|fair_share|JOBS_MAX_PER_USER" -- 'tldw_Server_API/tests/*.py' > "${TMPDIR:-/tmp}/t13848_files.txt"
wc -l < "${TMPDIR:-/tmp}/t13848_files.txt"
RUN_JOBS=1 $PY -m pytest -p no:cacheprovider -q $(cat "${TMPDIR:-/tmp}/t13848_files.txt") tldw_Server_API/tests/AudioJobs tldw_Server_API/tests/Jobs tldw_Server_API/tests/Docs tldw_Server_API/tests/lint
```

How to run it:
- Run it serially, without `-n`. Under xdist the workers collect different test sets, so nothing runs.
- It takes about 16 minutes.
- `tests/lint` is included because of the line baseline.

Expected:
- The `git grep` lists 16 files.
- The run gives `1 failed, 2737 passed, 7 skipped`, measured on the prototype. The failure, `tests/Jobs/test_retry_admission_fixture_registration.py::test_bridge_resolution_preserves_jobs_loop_and_isolation` (`ValueError: Pytest terminal summary report not found`), also fails on unchanged `origin/dev`.

If any other failure appears, run it on unchanged `origin/dev` before going on.

- [ ] **Step 8: Lint and security**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
uvx bandit -q -ll tldw_Server_API/app/services/audio_jobs_worker.py
uvx bandit -q -ll tldw_Server_API/app/core/Jobs/manager.py
```

Expected:
- **Ruff** reports only `test_audio_worker_smoke.py:1:1: I001` and `test_audio_worker_smoke.py:224:5: I001`. Both are on `origin/dev`, at the file's top import block and inside an earlier test; this diff touches neither.
- **Bandit on the worker** prints nothing and exits 0.
- **Bandit on `manager.py`** reports only the existing B608 query build at `manager.py:9789`. It is at `:9784` on `origin/dev`; this diff moves it down.

- [ ] **Step 9: Commit the fix**

```bash
git add tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py Helper_Scripts/ci/scope_predicate_baseline.txt Docs/API-related/Audio_Jobs_API.md Docs/Published/API-related/Audio_Jobs_API.md Docs/Operations/Env_Vars.md Docs/Published/Env_Vars.md
git commit -m "fix(audio): the audio jobs pipeline transcribes in-process, and a failed stage hand-off fails the job (TASK-13848)

Every POST /api/v1/audio/jobs/submit job stopped after conversion: the
worker queued audio_transcribe on 'transcribe', which the audio domain did
not allow, so create_job raised, the worker logged a loop error and the
convert job read completed with nothing after it. Only the GPU worker
container reads 'transcribe' anyway.

Transcription now runs in the in-process worker on 'default';
AUDIO_JOBS_GPU_TRANSCRIBE=true sends it to the GPU worker's 'transcribe'
queue, which is registered for the audio domain. The worker queues the next
stage before completing the current one, under an idempotency key, so a
failed hand-off fails the job with next_stage_enqueue_failed instead of
stranding it (permanently only for a refused queue). A hand-off is exempt
from the per-user fair-share limit (new create_job/admit_job flag), since
the next stage replaces the running job; without that, the running stage
would count against its own hand-off.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 10: Close the task and commit the plan**

The notes below quote the measured results. If Steps 3, 5, 7 or 8 printed different lines, put the actual ones in the notes.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13848 --append-notes "Fixed in app/services/audio_jobs_worker.py: _next_stage_queue() puts every next stage on default, except audio_transcribe on transcribe when AUDIO_JOBS_GPU_TRANSCRIBE is set (the owner's choice: in-process by default, GPU by opt-in flag where the audio jobs worker runs); transcribe is registered for the audio domain. The worker queues the next stage before completing the current one, with idempotency key audio-stage:<job id>:<next type>; a failed enqueue logs a warning and fails the current job with next_stage_enqueue_failed, permanently only when the queue is refused (a quota or transient error is retried). JobManager.create_job/admit_job gained fair_share_exempt (default False), which the hand-off passes: the next stage replaces the running job, which would otherwise count against its own hand-off under JOBS_MAX_PER_USER (the plan review's probe: with a limit of 1 the job failed). The scope-predicate line baseline was regenerated (7 manager.py entries moved, none added). The flag is a boolean rather than a queue name, so tts_jobs_queue()'s refusal of default and transcribe stays complete. Tests: RED 5 failed, 1 skipped on dev, GREEN 5 passed, 1 skipped; with the exemption switched off only test_audio_pipeline_runs_under_a_per_user_job_limit_of_one fails; every test file touching the changed code plus tests/AudioJobs, tests/Jobs, tests/Docs and tests/lint 1 failed, 2737 passed, 7 skipped. Ruff: only the two I001 in test_audio_worker_smoke.py that dev has. Bandit -ll: none on the worker; only the pre-existing B608 in Jobs/manager.py. Docs/API-related/Audio_Jobs_API.md and Docs/Operations/Env_Vars.md document the routing, the flag, the GPU prerequisites, the upgrade step and next_stage_enqueue_failed (mirrored). Upgrade: a GPU deployment that relied on JOBS_ALLOWED_QUEUES_AUDIO=transcribe must now set AUDIO_JOBS_GPU_TRANSCRIBE=true. Jobs whose audio_convert completed before the deploy have no next stage and must be re-submitted." --check-ac 1 --check-ac 2 --check-ac 3 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "The audio jobs pipeline transcribes in the in-process worker by default; AUDIO_JOBS_GPU_TRANSCRIBE=true routes transcription to the GPU worker's transcribe queue, now registered for the audio domain. Each stage queues the next before completing, under an idempotency key and exempt from per-user fair-share, so a failed hand-off fails the job with next_stage_enqueue_failed instead of stranding it." --status Done
git add -A backlog/tasks
git add Docs/superpowers/plans/2026-10-18-audio-pipeline-transcribe-in-process.md
git commit -m "docs(audio): plan for in-process transcription in the audio jobs pipeline; close TASK-13848

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: `git show --stat HEAD` lists the task file and the plan only.
