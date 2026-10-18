# TTS and Audio Job Workers Keep to Their Own Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Long-form TTS jobs and the audio pipeline stop taking each other's jobs. TTS jobs get their own `tts` queue, and the TTS worker takes only `tts_longform` jobs (TASK-13641).

**Architecture:** Both kinds of job live in the Jobs `audio` domain.
- `POST /api/v1/audio/speech/jobs` (`app/api/v1/endpoints/audio/audio_tts.py`) enqueued `tts_longform` jobs on `default`.
- The in-process audio pipeline worker (`app/services/audio_jobs_worker.py`) reads `default` with no job-type filter. It has one queue and six stage types, and `acquire_next_job` filters on a single type.
- The standalone TTS worker (`app/core/TTS/tts_jobs_worker.py`) also read `default` with no filter.

The fix:
1. One function, `tts_jobs_queue()` in `app/core/TTS/utils.py`, gives the queue for long-form TTS jobs: `TTS_JOBS_QUEUE`, default `tts`. Both modules already import that file. The endpoint enqueues there and the TTS worker reads there. It refuses `default` and `transcribe`, the queues the audio pipeline workers read unfiltered, so an old `TTS_JOBS_QUEUE=default` fails loudly instead of reviving the collision or stranding jobs.
2. `tts` is registered for the `audio` domain in `JobManager.DOMAIN_ALLOWED_QUEUES` (`app/core/Jobs/manager.py`). Without that, `create_job` refuses the queue.
3. The TTS worker passes `job_type=TTS_JOB_TYPE` to `WorkerSDK.run`, so it takes only TTS jobs even when pointed at a shared queue.

The audio pipeline worker is unchanged: with TTS jobs off `default`, it no longer sees them.

**Tech Stack:** Python, asyncio, the Jobs module (`JobManager`, `WorkerSDK`), pytest.

**Spec:** `backlog/tasks/task-13641 - TTS-and-audio-job-workers-may-take-each-other-s-jobs.md` (no design doc: a bounded bug fix). Its one acceptance criterion: "Verify whether both workers can run against the same queue; if so, each acquires only its own job types (separate queues or a job_type filter), with a test".

## What was verified (on `origin/dev` `f456e5c9bc`)

- **They do share a queue, and the collision does not need the TTS worker.**
  - A job submitted to `POST /api/v1/audio/speech/jobs` while the in-process audio worker runs is taken by that worker within half a second.
  - The worker fails it as `Unknown job_type: tts_longform` (retryable), and it ends `quarantined`.
  - The in-process worker starts when `AUDIO_JOBS_WORKER_ENABLED` is true (`app/services/startup_content_jobs_pollers.py`). Test: `test_audio_jobs_worker_leaves_a_submitted_speech_job_alone`.
- **The TTS worker, reading `default`, takes an `audio_convert` job and fails it** as an unsupported job type. Test: `test_tts_worker_takes_only_tts_jobs_from_its_queue`.
- **Nothing in the repository starts the TTS worker.** It runs only as `python -m tldw_Server_API.app.core.TTS.tts_jobs_worker`; there is no lifespan hook, Dockerfile or compose service for it.
- **The audio GPU worker reads only the `transcribe` queue**, so it never meets a TTS job.

## Global Constraints

- **Branch:** `fix/tts-audio-job-isolation` from `origin/dev`; the PR targets `dev`.
- **Queue, verbatim:** `tts_jobs_queue()` returns `(os.getenv("TTS_JOBS_QUEUE") or "").strip() or "tts"`, and raises `ValueError` for `default` and `transcribe`.
  - Those are the queues `audio_jobs_worker.py` and `audio_transcribe_gpu_worker.py` read without a job-type filter.
  - The worker hits the guard at startup and the endpoint as a 500.
  - The upgrade case it catches: an operator kept `TTS_JOBS_QUEUE=default` on the standalone worker, the value the old docstring named. Without the guard, the endpoint would write to `tts` while the worker read `default`, and every job would stay queued with nothing logged.
  - It is the only place the queue name is chosen.
  - `DOMAIN_ALLOWED_QUEUES` gains exactly `"audio": ("tts",)`.
  - A custom `TTS_JOBS_QUEUE` must be registered in `JOBS_ALLOWED_QUEUES_AUDIO`, the existing mechanism (`JobManager._get_allowed_queues`). Unregistered, the endpoint's `create_job` raises and the request fails with a 500. Before, the same setting made the worker read a queue the endpoint never wrote to, which failed silently. The docs line says so.
- **The TTS worker's filter:** `await sdk.run(handler=_handle_tts_job, job_type=TTS_JOB_TYPE)`. `WorkerSDK.run` already takes `job_type` (`app/core/Jobs/worker_sdk.py`).
- **The audio pipeline worker and the GPU worker are not changed.**
- **Test first:** the RED run below happens before the source change.
- **Docs:** `Docs/Architecture/audio.md`, item 8 of the TTS flow. It has no `Docs/Published` mirror. Run `bash Helper_Scripts/refresh_docs_published.sh` anyway and check that it changes nothing.
- **Backlog:**
  - Use `tools/backlog-py` only (`PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task ...`), with explicit ids, `--append-notes`, and `git add -A backlog/tasks`.
  - TASK-13848 is filed with an explicit id. Open PRs use ids up to 13820, and 13840–13847 are on `dev`; backlog-py refuses an id that exists.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never `git stash`.
- **Out of scope:**
  - **The audio pipeline's transcription hand-off is broken; filed here as TASK-13848, fixed next by the owner's choice.**
    - `audio_convert` queues the next stage on `transcribe`, which `audio` does not allow. On `dev`, the probe below shows `audio_convert` completing and then `Audio worker loop error: Queue 'transcribe' not allowed for domain 'audio'`, with no `audio_transcribe` job created.
    - Even with the queue allowed, only the separate GPU worker container reads it.
    - Registering `transcribe` is deliberately not done here: on its own it would turn a loud error into jobs nobody picks up.
  - **TTS jobs already queued on `default` at deploy are not picked up** by the TTS worker, which now reads `tts`.
    - It needs a hand-run TTS worker with jobs in flight, and with the audio worker on those jobs were already being quarantined.
    - Left queued, they count toward a user's active-job cap when fair share is on.
    - The PR gives operators one statement: `UPDATE jobs SET queue='tts' WHERE domain='audio' AND job_type='tts_longform' AND status='queued' AND queue='default';`
  - **No deployment runs the TTS worker, but the WebUI offers long-form TTS.**
    - `apps/packages/ui/src/services/tldw/domains/models-audio.ts` and `TldwApiClient.ts` call `POST /api/v1/audio/speech/jobs`.
    - There is no lifespan hook, sidecar entry, Dockerfile or service unit, so those jobs sit `queued` unless an operator runs the module.
    - Filed here as TASK-13849. The docs line now names the command and says nothing starts it.

## Review Focus

1. **`TTS_JOBS_QUEUE=default` or `transcribe`**, for example an old setting matching the endpoint's old hard-coded queue. It is refused: the TTS worker fails at startup and the endpoint with a 500, rather than reviving the collision or stranding jobs. Test: `test_tts_jobs_queue_refuses_the_audio_pipeline_queues`.
2. **A custom unregistered `TTS_JOBS_QUEUE`.** The endpoint fails loudly with a 500 instead of stranding jobs; documented. No test: the Jobs allowlist is already tested in `tests/Jobs`.
3. **The full speech-job round trip still works**, from the endpoint through a worker acquisition on the `tts` queue. This pins the default, the allowlist registration and the hand-off in one assertion. Test: `test_speech_jobs_flow_links_artifact_and_history`, which now acquires on `queue="tts"` instead of `default`.
4. **A TTS worker sharing a queue with an audio pipeline job** (a custom `TTS_JOBS_QUEUE` registered for both) takes only the TTS job. A `tts_longform` job on the same queue is the positive control: it is taken, which proves the worker polled the queue. That matters because `WorkerSDK.run` treats an acquisition error as "no job". Test: `test_tts_worker_takes_only_tts_jobs_from_its_queue`.
5. **The TTS worker passes the filter to the SDK.** Test: `test_tts_worker_returns_cleanly_when_sdk_stops_before_stop_event`. Its fake SDK now asserts `job_type == TTS_JOB_TYPE`; its `run` had no `job_type` parameter, so the old fake fails with a `TypeError`. Status, progress, artifacts and the admin listings read the `audio` domain by id or status, never by queue (`app/api/v1/endpoints/audio/audio_jobs.py`), so they are unaffected.

---

### Task 1: TTS jobs get their own queue, and the TTS worker takes only TTS jobs

**Files:**
- Modify: `tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py`: `import asyncio`; the flow test's acquire; one new test after it.
- Modify: `tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py`: two new tests before `test_standalone_worker_waits_for_override_bootstrap`.
- Modify: `tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py`: the fake SDK in `test_tts_worker_returns_cleanly_when_sdk_stops_before_stop_event`.
- Modify: `tldw_Server_API/app/core/TTS/utils.py`: add `tts_jobs_queue()`.
- Modify: `tldw_Server_API/app/core/TTS/tts_jobs_worker.py`: the module docstring, the utils import, `run_tts_jobs_worker`.
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py`: the utils import and `create_speech_job`'s `create_job`.
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`: `DOMAIN_ALLOWED_QUEUES`.
- Modify: `Docs/Architecture/audio.md`.
- Modify: `backlog/tasks/task-13641*.md`; create `task-13848*` and `task-13849*` (backlog-py only).
- Add: this plan, `Docs/superpowers/plans/2026-10-17-tts-audio-job-isolation.md`.

**Interfaces:**
- Produces: `tldw_Server_API.app.core.TTS.utils.tts_jobs_queue() -> str`, which raises `ValueError` for `default` and `transcribe`.
- Changes:
  - `POST /api/v1/audio/speech/jobs` enqueues on `tts_jobs_queue()` instead of `default`.
  - `run_tts_jobs_worker` reads `tts_jobs_queue()`, whose default moves from `default` to `tts`, and acquires only `tts_longform`.

Set once for every step: `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` and `export TLDW_TEST_NO_DOCKER=1`.

- [ ] **Step 1: Branch**

```bash
git fetch -q origin dev
git switch -c fix/tts-audio-job-isolation origin/dev
```

Copy this plan to `Docs/superpowers/plans/2026-10-17-tts-audio-job-isolation.md`.

- [ ] **Step 2: Write the failing tests**

Apply exactly this diff with `git apply`:

```diff
diff --git a/tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py b/tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py
index a355b276ad..d09a34dfe3 100644
--- a/tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py
+++ b/tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py
@@ -2230,8 +2230,9 @@ async def test_tts_worker_returns_cleanly_when_sdk_stops_before_stop_event(
         def __init__(self, _manager, _config):
             return None
 
-        async def run(self, *, handler):
+        async def run(self, *, handler, job_type=None):
             assert handler is tts_jobs_worker._handle_tts_job
+            assert job_type == tts_jobs_worker.TTS_JOB_TYPE
 
         def stop(self):
             return None
diff --git a/tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py b/tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py
index 286c754f82..7af92fec22 100644
--- a/tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py
+++ b/tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py
@@ -5,6 +5,7 @@ Tests the full request/response cycle with real components,
 no mocking except for external API calls.
 """
 
+import asyncio
 import contextlib
 import json
 import base64
@@ -1248,7 +1249,7 @@ class TestTTSJobsHistoryIntegration:
             worker_id = "tts-history-integration-worker"
             job = jm.acquire_next_job(
                 domain="audio",
-                queue="default",
+                queue="tts",
                 job_type="tts_longform",
                 lease_seconds=60,
                 worker_id=worker_id,
@@ -1301,6 +1302,52 @@ class TestTTSJobsHistoryIntegration:
             with contextlib.suppress(Exception):
                 audio_jobs._job_manager_cache.clear()
 
+    async def test_audio_jobs_worker_leaves_a_submitted_speech_job_alone(self, test_client, auth_headers, monkeypatch, tmp_path):
+        """The audio pipeline worker must not take a long-form TTS job (TASK-13641)."""
+        import tldw_Server_API.app.services.audio_jobs_worker as audio_worker
+
+        monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
+        monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")
+
+        async def _unlimited(user_id):
+            return {"concurrent_jobs": None}
+
+        async def _yes(user_id):
+            return True, ""
+
+        async def _noop(user_id):
+            return None
+
+        monkeypatch.setattr(audio_worker, "get_limits_for_user", _unlimited)
+        monkeypatch.setattr(audio_worker, "can_start_job", _yes)
+        monkeypatch.setattr(audio_worker, "increment_jobs_started", _noop)
+        monkeypatch.setattr(audio_worker, "finish_job", _noop)
+        with contextlib.suppress(Exception):
+            audio_jobs._job_manager_cache.clear()
+
+        try:
+            submit_resp = test_client.post(
+                "/api/v1/audio/speech/jobs",
+                json={"input": "Long-form speech", "voice": "alloy", "model": "tts-1", "response_format": "mp3"},
+                headers=auth_headers,
+            )
+            assert submit_resp.status_code == status.HTTP_200_OK
+            job_id = int(submit_resp.json()["job_id"])
+
+            stop = asyncio.Event()
+            task = asyncio.create_task(audio_worker.run_audio_jobs_worker(stop))
+            await asyncio.sleep(0.5)
+            stop.set()
+            await asyncio.wait_for(task, timeout=2.0)
+
+            job = audio_jobs.get_job_manager().get_job(job_id) or {}
+            assert job.get("status") == "queued"
+            assert int(job.get("retry_count") or 0) == 0
+            assert job.get("error_message") in (None, "")
+        finally:
+            with contextlib.suppress(Exception):
+                audio_jobs._job_manager_cache.clear()
+
 # ========================================================================
 # Provider Management Endpoint Tests
 # ========================================================================
diff --git a/tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py b/tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py
index 3f5327c6db..8d9463db6e 100644
--- a/tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py
+++ b/tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py
@@ -253,6 +253,49 @@ async def test_tts_jobs_worker_output_failure_after_synthesis_is_terminal(
     assert sentinel not in repr(exc_info.value)
 
 
+@pytest.mark.unit
+async def test_tts_worker_takes_only_tts_jobs_from_its_queue(monkeypatch, tmp_path):
+    """On a queue that also holds an audio pipeline job, the TTS worker takes only the TTS job (TASK-13641)."""
+    from tldw_Server_API.app.core.Jobs.manager import JobManager
+
+    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_AUDIO", "tts-shared")
+    monkeypatch.setenv("TTS_JOBS_QUEUE", "tts-shared")
+    jm = JobManager(tmp_path / "tts-shared-queue.db")
+    monkeypatch.setattr(tts_jobs_worker, "_jobs_manager", lambda: jm)
+    audio_row = jm.create_job(
+        domain="audio",
+        queue="tts-shared",
+        job_type="audio_convert",
+        payload={"local_path": "/tmp/audio.wav"},
+        owner_user_id="1",
+    )
+    # Positive control: proves the worker polled this queue (it fails: no speech_request).
+    tts_row = jm.create_job(
+        domain="audio", queue="tts-shared", job_type="tts_longform", payload={}, owner_user_id="1"
+    )
+
+    stop = asyncio.Event()
+    task = asyncio.create_task(tts_jobs_worker.run_tts_jobs_worker(stop))
+    await asyncio.sleep(0.5)
+    stop.set()
+    await asyncio.wait_for(task, timeout=5.0)
+
+    audio_job = jm.get_job(int(audio_row["id"])) or {}
+    assert audio_job.get("status") == "queued"
+    assert int(audio_job.get("retry_count") or 0) == 0
+    assert (jm.get_job(int(tts_row["id"])) or {}).get("status") != "queued"
+
+
+@pytest.mark.unit
+@pytest.mark.parametrize("queue", ["default", "transcribe"])
+def test_tts_jobs_queue_refuses_the_audio_pipeline_queues(monkeypatch, queue):
+    from tldw_Server_API.app.core.TTS import utils as tts_utils
+
+    monkeypatch.setenv("TTS_JOBS_QUEUE", queue)
+    with pytest.raises(ValueError, match="TTS_JOBS_QUEUE"):
+        tts_utils.tts_jobs_queue()
+
+
 @pytest.mark.unit
 async def test_standalone_worker_waits_for_override_bootstrap(monkeypatch):
     """Standalone dispatch cannot start before the first policy load succeeds."""
```

- [ ] **Step 3: Run them to see them fail**

```bash
$PY -m pytest -p no:cacheprovider -p no:randomly -q --tb=line \
  "tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py::TestTTSJobsHistoryIntegration" \
  "tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py::test_tts_worker_takes_only_tts_jobs_from_its_queue" \
  "tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py::test_tts_jobs_queue_refuses_the_audio_pipeline_queues" \
  "tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py::test_tts_worker_returns_cleanly_when_sdk_stops_before_stop_event"
```

Expected: `6 failed`. Measured on `origin/dev` `f456e5c9bc`:
- `test_speech_jobs_flow_links_artifact_and_history`: `assert None is not None`. The endpoint put the job on `default`, so nothing is acquired from `tts`.
- `test_audio_jobs_worker_leaves_a_submitted_speech_job_alone`: `AssertionError: assert 'quarantined' == 'queued'`. The audio worker took the speech job, failed it as an unknown job type, and it was quarantined.
- `test_tts_worker_takes_only_tts_jobs_from_its_queue`: `AssertionError: assert 'failed' == 'queued'`. The TTS worker took the `audio_convert` job and failed it.
- `test_tts_jobs_queue_refuses_the_audio_pipeline_queues[default]` and `[transcribe]`: `AttributeError: module 'tldw_Server_API.app.core.TTS.utils' has no attribute 'tts_jobs_queue'`.
- `test_tts_worker_returns_cleanly_when_sdk_stops_before_stop_event`: `AssertionError: assert None == 'tts_longform'`. No job-type filter is passed.

- [ ] **Step 4: Apply the source change**

Apply exactly this diff:

```diff
diff --git a/tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py b/tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py
index 9feab126ee..f5ac512df2 100644
--- a/tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py
+++ b/tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py
@@ -70,6 +70,7 @@ from tldw_Server_API.app.core.TTS.utils import (
     contains_tts_credential_fields,
     parse_bool,
     tts_history_text_length,
+    tts_jobs_queue,
 )
 from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat
 
@@ -675,7 +676,7 @@ async def create_speech_job(
 
     job = jm.create_job(
         domain="audio",
-        queue="default",
+        queue=tts_jobs_queue(),
         job_type="tts_longform",
         payload=payload,
         owner_user_id=str(current_user.id),
diff --git a/tldw_Server_API/app/core/Jobs/manager.py b/tldw_Server_API/app/core/Jobs/manager.py
index 7bb58ec913..5842e855a3 100644
--- a/tldw_Server_API/app/core/Jobs/manager.py
+++ b/tldw_Server_API/app/core/Jobs/manager.py
@@ -683,6 +683,8 @@ class JobManager:
         "writing": ("writing-review", "writing-ai"),
         "scheduled_tasks": ("scheduled-tasks",),
         "admin_webhooks": ("delivery",),
+        # Long-form TTS jobs, kept off "default", which the audio pipeline worker reads unfiltered.
+        "audio": ("tts",),
     }
 
     # --- Shutdown/acquisition gate (process-wide) ---
diff --git a/tldw_Server_API/app/core/TTS/tts_jobs_worker.py b/tldw_Server_API/app/core/TTS/tts_jobs_worker.py
index 07e9c5103b..ab3ec9ffc1 100644
--- a/tldw_Server_API/app/core/TTS/tts_jobs_worker.py
+++ b/tldw_Server_API/app/core/TTS/tts_jobs_worker.py
@@ -2,8 +2,8 @@
 TTS Jobs worker for long-form speech generation.
 
 - domain = "audio"
-- queue = os.getenv("TTS_JOBS_QUEUE", "default")
-- job_type = "tts_longform"
+- queue = utils.tts_jobs_queue(): TTS_JOBS_QUEUE, default "tts"
+- job_type = "tts_longform", and only that type is acquired
 """
 
 from __future__ import annotations
@@ -67,6 +67,7 @@ from tldw_Server_API.app.core.TTS.utils import (
     contains_tts_credential_fields,
     parse_bool,
     tts_history_text_length,
+    tts_jobs_queue,
 )
 
 TTS_DOMAIN = "audio"
@@ -935,7 +936,7 @@ async def _handle_tts_job(job: dict[str, Any]) -> dict[str, Any]:
 
 async def run_tts_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
     worker_id = (os.getenv("TTS_JOBS_WORKER_ID") or f"tts-jobs-{os.getpid()}").strip()
-    queue = (os.getenv("TTS_JOBS_QUEUE") or "default").strip() or "default"
+    queue = tts_jobs_queue()
     lease_seconds = _coerce_int(os.getenv("TTS_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"), 60)
     renew_jitter = _coerce_int(
         os.getenv("TTS_JOBS_RENEW_JITTER_SECONDS")
@@ -967,7 +968,8 @@ async def run_tts_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
 
     logger.info("TTS Jobs worker starting (queue={}, worker_id={})", queue, worker_id)
     try:
-        await sdk.run(handler=_handle_tts_job)
+        # The audio domain is shared with the audio pipeline: take only TTS jobs, whatever the queue.
+        await sdk.run(handler=_handle_tts_job, job_type=TTS_JOB_TYPE)
     finally:
         if _stop_watcher_task is not None and not _stop_watcher_task.done():
             _stop_watcher_task.cancel()
diff --git a/tldw_Server_API/app/core/TTS/utils.py b/tldw_Server_API/app/core/TTS/utils.py
index 491239f122..a05d5a51f6 100644
--- a/tldw_Server_API/app/core/TTS/utils.py
+++ b/tldw_Server_API/app/core/TTS/utils.py
@@ -61,6 +61,21 @@ def contains_tts_credential_fields(value: Any) -> bool:
     return False
 
 
+def tts_jobs_queue() -> str:
+    """The Jobs queue for long-form TTS jobs, shared by the enqueueing endpoint and the TTS worker.
+
+    The audio pipeline workers read the audio domain's ``default`` and ``transcribe`` queues
+    without a job-type filter, so a TTS job there would be taken and failed as an unknown job
+    type. Those two names are refused rather than silently reviving that collision.
+    """
+    queue = (os.getenv("TTS_JOBS_QUEUE") or "").strip() or "tts"
+    if queue in {"default", "transcribe"}:
+        raise ValueError(
+            f"TTS_JOBS_QUEUE must not be {queue!r}: the audio pipeline workers read it without a job-type filter"
+        )
+    return queue
+
+
 def parse_bool(value: Any, default: bool | None = False) -> bool:
     """Parse a value into a boolean in a tolerant, explicit way.
 
```

- [ ] **Step 5: Run the tests to see them pass**

Run the Step 3 command. Expected: `6 passed`.

- [ ] **Step 6: Document it**

Apply this diff, then run `bash Helper_Scripts/refresh_docs_published.sh` and `git status --short Docs`:

```diff
diff --git a/Docs/Architecture/audio.md b/Docs/Architecture/audio.md
index a9157a4385..41e7b3abda 100644
--- a/Docs/Architecture/audio.md
+++ b/Docs/Architecture/audio.md
@@ -27,7 +27,7 @@ All paths under `tldw_Server_API/` unless noted.
 5. All provider egress goes through the shared HTTP client's egress validation (SSRF allow/deny lists).
 6. Errors map through the TTS exception taxonomy (auth, rate-limit, quota, model-not-found/load, …) to HTTP.
 7. Response: audio bytes with optional sample-rate and alignment headers; metrics record totals, durations, TTFB.
-8. Long-form: `POST /speech/jobs` enqueues an audio-domain job forcing non-streaming; artifacts are listed per job.
+8. Long-form: `POST /speech/jobs` enqueues an audio-domain `tts_longform` job forcing non-streaming; artifacts are listed per job. The job goes on the `tts` queue. `TTS_JOBS_QUEUE` overrides it: set the same value for the API server and the TTS worker, and register a custom name in `JOBS_ALLOWED_QUEUES_AUDIO`. `default` and `transcribe` are refused, because the audio pipeline workers read them without a job-type filter. The TTS jobs worker (`python -m tldw_Server_API.app.core.TTS.tts_jobs_worker`) reads that queue and takes only `tts_longform` jobs; nothing starts it automatically.
 
 ## STT (condensed)
 
```

Expected: only `Docs/Architecture/audio.md` is modified under `Docs`. The page has no published mirror.

- [ ] **Step 7: Run every test file that touches the changed code, plus the AudioJobs, Jobs and Docs tests**

```bash
mkdir -p .superpowers/sdd
git grep -l -E "tts_jobs_worker|audio_tts|TTS\.utils|core\.TTS import [^\n]*utils|tts_longform|TTS_JOBS_QUEUE|_get_allowed_queues|DOMAIN_ALLOWED_QUEUES|audio_jobs_worker" -- 'tldw_Server_API/tests/*.py' > .superpowers/sdd/t13641_files.txt
wc -l < .superpowers/sdd/t13641_files.txt
RUN_JOBS=1 $PY -m pytest -p no:cacheprovider -q $(cat .superpowers/sdd/t13641_files.txt) tldw_Server_API/tests/AudioJobs tldw_Server_API/tests/Jobs tldw_Server_API/tests/Docs
```

Run this without `-n`. With `-n 4`, xdist stops at collection ("Different tests were collected between gw1 and gw0"), so no tests run. It takes about 30 minutes. `.superpowers/` is git-ignored.

Expected: the `git grep` lists 36 files. The run gives `1 failed, 3365 passed, 6 skipped` (measured on the prototype). The failure also fails on unchanged `origin/dev`:
- `tests/Jobs/test_retry_admission_fixture_registration.py::test_bridge_resolution_preserves_jobs_loop_and_isolation`: `ValueError: Pytest terminal summary report not found`.

If any other failure appears, run it on unchanged `origin/dev` before going on.

- [ ] **Step 8: Lint and security**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/app/core/TTS/utils.py tldw_Server_API/app/core/TTS/tts_jobs_worker.py tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py
uvx bandit -q -ll tldw_Server_API/app/core/TTS/utils.py tldw_Server_API/app/core/TTS/tts_jobs_worker.py tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py tldw_Server_API/app/core/Jobs/manager.py
```

Expected: ruff reports the 6 findings that `origin/dev` already has in two test files, at the line numbers below (each shifted by this diff's added lines):
- `test_tts_endpoints.py`:
  - `:8:1 I001`, `:12:8 F401` (`tempfile`), `:378:31 UP032`, `:926:9 I001` and `:1733:9 I001`;
  - on `dev` they are at `:8`, `:11`, `:377`, `:925` and `:1686`.
- `test_tts_jobs_worker.py`: `:1:1 I001`.

The source files are clean. Bandit reports one finding, B608 at `manager.py:9784`, which is `:9782` on `origin/dev`. It is a pre-existing query build that this diff shifts by two lines.

- [ ] **Step 9: Commit the fix**

```bash
git add tldw_Server_API/app/core/TTS/utils.py tldw_Server_API/app/core/TTS/tts_jobs_worker.py tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py Docs/Architecture/audio.md
git commit -m "fix(tts): long-form TTS jobs get their own queue, and the TTS worker takes only TTS jobs (TASK-13641)

POST /api/v1/audio/speech/jobs put tts_longform jobs on the audio domain's
default queue, which the in-process audio pipeline worker reads without a
job-type filter: with AUDIO_JOBS_WORKER_ENABLED it took each speech job and
failed it as an unknown job type until it was quarantined. The standalone
TTS worker read the same queue unfiltered and failed audio_* stage jobs.

Long-form TTS jobs now go on the tts queue (TTS_JOBS_QUEUE, registered for
the audio domain), chosen in one place for the endpoint and the worker, and
the TTS worker acquires only tts_longform jobs. TTS_JOBS_QUEUE=default or
transcribe is refused, so an old setting fails loudly instead of reviving
the collision or leaving the worker on a queue the endpoint no longer uses.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 10: File the two follow-ups, close the task, commit the plan**

The notes below quote the measured results. If Steps 3, 5, 7 or 8 printed different lines, put the actual ones in the notes.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Audio jobs pipeline stops after conversion: the transcribe stage's queue is not allowed" --id TASK-13848 --labels audio,jobs,bug --description "On dev, every POST /api/v1/audio/jobs/submit job stops after audio_convert. The in-process audio worker (app/services/audio_jobs_worker.py) queues the next stage, audio_transcribe, on the queue 'transcribe', which the Jobs audio domain does not allow (JobManager.DOMAIN_ALLOWED_QUEUES has no audio entry for it; standard queues are default, high, low). create_job raises 'Queue transcribe not allowed for domain audio', the worker logs 'Audio worker loop error' and nothing further runs, while the audio_convert job reads completed. Probe (stubbed convert_to_wav, worker run for 2 s, no JOBS_ALLOWED_QUEUES_AUDIO): jobs = audio_convert default completed; error as above. Registering 'transcribe' alone is not enough: only the separate GPU worker container (app/services/audio_transcribe_gpu_worker.py, Dockerfiles/Dockerfile.audio_gpu_worker) reads that queue, and the in-process worker, which has an audio_transcribe handler, reads only default. The existing tests miss it because they create audio_transcribe jobs on default by hand, and the opt-in pipeline smoke test asserts nothing. Found while fixing TASK-13641." --ac "A job submitted to /api/v1/audio/jobs/submit is transcribed by the in-process audio worker when no GPU worker is configured, with a test that runs the real hand-off from audio_convert" --ac "A deployment that runs the GPU transcription worker can still route audio_transcribe to it, and the queue it uses is allowed for the audio domain" --ac "A failure to queue the next stage fails the job visibly instead of leaving the previous stage completed and the job stranded"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Long-form TTS jobs have no worker in any deployment, but the WebUI offers them" --id TASK-13849 --labels tts,jobs,bug --description "POST /api/v1/audio/speech/jobs queues tts_longform jobs (on the tts queue since TASK-13641), and the WebUI calls it (apps/packages/ui/src/services/tldw/domains/models-audio.ts and TldwApiClient.ts). But nothing starts the TTS jobs worker: there is no lifespan hook in app/services/startup_content_jobs_pollers.py, no sidecar entry, no Dockerfile and no service unit. It runs only when an operator starts python -m tldw_Server_API.app.core.TTS.tts_jobs_worker, so on a default deployment a long-form speech job stays queued with nothing to say why. Found in the plan review for TASK-13641." --ac "A default deployment either runs the TTS jobs worker (for example a lifespan worker behind a flag, like the other Jobs workers) or tells the caller that long-form TTS needs it, instead of leaving jobs queued silently" --ac "The deployment docs say how to run the TTS jobs worker"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13641 --append-notes "Verified on dev f456e5c9bc: the collision is real and does not need the TTS worker. POST /api/v1/audio/speech/jobs enqueued tts_longform on the audio domain's default queue, which the in-process audio pipeline worker (AUDIO_JOBS_WORKER_ENABLED) reads unfiltered; it took a submitted speech job within half a second and failed it as an unknown job type until it was quarantined. The standalone TTS worker read the same queue unfiltered and failed an audio_convert job as unsupported. Nothing in the repository starts the TTS worker; it runs only as python -m tldw_Server_API.app.core.TTS.tts_jobs_worker. Fixed: core/TTS/utils.tts_jobs_queue() (TTS_JOBS_QUEUE, default tts) is the queue for long-form TTS jobs, used by the endpoint and the worker; tts is registered for the audio domain in JobManager.DOMAIN_ALLOWED_QUEUES; the TTS worker passes job_type=tts_longform to WorkerSDK.run; tts_jobs_queue() refuses default and transcribe (the queues the audio pipeline workers read unfiltered), so an old TTS_JOBS_QUEUE=default fails at worker startup and at the endpoint instead of reviving the collision or stranding jobs. A custom TTS_JOBS_QUEUE must be set the same on the API server and the worker and registered in JOBS_ALLOWED_QUEUES_AUDIO, otherwise the endpoint fails with a 500 (it used to strand jobs silently). TTS jobs already queued on default at deploy are not picked up by the TTS worker; move them with UPDATE jobs SET queue='tts' WHERE domain='audio' AND job_type='tts_longform' AND status='queued' AND queue='default'. Tests: RED 6 failed on dev (the endpoint queued on default; the audio worker quarantined the speech job; the TTS worker failed the audio_convert job; no tts_jobs_queue; no job-type filter), GREEN 6 passed; every test file touching the changed code plus tests/AudioJobs, tests/Jobs and tests/Docs 1 failed, 3365 passed, 6 skipped, the failure (test_retry_admission_fixture_registration's terminal-summary check) also failing on unchanged dev. Ruff: only the six findings dev already has in two test files. Bandit -ll: only the pre-existing B608 in Jobs/manager.py. Docs/Architecture/audio.md documents the queue. Found and filed: TASK-13848 (the audio pipeline stops after conversion; the transcribe queue is not allowed and only the GPU worker reads it) and TASK-13849 (nothing starts the TTS jobs worker, but the WebUI offers long-form TTS)." --check-ac 1 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "Long-form TTS jobs go on their own tts queue (TTS_JOBS_QUEUE, registered for the audio domain), chosen in one place for the endpoint and the TTS worker, so the audio pipeline worker no longer takes and quarantines them; the TTS worker acquires only tts_longform jobs, so it no longer fails audio pipeline jobs." --status Done
git add -A backlog/tasks
git add Docs/superpowers/plans/2026-10-17-tts-audio-job-isolation.md
git commit -m "docs(tts): plan for TTS and audio job isolation; close TASK-13641, file TASK-13848 and TASK-13849

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: `git show --stat HEAD` lists the three task files and the plan only.
