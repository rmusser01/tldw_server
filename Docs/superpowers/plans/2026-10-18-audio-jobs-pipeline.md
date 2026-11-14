# Audio Jobs Pipeline Hand-Off Implementation Plan (TASK-13850, TASK-13851, TASK-13852, TASK-13853)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Independent review and dry run (2026-10-18): 1 Blocker, 3 Important, 15 Minor; all folded in.
- Each change is marked "(Review B1)" to "(Review M15)". Full report: `scratchpad/review_plan_audio_pipeline.md`.
- The revision was re-run in two scratch worktrees on `origin/dev` `32eb11ae03`, with the reviewer's analysis-plan stand-in applied first. Every RED, GREEN and mutation result below that names a finding was observed there.

**Goal:** An audio pipeline runs each stage once, even when the stage outlasts its lease. It carries a multi-hour transcript through every stage. A refused hand-off fails at once instead of transcribing again. A cancelled pipeline stops. The worker's own hand-offs are never refused by the user's submit quotas.

**Architecture:**
- **Lease (TASK-13850).** Both workers start a small renewal loop, `start_lease_renewal` in `audio_jobs_worker.py`, for the acquired job. They cancel it in the loop's `finally`, so it stops when the stage completes or fails. (Review I3, M9)
  - Every half lease it calls `jm.renew_job_lease` with the worker's `worker_id` and `lease_id`.
  - It stops when a renewal is refused (the lease was lost, or the job was cancelled).
  - It logs a backend error and tries again at the next renewal, as `core_jobs_worker.py:128-148` does. WorkerSDK's private `_auto_renew` stops at the first error, so one locked-database error would let a multi-hour stage's lease expire.
- **Quotas (TASK-13853).** `JobManager.create_job`/`admit_job` gain `quota_exempt: bool = False`, the twin of PR #3260's `fair_share_exempt`. It skips the two submit quotas. Both workers' hand-offs pass it.
- **One hand-off (TASK-13851, TASK-13852).** A shared `hand_off()` in `audio_jobs_worker.py` replaces the in-process block and the GPU worker's older complete-then-create code. It does five things:
  - It classifies `create_job` errors. A `ValueError` or `TypeError` (an unserializable payload) is a refusal that repeats, so it fails the stage now. Any other error is retried in place, twice, before the stage is retried as a whole. (Review M5)
  - It completes the stage with the `result` its caller passes: the analysis plan's `{"analysis_error": ...}` for an analyze stage whose analysis failed for good. (Review B1)
  - It cancels the stage it just queued when the current stage turns out to have been cancelled.
  - It copies the pipeline's `batch_group`. The new `DELETE /api/v1/audio/jobs/{job_id}` uses it to cancel the whole pipeline. The route handles only the six pipeline job types, and it repeats list-then-cancel so that a stage queued while it cancels is caught. (Review I1, I2)
  - It carries the transcript once, and a stored result over the cap fails `audio_store` instead of leaving it processing.

**Tech Stack:** Python 3.12, asyncio, the Jobs `JobManager` (SQLite and Postgres), FastAPI, pytest. (WorkerSDK is no longer used; Review M9.)

**Spec:** the four backlog tasks on `origin/dev`:
- `backlog/tasks/task-13850 - Audio-jobs-renew-the-job-lease-while-a-long-stage-runs.md`
- `backlog/tasks/task-13851 - Audio-jobs-long-recordings-exceed-the-job-payload-cap-at-the-hand-off-and-a-refused-hand-off-re-runs-the-stage.md`
- `backlog/tasks/task-13852 - Audio-jobs-cancelling-a-running-stage-does-not-stop-the-pipeline.md`
- `backlog/tasks/task-13853 - Audio-jobs-stage-hand-offs-count-against-the-per-minute-and-queued-job-submit-quotas.md`

## What was measured (on `origin/dev` `e1f7770953`)

Probes ran in a scratch worktree with the real worker loop, a fresh SQLite Jobs DB and stubbed convert/STT, as `test_audio_worker_smoke.py`'s `pipeline_worker` fixture does.

- **TASK-13850.**
  - The in-process worker acquires with `JOBS_LEASE_SECONDS` (default 120, `audio_jobs_worker.py:127`), and the GPU worker with 180 (`audio_transcribe_gpu_worker.py:87`). Neither renews.
  - The probe ran two worker loops with `JOBS_LEASE_SECONDS=2` and an STT that takes 3 s. STT ran twice, and the transcribe job completed with `retry_count` 1. Recovery can repeat until the stage completes, so another run can see STT three times; the review's dry run did. (Review M2)
  - The retry is consumed by `acquire_next_job`, which runs `_recover_expired_processing_jobs` on every call (`manager.py:6379`) and applies the job's `expired_lease_policy`, default `consume_retry`. This confirms the task's implementation note.
- **TASK-13851.**
  - Payload size per minute of speech, using the test transcript in Task 5 (12 segments a minute, about 70 characters each, faster-whisper's segment shape). The table gives the bytes per minute and the length that fits under the default 1 MiB cap:

    | Payload | Without chunking | With chunking |
    | --- | --- | --- |
    | Today | 5,903 B/min, about 2.9 h | 7,270 B/min, about 2.4 h |
    | `text` + `segments` once, `normalized_stt` without them (Decision 1, recommended) | 2,952 B/min, about 5.9 h | 4,320 B/min, about 4.0 h |
    | `segments` only (Decision 1, option b) | 2,094 B/min, about 8.3 h | 3,462 B/min, about 5.0 h |

  - For the 3-hour test transcript, the transcribe→chunk hand-off is refused with `Payload too large: 1062513 bytes > limit 1048576`. STT ran twice and the job ended `quarantined` with `retry_count` 2.
  - A refused job type (`JOBS_ALLOWED_JOB_TYPES_AUDIO=audio_convert,audio_transcribe`): STT ran twice, then the job was quarantined.
  - One transient `create_job` error: STT ran twice, and the stage completed with `retry_count` 1.
  - An `audio_store` job whose result is 13 bytes over the cap stays `processing` with a lease. `complete_job` raises `ValueError` (`manager.py:7359`), and the loop's catch-all only logs it (`audio_jobs_worker.py:461`).
- **TASK-13852.**
  - The probe cancelled the transcribe job from inside STT. The transcribe job ended `cancelled`, but `audio_store` ended `completed`.
  - `audio_jobs.py` has no cancel route. The only Jobs cancel route is the admin `POST /api/v1/jobs/batch/cancel` (`jobs_admin.py:1450`, needs `X-Confirm`).
- **TASK-13853.**
  - With `JOBS_QUOTA_SUBMITS_PER_MIN_AUDIO=1`, the convert job is quarantined with `retry_count` 2 and the error `could not queue audio_transcribe: Quota exceeded: submits per minute`.
  - `JOBS_QUOTA_MAX_QUEUED_AUDIO=1` alone never stops a lone pipeline, because it has at most one queued row at a time. It does refuse the hand-off when the same user has another job queued (Task 2's test).
  - The per-minute window counts every row the owner created in the domain in the last 60 s (`count_recent_job_admissions`, `DB_Management/jobs_failed_requeue.py:197`). So stage rows also count toward the user's next submission (Decision 3).
- **GPU worker.**
  - It completes the stage before it queues the next one (`audio_transcribe_gpu_worker.py:122-132`), with no idempotency key and no exemptions.
  - A refused hand-off leaves the stage `completed` with nothing after it (the RED result of Task 4's GPU test).
  - Its `_handle_gpu_audio_transcribe_stage` (lines 22-70) is a copy of the in-process `_handle_audio_transcribe_stage`.
- **CI side effects.**
  - `Jobs/manager.py` is in `Helper_Scripts/ci/scope_predicate_baseline.txt` (lines 5522-6343). PR 1 moves those lines by 5, so the baseline is regenerated.
  - PR 2's new route changes `apps/packages/api-types/openapi.fingerprint.json` only (`schema_count` +1; `paths.d.ts` is unchanged, because the path already exists).
  - `refresh_docs_published.sh` on unchanged dev changes nothing.
- **Prototype.** Both PRs were prototyped in the scratch worktree; every RED and GREEN result below was observed there. The counts of the broad runs depend on how many tests the analysis PR adds, so the close-out tasks say "all pass; record the counts" and give each command. The counts below are the revision's dry run, with the reviewer's analysis stand-in. (Review M10)
  - Task 3's sweep: 450 passed, 70 skipped, in 3.5 minutes.
  - Task 3's `RUN_JOBS=1` run: 272 passed, with a local Postgres reachable. It took **17 minutes** in the review's run and 3 minutes in the revision's, on the same machine; allow for the longer time.
  - Task 9's sweep: 462 passed, 70 skipped. Its `tests/Docs` and API-contract run: 323 passed. The five required ratchet tests: 53 passed on each PR. (Review M1)
  - Bandit `-ll` reports no issues on the two workers and `audio_jobs.py`. On `manager.py` it reports one pre-existing Medium B608, at line 9790 on dev (9800 after the analysis PR and PR 1); nothing new. (Review M3)
  - `ruff` reports nothing new; its two findings in `audio_jobs.py` (I001, F401 `contextlib`, at line 25 on dev and 26 after PR 2's `import uuid`) already exist on dev.

## PRs

Two PRs, in this order, both after the analysis-stage PR (TASK-13845/13846/13847, `plan_audio_analysis.md`) has merged:

| PR | Tasks | Branch | Why separate |
| --- | --- | --- | --- |
| **PR 1** | TASK-13850 (lease), TASK-13853 (quotas) | `fix/audio-jobs-lease-and-quotas` | No owner decision blocks it. Each change is a few lines and stands alone. TASK-13853 must land before PR 2: PR 2 treats every `ValueError` (and `TypeError`, Review M5) from `create_job` as permanent, and a quota refusal is a `ValueError`. |
| **PR 2** | TASK-13851 (payload, refusals, store), TASK-13852 (cancel) | `fix/audio-jobs-hand-off` | Both rewrite the same hand-off into one shared function for both workers. They wait on Decisions 1 and 2. |

## Global Constraints

- Branch each PR from `origin/dev`, PR 2 after PR 1 has merged. Both target `dev`.
- **Builds on `plan_audio_analysis.md`, which lands first.** This plan uses its interfaces and does not redo them:
  - `stage_after_transcription(payload) -> str`, in `audio_jobs_worker.py`. The GPU worker imports it from there.
  - `_StageFailure(message, *, failure_code, retryable, backoff_seconds=1, quarantine=True)`.
  - `JobManager.fail_job(..., quarantine: bool = True)`, with the worker's final `fail_job` passing `getattr(failure, "backoff_seconds", 1)` and `getattr(failure, "quarantine", True)`.
  - Storing the transcript with `analysis_error` when analysis fails for good.
  - Its Decision 3: the hand-off branch completes the analyze job with `result={"analysis_error": ...}`, and its `test_audio_analyze_provider_setup_failure_keeps_the_transcript` asserts that result. Task 4's `hand_off` takes over that branch, so it carries the result; never edit that test to make Task 4 pass. (Review B1)

  Line numbers below are `origin/dev` `e1f7770953`; the revision re-measured the ones it cites on `32eb11ae03`, and they match. Where that plan moved a line, find the landmark text quoted in the step.
- **The analysis PR and the OpenAPI fingerprint.** That plan changes the `api_name` and `perform_analysis` descriptions on `SubmitAudioJobRequest`. The fingerprint hashes descriptions too, and that plan has no regeneration step. Task 8 Step 0 checks the fingerprint before this plan changes the API, so a stale one is caught before it is mixed with this PR's change. Flag the missing step to the analysis plan's executor. (Review M13)
- **Doc conflicts on rebase.** Task 4 edits `Audio_Jobs_API.md:40` and Task 5 adds a bullet after it. The analysis plan and `plan_summary_error_codes.md` (TASK-13844) both rewrite line 41. PR 2 is based after the analysis PR, so only TASK-13844 can conflict. The conflict is textual: keep both sides. (Review M14)
- **Test quality.** Since #3243, `backend-required` runs `tldw_Server_API/tests/CI/test_test_quality_ratchet.py`. A new test must not be status-only (asserting only `.status_code`) or tautological (asserting only a list its own stub appends to, or asserting inside a fake class). Each new test below also asserts on a job row or a result. Both close-out tasks run the ratchet. (Review M1)
- **Defaults stay:** `quota_exempt=False`; no new env vars; lease defaults unchanged (120 in-process, 180 GPU).
- **The transcript stays intact.** `JOBS_JSON_TRUNCATE` is not a fix (TASK-13851's note), and no change here enables it.
- No schema change and no migration.
- **Tests:** `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q <paths>`, with `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`.
  - Tests marked `jobs`/`pg_jobs` need `RUN_JOBS=1`.
  - The `postgres` cases need a reachable Postgres; they skip without one.
- **Bandit:** `uvx bandit -ll <files>`; the repo venv has no bandit.
- **Merging:** merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset.

## Decisions for the owner (recommendation first)

1. **TASK-13851 AC #1: how a long recording's payload stays under the cap.**
   - **Recommended (a): carry each form of the transcript once.**
     - `text` and `segments` stay in the payload, and `normalized_stt` keeps the artifact's other fields (`language`, `diarization`, `usage`, `metadata`) but not its second copy of `text` and `segments`.
     - One line in the shared transcribe handler; no reader changes. The chunk and analyze stages read `text`, and nothing reads `normalized_stt.text` or `normalized_stt.segments` (checked: `git grep normalized_stt` in app, tests and `apps/`).
     - At the default cap that is about 4 h with chunking (the default) and about 5.9 h without (table above). A longer recording fails at once at the hand-off with the sizes in the error (Task 4), and operators can raise `JOBS_MAX_JSON_BYTES`.
     - Revisit when `audio_store` gets real storage (it is still a placeholder), which is where the transcript belongs.
   - (b) **Segments only.** Also drop `text`; the chunk and analyze stages already rebuild it from `segments`. That gives about 5 h / 8.3 h, but `text` disappears from the stored result, and the GPU test's `payload.get("text")` assertion changes.
     - The code change is to delete `updated_payload["text"] = text_merged` in `_handle_audio_transcribe_stage`.
     - Task 5's test then asserts `"text" not in stored` instead of comparing `text`.
   - (c) **Store the transcript outside the payload**, as a Jobs attachment (`add_job_attachment(kind="artifact")`) or a file next to the WAV. This has no length ceiling, but it raises three new questions:
     - Attachments are not encrypted at rest when Jobs payload encryption is on.
     - Neither store has a retention rule.
     - The stored result becomes a reference.

     It would need its own plan.
2. **TASK-13852 AC #2: cancelling a submitted pipeline as a whole.**
   - **Recommended (a): add `DELETE /api/v1/audio/jobs/{job_id}`** for the owner or an admin, with the same checks as `GET /jobs/{job_id}`.
     - It cancels every queued or running stage of the pipeline the job belongs to. `job_id` may be the submitted id or any stage's id.
     - Stages find each other through `batch_group = "audio-pipeline:<uuid4 hex>"`, set at submit and copied at each hand-off. The column is indexed and `list_jobs(batch_group=...)` already filters on it on both backends.
     - This follows the media ingest cancel routes (`DELETE /media/ingest/jobs/{job_id}`, `POST /media/ingest/jobs/cancel` by `batch_group`).
     - It repeats list-then-cancel until a pass finds no queued or running stage, at most 6 passes, and sums `cancelled` across them. A stage that hands off between a pass's list and its cancel has queued the next stage, and the next pass finds it. (Review I1)
     - It handles only the six pipeline job types (`audio_download` … `audio_store`). Any other job in the `audio` domain, such as a long-form TTS job (`tts_longform`), gets 404: its id is not one this route handles. Cancelling TTS jobs stays out of scope. (Review I2)
     - Response `{job_id, cancelled}`; 400 only when the first pass finds nothing queued or running; 403 and 404 as for GET.
     - Pipelines submitted before PR 2 have no `batch_group`, so only the named job is cancelled.
   - (b) **Document only.** Users have no cancel today. An admin can cancel one stage with `POST /api/v1/jobs/batch/cancel {"domain": "audio", "job_id": <stage id>}` and `X-Confirm: true`, after finding the active stage in `GET /api/v1/audio/jobs/admin/list?owner_user_id=…`. With AC #1's fix, cancelling the active stage stops the rest. Under (b), Task 8 is replaced by that paragraph in `Audio_Jobs_API.md`.
3. **TASK-13853 AC #1: do stage rows count toward the user's next submission?**
   - **Recommended (a): yes, and document it.** The hand-off itself is never refused (both options do that, which is AC #2).
     - Each stage is a real job row of the user. It counts in the per-minute window for 60 s after it is queued, and a pipeline waiting between stages counts as one queued job, as its submission did.
     - The AC allows this outcome ("or the decision to count it is documented").
     - The cost: with a per-minute quota below about 4, a user's next submission can be refused in the minute after their pipeline's last stages are queued.
   - (b) **Exclude stage rows from the per-minute window.** Every hand-off already writes a marker on its row: `idempotency_key = 'audio-stage:<id>:<type>'`, the GPU worker's included after PR 2. So `count_recent_job_admissions` (`DB_Management/jobs_failed_requeue.py:197`) could skip rows with that prefix, with no schema change. The cost is an audio-specific prefix in generic Jobs code, which is why (a) stays the recommendation. A new `jobs` column would be the generic form, but it touches both backends, the archive table and every insert column list. Either would be a separate task. (Review M12)

## ADR assessment

- No new ADR is needed. Each change is additive with unchanged defaults, and none sets a new durable rule:
  - `quota_exempt` mirrors `fair_share_exempt`, with the same justification and default.
  - The new route follows the existing per-domain cancel routes.
  - Lease renewal is a local loop of the kind `core_jobs_worker.py` already runs. (Review I3)
  - There is no persistence change.
- Governing ADRs, all respected:
  - **ADR-003:** user-visible work stays on Jobs.
  - **ADR-011:** the new route authenticates through `get_request_user`, like the other audio routes.
  - **ADR-021 (Accepted):** stop-event workers with bounded shutdown. The renewal task is a child of a stop-event worker, and the loop's `finally` cancels it on every exit path. (Review M11)
  - **ADR-058:** completion still goes through `complete_job`'s locked row identity.
  - **ADR-069 (Proposed):** the `@tldw/api-types` contract. Task 8 regenerates the committed fingerprint, and the path union is unchanged. (Review M11)
- **CI selection.** PR 2 touches `apps/packages/api-types/` and `tldw_Server_API/app/api/v1/endpoints/**`, so it also selects the frontend checks and the E2E backend tiers. Expect those runs. (Review M11)
- **ADR-064 does not apply.** `JOBS_QUOTA_*` are Jobs admission quotas, not `limits.*` usage quotas.

## Review Focus

Failure modes the tests do not exercise, most likely first:

1. **Postgres.**
   - The worker tests run on SQLite only: lease renewal, `hand_off`, and the cancel route's `list_jobs(batch_group=...)` plus `cancel_job` on a `processing` row. Only `quota_exempt` has a Postgres case.
   - The GPU worker's renewal and hand-off are covered only through the shared helpers and two GPU tests (a refused hand-off, and a job without an owner).
2. **Payload encryption on for audio** (`JOBS_ENCRYPT` or `JOBS_ENCRYPT_AUDIO`). The cap is checked on the encrypted envelope (base64), so Decision 1's ceilings shrink by roughly a quarter. This is an estimate; it was not measured.
3. **A race with cancellation.** If the current stage is cancelled, another worker can still acquire the next stage in the milliseconds between `create_job` and `complete_job`. That stage runs, and the pipeline stops after it, because its own hand-off finds it cancelled. At most one extra stage runs.
   - `DELETE`'s repeated passes close a second window: a stage that hands off between the route's list and its cancel. The next pass finds the stage it queued (Task 8's race test). (Review I1)
   - What remains is the window above. A stage the route cancels while it runs still queues its next stage before `hand_off` sees the cancel, and another worker can acquire that stage first. By then the route has returned.
   - The passes stop at 6. A pipeline has six stages, and each pass that finds a stage either cancels it or finds the stage it handed off to on the next pass, so the cap is not reached in practice.
4. **A pipeline submitted before PR 2** (no `batch_group`) whose first stage has completed: `DELETE` returns 400 although later stages still run. Cancel by the running stage's id, from `/jobs/admin/list`.
5. **`JOBS_SECRET_REJECT=true` with a transcript that looks like it contains a secret.** The pipeline now fails at the hand-off the first time, where before it ran the stage twice. That is correct under that setting, but users will see it.
6. **A failed renewal with an even lease.** The renewal runs every `max(1, lease // 2)` seconds. After one failed renewal, the next one falls at about the moment the lease expires (120 s at the default). Another worker's acquire can recover the job in the few milliseconds between, and two failures in a row lose it for certain. Task 1's test uses a 3 s lease (renewal every second) so that it is deterministic. A shorter interval, such as a third of the lease, would leave margin; the owner chose half. (Review I3)
7. **`JOBS_DISABLE_LEASE_ENFORCEMENT=true`.** Renewal then binds only `id` and `status`, so it can extend a lease another worker now holds. These are that mode's existing semantics. (Review M15)
8. **Timing on a loaded runner.** The lease tests leave about 1 s between a renewal and the event it is measured against. The review measured 0.5 s margins in the first draft; Task 1 widens them. A runner stalled for a second or more can still fail them. (Review M15)

---

## PR 1: keep the lease; hand-offs are not submissions

Branch: `git switch -c fix/audio-jobs-lease-and-quotas origin/dev` (after the analysis-stage PR merged).

### Task 1: renew the stage lease while it runs (TASK-13850)

**Files:**
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py` (new helpers after `_StageFailure`, line 34; loop at 124-125, 319, 463). No new import. (Review M9)
- Modify: `tldw_Server_API/app/services/audio_transcribe_gpu_worker.py` (imports; loop at 94-96, 119, 141)
- Modify: `Docs/API-related/Audio_Jobs_API.md` (`JOBS_LEASE_SECONDS` line, 46)
- Test: `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py`

**Interfaces:**
- Consumes: `JobManager.renew_job_lease(job_id, *, seconds, worker_id, lease_id) -> bool` (`manager.py:6519` on dev). With lease enforcement on, the default, it binds `worker_id`, `lease_id` and `status='processing'`, so it returns False once the lease is lost or the job is cancelled or finished. (Review I3)
- Produces, in `audio_jobs_worker.py`:
  - `start_lease_renewal(jm: JobManager, job: dict[str, Any], *, worker_id: str, lease_seconds: int) -> asyncio.Task`. Its task sleeps `max(1, lease_seconds // 2)`, renews, returns on `False`, and logs a backend exception and carries on. It does not use the private `WorkerSDK._auto_renew`, which stops at the first exception. (Review I3, M9)
  - `async stop_lease_renewal(task: asyncio.Task | None) -> None`. It cancels the task and waits with `asyncio.wait`, which does not raise the task's cancellation, so a cancellation of the caller that arrives during the wait still propagates. (Review M7)

- [ ] **Step 1: failing tests.**
  - In `test_audio_worker_smoke.py`, add `import time` after `import tempfile`.
  - Generalize `_run_worker_until` (line 797) to run several worker loops, as several server processes do. Existing callers are unchanged: the default of 5 s is today's 100 × 0.05 s, and the shutdown wait grows from 2 s to 10 s, because a worker may be in the middle of a 3 s stage.

```python
async def _run_worker_until(worker, done, *, workers: int = 1, seconds: float = 5.0) -> None:
    """Run ``workers`` audio worker loops (as separate server processes would) until ``done()`` or ``seconds``."""
    stop = asyncio.Event()
    tasks = [asyncio.create_task(worker.run_audio_jobs_worker(stop)) for _ in range(workers)]
    try:
        for _ in range(int(seconds / 0.05)):
            await asyncio.sleep(0.05)
            for task in tasks:
                if task.done():
                    pytest.fail(f"audio worker crashed: {task.exception()!r}")
            if done():
                break
    finally:
        stop.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=10.0)
```

  Append at the end of the file:

```python
# --- the stage lease (TASK-13850) ---


def _stored(jm) -> bool:
    return any(j.get("status") == "completed" for j in _jobs_of(jm, "audio_store"))


@pytest.mark.unit
async def test_a_stage_longer_than_its_lease_runs_once(pipeline_worker, monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    worker, jm, stt_calls = pipeline_worker
    monkeypatch.setenv("JOBS_LEASE_SECONDS", "2")

    def _slow_stt(wav_path, model, *args, **kwargs):
        stt_calls.append(wav_path)
        time.sleep(3)  # longer than the lease
        return {"text": "hello pipeline", "segments": [{"Text": "hello pipeline"}]}

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _slow_stt)
    _convert_job(jm)
    # Two worker loops, as two server processes would run them: the second must not recover the lease.
    await _run_worker_until(worker, lambda: _stored(jm), workers=2, seconds=15)
    assert len(stt_calls) == 1
    assert [(j["status"], j["retry_count"]) for j in _jobs_of(jm, "audio_transcribe")] == [("completed", 0)]


@pytest.mark.unit
@pytest.mark.parametrize("stage_fails", [False, True], ids=["completes", "fails"])
async def test_lease_renewal_stops_when_the_stage_ends(pipeline_worker, monkeypatch, stage_fails):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    worker, jm, stt_calls = pipeline_worker
    monkeypatch.setenv("JOBS_LEASE_SECONDS", "4")  # a renewal every 2 s
    renewed = []
    original_renew = JobManager.renew_job_lease

    def _recording_renew(self, job_id, **kwargs):
        renewed.append(int(job_id))
        return original_renew(self, job_id, **kwargs)

    monkeypatch.setattr(JobManager, "renew_job_lease", _recording_renew)

    def _stt(wav_path, model, *args, **kwargs):
        time.sleep(3)  # one renewal, at 2 s
        if stage_fails:
            raise RuntimeError("stt failed")
        return {"text": "hello pipeline", "segments": [{"Text": "hello pipeline"}]}

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _stt)
    job_id = int(jm.create_job(
        domain="audio", queue="default", job_type="audio_transcribe",
        payload={"wav_path": "/tmp/p.wav", "model": "whisper-1", "perform_chunking": False, "perform_analysis": False},
        owner_user_id="7", max_retries=0,
    )["id"])
    await _run_worker_until(worker, lambda: (jm.get_job(job_id) or {}).get("status") in {"completed", "failed"})
    await asyncio.sleep(2)  # a renewal still running would fire again at 4 s
    assert (jm.get_job(job_id) or {}).get("status") == ("failed" if stage_fails else "completed")
    assert renewed.count(job_id) == 1


@pytest.mark.unit
async def test_a_renewal_error_does_not_end_the_renewals(pipeline_worker, monkeypatch):
    import sqlite3

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    worker, jm, stt_calls = pipeline_worker
    # A 3 s lease is renewed every second (3 // 2): the renewal after the failed one still comes a second
    # before the lease runs out.
    monkeypatch.setenv("JOBS_LEASE_SECONDS", "3")
    renewals = []
    original_renew = JobManager.renew_job_lease

    def _first_renewal_fails(self, job_id, **kwargs):
        renewals.append(int(job_id))
        if len(renewals) == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_renew(self, job_id, **kwargs)

    monkeypatch.setattr(JobManager, "renew_job_lease", _first_renewal_fails)

    def _slow_stt(wav_path, model, *args, **kwargs):
        stt_calls.append(wav_path)
        time.sleep(4)  # longer than the lease
        return {"text": "hello pipeline", "segments": [{"Text": "hello pipeline"}]}

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _slow_stt)
    _convert_job(jm)
    await _run_worker_until(worker, lambda: _stored(jm), workers=2, seconds=20)
    assert len(stt_calls) == 1
    assert [(j["status"], j["retry_count"]) for j in _jobs_of(jm, "audio_transcribe")] == [("completed", 0)]
```

  - The renewal test leaves about 1 s on each side: a renewal at 2 s, the stage ending at 3 s, and the check at about 5 s, after a leftover renewal would have fired at 4 s. The first draft used a 2 s lease and 0.5 s margins. (Review M15)
  - The renewal-error test needs an odd lease. With `JOBS_LEASE_SECONDS=2`, the renewal after a failed one falls at the moment the lease expires, and the result would depend on which timer fires first (Review Focus #6). (Review I3)

- [ ] **Step 2: run them; expect RED.**
  - Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k "longer_than_its_lease or renewal_stops or renewal_error"`
  - Expected: 4 failed.
    - `test_a_stage_longer_than_its_lease_runs_once`: `AssertionError: assert N == 1`, with N 2 or 3. Another worker recovers the expired lease and runs STT again, and recovery can repeat until the stage completes. The revision's dry run saw 2 and the review's saw 3. (Review M2)
    - Both `test_lease_renewal_stops_when_the_stage_ends` cases: `assert 0 == 1` (no renewal at all).
    - `test_a_renewal_error_does_not_end_the_renewals`: `AssertionError: assert N == 1`, with N 2 or more; the dry run saw 2. (Review I3)
- [ ] **Step 3: implement in `audio_jobs_worker.py`.** No new import: `asyncio`, `logger` and `JobManager` are already imported. Add after the `_StageFailure` class: (Review I3, M7, M9)

```python
def start_lease_renewal(jm: JobManager, job: dict[str, Any], *, worker_id: str, lease_seconds: int) -> asyncio.Task:
    """Renew the job's lease every half lease while its stage runs, so no other worker runs the stage again.

    The caller stops it with ``stop_lease_renewal`` when the stage ends. It stops by itself once a renewal is
    refused: the lease was lost, or the job was cancelled. A backend error (a locked database, a dropped
    connection) is logged and the next renewal is tried, so one error does not let a long stage's lease expire.
    """
    job_id = int(job["id"])
    lease_id = str(job.get("lease_id"))

    async def _renew() -> None:
        while True:
            await asyncio.sleep(max(1, lease_seconds // 2))
            try:
                if not jm.renew_job_lease(job_id, seconds=lease_seconds, worker_id=worker_id, lease_id=lease_id):
                    return
            except Exception as e:  # noqa: BLE001 - a backend error must not end renewal; the next one may pass
                logger.warning("Audio job {}: lease renewal failed ({}); trying again", job_id, type(e).__name__)

    return asyncio.create_task(_renew())


async def stop_lease_renewal(task: asyncio.Task | None) -> None:
    """Stop a renewal started by ``start_lease_renewal``; a no-op for None.

    ``asyncio.wait`` does not raise the renewal's own cancellation, so a cancellation of the caller that
    arrives while it waits still propagates.
    """
    if task is None:
        return
    task.cancel()
    await asyncio.wait({task})
```

  - The loop catches `Exception`, not a tuple: a backend error can be `sqlite3.Error` or `psycopg.Error`, neither of which is in `core_jobs_worker`'s tuple. `CancelledError` is not an `Exception`, so cancellation still ends the task. The log names only the error type, as `hand_off`'s does (Review M6).

  - In `run_audio_jobs_worker`, the per-iteration state (lines 124-125) gains `renewal: asyncio.Task | None = None` after `acquired_slot = False`.
  - Just before `payload: dict[str, Any] = job.get("payload") or {}` (line 319), after the concurrency slot is taken, add:

```python
            renewal = start_lease_renewal(jm, job, worker_id=worker_id, lease_seconds=lease_seconds)
```

  - The loop's `finally:` (line 463) starts with:

```python
        finally:
            await stop_lease_renewal(renewal)
```

  - In `audio_transcribe_gpu_worker.py`, the import from `audio_jobs_worker` (the analysis plan adds it for `stage_after_transcription`, after the `audio_quota` import) becomes:

```python
from tldw_Server_API.app.services.audio_jobs_worker import (
    stage_after_transcription,
    start_lease_renewal,
    stop_lease_renewal,
)
```

  - In the GPU loop, add `renewal: asyncio.Task | None = None` after `acquired_slot = False` (line 96).
  - Add `renewal = start_lease_renewal(jm, job, worker_id=worker_id, lease_seconds=lease_seconds)` just before `payload: dict[str, Any] = job.get("payload") or {}` (line 119).
  - Its `finally:` (line 141) also starts with `await stop_lease_renewal(renewal)`.
- [ ] **Step 4: run them; expect GREEN.**
  - Run the Step 2 command. Expected: `4 passed`, in about 20 s (3.5 s, 5 s, 5 s and 4 s). Three runs in a row passed in the dry run.
  - Then run the whole file and `tldw_Server_API/tests/Audio/test_audio_jobs_worker_helper.py`. Expected: all pass, 1 skipped (the ffmpeg smoke test). Record the counts. (Review M10)
- [ ] **Step 5: mutation checks.** Run each one alone; together, a renewal left running by the first steals the second's failing call.
  - Replace `await stop_lease_renewal(renewal)` in `audio_jobs_worker.py`'s `finally:` with `pass`, and rerun `-k renewal_stops`. Expected: 2 failed, `assert 2 == 1`. A renewal left running fires once more, after the stage has ended. Restore the line.
  - Replace the `logger.warning(...)` line in `_renew`'s `except` with `return` (WorkerSDK's behaviour), and rerun `-k renewal_error`. Expected: 1 failed, `assert 2 == 1`: the lease expires and another worker runs STT again. Restore the line. (Review I3)
- [ ] **Step 6: docs.** In `Docs/API-related/Audio_Jobs_API.md`, replace `` - `JOBS_LEASE_SECONDS`: lease duration (default 120)`` with:

```markdown
  - `JOBS_LEASE_SECONDS`: lease duration (default 120). The worker renews it every half lease while a stage runs, so another worker does not pick up a long transcription and run it again. The GPU worker does the same, with its own default of 180.
```

- [ ] **Step 7: commit.**

```bash
git add tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/app/services/audio_transcribe_gpu_worker.py \
  tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py Docs/API-related/Audio_Jobs_API.md
git commit -m "fix(audio-jobs): renew the stage lease while a stage runs, so a long transcription runs once (TASK-13850)" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: stage hand-offs are exempt from the submit quotas (TASK-13853)

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/manager.py` (`admit_job` 3498-3958: signature 3517, docstring 3536-3538, before 3769, arguments at 3826-3827 and 3917-3922; `create_job` 3960-4002)
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py` (hand-off `create_job`, 423-432)
- Modify: `tldw_Server_API/app/services/audio_transcribe_gpu_worker.py` (`create_job`, 125-132)
- Modify: `Helper_Scripts/ci/scope_predicate_baseline.txt` (regenerated)
- Modify: `Docs/API-related/Audio_Jobs_API.md`, `Docs/Code_Documentation/Jobs_Module.md`
- Test: `tldw_Server_API/tests/Jobs/test_fair_share_exempt.py`, `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py`

**Interfaces:**
- Produces: `JobManager.create_job(..., quota_exempt: bool = False)` and `JobManager.admit_job(..., quota_exempt: bool = False)`. When true, both `JOBS_QUOTA_MAX_QUEUED*` and `JOBS_QUOTA_SUBMITS_PER_MIN*` are treated as 0 (unset) for this create.

- [ ] **Step 1: failing tests.**
  - In `tests/Jobs/test_fair_share_exempt.py`, after the imports, add a fixture that runs on both backends. It uses the same shape as `test_failed_job_requeue_admission.py`'s `jobs` fixture.

```python
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_isolated_fixtures"]
USE_SHARED_JOBS_POSTGRES = True


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.pg_jobs)])
def jobs(request, tmp_path, monkeypatch) -> JobManager:
    if request.param == "postgres":
        return JobManager(backend="postgres", db_url=request.getfixturevalue("jobs_pg_dsn"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    return JobManager(db_path=tmp_path / "jobs.db")
```

  At the end of the file:

```python
@pytest.mark.unit
@pytest.mark.parametrize(
    ("quota", "refusal"),
    [("JOBS_QUOTA_SUBMITS_PER_MIN_AUDIO", "submits per minute"), ("JOBS_QUOTA_MAX_QUEUED_AUDIO", "max queued")],
)
def test_quota_exempt_skips_the_submit_quotas(jobs, monkeypatch, quota, refusal):
    monkeypatch.setenv(quota, "1")
    jobs.create_job(domain="audio", queue="default", job_type="audio_convert", payload={}, owner_user_id="7")

    with pytest.raises(ValueError, match=refusal):
        jobs.create_job(domain="audio", queue="default", job_type="audio_convert", payload={}, owner_user_id="7")

    exempt = jobs.create_job(
        domain="audio", queue="default", job_type="audio_transcribe", payload={}, owner_user_id="7", quota_exempt=True,
    )
    assert exempt["status"] == "queued"
```

  At the end of `test_audio_worker_smoke.py`:

```python
# --- stage hand-offs and the submit quotas (TASK-13853) ---


@pytest.mark.unit
async def test_audio_pipeline_runs_under_a_submit_quota_of_one(pipeline_worker, monkeypatch):
    worker, jm, stt_calls = pipeline_worker
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN_AUDIO", "1")
    convert_id = _convert_job(jm)
    await _run_worker_until(worker, lambda: _stored(jm) or (jm.get_job(convert_id) or {}).get("status") == "quarantined")
    assert [j["status"] for j in _jobs_of(jm, "audio_store")] == ["completed"]
    assert (jm.get_job(convert_id) or {}).get("retry_count") == 0


@pytest.mark.unit
async def test_a_hand_off_is_not_refused_by_the_queued_job_quota(pipeline_worker, monkeypatch):
    from datetime import datetime, timedelta, timezone

    worker, jm, stt_calls = pipeline_worker
    convert_id = _convert_job(jm)
    later = jm.create_job(  # the same user's next submission, queued for later
        domain="audio", queue="default", job_type="audio_convert",
        payload={"local_path": "/tmp/later.mp3", "model": "whisper-1"}, owner_user_id="7",
        available_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED_AUDIO", "1")  # that queued job fills the quota
    await _run_worker_until(worker, lambda: _stored(jm) or (jm.get_job(convert_id) or {}).get("status") == "quarantined")
    assert [j["status"] for j in _jobs_of(jm, "audio_store")] == ["completed"]
    assert (jm.get_job(int(later["id"])) or {}).get("status") == "queued"
```

- [ ] **Step 2: run them; expect RED.**
  - Run: `RUN_JOBS=1 TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Jobs/test_fair_share_exempt.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k "quota_exempt or submit_quota_of_one or queued_job_quota"`
  - Expected:
    - The quota-exempt cases fail with `TypeError: JobManager.create_job() got an unexpected keyword argument 'quota_exempt'`: 2 on SQLite, 2 more on Postgres when one is reachable.
    - Both pipeline tests fail with `AssertionError: assert [] == ['completed']`.
- [ ] **Step 3: implement in `manager.py`.**
  - In `admit_job`'s signature, after `fair_share_exempt: bool = False,` (line 3517), add `quota_exempt: bool = False,`.
  - In its docstring, after the `fair_share_exempt:` entry (ending "is about to complete (it replaces that job)."), add:

```python
            quota_exempt: Skip the submit quotas (JOBS_QUOTA_MAX_QUEUED*, JOBS_QUOTA_SUBMITS_PER_MIN*), for
                the same next stage: it continues a job the owner already submitted.
```

  - Just before `if self.backend == "postgres":` (line 3769, inside the `try:` after the job-type allowlist), add:

```python
            max_queued_quota = 0 if quota_exempt else self._quota_get("JOBS_QUOTA_MAX_QUEUED", domain, owner_user_id)
            submits_per_minute_quota = (
                0 if quota_exempt else self._quota_get("JOBS_QUOTA_SUBMITS_PER_MIN", domain, owner_user_id)
            )
```

  - In the Postgres call (lines 3826-3827), use:

```python
                    max_queued_quota=max_queued_quota,
                    submits_per_minute_quota=submits_per_minute_quota,
```

  - In the SQLite call (lines 3917-3922), replace the two arguments, including the five-line `submits_per_minute_quota=self._quota_get(...)`, with the same two lines, indented to that call.
  - In `create_job`'s signature, add `quota_exempt: bool = False,` after `fair_share_exempt: bool = False,` (line 3979). Pass `quota_exempt=quota_exempt,` after `fair_share_exempt=fair_share_exempt,` (line 4000).

  The workers:
  - In `audio_jobs_worker.py`'s hand-off `create_job` (`fair_share_exempt=True,`, line 431), add `quota_exempt=True,` after it.
  - In `audio_transcribe_gpu_worker.py`'s `create_job` (line 125), add `quota_exempt=True,` after `priority=5,`.
- [ ] **Step 4: run them; expect GREEN.**
  - Run the Step 2 command. Expected: all pass. The Postgres cases pass when Postgres is reachable and skip otherwise.
  - Then run `$PY Helper_Scripts/ci/scope_predicate_ratchet.py`. Expected: it fails with "Baseline entries that no longer match the tree" for `manager.py:5522` … `:6343`.
  - Run `$PY Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline`. Expected: `Recorded 28 conditional scope predicates.`
  - Rerun without the flag. Expected: `28 conditional scope predicates; baseline exact.`
- [ ] **Step 5: docs (Decision 3, recommendation a).**
  - In `Docs/API-related/Audio_Jobs_API.md` under `## Quotas & Fairness`, add after the first bullet:

```markdown
- The Jobs submit quotas (`JOBS_QUOTA_SUBMITS_PER_MIN*`, `JOBS_QUOTA_MAX_QUEUED*`) apply to `/jobs/submit` only; a stage hand-off is never refused by them. Each stage is still a job of the submitting user, so a stage queued in the last 60 seconds counts toward that user's per-minute quota, and a pipeline waiting between stages counts as one queued job.
```

  - In `Docs/Code_Documentation/Jobs_Module.md`, after `  - Precedence: domain+user > user > domain > global.`, add:

```markdown
  - `create_job(..., quota_exempt=True)` skips the queued and per-minute checks for a job that continues one its owner already submitted, such as an audio pipeline's next stage. The job still counts toward the owner's later checks.
```

- [ ] **Step 6: commit.**

```bash
git add tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/services/audio_jobs_worker.py \
  tldw_Server_API/app/services/audio_transcribe_gpu_worker.py Helper_Scripts/ci/scope_predicate_baseline.txt \
  tldw_Server_API/tests/Jobs/test_fair_share_exempt.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py \
  Docs/API-related/Audio_Jobs_API.md Docs/Code_Documentation/Jobs_Module.md
git commit -m "fix(audio-jobs): a stage hand-off is not refused by the user's submit quotas (TASK-13853)" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 3: PR 1 close-out

- [ ] **Mirrors.** Run `bash Helper_Scripts/refresh_docs_published.sh`. `git status --short Docs/Published` should list exactly `Docs/Published/API-related/Audio_Jobs_API.md` and `Docs/Published/Code_Documentation/Jobs_Module.md`. Then run `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs`.
- [ ] **Sweep.** Run every test that touches the changed code. Expected: all pass; record the counts. The revision's dry run gave `450 passed, 70 skipped` in 3.5 minutes; the number moves with the analysis PR's test count. (Review M10)

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/AudioJobs \
  tldw_Server_API/tests/Audio/test_audio_jobs_worker_helper.py tldw_Server_API/tests/Jobs/test_fair_share_exempt.py \
  tldw_Server_API/tests/Jobs/test_failed_job_requeue_admission.py tldw_Server_API/tests/Jobs/test_jobs_quotas_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_precedence_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_quotas_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_precedence_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_slides_generation_coordination_sqlite.py \
  tldw_Server_API/tests/Jobs/test_worker_owner_fallback_fails_closed.py tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  tldw_Server_API/tests/VN_Assets/test_parent_job_recovery.py \
  tldw_Server_API/tests/prompt_studio/unit/test_prompt_studio_quota_config.py \
  tldw_Server_API/tests/Services/test_shutdown_audio_jobs_worker.py tldw_Server_API/tests/Services/test_main_lifecycle_contract.py \
  tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py \
  tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py
```

  Then run the sweep's Jobs files with `RUN_JOBS=1`, plus the TTS worker test. **It can take up to 17 minutes** (the review's run; the revision's took 3). Expected: all pass; record the count (both dry runs: `272 passed`, with a local Postgres reachable). (Review M10)

```bash
RUN_JOBS=1 TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Jobs/test_fair_share_exempt.py \
  tldw_Server_API/tests/Jobs/test_failed_job_requeue_admission.py tldw_Server_API/tests/Jobs/test_jobs_quotas_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_precedence_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_quotas_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_precedence_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_slides_generation_coordination_sqlite.py \
  tldw_Server_API/tests/Jobs/test_worker_owner_fallback_fails_closed.py tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  "tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py::TestTTSJobsHistoryIntegration::test_audio_jobs_worker_leaves_a_submitted_speech_job_alone"
```

  - Record a baseline on unchanged `origin/dev` for any failure before blaming the change.
- [ ] **Required ratchets.** Run the five tests of `backend-required`'s "Enforce tenant isolation ratchets" step, as CI runs them. `test_test_quality_ratchet.py` is required on dev since #3243; it fails on a new status-only or tautological test. Expected: all pass (the revision's dry run: `53 passed`, in 3.5 minutes). (Review M1)

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
  $PY -m pytest -p no:cacheprovider -q --disable-warnings -p pytest_asyncio.plugin \
  tldw_Server_API/tests/CI/test_rls_coverage_ratchet.py tldw_Server_API/tests/CI/test_test_quality_ratchet.py \
  tldw_Server_API/tests/lint/test_route_auth_ratchet.py tldw_Server_API/tests/lint/test_rg_route_map_lint.py \
  tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py
```

  - If the test-quality ratchet fails, `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` names the file and flag. Fix the test; never raise a baseline count.
- [ ] **Lint and Bandit.**
  - `$PY -m ruff check tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/app/services/audio_transcribe_gpu_worker.py tldw_Server_API/app/core/Jobs/manager.py`: nothing new.
  - `uvx bandit -ll` on the same three files: nothing new. The workers are clean. `manager.py` has one pre-existing Medium B608, at line 9790 on dev and 9800 after the analysis PR and PR 1. (Review M3)
- [ ] **Backlog.** No task is created, so no new id is needed. Close both tasks:

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13850 -s Done --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "RED on dev: 4 failed (STT ran 2 or 3 times under a 2 s lease; no renewal at all, twice; STT ran twice under a 3 s lease whose first renewal raised sqlite3.OperationalError). GREEN: 4 passed. Mutations: removing the stop in the loop's finally fails both renewal cases (2 renewals); returning on a renewal error, as WorkerSDK._auto_renew does, fails the renewal-error test. Renewal is a local loop in audio_jobs_worker.py: it returns when a renewal is refused and logs a backend error and tries again. Bandit -ll: workers clean; manager.py has one pre-existing B608, nothing new." \
  --final-summary "Both audio workers renew the acquired job's lease every half lease while its stage runs and stop renewing when the stage ends, so a stage longer than JOBS_LEASE_SECONDS is not recovered and run again by another worker. A backend error during a renewal is logged and the next renewal tried; a refused renewal (lost lease, cancelled job) ends the loop."
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13853 -s Done --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "RED on dev: 4 failed on SQLite (TypeError quota_exempt twice; pipeline quarantined under JOBS_QUOTA_SUBMITS_PER_MIN_AUDIO=1; hand-off refused by JOBS_QUOTA_MAX_QUEUED_AUDIO=1 with another queued job), plus 2 Postgres TypeErrors when reachable. GREEN: all pass on both backends. Stage rows still count toward the user's next submission, documented in Audio_Jobs_API.md and Jobs_Module.md (owner Decision 3). scope_predicate_baseline.txt regenerated (manager.py lines +5). Bandit -ll: one pre-existing B608 in manager.py, nothing new." \
  --final-summary "create_job(quota_exempt=True) skips the queued and per-minute submit quotas; both audio workers pass it on every stage hand-off, so a submitted pipeline runs through every stage under JOBS_QUOTA_SUBMITS_PER_MIN_AUDIO=1. The docs say that stage rows still count toward the user's next submission."
git add -A backlog/tasks
git add Docs/Published
git commit -m "docs(audio-jobs): close TASK-13850 and TASK-13853; refresh Docs/Published" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

  If the observed RED or GREEN counts differ from the notes above, correct the notes before committing.
- [ ] **PR.**
  - Push, then open a PR against `dev`: `gh pr create --base dev`. The body says that PR 2 (TASK-13851/13852) follows, and why the tasks are split.
  - Before merge:
    - Run `/superpowers:requesting-code-review` in a clean subagent and address its findings.
    - Ask the owner for the ADR-004 Change summary. No recorded waiver covers TASK-13850–13853.
    - Merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset.

---

## PR 2: one hand-off for both workers

Branch: `git switch -c fix/audio-jobs-hand-off origin/dev`, after PR 1 has merged. Tasks 5 and 8 implement the recommended options of Decisions 1 and 2; switch them if the owner chose otherwise.

### Task 4: a refused hand-off fails at once; a transient one is retried in place (TASK-13851 AC #2)

**Files:**
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py` (new `_ENQUEUE_RETRY_DELAYS` and `hand_off` after `_next_stage_queue`; replace the hand-off block that starts `if ok and next_type:` and ends at the first `jm.complete_job(` after it)
- Modify: `tldw_Server_API/app/services/audio_transcribe_gpu_worker.py` (replace `# Complete and enqueue next stage` … `jm.create_job(...)`; the owner line, `owner = job.get("owner_user_id") or "0"`, line 102)
- Modify: `Docs/API-related/Audio_Jobs_API.md` (Worker bullet, line 40)
- Test: `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py`

**Interfaces:**
- Consumes: `_StageFailure`, `_next_stage_queue`, `create_job(..., fair_share_exempt=True, quota_exempt=True)` (PR 1), `stage_after_transcription` (analysis plan), and the analysis plan's Decision 3 result, `{"analysis_error": {...}}`.
- Produces: `_ENQUEUE_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0)` and `async hand_off(jm: JobManager, job: dict[str, Any], next_type: str, payload: dict[str, Any], *, worker_id: str, result: dict[str, Any] | None = None) -> _StageFailure | None`. Tasks 7 and 8 extend `hand_off`.
  - `result` is passed to `complete_job` unchanged, in every version of `hand_off`, Task 7's included. The in-process worker passes `{"analysis_error": ...}` when the payload has one, so the analyze job's `GET /api/v1/jobs/{id}` still shows why analysis failed. (Review B1)
  - A `ValueError` or `TypeError` from `create_job` is a refusal. (Review M5)
  - `hand_off` logs only the error's type and the job id; the `_StageFailure` message, which goes to the job row, keeps the detail. (Review M6)
- The GPU worker fails a job without an owner, as the in-process worker does (dev lines 216-218), instead of running it as user `"0"`. `hand_off` would otherwise queue the next stage for the owner `"None"`. (Review M8)

- [ ] **Step 1: failing tests.**
  - Rename `test_a_transient_hand_off_failure_is_retried` (line 1070) to `test_a_hand_off_that_keeps_failing_retries_the_stage`. It still fails `create_job` every time, so it now pins the last resort. Add as its first line after `worker, jm, stt_calls = pipeline_worker`:

```python
    monkeypatch.setattr(worker, "_ENQUEUE_RETRY_DELAYS", (0.0, 0.0), raising=False)
```

  - Append:

```python
# --- long recordings and refused hand-offs (TASK-13851) ---


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env", "value", "segment_extra"),
    [
        ("JOBS_MAX_JSON_BYTES", "600", {}),
        ("JOBS_ALLOWED_JOB_TYPES_AUDIO", "audio_convert,audio_transcribe", {}),
        ("JOBS_MAX_JSON_BYTES", "1048576", {"speakers": {"a", "b"}}),  # the default cap; a set is not JSON
    ],
    ids=["payload-over-the-cap", "job-type-not-allowed", "payload-not-json"],
)
async def test_a_refused_hand_off_fails_without_running_the_stage_again(
    pipeline_worker, monkeypatch, env, value, segment_extra
):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    worker, jm, stt_calls = pipeline_worker

    def _stt(wav_path, model, *args, **kwargs):
        stt_calls.append(wav_path)
        text = "hello pipeline " * 20  # puts the audio_store payload over a 600-byte cap
        return {"text": text, "segments": [{"Text": text, **segment_extra}]}

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _stt)
    monkeypatch.setenv(env, value)
    _convert_job(jm)
    await _run_worker_until(
        worker, lambda: any(j["status"] in {"failed", "quarantined"} for j in _jobs_of(jm, "audio_transcribe"))
    )
    assert [(j["status"], j["retry_count"], j["error_code"]) for j in _jobs_of(jm, "audio_transcribe")] == [
        ("failed", 0, "next_stage_enqueue_failed")
    ]
    assert len(stt_calls) == 1
    assert _jobs_of(jm, "audio_store") == []


@pytest.mark.unit
async def test_a_transient_enqueue_error_is_retried_without_running_the_stage_again(pipeline_worker, monkeypatch):
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    worker, jm, stt_calls = pipeline_worker
    monkeypatch.setattr(worker, "_ENQUEUE_RETRY_DELAYS", (0.0, 0.0), raising=False)
    logger_capture = _CaptureLogger()
    monkeypatch.setattr(worker, "logger", logger_capture)
    original_create = JobManager.create_job
    failed_once = []

    def _flaky_create(self, **kwargs):
        if kwargs.get("job_type") == "audio_store" and not failed_once:
            failed_once.append(True)
            raise RuntimeError("connection to server at db.internal (10.0.0.5) failed")
        return original_create(self, **kwargs)

    monkeypatch.setattr(JobManager, "create_job", _flaky_create)
    _convert_job(jm)
    await _run_worker_until(worker, lambda: _stored(jm))
    assert [(j["status"], j["retry_count"]) for j in _jobs_of(jm, "audio_transcribe")] == [("completed", 0)]
    assert len(stt_calls) == 1
    # The log names the error type only: a backend error can carry host and user names.
    rendered_logs = _render_log_records(logger_capture.records)
    assert "RuntimeError" in rendered_logs
    assert "db.internal" not in rendered_logs


@pytest.mark.unit
async def test_audio_gpu_worker_fails_a_job_without_an_owner(monkeypatch, tmp_path):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.services.audio_transcribe_gpu_worker as gpu_worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs_gpu_ownerless.db"))
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")

    async def _yes(user_id: int):
        return True, ""

    async def _noop(user_id: int):
        return None

    monkeypatch.setattr(gpu_worker, "can_start_job", _yes)
    monkeypatch.setattr(gpu_worker, "increment_jobs_started", _noop)
    monkeypatch.setattr(gpu_worker, "finish_job", _noop)
    stt_calls = []

    def _stt(wav_path, model, *args, **kwargs):
        stt_calls.append(wav_path)
        return {"text": "hello gpu", "segments": [{"Text": "hello gpu"}]}

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _stt)
    jm = JobManager()
    job_id = int(jm.create_job(
        domain="audio", queue="transcribe", job_type="audio_transcribe",
        payload={"wav_path": "/tmp/g.wav", "model": "whisper-1", "perform_chunking": False, "perform_analysis": False},
        owner_user_id=None,
    )["id"])
    stop = asyncio.Event()
    task = asyncio.create_task(gpu_worker.run_audio_transcribe_gpu_worker(stop))
    try:
        for _ in range(100):
            await asyncio.sleep(0.05)
            if (jm.get_job(job_id) or {}).get("status") not in {"queued", "processing"}:
                break
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=5.0)
    job = jm.get_job(job_id) or {}
    # Before: it ran as user "0"; through hand_off, its next stage would be queued for the owner "None".
    assert (job.get("status"), job.get("retry_count"), job.get("error_message")) == (
        "failed", 0, "missing owner_user_id"
    )
    assert stt_calls == []
    assert jm.list_jobs(domain="audio", job_type="audio_store") == []


@pytest.mark.unit
async def test_audio_gpu_worker_fails_a_stage_whose_hand_off_is_refused(monkeypatch, tmp_path):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.services.audio_transcribe_gpu_worker as gpu_worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs_gpu_refused.db"))
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("JOBS_ALLOWED_JOB_TYPES_AUDIO", "audio_transcribe")  # audio_store is refused

    async def _yes(user_id: int):
        return True, ""

    async def _noop(user_id: int):
        return None

    monkeypatch.setattr(gpu_worker, "can_start_job", _yes)
    monkeypatch.setattr(gpu_worker, "increment_jobs_started", _noop)
    monkeypatch.setattr(gpu_worker, "finish_job", _noop)
    stt_calls = []

    def _stt(wav_path, model, *args, **kwargs):
        stt_calls.append(wav_path)
        return {"text": "hello gpu", "segments": [{"Text": "hello gpu"}]}

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _stt)
    jm = JobManager()
    job_id = int(jm.create_job(
        domain="audio", queue="transcribe", job_type="audio_transcribe",
        payload={"wav_path": "/tmp/g.wav", "model": "whisper-1", "perform_chunking": False, "perform_analysis": False},
        owner_user_id="7",
    )["id"])
    stop = asyncio.Event()
    task = asyncio.create_task(gpu_worker.run_audio_transcribe_gpu_worker(stop))
    try:
        for _ in range(100):
            await asyncio.sleep(0.05)
            if (jm.get_job(job_id) or {}).get("status") not in {"queued", "processing"}:
                break
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=5.0)
    job = jm.get_job(job_id) or {}
    # Before: the stage completed first, so it read completed with nothing after it.
    assert (job.get("status"), job.get("retry_count"), job.get("error_code")) == ("failed", 0, "next_stage_enqueue_failed")
    assert len(stt_calls) == 1
```

  - The `payload-not-json` case: `json.dumps` raises `TypeError` for a set, and `admit_job` re-raises it as is. (Review M5)
  - The transient test's log assertions use `_CaptureLogger` (with the `warning` method the analysis plan adds) and `_render_log_records`, both at the top of the file. The three `test_audio_gpu_worker_sanitizes_*` tests patch only `gpu_worker.logger`, so they do not see `hand_off`'s log. (Review M6)
  - The ownerless test runs the GPU worker on a real Jobs DB, not a fake `JobManager`. Asserting only what a fake's `fail_job` collected would be a tautology to the test-quality ratchet. (Review M8, M1)

- [ ] **Step 2: run them; expect RED.**
  - Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k "fails_without_running_the_stage_again or retried_without_running or keeps_failing or whose_hand_off_is_refused or without_an_owner"`
  - Expected: 6 failed, 1 passed.
    - All three refused cases: `('quarantined', 2, 'next_stage_enqueue_failed') != ('failed', 0, 'next_stage_enqueue_failed')`.
    - The transient test: `assert [('completed', 1)] == [('completed', 0)]`.
    - The GPU refusal test: `('completed', 0, None) == ('failed', 0, 'next_stage_enqueue_failed')`.
    - The ownerless test: `assert ('completed', 0, None) == ('failed', 0, 'missing owner_user_id')`. The job ran as user `"0"`. (Review M8)
    - `keeps_failing` passes: it already holds on dev.
- [ ] **Step 3: implement.** In `audio_jobs_worker.py`, after `_next_stage_queue`:

```python
# Waits between attempts to queue the next stage. An error other than a refusal is retried this way, without
# running the stage again; after the last attempt the stage fails and is retried as a whole.
_ENQUEUE_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0)


async def hand_off(
    jm: JobManager,
    job: dict[str, Any],
    next_type: str,
    payload: dict[str, Any],
    *,
    worker_id: str,
    result: dict[str, Any] | None = None,
) -> _StageFailure | None:
    """Queue the next stage, then complete this one with ``result``. Return the failure to record, or None.

    The next stage is queued first, so a failed hand-off fails this stage instead of completing it with nothing
    after it. The key makes a re-run stage reuse the next stage it already queued, which also makes a repeated
    enqueue safe. The next stage continues this job, so it is exempt from the per-user fair-share limit and the
    submit quotas.
    """
    next_queue = _next_stage_queue(next_type)
    for delay in (*_ENQUEUE_RETRY_DELAYS, None):
        try:
            jm.create_job(
                domain=DOMAIN,
                queue=next_queue,
                job_type=next_type,
                payload=payload,
                owner_user_id=str(job.get("owner_user_id")),
                priority=5,
                idempotency_key=f"audio-stage:{job['id']}:{next_type}",
                fair_share_exempt=True,
                quota_exempt=True,
            )
            break
        except (TypeError, ValueError) as e:
            # A refusal (a payload over JOBS_MAX_JSON_BYTES or not JSON, a queue or job type that is not allowed,
            # a JOBS_SECRET_REJECT match) repeats every time: fail now instead of running the stage again.
            # The log names only the error type; the job row's error keeps the detail.
            logger.warning("Audio job {}: could not queue {} ({})", job["id"], next_type, type(e).__name__)
            return _StageFailure(
                f"could not queue {next_type}: {e}", failure_code="next_stage_enqueue_failed", retryable=False
            )
        except Exception as e:  # noqa: BLE001 - any other error may pass when the enqueue alone is retried
            logger.warning("Audio job {}: could not queue {} ({})", job["id"], next_type, type(e).__name__)
            if delay is None:
                return _StageFailure(
                    f"could not queue {next_type}: {e}", failure_code="next_stage_enqueue_failed", retryable=True
                )
            await asyncio.sleep(delay)
    lease_id = str(job.get("lease_id"))
    jm.complete_job(int(job["id"]), result=result, worker_id=worker_id, lease_id=lease_id, completion_token=lease_id)
    return None
```

  - Every `ValueError` or `TypeError` from `create_job` is a refusal that repeats. These are: queue not allowed, payload over the cap, job type not allowed, secret match, invalid execution controls (`ValueError`), and an unserializable payload (`TypeError`, from `json.dumps`). (Review M5)
  - Quota rejections are also `ValueError`, but PR 1 exempts hand-offs from them. Fair-share's `BadRequestError` is a `ValueError` too, and hand-offs are exempt from it.
  - Backend errors (`sqlite3.OperationalError`, psycopg errors, the `RuntimeError` of an admission conflict) are neither, so they are retried.
  - Both log lines name only the error type, because a psycopg error can carry host and user names; the `_StageFailure` message keeps the detail for the job row, as today. (Review M6)

  In `run_audio_jobs_worker`, replace the whole `if ok and next_type:` branch (its comment block, the `try: jm.create_job(...)`, the `except` that builds `_StageFailure(..., retryable=next_queue in jm._get_allowed_queues(DOMAIN))`, and the `else: jm.complete_job(...)`, which the analysis plan gave `result={"analysis_error": ...}`) with: (Review B1)

```python
            if ok and next_type:
                analysis_error = updated_payload.get("analysis_error")
                failure = await hand_off(
                    jm, job, next_type, updated_payload, worker_id=worker_id,
                    result={"analysis_error": analysis_error} if analysis_error else None,
                )
                if failure is not None:
                    ok = False
                    msg_err = str(failure)
```

  - In `audio_transcribe_gpu_worker.py`, extend the `audio_jobs_worker` import with `hand_off`, keeping the names sorted: `hand_off, stage_after_transcription, start_lease_renewal, stop_lease_renewal`.
  - Replace `owner = job.get("owner_user_id") or "0"` (line 102) with the in-process worker's check (dev lines 215-218): (Review M8)

```python
            owner = job.get("owner_user_id")
            if not owner:
                # As the in-process worker does: a job without an owner fails instead of running as user 0.
                jm.fail_job(int(job["id"]), error="missing owner_user_id", retryable=False, worker_id=worker_id, lease_id=str(job.get("lease_id")))
                continue
```

  - Replace the block from `# Complete and enqueue next stage` through the closing `)` of `jm.create_job(...)` with:

```python
            next_type = stage_after_transcription(payload)
            failure = await hand_off(jm, job, next_type, updated_payload, worker_id=worker_id)
            if failure is not None:
                lease_id = str(job.get("lease_id"))
                jm.fail_job(
                    int(job["id"]),
                    error=str(failure),
                    retryable=failure.retryable,
                    error_code=failure.failure_code,
                    worker_id=worker_id,
                    lease_id=lease_id,
                    completion_token=lease_id,
                )
```

- [ ] **Step 4: run them; expect GREEN.**
  - Run the Step 2 command. Expected: `7 passed`.
  - Then run the whole smoke file and `tests/Audio/test_audio_jobs_worker_helper.py`. Expected: all pass, 1 skipped; record the counts. (Review M10)
  - This run includes the analysis plan's `test_audio_analyze_provider_setup_failure_keeps_the_transcript`, which asserts the analyze job's `result["analysis_error"]["code"]`. Without the `result` pass-through, both its cases fail with `assert None == 'configuration'` (and `'authentication'`). Fix `hand_off`, never that test. (Review B1)
  - **Mutation check:** change `except (TypeError, ValueError) as e:` to `except ValueError as e:` and rerun `-k payload-not-json`. Expected: 1 failed, `('quarantined', 2, 'next_stage_enqueue_failed') != ('failed', 0, 'next_stage_enqueue_failed')`. Restore the line. (Review M5)
- [ ] **Step 5: docs.** In `Docs/API-related/Audio_Jobs_API.md`'s Worker bullet, replace "Only a refused queue fails it outright; any other refusal is retried by running the stage again, and the same failure twice in a row quarantines the job (`JOBS_QUARANTINE_THRESHOLD`)." with the text below. If TASK-13844 landed first and rewrote line 41, the rebase conflict is textual: keep both sides. (Review M14)

```markdown
A refusal (a payload over `JOBS_MAX_JSON_BYTES` or not JSON, a queue or job type the Jobs allowlists refuse, a `JOBS_SECRET_REJECT` match) fails it at once, without running the stage again. Any other error is retried twice, 1 and 2 seconds apart, without running the stage again; if it still fails, the stage is retried as a whole, and the same failure twice in a row quarantines the job (`JOBS_QUARANTINE_THRESHOLD`).
```

- [ ] **Step 6: commit.**

```bash
git add tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/app/services/audio_transcribe_gpu_worker.py \
  tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py Docs/API-related/Audio_Jobs_API.md
git commit -m "fix(audio-jobs): one hand-off for both workers; a refusal fails at once, a transient error is retried in place (TASK-13851)" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 5: carry the transcript once (TASK-13851 AC #1; Decision 1, recommendation a)

**Files:**
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py` (`_handle_audio_transcribe_stage`: docstring line 53, `normalized_stt` line 96)
- Modify: `tldw_Server_API/app/services/audio_transcribe_gpu_worker.py` (`_handle_gpu_audio_transcribe_stage`, lines 22-70; imports)
- Modify: `Docs/API-related/Audio_Jobs_API.md`
- Modify: `Docs/Product/STT_Module_IMPLEMENTATION_PLAN.md` (line 27; not mirrored under `Docs/Published`)
- Test: `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py`, `tldw_Server_API/tests/Audio/test_audio_jobs_worker_helper.py`

**Interfaces:**
- Consumes: `_handle_audio_transcribe_stage(payload) -> tuple[dict[str, Any], str | None]`.
- Produces: the payload shape. `text` and `segments` are unchanged; `normalized_stt` holds the artifact minus `text` and `segments`. The GPU worker builds the same shape through the same function.

- [ ] **Step 1: failing tests.**
  - In `test_audio_jobs_worker_helper.py`, change the docstring's "attach the full artifact as normalized_stt" to "keep the rest of the artifact (not its text and segments again) as normalized_stt".
  - Change its last assertion to:

```python
    assert updated_payload.get("normalized_stt") == {
        k: v for k, v in fake_artifact.items() if k not in ("text", "segments")
    }
```

  In `test_audio_worker_smoke.py`, under the TASK-13851 header:

```python
def _long_transcript(minutes: int) -> dict:
    """An STT artifact shaped like faster-whisper's: one segment every 5 s, about 150 words a minute."""
    segments = [
        {"start_seconds": 5.0 * i, "end_seconds": 5.0 * i + 5, "Time_Start": 5.0 * i, "Time_End": 5.0 * i + 5,
         "Text": f"Part {i} of the talk, where the speaker keeps on explaining the topic."}
        for i in range(minutes * 12)
    ]
    return {"text": " ".join(s["Text"] for s in segments), "language": "en", "segments": segments,
            "diarization": {"enabled": False, "speakers": None}, "usage": {"duration_ms": None, "tokens": None},
            "metadata": {"provider": "faster-whisper", "model": "large-v3"}}


@pytest.mark.unit
async def test_a_three_hour_recording_runs_through_every_stage(pipeline_worker, monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    worker, jm, stt_calls = pipeline_worker
    artifact = _long_transcript(180)

    def _long_stt(wav_path, model, *args, **kwargs):
        stt_calls.append(wav_path)
        return artifact

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _long_stt)
    jm.create_job(
        domain="audio", queue="default", job_type="audio_convert",
        payload={"local_path": "/tmp/long.mp3", "model": "whisper-1", "perform_chunking": True, "perform_analysis": False},
        owner_user_id="7",
    )
    await _run_worker_until(
        worker, lambda: _stored(jm) or any(j["status"] in {"failed", "quarantined"} for j in jm.list_jobs(domain="audio")),
        seconds=20,
    )
    assert [j["status"] for j in _jobs_of(jm, "audio_store")] == ["completed"], jm.list_jobs(domain="audio")
    assert len(stt_calls) == 1
    stored = _jobs_of(jm, "audio_store")[0]["payload"]
    assert stored["text"] == artifact["text"] and stored["segments"] == artifact["segments"]
    assert stored["chunks"]
```

- [ ] **Step 2: run them; expect RED.**
  - Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py tldw_Server_API/tests/Audio/test_audio_jobs_worker_helper.py -k "three_hour or normalized_artifact"`
  - Expected: 2 failed.
    - `three_hour`: `assert [] == ['completed']`. The job list shows the transcribe job failed with `could not queue audio_chunk: Payload too large: 1062513 bytes > limit 1048576`, failed at once because of Task 4.
    - The helper test: `normalized_stt` still equals the full artifact.
- [ ] **Step 3: implement.** In `_handle_audio_transcribe_stage`, replace the two comment lines and `updated_payload["normalized_stt"] = artifact` with:

```python
    # The artifact's metadata (language, usage, provider); its text and segments are already above, and
    # carrying them twice would put a long recording's payload over JOBS_MAX_JSON_BYTES.
    updated_payload["normalized_stt"] = {k: v for k, v in artifact.items() if k not in ("text", "segments")}
```

  - In the same function's docstring, replace `` - `normalized_stt`: full normalized STT artifact`` (line 53) with: (Review M4)

```python
      - `normalized_stt`: the rest of the STT artifact (language, diarization, usage, metadata)
```

  - In `Docs/Product/STT_Module_IMPLEMENTATION_PLAN.md:27`, replace "attach the normalized artifact to their `audio_transcribe` payloads (`normalized_stt`), and" with "attach the normalized artifact to their `audio_transcribe` payloads (`normalized_stt`, without the `text` and `segments` the payload already carries), and". The page is not mirrored under `Docs/Published`. (Review M4)
  - In `audio_transcribe_gpu_worker.py`, replace the body of `_handle_gpu_audio_transcribe_stage` (a copy of the in-process handler) with a call to the shared one. Keep the name: three GPU tests monkeypatch it.

```python
async def _handle_gpu_audio_transcribe_stage(payload: dict[str, Any]) -> dict[str, Any]:
    """The GPU `audio_transcribe` stage: the same transcription and payload as the in-process worker's."""
    updated_payload, _next_type = await _handle_audio_transcribe_stage(payload)
    return updated_payload
```

  - Add `_handle_audio_transcribe_stage` first in the `audio_jobs_worker` import list.
  - Delete the now-unused `from functools import partial` and `from pathlib import Path`.
- [ ] **Step 4: run them; expect GREEN.** Run the Step 2 command. Expected: `2 passed` (the 3-hour test takes about 2 s). Then run the whole smoke file and the helper file. Expected: all pass, 1 skipped; record the counts. That includes `test_audio_gpu_worker_normalizes_segments_and_text`. (Review M10)
- [ ] **Step 5: docs.** In `Audio_Jobs_API.md`, add a Worker bullet after the one Task 4 edited:

```markdown
- Each stage's payload carries the transcript once: `text`, `segments` with their timestamps, and `chunks` when chunking is on. `normalized_stt` keeps the STT artifact's other fields (`language`, `diarization`, `usage`, `metadata`). At the default `JOBS_MAX_JSON_BYTES` (1 MiB) that fits about 4 hours of speech with chunking, the default, and about 6 hours without. A longer recording fails at the hand-off that exceeds the cap, with `next_stage_enqueue_failed` and the sizes in the error; raise `JOBS_MAX_JSON_BYTES` for longer recordings.
```

- [ ] **Step 6: commit.**

```bash
git add tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/app/services/audio_transcribe_gpu_worker.py \
  tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py tldw_Server_API/tests/Audio/test_audio_jobs_worker_helper.py \
  Docs/API-related/Audio_Jobs_API.md Docs/Product/STT_Module_IMPLEMENTATION_PLAN.md
git commit -m "fix(audio-jobs): carry the transcript once, so a multi-hour recording fits the payload cap (TASK-13851)" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 6: a stored result over the cap fails `audio_store` (TASK-13851 AC #3)

**Files:**
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py` (the `elif ok:` completion, line 445-446)
- Modify: `Docs/API-related/Audio_Jobs_API.md`
- Test: `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py`

- [ ] **Step 1: failing test.**

```python
@pytest.mark.unit
async def test_audio_store_with_a_result_over_the_cap_fails_instead_of_staying_processing(pipeline_worker, monkeypatch):
    import json

    worker, jm, stt_calls = pipeline_worker
    payload = {"text": "x" * 5000, "model": "whisper-1"}
    # The payload fits; the stored result ({"payload": ...}) is 13 bytes larger and does not.
    monkeypatch.setenv("JOBS_MAX_JSON_BYTES", str(len(json.dumps(payload)) + 5))
    job_id = int(jm.create_job(domain="audio", queue="default", job_type="audio_store", payload=payload, owner_user_id="7")["id"])
    await _run_worker_until(worker, lambda: (jm.get_job(job_id) or {}).get("status") in {"completed", "failed"}, seconds=2)
    job = jm.get_job(job_id) or {}
    assert (job.get("status"), job.get("retry_count"), job.get("error_code")) == ("failed", 0, "result_too_large")
```

- [ ] **Step 2: run it; expect RED.**
  - Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k result_over_the_cap`
  - Expected: `assert ('processing', 0, None) == ('failed', 0, 'result_too_large')`.
- [ ] **Step 3: implement.** Replace the `elif ok:` branch with:

```python
            elif ok:
                try:
                    jm.complete_job(int(job["id"]), result={"payload": updated_payload}, worker_id=worker_id, lease_id=str(job.get("lease_id")), completion_token=str(job.get("lease_id")))
                except ValueError as e:
                    # A result over JOBS_MAX_JSON_BYTES is refused the same way every time: fail the job
                    # instead of leaving it processing until its lease expires.
                    ok = False
                    msg_err = str(e)
                    failure = _StageFailure(msg_err, failure_code="result_too_large", retryable=False)
```

  The `if not ok:` block right after it records the failure.
- [ ] **Step 4: run it; expect GREEN.** Run the Step 2 command. Expected: `1 passed`.
- [ ] **Step 5: docs.** Append to the bullet Task 5 added: ` The `audio_store` stage stores the payload as its result; a result over the cap fails that stage with `error_code` `result_too_large`.`
- [ ] **Step 6: commit.**

```bash
git add tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py \
  Docs/API-related/Audio_Jobs_API.md
git commit -m "fix(audio-jobs): a stored result over the cap fails audio_store instead of leaving it processing (TASK-13851)" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 7: cancelling a running stage stops the stages after it (TASK-13852 AC #1)

**Files:**
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py` (`hand_off`)
- Test: `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py` (new test; the fake `complete_job` in `test_audio_gpu_worker_sanitizes_owner_slot_release_log`, line 497-498)

- [ ] **Step 1: failing test.**
  - In `test_audio_gpu_worker_sanitizes_owner_slot_release_log`, change the fake's `complete_job` to `return True`. A real `complete_job` returns a bool, and `hand_off` now reads it.
  - Append:

```python
# --- cancelling a pipeline (TASK-13852) ---


@pytest.mark.unit
async def test_cancelling_a_running_stage_stops_the_stages_after_it(pipeline_worker, monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    worker, jm, stt_calls = pipeline_worker

    def _stt_cancelled_while_running(wav_path, model, *args, **kwargs):
        stt_calls.append(wav_path)
        assert jm.cancel_job(int(_jobs_of(jm, "audio_transcribe")[0]["id"]), reason="user")
        return {"text": "hello pipeline", "segments": [{"Text": "hello pipeline"}]}

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _stt_cancelled_while_running)
    _convert_job(jm)
    await _run_worker_until(
        worker, lambda: any(j["status"] not in {"queued", "processing"} for j in _jobs_of(jm, "audio_store"))
    )
    assert [j["status"] for j in _jobs_of(jm, "audio_transcribe")] == ["cancelled"]
    assert [j["status"] for j in _jobs_of(jm, "audio_store")] == ["cancelled"]
```

- [ ] **Step 2: run it; expect RED.**
  - Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k "stops_the_stages_after_it or owner_slot_release"`
  - Expected: 1 failed, `assert ['completed'] == ['cancelled']`; 1 passed.
- [ ] **Step 3: implement.** In `hand_off`:
  - Change `jm.create_job(` to `next_job = jm.create_job(`.
  - Replace its last three lines with the version below. It still passes `result` to `complete_job`. (Review B1)

```python
    lease_id = str(job.get("lease_id"))
    if not jm.complete_job(int(job["id"]), result=result, worker_id=worker_id, lease_id=lease_id, completion_token=lease_id):
        if (jm.get_job(int(job["id"])) or {}).get("status") == "cancelled":
            jm.cancel_job(int(next_job["id"]), reason="pipeline_cancelled")
    return None
```

  - Append to its docstring: "If this stage was cancelled while it ran, the stage just queued is cancelled too, so the pipeline stops here."
  - A `False` from `complete_job` for any other reason (the lease was recovered by another worker) leaves the next stage alone. The re-run reuses it through the idempotency key.
- [ ] **Step 4: run it; expect GREEN.** Run the Step 2 command. Expected: `2 passed`. Then run the whole smoke file. Expected: all pass, 1 skipped.
- [ ] **Step 5: commit.**

```bash
git add tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
git commit -m "fix(audio-jobs): cancelling a running stage cancels the stage it queued, so the pipeline stops (TASK-13852)" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 8: cancel a submitted pipeline as a whole (TASK-13852 AC #2; Decision 2, recommendation a)

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py` (imports; submit's `create_job`, 289-298; new route before `/jobs/{job_id}/progress/stream`, line 368)
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py` (`hand_off` copies `batch_group`)
- Modify: `apps/packages/api-types/openapi.fingerprint.json` (regenerated)
- Modify: `Docs/API-related/Audio_Jobs_API.md`
- Test: `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py`

**Interfaces:**
- Produces:
  - `CancelAudioJobResponse(job_id: int, cancelled: int)`.
  - `async cancel_audio_job(job_id, current_user, jm, principal) -> CancelAudioJobResponse`, served as `DELETE /api/v1/audio/jobs/{job_id}`.
    - It handles the six pipeline job types only (`_PIPELINE_JOB_TYPES` in `audio_jobs.py`). Any other job id, a `tts_longform` job's included, gets 404. (Review I2)
    - It repeats list-then-cancel until a pass finds no queued or processing stage, at most 6 passes, and returns the sum of `cancelled`. It returns 400 only when the first pass finds nothing active. (Review I1)
  - Submitted jobs carry `batch_group="audio-pipeline:<uuid4 hex>"`, which every hand-off copies.

- [ ] **Step 0: check the API contract before changing it.** Run `PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts`. Expected: exit 0. If it fails, the analysis PR changed `SubmitAudioJobRequest`'s descriptions without regenerating the fingerprint. Regenerate it in its own commit, `chore(api-types): regenerate the OpenAPI fingerprint for the analysis PR's field descriptions`, and tell the analysis plan's owner. (Review M13)
- [ ] **Step 1: failing tests.** The first test is the plan's original. The second races a hand-off with the route (Review I1). The third refuses a long-form TTS job (Review I2). Each asserts on job rows, not only on a status code (Review M1).

```python
@pytest.mark.unit
async def test_cancel_endpoint_cancels_the_rest_of_the_pipeline(monkeypatch, tmp_path):
    from fastapi import HTTPException

    import tldw_Server_API.app.services.audio_jobs_worker as worker
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_jobs
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager(db_path=tmp_path / "jobs_cancel.db")
    owner = User(id=7, username="owner", email="owner@example.com", role="user", is_active=True)
    stranger = User(id=8, username="stranger", email="stranger@example.com", role="user", is_active=True)
    no_claims = SimpleNamespace(roles=[], permissions=[])
    submitted = await audio_jobs.submit_audio_job(
        req=audio_jobs.SubmitAudioJobRequest(url="https://example.com/a.mp3", model="whisper-1"),
        current_user=owner, jm=jm, request=None,
    )
    # The first stage runs and hands over to the next one, as the worker would.
    first = jm.acquire_next_job(domain="audio", queue="default", lease_seconds=60, worker_id="w")
    assert first["id"] == submitted.id
    assert await worker.hand_off(jm, first, "audio_convert", {"local_path": "/tmp/a.mp3"}, worker_id="w") is None

    with pytest.raises(HTTPException) as refused:
        await audio_jobs.cancel_audio_job(job_id=submitted.id, current_user=stranger, jm=jm, principal=no_claims)
    assert refused.value.status_code == 403

    result = await audio_jobs.cancel_audio_job(job_id=submitted.id, current_user=owner, jm=jm, principal=no_claims)
    assert result.cancelled == 1
    statuses = {j["job_type"]: j["status"] for j in jm.list_jobs(domain="audio", owner_user_id="7")}
    assert statuses == {"audio_download": "completed", "audio_convert": "cancelled"}

    with pytest.raises(HTTPException) as finished:
        await audio_jobs.cancel_audio_job(job_id=submitted.id, current_user=owner, jm=jm, principal=no_claims)
    assert finished.value.status_code == 400


@pytest.mark.unit
async def test_cancel_endpoint_cancels_a_stage_queued_while_it_cancels(monkeypatch, tmp_path):
    import threading

    import tldw_Server_API.app.services.audio_jobs_worker as worker
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_jobs
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager(db_path=tmp_path / "jobs_cancel_race.db")
    owner = User(id=7, username="owner", email="owner@example.com", role="user", is_active=True)
    submitted = await audio_jobs.submit_audio_job(
        req=audio_jobs.SubmitAudioJobRequest(url="https://example.com/a.mp3", model="whisper-1"),
        current_user=owner, jm=jm, request=None,
    )
    first = jm.acquire_next_job(domain="audio", queue="default", lease_seconds=60, worker_id="w")
    list_jobs = jm.list_jobs

    def _list_then_the_worker_hands_off(**kwargs):
        stages = list_jobs(**kwargs)
        if (jm.get_job(int(first["id"])) or {}).get("status") == "processing":
            # The worker finishes the first stage right after the route listed it as running.
            hand_off = worker.hand_off(jm, first, "audio_convert", {"local_path": "/tmp/a.mp3"}, worker_id="w")
            runner = threading.Thread(target=asyncio.run, args=(hand_off,))
            runner.start()
            runner.join()
        return stages

    monkeypatch.setattr(jm, "list_jobs", _list_then_the_worker_hands_off)
    result = await audio_jobs.cancel_audio_job(
        job_id=submitted.id, current_user=owner, jm=jm, principal=SimpleNamespace(roles=[], permissions=[])
    )
    assert result.cancelled == 1
    statuses = {j["job_type"]: j["status"] for j in list_jobs(domain="audio", owner_user_id="7")}
    assert statuses == {"audio_download": "completed", "audio_convert": "cancelled"}


@pytest.mark.unit
async def test_cancel_endpoint_does_not_handle_other_audio_jobs(tmp_path):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints.audio import audio_jobs
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager(db_path=tmp_path / "jobs_cancel_tts.db")
    owner = User(id=7, username="owner", email="owner@example.com", role="user", is_active=True)
    tts_id = int(jm.create_job(
        domain="audio", queue="default", job_type="tts_longform", payload={"text": "hi"}, owner_user_id="7"
    )["id"])
    with pytest.raises(HTTPException) as refused:
        await audio_jobs.cancel_audio_job(
            job_id=tts_id, current_user=owner, jm=jm, principal=SimpleNamespace(roles=[], permissions=[])
        )
    assert refused.value.status_code == 404
    assert (jm.get_job(tts_id) or {}).get("status") == "queued"
```

  - The race test runs the real `hand_off` in a thread, inside the patched `list_jobs`, as the review's probe did with the route's single pass: stage 1 completes and stage 2 is queued between the route's first list and its cancel. (Review I1)

- [ ] **Step 2: run them; expect RED.**
  - Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py -k cancel_endpoint`
  - Expected: 3 failed, each with `AttributeError: module 'tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs' has no attribute 'cancel_audio_job'`.
  - The first test also pins the `batch_group` copy. With the route but without the copy, the second stage is not found: the route returns 400 and the test fails at `result.cancelled`.
  - Against the first draft's route (one pass, any audio job type), the dry run gave: the race test `HTTPException: 400: Nothing to cancel: the pipeline has finished`, with stage 2 left queued; the TTS test `Failed: DID NOT RAISE HTTPException`, because the TTS job was cancelled. (Review I1, I2)
- [ ] **Step 3: implement.**
  - In `audio_jobs.py`, add `import uuid` after `import threading`.
  - In `submit_audio_job`'s `create_job`, add after `owner_user_id=str(current_user.id),`:

```python
            # Every stage of this pipeline shares it, so DELETE /jobs/{job_id} can find and cancel them all.
            batch_group=f"audio-pipeline:{uuid.uuid4().hex}",
```

  Before `@router.get("/jobs/{job_id}/progress/stream", ...)`:

```python
# The audio pipeline's stages. Other jobs in the audio domain, such as long-form TTS, are not cancelled here.
_PIPELINE_JOB_TYPES = frozenset(
    {"audio_download", "audio_convert", "audio_transcribe", "audio_chunk", "audio_analyze", "audio_store"}
)


class CancelAudioJobResponse(BaseModel):
    job_id: int
    cancelled: int = Field(..., description="How many queued or running stages were cancelled")


@router.delete("/jobs/{job_id}", response_model=CancelAudioJobResponse, summary="Cancel an audio job's pipeline")
async def cancel_audio_job(
    job_id: int,
    current_user: Annotated[User, Depends(get_request_user)],
    jm: Annotated[JobManager, Depends(get_job_manager)],
    principal: Annotated[AuthPrincipal, Depends(get_auth_principal)],
):
    """
    Cancel every queued or running stage of the pipeline this job belongs to (owner or admin).

    Pass the id that /jobs/submit returned, or any stage's id. A stage that is running finishes its current
    step, and nothing after it runs. Other audio jobs, such as long-form TTS, are not handled here (404).
    """
    try:
        job = jm.get_job(int(job_id))
        if not job or str(job.get("domain") or "") != "audio" or job.get("job_type") not in _PIPELINE_JOB_TYPES:
            raise HTTPException(status_code=404, detail="Job not found")
        owner = str(job.get("owner_user_id") or "")
        if not (_principal_has_admin_claims(principal) or owner == str(current_user.id)):
            raise HTTPException(status_code=403, detail="Not authorized for this job")
        group = job.get("batch_group")
        cancelled = 0
        # A stage can hand off between a pass's list and its cancels; the next pass finds the stage it queued.
        # A pipeline has six stages, so six passes are enough.
        for attempt in range(6):
            stages = (
                jm.list_jobs(domain="audio", owner_user_id=owner, batch_group=group, limit=50)
                if group
                else [jm.get_job(int(job_id)) or {}]
            )
            active = [
                stage
                for stage in stages
                if stage.get("status") in {"queued", "processing"} and stage.get("job_type") in _PIPELINE_JOB_TYPES
            ]
            if not active:
                if attempt == 0:
                    raise HTTPException(status_code=400, detail="Nothing to cancel: the pipeline has finished")
                break
            cancelled += sum(1 for stage in active if jm.cancel_job(int(stage["id"]), reason="pipeline_cancelled"))
        return CancelAudioJobResponse(job_id=int(job_id), cancelled=cancelled)
    except HTTPException:
        raise
    except _AUDIO_JOBS_NONCRITICAL_EXCEPTIONS:
        logger.exception(f"Failed to cancel job: job_id={job_id}")
        raise HTTPException(status_code=500, detail="Failed to cancel job") from None
```

  - In `hand_off`'s `create_job`, add `batch_group=job.get("batch_group"),` after `owner_user_id=...`.
  - Its docstring's "it is exempt from the per-user fair-share limit and the submit quotas" becomes "it is exempt from the per-user fair-share limit and the submit quotas, and it keeps the pipeline's batch_group".
- [ ] **Step 4: run them; expect GREEN.**
  - Run the Step 2 command. Expected: `3 passed`.
  - Regenerate the API contract: `PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --fingerprint apps/packages/api-types/openapi.fingerprint.json --paths apps/packages/api-types/paths.d.ts`.
  - Expected: only `openapi.fingerprint.json` changes, with `schema_count` 3301 → 3302 on dev and a new `sha256`. `paths.d.ts` is unchanged, because `/api/v1/audio/jobs/{job_id}` already exists. The `_PIPELINE_JOB_TYPES` constant adds no schema.
  - Then run `PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts`. Expected: exit 0.
- [ ] **Step 5: docs.**
  - In `Audio_Jobs_API.md`, after the `GET /jobs/{job_id}` entry: (Review I1, I2)

```markdown
- DELETE `/jobs/{job_id}` - Cancel a submitted job's pipeline (owner or admin)
  - Path: `job_id` (int): the id `/jobs/submit` returned, or any stage's id.
  - Cancels every stage of that pipeline that is queued or running, including a stage queued while the request runs. A running stage finishes its current step; the stage it queues is cancelled, so nothing after it runs.
  - Response: `{ job_id, cancelled }`, the number of stages cancelled. 400 when no stage is left to cancel; 403 and 404 as for GET. Only audio pipeline stages are handled: another audio job, such as a long-form TTS job, gets 404.
  - A pipeline submitted before this endpoint existed has no pipeline id, so only the job named is cancelled.
```

  - In the Worker section, "Each stage runs as its own job in the `audio` domain; the submit response carries only the first stage's id." gains: " All stages of one pipeline share a `batch_group` (`audio-pipeline:<id>`), which `DELETE /jobs/{job_id}` uses to cancel the rest of the pipeline."
- [ ] **Step 6: commit.**

```bash
git add tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py tldw_Server_API/app/services/audio_jobs_worker.py \
  tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py apps/packages/api-types/openapi.fingerprint.json \
  Docs/API-related/Audio_Jobs_API.md
git commit -m "feat(audio-jobs): DELETE /audio/jobs/{job_id} cancels the rest of a submitted pipeline (TASK-13852)" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 9: PR 2 close-out

- [ ] **Mirrors.** Run `bash Helper_Scripts/refresh_docs_published.sh`. Expected: only `Docs/Published/API-related/Audio_Jobs_API.md` changes. Then run `tests/Docs` as in Task 3.
- [ ] **Sweep.** Expected for each command: all pass; record the counts. The counts move with the analysis PR's test count. (Review M10)
  - Rerun Task 3's sweep command. The revision's dry run: `462 passed, 70 skipped`, in 2.5 minutes (PR 1's 450 plus PR 2's 12 new tests and cases).
  - Rerun Task 3's `RUN_JOBS=1` command (up to 17 minutes). PR 2 adds no Jobs test, so expect the same count as PR 1's run.
  - Run `tests/Docs` and the API-contract tests. The revision's dry run: `323 passed`.

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs \
  tldw_Server_API/tests/Utils/test_pagination_openapi_contract.py tldw_Server_API/tests/CI/test_path_classifier.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py tldw_Server_API/tests/Logging/test_trace_context.py
```

- [ ] **Required ratchets.** Run Task 3's ratchet command. `test_route_auth_ratchet.py` judges the new DELETE route, and `test_test_quality_ratchet.py` judges the new tests. Expected: all pass (the revision's dry run: `53 passed`, in 3.5 minutes). (Review M1)
  - If a route lint fails with `assert -6 == 0` and `libc++abi: terminating due to uncaught exception ... recursive_mutex lock failed` in its output, its child process aborted while exiting. The revision's dry run hit that intermittently in the PR 2 worktree (a different test each time; the same tests then passed three runs in a row). Rerun, and record a baseline on unchanged `origin/dev` if it repeats.
- [ ] **Lint and Bandit** on `audio_jobs_worker.py`, `audio_transcribe_gpu_worker.py` and `audio_jobs.py`.
  - `ruff`: nothing new. I001 and F401 (`contextlib`) in `audio_jobs.py` already exist on dev; F401 moves from line 25 to 26 with `import uuid`.
  - `uvx bandit -ll`: no issues.
- [ ] **Backlog.**

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13851 -s Done --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "RED on dev+PR1: refused hand-off quarantined after running STT twice (payload cap, job-type allowlist, payload not JSON); transient enqueue error re-ran the stage (retry_count 1); GPU worker left a refused hand-off's stage completed, and ran an ownerless job as user 0; 3-hour chunked recording refused at transcribe->chunk (1062513 bytes > 1048576); store result over the cap left the job processing; helper test pinned the duplicated artifact. GREEN: all pass, including the analysis plan's Decision 3 test (hand_off passes result=analysis_error to complete_job). Mutation: catching only ValueError quarantines the payload-not-json case. hand_off logs only the error type. Decision 1 (owner): normalized_stt keeps the artifact's metadata only; about 4 h chunked, 6 h unchunked at the default cap. The GPU worker now shares the transcribe handler and hand_off, and fails an ownerless job as the in-process worker does. Bandit -ll clean." \
  --final-summary "Both workers hand off through one hand_off(): a ValueError or TypeError from create_job (payload over the cap or not JSON, refused queue or job type, secret match) fails the stage at once; any other error is retried twice in place before the stage is retried. The stage completes with the result its caller passes (analysis_error for a failed analysis). The payload carries the transcript once, so a multi-hour recording fits the default cap, and an audio_store result over the cap fails with result_too_large instead of staying processing."
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13852 -s Done --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "RED on dev+PR1: a stage cancelled while running still handed off and audio_store completed; no cancel route (3 AttributeErrors). Against a one-pass route: a stage handed off between the list and the cancel left the next stage queued (400), and a tts_longform job was cancelled. GREEN: the queued next stage is cancelled when the current one turns out cancelled; DELETE /api/v1/audio/jobs/{job_id} handles only the six pipeline job types (404 otherwise) and repeats list-then-cancel, at most 6 passes, through a per-pipeline batch_group (Decision 2, owner). OpenAPI fingerprint regenerated (schema_count +1). Bandit -ll clean." \
  --final-summary "Cancelling a running stage now cancels the stage it queues, so nothing after it runs. DELETE /api/v1/audio/jobs/{job_id} (owner or admin) cancels every queued or running stage of the pipeline, found through the batch_group that submit sets and every hand-off copies, including a stage queued while it cancels. Other audio-domain jobs, such as long-form TTS, get 404."
git add -A backlog/tasks
git add Docs/Published
git commit -m "docs(audio-jobs): close TASK-13851 and TASK-13852; refresh Docs/Published" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

  Correct the notes if the observed RED or GREEN counts differ.
- [ ] **PR.**
  - Open it against `dev`: `gh pr create --base dev`. The body records Decisions 1 and 2 as the owner made them, and says that cancelling long-form TTS jobs is out of scope (Review I2).
  - Before merge:
    - Run `/superpowers:requesting-code-review` in a clean subagent and address its findings.
    - Get the owner's ADR-004 Change summary.
    - Merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset.

## Out of scope

- **`GET /jobs/{job_id}` reports only the named stage's status**, so a user cannot see a pipeline's progress. That would be its own task.
- **Excluding stage rows from the per-minute window** (Decision 3, option b).
- **Storing the transcript outside the payload** (Decision 1, option c), until `audio_store` gets real storage.
- **Stopping the STT work of a stage that is cancelled while it runs.** The AC asks only that later stages stop; the stage's renewal ends at its next renewal, because a cancelled job cannot be renewed.
- **Cancelling long-form TTS jobs.** They share the `audio` domain (`core/TTS/tts_jobs_worker.py:73`, job type `tts_longform`), but TTS has its own dispatch-marker and history semantics, and the TTS worker's handling of a cancelled running job is untested. `DELETE /api/v1/audio/jobs/{job_id}` returns 404 for them. A TTS cancel would need an owner decision and its own task. (Review I2)
- **A rate limit on the new route.** It has no `check_rate_limit` dependency, unlike the media-ingest cancel routes it follows; the other audio-jobs routes have none either. Add one with the rest of the audio-jobs routes if wanted. (Review M15)
