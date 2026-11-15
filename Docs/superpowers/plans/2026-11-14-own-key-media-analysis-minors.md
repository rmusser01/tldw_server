# Own-key media analysis: PR D follow-up minors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended for this size) or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Act on the eight Minor findings of PR D's final review (#3470, TASK-15426): fix the three that have a fix worth making, and record why the other five need none.

**Architecture:** Test-only coverage for the media-analysis denylist; the ingest job's fire-and-forget embeddings task stops inheriting the job's admission and key; two import-time test fixtures become lazy. No behaviour changes for any request or job today.

**Tech Stack:** Python 3, pytest, FastAPI TestClient, asyncio ContextVars.

**Spec:** PR D's final review, the 8 Minors (numbered 2-9 there): `plans/final_review_15426_d.md` (also summarized in #3470's body, "Its Minors are deferred"). PR D's plan: `Docs/superpowers/plans/2026-11-07-own-key-media-analysis-close-out.md`.

**Base:** `origin/dev` after #3470 merges. Branch `fix/own-key-media-analysis-minors-15426`. Until #3470 merges, the plan's line numbers are against #3470's head `4b7c90d9ff`.

## Global Constraints

- PRs target `dev`.
- Tests make no real outbound network request. Every pytest run uses the probe: `PYTHONPATH=<plans dir> NETPROBE_LOG=<file> python -m pytest -p netprobe_15427 ...`; a run ends with 0 netprobe lines for the tests this plan names.
- Backlog edits only with backlog-py (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . ...`), `--append-notes` never `--notes`; stage with `git add -A backlog/tasks`.
- Never `git stash`, `--no-verify`, force-push or `--admin`.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- No trailing whitespace in any committed file, this plan's copy included.
- No doc changes: `Docs/Operations/Usage_Quotas.md`, the BYOK guide and ADR-083 are not edited.

## The eight Minors and what this plan does

| # | Finding | Action |
|---|---|---|
| 2 | `media_analysis_provider`'s nine denylist entries: only 2 pinned end to end; the server's claims default has no test | Task 1: unit test over every entry, plus the server-default and opt-out cases |
| 3 | The `ponytail:` comment and the Usage_Quotas sentence say a missed stage is refused at its own call / keeps the server's key; the final review named `chunk_method=rolling_summarize` as a stage that would run on the user's key | No change: both are true. On every media path the chunker gets no LLM function (only the `/chunking` endpoints pass `llm_call_func`, `chunking.py:393,570`), so `rolling_summarize` raises `ProcessingError: Rolling summarization provider is unavailable` (`rolling_summarize.py:229-234`) and never calls `analyze`; the plan review's dry run found no other missed stage that calls `analyze` on a media path |
| 4 | The worker's `asyncio.create_task(_schedule_embeddings(...))` copies the job's context, admission mark and key | Task 2: create the task under `llm_call_context(prechecked=False, provider_credentials=None)`, with a test |
| 5 | Every analysis request resolves the user's key before the quota check | No change: the owner's D-2 ("use their key": resolve whenever stored); the cost is one secrets read per request on BYOK installs, and a lookup error falls back to the server's key (PR A's helper semantics) |
| 6 | `ensure_llm_state_executor()` in `_handle_job` replaces a bare loop's default executor without shutting the old one down | No change: in production `llm_call_root` already installed it (idempotent); on a bare loop asyncio's default executor is created lazily, so there is usually none to shut down, and reaching it needs the private `loop._default_executor` |
| 7 | `_undo_module_reloads` restores namespaces, not a reload's side effects elsewhere | No change: adequate for that file (its reloads check import bindings); the general class is TASK-15497 |
| 8 | `test_media_llm_quota.py` builds `_wav()` and `_pdf()` at import time | Task 1: the parameters carry the builders; the test calls them |
| 9 | Two fixtures depend on the environment (puremagic not identifying `b"fake"`; ffmpeg accepting `b"ID3"` + zeros) | No change: not defects; each fails loudly (4xx or a failed assertion) if the environment changes, never passes silently |

## Review Focus

1. **Claims by the server's default** (`perform_claims_extraction=None`, `ENABLE_INGESTION_CLAIMS` on): the provider is None, so an over-limit user is refused. With the setting off, or with `perform_claims_extraction=False`, the provider is named.
2. **The embeddings task's context:** no key and no admission mark, but the same user (embeddings writes are the owner's).
3. **The job's own analysis still runs on the key** after Task 2 (the reset is scoped to the `create_task` call only).
4. **The lazy builders** still post real WAV and PDF bytes (the `[audios-*]` and `[pdfs-*]` cases keep passing with the same analysis counts).

---

### Task 0: File the task and copy the plan

**Files:**
- Create: `backlog/tasks/task-15500 - ....md` (backlog-py)
- Create: `Docs/superpowers/plans/2026-11-14-own-key-media-analysis-minors.md` (this plan, verbatim)

- [ ] **Step 1: Take the id.** `mkdir <scratchpad>/id_lock` (fails if held; wait), then `zsh plans/task_next.sh` prints the highest id; `15500` = that + 1. Release the lock (`rmdir`) once the PR's file list shows the task.

- [ ] **Step 2: File it.**

```bash
PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task create \
  "Own-key media analysis: act on PR D's final-review minors" \
  --id TASK-15500 -l byok,usage-quotas,tests \
  -d "Follow-up of TASK-15426 PR D (#3470). Its final review left eight Minor findings (numbered 2-9 there). Three get a fix: a unit test over every entry of media_analysis_provider's other-stage denylist, including claims by the server's ENABLE_INGESTION_CLAIMS default (2); the media ingest job's fire-and-forget embeddings task inheriting the job's admission and the owner's key (4); two test fixtures built at import time (8). Five need no change, each for a stated reason (3, 5, 6, 7, 9)." \
  --ac "media_analysis_provider has a unit test per denylist entry, and one for the server's claims default on, off, and opted out" \
  --ac "The media ingest job's embeddings task runs with no provider key and no admission mark, with a test that fails without the fix" \
  --ac "test_media_llm_quota.py builds no fixture bytes at import time"
```

- [ ] **Step 3: Copy the plan and commit.** Copy this file to `Docs/superpowers/plans/2026-11-14-own-key-media-analysis-minors.md`, with its task-id placeholder replaced by the id.

```bash
git add -A backlog/tasks Docs/superpowers/plans/2026-11-14-own-key-media-analysis-minors.md
git commit -m "chore(backlog): file TASK-15500 for PR D's final-review minors, with its plan"
```

---

### Task 1: Pin the denylist, build fixtures lazily (Minors 2, 8)

**Files:**
- Modify: `tldw_Server_API/tests/Media/test_media_llm_quota.py` (imports; `_OTHER_HANDLERS` at `:91-105`; `test_media_handlers_name_the_analysis_provider_to_the_llm_quota` at `:176-212`; new test after `test_analysis_is_requested_only_with_a_provider` at `:112-115`)

- [ ] **Step 1: Write the denylist test.** Add `media_analysis_provider` to the existing import from `media_processing_deps`, and `from tldw_Server_API.app.core.config import settings`. After `test_analysis_is_requested_only_with_a_provider`:

```python
_NO_OTHER_STAGE = {"api_name": "OpenAI", "perform_claims_extraction": False}


@pytest.mark.unit
@pytest.mark.parametrize(
    "stage",
    [
        {"perform_claims_extraction": True},
        {"enable_contextual_chunking": True},
        {"contextual_llm_model": "gpt-4o-mini"},
        {"context_window_size": 500},
        {"auto_chunking_use_llm": True},
        {"auto_apply_template": True},
        {"chunking_template_name": "academic_paper"},
        {"perform_confabulation_check_of_analysis": True},
        {"proposition_engine": "llm"},
        {"proposition_engine": " Auto "},
    ],
    ids=lambda stage: next(iter(stage)) + ("=" + str(stage["proposition_engine"]).strip() if "proposition_engine" in stage else ""),
)
def test_another_llm_stage_names_no_provider(stage: dict) -> None:
    """Every other LLM stage the boundary cannot place on the user's key makes the request name no provider, so an
    over-limit user is refused (TASK-15426)."""
    assert media_analysis_provider(SimpleNamespace(**_NO_OTHER_STAGE)) == "openai"
    assert media_analysis_provider(SimpleNamespace(**{**_NO_OTHER_STAGE, **stage})) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("server_default", "perform_claims_extraction", "expected"),
    [(True, None, None), (False, None, "openai"), (True, False, "openai")],
    ids=["server-default-on", "server-default-off", "opted-out"],
)
def test_claims_by_the_servers_default_name_no_provider(
    monkeypatch: pytest.MonkeyPatch, server_default: bool, perform_claims_extraction: bool | None, expected: str | None
) -> None:
    """Claims the server turns on by default (ENABLE_INGESTION_CLAIMS) are another LLM stage unless the request opts
    out with perform_claims_extraction=false (TASK-15426)."""
    monkeypatch.setitem(settings, "ENABLE_INGESTION_CLAIMS", server_default)
    form = SimpleNamespace(api_name="openai", perform_claims_extraction=perform_claims_extraction)
    assert media_analysis_provider(form) == expected
```

(If `monkeypatch.setitem` cannot undo on `LazySettings`, save `settings.get("ENABLE_INGESTION_CLAIMS")` and restore it in a `try/finally`, as `tests/Claims/test_claim_embeddings_chroma.py:18-67` does. Record which one was used.)

- [ ] **Step 2: Run them.** They pin shipped behaviour, so they pass on first run.

Run: `python -m pytest -p netprobe_15427 tldw_Server_API/tests/Media/test_media_llm_quota.py -k "no_provider" -v`
Expected: 13 passed, 0 netprobe lines.

- [ ] **Step 3: Sabotage check (the RED for a coverage test).** Temporarily make `claims_extraction_enabled` return `bool(value)` only (drop the settings fallback, `claims_utils.py:54-57`), re-run Step 2: expected exactly `[server-default-on]` fails. Then temporarily remove `"auto_apply_template",` from the denylist tuple (`media_processing_deps.py:695`): expected exactly `[auto_apply_template]` fails. Revert both (`git diff --stat` shows only the test file). Record the two results.

- [ ] **Step 4: Make the fixtures lazy (Minor 8).** In `_OTHER_HANDLERS`, pass the builders, not their bytes:

```python
    pytest.param("/api/v1/media/process-audios", _ANALYSIS, [("files", ("a.wav", _wav, "audio/wav"))], 1, id="audios"),
```

```python
        [("files", ("d.pdf", _pdf, "application/pdf"))],
```

and in `test_media_handlers_name_the_analysis_provider_to_the_llm_quota`, replace `response = client.post(path, data=data, files=files)` with:

```python
        files = [(field, (name, body() if callable(body) else body, mime)) for field, (name, body, mime) in files]
        response = client.post(path, data=data, files=files)
```

(Inside the `with block_live_network()` block, so building the bytes is also under the probe.)

- [ ] **Step 5: Run the file.**

Run: `python -m pytest -p netprobe_15427 tldw_Server_API/tests/Media/test_media_llm_quota.py -v`
Expected: every case passes (the file's previous count + 13), 0 netprobe lines; `grep -nE "_(wav|pdf)\(\)" tldw_Server_API/tests/Media/test_media_llm_quota.py` prints only the two `def` lines.

- [ ] **Step 6: Commit.**

```bash
git add tldw_Server_API/tests/Media/test_media_llm_quota.py
git commit -m "test(media): pin every other-LLM-stage entry of the media analysis denylist (TASK-15500)"
```

---

### Task 2: The embeddings task drops the job's key and admission (Minor 4)

**Files:**
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py:712-720`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py` (new test after `test_the_admission_ends_with_the_job`, `:252-270`)

- [ ] **Step 1: Write the failing test.**

```python
@pytest.mark.asyncio
async def test_the_embeddings_task_carries_no_key_and_no_admission(
    job_dbs: Path, under_the_limit: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fire-and-forget embeddings task outlives the job, so it starts with neither the job's admission nor the
    owner's key: an LLM call it made would be checked at the call, on the server's key (TASK-15500)."""
    seen: list[Any] = []
    started = asyncio.Event()

    async def _embeddings(**kwargs: Any) -> None:
        seen.append((current_llm_call_context(), kwargs["user_id"]))
        started.set()

    monkeypatch.setattr(worker, "_schedule_embeddings", _embeddings)
    monkeypatch.setattr(summary, "_summarize_via_adapter", lambda **_kwargs: "Cedar opens.")
    users_own_key(monkeypatch, "openai")
    manager = JobManager()
    row = _create_job(manager, job_dbs, _write_document(job_dbs, "Cedar"), "document", generate_embeddings=True)

    result = await worker._handle_job(manager.get_job(int(row["id"])), manager, worker._ProgressState())
    await asyncio.wait_for(started.wait(), timeout=5)

    assert result["status"] == "Success", result
    [(context, user_id)] = seen
    assert context.provider_credentials is None
    assert context.prechecked is False
    assert str(user_id) == "1"
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `python -m pytest -p netprobe_15427 tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py -k embeddings_task -v`
Expected: FAIL on `assert context.provider_credentials is None` (the handle for `sk-users-own-key`).

- [ ] **Step 3: Implement.** Replace the `create_task` block:

```python
        if media_id and getattr(form_data, "generate_embeddings", False):
            # The task outlives the job: it keeps the owner, not the job's admission or key.
            with llm_call_context(prechecked=False, provider_credentials=None):
                asyncio.create_task(
                    _schedule_embeddings(
                        media_id=int(media_id),
                        user_id=user_id,
                        db=db,
                        form_data=form_data,
                    )
                )
```

(`llm_call_context` is already imported at `:47`.)

- [ ] **Step 4: Run the file.**

Run: `python -m pytest -p netprobe_15427 tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py -v`
Expected: all pass (previous count + 1), 0 netprobe lines; `test_the_admission_ends_with_the_job` still passes (the job's own analysis ran on `sk-users-own-key`).

- [ ] **Step 5: Commit.**

```bash
git add tldw_Server_API/app/services/media_ingest_jobs_worker.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py
git commit -m "fix(media-jobs): start the embeddings task without the job's admission or the owner's key (TASK-15500)"
```

---

### Task 3: Check and close the task

**Files:**
- Modify: `backlog/tasks/task-15500 - ....md` (backlog-py)

- [ ] **Step 1: Run the docs and lint checks.**

Run: `python -m pytest -p netprobe_15427 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint -q`
Expected: no new failure against the base (the base's environment-only failures: `tests/lint/test_runtime_package_discovery.py` errors without `setuptools`; `test_jobs_postgres_exports_recorded_coverage` fails without `coverage`).

- [ ] **Step 2: Wider set.** Run every test file that names a touched symbol, as PR D did:

Run: `git grep -l -E "media_analysis_provider|_schedule_embeddings|media_ingest_jobs_worker|test_media_llm_quota" -- tldw_Server_API/tests` and run those files with `-n 4 --dist loadfile` under the probe.
Expected: 0 failed; netprobe lines only from TASK-15486's known tests.

- [ ] **Step 3: Close the task and commit.**

```bash
PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit TASK-15500 --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "Minors 3, 5, 6, 7 and 9 need no change: 3's comment and Usage_Quotas sentence are true, since on media paths rolling_summarize gets no LLM function and raises instead of calling analyze; 5 is the owner's D-2; 6 acts only on bare loops whose default executor is created lazily; 7 is adequate for its file and TASK-15497 covers the class; 9 fails loudly if the environment changes." \
  --final-summary "<what changed, the counts>" -s Done
git add -A backlog/tasks
git commit -m "chore(backlog): close TASK-15500"
```

(Check `backlog_py task edit --help` for the exact flag names before running.)
