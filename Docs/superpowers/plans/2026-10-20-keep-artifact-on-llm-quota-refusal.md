# Add-on LLM Work Keeps the User's Artifact on a Quota Refusal (TASK-13864)

> **Review folded in (2026-10-20).** Independent review and dry run (`review_plan_fu_keep_artifact_on_refusal.md`): 0 Blocker, 4 Important, 7 Minor, all folded in per the coordinator's rulings.
> - **I1:** TASK-13638 row 24 is kept. A job refused at its start passes the refusal into `process_batch_media`, so a reused transcript still records `video_lite.summary_error` (new Task 2). Review Focus 2 is corrected, and the row-24 sentence on the operator page is reworded.
> - **I2:** Task 6's script also edits `Usage_Quotas.md:94`, design doc lines 207, 227, 228 (second sentence) and 253, and the `llm_call_root` docstring.
> - **I3:** the scraping queue leak is fixed in this PR (new Task 4). Each job runs in its submitter's context, and a refusal fails only its job. (The clean worker start first added here was removed by re-review M4.)
> - **I4:** the neighbourhood commands are split as measured. The dev order dependence (a watchlists test, then the OpenAPI test) is recorded with its dev reproduction. The duplicate connectors operationId gets a follow-up.
> - **M1:** the conflicts table is rebuilt from the actual `plan_fu_*.md` files. The design-doc revision is appended at end of file.
> - **M2:** the AC numbering note is fixed.
> - **M3:** the plan is committed to `Docs/superpowers/plans/2026-10-20-keep-artifact-on-llm-quota-refusal.md`.
> - **M5:** one clause each for `Usage_Quotas.md:106`, `:109`, `:110` and the design's Risks bullet.
> - **M6:** a `summarize: false` control test, and the enrichment-fallback note.
> - **M7:** Decision 1's middle option.
> - **M8:** the count is 527 lines, and everything is re-measured on `2e6ceab9fb`.
> - **Found while re-running:** an admission must not outlive the job when `_handle_job` is called outside `WorkerSDK`'s `llm_call_root`. Task 1 now scopes it, with a test.
>
> **Re-review folded in (2026-10-20).** Scoped re-review (`rereview_plan_fu_keep_artifact.md`): ready; 0 Blocker, 0 Important, 4 Minor, folded in per the coordinator's rulings.
> - **M1:** the sweep line about the core Scheduler is corrected. It starts lazily, including inside `POST /watchlists/outputs`, and Review Focus 6 records that the admission reaches its tasks.
> - **M2:** Task 6's script also edits design-doc lines 206-207 (where the §4a mark reaches) and the §4b intro (dev line 239).
> - **M3:** the route is spelled `/media/ingest/jobs` in `Usage_Quotas.md:94` and design line 227, the only two places with the old spelling. Line 228's edit drops the route.
> - **M4:** the clean worker start is deleted from Task 4. The workers do no LLM or DB work outside a job, and the per-job context covers correctness.
>   - Re-measured on `e521605198`. RED and GREEN counts are unchanged.
>   - The code diff is now 78 added lines and 9 removed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** No path in TASK-13864 throws away work it already did because an add-on LLM step was refused.
- A media ingest job stores its item without analysis and says why. A reused transcript in such a job keeps its item-level record.
- The three HTTP endpoints refuse with the 402 before they fetch or render anything.
- Scraping queue jobs run as the request that submitted them.
- A Prompt Studio evaluation refused part-way keeps the links to the test runs it stored.

**Architecture:**
- **The rule, in one line:** a synchronous request that asks for an add-on LLM step is refused with a 402 before any work, as `/media/add` is. An asynchronous job keeps its artifact and records the refusal, as audio jobs do (TASK-13638 Decision 3).
- **Media ingest jobs (Decision 1).** The worker checks the owner's LLM quota once, when the job starts, if the job asks for analysis. It reuses functions that already exist:
  - `media_analysis_requested`, the `/media/add` predicate;
  - `user_to_check` and `check_user_async`, the check at the call (§4b);
  - `mark_llm_quota_prechecked`, the §4a admission mark.
  - **Refused:** the job runs with `perform_analysis=False` and stores the item. Its `result` carries `status: Warning`, a warning, and `analysis_error: {code: "llm_quota_exceeded", message}` (the audio-jobs shape).
  - **A refused job that reuses a transcript:** `process_batch_media` gets the refusal (`analysis_refusal=`), and `_populate_reused_transcript_summary` records `video_lite.summary_status: failed, summary_error: llm_quota_exceeded` without an LLM call. That keeps TASK-13638 row 24.
  - **Admitted:** the rest of the job runs `prechecked`. `_handle_job` wraps the job in `llm_call_context()`, so the mark ends with the job whoever calls the handler; `WorkerSDK`'s `llm_call_root` also resets it.
- **The three HTTP endpoints (Decision 2).** `/media/ingest-web-content` with analysis, `/media/process-web-scraping` with `summarize_checkbox`, and `POST /watchlists/outputs` with `summarize` join the §4a boundary table.
- **The scraping queue (I3, a defect).** `WebScrapingService` starts `ScrapingJobQueue`'s workers lazily, inside the first scraping request, so they inherited that request's user and, after Task 3, its admission. Every later job of every user ran in it (probe: `{'user_id': 7, 'prechecked': True}` for user 9's job).
  - The workers still start in the context of the request that starts them. Outside a job they only take jobs off the queue, log, and resolve futures, with no LLM or DB call, so that context decides nothing (re-review M4).
  - `add_job` stores `contextvars.copy_context()` on the job, and the worker runs `_execute_job` in a task created with that context.
  - A quota refusal raised by a job fails that job's future; on dev it killed the worker task.
- **Prompt Studio** (the task's Implementation Notes): the failed-record update gets `"test_run_ids": test_run_ids`.

**Tech Stack:** Python 3.12 (`asyncio.create_task(..., context=)`), FastAPI, contextvars, pytest (pytest-asyncio auto mode, pytest-xdist), loguru.

**Spec:**
- `backlog/tasks/task-13864 - Add-on-LLM-work-that-still-discards-the-user-s-artifact-on-a-quota-refusal.md` on `dev`.
  - AC1: each listed path keeps the transcript, page or output it produced and records `llm_quota_exceeded` where the user can see it, with a test per path.
  - AC2: `Usage_Quotas.md` says what each path keeps.
  - Implementation Notes: the Prompt Studio evaluation's failed update keeps `test_run_ids`, with a test that refuses the second of two cases.
  - Task 1 Step 0 replaces AC1 (Decision 2) and adds ACs for Prompt Studio and the scraping queue.
- Governing design: `Docs/Design/2026-10-09-llm-usage-accounting-design.md` §1, §4a, §4b. TASK-13638's plan: `Docs/superpowers/plans/2026-10-18-llm-quota-refusal-callers-rag-metric.md`, Decision 3 and row 24. Operator page: `Docs/Operations/Usage_Quotas.md`.

## Delivery: one PR

| PR | Task | Branch | Plan tasks |
|---|---|---|---|
| A | TASK-13864 | `fix/keep-artifact-on-llm-quota-refusal` | 1–6 |

It is one PR. All six tasks close one backlog task and share the operator page. The code diff is 78 added lines (87 with blank lines) and 9 removed, in 8 app files, measured on the prototype. Of those, Task 2 (row 24 kept) is 20 added and 4 removed in `persistence.py` plus 1 line in the worker. There is one commit per plan task, plus the backlog start (Task 1 Step 0) and close-out (Task 6 Step 6) commits.

## What was measured

Measured on `origin/dev` `2e6ceab9fb` (2026-10-20), in scratch worktrees (`wt_addon2` for the plan, `wt_base2` for dev baselines).
- The touched files are unchanged since the first draft's base `a6719bf24f`. Only `endpoints/workflows.py` moved, and this plan does not touch it.
- The review dry-ran the first draft on `df7c290fbf` and `454974e812`.
- Each probe makes the LLM call raise `LLMQuotaExceededError(category="llm_tokens_month", ...)`, as the check at the call does.
- Artefacts in the scratchpad:
  - `addon2_app.patch`, `addon2_tests.patch`, `addon2_docs.patch`, `addon2_new/`;
  - `red2.log`, `suite2_*.log`, `base2_media.log`, `order_dev.log`, `tq2.log`, `bandit2.log`;
  - `addon2_ruff.py`, `addon_docs2.py`, `addon_overlap.py`.

| # | Path | Today, on a refusal (measured) | What it keeps after this plan | Where the refusal is recorded | Tests that pin today's behaviour |
|---|---|---|---|---|---|
| 1 | Media ingest job, document-like (`/media/ingest/jobs`, `process_document_like_item`) | The processors' analysis `except` tuples (`_PLAINTEXT_…`, `_PDF_…`, `_BOOK_PROCESSING_…`, `_EMAIL_…`) hold no `HTTPException`, so the refusal escapes the processor with its extracted text. `persistence.py:5518` catches it. Job `completed`, `result.status` `Error`, `error` `Processing error: LLMQuotaExceededError: 402: {...}`, `media_id` `None`. | The stored document, without analysis | The job's `result`: `status: Warning`, warning `Analysis skipped: LLM quota exceeded (llm_tokens_month).`, `analysis_error: {"code": "llm_quota_exceeded", "message": ...}` | None pins the refusal. Neighbours: `MediaIngestion_NEW/unit/test_plaintext_analysis_outcomes.py::test_real_analysis_job_persists_source_and_truthful_outcome`, `MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py` (job result shape) |
| 2 | Media ingest job, audio/video (`process_batch_media`) | `_AUDIO_FILES_…` and `_VIDEO_…` hold no `HTTPException`; the refusal escapes `process_audio_files` with the transcript. `persistence.py:4618` marks every item of the call `Error`, `Failed to call processor: LLMQuotaExceededError`, `media_id` `None`. | The stored transcript | The same `result` fields | None |
| 3 | Media ingest batch (one job per item, one `batch_id`) | Every job makes its own refused call (2 jobs: 2 refused calls) and is marked `Error` as in rows 1–2. | Every item | Each job's `result` | None |
| 4 | A media ingest job that reuses an earlier transcript (TASK-13638 row 24) | On dev, the reuse summary call is refused and `persistence.py:1790` records `video_lite: {summary_status: failed, summary_error: llm_quota_exceeded}`. This is the only live path to that clause: `/media/add` is prechecked. With Task 1 alone, the job turns analysis off, so nothing is recorded on the item (measured: `KeyError: 'video_lite'`). | The reused transcript | The item's `video_lite` metadata, as today, and the job's `result` | `MediaIngestion_NEW/unit/test_reused_transcript_llm_quota_refusal.py` (the clause itself); `integration/test_media_ingest_jobs_video_dedupe.py` (reuse through the worker) |
| 5 | `/media/ingest-web-content` with analysis | 402. One page was fetched, then its summary call was refused; the response drops it, and the second URL is never fetched. | Nothing is fetched (Decision 2) | The 402 `limit_exceeded` body and `Retry-After` | `Media/test_web_summary_service_prompt.py` (admitted path); `Web_Scraping/test_phase4_extraction_failures.py::test_provider_quota_refusal_is_not_retried`. None pins the discard. |
| 6 | `/media/process-web-scraping` with summarization | 402. `Individual URLs`, `mode: persist`: both pages fetched, 1 refused call, nothing stored. `Sitemap`: the sitemap fetched, 1 refused call, nothing stored. | Nothing is fetched (Decision 2) | The 402 | Same file. None pins the discard. |
| 7 | `POST /watchlists/outputs` with `summarize` | 402 and no output written (`watchlists.py:7215` re-raises; TASK-13636). The run's items were loaded and the summary call made first. | Nothing is rendered (Decision 2) | The 402. `Watchlists_API.md:506` already says "a user over an LLM quota gets a 402 and no output is created". | `Watchlists/test_watchlists_api.py::test_output_summaries_return_a_quota_refusal_as_402` (stays green) |
| 8 | Scraping queue jobs (`ScrapingJobQueue`, fed by `scrape_multiple`: "Individual URLs" and the sitemap, URL-level and recursive crawls) | Workers inherit the context of the request that started them: user 9's job ran as `{'user_id': 7, 'prechecked': True}`. A job that raises the refusal kills its worker task (`_WEBSCRAPE_NONCRITICAL_EXCEPTIONS` has no `HTTPException`); the next job's future never resolves (`TimeoutError` after 5 s). | Each job's own result; the worker | The job's future (the refusal) | None |
| 9 | Prompt Studio evaluation refused part-way (`evaluation_manager.py:258-268`) | The record ends `failed` with `error_message`, but the update has no `test_run_ids`: the runs stored before the refusal stay in the DB, unlinked (`KeyError: 'test_run_ids'`). | The links to the stored test runs | The evaluation record's `test_run_ids` | `prompt_studio/unit/test_evaluation_llm_quota_refusal.py::test_evaluation_stops_at_the_refusal_and_marks_the_record_failed` (stays green) |

Not on the list, checked and unaffected:
- `/media/add` and the six `/media/process-*` endpoints run the same processors, but their boundary check marks the request `prechecked`, so the processors are never refused there.
- The reading service calls `process_document_like_item` with `perform_analysis=False` (`Collections/reading_service.py:497-503`).
- The auto-chunking LLM refinement catches `Exception` and degrades (`Chunking/auto_boundary_assistant.py:189,212`).
- The enhanced scraper allows LLM extraction only with `summarize` (`enhanced_web_scraping_service.py:496,550,618,683`), so Decision 2's check covers it for `/media/process-web-scraping`.
- **Other lazily started workers (I3 sweep).** `git grep` for long-lived `asyncio.create_task(self._…)` loops in `app/`:
  - `Chat/request_queue.py:507`: its items carry `capture_llm_call_state()` and run under `run_with_llm_call_state` (lines 720, 919, 1105), so the worker's own context does not decide the user.
  - `Buddy/turns.py:198`: one drain task per `(user_id, conversation_id)`. Every turn it drains belongs to that user, and Buddy has no boundary check, so no admission leaks.
  - `RAG/rag_service/batch_processing.py:538`: `BatchScheduler.start()` has no caller in the app (`rag_health.py` imports only `BatchProcessor`).
  - The core Scheduler can start lazily (`get_global_scheduler`, `Scheduler/scheduler.py:903-917`). It starts at startup only with `WORKFLOWS_SCHEDULER_ENABLED=true` (`startup_recurring_schedulers.py:79`); otherwise its first caller starts it.
    - One caller is `schedule_output_enrichment` (`output_enrichment_handler.py:133`), inside `POST /watchlists/outputs` (`watchlists.py:7761`). That runs after Task 3's mark when a request has `summarize` and enrichment.
    - Its LLM-calling handlers run under `llm_call_root`, which clears the admission and sets the owner: `watchlists_enrich_output` (`Scheduler/handlers/watchlists.py:97`) and `workflow_run` (`Workflows/engine.py:549,1140`).
    - See Review Focus 6.
  - The rest (audit, MCP, metrics, TTS, ACP, health loops) start at startup or per session, and make no LLM call on behalf of queued user work.
  - `Watchlists/fetchers.py:432` builds an `EnhancedWebScraper()` without `start()`, so it starts no workers.
  - `ScrapingJobQueue.add_job` is the one way a scraping job enters the queue (a retry re-queues the same job), so Task 4 fixes it there.

## Batch, settled

- **Media ingest jobs:** a batch is one job per item. Each job checks the quota when it starts. Once the user is over a limit, every later job is refused before any LLM work, at the cost of one counter read and no provider call. It then stores its item without analysis. No batch-wide flag is needed: the decision is per user and month, and each job's own check already gives the answer.
- **Within one job:** an admitted job runs `prechecked` and is never refused part-way. A refused job makes no LLM call at all.
- **The HTTP endpoints:** under Decision 2's recommendation, an admitted request runs to completion. Its several URLs, sitemap pages, crawl pages or run items cannot be refused part-way, including the scraping queue jobs it submits (Task 4 carries the admission to them).
- **Overshoot:** an admitted job or request can take the user past the limit by its own size. With `c` media ingest jobs running at once, up to `c` jobs can be admitted just under the limit. Task 6 adds that to `Usage_Quotas.md:106`.

## Prototype check (the whole plan applied in `wt_addon2` on `2e6ceab9fb`)

- **New tests, RED on unchanged app code** (one run, 14 new test ids plus the 2 existing Prompt Studio tests): `10 failed, 6 passed` (`red2.log`). The 6 that pass on dev are the admission-scope test (dev never marks), the three controls, and the two existing Prompt Studio tests. The failures:
  - Task 1: the document batch `('Error', "Processing error: LLMQuotaExceededError: 402: ...")`; audio `('Error', 'Failed to call processor: LLMQuotaExceededError')`; the admitted job `assert [1] == [None]`.
  - Task 2: `KeyError: 'analysis_error'` on dev. With Task 1 applied and Task 2 not: `KeyError: 'video_lite'`.
  - Task 3: the two web endpoints and the watchlist output answer 200 where 402 is expected.
  - Task 4: `assert [{'user_id': 7, 'prechecked': True}] == [{'user_id': 9, 'prechecked': False}]`, and `TimeoutError` for the job after a refused one.
  - Task 5: `KeyError: 'test_run_ids'`.
- **Within Task 1:** with the job-start check but without the `llm_call_context()` scope in `_handle_job`, the scope test fails with `assert True is False`. The Task 2 test then fails with `KeyError: 'analysis_error'`, because user 1's admission leaked into user 2's job in the same task.
- **GREEN:** the same ids plus `test_output_summaries_return_a_quota_refusal_as_402`: `17 passed`.
- **Neighbourhood suites on the prototype** (`-n 4`):
  - **A** (`MediaIngestion_NEW`, the worker scope test, two `Services` files, `Jobs/test_worker_sdk_llm_user.py`, `Chat/unit/test_llm_call_state_propagation.py`, `AudioJobs/test_audio_worker_smoke.py`, `tests/Usage`): `1 failed, 1320 passed, 14 skipped, 1 xfailed` (`suite2_media.log`).
    - The failure is `MediaIngestion_NEW/integration/test_media_add_endpoint.py::TestErrorHandling::test_transcription_failure_handling` (`Exception: Transcription failed` escapes).
    - The same command on unchanged dev gives the same failure: `1 failed, 1315 passed, 14 skipped, 1 xfailed` (`base2_media.log`).
    - The test passes alone, and its file passes alone (`22 passed`), so it is an existing xdist order dependence.
  - **B** (`tests/Media`, `tests/Web_Scraping`, `tests/WebScraping`, `tests/Watchlists`, the two AuthNZ files, `tests/lint`): `4064 passed, 37 skipped, 1 xpassed` (3 min 44 s; `suite2_web.log`), including `WebScraping/test_refactor_import_inventory.py` against the regenerated inventory.
  - **`Services/test_openapi_contracts.py` alone:** `74 passed` (`openapi2.log`). Run it alone. On unchanged dev, `Watchlists/test_watchlists_api.py::test_output_summaries_return_a_quota_refusal_as_402` followed by `test_openapi_operation_ids_are_present_and_unique` in one process gives `1 failed, 1 passed`: "Duplicate OpenAPI operationIds: provider_webhook_callback_api_v1_connectors_providers__provider__webhook_post: POST … and GET /api/v1/connectors/providers/{provider}/webhook" (`order_dev.log`). The review saw this ordering arise in a combined `-n 4` run (`1 failed, 4401 passed`). Task 6 files it.
  - **`tests/prompt_studio`:** `1181 passed, 17 skipped` (3 min 45 s; `suite2_ps.log`).
  - **`tests/Docs`** with every doc edit, the refreshed mirror and the plan copy: `212 passed` (`docs2.log`).
- **Gates:**
  - `test_quality_triage.py --enforce`: exit 0. A first draft of the queue test asserted only its collector, and was flagged `tautology_suspect`; the test now also asserts the job's result.
  - `tests/CI/test_test_quality_ratchet.py`: `1 passed`.
  - Ruff, multiset comparison over the 8 app files and 6 test files (`addon2_ruff.py`): 23 findings on dev, the same 23 after, nothing new.
  - `uvx bandit -q` on the 8 app files: no findings.
  - `check_shard_coverage.py`: "OK — no newly-unshared test files".
  - `scope_predicate_ratchet.py`: "28 conditional scope predicates; baseline exact".
  - The OpenAPI drift gate: "fingerprint matches" and "path union matches" (exit 0).
  - Regenerating the web scraping import inventory changes `Docs/Design/web_scraping_refactor_import_inventory.json` and `Docs/Design/WebScraping_Refactor_Import_Inventory.md`, because Task 4 shifts `enhanced_web_scraping.py` lines and adds a test that imports `ScrapingJobQueue`. Task 4 commits both, and `WebScraping/test_refactor_import_inventory.py` checks them.
  - `refresh_docs_published.sh` changes one mirrored file: `Docs/Published/API-related/Media_Ingest_Jobs_API.md`.
- **Re-review fold-in, re-run on `e521605198`** (`wt_addon3`; the touched files are unchanged since `2e6ceab9fb`):
  - GREEN: `17 passed` (`green3.log`). The RED split is unchanged, `10 failed, 6 passed`: dev's code did not change, and Task 4's tests do not depend on `start()`.
  - Neighbourhood B: `4064 passed, 37 skipped, 1 xpassed`, as before (`suite3_web.log`).
  - `tests/Docs` with the revised script, the refreshed mirror and the plan copy: `212 passed` (`docs3.log`).
  - Gates:
    - triage `--enforce`: exit 0;
    - ruff: 23 findings, the same 23, nothing new (`addon3_ruff.py`);
    - `scope_predicate_ratchet.py`: baseline exact;
    - `uvx bandit -q` on `enhanced_web_scraping.py` and `llm_roots.py`: no output.
  - After the revised script, `grep -c ingest-jobs` on both docs gives 0.

## Global Constraints

- **Python:** `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. **Tests:** `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q <paths>`. Do not pass `-p no:logging`: it removes the `caplog` fixture, and 4 tests in `MediaIngestion_NEW/integration/test_email_sensitive_logging_13376_2.py` then error.
- **Branch** `fix/keep-artifact-on-llm-quota-refusal` from `origin/dev`; the PR targets `dev`. Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never `git stash`.
- **Test first:** each task's RED run happens on source without that task's change, and fails for the reason given. Record a dev baseline for any other failure before blaming the change. Two are known on dev:
  - `test_transcription_failure_handling` in neighbourhood A;
  - the OpenAPI operationId test after a watchlists test, which is why `test_openapi_contracts.py` runs alone.
- **The test-quality ratchet is a required gate.**
  - Run `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` (exit 0) and `tldw_Server_API/tests/CI/test_test_quality_ratchet.py` (`1 passed`).
  - An integration-marked test may make at most 3 mock/patch calls in its own body; put the rest in fixtures.
  - A test that collects calls into a list must also assert a real output.
- **No new exception type, setting, `limits.*` key or response model.**
  - No change to `quota_checks`, `llm_quota_gate` or `llm_call_context`, and no behaviour change to `llm_roots` (its docstring gains one sentence).
  - The processors' analysis steps and `Summarization_General_Lib.py` are not changed.
  - The existing `except LLMQuotaExceededError: raise` at `watchlists.py:7215` stays.
- **The boundary predicate per endpoint:**
  - `/media/ingest-web-content`: `media_analysis_requested(request)`, the `/media/add` predicate. `perform_analysis` defaults to true, and without a provider no summary runs (`web_scraping_service._has_analysis_provider`).
  - `/media/process-web-scraping`: `payload.summarize_checkbox` alone. Its sitemap, URL-level and recursive summaries call `analyze(api_name=api_name or "openai")` (`enhanced_web_scraping_service.py:740`).
  - `POST /watchlists/outputs`: `payload.summarize`, after the provider is resolved, so a missing provider still answers 400.
- **Imports:** ruff's isort is case-insensitive, and puts `import x` before `from x import` within a block. Each task names the line the new import goes after; the lint comparison must show nothing new.
- **Lint comparison:** `$PY -m ruff check --output-format concise <touched .py files>` reports nothing the `origin/dev` copies do not. Compare multisets of findings with line and column numbers stripped (the scratchpad's `addon2_ruff.py`). Do not run `ruff --fix`.
- **Docs:**
  - `Usage_Quotas.md`, the design doc and the web scraping inventories have no `Docs/Published` mirror.
  - `Docs/API-related/Media_Ingest_Jobs_API.md` is mirrored: Task 6 runs `bash Helper_Scripts/refresh_docs_published.sh` and commits the `Docs/Published` diff.
  - Doc edits are anchored on quoted text, not line numbers, and the design-doc revision is appended at end of file.
- **Backlog:**
  - `tools/backlog-py` only (`PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task ...`), with `--append-notes`, explicit ids for new tasks, and `git add -A backlog/tasks`.
  - Mark TASK-13864 `In Progress` in the first commit and `Done` in the close-out commit.
  - Execution order with `plan_fu_anthropic_cache.md`, which also files follow-ups by the max-id rule: run one plan to its open PR before the other computes ids.
- **ADR-004:** the PR body needs a "## Change summary" from the owner. Ask the owner, before merge, whether the spec 2 waiver covers TASK-13864. The waiver covers TASK-13638 (2026-10-16) but does not name its follow-ups, this task among them.
- **Review and merge:** run `/superpowers:requesting-code-review` in a clean subagent before merge. Check `gh variable get MERGE_QUEUE` and follow CLAUDE.md "Merging into dev" for that mode.

## Conflicts with other plans

Measured against every `plan_fu_*.md` in the scratchpad on 2026-10-20 (`addon_overlap.py`), and against each plan's own conflicts table.

| Place | This plan | Other plan | How to merge |
|---|---|---|---|
| `Usage_Quotas.md:95` | Removes the three endpoints and the "discard that request's results" sentence | `plan_fu_fva_adjudicator.md` (TASK-13862) deletes "claims verification (`POST /claims/verify/fva`), " on the same line, and edits `:103` | Same line: git conflicts for whoever lands second. Re-apply your own edit by its quoted text; the two edits remove different phrases, so each script still matches once after a rebase. |
| `Usage_Quotas.md:94`, `:97`, `:98` | Adds a clause to `:94` and `:97`; replaces two sentences of `:98` and rewords the row-24 sentence | `plan_fu_workflow_cancel_siblings.md` (TASK-13863) replaces the last sentence of `:96` | Adjacent lines: git conflicts for whoever lands second. Keep the landed text and re-apply your own edit by quote. No sentence is shared. |
| `Usage_Quotas.md:106`, `:109`, `:110` | Adds one clause to each | `plan_fu_anthropic_cache.md` inserts a bullet before "Two settings defeat these limits" (between `:107` and `:108`); `plan_fu_workflow_cancel_siblings.md` only cites `:106` in prose | One unchanged line separates each hunk, so they apply in either order. If git still conflicts, re-apply by quote. |
| `Usage_Quotas.md` concurrency section | Not touched | `plan_fu_rg_redis_error.md` edits the Redis line (`:124`) | No conflict. |
| Design doc, end of file | Appends "Revision of 2026-10-20, TASK-13864" at end of file | `plan_fu_fva_adjudicator.md` appends "Revision of 2026-10-20, TASK-13862" at end of file | Git conflicts at the end for whoever lands second. Keep both paragraphs, in merge order: the script appends after whatever is there. |
| Design doc body (lines 206, 207, 221, 227, 228, 239, 253, Risks) | Quote-anchored edits | No other plan edits these lines | No conflict. |
| `llm_roots.py` | One docstring sentence | `plan_fu_chacha_pg_checkouts.md` cites `llm_roots.py:32-37,49` in prose and does not edit it | No conflict. |
| `Docs/Published/` | Refreshes `API-related/Media_Ingest_Jobs_API.md` | `plan_fu_summary_codes.md` (TASK-13865) refreshes `API-related/Audio_Jobs_API.md`; `plan_fu_rg_redis_error.md` refreshes ADRs | Different files. After a rebase, re-run `refresh_docs_published.sh` and `tests/Docs`. |
| `Summarization_General_Lib.py` | Not touched | TASK-13865 | No conflict. `plan_fu_summary_codes.md:133` says it does not touch `Usage_Quotas.md`. |

## Decisions for the owner (recommendation first)

1. **Media ingest jobs: where the refusal is caught (rows 1–4).**
   - **Recommended: check once, when the job starts.** If the job asks for analysis (`media_analysis_requested`), the worker runs the check at the call (`user_to_check` + `check_user_async`) before any work.
     - Refused: it switches analysis off, stores the item, and records the refusal in the job's `result`. A reused transcript keeps its item-level `video_lite.summary_error` (Task 2).
     - Admitted: it marks the rest of the job `prechecked`, so the job runs to completion as an admitted `/media/add` request does.
     - Why:
       - One site covers all eight media types, single and batch.
       - It is the rule `/media/add` already applies to the same processing.
       - A refused job makes no LLM call at all.
       - It adds no function to the quota modules.
     - What changes:
       - A media ingest job is no longer checked at each call. An admitted job can take the user past a limit by its own analysis (a long video's chunk summaries), and `c` concurrent jobs can each do so.
       - A check at the job's start that times out or cannot read the limits admits the whole job, where today it admits one call.
       - TASK-13638 row 24 still holds: its sentence on the operator page is reworded to say the refusal now comes at the job's start, with no LLM call.
       - The design doc gets body edits and a revision note; no ADR covers where jobs are checked (see "ADR assessment").
   - **Middle option: catch at the two persistence sites and redo the item without analysis.** Catch the refusal at `persistence.py:5518` (document-like) and `:4618` (audio/video), mark the result, and re-run the item with `perform_analysis=False`.
     - What it buys: strict checking at each call, with two catch sites instead of sixteen.
     - What it costs: the item is extracted twice. For audio and video that means a second download and a second transcription, minutes of compute per refused item.
     - Any analysis made before the refusal is lost.
     - No steps are written for it.
   - **Alternative: catch the refusal at each processor's analysis step,** and keep the text with the code, as transcript reuse does (`persistence.py:1715,1790`).
     - The sites, measured on dev: `Plaintext_Files.py:512,546`; `PDF_Processing_Lib.py:1176,1227`; `Book_Processing_Lib.py:868,899,1421,1442`; `Email_Processing_Lib.py:443`; `Audio_Files.py:1178,1301`; `Video_DL_Ingestion_Lib.py:1607,1617,1638,1657,1666`. That is 16 call sites in 6 modules, and each needs a clause that stops its chunk loop and skips its recursive pass.
     - What it buys: strict checking at each call, with no overshoot. An item refused part-way keeps the chunk summaries made before the refusal.
     - What it costs: about 6 tests (one per processor), in processors shared with `/media/add` and `/media/process-*`, where the new clauses never fire because those requests are prechecked.
     - No steps are written for it. If chosen, Tasks 1 and 2 are redrafted.
2. **The three HTTP endpoints: what the response is (rows 5–7).** The brief offered two shapes. A third avoids the question.
   - **Recommended: 402 before any work.** Add the §4a boundary check: one `await enforce_llm_quota(current_user.id)` per endpoint, guarded by the predicate in Global Constraints.
     - Why:
       - It is what `/media/add` and the six `/media/process-*` endpoints do for the same choice (analysis on, provider named). The design's own test for the §4a table is "each would do worse than a 402 before any work". These three do worse today: they fetch pages or load items, then drop them.
       - Nothing is discarded. An over-quota user's request does no work, and an admitted request is never refused part-way. That includes the scraping queue jobs it submits, after Task 4.
       - It adds 3 guarded lines and no new response field. Clients already get a 402 from these endpoints today, with the same body.
       - `Watchlists_API.md:506` and TASK-13636's test already promise "a 402 and no output is created" for `POST /watchlists/outputs`.
     - What it costs:
       - An over-quota user who still wants the raw pages must send the request again with summaries off; the 402 says why, and `Retry-After` says when.
       - AC1 is replaced (Task 1 Step 0), because these paths now refuse before producing anything instead of keeping what they produced.
   - **Alternative B: 200 with the results, and the refusal recorded per item.** This is Decision 3 taken literally.
     - Catch the refusal at each summary site:
       - `web_scraping_service.py:351` (`maybe_summarize_one`);
       - `enhanced_web_scraping_service.py:737` (`_summarize_content`);
       - `core/Web_Scraping/enhanced_web_scraping.py:1983` (the scraper's `_summarize_content`, run concurrently by `scrape_multiple`);
       - `watchlists.py:7208`.
     - Mark each article `analysis_status: failed`, `analysis_error: llm_quota_exceeded`, and render the watchlist output with `metadata.summary_error`.
     - It reverses TASK-13636's pinned test (`test_output_summaries_return_a_quota_refusal_as_402`) and `Watchlists_API.md:506`.
     - A client that reads only the status code sees success with no summaries.
     - No steps are written for it.
   - **Alternative C: 402 with the results in the body.** Not recommended.
     - The documented 402 body (`limit_exceeded`, `Usage_Quotas.md:68-70`) would gain a payload on 3 endpoints only.
     - Clients and proxies treat a 4xx body as an error and drop it.
     - It still needs every catch site of Alternative B.

## Review Focus

1. **`/media/ingest-web-content` without a named provider still discards on a refusal.**
   - What happens: `perform_analysis` defaults to true and allows the LLM extraction strategy (`web_scraping_service.py:440,499`). That strategy is in `DEFAULT_EXTRACTION_STRATEGY_ORDER` (`extraction/pipeline.py:48-55`) and runs when jsonld, schema and regex fail and a provider resolves from `llm_settings` or `RAG_DEFAULT_LLM_PROVIDER` (`strategies/llm.py:49-55`).
   - With no `api_name`, Decision 2's check does not run. The extraction call is checked at the call, and its refusal is re-raised (`strategies/llm.py:183`).
     - In the `individual` method it propagates through `scrape_article`'s `except Exception: ... raise` (`web_scraping_service.py:443-445`), and the endpoint answers 402 after scraping.
     - The `url_level` and `recursive_scraping` methods run their pages as scraping queue jobs: after Task 4 a refused page becomes a failed page (Review Focus 2), and on dev it ended a queue worker.
   - A user would expect either a 402 first or the page without LLM extraction.
   - No test here; Task 6 Step 5 files it.
2. **A refused scraping queue job becomes a failed page, not a 402.**
   - After Task 4, a queue job runs as its submitter, so a submitter that was not admitted can be refused at the call inside the job (LLM extraction). Task 4 makes that fail only the job's future; on dev it ended the worker.
   - `scrape_multiple` gathers the futures with `return_exceptions=True` and turns an exception into `{"error": str(exc), "extraction_successful": False}` (`enhanced_web_scraping.py:1950-1961`). So the requester sees a failed page carrying the 402 text, not a 402.
   - The queue's submitters are `scrape_multiple`'s callers:
     - "Individual URLs" (`enhanced_web_scraping_service.py:481`);
     - the scraper's sitemap, URL-level and recursive crawls (`enhanced_web_scraping.py:2066,2338,2557`), reached from `/media/process-web-scraping`, from `/media/ingest-web-content`'s `url_level` and `recursive_scraping` methods, and from the workflow `media_ingest` step (`Workflows/adapters/media/ingest.py:574`).
   - With summaries on, Task 3 admits the request, so its jobs are never refused. The reachable cases:
     - `/media/ingest-web-content` crawls with `perform_analysis` (the default) and no `api_name` (Review Focus 1);
     - workflow runs (§4b), whose jobs now also run as the workflow's user instead of the first caller's.
   - No test beyond Task 4's queue-level test. The follow-up filed in Task 6 Step 5 covers the ingest-web-content case.
3. **The fail-open check at a job's start.**
   - If the limits cannot be read, or the check times out (2 s), `check_user_async` admits, and the job is marked `prechecked`; none of its calls is checked again.
   - That is the design's fail-open rule (§4, `Usage_Quotas.md:109`), now for a whole job, and Task 6 says so on the operator page.
   - `tests/Usage/test_llm_quota_gate.py` covers `check_user_async`'s fail-open; no job-level test.
4. **Overshoot by an admitted job** (Decision 1): a long audio or video item with chunked and recursive summaries runs all of its analysis after admission. With several ingest workers, several jobs can be admitted at once. There is no test, as for `/media/add`.
5. **The watchlist enrichment fallback after an admitted request** (M6).
   - When the Scheduler is unavailable, `schedule_output_enrichment` runs enrichment as a `BackgroundTasks` task of the request (`output_enrichment_handler.py:72-86`). After an admitted `summarize` request, that task inherits the admission and is not refused.
   - `Usage_Quotas.md:97` gains that clause (Task 6); no test.
6. **The admission reaches a lazily started core Scheduler** (re-review M1).
   - When `WORKFLOWS_SCHEDULER_ENABLED` is off (the default), the process-global Scheduler starts on first use. If that first use is `schedule_output_enrichment` inside an admitted `POST /watchlists/outputs` request, its worker tasks inherit that request's admission and user, and every later scheduled task runs in them.
     - The re-review's probe `rr2ka/probe_scheduler_leak.py` shows this on dev code: user 9's task ran with `[{'user_id': 7, 'prechecked': True}]`.
   - It is harmless today. Every Scheduler handler that makes an LLM call runs under `llm_call_root`, which resets `prechecked` and sets the owner: `watchlists_enrich_output` and `workflow_run`.
     - `watchlist_run` has no root, but only the workflows recurring scheduler submits it, and that scheduler starts the global Scheduler at startup.
     - `acp_run` and `watchlists_deliver_briefing` make no in-process accounted LLM call.
   - A new handler that makes an LLM call without `llm_call_root` would run admitted, as another user. The identity and content-scope leak into those workers predates this plan. No test.

## ADR assessment

No ADR needed.
- ADR-064 (per-user quotas), ADR-067 (accounting in the provider base class) and ADR-076 govern, and none says where a job is checked. ADR-067 covers recording; ADR-064 covers how limits resolve.
- Where checks happen is in the design doc (§4a, §4b). Task 6 edits its body and appends a revision note, as TASK-13638 did for the workflow engine's retry.
- Decision 1 moves media ingest jobs from §4b to a §4a-style check at the job's start. That is a design-doc revision, not a new architecture rule.
- Task 4 makes scraping queue jobs follow §1 (a call is for the user whose work it is). That fixes code to match the design.

---

## PR A — TASK-13864

### Task 1: a media ingest job checks the quota once when it starts, keeps its item when refused, and its admission ends with it (rows 1–3; Decision 1)

**Files:**
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py`:
  - imports after line 14 (the blank line after `from loguru import logger`), line 23 (`scope_context`) and line 34 (`worker_sdk`);
  - `_handle_job` (lines 481-483);
  - new helpers before `_create_db` (line 356);
  - after the "prepare" progress update (lines 541-543);
  - after `job_result["chunking_plan"] = final_chunking_plan` (line 666).
- Create: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py`.
- Create: `Docs/superpowers/plans/2026-10-20-keep-artifact-on-llm-quota-refusal.md` (this plan; Step 0).
- Backlog: TASK-13864 (Step 0).

**Interfaces:**
- Consumes (all exist on dev):
  - `media_analysis_requested(form_data: Any) -> bool` (`api/v1/API_Deps/media_processing_deps.py:664`);
  - `user_to_check() -> int | None` and `async check_user_async(user_id: int) -> None`, which raises `LLMQuotaExceededError` when it refuses (`core/Usage/llm_quota_gate.py`);
  - `mark_llm_quota_prechecked() -> None` and `llm_call_context(**overrides)` (`core/Usage/llm_call_context.py`);
  - `refuse_llm_calls(monkeypatch)` (`tests/helpers/llm_quota.py`).
- Produces:
  - `async _admit_or_refuse_analysis(form_data: AddMediaForm) -> LLMQuotaExceededError | None`;
  - `_record_analysis_refusal(job_result: dict[str, Any], refusal: LLMQuotaExceededError) -> None`;
  - the local `analysis_refusal: LLMQuotaExceededError | None` in `_handle_job_in_scope`, which Task 2 passes on;
  - the job `result` keys `status`, `warnings` and `analysis_error`, which Task 6 documents.

- [ ] **Step 0: start the task, commit the plan, and amend the ACs per Decision 2**

```bash
git fetch origin dev --quiet
git switch -c fix/keep-artifact-on-llm-quota-refusal origin/dev
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
cp <scratchpad>/plan_fu_keep_artifact_on_refusal.md Docs/superpowers/plans/2026-10-20-keep-artifact-on-llm-quota-refusal.md
```

If the owner takes Decision 2's recommendation (402 before any work):

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13864 -s "In Progress" \
  --remove-ac 1 \
  --ac "Each listed path keeps the transcript or page it produced and records llm_quota_exceeded where the user can see it, or refuses with the 402 before it fetches or produces anything (LLM usage accounting design section 4a), with a test per path" \
  --ac "A Prompt Studio evaluation refused part-way keeps test_run_ids for the cases that ran, with a test" \
  --ac "A scraping queue job runs in the context of the request that submitted it, and a quota refusal fails only that job, with tests" \
  --append-notes "Plan: Docs/superpowers/plans/2026-10-20-keep-artifact-on-llm-quota-refusal.md. Per its Decision 2, AC1 is replaced: the HTTP endpoints refuse with the 402 before any work, so nothing is produced and discarded. The reworded criterion is now #2; #1 is 'Usage_Quotas.md says what each path keeps'; #3 is the Prompt Studio fix from these notes; #4 is the scraping queue defect found by the plan review."
```

If the owner takes Alternative B instead, keep AC1 and run `-s "In Progress"` with the same two `--ac` lines and `--append-notes "Plan: Docs/superpowers/plans/2026-10-20-keep-artifact-on-llm-quota-refusal.md (Decision 2, Alternative B). #3 is the Prompt Studio fix; #4 is the scraping queue defect."`. Either way the task ends with four ACs, numbered 1 to 4. Then:

```bash
git add -A backlog/tasks Docs/superpowers/plans/2026-10-20-keep-artifact-on-llm-quota-refusal.md
git commit -m "docs(backlog): start TASK-13864; add its plan; amend its acceptance criteria" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 1: write the failing tests.** Create `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py`:

```python
"""A media ingest job whose owner is over an LLM quota keeps its content and records the refusal (TASK-13864)."""

import asyncio
import wave
from pathlib import Path
from typing import Any

import pytest

import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summary
from tldw_Server_API.app.core.Usage import llm_accounting, quota_resolver
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
from tldw_Server_API.app.core.Usage.llm_quota_gate import user_to_check
from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_calls

pytestmark = pytest.mark.integration

_REFUSAL_MESSAGE = "Analysis skipped: LLM quota exceeded (llm_tokens_month)."


def _write_document(tmp_path: Path, name: str) -> Path:
    source = tmp_path / f"{name}.md"
    source.write_text(f"Project {name} opens on 24 January 2027.", encoding="utf-8")
    return source


def _write_wav(tmp_path: Path) -> Path:
    source = tmp_path / "standup.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)
    return source


def _create_job(manager: JobManager, tmp_path: Path, source: Path, media_type: str) -> dict[str, Any]:
    return manager.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        owner_user_id="1",
        payload={
            "batch_id": "quota-refusal",
            "temp_dir": str(tmp_path),
            "media_type": media_type,
            "source": str(source),
            "source_kind": "file",
            "input_ref": source.name,
            "options": {
                "media_type": media_type,
                "perform_analysis": True,
                "api_name": "openai",
                "perform_chunking": False,
                "generate_embeddings": False,
            },
        },
    )


async def _run_jobs(manager: JobManager, count: int) -> None:
    sdk = WorkerSDK(manager, WorkerConfig(domain="media_ingest", queue="default", worker_id="quota-refusal"))
    handled: list[int] = []

    async def handle(job: dict[str, Any]) -> dict[str, Any]:
        try:
            return await worker._handle_job(job, manager, worker._ProgressState())
        finally:
            handled.append(int(job["id"]))
            if len(handled) == count:
                sdk.stop()

    await asyncio.wait_for(sdk.run(handler=handle), timeout=30)


@pytest.fixture
def job_dbs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The jobs DB and the owner's media DB under tmp_path."""
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setattr(worker.DatabasePaths, "get_media_db_path", lambda _: tmp_path / "media.db")
    return tmp_path


@pytest.fixture
def refused(job_dbs: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every LLM quota check refuses, and so does the provider call; returns the provider calls made."""
    provider_calls: list[dict[str, Any]] = []

    def _refused_call(**kwargs: Any) -> str:
        """What the check at the call does on dev: refuse before the provider is reached."""
        provider_calls.append(kwargs)
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(summary, "_summarize_via_adapter", _refused_call)
    refuse_llm_calls(monkeypatch)
    return provider_calls


@pytest.fixture
async def under_the_limit(job_dbs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, with a monthly token limit the owner is under: every check admits."""

    async def _user_quota(_uid: object, key: str) -> int | None:
        return 1000 if key == "limits.llm_tokens_per_month" else None

    async def _no_tokens(_uid: object) -> float:
        return 0.0

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.user_quota", _user_quota)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.llm_tokens_this_month", _no_tokens)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.llm_quota_gate.usage_quotas_enabled", lambda: True)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    quota_resolver.invalidate_all()


@pytest.fixture
def transcribed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speech recognition returns a fixed transcript."""
    monkeypatch.setattr(
        audio,
        "speech_to_text",
        lambda **_kwargs: [{"start_seconds": 0, "end_seconds": 1, "Text": "Send the report on Friday."}],
    )
    monkeypatch.setattr(audio, "check_transcription_model_status", lambda _: {"available": True, "usable": True})


def _assert_kept_with_refusal(job: dict[str, Any], tmp_path: Path, expected_text: str) -> None:
    assert job["status"] == "completed", job
    result = job["result"]
    assert result["media_id"] is not None, (result["status"], result["error"])
    assert result["status"] == "Warning"
    assert result["analysis_error"] == {"code": "llm_quota_exceeded", "message": _REFUSAL_MESSAGE}
    assert _REFUSAL_MESSAGE in result["warnings"]
    db = MediaDatabase(str(tmp_path / "media.db"), client_id="quota-refusal-verifier")
    try:
        assert expected_text in db.get_media_by_id(result["media_id"])["content"]
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_a_refused_document_batch_keeps_every_document(job_dbs: Path, refused: list[dict[str, Any]]) -> None:
    """Each job of a batch is refused at its start: every document is stored, and no LLM call is made."""
    manager = JobManager()
    names = ("Cedar", "Birch")
    rows = [_create_job(manager, job_dbs, _write_document(job_dbs, name), "document") for name in names]

    await _run_jobs(manager, len(rows))

    for row, name in zip(rows, names):
        _assert_kept_with_refusal(manager.get_job(int(row["id"])), job_dbs, f"Project {name} opens")
    assert refused == []


@pytest.mark.asyncio
async def test_a_refused_audio_job_keeps_its_transcript(
    job_dbs: Path, refused: list[dict[str, Any]], transcribed: None
) -> None:
    """An audio job is transcribed and stored without analysis."""
    manager = JobManager()
    row = _create_job(manager, job_dbs, _write_wav(job_dbs), "audio")

    await _run_jobs(manager, 1)

    _assert_kept_with_refusal(manager.get_job(int(row["id"])), job_dbs, "Send the report on Friday.")
    assert refused == []


@pytest.mark.asyncio
async def test_an_admitted_job_is_not_checked_again_part_way(
    job_dbs: Path, under_the_limit: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job admitted at its start runs as an admitted request does: its LLM calls are not checked again."""
    checked_at_the_call: list[int | None] = []

    def _analysis(**_kwargs: Any) -> str:
        checked_at_the_call.append(user_to_check())
        return "Cedar opens in January."

    monkeypatch.setattr(summary, "_summarize_via_adapter", _analysis)
    manager = JobManager()
    row = _create_job(manager, job_dbs, _write_document(job_dbs, "Cedar"), "document")

    await _run_jobs(manager, 1)

    result = manager.get_job(int(row["id"]))["result"]
    assert result["status"] == "Success", result
    assert "analysis_error" not in result
    assert checked_at_the_call == [None]


@pytest.mark.asyncio
async def test_the_admission_ends_with_the_job(
    job_dbs: Path, under_the_limit: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job admitted at its start leaves no admission behind for whatever its caller runs next."""
    monkeypatch.setattr(summary, "_summarize_via_adapter", lambda **_kwargs: "Cedar opens in January.")
    manager = JobManager()
    row = _create_job(manager, job_dbs, _write_document(job_dbs, "Cedar"), "document")

    result = await worker._handle_job(manager.get_job(int(row["id"])), manager, worker._ProgressState())

    assert result["status"] == "Success", result
    assert current_llm_call_context().prechecked is False
```

Why each fixture is shaped this way:
- `refused` makes both checks refuse. `refuse_llm_calls` makes the new job-start check refuse. The `_summarize_via_adapter` stub raises the refusal the way dev's check at the call does, inside `analyze`, so the RED run shows today's discard.
- The audio test runs the real `process_batch_media` and `process_audio_files` on a real WAV, with only speech recognition replaced (as `tests/Media/test_audio_summary_service_prompt.py` does).
- The admitted test reads `user_to_check()` from inside the processor's executor thread. `None` there means the call runs `prechecked`.
- The last test calls `_handle_job` outside `WorkerSDK`, as the dedupe tests do, and checks that no admission is left in the caller's context.

- [ ] **Step 2: run them on unchanged source**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py`

Expected: `3 failed, 1 passed` (measured):
- the batch test: `AssertionError: ('Error', "Processing error: LLMQuotaExceededError: 402: {'error': 'limit_exceeded', 'category': 'llm_tokens_month', 'current': 0, 'limit': 0, 'message': 'Monthly LLM token limit reached'}")`;
- the audio test: `AssertionError: ('Error', 'Failed to call processor: LLMQuotaExceededError')`;
- the admitted test: `assert [1] == [None]`;
- `test_the_admission_ends_with_the_job` passes on dev, which never marks a job. It guards Step 4.

- [ ] **Step 3: implement the check.** In `media_ingest_jobs_worker.py`, add the imports:

```python
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import media_analysis_requested
from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm
```

```python
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
```

```python
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context, mark_llm_quota_prechecked
from tldw_Server_API.app.core.Usage.llm_quota_gate import check_user_async, user_to_check
from tldw_Server_API.app.core.Workspaces.status_projection import derive_workspace_source_media_status
```

Add before `def _create_db(user_id: str):`:

```python
async def _admit_or_refuse_analysis(form_data: AddMediaForm) -> LLMQuotaExceededError | None:
    """Check the owner's LLM quota once, before any work, when the job asks for analysis (TASK-13864).

    Admitted, the rest of the job runs as an admitted request does (LLM usage accounting
    spec §4a), so no refusal lands part-way through an item. Refused, the refusal is
    returned and the job runs without analysis.
    """
    if not media_analysis_requested(form_data):
        return None
    user_id = user_to_check()
    if user_id is None:
        return None
    try:
        await check_user_async(user_id)
    except LLMQuotaExceededError as refusal:
        return refusal
    mark_llm_quota_prechecked()
    return None


def _record_analysis_refusal(job_result: dict[str, Any], refusal: LLMQuotaExceededError) -> None:
    """Say in the job's result why the stored item has no analysis (TASK-13638 Decision 3)."""
    message = f"Analysis skipped: LLM quota exceeded ({refusal.category})."
    job_result["warnings"] = [*(job_result.get("warnings") or []), message]
    job_result["analysis_error"] = {"code": refusal.failure_code, "message": message}
    if job_result.get("status") == "Success":
        job_result["status"] = "Warning"
```

In `_handle_job_in_scope`, between the "prepare" progress update and `db = _create_db(user_id)`. This is before chunking resolution, so an admitted job's auto-chunking LLM call is admitted too:

```python
        progress.percent = 5.0
        progress.message = "prepare"
        jm.update_job_progress(job_id, progress_percent=progress.percent, progress_message=progress.message)

        analysis_refusal = await _admit_or_refuse_analysis(form_data)
        if analysis_refusal is not None:
            form_data = form_data.model_copy(update={"perform_analysis": False})

        db = _create_db(user_id)
```

After the chunking-plan lines of `job_result`:

```python
            if final_chunking_plan is not None:
                job_result["chunking_plan"] = final_chunking_plan
            if analysis_refusal is not None:
                _record_analysis_refusal(job_result, analysis_refusal)
```

Run the file. Expected (measured): `1 failed, 3 passed`. `test_the_admission_ends_with_the_job` fails with `assert True is False`: `mark_llm_quota_prechecked` is a plain `ContextVar` set, and outside `WorkerSDK`'s `llm_call_root` nothing resets it.

- [ ] **Step 4: end the admission with the job.** In `_handle_job`:

```python
    user_id = _resolve_user_id(job)
    # llm_call_context(): an admission the job marks (TASK-13864) ends with the job, whoever calls the handler.
    with scoped_context(user_id=int(user_id), is_admin=False), llm_call_context():
        return await _handle_job_in_scope(job, jm, progress)
```

`llm_call_context()` with no overrides sets a copy of the call context and resets it on exit, so a mark set inside ends there. Inside `WorkerSDK`, `llm_call_root` already did this; the default executor it installs carries the mark into the processors' threads.

- [ ] **Step 5: run the tests and the neighbourhood**

Run the new file. Expected: `4 passed`. Then neighbourhood A:

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -n 4 tldw_Server_API/tests/MediaIngestion_NEW \
  tldw_Server_API/tests/DB_Management/test_media_ingest_worker_content_scope.py \
  tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py \
  tldw_Server_API/tests/Services/test_shutdown_media_ingest_jobs_workers.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk_llm_user.py \
  tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py \
  tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py tldw_Server_API/tests/Usage
```

Expected: at most the one failure that dev has too, `test_media_add_endpoint.py::TestErrorHandling::test_transcription_failure_handling`.
- Measured with the whole plan: `1 failed, 1320 passed, 14 skipped, 1 xfailed`.
- Unchanged dev: `1 failed, 1315 passed, 14 skipped, 1 xfailed`, the same test.
- If it fails, run it alone (`1 passed` measured) before going on.
- Until Task 2 lands, Task 2's test is not yet in the tree.

Then the triage gate (exit 0) and the lint comparison (nothing new).

- [ ] **Step 6: commit** `fix(media): an ingest job checks the LLM quota when it starts, and a refused job keeps its item (TASK-13864)`, with the trailer.

### Task 2: a refused job still records the refusal on a reused transcript (row 4; I1, TASK-13638 row 24)

**Files:**
- Modify `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`:
  - a new helper before `_populate_reused_transcript_summary` (line 1748);
  - its signature and first lines (1748-1757) and its `except LLMQuotaExceededError` (1790-1794);
  - `_maybe_build_reused_transcript_process_result`'s signature (1800-1809) and its call (1902-1907);
  - `process_batch_media`'s signature and docstring (4086-4105) and its reuse call (4391-4399).
- Modify `tldw_Server_API/app/services/media_ingest_jobs_worker.py`: the `process_batch_media(...)` call (lines 582-595).
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py` (append after line 865).

**Interfaces:**
- Consumes: Task 1's `analysis_refusal` local.
- Produces:
  - `process_batch_media(..., analysis_refusal: LLMQuotaExceededError | None = None)`;
  - `_maybe_build_reused_transcript_process_result(..., analysis_refusal=None)`;
  - `_populate_reused_transcript_summary(..., analysis_refusal=None)`;
  - `_record_reused_summary_refusal(process_result, refusal) -> None`.
  - Every new parameter defaults to `None`, so `/media/add` (`persistence.py:3178`) is unchanged.

- [ ] **Step 1: failing test.** Append to `test_media_ingest_jobs_video_dedupe.py`:

```python


@pytest.fixture
async def cross_user_dedupe(monkeypatch, tmp_path):
    """Per-user media DBs and the shared AuthNZ dedupe registry under tmp_path."""
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool

    _set_jobs_db(monkeypatch, tmp_path)
    _allow_url_policy(monkeypatch)
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    await reset_db_pool()
    yield
    await reset_db_pool()


def _reusable_video_job(job_manager, source_url: str, user_id: str) -> dict:
    row = job_manager.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "batch_id": f"batch-refused-reuse-{user_id}",
            "media_type": "video",
            "source": source_url,
            "source_kind": "url",
            "input_ref": source_url,
            "options": {
                "media_type": "video",
                "transcription_model": "whisper-test",
                "perform_analysis": True,
                "api_name": "openai",
                "perform_chunking": False,
            },
        },
        owner_user_id=user_id,
    )
    return job_manager.get_job(int(row["id"]))


@pytest.mark.asyncio
async def test_a_job_refused_at_its_start_records_the_refusal_on_a_reused_transcript(
    cross_user_dedupe,
    monkeypatch,
):
    """TASK-13638 row 24 holds for a job refused when it starts (TASK-13864).

    The reused transcript is stored, no LLM call is made, and the item's video_lite
    metadata says why there is no summary.
    """
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.DB_Management.media_db.api import (
        create_media_database,
        get_full_media_details_rich,
    )
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_calls

    source_url = "https://www.youtube.com/watch?v=refusedreuse1"
    analyze_calls: list[object] = []

    async def _fake_process_videos(**kwargs):
        return {
            "results": [
                {
                    "status": "Success",
                    "input_ref": source_url,
                    "processing_source": source_url,
                    "media_type": "video",
                    "content": "Reusable transcript content.",
                    "segments": [],
                    "analysis": "processor summary",
                    "analysis_details": {"transcription_language": "en"},
                    "metadata": {"title": "Reusable Video", "source_url": source_url, "url": source_url},
                    "warnings": None,
                }
            ],
            "errors_count": 0,
            "errors": [],
        }

    def _refused_analyze(*args, **kwargs):
        """What the check at the call does on dev: refuse before the provider is reached."""
        analyze_calls.append(args)
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_videos",
        _fake_process_videos,
    )
    monkeypatch.setattr("tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.analyze", _refused_analyze)
    job_manager = JobManager()
    await worker._handle_job(_reusable_video_job(job_manager, source_url, "1"), job_manager, worker._ProgressState())
    refuse_llm_calls(monkeypatch)

    result = await worker._handle_job(
        _reusable_video_job(job_manager, source_url, "2"), job_manager, worker._ProgressState()
    )

    db = create_media_database(client_id="test-user-2", db_path=str(DatabasePaths.get_media_db_path("2")))
    try:
        details = get_full_media_details_rich(db, int(result["media_id"]))
    finally:
        db.close_connection()
    assert details["content"]["text"] == "Reusable transcript content."
    assert details["processing"]["safe_metadata"]["video_lite"] == {
        "summary_status": "failed",
        "summary_error": "llm_quota_exceeded",
    }
    assert result["analysis_error"]["code"] == "llm_quota_exceeded"
    assert analyze_calls == []
```

How the test works:
- User 1's job makes the transcript reusable, as `test_media_ingest_cross_user_dedupe_reuses_transcript_without_reprocessing` does.
- In this test environment, user 1's job is checked and admitted at its start (`user_to_check()` returned `1` in the prototype). So the test also proves Task 1 Step 4: without the scope, the admission leaks into user 2's job, and that job is never checked.

- [ ] **Step 2: run it** with Task 1 in place: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q "tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py::test_a_job_refused_at_its_start_records_the_refusal_on_a_reused_transcript"`. Expected (measured): `KeyError: 'video_lite'`. The job turned analysis off, so the reuse path recorded nothing on the item. (On dev it fails later, `KeyError: 'analysis_error'`: dev records `video_lite` through a refused call and has no job-level record.)

- [ ] **Step 3: implement.** In `persistence.py`, before `async def _populate_reused_transcript_summary(`:

```python
def _record_reused_summary_refusal(process_result: dict[str, Any], refusal: LLMQuotaExceededError) -> None:
    """Keep the reused transcript; record why there is no summary (TASK-13638 Decision 3)."""
    _ensure_warnings_list(process_result).append(f"Summary generation failed during dedupe reuse: {refusal}")
    process_result["video_lite_summary_status"] = "failed"
    process_result["video_lite_summary_error"] = refusal.failure_code
```

`_populate_reused_transcript_summary` gains a parameter and a branch:

```python
async def _populate_reused_transcript_summary(
    *,
    process_result: dict[str, Any],
    form_data: Any,
    chunk_options: dict[str, Any] | None,
    loop: asyncio.AbstractEventLoop,
    analysis_refusal: LLMQuotaExceededError | None = None,
) -> None:
    if process_result.get("analysis") or process_result.get("summary"):
        return
    if analysis_refusal is not None:
        # The job was refused when it started (TASK-13864): record it without an LLM call.
        _record_reused_summary_refusal(process_result, analysis_refusal)
        return
    if not getattr(form_data, "perform_analysis", False):
        return
```

and its existing clause becomes:

```python
    except LLMQuotaExceededError as refusal:
        _record_reused_summary_refusal(process_result, refusal)
```

`_maybe_build_reused_transcript_process_result` gains `analysis_refusal: LLMQuotaExceededError | None = None,` after `loop: asyncio.AbstractEventLoop,`, and passes `analysis_refusal=analysis_refusal,` to `_populate_reused_transcript_summary(...)`. `process_batch_media` gains the same parameter after `cancel_check`, with this paragraph after its docstring's first line:

```python
    analysis_refusal: LLMQuotaExceededError | None = None,
) -> list[dict[str, Any]]:
    """
    Core implementation of the audio/video batch processing helper used by `/media/add`.

    ``analysis_refusal``: a media ingest job refused at its start passes its refusal, so
    a reused transcript records it as an LLM call refused here would (TASK-13864).
```

and passes `analysis_refusal=analysis_refusal,` after `loop=loop,` in its `_maybe_build_reused_transcript_process_result(...)` call. In the worker's `process_batch_media(...)` call, after `cancel_check=cancel_check,`:

```python
                cancel_check=cancel_check,
                analysis_refusal=analysis_refusal,
            )
```

- [ ] **Step 4: run** the test (expected `1 passed`), then `tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py` and `tests/MediaIngestion_NEW/unit/test_reused_transcript_llm_quota_refusal.py`. Expected: `13 passed` together with Task 1's file, as measured. Then neighbourhood A (Task 1 Step 5): measured `1 failed, 1320 passed, 14 skipped, 1 xfailed`, the failure being the dev one. Then the lint comparison: nothing new. The `import ... as worker` line goes first in the test's local import block, or ruff reports `I001`.

- [ ] **Step 5: commit** `fix(media): a job refused at its start still records the refusal on a reused transcript (TASK-13864)`, with the trailer.

### Task 3: the three HTTP endpoints refuse before any work (rows 5–7; Decision 2)

If the owner takes Alternative B, skip this task; the plan is redrafted for it.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py`: imports after line 13 (`DB_Deps`) and line 19 (`Prompts_DB_Deps`); the check after the URL validation at lines 73-74.
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py`: import after line 21 (`Prompts_DB_Deps`); the check before the `try:` at line 75.
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`: import between lines 57 (`DB_Deps`) and 58 (`Watchlists_DB_Deps`); the check after the `llm_provider_required` 400 (lines 7191-7194), before `items_for_summary = [` (line 7195).
- Test: `tldw_Server_API/tests/Media/test_web_summary_service_prompt.py`: an import after line 27; tests appended at the end (after line 527).
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_api.py`: a helper and two tests appended at the end (after line 1863).

**Interfaces:**
- Consumes: `enforce_llm_quota(user_id, estimated_tokens=1)` (`api/v1/API_Deps/usage_quota_deps.py:47`), which raises `LLMQuotaExceededError` or marks the request `prechecked`; `media_analysis_requested`; `refuse_llm_work` and `assert_llm_quota_402` (`tests/helpers/llm_quota.py`).
- Produces: the admission that Task 4 carries into scraping queue jobs.

- [ ] **Step 1: write the failing tests.** In `test_web_summary_service_prompt.py`, after `from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage`:

```python
from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work
```

Append to the end of the file. The module's `context` fixture keeps orchestration and summarization real, and records every fetch in `context.fetches` and every model call in `context.calls`.

```python


WEB_SUMMARY_REQUESTS = [
    pytest.param(
        "/api/v1/media/ingest-web-content",
        {"urls": ["https://example.com/a"], "scrape_method": "individual", "api_name": "openai", "perform_chunking": False},
        "perform_analysis",
        id="ingest-web-content",
    ),
    pytest.param(
        "/api/v1/media/process-web-scraping",
        {
            "scrape_method": "Individual URLs",
            "url_input": "https://example.com/a",
            "api_name": "openai",
            "mode": "ephemeral",
            "perform_chunking": False,
        },
        "summarize_checkbox",
        id="process-web-scraping",
    ),
]


@pytest.mark.parametrize(("path", "body", "summarize_flag"), WEB_SUMMARY_REQUESTS)
def test_summaries_for_an_over_quota_user_are_refused_before_any_page_is_fetched(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, path: str, body: dict[str, Any], summarize_flag: str
) -> None:
    """The 402 comes first, so no page is fetched and then discarded (TASK-13864)."""
    refuse_llm_work(monkeypatch)
    response = context.client.post(path, json={**body, summarize_flag: True})
    assert_llm_quota_402(response)
    assert context.fetches == []
    assert context.calls == []


@pytest.mark.parametrize(("path", "body", "summarize_flag"), WEB_SUMMARY_REQUESTS)
def test_scraping_without_summaries_is_not_refused(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, path: str, body: dict[str, Any], summarize_flag: str
) -> None:
    """Without summaries there is no LLM work to refuse, so the page comes back."""
    refuse_llm_work(monkeypatch)
    response = context.client.post(path, json={**body, summarize_flag: False})
    assert response.status_code == 200, response.text
    assert context.fetches == ["https://example.com/a"]
```

(The leading blank lines keep two blank lines after the last test; the block has no trailing whitespace.)

Append to the end of `test_watchlists_api.py`:

```python


def _over_quota_run(c, monkeypatch) -> tuple[int, list[list[dict]]]:
    """A finished run for a user over an LLM quota; returns the run id and the summary calls made."""
    from tldw_Server_API.app.api.v1.endpoints import watchlists as watchlists_endpoint
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    monkeypatch.setenv("TEST_MODE", "1")
    summarized: list[list[dict]] = []

    async def _summarize(items, **_kwargs):
        summarized.append(items)
        return items

    monkeypatch.setattr(watchlists_endpoint, "summarize_items_for_output", _summarize)
    r = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "Quota Feed", "url": "https://example.com/quota.xml", "source_type": "rss", "tags": ["quota"]},
    )
    assert r.status_code == 200, r.text
    r = c.post("/api/v1/watchlists/jobs", json={"name": "Quota Digest", "scope": {"tags": ["quota"]}})
    assert r.status_code == 200, r.text
    r = c.post(f"/api/v1/watchlists/jobs/{r.json()['id']}/run")
    assert r.status_code == 200, r.text
    refuse_llm_work(monkeypatch)
    return int(r.json()["id"]), summarized


def test_output_summaries_refuse_an_over_quota_user_before_any_summary(client_with_user, monkeypatch):
    """An over-quota user asking for summaries gets the 402 before any summary call (TASK-13864)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402

    c = client_with_user
    run_id, summarized = _over_quota_run(c, monkeypatch)

    r = c.post("/api/v1/watchlists/outputs", json={"run_id": run_id, "summarize": True, "llm_provider": "openai"})

    assert_llm_quota_402(r)
    assert summarized == []
    r = c.get("/api/v1/watchlists/outputs", params={"run_id": run_id})
    assert r.status_code == 200, r.text
    assert r.json()["total"] == 0


def test_output_without_summaries_is_not_refused_for_an_over_quota_user(client_with_user, monkeypatch):
    """Without summarize there is no LLM work to refuse, so the output is written (TASK-13864)."""
    c = client_with_user
    run_id, summarized = _over_quota_run(c, monkeypatch)

    r = c.post("/api/v1/watchlists/outputs", json={"run_id": run_id, "summarize": False})

    assert r.status_code == 200, r.text
    assert r.json()["run_id"] == run_id
    assert summarized == []
```

`refuse_llm_work` makes only the boundary decision refuse. On dev these endpoints have no boundary check, so they answer 200. That is the defect this task fixes: today the refusal comes only at the call, after the work.

- [ ] **Step 2: run them on unchanged source**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  "tldw_Server_API/tests/Media/test_web_summary_service_prompt.py::test_summaries_for_an_over_quota_user_are_refused_before_any_page_is_fetched" \
  "tldw_Server_API/tests/Media/test_web_summary_service_prompt.py::test_scraping_without_summaries_is_not_refused" \
  "tldw_Server_API/tests/Watchlists/test_watchlists_api.py::test_output_summaries_refuse_an_over_quota_user_before_any_summary" \
  "tldw_Server_API/tests/Watchlists/test_watchlists_api.py::test_output_without_summaries_is_not_refused_for_an_over_quota_user"
```

Expected (measured): `3 failed, 3 passed`.
- Each failure is `assert_llm_quota_402`'s `assert response.status_code == 402`, with a 200 body:
  - `"status":"success","message":"Web content processed"` with `"analysis":"Summary result."`;
  - `"status":"ephemeral-ok"`;
  - the watchlist output `{"id":1,...,"type":"briefing_markdown",...}`.
- The three controls pass.

- [ ] **Step 3: implement.** `ingest_web_content.py` imports:

```python
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import media_analysis_requested
from tldw_Server_API.app.api.v1.API_Deps.media_route_deps import media_create_dependencies
```

```python
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota
```

and in the handler:

```python
    # Basic validation: require at least one URL.
    if not request.urls:
        raise HTTPException(status_code=400, detail="At least one URL is required")
    if media_analysis_requested(request):
        # Usage quotas (LLM usage accounting spec §4a), before any page is fetched.
        await enforce_llm_quota(current_user.id)
```

`process_web_scraping.py` import:

```python
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota
```

and at the top of the handler body, after the docstring, before `try:`:

```python
    if payload.summarize_checkbox:
        # Usage quotas (LLM usage accounting spec §4a), before any page is fetched. Summaries
        # default to a provider when api_name is empty, so the checkbox alone means LLM work.
        await enforce_llm_quota(current_user.id)
```

`watchlists.py` import:

```python
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota
from tldw_Server_API.app.api.v1.API_Deps.Watchlists_DB_Deps import get_watchlists_db_for_user
```

and in `create_output`, inside `if payload.summarize:`:

```python
        if not llm_provider:
            raise HTTPException(
                status_code=400,
                detail="llm_provider_required: set llm_provider in request or job output_prefs.llm.provider",
            )
        # Usage quotas (LLM usage accounting spec §4a), before any summary call and before the output is written.
        await enforce_llm_quota(current_user.id)
        items_for_summary = [
```

- [ ] **Step 4: run the tests and the neighbourhood.** Run Step 2's command plus `::test_output_summaries_return_a_quota_refusal_as_402`. Expected: `7 passed`. Then neighbourhood B, and the OpenAPI contracts **alone**:

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -n 4 tldw_Server_API/tests/Media tldw_Server_API/tests/Web_Scraping \
  tldw_Server_API/tests/WebScraping tldw_Server_API/tests/Watchlists \
  tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ/test_auth_dependency_contract.py tldw_Server_API/tests/lint
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Services/test_openapi_contracts.py
PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts
```

Expected:
- Neighbourhood B: no failures. Measured with the whole plan: `4064 passed, 37 skipped, 1 xpassed` (3 min 44 s; `suite2_web.log`), including `WebScraping/test_refactor_import_inventory.py` against the regenerated inventory.
  - On a fresh worktree's first run, `Web_Scraping/test_phase4_predecessor_fixtures.py` can hit "Fixture publication lock could not be opened" between xdist workers (review run 1). Re-run; alone it gives `14 passed`.
- `test_openapi_contracts.py` alone: `74 passed` (`openapi2.log`).
  - Do not add it to the `-n 4` run. After `Watchlists/test_watchlists_api.py::test_output_summaries_return_a_quota_refusal_as_402` in the same process, `test_openapi_operation_ids_are_present_and_unique` fails on unchanged dev too, with the connectors webhook's duplicate operationId (`order_dev.log`: `1 failed, 1 passed`). Task 6 Step 5 files it.
- The OpenAPI gate: "fingerprint matches" and "path union matches". The handler signatures do not change.

Then the lint comparison: nothing new.

- [ ] **Step 5: commit** `fix(quotas): web scraping and watchlist outputs with summaries refuse an over-quota user before any work (TASK-13864)`, with the trailer.

### Task 4: scraping queue jobs run as the request that submitted them (row 8; I3)

**Files:**
- Modify `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`:
  - `import contextvars` after `import contextlib` (line 19);
  - the `core.exceptions` import (line 45);
  - a `ScrapingJob` field after `cancel_requested` (line 266);
  - `_record_completed_job` (667-669) and `add_job` (706-708);
  - the job execution (758-759) and the worker's `except` (804).
- Create: `tldw_Server_API/tests/Web_Scraping/test_scraping_job_queue_call_context.py`.
- Regenerate: `Docs/Design/web_scraping_refactor_import_inventory.json`, `Docs/Design/WebScraping_Refactor_Import_Inventory.md`.

**Interfaces:**
- Consumes: Task 3's admission (set in the request's context before `add_job` is called).
- Produces: `ScrapingJob.context: Optional[contextvars.Context]`. It is not in `to_dict()`, and it is cleared when the job is recorded as completed.

- [ ] **Step 1: failing tests.** Create `test_scraping_job_queue_call_context.py`. The first test is the review's probe (`rvka/probe_queue_leak.py`):

```python
"""Scraping queue workers run each job as the request that submitted it (TASK-13864)."""

import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage.llm_call_context import (
    current_llm_call_context,
    llm_call_context,
    mark_llm_quota_prechecked,
)
from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ScrapingJob, ScrapingJobQueue

pytestmark = pytest.mark.unit


def _job(job_id: str) -> ScrapingJob:
    return ScrapingJob(job_id=job_id, url=f"https://example.com/{job_id}", method="trafilatura")


@pytest.mark.asyncio
async def test_a_job_runs_with_its_submitters_user_not_the_admission_of_the_request_that_started_the_workers() -> None:
    """The workers start lazily inside a request; a later request's job must not run as that request."""
    queue = ScrapingJobQueue(max_concurrent=1)
    seen: list[dict[str, Any]] = []

    async def _execute(job: ScrapingJob) -> dict[str, Any]:
        context = current_llm_call_context()
        seen.append({"user_id": context.user_id, "prechecked": context.prechecked})
        return {"extraction_successful": True}

    queue._execute_job = _execute  # type: ignore[method-assign]

    async def first_request() -> None:
        """User 7, admitted at the boundary, starts the service's workers."""
        with llm_call_context(user_id=7):
            mark_llm_quota_prechecked()
            await queue.start()

    async def second_request() -> dict[str, Any]:
        """User 9, not admitted."""
        with llm_call_context(user_id=9):
            future = await queue.add_job(_job("j2"))
            return await asyncio.wait_for(future, timeout=5)

    try:
        await asyncio.create_task(first_request())
        result = await asyncio.create_task(second_request())
    finally:
        await queue.stop()

    assert result == {"extraction_successful": True}
    assert seen == [{"user_id": 9, "prechecked": False}]


@pytest.mark.asyncio
async def test_a_refused_job_fails_with_the_refusal_and_the_worker_runs_the_next_job() -> None:
    """A quota refusal in one job is that job's failure, not the worker's."""
    queue = ScrapingJobQueue(max_concurrent=1)

    async def _execute(job: ScrapingJob) -> dict[str, Any]:
        if job.job_id == "refused":
            raise LLMQuotaExceededError(
                category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
            )
        return {"extraction_successful": True, "url": job.url}

    queue._execute_job = _execute  # type: ignore[method-assign]
    await queue.start()
    try:
        refused = await queue.add_job(_job("refused"))
        with pytest.raises(LLMQuotaExceededError):
            await asyncio.wait_for(refused, timeout=5)
        following = await queue.add_job(_job("next"))
        result = await asyncio.wait_for(following, timeout=5)
    finally:
        await queue.stop()

    assert result == {"extraction_successful": True, "url": "https://example.com/next"}
```

- [ ] **Step 2: run it.** `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Web_Scraping/test_scraping_job_queue_call_context.py`. Expected (measured): `2 failed`:
  - `assert [{'user_id': 7, 'prechecked': True}] == [{'user_id': 9, 'prechecked': False}]`;
  - `TimeoutError` inside `pytest.raises(LLMQuotaExceededError)`: the refusal escaped `_worker` and ended the worker task, so the job's future never resolved.

- [ ] **Step 3: implement**

```python
import asyncio
import atexit
import contextlib
import contextvars
import hashlib
```

```python
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError, NetworkError
```

```python
    metadata: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    # The submitting request's context; the job runs in it (TASK-13864). Dropped when the job ends.
    context: Optional[contextvars.Context] = field(default=None, repr=False, compare=False)
```

```python
    def _record_completed_job(self, job: ScrapingJob) -> None:
        job.context = None
        self._completed_jobs[job.job_id] = job
        self._trim_completed_jobs()
```

In `add_job`:

```python
            # Create future for job result
            future = asyncio.Future()
            job.context = contextvars.copy_context()
            self._job_futures[job.job_id] = future
```

In `_worker`:

```python
                # Execute the job as its submitter: their user, and their admission if any (TASK-13864).
                result = await asyncio.create_task(self._execute_job(job), context=job.context)
```

```python
            except asyncio.TimeoutError:
                continue
            except (*_WEBSCRAPE_NONCRITICAL_EXCEPTIONS, LLMQuotaExceededError) as e:
                logger.error(f"{worker_id} error: {e}")
```

Notes:
- `start()` is unchanged (re-review M4).
  - The workers keep the context of the request that starts them, but outside a job they do no LLM or DB work. They only take jobs off the queue, log, record completions and resolve futures.
  - Every job enters through `add_job`, so every job runs in its own submitter's context.
  - With `start()` changed or not, both tests pass; this was measured both ways.
- Retries keep `job.context` (they re-queue without `_record_completed_job`).
- Cancelling a worker cancels the job task it awaits.
- The `except` clause already marks the job `FAILED` and sets the exception on its future; a refusal now takes that path instead of ending the worker.

- [ ] **Step 4: regenerate the import inventory, and run.**

```bash
$PY Helper_Scripts/web_scraping_refactor_inventory.py --json Docs/Design/web_scraping_refactor_import_inventory.json --markdown Docs/Design/WebScraping_Refactor_Import_Inventory.md
git status --short Docs/Design/
$PY Helper_Scripts/ci/scope_predicate_ratchet.py
```

Expected (measured):
- Both inventory files change: `enhanced_web_scraping.py` import lines shift by one, and the new test's `ScrapingJob` and `ScrapingJobQueue` imports are added.
- "28 conditional scope predicates; baseline exact".

Then run the new file (`2 passed`) and neighbourhood B (Task 3 Step 4). `WebScraping/test_refactor_import_inventory.py` checks the regenerated inventory. Then the triage gate (exit 0) and the lint comparison.

- [ ] **Step 5: commit** `fix(web-scraping): queue jobs run as the request that submitted them, and a refusal fails only its job (TASK-13864)`, with the trailer. Include the two inventory files.

### Task 5: a refused Prompt Studio evaluation keeps its test_run_ids (row 9)

**Files:** Modify `tldw_Server_API/app/core/Prompt_Management/prompt_studio/evaluation_manager.py`, the failed update at lines 260-267. Test: `tldw_Server_API/tests/prompt_studio/unit/test_evaluation_llm_quota_refusal.py` (append at the end; it already has `_EvaluationDb`, whose `create_test_run` returns ids 1, 2, …, and `_refusal()`).

**Interfaces:** Consumes nothing new. Produces the `test_run_ids` key on the failed record, which Task 6 documents.

- [ ] **Step 1: failing test.** Append:

```python


@pytest.mark.asyncio
async def test_a_refused_evaluation_keeps_the_links_to_the_cases_that_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failed record still points to the test runs stored before the refusal (TASK-13864)."""
    db = _EvaluationDb()
    calls: list[dict[str, Any]] = []

    def _second_refused(self: EvaluationManager, **kwargs: Any) -> Any:
        calls.append(kwargs)
        if len(calls) > 1:
            raise _refusal()
        return "a"

    monkeypatch.setattr(EvaluationManager, "_call_adapter_text", _second_refused)

    with pytest.raises(LLMQuotaExceededError):
        await EvaluationManager(db).run_evaluation_with_existing_record(
            evaluation_id=5, prompt_id=1, test_case_ids=[2, 3], model="m"
        )

    assert len(db.runs) == 1
    assert db.updates[-1]["status"] == "failed"
    assert db.updates[-1]["test_run_ids"] == [1]
```

- [ ] **Step 2: run it.** `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/prompt_studio/unit/test_evaluation_llm_quota_refusal.py`. Expected: `1 failed, 2 passed`, with `KeyError: 'test_run_ids'` (measured).
- [ ] **Step 3: implement**

```python
            except LLMQuotaExceededError as refusal:
                # Stop at the user's limit and record it; the remaining cases would be refused too.
                self.db.update_evaluation(
                    evaluation_id,
                    {
                        "status": "failed",
                        "completed_at": datetime.utcnow(),
                        "test_run_ids": test_run_ids,
                        "error_message": f"{refusal.failure_code}: {refusal.detail['message']}",
                    },
                )
                raise
```

The completed path already writes `test_run_ids` through the same `update_evaluation` (`evaluation_manager.py:303-311`), and `PromptStudioDatabase.update_evaluation` serialises it (`PromptStudioDatabase.py:602`).

- [ ] **Step 4: run** the file (expected `3 passed`), then `tests/prompt_studio` with `-n 4` (measured: `1181 passed, 17 skipped`), the triage gate and the lint comparison.
- [ ] **Step 5: commit** `fix(prompt-studio): an evaluation refused part-way keeps the links to the test runs it stored (TASK-13864)`, with the trailer.

### Task 6: the operator page, the design doc, the jobs API doc, follow-ups and close-out (AC2)

**Files:**
- Modify: `Docs/Operations/Usage_Quotas.md` (lines 93, 94, 95, 97, 98, 106, 109, 110).
- Modify: `Docs/Design/2026-10-09-llm-usage-accounting-design.md`:
  - lines 206, 207, 221 (a new row after it), 227, 228, 239 (the §4b intro), 253 and 433 (Risks);
  - a revision appended at the end of the file.
- Modify: `Docs/API-related/Media_Ingest_Jobs_API.md` (after line 36), with its mirror `Docs/Published/API-related/Media_Ingest_Jobs_API.md`.
- Modify: `tldw_Server_API/app/core/Usage/llm_roots.py`, the `llm_call_root` docstring (lines 59-60).
- Backlog: TASK-13864, and two new tasks.

- [ ] **Step 1: apply the doc edits.** Every edit is found by its quoted text, and the script fails if a quote is not found exactly once. If another plan has rewritten a line, re-apply the edit by hand to the landed text (see "Conflicts with other plans"). The design revision is appended at end of file, after any other plan's revision. This is the scratchpad's `addon_docs2.py`, run on the prototype:

```bash
$PY - <<'EOF'
from pathlib import Path


def edit(rel, pairs):
    path = Path(rel)
    text = path.read_text()
    for old, new in pairs:
        assert text.count(old) == 1, (rel, old[:80])
        text = text.replace(old, new)
    path.write_text(text)


def append(rel, block):
    path = Path(rel)
    path.write_text(path.read_text().rstrip("\n") + "\n\n" + block)


edit("Docs/Operations/Usage_Quotas.md", [
    (
        "benchmark runs (`POST /evaluations/benchmarks/{name}/run` and `POST /benchmarks/{name}/run`);",
        "benchmark runs (`POST /evaluations/benchmarks/{name}/run` and `POST /benchmarks/{name}/run`);"
        " `/media/ingest-web-content` when analysis is on and a provider is named;"
        " `/media/process-web-scraping` with `summarize_checkbox`, with or without `api_name`"
        " (a summary without one uses a default provider); `POST /watchlists/outputs` with `summarize`;",
    ),
    (
        "Background jobs (`/media/ingest-jobs`, evaluation recipe runs),",
        "Background jobs (`/media/ingest/jobs`, evaluation recipe runs; a media ingest job that asks for"
        " analysis is checked once, when it starts, below),",
    ),
    (
        "returns the same 402: `/translate`, `/media/ingest-web-content` with analysis,"
        " `/media/process-web-scraping` with summarization, `POST /watchlists/outputs` with `summarize`,"
        " `/media/{id}/insights` and the manuscript analysis endpoints among them.",
        "returns the same 402: `/translate`, `/media/{id}/insights` and the manuscript analysis endpoints among them.",
    ),
    (
        " The web scraping endpoints and `POST /watchlists/outputs` discard that request's results.",
        "",
    ),
    (
        "and `enrichment_error: 402: {...}`, and the task is not retried.",
        "and `enrichment_error: 402: {...}`, and the task is not retried. When the Scheduler is unavailable,"
        " enrichment runs as a background task of the request instead, so after an admitted `summarize`"
        " request it is admitted with it and not refused.",
    ),
    (
        "A media ingest job marks the item `Error` (`Processing error: LLMQuotaExceededError: ...`);"
        " an audio or video batch marks every item in the batch.",
        "A media ingest job (`/media/ingest/jobs`) that asks for analysis is checked once, when it starts,"
        " not at each call. Refused, it stores the item without analysis: the job's `result` has"
        " `status: Warning`, the warning `Analysis skipped: LLM quota exceeded (<category>).` and"
        " `analysis_error.code = \"llm_quota_exceeded\"`. Admitted, it runs to completion, as an admitted"
        " request does. Each job of a batch is checked when it starts, so once the user is over a limit"
        " the rest of the batch is stored without analysis and makes no LLM call.",
    ),
    (
        "A media item that reuses an earlier transcript keeps it and gets no summary: summarizing stops at"
        " the first refusal, the item's warnings carry the 402,",
        "A media item that reuses an earlier transcript, in a job refused when it starts, keeps the transcript"
        " and gets no summary: no LLM call is made, the item's warnings carry the 402,",
    ),
    (
        "A background Prompt Studio evaluation ends `failed`, with `error_message` `llm_quota_exceeded: ...`.",
        "A Prompt Studio evaluation refused part-way, synchronous or background, ends `failed`, with"
        " `error_message` `llm_quota_exceeded: ...`, and its `test_run_ids` list the cases that ran before"
        " the refusal.",
    ),
    (
        "so a user can pass a limit by one request's worth of usage. A media batch or an evaluation batch can be large.",
        "so a user can pass a limit by one request's worth of usage. A media batch or an evaluation batch can be large."
        " So can each media ingest job admitted when it starts.",
    ),
    (
        "A check at the call that gets no answer within two seconds lets the call through, with a warning.",
        "A check at the call that gets no answer within two seconds lets the call through, with a warning;"
        " at a media ingest job's start, it admits the whole job.",
    ),
    (
        "so its synchronous LLM calls are neither checked nor recorded; the first one logs a warning.",
        "so its synchronous LLM calls are neither checked nor recorded; the first one logs a warning. A media"
        " ingest job's check when it starts still runs, on that process's own loop.",
    ),
])

DESIGN = "Docs/Design/2026-10-09-llm-usage-accounting-design.md"
edit(DESIGN, [
    (
        "the default executor and the chat request queue.",
        "the default executor, the chat request queue, and the scraping queue jobs the request submits"
        " (revision of 2026-10-20).",
    ),
    (
        "the next request on the same connection, or a job the request enqueues.",
        "the next request on the same connection, or a job the request enqueues in the Jobs queue.",
    ),
    (
        "A job runs under `llm_call_root(owner)` and is checked by §4b when it runs.",
        "A job runs under `llm_call_root(owner)` and is checked by §4b when it runs; a media ingest job that"
        " asks for analysis checks itself once, when it starts (revision of 2026-10-20).",
    ),
    (
        "`/media/ingest-jobs` and evaluation recipe runs. The job is checked by §4b when it runs.",
        "`/media/ingest/jobs` and evaluation recipe runs. The job is checked when it runs: a media ingest job"
        " that asks for analysis once, when it starts; everything else by §4b.",
    ),
    (
        "| Benchmark runs (`/evaluations/benchmarks/{name}/run`, `/benchmarks/{name}/run`) | Always. Checked only at the call, every item would be refused and counted as failed, and the run would still complete. |\n",
        "| Benchmark runs (`/evaluations/benchmarks/{name}/run`, `/benchmarks/{name}/run`) | Always. Checked only at the call, every item would be refused and counted as failed, and the run would still complete. |\n"
        "| `/media/ingest-web-content`, `/media/process-web-scraping` and `POST /watchlists/outputs` | When the request asks for summaries: analysis with a provider named, `summarize_checkbox`, or `summarize`. Checked only at the call, a refusal discarded the pages or items the request had already fetched. |\n",
    ),
    (
        "these HTTP endpoints are not in the table: `/media/ingest-web-content` with analysis,"
        " `/media/process-web-scraping` with summarization, `/media/{id}/insights`,",
        "these HTTP endpoints are not in the table: `/media/{id}/insights`,",
    ),
    (
        "So are background jobs (`/media/ingest-jobs`, evaluation recipe runs, audio analysis jobs),",
        "So are background jobs (evaluation recipe runs, audio analysis jobs, and media ingest jobs that do"
        " not ask for analysis),",
    ),
    (
        "A media ingestion job reports the item as failed.",
        "A media ingestion job that asks for analysis is checked when it starts instead; refused, it stores"
        " the item without analysis.",
    ),
    (
        "Background jobs, workflow runs, WebSocket sessions and any HTTP endpoint that is not in the table have"
        " no boundary check.",
        "Background jobs, workflow runs, WebSocket sessions and any HTTP endpoint that is not in the table have"
        " no boundary check (a media ingest job that asks for analysis checks itself when it starts, revision of"
        " 2026-10-20).",
    ),
    (
        "The overshoot is bounded by one request, not one call.",
        "The overshoot is bounded by one request, not one call, and by one media ingest job, which is admitted"
        " when it starts.",
    ),
])
append(
    DESIGN,
    "Revision of 2026-10-20, TASK-13864:\n"
    "\n"
    "- **A media ingest job that asks for analysis is checked once, when it starts,** and then runs `prechecked`,"
    " as an admitted request does; the admission ends with the job. Refused, it stores the item without analysis"
    " and records `analysis_error.code = \"llm_quota_exceeded\"` in its result, instead of failing the item; a"
    " reused transcript still records `video_lite.summary_error`. Other jobs are still checked at the call (§4b).\n"
    "- **The two web scraping endpoints and `POST /watchlists/outputs` join the §4a table.**\n"
    "- **Scraping queue jobs run as the request that submitted them.** The web scraping service starts its queue"
    " workers lazily, inside a request, and they ran every later job in that request's context. Each job now runs"
    " in a copy of its submitter's context, so its LLM calls are recorded and checked for that user and no"
    " request's admission reaches another's jobs (§1).\n",
)

edit("Docs/API-related/Media_Ingest_Jobs_API.md", [
    (
        "  - Response includes progress fields (`progress_percent`, `progress_message`), result summary,\n"
        "    and payload metadata (`media_type`, `source`, `source_kind`, `batch_id`).\n",
        "  - Response includes progress fields (`progress_percent`, `progress_message`), result summary,\n"
        "    and payload metadata (`media_type`, `source`, `source_kind`, `batch_id`).\n"
        "  - A job that asks for analysis while its owner is over an LLM quota still stores the item, without\n"
        "    analysis. Its `result` has `status: \"Warning\"`, the warning\n"
        "    `Analysis skipped: LLM quota exceeded (<category>).` and\n"
        "    `analysis_error: {\"code\": \"llm_quota_exceeded\", \"message\": ...}`.\n",
    ),
])

edit("tldw_Server_API/app/core/Usage/llm_roots.py", [
    (
        "    already set. A root is never admitted by the request that started it: its\n"
        "    calls are quota-checked when they run (LLM usage accounting spec §4a, §4b).\n",
        "    already set. A root is never admitted by the request that started it: its\n"
        "    calls are quota-checked when they run (LLM usage accounting spec §4a, §4b).\n"
        "    A media ingest job that asks for analysis checks once when it starts, and\n"
        "    then marks itself admitted until it ends (TASK-13864).\n",
    ),
])
EOF
```

If the owner takes Decision 1's middle option or alternative, replace the media ingest job sentences, the `:94`/`:106`/`:109`/`:110` clauses, the design body edits and the first revision bullet with that plan's wording. If the owner takes Decision 2's Alternative B, drop the boundary-list insertion, the table row and the second revision bullet, and leave the HTTP bullet's endpoint list in place.

- [ ] **Step 2: refresh the mirror and run the doc tests**

```bash
bash Helper_Scripts/refresh_docs_published.sh
git status --short Docs/ tldw_Server_API/app/core/Usage/
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs
```

Expected (measured): `git status` lists the three edited docs, `Docs/Published/API-related/Media_Ingest_Jobs_API.md` and `llm_roots.py` (the inventories and the plan copy were committed earlier). `tests/Docs`: `212 passed` (`docs2.log`).

- [ ] **Step 3: the close-out checks**
  - `$PY Helper_Scripts/ci/test_quality_triage.py --enforce`: exit 0.
  - `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/CI/test_test_quality_ratchet.py`: `1 passed`.
  - `$PY Helper_Scripts/ci/check_shard_coverage.py`: "OK — no newly-unshared test files".
  - `$PY Helper_Scripts/ci/scope_predicate_ratchet.py`: "28 conditional scope predicates; baseline exact."
  - The lint comparison over the 8 app files and 6 test files: nothing new (measured: 23 findings before, the same 23 after).
  - Bandit: `uvx bandit -q` on `media_ingest_jobs_worker.py`, `persistence.py`, `enhanced_web_scraping.py`, `llm_roots.py`, `ingest_web_content.py`, `process_web_scraping.py`, `watchlists.py` and `prompt_studio/evaluation_manager.py`. Expected: no output (measured; bandit is not in `.venv`). If `uvx` is unavailable, record the skip.

- [ ] **Step 4: commit the docs**

```bash
git add Docs/Operations/Usage_Quotas.md Docs/Design/2026-10-09-llm-usage-accounting-design.md \
  Docs/API-related/Media_Ingest_Jobs_API.md Docs/Published/API-related/Media_Ingest_Jobs_API.md \
  tldw_Server_API/app/core/Usage/llm_roots.py
git commit -m "docs(quotas): what each add-on LLM path keeps on a quota refusal (TASK-13864)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: file the two follow-ups.** Compute the highest task id with the AGENTS.md recipe. Page through open PRs with more than 100 files, because `gh pr list --json files` returns at most 100 per PR:

```bash
git fetch origin dev --quiet
HIGHEST=$( {
  git ls-tree -r --name-only origin/dev backlog/
  gh pr list --state open --limit 500 --json files --jq '.[].files[].path'
  for n in $(gh pr list --state open --limit 500 --json number,changedFiles --jq '.[] | select(.changedFiles > 100) | .number'); do
    gh api --paginate "repos/rmusser01/tldw_server/pulls/$n/files" --jq '.[].filename'
  done
} | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1 )
echo "$HIGHEST"
EXTRACT="TASK-$((HIGHEST + 1))"
OPID="TASK-$((HIGHEST + 2))"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Web content ingest discards scraped pages when its LLM extraction is refused" --id "$EXTRACT" -l quotas,media,backend \
  -d "/media/ingest-web-content allows the LLM extraction strategy whenever perform_analysis is true, which is the default (services/web_scraping_service.py, ingest_web_content_orchestrate: allow_llm_extraction=bool(request.perform_analysis)). The strategy is in DEFAULT_EXTRACTION_STRATEGY_ORDER (Web_Scraping/extraction/pipeline.py) and runs when jsonld, schema and regex fail and a provider resolves from llm_settings or RAG_DEFAULT_LLM_PROVIDER. TASK-13864 checks the quota at the request boundary only when a summary provider is named (media_analysis_requested), so with no api_name the extraction call is checked at the call; its refusal is re-raised (extraction/strategies/llm.py, call_llm_provider). The individual method then answers 402, discarding the pages already scraped in that request; the url_level and recursive_scraping methods run their pages as scraping queue jobs, where a refused page becomes a failed page carrying the 402 text (TASK-13864 Task 4). Found while planning TASK-13864." \
  --ac "An over-quota user's /media/ingest-web-content request with perform_analysis and no api_name gets the 402 before any page is fetched, or keeps its pages without LLM extraction, with a test" \
  --ac "Usage_Quotas.md says which"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Connectors webhook GET and POST share one OpenAPI operationId" --id "$OPID" -l backend,tests \
  -d "connectors.py declares /providers/{provider}/webhook with api_route(methods=[\"GET\", \"POST\"]), so both methods get the operationId provider_webhook_callback_api_v1_connectors_providers__provider__webhook_post. The connectors router is non-stable (default_stable=False) and is left out of the default schema, but an app that a test builds can mount it into the cached schema. On origin/dev 2e6ceab9fb, Watchlists/test_watchlists_api.py::test_output_summaries_return_a_quota_refusal_as_402 followed by Services/test_openapi_contracts.py::test_openapi_operation_ids_are_present_and_unique in one process gives 1 failed, 1 passed: Duplicate OpenAPI operationIds. A combined -n 4 run can order them that way (TASK-13864 plan review). Found while planning TASK-13864." \
  --ac "The connectors webhook's GET and POST have distinct operationIds (separate routes or explicit operation_id), with a test" \
  --ac "test_openapi_operation_ids_are_present_and_unique passes after test_output_summaries_return_a_quota_refusal_as_402 in one process"
```

Measured on 2026-10-20 without the paging loop: `13870`.

- [ ] **Step 6: close TASK-13864**

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13864 \
  --append-notes "RED on origin/dev (plan measured on 2e6ceab9fb): 10 failed, 6 passed over 14 new tests and the 2 existing Prompt Studio ones. Jobs 3 failed (document batch: Processing error: LLMQuotaExceededError, media_id None; audio: Failed to call processor: LLMQuotaExceededError; admitted job checked at the call, [1] != [None]); the admission-scope test fails only once the job-start check exists without its scope (assert True is False); reused transcript failed (KeyError 'analysis_error' on dev, 'video_lite' after Task 1); web endpoints and watchlist outputs 3 failed (200 where 402 is expected; 3 controls passed); scraping queue 2 failed (job ran as user 7 prechecked; worker died on a refusal); Prompt Studio 1 failed (KeyError: 'test_run_ids'). GREEN: <counts from Tasks 1-5>. Neighbourhood suites: <counts>; known dev failures: test_transcription_failure_handling under -n 4, and the OpenAPI operationId test after a watchlists test ($OPID). Test-quality triage: exit 0; ratchet: 1 passed; OpenAPI gate: OK." \
  --append-notes "Decisions taken: <the owner's choices for Decisions 1 and 2>. Batch: each media ingest job checks once when it starts; no batch-wide flag. Follow-ups: $EXTRACT (LLM extraction during web content ingest without a named provider), $OPID (connectors webhook duplicate operationId)." \
  --append-notes "Known gaps (plan Review Focus): a refused scraping queue job outside the app's routes becomes a failed page, not a 402; an admitted job can pass the limit by its own analysis, as an admitted request can; enrichment run as a request background task inherits the request's admission. Bandit on touched app files: <result>." \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A media ingest job that asks for analysis checks the owner's LLM quota once when it starts: refused, it stores the item without analysis and its result says why (status Warning, analysis_error.code llm_quota_exceeded), and a reused transcript still records video_lite.summary_error; admitted, it runs to completion like an admitted request, and the admission ends with the job. /media/ingest-web-content with analysis, /media/process-web-scraping with summaries and POST /watchlists/outputs with summarize refuse an over-quota user with the 402 before any page is fetched or output rendered. Scraping queue jobs run in their submitter's context; a refusal fails only that job. A Prompt Studio evaluation refused part-way keeps test_run_ids. Usage_Quotas.md, the design doc and the media ingest jobs API doc say what each path keeps." \
  -s Done
git add -A backlog/tasks
git commit -m "docs(backlog): close TASK-13864; file follow-ups $EXTRACT and $OPID" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: push and open the PR**

```bash
git push -u origin fix/keep-artifact-on-llm-quota-refusal
gh pr create --base dev --title "Add-on LLM work keeps the user's artifact on a quota refusal (TASK-13864)" --body-file <body>
```

The PR body has:
- the per-path table from "What was measured", with the Decisions taken;
- the batch answer and the scraping queue fix;
- the known gaps and the two known dev test failures;
- a "## Change summary" section, written by the owner or covered by the waiver they confirm (Global Constraints, ADR-004);
- the PR attribution line at the end.

Then run `/superpowers:requesting-code-review` in a fresh subagent and address its findings before merge. Merge per `gh variable get MERGE_QUEUE` and CLAUDE.md "Merging into dev".

---

## Out of scope

- **The processors' own analysis steps** (`Plaintext_Files.py`, `PDF_Processing_Lib.py`, `Book_Processing_Lib.py`, `Email_Processing_Lib.py`, `Audio_Files.py`, `Video_DL_Ingestion_Lib.py`). Under Decision 1's recommendation they never see a refusal from a media ingest job, and `/media/add` and `/media/process-*` already admit them. They still let a refusal escape with their extracted text, if anything ever calls them unadmitted.
- **LLM extraction during `/media/ingest-web-content` without a named provider** (Review Focus 1): filed in Task 6 Step 5.
- **The duplicate connectors operationId and its test order dependence:** filed in Task 6 Step 5.
- **`test_transcription_failure_handling` under `-n 4`:** fails on unchanged dev too; not filed here.
- **The workflow `media_ingest` step** (`Workflows/adapters/media/ingest.py`). A refused workflow step fails the step and the run (TASK-13638 Decision 1).
- **`Summarization_General_Lib.py`, the audio jobs worker and its docs:** TASK-13865.
