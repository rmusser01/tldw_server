# Output Summaries and Audio Analysis Reach the Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two LLM calls that never reached a provider. Watchlist output summaries (`outputs_service._summarize_single_article`, `_summarize_text_block`) passed `analyze` keyword arguments it does not take, so every summary fell back to the first 300 characters (TASK-13636). The audio jobs `audio_analyze` stage passed the transcript as the provider name, so every analysis job failed and ended quarantined (TASK-13637). Once the calls work, an LLM quota refusal must stop the work and be reported, not be recorded as a completed summary or retried.

**Architecture:** Two small fixes, each in its own task. Outputs: the helpers fill the prompt template once (`_fill_prompt`) and send it as `analyze`'s literal input, with a short system prompt of their own (`_OUTPUTS_SYSTEM_PROMPT`) instead of `analyze`'s bulleted-notes default; an `"Error:"` result counts as a failure; `LLMQuotaExceededError` is re-raised at the five places that swallowed it (the two helpers, the per-group summary loop, topic classification, and the `POST /watchlists/outputs` handler), so the request returns the 402 and output enrichment records `enrichment_status: failed`. Audio: the stage calls `analyze(provider, transcript, None, input_is_literal_text=True, raise_on_error=True)`, an `"Error:"` result fails the stage, and the failure path passes the exception's `retryable` flag and `failure_code` to `fail_job`.

**Tech Stack:** Python 3.12, FastAPI, the Jobs manager (SQLite in tests), pytest.

**Spec:** the two backlog tasks, `backlog/tasks/task-13636 - Output-summaries-pass-arguments-analyze-does-not-accept.md` and `backlog/tasks/task-13637 - Audio-analyze-job-stage-calls-analyze-with-the-wrong-arguments.md`, and for the refusal handling `Docs/Design/2026-10-09-llm-usage-accounting-design.md` §4b ("a refusal is not a provider outage") with `Docs/Operations/Usage_Quotas.md`.

## Global Constraints

- **`analyze`'s contract** (`core/LLM_Calls/Summarization_General_Lib.py:549`): `analyze(api_name, input_data, custom_prompt_arg, ..., model_override=None, input_is_literal_text=False, *, raise_on_error=False)`. It sends `f"{text}\n\n\n\n{custom_prompt_arg}"` as the user message and returns a string starting with `"Error:"` on most failures (with `raise_on_error=True` provider failures raise `SummaryProviderError` instead; some early returns are strings regardless). It does not catch `LLMQuotaExceededError`.
- **`LLMQuotaExceededError`** (`core/exceptions.py`) is an `HTTPException` 402 with `retryable = False` and `failure_code = "llm_quota_exceeded"`. It must reach the caller: a request returns the 402; background work records a failure. It is never a completed result and never retried.
- **Behaviour that stays:** a provider failure still falls back (the article's first 300 characters; an empty block summary; tag grouping); a per-group provider failure still leaves the other groups; an ordinary audio stage error is still retried.
- **The audio payload's `model` is the transcription model** (`whisper-1`); it is never passed to `analyze`. Jobs carry no prompt, so the stage uses `analyze`'s default system prompt (bulleted notes of the transcript) and the provider's configured model.
- **The output helpers send their own system prompt.** `analyze`'s default system prompt asks for "comprehensive bulleted notes", which contradicts "2-3 concise sentences" and the topic classifier's JSON array; the helpers pass `system_message=_OUTPUTS_SYSTEM_PROMPT` ("follow the instructions in the message exactly").
- **Every change was prototyped** in a scratch checkout of `origin/dev` and measured on `origin/dev` 9e43c298e6; it is given here as exact patches (the Appendix; the same patches as files in this plan's SDD workspace). Apply each with `git apply --check` first. If a patch does not apply, or a test does not behave as a step says, STOP and report it with the output. Do not hand-edit around a failed patch.
- **Test-first:** each code task applies its test patch, runs it and sees the named failures, then applies its source patch.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit ids, `git add -A backlog/tasks`.
- **Commits** end with exactly `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `git stash`. Never `--no-verify`.
- **Python:** `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (written `$PY`). Tests run with `TLDW_TEST_NO_DOCKER=1 RUN_JOBS=1 RUN_EVALUATIONS=1`.
- **Known on `dev`:** `tests/AudioJobs/test_audio_jobs_admin.py` skips 3 tests ("endpoints not mounted") depending on test order; under `-n 4` across many directories, order-dependent 404s and 401s on `/media` routes. Re-run any other failure alone, and at `origin/dev`, before reporting it.

## Review Focus

1. **A custom summary prompt without `{text}`** (`summarize_prompt`, `BriefingSummaryConfig.prompt` / `per_group_prompt`; the placeholder is documented nowhere). Expected: the text is appended after the prompt, not dropped. (Task 1, `test_summarize_single_article_sends_the_text_with_a_custom_prompt_that_has_no_placeholder`)
2. **An article or title containing `{` or `}`** (code, JSON). Expected: the summary is produced; nothing raises `KeyError`. (Task 1, `test_generate_group_summaries_sends_text_with_braces_once`)
3. **A user over an LLM quota who asks `POST /watchlists/outputs` for summaries.** Expected: HTTP 402 with the `limit_exceeded` body, no output created with fallback summaries. (Task 1, `test_output_summaries_return_a_quota_refusal_as_402`)
4. **Scheduled output enrichment for a user over a quota.** Expected: the output's metadata records `enrichment_status: failed` with the 402 text, not `completed` with empty summaries or tag grouping. (Task 1, the two `test_enrich_output_records_a_refused_*` tests)
5. **An audio analysis job for a user over a quota, and one whose provider fails.** Expected: the refused job fails at once (`retry_count` 0, `error_code` `llm_quota_exceeded`); the provider failure is retried. (Task 2)

## File Structure

| File | Change |
|---|---|
| `tldw_Server_API/app/services/outputs_service.py` | `_fill_prompt`; the two helpers call `analyze` correctly, treat `"Error:"` as failure and re-raise a refusal; `generate_group_summaries` and `classify_items_by_topic` pass the raw template and re-raise a refusal. |
| `tldw_Server_API/app/api/v1/endpoints/watchlists.py` | `create_output` re-raises `LLMQuotaExceededError` before its broad `_WATCHLISTS_NONCRITICAL_EXCEPTIONS` (which includes `HTTPException`). |
| `tldw_Server_API/app/services/audio_jobs_worker.py` | The `audio_analyze` call; an `"Error:"` result fails the stage; `fail_job` gets `retryable` and `error_code` from the exception. |
| `tldw_Server_API/tests/Services/test_outputs_service.py`, `tests/Watchlists/test_output_enrichment_scheduler.py`, `tests/Watchlists/test_watchlists_api.py` | 16 new tests (Task 1). |
| `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py` | 4 new tests (Task 2). |
| `tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py` | The fake `analyze` takes `(api_name, input_data, ...)`, not the old wrong order (Task 2). |
| `Docs/Operations/Usage_Quotas.md`, `Docs/Design/2026-10-09-llm-usage-accounting-design.md` | Where the refusal now surfaces (Task 3). Neither is mirrored under `Docs/Published`. |
| `Docs/API-related/Watchlists_API.md` and its `Docs/Published` mirror | One sentence: how `{text}` works in the summary prompts (Task 3). |

## Rulings made while planning

- **`_fill_prompt` replaces `str.format`** (a widening of TASK-13636, kept). The `{text}` placeholder is documented nowhere; with `.format` a prompt without it dropped the article, any other brace raised outside the `try`, and `generate_group_summaries` and `classify_items_by_topic` formatted twice, so a `{` in any title emptied the summary. `_fill_prompt` puts the text at `{text}`, or after the template when it has none, and every caller passes the raw template so the text is filled exactly once.
- **The prompt is sent once, as literal input** (`analyze(api_name, prompt, None, ..., input_is_literal_text=True)`): renaming the keywords alone would have sent the article twice (once in the prompt, once as input). Literal, so an article that looks like JSON is not parsed.
- **`{text}` is documented in `Docs/API-related/Watchlists_API.md`, not in the schema descriptions.** A description change would change the OpenAPI fingerprint and the generated frontend types; with `_fill_prompt` a prompt works with or without the placeholder. Only the first `{text}` is filled, so a prompt that repeats it does not send the text twice. Doubled braces (`{{`), which only `str.format` read as one, are now literal; no custom prompt ever reached a model, so none relied on it.
- **The output helpers pass `system_message=_OUTPUTS_SYSTEM_PROMPT`** (found by the plan review): without it every summary and the topic classification run under `analyze`'s bulleted-notes system prompt, and the classifier, which needs a JSON array, would often fall back to tag grouping.
- **Five swallow points re-raise the refusal**, not two (the AC named the two helpers): the endpoint's broad tuple contains `HTTPException`; `generate_group_summaries`' per-group `except` recorded `completed`; `classify_items_by_topic` fell back to tag grouping. `enrich_output` needs no change: its final `except` already records `enrichment_status: failed`. It is a Scheduler task (not a WorkerSDK job) and is not retried.
- **A refusal mid-way through `POST /watchlists/outputs` returns the 402 and discards that request's summaries,** as the web scraping endpoints do (operator page).
- **The audio stage passes no model** (the payload's `model` is the STT model; the test job carries `model: whisper-1` so a forwarded model would fail the test) and **no prompt** (jobs carry none; `analyze`'s default system prompt, bulleted notes of the transcript, is "a prompt" for TASK-13637's AC#1 and a fair analysis). `error_class` is not passed to `fail_job`: only `retryable` and `error_code` are needed.
- **The in-process enrichment fallback** (`fallback_submitter=background_tasks.add_task` in `endpoints/watchlists.py`, used when the Scheduler is unavailable) is unchanged: it runs `enrich_output` without `llm_call_root`, with the user taken from the request's content scope (which `BackgroundTasks` keep, design §4a), and `POST /watchlists/outputs` carries no `prechecked` mark, so a refusal there is checked at the call and recorded as `failed`, as on the Scheduler path.
- **A model reply that starts with `Error:`** is treated as a failure (it is `analyze`'s error convention); noted on TASK-13642.
- **Follow-ups filed in Task 3**, not fixed here: the audio stage retries configuration errors (`SummaryProviderError` has no `retryable`); the TTS and audio workers may take each other's jobs from the shared `audio`/`default` queue (not verified at runtime); the watchlist summary cache is never persisted (`update_item_metadata` exists on no DB class) and the 300-character fallback is shown as an LLM summary.

---

### Task 1: Watchlist output summaries reach the model, and a refusal is reported (TASK-13636)

**Files:**
- Modify: `tldw_Server_API/app/services/outputs_service.py`, `tldw_Server_API/app/api/v1/endpoints/watchlists.py` (Appendix A.2, `outputs-src.patch`)
- Test: `tldw_Server_API/tests/Services/test_outputs_service.py`, `tldw_Server_API/tests/Watchlists/test_output_enrichment_scheduler.py`, `tldw_Server_API/tests/Watchlists/test_watchlists_api.py` (Appendix A.1, `outputs-tests.patch`)

**Interfaces:**
- Produces: `outputs_service._fill_prompt(template: str, text: str) -> str` (fills the first `{text}`, or appends); `outputs_service._OUTPUTS_SYSTEM_PROMPT: str`; `_summarize_single_article` / `_summarize_text_block` keep their signatures and now raise `LLMQuotaExceededError`; so do `generate_group_summaries`, `generate_briefing_summary`, `classify_items_by_topic` and `summarize_items_for_output`.

- [ ] **Step 1: Apply the test patch**

The patches are in this plan's SDD workspace (`.superpowers/sdd/2026-10-14-outputs-and-audio-analyze-calls/`). From the worktree root:

```bash
git apply --check .superpowers/sdd/2026-10-14-outputs-and-audio-analyze-calls/outputs-tests.patch
git apply .superpowers/sdd/2026-10-14-outputs-and-audio-analyze-calls/outputs-tests.patch
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 RUN_JOBS=1 RUN_EVALUATIONS=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Services/test_outputs_service.py tldw_Server_API/tests/Watchlists/test_output_enrichment_scheduler.py tldw_Server_API/tests/Watchlists/test_watchlists_api.py`
Expected: `14 failed, 47 passed`, exactly these:

- `test_outputs_service.py::test_summarize_single_article_returns_the_model_summary_and_sends_the_text_once`
- `test_outputs_service.py::test_summarize_text_block_returns_the_model_summary_and_sends_the_text_once`
- `test_outputs_service.py::test_summarize_single_article_sends_the_text_with_a_custom_prompt_that_has_no_placeholder`
- `test_outputs_service.py::test_summarize_single_article_lets_a_quota_refusal_through`
- `test_outputs_service.py::test_summarize_text_block_lets_a_quota_refusal_through`
- `test_outputs_service.py::test_summarize_items_for_output_lets_a_quota_refusal_through`
- `test_outputs_service.py::test_generate_group_summaries_sends_text_with_braces_once`
- `test_outputs_service.py::test_generate_group_summaries_lets_a_quota_refusal_through`
- `test_outputs_service.py::test_generate_briefing_summary_lets_a_quota_refusal_through`
- `test_outputs_service.py::test_classify_items_by_topic_lets_a_quota_refusal_through`
- `test_output_enrichment_scheduler.py::test_enrich_output_stores_the_model_summaries`
- `test_output_enrichment_scheduler.py::test_enrich_output_records_a_refused_summary_as_failed`
- `test_output_enrichment_scheduler.py::test_enrich_output_records_a_refused_topic_grouping_as_failed`
- `test_watchlists_api.py::test_output_summaries_return_a_quota_refusal_as_402`

The two `Error:`-fallback guards (`test_summarize_single_article_falls_back_when_analyze_returns_an_error`, `test_summarize_text_block_returns_nothing_when_analyze_returns_an_error`) pass now, because the broken call falls back the same way; they fail if the `Error:` check is dropped after the fix. Every failure's log shows `analyze() got an unexpected keyword argument 'custom_prompt'` or the swallowed refusal.

- [ ] **Step 3: Apply the source patch**

```bash
git apply --check .superpowers/sdd/2026-10-14-outputs-and-audio-analyze-calls/outputs-src.patch
git apply .superpowers/sdd/2026-10-14-outputs-and-audio-analyze-calls/outputs-src.patch
```

- [ ] **Step 4: Run the tests to see them pass**

Run the Step 2 command.
Expected: `61 passed`.

- [ ] **Step 5: Lint**

Compare ruff finding counts with `origin/dev` for each changed file (`$PY -m ruff check --output-format concise <file> | grep -c ':[0-9]*:[0-9]*:'` against `git show origin/dev:<file> | $PY -m ruff check --output-format concise --stdin-filename <file> - | grep -c ':[0-9]*:[0-9]*:'`). On 9e43c298e6: `outputs_service.py` 1, `endpoints/watchlists.py` 6, `test_outputs_service.py` 0, `test_output_enrichment_scheduler.py` 1, `test_watchlists_api.py` 5. Counts must not grow.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/services/outputs_service.py tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/tests/Services/test_outputs_service.py tldw_Server_API/tests/Watchlists/test_output_enrichment_scheduler.py tldw_Server_API/tests/Watchlists/test_watchlists_api.py
git status --short
git commit -m "fix(outputs): watchlist summaries reach the model and a quota refusal is reported, not stored as a summary (TASK-13636)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

`git status --short` before the commit must show only these five files staged.

---

### Task 2: The audio analysis stage reaches the model, and a refusal fails the job at once (TASK-13637)

**Files:**
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py` (Appendix A.4, `audio-src.patch`)
- Test: `tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py`, `tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py` (Appendix A.3, `audio-tests.patch`)

**Interfaces:**
- Produces: the `audio_analyze` stage stores `analyze`'s string in `payload["analysis"]` and hands over to `audio_store`; a failed stage calls `JobManager.fail_job(job_id, *, error, retryable=bool(getattr(exc, "retryable", True)), error_code=getattr(exc, "failure_code", None), ...)`.

- [ ] **Step 1: Apply the test patch**

```bash
git apply --check .superpowers/sdd/2026-10-14-outputs-and-audio-analyze-calls/audio-tests.patch
git apply .superpowers/sdd/2026-10-14-outputs-and-audio-analyze-calls/audio-tests.patch
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 RUN_JOBS=1 RUN_EVALUATIONS=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py`
Expected: `4 failed, 12 passed, 1 skipped`, exactly these:

- `test_audio_worker_smoke.py::test_audio_analyze_stores_the_model_analysis`
- `test_audio_worker_smoke.py::test_audio_analyze_fails_the_stage_on_an_error_result`
- `test_audio_worker_smoke.py::test_audio_analyze_quota_refusal_fails_the_job_without_retry`
- `test_audio_worker_smoke.py::test_audio_analyze_provider_failure_still_retries`

(The `test_llm_roots_non_sdk.py` change only renames the fake's parameters; it passes before and after.)

- [ ] **Step 3: Apply the source patch**

```bash
git apply --check .superpowers/sdd/2026-10-14-outputs-and-audio-analyze-calls/audio-src.patch
git apply .superpowers/sdd/2026-10-14-outputs-and-audio-analyze-calls/audio-src.patch
```

- [ ] **Step 4: Run the tests to see them pass**

Run the Step 2 command.
Expected: `16 passed, 1 skipped`.

- [ ] **Step 5: Lint, bandit and the wider suites**

Ruff counts as in Task 1 Step 5 for `audio_jobs_worker.py` and the two test files. On 9e43c298e6: `audio_jobs_worker.py` 0, `test_audio_worker_smoke.py` 2, `test_llm_roots_non_sdk.py` 0.

Run: `uvx bandit -ll tldw_Server_API/app/services/outputs_service.py tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/services/audio_jobs_worker.py`
Expected: `No issues identified.`

Run (every test file that imports a changed module, plus the `analyze` adapter tests): `TLDW_TEST_NO_DOCKER=1 RUN_JOBS=1 RUN_EVALUATIONS=1 $PY -m pytest -q -p no:cacheprovider -n 4 tldw_Server_API/tests/Services tldw_Server_API/tests/Watchlists tldw_Server_API/tests/AudioJobs tldw_Server_API/tests/Usage tldw_Server_API/tests/Audio/test_audio_jobs_worker_helper.py tldw_Server_API/tests/Audio/test_tts_generation_default_policy.py tldw_Server_API/tests/AuthNZ/test_websocket_cookie_route_contract.py tldw_Server_API/tests/Collections/test_items_and_outputs_api.py tldw_Server_API/tests/DB_Management/test_output_storage_normalization.py tldw_Server_API/tests/Jobs/test_worker_owner_fallback_fails_closed.py tldw_Server_API/tests/lint/test_private_coercion_ratchet.py tldw_Server_API/tests/Web_Scraping/test_phase4_selectors.py tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py`
Expected: 0 failed; between 7 and 10 skipped; the only xpass is `test_full_pipeline_integration.py::test_opml_reimport_skips_duplicates` (on 9e43c298e6: `2747 passed, 10 skipped, 1 xpassed`, or `2750 passed, 7 skipped` when the 3 order-dependent `test_audio_jobs_admin.py` tests run; the other skips are `RUN_E2E` unset, live ingestion, the heavy audio smoke test and no Postgres `DATABASE_URL`). Classify any failure as Global Constraints says.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py
git status --short
git commit -m "fix(audio): the analysis stage reaches the model and a quota refusal fails the job without retry (TASK-13637)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

`git status --short` before the commit must show only these three files staged.

---

### Task 3: The operator page, the spec, and the backlog

**Files:**
- Modify: `Docs/Operations/Usage_Quotas.md`, `Docs/Design/2026-10-09-llm-usage-accounting-design.md`
- Modify: `backlog/tasks/task-13636*.md`, `backlog/tasks/task-13637*.md`; create three follow-up tasks (backlog-py only)

- [ ] **Step 1: The operator page**

In `Docs/Operations/Usage_Quotas.md`, use the Edit tool for each replacement (each `old` appears once):

1. Old: ``returns the same 402: `/translate`, `/media/ingest-web-content` with analysis, `/media/process-web-scraping` with summarization, `/media/{id}/insights` and the manuscript analysis endpoints among them. The web scraping endpoints discard that request's results.``
   New: ``returns the same 402: `/translate`, `/media/ingest-web-content` with analysis, `/media/process-web-scraping` with summarization, `POST /watchlists/outputs` with `summarize`, `/media/{id}/insights` and the manuscript analysis endpoints among them. The web scraping endpoints and `POST /watchlists/outputs` discard that request's results.``
2. Old: ``A WorkerSDK job records the error code `llm_quota_exceeded`; other workers record their own.``
   New: ``A WorkerSDK job and an audio analysis job (`/audio/jobs/submit` with `perform_analysis`) record the error code `llm_quota_exceeded`; other workers record their own.``
3. After the line that begins `  - **A workflow:**`, insert this line (same two-space indent):
   ``  - **Watchlist output enrichment** (briefing and per-group summaries, topic grouping) stops at the refusal: the output's metadata gets `enrichment_status: failed` and `enrichment_error: 402: {...}`, and the task is not retried.``

In `Docs/API-related/Watchlists_API.md`, after the line that begins ``- `template_name` and `template_version`: render a stored Watchlists template.``, insert:
``- `summarize` and `summarize_prompt`: summarize each included item with the LLM before rendering. In `summarize_prompt`, and in `briefing_summary.prompt` and `briefing_summary.per_group_prompt`, `{text}` marks where the text goes; without it the text is appended after the prompt. Other braces are literal. A user over an LLM quota gets a 402 (see `Docs/Operations/Usage_Quotas.md`).``

Then refresh the published copies: `bash Helper_Scripts/refresh_docs_published.sh`, and check `git status --short -- Docs/Published`. Expected (on 9e43c298e6): only `Docs/Published/API-related/Watchlists_API.md` changes. If the refresh also rewrites other stale files, carry them and list them in your report.

- [ ] **Step 2: The spec**

In `Docs/Design/2026-10-09-llm-usage-accounting-design.md`, with the Edit tool:
Old: ``(`/media/ingest-jobs`, evaluation recipe runs; audio jobs reach no provider until TASK-13637)``
New: ``(`/media/ingest-jobs`, evaluation recipe runs, audio analysis jobs)``

- [ ] **Step 3: Close the two tasks**

First check the follow-up ids are free: `ls backlog/tasks | grep -E '^task-(13640|13641|13642) '` must print nothing. If one is taken, use the next free id in this step and in Step 4, and say so in your report. Then, from the worktree root, one command at a time:

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13636 --append-notes "Fixed: _summarize_single_article and _summarize_text_block call analyze(api_name, prompt, None, model_override=..., input_is_literal_text=True), so the text is sent once; _fill_prompt fills the template once (at {text}, or appended when absent), which also fixes custom prompts without {text} and braces in titles, since generate_group_summaries and classify_items_by_topic formatted twice. An Error: result falls back as before. LLMQuotaExceededError is re-raised by the two helpers, the per-group summary loop, classify_items_by_topic and POST /watchlists/outputs (whose broad tuple includes HTTPException), so the endpoint returns the 402 and enrich_output records enrichment_status failed." --check-ac 1 --check-ac 2 --check-ac 3 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "Watchlist output summaries never reached a provider: the helpers passed analyze keyword arguments it does not take, so every summary fell back to the first 300 characters. They now call analyze correctly, with the prompt filled once and a system prompt that follows it, an Error: result still falls back, and an LLM quota refusal is reported (402 on POST /watchlists/outputs; enrichment_status failed for scheduled enrichment) instead of being stored as a summary. Follow-ups: TASK-13642 (summary cache never persisted; fallback shown as an LLM summary)." --status Done
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13637 --append-notes "Correction to the description: at base the TypeError stage ended quarantined (failure streak), not retried until its attempts ran out. Fixed: the stage calls analyze(api_name or 'openai', text, None, input_is_literal_text=True, raise_on_error=True) (the payload's model is the transcription model and is not passed; jobs carry no prompt, so analyze's default system prompt, bulleted notes, applies); an Error: result fails the stage; fail_job gets retryable and error_code from the exception, so a quota refusal fails the job at once with llm_quota_exceeded and an ordinary provider error is retried." --check-ac 1 --check-ac 2 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "The audio analyze stage passed the transcript as the provider name, so every analysis job failed and ended quarantined. It now calls analyze with the provider, the transcript and analyze's default system prompt (bulleted notes), stores the analysis, fails the stage on an Error: result, and passes the exception's retryable flag and failure code to fail_job: a quota refusal fails the job at once with llm_quota_exceeded. Follow-ups: TASK-13640 (configuration errors are retried), TASK-13641 (TTS and audio workers share a queue)." --status Done
```

(Bandit, DoD #4, ran in Task 2 Step 5.)

- [ ] **Step 4: File the follow-ups**

Using the ids checked in Step 3:

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Audio analysis stage retries configuration errors" --id TASK-13640 --status "To Do" --labels audio,jobs,bug --description "With raise_on_error=True the audio_analyze stage (app/services/audio_jobs_worker.py) gets SummaryProviderError for missing_model, authentication, invalid_provider and configuration failures, which carry no retryable flag, so they are retried until the job is quarantined, the same as rate_limit. An empty transcript (analyze returns 'Error: Could not extract text content.') is retried too. Found while fixing TASK-13637." --ac "Configuration and authentication failures, and an empty transcript, fail the stage without retry; rate limits and transient provider failures are still retried"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "TTS and audio job workers may take each other's jobs" --id TASK-13641 --status "To Do" --labels audio,tts,jobs,bug --description "core/TTS/tts_jobs_worker.py (WorkerSDK, no job_type filter) and app/services/audio_jobs_worker.py both acquire from domain audio, queue default with no job_type filter. If both run, the TTS worker would fail audio_* stage jobs as unsupported and the audio worker would fail tts_longform jobs as an unknown job_type. Not verified at runtime: the TTS worker appears to run only standalone (python -m). Found while fixing TASK-13637." --ac "Verify whether both workers can run against the same queue; if so, each acquires only its own job types (separate queues or a job_type filter), with a test"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Watchlist summaries: the cache is never written and the fallback looks like an LLM summary" --id TASK-13642 --status "To Do" --labels outputs,watchlists,bug --description "summarize_items_for_output (app/services/outputs_service.py) persists cached_summary only through update_item_metadata, which no DB class defines, so nothing is cached and every output re-summarizes. When a summary fails, the first 300 characters of the article are stored in item.llm_summary and has_llm_summaries is set, so the fallback renders as if the model wrote it. A model reply that itself starts with 'Error:' is also treated as a failure (analyze's error convention). Found while fixing TASK-13636." --ac "Summaries are cached by content hash and reused, or the dead cache code is removed" --ac "A fallback excerpt is not presented as an LLM summary"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task normalize --check backlog/tasks/task-13636*.md backlog/tasks/task-13637*.md backlog/tasks/task-13640*.md backlog/tasks/task-13641*.md backlog/tasks/task-13642*.md
```

Expected: each command succeeds; `normalize --check` lists nothing.

- [ ] **Step 5: Run the docs tests**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Docs`
Expected: all pass (on 9e43c298e6: `212 passed`).

- [ ] **Step 6: Commit**

```bash
git add Docs/Operations/Usage_Quotas.md Docs/Design/2026-10-09-llm-usage-accounting-design.md Docs/API-related/Watchlists_API.md
git add -A Docs/Published
git add -A backlog/tasks
git status --short
git commit -m "docs(usage): watchlist outputs and audio analysis jobs report a quota refusal; close TASK-13636/13637, file follow-ups

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

`git status --short` before the commit must show the three docs, the published Watchlists copy (and any other files the refresh rewrote), the two closed task files and the three new task files.

---

## Appendix: the patches

Each patch was made against `origin/dev` 9e43c298e6 and measured on 9e43c298e6. They are saved, byte for byte, as files in this plan's SDD workspace; apply the files, not a hand copy of these blocks.

### A.1 `outputs-tests.patch`

```diff
diff --git a/tldw_Server_API/tests/Services/test_outputs_service.py b/tldw_Server_API/tests/Services/test_outputs_service.py
index 59e340cbb1..5e280711af 100644
--- a/tldw_Server_API/tests/Services/test_outputs_service.py
+++ b/tldw_Server_API/tests/Services/test_outputs_service.py
@@ -5,6 +5,8 @@ from types import SimpleNamespace
 
 import pytest
 
+from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
+from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summarization
 from tldw_Server_API.app.services import outputs_service
 
 
@@ -154,3 +156,128 @@ async def test_resolve_tts_generation_defaults_omnivoice_blank_voice_falls_back_
     assert resolved_model == "omnivoice"
     assert resolved_voice == outputs_service.DEFAULT_OMNIVOICE_TTS_VOICE
     assert resolved_speed is None
+
+
+# --- LLM summaries through the real analyze() (TASK-13636) ---
+
+
+class _Adapter:
+    """The provider adapter below analyze(): records each request and replies or raises."""
+
+    def __init__(self) -> None:
+        self.reply = "Model summary."
+        self.error: BaseException | None = None
+        self.requests: list[dict] = []
+
+    def chat(self, request, timeout=None):
+        self.requests.append(request)
+        if self.error is not None:
+            raise self.error
+        return {"choices": [{"message": {"content": self.reply}}]}
+
+    def sent(self) -> str:
+        """The user message of the only request."""
+        assert len(self.requests) == 1
+        return self.requests[0]["messages"][0]["content"]
+
+
+@pytest.fixture
+def adapter(monkeypatch: pytest.MonkeyPatch) -> _Adapter:
+    fake = _Adapter()
+    monkeypatch.setattr(summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _provider: fake))
+    return fake
+
+
+def _refusal() -> LLMQuotaExceededError:
+    return LLMQuotaExceededError(
+        category="llm_tokens_month", current=12, limit=10, message="LLM token quota exceeded", retry_after=60
+    )
+
+
+_LLM = {"api_name": "openai", "model_override": "test-model"}
+
+
+@pytest.mark.unit
+def test_summarize_single_article_returns_the_model_summary_and_sends_the_text_once(adapter: _Adapter) -> None:
+    assert outputs_service._summarize_single_article("Cedar article body.", **_LLM) == "Model summary."
+    assert adapter.sent().count("Cedar article body.") == 1
+    assert adapter.requests[0]["model"] == "test-model"
+    assert adapter.requests[0]["system_message"] == outputs_service._OUTPUTS_SYSTEM_PROMPT
+
+
+@pytest.mark.unit
+def test_summarize_text_block_returns_the_model_summary_and_sends_the_text_once(adapter: _Adapter) -> None:
+    assert outputs_service._summarize_text_block("- Cedar story", **_LLM) == "Model summary."
+    assert adapter.sent().count("- Cedar story") == 1
+    assert adapter.requests[0]["system_message"] == outputs_service._OUTPUTS_SYSTEM_PROMPT
+
+
+@pytest.mark.unit
+def test_summarize_single_article_sends_the_text_with_a_custom_prompt_that_has_no_placeholder(
+    adapter: _Adapter,
+) -> None:
+    outputs_service._summarize_single_article("Cedar article body.", custom_prompt="Summarize in French.", **_LLM)
+    assert "Summarize in French." in adapter.sent()
+    assert adapter.sent().count("Cedar article body.") == 1
+
+
+@pytest.mark.unit
+def test_summarize_single_article_falls_back_when_analyze_returns_an_error(adapter: _Adapter) -> None:
+    adapter.reply = ""  # analyze() returns "Error: Provider returned no usable analysis..."
+    assert outputs_service._summarize_single_article("Cedar article body.", **_LLM) == "Cedar article body."
+
+
+@pytest.mark.unit
+def test_summarize_text_block_returns_nothing_when_analyze_returns_an_error(adapter: _Adapter) -> None:
+    adapter.reply = ""
+    assert outputs_service._summarize_text_block("- Cedar story", **_LLM) == ""
+
+
+@pytest.mark.unit
+def test_summarize_single_article_lets_a_quota_refusal_through(adapter: _Adapter) -> None:
+    adapter.error = _refusal()
+    with pytest.raises(LLMQuotaExceededError):
+        outputs_service._summarize_single_article("Cedar article body.", **_LLM)
+
+
+@pytest.mark.unit
+def test_summarize_text_block_lets_a_quota_refusal_through(adapter: _Adapter) -> None:
+    adapter.error = _refusal()
+    with pytest.raises(LLMQuotaExceededError):
+        outputs_service._summarize_text_block("- Cedar story", **_LLM)
+
+
+@pytest.mark.unit
+async def test_summarize_items_for_output_lets_a_quota_refusal_through(adapter: _Adapter) -> None:
+    adapter.error = _refusal()
+    with pytest.raises(LLMQuotaExceededError):
+        await outputs_service.summarize_items_for_output([{"id": 1, "content": "Cedar article body."}], **_LLM)
+
+
+@pytest.mark.unit
+async def test_generate_group_summaries_sends_text_with_braces_once(adapter: _Adapter) -> None:
+    groups = [{"name": "Rust", "items": [{"title": "Rust {macros} and {}"}]}]
+    result = await outputs_service.generate_group_summaries(groups, **_LLM)
+    assert result[0]["summary"] == "Model summary."
+    assert adapter.sent().count("Rust {macros} and {}") == 1
+
+
+@pytest.mark.unit
+async def test_generate_group_summaries_lets_a_quota_refusal_through(adapter: _Adapter) -> None:
+    adapter.error = _refusal()
+    with pytest.raises(LLMQuotaExceededError):
+        await outputs_service.generate_group_summaries([{"name": "g", "items": [{"title": "Cedar"}]}], **_LLM)
+
+
+@pytest.mark.unit
+async def test_generate_briefing_summary_lets_a_quota_refusal_through(adapter: _Adapter) -> None:
+    adapter.error = _refusal()
+    with pytest.raises(LLMQuotaExceededError):
+        await outputs_service.generate_briefing_summary([{"title": "Cedar"}], **_LLM)
+
+
+@pytest.mark.unit
+async def test_classify_items_by_topic_lets_a_quota_refusal_through(adapter: _Adapter) -> None:
+    adapter.error = _refusal()
+    with pytest.raises(LLMQuotaExceededError):
+        await outputs_service.classify_items_by_topic([{"title": "Cedar"}], **_LLM)
diff --git a/tldw_Server_API/tests/Watchlists/test_output_enrichment_scheduler.py b/tldw_Server_API/tests/Watchlists/test_output_enrichment_scheduler.py
index 016657a684..f9b817dd55 100644
--- a/tldw_Server_API/tests/Watchlists/test_output_enrichment_scheduler.py
+++ b/tldw_Server_API/tests/Watchlists/test_output_enrichment_scheduler.py
@@ -1,10 +1,14 @@
 from __future__ import annotations
 
+import json
 from types import SimpleNamespace
 from typing import Any
 
 import pytest
 
+from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
+from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
+from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summarization
 from tldw_Server_API.app.core.Watchlists import output_enrichment_handler
 
 
@@ -87,3 +91,81 @@ async def test_schedule_output_enrichment_uses_fallback_when_scheduler_submit_fa
             },
         )
     ]
+
+
+# --- enrich_output through the real analyze() (TASK-13636) ---
+
+
+class _Adapter:
+    """The provider adapter below analyze(): replies, or raises the quota refusal."""
+
+    def __init__(self) -> None:
+        self.error: BaseException | None = None
+
+    def chat(self, request, timeout=None):
+        if self.error is not None:
+            raise self.error
+        return {"choices": [{"message": {"content": "Model summary."}}]}
+
+
+class _CollectionsDB:
+    """An output with two items; records each metadata write."""
+
+    def __init__(self) -> None:
+        self.saved: list[dict[str, Any]] = []
+
+    def get_output_artifact(self, output_id: int) -> SimpleNamespace:
+        return SimpleNamespace(metadata_json=json.dumps({"item_ids": [1, 2]}))
+
+    def get_content_item(self, item_id: int) -> SimpleNamespace:
+        return SimpleNamespace(id=item_id, title=f"Story {item_id}", url="", summary=f"About {item_id}", tags=["tech"])
+
+    def update_output_artifact_metadata(self, output_id: int, metadata_json: str) -> None:
+        self.saved.append(json.loads(metadata_json))
+
+
+@pytest.fixture
+def enrichment(monkeypatch: pytest.MonkeyPatch) -> tuple[_Adapter, _CollectionsDB]:
+    adapter, db = _Adapter(), _CollectionsDB()
+    monkeypatch.setattr(summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _provider: adapter))
+    monkeypatch.setattr(CollectionsDatabase, "for_user", staticmethod(lambda user_id: db))
+    return adapter, db
+
+
+_SUMMARY = {"llm_provider": "openai", "llm_model": "test-model", "per_group_summaries": True}
+
+
+async def test_enrich_output_stores_the_model_summaries(enrichment) -> None:
+    _adapter, db = enrichment
+    await output_enrichment_handler.enrich_output(
+        output_id=5, user_id=7, grouping_config={"group_by": "tag"}, summary_config=_SUMMARY
+    )
+    assert db.saved[-1]["enrichment_status"] == "completed"
+    assert db.saved[-1]["briefing_summary_text"] == "Model summary."
+    assert db.saved[-1]["enriched_groups"][0]["summary"] == "Model summary."
+
+
+async def test_enrich_output_records_a_refused_summary_as_failed(enrichment) -> None:
+    adapter, db = enrichment
+    adapter.error = LLMQuotaExceededError(
+        category="llm_tokens_month", current=12, limit=10, message="over", retry_after=60
+    )
+    await output_enrichment_handler.enrich_output(
+        output_id=5, user_id=7, grouping_config={"group_by": "tag"}, summary_config=_SUMMARY
+    )
+    assert db.saved[-1]["enrichment_status"] == "failed"
+    assert "limit_exceeded" in db.saved[-1]["enrichment_error"]
+
+
+async def test_enrich_output_records_a_refused_topic_grouping_as_failed(enrichment) -> None:
+    adapter, db = enrichment
+    adapter.error = LLMQuotaExceededError(
+        category="llm_tokens_month", current=12, limit=10, message="over", retry_after=60
+    )
+    await output_enrichment_handler.enrich_output(
+        output_id=5,
+        user_id=7,
+        grouping_config={"group_by": "topic", "topic_llm_provider": "openai", "topic_llm_model": "test-model"},
+    )
+    assert db.saved[-1]["enrichment_status"] == "failed"
+    assert "limit_exceeded" in db.saved[-1]["enrichment_error"]
diff --git a/tldw_Server_API/tests/Watchlists/test_watchlists_api.py b/tldw_Server_API/tests/Watchlists/test_watchlists_api.py
index b1d0ea3eb5..16b368d2be 100644
--- a/tldw_Server_API/tests/Watchlists/test_watchlists_api.py
+++ b/tldw_Server_API/tests/Watchlists/test_watchlists_api.py
@@ -1820,3 +1820,39 @@ def test_outputs_generate_audio_trigger_failure_marks_enqueue_failed_metadata(
     assert r.status_code == 200, r.text
     run_payload = r.json()
     assert run_payload.get("stats", {}).get("audio_briefing_task_id") is None
+
+
+def test_output_summaries_return_a_quota_refusal_as_402(client_with_user, monkeypatch):
+    """A refused summary call is not swallowed as a non-critical summarization failure (TASK-13636)."""
+    from types import SimpleNamespace
+
+    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
+    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summarization
+
+    c = client_with_user
+    monkeypatch.setenv("TEST_MODE", "1")
+
+    class _RefusingAdapter:
+        def chat(self, request, timeout=None):
+            raise LLMQuotaExceededError(
+                category="llm_tokens_month", current=12, limit=10, message="over", retry_after=60
+            )
+
+    monkeypatch.setattr(summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _p: _RefusingAdapter()))
+
+    r = c.post(
+        "/api/v1/watchlists/sources",
+        json={"name": "Quota Feed", "url": "https://example.com/quota.xml", "source_type": "rss", "tags": ["quota"]},
+    )
+    assert r.status_code == 200, r.text
+    r = c.post("/api/v1/watchlists/jobs", json={"name": "Quota Digest", "scope": {"tags": ["quota"]}})
+    assert r.status_code == 200, r.text
+    r = c.post(f"/api/v1/watchlists/jobs/{r.json()['id']}/run")
+    assert r.status_code == 200, r.text
+
+    r = c.post(
+        "/api/v1/watchlists/outputs",
+        json={"run_id": r.json()["id"], "summarize": True, "llm_provider": "openai", "llm_model": "test-model"},
+    )
+    assert r.status_code == 402, r.text
+    assert r.json()["detail"]["error"] == "limit_exceeded"
```

### A.2 `outputs-src.patch`

```diff
diff --git a/tldw_Server_API/app/api/v1/endpoints/watchlists.py b/tldw_Server_API/app/api/v1/endpoints/watchlists.py
index 6813063e86..25dc19a105 100644
--- a/tldw_Server_API/app/api/v1/endpoints/watchlists.py
+++ b/tldw_Server_API/app/api/v1/endpoints/watchlists.py
@@ -70,7 +70,7 @@ from tldw_Server_API.app.core.AuthNZ.websocket_session_auth import (
 from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
 from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
 from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
-from tldw_Server_API.app.core.exceptions import TemplateValidationError
+from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError, TemplateValidationError
 from tldw_Server_API.app.core.Personalization.companion_activity import (
     build_watchlist_source_bulk_import_activity,
     record_companion_activity_events_bulk,
@@ -7214,6 +7214,8 @@ async def create_output(
                 item_id = entry.get("id")
                 if item_id is not None and entry.get("llm_summary"):
                     llm_summaries[int(item_id)] = entry["llm_summary"]
+        except LLMQuotaExceededError:
+            raise  # a 402, not a non-critical summarization failure
         except _WATCHLISTS_NONCRITICAL_EXCEPTIONS as exc:
             logger.warning(f"Watchlists output summarization failed (non-critical): {exc}")
 
diff --git a/tldw_Server_API/app/services/outputs_service.py b/tldw_Server_API/app/services/outputs_service.py
index a7a3f2729f..da01f72f20 100644
--- a/tldw_Server_API/app/services/outputs_service.py
+++ b/tldw_Server_API/app/services/outputs_service.py
@@ -20,7 +20,7 @@ from tldw_Server_API.app.core.DB_Management.db_path_utils import (
     DatabasePaths,
     normalize_output_storage_filename,
 )
-from tldw_Server_API.app.core.exceptions import InvalidStoragePathError
+from tldw_Server_API.app.core.exceptions import InvalidStoragePathError, LLMQuotaExceededError
 from tldw_Server_API.app.core.Utils.path_utils import safe_join
 from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2
 
@@ -636,6 +636,10 @@ def delete_outputs_by_ids(cdb, user_id: int, ids: list[int]) -> int:
 
 _SUMMARIZE_MAX_CHARS = 12_000
 _SUMMARIZE_DEFAULT_PROMPT = "Summarize the following article in 2-3 concise sentences:\n\n{text}"
+# analyze's own default system prompt asks for bulleted notes; these prompts say what they want.
+_OUTPUTS_SYSTEM_PROMPT = (
+    "You are a concise assistant. Follow the instructions in the message exactly and return only what they ask for."
+)
 
 
 def _content_cache_key(content: str) -> str:
@@ -663,6 +667,13 @@ def _get_cached_summary(metadata_json: str | None, content_hash: str) -> str | N
     return None
 
 
+def _fill_prompt(template: str, text: str) -> str:
+    """Put ``text`` at the template's ``{text}``, or after the template when it has none."""
+    if "{text}" in template:
+        return template.replace("{text}", text, 1)
+    return f"{template}\n\n{text}"
+
+
 def _summarize_single_article(
     text: str,
     *,
@@ -670,21 +681,25 @@ def _summarize_single_article(
     model_override: str | None = None,
     custom_prompt: str | None = None,
 ) -> str:
-    """Synchronously call LLM to summarize a single article."""
+    """Synchronously call LLM to summarize a single article. A quota refusal propagates."""
     from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
 
     truncated = text[:_SUMMARIZE_MAX_CHARS]
-    prompt = (custom_prompt or _SUMMARIZE_DEFAULT_PROMPT).format(text=truncated)
+    prompt = _fill_prompt(custom_prompt or _SUMMARIZE_DEFAULT_PROMPT, truncated)
     try:
         result = analyze(
-            input_data=truncated,
-            custom_prompt=prompt,
-            api_name=api_name,
-            api_key=None,
-            model=model_override,
+            api_name,
+            prompt,
+            None,
+            system_message=_OUTPUTS_SYSTEM_PROMPT,
+            model_override=model_override,
+            input_is_literal_text=True,
         )
-        if isinstance(result, str) and result.strip():
-            return result.strip()
+        summary = result.strip() if isinstance(result, str) else ""
+        if summary and not summary.startswith("Error:"):
+            return summary
+    except LLMQuotaExceededError:
+        raise
     except Exception as exc:
         logger.warning(f"outputs_service: LLM summarize failed: {exc}")
     return text[:300]
@@ -874,21 +889,25 @@ def _summarize_text_block(
     custom_prompt: str | None = None,
     max_chars: int = 20_000,
 ) -> str:
-    """Shared helper to summarize a block of text via LLM. Synchronous."""
+    """Shared helper to summarize a block of text via LLM. Synchronous. A quota refusal propagates."""
     from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
 
     truncated = text[:max_chars]
-    prompt = (custom_prompt or _BRIEFING_SUMMARY_DEFAULT_PROMPT).format(text=truncated)
+    prompt = _fill_prompt(custom_prompt or _BRIEFING_SUMMARY_DEFAULT_PROMPT, truncated)
     try:
         result = analyze(
-            input_data=truncated,
-            custom_prompt=prompt,
-            api_name=api_name,
-            api_key=None,
-            model=model_override,
+            api_name,
+            prompt,
+            None,
+            system_message=_OUTPUTS_SYSTEM_PROMPT,
+            model_override=model_override,
+            input_is_literal_text=True,
         )
-        if isinstance(result, str) and result.strip():
-            return result.strip()
+        summary = result.strip() if isinstance(result, str) else ""
+        if summary and not summary.startswith("Error:"):
+            return summary
+    except LLMQuotaExceededError:
+        raise
     except Exception as exc:
         logger.warning(f"outputs_service: LLM summarize_text_block failed: {exc}")
     return ""
@@ -915,15 +934,19 @@ async def generate_group_summaries(
             if title:
                 texts.append(f"- {title}: {summary}" if summary else f"- {title}")
         combined = "\n".join(texts)
-        prompt = (custom_prompt or _GROUP_SUMMARY_DEFAULT_PROMPT).format(text=combined)
         try:
             result = await loop.run_in_executor(
                 None,
-                lambda t=combined, p=prompt: _summarize_text_block(
-                    t, api_name=api_name, model_override=model_override, custom_prompt=p,
+                lambda t=combined: _summarize_text_block(
+                    t,
+                    api_name=api_name,
+                    model_override=model_override,
+                    custom_prompt=custom_prompt or _GROUP_SUMMARY_DEFAULT_PROMPT,
                 ),
             )
             group["summary"] = result
+        except LLMQuotaExceededError:
+            raise
         except Exception as exc:
             logger.warning(f"outputs_service: group summary failed for '{group.get('name')}': {exc}")
             group["summary"] = ""
@@ -1022,7 +1045,7 @@ async def classify_items_by_topic(
         summary = item.get("llm_summary") or item.get("summary", "")
         texts.append(f"[{idx}] {title}: {summary[:200]}" if summary else f"[{idx}] {title}")
     combined = "\n".join(texts)
-    prompt = _TOPIC_CLASSIFY_PROMPT.format(max_groups=max_groups, text=combined)
+    prompt = _TOPIC_CLASSIFY_PROMPT.replace("{max_groups}", str(max_groups))
 
     try:
         result = await loop.run_in_executor(
@@ -1072,6 +1095,8 @@ async def classify_items_by_topic(
                 "summary": None,
             })
         return groups if groups else None
+    except LLMQuotaExceededError:
+        raise
     except Exception as exc:
         logger.warning(f"outputs_service: topic classification failed: {exc}")
         return None
```

### A.3 `audio-tests.patch`

```diff
diff --git a/tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py b/tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
index 74982b8987..33652b6865 100644
--- a/tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
+++ b/tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
@@ -2,6 +2,7 @@ import os
 import shutil
 import asyncio
 import tempfile
+from types import SimpleNamespace
 import numpy as np
 import soundfile as sf
 import pytest
@@ -743,3 +744,128 @@ async def test_audio_worker_owner_over_limit_defers_repeatedly_without_quarantin
     finally:
         stop.set()
         await asyncio.wait_for(task, timeout=2.0)
+
+
+# --- audio_analyze through the real analyze() (TASK-13637) ---
+
+
+class _AnalysisAdapter:
+    """The provider adapter below analyze(): records each request and replies or raises."""
+
+    def __init__(self):
+        self.reply = "Model analysis."
+        self.error = None
+        self.requests = []
+
+    def chat(self, request, timeout=None):
+        self.requests.append(request)
+        if self.error is not None:
+            raise self.error
+        return {"choices": [{"message": {"content": self.reply}}]}
+
+
+@pytest.fixture
+def analyze_worker(monkeypatch, tmp_path):
+    """The audio worker with quota stubs, a fresh jobs DB and a fake provider adapter."""
+    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs_analyze.db"))
+    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")
+
+    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as summarization
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
+    adapter = _AnalysisAdapter()
+    monkeypatch.setattr(summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _provider: adapter))
+    monkeypatch.setattr(summarization, "load_and_log_configs", lambda: {"openai_api": {"model": "configured-model"}})
+    return worker, JobManager(), adapter
+
+
+async def _run_worker_until(worker, done) -> None:
+    stop = asyncio.Event()
+    task = asyncio.create_task(worker.run_audio_jobs_worker(stop))
+    try:
+        for _ in range(100):
+            await asyncio.sleep(0.05)
+            if task.done():
+                pytest.fail(f"audio worker crashed: {task.exception()!r}")
+            if done():
+                break
+    finally:
+        stop.set()
+        await asyncio.wait_for(task, timeout=2.0)
+
+
+def _analyze_job(jm, text: str) -> int:
+    row = jm.create_job(
+        domain="audio", queue="default", job_type="audio_analyze",
+        payload={"text": text, "api_name": "openai", "model": "whisper-1", "perform_analysis": True}, owner_user_id="7",
+    )
+    return int(row["id"])
+
+
+@pytest.mark.unit
+async def test_audio_analyze_stores_the_model_analysis(analyze_worker):
+    worker, jm, adapter = analyze_worker
+    _analyze_job(jm, "hello transcript")
+
+    def stored():
+        return jm.list_jobs(domain="audio", job_type="audio_store", owner_user_id="7")
+
+    await _run_worker_until(worker, stored)
+    assert stored(), "audio_analyze did not hand over to audio_store"
+    assert stored()[0]["payload"]["analysis"] == "Model analysis."
+    assert [r["messages"][0]["content"] for r in adapter.requests] == ["hello transcript"]
+    assert adapter.requests[0]["model"] == "configured-model"
+
+
+@pytest.mark.unit
+async def test_audio_analyze_fails_the_stage_on_an_error_result(analyze_worker):
+    worker, jm, adapter = analyze_worker
+    job_id = _analyze_job(jm, "")  # analyze() returns "Error: Could not extract text content."
+    await _run_worker_until(worker, lambda: int((jm.get_job(job_id) or {}).get("retry_count") or 0) >= 1)
+    job = jm.get_job(job_id) or {}
+    assert int(job.get("retry_count") or 0) >= 1
+    assert "Could not extract text content" in str(job.get("error_message") or job.get("last_error"))
+    assert jm.list_jobs(domain="audio", job_type="audio_store", owner_user_id="7") == []
+
+
+@pytest.mark.unit
+async def test_audio_analyze_quota_refusal_fails_the_job_without_retry(analyze_worker):
+    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
+
+    worker, jm, adapter = analyze_worker
+    adapter.error = LLMQuotaExceededError(
+        category="llm_tokens_month", current=12, limit=10, message="over", retry_after=60
+    )
+    job_id = _analyze_job(jm, "hello transcript")
+    await _run_worker_until(worker, lambda: (jm.get_job(job_id) or {}).get("status") == "failed")
+    job = jm.get_job(job_id) or {}
+    assert job.get("status") == "failed"
+    assert int(job.get("retry_count") or 0) == 0
+    assert job.get("error_code") == "llm_quota_exceeded"
+
+
+@pytest.mark.unit
+async def test_audio_analyze_provider_failure_still_retries(analyze_worker):
+    worker, jm, adapter = analyze_worker
+    adapter.reply = ""  # no usable analysis: a provider failure, not a refusal
+    job_id = _analyze_job(jm, "hello transcript")
+    await _run_worker_until(worker, lambda: int((jm.get_job(job_id) or {}).get("retry_count") or 0) >= 1)
+    job = jm.get_job(job_id) or {}
+    assert int(job.get("retry_count") or 0) >= 1
+    assert job.get("status") != "failed"
+    assert len(adapter.requests) >= 1
diff --git a/tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py b/tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py
index 6df2c70073..01a626c2b9 100644
--- a/tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py
+++ b/tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py
@@ -84,7 +84,7 @@ async def test_the_audio_jobs_worker_runs_analysis_for_the_job_owner(monkeypatch
 
     seen: list[int | None] = []
 
-    def _analyze(text: str, api_name: str, *args: object, **kwargs: object) -> str:
+    def _analyze(api_name: str, input_data: str, *args: object, **kwargs: object) -> str:
         """Record the user the analysis runs as."""
         seen.append(current_llm_user_id())
         return "analysis"
```

### A.4 `audio-src.patch`

```diff
diff --git a/tldw_Server_API/app/services/audio_jobs_worker.py b/tldw_Server_API/app/services/audio_jobs_worker.py
index f621c7e18a..e23d2f19b3 100644
--- a/tldw_Server_API/app/services/audio_jobs_worker.py
+++ b/tldw_Server_API/app/services/audio_jobs_worker.py
@@ -303,6 +303,7 @@ async def run_audio_jobs_worker(stop_event: asyncio.Event | None = None) -> None
             updated_payload = dict(payload)
             ok = True
             msg_err = ""
+            failure: Exception | None = None
             try:
                 if jtype == "audio_download":
                     # Download to temp path
@@ -351,7 +352,16 @@ async def run_audio_jobs_worker(stop_event: asyncio.Event | None = None) -> None
                         segs = updated_payload.get("segments") or []
                         text = " ".join((s.get("Text", "").strip() if isinstance(s, dict) else "") for s in segs).strip()
                     with llm_call_root(owner):
-                        analysis = await asyncio.to_thread(analyze, text, payload.get("api_name") or "openai")
+                        analysis = await asyncio.to_thread(
+                            analyze,
+                            payload.get("api_name") or "openai",
+                            text,
+                            None,  # jobs carry no prompt: analyze's default system prompt (bulleted notes)
+                            input_is_literal_text=True,
+                            raise_on_error=True,
+                        )
+                    if isinstance(analysis, str) and analysis.startswith("Error:"):
+                        raise RuntimeError(analysis)
                     updated_payload["analysis"] = analysis
                     next_type = "audio_store"
                 elif jtype == "audio_store":
@@ -365,6 +375,7 @@ async def run_audio_jobs_worker(stop_event: asyncio.Event | None = None) -> None
             except Exception as e:
                 ok = False
                 msg_err = str(e)
+                failure = e
 
             if ok:
                 if next_type:
@@ -381,7 +392,16 @@ async def run_audio_jobs_worker(stop_event: asyncio.Event | None = None) -> None
                 else:
                     jm.complete_job(int(job["id"]), result={"payload": updated_payload}, worker_id=worker_id, lease_id=str(job.get("lease_id")), completion_token=str(job.get("lease_id")))
             else:
-                jm.fail_job(int(job["id"]), error=msg_err, retryable=True, worker_id=worker_id, lease_id=str(job.get("lease_id")), completion_token=str(job.get("lease_id")))
+                # An error that says it is not retryable (an LLM quota refusal) fails the job now.
+                jm.fail_job(
+                    int(job["id"]),
+                    error=msg_err,
+                    retryable=bool(getattr(failure, "retryable", True)),
+                    error_code=getattr(failure, "failure_code", None),
+                    worker_id=worker_id,
+                    lease_id=str(job.get("lease_id")),
+                    completion_token=str(job.get("lease_id")),
+                )
 
         except asyncio.CancelledError:
             raise
```
