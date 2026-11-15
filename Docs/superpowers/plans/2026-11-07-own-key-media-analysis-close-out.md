# Media analysis and the media ingest job run on the user's stored key; close TASK-15426 (PR D of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Media analysis on `/media/add` and the six `/media/process-*` endpoints, and a media ingest job, run every analysis call on the user's own stored key for `api_name`; their boundaries name that provider, and the job's start check admits an over-limit owner whose analysis runs on their own key without marking the job admitted. Closes TASK-15426.

**Architecture:** The handle rides the LLM call context instead of a parameter through about ten ingestion libraries. `LLMCallContext` gets a `provider_credentials` field; `bind_provider_credentials(handle)` sets it for the rest of the task (a plain set, like `mark_llm_quota_prechecked`), and `scoped_provider_credentials(provider)` returns it only for its own provider. `Summarization_General_Lib._dispatch_to_api`, the single caller of `_summarize_via_adapter`, reads it when the caller brought no key, app config or handle of its own. Because the field is part of the call state, it travels wherever the call's user travels: tasks and `asyncio.to_thread` copy the context, and the default executor and the bounded daemons carry `capture_llm_call_state()`. The boundary is one line per endpoint, `await admit_media_analysis(current_user.id, form_data)` in place of `await enforce_llm_quota(current_user.id)` (the six `process_*` endpoints are frozen; the line count does not change). `media_analysis_provider(form_data)` names `api_name` to the quota, or None when the request turns on another LLM stage the boundary cannot name; the handle itself is resolved for `api_name` whatever else is on (owner, 2026-11-07, "use their key"), so an under-limit user's analysis runs on their key even then, and the None still refuses an over-limit user. The handle comes from PR A's merged `users_own_key_credentials`. The job's `_admit_or_refuse_analysis` resolves the owner's handle, keeps its gate, binds the handle unless it refuses, and returns no refusal (and no mark) for an over-limit owner whose handle was issued and who turned on no other stage. `_handle_job` installs the carrying executor itself, and `llm_call_root` drops a bound handle as it drops the admission mark.

**Tech Stack:** Python 3.12, FastAPI, pytest, the Jobs worker SDK.

**Spec:** the TASK-15426 task file (ACs #2-#3, the media family and the job's start check); ADR-083; the brief; PR A's merged helpers, `users_own_key_credentials` (`usage_quota_deps.py:91-122`) and `enforce_llm_quota` (`:125-152`; a request that names no provider is refused whatever `own_key` says, `:150`), and its test helper `users_own_key` (`tests/helpers/llm_quota.py:63-97`); `plan_15426_a.md` (Decisions 2-4); `Docs/superpowers/plans/2026-11-05-own-key-server-key-boundaries-close-out.md` (the pins flipped here, and why the job's marker exists, TASK-13864).

**Measured on:** `cb0e9bb0d4`, a simulated dev after PR C: `origin/dev` `7f30182940` (PR A #3444, the judge fix #3445 (TASK-15485) and PR B #3452, all merged; `7f30182940` is B's merge commit), then `git merge --no-ff` of PR C (#3460, `feat/own-key-evaluation-runs-15426` at `ef0ef78ecc`, open, not in dev); the merge did not conflict. Prototype: `proto_15426_d.patch` (plans dir; 21 files, +429/-75: 16 source and test files and 5 docs, 2 of them the `Docs/Published` mirrors); it applies to `cb0e9bb0d4`. C changes none of D's 16 source and test files (`git diff 7f30182940 cb0e9bb0d4` is empty for them), so those hunks also apply to dev alone; the docs hunks are written on C's final text and need C. `file:line` citations are for `cb0e9bb0d4`, and for the prototype where the code is new (said so). History: written on `cb70f0c5d1` + `proto_15426_a.patch`; folded on `7a020da14a` with the folded A, B and C prototypes stacked (see "Review fold (2026-11-07)"); refreshed onto this base (see "Refresh (2026-11-13)"); the refresh review folded in (see "Review fold (2026-11-13)" at the end).

**Order:** last. PR A (#3444, merged 2026-11-10), the judge fix (#3445) and PR B (#3452) are in dev; PR C (#3460, open) goes first. D's source files are disjoint from C's, but D's docs hunks rewrite C's final sentences, so D branches only once #3460 is in dev. D closes the task.

---

## What was verified (on `cb0e9bb0d4`)

| # | Question | Finding | Consequence |
|---|---|---|---|
| V1 | The boundaries | `add_media` (`media/add.py:83-85`, not frozen, 61 lines) and `process_{audios,documents,ebooks,emails,pdfs,videos}_endpoint` (`:160-162`, `:151-153`, `:208-210`, `:97-99`, `:158-160`, `:138-140`) call `enforce_llm_quota(current_user.id)` under `media_analysis_requested(form_data)` (`media_processing_deps.py:664-671`). The six are frozen at 357, 528, 413, 455, 428 and 329 lines and measure one under each on this base (356, 527, 412, 454, 427, 328): #3394 (merged 2026-11-09) took a net line out of each processing loop, and added an import line above it, the +1 shift of each `process_*` boundary since `cb70f0c5d1`. | One-line replacement; nothing else in those functions changes. |
| V2 | Where analysis runs | About ten libraries call `analyze(api_name, ..., api_key=None)` with no model: `Audio_Files.py:1190`, `:1313`; `Book_Processing_Lib.py:868-1442`; `Email_Processing_Lib.py:443`; `PDF_Processing_Lib.py:1176`, `:1227`; `Plaintext_Files.py:512`, `:546`; `XML_Ingestion_Lib.py:155`; `persistence.py:1667-1740`; the video library (`Video_DL_Ingestion_Lib.py:1595-1654`), which `video_batch.py:118` hands `api_name`. Every one reaches `_summarize_via_adapter` through `_dispatch_to_api` (`Summarization_General_Lib.py:535-587`), its only caller (git grep). | One read in `_dispatch_to_api` covers them all. |
| V3 | How the call context crosses threads today | The endpoints hop with `loop.run_in_executor(None, ...)` (`process_documents.py:442`, `process_ebooks.py:421`, `process_emails.py:249-364`, `audio_batch.py:164`, `video_batch.py:140`, `persistence.py:4560`, `:4625`, `:5404`) and `asyncio.to_thread`. The default executor is `LLMStateThreadPoolExecutor`, which carries `capture_llm_call_state()` into its threads (`Usage/llm_roots.py:33-52`), installed at startup (`main.py:1275-1277`) and by `llm_call_root` for every Jobs handler (`Jobs/worker_sdk.py:721`, `:963`); the bounded daemons carry the same state (`LLM_Calls/bounded_daemon.py:139-146`). "Only the LLM call state crosses into the thread; no other context does." The one explicit pool on these paths (`PDF_Processing_Lib.py:1603-1609`) runs OCR, not chat calls. | A field of `LLMCallContext` goes wherever the call's user goes; a separate `ContextVar` would be lost at the executor and daemon hops. |
| V4 | Other LLM stages a media request can turn on | Claims extraction (`perform_claims_extraction`, else the server's `ENABLE_INGESTION_CLAIMS`, `Claims_Extraction/claims_utils.py:44-57`), contextual chunking (`enable_contextual_chunking`, or inferred from `contextual_llm_model`/`context_window_size`, `chunking_options.py:171-196`), LLM chunk boundaries (`auto_chunking_use_llm`), LLM propositions (`proposition_engine` `llm`/`auto`), a chunking template (`auto_apply_template`, `chunking_template_name`), and the confabulation check. Each picks its own provider, and none of their calls in a media request or job goes through `_dispatch_to_api`: claims call the adapter directly with the server's key (`ingestion_claims.py:324-359`), chunk boundaries `perform_chat_api_call_async` (`Chunking/auto_boundary_assistant.py:342-344`), the confabulation check `run_geval` with a resolved server key (`Video_DL_Ingestion_Lib.py:1237-1243`). | Any of them names None: refused over a limit (fail closed). Under a limit they keep the server's key, as the docs say. |
| V5 | The job's start check | `_admit_or_refuse_analysis` (`media_ingest_jobs_worker.py:363-380`) uses the call-site gate (`user_to_check`, `check_user_async`, which hands the check to the accounting loop from a worker loop) and then `mark_llm_quota_prechecked()`: the marker keeps an admitted job from being refused part-way through an item (TASK-13864). `_handle_job` scopes the job's context with `llm_call_context()` (`:513-520`), so the mark ends with the job. The job calls it at `:598`, after building its form from the payload's options (`_build_form_data`, `:353-360`). | Keep the gate and the marker for everyone else; an own-key owner is admitted unmarked; a handle bound inside the job also ends with it. |
| V6 | The web-scraping endpoints | `/media/ingest-web-content` (`ingest_web_content.py:78-80`) and `/media/process-web-scraping` (`process_web_scraping.py:76-79`) check whenever analysis/summaries are on, with or without a provider: the crawl methods summarize with a default provider and the others may extract with an LLM (TASK-13889). | Out of scope here: they keep the server's key and the refusal (Decision 4), and their pin (`test_web_summary_service_prompt.py`) stays as it is (in the wider set, passes on both trees). |
| V7 | RED on the base | 23 failed, 9 passed, 1 collection error (`$FOCUSED_D` with D's tests on the base's source, serially, `--continue-on-collection-errors`, `test_llm_call_context.py` selected whole because it cannot import the new names; 0:47; 0 netprobe attempts). The failures: the three `analyze` cases (no `bind_provider_credentials`); every media handler and ebook case, own key and server key (the boundary asks about nothing, so `asked == []`; the own-key request gets the 402); the two other-stage tests (`([], []) != (["openai"], [])`; `[None] != ["sk-users-own-key"]`); the job's four new or extended tests (refused, `asked == []`; no handle reaches the analysis). The `media_analysis_requested` cases, the no-analysis ebook test and the job's three existing tests pass on the base, as they must. Same counts as the fold measured on `7a020da14a`. | |
| V8 | Open PRs (2026-11-12) | Every open PR's full file list (`gh api .../pulls/<n>/files`, 44 PRs; #3454, the v0.1.47 release from dev to main, skipped: 9898 files, all dev's) against D's 21 files, then a trial merge of each overlapping head with the prototype (`git merge-tree --write-tree`, objects only). #3451 (`e4a7d600b3`), #3455 (`35280c7490`) and #3458 (`2dbdfa4cf0`), the stacked TASK-13950 refactors, edit five `process_*` endpoints' imports (`deprecation_signals` and `input_contracts`, a few lines below the `usage_quota_deps` import this PR drops, with unchanged lines between) and #3458 also the worker's `_schedule_embeddings` (`:403`): each merges with the prototype with no conflict in D's files (each conflicts with dev in `flashcards.py` and `quizzes.py` only, on base and prototype alike). #3269 (`81b0bb5836`): its worker hunks are on dev; 230 conflicting paths with dev, none of D's, on both trees. #2738 (`f303f5e207`) already conflicts with dev in the worker; the prototype adds none. #3394, which the fold checked, merged on 2026-11-09 (V1). #3436 touches `ingest_web_content.py`, which this PR does not. No open PR touches `llm_call_context.py`, `llm_roots.py`, `Summarization_General_Lib.py`, `media_processing_deps.py`, `add.py`, `process_emails.py`, this PR's tests or its docs, except #3460 (PR C, in the base). | Either order with each. |
| V9 | The judge fix (TASK-15485, #3445) | No code path of this PR reaches `evaluation_manager.py`: the media endpoints and libraries, the worker, `media_processing_deps.py`, `core/Usage` and `Summarization_General_Lib.py` never name `evaluation_manager`, `EvaluationManager`, `evaluate_custom_metric` or `_generate_model_response` (git grep, empty). `evaluation_manager.py` calls `analyze` with its own `provider_credentials` keyword (`:605-613`, `:776-785`), and nothing binds a handle in a benchmark's context, so the new read in `_dispatch_to_api` never fires there. Its unscored contract is untouched. | This PR needs and changes nothing of it. |
| V10 | The carry rule (owner, B through D) | See "Plan-specific" below: where the handle lives, the one place that reads it, and every payload, record, log and root it does not reach, with lines. | The handle is a field of the in-memory call context only. |
| V11 | An own-key call with no model (B's Review Focus 6) | Measured through `/media/process-documents` with the adapter stubbed (it records the model it is asked for), `DEFAULT_MODEL_OPENAI=gpt-env-default` set, under the probe (0 attempts). On the user's key with the server's model in config (the shipped `config.txt` has `openai_model = gpt-4o`; the real `_build_app_config` copies it into the handle): the call reaches `gpt-4o` on the user's key, item `Success`. On the user's key with no model in config: no call; HTTP 200, item `Warning`, no analysis, warning `Analysis failed for chunk 1: Model is required for provider 'openai'`. On the server's key with no model in config: the call reaches `gpt-env-default`, item `Success`. The handle path takes the model from the handle's `app_config` only (`Summarization_General_Lib.py:303-311`), not the `DEFAULT_MODEL_<PROVIDER>` fallback of `resolve_provider_model` (`adapter_utils.py:160-180`). No media library passes a model (`model_override`, git grep empty). | Pre-existing for every handle caller; with this PR it also reaches media analysis, for under-limit users with a stored key too (Decision 7). Filed as `TASK-<FDM>` (Task 0 Step 2); D adds no guard (Decision 9). Both are controller rulings of 2026-11-13. |

## Decisions

1. **ADR-004 Change summary.** Waived for every TASK-15426 PR (owner, 2026-11-07); recorded as a full `## Change summary` section.
2. **A context-scoped handle read in `_dispatch_to_api`, not a parameter through each library (ruling; the brief's question).** Trade-off: the key a call uses is no longer visible at its call site. It is bounded: only `analyze`'s dispatch reads it, only for the handle's own provider, and only when the caller passed no key, app config or handle (an explicit one always wins). It is safe across the hops that matter because it is part of the LLM call state (V3): wherever the user travels, the handle travels. The residual risk is a hop that drops the call state, such as `run_in_executor(None, ...)` on a loop whose default executor is asyncio's own: the call then loses the user and the handle together, so it runs on the server's key, unrecorded and unchecked; for an over-limit owner admitted unmarked that would be spend past the limit. Today such a call is already unrecorded for everyone. Production installs the carrying executor on the app's loop at startup and on every Jobs handler's loop (V3); measured: calling `_handle_job` directly on a bare test loop dropped the handle and the user. So `_handle_job` now installs the carrying executor itself (`ensure_llm_state_executor()`, idempotent; review D-1, Task 3 Step 3): the job carries its own state whoever calls it, and the extended test calls `_handle_job` with no executor line of its own. Measured: the analysis calls of all seven converted endpoints (audios, videos, documents, PDFs, ebooks, emails and add, with the parsable inputs of review D-5, and with speech-to-text and the video library's transcription stubbed, so no ffmpeg runs) and of a job all receive the handle. The alternative threads `provider_credentials` through about thirty call sites in ten libraries and grows the frozen `process_*` endpoints and `add_media_orchestrate` (816). Read in `_dispatch_to_api` rather than `_summarize_via_adapter` so stub-based tests (which replace `_summarize_via_adapter`) see the handle as a keyword.
3. **Other LLM stages keep the refusal over a limit (ruling).** `media_analysis_provider` is a denylist (V4), `ponytail:` comment: a stage it misses is checked at its own call (the request is admitted unmarked), so it is refused there and never counted. Some listed switches may make no LLM call (claims in `heuristic` mode); listing them only refuses more. So does the claims switch on the six `/media/process-*` endpoints: of the seven endpoints and the job, only `/media/add` and the job run claims extraction (`persistence.py:4782`, `:5586`). The forms of five of them (audios, videos, documents, PDFs, ebooks) have no `perform_claims_extraction` field; only `get_add_media_form` (`media_add_deps.py:105`, also the ingest-job submit route's form, `ingest_jobs.py:602`) and `get_process_emails_form` (`media_processing_deps.py:558`) take it. So with the server's `ENABLE_INGESTION_CLAIMS` on, those five refuse an over-limit user's own-key analysis and the user cannot opt out. Measured in the 2026-11-13 fold: with the default on, `perform_claims_extraction=false` admits the request on `/media/add` and `/media/process-emails` (one analysis on `sk-users-own-key`) and not on `/media/process-documents` (the 402). Before D they refused that user anyway, so nothing regresses; the BYOK guide says it (Task 4 Step 1). The denylist decides only what the boundary names to the quota; it no longer decides whether the handle is resolved (Decision 7).
4. **The web-scraping endpoints keep the server's key (scope ruling; owner decision, 2026-11-07: out of scope, with watchlist output enrichment and `rag_pipeline` evaluation runs).** Their scrapers choose providers inside themselves (V6); placing them on one key needs the same audit RAG got in TASK-13956 B. The owner's "the /media endpoints" is taken as the seven that analyse with `api_name`.
5. **The job's marker (AC #3).** An under-limit owner: marked as before. An over-limit owner whose handle was issued and who turned on no other stage: admitted, not marked, so each call is checked where its key is known; the analysis calls carry the owner's key and are not checked, so nothing is refused part-way. Anyone else: refused at the start, as before.
6. **A key revoked mid-job (owner decision, 2026-11-07: keep).** Same as PR C Decision 3: the job finishes on the key resolved at its start; a key revoked at the vendor fails the remaining analysis calls (recorded on the item); no fallback to the server's key. Task 4 Step 1 extends PR C's BYOK sentence to the ingest job: a job already started keeps using the key; cancel it (`DELETE /api/v1/media/ingest/jobs/{job_id}`, `ingest_jobs.py:1033-1041`) to stop it.
7. **Under-limit users get their own key too, whatever other stage is on (owner decision, 2026-11-07, review D-2: "use their key").** The handle is resolved for `canonical_provider_name(api_name)` and bound whenever it is issued, whatever the quota and whatever other LLM stage the request turns on (claims extraction, contextual chunking and the rest of V4). The 402 rule for an over-limit user is unchanged: the boundary still names `media_analysis_provider(form_data)`, which is None with another stage on, and `enforce_llm_quota` refuses a None provider whatever `own_key` says (`usage_quota_deps.py:150`, `names and all(names)`). In the job, an over-limit owner with another stage on keeps the refusal, and nothing is bound for a refused job (its other stages stay as before: checked at their call, refused). Before the fold the handle was resolved for `media_analysis_provider(form_data)`, so such a user's analysis ran on the server's key and counted; the other-stage test's `asked == []` becomes `["openai"]`.
8. **A background root never inherits a handle (review D-4).** `llm_call_root` resets `provider_credentials` to None as it resets `prechecked`: a root is never admitted by, and never runs on the key of, the request that started it. No path starts a root from a context holding a handle today (the review's Secrets section); this closes it for good.
9. **No media-only guard for an own-key call with no config model (controller ruling, 2026-11-13; refresh review M2).** The refresh review offered an optional two-line guard: `media_analysis_credentials` returns None when the handle's `app_config` has no model for the provider (`configured_provider_model_from_snapshot`), so such a user keeps the server's key and the env default, as before D. D does not add it. B and C already ship the same gap on their paths without a guard, and a media-only fallback to the server's key would be inconsistent with them and with the owner's D-2 "use their key". The shipped config supplies a model for every provider, and the real `_build_app_config` copies it into the handle. `TASK-<FDM>` (Task 0 Step 2) fixes it for every caller, in `_summarize_via_adapter`. The cost if this is wrong: on a server whose only default model for a provider is `DEFAULT_MODEL_<PROVIDER>`, a user with a stored key gets a visible `Warning` with no media analysis (HTTP 200, no provider call; V11), under a limit too, until `TASK-<FDM>` lands. The PR body states it as a known limit (Task 4 Step 5).

## Global Constraints

Verbatim from the brief:
- Tests make no real outbound network requests; stub the provider adapter or its transport. A test that reaches a commercial provider through an endpoint calls the real API unless the adapter or call is stubbed. Show it with `tldw_Server_API/tests/helpers/offline_network.block_live_network()` or an equivalent socket probe.
- Function-size ratchet `tldw_Server_API/tests/lint/function_size_baseline.json`: frozen functions must not grow.
- Backlog:
  - edit tasks with backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`);
  - `--append-notes`, never `--notes`; stage with `git add -A backlog/tasks`; never hand-edit task files;
  - a new task's explicit id comes from `zsh <plans dir>/task_next.sh` (prints the HIGHEST id in use; the new id is that plus one), after `mkdir <scratchpad>/id_lock`; release it once the PR's file list shows the task. Write ids as placeholders in the plan. Do NOT allocate ids yourself now.
- Branch from `origin/dev`; the PR targets `dev`. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- Redis: never FLUSHALL or SCRIPT FLUSH. PostgreSQL only through the repo fixtures (`tldw_postgres_test`), with a session-unique `TEST_DB_NAME`; drop only your own objects.
- Mirrored docs (`Docs/Published`) need `bash Helper_Scripts/refresh_docs_published.sh` in the same PR.
- CI checks before the PR, in the last task:
  - `tests/Docs`, `tests/lint`, `tests/CI`, and backlog-py's tests;
  - shard coverage (`Helper_Scripts/ci/check_shard_coverage.py`) and `Helper_Scripts/ci/test_quality_triage.py --enforce`;
  - the route-auth ratchet and the OpenAPI export check;
  - Bandit (`/opt/homebrew/bin/uvx bandit -ll`) on touched source;
  - every test file that names a touched symbol (git grep), with a baseline on `origin/dev`; run PostgreSQL files serially.
  - A file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` whose lines shift fails backend-required (regenerate with `--write-baseline` in the same PR); check every file you touch.
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` exit 0.
  - The Frontend E2E journeys are not required checks; the PR step diffs the PR's E2E failures against a recent dev run.
- Agent shells don't keep the venv `PATH`: prefix commands that spawn `python` with `PATH="$(dirname $PY):$PATH"`.

Plan-specific:
- `$PY` is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`; `<plans dir>` is `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`; `<scratchpad>` is `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad`.
- The six `process_*` endpoints are frozen (ceilings 357, 528, 413, 455, 428, 329) and measure 356, 527, 412, 454, 427, 328 on the base and on the prototype: one line replaced by one line, and one import line replaced by a two-name import (outside the functions). `add_media_orchestrate` (ceiling 816, 815 on both) is not touched. Nothing else D touches is frozen or near 300 lines: `add_media` 61 → 61, `_admit_or_refuse_analysis` 18 → 24, `_handle_job` 8 → 10, `_dispatch_to_api` 53 → 56, `llm_call_root` 18 → 18, `analyze` 248 → 248 (untouched).
- The layering ratchet (`tests/lint/test_layering_contract.py`, `layering_baseline.txt:204`): `services -> api.v1.API_Deps` is at its baseline of 18 import statements on this base, so the worker reaches the new helpers through its existing `media_processing_deps` import (one statement, three names).
- Every test that can be admitted stubs `Summarization_General_Lib._summarize_via_adapter` and runs its request inside `block_live_network()`. A first prototype of the other-stage test did neither, was admitted (the form field it sent is not a `process-documents` field), and sent `POST https://api.openai.com/v1/chat/completions` (401, no key). The final test uses `auto_chunking_use_llm`, stubs, and probes.
- `scope_predicate_baseline.txt` lists none of this PR's 21 files (checked on `cb0e9bb0d4`).
- **Carry rule** (owner, for B through D): the own-key handle never goes inside a copied or persisted config, stored metadata, or a job payload. Where it holds for this PR (lines on `cb0e9bb0d4`, prototype lines where new):
  - **Where it lives.** One field of the in-memory call context, `LLMCallContext.provider_credentials` (prototype `llm_call_context.py:62`), set only by `bind_provider_credentials` (prototype `:237-245`), which only `admit_media_analysis` (prototype `media_processing_deps.py:729`) and `_admit_or_refuse_analysis` (prototype `media_ingest_jobs_worker.py:395`) call. The form (`form_data`) never gets it: the endpoints pass it on unchanged, and the job builds its form from the payload before the check (`_build_form_data`, `:353-360`) and only copies it to switch analysis off on a refusal (`:599-600`).
  - **The one reader.** `scoped_provider_credentials` (prototype `llm_call_context.py:248-255`), called only from `_dispatch_to_api` (prototype `Summarization_General_Lib.py:558-560`), which passes it as the `provider_credentials` keyword every handle caller already uses (`_summarize_via_adapter`, `:235`). `analyze` returns text; what the libraries store (the analysis, chunk metadata `analysis`, the media row) is built from that text, not from the context.
  - **The job.** The payload is written by the submit route (`ingest_jobs.py:670-676`, `:754-763`, `create_job` `:410`) from the request's options, before any handle exists; the worker never writes it. What the worker stores is built from the item's result: progress messages are fixed strings (`:594-596` and the later `update_job_progress` calls), the job result is `status`, media ids, `error`, `warnings`, `db_message`, `chunking_plan` and, on a refusal, `analysis_error` with the refusal's code and message (`:716-727`, `_record_analysis_refusal` `:383-389`); the Jobs manager stores it with `json.dumps` (`Jobs/manager.py:7951`), which raises on a handle rather than storing it. Collection-item status rows carry status and an error summary only. The embeddings job `/media/add` may enqueue gets explicit arguments (`persistence.py:2538-2549`), not the context.
  - **Roots and hops.** `llm_call_root` drops a bound handle (prototype `llm_roots.py:64`, Decision 8), so no Jobs job, workflow run or scheduled task started from a bound context inherits it; the embeddings worker runs under its own root (`worker_sdk.py:721`, `:963`). The other readers of the whole call state hand it to a thread in memory and store nothing: the carrying executor (`llm_roots.py:37`), the bounded daemons (`bounded_daemon.py:140`), the chat request queue's in-memory `QueuedRequest.llm_state` (`Chat/request_queue.py:391`, `:1105`), research, ingestion claims and the article extractor (`research.py:422`, `ingestion_claims.py:422`, `Article_Extractor_Lib.py:1207`). Usage accounting and the quota gate read named fields only (`usage_accounting.py:55-64`, `llm_quota_gate.py:60-61`). A FastAPI background task of the same request (`/media/add`'s `"background"` embeddings dispatch, `persistence.py:2586-2650`) runs in the request's context and so sees the handle and the admission mark, as it sees the mark today; it stores nothing from the context.
  - **Logs and audits.** No log line, audit record or usage event includes the call context (git grep for its readers, above); `_dispatch_to_api` logs the provider name only (`Summarization_General_Lib.py:556`).
  - **The backstop.** `ProviderCallCredentials` has a redacted `repr` (`provider_credential_runtime.py:269`), so a logged `LLMCallContext` would show `credentials=[REDACTED]`, and it refuses pickle, copy, deepcopy and pydantic serialization (`:271-290`): an accidental copy or store raises instead of leaking.
- Every test run in this plan runs with outbound connections blocked: `PYTHONPATH=<plans dir> NETPROBE_LOG=<file> $PY -m pytest -p netprobe_15427 ...`, or inside `block_live_network()`.
- `proto_15426_d.patch` is normative for docstrings and comments the plan does not quote; the docs prose is quoted in Task 4.
- ruff: no new finding on a touched file against the base (as PR A, Task 3 Step 3).

## Review Focus

1. **An over-limit user on the server's key, each endpoint.** Expected: the same 402 before any processing. Pinned by `test_media_handlers_name_the_analysis_provider_to_the_llm_quota[*-server-key]` and `test_ebook_analysis_names_its_provider_before_processing[server-key]` (Task 2).
2. **A request that also turns on another LLM stage.** Over the limit: the 402, no analysis, no network; the lookup runs (`asked == ["openai"]`) but cannot admit, because the boundary names no provider. Pinned by `test_another_llm_stage_keeps_the_refusal_for_an_own_key_user` and, for the job, `test_a_stored_key_for_another_provider_keeps_the_refusal[own-key-and-another-llm-stage]`. Under the limit: the analysis runs on the user's key. Pinned by `test_another_llm_stage_still_analyses_on_an_under_limit_users_own_key` (Decision 7; Tasks 2-3).
3. **A bound handle and an analysis call for another provider, or with its own key.** Expected: that call does not use the handle. Pinned by `test_analyze_runs_on_the_bound_handle_for_its_provider_without_a_key_of_its_own[another-provider|own-api-key]` (Task 1).
4. **The handle crosses an executor or daemon hop.** Expected: a thread given the call state sees it. Pinned by `test_a_bound_handle_serves_its_own_provider_and_crosses_threads_with_the_call_state` (Task 1) and the endpoint tests' analysis counts (Task 2).
5. **A job's key does not outlive the job, and a job carries its own state.** Expected: after `_handle_job` returns, the caller's context has no handle and no mark; called directly on a bare loop, the job's analysis still sees the handle. Pinned by `test_the_admission_ends_with_the_job` (extended, with no executor line of its own) (Task 3).
6. **A background root started from a bound context.** Expected: no handle inside the root. Pinned by `test_a_background_root_never_inherits_the_handle_of_the_request_that_started_it` (Task 1).
7. **The handle is never stored** (carry rule). Expected: no job payload, job result, media record, chunk, claim, log line or root carries it. Shown by reading (V10, the "Carry rule" constraint); `ProviderCallCredentials` raises on any copy or serialization.

## Files

- Modify: `tldw_Server_API/app/core/Usage/llm_call_context.py` (`LLMCallContext` `:51-60`; new `bind_provider_credentials`, `scoped_provider_credentials` at the end, after `:232`)
- Modify: `tldw_Server_API/app/core/Usage/llm_roots.py` (`llm_call_root`'s `overrides`, `:64`, review D-4)
- Modify: `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py` (import after `:56-59`; `_dispatch_to_api`, after the debug line `:556`)
- Modify: `tldw_Server_API/app/api/v1/API_Deps/media_processing_deps.py` (new `media_analysis_provider`, `media_analysis_credentials`, `admit_media_analysis` before `__all__` `:674`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/{add,process_audios,process_documents,process_ebooks,process_emails,process_pdfs,process_videos}.py` (imports; the boundary line, V1)
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py` (imports `:17`, `:40`; `_admit_or_refuse_analysis` `:363-380`; `_handle_job` `:513-520`, review D-1)
- Modify tests: `tldw_Server_API/tests/Usage/test_llm_call_context.py`, `tldw_Server_API/tests/LLM_Calls/test_summarization_runtime_credentials.py`, `tldw_Server_API/tests/Media/test_media_llm_quota.py`, `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py`
- Docs (Task 4): `Docs/Operations/Usage_Quotas.md`, `Docs/User_Guides/Server/BYOK_User_Guide.md`, `Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md` (+ the two mirrors; `Usage_Quotas.md` is not mirrored)
- Backlog: TASK-15426 (ACs #2-#3, DoD, final summary, Done); create `TASK-<FDM>` (Task 0 Step 2)
- Create (Task 0): `Docs/superpowers/plans/2026-11-07-own-key-media-analysis-close-out.md`

## Test commands

`$FOCUSED_D`:

```text
tldw_Server_API/tests/Usage/test_llm_call_context.py::test_a_bound_handle_serves_its_own_provider_and_crosses_threads_with_the_call_state
tldw_Server_API/tests/Usage/test_llm_call_context.py::test_a_background_root_never_inherits_the_handle_of_the_request_that_started_it
tldw_Server_API/tests/LLM_Calls/test_summarization_runtime_credentials.py::test_analyze_runs_on_the_bound_handle_for_its_provider_without_a_key_of_its_own
tldw_Server_API/tests/Media/test_media_llm_quota.py
tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py
```

Measured on `cb0e9bb0d4` under `-p netprobe_15427`: RED (D's tests, the base's source; serially, `--continue-on-collection-errors`, with `test_llm_call_context.py` selected whole because its two node ids do not collect on the base) `23 failed, 9 passed, 1 error (0:47)`; GREEN `34 passed (0:34, -n 4)`; 0 netprobe attempts in both. (The fold measured the same on `7a020da14a` with the folded A-C: RED 23 failed, 9 passed, 1 collection error; GREEN 34 passed.)

**The wider set:** the plan's grep, plus the callers it misses.

```bash
git grep -l -E 'LLMCallContext|current_llm_call_context|_dispatch_to_api|_summarize_via_adapter|media_processing_deps|media_analysis_requested|media_ingest_jobs_worker|/media/process-(audios|videos|documents|pdfs|ebooks|emails)|"/api/v1/media/add"' -- 'tldw_Server_API/tests/*.py' | grep '/test_[^/]*\.py$'
git grep -l -E 'llm_call_context|llm_roots|llm_call_root|ensure_llm_state_executor|LLMStateThreadPoolExecutor|Summarization_General_Lib|process-(audios|videos|documents|pdfs|ebooks|emails)|[}"]/add"' -- 'tldw_Server_API/tests/*.py' | grep '/test_[^/]*\.py$' | grep -v -E '^tldw_Server_API/tests/(e2e|server_e2e_tests)/'
```

The first grep gives 94 files on `cb0e9bb0d4` and on the prototype (87 on `cb70f0c5d1`). It misses 38 callers, which the second adds (132 files): 18 files that use the two touched `core/Usage` modules by their module name (`llm_call_context`, and `llm_call_root`/`ensure_llm_state_executor`, whose root now also drops the handle: `test_llm_quota_gate.py`, `test_llm_usage_recorded_through_executor.py`, `test_outputs_service.py`, the Claims, DataTables and quiz-generation root tests and twelve more), 14 more importers of `Summarization_General_Lib` (whose `_dispatch_to_api` changed and which now imports `llm_call_context`; among them `tests/lint/test_llm_calls_import_isolation.py`, which checks what importing it loads), 5 callers of the touched endpoints through a path built from a prefix or route introspection (`test_add_media_endpoint.py`'s `f"{API_PREFIX}/add"`, `test_research_discovery_media_add.py`, the media router import test and two route-binding contracts), and V6's pin `test_web_summary_service_prompt.py`. The second grep leaves out the e2e and server-e2e files it would add (they need a live server); the first grep's two, `e2e/workflows/test_workflow_audio_analysis.py` and `server_e2e_tests/test_rag_pipeline_workflow.py`, stay in, as before (they skip without a server; under load the second errors at its health check). Nine of the 132 take a PostgreSQL fixture (`pg_temp_db`, `pg_eval_params`, `pg_database_config`, `isolated_test_environment` and the like): `test_auth_principal_media_rag_invariants.py`, `test_media_permission_enforcement.py`, `test_rag_media_permissions_claims.py`, `test_fva_verify_endpoint.py`, `test_media_ingest_worker_content_scope.py`, `test_run_progress_response.py`, `test_eval_runner.py`, `test_study_assistant_adapter_contract.py`, `test_media_ingest_jobs_postgres_scope.py`. Run them serially with `TEST_DB_NAME` set, and the other 123 with `-n 4 --dist loadfile`.

Measured on `cb0e9bb0d4` under `-p netprobe_15427` (base → prototype). The 123 files with `-n 4 --dist loadfile`: `32 failed, 3310 passed, 38 skipped, 2 xfailed, 3 errors (42:24, load average about 65)` → `3 failed, 3355 passed, 40 skipped, 2 xfailed (11:13, load average about 20)`; 3384 → 3400 collected, the +16 being this PR's new cases (its four test files, 30 → 46). The three failures on both trees are pre-existing: `test_media_processing.py::TestProcessPdfs::test_process_pdf_upload_success[docling]` and `::test_process_pdf_multi_status_mixed[docling]` (they failed on `cb70f0c5d1` too), and `TestProcessAudios::test_process_audio_url_success_no_analysis_no_chunking`, a live download the blocker stops (TASK-15486). The base's other 29 failures and 3 errors were load: 25 in the two email-ingestion files (`test_email_offline_ingestion.py`, `test_email_sensitive_logging_13376_2.py`), 2 in `test_add_media_endpoint.py` (one also errors at teardown), the claims-engine deadline test, a video-dedupe job test and the two server-e2e health-check errors. Their six files re-run alone on the base pass (`157 passed, 9 skipped, 1 xfailed`, 0 attempts), and every one of them passes in the prototype's run. The nine PostgreSQL files, serially with `TEST_DB_NAME=tldw_test_refreshd` (PostgreSQL up, every case ran): `253 passed (18:48)` → `253 passed (8:22)`. Netprobe: the same 22 node ids on both trees (Task 0 Step 4), none in the PostgreSQL files.

---

## Task 0: Plan, branch, draft PR, baseline

- [ ] **Step 1: Branch** after PR C (#3460) merges (PR A, the judge fix and PR B are already in dev): `git fetch origin dev && git merge-base --is-ancestor origin/feat/own-key-evaluation-runs-15426 origin/dev && git switch -c feat/own-key-media-analysis-15426 origin/dev` (the check exits 0; if C's branch was deleted after merging, check that `gh pr view 3460 --json state` shows `MERGED` instead).
- [ ] **Step 2: Follow-up task, note and plan file** (controller ruling, 2026-11-13 fold: this PR files the `DEFAULT_MODEL_<PROVIDER>` follow-up V11 measured; "Ruled: own-key analysis with no model in config" at the end).
  - (a) **Id.** `mkdir <scratchpad>/id_lock` (if it exists, another session holds the lock: wait and retry; never remove a lock you did not create). `git fetch origin dev --quiet`, then `zsh <plans dir>/task_next.sh` from the repository root: it prints the HIGHEST id in use on `origin/dev` and in every open PR, so `<FDM>` = printed + 1. Check that `ls backlog/tasks backlog/completed backlog/archive/tasks | grep "^task-<FDM> "` prints nothing. From here on write the id literally in every command.
  - (b) **File TASK-<FDM>** (`task create` ignores `--append-notes`, `tools/backlog-py/src/backlog_py/cli/main.py:85-97`, so the facts go in `-d` and the cross-reference in (c)):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Own-key calls through analyze skip the DEFAULT_MODEL_<PROVIDER> fallback" \
  --id TASK-<FDM> -l byok,llm,bug \
  -d "Found by TASK-15426 PR B's plan review (B-6); measured by PR D's plan refresh (V11) and its review (2026-11-12). Pre-existing for every analyze call that carries a user's own-key handle. Summarization_General_Lib._summarize_via_adapter takes the model of a call with a provider_credentials handle only from the handle's app_config, the provider section's model or model_id (core/LLM_Calls/Summarization_General_Lib.py:303-311; lines on 7f30182940). A call on the server's key falls back to resolve_provider_model, which also reads the DEFAULT_MODEL_<PROVIDER> environment variable (core/LLM_Calls/adapter_utils.py:160-180). So on a server whose only default model for a provider is that variable (no model for it in config.txt), an own-key analyze call has no model and fails before any provider call. Measured through POST /api/v1/media/process-documents with PR D's prototype, the adapter stubbed, DEFAULT_MODEL_OPENAI=gpt-env-default set and no openai model in config: on the user's own key HTTP 200, the item a Warning with no analysis (Analysis failed for chunk 1: Model is required for provider 'openai'); on the server's key the call reaches gpt-env-default. Affected: PR B's benchmark judge and summaries, PR C's RAG and quality judges, and with PR D every media analysis of a user with a stored key, under or over a limit (PR D adds no media-only guard: controller ruling, 2026-11-13). Reach: the shipped config.txt sets a model for every provider and the real _build_app_config (core/Provider_Credentials/byok_runtime.py:975) copies it into the handle; no Dockerfile, compose file or .env.example sets DEFAULT_MODEL_*, so it needs an operator who removed the config model. Constraint on the fix: it belongs in _summarize_via_adapter, which every handle caller shares. The provider-override model policy is enforced at resolve time against the configured model only (core/Provider_Credentials/provider_credential_runtime.py:1063-1083), and _summarize_via_adapter does not check it again at dispatch, so a fallback model chosen at dispatch has to pass that policy too." \
  --ac "An own-key analyze call on a server whose only default model for the provider is DEFAULT_MODEL_<PROVIDER> (no model in config) reaches that model, with a test" \
  --ac "That model is checked against the provider-override model policy, as the configured model is, with a test"
```

  - (c) **Note on TASK-15426:** `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-15426 --append-notes "Plan, PR D of 4 (media analysis, media ingest job; closes the task): Docs/superpowers/plans/2026-11-07-own-key-media-analysis-close-out.md. Filed TASK-<FDM>: own-key calls through analyze skip the DEFAULT_MODEL_<PROVIDER> fallback (pre-existing; with PR D it also reaches media analysis)."` (the task id is required: a bare `15426` raises `KeyError: 'Task not found: 15426'`).
  - (d) **Plan file and commit.** Copy this plan to `Docs/superpowers/plans/2026-11-07-own-key-media-analysis-close-out.md` unchanged (it keeps the `<FDM>` placeholder, ends with one newline, and no line has trailing whitespace or a tab). `git add Docs/superpowers/plans/2026-11-07-own-key-media-analysis-close-out.md && git add -A backlog/tasks && git diff --cached --check` must print nothing; `git grep -n "TASK-<FDM>" -- ':!Docs/superpowers/plans'` must print nothing; `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` exit 0. Commit `chore(backlog): start TASK-15426 PR D, file TASK-<FDM> and add its plan`.

  Dry-run in the 2026-11-13 fold on `0d33c9f989` (dev `7f30182940` + C at `ef0ef78ecc`; its tree is `cb0e9bb0d4`'s), on a throwaway copy of `backlog/` with a throwaway id (TASK-99902, removed afterwards): (b) and (c) exit 0, the new file has the three labels and two ACs and its description matches `-d`, `check_backlog_task_ids.py <copy>` exits 0, and neither changed file has a trailing space or a tab.
- [ ] **Step 3: Draft PR** (`--base dev`), title `feat(media): media analysis and ingest jobs run on the user's stored key; close TASK-15426 (4 of 4)`. Then release the id lock: once `gh pr view <n> --json files --jq '.files[].path' | grep 'task-<FDM> '` prints the new task's file, `rmdir <scratchpad>/id_lock`.
- [ ] **Step 4: Baseline** the wider set (both greps of "Test commands", under the outbound blocker; PostgreSQL files serially): record counts, failing ids and netprobe lines. Expected, as measured on `cb0e9bb0d4`: 3384 tests collected in the 123 non-PostgreSQL files, the three failures named in "Test commands" (two `[docling]` PDF cases and the blocked `cdn.pixabay.com` download), and under load timing failures in the email-ingestion, `/media/add`, claims-deadline and video-dedupe files, which pass alone; the nine PostgreSQL files `253 passed`; 22 netprobe lines. A failure outside these is checked against a recent dev run before it is called flaky. Known and pre-existing, the same on base and prototype, so not a regression: 22 tests in the wider set attempt live DNS on dev, which the blocker stops (15 in `test_media_processing.py`: www.youtube.com, cdn.pixabay.com, filesamples.com, www.w3.org, raw.githubusercontent.com, example.com; 7 in `test_media_contextual_integration.py`: example.com). TASK-15486 (filed by the judge fix) lists all of them and stubs them; one of them, `TestProcessAudios::test_process_audio_url_success_no_analysis_no_chunking` (cdn.pixabay.com), fails when blocked.

## Task 1: The call context carries a handle, and `analyze` uses it (AC #2)

**Files:** `llm_call_context.py`, `llm_roots.py`, `Summarization_General_Lib.py`, `test_llm_call_context.py`, `test_summarization_runtime_credentials.py`.

**Interfaces:**
- Produces: `LLMCallContext.provider_credentials: Any = None`; `def bind_provider_credentials(handle: Any) -> None`; `def scoped_provider_credentials(provider: str) -> Any` (in `llm_call_context.py`).

- [ ] **Step 1: Write the failing tests.** In `test_llm_call_context.py`, add `import contextvars` at the top of the module and the two new names, `bind_provider_credentials` and `scoped_provider_credentials`, to its existing `llm_call_context` import list, in sorted place; `SimpleNamespace` is imported inside each test, as the prototype does:

```python
def test_a_bound_handle_serves_its_own_provider_and_crosses_threads_with_the_call_state() -> None:
    """A handle bound for the rest of a task answers for its provider only, and a thread given the call state
    sees it (TASK-15426)."""
    from types import SimpleNamespace

    handle = SimpleNamespace(provider="openai")
    seen: list[object] = []

    def _bound_in_a_task() -> None:
        bind_provider_credentials(None)  # binds nothing
        assert current_llm_call_context().provider_credentials is None
        bind_provider_credentials(handle)
        state = capture_llm_call_state()
        carried = threading.Thread(
            target=run_with_llm_call_state, args=(state, lambda: seen.append(scoped_provider_credentials("OpenAI")))
        )
        carried.start(), carried.join()
        seen.append(scoped_provider_credentials("anthropic"))

    contextvars.copy_context().run(_bound_in_a_task)
    assert seen == [handle, None]
    assert current_llm_call_context().provider_credentials is None
```

  In `test_summarization_runtime_credentials.py`, with every new import inside the test (a module-level import of `bind_provider_credentials` fails the whole file's collection on the base, its 7 existing tests included, instead of failing the 3 new cases):

```python
@pytest.mark.parametrize(
    ("api_name", "api_key", "uses_handle"),
    [("openai", None, True), ("anthropic", None, False), ("openai", "sk-raw", False)],
    ids=["bound-provider", "another-provider", "own-api-key"],
)
def test_analyze_runs_on_the_bound_handle_for_its_provider_without_a_key_of_its_own(
    monkeypatch: pytest.MonkeyPatch, api_name: str, api_key: str | None, uses_handle: bool
) -> None:
    """A request or job bound to the user's own key runs each analysis for that provider on it (TASK-15426)."""
    import contextvars
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Usage.llm_call_context import bind_provider_credentials

    handle = SimpleNamespace(provider="openai")
    seen: list[object] = []
    monkeypatch.setattr(sgl, "_summarize_via_adapter", lambda **kwargs: seen.append(kwargs["provider_credentials"]) or "ok")

    def _bound() -> object:
        bind_provider_credentials(handle)
        return sgl.analyze(api_name, "Cedar opens in January.", None, api_key, input_is_literal_text=True)

    assert contextvars.copy_context().run(_bound) == "ok"
    assert seen == [handle if uses_handle else None]
```

  (The `== "ok"` assert on `analyze`'s own output keeps `test_quality_triage.py --enforce` from flagging a collector-only test, `tautology_suspect`: measured on the first prototype.)

  In `test_llm_call_context.py` (review D-4; `contextvars` is the module-level import added above):

```python
def test_a_background_root_never_inherits_the_handle_of_the_request_that_started_it() -> None:
    """``llm_call_root`` drops a bound handle as it drops the admission mark (TASK-15426)."""
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Usage.llm_roots import llm_call_root

    def _root_started_from_a_bound_request() -> object:
        bind_provider_credentials(SimpleNamespace(provider="openai"))
        with llm_call_root(7):
            return current_llm_call_context().provider_credentials

    assert contextvars.copy_context().run(_root_started_from_a_bound_request) is None
```

- [ ] **Step 2: Run them.** Expected: `ImportError: cannot import name 'bind_provider_credentials'` (measured on `cb0e9bb0d4`: `test_llm_call_context.py` fails to collect; the summarization test imports the name inside the test, so its 3 cases fail with the same `ImportError`). With Step 3 but without the `llm_roots` line, the root test fails: `namespace(provider='openai') is None` (measured in the fold).
- [ ] **Step 3: Implement.** `LLMCallContext` gets `provider_credentials: Any = None` (comment: a runtime-issued handle for the user's own key, for `analyze` calls that bring no key). `bind_provider_credentials`: `if handle is not None: _CTX.set(replace(_CTX.get(), provider_credentials=handle))`. `scoped_provider_credentials`: the bound handle when `getattr(handle, "provider", None) == canonical_provider_name(provider)` (import inside the function, as `billed_credential_source` does), else None. Docstrings: the prototype's. In `_dispatch_to_api`, after the debug line (`:556`):

```python
        if provider_credentials is None and not api_key and app_config is None and not credentials_resolved:
            # A request or job that runs on the user's own key carries its handle (TASK-15426).
            provider_credentials = scoped_provider_credentials(_adapter_provider_name(api_name_lower))
```

  with `from tldw_Server_API.app.core.Usage.llm_call_context import scoped_provider_credentials` at the top (after the `provider_credential_runtime` import, `:56-59`). In `llm_roots.llm_call_root` (`:64`): `overrides: dict[str, Any] = {"prechecked": False, "provider_credentials": None}  # nor its own-key handle`.
- [ ] **Step 4: Run them.** Expected: 5 passed; both files whole: no new failure (`test_defaults_have_no_user_and_the_generic_labels` still holds: the new field defaults to None). `tests/lint/test_llm_calls_import_isolation.py` passes: the new import loads no AuthNZ module `Summarization_General_Lib` did not already load (measured, Task 3 Step 4).
- [ ] **Step 5: Commit** `feat(usage): the LLM call context carries the user's own-key handle for analyze (TASK-15426)`.

## Task 2: The media endpoints (ACs #2, #3)

**Files:** `media_processing_deps.py`, the seven endpoint modules, `test_media_llm_quota.py`.

**Interfaces:**
- Consumes: `users_own_key_credentials` (`usage_quota_deps.py:91`), `enforce_llm_quota` (`:125`), both PR A, merged; `bind_provider_credentials` (Task 1); in tests, `users_own_key` (`tests/helpers/llm_quota.py:63`, PR A).
- Produces: `def media_analysis_provider(form_data: Any) -> str | None`; `async def media_analysis_credentials(user_id: Any, form_data: Any) -> Any` (`users_own_key_credentials(user_id, canonical_provider_name(str(form_data.api_name)))`, whatever other stage is on, Decision 7); `async def admit_media_analysis(user_id: Any, form_data: Any) -> None` (raises `LLMQuotaExceededError`).

- [ ] **Step 1: Flip the pins.** In `test_media_llm_quota.py` import `users_own_key` (instead of `own_keys_for_every_provider`) and `block_live_network`, plus `io`, `wave`, `pymupdf`, `ebooklib.epub`, `Audio_Files` and `Video_DL_Ingestion_Lib` (`from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import Video_DL_Ingestion_Lib`, after the `Audio_Files` import) (review D-5: parsable inputs, so every converted endpoint is shown to analyse on the handle). Add a `transcribed` fixture (a fixture, so the test does not stack the patch calls): `Audio_Files.speech_to_text` returns a fixed one-segment transcript and `check_transcription_model_status` `{"available": True, "usable": True}`, as the ingest-job tests stub them, and `Video_DL_Ingestion_Lib.perform_transcription` returns `(None, segments)`, as `test_video_summary_service_prompt.py` stubs it (review M8; no model and no ffmpeg run):

```python
@pytest.fixture()
def transcribed(monkeypatch):
    """Speech recognition returns a fixed transcript (as the media ingest job tests stub it, and the video library's
    as ``test_video_summary_service_prompt.py`` does), so no model and no ffmpeg run."""
    segments = [{"start_seconds": 0, "end_seconds": 1, "Text": "Send the report on Friday."}]
    monkeypatch.setattr(Audio_Files, "speech_to_text", lambda **_kwargs: segments)
    monkeypatch.setattr(Audio_Files, "check_transcription_model_status", lambda _: {"available": True, "usable": True})
    monkeypatch.setattr(Video_DL_Ingestion_Lib, "perform_transcription", lambda **_kwargs: (None, segments))
```

  Add three builders: `_wav()` (0.1 s of 8 kHz mono silence through `wave` into a `BytesIO`), `_pdf()` (one `pymupdf` page with one line of text, `document.tobytes()`) and `_epub(tmp_path)` (one chapter through `ebooklib`, as `test_ebook_summary_service_prompt.py` builds one). Give `_OTHER_HANDLERS` a fourth value, the analysis calls an admitted request makes: audios 1 (`a.wav`, `_wav()`, `audio/wav`), videos 1 (`v.mp4`, `b"fake"`, `video/mp4`: puremagic, a core dependency, cannot identify the bytes, so the upload validator falls back to the extension, `Upload_Sink.py:668-703`; measured with ffmpeg off `PATH` too), documents 1, pdfs 1 (`_pdf()`, with `pdf_parsing_engine=pymupdf`), emails 1, add 1. Replace `test_other_media_handlers_refuse_analysis_for_an_over_quota_user` with `test_media_handlers_name_the_analysis_provider_to_the_llm_quota`, parametrized also by `own_key` (`own-key`, `server-key`) and taking `transcribed`: stub `_summarize_via_adapter` to record its kwargs and return `"Summary."`; `refuse_llm_work`; `asked = users_own_key(monkeypatch, "openai" if own_key else "anthropic")`; post inside `with block_live_network() as attempts:`. Assert `attempts == []` and `asked == ["openai"]`; own key: not 402 and `[getattr(k["provider_credentials"], "api_key", None) for k in summarized] == ["sk-users-own-key"] * analyses`; server key: `assert_llm_quota_402`, `summarized == []`. Replace `test_ebook_analysis_is_refused_before_processing` with `test_ebook_analysis_names_its_provider_before_processing`, parametrized `own_key`: wrap the real `media_mod.books.process_epub` (record, then call it) instead of stubbing it, stub `_summarize_via_adapter`, post `_epub(tmp_path)` inside `with block_live_network() as attempts:` (review D-7: the `[own-key]` request is admitted) and assert `attempts == []`, `asked == ["openai"]`; own key: not 402, `process_epub` called once and one analysis on `sk-users-own-key`; server key: the 402 and `(processed, summarized) == ([], [])`. Add:

```python
@pytest.mark.integration
def test_another_llm_stage_keeps_the_refusal_for_an_own_key_user(
    client_with_single_user, quota_service_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request that also asks for LLM chunk boundaries makes a call the boundary cannot place on the user's key,
    so an over-limit user gets the 402, though they stored a key for the analysis (TASK-15426)."""
    client, _usage_logger = client_with_single_user
    summarized: list[dict] = []
    monkeypatch.setattr(
        Summarization_General_Lib, "_summarize_via_adapter", lambda **kwargs: summarized.append(kwargs) or "Summary."
    )
    refuse_llm_work(monkeypatch)
    asked = users_own_key(monkeypatch, "openai")
    with block_live_network() as attempts:
        response = client.post(
            "/api/v1/media/process-documents",
            data={**_ANALYSIS, "auto_chunking_use_llm": "true"},
            files=[("files", ("d.txt", b"fake", "text/plain"))],
        )
    assert attempts == []
    assert_llm_quota_402(response)
    assert (asked, summarized) == (["openai"], [])


@pytest.mark.integration
def test_another_llm_stage_still_analyses_on_an_under_limit_users_own_key(
    client_with_single_user, quota_service_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under the limit, a request that also turns on another LLM stage still runs its analysis on the user's own key
    for the analysis provider; only the other stage keeps the server's key (TASK-15426). Chunking is off here, so
    the LLM chunk boundaries the request asks for make no call."""
    client, _usage_logger = client_with_single_user
    summarized: list[dict] = []
    monkeypatch.setattr(
        Summarization_General_Lib, "_summarize_via_adapter", lambda **kwargs: summarized.append(kwargs) or "Summary."
    )
    asked = users_own_key(monkeypatch, "openai")
    with block_live_network() as attempts:
        response = client.post(
            "/api/v1/media/process-documents",
            data={**_ANALYSIS, "auto_chunking_use_llm": "true", "perform_chunking": "false"},
            files=[("files", ("d.txt", b"Project Cedar opens on 24 January 2027.", "text/plain"))],
        )
    assert attempts == []
    assert response.status_code == 200, response.text
    assert [getattr(k["provider_credentials"], "api_key", None) for k in summarized] == ["sk-users-own-key"]
    assert asked == ["openai"]
```

  The other-stage test asks now (`asked == ["openai"]`, Decision 7); its 402 is unchanged. The under-limit test is review D-2's RED-first test: on the base source it fails `[None] == ["sk-users-own-key"]` (measured); with Step 3 it passes. It uses `auto_chunking_use_llm` because it is a `process-documents` form field (a field the form does not have is dropped, and the request then turns on no other stage), with chunking off so no chunk-boundary call is made.

- [ ] **Step 2: Run** the file. Expected on unchanged source: 16 failed, 6 passed (measured on `cb0e9bb0d4`, part of V7's RED run; the fold measured the same): own-key cases get the 402; `asked == []` everywhere, including the other-stage test; the under-limit test's analysis has no handle. The audio, video, PDF and ebook own-key cases expect one analysis each. With Step 3 but without the read in `_dispatch_to_api`, the `[videos-own-key]` case fails `[None] == ['sk-users-own-key']` (measured in the 2026-11-13 fold).
- [ ] **Step 3: Implement** in `media_processing_deps.py`, before `__all__` (`:674`; prototype text):
  - `media_analysis_provider(form_data)`: `None` when `claims_extraction_enabled(form_data)` or any of `enable_contextual_chunking`, `contextual_llm_model`, `context_window_size`, `auto_chunking_use_llm`, `auto_apply_template`, `chunking_template_name`, `perform_confabulation_check_of_analysis` is truthy, or `proposition_engine` is `llm`/`auto`; else `canonical_provider_name(str(form_data.api_name))`. The `ponytail:` comment of Decision 3.
  - `media_analysis_credentials(user_id, form_data)`: `return await users_own_key_credentials(user_id, canonical_provider_name(str(form_data.api_name)))` (imports inside the function; Decision 7: for `api_name`, not `media_analysis_provider(form_data)`). Docstring: resolved whatever other LLM stage the request turns on, so the analysis runs on the user's key whenever they stored one; `media_analysis_provider` still names no provider to the quota for such a request.
  - `admit_media_analysis(user_id, form_data)`: `handle = await media_analysis_credentials(user_id, form_data)`; `await enforce_llm_quota(user_id, provider=media_analysis_provider(form_data), own_key=handle is not None)` (a None provider refuses an over-limit user whatever `own_key` says); `bind_provider_credentials(handle)`, reached only when admitted. Imports inside the function (the module is imported by the media endpoints at startup). PR A's `enforce_llm_quota_on_own_key` does not fit here: it names to the quota the provider it resolves the handle for, and D resolves for `api_name` while naming `media_analysis_provider(form_data)`.
  - Add the three to `__all__`. In each of the seven endpoints replace `await enforce_llm_quota(current_user.id)` with `await admit_media_analysis(current_user.id, form_data)` and drop the `usage_quota_deps` import (`add.py:25`, `process_audios.py:40`, `process_documents.py:29`, `process_ebooks.py:36`, `process_emails.py:24`, `process_pdfs.py:37`, `process_videos.py:36`). In `add.py` the import (`:19`) becomes `from ...media_processing_deps import (admit_media_analysis, media_analysis_requested)`; in the six `process_*` endpoints put `admit_media_analysis` first in the existing sorted import list, before `get_process_<x>_form` (review D-6: placed after it, as the first prototype did, ruff reports a new `I001` in each of the six; measured: first, none).
- [ ] **Step 4: Run** the file → 22 passed; `$PY -m pytest tldw_Server_API/tests/lint/test_function_size_ratchet.py -q` → passes (each `process_*` endpoint at 356, 527, 412, 454, 427, 328, one under its ceiling, as on the base).
- [ ] **Step 5: Commit** `feat(media): media analysis runs on the user's stored key (TASK-15426)`.

## Task 3: The media ingest job's start check (ACs #2, #3)

**Files:** `media_ingest_jobs_worker.py`, `test_media_ingest_job_llm_quota_refusal.py`.

- [ ] **Step 1: Flip the job pin and add the admission test.** Import `users_own_key` (instead of `own_keys_for_every_provider`). `_create_job` takes `**options: Any`, merged last into the payload's `options`. Replace `test_a_stored_key_for_the_analysis_provider_does_not_admit_the_job` with `test_a_stored_key_for_another_provider_keeps_the_refusal`, parametrized `("stored", "options")` over `("anthropic", {})` and `("openai", {"perform_confabulation_check_of_analysis": True})` (ids `another-provider`, `own-key-and-another-llm-stage`; Decision 7: an own key with another stage on keeps the refusal; the confabulation check runs only after analysis, so a refused job makes no other-stage call): `asked = users_own_key(monkeypatch, stored)`, the `refused` fixture, the job created with `**options`; the job is stored with the refusal, `refused == []`, `asked == ["openai"]`. Add:

```python
@pytest.mark.asyncio
async def test_an_over_limit_owner_whose_analysis_runs_on_their_own_key_is_admitted_unmarked(job_dbs, monkeypatch) -> None:
    """The start check admits an over-limit owner whose analysis provider runs on their own key without marking
    the job admitted; the analysis runs on that key (TASK-15426)."""
    calls: list[tuple[Any, bool]] = []

    def _analysis(**kwargs: Any) -> str:
        calls.append((getattr(kwargs["provider_credentials"], "api_key", None), current_llm_call_context().prechecked))
        return "Cedar opens in January."

    monkeypatch.setattr(summary, "_summarize_via_adapter", _analysis)
    refuse_llm_calls(monkeypatch)
    asked = users_own_key(monkeypatch, "openai")
    manager = JobManager()
    row = _create_job(manager, job_dbs, _write_document(job_dbs, "Cedar"), "document")
    await _run_jobs(manager, 1)
    result = manager.get_job(int(row["id"]))["result"]
    assert result["status"] == "Success", result
    assert "analysis_error" not in result
    assert calls == [("sk-users-own-key", False)]
    assert asked == ["openai"]
```

  Extend `test_the_admission_ends_with_the_job`: `users_own_key(monkeypatch, "openai")`, record each analysis call's handle, and assert the one call used `sk-users-own-key` and that afterwards `current_llm_call_context().provider_credentials is None` (beside the existing `prechecked is False`). The test calls `_handle_job` directly, outside the SDK, on the test loop whose default executor drops the call state, and has no executor line of its own (review D-1): it fails until `_handle_job` installs the carrying executor itself (Step 3; measured in the fold on the pre-fold source: `[None] == ["sk-users-own-key"]`).
- [ ] **Step 2: Run** the file. Expected on unchanged source: 4 failed, 3 passed (measured on `cb0e9bb0d4`, part of V7's RED run; the same in the fold): both refusal cases (`asked == []`), the admission test, and the extended test (no handle). With Step 3 as before the fold (the handle for `media_analysis_provider`, no executor line), the other-stage case and the extended test still fail (measured in the fold).
- [ ] **Step 3: Implement** `_admit_or_refuse_analysis` (`:363-380`). Import `media_analysis_credentials` and `media_analysis_provider` in the worker's existing `media_processing_deps` import statement (`:17`), not `usage_quota_deps` directly: the layering ratchet counts `services -> api.v1.API_Deps` import statements (baseline 18, `layering_baseline.txt:204`) and a second statement fails `test_imports_follow_the_layering_contract` (measured: 19 > 18 on the first prototype; on `cb0e9bb0d4`, `test_layering_contract.py` passes with the three names in the one statement). Also import `bind_provider_credentials` and `current_llm_user_id` from `llm_call_context` (`:40`), and `ensure_llm_state_executor` from `Usage.llm_roots` on a line of its own after the `llm_quota_gate` import (`:41`), in sorted place, as the prototype has it (after the `llm_call_context` import it is unsorted, an `I001` the worker's existing `I001` hides):

```python
    if not media_analysis_requested(form_data):
        return None
    handle = await media_analysis_credentials(current_llm_user_id(), form_data)
    user_id = user_to_check()
    if user_id is not None:
        try:
            await check_user_async(user_id)
        except LLMQuotaExceededError as refusal:
            if handle is None or media_analysis_provider(form_data) is None:
                return refusal  # the server's key, or another LLM stage: refused as before, no key bound
        else:
            mark_llm_quota_prechecked()
    bind_provider_credentials(handle)
    return None
```

  A refused job binds nothing, so its other stages behave exactly as before (Decision 7). Extend the docstring: the analysis runs on the owner's own key when they stored one; an over-limit owner whose analysis runs on it, with no other LLM stage, is admitted without the mark, so each call is checked where its key is known. In `_handle_job` (`:513-520`), before the `with scoped_context(...), llm_call_context():` line (review D-1):

```python
    # The job's executor hops carry its call state (owner, admission, own-key handle) whoever calls the handler.
    ensure_llm_state_executor()
```

  The function-size ratchet freezes neither function (18 → 24 and 8 → 10 lines, measured).
- [ ] **Step 4: Run** `$FOCUSED_D` → 34 passed (measured on `cb0e9bb0d4`: 0:34, `-n 4`, 0 netprobe attempts); with `test_function_size_ratchet.py`, `test_layering_contract.py`, `test_llm_calls_import_isolation.py` and the two call-context and summarization files whole: 88 passed (1:39, `-n 4`, 0 netprobe attempts; the fold's 81 had no import-isolation lint).
- [ ] **Step 5: Commit** `feat(media): the ingest job's start check admits an own-key owner unmarked (TASK-15426)`.

## Task 4: Docs, verification, close-out, review, PR

- [ ] **Step 1: Docs** (one commit with the mirrors). Every edit starts from PR C's final text, as #3460 merges it (on `cb0e9bb0d4`), which keeps PR B's scoping of the raw-`api_key` sentences to the standalone `POST /api/v1/benchmarks/{name}/run` and PR A's narrowed BYOK sentence ("whenever it is a key that bills you as described above").
  - `Usage_Quotas.md`, after PR C's evaluation-run bullet (`:114`, ending `... and a run of a type that makes no LLM call;`), with Decision 7's sentence:

```text
  - media analysis on `/media/add` and the `/media/process-*` endpoints for audio, video, documents, PDFs, ebooks and email, and a media ingest job's check when it starts, for `api_name`, on which every analysis call of the request or job then runs. A request that also turns on another LLM stage is refused: claims extraction (itself or by the server's default), contextual chunking, LLM chunk boundaries or propositions, a chunking template, and the confabulation check. Under a limit such a request's analysis still runs on the user's own key; the other stage keeps the server's key. A job admitted over a limit on the user's own key is not marked admitted, so its calls are checked where their key is known;
```

    The "other boundaries" sentence (`:117`, `The other boundaries refuse an over-limit user whatever keys they have stored, because their LLM calls run on the server's key and never on a stored user key: media analysis (...) and a media ingest job's check when it starts.`) becomes `` `/media/ingest-web-content` and `/media/process-web-scraping` refuse an over-limit user whatever keys they have stored, because their scrapers pick LLM providers inside themselves and run on the server's key. `` (`Those calls count, so the refusal is the limit doing its job.` stays). PR A's parenthesis (`:119`) `(the study assistants, benchmark runs, watchlist output summaries and evaluation runs above)` becomes `(the study assistants, benchmark runs, watchlist output summaries, evaluation runs and media analysis above)`, keeping PR C's qualifier `except the runs and requests the list above says are refused`.
  - BYOK guide (Usage Quotas, the paragraph at `:97`):
    - the over-limit list: `evaluation runs, the judge-model evaluations` becomes ``evaluation runs, media analysis on `/media/add` and the `/media/process-*` endpoints for audio, video, documents, PDFs, ebooks and email (and media ingest jobs), the judge-model evaluations`` (refresh review M4: `/media/process-web-scraping` and `/media/process-code` are `/process-*` routes too);
    - PR A's narrowed sentence, as C left it, `The study assistants, benchmark runs, watchlist output summaries and evaluation runs use your own key for their provider whenever it is a key that bills you as described above (...)` gets media analysis: `The study assistants, benchmark runs, watchlist output summaries, evaluation runs and media analysis use your own key for their provider whenever it is a key that bills you as described above (...)` (the rest of the sentence unchanged);
    - the raw-`api_key` sentence (B's standalone scoping, C's RAG-pipeline and no-LLM runs) stays as C left it;
    - PR C's revoked-key sentence and its media sentence after it:

```text
An evaluation run already started keeps using the key it started with if you delete the key; cancel the run (`POST /api/v1/evaluations/runs/{run_id}/cancel`) to stop it. Media analysis never uses a key you stored: its calls run on the server's key, so over a limit it still refuses.
```

      become (Decision 6, the owner's decision; the claims sentence is refresh review M3, worded to the forms: only `/media/add`, `/media/process-emails` and the ingest-job submit route take `perform_claims_extraction`, Decision 3):

```text
An evaluation run or a media ingest job already started keeps using the key it started with if you delete the key; cancel it to stop it (`POST /api/v1/evaluations/runs/{run_id}/cancel`, `DELETE /api/v1/media/ingest/jobs/{job_id}`). Media analysis that also turns on another LLM stage (claims extraction, including the claims extraction the server turns on by default, contextual chunking and the like) is refused over a limit, and so are the web-scraping endpoints, whose scrapers run on the server's key. To turn the server's claims default off, send `perform_claims_extraction=false` with `/media/add`, `/media/process-emails` or a media ingest job (`POST /api/v1/media/ingest/jobs`); the `/media/process-*` endpoints for audio, video, documents, PDFs and ebooks take no such field, so while the server's default is on they refuse media analysis over a limit.
```

  - BYOK guide (Auditing and Updates, PR A's `last_used_at` clause as C extended it, `:207`): the parenthesis `(the study assistants, benchmark runs, watchlist output summaries and evaluation runs)` becomes `(the study assistants, benchmark runs, watchlist output summaries, evaluation runs, media analysis and media ingest jobs)`.
  - ADR-083, appended to PR A's amendment section (`## Amendment (2026-11-09)`, `:62`), after PR C's `**Evaluation runs**` bullet (`:68`, the file's last line; `<date>` is this PR's date, 2026-11-13 in the prototype):

```markdown
- **Media analysis and the media ingest job** (<date>): `api_name` on `/media/add` and the `/media/process-*` endpoints for audio, video, documents, PDFs, ebooks and email, and at a media ingest job's start. The handle is bound to the request's or job's call context, which threads, executors and queues already carry, and every `analyze` call for that provider that brings no key of its own runs on it. A request or job that also turns on another LLM stage names no provider and is refused over a limit; under a limit its analysis still runs on the user's key and the other stage on the server's. A job admitted on the user's own key is not marked admitted, so its calls are checked where their key is known. `/media/ingest-web-content` and `/media/process-web-scraping` keep the server's key and the refusal: their scrapers pick LLM providers inside themselves.
```

  - `bash Helper_Scripts/refresh_docs_published.sh`; only the two mirrors change (ADR-083 and the BYOK guide; measured on `cb0e9bb0d4`). `git diff --check` is clean (measured).
  - No other doc says media analysis stays on the server's key outside `Docs/Published`, design history and ADR-083's 2026-11-06 amendment, which the 2026-11-09 amendment supersedes and PRs A-C left as written (git grep).
- [ ] **Step 2: Wider set** with the PR: Task 0's counts plus the new tests, no new failing id (measured on `cb0e9bb0d4`, "Test commands").
- [ ] **Step 3: CI checks:** `PATH="$(dirname $PY):$PATH" PYTHONPATH=<plans dir> NETPROBE_LOG=<file> TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p netprobe_15427 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI tools/backlog-py/tests`; `$PY Helper_Scripts/ci/check_shard_coverage.py` (`new_uncovered=0`); `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` (OK); `PYTHONPATH=. $PY Helper_Scripts/ci/route_auth_ratchet.py` (baseline exact); `PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` (both OK); `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` (exit 0); `/opt/homebrew/bin/uvx bandit -ll` on the twelve source files. Expected, measured on `cb0e9bb0d4` with this PR's prototype: shard coverage `OK — no newly-unshared test files.`; triage `OK — no offenses beyond the baseline.`; route-auth `208 unauthenticated routes; baseline exact.`; OpenAPI fingerprint and path union both OK (no route signature changes); `check_backlog_task_ids.py` exit 0; `tests/Docs` with `tests/lint/test_function_size_ratchet.py`, `test_layering_contract.py`, `test_llm_calls_import_isolation.py` and `tests/CI/test_test_quality_ratchet.py` (`-n 4`, `-p netprobe_15427`): `297 passed` on base and prototype, 0 attempts (the `Docs/Published` mirror tests pass with the two refreshed mirrors); Bandit `-ll` on the twelve files: `No issues identified.` on base and prototype (5065 → 5151 lines of code, none `#nosec`). ruff on the 16 touched `.py` files against the base: no new finding (venv ruff 0.16.5, line numbers stripped: the same 2 findings on both, the worker's pre-existing `F401` and `I001`). Known environment-only results in `tests/lint` and `tests/CI` whole (from B's and C's runs; the shared venv lacks `setuptools` and `coverage`): `tests/lint/test_runtime_package_discovery.py` errors and `test_jobs_postgres_exports_recorded_coverage` (2 cases) fails; anything else is checked against a recent dev run.
- [ ] **Step 4: Close TASK-15426** (backlog-py; AC #1 was checked by PR A). Two commands, each with `--append-notes` (never `--notes`), the PR URL and the measured counts written in, the id written literally; then `git add -A backlog/tasks`. First, the PR's note and the ACs:

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-15426 \
  --check-ac 2 --check-ac 3 \
  --append-notes "PR D (<url>): media analysis on /media/add and the six /media/process-* endpoints, and a media ingest job, run on the user's own key (admit_media_analysis; the handle rides LLMCallContext, and Summarization_General_Lib._dispatch_to_api reads it for an analyze call that brings no key of its own). The job's start check admits an over-limit owner whose analysis runs on their own key without marking the job. With another LLM stage on, an under-limit user's analysis still runs on their key (owner, 2026-11-07) and an over-limit user is refused as before. _handle_job installs the carrying executor itself; llm_call_root drops a bound handle; no job payload, stored media metadata or log line carries it. /media/ingest-web-content and /media/process-web-scraping keep the server's key and the refusal. Built on PR C (#3460). Tests: <counts>. Bandit: no issues. Docs: Usage_Quotas.md, BYOK guide, ADR-083 amendment (mirrored)."
```

  Then the Definition of Done, the final summary and the status:

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-15426 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "TASK-15426 moved the four families onto the user's own stored key (owner, 2026-11-07: all four). When users_own_key_serves holds for a feature's provider (BYOK on, a provider that bills per key and reaches its vendor, allowlisted, an active key of the user's), the feature resolves that key once through ProviderCredentialRuntime (users_own_key_credentials: a handle only for credential_source 'user', never a team or org key) and passes the handle on every LLM call of the feature (for media, every analysis call; the other LLM stages a media request can turn on keep the server's key), so those calls are recorded credential_source = user and the per-user limits neither count nor check them. Each boundary names its provider to enforce_llm_quota with the handle as its answer (enforce_llm_quota_on_own_key; admit_media_analysis for media), so an over-limit user whose calls all run on their own key is admitted unmarked and everyone else gets the same 402. PR A (#3444): the flashcards and quizzes study assistants. The benchmark judge fix (TASK-15485, #3445) landed before PR B. PR B (#3452): benchmark runs (a raw api_key on the standalone route names no provider) and watchlist output summaries. PR C (#3460): evaluation runs (the runner's judge; rag_pipeline, raw api_key and no-LLM runs still refused). PR D (<url>): media analysis on /media/add and the six /media/process-* endpoints, and the media ingest job (the handle rides the LLM call context; the job is admitted unmarked). Owner decisions (2026-11-07): team and org keys are not used; a raw api_key keeps counting; watchlist output enrichment, /media/ingest-web-content, /media/process-web-scraping and rag_pipeline evaluation runs keep the server's key; a run or job keeps the key it resolved at its start (the BYOK guide says to cancel it); with another LLM stage on, a media request's analysis still runs on an under-limit user's key. Follow-ups: TASK-15486 (tests that reach a real host, among them the prompt-improvement test and the media tests' downloads), TASK-15491 (label choice and NLI score a failed judge call 0.0), TASK-15492 (an evaluator_model set to a model name is taken as the provider), TASK-<FDM> (own-key calls through analyze skip the DEFAULT_MODEL_<PROVIDER> fallback); the own-key helpers move into core/Usage when TASK-13950 cuts services -> api. Known skips: <none, or the PostgreSQL cases a down local server skipped, by id>." \
  -s Done \
  --append-notes "Closed by PR D (<url>): ACs #2-#3, the Definition of Done and the final summary."
```

  Dry-run on a throwaway copy of `backlog/` (`--cwd <copy>`, 2026-11-12, and again with the folded summary on 2026-11-13, the placeholders filled with dummies): both commands exit 0; ACs #2-#3 and DoD #1-#6 checked, the two notes appended after PR C's lines, the final summary set, `status: Done`, the file stays in `backlog/tasks/`; `check_backlog_task_ids.py <copy>` exit 0. Every flag exists in today's backlog-py (`task edit --help`: `--check-ac`, `--check-dod`, `--append-notes`, `--final-summary`, `-s`). The summary text has no backtick or `$`, so the double-quoted shell string passes it unchanged.
- [ ] **Step 5: Review, PR, merge:** the clean-session `/superpowers:requesting-code-review` on the branch diff with this plan and the task file (address every finding, re-run the focused set); then the PR body (Summary; a full `## Change summary` section recording the owner's waiver of 2026-11-07, "Yes, waive for all"; Test plan with the measured counts; Review Focus; ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`), mark ready, diff the PR's Frontend E2E failures against a recent dev run, and merge through `<plans dir>/pr_queue3.sh` (check `gh variable get MERGE_QUEUE` first: not set on 2026-11-12, so rebase only this PR when it is next, wait for the required statuses, then merge; the script reads dev's required checks from the ruleset). The PR body records the owner's decisions of 2026-11-07: a key revoked mid-run or mid-job is kept until the run or job ends (BYOK guide: cancel it); a raw `api_key` keeps counting; no team or org keys; watchlist output enrichment, `/media/ingest-web-content`, `/media/process-web-scraping` and `rag_pipeline` evaluation runs stay out of scope; "use their key" for an under-limit user with another LLM stage on (Decision 7). It lists, for the owner, the follow-ups the reviews found and no PR fixes: when TASK-13950 cuts `services -> api`, move `users_own_key_serves`, `users_own_key_credentials` and `media_analysis_provider` into `core/Usage` (review D-3: the worker's `media_processing_deps` import now carries credential resolution); own-key calls through `analyze` skip the `DEFAULT_MODEL_<PROVIDER>` fallback (review B-6, V11; filed as `TASK-<FDM>` in Task 0 Step 2); the 22 pre-existing live-DNS tests in the media wider set (Task 0 Step 4) are TASK-15486's. Its Review Focus ends with two known limits. (1) Decision 9: on a server whose only default model for a provider is `DEFAULT_MODEL_<PROVIDER>`, a user with a stored key gets a visible `Warning` with no media analysis (HTTP 200, no provider call), under a limit too, until `TASK-<FDM>` lands; D adds no media-only guard, as B and C add none on their paths. (2) Decision 3: with the server's `ENABLE_INGESTION_CLAIMS` on, the `/media/process-*` endpoints for audio, video, documents, PDFs and ebooks refuse an over-limit user's own-key analysis, because their forms take no `perform_claims_extraction` (they run no claims extraction); as before D, so no regression.

## Review fold (2026-11-07)

Folded from `review_plan_15426_cd.md` (D: 0 Blocker, 0 Important, 7 Minor), the A+B review's A-6 and B-6, and the owner's decisions of 2026-11-07. The pre-fold texts are `plan_15426_d.prefold.md` and `proto_15426_d.prefold.patch`. The fold was measured on `7a020da14a` with the folded A, B and C applied, outbound connections blocked (`-p netprobe_15427`, 0 attempts in every run); PostgreSQL was down, so no PostgreSQL case ran (`test_media_ingest_worker_content_scope.py` and the other PostgreSQL-fixture files' PostgreSQL cases were skipped).

| Finding / decision | Where it landed |
|---|---|
| D-1: the job's unmarked admission depended on its caller installing the carrying executor | Decision 2, Task 3 Steps 1 and 3, and the patch: `_handle_job` calls `ensure_llm_state_executor()`; the extended test lost its executor line. RED (pre-fold source): `[None] == ["sk-users-own-key"]`; GREEN with the line. |
| D-2: under-limit users with another LLM stage on got no handle | **Owner: "use their key."** Decision 7, Architecture, Review Focus 2, Task 2 Steps 1-3, Task 3 Steps 1-3, the docs and the patch: `media_analysis_credentials` resolves for `canonical_provider_name(api_name)`; the boundary still names `media_analysis_provider(form_data)` (None refuses an over-limit user); the worker refuses as before for another stage and binds nothing on a refusal. RED-first test `test_another_llm_stage_still_analyses_on_an_under_limit_users_own_key`: RED `[None] == ["sk-users-own-key"]`, GREEN 1 passed; the other-stage test now asks (`(["openai"], [])`); the job test gained `[own-key-and-another-llm-stage]` (RED `[] == ["openai"]`, GREEN). |
| D-3: the layering change adds debt | Task 4 Step 5: listed for the owner as a follow-up (move the helpers into `core/Usage` when TASK-13950 cuts `services -> api`). No code change. |
| D-4: `llm_call_root` did not reset `provider_credentials` | Decision 8, Task 1 Steps 1-4, Files and the patch: the `overrides` line and `test_a_background_root_never_inherits_the_handle_of_the_request_that_started_it` (RED `namespace(provider='openai') is None`, GREEN). |
| D-5: four endpoints tested for admission only | Task 2 Step 1 and the patch: a WAV with speech-to-text stubbed (a `transcribed` fixture), a `pymupdf` PDF with `pdf_parsing_engine=pymupdf`, and a generated EPUB through the real `process_epub`; `analyses` is 1 for audios, PDFs and ebooks (measured), videos stay 0. |
| D-6: six new ruff `I001` | Task 2 Step 3 and the patch: `admit_media_analysis` first in the sorted list (measured: no new ruff finding on any D file). |
| D-7: the ebook `[own-key]` request ran outside `block_live_network()` | Task 2 Step 1 and the patch: inside it, `attempts == []` asserted. |
| Dry-run gap: the BYOK sentence "... use your own key for their provider whenever you stored one" needs "and media analysis" | Task 4 Step 1 quotes it (with every other BYOK change). |
| A-6 (from PR A): `last_used_at` | Task 4 Step 1 completes the clause's parenthesis. |
| B-6 (from PR B): the `DEFAULT_MODEL_<PROVIDER>` fallback | Task 4 Step 5's follow-up list. |
| Pre-existing live-DNS tests (review, owner's list) | Task 0 Step 4 lists the 23 as known; Task 4 Step 5 lists them for the owner. |
| Owner: a run or job keeps the key resolved at its start | Decision 6; Task 4 Step 1 extends PR C's BYOK sentence to the ingest job with its cancel route. |
| Owner: a raw `api_key` keeps counting; no team or org keys; enrichment, the two web-scraping endpoints and `rag_pipeline` runs out of scope | Decision 4; Task 4 Steps 4-5 (final summary and PR body record them as answered; no open owner question). |
| Owner: the benchmark judge fix before B | Task 4 Step 4's final summary names `TASK-<JB>` (the old `TASK-<F_B>` follow-up is gone). |
| Found by the fold: "A job admitted that way" read wrongly after the D-2 sentence | Usage_Quotas text now says "A job admitted over a limit on the user's own key". |
| Fold measurements | Whole RED on layer C: 23 failed, 9 passed, 1 collection error (V7); `$FOCUSED_D` GREEN 34 passed; with the function-size and layering lints and the two call-context and summarization files whole: 81 passed. |

## Refresh (2026-11-13)

Refreshed onto its real base, `cb0e9bb0d4`: `origin/dev` `7f30182940` (PR A #3444, the judge fix #3445 and PR B #3452, all merged), then `git merge --no-ff` of PR C (#3460 at `ef0ef78ecc`, open); no conflict. The pre-refresh texts are `plan_15426_d.pre_refresh.md` and `proto_15426_d.pre_refresh.patch`. Every count was re-measured on that base, base and prototype, under `-p netprobe_15427`. No owner decision or ruling changed.

| What changed | Why |
|---|---|
| Header ("Spec", "Measured on", "Order"), Architecture | PR A's helpers are cited as merged code (`usage_quota_deps.py:91-122`, `:125-152`; `tests/helpers/llm_quota.py:63-97`), not `proto_15426_a.patch`; the base is dev + C, and C goes first because D's docs rewrite C's final sentences. Architecture names the merged helper; Task 2 Step 3 says why `enforce_llm_quota_on_own_key` does not fit. |
| V1-V6 | Re-cited. #3394 merged on 2026-11-09: every `process_*` boundary moved down a line and each frozen `process_*` endpoint now measures one under its ceiling (356, 527, 412, 454, 427, 328). V2 names the video library and `video_batch.py:118`; V4 adds where each other stage's call goes, so "the other stage keeps the server's key" is checked; V5 adds where the job builds its form and calls the check. |
| V7 | RED re-measured: 23 failed, 9 passed, 1 error, as in the fold. |
| V8 | Re-checked on 2026-11-12 with every open PR's file list and a trial merge of each overlapping head: #3451, #3455 and #3458 now touch five `process_*` endpoints near the dropped import, and merge with no conflict in D's files; #3269 and #2738 as before. |
| V9-V11, Review Focus 7 (new) | V9: the judge fix is untouched (git grep). V10 and the "Carry rule" constraint: where the handle lives, the one reader, and every payload, record, log and root it does not reach, with lines. V11: the `DEFAULT_MODEL_<PROVIDER>` measurement. |
| Decisions 6-7 | Cite the cancel route (`ingest_jobs.py:1033-1041`) and `usage_quota_deps.py:150`; text and rulings unchanged. |
| Plan-specific, Files, Tasks 1-3 | Re-cited lines; the frozen sizes and the layering baseline re-measured; the import-isolation lint added to Task 1 Step 4 and Task 3 Step 4. |
| Test commands, Task 0 Step 4, Task 4 Steps 2-3 | Re-measured. The wider set gains a second grep for the 38 callers the first misses; the old four-PR union run is history (pre-refresh copy). The 23 known live-DNS tests are 22 here (the Groq one belonged to another PR's set) and are TASK-15486's. |
| Task 0 Step 1 | Branch only once #3460 is in dev (`merge-base --is-ancestor`, or `gh pr view 3460` MERGED). |
| Task 0 Step 2 | The plan-file copy and its checks spelled out; an optional `TASK-<FDM>` filing, only if the owner says so. |
| Task 4 Step 1 (and the patch) | Every docs edit starts from C's final text: the BYOK sentence now reads "whenever it is a key that bills you as described above" (A's narrowing), and C's "Media analysis never uses a key you stored ..." is the sentence D rewrites; C's qualifier on the Usage_Quotas sentence is kept. ADR-083 dated 2026-11-13 in the prototype. Mirrors regenerated: the same two. |
| Task 4 Steps 4-5 | Two close-out commands, each with `--append-notes`, dry-run on a copy; the final summary names #3444, #3445, #3452, #3460 and this PR, and TASK-15486, TASK-15491 and TASK-15492 instead of the old placeholders; the merge step checks `MERGE_QUEUE`. |
| Prototype | Rebased: 21 files, +429/-75, as before. The source and test hunks applied unchanged (the endpoints at offset +1); the docs hunks were redone on C's text. Against the pre-refresh patch only the BYOK paragraph (both copies), the Usage_Quotas context and the ADR date differ; every source and test line is the same. |

Measured on `cb0e9bb0d4` (base → prototype), every run under `-p netprobe_15427`:
- `$FOCUSED_D`: RED `23 failed, 9 passed, 1 error` → GREEN `34 passed`; 0 netprobe attempts.
- The wider set (132 files): the 123 non-PostgreSQL files `32 failed, 3310 passed, 38 skipped, 2 xfailed, 3 errors` (load average about 65) → `3 failed, 3355 passed, 40 skipped, 2 xfailed`; the three are pre-existing on both trees, and the base's 29 other failures and 3 errors pass alone (`157 passed, 9 skipped, 1 xfailed`). The nine PostgreSQL files serially `253 passed` → `253 passed`. The same 22 netprobe node ids on both trees (TASK-15486's). `$FOCUSED_D` with the lints and the two files whole: `88 passed`.
- Function sizes: the six frozen endpoints 356, 527, 412, 454, 427, 328 → the same (ceilings 357, 528, 413, 455, 428, 329); `add_media_orchestrate` 815 → 815 (816); `add_media` 61 → 61, `_admit_or_refuse_analysis` 18 → 24, `_handle_job` 8 → 10, `_dispatch_to_api` 53 → 56, `llm_call_root` 18 → 18. The layering baseline is 18 for `services -> api.v1.API_Deps` on both; `test_layering_contract.py` passes on the prototype.
- `scope_predicate_baseline.txt`: lists none of the 21 files; no regeneration.
- Bandit `-ll` on the twelve source files: `No issues identified.` on both. ruff on the 16 `.py` files: the same 2 pre-existing findings on both.
- `tests/Docs` with the size, layering, import-isolation and test-quality lints: `297 passed` → `297 passed`, 0 attempts, with the two mirrors refreshed by `refresh_docs_published.sh`.
- Prototype checks: shard coverage OK, triage OK, route-auth `208; baseline exact`, OpenAPI fingerprint and path union OK, `check_backlog_task_ids.py` exit 0, `git diff --check` clean.

Not measured here: `tests/lint` and `tests/CI` whole and backlog-py's tests (only `tests/Docs`, the size, layering and import-isolation lints and the test-quality ratchet); the Frontend E2E journeys; the e2e and server-e2e files the second grep names (they need a live server).

Ruled (surfaced by the refresh as a question for the owner; controller ruling, 2026-11-13, recorded in "Review fold (2026-11-13)"):
- **Own-key analysis with no model in config (V11, review B-6).** On a server whose only default model for a provider is the `DEFAULT_MODEL_<PROVIDER>` environment variable (undocumented; the shipped `config.txt` sets `openai_model = gpt-4o` and `anthropic_model`), an own-key call through `analyze` gets no model and fails "Model is required"; on the server's key the variable supplies one. Measured for media analysis: HTTP 200, the item a `Warning` with no analysis and no provider call. It is pre-existing for every `analyze` handle caller (PR B's benchmark judge and summaries, PR C's RAG and quality judges); with this PR it also reaches media analysis, and for under-limit users with a stored key too (Decision 7). **Ruled: file it, do not fix it in D.** The fix belongs in `_summarize_via_adapter`, which every handle caller shares, so it changes B's and C's paths as well and needs tests across them; and a model the server picks for a user's key must still pass the provider-override model policy, which the runtime checks at resolve time against the configured model only (`provider_credential_runtime.py:1063-1083`; `_summarize_via_adapter` does not check again at dispatch, unlike the callers that pass `require_model=True`), so the fix needs its own review. Its reach is small (a server that clears the config model) and the failure is visible, not spend. Task 0 Step 2 files `TASK-<FDM>` unconditionally, with V11's facts. It is not an owner question: the gap is pre-existing, and B and C have the same gap on their paths. D adds no media-only guard (Decision 9).

## Review fold (2026-11-13)

Folded from `review_refresh_15426_d.md` (ready with fixes: 0 Blocker, 0 Important, 8 Minor) with the controller rulings of 2026-11-13. The pre-fold texts are `plan_15426_d.pre_fold.md` and `proto_15426_d.pre_fold.patch`. Base: `git fetch origin dev feat/own-key-evaluation-runs-15426`; #3460 is still open (head `ef0ef78ecc`), so the fold detached at `origin/dev` `7f30182940` and ran `git merge --no-ff --no-edit origin/feat/own-key-evaluation-runs-15426`: `0d33c9f989`, no conflict, its tree identical to `cb0e9bb0d4` (`git diff --quiet`). Every test ran under `-p netprobe_15427`.

| Finding / ruling | Where it landed |
|---|---|
| M1: `TASK-<FDM>` was conditional and had no command. **Ruling: fold; filed unconditionally.** | Task 0 Step 2 (a)-(d) as plan C's: the id lock and `task_next.sh` + 1; a literal `task create` (the review's title, `-l byok,llm,bug`, `-d` with V11's facts, the measured effect, who is affected and the policy constraint, the review's two `--ac`); the note on TASK-15426; the plan copy, the placeholder check and the commit. Task 0 Step 3 releases the lock once the PR's file list shows the task. V11, Files, Task 4 Step 4 (the summary names `TASK-<FDM>`, no conditional bracket) and Task 4 Step 5 (filed). "For the owner" is now "Ruled", a record of the ruling. |
| M2: D makes the gap a media regression; optional media-only guard. **Ruling: no guard in D.** | Decision 9 (the three reasons and the cost if wrong), V11's consequence, and Task 4 Step 5 (a known limit in the PR body's Review Focus). No code change. |
| M3: BYOK guide, the server's claims default | Task 4 Step 1's quoted sentence, the prototype's BYOK guide and its mirror. Field check: `perform_claims_extraction` is a form field of 2 of the 7 endpoints, `/media/add` (`get_add_media_form`, `media_add_deps.py:105`; also the ingest-job submit route's form, `ingest_jobs.py:602`) and `/media/process-emails` (`get_process_emails_form`, `media_processing_deps.py:558`); the forms of audios, videos, documents, PDFs and ebooks lack it (their `Process*Form` models inherit the field, but their form dependencies never set it, so it stays None and the server's default decides). The review's sentence would be false for five endpoints, so the sentence names where the field works and says the other five refuse while the default is on. Decision 3 records it, with the fact that the six `/process-*` endpoints never run claims extraction. |
| M4: "the `/media/process-*` endpoints" too wide | Task 4 Step 1's over-limit list, the prototype's BYOK guide and its mirror: "for audio, video, documents, PDFs, ebooks and email". |
| M5: "every LLM call it makes" | Task 4 Step 4's final summary: "every LLM call of the feature (for media, every analysis call; the other LLM stages a media request can turn on keep the server's key)". No backtick or `$` added. |
| M6: imports in Task 1 Step 1 | The three code blocks show their imports as the prototype has them: `import contextvars` and the two names at module level in `test_llm_call_context.py`, `SimpleNamespace` inside each test; the summarization test imports `contextvars`, `SimpleNamespace` and `bind_provider_credentials` inside the test. The prototype is unchanged. |
| M7: incomplete code blocks | Task 2 Step 1: `test_another_llm_stage_keeps_the_refusal_for_an_own_key_user` with its setup written out (as the prototype). Task 3 Step 3: the `ensure_llm_state_executor` import after the `llm_quota_gate` import. The prototype is unchanged. |
| M8: videos can be pinned. **Fold if it works: it works.** | Task 2 Steps 1-2, Decision 2, the Refresh's "Not measured" and the prototype's test file. The `transcribed` fixture also stubs `Video_DL_Ingestion_Lib.perform_transcription` (as `test_video_summary_service_prompt.py` does), and videos' `analyses` is 1. The test's `b"fake"` video passes the endpoint's validation without an ffmpeg-made fixture: puremagic (a core dependency in `pyproject.toml`, preferred over python-magic) raises `PureError`, and the validator falls back to the `.mp4` extension (`Upload_Sink.py:668-703`; logged `MIME magic detection failed; using extension fallback (error_type=PureError)`). The local-file video path runs no ffmpeg or ffprobe outside `perform_transcription`. No fixture is needed; the repo's `sample.mp4` (45 MB) is not used and no binary is added. |
| Found by the fold | The six `/media/process-*` endpoints never run claims extraction (only `/media/add` and the job: `persistence.py:4782`, `:5586`), so on five of them the server's claims default refuses an over-limit own-key user with no opt-out. Pre-existing in effect (they refused that user before D too); recorded in Decision 3, the BYOK sentence and the PR body's known limits. Not changed here: narrowing the denylist per endpoint is a design change. |

Re-measured on `0d33c9f989` (base) and `0d33c9f989` + the folded prototype, every run under `-p netprobe_15427`, 0 netprobe attempts in each:
- The claims-default probe (a temporary test, deleted): with `ENABLE_INGESTION_CLAIMS` on and an over-limit own-key user, `/media/process-documents`, `/media/process-emails` and `/media/add` all answer 402; with `perform_claims_extraction=false`, `/media/process-documents` still 402, `/media/process-emails` and `/media/add` 200 with one analysis on `sk-users-own-key`.
- Videos (M8): `-k videos`, `2 passed` with ffmpeg on `PATH` and `2 passed` with `PATH` limited to the venv and `/usr/bin:/bin:/usr/sbin:/sbin`.
- RED, `$FOCUSED_D` on the base's source (serially, `--continue-on-collection-errors`, `test_llm_call_context.py` whole): `23 failed, 9 passed, 1 error` (1:56), as V7; the two videos cases fail `[] == ['openai']`. On the prototype without the read in `_dispatch_to_api`: `[videos-own-key]` fails `[None] == ['sk-users-own-key']`, `[videos-server-key]` passes.
- GREEN, `$FOCUSED_D` (`-n 4`): `34 passed` (1:44). With the size, layering and import-isolation lints and the four test files whole: `88 passed` (2:22).
- `test_quality_triage.py --enforce`: `OK — no offenses beyond the baseline.`
- `tests/Docs` with the size, layering, import-isolation and test-quality lints (`-n 4`), the mirrors refreshed by `refresh_docs_published.sh`: `297 passed` (4:12); the script changes only the ADR-083 and BYOK mirrors, and a second run changes nothing.
- ruff 0.16.5 on `test_media_llm_quota.py`: `All checks passed!`. `git diff --check` on the prototype tree: clean. The prototype stays 21 files, +429/-75 (only the BYOK guide, its mirror and `test_media_llm_quota.py` differ from the pre-fold patch).
- Task 0 Step 2 (b)-(c) and Task 4 Step 4's two commands, extracted from this plan verbatim, on a throwaway copy of `backlog/` with TASK-99902 for `<FDM>` (removed afterwards): all exit 0; the new task has labels `byok`, `llm`, `bug` and two ACs; TASK-15426 gets both notes, ACs #2-#3, DoD #1-#6, the summary and `Done`; `check_backlog_task_ids.py` exits 0 on the copy and on the worktree.

Not re-measured (the fold changes nothing they cover): the wider set, Bandit, shard coverage, route-auth, the OpenAPI check and the function sizes; the fold touches no source file.
