# LLM Quota Refusals Reach the Caller, and the Dead RAG Completeness Metric Is Deleted (TASK-13638, TASK-13635)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Independent review and dry run (2026-10-18): 1 Blocker, 4 Important, 9 Minor; all folded in.

**Goal:**
- TASK-13638: no caller turns an LLM quota refusal (`LLMQuotaExceededError`) into output that looks like a result. Each caller lets the refusal through, or keeps what it already has and records the refusal. The workflow engine does not retry a refused step (Review M3). The operator page says what each caller does.
- TASK-13635 (Review I4): delete `RAGCustomMetrics.evaluate_response_completeness` and `evaluate_all_metrics`, its only caller. Nothing in the app, the WebUI or the tests outside their own sanitizer file calls either, and the LLM path cannot reach a provider. The task's ACs are amended to match (Decision 4). Appendix A keeps the alternative: grade with a provider the caller names.

**Architecture:**
- LLM usage accounting PR 2b (TASK-13437) raises `LLMQuotaExceededError` (an `HTTPException`, status 402, `retryable = False`, `failure_code = "llm_quota_exceeded"`) at the call. Design §4b: an endpoint that lets `HTTPException` through answers 402, a workflow step fails, and a job fails without retry.
- The callers below catch it in a broad `except` and carry on. The fix is the pattern PR 2b already uses (`Writing/manuscript_analysis.py:150`, `services/outputs_service.py:701`): `except LLMQuotaExceededError: raise` placed before the broad clause.
- **What a refusal does to a fan-out** (Review I3; measured with `rvq_probe_map.py`, one refused item and the others sleeping 0.1 s):
  - `parallel` with `fail_fast` off (the default) and the FVA batch (`process_batch`) wait for every sibling, then raise the refusal.
  - `map` and `parallel` with `fail_fast` on raise the first refusal at once. `asyncio.gather` does not cancel the other awaitables, so siblings already started, and those still queued behind the concurrency limit, run on after the step has failed. In the probe, `map` items 1-5 and `parallel(fail_fast)` steps 1-3 all finished after the step had raised.
  - The chunked transcript summary and a Prompt Studio evaluation are sequential loops. They stop at the first refusal.
  - A sibling's LLM call made after a refusal is refused too, before it reaches a provider: the check runs before the provider method (`LLM_Calls/usage_accounting.py:323-341`). It costs one quota counter read, and no provider spend or usage row. A non-LLM sibling runs to completion with its side effects, which is old behaviour for any sub-step that raises.
  - Cancelling siblings (`asyncio.TaskGroup`) is a follow-up that Task 9 files. This plan does not change the engine's fan-out.
- Three things are not a simple re-raise:
  - The benchmark run endpoints get the §4a boundary check (Decision 2): `await enforce_llm_quota(...)` first in each handler, as the evaluations module does (Review M6).
  - A media item that reuses an earlier transcript keeps the transcript and records the refusal, with the code `llm_quota_exceeded` (Decision 3, Review M4).
  - The workflow engine treats a `402` as not retriable (Decision 1, Review M3).
- TASK-13635 (Review I4): PR B deletes the two methods, the now-unused `analyze` and `logger` imports, and the sanitizer test file that covers only them. It adds no code.

**Tech Stack:** Python 3.12, FastAPI, asyncio, pytest, loguru.

**Spec:**
- `backlog/tasks/task-13638 - Callers-that-turn-an-LLM-quota-refusal-into-degraded-output.md`.
  - AC1: each listed caller lets the refusal through or records that it was refused.
  - AC2: the operator page lists what each one does.
  - Its implementation notes add the transcript-reuse loop, the Explainer node's error code and the HyDE legacy branch.
- `backlog/tasks/task-13635 - RAG-custom-completeness-metric-never-calls-a-provider.md`.
  - Today's AC1: the metric calls a configured provider through `analyze` with the right arguments, off the event loop.
  - Today's AC2: with no provider configured, it keeps the heuristic and says so in the result.
  - Under Decision 4's recommendation, Task 10 Step 0 replaces both ACs through backlog-py (`--remove-ac` / `--ac`; Review I4).
- Governing design: `Docs/Design/2026-10-09-llm-usage-accounting-design.md` §4a, §4b and §5. Operator page: `Docs/Operations/Usage_Quotas.md`.

## Delivery: two PRs

| PR | Task | Branch | Plan tasks |
|---|---|---|---|
| A | TASK-13638 | `fix/llm-quota-refusal-callers` | 1–9 |
| B | TASK-13635 | `fix/delete-rag-completeness-metric` (Appendix A: `fix/rag-completeness-metric-provider`) | 10–11 |

They are two PRs because they share no code or test file; both edit the design doc, in different sections. Decision 4 can hold PR B without holding PR A. PR A is one task with one operator-page change, so it stays one PR, with one commit per plan task.

## What was measured

Re-measured on `origin/dev` `7c94a2efb1` for the review's changes (Review B1, I1–I4, M3–M7). The first draft was measured on `e1f7770953`. The 11 commits between the two touch none of the plan's source files, and every line number below matches `7c94a2efb1`; they do add the test-quality ratchet (`tests/CI/test_test_quality_ratchet.py`, run by `backend-required.yml:261`) and its baseline.

All measurements come from probes and RED runs, recorded in the scratchpad (`probe_qc.py`, `probe_qc.out`, `probe_claims_usage*.out`, `qc_red_*.out`, `rvq_probe_*.py`, `rq_red*.log`, `rq_green1.log`, `rq_*suite*.log`). Each probe forces the LLM call to raise `LLMQuotaExceededError(category="llm_tokens_month", ...)`.

| # | Caller | What it does today with a refusal (measured) | Decision | Change |
|---|---|---|---|---|
| 1 | Workflow content steps: flashcard, quiz, outline, glossary, mindmap, slides, report, newsletter, diagram (`Workflows/adapters/content/generation.py`) | Returns `{"error": "<step>_generate_error", ...}`. The step succeeds and the run goes on. | Let through (Decision 1) | `except LLMQuotaExceededError: raise` before each of the 10 `except _GENERATION_NONCRITICAL_EXCEPTIONS` (that tuple includes `Exception`) |
| 2 | `notes_studio_generate`, same file (not in the ticket's list) | With a provider and model set: returns the deterministic fallback payload plus `warning: notes_studio_generate_warning`. | Let through | Same; one of the 10 sites |
| 3 | NLP steps: keyword (llm method), sentiment, topic, entity (`text/nlp.py`) | Error outputs. Sentiment returns `{"sentiment": "neutral", "score": 0.0, ...}`, which reads as a real result. | Let through | 4 sites |
| 4 | `reference_parse`, `literature_review` (`research/bibliography.py`) | Error outputs | Let through | 2 sites |
| 5 | `image_describe` (`content/image.py`) | `{"description": "", "error": "image_describe_error"}`. `image_gen` uses an image backend, not a chat provider, and is not gated. | Let through | 1 site |
| 6 | `summarize` (`content/summarize.py`) | `{"error": "summarize_error"}`. `analyze` itself lets the refusal through. | Let through | 1 site |
| 7 | `translate` (`llm/translate.py`) | `{"text": "Bonjour", "provider": "none", "fallback": True}`: the input is passed on as its own translation. | Let through | 1 site |
| 8 | `evaluations` step (`evaluation/eval.py:286`), around G-Eval, RAG and response-quality evaluation (Review I2) | `{"error": "evaluations_error"}`. The step succeeds and the run goes on (probe `rvq_probe_wf_eval.py`). G-Eval itself lets the refusal through (`ms_g_eval.py:537-538`). | Let through | 1 site |
| 9 | `parallel` (`control/flow.py`), `fail_fast` off, the default | The refused step's result is `None` and `errors` is `None`: the refusal is dropped. | Let through | Re-raise a refusal found in the `gather` results |
| 10 | `map` (`control/flow.py`) | Raises it already; siblings run on (Architecture, Review I3) | No change | None; follow-up filed in Task 9 |
| 11 | `retry` step (`control/state.py`) | Around an error output it calls the step 4 times (sleeps 1, 2, 4 s) and returns `success: True`. A raised refusal is not retried. | No change needed | Once 1–8 raise, nothing is retried. A test pins it. |
| 12 | The engine's own step retry (`retry:` on a step; `llm`, `prompt`, `tts` and `webhook` steps default to 1, others to 0) (Review M3) | `_reason_code_from_error` yields `"402"`, which is not in `_NON_RETRIABLE_REASONS` (`engine.py:85-91`). A step with `retry: 3` is attempted 4 times, each refused without a provider call (RED: `assert 4 == 1`). | Not retried (Decision 1) | Add `"402"` to `_NON_RETRIABLE_REASONS`, so `step_retry_suppressed` fires |
| 13 | `_fva_claims_analyze_call` (`Claims_Extraction/claims_service.py`), shared by `/claims/verify/fva`, generated-artifact verification (`artifact_verification.py`) and the workflow `claims_extract` step | Returns `""`, and the verifier decides without the LLM. Probe: "The sky is green." against the evidence "The sky is blue." came back **VERIFIED** ("Moderate evidence support"). The refused call also wrote a `claims_verify` usage row with status 200 and the estimated prompt tokens, so it counted against the user. | Let through | Re-raise before `except _CLAIMS_NONCRITICAL_EXCEPTIONS` (that tuple holds `HTTPException`) |
| 14 | `FVAPipeline.process_batch` (`Claims_Extraction/fva_pipeline.py`) | `gather(return_exceptions=True)` drops the refused claim: `results=[]`, `total_claims=1` | Let through | Re-raise a refusal found in the results |
| 15 | `TestRunner.run_test_case` (`prompt_studio/test_runner.py`) | Non-strict: stores a run with `{"error": "The chat service provider is currently unavailable.", "error_code": "provider_unavailable"}`, scored 0.0. Strict (the `/test-cases/run` endpoint and optimizers): raises `SanitizedProviderStreamError(provider_unavailable)`. | Let through | Re-raise before `except Exception` |
| 16 | `PromptExecutor._call_llm` (`prompt_studio/prompt_executor.py:559-568`), the optimizers' own LLM calls (Review I1) | Once the provider is dispatched it rewrites any error, the refusal included, as `RuntimeError("Provider returned an error response")` (RED: that `RuntimeError`). MIPRO's `_call_internal_llm` has no `try`, and `_generate_refinement` re-raises in a job, so the job sees a `RuntimeError`. | Let through | Re-raise before `except Exception` (line 559) |
| 17 | `PromptExecutor.execute_prompt` (`prompt_executor.py:192`), behind `POST /prompt-studio/prompts/execute` (Review I1) | Returns `success: False`, and the endpoint (`prompt_studio_prompts.py:855-861`) answers **502** "Upstream provider request failed." (RED: that 502) | Let through | Re-raise before `except Exception` (line 192) |
| 18 | `EvaluationManager.run_evaluation_with_existing_record` (`prompt_studio/evaluation_manager.py`) | Every case is scored 0.0. The evaluation ends `completed`, `average_score 0.0`. | Let through, after marking the record `failed` with `error_message "llm_quota_exceeded: <message>"` | New clause before `except Exception` |
| 19 | Background `run_evaluation_async` (`endpoints/prompt_studio/prompt_studio_evaluations.py`) | After 18, the refusal would escape the `BackgroundTask`: its except tuple has no `HTTPException`. | Catch and log; 18 already wrote the record | New clause |
| 20 | `_bounded_optimization_error` (`prompt_studio/services/jobs_worker.py`) (corrected, Review I1) | A refusal arrives as `provider_unavailable`, **retryable**, by either path: a scoring call (strict `run_single_test`, row 15) or an optimizer's own call (row 16). Probe `rvq_probe_executor.py`: `provider_unavailable retryable True`. Once rows 15 and 16 let it through, it would map to `prompt_studio_job_failed`. | Record `llm_quota_exceeded`, not retryable | New branch. With rows 15, 16 and 20, the probe gives `llm_quota_exceeded retryable False`. |
| 21 | `Evaluations/evaluation_manager.evaluate_custom_metric` (benchmarks) | `{"score": 0.0, "explanation": "Evaluation failed: 402: ..."}`. The benchmark loops count it as a success, so it pulls down `average_score`. | Let through | Re-raise before `except Exception`. The run loops then record the item as an error, counted under `failed` and left out of `average_score`. |
| 22 | Benchmark run endpoints (`/evaluations/benchmarks/{name}/run`, `/benchmarks/{name}/run`) | Every item is refused and scored 0.0 (row 21). The run completes. | §4a boundary check (Decision 2) | `await enforce_llm_quota(getattr(current_user, "id", None))` first in both handlers (Review M6) |
| 23 | `extract_character_memories` (`Character_Chat/modules/character_memory_extraction.py`) | Returns an empty result. The manual endpoint answers `extracted: 0`. | Let through | Re-raise before `except Exception`. The manual `/characters/{id}/memories/extract` answers 402. Automatic extraction runs as a background task of the boundary-checked character chat reply, so it is admitted, and its own `except Exception` logs anything else. |
| 24 | Transcript reuse (`Ingestion_Media_Processing/persistence.py`, `_synthesize_reused_transcript_analysis_sync`) | 20 chunks: 20 refused calls, 21 warnings, status `failed`, transcript kept. The only record is a free-text warning holding a dict repr (Review M4). | Keep the transcript, record the refusal with its code, stop at the first (Decision 3) | Re-raise in the chunk loop. `_populate_reused_transcript_summary` gets its own clause that also sets `video_lite_summary_error = "llm_quota_exceeded"`, and `_merge_video_lite_summary_safe_metadata` stores it as `video_lite.summary_error` (Review M4). |
| 25 | Explainer node expansion (`Explainer/jobs.py`) | Node error metadata says `generation_failed`, while the job says `llm_quota_exceeded` | Record the refusal's code | `error=getattr(exc, "failure_code", None) or "generation_failed"` |
| 26 | HyDE legacy branch (`RAG/rag_service/unified_pipeline.py:4133`; taken when there is no credential runtime) | The sync `generate_hypothetical_answer` runs on the loop thread (same thread id in the RED run), so the quota check is skipped (fail open, design §5) | Move it off the loop | `await asyncio.to_thread(...)`. A refusal there then degrades to the heuristic answer, like any provider failure of this optional stage. |
| 27 | `dispatch_prompt_improvement` (`Prompt_Management/prompt_improvement_dispatch.py:215`), behind `POST /prompts/improve` (not in the ticket's list; clean-session review M3) | `except Exception` maps the refusal through `_map_provider_exception` to `provider_unavailable`, `retryable: true`. The route has no boundary check. | Let through | Re-raise before `except Exception`; the route answers the 402 |

## Prototype check (the whole revised plan applied in a scratch worktree on `7c94a2efb1`)

- **Every new test:** RED on dev for the reason each task gives, and GREEN on the prototype.
  - PR A: 41 test ids: 23 workflow (22 adapter cases and the engine test) and 18 others (claims 3, Prompt Studio 7, benchmarks 3, memory 1, reuse 1, Explainer 2 including the control, HyDE 1). The first draft said "17 others"; it was 16 (Review M9), and Review I1 adds 2.
  - PR B: deletes 2 tests and adds none.
  - One GREEN run of all 41, with the rest of the four benchmark-route files, `test_engine_retry_classifier.py` and `test_claims_service_fva_analyze.py`: `68 passed, 1 failed` (`rq_green1.log`). The failure was the review's suggested `assert db.updates == []` (Review M5); with Task 3's assertion that file gives `2 passed`.
- **Neighbourhood suites, PR A:**
  - `tests/Workflows`, whole: `1682 passed, 6 skipped` (it includes every test that names `_is_retriable_error`, `_reason_code_from_error` or `step_retry_suppressed`, since Review M3 changes `_NON_RETRIABLE_REASONS`).
  - `tests/prompt_studio`, all 93 files: no failure or error (pytest progress tally: 1124 passed, 17 skipped; `rq_suiteA.log`). That run was then stopped early in `tests/Evaluations`, with other sessions holding the load average near 80.
  - Targeted at what the revision touches, in one run: the four benchmark-route files, `Evaluations/unit/test_evaluations_identity_route_bindings.py`, `Evaluations/unit/test_evaluation_manager.py`, `Evaluations/unit/test_eval_runner.py`, `Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py`, both character memory files, `tests/Explainer`, the three HyDE files, `Media/test_media_llm_quota.py`, `MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py` and `tests/Usage`: `507 passed, 1 skipped` (4 min 50 s; `rq_suiteC.log`).
  - `tests/MediaIngestion_NEW/unit`, with `tests/RAG`: see PR B.
  - Not re-run for the revision: `tests/Claims`, `tests/Claims_Extraction`, `tests/Flashcards` and the rest of `tests/Evaluations` (without `property/` and `integration/`). The review's dry run ran them with the same claims and evaluation-manager changes: `1 failed, 1705 passed, 1 skipped` for the first three, the failure a Hypothesis `FailedHealthCheck` that passes alone, and no new failure in `Evaluations`. Of what they import, the revision changes only the benchmark handlers, which the targeted files cover.
- **Neighbourhood suites, PR B:** `tests/RAG` (without `test_restricted_postgres_media_retrieval.py`) and `tests/MediaIngestion_NEW/unit`, in one run on the prototype with both PRs applied: `1274 passed, 8 skipped, 1 xfailed` (8 min 44 s; `rq_suiteB.log`). The two `eval_runner` files that import `get_custom_metrics` passed in the targeted run above.
- **Other checks** (all on the prototype):
  - `Helper_Scripts/ci/test_quality_triage.py --enforce`: exit 0, "OK — no offenses beyond the baseline". `tests/CI/test_test_quality_ratchet.py`: `1 passed` (Review B1).
  - `scope_predicate_ratchet.py`: "28 conditional scope predicates; baseline exact." Regenerating the web scraping import inventory changes nothing under `Docs/Design/`.
  - The OpenAPI drift gate exits 0 ("fingerprint matches", "path union matches").
  - Ruff on the 38 touched Python files: 35 findings on `origin/dev`, the same 35 after, none new (`rq_ruff.sh`).
  - `tests/Docs` with every doc edit of both PRs and the refreshed mirror: `212 passed`. `refresh_docs_published.sh` changes one mirrored file, the benchmark guide (Review M7).

## Global Constraints

- **Python:** `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. **Tests:** `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q <paths>`.
- **Branches** come from `origin/dev`, and both PRs target `dev`. Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never `git stash`.
- **Test first:** each task's RED run happens on unchanged source and fails for the reason given.
- **The test-quality ratchet is a required gate (Review B1).** Every PR's verification runs `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` (expected exit 0) and `tldw_Server_API/tests/CI/test_test_quality_ratchet.py` (expected `1 passed`). A new test needs at least one `assert` on a result, not only on `.status_code` or on its own stub collector.
- **The pattern, verbatim:** `except LLMQuotaExceededError:` then `raise`, placed directly before the broad clause.
  - No caller's other error handling changes. `_GENERATION_NONCRITICAL_EXCEPTIONS`, `_CLAIMS_NONCRITICAL_EXCEPTIONS` and the rest keep their members.
  - A provider outage keeps today's degraded output.
- **Imports (Review M1).** Use `from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError`, or `from ....core.exceptions import ...` in the Prompt Studio core modules, which import relatively. Ruff's isort is case-insensitive, so `...core.exceptions` sorts after `...core.Chat` and `...core.DB_Management` and before `...core.Explainer`, `...core.http_client`, `...core.LLM_Calls`, `...core.Notes` and `...core.testing`. Each task names the import line the new one goes next to.
- **Lint (the "lint comparison" in the tasks):** `$PY -m ruff check --output-format concise <touched .py files>` reports nothing that the same files from `origin/dev` do not. Export the dev copies with `git show origin/dev:<path>` into a temporary tree holding the repo's `pyproject.toml`, run the same command there, and diff the two outputs with the line and column numbers stripped (`rq_ruff.sh` in the scratchpad does this; measured on the prototype: 35 findings on dev, the same 35 after). Do not run `ruff --fix`: 12 of the touched files already have findings on dev (unsorted import blocks in `eval.py`, `image.py`, `claims_service.py` and others), and fixing them would rewrite lines this plan does not own.
- **No new exception type, setting, or `limits.*` key.** No change to `quota_checks`, `llm_quota_gate`, the provider base class or `chat_service`.
- **Line-number baselines:** in `Helper_Scripts/ci/scope_predicate_baseline.txt`, the only touched file is `unified_pipeline.py`, at line 3889. The inventory lists it at lines 3384 and 3413. The HyDE edit is at line 4133, after all three.
  - Verified on the prototype: `scope_predicate_ratchet.py` reports "baseline exact".
  - Regenerating `Docs/Design/web_scraping_refactor_import_inventory.json` and `Docs/Design/WebScraping_Refactor_Import_Inventory.md` changes neither file.
  - Task 8 re-runs both checks.
- **Docs:** `Docs/Operations/Usage_Quotas.md`, `Docs/Design/2026-10-09-llm-usage-accounting-design.md`, `Docs/Design/service-prompt-inventory.md` and `Docs/Design/RAG_Pipeline_Evaluation.md` have no `Docs/Published` mirror (`refresh_docs_published.sh` copies only `Operations/Env_Vars.md` from Operations, and nothing from `Docs/Design`). `Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md` is mirrored: Task 9 runs `bash Helper_Scripts/refresh_docs_published.sh` and commits the `Docs/Published` diff (Review M7).
- **Doc edits are anchored on quoted text, not line numbers** (Review M8). Other plans edit the same pages; see "Conflicts with other plans".
- **Backlog:**
  - `tools/backlog-py` only (`PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task ...`), `--append-notes`, explicit ids for new tasks, and `git add -A backlog/tasks`.
  - New ids come from the AGENTS.md recipe, extended to page through open PRs with more than 100 files (`gh pr list --json files` stops at 100 per PR). Task 9 Step 4 has the command.
  - Mark each task `In Progress` in its PR's first commit and `Done` in its close-out commit.
- **Merging:** merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset.

## Conflicts with other plans (Review M8)

Measured against `plan_audio_analysis.md` and `plan_llm_usage_recording.md` as they stand in the scratchpad on 2026-10-18.

| Place | This plan | Other plan | How to merge |
|---|---|---|---|
| `Usage_Quotas.md:95`, the job bullet (one long line) | PR A replaces the sentence "A media item that reuses an earlier transcript gets no summary." and adds Prompt Studio and Explainer sentences after it | `plan_audio_analysis.md` Decision 1 rewrites the audio-job clause of the same line ("A WorkerSDK job and an audio analysis job ... record the error code `llm_quota_exceeded`"); `plan_llm_usage_recording.md` inserts a new bullet right after the line ("The in-process claims rebuild ...") | Git conflicts on the line for whoever lands second. Take the landed line and re-apply your own sentence edit to it, found by its quoted text. Keep the recording plan's new bullet directly after it. The meanings agree: each keeps the artifact and records the refusal. |
| `Usage_Quotas.md:99` and `:100-101` | PR A replaces line 99, the "Degraded, not refused" bullet | `plan_llm_usage_recording.md` replaces line 100 (the MCP bullet) and deletes line 101 ("Not recorded today") | Adjacent hunks conflict textually. Keep this plan's line 99 text, the recording plan's new MCP bullet, and its deletion of line 101. No sentence is shared. |
| `Usage_Quotas.md:93` | PR A replaces the workflow bullet | `plan_llm_usage_recording.md` names this edit (its Review M6) but does not change the line | No conflict. |
| Design doc, end of file | PR A appends "Revision of 2026-10-18, TASK-13638" after "Review changes" | `plan_llm_usage_recording.md` appends "Revision of 2026-10-18, TASK-13585 and TASK-13610" at the same place | Keep both paragraphs, in merge order, each with its own heading line. |
| Design doc §4b | PR B rewrites the last bullet's sentence about the RAG custom metric | `plan_llm_usage_recording.md` replaces the §4b sentence "MCP tool calls carry no user, so they are neither recorded nor checked (§1)." | Different sentences: re-apply by quoted text. |
| Design doc §4a table | PR A adds a "Benchmark runs" row after "Evaluation runs" | Neither other plan touches the table | No conflict. |
| `claims_service.py` | PR A edits the import at line 71 and the `try` at 610-615 | `plan_llm_usage_recording.md` edits lines 634, 4599 and 4649 | No overlap; a rebase applies cleanly. |
| `plan_summary_error_codes.md` | PR B deletes `rag_custom_metrics`' only `analyze` call | Its new `SummaryProviderError` codes would have reached that call | No shared file. Its reader-audit table needs no `rag_custom_metrics` row once PR B lands. |

## Decisions for the owner (recommendation first)

1. **Workflow steps that make an LLM call (rows 1–9 and 12).**
   - **Recommended:** the refusal fails the step with the 402, and the engine does not retry it (Review M3). The run fails too, unless the step's `on_failure` routes elsewhere. This is what an `llm` step does today, minus its retry.
   - Why:
     - Today's outputs look like results: a neutral sentiment, an untranslated "translation", an empty flashcard list.
     - The next step would act on them.
     - The operator page already says the `llm` step fails the run.
     - A quota does not reset between attempts, so a retry only spends a counter read and a backoff sleep. Design §4b already says "retry loops do not retry it". With `"402"` in `_NON_RETRIABLE_REASONS`, the engine logs `step_retry_suppressed` instead, as it does for `validation_error`. Any adapter error whose message starts with `402` is treated the same way; a provider's own 402 (payment required) does not clear in seconds either.
   - Alternative: keep going, with each step returning `{"error": "llm_quota_exceeded"}`. This needs a matching branch in all 20 sites and in `parallel`, and a downstream step still cannot tell the output was refused. Separately, the engine's retry can be left as is (drop the `engine.py` edit and its test), and the operator page then keeps "the step is retried once and refused again".
   - It changes documented behaviour: `Usage_Quotas.md:93` says these steps "report the refusal in their outputs, and the run goes on". That line is also inaccurate today, because the outputs say `keyword_extract_error` and the like, not that the call was refused.
2. **Benchmark runs (rows 21–22).**
   - **Recommended:** add them to the §4a boundary check. In `evaluations_benchmarks.run_benchmark`, the first line is `await enforce_llm_quota(getattr(current_user, "id", None))`, as in `evaluations_crud.py:328` and `evaluations_unified.py` (Review M6). `benchmark_api.run_benchmark` gains `current_user: User = Depends(get_eval_request_user)`, which FastAPI resolves once per request because `require_eval_permissions` already depends on it, and makes the same call. An over-quota user gets the 402 before the dataset is loaded. An admitted run completes, as evaluation runs do.
   - Why: they always call an LLM. Without the check, every item of an over-quota user's run is refused and recorded as a failed item, and the run still completes and is stored.
   - The first draft used `Depends(require_llm_quota)`. That resolves `get_request_user` a second time, and two test apps then needed new overrides. The handler call needs none (Review M6).
   - Alternative: no boundary check. Each run loop records refused items as errors (row 21) and the run completes with `failed` equal to the item count. The two benchmark tests of Task 4, its two handler edits, the design-doc row and the operator-page boundary entry are then dropped, and the guide keeps only its mid-run bullet (Task 9 Steps 1-3).
   - Either way, `evaluate_custom_metric` lets the refusal through, so it is never scored 0.0.
3. **One rule for "the LLM part is an add-on to what the user asked for" (row 24, and the audio plan's Decision 1).**
   - **Recommended:** keep the artifact the user asked for, record the refusal with its code, and stop at the first one.
     - A reused transcript keeps its text. The summary is absent and the status is `failed`. `process_result["video_lite_summary_error"]` is `"llm_quota_exceeded"`, and the stored `video_lite` metadata says `summary_status: failed, summary_error: llm_quota_exceeded`; the warning still carries the 402 text (Review M4).
     - This matches `plan_audio_analysis.md` Decision 1, which stores the transcript with `analysis_error.code = "llm_quota_exceeded"`. Both keep the artifact and record a stable code. They store it in different places, because one is a media item's metadata and the other a job result.
     - A side effect: once the refusal escapes `_synthesize_reused_transcript_analysis_sync`, the warnings it collected before it and its `chunks` are not returned. The item then looks like the path with no `api_name` (`chunks` unset), which downstream already handles.
   - Alternative: fail the item, as a media ingest job does today when analysis is refused outside the reuse path (`Usage_Quotas.md:95`: "A media ingest job marks the item `Error`").
   - **Paths that would still discard the user's artifact under this rule** (Review M4), filed as a follow-up in Task 9 Step 4 if the recommendation is taken:
     - media ingest jobs, single and batch (the item is marked `Error`; a batch marks every item);
     - `/media/ingest-web-content` with analysis and `/media/process-web-scraping` with summarization, which discard the request's results;
     - `POST /watchlists/outputs` with `summarize`, which discards them too (`Usage_Quotas.md:92`).
4. **TASK-13635: the completeness metric (Review I4).**
   - **Recommended:** delete `evaluate_response_completeness` and `evaluate_all_metrics`, and amend the task's ACs through backlog-py (Task 10).
     - Verified on `7c94a2efb1`: `git grep -e evaluate_response_completeness -e evaluate_all_metrics -- tldw_Server_API apps` finds 7 lines: 3 in `rag_custom_metrics.py` (the two definitions and the aggregate's call) and 4 in `tests/RAG/test_rag_custom_metrics_sanitizers.py`. Nothing in the app, nothing in the WebUI (`apps/`), and no other test. `eval_runner.py:818-829` uses only `evaluate_retrieval_coverage` and `evaluate_retrieval_diversity`.
     - The LLM path has been dead since `6811ad14bf` (2025-08-18): `await analyze(prompt, "...")` raises `TypeError` before any provider call.
     - Under the first draft's option, PR B would add API surface (`api_name`, `model`), a helper and 3 tests to code only tests can reach. `evaluate_all_metrics` would pass no provider, so no path in the app would grade with an LLM even then.
     - Deleting is the smaller diff (152 lines and one test file removed, none added). It removes the sync-`analyze`-on-the-loop hazard for good, and git keeps the code for whoever wires completeness into `eval_runner`.
     - The ACs assume the metric is used. It is not, so the ACs change rather than the code growing to meet them.
     - The product question behind it: `Docs/Design/RAG_Pipeline_Evaluation.md:43,181` names completeness as a planned generation metric. Deleting it means that work starts from git history.
   - Alternative A, the caller names the provider: `evaluate_response_completeness(..., api_name=..., model=...)`, through `analyze` on a worker thread; with no provider, the heuristic with `details.reason = "no_provider"`. This keeps today's ACs. Appendix A has its complete Task 10 and Task 11, measured: `4 failed, 1 passed` RED and `5 passed` GREEN.
   - Alternative B, a config default as HyDE (`RAG_HYDE_PROVIDER` / `RAG_HYDE_MODEL`) or claims (`CLAIMS_LLM_PROVIDER`) have. That is a new setting, and spend no caller asked for. No steps are written for it.

## Review Focus

1. **Generated-artifact verification after row 13.**
   - Flashcard generation, slides, quiz and OSCE generation, research-workspace artifacts and the workflow verify endpoint now receive the refusal instead of a `needs_revision` verdict.
   - Flashcard generation (`flashcards.py:2621`) and slides (`slides.py:1356`) call the verifier outside any `try`, so they answer 402. Today the same refusal gives a misleading 422 `claim_verification_failed`.
   - The generation call comes first and is refused first. So the verifier sees a refusal only when the generation call itself took the user past a limit.
   - The `require_llm_quota` dependencies in `flashcards.py:2466` and `quizzes.py:784` are on the study-assistant routes, not on generation.
   - The quiz, OSCE and research-workspace call sites have no broad catch around the verifier; their callers decide.
2. **The FVA adjudicator's LLM fallback is not covered, and is filed as a follow-up.**
   - `ClaimAdjudicator._call_llm` (`adjudicator.py:291-325`) calls the sync `_fva_claims_analyze_call` directly from a coroutine, on the loop thread. The quota check is therefore skipped (design §5), and the call also blocks the loop.
   - Its `except Exception` (`adjudicator.py:231`) would swallow a refusal if the call ever moved off the loop.
   - Task 9 files it.
3. **`parallel` with `fail_fast` on** already raised the refusal. **The `retry` step** never caught `HTTPException`. Neither needed code, only tests.
4. **Siblings of a refused sub-step keep running (Review I3).** In `map` and `parallel(fail_fast=True)`, sub-steps already started, and those queued behind the semaphore, run after the step and the run are marked failed (`engine.py:882-935`). Their LLM calls are refused before reaching a provider; non-LLM sub-steps complete with their side effects. This is old behaviour for any raising sub-step, now reached by every content, NLP and bibliography step. Task 9 files the `asyncio.TaskGroup` follow-up; this plan does not change it.
5. **The engine's step retry (Review M3).** `"402"` joins `_NON_RETRIABLE_REASONS`. The token comes from `str(exc)`, so any adapter error whose text starts with `402` is no longer retried. Only `llm`, `prompt`, `tts` and `webhook` steps retry by default (`engine.py:1700-1721`).
6. **Prompt Studio optimizers after rows 16 and 20 (corrected, Review I1).**
   - The refusal now reaches the job as `LLMQuotaExceededError` from every optimizer call that lets errors through in a job: MIPRO's `_rephrase_instruction`, `_add_details` and `_simplify_instruction` (no `try`), `_generate_refinement` (re-raises when `on_provider_success` is set), MCTS segment improvement and MCTS `_rephrase_instruction` (their `_MCTS_NONCRITICAL_EXCEPTIONS` has no `HTTPException`), and the prompt quality scorer when strict. The job fails with `llm_quota_exceeded` and is not retried.
   - The genetic optimizer's `_generate_variation`, `_crossover` and `_mutate` still catch `Exception` and fall back, and so does the quality scorer when not strict. The next scoring call (`run_single_test`) is refused and fails the job the same way.
   - `execute_batch` has no caller in the app. Its `gather(return_exceptions=True)` turns any exception, a refusal included, into a per-item error dict. Not changed.
7. **`TestRunner.run_multiple_tests` non-strict** still records a per-test `{"error": ...}`. Its only caller (`run_batch_tests`, from `/test-cases/run`) passes `strict_provider_errors=True`, so that path is unreachable from the app.
8. **Notes Studio with no provider or model** still returns the deterministic payload with no LLM call. Only the refused-call path changes.
9. **HyDE legacy branch after the move.** A refusal now raises inside `analyze` on the worker thread. `_LLMClient.generate` catches it and returns `""`, and the heuristic answer is used. That is the documented degrade for optional RAG stages, but this branch writes no stage metadata. Its callers are internal: there is no credential runtime.
10. **Prompt Studio record status.** A synchronous evaluation (`run_async: false`) answers 402 and leaves the record `failed`. A background one ends `failed` the same way and logs `PS evaluation.async.refused`; its test pins that the generic handler writes nothing after the `running` update (Review M5).
11. **`web_search`'s optional query rewrite** (`Workflows/adapters/rag/search.py:348`, `except Exception` around `run_query_rewrite_adapter`) still degrades on a refusal: the search runs with the original query. That is an optional stage, like the RAG pipeline's, and the operator page says so (Review I2).
12. **Both benchmark routes are tested end to end** (Review M6, M9): each test posts to the route with a refusing quota and checks the 402 body. The first draft checked only the standalone route's declared dependency.

## ADR assessment

Not needed. ADR-064 (per-user quotas) and ADR-067 (accounting in the provider base class) govern, and neither changes. "A refusal is raised at the call and the caller lets it through" is design §4b, already approved; "retry loops do not retry it" covers the engine change. These tasks make the remaining callers follow it. Deleting an unused metric (PR B) changes no architecture rule.

---

## PR A — TASK-13638

### Task 1: workflow steps let the refusal through, and the engine does not retry it (rows 1–12)

**Files:**
- Modify: `tldw_Server_API/app/core/Workflows/adapters/content/generation.py`. Insert before the `except _GENERATION_NONCRITICAL_EXCEPTIONS` at dev lines 334, 398, 451, 500, 553, 630, 704, 789, 852 and 961. Import after the `Chat.prompt_template_manager` import (line 24).
- Modify: `tldw_Server_API/app/core/Workflows/adapters/text/nlp.py`, before the `except Exception` at lines 124, 200, 340 and 440. Line 260 (`language_detect`, no LLM) is left alone. Import before the `LLM_Calls.tokenizer_resolver` import (line 20).
- Modify: `tldw_Server_API/app/core/Workflows/adapters/research/bibliography.py`, before the `except Exception` at lines 201 and 359. Line 126 (DOI resolve, HTTP only) is left alone. Import before the `http_client` import (line 18).
- Modify: `tldw_Server_API/app/core/Workflows/adapters/content/image.py`, before line 388 (`image_describe`). Import after the `Chat.prompt_template_manager` import (line 24).
- Modify: `tldw_Server_API/app/core/Workflows/adapters/content/summarize.py`, before line 126. Import after the `Chat.prompt_template_manager` import (line 13).
- Modify: `tldw_Server_API/app/core/Workflows/adapters/llm/translate.py`, before line 71. Import after the `Chat.Chat_Deps` import (line 10).
- Modify: `tldw_Server_API/app/core/Workflows/adapters/evaluation/eval.py`, before the `except Exception:` at line 286 (Review I2). Import after the `DB_Management.db_path_utils` import (line 18).
- Modify: `tldw_Server_API/app/core/Workflows/adapters/control/flow.py`, at line 413 (the `gather` in `run_parallel_adapter`), and its import at line 17.
- Modify: `tldw_Server_API/app/core/Workflows/engine.py`, `_NON_RETRIABLE_REASONS` at lines 85-91 (Review M3).
- Create: `tldw_Server_API/tests/Workflows/adapters/test_adapters_llm_quota_refusal.py`
- Test: `tldw_Server_API/tests/Workflows/test_engine_retry_classifier.py` (append; its import at line 6).

- [ ] **Step 0: start the task and the branch**

```bash
git fetch origin dev --quiet && git switch -c fix/llm-quota-refusal-callers origin/dev
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13638 -s "In Progress" \
  --append-notes "Plan: plan_quota_callers_rag_metric.md (PR A, tasks 1-9). Each caller lets LLMQuotaExceededError through, or keeps its artifact and records the refusal; per-caller table in the plan."
git add -A backlog/tasks
```

- [ ] **Step 1: write the failing tests**

`test_adapters_llm_quota_refusal.py`:

```python
"""A workflow step that calls an LLM fails with the user's quota refusal; it does not return an error output (TASK-13638)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Workflows.adapters.content import generation, image, summarize
from tldw_Server_API.app.core.Workflows.adapters.control import flow, state
from tldw_Server_API.app.core.Workflows.adapters.evaluation import eval as evaluation_step
from tldw_Server_API.app.core.Workflows.adapters.llm import translate
from tldw_Server_API.app.core.Workflows.adapters.research import bibliography
from tldw_Server_API.app.core.Workflows.adapters.text import nlp

pytestmark = pytest.mark.unit

CHAT_CALL = "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async"
ANALYZE = "tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.analyze"
TEXT = "Plants turn light, water and air into sugar."
GENERATION_CONFIG = {
    "text": TEXT,
    "content": TEXT,
    "excerpt_text": TEXT,
    "topic": "plants",
    "provider": "openai",
    "model": "gpt-4o-mini",
}


def _refusal() -> LLMQuotaExceededError:
    return LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter_name",
    [
        "run_flashcard_generate_adapter",
        "run_quiz_generate_adapter",
        "run_outline_generate_adapter",
        "run_glossary_extract_adapter",
        "run_mindmap_generate_adapter",
        "run_slides_generate_adapter",
        "run_report_generate_adapter",
        "run_newsletter_generate_adapter",
        "run_notes_studio_generate_adapter",
        "run_diagram_generate_adapter",
    ],
)
async def test_content_generation_lets_the_refusal_through(adapter_name: str) -> None:
    with patch.object(generation, "perform_chat_api_call_async", AsyncMock(side_effect=_refusal())):
        with pytest.raises(LLMQuotaExceededError):
            await getattr(generation, adapter_name)(dict(GENERATION_CONFIG), {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter", "config"),
    [
        (nlp.run_keyword_extract_adapter, {"text": TEXT, "method": "llm"}),
        (nlp.run_sentiment_analyze_adapter, {"text": TEXT}),
        (nlp.run_topic_model_adapter, {"text": TEXT}),
        (nlp.run_entity_extract_adapter, {"text": TEXT}),
        (bibliography.run_reference_parse_adapter, {"citation": "Smith J. 2020. Plants. J Bot 1:1."}),
        (bibliography.run_literature_review_adapter, {"papers": [{"title": "Plants", "abstract": TEXT}]}),
    ],
)
async def test_text_and_research_steps_let_the_refusal_through(adapter, config: dict) -> None:
    with patch(CHAT_CALL, AsyncMock(side_effect=_refusal())):
        with pytest.raises(LLMQuotaExceededError):
            await adapter(dict(config), {})


@pytest.mark.asyncio
async def test_image_describe_lets_the_refusal_through() -> None:
    with patch.object(image, "perform_chat_api_call_async", AsyncMock(side_effect=_refusal())):
        with pytest.raises(LLMQuotaExceededError):
            await image.run_image_describe_adapter({"image_url": "https://example.com/plant.png"}, {})


@pytest.mark.asyncio
async def test_summarize_lets_the_refusal_through() -> None:
    with patch.object(summarize, "is_test_mode", return_value=False), patch(ANALYZE, side_effect=_refusal()):
        with pytest.raises(LLMQuotaExceededError):
            await summarize.run_summarize_adapter({"text": TEXT, "api_name": "openai"}, {})


@pytest.mark.asyncio
async def test_translate_does_not_return_the_input_as_a_translation() -> None:
    adapter = MagicMock()
    adapter.achat = AsyncMock(side_effect=_refusal())
    registry = MagicMock()
    registry.get_adapter.return_value = adapter
    with patch.object(translate, "is_test_mode", return_value=False), patch.object(
        translate, "get_registry", return_value=registry
    ):
        with pytest.raises(LLMQuotaExceededError):
            await translate.run_translate_adapter({"input": "Bonjour", "target_lang": "en"}, {})


@pytest.mark.asyncio
async def test_parallel_does_not_drop_a_refused_step() -> None:
    async def refused_step(_config, _context):
        raise _refusal()

    with patch.object(flow, "get_adapter", return_value=refused_step):
        with pytest.raises(LLMQuotaExceededError):
            await flow.run_parallel_adapter({"steps": [{"type": "summarize", "config": {}}]}, {})


@pytest.mark.asyncio
async def test_retry_does_not_retry_a_refused_step() -> None:
    analyze = MagicMock(side_effect=_refusal())
    with patch.object(summarize, "is_test_mode", return_value=False), patch(ANALYZE, analyze), patch.object(
        state.asyncio, "sleep", AsyncMock()
    ):
        with pytest.raises(LLMQuotaExceededError):
            await state.run_retry_adapter(
                {"step_type": "summarize", "step_config": {"text": TEXT, "api_name": "openai"}, "max_retries": 3}, {}
            )
    assert analyze.call_count == 1


@pytest.mark.asyncio
async def test_evaluations_step_lets_the_refusal_through() -> None:
    service = MagicMock()
    service.initialize = AsyncMock()
    service.evaluate_geval = AsyncMock(side_effect=_refusal())
    with patch.object(evaluation_step, "is_test_mode", return_value=False), patch(
        "tldw_Server_API.app.core.Evaluations.unified_evaluation_service.UnifiedEvaluationService",
        MagicMock(return_value=service),
    ):
        with pytest.raises(LLMQuotaExceededError):
            await evaluation_step.run_evaluations_adapter({"action": "geval", "response": TEXT, "context": TEXT}, {"user_id": "1"})
```

In `tests/Workflows/test_engine_retry_classifier.py` (Review M3), change line 6 to `from tldw_Server_API.app.core.exceptions import AdapterError, LLMQuotaExceededError` and append:

```python
@pytest.mark.asyncio
async def test_llm_quota_refusal_does_not_consume_retries(tmp_path, monkeypatch):
    """A step refused for the user's LLM quota is not retried, even when it sets retry (TASK-13638)."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = "run-llm-quota-refusal"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={
            "name": "llm-quota-refusal",
            "version": 1,
            "steps": [{"id": "s1", "type": "prompt", "retry": 3, "config": {"template": "ok"}}],
        },
    )

    async def _refused_adapter(_config, _context):
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _refused_adapter)

    async def _fast_sleep(_duration: float):
        return None

    monkeypatch.setattr(engine_mod.asyncio, "sleep", _fast_sleep)

    await engine_mod.WorkflowEngine(db).start_run(run_id)

    run = db.get_run(run_id)
    assert run is not None
    assert run.status == "failed"
    assert len(db.list_step_attempts(run_id=run_id, step_id="s1")) == 1
    assert any(event.get("event_type") == "step_retry_suppressed" for event in db.get_events(run_id))
```

- [ ] **Step 2: run them**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/adapters/test_adapters_llm_quota_refusal.py \
  "tldw_Server_API/tests/Workflows/test_engine_retry_classifier.py::test_llm_quota_refusal_does_not_consume_retries"
```

Expected: `23 failed`. Measured on `7c94a2efb1`: the 22 adapter cases with `Failed: DID NOT RAISE LLMQuotaExceededError`, and the engine test with `assert 4 == 1` (one attempt and three retries).

- [ ] **Step 3: implement**

In each of the seven leaf modules, add the import on the line Files names:

```python
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
```

Then put these two lines directly above every listed `except` line, at the same indentation as that `except`. That is 10 in `generation.py`, 4 in `nlp.py`, 2 in `bibliography.py` and 1 each in `image.py`, `summarize.py`, `translate.py` and `eval.py`:

```python
    except LLMQuotaExceededError:
        raise
```

For example, `eval.py` ends:

```python
        return {"error": f"unknown_action:{action}"}

    except LLMQuotaExceededError:
        raise
    except Exception:
        logger.error("Evaluations adapter error")
        return {"error": "evaluations_error"}
```

In `control/flow.py`, change line 17 to `from tldw_Server_API.app.core.exceptions import AdapterError, LLMQuotaExceededError`. Then replace the end of `run_parallel_adapter`:

```python
    tasks = [run_step(i, step) for i, step in enumerate(steps)]
    await asyncio.gather(*tasks, return_exceptions=not fail_fast)
```

with:

```python
    tasks = [run_step(i, step) for i, step in enumerate(steps)]
    outcomes = await asyncio.gather(*tasks, return_exceptions=not fail_fast)
    refusal = next((o for o in outcomes if isinstance(o, LLMQuotaExceededError)), None)
    if refusal is not None:
        raise refusal  # the user's limit fails the step; it is not dropped as a missing result
```

In `engine.py` (Review M3):

```python
_NON_RETRIABLE_REASONS: set[str] = {
    "validation_error",
    "authz_error",
    "acp_governance_blocked",
    "session_access_denied",
    "invariant_violation",
    "402",  # an LLM quota refusal (LLMQuotaExceededError): the limit does not reset between attempts
}
```

- [ ] **Step 4: run the new tests, then the neighbourhood**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows/adapters/test_adapters_llm_quota_refusal.py \
  "tldw_Server_API/tests/Workflows/test_engine_retry_classifier.py::test_llm_quota_refusal_does_not_consume_retries"
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Workflows
```

Expected: `23 passed`, then `1682 passed, 6 skipped` (measured on the prototype, 8 min 22 s). `tests/Workflows` holds every test that names `_is_retriable_error`, `_reason_code_from_error` or `step_retry_suppressed`. Any other failure gets a baseline run on unchanged `origin/dev` before it is attributed to this change. Then the lint comparison (Global Constraints): nothing new.

- [ ] **Step 5: commit**

```bash
git add -A backlog/tasks tldw_Server_API/app/core/Workflows tldw_Server_API/tests/Workflows/adapters/test_adapters_llm_quota_refusal.py tldw_Server_API/tests/Workflows/test_engine_retry_classifier.py
git commit -m "fix(workflows): an LLM quota refusal fails the step, unretried, instead of becoming an error output (TASK-13638)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: claims verification lets the refusal through (rows 13–14)

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_service.py`. The import at line 71; the `try` around `perform_chat_api_call` at lines 610-615.
- Modify: `tldw_Server_API/app/core/Claims_Extraction/fva_pipeline.py`. The import; `process_batch` after the `gather` at lines 406-409.
- Test: `tldw_Server_API/tests/Claims/test_claims_service_fva_analyze.py`, `tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py`

- [ ] **Step 1: write the failing tests**

In `tests/Claims/test_claims_service_fva_analyze.py`, replace the import block (`from __future__ import annotations` and `from tldw_Server_API.app.core.Claims_Extraction import claims_service`) with:

```python
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_engine, claims_service
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
```

Append:

```python
def _refused(**_kwargs):
    raise LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )


def test_fva_analyze_lets_an_llm_quota_refusal_through(monkeypatch):
    """A refusal is not turned into an empty reply (TASK-13638)."""
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call", _refused)

    with pytest.raises(LLMQuotaExceededError):
        claims_service._fva_claims_analyze_call("openai", "input", "prompt", None, None, 0.2)


@pytest.mark.asyncio
async def test_a_refused_verification_raises_and_records_no_usage(monkeypatch):
    """On dev the empty reply let the decision tree mark this claim verified, and wrote a claims_verify row (TASK-13638)."""
    recorder = AsyncMock()
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call", _refused)
    monkeypatch.setattr(claims_engine, "_log_claims_llm_usage", recorder)
    verifier = claims_engine.HybridClaimVerifier(claims_service._fva_claims_analyze_call)
    evidence = Document(id="d1", content="The sky is blue.", metadata={}, source=DataSource.MEDIA_DB, score=0.9)

    with pytest.raises(LLMQuotaExceededError):
        await verifier.verify(
            claim=claims_engine.Claim(id="c1", text="The sky is green."),
            query="sky colour",
            base_documents=[evidence],
            mode="llm",
        )
    recorder.assert_not_awaited()
```

Append to `tests/Claims_Extraction/test_fva_pipeline.py`:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_batch_does_not_drop_an_llm_quota_refusal():
    """A refused claim fails the batch; it is not silently left out of the results (TASK-13638)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    verifier = MockVerifier()
    verifier.verify = AsyncMock(
        side_effect=LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )
    )
    pipeline = FVAPipeline(MockClaimsEngine(verifier), MockRetriever())

    with pytest.raises(LLMQuotaExceededError):
        await pipeline.process_batch(claims=[MockClaim(id="1", text="Claim 1")], query="test", documents=[])
```

- [ ] **Step 2: run them**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Claims/test_claims_service_fva_analyze.py "tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py::test_process_batch_does_not_drop_an_llm_quota_refusal"
```

Expected: `3 failed, 3 passed`. The three new tests fail with `DID NOT RAISE LLMQuotaExceededError`. Measured on dev: the verify probe returned `VERIFIED` and wrote one `claims_verify` row with status 200, and `process_batch` returned `results=[]`.

- [ ] **Step 3: implement**

`claims_service.py`: change line 71 to `from tldw_Server_API.app.core.exceptions import EgressPolicyError, LLMQuotaExceededError, RetryExhaustedError`. At the end of `_fva_claims_analyze_call`:

```python
    try:
        response = perform_chat_api_call(**call_kwargs)
        return coerce_llm_response_text(response)
    except LLMQuotaExceededError:
        raise  # the user's limit: not an empty reply for the verifier to score without the LLM
    except _CLAIMS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"FVA analyze function failed: {exc}")
        return ""
```

`fva_pipeline.py`: add `from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError` after `from loguru import logger`, preceded by a blank line. In `process_batch`:

```python
        results = await asyncio.gather(
            *[process_with_semaphore(c) for c in claims],
            return_exceptions=True,
        )
        refusal = next((r for r in results if isinstance(r, LLMQuotaExceededError)), None)
        if refusal is not None:
            raise refusal  # the user's limit fails the batch; a refused claim is not dropped from it
```

The batch waits for every claim before it raises; each one's own LLM call is refused before it reaches a provider (Architecture, Review I3).

- [ ] **Step 4: run the new tests, then the neighbourhood**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Claims/test_claims_service_fva_analyze.py "tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py::test_process_batch_does_not_drop_an_llm_quota_refusal"
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Claims tldw_Server_API/tests/Claims_Extraction tldw_Server_API/tests/Flashcards
```

Expected: `6 passed`, then no failures. In the review's dry run one Hypothesis test, `test_claims_jobs_contracts.py::test_review_metrics_payload_inclusive_window_property`, failed with `FailedHealthCheck: Input generation is slow` while other suites loaded the machine, and passed alone. Rerun any such failure alone; any other failure gets a baseline run on unchanged `origin/dev`.

- [ ] **Step 5: commit** `fix(claims): an LLM quota refusal is raised, not verified heuristically or dropped from an FVA batch (TASK-13638)`, with the trailer.

### Task 3: Prompt Studio (rows 15–20)

**Files:**
- Modify: `tldw_Server_API/app/core/Prompt_Management/prompt_studio/test_runner.py`. Line 27 (import); before `except Exception as e:` at line 218.
- Modify: `tldw_Server_API/app/core/Prompt_Management/prompt_studio/prompt_executor.py` (Review I1). Line 30 (import); before `except Exception as e:` at line 192 (`execute_prompt`) and line 559 (`_call_llm`).
- Modify: `tldw_Server_API/app/core/Prompt_Management/prompt_studio/evaluation_manager.py`. The import after line 19; before `except Exception as e:` at line 257.
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_evaluations.py`. Line 61 (import); before `except (ByokResolutionError, *_PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS) as e:` at line 993.
- Modify: `tldw_Server_API/app/core/Prompt_Management/prompt_studio/services/jobs_worker.py`. Line 68 (import); `_bounded_optimization_error` at line 646.
- Test: `tldw_Server_API/tests/prompt_studio/unit/test_test_runner_strict_provider_errors.py` (append), `tldw_Server_API/tests/prompt_studio/unit/test_optimization_job_provider_runtime.py` (append). Create `tldw_Server_API/tests/prompt_studio/unit/test_evaluation_llm_quota_refusal.py` and `tldw_Server_API/tests/prompt_studio/unit/test_prompt_executor_llm_quota_refusal.py` (Review I1).

- [ ] **Step 1: write the failing tests**

Append to `test_test_runner_strict_provider_errors.py`. It already has `_RunnerDb`, `_MODEL_CONFIG`, `TestRunner` and `Any`.

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
async def test_single_test_lets_an_llm_quota_refusal_through(monkeypatch: pytest.MonkeyPatch, strict: bool) -> None:
    """A refusal is neither stored as a run scored 0.0 nor reported as a provider outage (TASK-13638)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    db = _RunnerDb()
    runner = TestRunner(db)

    def _refused(**_kwargs: Any) -> Any:
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(runner, "_call_adapter", _refused, raising=True)

    with pytest.raises(LLMQuotaExceededError):
        await runner.run_single_test(
            prompt_id=12, test_case_id=3, model_config=_MODEL_CONFIG, strict_provider_errors=strict
        )
    assert db.runs == []
```

Append to `test_optimization_job_provider_runtime.py`. It already has `jobs_worker` and `_assert_safe_job_error`.

```python
def test_llm_quota_refusal_is_bounded_and_not_retryable() -> None:
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    bounded = jobs_worker._bounded_optimization_error(
        LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )
    )

    _assert_safe_job_error(bounded, code="llm_quota_exceeded", retryable=False)
```

Create `test_prompt_executor_llm_quota_refusal.py` (Review I1). The endpoint test replaces `_call_llm` because the endpoint passes runtime-issued credentials, which a test cannot mint; the first test covers `_call_llm` itself.

```python
"""Prompt Studio's executor lets an LLM quota refusal through instead of reporting a provider error (TASK-13638)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_prompts as prompts_endpoint
from tldw_Server_API.app.api.v1.schemas.prompt_studio_schemas import ExecutePromptSimpleRequest
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import prompt_executor as executor_module
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor

pytestmark = pytest.mark.unit


def _refusal() -> LLMQuotaExceededError:
    return LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )


class _RefusingAdapter:
    def chat(self, _request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        raise _refusal()


class _PromptDb:
    client_id = "quota-refusal"

    def get_prompt(self, prompt_id: int) -> dict[str, Any]:
        return {"id": prompt_id, "deleted": False, "system_prompt": "Answer.", "user_prompt": "Q: {question}"}


@pytest.mark.asyncio
async def test_call_llm_lets_the_refusal_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optimizers' own calls go through _call_llm; a refusal is not rewritten as a retryable provider error."""

    async def _run_sync_call(function: Any, **_kwargs: Any) -> Any:
        return function()

    async def _on_success() -> None:
        raise AssertionError("a refused call is not a provider success")

    monkeypatch.setattr(
        executor_module, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _name: _RefusingAdapter())
    )
    monkeypatch.setattr(executor_module, "await_bounded_sync_call", _run_sync_call)

    with pytest.raises(LLMQuotaExceededError):
        await PromptExecutor(_PromptDb())._call_llm(
            provider="openai",
            model="m",
            prompt="p",
            api_key_override="k",
            app_config={},
            credentials_resolved=True,
            on_provider_success=_on_success,
        )


@pytest.mark.asyncio
async def test_prompt_execute_endpoint_answers_the_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /prompt-studio/prompts/execute answers 402, not 502 "Upstream provider request failed"."""

    async def _refused_call(_self: PromptExecutor, **_kwargs: Any) -> dict[str, Any]:
        raise _refusal()

    async def _allow_project(*_args: Any, **_kwargs: Any) -> bool:
        return True

    class _Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None) -> Any:
            return SimpleNamespace(api_key="k", app_config={}, credentials_resolved=True)

        async def close(self) -> None:
            return None

    monkeypatch.setattr(PromptExecutor, "_call_llm", _refused_call)
    monkeypatch.setattr(prompts_endpoint, "authoritative_prompt_project", lambda *_args, **_kwargs: (1, 7))
    monkeypatch.setattr(prompts_endpoint, "require_project_access", _allow_project)
    monkeypatch.setattr(
        prompts_endpoint, "derive_trusted_credential_scope", lambda *_args, **_kwargs: (7, [], [], False)
    )
    monkeypatch.setattr(prompts_endpoint, "ProviderCredentialRuntime", lambda **_kwargs: _Runtime())
    request = Request({"type": "http", "method": "POST", "path": "/api/v1/prompt-studio/prompts/execute", "headers": []})

    with pytest.raises(LLMQuotaExceededError) as exc_info:
        await prompts_endpoint.execute_prompt_simple(
            ExecutePromptSimpleRequest(prompt_id=1, inputs={"question": "q"}, provider="openai", model="m"),
            request,
            db=_PromptDb(),
            user_context={"user_id": "7"},
        )

    assert exc_info.value.detail["error"] == "limit_exceeded"
```

Create `test_evaluation_llm_quota_refusal.py`. Its background test ends with an assertion on the records (Review M5). The review's `assert db.updates == []` fails on the fixed code, because `run_evaluation_async` writes the `running` update itself (measured), so the test asserts that nothing follows it.

```python
"""A Prompt Studio evaluation stops at an LLM quota refusal and records it (TASK-13638)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager import EvaluationManager

pytestmark = pytest.mark.unit


def _refusal() -> LLMQuotaExceededError:
    return LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )


class _EvaluationDb:
    client_id = "quota-refusal"

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []

    def get_prompt(self, prompt_id: int) -> dict[str, Any]:
        return {"id": prompt_id, "project_id": 71, "deleted": False, "system_prompt": "Answer.", "user_prompt": "Q: {question}"}

    def get_test_cases_by_ids(self, ids: list[int]) -> list[dict[str, Any]]:
        return [
            {"id": i, "project_id": 71, "inputs": {"question": "q"}, "expected_outputs": {"response": "a"}} for i in ids
        ]

    def get_test_case(self, test_case_id: int) -> dict[str, Any]:
        return self.get_test_cases_by_ids([test_case_id])[0]

    def get_evaluation(self, evaluation_id: int) -> dict[str, Any]:
        return {
            "id": evaluation_id,
            "uuid": "eval-uuid",
            "prompt_id": 1,
            "test_case_ids": [2, 3],
            "model_configs": {"model": "m"},
        }

    def update_evaluation(self, evaluation_id: int, updates: dict[str, Any]) -> None:
        self.updates.append(dict(updates))

    def create_test_run(self, **kwargs: Any) -> dict[str, Any]:
        self.runs.append(kwargs)
        return {"id": len(self.runs)}


@pytest.mark.asyncio
async def test_evaluation_stops_at_the_refusal_and_marks_the_record_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _EvaluationDb()
    calls: list[dict[str, Any]] = []

    def _refused(self: EvaluationManager, **kwargs: Any) -> Any:
        calls.append(kwargs)
        raise _refusal()

    monkeypatch.setattr(EvaluationManager, "_call_adapter_text", _refused)

    with pytest.raises(LLMQuotaExceededError):
        await EvaluationManager(db).run_evaluation_with_existing_record(
            evaluation_id=5, prompt_id=1, test_case_ids=[2, 3], model="m"
        )

    assert len(calls) == 1
    assert db.runs == []
    assert db.updates[-1]["status"] == "failed"
    assert db.updates[-1]["error_message"] == "llm_quota_exceeded: Monthly LLM token limit reached"


@pytest.mark.asyncio
async def test_background_evaluation_does_not_raise_the_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_evaluations as eval_endpoint

    class _Manager:
        def __init__(self, _db: Any) -> None:
            return None

        async def run_evaluation_with_existing_record(self, **_kwargs: Any) -> dict[str, Any]:
            raise _refusal()

    class _Runtime:
        async def close(self) -> None:
            return None

    monkeypatch.setattr(eval_endpoint, "EvaluationManager", _Manager)
    monkeypatch.setattr(eval_endpoint, "provider_requires_api_key", lambda _provider: False)
    handle = SimpleNamespace(api_key=None, app_config={}, credentials_resolved=True)
    db = _EvaluationDb()

    await eval_endpoint.run_evaluation_async(
        5,
        db,
        provider="openai",
        model="m",
        credential_runtime=_Runtime(),
        provider_credentials=handle,
    )

    # Only the start is written here: the generic handler does not overwrite the manager's llm_quota_exceeded record.
    assert [update["status"] for update in db.updates] == ["running"]
```

- [ ] **Step 2: run them**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  "tldw_Server_API/tests/prompt_studio/unit/test_test_runner_strict_provider_errors.py::test_single_test_lets_an_llm_quota_refusal_through" \
  "tldw_Server_API/tests/prompt_studio/unit/test_optimization_job_provider_runtime.py::test_llm_quota_refusal_is_bounded_and_not_retryable" \
  tldw_Server_API/tests/prompt_studio/unit/test_evaluation_llm_quota_refusal.py \
  tldw_Server_API/tests/prompt_studio/unit/test_prompt_executor_llm_quota_refusal.py
```

Expected: `7 failed`. Measured on `7c94a2efb1`:
- `[False]`: `DID NOT RAISE`.
- `[True]`: raised `SanitizedProviderStreamError: The chat service provider is currently unavailable.`
- Bounded error: `'prompt_studio_job_failed' == 'llm_quota_exceeded'`.
- Evaluation: `DID NOT RAISE`. Both cases were scored 0.0 and the record ended `completed`.
- Background: the refusal escaped `run_evaluation_async`. On dev it comes from the fake manager.
- `_call_llm`: `RuntimeError: Provider returned an error response` (Review I1).
- Endpoint: `HTTPException: 502: Upstream provider request failed.` (Review I1).

- [ ] **Step 3: implement**

`test_runner.py`: change line 27 to `from ....core.exceptions import LLMQuotaExceededError, raise_detached_error`. In `run_test_case`:

```python
            actual_output = {"response": response_text}
            _log.info("PS testrun.llm.done time_ms={}", int((time.time() - start_time) * 1000))

        except LLMQuotaExceededError:
            raise  # the user's limit: neither a provider outage nor a failed run scored 0.0
        except Exception as e:
            safe_error = sanitized_provider_stream_exception(e)
```

`prompt_executor.py` (Review I1): change line 30 to `from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError, raise_detached_error`. At the end of `execute_prompt`:

```python
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

        except LLMQuotaExceededError:
            raise  # the user's limit, not a failed execution reported as a provider error
        except Exception as e:
            logger.error(
                "Prompt execution failed; error_type={}",
```

and at the end of `_call_llm`:

```python
            return {"content": content, "tokens": tokens}
        except LLMQuotaExceededError:
            raise  # the user's limit, not "Provider returned an error response"
        except Exception as e:
            logger.error(
                "LLM call failed for {}/{}; error_type={}",
```

`evaluation_manager.py`: add `from ....core.exceptions import LLMQuotaExceededError` after the `streaming_utils` import. In the per-test loop of `run_evaluation_with_existing_record`:

```python
            except LLMQuotaExceededError as refusal:
                # Stop at the user's limit and record it; the remaining cases would be refused too.
                self.db.update_evaluation(
                    evaluation_id,
                    {
                        "status": "failed",
                        "completed_at": datetime.utcnow(),
                        "error_message": f"{refusal.failure_code}: {refusal.detail['message']}",
                    },
                )
                raise
            except Exception as e:
                safe_error = sanitized_provider_stream_exception(e)
```

`prompt_studio_evaluations.py`: change line 61 to `from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError, raise_detached_error`. In `run_evaluation_async`, between the `except asyncio.CancelledError` block and the generic one:

```python
    except LLMQuotaExceededError:
        # EvaluationManager has marked the record failed with llm_quota_exceeded.
        get_ps_logger(
            request_id=request_id,
            ps_component="evaluation_bg",
            ps_job_kind="evaluations",
            traceparent=traceparent,
        ).info("PS evaluation.async.refused evaluation_id={}", evaluation_id)
    except (ByokResolutionError, *_PROMPT_STUDIO_EVAL_NONCRITICAL_EXCEPTIONS) as e:
```

`jobs_worker.py`: change line 68 to `from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError, raise_detached_error`. In `_bounded_optimization_error`:

```python
    if isinstance(exc, PromptStudioJobError):
        return exc
    if isinstance(exc, LLMQuotaExceededError):
        return PromptStudioJobError("LLM quota exceeded", retryable=False, failure_code=exc.failure_code)
    if isinstance(exc, ByokResolutionError):
```

- [ ] **Step 4: run the new tests, then `tests/prompt_studio`**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/prompt_studio
```

Expected: the 7 new tests pass, and the rest of `tests/prompt_studio` too (measured: no failure or error in its 93 files; progress tally 1124 passed, 17 skipped). Any failure gets a dev baseline. Then the lint comparison: nothing new.

- [ ] **Step 5: commit** `fix(prompt-studio): an LLM quota refusal stops a test run, an execution, an optimizer call or an evaluation, and is recorded as llm_quota_exceeded (TASK-13638)`, with the trailer.

### Task 4: benchmarks (rows 21–22; Decision 2)

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/evaluation_manager.py`. The import after line 36; before `except Exception as e:` at line 659.
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py`. The import after line 12; the first line of `run_benchmark`, before its `try:` at line 106 (Review M6).
- Modify: `tldw_Server_API/app/api/v1/endpoints/benchmark_api.py`. The imports at lines 14-15; `run_benchmark`'s signature at lines 201-205 and its first line, before the `try:` at line 220 (Review M6).
- Test: append to `tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py`, `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py` and `tldw_Server_API/tests/Evaluations/test_benchmark_api_auth.py`. No existing test changes (Review M6).

- [ ] **Step 1: write the failing tests**

Append to `tests/Evaluations/unit/test_evaluation_manager.py`. The `evaluation_manager` fixture is in `tests/Evaluations/conftest.py`.

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_evaluate_custom_metric_lets_an_llm_quota_refusal_through(evaluation_manager):
    """A refusal is not scored 0.0 (TASK-13638)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    refusal = LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )
    with patch("tldw_Server_API.app.core.Evaluations.evaluation_manager.analyze", side_effect=refusal):
        with pytest.raises(LLMQuotaExceededError):
            await evaluation_manager.evaluate_custom_metric(
                metric_name="accuracy",
                description="d",
                evaluation_prompt="Rate {answer}",
                input_data={"answer": "a"},
                scoring_criteria={"accuracy": "0-10"},
                api_name="openai",
            )
```

Append to `tests/Evaluations/test_evaluations_benchmarks_api.py`, which has `_build_app`, `benchmarks_ep` and `TestClient`. The body assertion keeps the test-quality ratchet green (Review B1): with only `assert loaded == []`, `test_quality_triage.py --enforce` exits 1 with `tautology_suspect=1 (new)` for this file.

```python
def test_benchmark_run_refuses_an_over_quota_user_before_loading_the_dataset(monkeypatch):
    """Benchmark runs check the LLM quota at the request boundary, as evaluation runs do (TASK-13638)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    refuse_llm_work(monkeypatch)
    loaded = []
    monkeypatch.setattr(benchmarks_ep, "load_benchmark_dataset", lambda *_a, **_k: loaded.append(True) or [])

    app = _build_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/benchmarks/bullshit_benchmark/run",
            json={"limit": 1, "api_name": "openai", "parallel": 1, "save_results": False},
        )

    assert_llm_quota_402(response)
    assert response.json()["detail"]["error"] == "limit_exceeded"
    assert loaded == []
```

Append to `tests/Evaluations/test_benchmark_api_auth.py`, which has `_make_app`, `_override_user`, `EVALS_MANAGE` and `TestClient`. This replaces the first draft's declared-dependency check in `test_benchmark_api_error_mapping.py` with a request (Review M6, M9). The file has a baseline of `status_only=3`, so the test asserts on the body.

```python
def test_run_route_refuses_an_over_quota_user_before_any_work(monkeypatch) -> None:
    """The standalone run route checks the LLM quota before it looks up the benchmark (TASK-13638)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    refuse_llm_work(monkeypatch)
    app = _make_app()
    _override_user(app, [EVALS_MANAGE])

    with TestClient(app) as client:
        response = client.post("/api/v1/benchmarks/does-not-exist/run", json={})

    assert_llm_quota_402(response)
    assert response.json()["detail"]["error"] == "limit_exceeded"
```

- [ ] **Step 2: run them**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  "tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py::test_evaluate_custom_metric_lets_an_llm_quota_refusal_through" \
  "tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py::test_benchmark_run_refuses_an_over_quota_user_before_loading_the_dataset" \
  "tldw_Server_API/tests/Evaluations/test_benchmark_api_auth.py::test_run_route_refuses_an_over_quota_user_before_any_work"
```

Expected: `3 failed`. Measured on `7c94a2efb1`:
- `DID NOT RAISE`.
- `{"detail":"Failed to load dataset for benchmark 'bullshit_benchmark'"}`: the run started, and there was no 402.
- `{"detail":"Benchmark 'does-not-exist' not found"}`: the lookup ran, and there was no 402.

- [ ] **Step 3: implement**

`Evaluations/evaluation_manager.py`: add `from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError` after the `Evaluations.scoring` import. In `evaluate_custom_metric`:

```python
                "raw_output": response
            }

        except LLMQuotaExceededError:
            raise  # the user's limit is not a 0.0 score
        except Exception as e:
            logger.error(f"Custom metric evaluation failed: {e}")
```

`evaluations_benchmarks.py` (Review M6): add `from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota` after the `auth_deps` import (line 12). Then:

```python
    current_user: User = Depends(get_eval_request_user),
):
    # Usage quotas (LLM usage accounting spec §4a): every item is an LLM call, so the run is
    # checked here, before the dataset loads, and then runs to completion.
    await enforce_llm_quota(getattr(current_user, "id", None))
    try:
        identity = get_evaluation_identity(current_user)
```

`benchmark_api.py` (Review M6): replace the imports at lines 14-15 with:

```python
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_rate_limiter_dep
from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import (
    get_eval_request_user,
    require_eval_permissions,
)
```

Then give `run_benchmark` the user that `require_eval_permissions` already resolved (FastAPI caches `get_eval_request_user` per request) and check it first:

```python
async def run_benchmark(
    benchmark_name: str,
    request: BenchmarkRunRequest,
    user_id: str = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_eval_request_user),
):
```

```python
        Summary of benchmark results.
    """
    # Usage quotas (LLM usage accounting spec §4a): checked before any work, as the evaluations route does.
    await enforce_llm_quota(getattr(current_user, "id", None))
    try:
        registry = get_registry()
```

- [ ] **Step 4: run the new tests, the benchmark files, then `tests/Evaluations` and `tests/Usage`.**
  - Expected: `3 passed`. The four benchmark-route test files (`test_benchmark_api_auth.py`, `test_evaluations_benchmarks_api.py`, `test_benchmark_api_error_mapping.py`, `test_evaluations_benchmarks_endpoint_sanitization.py`) all pass, with no override edits (Review M6).
  - Run `tests/Evaluations` without `property/` and `integration/`, and `tests/Usage` (measured, targeted: the four benchmark-route files, `Evaluations/unit/test_evaluations_identity_route_bindings.py`, `Evaluations/unit/test_evaluation_manager.py`, `Evaluations/unit/test_eval_runner.py`, `Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py`, both character memory files, `tests/Explainer`, the three HyDE files, `Media/test_media_llm_quota.py`, `MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py` and `tests/Usage`: `507 passed, 1 skipped`; the review's dry run found no new failure in the rest of `tests/Evaluations`).
  - Run the test-quality ratchet (Review B1): `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` exits 0 ("OK — no offenses beyond the baseline"), and `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/CI/test_test_quality_ratchet.py` gives `1 passed` (90 s on the prototype).
  - Run the OpenAPI drift gate from `backend-required`: `PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts`. Expected: exit 0. Measured on the prototype: the new parameter adds nothing to the schema, because `require_eval_permissions` already declared `get_eval_request_user`'s inputs.
- [ ] **Step 5: commit** `fix(evaluations): benchmark runs refuse an over-quota user up front, and a refusal is never scored 0.0 (TASK-13638)`, with the trailer.

### Task 5: character memory extraction (row 23)

**Files:** Modify `tldw_Server_API/app/core/Character_Chat/modules/character_memory_extraction.py`: the import after `from loguru import logger`, and before `except Exception:` at line 240. Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_memory.py` (append).

- [ ] **Step 1: failing test**

```python
@pytest.mark.unit
def test_memory_extraction_lets_an_llm_quota_refusal_through(monkeypatch):
    """A refusal is not reported as "no memories found" (TASK-13638)."""
    from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
        extract_character_memories,
    )
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    def _refused(**_kwargs):
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_orchestrator.chat_api_call", _refused)

    with pytest.raises(LLMQuotaExceededError):
        extract_character_memories(
            [{"sender": "user", "content": "I like green tea."}], "Bob", "Al", [], api_endpoint="openai"
        )
```

- [ ] **Step 2: run it.** Expected: `DID NOT RAISE` (measured: it returned an empty `ExtractionResult`).
- [ ] **Step 3: implement**

```python
from loguru import logger

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
```

```python
    except LLMQuotaExceededError:
        raise  # the user's limit, not "no memories found"
    except Exception:
        logger.exception("Character memory extraction LLM call failed")
        return ExtractionResult(unique=[], total_parsed=0, duplicates_skipped=0)
```

- [ ] **Step 4: run** `tests/Character_Chat_NEW/unit/test_character_memory.py` and `tests/Character_Chat_NEW/integration/test_character_memory_endpoint.py`. Expected: all pass.
- [ ] **Step 5: commit** `fix(character-chat): manual memory extraction answers a quota refusal instead of finding nothing (TASK-13638)`, with the trailer.

### Task 6: transcript reuse stops at the first refusal and records its code (row 24; Decision 3)

**Files:** Modify `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`: the import after line 53 (`scope_context`); the chunk loop's `except` at line 1709; `_populate_reused_transcript_summary`'s `except` at line 1782; and `_merge_video_lite_summary_safe_metadata`'s `failed` branch at lines 1326-1329 (Review M4). Create `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_reused_transcript_llm_quota_refusal.py`.

- [ ] **Step 1: failing test**

```python
"""A media item that reuses a transcript stops summarizing at the first LLM quota refusal (TASK-13638)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_reused_transcript_stops_at_the_first_refusal_and_keeps_the_transcript() -> None:
    analyze = MagicMock(
        side_effect=LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )
    )
    transcript = " ".join(f"Sentence number {i} is here." for i in range(400))  # 2000 words: 20 chunks of 100
    result = {"content": transcript}
    form = SimpleNamespace(
        api_name="openai",
        custom_prompt=None,
        system_prompt=None,
        perform_chunking=True,
        summarize_recursively=False,
        perform_analysis=True,
    )

    with patch("tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.analyze", analyze):
        await persistence._populate_reused_transcript_summary(
            process_result=result,
            form_data=form,
            chunk_options={"method": "words", "max_size": 100, "overlap": 0},
            loop=asyncio.get_running_loop(),
        )

    assert analyze.call_count == 1
    assert result["content"] == transcript
    assert not result.get("summary")
    assert result["video_lite_summary_status"] == "failed"
    assert result["video_lite_summary_error"] == "llm_quota_exceeded"
    assert persistence._merge_video_lite_summary_safe_metadata({}, process_result=result)["video_lite"] == {
        "summary_status": "failed",
        "summary_error": "llm_quota_exceeded",
    }
    assert any("limit_exceeded" in warning for warning in result["warnings"])
```

- [ ] **Step 2: run it.** Expected: `assert 20 == 1`, as measured on `7c94a2efb1`.
- [ ] **Step 3: implement**

```python
from tldw_Server_API.app.core.DB_Management.scope_context import ScopeContext, get_scope, scoped_context
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
```

In the chunk loop of `_synthesize_reused_transcript_analysis_sync`:

```python
        try:
            chunk_summary = analyze(
                api_name,
                chunk_text,
                custom_prompt,
                None,
                system_message=system_prompt,
            )
        except LLMQuotaExceededError:
            raise  # stop at the first refusal; the caller records it and keeps the transcript
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
```

At the end of `_populate_reused_transcript_summary` (Review M4):

```python
            process_result["video_lite_summary_status"] = "failed"
    except LLMQuotaExceededError as refusal:
        # Keep the transcript; record why there is no summary (Decision 3).
        _ensure_warnings_list(process_result).append(f"Summary generation failed during dedupe reuse: {refusal}")
        process_result["video_lite_summary_status"] = "failed"
        process_result["video_lite_summary_error"] = refusal.failure_code
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
        _ensure_warnings_list(process_result).append(f"Summary generation failed during dedupe reuse: {exc}")
        process_result["video_lite_summary_status"] = "failed"
```

In `_merge_video_lite_summary_safe_metadata`, so the code is stored with the item's safe metadata:

```python
    if summary_status == "failed":
        video_lite_meta["summary_status"] = "failed"
        if process_result.get("video_lite_summary_error"):
            video_lite_meta["summary_error"] = process_result["video_lite_summary_error"]
        merged["video_lite"] = video_lite_meta
        return merged
```

The single-call and recursive-summary `analyze` calls in the same function have no `try`, so a refusal there reaches the new clause too.

- [ ] **Step 4: run** the new file, `tests/MediaIngestion_NEW/unit`, `tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py` (it reads `safe_metadata.video_lite`) and `tests/Media/test_media_llm_quota.py`. Expected: all pass (measured: `MediaIngestion_NEW/unit` with `tests/RAG`, `1274 passed, 8 skipped, 1 xfailed`; the other two files passed in the targeted run). In the review's combined run, `MediaIngestion_NEW/unit/test_audio_transcription.py` crawled while other sessions loaded the machine; alone it gave `58 passed, 8 skipped in 6.60s`, and it imports nothing this plan changes.
- [ ] **Step 5: commit** `fix(media): a reused transcript stops summarizing at the first quota refusal and records llm_quota_exceeded (TASK-13638)`, with the trailer.

### Task 7: the Explainer node records the refusal's code (row 25)

**Files:** Modify `tldw_Server_API/app/core/Explainer/jobs.py`: the `except Exception:` at line 278, and `_build_error_metadata` at line 447. Test: `tldw_Server_API/tests/Explainer/test_explainer_jobs.py`. Add `from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError` after the `DB_Management.Explainer_DB` import (line 7): ruff sorts `exceptions` before `Explainer` (Review M1). Then append the test.

- [ ] **Step 1: failing test**

```python
@pytest.mark.asyncio()
@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (RuntimeError("provider unavailable"), "generation_failed"),
        (
            LLMQuotaExceededError(
                category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
            ),
            "llm_quota_exceeded",
        ),
    ],
)
async def test_job_handler_records_the_failure_code_on_the_node(
    explainer_repo: ExplainerRepository,
    error: Exception,
    expected_code: str,
) -> None:
    """The node's error metadata says why generation failed, as the job's error code does (TASK-13638)."""
    session = _create_session(explainer_repo)
    node_id = session.root_node_ids[0]

    async def failing_generator(_prompt):
        raise error

    with pytest.raises(type(error)):
        await handle_explainer_node_expansion_job(
            {
                "id": 103,
                "job_type": EXPLAINER_JOB_TYPE,
                "owner_user_id": "7",
                "payload": {"session_id": session.id, "node_id": node_id, "intent": "explain"},
            },
            repo=explainer_repo,
            generator=failing_generator,
        )

    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[node_id].generation_metadata["error"] == expected_code
```

- [ ] **Step 2: run it.** Expected: `1 failed, 1 passed`. The refusal case fails `'generation_failed' == 'llm_quota_exceeded'`. The `RuntimeError` case is the control and passes on dev.
- [ ] **Step 3: implement**

```python
    except Exception as exc:
        repo.update_node(
            session_id,
            node_id,
            owner_user_id=owner_user_id,
            status=ExplainerNodeStatus.ERROR.value,
            generation_metadata=_build_error_metadata(
                job_id=job_id,
                grounding=session.grounding,
                error=getattr(exc, "failure_code", None) or "generation_failed",
            ),
        )
        raise
```

```python
def _build_error_metadata(*, job_id: str, grounding: str, error: str = "generation_failed") -> dict[str, Any]:
    return {
        "jobId": job_id,
        "grounding": grounding,
        "error": error,
        "generatedAt": _utc_now(),
    }
```

- [ ] **Step 4: run** `tests/Explainer`. Expected: all pass. Then the lint comparison: nothing new (the import placement above is what keeps `I001` away).
- [ ] **Step 5: commit** `fix(explainer): a node that could not be expanded records the refusal's code (TASK-13638)`, with the trailer.

### Task 8: the HyDE legacy branch runs off the event loop (row 26)

**Files:** Modify `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py:4133`. Test: `tldw_Server_API/tests/RAG_NEW/unit/test_hyde.py` (append; the module already imports `threading`, `AsyncMock`, `patch`, `DataSource` and `unified_rag_pipeline`).

- [ ] **Step 1: failing test**

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_unified_pipeline_runs_legacy_hyde_generation_off_the_event_loop():
    """With no credential runtime, the sync HyDE call runs on a worker thread, where the LLM quota check runs (TASK-13638)."""
    loop_thread = threading.get_ident()
    generation_threads: list[int] = []

    def _generate(_query, _provider, _model):
        generation_threads.append(threading.get_ident())
        return "A hypothetical answer with enough words in it."

    class _FakeMediaRetriever:
        async def retrieve_hybrid(self, *args, **kwargs):
            return []

    class _FakeMultiRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {DataSource.MEDIA_DB: _FakeMediaRetriever()}

        async def retrieve(self, *args, **kwargs):
            return []

    with patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever", _FakeMultiRetriever), \
         patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.generate_hypothetical_answer", _generate), \
         patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.hyde_embed_text", new=AsyncMock(return_value=[0.1, 0.2, 0.3])):
        result = await unified_rag_pipeline(
            query="test hyde",
            sources=["media_db"],
            top_k=10,
            enable_hyde=True,
            adaptive_hybrid_weights=False,
            enable_generation=False,
        )

    assert result.metadata.get("hyde_applied") is True
    assert len(generation_threads) == 1
    assert generation_threads[0] != loop_thread
```

- [ ] **Step 2: run it.** Expected: `assert <id> != <id>` with the same thread id on both sides, as measured.
- [ ] **Step 3: implement** (one line becomes two; both are after line 3889)

```python
                if credential_runtime is None:
                    # Off the loop: a sync LLM call on the loop's thread blocks it and cannot be quota-checked.
                    hypo = await asyncio.to_thread(generate_hypothetical_answer, query, hyde_provider, hyde_model)
                    vec = await hyde_embed_text(hypo)
```

- [ ] **Step 4: run** `tests/RAG_NEW/unit/test_hyde.py`, `tests/RAG_NEW/unit/test_hyde_retrieval_merge.py` and `tests/RAG_NEW/unit/test_vector_retriever_hyde.py`. Then check both baselines:

```bash
$PY Helper_Scripts/ci/scope_predicate_ratchet.py
$PY Helper_Scripts/web_scraping_refactor_inventory.py --json Docs/Design/web_scraping_refactor_import_inventory.json --markdown Docs/Design/WebScraping_Refactor_Import_Inventory.md
git status --short Docs/Design/
```

Expected: the tests pass, `28 conditional scope predicates; baseline exact.`, and `git status` shows nothing under `Docs/Design/` (as measured on the revised prototype). If either check reports a change, regenerate (`scope_predicate_ratchet.py --write-baseline`) and commit the regenerated file with this task.
- [ ] **Step 5: commit** `fix(rag): the HyDE legacy branch calls the provider off the event loop, so it is quota-checked (TASK-13638)`, with the trailer.

### Task 9: the operator page, the design note, the benchmark guide, follow-up tasks and close-out (AC2)

**Files:** Modify `Docs/Operations/Usage_Quotas.md`, `Docs/Design/2026-10-09-llm-usage-accounting-design.md` and `Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md` with its `Docs/Published` mirror (Review M7). Backlog: TASK-13638 and two or three new tasks.

Every edit below is found by its quoted text (Review M8); the line numbers are dev's, for orientation only.

- [ ] **Step 1: `Usage_Quotas.md`**
  - **Line 90**, the boundary list: after "evaluation runs (`POST /evaluations/{eval_id}/runs`);" insert " benchmark runs (`POST /evaluations/benchmarks/{name}/run` and `POST /benchmarks/{name}/run`);". Drop this edit if the owner rejects Decision 2.
  - **Line 92**, the HTTP endpoint bullet: after "`/media/{id}/insights` and the manuscript analysis endpoints among them." insert (Review I1):

    > So do Prompt Studio prompt execution (`POST /prompt-studio/prompts/execute`), test runs (`POST /prompt-studio/test-cases/run`) and synchronous evaluations, claims verification (`POST /claims/verify/fva`), and manual character memory extraction (`POST /characters/{id}/memories/extract`).

  - **Line 93**, the workflow bullet: replace the whole bullet, from "  - **A workflow:** an `llm` step fails" to "and the run goes on.", with (Reviews I2, I3, M2, M3):

    > - **A workflow:** an `llm` step fails, and so does the run, with `error` set to `402: {'error': 'limit_exceeded', ...}`. Content generation, NLP, bibliography, image description, summarize, translate, claims extraction and evaluations steps fail the step, and the run, the same way; `web_search` skips its optional query rewrite instead. A refused step is not retried, even when it sets `retry` (the run's events show `step_retry_suppressed`), and a `retry` step does not retry it either. A `parallel` step fails when one of its steps is refused. A step's `on_failure` can route around a refused step. In a `map` or `parallel` step, sibling sub-steps that have started run on after the step fails; their LLM calls are refused before they reach the provider.

    (The bullet keeps its two-space indent under the "Everything else is checked at the call" item.)

  - **Line 95**, the job bullet: replace "A media item that reuses an earlier transcript gets no summary." with (Review M4, I1):

    > A media item that reuses an earlier transcript keeps it and gets no summary: summarizing stops at the first refusal, the item's warnings carry the 402, and its `video_lite` metadata records `summary_status: failed` and `summary_error: llm_quota_exceeded`. A Prompt Studio optimization job fails with `llm_quota_exceeded` and is not retried. A background Prompt Studio evaluation ends `failed`, with `error_message` `llm_quota_exceeded: ...`. An Explainer node that could not be expanded records `error: llm_quota_exceeded`.

  - **Line 98**, shared machinery: replace "is never scored 0.0 by an evaluator," with "is never scored 0.0 by an evaluator or a Prompt Studio evaluation (a benchmark run counts a refused item as `failed`, outside its `average_score`),".
  - **Line 99**: replace the whole bullet that starts "  - **Degraded, not refused:**" with:

    > - **Claims verification is refused, not answered without the LLM:** `/claims/verify/fva` answers 402, and a generated artifact whose claims cannot be checked reports the refusal instead of `claim_verification_failed`. Automatic character memory extraction after a character chat reply runs as part of that reply, which the boundary check already admitted.

  - If the owner keeps the engine retry (Decision 1's second alternative), replace "A refused step is not retried, even when it sets `retry` (the run's events show `step_retry_suppressed`), and a `retry` step does not retry it either." with "A step that sets `retry`, and an `llm` step by default, is retried and refused again without a provider call; a `retry` step does not retry it."

- [ ] **Step 2: the design doc**
  - §4a table: after the row that starts "| Evaluation runs (`POST /evaluations/{eval_id}/runs`) |", add (Review M6: a refused item is no longer scored 0.0 after row 21):

    ```markdown
    | Benchmark runs (`/evaluations/benchmarks/{name}/run`, `/benchmarks/{name}/run`) | Always. Checked only at the call, every item would be refused and counted as failed, and the run would still complete. |
    ```

  - At the end of the file (after "Review changes" and its revisions), add:

    ```markdown

    Revision of 2026-10-18, TASK-13638:

    - **Callers that turned a refusal into degraded output** let it through, or keep their artifact and record the refusal. Benchmark runs join the §4a table.
    - **The workflow engine does not retry a step refused with a 402.** This replaces the 2026-10-13 note that an `llm` step is retried once.
    ```

  - Not mirrored. If the owner rejects Decision 2, drop the table row and the benchmark sentence; if the owner keeps the engine retry, drop the second bullet.

- [ ] **Step 3: the benchmark guide (Review M7)**
  - In `Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md`, "Troubleshooting", insert before "### 404 benchmark not found":

    ```markdown
    ### 402 LLM quota

    - The user is over an LLM limit: `detail.error` is `limit_exceeded`, and `Retry-After` says when to try again. The run is refused before its dataset loads, so nothing is evaluated.
    - A user who reaches the limit during a run has the remaining items refused; they count under `failed` in the summary.

    ```

  - Run `bash Helper_Scripts/refresh_docs_published.sh`. Expected (measured): one changed file, `Docs/Published/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md`.
  - Run `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs`. Expected: `212 passed` (measured with every doc edit of both PRs and the refreshed mirror).
  - If the owner rejects Decision 2, keep only the second bullet, under the heading "### 402 LLM quota during a run".

- [ ] **Step 4: file the follow-up tasks.** Compute the highest task id with the AGENTS.md recipe, extended to page through open PRs with more than 100 files (`gh pr list --json files` returns at most 100 per PR):

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
ADJ="TASK-$((HIGHEST + 1))"; FANOUT="TASK-$((HIGHEST + 2))"; ADDON="TASK-$((HIGHEST + 3))"
```

Measured on 2026-10-18: `13857`, with 10 open PRs over 100 files.

The adjudicator (Review Focus 2):

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "FVA adjudicator calls the sync analyze on the event loop" --id "$ADJ" -l quotas,claims,backend \
  -d "ClaimAdjudicator._call_llm (Claims_Extraction/adjudicator.py, near lines 291-325) calls the sync analyze function (_fva_claims_analyze_call) directly from a coroutine, so the LLM quota check at the call is skipped (LLM usage accounting design section 5) and the server loop blocks for the provider call. Its except Exception in _assess_documents (near line 231) would also swallow a refusal once the call moves off the loop. Found while planning TASK-13638." \
  --ac "The adjudicator's LLM call runs off the event loop (asyncio.to_thread or the bounded adapter pool), with a test" \
  --ac "An LLM quota refusal there is let through, not counted as a failed assessment"
```

Sibling sub-steps (Review I3):

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Workflow map and parallel leave sibling sub-steps running after one fails" --id "$FANOUT" -l workflows,quotas,backend \
  -d "run_map_adapter and run_parallel_adapter (Workflows/adapters/control/flow.py) use asyncio.gather without return_exceptions for map, and for parallel with fail_fast. That raises the first error but does not cancel the other sub-steps: those already started, and those queued behind the concurrency limit, run on after the engine has marked the step and the run failed (probe in the TASK-13638 plan review: map items 1-5 finished after the step raised). Since TASK-13638 every content, NLP and bibliography sub-step raises on an LLM quota refusal, so a map over flashcard_generate reaches this. The siblings' own LLM calls are refused before reaching a provider, but non-LLM sub-steps complete with their side effects." \
  --ac "map and parallel run their sub-steps in an asyncio.TaskGroup, or cancel the remaining tasks, so the first raised error cancels the siblings, with a test that records which siblings finished after the step raised" \
  --ac "parallel with fail_fast off keeps collecting per-step errors, apart from an LLM quota refusal, which fails the step (TASK-13638)"
```

Add-on paths that still discard the artifact (Review M4), only if the owner takes Decision 3's recommendation:

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Add-on LLM work that still discards the user's artifact on a quota refusal" --id "$ADDON" -l quotas,media,backend \
  -d "TASK-13638 Decision 3 keeps the artifact the user asked for and records the refusal when only an add-on LLM step is refused (transcript reuse; plan_audio_analysis.md does the same for audio jobs). These paths still discard the artifact: media ingest jobs, single and batch (the item is marked Error, Processing error: LLMQuotaExceededError; a batch marks every item); /media/ingest-web-content with analysis and /media/process-web-scraping with summarization, which discard the request's results; POST /watchlists/outputs with summarize, which discards them too (Usage_Quotas.md, the HTTP endpoint bullet under Everything else is checked at the call)." \
  --ac "Each listed path keeps the transcript, page or output it produced and records llm_quota_exceeded where the user can see it, with a test per path" \
  --ac "Usage_Quotas.md says what each path keeps"
```

- [ ] **Step 5: close TASK-13638**

Run the close-out checks first (Review B1): `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` (exit 0), `tests/CI/test_test_quality_ratchet.py` (`1 passed`), `tests/Docs` (`212 passed`), and the lint comparison (nothing new). Bandit: `$PY -m bandit -q -r <touched app files>`; bandit is not installed in `.venv`, so use `uvx bandit -q -r <touched app files>` or record the skip.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13638 \
  --append-notes "RED on origin/dev 7c94a2efb1: 23 workflow tests (22 adapter cases DID NOT RAISE, including the evaluations step; the engine test assert 4 == 1); claims 3 failed (verify probe returned VERIFIED and wrote a claims_verify row); Prompt Studio 7 failed (including _call_llm RuntimeError and the execute endpoint's 502); benchmarks 3 failed; memory 1; transcript reuse 1 (20 refused calls); Explainer 1; HyDE 1 (same thread). GREEN: <counts from tasks 1-8>. Neighbourhood suites: <counts, with dev baselines for any failure>. Test-quality ratchet: exit 0." \
  --append-notes "Decisions taken: <the owner's choices for Decisions 1-3>. Follow-ups: $ADJ (FVA adjudicator sync call on the loop), $FANOUT (cancel sibling sub-steps)<, $ADDON (add-on paths that discard the artifact), if Decision 3 was taken>." \
  --append-notes "Bandit on touched app files: <result>." \
  --check-ac 1 --check-ac 2 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Every caller listed in the task, plus Prompt Studio's executor and the workflow evaluations step, lets LLMQuotaExceededError through, or keeps its artifact and records the refusal. Sequential loops stop at the first refusal; fan-outs fail with it, and their siblings' calls are refused before reaching a provider (cancelling the siblings is a follow-up). The workflow engine does not retry a 402. Benchmark runs are checked at the request boundary. Usage_Quotas.md lists what each caller does." \
  -s Done
git add -A backlog/tasks
```

- [ ] **Step 6: commit and PR**

```bash
git add Docs/Operations/Usage_Quotas.md Docs/Design/2026-10-09-llm-usage-accounting-design.md \
  Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md Docs/Published/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md
git commit -m "docs(quotas): what each caller does with an LLM quota refusal; close TASK-13638" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push -u origin fix/llm-quota-refusal-callers
gh pr create --base dev --title "LLM quota refusals reach the caller instead of degraded output (TASK-13638)" --body-file <body>
```

The PR body needs the per-caller table, the decisions taken, and a full "## Change summary" section: the owner's ADR-004 waiver of 2026-10-16 covers TASK-13635 and TASK-13638. The body ends with the PR attribution line. Then request the clean review (`/superpowers:requesting-code-review` in a fresh subagent). Merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset.

---

## PR B — TASK-13635 (Decision 4 recommended: delete the metric)

If the owner takes Alternative A instead, skip Tasks 10 and 11 and use Appendix A.

### Task 10: delete the completeness metric and its aggregate (Review I4)

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_custom_metrics.py`. Delete `evaluate_response_completeness` (lines 200-278) and `evaluate_all_metrics` (lines 432-499), and the imports they alone use: `from loguru import logger` (line 17) and `from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze` (line 20).
- Delete: `tldw_Server_API/tests/RAG/test_rag_custom_metrics_sanitizers.py`. Both of its tests cover only the two deleted methods (the completeness fallback log and `evaluate_all_metrics`' skipped-metric log); no other logging is left in the module to sanitize.
- Modify: `Docs/Design/service-prompt-inventory.md` (line 153), `Docs/Design/RAG_Pipeline_Evaluation.md` (line 43). Neither is mirrored or pinned by a test.

**Interface:** after this task `RAGCustomMetrics` keeps `evaluate_retrieval_coverage`, `evaluate_retrieval_diversity`, `evaluate_source_attribution`, `evaluate_response_coherence`, `evaluate_cost_efficiency` and `aggregate_metrics`; `get_custom_metrics` and `_LightRAGCustomMetrics` are unchanged. `MetricType.RESPONSE_COMPLETENESS` stays, like the enum's other members that no method computes.

- [ ] **Step 0: branch, and amend the task**

```bash
git fetch origin dev --quiet && git switch -c fix/delete-rag-completeness-metric origin/dev
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13635 -s "In Progress" \
  --remove-ac 2 --remove-ac 1 \
  --ac "evaluate_response_completeness and evaluate_all_metrics, which nothing in the app or the WebUI calls, are deleted, so rag_custom_metrics makes no LLM call" \
  --ac "The docs that name the metric (the service prompt inventory, the RAG pipeline evaluation design and the LLM usage accounting design section 4b) say it was deleted" \
  --append-notes "Plan: plan_quota_callers_rag_metric.md (PR B, tasks 10-11). Owner decision 4: delete the metric. The original ACs assumed a caller: git grep on origin/dev 7c94a2efb1 finds none in tldw_Server_API/app, apps/ or the tests other than test_rag_custom_metrics_sanitizers.py, and eval_runner uses only the coverage and diversity metrics. ACs amended to match."
git add -A backlog/tasks
```

The `--remove-ac 2 --remove-ac 1` order, then `--ac`, was checked in a scratch worktree: the task ends with exactly the two new ACs, numbered 1 and 2.

- [ ] **Step 1: confirm what calls them (the check that changes)**

```bash
git grep -n -e evaluate_response_completeness -e evaluate_all_metrics -- tldw_Server_API apps
```

Expected on dev (measured): 7 lines. `rag_custom_metrics.py:200` and `:432` (the definitions), `:453` (the aggregate's call), and `tests/RAG/test_rag_custom_metrics_sanitizers.py:41,63,86,91`. Nothing else. If anything else appears, stop: Decision 4's premise no longer holds.

- [ ] **Step 2: delete**

In `rag_custom_metrics.py`, delete the whole `async def evaluate_response_completeness(` method, from its `def` line through the heuristic fallback's `return CustomMetricResult(...)` (dev lines 200-278, ending just before `async def evaluate_source_attribution(`). Delete the whole `async def evaluate_all_metrics(` method (dev lines 432-499, ending just before `async def _extract_concepts(`). Then the import block reads:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Safe import of embeddings helpers to avoid heavy deps during app import
```

```bash
git rm tldw_Server_API/tests/RAG/test_rag_custom_metrics_sanitizers.py
```

- [ ] **Step 3: the docs**
  - `Docs/Design/service-prompt-inventory.md`, the row that starts "| RAG response-completeness evaluator |": replace its location cell `` `tldw_Server_API/app/core/RAG/rag_custom_metrics.py:200-242` `` with "Deleted in TASK-13635 (was `tldw_Server_API/app/core/RAG/rag_custom_metrics.py:200-242`)". The inventory is historical (its "Review state"), so the row stays.
  - `Docs/Design/RAG_Pipeline_Evaluation.md:43`: replace "`rag_custom_metrics` (attribution, completeness/coherence, response time, costs)." with "`rag_custom_metrics` (attribution, coherence, response time, costs; its completeness metric was deleted in TASK-13635, unused and unable to reach a provider)."

- [ ] **Step 4: verify**

```bash
git grep -n -e evaluate_response_completeness -e evaluate_all_metrics -- tldw_Server_API apps
$PY -m ruff check tldw_Server_API/app/core/RAG/rag_custom_metrics.py
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/RAG --ignore=tldw_Server_API/tests/RAG/test_restricted_postgres_media_retrieval.py
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py tldw_Server_API/tests/Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py
```

Expected (measured on the prototype): `git grep` prints nothing; ruff "All checks passed!"; `tests/RAG` with `tests/MediaIngestion_NEW/unit`, `1274 passed, 8 skipped, 1 xfailed`; the two eval_runner files, which import `get_custom_metrics`, pass (in the targeted run, `507 passed, 1 skipped`). `test_restricted_postgres_media_retrieval.py` is left out because, in the review's dry run, it failed and then hung on a local Postgres another session had started, on unchanged dev as well.
- [ ] **Step 5: commit** `refactor(rag): delete the completeness metric and evaluate_all_metrics, which nothing calls (TASK-13635)`, with the trailer. Stage `backlog/tasks` too, for Step 0's changes.

### Task 11: close TASK-13635

- [ ] **Step 1: update the design note.** In `Docs/Design/2026-10-09-llm-usage-accounting-design.md` §4b, the last bullet's sentence "The RAG custom metric named in an earlier draft makes no provider call: its `analyze` call is missing an argument and falls back to a heuristic (TASK-13635)." becomes:

  > The RAG custom completeness metric named in an earlier draft is deleted: nothing called it, and its `analyze` call could not reach a provider (TASK-13635).

  Not mirrored.
- [ ] **Step 2: run the checks** (Review B1): `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` (exit 0; a deleted test file removes offenses, it cannot add one), `tests/CI/test_test_quality_ratchet.py` (`1 passed`) and `tests/Docs` (`212 passed`, measured with this PR's doc edits applied).
- [ ] **Step 3: close the task**

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13635 \
  --append-notes "Before: git grep found 7 lines (2 definitions, the aggregate's call, 4 lines in test_rag_custom_metrics_sanitizers.py); after: none. Deleted 152 lines and the two-test sanitizer file, which covered only the deleted methods. tests/RAG: <counts>; eval_runner tests: <counts>; test-quality ratchet exit 0." \
  --append-notes "Bandit on rag_custom_metrics.py: <result>." \
  --check-ac 1 --check-ac 2 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "evaluate_response_completeness and its only caller, evaluate_all_metrics, are deleted: nothing in the app or the WebUI called them, and the LLM path raised TypeError before any provider call. rag_custom_metrics now makes no LLM call. The service prompt inventory, the RAG pipeline evaluation design and the LLM usage accounting design say so. Wiring a completeness metric into eval_runner starts from git history." \
  -s Done
git add -A backlog/tasks Docs/Design/2026-10-09-llm-usage-accounting-design.md Docs/Design/service-prompt-inventory.md Docs/Design/RAG_Pipeline_Evaluation.md
git commit -m "docs(rag): the completeness metric is deleted; close TASK-13635" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: PR** to `dev`, with a full "## Change summary" section (the 2026-10-16 waiver covers TASK-13635), the attribution line, and a clean review before merge.
  - Merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset.
  - Both PRs edit the design doc, in different sections (PR A: §4a and the end; PR B: §4b), so a conflict is textual only.

## Out of scope

- **Media ingest and the other add-on paths** that still discard the artifact (Decision 3): filed as a follow-up in Task 9 Step 4, if the owner takes the recommendation.
- **Cancelling sibling sub-steps** in `map` and `parallel` (Review I3): filed in Task 9 Step 4.
- **The FVA adjudicator's sync call on the loop**, filed in Task 9 Step 4.
- **`web_search`'s optional query rewrite**, which keeps degrading (Review Focus 11).
- **The genetic optimizer's fallbacks** and `execute_batch` (Review Focus 6).
- **Stage metadata for the HyDE legacy branch.**
- **PR B: the other heuristic metrics** (`evaluate_source_attribution`, `evaluate_response_coherence`, `evaluate_cost_efficiency`, `aggregate_metrics`) have no caller either once `evaluate_all_metrics` goes. They make no LLM call, so TASK-13635 leaves them.
- **A boundary check for Prompt Studio and FVA.** The design leaves Prompt Studio to §4b. FVA's LLM use depends on the NLI mode, so it is not "always".

---

## Appendix A: PR B if the owner keeps the metric (Decision 4, Alternative A)

Branch `fix/rag-completeness-metric-provider`. These replace Tasks 10 and 11; TASK-13635's ACs stay as they are. The code is complete; measured in the review's dry run on `7c94a2efb1`: RED `4 failed, 1 passed`, GREEN `5 passed`.

### Task 10A: the completeness metric grades with the caller's provider, off the loop

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_custom_metrics.py`. The import at line 20; `evaluate_response_completeness` at lines 200-278; a new `_heuristic_completeness`.
- Modify: `tldw_Server_API/tests/RAG/test_rag_custom_metrics_sanitizers.py`. Its first test mocks the broken call shape.
- Create: `tldw_Server_API/tests/RAG/test_rag_custom_metrics_completeness.py`

**Interface:**

```python
async def evaluate_response_completeness(
    self,
    query: str,
    response: str,
    expected_elements: Optional[list[str]] = None,
    *,
    api_name: Optional[str] = None,
    model: Optional[str] = None,
) -> CustomMetricResult
```

`details` is one of:
- `{"word_count": n, "method": "heuristic", "reason": "no_provider" | "llm_failed"}`;
- `{"missing_elements": ..., "expected_elements_found": ..., "method": "llm"}`.

`evaluate_all_metrics` is unchanged. It passes no provider, so it gets the heuristic, labelled `no_provider`.

- [ ] **Step 0: branch and task**

```bash
git fetch origin dev --quiet && git switch -c fix/rag-completeness-metric-provider origin/dev
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13635 -s "In Progress" --append-notes "Plan: plan_quota_callers_rag_metric.md (PR B, Appendix A); provider per Decision 4, Alternative A."
git add -A backlog/tasks
```

- [ ] **Step 1: failing tests** (`test_rag_custom_metrics_completeness.py`)

```python
"""The RAG completeness metric grades with the provider its caller names, off the event loop (TASK-13635)."""

import threading

import pytest

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.RAG import rag_custom_metrics
from tldw_Server_API.app.core.RAG.rag_custom_metrics import MetricType, RAGCustomMetrics

pytestmark = pytest.mark.unit

QUERY = "What do plants need to make sugar?"
RESPONSE = "Plants need light, water and carbon dioxide."


def _evaluator() -> RAGCustomMetrics:
    return RAGCustomMetrics(embedding_config={"embedding_config": {"default_model_id": ""}})


@pytest.mark.asyncio
async def test_completeness_grades_with_the_named_provider_off_the_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    calls = []

    def fake_analyze(*args, **kwargs):
        calls.append((threading.get_ident(), args, kwargs))
        return "Score: 8\nMissing: None"

    monkeypatch.setattr(rag_custom_metrics, "analyze", fake_analyze)

    result = await _evaluator().evaluate_response_completeness(QUERY, RESPONSE, api_name="openai", model="gpt-4o-mini")

    assert len(calls) == 1
    thread, args, kwargs = calls[0]
    assert thread != loop_thread
    assert args[0] == "openai"
    assert QUERY in args[1] and RESPONSE in args[1]
    assert args[2] is None
    assert kwargs["model_override"] == "gpt-4o-mini"
    assert kwargs["input_is_literal_text"] is True
    assert result.metric_type == MetricType.RESPONSE_COMPLETENESS
    assert result.score == pytest.approx(0.8)
    assert result.details["method"] == "llm"


@pytest.mark.asyncio
async def test_completeness_without_a_provider_keeps_the_heuristic_and_says_so(monkeypatch):
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("no provider was named, so no LLM call is made")

    monkeypatch.setattr(rag_custom_metrics, "analyze", must_not_run)

    result = await _evaluator().evaluate_response_completeness(QUERY, RESPONSE)

    assert result.details == {"word_count": 7, "method": "heuristic", "reason": "no_provider"}
    assert result.score == pytest.approx(7 / 200)


@pytest.mark.asyncio
async def test_completeness_lets_an_llm_quota_refusal_through(monkeypatch):
    def refused(*_args, **_kwargs):
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(rag_custom_metrics, "analyze", refused)

    with pytest.raises(LLMQuotaExceededError):
        await _evaluator().evaluate_response_completeness(QUERY, RESPONSE, api_name="openai")
```

In `test_rag_custom_metrics_sanitizers.py`, `test_response_completeness_heuristic_fallback_log_is_sanitized` keeps its purpose: the fallback log must not leak the query or response. Change it to the real call shape. Replace its `async def broken_analyze(_prompt, _system_prompt):` with:

```python
    def broken_analyze(*_args, **_kwargs):
        raise RuntimeError(f"LLM failed for query={query!r} response={response!r}")
```

its call with:

```python
        result = await evaluator.evaluate_response_completeness(query, response, api_name="openai")
```

and its `details` assertion with:

```python
    assert result.details == {"word_count": 3, "method": "heuristic", "reason": "llm_failed"}
```

- [ ] **Step 2: run them**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/RAG/test_rag_custom_metrics_completeness.py tldw_Server_API/tests/RAG/test_rag_custom_metrics_sanitizers.py
```

Expected: `4 failed, 1 passed`.
- `TypeError: ... unexpected keyword argument 'api_name'` (three times: two new tests and the edited sanitizer test).
- The no-provider test fails on `details`: `{'word_count': 7, 'method': 'heuristic'}`, without `reason`.
- `test_evaluate_all_metrics_skipped_metric_log_is_sanitized` passes.

- [ ] **Step 3: implement**

Add `from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError` above the `Summarization_General_Lib` import. Replace `evaluate_response_completeness` with the following, and add the helper below it:

```python
    async def evaluate_response_completeness(
        self,
        query: str,
        response: str,
        expected_elements: Optional[list[str]] = None,
        *,
        api_name: Optional[str] = None,
        model: Optional[str] = None,
    ) -> CustomMetricResult:
        """
        Evaluate completeness of the generated response.

        Grades with the caller's ``api_name`` and ``model``. With no provider, or when
        the grading call fails, it scores by length and ``details`` says why. An LLM
        quota refusal is raised, not scored.
        """
        if not api_name:
            return self._heuristic_completeness(response, reason="no_provider")

        prompt = f"""
        Evaluate the completeness of this response to the given query.

        Query: {query}

        Response: {response}

        Rate the completeness on a scale of 0-10 where:
        - 0-3: Response is incomplete, missing major elements
        - 4-6: Response covers basics but lacks depth
        - 7-8: Response is mostly complete with minor gaps
        - 9-10: Response is comprehensive and complete

        Also identify any missing elements.

        Format your response as:
        Score: [0-10]
        Missing: [list any missing elements or "None"]
        """

        try:
            # Off the event loop: analyze is sync, and the LLM quota check cannot run on the loop's thread.
            llm_response = await asyncio.to_thread(
                analyze,
                api_name,
                prompt,
                None,
                system_message="You are an expert at evaluating response completeness",
                temp=0.1,
                model_override=model,
                input_is_literal_text=True,
                raise_on_error=True,
            )

            # Parse score
            score_line = [l for l in llm_response.split('\n') if 'Score:' in l]
            score = float(score_line[0].split(':')[1].strip()) / 10 if score_line else 0.5

            # Parse missing elements
            missing_line = [l for l in llm_response.split('\n') if 'Missing:' in l]
            missing = missing_line[0].split(':')[1].strip() if missing_line else ""

            # Check expected elements if provided
            if expected_elements:
                found_elements = [e for e in expected_elements if e.lower() in response.lower()]
                element_score = len(found_elements) / len(expected_elements)
                score = (score + element_score) / 2

            suggestions = []
            if score < 0.7:
                suggestions.append("Response lacks completeness")
                if missing and missing != "None":
                    suggestions.append(f"Consider adding: {missing}")

            return CustomMetricResult(
                metric_type=MetricType.RESPONSE_COMPLETENESS,
                score=score,
                confidence=0.75,
                details={
                    "missing_elements": missing,
                    "expected_elements_found": found_elements if expected_elements else None,
                    "method": "llm",
                },
                suggestions=suggestions
            )

        except LLMQuotaExceededError:
            raise
        except Exception:
            logger.error("Failed to evaluate completeness; using heuristic fallback")
            return self._heuristic_completeness(response, reason="llm_failed")

    @staticmethod
    def _heuristic_completeness(response: str, *, reason: str) -> CustomMetricResult:
        """Score completeness by length; ``reason`` says why the LLM was not used."""
        word_count = len(response.split())
        return CustomMetricResult(
            metric_type=MetricType.RESPONSE_COMPLETENESS,
            score=min(1.0, word_count / 200),  # Assume 200 words is complete
            confidence=0.5,
            details={"word_count": word_count, "method": "heuristic", "reason": reason},
            suggestions=["Could not perform deep evaluation"],
        )
```

`input_data=prompt` with `custom_prompt_arg=None` sends the prompt as the whole user message (`_build_summary_prompt` appends the custom prompt only when it is set). `raise_on_error=True` turns a provider failure into `SummaryProviderError`, which takes the heuristic path, instead of an `"Error: ..."` string reaching the score parser.

- [ ] **Step 4: run** both files, then `tests/RAG` (without `test_restricted_postgres_media_retrieval.py`) and `tests/Evaluations` (without `property/` and `integration/`). Expected: `5 passed`, and the rest unchanged from dev. Rerun any Hypothesis `FailedHealthCheck` on its own before reading anything into it. Then the test-quality ratchet (exit 0).
- [ ] **Step 5: commit** `fix(rag): the completeness metric grades with the caller's provider, off the event loop (TASK-13635)`, with the trailer. Stage `backlog/tasks` too, for Step 0's status change.

### Task 11A: close TASK-13635

- [ ] **Step 1: update the design note.** In §4b, the sentence "The RAG custom metric named in an earlier draft makes no provider call: its `analyze` call is missing an argument and falls back to a heuristic (TASK-13635)." becomes:

  > The RAG completeness metric now grades off the loop with the provider its caller names, and without one keeps its heuristic (TASK-13635).

- [ ] **Step 2: close the task**

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13635 \
  --append-notes "RED on origin/dev: 4 failed (TypeError on api_name; heuristic details had no reason). GREEN: <counts>. Provider: Decision 4 Alternative A. 'Configured provider' is the api_name/model the caller passes, as RAGEvaluator and evaluate_custom_metric take them. No app code calls the metric yet (evaluate_all_metrics has no caller); wiring it into eval_runner is not in this task." \
  --append-notes "Bandit on rag_custom_metrics.py: <result>." \
  --check-ac 1 --check-ac 2 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "evaluate_response_completeness grades through analyze with the caller's provider and model on a worker thread, so the LLM quota check runs; with no provider, or when grading fails, it returns the heuristic with details.reason (no_provider or llm_failed); a quota refusal is raised." \
  -s Done
git add -A backlog/tasks Docs/Design/2026-10-09-llm-usage-accounting-design.md
git commit -m "docs(rag): close TASK-13635" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: PR** to `dev`, as in Task 11 Step 4. Merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset.
