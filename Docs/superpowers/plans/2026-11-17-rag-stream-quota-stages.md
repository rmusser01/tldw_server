# TASK-15488 Implementation Plan: the stream's quota check skips the stages the stream does not run

**Goal:** `/api/v1/rag/search/stream` no longer refuses (402) an over-limit own-key user for post-answer stages the stream never runs.

**Architecture:** `_rag_llm_providers` gets a keyword `stream: bool = False`. A new `_RAG_STREAM_SKIPPED_LLM_STAGES` frozenset (6 names) in `rag_unified.py` lists the `_RAG_OTHER_LLM_STAGES` entries that run only after the answer; with `stream=True` those add no `None` provider. Only the stream call site passes `stream=True`. A test file constant `_STREAM_RUNS_LLM_STAGES` lists the 6 that run, and a drift test pins that the two sets partition `_RAG_OTHER_LLM_STAGES` (as `_NO_OTHER_LLM_CALL` does for the schemas).

**Tech Stack:** FastAPI, pytest.

**Spec:** `backlog/tasks/task-15488 - RAG-stream-quota-check-refuses-for-post-answer-stages-the-stream-no-longer-runs.md`

## Classification (measured by reading `streaming_executor.py` and `unified_pipeline.py` at origin/dev 3883a73712)

The stream's retrieval step calls the pipeline with `enable_generation=False` (`streaming_executor.py:381`); the answer comes from `generate_streaming_response`, which is handed only the claims arguments (`:706-711`).

| Flag | Run by the stream? | Evidence |
|---|---|---|
| enable_query_classification | yes | retrieval, `unified_pipeline.py:3338` |
| enable_research_loop | yes | retrieval, `:3398`, `:3472` |
| enable_document_grading | yes | retrieval, `:5807` |
| enable_evidence_accumulation | yes | retrieval, `:5687` (before the generation block) |
| enable_evidence_chains | yes | "before generation" block, gate has no answer condition, `:6841` |
| enable_claims | yes | streamer, `streaming_executor.py:708` |
| enable_fast_hallucination_check | no | needs `result.generated_answer`, `:7534` |
| enable_utility_grading | no | needs `generated_answer`, `:8590` |
| enable_post_verification | no | needs `generated_answer`, `:8163` |
| enable_multi_turn_synthesis | no | inside the generation block, `:7067`; not passed to the streamer |
| enable_suggestions | no | needs `generated_answer`, `:7346` |
| enable_faithfulness_eval | no | needs `generated_answer`, `:8863` |

`strategy != standard`, `search_depth_mode == quality`, the LLM rerankers and the reformulation are retrieval-side: unchanged for the stream.

## Global Constraints

- Tests make no real outbound network requests. Every pytest run uses the network probe plugin: `PYTHONPATH=<plans dir> NETPROBE_LOG=<file> python -m pytest -p netprobe_15427 ...`. It blocks and logs every outbound DNS lookup and connection; a run must end with 0 log lines for the tests the plan adds or changes.
- A test that drives `/chat/completions`, an LLM adapter or a judge must stub the adapter or the provider call. A commercial provider otherwise calls the real API.
- A failed judge or model call, or a reply with no readable score, is unscored: `score: None`, a sample error, and out of every average. Never 0.0, never a success. This is the owner's rule, recorded 2026-11-10 and 2026-11-12.
- Backlog edits use backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`). Use `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit task files.
- Branch from `origin/dev`; the PR targets `dev`. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- PostgreSQL only through the repo fixtures, with a session-unique `TEST_DB_NAME`. Never restart the server. Redis: never FLUSHALL or SCRIPT FLUSH. (This change touches neither.)
- A mirrored doc needs `bash Helper_Scripts/refresh_docs_published.sh`, with the `Docs/Published` diff in the same PR. (No doc is edited here; see Task 1 Step 5.)
- Shifting lines in a file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` fails backend-required. (`rag_unified.py` is not listed, so no regeneration here.)
- Size-capped functions (the function-size ratchet in `tests/lint`) must not grow past their caps. `_rag_llm_providers` grows by about 4 lines and `unified_search_stream_endpoint` by 2: measure both before and after.
- A plan with embedded diffs fails trailing-whitespace when committed: keep this file free of trailing spaces.
- Environment note (measured): the shared venv lacks `pytest_mock`, which the root `conftest.py` loads. Local runs put an empty `pytest_mock.py` stub on `PYTHONPATH` (outside the repo); the tests here do not use `mocker`.

## Review Focus

1. The six skipped stages: is each really absent from the stream? Check `enable_multi_turn_synthesis` and `enable_fast_hallucination_check` hardest (neither is passed to the streamer, and both need an answer).
2. Fail-closed direction: the stream set can only shrink. A new flag is in neither set until classified, and the drift test fails; the unclassified default for a flag in the list is "run" (still refused), so nothing under-counts.
3. `enable_evidence_accumulation` and `enable_evidence_chains` are kept as "run" because their gates do not need an answer. If accumulation makes no LLM call, listing it only refuses more (the same stance as TASK-15427).
4. `/search`, batch, resume and advanced search keep `stream=False`: they are unchanged.

## Task 0: Start

**Files:** `backlog/tasks/task-15488 ...md`

- [ ] `git fetch origin dev && git checkout -b fix/rag-stream-quota-stages-15488 origin/dev`
- [ ] `PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit TASK-15488 -s "In Progress"`, then `git add -A backlog/tasks` and commit "chore(backlog): start TASK-15488".
- [ ] Push and open a draft PR with base `dev`.
- [ ] Baseline: run `tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py`, `test_rag_stream_parity.py`, `tests/RAG_NEW/unit/test_rag_provider_credentials.py` serially with the probe plugin. Record pass counts.

## Task 1: Skip the post-answer stages on the stream, with the pin

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` (new constant after `_RAG_OTHER_LLM_STAGES` `:215-224`; `_rag_llm_providers` `:225`; the stream call at `:2313-2314`)
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py` (appended after the TASK-15427 drift test; same file, already in a CI shard)

**Interfaces:**
- `_RAG_STREAM_SKIPPED_LLM_STAGES: frozenset[str]` = `enable_fast_hallucination_check`, `enable_utility_grading`, `enable_post_verification`, `enable_multi_turn_synthesis`, `enable_suggestions`, `enable_faithfulness_eval`.
- `_rag_llm_providers(settings, *, stream: bool = False) -> list[str | None]`.
- Test: `_STREAM_RUNS_LLM_STAGES` (6 names), `_POST_ANSWER_STAGES`, `_stream_without_pipeline(monkeypatch)`.

- [ ] **Step 1 (RED): tests.** Apply the test half of `proto_15488.patch`. It adds:
  1. `test_every_other_llm_stage_says_whether_the_stream_runs_it`: `listed - skipped - runs == set()`, `skipped & runs == set()`, `(skipped | runs) - listed == set()`.
  2. `test_rag_stream_admits_an_own_key_answer_with_only_post_answer_stages`: over-limit user with an own `openai` key, `stream_rag_events` patched to yield nothing (no pipeline, no provider call), body with `generation_provider: openai` and the six skipped stages on. Asserts the stream is not 402, `asked == ["openai"]`, and the same body on `/api/v1/rag/search` is 402. Reuses `_over_the_limit_with_own_keys`, which stubs the own-key lookup and the pipeline.
  3. (Test comment: `enable_research_loop` cannot be a body case; the schema returns 422 without `enable_query_classification`.) `test_rag_stream_still_refuses_an_own_key_answer_with_a_stage_the_stream_runs[classification|grading|claims|evidence-chains|evidence-accumulation]`.
  Run: `zsh <scratchpad>/plan15488_t.sh red "-k rag_stream or every_other"` (the script sets the probe plugin, the stub path and `-p no:cacheprovider`).
  Expected on origin/dev (measured): `2 failed, 5 passed` (the pin: `AttributeError` on `_RAG_STREAM_SKIPPED_LLM_STAGES`; the admit test: 402), `netlines: 0`. The five refuse cases pass on dev: they pin AC #3 and must stay green.
- [ ] **Step 2 (GREEN): source.** Apply the source half of the patch:
  - add `_RAG_STREAM_SKIPPED_LLM_STAGES` with a 3-line comment;
  - `_rag_llm_providers(..., *, stream: bool = False)`; the final `any(...)` skips `flag in _RAG_STREAM_SKIPPED_LLM_STAGES` when `stream`; docstring line;
  - the stream endpoint passes `stream=True`.
  Expected (measured): `7 passed, 28 deselected`, `netlines: 0`.
- [ ] **Step 3: wider set.** `git grep -l -E "_RAG_OTHER_LLM_STAGES|_rag_llm_providers|search/stream" -- tldw_Server_API/tests` names 5 files: `RAG/test_restricted_postgres_media_retrieval.py`, `RAG_NEW/integration/test_rag_agentic_api.py` (whole file), `RAG_NEW/integration/test_rag_stream_parity.py`, `RAG_NEW/unit/test_rag_provider_credentials.py`, `Services/test_openapi_contracts.py`. Run each serially with the probe plugin; PostgreSQL ones only through the repo fixtures with a unique `TEST_DB_NAME`. Compare against the Task 0 baseline.
- [ ] **Step 4: lint ratchets.** Measure `_rag_llm_providers` and `unified_search_stream_endpoint` line counts before and after against the function-size ratchet in `tests/lint`. `rag_unified.py` is not in `scope_predicate_baseline.txt` (reviewer-checked), so no baseline change.
- [ ] **Step 5: docs.** `Docs/Operations/Usage_Quotas.md:112` says a request that turns on another stage "is still refused" and names the list. Add one sentence: "On `/search/stream`, stages that run only after the answer (suggestions, post-verification, faithfulness and utility grading, the fast hallucination check, multi-turn synthesis) are not counted, because the stream makes no such call." `Usage_Quotas.md` is not mirrored under `Docs/Published`, so no refresh step. (Not in the prototype.)
- [ ] **Step 6: commit** "fix(usage): the RAG stream's quota check skips the post-answer stages the stream does not run (TASK-15488)" with the trailer.

## Task 2: Verify, close out, review, PR

- [ ] `tests/Docs`, `tests/lint`, `tests/CI` (known env-only failures: `test_runtime_package_discovery.py` errors, `test_jobs_postgres_exports_recorded_coverage` x2).
- [ ] `python Helper_Scripts/ci/test_quality_triage.py --enforce`; shard coverage (no new test file); `check_backlog_task_ids.py` exit 0; Bandit on `rag_unified.py`; `git diff --check`.
- [ ] Backlog via backlog-py: tick ACs 1-3 (AC1: admit test, with `/search` still 402 in the same test; AC2: the pin; AC3: the five refuse cases), `--append-notes` with RED/GREEN counts, tick DoD, final summary, status Done. `git add -A backlog/tasks`.
- [ ] Clean-session final review (`/superpowers:requesting-code-review`), address findings, mark the PR ready, queue behind the controller.

## AC check

| AC | Met by | RED on origin/dev |
|---|---|---|
| 1 | `stream=True` skips the 6; admit test asserts stream not 402 and `/search` 402 | `test_rag_stream_admits_...` fails (402) |
| 2 | `_RAG_STREAM_SKIPPED_LLM_STAGES` + `_STREAM_RUNS_LLM_STAGES` partition test | `test_every_other_llm_stage_says_...` fails (AttributeError) |
| 3 | five `still_refuses` cases (classification, grading, claims, evidence-chains, evidence-accumulation) | pass on dev by design (guard against over-skipping); mutation: adding `enable_document_grading` to the skipped set turns them red |

## Decisions

1. **Is the AC's "classified for the stream route" enough as a skip-set plus a run-set in the test?** Alternative: one list only, with a mutation probe. Reviewer and planner recommendation: keep both (as built); cost is a 6-name test constant.
2. **`enable_evidence_accumulation` kept as "run".** It may make no LLM call at all (TASK-15427 left it listed). Options: leave (fails closed, no new claim) or measure and move it to `_NO_OTHER_LLM_CALL` in a separate task. Reviewer and planner recommendation: leave as "run", guarded by the fifth refuse case (`evidence-accumulation`).
3. ADR-004 Change summary waiver: the controller will ask.

## Overlap with open PRs

Only #3480 ("Add portable large-PDF research with Claims checks and model evals", `codex/political-pdf-vip-fixes`) touches `endpoints/rag_unified.py`, `test_rag_agentic_api.py` or `Usage_Quotas.md` (one of the three). Not measured which hunks; a rebase may be needed after it merges.
