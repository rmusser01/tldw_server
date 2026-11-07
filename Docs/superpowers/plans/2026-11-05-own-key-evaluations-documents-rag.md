# Own-key evaluations, chat documents and RAG (TASK-13956, PR B of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The judge-model evaluations, document generation from a chat and RAG generation let an over-limit user through, unmarked, when every LLM call the request will make runs on the user's own key; a request that also makes a call the boundary cannot place on the user's key keeps the 402.

**Architecture:** `enforce_llm_quota(provider=...)` accepts several providers. An over-limit user is let through only when the request names at least one and every named provider runs on the user's own key; any missing name (None, "") refuses, so "a call the boundary cannot name" is spelled `None` (fail closed in one place). A caller's `own_key` answer counts only for a single provider. Each boundary names its providers before any LLM work: an evaluation its judge provider (a batch, each item's), document generation its provider, RAG the answer's provider (as the generator resolves it) and, with a chat history to reformulate, the reformulation's, plus `None` for any other LLM stage the request turns on. RAG reads the request as the pipeline will run it (`resolve_rag_request(...).payload`: a `rag_profile`'s switches and the server's `Search-Agent` defaults applied) at every site: search already did; stream and batch now resolve it at the check.

**Tech Stack:** Python 3.12, FastAPI, pytest (asyncio auto mode, xdist).

**Spec:** `backlog/tasks/task-13956 - Own-key-LLM-calls-the-router-s-usage-row-and-the-other-quota-boundaries.md` (AC #1 for these families); the planner brief `brief_plan_13956.md` ("Mixed providers ... Fail closed: when not every LLM call of the request runs on the user's own key, the boundary must count"); ADR-083; `Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md`; PR A's plan `plan_13956_a.md` (its split table).

**Measured on:** `origin/dev` at `484e393333` (2026-11-05, the merge of #3420; since `6e2dc0c2dc` it changed only a workflow file and one backlog task file), with `Docs/Operations/Usage_Quotas.md` from PR #3422's head `9791e52407` and `proto_13956_a.patch` applied: the tree `dev` will be once #3422 and PR A have merged. The prototype is `proto_13956_b.patch` (plans dir; relative to that tree; 10 files, +476/-25). Its source and test hunks also apply to `484e393333` on their own (`git apply --check`, measured; PR A's new function in `usage_quota_deps.py` sits at the file's end, outside this PR's hunks); its `Usage_Quotas.md` hunk needs PR A's. Every RED/GREEN count below was measured on that patch: the tests on unchanged code first, then the code. `file:line` citations are for `484e393333` (the same as `6e2dc0c2dc` for every file cited; `usage_quota_deps.py` before PR A's addition at its end).

**Order:** after PR A merges (or build Tasks 1-2 while A is in review, and rebase on `dev` before Task 3). PR C goes last.

---

## What was verified (on `6e2dc0c2dc`; the cited files are unchanged on `484e393333`)

The question for each boundary: does every LLM call of the request run on a stored user key through a credential handle, and can the boundary name each call's provider before any LLM work? Only then does naming the provider admit an own-key user; a call without a handle runs on the server's key (source `None` at the call: checked and counted), so admitting the request would only move the 402 into the middle of the work.

| # | Boundary (`file:line`) | LLM calls and their key | Provider known before LLM work | Verdict |
|---|---|---|---|---|
| V1 | `/evaluations/geval`, `/rag`, `/response-quality` (`evaluations_unified.py:991`, `:1208`, `:1406`) | One judge provider per request, resolved through the request's credential runtime: `_resolve_and_validate_eval_provider` (`:366-413`) issues a handle that the service passes to every judge call (`provider_credentials=credential_handle`, `:1056`, `:1277`, `:1474`; `ms_g_eval.py:370`, `rag_evaluator.py:489-992`, `response_quality_evaluator.py:250-586`). The RAG evaluator's similarity uses embeddings, which the LLM limits neither count nor check. | Yes: `canonical_provider_name((request.api_name or "openai").strip() or "openai")` (`:375`), pure. | name it |
| V2 | `/evaluations/batch` of those types (`:1694-1696`) | Each item's judge provider through the shared runtime (`_extract_provider_and_key`, `:1703-1725`: `item.get("api_name") or "openai"`). Items may name different providers. | Yes, per item. | name every item's |
| V3 | Document generation (`chat_documents.py:362`) | One call with the handle (`:370-426`, `document_generator.py:761-777`). The `async_generation` branch only records a job row (`document_generator.py:1092-1131`); nothing in the app processes it. | Yes: `provider_name` (`:344`). | name it |
| V4 | RAG search, stream, batch, resume, advanced (`rag_unified.py:1480`, `:2253`, `:1688`, `:2088`, `:2387`) | Generation resolves a handle from the request's runtime (`rag_service/generation.py:508-521`, `streaming_executor.py:663-669`). Other LLM stages: query classification and reformulation (`query_classifier.py`; provider `classifier_provider or generation_provider or "openai"`, `unified_pipeline.py:3320`, `:3419`; reformulation runs whenever `chat_history` is given and `enable_query_reformulation`, default True, `:3413`), HyDE (`quality` search depth, `:2602`; provider from config `RAG_HYDE_PROVIDER`, `:3984`), grading and the rewrite loop (`grading_provider or "openai"`, `:5790`, `:5962`), knowledge strips, the fast hallucination and utility graders, claims, post-verification, suggestions, the research loop, media search's reformulation, the LLM rerankers (`llm_scoring`, `two_tier`, `advanced_reranking.py:1548`, `:1884`; the `hybrid` reranker is FlashRank, diversity and multi-criteria, `:1438-1470`), and the agentic strategy. | The answer's: `resolve_generation_defaults(load_and_log_configs() or {}, generation_provider, generation_model)`, the generator's own resolution (`generation.py:776-782`). The reformulation's: from the request. The others: their own fields, config defaults or `"openai"`, chosen inside the pipeline. | name the answer's and the reformulation's; any other stage counts |
| V4b | Where RAG reads its switches (review B I1) | Search checks the resolved payload (`standard_bundle.resolved_request.payload`, `:1468-1480`). Stream checks before `_build_standard_request_bundle` (`:2253`, bundle at `:2266`) but runs the resolved request (profile switches and `Search-Agent` defaults, `request_resolution.py:130-241`; `streaming_executor.py:368-381`, `:705`); batch checks `request.enable_generation` (`:1686-1688`) and runs `_build_batch_request_bundle` (`:394-416`, `search_agent_allowed_fields=_BATCH_ROUND2_DEFAULT_FIELDS`). Reproduced on the first prototype (raw `request.model_dump()` at both): over-limit with an own OpenAI key, `{"rag_profile": "accuracy"}` (claims, the `two_tier` reranker, decomposition, structured responses, `profiles.py`) is refused on search and admitted on the stream; a `SEARCH_SUGGESTIONS` default is admitted on a batch. `resolve_rag_request` does no LLM work and is already imported (`:102`). | resolve at the check |
| V5 | RAG ablation (`rag_unified.py:823`) | Four pipeline runs, two of them agentic (`:858-905`). | Not every call. | unchanged |
| V6 | `enforce_llm_quota` (`API_Deps/usage_quota_deps.py:90-107`) | Takes one provider; `if provider and (own_key ... or users_own_key_serves(...))`. Reproduced: a list of providers is passed through to `users_own_key_serves` as one value (`TypeError: unhashable type: 'list'` in the test stub). With a list, a caller's `own_key=True` would admit without asking about each name (review B M6; no caller passes both). | | extend |
| V7 | Frozen functions | `function_size_baseline.json`: `generate_document` 466 (measured 466 before and after), `batch_evaluate` 547 (547, 547). Not frozen: `evaluate_geval` 214 → 215, `evaluate_rag` 195 → 196, `evaluate_response_quality` 199 → 200; the RAG handlers are not frozen (only `get_capabilities` is). | | constraint |
| V8 | Open PRs on these files (re-checked 2026-11-05 ~12:00, `gh api .../pulls/<n>/files --paginate` for all 38 open PRs on `dev`) | None touches `usage_quota_deps.py`, `evaluations_unified.py`, `chat_documents.py`, `rag_unified.py` or any of this PR's test files. #3394 [`82a7534446`] touches `function_size_baseline.json` (drops an unrelated entry). #3422 [`9791e52407`] edits `Usage_Quotas.md`'s Service tiers, Gemini and "Streams without a usage chunk" bullets, not the own-key bullet this PR replaces (measured: this PR's docs hunk applies on #3422's text with PR A's). | | note |

## Decisions

1. **ADR-004 Change summary.** AI-authored. **Answered (owner, 2026-11-05):** the waiver covers all three TASK-13956 PRs; the PR body records it as a full `## Change summary` section.
2. **Several providers, fail closed in one place (ruling).** `enforce_llm_quota(provider=)` takes a name or a collection. An over-limit user is let through only when the collection is non-empty, has no missing name, and every distinct provider runs on the user's own key (looked up once each, stopping at the first that does not). A boundary that cannot name a call passes `None` for it. Alternative: a second keyword (`providers=`); rejected, one parameter keeps one rule. `own_key=` stays a single-provider shortcut for `/chat/completions`: with a collection it is ignored and each name is looked up (review B M6; pinned by `test_own_key_answers_for_a_single_provider_only`).
3. **RAG: the answer, the reformulation, or the 402 (ruling).** The RAG pipeline picks many stages' providers inside itself (presets, config defaults, `"openai"` fallbacks). The boundary names what it can name exactly (the answer, the reformulation) and counts the rest: `_RAG_OTHER_LLM_STAGES` lists the request switches whose stage makes another LLM call; the `quality` search depth (HyDE, the two-tier reranker, web fallback), a non-standard strategy and the `llm_scoring`/`two_tier` rerankers count too. Some listed switches may make no LLM call (intent routing, evidence accumulation); listing them only refuses more. Every site reads those switches from the resolved request, so a `rag_profile` or a server `Search-Agent` default that turns a stage on counts (V4b). A stage the list misses, or one a pipeline-internal config default turns on (`RAG_*` settings the resolution does not read), is still checked at its own call: refused there and never counted, as a failover is on `/chat/completions` (`ponytail:` comment). Alternatives: name every stage's provider (an audit of a 9,000-line pipeline that drifts with each new stage); admit on the answer's provider alone and let other stages be refused at their calls (they degrade with `failure_code: provider_unavailable` instead of a 402; rejected by the brief's fail-closed rule).
4. **Advanced search names the default answer provider.** `advanced_search` (`unified_pipeline.py:9471-9498`) turns on query expansion (acronym, synonym, domain: no LLM), the hybrid reranker (no LLM), citations, table processing and generation, so its only LLM call is the answer on the default provider: `_rag_llm_providers({})`.
5. **Evaluation token reservations are unchanged** (owner decision, 2026-10-29): only the boundary changes; `limits.evaluation_tokens_per_day` keeps counting own-key calls.

## Global Constraints

Verbatim from the brief:
- Tests make no real outbound network requests; stub the provider adapter or its transport. A test that reaches a commercial provider through an endpoint (character chat, RAG generation, media analysis) calls the real API unless the adapter or call is stubbed.
- Function-size ratchet `tldw_Server_API/tests/lint/function_size_baseline.json`: frozen functions must not grow. Several of these endpoints are large; check whether each touched function is frozen and plan the change so it does not grow (extract a helper if needed).
- Backlog:
  - edit TASK-13956 with backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`);
  - `--append-notes`, never `--notes`; stage with `git add -A backlog/tasks`; never hand-edit task files;
  - if a new task must be filed, its explicit id comes from `zsh <plans dir>/task_next.sh` (prints the HIGHEST id in use; the new id is that plus one), after `mkdir <scratchpad>/id_lock`; release it once the PR's file list shows it.
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
  - Character chat: the Frontend E2E journeys are not required checks, and a change there broke character chat once (#3191). The plan's PR step diffs the PR's E2E failures against a recent dev run before merging.
- Agent shells don't keep the venv `PATH`: prefix commands that spawn `python` with `PATH="$(dirname $PY):$PATH"`.

Plan-specific:
- `$PY` is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`.
- `scope_predicate_baseline.txt` lists none of this PR's files (checked on `484e393333`).
- No trailing whitespace in this plan or the code; `git diff --cached --check` before each commit.
- Compatibility: `enforce_llm_quota`'s `provider` keeps accepting a string; the existing callers (`chat.py:4426`, `messages.py:2078`, PR A's character chat) are unchanged. No setting, schema, route or OpenAPI change.

## Review Focus

1. **A batch whose items name different judge providers, one of them not on the user's own key.** Expected: the 402 for the whole batch, before the rate limiter reserves anything. Pinned by `test_batch_evaluation_admits_an_over_limit_user_only_when_every_items_judge_is_their_own_key[anthropic]`. Task 1.
2. **A RAG chat with history, the reformulation on a provider the user has no key for.** Expected: the 402. Pinned by `test_rag_search_still_refuses_an_own_key_answer_with_another_llm_stage[history-on-a-server-key]`; the admitted twin `[reformulated-history]`. Task 2.
3. **A RAG request that turns on another LLM stage (claims, grading, an LLM reranker, the `quality` depth), itself or through a `rag_profile` or a server default, on any RAG route.** Expected: the 402 before any retrieval. Pinned by the four other cases of the same test (search), `test_rag_search_stream_names_the_providers_of_the_resolved_request[accuracy]` (stream, a profile) and `test_rag_batch_names_the_providers_of_the_resolved_request[suggestions-by-default]` (batch, a server default); the admitted twins `[fast]` and `[no-default]` pin that the resolution does not over-refuse. Task 2.
4. **An over-limit user on a server key, any of these endpoints.** Expected: the same 402 as before. Pinned by the existing `test_sync_evaluation_llm_quota_refuses_before_the_rate_limiter` (3 routes), `test_batch_evaluation_llm_quota_refuses_llm_types_and_admits_propositions`, `test_document_generate_refuses_an_over_quota_user_before_generation` (sync and async), `test_rag_search_with_generation_refuses_an_over_quota_user`, and the admit tests' own refused cases (`anthropic`). Tasks 1-2.
5. **An evaluation item without `api_name`.** Expected: judged by OpenAI, so the boundary asks about OpenAI (once, even when another item names it). Pinned by the `default` case of the batch test. Task 1.

Also for the reviewer:
- `_RAG_OTHER_LLM_STAGES` is a denylist; a missed stage degrades (checked at its call), it never spends past the limit (Decision 3).
- Stream and batch now resolve the request once at the check and again in their bundle (`resolve_rag_request` is pure: config and env reads, no LLM, no database); search reuses its bundle's resolution.
- The providers are computed before the check, for every generation request: one `load_and_log_configs()` dict build per request (its parser is `lru_cache`d), as the generator already does once per request. A lazy form (a callable) was not worth the second shape.
- Only `/rag/search`, the stream and the batch have admit tests; resume and advanced name their providers the same way and were probed admitted by the plan reviewer (review B M7).

## Files

- Modify: `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py` (`enforce_llm_quota`, `:90-107`; new `_own_key_serves_each`; import `Collection`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py` (new `_eval_provider_name`, `_batch_eval_providers` above `_resolve_and_validate_eval_provider` `:366`; `:375`, `:990-991`, `:1207-1208`, `:1405-1406`, `:1695-1696`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat_documents.py` (`:361-362`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` (import `Mapping`; new `_RAG_OTHER_LLM_STAGES`, `_rag_llm_providers`, `_resolved_rag_settings` after `_rag_provider_http_exception` `:196-209`; `:1479-1480`, `:1686-1688`, `:2088`, `:2252-2253`, `:2386-2387`)
- Modify tests: `tldw_Server_API/tests/Usage/test_llm_quota_decision.py`, `tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`, `tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py`, `tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py`, `tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py`
- Docs (Task 3): `Docs/Operations/Usage_Quotas.md`
- Create (Task 0): `Docs/superpowers/plans/2026-11-05-own-key-evaluations-documents-rag.md`

## Test commands

The focused set (`$FOCUSED_B`):

```text
tldw_Server_API/tests/Usage/test_llm_quota_decision.py::test_several_providers_are_let_through_only_when_each_runs_on_the_users_own_key
tldw_Server_API/tests/Usage/test_llm_quota_decision.py::test_own_key_answers_for_a_single_provider_only
tldw_Server_API/tests/Usage/test_llm_quota_decision.py::test_an_over_limit_user_whose_provider_runs_on_their_own_key_is_let_through_unmarked
tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py::test_sync_evaluation_admits_an_over_limit_user_whose_judge_runs_on_their_own_key
tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py::test_batch_evaluation_admits_an_over_limit_user_only_when_every_items_judge_is_their_own_key
tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py::test_sync_evaluation_llm_quota_refuses_before_the_rate_limiter
tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py::test_batch_evaluation_llm_quota_refuses_llm_types_and_admits_propositions
tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py::test_document_generate_admits_an_over_limit_user_whose_provider_runs_on_their_own_key
tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py::test_document_generate_refuses_an_over_quota_user_before_generation
tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py::test_rag_search_admits_an_over_limit_user_whose_llm_calls_run_on_their_own_keys
tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py::test_rag_search_still_refuses_an_own_key_answer_with_another_llm_stage
tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py::test_rag_batch_names_the_providers_of_the_resolved_request
tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py::test_rag_search_with_generation_refuses_an_over_quota_user
tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py::test_rag_search_stream_names_the_providers_of_the_resolved_request
```

```bash
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -rf $FOCUSED_B
```

Measured on `dev` with PR A (`484e393333` + #3422's `Usage_Quotas.md` + `proto_13956_a.patch`): tests on base code `15 failed, 15 passed`; with the prototype `30 passed`. No test in the set, and neither Task 2 Step 6 mutation, opens a connection to a non-loopback address (the socket probe of PR A's Test commands).

**The wider set:**

```bash
git grep --untracked -l -E 'enforce_llm_quota|usage_quota_deps|users_own_key_serves|evaluations_unified|eval_unified|/api/v1/evaluations/(geval|rag|response-quality|batch)|chat_documents|chat/documents/generate|rag_unified|/api/v1/rag/(search|batch|advanced|ablate)|_resolve_and_validate_eval_provider|resolve_generation_defaults' -- 'tldw_Server_API/tests/*.py' | grep '/test_[^/]*\.py$'
```

The file list depends on the tree: 100 files on `484e393333`, 102 once PR A is in (PR A's edits make `Character_Chat_NEW/unit/test_chat_completion_precheck.py` and `Chat_NEW/integration/test_roleplay_resume_completion.py` match), and the same 102 with this PR. Split them: the PostgreSQL part is the 2 files whose name contains `postgres` (`Evaluations/test_evaluations_postgres_crud.py`, `RAG/test_restricted_postgres_media_retrieval.py`; under `TLDW_TEST_NO_DOCKER=1` they skip, review B M3), run serially through the repo fixture; the SQLite part is the other 100, run with `-n 8 --dist loadfile` as in PR A:

```bash
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -n 8 --dist loadfile -rf $(cat <scratchpad>/wider_b_sqlite.txt)
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TEST_DB_NAME=tldw_test_13956_b_$$ /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -rf -p no:xdist $(cat <scratchpad>/wider_b_pg.txt)
```

Measured on the 102-file list, on `dev` with PR A (the Task 0 baseline after A merges) and with this PR (review B M1: the earlier numbers were taken on this list and labelled as base). SQLite part: base `3 failed, 1484 passed, 161 skipped` in 3:41; prototype `4 failed, 1505 passed, 161 skipped, 1 error` in 3:40 (the 23 new test ids pass). PostgreSQL part: `70 passed` on both. Expect Task 0's baseline plus this PR's 23 new ids, and no failure outside this list:
- `e2e/test_rbac_multi_user.py::TestMultiUserIsolation::test_10_...`, `test_11_...`, `test_12_...`: `httpx.ConnectError: [Errno 61] Connection refused` on base and prototype (they need a live server).
- `Evaluations/integration/test_embeddings_abtest_multi_user_api.py::test_dedicated_abtest_create_uses_string_tenant_scope_db` (`assert None is not None` on the tenant DB's A/B test) and a setup error in `RAG_NEW/integration/test_rag_health_endpoints.py::test_rag_regression_check_get_sanitizes_unexpected_error` (`DatabaseLockError: Database is temporarily locked` at app startup): both under `-n 8` load, both files pass alone on the prototype (`15 passed`, measured); neither names a symbol this PR changes.
- `RAG_NEW/integration/test_rag_doc_researcher_api.py::test_rag_search_natural_language_media_fallback` (`assert docs`) failed under `-n 8` for the plan reviewer on base and on the first prototype and passed alone (review B M2); it passed in both runs here.

If any of these fails in Task 3, re-run its file alone before judging.

---

## Task 0: Branch, plan, draft PR, baseline

- [ ] **Step 1: Branch** from `origin/dev` after PR A merges: `git fetch origin dev && git switch -c fix/own-key-evals-docs-rag-13956 origin/dev`. (If PR A is still open, branch now and rebase before Task 3.)
- [ ] **Step 2: Re-check open PRs** for this plan's files (as PR A's Task 0 Step 2).
- [ ] **Step 3: Note** on TASK-13956: `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13956 --append-notes "Plan, PR B of 3: Docs/superpowers/plans/2026-11-05-own-key-evaluations-documents-rag.md."`
- [ ] **Step 4: Plan file** at `Docs/superpowers/plans/2026-11-05-own-key-evaluations-documents-rag.md`; `git add -A backlog/tasks Docs/superpowers/plans && git diff --cached --check`; commit (message 0).
- [ ] **Step 5: Draft PR** (`--base dev`, title `fix(usage): own-key evaluations, chat documents and RAG (TASK-13956, 2 of 3)`).
- [ ] **Step 6: Baseline** the wider set (Test commands: the SQLite part, then the two PostgreSQL files serially) on the branch before any code change; record the counts and every failing id. Expected after PR A merges: SQLite `3 failed, 1484 passed, 161 skipped` (the rbac e2e ids), PostgreSQL `70 passed` (measured on `484e393333` with A; #3422's own tests may add to the count).

## Task 1: Several providers at the boundary; evaluations and chat documents name theirs

**Files:**
- Modify: `usage_quota_deps.py`, `evaluations_unified.py`, `chat_documents.py`
- Test: `tldw_Server_API/tests/Usage/test_llm_quota_decision.py`, `tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`, `tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py`

**Interfaces:**
- Produces: `enforce_llm_quota(user_id: Any, estimated_tokens: int = 1, *, provider: str | Collection[str | None] | None = None, own_key: bool | None = None) -> None`; `_eval_provider_name(api_name: Any) -> str`; `_batch_eval_providers(items: Any) -> list[str]`.
- Consumes: `users_own_key_serves(user_id, provider) -> bool` (`usage_quota_deps.py:53`).

- [ ] **Step 1: Write the failing tests.**

(a) `test_llm_quota_decision.py`, after `test_an_over_limit_user_whose_provider_runs_on_their_own_key_is_let_through_unmarked`:

```python
@pytest.mark.parametrize(
    ("providers", "admitted", "asked"),
    [
        (["openai", "groq", "openai"], True, ["openai", "groq"]),
        (["openai", "anthropic", "groq"], False, ["openai", "anthropic"]),
        (["openai", None], False, []),
        ([], False, []),
    ],
    ids=["each-own-key", "one-not", "one-unnamed", "none-named"],
)
async def test_several_providers_are_let_through_only_when_each_runs_on_the_users_own_key(
    monkeypatch: pytest.MonkeyPatch, providers: list[str | None], admitted: bool, asked: list[str]
) -> None:
    """A request naming several providers is admitted past a limit only if every one is the user's own (TASK-13956)."""
    looked_up: list[str] = []

    async def _own_key(_user_id: object, provider: str) -> bool:
        """The user has their own key for openai and groq."""
        looked_up.append(provider)
        return provider in {"openai", "groq"}

    async def _enforce() -> bool:
        await usage_quota_deps.enforce_llm_quota(7, provider=providers)
        return current_llm_call_context().prechecked

    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    _set_limits(monkeypatch, tokens=0)
    if admitted:
        assert await asyncio.create_task(_enforce()) is False
    else:
        with pytest.raises(LLMQuotaExceededError):
            await _enforce()
    assert looked_up == asked


async def test_own_key_answers_for_a_single_provider_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller's ``own_key`` answer is not taken for several providers: each is looked up (TASK-13956)."""
    looked_up: list[str] = []

    async def _own_key(_user_id: object, provider: str) -> bool:
        """The user has no own key for anthropic."""
        looked_up.append(provider)
        return False

    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    _set_limits(monkeypatch, tokens=0)
    with pytest.raises(LLMQuotaExceededError) as refused:
        await usage_quota_deps.enforce_llm_quota(7, provider=["anthropic"], own_key=True)
    assert refused.value.status_code == 402
    assert looked_up == ["anthropic"]
```

(The status assertion is the real output beside the collector: without it `test_quality_triage.py --enforce` flags the test `tautology_suspect`, measured.)

(b) `test_evaluations_stage1_route_and_error_regressions.py`, after `test_batch_evaluation_llm_quota_refuses_llm_types_and_admits_propositions`:

```python
def _past_the_limit_on_own_keys(monkeypatch, own: set[str]) -> tuple[list[str], list[bool]]:
    """An over-limit user with their own key for the providers in ``own``; the rate limiter, next after
    the quota check, answers 418 and records whether the request was marked admitted."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    asked: list[str] = []
    marked: list[bool] = []

    async def _own_key(_uid, provider):
        """Stands in for the lookup (the boundary is under test)."""
        asked.append(provider)
        return provider in own

    def _limiter_reached(_uid):
        marked.append(current_llm_call_context().prechecked)
        raise HTTPException(status_code=418, detail="the quota check let the request through")

    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", _limiter_reached)
    refuse_llm_work(monkeypatch)
    return asked, marked


@pytest.mark.parametrize("route", ["geval", "rag", "response-quality"])
def test_sync_evaluation_admits_an_over_limit_user_whose_judge_runs_on_their_own_key(monkeypatch, route):
    """Past a limit, an evaluation whose judge provider is the user's own key goes on, unmarked (TASK-13956)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402

    app = _build_eval_only_app(monkeypatch)
    asked, marked = _past_the_limit_on_own_keys(monkeypatch, {"openai"})

    with TestClient(app) as client:
        admitted = client.post(f"/api/v1/evaluations/{route}", json=_LLM_EVAL_BODIES[route])
        refused = client.post(f"/api/v1/evaluations/{route}", json={**_LLM_EVAL_BODIES[route], "api_name": "anthropic"})

    assert admitted.status_code == 418, admitted.text
    assert marked == [False]
    assert_llm_quota_402(refused)
    assert asked == ["openai", "anthropic"]


@pytest.mark.parametrize(
    ("second_provider", "admitted", "looked_up"),
    [
        ("groq", True, ["openai", "groq"]),
        ("anthropic", False, ["openai", "anthropic"]),
        (None, True, ["openai"]),  # an item without api_name is judged by OpenAI, so it is looked up once
    ],
    ids=["groq", "anthropic", "default"],
)
def test_batch_evaluation_admits_an_over_limit_user_only_when_every_items_judge_is_their_own_key(
    monkeypatch, second_provider, admitted, looked_up
):
    """A batch names each item's provider; one item off the user's own keys refuses the batch (TASK-13956)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402

    app = _build_eval_only_app(monkeypatch)
    asked, marked = _past_the_limit_on_own_keys(monkeypatch, {"openai", "groq"})
    second = {**_LLM_EVAL_BODIES["geval"], "api_name": second_provider}

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/batch",
            json={"evaluation_type": "geval", "parallel_workers": 1, "items": [_LLM_EVAL_BODIES["geval"], second]},
        )

    if admitted:
        assert response.status_code == 418, response.text
        assert marked == [False]
    else:
        assert_llm_quota_402(response)
    assert asked == looked_up
```

(c) `test_document_generation_endpoints.py`, after `test_document_generate_refuses_an_over_quota_user_before_generation`:

```python
def test_document_generate_admits_an_over_limit_user_whose_provider_runs_on_their_own_key(
    authenticated_client, monkeypatch
):
    """Past a limit, a document whose provider runs on the user's own key is generated, unmarked (TASK-13956)."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    asked: list[str] = []
    marked: list[bool] = []

    async def own_key(_uid, provider):
        """Stands in for the lookup answering True for the request's provider (the boundary is under test)."""
        asked.append(provider)
        return provider == "openai"

    class OwnKeyRuntime:
        def __init__(self, **_kwargs):
            return None

        async def resolve(self, provider, *, model=None):
            return SimpleNamespace(provider=provider, api_key="sk-own", app_config={}, credentials_resolved=True)

        async def mark_used(self, _handle):
            return None

        async def close(self):
            return None

    class OwnKeyStubService:
        def __init__(self, db):
            self._db = db

        def generate_document(self, *args, **kwargs):
            marked.append(current_llm_call_context().prechecked)
            return "Generated on your own key."

        def get_generated_documents(self, conversation_id=None, document_type=None, **_kwargs):
            return [
                {
                    "id": 1,
                    "conversation_id": conversation_id,
                    "title": "Doc",
                    "content": "Generated on your own key.",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "generation_time_ms": 1,
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                }
            ]

    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs

    monkeypatch.setattr(chat_docs, "ProviderCredentialRuntime", OwnKeyRuntime)
    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: OwnKeyStubService
    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", own_key)
    refuse_llm_work(monkeypatch)

    response = authenticated_client.post("/api/v1/chat/documents/generate", json=_make_payload())

    assert response.status_code == 200, response.text
    assert asked == ["openai"]
    assert marked == [False]
    response.close()
```

- [ ] **Step 2: Run them; they fail.** The command is `$FOCUSED_B`'s first nine ids. Expected (measured on `dev` with PR A): `11 failed, 7 passed`: the decision test's three named cases (`TypeError: unhashable type: 'list'`), `test_own_key_answers_for_a_single_provider_only` (`DID NOT RAISE LLMQuotaExceededError`: today's code takes `own_key=True` for the list), the three sync routes and the three batch cases (`402 == 418`, or `asked` for the refused batch), the document test (`402 == 200`); `none-named`, TASK-13955's own-key decision test and the five existing refusal cases pass.
- [ ] **Step 3: `usage_quota_deps.py`.** Add `from collections.abc import Collection` above `from dataclasses import asdict`. Replace `enforce_llm_quota` with:

```python
async def enforce_llm_quota(
    user_id: Any,
    estimated_tokens: int = 1,
    *,
    provider: str | Collection[str | None] | None = None,
    own_key: bool | None = None,
) -> None:
    """402 limit_exceeded when the user is over an LLM quota; otherwise mark the request admitted (spec §4a).

    An over-limit user whose ``provider`` runs on their own key is let through
    unmarked: each call is then checked where its key is known, and a call on
    the user's own key is not checked (TASK-13955). ``provider`` names the
    provider of every LLM call the request makes: one name, or several (an
    evaluation batch's items, TASK-13956). A request that names none, or names
    one the user has no own key for, is refused. ``own_key`` is the caller's
    ``users_own_key_serves`` answer for a single provider, when it already has
    one; None looks it up, and so does a collection, whatever ``own_key`` says.
    """
    refusal = await llm_quota_refusal(user_id, estimated_tokens)
    if refusal is None:
        mark_llm_quota_prechecked()
        return
    names = [provider] if isinstance(provider, str) else list(provider or ())
    if not isinstance(provider, str):
        own_key = None  # a caller's answer is for one provider; several are each looked up
    if names and all(names) and (own_key if own_key is not None else await _own_key_serves_each(user_id, names)):
        return
    raise LLMQuotaExceededError(**asdict(refusal))


async def _own_key_serves_each(user_id: Any, providers: list[str]) -> bool:
    """``users_own_key_serves`` for each provider once, stopping at the first that is not the user's own key."""
    for name in dict.fromkeys(providers):
        if not await users_own_key_serves(user_id, name):
            return False
    return True
```

- [ ] **Step 4: `evaluations_unified.py`.** Above `async def _resolve_and_validate_eval_provider(`, add:

```python
def _eval_provider_name(api_name: Any) -> str:
    """The judge provider an evaluation names, as its credential resolution reads it (OpenAI by default)."""
    return canonical_provider_name(str(api_name or "").strip() or "openai")


def _batch_eval_providers(items: Any) -> list[str]:
    """Each batch item's judge provider, as ``batch_evaluate`` resolves it (a non-dict item is OpenAI's)."""
    return [_eval_provider_name(item.get("api_name") if isinstance(item, dict) else None) for item in items or []]
```

In `_resolve_and_validate_eval_provider`, replace `provider_name = canonical_provider_name((request.api_name or "openai").strip() or "openai")` with `provider_name = _eval_provider_name(request.api_name)`. Replace each of the three blocks (geval, rag, response-quality)

```python
        # Usage quotas (LLM usage accounting spec §4a), before the rate limiter reserves anything.
        await enforce_llm_quota(current_user.id)
```

with

```python
        # Usage quotas (LLM usage accounting spec §4a), before the rate limiter reserves anything. An
        # over-limit user whose judge provider runs on their own key is let through unmarked (TASK-13956).
        await enforce_llm_quota(current_user.id, provider=_eval_provider_name(request.api_name))
```

and in `batch_evaluate` (frozen at 547: two lines for two lines) replace

```python
            # Usage quotas (LLM usage accounting spec §4a): these types make LLM calls inside the request.
            await enforce_llm_quota(current_user.id)
```

with

```python
            # Usage quotas (§4a): admitted past a limit only when every item's provider is the user's own key.
            await enforce_llm_quota(current_user.id, provider=_batch_eval_providers(request.items))
```

- [ ] **Step 5: `chat_documents.py`** (frozen at 466: one line for one line). Replace

```python
        # Usage quotas (LLM usage accounting spec §4a), before the job branch and any generation.
        await enforce_llm_quota(user_id_int)
```

with

```python
        # Usage quotas (§4a), before the job branch and any generation; an own-key provider is let through unmarked.
        await enforce_llm_quota(user_id_int, provider=provider_key)
```

- [ ] **Step 6: Run Step 2's command.** Expected: `18 passed`. Then `tldw_Server_API/tests/lint/test_function_size_ratchet.py`: `13 passed`. Mutation probe (revert after): delete the two `if not isinstance(provider, str):` / `own_key = None` lines → `test_own_key_answers_for_a_single_provider_only` fails with `DID NOT RAISE` (measured).
- [ ] **Step 7: Commit** (message 1).

## Task 2: RAG names the answer's provider (and the reformulation's), from the resolved request

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py`, `tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py`

**Interfaces:**
- Consumes: Task 1's `enforce_llm_quota(provider=list)`; `resolve_generation_defaults(config, provider=None, model=None) -> tuple[str, str | None]` (`RAG/rag_service/generation_defaults.py:10`); `load_and_log_configs()` (`core/config.py`); `resolve_rag_request(request, *, current_user, single_user_id_resolver, search_agent_setting_fn, search_agent_allowed_fields) -> ResolvedRAGRequest` (`RAG/rag_service/request_resolution.py:202`, already imported by `rag_unified.py`).
- Produces: `_RAG_OTHER_LLM_STAGES: tuple[str, ...]`; `_rag_llm_providers(settings: Mapping[str, Any]) -> list[str | None]`; `_resolved_rag_settings(request: Any, current_user: Any, allowed_fields: set[str] | None = None) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing tests** in `test_rag_agentic_api.py`, after `test_rag_search_profile_without_explicit_generation_is_not_refused_by_the_llm_quota`:

```python
def _over_the_limit_with_own_keys(monkeypatch, own: set[str]) -> tuple[list[str], list[bool]]:
    """An over-limit user with their own key for ``own``; the pipeline records whether it ran marked admitted."""
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    asked: list[str] = []
    marked: list[bool] = []

    async def _own_key(_uid, provider):
        """Stands in for the lookup (the boundary is under test)."""
        asked.append(provider)
        return provider in own

    async def _fake_pipeline(**kwargs):
        marked.append(current_llm_call_context().prechecked)
        return up.UnifiedSearchResult(documents=[], query=kwargs.get("query", ""), errors=[])

    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    monkeypatch.setattr(rag_ep, "unified_rag_pipeline", _fake_pipeline)
    refuse_llm_work(monkeypatch)
    return asked, marked


@pytest.mark.parametrize(
    ("extra", "own", "looked_up"),
    [
        ({}, {"openai"}, ["openai"]),
        ({"chat_history": [{"role": "user", "content": "Earlier"}], "classifier_provider": "groq"},
         {"openai", "groq"}, ["openai", "groq"]),
    ],
    ids=["answer-only", "reformulated-history"],
)
def test_rag_search_admits_an_over_limit_user_whose_llm_calls_run_on_their_own_keys(
    client_with_agentic_overrides, monkeypatch, extra, own, looked_up
):
    """Past a limit, a search whose answer (and reformulation) run on the user's own keys goes on, unmarked (TASK-13956)."""
    asked, marked = _over_the_limit_with_own_keys(monkeypatch, own)
    resp = client_with_agentic_overrides.post(
        "/api/v1/rag/search",
        json={"query": "q", "enable_generation": True, "generation_provider": "openai", **extra},
    )
    assert resp.status_code != 402, resp.text
    assert marked == [False]
    assert asked == looked_up


@pytest.mark.parametrize(
    "extra",
    [
        {"enable_claims": True},
        {"enable_document_grading": True, "grading_provider": "openai"},
        {"enable_reranking": True, "reranking_strategy": "llm_scoring"},
        {"search_depth_mode": "quality"},
        {"chat_history": [{"role": "user", "content": "Earlier"}], "classifier_provider": "groq"},
    ],
    ids=["claims", "grading", "llm-rerank", "quality-depth", "history-on-a-server-key"],
)
def test_rag_search_still_refuses_an_own_key_answer_with_another_llm_stage(client_with_agentic_overrides, monkeypatch, extra):
    """An LLM stage the boundary cannot place on the user's own key keeps the 402 before any retrieval (TASK-13956)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402

    _asked, marked = _over_the_limit_with_own_keys(monkeypatch, {"openai"})
    resp = client_with_agentic_overrides.post(
        "/api/v1/rag/search",
        json={"query": "q", "enable_generation": True, "generation_provider": "openai", **extra},
    )
    assert_llm_quota_402(resp)
    assert marked == []


@pytest.mark.parametrize(
    ("suggestions", "admitted"), [(None, True), ("true", False)], ids=["no-default", "suggestions-by-default"]
)
def test_rag_batch_names_the_providers_of_the_resolved_request(
    client_with_agentic_overrides, monkeypatch, suggestions, admitted
):
    """A server default that turns on another LLM stage keeps the 402 on a batch, as on search (TASK-13956)."""
    from fastapi import HTTPException

    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402

    asked, _marked = _over_the_limit_with_own_keys(monkeypatch, {"openai"})
    reached: list[bool] = []

    def _past_the_check(*_args, **_kwargs):
        """The batch's next step after the quota check: stop an admitted request before any work."""
        reached.append(True)
        raise HTTPException(status_code=418, detail="the quota check let the request through")

    monkeypatch.setattr(
        rag_ep, "_search_agent_setting", lambda env_key, _key: suggestions if env_key == "SEARCH_SUGGESTIONS" else None
    )
    monkeypatch.setattr(rag_ep, "_resolve_kanban_db_path", _past_the_check)
    resp = client_with_agentic_overrides.post(
        "/api/v1/rag/batch",
        json={"queries": ["q"], "enable_generation": True, "generation_provider": "openai"},
    )

    if admitted:
        assert resp.status_code == 418, resp.text
        assert asked == ["openai"]
    else:
        assert_llm_quota_402(resp)
        assert reached == []
```

(`_search_agent_setting` is patched for both cases, so the server's own `[Search-Agent]` settings cannot decide the `no-default` case; `_resolve_kanban_db_path` is the batch's first call after the check.)

And in `test_rag_stream_parity.py`, after `test_rag_search_stream_refuses_an_over_quota_user_with_a_real_402` (the end of the file):

```python
@pytest.mark.parametrize(("profile", "admitted"), [("accuracy", False), ("fast", True)], ids=["accuracy", "fast"])
def test_rag_search_stream_names_the_providers_of_the_resolved_request(
    monkeypatch: pytest.MonkeyPatch,
    client_with_stream_overrides: TestClient,
    profile: str,
    admitted: bool,
) -> None:
    """A profile that turns on another LLM stage keeps the 402 on the stream, as on search (TASK-13956)."""
    from fastapi import HTTPException

    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    asked: list[str] = []
    marked: list[bool] = []

    async def own_key(_uid: object, provider: str) -> bool:
        """The user has their own OpenAI key (the boundary is under test)."""
        asked.append(provider)
        return provider == "openai"

    def past_the_check(*_args: Any, **_kwargs: Any) -> Any:
        """The stream's next step after the quota check: stop an admitted request before any work."""
        marked.append(current_llm_call_context().prechecked)
        raise HTTPException(status_code=418, detail="the quota check let the request through")

    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", own_key)
    monkeypatch.setattr(rag_ep, "_apply_media_collection_scope", past_the_check)
    refuse_llm_work(monkeypatch)

    resp = client_with_stream_overrides.post(
        "/api/v1/rag/search/stream",
        json={"query": "q", "enable_generation": True, "generation_provider": "openai", "rag_profile": profile},
    )

    if admitted:
        assert resp.status_code == 418, resp.text
        assert marked == [False]
        assert asked == ["openai"]
    else:
        assert_llm_quota_402(resp)
        assert marked == []
        assert asked == []
```

(`accuracy` turns on claims, the `two_tier` reranker, query decomposition and structured responses, `profiles.py`; `fast` turns on no LLM stage. The repo's `config.txt` turns no `[Search-Agent]` LLM stage on, so the stream case needs no patch of `_search_agent_setting`. `_apply_media_collection_scope` is the stream's first call after the check.)

- [ ] **Step 2: Run** the last five ids of `$FOCUSED_B`. Expected (measured with Task 1's code in): `4 failed, 8 passed`: the two search admit cases (`assert 402 != 402`), the batch's `[no-default]` and the stream's `[fast]` (each the 402 instead of the sentinel 418); the five refuse cases, the existing refusal, the batch's `[suggestions-by-default]` and the stream's `[accuracy]` pass today, when every over-limit request is refused (they guard the change).
- [ ] **Step 3: The helpers.** Add `from collections.abc import Mapping` after `import time`. After `_rag_provider_http_exception`, add:

```python
# Request settings whose stage makes an LLM call besides the answer, on a provider the boundary cannot
# name before the pipeline runs (its own field, a config default or "openai"). TASK-13956.
_RAG_OTHER_LLM_STAGES = (
    "enable_query_classification", "enable_research_loop", "enable_query_rewriting_loop",
    "enable_document_grading", "enable_knowledge_strips", "enable_fast_hallucination_check",
    "enable_utility_grading", "enable_claims", "enable_post_verification", "enable_multi_turn_synthesis",
    "enable_query_decomposition", "enable_suggestions", "enable_web_fallback", "enable_image_search",
    "enable_video_search", "enable_structured_response", "enable_faithfulness_eval", "enable_intent_routing",
    "enable_dynamic_granularity", "enable_evidence_accumulation", "enable_evidence_chains", "enable_abstention",
    "adaptive_rerun_on_low_confidence", "adaptive_advanced_rewrites",
)


def _rag_llm_providers(settings: Mapping[str, Any]) -> list[str | None]:
    """The provider of each LLM call a RAG request makes, for ``enforce_llm_quota`` (TASK-13956).

    The answer's provider, as the generator resolves it, and the reformulation's when a chat history
    is reformulated. Any other LLM stage adds None, so an over-limit user is refused as before.
    """
    # ponytail: a stage this list misses is still checked at its own call (refused there, never counted).
    from tldw_Server_API.app.core.config import load_and_log_configs
    from tldw_Server_API.app.core.RAG.rag_service.generation_defaults import resolve_generation_defaults

    named = settings.get("generation_provider")
    providers: list[str | None] = [
        resolve_generation_defaults(load_and_log_configs() or {}, named, settings.get("generation_model"))[0]
    ]
    if settings.get("chat_history") and settings.get("enable_query_reformulation", True):
        providers.append(settings.get("classifier_provider") or named or "openai")
    if (
        str(settings.get("strategy") or "standard") != "standard"
        or settings.get("search_depth_mode") == "quality"
        or (settings.get("enable_reranking", True) and settings.get("reranking_strategy") in {"llm_scoring", "two_tier"})
        or any(settings.get(flag) for flag in _RAG_OTHER_LLM_STAGES)
    ):
        providers.append(None)
    return providers


def _resolved_rag_settings(request: Any, current_user: Any, allowed_fields: set[str] | None = None) -> dict[str, Any]:
    """The request as the pipeline runs it (a ``rag_profile``'s and the server's search defaults applied).

    For the quota check of a route that resolves its request only after the check (TASK-13956).
    """
    return resolve_rag_request(
        request,
        current_user=current_user,
        single_user_id_resolver=DatabasePaths.get_single_user_id,
        search_agent_setting_fn=_search_agent_setting,
        search_agent_allowed_fields=allowed_fields,
    ).payload
```

(The same arguments the stream's `_build_standard_request_bundle` and the batch's `_build_batch_request_bundle` pass, `:372-416`; `_search_agent_setting` is looked up when called.)

- [ ] **Step 4: The sites.** Search (`unified_search_endpoint`):

```python
        if bool(standard_bundle.resolved_request.payload.get("enable_generation", request.enable_generation)):
            await enforce_llm_quota(
                getattr(current_user, "id", None), provider=_rag_llm_providers(standard_bundle.resolved_request.payload)
            )
```

Batch (`unified_batch_endpoint`), replacing the comment and call under `if request.enable_generation:`:

```python
            # Usage quotas (LLM usage accounting spec §4a): one flag turns generation on for every query. The
            # providers come from the resolved request, which carries the server's search defaults (TASK-13956).
            resolved_settings = _resolved_rag_settings(request, current_user, _BATCH_ROUND2_DEFAULT_FIELDS)
            await enforce_llm_quota(getattr(current_user, "id", None), provider=_rag_llm_providers(resolved_settings))
```

Resume: `await enforce_llm_quota(getattr(execution_user, "id", None), provider=_rag_llm_providers(resume_config))` (the checkpoint stores the resolved batch payload). Stream (`unified_search_stream_endpoint`), replacing the comment and call before `request = _apply_media_collection_scope(request, collections_db)`:

```python
    # Usage quotas (LLM usage accounting spec §4a), before the streaming response starts. The providers come
    # from the resolved request, which carries a profile's switches and the server's search defaults (TASK-13956).
    resolved_settings = _resolved_rag_settings(request, current_user)
    await enforce_llm_quota(getattr(current_user, "id", None), provider=_rag_llm_providers(resolved_settings))
```

Advanced (`advanced_search_endpoint`), replacing its comment and call:

```python
        # Usage quotas (LLM usage accounting spec §4a): answer generation is checked before any work. Its only
        # LLM call is the answer, on the default provider (advanced_search turns on no other LLM stage).
        await enforce_llm_quota(getattr(current_user, "id", None), provider=_rag_llm_providers({}))
```

Ablation is unchanged (Decision 3).
- [ ] **Step 5: Run Step 2's command.** Expected: `12 passed`. Then the two whole files: `test_rag_agentic_api.py` `20 passed, 1 skipped`, `test_rag_stream_parity.py` `15 passed` (measured; the parity file's existing 13 still pass).
- [ ] **Step 6: Mutation probes** (revert each after; no request leaves the machine in either, measured with a socket probe as in PR A's Test commands): (i) drop `"enable_claims"` from `_RAG_OTHER_LLM_STAGES` → `test_rag_search_still_refuses_an_own_key_answer_with_another_llm_stage`: `1 failed, 4 passed` (`[claims]`, measured); (ii) set `resolved_settings = request.model_dump()` at the stream and the batch (the raw read of the first draft of this plan, review B I1) → `test_rag_search_stream_names_the_providers_of_the_resolved_request[accuracy]` and `test_rag_batch_names_the_providers_of_the_resolved_request[suggestions-by-default]` fail with `{"detail":"the quota check let the request through"}`: `2 failed, 2 passed` (measured).
- [ ] **Step 7: Commit** (message 2).

## Task 3: Docs, verify, notes, review, PR

- [ ] **Step 1: Rebase on `dev`** (PR A merged): `git fetch origin dev && git rebase origin/dev` (the branch has no merge commit yet). Re-run `$FOCUSED_B`.
- [ ] **Step 2: `Usage_Quotas.md`.** Replace PR A's "**An over-limit user on their own key.**" bullet with:

```markdown
- **An over-limit user on their own key.** An endpoint that names the provider of every LLM call it will make lets an over-limit user through when, for each of those providers, four things hold: BYOK is on, the provider bills per key and its base URL is the vendor's, the provider is on the BYOK allowlist, and the user has an active key for it. These endpoints do:
  - `/chat/completions` and the Messages API (`/messages`, `/v1/messages`), for the request's provider;
  - character chat, for the reply's provider once it is known: after the auto-router, or the saved provider of a roleplay resume. A saved reply in a chat with automatic memory extraction on is still refused, because extraction runs on the server's key;
  - `/evaluations/geval`, `/evaluations/rag`, `/evaluations/response-quality`, and `/evaluations/batch` of those types, for each item's judge provider (`api_name`, OpenAI when unset);
  - document generation from a chat, for the request's provider;
  - RAG search, streaming search, batch, batch resume and advanced search, for the answer's provider and, with a chat history to reformulate, the reformulation's (`classifier_provider`, else `generation_provider`, else OpenAI). A request that also turns on another LLM stage, itself or through a `rag_profile` or the server's `[Search-Agent]` defaults, is still refused: query classification, grading, claims, an LLM reranker (`llm_scoring`, `two_tier`), the `quality` search depth, the agentic strategy and the rest of `_RAG_OTHER_LLM_STAGES` in `rag_unified.py`. So is RAG ablation.

  Such a request is not marked admitted, so each of its calls is checked where its key is known. A call on the user's own key runs. A call on any other key, a fallback provider's for example, is refused at the call. On `/chat/completions` the hosted billing plan's check is skipped for such a request. Every other boundary endpoint refuses an over-limit user before the key is known, even when the call would have used the user's own key: benchmarks and evaluation runs, flashcards and quizzes, watchlists and media analysis. So does a media ingest job's check when it starts.
```

  Not mirrored; `bash Helper_Scripts/refresh_docs_published.sh` leaves `Docs/Published` unchanged. Commit (message 3).
- [ ] **Step 3: Verify:** `$FOCUSED_B` `30 passed`; the wider set against Task 0's baseline (the baseline plus 23 new ids; measured `4 failed, 1505 passed, 161 skipped, 1 error` and PostgreSQL `70 passed`, the failures and the error named under Test commands); `tests/Docs tests/lint tests/CI tools/backlog-py/tests` (`1661 passed, 6 skipped`, measured on `dev` + A + B once `lint/test_rg_route_map_lint.py` is re-run alone: in the full run its subprocess aborted under load, see the Folded section); shard coverage (`new_uncovered=0`); triage `--enforce` ("OK — no offenses beyond the baseline."); route-auth ("208 unauthenticated routes; baseline exact."); OpenAPI (both "OK" lines); the scope-baseline grep for `usage_quota_deps|evaluations_unified|chat_documents|rag_unified|test_llm_quota_decision|test_evaluations_stage1|test_document_generation_endpoints|test_rag_agentic_api|test_rag_stream_parity` (nothing, measured); `git diff origin/dev --check` (nothing). The commands are PR A's Task 3 Step 2.
- [ ] **Step 4: Bandit:** `/opt/homebrew/bin/uvx bandit -ll tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py tldw_Server_API/app/api/v1/endpoints/chat_documents.py tldw_Server_API/app/api/v1/endpoints/rag_unified.py`. Expected: "No issues identified.", Medium 0, High 0 (measured on `dev` + A + B).
- [ ] **Step 5: Notes** (backlog-py; fill every `<...>`):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13956 \
  --append-notes "PR B (#<n>): enforce_llm_quota takes several providers (admitted past a limit only when every one runs on the user's own key; a missing name refuses). Named: the judge-model evaluations (each batch item's), chat document generation, RAG search/stream/batch/resume/advanced (the answer's provider and, with a chat history, the reformulation's, read from the resolved request so a rag_profile or a server Search-Agent default counts; any other LLM stage in _RAG_OTHER_LLM_STAGES, the quality depth, a non-standard strategy or an LLM reranker counts; ablation unchanged). own_key= answers for a single provider only. Evaluation token reservations unchanged (owner, 2026-10-29). RED/GREEN: focused <red> -> <green>. Wider <base> -> <after>. Mutations: claims dropped from the list -> [claims] not 402; raw request at stream and batch -> [accuracy] and [suggestions-by-default] admitted; own_key taken for a list -> admitted. Bandit: <result>."
git add -A backlog/tasks && git diff --cached --check
```

  Commit (message 4).
- [ ] **Step 6: Review, PR body, Change summary, checks, merge:** as PR A's Task 3 Steps 5-8 and 10, with this plan's Review Focus. The non-required E2E diff (PR A's Step 9, against recent PR runs of "Frontend E2E Tiers") also runs here: the chat pages share `/chat/completions`, and the diff is cheap.

## Commit messages

Each ends with a blank line and `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

- 0: `docs(plan): plan own-key evaluations, chat documents and RAG (TASK-13956, 2 of 3)`
- 1: `fix(usage): the boundary names several providers; evaluations and chat documents name theirs (TASK-13956)`
- 2: `fix(usage): RAG lets an over-limit user through when its answer runs on their own key (TASK-13956)`
- 3: `docs(usage): own-key evaluations, chat documents and RAG (TASK-13956)`
- 4: `chore(backlog): TASK-13956 notes for PR B`

## Not measured

- RAG end to end with a real pipeline over seeded documents on the user's own key: the admit tests stub `unified_rag_pipeline`; the generator's handle use is covered by the existing RAG credential tests (`test_rag_convenience_endpoint_credentials.py`).
- Which `_RAG_OTHER_LLM_STAGES` entries make no LLM call in practice (listing one only refuses more).
- A pipeline-internal config default that turns on an LLM stage without the request or its resolution naming it (`RAG_*` settings read inside the pipeline): checked at that stage's call, not measured. A `rag_profile` and the `[Search-Agent]` defaults are measured (V4b).
- Admit tests for batch resume and advanced search (they name their providers as search does; the plan reviewer probed both admitted unmarked with an own OpenAI key, review B M7).

## Owner answers (2026-11-05, recorded by the controller)

- Plans A, B and C: "Approve (Recommended)", with both reviews' Important fixes and minors folded in (review_plan_13956_a.md, review_plan_13956_bc.md).
- AC #2 (and the matching part of AC #1): "Amend + follow-up (Recommended)": PR C amends the ACs with backlog-py to what holds, pins the server-key boundaries, and files the follow-up asking whether those features should use the stored key (the study assistants as the cheap first slice; media analysis about a day).
- Decision 1, ADR-004 Change summary: the waiver covers all three TASK-13956 PRs ("Yes, covers all three").

## Folded after the plan reviews (2026-11-05)

From `review_plan_13956_bc.md` (B: 1 Important, 6 Minor) and the owner's answers. Re-measured on `origin/dev` `484e393333` with #3422's `Usage_Quotas.md` and `proto_13956_a.patch` (the tree `dev` will be after #3422 and PR A); `proto_13956_b.patch` was rebuilt from this plan's text on that tree and matches it byte for byte (`cmp`).

| Finding | What changed |
|---|---|
| I1: streaming and batch RAG named providers from the raw request | New `_resolved_rag_settings(request, current_user, allowed_fields=None)` (`resolve_rag_request(...).payload` with the bundles' arguments); the stream names its providers from it and the batch from it with `_BATCH_ROUND2_DEFAULT_FIELDS`, so a `rag_profile` or a `[Search-Agent]` default that turns an LLM stage on counts. New tests: `test_rag_search_stream_names_the_providers_of_the_resolved_request` (`test_rag_stream_parity.py`; `[accuracy]` 402, `[fast]` admitted unmarked) and `test_rag_batch_names_the_providers_of_the_resolved_request` (`[suggestions-by-default]` 402, `[no-default]` admitted). RED against the unfixed code: on the first prototype `[accuracy]` and `[suggestions-by-default]` were admitted (the 418 sentinel), and Task 2 Step 6 mutation (ii) reproduces that. `test_rag_stream_parity.py` `15 passed` (its 13 existing tests and the two new). V4b, Decision 3, Review Focus 3, Architecture; the "raw request" reviewer note is gone. |
| M1: wider numbers labelled as base but taken on A's file list | The wider set states its list per tree (100 files on `dev`, 102 with A) and expects Task 0's baseline plus this PR's new ids; the numbers are re-measured on the 102-file list on `dev` + A. |
| M2: a fourth pre-existing load failure | Listed beside the rbac ids (`test_rag_search_natural_language_media_fallback`, which passed in these runs), with the two load failures seen here (the abtest id and a `DatabaseLockError` setup error in `test_rag_health_endpoints.py`; both files `15 passed` alone). |
| M3: "none PostgreSQL" was wrong | The two `postgres` files are the PostgreSQL part, run serially through the repo fixture (`TEST_DB_NAME=tldw_test_13956f`): `70 passed` on base and prototype. |
| M6: `own_key=True` with a list admitted without asking | `own_key` is ignored for a collection (two lines and the docstring); `test_own_key_answers_for_a_single_provider_only` is RED today (`DID NOT RAISE`) and asserts the 402's status beside its collector so the triage passes (it flagged `tautology_suspect` without it, measured). Decision 2. |
| M7: only `/rag/search` had an admit test | The stream and the batch now have admit cases; resume and advanced are under Not measured (probed by the reviewer). |
| M9: the docs said a RAG request asking for another stage is refused, untrue for streaming | Fixed with I1; the bullet now says "turns on another LLM stage, itself or through a `rag_profile` or the server's `[Search-Agent]` defaults". |
| Owner answer to Decision 1 | Recorded. |
| Interfaces with A | Task 3 Step 2 replaces PR A's own-key bullet by its literal text, which PR A's fold leaves unchanged; PR A's new `usage_quota_deps` function sits at the file's end, outside this PR's hunks. |

Re-measured (each against the same base code first):
- Task 1 (18 tests, was 17): RED `11 failed, 7 passed`; GREEN `18 passed`; ratchet `13 passed`.
- Task 2 (12 tests, was 8): RED with Task 1's code `4 failed, 8 passed`; GREEN `12 passed`; whole files `test_rag_agentic_api.py` `20 passed, 1 skipped`, `test_rag_stream_parity.py` `15 passed`.
- Focused set (14 ids, 30 tests, was 11 and 25): base `15 failed, 15 passed`; prototype `30 passed`; no outbound connection in the set or in any Step 6 mutation (socket probe).
- Wider set (102 files on `dev` + A: 100 SQLite, 2 PostgreSQL): SQLite base `3 failed, 1484 passed, 161 skipped` (3:41), prototype `4 failed, 1505 passed, 161 skipped, 1 error` (3:40), all failures and the error pre-existing or load-dependent (Test commands); PostgreSQL `70 passed` on both.
- `tests/Docs tests/lint tests/CI tools/backlog-py/tests` on `dev` + A + B: `1 failed, 1660 passed, 6 skipped` (5:52), the failure `lint/test_rg_route_map_lint.py::test_shipped_route_map_is_clean`, whose subprocess aborted with `libc++abi: ... recursive_mutex lock failed` (exit -6) under the load of three parallel runs; alone the file is `7 passed` and `Helper_Scripts/ci/rg_route_map_lint.py` exits 0 twice, and the same suite on `dev` + A + B + C is `1661 passed, 6 skipped`. `tests/Docs` `212 passed`. Shard coverage `new_uncovered=0`; triage `--enforce` OK; route-auth "208 unauthenticated routes; baseline exact."; OpenAPI both OK; function-size ratchet `13 passed` (`batch_evaluate` 547, `generate_document` 466, unchanged); Bandit: no issues; ruff: no new finding in the nine touched files; `git diff --check` clean.
- Open-PR overlap: V8 (re-checked against current heads).
