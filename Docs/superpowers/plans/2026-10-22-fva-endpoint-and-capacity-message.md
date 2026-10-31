# POST /claims/verify/fva Runs, for the Caller or for an Admin's Named User, and Challenges Claims; a Full Bounded Pool Names Its Call (TASK-13881, TASK-13888)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Re-review folded in (r3, 2026-10-22):** the scoped re-review and dry run of r2, `<scratchpad>/rereview_plan_fu2_claims.md`. Verdict: ready with fixes, with 0 Blocker, 1 Important and 3 Minor, all folded in. r2 is kept as `<scratchpad>/plan_fu2_claims.r2.md`.
- **I1 (the cost statement was wrong):**
  - **The corrected cost.** Decision 1 and Review Focus 9 now give the per-request worst case: up to 50 claims × (1 + `top_k` + 15) LLM calls, so 1,300 at the default `top_k` and 5,800 at the schema maximum.
  - **The quota is not the hard bound; the request schema is.** Usage quotas are off by default. With them on, the per-call check is fail-open, and bounds the overshoot only to one request's in-flight calls.
  - **A cap on the supporting side, prototyped and measured.** Adjudication assesses only the first 5 base documents, the verifier's window: `fva_pipeline.py` +6/−1, one test pinning it. The worst case becomes 50 × 21 = 1,050 calls per request at any `top_k`. The verdict rules are unchanged; only which supporting documents are averaged changes (Review Focus 11).
  - **Decision 1 now has three options:** A-capped (recommended, and implemented here as TASK-13881's new AC #6), A-uncapped, and B. Each comes with its measured worst case, and the plan lists the edits for the other two.
- **M1:** the doc sentence and the code comment say that the FTS "floor" is relative: scores are min-max normalised per query, so the best match always passes.
- **M2:** Review Focus 10 now states the signal risk: one document wrongly judged CONTRADICTS among many unfloored neighbours can produce REFUTED.
- **M3:** an `adjudicate` test pins that NEUTRAL counter-evidence does not count. It passes on dev, as a pin; mutant N1 shows it bites.
- The changed tests were re-run on the r3 prototype; the counts are under "What was measured". Logs are prefixed `fu2c_r3_`.

**Review folded in (r2, 2026-10-22):** the independent review and dry run, `<scratchpad>/review_plan_fu2_claims.md`. Verdict: ready with fixes, with 0 Blocker, 2 Important and 6 Minor, all folded in. r1 is kept as `<scratchpad>/plan_fu2_claims.r1.md`.
- **I1 (id lock):** Task 4 Step 1 takes `<scratchpad>/id_lock` before computing the next id. Step 7 releases it once `gh pr view --json files` lists the new task file.
- **I2 (Decision 1):** the anti-context fix (r1's follow-up F1) was prototyped and measured.
  - It is small: the post-fusion filter is deleted (a 6+/3− change in `anti_context_retriever.py`), and the FTS floor already applies before fusion.
  - The one design choice, what floors the vector list, is settled and stated: nothing does; the adjudicator's stance check decides.
  - So Decision 1 now recommends folding it in (option A). It is this plan's new Task 1 and TASK-13881's new AC #5, and it is no longer a follow-up.
  - The endpoint tests lose the floor lift, and gain vector-store variants.
  - Option B is described, with its corrected costs and the exact edits it needs.
- **M1:** the close-out blocks write the new task id literally, and set `PY` themselves.
- **M2:** the conflicts table has the `plan_fu2_workflows.md` row.
- **M3:** one real multi-user API-key test, on both backends.
- **M4:** TASK-13888's AC #2 is kept and checked (vacuously met), with the finding in the notes.
- **M5:** one test sends a body without `max_budget_usd`.
- **M6:** Task 3's sweep writes `fu2c_13888_tests.r2.txt`, not the planner's evidence file.

The re-measurements below were made on the same `origin/dev` `e06eeec68c` (dev had not moved under this plan's files when r2 was written). Logs are in `<scratchpad>`, prefixed `fu2c_r2_` and `fu2c_f1_`.

**Goal:**
- `POST /claims/verify/fva` stops answering 500 to every request. It runs the FVA pipeline for one user, finds counter-evidence and adjudicates it against the verifier's top 5 supporting documents, and its LLM quota refusals answer 402.
- Its `user_id` query parameter counts only for a platform admin, as `rebuild_claims` does. The media DB, the vector store, the quota decision and the usage rows then all belong to the named user. Otherwise all four belong to the authenticated user, by id.
- A full bounded pool reached through `await_bounded_daemon_with_timeout` raises `DaemonCapacityError("<name> capacity is exhausted")`, not the streaming default.

**Architecture:**
- **TASK-13881, counter-evidence and its cost (Task 1).** `AntiContextRetriever` asks for hybrid retrieval with `min_score=0.3`. The Media DB applies that to FTS scores min-max normalised within each query's results, before fusion (`database_retrievers.py:1907-1910`). So it trims each query's weakest full-text matches, and the best one always passes.
  - It then filtered the fused list again with the same 0.3. A fused (RRF) score is at most 1/61, so that second filter dropped every document, with or without a vector store.
  - The second filter is deleted.
  - No absolute relevance floor applies to either half: the adjudicator's stance check decides, and only documents it judges CONTRADICTS count (`adjudicator.py:177-183`).
  - **The cap.** Each supporting document costs one stance call per falsified claim. So `FVAPipeline.process_claim` hands the adjudicator only `documents[:5]`, the window `HybridClaimVerifier.verify` judged (its `top_k` is 5), not every base document the request retrieved.
- **TASK-13881, the user (Task 2).** `verify_claims_with_fva` resolves the owner first, then runs the body for that owner.
  - Its body moves unchanged into a private `_verify_claims_with_fva_for(owner_user_id, db, ...)`, apart from the three constructor fixes.
  - **For the caller,** nothing is switched. The request's content scope is already the caller's: `get_request_user` sets it.
  - **For an admin naming another user,** the call runs inside two context managers:
    - `scoped_context(user_id=<named>, is_admin=False)`;
    - the module's own `_claims_user_override_db(<named>)`, which `rebuild_all_media` and `rebuild_claims_fts` already use.
  - **One switch covers three of the four.** The content scope is what:
    - the media visibility filter reads (SQLite: `append_sqlite_media_visibility`; PostgreSQL: the media RLS policy, through `app.current_user_id` and `app.is_admin`);
    - the quota check reads (`current_llm_user_id()` falls back to the scope);
    - the chat recorder reads.
  - **The fourth is set explicitly.** The `user_id` given to `MultiDatabaseRetriever` picks the vector store. The `ClaimsJobContext(user_id=...)` writes the `claims_verify` usage rows by id.
- **TASK-13888 (Task 3):** `await_bounded_daemon_with_timeout` passes `exhaustion_message=f"{name} capacity is exhausted"` to `pool.start`. That is one line in shared code. One test pool that overrides `start` without accepting the argument is updated.

**Tech Stack:** Python 3.12, FastAPI, httpx `ASGITransport`, pytest (`asyncio_mode = "auto"`), SQLite and PostgreSQL media DBs, the AuthNZ SQLite DB for one real-auth test.

**Spec:**
- `backlog/tasks/task-13881 - POST-claims-verify-fva-answers-500-to-every-request.md`:
  - AC #1: the endpoint runs the FVA pipeline on the media DB and vector store of the user AC #3 resolves, with an endpoint test that checks a result;
  - AC #2: a quota refusal at its verify or adjudication call answers 402, with a test;
  - AC #3: `user_id` counts only for a platform admin, as `rebuild_claims` does (`_legacy_user_has_platform_admin_claims`). With an admin's `user_id`, the media DB, the vector store, the quota decision and the usage rows all belong to that user; otherwise all of them belong to the authenticated user, by id, not username. A test for each case;
  - AC #4: `Usage_Quotas.md` lists `POST /claims/verify/fva` among the endpoints that answer 402;
  - **AC #5, added in Task 0 (Decision 1, A-capped or A-uncapped):** "FVA's anti-context retrieval keeps the counter-evidence hybrid retrieval returns (its min_score trims each query's weakest full-text matches before fusion), so the adjudicator runs, with a test on a real MultiDatabaseRetriever with and without a vector store";
  - **AC #6, added in Task 0 (Decision 1, A-capped):** "Adjudication assesses at most the verifier's 5 base documents on the supporting side, with a test";
  - notes: the owner's decision of 2026-10-20 (repair the endpoint, admin-gate `user_id` as `rebuild_claims`), and the PR #3287 review's I1 reword.
- `backlog/tasks/task-13888 - A-full-sync-adapter-pool-logs-Streaming-worker-capacity-is-exhausted-for-non-streaming-calls.md`:
  - AC #1: a full pool reached through `await_bounded_daemon_with_timeout` raises `DaemonCapacityError` with a message naming the call, with a test;
  - AC #2: callers and tests that match on the old text are updated.
- Filed by PR #3287 (TASK-13862, merged). Its clean review is `<scratchpad>/clean3287_review.md` (I1, and Minor 3 for TASK-13888). Its plan is `Docs/superpowers/plans/2026-10-20-fva-adjudicator-off-loop.md`.
- Governing design: `Docs/Design/2026-10-09-llm-usage-accounting-design.md` §1 (who a call is for: an explicit context, else the content scope).

`<scratchpad>` is `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad`.

## What was measured

Measured on `origin/dev` `e06eeec68c` (2026-10-22), in an isolated worktree. The prototype is this plan applied there (`fu2c_prototype_r2.diff`). The probes, mutation scripts and logs are in `<scratchpad>`, prefixed `fu2c_`.

### TASK-13881

- **The endpoint** is `endpoints/claims.py:996-1040`. It takes `user_id: Optional[int]` (`:999`) and builds `resolved_user_id` (`:1024-1029`): the parameter as a string if set, else the **username**. It has one app caller path, `claims_service.verify_claims_with_fva` (`claims_service.py:4686`). `git grep` finds no other caller, and no test calls either.
- **The 500, with the reason** (probe `fu2c_probe_500.py`, a direct service call):
  - with `fva_config.max_budget_usd`: `TypeError: ClaimsJobContext.__init__() got an unexpected keyword argument 'job_id'`. The call is at `:4761-4764`; the dataclass at `budget_guard.py:45-50` has `user_id`, `api_key_id`, `request_id` and `endpoint`;
  - without it: `TypeError: MultiDatabaseRetriever.__init__() missing 1 required positional argument: 'db_paths'` (`:4772`; the signature is at `database_retrievers.py:4745-4755`);
  - `RetrievalConfig(top_k=..., search_mode="hybrid")` (`:4784-4787`) names two fields the dataclass (`database_retrievers.py:476-495`) does not have. Its fields are `max_results`, `use_fts`, `use_vector`, and so on. `use_fts` and `use_vector` default to true, which `MultiDatabaseRetriever.retrieve` turns into `retrieve_hybrid` (`:4931-4946`).
  - So nothing is exposed on dev: every request fails at a constructor before it reads any data or makes any LLM call (the review confirmed this).
- **What picks each of the four things AC #3 names:**
  - **Media DB.** The request's DB comes from `get_media_db_for_user`. On SQLite each user has a file. On PostgreSQL, every user's rows share one set of tables.
    - Reads are filtered by the content scope: on SQLite by `append_sqlite_media_visibility` (`media_search_repository.py:25-66`, applied at `:276-277`), on PostgreSQL by the media RLS policy (`postgres_rls.py:37-80`, `FORCE ROW LEVEL SECURITY` at `:135`).
    - An admin scope (`is_admin=True`) skips both. So under the admin's own scope, the named user's file would show all its rows on SQLite, and on PostgreSQL every account's rows.
  - **Vector store.** `MediaDBRetriever` builds it with `create_from_settings_for_user(settings, self.user_id)` (`database_retrievers.py:997`), and searches `user_{self.user_id}_media_embeddings` (`:2203`).
  - **Quota decision.** `current_llm_user_id()` returns the explicit `llm_call_context` user, else the scope's (`llm_call_context.py:44-50`). No HTTP path sets an explicit user (`git grep bind_llm_call_user`, `llm_call_context(user_id`: only WebSockets and tests). The claims router has no boundary check, so every LLM call is checked at the call.
  - **Usage rows.**
    - The verifier's call runs `records_itself=True` (`claims_engine.py:1206`). Its `claims_verify` row is written by `_log_claims_llm_usage` for `job_context.user_id`, else `current_llm_user_id()` (`:676-681`).
    - The adjudicator's calls are recorded by the provider base class for `current_llm_user_id()` (`usage_accounting.py:46-58`).
    - On dev, the job context gets the parameter string or the username.
- **The admin gate's sibling:** `rebuild_claims` (`claims_service.py:4575-4604`) uses `user_id` only when `_legacy_user_has_platform_admin_claims(current_user)` (`:282-289`, roles `admin`/`owner`/`super_admin` or permissions `*`/`system.configure`/`admin`), else `current_user.id`. It ignores a non-admin's `user_id` without an error. `rebuild_all_media` (`:4660-4662`) and `rebuild_claims_fts` (`:4673-4675`) open the named user's DB with `_claims_user_override_db` (`:1819-1828`).
  - On PostgreSQL that helper opens the shared content backend: `create_media_database` uses `runtime.backend_loader()` in PostgreSQL content mode (`media_db/runtime/factory.py:41-82`).
  - `managed_media_db_for_owner` in `API_Deps/DB_Deps.py` is not usable here. `tests/lint/test_core_to_api_import_boundary.py` freezes the core modules that import `api/v1/API_Deps`, and `claims_service.py` is not among them.
- **Single-user mode:** `get_single_user_instance()` has `role="admin"` and `roles=["admin"]` (`User_DB_Handling.py:283-309`). So, as for `rebuild_claims`, the single user's `user_id` is honoured.
- **In the app, FVA never finds counter-evidence, so the adjudicator and its LLM calls never run.**
  - `AntiContextRetriever` asks for hybrid retrieval with `min_score=0.3` (`anti_context_retriever.py:214-220`). The Media DB applies that floor to min-max-normalised FTS scores before fusion (`database_retrievers.py:1907-1910`).
  - Then it drops every fused document scored below `min_relevance_score = 0.3` (`:55`, `:187`).
  - Fused scores are reciprocal-rank-fusion scores, `1 / (60 + rank + 1)` per list (`database_retrievers.py:2652`, `:2658`, `:2679`), so the best possible is 1/61 ≈ 0.016. This holds without a vector store too: `_retrieve_vector` falls back to FTS, and the two lists are still fused.
  - `supporting_evidence` and `contradicting_evidence` are filled only from adjudication (`claims_service.py`, the result loop), so with the endpoint repaired but this unfixed, the response carries no evidence at all.
  - **The cost of leaving it:** every claim that triggers falsification still runs 3 anti-context hybrid retrievals, and throws their results away. With a vector store configured, those include 3 query embeddings. Which claims trigger: a type in `force_claim_types`, verification confidence below 0.7, or sparse evidence (`should_trigger_falsification`).
  - **One caller:** `AntiContextRetriever` and `FVAPipeline` have one app caller each, this endpoint's path (`git grep`). Nothing else in the app uses the class (the `min_relevance_score` names in `rag_schemas_unified.py` and `knowledge_strips.py` are unrelated).
- **The anti-context fix, prototyped (Task 1).**
  - **The change:** delete the post-fusion filter, and say why in a comment. `anti_context_retriever.py`: 6 lines added, 3 removed.
  - **RED on `e06eeec68c`** (`fu2c_f1_red.log`): `3 failed, 22 passed` in `test_anti_context_retriever.py`:
    - the replaced `test_filters_by_min_score`, `assert set() == {'1', '2'}`;
    - the new real-retriever test, without a vector store: `assert set() == {'1'}`;
    - the same test with a vector store: `assert set() == {'1', '2'}`.
  - **GREEN:** `25 passed`.
  - **End to end,** with the endpoint repaired (Task 2) but without Task 1 (`fu2c_r2_nof1.log`): the endpoint file gives `21 failed, 1 passed`. Twenty fail `anti_context_found > 0`, with and without a vector store. The 402-at-adjudication case gets a 200. Only the 402-at-verify case passes.
  - **The FTS "floor" is relative** (the re-review's M1). `_build_media_documents` min-max normalises each query's FTS scores (`database_retrievers.py:1903-1910`, `utils.py:176-181`). The best match scores 1.0, and a single or tied result 0.5, so both pass 0.3 whatever their absolute relevance. So neither half of the hybrid list has an absolute relevance floor.
- **What one request can cost** (the re-review's I1, re-checked here).
  - **Per claim:** 1 verify call (`HybridClaimVerifier.verify` judges the top 3 snippets of `base_documents[:5]` in one call, `claims_engine.py:1043`, `:1058`, `:1207`). Then, for a falsified claim, adjudication makes 1 stance call per document assessed (`_assess_documents` loops, `adjudicator.py:205-239`):
    - **the supporting side:** every base document, up to `top_k` (`fva_pipeline.py:271-276`; the fusion caps the list at `top_k`, `database_retrievers.py:2677`);
    - **the counter side:** at most 3 queries × 5 documents = 15, fewer after deduplication.
  - **Per request:** `FVAVerifyRequest` allows 50 claims and `top_k` up to 100 (`claims_schemas.py:793`, `:798`). So with Task 1 and no cap, the worst case is 50 × (1 + `top_k` + 15) LLM calls:
    - 1,300 at the default `top_k` of 10;
    - 5,800 at 100.
  - **The supporting calls repeat:** they assess the same base documents again for every falsified claim of the request.
  - **The quota is not the hard bound.**
    - Usage quotas are off by default (`usage_quotas_enabled()`, `config.py:2896-2905`).
    - With them on, every call is checked (`chat_service.py:2637`, for the request's user), but the check asks for 1 token and is fail-open (`llm_quota_gate.py:88-125`). Calls already in flight finish: up to `max_concurrent_falsifications` claims at once, default 5, at most 20.
    - So the request schema is the hard bound.
    - `falsification_timeout_seconds` bounds only the anti-context retrieval (`fva_pipeline.py:251-258`), not adjudication. `max_budget_usd` only gates entry to falsification (`fva_pipeline.py:452-466`).
- **The cap, prototyped (Task 1).**
  - **The change:** `fva_pipeline.py` +6/−1. A module constant, `_ADJUDICATED_SUPPORTING_DOCS = 5`, with a comment saying why, and `supporting_docs=documents[:_ADJUDICATED_SUPPORTING_DOCS]`.
  - **Why 5:** `process_claim` calls `verifier.verify(...)` without `top_k`, so the verifier uses its default 5. `documents` is fused-score order, so `documents[:5]` is exactly the window the verifier drew its evidence from.
  - **Worst case with the cap:** 1 + min(`top_k`, 5) + 15 ≤ 21 LLM calls per falsified claim, so 50 × 21 = **1,050 per request, at any `top_k` ≥ 5** (950 at `top_k` 3).
  - **RED** (`fu2c_r3_red.log`, the cap test on the r2 prototype, whose `fva_pipeline.py` is dev's): `test_adjudication_assesses_only_the_verifiers_top_five_supporting_documents` fails with `Left contains 3 more items, first extra item: 'Support 5'`, because all 8 base documents were assessed. **GREEN:** passes.
  - **The verdict logic holds.**
    - `_determine_status` is unchanged.
    - `support_score` is `_aggregate_score` over the supporting-side documents judged SUPPORTS. That is a confidence-weighted mean, not a sum, so assessing fewer documents does not lower it systematically.
    - What changes is which documents are averaged: base documents ranked 6 to `top_k`, which the verifier never saw, no longer count.
    - So in edge cases the verdict can move either way. If only a document ranked 6 or lower SUPPORTS, support drops to 0, and a strong contradiction (> 0.7) can then give REFUTED. If low-confidence supports ranked 6 or lower dragged the mean down, support rises.
    - `supporting_evidence` in the response has at most 5 entries.
    - The contradict side is unchanged.
  - **The NEUTRAL pin** (the re-review's M3): `test_neutral_counter_evidence_does_not_count` (one CONTRADICTS 0.9 and three NEUTRAL counter-documents; `contradict_score == 0.9`, one entry). It passes on dev source (`adjudicator.py` is unchanged by this plan), so it is a pin, not a RED. Mutant N1, which lets NEUTRAL documents into the mean, fails it.
- **An admin's `user_id` for an account with no media DB** (probe `fu2c_probe_missing_user.py`): `_claims_user_override_db` creates `<USER_DB_BASE_DIR>/<id>/Media_DB_v2.db`, and the search returns 0 rows. So the request answers 200 with no evidence, as `rebuild_claims_fts` does today for the same input.
- **RED of the endpoint tests** (Task 2 Step 2, with Task 1 applied and dev's `claims_service.py` and `claims.py`, `fu2c_r2_red_t2.log`): `22 failed`, all `500`: 20 `assert 500 == 200` and 2 `assert 500 == 402`. The PostgreSQL params ran against `tldw_postgres_test`.
- **GREEN:** `22 passed` (14 SQLite, 8 PostgreSQL; the real-auth cases included), about 11 s here and 24-45 s in the re-review. The cap does not touch these tests: their requests use `top_k` 3.
- **An intermittent abort at interpreter exit.**
  - **What happens:** the process ends with exit code `-6` and `libc++abi: terminating due to uncaught exception of type std::__1::system_error: recursive_mutex lock failed: Invalid argument`, after `22 passed` was already printed.
  - **How often:** 3 of 9 plain runs of the whole endpoint file here. It never happened in 0 of 4 runs with `PYTHONFAULTHANDLER=1`, or in 0 of 3 runs of either half of the file. The re-review saw it in 0 of 15 r2 runs.
  - **Already known:** it is a pre-existing macOS issue, after unrelated test files. The re-review found the same message in 6 earlier repo documents, among them TASK-13291, TASK-13850 and TASK-13851. On a `-6` after a full pass, rerun once and record both runs.
- **Mutants** (`fu2c_mutate_r3.py`, against the endpoint, anti-context, pipeline and adjudicator test files, 113 tests): every mutant fails at least one test. The re-review re-ran r2's nine and got the same counts.

  | Mutant | Tests failed |
  |---|---|
  | G1: `user_id` honoured for anyone | 6: the member's `?user_id=2` case (4 backend/store variants) and the real-auth member case (2 backends) |
  | G2: `user_id` ignored for everyone | 7: the admin case (4 variants), the real-auth admin case (2) and single-user `?user_id=2` |
  | S1: no scope switch | 7, the same |
  | S2: the scope switch keeps `is_admin=True` | 7, the same: the stranger's row (user 5) shows up in the evidence |
  | S3: no DB switch | 3: the SQLite admin case, the SQLite real-auth admin case, single-user `?user_id=2`. On PostgreSQL the DB is shared, and with a vector store the fake store returns the named user's rows, so the scope and the store decide, as designed |
  | V1: the vector store keyed by another user | 19 |
  | U1: usage rows by username | 20 |
  | F1: the post-fusion floor put back | 24 |
  | F2: no FTS floor before fusion (`min_score=0.0`) | 1: the contract test on the retrieval config |
  | C1: no cap on the supporting side (r3) | 1: the cap test |
  | N1: NEUTRAL counter-evidence counted (r3; `adjudicator.py`, which this plan does not change) | 1: the NEUTRAL pin |

- **Neighbourhood** (r3 prototype):
  - The four FVA test files (endpoint, anti-context, pipeline, adjudicator): `113 passed`.
  - `tests/Claims` + `tests/Claims_Extraction`: `1307 passed, 1 skipped, 1 failed` (97 s). The failure is the known Hypothesis `too_slow` test, `test_review_metrics_payload_inclusive_window_property`, which passed alone, so 1,308 pass when it does (r2: 1,306, plus the two r3 tests). The skip is `test_ingestion_claims_ner.py:21` (spaCy not available).
  - The two lint tests (r1): `8 passed`.
  - The test-quality ratchet: exit 0, "OK — no offenses beyond the baseline".
  - `tests/CI/test_test_quality_ratchet.py` + `tests/Docs`, with a plan copy and the doc edits applied: `213 passed`.
  - ruff:
    - `claims_service.py` I001, which dev's file also has;
    - `test_anti_context_retriever.py` I001 twice and F401 `MagicMock`, which dev's file also has (checked on an exported copy);
    - `test_adjudicator.py` I001 twice and F401 `typing.Optional`, which dev's file also has;
    - nothing in the new endpoint test file, `fva_pipeline.py` or `test_fva_pipeline.py`.
  - `uvx bandit -q -r` on the five touched source files: exit 0, no issues. The 19 `nosec encountered` warnings come from dev's existing comments.
  - `pg_roles` held no `fva_endpoint_%` role afterwards.
- **The review's extra probes** (its "Dry-run results" §6): real multi-user API-key auth through the real `get_request_user` (now a test in this plan, M3), and 8 interleaved concurrent requests on both backends. Neither found any crossing.

### TASK-13888

- **The default:** `BoundedDaemonPool.start(..., exhaustion_message="Streaming worker capacity is exhausted")` (`bounded_daemon.py:135`). `await_bounded_daemon_with_timeout` calls `pool.start(invoke, name=name, released_event=released)` (`:304`) without the argument.
- **Its 8 app callers**, with the `name=` each passes:
  - `audio_streaming.py:816`: `audio-stream-iterator`, `-next`, `-factory`, on `STREAM_DAEMON_POOL`;
  - `character_chat_sessions.py:555`: `name=name`;
  - `byok_testing.py:305`: `admin-provider-health-{provider}`;
  - `streaming_utils.py:896`: `owned-stream-close`;
  - `claims_engine.py:114`: `claims-provider-analyze`;
  - `rag_evaluator.py:106`: `rag-evaluation-analyze`;
  - `advanced_reranking.py:1728`: `rag-llm-reranker`;
  - `tts_jobs_worker.py:423`: `tts-job-rbac-revalidation`.

  None reads the error's text. The audio WebSocket sends a sanitized payload, "without exposing upstream text" (`audio_streaming.py:2589-2601`). The evaluation and document endpoints map `DaemonCapacityError` to a fixed 503 detail. TTS logs the type only (the review).
- **Matches on the old text:** `git grep "Streaming worker capacity is exhausted"` finds only `bounded_daemon.py:135` and the task file. No caller or test matches it, so AC #2 is met without a change (M4).
- **What does break:** test pools that override `start(self, target, *, name, released_event=None)` without `**kwargs`.
  - There are 13 such overrides, in `test_async_stream_task_capacity.py`, `test_chat_service_fallback.py` (4), `test_request_queue.py` (4), `test_streaming_utils.py` (2), `test_llm_streaming_and_security.py` and `test_mlx_provider.py`.
  - With the change, the six files: `1 failed, 571 passed, 1 skipped`. The failure is `test_async_stream_task_capacity.py::test_async_daemon_bridge_consumes_failure_when_cancelled_during_release_wait`, the only override reached through `await_bounded_daemon_with_timeout` (`:1042`, called at `:1063`).
    - `start` raised `TypeError`, which the helper re-raised after `released.set()`.
    - The test then spun forever in `while not release_wait_polled.is_set(): await asyncio.sleep(0)`. With `-o timeout=30 -o timeout_method=signal` it failed: `Timeout (>30.0s) from pytest-timeout`.
    - Under the repo's default `timeout = 300`, `timeout_method = "thread"` (`pyproject.toml:668-669`), the hang ends the whole pytest process instead. Task 3 runs the sweep with `timeout_method=signal`.
  - The other 12 are reached through `pool.start` directly, or (mlx) through `run_bounded_daemon_with_timeout`. Neither is changed.
- **The sync sibling** `run_bounded_daemon_with_timeout` (`:237`) also omits the argument. Its only app caller, `mlx_provider.py:139`, passes a `_RegistryWorkerPoolView` whose `start` takes no `exhaustion_message` (`:163-174`). It also turns `DaemonCapacityError` into `ChatRateLimitError("MLX busy (max concurrency reached)")` (`:146-153`), so the default text is never seen. Not changed.
- **RED:** the strengthened assertion in `test_daemon_drain_after_timeout_rejects_until_exit_then_raises` fails on dev with `- Streaming worker capacity is exhausted / + capacity-rejected capacity is exhausted` (1 failed).
- **GREEN, and the sweep** of every test file that names `await_bounded_daemon_with_timeout`, `BoundedDaemonPool`, `DaemonCapacityError`, one of the four pools, or one of the three claims/audio/RAG helpers (35 files): `2300 passed, 4 skipped` (869 s, with r1's Task 1 applied too); the review measured the same, 703 s. `test_async_stream_task_capacity.py` alone: `17 passed`. The four skips (the review, with `-rs`):
  - `test_chat_endpoint_simplified.py:766` (streaming with TestClient);
  - `test_streaming_utils.py:2055` (flaky coordination test);
  - `test_rag_evaluator_embeddings.py:200` and `:238` (API keys).
  - r2 changes no file in this sweep's scope, so it was not re-run.

## Global Constraints

- **Python:** `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Each command block below that uses `$PY` assumes it is set; set it at the top of the block if your shell does not keep variables between calls. **Tests:** `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q <paths>`. PostgreSQL params use the running `tldw_postgres_test` container through `pg_database_config`; they skip when it is unreachable, and a skip is recorded, not counted as a pass.
- **Branch** `fix/fva-endpoint-and-capacity-message` from `origin/dev`; one PR to `dev`. Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, `git stash` or force-push.
- **Test first:** each task's RED run is on unchanged source and fails for the reason measured above.
- **No API change.** The endpoint keeps its signature, docstring and `user_id: Optional[int]`, so the OpenAPI document and `apps/packages/api-types` do not change. Do not add a `Query(description=...)`.
- **No new helper outside `claims_service.py`, no new setting.** Reuse `_legacy_user_has_platform_admin_claims`, `_claims_user_override_db` and `scoped_context`. Do not import from `api/v1/API_Deps` in core: `tests/lint/test_core_to_api_import_boundary.py` freezes that set.
- **Isolation tests run on both backends** (the owner's rule: an isolation test is parameterized over the content backend). The member, admin-for-another-user and real-auth tests are; the admin-for-itself test is SQLite only (Review Focus 1).
- **Long-running sweeps:** pass `-o timeout=120 -o timeout_method=signal`. The repo default, `timeout_method = "thread"`, ends the whole process on one hang.
- **An exit code of `-6` after all tests passed** is the known macOS at-exit abort (What was measured). Rerun once and record both runs. Do not count it as a pass or a failure of this change without the rerun.
- **Lint:** `$PY -m ruff check --output-format concise <touched .py files>`. Expected: the findings dev's files already have, as listed above, and nothing else. Do not run `ruff --fix`: those import blocks are already unsorted on dev.
- **Line-number baselines:** none of the touched files is in `Helper_Scripts/ci/scope_predicate_baseline.txt` (checked).
- **Docs:** `Docs/Operations/Usage_Quotas.md`, `Docs/Design/2026-10-09-llm-usage-accounting-design.md` and `Docs/RAG/FVA_Pipeline.md` have no `Docs/Published` mirror. `Helper_Scripts/refresh_docs_published.sh` copies only `Operations/Env_Vars.md` from Operations, and nothing from `Design` or `RAG` (`:137-168`). Edits are anchored on quoted text.
- **Backlog:** `tools/backlog-py` only (`PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task ...`). Use `--append-notes`, explicit `--id` for new tasks, and `git add -A backlog/tasks`.
  - **New ids are taken under the id lock** (Task 4 Step 1). Write each id literally in later commands; a shell variable does not survive between agent calls (M1).
  - `TASK-<N>` below stands for the id Task 4 Step 1 prints.
- **Embedded code:** no block here has trailing whitespace; keep it that way when copying.

## Decisions for the owner (recommendation first)

1. **Whether this PR also makes FVA find counter-evidence, and how much adjudication may cost.**
   - **The problem.** Today FVA never adjudicates in the app: a 0.3 floor applied to fused scores (at most 1/61) drops every counter-evidence document (measured above).
   - **What bounds the cost.** Fixing that makes adjudication run, and adjudication makes one LLM call per document. The request schema (50 claims, `top_k` ≤ 100) is the only hard bound on how many:
     - usage quotas are off by default;
     - with them on, the per-call check is fail-open, and lets one request's in-flight calls finish;
     - `max_budget_usd` only gates entry to falsification.

   The worst cases below are LLM calls per request, all 50 claims falsified.

   - **Recommended: A-capped. Fold the fix in, with the supporting side capped at the verifier's 5 documents** (Task 1, TASK-13881's new AC #5 and AC #6). This is what the tasks below implement.
     - **Worst case:** 1 + min(`top_k`, 5) + 15 ≤ 21 calls per falsified claim, so **1,050 per request at any `top_k` ≥ 5**.
     - **Diff:**
       - `anti_context_retriever.py` +6/−3: one filter line deleted, with a comment saying why;
       - `fva_pipeline.py` +6/−1: one constant and one slice;
       - tests: `test_anti_context_retriever.py` +69/−13, `test_fva_pipeline.py` +22, `test_adjudicator.py` +24. The endpoint tests drop their floor lift.
     - **The verdict logic holds** (Review Focus 11). The rules are unchanged. `support_score` is a mean over the supporting documents judged SUPPORTS, now taken over the verifier's own window instead of all of `top_k`.
     - **The design choice, made:** neither half has an absolute relevance floor.
       - The FTS `min_score` is relative: it trims each query's weakest matches, and the best always passes.
       - The vector scale depends on the store and the embedding model, so no threshold was measured that would hold across them.
       - The adjudicator's stance check decides, and a NEUTRAL document does not count (pinned by the new adjudicator test).
       - The signal risk this leaves is Review Focus 10.
   - **Alternative: A-uncapped. Fold the fix in as r2 had it** (Task 1 without its cap steps).
     - **Worst case:** 1 + `top_k` + 15 calls per falsified claim, so **1,300 per request at the default `top_k` of 10, and 5,800 at 100**. The supporting calls repeat the same base documents for every falsified claim.
     - **Diff:** the above, less `fva_pipeline.py` and the cap test.
     - **What it changes in this plan:** "If Decision 1 is A-uncapped", after Task 4.
   - **Alternative: B. Land the repair only, and file the fix as a follow-up.**
     - **Worst case:** 1 verify call per claim, so **50 per request**. But every falsified claim also runs 3 anti-context retrievals whose results are discarded: up to 150 per request, and as many query embeddings when a vector store is configured (possibly paid). For that price the endpoint returns no evidence.
     - **Security:** holding the repair back exposes nothing. On dev the endpoint fails at a constructor before it reads any data.
     - **What it changes in this plan:** "If Decision 1 is B", after Task 4.

## Review Focus

1. **On PostgreSQL, an admin's own request (no `user_id`) reads every account's media.** The media RLS policy admits every row when `app.is_admin = '1'` (`postgres_rls.py`), as it does for any admin read, `/rag/search` included. The same holds for `?user_id=<own id>`, because of the short-circuit. On SQLite, the same request reads the admin's own file.
   - This plan does not change that, and AC #3 does not ask it to: "the authenticated user's media DB" is the admin's own session.
   - So `test_multi_user_admin_without_user_id_runs_for_itself` is SQLite only. A reviewer who reads "the authenticated user's" as "only rows the admin owns" should raise it as a product question about admin reads, not about this endpoint.
2. **The admin's named user is read with personal visibility only.**
   - `scoped_context(user_id=<named>, is_admin=False)` carries no team or org ids. So that user's team- or org-shared media is not evidence, though their own personal media is.
   - The code says so in a `ponytail:` comment. Adding the named user's memberships needs an AuthNZ lookup; add it if admins need shared media.
3. **A non-admin who names another user gets no signal.** The answer is a 200 with its own results, and its own quota and usage, exactly as AC #3 and `rebuild_claims` say. `_resolve_media_db(admin_required=True)` would answer 403 instead; the ACs chose "ignored".
4. **Real multi-user auth runs in one test; the rest inject it.**
   - `test_real_multi_user_api_key_auth_resolves_the_same_owner` creates an admin and a member in an AuthNZ SQLite DB with real API keys, through the review's recipe. The request then goes through the real `get_request_user`.
   - The other multi-user tests set `AUTH_MODE=multi_user` and override `get_request_user` with the `User` and content scope that real auth produces (`User_DB_Handling.py:1361-1368`). The review checked those values field by field.
   - Single-user mode runs for real: the `X-API-KEY` header goes through the real `get_request_user` and `authenticate_api_key_user`.
5. **What the stubs stand in for.**
   - **The provider stub:** it runs `llm_quota_gate.check_llm_call_sync()`, the check the provider base class runs first. It records `current_llm_user_id()`, the user the base class records the call for (`usage_accounting._plan_call`). The verifier's call is `records_itself`, so its row comes from the claims recorder, which the test spies on directly (`usage_tracker.log_llm_usage`).
   - **The vector-store variants:** `create_from_settings_for_user` returns a per-user fake collection. The query embedding is stubbed (`Embeddings_Create.create_embeddings_batch`) as a one-number toy vector, which marks an anti-context query by its template words. So a real `MediaDBRetriever` runs the real hybrid path, and the vector half's nearest neighbour is that user's counter-evidence row.
6. **The PostgreSQL fixture creates a role that cannot bypass RLS, then drops it.**
   - It follows the pattern of `tests/RAG/test_restricted_postgres_media_retrieval.py`, whose teardown can hang on a session left idle in a transaction (TASK-13900).
   - This fixture closes the pool first and sets `lock_timeout = '10s'` before `DROP OWNED BY`, so a leak fails the test instead of hanging the run.
   - After every r1, r2 and r3 run, including 27 mutant runs and the reviews' probes, `pg_roles` held no `fva_endpoint_%` role.
7. **The capacity message changes for all 8 callers**, the two streaming pools included, for example `audio-stream-next capacity is exhausted`. It is logged, not sent.
8. **`_verify_claims_with_fva_for` keeps the body as it was**, apart from the three constructor fixes and the owner id. One behaviour changes: `job_context` is now always built, not only with `max_budget_usd`, so every `claims_verify` row carries the owner's id explicitly. Before, a request without a budget fell back to the content scope; after the switch that is the same user.
9. **The request schema, not the quota, bounds FVA's LLM calls.**
   - **The worst case.** Up to 50 claims × (1 + min(`top_k`, 5) + 15) = 1,050 LLM calls per request with Task 1's cap. Without the cap it is 1,300 at the default `top_k` and 5,800 at 100.
   - **Usage quotas are off by default.**
   - **With quotas on,** every call is checked for the request's user. But the check asks for 1 token and is fail-open: a 2 s timeout, or no accounting loop, admits the call. Calls in flight finish: up to `max_concurrent_falsifications` claims at once, default 5, at most 20. So the quota limits the overshoot to one request's in-flight calls; it is not a hard bound.
   - **The other limits:**
     - `_can_afford_falsification` checks a fixed $0.01 per claim before falsification, and the stance calls are not reserved against `max_budget_usd`;
     - `falsification_timeout_seconds` bounds only the anti-context retrieval.
   - That is pre-existing FVA design, now reachable. A per-request call budget would be a separate change; it is not filed, because no AC or report asks for it.
10. **No absolute relevance floor on the counter-evidence** (Task 1's `ponytail:` comment).
    - **The cost:** a weakly related hit costs one stance call, then counts for nothing when it is NEUTRAL.
    - **The signal risk:** the contradict score is the confidence-weighted mean over CONTRADICTS documents only, so NEUTRAL ones do not dilute it. One document wrongly judged CONTRADICTS at confidence ≥ 0.7, among many unfloored neighbours, sets the contradict score to about its confidence. With support below 0.4, that gives REFUTED (`adjudicator.py:443-450`). Every extra document assessed is one more chance of that.
    - **The upgrade:** a per-store similarity floor, once a threshold is measured for the configured embedding model.
11. **The cap changes which supporting documents are averaged, not the rules.**
    - `support_score` is the mean over the first 5 base documents judged SUPPORTS: the verifier's window.
    - In edge cases a verdict can move either way:
      - if only a document ranked 6th or lower SUPPORTS, support becomes 0, and a contradiction above 0.7 then gives REFUTED;
      - if weak supports ranked 6th or lower dragged the mean down, support rises.
    - `supporting_evidence` lists at most 5 documents.

## ADR assessment

Not needed.
- **Governing ADRs, unchanged:**
  - ADR-064: quotas are per user;
  - ADR-067: rows are written for "the user the call is for, taken from an explicit `llm_call_context` or the content scope". This plan sets the content scope;
  - ADR-052's addendum: the `admin` permission is an administrator everywhere. `_legacy_user_has_platform_admin_claims` aliases `PLATFORM_ADMIN_ROLES` and `PLATFORM_ADMIN_PERMISSIONS`;
  - ADR-020: per-user paths come from `DatabasePaths`, through `get_user_media_db_path`.
- **No new durable rule.**
  - "A `user_id` parameter counts only for a platform admin" is the existing claims convention (`rebuild_claims`, `rebuild_all_media`, `rebuild_claims_fts`).
  - Design §1 already says the content scope decides who a call is for.
  - Task 1 fixes a filter inside one class with one caller, and limits how many documents one pipeline assesses.
- TASK-13888 changes a log message.

## PR split

**One PR.**
- The tasks are independent, but none holds another up.
- TASK-13888 is one line plus one test pool, and was found by the same review.
- Four commits keep them separate for review: Task 1, Task 2, Task 3, and the docs and backlog.

## Conflicts with other plans and PRs

Checked on 2026-10-22:
- every open PR's file list, paged through the files API for the 10 PRs with more than 100 files (`fu2c_bigpr_overlap.py`);
- the sibling plans in `<scratchpad>` (the review, with `git merge-file`).

| Place | This plan | Other | How to merge |
|---|---|---|---|
| `endpoints/claims.py`, `claims_service.py` | lines 1021-1036; the import at `:69` and `:4686-4850` | PR #3269 touches `claims.py:10`, `:46-53` and `claims_service.py:2842`, `:2991` | No shared or adjacent hunk; both orders apply cleanly |
| `Usage_Quotas.md:95` | Task 4 Step 2 rewrites the HTTP endpoint bullet | `plan_fu2_workflows.md` Task 4 Step 5 rewrites `:96`, the workflow bullet | Adjacent lines: `git merge-file` reports 1 conflict. Whichever PR lands second keeps the landed text and re-applies its own sentence by its quoted text |
| `Usage_Quotas.md` | as above | `plan_fu2_ingest_openapi.md` edits `:93`; `plan_fu2_pricing.md` edits `:108` | Both merge cleanly (`git merge-file`: 0 conflicts for `:93`; `:108` is far away) |
| everything else in this plan | — | no open PR or sibling plan | none |

---

### Task 0: branch, plan, tasks In Progress

**Files:** `Docs/superpowers/plans/2026-10-22-fva-endpoint-and-capacity-message.md` (this plan). Backlog: TASK-13881, TASK-13888.

- [ ] **Step 1**

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
git fetch origin dev --quiet && git switch -c fix/fva-endpoint-and-capacity-message origin/dev
cp <scratchpad>/plan_fu2_claims.md Docs/superpowers/plans/2026-10-22-fva-endpoint-and-capacity-message.md
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13881 -s "In Progress" \
  --ac "FVA's anti-context retrieval keeps the counter-evidence hybrid retrieval returns (its min_score trims each query's weakest full-text matches before fusion), so the adjudicator runs, with a test on a real MultiDatabaseRetriever with and without a vector store" \
  --ac "Adjudication assesses at most the verifier's 5 base documents on the supporting side, with a test" \
  --append-notes "Plan: Docs/superpowers/plans/2026-10-22-fva-endpoint-and-capacity-message.md. AC #5 and AC #6 added (owner Decision 1, A-capped): AntiContextRetriever filtered the fused hybrid list with min_relevance_score 0.3, but an RRF score is at most 1/61, so FVA never found counter-evidence and never adjudicated; the post-fusion filter is deleted (the retriever's min_score already trims each query's weakest full-text matches before fusion; no absolute floor, the adjudicator's stance check decides). Each supporting document costs one stance call per falsified claim, so adjudication assesses the verifier's 5 documents: worst case 1,050 LLM calls per request instead of 50 x (16 + top_k). verify_claims_with_fva resolves the owner first (user_id only for _legacy_user_has_platform_admin_claims, as rebuild_claims), then runs the pipeline for it; an admin naming another user runs inside scoped_context(user_id, is_admin=False) and _claims_user_override_db, so the media visibility filter (SQLite) or RLS (PostgreSQL), the quota decision and the usage rows all read that user. The retriever's user_id picks the vector store; ClaimsJobContext(user_id=owner id) writes the claims_verify rows."
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13888 -s "In Progress" \
  --append-notes "Plan: Docs/superpowers/plans/2026-10-22-fva-endpoint-and-capacity-message.md. await_bounded_daemon_with_timeout passes exhaustion_message=f'{name} capacity is exhausted' to pool.start."
git add -A backlog/tasks Docs/superpowers/plans/2026-10-22-fva-endpoint-and-capacity-message.md
```

Write the scratchpad path in full in place of `<scratchpad>`. For A-uncapped or B, change the `--ac` lines and the AC sentences of the note as their sections after Task 4 say.

### Task 1: FVA's anti-context retrieval keeps the counter-evidence it finds, and adjudication assesses the verifier's window (TASK-13881 AC #5, AC #6)

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/anti_context_retriever.py` (`:55`, `:186-187`).
- Modify: `tldw_Server_API/app/core/Claims_Extraction/fva_pipeline.py` (after the metrics fallback at `:61-65`; `:273`).
- Test: `tldw_Server_API/tests/Claims_Extraction/test_anti_context_retriever.py` (`:341-363`, and the end of the file).
- Test: `tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py` (the end of the file).
- Test: `tldw_Server_API/tests/Claims_Extraction/test_adjudicator.py` (the end of the file).

**Interfaces:**
- `AntiContextRetriever.retrieve_anti_context(...)` keeps its signature. It no longer filters by `doc.score`. Its retrieval config still carries `min_score=self.config.min_relevance_score`.
- `FVAPipeline.process_claim(...)` keeps its signature. It hands `adjudicate` at most `_ADJUDICATED_SUPPORTING_DOCS` (5) supporting documents.
- `ClaimAdjudicator` is unchanged; the new adjudicator test pins its existing behaviour.

- [ ] **Step 1: write the failing tests.** In `test_anti_context_retriever.py`, replace the test that pinned the old filter:

```python
    async def test_filters_by_min_score(self):
        """Should filter documents below minimum relevance score."""
        mock_docs = [
            Document(id="1", content="High score", metadata={}, score=0.8),
            Document(id="2", content="Low score", metadata={}, score=0.1),
        ]
        mock_retriever = MockRetriever(mock_docs)
        config = AntiContextConfig(min_relevance_score=0.5)
        retriever = AntiContextRetriever(mock_retriever, config)

        claim = MockClaim(id="1", text="Some claim")

        results = await retriever.retrieve_anti_context(
            claim=claim,
            original_doc_ids=set(),
        )

        all_docs = []
        for r in results:
            all_docs.extend(r.documents)

        # Should only include high-score document
        assert all(d.score >= 0.5 for d in all_docs)
```

with

```python
    async def test_min_score_is_the_retrievers_floor_not_a_filter_on_fused_scores(self):
        """The floor goes to the retriever, which applies it before fusion; fused scores are kept (TASK-13881)."""
        mock_docs = [
            Document(id="1", content="Fused rank 1", metadata={}, score=1 / 61),
            Document(id="2", content="Fused rank 2", metadata={}, score=1 / 62),
        ]
        mock_retriever = MockRetriever(mock_docs)
        config = AntiContextConfig(min_relevance_score=0.5)
        retriever = AntiContextRetriever(mock_retriever, config)

        results = await retriever.retrieve_anti_context(
            claim=MockClaim(id="1", text="Some claim"),
            original_doc_ids=set(),
        )

        assert {call["config"].min_score for call in mock_retriever.retrieve_calls} == {0.5}
        assert {d.id for r in results for d in r.documents} == {"1", "2"}
```

Then append to the end of the file, after `test_contrary_templates_exist` and two blank lines:

```python
class _NearestNeighbourStore:
    """A vector store whose nearest neighbour of every query is ``hit``; it notes the collections searched."""

    _initialized = True

    def __init__(self, hit) -> None:
        self.hit = hit
        self.collections: list[str] = []

    async def search(self, collection_name, query_vector, k=10, filter=None, include_metadata=True):
        self.collections.append(collection_name)
        return [self.hit]


@pytest.mark.unit
@pytest.mark.parametrize("with_vector_store", [False, True])
async def test_counter_evidence_from_a_real_hybrid_retriever_is_kept(tmp_path, monkeypatch, with_vector_store):
    """A fused (RRF) score is at most 1/61: a 0.3 floor on it dropped every document, so FVA never adjudicated.

    Without a vector store the FTS hit comes through. With one, so does the vector-only hit, which FTS cannot find.
    """
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create
    from tldw_Server_API.app.core.RAG.rag_service import database_retrievers
    from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import VectorSearchResult

    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="7")
    with scoped_context(user_id=7, is_admin=False):
        fts_hit, _, _ = db.add_media_with_keywords(
            title="Counter-evidence", media_type="text", owner_user_id=7, keywords=None,
            content="Evidence against the claim that the sky is green: it is blue.",
        )
        vector_hit, _, _ = db.add_media_with_keywords(
            title="Optics", media_type="text", owner_user_id=7, keywords=None,
            content="Rayleigh scattering colours daylight blue.",
        )
    store = _NearestNeighbourStore(
        VectorSearchResult(
            id=f"{vector_hit}_0", content="Rayleigh scattering colours daylight blue.",
            metadata={"media_id": str(vector_hit)}, score=0.42, distance=0.58,
        )
    )
    monkeypatch.setattr(
        database_retrievers, "create_from_settings_for_user", lambda settings, user_id="0": store if with_vector_store else None
    )
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: {})
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", lambda texts, *a, **k: [[0.1, 0.2, 0.3] for _ in texts])
    retriever = database_retrievers.MultiDatabaseRetriever({}, user_id="7", media_db=db)
    try:
        with scoped_context(user_id=7, is_admin=False):
            results = await AntiContextRetriever(retriever).retrieve_anti_context(
                claim=MockClaim(id="1", text="The sky is green."), original_doc_ids=set()
            )
    finally:
        db.close_connection()

    found = {d.id for r in results for d in r.documents}
    assert found == ({str(fts_hit), str(vector_hit)} if with_vector_store else {str(fts_hit)})
    assert store.collections == (["user_7_media_embeddings"] * 3 if with_vector_store else [])
```

The real-retriever test runs a real `MultiDatabaseRetriever` on a seeded SQLite media DB. Only the vector store and the query embedding are fakes.
- **Without a vector store,** the FTS hit must come through; on dev it is dropped.
- **With one,** a vector-only hit must come through too. Its content shares no word with the anti-context queries, so FTS cannot find it, and it shows that the vector half reaches adjudication with no floor.

Append to `test_fva_pipeline.py`, after `test_process_claim_lets_an_adjudication_refusal_through` and two blank lines (the cap, AC #6):

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_adjudication_assesses_only_the_verifiers_top_five_supporting_documents():
    """Each supporting document costs a stance call per falsified claim, so that side is the verifier's window (TASK-13881)."""
    low_confidence = MockClaimVerification(
        claim=MockClaim(id="1", text="Test"),
        confidence=0.5,
        evidence=[MockEvidence(doc_id="0", snippet="Support 0")],
    )
    claims_engine = MockClaimsEngine(MockVerifier(verification=low_confidence))
    anti_docs = [Document(id="anti_1", content="Contradicting evidence", metadata={}, score=0.7)]
    pipeline = FVAPipeline(claims_engine, MockRetriever(documents=anti_docs))
    base_docs = [Document(id=str(i), content=f"Support {i}", metadata={}, score=1.0 - i / 10) for i in range(8)]

    result = await pipeline.process_claim(claim=MockClaim(id="1", text="Test claim"), query="test", documents=base_docs)

    prompts = [call.args[2] for call in claims_engine._analyze.call_args_list]
    assessed = sorted(d.content for d in base_docs + anti_docs if any(f"Evidence: {d.content}\n" in p for p in prompts))
    assert assessed == ["Contradicting evidence"] + [f"Support {i}" for i in range(5)]
    assert {a.document.id for a in result.adjudication.supporting_evidence} == {"0", "1", "2", "3", "4"}
```

`MockClaimsEngine`'s default `_analyze` is an `AsyncMock` answering SUPPORTS; the test reads which documents reached it from its calls, so no fake is assigned to an attribute (the test-quality ratchet counts that as `stub_injection`).

Append to `test_adjudicator.py`, after `TestClaimAdjudicatorLLMQuota` and two blank lines (the pin the no-floor design rests on):

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_neutral_counter_evidence_does_not_count():
    """Only CONTRADICTS documents enter the contradict score; FVA's unfloored anti-context list rests on this (TASK-13881)."""

    def _analyze(api_endpoint, input_data, prompt, *args, **kwargs):
        if "Evidence: The sky is blue, not green." in prompt:
            return '{"stance": "CONTRADICTS", "confidence": 0.9}'
        return '{"stance": "NEUTRAL", "confidence": 0.6}'

    claim = MockClaim(id="1", text="The sky is green.")
    neutral = [Document(id=f"n{i}", content=f"Unrelated note {i}.", metadata={}, score=0.5) for i in range(3)]

    result = await ClaimAdjudicator(llm_analyze_fn=_analyze).adjudicate(
        claim=claim,
        supporting_docs=[],
        contradicting_docs=[Document(id="c", content="The sky is blue, not green.", metadata={}, score=0.5), *neutral],
        original_verification=MockClaimVerification(claim=claim),
    )

    assert result.contradict_score == 0.9
    assert [a.document.id for a in result.contradicting_evidence] == ["c"]
```

- [ ] **Step 2: RED**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Claims_Extraction/test_anti_context_retriever.py \
  "tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py::test_adjudication_assesses_only_the_verifiers_top_five_supporting_documents" \
  "tldw_Server_API/tests/Claims_Extraction/test_adjudicator.py::test_neutral_counter_evidence_does_not_count"
```

Expected on `e06eeec68c` (measured in one run, `fu2c_r3_red_t1.log`): `4 failed, 23 passed`:
- the replaced anti-context test: `assert set() == {'1', '2'}`;
- `test_counter_evidence_from_a_real_hybrid_retriever_is_kept[False]`: `assert set() == {'1'}`;
- `test_counter_evidence_from_a_real_hybrid_retriever_is_kept[True]`: `assert set() == {'1', '2'}`;
- the cap test: `Left contains 3 more items, first extra item: 'Support 5'` (all 8 base documents assessed);
- `test_neutral_counter_evidence_does_not_count` **passes**: it pins the adjudicator's existing filter, which this plan relies on and does not change. Mutant N1 shows that it bites.

- [ ] **Step 3: implement.** In `anti_context_retriever.py`, replace

```python
    min_relevance_score: float = 0.3
```

with

```python
    min_relevance_score: float = 0.3  # the retriever's min_score: trims each query's weakest full-text matches
```

and replace

```python
            # Filter by minimum score
            docs = [d for d in docs if d.score >= self.config.min_relevance_score]
```

with

```python
            # No score filter here. min_relevance_score goes to the retriever as min_score, which drops each query's
            # weakest full-text matches before hybrid fusion (scores are min-max normalised per query, so the best
            # match always passes). A fused (RRF) score is at most 1/61, so a floor on it dropped every document.
            # ponytail: no absolute relevance floor on either half; the adjudicator's stance check decides, and a
            # NEUTRAL document does not count. A similarity floor per vector store is the upgrade.
```

In `fva_pipeline.py`, replace

```python
    def observe_histogram(*args, **kwargs) -> None:
        pass

```

with

```python
    def observe_histogram(*args, **kwargs) -> None:
        pass


# Each supporting document costs one stance call per falsified claim, so adjudication assesses the documents the
# verifier judged (HybridClaimVerifier.verify's top_k, 5), not every base document the request retrieved.
_ADJUDICATED_SUPPORTING_DOCS = 5

```

and replace

```python
                                supporting_docs=documents,
                                contradicting_docs=anti_docs,
```

with

```python
                                supporting_docs=documents[:_ADJUDICATED_SUPPORTING_DOCS],
                                contradicting_docs=anti_docs,
```

- [ ] **Step 4: GREEN**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Claims_Extraction/test_anti_context_retriever.py \
  tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py tldw_Server_API/tests/Claims_Extraction/test_adjudicator.py
```

Expected: `91 passed` (measured on the r3 prototype).

- [ ] **Step 5: commit**

```bash
git add -A backlog/tasks Docs/superpowers/plans/2026-10-22-fva-endpoint-and-capacity-message.md \
  tldw_Server_API/app/core/Claims_Extraction/anti_context_retriever.py tldw_Server_API/app/core/Claims_Extraction/fva_pipeline.py \
  tldw_Server_API/tests/Claims_Extraction/test_anti_context_retriever.py tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py \
  tldw_Server_API/tests/Claims_Extraction/test_adjudicator.py
git commit -m "fix(claims): FVA keeps the counter-evidence hybrid retrieval finds, and adjudicates against the verifier's window (TASK-13881)" \
  -m "AntiContextRetriever filtered the fused list with the 0.3 relevance threshold, but a fused (RRF) score is at most 1/61, so every counter-evidence document was dropped; the second filter is deleted. Each supporting document costs one stance call per falsified claim, so adjudication assesses the verifier's 5 documents, not all of top_k: a request's worst case drops from 50 x (16 + top_k) LLM calls to 1,050." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: `POST /claims/verify/fva` runs, for the user the admin gate resolves (TASK-13881 AC #1-#3)

**Files:**
- Create: `tldw_Server_API/tests/Claims_Extraction/test_fva_verify_endpoint.py`.
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_service.py` (import after `:69`; `verify_claims_with_fva` at `:4686-4835`).
- Modify: `tldw_Server_API/app/api/v1/endpoints/claims.py` (`:1024-1037`).

**Interfaces:**
- Consumes:
  - `_legacy_user_has_platform_admin_claims(current_user) -> bool`;
  - `_claims_user_override_db(user_id: int)`, a context manager yielding `(MediaDatabase, db_path)`;
  - `scope_context.scoped_context(**kwargs)`;
  - `MultiDatabaseRetriever(db_paths, user_id="0", *, media_db=...)`;
  - `RetrievalConfig(max_results=...)`;
  - `ClaimsJobContext(user_id=...)`.
- Produces:
  - `verify_claims_with_fva(*, claims, query, sources=None, top_k=10, fva_config=None, user_id: int | None = None, current_user: User, db: MediaDatabase) -> dict`. `current_user` and `db` are now required; the endpoint always passed both;
  - private `_verify_claims_with_fva_for(owner_user_id: int, db, *, claims, query, sources, top_k, fva_config) -> dict`;
  - the endpoint's response model is unchanged.

- [ ] **Step 1: write the failing tests.** Create `tldw_Server_API/tests/Claims_Extraction/test_fva_verify_endpoint.py`:

```python
"""POST /claims/verify/fva end to end: who the request runs for, and its quota refusals (TASK-13881).

The claims router alone, on real media DBs: one SQLite file per user, or PostgreSQL's shared tables
under a role that cannot bypass row-level security, with or without a per-user vector store. Only the
provider call and the query embedding are stubbed: the provider stub runs the quota check the
instrumented provider runs first, and notes who the call is for.
"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi import Depends, FastAPI

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import claims as claims_endpoints
from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_single_user_instance
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Claims_Extraction import claims_engine, claims_service
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context, set_scope
from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create
from tldw_Server_API.app.core.RAG.rag_service import database_retrievers
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import VectorSearchResult
from tldw_Server_API.app.core.Usage import llm_accounting, llm_quota_gate, quota_resolver, usage_tracker
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id
from tldw_Server_API.app.core.Usage.quota_checks import LLMQuotaRefusal
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

pytestmark = pytest.mark.unit

OTHER_USER = 2
STRANGER = 5
MEMBER = 7
ADMIN = 9
_BODY = {
    "claims": [{"text": "The sky is green."}],
    "query": "clear day",
    "top_k": 3,
    "fva_config": {"force_claim_types": ["general"], "max_budget_usd": 5.0},
}
_REFUSAL = LLMQuotaRefusal("llm_tokens_month", 100, 100, "Monthly LLM token limit reached", 60)
_ANTI_QUERY_WORDS = ("against", "contradicts", "disproves")  # the first three negation templates
_BOTH_BACKENDS = pytest.mark.parametrize("fva", ["sqlite", "postgresql"], indirect=True)
_BACKENDS_AND_STORES = pytest.mark.parametrize(
    "fva", ["sqlite", "postgresql", "sqlite+vectors", "postgresql+vectors"], indirect=True
)


class _Spies:
    """What the request did, per AC #3: the evidence's DB, the vector store, the quota decisions, the usage rows."""

    def __init__(self, with_vectors: bool) -> None:
        self.with_vectors = with_vectors
        self.admitted = 10**6
        self.decided_for: list[int] = []
        self.calls: list[tuple[str, int | None]] = []
        self.usage_rows: list[tuple[str, int | None]] = []
        self.vector_store_users: list[str] = []
        self.vector_collections: list[str] = []


@pytest.fixture
def restricted_pg_backend(request):
    """A PostgreSQL content backend whose role cannot bypass RLS (as tests/RAG/test_restricted_postgres_media_retrieval.py)."""
    from psycopg import sql

    config = request.getfixturevalue("pg_database_config")
    admin = DatabaseBackendFactory.create_backend(config)
    role, password = "fva_endpoint_" + uuid4().hex, uuid4().hex
    with admin.transaction() as conn:
        conn.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS NOINHERIT NOCREATEDB NOCREATEROLE"
            ).format(sql.Identifier(role), sql.Literal(password))
        )
        conn.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA public TO {}").format(sql.Identifier(role)))
        conn.execute(
            sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                sql.Identifier(config.pg_database), sql.Identifier(role)
            )
        )
    backend = DatabaseBackendFactory.create_backend(
        replace(config, pg_user=role, pg_password=password, connection_string=None, pool_size=2, max_overflow=4)
    )
    try:
        yield backend
    finally:
        backend.get_pool().close_all()
        with admin.transaction() as conn:
            conn.execute("SET LOCAL lock_timeout = '10s'")  # fail, not hang, on a leaked session (TASK-13900)
            conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
        admin.get_pool().close_all()


@pytest.fixture
def fva(request, monkeypatch, tmp_path: Path):
    """The claims router, a seeded media DB per user, and spies on everything the request decides per user."""
    param = getattr(request, "param", "sqlite")
    backend = request.getfixturevalue("restricted_pg_backend") if param.startswith("postgresql") else None
    spies = _Spies(with_vectors=param.endswith("+vectors"))

    def _open(owner: int, file_owner: int | None = None) -> MediaDatabase:
        """``owner``'s media DB: their own SQLite file (or ``file_owner``'s), else the shared PostgreSQL tables."""
        if backend is not None:
            return MediaDatabase(db_path=":memory:", client_id=str(owner), backend=backend)
        return MediaDatabase(db_path=str(tmp_path / f"user_{file_owner or owner}.db"), client_id=str(owner))

    seeds = [(owner, owner) for owner in (OTHER_USER, MEMBER, ADMIN, int(get_single_user_instance().id))]
    # A row of another account in OTHER_USER's file, as PostgreSQL's shared tables hold. Only OTHER_USER's
    # own content scope, not an admin's, keeps it out of the evidence.
    seeds.append((STRANGER, OTHER_USER))
    rows: dict[int, list[tuple[int, str]]] = {}  # each owner's (media id, content): sky first, counter-evidence second
    for owner, file_owner in seeds:
        db = _open(owner, file_owner)
        with scoped_context(user_id=owner, is_admin=False):
            for title, content in (
                (f"Sky of user {owner}", f"From the home of user {owner} the sky is blue on a clear day."),
                (
                    f"Counter-evidence of user {owner}",
                    f"Evidence against the claim that the sky is green, by user {owner}: it is blue.",
                ),
            ):
                media_id, _, _ = db.add_media_with_keywords(
                    title=title, media_type="text", content=content, owner_user_id=owner, keywords=None
                )
                rows.setdefault(owner, []).append((media_id, content))
        db.close_connection()

    async def _decide(user_id, requested_tokens=1):
        spies.decided_for.append(user_id)
        return None if len(spies.decided_for) <= spies.admitted else _REFUSAL

    def _provider_call(**kwargs):
        kind = "adjudicate" if "supports, contradicts, or is neutral" in str(kwargs.get("system_message")) else "verify"
        spies.calls.append((kind, current_llm_user_id()))  # the user the call's usage row is written for
        llm_quota_gate.check_llm_call_sync()  # what the instrumented provider runs first
        if kind == "adjudicate":
            return json.dumps({"stance": "CONTRADICTS", "confidence": 0.95})
        return json.dumps({"label": "supported", "confidence": 0.95, "rationale": "ok"})

    async def _log_llm_usage(*, user_id, operation, **_kwargs):
        spies.usage_rows.append((operation, user_id))

    class _UserVectorStore:
        """One user's collection: an anti-context query's nearest neighbour is their counter-evidence row."""

        _initialized = True

        def __init__(self, owner: int) -> None:
            self.owner = owner

        async def search(self, collection_name, query_vector, k=10, filter=None, include_metadata=True):
            spies.vector_collections.append(collection_name)
            media_id, content = rows[self.owner][1 if query_vector[0] else 0]
            return [VectorSearchResult(f"{media_id}_0", content, {"media_id": str(media_id)}, 0.8, 0.2)]

    def _vector_store(settings, user_id="0"):
        spies.vector_store_users.append(str(user_id))
        return _UserVectorStore(int(user_id)) if spies.with_vectors else None  # None: FTS alone

    def _embed(texts, *args, **kwargs):
        return [[1.0 if any(word in text.lower() for word in _ANTI_QUERY_WORDS) else 0.0] for text in texts]

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _decide)
    monkeypatch.setattr(chat_service, "perform_chat_api_call", _provider_call)
    monkeypatch.setattr(usage_tracker, "log_llm_usage", _log_llm_usage)
    monkeypatch.setattr(database_retrievers, "create_from_settings_for_user", _vector_store)
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: {})
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", _embed)
    monkeypatch.setattr(claims_engine, "_claims_local_nli_enabled", lambda: False)
    # The real _claims_user_override_db opens the admin's target: their file, or the shared backend.
    monkeypatch.setattr(claims_service, "get_user_media_db_path", lambda uid: str(tmp_path / f"user_{int(uid)}.db"))
    if backend is not None:
        real_managed = claims_service.managed_media_database
        monkeypatch.setattr(claims_service, "managed_media_database", lambda **kw: real_managed(**kw, backend=backend))

    async def _own_media_db(current_user: User = Depends(get_request_user)):
        db = _open(int(current_user.id))
        try:
            yield db
        finally:
            db.close_connection()

    app = FastAPI()
    app.include_router(claims_endpoints.router, prefix="/api/v1")
    app.dependency_overrides[get_auth_principal] = lambda: None
    app.dependency_overrides[get_media_db_for_user] = _own_media_db

    async def post(query_string: str = "", headers: dict[str, str] | None = None, body: dict | None = None):
        monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                f"/api/v1/claims/verify/fva{query_string}", json=body or _BODY, headers=headers or {}
            )

    def as_multi_user(user_id: int, role: str) -> None:
        """Multi-user mode: the account and content scope multi-user auth resolves for ``role``."""
        monkeypatch.setenv("AUTH_MODE", "multi_user")
        reset_settings()

        async def _account() -> User:
            set_scope(user_id=user_id, is_admin=role == "admin")
            return User(id=user_id, username=f"name-{user_id}", role=role, roles=[role], is_admin=role == "admin")

        app.dependency_overrides[get_request_user] = _account

    return spies, post, as_multi_user


@pytest.fixture
async def real_api_keys(tmp_path, monkeypatch):
    """Multi-user mode for real: an AuthNZ SQLite DB with an admin (ADMIN) and a member (MEMBER), each with an API key."""
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-fva-endpoint-1234567890")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("VIRTUAL_KEYS_ENABLED", "false")
    reset_settings()
    await reset_db_pool()
    try:
        pool = await get_db_pool()
        ensure_authnz_tables(Path(pool.db_path))
        keys: dict[int, str] = {}
        manager = APIKeyManager()
        await manager.initialize()
        for user_id, role in ((ADMIN, "admin"), (MEMBER, "user")):
            await create_authnz_test_user(
                pool, username=f"u{user_id}", email=f"u{user_id}@example.com", role=role, user_id=user_id, is_verified=True
            )
            role_row = await pool.fetchrow("SELECT id FROM roles WHERE name = ?", role)
            role_id = int(role_row["id"] if isinstance(role_row, dict) else role_row[0])
            async with pool.transaction() as conn:
                await conn.execute("INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)", (user_id, role_id))
            created = await manager.create_api_key(user_id=user_id, name=f"k{user_id}", scope="write", expires_in_days=7)
            keys[user_id] = str(created["key"])
        yield keys
    finally:
        await reset_db_pool()


def _assert_all_for(spies: _Spies, response: httpx.Response, owner: int) -> None:
    """The evidence, the vector store, every quota decision and every usage row belong to ``owner``."""
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["anti_context_found"] > 0 and result["adjudication"] is not None, result
    evidence = result["supporting_evidence"] + result["contradicting_evidence"]
    assert evidence and all(f"user {owner}" in e["snippet"] for e in evidence), evidence  # the media DB
    assert set(spies.vector_store_users) == {str(owner)}
    assert set(spies.vector_collections) == ({f"user_{owner}_media_embeddings"} if spies.with_vectors else set())
    assert spies.decided_for and set(spies.decided_for) == {owner}
    assert {user for _op, user in spies.usage_rows} == {owner}  # claims_verify rows, by id
    assert {kind for kind, _user in spies.calls} == {"verify", "adjudicate"}
    assert {user for _kind, user in spies.calls} == {owner}


@_BACKENDS_AND_STORES
@pytest.mark.parametrize("query_string", ["", f"?user_id={OTHER_USER}"])
async def test_multi_user_member_runs_for_itself_whatever_user_id_it_sends(fva, query_string):
    """AC #1, AC #3: a non-admin's user_id is ignored for the media DB, vector store, quota and usage rows."""
    spies, post, as_multi_user = fva
    as_multi_user(MEMBER, "user")

    _assert_all_for(spies, await post(query_string), MEMBER)


@_BACKENDS_AND_STORES
async def test_multi_user_admin_runs_for_the_user_it_names(fva):
    """AC #3: an admin's user_id switches all four to that user, and only that user's rows are evidence."""
    spies, post, as_multi_user = fva
    as_multi_user(ADMIN, "admin")

    _assert_all_for(spies, await post(f"?user_id={OTHER_USER}"), OTHER_USER)


async def test_multi_user_admin_without_user_id_runs_for_itself(fva):
    """SQLite only: on PostgreSQL an admin's own scope reads every account's rows, as for any admin read."""
    spies, post, as_multi_user = fva
    as_multi_user(ADMIN, "admin")

    _assert_all_for(spies, await post(), ADMIN)


async def test_a_request_without_max_budget_usd_runs_too(fva):
    """On dev this body failed at MultiDatabaseRetriever(media_db=db), the body with a budget at ClaimsJobContext."""
    spies, post, as_multi_user = fva
    as_multi_user(MEMBER, "user")

    _assert_all_for(spies, await post(body={**_BODY, "fva_config": {"force_claim_types": ["general"]}}), MEMBER)


@_BOTH_BACKENDS
@pytest.mark.parametrize(("caller", "owner"), [(ADMIN, OTHER_USER), (MEMBER, MEMBER)])
async def test_real_multi_user_api_key_auth_resolves_the_same_owner(fva, real_api_keys, caller, owner):
    """The real multi-user get_request_user and AuthNZ DB: the admin's ?user_id= counts, the member's does not."""
    spies, post, _as_multi_user = fva

    response = await post(f"?user_id={OTHER_USER}", headers={"X-API-KEY": real_api_keys[caller]})

    _assert_all_for(spies, response, owner)


@pytest.mark.parametrize("names_other_user", [False, True])
async def test_single_user_mode_honours_user_id_because_the_single_user_is_an_admin(fva, monkeypatch, names_other_user):
    """Real single-user API-key auth: the single user has the admin role, so user_id counts, as for rebuild_claims."""
    spies, post, _as_multi_user = fva
    monkeypatch.setenv("AUTH_MODE", "single_user")
    reset_settings()
    single_user = int(get_single_user_instance().id)

    response = await post(
        f"?user_id={OTHER_USER}" if names_other_user else "",
        headers={"X-API-KEY": get_settings().SINGLE_USER_API_KEY},
    )

    _assert_all_for(spies, response, OTHER_USER if names_other_user else single_user)


@pytest.mark.parametrize(("admitted", "refused"), [(0, "verify"), (1, "adjudicate")])
async def test_a_quota_refusal_at_the_verify_or_adjudication_call_answers_402(fva, admitted, refused):
    """AC #2: the refusal reaches the client as the 402 limit_exceeded body, for the request's user."""
    spies, post, as_multi_user = fva
    as_multi_user(MEMBER, "user")
    spies.admitted = admitted

    response = await post()

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == "limit_exceeded"
    assert spies.calls[-1] == (refused, MEMBER)
    assert set(spies.decided_for) == {MEMBER}
```

Notes on the design of the tests:
- **What a non-admin gets for another user's `user_id`:** HTTP 200, with results drawn from its own media DB and vector store, its own quota decisions and its own usage rows. The parameter is ignored, without an error or a warning.
  - `test_multi_user_member_runs_for_itself_whatever_user_id_it_sends[?user_id=2-*]` pins all four: on both backends, with and without a vector store.
  - The real-auth member case pins it through real API-key auth.
- **Every 200 is checked for adjudication.** `_assert_all_for` requires `anti_context_found > 0` and an adjudication, so the evidence it checks came through the real anti-context path. There is no floor lift; Task 1 makes that path real.
- **The stranger's row** (user 5) is a personal row of another account inside user 2's SQLite file, as PostgreSQL's shared tables hold one. Only user 2's own non-admin scope keeps it out. That is why mutant S2 (a switch that keeps `is_admin=True`) fails on SQLite too.
- **Usernames differ from ids** (`name-7`, and `u7` in the real-auth DB), so a row written for the username fails (mutant U1).
- **`max_budget_usd` is in the default body,** so the `ClaimsJobContext` path that held constructor bug (1) runs in nearly every test. `test_a_request_without_max_budget_usd_runs_too` covers the body that failed on constructor (2) (M5).
- **The real-auth fixture** builds the AuthNZ SQLite DB under `tmp_path`, through the review's recipe (`create_authnz_test_user`, a `user_roles` row, `APIKeyManager.create_api_key`). It always calls `reset_db_pool()` on the way out, so no later test sees its pool.

- [ ] **Step 2: run them on the source as it stands after Task 1**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -o timeout=150 tldw_Server_API/tests/Claims_Extraction/test_fva_verify_endpoint.py
```

Expected: `22 failed` (measured): 20 `assert 500 == 200` and 2 `assert 500 == 402`. The 500 is the `TypeError` from `ClaimsJobContext(job_id=...)`, or from `MultiDatabaseRetriever(media_db=db)` for the body without a budget. If the PostgreSQL params skip, the container is unreachable: record `14 failed, 8 skipped` and the skip reason.

- [ ] **Step 3: implement.** In `claims_service.py`, after

```python
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
```

add

```python
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
```

Then make five replacements in `verify_claims_with_fva`. Each anchor is unique in the file on `e06eeec68c`.

(a) The signature's last three parameters. Replace

```python
    fva_config: dict[str, Any] | None = None,
    user_id: str | None = None,
    current_user: User | None = None,
    db: MediaDatabase | None = None,
) -> dict[str, Any]:
```

with

```python
    fva_config: dict[str, Any] | None = None,
    user_id: int | None = None,
    current_user: User,
    db: MediaDatabase,
) -> dict[str, Any]:
```

(b) The docstring's `Args` entries, and the split of the body into the private function. Replace

```python
        user_id: User ID for scoped retrieval
        current_user: Current authenticated user
        db: MediaDatabase instance

    Returns:
        FVA verification results with status changes and timing
    """
    import time
```

with

```python
        user_id: The user to verify for; honoured only for a platform admin, as in
            rebuild_claims. Anyone else's is ignored and the request runs for current_user.
        current_user: Current authenticated user
        db: The authenticated user's MediaDatabase

    Returns:
        FVA verification results with status changes and timing
    """
    owner_user_id = int(current_user.id)
    if user_id is None or int(user_id) == owner_user_id or not _legacy_user_has_platform_admin_claims(current_user):
        return await _verify_claims_with_fva_for(
            owner_user_id, db, claims=claims, query=query, sources=sources, top_k=top_k, fva_config=fva_config
        )
    # An admin verifying for another user: that user's media DB and content scope. The media visibility
    # filter, the quota decision and the usage rows all read the scope (TASK-13881).
    # ponytail: the user's personal media only; give the scope their team and org ids if admins need shared media.
    with scoped_context(user_id=int(user_id), is_admin=False), _claims_user_override_db(int(user_id)) as (owner_db, _):
        return await _verify_claims_with_fva_for(
            int(user_id), owner_db, claims=claims, query=query, sources=sources, top_k=top_k, fva_config=fva_config
        )


async def _verify_claims_with_fva_for(
    owner_user_id: int,
    db: MediaDatabase,
    *,
    claims: list[dict[str, Any]],
    query: str,
    sources: list[str] | None,
    top_k: int,
    fva_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run the FVA pipeline on ``db`` and ``owner_user_id``'s vector store, for that user."""
    import time
```

(c) Constructor calls (1) and (2), and the username fallback. Replace

```python
    budget: ClaimsJobBudget | None = None
    job_context: ClaimsJobContext | None = None
    max_budget_usd = config_dict.get("max_budget_usd")
    if max_budget_usd is not None:
        budget = ClaimsJobBudget(max_cost_usd=float(max_budget_usd))
        job_context = ClaimsJobContext(
            job_id=str(uuid4()),
            user_id=user_id or (current_user.username if current_user else None),
        )

    # Resolve user_id for retrieval
    resolved_user_id = user_id
    if resolved_user_id is None and current_user is not None:
        resolved_user_id = str(current_user.username) if hasattr(current_user, "username") else None

    # Build retriever
    retriever = MultiDatabaseRetriever(media_db=db)
```

with

```python
    budget: ClaimsJobBudget | None = None
    max_budget_usd = config_dict.get("max_budget_usd")
    if max_budget_usd is not None:
        budget = ClaimsJobBudget(max_cost_usd=float(max_budget_usd))
    job_context = ClaimsJobContext(user_id=owner_user_id)  # the claims_verify usage rows, by id

    # The user_id picks the vector store (collection user_{id}_media_embeddings).
    retriever = MultiDatabaseRetriever({}, user_id=str(owner_user_id), media_db=db)
```

(d) Constructor call (3). Replace

```python
    retrieval_config = RetrievalConfig(
        top_k=top_k,
        search_mode="hybrid",
    )
```

with

```python
    retrieval_config = RetrievalConfig(max_results=top_k)  # FTS and vector, merged: the hybrid search
```

(e) The pipeline's user. In the `pipeline.process_batch(` call, replace

```python
        user_id=resolved_user_id,
```

with

```python
        user_id=str(owner_user_id),
```

`uuid4` stays imported: the claim ids still use it. `MultiDatabaseRetriever({}, ...)` is the shape `endpoints/rpg.py:145-149` uses; with `media_db` given, the Media DB retriever needs no path.

In `endpoints/claims.py`, delete the "Resolve user_id string" block and pass the parameter through. Replace

```python
    # Resolve user_id string
    resolved_user_id: Optional[str] = None
    if user_id is not None:
        resolved_user_id = str(user_id)
    elif current_user is not None:
        resolved_user_id = str(getattr(current_user, "username", None) or getattr(current_user, "id", None))

    return await claims_service.verify_claims_with_fva(
        claims=claims_data,
        query=payload.query,
        sources=payload.sources,
        top_k=payload.top_k,
        fva_config=fva_config,
        user_id=resolved_user_id,
```

with

```python
    return await claims_service.verify_claims_with_fva(
        claims=claims_data,
        query=payload.query,
        sources=payload.sources,
        top_k=payload.top_k,
        fva_config=fva_config,
        user_id=user_id,
```

`Optional` stays imported in `claims.py`: the signatures still use it.

- [ ] **Step 4: GREEN, then the neighbourhood**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -o timeout=150 tldw_Server_API/tests/Claims_Extraction/test_fva_verify_endpoint.py
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -rs -o timeout=150 -o timeout_method=signal tldw_Server_API/tests/Claims tldw_Server_API/tests/Claims_Extraction
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/lint/test_core_to_api_import_boundary.py tldw_Server_API/tests/lint/test_platform_admin_roles_single_definition.py
docker exec tldw_postgres_test psql -U tldw_user -d postgres -Atc "select count(*) from pg_roles where rolname like 'fva_endpoint_%'"
```

Expected:
- `22 passed`. An exit code of `-6` after `22 passed` is the known at-exit abort; rerun once (Global Constraints);
- `1308 passed, 1 skipped` with Task 1 in (the r3 prototype gave `1307 passed, 1 failed, 1 skipped`, 97 s; the failure is the Hypothesis test below, which passed alone; the skip is spaCy). Under A-uncapped it is 1,307, and under B 1,304 (no Task 1 tests);
- `8 passed`;
- `0`: no test role is left behind.

The Hypothesis test `test_claims_jobs_contracts.py::test_review_metrics_payload_inclusive_window_property` is known to fail `too_slow` under load, on dev too (TASK-13862's notes; it did once in r2), and to pass alone. Rerun it alone and record both results. Before attributing any other failure to this change, run it on unchanged `origin/dev`.

- [ ] **Step 5 (optional, for the reviewer): the mutation run.** `$PY <scratchpad>/fu2c_mutate_r2.py` from the worktree root. It restores each mutated file after its mutant. Expected: the table under "What was measured".

- [ ] **Step 6: commit**

```bash
git add tldw_Server_API/app/core/Claims_Extraction/claims_service.py tldw_Server_API/app/api/v1/endpoints/claims.py \
  tldw_Server_API/tests/Claims_Extraction/test_fva_verify_endpoint.py
git commit -m "fix(claims): POST /claims/verify/fva runs, for the caller or an admin's named user (TASK-13881)" \
  -m "The three constructor calls that made every request a 500 are fixed. user_id counts only for a platform admin, as rebuild_claims; the named user's media DB and content scope then decide the evidence, the vector store, the quota decision and the usage rows, on SQLite and PostgreSQL alike." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 3: a full bounded pool names the rejected call (TASK-13888)

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/bounded_daemon.py:304`.
- Test: `tldw_Server_API/tests/Chat/unit/test_async_stream_task_capacity.py` (`:920`, `:1042-1053`).

**Interfaces:** `await_bounded_daemon_with_timeout(...)` keeps its signature. A `DaemonCapacityError` it raises now reads `"<name> capacity is exhausted"`.

- [ ] **Step 1: strengthen the existing capacity assertion.** In `test_daemon_drain_after_timeout_rejects_until_exit_then_raises`, replace

```python
        with pytest.raises(DaemonCapacityError):
            await bounded_daemon.await_bounded_daemon_with_timeout(
                blocked_call,
                pool=pool,
                name="capacity-rejected",
```

with

```python
        # The rejection names the call, not the pool's streaming default (TASK-13888).
        with pytest.raises(DaemonCapacityError, match="^capacity-rejected capacity is exhausted$"):
            await bounded_daemon.await_bounded_daemon_with_timeout(
                blocked_call,
                pool=pool,
                name="capacity-rejected",
```

That test already fills a one-slot pool and calls the helper a second time, so it is AC #1's "full pool reached through `await_bounded_daemon_with_timeout`".

- [ ] **Step 2: RED**

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q \
  "tldw_Server_API/tests/Chat/unit/test_async_stream_task_capacity.py::test_daemon_drain_after_timeout_rejects_until_exit_then_raises"
```

Expected: `1 failed`, `- Streaming worker capacity is exhausted` / `+ capacity-rejected capacity is exhausted` (measured).

- [ ] **Step 3: implement.** In `await_bounded_daemon_with_timeout`, replace

```python
        pool.start(invoke, name=name, released_event=released)
    except BaseException:
        released.set()
        raise

    loop_result = asyncio.wrap_future(result)
```

with

```python
        pool.start(invoke, name=name, released_event=released, exhaustion_message=f"{name} capacity is exhausted")
    except BaseException:
        released.set()
        raise

    loop_result = asyncio.wrap_future(result)
```

The anchor includes `loop_result`, which only the async helper has; `run_bounded_daemon_with_timeout` (`:237`) stays as it is.

Then let the one test pool that overrides `start` and is reached through this helper forward the argument. In `test_async_daemon_bridge_consumes_failure_when_cancelled_during_release_wait`, replace

```python
        def start(self, target, *, name, released_event=None):
            def gated_target() -> None:
                target()
                target_finished.set()
                allow_release.wait(timeout=2.0)

            return super().start(
                gated_target,
                name=name,
                released_event=released_event,
            )
```

with

```python
        def start(self, target, *, name, released_event=None, **kwargs):
            def gated_target() -> None:
                target()
                target_finished.set()
                allow_release.wait(timeout=2.0)

            return super().start(
                gated_target,
                name=name,
                released_event=released_event,
                **kwargs,
            )
```

`**kwargs`, not a required `exhaustion_message`: a fake that rejects the argument turns a regression into a hang (measured above), and the message is pinned by Step 1.

- [ ] **Step 4: GREEN, then the sweep.** The file list is every test file that names a symbol the change reaches. It is written to a new file, so the planner's evidence file stays as it is (M6):

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_async_stream_task_capacity.py
git grep -l -e "await_bounded_daemon_with_timeout" -e "BoundedDaemonPool" -e "DaemonCapacityError" \
  -e "SYNC_ADAPTER_CALL_POOL" -e "STREAM_DAEMON_POOL" -e "STREAM_CLEANUP_DAEMON_POOL" -e "TTS_JOB_RBAC_DAEMON_POOL" \
  -e "_run_bounded_claims_analyze" -e "_run_bounded_audio_sync_call" -e "_run_bounded_rag_analyze" \
  -- tldw_Server_API/tests tldw_Server_API/app/core/MCP_unified/tests > <scratchpad>/fu2c_13888_tests.r2.txt
wc -l < <scratchpad>/fu2c_13888_tests.r2.txt
TLDW_TEST_NO_DOCKER=1 xargs $PY -m pytest -p no:cacheprovider -q -rs -o timeout=120 -o timeout_method=signal < <scratchpad>/fu2c_13888_tests.r2.txt
```

If your environment refuses `xargs` (the review's worktree guard did), run the list through a short Python wrapper that calls pytest with the same arguments, as the review's `rv_fu2c/run_sweep.py` does.

Expected:
- `17 passed`;
- `35`;
- `2300 passed, 4 skipped` (measured twice, 869 s and 703 s), with the four skip reasons listed under "What was measured".

For any failure, run the same test on unchanged `origin/dev` before attributing it.

- [ ] **Step 5: commit**

```bash
git add tldw_Server_API/app/core/Chat/bounded_daemon.py tldw_Server_API/tests/Chat/unit/test_async_stream_task_capacity.py
git commit -m "fix(chat): a full bounded pool names the rejected call, not the streaming default (TASK-13888)" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 4: docs, the follow-up, close-out, PR

**Files:** `Docs/Operations/Usage_Quotas.md`, `Docs/Design/2026-10-09-llm-usage-accounting-design.md`, `Docs/RAG/FVA_Pipeline.md`. Backlog: TASK-13881, TASK-13888, and one new task.

- [ ] **Step 1: take the id lock and file the follow-up.**
  - Take the id lock: `mkdir <scratchpad>/id_lock`. If it exists, wait. Never break another session's lock.
  - Compute the highest id across `origin/dev` and every open PR. This is AGENTS.md's command, paging PRs with more than 100 files:

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
git fetch origin dev --quiet
{ git ls-tree -r --name-only origin/dev backlog/
  gh pr list --state open --limit 500 --json number,changedFiles,files \
    --jq '.[] | if .changedFiles > 100 then "PR#\(.number)" else .files[].path end' \
    | while IFS= read -r line; do
        if [[ $line == PR#* ]]; then
          gh api --paginate "repos/rmusser01/tldw_server/pulls/${line#PR#}/files" --jq '.[].filename'
        else
          printf '%s\n' "$line"
        fi
      done
} | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1
```

Use the next id for `TASK-<N>`, and write it literally in this step and in Step 5:

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "FVA adjudicator never uses the verifier's NLI pipeline" --id "TASK-<N>" -l claims,backend \
  -d "FVAPipeline.__init__ (Claims_Extraction/fva_pipeline.py near lines 164-166) reads the verifier's _nli when the pipeline is built, but HybridClaimVerifier loads it lazily on first use (claims_engine.py near lines 970-1008), so ClaimAdjudicator always gets nli_pipeline=None and assesses every document with the LLM: one stance call per supporting and per counter-evidence document. Listed in TASK-13881's description, outside its acceptance criteria." \
  --ac "With local NLI enabled, the adjudicator uses the verifier's loaded NLI pipeline, with a test" \
  --ac "Without local NLI, the adjudicator keeps the LLM path"
git add -A backlog/tasks
```

The lock is released in Step 7, after the PR is open.

- [ ] **Step 2: `Usage_Quotas.md` (AC #4).** In the HTTP endpoint bullet (line 95), replace

> prompt improvement (`POST /prompts/improve`), and manual character memory extraction (`POST /characters/{id}/memories/extract`). An endpoint that catches errors broadly

with

> prompt improvement (`POST /prompts/improve`), manual character memory extraction (`POST /characters/{id}/memories/extract`), and FVA claim verification (`POST /claims/verify/fva`) at its verify or adjudication call; an admin's `?user_id=` runs that request, and charges it, for the named user. An endpoint that catches errors broadly

In the claims bullet (line 103), delete the sentence "`POST /claims/verify/fva` answers 500 to every request until TASK-13881 is fixed. " together with its trailing space.

- [ ] **Step 3: the design note and the FVA page.**
  - **The design doc.** Append at the end of `Docs/Design/2026-10-09-llm-usage-accounting-design.md`, after the TASK-13864 revision's last bullet. Use the commit date if it is not 2026-10-22. The 2026-10-20 TASK-13862 revision stays as it was written.

```markdown

Revision of 2026-10-22, TASK-13881:

- **`POST /claims/verify/fva` runs,** and a refusal at its verify or adjudication call answers the 402. Its `user_id` query parameter counts only for a platform admin. The request then runs in that user's content scope, so its calls are checked and recorded for that user (§1). Anyone else's `user_id` is ignored.
```

  - **The FVA page, the anti-context section.** In `Docs/RAG/FVA_Pipeline.md`, after the bullet `- **Alternative queries**: "[different perspective on claim]"`, insert a blank line and:

```markdown
Each query runs the hybrid (full-text and vector) search of the user's media DB, at most 5 documents per query. `min_relevance_score` (0.3) applies to the full-text scores, min-max normalised within each query's results, before they are fused with the vector results: it drops the weakest full-text matches, and the best one always passes. Fused scores are not filtered again, and vector results have no floor. So no absolute relevance floor applies: the adjudicator's stance check decides which documents count, and a neutral one does not.

Adjudication assesses every counter-evidence document and, on the supporting side, the first 5 retrieved documents, the window the verifier draws its evidence from. Each assessment is one LLM call.
```

  - **The FVA page, the endpoint section.** Between the "### Endpoint" code block and "### Request Body", insert:

```markdown
### Query Parameter

- `user_id` (optional): the user to verify for. It counts only for a platform admin, as for the claims rebuild endpoints. The request then reads that user's media DB and vector store, and its LLM calls are quota-checked and recorded for that user. Anyone else's `user_id` is ignored, and the request runs for the caller.

An LLM call refused by the user's quota answers HTTP 402 with the `limit_exceeded` body (`Docs/Operations/Usage_Quotas.md`).
```

- [ ] **Step 4: the checks**

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
$PY Helper_Scripts/ci/test_quality_triage.py --enforce
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/CI/test_test_quality_ratchet.py tldw_Server_API/tests/Docs
$PY -m ruff check --output-format concise tldw_Server_API/app/core/Claims_Extraction/claims_service.py \
  tldw_Server_API/app/core/Claims_Extraction/anti_context_retriever.py tldw_Server_API/app/core/Claims_Extraction/fva_pipeline.py \
  tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py tldw_Server_API/tests/Claims_Extraction/test_adjudicator.py \
  tldw_Server_API/app/api/v1/endpoints/claims.py tldw_Server_API/app/core/Chat/bounded_daemon.py \
  tldw_Server_API/tests/Claims_Extraction/test_fva_verify_endpoint.py tldw_Server_API/tests/Claims_Extraction/test_anti_context_retriever.py \
  tldw_Server_API/tests/Chat/unit/test_async_stream_task_capacity.py
uvx bandit -q -r tldw_Server_API/app/core/Claims_Extraction/claims_service.py tldw_Server_API/app/core/Claims_Extraction/anti_context_retriever.py \
  tldw_Server_API/app/core/Claims_Extraction/fva_pipeline.py \
  tldw_Server_API/app/api/v1/endpoints/claims.py tldw_Server_API/app/core/Chat/bounded_daemon.py
```

Expected:
- the triage exits 0 with "OK — no offenses beyond the baseline" (measured on r3);
- `213 passed` (measured on r3, with the doc edits applied);
- ruff: dev's own findings only: `claims_service.py` I001, `test_anti_context_retriever.py` I001 ×2 and F401, `test_adjudicator.py` I001 ×2 and F401;
- bandit: exit 0, no issues, and 19 `nosec encountered` warnings for dev's existing comments (measured on r3).

If a guard or the environment refuses the triage or `uvx`, record the skip. The review's worktree guard refused both, so for the dry run these rest on the planner's logs.

- [ ] **Step 5: close TASK-13881.** Write the numbers your runs gave; these are the r3 prototype's. Write the Step 1 id in place of `TASK-<N>`.

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13881 \
  --append-notes "RED on origin/dev e06eeec68c: Task 1's tests 4 failed, 23 passed (the fused-score floor dropped every document: set() == {'1'} without a vector store, set() == {'1', '2'} with one; all 8 supporting documents adjudicated where the cap allows 5; the NEUTRAL pin passes on dev, as a pin); with Task 1 in, test_fva_verify_endpoint.py 22 failed (20 assert 500 == 200, 2 assert 500 == 402; TypeError: ClaimsJobContext got an unexpected keyword argument 'job_id', or MultiDatabaseRetriever missing db_paths without max_budget_usd). GREEN: 91 passed (anti-context, pipeline and adjudicator files) and 22 passed (the endpoint file: SQLite and PostgreSQL under a role that cannot bypass RLS, with and without a vector store, one real multi-user API-key case per backend and role). Mutants all killed: user_id honoured for anyone, ignored for everyone, no scope switch, a scope switch that keeps is_admin, no DB switch (SQLite), another user's vector store, usage rows by username, the fused-score floor put back, no FTS floor, no cap, NEUTRAL counter-evidence counted. tests/Claims + tests/Claims_Extraction: 1308 passed, 1 skipped (one run: 1307 and the known Hypothesis too_slow test, which passed alone). Test-quality ratchet exit 0; ruff: dev's own findings only; bandit clean." \
  --append-notes "A non-admin that passes another user's user_id gets 200 with its own results; the parameter is ignored for the media DB, vector store, quota decision and usage rows (AC #3, owner decision 2026-10-20). Single-user mode honours user_id: the single user has the admin role, as for rebuild_claims. AC #5 and AC #6 (owner Decision 1, A-capped): the anti-context retriever no longer filters fused scores; no absolute relevance floor applies, and the adjudicator's stance check decides; adjudication assesses the verifier's 5 supporting documents, so a request's worst case is 1,050 LLM calls (50 claims x 21), bounded by the request schema, not the quota (off by default, fail-open when on). Filed: TASK-<N> (the adjudicator never uses NLI). Not changed: on PostgreSQL an admin's own request reads every account's media, as any admin read does; an admin's user_id for an account with no media DB creates an empty one and answers 200, as rebuild_claims_fts does; max_budget_usd gates only the entry to falsification." \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "POST /claims/verify/fva runs: the three constructor calls that made every request a 500 are fixed, and FVA now finds counter-evidence (the anti-context retriever no longer drops every fused hybrid result), so claims are adjudicated, against the verifier's 5 supporting documents (at most 1,050 LLM calls per request). user_id counts only for a platform admin, as rebuild_claims does; an admin's switches the media DB and the content scope, so the evidence, the vector store, the quota decision and the claims usage rows all belong to the named user, on SQLite and PostgreSQL. Otherwise all four are the authenticated user's, by id. A quota refusal at the verify or adjudication call answers 402. Usage_Quotas.md lists the endpoint. Follow-up: TASK-<N> (the adjudicator never uses NLI)." \
  -s Done
git add -A backlog/tasks
```

For A-uncapped or B, use the variant in that option's section.

- [ ] **Step 6: close TASK-13888.** AC #2 is kept and checked: no caller or test matched the old text, so it is met without a change (M4).

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13888 \
  --append-notes "AC #2: git grep finds no caller or test matching 'Streaming worker capacity is exhausted' outside bounded_daemon.py and this task, so it is met without a change. What the change did break was a test pool whose start() override took no exhaustion_message: test_async_daemon_bridge_consumes_failure_when_cancelled_during_release_wait, the only one of 13 such overrides reached through await_bounded_daemon_with_timeout, then spun forever (Timeout >30 s with timeout_method=signal; the repo default, timeout_method=thread, ends the whole run). It now forwards **kwargs." \
  --append-notes "RED: test_daemon_drain_after_timeout_rejects_until_exit_then_raises 1 failed ('Streaming worker capacity is exhausted' != 'capacity-rejected capacity is exhausted'). GREEN: 17 passed. Sweep of the 35 test files that name the helper, the pools or their callers' helpers: 2300 passed, 4 skipped. run_bounded_daemon_with_timeout also omits the message, but its only caller (mlx_provider) maps DaemonCapacityError to its own 'MLX busy' error, so it is unchanged. Bandit clean." \
  --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "await_bounded_daemon_with_timeout passes exhaustion_message=f'{name} capacity is exhausted', so a full pool names the rejected call (for example 'claims-provider-analyze capacity is exhausted') for all 8 app callers instead of the streaming default. One test pool override now forwards the argument." \
  -s Done
git add -A backlog/tasks
```

- [ ] **Step 7: commit, PR, release the lock**

```bash
git add Docs/Operations/Usage_Quotas.md Docs/Design/2026-10-09-llm-usage-accounting-design.md Docs/RAG/FVA_Pipeline.md
git add -A backlog/tasks
git commit -m "docs(claims): /claims/verify/fva answers 402, admin-gates user_id and adjudicates; close TASK-13881 and TASK-13888; file the NLI follow-up" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push -u origin fix/fva-endpoint-and-capacity-message
gh pr create --base dev --title "fix(claims): /claims/verify/fva runs for the resolved user and adjudicates; a full pool names its call (TASK-13881, TASK-13888)" --body-file <scratchpad>/pr_body_fu2_claims.md
gh pr view --json files --jq '.files[].path' | grep "backlog/tasks/task-<N> "
```

When that `grep` prints the new task file, release the lock: `rmdir <scratchpad>/id_lock`.

The PR body needs:
- what changed and why;
- the measured 500 and its three causes, the fused-score floor, and the cap with its worst case (1,050 LLM calls per request, bounded by the request schema, not the quota);
- what a non-admin gets for another user's `user_id`;
- both backends' tests, the vector-store and real-auth variants, and the mutants;
- Decision 1 as taken, and the follow-up id;
- a "## Change summary" section. ADR-004: AI-authored PRs need the owner's own Change summary, and no recorded waiver names TASK-13881 or TASK-13888, so ask the owner for it;
- the line `🤖 Generated with [Claude Code](https://claude.com/claude-code)` at the end.

Then:
- request a clean review (`/superpowers:requesting-code-review` in a fresh subagent) and address it;
- check `gh variable get MERGE_QUEUE` and merge as `CLAUDE.md` says for that mode.

### If Decision 1 is A-uncapped

Everything stays as written, except:

- **Task 0:** leave out the second `--ac` line (AC #6). In the note, replace "AC #5 and AC #6 added (owner Decision 1, A-capped)" with "AC #5 added (owner Decision 1, A-uncapped)", and drop its last sentence (the cap).
- **Task 1:** leave out the `fva_pipeline.py` replacements and the cap test (`test_adjudication_assesses_only_the_verifiers_top_five_supporting_documents`). Keep the NEUTRAL pin. Step 2 then expects `3 failed, 23 passed`, and Step 4 `90 passed`. Drop the cap from the commit message.
- **Task 4 Step 3:** drop the FVA page's paragraph that starts "Adjudication assesses every counter-evidence document".
- **Task 4 Step 4:** drop `fva_pipeline.py` from the bandit and ruff lists.
- **Task 4 Step 5:** check AC #1 to #5. Replace the worst case in the notes and final summary with "up to 50 x (16 + top_k) LLM calls per request: 1,300 at the default top_k, 5,800 at 100".
- **The PR body:** state that worst case, and that the request schema, not the quota, bounds it.

### If Decision 1 is B

Skip Task 1 entirely (the cap only matters once adjudication runs). Then make these changes; everything else stays as written.

- **Task 0:** leave out both `--ac` lines, and the AC sentences of the note.
- **Task 2 Step 1:** the endpoint test file needs the floor lift that r1 used, or no 200 case can reach adjudication.
  - Add `from tldw_Server_API.app.core.Claims_Extraction.anti_context_retriever import AntiContextRetriever` to its imports.
  - Add this to the `fva` fixture, before `monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")`. Write the follow-up's id in place of `TASK-<F1>`:

```python
    real_anti_context = AntiContextRetriever.retrieve_anti_context

    async def _anti_context_without_score_floor(self, *args, **kwargs):
        # Hybrid (RRF) scores are at most 1/61, under the 0.3 floor, so no counter-evidence ever passes it
        # and the adjudicator never runs (follow-up TASK-<F1>). Retrieval itself stays real.
        self.config.min_relevance_score = 0.0
        return await real_anti_context(self, *args, **kwargs)

    monkeypatch.setattr(AntiContextRetriever, "retrieve_anti_context", _anti_context_without_score_floor)
```

  - Step 2's RED is then measured on unchanged dev source.
- **Task 4 Step 1:** file two tasks under the same lock.
  - **F1:** r1's text. Its title is "FVA anti-context retrieval drops every hybrid document, so the adjudicator never runs". Its description and ACs are in `<scratchpad>/plan_fu2_claims.r1.md`, Task 3 Step 1; add this plan's Task 1 as the candidate fix.
  - **F2:** as above.
- **Task 4 Step 3:**
  - In place of the anti-context paragraph, add "Known limit (TASK-<F1>): no counter-evidence passes the anti-context relevance filter today, so falsification finds none, no adjudication runs, and `final_status` is the verifier's." after the 402 sentence of the endpoint section.
  - Add F1's cost to the PR body: 3 discarded retrievals, and up to 3 embeddings, per falsified claim.
- **Task 4 Step 4:** drop `fva_pipeline.py` from the bandit and ruff lists.
- **Task 4 Step 5:** check AC #1 to #4 only. Name both follow-ups in the notes and the final summary. Worst case: 50 verify calls per request, plus up to 150 discarded anti-context retrievals (and as many query embeddings with a vector store).

## Follow-ups (filed in Task 4 Step 1)

- **`TASK-<N>`: the FVA adjudicator never uses the verifier's NLI pipeline.** It is in TASK-13881's description, not its ACs. With Task 1 in, it costs one LLM stance call per document.

Considered and not filed:
- **r1's F1 (the fused-score floor):** fixed in Task 1, under A-capped and A-uncapped.
- **The admin's own reads on PostgreSQL** (Review Focus 1): product-wide admin semantics, not this endpoint's.
- **`run_bounded_daemon_with_timeout`'s message:** never seen (mlx maps the error).
- **An admin's `user_id` for a missing account creating an empty DB:** the existing behaviour of the sibling rebuild endpoints.
- **`max_budget_usd` not reserving the stance calls, and no per-request call budget** (Review Focus 9): pre-existing FVA design. With the cap, the request schema bounds a request at 1,050 LLM calls.

## Out of scope

- Dropping or documenting `user_id` in the OpenAPI schema (the owner kept the parameter).
- A boundary quota check for FVA. Its LLM use depends on the claims and the evidence, so it is checked at the call (TASK-13638, TASK-13862).
- The team and org memberships of an admin's named user (Review Focus 2).
- A similarity floor for the vector list (Review Focus 10).
- A per-request LLM call budget for FVA (Review Focus 9).
