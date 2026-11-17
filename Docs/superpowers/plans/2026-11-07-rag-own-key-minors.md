# PR B's deferred minors: RAG LLM-stage drift test, reformulation canonicalization, a stream test's outbound call (TASK-15427) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Settle TASK-15427's three ACs (no outbound call from the provider-override stream test; a drift test for the RAG request switches; the reformulation's provider canonicalized and asserted), and file the follow-ups the owner asked to track.

**Architecture:**
- AC #1 is test-only: the stream test stubs `chat_service.perform_chat_api_call_async`, the answer call the standard pipeline makes, and asserts an empty `block_live_network()` record.
- AC #2 classifies all 98 boolean switches of the two RAG request schemas. Twelve make an LLM call besides the answer and the reformulation; they stay in `rag_unified._RAG_OTHER_LLM_STAGES`. Six that were listed make none of their own, so they leave the list. The test file declares the other 86 in `_NO_OTHER_LLM_CALL`, and a drift test fails on a switch in neither list, in both, or in neither schema.
- AC #3: `_rag_llm_providers` passes the reformulation's provider through `canonical_provider_name`, the call `resolve_generation_defaults` already makes for the answer. The refusal test then asserts which providers were looked up.

**Tech Stack:** Python 3.12, pytest, FastAPI TestClient, Pydantic v2 `model_fields`, backlog-py.

**Spec:** `backlog/tasks/task-15427 - PR-B-TASK-13956-deferred-minors-RAG-LLM-stage-drift-test-reformulation-provider-canonicalization-a-stream-test-s-outbound-Groq-call.md` (the ACs), and the planner brief `brief_plan_15427.md` (plans dir). Context: PR B's plan `Docs/superpowers/plans/2026-11-05-own-key-evaluations-documents-rag.md`, `final_review_13953.md`, `plan_13953.md` (V4), `final_review_13955.md` and `plan_byok_quota_exempt.folded.md` (plans dir).

**Measured on:** `origin/dev` at `cb70f0c5d134ac585c70fc30b5a99bf0ced98562` (`cb70f0c5d1`, 2026-11-07, "Merge pull request #3400"). All counts were measured in a detached worktree, which was reverted afterwards. `file:line` citations are for `cb70f0c5d1`.

**Prototype:** `proto_15427.patch` (plans dir): 3 files, +100/-23. It holds every code and test change in this plan, but not the plan copy or the task files. RED was measured by applying each task's tests to unchanged code, then GREEN with that task's code.

---

## What was verified

| Item | Evidence on `cb70f0c5d1` | Verdict |
|---|---|---|
| AC #1 | `test_rag_stream_parity.py::test_rag_streaming_generation_provider_override` looks up `api.groq.com`. **Measured:** wrapping its request in `block_live_network()` gives `AssertionError: assert [('dns', 'api.groq.com')] == []`. A stack probe found the caller on a `provider-sync-adapter` thread (`LLM_Calls/bounded_daemon.py:153` → `:176` → `usage_accounting.py:374` → `providers/groq_adapter.py:135`). It started from the stream's **retrieval prefetch**: `streaming_executor.py:391` `_retrieve_standard_documents` → `unified_pipeline.py:7255` `execute_generation_phase` → `generation.py:815`/`:451`/`:531` → `runtime_provider_call.py:74` → `gateway.py:1354`. The test stubs only `generate_streaming_response`, the second, streamed answer. The same prefetch answer makes four cases of `RAG/test_restricted_postgres_media_retrieval.py`'s two stream tests reach `api.openai.com` on `dev` (probe log; same stack). | **stub** the pipeline's answer call (Task 1); the prefetch's own answer is a product bug, **TASK-<SP>** (Decision 2) |
| AC #2 | `_RAG_OTHER_LLM_STAGES` (`rag_unified.py:215-223`) lists 18 switches. The schemas (`RAG/rag_schemas_unified.py`: `UnifiedRAGRequest`, `UnifiedBatchRequest`) have 98 `bool`/`Optional[bool]` fields: 49 + 2 `enable_*` and 47 others. Every one is classified below. 12 of the 18 make an LLM call; 6 do not (`enable_abstention`, `enable_knowledge_strips`, `enable_web_fallback`, `enable_query_rewriting_loop`, `adaptive_rerun_on_low_confidence`, `adaptive_advanced_rewrites`). No unlisted switch makes one. | drift test; **trim the 6** (Task 2, Review Focus 1) |
| AC #3 | The answer's provider is `resolve_generation_defaults(...)[0]`, which canonicalizes (`generation_defaults.py:24`, `canonical_provider_name`). The reformulation's is `settings.get("classifier_provider") or named or "openai"`, raw (`rag_unified.py:241`). `enforce_llm_quota` dedups with `dict.fromkeys` (`usage_quota_deps.py:121-126`, the dedup at `:123`) before `users_own_key_serves`, which canonicalizes itself (`:75`). So a spelling such as `OAI` asks twice, with the same decision. **Measured:** asked `['openai', 'OAI']`. `[history-on-a-server-key]` discards `asked` (`test_rag_agentic_api.py:324`). | canonicalize; assert `asked` (Task 3) |
| RG follow-up | **Reproduced, but already filed as TASK-15421** (`f9b792623d`, "A retried /chat/completions request commits nothing to the Resource Governor for its successful attempt"). Not re-filed; Task 0 appends the new evidence. The failed attempt's refund is `rg_governor.commit(handle, actuals={"tokens": 0})` (`endpoints/chat.py:5676-5685`), called from `_finalize_stream` when nothing was received (`Chat/chat_service.py:5583-5585`). It finalizes the handle: the memory governor pops it (`governor.py:861`), and a later commit on it returns at `:790-792`. The retry (`chat.py:5812-5900`, the same `_rg_handle_id` from `:4242-4287`) then commits through `_build_streaming_commit_cb` (`:6575`) as a no-op. Billing's `record_actual`/`apply_usage_delta` (`:6555-6560`) run once, for the retry. **The Redis governor behaves the same:** its commit returns when the handle record is gone (`governor_redis.py:2263-2271`) and deletes it at the end of the first commit (`:2404-2407`). **Measured:** (a) the endpoint, via a temporary copy of `test_streaming_oauth_retry_records_one_usage_row_for_the_successful_attempt` with a recording `MemoryResourceGovernor` on `app.state`, gave `[('reserve', 309), ('commit', 0, 'live'), ('commit', 19, 'finalized')]`; (b) both governors, Redis through the in-memory stub client: reserve 500, commit 0, commit 300, and a 701-token reserve is still allowed (refused in the control without the refund). | note on TASK-15421 (Task 0) |
| HP follow-up | **Reproduced.** `_own_key_before_the_hosted_plan` (`chat.py:419-426`) answers True for an own-key provider, and then no `LimitEnforcer` is entered (`:4415-4429`). With fallback on (`ENABLE_PROVIDER_FALLBACK`, `:517`, default False; `allow_provider_fallback_for_request`, `:3600-3603`), a failover runs on the fallback provider's key: the stream retry at `chat.py:5861-5900`, `execute_streaming_call` at `chat_service.py:4667`, `_execute_non_stream_call_impl` at `:6107-6125`. That key is usually the server's. At the call it is checked only against the per-user `limits.llm_*` (`Usage/llm_quota_gate.py:50-69`), and not at all when the user is within them, because the request is then marked prechecked. The hosted plan never checks it. Its row (`server_default`) joins the plan's sum at the next cache refresh: `BILLING_CACHE_TTL_SECONDS` is 60 s (`Billing/enforcement.py:26`), and `_OWN_KEY_EXCLUDED` (`:54`) leaves out only `user` rows. The hot cache gets nothing, because the enforcer is None (`chat.py:6555-6560`). This is code reading; it was not driven end to end (Not measured). | file **TASK-<HP>** (Task 0) |

### AC #2: every switch, classified

Paths: `UP` = `RAG/rag_service/unified_pipeline.py`, `EP` = `api/v1/endpoints/rag_unified.py`, `AE` = `RAG/rag_service/agentic_execution.py`, `AC` = `RAG/rag_service/agentic_chunker.py`, `SE` = `RAG/rag_service/streaming_executor.py`.

**A. An LLM call besides the answer and the reformulation; stays in `_RAG_OTHER_LLM_STAGES` (12).**

| Switch | Read at | The call |
|---|---|---|
| `enable_query_classification` | `UP:3317` | `query_classifier.py:286-317` (`perform_chat_api_call_async`); provider `UP:3320` |
| `enable_research_loop` | `UP:3451` | `research_agent.py:1315-1341` |
| `enable_document_grading` | `UP:5786` | `document_grader.py:174-183` (`sgl.analyze`) |
| `enable_fast_hallucination_check` | `UP:7513` | `quality_graders.py:728-756` (provider default `openai`) |
| `enable_utility_grading` | `UP:8569` | `quality_graders.py:486-495` |
| `enable_claims` | `UP:7550` | `UP:7569-7590` (`sgl.analyze`) |
| `enable_post_verification` | `UP:8142` | `post_generation_verifier.py:249-277` |
| `enable_multi_turn_synthesis` | `UP:7046` | the critique, `UP:7099-7101` (`api_name="openai"`) |
| `enable_suggestions` | `UP:7325` | `suggestion_generator.py:205-234`; provider `UP:7333` |
| `enable_faithfulness_eval` | `UP:8842` | `faithfulness.py:265-272`, through `UP:8873-8877` |
| `enable_evidence_accumulation` | `UP:5666` | `evidence_accumulator.py:423-431` (`AnswerGenerator`) |
| `enable_evidence_chains` | `UP:6820-6826` | `evidence_chains.py:377-387` (`enable_llm_extraction=True`, raw `generation_provider`) |

**B. Their call is one `_rag_llm_providers` names (3):** `enable_generation` (the answer, `EP:236-239`); `enable_query_reformulation` (`EP:240-241`, `UP:3413-3419`); `enable_reranking` (LLM only with `llm_scoring`/`two_tier`, `EP:245`; `UP:6057`, `UP:6146-6148`).

**C. Read only by the agentic strategy (16).** `_rag_llm_providers` refuses that strategy as non-standard (`EP:243`).

| Switches | Read at |
|---|---|
| `agentic_extractive_only`, `agentic_quote_spans`, `agentic_enable_tools`, `agentic_use_llm_planner`, `agentic_debug_trace`, `agentic_enable_query_decomposition`, `agentic_enable_semantic_within`, `agentic_enable_section_index`, `agentic_prefer_structural_anchors`, `agentic_enable_table_support`, `agentic_adaptive_budgets`, `agentic_enable_metrics` | `AE:246-273` (`_payload_bool`) |
| `agentic_enable_vlm_late_chunking`, `agentic_vlm_detect_tables_only`, `agentic_use_provider_embeddings_within` | `AC:393`, `AC:432`/`:455`, `AC:504` and `AE:750` |
| `explain_only` | `AC:660`; `SE:547` (agentic prefetch) |

**D. Read only inside a listed stage (13).**

| Switch | Read at | Inside |
|---|---|---|
| `enable_image_search`, `enable_video_search` | `UP:3589`; `research_agent.py:896-898` | the research loop (`UP:3451`) |
| `enable_research_action_dedup` | `UP:3522` | the research loop |
| `enable_research_progress` | `UP:3493`; `SE:383` (a progress callback only) | the research loop |
| `search_url_scraping` | `UP:3499`, `UP:3521` | the research loop |
| `grading_fallback_to_score` | `UP:5794`, `UP:5966` | document grading |
| `enable_query_rewriting_loop` **(was listed)** | `UP:5834` (in grading); `UP:5877-5881` (needs grading's `low_relevance_detected`) | document grading. The rewriter is the rule-based `QueryRewriter` (`query_features.py`, no LLM import). |
| `doc_only_verification`, `generate_verification_report` | `UP:7824`, `UP:7935` | claims (`UP:7550`) |
| `adaptive_advanced_rewrites` **(was listed)**, `adaptive_rerun_on_low_confidence` **(was listed)**, `adaptive_rerun_bypass_cache`, `adaptive_rerun_include_generation` | `UP:8148`, `UP:8249`, `UP:8273`, `UP:8309` | post-verification (`UP:8142`) |

**E. No LLM call of their own (54).**

| Switch | Read at | Why no LLM call |
|---|---|---|
| `enable_abstention` **(was listed)** | `UP:7424` (only read) | `UP:7438-7451` replaces the answer with a fixed sentence |
| `enable_knowledge_strips` **(was listed)** | `UP:6748-6756` | passes `use_llm_grading=False`; the only LLM path is gated on it (`knowledge_strips.py:593-604`) |
| `enable_web_fallback` **(was listed)** | `UP:6435-6465` | `web_fallback.py:358-363` builds `WebFallbackConfig(engine, result_count)` (`subquery_generation=False`, `:33`); `WebSearch_APIs.py:571-574` is the only LLM call in `generate_and_search`, and it is gated on that. A web search, not an LLM call. |
| `enable_structured_response` | `UP:6997` | the answer's prompt |
| `enable_query_decomposition` | `UP:4543-4570` | `AE:719-728` (regex split) |
| `enable_intent_routing`, `enable_dynamic_granularity` | `UP:3853`, `UP:3889` | `QueryRouter`, `GranularityRouter` (rule-based) |
| `enable_pre_retrieval_clarification` | `UP:3163-3173` | no `llm_call` is passed, so `clarification_gate.py:59-75` is heuristic |
| `enable_numeric_fidelity` | `UP:8001`; the retry `UP:8081-8092` | `AnswerGenerator(provider=generation_provider)` resolves as the answer does (`generation.py:782`): the answer's provider |
| `strict_extractive`, `require_hard_citations` | `UP:6955`; `UP:3683`, `UP:3699` | extractive assembly; heuristic citation check |
| `expand_query`, `spell_check` | `UP:3742-3790`; `UP:3205` | `query_expansion.py:546` and `QueryRewriter` (rule-based); `quick_wins.py:800` |
| `enable_discussion_search` | `UP:3242` | retrieval |
| `enable_multi_vector_passages`, `mv_flatten_to_spans` | `UP:4908`, `UP:4939` | spans and embeddings |
| `enable_text_late_chunking` | `database_retrievers.py:1277-1280`; `UP:2826` | chunking |
| `enable_enhanced_chunking`, `include_page_numbers`, `enable_batch` | forwarded to the rerun only (`UP:8291`, `UP:8307`); `UP:2230` signature only | capabilities lists (`EP:2572`, `EP:2531`, `EP:2576`) |
| `enable_vlm_late_chunking`, `vlm_detect_tables_only` | `UP:5520`, `UP:5583` | VLM backends `docling_vlm`, `hf_table_transformer` (local models) |
| `enable_table_processing`, `enable_numeric_table_boost` | `UP:5483`, `UP:4998` | `table_serialization.py:494` |
| `enable_parent_expansion`, `include_parent_document`, `include_sibling_chunks` | `UP:3906`, `UP:3910`, `UP:6606` | retrieval context |
| `enable_citations`, `enable_chunk_citations` | `UP:6670`, `UP:6695` | `CitationGenerator`; its LLM path is evidence chains' (`UP:6687`) |
| `enable_security_filter`, `detect_pii`, `redact_pii`, `content_filter`, `enable_content_policy_filter`, `enable_injection_filter`, `enable_html_sanitizer` | `UP:5342`, `UP:5362`, `UP:5401`, `security_filters.py:780`, `UP:5310`, `UP:5213`, `UP:5291` | regex filters |
| `enable_cache`, `adaptive_cache`, `use_embedding_cache`, `use_connection_pool` | `UP:2690-2703`, `UP:2853`; the last two read nowhere (`EP:1268-1269`, capabilities) | cache |
| `enable_resilience`, `circuit_breaker` | `UP:2907`, `UP:2931` | retries and breaker |
| `enable_checkpoint` | `EP:1774` | batch checkpointing |
| `collect_feedback`, `apply_feedback_boost` | `UP:8592`; `UP:5775`, `UP:8601` | feedback store |
| `highlight_results`, `highlight_query_terms`, `include_rerank_debug_documents`, `debug_mode`, `track_cost` | `UP:8647`, `UP:8653`, `UP:6225`, `EP:1615`, `UP:8668` | formatting, telemetry |
| `enable_analytics`, `enable_monitoring`, `enable_observability`, `enable_performance_analysis` | `analytics_system.py:882-884`, `EP:1613`, `UP:4171`, `UP:8756` | telemetry (`PerformanceMonitor.analyze` reads timings) |

The method: `git grep -w` of each name over `app/core/RAG` and `rag_unified.py` (scratch script `uses.py`). Each read site was then followed to the code it gates. An LLM call means `perform_chat_api_call*`, `sgl.analyze` or `AnswerGenerator`. None of the RAG modules that the E rows lean on imports one: `query_features`, `quick_wins`, `security_filters`, `observability`, `analytics_system`, `feedback_system`, `table_serialization`, `query_expansion` and the VLM backends (grep).

## Decisions

1. **ADR-004 Change summary.** This PR is AI-authored. The recorded waivers do not name TASK-15427.
   - **Recommendation:** ask the owner to extend the waiver to this PR, as for each earlier follow-up. Otherwise the owner writes the `## Change summary` before merge.
2. **TASK-<SP>, a third follow-up, is outside the brief.** Every `/rag/search/stream` request runs a full non-streaming answer in its retrieval step, plus any post-answer stage it turns on (claims, post-verification, suggestions), and throws it away (What was verified, AC #1). So every streaming RAG request pays for two answers and waits for the first.
   - **Priority:** user-facing. Every Knowledge QA question pays for two answers and waits for the first before streaming, since 2026-04-24. Recommend it next after this PR; ask the owner for its priority.
   - **Recommendation:** file it in Task 0 (drafted below), and fix it in its own PR, not this one. Its fix also needs the effective query to reach the stream (the prefetch syncs it into the shared request, `unified_pipeline.py:2507-2518`; the stream reads it, `streaming_executor.py:696`). Removing the prefetch's answer also turns off the stream's default pre-retrieval clarification, which follows the effective `enable_generation` (`UP:3163-3167`). That needs its own test and review.
   - If the owner says no, skip Task 0 Step 3(b). Leave `TASK-<SP>` out of the notes and the final summary, and write the Task 1 stub's docstring without it: `(the stream's retrieval makes one)`.
3. **TASK-<HP>'s AC #1 is the owner's choice**, as the brief says, and is left open in the task. Recommendation for when it is picked up: **refuse the fallback**. Set `allow_provider_fallback_for_request = False` once `own_key is True` (`chat.py:4415`). It is a few lines, fails closed (`own_key` is True only when the hosted plan applies), and the user asked for the provider their key serves. It covers all four failover paths, including the pre-dispatch health-check switch (`chat.py:5246-5276`). Trade-off: it removes failover for every own-key request under a hosted plan, including orgs within their plan; keeping it for them needs a plan check at the boundary. Checking the plan at the call would need a billing hook in the LLM call layer. No hosted deployment exists, so it stays low priority.

Not owner questions (rulings for the reviewer, each in Review Focus):
- Trimming the six switches (Review Focus 1).
- The drift test covers the boolean switches only (Review Focus 2).
- TASK-15421 gets a note, not a duplicate task.

---

## Global Constraints

Carried verbatim from the brief:

- Tests make no real outbound network requests; stub the provider adapter or its transport.
- Function-size ratchet `tldw_Server_API/tests/lint/function_size_baseline.json`: frozen functions must not grow.
- Backlog:
  - edit tasks with backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`);
  - `--append-notes`, never `--notes`; stage with `git add -A backlog/tasks`; never hand-edit task files;
  - a new task's explicit id comes from `zsh <plans dir>/task_next.sh` (prints the HIGHEST id in use; the new id is that plus one), after `mkdir <scratchpad>/id_lock`; release it once the PR's file list shows the task. The plan writes ids as placeholders (`TASK-<RG>`, `TASK-<HP>`) filled at Task 0. Do NOT allocate ids yourself now.
- Branch from `origin/dev`; the PR targets `dev`. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- Redis: never FLUSHALL or SCRIPT FLUSH. PostgreSQL only through the repo fixtures, with a session-unique `TEST_DB_NAME`; drop only your own objects.
- Mirrored docs (`Docs/Published`) need `bash Helper_Scripts/refresh_docs_published.sh` in the same PR.
- CI checks before the PR, in the last task:
  - `tests/Docs`, `tests/lint`, `tests/CI`, and backlog-py's tests;
  - shard coverage (`Helper_Scripts/ci/check_shard_coverage.py`) and `Helper_Scripts/ci/test_quality_triage.py --enforce`;
  - the route-auth ratchet and the OpenAPI export check;
  - Bandit (`/opt/homebrew/bin/uvx bandit -ll`) on touched source;
  - every test file that names a touched symbol (git grep), with a baseline on `origin/dev`.
  - A file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` whose lines shift fails backend-required (regenerate with `--write-baseline` in the same PR); check every file you touch.
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` exit 0.
- Agent shells don't keep the venv `PATH`: prefix commands that spawn `python` with `PATH="$(dirname $PY):$PATH"`.

Plan-specific:

- **Ids.** The RG follow-up is already TASK-15421, so `TASK-<RG>` is not allocated. Task 0 allocates `TASK-<HP>` = printed + 1 and `TASK-<SP>` = printed + 2 under one lock. Write the real ids wherever this plan says `<HP>`/`<SP>`. Before the last commit, `git grep -n -E "TASK-<(HP|SP|RG)>" -- ':!Docs/superpowers/plans'` must print nothing.
- **`$PY`** is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Write it out, or set it in each call, because shell variables do not survive between agent calls. `<scratchpad>` is the executing session's scratchpad. `<plans dir>` is `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`.
- **`task_next.sh` does not take the lock.** Take it yourself with `mkdir <scratchpad>/id_lock`. If the directory exists, wait; never break another session's lock. Run the script from the repository root with `zsh` (it is a zsh script).
- **TDD.** Write each task's tests first, run them, and see them fail for the stated reason. Record the real counts.
- **No network: run every pytest command that runs RAG tests under the probe plugin.** `<pytest>` below stands for `PATH="$(dirname $PY):$PATH" PYTHONPATH=<plans dir> NETPROBE_LOG=<scratchpad>/netprobe.log TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -p netprobe_15427`; check that the log is empty (or holds only the base lines named here) after each run. `<plans dir>/netprobe_15427.py` runs each test inside the repo's `block_live_network()` and appends any blocked attempt to `$NETPROBE_LOG`. Add `PYTHONPATH=<plans dir> NETPROBE_LOG=<scratchpad>/netprobe_<label>.log` to the environment and `-p netprobe_15427` to the pytest arguments. On `dev`, five tests in the focused and wider sets try to reach a provider: this AC's test (`api.groq.com`), and four cases of `RAG/test_restricted_postgres_media_retrieval.py`'s two stream tests (`api.openai.com`, the same discarded prefetch answer, TASK-<SP>). Without the plugin, they make real DNS lookups. With it, the base logs those 5 lines and the final code logs 4: everything but this AC's.
- **Probe ceilings.** The plugin works per test window (checked once beside a session-wide blocker: nothing outside test windows). An attempt from a thread that outlives its test runs unblocked after teardown; a test with its own `block_live_network()` hides its attempts from the log (the inner patch records and raises first); a server subprocess (`server_e2e_tests`, `e2e/`) is never probed.
- **Services.** Nothing here needs Redis. The RG evidence used the in-memory Redis stub client, never a server. The wider set's PostgreSQL cases use `pg_temp_db`, which creates and drops a `tldw_test_<uuid>` database per test (`tests/_plugins/postgres.py:207`). The script still sets a session-unique `TEST_DB_NAME`.
- **Git.** Stage named files. `git diff --cached --check` must print nothing before each commit.
- **Go cache.** Test runs can leave a read-only Go module cache under `tldw_Server_API/Config_Files/acp_runner_home`. Clean it with `chmod -R u+w tldw_Server_API/Config_Files/acp_runner_home && git clean -ffdx -- tldw_Server_API/Config_Files/acp_runner_home`. Never `rm -rf` the directory: it holds a tracked file. The last `git status --short` must be empty.
- **Scope baseline and function sizes.** No touched file is in `Helper_Scripts/ci/scope_predicate_baseline.txt` (grep: nothing). `_rag_llm_providers` is not in `function_size_baseline.json`; the only `rag_unified` entry is `get_capabilities`, which is untouched.
- **Docs.** No doc changes. `Docs/Operations/Usage_Quotas.md:111` names the list by name ("the rest of `_RAG_OTHER_LLM_STAGES`"), and ADR-083's `:56` ("a RAG request that turns on another LLM stage") stays true. Neither is reworded, so nothing is mirrored.

## Review Focus

1. **Six switches stop refusing an over-limit user on their own key:** `enable_abstention`, `enable_knowledge_strips`, `enable_web_fallback`, `enable_query_rewriting_loop`, `adaptive_rerun_on_low_confidence` and `adaptive_advanced_rewrites`. Each makes no LLM call of its own, or makes it only inside a stage that stays listed (classification rows D and E). The WebUI sends them: Knowledge QA's web-fallback toggle (`apps/packages/ui/.../KnowledgeQA/SearchBar.tsx`), and the Knowledge settings' abstention and low-confidence rerun toggles (`GenerationSection.tsx:184`, `AdvancedSection.tsx:485`). A wrong trim cannot spend unchecked, because the request is admitted unmarked and every call is still checked at its own call where the call-site gate runs (`rag_unified.py:232`; `llm_quota_gate.py`: `check_user_sync` lets a synchronous call through unchecked on the accounting loop's own thread or with no accounting loop, `:97-107`; none of the six makes a call). The reviewer should confirm each of the six against its cited lines. Pinned by the new `[stages-without-an-llm-call]` admit case.
2. **The drift test covers boolean switches only:** all 98 `bool`/`Optional[bool]` fields of both schemas, which includes the two non-`enable_*` entries the list had. A string-valued mode that adds an LLM call is not caught: `strategy`, `search_depth_mode` and `reranking_strategy`, which `_rag_llm_providers` checks by value, or a new `*_behavior` value. That is the known ceiling. The test also fails on a stale name in either list (`(listed | declared) - switches`), and on a switch in both lists.
3. **Canonicalization changes no decision.** `users_own_key_serves` canonicalizes too (`usage_quota_deps.py:75`), so the only effect is one lookup fewer for an aliased reformulation provider. A blank `classifier_provider` (`"  "`) becomes `""`, so `all(names)` is False and the request is refused without a lookup. Before, `"  "` was looked up and answered False. Both refuse.
4. **The stream test asserts no network, not that the stub is reached.** That is deliberate. Today the stub serves the prefetch's discarded answer (TASK-<SP>), and asserting that call would pin that bug. Once TASK-<SP> lands, the stub is unused and its AC removes it.
5. **The other refusal cases assert `asked == []`.** A stage with an unnamed provider appends None, and `enforce_llm_quota` stops at `all(names)` before any lookup (`usage_quota_deps.py:116`). This pins "no lookup when the answer is refused anyway". If that order is ever changed on purpose, these cases change with it.

## Files

| File | Task | Change |
|---|---|---|
| `tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py` | 1 | `test_rag_streaming_generation_provider_override`: stub `perform_chat_api_call_async`, wrap the request in `block_live_network()`, assert no attempt |
| `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` | 2, 3 | `_RAG_OTHER_LLM_STAGES` loses 6 names, and its comment points to the drift test (2). `_rag_llm_providers` canonicalizes the reformulation's provider (3). |
| `tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py` | 2, 3 | `_NO_OTHER_LLM_CALL` and the drift test; admit cases `[stages-without-an-llm-call]` (2) and `[reformulated-on-an-alias-of-the-answers-provider]` (3); the admit test asserts `asked` first; the refusal test asserts `asked` per case (3) |
| `Docs/superpowers/plans/2026-11-07-rag-own-key-minors.md`; `backlog/tasks/task-<HP> - ...md`, `task-<SP> - ...md`, `task-15421 - ...md` (notes), `task-15427 - ...md` | 0, 4 | this plan; the tasks |

## Test commands

- **Focused set** (the two test files this plan edits):
  `PATH="$(dirname $PY):$PATH" PYTHONPATH=<plans dir> NETPROBE_LOG=<scratchpad>/netprobe_focused_<label>.log TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -p netprobe_15427 -q tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py`
  Measured:
  - On `cb70f0c5d1`: `40 passed, 1 skipped`. The log has one line, `...test_rag_streaming_generation_provider_override [('dns', 'api.groq.com')]`.
  - With the prototype: `43 passed, 1 skipped`, about 2-4 minutes, and the log is empty. That is +3 tests: the drift test and two admit cases.
  - The prototype also passed under `<plans dir>/netblock_13956.py` (TASK-13956's connect blocker): `netblock: 0 blocked outbound connect(s)`.
- **Wider set:** every test file that names `_RAG_OTHER_LLM_STAGES`, `_rag_llm_providers`, `rag_unified` or `/api/v1/rag/`. Save it as `<scratchpad>/wider_15427.sh` and run `zsh <scratchpad>/wider_15427.sh <label>` from the worktree root:

```zsh
#!/bin/zsh
S=<scratchpad>
P=<plans dir>
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
L=${1:-run}
git grep -l -E "_RAG_OTHER_LLM_STAGES|_rag_llm_providers|rag_unified|/api/v1/rag/" \
  -- 'tldw_Server_API/tests/*.py' 'tldw_Server_API/app/*/tests/*.py' | grep '/test_[^/]*\.py$' | sort -u > $S/wider_$L.txt
echo "files: $(wc -l < $S/wider_$L.txt)"
rm -f $S/netprobe_wider_$L.log
PATH="$(dirname $PY):$PATH" TEST_DB_NAME=tldw_test_15427 PYTHONPATH=$P NETPROBE_LOG=$S/netprobe_wider_$L.log TLDW_TEST_NO_DOCKER=1 \
  $PY -m pytest -p no:cacheprovider -p netprobe_15427 -q -n 8 --dist loadfile -rf $(cat $S/wider_$L.txt) > $S/wider_$L.out 2>&1
echo "wider: $(grep -E '^=+ .*(passed|failed)' $S/wider_$L.out | tail -1)"
grep -E '^FAILED|^ERROR' $S/wider_$L.out | cut -c1-220
echo "netprobe attempts: $(cat $S/netprobe_wider_$L.log 2>/dev/null | wc -l)"
cut -c1-220 $S/netprobe_wider_$L.log 2>/dev/null
```

  Measured: 57 files, about 6-10 minutes under load. See the Wider-set results under Task 4 Step 2.

---

## Task 0: Ids, follow-up tasks, branch, draft PR, baseline

**Files:** `backlog/tasks/` (two new tasks; notes on TASK-15421 and TASK-15427); `Docs/superpowers/plans/2026-11-07-rag-own-key-minors.md`.

- [ ] **Step 1: Allocate the ids under the lock.**
  - `mkdir <scratchpad>/id_lock` (see Global Constraints).
  - `git fetch origin dev --quiet`, then `zsh <plans dir>/task_next.sh` from the repository root. `<HP>` is the printed value + 1, and `<SP>` is the printed value + 2.
  - Check that `ls backlog/tasks backlog/completed backlog/archive/tasks | grep -E "^task-(<HP>|<SP>) "` prints nothing.
  - `git switch -c fix/rag-own-key-minors-15427 origin/dev`.
- [ ] **Step 2: Note the RG evidence on TASK-15421** (no new task; it already holds this finding):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-15421 \
  --append-notes "Re-verified by TASK-15427 on cb70f0c5d1 (2026-11-07). Endpoint: a copy of test_streaming_oauth_retry_records_one_usage_row_for_the_successful_attempt with a recording MemoryResourceGovernor on app.state recorded reserve 309, commit(tokens=0) on the live handle (the failed attempt's rg_refund_cb, endpoints/chat.py:5676-5685, from _finalize_stream's nothing-received branch, Chat/chat_service.py:5583-5585), then commit(tokens=19) on the finalized handle (the retry's _build_streaming_commit_cb, chat.py:6575): a no-op (governor.py:790-792). The Redis governor behaves the same: RedisResourceGovernor.commit returns when the handle record is gone (governor_redis.py:2263-2271) and deletes it at the end of the first commit (:2404-2407). Measured on both governors (Redis through the in-memory stub client): reserve 500 tokens, commit 0, commit 300, and a 701-token reserve in the same minute is still allowed (refused without the first commit). The chat handle reserves tokens only (chat.py:4242-4247), so no stream lease is involved. AC #1's test should run against both governors (the Redis one through the in-memory stub client)."
```

- [ ] **Step 3: File the two follow-ups** (write the ids literally).

  (a) **TASK-<HP>:**

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "The hosted plan never checks a server-key failover reply inside an own-key /chat/completions request" \
  --id TASK-<HP> -l quotas,billing,backend \
  -d "Follow-up of TASK-13955 (ADR-083), from its final review (declined to judge) and its plan's Task 3 ruling, an owner-approved trade-off. /chat/completions asks users_own_key_serves once, before the hosted plan's pre-check (_own_key_before_the_hosted_plan, endpoints/chat.py:419-426 on cb70f0c5d1); when the provider runs on the user's own key no LimitEnforcer is entered (:4415-4429), so the request is neither checked against the org's plan (llm_tokens_month) nor added to its hot cache. With provider fallback on (ENABLE_PROVIDER_FALLBACK, :517, default off; allow_provider_fallback_for_request, :3600-3603), a failover runs on the fallback provider's key, usually the server's, by four paths: the pre-dispatch health-check switch (chat.py:5246-5276, which swaps selected_provider for a healthy one after the own-key decision at :4415), the stream retry loop (:5861-5900), execute_streaming_call (Chat/chat_service.py:4667) and _execute_non_stream_call_impl (:6107-6125). That call is checked at the call only against the per-user limits.llm_* (Usage/llm_quota_gate.py:50-69; not at all when the user is within them, because the request is then marked prechecked), never against the hosted plan, so an org over its plan gets a server-key reply. Its usage row (server_default) joins the plan's sum at the next billing cache refresh (BILLING_CACHE_TTL_SECONDS, 60 s, Billing/enforcement.py:26; _OWN_KEY_EXCLUDED, :54, leaves out user rows only); the hot cache gets nothing, because the enforcer is None (_build_streaming_commit_cb, chat.py:6555-6560). No hosted deployment exists, so this is low priority. Options: check the plan at the call for a server-key call inside an own-key request; refuse provider fallback for an own-key request when the hosted plan applies (set allow_provider_fallback_for_request False once own_key is True, chat.py:4415; recommended by TASK-15427's plan); or accept and document. Trade-off of the refusal: own_key is True only when the hosted plan applies (:419-426), so it is scoped and fail-closed (every later failover reads allow_provider_fallback_for_request: :5246, :5661, :5865, :5965), but it removes failover for every own-key request under a hosted plan, including orgs within their plan. Keeping failover for orgs within their plan needs a plan check at the boundary." \
  --ac "The owner chooses, and the notes and ADR-083 record it: check the hosted plan at the call, refuse provider fallback for an own-key request when the plan applies, or accept and document the gap" \
  --ac "If a check or a refusal is chosen: an own-key /chat/completions request from an org over its plan gets no server-key failover reply, stream and non-stream, by any of the four failover paths, with tests; if the refusal is chosen, the notes and docs say that own-key requests under a hosted plan never fail over (if the check at the call is chosen, an org within its plan keeps failover)" \
  --ac "Docs/Operations/Usage_Quotas.md and ADR-083 (and its Docs/Published mirror) state the behaviour"
```

  (b) **TASK-<SP>** (Decision 2; skip this step if the owner declines):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "The streaming RAG search's retrieval step generates a full answer and discards it" \
  --id TASK-<SP> -l rag,backend,bug \
  -d "User-facing: every Knowledge QA question (the WebUI streams) pays for two answers, and waits for the first before its stream starts, since 2026-04-24; recommended next after TASK-15427. Found by TASK-15427 (the outbound Groq call in test_rag_streaming_generation_provider_override). /api/v1/rag/search/stream retrieves through the standard pipeline before it streams the answer. _retrieve_standard_documents sets kwargs['enable_generation'] = False (RAG/rag_service/streaming_executor.py:381 on cb70f0c5d1) but passes the endpoint's resolved_request (:379; rag_unified.py:2340), whose payload says enable_generation True (the stream route requires it, rag_unified.py:2308-2309), and unified_rag_pipeline prefers the payload (_resolve_effective_enable_generation, unified_pipeline.py:2390-2393, re-read at :6900; since cd184c9fb9, 2026-04-24). So the retrieval step runs the full non-streaming answer (execute_generation_phase, unified_pipeline.py:7255, to the provider call at generation.py:531) and any post-answer stage the request turns on (claims, post-verification, suggestions), keeps only the documents (streaming_executor.py:391-431), and _stream_generation_events (:685-722) then generates the answer again. Every streaming RAG request pays for two answers and waits for the first before its stream starts; on the user's own key their provider bills both. Measured by a socket probe: the prefetch's answer reached api.groq.com from a provider-sync-adapter thread, and on dev four cases of RAG/test_restricted_postgres_media_retrieval.py's two stream tests reach api.openai.com the same way (stack through streaming_executor.py:391 and unified_pipeline.py:7255). Fix sketch: give the prefetch a request with enable_generation False. Caveat: a plain copy (dataclasses.replace) is wrong. The prefetch's pipeline writes the effective query (spell correction, classification, chat-history reformulation) back into the shared resolved request (_sync_effective_query, unified_pipeline.py:2507-2518: resolved_request.query and payload['query']) and the streamed answer reads resolved_request.query (_stream_generation_events, streaming_executor.py:696; wired at :841-842), so with a copy the stream would answer the raw follow-up (what about it?). The copy must carry the effective query back, or the override must not replace the object. Knowledge QA takes this route (chat-rag.ts:592, knowledgeQaClient.ts:51-52). With suggestions on, the prefetch also makes a suggestions call whose result it discards. Pre-retrieval clarification defaults to the effective enable_generation (unified_pipeline.py:3163-3167), so set enable_pre_retrieval_clarification explicitly when the request leaves it None, to keep the stream's clarification." \
  --ac "A streaming RAG search makes one answer call, the streamed one: its retrieval step runs no generation and no post-answer stage, with a test that counts the provider calls" \
  --ac "The streamed answer still uses the effective query (chat-history reformulation, spell correction), with a test" \
  --ac "The stream still runs pre-retrieval clarification when neither the request nor its profile sets enable_pre_retrieval_clarification, as today, with a test" \
  --ac "test_rag_streaming_generation_provider_override passes its socket probe without its provider stub (the stub is removed), and test_restricted_postgres_media_retrieval.py's stream tests reach no provider"
```

- [ ] **Step 4: Start TASK-15427:** `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-15427 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-11-07-rag-own-key-minors.md. Follow-ups: the RG under-commit is TASK-15421 (re-verified, Redis governor included; not re-filed); TASK-<HP> (hosted plan and failover); TASK-<SP> (the stream's discarded prefetch answer). Owner answers: Decision 1 (Change summary) <answer>; Decision 2 (TASK-<SP>) <answer>."`
- [ ] **Step 5: Plan copy, commit, draft PR.**
  - Copy this plan to `Docs/superpowers/plans/2026-11-07-rag-own-key-minors.md`, unchanged: it keeps the placeholders.
  - `git add Docs/superpowers/plans/2026-11-07-rag-own-key-minors.md && git add -A backlog/tasks && git diff --cached --check`. It must print nothing.
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` must exit 0.
  - Commit (message 0).
  - `git push -u origin fix/rag-own-key-minors-15427`.
  - `gh pr create --draft --base dev --title "fix(usage): PR B's deferred minors: RAG LLM-stage drift test, reformulation canonicalization, no outbound call in a stream test (TASK-15427)"`, with a one-line body naming the task and ending with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - Release `<scratchpad>/id_lock` once `gh pr view <n> --json files` lists both new task files.
- [ ] **Step 6: Baseline.** On the unchanged branch, run both sets under the probe plugin (measured). Record both in the ledger. Any later failure is compared with this baseline; never dismiss one as environment-only without it.
  - The focused set: `40 passed, 1 skipped`, and one log line, `...test_rag_streaming_generation_provider_override [('dns', 'api.groq.com')]`.
  - The wider set with label `base`: `3 failed, 1035 passed, 58 skipped, 2 errors`, and 5 log lines: the groq line, plus `RAG/test_restricted_postgres_media_retrieval.py::test_owner_retrieves_chunked_media_through_stream_endpoint[sqlite|postgresql]` and `::test_permission_first_auth_stream_without_preseeded_scope[sqlite-Rowan-True|postgresql-Rowan-True]`, each `[('dns', 'api.openai.com')]`.
    - The 3 failures are `e2e/test_rbac_multi_user.py::TestMultiUserIsolation::test_10_isolation_media_retrieval_and_search`, `test_11_isolation_embedding_jobs` and `test_12_self_virtual_key_access` (`Connection refused`).
    - The 2 errors are the setups of `server_e2e_tests/test_rag_pipeline_workflow.py::test_rag_pipeline_local_workflow` and `::test_rag_pipeline_external_embedding_workflow` (`Server 'server-e2e' failed health check`). These tests need a live server.
  - The e2e lifecycle helper can leave a uvicorn process and an untracked `server-server-e2e.pid` (with an ignored `server-server-e2e.log`) in the worktree (seen once). Stop that pid only if `lsof -p <pid> -a -d cwd` shows this worktree, then delete both files.

## Task 1: The provider-override stream test makes no outbound call (AC #1)

**Files:** Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py` (`test_rag_streaming_generation_provider_override`, `:161-215`).

**Interfaces:** Consumes `tldw_Server_API.tests.helpers.offline_network.block_live_network() -> Iterator[list[tuple[str, str]]]` (blocks and records non-loopback DNS and connects; `testserver` is loopback). Produces nothing.

- [ ] **Step 1: Write the failing probe.** In the test, add `from tldw_Server_API.tests.helpers.offline_network import block_live_network` after its `rag_ep` import. Replace its `with client_with_stream_overrides.stream(...) as resp:` block (`:208-210`) with this, and put `assert attempts == []` before `generation_config = captured["generation_config"]`:

```python
    with (
        block_live_network() as attempts,
        client_with_stream_overrides.stream("POST", "/api/v1/rag/search/stream", json=payload) as resp,
    ):
        assert resp.status_code == 200
        list(resp.iter_lines())

    assert attempts == []
```

- [ ] **Step 2: Run it.** `<pytest> -q "tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py::test_rag_streaming_generation_provider_override"`
  Expected: `1 failed` with `AssertionError: assert [('dns', 'api.groq.com')] == []` (measured, about 35 s).
- [ ] **Step 3: Stub the pipeline's answer call.** Add `from tldw_Server_API.app.core.Chat import chat_service` to the test's imports. `generation.py:500` imports `perform_chat_api_call_async` from that module at call time, so patching the attribute is enough. After `fake_generate_streaming_response`, add the stub and patch it with the two existing patches:

```python
    async def no_provider_call(**_kwargs: Any) -> dict[str, Any]:
        """The standard pipeline's answer call (the stream's retrieval makes one, TASK-<SP>): nothing leaves."""
        return {"choices": [{"message": {"content": "stub answer"}}]}
```

  `monkeypatch.setattr(chat_service, "perform_chat_api_call_async", no_provider_call)`. Do not assert on its calls (Review Focus 4).
- [ ] **Step 4: Run it.** The same command. Expected: `1 passed` (measured).
- [ ] **Step 5: Commit** (message 1): `git add tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py && git diff --cached --check`.

## Task 2: A drift test for every RAG request switch; six switches leave the list (AC #2)

**Files:** Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py:213-223`. Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py`.

**Interfaces:**
- Consumes `UnifiedRAGRequest` and `UnifiedBatchRequest` from `tldw_Server_API.app.api.v1.schemas.rag_schemas_unified`, the import the routes use.
- Produces `_RAG_OTHER_LLM_STAGES: tuple[str, ...]` with these 12 names, in this order: `enable_query_classification`, `enable_research_loop`, `enable_document_grading`, `enable_fast_hallucination_check`, `enable_utility_grading`, `enable_claims`, `enable_post_verification`, `enable_multi_turn_synthesis`, `enable_suggestions`, `enable_faithfulness_eval`, `enable_evidence_accumulation`, `enable_evidence_chains`. Also `_NO_OTHER_LLM_CALL: frozenset[str]` (86 names), test-module level.

- [ ] **Step 1: Write the tests.**

  (a) In `test_rag_search_admits_an_over_limit_user_whose_llm_calls_run_on_their_own_keys`'s parametrize list, add this case after `switches-without-an-llm-call`, and add `"stages-without-an-llm-call"` to the end of `ids`:

```python
        ({"enable_abstention": True, "enable_knowledge_strips": True, "enable_web_fallback": True,
          "enable_query_rewriting_loop": True, "adaptive_rerun_on_low_confidence": True,
          "adaptive_advanced_rewrites": True},
         {"openai"}, ["openai"]),
```

  (b) After `test_rag_search_still_refuses_an_own_key_answer_with_another_llm_stage`, add the declared list and the drift test. The groups follow the classification (B, C, D, E):

```python
# Every boolean switch of the RAG request schemas makes an LLM call besides the answer and the reformulation
# (rag_unified._RAG_OTHER_LLM_STAGES) or is declared here (TASK-15427).
_NO_OTHER_LLM_CALL = frozenset({
    # Their call is one _rag_llm_providers names: the answer, the reformulation, LLM reranking (by its strategy).
    "enable_generation", "enable_query_reformulation", "enable_reranking",
    # Read only by the agentic strategy, which _rag_llm_providers refuses as a non-standard strategy.
    "agentic_adaptive_budgets", "agentic_debug_trace", "agentic_enable_metrics", "agentic_enable_query_decomposition",
    "agentic_enable_section_index", "agentic_enable_semantic_within", "agentic_enable_table_support",
    "agentic_enable_tools", "agentic_enable_vlm_late_chunking", "agentic_extractive_only",
    "agentic_prefer_structural_anchors", "agentic_quote_spans", "agentic_use_llm_planner",
    "agentic_use_provider_embeddings_within", "agentic_vlm_detect_tables_only", "explain_only",
    # Read only inside a listed stage: the research loop, document grading, claims, post-verification.
    "enable_image_search", "enable_video_search", "enable_research_action_dedup", "enable_research_progress",
    "search_url_scraping", "grading_fallback_to_score", "enable_query_rewriting_loop", "doc_only_verification",
    "generate_verification_report", "adaptive_advanced_rewrites", "adaptive_rerun_on_low_confidence",
    "adaptive_rerun_bypass_cache", "adaptive_rerun_include_generation",
    # No LLM call: rule-based, retrieval, filtering, formatting, telemetry, or the answer's own generator again.
    "enable_structured_response", "enable_query_decomposition", "enable_intent_routing",
    "enable_dynamic_granularity", "enable_abstention", "enable_knowledge_strips", "enable_web_fallback",
    "enable_pre_retrieval_clarification", "enable_numeric_fidelity", "strict_extractive", "require_hard_citations",
    "expand_query", "spell_check", "enable_discussion_search", "enable_multi_vector_passages", "mv_flatten_to_spans",
    "enable_text_late_chunking", "enable_enhanced_chunking", "enable_vlm_late_chunking", "vlm_detect_tables_only",
    "enable_table_processing", "enable_numeric_table_boost", "enable_parent_expansion", "include_parent_document",
    "include_sibling_chunks", "enable_citations", "enable_chunk_citations", "include_page_numbers",
    "enable_security_filter", "detect_pii", "redact_pii", "content_filter", "enable_content_policy_filter",
    "enable_injection_filter", "enable_html_sanitizer", "enable_cache", "adaptive_cache", "use_embedding_cache",
    "use_connection_pool", "enable_resilience", "circuit_breaker", "enable_batch", "enable_checkpoint",
    "collect_feedback", "apply_feedback_boost", "highlight_results", "highlight_query_terms",
    "include_rerank_debug_documents", "debug_mode", "track_cost", "enable_analytics", "enable_monitoring",
    "enable_observability", "enable_performance_analysis",
})


def test_every_rag_request_switch_says_whether_it_makes_another_llm_call():
    """A new switch of the RAG request schemas must join _RAG_OTHER_LLM_STAGES or _NO_OTHER_LLM_CALL (TASK-15427)."""
    from typing import Optional

    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedBatchRequest, UnifiedRAGRequest

    switches = {
        name
        for model in (UnifiedRAGRequest, UnifiedBatchRequest)
        for name, field in model.model_fields.items()
        if field.annotation in (bool, Optional[bool])
    }
    listed = set(rag_ep._RAG_OTHER_LLM_STAGES)
    assert switches - listed - _NO_OTHER_LLM_CALL == set(), "add each to _RAG_OTHER_LLM_STAGES or _NO_OTHER_LLM_CALL"
    assert listed & _NO_OTHER_LLM_CALL == set()
    assert (listed | _NO_OTHER_LLM_CALL) - switches == set()
```

  `Optional[bool]` compares equal to `bool | None`, so both spellings match. The 98 switches use only `bool` and `Optional[bool]` (measured).
- [ ] **Step 2: Run them.** `<pytest> -q tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py -k "says_whether_it_makes_another_llm_call or stages-without-an-llm-call"`
  Expected: `2 failed, 25 deselected` (measured).
  - `[stages-without-an-llm-call]` fails with `assert 402 != 402`.
  - The drift test fails on its second assertion, `assert {'adaptive_ad...web_fallback'} == set()`, which names the six. Its first assertion passes: the declared list covers every other switch.
- [ ] **Step 3: Trim the list in `rag_unified.py`.** Drop the six from `_RAG_OTHER_LLM_STAGES`, leaving the 12 of the Interfaces block in that order. Keep its two-line comment above it. Replace the two-line `# Not listed, ...` comment below the tuple with:

```python
# Every other switch of the RAG request schemas makes no LLM call of its own, or makes it only inside a listed
# stage; test_rag_agentic_api.py's _NO_OTHER_LLM_CALL declares each one (TASK-15427).
```

- [ ] **Step 4: Run them.** The same command. Expected: `2 passed, 25 deselected` (measured).
- [ ] **Step 5: Mutation probes.** Revert each mutation after its run. Each was measured to fail:
  - Add `enable_probe_switch: bool = Field(default=False)` to `UnifiedRAGRequest` (`tldw_Server_API/app/core/RAG/rag_schemas_unified.py`, above `enable_pre_retrieval_clarification`, `:1015-1019`; `app/api/v1/schemas/rag_schemas_unified.py` only re-exports it). The drift test gives `AssertionError: add each to ...` / `assert {'enable_probe_switch'} == set()`.
  - Drop `"enable_post_verification"` from `_RAG_OTHER_LLM_STAGES`. The drift test gives `assert {'enable_post_verification'} == set()`.
- [ ] **Step 6: Commit** (message 2): `git add tldw_Server_API/app/api/v1/endpoints/rag_unified.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py && git diff --cached --check`.

## Task 3: The reformulation's provider is canonical; the refusal test asserts the lookups (AC #3)

**Files:** Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` (`_rag_llm_providers`, `:226-249`). Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py`.

**Interfaces:** Consumes `canonical_provider_name(provider: str) -> str` from `tldw_Server_API.app.core.LLM_Calls.provider_identity`, the import `generation_defaults.py:7` uses. `_rag_llm_providers(settings: Mapping[str, Any]) -> list[str | None]` keeps its signature.

- [ ] **Step 1: Write the tests.**

  (a) In the admit test's parametrize list, add this case after `reformulated-history`, and the id `"reformulated-on-an-alias-of-the-answers-provider"` after `"reformulated-history"` in `ids`:

```python
        ({"chat_history": [{"role": "user", "content": "Earlier"}], "classifier_provider": "OAI"},
         {"openai"}, ["openai"]),
```

  In the admit test's body, move `assert asked == looked_up` to be the first assertion, before `assert resp.status_code != 402, resp.text`. That way the RED names the lookups, not the 402 that the stand-in lookup (`provider in own`, no canonicalization) adds.

  (b) Parametrize `test_rag_search_still_refuses_an_own_key_answer_with_another_llm_stage` with `("extra", "looked_up")`. Each case gets `[]`, except `history-on-a-server-key`, which gets `["openai", "groq"]`. The ids are unchanged. Add `looked_up` to its signature, and wrap the signature over three lines (`def ...(`, the four parameters, `):`): it is 126 characters on `cb70f0c5d1`, and 137 with the parameter. Add a second docstring paragraph: `A stage on an unnamed provider is refused without a lookup; a named one on a server key, after it (TASK-15427).` Rename `_asked` to `asked`, and add `assert asked == looked_up` after `assert marked == []`.
- [ ] **Step 2: Run them.** `<pytest> -q tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py -k "admits_an_over_limit_user or still_refuses_an_own_key_answer"`
  Expected: `1 failed, 11 passed, 16 deselected` (measured). `[reformulated-on-an-alias-of-the-answers-provider]` fails with `assert ['openai', 'OAI'] == ['openai']`. The refusal cases pass: they pin behaviour that already holds.
- [ ] **Step 3: Canonicalize.** In `_rag_llm_providers`:
  - add the `canonical_provider_name` import beside the function's two local imports;
  - wrap the reformulation's provider: `providers.append(canonical_provider_name(settings.get("classifier_provider") or named or "openai"))`;
  - change the docstring's second paragraph to `The answer's provider, as the generator resolves it, and the reformulation's when a chat history is reformulated, both canonical (TASK-15427). Any other LLM stage adds None, so an over-limit user is refused as before.`
- [ ] **Step 4: Run them.** The same command. Expected: `12 passed, 16 deselected` (measured).
- [ ] **Step 5: Mutation probe.** Change the reformulation's append to `providers.append(None)`, run `-k history-on-a-server-key`, and revert. Expected: `1 failed` with `assert [] == ['openai', 'groq']` (measured). The case now tells "groq is not the user's key" from "a None was appended", which was the deferred finding.
- [ ] **Step 6: Commit** (message 3): `git add tldw_Server_API/app/api/v1/endpoints/rag_unified.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py && git diff --cached --check`.

## Task 4: Verify, close, review, PR

- [ ] **Step 1: Focused set** (with the probe plugin). Expected: `43 passed, 1 skipped`, and an empty log (measured).
- [ ] **Step 2: Wider set** with label `final`, compared with Task 0's baseline. Expected: `3 failed, 1038 passed, 58 skipped, 2 errors`, with the same 3 failures and 2 errors as the base, +3 passed (this plan's new tests), and 4 log lines (the base's, without the groq line). Measured.
  - One earlier prototype run under heavy machine load also gave 9 setup errors `DatabaseLockError: Database is temporarily locked` (AuthNZ SQLite; RAG endpoint tests and two OpenAPI-doc tests) and `AuthNZ/integration/test_rag_media_permissions_claims.py::test_media_process_videos_requires_create_permission` (`TransactionError ... PostgreSQL transaction`). A rerun on the same code was clean.
  - Re-run such a test alone before judging it. Any other failure not in the baseline is this plan's.
  - Then check for a leftover `server-server-e2e.pid` (Task 0 Step 6).
- [ ] **Step 3: CI checks.** Each expected result was measured on the prototype:
  - `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -n 8 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI`: `1497 passed, 6 skipped` (about 4 minutes). It includes the function-size ratchet. This was measured without the plan copy; the plan copies already under `Docs/superpowers/plans` pass the same suite.
  - `PATH="$(dirname $PY):$PATH" $PY -m pytest -p no:cacheprovider -q tools/backlog-py/tests`: `164 passed`.
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_shard_coverage.py`: `new_uncovered=0` and "OK — no newly-unshared test files."
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/test_quality_triage.py --enforce`: exit 0, "OK — no offenses beyond the baseline."
  - `PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/ci/route_auth_ratchet.py`: "208 unauthenticated routes; baseline exact."
  - `PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts`: exit 0, "OK — OpenAPI fingerprint matches the checked-in snapshot." and "OK — committed path union matches the canonical contract."
  - `/opt/homebrew/bin/uvx bandit -ll tldw_Server_API/app/api/v1/endpoints/rag_unified.py`: "No issues identified.", Medium 0, High 0.
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .`: exit 0.
  - `$PY -m ruff check` on the three touched files: the 7 findings of `test_rag_agentic_api.py` on `cb70f0c5d1` (I001 ×5, F401, N801), and none new. The other two files are clean (measured).
  - `git grep -n -E "rag_unified|test_rag_agentic_api|test_rag_stream_parity" Helper_Scripts/ci/scope_predicate_baseline.txt`: nothing.
  - `git diff origin/dev --check`: nothing.
  - `git grep -n -E "TASK-<(HP|SP|RG)>" -- ':!Docs/superpowers/plans'`: nothing.
- [ ] **Step 4: Close TASK-15427** (backlog-py; fill every `<...>` first):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-15427 \
  --append-notes "RED/GREEN (record the real counts): Task 1 1 failed (assert [('dns', 'api.groq.com')] == []) -> 1 passed; Task 2 2 failed -> 2 passed (mutations: a new bool field on UnifiedRAGRequest, enable_post_verification dropped from the list: 1 failed each); Task 3 1 failed, 11 passed -> 12 passed (assert ['openai', 'OAI'] == ['openai']; mutation: the reformulation appends None -> history-on-a-server-key 'assert [] == ['openai', 'groq']'). Focused set 40 -> 43 passed (1 skipped), 0 outbound attempts under a socket probe. Wider set <base> -> <after>. tests/Docs + lint + CI <n> passed; backlog-py <n> passed; shard coverage, test-quality ratchet, route-auth ratchet and OpenAPI unchanged." \
  --append-notes "Decisions: all 98 boolean switches of UnifiedRAGRequest and UnifiedBatchRequest classified with file:line (plan, 'AC #2: every switch, classified'); 12 make another LLM call and stay in _RAG_OTHER_LLM_STAGES; enable_abstention, enable_knowledge_strips, enable_web_fallback, enable_query_rewriting_loop, adaptive_rerun_on_low_confidence and adaptive_advanced_rewrites make none of their own (or only inside a listed stage) and leave it, so an over-limit user on their own key with them on is admitted (every call is still checked at its own call where the call-site gate runs); the other 86 are declared in test_rag_agentic_api._NO_OTHER_LLM_CALL. The drift test covers boolean switches only; string modes (strategy, search_depth_mode, reranking_strategy) stay checked by value in _rag_llm_providers. The stream test stubs perform_chat_api_call_async and asserts no network, not the stub's calls (they are TASK-<SP>'s discarded prefetch answer). Follow-ups: TASK-15421 (RG under-commit, re-verified incl. the Redis governor), TASK-<HP>, TASK-<SP>." \
  --append-notes "Docs: none (Usage_Quotas.md names the list by name; ADR-083 unchanged). Bandit: <result>." \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "test_rag_streaming_generation_provider_override stubs the standard pipeline's answer call and asserts, through block_live_network(), that it opens no connection; the call it made to api.groq.com came from the stream's retrieval step, which runs and discards a full answer (TASK-<SP>). Every boolean switch of the RAG request schemas is classified: _RAG_OTHER_LLM_STAGES keeps the 12 that make an LLM call besides the answer and the reformulation and drops six that make none of their own; a drift test fails on a switch in neither list, in both, or in neither schema. _rag_llm_providers canonicalizes the reformulation's provider as the answer's is canonicalized, so an alias is looked up once, and the refusal test asserts the providers asked. Follow-ups filed: TASK-<HP>, TASK-<SP>; TASK-15421 annotated." \
  -s Done
```

  `--check-dod 3` holds: no doc needed a change (Global Constraints, Docs). Then:
  - `git grep -n -E "<(answer|base|after|n|result)>" -- 'backlog/tasks/task-15427 *' 'backlog/tasks/task-15421 *' 'backlog/tasks/task-<HP> *' 'backlog/tasks/task-<SP> *'` must print nothing (write the real ids; the unscoped grep prints 4 pre-existing lines in task-13404 and task-13921);
  - `git add -A backlog/tasks && git diff --cached --check`, and commit (message 4).
- [ ] **Step 5: Review before the PR.** Run `/superpowers:requesting-code-review` in a clean subagent on `origin/dev..HEAD`. Give it this plan's Review Focus. Address the findings before the PR is marked ready, and record each ruling in the ledger.
- [ ] **Step 6: Finish the PR and merge.**
  - `git push`. Replace the draft's body with `gh pr edit <n> --body-file <file>`, then `gh pr ready <n>`.
  - The body states:
    - the three ACs and how each is met;
    - the classification summary (12 / 6 trimmed / 86) and the six switches by name;
    - the follow-ups (TASK-15421 note, TASK-<HP>, TASK-<SP>);
    - the Review Focus items;
    - the measured counts;
    - the `## Change summary` (or the waiver; Decision 1).
  - It ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - Merging: `gh variable get MERGE_QUEUE` (unset when this plan was written). When reviews are settled and the Change summary is in, run `QUEUE="<n>" sh <plans dir>/pr_queue3.sh` in the background. It reads dev's required checks (8) from the ruleset, rebases server-side when BEHIND, and merges when all pass. Wait for each body edit's re-triggered license status before merging. Confirm MERGED before deleting the branch.

## Commit messages

Each ends with a blank line and `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

- 0: `chore(backlog): file TASK-<HP> and TASK-<SP>, note TASK-15421, start TASK-15427 and add its plan`
- 1: `test(rag): the provider-override stream test opens no outbound connection (TASK-15427)`
- 2: `fix(usage): RAG's other-LLM-stage list drops six switches that make no LLM call; a drift test holds every switch to a list (TASK-15427)`. Body: the six, and why each makes no call of its own.
- 3: `fix(usage): RAG canonicalizes the reformulation's provider before the own-key lookups (TASK-15427)`
- 4: `chore(backlog): close TASK-15427`

## Conflicts with open PRs

None. 41 open PRs target `dev` (2026-11-07). None of them touches `rag_unified.py`, `test_rag_agentic_api.py`, `test_rag_stream_parity.py`, `rag_schemas_unified.py`, `usage_quota_deps.py` or the two test helpers. Checked with `gh api repos/rmusser01/tldw_server/pulls/<n>/files --paginate` per PR. Neither PR needs to land first.

## Not measured

- **TASK-<HP>, end to end.** No `/chat/completions` request was driven with a billing org, an own-key provider and a failover. That claim is code reading (cited lines).
- **The real-Redis Lua path of the RG commit.** It was measured only through the in-memory stub client. The commit code is shared, and `delete(hkey)` is unconditional (`governor_redis.py:2404-2407`).
- **The six trimmed switches, end to end.** Each was classified by reading its gate (cited). No pipeline run counted LLM calls per switch, and the admit test stubs the pipeline.
- **Ids.** `task_next.sh` was not run: the ids are allocated at execution.

## Review fold (2026-11-07)

Folded from `review_plan_15427.md` (the pre-fold copy is `plan_15427.prefold.md`). The prototype `proto_15427.patch` is unchanged: every finding is about task texts and plan steps.

- **I1** (fix sketch would answer the raw query): Task 0 Step 3(b), TASK-<SP> description (caveat, `unified_pipeline.py:2507-2518`, `streaming_executor.py:696`) and a new AC; Decision 2 (recommendation).
- **I2** (AC #2 vs the remedy; fourth failover path): Task 0 Step 3(a), TASK-<HP> description (trade-off; the health-check switch `chat.py:5246-5276`) and AC #2 reworded; Decision 3 (trade-off and fourth path). AC #1 stays the owner's choice.
- **M1** (placeholder guard): Task 4 Step 4, grep scoped to the four touched task files.
- **M2** (probe ceilings): Global Constraints, "Probe ceilings".
- **M3** (TASK-15421 note): Task 0 Step 2, note ends with the both-governors instruction.
- **M4** (TASK-<SP> priority): Task 0 Step 3(b) description opens with the user-facing line; Decision 2 "Priority".
- **M5** (call-site claim): Review Focus 1 and the Task 4 Step 4 notes.
- **M6** (ambiguous path): Task 2 Step 5, core path named.
