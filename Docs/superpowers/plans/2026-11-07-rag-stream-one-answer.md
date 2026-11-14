# The streaming RAG search answers once: its retrieval step runs no generation (TASK-15482) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/api/v1/rag/search/stream` makes one answer call, the streamed one. Its retrieval step runs no generation and no post-answer stage. The streamed answer keeps the effective query and the stream keeps its default pre-retrieval clarification.

**Architecture:**
- The retrieval step already asks for retrieval only (`kwargs["enable_generation"] = False`, `streaming_executor.py:381`). `unified_rag_pipeline` ignores it, because `_resolve_effective_enable_generation` (`unified_pipeline.py:2390-2391`) prefers the resolved payload's `True`. The fix makes an explicit `False` win. It rewrites the nested function's return line and adds a comment to its `def` line, so the line count does not change.
- The pipeline keeps the shared `ResolvedRAGRequest`, so `_sync_effective_query` (`:2507-2518`) still writes the effective query where the stream reads it (`streaming_executor.py:696`). No copy is made.
- Pre-retrieval clarification defaults to the effective `enable_generation` (`:3163-3167`), which is now `False` in the retrieval step. When the request and its profile leave `enable_pre_retrieval_clarification` unset, the retrieval step sets it from the request's own `enable_generation`, so the gate runs as it does today.

**Tech Stack:** Python 3.12, FastAPI `TestClient`, pytest, backlog-py.

**Spec:** `backlog/tasks/task-15482 - The-streaming-RAG-search-s-retrieval-step-generates-a-full-answer-and-discards-it.md` (its description and four ACs; filed by PR #3439, TASK-15427). Planner brief: `brief_plan_15482.md` (plans dir).

**Measured on:** `origin/dev` at `7a020da14ab28e220fab652269723e3a2a4abbcb` (`7a020da14a`, 2026-11-07, "Merge pull request #3432") with `proto_15427.patch` (plans dir) applied, in a detached worktree that was reverted afterwards. `rag_service/`, `rag_unified.py`, `generation.py`, `gateway.py` and the RAG tests are unchanged between `cb70f0c5d1` (the task's line numbers) and `7a020da14a`, so every `file:line` below holds on both.

**Prototype:** `proto_15482.patch` (plans dir), on top of `proto_15427.patch`: 4 files, +131/-11 (source +6/-3; tests +125/-8). It holds every code and test change in Task 1, but not the plan copy or the task file. It applies with `git apply` after `proto_15427.patch`. On the branch, after #3439 has merged, the stub's docstring names `TASK-15482` where the prototype base has `TASK-<SP>`. That one hunk then needs `git apply --3way`, or Task 1 Step 1 done by hand.

Paths: `UP` = `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`, `SE` = `.../rag_service/streaming_executor.py`, `EP` = `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`, `GEN` = `.../rag_service/generation.py`.

---

## What was verified

| Item | Evidence | Verdict |
|---|---|---|
| The bug | `SE:381` sets `kwargs["enable_generation"] = False`, but the route's `resolved_request` (`EP:2340`, passed on at `SE:379`) has `payload["enable_generation"] = True` (the route refuses anything else, `EP:2308-2309`). `_resolve_effective_enable_generation` returns `bool(payload.get("enable_generation", enable_generation))` (`UP:2390-2391`; read again at `UP:6900`). So the retrieval step runs the generation branch (`UP:6929`), `execute_generation_phase` (`UP:7255`) and its provider call (`GEN:531`, `chat_service.perform_chat_api_call_async`). **Measured:** the new counting test records `[False, False, True]`: the discarded answer, the discarded suggestions call (`UP:7325`, after the answer), then the streamed answer. | fix (Task 1) |
| AC #1 | Every post-answer stage is gated on `result.generated_answer` (table below), which stays `None` without the generation branch. **Measured:** after the fix the same test records `[True]`. | Task 1 |
| AC #2 | The pipeline writes the effective query into the shared request (`UP:2515-2516`: `resolved_request.query` and `payload["query"]`); the stream reads `resolved_request.query` (`SE:696`, wired at `SE:841-842`). **Measured:** the effective-query test passes on the base (2 cases) and with the fix. With the task's fix sketch instead (a `dataclasses.replace` copy with `enable_generation: False`, the pipeline line unchanged), AC #1's test passes but both cases fail: the streamed prompt says `Question: Who dirrects the observatory?` and `Question: Who directs it?`. | Task 1 (guard) |
| AC #3 | `effective_pre_clarify = enable_pre_retrieval_clarification if not None else effective_enable_generation` (`UP:3163-3167`). The route passes the payload's value through `pipeline_kwargs` (`transport.py:82-104`); unset is `None` (schema default, `RAG/rag_schemas_unified.py:1020-1024`, "When None, defaults to true if enable_generation=true"). **Measured:** with only the pipeline line fixed, the unset case streams an answer (`['contexts', 'reasoning', 'delta', 'complete']`) instead of `['clarification', 'delta', 'complete']`. With the retrieval step's explicit value, it passes. | Task 1 |
| AC #4 | **Measured on the base without the stub:** `assert [('dns', 'api.groq.com')] == []`. The probe plugin logs, for the four files, `test_owner_retrieves_chunked_media_through_stream_endpoint[sqlite]` and `test_permission_first_auth_stream_without_preseeded_scope[sqlite-Rowan-True]`, each `[('dns', 'api.openai.com')]`. **After the fix:** the test passes without the stub, and the log is empty. The two `postgresql` cases could not run (PostgreSQL down). | Task 1 |

### The smallest correct fix

| Option | Effective query | Clarification | Other state | Diff |
|---|---|---|---|---|
| A. Override on the shared payload: set `resolved_request.payload["enable_generation"] = False` around the call, restore it in `finally` | kept (same object) | needs the explicit value | every write-back kept | ~7 lines in `SE`. It mutates shared state for the call's length. |
| B. The task's sketch: a `dataclasses.replace` copy with `enable_generation: False`, then copy `.query` back | only what is copied back; `payload["query"]` and `payload["sources"]` (`UP:2996`) stay on the copy (the stream reads neither today) | needs the explicit value | a later write-back is lost silently | ~4 lines and an import in `SE`. Without the copy-back, AC #2 fails (measured). |
| **C. Recommended: the pipeline honours an explicit `enable_generation=False`** (`return bool(enable_generation) and bool(payload.get("enable_generation", True))`) | kept (same object) | needs the explicit value | every write-back kept | 2 lines rewritten in `UP` (same line count); 3 added in `SE`. |

- **Why C:** the retrieval step already says retrieval only (`SE:381`), and the pipeline drops it. C fixes it where it is dropped, and it copies nothing and mutates nothing.
- **Who else is affected:** C changes only one case, an explicit `False` with a payload that says `True`. Every other `unified_rag_pipeline` caller either passes no `resolved_request` (the pipeline builds one from its kwargs, `UP:2370-2385`; for example the adaptive rerun `UP:8254-8309`, `/advanced`, slides, evaluations, workflows), or builds its kwargs from the same payload (`/search` `EP:1601-1604`, MCP `rag_module.py:418-424` and `:1000`, batch `UP:9211-9225`). Two indirect call sites are not in the grep count: the stream's own call (`SE:394`, `standard_pipeline(**kwargs)`, the one this fix targets) and the shared-workspace chat service's `self.rag_pipeline(...)` (`shared_workspace_chat_service.py:607`, `enable_generation=False`, `resolved_request=None`, so unaffected). Neither changes the conclusion. Checked with `git grep -n "unified_rag_pipeline("` over `app/`: 21 hits, which are 20 call sites and one docstring example (`UP:2312`).
- **Kept by C:** every case `cd184c9fb9` pinned still holds. A payload `False` with the default kwarg still gives retrieval only (`test_unified_pipeline_coordinates_retrieval_only_result`, which becomes the `[payload-off]` case).

### State the stream reads from its retrieval step

From the result (`SE:391-431`):
1. `documents` (or `sources`), for the contexts and the answer;
2. `errors`, for the retrieval-failure terminal event (`SE:393-402`);
3. `metadata.clarification` and `generated_answer`, for the clarification events (`SE:408-417`, emitted at `SE:793-797`);
4. `metadata.security_filter` (`SE:418-424`);
5. `metadata.source_status` (`SE:429`);
6. research progress, through the callback (`SE:383-389`).

Through the shared object: 7. `resolved_request.query`, the effective query (`SE:696`).

Not read after the call: `resolved_request.payload`. The stream works from its own copy, taken before the call (`SE:737`), for the generation config, citations and claims options.

Under C, items 1-6 come from the retrieval-only result, which is `build_retrieval_only_result` + `coordinate_standard_result_evidence` (`UP:6911-6926`). Neither filters documents (`post_retrieval_coordinator.py:60-103`). Item 7 is unchanged. The differences, all caused by the discarded answer: (a) with post-verification and its adaptive rerun on, an adopted rerun replaced `result.documents` (`UP:8249`, `UP:8466-8471`); (b) that adopted rerun's `result.metadata.update(new_result.metadata)` (`UP:8470`) could overwrite the `security_filter` and `source_status` items 4 and 5 read; (c) the numeric-fidelity retry (`enable_numeric_fidelity` with `numeric_fidelity_behavior="retry"`, `UP:8060-8077`) merged extra documents. So today the stream's contexts could come from the rerun or the retry of a discarded answer. Now they come from the stream's own retrieval (Review Focus 2).

### Post-answer stages the retrieval step runs today

All of them are skipped after the fix. Each needs the generation branch (`UP:6929`) or a generated answer, which stays `None` (the cache hit sets `None`, `UP:4148`; the generation branch is the only other writer before these stages):

| Stage | Gate (`UP`) | Makes an LLM call |
|---|---|---|
| strict extractive, multi-turn synthesis, the answer | inside `:6929` | the answer (`:7255`); synthesis's critique (`:7046`) |
| suggestions | `:7325` | yes (`suggestion_generator.py:205-234`, through `chat_service.perform_chat_api_call_async`) |
| gated abstention (ask/decline) | `:7386` (needs the effective flag) | no |
| hard citations, quote citations | `:7460`, `:7501` | no |
| fast groundedness | `:7513` | yes |
| claims | `:7550` | yes |
| numeric fidelity (and its retry) | `:7982` | yes, the retry |
| post-verification and the adaptive rerun | `:8142`, `:8249` | yes |
| post-generation evidence chains | `:8500` | yes |
| utility grading | `:8569` | yes |
| faithfulness eval | `:8842` | yes |

**The stream itself** runs one of them, claims, once. `_stream_generation_events` passes `enable_claims` to `generate_streaming_response` (`SE:703-709`). Its overlay is heuristic: `GEN:859-861` passes an `_analyze` that returns no claims, so there is no LLM call. The others never reached the stream's client: nothing emits them. Their loss is the AC.

### Usage and quota effects

- **Per call:** each provider call runs `check_llm_call_async` and the usage post-call hooks (`gateway.py:1340-1357`). A stream request now makes one answer call, not two, plus no post-answer-stage calls. So it writes one usage row fewer (more with suggestions and the like), and runs one call-site quota check fewer.
- **At the boundary:** `enforce_llm_quota(..., provider=_rag_llm_providers(...))` (`EP:2313`) already names the answer's provider once. It never assumed two calls, and it is unchanged. `_log_rag_queries_for_org(units=1)` (`EP:2317`) is unchanged.
- **Tests:** none pins two rows or two checks for the stream. `git grep "search/stream"` over the tests turns up the boundary tests only (`test_rag_stream_parity.py`'s `test_rag_search_stream_refuses_an_over_quota_user_with_a_real_402` and `test_rag_search_stream_names_the_providers_of_the_resolved_request`, which stop before any work), and `test_rag_provider_credentials.py`'s fake pipelines. Those honour `enable_generation is False` already (`test_rag_provider_credentials.py:364-365`).
- **Own keys:** with `credential_runtime`, the retrieval step's answer resolved the user's credential and billed their provider. Now only the stream's call does.
- **Fail-closed leftover (Review Focus 4):** on the stream route, `_RAG_OTHER_LLM_STAGES` still refuses an over-limit own-key user who turns on a post-answer stage that no longer runs there. Those stages are suggestions, claims, post-verification, fast groundedness, utility grading, multi-turn synthesis and faithfulness eval.

### Docs

None changes. No doc describes the retrieval step's answer:
- `RAG-API-Guide.md:134-136` and `RAG_API_Documentation.md:128-145` say only that streaming requires generation and list the event types;
- `RAG-Developer-Guide.md:75` and `:207` say the same;
- `Docs/Architecture/rag.md:31` and `:49` say the same;
- `claims-extraction.md:36` names `claims_overlay` and `final_claims`;
- `app/core/RAG/README.md:140` says the same as the guides.

The schema's `enable_pre_retrieval_clarification` description (`rag_schemas_unified.py:1022`) stays true. Nothing is mirrored, so `Docs/Published` is untouched.

### The WebUI's event contract

Knowledge QA consumes this route (`chat-rag.ts:581-592`, `knowledgeQaClient.ts:51-55`, `KnowledgeQAProvider.tsx:2715`). The fix changes what the retrieval step computes, not what the executor emits. The executor still emits `heartbeat`, `research_*`, then either `clarification`/`delta`/`complete`, or `contexts`/`reasoning`/`delta`.../`claims_overlay`/`final_claims`/`complete`, or `error` (`SE:728-875`), in the same order on success. The clarification case keeps its three events (pinned by the AC #3 test). Only the time to the first `contexts` event shrinks.

**One order changes, on a failure of the answer's provider or credential** (measured with a credential runtime whose `resolve` raises `ByokResolutionError("invalid_provider_credentials")`; the route always builds a runtime, `EP:164-176`). Base: the retrieval step's answer fails first, and the stream emits only `[error]`. With the fix: `[contexts, reasoning, error]`, with the same error fields (`invalid_provider_credentials`, `upstream_dispatched=true`, `allow_non_stream_fallback=false`). Knowledge QA shows the sources, then the same non-replayable error, which is what any failure of the streamed call produces today. It matches the stream before `cd184c9fb9` and the agentic stream. Without a credential runtime (the test fixture), the base already emitted `contexts`, `reasoning`, `error`.

## Decisions

1. **ADR-004 Change summary.** This PR is AI-authored, and no recorded waiver names TASK-15482.
   - **Recommendation:** ask the owner to extend the waiver to this PR, as for the TASK-15427 follow-ups. Otherwise the owner writes the `## Change summary` before merge. The controller asks.

Not owner questions (rulings for the reviewer, each in Review Focus):
- Option C over A/B (above).
- The fail-closed boundary leftover (Review Focus 4). It is not filed: it refuses more, never less, and no hosted deployment exists.

---

## Global Constraints

Carried verbatim from the brief:

- Tests make no real outbound network requests; stub the provider adapter or its transport. Run every RAG test with the probe plugin. Known leaks on dev: the stream parity test (Groq; stubbed by #3439) and four `test_restricted_postgres_media_retrieval.py` stream cases (OpenAI; this task removes them).
- Function-size ratchet `tldw_Server_API/tests/lint/function_size_baseline.json`: frozen functions must not grow (`unified_pipeline.py` has large functions: check each one you touch).
- Backlog: edit tasks with backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`); `--append-notes`, never `--notes`; stage with `git add -A backlog/tasks`; never hand-edit task files. A new task's explicit id comes from `zsh <plans dir>/task_next.sh` (highest id in use; new = that + 1) after `mkdir <scratchpad>/id_lock`; placeholders in the plan; do NOT allocate ids now.
- Branch from `origin/dev`; the PR targets `dev`. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- Redis: never FLUSHALL or SCRIPT FLUSH. PostgreSQL only through the repo fixtures, with a session-unique `TEST_DB_NAME`; drop only your own objects. The shared PostgreSQL at 127.0.0.1:5432 is currently down (closes every connection); do not restart it; say which PG cases you could not run.
- Mirrored docs (`Docs/Published`) need `bash Helper_Scripts/refresh_docs_published.sh` in the same PR.
- CI checks before the PR, in the last task: `tests/Docs`, `tests/lint`, `tests/CI`, backlog-py's tests; shard coverage; `test_quality_triage.py --enforce`; route-auth ratchet; OpenAPI export check; Bandit on touched source; every test file that names a touched symbol (git grep), baseline on the base; `scope_predicate_baseline.txt`; `check_backlog_task_ids.py` exit 0. The Frontend E2E journeys are not required checks (Knowledge QA streams): the PR step diffs the PR's E2E failures against a recent dev run.
- Agent shells don't keep the venv `PATH`: prefix commands that spawn `python` with `PATH="$(dirname $PY):$PATH"`.

Plan-specific:

- **Base.** Start only after PR #3439 (TASK-15427) has merged: `gh pr view 3439 --json state` must say `MERGED`. Its Task 1 adds the provider stub that this plan removes. Before that, the branch would not have the stub or the task file.
- **No new ids.** This plan files no task. TASK-15482 exists on `dev` once #3439 merges.
- **`$PY`** is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Write it out in each call, because shell variables do not survive between agent calls. `<plans dir>` is `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`. `<scratchpad>` is the executing session's scratchpad.
- **`<pytest>`** stands for `PATH="$(dirname $PY):$PATH" PYTHONPATH=<plans dir> NETPROBE_LOG=<scratchpad>/netprobe_<label>.log TLDW_TEST_NO_DOCKER=1 TEST_DB_NAME=tldw_test_15482_<session> $PY -m pytest -p no:cacheprovider -p netprobe_15427`.
  - Delete the log before each run, and read it after. `netprobe_15427.py` runs each test inside the repo's `block_live_network()` and appends `<nodeid> <attempts>` for any blocked attempt.
  - **Probe ceilings:** a test with its own `block_live_network()` hides its attempts from the log (it records and raises first), so `test_rag_streaming_generation_provider_override` reports through its own assertion. A thread that outlives its test is not probed after teardown.
- **PostgreSQL is down.** Add `-k "not postgresql"` to every run. Otherwise each `postgresql` case waits on the dead server before it skips: the four focused files took more than 25 minutes for about 60 of 175 items. The deselected cases are reported as not run. The two `postgresql` stream cases of `test_restricted_postgres_media_retrieval.py` are among them.
- **TDD.** Run each test you write before its code, and see it fail for the stated reason. Record the real output.
- **Sizes and baselines.**
  - `unified_rag_pipeline` is frozen at 7063 lines (`function_size_baseline.json:163`). The fix replaces 2 lines with 2, so the count is unchanged; do not add a line to it.
  - `_retrieve_standard_documents` (`SE:366-431`, 66 lines) is not frozen; it grows by 3.
  - `Helper_Scripts/ci/scope_predicate_baseline.txt:34` pins `unified_pipeline.py:3748`. The fix is at `:2390-2391` and shifts no line. `streaming_executor.py` and the test files are not listed.
- **Git.** Stage named files. `git diff --cached --check` must print nothing before each commit.
- **Go cache.** Test runs can leave a read-only Go module cache under `tldw_Server_API/Config_Files/acp_runner_home`. Clean it with `chmod -R u+w tldw_Server_API/Config_Files/acp_runner_home && git clean -ffdx -- tldw_Server_API/Config_Files/acp_runner_home`. Never `rm -rf` the directory. The last `git status --short` must be empty.
- **Network in the new tests.** The three new route tests patch `chat_service.perform_chat_api_call_async`. Every provider call they can reach goes through it: the answer (`GEN:500`), the streamed answer, and suggestions (`suggestion_generator.py:205`). The reformulation is patched at `query_classifier.reformulate_query`, and spell check and the clarification gate are local. The probe log stays empty (measured).

## Review Focus

1. **The pipeline contract change (Option C).** An explicit `enable_generation=False` now wins over a resolved payload that says `True`; nothing else changes. The reviewer should confirm that no caller passes `enable_generation=False` with a `resolved_request` whose payload says `True` and expects an answer (call sites in What was verified, "Who else is affected"). Pinned by `[explicit-off-beats-the-payload]`.
2. **The stream's contexts with post-verification or numeric fidelity on.** Before, an adopted adaptive rerun of the discarded answer replaced the documents the stream showed and answered over (`UP:8249`, `:8466-8471`), and could overwrite the `security_filter` and `source_status` metadata the stream emits (`UP:8470`); the numeric-fidelity retry (`UP:8060-8077`) merged extra documents. Now they are the stream's own retrieval. That is the intended behaviour, and no test pins the old one.
3. **An explicit clarification setting still wins.** The retrieval step fills `enable_pre_retrieval_clarification` only when it is `None`, so a request or profile `False` turns the gate off, as today. A profile `True` stays on. Pinned by `[turned-off]`. Forcing the value on fails that case (measured: `assert [] == [True]`).
4. **The boundary still refuses for stages the stream no longer runs.** On `/search/stream`, `_rag_llm_providers` names `None` for suggestions, claims, post-verification and the other post-answer stages (`EP:215-220` after #3439). So an over-limit user on their own key with one of them on is still refused, though no such call happens now. This fails closed. Narrowing the stream's list would need its own drift reasoning (TASK-15427's classification), so it is left out.
5. **One answer call, so one usage row and one call-site check fewer per stream request.** Nothing assumed two; see Usage and quota effects. With the user's own key, their provider now bills one answer.
6. **The event order on a failure of the answer's provider or credential.** Base: only `[error]` (the retrieval step's answer fails first). With the fix: `contexts` and `reasoning`, then the same `error` with unchanged fields (see "The WebUI's event contract"). Knowledge QA already handles that sequence, and it matches the stream before `cd184c9fb9` and the agentic stream. Recommended for acceptance; the success path is unchanged.

## Files

| File | Change |
|---|---|
| `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py` | `:2390-2391`: an explicit `enable_generation=False` wins (same line count) |
| `tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py` | `_retrieve_standard_documents` (`:381`): sets `enable_pre_retrieval_clarification` from the request's `enable_generation` when it is unset |
| `tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py` | `test_rag_streaming_generation_provider_override` loses its provider stub. New: the helper `_stream_on_recorded_providers` and three tests (5 cases) |
| `tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py` | `test_unified_pipeline_coordinates_retrieval_only_result` gets two cases |
| `Docs/superpowers/plans/2026-11-07-rag-stream-one-answer.md`; `backlog/tasks/task-15482 - ...md` | this plan; the task |

## Test commands

- **Focused set** (the four files):
  `<pytest> -q -n 4 -rfE -k "not postgresql" tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py tldw_Server_API/tests/RAG/test_restricted_postgres_media_retrieval.py tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py`
  - Base (`7a020da14a` + `proto_15427.patch`): `140 passed` (35 `postgresql` cases deselected), about 40 s. The log has 2 lines: `...test_permission_first_auth_stream_without_preseeded_scope[sqlite-Rowan-True]` and `...test_owner_retrieves_chunked_media_through_stream_endpoint[sqlite]`, each `[('dns', 'api.openai.com')]`.
  - Prototype: `146 passed` (+6: the counting test, 2 effective-query cases, 2 clarification cases, one pipeline case), about 35 s, and the log is empty.
- **Wider set:** every test file that names a touched symbol. Save it as `<scratchpad>/wider_15482.sh` and run `zsh <scratchpad>/wider_15482.sh <label>` from the worktree root:

```zsh
#!/bin/zsh
S=<scratchpad>
P=<plans dir>
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
L=${1:-run}
git grep -l -E "unified_pipeline|unified_rag_pipeline|streaming_executor|stream_rag_events|search/stream|enable_pre_retrieval_clarification|_retrieve_standard_documents" \
  -- 'tldw_Server_API/tests/*.py' | grep '/test_[^/]*\.py$' | sort -u > $S/wider_$L.txt
echo "files: $(wc -l < $S/wider_$L.txt)"
rm -f $S/netprobe_wider_$L.log
PATH="$(dirname $PY):$PATH" TEST_DB_NAME=tldw_test_15482_$L PYTHONPATH=$P NETPROBE_LOG=$S/netprobe_wider_$L.log TLDW_TEST_NO_DOCKER=1 \
  $PY -m pytest -p no:cacheprovider -p netprobe_15427 -q -n 4 --dist loadfile -rfE -k "not postgresql" $(cat $S/wider_$L.txt) > $S/wider_$L.out 2>&1
echo "wider: $(grep -E '^=+ .*(passed|failed)' $S/wider_$L.out | tail -1)"
grep -E '^FAILED|^ERROR' $S/wider_$L.out | cut -c1-220
echo "netprobe attempts: $(cat $S/netprobe_wider_$L.log 2>/dev/null | wc -l)"
cut -c1-220 $S/netprobe_wider_$L.log 2>/dev/null
```

  Measured: 77 files, about 12 minutes each.
  - **Base** (`7a020da14a` + `proto_15427.patch`): `6 failed, 1702 passed, 29 skipped`. The log is empty.
  - **Prototype:** `7 failed, 1707 passed, 29 skipped`. The log is empty.
  - **Failures in both runs (5).** All are environment or order failures:
    - `test_rag_profiles.py::TestSlidesSourceLocalRerankers::test_flashrank_accepts_complete_non_listwise_local_bundle` fails alone on both (`RAGConfigurationError: FlashRank local model validation is unavailable`: no local FlashRank model).
    - Four cases of `test_restricted_postgres_media_retrieval.py` pass alone and in the focused set on both, so they fail only in this file order: `test_permission_first_auth_trash_restore_without_preseeded_scope[sqlite-X-API-KEY|sqlite-Authorization]` (`assert 404 == 200` on `GET /media/1`), `test_permission_first_auth_stream_without_preseeded_scope[sqlite-Rowan-True]`, and `test_failed_selected_media_read_emits_error_without_answer[sqlite]`. In this order the media row is not visible to them. That is also why this run's log misses the two leaking stream cases that the focused set shows.
  - **Only in one run.** Each passes alone on both codes: on the base, `test_rag_doc_researcher_api.py::test_rag_search_natural_language_media_fallback`; on the prototype, `test_retrieval_plan_usage.py::test_multi_database_retriever_plan_overrides_conflicting_legacy_namespace` (`RAGDatabaseError`, after an `AttributeError` in the media retriever) and `Evaluations/test_embeddings_abtest_retrieval.py::test_abtest_rejects_fts_search_mode`. Neither reaches the changed code: one calls `MultiDatabaseRetriever.retrieve_from_plan`, the other the A/B service.
  - **The arithmetic:** 1702 + 6 new = 1708. Two prototype-only failures and one base-only failure that passes leave 1707.
  - **The failure set varies run to run.** A review run measured `6 failed, 1708 passed, 29 skipped`, log empty: the five shared failures matched, the base-only case passed, the two prototype-only cases did not recur, and a new one appeared (`RAG_NEW/property/test_rag_properties.py::TestPipelineContextProperties::test_context_query_preservation`, `hypothesis.errors.FailedHealthCheck: Input generation is slow`). It passes alone, so it is load. Rule: a failure not in the baseline is re-run alone; one that passes alone is load. Hypothesis "too slow" health-check cases, such as that one, are known load cases.

---

## Task 0: Start TASK-15482, branch, draft PR, baseline

**Files:** `backlog/tasks/task-15482 - ...md` (status and notes); `Docs/superpowers/plans/2026-11-07-rag-stream-one-answer.md`.

- [ ] **Step 1: Branch after #3439.** `gh pr view 3439 --json state --jq .state` prints `MERGED`. Then `git fetch origin dev --quiet && git switch -c fix/rag-stream-one-answer-15482 origin/dev`. Check that the stub is there: `git grep -n "no_provider_call" -- tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py` prints 2 lines.
- [ ] **Step 2: Start the task.**

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-15482 -s "In Progress" \
  --append-notes "Plan: Docs/superpowers/plans/2026-11-07-rag-stream-one-answer.md. Fix: unified_rag_pipeline honours an explicit enable_generation=False over a resolved payload that says True (_resolve_effective_enable_generation, unified_pipeline.py:2390-2391), so the stream's retrieval step (streaming_executor.py:381, which already passes False) runs retrieval only on the shared request, which keeps the effective query; the step sets enable_pre_retrieval_clarification from the request's enable_generation when it is unset. Owner answer, Change summary: <answer>."
```

- [ ] **Step 3: Plan copy, commit, draft PR.**
  - Copy this plan to `Docs/superpowers/plans/2026-11-07-rag-stream-one-answer.md`, unchanged.
  - `git add Docs/superpowers/plans/2026-11-07-rag-stream-one-answer.md && git add -A backlog/tasks && git diff --cached --check`. It must print nothing.
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` must exit 0.
  - Commit (message 0).
  - `git push -u origin fix/rag-stream-one-answer-15482`.
  - `gh pr create --draft --base dev --title "fix(rag): the streaming search answers once; its retrieval step runs no generation (TASK-15482)"`. Give it a one-line body naming the task, ending with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- [ ] **Step 4: Baseline.** On the unchanged branch, run the focused set and the wider set (label `base`) under the probe plugin. Record both in the ledger. Expected: the focused base above. Any later failure is compared with this baseline; never dismiss one as environment-only without it.

## Task 1: The retrieval step runs retrieval only, and keeps the effective query and clarification (AC #1-#4)

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py:2390-2391`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py:381`
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py:143-204`

**Interfaces:**
- Consumes: `unified_rag_pipeline(**kwargs)`, with `enable_generation: bool = True` and `enable_pre_retrieval_clarification: Optional[bool] = None` (`UP:2088`, `:2093`). `chat_service.perform_chat_api_call_async(**call_kwargs)`, which is `GEN:500`'s call-time import. Its `stream` flag is `False` for the answer and suggestions, `True` for the streamed answer.
- Produces: the pipeline contract "an explicit `enable_generation=False` is retrieval only"; the test helper `_stream_on_recorded_providers(monkeypatch, client, payload) -> tuple[list[dict], list[dict]]` (events, provider-call kwargs).

- [ ] **Step 1: Remove the provider stub (AC #4).** In `test_rag_streaming_generation_provider_override`, delete the `chat_service` import, the `no_provider_call` stub and its `monkeypatch.setattr(chat_service, ...)` line. Keep `block_live_network()` and `assert attempts == []`.
- [ ] **Step 2: Write the route tests (AC #1-#3).** Insert them before `test_rag_streaming_emits_research_progress_before_generation`:

```python
def _stream_on_recorded_providers(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, payload: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stream one retrieved document through the real pipeline and generator; record each provider call."""
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

    calls: list[dict[str, Any]] = []

    class OneDocumentRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.retrievers: dict[str, Any] = {}

        async def retrieve(self, query: str, **kwargs: Any) -> list[Document]:
            return [
                Document(
                    id="doc-1",
                    content="Rowan directs the observatory.",
                    metadata={"title": "Observatory"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    async def provider_call(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if not kwargs.get("stream"):
            return {"choices": [{"message": {"content": "A discarded answer."}}]}

        async def chunks() -> Any:
            yield {"choices": [{"delta": {"content": "The streamed answer."}}]}

        return chunks()

    monkeypatch.setattr(rag_ep, "MultiDatabaseRetriever", OneDocumentRetriever)
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", provider_call)
    request = {"sources": ["media_db"], "enable_generation": True, "enable_cache": False, "enable_query_classification": False, **payload}
    with client.stream("POST", "/api/v1/rag/search/stream", json=request) as resp:
        assert resp.status_code == 200
        events = [json.loads(raw) for raw in resp.iter_lines() if raw]
    return events, calls
```

  The three tests, each with `(monkeypatch: pytest.MonkeyPatch, client_with_stream_overrides: TestClient, ...) -> None` and a docstring ending `(TASK-15482)`:
  - `test_rag_stream_makes_one_provider_call_the_streamed_answer`. Call the helper with `{"query": "Who directs the observatory?", "enable_suggestions": True}`. Then:

```python
    assert [call["stream"] for call in calls] == [True]
    assert [event["text"] for event in events if event["type"] == "delta"] == ["The streamed answer."]
    assert events[-1]["type"] == "complete"
```

  - `test_rag_stream_answers_the_effective_query`, parametrized on `extra` with ids `["spell-correction", "chat-history-reformulation"]`:
    - `{"query": "Who dirrects the observatory?", "spell_check": True}`;
    - `{"query": "Who directs it?", "chat_history": [{"role": "user", "content": "Tell me about the observatory."}]}`.

    Patch both `unified_pipeline.spell_check_query` and `query_classifier.reformulate_query` (both from `tldw_Server_API.app.core.RAG.rag_service`) with one `async def effective_query(*_args, **_kwargs) -> str` that returns `"Who directs the observatory?"`. The pipeline reads the first as a module global (`UP:3208`) and imports the second at call time (`UP:3416`). Then:

```python
    prompts = [call["messages"][-1]["content"] for call in calls if call["stream"]]
    assert len(prompts) == 1
    assert "Who directs the observatory?" in prompts[0]
    assert extra["query"] not in prompts[0]
```

  - `test_rag_stream_keeps_its_pre_retrieval_clarification`, parametrized on `("setting", "clarified")`, values `[({}, True), ({"enable_pre_retrieval_clarification": False}, False)]`, ids `["unset-follows-generation", "turned-off"]`. It uses the real heuristic gate: a pronoun with no chat history asks (`clarification_gate.py:44-51`). Call the helper with `{"query": "Who directs it?", **setting}`. Then:

```python
    if clarified:
        assert [event["type"] for event in events] == ["clarification", "delta", "complete"]
        assert calls == []
    else:
        assert [call["stream"] for call in calls] == [True]
```

- [ ] **Step 3: Write the pipeline case.** Parametrize `test_unified_pipeline_coordinates_retrieval_only_result` on `("payload_generation", "explicit")`, values `[(False, {}), (True, {"enable_generation": False})]`, ids `["payload-off", "explicit-off-beats-the-payload"]`. Set `resolved.payload["enable_generation"] = payload_generation`, and pass `**explicit` to `unified_rag_pipeline`. Its `fake_generation_phase` already raises `AssertionError("generation phase should not run")`, and the pipeline does not catch that.
  - **Optional (not in the prototype):** a failure-order test. A five-line variant of `_stream_on_recorded_providers` with a credential runtime whose `resolve` raises `ByokResolutionError("invalid_provider_credentials")`, asserting `[e["type"] for e in events] == ["contexts", "reasoning", "error"]`. Skip it if the review does not ask.
  - The one-call test's helper sets `"enable_query_classification": False`, so `SEARCH_QUERY_CLASSIFICATION=1` in the environment cannot add a non-stream call and break the exact `[True]` assertion.
- [ ] **Step 4: Run them (RED).**
  `<pytest> -q -rfE "tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py::test_rag_streaming_generation_provider_override" "tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py::test_rag_stream_makes_one_provider_call_the_streamed_answer" "tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py::test_rag_stream_answers_the_effective_query" "tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py::test_rag_stream_keeps_its_pre_retrieval_clarification" "tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py::test_unified_pipeline_coordinates_retrieval_only_result"`
  Expected: `4 failed, 4 passed` (measured, about 30 s):
  - `test_rag_streaming_generation_provider_override`: `AssertionError: assert [('dns', 'api.groq.com')] == []`;
  - `test_rag_stream_makes_one_provider_call_the_streamed_answer`: `assert [False, False, True] == [True]`;
  - `[turned-off]`: `assert [False, True] == [True]`;
  - `[explicit-off-beats-the-payload]`: `AssertionError: generation phase should not run`.

  The two effective-query cases, `[unset-follows-generation]` and `[payload-off]` pass on the base: they guard what the fix must keep. The log is empty, because the override test's own blocker hides its attempt.
- [ ] **Step 5: The pipeline honours an explicit `False`.** In `UP:2390-2391`, replace the two lines with exactly these two. One more line would grow the frozen `unified_rag_pipeline`.

```python
    def _resolve_effective_enable_generation() -> bool:  # an explicit False is retrieval only (TASK-15482)
        return bool(enable_generation) and bool((resolved_request.payload or {}).get("enable_generation", True))
```

- [ ] **Step 6: Run them.** The same command. Expected: `1 failed, 7 passed` (measured). `[unset-follows-generation]` fails with `assert ['contexts', ...', 'complete'] == ['clarificati...', 'complete']`: the gate now follows the retrieval step's `False`. That is AC #3's RED.
- [ ] **Step 7: Keep the stream's clarification.** In `_retrieve_standard_documents`, replace `SE:381` with these four lines (the old line gains a comment):

```python
    kwargs["enable_generation"] = False  # retrieval only: the stream makes the one answer call (TASK-15482)
    if kwargs.get("enable_pre_retrieval_clarification") is None:
        # Unset, the gate follows enable_generation: the request's, not this retrieval step's.
        kwargs["enable_pre_retrieval_clarification"] = bool(payload.get("enable_generation"))
```

  `payload` is the function's parameter, the stream's copy of the request payload (`SE:737`). Today's default was `bool(payload.get("enable_generation", False))` (`UP:2391` with the kwarg `False`), so this value is the same.
- [ ] **Step 8: Run them.** The same command. Expected: `8 passed`, empty log (measured).
  - **Mutation check for AC #2** (measured; revert it afterwards). Undo Step 5, and set `kwargs["resolved_request"] = dataclasses.replace(resolved_request, payload={**resolved_request.payload, "enable_generation": False})` in place of the `setdefault` at `SE:379`. That is the task's sketch. Expected: `3 failed, 5 passed`: both `test_rag_stream_answers_the_effective_query` cases (`assert 'Who directs the observatory?' in "...Question: Who dirrects the observatory?..."`) and the pipeline case `[explicit-off-beats-the-payload]` (`AssertionError: generation phase should not run`, because the mutation undoes Step 5, which that case pins). AC #1's test passes.
- [ ] **Step 9: Focused set.** Expected: `146 passed`, empty log (measured).
- [ ] **Step 10: Commit** (message 1): `git add tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py tldw_Server_API/tests/RAG_NEW/unit/test_standard_core_contract_threading.py && git diff --cached --check`.

## Task 2: Verify, close, review, PR

- [ ] **Step 1: Focused set** (probe plugin). Expected: `146 passed`, empty log.
- [ ] **Step 2: Wider set** with label `final`, compared with Task 0's base. Expected: the base's failures and +6 passed. The prototype measured `7 failed, 1707 passed, 29 skipped` against the base's `6 failed, 1702 passed, 29 skipped`; see Test commands for the failures, all environment or order, and for the rule that the exact failure set varies. Re-run any failure not in the base alone before judging it: one that passes alone is load (for example a Hypothesis "too slow" health check, such as `test_rag_properties.py::TestPipelineContextProperties::test_context_query_preservation`). Any failure that is not in the base, and that fails alone, is this plan's.
- [ ] **Step 3: CI checks.** Each expected result was measured on the prototype:
  - `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -n 4 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI`: `1497 passed, 6 skipped` (about 5 minutes). It includes the function-size ratchet. This was measured without the plan copy. With the copy in `Docs/superpowers/plans/`, `tests/Docs` alone gave `212 passed`, and the copy has no trailing whitespace.
  - `PATH="$(dirname $PY):$PATH" $PY -m pytest -p no:cacheprovider -q tools/backlog-py/tests`: `164 passed`.
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_shard_coverage.py`: `new_uncovered=0` and "OK — no newly-unshared test files."
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/test_quality_triage.py --enforce`: exit 0, "OK — no offenses beyond the baseline."
  - `PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/ci/route_auth_ratchet.py`: "208 unauthenticated routes; baseline exact."
  - `PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts`: exit 0, "OK — OpenAPI fingerprint matches the checked-in snapshot." and "OK — committed path union matches the canonical contract."
  - `/opt/homebrew/bin/uvx bandit -ll tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`: "No issues identified.", Medium 0, High 0.
  - `$PY -m ruff check` on the four touched files: the 35 `I001` findings `unified_pipeline.py` has on the base, and none new; the other three files are clean (measured on base and prototype).
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .`: exit 0.
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/scope_predicate_ratchet.py`: "28 conditional scope predicates; baseline exact."
  - `git diff origin/dev --check`: nothing.
- [ ] **Step 4: Close TASK-15482** (fill every `<...>` first):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-15482 \
  --append-notes "RED/GREEN (record the real output): base 4 failed, 4 passed (provider-override [('dns', 'api.groq.com')] == []; one-call [False, False, True] == [True]; clarification[turned-off] [False, True] == [True]; pipeline[explicit-off-beats-the-payload] 'generation phase should not run'); pipeline line only 1 failed, 7 passed (clarification[unset-follows-generation] streamed an answer); with the retrieval step's explicit clarification 8 passed. Mutations: the task's plain-copy sketch passes AC #1 and fails both effective-query cases (the prompt asks the raw query); forcing clarification on fails [turned-off] (assert [] == [True]). Focused set <base> -> <after>, probe log 2 lines (the two sqlite stream cases of test_restricted_postgres_media_retrieval.py, api.openai.com) -> 0. Wider set <base> -> <after>. tests/Docs + lint + CI <n> passed; backlog-py <n> passed; shard coverage, test-quality ratchet, route-auth ratchet, OpenAPI and the scope-predicate baseline unchanged. Not run: the postgresql cases (the shared PostgreSQL is down), incl. the two postgresql stream cases." \
  --append-notes "Decisions: unified_rag_pipeline honours an explicit enable_generation=False over a payload that says True (no other caller passes that pair; the 20 call sites in the plan), so no copy of the shared request is made and _sync_effective_query keeps writing the effective query where the stream reads it. Every post-answer stage is gated on generated_answer and is skipped; the stream's own claims overlay is heuristic (no LLM call). With post-verification on, the stream's contexts are its own retrieval, no longer an adopted rerun of the discarded answer. One answer call means one usage row and one call-site quota check fewer per stream request; the boundary check is unchanged; on /search/stream it still refuses an over-limit own-key user for post-answer stages that no longer run there (fail-closed, not filed)." \
  --append-notes "Docs: none (no doc describes the retrieval step's answer; the event contract is unchanged). Bandit: <result>." \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "/api/v1/rag/search/stream makes one answer call, the streamed one. Its retrieval step already asked for retrieval only, but unified_rag_pipeline preferred the resolved payload's enable_generation True (since cd184c9fb9), so it ran and discarded a full answer and every post-answer stage the request turned on. The pipeline now honours an explicit enable_generation=False; the retrieval step keeps the shared request, so the stream still answers the effective query (spell correction, chat-history reformulation), and it sets pre-retrieval clarification from the request's enable_generation when unset, so the stream still clarifies by default. test_rag_streaming_generation_provider_override passes its socket probe without a provider stub, and test_restricted_postgres_media_retrieval.py's stream tests reach no provider." \
  -s Done
```

  `--check-dod 3` holds: no doc needed a change (What was verified, Docs). Then `git grep -n -E "<(answer|base|after|n|result)>" -- 'backlog/tasks/task-15482 *'` must print nothing, `git add -A backlog/tasks && git diff --cached --check`, and commit (message 2).
- [ ] **Step 5: Review before the PR.** Run `/superpowers:requesting-code-review` in a clean subagent on `origin/dev..HEAD`, and give it this plan's Review Focus. Address the findings before the PR is marked ready, and record each ruling in the ledger.
- [ ] **Step 6: Finish the PR and merge.**
  - `git push`. Replace the draft's body with `gh pr edit <n> --body-file <file>`, then `gh pr ready <n>`.
  - The body states:
    - the four ACs and how each is met;
    - Option C and why (A/B in one line each);
    - the post-answer stages skipped;
    - the usage and quota effects;
    - the Review Focus items;
    - the failure-path event order: with the answer's provider or credential unusable, the stream now emits `contexts` and `reasoning` before the same `error` (base: only the `error`); it matches the stream before `cd184c9fb9` and the agentic stream;
    - the measured counts;
    - the `## Change summary` (or the waiver; Decision 1).
  - It ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - **E2E (not required; Knowledge QA streams):** once the PR's `Frontend E2E` journeys finish, diff its failed test ids with the latest `dev` run of the same workflow (`gh run list --branch dev --workflow <name> --limit 1`, then `gh run view <id> --log-failed`). A failure only on the PR in `knowledge-qa.spec.ts` or `research-workspace.spec.ts` is this PR's until shown otherwise. Most of their stream cases mock the route (`page.route("**/api/v1/rag/search/stream", ...)`, e.g. `knowledge-qa.spec.ts:487`).
  - **Merging:** check `gh variable get MERGE_QUEUE`. It was unset when this plan was written, and `dev` requires 8 statuses. If it is `on`, arm auto-merge instead (CLAUDE.md, "Merging into dev"). When reviews are settled and the Change summary is in, run `QUEUE="<n>:fix/rag-stream-one-answer-15482" sh <plans dir>/pr_queue3.sh` in the background. It reads dev's required checks from the ruleset, rebases server-side when BEHIND, and merges when all pass. Wait for each body edit's re-triggered license status before merging. Confirm MERGED before deleting the branch.

## Commit messages

Each ends with a blank line and `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

- 0: `chore(backlog): start TASK-15482 and add its plan`
- 1: `fix(rag): the streaming search's retrieval step runs no generation, so the stream answers once (TASK-15482)`. Body: the pipeline honours an explicit `enable_generation=False`; the shared request keeps the effective query; the retrieval step keeps the default clarification; the provider stub of the override test is gone.
- 2: `chore(backlog): close TASK-15482`

## Conflicts with open PRs

- **#3439** (TASK-15427) must merge first (Global Constraints, Base). This plan's prototype applies on top of `proto_15427.patch`.
- **#3394** ("UAT: fix workflow persistence, recovery, and account isolation", `CONFLICTING` with `dev`) also touches two of these files:
  - `unified_pipeline.py`: hunks at `:588` and `:2567-2629` (search-depth presets), not near `:2390`;
  - `test_restricted_postgres_media_retrieval.py`: +49, which this plan does not touch.

  No textual conflict is expected. Whichever lands second re-runs the stream tests.
- No other open PR touches these files. 41 PRs target `dev` (2026-11-07). The files checked: `streaming_executor.py`, `unified_pipeline.py`, `rag_unified.py`, `test_rag_stream_parity.py`, `test_restricted_postgres_media_retrieval.py`, `test_streaming_executor.py` and `test_rag_agentic_api.py`, with `gh api repos/rmusser01/tldw_server/pulls/<n>/files --paginate` per PR.

## Not measured

- **The PostgreSQL cases.** The shared server is down, so every `postgresql` case was deselected. Among them are `test_owner_retrieves_chunked_media_through_stream_endpoint[postgresql]` and `test_permission_first_auth_stream_without_preseeded_scope[postgresql-Rowan-True]`, the other two of the four leaking stream cases. Their path is the same as the sqlite cases' (the retrieval step's answer), and the sqlite cases are clean after the fix.
- **A real provider and real usage rows.** The "one row fewer" claim is from the call path (`gateway.py:1340-1357`). It is not from a usage-table count.
- **Post-verification's adaptive rerun.** That the stream's contexts no longer come from an adopted rerun is code reading (`UP:8249`, `:8466-8471`); no test drove it.
- **The Frontend E2E journeys**, until the PR runs them (Task 2 Step 6).

## Review fold (2026-11-07)

The independent review (`review_plan_15482.md`: ready with fixes, 0 Blocker, 0 Important, 6 Minor) is folded in. The pre-fold copies are `plan_15482.prefold.md` and `proto_15482.prefold.patch`.

- **M1** (mutation check count): Task 1 Step 8 now expects `3 failed, 5 passed`, naming `[explicit-off-beats-the-payload]` as the third.
- **M2** (flaky set varies): Test commands, after "The arithmetic", gives the rule (re-run a non-baseline failure alone; passes alone means load) and names the Hypothesis "too slow" case `test_rag_properties.py::TestPipelineContextProperties::test_context_query_preservation`; Task 2 Step 2 repeats the rule.
- **M3** (failure-path order): "The WebUI's event contract" has the measured change; Review Focus 6 and the Step 6 PR-body list say so; Task 1 Step 3 holds the optional failure-order test as an explicit optional bullet.
- **M4** (Review Focus 2 incomplete): "State the stream reads" and Review Focus 2 name the numeric-fidelity retry and the `security_filter`/`source_status` overwrite by an adopted rerun.
- **M5** (environment-dependent one-call test): the helper's `request` dict sets `"enable_query_classification": False` in the plan text and in `proto_15482.patch` (same line, so the hunk counts are unchanged); a note in Task 1 Step 3 says why.
- **M6** (caller census): "Who else is affected" adds the two indirect call sites (`SE:394`, `shared_workspace_chat_service.py:607`); the conclusion stands.
