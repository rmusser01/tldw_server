# The study assistants run on the user's stored key (TASK-15426, PR A of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The flashcards and quizzes study assistants run their reply on the user's own stored key for the reply's provider, and their boundary names that provider to `enforce_llm_quota`, so an over-limit user whose reply runs on their own key is answered unmarked while every other user gets the same 402 at the same point. Adds the two helpers PRs B-D reuse.

**Architecture:** One helper resolves "the user's own key" everywhere in TASK-15426: `users_own_key_credentials(user_id, provider, model=None)` asks `users_own_key_serves` (the brief's definition: BYOK on, a provider that bills per key and reaches its vendor, allowlisted, an active key), and only then builds a `ProviderCredentialRuntime` for the user alone (no team or org scope, no base-URL override, the admin override snapshot as on `/chat/completions`), resolves the provider, closes the runtime and returns the runtime-issued handle when its `credential_source` is `user`, else None. None means "keep the server's key and today's call path, byte for byte". `enforce_llm_quota_on_own_key(user_id, provider, model=None)` resolves that handle first and passes `own_key=handle is not None` to `enforce_llm_quota`, so the boundary asks once and admits an over-limit user, unmarked, only when the handle was issued. The study-assistant routes swap `Depends(require_llm_quota)` for a dependency that resolves the reply's target as the reply does and returns the handle; the routes pass it to `generate_study_assistant_reply`, which hands it to `perform_chat_api_call_async` as `_provider_call_credentials` (the gateway already binds it, `gateway.py:821-889`).

**Tech Stack:** Python 3.12, FastAPI, pytest (asyncio auto mode, xdist), backlog-py.

**Spec:** `backlog/tasks/task-15426 - Use-the-user-s-stored-key-for-media-analysis-the-study-assistants-watchlist-summaries-evaluation-runs-and-benchmarks.md` (AC #1 answered by the owner, ACs #2-#3 for this family); ADR-083 (`Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md`) and its 2026-11-06 amendment; the planner brief `brief_plan_15426.md`; `Docs/superpowers/plans/2026-11-05-own-key-server-key-boundaries-close-out.md` (the pins this PR flips).

**Measured on:** `origin/dev` at `cb70f0c5d1` (2026-11-07, the merge of #3400). Prototype: `proto_15426_a.patch` (plans dir; 15 files, +240/-39, of which 5 docs files; applies to `cb70f0c5d1`). Every `file:line` below is for `cb70f0c5d1`. Review fold (2026-11-07): the prototype was regenerated (15 files, +292/-41; it applies to `7a020da14a`, and the four applied in order give the fold's final tree, checked) on `7a020da14a` (dev one merge later, #3432, which touches none of this PR's files) and every count the fold changed was re-measured there (see "Review fold" at the end); the `file:line` citations still hold.

**Order:** first of four (cheapest first, owner 2026-11-07). PRs B, C and D build on the two helpers and the test helper this PR adds; each is a separate plan (`plan_15426_b.md`, `_c.md`, `_d.md`) and only D closes TASK-15426. The benchmark judge fix (`TASK-<JB>`, planned and filed separately, its own PR) lands before PR B (owner, 2026-11-07); it does not touch this PR.

---

## The split (all four PRs)

| PR | Families | Why this cut | Depends on | Size (prototype) |
|---|---|---|---|---|
| A | flashcards and quizzes study assistants | One provider per reply, request-scoped, the gateway already binds a handle; it adds the shared helpers. | - | 15 files, +240/-39 (fold: +292/-41) |
| B | benchmark runs (both routes) and watchlist output summaries | Both run inline in the request and call `Summarization_General_Lib.analyze`, which already takes `provider_credentials`; one provider per request. Merged: the review surface is two thin route edits on the same helper. | A, and `TASK-<JB>` first | 15 files, +162/-60 (fold: +211/-63) |
| C | evaluation runs | The run's judge provider lives in the stored evaluation and the run config, and the calls run in a task that outlives the request. | A | 12 files, +159/-25 (fold: +228/-27) |
| D | media analysis (7 endpoints) and the media ingest job | About ten ingestion libraries call `analyze` without a key; the handle rides the LLM call context instead of a parameter through each. | A | 20 files, +253/-56 (fold: 21 files, +429/-75) |

B also waits for `TASK-<JB>` (the benchmark judge call never reaches the provider, so every benchmark item scores the 0.5 default; fixed first, in its own PR, owner 2026-11-07). B, C and D touch disjoint source files, so they can be built and reviewed in parallel once A has merged (their prototypes stack on A only through the helpers; their docs hunks stack A→B→C→D on the same three docs, so whichever merges later rebases its docs hunk). Merges are serial anyway (`pr_queue3.sh`, MERGE_QUEUE unset).

## What was verified (on `cb70f0c5d1`)

| # | Question | Finding | Consequence |
|---|---|---|---|
| V1 | How does `/chat/completions` build its runtime and pass the handle? | A `ProviderCredentialRuntime` per request from the request's trusted scope (`chat.py:3697-3709`); `enforce_llm_quota(..., own_key=...)` with the provider asked once (`chat.py:419-426`, `:4432`); the handle goes on every call as `_provider_call_credentials` (`chat.py:5135`). Character chat resolves, calls, and closes the runtime in `finally` (`character_chat_sessions.py:6824-6866`, `:6941-6943`). | The same runtime and the same kwarg; the boundary asks once. |
| V2 | The study assistants today | `dependencies=[Depends(require_llm_quota)]` (`flashcards.py:2466`, `quizzes.py:784`) checks before the route body, naming no provider. `generate_study_assistant_reply` resolves the target with `resolve_chat_target` (`core/Flashcards/study_assistant.py:457`) and calls `perform_chat_api_call_async` with no key or handle (`:460-468`), so the server's key. | Name the target's provider; pass the handle. |
| V3 | Does the gateway take a handle? | `_build_adapter_request_from_chat_args` binds `_provider_call_credentials` (`gateway.py:821-825`) and puts it on the adapter request (`:888-889`); the check at the call reads it (`call_credential_source`, `gateway.py:1345`; `llm_call_context.py:81-101`). | A handle for the user's key is recorded `user` and not checked. |
| V4 | Can a handle be used after its runtime closes? | `bind_provider_call_credentials` checks only that the handle was runtime-issued and for this provider (`adapter_utils.py:57-105`, `provider_credential_runtime.py:322-338`); the handle holds its key (`:179-262`). The repo's own tests use a handle from a closed runtime (`tests/LLM_Calls/test_summarization_runtime_credentials.py`, `_issue_credentials`). After `close()` the handle cannot be refreshed or `mark_used`. | Resolve once and close: no runtime lifetime to manage in routes, tasks or jobs (Decision 2). |
| V5 | What "the user's stored key" means | `users_own_key_serves` (`usage_quota_deps.py:54-88`) is the brief's definition. | The helper asks it first, so a user without such a key never builds a runtime. |
| V6 | The RED on the base | With this PR's tests and no source change (measured): 11 failed, 6 passed: the three helper tests (`AttributeError`: no `users_own_key_credentials`), `[own-key]` and `[server-key]` of both endpoint tests (the boundary asks about nothing, so `asked == []`; the own-key reply gets the 402), and `[own-key]` of the adapter test for both routes on SQLite and PostgreSQL (the reply runs on the server's key). `[no-usable-provider]` and the adapter test's `[server-key]` cases pass on the base, as they must. | |
| V7 | Frozen functions (`tests/lint/function_size_baseline.json`) | None of this PR's functions is listed. | No constraint. |
| V8 | Open PRs (41 open on `dev`, file lists fetched 2026-11-07) | #3394 [`c6f023f9d8`, already CONFLICTING with dev] edits `quizzes.py` at `:935` only (`generate_quiz`); #3269 [`81b0bb5836`] shows a `flashcards.py` import hunk at `:13-19`, but that change is already on dev (its base is stale), so nothing new overlaps. No open PR touches `usage_quota_deps.py`, `study_assistant_deps.py`, `study_assistant.py`, `tests/helpers/llm_quota.py`, the docs, or this PR's tests except #3269's `test_flashcards_endpoint_integration.py` hunk at `:5698` (far from `:3323`). TASK-15427's plan touches `rag_unified.py` and RAG tests only. | Either order. |

## Decisions

1. **ADR-004 Change summary.** AI-authored. **Answered (owner, 2026-11-07): "Yes, waive for all"**, covering every TASK-15426 PR. The PR body records it as a full `## Change summary` section.
2. **One helper, resolve once, close at once (ruling, all four PRs).** `users_own_key_credentials` closes the runtime as soon as the handle is issued. Alternatives: hold the runtime open around the calls (chat's shape) needs a lifetime in a route body, a run's task and a job, for two things these features do not use: refreshing an OAuth key after a 401, and stamping the key's `last_used_at` (`mark_used`). `ponytail:` comment in the helper names both and the upgrade (hold it open). Where each family gets its handle: `/chat/completions` scopes a runtime to the request (built from the request's trusted scope, closed when the response ends, V1). Here the study assistants (A), benchmark runs and watchlist summaries (B) and the media endpoints (D) resolve at the request, before any LLM work. An evaluation run (C) resolves at the request too and hands the handle to the task that outlives it, in memory only. A media ingest job (D) has no request: it resolves at its start from the job's owner (`current_llm_user_id()`, the scope `_handle_job` sets from the job row). Resolving at each call was rejected: a database read per call, and a half-done run or job could move onto the server's key. A key revoked mid-run or mid-job is used until the run or job ends (**owner decision, 2026-11-07: keep**; PRs C and D add the BYOK-guide sentence: a run or job already started keeps using the key; cancel it to stop). The cost of not calling `mark_used`: the key's `last_used_at` is not updated by these features' calls, so Task 3 adds that to the BYOK guide's "Auditing and Updates" line (review A-6); their usage rows still record the calls as the user's.
3. **Only the user's own key; everyone else unchanged (ruling).** The runtime gets no team or org scope and no base-URL override, and the helper returns a handle only for `credential_source == "user"`. A user without such a key gets None and the feature keeps its server-key path exactly as today: no runtime, no handle, no new failure mode (a credential-store outage, the admin override snapshot) for them. Chat also uses an active team's or org's key; those keys count anyway, and the task is about the user's own key. **Owner decision, 2026-11-07: no team or org keys.** (Had they been wanted: pass the request's scope, `derive_trusted_credential_scope`, and return any non-`server_default` handle; that widens the change to server-key users.)
4. **A failed resolution keeps the server's key (ruling).** If the lookup said yes but the runtime fails (store down, policy denial, the key deleted in between), the helper logs a warning and returns None. For an over-limit user, `enforce_llm_quota_on_own_key` then refuses (own_key False), so nothing runs past the limit; an under-limit user runs on the server's key as today.
5. **The boundary stays a dependency (ruling).** `study_assistant_own_key(payload, current_user)` replaces `require_llm_quota` in the same position (FastAPI resolves it before the body), shares the body with `get_study_assistant_guidance` as that dependency already does (`study_assistant_deps.py:14-18`), resolves the target with the same `resolve_chat_target` call, and returns the handle. A target that cannot be resolved (`ChatConfigurationError`, `ByokResolutionError`) names none: an over-limit user gets the 402 before the reply's 400, as today. Resolution runs twice per reply (here and in `generate_study_assistant_reply`); both are local and pure.
6. **A key revoked mid-reply.** One call per reply: the handle resolved at the boundary is used once. Nothing to decide here (PRs C and D decide it for runs and jobs).
7. **Docs.** `Usage_Quotas.md`, the BYOK guide and ADR-083 change in this PR; ADR-083 gets a new dated amendment section that PRs B-D extend with one bullet each (appending, never rewriting the 2026-11-06 text). ADR-083 and the BYOK guide are mirrored: `bash Helper_Scripts/refresh_docs_published.sh` in the same commit.

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
- `$PY` is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`; `<plans dir>` is `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`.
- `proto_15426_a.patch` is normative for docstrings and comments the plan does not quote; the docs prose is quoted in Task 3.
- Every test run in this plan runs with outbound connections blocked: `PYTHONPATH=<plans dir> NETPROBE_LOG=<file> $PY -m pytest -p netprobe_15427 ...` (an empty log means no attempt), or inside `block_live_network()`.
- ruff: no new finding on a touched file against the base (`ruff check` on each touched file, and on `git show origin/dev:<file>` through `--stdin-filename`); the CI ruff step is non-blocking, the Definition of Done is not.
- `scope_predicate_baseline.txt` lists none of this PR's files (checked).
- `git diff --cached --check` before each commit (no trailing whitespace).
- Compatibility: `require_llm_quota` stays (other callers may use it); `generate_study_assistant_reply` keeps its signature plus one keyword. No setting, schema or route path changes.

## Review Focus

1. **An over-limit user without a stored key for the reply's provider, sending a valid request.** Expected: the same 402 body before the route body, as before, and the assistant never called. Pinned by `[server-key]` of both endpoint tests (Task 2). An invalid body now gets FastAPI's 422 before the 402: `study_assistant_own_key` reads the body, and FastAPI does not run a dependency whose body fails validation (the decorator dependency it replaces answered the 402 first). No LLM work runs either way.
2. **A user under the limit with a stored key.** Expected: the reply runs on their key (the request carries the runtime handle, recorded `user`), not the server's. Pinned by `test_a_reply_runs_on_the_users_stored_key_else_the_servers[own-key]`, which goes through the real routes, the real target resolution and the real gateway to a terminal adapter (Task 2).
3. **A request whose provider or model cannot be resolved.** Expected: an over-limit user gets the 402 (not the 400) and no lookup runs. Pinned by `[no-usable-provider]` of both endpoint tests (Task 2).
4. **The credential store fails after the lookup said yes.** Expected: no handle; an over-limit user is refused, an under-limit user keeps the server's key. Pinned by `test_a_failed_resolution_keeps_the_servers_key` and `test_enforce_on_own_key_asks_once_and_admits_only_with_a_handle` (Task 1).
5. **A stored key for another provider.** Expected: the server's key and no handle for this reply. Pinned by `test_the_users_own_key_credentials_are_a_runtime_handle_for_that_key_only` (`anthropic` → None) and `[server-key]` of the adapter test (Task 1, Task 2).

## Files

- Modify: `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py` (new `users_own_key_credentials` after `users_own_key_serves` `:54-88`; new `enforce_llm_quota_on_own_key` after `enforce_llm_quota` `:91-118`)
- Modify: `tldw_Server_API/app/api/v1/API_Deps/study_assistant_deps.py` (new `study_assistant_own_key`; imports)
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py` (`:19-20` imports, `:2463-2489` route), `tldw_Server_API/app/api/v1/endpoints/quizzes.py` (`:12-13` imports, `:781-807` route)
- Modify: `tldw_Server_API/app/core/Flashcards/study_assistant.py` (`generate_study_assistant_reply`, `:428-482`)
- Modify tests: `tldw_Server_API/tests/helpers/llm_quota.py` (new `users_own_key`), `tldw_Server_API/tests/Usage/test_llm_quota_decision.py`, `tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py` (`:3323-3361`), `tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py` (`:1746-1781`), `tldw_Server_API/tests/Flashcards/test_study_assistant_adapter_contract.py`
- Docs (Task 3): `Docs/Operations/Usage_Quotas.md`, `Docs/User_Guides/Server/BYOK_User_Guide.md`, `Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md`, and the two `Docs/Published` copies
- Backlog: TASK-15426 (status, AC #1, notes)
- Create (Task 0): `Docs/superpowers/plans/2026-11-07-own-key-study-assistants.md` (this plan)

## Test commands

`$FOCUSED_A`:

```text
tldw_Server_API/tests/Usage/test_llm_quota_decision.py::test_the_users_own_key_credentials_are_a_runtime_handle_for_that_key_only
tldw_Server_API/tests/Usage/test_llm_quota_decision.py::test_a_failed_resolution_keeps_the_servers_key
tldw_Server_API/tests/Usage/test_llm_quota_decision.py::test_enforce_on_own_key_asks_once_and_admits_only_with_a_handle
tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py::test_flashcard_assistant_respond_names_its_provider_to_the_llm_quota
tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py::test_quiz_attempt_question_assistant_respond_names_its_provider_to_the_llm_quota
tldw_Server_API/tests/Flashcards/test_study_assistant_adapter_contract.py::test_a_reply_runs_on_the_users_stored_key_else_the_servers
```

```bash
PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q -rf -n 6 $FOCUSED_A
```

Measured: RED (tests only, on `cb70f0c5d1`) 11 failed, 6 passed (4:51); GREEN 17 passed (3:29), with the first A patch; the later fixes (the helper's base-URL-free resolution, the existing stubs' keyword) touch no focused test's outcome; with them, the focused tests and the four files they live in pass on the final prototype even with a leaked `MOCK_OPENAI_BASE_URL` (the wider-set recheck below; the quizzes file then 77 passed). The adapter-contract test runs each case against SQLite and the local PostgreSQL the `study_database` fixture provides.

**The wider set** (every test file that names a touched symbol or the changed test helper):

```bash
git grep -l -E 'helpers\.llm_quota|usage_quota_deps|study_assistant_deps|generate_study_assistant_reply|/assistant/respond' -- 'tldw_Server_API/tests/*.py' | grep '/test_[^/]*\.py$'
```

34 files on `cb70f0c5d1`. Measured as one run of the union of the four PRs' wider sets (183 files: 176 with `-n 8 --dist loadfile`, the 7 that take a PostgreSQL fixture serially with a session-unique `TEST_DB_NAME`), on `cb70f0c5d1` and on the four prototypes stacked. Base: 3 failed, 3751 passed, 67 skipped, 1 xfailed, 1 xpassed (44:38); PostgreSQL 99 passed, 59 skipped. The base failures: `test_media_processing.py::TestProcessPdfs::test_process_pdf_upload_success[docling]` and `::test_process_pdf_multi_status_mixed[docling]` (fail on both trees), and `test_evaluations_stage1_route_and_error_regressions.py::test_response_quality_capacity_failure_drains_admitted_siblings_before_runtime_close` (flaky: passes with the prototype). Prototype (a run started before the last fixes): 51 failed, 3733 passed, 65 skipped, 3 errors (53:08); PostgreSQL 108 passed, 63 skipped. Its new failures were, and are fixed in the patches: every own-key test (a `MOCK_OPENAI_BASE_URL` left by another test in the same worker made the test helper's handle `user_unmetered`: the helper's resolution now carries no base URL), eight existing study-assistant stubs and one evaluation-identity stub (see PR A Task 2, PR C Task 1); 22 in `test_email_offline_ingestion.py` were load (the file passes alone on both trees, 42 passed on the base), and the 3 errors were environment (a PostgreSQL teardown lock, a server-e2e health-check timeout, a fixture finalizer). Every file that failed, re-run on the final prototype with `MOCK_OPENAI_BASE_URL=http://127.0.0.1:9` set for the whole run: 531 passed, 30 skipped, 1 failed (a quizzes stub, fixed after: the quizzes file then 77 passed).

---

## Task 0: Status, plan, branch, draft PR, baseline

- [ ] **Step 1: Branch** from `origin/dev`: `git fetch origin dev && git switch -c feat/own-key-study-assistants-15426 origin/dev`.
- [ ] **Step 2: TASK-15426 In Progress, AC #1 answered** (backlog-py only):

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-15426 -s "In Progress" --check-ac 1 \
  --append-notes "AC #1 answered (owner, 2026-11-07): all four families use the user's stored key, built cheapest first: (a) the flashcards and quizzes study assistants; (b) benchmark runs and watchlist output summaries; (c) evaluation runs; (d) media analysis (the /media endpoints) and media ingest jobs. ADR-004 Change summary waived for every TASK-15426 PR (owner, 2026-11-07: \"Yes, waive for all\")." \
  --append-notes "Plan, PR A of 4 (study assistants): Docs/superpowers/plans/2026-11-07-own-key-study-assistants.md."
```

  `sed -n '/AC:BEGIN/,/AC:END/p' "backlog/tasks/task-15426 - Use-the-user-s-stored-key-for-media-analysis-the-study-assistants-watchlist-summaries-evaluation-runs-and-benchmarks.md"` shows `- [x] #1`.
- [ ] **Step 3: Plan file** at `Docs/superpowers/plans/2026-11-07-own-key-study-assistants.md` (this plan). `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .`: exit 0. `git add -A backlog/tasks Docs/superpowers/plans && git diff --cached --check`; commit `docs(plan): own-key study assistants (TASK-15426, 1 of 4)`.
- [ ] **Step 4: Draft PR**, `--base dev`, title `feat(usage): the study assistants run on the user's stored key (TASK-15426, 1 of 4)`.
- [ ] **Step 5: Baseline** the wider set on the branch (`-n 8 --dist loadfile`, under the outbound blocker; run its PostgreSQL-parametrized files serially, below); record counts, every failing id and every netprobe line. Expected: no failure in the 34 files (none of the union run's three base failures is in this set). A failure outside that is checked against a recent dev run before it is called flaky. Known, pre-existing and identical on the PR tree, so not a regression: two tests in this set attempt live DNS on dev, which the blocker stops: `RAG_NEW/.../test_rag_stream_parity.py::test_rag_streaming_generation_provider_override` (Groq; on the brief's known list) and `Prompt_Management_NEW/integration/test_prompt_improvement_api.py::test_llm_quota_refusal_answers_402_not_retryable_provider_unavailable` (`api.openai.com` x2; owned by the live-network-tests task the benchmark-judge PR files). Under the DNS-blocking probe `test_security.py::...test_ssrf_protection_private_ip` fails as a probe artifact (it passes under the connect-only `netblock_13956`).

## Task 1: The own-key helpers (shared by all four PRs)

**Files:** `usage_quota_deps.py`, `tests/helpers/llm_quota.py`, `tests/Usage/test_llm_quota_decision.py`.

**Interfaces:**
- Produces: `async def users_own_key_credentials(user_id: Any, provider: str | None, *, model: str | None = None) -> Any` (a runtime-issued `ProviderCallCredentials` whose `credential_source == "user"`, or None).
- Produces: `async def enforce_llm_quota_on_own_key(user_id: Any, provider: str | None, *, model: str | None = None) -> Any` (raises `LLMQuotaExceededError` like `enforce_llm_quota`; returns the handle or None).
- Produces (tests): `users_own_key(monkeypatch, provider="openai", api_key="sk-users-own-key") -> list[str]` in `tests/helpers/llm_quota.py`: `usage_quota_deps.users_own_key_serves` answers True for `provider` only and records each provider asked; `provider_credential_runtime.resolve_byok_credentials` returns `ResolvedByokCredentials(provider, api_key, app_config={section: {"model": <the snapshot's model for that section>}}, credential_fields={}, source="user", allowlisted=True)` for `provider` and a user id (`section` from `PROVIDER_APP_CONFIG_KEYS`), and the real resolution for anything else. No base URL on purpose: measured, a first version copied the whole server snapshot, and in the wider xdist run a `MOCK_OPENAI_BASE_URL` left in the environment by another test in the same worker made every own-key handle `user_unmetered`, so all own-key tests failed there while passing alone.

- [ ] **Step 1: The test helper.** Add `users_own_key` to `tests/helpers/llm_quota.py` above `assert_llm_quota_402` (the module already imports `Any` and `pytest`; the local imports are sorted, so ruff reports no `I001`):

```python
def users_own_key(monkeypatch: pytest.MonkeyPatch, provider: str = "openai", api_key: str = "sk-users-own-key") -> list[str]:
    """The user stored their own key for ``provider``; returns each provider a boundary asks about.

    The quota's lookup answers True for ``provider`` and False for every other, and the credential
    runtime resolves that key for any user id (source ``user``), with the server's model for it and no base
    URL: a base URL another test left in the environment (``MOCK_OPENAI_BASE_URL``) would make it
    ``user_unmetered``. Every other provider resolves as before.
    """
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.app.core.Provider_Credentials import provider_credential_runtime as runtime_module
    from tldw_Server_API.app.core.Provider_Credentials.byok_config import PROVIDER_APP_CONFIG_KEYS
    from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ResolvedByokCredentials

    asked: list[str] = []
    real_resolver = runtime_module.resolve_byok_credentials

    async def _own_key(_uid: object, name: str) -> bool:
        """Only ``provider`` runs on the user's own key."""
        asked.append(name)
        return name == provider

    async def _resolve(name: str, **kwargs: Any) -> Any:
        """The user's key for ``provider``; the real resolution for everything else."""
        if name != provider or kwargs.get("user_id") is None:
            return await real_resolver(name, **kwargs)
        section = PROVIDER_APP_CONFIG_KEYS[name]
        model = ((kwargs.get("server_config_snapshot") or {}).get(section) or {}).get("model")
        return ResolvedByokCredentials(
            provider=name, api_key=api_key, app_config={section: {"model": model}}, credential_fields={},
            source="user", allowlisted=True,
        )

    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    monkeypatch.setattr(runtime_module, "resolve_byok_credentials", _resolve)
    return asked
```
- [ ] **Step 2: Write the failing tests** at the end of `tests/Usage/test_llm_quota_decision.py`:

```python
async def test_the_users_own_key_credentials_are_a_runtime_handle_for_that_key_only(monkeypatch) -> None:
    """A handle for the user's own key, recorded as theirs; None for any other provider and for none (TASK-15426)."""
    from tldw_Server_API.app.core.Provider_Credentials.provider_credential_runtime import (
        is_runtime_issued_provider_call_credentials,
    )
    from tldw_Server_API.tests.helpers.llm_quota import users_own_key

    asked = users_own_key(monkeypatch, "openai")
    handle = await usage_quota_deps.users_own_key_credentials(1, "openai")
    assert is_runtime_issued_provider_call_credentials(handle, provider="openai")
    assert (handle.api_key, handle.credential_source) == ("sk-users-own-key", "user")
    assert await usage_quota_deps.users_own_key_credentials(1, "anthropic") is None
    assert await usage_quota_deps.users_own_key_credentials(1, None) is None
    assert asked == ["openai", "anthropic"]
```

  `test_a_failed_resolution_keeps_the_servers_key`: `users_own_key(monkeypatch, "openai")`, then patch `provider_credential_runtime.resolve_byok_credentials` with a coroutine raising `ByokResolutionError("credential_store_unavailable", provider)`; `assert await usage_quota_deps.users_own_key_credentials(1, "openai") is None`.

  `test_enforce_on_own_key_asks_once_and_admits_only_with_a_handle`: `refuse_llm_work`, `asked = users_own_key(monkeypatch, "openai")`; in a fresh task (`asyncio.create_task`, as the neighbouring tests do), `enforce_llm_quota_on_own_key(1, "openai")` returns a handle with `api_key == "sk-users-own-key"` and leaves `current_llm_call_context().prechecked` False; for `"anthropic"` and for `None` it raises `LLMQuotaExceededError`; `assert asked == ["openai", "anthropic"]` (one lookup per call, none for None).
- [ ] **Step 3: Run them.** `$PY -m pytest -q <the three ids>`. Expected: 3 failed, `AttributeError: module ... has no attribute 'users_own_key_credentials'` (and `enforce_llm_quota_on_own_key`).
- [ ] **Step 4: Implement** both helpers in `usage_quota_deps.py` (as below; the prototype is the same code). `users_own_key_credentials`: return None when `not provider` or `not await users_own_key_serves(user_id, provider)`; otherwise, in one `try` that logs `logger.bind(exception_type=...).warning("Own-key credential resolution failed")` and returns None on any `Exception` (`# noqa: BLE001`), build `ProviderCredentialRuntime(user_id=as_quota_user_id(user_id), team_ids=None, org_ids=None, trusted_base_url_override=False, override_snapshot_resolver=capture_provider_override_call_snapshot)` (both imported inside the function, as `users_own_key_serves` does), `handle = await runtime.resolve(provider, model=model)` with `await runtime.close()` in `finally`; return `handle if handle.credential_source == USER_KEY_SOURCE else None`. Put this `ponytail:` comment above the `try`:

```python
    # ponytail: resolved once and closed, so the handle is never refreshed (an OAuth key that expires mid-job
    # fails its calls) and never marks the key used; hold the runtime open around the calls if either matters.
```

  `enforce_llm_quota_on_own_key`: `handle = await users_own_key_credentials(user_id, provider, model=model)`, then `await enforce_llm_quota(user_id, provider=provider, own_key=handle is not None)`, return `handle`. Docstrings (the prototype's): `users_own_key_credentials`, "A runtime-issued credential handle for the user's own stored key for ``provider``, or None (TASK-15426). None unless ``users_own_key_serves`` holds, so a user without such a key keeps the server's key and the call path it had. So does a resolution that fails or no longer finds the key. The runtime is closed once the handle is issued: the handle keeps its key for the caller's calls."; `enforce_llm_quota_on_own_key`, "``enforce_llm_quota`` for a request whose LLM calls all run on ``provider``, with the handle for the user's own key for it (``users_own_key_credentials``) as its answer; returns that handle, or None for the server's key (TASK-15426). An over-limit user is let through, unmarked, only when the handle was issued."
- [ ] **Step 5: Run them.** Expected: 3 passed. Also `tldw_Server_API/tests/Usage/test_llm_quota_decision.py` whole: all pass (35 passed, measured inside the serial socket-probe run below).
- [ ] **Step 6: Commit** `feat(usage): resolve the user's own key once for a boundary (TASK-15426)`.

## Task 2: The study assistants run on it (ACs #2, #3)

**Files:** `study_assistant_deps.py`, `flashcards.py`, `quizzes.py`, `core/Flashcards/study_assistant.py`, the three test files.

**Interfaces:**
- Consumes: `enforce_llm_quota_on_own_key` (Task 1), `users_own_key` (Task 1).
- Produces: `async def study_assistant_own_key(payload: StudyAssistantRespondRequest, current_user: User = Depends(get_request_user)) -> Any` in `study_assistant_deps.py`; `generate_study_assistant_reply(..., provider_credentials: Any = None)`.

- [ ] **Step 1: Flip the pins.** Replace `test_flashcard_assistant_respond_refuses_an_over_quota_user` (`test_flashcards_endpoint_integration.py:3323-3361`) with:

```python
@pytest.mark.parametrize(
    ("provider", "stored", "asks"),
    [("openai", "openai", ["openai"]), ("openai", "anthropic", ["openai"]), ("not-a-provider", "openai", [])],
    ids=["own-key", "server-key", "no-usable-provider"],
)
def test_flashcard_assistant_respond_names_its_provider_to_the_llm_quota(
    client_with_flashcards_db: TestClient, monkeypatch, provider: str, stored: str, asks: list[str],
):
    """Past a limit, a reply whose provider runs on the user's own key is answered, unmarked; on the server's
    key, or with no usable provider, it gets the 402 and the assistant never calls the LLM (TASK-15426)."""
```

  Body: create the card as today; the recording stub appends `{**kwargs, "prechecked": current_llm_call_context().prechecked}` and returns `{"assistant_text": "Filtration.", "structured_payload": {}, "provider": "openai", "model": "gpt-test"}`; `refuse_llm_work(monkeypatch)`; `asked = users_own_key(monkeypatch, stored)`; post `{"action": "explain", "provider": provider, "model": "gpt-test"}`. Then:

```python
    assert asked == asks
    if stored == provider:
        assert response.status_code == 200, response.text
        assert [(call["provider_credentials"].api_key, call["prechecked"]) for call in calls] == [("sk-users-own-key", False)]
    else:
        assert_llm_quota_402(response)
        assert calls == []
```

  Replace `test_quiz_attempt_question_assistant_respond_refuses_an_over_quota_user` (`test_quizzes_endpoint_integration.py:1746-1781`) the same way (`test_quiz_attempt_question_assistant_respond_names_its_provider_to_the_llm_quota`, the attempt fixture as today). The model is explicit (`gpt-test`) so the target resolves without a server default.

  The routes now always pass `provider_credentials=` (None for a server-key user), so every existing stub of `generate_study_assistant_reply` with an explicit signature gains `provider_credentials: Any = None,` after `guidance`: six `fake_generate_reply` in the flashcards file; three `fake_generate_reply`, `raise_input_error` and `raise_db_error` in the quizzes file (the `**_kwargs` stubs need nothing). Measured: without it eight existing tests (respond persists, 400 input error, 409 conflict, fact check) answer 500 (`TypeError: unexpected keyword argument 'provider_credentials'`), and the quizzes DB-error test passes for the wrong reason (the `TypeError` maps to the same 500).
- [ ] **Step 2: The end-to-end test.** In `test_study_assistant_adapter_contract.py`, before `test_study_version_conflict_still_stops_before_dispatch_or_new_messages`:

```python
@pytest.mark.integration
@pytest.mark.parametrize(
    ("stored", "api_key", "source"),
    [(True, "sk-users-own-key", "user"), (False, "synthetic-test-key", None)],
    ids=["own-key", "server-key"],
)
def test_a_reply_runs_on_the_users_stored_key_else_the_servers(
    study_route, provider_boundary, monkeypatch, stored, api_key, source
):
    """The reply's provider call carries a runtime-issued handle for the user's own key when they stored one;
    without one it keeps the server's key and no handle (TASK-15426)."""
    from tldw_Server_API.app.core.Usage import llm_quota_gate
    from tldw_Server_API.app.core.Usage.llm_call_context import call_credential_source
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_calls, users_own_key

    if stored:
        users_own_key(monkeypatch, "openai")
        # Over the limit, so a call without the handle is refused at the call; the fixture's user override sets
        # no content scope, so the check is told the user.
        refuse_llm_calls(monkeypatch)
        monkeypatch.setattr(llm_quota_gate, "current_llm_user_id", lambda: 1)
    payload = {"action": "follow_up", "message": "Which day?", "provider": "openai", "model": "remote-explicit"}
    response = study_route.client.post(study_route.path + "/respond", json=payload)
    assert response.status_code == 200, response.text
    [(provider, request)] = provider_boundary.calls
    assert provider == "openai"
    assert (request["api_key"], call_credential_source(request)) == (api_key, source)
```

  `[own-key]` runs over the limit end to end (review A-4): the boundary admits it unmarked, and the gateway's check at the call (`check_llm_call_async(call_credential_source(kwargs))`, `gateway.py:1345`) must skip the call because it carries the handle. The `current_llm_user_id` line is needed: `refuse_llm_calls` alone left the check idle in this fixture (no LLM user), measured by dropping the handle at the call, which still answered 200 on the server's key; with the line, that sabotage answers the 402 at the call (2 failed, `assert 402 == 200`), and the real code passes. Also update the `study_route` fixture comment (`:227`) to name `study_assistant_own_key`.
- [ ] **Step 3: Run them.** `$FOCUSED_A` minus Task 1's ids. Expected on unchanged source: `8 failed, 6 passed` (the `[own-key]` cases fail: 402 or the server's key; `asked == []`).
- [ ] **Step 4: Implement.**
  - `generate_study_assistant_reply(..., guidance=None, provider_credentials: Any = None)`: import `PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY` from `provider_credential_runtime` with the function's other local imports, and add `**({PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: provider_credentials} if provider_credentials is not None else {}),` as the last argument of the `perform_chat_api_call_async` call. Docstring: the handle runs the call on its key; without one the server's.
  - `study_assistant_own_key` in `study_assistant_deps.py` (imports `enforce_llm_quota_on_own_key`, `ChatConfigurationError`, `resolve_chat_target`, `ByokResolutionError`, `typing.Any`, sorted):

```python
    try:
        target = resolve_chat_target(requested_provider=payload.provider, requested_model=payload.model)
    except (ChatConfigurationError, ByokResolutionError):
        return await enforce_llm_quota_on_own_key(current_user.id, None)
    return await enforce_llm_quota_on_own_key(current_user.id, target.provider, model=target.model)
```

  - Both routes: delete `dependencies=[Depends(require_llm_quota)],` and the `require_llm_quota` import; import `study_assistant_own_key` beside `get_study_assistant_guidance`; add `own_key: Any = Depends(study_assistant_own_key),` right after `payload` (so it resolves first, as the decorator dependency did); pass `provider_credentials=own_key` to `generate_study_assistant_reply`.
- [ ] **Step 5: Run** `$FOCUSED_A`. Expected `17 passed (3:29)` (fold, on `7a020da14a` with PostgreSQL down: 13 passed, 4 skipped, the four PostgreSQL cases; 0 netprobe attempts). Then the two endpoint files and the adapter-contract file whole: no new failure against Task 0's baseline (fold: the adapter-contract file and `test_llm_quota_decision.py` whole, 78 passed, 30 skipped, all PostgreSQL; 0 attempts).
- [ ] **Step 6: Network.** Wrap one run of the six ids in the socket probe (`block_live_network()` from a `conftest`-free one-off: `$PY - <<'EOF' ... EOF` that runs `pytest.main([...])` inside the context) and confirm `attempts == []`. Measured on the prototype: all of $FOCUSED_A and the rest of the focused sets of PRs B-D, run serially on the stacked prototype under the outbound-connect blocker (`-p netblock_13956`, plans dir on `PYTHONPATH`): 131 passed, `netblock: 0 blocked outbound connect(s)`. The endpoint tests stub `generate_study_assistant_reply`; the adapter test stops at the terminal adapter.
- [ ] **Step 7: Commit** `feat(flashcards,quizzes): study assistants run on the user's stored key (TASK-15426)`.

## Task 3: Docs, verification, close-out notes, review, PR

- [ ] **Step 1: Docs** (text in the prototype; one commit with the mirror):
  - `Docs/Operations/Usage_Quotas.md`: in "An over-limit user on their own key", after the document-generation bullet add `  - the flashcards and quizzes study assistants, for the reply's provider (the request's `provider` and `model`, else the server's default), on which the reply then runs;`; in the "other boundaries" sentence drop `the flashcards and quizzes study assistants, `; and after the paragraph that ends `Those calls count, so the refusal is the limit doing its job.` (still inside the bullet, before `- **How the cost keys compare.**`) add the paragraph that starts `  The features TASK-15426 moved onto a stored key (the study assistants above) run on the user's own key whenever it serves, over a limit or not: ...` (PRs B-D extend its parenthesis).
  - `Docs/User_Guides/Server/BYOK_User_Guide.md` (Usage Quotas): add `the flashcards and quizzes study assistants, ` to the list of endpoints that answer an over-limit own-key request; replace the sentence that starts `Media analysis, the flashcards and quizzes study assistants, watchlist summaries,` with `The study assistants use your own key for the reply's provider whenever you stored one, so those replies are recorded as yours and do not count. Media analysis, watchlist summaries, evaluation runs and benchmark runs never use a key you stored:` (the rest of that sentence unchanged).
  - The BYOK guide's "Auditing and Updates" bullet that says `last_used_at` updates on successful provider calls (`BYOK_User_Guide.md:207`) gets a second sentence (review A-6; PRs B-D extend its parenthesis): `Calls by the features that resolve your key once, before their calls (the study assistants), do not update it; their usage is still recorded as yours.`
  - ADR-083: append (`<date>` is the commit date):

```markdown

## Amendment (<date>): the features that run on the user's stored key (TASK-15426)

TASK-15426 moves the features that the 2026-11-06 amendment lists under "Some boundaries never run on a stored key" onto the user's stored key, one family per PR (owner decision, 2026-11-07: all four families). For each family listed below, when `users_own_key_serves` holds for the feature's provider (BYOK on, a provider that bills per key and reaches its vendor, allowlisted, and an active key of the user's), the feature resolves that key through `ProviderCredentialRuntime` and passes the issued handle on every LLM call it makes, so the calls are recorded `credential_source = 'user'` and the per-user limits neither count nor check them. Only the user's own key is used, never a team or org key. A user without one keeps the server's key and the call path the feature had. The boundary names the feature's provider to `enforce_llm_quota` with the handle as its answer, so an over-limit user is let through, unmarked, only when the handle was issued; every other user gets the same 402 at the same point.

- **The flashcards and quizzes study assistants** (<date>): the reply's provider, resolved as the reply resolves it; a request with no usable provider names none, so an over-limit user gets the 402 first, as before.
```

  - `bash Helper_Scripts/refresh_docs_published.sh`; `git status --short Docs/Published` shows exactly the two mirrored files.
- [ ] **Step 2: Wider set** with the PR (same command as Task 0 Step 5). Expected: Task 0's counts plus the new tests (measured: no failing id that is not on the base list or explained above (load, environment), once the fixes in this plan are in), no new failing id.
- [ ] **Step 3: CI checks** (all measured on the prototype stack, measured on the four prototypes stacked (A-D): shard coverage `new_uncovered=0`; triage `OK — no offenses beyond the baseline` (after two fixes the first run found: PR C's runner test split so no test stacks four monkeypatch calls, PR D's `analyze` test asserting `analyze`'s own output); route-auth `208 unauthenticated routes; baseline exact.`; OpenAPI fingerprint and path union both OK (no route signature change is visible: the study-assistant dependency shares the body and the user dependency); `check_backlog_task_ids.py` exit 0; `tests/Docs tests/lint tests/CI tools/backlog-py/tests` 1657 passed, 6 skipped, 4 failed on the first run: the layering ratchet (`services -> api.v1.API_Deps` 19 > 18, fixed in PR D), the triage ratchet (the two fixes above), and two that failed on load (a land-gates helper subprocess timeout, an xdist worker crash in the route-map lint), which pass alone (37 passed with the triage ratchet); after the fixes `tests/Docs` with the layering, function-size, route-map, land-gates and triage-ratchet files: all pass but the two load-sensitive ones, which pass alone; Bandit `-ll` on A's five source files: no issues):
  - `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 8 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI tools/backlog-py/tests`
  - `$PY Helper_Scripts/ci/check_shard_coverage.py` → `new_uncovered=0`; `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` → OK
  - `PYTHONPATH=. $PY Helper_Scripts/ci/route_auth_ratchet.py` → baseline exact; `PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` → both OK (the dependency swap adds no parameter: the body is shared and `current_user` was already a sub-dependency)
  - `/opt/homebrew/bin/uvx bandit -ll` on the five source files → no issues
  - ruff on every touched `.py` file against the base (Plan-specific constraint) → no new finding (fold, measured on the A+B stack: every touched file has the same findings as on `7a020da14a`; the prototype's `I001` in `tests/helpers/llm_quota.py`, review A-2, is gone with Task 1 Step 1's sorted imports)
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` → exit 0; `scope_predicate_baseline.txt` lists none of the files.
- [ ] **Step 4: (no filing.)** The prompt-improvement live-DNS test (review A-3) is owned by the live-network-tests task the benchmark-judge PR files (`TASK-<NT>`, owner 2026-11-07); this PR files nothing. Controller ruling 2026-11-09.
- [ ] **Step 5: Close-out notes** (TASK-15426 stays In Progress; AC #2 and #3 are checked by PR D):

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-15426 \
  --append-notes "PR A (<PR url>): the flashcards and quizzes study assistants run on the user's own key for the reply's provider (users_own_key_credentials + enforce_llm_quota_on_own_key in usage_quota_deps.py; study_assistant_own_key replaces require_llm_quota on both routes). Over a limit: own key admitted unmarked, server key and no usable provider refused with the same 402. Tests: <focused counts>; wider set <counts>. Bandit: no issues. Docs: Usage_Quotas.md, BYOK guide (including last_used_at), ADR-083 amendment (mirrored). The prompt-improvement live-DNS test is left to the live-network-tests task (filed by the benchmark-judge PR)."
```

- [ ] **Step 6: Review.** Clean-session `/superpowers:requesting-code-review` on the branch diff, with this plan and the task file; address every finding; re-run the focused set.
- [ ] **Step 7: PR body** (ends with the attribution line): Summary; `## Change summary` (the owner's waiver of 2026-11-07, "Yes, waive for all", recorded in full as the section); Test plan with the measured counts; Review Focus. Mark ready. Before merging, diff the PR's Frontend E2E failures against a recent dev run (not required checks). Merge through `<plans dir>/pr_queue3.sh` (MERGE_QUEUE unset; dev requires 8 statuses).

## Review fold (2026-11-07)

Folded from `review_plan_15426_ab.md` (A: 0 Blocker, 0 Important, 6 Minor) and the owner's decisions of 2026-11-07. The pre-fold texts are `plan_15426_a.prefold.md` and `proto_15426_a.prefold.patch`. The fold was measured on `7a020da14a` with outbound connections blocked (`-p netprobe_15427`: 0 attempts in every run); PostgreSQL at 127.0.0.1:5432 was down, so the four PostgreSQL cases of the adapter-contract test (and the file's other PostgreSQL cases) were skipped, not run.

| Finding / decision | Where it landed |
|---|---|
| A-1: steps pointed to the prototype for text | Task 1 Step 1 quotes `users_own_key`; Task 1 Step 4 quotes both docstrings; Task 3 Step 1 quotes the ADR amendment; Plan-specific constraint: the patch is normative for any unquoted docstring or comment. |
| A-2: the prototype's `users_own_key` imports failed ruff `I001` | Sorted in the patch and in Task 1 Step 1; Task 3 Step 3 checks ruff against the base (measured: no new finding on any A or B file). |
| A-3: garbled baseline line; two pre-existing live-DNS tests not listed | Task 0 Step 5 rewritten with both tests as known (and the DNS-probe SSRF artifact); the prompt-improvement one is filed as `TASK-<F_A>` (Task 3 Step 4, new); notes and Files updated. |
| A-4: no over-limit end-to-end test through the real gateway | Task 2 Step 2 and the patch: `[own-key]` runs with `refuse_llm_calls` and the gate told the user (the fixture sets no scope; without that line the check stayed idle, measured). Sabotage (handle dropped at the call): 2 failed with the 402 at the call; real code: 13 passed, 4 skipped (PostgreSQL). |
| A-5: an invalid body now gets 422 before the 402 | Review Focus 1 ("sending a valid request", and the 422 note). No code change. |
| A-6: `last_used_at` is not updated by these features | Decision 2 and Task 3 Step 1: the BYOK guide's "Auditing and Updates" bullet gets the clause (patch updated; PRs B-D extend its parenthesis). |
| Owner: a run or job keeps the key resolved at its start | Decision 2 records it; the BYOK sentence lands in PRs C and D. |
| Owner: no team or org keys | Decision 3 records it. |
| Owner: the benchmark judge fix lands before B in its own PR (`TASK-<JB>`) | "Order" and the split note; nothing in this PR changes. |
| Owner: a raw `api_key` keeps counting; the other server-key paths stay out of scope | Recorded in PRs B-D (not this PR's surface). |
