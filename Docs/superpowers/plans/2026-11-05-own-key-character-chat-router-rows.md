# Own-key character chat, and the auto-router's usage row (TASK-13956, PR A of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Character chat (a reply and a roleplay resume) lets an over-limit user through, unmarked, when the reply's provider runs on their own key, and answers the 402 (not a 502) when such a reply is then refused at the call; the auto-router's usage row records the router call's own `credential_source` on `/chat/completions`, character chat and the shared route resolver; and a router call refused at the call that leaves no route answers the 402, not a 503, while every other routing error keeps its status.

**Architecture:** `enforce_llm_quota(user_id, provider=...)` (TASK-13955) is the only seam. Character chat moves its check from before the context build to just after provider resolution (below the auto-router, as `/chat/completions` already is), through one helper that names the reply's provider unless the turn may start memory extraction, which runs on the server's key. The resume names its saved provider. `_execute_character_provider_call` lets `LLMQuotaExceededError` through instead of rewrapping it as `502 Chat provider error`. `log_model_router_usage` gains an optional `credential_source`; each caller passes the source of the router call it just made (the runtime handle's on chat and character chat; on the resolver, which passes a raw key, the `billed_credential_source` of its BYOK resolution, which also marks the router call's context). A new `enforce_llm_quota_if_the_router_was_refused` (in `usage_quota_deps.py`) runs the check only when routing found candidates and its router call was refused by the quota, so that case answers the 402 on both endpoints and nothing else changes status.

**Tech Stack:** Python 3.12, FastAPI, pytest (asyncio auto mode, xdist), loguru.

**Spec:** `backlog/tasks/task-13956 - Own-key-LLM-calls-the-router-s-usage-row-and-the-other-quota-boundaries.md` (ACs #1 for character chat, #3); the planner brief `brief_plan_13956.md` (plans dir); ADR-083 (`Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md`); the executed plan `Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md` (TASK-13955, its Decision 3 table and Decision 8).

**Measured on:** `origin/dev` at `484e393333` (2026-11-05, the merge of #3420; it changed only `.github/workflows/backend-required.yml` and one backlog task file since `6e2dc0c2dc`, so every `file:line` below holds on both), with `Docs/Operations/Usage_Quotas.md` taken from PR #3422's head `9791e52407` (read-only; #3422 rewrites the "Streams without a usage chunk, and router calls" bullet this PR appends to). In a detached worktree, reverted afterwards. The prototype is `proto_13956_a.patch` (plans dir; relative to `484e393333` with that `Usage_Quotas.md`; 11 files, +578/-17; code, tests and the `Usage_Quotas.md` edit, not the plan copy or the task file). Its code and test hunks apply to `484e393333` alone. Every RED/GREEN count below was measured on that patch: the tests on unchanged code first, then the code.

**Order:** after PR #3422 (TASK-13952) merges. Both edit the same one-line bullet of `Usage_Quotas.md`; this plan's docs step is anchored on the text as it reads after #3422 (its anchor sentence survives #3422's edit, so it also applies before, but #3422 would then conflict).

**The split (3 PRs, serial):**

| PR | Plan | Scope | Closes TASK-13956 |
|---|---|---|---|
| A | this plan | character chat (reply, resume), the router's row on the three router callers, the no-route 402 | no |
| B | `plan_13956_b.md` | `enforce_llm_quota` with several providers; the judge-model evaluations, chat document generation, RAG | no |
| C | `plan_13956_c.md` | the boundaries whose calls never use a stored key (media, the ingest job, study assistants, watchlists, evaluation runs, benchmarks): pins, ADR-083 amendment, AC amendment, follow-up, close-out | yes |

A first (character chat is the primary UI surface). B's code is disjoint from A's, so B can be built while A is in review, but its docs step edits the same `Usage_Quotas.md` bullet as A, so B rebases on `dev` after A merges. C goes last: it closes the task and its ADR amendment describes A and B.

---

## What was verified (on `6e2dc0c2dc`; the cited files are unchanged on `484e393333`)

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| V1 | The router's row carries no source, so it counts | `log_model_router_usage` (`LLM_Calls/routing/accounting.py:51-89`) calls `log_llm_usage` without `credential_source`, which `log_llm_usage` takes (`Usage/usage_tracker.py:365`); a NULL source counts (`AuthNZ/repos/usage_repo.py:106`). Callers: `chat.py:1783-1810`, `character_chat_sessions.py:1175-1198`, `Chat/chat_route_resolver.py:202-222`. Reproduced: the three new row tests fail on base with `None != 'user'` and `TypeError: log_model_router_usage() got an unexpected keyword argument 'credential_source'`. | fix |
| V2 | Where each router call's key comes from | `/chat/completions`: a runtime handle (`chat.py:1687-1691`, kept in `router_credential_handle`) passed under `PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY` (`:1721`); the handle's `credential_source` is already billed with its `app_config` (`Provider_Credentials/provider_credential_runtime.py:956`). Character chat: the same, a handle per call (`character_chat_sessions.py:1143-1166`). The resolver: `resolve_byok_credentials` and a raw key (`chat_route_resolver.py:181-198`), under `llm_call_context(records_itself=True)` with no `credential_source`, so its check at the call (`LLM_Calls/gateway.py:1345`, `call_credential_source`) sees None: an over-limit user's own-key router call is refused there today. Its only caller is prompt improvement (`Prompt_Management/prompt_improvement_dispatch.py:118`), whose own reply call already marks its context with `billed_credential_source(provider, byok.source, byok.app_config, load_server_config_snapshot())` (`:205-210`). | fix |
| V3 | Character chat checks before the provider is known | Reply: `enforce_llm_quota(current_user.id)` at `character_chat_sessions.py:6989`; the provider is resolved after the context build and the auto-router, `:7222-7286`. Resume: the check at `:6764`, the saved provider at `:6768-6769`, nothing in between makes an LLM call. Reproduced: an over-limit user whose provider runs on their own key gets the 402 on both (the two admit tests fail on base with `assert 402 == 200`). | fix |
| V4 | Every LLM call a character chat turn makes | The router (`:1154`, a handle), the reply (`:7423`, a handle), the resume's reply (`:6825`, a handle), and automatic memory extraction: opt-in per chat (`settings.characterMemoryExtraction.enabled`), started for a saved reply (`:7982`, inside the persist branch) as a background task (`:3778-3863`) that calls `extract_character_memories(api_endpoint=provider)` with no key (`Character_Chat/modules/character_memory_extraction.py:224-235`, `chat_orchestrator.chat_api_call`): the server's key. The background task inherits the request's context (`mark_llm_quota_prechecked` is a plain set). Nothing between `:6989` and `:7286` makes an LLM call: the auto summary is deterministic (`:4257`), the world book and memory injection read the database. | design |
| V5 | A router call refused at the call can leave no route | After the move, an over-limit user's router call on a server key is refused at the call; `select_llm_router_choice` turns the exception into "no choice" and records its class name (`"error": "LLMQuotaExceededError"`, `LLM_Calls/routing/runtime.py:218-236`), which both endpoints keep under `routing_debug["llm_router"]` (`chat.py:1928`, `character_chat_sessions.py:1308`); `chat.py`'s router call re-raises the quota error as itself (`:1734-1737`), and character chat's does not catch it (`:1142-1173`). `route_model` returns None when the request's `routing.failure_mode` is `"error"` (`routing/service.py:95-96`), so character chat would answer `503 auto_routing_failed` (`:7239-7252`). `/chat/completions` already does, since its check runs after routing (`chat.py:3725-3738`, check at `:4426`). Reproduced: `test_a_refused_router_call_that_leaves_no_route_still_gets_the_402` fails on base with `{"detail":{"error_code":"auto_routing_failed",...}}`; with the character chat move but without the guard, the character chat twin fails the same way. The guard runs only when routing found candidates and the router's error is `LLMQuotaExceededError` (review A M2): no candidates keeps `400 auto_routing_no_candidates`, and a router call that failed for another reason (an outage, an unusable answer) keeps the 503, for every user. | fix |
| V6 | Frozen functions | `tests/lint/function_size_baseline.json`: `character_chat_completion` 1100 (measured 1100 on base, 1099 with the prototype), `create_chat_completion` 2908 (2905, 2906). Not frozen: `_complete_roleplay_resume` (191, 192), `_select_auto_character_llm_router_choice` (107, 111), `_execute_character_provider_call` (90, 92), `chat._select_auto_chat_llm_router_choice` (211, 212), `chat_route_resolver._select_auto_llm_router_choice` (91, 103), the new `_enforce_character_reply_quota` (12) and `enforce_llm_quota_if_the_router_was_refused` (9). No ceiling changes. | constraint |
| V7 | The refusal other users see | The body is `LLMQuotaExceededError` (`core/exceptions.py:2215-2240`), unchanged. Pinned before and after by `test_complete_v2_refuses_an_over_quota_user_before_any_llm_call` (stream and not) and `test_protected_resume_generation_enforces_llm_quota` (three limits): an over-limit user on a server key gets the same 402, and the provider is never called. | pin |
| V8 | Open PRs on the same files (re-checked 2026-11-05 ~12:00 with `gh pr list --base dev` and `gh api .../pulls/<n>/files --paginate` for all 38 open PRs; heads in brackets) | `Usage_Quotas.md`: #3422 [`9791e52407`] rewrites the "Streams without a usage chunk, and router calls" bullet, one physical line that Task 3's edit 5 appends to: this PR goes after #3422 (Order above); the anchor sentence survives #3422's edit. `character_chat_sessions.py`: #3394 [`82a7534446`], #3269 [`81b0bb5836`], #3071 [`e4482bd198`], #3020 [`332a95cd28`, patch too large for the API], #2869 [`3a9d1b61c6`]. `chat.py`: #3269, #3071, #854 [`2e71b56400`]. `test_chat_completion_precheck.py`: #3394 (appends at the end of the file; this PR inserts mid-file). `function_size_baseline.json`: #3394 (drops an unrelated entry). No changed line in those hunks names an anchor this plan uses (`enforce_llm_quota`, `usage_quota_deps`, `routing_decision is None`, `candidate_count`, `log_model_router_usage`, `estimated=usage`, `effective_completion`, `router_credential_handle`, `raise_detached_error`, `except ChatAPIError as e:`, `_maybe_trigger_character_memory_extraction`). None touches `usage_quota_deps.py`, `chat_route_resolver.py`, `routing/accounting.py` or the other test files. #3269 and #3071 also grow `create_chat_completion`: whichever lands second re-measures it against its ceiling. Every step below anchors on text, not on a line. | note |
| V9 | A reply refused at the call becomes a 502 (review A I1) | `_execute_character_provider_call` (`character_chat_sessions.py:554-643`) catches `_CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS`, which lists `HTTPException`; `LLMQuotaExceededError` is one, so the gate's refusal at the call is rewrapped as `502 {"detail": "Chat provider error"}`. Unreachable before this PR (an over-limit user was refused at the boundary, an admitted one is marked). Once an over-limit user is admitted unmarked on the lookup's word, a reply whose handle then bills another key (a base URL stored with the key, a stored row that does not extract) is refused at the call. Reproduced: `test_complete_v2_answers_402_when_an_admitted_own_key_reply_is_refused_at_the_call` (stream and not) fails after the move with `{"detail":"Chat provider error"}`; the WebUI treats any 5xx as an unreachable model (`chat-model-readiness.ts`) and has a billing-error parser for the 402 (`utils/billing-error.ts`). `/chat/completions` already passes this 402 through. Both the reply and the resume call through this helper. | fix |
| V10 | A refused request can write the auto summary (review A M1) | With the check after the context build, `_apply_auto_summary_to_prompt_messages` can persist the deterministic summary into the chat settings (`_persist_auto_summary_to_settings`, `:4089`) before the 402, when auto-summary is on and the chat is past its threshold (reviewer's probe P4). No LLM call; idempotent; the next admitted turn writes the same summary. | note |

## Decisions

1. **ADR-004 Change summary.** This PR is AI-authored. **Answered (owner, 2026-11-05):** the waiver covers all three TASK-13956 PRs, so the PR body records it as a full `## Change summary` section (Task 3 Step 7).
2. **Memory extraction counts the turn (ruling, fail closed).** A saved reply in a chat with automatic memory extraction on may start an extraction call on the server's key. The brief's rule is that a boundary counts when not every LLM call runs on the user's own key, so `_enforce_character_reply_quota` passes no provider for such a turn and an over-limit user is refused as before. It refuses every such turn, not only the turns that reach the extraction interval (the counter is per process and not read at the boundary); a `ponytail:` comment names the ceiling. Alternatives: let the reply through and let extraction be refused at its call (it is a background task that logs and gives up: the user loses that extraction silently); read `_extraction_message_counters` to refuse only on the interval turn (racy across workers). A streamed reply or `save_to_db: false` starts no extraction, so it is admitted.
3. **The check moves below the router (as the task says).** For an over-limit user with `model: "auto"`, the router call runs before the check and is checked at the call (it carries its handle): on the user's own key it runs, on another key it is refused and routing falls back to rules, as on `/chat/completions` (`test_a_refused_router_call_is_not_a_provider_failure_and_the_request_gets_the_402`). The check then asks about the provider the router chose (`test_complete_v2_asks_about_the_routed_provider`). Consequence: request errors checked between the old and the new position now come first for an over-limit user on character chat (`400` for a `directed_character_id` that is not a participant, the persist-under-sync conflict, routing's `400 auto_routing_no_candidates`); a valid request gets the same 402. A refused request may also have written the chat's deterministic auto summary first (V10).
4. **The no-route 402, on both endpoints, only for a refused router call (ruling; review A M2).** When routing found candidates but no decision, `enforce_llm_quota_if_the_router_was_refused(user, routing_debug)` runs `enforce_llm_quota(user)` with no provider only if the router's recorded error is `LLMQuotaExceededError`: an over-limit user gets the 402, and the request otherwise keeps the 503. No candidates keeps the 400. So every existing status holds for every other case: `/chat/completions`' `400 auto_routing_no_candidates` for an over-limit user, and the 503 for an own-key user whose router call failed for another reason, which a guard at the top of `if routing_decision is None:` turned into 402s. Every branch raises, so no spend is admitted. Pinned by `test_an_over_limit_user_keeps_the_routing_error_when_no_router_call_was_refused` (no decision: 503; no candidates: 400). Alternative: the reviewer's simpler move of the unconditional guard inside `if candidate_count > 0:` (keeps the 400, but still answers an over-limit own-key user's router outage with a 402). `/chat/completions` had the same gap since TASK-13955 (V5); it is one line in `create_chat_completion` (2905 to 2906, under its 2908 ceiling), so it is fixed here rather than filed.
5. **The resolver's source reads its endpoint.** The brief suggests `billed_credential_source(provider, byok_resolution.source)`. That form reads only the provider's default host, so a user key whose provider the operator points at its own server (`OPENAI_API_BASE_URL`, `config.txt`) would read `user`. The plan uses prompt improvement's form, `billed_credential_source(provider, source, byok_resolution.app_config, load_server_config_snapshot())` (V2): the adapter takes a raw key, so its base URL is the `app_config`'s or else the environment's. Pinned by the `own-key-on-operator-endpoint` case (`user_unmetered`).

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
- `$PY` is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (write it out; shell variables do not survive between agent calls).
- `scope_predicate_baseline.txt` lists none of this PR's files (checked on `484e393333`; re-check in Task 3).
- This plan is committed in the PR, so it must have no trailing whitespace; `git diff --cached --check` before each commit.
- Compatibility: `log_model_router_usage` gains one optional keyword (`credential_source=None`); `usage_quota_deps` gains `enforce_llm_quota_if_the_router_was_refused`; nothing else changes signature. No setting, schema, route or OpenAPI change.
- The E2E line above, as it holds today (checked 2026-11-05 with `gh api repos/rmusser01/tldw_server/rulesets/21824526`): `E2E Critical (Tier 1 + Journeys)` is one of the 7 required contexts on `dev`, and it runs the `journeys` project, which holds the character chat journeys (`character-chat.spec.ts`: a native turn and the roleplay resume through `/complete-v2`; `character-chat-phase6.spec.ts`; `character-chat-phase7-readiness.spec.ts`). The non-required jobs of the "Frontend E2E Tiers" workflow are `E2E Features (Tier 2 + 3)` (holds `tier-2-features/characters.spec.ts`), `E2E Admin (Tier 4 + 5)` and `E2E UX Regression (notes/chat)`. The workflow runs only on `pull_request` and `workflow_dispatch`, so there are no `dev` runs to diff against (the last is from 2026-07-09); Task 3 Step 9 diffs against recent PR runs.

## Review Focus

Inputs most likely to bite a user, most likely first, each with the test that pins it:

1. **An over-limit user on a server key, character chat, any request shape.** Expected: the same 402 body, the provider never called. Pinned by `test_complete_v2_refuses_an_over_quota_user_before_any_llm_call` (stream and not), `test_protected_resume_generation_enforces_llm_quota` (three limits), and, for `model: "auto"` with `routing.failure_mode: "error"`, `test_complete_v2_answers_402_when_the_refused_router_call_leaves_no_route` and its `/chat/completions` twin `test_a_refused_router_call_that_leaves_no_route_still_gets_the_402`; any other lost route keeps its 503 or 400 (`test_an_over_limit_user_keeps_the_routing_error_when_no_router_call_was_refused`). Task 2.
2. **An over-limit user on their own key with memory extraction on.** Expected: a saved reply is refused (extraction runs on the server's key), and no request leaves the machine even if that regresses; a reply with `save_to_db: false` is admitted. Pinned by `test_complete_v2_still_refuses_an_own_key_turn_that_may_extract_memories` (extraction stubbed, `extractions == []`) and `test_complete_v2_admits_an_over_limit_user_whose_reply_runs_on_their_own_key` (unsaved). Task 2.
3. **Admitted on their own key, the reply must run unmarked, and a call refused there is the 402** (so any call on another key is refused at its call, and the user sees their limit, not a broken provider). Pinned by `admitted == [False]` in the two admit tests and by `test_complete_v2_answers_402_when_an_admitted_own_key_reply_is_refused_at_the_call` (stream and not). Task 2.
4. **A router call on a user key whose provider the operator points at its own server.** Expected: `user_unmetered`, so it counts and is checked. Pinned by the `own-key-on-operator-endpoint` case of `test_the_router_call_and_its_usage_row_carry_whose_key_it_used`. Task 1.
5. **The WebUI's character chat journeys** (#3191 broke character chat once). Expected: no new E2E failure. They run in the required `E2E Critical (Tier 1 + Journeys)`; Task 3 Step 9 also diffs the non-required E2E jobs' failures against recent PR runs of the same workflow.

Also for the reviewer:
- The check is now after the context build, so an over-limit request does the database reads (history, world book, memories) before its 402. No LLM work runs there (V4). It can also write: with auto-summary on and the chat past its threshold, the deterministic summary is persisted into the chat settings before the 402 (V10; idempotent, the next admitted turn writes the same).
- An over-limit request that is also invalid now gets the validation error first (Decision 3).
- `chat_route_resolver` loads the server config snapshot once per router call (only with `model: "auto"` and the LLM router strategy), as prompt improvement already does for its reply.
- With the check below the router, an under-limit user's character chat router call is no longer covered by the boundary's admission, so it is checked at the call (one quota read for a user with an LLM limit; none once cached limits set no LLM key), as `/chat/completions`' router call already is.

## Files

- Modify: `tldw_Server_API/app/core/LLM_Calls/routing/accounting.py` (`log_model_router_usage`, `:51-89`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py` (import `:308`; `_log_router_usage` `:1783-1810`; the no-route branch `:3725-3727`)
- Modify: `tldw_Server_API/app/core/Chat/chat_route_resolver.py` (imports `:32-37`; `execute_router_call` `:180-200`; `log_router_usage` `:202-222`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py` (imports `:55`, `:168`; `_execute_character_provider_call` `:614-616`; the router `:1132-1191`; new helper above `_maybe_trigger_character_memory_extraction` `:3778`; resume `:6764-6769`; reply `:6988-6989`, `:7239-7241`, `:7286`)
- Modify: `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py` (new `enforce_llm_quota_if_the_router_was_refused` after `require_llm_quota`, `:110-112`)
- Modify tests: `tldw_Server_API/tests/Chat/unit/test_model_router_accounting.py`, `tldw_Server_API/tests/Chat/unit/test_chat_route_resolver.py`, `tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py`, `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py`, `tldw_Server_API/tests/Chat_NEW/integration/test_roleplay_resume_completion.py`
- Docs (Task 3): `Docs/Operations/Usage_Quotas.md` (not mirrored)
- Create (Task 0): `Docs/superpowers/plans/2026-11-05-own-key-character-chat-router-rows.md` (this plan)

## Test commands

The focused set (`$FOCUSED_A` below is these fourteen ids, space-separated):

```text
tldw_Server_API/tests/Chat/unit/test_model_router_accounting.py
tldw_Server_API/tests/Chat/unit/test_chat_route_resolver.py::test_the_router_call_and_its_usage_row_carry_whose_key_it_used
tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_auto_router_usage_row_records_whose_key_the_router_call_used
tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_complete_v2_admits_an_over_limit_user_whose_reply_runs_on_their_own_key
tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_complete_v2_answers_402_when_an_admitted_own_key_reply_is_refused_at_the_call
tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_complete_v2_still_refuses_an_own_key_turn_that_may_extract_memories
tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_complete_v2_answers_402_when_the_refused_router_call_leaves_no_route
tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_complete_v2_asks_about_the_routed_provider
tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_complete_v2_refuses_an_over_quota_user_before_any_llm_call
tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py::test_chat_endpoint_auto_routing_runs_llm_router_logs_usage_and_wires_sticky_mode
tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py::test_a_refused_router_call_that_leaves_no_route_still_gets_the_402
tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py::test_an_over_limit_user_keeps_the_routing_error_when_no_router_call_was_refused
tldw_Server_API/tests/Chat_NEW/integration/test_roleplay_resume_completion.py::test_protected_resume_generation_admits_an_over_limit_user_on_their_own_key
tldw_Server_API/tests/Chat_NEW/integration/test_roleplay_resume_completion.py::test_protected_resume_generation_enforces_llm_quota
```

```bash
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -rf $FOCUSED_A
```

Measured on `484e393333`: tests on base code `13 failed, 15 passed`; with the prototype `28 passed`. No test in the set opens a connection to a non-loopback address, on base code, with the prototype, or under any Task 2 Step 11 mutation (checked with a pytest plugin, loaded with `-p`, that blocks and logs `socket.socket.connect` to any non-loopback address; with the extraction stub removed it caught the extraction's four attempts on port 443).

**The wider set** (Task 0 baseline and Task 3): every test file that names a touched symbol:

```bash
git grep --untracked -l -E 'character_chat_sessions|complete-v2|chat_route_resolver|resolve_chat_route|log_model_router_usage|RoutingUsageContext|routing\.accounting|routing import|_select_auto_chat_llm_router_choice|_resolve_auto_chat_routing_decision|_select_auto_character_llm_router_choice|_complete_roleplay_resume|tldw_roleplay_resume_v1|enforce_llm_quota|usage_quota_deps|prompt_improvement_dispatch|/prompts/improve|endpoints import chat as|endpoints\.chat import|model="auto"|"model": "auto"' -- 'tldw_Server_API/tests/*.py' | grep '/test_[^/]*\.py$'
```

175 files on `484e393333`, and the same 175 with the prototype. Split them: the PostgreSQL part is every file under `AuthNZ_Postgres/` or whose name contains `postgres` or ends `_pg.py` (2 files: `Jobs/test_jobs_admin_endpoints_postgres.py`, `Jobs/test_jobs_events_outbox_postgres.py`); the SQLite part is the other 173.

```bash
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -n 8 --dist loadfile -rf $(cat <scratchpad>/wider_a_sqlite.txt)
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" RUN_JOBS=1 TEST_DB_NAME=tldw_test_13956_a_$$ /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -rf $(cat <scratchpad>/wider_a_pg.txt)
```

Measured on `484e393333` (SQLite part, `-n 8 --dist loadfile`, load average about 12-14): base `5071 passed, 53 skipped` in 16:23; prototype `5089 passed, 53 skipped` in 21:30 (the 18 new test ids; no failure on either). PostgreSQL part (Jobs tests, matched by the broad pattern; serially, through the repo fixture, `RUN_JOBS=1`; without it all 6 skip: "Jobs tests run only in the jobs-suite CI workflow"; the 2 skips are the outbox tests' `RUN_PG_JOBS_TESTS` gate): `4 passed, 2 skipped` on base and on the prototype.

---

## Task 0: Start TASK-13956, branch, plan, draft PR, baseline

**Files:** Create `Docs/superpowers/plans/2026-11-05-own-key-character-chat-router-rows.md`; modify `backlog/tasks/task-13956 - ...md` (backlog-py).

- [ ] **Step 1: Branch, after #3422 merges.** `gh pr view 3422 --json state` must say `MERGED` (Order). `git fetch origin dev && git switch -c fix/own-key-character-chat-13956 origin/dev`. Record `git rev-parse --short HEAD`. Run `git diff --stat 484e393333 HEAD -- <each file in Files>`: only `Docs/Operations/Usage_Quotas.md` should differ (by #3422's edit, which Task 3 Step 1's anchors already assume); a moved anchor is found by its text below.
- [ ] **Step 2: Re-check open PRs** for this plan's files: `gh pr list --base dev --state open --limit 200 --json number,headRefOid`, then `gh api repos/rmusser01/tldw_server/pulls/<n>/files --paginate --jq '.[].filename'` for every open PR (`gh pr list --json files` stops at 100 files per PR). For each overlap, read the PR's hunks in that file (`--jq '.[] | select(.filename == "<file>") | .patch'`) and look for this plan's anchors (V8). Note any new overlap in the PR body.
- [ ] **Step 3: Mark the task started:**

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13956 -s "In Progress" --append-notes "Plan, PR A of 3: Docs/superpowers/plans/2026-11-05-own-key-character-chat-router-rows.md (character chat, the router's row, the no-route 402). PR B: evaluations, chat documents, RAG. PR C: the boundaries whose calls never use a stored key, ADR-083 amendment, close-out."
```

- [ ] **Step 4: Plan file.** Copy this plan to `Docs/superpowers/plans/2026-11-05-own-key-character-chat-router-rows.md`. `git add -A backlog/tasks Docs/superpowers/plans && git diff --cached --check` (must print nothing), then commit (message 0).
- [ ] **Step 5: Draft PR.** `git push -u origin fix/own-key-character-chat-13956`, then `gh pr create --draft --base dev --title "fix(usage): own-key character chat, and the auto-router's usage row (TASK-13956, 1 of 3)" --body "Draft. Plan: Docs/superpowers/plans/2026-11-05-own-key-character-chat-router-rows.md"`.
- [ ] **Step 6: Baseline.** Build the wider set (Test commands) into `<scratchpad>/wider_a_sqlite.txt` and `wider_a_pg.txt` and run both on this branch before any code change. Expected: SQLite `5071 passed, 53 skipped`, PostgreSQL `4 passed, 2 skipped` (measured on `484e393333`; #3422 adds test files of its own, so after it merges expect more); record the real counts and every failing id.

## Task 1: The auto-router's usage row records the router call's source (AC #3)

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/routing/accounting.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_route_resolver.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_model_router_accounting.py`, `tldw_Server_API/tests/Chat/unit/test_chat_route_resolver.py`, `tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py`, `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py`

**Interfaces:**
- Produces: `log_model_router_usage(..., usage_logger=log_llm_usage, credential_source: str | None = None) -> None`, which passes `credential_source=` to `usage_logger`.
- Consumes: `billed_credential_source(provider, source, *configs)` (`Usage/llm_call_context.py:104`), `load_server_config_snapshot()` (`Provider_Credentials/byok_helpers.py`), `ProviderCallCredentials.credential_source`.

- [ ] **Step 1: Write the failing tests.**

(a) `test_model_router_accounting.py`: in `test_log_model_router_usage_matches_log_llm_usage_signature`, the strict logger mirrors `log_llm_usage`, so add its new parameter. After `conversation_id=None,` in the signature add `credential_source=None,`; after `"conversation_id": conversation_id,` in the recorded dict add `"credential_source": credential_source,`. Append:

```python
@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["user", "server_default", None])
async def test_log_model_router_usage_records_whose_key_the_router_call_used(source):
    """The router's row carries the router call's own source, so a call on the user's key does not count (TASK-13956)."""
    calls: list[dict[str, object]] = []

    async def fake_usage_logger(**kwargs: Any) -> None:
        calls.append(kwargs)

    await log_model_router_usage(
        context=RoutingUsageContext(surface="chat", endpoint="POST:/api/v1/chat/completions", user_id=1),
        provider="openai",
        model="gpt-4.1-mini",
        prompt_tokens=10,
        completion_tokens=2,
        usage_logger=fake_usage_logger,
        credential_source=source,
    )

    assert [(call["operation"], call["credential_source"]) for call in calls] == [("chat_router", source)]
```

(b) `test_chat_route_resolver.py`, append:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "app_config", "expected"),
    [
        ("user", {}, "user"),
        ("server_default", {}, "server_default"),
        ("user", {"openai_api": {"api_base_url": "http://models.internal:8000/v1"}}, "user_unmetered"),
    ],
    ids=["own-key", "server-key", "own-key-on-operator-endpoint"],
)
async def test_the_router_call_and_its_usage_row_carry_whose_key_it_used(monkeypatch, source, app_config, expected):
    """The router's raw-key call marks its context, and its row records the same source (TASK-13956)."""
    from tldw_Server_API.app.core.Chat import chat_route_resolver
    from tldw_Server_API.app.core.LLM_Calls.routing.models import RouterRequest, RoutingPolicy
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context

    seen: list[str | None] = []
    rows: list[dict] = []

    async def _probe(**_kwargs):
        seen.append(current_llm_call_context().credential_source)
        return {
            "choices": [{"message": {"content": '{"provider":"openai","model":"gpt-test"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def _byok(*_args, **_kwargs):
        async def _touch():
            return None

        return SimpleNamespace(api_key="k", app_config=app_config, source=source, touch_last_used=_touch)

    async def _log(**kwargs):
        rows.append(kwargs)

    monkeypatch.setattr(chat_route_resolver, "load_server_config_snapshot", lambda: {}, raising=False)
    await chat_route_resolver._select_auto_llm_router_choice(
        router_request=RouterRequest(model="auto", surface="chat", latest_user_turn="route me", scope="chat-1"),
        policy=RoutingPolicy(
            request_model="auto", server_default_provider="openai", boundary_mode="server_default_provider"
        ),
        candidates=[{"provider": "openai", "model": "gpt-test"}, {"provider": "openai", "model": "gpt-alt"}],
        provider_listing={
            "default_provider": "openai",
            "providers": [
                {
                    "name": "openai",
                    "default_model": "router-model",
                    "is_configured": True,
                    "models_info": [{"name": "router-model"}],
                }
            ],
        },
        request=SimpleNamespace(state=SimpleNamespace(user_id="user-1", api_key_id="key-1")),
        current_user=SimpleNamespace(id="user-1", id_int=1),
        request_id="request-1",
        surface="chat",
        endpoint="POST:/api/v1/chat/completions",
        resolve_provider_api_key_fn=lambda *a, **k: ("k", {}),
        resolve_byok_credentials_fn=_byok,
        perform_chat_api_call_async_fn=_probe,
        log_model_router_usage_fn=_log,
    )

    assert seen == [expected]
    assert [row.get("credential_source") for row in rows] == [expected]
```

(`raising=False` lets the test fail on base for its assertion, not for the missing module attribute.)

(c) `test_chat_completion_precheck.py`, after `test_auto_router_logs_only_error_types`:

```python
@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["user", "server_default"])
async def test_auto_router_usage_row_records_whose_key_the_router_call_used(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """The router's row carries its own handle's source, not the reply's (TASK-13956)."""
    rows: list[dict[str, Any]] = []
    handle = SimpleNamespace(
        provider="openai", api_key="router-key", app_config={}, credentials_resolved=True, credential_source=source
    )

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, _handle: Any) -> None:
            return None

        async def close(self) -> None:
            return None

    async def provider_call(**_kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": '{"provider":"openai","model":"routed"}'}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        }

    async def usage_log(**kwargs: Any) -> None:
        rows.append(kwargs)

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call_async", provider_call)
    monkeypatch.setattr(character_chat_sessions, "log_model_router_usage", usage_log)

    choice, _diagnostics = await character_chat_sessions._select_auto_character_llm_router_choice(
        router_request=RouterRequest(model="auto", surface="character_chat", latest_user_turn="route me", scope="chat-1"),
        policy=RoutingPolicy(
            request_model="auto", server_default_provider="openai", boundary_mode="server_default_provider"
        ),
        candidates=[{"provider": "openai", "model": "routed"}, {"provider": "openai", "model": "alternate"}],
        provider_listing={
            "default_provider": "openai",
            "providers": [
                {
                    "name": "openai",
                    "default_model": "router-model",
                    "is_configured": True,
                    "models_info": [{"name": "router-model"}],
                }
            ],
        },
        current_user=SimpleNamespace(id=1),
        credential_runtime=Runtime(),
    )

    assert choice == {"provider": "openai", "model": "routed"}
    assert [(row["provider"], row.get("credential_source")) for row in rows] == [("openai", source)]
```

(d) `test_chat_endpoint_auto_routing.py`, in `test_chat_endpoint_auto_routing_runs_llm_router_logs_usage_and_wires_sticky_mode`, after `assert captured["router_usage"][0]["provider"] == "openrouter"`:

```python
    # The fixture's runtime resolves the user's own key, so the router's row says so (TASK-13956).
    assert captured["router_usage"][0].get("credential_source") == "user"
```

- [ ] **Step 2: Run them; they fail.**

```bash
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -rf tldw_Server_API/tests/Chat/unit/test_model_router_accounting.py "tldw_Server_API/tests/Chat/unit/test_chat_route_resolver.py::test_the_router_call_and_its_usage_row_carry_whose_key_it_used" "tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_auto_router_usage_row_records_whose_key_the_router_call_used" "tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py::test_chat_endpoint_auto_routing_runs_llm_router_logs_usage_and_wires_sticky_mode"
```

Expected: `9 failed, 4 passed`: the 3 accounting cases with `TypeError: log_model_router_usage() got an unexpected keyword argument 'credential_source'`, the 3 resolver cases with `assert [None] == [...]`, the 2 character chat cases with `[('openai', None)] == [('openai', ...)]`, the chat case with `assert None == 'user'`.

- [ ] **Step 3: `accounting.py`.** In `log_model_router_usage`, after `usage_logger: UsageLogger = log_llm_usage,` add `credential_source: str | None = None,`; replace the docstring with:

```python
    """Record an LLM router call as a distinct llm_usage_log operation.

    ``credential_source`` is whose key the router call used, as its caller resolved it;
    the LLM quotas leave out a ``user`` row (ADR-083).
    """
```

and after `conversation_id=context.conversation_id,` in the `usage_logger(...)` call add `credential_source=credential_source,`.

- [ ] **Step 4: `chat.py`.** In `_select_auto_chat_llm_router_choice`'s `_log_router_usage`, after `estimated=usage["total_tokens"] == 0,` add:

```python
                    credential_source=getattr(router_credential_handle, "credential_source", None),
```

(`router_credential_handle` is the handle of the router call just made: `select_llm_router_choice` makes one call and logs it right after, `routing/runtime.py:217-244`.)

- [ ] **Step 5: `chat_route_resolver.py`.** Imports: add `from tldw_Server_API.app.core.Provider_Credentials.byok_helpers import load_server_config_snapshot` above the `byok_runtime` import, and change the `llm_call_context` import to `from tldw_Server_API.app.core.Usage.llm_call_context import billed_credential_source, llm_call_context`. Replace `execute_router_call` (from `async def execute_router_call` through `await byok_resolution.touch_last_used()`) with:

```python
    router_credential_source: str | None = None  # whose key the router call used, for its usage row

    async def execute_router_call(router_model: Any, router_messages: list[dict[str, str]]) -> Any:
        nonlocal router_credential_source
        byok_resolution = await resolve_byok_credentials_fn(
            router_model.provider,
            user_id=user_id_int,
            request=request,
            fallback_resolver=fallback_resolver,
        )
        # The adapter takes a raw key here, so its base URL is the app_config's or else the environment's
        # (in the snapshot): both decide whether a user key is the user's own spend (ADR-083).
        router_credential_source = billed_credential_source(
            router_model.provider,
            getattr(byok_resolution, "source", None),
            getattr(byok_resolution, "app_config", None),
            load_server_config_snapshot(),
        )
        try:
            with llm_call_context(records_itself=True, credential_source=router_credential_source):
                return await perform_chat_api_call_async_fn(
                    api_endpoint=router_model.provider,
                    messages_payload=router_messages,
                    api_key=byok_resolution.api_key,
                    model=router_model.model,
                    max_tokens=64,
                    streaming=False,
                    user_identifier=str(getattr(current_user, "id", "auto-router")),
                    app_config=byok_resolution.app_config,
                )
        finally:
            await byok_resolution.touch_last_used()
```

and in `log_router_usage`, after `estimated=usage["total_tokens"] == 0,` add `credential_source=router_credential_source,`.

- [ ] **Step 6: `character_chat_sessions.py` router.** In `_select_auto_character_llm_router_choice`, after the `if credential_runtime is None:` block that builds the runtime, add `router_credential_source: Optional[str] = None  # whose key the router call used, for its usage row`; in `_execute_router_call`, add `nonlocal router_credential_source` as its first line and, right after `provider_credentials = await credential_runtime.resolve(...)`, add `router_credential_source = getattr(provider_credentials, "credential_source", None)`; in `_log_router_usage`, after `estimated=usage["total_tokens"] == 0,` add `credential_source=router_credential_source,`.
- [ ] **Step 7: Run Step 2's command.** Expected: `13 passed`.
- [ ] **Step 8: Commit** (message 1): `git add` the four source files and four test files; `git diff --cached --check`.

## Task 2: Character chat names the reply's provider; a reply refused at the call is the 402; the no-route 402 (AC #1, character chat)

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`, `tldw_Server_API/app/api/v1/endpoints/chat.py`, `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py`, `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py`, `tldw_Server_API/tests/Chat_NEW/integration/test_roleplay_resume_completion.py`

**Interfaces:**
- Consumes: `enforce_llm_quota(user_id, estimated_tokens=1, *, provider=None, own_key=None)` (`API_Deps/usage_quota_deps.py:90`), `current_llm_call_context()`, `LLMQuotaExceededError` (`core/exceptions.py`).
- Produces: `async def _enforce_character_reply_quota(user_id: Any, provider: str, settings_row: Any, will_persist: bool) -> None` (module-private, `character_chat_sessions.py`); `async def enforce_llm_quota_if_the_router_was_refused(user_id: Any, routing_debug: Any) -> None` (`usage_quota_deps.py`, used by `chat.py` and `character_chat_sessions.py`).

- [ ] **Step 1: Write the failing tests.**

(a) `test_chat_completion_precheck.py`, after `test_complete_v2_refuses_an_over_quota_user_before_any_llm_call`:

```python
def _new_chat(test_client, auth_headers, settings: dict | None = None) -> str:
    """A chat with the first character, and the chat settings given."""
    characters = test_client.get("/api/v1/characters/", headers=auth_headers)
    assert characters.status_code == 200, characters.text
    created = test_client.post(
        "/api/v1/chats/", headers=auth_headers, json={"character_id": characters.json()[0]["id"]}
    )
    assert created.status_code == 201, created.text
    chat_id = created.json()["id"]
    if settings is not None:
        saved = test_client.put(f"/api/v1/chats/{chat_id}/settings", headers=auth_headers, json={"settings": settings})
        assert saved.status_code == 200, saved.text
    return chat_id


@pytest.mark.integration
def test_complete_v2_admits_an_over_limit_user_whose_reply_runs_on_their_own_key(
    monkeypatch, test_client, auth_headers, populated_character_db
):
    """Past a limit, a reply whose provider runs on the user's own key is let through unmarked (TASK-13956)."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    chat_id = _new_chat(test_client, auth_headers)
    asked: list[str] = []
    admitted: list[bool] = []

    async def _own_key(_uid, provider):
        """Stands in for the lookup answering True (the boundary is under test)."""
        asked.append(provider)
        return True

    def _completion(**_kwargs):
        admitted.append(current_llm_call_context().prechecked)
        return {"choices": [{"message": {"content": "On your own key."}}]}

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", _completion)
    monkeypatch.setattr(character_chat_sessions, "is_model_known_for_provider", lambda *args: None)
    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    refuse_llm_work(monkeypatch)
    response = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        headers=auth_headers,
        json={"provider": "llama", "model": "garden-model", "append_user_message": "Hi", "save_to_db": False},
    )

    assert response.status_code == 200, response.text
    assert response.json()["assistant_content"] == "On your own key."
    assert asked == [response.json()["provider"]]
    assert admitted == [False]  # unmarked: the call is checked where its key is known


@pytest.mark.integration
@pytest.mark.parametrize("stream", [False, True])
def test_complete_v2_answers_402_when_an_admitted_own_key_reply_is_refused_at_the_call(
    monkeypatch, test_client, auth_headers, populated_character_db, stream
):
    """Admitted on their own key, a reply refused at the call is the user's limit: 402, not 502 (TASK-13956)."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    chat_id = _new_chat(test_client, auth_headers)

    async def _own_key(_uid, _provider):
        """The lookup says the reply's provider runs on the user's own key."""
        return True

    def _refused_at_the_call(**_kwargs):
        """What the check at the call raises when the reply's key turns out not to be the user's own."""
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", _refused_at_the_call)
    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    refuse_llm_work(monkeypatch)
    response = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        headers=auth_headers,
        json={"provider": "llama", "model": "garden-model", "append_user_message": "Hi", "save_to_db": False, "stream": stream},
    )

    assert_llm_quota_402(response)


@pytest.mark.integration
def test_complete_v2_still_refuses_an_own_key_turn_that_may_extract_memories(
    monkeypatch, test_client, auth_headers, populated_character_db
):
    """Memory extraction runs on the server's key, so a saved turn in a chat that has it on is refused (TASK-13956)."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.app.core.Character_Chat.modules import character_memory_extraction
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    chat_id = _new_chat(
        test_client, auth_headers, {"characterMemoryExtraction": {"enabled": True, "intervalMessages": 1}}
    )
    calls: list[dict] = []
    extractions: list[dict] = []

    async def _own_key(_uid, _provider):
        """The reply's provider would run on the user's own key."""
        return True

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", lambda **kwargs: calls.append(kwargs))
    # A regression would start extraction, which calls the server's provider for real: stub it.
    monkeypatch.setattr(
        character_memory_extraction, "extract_character_memories", lambda **kwargs: extractions.append(kwargs)
    )
    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    refuse_llm_work(monkeypatch)
    response = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        headers=auth_headers,
        json={"provider": "llama", "model": "garden-model", "append_user_message": "Hi", "save_to_db": True},
    )

    assert_llm_quota_402(response)
    assert response.json()["detail"]["error"] == "limit_exceeded"
    assert calls == []
    assert extractions == []


@pytest.mark.integration
def test_complete_v2_answers_402_when_the_refused_router_call_leaves_no_route(
    monkeypatch, test_client, auth_headers, populated_character_db
):
    """The check now runs after the auto-router; a refused router call under failure_mode=error is still the 402."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    chat_id = _new_chat(test_client, auth_headers)
    calls: list[dict] = []

    async def _refused_router_call(**_kwargs):
        """What the check at the call raises for an over-limit user's router call on a server key."""
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(
        character_chat_sessions,
        "get_configured_providers",
        lambda: {
            "default_provider": "local-llm",
            "providers": [
                {
                    "name": "local-llm",
                    "is_configured": True,
                    "default_model": "router-model",
                    "models_info": [{"name": "router-model"}, {"name": "routed-model"}],
                }
            ],
        },
    )
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call_async", _refused_router_call)
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call", lambda **kwargs: calls.append(kwargs))
    refuse_llm_work(monkeypatch)
    response = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        headers=auth_headers,
        json={
            "model": "auto",
            "routing": {"failure_mode": "error"},
            "append_user_message": "Route this",
            "save_to_db": False,
        },
    )

    assert_llm_quota_402(response)
    assert "error_code" not in response.json()["detail"]  # not the 503 auto_routing_failed
    assert calls == []


@pytest.mark.integration
def test_complete_v2_asks_about_the_routed_provider(monkeypatch, test_client, auth_headers, populated_character_db):
    """With model auto, the quota check asks about the provider the auto-router chose (TASK-13956)."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    chat_id = _new_chat(test_client, auth_headers)
    asked: list[str] = []

    async def _router_call(**_kwargs):
        """The auto-router picks the second model."""
        return {
            "choices": [{"message": {"content": '{"provider":"local-llm","model":"routed-model"}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

    async def _own_key(_uid, provider):
        """The routed provider runs on the user's own key."""
        asked.append(provider)
        return True

    monkeypatch.setattr(
        character_chat_sessions,
        "get_configured_providers",
        lambda: {
            "default_provider": "local-llm",
            "providers": [
                {
                    "name": "local-llm",
                    "is_configured": True,
                    "default_model": "router-model",
                    "models_info": [{"name": "router-model"}, {"name": "routed-model"}],
                }
            ],
        },
    )
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call_async", _router_call)
    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    refuse_llm_work(monkeypatch)
    response = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        headers=auth_headers,
        json={"model": "auto", "append_user_message": "Route this", "save_to_db": False},
    )

    assert response.status_code == 200, response.text
    assert (response.json()["provider"], response.json()["model"]) == ("local-llm", "routed-model")
    assert asked == ["local-llm"]
```

(The refusal tests each assert a real output beside their collector, and no new test stacks four `monkeypatch` calls: `test_quality_triage.py --enforce` flags either shape in an integration-marked file. A first prototype failed it with `mock_density=2 (baseline 1)` and `tautology_suspect=2 (new)` on this file. The extraction test drops the earlier `is_model_known_for_provider` patch, which strict model selection makes redundant under pytest (`_is_char_chat_test_mode` reads `PYTEST_CURRENT_TEST`), to stub extraction within the same budget. `local-llm` runs the offline simulation, so the routed-provider test needs no reply stub.)

(b) `test_chat_endpoint_auto_routing.py`, after `test_a_refused_router_call_is_not_a_provider_failure_and_the_request_gets_the_402`:

```python
@pytest.mark.integration
def test_a_refused_router_call_that_leaves_no_route_still_gets_the_402(
    persona_chat_client,
    persona_chat_db,
    execution_scoped_provider_credentials,
    monkeypatch,
):
    """Under failure_mode=error a refused router call leaves no route: the over-limit user gets the 402, not a 503."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work

    authenticated_client, headers, _ = persona_chat_client
    refuse_llm_work(monkeypatch)
    request_data = ChatCompletionRequest(
        model="auto",
        api_provider="openrouter",
        routing={"failure_mode": "error"},
        messages=[ChatCompletionUserMessageParam(role="user", content="Summarize this")],
    )

    async def refused_router_call(**kwargs):
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    with (
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.get_configured_providers",
            return_value={
                "providers": [
                    {
                        "name": "openrouter",
                        "default_model": "anthropic/claude-4.5-sonnet",
                        "models_info": [
                            {"name": "anthropic/claude-4.5-sonnet", "tool_support": True, "vision_support": True, "quality_rank": 20},
                            {"name": "openai/gpt-4.1-mini", "tool_support": True, "vision_support": True, "quality_rank": 10},
                        ],
                    }
                ],
                "default_provider": "openrouter",
            },
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call_async",
            side_effect=refused_router_call,
            create=True,
        ),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", headers=headers, json=request_data.model_dump())

    assert_llm_quota_402(response)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("providers", "expected"),
    [
        (
            [{"name": "openai", "models_info": [{"name": "gpt-test", "tool_support": True, "quality_rank": 1}]}],
            (503, "auto_routing_failed"),
        ),
        ([], (400, "auto_routing_no_candidates")),
    ],
    ids=["no-decision", "no-candidates"],
)
def test_an_over_limit_user_keeps_the_routing_error_when_no_router_call_was_refused(
    authenticated_client, mock_chacha_db, setup_dependencies, monkeypatch, providers, expected
):
    """Only a router call refused at the call turns a lost route into the 402; other routing errors stay (TASK-13956)."""
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    refuse_llm_work(monkeypatch)
    request_data = ChatCompletionRequest(
        model="auto",
        messages=[ChatCompletionUserMessageParam(role="user", content="Route this")],
    )

    with (
        patch("tldw_Server_API.app.api.v1.endpoints.chat.route_model", return_value=None),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.get_configured_providers",
            return_value={"providers": providers, "default_provider": "openai"},
        ),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert (response.status_code, response.json()["detail"].get("error_code")) == expected
```

(With one candidate the router is skipped (`single_candidate`) and `route_model` is patched to None: a lost route whose router call was not refused, the stand-in for an outage. Both cases pass on base code, where `/chat/completions` checks after routing; a guard that ignores why the route was lost fails them with `(402, None)`.)

(c) `test_roleplay_resume_completion.py`, after `test_protected_resume_generation_enforces_llm_quota`:

```python
def test_protected_resume_generation_admits_an_over_limit_user_on_their_own_key(resume_api, monkeypatch) -> None:
    """Past a limit, a resume whose saved provider runs on the user's own key is let through unmarked (TASK-13956)."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as endpoint
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    client, db, cid, headers = resume_api
    body = generation_request(client, db, cid, headers)
    asked: list[str] = []
    admitted: list[bool] = []

    async def own_key(_uid: object, provider: str) -> bool:
        """Stands in for the lookup answering True for the saved provider (the boundary is under test)."""
        asked.append(provider)
        return provider == "local-llm"

    def provider_call(**_kwargs):
        admitted.append(current_llm_call_context().prechecked)
        return {"choices": [{"message": {"role": "assistant", "content": "Resumed on your key."}}]}

    monkeypatch.setattr(endpoint, "perform_chat_api_call", provider_call)
    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", own_key)
    refuse_llm_work(monkeypatch)
    response = client.post(f"/api/v1/chats/{cid}/complete-v2", headers=headers, json=body)

    assert response.status_code == 200, response.text
    assert response.json()["assistant_content"] == "Resumed on your key."
    assert asked == ["local-llm"]
    assert admitted == [False]
```

- [ ] **Step 2: Run them with the two existing pins; the new admits fail, the pins pass.**

```bash
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -rf "tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_complete_v2_admits_an_over_limit_user_whose_reply_runs_on_their_own_key" "tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_complete_v2_answers_402_when_an_admitted_own_key_reply_is_refused_at_the_call" "tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_complete_v2_still_refuses_an_own_key_turn_that_may_extract_memories" "tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_complete_v2_answers_402_when_the_refused_router_call_leaves_no_route" "tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_complete_v2_asks_about_the_routed_provider" "tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py::test_complete_v2_refuses_an_over_quota_user_before_any_llm_call" "tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py::test_a_refused_router_call_that_leaves_no_route_still_gets_the_402" "tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py::test_an_over_limit_user_keeps_the_routing_error_when_no_router_call_was_refused" "tldw_Server_API/tests/Chat_NEW/integration/test_roleplay_resume_completion.py::test_protected_resume_generation_admits_an_over_limit_user_on_their_own_key" "tldw_Server_API/tests/Chat_NEW/integration/test_roleplay_resume_completion.py::test_protected_resume_generation_enforces_llm_quota"
```

Expected (measured with Task 1's code in): `4 failed, 11 passed`: the reply admit test, the routed-provider test and the resume admit test with `assert 402 == 200`, the `/chat/completions` no-route test with `auto_routing_failed` in the body. The 402-at-the-call test (both), the extraction test, the character chat no-route test and the two routing-error cases pass on base (the check runs before or after all routing today); they guard the move.

- [ ] **Step 3: The helper.** In `character_chat_sessions.py`, directly above `def _maybe_trigger_character_memory_extraction(`, add:

```python
async def _enforce_character_reply_quota(user_id: Any, provider: str, settings_row: Any, will_persist: bool) -> None:
    """The LLM quota check, once the reply's provider is known (LLM usage accounting spec §4a, TASK-13956).

    An over-limit user whose reply runs on their own key is let through unmarked, so each call is
    checked where its key is known. Not when the turn may start memory extraction (a saved reply in
    a chat that has it on): extraction runs on the server's key, so the check refuses as before.
    """
    settings = (settings_row.get("settings") or {}) if isinstance(settings_row, dict) else {}
    extraction = settings.get("characterMemoryExtraction") if isinstance(settings, dict) else None
    extracts = will_persist and isinstance(extraction, dict) and bool(extraction.get("enabled"))
    # ponytail: refuses every turn of such a chat, not only the turns that reach the extraction interval.
    await enforce_llm_quota(user_id, provider=None if extracts else provider)
```

(It reads the setting as `_maybe_trigger_character_memory_extraction` does.)

- [ ] **Step 4: Move the reply's check.** In `character_chat_completion`, delete these three lines (after the persona check):

```python
        # Usage quotas (LLM usage accounting spec §4a): refuse before any LLM call or streaming response.
        await enforce_llm_quota(current_user.id)

```

After the provider/model resolution block, which ends with:

```python
                    "message": "Failed to resolve provider and model for character chat completion.",
                },
            ) from exc
```

add, before the blank line and `if strict_model_selection and explicit_model_requested:`:

```python
        await _enforce_character_reply_quota(current_user.id, provider, settings_row, will_persist)
```

In the auto-routing branch, make the shared guard (Step 5) the first line of `if candidate_count > 0:`, before the 503:

```python
            if routing_decision is None:
                candidate_count = int((routing_debug or {}).get("candidate_count") or 0)
                if candidate_count > 0:
                    await enforce_llm_quota_if_the_router_was_refused(current_user.id, routing_debug)
                    raise HTTPException(
```

and replace the import `from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota` with:

```python
from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import (
    enforce_llm_quota,
    enforce_llm_quota_if_the_router_was_refused,
)
```

- [ ] **Step 5: The shared no-route guard.** In `usage_quota_deps.py`, after `require_llm_quota` (the end of the file), add:

```python


async def enforce_llm_quota_if_the_router_was_refused(user_id: Any, routing_debug: Any) -> None:
    """The 402 when auto-routing found candidates but no route because its router call was refused at the call.

    The quota refused the router call, so the request gets the 402, not ``503 auto_routing_failed``
    (TASK-13956). A router call that failed for any other reason keeps the 503, for every user.
    """
    router = routing_debug.get("llm_router") if isinstance(routing_debug, dict) else None
    if isinstance(router, dict) and router.get("error") == LLMQuotaExceededError.__name__:
        await enforce_llm_quota(user_id)
```

(`select_llm_router_choice` records the router call's exception class as `error`; both endpoints keep that under `routing_debug["llm_router"]`, V5.)

- [ ] **Step 6: The resume.** In `_complete_roleplay_resume`, replace

```python
    await enforce_llm_quota(current_user.id)
    limiter = get_character_rate_limiter()
    current_count = await run_in_threadpool(db.count_messages_for_conversation, chat_id)
    await limiter.check_message_limit(chat_id, current_count + 1)
    effective = prepared["effective_completion"]
    provider, model = effective["provider"], effective["model"]
```

with

```python
    effective = prepared["effective_completion"]
    provider, model = effective["provider"], effective["model"]
    # An over-limit user whose saved provider runs on their own key is let through unmarked (TASK-13956).
    await enforce_llm_quota(current_user.id, provider=provider)
    limiter = get_character_rate_limiter()
    current_count = await run_in_threadpool(db.count_messages_for_conversation, chat_id)
    await limiter.check_message_limit(chat_id, current_count + 1)
```

- [ ] **Step 7: `/chat/completions`' no-route branch.** In `create_chat_completion`, make the same guard the first line of `if candidate_count > 0:`:

```python
            if routing_decision is None:
                candidate_count = int((routing_debug or {}).get("candidate_count") or 0)
                if candidate_count > 0:
                    await enforce_llm_quota_if_the_router_was_refused(getattr(current_user, "id", None), routing_debug)
                    raise HTTPException(
```

and replace the import `from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota, users_own_key_serves` with:

```python
from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import (
    enforce_llm_quota,
    enforce_llm_quota_if_the_router_was_refused,
    users_own_key_serves,
)
```

- [ ] **Step 8: Run Step 2's command; the moved check exposes the 502.** Expected: `2 failed, 13 passed`: `test_complete_v2_answers_402_when_an_admitted_own_key_reply_is_refused_at_the_call[False]` and `[True]`, each `AssertionError: {"detail":"Chat provider error"}` (the admitted reply's refusal at the call, rewrapped as a 502, V9).
- [ ] **Step 9: Let the refusal through.** In `character_chat_sessions.py`, replace the import `from tldw_Server_API.app.core.exceptions import raise_detached_error` with `from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError, raise_detached_error`, and in `_execute_character_provider_call` replace

```python
    except asyncio.CancelledError:
        raise
    except ChatAPIError as e:
```

with

```python
    except asyncio.CancelledError:
        raise
    except LLMQuotaExceededError:
        raise  # the user's limit, refused at the call: the route answers the 402
    except ChatAPIError as e:
```

- [ ] **Step 10: Run Step 2's command.** Expected: `15 passed`. Then the size ratchet: `PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/lint/test_function_size_ratchet.py`: `13 passed` (`character_chat_completion` 1099 of 1100, `create_chat_completion` 2906 of 2908).
- [ ] **Step 11: Mutation probes** (revert each after its run; none makes a request leave the machine, measured with the socket probe described under Test commands): (i) drop the guard line in character chat's `if candidate_count > 0:` → `test_complete_v2_answers_402_when_the_refused_router_call_leaves_no_route` fails with `auto_routing_failed` (measured); (ii) pass `provider=provider` unconditionally in `_enforce_character_reply_quota` → `test_complete_v2_still_refuses_an_own_key_turn_that_may_extract_memories` fails with a 200, and the stubbed extraction keeps the turn's extraction call on the machine (measured; without the stub the same run sent `POST https://api.openai.com/v1/chat/completions`, review A I2); (iii) replace the `if` in `enforce_llm_quota_if_the_router_was_refused` with `if True:` → `test_an_over_limit_user_keeps_the_routing_error_when_no_router_call_was_refused[no-decision]` fails with `(402, None) == (503, 'auto_routing_failed')` (measured); (iv) Step 8 is the RED of the `except LLMQuotaExceededError` line.
- [ ] **Step 12: Commit** (message 2): `git add tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py tldw_Server_API/tests/Chat_NEW/integration/test_roleplay_resume_completion.py && git diff --cached --check`.

## Task 3: Docs, verify, notes, review, PR

**Files:** `Docs/Operations/Usage_Quotas.md`; `backlog/tasks/task-13956 - ...md` (backlog-py).

- [ ] **Step 1: `Usage_Quotas.md`** (not mirrored). Five edits, each found by its text (the blocks are literal text), on the file as it reads after #3422 (measured on #3422's head `9791e52407`: #3422 rewrites edit 5's bullet but keeps its anchor sentence, and touches none of the other four anchors).
  1. In the own-key bullet of "What ... count", replace

```text
and rows whose key is unknown (`NULL`): rows written before the column existed, and the auto-router's call. If the column
```

  with

```text
and rows whose key is unknown (`NULL`), such as rows written before the column existed. An auto-router call's row records the router call's own key, so a router call on the user's own key does not count either. If the column
```

  2. In "Where a refusal happens", replace `character chat replies; RAG search,` with

```text
character chat replies (with `model: "auto"`, after the auto-router's call, as on `/chat/completions`); RAG search,
```

  3. In "Shared machinery is not affected", replace

```text
including the auto-router's call on `/chat/completions`, which runs before that endpoint's own check.
```

  with

```text
including the auto-router's call on `/chat/completions` and character chat, which runs before that endpoint's own check. When the refused router call leaves no route (a request with `routing.failure_mode: "error"`), the request gets the 402, not `auto_routing_failed`.
```

  4. Replace the whole "**An over-limit user on their own key.**" bullet with:

```markdown
- **An over-limit user on their own key.** `/chat/completions`, the Messages API (`/messages`, `/v1/messages`) and character chat let an over-limit user through when four things hold: BYOK is on, the request's provider bills per key and its base URL is the vendor's, the provider is on the BYOK allowlist, and the user has an active key for it. Character chat asks for the reply's provider once it is known: after the auto-router, or the saved provider of a roleplay resume. The request is not marked admitted, so each of its calls is checked where its key is known. A call on the user's own key runs. A call on any other key, a fallback provider's for example, is refused at the call. A saved character chat reply in a chat with automatic memory extraction on is still refused, because extraction runs on the server's key. On `/chat/completions` the hosted billing plan's check is skipped for such a request. Every other boundary endpoint refuses an over-limit user before the key is known, even when the call would have used the user's own key: RAG generation, evaluations and benchmarks, flashcards and quizzes, chat document generation, watchlists and media analysis. So does a media ingest job's check when it starts.
```

  5. In "Streams without a usage chunk, and router calls", after `is recorded from the provider's token counts without its usage block, so it costs the standard rate.` add one space and:

```text
Its row records whose key the router call used (`credential_source`), as the reply's row does; so does the router call that prompt improvement (`POST /prompts/improve`) makes with `model: "auto"`.
```

  ADR-083 is not touched here (PR C amends it once for all three PRs). `bash Helper_Scripts/refresh_docs_published.sh` must leave `git status --short Docs/Published` empty (measured). Commit (message 3).
- [ ] **Step 2: Verify** (each expected result measured on the prototype):
  - The focused set: `28 passed`.
  - The wider set, compared with Task 0's baseline: the baseline plus this PR's 18 new test ids, no new failure (measured on `484e393333`: SQLite `5089 passed, 53 skipped`, base `5071 passed, 53 skipped`; PostgreSQL `4 passed, 2 skipped`, base the same). Any other failure is this PR's.
  - `PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI tools/backlog-py/tests`: `1661 passed, 6 skipped` (measured on `484e393333` + #3422's `Usage_Quotas.md` + A, 6:29; `tests/Docs` alone `212 passed`; the lint part holds the function-size and other ceiling ratchets and the test-quality ratchet). If `lint/test_rg_route_map_lint.py` aborts under load, re-run that file alone before judging.
  - `PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/ci/check_shard_coverage.py`: `new_uncovered=0`, "OK — no newly-unshared test files." (measured).
  - `PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/ci/test_quality_triage.py --enforce`: "OK — no offenses beyond the baseline." (measured; see the note under Task 2 Step 1 for the two shapes it rejects).
  - `ruff check` on the ten touched `.py` files: the same 6 findings as on base (3 in `accounting.py`, 2 in `character_chat_sessions.py`, 1 in `chat.py`; import order, `UP035`, `F401`), none new (measured).
  - `PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/ci/route_auth_ratchet.py`: "208 unauthenticated routes; baseline exact." (measured).
  - `PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts`: "OK — OpenAPI fingerprint matches the checked-in snapshot." and "OK — committed path union matches the canonical contract." (measured).
  - `git grep -n -E "character_chat_sessions|endpoints/chat.py|chat_route_resolver|routing/accounting|usage_quota_deps|test_chat_completion_precheck|test_chat_endpoint_auto_routing|test_chat_route_resolver|test_model_router_accounting|test_roleplay_resume_completion" Helper_Scripts/ci/scope_predicate_baseline.txt`: nothing (measured).
  - `git diff origin/dev --check`: nothing.
- [ ] **Step 3: Bandit:** `/opt/homebrew/bin/uvx bandit -ll tldw_Server_API/app/core/LLM_Calls/routing/accounting.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/core/Chat/chat_route_resolver.py tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py`. Expected: "No issues identified.", Medium 0, High 0 (measured on `484e393333` + A).
- [ ] **Step 4: Notes on TASK-13956** (backlog-py; fill every `<...>`; do not check ACs or close the task, PR C does):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13956 \
  --append-notes "PR A (#<n>): character chat names the reply's provider after the auto-router (_enforce_character_reply_quota; a saved reply in a chat with memory extraction on still counts: extraction runs on the server's key) and the resume its saved provider; an admitted reply refused at the call answers the 402, not 502 Chat provider error (_execute_character_provider_call lets LLMQuotaExceededError through); the router's row records the router call's credential_source on /chat/completions, character chat and chat_route_resolver (whose raw-key router call also marks its context with billed_credential_source(provider, source, app_config, server snapshot)); a router call refused at the call that leaves no route answers the 402 on /chat/completions and character chat, not 503 auto_routing_failed (enforce_llm_quota_if_the_router_was_refused; any other lost route keeps its 503 or 400). RED/GREEN: focused <red> -> <green>. Wider SQLite <base> -> <after>; PostgreSQL <base> -> <after>. Mutations: no-route guard dropped -> auto_routing_failed; extraction ignored -> 200 (no request leaves the machine); guard unconditional -> 402 for a router outage. Bandit: <result>."
git add -A backlog/tasks && git diff --cached --check
```

  Commit (message 4).
- [ ] **Step 5: Review before the PR.** Run `/superpowers:requesting-code-review` in a clean subagent on `origin/dev..HEAD`, with this plan's Review Focus. Address findings; record each ruling.
- [ ] **Step 6: PR body.** `git push`; `gh pr edit <n> --body-file <file>`, then `gh pr ready <n>`. The body states: what changes for an over-limit user on their own key (character chat; a reply refused at the call is the 402) and on a server key (nothing: same 402; validation errors first, and possibly the chat's deterministic auto summary written first, Decision 3 and V10); the router row; the no-route 402 and the routing errors it leaves alone (Decision 4); the measured counts; Review Focus; the open-PR overlap (V8); the `## Change summary` (Step 7). It ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- [ ] **Step 7: Change summary.** The owner's ADR-004 waiver covers this PR (Decision 1, 2026-11-05): the body carries a full `## Change summary` section that records it (what changed and why, and that the owner waived writing it for the three TASK-13956 PRs on 2026-11-05).
- [ ] **Step 8: Required checks.** `gh variable get MERGE_QUEUE` ("not found" on 2026-11-05: rebase only the PR about to merge). Wait for the 8 required statuses (the 7 contexts of ruleset `21824526`, `E2E Critical (Tier 1 + Journeys)` among them, plus the license status).
- [ ] **Step 9: Non-required E2E.** The character chat journeys gate the merge through `E2E Critical (Tier 1 + Journeys)` (Plan-specific constraints). The other jobs of "Frontend E2E Tiers" (`E2E Features (Tier 2 + 3)`, which holds `tier-2-features/characters.spec.ts`; `E2E Admin (Tier 4 + 5)`; `E2E UX Regression (notes/chat)`) do not. That workflow never runs on `dev`, so diff against recent PR runs of it: `gh run list --workflow "Frontend E2E Tiers" --event pull_request --limit 10 --json databaseId,headBranch,conclusion`, then `gh run view <id> --log-failed` for this PR's run and for two or three recent runs of other PRs, and compare the failing test ids per job. A failure on this PR that the other runs do not have, in a character, chat or chat-settings spec, is this PR's until shown otherwise.
- [ ] **Step 10: Merge** with `QUEUE="<n>" sh <plans dir>/pr_queue3.sh` in the background once reviews are settled, the Change summary is in and Step 9 is clean. Wait for each body edit's re-triggered license status. Confirm MERGED before deleting the branch.

## Commit messages

Each ends with a blank line and `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

- 0: `docs(plan): plan own-key character chat and the router's usage row (TASK-13956, 1 of 3)`
- 1: `fix(usage): the auto-router's usage row records whose key the router call used (TASK-13956)`
- 2: `fix(usage): character chat lets an over-limit user through on their own key, and a call refused there or a refused router call that leaves no route is a 402 (TASK-13956)`
- 3: `docs(usage): character chat and the router's row on the user's own key (TASK-13956)`
- 4: `chore(backlog): TASK-13956 notes for PR A`

## Not measured

- The Frontend E2E jobs on the PR (Task 3 Steps 8-9). The plan reviewer ran the three character chat journey specs locally on the first prototype (7 passed); not re-run after the fold.
- A character chat stream admitted on the user's own key end to end through a real adapter: the admit test stubs `perform_chat_api_call`, so the check at the call is covered by TASK-13955's call-layer tests, not here; the 402-at-the-call test stubs the refusal too.
- The rules fallback after a router call refused at the call, with a reply on the user's own key, on character chat, end to end with the real gate: the `/chat/completions` twin is pinned (`test_a_refused_router_call_is_not_a_provider_failure_and_the_request_gets_the_402`), and `test_complete_v2_asks_about_the_routed_provider` pins that character chat asks about the routed provider; the plan reviewer's probe P2 measured the fallback with the real gate.

## Owner answers (2026-11-05, recorded by the controller)

- Plans A, B and C: "Approve (Recommended)", with both reviews' Important fixes and minors folded in (review_plan_13956_a.md, review_plan_13956_bc.md).
- AC #2 (and the matching part of AC #1): "Amend + follow-up (Recommended)": PR C amends the ACs with backlog-py to what holds, pins the server-key boundaries, and files the follow-up asking whether those features should use the stored key (the study assistants as the cheap first slice; media analysis about a day).
- Decision 1, ADR-004 Change summary: the waiver covers all three TASK-13956 PRs ("Yes, covers all three").

## Folded after the plan reviews (2026-11-05)

From `review_plan_13956_a.md` (0 Blocker, 2 Important, 5 Minor) and the owner's answers. Re-measured on `origin/dev` `484e393333` with #3422's `Usage_Quotas.md` (`9791e52407`, read-only); `proto_13956_a.patch` was rebuilt from this plan's text on that base and matches it byte for byte (`cmp`).

| Finding | What changed |
|---|---|
| I1: a reply refused at the call became `502 Chat provider error` | Task 2 Step 9: `except LLMQuotaExceededError: raise` before `except ChatAPIError` in `_execute_character_provider_call`, and the import. New `test_complete_v2_answers_402_when_an_admitted_own_key_reply_is_refused_at_the_call` (stream and not): passes on base (refused at the boundary), fails after the move with `{"detail":"Chat provider error"}` (Step 8, measured), passes with the fix. V9, Goal, Architecture, Review Focus 3, ADR text in PR C. |
| I2: the extraction test reached `api.openai.com` | The test stubs `character_memory_extraction.extract_character_memories` (replacing the redundant `is_model_known_for_provider` patch, so still three patches) and asserts `extractions == []`. Mutation (ii) still fails it with the 200; a socket probe saw no outbound connection (without the stub the same run made four blocked attempts on port 443, the extraction's call to the server's OpenAI endpoint). |
| M1: the auto summary can be written before the 402 | V10, Decision 3, Review Focus ("Also for the reviewer"), Task 3 Step 6's PR body. No code change. |
| M2: the no-route guard turned a 400 and an outage's 503 into 402s | The code fix, keeping every existing status: `enforce_llm_quota_if_the_router_was_refused(user_id, routing_debug)` (new, `usage_quota_deps.py`) is the first line of `if candidate_count > 0:` on both endpoints and runs `enforce_llm_quota(user_id)` only when the router's recorded error is `LLMQuotaExceededError`. No candidates keeps the 400; any other lost route keeps the 503; every branch raises, so no spend is admitted. Pinned by `test_an_over_limit_user_keeps_the_routing_error_when_no_router_call_was_refused` (`no-decision` 503, `no-candidates` 400; both pass on base; mutation (iii), the guard unconditional, fails `[no-decision]` with `(402, None)`). The reviewer's simpler option (the unconditional guard inside `if candidate_count > 0:`) keeps the 400 but still answers an over-limit own-key user's router outage with a 402. One line per endpoint: sizes unchanged (1099 of 1100, 2906 of 2908). Decisions 3 and 4 rewritten. |
| M3: Step 9 diffed against `dev` runs that do not exist | A plan-specific constraint under the verbatim Global Constraints: `E2E Critical (Tier 1 + Journeys)` is required (ruleset `21824526`, re-checked) and runs the character chat journeys. Task 3 Step 9 diffs the non-required jobs against recent PR runs of "Frontend E2E Tiers" (its last `dev` run is from 2026-07-09); Step 8 and Review Focus 5 follow. |
| M4: V8 stale; #3422 edits the same bullet | V8 rebuilt from the full file lists and hunks of all 38 open PRs (heads named). #3422 has not merged: the docs step is anchored on #3422's text (edit 5's anchor sentence survives its rewrite; edits 1-4 are outside it), the prototype's docs hunk is relative to it, and this PR executes after #3422 merges (Order, Task 0 Step 1). |
| M5: nothing pinned that the check asks about the routed provider | `test_complete_v2_asks_about_the_routed_provider` (RED on base `assert 402 == 200`, GREEN with A); the Not-measured entry rewritten. |
| Not findings: Task 2's commit names no files | Task 2 Step 12 names them. |
| Owner answer to Decision 1 | Recorded; Task 3 Step 7 writes the `## Change summary` section. |

Re-measured (each against the same base code first):
- Task 1: RED `9 failed, 4 passed`; GREEN `13 passed` (unchanged).
- Task 2 (15 tests, was 10): RED with Task 1's code `4 failed, 11 passed`; after the move, before Step 9, `2 failed, 13 passed`; GREEN `15 passed`.
- Focused set (14 ids, 28 tests, was 11 and 23): base code `13 failed, 15 passed`; prototype `28 passed`.
- Wider set, 175 files on base and prototype (173 SQLite, 2 PostgreSQL): SQLite base `5071 passed, 53 skipped` (16:23), prototype `5089 passed, 53 skipped` (21:30), no failure; PostgreSQL (serial, repo fixture, `TEST_DB_NAME=tldw_test_13956f`) `4 passed, 2 skipped` on both.
- `tests/Docs tests/lint tests/CI tools/backlog-py/tests` on A: `1661 passed, 6 skipped` (6:29); `tests/Docs` `212 passed`. Shard coverage `new_uncovered=0`; triage `--enforce` OK; route-auth "208 unauthenticated routes; baseline exact."; OpenAPI both OK; function-size ratchet `13 passed`; Bandit on the five source files: no issues; ruff: the same 6 findings as on base, none new; `scope_predicate_baseline.txt` lists no file of PRs A, B or C; `git diff --check` clean.
- Open-PR overlap: V8 (re-checked against current heads).
