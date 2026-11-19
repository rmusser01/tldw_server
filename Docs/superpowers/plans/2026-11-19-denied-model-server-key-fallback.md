# TASK-15571 Implementation Plan: a model the provider override denies falls back to the server key

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One rule for a model the admin provider-override policy denies (`model_not_allowed`), on every `analyze`
call and on the MCP model-completion path. The call runs on the server key, and is counted and checked as a
server-key call; it fails closed when no quota check can answer for it. A disabled provider, or an override store
that cannot be read, refuses the call. Every denial writes one warning naming the provider and the reason.

**Architecture:** One new helper, `server_key_fallback(provider, model) -> bool` in `llm_provider_overrides.py`,
reads the policy. It returns True only for `model_not_allowed`. It raises, and logs, `ProviderOverridePolicyError`
for a disabled provider and `ByokResolutionError("credential_store_unavailable")` when the policy cannot be read.
A second helper, `log_server_key_fallback(provider, outcome)`, writes the one line for a denied model once its
caller knows what became of the call. The new helper has two callers, the two places every affected call routes
through:
- **`_summarize_via_adapter`**, for every `analyze` call, through `_override_outcome` once the model is chosen.
  - A refusal returns `configuration`.
  - A fallback drops the handle and keeps the model, but only after the quota answers for the call. The answer
    comes either from the boundary's mark or from a strict check (`user_to_check(strict=True)`, then
    `check_user_sync(user_id, strict=True)`).
  - A refusal by the check raises the 402. A check that cannot answer refuses with `configuration` and its own
    message.
- **The authoritative branch of `ProviderCredentialRuntime._resolve_entry`**, the MCP adapter's only route,
  after the scope check. A fallback re-resolves the server key from the frozen snapshot, and the MCP
  reservation's existing fail-closed checks then apply. The helper's raises fail the completion closed.

The media ingest worker's start check also gets a strict mode (`check_user_async(user_id, strict=True)`), so it
marks a job admitted only when the check answered. The resolve-time path in `users_own_key_credentials` already
falls back and already re-checks quota at the boundary. It keeps its catch-all and gains a log line.

**Tech Stack:** Python 3.12, FastAPI dependencies, loguru, pytest with pytest-asyncio.

**Spec:** `backlog/tasks/task-15571 - The-provider-override-model-allowlist-gives-different-outcomes-depending-on-where-the-model-came-from.md`,
the owner decision below, the coordinator's rulings below, the task review `final_review_15571.md`, and the plan
review `plan_review_15571.md` (both in the plans dir).
**Prototype:** `proto_15571.patch` (plans dir) is the measured text of every change below. `git apply --check` is
clean on `origin/dev` e656a55f80 (17 files). Each task can apply its own files from it with
`git apply --include=<path>`, or type the code from this plan. The plan is the authority. The patch's
`Usage_Quotas.md` carries the placeholder `TASK-15577`; Task 5 replaces it.

## Rulings (coordinator, 2026-11-19, on `plan_review_15571.md`)

| Finding | Ruling | Applied in |
|---|---|---|
| B1 | On the analyze path, a server-key fallback fails closed when the call-site quota check cannot answer. That covers no accounting loop, a sync call on the loop's thread, and a timeout or error (`llm_quota_gate.py:91-133`). The rule is the owner's TASK-15481 fail-closed rule. MCP is unchanged: its reservation already fails closed. | Task 1. `check_user_sync(..., strict=True)` and `_override_outcome` refuse with `configuration`. Pinned by `test_a_fallback_no_quota_check_answers_for_is_refused[no-accounting-loop\|on-the-loop-thread\|check-times-out\|check-fails]`: over the limit, refused, nothing on `sk-server`. |
| I1 | One autouse fixture in the MCP tests directory's conftest gives the whole directory a healthy, empty override store; patching only `composed` is not enough. | Task 3 Step 1. `test_model_completion_adapter.py`: 136 passed, as on dev. |
| I2 | Only `model_not_allowed` falls back. `provider_disabled` refuses: `configuration` on analyze, as #3500 did; the policy error on MCP. | Tasks 1 and 3. `test_a_disabled_provider_or_an_unreadable_policy_refuses_the_call[provider-disabled-*]` and the MCP `[provider-disabled]` case. |
| I3 | `credential_store_unavailable` fails closed on the new paths: `configuration` on analyze, `ByokResolutionError("credential_store_unavailable")` on MCP (SHARED_INFRASTRUCTURE). `users_own_key_credentials` keeps its catch-all, unchanged. | Tasks 1 and 3. The `[override-cache-unhealthy-*]` cases and the MCP `[store-unavailable]` case. |
| I4 | Amend ADR-086 (Consequences) for the MCP denial fallback after the scope check, and refresh `Docs/Published`. | Task 4 Step 3. |
| I5 + gaps | File two follow-up tasks with backlog-py in this PR, and name both in the PR body. `TASK-15576`: the 19 resolve-time runtimes still refuse; the owner decides whether the rule extends to them. `TASK-15577`: `users_own_key_credentials`' catch-all runs `provider_disabled` and a store outage on the server key. The controller gives the two ids in the dispatch. | Task 5. |
| Minor 1 | The `test_model_completion_integration.py` baseline is 79, not 80. | Task 0 |
| Minor 2 | Commit the Flashcards `Docs/Published` drift that the refresh writes, and note it in the PR. | Task 4 Step 4, Task 5 |
| Minor 3 | Run lint with a `--basetemp` outside the repo. | Global Constraints, Task 5 |
| Minor 4 | Delete the model re-resolution; the fallback keeps `model`. | Task 1 Step 6 |
| Minor 5 | The BYOK guide sentence is appended to the same paragraph. | Task 4 Step 2 |
| Minor 6 | The integration file's new import goes after the `MCP_unified.tests` imports. | Task 3 Step 2 |

Re-review rulings (coordinator, 2026-11-19, on the "Re-review" section of `plan_review_15571.md`):

| Finding | Ruling | Applied in |
|---|---|---|
| I-R1 | `check_user_async` gets the same `strict` flag, returning whether the check answered. `media_ingest_jobs_worker._admit_or_refuse_analysis` marks the job admitted only when it did. Check the other callers that mark a request admitted. | Task 1. Pinned by `test_a_job_admitted_without_an_answer_is_not_marked_so_its_fallback_is_checked[check-times-out\|check-fails]`, which fails on the previous revision (the fallback ran on `sk-server`). The other callers are covered in Decision 5. |
| R6 | `user_to_check`'s own failure counts as no answer on the fallback path, so the fallback refuses. | Task 1: `user_to_check(strict=True)` raises, and `_override_outcome` reads that as no answer. Pinned by `test_a_fallback_whose_check_cannot_start_is_refused_with_its_own_message`. |
| Decision 1 / R5 | Keep `configuration`, with a distinct legacy message for "no quota answer": "Error: The LLM usage limit could not be checked." | Task 1 Step 6; the same test pins the text. |
| R1 | The `test_summarization_adapter.py` baseline is 128. | Task 0 |
| R2 | `tests/lint` runs with `--ignore=tldw_Server_API/tests/lint/test_runtime_package_discovery.py`. | Task 5 |
| R3 | `test_eval_runner.py` runs with pytest's default basetemp; an outside basetemp gives 46 environment-only failures. | Global Constraints, Task 1 Step 8 |
| R4 | The denied-model line is written after the quota decision, and states the outcome: one line per call. | Task 1 (`log_server_key_fallback`). The B1 test now asserts exactly one line. MCP writes its line when it moves to the server key. |
| R7 | The Usage_Quotas wording names the counter-failure exception and the worker's unmarked job. | Task 4 Step 1 |
| R8 | `TASK-15577` cites `users_own_key_credentials`' `except`, not a line number that Task 2 shifts. | Task 5 |

## Owner decision (AC #1), 2026-11-19

> A model the override policy denies falls back to the server key, whatever its source and whatever the key: a
> configured model, the DEFAULT_MODEL_* fallback, `model_override`, or the MCP authoritative path. The call is not
> refused for the denial itself.

This reverses PR #3500 (TASK-15496, merge b7d0b2dcb0), which failed a denied `DEFAULT_MODEL_*` fallback with
`configuration`. The fallback never runs an over-limit user on the server key: TASK-15481's rule is fail-closed.

## The code on origin/dev (e656a55f80), and after

| Model source | Handle on the user's, a team's or an org's key: dev | After (`model_not_allowed`) | No handle (server key) |
|---|---|---|---|
| configured | `users_own_key_credentials` resolves with the override and turns a denial into None, the server key (`usage_quota_deps.py:119-121`). The boundary then passes `own_key=False`, so an over-limit user gets a 402. The log names neither provider nor reason. A policy change after resolve is never checked at dispatch. | Resolve: the same, and the log names both. Dispatch: server key, after a strict quota check | runs, no check (unchanged) |
| `DEFAULT_MODEL_<PROVIDER>` | `configuration`, no log (`Summarization_General_Lib.py:326-336`) | server key, after a strict quota check | runs, no check |
| `model_override` | not checked; runs on the caller's key (`:319-320`) | server key, after a strict quota check | runs, no check |
| MCP authoritative (`settings.model`) | not checked; the caller's key (`provider_credential_runtime.py:521-538`) | scope checked, then a `server_default` handle; the reservation's checks apply | already `server_default`, not checked |

`provider_disabled` and `credential_store_unavailable` on a handle that is not the server's: dev refuses only the
`DEFAULT_MODEL_*` case. After the change, every case refuses: `configuration` on analyze, the policy error or
`credential_store_unavailable` on MCP.

Quota facts the design rests on:
- **Analyze path.** It is governed by the per-user LLM quota; the hosted plan of PR #3492 (TASK-15481) is only in
  `chat.py` and audio transcriptions. TASK-15426's `enforce_llm_quota(own_key=...)` admits an over-limit user who
  has a handle without the `prechecked` mark. The adapter wrappers (`usage_accounting.py:362-475`) then check
  each call whose request carries no runtime handle.
- **The gate admits a call it cannot check.** That happens with no accounting loop, on the loop's thread, on a
  timeout or error (`llm_quota_gate.py:91-160`), and when `user_to_check` fails. So the fallback runs its own
  strict check first (B1, R6).
- **The media ingest worker** marks a job admitted after `check_user_async` returns, which it also does when no
  answer came (`media_ingest_jobs_worker.py:390-399`). It now marks the job only when the check answered (I-R1).
- **Counter failures admit by design.** A usage counter that fails reads as no limit, at every check (spec 2 §2,
  `quota_checks.py:70-74`). That is an answer, not an unanswered check, and it is unchanged.
- **MCP.** `ModelCompletionAccounting.reserve` checks the user's quota (uncached, fail-closed) and the hosted
  plan's MCP token limit for every source that is not `user` (`accounting.py:176-255`).

## Global Constraints

- **No real outbound network calls.** Every pytest run loads the network probe:
  `PYTHONPATH=<plans dir>:<plans dir>/pyext NETPROBE_LOG=<file> python -m pytest -p no:cacheprovider -p netprobe_15427 --basetemp <dir outside the repo> -o cache_dir=<dir outside the repo> ...`.
  - The log must stay empty for every file this plan changes.
  - Every test that reaches an adapter stubs it with a recording adapter or a local `ChatProvider` subclass. A
    commercial-provider chat test that reaches a real adapter calls the real API.
- **PR base.** Branch from `origin/dev`; the PR targets `dev`.
- **Backlog edits.** Use only `tools/backlog-py`:
  `PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit|create ...`. Use `--append-notes`,
  never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit a task file.
- **Close-out.** Task 0 records the owner decision and these rulings in TASK-15571's notes; Task 5 checks ACs 1-4.
- **Follow-up ids.** Write the two ids exactly as `TASK-15576` and `TASK-15577` until the dispatch supplies them.
- **Scope ratchet.** If a file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` shifts lines, run
  `python Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline` in the same PR. Measured: no file this
  plan touches is listed. Task 5 runs the ratchet.
- **Whitespace.** No trailing whitespace in this plan or its embedded code (`git diff --check`).
- **Mirrored docs.** A doc mirrored under `Docs/Published` needs `bash Helper_Scripts/refresh_docs_published.sh`,
  with the mirror diff in the same PR. `BYOK_User_Guide.md` and ADR-086 are mirrored; `Usage_Quotas.md` is not.
- **Docs to update.** `Usage_Quotas.md` describes the rule (Task 4). No operator doc describes PR #3500's
  `configuration` error: measured, `git grep -n "Provider credential context is invalid" -- Docs` matches only
  `Docs/superpowers/plans/`. Plans are records; do not edit them.
- **Attribution.** Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The PR body ends
  with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Forbidden.** Never `--no-verify`, never force-push, never `git stash`. Use PostgreSQL only through the repo
  fixtures, and never `FLUSHALL` Redis. Nothing here uses either.
- **Imports.** Import `llm_provider_overrides`, `byok_runtime` and `llm_quota_gate` inside functions in
  `Summarization_General_Lib.py`, because of the LLM_Calls import-isolation ratchet. Import
  `llm_provider_overrides` inside a function in `provider_credential_runtime.py`, because of an import cycle.
- **Size and style.** No touched function is in `tests/lint/function_size_baseline.json`. Lines stay within 120
  characters.
- **ADR-004.** The controller asks for the Change summary; no recorded waiver covers TASK-15571.
- **Test isolation.** Run each test file in its own pytest process, with `--basetemp` outside the repo.
  - A basetemp inside the checkout makes four `tests/lint` ratchets see scratch files (plan review, Minor 3).
  - One exception: run `test_eval_runner.py` with pytest's default basetemp. An outside basetemp gives 46
    environment-only `InvalidStoragePathError` failures, the same on dev (R3).
  - Measured on dev: `test_byok_runtime_sqlite.py` fails 2 and errors 2 when it runs after the MCP transport and
    adapter files in one process.

## Review Focus

1. **An over-limit user's fallback never runs unchecked.** It is refused by the 402, or refused with
   `configuration` when no check answers. That covers no accounting loop (for example a standalone
   `media_ingest_jobs_worker`), the loop's thread, a timeout, an error, a `user_to_check` that fails, and a worker
   job whose start check got no answer. A user within the limit runs on the server key and is recorded.
   - Pinned by `test_a_denied_model_falls_back_to_the_server_key_under_the_users_limit` (4 cases),
     `test_a_fallback_no_quota_check_answers_for_is_refused` (4 cases),
     `test_a_fallback_whose_check_cannot_start_is_refused_with_its_own_message`, and
     `test_a_job_admitted_without_an_answer_is_not_marked_so_its_fallback_is_checked` (2 cases), all Task 1.
   - On MCP, pinned by `test_an_own_key_completion_whose_model_the_override_denies_obeys_the_users_limit`
     (Task 3).
2. **The fallback request never carries the caller's key or handle.** If it did, the call-site check would read
   `user` and skip the quota. Pinned by every `(model, "server-key", False)` assertion and by
   `test_a_denied_model_never_takes_the_callers_key_to_the_server_key_path` (Task 1).
3. **Only `model_not_allowed` falls back.**
   - A disabled provider or an unreadable store refuses, from any model source. Pinned by
     `test_a_disabled_provider_or_an_unreadable_policy_refuses_the_call` (6 cases, Task 1) and by MCP's
     `test_a_disabled_provider_or_an_unreadable_store_refuses_an_own_key_completion` (Task 3).
   - The resolve-time catch-all that still runs them on the server key is `TASK-15577`.
4. **The server key is checked by no one.** A call with no handle, or a `server_default` handle, runs a denied
   model with no check and no log. AC #4 is read per checked call. Pinned by
   `test_a_denied_model_on_the_server_key_path_runs_unchecked`,
   `test_a_denied_model_on_a_server_default_handle_runs_unchecked`, and MCP's `[server]` case.
5. **MCP checks the scope before it falls back.** A revoked scope still fails, the reservation records
   `server_default`, and no log carries a secret.
   - Pinned by `test_authoritative_runtime_checks_the_scope_before_a_denied_model_falls_back` and
     `test_a_model_the_provider_override_denies_runs_on_the_server_key` (Task 3).
   - `composed`'s teardown scans every log for the key sentinels.

---

### Task 0: Start

**Files:** `backlog/tasks/task-15571 - ...md`, `Docs/superpowers/plans/2026-11-19-denied-model-server-key-fallback.md`

- [ ] Create the branch: `git fetch origin dev && git switch -c fix/denied-model-server-key-15571 origin/dev`.
- [ ] Copy this plan to `Docs/superpowers/plans/2026-11-19-denied-model-server-key-fallback.md`.
- [ ] Record the decision and the rulings. The AC text stays as it is.

```bash
PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit TASK-15571 -s "In Progress" \
  --append-notes "Owner decision (AC #1), 2026-11-19: a model the provider-override policy denies falls back to the server key, whatever its source (configured model, DEFAULT_MODEL_<PROVIDER> fallback, model_override, the MCP authoritative path) and whatever the key (the user's own, a team's, an org's); the call is not refused for the denial itself. This reverses PR #3500 (TASK-15496), which failed a denied DEFAULT_MODEL_* fallback with 'configuration'. Coordinator rulings, 2026-11-19, on the plan review: only model_not_allowed falls back; a disabled provider refuses ('configuration' on analyze, the policy error on MCP); an override store that cannot be read fails closed ('configuration' on analyze, credential_store_unavailable on MCP); on analyze the fallback needs a quota check that answers, and refuses when none does (TASK-15481's fail-closed rule), with 'configuration' and the message 'The LLM usage limit could not be checked.'; a user_to_check failure is no answer; the media ingest worker marks a job admitted only when its start check answered; MCP's reservation already fails closed. With no handle, or a server_default handle, nothing changes and no check applies; AC #4 is read per checked call. AC #2's 'as chat does' is read under this rule: chat refuses a denied model with a 403; the analyze path falls back instead. Plan: Docs/superpowers/plans/2026-11-19-denied-model-server-key-fallback.md."
git add -A backlog/tasks Docs/superpowers/plans/2026-11-19-denied-model-server-key-fallback.md
git commit -m "chore(backlog): start TASK-15571 with the owner's rule, the rulings and the plan"
```

- [ ] Push, then `gh pr create --draft --base dev`.
- [ ] Baseline: run each file below in its own process and record the counts; Task 5 compares against them. Paths
  are under `tldw_Server_API/`. Dev counts:
  - `tests/LLM_Calls/test_provider_adapter_runtime_boundary.py`: 91
  - `tests/LLM_Calls/test_summarization_adapter.py`: 128
  - `tests/Usage/test_llm_quota_byok_calls.py`: 73
  - `tests/Usage/test_llm_quota_decision.py`: 35
  - `tests/Usage/test_llm_quota_gate.py`: 12
  - `tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py`: 8
  - `tests/AuthNZ_Unit/test_provider_credential_runtime.py`: 58
  - `app/core/MCP_unified/tests/test_model_completion_integration.py`: 79
  - `app/core/MCP_unified/tests/test_model_completion_quota_compatibility.py`: 7
  - `app/core/MCP_unified/tests/test_model_completion_adapter.py`: 136
  - `app/core/MCP_unified/tests/test_model_completion_transport.py`: 187
  - `app/core/MCP_unified/tests/test_model_completion_factory.py`: 89
  - The wider set:
    - `tests/LLM_Calls/test_summarization_runtime_credentials.py`
    - `tests/Chat/unit/test_authoritative_adapter_translation.py`
    - `tests/AuthNZ_Unit/test_byok_runtime.py`
    - `tests/AuthNZ_Unit/test_llm_provider_overrides.py`
    - `tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py`
    - `tests/Evaluations/unit/test_eval_runner.py`
    - `tests/Evaluations/unit/test_evaluation_manager.py`
    - `tests/Media/test_audio_summary_service_prompt.py`
    - `tests/Media/test_email_summary_service_prompt.py`
    - `tests/Media/test_media_llm_quota.py`
    - `tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py`
    - `tests/Services/test_outputs_service.py`
    - `tests/lint/test_llm_calls_import_isolation.py`
    - `tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py`
    - `tests/DB_Management/test_media_ingest_worker_content_scope.py`
    - `tests/MediaIngestion_NEW/unit/test_plaintext_analysis_outcomes.py`

### Task 1: The helpers, the strict quota checks, and the dispatch guard (AC #2, AC #4; B1, I2, I3, Minor 4, I-R1, R4-R6)

**Files:**
- Modify `tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py`: two new functions after
  `validate_provider_override` (`:888-889`).
- Modify `tldw_Server_API/app/core/Usage/llm_quota_gate.py`: `user_to_check` (`:50-72`), `check_user_sync`
  (`:89-125`) and `check_user_async` (`:128-160`).
- Modify `tldw_Server_API/app/services/media_ingest_jobs_worker.py`: `_admit_or_refuse_analysis` (`:390-399`).
- Modify `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`: the `core.exceptions` import
  (`:33-41`), `_model_allowed_by_override` (`:236-248`) and the model block (`:319-338`).
- Test `tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py`: `_issue_credentials`
  (`:47-82`), and the TASK-15496 block (`:432-554`).
- Test `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py` (`:833-849`).
- Test `tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py`: insert before `def _chat_request` (`:478`).

**Interfaces:**
- **Produces** `server_key_fallback(provider: str, model: str | None) -> bool` in `llm_provider_overrides`.
  - It returns True for `model_not_allowed` and False when the policy allows the model.
  - It raises `ProviderOverridePolicyError("provider_disabled", ...)` for a disabled provider, and lets
    `ByokResolutionError("credential_store_unavailable")` escape when the policy cannot be read. Both refusals
    log `f"Provider override for {canonical}: {reason}; the call is refused"`.
  - It does not log `model_not_allowed`; its caller does that once it knows the outcome. Task 3 calls it.
- **Produces** `log_server_key_fallback(provider: str, outcome: str) -> None` in `llm_provider_overrides`. It writes
  the one WARNING for a denied model: `f"Provider override for {canonical}: model_not_allowed; {outcome}"`.
- **Produces** three `strict` keywords in `llm_quota_gate`, all defaulting to False. Without `strict`, behaviour
  and log text are unchanged.
  - `user_to_check(credential_source=None, *, strict=False) -> int | None`: with `strict`, its own failure raises.
  - `check_user_sync(user_id, *, strict=False) -> bool`: with `strict`, a timeout or a failure returns False.
  - `check_user_async(user_id, *, strict=False) -> bool` now returns True once a decision ran (or, without
    `strict`, admitted after a failure). With `strict`, a timeout, a cancellation by the accounting loop or a
    failure returns False.
  - Strict outcomes log `...; call not admitted`.
- **Produces (private)** `_override_outcome(provider: str, handle: ProviderCallCredentials, model: str) -> str`,
  which returns `"handle"`, `"server_key"`, `"refused"` or `"unchecked"`. It replaces `_model_allowed_by_override`.
  Measured with `git grep`, the only other reference is the test stub that Step 2 removes.

**The PR #3500 assertions that flip:**
- `test_own_key_summary_default_model_environment_variable_meets_the_override_policy[model-not-allowed]`
  asserted `models == []` and `result == _POLICY_DENIED`. It now answers `"summary"` on `server-key` without
  the handle, and lives at `test_a_denied_model_on_the_users_own_key_runs_on_the_server_key[default-model-env]`.
- Its `[provider-disabled|override-cache-unhealthy]` cases keep the refusal and gain the log line. They move to
  `test_a_disabled_provider_or_an_unreadable_policy_refuses_the_call`, which now covers all three model sources.
- `test_own_key_summary_named_model_beats_the_environment_default_and_skips_the_fallback_policy[config-model|model-override]`
  asserted the model ran unchecked on the caller's key. It now runs on `server-key` without the handle (`...[configured|model-override]`
  above). Its comment ("checked at resolve time, not at this fallback") goes.
- The `server_default` handle with a denied `DEFAULT_MODEL_*` failed `configuration` on dev. It now runs
  unchecked on its own key (`test_a_denied_model_on_a_server_default_handle_runs_unchecked`).
- `test_summarization_adapter.py::test_explicit_missing_summary_model_uses_default_model_environment_under_the_override_policy`
  had a second half asserting `SummaryProviderError("configuration")` and one request. That call now answers
  `"Cedar summary"` with `("blocked-model", "server-key")`. The stub of the deleted `_model_allowed_by_override`
  becomes a real allowlist, and the test is renamed.
- These keep their behaviour in the new helper shape:
  `test_own_key_summary_falls_back_to_the_default_model_environment_variable`,
  `test_own_key_summary_with_an_allowlist_and_no_default_model_stays_missing_model`, and
  `test_explicit_missing_summary_model_without_default_model_environment_is_missing_model`.

- [ ] **Step 1: The dispatch tests.** In `_issue_credentials`, add `source: str = "user"` after
  `credential_fields` and pass `source=source` in place of `source="user"`. Then replace `:432-554` with:

```python
_POLICY_DENIED = "Error: Provider credential context is invalid."
_OVERRIDE_MODULE = "tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides"
# The one warning a denied model writes (TASK-15571).
_FALLBACK_LINE = "Provider override for openai: model_not_allowed; the call falls back to the server key"


def _refused_line(reason: str) -> str:
    """The one warning a refused call writes (TASK-15571)."""
    return f"Provider override for openai: {reason}; the call is refused"


async def _summary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source: str | None = "user",
    overrides: dict[str, Any] | None = None,
    healthy: bool = True,
    env_model: str | None = "gpt-env-default",
    config_model: str | None = None,
    model_override: str | None = None,
    server_key: str | None = "server-key",
) -> tuple[str | None, list[tuple[str, str | None, bool]], list[str]]:
    """Run one summary through the shared adapter path (TASK-15496, TASK-15571).

    ``source`` is whose key the handle carries; None sends no handle (the server-key path). The server's
    config has the same provider section as the handle's and its key is ``server_key``. Returns the result,
    (model, key, handle in the request) for each adapter request, and the override module's warnings.
    """
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        set_llm_provider_overrides_cache_for_tests,
    )

    openai_section = {"model": config_model} if config_model else {}
    handle = None
    if source is not None:
        handle = await _issue_credentials(
            "openai", api_key="handle-key", app_config={"openai_api": openai_section}, source=source
        )
    calls: list[tuple[str, str | None, bool]] = []
    warnings: list[str] = []

    class RecordingAdapter:
        def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            calls.append((request["model"], request["api_key"], PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY in request))
            return {"choices": [{"message": {"content": "summary"}}]}

    class Registry:
        def get_adapter(self, provider: str) -> RecordingAdapter:
            return RecordingAdapter()

    monkeypatch.setattr(sgl, "get_registry", lambda: Registry())
    monkeypatch.setattr(sgl, "load_and_log_configs", lambda: {"openai_api": dict(openai_section)})
    monkeypatch.setattr(sgl, "resolve_provider_api_key_from_config", lambda *_args, **_kwargs: server_key)
    if env_model:
        monkeypatch.setenv("DEFAULT_MODEL_OPENAI", env_model)
    else:
        monkeypatch.delenv("DEFAULT_MODEL_OPENAI", raising=False)
    set_llm_provider_overrides_cache_for_tests(overrides, healthy=healthy)
    sink = logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING", filter=_OVERRIDE_MODULE
    )
    try:
        result = sgl._summarize_via_adapter(
            api_name="openai",
            text_to_summarize="source",
            custom_prompt_arg=None,
            api_key=None,
            temp=0.2,
            system_message=None,
            streaming=False,
            model_override=model_override,
            app_config=None,
            credentials_resolved=False,
            provider_credentials=handle,
            raise_on_error=False,
        )
    finally:
        logger.remove(sink)
        set_llm_provider_overrides_cache_for_tests(None)
    return result, calls, warnings


def _allowlist(*models: str) -> dict[str, Any]:
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import LLMProviderOverride

    return {"openai": LLMProviderOverride(provider="openai", allowed_models=list(models))}


# Each source of the model: the handle's config, the DEFAULT_MODEL_OPENAI fallback, the request's model_override.
_MODEL_SOURCES = pytest.mark.parametrize(
    ("config_model", "model_override", "expected"),
    [("cfg-m", None, "cfg-m"), (None, None, "gpt-env-default"), (None, "ovr-m", "ovr-m")],
    ids=["configured", "default-model-env", "model-override"],
)


@pytest.mark.asyncio
async def test_own_key_summary_falls_back_to_the_default_model_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, warnings = await _summary(monkeypatch)

    assert result == "summary"
    assert calls == [("gpt-env-default", "handle-key", True)]
    assert warnings == []


@pytest.mark.asyncio
@_MODEL_SOURCES
async def test_a_denied_model_on_the_users_own_key_runs_on_the_server_key(
    monkeypatch: pytest.MonkeyPatch, config_model: str | None, model_override: str | None, expected: str
) -> None:
    """TASK-15571: whatever its source, a model the override denies runs on the server key, without the handle."""
    result, calls, warnings = await _summary(
        monkeypatch, overrides=_allowlist("x"), config_model=config_model, model_override=model_override
    )

    assert result == "summary"
    assert calls == [(expected, "server-key", False)]
    assert warnings == [_FALLBACK_LINE]


@pytest.mark.asyncio
@_MODEL_SOURCES
@pytest.mark.parametrize(
    ("override_kwargs", "healthy", "reason"),
    [({"is_enabled": False}, True, "provider_disabled"), ({}, False, "credential_store_unavailable")],
    ids=["provider-disabled", "override-cache-unhealthy"],
)
async def test_a_disabled_provider_or_an_unreadable_policy_refuses_the_call(
    monkeypatch: pytest.MonkeyPatch,
    override_kwargs: dict[str, Any],
    healthy: bool,
    reason: str,
    config_model: str | None,
    model_override: str | None,
    expected: str,
) -> None:
    """TASK-15571 rulings: only a denied model falls back. A disabled provider, or a policy that cannot be read,
    refuses the call with 'configuration' (as TASK-15496 did) and says why."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import LLMProviderOverride

    overrides = {"openai": LLMProviderOverride(provider="openai", **override_kwargs)}

    result, calls, warnings = await _summary(
        monkeypatch, overrides=overrides, healthy=healthy, config_model=config_model, model_override=model_override
    )

    assert result == _POLICY_DENIED
    assert calls == []
    assert warnings == [_refused_line(reason)]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["team", "org"])
async def test_a_denied_model_on_a_shared_key_runs_on_the_server_key(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """TASK-15571: the rule holds whatever the key, a team's or an org's too."""
    result, calls, warnings = await _summary(
        monkeypatch, source=source, overrides=_allowlist("x"), config_model="cfg-m"
    )

    assert result == "summary"
    assert calls == [("cfg-m", "server-key", False)]
    assert warnings == [_FALLBACK_LINE]


@pytest.mark.asyncio
async def test_a_denied_model_never_takes_the_callers_key_to_the_server_key_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK-15571: with no server key for the provider, the fallback request has no key, never the handle's."""
    _result, calls, warnings = await _summary(
        monkeypatch, overrides=_allowlist("x"), config_model="cfg-m", server_key=None
    )

    assert calls == [("cfg-m", None, False)]
    assert warnings == [_FALLBACK_LINE]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_model", "model_override", "expected"),
    [("cfg-m", None, "cfg-m"), (None, "ovr-m", "ovr-m")],
    ids=["configured", "model-override"],
)
async def test_an_allowed_model_stays_on_the_users_own_key(
    monkeypatch: pytest.MonkeyPatch, config_model: str | None, model_override: str | None, expected: str
) -> None:
    result, calls, warnings = await _summary(
        monkeypatch, overrides=_allowlist(expected), config_model=config_model, model_override=model_override
    )

    assert result == "summary"
    assert calls == [(expected, "handle-key", True)]
    assert warnings == []


@pytest.mark.asyncio
@_MODEL_SOURCES
async def test_a_denied_model_on_the_server_key_path_runs_unchecked(
    monkeypatch: pytest.MonkeyPatch, config_model: str | None, model_override: str | None, expected: str
) -> None:
    """TASK-15571: with no handle the call is already on the server key; no override check applies (unchanged)."""
    result, calls, warnings = await _summary(
        monkeypatch, source=None, overrides=_allowlist("x"), config_model=config_model, model_override=model_override
    )

    assert result == "summary"
    assert calls == [(expected, "server-key", False)]
    assert warnings == []


@pytest.mark.asyncio
async def test_a_denied_model_on_a_server_default_handle_runs_unchecked(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK-15571: a handle the runtime resolved to the server's key keeps it; the model is not checked."""
    result, calls, warnings = await _summary(monkeypatch, source="server_default", overrides=_allowlist("x"))

    assert result == "summary"
    assert calls == [("gpt-env-default", "handle-key", True)]
    assert warnings == []


@pytest.mark.asyncio
async def test_own_key_summary_with_an_allowlist_and_no_default_model_stays_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, warnings = await _summary(monkeypatch, overrides=_allowlist("gpt-allowed"), env_model=None)

    assert calls == []
    assert result == "Error: Model is required for provider 'openai'"
    assert warnings == []
```

- [ ] **Step 2: Flip the test in `test_summarization_adapter.py`.** Replace `:833-849` with the test below. It
  runs with no user in the call context, so no quota check applies. The root conftest's autouse
  `_healthy_empty_llm_provider_override_store` restores the store afterwards.

```python
@pytest.mark.unit
def test_explicit_missing_summary_model_uses_default_model_environment_and_a_denied_one_runs_on_the_server_key(
    monkeypatch,
):
    # TASK-15496 gave a handle call the DEFAULT_MODEL_<PROVIDER> fallback. TASK-15571: a fallback model the
    # provider override denies runs on the server key (under TASK-15496 it failed 'configuration').
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    adapter = _Adapter(response={"choices": [{"message": {"content": "Cedar summary"}}]})
    monkeypatch.setenv("DEFAULT_MODEL_MOONSHOT", "server-env-model")
    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(adapter))
    monkeypatch.setattr(sgl, "load_and_log_configs", dict)
    monkeypatch.setattr(sgl, "resolve_provider_api_key_from_config", lambda *_args, **_kwargs: "server-key")
    set_llm_provider_overrides_cache_for_tests(
        {"moonshot": LLMProviderOverride(provider="moonshot", allowed_models=["server-env-model"])}
    )

    assert _analyze_with_empty_moonshot_handle() == "Cedar summary"
    monkeypatch.setenv("DEFAULT_MODEL_MOONSHOT", "blocked-model")
    assert _analyze_with_empty_moonshot_handle() == "Cedar summary"

    assert [(request["model"], request["api_key"]) for request, _timeout in adapter.requests] == [
        ("server-env-model", "explicit-key"),
        ("blocked-model", "server-key"),
    ]
```

- [ ] **Step 3: The quota tests (B1, R4, R5, R6, I-R1).** In `test_llm_quota_byok_calls.py`:
  - change the import to `from collections.abc import AsyncIterator, Callable, Iterable`;
  - add `from loguru import logger` after `import pytest`;
  - insert this before `def _chat_request`:

```python
_OVERRIDE_MODULE = "tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides"
_NO_ANSWER_LINE = "Provider override for openai: model_not_allowed; no LLM quota check answered, so the call is refused"


async def _denied_model_fallback(
    monkeypatch: pytest.MonkeyPatch, *, streaming: bool = False, raise_on_error: bool = True
) -> tuple[list[str], Callable[[], str]]:
    """A summary on the user's own key whose model the override denies (TASK-15571).

    The provider is accounted and records the key each call runs on; the server's key is ``sk-server``.
    Returns those keys and the call, which raises on a refusal when ``raise_on_error``.
    """
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as sgl
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    keys: list[str] = []

    class _Summarizer(_Provider):
        """The accounted provider, recording the key each call runs on."""

        def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            """Record the key, then answer."""
            keys.append(request["api_key"])
            return {"choices": [{"message": {"content": "hi"}}], "usage": dict(USAGE)}

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
            """Record the key, then stream."""
            keys.append(request["api_key"])
            yield from super().stream(request, timeout=timeout)

    handle = await _handle("user")
    monkeypatch.setattr(sgl, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _provider: _Summarizer()))
    monkeypatch.setattr(sgl, "load_and_log_configs", dict)
    monkeypatch.setattr(sgl, "resolve_provider_api_key_from_config", lambda *_args, **_kwargs: "sk-server")
    set_llm_provider_overrides_cache_for_tests({"openai": LLMProviderOverride(provider="openai", allowed_models=["x"])})

    def _summarize() -> str:
        result = sgl._summarize_via_adapter(
            api_name="openai", text_to_summarize="source", custom_prompt_arg=None, api_key=None, temp=None,
            system_message=None, streaming=streaming, model_override="gpt-denied", app_config=None,
            credentials_resolved=False, raise_on_error=raise_on_error, provider_credentials=handle,
        )
        return result if isinstance(result, str) else "".join(result)

    return keys, _summarize


@pytest.mark.parametrize("streaming", [False, True], ids=["chat", "stream"])
@pytest.mark.parametrize("admitted", [False, True], ids=["over-limit-unmarked", "admitted-at-the-boundary"])
async def test_a_denied_model_falls_back_to_the_server_key_under_the_users_limit(
    refused, monkeypatch, admitted, streaming
) -> None:
    """TASK-15571: the fallback call carries no handle, so an over-limit user admitted unmarked on their own key is
    refused before the provider runs; an admitted user's fallback runs on the server key and counts."""
    keys, summarize = await _denied_model_fallback(monkeypatch, streaming=streaming)

    with llm_call_context(user_id=7, prechecked=admitted):
        if admitted:
            assert await asyncio.to_thread(summarize) == "hi"  # off the loop, where a check can run
        else:
            with pytest.raises(LLMQuotaExceededError):
                await asyncio.to_thread(summarize)
    await llm_accounting.drain_accounting()
    assert keys == (["sk-server"] if admitted else [])
    assert [row["credential_source"] for row in refused] == ([None] if admitted else [])


async def _slow_decision(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
    """A decision that outlasts the check's timeout."""
    await asyncio.sleep(1)
    return REFUSAL


async def _failing_decision(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
    """A decision that fails, like a broken pool."""
    raise RuntimeError("pool gone")


@pytest.mark.parametrize(
    ("accounting_loop", "decision", "on_the_loop"),
    [(False, None, False), (True, None, True), (True, _slow_decision, False), (True, _failing_decision, False)],
    ids=["no-accounting-loop", "on-the-loop-thread", "check-times-out", "check-fails"],
)
async def test_a_fallback_no_quota_check_answers_for_is_refused(
    refused, monkeypatch, accounting_loop, decision, on_the_loop
) -> None:
    """TASK-15571 (B1): where the call-site check would admit an over-limit user unchecked, the fallback is refused
    instead (TASK-15481's fail-closed rule), and nothing runs on the server key."""
    from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import SummaryProviderError

    if not accounting_loop:
        monkeypatch.setattr(llm_accounting, "_accounting_loop", None)
    if decision is not None:
        monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", decision)
        monkeypatch.setattr(llm_quota_gate, "CHECK_TIMEOUT_SECONDS", 0.05)
    keys, summarize = await _denied_model_fallback(monkeypatch)
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(message.record["message"]), level="WARNING", filter=_OVERRIDE_MODULE)
    try:
        with llm_call_context(user_id=7), pytest.raises(SummaryProviderError) as refusal:
            if on_the_loop:
                summarize()
            else:
                await asyncio.to_thread(summarize)
    finally:
        logger.remove(sink)
    assert (refusal.value.code, keys, lines) == ("configuration", [], [_NO_ANSWER_LINE])


async def test_a_fallback_whose_check_cannot_start_is_refused_with_its_own_message(refused, monkeypatch) -> None:
    """TASK-15571 (R6, R5): a check that cannot start is no answer either; a caller that takes error strings is told
    the limit could not be checked, not that the credentials are invalid."""

    def _broken_cache(_user_id: Any) -> dict[str, Any]:
        """Fail like a broken limits cache."""
        raise RuntimeError("cache broken")

    keys, summarize = await _denied_model_fallback(monkeypatch, raise_on_error=False)
    monkeypatch.setattr(llm_quota_gate, "cached_limits", _broken_cache)
    with llm_call_context(user_id=7):
        assert await asyncio.to_thread(summarize) == "Error: The LLM usage limit could not be checked."
    assert keys == []


@pytest.mark.parametrize("decision", [_slow_decision, _failing_decision], ids=["check-times-out", "check-fails"])
async def test_a_job_admitted_without_an_answer_is_not_marked_so_its_fallback_is_checked(
    refused, monkeypatch, decision
) -> None:
    """TASK-15571 (I-R1): the media ingest job's start check marks the job admitted only when the check answered.
    An over-limit owner's later denied-model fallback then meets the strict check; nothing runs on the server key."""
    from tldw_Server_API.app.services import media_ingest_jobs_worker as worker

    handle = await _handle("user")

    async def _credentials(_user_id: Any, _form: Any) -> Any:
        """The owner's own key."""
        return handle

    monkeypatch.setattr(worker, "media_analysis_requested", lambda _form: True)
    monkeypatch.setattr(worker, "media_analysis_credentials", _credentials)
    monkeypatch.setattr(worker, "media_analysis_provider", lambda _form: "openai")
    monkeypatch.setattr(llm_quota_gate, "CHECK_TIMEOUT_SECONDS", 0.05)
    keys, summarize = await _denied_model_fallback(monkeypatch)
    refusing = llm_quota_gate.llm_quota_refusal

    async def _job() -> None:
        """A job off the accounting loop, where a check can time out: no answer at the start, then one analysis."""
        monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", decision)
        assert await worker._admit_or_refuse_analysis(object()) is None
        monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", refusing)
        summarize()

    with llm_call_context(user_id=7), pytest.raises(LLMQuotaExceededError):
        await asyncio.to_thread(asyncio.run, _job())
    assert keys == []
```

  What the fixtures and setup do:
  - The `refused` fixture turns quotas on, puts user 7 over a limit, and makes the test's loop the accounting
    loop. `asyncio.to_thread` copies the call context.
  - In the I-R1 test, the job runs on a second loop, off the accounting loop, so its start check can time out
    there. The fixture's refusing decision is restored before the analysis, so the strict check answers with the
    402.

- [ ] **Step 4: RED.** Run each file in its own process. The counts below were measured with dev's sources and
  these tests.
  - `test_provider_adapter_runtime_boundary.py`: `13 failed, 91 passed`.
    - Five fail on the key (`handle-key, True`), because dev checks nothing at dispatch for them:
      `test_a_denied_model_on_the_users_own_key_runs_on_the_server_key[configured|model-override]`, both
      `[team|org]` cases, and `test_a_denied_model_never_takes_...`.
    - `...[default-model-env]` and the `server_default` case fail on `_POLICY_DENIED == 'summary'` (PR #3500).
    - `test_a_disabled_provider_or_an_unreadable_policy_refuses_the_call` fails in all 6 cases. The `configured`
      and `model-override` cases ran on the handle; the `default-model-env` cases refuse on dev but log nothing.
  - `test_summarization_adapter.py`: `1 failed, 127 passed`. The second call raises
    `SummaryProviderError(configuration)`.
  - `test_llm_quota_byok_calls.py`: `11 failed, 73 passed`. Every case runs on `sk-test`, the user's key.
  - Against the previous revision of this plan (the 16-file patch), the same file gave `7 failed, 77 passed`:
    - the 4 B1 cases logged a fallback that did not happen (R4);
    - the R6 case and both I-R1 cases ran on `sk-server`.
  - `test_llm_quota_gate.py` (`12 passed`) and `test_media_ingest_job_llm_quota_refusal.py` (`8 passed`) are
    unchanged pins.

- [ ] **Step 5: The helpers.** In `llm_provider_overrides.py`, after `validate_provider_override`:

```python
def server_key_fallback(provider: str, model: str | None) -> bool:
    """Whether a call for ``model`` on a key that is not the server's falls back to the server key (TASK-15571).

    True when the override's allowlist denies the model (``model_not_allowed``): the owner's rule. The caller
    logs that denial once it knows the outcome (``log_server_key_fallback``). A provider the override disables
    raises its ``ProviderOverridePolicyError``, and a policy that cannot be read raises
    ``ByokResolutionError("credential_store_unavailable")``: both refuse the call, and are logged here.
    """
    provider_norm = canonical_provider_name(provider)
    try:
        error = capture_provider_override_call_snapshot(provider_norm).policy_error(model)
    except ByokResolutionError as exc:
        logger.warning("Provider override for {}: {}; the call is refused", provider_norm, exc.code)
        raise
    if error is None:
        return False
    reason = error["error_code"]
    if reason != "model_not_allowed":
        logger.warning("Provider override for {}: {}; the call is refused", provider_norm, reason)
        raise ProviderOverridePolicyError(reason, provider_norm)
    return True


def log_server_key_fallback(provider: str, outcome: str) -> None:
    """The one warning for a denied model, written with what became of the call (TASK-15571)."""
    logger.warning("Provider override for {}: model_not_allowed; {}", canonical_provider_name(provider), outcome)
```

- [ ] **Step 6: The strict checks, the worker, and the guard.**

  **The gate.** In `llm_quota_gate.py`:
  - **`user_to_check`.** Add the keyword `strict: bool = False`, and add the docstring line "A failure here admits
    the call; ``strict`` raises it instead (TASK-15571)." In its `except`, put `if strict: raise` (on two lines)
    before the warning.
  - **`check_user_sync`.**
    - Add the keyword `strict: bool = False`, and set `outcome = "not admitted" if strict else "admitted"` at the
      top.
    - Change the docstring's last sentence to "A check that times out or fails admits the call and returns True;
      ``strict`` returns False for it too, for a caller that must not admit a call no check answered
      (TASK-15571)."
    - In the three branches that admit after a failure (the `run_coroutine_threadsafe` failure, the
      `TimeoutError` and the final `except Exception`), log `"...; call {}"` with `outcome`, and
      `return not strict` in place of `return True`.
  - **`check_user_async`.**
    - Change the signature to `(user_id: int, *, strict: bool = False) -> bool`, with the same `outcome`
      variable.
    - Change the docstring's last sentence to "A check that times out or fails admits the call and returns True;
      ``strict`` returns False for it, for a caller that must not admit a call no check answered (TASK-15571)."
    - In the `TimeoutError` branch, the cancelled-by-the-accounting-loop branch and the final `except Exception`,
      log `"...; call {}"` with `outcome` and `return not strict`. After `_raise_if_refused(refusal)`, add
      `return True`.
    - The usage-accounting callers ignore the return value, so they are unchanged.

  **The worker (I-R1).** In `media_ingest_jobs_worker._admit_or_refuse_analysis`, replace the `try`/`else` body
  with:

```python
        try:
            answered = await check_user_async(user_id, strict=True)
        except LLMQuotaExceededError as refusal:
            if handle is None or media_analysis_provider(form_data) is None:
                return refusal  # the server's key, or another LLM stage: refused as before, no key bound
        else:
            # A check that timed out or failed admits the job unmarked, so each call is checked (TASK-15571).
            if answered:
                mark_llm_quota_prechecked()
```

  **The guard.** In `Summarization_General_Lib.py`:
  - Add `LLMQuotaExceededError` to the `tldw_Server_API.app.core.exceptions` import, between `EgressPolicyError`
    and `NetworkError`.
  - Replace `_model_allowed_by_override` (`:236-248`) with:

```python
def _override_outcome(provider: str, handle: ProviderCallCredentials, model: str) -> str:
    """How a call on ``handle`` meets the provider override (TASK-15571).

    ``"handle"``: it stays on the handle, which is the server's key or holds an allowed model. ``"refused"``: a
    disabled provider, or a policy that cannot be read. ``"server_key"``: a denied model falls back, once the
    user's LLM quota answers for the call (the boundary admitted the request, or a strict check admits it now).
    ``"unchecked"``: no check answered, so the fallback is refused (TASK-15481's fail-closed rule). A check that
    refuses raises the 402.
    """
    if handle.credential_source in {"server_default", "none"}:
        return "handle"
    # Imported here: a module-level import would add AuthNZ modules to the LLM_Calls load ratchet.
    from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        log_server_key_fallback,
        server_key_fallback,
    )
    from tldw_Server_API.app.core.Usage.llm_quota_gate import check_user_sync, user_to_check

    try:
        if not server_key_fallback(provider, model):
            return "handle"
    except ByokResolutionError:  # logged where it was raised
        return "refused"
    try:
        user_id = user_to_check(strict=True)
        answered = user_id is None or check_user_sync(user_id, strict=True)
    except LLMQuotaExceededError:
        log_server_key_fallback(provider, "the user's LLM limit refuses the call on the server key")
        raise
    except Exception:  # noqa: BLE001 - the check could not start: no answer
        answered = False
    if not answered:
        log_server_key_fallback(provider, "no LLM quota check answered, so the call is refused")
        return "unchecked"
    log_server_key_fallback(provider, "the call falls back to the server key")
    return "server_key"
```

  In `_summarize_via_adapter`, replace `:326-338` (from `if not model:` in the `elif explicit_credentials:` branch
  through the `else:` branch) with the code below. Keep `:321-325` above it. Keep `model` as chosen: there is no
  re-resolution (Minor 4).

```python
        # Same fallback as the server-key path, DEFAULT_MODEL_<PROVIDER> (TASK-15496).
        model = model or resolve_provider_model(provider, effective_app_config)
    else:
        model = resolve_provider_model(provider, effective_app_config)
    outcome = None
    if model and provider_credentials is not None:
        outcome = _override_outcome(provider, provider_credentials, model)
    if outcome in {"refused", "unchecked"}:
        return _summary_failure(
            code="configuration",
            provider=provider,
            raise_on_error=raise_on_error,
            legacy_message=(
                "Error: The LLM usage limit could not be checked."
                if outcome == "unchecked"
                else "Error: Provider credential context is invalid."
            ),
        )
    if outcome == "server_key":
        # The same model on the server key, as a call with no handle (TASK-15571). The request then carries no
        # handle, so the call-site check counts it as a server-key call.
        provider_credentials = api_key = None
        explicit_credentials = False
        effective_app_config = ensure_app_config(load_and_log_configs())
```

  The rest of `_summarize_via_adapter` is unchanged.
  - With no handle, the existing code takes the key from `resolve_provider_api_key_from_config`, leaves
    `credentials_resolved` and the handle out of the request, and skips `missing_credentials`.
  - The adapter wrapper checks the fallback call once more, without `strict`. That second quota read happens only
    on this rare path.

- [ ] **Step 7: GREEN.** All measured, with an empty netprobe log:
  - `104 passed`, `128 passed` and `84 passed`;
  - `test_llm_quota_gate.py` `12 passed`;
  - `test_media_ingest_job_llm_quota_refusal.py` `8 passed`.

- [ ] **Step 8: Wider set.** Run the Task 0 wider set, each file alone, and expect the Task 0 counts. Run
  `test_eval_runner.py` with pytest's default basetemp (R3). Measured on the prototype:
  - `test_media_llm_quota.py` 35;
  - `test_summarization_runtime_credentials.py` 10, `test_authoritative_adapter_translation.py` 23,
    `test_llm_provider_overrides.py` 66;
  - `test_media_ingest_jobs_worker.py` 31, `test_media_ingest_worker_content_scope.py` 48,
    `test_plaintext_analysis_outcomes.py` 11.

  The rest of the wider set is in "Measured". The plan review measured `test_eval_runner.py` at 75 with the default
  basetemp.

- [ ] **Step 9: Commit** the seven files: `fix(llm): a denied model falls back to the server key at dispatch, fail-closed (TASK-15571)`.

### Task 2: The resolve-time path names the provider and the reason (AC #2 own-key configured model, AC #4)

**Files:**
- Modify `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py`: in `users_own_key_credentials`, the imports
  (`:100-103`) and the `except` (`:119-121`). The catch-all itself stays (I3).
- Test `tldw_Server_API/tests/Usage/test_llm_quota_decision.py`: insert before
  `test_enforce_on_own_key_asks_once_and_admits_only_with_a_handle` (`:380`).

**Interfaces:** none new. The log text is exactly
`f"Own-key credential resolution for {provider} failed: {reason}; the server key is used"`. The reason is chosen
in this order:
- `error.policy_code`, for a `ProviderOverridePolicyError`;
- `error.code`, for any other `ByokResolutionError`;
- `type(error).__name__` otherwise.

It is never `str(error)`.

- [ ] **Step 1: The test.**

```python
async def test_a_configured_model_the_override_denies_keeps_the_servers_key_and_its_limit(monkeypatch) -> None:
    """TASK-15571: denied at resolution, so no handle: the server's key, and an over-limit user is refused at the
    boundary. One warning names the provider and the reason, and no key."""
    from loguru import logger

    from tldw_Server_API.app.core.Provider_Credentials import provider_credential_runtime as runtime_module
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work, users_own_key

    users_own_key(monkeypatch, "openai")
    monkeypatch.setattr(runtime_module, "load_server_config_snapshot", lambda: {"openai_api": {"model": "gpt-denied"}})
    set_llm_provider_overrides_cache_for_tests({"openai": LLMProviderOverride(provider="openai", allowed_models=["x"])})
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING")
    try:
        assert await usage_quota_deps.users_own_key_credentials(1, "openai") is None
    finally:
        logger.remove(sink)
    assert warnings == ["Own-key credential resolution for openai failed: model_not_allowed; the server key is used"]
    refuse_llm_work(monkeypatch)
    with pytest.raises(LLMQuotaExceededError):
        await usage_quota_deps.enforce_llm_quota_on_own_key(1, "openai")
```

- [ ] **Step 2: RED.** Expect `1 failed, 35 passed` (measured), failing only on `warnings`; dev logs
  `'Own-key credential resolution failed'`. On dev, the `is None` assertion before it passes, and the 402 comes
  from code this plan does not change. So the existing path already falls back and already re-checks the quota at
  the boundary (`own_key=False`); this test pins both.
- [ ] **Step 3: Implement.** Add `from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError`
  to the function-local imports, and make the `except` body:

```python
        # A bounded code, never the message: a policy denial's code (TASK-15571), a resolution code, or the type.
        reason = getattr(error, "policy_code", None) or (
            error.code if isinstance(error, ByokResolutionError) else type(error).__name__
        )
        logger.bind(exception_type=type(error).__name__).warning(
            "Own-key credential resolution for {} failed: {}; the server key is used", provider, reason
        )
        return None
```

- [ ] **Step 4: GREEN.** Expect `36 passed` (measured) and an empty netprobe log. Then rerun
  `test_media_ingest_job_llm_quota_refusal.py` and `test_media_llm_quota.py`; their counts should not change.
- [ ] **Step 5: Commit** the two files: `fix(quota): an own-key resolution failure logs its provider and reason (TASK-15571)`.

### Task 3: The MCP authoritative path applies the rule (AC #3; I1, I2, I3, Minor 6)

**Files:**
- Modify `tldw_Server_API/app/core/Provider_Credentials/provider_credential_runtime.py`, in three places:
  - in `resolve`, the authoritative `_resolve_entry(` call (`:526-531`);
  - the `_resolve_entry` signature (`:720-728`);
  - after the resolver call (`:801-804`).
- Test `tldw_Server_API/app/core/MCP_unified/tests/conftest.py`: an autouse fixture before `mcp_ws_client`.
- Test `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py`: an import after the
  `MCP_unified.tests.test_model_completion_transport` imports, and two tests before `class Gate:` (`:376`).
- Test `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_quota_compatibility.py`: an import
  before `from tldw_Server_API.app.core.Usage import quota_resolver`, and a test before
  `test_own_key_still_obeys_explicit_mcp_operator_ceiling`.
- Test `tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py`: a test after
  `test_authoritative_runtime_revalidates_cached_credentials` (`:1847-1891`).

**Interfaces:**
- Consumes `server_key_fallback` (Task 1). Its raises fail the completion closed.
- Produces (private) `_resolve_entry(..., rejected_credential_generation=None, model: str | None = None)`. Only
  the authoritative call passes `model`.

- [ ] **Step 1: The directory's override store (I1).** `app/core/MCP_unified/tests` is outside
  `tldw_Server_API/tests`, so it runs on the import-time, unhealthy store. Once this task's runtime change reads
  that store, every key-seeded MCP test would fail closed with `credential_store_unavailable`. The plan review
  hung `test_model_completion_adapter.py` this way.

  In `conftest.py`, before `mcp_ws_client`, add:

```python
@pytest.fixture(autouse=True)
def _healthy_empty_llm_provider_override_store(monkeypatch):
    """Every test starts with the healthy, empty provider-override store app startup gives production.

    The tldw_Server_API/tests conftest does the same for its tests; this directory is outside it. The MCP
    authoritative path reads the store since TASK-15571, and fails closed when it cannot.
    """
    from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides

    monkeypatch.setattr(llm_provider_overrides, "_OVERRIDE_CACHE", {})
    monkeypatch.setattr(llm_provider_overrides, "_OVERRIDE_CACHE_HEALTHY", True)
    monkeypatch.setattr(llm_provider_overrides, "_OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS", True)
```

  A test's `set_llm_provider_overrides_cache_for_tests(...)` writes into the patched dict, so it is gone after
  the test.

- [ ] **Step 2: The tests.** In `test_model_completion_integration.py`, add
  `from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides` after the
  `MCP_unified.tests.test_model_completion_transport` imports (Minor 6). Then add these tests before
  `class Gate:`:

```python
@pytest.mark.parametrize("selected", ["user", "team", "org", "server"])
async def test_a_model_the_provider_override_denies_runs_on_the_server_key(composed, selected):
    """TASK-15571: the authoritative scope is still checked; then the operator's model, which the provider override
    denies, runs on the server key and is reserved as a server-key call. A server key is not checked again."""
    env = composed
    sources = ("user", "team", "org")
    await seed_keys(env, sources[sources.index(selected) :] if selected != "server" else ())
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"openai": llm_provider_overrides.LLMProviderOverride(provider="openai", allowed_models=["x"])}
    )
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    result = await env.build().complete(request(), invocation)
    assert result.content == CONTENT.replace("\r\n", "\n")
    assert [incoming.headers["authorization"] for incoming in env.provider.requests] == [f"Bearer {KEYS['server']}"]
    row = await assert_safe_surfaces(env, invocation, result=result)
    assert row["credential_source"] == "server_default"
    denials = [line for line, _extra, _exception in env.logs if "the call falls back to the server key" in line]
    assert len(denials) == (0 if selected == "server" else 1)
    assert all("Provider override for openai: model_not_allowed" in line for line in denials)


@pytest.mark.parametrize(
    ("override_kwargs", "healthy", "code", "domain"),
    [
        ({"is_enabled": False}, True, "invalid_provider_credentials", ModelFailureDomain.CREDENTIAL_SCOPE),
        ({}, False, "credential_store_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE),
    ],
    ids=["provider-disabled", "store-unavailable"],
)
async def test_a_disabled_provider_or_an_unreadable_store_refuses_an_own_key_completion(
    composed, override_kwargs, healthy, code, domain
):
    """TASK-15571: only a denied model falls back. A disabled provider and an unreadable store fail closed after the
    scope check, before any reservation or provider request."""
    env = composed
    await seed_keys(env, ("user",))
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"openai": llm_provider_overrides.LLMProviderOverride(provider="openai", **override_kwargs)}, healthy=healthy
    )
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    with pytest.raises(ModelCompletionFailure) as caught:
        await env.build().complete(request(), invocation)
    assert_failure(caught.value, code, domain)
    assert env.provider.requests == []
    assert await assert_safe_surfaces(env, invocation) is None
```

  In `test_model_completion_quota_compatibility.py`, add
  `from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides` and the MCP side of B1's
  constraint:

```python
async def test_an_own_key_completion_whose_model_the_override_denies_obeys_the_users_limit(composed, monkeypatch):
    """TASK-15571: the denied model falls back to the server key, so the user's limit applies before dispatch."""
    env = composed
    await seed_keys(env, ("user",))
    await set_user_limit(env, monkeypatch, USER_LIMITS[0])
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"openai": llm_provider_overrides.LLMProviderOverride(provider="openai", allowed_models=["x"])}
    )
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    port = env.build(server_config=VENDOR_CONFIG)
    try:
        with pytest.raises(ModelCompletionFailure) as error:
            await port.complete(request(), invocation)
        assert (error.value.code, error.value.domain) == ("model_quota_exceeded", ModelFailureDomain.REQUEST)
        assert env.provider.requests == []
    finally:
        quota_resolver.invalidate_all()
```

  In `test_provider_credential_runtime.py`, add a pin for Review Focus 5. It passes on dev too, and catches the
  check moving ahead of the scope check:

```python
@pytest.mark.asyncio
async def test_authoritative_runtime_checks_the_scope_before_a_denied_model_falls_back(monkeypatch):
    """TASK-15571: a revoked scope fails as before; the override's server-key fallback never widens it."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.repos.provider_scope_result import (
        ProviderScopeResult,
        ProviderScopeStatus,
    )
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    class Repo:
        async def resolve_authorized_secret(self, user_id, provider):
            return ProviderScopeResult(ProviderScopeStatus.UNAUTHORIZED)

    async def repo():
        return Repo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    set_llm_provider_overrides_cache_for_tests({"openai": LLMProviderOverride(provider="openai", allowed_models=["x"])})
    runtime = ProviderCredentialRuntime(
        user_id=7,
        team_ids=[],
        org_ids=[],
        trusted_base_url_override=False,
        authoritative_scope=byok_runtime.AuthoritativeProviderScope(7),
        server_config_snapshot={"openai_api": {"api_key": "server-key", "api_base_url": "https://fixed.example/v1"}},
    )
    try:
        with pytest.raises(ByokResolutionError, match="credential_scope_revoked"):
            await runtime.resolve("openai", model="denied-model")
    finally:
        await runtime.close()
```

- [ ] **Step 3: RED.** These counts were measured with dev's sources plus Steps 1 and 2.
  - `test_model_completion_integration.py`: `5 failed, 80 passed`. The `[user|team|org]` cases fail on the
    header, for example `'Bearer sk-integration-team-sentinel' != 'Bearer sk-integration-server-sentinel'`. Both
    refusal cases fail with `DID NOT RAISE`. `[server]` passes.
  - `test_model_completion_quota_compatibility.py`: `1 failed, 7 passed`. The user's own vendor key skips the
    limit, so nothing is raised.
  - `test_provider_credential_runtime.py`: `59 passed`. The new test is a pin.
- [ ] **Step 4: Implement.** In `resolve`, pass `model=model` in the authoritative `_resolve_entry(...)` call. Add
  `model: str | None = None` as the last parameter of `_resolve_entry`. Directly after
  `resolution = await self._resolver(provider, **resolver_kwargs)`, inside the same `try`, add:

```python
            if (
                self._authoritative_scope is not None
                and isinstance(resolution, ResolvedByokCredentials)
                and resolution.uses_byok
            ):
                # Imported here: the override module imports the adapter registry, which imports this module.
                from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
                    log_server_key_fallback,
                    server_key_fallback,
                )

                # Raises, so fails closed, for a disabled provider or a policy that cannot be read (TASK-15571).
                if server_key_fallback(provider, model):
                    # The scope's authority was checked above. A model the override denies runs on the server key
                    # from the same frozen snapshot the scope falls back to (TASK-15571).
                    log_server_key_fallback(
                        provider, "the call falls back to the server key, which its reservation checks"
                    )
                    resolution = await self._resolver(
                        provider,
                        user_id=None,
                        team_ids=[],
                        org_ids=[],
                        server_config_snapshot=copy.deepcopy(self._server_config_snapshot),
                        force_oauth_refresh=False,
                        trusted_base_url_override=False,
                    )
```

  - Called with `user_id=None`, `resolve_byok_credentials` returns the same `_fallback_result` that the
    authoritative path ends with when no key exists (`byok_runtime.py:2595-2610` and `:2963-2971`).
  - `_new_handle` still takes the endpoint from the authoritative snapshot, so the MCP transport accepts the handle.
  - Both raises are `ByokResolutionError`s, so `except ByokResolutionError` detaches them and the adapter maps
    them. `invalid_provider_credentials` maps to CREDENTIAL_SCOPE, and `credential_store_unavailable` maps to
    SHARED_INFRASTRUCTURE.
  - `uses_byok` covers only `user`, `team` and `org`, so the server key is never checked.
- [ ] **Step 5: GREEN.** Expect `85 passed`, `8 passed` and `59 passed` (measured), with an empty netprobe log.
  Then run these, each alone:
  - `test_model_completion_adapter.py`: 136, as on dev (I1).
  - `test_model_completion_transport.py`: 187.
  - `test_model_completion_factory.py`: 89.
  - `test_byok_runtime.py`: 198 passed and 1 skipped.
  - `test_byok_runtime_sqlite.py`: 25.

  All of these were measured. In one process, the whole `app/core/MCP_unified/tests` directory measured 25 failed
  and 53 errors on both dev and the change, with identical test ids. The change adds 7 passes (4266 against 4259).
- [ ] **Step 6: Commit** the five files: `fix(mcp): a denied model falls back to the server key after the scope check, fail-closed otherwise (TASK-15571)`.

### Task 4: Docs (I4, Minors 2 and 5)

**Files:**
- `Docs/Operations/Usage_Quotas.md:120`
- `Docs/User_Guides/Server/BYOK_User_Guide.md:97`
- `Docs/ADR/086-authoritative-provider-credential-scope.md`
- the `Docs/Published` mirrors of the last two, plus `Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`

- [ ] **Step 1: `Usage_Quotas.md`.** Append the following to the paragraph at `:120`, after "...a user without
  one keeps the server's key and the call path the feature had.". Keep the `TASK-15577` placeholder until Task 5:

> A call whose model the admin provider override does not allow (outside its `allowed_models`) runs on the server's key instead, whichever key it had (the user's own, a team's or an org's) and wherever its model came from (the provider's configured model, `DEFAULT_MODEL_<PROVIDER>`, or a model the request names). That call counts like any other server-key call: over a limit it is refused, and so is a call whose limit no check answered (no accounting loop, as in a job worker run outside the server process, or a check that times out or fails); a media ingest job whose start check got no answer is not marked admitted, so its calls are each checked. As with every LLM quota, a usage counter that fails is read as no limit, so it admits the call. A provider the override disables, or an override store that cannot be read, refuses such a call instead; when the request's own-key lookup meets either one first, the request keeps the server's key, as for a user without a stored key (TASK-15577). The server logs one warning per such call, naming the provider and the reason (`model_not_allowed`, `provider_disabled` or `credential_store_unavailable`), never a key. The MCP model-completion tool follows the same rule for its configured model, after its scope check; its reservation then counts and checks the call as a server-key call (TASK-15571).

- [ ] **Step 2: The BYOK guide (Minor 5).** Append this sentence to the same paragraph at `:97`, directly after
  "...so those calls are recorded as yours and do not count.":

> One of those calls whose model the operator's provider override does not allow runs on the server's key instead: it counts, and over a limit, or when the limit cannot be checked, it is refused.

- [ ] **Step 3: ADR-086 (I4).** Add this paragraph to Consequences, after its second paragraph ("Checks are
  authoritative at the database read; ..."):

> Amendment (TASK-15571, 2026-11-19): after every supplied scope is validated, a user, team or organization credential whose model the admin provider override's allowlist denies (`model_not_allowed`) is replaced by the server credential from the same frozen snapshot. This is the owner's rule that a denied model falls back to the server key. It is a policy fallback after authority is established, not a fallback on absence, and the reservation then counts and checks the call as a server-key call. A provider the override disables fails closed with its policy error, and an override store that cannot be read fails closed with `credential_store_unavailable`, as other store failures do. A server credential is not checked.

- [ ] **Step 4: Refresh the mirrors (Minor 2).** Run `bash Helper_Scripts/refresh_docs_published.sh`. Commit
  every mirror it writes. That includes `Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`,
  existing drift from TASK-15569's 63598e8fc2 (+9/-1; measured: the source guide's "Create a Study Pack from
  Media" section is missing from the mirror).
- [ ] **Step 5:** Run `tldw_Server_API/tests/Docs`. The plan review measured `214 passed` after the refresh. On
  dev, without it, `test_docs_published_refresh.py` fails 2.
- [ ] **Step 6: Commit** `docs: a denied model falls back to the server key; ADR-086 amendment; mirror refresh (TASK-15571)`.

### Task 5: Follow-ups, verify, close out, review, PR (I5, Minor 3)

- [ ] **File the two follow-ups** with the ids the dispatch gives. Write them as `TASK-15576` and `TASK-15577`
  until then.

```bash
PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task create \
  "Decide whether the resolve-time provider-override runtimes follow TASK-15571's server-key fallback" \
  --id TASK-15576 -s "To Do" -l llm,byok,policy \
  -d "Found by TASK-15571's plan review (I5). TASK-15571 applies the owner's rule (a model the provider-override policy denies falls back to the server key) at analyze dispatch (_summarize_via_adapter), in users_own_key_credentials, and on the MCP authoritative path. The 19 other runtimes built with override_snapshot_resolver still refuse a denied model at resolve time (ProviderCredentialRuntime._capture_override_snapshot / _enforce_call_policy). That covers chat, character chat, chat documents, messages, audio streaming, TTS and realtime, RAG unified, evaluations unified, Prompt Studio and its jobs worker, chunking, moderation, the auto boundary assistant and speech chat. Chat also answers 403 per request (validate_provider_override, chat.py:4468). Some of their handles reach _summarize_via_adapter, for example RAG HyDE (hyde.py:132), so a denied configured model still has two outcomes by endpoint." \
  --ac "The owner decides whether the server-key fallback extends to these runtimes and to chat's 403" \
  --ac "If it does: each runtime family falls back with the same fail-closed quota check, with tests; if not: the docs say which endpoints refuse"
PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task create \
  "users_own_key_credentials runs a disabled provider or an unreadable override store on the server key" \
  --id TASK-15577 -s "To Do" -l llm,byok,policy \
  -d "Found by TASK-15571's plan review. The except clause of users_own_key_credentials (API_Deps/usage_quota_deps.py) catches every resolution failure and returns None, so the request keeps the server key, and the no-handle analyze path checks nothing. A provider the override disables, or an override store that cannot be read, therefore still runs an own-key user's analysis on the server key. That is unlike TASK-15571's dispatch rule, which refuses both (provider_disabled, credential_store_unavailable)." \
  --ac "A disabled provider and an unreadable override store refuse an own-key analyze request at the boundary instead of running it on the server key, with tests" \
  --ac "The other resolution failures keep their current server-key behaviour, or the owner decides otherwise"
git add -A backlog/tasks
```

  Replace `TASK-15577` in `Docs/Operations/Usage_Quotas.md` with the real id.
- [ ] **Verify.** Rerun every file in the Task 0 list, each alone, and compare against the Task 0 baseline. A
  failure counts as environment-only only if it also fails on the baseline.
- [ ] **Docs, lint and CI suites.** Run `tests/Docs`, `tests/lint` and `tests/CI`, with `--basetemp` outside the
  repo (Minor 3).
  - Run `tests/lint` with `--ignore=tldw_Server_API/tests/lint/test_runtime_package_discovery.py` (R2). Without
    setuptools, that module is a collection error that stops the whole run.
  - Known environment-only failures: the 2 jobs-postgres coverage cases, and the CI cases that need a `python` on
    PATH.
  - The plan review measured `tests/CI` with `6 failed`, all environment-only and the same on dev. It measured
    `tests/lint` (ignore set, basetemp outside) at `318 passed`.
- [ ] **Ratchets and static checks.**
  - `python Helper_Scripts/ci/test_quality_triage.py --enforce`: exit 0 measured.
  - `python Helper_Scripts/ci/check_shard_coverage.py`: no new test file.
  - `python Helper_Scripts/ci/check_backlog_task_ids.py`: must exit 0 with the two new tasks.
  - `python Helper_Scripts/ci/scope_predicate_ratchet.py`: passes with no baseline change.
  - Bandit on the six source files, with `uvx bandit -q <files>` where available; otherwise record it as not run
    under DoD 4.
  - `git diff --check origin/dev...`.
- [ ] **Close out the backlog task.**

```bash
PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit TASK-15571 \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "<measured RED/GREEN counts per file, the wider set, the rulings as applied, the Bandit result or why it was not run, and follow-ups TASK-15576 (the resolve-time runtimes) and TASK-15577 (users_own_key_credentials' catch-all)>" \
  --final-summary "<one rule: a model the override denies falls back to the server key at analyze dispatch and on the MCP authoritative path, after a quota check that answers (fail-closed); a disabled provider or an unreadable store refuses; the resolve-time path logs provider and reason; ADR-086 amended; docs; follow-ups TASK-15576, TASK-15577>" \
  -s Done
git add -A backlog/tasks Docs/Operations/Usage_Quotas.md
git commit -m "chore(backlog): close TASK-15571; file TASK-15576 and TASK-15577"
```

- [ ] **Review and PR.**
  - Run a clean-session final review (`/superpowers:requesting-code-review`) and address its findings.
  - The PR body names both follow-ups (`TASK-15576`, `TASK-15577`) and the Flashcards mirror drift.
  - Ask the controller for the ADR-004 Change summary.
  - Mark the PR ready, with base `dev`. The merge goes through the controller's queue.

## Decisions left to the reviewer

1. **The code for a fallback no check answers stays `configuration` (ruled).** It is non-retryable, and every
   caller already maps it. Callers that take error strings get their own message, "Error: The LLM usage limit
   could not be checked." (R5). The policy refusals keep "Error: Provider credential context is invalid.".
2. **`server_default` and `none` handles are not checked.** Dropping such a handle would swap an admin-override
   server key for the config key. AC #4 is read per checked call.
3. **One log line per denied call.**
   - It is written after the quota decision and states the outcome (R4):
     - "...; the call falls back to the server key";
     - "...; no LLM quota check answered, so the call is refused";
     - "...; the user's LLM limit refuses the call on the server key".
   - A strict check that cannot answer also gets the gate's own warning about the check: "...; call not
     admitted", or the once-per-process note for no accounting loop or the loop's thread.
   - A media request of N chunks logs N lines.
4. **The strict check reads the quota a second time.** The adapter wrapper checks the fallback call again,
   without `strict`. That costs one more quota read, only on this rare path. Do not suppress it.
5. **The other callers that mark a request admitted (I-R1's audit) need no change.**
   - **`enforce_llm_quota`** (`usage_quota_deps.py:143-145`) awaits `llm_quota_refusal` directly, so a failure
     raises instead of admitting. Its admission on a failed usage counter is spec 2 §2's by-design "no limit".
     The strict check returns the same answer, and Usage_Quotas now says so (R7).
   - **The gateway's two `llm_call_context(prechecked=True)` blocks** (`gateway.py:1318` after a non-strict sync
     check, `:1370` after the async one) wrap only that one adapter call. `_summarize_via_adapter` never runs
     inside them, so `_override_outcome` never reads that mark.
   - **`Moderation/classification.py:152`** sets `prechecked=False`.
   - **The media ingest worker** was the one caller that marked after an unanswered check. It is fixed (Task 1).

## What the task and brief got wrong on dev

- PR #3492 (TASK-15481) is chat's hosted plan, not the analyze path's mechanism. `admit_media_analysis`,
  `users_own_key_credentials` and the mark helpers come from TASK-15426 (PRs #3444-#3472).
- `users_own_key_credentials` (its `except`, `usage_quota_deps.py:119-121` on dev) already re-checks quota at the
  boundary. Its catch-all also runs `provider_disabled` and store outages on the server key (`TASK-15577`).
- AC #2's "as chat does" conflicts with the owner's rule on the server-key path. It is recorded as a reading in the
  notes.
- AC #3's path is reachable only from the MCP adapter. The guard sits in the runtime's authoritative branch,
  because the MCP transport accepts only an authoritative server-config endpoint.
- `app/core/MCP_unified/tests` lacks the root conftest's healthy override store (I1).
- The media ingest worker marked a job admitted after a start check that got no answer (I-R1).

## Measured (scratch copies of e656a55f80, one process per file, every netprobe log empty)

| File | dev sources + new tests | change |
|---|---|---|
| `test_provider_adapter_runtime_boundary.py` | 13 failed, 91 passed | 104 passed |
| `test_summarization_adapter.py` | 1 failed, 127 passed | 128 passed |
| `test_llm_quota_byok_calls.py` | 11 failed, 73 passed (previous revision: 7 failed, 77 passed) | 84 passed |
| `test_llm_quota_decision.py` | 1 failed, 35 passed | 36 passed |
| `test_llm_quota_gate.py` | 12 passed | 12 passed |
| `test_media_ingest_job_llm_quota_refusal.py` | 8 passed | 8 passed |
| `test_model_completion_integration.py` | 5 failed, 80 passed (pristine dev: 79) | 85 passed |
| `test_model_completion_quota_compatibility.py` | 1 failed, 7 passed | 8 passed |
| `test_provider_credential_runtime.py` | 59 passed (a pin) | 59 passed |
| `test_model_completion_adapter.py` | 136 passed | 136 passed |
| `test_model_completion_transport.py` / `_factory.py` | 187 / 89 (plan review, dev) | 187 / 89 passed |

- The wider set on the change, each file alone:
  - `test_media_llm_quota.py` 35;
  - `test_summarization_runtime_credentials.py` 10, `test_authoritative_adapter_translation.py` 23,
    `test_llm_provider_overrides.py` 66;
  - `test_evaluation_manager.py` 56, `test_audio_summary_service_prompt.py` 22,
    `test_email_summary_service_prompt.py` 25;
  - `test_outputs_service.py` 25, `test_llm_calls_import_isolation.py` 7;
  - `test_byok_runtime.py` 198 passed and 1 skipped, `test_byok_runtime_sqlite.py` 25;
  - `test_llm_usage_recorded_end_to_end.py` 1, `test_llm_roots.py` 8;
  - `test_media_ingest_jobs_worker.py` 31, `test_media_ingest_worker_content_scope.py` 48,
    `test_plaintext_analysis_outcomes.py` 11.
- The whole `app/core/MCP_unified/tests` directory, in one process, gave the same 25 failures and 53 errors on
  pristine dev and on the change; the change adds 7 passes. That was measured before the MCP log-line change,
  which only adds a log call.
- `ruff` (E501, F) reports nothing new.
- `git apply --check proto_15571.patch` on `origin/dev` is clean (17 files).
- Measured only by the plan review in a full checkout: `tests/Docs` 214 after the refresh, `tests/CI`
  (environment-only failures), `tests/lint` 318, and `test_eval_runner.py` 75 (default basetemp).
- Bandit is not in the shared venv.
