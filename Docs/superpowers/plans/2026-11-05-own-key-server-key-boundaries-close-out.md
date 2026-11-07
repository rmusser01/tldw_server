# The boundaries that never run on a stored key, ADR-083's amendment and the close-out (TASK-13956, PR C of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin that the boundaries whose LLM calls run on the server's key (media analysis and the media ingest job's start, the flashcards and quizzes study assistants, watchlist summaries, evaluation runs, benchmarks) keep refusing an over-limit user whatever keys the user stored; say so in the docs and in ADR-083's amendment for all three TASK-13956 PRs; amend the two ACs whose premise does not hold; file the follow-up that would make those features use a stored key; close TASK-13956.

**Architecture:** No source change. The task's premise for these boundaries was that knowing the provider is enough. It is not: their calls never use a stored user key (no credential handle, no context mark), so they are the operator's spend and count (ADR-083), and naming their provider would only move the 402 into the middle of the work (after a download or a transcription). One test helper, `own_keys_for_every_provider`, makes every provider look like the user's own key and records each lookup; each family's existing refusal test uses it and asserts the boundary still refuses without asking.

**Tech Stack:** pytest, backlog-py, the docs mirror script.

**Spec:** `backlog/tasks/task-13956 - ...md` (ACs #1 and #2, amended in Task 0; AC #3 is PR A's); the brief `brief_plan_13956.md`; ADR-083; plans `plan_13956_a.md` and `plan_13956_b.md`.

**Measured on:** `origin/dev` at `484e393333` (2026-11-05, the merge of #3420; since `6e2dc0c2dc` it changed only a workflow file and one backlog task file), with `Docs/Operations/Usage_Quotas.md` from PR #3422's head `9791e52407`, then `proto_13956_a.patch` and `proto_13956_b.patch` applied (PR A goes after #3422, PR C after A and B). The prototype is `proto_13956_c.patch` (plans dir; relative to that A+B tree; 15 files, +146/-22). Its test hunks apply to `484e393333` on their own; its docs hunks apply on top of A and B. The ADR block below carries `TASK-15426`: Task 0 Step 3 allocates the id (under the lock), Task 2 Step 4 writes it into the ADR, and Task 2 Step 6's grep checks that no `TASK-15426` or `TASK-<FOLLOW_UP>` is left outside the plan copy.

**Order:** last, after PRs A and B merge (and so after #3422).

---

## What was verified (on `6e2dc0c2dc`; the cited files are unchanged on `484e393333`)

| # | Boundary (`file:line`) | How its LLM calls get a key | Verdict |
|---|---|---|---|
| V1 | Media analysis: `media/add.py:85`, `ingest_web_content.py:80`, `process_audios.py:161`, `process_documents.py:152`, `process_ebooks.py:209`, `process_emails.py:98`, `process_pdfs.py:159`, `process_videos.py:139`, `process_web_scraping.py:79` | Every analysis call is `Summarization_General_Lib.analyze(api_name, ..., api_key=None)` with no `provider_credentials` (`Audio/Audio_Files.py:1193`, `:1316`; `Books/Book_Processing_Lib.py:868-1442`; `Email/Email_Processing_Lib.py:443`; `PDF/PDF_Processing_Lib.py:1176`, `:1227`; `Plaintext/Plaintext_Files.py:512`, `:546`; `XML_Ingestion_Lib.py:155`; `persistence.py:1664-1737`; the video batch says "api_key removed - retrieved from server config", `video_batch.py:119`). `_summarize_via_adapter` then takes the key from the server's config (`Summarization_General_Lib.py:319-327`). No media module resolves BYOK (`git grep -l "resolve_byok_credentials\|provider_credentials=" -- tldw_Server_API/app/api/v1/endpoints/media tldw_Server_API/app/core/Ingestion_Media_Processing`: nothing). | keep refusing |
| V2 | The media ingest job's start (`services/media_ingest_jobs_worker.py:363-380`) | The same libraries, through `process_batch_media` (`:639`). The check is `user_to_check()` and `check_user_async`, then `mark_llm_quota_prechecked()` (`:379`): the admission marker that keeps an admitted job from being refused part-way (TASK-13864). | keep refusing, keep the marker |
| V3 | Study assistants: `flashcards.py:2466`, `quizzes.py:784` (`Depends(require_llm_quota)`) | `generate_study_assistant_reply` calls `perform_chat_api_call_async(api_provider=..., model=...)` with no key or handle (`core/Flashcards/study_assistant.py:457-468`): the server's key. | keep refusing; the dependency stays |
| V4 | Watchlist summaries (`watchlists.py:7119`) | `summarize_items_for_output` → `_llm_reply` → `analyze(api_name, prompt, None, ...)` (`services/outputs_service.py:652-715`). | keep refusing |
| V5 | Evaluation runs (`evaluations_crud.py:328`) | The run's task uses `EvaluationRunner`, whose key is `eval_spec/run_config["api_key"]` or `resolve_provider_api_key_from_config` (`core/Evaluations/eval_runner.py:108-218`). | keep refusing |
| V6 | Benchmarks (`evaluations_benchmarks.py:109`, `benchmark_api.py:226`) | `evaluate_custom_metric(api_name=..., api_key=request.api_key)` (`benchmark_api.py:280-289`): a raw key from the request or the server's. | keep refusing |
| V7 | Why naming the provider would be wrong here | `enforce_llm_quota(provider=X)` lets an over-limit user through unmarked when `users_own_key_serves(X)`; the calls above then carry no source (`call_credential_source` → None, `Usage/llm_call_context.py:81-101`), so each is checked at the call and refused there. Measured: making `/media/add` pass `provider=form_data.api_name` turns `test_other_media_handlers_refuse_analysis_for_an_over_quota_user[add]` (with the new helper) into a failure. Without a stub that admitted request ran analysis through the real adapter and sent `POST https://api.openai.com/v1/chat/completions` (review C I2: the check at the call does not see the user on this path, so `refuse_llm_calls` does not stop it); the pin now stubs `Summarization_General_Lib._summarize_via_adapter`, the choke point every `analyze` call goes through. | pin |
| V8 | The routes, OpenAPI and route-auth if the study-assistant dependency became an inline call (the brief's question) | Not converted (V3). TASK-13955 measured the same change for the Messages routes: no OpenAPI change. The study-assistant routes also depend on `get_chacha_db_for_user`, so they stay authenticated without `require_llm_quota`. | not applicable |
| V9 | Open PRs on these files (re-checked 2026-11-05 ~12:00, `gh api .../pulls/<n>/files --paginate` for all 38 open PRs on `dev`; heads in brackets) | Source this PR only reads: `media/process_*.py` (6), `quizzes.py`, `media_ingest_jobs_worker.py`: #3394 [`82a7534446`]; `flashcards.py`, `watchlists.py`, `media_ingest_jobs_worker.py`: #3269 [`81b0bb5836`]; `watchlists.py`: #2903; `media_ingest_jobs_worker.py`: #2738. This PR changes none of those source files. Its own files: #3269 also edits `test_flashcards_endpoint_integration.py` (one hunk at its `:5698`) and `test_watchlists_api.py` (`:305`); this PR's edits there are single tests at `:3323` and `:1919`, no conflict (review C M10). `Usage_Quotas.md`: #3422 [`9791e52407`] rewrites the Service tiers, Gemini and "Streams without a usage chunk" bullets; this PR edits PR B's own-key bullet, which applies cleanly beside #3422 (measured on #3422's text). No open PR touches ADR-083, the BYOK guide, `tests/helpers/llm_quota.py` or the other test files; `Docs/Published/ADR/README.md` (#3021, #3226, #3269, #3294, #3394) is not changed by this PR's mirror refresh. | note |

## Decisions

1. **ADR-004 Change summary.** AI-authored. **Answered (owner, 2026-11-05):** the waiver covers all three TASK-13956 PRs; the PR body records it as a full `## Change summary` section.
2. **AC #2's premise does not hold (owner question; answered 2026-11-05: "Amend + follow-up", the recommendation below).** AC #2 asks the ingest job's start check to admit an over-limit user "whose analysis provider resolves to their own key". Job analysis never resolves a user key (V1, V2): it always runs on the server's key, so for the quotas it is the operator's spend and the refusal is correct. AC #1 has the same gap for the media endpoints, the study assistants, watchlists, evaluation runs and benchmarks.
   - **Recommendation:** amend AC #1 and AC #2 to what holds (Task 0 Step 3 has the text), pin the refusal, and file a follow-up, TASK-<FOLLOW_UP>, for the product question: should those features use the user's stored key at all? Doing so moves their cost from the operator to the user's key, which the BYOK guide does not promise today. Once a feature resolves the key through `ProviderCredentialRuntime` and passes the handle, its boundary names its provider in one line, as PRs A and B did.
   - *If the owner wants AC #2 as written now:* the job must resolve the owner's key for `form_data.api_name` (a `ProviderCredentialRuntime` per job) and hand it to `_summarize_via_adapter`, the single choke point every `analyze` call goes through, by a context-scoped handle or a parameter through about ten libraries; then `_admit_or_refuse_analysis` asks `users_own_key_serves` and returns without `mark_llm_quota_prechecked()` for an own key. About a day with tests, and a new plan.
   - *Cost if wrong:* until the follow-up, a BYOK user over a limit gets the 402 on media analysis and the other features listed, as today; their usage of those features is the operator's and counts.
3. **The pins are cheap and catch the likely regression (ruling).** A later change that names a provider at one of these boundaries without giving its calls the user's key would pass every existing test (BYOK is off in tests, so the real lookup answers False). `own_keys_for_every_provider` answers True for every provider, so such a change fails its family's pin, provided the pinned request names a provider for the change to pass on. Media (`api_name: openai`), watchlists (`llm_provider: openai`) and both benchmark routes (`api_name` defaults to `openai`) do; the flashcards and quizzes pins did not (`StudyAssistantRespondRequest.provider` defaults to None, and `provider=None` refuses without asking), so they now send `"provider": "openai"` (review C I3). Measured on `/media/add` and on the flashcards route. Two pins catch only some regressions, and say so: the ingest-job pin misses a worker that imports `users_own_key_serves` by name at module level (the helper patches the module attribute; a `ponytail:` line in the helper names it), and the evaluation-run pin cannot name a provider (a run's provider lives in the stored eval spec, which the stub service never loads).
4. **One ADR amendment for the three PRs.** ADR-083's Consequences say the other boundaries still refuse before the key is known, and its Follow-up names this task. The amendment (dated the PR's day) states what A, B and C changed; the BYOK guide's paragraph is rewritten. Both are mirrored.

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
- No source file changes, so Bandit has no source to scan: record "no source touched" (DoD #4's skip).
- `scope_predicate_baseline.txt` lists none of this PR's files (checked).
- `Docs/ADR/083-...md` and `Docs/User_Guides/Server/BYOK_User_Guide.md` are mirrored; `Usage_Quotas.md` is not.
- `git grep -n "TASK-<F>\|TASK-<FOLLOW_UP>" -- ':!Docs/superpowers/plans'` must print nothing before the last commit.
- No pin may send a request off the machine when the boundary it pins regresses (review C I2). The media handlers pin now stubs `Summarization_General_Lib._summarize_via_adapter`; the others already stop before any LLM call even when admitted: the ebook pin stubs `process_epub`, the web-summary pin its fetch and summary calls, the study-assistant pins `generate_study_assistant_reply`, the watchlist pin `summarize_items_for_output`, the ingest-job test's `refused` fixture `_summarize_via_adapter`, the evaluation-run pin the service; the benchmark pins load an empty dataset or name a benchmark that does not exist.

## Review Focus

1. **A boundary here starts naming its provider without giving its calls the user's key** (the regression V7 measures). Expected: its pin fails (`asked != []`, or not the 402), and nothing leaves the machine. Pinned by the nine pins in Task 1 (the flashcards and quizzes ones name a provider in their bodies so the regression has a name to pass; the media pin stubs the analysis call).
2. **A BYOK user reads the docs and expects their key to be used for media analysis.** Expected: `Usage_Quotas.md` and the BYOK guide say these features use the server's key and still refuse. Task 2.
3. **The ingest job's admission marker.** Expected: unchanged; an admitted job is not checked again part-way. Pinned by the existing `test_an_admitted_job_is_not_checked_again_part_way` and `test_the_admission_ends_with_the_job` (in the wider set), plus the new `test_a_stored_key_for_the_analysis_provider_does_not_admit_the_job`.
4. **The ADR amendment describes A and B as merged.** Expected: it matches the code on `dev` when this PR merges (re-read A's and B's final diffs in Task 2 Step 1).
5. **The follow-up's id** is unique across `dev` and open PRs (`check_backlog_task_ids.py`). Task 0.

## Files

- Modify tests: `tldw_Server_API/tests/helpers/llm_quota.py` (new `own_keys_for_every_provider`); `tldw_Server_API/tests/Media/test_media_llm_quota.py`; `tldw_Server_API/tests/Media/test_web_summary_service_prompt.py`; `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py`; `tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py`; `tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py`; `tldw_Server_API/tests/Watchlists/test_watchlists_api.py`; `tldw_Server_API/tests/Evaluations/test_evaluations_crud_create_run_api.py`; `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py`; `tldw_Server_API/tests/Evaluations/test_benchmark_api_auth.py`
- Docs: `Docs/Operations/Usage_Quotas.md`, `Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md`, `Docs/User_Guides/Server/BYOK_User_Guide.md`, and their `Docs/Published` copies
- Backlog: TASK-13956 (ACs, notes, close), the new follow-up task file
- Create (Task 0): `Docs/superpowers/plans/2026-11-05-own-key-server-key-boundaries-close-out.md`

## Test commands

The focused set (`$FOCUSED_C`):

```text
tldw_Server_API/tests/Media/test_media_llm_quota.py::test_ebook_analysis_is_refused_before_processing
tldw_Server_API/tests/Media/test_media_llm_quota.py::test_other_media_handlers_refuse_analysis_for_an_over_quota_user
tldw_Server_API/tests/Media/test_web_summary_service_prompt.py::test_summaries_for_an_over_quota_user_are_refused_before_any_page_is_fetched
tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py::test_a_stored_key_for_the_analysis_provider_does_not_admit_the_job
tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py::test_a_refused_document_batch_keeps_every_document
tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py::test_flashcard_assistant_respond_refuses_an_over_quota_user
tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py::test_quiz_attempt_question_assistant_respond_refuses_an_over_quota_user
tldw_Server_API/tests/Watchlists/test_watchlists_api.py::test_output_summaries_refuse_an_over_quota_user_before_any_summary
tldw_Server_API/tests/Evaluations/test_evaluations_crud_create_run_api.py::test_create_run_refuses_an_over_quota_user_before_the_run_starts
tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py::test_benchmark_run_refuses_an_over_quota_user_before_loading_the_dataset
tldw_Server_API/tests/Evaluations/test_benchmark_api_auth.py::test_run_route_refuses_an_over_quota_user_before_any_work
```

Measured: with the new tests on `dev` + A + B (no source change) `21 passed`, and no test in the set opens a connection to a non-loopback address (the socket probe of PR A's Test commands). These are pins: they pass on base by design, and the mutations in Task 1 Step 3 are their RED.

**The wider set** (every test file that names the helper module):

```bash
git grep --untracked -l -E 'helpers\.llm_quota|helpers/llm_quota' -- 'tldw_Server_API/tests/*.py' | grep '/test_[^/]*\.py$'
```

The file list depends on the tree: 29 files on `484e393333`, 30 once PR A is in (its new test in `Chat_NEW/integration/test_roleplay_resume_completion.py` imports the helpers), the same 30 with PR B and with this PR; none PostgreSQL. Run with `-n 8 --dist loadfile`. Measured on the 30-file list, on `dev` + A + B (the Task 0 baseline once A and B have merged) and with this PR (review C M1: the earlier base number was this list's): base `896 passed, 1 skipped` in 3:04; prototype `897 passed, 1 skipped` in 3:01 (the one new test; the pins are edits of existing tests). Expect Task 0's baseline plus one.

---

## Task 0: Owner answer, follow-up task, AC amendment, branch, draft PR

- [ ] **Step 1: Owner answer (recorded).** The owner answered Decision 2 on 2026-11-05: "Amend + follow-up (Recommended)" (Owner answers, below). Steps 3-4 use it: `<date>` is `2026-11-05`, `<answer>` is `Amend + follow-up`.
- [ ] **Step 2: Branch** from `origin/dev` after PRs A and B merge: `git fetch origin dev && git switch -c test/own-key-server-key-boundaries-13956 origin/dev`.
- [ ] **Step 3: Allocate and file the follow-up.** `mkdir <scratchpad>/id_lock` (if it exists, another session holds it: wait). `zsh <plans dir>/task_next.sh` prints the highest id in use; the follow-up's id `<F>` is that plus one. Then:

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Use the user's stored key for media analysis, the study assistants, watchlist summaries, evaluation runs and benchmarks" --id TASK-15426 -l quotas,backend,byok \
  -d "Follow-up of TASK-13956. These features call their provider with the server's key and never with a stored user key: media analysis (the /media endpoints and media ingest jobs, every call through Summarization_General_Lib._summarize_via_adapter with no provider_credentials), the flashcards and quizzes study assistants (study_assistant.generate_study_assistant_reply, perform_chat_api_call_async with no key), watchlist output summaries (outputs_service._llm_reply), evaluation runs (eval_runner, resolve_provider_api_key_from_config) and benchmarks (a raw api_key from the request or the server's). So their calls are the operator's spend and count against the per-user LLM limits, and their boundaries refuse an over-limit user even when the user stored a key for the provider. Product question for the owner: should these features use the user's stored key (moving their cost to that key)? If yes, resolve it through ProviderCredentialRuntime and pass the handle; then each boundary names its provider to enforce_llm_quota in one line, and the ingest job's start check (_admit_or_refuse_analysis) admits an over-limit user whose analysis provider runs on their own key without marking the job admitted. Cost differs by family (TASK-13956 plan review): the study assistants are the cheap first slice (build a runtime in the route, resolve, pass the handle; perform_chat_api_call_async already binds it); benchmarks and watchlist summaries are small to medium (analyze takes provider_credentials); evaluation runs are medium (a runtime that outlives the request); media analysis and the ingest job are about a day with tests (a handle through about ten ingestion libraries, or a context-scoped handle read in _summarize_via_adapter, plus a runtime per job)." \
  --ac "The owner decides whether each listed feature uses the user's stored key" \
  --ac "Each feature the owner approves resolves the user's key through ProviderCredentialRuntime and passes the handle on every LLM call, with a test" \
  --ac "Its boundary names its provider to enforce_llm_quota, and the media ingest job's start check admits an over-limit user whose analysis provider runs on their own key without marking the job admitted, with a test"
```

- [ ] **Step 4: Amend TASK-13956's ACs** (owner-approved in Step 1; the old texts go into the notes):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13956 \
  --remove-ac 3 --remove-ac 2 --remove-ac 1 \
  --ac "Each LLM quota boundary whose LLM calls run on the user's stored key names every call's provider to enforce_llm_quota before any LLM work, so an over-limit user whose calls all run on their own key is let through unmarked; a request with a call the boundary cannot place on the user's key still counts; a boundary whose calls run on the server's key keeps refusing; a test per endpoint family, character chat first" \
  --ac "A media ingest job's start check keeps refusing an over-limit user who stored a key for the analysis provider, because job analysis runs on the server's key, and keeps the job admission marker; making job analysis use the stored key is TASK-15426; with a test" \
  --ac "The auto-router's usage row on /chat/completions, character chat and chat_route_resolver records the router call's credential_source, with a test" \
  --append-notes "ACs #1 and #2 amended (owner answer <date>: <answer>). Old #1: Each LLM quota boundary that knows its provider before any LLM work passes it to enforce_llm_quota, with a test per endpoint family, character chat first. Old #2: A media ingest job's start check admits analysis for an over-limit user whose analysis provider resolves to their own key, without marking the job admitted, with a test. Why: media analysis (and the ingest job), the study assistants, watchlist summaries, evaluation runs and benchmarks never resolve a stored user key, so their calls are the operator's and count; naming their provider would move the 402 into the middle of the work. Follow-up TASK-15426. AC #3 unchanged." \
  --append-notes "Plan, PR C of 3: Docs/superpowers/plans/2026-11-05-own-key-server-key-boundaries-close-out.md."
```

  `sed -n '/AC:BEGIN/,/AC:END/p' "backlog/tasks/task-13956 - ...md"` shows #1-#3 in that order (measured: `--remove-ac 3 --remove-ac 2 --remove-ac 1` then three `--ac` keeps the order).
- [ ] **Step 5: Plan file** at `Docs/superpowers/plans/2026-11-05-own-key-server-key-boundaries-close-out.md`, with `<F>` filled. `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .`: exit 0. `git add -A backlog/tasks Docs/superpowers/plans && git diff --cached --check`; commit (message 0).
- [ ] **Step 6: Draft PR** (`--base dev`, title `test(usage): the boundaries that never run on a stored key keep refusing; close TASK-13956 (3 of 3)`). Once `gh pr view <n> --json files` lists `backlog/tasks/task-15426 - ...md`, `rmdir <scratchpad>/id_lock`.
- [ ] **Step 7: Baseline** the wider set on the branch (expected after A and B merge: `896 passed, 1 skipped`, measured on `484e393333` + A + B); record the counts and every failing id.

## Task 1: Pin the refusal at the server-key boundaries (amended AC #1, AC #2)

**Files:** the ten test files in Files.

**Interfaces:**
- Produces: `own_keys_for_every_provider(monkeypatch: pytest.MonkeyPatch) -> list[str]` in `tests/helpers/llm_quota.py`.

- [ ] **Step 1: The helper.** In `tests/helpers/llm_quota.py`, above `def assert_llm_quota_402`:

```python
def own_keys_for_every_provider(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The user has their own key for every provider; returns each provider a boundary asks about.

    For a boundary whose LLM calls run on the server's key: it must refuse an over-limit user
    anyway, without asking (TASK-13956).
    """
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps

    # ponytail: patches the module attribute, so code that imported users_own_key_serves by name at module
    # level still gets the real lookup (False in tests): such a regression passes. Patch that module too then.
    asked: list[str] = []

    async def _own_key(_uid: object, provider: str) -> bool:
        """Every provider runs on the user's own key."""
        asked.append(provider)
        return True

    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    return asked
```

- [ ] **Step 2: The pins.** Each pin gains `asked = own_keys_for_every_provider(monkeypatch)` and `assert asked == []`, a docstring that says why, and the helper in its `helpers.llm_quota` import. Every edit is literal; "replace A with B" finds A once inside the named test.
  - `test_media_llm_quota.py`: replace the module import `from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work` with

```python
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib
from tldw_Server_API.tests.helpers.llm_quota import (
    assert_llm_quota_402,
    own_keys_for_every_provider,
    refuse_llm_work,
)
```

    In `test_ebook_analysis_is_refused_before_processing`, after `    refuse_llm_work(monkeypatch)` add the line `    asked = own_keys_for_every_provider(monkeypatch)  # analysis runs on the server's key (TASK-13956)`, and after `    assert processed == []` add `    assert asked == []`. Replace `test_other_media_handlers_refuse_analysis_for_an_over_quota_user`'s body (from its docstring to its end) with (review C I2: the stub keeps a regression, or the RED mutation of Step 3, on the machine):

```python
    """Each remaining handler answers the 402 before touching its input, whatever keys the user has stored."""
    client, _usage_logger = client_with_single_user
    summarized: list[dict] = []
    # A regression would admit the request and analyse on the server's key for real: stub that call.
    monkeypatch.setattr(
        Summarization_General_Lib, "_summarize_via_adapter", lambda **kwargs: summarized.append(kwargs) or ""
    )
    refuse_llm_work(monkeypatch)
    asked = own_keys_for_every_provider(monkeypatch)
    response = client.post(path, data=data, files=files)
    assert_llm_quota_402(response)
    assert asked == []
    assert summarized == []
```

  - `test_web_summary_service_prompt.py`: replace the module import `from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work` with the parenthesized three-name `helpers.llm_quota` import above (without the `Summarization_General_Lib` line). In `test_summaries_for_an_over_quota_user_are_refused_before_any_page_is_fetched`, replace

```python
    """The 402 comes first, so no page is fetched and then discarded (TASK-13864)."""
    refuse_llm_work(monkeypatch)
```

    with

```python
    """The 402 comes first, so no page is fetched and then discarded (TASK-13864); summaries run on the
    server's key, so a key the user stored does not lift it (TASK-13956)."""
    refuse_llm_work(monkeypatch)
    asked = own_keys_for_every_provider(monkeypatch)
```

    and after `    assert context.calls == []` add `    assert asked == []`.
  - `test_flashcards_endpoint_integration.py`, in `test_flashcard_assistant_respond_refuses_an_over_quota_user`: replace

```python
    """An over-quota user gets the 402 and the study assistant never calls the LLM."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work
```

    with

```python
    """An over-quota user gets the 402 and the study assistant never calls the LLM, whatever keys they stored."""
    from tldw_Server_API.tests.helpers.llm_quota import (
        assert_llm_quota_402,
        own_keys_for_every_provider,
        refuse_llm_work,
    )
```

    after `    refuse_llm_work(monkeypatch)` add `    asked = own_keys_for_every_provider(monkeypatch)  # the assistant runs on the server's key (TASK-13956)`; replace `        json={"action": "explain"},` with `        json={"action": "explain", "provider": "openai"},` (review C I3: the request names a provider, so a boundary that named `payload.provider` would ask about it and fail the pin; with no provider it refuses without asking and the pin passes, measured in Step 3); and after `    assert calls == []` add `    assert asked == []`.
  - `test_quizzes_endpoint_integration.py`, in `test_quiz_attempt_question_assistant_respond_refuses_an_over_quota_user`: the same four edits, with the same texts.
  - `test_watchlists_api.py`, in `test_output_summaries_refuse_an_over_quota_user_before_any_summary`: replace

```python
    """An over-quota user asking for summaries gets the 402 before any summary call (TASK-13864)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402
```

    with

```python
    """An over-quota user asking for summaries gets the 402 before any summary call (TASK-13864), whatever keys
    they stored: summaries run on the server's key (TASK-13956)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, own_keys_for_every_provider
```

    after `    run_id, summarized = _over_quota_run(c, monkeypatch)` add `    asked = own_keys_for_every_provider(monkeypatch)`; and after `    assert summarized == []` (before the follow-up `GET`) add `    assert asked == []`.
  - `test_evaluations_crud_create_run_api.py`, in `test_create_run_refuses_an_over_quota_user_before_the_run_starts`: replace

```python
    """A run's samples are LLM calls made in a task the request starts, so the run is checked at creation (§4a)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work
```

    with

```python
    """A run's samples are LLM calls made in a task the request starts, so the run is checked at creation (§4a).

    They run on the server's key, so a key the user stored does not lift the refusal (TASK-13956).
    """
    from tldw_Server_API.tests.helpers.llm_quota import (
        assert_llm_quota_402,
        own_keys_for_every_provider,
        refuse_llm_work,
    )
```

    after `    refuse_llm_work(monkeypatch)` add `    asked = own_keys_for_every_provider(monkeypatch)`; and after `    assert started == []` add `    assert asked == []`.
  - `test_evaluations_benchmarks_api.py`, in `test_benchmark_run_refuses_an_over_quota_user_before_loading_the_dataset`: replace

```python
    """Benchmark runs check the LLM quota at the request boundary, as evaluation runs do (TASK-13638)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work
```

    with

```python
    """Benchmark runs check the LLM quota at the request boundary, as evaluation runs do (TASK-13638).

    Their items run on the server's key (or a raw ``api_key``), so a stored key does not lift it (TASK-13956).
    """
    from tldw_Server_API.tests.helpers.llm_quota import (
        assert_llm_quota_402,
        own_keys_for_every_provider,
        refuse_llm_work,
    )
```

    after `    refuse_llm_work(monkeypatch)` add `    asked = own_keys_for_every_provider(monkeypatch)`; and after `    assert loaded == []` add `    assert asked == []`.
  - `test_benchmark_api_auth.py`, in `test_run_route_refuses_an_over_quota_user_before_any_work`: replace

```python
    """The standalone run route checks the LLM quota before it looks up the benchmark (TASK-13638)."""
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work
```

    with

```python
    """The standalone run route checks the LLM quota before it looks up the benchmark (TASK-13638), whatever keys
    the user stored: its items run on the server's key or a raw ``api_key`` (TASK-13956)."""
    from tldw_Server_API.tests.helpers.llm_quota import (
        assert_llm_quota_402,
        own_keys_for_every_provider,
        refuse_llm_work,
    )
```

    after `    refuse_llm_work(monkeypatch)` add `    asked = own_keys_for_every_provider(monkeypatch)`; and after `    assert response.json()["detail"]["error"] == "limit_exceeded"` add `    assert asked == []`.

  And in `test_media_ingest_job_llm_quota_refusal.py`, replace the import `from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_calls` with `from tldw_Server_API.tests.helpers.llm_quota import own_keys_for_every_provider, refuse_llm_calls`, and after `test_a_refused_document_batch_keeps_every_document` add:

```python
@pytest.mark.asyncio
async def test_a_stored_key_for_the_analysis_provider_does_not_admit_the_job(
    job_dbs: Path, refused: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job analysis runs on the server's key, never the owner's, so an own key leaves the start check as it was.

    Refused, the job is stored without analysis and makes no LLM call (TASK-13956).
    """
    asked = own_keys_for_every_provider(monkeypatch)
    manager = JobManager()
    row = _create_job(manager, job_dbs, _write_document(job_dbs, "Cedar"), "document")

    await _run_jobs(manager, 1)

    _assert_kept_with_refusal(manager.get_job(int(row["id"])), job_dbs, "Project Cedar opens")
    assert refused == []
    assert asked == []
```

- [ ] **Step 3: Run them, then the mutations that are their RED.** `$FOCUSED_C`: `21 passed`. The mutations run only once Step 2's `_summarize_via_adapter` stub is in: without it the `/media/add` mutation sends `POST https://api.openai.com/v1/chat/completions` (review C I2). Revert each (`git diff --stat -- tldw_Server_API/app` must print nothing afterwards). Neither makes a request leave the machine (measured with a pytest plugin that blocks and logs `socket.connect` to any non-loopback address).
  1. In `media/add.py` change `await enforce_llm_quota(current_user.id)` to `await enforce_llm_quota(current_user.id, provider=form_data.api_name)` and run `"tldw_Server_API/tests/Media/test_media_llm_quota.py::test_other_media_handlers_refuse_analysis_for_an_over_quota_user"`: expected `1 failed, 5 passed` (`[add]`: the request is admitted and answered, measured).
  2. In `flashcards.py`, drop `dependencies=[Depends(require_llm_quota)],` from the `/{card_uuid}/assistant/respond` route, add `current_user: User = Depends(get_request_user),` to `respond_flashcard_assistant`'s parameters and, as its first lines, `from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota` and `await enforce_llm_quota(current_user.id, provider=payload.provider)` (the "route dependency to an inline call" the task once proposed). Run `"tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py::test_flashcard_assistant_respond_refuses_an_over_quota_user"`: expected `1 failed` (answered 200, measured). With the pin's earlier body (`{"action": "explain"}`, no provider) the same mutation passes it (`1 passed`, measured): why the body names a provider.
- [ ] **Step 4: Triage.** `PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/ci/test_quality_triage.py --enforce`: "OK — no offenses beyond the baseline." (measured with A+B+C). The pins pair each collector assertion with the 402's; the only `monkeypatch` call a pin gains is the media handlers pin's analysis stub.
- [ ] **Step 5: Commit** (message 1).

## Task 2: Docs, ADR-083's amendment, verify, close TASK-13956, review, PR

- [ ] **Step 1: Re-read** PR A's and PR B's merged diffs (`git log --oneline origin/dev -- tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`); the amendment below must match them.
- [ ] **Step 2: `Usage_Quotas.md`.** In PR B's own-key bullet, replace its last two sentences (`Every other boundary endpoint refuses ... So does a media ingest job's check when it starts.`) with:

```markdown
The other boundaries refuse an over-limit user whatever keys they have stored, because their LLM calls run on the server's key and never on a stored user key: media analysis (`/media/add`, `/media/ingest-web-content`, `/media/process-web-scraping` and the other `/media/process-*` endpoints) and a media ingest job's check when it starts, the flashcards and quizzes study assistants, watchlist output summaries, evaluation runs and benchmark runs. Those calls count, so the refusal is the limit doing its job.
```

- [ ] **Step 3: BYOK guide.** In `Docs/User_Guides/Server/BYOK_User_Guide.md`, replace (literal text)

```text
If you are already over a limit, `/chat/completions` and the Messages API still answer a request whose provider uses your own key. Other endpoints, such as character chat and RAG, still refuse it for now.
```

  with:

```markdown
If you are already over a limit, an endpoint still answers a request whose LLM calls all use your own key: `/chat/completions`, the Messages API, character chat, document generation from a chat, the judge-model evaluations and RAG searches that make no LLM call besides the answer (and, with a chat history, its reformulation). Media analysis, the flashcards and quizzes study assistants, watchlist summaries, evaluation runs and benchmark runs use the server's key, not yours, so over a limit they still refuse.
```

- [ ] **Step 4: ADR-083.** Append to `Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md`, after the 2026-10-31 amendment, one blank line and (date it the day this PR opens; fill `<F>`; the prototype carries the block as written, `<F>` unfilled):

```markdown
## Amendment (2026-11-05): the other boundaries, and the auto-router's row

TASK-13956 closes the two gaps this ADR's Follow-up names. It replaces the sentence in Consequences that the other boundary endpoints and a media ingest job's start still refuse an over-limit user before the key is known.

- **A boundary names every LLM call it will make.** `enforce_llm_quota` takes one provider or several, and lets an over-limit user through, unmarked, only when every one runs on the user's own key; a request that names none, or one that is not, is refused as before. Character chat names the reply's provider once the auto-router has picked it (the check moved below the router, as on `/chat/completions`) and a roleplay resume its saved provider; a reply admitted that way and then refused at the call answers the 402, as on `/chat/completions`. The judge-model evaluations name each item's judge provider. Document generation from a chat names its provider. RAG search, streaming, batch, batch resume and advanced search name the answer's provider and, with a chat history to reformulate, the reformulation's; they read the request as the pipeline runs it, with a `rag_profile`'s switches and the server's search defaults applied.
- **A call the boundary cannot place on the user's key keeps the refusal.** A saved character chat reply in a chat with automatic memory extraction on (extraction runs on the server's key), a RAG request that turns on another LLM stage (itself, through a profile or through a server default), and RAG ablation are refused as before. A stage the boundary does not know about is still checked at its own call, as a failover is.
- **Some boundaries never run on a stored key.** Media analysis (the media endpoints and a media ingest job's start), the flashcards and quizzes study assistants, watchlist summaries, evaluation runs and benchmark runs call their provider with the server's key. Their calls count, so these boundaries refuse an over-limit user whatever keys the user has stored. Making those features use a stored key is follow-up TASK-15426.
- **The auto-router's row.** The router call's usage row records the router call's own `credential_source`, on `/chat/completions`, character chat and the shared route resolver (prompt improvement with `model: "auto"`). The resolver passes a raw key, so its router call also marks its call context, and its check at the call skips a call on the user's own key.
- **A refused router call that leaves no route.** With `routing.failure_mode: "error"`, a router call refused at the call left no route, and `/chat/completions` answered `503 auto_routing_failed`. It and character chat now answer the 402. Any other lost route keeps its answer: a router call that failed for another reason the 503, no candidates the 400.
- Evaluation token reservations (`limits.evaluation_tokens_per_day`) keep counting own-key calls (owner decision, 2026-10-29).
```

- [ ] **Step 5: Mirror.** `PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" bash Helper_Scripts/refresh_docs_published.sh`; `git status --short Docs/Published` shows exactly `Docs/Published/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md` and `Docs/Published/User_Guides/Server/BYOK_User_Guide.md` (measured). `git add` the five docs, then `tests/Docs`: `212 passed` (measured). Commit (message 2).
- [ ] **Step 6: Verify:** `$FOCUSED_C` `21 passed`; the wider set against Task 0's baseline (the baseline plus one; measured `897 passed, 1 skipped`); `tests/Docs tests/lint tests/CI tools/backlog-py/tests` (`1661 passed, 6 skipped`, measured on `dev` + A + B + C); shard coverage (`new_uncovered=0`); triage ("OK — no offenses beyond the baseline."); route-auth ("208 unauthenticated routes; baseline exact."); OpenAPI (both "OK" lines); the commands are PR A's Task 3 Step 2; `check_backlog_task_ids.py .` exit 0; `git diff origin/dev --check` nothing; `git grep -n "TASK-<F>\|TASK-<FOLLOW_UP>" -- ':!Docs/superpowers/plans'` nothing. Bandit: no source touched.
- [ ] **Step 7: Close TASK-13956** (fill every `<...>`):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13956 \
  --append-notes "PR C (#<n>): pins (own_keys_for_every_provider) that media analysis and the ingest job's start, the study assistants, watchlist summaries, evaluation runs and benchmarks refuse an over-limit user whatever keys they stored, without asking (their calls run on the server's key); mutations: /media/add naming its provider fails its pin, the flashcards route naming payload.provider fails its pin (whose request now names openai), and neither sends a request off the machine (the media pin stubs _summarize_via_adapter). ADR-083 amended (2026-11-05), Usage_Quotas.md and the BYOK guide updated, mirror refreshed. Focused <n> passed; wider <base> -> <after>; tests/Docs + lint + CI + backlog-py <n> passed; shard, triage, route-auth, OpenAPI as before. Bandit: no source touched in PR C (PRs A and B: no issues)." \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Over-limit users whose LLM calls all run on their own key are let through, unmarked, at every boundary that can name its calls' providers: character chat (the reply's provider after the auto-router; a roleplay resume's saved provider; not a saved reply in a chat with memory extraction on), the judge-model evaluations (each batch item's provider), chat document generation, and RAG search, stream, batch, resume and advanced search (the answer's provider and the reformulation's, read from the resolved request; any other LLM stage counts, including one a rag_profile or a server default turns on). enforce_llm_quota takes several providers and refuses when any is missing or not the user's own. An admitted character chat reply refused at the call answers the 402, not 502. The auto-router's usage row records the router call's credential_source on /chat/completions, character chat and chat_route_resolver, whose raw-key router call also marks its context. A refused router call that leaves no route answers the 402, not 503; any other routing error keeps its status. Media analysis and the ingest job, the study assistants, watchlist summaries, evaluation runs and benchmarks run on the server's key, so they keep refusing (ACs #1 and #2 amended, owner <date>); follow-up TASK-15426. Evaluation token reservations unchanged. PRs #<A>, #<B>, #<C>." \
  -s Done
git grep -n -E "<(n|A|B|C|F|base|after|date|answer)>" -- "backlog/tasks/task-13956 - *" "backlog/tasks/task-15426 - *"
git add -A backlog/tasks && git diff --cached --check
```

  The `git grep` (only the two task files this PR writes, with `<F>` filled; other task files hold such text already, `task-13404` a `<n>` and `task-13921` a `<base>`, review C M4) must print nothing. Commit (message 3).
- [ ] **Step 8: Review, PR body, Change summary, checks, merge:** as PR A's Task 3 Steps 5-8 and 10. The body names the AC amendment and the owner's answer, the follow-up, and the three PRs.

## Commit messages

Each ends with a blank line and `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

- 0: `chore(backlog): file TASK-15426, amend TASK-13956's ACs, and plan PR C (TASK-13956, 3 of 3)`
- 1: `test(usage): the boundaries whose calls run on the server's key refuse whatever keys the user stored (TASK-13956)`
- 2: `docs(usage): ADR-083 amendment for TASK-13956; which endpoints admit an own-key user (TASK-13956)`
- 3: `chore(backlog): close TASK-13956`

## Not measured

- The follow-up's id (taken at execution, under the lock).
- The owner's answer to Decision 2.
- The ADR amendment against the final merged diffs of A and B (Task 2 Step 1 re-reads them).

## Owner answers (2026-11-05, recorded by the controller)

- Plans A, B and C: "Approve (Recommended)", with both reviews' Important fixes and minors folded in (review_plan_13956_a.md, review_plan_13956_bc.md).
- AC #2 (and the matching part of AC #1): "Amend + follow-up (Recommended)": PR C amends the ACs with backlog-py to what holds, pins the server-key boundaries, and files the follow-up asking whether those features should use the stored key (the study assistants as the cheap first slice; media analysis about a day).
- Decision 1, ADR-004 Change summary: the waiver covers all three TASK-13956 PRs ("Yes, covers all three").

## Folded after the plan reviews (2026-11-05)

From `review_plan_13956_bc.md` (C: 2 Important, 5 Minor, and M9 shared with B), `review_plan_13956_a.md` (its fixes change the ADR text), and the owner's answers. Re-measured on `origin/dev` `484e393333` with #3422's `Usage_Quotas.md`, `proto_13956_a.patch` and `proto_13956_b.patch`; `proto_13956_c.patch` was rebuilt from this plan's text on that tree and matches it byte for byte (`cmp`). The first prototype differed from its plan in three places (the follow-up id, a docstring wrap, an assert's position); Task 1 Step 2 now gives every edit literally, and the patch carries the ADR block as written (`TASK-15426` unfilled).

| Finding | What changed |
|---|---|
| I2: the RED mutation reached `api.openai.com` | `test_other_media_handlers_refuse_analysis_for_an_over_quota_user` stubs `Summarization_General_Lib._summarize_via_adapter` and asserts `summarized == []`; Task 1 Step 3 says the mutation runs only with the stub in; a plan-specific constraint lists how every other pin already stops before an LLM call. Mutation 1 is still RED (`1 failed, 5 passed`, `[add]`) and a socket probe saw no outbound connection. |
| I3: the flashcards and quizzes pins named no provider | Both requests send `"provider": "openai"`. New mutation 2 (the flashcards route names `payload.provider` inline instead of `require_llm_quota`): the pin fails (`1 failed`, answered 200); with the old body it passed (`1 passed`), both measured. Decision 3 corrected (which pins catch which regression). |
| M1: wider base mislabelled | 29 files on `dev`, 30 with A (and with B, and with C); re-measured on the 30-file list on `dev` + A + B; Task 0 Step 7 and Task 2 Step 6 expect the baseline plus one. |
| M4: the close-out placeholder grep could not print nothing | Task 2 Step 7 greps only the two task files this PR writes. |
| M5: the job and evaluation-run pins catch only some regressions | A `ponytail:` comment in `own_keys_for_every_provider` names the module-level-import ceiling; Decision 3 names both. |
| M8: header wording on `TASK-<FOLLOW_UP>` | The header says where `<F>` is allocated (Task 0 Step 3), written (Task 2 Step 4) and checked (Task 2 Step 6). |
| M9 (B, C docs) | With B's I1 the BYOK guide sentence holds; the ADR amendment says RAG reads the resolved request and that a profile or a server default turning a stage on counts. |
| M10: overlap list incomplete | V9 rebuilt from all 38 open PRs (#3269's hunks in two of this PR's test files, no conflict; #3422 on `Usage_Quotas.md`, applies cleanly beside this PR's edit). |
| PR A's fold, in the docs | The ADR amendment says an admitted character chat reply refused at the call answers the 402, and that any other lost route keeps its 503 or 400; the close-out's final summary says the same. |
| Owner answers | Decision 2 answered "Amend + follow-up" (Task 0 Step 1 records it and fills `<date>`, `<answer>`); the follow-up's description names the study assistants as the cheap first slice and media analysis at about a day. Decision 1 recorded. |

Re-measured:
- Focused set: `21 passed` on `dev` + A + B (no source change); no outbound connection (socket probe).
- Mutations (Task 1 Step 3): 1 `1 failed, 5 passed` (`[add]`); 2 `1 failed`, and `1 passed` with the pin's old body; no outbound connection in either.
- Wider set (30 files on `dev` + A + B): base `896 passed, 1 skipped` (3:04); prototype `897 passed, 1 skipped` (3:01).
- `tests/Docs tests/lint tests/CI tools/backlog-py/tests` on `dev` + A + B + C: `1661 passed, 6 skipped` (5:23); `tests/Docs` `212 passed`. Shard coverage `new_uncovered=0`; triage `--enforce` OK; route-auth "208 unauthenticated routes; baseline exact."; OpenAPI both OK; `check_backlog_task_ids.py .` exit 0; mirror refresh changes exactly the two `Docs/Published` files; ruff: no new finding in the ten test files; no source touched (no Bandit run); `git diff --check` clean.
- Open-PR overlap: V9 (re-checked against current heads).

## Candidates from PR A's final review (2026-11-05, recorded by the controller)

PR A (#3423) final review (`final_review_13956_a.md`, Ready, 0C/0I/7M) suggested three one-sentence docs items for this PR's docs/ADR step. Not yet folded into the steps; decide when executing (each is docs only):
- `Docs/User_Guides/Server/BYOK_User_Guide.md`: an over-limit user on their own key in a chat with automatic memory extraction on is refused (extraction runs on the server's key); to chat past the limit, turn extraction off, or stream / `save_to_db: false`.
- `Usage_Quotas.md` (character chat errors-first): say once that a request error found before the check is answered first, instead of enumerating (the list misses the BYOK credential error, `500 provider_model_resolution_failed`, and `503` when the router's handle cannot resolve).
- `Usage_Quotas.md:60` (extraction "runs as part of that reply, which the boundary check already admitted"): add that an unmarked own-key admission never reaches extraction (such a turn is refused). ADR-083 `:35` ("as the auto-router's call does today") is already in this plan's amendment.

## Candidates from PR B's execution (2026-11-06, recorded by the controller)

- Test hygiene, not this work stream's code: the pre-existing `RAG_NEW/integration/test_rag_stream_parity.py::test_rag_streaming_generation_provider_override` makes a real outbound call to `api.groq.com:443` (stack: `groq_adapter.chat` via `usage_accounting.chat` on a `bounded_daemon` thread; the test stubs `generate_streaming_response` only). Seen with the netblock probe on PR B's Task 2, identical without PR B's code (`netwhere_13956_b.sh`). Candidate: file it as a follow-up task under the id lock with PR C's follow-up, or fold a stub into PR C if it stays one line.
- PR B final review (fable) recommendation: a drift test for `rag_unified._RAG_OTHER_LLM_STAGES` against the RAG request schema's `enable_*` fields, so a new switch has to be declared as an LLM stage or not (the list was trimmed to the switches that call an LLM in 3a402df4e6; the rule-based ones are named in the comment under it). Also two deferred minors there: canonicalize the reformulation's provider before the dedup (`rag_unified.py` `_rag_llm_providers`), and assert `asked` in `test_rag_search_still_refuses_an_own_key_answer_with_another_llm_stage[history-on-a-server-key]`.

## Rulings on the candidates (2026-11-06, controller, before Task 0)

- Ruling: fold two of PR A's three docs items into Task 2 Steps 2-3; drop the third — docs only, inside the approved docs step, each corrects text a reader acts on — cost if wrong: two sentences to revert.
  - Dropped: the "errors first" enumeration. It lives in PR A's plan (Decision 3) and PR body, not in any shipped doc (`git grep` of Usage_Quotas.md, ADR-083 and the BYOK guide on `2f402bd7e3`: no such list), so there is nothing to correct.
  - Step 2 addition, `Usage_Quotas.md` (the "Claims verification is refused" bullet): replace `Automatic character memory extraction after a character chat reply runs as part of that reply, which the boundary check already admitted.` with `Automatic character memory extraction after a character chat reply runs as part of that reply, which the boundary check already admitted; a reply admitted on the user's own key never reaches it, because a reply that would start extraction is refused (below).`
  - Step 3 addition, BYOK guide: append to the rewritten paragraph, on the same line after one space:

```markdown
In a character chat with automatic memory extraction on, a saved reply is still refused, because extraction runs on the server's key: turn extraction off, stream the reply, or send `save_to_db: false`.
```

    (code: `will_persist = bool(save_to_db) and not stream_requested`, `character_chat_sessions.py:7083`; `_enforce_character_reply_quota` refuses only when `will_persist` and extraction is enabled.)
- Ruling: do NOT fold PR B's items (Groq outbound test, `_RAG_OTHER_LLM_STAGES` drift test, reformulation canonicalization, `asked` in `[history-on-a-server-key]`); file them as a second follow-up, TASK-15427, in Task 0 Step 3 under the same lock — they touch RAG source and tests outside this approved plan (which has no source change), and the owner's pattern is a follow-up PR for deferred minors; a stage missing from the list is still refused at its own call, so nothing leaks meanwhile — cost if wrong: one more small PR later.
  Follow-up text: title "PR B (TASK-13956) deferred minors: RAG LLM-stage drift test, reformulation provider canonicalization, a stream test's outbound Groq call"; labels quotas,backend,tests; ACs: (1) test_rag_stream_parity.py::test_rag_streaming_generation_provider_override makes no outbound connection (stub the Groq adapter or the generator's LLM call), with the socket probe showing 0; (2) a test fails when a RAG request schema switch is neither in rag_unified._RAG_OTHER_LLM_STAGES nor in a declared no-LLM list; (3) _rag_llm_providers canonicalizes the reformulation provider before the dedup, and [history-on-a-server-key] asserts the providers asked.
