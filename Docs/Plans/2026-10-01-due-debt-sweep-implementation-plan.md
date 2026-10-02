# Due-Debt Sweep Implementation Plan (Credit Batch 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clear the repo's past-due maintenance debt — expired deprecation compat paths, dead code, zero-coverage data-deletion endpoints, and untested crypto — as small, independently mergeable PRs.

**Architecture:** Independent backend-only cleanup slices. Each stage is one reviewable PR with its own test cycle. No stage depends on another; stages may run in parallel across worktrees.

**Tech Stack:** Python (FastAPI backend), pytest, Bandit, Backlog.md CLI.

**Spec:** This plan; findings verified 2026-10-01 on branch `codex/post2970-uat-20260920`. Related existing tasks: TASK-12113, TASK-13100. Stages without an existing task need one Backlog.md task each before edits (repo rule §0; proposed titles below).

## Global Constraints

- `source .venv/bin/activate` before any `python` / `pip` / `pytest` command.
- Backend scope only — do not modify `apps/**` (console-UX / performance workstreams own it).
- One Backlog.md task per stage before file edits — stage tasks already exist: TASK-13399 (Stage 1), TASK-13400 (Stage 2), TASK-13401 (Stage 3), TASK-13402 (Stage 4); Stage 5 uses existing TASK-12113/TASK-13100. Use `backlog-py` (tools/backlog-py) for Backlog CLI operations; the bun `backlog` CLI's `task create` still crashes.
- Run `python -m bandit -r <touched_paths> -f json -o /tmp/bandit_<stage>.json` before finishing each stage; fix new findings in changed code.
- Line numbers below verified 2026-10-01; re-verify with grep before editing (branch drifts daily).
- Never delete other agents' plan files at repo root.

---

## Stage 1: Remove expired deprecation compat paths

**Goal:** Delete the three past-sunset compat paths and their fallback call sites; remove deprecated `/me` endpoints.
**Success Criteria:** Registry contains no entry with sunset < 2026-10-01; all call sites use successor paths only; full backend test suite green.
**Tests:** Extended deprecation-registry tests asserting expired keys are gone/raising; endpoint tests for removed `/me` routes assert 404.
**Status:** Not Started

Backlog task: TASK-13399.

- [ ] **Step 1: Inventory.** Read `tldw_Server_API/app/core/deprecations/runtime_registry.py`. Three entries are past sunset as of 2026-10-01: `web_scraping_legacy_fallback` (sunset 2026-06-30, line ~12), `llm_chat_legacy_session` (2026-07-15, ~17), `auth_db_execute_compat` (2026-08-01, ~22).
- [ ] **Step 2: Find all call sites.** `grep -rn "web_scraping_legacy_fallback\|llm_chat_legacy_session\|auth_db_execute_compat" tldw_Server_API/app tldw_Server_API/tests`. Known: `LLM_Calls/chat_calls.py:79,128`; `services/auth_service.py:65,80`; `services/web_scraping_service.py:364`.
- [ ] **Step 3: Write failing tests.** Find the existing deprecation tests (`grep -rln "runtime_registry\|deprecat" tldw_Server_API/tests | head`) and add assertions that each expired key raises / is no longer registered. Run them; expect FAIL.
- [ ] **Step 4: Remove fallback call sites.** In each file above, delete the legacy branch and keep only the successor path. Follow the successor named in each registry entry.
- [ ] **Step 5: Delete the three registry entries.** Run the tests from Step 3; expect PASS.
- [ ] **Step 6: Deprecated `/me` endpoints.** `tldw_Server_API/app/api/v1/endpoints/users.py:544,589` — check test references first (`grep -rn "/me" tldw_Server_API/tests | grep -i user`), update tests to expect 404, then delete routes.
- [ ] **Step 7: Straggler sweep.** Re-run the Step 2 grep; expect zero hits in `app/`.
- [ ] **Step 8: Bandit + full targeted tests + commit.** `python -m pytest tldw_Server_API/tests -k "deprecat or users" -v`; commit `fix: remove compat paths past sunset (TASK-<id>)`.
- [ ] **Step 9 (decision gate): `USER_DB_BASE` legacy alias.** `db_path_utils.py:475` deprecated; `allow_legacy_alias=True` at `:164` and `:870`. Grep for runtime users. If only tests use it, remove in this PR; if deployers may depend on it, file a follow-up task with evidence and skip.

## Stage 2: Dead-code and packaging housekeeping

**Goal:** Drop the dead `gradio` extras group and other verified-dead code.
**Success Criteria:** `pip install -e .` resolves without gradio; no grep hits for removed code; tests green.
**Tests:** Existing `tldw_Server_API/tests/Utils/test_quick_launch_scripts.py:300` (enforces no gradio regression); `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`.
**Status:** Not Started

Backlog task: TASK-13400.

- [ ] **Step 1:** `pyproject.toml` — delete the `gradio = [...]` group (lines ~470-471) and remove `gradio` from the `all` extras list (line ~480).
- [ ] **Step 2:** Validate parse + install: `python -c "import tomllib; ..."` then `pip install -e . --dry-run` (or full install in the venv).
- [ ] **Step 3:** Delete commented-out Gradio block `app/core/Metrics/metrics_logger.py:226` and the stale docstring note `app/core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py:925`.
- [ ] **Step 4:** Delete commented-out Elasticsearch workflow functions + "Dead code FIXME" `app/core/DB_Management/DB_Manager.py:1380-1397`.
- [ ] **Step 5:** Replace the deprecated stub `app/core/DB_Management/media_db/legacy_maintenance.py:125` (returns `True, "Deprecated"`) per its callers' needs — if all callers are gone, delete the module; check `grep -rn legacy_maintenance tldw_Server_API/app`.
- [ ] **Step 6:** Root plan-file audit — **index only, never delete others' plans.** Write `Docs/plans/2026-10-01-root-implementation-plan-index.md` listing the 27 root `IMPLEMENTATION_PLAN_*.md` files with done/in-progress/unknown status (cross-check against `backlog/`).
- [ ] **Step 7:** Bandit, run quick-launch tests, commit `chore: drop gradio extras and dead code (TASK-<id>)`.

## Stage 3: WebSearch_APIs.py hygiene (behavioral — own PR)

**Goal:** Evict inline smoke-test functions from production code; fix the broken Google advanced-args path; drop Bing remnants.
**Success Criteria:** No `test_perform_websearch_*` definitions inside `app/core/Web_Scraping/WebSearch_APIs.py`; Google path has a passing test; Bing code removed or isolated.
**Tests:** New unit tests for the Google arg-formatting fix; moved smoke functions become skipped/marked tests or are deleted if redundant.
**Status:** Not Started

Backlog task: TASK-13401.

- [ ] **Step 1:** Read `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py` lines ~1547-1820: ~8 inline `test_perform_websearch_*` functions live in production code, several self-flagged FIXME.
- [ ] **Step 2:** Move any still-valuable ones to `tldw_Server_API/tests/Web_Scraping/test_websearch_smoke.py` with `@pytest.mark.external_api`; delete redundant ones.
- [ ] **Step 3:** Fix the Google advanced-args FIXME at `:1747` ("Fails. Need to fix arg formatting") — write a failing unit test for the arg builder first (mock HTTP), then fix formatting.
- [ ] **Step 4:** Remove the deprecated Bing provider remnants (mid-function `raise ValueError` and no-op test placeholders).
- [ ] **Step 5:** `python -m pytest tldw_Server_API/tests -k websearch -v`; Bandit; commit `fix: websearch provider hygiene (TASK-<id>)`.

## Stage 4: Tests for zero-coverage endpoints and crypto

**Goal:** Cover the untested data-deletion and user-file endpoints, and the cookie crypto module.
**Success Criteria:** Each listed endpoint file has integration tests exercising happy path + auth failure; `cookie_cloner.py` has round-trip and negative crypto tests.
**Tests:** This stage *is* tests — integration tests under `tldw_Server_API/tests/`, mirroring the existing `tests/Storage/` patterns where present.
**Status:** Not Started

Backlog task: TASK-13402.

- [ ] **Step 1: Verify the gap.** For each of `storage_trash.py`, `storage_user_files.py`, `storage_user_folders.py`, `quizzes_osce.py`, `discord_oauth_admin.py`, `slack_oauth_admin.py` in `app/api/v1/endpoints/`, run `grep -rln "<module_name>" tldw_Server_API/tests`. Only write tests for files with zero references (the `tests/Storage/` dir may already cover some).
- [ ] **Step 2:** Data-deletion first: `storage_trash`, `storage_user_files`, `storage_user_folders` — test delete/restore/list happy paths plus authz denial (wrong user) using the existing httpx/pytest fixtures pattern from neighboring storage tests. Deletion logic is the highest-risk untested area.
- [ ] **Step 3:** `cookie_cloner.py` (`app/core/Web_Scraping/cookie_scraping/`, 418 lines PBKDF2/AES): unit tests for encrypt→decrypt round-trip, wrong-passphrase failure, malformed ciphertext rejection, and tampered-ciphertext error (no plaintext leak in exceptions).
- [ ] **Step 4:** `quizzes_osce.py`, `discord_oauth_admin.py`, `slack_oauth_admin.py` endpoint tests (auth failure + one happy path each, mocking OAuth providers).
- [ ] **Step 5:** `python -m pytest <new test files> -v`; Bandit; commit `test: cover storage deletion, oauth admin, OSCE, cookie crypto (TASK-<id>)`.

## Stage 5: Security smalls (execute under existing tasks)

**Goal:** Close the two small open security items.
**Success Criteria:** TASK-12113 and TASK-13100 acceptance criteria met per their task files.
**Tests:** As specified in each task.
**Status:** Not Started

- [ ] **Step 1: TASK-12113** — move the voice/STT WebSocket auth token out of the URL query string (header or subprotocol). Read `backlog/tasks/task-12113*.md` for AC; WS endpoint lives under `app/api/v1/endpoints/audio.py`.
- [ ] **Step 2: TASK-13100** — remediate global plaintext web-scraping cookie exposure (`backlog/tasks/task-13100*.md`). This is also the prerequisite for authenticated retrieval in the research program — sequence it before Credit Batch 2 Wave 1 if possible.
- [ ] **Step 3:** Bandit on touched paths; update both task files with verification evidence and final summary.
