# LLM Usage Accounting PR 3 (the evaluation cost-cap removals) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the evaluation cost caps, which never fired: every caller passed a cost of 0.0, the monthly cap was never compared, and `/evaluations/rate-limits` reported the tier's cost numbers truncated to integers. Evaluation LLM spend is capped from now on by the general `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd` quotas (PR 2a and 2b). Closes TASK-13436.

**Architecture:** A removal. `core/Evaluations/user_rate_limiter.py` loses the cost fields, the cost branch of the daily reservation and every cost parameter; its callers in `evaluations_unified.py` stop passing them; `evaluations_auth.py` stops sending the cost headers; the tier config (dataclass, YAML, validation, CLI default) and the frontend type lose the cost fields; three operator and API docs drop their cost-cap sentences. The database keeps its cost columns (nothing new writes them, they default to 0), so there is no migration.

**Tech Stack:** Python 3.12, FastAPI, SQLite (the evaluations limiter has no PostgreSQL path), TypeScript (frontend type only), pytest, vitest.

**Spec:** `Docs/Design/2026-10-09-llm-usage-accounting-design.md`: owner decision 1, the Problem section's "Evaluation cost caps are dead (TASK-13436)" bullet, §8 and Delivery "PR 3". §8 is the list this plan implements, plus the gaps a survey of the code found (see Rulings).

## Global Constraints

- **This plan runs after PR #3234 (LLM usage accounting PR 2b) has merged**, on a branch from the `dev` that contains it. PR 2b edits the same spec's Status line; Task 3 builds on its wording.
- **Behaviour that stays:** the per-minute rate limits, the daily evaluations and evaluation-tokens quotas (`limits.evaluations_per_day`, `limits.evaluation_tokens_per_day`), the Resource Governor path, and every evaluation result. Only the cost caps, their parameters, their headers and their report fields go.
- **No migration.** The `max_cost_per_day`, `max_cost_per_month` (`user_rate_limits`), `cost` (`rate_limit_tracking`) and `total_cost` (`daily_usage`) columns and their DDL stay. Nothing writes or reads them after this PR; existing rows keep their values.
- **Old config keeps loading.** An operator's `evaluations_config.yaml` that still has `max_cost_*` keys must load and validate.
- **Every change was prototyped** in a scratch checkout of `origin/dev` and re-measured on `origin/dev` 950404fe74 (PR 2b merged); it is given here as exact patches (the Appendix; the same patches as files in this plan's SDD workspace). Each test patch was seen failing against `origin/dev` and passing with its source patch. Apply each patch with `git apply --check` first. If a patch does not apply cleanly, or a test does not behave as a step says, STOP and report it with the output. Do not hand-edit around a failed patch.
- **Test-first:** each task applies its test patch, runs it and sees it fail, then applies its source patch.
- **The full Docs/Published refresh is part of this PR.** `Docs/Published` on `dev` is stale against `Docs/` (two `tests/Docs/test_docs_published_refresh.py` tests fail on `dev` since #3227); running `Helper_Scripts/refresh_docs_published.sh` rewrites the stale files too, and the PR carries all of them.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit ids, `git add -A backlog/tasks`.
- **Commits** end with exactly `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `git stash`. Never `--no-verify`.
- **Python:** `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (written `$PY`). Tests run with `TLDW_TEST_NO_DOCKER=1 RUN_EVALUATIONS=1`.
- **Known failures on `dev`:** `tests/Docs/test_docs_published_refresh.py` (2, fixed by this PR's refresh); under `-n 4` across many directories, order-dependent 404s and 401s on `/media` routes and in the vector-store tests. Re-run any other failure alone, and at `origin/dev`, before reporting it.

## Review Focus

1. **An operator with an old `evaluations_config.yaml` that still sets `max_cost_*`.** Expected: it loads and validates; the keys are ignored, and one warning per tier names them and points to `limits.llm_cost_per_day_usd` / `limits.llm_cost_per_month_usd`. (Task 1, `test_no_evaluation_cost_caps.py`)
2. **An admin who calls `upgrade_user_tier` with `max_cost_*` in `custom_limits`.** Expected: the upgrade succeeds, the other limits apply, and a warning names the ignored key. (Task 1)
3. **A client that reads `GET /evaluations/rate-limits` or the `X-RateLimit-*` headers.** Expected: the evaluations and tokens figures are unchanged; the cost fields and the two cost headers are gone. (Task 1)
4. **Existing database rows with cost values.** Expected: reads ignore them; no write touches them. (Task 1)
5. **The frontend evaluations page.** Expected: compiles without the six cost fields; nothing read them. (Task 2)

## File Structure

| File | Change |
|---|---|
| `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py` | Cost fields, cost branch, cost parameters, cost reads and writes, the dead cost header, stale docs; `rate_limit_source` `legacy_cost_only` becomes `legacy_local`; `upgrade_user_tier` warns on unknown `custom_limits` keys. |
| `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py` | 7 `check_rate_limit` and 5 `record_actual_usage` call sites; the `/rate-limits` handler's cost keys. |
| `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py` | The two `*-Cost-Remaining` headers. |
| `tldw_Server_API/app/core/Evaluations/config_manager.py`, `Config_Files/evaluations_config.yaml`, `cli/utils/config.py` | Tier cost fields, validation's required field, YAML entries, CLI default; a warning when an operator's YAML still sets the removed keys. |
| `tldw_Server_API/app/api/v1/schemas/webhook_schemas.py` | The cost keys in a dead duplicate `RateLimitStatusResponse` example (the module is imported nowhere). |
| `apps/packages/ui/src/services/evaluations.ts` and its design-system test fixture | Six cost fields. |
| Three docs and `Docs/Published` | Cost-cap sentences; the full refresh. |
| `tldw_Server_API/tests/...` (11 files, 1 new) | See Task 1. |

## Rulings made while planning

- **Beyond §8** (a survey of the code found these): the `X-RateLimit-Daily-Cost-Remaining` and `X-RateLimit-Monthly-Cost-Remaining` response headers in `evaluations_auth.py`; the dead `X-RateLimit-Cost-Remaining` line in `_generate_rate_limit_headers`; the `rate_limit_source` value `legacy_cost_only`; the module docstring and comments that call the cost caps the only enforced legacy layer; the positional `cost` parameter of `_reserve_request_usage` and of `record_actual_usage` (a leftover argument would land in `reserved_tokens`) and their callers, including `tests/Usage/test_quota_evals_chatbooks.py`; `config_manager`'s `required_fields`; the CLI default; the frontend's `daily_cost` and `monthly_cost`, which §8 omits; two more mirrored docs (`Evaluations_API_Unified_Reference.md`, `Evaluations_Code_Guide.md`).
- **`_reserve_request_usage` drops `config` too.** It is unused once the cost branch goes.
- **`rate_limit_source` becomes `legacy_local`.** It names the path that skips the Resource Governor and applies only the local daily caps. Nothing in the app reads the string; three tests did.
- **`upgrade_user_tier` keeps accepting `max_cost_*` in `custom_limits`, ignores them, and logs a warning for each unknown key.** Three callers and tests pass them; failing would break them for a field that no longer matters; the existing `hasattr` guard ignored unknown keys silently, and a warning stops an operator from believing a cost cap is still in force. Any other unknown key is now warned about too.
- **An operator's YAML that still sets `max_cost_*` gets the same warning**, once per tier at load (`config_manager._parse_rate_limit_configs`). The YAML is where operators set these caps, so it is where they would believe one is still in force.
- **The profile's `quotas.evaluations` block needs no code change:** it passes the limiter's summary through, so it loses the cost keys with it. A test covers it.
- **One test is deleted and two are rewritten.** `test_evaluations_core_hardening.py::test_rate_limiter_cost_reservation_is_atomic` proved cost-cap atomicity; it is rewritten as `test_daily_cap_reservation_is_atomic_under_concurrency` (two concurrent reservations against a daily evaluations cap of 1, exactly one admitted), because it was the only test that ran two reservations at once (`tests/Usage/test_quota_evals_chatbooks.py::test_daily_evaluation_cap_is_atomic_with_the_reservation` is sequential). `test_rg_cutover_evals_authnz_character_web.py::test_evaluations_rg_allows_with_zero_cost_skips_cost_check` is deleted: a duplicate of `test_evaluations_rg_allows_bypasses_legacy` once cost is gone. `..._still_enforces_cost_caps` is rewritten as `..._still_enforces_local_daily_caps`: the "Resource Governor allows, the local reservation still denies" path has no other test.
- **The OpenAPI schema does not change:** `RateLimitStatusResponse`'s inner fields are free-form dicts. Task 1 checks it.
- **The dead `_generate_rate_limit_headers` method stays** (only its cost line goes); deleting it is a separate cleanup. The dead duplicate `RateLimitStatusResponse` in `schemas/webhook_schemas.py` loses the cost keys from its example, so nobody takes it for the endpoint's shape.
- **The Resource Governor PRD** (`Docs/Product/Completed/AuthNZ-Refactor/Resource_Governor_PRD.md`) still describes the cost caps as live legacy enforcement. It is a completed, unmirrored historical document and is left as it is.
- **TASK-13436's acceptance criterion is reworded** to the owner's 2026-10-09 re-scope before it is checked: as written it asks for a cost cap that blocks, which the re-scope replaced with the general LLM cost quota (PR 2a).

---

### Task 1: The backend loses the cost caps

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py`, `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`, `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py`, `tldw_Server_API/app/core/Evaluations/config_manager.py`, `tldw_Server_API/Config_Files/evaluations_config.yaml`, `tldw_Server_API/cli/utils/config.py`, `tldw_Server_API/app/api/v1/schemas/webhook_schemas.py` (the source patch, Appendix A.2)
- Create: `tldw_Server_API/tests/Evaluations/unit/test_no_evaluation_cost_caps.py` (19 cases)
- Modify (tests): `tests/Config/test_module_yaml_integration.py`, `tests/Evaluations/integration/test_rate_limits_endpoint.py`, `tests/Evaluations/test_evaluations_core_hardening.py`, `tests/Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py`, `tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py`, `tests/Evaluations/test_production_features_integration.py`, `tests/Evaluations/test_rate_limits_view_resolver.py`, `tests/Evaluations/unit/test_user_rate_limiter_minute_exact_and_reset.py`, `tests/Resource_Governance/test_rg_cutover_evals_authnz_character_web.py`, `tests/Usage/test_quota_evals_chatbooks.py` (all under `tldw_Server_API/`; the test patch, Appendix A.1)

**Interfaces:**
- Produces: `UserRateLimiter.check_rate_limit(self, user_id, endpoint, is_batch=False, tokens_requested=0, evaluations_requested=1)` (no `estimated_cost`); `record_actual_usage(self, user_id, endpoint, tokens_used, reserved_tokens=0)` (the positional cost is gone, so `reserved_tokens` is now the fourth positional); `_reserve_request_usage(self, user_id, endpoint, tokens_used, max_evaluations_per_day=None, max_tokens_per_day=None, evaluations_requested=1)` (no `cost`, no `config`); `RateLimitConfig` and `RateLimitTierConfig` without `max_cost_per_day` / `max_cost_per_month`; `get_usage_summary()` without cost keys; `rate_limit_source == "legacy_local"`.

- [ ] **Step 1: Apply the test patch**

The patch is Appendix A.1, also saved as `pr3-tests.patch` in this plan's SDD workspace (`.superpowers/sdd/2026-10-13-llm-usage-accounting-pr-3-cost-cap-removal/`). From the worktree root:

```bash
git apply --check .superpowers/sdd/2026-10-13-llm-usage-accounting-pr-3-cost-cap-removal/pr3-tests.patch
git apply .superpowers/sdd/2026-10-13-llm-usage-accounting-pr-3-cost-cap-removal/pr3-tests.patch
```

Expected: both succeed silently.

- [ ] **Step 2: Run the tests to see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 RUN_EVALUATIONS=1 $PY -m pytest -q -p no:cacheprovider -n 4 tldw_Server_API/tests/Evaluations tldw_Server_API/tests/Usage tldw_Server_API/tests/Resource_Governance/test_rg_cutover_evals_authnz_character_web.py tldw_Server_API/tests/Config/test_module_yaml_integration.py`
Expected: 27 failed (on `dev` 950404fe74: `27 failed, 1210 passed, 11 skipped`). The failures: `test_evaluations_core_hardening.py::test_daily_cap_reservation_is_atomic_under_concurrency` (`TypeError`: `dev`'s `_reserve_request_usage` still requires `cost` and `config`); 13 of the 19 cases in `test_no_evaluation_cost_caps.py` (the five tiers, the dataclass fields, the signatures, the shipped YAML, the summary keys, the `custom_limits` warning, the old operator YAML's warning, `legacy_local`, the profile block); `test_rate_limits_view_resolver.py::test_no_cost_headers_are_sent`; `test_rate_limits_endpoint.py::test_get_rate_limits_shape`; `test_user_rate_limiter_minute_exact_and_reset.py::test_custom_tier_skips_minute_checks` and two `test_production_features_integration.py` tests (`legacy_local` expected); stage2 `test_ocr_pdf_uses_consistent_endpoint_key_for_check_and_record` and stage3 `test_batch_parallel_strict_fail_fast_cancels_remaining` and `test_batch_parallel_sanitizes_item_failure` (their fake limiters no longer accept `estimated_cost`, so `dev`'s endpoints raise `TypeError`); five `tests/Usage/test_quota_evals_chatbooks.py` tests (the new `_reserve_request_usage` signature). The other six new cases (the columns staying at 0 and the other guards) pass. A different count is not a failure of the plan if every extra failure is a known one in Global Constraints; anything else, stop and report.

- [ ] **Step 3: Apply the source patch**

```bash
git apply --check .superpowers/sdd/2026-10-13-llm-usage-accounting-pr-3-cost-cap-removal/pr3-src.patch
git apply .superpowers/sdd/2026-10-13-llm-usage-accounting-pr-3-cost-cap-removal/pr3-src.patch
```

(Appendix A.2.) The one subtle edit: `_get_user_config` reads `expires_at` from `row[6]` now that the two cost columns are no longer selected.

- [ ] **Step 4: Run the tests to see them pass**

Run the Step 2 command, and add `tldw_Server_API/tests/UserProfile`.
Expected: all pass (on `dev` 950404fe74: `1596 passed, 11 skipped`).

If a run passes and then aborts at interpreter exit with `libc++abi ... recursive_mutex lock failed`, re-run it once: the reviewer's dry run saw this after two unrelated test files and attributes it to the environment.

- [ ] **Step 5: Check nothing else uses the cost fields**

Run: `git grep -n -E "max_cost_per|estimated_cost|legacy_cost_only|cost_remaining|Cost-Remaining|daily_cost|monthly_cost|cost_today|cost_month|cost_per_day|cost_per_month" -- tldw_Server_API/app/core/Evaluations tldw_Server_API/app/api/v1/endpoints/evaluations tldw_Server_API/app/core/UserProfiles tldw_Server_API/cli/utils/config.py tldw_Server_API/Config_Files/evaluations_config.yaml`
Expected: exactly six lines: two in `config_manager.py` (the YAML warning's key list and its message), and four in `user_rate_limiter.py` (the module docstring naming `limits.llm_cost_per_day_usd` / `limits.llm_cost_per_month_usd`, the two DDL lines `max_cost_per_day REAL DEFAULT 10.0` and `max_cost_per_month REAL DEFAULT 100.0`, and the `upgrade_user_tier` comment). (The same names appear elsewhere in the app for unrelated features: audit, claims, ACP, the LLM quota keys, Resource Governor cost units.)

- [ ] **Step 6: The OpenAPI schema and lint**

Run: `$PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json`
Expected: `OK — OpenAPI fingerprint matches the checked-in snapshot.`

Compare ruff finding counts with `origin/dev` for every changed Python file (`$PY -m ruff check --output-format concise <file> | grep -c ':[0-9]*:[0-9]*:'` against `git show origin/dev:<file> | $PY -m ruff check --output-format concise --stdin-filename <file> - | grep -c ':[0-9]*:[0-9]*:'`). On `dev` 950404fe74: `evaluations_auth.py` 1, `evaluations_unified.py` 1, `config_manager.py` 0, `user_rate_limiter.py` 1, `cli/utils/config.py` 0, `webhook_schemas.py` 0, `test_module_yaml_integration.py` 0, `test_rate_limits_endpoint.py` 1, `test_evaluations_core_hardening.py` 1, stage2 0, stage3 0, `test_production_features_integration.py` 13, `test_rate_limits_view_resolver.py` 0, `test_user_rate_limiter_minute_exact_and_reset.py` 2, `test_rg_cutover_evals_authnz_character_web.py` 1, `test_quota_evals_chatbooks.py` 0; the new file must be clean. Counts must not grow.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Evaluations/user_rate_limiter.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py tldw_Server_API/app/core/Evaluations/config_manager.py tldw_Server_API/Config_Files/evaluations_config.yaml tldw_Server_API/cli/utils/config.py tldw_Server_API/app/api/v1/schemas/webhook_schemas.py tldw_Server_API/tests/Evaluations/unit/test_no_evaluation_cost_caps.py tldw_Server_API/tests/Config/test_module_yaml_integration.py tldw_Server_API/tests/Evaluations/integration/test_rate_limits_endpoint.py tldw_Server_API/tests/Evaluations/test_evaluations_core_hardening.py tldw_Server_API/tests/Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py tldw_Server_API/tests/Evaluations/test_production_features_integration.py tldw_Server_API/tests/Evaluations/test_rate_limits_view_resolver.py tldw_Server_API/tests/Evaluations/unit/test_user_rate_limiter_minute_exact_and_reset.py tldw_Server_API/tests/Resource_Governance/test_rg_cutover_evals_authnz_character_web.py tldw_Server_API/tests/Usage/test_quota_evals_chatbooks.py
git status --short
git commit -m "refactor(evals): remove the evaluation cost caps, which never fired; LLM spend is capped by limits.llm_cost_* (TASK-13436)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

`git status --short` before the commit must show only these files staged.

---

### Task 2: The frontend type loses the cost fields

**Files:**
- Modify: `apps/packages/ui/src/services/evaluations.ts`, `apps/packages/ui/src/components/Option/Evaluations/components/__tests__/EvaluationComponentAlerts.design-system.test.tsx` (Appendix A.3)

- [ ] **Step 1: Apply the patch**

```bash
git apply --check .superpowers/sdd/2026-10-13-llm-usage-accounting-pr-3-cost-cap-removal/pr3-frontend.patch
git apply .superpowers/sdd/2026-10-13-llm-usage-accounting-pr-3-cost-cap-removal/pr3-frontend.patch
```

- [ ] **Step 2: Nothing else reads the fields**

Run: `git grep -n -E "cost_per_day|cost_per_month|cost_today|cost_month|daily_cost|monthly_cost" -- apps`
Expected: no output.

- [ ] **Step 3: Run the fixture's test, if the frontend's dependencies are installed**

If `apps/packages/ui/node_modules/.bin/vitest` exists: `cd apps/packages/ui && node_modules/.bin/vitest run src/components/Option/Evaluations/components/__tests__/EvaluationComponentAlerts.design-system.test.tsx` (expected `5 passed`; return to the worktree root afterwards). If it does not exist, say so and skip. The package has no typecheck script, and a full `tsc --noEmit` needs `NODE_OPTIONS=--max-old-space-size=8192`; it is optional, since the change only deletes six optional fields from a type and a fixture and Step 2 shows nothing reads them.

- [ ] **Step 4: Commit**

```bash
git add apps/packages/ui/src/services/evaluations.ts apps/packages/ui/src/components/Option/Evaluations/components/__tests__/EvaluationComponentAlerts.design-system.test.tsx
git commit -m "refactor(evals-ui): the rate-limit status type loses the cost fields the server no longer sends (TASK-13436)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: The docs, the spec and the backlog

**Files:**
- Modify: `Docs/User_Guides/Server/Production_Hardening_Checklist.md`, `Docs/API-related/Evaluations_API_Unified_Reference.md`, `Docs/Code_Documentation/Guides/Evaluations_Code_Guide.md` (Appendix A.4), `Docs/Published/**` (the refresh), `Docs/Design/2026-10-09-llm-usage-accounting-design.md`
- Modify: `backlog/tasks/task-13436*.md` (through backlog-py only)

- [ ] **Step 1: Apply the docs patch and refresh the published copies**

```bash
git apply --check .superpowers/sdd/2026-10-13-llm-usage-accounting-pr-3-cost-cap-removal/pr3-docs.patch
git apply .superpowers/sdd/2026-10-13-llm-usage-accounting-pr-3-cost-cap-removal/pr3-docs.patch
bash Helper_Scripts/refresh_docs_published.sh
git status --short -- Docs/Published
```

Expected: the three docs' `Docs/Published` copies change, and so do the files that were already stale on `dev` (on the prototype's base: four Embeddings guides, `Evaluations/Evaluation_Plans.md`, `User_Guides/Server/Multi-User_Deployment_Guide.md`, `API-related/Audio_Transcription_API.md`). The stale set may differ by the time this runs; carry whatever the refresh produces.

- [ ] **Step 2: The spec**

In `Docs/Design/2026-10-09-llm-usage-accounting-design.md`, use the Edit tool:

1. In the **Status** line (line 4), replace `PR 2b in progress.` with `PR 2b (#3234) merged; PR 3 in progress.` That text is there once PR #3234 has merged (this plan's precondition); if it is not, stop and report what the line says.
2. At the end of §8, after `Evaluation LLM spend is capped by the general LLM quotas from then on.`, add:

```markdown

Found while building PR 3, and removed with the rest: the `X-RateLimit-Daily-Cost-Remaining` and `X-RateLimit-Monthly-Cost-Remaining` response headers (`evaluations_auth.py`) and the dead `X-RateLimit-Cost-Remaining` line; the `rate_limit_source` value `legacy_cost_only`, now `legacy_local`; the positional `cost` of `_reserve_request_usage` and `record_actual_usage`; `config_manager`'s required `max_cost_per_day`; the CLI default; the frontend's `daily_cost` and `monthly_cost`; two more mirrored docs. `upgrade_user_tier` ignores `max_cost_*` in `custom_limits` with a warning, and an old `evaluations_config.yaml` that still sets them loads with a warning per tier.
```

- [ ] **Step 3: The backlog**

`TASK-13436` is `To Do` on `dev`, and its one acceptance criterion predates the owner's 2026-10-09 re-scope ("A per-user daily and monthly evaluation cost cap blocks once recorded cost reaches it"). The re-scope replaced that cap with the general LLM cost quota, so the criterion is reworded before it is checked. First run bandit on the changed Python source (DoD #4):

```bash
uvx bandit -ll tldw_Server_API/app/core/Evaluations/user_rate_limiter.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py tldw_Server_API/app/core/Evaluations/config_manager.py tldw_Server_API/cli/utils/config.py tldw_Server_API/app/api/v1/schemas/webhook_schemas.py
```

Expected: `No issues identified.` (on `dev` 950404fe74 with the patches: 0 medium or high, 2 low below the `-ll` threshold). If it reports a medium or high issue, stop and report it. Then, from the worktree root, run one command at a time:

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13436 --append-notes "PR 3: the evaluation cost caps are removed: RateLimitConfig and the tier config lose max_cost_per_day/month, the daily reservation's cost branch and every cost parameter of check_rate_limit, _reserve_request_usage, _record_request and record_actual_usage go, /evaluations/rate-limits and the X-RateLimit-*-Cost-Remaining headers stop reporting cost, and the frontend type loses the six cost fields. The DB columns stay (no migration). Evaluation LLM spend is capped by limits.llm_cost_per_day_usd and limits.llm_cost_per_month_usd."
```

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13436 --remove-ac 1 --ac "The evaluation-only cost caps are removed (no cost field, parameter, header or YAML key remains in the evaluations limiter, its config, GET /evaluations/rate-limits or the frontend type), and evaluation LLM spend is capped by limits.llm_cost_per_day_usd and limits.llm_cost_per_month_usd (PR 2a, #3230)" --check-ac 1 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "The evaluation cost caps never fired: every caller passed a cost of 0.0, the monthly cap was never compared, and /evaluations/rate-limits reported the tier's numbers truncated to integers. PR 3 removes them from the limiter, its config and YAML, the endpoints, the response headers, the frontend type and the docs; the database columns stay (no migration). Evaluation LLM spend is capped by the general limits.llm_cost_per_day_usd and limits.llm_cost_per_month_usd quotas. Acceptance criterion 1 was reworded to the 2026-10-09 re-scope." --status Done
```

`--check-ac` applies after `--remove-ac` and `--ac`. Then:

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task normalize --check backlog/tasks/task-13436*.md
```

Expected: the edit succeeds (the task shows `Done`) and no file is listed.

- [ ] **Step 4: Run the docs tests**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Docs`
Expected: all pass (the refresh fixes the two `test_docs_published_refresh.py` tests that fail on `dev`).

- [ ] **Step 5: Commit**

```bash
git add Docs/User_Guides/Server/Production_Hardening_Checklist.md Docs/API-related/Evaluations_API_Unified_Reference.md Docs/Code_Documentation/Guides/Evaluations_Code_Guide.md Docs/Design/2026-10-09-llm-usage-accounting-design.md
git add -A Docs/Published backlog/tasks
git status --short
git commit -m "docs(evals): the cost caps are gone from the operator and API docs; refresh Docs/Published (TASK-13436)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

## Appendix: the patches

Each patch was made against `origin/dev` c65df167b9 and applies cleanly to 950404fe74 (PR 2b merged), where every count in this plan was measured; it applies to any later `dev` that has not changed these files. They are also saved, byte for byte, as files in this plan's SDD workspace; apply the files, not a hand copy of these blocks.

### A.1 `pr3-tests.patch`

```diff
diff --git a/tldw_Server_API/tests/Config/test_module_yaml_integration.py b/tldw_Server_API/tests/Config/test_module_yaml_integration.py
index 6623ccc28c..17656a1ca1 100644
--- a/tldw_Server_API/tests/Config/test_module_yaml_integration.py
+++ b/tldw_Server_API/tests/Config/test_module_yaml_integration.py
@@ -297,8 +297,6 @@ def test_evaluations_precedence_env_over_config_over_yaml(tmp_path, monkeypatch)
                 "      evaluations_per_day: 1",
                 "      total_tokens_per_day: 1",
                 "      burst_size: 1",
-                "      max_cost_per_day: 1.0",
-                "      max_cost_per_month: 1.0",
             ]
         )
         + "\n",
diff --git a/tldw_Server_API/tests/Evaluations/integration/test_rate_limits_endpoint.py b/tldw_Server_API/tests/Evaluations/integration/test_rate_limits_endpoint.py
index 12376c8463..ebebdaf58c 100644
--- a/tldw_Server_API/tests/Evaluations/integration/test_rate_limits_endpoint.py
+++ b/tldw_Server_API/tests/Evaluations/integration/test_rate_limits_endpoint.py
@@ -33,8 +33,6 @@ def test_get_rate_limits_shape():
         "evaluations_per_minute",
         "evaluations_per_day",
         "tokens_per_day",
-        "cost_per_day",
-        "cost_per_month",
     ]:
         assert k in lim, f"Missing limits key: {k}"
         if k in ("evaluations_per_day", "tokens_per_day"):
@@ -47,8 +45,6 @@ def test_get_rate_limits_shape():
     for k in [
         "evaluations_today",
         "tokens_today",
-        "cost_today",
-        "cost_month",
     ]:
         assert k in usage, f"Missing usage key: {k}"
         assert isinstance(usage[k], int)
@@ -58,14 +54,14 @@ def test_get_rate_limits_shape():
     for k in [
         "daily_evaluations",
         "daily_tokens",
-        "daily_cost",
-        "monthly_cost",
     ]:
         assert k in rem, f"Missing remaining key: {k}"
-        if k in ("daily_evaluations", "daily_tokens"):
-            assert rem[k] is None or isinstance(rem[k], int)  # None is unlimited
-        else:
-            assert isinstance(rem[k], int)
+        assert rem[k] is None or isinstance(rem[k], int)  # None is unlimited
+
+    # Evaluation cost caps were removed (TASK-13436): no cost fields anywhere.
+    assert set(lim) == {"evaluations_per_minute", "evaluations_per_day", "tokens_per_day"}
+    assert set(usage) == {"evaluations_today", "tokens_today"}
+    assert set(rem) == {"daily_evaluations", "daily_tokens"}
 
 
 def test_rate_limits_daily_caps_are_null_when_unlimited(monkeypatch):
diff --git a/tldw_Server_API/tests/Evaluations/test_evaluations_core_hardening.py b/tldw_Server_API/tests/Evaluations/test_evaluations_core_hardening.py
index b064fb47bc..40fe68ea54 100644
--- a/tldw_Server_API/tests/Evaluations/test_evaluations_core_hardening.py
+++ b/tldw_Server_API/tests/Evaluations/test_evaluations_core_hardening.py
@@ -24,7 +24,7 @@ from tldw_Server_API.app.core.Evaluations.db_adapter import (
 from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner
 from tldw_Server_API.app.core.Evaluations.synthetic_eval_service import SyntheticEvalWorkflowService
 from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import UnifiedEvaluationService
-from tldw_Server_API.app.core.Evaluations.user_rate_limiter import UserRateLimiter, UserTier
+from tldw_Server_API.app.core.Evaluations.user_rate_limiter import UserRateLimiter
 from tldw_Server_API.app.core.Evaluations.webhook_manager import WebhookPayload, WebhookManager
 from tldw_Server_API.app.core.Evaluations.webhook_security import WebhookSecurityValidator
 
@@ -318,48 +318,16 @@ def test_postgresql_adapter_is_not_constructible_until_implemented() -> None:
 
 
 @pytest.mark.asyncio
-async def test_rate_limiter_cost_reservation_is_atomic(tmp_path, monkeypatch) -> None:
+async def test_daily_cap_reservation_is_atomic_under_concurrency(tmp_path) -> None:
+    """Two concurrent reservations against a daily cap of 1: exactly one is admitted."""
     limiter = UserRateLimiter(db_path=str(tmp_path / "rate_limits.db"))
-    user_id = "cost-user"
-    endpoint = "/api/v1/evaluations/geval"
-    await limiter.upgrade_user_tier(
-        user_id=user_id,
-        new_tier=UserTier.CUSTOM,
-        custom_limits={
-            "evaluations_per_minute": 100,
-            "batch_evaluations_per_minute": 100,
-            "evaluations_per_day": 100,
-            "total_tokens_per_day": 100_000,
-            "burst_size": 0,
-            "max_cost_per_day": 0.05,
-            "max_cost_per_month": 10.0,
-        },
-    )
-
-    barrier = asyncio.Event()
-    waiters = 0
-    original_record_request = limiter._record_request
-
-    async def _delayed_record_request(*args, **kwargs):
-        nonlocal waiters
-        waiters += 1
-        if waiters == 2:
-            barrier.set()
-        await asyncio.wait_for(barrier.wait(), timeout=1)
-        return await original_record_request(*args, **kwargs)
-
-    monkeypatch.setattr(limiter, "_record_request", _delayed_record_request)
-
     results = await asyncio.gather(
-        limiter.check_rate_limit(user_id, endpoint, estimated_cost=0.04),
-        limiter.check_rate_limit(user_id, endpoint, estimated_cost=0.04),
+        limiter._reserve_request_usage("7", "/e", 0, max_evaluations_per_day=1),
+        limiter._reserve_request_usage("7", "/e", 0, max_evaluations_per_day=1),
     )
-
-    allowed_count = sum(1 for allowed, _metadata in results if allowed)
-    denied_metadata = [metadata for allowed, metadata in results if not allowed]
-    assert allowed_count == 1  # nosec B101
-    assert len(denied_metadata) == 1  # nosec B101
-    assert denied_metadata[0]["error"] == "Daily cost limit exceeded"  # nosec B101
+    assert sum(1 for allowed, _metadata in results if allowed) == 1  # nosec B101
+    denied = [metadata for allowed, metadata in results if not allowed]
+    assert len(denied) == 1 and denied[0]["error"] == "Daily evaluation limit exceeded"  # nosec B101
 
 
 def test_synthetic_workflow_requires_user_for_owner_scoped_operations(tmp_path) -> None:
diff --git a/tldw_Server_API/tests/Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py b/tldw_Server_API/tests/Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py
index 03ab8d5406..785cfff125 100644
--- a/tldw_Server_API/tests/Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py
+++ b/tldw_Server_API/tests/Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py
@@ -107,10 +107,9 @@ def test_ocr_pdf_uses_consistent_endpoint_key_for_check_and_record(monkeypatch):
             endpoint: str,
             is_batch: bool,
             tokens_requested: int,
-            estimated_cost: float,
             evaluations_requested: int = 1,
         ):
-            _ = (is_batch, tokens_requested, estimated_cost, evaluations_requested)
+            _ = (is_batch, tokens_requested, evaluations_requested)
             self.checked_endpoints.append(endpoint)
             return True, {"retry_after": 0}
 
@@ -119,7 +118,6 @@ def test_ocr_pdf_uses_consistent_endpoint_key_for_check_and_record(monkeypatch):
             _user_id: str,
             endpoint: str,
             _tokens_used: int,
-            _cost: float = 0.0,
             reserved_tokens: int = 0,
         ):
             _ = reserved_tokens
diff --git a/tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py b/tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py
index a92bcef6d2..d637ff4620 100644
--- a/tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py
+++ b/tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py
@@ -264,10 +264,9 @@ def test_batch_parallel_strict_fail_fast_cancels_remaining(monkeypatch):
             endpoint: str,
             is_batch: bool,
             tokens_requested: int,
-            estimated_cost: float,
             evaluations_requested: int = 1,
         ):
-            _ = (endpoint, is_batch, tokens_requested, estimated_cost, evaluations_requested)
+            _ = (endpoint, is_batch, tokens_requested, evaluations_requested)
             return True, {"retry_after": 0}
 
     started_items: list[int] = []
@@ -378,10 +377,9 @@ def test_batch_parallel_sanitizes_item_failure(monkeypatch):
             endpoint: str,
             is_batch: bool,
             tokens_requested: int,
-            estimated_cost: float,
             evaluations_requested: int = 1,
         ):
-            _ = (endpoint, is_batch, tokens_requested, estimated_cost, evaluations_requested)
+            _ = (endpoint, is_batch, tokens_requested, evaluations_requested)
             return True, {"retry_after": 0}
 
     class _Service:
diff --git a/tldw_Server_API/tests/Evaluations/test_production_features_integration.py b/tldw_Server_API/tests/Evaluations/test_production_features_integration.py
index 4362caa258..20baa351e5 100644
--- a/tldw_Server_API/tests/Evaluations/test_production_features_integration.py
+++ b/tldw_Server_API/tests/Evaluations/test_production_features_integration.py
@@ -206,7 +206,7 @@ class TestUserRateLimiter:
 
     @pytest.mark.asyncio
     async def test_rate_limit_enforcement(self, rate_limiter):
-        """Test current Phase-2 behavior for CUSTOM tier (legacy cost-only path)."""
+        """Test current Phase-2 behavior for CUSTOM tier (legacy local path)."""
         user_id = "test_user"
         endpoint = "/api/v1/evaluations"
 
@@ -218,13 +218,13 @@ class TestUserRateLimiter:
             custom_limits={"evaluations_per_minute": 2}
         )
 
-        # All requests should be allowed in legacy_cost_only mode.
+        # All requests should be allowed in legacy_local mode.
         for i in range(3):
             allowed, metadata = await rate_limiter.check_rate_limit(
                 user_id, endpoint
             )
             assert allowed is True
-            assert metadata.get("rate_limit_source") == "legacy_cost_only"
+            assert metadata.get("rate_limit_source") == "legacy_local"
 
     @pytest.mark.asyncio
     async def test_tier_upgrade(self, rate_limiter):
@@ -256,8 +256,7 @@ class TestUserRateLimiter:
         for _ in range(3):
             await rate_limiter.check_rate_limit(
                 user_id, "/api/v1/evaluations",
-                tokens_requested=100,
-                estimated_cost=0.01
+                tokens_requested=100
             )
 
         # Get usage summary
@@ -266,7 +265,6 @@ class TestUserRateLimiter:
         assert summary["user_id"] == user_id
         assert summary["usage"]["today"]["evaluations"] == 3
         assert summary["usage"]["today"]["tokens"] == 300
-        assert summary["usage"]["today"]["cost"] == pytest.approx(0.03)
 
     @pytest.mark.asyncio
     async def test_burst_allowance(self, rate_limiter):
@@ -289,7 +287,7 @@ class TestUserRateLimiter:
                 user_id, "/api/v1/evaluations"
             )
             assert allowed is True
-            assert metadata.get("rate_limit_source") == "legacy_cost_only"
+            assert metadata.get("rate_limit_source") == "legacy_local"
 
 
 @pytest_asyncio.fixture
@@ -749,8 +747,7 @@ class TestIntegration:
         allowed, rate_metadata = await rate_limiter.check_rate_limit(
             user_id,
             "/api/v1/evaluations",
-            tokens_requested=1000,
-            estimated_cost=0.10
+            tokens_requested=1000
         )
         assert allowed is True
 
@@ -797,7 +794,7 @@ class TestIntegration:
         # 5. Check usage
         usage = await rate_limiter.get_usage_summary(user_id)
         assert usage["usage"]["today"]["evaluations"] == 1
-        assert usage["usage"]["today"]["cost"] == pytest.approx(0.10)
+        assert usage["usage"]["today"]["tokens"] == 1000
 
 
 # Run tests with pytest
diff --git a/tldw_Server_API/tests/Evaluations/test_rate_limits_view_resolver.py b/tldw_Server_API/tests/Evaluations/test_rate_limits_view_resolver.py
index 9ff39fc78d..9ab45c98e2 100644
--- a/tldw_Server_API/tests/Evaluations/test_rate_limits_view_resolver.py
+++ b/tldw_Server_API/tests/Evaluations/test_rate_limits_view_resolver.py
@@ -71,3 +71,10 @@ async def test_daily_headers_absent_when_unlimited(caps: dict, tmp_path) -> None
     assert "x-ratelimit-daily-limit" not in headers
     assert "x-ratelimit-daily-remaining" not in headers
     assert "x-ratelimit-tokens-remaining" not in headers
+
+
+async def test_no_cost_headers_are_sent(caps: dict, tmp_path) -> None:
+    """The evaluation cost caps are gone, so no X-RateLimit-*-Cost-Remaining header is advertised."""
+    caps["limits.evaluations_per_day"] = 5
+    headers = await _headers_for(tmp_path)
+    assert not [h for h in headers if "cost" in h]
diff --git a/tldw_Server_API/tests/Evaluations/unit/test_no_evaluation_cost_caps.py b/tldw_Server_API/tests/Evaluations/unit/test_no_evaluation_cost_caps.py
new file mode 100644
index 0000000000..d332822465
--- /dev/null
+++ b/tldw_Server_API/tests/Evaluations/unit/test_no_evaluation_cost_caps.py
@@ -0,0 +1,149 @@
+"""The evaluation cost caps are gone (TASK-13436); LLM spend is capped by the general llm_cost_* quotas."""
+
+import dataclasses
+import inspect
+import sqlite3
+from pathlib import Path
+
+import pytest
+
+from tldw_Server_API.app.core.Evaluations import user_rate_limiter
+from tldw_Server_API.app.core.Evaluations.config_manager import (
+    EvaluationsConfigManager,
+    RateLimitTierConfig,
+)
+from tldw_Server_API.app.core.Evaluations.user_rate_limiter import (
+    RateLimitConfig,
+    UserRateLimiter,
+    UserTier,
+)
+
+pytestmark = pytest.mark.unit
+
+_SHIPPED_YAML = Path(__file__).resolve().parents[3] / "Config_Files" / "evaluations_config.yaml"
+
+
+def _cost_fields(cls) -> list[str]:
+    """Dataclass fields of `cls` that mention cost."""
+    return [f.name for f in dataclasses.fields(cls) if "cost" in f.name]
+
+
+def test_rate_limit_dataclasses_have_no_cost_fields() -> None:
+    assert _cost_fields(RateLimitConfig) == []
+    assert _cost_fields(RateLimitTierConfig) == []
+
+
+@pytest.mark.parametrize("tier", list(UserTier))
+def test_every_tier_builds_without_cost_numbers(tier: UserTier) -> None:
+    assert not [k for k in vars(RateLimitConfig.for_tier(tier)) if "cost" in k]
+
+
+def test_limiter_methods_take_no_cost_parameter() -> None:
+    for name in ("check_rate_limit", "record_actual_usage", "_reserve_request_usage", "_record_request"):
+        params = inspect.signature(getattr(UserRateLimiter, name)).parameters
+        assert not [p for p in params if "cost" in p], (name, list(params))
+    assert not hasattr(UserRateLimiter, "_check_cost_limits")
+
+
+def test_shipped_yaml_validates_with_no_cost_fields() -> None:
+    manager = EvaluationsConfigManager(config_path=_SHIPPED_YAML, enable_hot_reload=False)
+    assert manager.validate_config() == []
+    assert "max_cost" not in _SHIPPED_YAML.read_text()
+    for tier in ("free", "basic", "premium", "enterprise", "custom"):
+        assert manager.get_rate_limit_tier_config(tier) is not None
+
+
+def test_an_operator_yaml_that_still_has_cost_keys_still_loads(tmp_path: Path) -> None:
+    """Old operator files keep working: the keys are ignored, not required and not rejected."""
+    from loguru import logger
+
+    yaml_path = tmp_path / "evaluations_config.yaml"
+    yaml_path.write_text(
+        "rate_limiting:\n"
+        "  tiers:\n"
+        "    free:\n"
+        "      evaluations_per_minute: 7\n"
+        "      evaluations_per_day: 70\n"
+        "      total_tokens_per_day: 7000\n"
+        "      max_cost_per_day: 1.0\n"
+        "      max_cost_per_month: 10.0\n"
+    )
+    messages: list[str] = []
+    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
+    try:
+        manager = EvaluationsConfigManager(config_path=yaml_path, enable_hot_reload=False)
+    finally:
+        logger.remove(sink)
+    assert manager.validate_config() == []
+    assert manager.get_rate_limit_tier_config("free").evaluations_per_minute == 7
+    # The operator is told the keys do nothing, so nobody believes a cost cap is in force.
+    warned = [m for m in messages if "'free'" in m and "max_cost_per_day" in m and "llm_cost_per_day_usd" in m]
+    assert len(warned) == 1
+
+
+async def test_usage_summary_has_no_cost_keys(tmp_path) -> None:
+    summary = await UserRateLimiter(db_path=str(tmp_path / "evals.db")).get_usage_summary("7")
+    assert set(summary["limits"]) == {"per_minute", "daily"}
+    assert set(summary["limits"]["daily"]) == {"evaluations", "tokens"}
+    assert set(summary["usage"]) == {"today"}
+    assert set(summary["usage"]["today"]) == {"evaluations", "tokens"}
+    assert set(summary["remaining"]) == {"daily_evaluations", "daily_tokens"}
+
+
+async def test_custom_limits_with_cost_keys_are_ignored_and_warned_about(tmp_path) -> None:
+    """upgrade_user_tier still succeeds when a caller passes the removed max_cost_* keys."""
+    from loguru import logger
+
+    messages: list[str] = []
+    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
+    try:
+        limiter = UserRateLimiter(db_path=str(tmp_path / "evals.db"))
+        ok = await limiter.upgrade_user_tier(
+            "7", UserTier.CUSTOM, custom_limits={"evaluations_per_minute": 3, "max_cost_per_day": 0.01}
+        )
+    finally:
+        logger.remove(sink)
+    assert ok is True
+    assert (await limiter._get_user_config("7")).evaluations_per_minute == 3
+    assert any("max_cost_per_day" in m for m in messages)
+
+
+async def test_nothing_writes_the_cost_columns(tmp_path) -> None:
+    """The columns stay in the tables (no migration) but a reservation and a record leave them at 0."""
+    db_path = tmp_path / "evals.db"
+    limiter = UserRateLimiter(db_path=str(db_path))
+    allowed, _meta = await limiter.check_rate_limit("7", "evals:geval", tokens_requested=50)
+    assert allowed is True
+    await limiter.record_actual_usage("7", "evals:geval", 40, reserved_tokens=50)
+    with sqlite3.connect(db_path) as conn:
+        assert conn.execute("SELECT SUM(total_cost) FROM daily_usage").fetchone()[0] == 0.0
+        assert conn.execute("SELECT SUM(cost) FROM rate_limit_tracking").fetchone()[0] == 0.0
+        assert conn.execute("SELECT total_evaluations, total_tokens FROM daily_usage").fetchone() == (1, 40)
+
+
+async def test_the_old_cost_only_source_name_is_gone(tmp_path, monkeypatch) -> None:
+    monkeypatch.setenv("RG_ENABLED", "0")
+    limiter = UserRateLimiter(db_path=str(tmp_path / "evals.db"))
+    _ok, meta = await limiter.check_rate_limit("7", "evals:geval")
+    assert meta["rate_limit_source"] == "legacy_local"
+
+
+async def test_profile_evaluations_quotas_have_no_cost_keys(tmp_path, monkeypatch) -> None:
+    """quotas.evaluations in the user profile is the limiter's summary, so it loses the cost keys too."""
+    from tldw_Server_API.app.core.UserProfiles.service import UserProfileService
+    from tldw_Server_API.app.services import storage_quota_service
+
+    limiter = UserRateLimiter(db_path=str(tmp_path / "evals.db"))
+    monkeypatch.setattr(user_rate_limiter, "get_user_rate_limiter_for_user", lambda _uid: limiter)
+
+    async def _no_storage():
+        raise RuntimeError("storage not needed for this test")
+
+    monkeypatch.setattr(storage_quota_service, "get_storage_service", _no_storage)
+
+    service = UserProfileService(object())  # type: ignore[arg-type]
+    quotas = await service._build_quotas({"id": 7, "storage_quota_mb": 1, "storage_used_mb": 0.0})
+    block = quotas["evaluations"]
+    assert set(block["limits"]["daily"]) == {"evaluations", "tokens"}
+    assert set(block["usage"]["today"]) == {"evaluations", "tokens"}
+    assert set(block["remaining"]) == {"daily_evaluations", "daily_tokens"}
diff --git a/tldw_Server_API/tests/Evaluations/unit/test_user_rate_limiter_minute_exact_and_reset.py b/tldw_Server_API/tests/Evaluations/unit/test_user_rate_limiter_minute_exact_and_reset.py
index 22babd55cf..d66ce01d07 100644
--- a/tldw_Server_API/tests/Evaluations/unit/test_user_rate_limiter_minute_exact_and_reset.py
+++ b/tldw_Server_API/tests/Evaluations/unit/test_user_rate_limiter_minute_exact_and_reset.py
@@ -12,8 +12,8 @@ from tldw_Server_API.app.core.Evaluations import user_rate_limiter as evals_rl
 
 
 @pytest.mark.unit
-def test_custom_tier_enforces_cost_only(tmp_path, monkeypatch):
-    """Phase 2: CUSTOM tier skips minute/daily checks; only cost caps enforced."""
+def test_custom_tier_skips_minute_checks(tmp_path, monkeypatch):
+    """Phase 2: CUSTOM tier skips the minute checks (the local daily caps still apply)."""
     db_path = tmp_path / "evals_rate_limit.db"
     limiter = UserRateLimiter(db_path=str(db_path))
     monkeypatch.setattr(evals_rl, "_EVALS_DEPRECATION_WARNED", False)
@@ -31,8 +31,6 @@ def test_custom_tier_enforces_cost_only(tmp_path, monkeypatch):
                 "evaluations_per_day": 100,
                 "total_tokens_per_day": 100000,
                 "burst_size": 0,
-                "max_cost_per_day": 10.0,
-                "max_cost_per_month": 100.0,
             },
         )
 
@@ -49,10 +47,9 @@ def test_custom_tier_enforces_cost_only(tmp_path, monkeypatch):
                     endpoint=endpoint,
                     is_batch=False,
                     tokens_requested=0,
-                    estimated_cost=0.0,
                 )
                 assert ok is True
-                assert meta.get("rate_limit_source") == "legacy_cost_only"
+                assert meta.get("rate_limit_source") == "legacy_local"
                 assert meta.get("tier") == "custom"
 
             deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
diff --git a/tldw_Server_API/tests/Resource_Governance/test_rg_cutover_evals_authnz_character_web.py b/tldw_Server_API/tests/Resource_Governance/test_rg_cutover_evals_authnz_character_web.py
index 55801c22d0..96c6541d84 100644
--- a/tldw_Server_API/tests/Resource_Governance/test_rg_cutover_evals_authnz_character_web.py
+++ b/tldw_Server_API/tests/Resource_Governance/test_rg_cutover_evals_authnz_character_web.py
@@ -71,7 +71,6 @@ async def test_evaluations_rg_allows_bypasses_legacy(monkeypatch):
         user_id="user-123",
         endpoint="/api/v1/evaluations",
         tokens_requested=123,
-        estimated_cost=0.0,
     )
 
     assert allowed is True
@@ -81,7 +80,7 @@ async def test_evaluations_rg_allows_bypasses_legacy(monkeypatch):
 
 @pytest.mark.asyncio
 async def test_evaluations_rg_unavailable_fail_open(monkeypatch):
-    """Phase 2: RG returns None → fail-open for eval/token caps, cost-only enforcement."""
+    """Phase 2: RG returns None → fail-open for RG's eval/token caps; the local daily caps still apply."""
     monkeypatch.setenv("RG_ENABLED", "1")
     monkeypatch.setattr(evals_rl, "_rg_evals_fallback_logged", False)
     monkeypatch.setattr(evals_rl, "_EVALS_DEPRECATION_WARNED", False)
@@ -100,7 +99,6 @@ async def test_evaluations_rg_unavailable_fail_open(monkeypatch):
             user_id="user-123",
             endpoint="/api/v1/evaluations",
             tokens_requested=123,
-            estimated_cost=0.0,
         )
 
         assert allowed is True
@@ -217,10 +215,10 @@ async def test_web_scraping_rg_unavailable_fail_open(monkeypatch):
 
 
 @pytest.mark.asyncio
-async def test_evaluations_rg_allows_still_enforces_cost_caps(monkeypatch):
+async def test_evaluations_rg_allows_still_enforces_local_daily_caps(monkeypatch):
     """
     When RG allows a request, the Evaluations limiter must still enforce
-    cost caps locally (since RG does not handle cost limits).
+    the per-user daily evaluations/tokens quotas locally.
     """
     monkeypatch.setenv("RG_ENABLED", "1")
     fake = _FakeGovernor(allowed=True, retry_after=None)
@@ -235,46 +233,20 @@ async def test_evaluations_rg_allows_still_enforces_cost_caps(monkeypatch):
         nonlocal usage_denied
         usage_denied = True
         return False, {
-            "error": "Daily cost limit exceeded",
-            "limit": 1.0,
-            "used": 1.0,
-            "requested": 0.5,
+            "error": "Daily evaluation limit exceeded",
+            "limit": 1,
+            "used": 1,
             "retry_after": 3600,
         }
 
     monkeypatch.setattr(limiter, "_reserve_request_usage", _deny_usage)
 
     allowed, meta = await limiter.check_rate_limit(
-        user_id="user-cost-test",
+        user_id="user-cap-test",
         endpoint="/api/v1/evaluations",
         tokens_requested=0,
-        estimated_cost=0.5,
     )
 
     assert usage_denied is True
     assert allowed is False
-    assert "cost" in meta.get("error", "").lower()
-
-
-@pytest.mark.asyncio
-async def test_evaluations_rg_allows_with_zero_cost_skips_cost_check(monkeypatch):
-    """
-    Unified eval endpoints pass estimated_cost=0.0. When RG allows and
-    cost is zero, the cost check should pass (no denial).
-    """
-    monkeypatch.setenv("RG_ENABLED", "1")
-    fake = _FakeGovernor(allowed=True, retry_after=None)
-    monkeypatch.setattr(evals_rl, "_rg_evals_governor", fake)
-    monkeypatch.setattr(evals_rl, "_rg_evals_loader", None)
-
-    limiter = evals_rl.UserRateLimiter()
-
-    allowed, meta = await limiter.check_rate_limit(
-        user_id="user-zero-cost",
-        endpoint="/api/v1/evaluations",
-        tokens_requested=100,
-        estimated_cost=0.0,
-    )
-
-    assert allowed is True
-    assert meta.get("rate_limit_source") == "resource_governor"
+    assert meta.get("error") == "Daily evaluation limit exceeded"
diff --git a/tldw_Server_API/tests/Usage/test_quota_evals_chatbooks.py b/tldw_Server_API/tests/Usage/test_quota_evals_chatbooks.py
index e765d0a672..43c8b54d59 100644
--- a/tldw_Server_API/tests/Usage/test_quota_evals_chatbooks.py
+++ b/tldw_Server_API/tests/Usage/test_quota_evals_chatbooks.py
@@ -18,37 +18,33 @@ def _limiter(tmp_path) -> UserRateLimiter:
 async def test_daily_evaluation_cap_is_atomic_with_the_reservation(tmp_path) -> None:
     """The Nth+1 evaluation of the day is refused inside the same transaction that counts it."""
     limiter = _limiter(tmp_path)
-    config = await limiter._get_user_config("7")
     for _ in range(2):
-        ok, _meta = await limiter._reserve_request_usage("7", "/e", 0, 0.0, config, max_evaluations_per_day=2)
+        ok, _meta = await limiter._reserve_request_usage("7", "/e", 0, max_evaluations_per_day=2)
         assert ok is True
-    ok, meta = await limiter._reserve_request_usage("7", "/e", 0, 0.0, config, max_evaluations_per_day=2)
+    ok, meta = await limiter._reserve_request_usage("7", "/e", 0, max_evaluations_per_day=2)
     assert ok is False and meta["error"] == "Daily evaluation limit exceeded" and meta["limit"] == 2
 
 
 async def test_daily_evaluation_cap_of_zero_refuses_the_first_evaluation(tmp_path) -> None:
     """max_evaluations_per_day == 0 must block, not mean unlimited (spec 2 review B2)."""
     limiter = _limiter(tmp_path)
-    config = await limiter._get_user_config("7")
-    ok, meta = await limiter._reserve_request_usage("7", "/e", 0, 0.0, config, max_evaluations_per_day=0)
+    ok, meta = await limiter._reserve_request_usage("7", "/e", 0, max_evaluations_per_day=0)
     assert ok is False and meta["error"] == "Daily evaluation limit exceeded" and meta["limit"] == 0
 
 
 async def test_daily_evaluation_token_cap(tmp_path) -> None:
     """A request whose tokens would pass the daily token cap is refused."""
     limiter = _limiter(tmp_path)
-    config = await limiter._get_user_config("7")
-    assert (await limiter._reserve_request_usage("7", "/e", 900, 0.0, config, max_tokens_per_day=1000))[0] is True
-    ok, meta = await limiter._reserve_request_usage("7", "/e", 101, 0.0, config, max_tokens_per_day=1000)
+    assert (await limiter._reserve_request_usage("7", "/e", 900, max_tokens_per_day=1000))[0] is True
+    ok, meta = await limiter._reserve_request_usage("7", "/e", 101, max_tokens_per_day=1000)
     assert ok is False and meta["error"] == "Daily evaluation token limit exceeded"
 
 
 async def test_batch_of_four_is_refused_against_a_cap_of_three(tmp_path) -> None:
     """A batch reserving 4 evaluations against a cap of 3 is refused (Qodo Q1)."""
     limiter = _limiter(tmp_path)
-    config = await limiter._get_user_config("7")
     ok, meta = await limiter._reserve_request_usage(
-        "7", "/e", 0, 0.0, config, max_evaluations_per_day=3, evaluations_requested=4
+        "7", "/e", 0, max_evaluations_per_day=3, evaluations_requested=4
     )
     assert ok is False and meta["error"] == "Daily evaluation limit exceeded" and meta["limit"] == 3
 
@@ -56,9 +52,8 @@ async def test_batch_of_four_is_refused_against_a_cap_of_three(tmp_path) -> None
 async def test_batch_of_three_is_admitted_and_records_three(tmp_path) -> None:
     """A batch reserving 3 evaluations against a cap of 3 is admitted and counts as 3 (Qodo Q1)."""
     limiter = _limiter(tmp_path)
-    config = await limiter._get_user_config("7")
     ok, _meta = await limiter._reserve_request_usage(
-        "7", "/e", 0, 0.0, config, max_evaluations_per_day=3, evaluations_requested=3
+        "7", "/e", 0, max_evaluations_per_day=3, evaluations_requested=3
     )
     assert ok is True
     summary = await limiter.get_usage_summary("7")
```

### A.2 `pr3-src.patch`

```diff
diff --git a/tldw_Server_API/Config_Files/evaluations_config.yaml b/tldw_Server_API/Config_Files/evaluations_config.yaml
index 50d8e4ca9d..a6462ae845 100644
--- a/tldw_Server_API/Config_Files/evaluations_config.yaml
+++ b/tldw_Server_API/Config_Files/evaluations_config.yaml
@@ -12,8 +12,6 @@ rate_limiting:
       evaluations_per_day: 100
       total_tokens_per_day: 100000
       burst_size: 5
-      max_cost_per_day: 1.0
-      max_cost_per_month: 10.0
       description: "Free tier with basic usage limits"
 
     basic:
@@ -22,8 +20,6 @@ rate_limiting:
       evaluations_per_day: 1000
       total_tokens_per_day: 1000000
       burst_size: 10
-      max_cost_per_day: 10.0
-      max_cost_per_month: 100.0
       description: "Basic paid tier with increased limits"
 
     premium:
@@ -32,8 +28,6 @@ rate_limiting:
       evaluations_per_day: 10000
       total_tokens_per_day: 10000000
       burst_size: 25
-      max_cost_per_day: 100.0
-      max_cost_per_month: 1000.0
       description: "Premium tier for power users"
 
     enterprise:
@@ -42,8 +36,6 @@ rate_limiting:
       evaluations_per_day: 100000
       total_tokens_per_day: 100000000
       burst_size: 100
-      max_cost_per_day: 1000.0
-      max_cost_per_month: 10000.0
       description: "Enterprise tier with high limits"
 
     custom:
@@ -52,8 +44,6 @@ rate_limiting:
       evaluations_per_day: 100
       total_tokens_per_day: 100000
       burst_size: 5
-      max_cost_per_day: 1.0
-      max_cost_per_month: 10.0
       description: "Custom tier (configured per user)"
 
   # Global rate limiting settings
diff --git a/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py b/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py
index 8aacb2cad7..8e8d8a1149 100644
--- a/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py
+++ b/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py
@@ -368,8 +368,6 @@ async def _apply_rate_limit_headers(limiter, user_id: str, response: Response, m
         ):
             if value is not None:
                 response.headers[header] = str(value)
-        response.headers["X-RateLimit-Daily-Cost-Remaining"] = f"{remaining.get('daily_cost', 0):.2f}"
-        response.headers["X-RateLimit-Monthly-Cost-Remaining"] = f"{remaining.get('monthly_cost', 0):.2f}"
         response.headers["RateLimit-Limit"] = str(per_min_limit)
         if isinstance(meta, dict) and "requests_remaining" in meta:
             response.headers["RateLimit-Remaining"] = str(int(meta.get("requests_remaining") or 0))
diff --git a/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py b/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py
index 0ce1b760d1..70bfa58cf9 100644
--- a/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py
+++ b/tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py
@@ -853,21 +853,15 @@ async def get_rate_limit_status(
             limits={
                 "evaluations_per_minute": summary.get("limits", {}).get("per_minute", {}).get("evaluations", 0),
                 "evaluations_per_day": summary.get("limits", {}).get("daily", {}).get("evaluations"),
-                "tokens_per_day": summary.get("limits", {}).get("daily", {}).get("tokens"),
-                "cost_per_day": int(summary.get("limits", {}).get("daily", {}).get("cost", 0)),
-                "cost_per_month": int(summary.get("limits", {}).get("monthly", {}).get("cost", 0))
+                "tokens_per_day": summary.get("limits", {}).get("daily", {}).get("tokens")
             },
             usage={
                 "evaluations_today": summary.get("usage", {}).get("today", {}).get("evaluations", 0),
-                "tokens_today": summary.get("usage", {}).get("today", {}).get("tokens", 0),
-                "cost_today": int(summary.get("usage", {}).get("today", {}).get("cost", 0)),
-                "cost_month": int(summary.get("usage", {}).get("month", {}).get("cost", 0))
+                "tokens_today": summary.get("usage", {}).get("today", {}).get("tokens", 0)
             },
             remaining={
                 "daily_evaluations": summary.get("remaining", {}).get("daily_evaluations"),
-                "daily_tokens": summary.get("remaining", {}).get("daily_tokens"),
-                "daily_cost": int(summary.get("remaining", {}).get("daily_cost", 0)),
-                "monthly_cost": int(summary.get("remaining", {}).get("monthly_cost", 0))
+                "daily_tokens": summary.get("remaining", {}).get("daily_tokens")
             },
             reset_at=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
         )
@@ -1005,7 +999,7 @@ async def evaluate_geval(
             provider=getattr(request, "api_name", None),
             model=request.model,
         )
-        allowed, meta = await limiter.check_rate_limit(str(current_user.id), endpoint="evals:geval", is_batch=False, tokens_requested=tokens_est, estimated_cost=0.0)
+        allowed, meta = await limiter.check_rate_limit(str(current_user.id), endpoint="evals:geval", is_batch=False, tokens_requested=tokens_est)
         if not allowed:
             retry_after = meta.get("retry_after", 60)
             raise HTTPException(
@@ -1070,7 +1064,7 @@ async def evaluate_geval(
         try:
             usage = result.get("usage") if isinstance(result, dict) else None
             if usage and isinstance(usage, dict):
-                await limiter.record_actual_usage(str(current_user.id), "evals:geval", int(usage.get("total_tokens", 0)), float(usage.get("cost", 0.0) or 0.0), reserved_tokens=tokens_est)
+                await limiter.record_actual_usage(str(current_user.id), "evals:geval", int(usage.get("total_tokens", 0)), reserved_tokens=tokens_est)
         except asyncio.CancelledError:
             raise
         except _EVALS_NONCRITICAL_EXCEPTIONS:
@@ -1224,7 +1218,7 @@ async def evaluate_rag(
             provider=getattr(request, "api_name", None),
             model=request.model,
         )
-        allowed, meta = await limiter.check_rate_limit(str(current_user.id), endpoint="evals:rag", is_batch=False, tokens_requested=tokens_est, estimated_cost=0.0)
+        allowed, meta = await limiter.check_rate_limit(str(current_user.id), endpoint="evals:rag", is_batch=False, tokens_requested=tokens_est)
         if not allowed:
             retry_after = meta.get("retry_after", 60)
             raise HTTPException(
@@ -1290,7 +1284,7 @@ async def evaluate_rag(
         try:
             usage = result.get("usage") if isinstance(result, dict) else None
             if usage and isinstance(usage, dict):
-                await limiter.record_actual_usage(str(current_user.id), "evals:rag", int(usage.get("total_tokens", 0)), float(usage.get("cost", 0.0) or 0.0), reserved_tokens=tokens_est)
+                await limiter.record_actual_usage(str(current_user.id), "evals:rag", int(usage.get("total_tokens", 0)), reserved_tokens=tokens_est)
         except asyncio.CancelledError:
             raise
         except _EVALS_NONCRITICAL_EXCEPTIONS:
@@ -1421,7 +1415,7 @@ async def evaluate_response_quality(
             provider=getattr(request, "api_name", None),
             model=request.model,
         )
-        allowed, meta = await limiter.check_rate_limit(str(current_user.id), endpoint="evals:response_quality", is_batch=False, tokens_requested=tokens_est, estimated_cost=0.0)
+        allowed, meta = await limiter.check_rate_limit(str(current_user.id), endpoint="evals:response_quality", is_batch=False, tokens_requested=tokens_est)
         if not allowed:
             retry_after = meta.get("retry_after", 60)
             raise HTTPException(
@@ -1487,7 +1481,7 @@ async def evaluate_response_quality(
         try:
             usage = result.get("usage") if isinstance(result, dict) else None
             if usage and isinstance(usage, dict):
-                await limiter.record_actual_usage(str(current_user.id), "evals:response_quality", int(usage.get("total_tokens", 0)), float(usage.get("cost", 0.0) or 0.0), reserved_tokens=tokens_est)
+                await limiter.record_actual_usage(str(current_user.id), "evals:response_quality", int(usage.get("total_tokens", 0)), reserved_tokens=tokens_est)
         except asyncio.CancelledError:
             raise
         except _EVALS_NONCRITICAL_EXCEPTIONS:
@@ -1620,7 +1614,6 @@ async def evaluate_propositions_endpoint(
             endpoint="evals:propositions",
             is_batch=False,
             tokens_requested=tokens_est,
-            estimated_cost=0.0,
         )
         if not allowed:
             retry_after = meta.get("retry_after", 60)
@@ -1790,7 +1783,6 @@ async def batch_evaluate(
             endpoint=f"evals:batch:{etype}",
             is_batch=True,
             tokens_requested=tokens_total,
-            estimated_cost=0.0,
             evaluations_requested=max(1, len(request.items or [])),
         )
         if not allowed:
@@ -2266,7 +2258,7 @@ async def evaluate_ocr_endpoint(
         except _EVALS_NONCRITICAL_EXCEPTIONS:
             pass
         tokens_est = _estimate_tokens_from_texts("\n".join(texts))
-        allowed, meta = await limiter.check_rate_limit(str(current_user.id), endpoint="evals:ocr", is_batch=False, tokens_requested=tokens_est, estimated_cost=0.0)
+        allowed, meta = await limiter.check_rate_limit(str(current_user.id), endpoint="evals:ocr", is_batch=False, tokens_requested=tokens_est)
         if not allowed:
             retry_after = meta.get("retry_after", 60)
             raise HTTPException(status_code=429, detail=meta.get("error", "Rate limit exceeded"), headers={"Retry-After": str(retry_after)})
@@ -2283,7 +2275,7 @@ async def evaluate_ocr_endpoint(
         try:
             usage = result.get("usage") if isinstance(result, dict) else None
             if usage and isinstance(usage, dict):
-                await limiter.record_actual_usage(str(current_user.id), "evals:ocr", int(usage.get("total_tokens", 0)), float(usage.get("cost", 0.0) or 0.0), reserved_tokens=tokens_est)
+                await limiter.record_actual_usage(str(current_user.id), "evals:ocr", int(usage.get("total_tokens", 0)), reserved_tokens=tokens_est)
         except _EVALS_NONCRITICAL_EXCEPTIONS:
             pass
         # Apply headers
@@ -2334,7 +2326,7 @@ async def evaluate_ocr_pdf_endpoint(
         except _EVALS_NONCRITICAL_EXCEPTIONS:
             pass
         tokens_est = max(0, size_est // 4)
-        allowed, meta = await limiter.check_rate_limit(str(current_user.id), endpoint="evals:ocr_pdf", is_batch=False, tokens_requested=tokens_est, estimated_cost=0.0)
+        allowed, meta = await limiter.check_rate_limit(str(current_user.id), endpoint="evals:ocr_pdf", is_batch=False, tokens_requested=tokens_est)
         if not allowed:
             retry_after = meta.get("retry_after", 60)
             raise HTTPException(status_code=429, detail=meta.get("error", "Rate limit exceeded"), headers={"Retry-After": str(retry_after)})
@@ -2410,7 +2402,7 @@ async def evaluate_ocr_pdf_endpoint(
         try:
             usage = result.get("usage") if isinstance(result, dict) else None
             if usage and isinstance(usage, dict):
-                await limiter.record_actual_usage(str(current_user.id), "evals:ocr_pdf", int(usage.get("total_tokens", 0)), float(usage.get("cost", 0.0) or 0.0), reserved_tokens=tokens_est)
+                await limiter.record_actual_usage(str(current_user.id), "evals:ocr_pdf", int(usage.get("total_tokens", 0)), reserved_tokens=tokens_est)
         except _EVALS_NONCRITICAL_EXCEPTIONS:
             pass
         try:
diff --git a/tldw_Server_API/app/api/v1/schemas/webhook_schemas.py b/tldw_Server_API/app/api/v1/schemas/webhook_schemas.py
index 052624c8dd..53f9e469fe 100644
--- a/tldw_Server_API/app/api/v1/schemas/webhook_schemas.py
+++ b/tldw_Server_API/app/api/v1/schemas/webhook_schemas.py
@@ -204,25 +204,18 @@ class RateLimitStatusResponse(BaseModel):
                 },
                 "daily": {
                     "evaluations": 10000,
-                    "tokens": 10000000,
-                    "cost": 100.0
+                    "tokens": 10000000
                 }
             },
             "usage": {
                 "today": {
                     "evaluations": 150,
-                    "tokens": 50000,
-                    "cost": 1.50
-                },
-                "month": {
-                    "cost": 45.00
+                    "tokens": 50000
                 }
             },
             "remaining": {
                 "daily_evaluations": 9850,
-                "daily_tokens": 9950000,
-                "daily_cost": 98.50,
-                "monthly_cost": 955.00
+                "daily_tokens": 9950000
             }
         }
     })
diff --git a/tldw_Server_API/app/core/Evaluations/config_manager.py b/tldw_Server_API/app/core/Evaluations/config_manager.py
index 9f5501d1a5..c509b73464 100644
--- a/tldw_Server_API/app/core/Evaluations/config_manager.py
+++ b/tldw_Server_API/app/core/Evaluations/config_manager.py
@@ -47,8 +47,6 @@ class RateLimitTierConfig:
     evaluations_per_day: int
     total_tokens_per_day: int
     burst_size: int
-    max_cost_per_day: float
-    max_cost_per_month: float
     description: str = ""
 
     @classmethod
@@ -61,8 +59,6 @@ class RateLimitTierConfig:
             evaluations_per_day=data.get("evaluations_per_day", 100),
             total_tokens_per_day=data.get("total_tokens_per_day", 100000),
             burst_size=data.get("burst_size", 5),
-            max_cost_per_day=data.get("max_cost_per_day", 1.0),
-            max_cost_per_month=data.get("max_cost_per_month", 10.0),
             description=data.get("description", "")
         )
 
@@ -268,6 +264,12 @@ class EvaluationsConfigManager:
             try:
                 tier_config = RateLimitTierConfig.from_dict(tier_name, tier_data)
                 self._rate_limit_tiers[tier_name] = tier_config
+                removed = [key for key in ("max_cost_per_day", "max_cost_per_month") if key in tier_data]
+                if removed:
+                    logger.warning(
+                        f"Ignoring {', '.join(removed)} in rate limit tier '{tier_name}': evaluation cost caps were "
+                        "removed; cap LLM spend with limits.llm_cost_per_day_usd / limits.llm_cost_per_month_usd"
+                    )
             except _EVAL_CONFIG_NONCRITICAL_EXCEPTIONS as e:
                 logger.error(f"Invalid rate limit config for tier '{tier_name}': {e}")
 
@@ -486,7 +488,7 @@ class EvaluationsConfigManager:
 
                         required_fields = [
                             "evaluations_per_minute", "evaluations_per_day",
-                            "total_tokens_per_day", "max_cost_per_day"
+                            "total_tokens_per_day"
                         ]
 
                         for field in required_fields:
diff --git a/tldw_Server_API/app/core/Evaluations/user_rate_limiter.py b/tldw_Server_API/app/core/Evaluations/user_rate_limiter.py
index 132401974b..cf339211ce 100644
--- a/tldw_Server_API/app/core/Evaluations/user_rate_limiter.py
+++ b/tldw_Server_API/app/core/Evaluations/user_rate_limiter.py
@@ -4,11 +4,13 @@ Per-user rate limiting for Evaluations module.
 **Phase 2 Deprecation Notice**:
 - When ``RG_ENABLED=true``, evaluations/minute and evaluations/tokens daily
   caps are enforced by the ResourceGovernor via ``_maybe_enforce_with_rg_evaluations``.
-- Cost caps (``max_cost_per_day``, ``max_cost_per_month``) remain **legacy-only**
-  and are enforced regardless of RG status.
-- When RG is disabled or unavailable, minute/daily eval/token checks are
-  skipped (fail-open with deprecation warning). Only cost caps are enforced.
-- CUSTOM tier bypasses RG; only cost caps are enforced with a deprecation warning.
+- Evaluation cost caps were removed; LLM spend is capped by the general
+  ``limits.llm_cost_per_day_usd`` / ``limits.llm_cost_per_month_usd`` quotas.
+- When RG is disabled or unavailable, the per-minute checks are skipped
+  (fail-open with deprecation warning). The per-user daily evaluations/tokens
+  quotas (``limits.evaluations_per_day``, ``limits.evaluation_tokens_per_day``)
+  are still enforced locally by ``_reserve_request_usage``.
+- CUSTOM tier bypasses RG with a deprecation warning.
 - This shim will be removed in a future release.
 """
 
@@ -123,8 +125,6 @@ class RateLimitConfig:
     evaluations_per_day: int
     total_tokens_per_day: int
     burst_size: int
-    max_cost_per_day: float
-    max_cost_per_month: float
 
     @classmethod
     def for_tier(cls, tier: UserTier) -> "RateLimitConfig":
@@ -140,8 +140,6 @@ class RateLimitConfig:
                 evaluations_per_day=tier_config.evaluations_per_day,
                 total_tokens_per_day=tier_config.total_tokens_per_day,
                 burst_size=tier_config.burst_size,
-                max_cost_per_day=tier_config.max_cost_per_day,
-                max_cost_per_month=tier_config.max_cost_per_month
             )
 
         # Fallback to hardcoded defaults if external config not available
@@ -153,8 +151,6 @@ class RateLimitConfig:
                 evaluations_per_day=100,
                 total_tokens_per_day=100_000,
                 burst_size=5,
-                max_cost_per_day=1.0,
-                max_cost_per_month=10.0
             ),
             UserTier.BASIC: cls(
                 tier=UserTier.BASIC,
@@ -163,8 +159,6 @@ class RateLimitConfig:
                 evaluations_per_day=1000,
                 total_tokens_per_day=1_000_000,
                 burst_size=10,
-                max_cost_per_day=10.0,
-                max_cost_per_month=100.0
             ),
             UserTier.PREMIUM: cls(
                 tier=UserTier.PREMIUM,
@@ -173,8 +167,6 @@ class RateLimitConfig:
                 evaluations_per_day=10000,
                 total_tokens_per_day=10_000_000,
                 burst_size=25,
-                max_cost_per_day=100.0,
-                max_cost_per_month=1000.0
             ),
             UserTier.ENTERPRISE: cls(
                 tier=UserTier.ENTERPRISE,
@@ -183,8 +175,6 @@ class RateLimitConfig:
                 evaluations_per_day=100000,
                 total_tokens_per_day=100_000_000,
                 burst_size=100,
-                max_cost_per_day=1000.0,
-                max_cost_per_month=10000.0
             ),
             UserTier.CUSTOM: cls(
                 tier=UserTier.CUSTOM,
@@ -193,8 +183,6 @@ class RateLimitConfig:
                 evaluations_per_day=100,
                 total_tokens_per_day=100_000,
                 burst_size=5,
-                max_cost_per_day=1.0,
-                max_cost_per_month=10.0
             )
         }
 
@@ -343,7 +331,6 @@ class UserRateLimiter:
         endpoint: str,
         is_batch: bool = False,
         tokens_requested: int = 0,
-        estimated_cost: float = 0.0,
         evaluations_requested: int = 1,
     ) -> tuple[bool, dict[str, Any]]:
         """
@@ -354,7 +341,6 @@ class UserRateLimiter:
             endpoint: API endpoint being accessed
             is_batch: Whether this is a batch evaluation
             tokens_requested: Estimated tokens for request
-            estimated_cost: Estimated cost for request
             evaluations_requested: How many evaluation units this call represents
                 against the daily evaluations cap (e.g. item count for a batch)
 
@@ -378,7 +364,6 @@ class UserRateLimiter:
                 endpoint=endpoint,
                 is_batch=is_batch,
                 tokens_requested=tokens_requested,
-                estimated_cost=estimated_cost,
                 policy_id=policy_id,
             )
             if rg_decision is not None:
@@ -391,14 +376,12 @@ class UserRateLimiter:
                         "rate_limit_source": "resource_governor",
                     }
 
-                # Enforce cost caps locally (RG does not handle cost limits)
-                # and reserve the estimated request usage atomically.
+                # Enforce the daily caps locally and reserve the estimated
+                # request usage atomically.
                 usage_ok, usage_meta = await self._reserve_request_usage(
                     user_id,
                     endpoint,
                     tokens_requested,
-                    estimated_cost,
-                    config,
                     **daily_caps,
                     evaluations_requested=evaluations_requested,
                 )
@@ -414,13 +397,11 @@ class UserRateLimiter:
 
             _log_rg_evals_fallback("rg_decision_unavailable")
             _emit_evals_legacy_deprecation("rg_decision_unavailable")
-            # Fail-open for eval/token caps; still enforce cost.
+            # Fail-open for RG's eval/token caps; still enforce the local daily caps.
             usage_ok, usage_meta = await self._reserve_request_usage(
                 user_id,
                 endpoint,
                 tokens_requested,
-                estimated_cost,
-                config,
                 **daily_caps,
                 evaluations_requested=evaluations_requested,
             )
@@ -432,22 +413,20 @@ class UserRateLimiter:
                 **usage_meta,
             }
 
-        # RG disabled or CUSTOM tier → skip minute/daily checks (Phase 2),
-        # enforce cost caps only.
+        # RG disabled or CUSTOM tier → skip the minute checks (Phase 2);
+        # the local daily caps are still enforced.
         _emit_evals_legacy_deprecation("rg_disabled_or_custom_tier")
         usage_ok, usage_meta = await self._reserve_request_usage(
             user_id,
             endpoint,
             tokens_requested,
-            estimated_cost,
-            config,
             **daily_caps,
             evaluations_requested=evaluations_requested,
         )
         if not usage_ok:
             return False, usage_meta
         return True, {
-            "rate_limit_source": "legacy_cost_only",
+            "rate_limit_source": "legacy_local",
             "tier": config.tier.value,
             **usage_meta,
         }
@@ -467,7 +446,7 @@ class UserRateLimiter:
             cursor.execute("""
                 SELECT tier, evaluations_per_minute, batch_evaluations_per_minute,
                        evaluations_per_day, total_tokens_per_day, burst_size,
-                       max_cost_per_day, max_cost_per_month, expires_at
+                       expires_at
                 FROM user_rate_limits
                 WHERE user_id = ?
             """, (user_id,))
@@ -477,9 +456,9 @@ class UserRateLimiter:
             if row:
                 # Check if custom limits have expired (tolerate naive or aware timestamps)
                 expired = False
-                if row[8]:
+                if row[6]:
                     try:
-                        _exp_raw = row[8]
+                        _exp_raw = row[6]
                         exp_dt = datetime.fromisoformat(_exp_raw)
                         if exp_dt.tzinfo is None:
                             # Compare naive timestamps in UTC
@@ -509,8 +488,6 @@ class UserRateLimiter:
                         evaluations_per_day=row[3],
                         total_tokens_per_day=row[4],
                         burst_size=row[5],
-                        max_cost_per_day=row[6],
-                        max_cost_per_month=row[7]
                     )
             else:
                 # Create default configuration for new user.
@@ -525,22 +502,19 @@ class UserRateLimiter:
                 cursor.execute("""
                     INSERT INTO user_rate_limits (
                         user_id, tier, evaluations_per_minute, batch_evaluations_per_minute,
-                        evaluations_per_day, total_tokens_per_day, burst_size,
-                        max_cost_per_day, max_cost_per_month
-                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
+                        evaluations_per_day, total_tokens_per_day, burst_size
+                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                     ON CONFLICT(user_id) DO NOTHING
                 """, (
                     user_id, config.tier.value, config.evaluations_per_minute,
                     config.batch_evaluations_per_minute, config.evaluations_per_day,
                     config.total_tokens_per_day, config.burst_size,
-                    config.max_cost_per_day, config.max_cost_per_month
                 ))
                 conn.commit()
 
                 cursor.execute("""
                     SELECT tier, evaluations_per_minute, batch_evaluations_per_minute,
-                           evaluations_per_day, total_tokens_per_day, burst_size,
-                           max_cost_per_day, max_cost_per_month
+                           evaluations_per_day, total_tokens_per_day, burst_size
                     FROM user_rate_limits
                     WHERE user_id = ?
                 """, (user_id,))
@@ -553,8 +527,6 @@ class UserRateLimiter:
                         evaluations_per_day=persisted[3],
                         total_tokens_per_day=persisted[4],
                         burst_size=persisted[5],
-                        max_cost_per_day=persisted[6],
-                        max_cost_per_month=persisted[7],
                     )
             return config
 
@@ -652,69 +624,21 @@ class UserRateLimiter:
         except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
             logger.debug(f"Evaluations: legacy daily_usage backfill skipped: {exc}")
 
-    async def _check_cost_limits(
-        self,
-        user_id: str,
-        estimated_cost: float,
-        config: RateLimitConfig,
-    ) -> tuple[bool, dict[str, Any]]:
-        """Check daily cost limits only (RG handles evaluations/tokens)."""
-        try:
-            cost = float(estimated_cost or 0.0)
-        except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
-            cost = 0.0
-        if cost <= 0:
-            return True, {}
-
-        today = datetime.now(timezone.utc).date()
-        day_str = str(today)
-        row = await self._run_db(
-            lambda conn: conn.execute(
-                "SELECT total_cost FROM daily_usage WHERE user_id = ? AND date = ?",
-                (user_id, day_str),
-            ).fetchone()
-        )
-        total_cost = float(row[0] or 0.0) if row else 0.0
-
-        if total_cost + cost > float(config.max_cost_per_day):
-            reset_at = datetime.combine(
-                today + timedelta(days=1),
-                datetime.min.time(),
-                tzinfo=timezone.utc,
-            )
-            retry_after = max(1, int((reset_at - datetime.now(timezone.utc)).total_seconds()))
-            return False, {
-                "error": "Daily cost limit exceeded",
-                "limit": config.max_cost_per_day,
-                "used": total_cost,
-                "requested": cost,
-                "retry_after": retry_after,
-                "resets_at": reset_at.isoformat(),
-            }
-
-        return True, {
-            "cost_remaining": float(config.max_cost_per_day) - total_cost - cost,
-        }
-
     async def _reserve_request_usage(
         self,
         user_id: str,
         endpoint: str,
         tokens_used: int,
-        cost: float,
-        config: RateLimitConfig,
         max_evaluations_per_day: float | None = None,
         max_tokens_per_day: float | None = None,
         evaluations_requested: int = 1,
     ) -> tuple[bool, dict[str, Any]]:
-        """Atomically enforce cost caps and record estimated request usage."""
+        """Atomically enforce the daily caps and record estimated request usage."""
         allowed, metadata, now, normalized_tokens = await asyncio.to_thread(
             self._reserve_request_usage_sync,
             user_id,
             endpoint,
             tokens_used,
-            cost,
-            config,
             max_evaluations_per_day,
             max_tokens_per_day,
             evaluations_requested,
@@ -735,20 +659,13 @@ class UserRateLimiter:
         user_id: str,
         endpoint: str,
         tokens_used: int,
-        cost: float,
-        config: RateLimitConfig,
         max_evaluations_per_day: float | None = None,
         max_tokens_per_day: float | None = None,
         evaluations_requested: int = 1,
     ) -> tuple[bool, dict[str, Any], datetime, int]:
         now = datetime.now(timezone.utc)
         today = now.date()
-        try:
-            normalized_cost = float(cost or 0.0)
-        except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
-            normalized_cost = 0.0
         normalized_tokens = max(0, int(tokens_used or 0))
-        cost_remaining: Optional[float] = None
 
         # Already dispatched to a worker thread by _reserve_request_usage();
         # keep it synchronous and only take the connection policy from _connect().
@@ -781,33 +698,6 @@ class UserRateLimiter:
                             "resets_at": reset_at.isoformat(),
                         }, now, normalized_tokens
 
-                if normalized_cost > 0:
-                    cursor = conn.cursor()
-                    cursor.execute(
-                        "SELECT total_cost FROM daily_usage WHERE user_id = ? AND date = ?",
-                        (user_id, str(today)),
-                    )
-                    row = cursor.fetchone()
-                    total_cost = float(row[0] or 0.0) if row else 0.0
-                    daily_limit = float(config.max_cost_per_day)
-                    if total_cost + normalized_cost > daily_limit:
-                        conn.rollback()
-                        reset_at = datetime.combine(
-                            today + timedelta(days=1),
-                            datetime.min.time(),
-                            tzinfo=timezone.utc,
-                        )
-                        retry_after = max(1, int((reset_at - datetime.now(timezone.utc)).total_seconds()))
-                        return False, {
-                            "error": "Daily cost limit exceeded",
-                            "limit": config.max_cost_per_day,
-                            "used": total_cost,
-                            "requested": normalized_cost,
-                            "retry_after": retry_after,
-                            "resets_at": reset_at.isoformat(),
-                        }, now, normalized_tokens
-                    cost_remaining = daily_limit - total_cost - normalized_cost
-
                 self._write_request_usage(
                     conn,
                     user_id=user_id,
@@ -815,7 +705,6 @@ class UserRateLimiter:
                     now=now,
                     today=today,
                     tokens_used=normalized_tokens,
-                    cost=normalized_cost,
                     evaluations_delta=evaluations_requested,
                 )
                 conn.commit()
@@ -825,16 +714,13 @@ class UserRateLimiter:
         finally:
             conn.close()
 
-        if cost_remaining is None:
-            return True, {}, now, normalized_tokens
-        return True, {"cost_remaining": cost_remaining}, now, normalized_tokens
+        return True, {}, now, normalized_tokens
 
     async def _record_request(
         self,
         user_id: str,
         endpoint: str,
         tokens_used: int,
-        cost: float,
         reserved_tokens: int = 0,
     ):
         """Record a request's actual usage, replacing any tokens already reserved."""
@@ -843,7 +729,6 @@ class UserRateLimiter:
             user_id,
             endpoint,
             tokens_used,
-            cost,
             reserved_tokens,
         )
         await self._shadow_write_daily_usage(
@@ -858,13 +743,11 @@ class UserRateLimiter:
         user_id: str,
         endpoint: str,
         tokens_used: int,
-        cost: float,
         reserved_tokens: int = 0,
     ) -> tuple[datetime, int]:
         now = datetime.now(timezone.utc)
         today = now.date()
         normalized_tokens = max(0, int(tokens_used or 0))
-        normalized_cost = float(cost or 0.0)
         # This call site already reserved an estimate via check_rate_limit()'s
         # _reserve_request_usage(), which wrote +1 evaluation and the estimate's
         # tokens. Replace the estimate with the actual token count (may be
@@ -882,7 +765,6 @@ class UserRateLimiter:
                 now=now,
                 today=today,
                 tokens_used=token_delta,
-                cost=normalized_cost,
                 evaluations_delta=0,
             )
             conn.commit()
@@ -899,7 +781,6 @@ class UserRateLimiter:
         now: datetime,
         today: Any,
         tokens_used: int,
-        cost: float,
         evaluations_delta: int = 1,
     ) -> None:
         """Record one request's usage; `tokens_used`/`evaluations_delta` are deltas, not totals.
@@ -909,19 +790,18 @@ class UserRateLimiter:
         """
         # Record in tracking table
         conn.execute(
-            "INSERT INTO rate_limit_tracking (user_id, endpoint, timestamp, tokens_used, cost) VALUES (?, ?, ?, ?, ?)",
-            (user_id, endpoint, now.isoformat(), tokens_used, cost)
+            "INSERT INTO rate_limit_tracking (user_id, endpoint, timestamp, tokens_used) VALUES (?, ?, ?, ?)",
+            (user_id, endpoint, now.isoformat(), tokens_used)
         )
 
         # Update daily usage
         conn.execute("""
-            INSERT INTO daily_usage (user_id, date, total_evaluations, total_tokens, total_cost)
-            VALUES (?, ?, ?, MAX(0, ?), ?)
+            INSERT INTO daily_usage (user_id, date, total_evaluations, total_tokens)
+            VALUES (?, ?, ?, MAX(0, ?))
             ON CONFLICT(user_id, date) DO UPDATE SET
                 total_evaluations = total_evaluations + ?,
-                total_tokens = MAX(0, total_tokens + ?),
-                total_cost = total_cost + ?
-        """, (user_id, str(today), evaluations_delta, tokens_used, cost, evaluations_delta, tokens_used, cost))
+                total_tokens = MAX(0, total_tokens + ?)
+        """, (user_id, str(today), evaluations_delta, tokens_used, evaluations_delta, tokens_used))
 
     async def _shadow_write_daily_usage(
         self,
@@ -983,7 +863,6 @@ class UserRateLimiter:
                 "X-RateLimit-Daily-Limit": str(config.evaluations_per_day),
                 "X-RateLimit-Daily-Remaining": str(daily_metadata.get("evaluations_remaining", 0)),
                 "X-RateLimit-Tokens-Remaining": str(daily_metadata.get("tokens_remaining", 0)),
-                "X-RateLimit-Cost-Remaining": f"{daily_metadata.get('cost_remaining', 0):.2f}"
             },
             "tier": config.tier.value,
             "limits": {
@@ -1022,6 +901,9 @@ class UserRateLimiter:
                 for key, value in custom_limits.items():
                     if hasattr(config, key):
                         setattr(config, key, value)
+                    else:
+                        # e.g. the removed max_cost_per_day / max_cost_per_month
+                        logger.warning(f"Ignoring unknown custom limit {key!r} for user {user_id}")
 
             def _apply(conn: sqlite3.Connection) -> None:
                 cursor = conn.cursor()
@@ -1029,13 +911,11 @@ class UserRateLimiter:
                     UPDATE user_rate_limits
                     SET tier = ?, evaluations_per_minute = ?, batch_evaluations_per_minute = ?,
                         evaluations_per_day = ?, total_tokens_per_day = ?, burst_size = ?,
-                        max_cost_per_day = ?, max_cost_per_month = ?,
                         expires_at = ?, updated_at = CURRENT_TIMESTAMP
                     WHERE user_id = ?
                 """, (
                     new_tier.value, config.evaluations_per_minute, config.batch_evaluations_per_minute,
                     config.evaluations_per_day, config.total_tokens_per_day, config.burst_size,
-                    config.max_cost_per_day, config.max_cost_per_month,
                     expires_at, user_id
                 ))
 
@@ -1045,13 +925,12 @@ class UserRateLimiter:
                         INSERT INTO user_rate_limits (
                             user_id, tier, evaluations_per_minute, batch_evaluations_per_minute,
                             evaluations_per_day, total_tokens_per_day, burst_size,
-                            max_cost_per_day, max_cost_per_month, expires_at
-                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
+                            expires_at
+                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                     """, (
                         user_id, new_tier.value, config.evaluations_per_minute,
                         config.batch_evaluations_per_minute, config.evaluations_per_day,
                         config.total_tokens_per_day, config.burst_size,
-                        config.max_cost_per_day, config.max_cost_per_month,
                         expires_at
                     ))
 
@@ -1089,7 +968,7 @@ class UserRateLimiter:
 
             # Get today's usage
             cursor.execute("""
-                SELECT total_evaluations, total_tokens, total_cost
+                SELECT total_evaluations, total_tokens
                 FROM daily_usage
                 WHERE user_id = ? AND date = ?
             """, (user_id, str(today)))
@@ -1097,22 +976,13 @@ class UserRateLimiter:
             row = cursor.fetchone()
 
             if row:
-                total_evaluations, total_tokens, total_cost = row
+                total_evaluations, total_tokens = row
             else:
-                total_evaluations = total_tokens = total_cost = 0
+                total_evaluations = total_tokens = 0
 
-            # Get monthly cost
-            month_start = today.replace(day=1)
-            cursor.execute("""
-                SELECT SUM(total_cost)
-                FROM daily_usage
-                WHERE user_id = ? AND date >= ?
-            """, (user_id, str(month_start)))
-
-            monthly_cost = cursor.fetchone()[0] or 0.0
-            return total_evaluations, total_tokens, total_cost, monthly_cost
+            return total_evaluations, total_tokens
 
-        total_evaluations, total_tokens, total_cost, monthly_cost = await self._run_db(_summary)
+        total_evaluations, total_tokens = await self._run_db(_summary)
 
         quota_uid = as_quota_user_id(user_id)
         if quota_uid is None:
@@ -1134,28 +1004,18 @@ class UserRateLimiter:
                 },
                 "daily": {
                     "evaluations": evals_cap,
-                    "tokens": tokens_cap,
-                    "cost": config.max_cost_per_day
-                },
-                "monthly": {
-                    "cost": config.max_cost_per_month
+                    "tokens": tokens_cap
                 }
             },
             "usage": {
                 "today": {
                     "evaluations": total_evaluations,
-                    "tokens": total_tokens,
-                    "cost": total_cost
-                },
-                "month": {
-                    "cost": monthly_cost
+                    "tokens": total_tokens
                 }
             },
             "remaining": {
                 "daily_evaluations": None if evals_cap is None else max(0, evals_cap - total_evaluations),
-                "daily_tokens": None if tokens_cap is None else max(0, tokens_cap - total_tokens),
-                "daily_cost": config.max_cost_per_day - total_cost,
-                "monthly_cost": config.max_cost_per_month - monthly_cost
+                "daily_tokens": None if tokens_cap is None else max(0, tokens_cap - total_tokens)
             }
         }
 
@@ -1164,7 +1024,6 @@ class UserRateLimiter:
         user_id: str,
         endpoint: str,
         tokens_used: int,
-        cost: float = 0.0,
         reserved_tokens: int = 0,
     ) -> None:
         """Record actual usage after a request completes (if provider returns usage).
@@ -1178,7 +1037,6 @@ class UserRateLimiter:
                 user_id,
                 endpoint,
                 max(0, int(tokens_used or 0)),
-                float(cost or 0.0),
                 reserved_tokens=max(0, int(reserved_tokens or 0)),
             )
         except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
@@ -1372,7 +1230,6 @@ async def _maybe_enforce_with_rg_evaluations(
     endpoint: str,
     is_batch: bool,
     tokens_requested: int,
-    estimated_cost: float,
     policy_id: str,
 ) -> Optional[dict[str, object]]:
     """
@@ -1395,9 +1252,6 @@ async def _maybe_enforce_with_rg_evaluations(
             tu = 0
         if tu > 0:
             categories["tokens"] = {"units": tu}
-        # Cost/day caps remain legacy-only (and unified eval endpoints currently
-        # pass estimated_cost=0.0). RG-first enforcement covers evaluations/tokens;
-        # keep estimated_cost in tags for observability only.
 
         decision, handle = await gov.reserve(
             RGRequest(
@@ -1409,7 +1263,6 @@ async def _maybe_enforce_with_rg_evaluations(
                     "endpoint": endpoint,
                     "is_batch": str(bool(is_batch)).lower(),
                     "tokens_est": str(int(tu)),
-                    "cost_est": str(float(estimated_cost or 0.0)),
                 },
             ),
             op_id=op_id,
diff --git a/tldw_Server_API/cli/utils/config.py b/tldw_Server_API/cli/utils/config.py
index 20ec2c1685..dc9e78c265 100644
--- a/tldw_Server_API/cli/utils/config.py
+++ b/tldw_Server_API/cli/utils/config.py
@@ -149,8 +149,7 @@ def _apply_defaults(config: dict[str, Any]) -> dict[str, Any]:
                     'evaluations_per_minute': 10,
                     'evaluations_per_day': 100,
                     'total_tokens_per_day': 100000,
-                    'burst_size': 5,
-                    'max_cost_per_day': 1.0
+                    'burst_size': 5
                 }
             }
         },
```

### A.3 `pr3-frontend.patch`

```diff
diff --git a/apps/packages/ui/src/components/Option/Evaluations/components/__tests__/EvaluationComponentAlerts.design-system.test.tsx b/apps/packages/ui/src/components/Option/Evaluations/components/__tests__/EvaluationComponentAlerts.design-system.test.tsx
index 6afbb7e741..3e8da7e11a 100644
--- a/apps/packages/ui/src/components/Option/Evaluations/components/__tests__/EvaluationComponentAlerts.design-system.test.tsx
+++ b/apps/packages/ui/src/components/Option/Evaluations/components/__tests__/EvaluationComponentAlerts.design-system.test.tsx
@@ -120,21 +120,15 @@ describe("Evaluation component product-state alerts", () => {
           limits: {
             evaluations_per_minute: 10,
             evaluations_per_day: null,
-            tokens_per_day: null,
-            cost_per_day: 0,
-            cost_per_month: 0
+            tokens_per_day: null
           },
           usage: {
             evaluations_today: 4,
-            tokens_today: 1200,
-            cost_today: 0,
-            cost_month: 0
+            tokens_today: 1200
           },
           remaining: {
             daily_evaluations: null,
-            daily_tokens: null,
-            daily_cost: 0,
-            monthly_cost: 0
+            daily_tokens: null
           },
           reset_at: "2026-05-29T23:00:00Z"
         }}
diff --git a/apps/packages/ui/src/services/evaluations.ts b/apps/packages/ui/src/services/evaluations.ts
index 2a968f06f5..ccb8221586 100644
--- a/apps/packages/ui/src/services/evaluations.ts
+++ b/apps/packages/ui/src/services/evaluations.ts
@@ -75,20 +75,14 @@ export type EvaluationRateLimitStatus = {
     evaluations_per_minute: number
     evaluations_per_day: number | null
     tokens_per_day: number | null
-    cost_per_day: number
-    cost_per_month: number
   }
   usage: {
     evaluations_today: number
     tokens_today: number
-    cost_today: number
-    cost_month: number
   }
   remaining: {
     daily_evaluations: number | null
     daily_tokens: number | null
-    daily_cost: number
-    monthly_cost: number
   }
   reset_at: string
 }
```

### A.4 `pr3-docs.patch`

```diff
diff --git a/Docs/API-related/Evaluations_API_Unified_Reference.md b/Docs/API-related/Evaluations_API_Unified_Reference.md
index 02280eeb72..935b836c41 100644
--- a/Docs/API-related/Evaluations_API_Unified_Reference.md
+++ b/Docs/API-related/Evaluations_API_Unified_Reference.md
@@ -96,12 +96,12 @@ curl -H "Authorization: Bearer sk-..." https://api.example.com/api/v1/evaluation
 - **Burst Protection**: 10 requests/second max
 
 ### User-Based Limits (Multi-User Mode)
-| Tier       | Requests/Min | Tokens/Min | Batch Size | Cost/Month |
-|------------|-------------|------------|------------|------------|
-| Free       | 10          | 10,000     | 10         | $0         |
-| Basic      | 30          | 50,000     | 50         | $10        |
-| Premium    | 100         | 200,000    | 200        | $50        |
-| Enterprise | Unlimited   | Unlimited  | 1000       | Custom     |
+| Tier       | Requests/Min | Tokens/Min | Batch Size |
+|------------|-------------|------------|------------|
+| Free       | 10          | 10,000     | 10         |
+| Basic      | 30          | 50,000     | 50         |
+| Premium    | 100         | 200,000    | 200        |
+| Enterprise | Unlimited   | Unlimited  | 1000       |
 
 ### Rate Limit Response Headers
 Unified evaluation endpoints include standard response headers that surface current limits and remaining allowances:
@@ -112,8 +112,6 @@ Unified evaluation endpoints include standard response headers that surface curr
 - X-RateLimit-Daily-Limit
 - X-RateLimit-Daily-Remaining
 - X-RateLimit-Tokens-Remaining
-- X-RateLimit-Daily-Cost-Remaining
-- X-RateLimit-Monthly-Cost-Remaining
 
 In addition, draft standard headers are provided for compatibility with proxies and SDKs:
 
@@ -121,7 +119,7 @@ In addition, draft standard headers are provided for compatibility with proxies
 - RateLimit-Remaining
 - RateLimit-Reset (seconds until the current minute window resets)
 
-Note: Remaining values are based on the latest limiter decision; actual token/cost usage may adjust after the request completes when providers return precise usage metadata.
+Note: Remaining values are based on the latest limiter decision; actual token usage may adjust after the request completes when providers return precise usage metadata.
 
 ## Core API Endpoints
 
@@ -683,8 +681,6 @@ Responses from evaluation endpoints also include standard `X-RateLimit-*` header
 - `X-RateLimit-Daily-Limit`
 - `X-RateLimit-Daily-Remaining`
 - `X-RateLimit-Tokens-Remaining`
-- `X-RateLimit-Daily-Cost-Remaining`
-- `X-RateLimit-Monthly-Cost-Remaining`
 
 ### Idempotency
 
@@ -856,7 +852,7 @@ curl "$BASE/embeddings/abtest/abtest_123/events" -H "X-API-KEY: $API_KEY"
 1. **Monitor token usage** via metrics endpoint
 2. **Use smaller models** for initial testing
 3. **Implement sampling** for large datasets
-4. **Set spending limits** via configuration
+4. **Set spending limits** with `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd` (see `Docs/Operations/Usage_Quotas.md`)
 
 ### Security
 1. **Rotate API keys** regularly
diff --git a/Docs/Code_Documentation/Guides/Evaluations_Code_Guide.md b/Docs/Code_Documentation/Guides/Evaluations_Code_Guide.md
index 497ae61d3a..c9fd6713b8 100644
--- a/Docs/Code_Documentation/Guides/Evaluations_Code_Guide.md
+++ b/Docs/Code_Documentation/Guides/Evaluations_Code_Guide.md
@@ -133,7 +133,7 @@ results = await rag_eval.evaluate(
 
 **Auth, Limits, and Error Handling**
 - Auth modes handled in `evaluations_auth.verify_api_key`; admin gating is claim-first (`get_auth_principal` + `RequireRole("admin")` / heavy-admin claim checks).
-- Path limiter (`check_evaluation_rate_limit`) + per‑user `UserRateLimiter` estimate tokens and enforce daily/cost quotas; usage headers are applied by helpers.
+- Path limiter (`check_evaluation_rate_limit`) + per‑user `UserRateLimiter` estimate tokens and enforce the daily evaluations/tokens quotas (LLM spend is capped by the general `limits.llm_cost_*` quotas); usage headers are applied by helpers.
 - Errors are normalized via `create_error_response` and `sanitize_error_message` for consistent API error shapes.
 
 **API Usage Examples (curl)**
diff --git a/Docs/User_Guides/Server/Production_Hardening_Checklist.md b/Docs/User_Guides/Server/Production_Hardening_Checklist.md
index 81a2bbe9c9..3e50026481 100644
--- a/Docs/User_Guides/Server/Production_Hardening_Checklist.md
+++ b/Docs/User_Guides/Server/Production_Hardening_Checklist.md
@@ -45,7 +45,7 @@ CORS & CSRF
 
 Rate Limiting & Abuse Prevention
 - RG (rate gating) global rate limiter is enabled by default in production unless you disable it with `RG_ENABLED=0` or in `config.txt`; in test environments, RG is disabled by default unless you explicitly enable it (for example via `RG_ENABLED=1` or `config.txt`).
-- Treat `RG_ENABLED=0` as an emergency/debug-only setting. RG enforcement and the general `check_rate_limit` fallback are off, but several independent layers stay on regardless (ADR-056): the auth brute-force floor (`check_auth_rate_limit`, 30/min, `AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT`); the AuthNZ login lockout ("Too many failed login attempts...", 429, `MAX_LOGIN_ATTEMPTS`/`LOCKOUT_DURATION_MINUTES`); the audio daily-minutes 402 (`AUDIO_TIER_LIMITS_JSON`); the Evaluations legacy cost caps (`max_cost_per_day`/`max_cost_per_month` in `evaluations_config.yaml`); and the kanban per-action limiter (`KANBAN_RATE_LIMIT_*`). Layers entirely outside RG (RBAC rate classes, billing plan limits, virtual-key budgets) also keep their own switches; see `Docs/Operations/Rate_Limits_Troubleshooting.md`.
+- Treat `RG_ENABLED=0` as an emergency/debug-only setting. RG enforcement and the general `check_rate_limit` fallback are off, but several independent layers stay on regardless (ADR-056): the auth brute-force floor (`check_auth_rate_limit`, 30/min, `AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT`); the AuthNZ login lockout ("Too many failed login attempts...", 429, `MAX_LOGIN_ATTEMPTS`/`LOCKOUT_DURATION_MINUTES`); the audio daily-minutes 402 (`AUDIO_TIER_LIMITS_JSON`); and the kanban per-action limiter (`KANBAN_RATE_LIMIT_*`). Layers entirely outside RG (RBAC rate classes, billing plan limits, virtual-key budgets) also keep their own switches; see `Docs/Operations/Rate_Limits_Troubleshooting.md`.
 - Tune per-module rate limiters (Chat/RAG/Evals) via their respective settings.
 - Consider a network-level rate limit at the reverse proxy for additional protection.
 
```
