# Extract a Provider_Credentials Context from AuthNZ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the BYOK and provider-credential code out of `core/AuthNZ` into a new `core/Provider_Credentials` package, the LLM budget modules into `core/Usage` and the AuthNZ scheduler into `services`, so that `core.LLM_Calls -> core.AuthNZ` and `core.AuthNZ -> core.LLM_Calls/Chat/TTS` both reach 0 while every old import path and patch target keeps working and credential storage behaves exactly as before on SQLite and Postgres.

**Architecture:** `git mv` 16 modules with no change other than import paths (one perl codemod over `app/` and `Helper_Scripts/`). Fifteen old paths become 11-line `sys.modules` aliases of the moved modules, not re-exports. Tests patch these modules through the old paths about 690 times, and a re-export would let those patches miss the code that runs. A required-gate ratchet keeps app code off the aliases. The scheduler, the sixteenth module, moves without a shim; its references are retargeted in the same commit, and the AuthNZ initialize CLI stops starting it. The layering contract locks the result in: slice A deletes `core.LLM_Calls -> core.AuthNZ`, new sibling bans make any `core.AuthNZ -> core.LLM_Calls/Chat/TTS` import fail, and a module-keyed ratchet freezes which AuthNZ modules LLM_Calls still loads indirectly.

**Tech Stack:** Python 3.12, pytest, pytest-cov (CI), perl (codemod), SQLite and PostgreSQL 18 (the AuthNZ database), AST lint tests under `tldw_Server_API/tests/lint/`.

**Spec:** `backlog/tasks/task-13950.17 - Extract-a-Provider-Credentials-context-from-AuthNZ.md` (AC #1-#5; AC1, AC3 and AC4 as corrected 2026-10-26), and the `.17` row of `## Phase 1` in `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md` (effort M, "Security path: run `tests/AuthNZ` in both shards; LLM_Calls imports the new package directly so `LLM_Calls -> AuthNZ` reaches 0", depends on `.13`).

**Measured on:** `origin/dev` 35e711080a merged with `origin/chore/13950-9-layering-contract` 9bff409144 (scratch merge 0d270b00e4), 2026-10-26. This plan was dry-run twice on that tree, by its writer and by an independent dry run. The dry runs cover slice A as written and the first test step of slice B. This revision's new gate and tests were also run there: the alias gate (red with 233 uses on the base, green after the move, its seeded self-test green), the per-module alias tests, the CI-posture and guard contract tests, the widened guardrails, the golden-envelope and `team` checks (SQLite), the CLI test, and the dotted `--cov` targets. They were not run on Postgres (Docker was down), and the laundering ratchet was not run (it needs `.13`'s file). Results:
- K3 printed nothing.
- `--tighten` deleted only `core.LLM_Calls -> core.AuthNZ: 7`; pairs and statements went from 262/683 to 261/676.
- The aliases held in both import orders, for string and module-object patches, and across pickles.
- BYOK rows written by the base code decrypt through the moved code.
- `tests/AuthNZ_SQLite` and `tests/AuthNZ_Unit` gave 1443 passed, 1 skipped on the base, and 1460 passed, 1 skipped on the slice A head (the 17 new tests).
- The Postgres storage tests gave 5 passed on the base and after the move.

`.13` was not applied, so the two `gateway.py` rows below were not part of either run.

## Global Constraints

From the roadmap, verbatim:
- **Branching and PR size.** Every PR targets `dev`. One backlog task per PR. A task with more than about 400 changed lines is split into ordered PRs, each passing CI on its own.
- **Ratchet rules** (this task adds two ratchets and changes the layering contract, two guardrails and `.13`'s isolation test). Every new ratchet:
  - runs in a required gate: appended to backend-required's "Enforce CI contracts and code ratchets" step (frontend: frontend-required), with a workflow-contract assertion like `tests/CI/test_backlog_task_format_ratchet.py`; a ratchet that only runs in the optional full-suite shards enforces nothing (today `test_core_to_api_import_boundary.py` is in that state, and `.9` wires it in);
  - ships with a self-test that seeds one violation and asserts the ratchet fails.
- **Raising a baseline.** It is allowed only through a documented bump-with-reason entry, reviewed in the PR, so that security fixes inside frozen functions are not blocked.
- **Line-keyed legacy baseline.** `Helper_Scripts/ci/scope_predicate_baseline.txt` is still keyed by line number until `.23` re-keys it. Any PR that shifts lines in a file it lists regenerates it with `--write-baseline`.
- **Shims.** A move keeps a re-export shim at the old import path until `git grep` shows no importer and no string monkeypatch target still uses it.
- **Postgres proof.** Postgres-dependent acceptance criteria are proven in the Postgres shard with `TLDW_TEST_POSTGRES_REQUIRED=1`. A local skip is not evidence.
- **Backlog edits** use `tools/backlog-py` only; stage with `git add -A backlog/tasks`. Cite duplicate ids (TASK-12108, TASK-12013) by title.
- **Review gates.** Each AI-authored PR carries a `## Change summary` (ADR-004) and gets a clean-subagent `/superpowers:requesting-code-review` before merge. Each task's own plan gets an independent review plus a dry run before execution.
- **Owner decision (2026-10-25), binding here:** **Scheduler** folds into Jobs (`.48`; no fold work existed when re-checked on 2026-10-25). `.39` covers the recurring APScheduler wrappers and test collection around it.

Task-specific:
- **Moves only.** A moved file differs from its original only in import paths (K3 proves it). The one behaviour change is deleting the AuthNZ initialize CLI's scheduler start. Both dry runs showed it never runs a job before the CLI exits.
- **Aliases, not re-exports.** Every old module path is the `sys.modules` alias in Task 2, Step 3. Never use `from <new> import *`. App code may not import an alias path (`tests/lint/test_authnz_alias_imports.py`, count 0). Tests may, until the retarget follow-up.
- **No new upward edge.** `core.AuthNZ -> services` stays at 1 (`email_service`).
- **Every guard this task adds or widens runs in backend-required's "Enforce CI contracts and code ratchets" step,** with a workflow-contract assertion in `tests/CI/test_license_first_workflow_contracts.py`. That covers the alias gate, the two AuthNZ guardrails it widens, the CI-posture pins (e2e trigger, coverage floor), the layering contract, and `.13`'s isolation test, which Task 6 extends.
- **The line-keyed baseline here:** the codemod rewrites lines in place, and slice B shifts lines only in `initialize.py`, which `scope_predicate_baseline.txt` does not list. The ratchet still runs in K2.
- **Repo/tool rules:** use the venv python `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (written `$PY` below). Run pytest with `-p no:cacheprovider --basetemp "$TMPDIR/<name>"`. The shell is zsh, and macOS `/bin/bash` is 3.2 (no `mapfile`), so run word-splitting scripts with `bash`. Never `git stash`, never `git clean -fdx`. Scratch worktrees go under `$TMPDIR` and are removed after use. Commit messages end with the session's `Co-Authored-By:` trailer.

## Review Focus

1. **Old-path patches that stop reaching the code.** Tests patch moved modules through old paths in three ways:
   - 596 times on module objects (`from ...AuthNZ import byok_runtime`, then `monkeypatch.setattr(byok_runtime, ...)`), in 45 files;
   - 4 times by dotted string (`"...AuthNZ.provider_credential_runtime.load_server_config_snapshot"`);
   - 38 times via `importlib.import_module("...AuthNZ.llm_provider_overrides")`.

   With a re-export shim, every one of them would patch the shim while the moved code runs unpatched. Pinned in Task 2 by `test_old_authnz_path_is_the_moved_module`, `test_patching_the_old_path_patches_the_code_that_runs`, and `test_old_path_aliases_when_imported_first` (one fresh interpreter per module, old path first). Task 4 extends `MOVED`.
2. **Fake modules injected at an old path that stop intercepting, or app code that drifts back onto an alias.**
   - `test_startup_auth.py` installs a fake `AuthNZ.llm_provider_overrides` into `sys.modules` (lines 96, 152, 568). Once `services/startup_auth.py` imports the new path, `test_init_auth_services_runs_sqlite_startup_chain` and `test_init_auth_services_skips_provider_override_runtime_failures` fail until those 3 lines are retargeted.
   - The root `tests/conftest.py:75` no-ops the scheduler inside `except Exception`, so a missed retarget would silently start real schedulers in every test.
   - Open PR #3187 adds 11 new app imports of old paths.

   Pins: the retargeted `test_startup_auth.py` (Task 2), `test_app_code_imports_no_authnz_alias_path` (Task 2, required gate), and `test_root_conftest_disables_the_moved_authnz_scheduler` (Task 5).
3. **Credential storage that works on SQLite but breaks on Postgres.** The risks are plaintext at rest, cross-user or cross-org reads after the repositories move, and rows written before the move no longer decrypting. No Postgres test covers the user or org secret repositories today. Pinned in Task 1 by `assert_user_secret_encrypted_and_owner_scoped`, `assert_org_secret_encrypted_and_scope_bound` (including the same id under `scope_type="team"`) on both backends, and by `assert_pre_move_envelope_decrypts` against an envelope the base code wrote.
4. **Security posture that drops silently for moved code.** Four things key on `core/AuthNZ` and would stop covering the moved files:
   - the e2e trigger in `Helper_Scripts/ci/path_classifier.py`;
   - the AuthNZ coverage floor (4,241 statements move in slice A, and 1,357 more lines in slice B);
   - the backend-probe guard `tests/AuthNZ/unit/test_backend_detection_guardrail.py`;
   - the guardrail-table audit `tests/AuthNZ/unit/test_guardrail_sql_audit.py`.

   The RLS state of the five provider tables is also pinned. Pins: `test_moved_provider_credential_code_keeps_authnz_ci_posture` (required gate; Tasks 3-5), the widened guardrails and their scope assertions (Tasks 2, 4, 5; required gate), and `test_provider_credential_tables_keep_their_rls_posture_postgres` (Task 1).
5. **A cut that passes for the wrong reason.** This covers an `LLM_Calls -> AuthNZ` import turned into `importlib`, either `gateway.py` lookup from `.13` being left behind, AuthNZ reaching the LLM stack again, and LLM_Calls loading more AuthNZ modules indirectly. AC4's static pair reaches 0, but importing `Provider_Credentials.provider_credential_runtime` still loads 24 AuthNZ modules (the package plus 23 submodules). Pins:
   - `--tighten` must delete the baseline line, and Task 2, Step 6 greps for `importlib`;
   - `test_authnz_imports_no_llm_calls_chat_or_tts_but_provider_credentials_may` (Tasks 3 and 6);
   - `test_llm_calls_loads_only_the_known_authnz_modules` (Task 6), a module-keyed equality ratchet in `.13`'s isolation test.

   Cutting the indirect load is the follow-up "Invert provider credential resolution" (close-out).

---

## Inventory

**Moves** (lines as of 0d270b00e4):

| Module (old: `core/AuthNZ/...`) | Lines | New home | Why |
|---|---|---|---|
| `byok_config.py` | 309 | `core/Provider_Credentials/byok_config.py` | AC1; imported by LLM_Calls |
| `byok_helpers.py` | 472 | `core/Provider_Credentials/byok_helpers.py` | AC1; 2 LLM_Calls and 2 TTS imports |
| `byok_rotation.py` | 327 | `core/Provider_Credentials/byok_rotation.py` | AC1 |
| `byok_runtime.py` | 2,879 | `core/Provider_Credentials/byok_runtime.py` | AC1 |
| `byok_testing.py` | 699 | `core/Provider_Credentials/byok_testing.py` | AC1; 4 LLM_Calls, 4 Chat and 2 TTS imports |
| `user_provider_secrets.py` | 139 | `core/Provider_Credentials/user_provider_secrets.py` | AC1; BYOK envelope encryption |
| `provider_credential_runtime.py` | 1,005 | `core/Provider_Credentials/provider_credential_runtime.py` | AC1; imported by LLM_Calls |
| `llm_provider_overrides.py` | 1,092 | `core/Provider_Credentials/llm_provider_overrides.py` | AC1; 3 LLM_Calls imports |
| `repos/user_provider_secrets_repo.py` | 619 | `core/Provider_Credentials/repos/` | AC1 provider repos |
| `repos/org_provider_secrets_repo.py` | 717 | same | AC1 |
| `repos/llm_provider_overrides_repo.py` | 482 | same | AC1; 2 LLM_Calls imports |
| `repos/byok_oauth_state_repo.py` | 488 | same | AC1 |
| `repos/byok_validation_runs_repo.py` | 298 | same | AC1 |
| `llm_budget_guard.py` | 144 | `core/Usage/llm_budget_guard.py` | AC2; beside `llm_quota_gate.py` and `llm_accounting.py` (Billing holds plans and enforcement, not LLM budgets) |
| `llm_budget_middleware.py` | 395 | `core/Usage/llm_budget_middleware.py` | AC2; 1 LLM_Calls import |
| `scheduler.py` | 818 | `services/authnz_scheduler.py` | AC2; same naming as the other `services/*_scheduler.py` |

The 10 files the backlog counts as "about 7.5K lines" are the top 8 rows plus the two budget modules: 7,461 lines. Line counts are pre-#3191; #3191 edits `llm_provider_overrides.py`.

**Stays in AuthNZ** (these use credentials but do not import the LLM stack):
- `settings.py`: holds `BYOK_ENCRYPTION_KEY` and `BYOK_SECONDARY_ENCRYPTION_KEY`; the key material stays with AuthNZ config.
- `migrations.py` and `pg_migrations_extra.py`: DDL for the five tables, kept in the AuthNZ migration chain (same DB).
- `repos/orgs_teams_repo.py`: org deletion cascades over `org_provider_secrets`.
- `federation/oidc_service.py`, `admin_webhook_secrets.py`, `secret_backends/*`, `repos/managed_secret_refs_repo.py` and `repos/workspace_provider_installations_repo.py`: consumers of the BYOK envelope crypto and repositories (MCP credential broker, OIDC, webhooks, workspace integrations).
- `crypto_utils.py`, `api_key_crypto.py`, `key_resolution.py` and `virtual_keys.py`: AuthNZ API keys, not provider keys.

## Edges

**`core.LLM_Calls -> core.AuthNZ`.** This task owns all 9 statements on its PR base. The baseline line is `core.LLM_Calls -> core.AuthNZ: 7` on 0d270b00e4 (line 206 of `layering_baseline.txt`). `.13` raises it to 8 in PR C1 and to 9 in PR C2, each time with a `# bump:` that names TASK-13950.17. This task deletes the line.

| Importer | Line (0d270b00e4) | Level | Target | Source |
|---|---|---|---|---|
| `core/LLM_Calls/adapter_utils.py` | 7 | module | `AuthNZ.provider_credential_runtime` | dev |
| `core/LLM_Calls/Summarization_General_Lib.py` | 24 | module | `AuthNZ.provider_credential_runtime` | dev |
| `core/LLM_Calls/providers/custom_openai_adapter.py` | 8 | module | `AuthNZ.provider_credential_runtime` | dev |
| `core/LLM_Calls/providers/huggingface_adapter.py` | 10 | module | `AuthNZ.byok_config` | dev |
| `core/LLM_Calls/providers/local_adapters.py` | 8 | module | `AuthNZ.provider_credential_runtime` | dev |
| `core/LLM_Calls/openai_credentials.py` | 88 | function | `AuthNZ.provider_credential_runtime` | dev |
| `core/LLM_Calls/payload_utils.py` | 141 | function | `AuthNZ.byok_config` | dev |
| `core/LLM_Calls/gateway.py`, in `resolve_provider_api_key` | new | function | `AuthNZ.llm_provider_overrides` | `.13` C1 (from `chat_service.py:2195`) |
| `core/LLM_Calls/gateway.py`, in `_resolve_base_url_override` | new | function | `AuthNZ.byok_helpers` | `.13` C2 (from `chat_service.py:2294`) |

All 9 targets are moved modules, so the K1 codemod cuts all of them.

**`core.AuthNZ -> core.LLM_Calls/Chat/TTS`.** This task owns all 27 statements. None of them is in the baseline, because all four packages are in the `domain` layer and the contract leaves same-layer pairs unruled. This plan adds them to `SIBLING_BANS`.
- **LLM_Calls, 19:**
  - `byok_helpers` 2, `byok_rotation` 1, `byok_runtime` 1, `byok_testing` 4;
  - `llm_budget_middleware` 1, `llm_provider_overrides` 3, `provider_credential_runtime` 2, `user_provider_secrets` 1;
  - `repos.llm_provider_overrides_repo` 2, `repos.org_provider_secrets_repo` 1, `repos.user_provider_secrets_repo` 1.
- **Chat, 4:** all in `byok_testing`.
- **TTS, 4:** `byok_helpers` 2 (function-level) and `byok_testing` 2.

After slice A, 1 statement is left (`llm_budget_middleware` -> LLM_Calls). After slice B, none is left.

**Coordination with `.13`** (`Docs/superpowers/plans/2026-10-26-task-13950-13-llm-gateway.md`, PRs A, B, C1, C2, D in that order):
- **Who cuts what.**
  - `.13` owns `core.LLM_Calls -> core.Chat`: 47 statements to 0. It never lowers `core.LLM_Calls -> core.AuthNZ`.
  - `.17` cuts all 9 `core.LLM_Calls -> core.AuthNZ` statements in the table above, and all 27 `core.AuthNZ -> core.LLM_Calls/Chat/TTS` statements.
  - `AuthNZ/llm_provider_overrides.py`, which the gateway reads, goes to `core/Provider_Credentials/` (Task 2). The gateway's lookup is rewritten by the K1 codemod.
- **When each `.17` slice can start.**
  - PR A1 (tests only) can start any time.
  - PR A2 needs `.13` PR C2 merged. C2 is the last `.13` slice that brings an AuthNZ import into LLM_Calls. If A2 ran earlier, C2 would move `_resolve_base_url_override` with an old-path `byok_helpers` import, which the alias gate would then reject.
  - PRs D (hooks) and D2 (ADR-025) add no AuthNZ import, so A2 could run alongside them, but the frontmatter dependency (`.13` Done) holds A2 until D2 merges unless the owner relaxes it.
  - Slice B depends only on A2.
- **Avoiding edits to the same lines.**
  - In the shared LLM_Calls files (`adapter_utils.py`, `Summarization_General_Lib.py`, `providers/custom_openai_adapter.py`, `providers/local_adapters.py`, `payload_utils.py`, `openai_credentials.py`), `.13` rewrites the `core.Chat`/`Chat_Deps` import lines and `.17` rewrites only the `core.AuthNZ.<moved module>` path token on its own lines.
  - In `gateway.py`, `.17` rewrites only the path token of the two function-level imports. D's hooks edit `perform_chat_api_call(_async)`, a different hunk.
  - In `layering_baseline.txt`, `.17` deletes the line that `.13` last set to 9.
  - `.13`'s `tests/lint/direct_llm_adapter_calls_baseline.json` keys `core/AuthNZ/byok_testing.py: 1`. Task 2 re-keys it with a "moved from" reason.
  - `.13`'s `tests/lint/test_llm_calls_import_isolation.py` gains Task 6's ratchet at the end of the file.
  - In `tests/performance/test_authnz_multiuser_sqlite_load.py`, `.13` C1 edits `pytestmark` and slice B edits the scheduler import at line 72.
- **What `.13` adds that the aliases serve.** `.13`'s new tests import `AuthNZ.llm_provider_overrides` through the old path: `tests/LLM_Calls/test_gateway_module.py`, and `healthy_empty_override_store` in `tests/helpers/provider_keys.py`. Both keep working through the alias. Task 2, Step 5 runs them, and runs `.13`'s isolation guard.
- **If an AuthNZ edge cannot be cut.** If the PR-base scan shows an `LLM_Calls -> AuthNZ` statement whose target is not a moved module, the codemod cannot cut it. Stop and apply the contract's fix ("have the caller pass it in"). The `.13` plan adds none.

## Shims and patch targets

Counts on 0d270b00e4, made with regex and AST counters over `git ls-files '*.py'`. "Old path" means `tldw_Server_API.app.core.AuthNZ.<module>`. The test counts depend on the regex, since multi-line imports count differently. `.13` and `.14` add a few old-path test references to `llm_provider_overrides`; close-out re-measures.

| Group | App importers (retargeted here) | Test import lines (kept) | Dotted string patch targets | Module-path strings | Patches on old-path module objects |
|---|---|---|---|---|---|
| 13 credential modules | K1 changed 234 lines in 109 files in the dry run, plus 3 lines in `tests/Services/test_startup_auth.py`. That includes 28 lines inside 11 moved files, 1 string in `services/startup_auth.py:83`, `Helper_Scripts/AuthNZ/rotate_byok_keys.py:19`, and 2 relative `....core.AuthNZ.provider_credential_runtime` imports in `Prompt_Management/prompt_studio/{evaluation_manager,test_runner}.py`. The alias gate finds 233 uses on the base. #3191 adds 1 more. | about 520-540 | 4: `tests/Chat/integration/test_chat_endpoint.py:628,675` and `tests/Chat/integration/test_persona_backed_chat_conversations.py:86,233` (kept, work via alias) | 41: 38 `importlib.import_module` calls in `tests/AuthNZ_Unit/test_llm_provider_overrides.py` (kept), and 3 `_install_module` calls in `tests/Services/test_startup_auth.py:96,152,568` (**retargeted in Task 2**) | 596 in 45 files (kept) |
| 2 budget modules | 4: `api/v1/endpoints/chat.py:320`, `prompts.py:46`, `rpg.py:42`, `main.py:2476` | 27 | 9: `tests/AuthNZ_Unit/test_auth_governor_budget.py:270,274,278,476,480,484,527,531,535` (kept) | 0 | 45 in 4 files (kept) |
| scheduler (no shim) | 4: `core/AuthNZ/initialize.py:45` (deleted with the CLI start), `services/startup_recurring_schedulers.py:594,600`, and the test shim `core/Scheduler/core/AuthNZ/scheduler.py:6` (deleted) | 7 (all retargeted) | 2: `core/Scheduler/tests/test_security_fixes.py:298` and `tests/AuthNZ/integration/test_monitoring_metrics_summary.py:126` (retargeted) | 2: `tests/Services/test_shutdown_coordinator.py:121,122` (retargeted) | 9 in 3 files (retargeted with their imports) |

## PR slices

Slice A grew past 400 hand-written lines once the alias gate, the guardrail widening and the required-gate pins were added. It is split in two: the characterization tests (A1) pass on the base on their own, and they give the before-evidence that A2 must preserve.

| PR | Branch | Base | Tasks | Hand-written lines | Mechanical |
|---|---|---|---|---|---|
| A1: storage invariants | `test/13950-17a1-provider-credential-invariants` | `dev` (any time; needs only the existing AuthNZ fixtures) | 1 | about 170 | none |
| A2: `core/Provider_Credentials` | `refactor/13950-17a2-provider-credentials` | `dev` after A1, `.13` PR C2 and #3191; after `.13` PR D too, unless the owner relaxes the frontmatter dependency | 2-3 | about 300: alias gate ~110, alias tests ~45, CI-posture and guard contract tests ~40, layering bans and self-test ~35, the shim template and loop ~20, baselines and config ~25, guardrail roots ~10, `test_startup_auth.py` 3 | the 13 renames (9.5K lines; K3 proves the diffs are import-only), the 234-line codemod, the 13 generated aliases |
| B: budgets, scheduler, last ban, laundering ratchet | `refactor/13950-17b-budget-scheduler` | `dev` after A2 merges | 4-6 | about 170: laundering ratchet with its seed ~60, CLI and conftest tests ~20, posture, coverage and guardrail extensions ~20, the CLI deletion ~6, docs ~10, budget aliases, importers and snapshot ~15, scheduler retargets 17 | 3 renames, 2 generated aliases, the `Docs/Published` mirror |

Each PR is green on its own. After A2, the only `AuthNZ -> LLM_Calls` statement left is the unbanned one in `llm_budget_middleware`, until slice B.

**Conflict hazards:**
- **#3191** (`llm_provider_overrides.py`; it adds 1 old-path app import) will be on `dev` before `.17` runs, and this plan assumes it is merged. K1 and the alias gate then cover its import. If it is still open when A2 starts, port it with the recipe below, after A2 merges.
- **#3187 and #2821** edit moved files: #3187 edits `byok_runtime.py`, `provider_credential_runtime.py`, `repos/org_provider_secrets_repo.py` and `repos/user_provider_secrets_repo.py`; #2821 edits `byok_runtime.py`, `repos/org_provider_secrets_repo.py` and `user_provider_secrets.py`. Both conflict with `dev` already, and stopped moving on 2026-10-04/05. Git sees no rename, because the old path survives as an alias, and K1 rewrote the import lines in their patches' context. Post this porting recipe on both PRs and in PR A2's body:
  1. In the branch, before merging dev: `MB=$(git merge-base HEAD origin/dev); TIP=$(git rev-parse HEAD)`.
  2. `git merge origin/dev`. Expect a conflict on every moved file the branch edited.
  3. For each such file `f` (a path under `core/AuthNZ/` without `.py`, for example `byok_runtime` or `repos/org_provider_secrets_repo`):
     - `git checkout origin/dev -- tldw_Server_API/app/core/AuthNZ/$f.py` (keep the alias);
     - `git diff $MB $TIP -- tldw_Server_API/app/core/AuthNZ/$f.py | sed 's#core/AuthNZ/#core/Provider_Credentials/#g' | git apply -3`;
     - the conflicts are on import lines only. In each conflict block, keep dev's `Provider_Credentials` lines and the branch's added lines, and drop exact duplicates;
     - `perl -pi "$TMPDIR/t17_codemod.pl" tldw_Server_API/app/core/Provider_Credentials/$f.py`, then `git add` both paths.
  4. Run K1 over every app and Helper_Scripts file the branch touches: `git diff --name-only -z origin/dev...HEAD -- 'tldw_Server_API/app/*.py' 'Helper_Scripts/*.py' | xargs -0 perl -pi "$TMPDIR/t17_codemod.pl"`. For example, #3187 adds 11 old-path imports in `MCP_unified/adapters/model_completion/*`, `Billing/enforcement.py`, AuthNZ repos and `byok_runtime`.
  5. New modules the branch adds under `core/AuthNZ` stay there (`Provider_Credentials -> AuthNZ` is allowed). #3187 adds `repos/provider_scope_result.py` and `repos/provider_usage_reservations_repo.py`. Move a new module to `core/Provider_Credentials/repos/` (`git mv`; it needs no alias, because dev never had it) only if the AuthNZ sibling bans fail on it.
  6. Run the alias gate, the alias tests, the layering contract, and `git grep -nE -A3 "_install_module\(|setitem\(\s*sys\.modules" -- tldw_Server_API/tests | grep -E "AuthNZ\.(byok_|user_provider|provider_credential|llm_provider_overrides|repos\.(user|org|llm|byok))"` (`-A3` catches calls split across lines, such as `tests/Services/test_startup_auth.py:96`). Expect no output. Point any fake-module injection at the new path.
- **#854** edits the import block of `providers/local_adapters.py`: rebase it, then run K1 on the file.
- **#3331 (`.6`), if merged first:** `tests/lint/test_test_seam_env_reads_guarded.py` lists `core/AuthNZ/byok_testing.py` (re-key in Task 2), and `core/AuthNZ/llm_budget_guard.py` and `core/AuthNZ/llm_budget_middleware.py` (re-key in Task 4).
- **`.39`, if merged first:** its `AsyncIOScheduler(` cap may key `core/AuthNZ/scheduler.py`; re-key it in Task 5. `.39` also collects `core/Scheduler/tests`, whose AuthNZ import Task 5 retargets.
- **`.12`, Task 10:** it appends pins at the end of `test_layering_contract.py`, where Task 3 appends its self-test; take both on rebase. `.12` Task 2 moves `LLM_Calls/provider_identity.py` to `core/Utils` behind a shim. 10 of the 18 `Provider_Credentials -> LLM_Calls` statements import it, and they keep working through `.12`'s shim.
- **`.16`:** it refactors `_targets` and `_scan` in `test_layering_contract.py`. The alias gate imports only `_module_name` from that file; if `.16` renames it, follow the rename.
- **`.14`:** it adds old-path `llm_provider_overrides` test imports (they work through the alias), and edits `test_authnz_multiuser_sqlite_load.py`, the same file as Task 5 but other lines.

## Shared commands

**K1: the codemod** (old credential module paths to `core.Provider_Credentials`). It matches absolute, relative and string forms:

```bash
cat > "$TMPDIR/t17_codemod.pl" <<'EOF'
my $top = qr/byok_config|byok_helpers|byok_rotation|byok_runtime|byok_testing|user_provider_secrets|provider_credential_runtime|llm_provider_overrides/;
my $repos = qr/user_provider_secrets_repo|org_provider_secrets_repo|llm_provider_overrides_repo|byok_oauth_state_repo|byok_validation_runs_repo/;
s/\bcore\.AuthNZ\.repos\.($repos)(?![A-Za-z0-9_])/core.Provider_Credentials.repos.$1/g;
s/\bcore\.AuthNZ\.($top)(?![A-Za-z0-9_])/core.Provider_Credentials.$1/g;
EOF
```

It does not rewrite `from tldw_Server_API.app.core.AuthNZ import byok_runtime`. That form appears only in tests (none in `app/` on 0d270b00e4); the alias gate rejects it in app code. Re-run K1 after any porting.

**K2: backend-required's two ratchet steps, exactly as CI runs them**, from the worktree root:

```bash
cat > "$TMPDIR/t17_gates.py" <<'EOF'
import os, subprocess, sys
import yaml
PY = "/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python"
steps = yaml.safe_load(open(".github/workflows/backend-required.yml"))["jobs"]["backend-required"]["steps"]
base = subprocess.check_output(["git", "merge-base", "origin/dev", "HEAD"], text=True).strip()
failed = 0
for name in ("Enforce tenant isolation ratchets", "Enforce CI contracts and code ratchets"):
    step = next(s for s in steps if s.get("name") == name)
    env = {**os.environ, **{k: str(v) for k, v in step.get("env", {}).items()}, "BACKLOG_TASK_FORMAT_BASE": base}
    tmp = os.path.join(os.environ["TMPDIR"], f"t17-gate-{len(name)}")
    cmd = step["run"].replace("pytest -q", f"{PY} -m pytest -q -p no:cacheprovider --basetemp {tmp}", 1)
    code = subprocess.run(["bash", "-c", cmd], env=env).returncode
    print(f"{name}: exit {code}")
    failed |= code
sys.exit(failed)
EOF
$PY "$TMPDIR/t17_gates.py"
```

Pass means both steps print `exit 0`. A failure counts only if it does not also fail when K2 runs in a base worktree.
- **Known base failures on 0d270b00e4:** `test_function_size_ratchet.py` (dev outgrew `.8`'s seed, for example `endpoints.chat:create_chat_completion` 2870 > 2865) and `test_charactersragdb_surface_ratchet.py`. `.8` re-seeds these before it merges.
- **Local flake:** `test_route_auth_ratchet.py::test_force_enabled_routers_are_mounted` can abort locally with `libc++abi ... recursive_mutex lock failed`. That is a macOS subprocess flake: re-run it alone.
- **Dry runs on a scratch merge** made before `.9` merges must set the merge base to that merge. On the real PR, `git merge-base origin/dev HEAD` is right.

**K3: move integrity.** Every moved credential file equals its base version with K1 applied:

```bash
bash -c 'BASE=$(git merge-base origin/dev HEAD); A=tldw_Server_API/app/core/AuthNZ; P=tldw_Server_API/app/core/Provider_Credentials
for m in byok_config byok_helpers byok_rotation byok_runtime byok_testing user_provider_secrets provider_credential_runtime llm_provider_overrides repos/user_provider_secrets_repo repos/org_provider_secrets_repo repos/llm_provider_overrides_repo repos/byok_oauth_state_repo repos/byok_validation_runs_repo; do
  diff <(git show "$BASE:$A/$m.py" | perl -p "$TMPDIR/t17_codemod.pl") <(git show "HEAD:$P/$m.py") >/dev/null || echo "DIFFERS: $m"; done'
```

Pass means no output. Slice B checks its budget modules the same way, with no codemod: `bash -c 'BASE=$(git merge-base origin/dev HEAD); for m in llm_budget_guard llm_budget_middleware; do diff <(git show "$BASE:tldw_Server_API/app/core/AuthNZ/$m.py") <(git show "HEAD:tldw_Server_API/app/core/Usage/$m.py") >/dev/null || echo "DIFFERS: $m"; done'`. The scheduler must show `R100`. GitHub's PR view shows the moved files as about 9.5K added lines, so tell reviewers to read the diff with `git diff -B -M origin/dev...HEAD`, which shows each move as a copy with only its import lines changed.

**K4: the AuthNZ suites at CI parity, base against head.** Every run uses `TLDW_TEST_POSTGRES_REQUIRED=1`, so it needs the local Postgres that the AuthNZ fixtures use. The fixtures start Docker on `localhost:5432` with `tldw_user` / `TestPassword123!` when it is not running.

The full local Postgres run (`tests/AuthNZ` plus `tests/AuthNZ_Postgres`, about 2,218 tests) takes 4-5 hours, so K4 runs a per-slice subset:
- **A2:** `tests/AuthNZ_Postgres`, plus the `tests/AuthNZ` files that import a credential module. That is 150 tests on 0d270b00e4, plus the 3 new ones.
- **B:** the files that import the budget modules or the scheduler, plus `test_allowlists_budget_402_pg.py`.

The `Full Suite shard` jobs in CI (`auth-*`, with a Postgres service) are the complete Postgres evidence.

```bash
cat > "$TMPDIR/t17_authnz_runs.sh" <<'EOF'
#!/usr/bin/env bash
# Usage, from a worktree root: bash "$TMPDIR/t17_authnz_runs.sh" <label> <A|B>
set -uo pipefail
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
label=$1; slice=$2
run() {
  name=$1; shift
  TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -p no:cacheprovider \
    --basetemp "$TMPDIR/t17-$label-$name" -rfE "$@" > "$TMPDIR/t17-$label-$name.log" 2>&1
  grep -E '^(FAILED|ERROR) ' "$TMPDIR/t17-$label-$name.log" | sed 's/ - .*//' | sort > "$TMPDIR/t17-$label-$name.failed"
  echo "$name: $(grep -E '^=+ .* in [0-9.]+s' "$TMPDIR/t17-$label-$name.log" | tail -1)"
}
cred='core\.(AuthNZ|Provider_Credentials)(\.repos)?(\.| import .*)(byok_|user_provider_secrets|provider_credential_runtime|llm_provider_overrides|org_provider_secrets_repo)'
if [ "$slice" = A ]; then
  run pg tldw_Server_API/tests/AuthNZ_Postgres $(git grep -lE "$cred" -- 'tldw_Server_API/tests/AuthNZ/*test_*.py')
else
  run pg tldw_Server_API/tests/AuthNZ_Postgres/test_allowlists_budget_402_pg.py $(git grep -lE 'llm_budget_|AuthNZ(\.| import )scheduler|authnz_scheduler' -- 'tldw_Server_API/tests/AuthNZ/*test_*.py' 'tldw_Server_API/tests/AuthNZ_Postgres/*test_*.py')
fi
run sqlite tldw_Server_API/tests/AuthNZ_SQLite tldw_Server_API/tests/AuthNZ_Unit
users=$(git grep -lE "core\.(AuthNZ|Provider_Credentials|Usage)(\.repos)?(\.| import .*)(byok_|user_provider_secrets|provider_credential_runtime|llm_provider_overrides|org_provider_secrets_repo|llm_budget_)" -- 'tldw_Server_API/tests/*test_*.py' | grep -vE '^tldw_Server_API/tests/AuthNZ(_Unit|_SQLite|_Postgres)?/')
run importers $users
EOF
git worktree add --detach "$TMPDIR/t17-base" "$(git merge-base origin/dev HEAD)"
(cd "$TMPDIR/t17-base" && bash "$TMPDIR/t17_authnz_runs.sh" base A)
bash "$TMPDIR/t17_authnz_runs.sh" head A
for n in pg sqlite importers; do comm -13 "$TMPDIR/t17-base-$n.failed" "$TMPDIR/t17-head-$n.failed"; done
```

For slice B, pass `B` instead of `A`.

Pass means three things:
- the `comm` loop prints nothing (no new failure in any run);
- each head summary line is a pytest summary (`=== N passed ... in Ns ===`);
- no head line shows a skip that the base did not have.

The `importers` run covers the 113 test files outside the AuthNZ suites that import a moved module. On 0d270b00e4, 22 of them fail on both trees. Keep `$TMPDIR/t17-base` until the bandit comparison is done, then run `git worktree remove --force "$TMPDIR/t17-base"`.

---

### Task 1 (PR A1): Storage invariants on both backends (characterization)

This task writes tests that pass on the base, through the old paths, as the record of today's posture. Task 2 points them at the new package, and they must still pass.

**Files:**
- Create: `tldw_Server_API/tests/helpers/provider_credential_invariants.py`
- Create: `tldw_Server_API/tests/AuthNZ_SQLite/test_provider_credentials_storage_sqlite.py` (shard `auth-sqlite`)
- Create: `tldw_Server_API/tests/AuthNZ_Postgres/test_provider_credentials_storage_pg.py` (shard `auth-postgres`)

**Interfaces:**
- Consumes:
  - `_setup_byok_sqlite(tmp_path, monkeypatch) -> dict` (keys `pool`, `admin`, `user`, `org`, `team`) from `tests/AuthNZ_SQLite/test_byok_endpoints_sqlite.py:54`. It sets `BYOK_ENCRYPTION_KEY` to base64 of `b"k" * 32`.
  - `_create_postgres_runtime_scope(test_db_pool) -> tuple[int, int, int]` (user, org, team) from `tests/AuthNZ_Postgres/test_byok_oauth_endpoints_pg.py:180`.
  - the `test_db_pool` fixture (`tests/AuthNZ/conftest.py:1831`, re-exported by `tests/AuthNZ_Postgres/conftest.py`).
- Produces:
  - `async def assert_user_secret_encrypted_and_owner_scoped(pool: Any, *, owner_id: int, other_id: int) -> None`;
  - `async def assert_org_secret_encrypted_and_scope_bound(pool: Any, *, org_id: int, other_org_id: int) -> None`;
  - `def assert_pre_move_envelope_decrypts() -> None`;
  - `PLAINTEXT` and `GOLDEN_ENVELOPE`.

- [ ] **Step 1: Write the shared assertions** in `tests/helpers/provider_credential_invariants.py`. The Task 2 codemod later points the imports at the new package.

```python
"""Storage invariants for provider credentials, shared by the SQLite and Postgres suites (TASK-13950.17)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import AuthnzOrgProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    build_secret_payload,
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    key_hint_for_api_key,
    loads_envelope,
)

PLAINTEXT = "sk-t13950-17-invariant-0001"
# Written by the pre-move code (0d270b00e4) with BYOK_ENCRYPTION_KEY = base64(b"k" * 32).
GOLDEN_ENVELOPE = (
    '{"_enc": "aesgcm:v1", "nonce": "MRxmSm87PadxTUYn", '
    '"ct": "FJf6UufOTYiOtvbRSWKiv0Qh8K/pfvZQT7Mpplz6mUxheW+XH727k315", "tag": "y9c2ilJnbPceEmTDbCkJVg=="}'
)


def _blob() -> str:
    return dumps_envelope(encrypt_byok_payload(build_secret_payload(PLAINTEXT)))


def _assert_encrypted(blob: Any) -> None:
    assert isinstance(blob, str) and blob
    assert PLAINTEXT not in blob
    assert decrypt_byok_payload(loads_envelope(blob))["api_key"] == PLAINTEXT


def assert_pre_move_envelope_decrypts() -> None:
    assert decrypt_byok_payload(loads_envelope(GOLDEN_ENVELOPE)) == {"api_key": PLAINTEXT}


async def assert_user_secret_encrypted_and_owner_scoped(pool: Any, *, owner_id: int, other_id: int) -> None:
    repo = AuthnzUserProviderSecretsRepo(pool)
    await repo.ensure_tables()
    await repo.upsert_secret(
        user_id=owner_id,
        provider="openai",
        encrypted_blob=_blob(),
        key_hint=key_hint_for_api_key(PLAINTEXT),
        metadata=None,
        updated_at=datetime.now(timezone.utc),
    )
    raw = await pool.fetchval(
        "SELECT encrypted_blob FROM user_provider_secrets WHERE user_id = ? AND provider = ?", owner_id, "openai"
    )
    _assert_encrypted(raw)
    assert (await repo.fetch_secret_for_user(owner_id, "openai"))["encrypted_blob"] == raw
    assert await repo.fetch_secret_for_user(other_id, "openai") is None
    assert await repo.list_secrets_for_user(other_id) == []


async def assert_org_secret_encrypted_and_scope_bound(pool: Any, *, org_id: int, other_org_id: int) -> None:
    repo = AuthnzOrgProviderSecretsRepo(pool)
    await repo.ensure_tables()
    await repo.upsert_secret(
        scope_type="org",
        scope_id=org_id,
        provider="openai",
        encrypted_blob=_blob(),
        key_hint=key_hint_for_api_key(PLAINTEXT),
        metadata=None,
        updated_at=datetime.now(timezone.utc),
    )
    raw = await pool.fetchval(
        "SELECT encrypted_blob FROM org_provider_secrets WHERE scope_type = ? AND scope_id = ? AND provider = ?",
        "org",
        org_id,
        "openai",
    )
    _assert_encrypted(raw)
    assert await repo.fetch_secret("org", other_org_id, "openai") is None
    assert await repo.fetch_secret("team", org_id, "openai") is None
    assert await repo.list_secrets(scope_type="org", scope_id=other_org_id) == []
```

`DatabasePool.fetchval` converts `?` to `$n` on Postgres (`AuthNZ/database.py:1595`), so one query string serves both backends. The golden envelope was checked against both the base and the moved code.

- [ ] **Step 2: Write the SQLite tests** (`pytestmark = pytest.mark.integration`):

```python
@pytest.mark.asyncio
async def test_user_secret_is_encrypted_at_rest_and_owner_scoped_sqlite(tmp_path, monkeypatch) -> None:
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    await assert_user_secret_encrypted_and_owner_scoped(
        state["pool"], owner_id=int(state["user"]["id"]), other_id=int(state["admin"]["id"])
    )


@pytest.mark.asyncio
async def test_org_secret_is_encrypted_at_rest_and_scope_bound_sqlite(tmp_path, monkeypatch) -> None:
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    org_id = int(state["org"]["id"])
    await assert_org_secret_encrypted_and_scope_bound(state["pool"], org_id=org_id, other_org_id=org_id + 1)


@pytest.mark.asyncio
async def test_a_pre_move_envelope_still_decrypts(tmp_path, monkeypatch) -> None:
    await _setup_byok_sqlite(tmp_path, monkeypatch)
    assert_pre_move_envelope_decrypts()
```

- [ ] **Step 3: Write the Postgres tests.** Use `pytestmark = pytest.mark.integration`, and a `byok_key` fixture that does `monkeypatch.setenv("BYOK_ENCRYPTION_KEY", base64.b64encode(b"k" * 32).decode("ascii"))` and `reset_settings()` from `core.AuthNZ.settings` before and after `yield`.

```python
PROVIDER_TABLES = ("byok_oauth_state", "byok_validation_runs", "llm_provider_overrides", "org_provider_secrets", "user_provider_secrets")


@pytest.mark.asyncio
async def test_user_secret_is_encrypted_at_rest_and_owner_scoped_postgres(test_db_pool, byok_key) -> None:
    owner_id, _, _ = await _create_postgres_runtime_scope(test_db_pool)
    other_id, _, _ = await _create_postgres_runtime_scope(test_db_pool)
    await assert_user_secret_encrypted_and_owner_scoped(test_db_pool, owner_id=owner_id, other_id=other_id)


@pytest.mark.asyncio
async def test_org_secret_is_encrypted_at_rest_and_scope_bound_postgres(test_db_pool, byok_key) -> None:
    _, org_id, _ = await _create_postgres_runtime_scope(test_db_pool)
    _, other_org_id, _ = await _create_postgres_runtime_scope(test_db_pool)
    await assert_org_secret_encrypted_and_scope_bound(test_db_pool, org_id=org_id, other_org_id=other_org_id)


@pytest.mark.asyncio
async def test_provider_credential_tables_keep_their_rls_posture_postgres(test_db_pool) -> None:
    """Pins an absence: RLS is off and no policy exists on these tables today.

    The move cannot change that, because the DDL stays in the AuthNZ migrations. Isolation is
    the repositories' owner predicates, which the two scope tests above prove. The three owned
    tables are listed in Helper_Scripts/ci/rls_coverage_baseline.txt. Adding RLS to them is a
    deliberate change that edits this test.
    """
    for ensure in (ensure_user_provider_secrets_pg, ensure_org_provider_secrets_pg, ensure_byok_oauth_state_pg,
                   ensure_llm_provider_overrides_pg, ensure_byok_validation_runs_table_pg):
        assert await ensure(test_db_pool) is True
    names = ", ".join(f"'{table}'" for table in PROVIDER_TABLES)  # constants, not input
    rows = await test_db_pool.fetchall(
        f"SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class WHERE relkind = 'r' AND relname IN ({names})"
    )
    assert {row["relname"]: (row["relrowsecurity"], row["relforcerowsecurity"]) for row in rows} == {
        table: (False, False) for table in PROVIDER_TABLES
    }
    assert await test_db_pool.fetchall(f"SELECT tablename FROM pg_policies WHERE tablename IN ({names})") == []
```

The five `ensure_*_pg` functions come from `tldw_Server_API.app.core.AuthNZ.pg_migrations_extra`. A list bound to `ANY($1)` does not work here, because `DatabasePool` flattens list parameters. That is why the names are inlined constants.

- [ ] **Step 4: Run them on the base tree.** `TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. TEST_MODE=true $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t17-t1" tldw_Server_API/tests/AuthNZ_Postgres/test_provider_credentials_storage_pg.py tldw_Server_API/tests/AuthNZ_SQLite/test_provider_credentials_storage_sqlite.py`.
  - Expected: `6 passed`, with nothing skipped.
  - In the dry runs on 0d270b00e4, the first 5 passed on Postgres and SQLite. The SQLite golden-envelope and `team`-scope checks were added later; they passed on SQLite with the moved code, and are not yet run on Postgres.
  - An asyncpg `another operation is in progress` error during fixture bootstrap is a known local flake; re-run.

- [ ] **Step 5: Commit and open PR A1.**
  - `git add` the three files, then `git commit -m "test(authnz): pin provider-credential encryption, owner scoping and RLS posture on both backends (TASK-13950.17)"`.
  - Open the PR against `dev`. Its body carries the Change summary and the Step 4 output. CI's `auth-sqlite` and `auth-postgres` shards must run the new files green.
  - Run the clean-subagent review.

### Task 2 (PR A2): Move the 13 credential modules behind aliases, and gate the aliases

**Files:**
- Create:
  - `tldw_Server_API/tests/lint/test_authnz_alias_imports.py`;
  - `tldw_Server_API/tests/AuthNZ_Unit/test_moved_authnz_module_aliases.py` (shard `auth-unit-m-z`);
  - `tldw_Server_API/app/core/Provider_Credentials/__init__.py` and `.../Provider_Credentials/repos/__init__.py`;
  - 13 alias files at the old paths.
- Move (`git mv`): the 13 credential rows of the Inventory.
- Modify (K1): every `.py` under `tldw_Server_API/app` and `Helper_Scripts` that names a moved module (109 files in the dry run); `tldw_Server_API/tests/Services/test_startup_auth.py:96,152,568`; and the three Task 1 files.
- Modify:
  - `pyproject.toml:891,892,893,901,914,915,921,927,939` (ruff `BLE001` per-file paths);
  - `tldw_Server_API/tests/lint/function_size_baseline.json` and `tldw_Server_API/tests/lint/direct_llm_adapter_calls_baseline.json` (re-key one entry each);
  - `tldw_Server_API/tests/lint/layering_baseline.txt` (delete `core.LLM_Calls -> core.AuthNZ: 9`);
  - `tldw_Server_API/tests/AuthNZ/unit/test_backend_detection_guardrail.py` and `tldw_Server_API/tests/AuthNZ/unit/test_guardrail_sql_audit.py` (scan both roots);
  - `tldw_Server_API/tests/lint/test_test_seam_env_reads_guarded.py`, only if #3331 merged.

**Interfaces:**
- Produces:
  - package `tldw_Server_API.app.core.Provider_Credentials`, with the same public and private names as the moved modules;
  - `MOVED: dict[str, str]` in `tests/lint/test_authnz_alias_imports.py` (old module to moved module, both relative to `tldw_Server_API.app.core.`). It is the single source for the alias tests and the gate. Task 4 extends it.
  - `violations(root: Path = APP_ROOT) -> list[str]` in the same file.

- [ ] **Step 1: Write the alias gate and the alias tests.**

`tests/lint/test_authnz_alias_imports.py` (`pytestmark = pytest.mark.unit`):
- Module docstring: the rule. App code may not use an old path from `MOVED`: at module or function level, absolute or relative, as `from ...AuthNZ import byok_runtime`, or as a string equal to an alias path (`importlib.import_module`). In-app `tests` packages are skipped, as in the layering contract. Fix: run K1, or import the module named in `MOVED`.
- `APP = "tldw_Server_API.app"`, and `MOVED` with the 13 entries `"AuthNZ.<m>": "Provider_Credentials.<m>"` and `"AuthNZ.repos.<r>": "Provider_Credentials.repos.<r>"`.
- `_names(node: ast.AST, module: str, is_package: bool) -> Iterator[str]` yields:
  - every `ast.Import` alias name;
  - for an `ast.ImportFrom`, its resolved base and `base.<name>` for each name. Resolve relative imports as `test_layering_contract._targets` does: `parts[: len(parts) - level + 1]`, where `parts` is the module path, minus the last part for a non-package.
  - every `str` constant.
- `violations(root)` walks `parse_tree(root)` (from `_ast_cache`) with `ast.walk`, names each module with `test_layering_contract._module_name`, and reports `"<relative path>:<line> uses <old>; use <new>"`.

```python
def test_app_code_imports_no_authnz_alias_path() -> None:
    problems = violations()
    assert not problems, "\n".join(problems)


def test_a_seeded_alias_import_fails(tmp_path: Path) -> None:
    files = {
        "core/X/a.py": f"from {APP}.core.AuthNZ.byok_runtime import y\n",
        "core/X/b.py": f"def f():\n    from {APP}.core.AuthNZ import byok_runtime\n",
        "core/X/c.py": "from ..AuthNZ.repos.user_provider_secrets_repo import R\n",
        "core/X/d.py": f"import importlib\nm = importlib.import_module('{APP}.core.AuthNZ.llm_provider_overrides')\n",
        "core/X/ok.py": f"from {APP}.core.Provider_Credentials.byok_runtime import y\nfrom {APP}.core.AuthNZ import settings\n",
        "core/X/tests/test_t.py": f"from {APP}.core.AuthNZ.byok_runtime import y\n",
    }
    for relative, source in files.items():
        path = tmp_path / "app" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    found = sorted(problem.split(":")[0] for problem in violations(tmp_path / "app"))

    assert found == ["core/X/a.py", "core/X/b.py", "core/X/c.py", "core/X/d.py"]
```

`tests/AuthNZ_Unit/test_moved_authnz_module_aliases.py` (`pytestmark = pytest.mark.unit`, `APP = "tldw_Server_API.app.core."`). It imports `MOVED` from the gate and `REPO_ROOT` from `tests.lint._ast_cache`.

```python
@pytest.mark.parametrize(("old", "new"), sorted(MOVED.items()))
def test_old_authnz_path_is_the_moved_module(old: str, new: str) -> None:
    moved = importlib.import_module(APP + new)
    assert importlib.import_module(APP + old) is moved
    parent, _, leaf = (APP + old).rpartition(".")
    assert getattr(importlib.import_module(parent), leaf) is moved
    assert Path(moved.__file__).as_posix().endswith("core/" + new.replace(".", "/") + ".py")


def test_patching_the_old_path_patches_the_code_that_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Provider_Credentials import provider_credential_runtime as moved

    sentinel = object()
    monkeypatch.setattr(APP + "AuthNZ.provider_credential_runtime.load_server_config_snapshot", sentinel)
    assert moved.load_server_config_snapshot is sentinel


@pytest.mark.parametrize(("old", "new"), sorted(MOVED.items()))
def test_old_path_aliases_when_imported_first(old: str, new: str) -> None:
    code = (
        "import importlib, sys\n"
        f"first = importlib.import_module({APP + old!r})\n"
        f"assert first is importlib.import_module({APP + new!r})\n"
        f"assert sys.modules[{APP + old!r}] is sys.modules[{APP + new!r}]\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-2000:]
```

- [ ] **Step 2: Run them.** `PYTHONPATH=. TEST_MODE=true $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t17-t2a" tldw_Server_API/tests/lint/test_authnz_alias_imports.py tldw_Server_API/tests/AuthNZ_Unit/test_moved_authnz_module_aliases.py`. Expected: `28 failed, 1 passed`.
  - The gate reports 233 uses on 0d270b00e4, plus #3191's.
  - The 27 alias cases fail with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Provider_Credentials'`.
  - The seeded self-test passes.

- [ ] **Step 3: Move, alias, codemod.** Create K1 first, then:

```bash
bash -c 'set -euo pipefail
A=tldw_Server_API/app/core/AuthNZ; P=tldw_Server_API/app/core/Provider_Credentials
TOP="byok_config byok_helpers byok_rotation byok_runtime byok_testing user_provider_secrets provider_credential_runtime llm_provider_overrides"
REPOS="user_provider_secrets_repo org_provider_secrets_repo llm_provider_overrides_repo byok_oauth_state_repo byok_validation_runs_repo"
mkdir -p $P/repos
for m in $TOP; do git mv $A/$m.py $P/$m.py; done
for m in $REPOS; do git mv $A/repos/$m.py $P/repos/$m.py; done
printf "%s\n" "\"\"\"Provider credentials: BYOK secrets, provider overrides and the per-call credential runtime (TASK-13950.17).\"\"\"" > $P/__init__.py
printf "%s\n" "\"\"\"Repositories for the provider-credential tables in the AuthNZ database (TASK-13950.17).\"\"\"" > $P/repos/__init__.py
git ls-files -z "tldw_Server_API/app/*.py" "Helper_Scripts/*.py" | xargs -0 perl -pi "$TMPDIR/t17_codemod.pl"
perl -pi "$TMPDIR/t17_codemod.pl" tldw_Server_API/tests/Services/test_startup_auth.py tldw_Server_API/tests/helpers/provider_credential_invariants.py tldw_Server_API/tests/AuthNZ_SQLite/test_provider_credentials_storage_sqlite.py tldw_Server_API/tests/AuthNZ_Postgres/test_provider_credentials_storage_pg.py
alias_to() {  # <old file> <new dotted module>
  cat > "$1" <<PY
"""Moved to $2 (TASK-13950.17).

An alias, not a re-export: the old name *is* the moved module, so old imports and
monkeypatch targets (string or module object) reach the code that runs.
"""

import sys

from ${2%.*} import ${2##*.} as _moved

sys.modules[__name__] = _moved
PY
}
for m in $TOP; do alias_to $A/$m.py tldw_Server_API.app.core.Provider_Credentials.$m; done
for m in $REPOS; do alias_to $A/repos/$m.py tldw_Server_API.app.core.Provider_Credentials.repos.$m; done
git add -A tldw_Server_API/app/core/AuthNZ tldw_Server_API/app/core/Provider_Credentials'
```

Python's import system returns `sys.modules[name]` after it runs a module, and binds that object on the parent package. So the alias is the moved module, whichever path a process imports first. The independent dry run checked all 13 modules in fresh interpreters in both orders, including `importlib.reload`. This is pinned by `test_old_path_aliases_when_imported_first`.

- [ ] **Step 4: Update the path-keyed config, baselines and guardrails.**
  - **Inventory of path-keyed guards, before any re-key:** `git grep -nE "core[/.]AuthNZ[/.](repos[/.])?(byok_(config|helpers|rotation|runtime|testing)|user_provider_secrets|provider_credential_runtime|llm_provider_overrides|org_provider_secrets_repo|byok_oauth_state_repo|byok_validation_runs_repo)" -- tldw_Server_API/tests/lint tldw_Server_API/tests/CI Helper_Scripts/ci`. Expected:
    - `function_size_baseline.json` (the `byok_runtime:resolve_byok_credentials` entry);
    - `direct_llm_adapter_calls_baseline.json` (`core/AuthNZ/byok_testing.py`, from `.13` PR A);
    - if #3331 merged, `test_test_seam_env_reads_guarded.py` (`"core/AuthNZ/byok_testing.py",`).
    Re-key each one below. Afterwards, the same grep prints only the two `moved from` reasons.
  - **Re-key the two ceiling baselines**, keeping their numbers. Run `PYTHONPATH=. $PY - <<'EOF'` with this body, then `EOF`:

```python
from pathlib import Path
from tldw_Server_API.tests.lint import _ceilings

def rekey(path: str, old: str, new: str) -> None:
    baseline = Path(path)
    entries = _ceilings.load(baseline)
    if old in entries:
        entries[new] = {**entries.pop(old), "reason": f"moved from {old} (TASK-13950.17)"}
        _ceilings.dump(baseline, entries)

rekey("tldw_Server_API/tests/lint/function_size_baseline.json",
      "tldw_Server_API.app.core.AuthNZ.byok_runtime:resolve_byok_credentials",
      "tldw_Server_API.app.core.Provider_Credentials.byok_runtime:resolve_byok_credentials")
rekey("tldw_Server_API/tests/lint/direct_llm_adapter_calls_baseline.json",
      "core/AuthNZ/byok_testing.py", "core/Provider_Credentials/byok_testing.py")
```

  - **Seam guard (#3331):** change its line to `"core/Provider_Credentials/byok_testing.py",`, keeping the ALLOWLIST sorted.
  - **pyproject:** `perl -pi -e 's#tldw_Server_API/app/core/AuthNZ/((?:repos/)?(?:byok_config|byok_helpers|byok_rotation|byok_runtime|byok_testing|user_provider_secrets|provider_credential_runtime|llm_provider_overrides|user_provider_secrets_repo|org_provider_secrets_repo|llm_provider_overrides_repo|byok_oauth_state_repo|byok_validation_runs_repo))\.py#tldw_Server_API/app/core/Provider_Credentials/$1.py#' pyproject.toml`. Expected: 9 lines change.
  - **Guardrails.** In both `test_backend_detection_guardrail.py` and `test_guardrail_sql_audit.py`, replace the single `authnz_core` root with `roots = [project_root / "app" / "core" / name for name in ("AuthNZ", "Provider_Credentials")]` and `files = [path for root in roots for path in root.rglob("*.py")]`. Iterate over `files`, and keep each `exists()` assertion, now for every root. Add `assert project_root / "app" / "core" / "Provider_Credentials" / "byok_runtime.py" in files`, so that losing the scope fails. The allowed-files set of the SQL audit is unchanged. The dry runs found no forbidden pattern in the moved code.
  - **Layering:** run `PYTHONPATH=. $PY -m tldw_Server_API.tests.lint.test_layering_contract --tighten`. Expected `git diff tldw_Server_API/tests/lint/layering_baseline.txt`: exactly one deleted line, `-core.LLM_Calls -> core.AuthNZ: 9  # bump: gateway._resolve_base_url_override, moved from chat_service, keeps its byok_helpers lookup; TASK-13950.17 cuts LLM_Calls -> AuthNZ to 0 (TASK-13950.13)`, and nothing else. The dry runs, made before `.13`, deleted `core.LLM_Calls -> core.AuthNZ: 7`; pairs and statements went from 262/683 to 261/676.

- [ ] **Step 5: Run the move checks.**
  - `PYTHONPATH=. TEST_MODE=true $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t17-t2b" tldw_Server_API/tests/lint/test_authnz_alias_imports.py tldw_Server_API/tests/AuthNZ_Unit/test_moved_authnz_module_aliases.py tldw_Server_API/tests/AuthNZ/unit/test_backend_detection_guardrail.py tldw_Server_API/tests/AuthNZ/unit/test_guardrail_sql_audit.py tldw_Server_API/tests/Services/test_startup_auth.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_crypto.py`. Expected: all pass. In the dry runs these were 2 + 27 + 2 + 17 + 296 passed, 1 skipped.
  - Task 1's command. Expected: `6 passed`, now through the new paths.
  - `.13`'s guards and gateway tests: `PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t17-t2c" tldw_Server_API/tests/lint/test_llm_calls_import_isolation.py tldw_Server_API/tests/lint/test_direct_llm_adapter_calls_ratchet.py tldw_Server_API/tests/LLM_Calls/test_gateway_module.py`. Expected: all pass.
  - K3. Expected: no output.

- [ ] **Step 6: Confirm that AC4 holds for the right reason.** `git grep -nE "core\.AuthNZ|import_module\(.*AuthNZ" -- 'tldw_Server_API/app/core/LLM_Calls/*.py'`. Expected: no output. The alias gate in Step 5 shows that no app module uses an alias path.

- [ ] **Step 7: Commit.** `git add -A tldw_Server_API Helper_Scripts pyproject.toml`, then `git commit -m "refactor(authnz): move BYOK and provider credentials to core/Provider_Credentials behind module aliases (TASK-13950.17)"`

### Task 3 (PR A2): Lock the cut, gate the guards, keep the CI posture, then open A2

**Files:**
- Modify: `tldw_Server_API/tests/lint/test_layering_contract.py`:
  - `SIBLING_BANS` (line 144 on 0d270b00e4);
  - `FIXES["sibling ban"]` (line 153);
  - the "Rules." docstring bullet (line 64);
  - a new self-test at the end.
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py` (two new tests)
- Modify: `.github/workflows/backend-required.yml` (the "Enforce CI contracts and code ratchets" step)
- Modify: `Helper_Scripts/ci/path_classifier.py` (`E2E_BACKEND_GLOBS`) and `.github/workflows/coverage-required.yml:229` (the "AuthNZ coverage floor" step)

**Interfaces:**
- Produces:
  - `SIBLING_BANS` gains `("core.AuthNZ", "core.Chat")` and `("core.AuthNZ", "core.TTS")`. Task 6 adds `("core.AuthNZ", "core.LLM_Calls")`.
  - `E2E_BACKEND_GLOBS` gains `"tldw_Server_API/app/core/Provider_Credentials/**"`.
  - `test_moved_provider_credential_code_keeps_authnz_ci_posture`, with module-level lists `MOVED_E2E_PATHS` and `AUTHNZ_FLOOR_TARGETS`, which Tasks 4 and 5 extend.

- [ ] **Step 1: Write the failing tests.**

```python
# tests/lint/test_layering_contract.py
def test_authnz_imports_no_llm_calls_chat_or_tts_but_provider_credentials_may(tmp_path: Path) -> None:
    """TASK-13950.17: provider credentials left AuthNZ, so AuthNZ needs none of the LLM stack."""
    files = {
        "core/AuthNZ/x.py": f"""
            from {APP}.core.Chat import a
            from {APP}.core.Provider_Credentials import c
            def f():
                from {APP}.core.TTS import d
        """,
        "core/Provider_Credentials/y.py": f"from {APP}.core.LLM_Calls import e\nfrom {APP}.core.AuthNZ import f\n",
        "core/LLM_Calls/z.py": f"from {APP}.core.Provider_Credentials import g\n",
    }
    root = _seed(tmp_path / "app", files)

    assert {pair: rule(*pair) for pair in measure(root)} == {
        ("core.AuthNZ", "core.Chat"): "sibling ban",
        ("core.AuthNZ", "core.TTS"): "sibling ban",
    }
```

```python
# tests/CI/test_license_first_workflow_contracts.py (yaml and REPO_ROOT are already imported there)
MOVED_E2E_PATHS = [
    "tldw_Server_API/app/core/Provider_Credentials/byok_runtime.py",
    "tldw_Server_API/app/core/Provider_Credentials/repos/user_provider_secrets_repo.py",
]
AUTHNZ_FLOOR_TARGETS = {"tldw_Server_API/app/core/AuthNZ", "tldw_Server_API/app/core/Provider_Credentials"}


@pytest.mark.unit
def test_moved_provider_credential_code_keeps_authnz_ci_posture() -> None:
    """TASK-13950.17: code moved out of core/AuthNZ keeps AuthNZ's e2e trigger and coverage floor."""
    import shlex

    from Helper_Scripts.ci.path_classifier import classify_paths

    assert [path for path in MOVED_E2E_PATHS if not classify_paths([path])["e2e_changed"]] == []  # nosec B101
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/coverage-required.yml").read_text())
    floor = next(
        step for step in workflow["jobs"]["coverage-required"]["steps"] if step.get("name") == "AuthNZ coverage floor"
    )
    words = shlex.split(floor["run"].replace("\\\n", " "), comments=True)
    assert {word.split("=", 1)[1] for word in words if word.startswith("--cov=")} >= AUTHNZ_FLOOR_TARGETS  # nosec B101
    assert "--cov-fail-under=35" in words  # nosec B101


@pytest.mark.unit
@pytest.mark.parametrize(
    "guard",
    [
        "tldw_Server_API/tests/lint/test_authnz_alias_imports.py",
        "tldw_Server_API/tests/AuthNZ/unit/test_backend_detection_guardrail.py",
        "tldw_Server_API/tests/AuthNZ/unit/test_guardrail_sql_audit.py",
        "tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py",
    ],
)
def test_backend_required_enforces_provider_credential_guards(guard: str) -> None:
    """TASK-13950.17: the alias gate, the widened AuthNZ guardrails and the CI-posture pins run in a required step."""
    import shlex

    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/backend-required.yml").read_text())
    gate = next(
        step for step in workflow["jobs"]["backend-required"]["steps"]
        if step.get("name") == "Enforce CI contracts and code ratchets"
    )
    assert not gate.get("continue-on-error", False)  # nosec B101
    assert guard in shlex.split(gate["run"].replace("\\\n", " "), comments=True)  # nosec B101
```

- [ ] **Step 2: Run them.** `PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t17-t3a" tldw_Server_API/tests/lint/test_layering_contract.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py -k "authnz_imports_no or ci_posture or provider_credential_guards"`. Use a single `-k`, because pytest keeps only the last one. Expected: 5 failed and 1 passed.
  - The 5 failures: the bans are missing; the e2e and `--cov` pins fail; and 3 guard paths are missing from the step.
  - The 1 pass: `test_license_first_workflow_contracts.py` is already in the step.

- [ ] **Step 3: Implement.**
  - **`SIBLING_BANS`:** the four-pair frozenset `("core.LLM_Calls", "core.Chat")`, `("core.LLM_Calls", "core.AuthNZ")`, `("core.AuthNZ", "core.Chat")`, `("core.AuthNZ", "core.TTS")`, one pair per line.
  - **`FIXES["sibling ban"]`:** `"core.LLM_Calls must not depend on core.Chat or core.AuthNZ, and core.AuthNZ must not depend on the LLM stack (provider credentials live in core.Provider_Credentials, LLM budgets in core.Usage): have the caller pass it in."`
  - **Docstring:** the bullet becomes `* is a sibling ban ("sibling ban"): core.LLM_Calls -> core.Chat and -> core.AuthNZ; core.AuthNZ -> core.Chat and -> core.TTS (TASK-13950.17).`
  - **backend-required.yml:** after `tldw_Server_API/tests/lint/test_core_to_api_import_boundary.py \`, append `tldw_Server_API/tests/lint/test_authnz_alias_imports.py \`, `tldw_Server_API/tests/AuthNZ/unit/test_backend_detection_guardrail.py \` and `tldw_Server_API/tests/AuthNZ/unit/test_guardrail_sql_audit.py \`. Under the step's flags (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `-p pytest_asyncio.plugin`), the three ran in about 42 s with Docker down.
  - **Path classifier:** append `"tldw_Server_API/app/core/Provider_Credentials/**"` to `E2E_BACKEND_GLOBS`.
  - **Coverage floor:** `coverage-required.yml:229` becomes `--cov=tldw_Server_API/app/core/AuthNZ --cov=tldw_Server_API/app/core/Provider_Credentials --cov-report=term --cov-fail-under=35`. Measured with the step's flags:
    - base: 42.01% of 28,570 statements;
    - the moved 13 modules: 4,241 statements at 58.38%;
    - AuthNZ alone after the move: 39.15%;
    - AuthNZ plus Provider_Credentials: the base figure.
    Both stay over 35, but only the second keeps the moved code under the floor.

- [ ] **Step 4: Run them again.** Same command as Step 2. Expected: all pass. Then run `PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t17-t3b" tldw_Server_API/tests/lint/test_layering_contract.py tldw_Server_API/tests/lint/test_core_to_api_import_boundary.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py tldw_Server_API/tests/CI/test_path_classifier.py tldw_Server_API/tests/CI/test_required_workflow_contracts.py`. Expected: all pass. `test_backend_required_enforces_layering_ratchets` keeps the changed contract in the gate (AC5).

- [ ] **Step 5: Commit.** `git add -A tldw_Server_API Helper_Scripts .github`, then `git commit -m "test(lint): ban AuthNZ -> Chat/TTS, gate the aliases and keep moved credentials under AuthNZ's e2e trigger and coverage floor (TASK-13950.17)"`

- [ ] **Step 6: Verify A2.**
  - K2: both steps `exit 0`, or base-only failures as described in K2.
  - K4 with `A`: the `comm` loop prints nothing. This is the "both shards" requirement locally; the CI shards in Step 7 are the full run.
  - mypy (advisory; `continue-on-error` in backend-required): `bash -c 'uvx --from mypy mypy --follow-imports=silent --ignore-missing-imports $(git diff --name-only --diff-filter=d origin/dev...HEAD -- "tldw_Server_API/*.py")' | tail -3`. No error may cite a `Provider_Credentials` import or a moved name. The dry run found 32 errors in the moved files, equal to the same 13 files on the base.
  - Bandit: `uvx bandit -q -r tldw_Server_API/app/core/Provider_Credentials | grep -c '>> Issue'` must equal the same count in `$TMPDIR/t17-base` for `tldw_Server_API/app/core/AuthNZ/{byok_config,byok_helpers,byok_rotation,byok_runtime,byok_testing,user_provider_secrets,provider_credential_runtime,llm_provider_overrides}.py` plus `tldw_Server_API/app/core/AuthNZ/repos/{user_provider_secrets_repo,org_provider_secrets_repo,llm_provider_overrides_repo,byok_oauth_state_repo,byok_validation_runs_repo}.py` (zsh brace expansion). The dry run found 0 = 0. Then remove the base worktree.
  - OpenAPI drift check: not needed. No schema or route moves, and endpoint modules change only on import lines.

- [ ] **Step 7: Open PR A2** against `dev` (`gh pr create --base dev`). The body carries:
  - `## Change summary`;
  - the edge tables;
  - K3's empty output, and the instruction to read the diff with `git diff -B -M origin/dev...HEAD`;
  - the K4 `comm` result and its head summary lines;
  - the layering report: 261 pairs, statements down by 9, and the package SCC from 102/130 to 103/131. Add the folded figure below, which equals the base.
  - the coverage-floor numbers;
  - the porting recipe for #3187 and #2821, also posted as a comment on each;
  - the note that the advisory ruff step grows by about 77 `I001` errors (renamed entries in sorted import blocks), and that this PR does not re-sort them because K3 forbids it;
  - "tests keep old paths through aliases; follow-up filed at close-out".

  Then run the clean-subagent `/superpowers:requesting-code-review`. In CI, these `Full Suite shard` jobs must be green, or fail only on tests that fail on `dev` too: `auth-postgres`, `auth-sqlite`, `auth-unit-a-l`, `auth-unit-m-z`, `auth-core-*` and `auth-integration-*` (Postgres service, `TLDW_TEST_POSTGRES_REQUIRED=1`).

  The folded SCC treats Provider_Credentials as part of AuthNZ. On the dry-run tree it gives `{'module_level': 102, 'with_function_level': 130}`. The +1 is the new node's bookkeeping, not a new cycle: Provider_Credentials imports SCC members, and 26 SCC packages import it. Run `PYTHONPATH=. $PY - <<'EOF'` with this body, then `EOF`:

```python
from tldw_Server_API.tests.lint.test_layering_contract import _largest_scc, package_of, scan
fold = {"core.Provider_Credentials": "core.AuthNZ"}
records = scan()
def edges(pairs):
    return {(fold.get(a, a), fold.get(b, b)) for a, b in pairs if fold.get(a, a) != fold.get(b, b)}
module_level = edges((package_of(r.module), r.target) for r in records if not r.function_level)
every = edges((package_of(r.module), r.target) for r in records)
print({"module_level": _largest_scc(module_level), "with_function_level": _largest_scc(every)})
```

### Task 4 (PR B): LLM budget modules move to core/Usage

**Files:**
- Move: `core/AuthNZ/llm_budget_guard.py` to `core/Usage/llm_budget_guard.py`, and `core/AuthNZ/llm_budget_middleware.py` to `core/Usage/llm_budget_middleware.py`
- Create: 2 alias files at the old paths (the Task 2 template)
- Modify: `api/v1/endpoints/chat.py:320`, `api/v1/endpoints/prompts.py:46`, `api/v1/endpoints/rpg.py:42`, `main.py:2476`
- Modify: `tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json:4065,6556` (`"module"` of `enforce_llm_budget`)
- Modify: `tests/lint/test_authnz_alias_imports.py` (`MOVED`)
- Modify: `tests/CI/test_license_first_workflow_contracts.py` (`MOVED_E2E_PATHS`, `AUTHNZ_FLOOR_TARGETS`), `Helper_Scripts/ci/path_classifier.py` and `.github/workflows/coverage-required.yml:229`
- Modify: both AuthNZ guardrails (`files`)
- Modify: `test_test_seam_env_reads_guarded.py`, only if #3331 merged (its `core/AuthNZ/llm_budget_guard.py` and `core/AuthNZ/llm_budget_middleware.py` entries)

**Interfaces:**
- Consumes: `MOVED` (Task 2), the `alias_to` template (Task 2, Step 3), and `MOVED_E2E_PATHS` and `AUTHNZ_FLOOR_TARGETS` (Task 3).
- Produces: `tldw_Server_API.app.core.Usage.llm_budget_guard.enforce_llm_budget` and `tldw_Server_API.app.core.Usage.llm_budget_middleware.LLMBudgetMiddleware`, the same objects as at the old paths.

- [ ] **Step 1: Extend the tests.**
  - Add `"AuthNZ.llm_budget_guard": "Usage.llm_budget_guard"` and `"AuthNZ.llm_budget_middleware": "Usage.llm_budget_middleware"` to `MOVED`.
  - Add the two `core/Usage/llm_budget_*.py` paths to `MOVED_E2E_PATHS`.
  - Add `"tldw_Server_API.app.core.Usage.llm_budget_guard"` and `"tldw_Server_API.app.core.Usage.llm_budget_middleware"` to `AUTHNZ_FLOOR_TARGETS`.
  - In both guardrails, add `assert project_root / "app" / "core" / "Usage" / "llm_budget_guard.py" in files`.
- [ ] **Step 2: Run them.** `PYTHONPATH=. TEST_MODE=true $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t17-t4a" tldw_Server_API/tests/AuthNZ_Unit/test_moved_authnz_module_aliases.py tldw_Server_API/tests/lint/test_authnz_alias_imports.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py -k "authnz or alias or ci_posture" tldw_Server_API/tests/AuthNZ/unit/test_backend_detection_guardrail.py tldw_Server_API/tests/AuthNZ/unit/test_guardrail_sql_audit.py`. Expected failures:
  - 4 alias cases, the 2 new parameters of each parametrized test;
  - the alias gate, which now flags the 4 budget importers;
  - the CI-posture test;
  - the 2 guardrails.
- [ ] **Step 3: Implement.**
  - Run `git mv` for both files into `tldw_Server_API/app/core/Usage/`.
  - Run `git ls-files -z 'tldw_Server_API/app/*.py' | xargs -0 perl -pi -e 's/\bcore\.AuthNZ\.(llm_budget_guard|llm_budget_middleware)(?![A-Za-z0-9_])/core.Usage.$1/g'`. It rewrites the 4 importers.
  - Write the two aliases: in a new `bash -c '...'`, paste the `alias_to` function from Task 2, Step 3, followed by `for m in llm_budget_guard llm_budget_middleware; do alias_to tldw_Server_API/app/core/AuthNZ/$m.py tldw_Server_API.app.core.Usage.$m; done`.
  - In the privilege snapshot, `perl -pi -e 's/"tldw_Server_API\.app\.core\.AuthNZ\.llm_budget_guard"/"tldw_Server_API.app.core.Usage.llm_budget_guard"/g'`. That rewrites 2 lines; `enforce_llm_budget.__module__` is now the Usage path.
  - Append `"tldw_Server_API/app/core/Usage/llm_budget_*.py"` to `E2E_BACKEND_GLOBS`.
  - Add `--cov=tldw_Server_API.app.core.Usage.llm_budget_guard --cov=tldw_Server_API.app.core.Usage.llm_budget_middleware` to the floor step. Coverage accepts importable module names, and a module name measures that file only. This was checked with pytest-cov 7.1 and `--cov=<dotted module>`: the report listed exactly those files.
  - Add the two `core/Usage/llm_budget_*.py` paths to `files` in both guardrails.
- [ ] **Step 4: Run them again.** The Step 2 command, plus `tldw_Server_API/tests/AuthNZ/unit/test_llm_budget_middleware.py tldw_Server_API/tests/AuthNZ/unit/test_llm_budget_guard_state_failure.py tldw_Server_API/tests/AuthNZ_Unit/test_auth_governor_budget.py tldw_Server_API/tests/Privileges/test_privilege_introspection.py tldw_Server_API/tests/Config/test_route_and_cors_guards.py`. Expected: all pass. The dry run showed `test_privilege_registry_snapshot_matches_live_app` failing until the snapshot edit, then 10 passed. Then check with K3's budget-module variant: it should print nothing.
- [ ] **Step 5: Commit.** `git add -A tldw_Server_API Helper_Scripts .github`, then `git commit -m "refactor(usage): move the LLM budget guard and middleware to core/Usage behind aliases (TASK-13950.17)"`

### Task 5 (PR B): The AuthNZ scheduler moves to services, and the CLI stops starting it

**Files:**
- Move: `core/AuthNZ/scheduler.py` to `services/authnz_scheduler.py`
- Delete: `core/Scheduler/core/AuthNZ/__init__.py` and `core/Scheduler/core/AuthNZ/scheduler.py`. This is a test-only re-export; keeping it would add a new `core.Scheduler -> services` upward pair.
- Modify: `core/AuthNZ/initialize.py`:
  - delete the import at `:45`;
  - in `start_services()` (`:1356`), delete the `# Start scheduler` comment, `await start_authnz_scheduler()` and `print("✅ Scheduler started")` (`:1361-1363`), and print `"ℹ️  Scheduled AuthNZ jobs run inside the API server."` in their place;
  - in the `--test-setup` help text (`:1550`), "and auto-starts background services)" becomes "and runs the monitoring check)".
- Modify (`services.authnz_scheduler` replaces `core.AuthNZ.scheduler`):
  - app: `services/startup_recurring_schedulers.py:594,600`;
  - in-app test: `core/Scheduler/tests/test_security_fixes.py:288` (the relative `from ..core.AuthNZ.scheduler import AuthNZScheduler`, which becomes the absolute services import) and `:298`;
  - tests: `tldw_Server_API/tests/conftest.py:75`, `tests/AuthNZ/conftest.py:36`, `tests/AuthNZ/integration/test_monitoring_metrics_summary.py:16,126`, `tests/AuthNZ/integration/test_scheduler_auth_failure_monitor_postgres.py:11`, `tests/AuthNZ_SQLite/test_scheduler_auth_failure_monitor_sqlite.py:10`, `tests/Privileges/test_privilege_snapshot_retention.py:10`, `tests/performance/test_authnz_multiuser_sqlite_load.py:72` and `tests/Services/test_shutdown_coordinator.py:121,122`.
- Modify: `MOVED_E2E_PATHS`, `AUTHNZ_FLOOR_TARGETS`, `path_classifier.py`, the floor step, and both guardrails' `files`. Also `.39`'s scheduler cap, only if it merged.

**Interfaces:**
- Produces: `tldw_Server_API.app.services.authnz_scheduler`, with `AuthNZScheduler`, `get_authnz_scheduler`, `start_authnz_scheduler`, `stop_authnz_scheduler` and `reset_authnz_scheduler` unchanged. `tldw_Server_API.app.core.AuthNZ.scheduler` no longer exists. `core.AuthNZ -> services` stays at 1.

- [ ] **Step 1: Write the failing tests.**
  - Add `"tldw_Server_API/app/services/authnz_scheduler.py"` to `MOVED_E2E_PATHS`, and `"tldw_Server_API.app.services.authnz_scheduler"` to `AUTHNZ_FLOOR_TARGETS`.
  - Add `assert project_root / "app" / "services" / "authnz_scheduler.py" in files` to both guardrails.
  - Add to `test_moved_authnz_module_aliases.py`:

```python
def test_root_conftest_disables_the_moved_authnz_scheduler() -> None:
    import importlib.util

    from tldw_Server_API.app.services import authnz_scheduler

    assert authnz_scheduler.start_authnz_scheduler.__name__ == "_noop"
    assert importlib.util.find_spec("tldw_Server_API.app.core.AuthNZ.scheduler") is None


def test_the_authnz_initialize_cli_imports_no_services() -> None:
    import ast

    tree = ast.parse((REPO_ROOT / "tldw_Server_API/app/core/AuthNZ/initialize.py").read_text(encoding="utf-8"))
    names = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    names |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert sorted(name for name in names if name.startswith("tldw_Server_API.app.services")) == []
```

- [ ] **Step 2: Run them.** `PYTHONPATH=. TEST_MODE=true $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t17-t5a" tldw_Server_API/tests/AuthNZ_Unit/test_moved_authnz_module_aliases.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py tldw_Server_API/tests/AuthNZ/unit/test_backend_detection_guardrail.py tldw_Server_API/tests/AuthNZ/unit/test_guardrail_sql_audit.py -k "scheduler or initialize_cli or ci_posture or guardrail or backend_probing"`. Expected failures:
  - the conftest test, with `ModuleNotFoundError: No module named 'tldw_Server_API.app.services.authnz_scheduler'`;
  - the CI-posture test;
  - the 2 guardrails.

  `test_the_authnz_initialize_cli_imports_no_services` passes for now. It guards Step 3: it fails as soon as the codemod points `initialize.py:45` at `services`, and stays failing until the CLI lines go.
- [ ] **Step 3: Implement.**
  - Run `git mv tldw_Server_API/app/core/AuthNZ/scheduler.py tldw_Server_API/app/services/authnz_scheduler.py` and `git rm -r tldw_Server_API/app/core/Scheduler/core/AuthNZ`.
  - Run `git ls-files -z 'tldw_Server_API/*.py' | xargs -0 perl -pi -e 's/tldw_Server_API\.app\.core\.AuthNZ\.scheduler(?![A-Za-z0-9_])/tldw_Server_API.app.services.authnz_scheduler/g; s/from tldw_Server_API\.app\.core\.AuthNZ import scheduler as/from tldw_Server_API.app.services import authnz_scheduler as/g'`.
  - Hand-edit `test_security_fixes.py:288`.
  - Make the `initialize.py` edits under Files. No import of `services` remains. The CLI's start never ran a job: the scheduler's first triggers are at least 5 minutes out, the CLI's event loop closes about 3 s later, and only `--test-setup` or an interactive "y" ever reached it. The entrypoints (`Dockerfiles/entrypoints/tldw-app-first-run.sh:402`, `Helper_Scripts/quickstart.sh:86`, the UAT runners) pass `--non-interactive`, which never starts services.
  - Append `"tldw_Server_API/app/services/authnz_scheduler.py"` to `E2E_BACKEND_GLOBS`.
  - Add `--cov=tldw_Server_API.app.services.authnz_scheduler` to the floor step.
  - Add the file to both guardrails' `files`.
- [ ] **Step 4: Check that no old path is left.**
  - `git grep -nE "core\.AuthNZ\.scheduler|AuthNZ import scheduler|\.core\.AuthNZ\.scheduler|start_authnz_scheduler" -- tldw_Server_API/app/core`. Expected: no output.
  - `git add -A tldw_Server_API && git diff --cached -M --name-status | grep authnz_scheduler`. Expected: `R100	tldw_Server_API/app/core/AuthNZ/scheduler.py	tldw_Server_API/app/services/authnz_scheduler.py`.
- [ ] **Step 5: Run the tests.** `PYTHONPATH=. TEST_MODE=true $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t17-t5" tldw_Server_API/tests/AuthNZ_Unit/test_moved_authnz_module_aliases.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py tldw_Server_API/tests/AuthNZ/unit/test_backend_detection_guardrail.py tldw_Server_API/tests/AuthNZ/unit/test_guardrail_sql_audit.py tldw_Server_API/tests/Services/test_shutdown_coordinator.py tldw_Server_API/tests/AuthNZ_SQLite/test_scheduler_auth_failure_monitor_sqlite.py tldw_Server_API/tests/Privileges/test_privilege_snapshot_retention.py "tldw_Server_API/app/core/Scheduler/tests/test_security_fixes.py::TestSQLInjectionFixes::test_parameterized_queries"`.
  - Expected: all pass. The last test is not in `testpaths`, so it is named explicitly.
  - Then run `PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t17-t5b" tldw_Server_API/tests/lint/test_layering_contract.py`. Expected: all pass, with `layering_baseline.txt` untouched.
- [ ] **Step 6: Commit.** `git add -A tldw_Server_API Helper_Scripts .github`, then `git commit -m "refactor(services): move the AuthNZ scheduler to services/authnz_scheduler.py; the initialize CLI no longer starts it (TASK-13950.17)"`

### Task 6 (PR B): Ban AuthNZ -> LLM_Calls, ratchet the indirect AuthNZ load, update the docs, then open B and close out

**Files:**
- Modify: `tldw_Server_API/tests/lint/test_layering_contract.py` (`SIBLING_BANS`, the docstring bullet, the self-test)
- Modify: `tldw_Server_API/tests/lint/test_llm_calls_import_isolation.py` (`.13`'s; the laundering ratchet)
- Modify (docs):
  - `tldw_Server_API/app/core/AuthNZ/README.md:33,94,96,101`;
  - `Docs/Code_Documentation/Guides/AuthNZ_Code_Guide.md:20` and `Docs/Code_Documentation/Data_Flow_Atlas.md:718` (both mirrored);
  - `Docs/Architecture/llm-providers.md:20` and `Docs/Operations/Rate_Limits_Troubleshooting.md:16`;
  - the `Docs/Published/` mirror.

  Historical docs that cite moved paths stay as they are: ADR-052, the CodeQL evidence doc, `BYOK-PRD.md` and design docs.

**Interfaces:**
- Consumes: `loaded_after_import(module: str) -> list[str]` from `.13`'s `test_llm_calls_import_isolation.py`.
- Produces: `AUTHNZ_LOADED_BY_LLM_CALLS: frozenset[str]` and `authnz_load_drift(loaded: Iterable[str], allowed: frozenset[str]) -> list[str]` in the same file.

- [ ] **Step 1: Make the self-tests fail.**
  - In `test_authnz_imports_no_llm_calls_chat_or_tts_but_provider_credentials_may`, add `from {APP}.core.LLM_Calls import b` to `core/AuthNZ/x.py`, and `("core.AuthNZ", "core.LLM_Calls"): "sibling ban"` to the expected dict.
  - Append the laundering ratchet to `test_llm_calls_import_isolation.py`:

```python
AUTHNZ = "tldw_Server_API.app.core.AuthNZ"
# TASK-13950.17: the AuthNZ modules the LLM gateway still loads through core.Provider_Credentials.
# Module-keyed and exact, so it fails on growth and on a stale entry, and it fails if any path
# loads an old-path alias. Shrink it as the inversion follow-up removes them.
AUTHNZ_LOADED_BY_LLM_CALLS: frozenset[str] = frozenset()  # seeded in Step 2


def authnz_load_drift(loaded: Iterable[str], allowed: frozenset[str]) -> list[str]:
    now = {m for m in loaded if m == AUTHNZ or m.startswith(AUTHNZ + ".")}
    return [f"new: {m}" for m in sorted(now - allowed)] + [f"stale: {m}" for m in sorted(allowed - now)]


def test_llm_calls_loads_only_the_known_authnz_modules() -> None:
    drift = authnz_load_drift(loaded_after_import("tldw_Server_API.app.core.LLM_Calls.gateway"), AUTHNZ_LOADED_BY_LLM_CALLS)
    assert drift == [], "LLM_Calls -> AuthNZ indirect load changed (TASK-13950.17):\n" + "\n".join(drift)


def test_the_authnz_load_ratchet_sees_growth_and_stale_entries() -> None:
    allowed = frozenset({f"{AUTHNZ}.settings", f"{AUTHNZ}.gone"})
    loaded = [f"{AUTHNZ}.settings", f"{AUTHNZ}.database", "tldw_Server_API.app.core.LLM_Calls"]
    assert authnz_load_drift(loaded, allowed) == [f"new: {AUTHNZ}.database", f"stale: {AUTHNZ}.gone"]
```

  - Run `PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t17-t6a" tldw_Server_API/tests/lint/test_layering_contract.py tldw_Server_API/tests/lint/test_llm_calls_import_isolation.py -k "authnz"`. Expected: 2 failed and 1 passed. The layering dict is missing the pair; the ratchet's seed is empty; the self-test passes.
- [ ] **Step 2: Implement.**
  - Add `("core.AuthNZ", "core.LLM_Calls")` to `SIBLING_BANS`. The docstring bullet becomes `core.AuthNZ -> core.LLM_Calls, -> core.Chat and -> core.TTS (TASK-13950.17)`.
  - Seed the ratchet from the PR base: `PYTHONPATH=. $PY -c "from tldw_Server_API.tests.lint.test_llm_calls_import_isolation import loaded_after_import as f; print(sorted(m for m in f('tldw_Server_API.app.core.LLM_Calls.gateway') if m.startswith('tldw_Server_API.app.core.AuthNZ')))"`. Paste the list into the frozenset, one name per line.
  - Expect about 24 names (`tldw_Server_API.app.core.AuthNZ` plus 23 submodules), and no alias name such as `AuthNZ.byok_runtime`. On the dry-run tree, importing `Provider_Credentials.provider_credential_runtime` loaded exactly 24. Six of them load only because `AuthNZ/repos/__init__.py` eagerly imports five unrelated repos.
  - The gateway imports the adapter registry, the providers and `adapter_utils` (`.13` Review Focus 5), so it reaches every module-level `LLM_Calls -> Provider_Credentials` path.
  - Re-run Step 1's command, then the whole of both files. Expected: all pass. The dry run passed 19 contract tests with all three bans. `test_backend_required_enforces_llm_gateway_guards` (`.13`) keeps the isolation file in the required step.
- [ ] **Step 3: Update the docs.**
  - Replace each moved path with its new home: `core/Provider_Credentials/byok_runtime.py`, `core/Provider_Credentials/byok_helpers.py`, `core/Usage/llm_budget_guard.py`, `core/Usage/llm_budget_middleware.py`, `services/authnz_scheduler.py`.
  - Run `bash Helper_Scripts/refresh_docs_published.sh`.
  - Run `PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t17-t6d" tldw_Server_API/tests/Docs/test_docs_published_refresh.py`. Expected: pass.
  - Commit everything, `Docs/Published` included: `git add -A tldw_Server_API Docs`, then `git commit -m "docs: point AuthNZ guides at Provider_Credentials, core/Usage budgets and services/authnz_scheduler; ban AuthNZ -> LLM_Calls and ratchet its indirect AuthNZ load (TASK-13950.17)"`.
- [ ] **Step 4: Verify slice B.**
  - K2: `exit 0` twice.
  - K4 with `B`: the `comm` loop prints nothing.
  - mypy and bandit as in Task 3, Step 6.
  - `PYTHONPATH=. $PY -m tldw_Server_API.tests.lint.test_layering_contract`. Expected, relative to B's base: the same pairs and statements (B adds and removes no ratcheted edge), and the same SCC.
  - Record the AuthNZ coverage floor's percentage from the `coverage-required` job of PR B and of `dev`. Expected: within a fraction of a point, because the floor measures the same code at its new paths.
- [ ] **Step 5: Open PR B** against `dev`, with the same body items as PR A2 that apply. Add:
  - the CLI scheduler-start deletion and its evidence;
  - the floor percentages;
  - the seeded ratchet size.

  Run the clean-subagent review, and check the same CI shards.
- [ ] **Step 6: Close out after PR B merges.** Record the facts with backlog-py, then `git add -A backlog/tasks` and open a backlog-only PR:

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13950.17 \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 \
  --append-notes "<PR A1/A2/B numbers; baseline pairs and statements before -> after; LLM_Calls->AuthNZ 9 -> 0, next to: LLM_Calls still loads <N, about 24> AuthNZ modules indirectly, frozen by AUTHNZ_LOADED_BY_LLM_CALLS; AuthNZ->LLM_Calls/Chat/TTS 27 -> 0 and banned; core.AuthNZ->services stays 1; package SCC raw 102/130 -> 103/131, folded 102/130; K4 base-vs-head result; coverage floor before/after>" \
  --final-summary "<what moved where; aliases kept and gated; the CLI no longer starts the scheduler; follow-up ids>"
```

  Record the same SCC (raw and folded) and the baseline numbers in TASK-13950's notes for the Phase 1 exit check.

  Follow-ups for the coordinator to file:
  - "Invert provider credential resolution: the gateway passes resolved credentials, and LLM_Calls stops loading AuthNZ". It empties `AUTHNZ_LOADED_BY_LLM_CALLS`; L-sized; touches `.13`'s gateway.
  - "Retarget test references to the moved AuthNZ modules and delete the 15 aliases". At close-out that is about 560 test import lines, 13 dotted string targets, 38 `import_module` strings and `.13`'s and `.14`'s additions. Do it after #3187 and #2821 are ported or closed. The codemod is K1 extended to tests, plus the budget rule. When it lands, the alias gate's `MOVED` becomes the deletion list.
  - "Re-dismiss CodeQL alerts #2613, #2614 and #2663 at `core/Provider_Credentials/byok_runtime.py`". These are `py/weak-sensitive-data-hashing`, justified in `Docs/Evidence/PR2761-codeql-python-boundaries.md`. Do it once the move reaches `main`.

## Self-review

Rechecked against the spec, both reports and the binding rulings of 2026-10-26.

**Spec coverage:**

| AC | Proven by |
|---|---|
| AC1 | Task 2: `test_old_authnz_path_is_the_moved_module`, the per-module fresh-interpreter test, `test_patching_the_old_path_patches_the_code_that_runs`, K3, and the alias gate |
| AC2 | Task 4 (`MOVED` and the floor), Task 5 (`find_spec` returns None, the root conftest no-op, and the CLI imports no `services`) |
| AC3 | Task 3 and Task 6 self-tests plus the full contract |
| AC4 | `--tighten` deletes the 9-statement line; Step 6's grep; the alias gate; and the laundering ratchet for the indirect load |
| AC5 | The layering contract (`.9` assertion); the alias gate, the widened guardrails and the CI-posture pins (`test_backend_required_enforces_provider_credential_guards`); and the laundering ratchet (`.13`'s `test_backend_required_enforces_llm_gateway_guards`) |

**Important findings, with where each is fixed:**

| Finding | Fixed in |
|---|---|
| Review I1 / dry-run I2: CLI scheduler start | Task 5 |
| Review I2: pins in a required gate | Task 3 |
| Review I3: slice B coverage floor | Tasks 4 and 5 |
| Review I4: guardrail roots | Tasks 2, 4 and 5 |
| Review I5: `.13` baseline re-key | Task 2, Step 4 |
| Review I6: laundering ratchet and follow-up | Task 6 and close-out |
| Review I7 / dry-run I1: porting recipe and alias gate | Conflict hazards, Task 2 |

**Minors folded:**
- Review:
  - M1: SCC explanation, and the folded measure in Task 3 Step 7 and close-out;
  - M2: K3 budget variant, `R100` for the scheduler only, and `git diff -B -M`;
  - M3: one interpreter per module;
  - M4, partly: the golden envelope;
  - M5: docstring and the `team` scope check;
  - M6: `.12`/`.14`/`.16` hazards.
- Dry run:
  - M1: counts restated as measured or "about";
  - M2: 1443 base, 1460 head;
  - M3: the grep runs before the re-key;
  - M4: Step 6 now relies on the alias gate, which covers the two repo names the regex missed;
  - M5: K4 summary grep and per-slice Postgres subsets;
  - M6: SCC growth reported in PR A2;
  - M7: ruff note;
  - M8: CodeQL follow-up;
  - M9: K2 note.

**Minors declined:**
- **Review M4, the rest:** a Postgres pin on `llm_provider_overrides.secret_blob` and on `byok_oauth_state.pkce_verifier_encrypted`. These repositories store blobs that callers in services and endpoints encrypt with `encrypt_byok_payload`/`dumps_envelope`. Those call sites do not move; the crypto they call does, unchanged (K3). The golden envelope proves that the moved `decrypt_byok_payload`, which every one of these fields uses, reads pre-move data. The existing SQLite suites and `test_byok_oauth_endpoints_pg.py` exercise the writers.
- **Review M6, the optional part:** retargeting the 10 `provider_identity` imports to `core.Utils`. `.12` owns its shim and that shim's retarget follow-up; doing 10 of its 27 importers here would split that follow-up and add a second rule to K1 and K3.

**Type and name consistency:**
- `MOVED` is defined once, in the gate, and imported by the alias tests.
- `MOVED_E2E_PATHS` and `AUTHNZ_FLOOR_TARGETS` are defined in Task 3 and extended in Tasks 4 and 5.
- `AUTHNZ_LOADED_BY_LLM_CALLS` and `authnz_load_drift` are defined and used in Task 6.
- K1-K4 are the plan's commands; C1, C2 and D are `.13`'s PRs.
