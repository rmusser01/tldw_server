# Final Startup Owner Fix Report

## Status

Implemented the single P2 fix wave against `127cac708628dd4a21d618beefb82b5ef1bd0823` in `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/persona-workspace-parity-plan`. This report is included in the same fix commit despite the SDD directory's normal ignore rule, as requested. The commit containing this report is the stable handoff; its hash is returned separately to the parent.

TASK-13245.4 remains In Progress; AC5 and DoD1 remain unchecked. The original uncommitted parent finding paragraph is included unchanged. Task updates used the official CLI in this worktree; no task-file manual edits. MCP discovery exposed no workflow resource or instructions tool. No subagents, full acceptance matrix reruns, publication, push, PR, merge, rebase, cleanup or PostgreSQL provisioning occurred. Parent owns both refreshed broad acceptance gates and scoped independent re-review.

## Cause And Fix

The trusted per-user cache is keyed by `DatabasePaths.get_user_base_directory(user_id)`. Its healthy-hit and initialization-waiter paths return the first published handle without replacing `client_id`. Voice supplies `voice_assistant`; workers also supply attribution aliases. The previous startup guards treated this attribution string as the owner, incorrectly rejecting legitimate REST owners.

`CharactersRAGDB` now accepts keyword-only `owner_user_id` and retains it as a read-only construction-time property. `_create_and_prepare_db` supplies `str(user_id)` from its existing trusted dependency argument, separately from the unchanged caller `client_id`, before seeding or publishing the handle. Standalone construction defaults the owner to the initial `client_id`, preserving existing owner-scoped direct callers. No owner identity comes from request metadata or path parsing.

Both startup guards now compare against this owner; the creation payload must still match the authenticated owner, and all existing scope/lineage checks remain. Public Character library projection passes the same owner rather than its writer alias. Cache keys, locks, coalescing, health probes, lifecycle, client attribution, service call sites, database schema and PostgreSQL session/RLS behavior are unchanged. This is a startup-boundary correction, not a general redesign of historical client-ID semantics.

## Exact Changed Paths

Paths below are relative to the isolated worktree above:

| Path | Reason |
| --- | --- |
| `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` | Add explicit construction-time owner with a read-only property and compatible standalone default. |
| `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py` | Supply trusted canonical owner independently of caller attribution at normal construction. |
| `tldw_Server_API/app/core/Workspaces/assistant_defaults.py` | Use the canonical owner in creation and reference-projection guards. |
| `tldw_Server_API/app/core/Character_Chat/modules/character_chat.py` | Pass canonical owner through the common public library projector. |
| `tldw_Server_API/tests/API_Deps/test_chacha_startup_owner.py` | Add 19 real-cache/on-disk SQLite cases for alias-first REST creation/projection, public facade visibility and client filtering, wrong-owner/alias/payload denial, owner isolation, shutdown/reopen, and a synchronized initialization waiter. |
| `tldw_Server_API/tests/Chat/test_chacha_db_deps_error_mapping.py` | Update one constructor double to assert the new trusted owner while retaining the attribution and sanitized-error assertions. |
| `tldw_Server_API/tests/Visual_Identities/test_builtin_pixel_migu.py` | Forward the new constructor keyword in one existing seed-error lifecycle test; no feature change. |
| `Docs/Code_Documentation/Workspace_Persona_Defaults.md` | Clarify trusted owner versus writer attribution and standalone construction responsibility. |
| `backlog/tasks/task-13245.4 - Implement-local-Workspace-Persona-startup-provenance.md` | Retain the parent note and append fix execution, verification and remaining gates via CLI. |
| `.superpowers/sdd/2026-09-13-persona-startup-provenance-implementation-plan/final-fix-report.md` | Commit this bounded evidence/handoff report. |

## TDD And Runtime Evidence

All pytest runs activated `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate`, ran from the isolated worktree, and used installed `-n 4 --benchmark-disable` with the real macOS `${TMPDIR%/}` basetemp. No cache-hit dictionary or DB implementation was substituted in the new tests. `USER_DB_BASE_DIR` points normal path resolution at each test's temporary directory; initialization, bundled seeding, default-character tasks, SQLite health checks, cache publication and REST dependency are real. The authenticated-user argument is a small `id=42` fixture, not a claim of full voice/HTTP end-to-end coverage. The concurrency test pauses real construction before publication and observes REST waiting on the actual initialization event; it releases and joins both tasks in `finally`. Async fixture teardown drains tasks/executor and closes cached handles through the normal lifecycle.

| Run | Result | Log |
| --- | --- | --- |
| New tests before production edits | 11 failed, 8 passed, 12 warnings; 41.70s; exit 1 | `/tmp/persona-final-fix-red.log` |
| Same new tests after production fix | 19 passed, 12 warnings; 41.51s; exit 0 | `/tmp/persona-final-fix-green.log` |
| Initial existing dependency gate | 2 failed, 47 passed, 12 warnings; 40.97s; exit 1 | `/tmp/persona-final-fix-deps.log` |
| Final combined focused gate | 68 passed, zero failures/skips, 12 warnings; 46.31s; exit 0 | `/tmp/persona-final-fix-targeted.log` |

RED reproduced `Conversation owner must match the scoped database owner` and `Startup projection owner must match the scoped database owner` for both `voice_assistant` and `study-pack-worker-42`. Legacy explicit-owner insertion worked. Tests also exposed absent canonical-owner attributes and alias callers reaching default resolution instead of owner rejection. Public library facade tests initially passed with the old alias-based guard; they remain passing after both guard and facade adopt canonical ownership.

The two intermediate dependency failures were the existing constructor doubles receiving unexpected `owner_user_id`; only their signatures/forwarding and one owner assertion changed. No tests were disabled or security checks weakened. Pytest reported 12 warnings on each run; no warning-free claim is made. The configured output omits a detailed pytest warning summary, so individual warning attribution was not established by these logs.

Exact final gate:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/API_Deps/test_chacha_startup_owner.py \
  tldw_Server_API/tests/API_Deps/test_chacha_notes_db_deps_error_mapping.py \
  tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_sqlite_policy.py \
  tldw_Server_API/tests/Chat/test_chacha_db_deps_error_mapping.py \
  tldw_Server_API/tests/Chat/test_chacha_runtime_contract.py \
  tldw_Server_API/tests/VoiceAssistant/test_ws_integration.py::TestWebSocketAuthentication \
  tldw_Server_API/tests/Visual_Identities/test_builtin_pixel_migu.py::test_production_db_factory_seeds_before_return \
  tldw_Server_API/tests/Visual_Identities/test_builtin_pixel_migu.py::test_production_factory_preserves_preexisting_deleted_name \
  tldw_Server_API/tests/Visual_Identities/test_builtin_pixel_migu.py::test_production_factory_closes_connection_after_seed_error \
  -n 4 --benchmark-disable --basetemp="${TMPDIR%/}/persona-final-fix-targeted" \
  -q -rs --tb=short
```

RED/GREEN select only the new file and use basetemps `persona-final-fix-red` and `persona-final-fix-green`; the intermediate dependency run selects the final gate minus the new file and uses `persona-final-fix-deps`. Counts are overlapping, not additive. No live PostgreSQL run is claimed for this wave. Parent's latest instruction reserves both full acceptance matrices, including affected Workspace/HTTP and Character suites, for the stable fix commit.

## Static And Security Evidence

All scans used the shared activated venv. Exact scope is the seven Python paths in Changed Paths, not the earlier 29-file scope. Baseline existing files plus `pyproject.toml` were extracted with `git archive 127cac708628dd4a21d618beefb82b5ef1bd0823` into `/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/persona-final-fix-base.TnEtIa`, with no checkout or worktree mutation. This includes newly touched dependency and test paths, not just the prior Stage 5 paths. The entirely new test file has no baseline; its Ruff and Bandit results are zero findings.

| Gate | Result | Artifacts |
| --- | --- | --- |
| `python -m py_compile` on all seven Python paths | Exit 0 | Command output verified |
| `python -m ruff check <seven paths> --output-format json` | Exit 1: one unchanged `I001` at `character_chat.py:9`; baseline also one; normalized file/code/message comparison has no additions | `/tmp/persona-final-fix-ruff.json`, `/tmp/persona-final-fix-base-ruff.json` |
| `python -m bandit <four production paths> -f json -o ...` | Current and baseline exit 0, zero findings/errors | `/tmp/persona-final-fix-bandit-production.json`, `/tmp/persona-final-fix-base-bandit-production.json` |
| `python -m bandit <three test paths> -s B101 -f json -o ...` | Current and baseline exit 0, zero findings/errors; only assertion rule B101 excluded | `/tmp/persona-final-fix-bandit-tests.json`, `/tmp/persona-final-fix-base-bandit-tests.json` |
| `git diff --check` | Exit 0 before final report/staging; checked again at commit boundary | Git verification |

No suppressions, dependency changes or unrelated formatting edits were added. Existing broader-stage warning/security records in `task-5-report.md` remain historical evidence, not substituted for this wave's scans. Owned pytest and scan sessions have all exited.

## Remaining Concerns And Parent Gates

- Parent must rerun both prepared acceptance matrices and perform the scoped independent re-review before approval/publication. No approval or full-stage completion is claimed here.
- Existing PostgreSQL Character WorldBook preflight limitation remains unchanged and uncertified; see `database-context.md` and `task-5-report.md`. The task-owned PostgreSQL container on port 55461 was not used or altered by this worker.
- Standalone callers with separate writer attribution must explicitly supply their trusted owner. The new field does not retrofit every historical use of `client_id`, change RLS, or authorize caller-supplied metadata.
- Offline maintenance and compatible restart requirements remain. No old-handle hot-upgrade mechanism, schema migration, receipt/provisioning/profile/RAG/Buddy/animation work or cleanup was introduced.
- TASK-13245.4, the parent task, later stages and the human-written Change summary merge gate remain open as previously documented.
