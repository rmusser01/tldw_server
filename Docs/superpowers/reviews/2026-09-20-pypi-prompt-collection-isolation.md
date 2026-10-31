# PyPI 0.1.42 Prompt Studio collection isolation repair

Worktree: `/Users/macbook-dev/.codex/worktrees/pypi-0142-test-gate-recovery/tldw_server2`, branch `codex/release-0.1.42-pypi-test-gate`, source baseline `cd2dbc792b`. TASK-13013.3 authorized before edits. No commits/push. Only test harness/test files changed; parent owns Backlog and publication decisions.

## Causality and repair

Prompt Studio conftest wrote MINIMAL_TEST_APP=0, TEST_MODE, AUTH_MODE and CSRF_ENABLED at import time, then imported/reloaded the process-wide main module. Merely collecting a Prompt Studio file therefore forced other suites to use its full-app routing profile. Sandbox tests import main.app during collection, before their fixture sets ROUTES_ENABLE=sandbox; full-app policy excluded that route. Their HTTP artifact request returned404. This was reproduced without executing any Prompt Studio test.

Reduced red command (project venv activated; env PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1):
`python -m pytest -p pytest_asyncio.plugin tldw_Server_API/tests/prompt_studio/unit/test_ps_create_request_id.py tldw_Server_API/tests/sandbox/test_artifacts_api.py -k test_artifacts_list_and_download_roundtrip -x -q --tb=short`
Result: **1failed/1deselected**, actual HTTP404; `/tmp/pypi0142-artifacts-prompt-minimal-red.log`. An earlier sandbox-restricted run could not write native-worktree runtime DB and failed integrity startup instead; that run is not claimed as404 evidence. Authorized native-worktree run removed that infrastructure obstacle.

Added regression loads Prompt Studio conftest after the shared app exists and asserts environment/app identity/routes unchanged. Red: **1failed** at environment mutation; `/tmp/pypi0142-collection-isolation-red.log`.

Repair removes all conftest collection-time env writes/main import/reload. A function-scoped `app` fixture uses existing `app_main_isolated()` + `reload_app_main()` with monkeypatch-scoped settings. Five owning modules now request that fixture instead of binding main.app at collection; integration module's redundant collection-time auth/CSRF writes removed. Dual-backend client fixture uses the same app fixture.

The fixture still constructs the **real production application**, preserving actual router composition, auth dependencies, request-ID middleware, exception behavior and TestClient lifespan. No substitute mini-app or production-router changes. Existing isolation helper restores original sys.modules/package-main identity at scope exit, even on failure. Tradeoff: these HTTP tests now pay full application construction per test; this is confined to tests requesting the fixture. Direct unit tests do not construct it.

## Changed test files
- `tldw_Server_API/tests/prompt_studio/conftest.py`
- `tldw_Server_API/tests/prompt_studio/integration/test_api_endpoints.py`
- `tldw_Server_API/tests/prompt_studio/unit/test_evaluation_async_request_id.py`
- `tldw_Server_API/tests/prompt_studio/unit/test_evaluation_bg_propagation_unit.py`
- `tldw_Server_API/tests/prompt_studio/unit/test_ps_compare_strategies_request_id.py`
- `tldw_Server_API/tests/prompt_studio/unit/test_ps_create_request_id.py`
- New `tldw_Server_API/tests/Infrastructure/test_prompt_studio_collection_isolation.py`

## Verification so far
- New import regression + request-ID owning test + sandbox artifact roundtrip: **3passed**,9.78s; `/tmp/pypi0142-collection-isolation-green.log`.
- Original full collection with `-k test_artifacts_list_and_download_roundtrip`: **1passed,12skipped,63,144deselected**,63,145collected (including new regression),28.94s test runtime. Process exited0. `/tmp/pypi0142-artifacts-full-collection-green.log`.
- Five changed owning test modules: **161passed**,198.76s; `/tmp/pypi0142-prompt-owning-tests.log`.
- Required Bandit across seven changed Python files:396 findings, exactly baseline396 (392test asserts,4existing test sentinel/password heuristics), **zero new findings** by file/test-id/exact-trigger-line comparison. Reports `/tmp/pypi0142-collection-isolation-bandit.json` and `...-bandit-base.json`. New regression assertions carry test-only B101 nosec comments.
- `git diff --check`: passed. `git diff --name-only -- tldw_Server_API/app pyproject.toml .github/workflows`: empty. Production/package/workflow bytes unchanged.

This closes one reproduced collection-time404 cause; it does not establish that all128publication markers are repaired. Parent requested recorded-seed bare fullcollection selection of representative Media versions/Integrations/Sandbox files after owning tests; pending.

## Requested next failure diagnostic

After the161-case owning suite completed successfully, ran bare plugin autoload with recorded seed9042326 across full collection:
`env -u PYTEST_DISABLE_PLUGIN_AUTOLOAD -u TEST_MODE -u DISABLE_HEAVY_STARTUP -u PYTEST_ADDOPTS python -m pytest --randomly-seed=9042326 -k 'test_media_versions or test_integrations_control_plane_endpoints or test_sandbox_api or test_artifacts_list_and_download_roundtrip' -x -vv --tb=long`

Result **1failed,1passed,12skipped,63,053deselected**,29.50s, exit1. Sandbox artifact roundtrip passed first. Next `test_media_versions.py::TestSecurityAndPerformance::test_sql_injection_attempt_param` failed at line1074: GET `/api/v1/media?page=1;DROP TABLE Media;` returned404, expected422. `/tmp/pypi0142-representative-full-collection-bare.log` captures full traceback and HTTP404. No Media/Chat harness edits made. Parent already identified other collection-time ROUTES_DISABLE writes for follow-up; this report does not claim those are proven sole cause of the remaining failure.

Patch stable for parent review. Production/package/workflow inputs remain untouched. All161owning Prompt Studio cases passed, process exit0 confirmed after session cleanup.
