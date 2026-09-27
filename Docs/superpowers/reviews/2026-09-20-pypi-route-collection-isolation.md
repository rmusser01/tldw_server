# Media route collection isolation repair

Continuation of TASK-13013.3 in native main-cd2 recovery worktree. Test-only scope; no production/package/workflow edits or commits. Existing Prompt Studio repair is separately documented in `/tmp/pypi0142-prompt-collection-isolation-results.md`.

## Reproduced causes

Three unrelated test files mutated ROUTES_DISABLE at import time solely to reduce import cost:
- `tests/Chat_NEW/conftest.py` added media.
- `tests/Chat_Macros/integration/test_chat_macros_api.py` added media/audio/audio-websocket.
- `tests/Skills/integration/test_skills_api.py` added media/audio/audio-websocket.

Each independently reproduced `Media_Ingestion_Modification/test_media_versions.py::TestSecurityAndPerformance::test_sql_injection_attempt_param` returning404 instead of422 when its file was collected first. No tests in the contaminating suite needed to execute. Commands used project venv and env PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1:

`python -m pytest -p pytest_asyncio.plugin <collecting_file> tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py -k test_sql_injection_attempt_param -x -q --tb=short`

Chat_NEW collecting file was `unit/test_credential_fixtures.py`; reduced result1failed/3deselected (used exact Media::node). Macro result1failed/49deselected; Skills1failed/132deselected. Logs `/tmp/pypi0142-media-pair-test_credential_fixtures-red.log`, `/tmp/pypi0142-media-ordered-pair-test_chat_macros_api-red.log`, `/tmp/pypi0142-media-ordered-pair-test_skills_api-red.log`.

Diagnostic nuance: with an explicit Media::node argument, pytest eagerly collected Media before ordinary Macro/Skills files, masking those two module-level mutations. Selecting via full files plus `-k` restored the harmful order. Those initial pass results are not claimed as red evidence.

## Minimal repair

Deleted just the three collection-time route-disable blocks; retained standard minimal app/test defaults, real production apps, current auth/DB/client fixtures, middleware and lifespans. The import-speed optimization is unnecessary and cannot safely alter another suite's route policy. There is no new route allowlist, test skip, fabricated app, or production behavior change.

New `tests/Infrastructure/test_route_collection_isolation.py` parametrizes imports of all three sources with an unrelated preexisting ROUTES_DISABLE sentinel. Tests fail if imports change that value; patch.dict restores the entire environment after each import.

Red canaries:3failed, `/tmp/pypi0142-route-collection-red.log`.
Green canaries + exact failing Media node while all three suites collect together: **4passed,142deselected**,6.86s; `/tmp/pypi0142-route-collection-green.log`.

## Validation

Owning suite **403passed,9skipped**,351.98s,exit0 (412collected): all Chat_NEW/unit, Chat_NEW/integration/test_chat_completions_api.py, Chat_Macros/integration/test_chat_macros_api.py, Skills/integration/test_skills_api.py. `/tmp/pypi0142-route-owning-tests.log`. Skips are existing missing OpenAI/Anthropic live-provider credential markers; none added.

Required Bandit on three changed files + new test:465baseline/465current, **0new findings**, exact file/test-id/triggering-line comparison. `/tmp/pypi0142-route-collection-bandit.json` and `...-bandit-base.json`.

Ruff across ALL currently touched Python files, including Prompt Studio repair and both new tests:54baseline/30current, **0new findings** by file/code/message multiplicity; both new tests clean. `/tmp/pypi0142-touched-ruff.json` and `...-ruff-base.json`. Fixed only introduced top-import spacing in two owning files and formatted new test files; no whole-file reformatting of legacy files. Used `--no-cache` because native worktree cache isn't writable under ordinary read-only tool sandbox.

`git diff --check` passed. Parent owns integration and independently verified existing Prompt Studio3test subset.

After owning cases complete, repeat recorded-seed bare fullcollection representative selector to find any next residual failure. No claim that all128publication markers are fixed.

## Recorded-seed fullcollection follow-up

After all owning tests exited0, repeated the exact bare selector from the preceding report with seed9042326. Result **1failed,9passed,12skipped,63,056deselected**,29.43s,exit1. Both prior404tests pass, along with additional Media list/detail validation.

Next distinct failure: `test_media_versions.py::TestMediaListDetailEndpoints::test_update_media_item_title`, line1022 expects200 but PUT returns500. Captured traceback: media_db/runtime/fts_ops.py:96 fails updating media_fts with SQLite `database disk image is malformed`. Log `/tmp/pypi0142-representative-full-collection-after-route-fix.log`.

This is a DB/FTS state failure beyond the repaired collection-time routing defect. Session-scoped test database uses direct SQL seed/cleanup and reused IDs, which warrants a separate reproduction; no fixture or application DB changes made here.

Final formatted isolation regression run: **4passed**,6.90s,exit0; `/tmp/pypi0142-final-isolation-canaries.log`. All agent-owned pytest processes exited. Patch ready for independent review; no further source edits pending.
