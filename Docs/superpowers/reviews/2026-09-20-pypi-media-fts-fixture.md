# Media versions FTS fixture repair (third independent root cause)

TASK-13013.3; native recovery worktree, after committed collection-isolation repairs `3b96a3cb04`. Only `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py` edited; no production/package/workflow changes or commits.

## Reproduction and diagnosis

- Fullcollection recordedseed diagnostic had progressed past both route404defects and stopped at `TestMediaListDetailEndpoints::test_update_media_item_title`: HTTP500, SQLite `database disk image is malformed` from `_update_fts_media`.
- Entire Media file alone with seed9042326: **1failed/8passed**,7.42s, same failure; `/tmp/pypi0142-media-versions-seed-alone.log`.
- Exact title-update node alone: **1failed**,7.40s, same failure; `/tmp/pypi0142-media-update-title-alone.log`. Thus this failure does not require cross-test history.
- Read-only pytest plugin before request: Media rows `(1,'Multi Test Doc'),(2,'Multi Test Video'),(3,'Multi Test Audio')`; `media_fts MATCH '"Multi Test Doc"'` returned no rows. `/tmp/pypi0142-media-fts-readonly-probe.log`. `seeded_multi_media` inserted Media rows directly without FTS maintenance. Existing `seeded_document_media` already calls the established `_update_fts_media` helper.
- Added behavioral checks for all three initial titles in real MATCH search, then old-title removal/new-title match after the real endpoint update. Red before fixture repair: **1failed** at `[] != [1]`, before PUT; `/tmp/pypi0142-media-missing-match-red.log`.
- Separately verified cleanup defect with a read-only `pytest_fixture_post_finalizer` probe after existing document test: `Media COUNT=0` but `media_fts MATCH '"Test Document"'` still returned rowid1. `/tmp/pypi0142-media-fts-cleanup-probe.log`; original document test itself passed. This proves raw DELETE cleanup left stale index records and justifies fresh per-test DB lifetime independently of missing initial index.

## Minimal repair

Multi-media seed now uses the existing `_update_fts_media` method once for each inserted Media row inside the existing transaction. It matches the neighboring document seed convention; no production tolerance changes.

The existing temp_db fixture is function-scoped and clearly named `db_instance`. Client and `db_session` receive that same per-test instance. Removed raw table DELETE/autoincrement reset cleanup, which did not maintain FTS. Existing temporary-file/backend teardown and DB connection cleanup remain. All actual endpoint, auth override, versioning and DB result assertions remain intact.

## Verification

- Entire43-case Media file with plugin autoload and recorded seed9042326: **43passed**,23.59s,exit0; `/tmp/pypi0142-media-versions-fts-green.log`.
- Ruff current5/baseline11, **0new findings**; `/tmp/pypi0142-media-fixture-ruff-{base,current}.json`.
- Bandit current212/baseline212, **0new findings**; `/tmp/pypi0142-media-fixture-bandit-{base,current}.json`. New test assertions use test-only B101 nosec; real query values remain parameterized.
- `git diff --check`: passed.
- Recordedseed bare fullcollection representative92case selector running; `/tmp/pypi0142-representative-full-collection-after-fts-fix.log`.

No claim that remaining publication failures are all repaired. This is a separately reduced, evidenced test-data/index-lifetime repair after the two collection-routing fixes.

## Broader diagnostic after repair

Recordedseed bare fullcollection selector exited1 after **61passed,1failed,12skipped,63,056deselected**,60.07s. All43Media cases,16Sandbox API cases,artifact roundtrip and first Integrations case passed. Next failure is independent test-loop setup: `Integrations::test_personal_slack_update_route_disables_all_provider_installations` never reaches its body; pytest_asyncio.wrap_in_sync raises `RuntimeError: There is no current event loop in thread 'MainThread'` while acquiring the loop. `/tmp/pypi0142-representative-full-collection-after-fts-fix.log`.

Environment caveat: local shared venv has pytest-asyncio1.1.0/Python3.11, while publication log reported1.4.0/Python3.12. Do not assume that exact plugin failure explains remote F markers without equivalent reproduction. No event-loop or Integrations edits made here. All agent-owned test processes exited; third repair ready for independent review.

## Temporary publication-version runner comparison

Installed ONLY into `/tmp/pypi0142-ci-pytest-plugins` using activated shared venv and `pip install --target ... --no-deps pytest==9.1.1 pytest-asyncio==1.4.0 pytest-randomly==5.0.0`. Shared venv unchanged (verified it still reports pytest8.4.1/asyncio1.1.0/randomly4.1.0 without target PYTHONPATH). No repository dependency edits.

Validated target module locations, Requires-Python>=3.10, and active requirements against installed iniconfig2.1.0,packaging26.0,pluggy1.6.0,pygments2.19.2,typing-extensions4.15.0; all satisfy constraints. Metadata `/tmp/pypi0142-ci-pytest-plugin-metadata.json`; install log `/tmp/pypi0142-ci-pytest-plugin-install.log`.

With `PYTHONPATH=/tmp/pypi0142-ci-pytest-plugins:.` and same bare environment/seed, exact previously failing Integration node now **1passed**,9.16s,exit0; `/tmp/pypi0142-integration-ci-plugins-alone.log`. This supports a local runner-version contribution; fullcollection comparison is still required to distinguish order effects. Fullcollection same representative selector running: `/tmp/pypi0142-representative-full-collection-ci-plugins.log`.

Only these three test-runner versions are aligned. Local Python3.11/macOS and other shared dependencies still differ from publication Python3.12/Ubuntu; this is a controlled diagnostic, not a complete reproduction of the hosted environment.

### Completed aligned-runner comparison

The same bare full-collection representative selection completed with **91 passed, 13 skipped, 63,056 deselected, 188 warnings in 135.57 seconds; process exit 0**. Log: `/tmp/pypi0142-representative-full-collection-ci-plugins.log`. All four requested representative files passed, including all eight Integration control-plane endpoint tests. Collection accounts for 12 skips; the single selected skip is the existing `MediaIngestion_NEW/integration/test_media_versions_integration.py::test_rollback_to_current_version_conflict`.

Exact command (after activating shared venv, from the native recovery worktree):

```sh
env -u PYTEST_DISABLE_PLUGIN_AUTOLOAD -u TEST_MODE \
  -u DISABLE_HEAVY_STARTUP -u PYTEST_ADDOPTS \
  PYTHONPATH=/tmp/pypi0142-ci-pytest-plugins:. \
  python -m pytest --randomly-seed=9042326 \
  -k 'test_media_versions or test_integrations_control_plane_endpoints or test_sandbox_api or test_artifacts_list_and_download_roundtrip' \
  -x -vv --tb=long
```

The prior local event-loop failure did not recur after aligning only the three runner packages. This is evidence of a local runner-version artifact for that observed failure; no asyncio or Integration source repair was needed or made. It does not establish that the full publication gate passes: only the 92 selected cases ran after full collection, and Python/platform/other dependencies remain different (local Python 3.11.13/macOS versus publication Python 3.12/Ubuntu).

Parent committed the test-only FTS repair as `02b42324d1`; working tree was clean after the diagnostic. Shared venv and repository dependencies remain unchanged.

An isolated `-q -rs` rerun of the selected skipped node with the same aligned runner/seed exited 0: **1 skipped, 2 warnings in 9.43 seconds**. Actual reason at line 234: `Backend did not enforce 'rollback to current' conflict; skipping assertion`. This is an existing conditional skip, not added by the repairs. Log: `/tmp/pypi0142-aligned-selected-skip-reason.log`. The original broad run did not include `-rs`, so this exact reason was confirmed in the isolated rerun. No agent-owned test processes remain active.
