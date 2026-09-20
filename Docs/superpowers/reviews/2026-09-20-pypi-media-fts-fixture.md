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
