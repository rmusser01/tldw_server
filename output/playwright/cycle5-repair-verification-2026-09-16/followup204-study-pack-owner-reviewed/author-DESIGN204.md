# UAT204 / TASK13260.142 — canonical StudyPack worker owner

## Causal evidence

The real cold factory accepts `study-pack-worker-2` as the DB client ID and publishes that object under the same user-directory cache key used by the numeric owner accessor. Actual source resolution, fake-model generation, and persistence store deck/card rows under that label; an independent canonical owner2 DB cannot read the deck. A later owner accessor reuses the wrong object. Numeric-owner-first warm cache passes. Official fixture probe:1 expected PostgreSQL failure/3 warm+SQLite controls,0skip; exact receipt under neighboring `uat-study-pack-owner-diagnosis-20260917`. Initial harness pool-close failure is retained separately.

## Minimal production design

Omit the custom `client_id` argument only in StudyPack `_get_databases_for_user`. Existing `get_chacha_db_for_user_id` already defaults to `str(user_id)` and keeps normal default-character maintenance scheduling. Keep the181 independent operation, acquisition, service behavior, transactions, cleanup and errors unchanged. No generic cache policy, database helper, other-worker, SQL, schema or migration edits.

SQLite keeps physical per-user-file isolation. Newly generated rows on a cold worker use the numeric sync/client label, matching the already-working warm path. Existing rows with the old worker label remain readable and are not rewritten. A control seeds such historical rows and verifies visibility and unchanged labels after generation. This deliberately makes new row metadata independent of cache warmness; it does not migrate historical PostgreSQL rows or repair already-published poisoned cache entries.

## Verification stages

1. **Causal permanent RED — Complete.** Real cold/warm accessor/factory + service/to_thread + fake model + independent numeric reader and same-cache owner lookup, both backends. Add restricted-role actual note-source resolution and SQLite legacy-label preservation.
2. **Minimal GREEN — Complete.** Remove only optional custom client ID. Adapt the existing media-lookup-failure test double to the default canonical accessor signature while retaining its cleanup/error assertions.
3. **Adjacent validation and freeze — Author Complete; independent review pending.** New tests plus unchanged181 worker lifetime source-error, successful/repeated persistence, cancellation and outer transaction controls; existing StudyPack worker tests. Scoped Ruff/Bandit and exact baseline/source attribution. Root owns independent review and integration.

## Scope and limits

Owned production: `app/services/study_pack_jobs_worker.py`. New `tests/DB_Management/test_study_pack_worker_owner_contract.py`; minimal existing `tests/StudyPacks/test_study_pack_jobs_worker.py` fixture adaptation. Existing181 tests are run unchanged. Full prefix-based lifetime cleanup is separate and already preserved. Restricted-role control uses actual existing RLS and actual source resolver with NOSUPERUSER/NOBYPASSRLS on a fixture connection; it does not claim a complete cold-login/bootstrap/model job under a restricted app role. No native/runtime/browser/config/task/tracker/git actions.
