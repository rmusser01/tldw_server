# PR2761: authoritative selected DSR preview coverage

Task: TASK-13013.8. Base: `0929b44a5a`. Verified 2026-09-10.

## Change and bounded design

The preview service previously converted Chroma manager/list/count failures to
zero or a partial count. It also counted every category before filtering the
summary, so an unrelated missing store could reject a notes-only preview.

The summary now gathers only selected counters in the existing category order.
Embedding collection failures raise `DataSubjectRequestCoverageUnavailableError`;
the existing preview boundary returns the fixed HTTP 500 detail
`requester_data_unavailable`, and intake does not persist a record.

When the optional Chroma manager cannot initialize, the service uses
`DatabasePaths.resolve_user_base_directory` with the same explicit
`USER_DB_BASE_DIR` setting/default as its manager factory. This matches the
manager's path normalization without creating directories. Only confirmed absent
`chroma_storage` permits zero. Existing storage, unresolved paths, and failed
filesystem inspection reject coverage. Service logs retain exception classes,
excluding exception bodies.

Actual erasure handlers and attachment RESTRICT/retention behavior are unchanged.
This is preview correctness evidence, not whole-account, backup, or physical
Chroma erasure certification. Chroma boundary failures are injected; SQLite and
AuthNZ API fixtures are real local databases. No Docker, dependency installation,
or network service was needed.

## Regression and verification evidence

- Initial service regressions: **6 failed, 1 passed** before the production fix.
  Failures cover existing/unknown storage, list failure, partial count failure,
  and selection; confirmed absent storage is the passing control.
  Log: `/tmp/pr2761-dsr-preview-red.log`.
- API regressions against the original service: **7 failed** as intended.
  Manager/list/count failures each returned false HTTP 200 for preview and intake;
  intake persisted a synthetic zero-count record. Notes-only preview returned 500.
  Log: `/tmp/pr2761-dsr-preview-api-red3.log`.
- API fixture repair: the old raw users insert was rejected by the existing
  profile-write guard. Tests now reserve the configured single-user login through
  the canonical repository bootstrap and create subjects through
  `tests.helpers.authnz_seed.ensure_test_user`, reusing returned IDs for stores.
  This also prevents app startup from treating subject fixtures as login storage.
- Final combined suite: **50 passed, 0 failed, 0 skipped, 2 warnings**.
  Log: `/tmp/pr2761-dsr-preview-final.log`;
  JUnit: `/tmp/pr2761-dsr-preview-final.xml`.
- Existing endpoint sanitizer suite: **4 passed, 6 warnings** via
  `python -m pytest tldw_Server_API/tests/Admin/test_admin_data_ops_dsr_sanitizers.py -q`.
  Log: `/tmp/pr2761-dsr-preview-sanitizers.log`.
- Ruff and changed-range Black checks passed. Bandit: **0 findings, 0 errors**
  for production and changed tests (test assertions excluded with `-s B101`).
  Reports: `/tmp/bandit_pr2761_dsr_preview_service.json` and
  `/tmp/bandit_pr2761_dsr_preview_tests.json`.

The earlier combined run was invalidated by host ENOSPC during SQLite/JUnit
writes. Known completed synthetic test directories were cleaned, and the final
run above completed successfully with an explicit isolated temporary directory.

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_TEST_NO_DOCKER=1 python -m pytest \
  tldw_Server_API/tests/Admin/test_dsr_preview_coverage.py \
  tldw_Server_API/tests/Admin/test_dsr_embeddings_erasure.py \
  tldw_Server_API/tests/Admin/test_data_subject_requests_api.py \
  tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py \
  tldw_Server_API/tests/Admin/test_dsr_sqlite_erasure_integration.py \
  -q --basetemp=/tmp/pr2761-dsr-preview-final-temp \
  --junitxml=/tmp/pr2761-dsr-preview-final.xml
```
