## Stage 1: Verify Migration Retry Finding
**Goal**: Confirm migration 087 fails or duplicates data when a partial `share_tokens_new` rebuild table is left behind.
**Success Criteria**: A focused migration test fails against the current `CREATE TABLE IF NOT EXISTS` retry path.
**Tests**: Focused pytest for migration 087 retry behavior.
**Status**: Complete

Notes:
- Focused red run failed with `sqlite3.IntegrityError: UNIQUE constraint failed: share_tokens_new.id`.

## Stage 2: Make Migration 087 Rerunnable
**Goal**: Ensure each migration retry starts from a clean scratch table.
**Success Criteria**: Migration 087 drops any stale `share_tokens_new` table before rebuilding and leaves no scratch table behind.
**Tests**: Focused migration test and relevant sharing/prototype link tests.
**Status**: Complete

Notes:
- Migration 087 now drops a stale `share_tokens_new` scratch table before skip/rebuild decisions when the source `share_tokens` table still exists.

## Stage 3: Verify and Publish
**Goal**: Run focused tests, Bandit on touched migration code, diff checks, then push/reply to the review thread.
**Success Criteria**: Local verification passes and the migration review thread has a reply.
**Tests**: Focused pytest, Bandit, `git diff --check`.
**Status**: Complete

Notes:
- Focused migration retry test passed.
- `tldw_Server_API/tests/Sharing` plus `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py` passed with 141 tests.
- Ruff passed for the new migration test file. Ruff on full `migrations.py` still reports unrelated existing SIM118 findings around line 4617.
- Bandit on `migrations.py` still reports one unrelated existing B608 at line 616; no findings were reported near migration 087.
- `git diff --check` passed.
