# Workspace Latest-Dev Integration

Tracking: TASK-12020.50. Continue the accepted remediation sequence after R1-R6.
Do not enable owned routes, implement deletion Stage 2, or imply merge readiness.

## Stage 1: Preserve And Transfer
**Goal**: Preserve the existing verified dirty worktree and transfer its complete
tracked/untracked patch into an isolated checkout of fetched dev.
**Success Criteria**: Original files/index untouched apart from tracking notes;
snapshot manifest records old base, dev revision and checksums. No missing files.
**Tests**: Compare source snapshot, transfer manifest, and three-way apply result.
**Status**: Complete

## Stage 2: Reconcile Contracts
**Goal**: Keep upstream improvements and all workspace authorization, durability
and lifecycle corrections across overlapping files.
**Success Criteria**: Review DB operation ownership and connection-session changes;
resolve textual conflicts semantically; regenerate OpenAPI from combined source.
**Tests**: Targeted connection and operation-lifetime regressions, including new
test-first cases if integration exposes a behavior change.
**Status**: Complete (auth/readiness, operation ownership and deletion sync
contracts reconciled; scoped independent reviews clear)

## Stage 3: Verify Integrated Checkpoint
**Goal**: Validate combined workspace behavior before returning to deletion work.
**Success Criteria**: Frontend/backend scoped regressions, type/lint, Bandit and
actual backend probes pass or have explicitly classified blockers. Independent
review and task/evidence records distinguish integration from full acceptance.
**Tests**: Existing 54-file frontend set; workspace/sharing/DB regressions; real
HTTP probes; OpenAPI fingerprint/type generation; diff check.
**Status**: Complete (1583 frontend tests; 243 scoped backend tests plus final
262-test SQLite/PostgreSQL regression; 76 real HTTP checks; type/OpenAPI/Bandit
verification and scoped independent reviews recorded)

Source: `.worktrees/shared-workspace-clone-ux`, HEAD
`c70387f496d82fcee92926bf3715bf5cd240ba88`.
Fetched dev: `d72b1d2850ea947b6d12cac19f6b95867b68a580` (446 commits ahead).
Seven overlapping tracked files: common locale, TldwAuth, connection store/tests,
OpenAPI fingerprint, persona endpoint, and ChaChaNotes DB. No untracked collisions.

## Integration Record

Active worktree: `/Users/macbook-dev/.codex/worktrees/shared-workspace-dev-integration/tldw_server2`,
branch `codex/shared-workspace-dev-integration`. Original checkpoint remains
unchanged in `.worktrees/shared-workspace-clone-ux`; 112 file checksums verified.
Snapshot: `/tmp/workspace-dev-checkpoint-aLJkQG/progress.patch`, SHA-256
`14261118895fe99cf57f38fb073f3b5bb7150097a024a7b57bd4829d8b42b4ab`.

Three-way transfer conflicts: TldwAuth, connection store/tests, OpenAPI fingerprint.
Auth cleanup retains both new upstream Flashcards cleanup and clone recovery
cleanup. Upstream org selection, refresh invalidation and authority-generation
guards are preserved.

Ruling: retain upstream `/api/v1/auth/sessions` for global readiness instead of
the earlier R6 profile probe. Upstream explicitly supports authenticated but
email-unverified users; profile reads impose a stricter verification requirement.
Both avoid disabled legacy routes and operator-only health. Workspace opening
still uses canonical identity/profile authorization. Cost if this is wrong:
global readiness would need another authenticated probe and corresponding tests;
no workspace principal validation is weakened by this choice.

Focused merge check initially had three failures from the older profile-path
expectation; upstream unverified-session and authority tests passed. Retain the
legacy-disabled 200/401/403 matrix while aligning it with the upstream session
contract, including the captured readiness transport argument.

PostgreSQL integration exposed two additional contracts. Deletion/status worker
callbacks must preserve an inherited request-owned checkout, including after
cancellation; only legacy worker-local connections are explicitly closed. Capture
ownership before DB work because querying a subsequently closed owner can mask
the worker outcome. Eight new regression cases cover normal reuse and cancelled
success/failure; independent review found no actionable defect.

The first full PostgreSQL run also exposed missing message deletion sync events.
This is longstanding missing PostgreSQL behavior, not a reason to skip the new
workspace cascade assertion. SQLite has a delete trigger; PostgreSQL schema
conversion omits it. Add a PostgreSQL-only event in MessageStore's existing
transaction, matching the SQLite payload and both supported sync identifier
columns. This follows existing manual sync-write practice and avoids a broad
trigger migration. It is deliberately not full create/update/restore sync parity.
If PostgreSQL message triggers are added later, this manual write must be removed
in the same change to avoid duplicate events. Verify retries, stale versions,
transaction rollback and sync-write failure before closing this checkpoint.

Final evidence: `../../Development/workspace-dev-integration-2026-09-20.md`.
The final backend process exited 0 with 262 passed and no skips. Source snapshot
checksums still match all 112 entries, temporary dependency links are removed,
and all task-owned probes/test sessions and reviewers have finished. No commit,
push, route enablement, or full acceptance claim. Continue deletion Stage 2.
