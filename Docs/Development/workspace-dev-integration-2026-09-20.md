# Workspace Dev Integration Checkpoint

Tracking: TASK-12020.50. This record covers reconciliation with current dev, not
completion of the workspace clone acceptance matrix or deletion feature.

## Source Preservation

- Original worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/shared-workspace-clone-ux`.
- Original base: `c70387f496d82fcee92926bf3715bf5cd240ba88`.
- Integration base, fetched this pass: `d72b1d2850ea947b6d12cac19f6b95867b68a580`.
- Active worktree: `/Users/macbook-dev/.codex/worktrees/shared-workspace-dev-integration/tldw_server2`.
- Active branch: `codex/shared-workspace-dev-integration`.

Captured 112 changed files, including all untracked additions, with a temporary
Git index. Original files and staging index were not changed by the transfer.
Patch and checksum manifest: `/tmp/workspace-dev-checkpoint-aLJkQG`.
Patch SHA-256: `14261118895fe99cf57f38fb073f3b5bb7150097a024a7b57bd4829d8b42b4ab`.
All 112 source checksums matched afterward. The new tree's 104 non-overlapping
content files matched the snapshot exactly; its integration plan was intentionally
updated. The seven upstream-overlapping files were reconciled separately.

The new worktree uses Git's index for the three-way transfer; staged changes are
not commits. Neither branch was pushed, and the original checkpoint was not
rebased or stashed.

## Contract Decisions

- Retain upstream org selection, refresh-session invalidation, authority-generation
  fencing and captured readiness transport.
- Retain both Flashcards handoff cleanup and clone recovery cleanup at logout.
- Retain upstream `/api/v1/auth/sessions` for global connection readiness. This
  supersedes the earlier R6 profile-path implementation, not its security goal:
  no disabled legacy endpoint, no operator health permission, and no public
  liveness fallback on 401/403. Global connectivity is not email verification.
- Owned-workspace opening still uses `/users/me/profile?sections=identity` and
  expected-user headers. An unverified session can be connected while workspace
  identity authorization is denied.
- Regenerate the OpenAPI contract from combined source instead of selecting
  either conflicting fingerprint. Result: 2098 paths, 3209 schemas; generated
  TypeScript and fingerprint recheck pass.

The frontend integration review found no actionable issues in auth cleanup,
authority ownership, readiness transport or retained regression coverage.
The database-operation review identified an integration defect: deletion/status
callbacks unconditionally closed a PostgreSQL checkout owned by the request.
They now capture ownership before DB work and leave inherited checkout cleanup
to the request scope. Worker-local SQLite/legacy cleanup is unchanged. Capturing
ownership before cancellation also prevents closed-scope lookup from masking the
worker's success or failure. New tests failed six cases before the fix; all eight
new cases then passed, including real PostgreSQL execution. Independent review
found no actionable defect. Cancellation precisely around ownership capture and
unscoped PostgreSQL endpoint cleanup lack dedicated new endpoint tests; broader
operation-scope coverage remains part of the regression run.

The parent and implementer full backend runs both found one separate failure:
PostgreSQL deleted messages had no sync events. This newly added stream assertion
exposed longstanding missing PostgreSQL functionality; it was not dismissed as
an inherited test failure. SQLite's message-delete trigger is omitted by the
PostgreSQL schema converter and has no native replacement. MessageStore now
writes the equivalent PostgreSQL delete event in the same transaction, supporting
both `entity_id` and shared-schema `entity_uuid`. Four new regressions failed
before the fix; those and the two-backend cascade test passed afterward. Additional
cases cover sync-metadata and insert failures rolling back the workspace graph.
No test was skipped or weakened to hide the gap. Independent review of the
message-store fix and failure-path tests found no actionable issues.

This is a deletion-specific compatibility fix, not a migration or full PostgreSQL
message create/update/restore sync implementation. A future native message-delete
trigger must replace, not supplement, the manual write to avoid duplicate events.

## Verification

### September 25 PR #3020 Rebase

The two implementation commits were rebased onto dev
`59bd5845038342013a2d84d0130f6164f14b54fd`. Only the OpenAPI fingerprint
conflicted; it was regenerated rather than choosing either side. Range review
preserves the workspace changes and upstream stream-error sanitization, sharing
admin-role authorization and PostgreSQL note-store fixes. Four fixture-plugin
changes are already present upstream and no longer appear in the PR diff.

| Rebase Check | Result | Evidence |
| --- | --- | --- |
| Workspace restore/settings/selection and note endpoint | 45 passed, no skips, 4 warnings | `/tmp/workspace-rebase-fences.log`, `.xml` |
| Sharing endpoints and note store | 89 passed, no skips, 7 warnings | `/tmp/workspace-rebase-upstream.log`, `.xml` |
| Canonical schema and TypeScript generation | 2098 paths, 3211 schemas; passed | `/tmp/workspace-rebase-openapi.log`, `/tmp/workspace-rebase-schema.d.ts` |
| Integration-touched production Bandit | 0 findings/errors | `/tmp/workspace-rebase-bandit.json` |
| Full feature diff whitespace | Passed | `git diff --check origin/dev` |

The workspace matrix uses the official PostgreSQL fixture on the task-owned
5434 container and SQLite. This is bounded rebase verification, not a rerun of
the historical frontend suite, full-project tests, WebUI/CDP acceptance or CI.
The owned route remains disabled; PR #3020 remains draft.

| Check | Result | Evidence |
| --- | --- | --- |
| Workspace and upstream auth regression | 1583 passed, 58 files | `/tmp/workspace-dev-regression.log` |
| Focused merged auth/connection | 70 passed, 2 files | `/tmp/workspace-dev-connection-green.log` |
| Sharing, workspace subresources, principal binding, clone media, persona | 243 passed, 11 warnings | `/tmp/workspace-dev-backend-regression.log` |
| Scoped TypeScript, 34 entrypoints | Passed | `/tmp/workspace-dev-tsc.log` |
| Integration-file ESLint | 0 errors; 22 existing `any` warnings and pages-path notice | `/tmp/workspace-dev-eslint.log` |
| Deletion/message-store/test Ruff | Message store and tests clean; 4 endpoint BLE001 findings also present at dev HEAD | `/tmp/workspace-dev-python-lint.json`, `/tmp/workspace-dev-python-lint-base.json` |
| Changed Python production Bandit, including message store | 0 findings/errors | `/tmp/workspace-dev-bandit-final.json` |
| New sync parity plus SQLite/PostgreSQL cascade | 6 passed, no skips | `/tmp/workspace-dev-sync-green.log` |
| New sync regression RED | 4 failed as expected, missing events | `/tmp/workspace-dev-sync-red.log` |
| Parent operation regression before sync repair | 230 passed, 1 failed, no skips | `/tmp/workspace-dev-operation-parent.log` |
| Final workspace, message store, sync schema and operation-lifecycle regression | 262 passed, 23 warnings, no skips; exit 0 | `/tmp/workspace-dev-backend-final.log` |
| OpenAPI/types/fingerprint | Passed | `/tmp/workspace-dev-openapi.log`, `/tmp/workspace-dev-codegen.log`, `/tmp/workspace-dev-openapi-check.log` |
| Real JWT/cookie HTTP | 30 requests passed | `/tmp/workspace-dev-readiness-live.log` |
| Real principal-bound workspace HTTP | 23 requests passed | `/tmp/workspace-dev-principal-live.log` |
| Real deletion/tombstone HTTP, after final sync repair | 23 requests passed | `/tmp/workspace-dev-deletion-live-final.log` |

Live probes used isolated SQLite databases and production FastAPI/authentication.
They started and stopped their own loopback backend processes. Application
lifespan was disabled; auth schema/profile bootstrap used production helpers.
No authentication/DB overrides were used. For the unverified-user case, the probe
changed only its synthetic user's verification flag using UsersDB, then restored
it. That demonstrated session 200 versus profile 403. Expired JWTs used a correctly
signed synthetic token with an expired timestamp. Other cases covered valid
sessions, legacy 410, missing/invalid/revoked credentials, expected-user mismatch,
version conflict, active/deleted/missing state and no-store responses.

Temporary live fixtures: `workspace-r6-multi_user-28zp17xw`,
`workspace-r6-single_user-d82am3f2`, `tldw-workspace-p1-http-7pw24ezc`, and
`tldw-owned-delete-http-pubqg5qv` under the OS temporary directory.

## Remaining Work

This bounded integration checkpoint is complete. The final backend run includes
all 19 operation-scope tests, 53 PostgreSQL HTTP lifecycle tests, both sync column
variants, rollback/failure/retry checks, and the formerly failing workspace
cascade test. The process exited successfully after a slow interpreter-finalization
GC pass, confirmed by `/tmp/workspace-dev-pytest-teardown.sample.txt`. Existing
pytest temporary-directory cleanup warnings remain; no checks were cancelled.

All 112 original snapshot entries still match, including the intentionally deleted
file. The three temporary frontend dependency links were verified and removed;
task-owned probes/test sessions and reviewers are finished. Work remains
uncommitted in the new worktree, and the preserved source worktree is unchanged.

Next is deletion Stage 2: writer fences and sharing cleanup state/retry. Beyond
that, deletion UI, explicit draft conflict
resolution/journal lifecycle, and full WebUI/CDP acceptance remain unfinished.
PostgreSQL ran through the official real database fixtures without skips; this
does not certify a full production PostgreSQL deployment, startup, or WebUI/CDP
workflow. The owned route remains disabled and TASK-12020.50 remains In Progress.
