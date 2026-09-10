# Regression repair for GitHub issues 2935–2938

## Scope and decisions

The current dev branch already passes a Uint8Array signature to WebCrypto. Preserve that fix, add a regression that detects the old cross-realm ArrayBuffer failure, and validate malformed-token behavior at the middleware boundary.

Production PostgreSQL sessions omit last_activity while session listing, touch, and refresh use it. Add the column through production bootstrap and upgrade handling, including historical values for existing rows. Tests must cover production-created tables so fixture-only columns cannot mask the defect.

DatabasePool supports variadic arguments and a single sequence of arguments. A PostgreSQL array is one argument, so affected profile override callers must pass a sequence containing that array. Keep the shared compatibility contract; do not inspect SQL text to guess whether a list means parameters or one array. Audit other array queries at the actual pool/connection boundary.

Chat metrics may label an absent model unknown, but execution must retain absence until a real configured default is found. Preserve explicit model and administrator default precedence, consult existing provider configuration for local defaults, and reject missing configuration before an upstream call. Audit users of shared resolution helpers and both streaming modes.

## Work and verification record

Tracked in TASK-13235, TASK-13236, and TASK-13237, initially based on fetched dev
commit `751563a966`. TASK-13238 tracks the migration-test follow-up and PR
publication. Before publication, the branch was rebased onto dev commit
`177d58ac6fee87678d65ce3a9db0216021b6b68e` without conflicts. The task-owned
implementation plan is removed when work is complete; this note retains the
design, audit, and verification record.

## Related-pattern audit

The PostgreSQL array audit followed `ANY`/`ALL` queries through their actual
connection boundary. Seven methods needed changes:

- Organization and team profile override lists and latest-update queries (four
  methods from issue 2937).
- `ManagedSecretRefsRepo.list_refs_by_ids`, including revoked-reference lookup.
- `AuthnzOrgsTeamsRepo.list_organizations` when the organization-ID array is the
  only count filter.
- `AuthnzUsersRepo.list_users` under the same count-filter condition.

Raw acquired/transaction connections retain asyncpg argument semantics and do
not flatten a single sequence. Billing, generated-file, admin analytics, RBAC,
data-subject-request, credential-alias, and migration queries using those
connections therefore do not need the extra wrapper. Queries with multiple
arguments already preserve their array argument. DatabasePool continues to
accept both ordinary variadic arguments and a single parameter sequence.

The regression uses the real DatabasePool parameter handling and mocks only the
external driver. It covers empty, single, and multiple IDs, normalized credential
IDs, scoped counts, and the existing parameter conventions. A Hypothesis property
checks preservation of arbitrary positive integer arrays. Separate integration
tests use the standard isolated PostgreSQL fixture and production schema setup.

The session audit includes fresh and upgraded SQLite databases, historical-row
backfill, new-session initialization after upgrade, and preservation of existing
activity timestamps. PostgreSQL tests recreate the sessions table through
production bootstrap so the fixture's pre-existing activity column cannot hide
the original defect.

The JWT audit found malformed signature encoding and non-string algorithm
headers could raise outside the verification error boundary. Both now fail
closed through the actual middleware. The existing Uint8Array signature argument
is retained, with a regression that rejects reintroducing the raw ArrayBuffer.

The chat audit follows execution-model values independently of metrics labels
through shared resolution and both streaming modes. Explicit models and existing
administrator, DEFAULT_MODEL environment, and Chat-Module defaults retain their
precedence before provider-configuration fallback. Tests exercise real
LOCAL_LLM_MODEL environment capture and Local-API.ollama_model parsing. The
Messages helper also used the metrics placeholder as an execution fallback; it
now rejects missing models. Review additionally found whitespace-only models
could bypass the missing-model guard. Both resolution and outgoing payload
construction now treat blank input like omission. Macro context and tool-policy
keys use the normalized model rather than falling back to the raw request.

## Verification

- Array regression before fixes: 16 failed and 14 passed, with failures at the
  actual driver argument boundary.
- Array regression and existing override-readiness tests after fixes: 49 passed.
- Real PostgreSQL array integration: 3 passed, covering profile overrides,
  scoped counts, and bulk managed-secret references.
- Combined auth/profile suite: 86 passed, including the new PostgreSQL session
  and array integration tests, session migration/read/refresh coverage, override
  readiness, and SQLite compatibility.
- Final PR verification on the rebased branch expanded that suite with the
  API-key, lockout-scope, and usage-truthiness migration tests: 100 passed.
  The profile-version migration file passed separately with 22 tests.
- Session worker checks: 24 focused unit tests and 4 PostgreSQL tests passed.
  The older session integration tests now seed through UsersDB and the standard
  isolated fixture, allowing their existing refresh/revocation checks to run
  with the production profile-write guard intact.
- Bandit for the array production scope: no findings or scan errors.
- Bandit for the session migration/repository scope: no findings or scan errors.
- Both new array test files pass Ruff and Black checks. The production scope has
  one existing Ruff SIM118 finding in unchanged managed-secret row conversion.

JWT checks pass: 11 tests in four focused Vitest files, and the six JWT cases on
both Node 20 and Node 24. Scoped ESLint and TypeScript checks pass. Independent
review cleared JWT, array binding, and session migration changes.

Final chat verification:

| Scope | Result |
| --- | --- |
| Provider/model resolution, target defaults, default provider, payload construction | 146 passed |
| Full simplified endpoint file, including omitted/blank/configured/explicit models in both streaming modes | 215 passed, 1 existing skip |
| Messages usage | 209 passed |
| Messages endpoints, overrides, defaults, native errors | 111 passed |
| Character prechecks and Notes suggestion providers | 30 passed |
| Chat macros | 7 passed |

After the whitespace follow-up and rebase onto dev, the resolver/default/payload
tests and full simplified endpoint file passed 361 tests combined, with one
existing skip. That skip is the older streaming test marked as hanging with
TestClient; the new streaming payload regressions ran successfully. The four
focused UI middleware/auth files also passed again on the rebased branch
(11 tests).

Bandit reports no findings or scan errors across all twelve touched backend
production files. Python compilation, repository test guards, and git whitespace
checks pass. Scoped Ruff checks find no new issues; three existing import-order
findings in chat.py and the existing managed-secret SIM118 finding are identical
on the original base. Existing whole-file Black drift was left outside the
behavioral repair. Independent review identified the malformed-algorithm and
blank-model cases, then cleared all fixes after follow-up.

## Migration-test follow-up

The broader SQLite/AuthNZ migration run passed 52 tests and failed
`test_sqlite_upgrade_preserves_custom_users_schema_objects_and_foreign_keys` in
`tests/AuthNZ/unit/test_profile_version_migration.py`. Running that single test in
a detached worktree at the original base `751563a966` reproduced the identical
failure with migrations 91–97. Migration 98 does not alter users.

The follow-up isolated the stale expectation: migration 91 preserves the custom
users schema and appends profile_version, while migration 93 legitimately appends
uuid, is_active, is_superuser, email_verified, is_verified, storage_quota_mb, and
storage_used_mb. No production migration change is needed. The regression now
checks the preserved column prefix and runs both migration 91 alone and the full
startup upgrade. All existing row, default, constraint, index, trigger, sequence,
and parent/child foreign-key checks remain intact.

Before the assertion repair, the migration-91 case passed and the latest-upgrade
case reproduced the failure. After repair, all 22 tests in the profile migration
file pass. Independent review found no outstanding issues. Ruff, compilation,
and whitespace checks pass. Bandit reports no findings or scan errors in the
changed test file with the expected test assertions excluded via B101.

The wider auth run also exposed an order-dependent Hypothesis generation health
check in the new array property regression. Two wider runs passed 99 tests and
failed that check, while the same property seed passed alone. Profiling a smaller
reproduction traced 1.324 seconds of a draw to Hypothesis's first scan of constants
from 2,159 imported modules. The tested array binding itself was not slow.
Independent review recommended a test-local, finite 1,000 ms deadline: in the
installed Hypothesis 6.138.2, this allows five seconds for generation health
checks. The positive-integer array strategy, length bound, example count, and
assertion are unchanged, and no health checks are suppressed. This avoids
patching Hypothesis internals or reducing coverage.

The final wider run passed all 100 tests using the original test-order seed
`182453889` and Hypothesis seed `219652731782752022987833063022846811844`.
Hypothesis completed 100 passing examples with typical runtimes below 1 ms.
Both follow-up test files pass Ruff, compilation, and Bandit with only expected
test assertions excluded via B101; the array test file also passes Black.

## Known verification limits

The full backend and admin UI suites were not run. PostgreSQL verification uses
the repository's standard local isolated fixture; it does not measure migration
time or lock behavior on large production session tables.
