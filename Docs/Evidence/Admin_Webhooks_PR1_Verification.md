# Admin Webhooks PR 1 Verification

## Verification Identity

- Tested application source commit:
  `2ad82ff4bcc7faee3cac127aa31a11733c5eb550`
- Final reviewed test-only source commit:
  `ca304f4b00536593a70844db268a7baa14886d7f`
- Final exact-base CI-ratchet source commit:
  `a53fe714f9610c6b908504381838409a039051ba`
- Final reviewer-remediated CI/E2E source commit:
  `dac56c2004b859103ceef393f023927014a988da`
- Latest Qodo and independent-review remediation source commit:
  `517d7b016089e220fa55eab3483f212031b6f5cb`
- Exact-head Qodo follow-up test commit:
  `7253450461a58f0724fba77de84c97e5ec26b548`
- Last immutable pushed head before the lifecycle-race follow-up:
  `cc885b47c86ec5fc64a1cdcc901839136c1a5909`
- Rebased onto: `origin/dev` at
  `9ee0b5a16dca9f5cf6372a3dd2798b84075501fc`
- Final ratchet comparison base: `origin/dev` at
  `9ee0b5a16dca9f5cf6372a3dd2798b84075501fc`
- Final verification timestamp: `2026-08-27T07:20:04Z`
- Host: macOS 26.5.2 (25F84), arm64
- Python: 3.11.13
- Node.js: 20.19.5 (the version family pinned by repository UI CI)
- Bun: 1.3.2
- Next.js: 16.2.2
- PostgreSQL: 18.6 (`postgres:18`, Debian 18.6-1.pgdg13+2)

The tested application source commit includes the control plane, dual-backend
persistence, legacy importer, key rotation, route selector, admin UI, runbooks,
and all runtime gate fixes described below. Later immutable commits contain the
analyzer testability remediation and the exact-base CI ratchet described below.
This evidence file is a documentation-only follow-up to those source trees.

## Result Summary

| Gate | Result |
| --- | --- |
| OpenAPI fingerprint and drift | PASS |
| Complete PR 1 Python matrix | PASS at final-rebase source: 460 non-PostgreSQL + 24 required PostgreSQL |
| PostgreSQL-required matrix | PASS at final-rebase source: 24 passed, 0 skipped |
| Direct pytest marker policy | PASS |
| CI shard coverage guard | PASS: 0 newly uncovered test files |
| Canonical webhook full-suite ownership | PASS: dedicated shard in all five matrices |
| Admin Webhooks non-PostgreSQL matrix | PASS: 301 passed |
| Chat persistence ordering regression | PASS: exact failing E2E plus 4-test surrounding set |
| Ruff | PASS |
| Focused Python typecheck | PASS |
| Bandit | PASS |
| Backend sensitive-log scans | PASS |
| Focused admin UI matrix | PASS: 77 passed |
| Qodo analyzer testability remediation | PASS: 5 passed; package typecheck and lint passed |
| TypeScript typecheck | PASS |
| Changed-file ESLint | PASS |
| Production admin UI build | PASS |
| Chromium control-plane journey | PASS: 1 passed |
| UI persistence/console sink scan | PASS |
| Package-wide admin UI tests | STRICT RATCHET PASS: 41 inherited failures, 0 regressions; all safety counters zero |
| Package-wide admin UI lint | PASS: 0 errors, 41 unchanged warnings |
| Two-project real-backend Playwright | PASS: JWT 26 passed/1 expected skip; single-user 1 passed/26 expected skips |
| Post-ratchet review remediation | PASS locally: 4 Qodo findings and 7 independent-review findings closed |
| Credential destination proof | PASS: hostile request Host never received the single-user API key |
| Exact-head Qodo follow-up | 2 valid test findings fixed; incorrect suite-count claim refuted with Vitest runtime/source evidence |

PR 1 remains default-off. These results do not authorize outbound webhook
delivery or canonical activation.

## OpenAPI Contract Review

Commands:

```bash
make CI_LOCAL_PYTHON=../../.venv/bin/python openapi-fingerprint
make CI_LOCAL_PYTHON=../../.venv/bin/python openapi-drift-check
git diff -- apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

Result:

```text
path_count:   2039
schema_count: 3031
sha256:       41d99488f7bb295e7c20d6c05085f788e54619c1079fbbf8053522c7dde949a9
drift-check:  PASS
```

The Make override is required because the shared Python 3.11 environment is the
repository-supported interpreter; the host default interpreter could not parse
the application type syntax.

The checked-in `origin/dev` fingerprint was already stale by one path and one
schema after an unrelated concurrent merge. Full schemas were therefore
exported from both actual trees and compared directly. The reviewed branch
delta was limited to the canonical webhook surface:

- added `/api/v1/admin/webhooks/catalog`;
- added `/api/v1/admin/webhooks/status`;
- added `/api/v1/admin/webhooks/{webhook_id}/rotate-secret`;
- added canonical GET for `/api/v1/admin/webhooks/{webhook_id}`;
- replaced list/create/PATCH/delete operations with canonical contracts;
- removed `/api/v1/admin/webhooks/{webhook_id}/test`;
- removed `/api/v1/admin/webhooks/{webhook_id}/deliveries`;
- removed `/api/v1/admin/incidents/{incident_id}/notify-webhooks`;
- added only the canonical catalog, status, registration, error, limit,
  migration, patch, delete, and one-time-secret schemas;
- removed only the superseded numeric webhook/delivery schemas.

Initial generation revealed that canonical classes named
`WebhookRegistrationResponse` and `WebhookStatusResponse` renamed the unrelated
evaluation-webhook component references. The canonical classes now use the
`AdminWebhook...` prefix, and a combined-router regression test proves the
existing evaluation references remain unchanged.

## Python And Database Gates

Complete matrix:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks \
  tldw_Server_API/tests/Admin/test_admin_system_ops_service.py \
  tldw_Server_API/tests/Admin/test_admin_webhooks_service.py \
  tldw_Server_API/tests/Admin/test_admin_webhooks_schemas.py \
  tldw_Server_API/tests/Admin/test_admin_ops_webhooks_reports.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_admin_webhook_migration_sqlite.py \
  tldw_Server_API/tests/Security/test_egress.py \
  tldw_Server_API/tests/Workflows/test_webhook_admin_endpoints.py
```

Final post-review, final-rebase proof was run in two explicit provider groups:
`460 passed, 6 warnings in 46.72s` for the complete non-PostgreSQL matrix and
`24 passed, 50 warnings in 97.37s` for the required PostgreSQL matrix. No test
was skipped. Together these runs execute all 484 current test bodies across
SQLite, PostgreSQL, API, authorization, egress, system-ops, and workflow paths.

One restricted-sandbox attempt failed only because the unchanged workflow test
creates `Databases/test_wf_dlq.db` inside the external worktree. The exact test
passed with normal worktree write access, followed by the complete passing run
above under the same valid permissions.

An earlier pre-review complete run executed all 466 then-current test bodies but
exposed a teardown error in an egress timing test. That test had patched
`time.monotonic` on the shared standard-library module, which also affected
asyncio teardown. Its fake clock now replaces only the `egress.time` module
binding. The focused regression passed, followed by the later complete passing
runs.

Required PostgreSQL matrix:

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. \
  ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_legacy_import_postgres.py
```

Result: `24 passed, 50 warnings in 97.37s`; zero skips. The required flag was
set, and the tests used the running disposable PostgreSQL 18.6 container rather
than SQLite or an availability skip.

## Pre-PR Review Corrections

Two independent review passes found and closed the following issues before the
source commit was frozen:

- rollback-backup resume now compares the authenticated backup's strict-parsed
  webhook subtree with the durable source fingerprint, allowing unrelated
  `system_ops` changes without accepting a different webhook source;
- rollback extraction performs a final eligibility check under the shared
  migration-state lock and holds that lock through plaintext publication, so
  canonical activity and artifact retirement serialize on both backends;
- cancellation before transaction exit removes only the exact plaintext inode
  created by that invocation, while a replacement pathname is preserved;
- mandatory audit failures survive both shared transaction wrapping and
  repository busy-cause inspection, including lock-shaped underlying causes;
- projected registration capacity excludes tombstones while their IDs remain
  reserved for deterministic collision allocation.

Regression coverage includes both rollback-closing lock orders for activity and
retirement on SQLite and PostgreSQL, fail-once audit sinks, cancellation and
replacement-inode cleanup, unrelated-store resume, authenticated wrong-subtree
rejection, and tombstone/live-count parity. The three newly exposed defects
first produced the expected red result (`3 failed, 3 passed`); the corrected
focused matrix then passed `6/6`.

## Qodo Review Corrections

Qodo review comment `5383466884` reported zero bugs and six compliance
findings. All six were accepted and corrected:

- best-effort read-audit failures now emit a sanitized warning containing only
  the static action, normalized request ID, and exception type; exception text
  and request data are not logged;
- the redacting route handler now declares its response type, and both the
  handler and request-ID helper have explicit docstrings;
- `WebhookError` is defined in `app/core/exceptions.py` and remains available
  through the existing domain import contract;
- the 99%-similarity repository rename places all canonical webhook SQL and
  unit-of-work code in
  `app/core/DB_Management/admin_webhooks_repository.py`;
- production, test, CLI, and implementation-plan imports point to the new
  database boundary, and no raw SQL remains under `app/core/Admin_Webhooks`;
- every test function in `tests/Admin_Webhooks` and the six additional
  webhook-related PR test modules now has exactly one direct accepted unit or
  integration marker; inherited accepted markers and redundant `asyncio`
  markers were removed while required `postgres` execution markers remain.

Focused architecture tests first reproduced the three material gaps as
`3 failed`, then passed `3/3` after correction. Qodo's incremental review then
showed that module-level markers did not satisfy its direct-marker rule. An AST
regression reproduced that policy gap across all PR-related webhook tests,
including 70 additional unmarked functions outside `tests/Admin_Webhooks`, and
now fails if a covered function has zero or multiple accepted direct markers.
The six expanded modules passed `108/108` with 2 warnings. The complete
`tests/Admin_Webhooks` scope passed `325/325` with 142 warnings. The explicit
required-PostgreSQL matrix passed `24/24` with 50 warnings and zero skips. The
complete PR 1 matrix passed `483/483` with 459 warnings and zero skips.

Ruff, Bandit, `git diff --check`, the sensitive-logger scan, raw-SQL boundary
scan, and stale-import scan all passed. Mypy passed all seven other changed
production modules. A direct check of the newly touched centralized
`app/core/exceptions.py` still reports its three pre-existing errors at
unchanged code outside this PR's hunks (current lines 127, 748, and 1503); the
new exception definition and all dependent changed modules type-check.

## Static And Sensitive-Data Gates

Ruff was run over the complete `Admin_Webhooks` package, touched AuthNZ,
security, system-ops, API, schema, and CLI sources, plus the Task 11 regression
tests. Result: `All checks passed!`.

Bandit command:

```bash
../../.venv/bin/python -m bandit -q -r \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/core/Security/egress.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/services/admin_system_ops_service.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py \
  tldw_Server_API/app/api/v1/endpoints/admin/__init__.py \
  tldw_Server_API/cli/admin_webhooks_cli.py \
  tldw_Server_API/cli/commands/admin_webhooks.py
```

Result: PASS with no findings. Bandit printed only diagnostics for existing
`nosec` annotations whose selected rule was not triggered.

Both required ripgrep gates passed with no matches:

- no canonical logger/audit metadata pattern carrying URL, secret, payload, or
  response body;
- no newly added compatibility logger/audit metadata pattern carrying those
  values relative to `origin/dev`.

After the pre-PR corrections, Ruff passed on every modified production/test
file, mypy reported no issues in the three modified production modules, Bandit
reported no findings in those modules, and `git diff --check` passed.

## Admin UI Gates

Focused tests covering the webhook page, authenticated proxy, HTTP helper,
server-auth forwarding, API client, privileged confirmation, and in-memory
idempotent commands:

```bash
bunx vitest run \
  app/api/proxy/__tests__/route.test.ts \
  app/webhooks/__tests__/page.test.tsx \
  components/ui/privileged-action-dialog.test.tsx \
  lib/__tests__/idempotent-command.test.ts \
  lib/api-client-webhooks.test.ts \
  lib/http.test.ts \
  lib/server-auth.test.ts --reporter=dot
```

Result: `7 passed` files, `77 passed` tests.

The following commands passed under Node 20.19.5:

```bash
bun run typecheck
bun run build
bunx playwright test tests/e2e/webhooks-control-plane.spec.ts --reporter=line
```

The production build compiled, typechecked, and generated all 49 static pages,
including `/webhooks`. The final unmodified Playwright auto-start command passed
one Chromium journey. It proves create/copy/acknowledge, redaction, fresh ETag
review, 412 re-review, exact-key lost-response rotation replay, synchronous
page-exit secret clearing, deletion, and absence of canonical delivery/test
requests.

The first Node 20 browser run exposed a generic admin middleware defect before
the page rendered: converting the decoded JWT signature to `.buffer` produced a
realm-sensitive `ArrayBuffer` rejected by Next Edge WebCrypto. Passing the
already decoded typed array directly, with its actual `ArrayBuffer` type
contract, fixed the supported runtime. Typecheck and the exact browser journey
then passed under Node 20; a Node 24 compatibility run also passed. One later
auto-start attempt timed out waiting for the dev server without executing a
test. The exact injected environment and journey passed manually, and the final
unmodified auto-start command passed; no server process was left running.

The required UI sink scan found no `localStorage`, `sessionStorage`,
`document.cookie`, or `console.*` use in the webhook page, idempotent command,
API client, or webhook types.

## PR CI Remediation Follow-up

CI remediation was performed from exact PR head `89f03feda2` and frozen as
source commit `575616ed8b`. The changes are limited to CI scheduling, a
cross-platform chat timestamp normalization defect, a time-dependent test
fixture, and the three package lint errors described in the original evidence.

The Admin Webhooks directory is assigned to the dedicated
`admin-webhooks-canonical` shard in all five duplicated workflow matrices.
The legacy `admin-watchlists-webhooks` shard remains restricted to
`tldw_Server_API/tests/Admin/test_admin_w*.py`.
The repository guard passed:

```text
[shard-coverage] shards=783 test_files=4413 ignored=4 baseline=130 new_uncovered=0
```

The chat regression was established with a red/green cycle. At the unmodified
PR head, a whole-second value normalized to `2026-08-23T04:28:54Z`; the new
regression requires the same UTC millisecond representation used by persisted
SQLite values, `2026-08-23T04:28:54.000Z`. After applying one common formatter
to numeric, string, and datetime inputs:

- the timestamp unit regression passed;
- the exact previously failing
  `test_chat_completions_save_to_db_persists_and_exposes_conversation` E2E
  passed;
- the four-test surrounding chat-persistence set passed.

Scheduling the previously unassigned webhook suite exposed a test-only
wall-clock dependency. Its fixed rotation cutoff used the module's `NOW`, but
the replay-secret expiry used `datetime.now() - 1 day`; after the real clock
passed the fixed cutoff, the supposedly expired row entered the inventory.
The fixture now derives both sides from `NOW`. The isolated regression and all
301 locally runnable, non-PostgreSQL Admin Webhooks tests passed. The 24
PostgreSQL cases were not rerun after this fixture-only correction because no
local PostgreSQL service was configured; their most recent required-provider
run remains the recorded `24 passed, 0 skipped` result at the direct-marker
source.

The package lint errors are removed without suppressions. The security-header
test uses an ESM import, and the active configuration is now
`next.config.mjs` with static ESM plugin imports and equivalent conditional
wrappers. Under Node 20.19.5 and Bun 1.3.2:

```text
bun run lint:      PASS, 0 errors and 41 unchanged warnings
bun run typecheck: PASS
security headers:  3 passed
bun run build:     PASS, 49 static pages generated
```

The build's first restricted-sandbox attempt could not bind Turbopack's local
port. The same command passed with normal process permissions; this was an
environment restriction, not an application failure.

## Refreshed Qodo Review

A manual Qodo review at remediation head `069c8c94cf` reported no security
concerns and one pagination observation: the public control-plane `list()`
method forwarded `limit`, `offset`, and `before_id` without repeating the
bounds enforced by `list_page()`.

The observation did not expose a database bypass. Both SQLite and PostgreSQL
use the same unit-of-work implementation, which rejects limits outside 1-100,
offsets outside 0-1,000, and non-positive `before_id` values before executing
the query. Repository-wide caller and design review found a narrower issue:
`list()` was an unsupported dead keyset-pagination surface. The canonical HTTP
route, approved design, and admin UI all use bounded offset pagination through
`list_page()`; only three control-plane test assertions called `list()`.

Source commit `e54701a402` removes that dead method and moves the gate/read
assertions to the supported `list_page()` path. Verification after the change:

```text
affected control-plane/API tests:        2 passed
complete control-plane module:           40 passed
non-PostgreSQL Admin Webhooks suite:      301 passed, 24 deselected
focused Ruff:                            PASS
focused mypy:                            PASS
```

The first complete control-plane run identified the two additional test-only
gate calls as `39 passed, 1 failed`; after moving both to `list_page()`, the
fresh complete run passed `40/40`. No repository, SQL, migration, or persistence
code changed, so the most recent required-PostgreSQL proof remains applicable.
Qodo marked `/review` deprecated in favor of `/agentic_review`; the latter is
the required post-push confirmation command for this source.

## Agentic Qodo Follow-Up

Qodo's refreshed agentic review at head `a66e255495` reported three rule
violations. Source commit `ebbe6d30da` addresses all three:

- renamed the active Next configuration to `next.config.mjs`, replaced both
  conditional CommonJS imports with static ESM imports, and removed both lint
  suppressions while retaining conditional analyzer and Sentry wrapping;
- removed the exact mocked `logger.warning` call assertion from the read-audit
  failure test while preserving the observable successful response and
  response-redaction assertions;
- removed redundant `asyncio` markers from the complete PR-touched Python test
  surface because repository `asyncio_mode=auto` supplies execution handling;
  direct unit/integration classifications and PostgreSQL execution markers are
  unchanged.

Verification against that source:

```text
focused API/control-plane/system-ops tests: 42 passed, 6 warnings
affected non-PostgreSQL Python matrix:      323 passed, 24 deselected
focused Ruff:                              PASS
redundant-marker/suppression scan:          PASS, no matches
admin security-header tests:               3 passed
admin package lint:                        PASS, 0 errors, 41 baseline warnings
admin typecheck:                           PASS
admin production build:                    PASS, 49 pages
analyzer and Sentry config branches:       PASS
git diff --cached --check:                 PASS
```

The first production-build attempt failed only because the restricted sandbox
forbids Turbopack's internal local port bind. The identical Node 20 command
passed with normal process permissions. These Python changes are test-only and
do not alter repository, migration, or PostgreSQL behavior, so the prior
required-provider proof remains `24 passed, 0 skipped`. Post-push Qodo
confirmation and GitHub-hosted CI remain pending.

## Eager-Import Qodo Follow-Up

Qodo's refreshed review at PR head `90907cd2de` identified one reliability
bug: `next.config.mjs` statically imported the development-only
`@next/bundle-analyzer` package, so a production-only dependency install could
fail while loading the config even when `ANALYZE` was disabled. The current
Docker builder installs all dependencies, but that does not make the config
safe for other supported build environments.

Source commit `7c104a7500` replaces the static import with a guarded dynamic
import that is evaluated only when `ANALYZE=true`. A focused regression rejects
eager analyzer imports while the existing config tests continue to exercise the
disabled path and security settings.

Independent review found no implementation or ESM/Next.js compatibility defect,
but correctly identified that the original source-regex regression could miss an
unconditional dynamic import or a multiline static import. Follow-up test commit
`1263677011` replaces that check with an isolated Node loader: analyzer
resolution is forced to fail while disabled, while the enabled path receives a
stub analyzer and must apply its wrapper. Mutating the config back to the buggy
static import produced the expected RED result; restoring the guarded import
returned the suite to GREEN.

Independent re-review of `1263677011` reported no remaining findings. The
reviewer separately reproduced the `4/4` focused pass on Node 20, 22, and 24,
plus targeted ESLint and `git diff --check` passes.

Qodo's next pass at pushed head `a1e4557489` reported one performance-rule
violation: the async regression used `spawnSync` for its isolated child
processes. Test commit `837b209884` replaces that call with promisified
`execFile`, preserving child failures as rejected test promises without
blocking the Vitest event loop. The static-import mutation again failed `1/4`
for the expected eager-resolution error; the restored guarded import passed
`4/4` under the current Node runtime and separately under Node 20. Targeted
ESLint, typecheck, synchronous-child scan, and `git diff --check` passed.
Independent re-review of `837b209884` reported no findings and separately
verified the focused test on Node 20, 22, and 24, forced child-error diagnostics,
shell-free invocation portability, and temporary-directory cleanup.

The refreshed Qodo review is bound to pushed head `8b00250005` and reports
`0` bugs and `0` rule violations. GitHub GraphQL reports no unresolved review
threads. Because the PR is merge-conflicted against the advanced `dev` branch,
normal `pull_request` workflows are not available at this head; the complete
hosted CI gate remains assigned to the final post-PR-2808 rebase.

```text
behavioral mutation RED:                   1 failed, 3 passed
behavioral guard GREEN:                     4 passed
Node 20 ANALYZE=false config load:          PASS
Node 20 ANALYZE=true config load:           PASS
targeted ESLint:                            PASS
admin package lint:                         PASS, 0 errors, 41 baseline warnings
admin typecheck:                            PASS
admin production build:                    PASS, 49 pages
git diff --check:                           PASS
```

The first production-build attempt failed only because the restricted sandbox
forbids Turbopack's internal local port bind. The identical Node 20 command
passed with normal process permissions. No dependency, runtime image, webhook,
database, migration, or PostgreSQL behavior changed; the prior required-
PostgreSQL proof remains applicable.

## Final Rebase Reconciliation

PR #2808 advanced `dev` through
`9ee0b5a16dca9f5cf6372a3dd2798b84075501fc` before PR #2806 could merge. All
32 PR commits were rebased onto that exact base. The rebase preserved current
`dev` behavior and made the following explicit reconciliations:

- current `dev` owns SQLite AuthNZ migrations 091 through 093 for profile
  versioning, candidate timestamp hardening, and users write-column
  harmonization; the canonical webhook migration is now additive migration
  094, including registry, rollback, upgrade, and implementation-plan
  references;
- PostgreSQL startup executes user-timestamp, AuthNZ core, canonical webhook,
  sharing, and remaining readiness ensures together while preserving the
  current readiness-error contract;
- direct unit/integration markers remain on every covered test, redundant
  module-level and `asyncio` markers were removed, and the current `dev`
  startup tests were included in the marker-policy gate;
- the OpenAPI fingerprint was regenerated from the rebased application and
  drift-checked at 2,039 paths, 3,031 schemas, and SHA-256
  `41d99488f7bb295e7c20d6c05085f788e54619c1079fbbf8053522c7dde949a9`;
- current `dev` sanitizes ordinary exceptions leaving a managed transaction.
  `WebhookError`, `WebhookRepositoryError`, and `LegacyImportError` now share a
  closed, documented `TransactionPassthroughError` marker so their already
  sanitized fixed error codes survive rollback; arbitrary runtime exceptions
  remain translated to generic SQLite or PostgreSQL `TransactionError`
  messages;
- PostgreSQL test-only `DROP` and `TRUNCATE` setup now uses a plain connection
  to the ephemeral test database, matching the shared AuthNZ cleanup pattern,
  while every application query remains on the managed users-write firewall.

The first complete current-base run provided the expected RED evidence:
`18 failed, 442 passed, 24 skipped`. Eighteen failures came from the new
transaction sanitization contract; the skips reflected the sandbox's inability
to reach the host-published disposable PostgreSQL port. After the transaction
reconciliation, the affected repository, control-plane, importer, and rotation
set passed `114/114`, and the complete non-PostgreSQL matrix passed `460/460`.

Running the required PostgreSQL matrix with normal localhost access then
exposed the current users-write firewall in destructive test setup and two
remaining raw-message assertions (`6 failed, 4 passed, 14 errors`). The two
representative fixture paths passed after moving destructive DDL to an
unmanaged test-only connection; the corrected rollback pair passed `2/2`; and
the final required-provider run passed `24/24` with zero skips against
PostgreSQL 18.6.

Final source commit `2ad82ff4bcc7faee3cac127aa31a11733c5eb550`
therefore has the following local proof:

```text
affected transaction matrix:              114 passed
complete non-PostgreSQL PR matrix:         460 passed, 0 skipped
required PostgreSQL matrix:                 24 passed, 0 skipped
direct pytest marker policy:                 1 passed
OpenAPI drift check:                        PASS
focused Ruff, reconciliation files:         PASS
Bandit, reconciliation production files:    PASS
focused mypy, repository/startup modules:    PASS
git diff --check:                            PASS
```

Broader diagnostic sweeps still show unrelated current-tree baselines outside
the reconciliation hunks: `chat.py` has pre-existing import-layout findings,
the OpenAPI schema example produces two low-confidence Bandit B105 false
positives, and the large AuthNZ database/legacy-import modules retain existing
mypy debt. These were not rewritten in this already broad PR. The focused
changed modules and all runtime provider matrices above are green.

The prior Qodo result at pre-rebase head `8b00250005` reported zero bugs and
zero rule violations. The first refreshed post-rebase review had not completed,
but a GitHub review-thread audit exposed one unresolved testability finding in
the analyzer configuration regression. That finding is remediated and verified
below. A new Qodo review and complete GitHub-hosted CI run against the pushed
exact head remain mandatory; this local evidence does not substitute for those
gates.

## Exact-Head Qodo Testability Remediation

Qodo reported that the analyzer configuration regression combined the disabled
and enabled branches in one Vitest case. The finding was valid: although the
isolated child-process harness covered both branches correctly, a failure did
not identify which mode had regressed.

Test-only source commit `ca304f4b00536593a70844db268a7baa14886d7f`
extracts the temporary loader setup into a shared helper and gives each branch
an independent test. No production file or application behavior changed.

```text
security-header/analyzer tests:             5 passed
targeted ESLint:                            PASS
admin-ui package typecheck:                 PASS
admin-ui package lint:                      PASS (0 errors, 41 baseline warnings)
git diff --check:                           PASS
```

The two unrelated untracked watchlist templates remained excluded. The Qodo
thread must be resolved and the review rerun against the pushed head before
review closure.

## Exact-Base Admin UI CI Ratchet

GitHub Actions run `32801997622` passed every other required check reported on
PR #2806. Its only failure was `frontend-required`, where the package-wide
admin UI Vitest command ended at 41 failed and 713 passed tests across 17
failed files. The 16 focused webhook page tests, package lint, typecheck,
frontend unit shards, backend/security checks, CodeQL, and the reported E2E
smoke and onboarding/UX checks passed in that hosted run.

Source commit `a53fe714f9610c6b908504381838409a039051ba` introduced the
exact-base ratchet. Reviewer-remediated source
`dac56c2004b859103ceef393f023927014a988da` closed the initial report,
provenance, and runner gaps. Latest remediation source
`517d7b016089e220fa55eab3483f212031b6f5cb` closes the subsequent Qodo,
independent-review, and old-head CI findings. Test-only source
`7253450461a58f0724fba77de84c97e5ec26b548` closes the valid findings from
the next exact-head Qodo pass. The workflow now:

1. installs frozen head dependencies and runs the complete head suite with
   human-readable, JSON, and minimal safety reporters;
2. rejects zero-test runs, missing file-level messages, assertion or nested
   suite counter inconsistencies, unhandled errors, module/suite errors, and
   failed or inconsistent hook lifecycles;
3. fingerprints each failed assertion by package-relative file, full name,
   normalized failure messages, and duplicate multiplicity;
4. extracts only validated failed files, rejects every changed failed-test
   file, and replays those files from a detached worktree at the exact base;
5. uses NUL-delimited repository-relative changed paths and `./`-prefixed
   replay paths, then permits only exact unchanged base failure fingerprints;
6. pins the checked-out helper and safety reporter to workflow constants with
   SHA-256, then revalidates after the head run, before base replay, and before
   comparison;
7. publishes manual runs as `frontend-required-diagnostic`, so a manually
   dispatched diagnostic cannot satisfy the protected required-check name.

The digest check has an event-dependent trust boundary. A default-branch
`workflow_run` authenticates the checked-out head artifacts against the
default-branch workflow definition. A direct `pull_request` run is only a
same-workflow consistency check because the PR controls the workflow, digest
constants, and checked-out artifacts. A manual `workflow_dispatch` inherits
the trust of the selected workflow ref. The final pinned digests are:

```text
ratchet helper:  1426966c059f9ff8080e33ffedba02cf6c3794369251283d715b4342908890e4
safety reporter: 433e8ab9a163694775fa4a50ceae2f7722358331d1f8f4426ec7ec31e36e93f3
```

The final exact comparison used Node 20.19.5, Bun 1.3.2, and base
`9ee0b5a16dca9f5cf6372a3dd2798b84075501fc`:

```text
full head:             41 failed, 728 passed, 769 total, 17 failed files
head suites:           346 total, 312 passed, 34 failed, 0 pending
head safety:           modules=146, unhandled=0, module=0, hook=0
exact-base replay:     41 failed,  89 passed, 130 total, 17 failed files
base suites:            43 total,   9 passed, 34 failed, 0 pending
base safety:           modules=17,  unhandled=0, module=0, hook=0
ratchet comparison:    inherited=41 regressions=0, exit 0
```

The final failed-file list exactly matched the prior strict run. None of those
17 test files is changed by the PR. Earlier synthetic verification marked
`admin-ui/lib/navigation.test.ts` as changed and was rejected with two
regressions, proving that an otherwise inherited failure becomes blocking when
its owning test file changes.

Qodo comment `5383466884` was refreshed against the previous pushed head and
reported four valid findings. The remediation prevents request-controlled
hostnames and protocols from selecting a credential-bearing backend, replaces
synchronous filesystem calls in async reporter tests, documents every
`compare_reports` option and failure contract, and honors project-specific
backend URL overrides. The request now selects only the known `3101` or `3102`
project; the destination comes from its explicit project environment variable
or trusted loopback default.

Independent review additionally found a digest revalidation gap, incomplete
strict suite-metadata reconciliation, direct manual-input interpolation into
Bash, a manual required-check self-ratchet path, swallowed teardown failures,
and incomplete hook-lifecycle coverage. All were reproduced or inspected and
closed. The reviewer suggestion to equate suite count with file count was not
adopted because real Vitest output contains nested suites: the final report has
346 suites across 146 files. The helper instead derives each file root and
every `ancestorTitles` prefix, applies failed-over-passed-over-pending status
precedence, and exactly reconciles the real head and base reports.

The prior remote head's hosted run failed all eight frontend unit shards for a
single workflow-policy reason: dependency impact exceeded 500 tests and no
frontend test file was directly changed. The workflow previously exited before
sharding. It now retains and shards the complete dependency-impact set in that
case; directly changed tests remain the bounded fallback when such tests exist.

A production build/start proof used JWT and single-user overrides on ports
`9101` and `9102`, started the single-user UI on `3102`, and sent a request with
`Host: attacker.example:3102` plus an API-key cookie. The UI returned 200 while
the mock backend recorded `GET /api/v1/users/me`, the API-key header, and
`Host: 127.0.0.1:9102`. The hostile hostname received no request or credential.

Qodo's exact-head pass at `c97460d31b` then reported one bug and two testability
findings. The two testability findings were valid and are fixed in
`7253450461`: all three new Python helper tests now carry direct `unit` markers,
and the readiness-route test asserts the observable trusted URL passed to
`fetch` rather than an internal helper call.

The reported suite-count bug was not valid. The actual Node 20/Vitest 4.0.18
report contains 346 suites across 146 files, while independently deriving each
file root and every `ancestorTitles` prefix also yields exactly 346. Treating
files as suites would therefore reject the real report. Vitest's own
`JsonReporter.onTestRunEnd` computes these counters from `getSuites(files)`, not
from `files.length`. The nested-suite regression now states that semantic
contract explicitly and also proves its fixture contains one file but three
suites: the file root, outer describe, and inner describe.

TDD and final verification:

```text
real failed-beforeEach lifecycle regression:  RED before hook accounting
all four hook lifecycle cases:                 PASS
reporter/routing/health/teardown matrix:        5 files, 21 passed
ratchet helper/workflow contracts:             34 passed
embedded workflow Bash syntax:                 22 steps passed bash -n
strict exact-base comparison:                  inherited=41 regressions=0
Ruff and git diff --check:                     PASS
admin package lint:                            PASS, 0 errors, 41 baseline warnings
admin package typecheck:                       PASS
admin production build:                       PASS, 49/49 pages
real-backend JWT project:                      26 passed, 1 expected skip
real-backend single-user project:              1 passed, 26 expected skips
```

`actionlint` was not installed in the local environment. The workflow parsed
through PyYAML in the 34 passing contract tests, and all 22 Bash `run` blocks
passed `bash -n`; GitHub's exact-head workflow validation remains required.

The complete local `tests/CI` collection finished with 225 passed and two
failures. Both reproduce unchanged on the exact base: the stale
`ui-watchlists-extension-e2e.yml` route manifest and a PostgreSQL schema test
that rejects the valid `public.users` qualification. They are documented as
current-base defects and were not folded into this already broad webhook PR.

## Frontend Dependency-Impact Shard Remediation

The exact-head frontend unit job `98053488515` failed in shard 5 after all
other completed checks had passed. The shard ran 1,942 tests across 522 suites.
It reported 10 failed assertions and one collection failure caused by the
unchanged `Playground.cockpit-shell.test.tsx` mock omitting the upstream
`LEGACY_SERVICE_PROMPT_DEFAULTS` export. The test and its source import chain
are unchanged from exact base `9ee0b5a16dca9f5cf6372a3dd2798b84075501fc`.

The first exact-base policy replayed only the seven files reported as failed.
That isolated replay reproduced six failed assertions plus the collection
failure on head and base, yielding `inherited=7 regressions=0`. Five other
failures from the hosted shard passed in isolation because they depend on the
complete shard's module and test-order context. Treating that mismatch as a
product regression would be a false red; accepting it without an exact-base
comparison would be a false green.

The package ratchet now keeps its fast failed-file replay. When that
well-formed comparison returns the normal regression status `1` after the head
ran in dependency-impact mode, it extracts the validated execution manifest
and replays those modules at the exact base. This applies to bounded dependency
impact and the existing greater-than-500/no-direct-test fallback; direct-test
mode remains bounded and does not enter context replay. The explicit manifest
is not sharded a second time.

Both revisions run through a pinned path-ordering sequencer, and a separately
pinned reporter records every `onTestModuleStart` event. This distinction is
required because Vitest 4.0.18 sorts the execution pool through the sequencer
but gives its built-in JSON reporter the original pre-sequencing specification
array. A first hardened comparison correctly rejected the head/base JSON array
order even though sorting the assertion-bearing file start timestamps showed
the same order. The final policy does not infer execution order from JSON.
It validates the reporter sidecar against the JSON module set, extracts the head
manifest in observed runtime order, and requires a byte-equivalent base runtime
order. Test identities and file-local order, suite identities and statuses, and
every reconciled test/suite counter must also match before failures are compared.

Fast and context replays use separate, newly-created detached worktrees for
each package, each with a frozen dependency install. This prevents Vitest cache
state or test side effects from crossing packages or replay modes. The workflow
rejects non-test Vitest exit codes, malformed comparator status,
absolute/traversal/CR paths, files absent from either revision, empty context,
and context above 5,000 modules. Extracted paths are prefixed with `./` before
reaching Vitest.

Non-strict package ratchets can now compare a zero-assertion collection
failure by package-relative file plus its exact normalized diagnostic message.
Checkout-specific package and repository roots are normalized while diagnostic
suffixes remain exact. Missing diagnostics, changed diagnostics, failures in
changed test files, and file-level failures after assertion collection remain
blocking. Strict admin UI validation still rejects every file-level error
before this comparison, so its safety contract is unchanged.

Final review also closed the remaining fail-open edges. Package comparisons now retain
duplicate failure multiplicity and require normalized assertion diagnostics in
both strict and non-strict modes. Package reports also reject a nonempty
file-level diagnostic whenever assertions were collected, so an inherited
assertion cannot mask a new hook error; a nonempty file diagnostic is permitted
only for a failed zero-assertion collection result in package mode.
Runtime-order sidecars accept only integer schema version `1`. Suite
reconciliation now follows observed Vitest 4.0.18 behavior for completed
skipped/disabled assertions: the assertions remain pending-test counts while
their completed containing suites count as passed. A captured all-skipped
Vitest JSON report is retained as the regression fixture.

Fresh local proof against the exact base:

```text
dependency-impact discovery: 2,028 candidate files
head shard 5 manifest:        253 unique runtime-ordered modules
fresh exact-base manifest:    253 byte-identical runtime-ordered modules
head shard 5:                 1,942 tests, 522 suites, 10 failed assertions
fresh exact-base replay:      1,942 tests, 522 suites, 10 failed assertions
test identities/status/order: exact match after canonical module serialization
suite identities/statuses:    exact match with counters reconciled to structure
file-level failures:          1 matching collection diagnostic on each side
ratchet comparison:           inherited=11 regressions=0, exit 0
Python CI contracts:          102 passed, 6 warnings
Ruff:                         PASS
embedded workflow Bash:       PASS (bash -n)
git diff --check:             PASS
```

The helper remains pinned at both package and strict admin ratchet call sites.
The package-only sequencer and runtime-order reporter are pinned with it:

```text
ratchet helper:          1426966c059f9ff8080e33ffedba02cf6c3794369251283d715b4342908890e4
path sequencer:          5393dd2f8652fe784c0cc268f3b52b4438bedb7dfdce23e579babf972c42dafd
runtime-order reporter:  6c0c06d6a85b8638868d63e243feb757e61ba399c1f5a1da9c55d8e2fb17cb8e
```

This is local exact-revision evidence. A new pushed source commit, refreshed
Qodo review, and exact-head GitHub Actions run remain required before the task
or PR can be considered complete.

## Upstream Admin UI Baselines

The measurements below are historical pre-ratchet evidence. The authoritative
current comparison and acceptance policy are recorded in the preceding
section.

`bun run test` is not green on the exact PR base. An isolated detached worktree
at `d736368d17` using a clean frozen install, Node 20.19.5, and Bun 1.3.2
produced:

```text
47 failed, 653 passed, 700 total
```

The final remediation source, under the same clean-install conditions,
produced:

```text
42 failed, 710 passed, 752 total
```

The branch therefore improves the base count by five failures while adding its
new tests. One other branch run returned `41 failed, 711 passed`, exposing an
unrelated timing fluctuation. Stable representative failures reproduce in
isolation: BYOK, Plans, and Resource Governor tests render production components
without the required `ConfirmProvider`, while navigation tests retain stale
section-order and href assertions. Changed-only Vitest mode still reports 33
failures out of 394 tests, so this remediation did not weaken or bypass the
required package gate.

The package lint gate is now clean of errors:

```text
41 problems: 0 errors, 41 warnings
```

The warnings are unchanged upstream debt. Targeted ESLint across every changed
TypeScript/TSX file, including `middleware.ts` and the Playwright helpers,
passed with zero findings.

An earlier two-project invocation attempted to start concurrent Next 16 dev
servers from one `admin-ui` tree and failed on the shared `.next/dev` lock
before browser tests. Final source `dac56c2004b859103ceef393f023927014a988da`
resolves that runner defect. `bun run test:real-backend` now creates one
production-mode E2E build, then runs the JWT and single-user Playwright
projects sequentially in separate `next start` processes. Each process starts
only its requested Python backend and UI server. Standalone output is disabled
only for that E2E build; the normal production build retains standalone output.

The one-build runner exposed and fixed a separate routing defect: single-user
login used the request-mapped backend on port 8102, but middleware cookie
revalidation used the build-time port 8101. Middleware now resolves the backend
from each request while preserving the existing production fallback outside
the explicit real-backend E2E mode. Final results were:

```text
JWT process:         26 passed, 1 expected project skip
single-user process:  1 passed, 26 expected project skips
normal build:         49/49 pages
```

Build and both project runs left tracked `next-env.d.ts` and `tsconfig.json`
unchanged. The historical package failures above are not represented as a
clean suite; they are governed by the strict exact-base ratchet.

## Final Safety Checks

- `git diff --check`: PASS at tested source commit
  `7253450461a58f0724fba77de84c97e5ec26b548`.
- Exact-base admin UI comparison: PASS with 41 inherited failures and 0
  regressions; head and base unhandled/module/hook safety counters are all zero.
- Real-backend admin UI: PASS in sequential JWT and single-user processes;
  normal standalone production build generated all 49 pages.
- Canonical Admin Webhooks tests have a dedicated shard in all five full-suite
  matrices; shard coverage reports 0 newly uncovered files.
- OpenAPI evaluation-webhook schema isolation: PASS.
- Canonical mode default remains `off`.
- Outbound HTTP, Jobs delivery workers, automatic event producers, test sends,
  delivery history, and activation readiness remain absent from PR 1.
- Two unrelated untracked watchlist template files were excluded from every
  commit and verification artifact.

## Final Rebase Onto Advanced Dev

After source `51f204b5c1ed779bc335088af673f227f715bc89` was pushed,
`dev` had advanced by 106 commits and GitHub reported the PR as conflicted. The
45 PR commits were rebased onto exact base
`2306c1939f3b460f9c62da8ae83a1aa47c02ee0d`. The only conflict was the
generated OpenAPI fingerprint. It was regenerated from the fully rebased API,
and the frontend API generation command completed with the dependency-complete
Python 3.11 environment:

```text
OpenAPI paths:    2,040
OpenAPI schemas:  3,037
OpenAPI SHA-256:  a4c4b6b24aea9cfbaa4102595c29988e5ed8041e10574ff798773bf0d0ef09b4
drift check:      PASS
```

The rebase exposed an inherited `dev` CI coverage gap. Two new
ChaChaNotesDB moodboard-studio tests were absent from every full-suite shard,
and the required workflow contract failed identically on exact base. Both
files are now assigned to the existing `chacha-content-persona` shard in all
five matrices. Running the resulting 95-case shard order exposed one more
exact-base test defect: its finite fake clock replaced the process-wide
`time.monotonic`, so unrelated logging could consume a sample and raise
`StopIteration`. The production migration was unchanged; the test now replaces
only the `ChaChaNotes_DB.time` module reference with a deterministic fake.

Final post-rebase local evidence:

```text
CI ratchet/workflow contracts:       116 passed, 2 warnings
frontend-required embedded Bash:      22/22 syntax checks passed
new ChaChaNotesDB shard files:         67 passed, 28 expected PostgreSQL skips
Admin_Webhooks non-PostgreSQL suite:  302 passed, 24 deselected
admin webhook UI focused suite:        77 passed across 7 files
admin UI typecheck:                    PASS
OpenAPI drift check:                   PASS
focused Ruff and py_compile:           PASS
```

The initial 102-test ratchet/workflow subset omitted the license-first freeze
contracts. Adding that module to the final local gate exposed three stale
expectations from this PR's intentional exact-base workflow changes: the
`workflow_dispatch` trigger digest, the direct pull-request base-reference
count, and the removed inline event-branching script. The contract now freezes
the required `base_sha` dispatch input and verifies both frontend and admin
ratchet steps receive the same structured `RATCHET_BASE_SHA` expression. The
complete 116-test contract set passes after that reconciliation.

The upstream frontend typecheck reports 80 errors, but the extracted error
lines are byte-identical on exact base `2306c1939f`; none of the reported source
files differs in this PR. That inherited base failure is not represented as a
PR regression. The worktree-local OpenAPI Make target lacked FastAPI, and the
first frontend generator attempt selected a pre-3.10 system Python; both were
environment-only failures before the successful Python 3.11 generation above.

The 253-module deterministic ratchet proof in the earlier section is bound to
the pre-rebase exact revisions. A fresh Qodo review and complete hosted CI run
at the rebased PR SHA remain mandatory; those hosted checks are the final
exact-head proof against the new base.

## Five-Finding Qodo Remediation

Source commit `c253e20467` addresses the five active findings in Qodo comment
`5383466884` at prior PR head `bdf3bdfec8`:

- canonical create, legacy create, and destination replacement now share an
  explicit browser-side invariant URL validator; deployment-specific HTTP,
  DNS, and egress decisions remain authoritative on the backend;
- `WebhooksPageContent` retains presentation while control-plane loading,
  pagination/conflict recovery, one-time-secret/retry state, and CRUD/form
  orchestration live in focused hooks and a composing controller;
- proxy route tests mock backend `fetch` but exercise the production
  `buildProxyResponse` and assert the returned public `Response` contract;
- the execution-order reporter is imported and exercised behaviorally for
  ordered path normalization, exact JSON, atomic replacement, temporary-file
  cleanup, and the CI environment-path constructor; and
- the newly added ratchet status line uses `sys.stdout.write` with an inline
  dependency-boundary explanation. The frontend workflow invokes this helper
  before installing any Python dependencies, so importing Loguru there would
  break the gate. Both helper digest pins were refreshed.

TDD reproduced the URL gap as three failing page cases before the validator
was wired. The proxy mutation proof failed 1 of 16 tests when the mocked
response builder copied a backend `content-length`; the same assertion passes
only after the test exercises the production response builder.

Final local evidence for source `c253e20467`:

```text
focused admin UI matrix:              94 passed across 9 files
admin UI package typecheck:           PASS
admin UI package lint:                PASS, 0 errors/41 inherited warnings
admin UI production build:            PASS, 49/49 routes
Chromium webhook lifecycle:           1 passed
CI/workflow contracts:                129 passed
embedded frontend-required Bash:      22/22 passed bash -n
workflow integrity digests:           PASS
OpenAPI drift check:                   PASS
Admin_Webhooks non-PostgreSQL matrix: 302 passed, 24 deselected
Ruff, node/python syntax, diff checks: PASS
```

The first production-build attempt failed only because the restricted sandbox
denied Turbopack's internal local port bind; the identical command passed with
normal process permissions. Raw Bandit reports one existing low-confidence
`B101` at the unchanged strict-counter assertion; excluding that established
non-diff finding leaves no findings. The 24 PostgreSQL tests were not rerun for
this remediation because it changes no persistence, schema, or backend webhook
behavior; the prior required-provider proof remains applicable. Independent
review, fresh Qodo analysis, exact-head hosted CI, and the final review-thread
audit remain mandatory before merge.

## Follow-up Review and Hosted Ratchet Remediation

Independent review of source `c253e20467` found four additional issues. The
follow-up rejects raw URL authorities containing `@` or percent-encoded host
syntax before `URL` normalization; invalid create destinations now use a
dedicated field error instead of marking the URL invalid for unrelated command
errors; late canonical or legacy create responses cannot restore a signing
secret after `pagehide` or unmount cleanup; and a shared execution-order
reporter change now selects the admin UI unit gate that owns its behavioral
test.

The old hosted frontend shard also exposed a separate ratchet defect. Run
`33037959693`, job `98406051146`, completed the 255-file UI shard with 25 failed
and 1,933 passed assertions plus three unhandled errors. Its exact-base fast
replay found inherited failures but the full-context fallback rejected the
head report because `numTotalTestSuites` did not equal the suite hierarchy
reconstructable from serialized assertions.

A complete local reproduction produced the same 255 modules, 1,958 assertions,
and 556 raw Vitest suites. Only 555 suites were observable in JSON because
Vitest 4 counts module and suite tasks directly while its JSON assertion
records omit empty and skipped suite nodes. A real minimal fixture proved both
sides of the contract:

- one module, one visible suite, and one empty skipped suite reported three
  raw/runtime suites while JSON assertions exposed only two; strict validation
  accepted the clean reporter proof;
- adding an ordinary empty suite reported four suites and one module error;
  the safety reporter preserved the exact suite count and strict validation
  rejected the nonzero error rather than accepting an incomplete report.

The execution-order reporter is now schema version 2 and records the exact
runtime suite tree using module-relative child-index paths, names, terminal
states, and suite modes. Mode is required because Vitest reports both ordinary
skipped suites and todo suites with terminal state `skipped`, but counts only
todo suites as pending. Vitest also rewrites a normal suite containing only
`test.todo` cases to `mode=skip`, so assertion-derived pending parent statuses
remain unknown lower bounds rather than guessed categories. The runtime tree
must independently reconcile all passed, failed, and pending totals with the
raw JSON counters. Context validation also requires reporter `suiteCount` to
equal the raw Vitest total, requires module roots and complete parent paths,
and compares the exact runtime tree across head and base. Schema version 1
remains accepted only when JSON alone proves the complete suite total.

The admin safety reporter is also schema version 2. It independently records
the exact passed, failed, and pending suite totals using Vitest's JSON reporter
semantics. Strict validation requires those categories and their sum to match
the raw JSON counters, rejects a success flag paired with any failed test or
suite counter, and still rejects unhandled, module, or hook errors. The old
optimized-away Python assertion was replaced with an explicit fail-closed
counter guard.

A second independent review also found lifecycle races outside the reporter
contract. Signing-secret commands are now serialized before any request can
start, including non-retryable legacy creation; all webhook mutations and
legacy create fields remain disabled until the one-time secret is stored or
cleared; stale clipboard promises cannot mark a newer secret as copied; late
legacy request failures or successes cannot mutate post-cleanup state; and raw
fragment delimiters are rejected before URL normalization. Focused tests cover
same-tick canonical and legacy command starts, cross-row locking, stale
clipboard completion, page-unload cleanup, late legacy completion, and
fragment-bearing or empty-literal-authority destinations.

Final independent re-review found two remaining CI fail-closed gaps. Vitest's
JSON counters combine legitimate skip/todo work with tasks still pending at
report time, so the safety reporter now records separate incomplete suite and
test counters and strict validation requires both to be zero. Assertion JSON
with status `pending` and runtime suite manifests with state `pending` or
`queued` are rejected directly. The path classifier also routes the shared
ratchet helper to backend, package-frontend, and admin-UI gates; the deterministic
config to the package gate; the order reporter to both consuming frontend
gates; and the classifier plus its output emitter to their Python CI-test gate.
Table-driven tests freeze each routing contract.

Final local evidence for the follow-up source:

```text
focused webhook/reporter matrix:      77 passed across 6 files
admin UI package typecheck:           PASS
admin UI package lint:                PASS, 0 errors/41 inherited warnings
admin UI production build:            PASS, 49/49 routes
Chromium webhook lifecycle:           1 passed
CI/workflow/ratchet contracts:         165 passed
embedded frontend-required Bash:      22/22 passed bash -n
real hidden-suite reporter proof:      PASS
Python compile, Ruff, and Bandit:      PASS
Node reporter syntax:                 PASS
git diff --check:                      PASS
```

The unrestricted production build passed after the restricted sandbox denied
Turbopack's internal local port bind. The prior complete admin UI baseline run
reported 117 failures and 673 passes across unrelated surfaces; all 77 tests in
the changed webhook/reporter surface pass at this follow-up source. Those
repository-wide failures remain governed by the exact-base ratchet and were
not altered by this remediation.

Pinned SHA-256 values at this source are:

```text
vitest_base_ratchet.py:               84e744941a29724319f9783f4a02199646399d7c1eae51fb73f182338276839f
vitest_execution_order_reporter.mjs: af01d572f95d69faaa32261e50d1f2d4c8924d6106af806b913b33f558a3f3d1
vitest-safety-reporter.mjs:          cf7621d1658a1a3e1b1d16c392dc9af3fe6751072bdb5ef967169f06972ecf4d
```

Independent re-review, fresh Qodo analysis, exact-head hosted CI, and the final
review-thread audit remain mandatory before merge.

## Exact-Head Lifecycle and Docs-Gate Follow-Up

Qodo's review of pushed head `cc885b47c8` identified two valid follow-up
findings. The new parameterized path-classifier contract lacked an accepted
direct pytest marker. The legacy delivery-history controller also allowed an
older row's request to publish success, failure, or loading completion after
the administrator expanded a different row; the post-test refresh used the
same stale expanded-row closure.

TDD reproduced both controller failures before implementation. One deferred
history response completed after a row switch and published under the new row,
including clearing its loading state. A deferred test delivery completed after
a row switch and performed an obsolete history refresh for the original row.
The controller now assigns each legacy history request an identity token tied
to the current expanded row. Row switches, collapse, and unmount synchronously
invalidate ownership, and success, failure, and `finally` state changes all
recheck both row and token. Test delivery captures whether its row was expanded
at command start and refreshes only when that same row still owns expansion.
The classifier test now carries `@pytest.mark.unit` above parameterization.

Independent re-review found no Critical or Important findings in these changes.
It specifically confirmed stale success, failure, and `finally` suppression,
collapse and unmount invalidation, same-row refresh behavior, and direct marker
coverage.

The prior pushed head's `onboarding-docs-gate` then failed 1 of 190 tests in
[run 33047266119](https://github.com/rmusser01/tldw_server/actions/runs/33047266119).
The failing contract showed that exact base `origin/dev` at `2306c1939f3b`
already contained `Docs/ADR/040-synchronized-moodboards-and-studio-authority.md`
without its required tracked `Docs/Published/ADR` copy. All docs boundary,
command-boundary, and endpoint-drift steps passed. The canonical refresh also
exposed six unrelated content drifts. A direct local suite run without the
workflow's required refresh reproduced those as two content-identity failures
with 188 other tests passing. The exact hosted order, refresh followed by the
complete docs suite, passed all 190 tests. The six unrelated generated changes
were then restored and remain excluded from this PR follow-up. Only the missing
ADR-040 published artifact is retained, and the previously failing focused docs
contract passes locally.

Fresh local evidence for this follow-up:

```text
focused webhook/reporter matrix:      77 passed across 6 files
CI/workflow/ratchet contracts:        165 passed
admin UI package typecheck:           PASS
admin UI package lint:                PASS, 0 errors/41 inherited warnings
admin UI production build:            PASS, 49/49 routes
Chromium webhook lifecycle:           1 passed
focused docs published-file contract: 1 passed
hosted-order complete docs suite:      190 passed
focused Ruff and Python compilation:  PASS
```

The reporter and ratchet source files are unchanged, so their pinned SHA-256
values remain the values recorded immediately above. Fresh exact-head Qodo
analysis, hosted CI, and a final unresolved-thread audit remain required before
merge.

## Same-Row Legacy Test Ownership Follow-Up

Qodo's review of pushed head `43c56eef80` identified one additional valid
lifecycle race. A test delivery started while a legacy row was expanded could
complete after that same row was collapsed and re-expanded, or after a newer
test started for the same expansion, then reacquire delivery-history ownership
and perform an obsolete refresh.

TDD reproduced both paths before implementation. The collapse/re-expansion
case and the overlapping-test case each observed three history requests where
only the initial load and current refresh were permitted. The controller now
assigns an identity token only to tests started for the expanded row. A newer
test supersedes the prior token, while row switches, collapse, and unmount
synchronously invalidate it. Completion may refresh history only when the test
still owns that token and its row remains expanded. Tests started on unrelated
collapsed rows do not interfere with the active expansion.

Independent review then identified a third ordering: an older test could pass
its ownership check and start a history request before a newer same-row test
began. TDD reproduced the stale publication as `older.completed` appearing
after the newer test started. Test-triggered history requests now retain their
originating test token. A newer expanded-row test invalidates only an older
test-triggered history request and clears its loading state; ordinary manual
history requests remain independent. The shared owner-aware guard suppresses
stale success, error, and `finally` publication. Independent re-review found no
remaining Critical or Important issues.

Qodo also treated the recorded repository-wide admin UI baseline failures as a
new PR failure. That claim does not match this repository's required exact-base
ratchet policy: the failures are inherited, unchanged debt, the changed
webhook/reporter surface is green, and the ratchet contracts remain green. The
truthful baseline evidence is retained; no test or evidence was removed to
mask inherited failures.

Fresh local evidence at `2026-08-27T07:53:41Z`:

```text
same-row lifecycle TDD RED:            2 failed (3 calls observed, 2 expected)
in-flight refresh TDD RED:              1 failed (stale history published)
same-row ownership regressions GREEN:  3 passed
complete webhook page suite:           33 passed
focused webhook/reporter matrix:       80 passed across 6 files
CI/workflow/ratchet contracts:         165 passed
admin UI package typecheck:            PASS
admin UI package lint:                 PASS, 0 errors/41 inherited warnings
admin UI production build:             PASS, 49/49 routes
Chromium webhook lifecycle:            1 passed
independent re-review:                  PASS, no Critical/Important findings
focused Ruff and Python compilation:   PASS
reporter/ratchet SHA-256 pins:          PASS, unchanged
git diff --check:                       PASS
```

The restricted production-build attempt failed only because the sandbox denied
Turbopack's local helper port bind; the identical unrestricted command passed.
Playwright's generated `next-env.d.ts` development-path drift was restored and
is excluded. Commit/push, refreshed exact-head Qodo analysis, hosted CI, and a
final unresolved-thread audit remain required before merge.
