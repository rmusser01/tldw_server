# Admin Webhooks PR 1 Verification

## Verification Identity

- Tested source commit: `ebbe6d30da`
- Rebased onto: `origin/dev` at
  `d736368d17c92f879d0b5364b45f23488629f5b8`
- Final verification timestamp: `2026-08-23T19:56:55Z`
- Host: macOS 26.5.2 (25F84), arm64
- Python: 3.11.13
- Node.js: 20.19.5 (the version family pinned by repository UI CI)
- Bun: 1.3.2
- Next.js: 16.2.2
- PostgreSQL: 18.6 (`postgres:18`, Debian 18.6-1.pgdg13+2)

The tested source commit includes the control plane, dual-backend persistence,
legacy importer, key rotation, route selector, admin UI, runbooks, and all gate
fixes described below. This evidence file is a documentation-only follow-up to
that immutable source tree.

## Result Summary

| Gate | Result |
| --- | --- |
| OpenAPI fingerprint and drift | PASS |
| Complete PR 1 Python matrix | PASS at direct-marker source: 483 passed, 0 skipped |
| PostgreSQL-required matrix | PASS at direct-marker source: 24 passed, 0 skipped |
| Direct pytest marker policy | PASS |
| CI shard coverage guard | PASS: 0 newly uncovered test files |
| Admin Webhooks non-PostgreSQL matrix | PASS: 301 passed |
| Chat persistence ordering regression | PASS: exact failing E2E plus 4-test surrounding set |
| Ruff | PASS |
| Focused Python typecheck | PASS |
| Bandit | PASS |
| Backend sensitive-log scans | PASS |
| Focused admin UI matrix | PASS: 77 passed |
| TypeScript typecheck | PASS |
| Changed-file ESLint | PASS |
| Production admin UI build | PASS |
| Chromium control-plane journey | PASS: 1 passed |
| UI persistence/console sink scan | PASS |
| Package-wide admin UI tests | KNOWN UPSTREAM BASELINE: final source 42 failed, 710 passed; base 47 failed, 653 passed |
| Package-wide admin UI lint | PASS: 0 errors, 41 unchanged warnings |
| Two-project real-backend Playwright | KNOWN UPSTREAM RUNNER FAILURE: Next 16 `.next/dev` lock collision before tests |

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
path_count:   2013
schema_count: 2948
sha256:       2b79239eeb8805e9801cc1cb03af9b952d6de06d2445b29c81bf76d68f650755
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

Final post-review, post-rebase result:
`483 passed, 459 warnings in 148.36s`; zero skips. This aggregate run executed
the real PostgreSQL-marked cases as well as the SQLite, API, authorization,
egress, system-ops, and workflow cases.

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

Result: `24 passed, 50 warnings in 95.74s`; zero skips. The required flag was
set, and the tests used the running disposable PostgreSQL 18.6 container rather
than SQLite or an availability skip. The post-rebase aggregate run then
executed the same 24 PostgreSQL cases with zero skips.

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

The Admin Webhooks directory is now assigned to the existing
`admin-watchlists-webhooks` shard in all five duplicated workflow matrices.
The repository guard passed:

```text
[shard-coverage] shards=774 test_files=4298 ignored=4 baseline=130 new_uncovered=0
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

## Upstream Admin UI Baselines

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

The required two-project real-backend command was also run exactly as CI
invokes it:

```bash
bun run test:real-backend -- \
  --project=chromium-real-jwt \
  --project=chromium-real-single-user \
  --reporter=line
```

It did not reach any browser test. Playwright starts multiple Next 16 dev
servers from the same `admin-ui` directory, and they collide on the shared
`.next/dev` lock with `Another next dev server is already running`. The same
multi-server structure exists at the exact PR base. The PR's Playwright change
only prevents unrelated mocked invocations from starting these real-backend
servers. No process remained after the failed attempt.

The package test and real-backend runner failures are not represented as
passing gates. They remain explicit upstream debt pending Linux CI and a
separate remediation decision; the focused tests, typecheck, production build,
and webhook browser journey establish the scoped PR behavior.

## Final Safety Checks

- `git diff --check`: PASS at tested source/test head `1263677011`.
- OpenAPI evaluation-webhook schema isolation: PASS.
- Canonical mode default remains `off`.
- Outbound HTTP, Jobs delivery workers, automatic event producers, test sends,
  delivery history, and activation readiness remain absent from PR 1.
- Two unrelated untracked watchlist template files were excluded from every
  commit and verification artifact.
