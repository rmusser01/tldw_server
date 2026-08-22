# Admin Webhooks PR 1 Verification

## Verification Identity

- Tested source commit: `a79fe91f2f`
- Rebased onto: `origin/dev` at
  `f7cc3d084affed81a7ae9e8fbbde9f5d96969fd1`
- Final verification timestamp: `2026-08-22T23:04:02Z`
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
| Complete PR 1 Python matrix | PASS: 466 passed, 0 skipped |
| PostgreSQL-required matrix | PASS: 19 passed, 0 skipped |
| Ruff | PASS |
| Bandit | PASS |
| Backend sensitive-log scans | PASS |
| Focused admin UI matrix | PASS: 77 passed |
| TypeScript typecheck | PASS |
| Changed-file ESLint | PASS |
| Production admin UI build | PASS |
| Chromium control-plane journey | PASS: 1 passed |
| UI persistence/console sink scan | PASS |
| Package-wide admin UI tests | KNOWN UPSTREAM BASELINE: 41 failed, 711 passed; no stable new failure |
| Package-wide admin UI lint | KNOWN UPSTREAM BASELINE: 3 errors, 41 warnings; no changed-file finding |

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

Result: `466 passed, 449 warnings in 115.15s`; zero skips.

The first complete run executed all 466 test bodies but exposed a teardown
error in an egress timing test. That test had patched `time.monotonic` on the
shared standard-library module, which also affected asyncio teardown. Its fake
clock now replaces only the `egress.time` module binding. The focused regression
passed, followed by the complete passing run above.

Required PostgreSQL matrix:

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. \
  ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_legacy_import_postgres.py
```

Result: `19 passed, 40 warnings in 79.57s`; zero skips. The required flag was
set, and the tests used the running disposable PostgreSQL 18.6 container rather
than SQLite or an availability skip.

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

## Upstream Admin UI Baselines

`bun run test` is not green on clean `origin/dev`. An isolated detached
`origin/dev` worktree using the same dependency tree and Node 20 produced:

```text
47 failed, 653 passed, 700 total
```

The tested branch produced:

```text
41 failed, 711 passed, 752 total
```

The six resolved baseline failures are all superseded legacy Webhooks-page
tests. Structured assertion-name comparison found no stable new branch failure.
One unchanged AI Ops timing test failed in one concurrent structured branch run
while still loading and passed `8/8` immediately in isolation; the final exact
package run returned the stable `41 failed, 711 passed` result.

`bun run lint` also reproduces the previously recorded package baseline:

```text
44 problems: 3 errors, 41 warnings
```

The three errors are pre-existing `@typescript-eslint/no-require-imports`
findings in `lib/__tests__/security-headers.test.ts` and `next.config.js`.
Targeted ESLint across every changed TypeScript/TSX file, including
`middleware.ts` and the Playwright helpers, passed with zero findings.

These baseline failures are not represented as passing gates. They are retained
as explicit upstream debt; the focused tests, typecheck, production build, and
browser journey establish no regression for this PR.

## Final Safety Checks

- `git diff --cached --check`: PASS for the tested source/docs commit.
- OpenAPI evaluation-webhook schema isolation: PASS.
- Canonical mode default remains `off`.
- Outbound HTTP, Jobs delivery workers, automatic event producers, test sends,
  delivery history, and activation readiness remain absent from PR 1.
- Two unrelated untracked watchlist template files were excluded from every
  commit and verification artifact.
