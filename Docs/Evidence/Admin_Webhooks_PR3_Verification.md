# Admin Webhooks PR 3 Verification

## Verification Identity

- Branch: `codex/admin-webhooks-durable-producers-runtime`
- Pull request: https://github.com/rmusser01/tldw_server/pull/2855
- Final integrated implementation head:
  `ecd801b6889f93def9e2a9e3b82a2e0ac63a8c8f`
- Final verified pre-evidence branch head:
  `ecd801b6889f93def9e2a9e3b82a2e0ac63a8c8f`
- Final branch merge base and observed `origin/dev`:
  `c85fb8db6b6efc338162276a52a193fc5d2d0ce5`
- Verification date: `2026-09-01`
- Host: macOS 26.5.2 build 25F84, arm64
- Project Python: 3.11.13
- Pytest: 8.4.1
- Bun: 1.3.2
- Node: v26.0.0
- Next.js: 16.2.2
- Playwright: 1.58.2
- Aggregate pytest seed: `1831171713`
- Required PostgreSQL pytest seed: `2138377250`

The evidence-only update that records the final integrated gates necessarily
follows the implementation head above and is not self-referential. Earlier
verification and rebase identities remain in their chronological sections.
The complete nineteen-commit pre-evidence PR branch rebased without conflicts
onto the final observed `origin/dev` and became nineteen commits ahead and zero
behind. No production activation occurred.

The verified branch was published as pull request
https://github.com/rmusser01/tldw_server/pull/2855 for normal CI and review.

## Result Summary

| Gate | Result |
| --- | --- |
| Complete backend aggregate | PASS: 1,169 passed, 0 skipped, 2,064 warnings |
| Post-rebase backend aggregate | PASS: 1,169 passed, 0 skipped, 2,064 warnings |
| Post-review backend aggregate | PASS: 1,178 passed, 0 skipped, 2,070 warnings |
| Final integrated backend aggregate | PASS: 1,201 passed, 0 skipped, 2,110 warnings |
| Post-publication base-advance aggregate | PASS: 1,201 passed, 0 skipped, 2,110 warnings |
| Final Qodo-remediation backend aggregate | PASS: 1,206 passed, 0 skipped, 2,116 warnings |
| Final Qodo-remediation focused backend | PASS: 113 passed, 0 skipped, 6 warnings |
| Final defensive-review backend matrix | PASS: 164 passed, 0 skipped |
| Required PostgreSQL producer/recovery matrix | PASS: 79 passed, 0 skipped, 160 warnings |
| Task 8 controlled receiver matrix | PASS: 6 passed, 0 skipped across all four backend combinations |
| Task 8 expanded producer/recovery matrix | PASS: 176 passed, 0 skipped |
| Task 8 security/support union | PASS: 506 passed |
| Final admin UI webhook/incident unit matrix | PASS: 94 passed |
| Post-review changed UI matrix | PASS: 128 passed |
| Final Qodo-remediation UI matrix | PASS: 121 passed across 7 files |
| Final real-backend browser lifecycle | PASS: 1 passed, including guarded link and Back navigation |
| Admin UI TypeScript typecheck | PASS |
| OpenAPI drift correction | PASS: 2,067 paths, 3,122 schemas, SHA `72a49730dfab...` |
| Generated frontend OpenAPI declaration | PASS: generated and focused `tsc` check exited 0 |
| ESLint | PASS with 36 unrelated warnings and 0 errors |
| Next.js production build | PASS: 49 pages, including `/webhooks` and `/incidents` |
| Changed-path Ruff | PASS |
| Broad planned Ruff scope | REVIEWED BASELINE: 13 errors in branch-unmodified admin files |
| Bandit | REVIEWED: 11 Low, 0 Medium, 0 High |
| Final Qodo-remediation Bandit scope | PASS: 0 findings across 4,978 production LOC |
| CI shard coverage guard | PASS: 4,518 test files, 0 newly uncovered |
| Markdown local links and diff whitespace | PASS |

## Test-Contract Correction

The first aggregate run deterministically exposed strict test doubles that had
not been updated when Task 8 added the test-only `allow_e2e_loopback` policy
argument. Production code was unchanged. RED evidence was:

- `22 failed, 59 passed` across control-plane, delivery composition, and runtime
  recovery tests;
- the validator and executor doubles rejected the new keyword;
- the runtime recovery `SimpleNamespace` lacked the new setting, which the
  fail-closed runtime correctly caught as an unavailable generation.

The minimal correction updated those signatures/settings and the matching
PostgreSQL validator double. The deterministic seed rerun passed:

```text
81 passed, 4 warnings in 19.04s
```

The exact focused command was:

```bash
RUN_JOBS=1 PYTHONPATH=.:packages/tldw_profile_core/src \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  -m pytest -q --tb=short --randomly-seed=1831171713 \
  tldw_Server_API/tests/Admin_Webhooks/test_retention_health_runtime.py \
  tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_mode_guard.py
```

## Independent Review Remediation

A first independent review found six Important integration defects and one
documentation mismatch after Task 9. All were reproduced against the branch
before implementation changes:

1. the admin UI accepted only UUID command IDs while the backend returns
   `sha256:<64 lowercase hex>`;
2. incident notification used the ordinary admin router, which bypassed the
   canonical bounded error envelope and accepted unknown request fields;
3. same-key incident replay rebuilt expected data from mutable current incident
   state;
4. a notify/reconciler race could retain a conflicting duplicate marker because
   request IDs changed between attempts;
5. the UI preview was not bound to the incident version accepted by the server;
6. user deactivation committed before the required durable account audit;
7. receiver documentation incorrectly claimed narrative whitespace was not
   trimmed.

The remediation uses a keyed canonical request fingerprint as the stable
incident command source identity, validates replay from the stored encrypted
canonical event, requires `expected_resource_version`, rejects stale new
commands, and preserves same-command replay after later incident mutation. The
incident route now runs under the canonical webhook route class with closed
request validation and redacted 404/409/412/422/503 envelopes. User
deactivation inserts its required AuthNZ audit row in the same transaction as
the source mutation, event, and automatic deliveries; a forced audit failure
test proves complete rollback.

The first enforced PostgreSQL run exposed a concrete backend-parity defect in
that audit change: PostgreSQL defines `audit_logs.resource_id` as `INTEGER`, but
the new insert passed a string that SQLite accepted. The focused test failed,
the shared insert was corrected to pass the integer user ID, the focused
PostgreSQL test passed, and the complete required PostgreSQL matrix then passed
79/79 with zero skips.

New regression coverage proves mutable-state replay, stale-preview rejection,
notify/reconciler convergence, canonical API envelopes and redaction, exact
OpenAPI requirements, SQLite/PostgreSQL durable audit rows, audit-failure
rollback, the backend command-ID contract, and preview version submission. The
real-backend browser lifecycle now also selects `incident.notify`, submits the
reviewed UI preview, verifies the signed privacy-bounded receiver payload, and
finds the persisted delivery in admin history.

## Final Defensive Review Remediation

A subsequent independent full-diff review found five additional defects, and a
provider-contract self-review found two more. All were reproduced before their
production fixes:

1. ordinary system-ops writes could replace malformed durable state with
   defaults;
2. retained true marker conflicts left reconciler health falsely ready;
3. a self-consistent rotate response could identify the wrong registration,
   and registration event types were not catalog-bound;
4. in-document navigation could discard a one-time secret or ambiguous command;
5. the lifetime 1,000-command stakeholder-email cap had no terminal retention;
6. a provider result of `false` was recorded as sent;
7. provider initialization happened after the recipient crossed the durable
   sending boundary.

Every system-ops mutation now uses the bounded strict parser. The reconciler
continues past a conflicting marker so later work can converge, then raises the
bounded conflict so the runtime records a degraded heartbeat. Client ETag,
registration identity, closed shape, and the immutable six-event catalog are
validated together. The shared memory-only navigation guard covers unload,
links, History/Navigation APIs, browser Back, and global keyboard shortcuts
before `router.push`. Stakeholder email initializes its provider before a
claim, treats only literal `true` as sent, and admits new work at capacity by
pruning only fully terminal commands older than the 30-day replay window.
Pending and `sending`/unknown outcomes are never automatically pruned or resent.

The verification review then found two remaining client gaps: direct keyboard
shortcut navigation started before the History wrapper, and ordinary
GET/PATCH/list/catalog responses still relied on TypeScript casts. Red tests
proved both. Shortcut navigation now performs synchronous shared admission
before calling Next, while security/session redirects remain intentionally
unblocked. GET, PATCH, list, catalog, create, and one-time-secret responses now
use closed runtime validators; create non-replay and PATCH results are also
bound to the submitted command.

The final verification pass also corrected two bounded validator edge cases:
accepted trailing-dot DNS targets are compared using the same normalized origin
the server returns, and catalog registration limits are capped at the server
schema maximum of 1,000.

Pre-integration verification passed `164/164` focused backend tests and
`128/128` focused UI tests. Required PostgreSQL parity passed `79/79` with zero
skips in 468.40 seconds. Typecheck, changed-path Ruff, Python compilation,
diff whitespace, full lint with only 36 unrelated baseline warnings, and the
49-page production build passed. Mocked Chromium passed `1/1`. The amended
real-backend lifecycle passed `1/1` in 31.5 seconds after proving that a
cancelable in-app navigation and browser Back both left the URL and one-time
secret dialog intact, then completing the full signed receiver lifecycle.
The final independent read-only verification found no remaining Critical, P1,
or P2 correctness/security issue.

## Backend Gates

### Complete aggregate

The authoritative aggregate disables Docker auto-provisioning because required
PostgreSQL behavior has its own zero-skip gate below. An already available
local disposable fixture may still satisfy optional PostgreSQL cases.

```bash
TLDW_TEST_NO_DOCKER=1 RUN_JOBS=1 \
  PYTHONPATH=.:packages/tldw_profile_core/src \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  -m pytest -q --tb=short --randomly-seed=1831171713 \
  tldw_Server_API/tests/Admin_Webhooks \
  tldw_Server_API/tests/Admin/test_incidents_service.py \
  tldw_Server_API/tests/Admin/test_admin_ops_new_endpoints.py \
  tldw_Server_API/tests/Admin/test_admin_account_audit_events.py \
  tldw_Server_API/tests/Services/test_startup_optional_workers.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Security/test_http_hop_transport.py \
  tldw_Server_API/tests/Security/test_http_hop_streaming.py
```

Result: exit 0, `1,169 passed, 2,064 warnings in 1153.44s (0:19:13)`,
zero skips. An already available disposable PostgreSQL fixture satisfied every
optional PostgreSQL case despite the no-auto-provision guard.

### Post-rebase aggregate

The branch was rebased without conflicts onto
`origin/dev` `2346700d0e64e2e7564853473dfea0b7f53928ab`. The upstream-only changes
were limited to shared pytest isolation and CI-concurrency tests, so the same
complete backend aggregate was rerun with the same seed after integration.

Result: exit 0, `1,169 passed, 2,064 warnings in 1157.68s (0:19:17)`,
zero skips. After the rebase the branch was ten commits ahead and zero behind
the observed `origin/dev`.

### Post-review aggregate

After the independent-review remediation, the same aggregate was rerun with
host loopback and disposable Docker/PostgreSQL access so no backend or receiver
case was skipped.

Result: exit 0, `1,178 passed, 2,070 warnings in 1135.97s (0:18:55)`,
zero skips. A prior sandboxed attempt reached `1,017 passed, 159 skipped` and
failed only the two tests whose explicit loopback binds were denied; both passed
immediately with loopback permission before the complete authoritative rerun.

### Final integration aggregate

The complete eleven-commit branch rebased without conflicts onto the final
observed `origin/dev` commit
`e3c198224bb63a995190863e9dcb9adbd95204b2`. The integrated implementation
head became `1298ee5d0d9208e5b457c783d82e8f0110a11498`, eleven commits ahead and zero
behind. The same deterministic aggregate command and seed recorded under
Complete aggregate were then rerun against that exact integrated tree.

Result: exit 0, `1,201 passed, 2,110 warnings in 1131.73s (0:18:51)`, zero
skips. The available disposable PostgreSQL fixture executed the PostgreSQL
cases rather than skipping them. No production activation occurred.

### Post-publication base advance

After PR #2855 opened, `origin/dev` advanced to
`21c1acc5bbac2df7d53ce5b759b0c79ab3a260ba` with three shared database commits
limited to `schema_once` verification and its Collections/Watchlists callers.
All fourteen PR commits rebased without conflicts. The rebased implementation
commit is `304a04a9da7dfd04d6f1d8e32878f4e22583ac4d`; the verified pre-evidence branch
head is `0525aff3b85fdb285bf6431100d75a727c05ac91`, fourteen commits ahead and zero
behind.

The required PostgreSQL matrix first passed `79/79`, zero skips, 160 warnings
in 475.72 seconds. The same complete aggregate and seed were then run with host
loopback permission against the live disposable fixture.

Result: exit 0, `1,201 passed, 2,110 warnings in 1129.96s (0:18:49)`, zero
skips. No production activation occurred.

Before the force-push, `origin/dev` advanced again to
`56def76c50acb61152c11bfba70c3f09388db375`. That delta changed only two
Embeddings test-fixture files. All fifteen PR commits rebased without conflicts;
the final implementation commit became
`15ad4e76c05b09581d8133f8eb91ef8c9f466abf` and the pre-evidence branch head
became `23f779bbf1c48fad98d810eedb4632c9a1710a19`, fifteen commits ahead and zero
behind. Because no production or aggregate test path changed, the 1,201-test
gate remained applicable. The four upstream `schema_once` caller regressions
and one Embeddings import regression passed `5/5` in 0.74 seconds on the final
tree, and the shard guard again reported zero newly uncovered files.

### Required PostgreSQL matrix

```bash
RUN_JOBS=1 ADMIN_WEBHOOKS_TEST_POSTGRES=1 \
  PYTHONPATH=.:packages/tldw_profile_core/src \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_user_producers_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_incident_marker_reconciler_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py
```

Result: exit 0, `79 passed, 160 warnings in 472.60s (0:07:52)`, zero
skips.

The fixture class was the repository's disposable local Docker PostgreSQL
fixture, default image `postgres:18` and container identity
`tldw_postgres_test`. The observed PostgreSQL server was 18.6
(`Debian 18.6-1.pgdg13+2`). No DSN, username, password, port, or private database
name is recorded.

After the post-publication base advance, the enforced matrix passed again:
`79 passed, 160 warnings in 475.72s (0:07:55)`, zero skips.

## Receiver And Browser Proof

The Task 8 controlled receiver matrix passed all six production events, exact
raw-body HMAC verification, duplicate network delivery, one retryable 503 then
success, terminal 400, the test header, and historical-event/new-delivery
manual redelivery. It ran across SQLite/PostgreSQL AuthNZ and every independent
AuthNZ/Jobs backend combination with zero skips.

The final browser rerun used explicit linked-worktree interpreter and package
paths:

```bash
cd admin-ui
PYTHONPATH=.:packages/tldw_profile_core/src \
  TLDW_ADMIN_E2E_PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  bun run test:real-backend:webhooks
```

Result: exit 0. The build generated 49 pages and Chromium passed `1/1` in
24.3 seconds. The flow created an inactive registration, acknowledged the
one-time secret, enabled it, produced user and incident events, previewed and
submitted a version-bound `incident.notify` command, verified its signed public
payload, inspected persisted history, tested, manually redelivered, disabled,
rotated, and proved secret/URL redaction. Test-only receiver addresses and key
material are omitted.

No production receiver was called and no production registration was enabled.

After the final rebase and defensive-review remediation, the same real-backend
gate passed again `1/1` in 32.9 seconds. That run additionally proved that
cancelable in-app navigation and a real browser Back action retain the URL and
one-time secret dialog before completing the signed receiver lifecycle.

## Admin UI Gates

```bash
cd admin-ui
bunx vitest run \
  app/webhooks/__tests__/page.test.tsx \
  app/incidents/__tests__/page.test.tsx \
  lib/api-client-webhooks.test.ts \
  lib/http.test.ts \
  lib/incident-workflow.test.ts
bun run typecheck
bun run lint
bun run build
```

Results:

- focused Vitest: 5 files, 94 tests passed;
- typecheck: exit 0;
- lint: exit 0, 36 existing warnings and no errors;
- production build: exit 0, 49 pages.

After review remediation, the directly changed UI matrix passed 62 tests across
the incident page, strict webhook API client, and HTTP helper. Typecheck, lint,
the 49-page production build, the focused Chromium incident-notify flow, and the
real-backend lifecycle all passed again.

The required broad `bun run test` baseline remains red: 131 files passed, 17
unrelated files failed, 778 tests passed, 41 unrelated tests failed, and three
unrelated snapshots failed. None of the failing files is in the PR 3 admin UI
diff. The focused webhook/incident matrix, typecheck, lint, build, and real
backend lifecycle are green. This baseline is not represented as a passing
whole-repository test gate.

## Qodo Review Remediation

Qodo reported eleven actionable findings after publication: nine Medium and
two High. Each finding was validated against the branch before implementation.
The final remediation commit is
`4a65f68c659ca0dd8260523d30448693d2b2d385` after the conflict-free rebase
onto `origin/dev` `8140c679f3ea0334cea2dc1be32feb5b80e22ebe`.

The two High findings were resolved as follows:

- Permanently malformed, catalog-invalid, duplicate, or cryptographically
  corrupt incident markers are now atomically moved from
  `webhook_pending_events` to `webhook_quarantined_events`. Later valid markers
  reconcile in the same pass, and the pass then reports
  `VALIDATION_FAILED` so runtime health degrades visibly. Missing and unknown
  keys remain retryable and are not quarantined.
- The shared sensitive-navigation guard now captures the protected URL and
  state when installed. A blocked browser Back operation restores that exact
  snapshot instead of preserving the destination URL.

The nine Medium findings were resolved as follows:

- incident notify now uses the canonical admin-webhook rate limiter;
- the service owns completion of the mandatory audit sink before returning;
- blocking incident store reads and writes run through `asyncio.to_thread`;
- activation-check exceptions retain phase and stack context in operator logs
  while CLI output remains closed and sanitized;
- deferred marker-capture logs include operation, event ID, request ID, error
  type, and exception context without payload or secret material;
- incident absence uses typed `WebhookError(NOT_FOUND)` rather than exception
  string inspection;
- the two event validation helpers now document their closed contracts; and
- legacy async incident tests now have explicit unit markers, argument types,
  and return annotations.

Regression tests were written before the implementation changes and failed for
the intended missing rate-limit, audit-ordering, event-loop lock, typed-error,
structured-log, poison-marker, and Back-restoration behavior. Final focused
backend verification passed 113 tests with six warnings. The focused admin UI
matrix passed 121 tests across seven files; typecheck passed, full lint returned
zero errors and the established 36 unrelated warnings, and the production build
generated all 49 pages.

Because `origin/dev` advanced with Reading List snapshot and shared Collections
schema-bootstrap changes, all seventeen PR commits were rebased and the
database-sensitive gates were repeated. The required PostgreSQL producer and
recovery matrix passed 79/79 with zero skips and 160 warnings in 483.09 seconds.
The deterministic impacted aggregate with seed `1831171713` passed 1,206/1,206
with zero skips and 2,116 warnings in 1,167.09 seconds. The final real-backend
browser lifecycle rebuilt the UI and passed 1/1 in 31.6 seconds.

Changed-path Ruff, Python compilation, the CI shard guard, both committed and
worktree diff checks, and focused Bandit passed. Bandit reported zero findings
across 4,978 lines of the five Qodo-remediated production Python files. The
previous complete-branch Bandit classification remains 11 reviewed Low, zero
Medium, and zero High because it covers older branch modules outside this
focused remediation scope.

The broad admin UI baseline remains unrelated and red: the latest diagnostic
run passed 133 files and 807 tests while 17 files and 41 tests failed in the
previously identified navigation metadata, BYOK, plans, resource governor,
snapshot, and other non-webhook areas. It is not represented as a passing gate.
One mocked Chromium webhook test also timed out before navigation because its
stale event-catalog fixture rendered no event checkboxes; the canonical
real-backend lifecycle above is the authoritative browser proof.

No production activation, migration, receiver enrollment, or traffic change
was performed as part of review remediation.

## CI OpenAPI Artifact Correction

The first post-Qodo `backend-required` CI run failed only the committed OpenAPI
drift gate:
https://github.com/rmusser01/tldw_server/actions/runs/33578798713/job/100088605503.
The checked-in fingerprint described 2,066 paths and 3,120 schemas with SHA
`1a9cad1bded6eac428093e0c222fa941bf645dada6f51ebf890df28183f94cfa`;
the branch contract contained 2,067 paths and 3,122 schemas with SHA
`72a49730dfab56021a5892747fbb2d1ce4f319a4730bb1ae44d24814e27f4e31`.

Canonical export inspection confirmed that the complete delta is the intended
`POST /api/v1/admin/incidents/{incident_id}/notify-webhooks` operation plus
`IncidentWebhookNotifyRequest` and `IncidentWebhookNotifyResponse`. The route
requires the bounded `Idempotency-Key`, closed request body, and expected 202
response schema. The refreshed fingerprint is committed at
`ecd801b6889f93def9e2a9e3b82a2e0ac63a8c8f` after the final rebase.

The canonical schema, fingerprint, and ignored frontend declaration were
regenerated with the repository exporter and `openapi-typescript`. Verification
then completed as follows:

```bash
PYTHONPATH=.:packages/tldw_profile_core/src \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  Helper_Scripts/export_openapi_schema.py \
  --out apps/tldw-frontend/lib/api/generated/openapi.json \
  --fingerprint apps/tldw-frontend/lib/api/openapi.fingerprint.json

cd apps/tldw-frontend
bun x openapi-typescript \
  lib/api/generated/openapi.json \
  -o lib/api/generated/schema.d.ts
```

- OpenAPI drift check: exit 0, fingerprint matches the checked-in snapshot.
- Focused OpenAPI tests: 12 passed, 6 warnings in 13.23 seconds.
- Generated declaration check: focused `tsc --noEmit --skipLibCheck` exited 0.
- Tracked delta: only the three fingerprint values changed before this evidence
  update; the full OpenAPI JSON and generated declaration remain ignored build
  artifacts.

The whole upstream frontend `bun run typecheck` is not represented as a green
gate for this correction. This isolated worktree has no installed workspace
dependencies; temporarily exposing the primary checkout's app dependencies
still left shared `packages/ui` dependencies unresolved and also surfaced
unrelated existing script-test type errors. The generated declaration was
therefore checked directly with the installed TypeScript compiler. The already
recorded admin UI typecheck remains green. No production activation occurred.

After the artifact correction was first pushed, `origin/dev` advanced to
`c85fb8db6b6efc338162276a52a193fc5d2d0ce5` through Personal Context
documentation and exact shared-core/Chatbook byte-parity PRs. The only source
change was import ordering and a blank line in the shared profile model; the
upstream PR explicitly preserved schemas and runtime behavior. All twenty PR
commits rebased without conflicts. On the rebased tree, the OpenAPI drift check
still matched and the combined webhook OpenAPI, server Personal Context, and
shared-core public contract matrix passed 19/19 with six warnings in 14.22
seconds. The verified pre-evidence implementation head remained nineteen
commits ahead and zero behind. No production activation occurred.

## CI Path Classification And Shard Contract Correction

After the fingerprint correction, workflow run
https://github.com/rmusser01/tldw_server/actions/runs/33589092145 passed every
required gate except upstream frontend unit shards 1 and 5. Retrying only the
failed jobs produced failures in the same shards but attributed regressions to
different unchanged tests. Exact-base replay also retained 36 and 16 baseline
failures respectively, while full-context base comparison changed those
counts. This was evidence of the shared Vitest ratchet's existing nondeterminism,
not a branch runtime-frontend regression. A local serial rerun of the three
first-attempt files passed all 85 tests.

The path classifier treated the generated
`apps/tldw-frontend/lib/api/openapi.fingerprint.json` metadata as executable
upstream frontend code and therefore selected all runtime unit shards. The
classifier now excludes that exact metadata path from upstream runtime and E2E
selection while preserving admin UI selection and all backend/API gates.
Regression tests first failed on the old behavior, then passed 14/14 after the
change, including fingerprint-only and fingerprint-plus-admin-UI cases.

The expanded CI contract run then exposed two branch-owned backend shard
omissions. All five repeated matrices still referenced the deleted
`test_admin_ops_webhooks_reports.py`, and six new activation/backup tests in
`test_admin_e2e_support_api.py` had no node-id assignment. The stale path was
removed and the six tests were assigned to `admin-e2e-reset-backups` in every
matrix. The final classifier, required-gate detector, frontend workflow,
backend workflow, and license-first contract matrix passed 79/79:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  -m pytest -q \
  tldw_Server_API/tests/CI/test_path_classifier.py \
  tldw_Server_API/tests/CI/test_detect_required_gate_changes_action.py \
  tldw_Server_API/tests/CI/test_frontend_required_workflow.py \
  tldw_Server_API/tests/CI/test_required_workflow_contracts.py \
  tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py
```

Before publication, `origin/dev` advanced by two commits to
`5fdd610df6e458957a34d07c5c5ac2a4b5d28d9f`; the only upstream file change
removed an obsolete Embeddings test fixture. All 21 branch commits rebased
without conflicts. The same 79-test CI contract matrix, Ruff lint/format
checks, and branch diff whitespace check passed on the rebased tree. The branch
was 21 commits ahead and zero behind before this evidence amendment.

No production activation occurred.

## Static And Security Gates

The plan's broad Ruff command returned 13 errors exclusively in admin endpoint
files unchanged from the branch merge base. The changed production/test path
rerun returned `All checks passed!`. The errors and their unrelated ownership
are recorded rather than silently fixed in this webhook task.

Bandit scanned the canonical package and shared producer/runtime services. It
returned raw exit 1 with 11 Low, 0 Medium, and 0 High findings:

- ten `B110`/`B112` findings are documented fail-open metrics/status paths whose
  exceptions cannot alter durable webhook truth;
- one `B105` finding is the enum reason string
  `canceled_secret_rotation`, not a password.

No finding exposes secret material or permits failed-closed delivery work to
continue. These reviewed findings are not described as a raw Bandit exit-0
pass.

Local Markdown link extraction found eight relative links in the changed
webhook guides, and every target exists. `git diff --check` is the final
whitespace gate.

## Invalid Attempts

Invalid attempts are retained here so they are not mistaken for product
failures or omitted from the historical record:

1. The first aggregate found the stale Task 8 test doubles described above.
2. An unguarded aggregate was interrupted at 84% after the non-PostgreSQL gate
   repeatedly waited for Docker provisioning before skipping PostgreSQL-only
   tests. No assertion had failed at interruption.
3. The first guarded aggregate ran inside the filesystem sandbox. It reported
   `1008 passed, 159 skipped` plus two `PermissionError` failures when transport
   tests attempted to bind an ephemeral loopback socket. The authoritative
   rerun used host-loopback permission.
4. The first production build attempt failed when sandboxed Turbopack was
   denied its internal loopback bind. The identical host-permitted build passed.
5. Two browser setup attempts reached no test assertion. The first selected
   system Python 3.9 and failed on `dataclass(slots=True)`; the second selected
   Python 3.11 but lacked the linked package `PYTHONPATH`. The command recorded
   above fixes both worktree prerequisites and passed.
6. The first post-review PostgreSQL command ran inside the filesystem sandbox
   and skipped all 79 tests because it could not access the Docker socket. The
   required host-permitted rerun enforced `TLDW_TEST_POSTGRES_REQUIRED=1`.
7. That first enforced PostgreSQL run passed 78 tests and failed the new
   deactivation audit test because a string was passed to the integer
   `resource_id` column. The integer-bound fix passed the focused regression and
   the complete 79-test matrix.
8. The first post-review aggregate ran without loopback permission. It had no
   product assertion failure, but two transport tests received `PermissionError`
   from `asyncio.start_server`. Both passed with host permission, followed by
   the authoritative 1,178-test zero-skip aggregate above.
9. A final mocked Chromium attempt ran inside the filesystem sandbox and
   reached no assertion because the local Next server could not bind
   `127.0.0.1:3001`. The host-permitted rerun passed `1/1`.
10. The first amended real-browser navigation assertion attempted a physical
    click on the sidebar while the signing-secret modal correctly owned the
    pointer overlay. Playwright timed out before dispatching a click, so this
    was a harness error rather than a product failure. The test now dispatches
    the same cancelable anchor event directly, retains a real `history.back()`
    assertion, and passed the complete lifecycle.
11. PR #2855's first `Shard coverage guard` run correctly found that the newly
    extracted `test_admin_ops_reports.py` was in no CI shard. The test was added
    beside the related admin reports coverage in all five duplicated backend
    matrices. The local guard then reported 4,517 test files and zero newly
    uncovered files, and the test itself passed 29/29.
12. The first aggregate after `origin/dev` advanced intentionally retained the
    no-auto-provision guard but ran before recreating the disposable PostgreSQL
    fixture and without host loopback permission. It reported `1,040 passed,
    159 skipped` plus the same two loopback-bind `PermissionError` failures as
    earlier sandboxed attempts. The enforced PostgreSQL matrix then passed
    79/79 and recreated the fixture; the host-permitted aggregate passed all
    1,201 tests with zero skips as recorded above.
13. The first local fingerprint command omitted the linked package source from
    `PYTHONPATH` and failed during import with
    `ModuleNotFoundError: tldw_profile_core`; no artifact was written by that
    attempt.
14. The first direct schema-export attempt was launched from the frontend
    directory with a root-relative script path, and the next attempt omitted
    creation of the ignored output directory. Both failed before writing the
    schema. The corrected root command above completed successfully.
15. The first `bun x openapi-typescript` attempt was denied access to Bun's
    sandbox-external temporary directory. The approved host-permitted rerun
    generated the declaration and exited 0.
16. The first whole-frontend typecheck found no `cross-env` because the isolated
    worktree has no dependency install. A temporary app dependency link allowed
    the command to start, but shared workspace packages remained unresolved and
    unrelated baseline script-test errors were reported. The link was removed;
    the focused generated-declaration check above exited 0.

## Documentation And Release Decision

`Docs/Admin_Webhooks_Receiver_Guide.md` is the public contract for:

- the exact six subscription schemas and reserved `webhook.test` schema;
- raw-body HMAC-SHA256, five-minute recommended timestamp tolerance, and
  constant-time comparison;
- at-least-once unordered delivery, event-ID business deduplication,
  delivery-ID diagnostics, retry timing, test headers, and manual redelivery;
- current secret-version overlap and receiver logging boundaries.

The control-plane, migration, key-rotation, and delivery runbooks now cover the
canonical-only rollout, coordinated pending-marker backup/restore/readback,
key-loss behavior, dead-delivery inspection, safe disable, and the permanent
forward-fix boundary after first canonical activity. Release notes state that
the implementation remains default-off.

This evidence does not authorize production activation. Private-beta
activation remains a separate reviewed operator change: finish migration and
readback in `migrate`, rotate imported secrets, provision the key ring, pass
predeploy, deploy one no-traffic `on` canary, pass live, perform controlled
automatic/test/redelivery probes, and expand gradually. After first
`event_capture`, rollback is mode `off` plus forward-fix, never the legacy
writer.

## Residual Risks And Follow-Up

- The branch is integrated with the observed `origin/dev`, and the complete
  impacted backend aggregate passed after the conflict-free rebase. Recheck
  divergence and repeat affected gates if `origin/dev` advances before merge.
- The whole admin UI test baseline and broad admin-endpoint Ruff baseline remain
  red outside this branch's changed files. They require separately owned cleanup.
- Independent review is complete with no unresolved Critical, P1, or P2
  finding. Normal PR review and CI remain required before merge.
- Pending incident-marker backup/readback currently uses reviewed maintenance
  access to private strict readers rather than a public one-command CLI. Keep
  the procedure operator-reviewed and automate it before public rollout.
- Disposable PostgreSQL and controlled receiver tests prove implementation
  behavior, not provider backup/restore or production capacity.
- The private-beta activation sequence has not been executed against the live
  deployment and remains intentionally blocked on a separate change record.
