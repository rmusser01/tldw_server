# Admin Webhooks PR 3 Verification

## Verification Identity

- Branch: `codex/admin-webhooks-durable-producers-runtime`
- Verified pre-documentation head:
  `616ea4d5b6a1c0c98dd47a87a351ba5e47728d12`
- Branch merge base:
  `256ff515b3b3e3b6b624264ed67a12da2d74363b`
- Observed `origin/dev` during final verification:
  `2346700d0e64e2e7564853473dfea0b7f53928ab`
- Post-rebase pre-evidence-update head:
  `79ff79543a07c7758304d847a888fe47bafa1af0`
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

The final documentation and four corrected strict test doubles were verified as
an uncommitted working tree based on the head above. The Task 9 documentation
commit necessarily follows this evidence file and is not self-referential.

The observed `origin/dev` reference advanced independently during the initial
verification. At that identity snapshot, the branch was eight commits behind
and nine commits ahead. The completed branch was subsequently rebased without
conflicts onto that exact `origin/dev` commit and became ten commits ahead and
zero behind. No push or PR creation is part of this verification task.

## Result Summary

| Gate | Result |
| --- | --- |
| Complete backend aggregate | PASS: 1,169 passed, 0 skipped, 2,064 warnings |
| Post-rebase backend aggregate | PASS: 1,169 passed, 0 skipped, 2,064 warnings |
| Post-review backend aggregate | PASS: 1,178 passed, 0 skipped, 2,070 warnings |
| Final defensive-review backend matrix | PASS: 164 passed, 0 skipped |
| Required PostgreSQL producer/recovery matrix | PASS: 79 passed, 0 skipped, 160 warnings |
| Task 8 controlled receiver matrix | PASS: 6 passed, 0 skipped across all four backend combinations |
| Task 8 expanded producer/recovery matrix | PASS: 176 passed, 0 skipped |
| Task 8 security/support union | PASS: 506 passed |
| Final admin UI webhook/incident unit matrix | PASS: 94 passed |
| Post-review changed UI matrix | PASS: 128 passed |
| Final real-backend browser lifecycle | PASS: 1 passed, including guarded link and Back navigation |
| TypeScript typecheck | PASS |
| ESLint | PASS with 36 unrelated warnings and 0 errors |
| Next.js production build | PASS: 49 pages, including `/webhooks` and `/incidents` |
| Changed-path Ruff | PASS |
| Broad planned Ruff scope | REVIEWED BASELINE: 13 errors in branch-unmodified admin files |
| Bandit | REVIEWED: 11 Low, 0 Medium, 0 High |
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
