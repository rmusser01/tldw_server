# Canonical Admin Webhook Delivery Runbook

## Scope And Release State

Canonical admin webhook delivery defaults to off. The supervised runtime starts
only when both of these validated conditions hold:

- `TLDW_ADMIN_WEBHOOKS_MODE=on`
- `TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT=false`, which selects the canonical route

The lifecycle task is named `admin_webhook_delivery_runtime_task`. It owns the
canonical prepared worker, reconciler, and retention loops. It does not call or
alias the legacy `jobs_webhooks_task` service.

This PR 2 substrate is not a release activation. User and incident producers and
the operational admin UI remain disconnected until PR 3. Do not enable canonical
mode in deployment configuration until the PR 3 activation gate passes.

## Configuration

Provision the dedicated webhook key ring before migration or delivery:

- `TLDW_ADMIN_WEBHOOK_KEYS_JSON`: JSON object mapping stable key IDs to strict
  base64-encoded 32-byte keys.
- `TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID`: one key ID present in that object.

The active primary key ID must match the completed durable migration state.
Keep every key needed by retained ciphertext available during recovery.

Delivery settings are validated at process startup:

| Setting | Default | Accepted value |
| --- | ---: | --- |
| `TLDW_ADMIN_WEBHOOKS_MODE` | `off` | `off`, `migrate`, or `on` |
| `TLDW_ADMIN_WEBHOOK_DELIVERY_CLAIM_TTL_SECONDS` | 60 | 5-300 |
| `TLDW_ADMIN_WEBHOOK_DELIVERY_LOOP_INTERVAL_SECONDS` | 1 | 1-60 |
| `TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_INTERVAL_SECONDS` | 10 | 1-60 |
| `TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_FRESHNESS_SECONDS` | 30 | 1-60 and greater than heartbeat interval |
| `TLDW_ADMIN_WEBHOOK_REGISTRATION_LIMIT` | 100 | 1-1000 |
| `TLDW_ADMIN_WEBHOOK_ACTIVE_LIMIT` | 25 | 1-1000 and no greater than registration limit |
| `TLDW_ADMIN_WEBHOOKS_ALLOW_HTTP_DEV` | `false` | strict `true` or `false`; `true` is rejected in production |

The work loops run at the configured loop interval. A component writes durable
heartbeat evidence after its bounded pass or acquisition guard, so active loops
may publish more often than the configured heartbeat target. Freshness must stay
strictly greater than the heartbeat interval.

Jobs must expose database access and the exact canonical registration:

```text
domain: admin_webhooks
queue: delivery
job type: admin_webhook_delivery
```

If `JOBS_ALLOWED_JOB_TYPES` or
`JOBS_ALLOWED_JOB_TYPES_ADMIN_WEBHOOKS` is set, its combined allowlist must
contain `admin_webhook_delivery`.

## Readiness

Inspect `GET /api/v1/admin/webhooks/status` as a platform administrator. The
delivery object is sanitized: it contains no instance ID, URL, payload, secret,
Jobs ID, token, or exception text.

Foundational `acquisition_ready` requires all of the following:

1. canonical schema version 1 and the migration-095 delivery extension;
2. completed migration;
3. an available key ring whose primary matches durable state;
4. Jobs database access and the exact domain, queue, and job type;
5. at least one fresh ready reconciler heartbeat.

Worker readiness is deliberately not part of acquisition preflight. The worker
uses that preflight, then publishes its own heartbeat. This avoids startup
depending on a heartbeat that the worker has not yet had a chance to write.
Full `delivery_capability_ready`, required to activate a registration, adds at
least one fresh ready worker heartbeat. Retention readiness is reported but is
not an acquisition prerequisite.

For each component, any fresh ready instance wins. If none exists, status uses
the freshest bounded row: an old row reports `heartbeat_stale`, a current
unready row reports its closed reason, and no row reports
`worker_unavailable`, `reconciler_unavailable`, or `retention_unavailable`.
Heartbeat evidence is valid through an inclusive five-second future-skew bound.
A row farther in the future is never ready and reports `heartbeat_stale` when
it is the only evidence; a genuinely fresh ready row takes precedence over an
invalid future ready or unready row.

When AuthNZ delivery health is readable, status keeps its factual schema,
migration, key, backlog, and heartbeat fields even while degraded. The final
acquisition gate reports `mode_off` or `mode_migrate` for those modes and
`jobs_unavailable` for a Jobs dependency failure. `database_unavailable` is
reserved for a genuine delivery-health read failure; already loaded migration
and key facts remain visible.

## Metrics And Triage

All canonical families begin with `admin_webhooks_` and accept only closed,
low-cardinality labels. Metric failures are fail-open and never change a durable
mutation, attempt, reconciliation, or retention result.

Attempt and delivery counters are best-effort post-commit observations. The
owner of a synchronous test completion, worker terminal transition, expiry or
recovery transition, or lifecycle cancellation emits once after its owned
durable commit. Rollback, stale compare-and-swap, and idempotent replay emit
nothing. A process can still exit between commit and observation; without a
telemetry outbox these metrics do not claim crash-proof exactly-once delivery.
Registration gauges initialize and refresh from one bounded current-count
snapshot.

Start outage triage with:

- `admin_webhooks_heartbeat_ready` and
  `admin_webhooks_heartbeat_age_seconds` by `component`;
- `admin_webhooks_backlog` by nonterminal state;
- `admin_webhooks_oldest_nonterminal_age_seconds`;
- `admin_webhooks_enqueue_failures_total` by closed reason and backend;
- `admin_webhooks_key_errors_total` and
  `admin_webhooks_migration_errors_total`;
- `admin_webhooks_expiries_total` and
  `admin_webhooks_retention_deletions_total`.

Interpret backlog states as follows:

- `pending`: AuthNZ work has not been claimed for Jobs admission.
- `enqueue_claimed`: an expiring AuthNZ enqueue claim owns the handshake.
- `queued`: the exact Jobs row is attached.
- `processing`: one attempt currently owns the I/O boundary.
- `retry_wait`: AuthNZ has committed a retry outcome and Jobs owns scheduling.

A growing `pending` or `enqueue_claimed` backlog points first to reconciler or
Jobs readiness. Growing `queued` or `retry_wait` with a fresh reconciler points
to the worker, acquisition guard, leases, or receiver outcomes. A growing
oldest age is more actionable than a short-lived count spike.

## Delivery And Recovery

Jobs is the only automatic retry scheduler. The fixed network-attempt policy is
one initial attempt plus delays of 1 minute, 5 minutes, and 30 minutes, for a
hard maximum of four network attempts. Valid receiver `Retry-After` values on
429 or 503 are clamped to 1-1800 seconds and can only lengthen the fixed delay.

Automatic work expires after 72 hours. The reconciler terminalizes ordinary
due unattached work atomically. A claimed enqueue whose Jobs row may have been
created before AuthNZ attachment is excluded from blind expiry: enqueue
reconciliation performs the existing lookup-only identity recovery, persists
one exact cancel token, then applies and acknowledges Jobs cancellation. For
attached work expiry also stores one exact cancel token and uses the same
lookup/apply/ack boundary. Live processing rows, current attempts, and rows
already carrying a disposition coordinate are not overwritten. No AuthNZ
transaction mutates the Jobs database.

The runtime builds a complete Jobs generation privately and promotes it only
after database, queue, and job-type capability checks pass. Every supervised
worker start uses a fresh manager, SDK, worker ID, and handler; a stopped SDK is
never reused. Initial or transient construction failure leaves closed
unavailable queue/probe delegates, reports unready heartbeats, and retries at
the interruptible delivery-loop cadence. Reconciler access recovers when the
new generation is promoted, while retention continues independently.

Common terminal reasons include:

- `delivery_expired` and `attempt_budget_exhausted`;
- `canceled_disabled`, `canceled_deleted`, and
  `canceled_secret_rotation`;
- `superseded_config`;
- `test_attempt_interrupted` and `outcome_unknown`;
- `target_invalid`, `target_rejected`, and the closed `http_hop_*` SSRF,
  DNS, TLS, timeout, protocol, and bounded-response reasons;
- `http_client_error`, `http_redirect`, `http_rate_limited`,
  `http_server_error`, and `transport_error`.

Do not infer that an `outcome_unknown` attempt was not received. A worker can
lose its lease or process after the receiver accepts a request but before the
terminal AuthNZ commit. Delivery is at-least-once and unordered; recovery does
not claim exactly-once network I/O.

Disabling, deleting, rotating, or changing configuration cancels work that has
not crossed the final pre-I/O boundary. It cannot recall an HTTP request already
in flight. A real in-flight result remains recorded, with changed-configuration
evidence where applicable.

Manual redelivery creates a new delivery against the current active
configuration. When its delivery-config version differs from the source, the
operator must review the current ETag/version and send
`confirm_changed_configuration=true`. It never rewrites the original history.

The synchronous test path does not create a Jobs row or retry. It persists one
direct processing attempt, sends `X-TLDW-Webhook-Test: true`, and may test an
inactive registration. A process loss closes a stale test as
`test_attempt_interrupted`; an exact in-progress retry returns bounded `202`
state rather than sending a second request.

## Retention

Terminal delivery metadata remains through 29 days, 23 hours, and 59 minutes
after `terminal_at` and becomes eligible at 30 days. Creation time and expiry do
not start this clock. Nonterminal rows are never retained-purge candidates, and
a terminal delivery with an unacknowledged Jobs disposition remains protected.

Each transaction has one total 1-200 row budget and drains in this order:

1. eligible terminal deliveries;
2. newly orphaned events;
3. expired idempotency rows;
4. stale runtime heartbeat instances;
5. eligible registration tombstones.

Repeated partial batches deterministically continue until every finite eligible
category drains. A failed transaction leaves its rows intact and publishes an
unready retention heartbeat when the heartbeat store is reachable.

## Receiver Verification

The receiver must preserve the exact raw request body. Verify
`X-TLDW-Webhook-Signature` with constant-time comparison against:

```text
v1=hex(HMAC-SHA256(full_whsec_value, unix_timestamp + "." + raw_body_bytes))
```

Reject stale timestamps outside the receiver's chosen window; five minutes is
recommended. Deduplicate by event ID for business effects and retain delivery ID
for attempt diagnostics. Event and delivery IDs stay stable across automatic
retries, while timestamp and signature change on each attempt.

Published vector:

```text
secret: whsec_1111111111111111111111111111111111111111111111111111111111111111
timestamp: 1787443200
body: {"api_version":"2026-07-01","created_at":"2026-08-23T00:00:00Z","data":{"synthetic":true},"id":"00000000-0000-4000-8000-000000000001","type":"user.created"}
signature: v1=294bc280642cfd89fd011f606fbbe39633a77372db8ae9efd4281b2a3e509811
```

The delivery path follows no redirects, uses no ambient proxy, revalidates URL
policy and DNS at attempt time, verifies the connected peer, and uses the
central status-only HTTP hop. It does not buffer or retain receiver bodies or
ordinary response headers. Never add a raw HTTP client as an operational
workaround for a policy denial.

## Disable And Forward-Fix

To stop new canonical acquisition, set mode to `off` through the normal reviewed
configuration process and restart all nodes consistently. Expect final unready
heartbeats; retain AuthNZ and Jobs data and every required encryption key. Do
not delete pending dispositions or delivery rows manually.

Before the first canonical mutation or delivery, rollback may follow the
separate migration runbook while its durable rollback conditions remain true.
After any canonical create, update, rotation, redelivery, event capture, or
delivery attempt, do not restore the legacy writer. Keep canonical mode off,
diagnose from sanitized status/history and closed metrics, deploy a forward fix,
then restore `mode=on` only after schema, key, Jobs, reconciler, and worker
readiness are healthy. Schema drops, Jobs SQL repair, and legacy-service fallback
are not recovery procedures.
