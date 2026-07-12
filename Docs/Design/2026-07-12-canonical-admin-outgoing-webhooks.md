# Canonical Admin Outgoing Webhooks Design

Date: 2026-07-12

Status: Approved conversational design; written specification under review

Backlog: TASK-12950

## Summary

Replace the repository's two incompatible admin webhook implementations with one
public, mounted, secure outgoing-webhook capability. The final system has one
`admin_webhooks` router, one API contract, one AuthNZ repository, one secret and
target encryption boundary, and one Jobs-backed delivery path.

The control plane generates signing secrets on the server and reveals each new
secret only in the successful create or rotate response. The data plane records
immutable source events, creates one delivery per matching active registration,
and enqueues opaque delivery references into Jobs. Jobs owns retry timing, lease
recovery, and crash recovery. The webhook worker owns payload construction,
signing, egress-safe HTTP delivery, and terminal delivery state.

This is a generic GPL upstream feature. Hosted deployment, billing, invite
management, and other commercial behavior are outside this design.

## Current State And Failure

The current repository contains two different implementations under the same
conceptual `/admin/webhooks` surface:

- Mounted routes in `admin_ops` use string IDs, `events` and `enabled` fields,
  server-generated secrets, and plaintext registrations in `system_ops.json`.
- The unmounted `admin_webhooks` router and admin UI use numeric IDs,
  `event_types` and `active`, AuthNZ database tables, and encrypted secrets.
- SQLite migrations 080 and 082 create and harden the unmounted implementation's
  tables. Equivalent PostgreSQL DDL is absent.
- The unmounted service performs SQL directly, including SQLite
  `INSERT ... RETURNING` calls through a pool path that can close without a
  durable commit.
- The intended service sends with raw `httpx`, retries in-process, logs full
  destination URLs on failure, and persists payloads, signatures, and response
  bodies.
- URL policy is checked when configuration is accepted, but the delivery path
  does not provide a complete delivery-time DNS, proxy, and SSRF boundary.
- Only `incident.notify` calls the intended `dispatch_event`; the UI advertises
  event names that do not have durable producers.
- The admin UI expects a create-response secret that the API response model
  deliberately omits, so an operator can create a registration without ever
  receiving the server-generated signing secret.

The code was previously mounted in commit `2f6bc764e2`. Later integration kept
much of the intended code but dropped the mount and reintroduced the legacy
handlers. The same split remains on current `dev`; this is not a hosted-only
problem.

## Goals

- Expose exactly one canonical admin webhook API and router.
- Require platform-admin authorization and auditable privileged actions.
- Support SQLite and PostgreSQL with equivalent schema and behavior.
- Generate, encrypt, rotate, and reveal signing secrets safely.
- Encrypt full destination URLs because path and query values commonly contain
  receiver credentials.
- Migrate legacy JSON and existing database registrations without silent loss
  or duplicate delivery.
- Persist immutable events before asynchronous delivery.
- Use Jobs as the only automatic retry and lease authority.
- Prevent SSRF, DNS rebinding, redirect, and ambient-proxy bypasses at delivery
  time.
- Publish a stable, documented signing and payload protocol.
- Provide actionable health, backlog, delivery-history, and redelivery controls
  in the existing admin UI.
- Make disabled, migration-pending, key-unavailable, worker-offline, and degraded
  states explicit and fail closed.

## Non-Goals

- No inbound webhook receiver is added. Existing dedicated inbound integrations,
  including hosted Postmark invite events, remain separate.
- No tenant-created webhook API is added. This first version is platform-admin
  only.
- No wildcard event subscription is supported in the canonical API.
- No exactly-once delivery claim is made. The protocol is at-least-once.
- No ordering guarantee is made across event types, registrations, workers, or
  retries.
- No arbitrary custom headers, HTTP methods, request templates, or payload
  transformations are added.
- No receiver response body or arbitrary response header is retained.
- No broad Jobs rewrite is included. Only supported extension points required by
  this worker may be added.
- No destructive downgrade migration is promised after canonical writes begin.

## Final Architecture

### Router And Authorization

`admin_webhooks` becomes the sole mounted router for the canonical surface. The
legacy webhook CRUD, test, and delivery handlers are removed from `admin_ops`.
Incident management remains in `admin_ops`, but emits through the canonical event
producer interface.

All webhook endpoints require a platform-admin principal. The route layer owns:

- authentication and platform-admin authorization;
- request and response validation;
- idempotency-header validation;
- privileged-action confirmation where the existing admin framework requires it;
- audit records containing actor, action, webhook ID, target hostname, event
  type, outcome, request ID, and reason code;
- stable HTTP status and error-code mapping.

Audit records and logs never contain the full destination URL, URL path, query,
signing secret, signature, event body, receiver response body, or approved
incident narrative.

### Component Boundaries

The implementation is divided into focused units:

1. **Router and schemas**
   Defines the public admin API and maps service failures into stable responses.
2. **AuthNZ repository**
   Owns all webhook SQL, transactions, backend adaptation, claim leases,
   idempotency records, migration state, and retention queries.
3. **Control-plane service**
   Owns registration rules, server secret generation, encryption, rotation,
   catalog validation, lifecycle transitions, and audit inputs.
4. **Event producer**
   Creates immutable, deduplicated source events and automatic delivery rows.
5. **Delivery reconciler**
   Claims unqueued delivery rows and idempotently creates Jobs jobs even when the
   AuthNZ and Jobs databases are different backends.
6. **Webhook Jobs worker**
   Loads an opaque delivery reference, validates the current configuration
   version, builds and signs exact bytes, performs one HTTP attempt, records the
   attempt result, and lets Jobs decide whether and when to retry.
7. **Central egress helper**
   Provides a status-only, non-redirecting, bounded request path with delivery-
   time URL policy, DNS pinning, proxy restrictions, and no response buffering.
8. **Legacy importer**
   Dry-runs and imports existing JSON/database registrations, re-encrypts secret
   material, records source hashes and mappings, and sanitizes the active JSON
   file only after committed readback verification.

The router never performs SQL or outbound HTTP. The repository never sends HTTP.
The worker never accepts caller-supplied payloads or secrets from a Jobs job.

## Canonical Admin API

The canonical prefix remains `/api/v1/admin/webhooks`. Static routes must be
declared and tested before `/{webhook_id}` routes.

### Read And Status Endpoints

- `GET /catalog`
  Returns the supported event catalog, payload API version, limits, and event
  descriptions. It never returns a wildcard.
- `GET /status`
  Returns feature mode, schema/import state, encryption-key state, reconciler
  heartbeat, worker heartbeat, queue/backlog counts, oldest pending age, and
  retention status. Sensitive values are redacted.
- `GET /`
  Returns paginated, non-deleted registration metadata.
- `GET /{webhook_id}`
  Returns one registration's metadata.
- `GET /{webhook_id}/deliveries`
  Returns paginated delivery metadata and stable terminal reason codes.

List and get responses expose only a redacted destination display such as scheme
and hostname. The encrypted path/query is never returned after create or update.

### Mutation Endpoints

- `POST /`
  Creates an inactive registration and returns a create-specific response with
  the generated signing secret exactly once.
- `PATCH /{webhook_id}`
  Updates description, destination URL, event subscriptions, timeout, or active
  state. It never accepts a signing secret.
- `DELETE /{webhook_id}`
  Soft-deletes the registration and cancels automatic work that has not entered
  an HTTP request.
- `POST /{webhook_id}/rotate-secret`
  Requires the registration to be inactive and returns a rotate-specific
  response with the new secret exactly once.
- `POST /{webhook_id}/test`
  Performs one synchronous, bounded test attempt. It is allowed while inactive
  and does not use the automatic retry scheduler.
- `POST /{webhook_id}/deliveries/{delivery_id}/redeliver`
  Creates a new manual delivery row with the same event ID, a new delivery ID,
  and `redelivery_of_id` pointing to the selected delivery.

Create, rotate, and manual redelivery require an `Idempotency-Key` header. Keys
are scoped to actor, operation, and request body for 24 hours. An exact replay
returns the original result, including the encrypted-and-recovered one-time
secret for create or rotate. Reusing the same key with a different body returns
`409 idempotency_conflict`. Idempotency records expire and are removed by the
retention worker.

### Registration Contract

A registration includes:

- numeric `id`;
- operator-facing `description`;
- redacted destination display and `target_hostname`;
- explicit `event_types` from the current catalog;
- `active`, default `false`;
- `timeout_seconds`, default 10 and maximum 30;
- `delivery_config_version`;
- `secret_version`;
- creator/updater identity and timestamps;
- soft-delete metadata.

The server rejects caller-supplied signing secrets and wildcard subscriptions.
Changing URL, events, timeout, active state, or secret increments
`delivery_config_version`. Rotating the signing secret also increments
`secret_version`. Description-only changes do not supersede delivery work.

Disabling a registration cancels pending or retrying automatic deliveries with
reason `canceled_disabled`. Re-enabling affects future events only. Rotating a
secret cancels pending or retrying work from the prior version with reason
`canceled_secret_rotation`. Updating delivery configuration marks older work
`superseded_config`. Deleting is a tombstone operation and retains delivery
history until normal retention expires.

An already-running HTTP request cannot be recalled. The API and UI state this
when disable, rotate, update, or delete races an in-flight attempt.

## Secret And Destination Protection

### Signing Secret Format

Create and rotate generate 32 cryptographically random bytes on the server and
encode them as:

```text
whsec_<64 lowercase hexadecimal characters>
```

The full string is the HMAC key. It is returned only by the successful
create-specific or rotate-specific response, or by an exact idempotent replay
within the bounded replay window. List, get, update, status, audit, log, and
delivery-history responses never reveal it.

### Dedicated Key Ring

Webhook target URLs and signing secrets use a dedicated encryption key ring with
stable operator-assigned key IDs and one configured primary key ID. The key ring
uses the repository's AES-GCM JSON-envelope primitive but does not derive runtime
keys from BYOK, session, JWT, API-key, or other unrelated credentials.

The stored envelope records its key ID. Reads can decrypt with the primary or a
configured previous key. New writes always use the primary. A rotation command
re-encrypts every target and secret under the new primary, verifies readback,
and only then permits removal of the old key.

Legacy BYOK/session/JWT/API-key candidates may be loaded only inside the explicit
legacy migration command to decrypt old rows. Canonical runtime decryption never
falls back to them.

If no usable dedicated key is available:

- metadata list, disable, soft delete, delivery history, and status remain
  available;
- create, URL update, enable, rotate, test, automatic delivery, and manual
  redelivery return `503 admin_webhook_key_unavailable`;
- an exact create/rotate idempotency replay that would reveal a secret also
  returns `503` and remains replayable when the same key ring is restored; it
  never generates a replacement resource or secret;
- same-key/different-body detection can still return `409` from the stored
  request hash without decrypting replay material;
- no plaintext fallback is written.

Key rotation places secret-returning mutations and replays in a bounded
`key_rotation_in_progress` maintenance state. It re-encrypts registration
targets, registration secrets, and unexpired idempotency replay secrets before
readback verification. The previous key remains configured until every row is
verified. A request interrupted by rotation receives `503` and can retry with
the same idempotency key after rotation; it does not receive a partial or newly
generated response.

Rotation progress is durable in migration state: operation ID, source and target
key IDs, phase, last processed table/key cursor, processed count, verified count,
start time, and completion time. Each row update is idempotent because its
envelope records the target key ID. After a crash, the operator resumes the same
operation; the scanner skips already re-encrypted rows and continues from the
durable cursor, then performs a complete readback pass. Once any row has moved,
rotation is forward-resume only. The source key cannot be removed and
secret-returning operations remain unavailable until verification commits the
completed state.

The full destination URL is encrypted at rest. Separate non-secret hostname and
redacted-display columns support listing, policy review, and audit without
decrypting path/query credentials.

## Persistence Model

The canonical tables use new names rather than treating SQLite migrations 080
and 082 as final. This makes existing `admin_webhooks` and
`admin_webhooks_delivery_log` tables explicit legacy sources and permits an
expand/migrate/contract rollout.

### `admin_webhook_registrations`

Stores canonical registration metadata, encrypted target URL and signing secret,
key IDs, explicit event set, active/tombstone state, timeout, configuration and
secret versions, actor IDs, and timestamps.

### `admin_webhook_events`

Stores immutable event ID, event type, API version, aggregate type and ID,
aggregate version or command ID, creation time, bounded canonical payload JSON,
and source identity. A unique source key prevents duplicate producer writes.

The producer uniqueness key is either:

```text
(event_type, aggregate_id, aggregate_version)
```

or an explicit stable command ID for command-like events such as
`incident.notify`.

The schema represents these alternatives without nullable-unique ambiguity:

- `source_kind` is `aggregate` or `command`;
- aggregate rows require `aggregate_type`, `aggregate_id`, and
  `aggregate_version` and leave `source_command_id` null;
- command rows require `source_command_id` and leave aggregate identity null;
- a partial unique index covers
  `(event_type, aggregate_type, aggregate_id, aggregate_version)` where
  `source_kind='aggregate'`;
- a second partial unique index covers `(event_type, source_command_id)` where
  `source_kind='command'`.

SQLite and PostgreSQL migrations create equivalent check constraints and partial
indexes. Aggregate IDs and versions use bounded text so existing integer, UUID,
and file-version identities share one representation without lossy conversion.

### `admin_webhook_deliveries`

Stores delivery ID, event ID, webhook ID, kind (`automatic`, `manual`, or
`test`), snapshotted delivery/secret versions, Jobs ID, enqueue claim token and
expiry, state, attempt count, status code, latency, bounded error/reason code,
terminal time, expiry time, and optional `redelivery_of_id`.

It does not store a signature, response body, response headers, decrypted URL,
or duplicate payload body. Automatic rows have a deterministic unique delivery
key so one event produces at most one automatic delivery per registration.

### `admin_webhook_idempotency`

Stores a hashed/scoped idempotency key, operation, request hash, resource ID,
encrypted replay secret when applicable, response metadata, and expiry. Plain
idempotency keys are not retained.

### `admin_webhook_migration_state`

Stores the expected canonical schema version, legacy importer phase, source file
content hash, source table fingerprint, mapping/report digest, completion time,
and operator identity. Both SQLite and PostgreSQL expose the same logical state.

### Transaction Ownership

The concrete AuthNZ repository owns all SQL and uses backend-specific parameter
and transaction helpers. Insert-and-return operations must commit durably on
SQLite and PostgreSQL. Service methods cannot use raw pool `fetchone` as a write
shortcut.

## Legacy Migration

The importer supports two independent sources:

- mounted legacy registrations in `system_ops.json`;
- rows in the existing `admin_webhooks` table, including rows encrypted with the
  temporary unrelated-key fallback.

The importer is dry-run first and all imported registrations are inactive. It
validates destination policy, expands legacy `*` to only the event catalog that
exists at migration time, bounds timeouts, re-encrypts target and secret values
with the dedicated key ring, and produces a source-to-canonical mapping report.
It never silently merges conflicting registrations. Invalid or undecryptable
records remain in the report and keep migration status degraded until an
operator resolves or explicitly rejects them.

For `system_ops.json`, migration follows this sequence while legacy mutation
routes are quiesced:

1. Acquire the existing system-ops file lock.
2. Parse the file structurally, compute its content hash, and create a `0600`
   backup that is flushed and fsynced.
3. In one AuthNZ transaction, insert canonical registrations and the migration
   marker containing the source hash and mapping digest.
4. Commit, decrypt/read back every imported registration, and verify counts and
   mappings.
5. Reacquire the lock, require the same source hash, remove only legacy webhook
   fields from the active JSON object, preserve incident and unrelated data, and
   publish the sanitized file by atomic replace plus directory fsync.
6. Retain the protected plaintext backup for a bounded default seven-day
   rollback window, then purge it through an explicit, auditable command. The
   configured window may not exceed 30 days.

If the process crashes after the database commit but before JSON sanitization,
the committed source hash and mapping make rerun idempotent. A changed source
hash stops the importer for operator review.

Existing legacy database tables are retained during expand/migrate. A later,
separately reviewed contract migration removes or sanitizes obsolete rows only
after canonical readback, backup evidence, and rollback-window completion.

Feature status remains `migration_pending` until schema inspection, both legacy
source checks, readback, and active-file sanitization are complete.

## Event Catalog And Privacy

The initial automatic catalog contains exactly:

- `user.created`
- `user.deleted`
- `incident.created`
- `incident.updated`
- `incident.resolved`
- `incident.notify`

`webhook.test` is reserved for the synchronous test endpoint and is not a
subscription option. New event types require a future API-versioned catalog
change; existing registrations do not receive them automatically.

Routine user events contain stable user ID, lifecycle status, resource version,
and timestamps. They do not include email address, credentials, session data,
API keys, profile text, or billing data.

Routine incident events contain incident ID, state, severity, resource version,
and timestamps. They omit title, summary, tags, free text, evidence, and operator
notes. `incident.notify` may include an operator-approved narrative because
sending that narrative is the explicit command being requested. Its API and UI
must show the outbound content before confirmation.

The final UTF-8 encoded event body is limited to 64 KiB before persistence.
Oversized events fail at the producer with a stable reason code; they are not
truncated into misleading payloads.

### Durable Producers

Each initial event has one explicit durable source identity:

| Event | Durable source | Canonical source identity |
| --- | --- | --- |
| `user.created` | AuthNZ user-create transaction | operation command ID generated before the transaction and persisted with the event |
| `user.deleted` | AuthNZ user-delete transaction | operation command ID generated before the transaction and persisted with the deletion/event |
| `incident.created` | incident record plus pending marker in `system_ops.json` | incident ID and incident version |
| `incident.updated` | incident record plus pending marker in `system_ops.json` | incident ID and incident version |
| `incident.resolved` | incident record plus pending marker in `system_ops.json` | incident ID and incident version |
| `incident.notify` | explicit notify command marker in `system_ops.json` | operator command ID generated before the locked save |

User command IDs are created by the service before entering the database
transaction and are reused if that operation is retried. Incident versions are
incremented in the same locked file mutation that writes the marker. Notify
command IDs are reused across request/idempotency retries. The repository's
unique source key turns repeated producer or reconciler calls into reads of the
existing event rather than new events.

Database-backed user mutations insert the source event in the same AuthNZ
transaction as the user change. A committed user mutation cannot lose its event,
and a rolled-back mutation cannot emit one.

Incident state remains file-backed. The incident mutation writes a minimal
pending event marker under the same `system_ops.json` lock and atomic save. A
reconciler inserts the canonical DB event using the marker's stable identity,
then removes the marker under the file lock only after database commit. A crash
can duplicate reconciliation attempts but cannot duplicate the canonical event.

Ordering is not guaranteed. Every payload includes aggregate version and event
creation time so receivers can reject stale state.

## Delivery And Jobs

### Event Expansion

When an event is persisted, the producer creates one automatic delivery row for
each matching active, non-deleted registration using that registration's current
delivery and secret versions. Event and delivery creation occur in the same
AuthNZ transaction.

Changing a registration later never retargets pending automatic work. Before an
attempt, the worker compares snapshotted and current versions. A mismatch becomes
terminal `superseded_config`; disable, rotation, and deletion use their more
specific cancellation reason codes.

Manual redelivery is an explicit exception: it creates a new row using the
current active configuration and the historical event. If the configuration
version differs from the original, the API requires an explicit confirmation,
the UI displays both versions and the current hostname, and the audit record
marks `redelivery_to_changed_config=true`.

### Synchronous Test Attempts

The test endpoint is persisted but does not create a Jobs job. In one AuthNZ
transaction it creates a `webhook.test` event with a new command ID and a
`kind=test` delivery row using the current registration/configuration versions.
The service generates the test-attempt token before that transaction and commits
the delivery directly in `processing` with token, attempt number one, and
`started_at`. There is no committed `pending` test state and therefore no gap in
which the row lacks both a recovery token and a Jobs job. After commit, the
endpoint invokes the same bounded `DeliveryAttemptExecutor` used by the Jobs
worker and waits for exactly one attempt.

The executor owns deterministic payload bytes, signature headers, final
configuration checks, central egress I/O, and bounded attempt metadata. The Jobs
worker supplies Jobs ID, lease ID, and attempt number. The synchronous test path
instead supplies a random test-attempt token and attempt number one. It never
calls Jobs retry APIs and never schedules another attempt.

Test requires completed migration and an available encryption key but does not
require worker heartbeat. It is allowed while the registration is inactive,
sets `X-TLDW-Webhook-Test: true`, and retains its event/delivery metadata under
normal 30-day terminal retention. If the API process dies after marking a test
attempt `processing` but before committing a terminal result, recovery marks the
row `dead` with reason `test_attempt_interrupted`; it does not retry an
operator's test implicitly. A test is stale after the maximum 30-second request
timeout plus a 90-second recovery margin. Recovery conditionally matches the
test-attempt token so a late completion cannot overwrite a recovered terminal
row.

### Cross-Database Enqueue Handshake

AuthNZ and Jobs can each use SQLite or PostgreSQL, including different databases.
The reconciler cannot rely on a distributed transaction. It uses this recoverable
handshake:

1. Atomically claim an unqueued delivery in AuthNZ with a random claim token and
   expiry.
2. Create or read a Jobs job using idempotency key
   `admin-webhook-delivery:<delivery_id>` and payload containing only the
   delivery ID.
3. Conditionally attach the Jobs ID to the delivery using the same claim token.
4. Clear the claim and mark the row queued.

If the reconciler crashes at any point, an expired claim can be recovered. A
reconciler that finds an existing Jobs job by idempotency key attaches it rather
than creating another. Tests cover every crash point with all four AuthNZ/Jobs
backend combinations.

### Delivery State Machine And Ownership

AuthNZ stores the operator-facing delivery state. Jobs stores lease, schedule,
retry, and job terminal state. Neither database is treated as a distributed
transaction participant.

| Delivery state | Writer | Meaning and allowed next states |
| --- | --- | --- |
| `pending` | event producer | Durable automatic/manual row exists but has no enqueue claim; next is `enqueue_claimed`, a terminal lifecycle state, or `dead` on expiry; test rows are never committed in this state |
| `enqueue_claimed` | reconciler | A claim token/expiry owns the enqueue handshake; next is `queued`, back to `pending` after safe recovery, or a terminal lifecycle state |
| `queued` | reconciler | One Jobs ID is attached; next is `processing`, `canceled`, `superseded`, or `dead` on expiry |
| `processing` | worker or synchronous test service | One Jobs-leased attempt or one committed test-attempt token owns execution; the executor still performs final pre-I/O checks; next is `succeeded`, `retry_wait` for Jobs work only, `dead`, or a terminal lifecycle state observed before I/O |
| `retry_wait` | worker | Attempt metadata is recorded and Jobs owns the next availability time; next is `processing`, `canceled`, `superseded`, or `dead` |
| `succeeded` | worker | Receiver returned 2xx; terminal |
| `dead` | worker or expiry reconciler | Nonretryable response, exhausted retries, or `delivery_expired`; terminal |
| `canceled` | control-plane service or worker | Disabled, rotated, or deleted before an HTTP request began; terminal with a specific reason code |
| `superseded` | control-plane service or worker | Delivery configuration no longer matches and no HTTP request began; terminal |

The producer and reconciler update AuthNZ only. The worker conditionally marks
`processing` with Jobs ID, lease ID, and attempt number before I/O. The test
service conditionally marks `processing` with its test-attempt token. A retryable
Jobs failure records bounded attempt metadata and `retry_wait`, then raises the
typed Jobs retry signal; it does not calculate or poll its own timer. On
reacquisition, the same Jobs ID and delivery ID return to `processing` with a
higher attempt. A test failure always transitions directly to `dead` with its
HTTP/network reason and no retry.

Lifecycle cancellation uses conditional updates and asks Jobs to cancel queued
or retrying work. It cannot convert an already-started HTTP request into a claim
that nothing was sent. If disable, rotate, update, or delete wins before the
worker's final pre-I/O compare-and-set, the worker records the matching terminal
reason and sends nothing. If the worker has already crossed that boundary, it
records the real attempt result. A successful response remains `succeeded` with
an audit/metadata flag such as `completed_after_config_change`; it is not
rewritten as canceled.

Terminal AuthNZ states are monotonic. Recovery code may repair a stale
nonterminal mirror from the authoritative Jobs record, but cannot overwrite a
terminal delivery with an older Jobs observation. A missing Jobs row for a
claimed/queued delivery is reconciled through the same idempotency key; a Jobs
terminal state without an AuthNZ terminal state is repaired idempotently from
bounded job result metadata.

### Worker Contract

The Jobs domain and queue are dedicated to admin webhooks. The job payload is:

```json
{"delivery_id":"<opaque delivery UUID>"}
```

The worker loads event, registration, encrypted destination, and signing secret
from AuthNZ at attempt time. It checks feature mode, tombstone/active state,
versions, delivery expiry, and key availability before performing network I/O.

Jobs is the sole automatic retry schedule authority. The delivery table mirrors
attempt and terminal state for admin visibility but never runs its own retry
timer. The Jobs worker API gains supported per-worker/per-job controls for:

- an exact retry-delay override so the webhook schedule is not multiplied by a
  second generic exponential calculation;
- a quarantine threshold above this worker's maximum attempts.

For this worker, `max_retries=3`, producing one initial attempt plus three
retries. The supported quarantine threshold is set above four attempts so normal
retryable HTTP failures do not hit the global poison-message default of two.

### Retry Classification

Success is any `2xx` response. The automatic retry schedule is:

- first retry: 1 minute;
- second retry: 5 minutes;
- third retry: 30 minutes.

Network failures, timeouts, HTTP 408, HTTP 429, and HTTP 5xx are retryable.
Redirects and all other 4xx responses are terminal. Redirects are never followed.

For 429 and 503, the status-only egress helper may expose only a syntactically
valid `Retry-After` value. Delta seconds and RFC-compliant HTTP dates are parsed
with standard-library code and clamped to 1 through 1800 seconds. Effective
delay is the greater of the scheduled delay and the bounded receiver value. No
other response header is retained.

Automatic deliveries that have not succeeded within 72 hours become terminal
`dead` with reason `delivery_expired`, even if worker downtime prevented their
first attempt.

### At-Least-Once Semantics

A worker can lose its lease after the receiver accepts an attempt but before the
terminal state commits. The same delivery ID may therefore be sent again.
Receivers must deduplicate by event ID or delivery ID as appropriate. This is an
explicit protocol property, not an error hidden by the service.

## External Delivery Protocol

### Body

Every event uses this envelope:

```json
{
  "id": "event UUID",
  "type": "incident.updated",
  "api_version": "2026-07-01",
  "created_at": "2026-07-12T19:00:00Z",
  "data": {}
}
```

The body is UTF-8 JSON encoded deterministically with sorted keys, compact
separators, and no NaN/Infinity values. Retries of the same event use identical
body bytes.

### Headers

```text
Content-Type: application/json
X-TLDW-Webhook-Event: incident.updated
X-TLDW-Webhook-Event-Id: <event UUID>
X-TLDW-Webhook-Delivery-Id: <delivery UUID>
X-TLDW-Webhook-Timestamp: <Unix seconds for this attempt>
X-TLDW-Webhook-Secret-Version: <integer>
X-TLDW-Webhook-Signature: v1=<lowercase hex HMAC>
X-TLDW-Webhook-Test: true                 # test attempts only
```

The signature is HMAC-SHA256 over:

```text
<unix_timestamp>.<exact_raw_body_bytes>
```

using the full `whsec_...` value as the key. Receiver documentation recommends
constant-time signature comparison, a five-minute timestamp freshness window,
and event-ID deduplication. The timestamp and signature are regenerated for each
attempt; event and delivery IDs remain stable across automatic retries.

## Egress Security

Registration validation and every delivery attempt use the central Security
egress policy. The webhook service does not instantiate raw `httpx` clients.

The central helper contract for webhooks is:

- HTTPS only by default; development HTTP requires an explicit non-production
  override.
- No URL user-info, fragments, unsupported ports, or malformed hostnames.
- Global and webhook-specific allow/deny policy support.
- Private, loopback, link-local, multicast, documentation, translation, and
  reserved IPv4/IPv6 ranges blocked by default.
- DNS resolution is bounded and fail-closed.
- The transport connects only to an address approved for the original hostname
  while preserving hostname-based TLS verification and `Host` semantics.
- Policy and DNS checks are repeated for every attempt.
- Redirects are disabled.
- Ambient proxy environment variables are ignored. Any explicit proxy must pass
  the central proxy allowlist.
- TLS verification cannot be disabled by registration configuration.
- Request timeout defaults to 10 seconds and cannot exceed 30 seconds.
- Response body and ordinary headers are neither buffered nor persisted. The
  helper returns status, latency, and bounded parsed `Retry-After` only.
- Logs receive webhook ID, target hostname, status/reason code, attempt, and
  latency, never the full URL.

The helper is reusable public infrastructure, but this task does not claim to
migrate every existing outbound caller to it.

## Feature Modes And Degraded States

`TLDW_ADMIN_WEBHOOKS_MODE` has three values:

- `off` (default): canonical delivery and mutations are unavailable; platform
  admins can read sanitized status.
- `migrate`: schema/import status and migration tooling are available; CRUD and
  delivery remain unavailable.
- `on`: canonical API, producers, reconciler, and worker are enabled if preflight
  succeeds.

During the first stacked upstream PR, legacy runtime behavior may remain behind
an explicit compatibility mode so the repository is not released in a partially
migrated state. The second PR removes that temporary compatibility path. The
final state always has one mounted canonical router.

When migration is pending, only status is available and other calls return
`503 admin_webhook_migration_pending`. When keys are unavailable, the restricted
metadata operations described above remain available. When the worker is
offline, configuration remains visible and durable source events continue to
queue; status and UI show the outage and backlog age. Enabling a previously
inactive registration requires current worker health, but already-active
registrations are not silently disabled during a transient worker outage.

Worker and reconciler heartbeats have explicit freshness thresholds and are
included in health metrics. Preflight checks schema version, migration state,
key ring, Jobs database access, queue registration, worker heartbeat, reconciler
heartbeat, and oldest pending delivery.

## Retention And Observability

Terminal delivery metadata is retained for 30 days after `terminal_at`. Events
remain until all dependent deliveries are terminal and the latest terminal row
has passed retention. Pending expiry does not start the retention clock until a
terminal state is committed.

Retention removes expired idempotency records, terminal delivery rows, orphaned
events, and expired migration backups through explicit bounded batches. It never
deletes active or nonterminal work.

Metrics include registrations by state, events created, enqueue claims and
recoveries, Jobs enqueue failures, deliveries by state/reason/status class,
attempt latency, retries, expired deliveries, worker/reconciler heartbeat age,
backlog size, oldest pending age, key errors, migration state, and SSRF denials.
Metrics and labels contain no full URL, user email, narrative, payload, or secret.

## Admin UI

The existing admin Webhooks page consumes the canonical API rather than carrying
its own event catalog or secret assumptions. It provides:

- catalog-driven create and edit forms;
- inactive-by-default creation;
- a one-time create/rotate secret dialog with copy and acknowledgement controls;
- no secret in list/get fixtures or browser storage;
- redacted hostname display;
- explicit migration, key, reconciler, and worker health states;
- delivery history with reason codes and retry metadata;
- disable, rotate, soft delete, test, and manual redelivery controls;
- a changed-configuration warning and explicit confirmation for redelivery to a
  newer configuration;
- a warning that in-flight HTTP requests cannot be recalled.

The page does not expose arbitrary payload editing, custom headers, wildcard
events, or receiver response bodies.

## Rollout And Rollback

The upstream feature defaults off. Schema expansion is additive. Legacy import
is dry-run first and explicit. Canonical activation requires successful readback,
key and worker preflight, and operator action.

Rollback has three boundaries:

1. **Before import:** code and additive schema can be reverted normally.
2. **After import but before any canonical mutation or delivery:** disable the
   feature and use the protected mapping/backup for an offline legacy restore if
   necessary.
3. **After the first canonical create, update, rotation, redelivery, or automatic
   event:** do not revert to the legacy writer. Disable delivery and forward-fix.
   Legacy restore at this point can lose canonical-only state.

Encryption keys required by stored canonical rows must remain available through
rollback and recovery. No rollback drops canonical tables or delivery history.

## Upstream Review Units

### PR 1: Canonical Control Plane And Migration

- canonical schemas, router contract, platform-admin authorization, and audit;
- repository-owned SQLite/PostgreSQL schema and transaction behavior;
- dedicated key ring, encrypted target URLs, server-generated secret lifecycle,
  and idempotent one-time responses;
- event catalog and status surface;
- dry-run/import/readback/sanitization tooling for legacy JSON and DB rows;
- route uniqueness and temporary compatibility-mode controls;
- admin UI control-plane contract updates where needed.

The feature remains non-releaseable in canonical `on` mode until PR 2 lands.

### PR 2: Durable Events And Delivery

- durable user and incident producers;
- event and delivery expansion;
- reconciler claim lease and cross-database Jobs handshake;
- supported Jobs exact-delay and quarantine-threshold extension points;
- webhook Jobs worker, central status-only egress helper, signing, retry, expiry,
  retention, metrics, and health;
- test and manual redelivery behavior;
- delivery-history and operational admin UI;
- removal of temporary legacy compatibility routing;
- receiver and operator documentation.

No release may enable canonical mode with only PR 1 present.

## Verification Gates

### PR 1 Gates

- exactly one canonical route per method/path and no final legacy duplicate;
- platform-admin allow/deny tests and complete mutation audit tests;
- OpenAPI request/response and stable error-code tests;
- SQLite fresh install and upgrades from pre-080, 080, and 082 states;
- PostgreSQL fresh install and representative legacy upgrade;
- durable SQLite write/commit regression tests;
- JSON and legacy-table dry run, import, conflict, crash-point, readback,
  sanitization, and rerun tests;
- dedicated key absence, primary/previous rotation, legacy re-encryption, and
  runtime-fallback rejection tests;
- create/rotate idempotent replay and same-key/different-body conflict tests;
- encrypted target and secret at-rest assertions;
- event catalog, wildcard rejection, timeout, payload-size, and static-route
  ordering tests;
- admin UI create/rotate one-time-secret tests and redacted list/get fixtures;
- Bandit on touched Python and generated OpenAPI review.

### PR 2 Gates

- transactional user producers and file-marker incident crash recovery;
- producer source-identity uniqueness and duplicate reconciliation;
- one automatic delivery per matching registration;
- all cross-database enqueue crash points for SQLite/SQLite,
  SQLite/PostgreSQL, PostgreSQL/SQLite, and PostgreSQL/PostgreSQL;
- Jobs exact retry schedule, max attempts, quarantine behavior, lease recovery,
  cancellation, and idempotent enqueue;
- configuration supersession, disable, rotation, soft delete, manual redelivery,
  in-flight race, 72-hour expiry, and 30-day retention;
- payload privacy and 64 KiB enforcement for all six event types;
- deterministic body and published signature test vectors;
- delivery-time SSRF, DNS change, proxy, redirect, TLS, timeout, no-buffer, and
  URL-redaction tests;
- retry classification and bounded `Retry-After` parsing;
- local controlled receiver end-to-end tests for signature, duplicates, retry,
  and test headers;
- worker/reconciler health and backlog preflight tests;
- complete admin UI test, typecheck, lint, and production build;
- Bandit on touched Python.

PostgreSQL tests use disposable databases through the repository's existing
fixtures. They never reuse staging or production data.

## Documentation Deliverables

- public API and event catalog reference;
- receiver signature-verification and deduplication guide;
- migration/import and legacy-backup runbook;
- encryption-key provisioning and rotation runbook;
- worker, backlog, dead-delivery, redelivery, and retention runbook;
- rollout, disable, forward-fix, and rollback-boundary runbook;
- release note stating at-least-once and unordered delivery semantics.

## Acceptance Criteria

- One canonical `admin_webhooks` router is mounted and legacy webhook handlers
  are absent from the final runtime.
- SQLite and PostgreSQL provide equivalent canonical schema, migration state,
  repository behavior, and legacy import.
- Signing secrets and destination URLs are encrypted with a dedicated rotatable
  key ring; create and rotate reveal server-generated secrets only through their
  specific idempotent responses.
- Six documented event producers create immutable, deduplicated events and
  versioned automatic deliveries.
- Jobs is the only automatic retry authority and cross-database enqueue is
  crash-recoverable.
- Delivery uses the published protocol and central egress-safe, status-only HTTP
  path with no full URL, payload, signature, or receiver body in logs/history.
- Degraded modes, retention, health, manual redelivery, expiry, and rollback
  boundaries are explicit and tested.
- PR 1 and PR 2 pass their defined SQLite, PostgreSQL, security, admin UI,
  Bandit, and end-to-end gates before canonical mode can be enabled.
