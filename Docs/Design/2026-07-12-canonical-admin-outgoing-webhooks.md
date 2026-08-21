# Canonical Admin Outgoing Webhooks Design

Date: 2026-07-12

Revalidated: 2026-08-21 against `origin/dev` at
`2e0815c1e4577902a220044619822ab6b1cb395f`

Status: Approved conversational design; written specification under review

Backlog: TASK-13013 (replaces the colliding historical TASK-12950 record)

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
handlers. The same split remains at the revalidated `origin/dev` commit above;
this is not a hosted-only problem. No webhook-specific runtime repair landed
between the original 2026-07-12 review base and that commit.

The surrounding platform did change materially. Current `dev` now includes the
peer-verified one-hop primitive in `Security/http_hop.py`, and Jobs admission and
lease operations have been substantially refactored. Implementation planning
must extend those current public contracts rather than replaying assumptions
from the older review base.

## Goals

- Expose exactly one canonical admin webhook API and router.
- Require platform-admin authorization and auditable privileged actions.
- Support SQLite and PostgreSQL with equivalent schema and behavior.
- Generate, encrypt, rotate, and reveal signing secrets safely.
- Encrypt full destination URLs because path and query values commonly contain
  receiver credentials.
- Encrypt exact canonical event bodies at rest, including an approved
  `incident.notify` narrative.
- Prevent lost updates and stale privileged actions through explicit resource
  revisions and conditional mutations.
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
   AuthNZ and Jobs databases are different backends. It also repairs durable
   pending Jobs dispositions and cancellation requests after cross-database
   crash windows.
6. **Webhook Jobs worker**
   Loads an opaque delivery reference, validates the current configuration
   version, builds and signs exact bytes, performs one HTTP attempt, records an
   append-only attempt result plus a durable Jobs disposition, and lets Jobs
   decide whether and when to retry.
7. **Central egress helper**
   Adapts the existing peer-verified `Security/http_hop.py` primitive into a
   status-only, non-redirecting, bounded request path with delivery-time URL
   policy, DNS pinning, proxy restrictions, and no response buffering.
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
  Returns one registration's metadata and a strong revision ETag.
- `GET /{webhook_id}/deliveries`
  Returns paginated delivery metadata and stable terminal reason codes.

List and get responses expose only a redacted destination display such as scheme
and hostname. The encrypted path/query is never returned after create or update.
Registration responses include integer `revision`; the response ETag is derived
from webhook ID and revision without containing secret material. Create, get,
PATCH, and rotate responses that carry a registration representation also carry
its current strong ETag so the UI never has to synthesize one.

### Mutation Endpoints

- `POST /`
  Creates an inactive registration and returns a create-specific response with
  the generated signing secret under the bounded version-bound replay contract.
- `PATCH /{webhook_id}`
  Updates description, destination URL, event subscriptions, timeout, or active
  state. It never accepts a signing secret and requires the current ETag in
  `If-Match`.
- `DELETE /{webhook_id}`
  Requires `If-Match`, soft-deletes the registration, and cancels automatic work
  that has not entered an HTTP request.
- `POST /{webhook_id}/rotate-secret`
  Requires `If-Match` and an inactive registration, then returns a
  rotate-specific response with the new secret through the bounded idempotency
  replay contract.
- `POST /{webhook_id}/test`
  Requires `If-Match` plus the reviewed `delivery_config_version` and performs
  one synchronous, bounded test attempt. It is allowed while inactive and does
  not use the automatic retry scheduler.
- `POST /{webhook_id}/deliveries/{delivery_id}/redeliver`
  Creates a new manual delivery row with the same event ID, a new delivery ID,
  and `redelivery_of_id` pointing to the selected delivery. It requires
  `If-Match`, and the request carries the delivery-configuration version the
  operator reviewed; a change before commit returns `412 precondition_failed`.

Missing required `If-Match` returns `428 precondition_required`; a syntactically
valid but noncurrent value returns `412 precondition_failed`. Successful
mutations return the resulting revision/ETag where applicable.

Create, rotate, test, and manual redelivery require an `Idempotency-Key` header.
Keys are scoped to actor, operation, canonical route identity, and request body
for 24 hours. Route identity includes the webhook and delivery IDs when present,
so an empty rotate body cannot replay another resource's result. The request
hash also includes normalized conditional headers and reviewed version fields.
Reusing the same scoped key with a different request returns
`409 idempotency_conflict`. Keys are 16-255 characters from
`[A-Za-z0-9._:-]`; the admin UI encodes 16 random bytes as 32 lowercase
hexadecimal characters and documentation tells other clients to use at least
equivalent entropy.

For an existing scoped key, lookup and request-hash comparison happen before a
current-resource precondition check. An exact replay can therefore return its
recorded result even though the successful first request changed the revision;
a conflicting request cannot use that ordering to bypass `If-Match`. A new
operation claims its idempotency record and validates every current precondition
in the same transaction as its mutation.

An exact create/rotate replay recovers the original secret only while that
resource still uses the recorded `secret_version` and is not deleted. If a later
rotation or deletion superseded the response, replay returns
`409 idempotency_result_superseded` and never reveals an obsolete secret or
creates a replacement. A successful replay identifies itself as a replay and
returns the recorded resource and secret versions. Idempotency records expire
and are removed by the retention worker.

### Registration Contract

A registration includes:

- numeric `id`;
- operator-facing `description`;
- redacted destination display and `target_hostname`;
- explicit `event_types` from the current catalog;
- `active`, default `false`;
- `timeout_seconds`, default 10 and maximum 30;
- monotonic `revision` for optimistic concurrency;
- `delivery_config_version`;
- `secret_version`;
- creator/updater identity and timestamps;
- soft-delete metadata.

The server rejects caller-supplied signing secrets and wildcard subscriptions.
Changing URL, events, timeout, active state, or secret increments
`delivery_config_version`. Rotating the signing secret also increments
`secret_version`. Every effective mutation increments `revision`.
Description-only changes do not supersede delivery work. A PATCH that already
matches persisted values is a no-op: it does not increment any version and may
return the current representation. Stale `If-Match` values fail before mutation.

Disabling a registration cancels pending or retrying automatic deliveries with
reason `canceled_disabled`. Re-enabling affects future events only. Rotating a
secret cancels pending or retrying work from the prior version with reason
`canceled_secret_rotation`. Updating delivery configuration marks older work
`superseded_config`. Deleting is a tombstone operation and retains delivery
history until normal retention expires.

An already-running HTTP request cannot be recalled. The API and UI state this
when disable, rotate, update, or delete races an in-flight attempt.

### Admission And Fanout Bounds

The first release has no unbounded registration mode. Validated deployment
settings default to 100 non-deleted registrations and 25 active registrations;
both are positive, the active limit cannot exceed the non-deleted limit, and
neither may exceed the hard implementation ceiling of 1,000. Catalog and status
responses expose the effective limits. Create or activation beyond a limit
fails with `409 admin_webhook_registration_limit` or
`409 admin_webhook_active_limit` and does not partially mutate state.

The active-registration bound makes delivery fanout inside a source mutation
finite. Event expansion uses a set-based repository operation rather than one
application query per registration. Lowering a configured limit below current
usage never deletes or silently disables registrations: preflight becomes
degraded, further create/activation is rejected, existing durable producers
continue to fan out to the finite current set, and status requires operator
action.

A soft-deleted registration is purged only after its 30-day minimum retention,
all dependent deliveries/events and unexpired idempotency records are gone, and
no migration or rollback reference remains. Audit retention is independent.

## Secret And Destination Protection

### Signing Secret Format

Create and rotate generate 32 cryptographically random bytes on the server and
encode them as:

```text
whsec_<64 lowercase hexadecimal characters>
```

The full string is the HMAC key. It is returned only by the successful
create-specific or rotate-specific response, or by an eligible exact idempotent
replay within the bounded replay window. List, get, update, status, audit, log,
and delivery-history responses never reveal it.

### Dedicated Key Ring

Webhook target URLs, signing secrets, canonical event bodies, and secret-bearing
idempotency responses use a dedicated encryption key ring with stable operator-
assigned key IDs and one configured primary key ID. The key ring uses the
repository's AES-GCM JSON-envelope primitive but does not derive runtime keys
from BYOK, session, JWT, API-key, or other unrelated credentials.

The stored envelope records its key ID. Reads can decrypt with the primary or a
configured previous key. New writes always use the primary. A rotation command
re-encrypts every protected value under the new primary, verifies readback, and
only then permits removal of the old key.

Each encrypted plaintext includes and validates a purpose plus stable row
identity: registration ID and delivery/secret version for targets and secrets,
event ID and API version for event bytes, and operation/resource/version for
idempotency replay material. Pending incident markers bind command/source
identity and event API version. Exact body bytes are stored in a byte-safe
encoded field inside that contextual envelope. This binds ciphertext to its
intended row/marker even though the existing AES-GCM JSON helper has no
associated-data API; copying an envelope to another row, marker, or field fails
closed after decryption.

Legacy BYOK/session/JWT/API-key candidates may be loaded only inside the explicit
legacy migration command to decrypt old rows. Canonical runtime decryption never
falls back to them.

If no usable dedicated key is available:

- metadata list, disable, soft delete, delivery history, and status remain
  available;
- while mode is `on`, user/incident source mutations fail before their domain
  commit with `503 admin_webhook_key_unavailable`; the system never commits a
  source mutation while omitting its required encrypted event;
- fresh create, URL update, enable, rotate, test, and manual redelivery return
  `503 admin_webhook_key_unavailable`; automatic workers acquire no new work and
  any already-leased pre-I/O work uses no-attempt deferral;
- an exact create/rotate idempotency replay that would reveal a secret also
  returns `503` and remains replayable when the same key ring is restored; it
  never generates a replacement resource or secret;
- same-key/different-body detection can still return `409` from the stored
  request hash without decrypting replay material;
- exact test/manual-redelivery replays may return their existing bounded metadata
  without decrypting event or destination material and never perform I/O;
- no plaintext fallback is written.

An operator can restore the key ring and retry the source mutation, or explicitly
set webhook mode `off` to restore ordinary product mutations without event
capture. That mode change is an acknowledged availability-over-delivery decision
shown in status/audit; key failure never makes that decision implicitly.

Key rotation places secret-returning mutations and replays in a bounded
`key_rotation_in_progress` maintenance state. It re-encrypts registration
targets, registration secrets, retained canonical event bodies, pending
encrypted incident markers, and unexpired idempotency replay secrets before
readback verification. The previous key remains configured until every database
row and file marker is verified. A request interrupted by rotation receives
`503` and can retry with the same idempotency key after rotation; it does not
receive a partial or newly generated response.

Rotation progress is durable in migration state: operation ID, source and target
key IDs, phase, last processed table/key cursor, processed count, verified count,
start time, and completion time. Each row update is idempotent because its
envelope records the target key ID. After a crash, the operator resumes the same
operation; the scanner skips already re-encrypted rows and continues from the
durable cursor, then performs a complete readback pass. Once any row has moved,
rotation is forward-resume only. The source key cannot be removed and
secret-returning operations remain unavailable until verification commits the
completed state.

Pending incident-marker scans and rewrites use the existing system-ops file lock
and atomic-save contract. Final verification repeats until no envelope using the
source key exists in either database rows or the locked active file.

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
secret versions, resource revision, actor IDs, and timestamps.

### `admin_webhook_events`

Stores immutable event ID, event type, API version, aggregate type and ID,
aggregate version or command ID, creation time, encrypted exact canonical body
bytes with envelope key ID, and source identity. The decrypted body is bounded
to 64 KiB and validated against the plaintext identity columns. A unique source
key prevents duplicate producer writes. Event content, including any approved
`incident.notify` narrative, is never persisted as plaintext JSON.

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
expiry, state, attempt count, current attempt ID, status code, latency, bounded
error/reason code, durable pending Jobs disposition and its application state,
terminal time, expiry time, and optional `redelivery_of_id`.

It does not store a signature, response body, response headers, decrypted URL,
or duplicate payload body. Automatic rows have a deterministic unique delivery
key so one event produces at most one automatic delivery per registration.

### `admin_webhook_delivery_attempts`

Stores one append-only row per reserved network-attempt slot: opaque attempt ID,
delivery ID, monotonic delivery-local sequence, Jobs/lease or synchronous-test
token, start and finish times, state, status code, latency, bounded reason code,
bounded requested retry delay, and whether the Jobs disposition was applied.
It never stores destination URLs, payloads, signatures, response bodies, or
ordinary response headers.

After lease/configuration/attempt-budget checks, the repository inserts the
attempt row and marks the parent delivery processing in one compare-and-set
transaction immediately before network I/O. Completion conditionally matches
the attempt token and records both the immutable attempt outcome and delivery
summary. A crash in the narrow post-reservation/pre-I/O gap is conservatively
indistinguishable from a send. Recovery closes the stale attempt as
`outcome_unknown` before any later attempt; history therefore does not falsely
claim that an ambiguous attempt was never sent.

### `admin_webhook_idempotency`

Stores a hashed/scoped idempotency key, operation, canonical route identity,
request hash, lifecycle state, resource and version IDs, encrypted replay secret
when applicable, response metadata, and expiry. Test records point to their
single delivery/attempt so an in-progress or completed request can be replayed
without another network call. Plain idempotency keys are not retained.

### `admin_webhook_migration_state`

Stores the expected canonical schema version, legacy importer phase, source file
webhook-subtree hash, protected full-backup digest, source table fingerprint,
mapping/report digest, completion time, and operator identity. Both SQLite and
PostgreSQL expose the same logical state.

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

Dry-run also reports projected non-deleted/active counts. Import writes nothing
when accepted sources would exceed the configured or hard registration ceiling;
the operator must raise the configured bound within 1,000 or explicitly reject/
archive named legacy entries. Migration never truncates the source set to fit.

For `system_ops.json`, migration follows this sequence while legacy webhook and
incident-notify mutation routes are quiesced:

1. Acquire the existing system-ops file lock.
2. Parse the file structurally and hash the canonicalized legacy webhook
   subtree. Create a full-file `0600` backup encrypted under a one-time rollback
   key held outside the data directory; flush and fsync the ciphertext and
   record its digest. The key is never stored beside or inside the backup.
3. In one AuthNZ transaction, insert canonical registrations and the migration
   marker containing the webhook-subtree hash, backup digest, and mapping
   digest.
4. Commit, decrypt/read back every imported registration, and verify counts and
   mappings.
5. Reacquire the lock, require the same webhook-subtree hash, remove only legacy
   webhook fields from the current JSON object, preserve incident and unrelated
   changes made between lock windows, and publish the sanitized file by atomic
   replace plus directory fsync.
6. Retain the encrypted backup and separate rollback key for a bounded default
   seven-day rollback window, then purge the active backup and destroy the key
   through an explicit, auditable command. The configured active window may not
   exceed 30 days. Infrastructure-backup retention is documented separately;
   destroying the one-time key makes retained ciphertext unusable after the
   rollback window.

If the process crashes after the database commit but before JSON sanitization,
the committed webhook-subtree hash and mapping make rerun idempotent. A changed
webhook subtree stops the importer for operator review; unrelated incident-file
changes do not invalidate the mapping or get overwritten.

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
and a rolled-back mutation cannot emit one. In mode `on`, the service resolves a
writable primary webhook key before opening that transaction; key failure aborts
the source mutation rather than writing a plaintext/missing event.

Incident state remains file-backed. The incident mutation writes a minimal
pending event marker under the same `system_ops.json` lock and atomic save. A
reconciler inserts the canonical DB event using the marker's stable identity,
then removes the marker under the file lock only after database commit. A crash
can duplicate reconciliation attempts but cannot duplicate the canonical event.
The locked source mutation likewise requires a writable primary key in mode
`on`, so its pending marker carries the encrypted canonical body rather than a
plaintext narrative or deferred best-effort payload.

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
transaction it claims the idempotency key, checks the reviewed registration
revision/configuration, creates a `webhook.test` event with a new command ID and
a `kind=test` delivery row using those versions, and inserts its first delivery-
attempt row. The service generates the test-attempt ID and token before that
transaction and commits the delivery and attempt directly in `processing` with
attempt sequence one and `started_at`. There is no committed `pending` test state
and therefore no gap in which the row lacks both recovery identity and a Jobs
job. After commit, the endpoint invokes the same bounded
`DeliveryAttemptExecutor` used by the Jobs worker and waits for exactly one
attempt.

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
attempt `outcome_unknown` and the delivery `dead` with reason
`test_attempt_interrupted`; it does not retry an operator's test implicitly. A
test is stale after the maximum 30-second request timeout plus a 90-second
recovery margin. Recovery conditionally matches the test-attempt token so a late
completion cannot overwrite a recovered terminal row or attempt.

An exact retry of a processing test returns `202` with the original delivery ID
and bounded retry guidance; it never starts an executor. A retry after terminal
commit returns the stored bounded result and identifies itself as an idempotent
replay. The original caller may still lose its HTTP response after the receiver
accepted the request, but retrying that caller request cannot create a second
test attempt during the 24-hour idempotency window.

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

The post-attempt direction has an equivalent durable handshake. Before leaving
the attempt executor, the worker transactionally records the attempt outcome,
updates the delivery summary, and stores a pending Jobs disposition: complete,
retry with an exact bounded delay, fail terminally, or cancel. It then asks Jobs
through `WorkerSDK`'s prepared-disposition contract to apply that disposition
under its current lease and conditionally marks the disposition applied in
AuthNZ.

If the process dies after AuthNZ commits but before Jobs changes state, the next
lease holder observes the pending disposition and applies it without performing
another HTTP request. If Jobs changed state but the final AuthNZ acknowledgement
was lost, the reconciler proves the matching Jobs state and marks the disposition
applied. Cancellation requests use the same repair path. This prevents a
cross-database crash from turning one recorded receiver failure into an extra
unscheduled network attempt or leaving canceled work runnable indefinitely.

### Delivery State Machine And Ownership

AuthNZ stores the operator-facing delivery state. Jobs stores lease, schedule,
retry, and job terminal state. Neither database is treated as a distributed
transaction participant.

| Delivery state | Writer | Meaning and allowed next states |
| --- | --- | --- |
| `pending` | event producer | Durable automatic/manual row exists but has no enqueue claim; next is `enqueue_claimed`, a terminal lifecycle state, or `dead` on expiry; test rows are never committed in this state |
| `enqueue_claimed` | reconciler | A claim token/expiry owns the enqueue handshake; next is `queued`, back to `pending` after safe recovery, or a terminal lifecycle state |
| `queued` | reconciler | One Jobs ID is attached; next is `processing`, `canceled`, `superseded`, or `dead` on expiry |
| `processing` | worker or synchronous test service | One append-only attempt row plus Jobs lease or test token owns execution; the executor still performs final pre-I/O checks; next is `succeeded`, `retry_wait` for Jobs work only, `dead`, or a terminal lifecycle state observed before I/O |
| `retry_wait` | worker or disposition reconciler | Attempt metadata and requested delay are durable; Jobs either has the pending retry disposition or owns the next availability time; next is `processing`, `canceled`, `superseded`, or `dead` |
| `succeeded` | worker, test service, or disposition reconciler | Receiver returned 2xx; terminal |
| `dead` | worker, test service, or reconciler | Nonretryable response, exhausted retries, interrupted test, or `delivery_expired`; terminal |
| `canceled` | control-plane service or worker | Disabled, rotated, or deleted before an HTTP request began; terminal with a specific reason code |
| `superseded` | control-plane service or worker | Delivery configuration no longer matches and no HTTP request began; terminal |

The producer and enqueue reconciler update AuthNZ only. The worker conditionally
inserts an attempt and marks `processing` with Jobs ID, lease ID, attempt ID, and
monotonic sequence before I/O. The test service uses its test-attempt identity.
A retryable Jobs failure records bounded attempt metadata, the exact requested
delay, and `retry_wait`; Jobs calculates no second exponential delay. On a later
scheduled acquisition, the same Jobs ID and delivery ID return to `processing`
with a new attempt row and higher sequence. A test failure always transitions
directly to `dead` with its HTTP/network reason and no retry.

Lifecycle cancellation uses conditional updates and asks Jobs to cancel queued
or retrying work through a durable pending disposition. It cannot convert an
already-started HTTP request into a claim that nothing was sent. If disable,
rotate, update, or delete wins before the worker's final pre-I/O compare-and-set,
the worker records the matching terminal reason and sends nothing. If the worker
has already crossed that boundary, it records the real attempt result. A
successful response remains `succeeded` with an audit/metadata flag such as
`completed_after_config_change`; it is not rewritten as canceled.

Terminal AuthNZ states are monotonic. Recovery code may repair a stale
nonterminal mirror from the authoritative Jobs record, but cannot overwrite a
terminal delivery with an older Jobs observation. A missing Jobs row for a
claimed/queued delivery is reconciled through the same idempotency key; a Jobs
terminal state without an AuthNZ terminal state is repaired idempotently from
bounded job result metadata. A stale processing attempt is closed as
`outcome_unknown` before a new attempt starts; it is never deleted or rewritten
as an unsent attempt.

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
timer. The implementation extends the current Jobs operations and `WorkerSDK`
contracts rather than bypassing them. Jobs gains supported per-worker/per-job
controls for:

- a fail-closed acquisition guard and exact no-attempt deferral that releases or
  reschedules infrastructure-blocked work without incrementing HTTP retry or
  quarantine counters;
- observable lease-renewal state plus an `ensure_lease_horizon` operation so a
  handler cannot reserve an attempt unless its lease extends beyond the maximum
  HTTP timeout and terminal-commit margin;
- a typed disposition or retry signal carrying an exact retry-delay override so
  the webhook schedule is not multiplied by a second generic exponential
  calculation;
- a validated quarantine threshold captured for this job/worker above the
  webhook maximum attempts;
- idempotent recovery of an already-recorded disposition under a replacement
  lease without invoking the webhook handler's network path.

Webhook handling uses one prepared-disposition return contract. After AuthNZ
commits `complete`, `retry`, `fail`, `cancel`, or no-attempt `defer`, the handler
returns that typed disposition to `WorkerSDK`; the SDK applies it under the
current lease instead of also running its default success/failure finalizer. A
bounded post-apply acknowledgement marks the AuthNZ disposition applied, and a
lost acknowledgement is repaired by the reconciler. Direct handler finalization
plus default SDK completion is forbidden and tested.

The webhook worker refuses queue acquisition when mode, key-ring, or database
preflight is unavailable. If one of those conditions fails after acquisition but
before attempt reservation, it uses the bounded no-attempt deferral; expiry still
terminates the delivery after 72 hours. Acquisition-guard exceptions fail closed
for this worker. Lease renewal runs during I/O, and renewal loss is visible to
the handler. Before crossing the reservation/I/O boundary, the worker proves a
remaining lease horizon of at least the configured request timeout plus a
30-second commit margin. The same pre-I/O transaction requires that much time to
remain before delivery expiry; otherwise it commits
`dead:delivery_expired` without reserving an attempt. Lease-horizon failure sends
nothing and consumes no attempt slot.

Lease loss after the boundary remains an unavoidable at-least-once ambiguity.
The stale attempt becomes `outcome_unknown`, counts against the hard attempt
budget, and a late worker cannot commit over a replacement lease/attempt token.

A replacement lease that sees a still-`processing` attempt does not start
another request. It uses no-attempt deferral until that attempt's deterministic
stale time (`started_at + timeout_seconds + 90 seconds`). Only then may recovery
close the prior slot as `outcome_unknown` and recheck the hard budget. If budget
remains, recovery records the exact next scheduled retry disposition; Jobs
applies that disposition, and only a later scheduled acquisition may reserve the
new slot. Recovery never jumps directly from stale detection to HTTP I/O. This
prevents ordinary lease turnover from creating overlapping or unscheduled
requests while preserving recovery after a dead worker.

For this worker, `max_retries=3`, producing one initial attempt plus three
retries. The supported quarantine threshold is set above four attempts so normal
retryable HTTP failures do not hit the global poison-message default of two.
AuthNZ stores each already-decided retry delay with the attempt, so disposition
recovery after a crash reuses the original delay and does not recompute policy.

Append-only attempt history also enforces a hard four-network-attempt safety cap,
counting `outcome_unknown` attempts caused by lease or process loss. A lease
recovery may apply a recorded pending disposition without consuming a new
network attempt, but it may not send a fifth request merely because Jobs did not
increment `retry_count` before the earlier process died. Reaching this cap marks
the delivery `dead:attempt_budget_exhausted` and terminally reconciles Jobs. This
is an I/O safety ceiling; Jobs remains the only authority that schedules retries.

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
Current `dev` already provides peer-verified one-hop transport in
`tldw_Server_API/app/core/Security/http_hop.py`; webhook delivery extends that
primitive with a status-only response mode instead of introducing a second DNS,
socket, or TLS implementation.

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
- Status-only mode validates bounded response headers, extracts at most the
  allowed `Retry-After` value, closes the response stream without buffering a
  body, and does not expose ordinary headers to the webhook service. The helper
  returns status, latency, and bounded parsed `Retry-After` only.
- Logs receive webhook ID, target hostname, status/reason code, attempt, and
  latency, never the full URL.

The lower-level one-hop helper remains reusable public infrastructure, but this
task does not claim to migrate every existing outbound caller to status-only
mode.

## Feature Modes And Degraded States

`TLDW_ADMIN_WEBHOOKS_MODE` has three values:

- `off` (default): canonical delivery and mutations are unavailable; platform
  admins can read sanitized status.
- `migrate`: schema/import status and migration tooling are available; CRUD and
  delivery remain unavailable.
- `on`: canonical API, producers, reconciler, and worker are enabled if preflight
  succeeds.

During the first two stacked upstream PRs, legacy runtime behavior may remain
behind an explicit compatibility mode so the repository is not released in a
partially migrated state. PR 3 removes that temporary compatibility path and
performs final route activation. The final state always has one mounted
canonical router.

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
terminal state is committed. Eligible registration tombstones are then purged
under the admission-bound rules above.

Retention removes expired idempotency records, terminal delivery rows, orphaned
events, and expired migration backups through explicit bounded batches. It never
deletes active or nonterminal work.

Metrics include registrations by state, admission-limit denials, events created,
enqueue claims and recoveries, Jobs enqueue failures, deliveries by state/reason/
status class,
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
- automatic ETag handling and one fresh idempotency key per side-effecting
  create/rotate/test/redelivery command, reused only for that command's transport
  retries;
- effective registration limits and actionable limit/degraded states;
- a changed-configuration warning and explicit confirmation for redelivery to a
  newer configuration;
- a warning that in-flight HTTP requests cannot be recalled.

The page does not expose arbitrary payload editing, custom headers, wildcard
events, or receiver response bodies.

An in-progress command keeps its key and normalized request in memory; automatic
retry reuses both. The UI does not persist a secret-replay-capable key in local
or session storage. If navigation/reload loses a create/rotate response, the UI
refetches the inactive registration and directs the operator through a new
rotation rather than claiming the original secret is recoverable. A `412`
response reloads the current representation/ETag and requires fresh operator
review; it is never auto-retried against changed configuration.

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
- resource revisions, conditional mutations, and no-op PATCH behavior;
- dedicated key ring, encrypted target URLs/event bodies, server-generated
  secret lifecycle, and bounded idempotent replay;
- event catalog and status surface;
- dry-run/import/readback/sanitization tooling for legacy JSON and DB rows;
- route uniqueness and temporary compatibility-mode controls;
- admin UI control-plane contract updates where needed.

The feature remains non-releaseable in canonical `on` mode until PR 3 lands.

### PR 2: Delivery Substrate And Recovery

- event, delivery, and append-only attempt repositories;
- event expansion and reconciler enqueue claims;
- cross-database enqueue, disposition, and cancellation recovery handshakes;
- supported Jobs exact-delay, disposition-recovery, and quarantine-threshold
  extension points, plus fail-closed acquisition, no-attempt deferral, and
  observable lease-horizon enforcement;
- webhook Jobs worker, central status-only egress helper, signing, retry, expiry,
  retention, metrics, and health;
- test and manual redelivery behavior;
- delivery-history API and operational service contracts.

This PR remains behind the disabled canonical mode and proves the data plane with
synthetic events. It does not yet connect user or incident mutations.

### PR 3: Durable Producers And Activation

- durable user and incident producers;
- source-identity deduplication and automatic delivery expansion;
- final canonical router mounting and removal of legacy webhook handlers;
- delivery-history and operational admin UI;
- removal of temporary legacy compatibility routing;
- receiver and operator documentation.

No release may enable canonical mode until all three PRs are present and the PR 3
activation gate passes. This split keeps current Jobs and egress extension work
reviewable instead of combining it with six transactional producers and final
route cutover.

## Verification Gates

### PR 1 Gates

- temporary mode-routing tests prove canonical and legacy handlers are never
  mounted for the same method/path in one runtime;
- platform-admin allow/deny tests and complete mutation audit tests;
- OpenAPI request/response and stable error-code tests;
- revision ETag, stale `If-Match`, no-op PATCH, and concurrent mutation tests;
- create/active registration bounds, configuration validation, degraded
  over-limit behavior, set-based fanout, and tombstone-purge tests;
- SQLite fresh install and upgrades from pre-080, 080, and 082 states;
- PostgreSQL fresh install and representative legacy upgrade;
- durable SQLite write/commit regression tests;
- JSON and legacy-table dry run, import, conflict, crash-point, readback,
  encrypted-backup/key-destruction, sanitization, unrelated-file-change, and
  rerun tests;
- dedicated key absence, primary/previous rotation, legacy re-encryption, and
  runtime-fallback rejection tests;
- create/rotate idempotent replay, route scoping, superseded-secret replay, and
  same-key/different-request conflict tests;
- replay-before-precondition ordering and concurrent idempotency-claim tests;
- encrypted target, secret, and replay-material at-rest assertions, including
  cross-row/cross-purpose envelope substitution rejection;
- event catalog, wildcard rejection, timeout, payload-size, and static-route
  ordering tests;
- admin UI create/rotate one-time-secret tests, same-command transport retry,
  reload-to-new-rotation recovery, stale-ETag review, and redacted list/get
  fixtures;
- Bandit on touched Python and generated OpenAPI review.

### PR 2 Gates

- one automatic delivery per matching registration;
- synthetic event-body encryption/decryption, key rotation, and 64 KiB boundary
  assertions;
- all enqueue, post-attempt disposition, and cancellation crash points for
  SQLite/SQLite, SQLite/PostgreSQL, PostgreSQL/SQLite, and
  PostgreSQL/PostgreSQL;
- append-only attempt sequencing, stale-attempt `outcome_unknown`, and no extra
  HTTP request while recovering a pending Jobs disposition;
- hard four-network-attempt enforcement across repeated lease-loss recovery;
- Jobs exact retry schedule, max attempts, quarantine behavior, lease recovery,
  cancellation, idempotent enqueue, fail-closed acquisition, no-attempt
  infrastructure deferral, and pre-I/O lease-horizon enforcement;
- prepared-disposition application for complete/retry/fail/cancel/defer with no
  default-SDK double finalization and lost-acknowledgement repair;
- lease-renewal loss before/after the I/O boundary, replacement-lease deferral
  before deterministic staleness, exact Jobs-scheduled disposition afterward,
  no overlapping normal attempt, and token-rejected late completion;
- pre-I/O expiry-horizon rejection proving no request can begin too late to
  finish and commit within the 72-hour window;
- configuration supersession, disable, rotation, soft delete, manual redelivery,
  in-flight race, 72-hour expiry, and 30-day retention;
- deterministic body and published signature test vectors;
- delivery-time SSRF, DNS change, proxy, redirect, TLS, timeout, no-buffer, and
  URL-redaction tests;
- retry classification and bounded `Retry-After` parsing;
- synchronous-test direct-processing, interrupted-attempt recovery, one-attempt,
  stale-screen precondition, in-progress/terminal idempotent replay, and no-Jobs/
  no-retry tests;
- worker/reconciler health and backlog preflight tests;
- Bandit on touched Python.

### PR 3 Gates

- transactional user producers and file-marker incident crash recovery;
- key rotation/readback for pending encrypted file markers and key-loss-before-
  source-commit tests proving no domain write without its event;
- producer source-identity uniqueness and duplicate reconciliation;
- payload privacy, encryption, and 64 KiB enforcement for all six event types;
- exactly one final canonical route per method/path and no legacy duplicate;
- local controlled receiver end-to-end tests for signatures, duplicates, retry,
  test headers, automatic producers, and manual redelivery;
- complete admin UI test, typecheck, lint, and production build;
- complete SQLite and PostgreSQL activation/preflight matrix;
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
  key ring; exact canonical event bodies and secret-bearing replay material are
  encrypted too. Create and rotate reveal server-generated secrets only through
  eligible version-bound idempotent responses.
- Registration mutations are revision-conditional and stale or superseded
  privileged actions fail closed.
- Six documented event producers create immutable, deduplicated events and
  versioned automatic deliveries.
- Jobs is the only automatic retry authority and cross-database enqueue is
  crash-recoverable. Post-attempt dispositions and cancellations are also
  recoverable without an unintended extra HTTP attempt.
- Append-only attempt rows preserve ambiguous outcomes and bounded retry
  evidence without persisting receiver content.
- Delivery uses the published protocol and central egress-safe, status-only HTTP
  path with no full URL, payload, signature, or receiver body in logs/history.
- Degraded modes, retention, health, manual redelivery, expiry, and rollback
  boundaries are explicit and tested.
- PR 1, PR 2, and PR 3 pass their defined SQLite, PostgreSQL, security, admin UI,
  Bandit, and end-to-end gates before canonical mode can be enabled.
