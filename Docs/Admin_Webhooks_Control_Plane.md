# Admin Webhooks Control Plane

## Scope And Release Boundary

This document covers the canonical admin-webhook control plane, durable delivery
runtime, and activation status. The subsystem remains default-off until the
operator completes migration and the two-phase activation procedure.

## Access And Authorization

All routes are below `/api/v1/admin` and require an authenticated platform-admin
principal. Mutations additionally require a user-backed principal with a
positive user ID. Service principals and any other principal without a positive
numeric user ID cannot mutate registrations.

The admin Web UI is the normal management surface. Direct API clients must use
the deployment's existing authenticated admin transport and must preserve the
conditional and idempotency headers described below.

## Environment Reference

All values are process startup configuration. Every application process must
receive the same values before traffic is admitted.

| Variable | Default | Accepted values and effect |
| --- | --- | --- |
| `TLDW_ADMIN_WEBHOOKS_MODE` | `off` | `off`, `migrate`, or `on`. CRUD is unavailable in `off`; import tooling is used in `migrate`; canonical CRUD is available in `on` only after migration is complete. |
| `TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT` | `false` | Deprecated deployment input. Omit it or use exact `false`; exact `true` is a sanitized startup configuration error. It never selects routes. |
| `TLDW_ADMIN_WEBHOOK_REGISTRATION_LIMIT` | `100` | Integer 1-1,000. Bounds all non-deleted registrations. |
| `TLDW_ADMIN_WEBHOOK_ACTIVE_LIMIT` | `25` | Integer 1-1,000 and no greater than the registration limit. Bounds active registrations. |
| `TLDW_ADMIN_WEBHOOKS_ALLOW_HTTP_DEV` | `false` | Exact `true` or `false`. Allows `http` targets only in a validated non-production environment. Enabling it in production is a startup error. |
| `TLDW_ADMIN_WEBHOOK_ROLLBACK_WINDOW_DAYS` | `7` | Integer 1-30. Controls how long the encrypted legacy-file backup and separate rollback key may remain usable. |
| `TLDW_ADMIN_WEBHOOK_ACTIVATION_MAX_BACKLOG_AGE_SECONDS` | `300` | Integer 1-86,400. Maximum oldest nonterminal delivery age accepted by the live activation check. |
| `TLDW_ADMIN_WEBHOOK_KEYS_JSON` | none | Strict JSON object mapping key IDs to canonical base64 encodings of exactly 32 random bytes. Required for migration and protected operations. Duplicate IDs, invalid JSON/base64, and wrong key sizes fail closed. |
| `TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID` | none | Key ID present in `TLDW_ADMIN_WEBHOOK_KEYS_JSON`; 1-64 characters from `[A-Za-z0-9._-]`. New protected values use this key. |

The idempotency retention period is fixed at 86,400 seconds and has no
environment override.

### Egress Policy Composition

Target creation and replacement apply the central platform-webhook policy after
the stricter canonical URL parser. The parser rejects user-info, fragments,
control characters, backslashes, malformed hosts or ports, and non-HTTPS URLs
unless the non-production HTTP override is enabled.

The platform adapter unions these comma-separated host allowlists:

- `EGRESS_ALLOWLIST`
- `WORKFLOWS_EGRESS_ALLOWLIST`
- `WORKFLOWS_WEBHOOK_ALLOWLIST`

It separately unions these denylists:

- `EGRESS_DENYLIST`
- `WORKFLOWS_EGRESS_DENYLIST`
- `WORKFLOWS_WEBHOOK_DENYLIST`

A match in any denylist wins. When `WORKFLOWS_EGRESS_PROFILE=strict`, at least
one allowlist entry must match. If the profile is unset, production defaults to
`strict` and non-production defaults to `permissive`. In permissive/custom mode,
a non-empty combined allowlist is still enforced. Host entries match the exact
host and its subdomains.

The shared policy also uses:

| Variable | Default | Effect |
| --- | --- | --- |
| `WORKFLOWS_EGRESS_PROFILE` | `strict` in production; `permissive` otherwise | `strict`, `permissive`, or `custom` host-policy behavior. |
| `WORKFLOWS_EGRESS_BLOCK_PRIVATE` | `true` | Blocks private and special-purpose destinations. Keep enabled. |
| `WORKFLOWS_EGRESS_ALLOWED_PORTS` | `80,443,8080` | Comma-separated allowed destination ports. Canonical HTTPS is still required by default. |
| `WORKFLOWS_EGRESS_DNS_MAX_OUTSTANDING` | `64` | Positive bound on concurrent resolver work. |
| `WORKFLOWS_EGRESS_DNS_SLOT_WAIT_SECONDS` | `0.05` | Non-negative bounded wait for a resolver slot. |

Registration-time validation does not guarantee future delivery. Delivery
repeats DNS and egress checks for every attempt.

## Canonical Routing

Status and every canonical route are mounted in every mode. `off` returns the
bounded disabled error for blocked operations, `migrate` returns the bounded
migration-pending error, and `on` permits operations only when their canonical
readiness rules pass. Status retains `route_selection=canonical` for one
response-compatibility cycle; there is no selectable legacy route family.

Clients must never treat a 404, network failure, or malformed status response
as permission to use a different API. A multi-process deployment is not
switched by changing one node: drain/restart and verify every process reports
the intended mode.

## Canonical API

| Method and path | Required headers | Result |
| --- | --- | --- |
| `GET /api/v1/admin/webhooks/status` | normal admin auth | Mode, canonical route literal, schema/key/migration state, limits, delivery runtime health, backlog, and rollback eligibility. |
| `GET /api/v1/admin/webhooks/catalog` | normal admin auth | Immutable event catalog (`2026-07-01`) and effective limits. |
| `GET /api/v1/admin/webhooks?limit=50&offset=0` | normal admin auth | Redacted registrations ordered by numeric ID descending. `limit` is 1-100; `offset` is 0-1,000. |
| `POST /api/v1/admin/webhooks` | `Idempotency-Key` | Creates an inactive registration and returns its signing secret once. |
| `GET /api/v1/admin/webhooks/{id}` | normal admin auth | One redacted registration plus its strong `ETag`. |
| `PATCH /api/v1/admin/webhooks/{id}` | `If-Match` | Updates one or more non-null fields: `description`, full replacement `url`, `event_types`, `active`, or `timeout_seconds`. |
| `DELETE /api/v1/admin/webhooks/{id}` | `If-Match` | Soft-deletes a registration. |
| `POST /api/v1/admin/webhooks/{id}/rotate-secret` | `If-Match`, `Idempotency-Key` | Rotates an inactive registration and returns the new signing secret once. |
| `POST /api/v1/admin/webhooks/{id}/test` | `If-Match`, `Idempotency-Key` | Persists one synthetic test delivery command. |
| `GET /api/v1/admin/webhooks/{id}/deliveries` | normal admin auth | Sanitized durable delivery and attempt history. |
| `POST /api/v1/admin/webhooks/{id}/deliveries/{delivery_id}/redeliver` | `If-Match`, `Idempotency-Key` | Persists one eligible manual redelivery command. |
| `POST /api/v1/admin/incidents/{id}/notify-webhooks` | `Idempotency-Key` | Persists one durable incident notification event. |
| `POST /api/v1/admin/incidents/{id}/notify` | `Idempotency-Key` | Separately persists and sends one at-most-once stakeholder-email command. |

The catalog currently contains:

- `user.created`
- `user.deleted`
- `incident.created`
- `incident.updated`
- `incident.resolved`
- `incident.notify`

Wildcards and unknown events are rejected. Creation is always inactive.
Timeouts are 1-30 seconds and default to 10.

The public request envelope, exact event-specific payload schemas, signature
algorithm, deduplication rules, and response behavior are defined in the
[Admin Webhooks Receiver Guide](Admin_Webhooks_Receiver_Guide.md). Treat that
guide and the catalog API version as one compatibility contract.

## Conditional Requests And Idempotency

Registration ETags are strong and have this exact shape:

```text
"admin-webhook-<positive-id>-r<positive-revision>"
```

Fetch the current registration immediately before PATCH, delete, or secret
rotation and send the returned ETag verbatim in `If-Match`. Missing ETags return
`428 precondition_required`; malformed, wrong-resource, or stale ETags return
`412 precondition_failed`. After either response, fetch the current
registration and require fresh operator review. Do not auto-retry against a new
revision.

Create and rotate idempotency keys are 16-255 characters from
`[A-Za-z0-9._:-]`. Generate at least 16 random bytes; the admin UI encodes those
bytes as 32 lowercase hexadecimal characters. Reuse a key only for an exact
retry of the same command after an ambiguous transport failure. A changed body,
resource, operation, actor, or conditional version is a different command and
requires a new key.

Exact create/rotate replays can return the original secret only while the
recorded secret version remains current, the registration is not deleted, and
the 24-hour replay record remains. A later rotation or deletion returns
`409 idempotency_result_superseded` instead of revealing an obsolete secret.

## One-Time Secrets And Redaction

Create and rotate generate a server-side secret with this shape:

```text
whsec_<64 lowercase hexadecimal characters>
```

The full value is returned only in the successful create/rotate response or an
eligible exact replay. It is never returned by list, get, PATCH, status, or
audit. Secret-bearing responses set `Cache-Control: no-store` and
`Pragma: no-cache`.

Store the secret in the receiver's secret manager before acknowledging the UI
dialog. The admin UI keeps both the secret and same-command retry key in memory
only and clears them on page exit/restoration. If the response is lost and the
same in-memory retry is no longer available, fetch the inactive registration
and perform a new rotation. There is no secret-recovery endpoint.

The full destination path and query are encrypted and non-retrievable. Normal
responses show only `target_display` (origin) and `target_hostname`. Metadata
edits omit `url`. Destination replacement starts with a blank field and requires
the complete new URL; never submit `target_display` as though it were the full
destination.

## API Examples

These examples use reserved `.example` hosts and fake credentials. Run direct
secret-bearing requests only from an approved operator workstation whose
terminal output is not recorded.

### Create An Inactive Registration

```bash
API_BASE=https://admin-api.example/api/v1/admin
ADMIN_TOKEN=fake-admin-token
IDEMPOTENCY_KEY="$(openssl rand -hex 16)"

curl --fail-with-body --silent --show-error --include \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: ${IDEMPOTENCY_KEY}" \
  --data '{"url":"https://receiver.example/hooks/private","event_types":["incident.created"],"description":"Primary incident receiver","timeout_seconds":10}' \
  "${API_BASE}/webhooks"
```

The response is `201`, includes an ETag such as
`"admin-webhook-41-r1"`, and contains a one-time fake-shaped value such as
`whsec_0000000000000000000000000000000000000000000000000000000000000000`.
Store the actual returned value immediately.

### Rotate An Inactive Registration

First fetch `/webhooks/41` and copy its current response ETag. Then generate a
new command key:

```bash
IDEMPOTENCY_KEY="$(openssl rand -hex 16)"

curl --fail-with-body --silent --show-error --include \
  -X POST \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Idempotency-Key: ${IDEMPOTENCY_KEY}" \
  -H 'If-Match: "admin-webhook-41-r1"' \
  "${API_BASE}/webhooks/41/rotate-secret"
```

The registration must be inactive. Update the receiver with the newly returned
secret before any later activation attempt.

## Error Contract

Expected canonical failures use a bounded, redacted body:

```json
{
  "error": {
    "code": "admin_webhook_validation_failed",
    "message": "Webhook request validation failed",
    "request_id": "4aa1324c-7fb7-49cf-9058-ce0df25d5932"
  }
}
```

The same sanitized request ID is returned in `X-Request-ID`, and expected error
responses set `Cache-Control: no-store`. Submitted URLs, query credentials,
secrets, conditional headers, idempotency keys, exception details, and Pydantic
validation details are never reflected.

Common statuses are:

| Status | Meaning |
| --- | --- |
| `401`, `403` | Authentication or platform-admin authorization failed. |
| `409` | Idempotency conflict/supersession, registration/active limit, required signing-secret rotation, or active registration blocked the command. |
| `412`, `428` | Stale/invalid or missing `If-Match`. |
| `422` | Bounded request, event, idempotency-key, or target validation failure. |
| `503` | Mode/migration/key/rotation/database/audit/delivery precondition is unavailable. |

Use the closed `error.code` and `request_id` for response handling and support
correlation. Do not parse human-readable message text.

Stakeholder email is not webhook delivery. Each recipient is durably claimed
before the provider call; a crash after that boundary is reported as `unknown`
and is never resent automatically. Only a literal provider result of `true` is
recorded as sent. Exact command replays are retained for at least 30 days.
When the 1,000-command store reaches capacity, only fully `sent`/`failed`
commands older than 30 days are eligible for pruning; `pending` and `sending`
commands are never pruned automatically. If no command is eligible, admission
fails closed until an operator resolves capacity. Never reuse an idempotency
key for a changed recipient list or message.

## Operator Checks

Before any canonical management session:

1. Read `/webhooks/status` and confirm every process reports the intended mode
   and route selection.
2. Confirm `schema_ready=true` and migration phase `complete` before mode `on`.
3. Confirm `key_state=available` and configured limits are not exceeded.
4. Treat `delivery_capability_ready=false` as a hard live-activation stop.
5. Run `tldw-admin-webhooks activation-check --phase predeploy` while every
   node remains in `migrate` before deploying an `on` canary.
6. Run `tldw-admin-webhooks activation-check --phase live` only against the
   no-traffic canary before enabling a registration or admitting producer
   traffic.
7. Use the migration and key-rotation runbooks for all offline state changes.

See [Admin Webhooks Migration Runbook](Admin_Webhooks_Migration_Runbook.md) and
[Admin Webhooks Key Rotation Runbook](Admin_Webhooks_Key_Rotation_Runbook.md).
The complete private-beta rollout, controlled probes, dead-delivery triage, and
safe-disable procedure are in the
[Admin Webhooks Delivery Runbook](Admin_Webhooks_Delivery_Runbook.md).
