# Admin Webhooks Receiver Guide

## Scope

This guide defines the public receiver contract for canonical tldw admin
webhooks API version `2026-07-01`. It covers request authentication, the six
subscription events, test deliveries, retry and ordering behavior, and safe
receiver operations.

The sender performs HTTPS `POST` requests. Receiver response bodies are ignored
and are not stored by tldw. A receiver should return a `2xx` response only after
it has durably accepted the event or completed an idempotent duplicate check.

## Request Contract

Every request includes these webhook headers:

```text
Content-Type: application/json
X-TLDW-Webhook-Event: incident.updated
X-TLDW-Webhook-Event-Id: 00000000-0000-4000-8000-000000000001
X-TLDW-Webhook-Delivery-Id: 00000000-0000-4000-8000-000000000002
X-TLDW-Webhook-Timestamp: 1787443200
X-TLDW-Webhook-Secret-Version: 1
X-TLDW-Webhook-Signature: v1=<64 lowercase hexadecimal characters>
```

Test requests also include:

```text
X-TLDW-Webhook-Test: true
```

Header names are case-insensitive under HTTP. Header values are not. Event and
delivery IDs are UUIDv4 values. Secret versions are positive integers. The
timestamp is Unix seconds for this network attempt.

Normal HTTP transport may add standard headers such as `Host` and
`Content-Length`. Do not include those headers in signature input.

## Verify Before Processing

Each registration has a server-generated signing secret shaped like
`whsec_<64 lowercase hexadecimal characters>`. Store the complete value in a
secret manager. The `whsec_` prefix is part of the HMAC key.

For each request:

1. Read and retain the exact raw request body bytes. Do not parse and
   reserialize the body before verification.
2. Reject a missing, malformed, or stale timestamp. A five-minute past/future
   tolerance is recommended.
3. Build the signed bytes as the ASCII timestamp, one literal period, and the
   exact raw body bytes.
4. Compute HMAC-SHA256 with the complete registration secret.
5. Compare `v1=<lowercase hex digest>` to the signature header in constant
   time.
6. Only after signature verification, parse the JSON and validate the envelope,
   API version, event type, and event ID against the headers.
7. Apply the receiver's durable deduplication transaction before business side
   effects.

Python standard-library example:

```python
from __future__ import annotations

import hashlib
import hmac
import time


def verify_tldw_signature(
    *,
    body: bytes,
    timestamp: str,
    signature: str,
    secret: str,
    now: int | None = None,
    tolerance_seconds: int = 300,
) -> bool:
    if not timestamp.isascii() or not timestamp.isdigit():
        return False
    attempt_time = int(timestamp)
    current_time = int(time.time()) if now is None else now
    if abs(current_time - attempt_time) > tolerance_seconds:
        return False
    signed = timestamp.encode("ascii") + b"." + body
    expected = "v1=" + hmac.new(
        secret.encode("ascii"),
        signed,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(signature, expected)
```

Do not log the raw body, full destination URL, signature, signing secret, or
headers that may become replay evidence. Log bounded event/delivery IDs,
validated event type, secret version, and the receiver's own outcome code.

## Event Envelope

Every body is deterministic compact UTF-8 JSON with this closed envelope:

```json
{
  "api_version": "2026-07-01",
  "created_at": "2026-08-31T12:05:00Z",
  "data": {},
  "id": "00000000-0000-4000-8000-000000000001",
  "type": "incident.updated"
}
```

The wire representation uses sorted keys and no insignificant whitespace. The
decoded object always contains exactly:

| Field | Contract |
| --- | --- |
| `id` | UUIDv4 equal to `X-TLDW-Webhook-Event-Id`. |
| `type` | Event name equal to `X-TLDW-Webhook-Event`. |
| `api_version` | Exact string `2026-07-01`. Reject an unsupported version. |
| `created_at` | Canonical UTC timestamp ending in `Z`; this is event creation, not attempt time. |
| `data` | Closed event-specific object described below. |

The complete encoded body is at most 65,536 bytes and is never truncated.
Receivers should still enforce their own bounded request-body limit.

All public timestamps use canonical UTC `YYYY-MM-DDTHH:MM:SS[.ffffff]Z`
format. Event-specific objects do not gain fields automatically. A future field
or event requires a future versioned contract; treat unsupported versions as a
controlled receiver error rather than silently ignoring unknown semantics.

## User Events

### `user.created`

```json
{
  "created_at": "2026-08-31T12:00:00Z",
  "resource_version": "2026-08-31T12:05:00Z",
  "status": "active",
  "updated_at": "2026-08-31T12:05:00Z",
  "user_id": 7
}
```

`user_id` is a positive integer. `status` is `active` or `inactive` and reflects
the persisted lifecycle state created by the source transaction.
`resource_version` is the canonical profile-version timestamp.

### `user.deleted`

```json
{
  "created_at": "2026-08-31T12:00:00Z",
  "resource_version": "2026-08-31T12:10:00Z",
  "status": "inactive",
  "updated_at": "2026-08-31T12:10:00Z",
  "user_id": 7
}
```

Deletion is a durable lifecycle deactivation. `status` is always `inactive`.

User events never include username, email, password material, sessions, API
keys, profile text, organization/invitation data, or billing data.

## Incident Events

Incident data uses this common closed shape:

```json
{
  "created_at": "2026-08-31T12:00:00Z",
  "incident_id": "inc_123",
  "resolved_at": null,
  "resource_version": 2,
  "severity": "critical",
  "state": "investigating",
  "updated_at": "2026-08-31T12:05:00Z"
}
```

The constraints are:

- `incident_id`: non-empty string, at most 255 characters;
- `resource_version`: positive integer that increases for effective incident
  mutations;
- `state`: `open`, `investigating`, `mitigating`, or `resolved`;
- `severity`: `low`, `medium`, `high`, or `critical`;
- `resolved_at`: canonical UTC timestamp only when state is `resolved`, otherwise
  `null`.

### `incident.created`

Published with the first persisted incident version. Receivers should create or
upsert their projection by `(incident_id, resource_version)`.

### `incident.updated`

Published for every effective non-resolution mutation, including a timeline
append. A no-op update emits nothing. Compare `resource_version` before applying
the event so an older out-of-order event cannot overwrite newer state.

### `incident.resolved`

Published on transition into `resolved`. `state` is `resolved`, and
`resolved_at` is a canonical UTC timestamp no later than `updated_at`.

### `incident.notify`

This explicit operator command adds exactly one field to the common incident
shape:

```json
{
  "created_at": "2026-08-31T12:00:00Z",
  "incident_id": "inc_123",
  "narrative": "Mitigation is in progress.",
  "resolved_at": null,
  "resource_version": 2,
  "severity": "high",
  "state": "investigating",
  "updated_at": "2026-08-31T12:05:00Z"
}
```

`narrative` is either `null` or the exact operator-reviewed string with a
maximum length of 4,096 characters. It is not trimmed or rewritten. This is the
only routine incident event that may contain operator-authored free text.

Incident events never include title, summary, tags, timeline messages,
evidence, assignee, root cause, impact, runbook URL, action items, or recipient
email addresses.

## Test Deliveries

`webhook.test` is reserved and cannot be selected as a subscription event. Its
body uses the normal envelope and this data object:

```json
{
  "test": true,
  "webhook_id": 41
}
```

The request includes `X-TLDW-Webhook-Test: true`. It is one synchronous,
persisted attempt with no Jobs row and no automatic retry. It may be sent while
the registration is inactive. Respond as for a normal signed request, but keep
test effects separate from production business processing.

An interrupted test may be recorded by tldw as `outcome_unknown` or
`test_attempt_interrupted`; tldw does not send an implicit replacement request.

## Deduplication And Ordering

Delivery is at-least-once and unordered.

Use the identifiers for distinct purposes:

| Identifier | Receiver use |
| --- | --- |
| Event ID | Idempotency key for business effects. One event can have multiple delivery rows. |
| Delivery ID | Diagnostics and deduplication of repeated attempts for one delivery row. |
| `(event ID, delivery ID)` | Exact receiver identity for one logical delivery. Multiple network attempts may share it. |
| Aggregate `resource_version` | Reject stale out-of-order user/incident projection updates. |

Automatic retries reuse both event ID and delivery ID and reuse the exact body
bytes. Their timestamp and signature are regenerated for each network attempt.
A receiver may therefore observe the same `(event ID, delivery ID)` more than
once after an ambiguous timeout or process/lease loss.

Manual redelivery creates a new delivery ID for the historical event ID and
uses the current active registration configuration. Deduplicating only by
delivery ID can repeat a business side effect; deduplicate business effects by
event ID. Keep delivery IDs in diagnostics so operators can distinguish the
original delivery from an explicit redelivery.

Events for different sources or aggregates have no global ordering guarantee.
Commit the deduplication record and business effect atomically where the
receiver's persistence model permits it.

## Responses And Retry Schedule

Any `2xx` response is success. The automatic attempt schedule is:

| Attempt | Timing |
| --- | --- |
| 1 | Initial attempt |
| 2 | At least 1 minute after attempt 1 |
| 3 | At least 5 minutes after attempt 2 |
| 4 | At least 30 minutes after attempt 3 |

Network failures, timeouts, HTTP 408, HTTP 429, and HTTP 5xx are retryable.
Redirects and every other 4xx are terminal. Redirects are never followed.

For 429 and 503 only, a valid `Retry-After` delta or HTTP date may lengthen the
fixed delay. tldw clamps that value to 1-1,800 seconds. It never shortens the
fixed schedule. Automatic work becomes terminal after 72 hours or four network
attempts, including attempts whose outcome became unknown.

Choose responses deliberately:

- return `2xx` after durable acceptance or an idempotent duplicate;
- return 400-series terminal errors for requests that retry cannot repair;
- return 429 when overloaded and optionally include bounded `Retry-After`;
- return 500-series errors for temporary server failures;
- do not redirect to another receiver URL.

## Secret Rotation

`X-TLDW-Webhook-Secret-Version` identifies the registration signing-secret
version used for the attempt. Store the new one-time secret before re-enabling a
rotated registration. Keep an explicitly bounded overlap only if your receiver
deployment requires it; select the candidate secret by version. Rotation
cancels work that has not crossed tldw's final pre-I/O boundary, but it cannot
recall an already-running request. Keep the old version through at least the
registration's maximum in-flight timeout and until delivery history shows no
unresolved old-version attempt, then remove it under the receiver's secret
retention procedure.

Never retrieve a secret from list/history APIs or logs. tldw exposes a new
secret only in the eligible create/rotate response and does not provide a
secret-recovery endpoint.

## Receiver Checklist

- HTTPS endpoint with a valid certificate and no redirect dependency.
- Exact raw-body capture with a body-size limit.
- Five-minute timestamp freshness check.
- HMAC-SHA256 verification with constant-time comparison.
- Closed API-version and event-schema validation.
- Durable event-ID deduplication in the same transaction as business effects.
- Delivery-ID diagnostics for retries and manual redelivery.
- Aggregate resource-version checks for out-of-order events.
- Bounded processing time below the registration timeout (1-30 seconds).
- Sanitized logs and metrics with no body, URL path/query, signature, or secret.
- Separate handling for `X-TLDW-Webhook-Test: true`.

Operators should use the canonical admin Webhooks UI for registration, test,
history, redelivery, disable, and secret rotation. See
[Admin Webhooks Delivery Runbook](Admin_Webhooks_Delivery_Runbook.md) for tldw
runtime triage and [Admin Webhooks Control Plane](Admin_Webhooks_Control_Plane.md)
for the management API contract.
