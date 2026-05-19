# Webhooks

Webhooks let the admin UI push event notifications to external systems in real time. When a subscribed event fires (e.g. user created, incident opened, key rotated), the platform sends an HTTP POST with a JSON payload to the configured URL.

---

## Creating a Webhook

1. Navigate to **Settings > Webhooks**.
2. Click **Create Webhook**.
3. Provide a **URL** (must be HTTPS in production).
4. Select one or more **event types** to subscribe to:
   - `user.created`, `user.deleted`, `user.role_changed`
   - `org.created`, `org.deleted`, `org.member_added`, `org.member_removed`
   - `incident.created`, `incident.resolved`, `incident.status_changed`
   - `api_key.created`, `api_key.revoked`
   - `policy.created`, `policy.updated`, `policy.deleted`
5. Optionally set a **description** for your own reference.
6. Click **Save**. The webhook secret is shown once -- copy it immediately.

---

## HMAC Signature Verification

Every delivery includes an `X-Webhook-Signature` header containing an HMAC-SHA256 hex digest computed over the raw request body using the webhook secret.

### Verification steps

```python
import hmac
import hashlib

def verify_signature(body: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

```typescript
import { createHmac, timingSafeEqual } from 'crypto';

function verify(body: Buffer, signature: string, secret: string): boolean {
  const expected = `sha256=${createHmac('sha256', secret).update(body).digest('hex')}`;
  return timingSafeEqual(Buffer.from(signature), Buffer.from(expected));
}
```

Always use a constant-time comparison to prevent timing attacks.

---

## Delivery Monitoring

Each webhook has a **Deliveries** tab showing recent attempts:

| Column         | Description                                       |
|----------------|---------------------------------------------------|
| Event          | The event type that triggered the delivery.       |
| Status         | HTTP response code or `timeout` / `error`.        |
| Latency        | Round-trip time in milliseconds.                  |
| Attempted At   | Timestamp of the delivery attempt.                |

Failed deliveries are retried with exponential backoff (1 min, 5 min, 30 min) up to 3 attempts. After exhaustion the delivery is marked as permanently failed.

### Inspecting a delivery

Click a delivery row to see:

- **Request headers** and **body** (JSON payload).
- **Response headers** and **body** from the receiver.
- **Error details** if the delivery failed.

---

## Testing a Webhook

1. Open the webhook detail page.
2. Click **Send Test Event**.
3. Choose an event type from the dropdown.
4. The platform sends a synthetic payload to your URL with the header `X-Webhook-Test: true`.
5. Check the delivery log to confirm the result.

Use this to validate your endpoint before subscribing to production events.

---

## Payload Format

All payloads share a common envelope:

```json
{
  "id": "evt_abc123",
  "type": "user.created",
  "created_at": "2026-03-27T12:00:00Z",
  "data": {
    "user_id": 42,
    "username": "alice"
  }
}
```

The `data` field varies by event type. Refer to the API reference for per-event schemas.

---

## Best Practices

- **Respond quickly.** Return a 2xx within 10 seconds. Process asynchronously if your handler is slow.
- **Idempotency.** Use the `id` field to deduplicate deliveries.
- **Secret rotation.** Rotate webhook secrets periodically via the admin UI and update your verification code.
- **Monitoring.** Set up alerts on repeated delivery failures to catch endpoint issues early.
