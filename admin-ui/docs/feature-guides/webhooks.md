# Webhooks

The Webhooks page manages the canonical outgoing-webhook control plane. The
current PR 1 release manages registrations and migration state only. It cannot
send webhook traffic, test a receiver, show canonical delivery history, or
activate a registration.

## Availability

The page reads `/api/v1/admin/webhooks/status` before selecting an API family.
It does not infer compatibility mode from a failed request.

| Status | UI behavior |
| --- | --- |
| Canonical mode `off` | Shows the disabled warning; canonical registrations cannot be loaded or changed. |
| Canonical mode `migrate` or incomplete migration | Shows migration state; canonical registrations cannot be loaded or changed. |
| Canonical mode `on`, migration complete | Loads the catalog and redacted registration list. PR 1 still reports delivery unavailable and blocks activation. |
| `route_selection=legacy` | Shows a prominent compatibility warning and uses only the typed legacy adapter. Canonical ETags and secret rotation are unavailable. |
| Status unavailable or malformed | Shows the bounded error and retry action; no automatic downgrade occurs. |

The page requires a platform-admin account. Canonical mutations also require a
user-backed admin principal.

## Create A Canonical Registration

1. Open **Webhooks** and confirm the operational alerts show the expected mode,
   completed migration, available key, and acceptable limits.
2. Select **Add webhook**.
3. Enter the complete destination URL. HTTPS is required unless the deployment
   explicitly allows HTTP in non-production.
4. Optionally enter a description and set a timeout from 1 to 30 seconds.
5. Select one or more events from the server-provided catalog.
6. Select **Create**.
7. Copy the generated signing secret into the receiver's secret manager.
8. Check the acknowledgement only after the secret is stored, then select
   **Done**.

Creation is always inactive. The secret is shown once and cannot be retrieved
from list/get responses. The page keeps it and the same-command retry key in
memory only; it does not use local storage, session storage, cookies, URLs, or
console output. Page exit and back-forward-cache restoration clear both.

If a transport failure leaves the create result unknown, use **Retry same
create** only while that action remains visible. It sends the exact same body
and in-memory idempotency key. After navigation/reload, fetch the resulting
inactive registration and rotate it to obtain a new secret instead of trying to
recover the original.

## Review Registrations

Canonical rows show only the destination origin and hostname. The full path and
query are encrypted and are not retrievable after submission. Rows also show:

- active/inactive state;
- subscribed catalog events;
- timeout and description;
- revision and delivery/secret versions;
- imported-secret rotation requirement;
- created/updated actor and time metadata.

Pagination is server-backed. Operational alerts report registration and active
limits, key state, migration state, rollback eligibility, imported secrets that
require rotation, and delivery availability.

## Edit Metadata Or Replace A Destination

**Edit metadata** changes description, events, timeout, or active state without
submitting a URL. **Replace destination** is separate and starts with a blank
field because the full existing destination is intentionally unavailable.

Before any PATCH, the page fetches the current registration and uses its strong
ETag. The confirmation dialog shows the current reviewed registration. If the
server returns `412` or `428`, the page fetches and displays the new current
state and requires another explicit review. It never retries a conditional
mutation against a changed revision automatically.

PR 1 blocks activation because `delivery_capability_ready=false`. Imported
registrations are also blocked until their signing secret is rotated.

## Rotate A Signing Secret

1. Keep or make the registration inactive.
2. Select **Rotate secret**.
3. Review the freshly fetched registration and confirm.
4. Copy the new secret to the receiver's secret manager.
5. Acknowledge storage and close the dialog.

Rotation uses a fresh in-memory idempotency key and the current ETag. A lost
response can be retried only through the visible same-command retry action. An
exact retry returns the same secret while its recorded version remains current.
If that in-memory action is gone, perform a new rotation.

Do not reactivate a receiver until its configuration uses the new secret. In PR
1, final activation remains unavailable regardless.

## Delete A Registration

Delete first fetches the current registration, presents it for review, and
sends its ETag. Deletion is a soft delete and cannot be undone through the UI.
A stale ETag refreshes current state and requires a new confirmation.

## Legacy Compatibility Mode

Legacy compatibility exists only as a temporary migration bridge while
canonical mode is `off`. The page can expose historical create, enable/disable,
test, delivery-history, and delete controls only when authenticated status
explicitly selects the legacy route family.

Legacy behavior lacks canonical ETags, destination redaction, and signing-secret
rotation guarantees. Do not use a failed canonical request as a reason to switch
or fall back. Complete the reviewed migration runbook before canonical use.

## Current PR 1 Limitations

- No outbound webhook delivery.
- No canonical test sends or delivery history.
- No retry worker, reconciler, or producer health.
- No receiver payload/signature contract for production use.
- No canonical activation while delivery capability is unavailable.

The delivery substrate, durable producers, receiver guide, and final activation
gate are separate follow-up releases. Do not configure a receiver on the
assumption that PR 1 will send events.

Operator environment, migration, rollback, and key-rotation procedures are in
`Docs/Admin_Webhooks_Control_Plane.md`,
`Docs/Admin_Webhooks_Migration_Runbook.md`, and
`Docs/Admin_Webhooks_Key_Rotation_Runbook.md`.
