# Accepted Buddy conversation turns

Buddy turns use the authenticated `POST /api/v1/chat/completions` route through a fixed in-process ASGI transport. Chat continues to own authentication, token scope, budgets, moderation, effective credentials, provider admission, accounting, message persistence and Persona memory. The adapter does not call provider functions directly or accept an arbitrary endpoint URL.

## API

- `POST /api/v1/buddies/turns?client_slot=default` accepts `{conversation_id, text, client_request_id, expected_attachment_version, model?, provider?}` and returns HTTP 202 with the turn record.
- `GET /api/v1/buddies/turns?client_slot=default&limit=50&offset=0` returns `{turns, limit, offset}`. The default limit is 50 and maximum is 100. Add `status=active` to select only queued/running turns before pagination, so newer completed history cannot hide ongoing work. Omit `status` for unfiltered history; other status values return 422.
- `GET /api/v1/buddies/turns/{id}` reads one authenticated owner's record.
- `POST /api/v1/buddies/turns/{id}/stop` revokes its publication authority. This is the only interaction action that means Stop.

Records contain `id`, `client_slot`, `client_request_id`, `conversation_id`, `conversation_title`, `workspace_id`, `attachment_version`, `status`, `result_message_id`, `error_code`, `created_at` and `updated_at`. Status is `queued`, `running`, `completed`, `failed` or `stopped`. Read completed content through the existing authorized conversation/message API using the exact result message ID. The ledger contains no message bodies or credentials.

Text is limited to 12,000 characters. The caller must provide a provider/model or the selected conversation must have saved effective completion settings. Missing configuration returns 422; the adapter never chooses a model from the server catalog. Slash commands are rejected; the narrow request supplies no tools and requests `tool_choice: none`. Existing Chat policy remains authoritative.

Reuse the same `client_request_id` and identical body after an ambiguous acceptance response. A repeated key returns the existing record and never dispatches again, including failed/stopped work. Changing its input returns 409. A deliberate new action needs a new key.

## Ownership and ordering

Acceptance checks the authenticated principal's active attachment, its expected version and the exact selected private conversation. Workspace attachments can target only conversations currently in that owned workspace. The attachment version is locked again when the ledger record commits. The adapter captures the conversation ID, revision and workspace; subsequent execution does not depend on the attachment preference. Detaching, rebinding, closing a popover or navigating does not stop queued or running turns.

The process retains acceptance tasks independently of request cancellation, then drains one FIFO per principal/conversation. Different conversations can run concurrently. The default process capacity is 64 accepted turns, with at most 16 per principal, including running work. Full queues return 429. Bodies and forwarded admission headers remain only in these bounded tasks and are cleared when each turn exits.

Before dequeue and inside every canonical Chat message transaction, the adapter checks the owner lease, active turn status, conversation owner/revision/scope and workspace availability. The transaction serializes these checks with the message and its ledger result identity. A changed/deleted/moved conversation or deleted workspace fails closed. Ordinary Chat requests do not join the Buddy FIFO; this adapter does not claim to serialize every writer to a conversation.

## Ordinary Console requests

This acceptance guarantee applies only to turns sent through the Buddy turn API. The Console's abort-controller provider is mounted above route children, and the inspected plain client-side navigation paths do not themselves call Stop. Its ordinary completion streams still run from browser JavaScript, however: explicit Stop aborts their controller, server/credential scope changes invalidate their request scope, and page close, reload or transport loss has no Buddy ledger or recovery guarantee. Main Chat streaming handles request cancellation as a stream cancellation. Other existing run APIs can have different ownership; this adapter does not change them.

## Stop, failures and restart

Stop updates the SQL record before requesting cancellation of the local ASGI call. Another process can revoke a record even if it cannot cancel the owning process's coroutine. The publication transaction prevents later messages from a stopped or superseded owner. A reply already committed before Stop remains published and returns `completed`; Stop does not erase it.

Chat can persist a user message before contacting a provider. Timeouts, interruption or provider failures can therefore leave a user message or other effects already performed by Chat. Cancellation cannot undo provider or tool effects. `interrupted_unknown` and `completion_unknown` explicitly do not mean rollback, and the adapter never retries them. If Chat already committed a reply and its result ID, runtime failure or shutdown preserves that completed result. A transient database error before dispatch finalizes the affected turn without discarding the remaining FIFO. Existing retry policy within one ordinary Chat execution is unchanged.

A metadata-only owner lease is renewed every five seconds and expires after 30 seconds. On status read or ownership claim, an expired owner's queued/running records become terminal: an atomically recorded reply remains `completed` with its exact result ID; work without a committed reply becomes `failed` with `interrupted_unknown`. Expiry is reconciled before lease renewal, including reacquisition by the same process, so old work cannot regain publication authority. SQL publication checks prevent a superseded process from publishing. Credentials and pending bodies cannot be recovered, so a restart never replays work. Graceful application shutdown revokes pending records before cancelling tasks.

While a principal has a live owner lease, a different server process returns 503 with `Retry-After: 5` for new acceptance. Status and Stop remain available on any process. Multi-worker deployments should use principal affinity for acceptance; this is process-owned execution, not a distributed durable job queue. Expired work is reconciled when its owner next reads/sends, within the 30-second lease bound rather than through credential replay.

Related decision: [ADR-005](../../backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md).
