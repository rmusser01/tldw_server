# tldw Mobile Sync Companion Design

Date: 2026-07-05
Task: TASK-12151
Status: Draft for review

## Context

tldw_server already has the hard pieces a mobile client needs: AuthNZ in single-user API-key mode and multi-user JWT mode, Sync v2 endpoints, chat endpoints, note/chat domain concepts, and typed web/extension client patterns. The mobile app should not start by recreating the WebUI. It should start as a small offline companion that proves mobile sync correctness.

The app is open source and self-hosted-first. Later it should also connect cleanly to the primary commercial hosted tldw instance without forking the client or adding hosted-only behavior to the core app.

## Decision

Build an Expo/React Native app for iOS and Android as the MVP. Use local SQLite for mobile state, platform secure storage for credentials, and Sync v2 for foreground/manual synchronization. The first useful product is chat-first: browse chat history, draft messages offline, edit notes, and queue URL/text captures.

Do not include offline media blobs, background sync guarantees, hosted billing/account flows, or local model inference in MVP. Local model fallback remains a later product goal, likely requiring native modules or a dev-client/eject path.

## MVP Scope

- iOS and Android from day one.
- Self-hosted tldw_server instances first.
- Single-user `X-API-KEY` auth and multi-user bearer JWT auth.
- Chat-first navigation with Chat, Notes, Capture, and Sync/Settings.
- Offline browse/edit/draft behavior backed by local SQLite.
- Foreground/manual Sync v2 only: app open, pull-to-sync, save-triggered sync, explicit sync.
- URL/text capture queue only, synced as metadata via `source_cache.entry`.
- Sync v2 `server_trusted_v1` for MVP.

## Explicit Non-Goals

- No offline media blob sync.
- No background delivery guarantees.
- No local model inference in MVP.
- No custom credential encryption beyond platform secure storage.
- No automatic conflict merge.
- No hosted-only client flows in the shared mobile core.
- No file capture until upload and ingestion compatibility is proven.
- No capture ingestion on mobile. Mobile records metadata; server-side ingestion remains separate.

## Desired Product Direction

After the MVP proves sync correctness, grow the app toward a full mobile knowledge client: richer capture, media metadata, hosted login, better conflict UX, background sync where platform limits allow it, and optional local model fallback.

For local inference, the current direction is a separate spike using native-backed llama.cpp bindings, such as `swift-llama-cpp` on iOS and a comparable Android path based on llama.cpp Android support. This should remain isolated from MVP sync work because model storage, battery, thermal behavior, native packaging, and UX expectations are independent risks.

## Architecture

Create a new mobile workspace, likely `apps/tldw-mobile`. Keep it TypeScript and Expo-first. Reuse protocol knowledge from `apps/packages/ui` where cheap, especially API request behavior, auth conventions, typed Sync v2 shapes, and normalization helpers. Do not reuse desktop screens.

Core modules:

- `connection`: server URL, auth mode, health/capability checks, credential lookup, token invalidation.
- `sync`: Sync v2 capabilities/profile, device registration, dataset enrollment, push, pull, conflict listing, cursors.
- `local-db`: SQLite tables for notes, conversations, messages, capture queue, sync cursors, pending envelopes, and conflict payloads.
- `domains`: mobile adapters for notes, conversations, messages, and capture queue records.
- `ui`: native React Native screens for Chat, Notes, Capture, and Sync/Settings.

The backend should stay mostly unchanged for MVP. Add only small Sync v2 or API compatibility fixes if the mobile gap audit finds missing fields or unstable domain behavior.

## Data Flow

First connect:

1. User enters server URL and chooses API key or username/password JWT auth.
2. App validates auth and server reachability.
3. App calls Sync v2 capabilities/profile.
4. App checks the required capability matrix: protocol version, domains, operations, encryption policy, and batch/payload limits.
5. App registers a mobile device ID and enrolls or joins the personal dataset.
6. App stores credentials only in platform secure storage.

Normal use:

1. Local edits write to SQLite first.
2. Each write creates a pending Sync v2 envelope with stable object ID, base metadata, payload hash, and local envelope status.
3. Foreground sync pushes pending envelopes, pulls remote envelopes by domain cursor, applies accepted changes, and updates cursors.
4. Online chat sends through server chat endpoints, then records conversation/message sync state.
5. Offline chat stores queued drafts only; it does not invent a local assistant answer.
6. Capture stores URL/text locally and syncs it through `source_cache.entry` metadata envelopes. Capture ingestion is not part of mobile sync.

Track both object state and envelope state. An object can have a synced base and a pending local revision at the same time.

## Compatibility Gate

The app should fail closed around sync compatibility, but not around offline use.

On first setup, unsupported servers block sync setup. After a compatible server has been verified, an unreachable server must not block local browse/edit/draft/capture. If a later server response shows missing required domains or incompatible Sync v2 behavior, the app blocks sync writes and surfaces the compatibility problem instead of attempting partial sync.

Required MVP domains:

- `notes.note`
- `chat.conversation`
- `chat.message`
- `source_cache.entry`

Required MVP operations:

- `notes.note`: `upsert`, `tombstone`
- `chat.conversation`: `upsert`, `tombstone`
- `chat.message`: `append`, `tombstone`
- `source_cache.entry`: `upsert`, `tombstone`

The app must call `GET /api/v1/sync/capabilities` before enrollment and require:

- `protocol_version` is `sync-v2-m1`.
- `min_supported_protocol_version` is `sync-v2-m1`.
- `domains` contains all required MVP domains.
- `operations` contains all required MVP operations.
- `encryption_policies` contains `server_trusted_v1`.
- `encryption.policy` is `server_trusted_v1`.
- `encryption.ready` is `true`.
- `max_batch_size` is at least `1`.
- `max_envelope_payload_bytes` is present and used as the mobile payload cap.

`blob_transfer.supported` and `supports_attachments` may be false. Do not require `attachment.ref`, media, workspace, or blob domains in MVP.

URL/text captures use `source_cache.entry` with operation `upsert`. The envelope must include `source_id`, `content_hash`, and provenance metadata (`url`, `source_uri`, `origin`, or `provider`) through payload or routing metadata, matching the existing source-cache adapter requirements. The payload is metadata-only: URL, optional title, optional selected text, created timestamp, and capture source.

## Error Handling

User-visible connection states should distinguish unreachable server, invalid API key, expired/invalid JWT, unsupported server, missing Sync v2 domain, local-network permission issue, and TLS/certificate failure where the platform exposes it.

JWT refresh is out of MVP. Multi-user login uses `POST /api/v1/auth/login` with form-encoded `username` and `password`, stores only the returned `access_token` in platform secure storage, and sends it as `Authorization: Bearer <token>`. On `401`, the app deletes the token, marks auth expired, and asks the user to log in again. Do not store the password.

Secrets stay out of SQLite and logs. Auth headers must never be logged.

Conflicts preserve both versions durably across app restarts. MVP conflict actions are:

- Keep local.
- Keep remote.
- Duplicate.

Capture failures leave queue entries in a retryable failed state. Mobile records capture intent; server ingestion remains server-owned.

## Testing And Verification

Start with a Sync v2 compatibility fixture that proves capabilities and required MVP domains before mobile code depends on them.

Minimum automated coverage:

- Local SQLite/domain-adapter tests for create, edit, delete, pending envelopes, conflict persistence, and cursor updates.
- Sync adapter tests for push/pull ordering, rejected envelopes, conflicts, and idempotent retry.
- API integration path for single-user API key auth.
- API integration path for multi-user JWT auth.
- Compatibility tests for missing domain, missing operation, unsupported protocol version, unsupported encryption policy, encryption not ready, and older server behavior.

Manual device checks:

- iOS simulator/device and Android emulator/device.
- LAN HTTP server URL.
- Self-signed or local TLS where supported.
- Unreachable server after successful setup.
- Invalid/expired credentials.
- App restart with pending envelopes.
- App restart with conflicts.

## Main Risks

- Sync v2 domain contract drift between server and mobile.
- Capture becoming ingestion work instead of a queue.
- Note conflict UX expanding beyond MVP.
- Expo managed workflow limits once native local inference is added.
- Local network and TLS behavior varying between iOS and Android.

## Open Implementation Questions

- Whether `attachment.ref` should enter the first post-MVP media metadata slice.
- Exact hosted-login UX once commercial-hosted support starts.
- Whether local inference should use Expo dev-client native modules or a separate native shell.

## References

- `tldw_Server_API/app/core/Sync/README.md`
- `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- `tldw_Server_API/app/api/v1/endpoints/sync.py`
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- `apps/tldw-frontend/lib/api.ts`
- [swift-llama-cpp](https://swiftpackageindex.com/pgorzelany/swift-llama-cpp)
- [llama.cpp Android docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/android.md)
