# Audio Studio Large Artifact Media Tickets Design

## Context

Audio Studio already has an authenticated artifact media endpoint:

- `GET /api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/media`
- It validates project ownership through normal auth, resolves artifact paths under safe per-user output roots, rejects path traversal and symlink escapes, serves allowlisted audio media, and supports browser byte ranges.
- The WebUI MVP uses authenticated Blob fetches for small selected-clip audio artifacts and deliberately avoids putting raw media endpoint URLs in DOM attributes because native `<audio>` and download links cannot attach API headers.

`TASK-2358` extends this to large artifacts for both WebUI and extension/shared UI contexts. The result must support native large audio playback and large artifact downloads without query-string API secrets and without loading large files fully into browser memory.

## Goals

- Support native large audio playback with browser `Range` requests and seeking.
- Support large downloads for project-owned artifacts without full in-memory Blob buffering.
- Work from the WebUI and extension/shared UI service layer.
- Keep API keys, JWTs, provider secrets, and filesystem paths out of URLs and DOM attributes.
- Preserve strict project/user authorization and safe storage-root checks.
- Keep small-artifact Blob preview/download behavior in place to reduce churn.

## Non-Goals

- Do not implement service-worker header injection for this slice.
- Do not build a generic public file-sharing system.
- Do not support inline browser preview for non-audio artifacts in this slice.
- Do not persist ticket URLs in project data, clip settings, localStorage, application telemetry, or app-managed history.
- Do not add provider-adapter work, ACE-Step support, or full timeline mixing.

## Considered Approaches

### 1. DB-Backed Scoped Media Tickets

An authenticated API mints an opaque, short-lived ticket for one project artifact and purpose. Native browser media/download elements redeem the ticket through a normal URL.

Pros:
- Works with native `<audio>` range requests and browser downloads.
- Shared UI and extension contexts can mint tickets through existing authenticated request helpers.
- Tickets are scoped, revocable, auditable, and do not expose API credentials.
- Single-use downloads can be enforced atomically in the database.

Cons:
- Requires ticket storage, cleanup, and token redaction.
- Bearer ticket URLs can still appear in browser history or infrastructure access logs.

### 2. Service-Worker Header Injection

A local service worker intercepts media URLs and injects auth headers before forwarding to the authenticated media endpoint.

Pros:
- Avoids bearer ticket URLs.
- Could reuse the existing authenticated endpoint directly.

Cons:
- Fragile across WebUI, extension, browser-app, service-worker scopes, CORS, lifecycle, and cache behavior.
- Harder to test and reason about than explicit ticket redemption.

### 3. Streamed Fetch Bridge

The frontend fetches with auth headers and pipes response streams into playback or download surfaces.

Pros:
- Avoids bearer URLs for downloads in browsers that support stream-saving APIs.

Cons:
- Does not map cleanly to native audio seeking.
- Browser compatibility is more complex.
- Easy to fall back into accidental large in-memory buffering.

## Decision

Use **DB-backed scoped media tickets**.

Tickets are opaque bearer credentials for one artifact and one purpose. Playback tickets are reusable for native range playback for 30 minutes. Download tickets are single-use and expire after 10 minutes. Raw ticket tokens are never stored; the database stores only a SHA-256 token hash and metadata.

## API Contract

### Mint Ticket

`POST /api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/tickets`

Auth: normal API key/JWT auth.

Request:

```json
{
  "purpose": "playback"
}
```

`purpose` is one of:

- `playback`: audio-only, reusable, range-capable.
- `download`: download-only, generic project artifact unless blocked as dangerous, single-use.

Response:

```json
{
  "ticket_url": "/api/v1/audio-studio/media-tickets/{token}",
  "expires_at": "2026-06-24T12:00:00Z",
  "purpose": "playback",
  "artifact_id": "artifact_123"
}
```

The response must not include token hashes, filesystem paths, provider secrets, or storage roots.

### Redeem Ticket

`GET /api/v1/audio-studio/media-tickets/{token}`

Auth: none beyond the bearer ticket.

Behavior:

- Hash the path token and look up the ticket row by hash.
- Reject malformed, unknown, expired, revoked, or consumed tickets.
- Use the ticket's stored `user_id`, `project_id`, and `artifact_id` to reopen the correct user-scoped project/artifact context.
- Revalidate project/artifact ownership, artifact existence, storage root containment, file existence, symlink safety, file size/hash when available, and purpose-specific type rules.
- Stream the artifact without exposing filesystem paths.

Playback tickets:

- Must be audio artifacts.
- Use `Content-Disposition: inline`.
- Support single byte-range requests and existing `416` behavior.
- Are reusable until expiry or revocation.

Download tickets:

- Force `Content-Disposition: attachment`.
- Do not support range requests in the MVP.
- Are atomically marked consumed before streaming starts.
- If transfer fails after consumption, the user can mint another ticket through a new click.

## Ticket Storage

Use a DB-backed table available to the API process:

- `id`: internal row id.
- `token_hash`: SHA-256 hash of the raw token, unique.
- `user_id`: authenticated user id that minted the ticket.
- `project_id`: Audio Studio project id.
- `artifact_id`: Audio Studio artifact id.
- `purpose`: `playback` or `download`.
- `expires_at`: ticket expiry timestamp.
- `consumed_at`: set for download tickets before streaming.
- `revoked_at`: administrative or cleanup revocation timestamp, nullable.
- `created_at`: creation timestamp.
- `created_by_auth_mode`: audit field for single-user/JWT mode, nullable.
- `last_redeemed_at`: audit timestamp, nullable.

Generate at least 128 bits of entropy; prefer 256-bit URL-safe tokens. Store only `token_hash`, never the raw token.

Cleanup starts as opportunistic deletion on mint and redemption. Expired, consumed, and revoked rows may be removed after a retention window. A scheduled cleanup task can be added later without changing the ticket API.

## Artifact Eligibility

Issuance validates eligibility for a good user experience, but redemption repeats all checks.

Playback:

- Allowed only for audio-playable artifacts such as `audio`, `clip_audio`, `generated_audio`, `tts_audio`, `normalized_audio`, `reference_audio`, `preview_mix`, `final_mix`, and audio `alternate_format`.
- MIME type must be audio-compatible.
- Path must resolve under the user's Audio Studio output roots.

Download:

- Allowed for any owned project artifact under safe roots unless blocked as dangerous active/executable content.
- Non-audio artifacts are download-only.
- Archives/packages are allowed when they are valid project artifacts and safe-root-contained.
- Dangerous extension/MIME blocks include active web content and executable/script formats such as HTML, SVG, JavaScript, shell scripts, batch files, platform executables, app bundles, and other known executable binaries.
- Generic downloads always use `attachment` and `X-Content-Type-Options: nosniff`.

## Security Requirements

- Ticket URLs are scoped bearer credentials, not API keys or JWTs.
- Use short TTLs: playback 30 minutes, download 10 minutes.
- Store token hashes only.
- Redact ticket tokens from application logs and error telemetry.
- Document that reverse-proxy access-log redaction is an operator responsibility.
- Return `Cache-Control: private, no-store`.
- Return `Referrer-Policy: no-referrer`.
- Return `X-Content-Type-Options: nosniff`.
- Return `Cross-Origin-Resource-Policy: same-origin` on ticket redemption responses unless a tested WebUI or extension context proves it blocks required same-origin media behavior.
- Do not place API keys, JWTs, provider secrets, raw storage paths, or ticket hashes in responses.
- Revalidate all important artifact and path checks at redemption.
- Download tickets must be consumed atomically so parallel redeems cannot double-use the ticket.

## Error Handling

Use stable errors where the holder already has a valid-looking ticket and a generic response where detail would aid enumeration.

- Malformed or unknown token: generic `404` without confirming ticket existence.
- Expired ticket: `410 Gone`, `audio_studio_media_ticket_expired`.
- Consumed download ticket: `410 Gone`, `audio_studio_media_ticket_consumed`.
- Revoked ticket: `410 Gone`, `audio_studio_media_ticket_revoked`.
- Artifact missing at redemption: existing artifact `404` semantics.
- Path invalid, symlink escape, or unsafe root at redemption: existing artifact path error semantics.
- File size/hash mismatch: existing conflict semantics.
- Playback range malformed or unsatisfiable: existing `416` behavior.
- Download range request: ignore `Range`, serve the full attachment with `200`, and advertise no range support for download tickets.

## Frontend Behavior

Shared service helpers:

- Add a `mintAudioStudioArtifactMediaTicket(projectId, artifactId, purpose)` helper.
- Return runtime-only ticket URL and expiry metadata.
- Do not persist ticket URLs or raw tokens.

Playback:

- For small audio artifacts, keep current Blob preview/download behavior.
- For over-threshold audio artifacts, mint a playback ticket and set `<audio src={ticket_url} referrerPolicy="no-referrer">`.
- If a playback ticket expires while the audio element is open, handle the media error by attempting one remint and restoring `currentTime` where feasible.
- Do not retry forever.

Download:

- For large downloads and non-audio download-only artifacts, mint a download ticket only in direct response to a user action.
- Immediately trigger download with the returned URL.
- Do not pre-render durable single-use ticket URLs in anchors where browser prefetchers or accidental duplicate clicks can consume them.
- If a user clicks again, mint a fresh ticket.

UI states:

- Keep compact states for unavailable, unsafe, expired, consumed, and failed ticket mint/redeem cases.
- Non-audio artifacts get download-only controls.
- Dangerous blocked artifacts show a disabled state with concise copy.

## Rollout

- Keep the existing small Blob path unchanged.
- Use tickets first for artifacts above the current Blob threshold and for generic non-audio downloads.
- Extend `Docs/Audio_Studio.md` with the media ticket contract and operator log-redaction note.
- Mark `TASK-2358` done only after tests, docs, Bandit, and known skips are recorded.

## Testing

Backend tests:

- Mint playback ticket for owned audio artifact.
- Reject playback ticket for non-audio artifact.
- Mint download ticket for owned non-audio artifact.
- Reject mint across users/projects.
- Redemption supports playback `Range`.
- Download redemption is single-use and atomically consumed.
- Expired, revoked, and consumed tickets return stable errors.
- Redemption revalidates safe roots, symlink escapes, missing files, size/hash mismatch, and dangerous MIME/extension blocks.
- Raw tokens are not stored in DB rows.

Frontend tests:

- Large audio artifact mints playback ticket and uses ticket URL in `<audio>`.
- Expired playback ticket error remints once and restores current time where feasible.
- Large download mints only on click and triggers download with `referrerPolicy`.
- Unknown, expired, and consumed ticket errors show compact states.
- Small Blob path remains unchanged.
- Extension/shared UI path uses the same service helpers.

Security verification:

- Bandit on touched backend ticket/media code.
- `git diff --check`.
- Focused frontend service/component tests.
