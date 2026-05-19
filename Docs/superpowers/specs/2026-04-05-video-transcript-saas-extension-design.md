# Video Transcript SaaS Extension Design

## Summary

This design defines a private browser-extension plus hosted-web product for users who want to ingest video URLs, read transcripts, view summaries, and chat against transcript content without entering the full `tldw` product surface.

The product should ship from a private overlay repo outside `tldw_server`, but it should reuse the existing hosted backend and existing backend contracts. The extension stays narrow and launcher-oriented. The hosted app is the primary workspace.

The key architectural rule is:

- no new product-specific OSS backend endpoints for `video-lite`
- reuse existing media ingest jobs, media detail, auth, billing, and chat surfaces
- put any dedupe optimization behind the existing ingest-job worker path rather than in a new public API contract

## Goals

- Offer a browser-store-distributed entry point for transcript-based video analysis.
- Let users ingest a YouTube URL directly from the current page with minimal friction.
- Support a reduced hosted workspace centered on:
  - `Transcript`
  - `Summary`
  - `Chat`
- Reuse the existing backend for ingestion, transcript retrieval, summarization, and chat.
- Keep the store-facing extension and hosted app in a separate private project so the product is not directly packaged as part of the open-source monorepo.
- Reuse the existing app through an upstream sync plus private patch-overlay model rather than rebuilding the frontend from scratch.
- Require sign-in and an active subscription before users can ingest content or enter the transcript-backed workspace.
- Keep transcripts and summaries stored per-user even when backend processing work is deduplicated internally.

## Non-Goals

- Adding new public `video-lite` endpoints to `tldw_server`.
- Rebuilding the existing backend around a new product-specific media model.
- Shipping a greenfield standalone SaaS frontend unrelated to the existing app surfaces in V1.
- Reproducing the full `tldw` workspace, notebook, or library feature set inside the lightweight mode.
- Building a rich extension-side history manager.
- Exposing all ingestion, RAG, or model configuration knobs in the lightweight experience.
- Sharing user transcript records or user summary records across users.

## Requirements Confirmed With User

- The product should function as a SaaS-oriented frontend distributed via the extension store.
- The backend should be the existing hosted `tldw` project.
- The store-facing product should live in a new project folder outside `tldw_server`.
- The extension UI should stay limited to:
  - quick ingest
  - launcher actions for transcript and chat
  - shallow status and handoff behavior
- V1 should not render full transcript reading or persistent chat directly inside the extension.
- The hosted experience should be the main product surface.
- The extension should detect the current YouTube page and show:
  - `Ingest`
  - `Open transcript`
  - `Quick chat`
- V1 should require sign-in and an active subscription before users can:
  - ingest a source
  - open the transcript-backed workspace
  - view generated summaries
  - chat against transcript content
- The main hosted workspace should be a compact video workspace with three tabs:
  - `Transcript`
  - `Summary`
  - `Chat`
- Hosted intake should use the existing async ingest-jobs flow, not the synchronous add-media flow.
- The private frontend should not perform cross-user search.
- Reuse should happen via a backend dedupe shim behind the existing ingest-job worker.
- Dedupe should still create a normal ingest job that finishes quickly.
- Transcripts and summaries should still be stored per-user.
- On a dedupe hit, the backend should not copy another user’s summary; it should generate the new user’s summary as part of that user’s ingest lifecycle.

## Current State

The backend already exposes the core public primitives needed for this product:

- `POST /api/v1/media/ingest/jobs`
- `GET /api/v1/media/ingest/jobs/{job_id}`
- `GET /api/v1/media/ingest/jobs?batch_id=...`
- `GET /api/v1/media/{media_id}`
- existing auth and billing surfaces
- existing chat surfaces such as `/api/v1/chat/completions`

The existing media-ingest worker already returns `result.media_id` in completed job status. That means the private frontend can submit a normal ingest job, poll status, and open the resulting media item without any new orchestration endpoint.

The existing media detail response already exposes the core hosted workspace fields the private app needs:

- transcript text via `content.text`
- source URL via `source.url`
- stable per-user summary storage via `processing.safe_metadata`, which should carry a reserved `video_lite` summary payload for this product

## Recommended Approach

Use a private overlay repo on top of the existing app, but reuse existing public backend contracts only.

The private product should:

- submit ingest through existing ingest-job endpoints
- poll existing job status endpoints
- open existing media detail by `media_id`
- use existing chat/auth/billing surfaces
- keep all product-specific logic in the private frontend and in internal backend worker behavior

It should not add a public `video-lite` backend contract.

## Product Architecture

### Extension Surface

The extension is the discovery and launch surface.

Its responsibilities are:

- detect supported YouTube pages
- extract current tab URL and basic metadata when available
- show the three primary actions:
  - `Ingest`
  - `Open transcript`
  - `Quick chat`
- hand the user into the hosted app with preserved intent

It should not own full transcript reading, durable chat history, or broad media-library functionality.

### Hosted Lightweight Web Surface

The hosted app is the primary workspace.

This should live in a separate private hosted app repo, not inside the open-source `tldw` app shell. The core screen is a compact video workspace with three tabs:

- `Transcript`
- `Summary`
- `Chat`

The private hosted app should act as an orchestrator over existing backend endpoints, not as a client for new product-specific backend routes.

### Backend Surface

The backend remains the existing `tldw_server` platform.

The public backend surface used by this product should remain:

- existing ingest-job submission and polling
- existing media detail retrieval
- existing auth, billing, and chat endpoints

The product should not add public `video-lite` source-state, workspace, or summary-refresh endpoints.

## Same-User Idempotency

Repeated submits for the same normalized source must not create duplicate media rows for the same user.

Rules:

- if the same user already has an active media record for the same normalized source and that record already has a transcript, the ingest path should resolve to that existing `media_id`
- the backend may still create a normal ingest job for user-facing consistency, but the job should complete quickly and return the existing `media_id`
- `Ingest`, `Open transcript`, and `Quick chat` must all converge on that same-user idempotent behavior

This is separate from cross-user dedupe. Cross-user dedupe is an internal compute optimization. Same-user idempotency is a user-facing correctness rule.

## Dedupe Boundary

Dedupe should be an internal compute and artifact optimization behind the existing ingest-job worker path.

What may be reused internally:

- normalized source identity
- downloaded media artifacts
- canonical transcript artifacts
- other non-user-owned processing artifacts that are safe to reuse internally

V1 should scope reuse conservatively:

- prefer reuse for sources with stable canonical identity, such as normalized YouTube video IDs
- for mutable or weakly identified sources, only reuse artifacts when a freshness or canonical-source fingerprint check passes
- if freshness cannot be established cheaply, fall back to the normal ingest path

What must remain per-user:

- media row
- transcript record stored in that user’s media DB
- summary stored in that user’s record
- any later edits, keywords, notes, chat history, or reprocessing state

Operationally:

- on no dedupe hit:
  - run normal download and transcription
  - persist transcript into the user’s record
  - generate first summary for that user
- on same-user repeat ingest with an existing active transcript-backed record:
  - skip duplicate media-row creation
  - return the existing `media_id`
  - do not re-transcribe
  - do not regenerate summary unless the existing record is missing its dedicated summary artifact
- on dedupe hit:
  - skip download and transcription work
  - materialize transcript into that user’s record
  - generate first summary for that user
  - complete the job quickly with the normal `result.media_id` payload

The private frontend should not know whether a job was deduped. It should only observe a normal ingest-job lifecycle.

## Hosted Flow

The hosted flow should be:

1. User lands on the private hosted intake screen, typically from extension handoff.
2. Hosted app checks sign-in and subscription state using existing auth and billing surfaces.
3. Hosted app submits `POST /api/v1/media/ingest/jobs` with the source URL.
4. Hosted app stores the returned `job_id` and `batch_id`.
5. Hosted app polls `GET /api/v1/media/ingest/jobs/{job_id}` or the batch list endpoint.
6. Once the job reaches `completed`, the hosted app reads `result.media_id`.
7. Hosted app loads `GET /api/v1/media/{media_id}`.
8. Workspace renders:
   - transcript from `content.text`
   - summary from the dedicated `processing.safe_metadata.video_lite.summary` value
   - chat entry using the existing media/chat flows

If the completed job result does not include `media_id`, the hosted app should treat that as failure.

For V1, ingest job completion should mean:

- transcript persistence is finished
- the initial summary attempt has also finished

That keeps the lightweight hosted flow simple. The Summary tab should not need a long-lived “summary still generating” state after job completion. If summary generation fails, the backend should persist a terminal summary status for that user and the hosted app should render that failure or unavailable state.

## Launcher Behavior

The extension should not depend on a backend `launcher_access` route.

Instead:

- `Ingest` should deep-link into the hosted intake screen
- `Open transcript` should deep-link into the hosted app with transcript intent
- `Quick chat` should deep-link into the hosted app with chat intent

The hosted app should then decide whether to:

- route the user into login
- route the user into upgrade
- submit a new ingest job
- reopen a locally known in-progress job
- open an already-known media item in the workspace

This keeps auth and subscription routing in the hosted app rather than in a new backend launcher contract.

## Summary Lifecycle

Summary generation should remain tied to first ingest and explicit full-app re-request behavior.

Rules:

- the lightweight hosted app should not trigger summary generation through a dedicated lite endpoint
- first-ingest jobs should generate the user’s summary automatically
- dedupe hits should still generate the user’s summary automatically after the transcript is materialized into that user’s record
- later regeneration should happen through existing full-app behavior, not a lightweight product-specific API

The lightweight product must not treat generic `processing.analysis` as its canonical summary field. That field represents the last analysis or summary generated in the broader app and can be overwritten by unrelated later actions.

Instead, V1 should reserve a stable summary slot inside existing per-user media metadata, exposed through the existing detail response. The recommended shape is:

- `processing.safe_metadata.video_lite.summary`
- optional companion fields such as:
  - `processing.safe_metadata.video_lite.summary_status`
  - `processing.safe_metadata.video_lite.summary_generated_at`
  - `processing.safe_metadata.video_lite.summary_source`

The hosted Summary tab should read that reserved summary payload only.

## Error Handling

The product should handle these states explicitly:

- sign-in required
- active subscription required
- ingest job queued or processing
- ingest job failed
- completed job missing `media_id`
- media detail fetch failed
- transcript unavailable after ingest failure
- summary unavailable or failed after job completion

The lightweight experience should remain intentionally small and avoid exposing low-level backend controls.

## Privacy And Data Ownership

This product should not share transcript or summary records across users.

Even when internal dedupe avoids repeat compute, the resulting user-visible content should be persisted per-user in that user’s own backend records.

That keeps user data ownership aligned with the rest of the hosted platform and avoids cross-user product semantics.

## Testing Requirements

Verification should cover:

- private hosted intake submitting existing ingest jobs correctly
- dedupe-hit worker behavior still returning a normal completed job with `result.media_id`
- same-user repeated ingest resolving to the existing `media_id` instead of creating duplicate rows
- per-user transcript persistence after dedupe
- per-user first summary generation after dedupe
- hosted workspace rendering transcript from `GET /api/v1/media/{media_id}`
- hosted workspace rendering summary from the reserved `processing.safe_metadata.video_lite.summary` field
- an integration path that submits a real ingest job, observes completed status, and then fetches the resulting media detail successfully
- extension handoff into hosted ingest and workspace flow
- paid-only gating through existing login and subscription surfaces

## Design Rules

- No new public OSS backend endpoints for this product.
- Reuse existing ingest jobs as the canonical ingest path.
- Keep dedupe internal to backend worker logic.
- Keep transcript and summary storage per-user.
- Keep the extension narrow and launcher-oriented.
- Keep the hosted app as the primary workspace.
