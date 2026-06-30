# Audio Studio

Audio Studio is the server-backed workspace for Narration, Podcast, and Briefing workflows. It replaces the Audiobook-first studio surface with `/audio-studio` while keeping `/audiobook-studio` as a compatibility interstitial/fallback route into Narration until migration is stable.

Music and SFX generation remain planned Audio Studio expansion areas. The remaining roadmap defers standalone music composition and full ACE-Step workflow work until the spoken-audio creator MVP and shared platform contracts are stable.

## Workflows

- Narration: long-form chapter narration, voice settings, subtitle-oriented output, and legacy Audiobook migration.
- Podcast: multi-speaker scripts with speaker-specific sections and speech generation.
- Briefing: source-driven sections for summaries, updates, and analyst notes.
- Music/SFX: planned prompt-based cues, beds, stingers, loops, and ambience through provider adapters such as ACE-Step.

The stabilization priority treats `narration`, `podcast`, and `briefing` as first-class spoken-audio workflows. Clients should pass workflow ids instead of inferring behavior from route names. `music` may appear in compatibility or experimental surfaces, but implementation planning treats ACE-Step and standalone music composition as later roadmap slices.

## Routes

- `/audio-studio`: canonical WebUI route.
- `/audio-studio?workflow=narration`: opens the Narration workflow.
- `/audio-studio?workflow=podcast`: opens the Podcast workflow.
- `/audio-studio?workflow=briefing`: opens the Briefing workflow.
- `/audio-studio?workflow=music`: planned/follow-up Music workflow; deployments may hide or disable it until the music adapter slice lands.
- `/audiobook-studio`: compatibility interstitial/fallback route that checks for local Audiobook projects and routes users into Narration without requiring a hard redirect during stabilization.

## API Overview

Base path: `/api/v1/audio-studio`

- `GET /workflows`: list supported workflow ids and labels.
- `GET /providers`: list configured generation providers without secrets.
- `POST /projects`: create a project.
- `GET /projects?workflow=narration`: list projects for the current user.
- `GET /projects/{project_id}`: fetch one project.
- `PATCH /projects/{project_id}`: update project metadata/settings with `base_revision_id`.
- `DELETE /projects/{project_id}`: archive a project with `base_revision_id`.
- `PUT /projects/{project_id}/sections/{section_id}`: upsert narration, podcast, or briefing sections.
- `PUT /projects/{project_id}/tracks/{track_id}`: upsert tracks for generated or imported assets.
- `PUT /projects/{project_id}/clips/{clip_id}`: upsert timeline clips.
- `POST /projects/{project_id}/generations`: queue speech or music generation.
- `GET /projects/{project_id}/generations/{job_id}`: inspect a generation job.
- `GET /projects/{project_id}/artifacts`: list generated, render, and export artifacts.
- `GET /projects/{project_id}/artifacts/{artifact_id}/media`: stream one allowed audio artifact.
- `POST /projects/{project_id}/renders`: queue a render job.
- `GET /projects/{project_id}/renders/{job_id}`: inspect a render job.
- `POST /projects/{project_id}/exports`: queue an export job.
- `GET /projects/{project_id}/exports/{job_id}`: inspect an export job.
- `POST /migrations/audiobook/preview`: preview legacy Audiobook import counts.
- `POST /migrations/audiobook/commit`: commit selected legacy Audiobook projects into Narration.

Mutating project, section, track, clip, generation, render, and export calls use optimistic concurrency through revision ids. For existing projects, send the latest `base_revision_id` or target revision id returned by the server. Stale revision ids are rejected instead of silently overwriting user work.

Generation, render, and export requests require a caller-provided `idempotency_key` between 16 and 200 characters. Reusing the same key for the same project, target resource, and target revision returns the existing job instead of enqueuing duplicate work.

## Roadmap Alignment

The accepted remaining-work roadmap is `Docs/superpowers/specs/2026-06-24-audio-studio-remaining-roadmap-design.md`. It supersedes the earlier MVP timing for first-class music generation and ACE-Step while preserving the adapter-pattern, external HTTP provider, allowlisting, and secret-handling requirements.

The next implementation slice is artifact playback/download. Provider capability metadata is a separate follow-up by default unless code inspection proves it is genuinely tiny and safe to include without blurring the artifact-access review boundary.

## Artifact Media Access

Artifact metadata is listed through `GET /projects/{project_id}/artifacts`. Artifact bytes are served separately through `GET /projects/{project_id}/artifacts/{artifact_id}/media` so clients never receive filesystem paths.

The media endpoint uses the normal Audio Studio auth path, including single-user API key mode and multi-user request scoping. It verifies the artifact belongs to the authenticated user's project before reading bytes. Storage paths must resolve under the user's configured Audio Studio output roots, URL-like paths are rejected, symlink escapes are rejected, and only allowlisted audio MIME/extension pairs are served.

The endpoint supports authenticated streaming responses, `Range` requests, `Accept-Ranges: bytes`, safe `Content-Disposition`, and `X-Content-Type-Options: nosniff`. The WebUI fetches small selected-clip artifacts as authenticated Blobs through the background proxy and renders only Blob URLs in `<audio>` and download links. It does not put raw `/api/v1/audio-studio/.../media` URLs into DOM attributes.

## Large Artifact Media Tickets

Audio Studio supports short-lived media tickets for native browser playback and downloads when a Blob fetch would be too large or when the artifact is not audio-previewable.

- Mint endpoint: `POST /api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/tickets`
- Redeem endpoint: `GET /api/v1/audio-studio/media-tickets/{token}`
- Playback tickets are audio-only, reusable for 30 minutes, support `Range`, and use `Content-Disposition: inline`.
- Download tickets are single-use, expire after 10 minutes, ignore browser `Range` headers, and force `Content-Disposition: attachment`.
- Download ticket filenames preserve the artifact id plus a safe, non-dangerous file suffix such as `.json` or `.zip`.
- The server stores only the SHA-256 hash of the ticket token.
- Redemption repeats ownership, artifact existence, safe-root containment, symlink, file size, content hash, MIME, and extension checks.
- Playback ticket content-hash verification is reused only while the resolved file path and stat identity remain unchanged, avoiding repeated full-file hashes for browser range requests.
- Responses use `Cache-Control: private, no-store`, `Referrer-Policy: no-referrer`, and `X-Content-Type-Options: nosniff`.
- `Cross-Origin-Resource-Policy: same-origin` is intentionally not set for ticket media responses until WebUI and extension/shared UI playback compatibility is verified.

Application access logs and intercepted stdlib/uvicorn access log messages redact media ticket tokens. Operators running a reverse proxy should also redact or suppress `/api/v1/audio-studio/media-tickets/{token}` in proxy access logs because the token is a short-lived bearer credential.

The WebUI uses Blob transport for small known-size audio artifacts, playback tickets for oversized or unknown-size audio artifacts, and click-only download tickets for ticket-backed audio and non-audio artifacts. Download ticket URLs are held only long enough to click a temporary hidden anchor.

The current regression coverage includes single-user API key access, per-user isolation, traversal and symlink rejection, duplicate relative-path disambiguation, range handling, download headers, WebUI Blob URL download/preview, media-ticket playback and download flows, hash-verification reuse for repeated range playback, no-artifact and missing-metadata states, fetch failures, stale async UI guards, and oversized or unknown-size ticket playback.

## Provider Adapters

Audio Studio providers use an adapter registry. External HTTP services remain the preferred shape for generation work, and Audio Studio should not execute ACE-Step locally inside the WebUI. Full ACE-Step/music workflow integration is deferred behind artifact playback, minimum provider capabilities, migration compatibility, render/export UI, spoken-workflow stabilization, and platform hardening.

Provider requests are normalized before they are persisted in Jobs:

- Client payloads must not include API keys, bearer tokens, passwords, callback URLs, provider base URLs, or other external URL-bearing fields.
- Provider secrets are read from environment variables or secret config only.
- External provider base URLs must pass `AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST`.
- HTTP endpoints are rejected unless explicitly allowed for local development.
- Provider errors are redacted before they are stored or returned.

### ACE-Step

ACE-Step is planned as the `ace_step` music provider behind the shared external HTTP adapter system. Treat this configuration as the target provider shape for the later music adapter slice rather than a prerequisite for the spoken-audio stabilization work.

```bash
AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST=
AUDIO_STUDIO_ALLOW_HTTP_ENDPOINTS=false
AUDIO_STUDIO_ACE_STEP_BASE_URL=
AUDIO_STUDIO_ACE_STEP_TIMEOUT_SECONDS=60
AUDIO_STUDIO_ACE_STEP_API_KEY=
```

For local development with an HTTP ACE-Step sidecar, set an allowlist entry for that exact loopback origin and set `AUDIO_STUDIO_ALLOW_HTTP_ENDPOINTS=true`. Do not allow broad internal networks or wildcard hosts. The adapter should read these environment variables server-side; `config.txt` keys are not active for Audio Studio providers yet.

## Render And Export Separation

Generation jobs produce source artifacts such as speech clips or music beds. Render jobs compose approved project artifacts into a mixed output. Export jobs package a render or project artifact into a user-facing format.

This separation matters because each step has different provenance:

- Generation artifact: provider, source resource kind/id, source revision id, content hash.
- Render artifact: render id, source artifact ids, target revision id, mix settings, content hash.
- Export artifact: export id, source render/artifact id, manifest, package format, content hash.

Render services validate that source artifacts belong to the current user, project, and expected revision before reading storage paths. Export manifests include source artifact hashes so packages can be audited later.

## Legacy Audiobook Migration

The compatibility route exports local Dexie Audiobook projects as sanitized payloads and sends them to the server in two steps:

1. Preview reports how many projects, chapters, and audio assets would be imported. Preview does not create server projects and does not mutate local Dexie rows.
2. Commit creates Narration projects and sections. The MVP imports legacy audio asset metadata for preview counts but does not copy local Dexie blobs or create renderable clip artifacts from metadata-only references; blob upload/import is a later migration slice. Local Dexie rows are marked migrated only after commit succeeds.

The server never deletes local Dexie data. Cleanup remains a client-side decision after a successful migration.

## Security Notes

- Never send provider secrets in Audio Studio request bodies.
- Never store external provider URLs in project, section, track, clip, job, or migration payloads.
- Use exact allowlist entries for provider origins.
- Prefer HTTPS provider endpoints. Only enable HTTP for local loopback development.
- Treat migration payloads as untrusted user input and keep them under API size limits.
- Use `base_revision_id`, target revision ids, and idempotency keys on every mutating workflow.
