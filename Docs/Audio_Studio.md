# Audio Studio

Audio Studio is the server-backed workspace for narration, podcast, briefing, and music workflows. It replaces the Audiobook-first studio surface with `/audio-studio` while keeping `/audiobook-studio` as a compatibility route into Narration until migration is stable.

## Workflows

- Narration: long-form chapter narration, voice settings, subtitle-oriented output, and legacy Audiobook migration.
- Podcast: multi-speaker scripts with speaker-specific sections and speech generation.
- Briefing: source-driven sections for summaries, updates, and analyst notes.
- Music: prompt-based music cues and beds through provider adapters such as ACE-Step.

The API treats these as first-class workflows. Clients should pass one of `narration`, `podcast`, `briefing`, or `music` instead of inferring behavior from route names.

## Routes

- `/audio-studio`: canonical WebUI route.
- `/audio-studio?workflow=narration`: opens the Narration workflow.
- `/audio-studio?workflow=podcast`: opens the Podcast workflow.
- `/audio-studio?workflow=briefing`: opens the Briefing workflow.
- `/audio-studio?workflow=music`: opens the Music workflow.
- `/audiobook-studio`: compatibility route that checks for local Audiobook projects and routes users into Narration.

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
- `POST /projects/{project_id}/renders`: queue a render job.
- `GET /projects/{project_id}/renders/{job_id}`: inspect a render job.
- `POST /projects/{project_id}/exports`: queue an export job.
- `GET /projects/{project_id}/exports/{job_id}`: inspect an export job.
- `POST /migrations/audiobook/preview`: preview legacy Audiobook import counts.
- `POST /migrations/audiobook/commit`: commit selected legacy Audiobook projects into Narration.

Mutating project, section, track, clip, generation, render, and export calls use optimistic concurrency through revision ids. For existing projects, send the latest `base_revision_id` or target revision id returned by the server. Stale revision ids are rejected instead of silently overwriting user work.

Generation, render, and export requests require a caller-provided `idempotency_key` between 16 and 200 characters. Reusing the same key for the same project, target resource, and target revision returns the existing job instead of enqueuing duplicate work.

## Provider Adapters

Audio Studio providers use an adapter registry. The MVP prefers external HTTP services for generation work and does not execute ACE-Step locally.

Provider requests are normalized before they are persisted in Jobs:

- Client payloads must not include API keys, bearer tokens, passwords, callback URLs, provider base URLs, or other external URL-bearing fields.
- Provider secrets are read from environment variables or secret config only.
- External provider base URLs must pass `AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST`.
- HTTP endpoints are rejected unless explicitly allowed for local development.
- Provider errors are redacted before they are stored or returned.

### ACE-Step

ACE-Step support is exposed as the `ace_step` music provider when configured.

```bash
AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST=
AUDIO_STUDIO_ALLOW_HTTP_ENDPOINTS=false
AUDIO_STUDIO_ACE_STEP_BASE_URL=
AUDIO_STUDIO_ACE_STEP_TIMEOUT_SECONDS=60
AUDIO_STUDIO_ACE_STEP_API_KEY=
```

For local development with an HTTP ACE-Step sidecar, set an allowlist entry for that exact loopback origin and set `AUDIO_STUDIO_ALLOW_HTTP_ENDPOINTS=true`. Do not allow broad internal networks or wildcard hosts. The MVP adapter reads these environment variables; `config.txt` keys are not active for Audio Studio providers yet.

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
