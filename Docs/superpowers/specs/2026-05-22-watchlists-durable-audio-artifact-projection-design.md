# Watchlists Durable Audio Artifact Projection Design

Status: Draft for spec review
Date: 2026-05-22
Backlog: TASK-481
Parent PRD: `Docs/superpowers/specs/2026-05-18-watchlists-digest-audio-briefing-prd-design.md`
P0 addendum: `Docs/superpowers/specs/2026-05-22-watchlists-staged-demo-remediation-design.md`

## Summary

Watchlists audio briefing artifacts should remain canonical in Workflows. Watchlists should store a compact, repairable projection of those artifacts in run stats and the canonical digest output metadata so `/watchlists` can show durable script, per-speaker audio, final audio, fallback, and retry state after refreshes, restarts, or transient Workflows/Scheduler lookup failures.

This design is the durable-audio follow-up after the P0 demo-rescue work. It does not introduce a Watchlists-owned binary artifact store. It adds a Watchlists-owned projection contract over existing Workflows artifacts.

## Current Evidence

The implementation already creates most of the raw material this feature needs:

- `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
  - `trigger_audio_briefing()` submits an async `workflow_run` task with metadata containing `source`, `watchlist_job_id`, and `watchlist_run_id`.
  - `AUDIO_BRIEFING_WORKFLOW_DEF` composes the script, cleans it, runs `multi_voice_tts`, and falls back to single-voice TTS when multi-voice generation fails.
- `tldw_Server_API/app/core/Scheduler/handlers/workflows.py`
  - The async `workflow_run` Scheduler task currently returns `"queued"` after submitting the Workflow engine run; Scheduler task completion is therefore not the same as audio workflow completion.
  - The handler accepts payload metadata today, but the current `WorkflowsDatabase.create_run()` path does not persist a run-level metadata field. The durable projection implementation must make correlation metadata first-class instead of relying on mocked `metadata_json` behavior.
- `tldw_Server_API/app/core/DB_Management/Workflows_DB.py`
  - `workflow_runs` stores `inputs_json`, `outputs_json`, and `definition_snapshot_json`, but not durable run metadata in the current schema.
- `tldw_Server_API/app/core/Workflows/adapters/content/audio_briefing.py`
  - `_register_script_artifact()` persists an `audio_script` artifact and returns `script_artifact_id` / `script_artifact_uri`.
- `tldw_Server_API/app/core/Workflows/adapters/audio/multi_voice_tts.py`
  - Registers per-speaker `tts_audio` artifacts with `speaker_artifact: true`.
  - Registers the final `tts_audio` artifact with `final_artifact: true`.
  - Marks fallback speaker artifacts and final background-mix state in metadata.
- `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - `GET /api/v1/watchlists/runs/{run_id}/audio` scans Workflows runs by `watchlist_run_id`, summarizes script/speaker/final artifacts, and falls back to Scheduler status when artifacts are not ready.
  - The endpoint currently does not persist the summarized artifact graph back into Watchlists metadata.
- `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts`
  - Already understands nested `metadata.audio`, flat `audio_briefing_*` fields, script artifacts, speaker artifacts, final artifacts, task IDs, queue names, fallback reasons, and live status merging.
- `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
  - Can render an audio artifact graph and can poll `getWatchlistRunAudio(runId)`.
- `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
  - Shows run audio status and retry, but currently emphasizes final audio rather than the full artifact graph.

The remaining product problem is not absence of artifacts. It is that `/watchlists` treats Workflows artifact discovery as transient live state rather than a durable Watchlists projection.

## Problem

Users need to know what happened to an audio briefing after a digest is generated:

- Was script generation completed?
- Which speaker artifacts exist?
- Is the final mix available?
- Did the workflow fall back to a single voice or narration-only output?
- Is the artifact visible after reload, restart, retry, or temporary Workflows/Scheduler lookup failure?

Today the answer depends on a live bridge from Watchlist run stats to Workflows run/artifact lookup. That bridge is useful, but it is not enough for stable product UX because:

- Watchlists run stats persist only trigger/task state, not the artifact graph.
- Output metadata can contain audio state only when it was manually populated at output creation time.
- Workflow run lookup by only `watchlist_run_id` is ambiguous after retries.
- Scheduler status and Workflow status use different status vocabularies.
- A transient Workflows DB lookup failure can collapse a completed audio briefing into an unclear or unknown state.

## Goals

- Keep Workflows artifacts as the source of truth for script, speaker clips, final mix, fallback artifacts, and download URLs.
- Mirror a compact audio artifact graph into Watchlist run stats.
- Mirror the same compact graph into the canonical non-audio Watchlist output artifact metadata for the run.
- Add an `audio_request_id` so retries and historical artifacts can be distinguished deterministically.
- Use lazy read-repair in `/runs/{run_id}/audio` to rebuild or update stale Watchlists projections from canonical Workflows artifacts.
- Add a best-effort proactive projection path owned by Watchlists, not a hard Workflows-to-Watchlists dependency.
- Preserve existing Watchlists outputs, reports, runs, retry, and advanced OSINT/news workflows.
- Avoid exposing local filesystem paths in mirrored Watchlists metadata.

## Non-Goals

- No new Watchlists binary artifact table.
- No copying audio/script files from Workflows storage into Watchlists storage.
- No broad `/watchlists` redesign.
- No script editor, waveform editor, or podcast studio UI in this follow-up.
- No email attachment delivery of audio artifacts in this follow-up.
- No new artifact retention policy.
- No removal of existing raw workflow, template, voice-map, or power-user controls.

## Decisions

### 0. Correlation Metadata Must Be First-Class

The projection depends on deterministic correlation from a Watchlist audio request to a Workflow run and its artifacts. The implementation must add or use a durable, queryable correlation path before relying on `audio_request_id`.

Recommended implementation:

- Add run-level metadata support to Workflows runs, such as a `metadata_json` column plus `WorkflowsDatabase.create_run(..., metadata=...)`.
- Pass Scheduler payload metadata through the `workflow_run` handler into that run metadata.
- Include `audio_request_id`, `watchlist_run_id`, `watchlist_job_id`, and `source: "watchlist_audio_briefing"` in run metadata.
- Also include the same correlation fields in Workflow inputs or definition metadata so adapters can attach them to generated artifacts.

Fallback implementation if a Workflows run metadata migration is deferred:

- Persist the correlation fields in `inputs_json` and teach the projection lookup to read them.
- This is acceptable only as a compatibility bridge. It should not become the long-term contract because workflow inputs should describe execution inputs, not cross-product ownership metadata.

The current endpoint's scan-by-`metadata_json` behavior is not enough by itself until real Workflow runs persist that metadata.

### 1. Workflows Is Canonical

Workflows remains canonical for artifact rows, binary locations, metadata, retention, and download endpoints.

Watchlists stores only a projection. The projection is a cache for UX, history, and recovery. If the projection disagrees with Workflows, Workflows wins and Watchlists should be repaired.

### 2. Watchlists Mirrors Run Stats And Canonical Output Metadata

The projection should be written to:

- Watchlist run stats, so Run Detail can show durable audio state.
- The canonical non-audio Watchlist output metadata for that run, so Reports/Output Preview can render durable audio state before or without live polling.

"Canonical output" means the base Watchlist output artifact for the run:

- `origin == "watchlists"`
- not an audio output row
- not a generated variant (`variant_of` is absent and `variant_kind` is absent)
- text-like format such as Markdown or HTML
- preferably matching the current `audio_request_id` when available
- otherwise the newest canonical base output for the run

If no canonical output exists, the projection should still update run stats.

### 3. Projection Is Idempotent And Repairable

One backend helper should build and persist the graph. It must be safe to call repeatedly from:

- `GET /api/v1/watchlists/runs/{run_id}/audio` after canonical artifacts are found.
- A proposed Watchlists-owned background projection task.
- Retry handling when a new request supersedes old artifacts.

Repeated calls with the same canonical artifact IDs should not create noisy metadata churn.

### 4. Add `audio_request_id`

Every audio trigger/retry should generate a stable request ID before enqueueing Workflows.

Persist it in:

- Watchlist run stats.
- Canonical output metadata when an output exists.
- Scheduler task metadata.
- Workflow run metadata.
- Workflows artifact metadata where context is available.

Use it before `watchlist_run_id` when matching Workflow runs/artifacts to a Watchlists audio request. `watchlist_run_id` remains a compatibility fallback for pre-request-ID runs.

Retry creates a new `audio_request_id`. Older projections should be marked `stale` or `superseded` and should not be presented as the active retry result.

The Scheduler idempotency key must include the active `audio_request_id`. Otherwise a retry can request a new audio generation but receive the previous idempotent task for `watchlist-audio-briefing:{user}:{job}:{run}`.

### 5. Do Not Mirror Raw `file://` Paths

The Watchlists projection should not persist raw artifact URIs that expose local filesystem paths.

Mirrored artifact summaries may include:

- `artifact_id`
- `type`
- `title`
- `download_url`
- `size_bytes`
- `mime_type`
- `speaker_id`
- `voice`
- compact metadata such as `format`, `multi_voice`, `background_mixed`, `fallback_artifact`

Raw Workflows artifact `uri` values stay inside the canonical Workflows lookup path and API response internals where already used. The frontend should display/open `download_url`, not local paths.

For compatibility, `/runs/{run_id}/audio` may continue returning legacy `audio_uri` / artifact `uri` fields while existing clients depend on them. The durable Watchlists mirror must not persist those raw URI values, and the `/watchlists` UI must prefer `download_url` for links and display names.

### 6. Normalize Statuses At The Projection Boundary

The projection should normalize backend status vocabularies into the Watchlists audio UX vocabulary:

- Workflows `succeeded` -> `completed`
- Workflows/Scheduler `queued` or `pending` -> `queued` when a task ID exists, otherwise `pending`
- `running` -> `running`
- `failed` -> `failed`
- `cancelled` / `canceled` -> `cancelled`
- `dead` -> `dead`
- unavailable lookup -> preserve the best mirrored status, otherwise `unknown`

The same normalized status should be returned by `/runs/{run_id}/audio` and mirrored into `metadata.audio.status`.

## Proposed Data Contract

The durable projection lives under `metadata.audio` in output metadata and `audio` in run stats. Flat `audio_briefing_*` compatibility fields should continue to be maintained.

Example:

```json
{
  "audio": {
    "schema_version": 1,
    "requested": true,
    "status": "completed",
    "task_id": "task_audio_123",
    "audio_request_id": "wla_20260522_abc123",
    "workflow_run_id": "wf_run_456",
    "queue_name": "workflows",
    "synced_at": "2026-05-22T17:20:00Z",
    "source": "workflows",
    "stale": false,
    "fallback_reason": null,
    "script_artifact": {
      "artifact_id": "art_script",
      "type": "audio_script",
      "title": "Briefing script",
      "download_url": "/api/v1/workflows/artifacts/art_script/download",
      "size_bytes": 2048,
      "mime_type": "text/markdown"
    },
    "speaker_artifacts": [
      {
        "artifact_id": "art_host",
        "type": "tts_audio",
        "title": "Host",
        "download_url": "/api/v1/workflows/artifacts/art_host/download",
        "size_bytes": 123456,
        "mime_type": "audio/mpeg",
        "speaker_id": "HOST",
        "voice": "af_bella"
      }
    ],
    "final_artifact": {
      "artifact_id": "art_final",
      "type": "tts_audio",
      "title": "Final mix",
      "download_url": "/api/v1/workflows/artifacts/art_final/download",
      "size_bytes": 4567890,
      "mime_type": "audio/mpeg"
    }
  },
  "audio_briefing_requested": true,
  "audio_briefing_status": "completed",
  "audio_briefing_task_id": "task_audio_123",
  "audio_request_id": "wla_20260522_abc123"
}
```

The mirror is intentionally compact. It should not include full script text, raw per-section TTS metadata, large provenance payloads, or local file URIs.

## Backend Architecture

### New Projection Module

Add a Watchlists-side module:

`tldw_Server_API/app/core/Watchlists/audio_artifact_projection.py`

Responsibilities:

- Resolve the current Watchlists audio request for a run.
- Resolve the matching Workflow run using `audio_request_id` first and `watchlist_run_id` as legacy fallback.
- Read correlation metadata from Workflow run metadata first, then compatibility locations such as `inputs_json` or `definition_snapshot_json` if needed for older runs.
- Build a compact artifact graph from Workflows artifacts.
- Normalize statuses.
- Persist the graph into run stats.
- Persist the graph into the canonical output metadata when available.
- Return mirrored state when canonical lookup is unavailable.
- Avoid leaking local paths or raw exception text.

The module should accept explicit database/service dependencies rather than importing FastAPI endpoint dependencies. This keeps it testable and usable from endpoints and background jobs.

Projection writes must merge into existing run/output metadata rather than replacing unrelated keys. Existing delivery results, template metadata, retention fields, Chatbook references, variant metadata, and unknown future keys must be preserved.

Admin `target_user_id` paths must resolve both the target Watchlists DB and the target Collections DB consistently. The projector should not update the current user's Collections DB when the endpoint is serving an authorized target user.

### Artifact Graph Builder

Artifact classification should be deterministic:

- Script artifact:
  - `metadata.script_artifact == true`, or
  - `type == "audio_script"`, or
  - legacy types such as `"briefing_script"` / `"script"`.
- Speaker artifacts:
  - `metadata.speaker_artifact == true`, or
  - `metadata.speaker_id` exists.
  - Speaker artifacts must never be final candidates unless they are explicitly also marked final/fallback.
- Final artifact:
  - prefer `metadata.final_artifact == true`
  - then `metadata.background_mixed == true`
  - then explicit fallback/final aliases already supported by the current endpoint
  - rank by created time/artifact order only after final hints
- Single-voice fallback artifact:
  - should be tagged as the active final fallback artifact when the `tts_single_voice_fallback` step succeeds, or
  - if not tagged for legacy runs, projection should infer it from the fallback step's terminal output when it is the only completed audio artifact for the active request
- Fallback reason:
  - Workflow run metadata first
  - final artifact metadata second
  - speaker artifact fallback metadata third
  - scheduler/task error category last

The helper should continue to support legacy runs whose artifacts do not yet carry `audio_request_id`.

### Lazy Read-Repair

`GET /api/v1/watchlists/runs/{run_id}/audio` should use this order:

1. Load Watchlist run stats and current `audio_request_id` / task ID.
2. Try canonical Workflows lookup.
3. If canonical artifacts or terminal Workflow status are found, build a graph and mirror it.
4. Return the canonical graph response.
5. If canonical lookup fails, return the mirrored Watchlists graph if present.
6. If no mirror exists, return Scheduler status/pending/unknown as today.

This gives users useful state even if Workflows lookup fails later, while still repairing stale projections when canonical artifacts are available.

### Proactive Projection

Proactive sync should be Watchlists-owned and best effort. It must not add a direct import from Workflows engine into Watchlists, and it must not cause a Workflows run to fail if projection fails.

Preferred implementation proposal:

- When Watchlists successfully submits an audio workflow request, enqueue a lightweight Watchlists projection task keyed by `user_id`, `run_id`, `job_id`, `audio_request_id`, and `task_id`.
- The projection task polls/looks up Workflows by request ID until it finds a matching run/artifacts or reaches a bounded retry/timeout policy.
- The projection task must not treat Scheduler task terminal status as Workflow terminal status. The async `workflow_run` Scheduler task can complete after it queues the Workflow engine, before the Workflow run itself has finished.
- When terminal status or artifacts are found, it calls the same mirror helper used by lazy read-repair.
- Failure to project is logged as a Watchlists-side projection failure and can be repaired by the next `/runs/{run_id}/audio` read.
- Do not add this proactive task unless its queue/worker path is already guaranteed or explicitly scaled. The feature should not recreate the previous "queued forever" failure mode on a new projection queue.

Fallback implementation:

- If adding a new task is too much for the first durable-artifacts PR, implement the shared projection helper and lazy read-repair first.
- The design still requires proactive sync as a follow-up, but lazy read-repair remains the hard reliability guarantee.

### Retry Semantics

On audio retry:

- Generate a new `audio_request_id`.
- Persist the new request ID and task ID.
- Mark the previous mirrored `metadata.audio` graph as `stale: true` or move it to a small `previous_audio` summary if needed for diagnostics.
- Clear active `final_artifact`, `script_artifact`, `speaker_artifacts`, `download_url`, and terminal status for the active request unless the new canonical graph is already available.
- The UI must show the retry as queued/running for the new request, not completed because an old final artifact exists.

### Failure Handling

- If Workflows DB is unavailable, return mirrored metadata if present.
- If Scheduler status is unavailable, return mirrored status if present, otherwise `unknown` or pending fallback.
- If only script exists, mirror and show script.
- If speakers exist but final mix fails, mirror and show speakers plus failure/fallback reason.
- If final exists but script/speaker artifacts are missing, return final and mark missing intermediates by absence, not fabricated placeholders.
- If projection write fails, return the canonical response to the user and log a sanitized warning.
- Lazy read-repair is allowed to mutate state from a GET request only because the write is idempotent, best effort, and repairs an already-created artifact projection. It should avoid writes when the mirrored graph already matches the canonical graph.

## Frontend Scope

The frontend should use the mirror as durable cached state and live `/runs/{run_id}/audio` as the canonical refresh path.

Required behavior:

- Output Preview renders mirrored `metadata.audio` immediately when present.
- Output Preview polls `/runs/{run_id}/audio` while status is non-terminal or task ID exists.
- Live status/artifacts override stale mirrored metadata when returned.
- If live lookup fails, keep the mirrored graph visible and show a small unavailable/error line.
- Run Detail should show script, speaker, final artifact summaries, fallback reason, and retry state, not just "Final audio available."
- Artifact links should use `download_url`.
- Raw `file://` paths should not appear in visible UI.
- Retry should show the new request as queued/running and should not present old final audio as the active result.

Out of scope:

- new player/mixer UI
- waveform previews
- script editing
- per-speaker regeneration controls
- batch artifact export
- audio delivery attachments

## API And Schema Impact

Backend:

- Extend or reuse `WatchlistRunAudioResponse` with `audio_request_id`, `workflow_run_id`, `schema_version`, `synced_at`, and `stale` if needed.
- Keep existing `script_artifact`, `speaker_artifacts`, `final_artifact`, `download_url`, `fallback_reason`, `queue_name`, and `error` fields.
- Keep `audio_uri` as a compatibility field where necessary, but do not use it as the durable mirrored link field.
- Preserve flat `audio_briefing_*` metadata fields for compatibility.
- For admin `target_user_id` reads, artifact download URLs must resolve to the target user's artifact context. If the generic Workflows artifact download endpoint cannot do that, return a Watchlists-scoped proxy/download URL or include an explicit supported target parameter rather than emitting a broken current-user link.

Frontend:

- Extend `WatchlistRunAudioStatus` with the same optional fields.
- Keep existing `outputMetadata.ts` merge behavior, but make stale/request ID handling explicit.
- Make `RunDetailDrawer` render the full artifact graph using the same normalized shape as Output Preview.

## Testing Strategy

### Backend Unit Tests

- Builds graph with script, speakers, and final mix.
- Builds graph with script-only partial state.
- Speaker artifacts do not become final audio candidates.
- Final/mixed artifact precedence wins over raw/intermediate audio.
- Fallback reason propagates from Workflow run metadata and artifacts.
- `succeeded` normalizes to `completed`.
- `audio_request_id` match wins over older same-run artifacts.
- Legacy `watchlist_run_id` matching still works.
- Raw `file://` URIs are not mirrored into Watchlists output metadata.
- Single-voice fallback artifacts are surfaced as the active final/fallback artifact.

### Backend API Tests

- `/runs/{run_id}/audio` returns canonical Workflows artifacts and mirrors them into run stats and canonical output metadata.
- `/runs/{run_id}/audio` returns mirrored metadata when Workflows DB lookup fails.
- `/runs/{run_id}/audio` returns Scheduler pending when neither canonical artifacts nor mirror exist.
- Retry creates a new request ID and marks/clears stale active artifacts.
- Retry uses an idempotency key that includes the active `audio_request_id`.
- Projection write failure does not prevent returning canonical artifact response.
- Admin `target_user_id` paths resolve the correct Watchlists and Collections DBs.
- Admin `target_user_id` artifact links resolve in the target user's context or use a Watchlists-scoped proxy.
- Real Workflow runs persist or otherwise expose correlation metadata used by the projection; mocked `metadata_json` alone is not sufficient.

### Frontend Tests

- Output Preview renders mirrored artifacts before live polling resolves.
- Live response overrides stale mirrored artifacts.
- Live lookup failure keeps mirrored artifacts visible.
- Run Detail renders script, speaker, and final artifacts.
- Retry state does not show old final artifacts as current.
- UI never displays raw local artifact paths.

### Demo/E2E Test

- Create or use a run with audio enabled.
- Generate a digest output.
- Trigger audio.
- Confirm `/runs/{run_id}/audio` returns status, request ID, script artifact, speaker artifacts when available, final artifact when available, and download URLs.
- Reload `/watchlists` and confirm the artifact graph remains visible from mirrored metadata.

## Acceptance Criteria

- Completed audio briefings remain visible in `/watchlists` after refresh/restart without relying only on Scheduler task status.
- `/runs/{run_id}/audio` can rebuild missing/stale Watchlists projections from Workflows artifacts.
- Watchlists projections do not duplicate binary files and do not become canonical artifact storage.
- Retries do not confuse old final artifacts with the active audio request.
- Partial artifacts are visible when final audio fails.
- Existing Watchlists digest/output/run flows continue to work.
- Raw local file paths are not persisted into Watchlists output metadata or displayed in UI.

## Risks And Mitigations

- Risk: Projection drift.
  - Mitigation: Workflows remains canonical; lazy read-repair overwrites stale projection by `audio_request_id` and artifact IDs.
- Risk: Workflow-to-Watchlists coupling.
  - Mitigation: Proactive sync is Watchlists-owned and best effort; no Workflows engine import of Watchlists projection code.
- Risk: Retry ambiguity.
  - Mitigation: `audio_request_id` is generated per request and used before `watchlist_run_id`.
- Risk: Metadata bloat.
  - Mitigation: Mirror only compact summaries; large text, per-section metadata, and binaries remain in Workflows.
- Risk: Admin/user DB routing mistakes.
  - Mitigation: API tests cover `target_user_id` and explicit Collections DB resolution.
- Risk: User sees stale success during retry.
  - Mitigation: Retry marks previous active graph stale and clears active terminal artifacts for the new request.
- Risk: Correlation metadata is only present in tests.
  - Mitigation: Add a real Workflows run metadata persistence path, or a clearly temporary inputs-based compatibility path, before implementing projection matching.
- Risk: Proactive projection repeats the earlier stuck-queue failure.
  - Mitigation: Do not ship proactive projection without an ensured worker path; lazy read-repair remains the first reliability guarantee.
- Risk: Admin artifact links point at the wrong user's Workflows DB.
  - Mitigation: Use target-aware download URLs or a Watchlists proxy for target-user reads.

## Deferred Work

- Script editing and approval gates inside `/watchlists`.
- Per-speaker regeneration and final remixing controls.
- Audio artifact delivery by email attachment.
- Batch export of digest/audio artifact bundles.
- Retention policy UI for Workflows audio artifacts.
- Power-user presets for reusable casts and audio output pipelines.
