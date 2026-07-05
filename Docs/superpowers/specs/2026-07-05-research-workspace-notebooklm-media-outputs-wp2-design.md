# Research Workspace NotebookLM Media Outputs WP2 Design

Date: 2026-07-05
Status: Reviewer-approved
Backlog: TASK-12160

## Summary

WP2 adds real NotebookLM-style media outputs to Research Workspace:

- Video Overview is a narrated slideshow rendered to a downloadable video.
- Infographic is an actual generated image produced by the configured image
  generation backend.

This replaces the earlier cheap script/storyboard/visual-brief idea. The user
expectation is real media, not text labels pretending to be media.

The implementation should reuse existing tldw systems:

- Jobs for long-running output generation.
- Slides generation and presentation rendering for slideshow video.
- TTS for narration audio.
- Image generation adapters for infographic images.
- Output artifacts plus workspace artifacts for durable preview/download
  metadata.

No new video engine, image backend, or frontend media stitching layer is needed.

## Goals

- Add `video_overview` and `infographic` as first-class Research Workspace
  artifact types.
- Run both outputs through backend jobs so generation survives slow TTS, image,
  and ffmpeg work.
- Store final media as durable output artifacts with `/api/v1/outputs/{id}/download`
  URLs.
- Record traceability in workspace artifact `producer_metadata`,
  `source_lineage`, `version_metadata`, and `export_refs`.
- Show completed video/image previews and download controls in the Studio pane.
- Fail visibly and recoverably when sources or required backends are missing.

## Non-Goals

- No cinematic video generation.
- No arbitrary image editor or multi-image storyboard system.
- No new storage subsystem.
- No frontend-side ffmpeg, canvas video assembly, or browser media stitching.
- No clone of Google-specific NotebookLM templates, quotas, or Drive sync.

## Approach

Use one small Research Workspace output job layer.

The WebUI submits an output request for a workspace and selected source ids. The
backend creates a pending workspace artifact, creates a Jobs row with
`job_type="research_workspace_output"`, and the worker updates the artifact as
the job progresses.

The worker must be registered with the server's existing startup worker group or
documented as a required standalone worker entrypoint. A submit-only API is not
acceptable for WP2 because Research Workspace jobs must drain without manual
operator intervention in the normal local server path.

This job layer should coordinate existing primitives directly. In particular,
Video Overview should call `render_presentation_video` directly instead of
submitting a second Presentation Render job and then polling it from the
Research job. The existing presentation render worker is useful precedent, but
nested jobs would add timing, retry, and ownership failure modes for no useful
benefit.

## Backend Contract

Add Research Workspace output endpoints under the workspace API surface:

- `POST /api/v1/workspaces/{workspace_id}/outputs`
- `GET /api/v1/workspaces/{workspace_id}/outputs/{job_id}`

The submit request includes:

- `artifact_type`: `video_overview` or `infographic`.
- `source_ids`: selected workspace source ids.
- Optional generation settings already exposed in Studio where applicable:
  model/provider, slide style, TTS voice/provider, image backend, and image
  dimensions. Image output format is PNG for WP2.

The submit response includes:

- `job_id`
- `status`
- `workspace_id`
- `artifact_id`
- `artifact_type`

The status response includes:

- `job_id`
- `status`
- `progress_percent`
- `progress_message`
- `artifact`, when available
- `error`, when failed

Use existing Jobs status naming where possible.

## Video Overview Flow

1. Validate the workspace, selected source ids, and source readiness.
2. Build a grounded source context from selected sources.
3. Generate a 5-8 slide presentation from that context.
4. Put narration text in each slide's `speaker_notes`.
5. Synthesize per-slide narration with the selected/default TTS provider.
6. Persist each narration audio clip as a durable output artifact with audio
   format and content-type metadata.
7. Attach each clip to its slide metadata as `metadata.studio.audio.asset_ref`
   using `output:<id>`.
8. Persist the presentation snapshot through the existing Slides DB path.
9. Call `render_presentation_video` directly with the presentation version.
10. Persist the final MP4 as an output artifact with metadata origin
    `research_workspace`.
11. Update the workspace artifact to complete with a video preview/download
    `export_ref`.

The `output:<id>` requirement is important: Slides asset resolution currently
accepts output artifact references only. Generated-file ids alone will not
resolve during rendering.

Do not rely on `save_and_register_tts_audio` alone for slide narration. It
tracks generated files for quota/storage, but the renderer resolves slide media
through output artifact rows.

Default format is MP4. WebM is future-only for WP2 and should be added later
only if the UI needs it.

## Infographic Flow

1. Validate workspace, selected source ids, source readiness, and image backend
   capability.
2. Build a grounded source context from selected sources.
3. Generate a concise infographic prompt from that context.
4. Call the configured image generation backend through the existing image
   adapter contract to obtain image bytes.
5. Persist the final PNG as a durable output artifact. WebP can be added later
   as an explicit user option if needed.
6. Update the workspace artifact to complete with image preview/download
   `export_ref`.

Do not rely on file-artifact export URLs as the final user-facing preview. Those
exports are TTL-oriented. Research Workspace artifacts need durable output
artifact URLs.

Avoid using the file-artifact export finalization path for the final preview
asset. It is useful reference code, but WP2 needs durable output artifact
storage for the image the user sees.

## Capability Gates

Extend the Research Workspace capability contract with:

- `video_overview_generation`
- `infographic_generation`
- `image_generation`

Composition:

- `video_overview_generation` requires source browsing, LLM, Slides, TTS, and
  presentation render availability.
- `infographic_generation` requires source browsing, LLM, and image generation.
- Existing `slides_generation` and `audio_summary` remain unchanged.

The UI maps `video_overview` to `video_overview_generation` and `infographic` to
`infographic_generation`.

## Workspace Artifact Shape

Use existing workspace artifact fields instead of adding a second media artifact
store.

For completed media artifacts:

- `artifact_type`: `video_overview` or `infographic`
- `status`: `complete`
- `content_type`: `video/mp4` or `image/png` for WP2
- `preview_text`: short source-grounded summary
- `producer_metadata`: provider/model/render settings and job id
- `source_lineage`: selected source ids and retrieval/context metadata
- `version_metadata`: artifact version, previous version, and regeneration info
- `export_refs`: output artifact id, download URL, format, byte size, and
  content type

Failed jobs update the same pending artifact to `failed` and store a sanitized
reason in metadata.

## Frontend

Add two Studio output buttons:

- Video Overview
- Infographic

Both buttons use the same capability-aware disabled pattern as existing output
types. Starting generation creates a pending artifact card and polls the backend
job status.

Completed previews:

- Video Overview renders an HTML video player pointed at the output download
  URL, plus download/regenerate/discuss/save controls consistent with existing
  artifacts.
- Infographic renders an image preview with download/regenerate/discuss/save
  controls.

Avoid visible in-app explanatory copy beyond existing status/error text. The UI
should present these as normal work products, not as a tutorial.

## Error Handling

Fail before expensive work when possible:

- No selected/usable sources.
- Source context is empty.
- LLM unavailable.
- Slides unavailable for Video Overview.
- TTS unavailable for Video Overview.
- Image backend unavailable for Infographic.
- ffmpeg unavailable for Video Overview.
- Image generation request rejected by backend capability validation.

Partial video outputs are not exposed. A workspace artifact gets media preview
metadata only after final output artifact persistence succeeds.

Retryable backend failures remain retryable Jobs failures where existing Jobs
semantics allow it. Validation and missing-capability failures are final.

## Security And Storage

- Validate artifact type, output format, dimensions, selected sources, and
  workspace ownership at the API boundary.
- Keep secrets out of job payloads and artifact metadata.
- Use output artifact storage paths only; do not expose raw filesystem paths.
- Apply existing renderer limits for slide count, per-slide duration, total
  duration, and asset size.
- Register durable output artifacts for final media so previews do not expire.

## Tests

Backend:

- Request validation rejects unknown artifact types and empty source ids.
- Worker registration/dispatch test proves `research_workspace_output` jobs are
  acquired by the Research Workspace output worker path.
- Video worker test with mocked source context, slides, TTS, output artifact
  persistence, and direct render call.
- Infographic worker test with mocked source context, prompt generation, image
  adapter output, and durable output artifact persistence.
- Capability contract test for the new video/image capability ids.

Frontend:

- Studio renders Video Overview and Infographic buttons.
- Capability blocks disable the buttons and show the existing unavailable
  messaging pattern.
- Completed Video Overview artifact renders a video preview and download action.
- Completed Infographic artifact renders an image preview and download action.

Run the narrow Python and Vitest suites for touched backend/frontend files, then
Bandit on touched backend code before implementation completion.

## Risks

- The renderer currently creates still-frame slideshow videos, not animated
  motion graphics. That is acceptable for WP2 because the requested artifact is
  a narrated slideshow.
- TTS clip registration must create output artifacts; otherwise slide audio
  assets will not resolve.
- A backend endpoint that only submits Jobs but does not start/register a worker
  would leave WP2 artifacts permanently pending.
- Image artifact generation can be slow or backend-specific. Keep the first
  implementation to one generated image and existing backend settings.
- If source context is too large, the worker should reuse existing
  summarization/context limits rather than introducing a new retrieval stack.

## Implementation Notes

- Prefer adding a small `Research_Workspace` output jobs module over expanding
  the already-large Studio hook first.
- Reuse `CollectionsDatabase.create_output_artifact` for final video, final
  infographic, and per-slide narration clips.
- Add the Research Workspace output worker to the existing startup worker group
  or provide an equivalent default startup path.
- Reuse `render_presentation_video` instead of shelling out to ffmpeg directly.
- Reuse the image adapter/backend contract, but persist the final image as an
  output artifact after generation.
- Keep generated prompts/source excerpts in metadata concise to avoid bloating
  workspace artifacts.
