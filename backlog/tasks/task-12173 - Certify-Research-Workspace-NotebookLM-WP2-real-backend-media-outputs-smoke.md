---
id: TASK-12173
title: Certify Research Workspace NotebookLM WP2 real-backend media outputs smoke
status: Done
labels:
- research-workspace
- notebooklm
- wp2
- verification
ordinal: 12156
references:
- TASK-12160
- https://github.com/rmusser01/tldw_server/pull/2668
- https://github.com/rmusser01/tldw_server/pull/2669
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run a local real-backend smoke for Research Workspace WP2 media outputs after the closeout PR merged. Scope: confirm required local services/capabilities, generate one Infographic PNG and one Video Overview MP4 through Research Workspace, and verify preview/download paths. Do not change product code unless the smoke exposes a concrete defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local capability checks for API, ffmpeg, TTS readiness, image generation readiness, and configured LLM are recorded.
- [x] #2 One Infographic job completes with a durable PNG output artifact and authenticated preview/download works, or the exact blocker is recorded.
- [x] #3 One Video Overview job completes with a durable MP4 output artifact and authenticated preview/download works, or the exact blocker is recorded.
- [x] #4 No product-code changes are made unless a verified defect is found and separately documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Smoke environment:

- Worktree/branch: `.worktrees/research-workspace-wp2-smoke`, `codex/research-workspace-wp2-smoke`, based on `origin/dev` after PR #2668 merge commit `903a139a0c9044b0acbc15a29cf70b26d4999a87`.
- API server: `uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 18002`, `AUTH_MODE=single_user`, `SINGLE_USER_API_KEY=wp2-smoke-key-123456`.
- LLM: local llama.cpp at `http://127.0.0.1:9099/v1`, `/v1/models` returned `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`; `/health` returned `{"status":"ok"}`.
- ffmpeg: available, startup preflight resolved `/opt/homebrew/bin/ffmpeg` (`ffmpeg 8.1`).
- Source fixture: media id `1`, workspace `wp2-smoke`, source `src-wp2-smoke`; `/api/v1/media/1` returned `chunking_status: completed` and source text.
- Source readiness: `/api/v1/workspaces/wp2-smoke/sources/status` returned `partially_queryable`, reason `vector_index_pending`, with `metadata_ready`, `text_extracted`, `fts_ready`, `citation_ready`, and `tool_accessible` true.
- Research Workspace media-output capability gate: `/api/v1/research-workspace/capabilities` returned `status: degraded`.

Capability results:

- `source_browse`: ready/allow.
- `video_overview_generation`: degraded/warn, `reason_code: llm_degraded`.
- `image_generation`: unavailable/block, `reason_code: image_backend_unknown`.
- `infographic_generation`: unavailable/block, `reason_code: image_backend_unknown`.
- `export_download`: ready/allow.

Infographic result:

- No Infographic job was submitted because the production capability endpoint blocked image generation before job submission.
- Exact blocker: `infographic_generation` unavailable/block due to `image_backend_unknown`.
- Local config has `stable_diffusion_cpp` enabled, but no Stable Diffusion model path/model asset is configured in this worktree.

Video Overview result:

- Submitted `POST /api/v1/workspaces/wp2-smoke/outputs` with `artifact_type=video_overview`, `source_ids=["src-wp2-smoke"]`, provider `custom_openai_api`, and model `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`.
- API response: `202 Accepted`, job id `2`, artifact id `video_overview-03957fe75ef04edc9bd1c77c8e713218`.
- Initial API-only run left the job queued because `RESEARCH_WORKSPACE_OUTPUT_JOBS_WORKER_ENABLED` was not set for the API process.
- Started the production sidecar worker entrypoint manually with `run_research_workspace_output_jobs_worker`; the worker claimed job `2`, generated slides, and reached `generate_narration`.
- Job failed at TTS generation on both attempts. Worker log error: `Error generating speech with kitten_tts: KittenTTS generation failed`.
- Public status: `/api/v1/workspaces/wp2-smoke/outputs/2` returned `status: failed`, `progress_percent: 50.0`, `progress_message: generate_narration`; workspace artifact status `failed`, producer metadata error `tts_generation_failed`, top-level error `worker_exception`, `export_refs: []`.
- Exact blocker: local `kitten_tts` adapter initializes but fails during speech generation, so no narrated MP4 artifact/download could be produced.

Potential follow-up:

- The Research Workspace capability endpoint currently allowed/warned Video Overview because TTS readiness is config-derived. The smoke shows actual local TTS generation can fail after capability reports enough readiness to submit a job. A follow-up should harden TTS capability probing or expose a more conservative TTS readiness state for media-output generation.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
WP2 real-backend smoke completed against a local API, local llama.cpp, the real Research Workspace output API, and the production output worker. Infographic is blocked by missing image backend configuration (`image_backend_unknown`). Video Overview submission and slide generation reached the backend worker, but MP4 production is blocked by real TTS generation failure (`tts_generation_failed` from `kitten_tts`). No product code was changed.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed or blockers recorded with evidence.
- [x] #2 Smoke commands/results recorded.
- [x] #3 Bandit skipped with rationale: no backend/product code was changed; only this Backlog task record was updated.
- [x] #4 Final summary added.
<!-- DOD:END -->
