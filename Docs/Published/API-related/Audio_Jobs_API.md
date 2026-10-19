# Audio Jobs API

Submit and manage background audio processing jobs using the Jobs module.

Base path: `/api/v1/audio`

## Endpoints

- POST `/jobs/submit` - Submit an audio job
  - Body (JSON):
    - `url` (string, optional): Remote audio/video URL to download.
    - `local_path` (string, optional): Server-local absolute file path.
    - `model` (string, default `whisper-1`): Transcription model.
    - `perform_chunking` (bool, default true)
    - `perform_analysis` (bool, default false)
    - `api_name` (string, optional): LLM provider for analysis. Without it the analysis stage is skipped, and the payload stored as the `audio_store` stage job's result records `analysis_skipped: "no api_name"`.
  - Response: `{ id, uuid, domain, queue, job_type, status }`
  - Auth: Single-user X-API-KEY or multi-user JWT.

- GET `/jobs/{job_id}` - Get job status (owner or admin)
  - Path: `job_id` (int)
  - Response: Job row fields (`id, job_type, status, priority, retry_count, ...`).

- GET `/jobs/admin/list` - List jobs (admin)
  - Query: `status` (optional), `owner_user_id` (optional), `limit` (1-200)
  - Response: `{ jobs: [...] }`

- GET `/jobs/admin/summary` - Counts by status (admin)
  - Query: `owner_user_id` (optional)
  - Response: `{ counts_by_status, total, owner_user_id }`

- GET `/jobs/admin/owner/{owner_user_id}/processing` - Active processing (admin)
  - Path: `owner_user_id` (string/int)
  - Response: `{ owner_user_id, processing, limit }`

## Worker

- Service: runs the pipeline stages `audio_download → audio_convert → audio_transcribe → audio_chunk → audio_analyze → audio_store`.
- Each stage runs as its own job in the `audio` domain; the submit response carries only the first stage's id.
- Every stage, transcription included, runs in this worker on the `default` queue, unless `AUDIO_JOBS_GPU_TRANSCRIBE` sends transcription to the GPU worker (below). A stage that cannot queue the next one fails with `error_code` `next_stage_enqueue_failed` instead of completing with nothing after it. Only a refused queue fails it outright; any other refusal is retried by running the stage again, and the same failure twice in a row quarantines the job (`JOBS_QUARANTINE_THRESHOLD`).
- Analysis failures: these fail the `audio_analyze` stage job at once, with the reason in that job row's `error_code`: `missing_model`, `authentication`, `configuration`, `invalid_provider`, `adapter_unavailable`, `request_rejected`, `provider_quota_exhausted`, `llm_quota_exceeded` or `empty_transcript`. A missing API key shows up as `authentication` or `configuration`. A provider switched off with `LLM_ADAPTERS_NATIVE_HTTP_<PROVIDER>=0` shows up as `configuration`. `request_rejected` means the provider refused the request (an unknown model, a transcript too long for the model); `provider_quota_exhausted` means the provider account's billing quota is spent (OpenAI `insufficient_quota`). A rate limit (`rate_limit`) and any other provider failure (`provider_failure`, which includes a provider's transient 408 or 498 reply) are retried. The same failure twice in a row quarantines the job (`JOBS_QUARANTINE_THRESHOLD`, default 2).
- Env flags:
  - `AUDIO_JOBS_WORKER_ENABLED`: `true|false` (default false)
  - `AUDIO_JOBS_OWNER_STRICT`: enable owner-aware acquisition (default false)
  - `AUDIO_JOBS_GPU_TRANSCRIBE`: `true` queues `audio_transcribe` on the `transcribe` queue for the GPU worker instead of transcribing in this worker (default false). It is read by this worker, so set it where this worker runs, and only when a GPU worker reads the same Jobs database; otherwise transcription waits for a worker that never comes.
  - `JOBS_LEASE_SECONDS`: lease duration (default 120)

### GPU Worker (stub)

- A GPU-oriented worker stub is provided to process only the `audio_transcribe` stage on GPU nodes.
- Location: `tldw_Server_API/app/services/audio_transcribe_gpu_worker.py`
- Container: `Dockerfiles/Dockerfile.audio_gpu_worker`
- Behavior: acquires `audio` domain jobs from the `transcribe` queue and processes only `audio_transcribe`; it then queues the next stage on `default` for the CPU audio worker, which still runs every other stage.
- To use it, set `AUDIO_JOBS_GPU_TRANSCRIBE=true` where the audio jobs worker runs (the API server, or a standalone `python -m tldw_Server_API.app.services.audio_jobs_worker`). The GPU container must share the Jobs database (`JOBS_DB_URL`) and the filesystem behind the converted WAV files. The `transcribe` queue is registered for the `audio` domain.
- Upgrading: a deployment that made the GPU worker work by setting `JOBS_ALLOWED_QUEUES_AUDIO=transcribe` must now set `AUDIO_JOBS_GPU_TRANSCRIBE=true` as well, or transcription moves in-process; the allowlist setting is no longer needed.

## Quotas & Fairness

- Per-user concurrent job cap enforced both pre- and post-acquisition.
- Daily minutes quota is enforced by HTTP/WS paths (streaming + file) and recorded in `audio_usage_daily`.

## Notes

- This API complements synchronous `/audio/transcriptions` and real-time WS; it does not replace them.
- Audio Studio uses the Jobs module under the `audio_studio` domain for generation, render, and export work. Audiobook migration commit is currently a synchronous server-side transaction. See [Audio Studio](https://github.com/rmusser01/tldw_server/blob/main/Docs/Audio_Studio.md) for the project/revision API and artifact lifecycle.
