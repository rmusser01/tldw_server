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
    - `api_name` (string, optional): LLM provider for analysis.
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
- Env flags:
  - `AUDIO_JOBS_WORKER_ENABLED`: `true|false` (default false)
  - `AUDIO_JOBS_OWNER_STRICT`: enable owner-aware acquisition (default false)
  - `JOBS_LEASE_SECONDS`: lease duration (default 120)

### GPU Worker (stub)

- A GPU-oriented worker stub is provided to process only the `audio_transcribe` stage on GPU nodes.
- Location: `tldw_Server_API/app/services/audio_transcribe_gpu_worker.py`
- Container: `Dockerfiles/Dockerfile.audio_gpu_worker`
- Behavior: acquires `audio` domain jobs and processes only `audio_transcribe`; other stages are re-queued with a short backoff for CPU workers.

## Quotas & Fairness

- Per-user concurrent job cap enforced both pre- and post-acquisition.
- Daily minutes quota is enforced by HTTP/WS paths (streaming + file) and recorded in `audio_usage_daily`.

## Notes

- This API complements synchronous `/audio/transcriptions` and real-time WS; it does not replace them.
- Audio Studio uses the Jobs module under the `audio_studio` domain for generation, render, and export work. Audiobook migration commit is currently a synchronous server-side transaction. See [Audio Studio](https://github.com/rmusser01/tldw_server/blob/main/Docs/Audio_Studio.md) for the project/revision API and artifact lifecycle.
