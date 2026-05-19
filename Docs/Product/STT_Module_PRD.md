**STT Module PRD v1.0**

- **Meta**
  - Owner: Core Voice & API Team
  - Status: Current-state contract updated for the STT vNext rollout on `codex/stt-vnext-slice-1-config`
  - Implementation Progress:
    - Provider registry + adapters implemented (`stt_provider_adapter.py`) and used by REST `/api/v1/audio/transcriptions`, ingestion persistence, and Jobs (CPU and GPU workers).
    - Normalized STT artifact shape in place (`to_normalized_stt_artifact`, adapter `transcribe_batch`) and used internally by REST, `/media/add` ingestion, and Jobs; transcripts now persist as append-only run-history rows with `Transcripts.transcription_run_id`, optional `idempotency_key`, and `Media.latest_transcription_run_id` / `Media.next_transcription_run_id` tracking the effective/default run.
    - Unified batch helpers for ingestion and Jobs (`run_stt_batch_via_registry`, `run_stt_job_via_registry`) wired into `perform_transcription` and `audio_jobs_worker`/`audio_transcribe_gpu_worker`; Stage 5 hardening/release artifacts are published in `Docs/Product/STT_Module_Release_Report_20260207.md`.
    - WebSocket STT control v2 is implemented behind explicit `protocol_version=2` negotiation for `/api/v1/audio/stream/transcribe` and `/api/v1/audio/chat/stream`, while legacy top-level `commit` / `reset` / `stop` remain supported.
    - Final/full transcript frames now carry deterministic diagnostics (`auto_commit`, `vad_status`, `diarization_status`, optional bounded `diarization_details`).
    - Tenant-aware STT retention/redaction policy is implemented for REST, WS, and persistence paths, with org admin routes in multi-user mode and global config defaults in single-user mode.
    - Bounded `audio_stt_*` metrics are registered and emitted for request, session, error, redaction, write-result, read-path, and latency visibility.

- **Project Summary**
  - Speech-to-Text (STT) powers `/api/v1/audio/transcriptions` and `/api/v1/audio/stream/transcribe`.
  - Current providers live under `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/` and include faster-whisper (CPU/GPU), NVIDIA NeMo (Parakeet/Canary), and Qwen2Audio.
  - The STT module should unify these providers, expose consistent REST/WebSocket behaviors, and integrate cleanly with Jobs, Media ingestion, Embeddings, and AuthNZ.

- **Problem Statement**
  Contributors need a single, predictable STT subsystem that accepts uploaded or streamed audio, selects the appropriate provider, and yields normalized transcripts plus metadata suitable for RAG pipelines. Current state: REST + WS parity exists, WS control v2 and deterministic transcript diagnostics are available, transcript run history is explicit in Media DB v2, tenant-aware retention/redaction policy is enforced, and bounded STT metrics are wired. Remaining gaps are limited to rollout tuning, operational thresholds, and follow-on enhancements tracked in the known-issues list and vNext ledger.

- **Goals**
  - Keep turn detection/auto-commit in unified WS STT deterministic and observable (fail-open if VAD unavailable).
  - Add phoneme/lexicon overrides for Kokoro (per-request/provider/global precedence).
  - Add optional WS TTS endpoint with backpressure/auth parity.
  - Keep current docs aligned with the shipped STT control, persistence, policy, and metrics contracts.
  - Keep unified provider interface and deterministic REST/WS behavior with normalized transcript artifacts (timestamps, diarization, confidence, language detection), explicit run history, and existing observability intact.

- **Non-Goals**
  - No SIP ingestion, telephony gateways, or mobile SDKs in this release.
  - No training or fine-tuning workflows; assume pre-trained models.
  - No automatic summarization of transcripts (handled by downstream RAG).

- **Primary Users**
  - Backend contributors extending STT or adding providers.
  - Worker authors (GPU/CPU) using the Jobs/worker helpers (informally the Worker SDK).
  - QA engineers writing API and WebSocket tests.
  - DevOps configuring deployments and monitoring.

- **Use Cases**
  - Upload or URL-based transcription returning transcript + job artifact.
  - Live streaming transcription with partial results and controllable latency.
  - Background jobs triggered by media ingestion (priority queue).
  - Benchmarking/evaluation across providers (accuracy, latency).
  - Replay transcripts for downstream embeddings and RAG.

- **Functional Requirements**
  - Provider abstraction located in `app/core/Ingestion_Media_Processing/Audio` implements batch, stream, diarization, and capability discovery.
  - REST STT endpoint (`/api/v1/audio/transcriptions`) handles multipart uploads (OpenAI-compatible); URL-based and yt-dlp flows remain in media ingestion endpoints that call into the shared STT module.
  - Streaming endpoints support partial/interim updates and final transcripts, accept legacy top-level `auth` / `config` / `audio` / `commit` / `reset` / `stop` messages, and also support WS control v2 via `protocol_version=2` plus `{"type":"control","action":"pause|resume|commit|stop"}`. v2-only `status` / `warning` frames are additive and gated by explicit negotiation.
  - Transcript schema includes segments with timestamps, speaker labels, confidence, optional diarization, usage stats, and deterministic final-frame diagnostics (`auto_commit`, `vad_status`, `diarization_status`, optional bounded `diarization_details`).
  - Persist transcripts and metadata into Media DB v2 as explicit run history: normalized STT artifacts (including `text`, `segments`, `language`, `diarization`, `usage`, `metadata`) serialized into `Transcripts.transcription`, with `transcription_run_id`, optional `idempotency_key`, and `supersedes_run_id` on transcript rows plus `Media.latest_transcription_run_id` / `Media.next_transcription_run_id` tracking the effective/default run. `whisper_model` is retained for compatibility/indexing. Segments/chunks are written via `MediaChunks` / `UnvectorizedMediaChunks`; ingestion result JSON continues to carry transcript content/segments in the established shape and may include `normalized_stt` for internal consumers.
  - Config-driven defaults with request-level overrides (bounded validation).
  - Rate limiting via RG ingress policies plus per-provider thresholds.
  - Retry/backoff policy with provider-specific fatal vs retriable errors; fallbacks must not mask auth/quota/validation failures and should be configurable per environment.
  - Metrics and audit: register bounded `audio_stt_*` counters/histograms and emit the request/session/error/write-result/redaction/read-path families in current REST/WS/ingestion paths. `audio_stt_queue_wait_seconds` and `audio_stt_streaming_token_latency_seconds` are reserved for paths that can compute those timings without guesswork. Keep existing `stt_final_latency_seconds`, `tts_ttfb_seconds`, and `voice_to_voice_seconds` documented.
  - Failure modes: fail-open when VAD/diarization unavailable (log once, continue streaming) rather than blocking; record a clear metric/log label when operating in fail-open mode so latency/quality are interpreted correctly.
  - Worker SDK updates: strict worker_id/lease_id enforcement, explicit completion tokens, provider context metadata.

- **Non-Functional Requirements**
  - Performance: on the reference setup described in `Docs/Product/STT_Module_IMPLEMENTATION_PLAN.md`, target p50 <10s latency for a 5-minute clip on GPU-backed models; target p50 <250ms end-of-speech → partial-result latency for unified WS streaming. Reference setup is: 8-core CPU, optional NVIDIA GPU (Parakeet GPU path), macOS 14 or Ubuntu 22.04, Python 3.11, ffmpeg ≥6.0, av ≥11.0.0, localhost loopback, 10s 16 kHz float32 single-speaker fixture with 250 ms trailing silence.
  - Accuracy: allow measurement hooks for WER; prefer baseline comparisons or non-blocking checks over hard per-provider thresholds in unit tests.
  - Scalability: multi-worker queue processing, horizontal scaling with Jobs; avoid single-node bottlenecks.
  - Resilience: fallback from GPU to CPU or alternate provider with clear logging.
  - Security: authenticated endpoints (API key/JWT), MIME/type validation, file size limits, sanitized metadata.
  - Compatibility: Python 3.10+, works on macOS/Linux/Windows; GPU use optional.
  - Observability: structured logging (loguru), trace correlation, optional OpenTelemetry when available.

- **Architecture Overview**
  1. **API Layer (`app/api/v1/endpoints/audio.py`)**
     - Validates requests, resolves provider, triggers Jobs or direct processing.
     - Shared schemas in `schemas/audio_schemas.py` to keep OpenAPI consistent.
  2. **Service Layer (`app/services/audio_transcribe_*` and helpers in `app/core/Ingestion_Media_Processing/Audio/`)**
     - Provider selection, chunking, diarization, metadata assembly.
     - Works with Jobs manager for background processing; streaming path handled via WebSocket handler.
  3. **Providers (modules such as `Audio_Transcription_Lib.py`, `Audio_Transcription_Nemo.py`, `Audio_Transcription_External_Provider.py`)**
     - Implement provider logic; expose capability metadata, config parsing, health checks.
  4. **Workers (`app/services/audio_jobs_worker.py`, `audio_transcribe_gpu_worker.py`)**
     - Poll Jobs queue, enforce leases, call providers, push transcripts, emit metrics.
  5. **Storage & Integration**
     - Media DB (`Media_DB_v2`) for transcript storage, `UnvectorizedMediaChunks` for pending embedding data.
     - Embeddings pipeline triggered on completion when configured.
     - Audit bridge logs creation/completion/failure events when audit enabled.
  6. **Observability**
     - Metrics: counters, histograms; logs with request IDs; optional audit records.

- **Provider Specifications**
  - Provider registry with configuration metadata (name, model, streaming support, hardware requirements), backed by a concrete adapter interface (e.g., `SttProviderAdapter` in `app/core/Ingestion_Media_Processing/Audio`).
  - Standard adapter methods (final names to match the implementation): synchronous/async batch entry point (e.g., `transcribe_batch`), streaming entry point (e.g., `transcribe_stream` or equivalent chunk handler), optional `detect_language`, and `healthcheck` for readiness.
  - Config via config loader (`core/config`) reading `.env` and `config.txt` `[STT-Settings]`, with defaults for beam size, temperature, chunk length, and device/variant selection per provider.
  - Fallback order: try preferred provider, then a bounded, explicitly configured fallback chain depending on capabilities and error type (never fallback on auth/quota/validation errors).

- **Data Contracts**
  - **Transcription Response**
    ```
    {
      "text": str,
      "language": str,
      "segments": [
        {"start": float, "end": float, "speaker": str | None, "confidence": float | None, "text": str}
      ],
      "diarization": {"enabled": bool, "speakers": int | None},
      "usage": {"duration_ms": int, "tokens": int | None},
      "metadata": {"provider": str, "model": str, "queue_latency_ms": int, ...}
    }
    ```
  - Internal normalized artifact: this response shape is used inside ingestion/Jobs as the normalized STT transcript; the public OpenAI-compatible `/api/v1/audio/transcriptions` endpoint continues to use the `OpenAITranscriptionResponse` schema in `audio_schemas.py`.
  - **Streaming Events**: partial/interim results (e.g., `partial`/`transcription`), final/full transcripts (e.g., `final`/`full_transcript`), `error` frames, and additive `status` / `warning` frames. In v2 sessions, `status=configured|paused|resumed|closing` and `warning_type=audio_dropped_during_pause` are part of the public contract. Final/full transcript payloads always include `auto_commit`, `vad_status`, and `diarization_status`, plus bounded `diarization_details` when relevant.
  - **Job Payload**: audio reference, provider override, diarization flag, chunk config, audit context.
  - **Database Fields**: transcripts stored in `Transcripts.transcription` with append-only run-history fields (`transcription_run_id`, `supersedes_run_id`, `idempotency_key`) and `Media.latest_transcription_run_id` / `Media.next_transcription_run_id` for default-run resolution. `Media.transcription_model` remains a compatibility/reporting field; segments/chunks persist via `MediaChunks` / `UnvectorizedMediaChunks` when chunking is enabled.

- **Configuration & Deployment**
  - Process environment: provider API keys and STT-related env flags (e.g., `STT_SKIP_AUDIO_PREVALIDATION`, transcript-cache bounds); see `tldw_Server_API/Config_Files/README.md` and `Docs/Published/Env_Vars.md` for the curated list. Optional `.env` / `Config_Files/.env` files may be loaded via the existing config/secret helpers.
  - `Config_Files/config.txt` `STT-Settings` section: default provider/transcriber, diarization boolean, streaming toggles, buffering and model-variant settings, and the vNext controls exposed by `get_stt_config()` (`ws_control_v2_enabled`, `paused_audio_queue_cap_seconds`, `overflow_warning_interval_seconds`, `transcript_diagnostics_enabled`, `delete_audio_after_success`, `audio_retention_hours`, `redact_pii`, `allow_unredacted_partials`, `redact_categories`).
  - Feature flags: `STT_WS_CONTROL_V2_ENABLED`, `STT_TRANSCRIPT_DIAGNOSTICS_ENABLED`, `STT_DELETE_AUDIO_AFTER_SUCCESS`, `STT_AUDIO_RETENTION_HOURS`, `STT_REDACT_PII`, `STT_ALLOW_UNREDACTED_PARTIALS`, and `STT_DEBUG_DUMP_AUDIO` (debug dumping is dev/test only, writes into a configurable debug directory, and must remain disabled in shared/production deployments).
  - In multi-user mode, org admins can override the global STT policy via `/api/v1/admin/orgs/{org_id}/stt/settings`; in single-user mode, global config defaults are authoritative.
  - Deployment checklist: FFmpeg installed, CUDA (optional), model weights downloaded, Jobs workers configured, network egress allowed for remote providers.

- **Testing Strategy**
  - Unit tests: provider adapters (success, failure, retry, parameter validation), chunking helpers, metadata cleaning (no empty arrays).
  - Integration tests: REST upload using small fixtures, Jobs pipeline end-to-end, WebSocket streaming (use deterministic mock provider).
  - Property tests: ensure timestamps monotonic, segments cover audio duration, transcripts normalized (no control chars).
  - Contract tests: DB writes align with schema; ensure transcripts survive round-trip.
  - Load tests (optional): evaluate throughput under concurrency.
  - CI: default to mock providers; allow opt-in runs with real providers (guarded by env).
  - Manual QA: real audio on GPU, long-form streaming with network interruptions, diarization accuracy sampling.

- **Metrics & Observability**
  - Counters: `audio_stt_requests_total`, `audio_stt_streaming_sessions_started_total`, `audio_stt_streaming_sessions_ended_total`, `audio_stt_errors_total`, `audio_stt_run_writes_total`, `audio_stt_redaction_total`, and `audio_stt_transcript_read_path_total` via the central metrics manager.
  - Histograms: `audio_stt_latency_seconds`, plus registered `audio_stt_queue_wait_seconds` / `audio_stt_streaming_token_latency_seconds` families for paths that compute those timings, alongside `stt_final_latency_seconds{model,variant,endpoint}`, `tts_ttfb_seconds{provider,voice,format}`, and `voice_to_voice_seconds{provider,route}`.
  - Gauges: active workers, GPU memory usage (if available).
  - Logs: include request/job ID, provider, timing, errors.
  - Audit: create, updated, failed transcripts recorded when audit bridge enabled.
  - Alerts: high error rate, backlog growth, extreme latency, no transcripts returned.

- **Security & Privacy**
  - Input validation: MIME/type check, duration limit, safe temporary storage, optional virus scan hook.
  - Authentication: enforce API key/JWT per AuthNZ mode.
  - Secret handling: never log raw keys; sanitize provider responses.
  - Data retention: delete-after-success, retention-hours, and transcript redaction are active STT policy controls. Effective policy resolves as org override > global default in multi-user mode, and global default only in single-user mode. Request-level overrides may only be stricter than the effective policy.
  - Multi-tenant: respect user IDs in path sanitization, separate storage directories.

- **Dependencies & Integration Points**
  - Jobs manager (enqueue, lease, completion).
  - Worker SDK (lease enforcement, metrics).
  - Media ingestion & DB (store transcripts).
  - Embeddings service (optional follow-up).
  - AuthNZ settings (single user vs multi).
  - Filesystem for temp storage; FFmpeg/yt-dlp for preprocessing.
  - Config loader (`core/config`) for provider settings.

- **Milestones**
  1. **M1 - Turn Detection/Auto-Commit**: Silero VAD-driven turn detection in unified WS STT; auto-commit finals within target latency on reference audio (fail-open if VAD missing).
     - **Acceptance**: On the reference fixture, p50 `stt_final_latency_seconds` for unified WS STT ≤ 600ms with VAD enabled; fail-open mode emits a clear metric/log label and does not regress quotas/auth.
     - **Status**: Implemented in unified WS path (fail-open); needs real-world threshold tuning + doc polish.
  2. **M2 - Phoneme/Lexicon Overrides**: configurable pronunciation maps for Kokoro with safe defaults and precedence rules; demo request shows changed pronunciation.
     - **Acceptance**: Given a documented test phrase and phoneme map, Kokoro output changes as expected in at least one integration fixture; added latency from applying overrides is ≤ 5% vs baseline on the reference setup.
     - **Status**: Implemented. See `tldw_Server_API/app/core/TTS/phoneme_overrides.py`, `tldw_Server_API/tests/TTS/test_phoneme_overrides.py`, and docs in `Docs/User_Guides/WebUI_Extension/TTS_Getting_Started.md`.
 3. **M3 - Optional WS TTS**: `/api/v1/audio/stream/tts` with backpressure/auth/quota parity and PCM streaming; passes slow-reader/disconnect tests. Ownership split with TTS PRD.
     - **Acceptance**: p50 `tts_ttfb_seconds` over WS ≤ 200ms on the reference setup; slow-reader and disconnect tests complete without resource leaks and emit `audio_stream_underruns_total`/error metrics as expected.
     - **Status**: Implemented with protocol/runbook/sign-off artifacts. See `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`, `tldw_Server_API/tests/Audio/test_ws_tts_endpoint.py`, `Docs/Audio_Streaming_Protocol.md`, `Docs/Operations/Audio_Streaming_Backpressure_Runbook.md`, and `Docs/Product/STT_TTS_WS_TTS_SIGNOFF_20260207.md`.
 4. **M4 - Docs & Harness**: refreshed STT/TTS docs plus a lightweight voice-to-voice latency harness consuming existing metrics; documented CLI/outputs.
     - **Acceptance**: `Helper_Scripts/voice_latency_harness/run.py --out out.json --short` (or equivalent) runs against the reference setup and outputs JSON with at least p50/p90 for `stt_final_latency_seconds`, `tts_ttfb_seconds`, and `voice_to_voice_seconds`; short mode suitable for CI; docs reference the harness and the VAD/metrics knobs.
     - **Status**: Implemented. See `Helper_Scripts/voice_latency_harness/run.py`, `Helper_Scripts/voice_latency_harness/README.md`, and sample output `Docs/Product/stt_stage4_voice_latency_harness_sample_20260207.jsonc`.
 5. **M5 - Production Hardening & Release Readiness**: release-report closure, known-issues publication, rollback playbook, and operations/support handoff.
    - **Acceptance**: All four artifacts exist and are cross-linked from the execution tracker; known issues include severity + workaround + owner; rollback includes concrete STT/WS/TTS actions with validation steps.
    - **Status**: Implemented. See `Docs/Product/STT_Module_Release_Report_20260207.md`, `Docs/Product/STT_Module_Known_Issues_20260207.md`, `Docs/Operations/Audio_Streaming_Backpressure_Runbook.md`, and `Docs/Audio_STT_Module.md`.

- **Risks & Mitigations**
  - Provider downtime/unavailability → multi-provider fallbacks, local deterministic mock, error escalation.
  - GPU resource contention → configurable concurrency, CPU fallback, preflight health checks.
  - Large audio causing timeouts → enforce max duration, chunk processing with partial saves, send progress updates.
  - Streaming resource leaks → ensure queue shutdown, WebSocket cleanup, tests verifying teardown.
  - Sensitive data in transcripts → provide configuration for redaction or manual review before storage.

- **Open Questions**
  1. Should prompt biasing/custom vocabulary be part of v1 or deferred?
  2. How aggressively do we support diarization (default on/off, provider coverage)?
  3. Where should transcripts live when multiple transformations occur (versioning, delta storage)?
  4. Do we ship sample models/providers as dev dependencies or expect manual installation?
  5. How do we expose quality metrics (WER) to contributors-CI dashboards, logs, or manual scripts?

- **Documentation Deliverables**
  - `/Docs/Audio_STT_Module.md`: architecture, provider setup, API/WS diagrams, troubleshooting.
  - Sample worker scripts (e.g., `Helper_Scripts/Samples/Jobs/batch_worker_example.py` as a template for STT workers).
  - Integration examples for REST & WebSocket clients.
  - Troubleshooting matrix (common errors, diagnostics, remediation).
  - Migration notes for legacy Gradio STT flows.

- **Future Work & vNext**
  - Remaining follow-up work beyond the current implementation is tracked in `Docs/Product/STT_Module_vNext_PRD.md` as a design ledger and rollout backlog. The canonical current contract for shipped WS control v2, transcript run history, deterministic diagnostics, STT policy enforcement, and bounded `audio_stt_*` metrics is this PRD plus `Docs/API-related/Audio_Transcription_API.md`.
