## Stage 1: Provider Abstraction & Config Unification
**Goal**: Unify faster‑whisper, NeMo (Parakeet/Canary), and Qwen2Audio behind a single STT provider layer with consistent configuration.
**Success Criteria**: A clear adapter/registry in `app/core/Ingestion_Media_Processing/Audio` (e.g., `SttProviderAdapter`) is used by REST STT, media ingestion, and Jobs to select providers; STT defaults and overrides come from `get_stt_config()` / `[STT-Settings]`; capability discovery (batch/stream/diarization) is test‑covered.
**Tests**:
- Unit: adapter methods for each provider (success/failure paths, retry & fallback behavior, config parsing, capability flags).
- Integration: `/api/v1/audio/transcriptions` exercises at least two providers; a Jobs‑driven transcription run uses the shared abstraction.
**Status**: In Progress

**Current Progress**:
- Implemented `SttProviderAdapter` + registry in `stt_provider_adapter.py` with capability metadata and a `transcribe_batch` API for faster‑whisper, Parakeet, Canary, Qwen2Audio, and external providers.
- REST `/api/v1/audio/transcriptions` now resolves providers via the registry and delegates batch work through adapters (still returning the same OpenAI‑compatible response shapes).
- Media ingestion uses the registry and a unified helper (`run_stt_batch_via_registry`) inside `perform_transcription`; ingestion persistence (`persist_primary_av_item`) upserts transcripts into `Transcripts` via `upsert_transcript` and attaches a normalized STT artifact to `process_result`.
- Jobs (`audio_jobs_worker`, `audio_transcribe_gpu_worker`) now delegate STT to the shared registry helpers (`run_stt_job_via_registry`), and their `audio_transcribe` stages populate both legacy `text`/`segments` fields and a normalized STT artifact on the job payload.

## Stage 2: REST/DB/Jobs Contract Hardening
**Goal**: Make REST STT and media ingestion share a normalized transcription artifact and persist it consistently into Media DB v2.
**Success Criteria**: A single normalized STT result shape (text, segments, diarization, usage, metadata) is produced by the STT module and used across REST, ingestion, and Jobs; transcripts land in `Transcripts.transcription` keyed by `(media_id, whisper_model)` with `Media.transcription_model` tracking the effective/default model; segments/chunks are written to `MediaChunks`/`UnvectorizedMediaChunks`; ingestion result JSON reflects this shape and is documented.
**Tests**:
- Integration: media ingestion (audio URLs/files) writes expected rows into `Transcripts`, `MediaChunks`, and `UnvectorizedMediaChunks`, and returns a result JSON matching the documented contract.
- Contract: round‑trip tests that reload a transcript from DB and compare to the original normalized artifact (within expected lossy fields such as floating‑point timestamps).
**Status**: In Progress

**Current Progress**:
- Normalized STT artifact helper (`to_normalized_stt_artifact`) is implemented and used by adapters and ingestion to represent STT results in a single internal shape.
- REST `/api/v1/audio/transcriptions` and batch ingestion paths now produce/consume this normalized artifact internally, while preserving their existing public response JSON.
- `persist_primary_av_item` upserts transcripts into `Transcripts` (via `upsert_transcript`) with `whisper_model` derived from the registry, and stores the normalized artifact on `process_result["normalized_stt"]`; chunking continues through the existing `chunks` → `UnvectorizedMediaChunks`/`MediaChunks` pipeline.
- Jobs (CPU and GPU workers) attach the normalized artifact to their `audio_transcribe` payloads (`normalized_stt`, without the `text` and `segments` the payload already carries), and dedicated DB round‑trip/contract tests for the normalized artifact remain the primary outstanding item for this stage.

## Stage 3: Streaming STT Latency & Quotas (M1 Alignment)
**Goal**: Finalize unified WS STT (turn detection + quotas) to meet latency and fairness requirements while failing open safely when VAD is unavailable.
**Success Criteria**: Silero VAD‑based auto‑commit is wired into unified WS STT as described in `STT-IMPLEMENTATION_PLAN.md` Stage 1; on the reference fixture, p50 `stt_final_latency_seconds{endpoint="audio_unified_ws"}` ≤ 600 ms with VAD enabled; WS `/api/v1/audio/stream/transcribe` remains backward compatible (auth/config/audio/commit only), enforces per‑user concurrent streams and daily minutes, and emits clear logs/metrics when running in fail‑open mode (no VAD).
**Tests**:
- Unit: VAD threshold/min‑silence/turn‑stop edge cases; mapping VAD end‑of‑speech to final transcripts; quota helpers (streams and jobs), including DB failure paths.
- Integration: WS streams with synthetic audio (pauses, silence, long clips) assert final latency, absence of duplicate finals, and correct quota/rate‑limit error behavior; metrics registry shows non‑zero `stt_final_latency_seconds` entries with expected labels.
**Status**: In Progress (auto‑commit wiring and latency metrics largely implemented per `STT-IMPLEMENTATION_PLAN.md`; threshold tuning and compatibility tests outstanding)

## Stage 4: TTS‑Adjacent Features (Kokoro Overrides & WS TTS)
**Goal**: Implement Kokoro pronunciation overrides and optional WS TTS endpoint with behavior aligned to the existing REST TTS API and latency metrics.
**Success Criteria**: Kokoro phoneme/lexicon overrides are loaded from a config file, applied safely (word boundaries, case handling), and can be toggled per‑request/provider/global; a documented sample shows changed pronunciation for a test phrase with ≤ 5% added latency on the reference setup. If enabled, `/api/v1/audio/stream/tts` exists (coordinated with TTS PRD), mirrors REST TTS semantics, respects auth/quota/backpressure parity with WS STT, and records `tts_ttfb_seconds` and `audio_stream_underruns_total` appropriately.
**Tests**:
- Unit: phoneme mapping correctness (boundaries, case, overlaps, invalid entries); Kokoro adapter behavior with and without overrides.
- Integration: REST PCM TTS path remains stable; WS TTS (when enabled) passes slow‑reader, disconnect, and quota tests; latency metrics for `tts_ttfb_seconds` update as expected.
**Status**: Complete (Kokoro overrides + WS TTS parity shipped; see `tldw_Server_API/app/core/TTS/phoneme_overrides.py`, `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`, and `Docs/Product/STT_TTS_WS_TTS_SIGNOFF_20260207.md`)

## Stage 5: Metrics Harness, Docs & Operational Hardening (M4 Alignment)
**Goal**: Provide a small voice‑to‑voice latency harness and refreshed docs so contributors and operators can validate STT/TTS metrics and behavior end‑to‑end.
**Success Criteria**: A harness under `Helper_Scripts/voice_latency_harness/` (or similar) can be run on the reference setup to produce JSON with at least p50/p90 for `stt_final_latency_seconds`, `tts_ttfb_seconds`, and `voice_to_voice_seconds`; a `--short` mode is suitable for CI; docs (`/Docs/Audio_STT_Module.md`, API docs) describe STT providers, REST/WS contracts, VAD knobs, quotas, and how to interpret latency metrics and common alerts.
**Tests**:
- Harness: manual and (optionally) automated runs of the harness in short mode that verify JSON output structure and non‑zero metrics; basic failure handling (server unavailable, auth errors).
- Docs: link/lint checks; targeted integration tests that perform one REST STT call and one WS STT session and assert that key metrics and logs are emitted.
**Status**: Complete (harness schema, sample artifact, and docs/troubleshooting shipped; see `Helper_Scripts/voice_latency_harness/run.py`, `Helper_Scripts/voice_latency_harness/README.md`, and `Docs/Product/stt_stage4_voice_latency_harness_sample_20260207.jsonc`)
