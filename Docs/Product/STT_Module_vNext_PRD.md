**STT Module vNext / Design Ledger**

- **Relationship to Existing Docs**
  - This document started as a *future-work* STT backlog. The major contract items in it are now implemented on `codex/stt-vnext-slice-1-config` and published in `Docs/Product/STT_Module_PRD.md`, `Docs/API-related/Audio_Transcription_API.md`, and the operator docs.
  - It now serves as design history plus a place to track any remaining rollout/tuning follow-ons.
  - It should be read alongside:
    - `Docs/Product/STT_Module_PRD.md` (current STT behavior and near-term goals).
    - `Docs/Product/Realtime_Voice_Latency_PRD.md` (end-to-end voice latency, WS TTS, and metrics).

- **Status**
  - Owner: Core Voice & API Team
  - Status: Partially implemented; canonical shipped behavior has moved into current-state product/API/operator docs.

- **Implementation Snapshot (2026-04-05)**
  - Shipped on the current branch:
    - WS control v2 with explicit `protocol_version=2` negotiation and bounded pause-queue handling.
    - Transcript run history in Media DB v2 with `latest_transcription_run_id`, `next_transcription_run_id`, `transcription_run_id`, and `idempotency_key`.
    - Deterministic final/full transcript diagnostics (`auto_commit`, `vad_status`, `diarization_status`, optional bounded `diarization_details`).
    - Tenant/global STT retention and redaction policy enforcement across REST, WS, and persistence.
    - Bounded `audio_stt_*` metrics families and read-path/write-result telemetry.
  - Remaining work is limited to rollout tuning, ops thresholds, and any new follow-on capabilities beyond these shipped contracts.

- **Product Decisions (Locked: 2026-02-09)**
  - WS protocol versioning is **mandatory** once control frames ship.
  - `latest_run_id` is global per `media_id` (not per provider/model).
  - PII redaction policy precedence is **tenant-level**.

- **Historical vNext Themes**
  - Richer WebSocket control/status protocol for STT.
  - Transcript versioning and run history in Media DB v2.
  - Enhanced streaming diagnostics (auto-commit / VAD / diarization flags).
  - Explicit retention/PII controls for audio and transcripts.
  - Finer-grained STT metrics series for dashboards.

- **1. Rich WS Control & Status Protocol**
  - Goal: allow clients to explicitly pause, resume, and stop streaming, with clear server acknowledgments and bounded backpressure behavior.
  - Versioning:
    - `v1` (legacy): existing top-level `auth` / `config` / `audio` / `commit` / `reset` / `stop` behavior.
    - `v2` (new): adds control envelope frames and state-machine semantics.
    - `v2` is opt-in via the initial `config` frame field `protocol_version=2`; if omitted, the server defaults to `v1`.
    - Server echoes the negotiated protocol version in the first success acknowledgment for the session.
    - Control frames are rejected without explicit `v2` negotiation.
  - Proposed client -> server control frames (v2):
    - JSON frames of the form: `{ "type": "control", "action": "pause" | "resume" | "stop" | "commit" }`.
    - `pause`: stop processing newly received audio and buffer it up to a bounded queue.
    - `resume`: continue processing queued audio in FIFO order according to backpressure policy.
    - `stop`: server flushes finals for already-processed audio, emits completion marker, drops any still-unprocessed paused queue, then closes WS.
    - `commit`: force a final transcript for already-processed audio in the current utterance; queued paused audio is not drained implicitly.
  - Proposed server -> client frames:
    - Status acknowledgments (v2): `{ "type": "status", "state": "configured" | "paused" | "resumed" | "closing", "protocol_version": 2 }` (idempotent where applicable).
    - Legacy reset acknowledgment remains `{ "type": "status", "state": "reset" }` in both versions.
    - Control errors: `{ "type": "error", "error_type": "invalid_control", "message": "..." }`.
    - Backpressure warnings: `{ "type": "warning", "warning_type": "audio_dropped_during_pause", "message": "..." }`.
  - Backpressure policy (v2 requirement):
    - Default paused-audio queue cap: 2.0 seconds (configurable).
    - Queue accounting is measured in submitted audio duration derived from configured input format/sample rate; if exact duration cannot be derived, use the same byte-to-seconds fallback used for quota accounting.
    - Overflow policy: `drop_oldest` (default), with warning emission rate-limited to once per 5 seconds.
  - Compatibility:
    - Existing v1 frames remain valid and unchanged, including top-level `reset`.
    - v1 sessions only receive legacy response frame types plus additive-safe transcript fields; new control-lifecycle `status` and `warning` frames are v2-only.
  - **Definition of Done**
    - Acceptance:
      - Control frames only accepted for `protocol_version=2`; otherwise return `invalid_control`.
      - `pause`/`resume` are idempotent and always ack with current state.
      - `stop` flushes finals and closes cleanly.
      - Paused queue is bounded and enforces documented overflow policy.
    - Tests:
      - Unit: control parser, invalid action/type handling, state transitions, queue overflow logic.
      - Integration: pause/resume with synthetic audio, commit during pause, stop flush semantics, mixed v1/v2 clients.
    - Metrics:
      - `audio_stt_streaming_sessions_started_total{provider}` increments on session open.
      - `audio_stt_streaming_sessions_ended_total{provider,session_close_reason}` increments on close.
      - `audio_stt_errors_total{endpoint,provider,reason}` records invalid control and overflow paths.
    - Rollback:
      - Feature flag disables v2 control handling and reverts all clients to v1 semantics.

- **2. Transcript Versioning & Run History**
  - Goal: make transcript history explicit and append-only, with a clear notion of runs and which run is effective/default per media item.
  - Proposed schema evolution for `Transcripts` (Media DB v2):
    - Introduce `transcription_run_id` and use `(media_id, transcription_run_id)` as logical run key (append-only).
    - Add nullable `idempotency_key` with uniqueness scoped to `(media_id, idempotency_key)` when present.
    - Track `Media.latest_transcription_run_id` and `Media.next_transcription_run_id` for global default selection and atomic run allocation.
    - Keep `whisper_model` (or normalized provider/model identifier) for indexing and compatibility.
    - Maintain existing soft-delete and sync/versioning semantics (`version`, `prev_version`, `deleted`, `merge_parent_uuid`).
  - Proposed ingestion contract changes:
    - Ingestion result JSON includes:
      - `latest_run_id`: globally current/default run id for the `media_id`.
      - `supersedes`: optional run id replaced by new run.
      - `idempotency_key`: optional client key to dedupe reruns safely.
    - Downstream consumers use `latest_run_id` when they need default transcript; advanced tools may navigate full run history.
  - Concurrency and idempotency rules:
    - New run ids are monotonically increasing per `media_id` and allocated by a per-media transactional counter (`Media.next_transcription_run_id` or equivalent).
    - Same `idempotency_key` for same `media_id` returns existing run without creating a new run.
    - Competing writes resolve by transaction order; final committed writer updates `latest_run_id`.
  - **Definition of Done**
    - Acceptance:
      - Multiple runs per media item can coexist.
      - `latest_run_id` resolves globally per `media_id`.
      - Idempotent reruns do not duplicate rows.
    - Tests:
      - Unit: run id allocator, idempotency key dedupe, supersedes validation.
      - Integration: concurrent reruns, rollback/retry paths, downstream retrieval by latest vs explicit run.
    - Metrics:
      - `audio_stt_run_writes_total{provider,write_result}` tracks `created`, `deduped`, `superseded`, and `failed` outcomes.
    - Rollback:
      - Read path can be switched back to legacy `(media_id, whisper_model)` selector via feature flag.

- **3. Enhanced Streaming Diagnostics**
  - Goal: give clients/operators more insight into why/when a final transcript was emitted and which features were active.
  - Final transcript diagnostics (v2 and v1 additive-safe):
    - `auto_commit`: `true` when final emitted by server-side VAD/turn detection; else `false`.
    - `vad_status`: one of `enabled`, `disabled`, `fail_open`.
    - `diarization_status`: one of `enabled`, `disabled`, `unavailable`.
    - Optional `diarization_details` when initialization/persistence fails. This must be a structured object with bounded `code` enum and optional safe summary string; raw exception text must never be emitted or used as metric label content.
  - Contract rule:
    - These fields are always present on final/full transcript frames once this feature is enabled, to keep client parsing deterministic.
  - **Definition of Done**
    - Acceptance:
      - Final frames always include `auto_commit`, `vad_status`, and `diarization_status`.
      - Status values match enumerated set.
    - Tests:
      - Unit: enum validation and payload composer defaults.
      - Integration: VAD enabled/disabled/fail-open and diarization enabled/unavailable scenarios.
    - Metrics:
      - `audio_stt_errors_total{endpoint,provider,reason}` tracks fail-open initialization reasons.
    - Rollback:
      - Feature flag removes extra diagnostic fields while preserving existing final-frame shape.

- **4. Retention & PII Controls**
  - Goal: provide first-class configuration for raw audio retention and optional transcript PII redaction.
  - Proposed environment/config flags (names subject to final design review):
    - `STT_DELETE_AUDIO_AFTER` (bool):
      - When `true` (recommended in production), delete persisted raw audio after successful transcription unless explicit retention is enabled.
    - `STT_AUDIO_RETENTION_HOURS` (int/float):
      - Max retention window for persisted raw audio; default `0` in production.
    - `STT_REDACT_PII` (bool or enum; experimental):
      - When enabled, redact configured PII categories before persistence and response emission.
  - Policy precedence (locked):
    - Tenant-level policy is authoritative.
    - Global defaults apply only when tenant policy is unset.
    - Request-level overrides may only be stricter than tenant policy, never weaker.
  - Strictness rules:
    - A request may shorten retention TTL or force delete-after-success, but may not lengthen retention beyond tenant policy.
    - A request may add PII categories to redact, but may not remove tenant-required categories or disable redaction when tenant policy requires it.
  - Required processing order:
    - 1) Transcribe -> 2) Normalize artifact -> 3) Apply tenant PII policy -> 4) Persist transcript -> 5) Build response -> 6) Apply audio retention/deletion policy.
  - Streaming rule:
    - WS final/full transcript frames must honor the same tenant redaction policy as REST responses.
    - Partial transcript frames may only bypass redaction if tenant policy explicitly allows unredacted partials; otherwise they are redacted before emission.
  - Documentation requirements:
    - Mark PII redaction as best-effort, not a hard compliance guarantee.
    - Document performance/accuracy implications for retrieval pipelines.
  - **Definition of Done**
    - Acceptance:
      - Tenant policy enforcement is consistent across REST, WS, and ingestion paths.
      - Retention TTL and delete-after behavior enforced deterministically.
    - Tests:
      - Unit: precedence resolution and category masking.
      - Integration: tenant policy application in transcript persistence and API output; retention TTL cleanup.
    - Metrics:
      - `audio_stt_errors_total{endpoint,provider,reason}` includes policy-enforcement failures.
      - `audio_stt_redaction_total{endpoint,redaction_outcome}` tracks redaction application outcomes with bounded enums.
    - Rollback:
      - Disable STT-specific redaction/retention feature flags and fall back to existing persistence behavior.

- **5. Fine-Grained STT Metrics**
  - Goal: complement existing latency metrics with request/session counters and queue/streaming latency histograms.
  - Proposed metrics (Prometheus-safe snake_case names; labels finalized via metrics policy below):
    - Counters:
      - `audio_stt_requests_total{endpoint,provider,model,status}`
      - `audio_stt_streaming_sessions_started_total{provider}`
      - `audio_stt_streaming_sessions_ended_total{provider,session_close_reason}`
      - `audio_stt_errors_total{endpoint,provider,reason}`
      - `audio_stt_run_writes_total{provider,write_result}`
      - `audio_stt_redaction_total{endpoint,redaction_outcome}`
    - Histograms:
      - `audio_stt_latency_seconds{endpoint,provider,model}`
      - `audio_stt_queue_wait_seconds{endpoint}`
      - `audio_stt_streaming_token_latency_seconds{provider,model}`
  - Interaction with existing metrics:
    - Additive to existing:
      - `stt_final_latency_seconds{model,variant,endpoint}`
      - `tts_ttfb_seconds{provider,voice,format}`
      - `voice_to_voice_seconds{provider,route}`
  - **Definition of Done**
    - Acceptance:
      - All proposed metric families are registered and emitted in every applicable STT path (REST, WS, or ingestion) without leaking unsupported labels into unrelated paths.
      - Label values conform to bounded enums and cardinality budget.
    - Tests:
      - Unit: registration idempotence and bounded-label mapping.
      - Integration: known success/failure/queue scenarios emit expected series.
    - Metrics:
      - Cardinality alerting enabled per policy budget thresholds.
    - Rollback:
      - Disable new metric emission via feature flag while preserving existing latency histograms.

- **6. WS State Machine & Message Contract (Required)**
  - Version negotiation:
    - Client proposes protocol version in the initial `config` frame.
    - If `protocol_version` is absent, server treats the session as `v1`.
    - `v2`-only control/status/warning semantics are unavailable until configuration succeeds.
  - State machine (v2 control path):

| Current State | Event | Next State | Server Behavior |
| --- | --- | --- | --- |
| `awaiting_config` | `config` valid (`protocol_version=1` or absent) | `running` | Configure session without v2 control semantics |
| `awaiting_config` | `config` valid (`protocol_version=2`) | `running` | Emit `status=configured` with `protocol_version=2` |
| `running` | `control.pause` | `paused` | Emit `status=paused` |
| `paused` | `control.pause` | `paused` | Emit `status=paused` (idempotent) |
| `running` | `control.resume` | `running` | Emit `status=resumed` (idempotent) |
| `paused` | `control.resume` | `running` | Emit `status=resumed` |
| `running` or `paused` | `control.commit` | unchanged | Emit `full_transcript` from already-processed audio only |
| `running` or `paused` | `reset` (legacy top-level) | `running` | Clear current transcript/diarization state and emit `status=reset` |
| `running` or `paused` | `control.stop` | `closing` | Flush finals, drop any still-unprocessed paused queue, emit completion marker, close |
| any | invalid `control.action` | unchanged | Emit `error_type=invalid_control` |
| any | WS disconnect | `closed` | Cleanup resources |

  - Message contract table:

| Direction | Frame | Allowed Versions | Allowed States | Notes |
| --- | --- | --- | --- | --- |
| C -> S | `{type:"config", protocol_version:1|2, ...}` | v1, v2 | `awaiting_config` | If `protocol_version` omitted, default to `1` |
| C -> S | `{type:"audio", data:"..."}` | v1, v2 | `running`, `paused` | In `paused`, data is queued subject to cap/overflow policy |
| C -> S | `{type:"commit"}` | v1, v2 | `running`, `paused` | Legacy alias; equivalent to v2 `control.commit` |
| C -> S | `{type:"reset"}` | v1, v2 | `running`, `paused` | Legacy top-level reset retained for compatibility |
| C -> S | `{type:"stop"}` | v1, v2 | `running`, `paused` | Legacy alias; equivalent to v2 `control.stop` |
| C -> S | `{type:"control", action:"pause|resume|commit|stop"}` | v2 only | per state machine | Rejected with `invalid_control` if v2 not negotiated |
| S -> C | `{type:"status", state:"reset"}` | v1, v2 | any | Existing legacy reset acknowledgment |
| S -> C | `{type:"status", state:"configured|paused|resumed|closing", protocol_version:2}` | v2 only | any | v2 lifecycle/control status frames |
| S -> C | `{type:"warning", warning_type:"audio_dropped_during_pause", ...}` | v2 | `paused` | Rate-limited warning on overflow |
| S -> C | `{type:"error", error_type:"invalid_control", ...}` | v1, v2 | any | Structured control error |
| S -> C | `{type:"full_transcript", ...}` | v1, v2 | `running`, `paused` | Final transcript payload (plus diagnostics when enabled) |

- **7. Transcript Run-History Migration Plan**
  - DDL phase (non-breaking):
    - Add `Transcripts.transcription_run_id` (nullable initially).
    - Add `Transcripts.supersedes_run_id` (nullable).
    - Add `Transcripts.idempotency_key` (nullable).
    - Add `Media.latest_transcription_run_id` (nullable).
    - Add `Media.next_transcription_run_id` with default `1`.
    - Add unique index on `(media_id, transcription_run_id)` for active rows.
    - Add unique index on `(media_id, idempotency_key)` when `idempotency_key IS NOT NULL`.
  - Backfill phase:
    - Existing transcript rows are grouped by `media_id`, ordered deterministically by `created_at`, `last_modified`, then `id`, and assigned distinct `transcription_run_id` values `1..N`.
    - `Media.latest_transcription_run_id` is initialized to the backfilled row that matches the current effective/default transcript policy; absent a stronger existing signal, use the most recently modified active row.
    - `Media.next_transcription_run_id` is initialized to `max(transcription_run_id) + 1` per `media_id`.
    - `supersedes_run_id` remains null for backfilled rows unless explicit provenance already exists.
    - Backfill is resumable and idempotent, tracked by migration progress table.
  - Dual-write phase:
    - New writes populate both legacy compatibility fields and run-history fields.
    - Run ids assigned transactionally via `Media.next_transcription_run_id` (or equivalent per-media allocator).
    - `supersedes_run_id` set when reruns replace previous default.
    - `idempotency_key` lookup/insert occurs in the same transaction as run creation.
  - Dual-read phase:
    - Primary read path: `Media.latest_transcription_run_id`.
    - Fallback 1: highest active `transcription_run_id` per `media_id`.
    - Fallback 2 (temporary): legacy `(media_id, whisper_model)` path only for callers that explicitly request model-scoped selection.
    - Emit explicit fallback counter for readiness tracking.
  - Cutoff criteria:
    - 100% backfill completion.
    - 0 fallback-2 reads for 7 consecutive days.
    - Error budget for transcript retrieval unchanged from baseline.
  - Post-cutoff:
    - Remove legacy fallback path.
    - Keep `whisper_model` for indexing/reporting compatibility.
  - Rollback strategy:
    - Flip read-path feature flag to legacy selector.
    - Keep new columns/indexes in place (no destructive rollback).

- **8. Metrics Label Policy & Cardinality Budget**
  - Label value policy (bounded enums):
    - `endpoint`: one of `audio.transcriptions`, `audio.stream.transcribe`, `audio.chat.stream`, `ingestion`.
    - `status`: one of `ok`, `quota_exceeded`, `bad_request`, `provider_error`, `model_unavailable`, `internal_error`.
    - `reason`: one of `auth`, `quota`, `provider_error`, `model_unavailable`, `invalid_control`, `validation_error`, `timeout`, `internal`.
    - `session_close_reason`: one of `client_stop`, `client_disconnect`, `server_shutdown`, `error`.
    - `write_result`: one of `created`, `deduped`, `superseded`, `failed`.
    - `redaction_outcome`: one of `applied`, `not_requested`, `skipped`, `failed`.
    - `provider`: normalized lowercase provider key from configured registry.
    - `model`: normalized allowlisted model slug; unknown values bucketed to `other`.
  - Prohibited:
    - Free-text labels, exception messages, raw diarization/policy error strings, user IDs, request IDs, arbitrary error strings.
  - Cardinality budget:
    - Total active series across `audio_stt_*` <= 1000 per deployment environment.
    - Per metric family target <= 300 active series.
    - Alert at 80% of either threshold; block release if thresholds exceeded in staging.
  - Change governance:
    - Any new label key or enum value requires metrics-team review and PRD update.

- **9. Out-of-Scope / To Be Designed Elsewhere**
  - Breaking protocol changes for existing WS clients (outside v2 negotiation scope) are out of scope; changes must go through dedicated compatibility design.
  - Major schema changes to Media DB v2 beyond transcript run history should be captured in a dedicated DB migration PRD/design.
  - Full PII/compliance frameworks (classification engines, policy orchestration) are separate projects; this doc tracks a minimal STT-specific control surface.

- **10. Next Steps**
  - Convert each theme above into scheduled stages in task-specific implementation plans (for example `IMPLEMENTATION_PLAN_stt_ws_control_v2.md`) with owner, ETA, and test gate.
  - Open focused design docs derived from this PRD:
    - `Transcript Versioning Design`
    - `STT WS Control v2 Design`
    - `STT Retention and PII Policy Design`
    - `STT Metrics Rollout Design`
  - When implementation begins, update:
    - `STT_Module_PRD.md` with high-level requirements.
    - The design docs with API/schema migration details and release playbooks.
