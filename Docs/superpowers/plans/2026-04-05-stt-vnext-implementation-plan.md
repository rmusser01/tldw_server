# STT vNext Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gap between `STT_Module_vNext_PRD.md` and the current STT implementation by landing transcript run history, WS control v2, deterministic diagnostics, tenant-level retention/redaction policy, and bounded STT metrics.

**Architecture:** Do not ship this as one large branch. Execute it as sequential PR slices where shared STT config lands first, then Media DB transcript run-history, then WS protocol v2 and diagnostics, then tenant retention/redaction, then metrics and docs. Reuse the existing Media DB migration framework, current audio endpoints, `generated_files` index, and internal Prometheus-style metrics registry rather than inventing a second STT stack.

**Tech Stack:** FastAPI, WebSocket endpoints, Media DB v2 (SQLite/PostgreSQL), AuthNZ org settings, `config.txt` + env overrides, Loguru, pytest, Bandit

---

## Scope Lock

Keep these decisions fixed during implementation:

- `latest_run_id` is global per `media_id`
- WS control frames are `v2` only and require explicit `protocol_version=2` in the initial `config` frame
- legacy top-level WS `reset` remains supported
- final/full transcript frames must always carry deterministic diagnostics once the diagnostics slice lands
- tenant-level STT policy is authoritative
- request-level overrides may only be stricter than tenant policy
- new STT metrics must use snake_case Prometheus-safe names and bounded labels only
- do not remove legacy transcript reads until fallback telemetry shows the migration is safe
- do not implement all five themes in one PR

## Execution Preconditions

1. Create an isolated git worktree/branch before implementation so subagents do not collide with unrelated local changes.
2. Activate the project virtual environment before every `python`, `pytest`, or `bandit` command: `source .venv/bin/activate` from the repo root, or `source ../../.venv/bin/activate` from a `.worktrees/<branch>` checkout.
3. Keep one PR slice per branch or stacked branch. Do not assign overlapping write sets to concurrent workers.
4. Treat "tenant" as `org_id` in multi-user mode. In single-user mode, there is no tenant row; STT policy resolves from global config defaults only.

## Recommended Delivery Order

1. Shared STT config/feature-flag plumbing
2. Transcript run history and Media DB migration
3. WS control v2 and state-machine behavior
4. Final-frame diagnostics contract
5. Tenant retention/redaction policy and cleanup
6. Metrics rollout, docs, and release gates

## Safe Parallelism Windows

- After Slice 1 lands, one worker may own Slice 2 while another owns Slice 5 Step 1-4, because the DB files and AuthNZ files are disjoint.
- Keep Slices 3 and 4 on the same branch or stacked branches because they share the WS streamer files.
- Keep Slice 6 last because it depends on stable labels and telemetry points from Slices 2, 3, and 5.

## File Structure

- `Docs/Product/STT_Module_vNext_PRD.md`
  Purpose: source PRD for locked behavior, DoD, migration policy, and metrics policy.
- `Docs/Product/STT_Module_PRD.md`
  Purpose: current-state PRD that must absorb shipped vNext behavior once implementation lands.
- `Docs/API-related/Audio_Transcription_API.md`
  Purpose: public REST + WS contract documentation for `/api/v1/audio/transcriptions` and `/api/v1/audio/stream/transcribe`.
- `Docs/Operations/Env_Vars.md`
  Purpose: operator-facing settings reference for WS protocol, retention, and metrics toggles.
- `Docs/Deployment/Monitoring/Metrics_Cheatsheet.md`
  Purpose: operator-facing metrics contract and dashboard label guidance.
- `tldw_Server_API/app/core/config.py`
  Purpose: canonical STT config parsing from `[STT-Settings]` and env overrides.
- `tldw_Server_API/app/core/config_sections/stt.py`
  Purpose: new focused STT config loader for vNext flags and bounded defaults.
- `tldw_Server_API/app/core/config_sections/__init__.py`
  Purpose: export the new STT config section.
- `tldw_Server_API/app/core/DB_Management/migrations/023_transcript_run_history.sql`
  Purpose: packaged SQLite migration script for file-backed Media DB upgrades to transcript run-history.
- `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`
  Purpose: Media DB schema version bump, fresh bootstrap schema, and migration registration.
- `tldw_Server_API/app/core/DB_Management/media_db/legacy_reads.py`
  Purpose: central read-path resolution for latest/default transcript run and legacy fallback behavior.
- `tldw_Server_API/app/core/DB_Management/media_db/legacy_transcripts.py`
  Purpose: preserve legacy helper compatibility while dual-writing run-history fields.
- `tldw_Server_API/app/core/DB_Management/media_db/api.py`
  Purpose: package-level transcript read facade that must keep exposing stable helpers during the migration window.
- `tldw_Server_API/app/core/DB_Management/media_db/schema/migrations.py`
  Purpose: register PostgreSQL migration bodies for new Media DB schema version.
- `tldw_Server_API/app/core/DB_Management/media_db/schema/migration_bodies/__init__.py`
  Purpose: export the new PostgreSQL transcript run-history migration body.
- `tldw_Server_API/app/core/DB_Management/media_db/schema/migration_bodies/postgres_transcript_run_history.py`
  Purpose: new PostgreSQL migration body for transcript run-history columns/indexes/backfill helpers.
- `tldw_Server_API/app/core/DB_Management/media_db/schema/sqlite_post_core_structures.py`
  Purpose: SQLite table/column ensure path for non-breaking transcript run-history additions.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
  Purpose: ingestion persistence path for transcript writes and default-run updates.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py`
  Purpose: shared STT WebSocket processing, final transcript emission, diagnostics, and v2 control behavior.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/ws_control_protocol.py`
  Purpose: new helper for control parsing, protocol negotiation, pause-queue accounting, and state transitions.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_policy.py`
  Purpose: new helper for tenant/request STT retention + PII policy resolution and enforcement.
- `tldw_Server_API/app/core/Audio/streaming_service.py`
  Purpose: websocket auth/session context plumbing that must carry enough identity to resolve org policy on WS paths.
- `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
  Purpose: WS endpoint integration, config negotiation, auth/quota glue, and metrics emission.
- `tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py`
  Purpose: REST STT retention/redaction enforcement and request metrics.
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
  Purpose: existing org-policy resolver that should be reused for audio REST/WS policy resolution, including single-user fallback semantics.
- `tldw_Server_API/app/core/AuthNZ/migrations.py`
  Purpose: SQLite AuthNZ schema additions for org-level STT settings.
- `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
  Purpose: PostgreSQL AuthNZ schema additions for org-level STT settings.
- `tldw_Server_API/app/core/AuthNZ/repos/org_stt_settings_repo.py`
  Purpose: new tenant policy repo for org-level STT retention/redaction settings.
- `tldw_Server_API/app/core/AuthNZ/repos/generated_files_repo.py`
  Purpose: durable artifact index for retained STT audio and cleanup metadata.
- `tldw_Server_API/app/api/v1/schemas/org_team_schemas.py`
  Purpose: request/response schemas for org-scoped STT policy endpoints.
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_orgs.py`
  Purpose: admin API for tenant-level STT settings.
- `tldw_Server_API/app/services/admin_orgs_service.py`
  Purpose: org-scoped STT policy service logic, backend selection, and admin access enforcement.
- `tldw_Server_API/app/core/Metrics/metrics_manager.py`
  Purpose: register new `audio_stt_*` metric families.
- `tldw_Server_API/app/core/Metrics/stt_metrics.py`
  Purpose: new bounded-label mapping helpers and one place to emit STT counters/histograms safely.
- `tldw_Server_API/tests/DB_Management/test_media_db_legacy_transcripts.py`
  Purpose: lock legacy transcript helper compatibility.
- `tldw_Server_API/tests/Sharing/test_clone_service.py`
  Purpose: protect latest-run fallback and clone behavior when run pointers are stale.
- `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingestion_audio_transcripts.py`
  Purpose: update ingestion expectations from one transcript per model to multiple runs per media item.
- `tldw_Server_API/tests/Audio/test_ws_control_protocol_v2.py`
  Purpose: new tests for WS v2 pause/resume/commit/stop behavior.
- `tldw_Server_API/tests/Audio/test_ws_vad_turn_detection.py`
  Purpose: existing final transcript coverage that must be extended for diagnostics.
- `tldw_Server_API/tests/Audio/test_ws_diarization_persistence_status.py`
  Purpose: existing diarization-status coverage to extend for bounded details.
- `tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py`
  Purpose: new tests for REST/WS policy enforcement and artifact cleanup.
- `tldw_Server_API/tests/Audio/test_stt_vnext_config_flags.py`
  Purpose: new tests for STT vNext config parsing and defaults.
- `tldw_Server_API/tests/Audio/test_ws_metrics_audio.py`
  Purpose: existing WS metrics coverage to extend for `audio_stt_*`.
- `tldw_Server_API/tests/Metrics/test_audio_stt_metrics.py`
  Purpose: new metrics contract tests for bounded labels and expected emission.
- `tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py`
  Purpose: enforce the cardinality budget with the new STT metric families.
- `tldw_Server_API/tests/AuthNZ_SQLite/test_org_stt_settings_repo_sqlite.py`
  Purpose: new SQLite tests for org-level STT settings CRUD and precedence.
- `tldw_Server_API/tests/AuthNZ/unit/test_admin_orgs_service_backend_selection.py`
  Purpose: keep org STT settings service behavior consistent across SQLite and PostgreSQL paths.
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_generated_files_repo_backend_selection.py`
  Purpose: preserve generated-files repo backend compatibility if STT retention adds a new file category/source feature.
- `tldw_Server_API/tests/AuthNZ_Unit/test_org_policy_from_principal.py`
  Purpose: lock single-user synthetic-org fallback and org-claim precedence for audio policy resolution.
- `tldw_Server_API/tests/AuthNZ_Postgres/test_org_stt_settings_pg.py`
  Purpose: verify org STT settings ensure/migration behavior on the PostgreSQL AuthNZ path.
- `tldw_Server_API/tests/Admin/test_admin_orgs_stt_settings.py`
  Purpose: new admin API tests for tenant-level STT settings.

## PR Slice 1: Shared STT Config Surface

### Task 1: Add Canonical STT vNext Config Parsing

**Files:**
- Create: `tldw_Server_API/app/core/config_sections/stt.py`
- Modify: `tldw_Server_API/app/core/config_sections/__init__.py`
- Modify: `tldw_Server_API/app/core/config.py`
- Test: `tldw_Server_API/tests/Audio/test_stt_vnext_config_flags.py`

- [ ] **Step 1: Write the failing config test**

```python
def test_stt_vnext_defaults_are_bounded(config_parser):
    config_parser["STT-Settings"] = {}
    cfg = load_stt_config(config_parser, env={})
    assert cfg.ws_control_v2_enabled is False
    assert cfg.paused_audio_queue_cap_seconds == 2.0
    assert cfg.delete_audio_after_success is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_stt_vnext_config_flags.py -v`
Expected: FAIL because `load_stt_config(...)` does not exist yet.

- [ ] **Step 3: Implement the config loader**

Implement a small `STTConfig` dataclass and loader in `tldw_Server_API/app/core/config_sections/stt.py` that reads:

- WS v2 enable flag
- paused queue cap seconds
- overflow warning interval seconds
- transcript diagnostics enable flag
- delete-after-success default
- retention hours default
- redaction default

Prefer `[STT-Settings]` values with optional env overrides. Keep names aligned with the PRD and document any unavoidable renames.

- [ ] **Step 4: Thread the loader into the central config exports**

Make `tldw_Server_API/app/core/config.py` and `tldw_Server_API/app/core/config_sections/__init__.py` expose the parsed STT config in the same style as other config sections.

- [ ] **Step 5: Re-run the test**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_stt_vnext_config_flags.py -v`
Expected: PASS

- [ ] **Step 6: Run Bandit on the touched config scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/config.py \
  tldw_Server_API/app/core/config_sections/stt.py \
  -f json -o /tmp/bandit_stt_config.json
```

Expected: no new findings in changed code.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/config_sections/stt.py \
  tldw_Server_API/app/core/config_sections/__init__.py \
  tldw_Server_API/app/core/config.py \
  tldw_Server_API/tests/Audio/test_stt_vnext_config_flags.py
git commit -m "feat: add canonical stt vnext config loader"
```

## PR Slice 2: Transcript Run History and Media DB Migration

### Task 2: Add Run-History Schema, Backfill, and Dual-Write

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/migrations/023_transcript_run_history.sql`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/legacy_reads.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/legacy_transcripts.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/api.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/schema/migrations.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/schema/migration_bodies/__init__.py`
- Create: `tldw_Server_API/app/core/DB_Management/media_db/schema/migration_bodies/postgres_transcript_run_history.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/schema/sqlite_post_core_structures.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Test: `tldw_Server_API/tests/Sharing/test_clone_service.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_legacy_transcripts.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_transcripts_upsert.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingestion_audio_transcripts.py`

- [ ] **Step 1: Write failing DB tests**

Write tests for:

- two existing transcript rows for one `media_id` backfill to distinct run ids
- file-backed SQLite DBs upgrade cleanly via the packaged migration script
- same `idempotency_key` returns the same run
- `latest_transcription_run_id` tracks the global default run
- `get_latest_transcription(...)` prefers `Media.latest_transcription_run_id` and falls back deterministically when absent
- legacy helper still returns a stable payload shape

```python
def test_upsert_transcript_allocates_new_run_ids_for_distinct_idempotency_keys():
    created = create_transcript_run(db, media_id=media_id, idempotency_key="a")
    rerun = create_transcript_run(db, media_id=media_id, idempotency_key="b")
    assert created["transcription_run_id"] == 1
    assert rerun["transcription_run_id"] == 2
```

- [ ] **Step 2: Run the DB tests and verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sharing/test_clone_service.py \
  tldw_Server_API/tests/DB_Management/test_media_db_legacy_transcripts.py \
  tldw_Server_API/tests/DB_Management/test_media_transcripts_upsert.py \
  tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py \
  tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py \
  -v
```

Expected: FAIL because the new columns, allocator, and helpers do not exist.

- [ ] **Step 3: Implement schema changes**

Add to `Transcripts`:

- `transcription_run_id`
- `supersedes_run_id`
- `idempotency_key`

Add to `Media`:

- `latest_transcription_run_id`
- `next_transcription_run_id`

Requirements:

- bump `MediaDatabase._CURRENT_SCHEMA_VERSION`
- add a packaged SQLite migration script under `app/core/DB_Management/migrations/`
- register a new PostgreSQL migration body
- export the new migration body from `schema/migration_bodies/__init__.py`
- ensure fresh DB bootstrap includes the new columns
- ensure SQLite existing DBs get non-breaking `ALTER TABLE` coverage
- drop/replace the current `UNIQUE (media_id, whisper_model)` constraint
- create partial/filtered unique indexes where supported, including one run id per media and optional idempotency-key dedupe
- backfill distinct run ids per existing transcript row for one `media_id`
- seed `next_transcription_run_id` to one greater than the highest assigned run id per `media_id`

- [ ] **Step 4: Implement dual-write and dual-read helpers**

In `legacy_transcripts.py`, `legacy_reads.py`, `api.py`, and `persistence.py`:

- keep legacy callers working
- allocate monotonic run ids transactionally per `media_id`
- set `latest_transcription_run_id`
- dedupe by `(media_id, idempotency_key)` when present
- preserve `whisper_model` for legacy/reporting compatibility
- centralize read fallback in `legacy_reads.py`; do not duplicate legacy-order fallback in downstream callers
- emit structured `legacy_transcript_fallback` telemetry whenever a read misses `latest_transcription_run_id`

- [ ] **Step 5: Update integration expectations**

Change ingestion tests so they no longer assume “at most one primary transcript per media/model.” Replace that with:

- at least one transcript row exists
- the latest/default run can be resolved
- reruns do not corrupt the original row

- [ ] **Step 6: Run the targeted tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sharing/test_clone_service.py \
  tldw_Server_API/tests/DB_Management/test_media_db_legacy_transcripts.py \
  tldw_Server_API/tests/DB_Management/test_media_transcripts_upsert.py \
  tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py \
  tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingestion_audio_transcripts.py \
  -v
```

Expected: PASS

- [ ] **Step 7: Run Bandit on the touched DB scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/DB_Management/media_db \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  -f json -o /tmp/bandit_stt_run_history.json
```

Expected: no new findings in changed code.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/migrations/023_transcript_run_history.sql \
  tldw_Server_API/app/core/DB_Management/media_db \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  tldw_Server_API/tests/Sharing/test_clone_service.py \
  tldw_Server_API/tests/DB_Management/test_media_db_legacy_transcripts.py \
  tldw_Server_API/tests/DB_Management/test_media_transcripts_upsert.py \
  tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py \
  tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingestion_audio_transcripts.py
git commit -m "feat: add transcript run history to media db"
```

## PR Slice 3: WS Control v2

### Task 3: Implement WS Protocol Negotiation, State Machine, and Backpressure

**Files:**
- Create: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/ws_control_protocol.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
- Test: `tldw_Server_API/tests/Audio/test_ws_control_protocol_v2.py`
- Test: `tldw_Server_API/tests/Audio/test_ws_invalid_json_error.py`
- Test: `tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py`

- [ ] **Step 1: Write failing WS v2 tests**

Cover:

- v2 negotiation is rejected or downgraded when the config flag is off
- `protocol_version=2` negotiation
- `control.pause`, `control.resume`, `control.commit`, `control.stop`
- idempotent `pause` and `resume`
- `invalid_control` when `v2` was not negotiated
- top-level `reset` still works

```python
def test_control_pause_requires_v2(ws_client):
    ws_client.send_json({"type": "config", "sample_rate": 16000})
    ws_client.send_json({"type": "control", "action": "pause"})
    frame = ws_client.receive_json()
    assert frame["type"] == "error"
    assert frame["error_type"] == "invalid_control"
```

- [ ] **Step 2: Run the WS tests and verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Audio/test_ws_control_protocol_v2.py \
  tldw_Server_API/tests/Audio/test_ws_invalid_json_error.py \
  -v
```

Expected: FAIL because the protocol helper and new frames do not exist.

- [ ] **Step 3: Implement a focused WS control helper**

`ws_control_protocol.py` should own:

- protocol version negotiation from the initial `config` frame
- bounded pause-queue accounting in seconds
- overflow policy (`drop_oldest`)
- idempotent state transitions
- mapping legacy aliases (`commit`, `stop`, `reset`) without breaking current sessions
- a reusable parser/state layer that `audio_streaming.py` can apply to both `/audio/stream/transcribe` and `/audio/chat/stream`

Do not bury the state machine directly inside `Audio_Streaming_Unified.py`.

- [ ] **Step 4: Wire the helper into both audio streaming surfaces**

Update the shared streamer and endpoint wrapper so:

- `/api/v1/audio/stream/transcribe` supports v2
- `/api/v1/audio/chat/stream` either adopts the same helper or explicitly rejects `protocol_version=2` until parity lands
- do not assume `/audio/chat/stream` inherits the transcribe control path automatically; it has its own nested config/turn state machine today
- `ws_control_v2_enabled=False` keeps the server on v1-only behavior by default
- v1 sessions keep legacy behavior and do not receive v2-only status/warning frames

- [ ] **Step 5: Re-run the targeted WS tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Audio/test_ws_control_protocol_v2.py \
  tldw_Server_API/tests/Audio/test_ws_invalid_json_error.py \
  tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py \
  -v
```

Expected: PASS

- [ ] **Step 6: Run Bandit on the touched WS scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/ws_control_protocol.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py \
  -f json -o /tmp/bandit_stt_ws_control.json
```

Expected: no new findings in changed code.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/ws_control_protocol.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py \
  tldw_Server_API/tests/Audio/test_ws_control_protocol_v2.py \
  tldw_Server_API/tests/Audio/test_ws_invalid_json_error.py \
  tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py
git commit -m "feat: add stt websocket control protocol v2"
```

## PR Slice 4: Diagnostics Contract

### Task 4: Make Final Transcript Diagnostics Deterministic

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
- Test: `tldw_Server_API/tests/Audio/test_ws_vad_turn_detection.py`
- Test: `tldw_Server_API/tests/Audio/test_ws_diarization_persistence_status.py`
- Test: `tldw_Server_API/tests/Audio/test_audio_streaming_truthiness_flags.py`

- [ ] **Step 1: Write/extend failing diagnostics tests**

Lock these invariants:

- `auto_commit` always present on final/full transcript frames
- `vad_status` always present and one of `enabled|disabled|fail_open`
- `diarization_status` always present and one of `enabled|disabled|unavailable`
- `diarization_details` is structured and bounded, not raw exception text

- [ ] **Step 2: Run the diagnostics tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Audio/test_ws_vad_turn_detection.py \
  tldw_Server_API/tests/Audio/test_ws_diarization_persistence_status.py \
  tldw_Server_API/tests/Audio/test_audio_streaming_truthiness_flags.py \
  -v
```

Expected: FAIL because the current payload only partially exposes these fields.

- [ ] **Step 3: Implement bounded diagnostics emission**

In the full/final transcript emission path:

- always set `auto_commit` to `True` or `False`
- derive `vad_status` from actual detector state, including fail-open cases
- derive `diarization_status` from initialization/runtime availability
- only emit structured `diarization_details` with a bounded `code` plus optional short summary

- [ ] **Step 4: Re-run the diagnostics tests**

Run the same command from Step 2.
Expected: PASS

- [ ] **Step 5: Run Bandit on the touched diagnostics scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py \
  -f json -o /tmp/bandit_stt_diagnostics.json
```

Expected: no new findings in changed code.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py \
  tldw_Server_API/tests/Audio/test_ws_vad_turn_detection.py \
  tldw_Server_API/tests/Audio/test_ws_diarization_persistence_status.py \
  tldw_Server_API/tests/Audio/test_audio_streaming_truthiness_flags.py
git commit -m "feat: add deterministic stt streaming diagnostics"
```

## PR Slice 5: Tenant Retention and PII Policy

### Task 5: Add Tenant Policy Storage and Admin API

**Files:**
- Create: `tldw_Server_API/app/core/AuthNZ/repos/org_stt_settings_repo.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/org_team_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_orgs.py`
- Modify: `tldw_Server_API/app/services/admin_orgs_service.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_org_stt_settings_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_admin_orgs_service_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ_Postgres/test_org_stt_settings_pg.py`
- Test: `tldw_Server_API/tests/Admin/test_admin_orgs_stt_settings.py`

- [ ] **Step 1: Write failing org-settings tests**

Lock:

- create/update/get org STT settings
- unset values fall back to global defaults
- multi-user mode treats tenant policy as org-scoped policy
- request payload validation forbids weaker-than-tenant overrides
- single-user mode has no org policy row and relies on global defaults only
- PostgreSQL ensure/migration path creates the same STT settings table/indexes as SQLite

- [ ] **Step 2: Run the org-settings tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/AuthNZ_SQLite/test_org_stt_settings_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ/unit/test_admin_orgs_service_backend_selection.py \
  tldw_Server_API/tests/AuthNZ_Postgres/test_org_stt_settings_pg.py \
  tldw_Server_API/tests/Admin/test_admin_orgs_stt_settings.py \
  -v
```

Expected: FAIL because the repo, table, and admin route do not exist.

- [ ] **Step 3: Implement the tenant settings storage**

Add a dedicated STT settings table keyed by `org_id` in multi-user mode, not a free-form JSON blob hidden in unrelated settings. Do not invent a separate tenant identifier in this repo unless the broader AuthNZ model changes first.

Suggested columns:

- `org_id`
- `delete_audio_after_success`
- `audio_retention_hours`
- `redact_pii`
- `allow_unredacted_partials`
- `redact_categories_json`

- [ ] **Step 4: Add the schemas, service methods, and admin API**

Add request/response schemas in `org_team_schemas.py`, service methods in `admin_orgs_service.py`, and `GET` plus `PUT/PATCH` handlers to `admin_orgs.py` following the same shape as other org-scoped settings routes.

- [ ] **Step 5: Re-run the org-settings tests**

Run the same command from Step 2.
Expected: PASS

- [ ] **Step 6: Run Bandit on the touched AuthNZ/admin scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/AuthNZ \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_orgs.py \
  tldw_Server_API/app/services/admin_orgs_service.py \
  -f json -o /tmp/bandit_stt_org_policy.json
```

Expected: no new findings in changed code.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/repos/org_stt_settings_repo.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/api/v1/schemas/org_team_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_orgs.py \
  tldw_Server_API/app/services/admin_orgs_service.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_org_stt_settings_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ/unit/test_admin_orgs_service_backend_selection.py \
  tldw_Server_API/tests/AuthNZ_Postgres/test_org_stt_settings_pg.py \
  tldw_Server_API/tests/Admin/test_admin_orgs_stt_settings.py
git commit -m "feat: add org-level stt policy settings"
```

### Task 6: Enforce STT Policy in REST, WS, and Persistence Paths

**Files:**
- Create: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_policy.py`
- Modify: `tldw_Server_API/app/core/Audio/streaming_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/generated_files_repo.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Test: `tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py`
- Test: `tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_org_policy_from_principal.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_generated_files_repo_backend_selection.py`

- [ ] **Step 1: Write failing policy-enforcement tests**

Cover:

- stricter request override accepted
- weaker request override rejected
- single-user mode falls back to global STT defaults with no org lookup
- REST response text is redacted when tenant policy requires it
- WS final/full transcript frames follow the same redaction rule
- partial transcript behavior matches `allow_unredacted_partials`
- delete-after-success produces deterministic cleanup behavior
- retained raw audio is indexed through `generated_files` before TTL-based retention is allowed

- [ ] **Step 2: Run the policy-enforcement tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py \
  tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_org_policy_from_principal.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_generated_files_repo_backend_selection.py \
  -v
```

Expected: FAIL because there is no shared STT policy helper yet.

- [ ] **Step 3: Implement `stt_policy.py`**

Centralize:

- tenant policy lookup
- global default fallback
- stricter-only request merge
- transcript redaction application
- artifact retention decision (`delete now`, `retain until cutoff`)
- reuse the existing single-user synthetic-org fallback semantics instead of inventing a second audio-only tenant resolver

Do not duplicate this logic separately in REST and WS handlers.

- [ ] **Step 4: Wire the helper into REST, WS, and ingestion persistence**

Make sure:

- REST responses redact before response serialization
- WS final/full transcript frames redact before emission
- persisted transcript text is redacted before persistence when policy requires it
- WS auth/session context carries enough identity to resolve org policy with the same precedence as HTTP
- retained raw-audio cleanup metadata is recorded in one place via `generated_files`
- if `generated_files` indexing cannot be completed in the slice, retention TTL stays behind a hard guard and `delete_audio_after_success=True` remains enforced

- [ ] **Step 5: Re-run the policy tests**

Run the same command from Step 2.
Expected: PASS

- [ ] **Step 6: Run Bandit on the touched STT policy scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py \
  tldw_Server_API/app/api/v1/API_Deps/auth_deps.py \
  tldw_Server_API/app/core/Audio/streaming_service.py \
  tldw_Server_API/app/core/AuthNZ/repos/generated_files_repo.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_policy.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  -f json -o /tmp/bandit_stt_policy.json
```

Expected: no new findings in changed code.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_policy.py \
  tldw_Server_API/app/core/Audio/streaming_service.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py \
  tldw_Server_API/app/api/v1/API_Deps/auth_deps.py \
  tldw_Server_API/app/core/AuthNZ/repos/generated_files_repo.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py \
  tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_org_policy_from_principal.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_generated_files_repo_backend_selection.py
git commit -m "feat: enforce tenant stt retention and redaction policy"
```

## PR Slice 6: Metrics and Rollout

### Task 7: Register and Emit the New `audio_stt_*` Metrics

**Files:**
- Create: `tldw_Server_API/app/core/Metrics/stt_metrics.py`
- Modify: `tldw_Server_API/app/core/Metrics/metrics_manager.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Test: `tldw_Server_API/tests/Metrics/test_audio_stt_metrics.py`
- Test: `tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py`
- Test: `tldw_Server_API/tests/Audio/test_ws_metrics_audio.py`

- [ ] **Step 1: Write failing metrics tests**

Lock:

- registration of the new metric families
- bounded mapping of `endpoint`, `provider`, `model`, `status`, `reason`, `session_close_reason`, `write_result`, `redaction_outcome`
- `unknown`/unallowlisted models bucket to `other`
- active-series cap stays inside the PRD budget
- existing STT latency metrics do not continue exposing raw request model identifiers after the slice lands

- [ ] **Step 2: Run the metrics tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Metrics/test_audio_stt_metrics.py \
  tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py \
  tldw_Server_API/tests/Audio/test_ws_metrics_audio.py \
  -v
```

Expected: FAIL because the new STT metric families are not registered.

- [ ] **Step 3: Add a focused STT metrics helper**

`stt_metrics.py` should own:

- enum normalization
- bucketing unknown models to `other`
- safe wrappers around `increment_counter(...)` and `observe_histogram(...)`
- one place to emit run-write, session, redaction, and error metrics

- [ ] **Step 4: Register and emit metrics**

Add the new metric families to `metrics_manager.py` and instrument:

- REST transcription success/failure
- WS session start/end and control errors
- transcript run-history writes
- transcript read-path resolution and legacy fallback usage
- redaction outcomes
- normalize or deprecate any existing STT metric label paths that still emit raw `model` or `variant` values

- [ ] **Step 5: Re-run the metrics tests**

Run the same command from Step 2.
Expected: PASS

- [ ] **Step 6: Run Bandit on the touched metrics scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Metrics \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  -f json -o /tmp/bandit_stt_metrics.json
```

Expected: no new findings in changed code.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Metrics/stt_metrics.py \
  tldw_Server_API/app/core/Metrics/metrics_manager.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  tldw_Server_API/tests/Metrics/test_audio_stt_metrics.py \
  tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py \
  tldw_Server_API/tests/Audio/test_ws_metrics_audio.py
git commit -m "feat: add bounded stt metrics families"
```

### Task 8: Update Public Docs and Release Gates

**Files:**
- Modify: `Docs/Product/STT_Module_PRD.md`
- Modify: `Docs/API-related/Audio_Transcription_API.md`
- Modify: `Docs/Operations/Env_Vars.md`
- Modify: `Docs/Deployment/Monitoring/Metrics_Cheatsheet.md`
- Modify: `Docs/Product/STT_Module_vNext_PRD.md`

- [ ] **Step 1: Update the current-state STT PRD**

Move newly shipped behavior from `STT_Module_vNext_PRD.md` into `STT_Module_PRD.md`. Do not leave shipped behavior documented only as “future”.

- [ ] **Step 2: Update public API and operator docs**

Document:

- WS `protocol_version=2`
- legacy `reset` compatibility
- final-frame diagnostics
- new org-level retention/redaction policy behavior
- new `audio_stt_*` metric families and bounded labels

- [ ] **Step 3: Run the full targeted verification set**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sharing/test_clone_service.py \
  tldw_Server_API/tests/DB_Management/test_media_db_legacy_transcripts.py \
  tldw_Server_API/tests/DB_Management/test_media_transcripts_upsert.py \
  tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py \
  tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingestion_audio_transcripts.py \
  tldw_Server_API/tests/Audio/test_stt_vnext_config_flags.py \
  tldw_Server_API/tests/Audio/test_ws_control_protocol_v2.py \
  tldw_Server_API/tests/Audio/test_ws_invalid_json_error.py \
  tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py \
  tldw_Server_API/tests/Audio/test_ws_vad_turn_detection.py \
  tldw_Server_API/tests/Audio/test_ws_diarization_persistence_status.py \
  tldw_Server_API/tests/Audio/test_audio_streaming_truthiness_flags.py \
  tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py \
  tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_org_policy_from_principal.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_org_stt_settings_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ/unit/test_admin_orgs_service_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_generated_files_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ_Postgres/test_org_stt_settings_pg.py \
  tldw_Server_API/tests/Admin/test_admin_orgs_stt_settings.py \
  tldw_Server_API/tests/Metrics/test_audio_stt_metrics.py \
  tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py \
  tldw_Server_API/tests/Audio/test_ws_metrics_audio.py \
  -v
```

Expected: PASS

- [ ] **Step 4: Run Bandit on the final touched scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/audio \
  tldw_Server_API/app/api/v1/API_Deps/auth_deps.py \
  tldw_Server_API/app/core/Audio \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio \
  tldw_Server_API/app/core/DB_Management/media_db \
  tldw_Server_API/app/core/AuthNZ \
  tldw_Server_API/app/core/Metrics \
  -f json -o /tmp/bandit_stt_vnext_final.json
```

Expected: no new findings in changed code.

- [ ] **Step 5: Commit**

```bash
git add Docs/Product/STT_Module_PRD.md \
  Docs/API-related/Audio_Transcription_API.md \
  Docs/Operations/Env_Vars.md \
  Docs/Deployment/Monitoring/Metrics_Cheatsheet.md \
  Docs/Product/STT_Module_vNext_PRD.md
git commit -m "docs: publish stt vnext rollout contracts"
```

## Rollout Notes

- Keep transcript run-history dual-read fallback until structured fallback logs are quiet in staging, then until `audio_stt_transcript_read_path_total{path="legacy_fallback"}` is zero for 7 days in production.
- Ship WS control v2 dark by default; enable only after the config parser, metrics, and diagnostics slices are already merged.
- Do not advertise `/audio/chat/stream` as v2-capable until it either adopts the shared control helper or explicitly passes parity/regression coverage.
- Do not remove the v1 WS path until protocol-v2-disabled sessions have explicit regression coverage.
- Do not expose tenant retention/redaction admin routes without tests for weaker-than-tenant request overrides.
- In single-user mode, do not expose fake tenant policy surfaces; rely on global config defaults and the shared policy resolver.
- If retained raw audio artifacts are not indexed through `generated_files` in your branch, keep `delete_audio_after_success=True` as the enforced default and block enabling retention TTL in production until that integration exists.

## Suggested Branch / PR Split

- PR 1: worktree/bootstrap + config surface
- PR 2: transcript run history
- PR 3: WS control v2
- PR 4: diagnostics contract
- PR 5: tenant policy storage + policy enforcement
- PR 6: metrics + docs + rollout gates
