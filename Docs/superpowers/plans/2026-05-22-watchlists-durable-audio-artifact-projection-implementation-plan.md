# Watchlists Durable Audio Artifact Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Watchlists audio briefing artifacts durable in `/watchlists` by projecting canonical Workflows audio artifacts into Watchlist run stats and canonical output metadata.

**Architecture:** Workflows remains the canonical artifact store. Watchlists adds request correlation, a compact projection helper, lazy read-repair in `/runs/{run_id}/audio`, retry stale-state handling, and frontend rendering of mirrored artifacts. Proactive projection is Watchlists-owned and best effort; it must not create a hard Workflows-to-Watchlists dependency.

**Tech Stack:** FastAPI, SQLite/PostgreSQL-backed Workflows DB, Watchlists DB, Collections DB, Scheduler, pytest, Vitest/React Testing Library, TypeScript.

---

## Scope Check

This spec touches several layers, but they are tightly coupled around one product contract: durable Watchlists audio artifact projection. Implement in this order so every task leaves the system in a testable state:

1. Make Workflow run correlation real.
2. Add Watchlists audio request IDs and retry-safe enqueue semantics.
3. Tag or infer audio artifacts consistently.
4. Add the Watchlists projection helper.
5. Wire `/runs/{run_id}/audio` to lazy read-repair and mirrored fallback.
6. Update frontend durable graph and retry state handling.
7. Add best-effort proactive projection only after lazy read-repair is solid.

If time is constrained, implementation tasks 1-7 are the MVP for the durable `/watchlists` user experience. Task 8, proactive projection, can be a follow-up PR because lazy read-repair is the reliability guarantee. A backend-only slice through implementation task 6 is acceptable only as an intermediate PR, not as the final durable UX.

## File Map

Backend:

- Modify `tldw_Server_API/app/core/DB_Management/Workflows_DB.py`
  - Add durable run metadata support to `workflow_runs`.
  - Add migrations for SQLite and backend/PostgreSQL schema version.
  - Include `metadata_json` in `WorkflowRun`, `create_run`, `get_run`, and `list_runs`.
- Modify `tldw_Server_API/app/core/Scheduler/handlers/workflows.py`
  - Pass payload `metadata` into `WorkflowsDatabase.create_run(...)`.
  - Merge payload `metadata` into the persisted workflow definition snapshot so Workflow adapters receive it through `context["workflow_metadata"]`.
- Modify `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
  - Generate/preserve `audio_request_id`.
  - Include request ID in Scheduler metadata, Workflow payload metadata, Workflow inputs, and idempotency key.
  - Return request ID through `AudioBriefingTriggerResult`.
- Modify `tldw_Server_API/app/core/Workflows/adapters/content/audio_briefing.py`
  - Add Watchlists correlation metadata to script artifacts.
- Modify `tldw_Server_API/app/core/Workflows/adapters/audio/multi_voice_tts.py`
  - Add Watchlists correlation metadata to speaker and final artifacts.
- Modify `tldw_Server_API/app/core/Workflows/adapters/audio/tts.py`
  - Add optional artifact metadata passthrough so the single-voice fallback can be tagged as final/fallback.
- Modify `tldw_Server_API/app/core/Workflows/adapters/audio/_config.py`
  - Add optional `artifact_metadata` to TTS config if validation requires it.
- Create `tldw_Server_API/app/core/Watchlists/audio_artifact_projection.py`
  - Build compact audio graph, normalize statuses, sanitize artifact summaries, mirror run/output metadata, read mirrored state, mark stale on retry.
- Modify `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
  - Add optional audio projection fields to `WatchlistRunAudioResponse`.
- Modify `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Inject/resolve `CollectionsDatabase` for `/runs/{run_id}/audio`.
  - Inject/resolve `CollectionsDatabase` for `/runs/{run_id}/retry-audio` when stale-state updates need canonical output metadata writes.
  - Resolve Workflows DB through the same factory path as the Scheduler/Workflows API instead of manually constructing a target-user SQLite path.
  - Use projection helper for canonical lookup, lazy mirror write, mirrored fallback, retry stale-state handling, and target-user-safe links.

Backend tests:

- Create or modify `tldw_Server_API/tests/Workflows/test_workflows_run_metadata.py`
- Modify `tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py`
- Modify `tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py`
- Create `tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py`
- Modify `tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py`
- Modify `tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py` if retry semantics need coverage there.

Frontend:

- Modify `apps/packages/ui/src/types/watchlists.ts`
- Modify `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts`
- Modify `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx` only if stale/error rendering needs adjustment.
- Modify `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
- Modify locale files:
  - `apps/packages/ui/src/assets/locale/en/watchlists.json`
  - `apps/packages/ui/src/public/_locales/en/watchlists.json`

Frontend tests:

- Modify `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts`
- Modify `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx`
- Modify `apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx`

Docs/backlog:

- Update `backlog/tasks/task-482 - Plan-Watchlists-durable-audio-artifact-projection-implementation.md` during planning.
- Implementation tasks should create or update a separate Backlog task before code edits begin.

---

### Task 1: Persist Workflow Run Correlation Metadata

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Workflows_DB.py`
- Modify: `tldw_Server_API/app/core/Scheduler/handlers/workflows.py`
- Test: `tldw_Server_API/tests/Workflows/test_workflows_run_metadata.py`

- [x] **Step 1: Write failing SQLite run metadata tests**

Add tests that create a temporary `WorkflowsDatabase`, call `create_run(..., metadata={...})`, and assert `get_run()` / `list_runs()` return `metadata_json`.

```python
def test_create_run_persists_metadata_json(tmp_path):
    from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase

    db = WorkflowsDatabase(db_path=str(tmp_path / "workflows.db"))
    db.create_run(
        run_id="wf_audio_1",
        tenant_id="default",
        user_id="1",
        inputs={"items": []},
        definition_snapshot={"name": "audio_briefing", "steps": []},
        metadata={
            "source": "watchlist_audio_briefing",
            "watchlist_run_id": 7,
            "watchlist_job_id": 3,
            "audio_request_id": "wla_test_1",
        },
    )

    run = db.get_run("wf_audio_1")
    assert run is not None
    assert run.metadata_json is not None
    assert json.loads(run.metadata_json)["audio_request_id"] == "wla_test_1"
```

- [x] **Step 2: Run failing metadata test**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/test_workflows_run_metadata.py -q
```

Expected: fails because `create_run()` does not accept `metadata` and `WorkflowRun` has no `metadata_json`.

- [x] **Step 3: Implement Workflows DB metadata support**

In `Workflows_DB.py`:

- Bump `_CURRENT_SCHEMA_VERSION` from its current value to the next integer (`8` to `9` on the current base).
- Add `metadata_json: str | None = None` to `WorkflowRun`.
- Add `metadata: dict[str, Any] | None = None` to `create_run(...)`.
- Add `metadata_json` to initial `workflow_runs` schemas for SQLite and backend/PostgreSQL.
- Add SQLite and backend migrations to add `workflow_runs.metadata_json`.
- Insert `json.dumps(metadata or {})` into `workflow_runs.metadata_json`.
- Prefer `TEXT` for `workflow_runs.metadata_json` in both SQLite and PostgreSQL backend schemas so `WorkflowRun.metadata_json` keeps the same string contract as `inputs_json` and `definition_snapshot_json`. If a backend migration uses `JSONB`, normalize backend row values to a JSON string before constructing `WorkflowRun`.

Minimal shape:

```python
metadata_json = json.dumps(metadata or {})
```

- [x] **Step 4: Pass Scheduler payload metadata into Workflow runs**

In `tldw_Server_API/app/core/Scheduler/handlers/workflows.py`, pass sanitized payload metadata:

```python
metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
db.create_run(
    ...
    metadata=metadata,
)
```

Also merge the same metadata into a copied `definition_snapshot["metadata"]` before `create_run(...)` so the Workflow engine injects it into adapter context as `context["workflow_metadata"]`. Do not mutate the caller-owned payload dict in place.

```python
if isinstance(definition_snapshot, dict):
    definition_snapshot = dict(definition_snapshot)
    existing_definition_metadata = definition_snapshot.get("metadata")
    merged_definition_metadata = {
        **(existing_definition_metadata if isinstance(existing_definition_metadata, dict) else {}),
        **metadata,
    }
    if merged_definition_metadata:
        definition_snapshot["metadata"] = merged_definition_metadata
```

This is separate from `workflow_runs.metadata_json`: run metadata makes lookup durable; definition metadata makes artifact tagging possible.

- [x] **Step 5: Add handler test for payload metadata propagation**

Patch `_get_wf_db()` with a fake DB and assert:

- `create_run(..., metadata=payload["metadata"])` is called
- the saved `definition_snapshot["metadata"]` contains the payload metadata
- the original `payload["definition_snapshot"]` object is unchanged

- [x] **Step 6: Run metadata tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Workflows/test_workflows_run_metadata.py \
  -q
```

Expected: pass.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/Workflows_DB.py \
  tldw_Server_API/app/core/Scheduler/handlers/workflows.py \
  tldw_Server_API/tests/Workflows/test_workflows_run_metadata.py
git commit -m "feat: persist workflow run metadata"
```

---

### Task 2: Add Audio Request IDs And Retry-Safe Enqueue

**Files:**
- Modify: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py`

- [x] **Step 1: Write failing trigger test for `audio_request_id`**

Add coverage to `test_audio_briefing_workflow.py`:

```python
assert result.audio_request_id
assert payload["metadata"]["audio_request_id"] == result.audio_request_id
assert payload["inputs"]["audio_request_id"] == result.audio_request_id
assert submit_kwargs["idempotency_key"].endswith(f":{result.audio_request_id}")
```

Also add a regression assertion that a stale or user-supplied `output_prefs["audio_request_id"]` is ignored unless passed through the explicit internal test hook.

- [x] **Step 2: Run failing trigger test**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py::TestTriggerAudioBriefing \
  -q
```

Expected: fails because `AudioBriefingTriggerResult` has no `audio_request_id` and idempotency key does not include it.

- [x] **Step 3: Implement request ID creation and propagation**

In `audio_briefing_workflow.py`:

- Add `audio_request_id: str | None = None` to `AudioBriefingTriggerResult`.
- Generate a request ID before Workflow input construction, using a collision-resistant prefix.
- Do not read `audio_request_id` from user/job `output_prefs`; it is correlation state, not a user preference.
- If tests or future internal callers need determinism, add an explicit keyword-only internal parameter such as `audio_request_id: str | None = None` and validate that it starts with `wla_`.
- Add `audio_request_id` to Workflow inputs and metadata.
- Include `audio_request_id` in Scheduler task metadata.
- Include `audio_request_id` in the idempotency key.

Suggested helper:

```python
def _new_audio_request_id() -> str:
    return f"wla_{uuid.uuid4().hex}"
```

Idempotency key:

```python
idempotency_key=f"watchlist-audio-briefing:{user_id}:{job_id}:{run_id}:{audio_request_id}"
```

- [x] **Step 4: Persist request ID in trigger metadata application**

Update `apply_audio_briefing_result_metadata(...)` to set or clear `audio_request_id` consistently:

```python
if result.audio_request_id:
    target["audio_request_id"] = result.audio_request_id
else:
    target.pop("audio_request_id", None)
```

Keep flat compatibility fields unchanged.

- [x] **Step 5: Update retry endpoint stale-state setup**

In `retry_run_audio(...)`, make retry create a new request ID via `trigger_audio_briefing(...)`. Do not reuse old `audio_request_id`, and do not let a stale value in `output_prefs` override the generated request ID.

Before applying the new result, preserve or mark old audio metadata stale in run stats:

```python
old_audio = run_stats.get("audio")
if isinstance(old_audio, dict):
    old_audio = {**old_audio, "stale": True, "superseded_by": audio_result.audio_request_id}
    run_stats["previous_audio"] = old_audio
run_stats.pop("audio", None)
```

The exact helper will move into Task 4, but add test coverage now.

- [x] **Step 6: Run Watchlists audio workflow tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py::TestGetRunAudioEndpoint \
  -q
```

Expected: pass after implementation.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py
git commit -m "feat: add watchlists audio request ids"
```

---

### Task 3: Tag Script, Speaker, Final, And Fallback Artifacts

**Files:**
- Modify: `tldw_Server_API/app/core/Workflows/adapters/content/audio_briefing.py`
- Modify: `tldw_Server_API/app/core/Workflows/adapters/audio/multi_voice_tts.py`
- Modify: `tldw_Server_API/app/core/Workflows/adapters/audio/tts.py`
- Modify: `tldw_Server_API/app/core/Workflows/adapters/audio/_config.py`
- Modify: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
- Test: `tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py`
- Test: `tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py`

- [x] **Step 1: Write failing artifact metadata tests**

Add tests that set `context["workflow_metadata"]`:

```python
context["workflow_metadata"] = {
    "source": "watchlist_audio_briefing",
    "watchlist_run_id": 7,
    "watchlist_job_id": 3,
    "audio_request_id": "wla_test_1",
}
```

Assert script, speaker, final, and fallback TTS artifact metadata include those fields.

- [x] **Step 2: Run failing adapter tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py::TestMultiVoiceTTSAdapter \
  tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py \
  -q
```

Expected: new metadata assertions fail.

- [x] **Step 3: Add shared correlation metadata helper locally**

Keep implementation small. In each touched adapter, derive only safe Watchlists keys from `context["workflow_metadata"]`:

```python
def _watchlist_artifact_metadata(context: dict[str, Any]) -> dict[str, Any]:
    meta = context.get("workflow_metadata") if isinstance(context.get("workflow_metadata"), dict) else {}
    return {
        key: meta[key]
        for key in ("source", "watchlist_job_id", "watchlist_run_id", "audio_request_id")
        if key in meta
    }
```

If duplication becomes distracting, move the helper into `tldw_Server_API/app/core/Workflows/adapters/_common.py`, but do not over-refactor.

Before changing adapters, confirm the Task 1 handler test proves payload metadata reaches `definition_snapshot["metadata"]`; otherwise these adapter changes will pass only synthetic unit tests and fail in the real workflow path.

- [x] **Step 4: Merge correlation metadata into artifacts**

Merge the helper result into:

- `audio_script` artifact metadata in `audio_briefing.py`.
- Per-speaker and final `tts_audio` artifact metadata in `multi_voice_tts.py`.
- Generic `tts_audio` metadata in `tts.py`.

For fallback TTS, add optional `artifact_metadata` in config and merge it last after base model/voice/format metadata.

- [x] **Step 5: Mark Watchlists fallback TTS as final fallback**

In `AUDIO_BRIEFING_WORKFLOW_DEF`, add fallback TTS config metadata:

```python
"artifact_metadata": {
    "final_artifact": True,
    "fallback_artifact": True,
    "single_voice_fallback": True,
    "fallback_reason": "multi_voice_tts_failed",
}
```

Keep this limited to the Watchlists fallback step.

- [x] **Step 6: Run adapter tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py \
  tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py \
  -q
```

Expected: pass.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Workflows/adapters/content/audio_briefing.py \
  tldw_Server_API/app/core/Workflows/adapters/audio/multi_voice_tts.py \
  tldw_Server_API/app/core/Workflows/adapters/audio/tts.py \
  tldw_Server_API/app/core/Workflows/adapters/audio/_config.py \
  tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py \
  tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py \
  tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py
git commit -m "feat: tag watchlists audio artifacts"
```

---

### Task 4: Add Watchlists Audio Projection Helper

**Files:**
- Create: `tldw_Server_API/app/core/Watchlists/audio_artifact_projection.py`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py`

- [x] **Step 1: Write failing projection tests**

Cover:

- graph with script, two speaker artifacts, final mix
- script-only partial graph
- speaker-only artifacts do not become final
- `final_artifact` / `background_mixed` precedence
- `succeeded` normalizes to `completed`
- `audio_request_id` match wins over old same-run artifacts
- raw `file://` URI is not included in mirrored summaries
- metadata merge preserves delivery/template/Chatbook fields

Example assertion:

```python
projection = build_audio_projection(
    run_id=91,
    task_id="task_graph",
    audio_request_id="wla_current",
    workflow_run=workflow_run,
    artifacts=artifacts,
)
assert projection["status"] == "completed"
assert projection["final_artifact"]["artifact_id"] == "art_final"
assert "uri" not in projection["final_artifact"]
```

- [x] **Step 2: Run failing projection tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py \
  -q
```

Expected: fails because module does not exist.

- [x] **Step 3: Implement projection dataless helpers**

Create pure helpers first:

- `normalize_audio_status(status: Any, *, task_id: Any = None) -> str`
- `artifact_download_url(artifact_id: Any, *, target_user_id: int | None = None) -> str | None`
- `summarize_audio_artifact(...) -> dict[str, Any]`
- `build_audio_projection(...) -> dict[str, Any]`
- `extract_workflow_run_metadata(workflow_run: Any) -> dict[str, Any]`

Keep these pure and easy to test.

`extract_workflow_run_metadata(...)` must read correlation data in this order:

1. `workflow_run.metadata_json`
2. `workflow_run.definition_snapshot_json["metadata"]`
3. `workflow_run.inputs_json`

This preserves new run-level metadata while supporting compatibility runs where only definition metadata or inputs carry the watchlist correlation fields.

`artifact_download_url(...)` must not append unsupported `target_user_id` query parameters to `/api/v1/workflows/artifacts/{artifact_id}/download`. The existing Workflows endpoint authorizes same-tenant admins by run ownership. If tests show that is insufficient for a target-user Watchlists read, add a Watchlists-scoped proxy endpoint in a later task instead of emitting links that look valid but 404.

- [x] **Step 4: Implement metadata merge helpers**

Add:

- `merge_audio_projection_metadata(existing: dict[str, Any], projection: dict[str, Any]) -> dict[str, Any]`
- `mark_audio_projection_stale(existing: dict[str, Any], *, superseded_by: str | None) -> dict[str, Any]`

Preserve unrelated keys:

```python
merged = dict(existing)
merged["audio"] = projection
merged["audio_briefing_status"] = projection["status"]
merged["audio_request_id"] = projection.get("audio_request_id")
```

- [x] **Step 5: Implement DB-facing helpers**

Add functions that accept DB instances explicitly:

- `mirror_audio_projection(run_db, collections_db, run, projection, *, user_id: int) -> bool`
- `get_mirrored_audio_projection(run) -> dict[str, Any] | None`
- `find_matching_workflow_run(workflow_db, *, tenant_id: str, user_id: str, run_id: int, audio_request_id: str | None) -> Any | None`
- `find_canonical_watchlist_output(collections_db, run_id, audio_request_id=None) -> Any | None`

Rules:

- Update run stats always when possible.
- Update canonical output metadata when a base non-audio output exists.
- Skip writes when existing mirrored graph already matches.
- Catch noncritical persistence failures and return `False`; callers should still return canonical responses.
- Keep DB helpers synchronous. Async endpoints must call blocking read/write helpers via `run_in_threadpool(...)` rather than performing SQLite/Collections writes directly on the event loop.

- [x] **Step 6: Run projection tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py \
  -q
```

Expected: pass.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Watchlists/audio_artifact_projection.py \
  tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py
git commit -m "feat: add watchlists audio projection helper"
```

---

### Task 5: Wire `/runs/{run_id}/audio` To Projection And Mirrored Fallback

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py`

- [ ] **Step 1: Write failing endpoint tests**

Add tests for:

- canonical Workflows artifacts mirror into run stats and canonical output metadata
- Workflows DB lookup failure returns mirrored metadata
- endpoint uses the same Workflows DB factory path as Scheduler/Workflows API, not `DatabasePaths.get_user_base_directory(...)/workflows/workflows.db`
- target user path resolves target Collections DB
- admin target links are target-aware or Watchlists-scoped
- projection write failure still returns canonical response

- [ ] **Step 2: Run failing endpoint tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py::TestGetRunAudioEndpoint \
  -q
```

Expected: new tests fail because endpoint does not use projection helper.

- [ ] **Step 3: Add `collections_db` dependency to `get_run_audio`**

Update signature:

```python
async def get_run_audio(..., db=Depends(get_watchlists_db_for_user), collections_db=Depends(get_collections_db_for_user)):
```

Resolve target DBs:

```python
target_collections_db = _resolve_collections_db_for_target_user(
    current_user=current_user,
    current_db=collections_db,
    target_user_id=resolved_user_id,
)
```

Resolve the Workflow DB with the same factory used by `tldw_Server_API/app/core/Scheduler/handlers/workflows.py` and `tldw_Server_API/app/api/v1/endpoints/workflows.py`:

```python
from tldw_Server_API.app.core.DB_Management.DB_Manager import (
    create_workflows_database,
    get_content_backend_instance,
)

workflow_db = create_workflows_database(backend=get_content_backend_instance())
```

Do not manually construct `DatabasePaths.get_user_base_directory(resolved_user_id) / "workflows" / "workflows.db"` inside the endpoint. That path can diverge from where the Scheduler writes Workflow runs, especially outside single-user SQLite mode.

- [ ] **Step 4: Replace inline artifact classification with projection helper**

Keep endpoint behavior, but delegate:

- matching Workflow run resolution
- artifact graph construction
- mirrored fallback
- mirror persistence

Do this incrementally. Preserve existing status fallback tests while moving logic.

Call blocking projection lookups and mirror writes through `run_in_threadpool(...)` from this async endpoint.

- [ ] **Step 5: Extend response schema**

In `WatchlistRunAudioResponse`, add optional:

- `audio_request_id`
- `workflow_run_id`
- `schema_version`
- `synced_at`
- `stale`

- [ ] **Step 6: Ensure raw URI mirror boundary**

Endpoint may continue returning `audio_uri` for compatibility, but mirrored `metadata.audio` must use `download_url` and no raw file path display fields.

- [ ] **Step 7: Run endpoint tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py \
  -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py
git commit -m "feat: mirror watchlists audio artifacts"
```

---

### Task 6: Harden Retry Stale-State Handling

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Modify: `tldw_Server_API/app/core/Watchlists/audio_artifact_projection.py`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py`

- [ ] **Step 1: Write failing retry stale-state tests**

Test setup:

- run stats contain completed `audio` graph with old `audio_request_id`
- output metadata contains same completed graph
- retry endpoint queues a new request

Assert:

- new `audio_request_id` differs from old
- active `audio.status` is queued/pending
- old graph is stale or moved to `previous_audio`
- active final artifact is not the old final artifact

- [ ] **Step 2: Run failing retry tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py::TestGetRunAudioEndpoint \
  tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py \
  -q
```

- [ ] **Step 3: Use projection stale helper in retry path**

In `retry_run_audio(...)`, add `collections_db=Depends(get_collections_db_for_user)` and resolve target Collections DB the same way `retry_run_delivery(...)` does:

```python
target_collections_db = _resolve_collections_db_for_target_user(
    current_user=current_user,
    current_db=collections_db,
    target_user_id=resolved_user_id,
)
```

Then, before or after applying the new trigger result:

- read run stats
- call `mark_audio_projection_stale(...)`
- clear active artifacts for the new request
- update canonical output metadata if a canonical output exists

- [ ] **Step 4: Run retry tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py \
  tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py \
  -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/core/Watchlists/audio_artifact_projection.py \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py \
  tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py
git commit -m "fix: prevent stale watchlists audio retry artifacts"
```

---

### Task 7: Frontend Durable Audio Graph Rendering

**Files:**
- Modify: `apps/packages/ui/src/types/watchlists.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/watchlists.json`
- Modify: `apps/packages/ui/src/public/_locales/en/watchlists.json`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx`

- [ ] **Step 1: Write failing frontend metadata tests**

Add coverage for:

- `metadata.audio.stale === true` produces a stale/superseded indicator
- live response with newer `audio_request_id` overrides mirrored metadata
- mirrored final artifact uses `download_url` and never displays raw `file://`
- old mirrored final does not appear active when status is queued with new request ID

- [ ] **Step 2: Run failing frontend metadata tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts
```

- [ ] **Step 3: Extend frontend types and normalizer**

In `types/watchlists.ts`, add optional fields to `WatchlistRunAudioStatus`:

- `audio_request_id`
- `workflow_run_id`
- `schema_version`
- `synced_at`
- `stale`

In `outputMetadata.ts`, add these to `AudioStatusSummary` and merge logic. Live summaries should override mirrored summaries when the live `audio_request_id` differs or when live has artifacts.

- [ ] **Step 4: Render full graph in Run Detail**

Refactor small local rendering in `RunDetailDrawer.tsx` to show:

- script artifact
- speaker artifacts
- final artifact
- fallback reason
- stale/superseded state

Reuse the same labels as Output Preview where practical. Do not create nested cards.

- [ ] **Step 5: Add or align locale keys**

Add labels for stale/superseded audio and script/speaker/final artifact headings if missing.

- [ ] **Step 6: Run frontend focused tests**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/types/watchlists.ts \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx \
  apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx \
  apps/packages/ui/src/assets/locale/en/watchlists.json \
  apps/packages/ui/src/public/_locales/en/watchlists.json \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx
git commit -m "feat: show durable watchlists audio graph"
```

---

### Task 8: Best-Effort Proactive Projection

**Files:**
- Modify or create: `tldw_Server_API/app/core/Watchlists/audio_projection_tasks.py`
- Modify: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
- Modify: `tldw_Server_API/app/core/Scheduler/handlers/watchlists.py` if a Watchlists Scheduler handler exists and is the established pattern
- Test: `tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py`

- [ ] **Step 1: Verify the existing Watchlists background execution pattern**

Inspect Scheduler handlers and Watchlists services. Use an existing Watchlists queue/handler if one exists. Do not create a new unserved queue.

Run:

```bash
rg -n "watchlist.*@task|queue=.*watch|watchlist_run|scale_workers" tldw_Server_API/app/core tldw_Server_API/app/services
```

- [ ] **Step 2: Decide whether proactive projection is safe for this PR**

If there is no ensured worker path, do not implement this task in the first PR. Record it as deferred in the implementation task. Lazy read-repair is already sufficient for durability on read.

- [ ] **Step 3: Write failing proactive projection tests if safe**

Test that the proactive task:

- does not treat Scheduler `workflow_run` task completion as Workflow completion
- polls or looks up the real Workflow run by `audio_request_id`
- stops after a bounded retry count
- calls `mirror_audio_projection(...)` when terminal artifacts exist
- does not raise back into audio workflow submission if projection scheduling fails

- [ ] **Step 4: Implement best-effort projection scheduling**

Only if Step 2 found a safe queue path:

- after successful audio workflow submit, submit a projection task with request correlation
- ensure the projection task queue has a worker, or use an already-served queue
- catch and log projection enqueue failures without changing `AudioBriefingTriggerResult(status="submitted")`

- [ ] **Step 5: Run proactive tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py \
  tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py \
  -q
```

- [ ] **Step 6: Commit or record deferral**

If implemented:

```bash
git add tldw_Server_API/app/core/Watchlists/audio_projection_tasks.py \
  tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py \
  tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py \
  tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py
git commit -m "feat: add watchlists audio projection task"
```

If deferred, update the implementation Backlog task with the reason and do not leave partial code.

---

### Task 9: Integration Verification And Cleanup

**Files:**
- Modify only if tests reveal issues.
- Update implementation Backlog task with verification results.

- [ ] **Step 1: Run backend focused suite**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Workflows/test_workflows_run_metadata.py \
  tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py \
  tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py \
  tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py \
  tldw_Server_API/tests/Watchlists/test_audio_artifact_projection.py \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py \
  tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Run frontend focused suite**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx
```

Expected: pass.

- [ ] **Step 3: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Watchlists \
  tldw_Server_API/app/core/Workflows/adapters/audio \
  tldw_Server_API/app/core/Workflows/adapters/content/audio_briefing.py \
  tldw_Server_API/app/core/Scheduler/handlers/workflows.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  -f json -o /tmp/bandit_watchlists_audio_projection.json
```

Expected: no new findings in touched code. Fix new findings before finishing.

- [ ] **Step 4: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 5: Manual API smoke if a local server is available**

Use a fixture or local run with `audio_briefing_task_id` and Workflows artifacts:

```bash
curl -sf "http://127.0.0.1:8000/api/v1/watchlists/runs/<RUN_ID>/audio" \
  -H "X-API-Key: <KEY>"
```

Expected: response includes `audio_request_id`, normalized `status`, script/speaker/final artifacts when available, and `download_url` values.

- [ ] **Step 6: Update implementation Backlog task**

Record:

- tests run
- Bandit output path
- proactive projection implemented or deferred
- known skips/blockers

- [ ] **Step 7: Final commit**

If any cleanup changes were made:

```bash
git add <changed files>
git commit -m "test: verify watchlists audio projection"
```

---

## Implementation Notes

- Do not remove existing `/runs/{run_id}/audio` compatibility fields. Preserve `audio_uri` in API responses where current clients expect it, but do not persist raw `file://` URIs into Watchlists metadata.
- Keep projection writes idempotent. A GET endpoint mutating state is acceptable here only as read-repair and should skip unchanged writes.
- Do not let projection failures fail audio generation.
- Do not create a proactive projection queue without proving it has workers.
- Do not replace unrelated output metadata. Merge only `audio` and flat audio compatibility fields.
- Preserve existing Watchlists news/OSINT/CTI workflows, tabs, templates, and advanced controls.
