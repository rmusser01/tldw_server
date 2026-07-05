# Research Workspace NotebookLM Media Outputs WP2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real Research Workspace `video_overview` and `infographic` outputs: backend Jobs produce durable MP4/PNG output artifacts, and the WebUI submits, polls, previews, and downloads them.

**Architecture:** Add a small Research Workspace output-job layer that creates a pending workspace artifact, enqueues a Jobs row, and lets a startup-registered worker update the artifact when media is complete. Video Overview reuses `SlidesGenerator`, `SlidesDatabase`, per-slide TTS output artifacts, and `render_presentation_video`; Infographic reuses the image generation adapter and persists the returned bytes as a durable output artifact. The frontend only submits jobs, polls status, and renders existing download URLs.

**Tech Stack:** FastAPI, Pydantic, Jobs `WorkerSDK`, `CollectionsDatabase` output artifacts, `SlidesGenerator`, `SlidesDatabase`, TTS service v2, image generation `ImageAdapter`, Next.js/React, Ant Design, Vitest, Pytest, Bandit.

---

## Scope Check

This is one reviewable WP2 feature because the backend contract, worker, storage shape, and Studio pane UI must ship together for either output type to work. The work is split into small tasks, but do not merge a backend-only submit endpoint that leaves jobs permanently pending.

Do not add a new video engine, image backend, browser ffmpeg path, or media artifact table. Use existing Jobs, Slides, TTS, image adapter, output artifacts, and workspace artifacts.

## Pre-Implementation Issues To Guard Against

- Submit-only backend endpoints are not acceptable. The worker must be registered in normal server startup through `startup_content_jobs_pollers.py`.
- `slides_assets.py` resolves slide media only from `output:<id>` refs. Per-slide narration clips must be `CollectionsDatabase.create_output_artifact(...)` rows.
- File Artifact export URLs are TTL-oriented. The final infographic preview/download must be a durable output artifact URL.
- Keep job payloads free of secrets. Store provider names and settings, not API keys.
- The UI should preview media from `exportRefs` but keep downloads on the authenticated output-artifact path when `serverId` is available.

## File Map

Backend files to create:

- `tldw_Server_API/app/api/v1/schemas/research_workspace_outputs.py`
  - Pydantic request/response schemas for output job submission and status.
- `tldw_Server_API/app/core/Research_Workspace/output_jobs.py`
  - Constants, source-context assembly, job submission, worker dependency loading, status projection, durable output persistence, video generation, infographic generation.
- `tldw_Server_API/app/services/research_workspace_output_jobs_worker.py`
  - Long-lived Jobs `WorkerSDK` runner for `research_workspace_output` jobs.
- `tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py`
  - API validation, pending artifact creation, status projection tests.
- `tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py`
  - Worker processing tests for video, infographic, and failure update behavior.
- `tldw_Server_API/tests/Research_Workspace/test_output_jobs_startup.py`
  - Startup registration/worker-spec test.

Backend files to modify:

- `tldw_Server_API/app/api/v1/schemas/research_workspace_capabilities.py`
  - Add `video_overview_generation`, `infographic_generation`, and `image_generation` capability IDs.
- `tldw_Server_API/app/core/Research_Workspace/capabilities.py`
  - Add image health collection and compose the new capabilities.
- `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
  - Add `POST /{workspace_id}/outputs` and `GET /{workspace_id}/outputs/{job_id}` routes.
- `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
  - Register and start the Research Workspace output worker.
- `tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py`
  - Extend existing capability tests for image/video gates.
- `tldw_Server_API/tests/Research_Workspace/test_capability_endpoint.py`
  - Extend endpoint contract if new collector arguments change the returned shape.

Frontend files to modify:

- `apps/packages/ui/src/types/workspace.ts`
  - Add `video_overview` and `infographic` artifact types and output config.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/research-workspace-capabilities.ts`
  - Add new capability IDs and artifact mapping.
- `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
  - Add output submit/status request methods and response types.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx`
  - Add buttons/icons/groups and modal branching for media previews.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`
  - Submit backend jobs for media outputs and poll until completion/failure.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/ArtifactModalContent.tsx`
  - Add video and image preview viewers.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactExport.tsx`
  - Add MP4/PNG extensions and durable media download handling.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/research-workspace-capabilities.test.ts`
  - Extend capability normalization and mapping tests.
- `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts`
  - Add submit/status path tests.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx`
  - Add button/preview tests, or create `StudioPane.media-outputs.test.tsx` if the existing file becomes too broad.

## Task 1: Backend Capability Contract

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/research_workspace_capabilities.py`
- Modify: `tldw_Server_API/app/core/Research_Workspace/capabilities.py`
- Modify: `tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py`

- [ ] **Step 1: Write failing capability tests**

Add tests proving ready dependencies expose the new IDs, image backend absence blocks only image/infographic, and TTS/slides/ffmpeg absence blocks video.

```python
def test_ready_dependencies_allow_media_output_actions():
    response = build_research_workspace_capabilities(
        **_health_inputs(
            render_health={"status": "healthy"},
            image_health={"status": "healthy", "providers": {"available": 1}},
        )
    )

    assert response.capabilities["image_generation"].mode == "allow"
    assert response.capabilities["infographic_generation"].mode == "allow"
    assert response.capabilities["video_overview_generation"].mode == "allow"


def test_image_unavailable_blocks_infographic_only():
    response = build_research_workspace_capabilities(
        **_health_inputs(image_health={"status": "unavailable", "reason_code": "image_backend_unavailable"})
    )

    assert response.capabilities["image_generation"].mode == "block"
    assert response.capabilities["infographic_generation"].mode == "block"
    assert response.capabilities["video_overview_generation"].mode == "allow"
    assert response.capabilities["slides_generation"].mode == "allow"


def test_ffmpeg_unavailable_blocks_video_overview_only():
    response = build_research_workspace_capabilities(
        **_health_inputs(render_health={"status": "unavailable", "reason_code": "presentation_render_ffmpeg_unavailable"})
    )

    assert response.capabilities["video_overview_generation"].mode == "block"
    assert response.capabilities["video_overview_generation"].reason_code == "presentation_render_ffmpeg_unavailable"
    assert response.capabilities["infographic_generation"].mode == "allow"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py -v
```

Expected: FAIL because the schema Literal and capability builder do not know the new IDs, `render_health`, or `image_health`.

- [ ] **Step 3: Extend schema and capability builder**

In `research_workspace_capabilities.py`, extend the Literal:

```python
ResearchWorkspaceCapabilityId = Literal[
    "source_browse",
    "chat",
    "artifact_text_generation",
    "slides_generation",
    "audio_summary",
    "video_overview_generation",
    "image_generation",
    "infographic_generation",
    "export_download",
    "sync_share",
]
```

In `capabilities.py`, add render and image health to the collector dataclass and builder:

```python
RESEARCH_WORKSPACE_CAPABILITY_IDS = (
    "source_browse",
    "chat",
    "artifact_text_generation",
    "slides_generation",
    "audio_summary",
    "video_overview_generation",
    "image_generation",
    "infographic_generation",
    "export_download",
    "sync_share",
)


@dataclass(frozen=True)
class ResearchWorkspaceHealthCollectors:
    aggregate_health: HealthCollector
    rag_health: HealthCollector
    llm_health: HealthCollector
    slides_health: SlidesHealthCollector
    tts_health: HealthCollector
    render_health: HealthCollector
    image_health: HealthCollector
```

Add `_presentation_render_capability()`. This is not exposed as a standalone capability ID in WP2; it is an internal dependency for `video_overview_generation`.

```python
def _presentation_render_capability(render_health: Mapping[str, Any] | None) -> ResearchWorkspaceCapability:
    status = _status(render_health)
    reason = _reason_code(render_health)
    if status == "ready":
        return _cap("ready", "allow", ["presentation_render"])
    if status == "degraded":
        return _cap("degraded", "warn", ["presentation_render"], reason or "presentation_render_degraded")
    if status == "unavailable":
        return _cap(
            "unavailable",
            "block",
            ["presentation_render"],
            reason or "presentation_render_unavailable",
        )
    return _cap("unknown", "warn", ["presentation_render"], reason or "presentation_render_unknown")
```

Add `_image_capability()` matching `_tts_capability()` style. Keep reason codes sanitized through the existing `_reason_code()` path.

```python
def _image_capability(image_health: Mapping[str, Any] | None) -> ResearchWorkspaceCapability:
    providers = _mapping_value(image_health, "providers")
    available = providers.get("available") if isinstance(providers, Mapping) else None
    status = _status(image_health)
    reason = _reason_code(image_health)

    if available == 0 or status == "unavailable":
        return _cap("unavailable", "block", ["image_generation"], reason or "image_backend_unavailable")
    if status == "degraded":
        return _cap("degraded", "warn", ["image_generation"], reason or "image_backend_degraded")
    if status == "ready":
        return _cap("ready", "allow", ["image_generation"])
    return _cap("unknown", "warn", ["image_generation"], reason or "image_backend_unknown")
```

Compose new actions:

```python
render = _presentation_render_capability(render_health)
image = _image_capability(image_health)
capabilities = {
    ...
    "video_overview_generation": _compose_capability(
        dependencies=["source_browse", "llm", "slides", "tts", "presentation_render"],
        required=[source, llm, slides, tts, render],
    ),
    "image_generation": image,
    "infographic_generation": _compose_capability(
        dependencies=["source_browse", "llm", "image_generation"],
        required=[source, llm, image],
    ),
    ...
}
```

Update `collect_research_workspace_capabilities()` to gather render and image health. The render collector must check the same ffmpeg resolution contract used by the renderer:

```python
def _collect_presentation_render_health() -> Mapping[str, Any]:
    ffmpeg_path = (os.getenv("FFMPEG_PATH") or "").strip() or shutil.which("ffmpeg")
    if not ffmpeg_path:
        return {"status": "unavailable", "reason_code": "presentation_render_ffmpeg_unavailable"}
    return {"status": "healthy", "components": {"ffmpeg": {"status": "healthy"}}}
```

The default image collector can be a lightweight function that imports `tldw_Server_API.app.core.Image_Generation.adapter_registry.get_registry()` and returns ready when at least one backend is resolvable/configured. If no stable health helper exists, start with conservative `"unknown"` rather than overclaiming ready.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/research_workspace_capabilities.py \
  tldw_Server_API/app/core/Research_Workspace/capabilities.py \
  tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py
git commit -m "feat: expose research workspace media capabilities"
```

## Task 2: Backend Output Job Schemas And API Skeleton

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/research_workspace_outputs.py`
- Create: `tldw_Server_API/app/core/Research_Workspace/output_jobs.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Create: `tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py`

- [ ] **Step 1: Write failing API/schema tests**

Test that unknown types and empty sources are rejected, valid submit creates a pending artifact and Jobs row, and status returns job progress plus artifact.

```python
def test_submit_output_rejects_empty_source_ids():
    with pytest.raises(ValidationError):
        ResearchWorkspaceOutputSubmitRequest(
            artifact_type="infographic",
            source_ids=[],
        )


def test_submit_output_creates_pending_artifact_and_job(fake_workspace_db, fake_job_manager):
    result = submit_research_workspace_output_job(
        workspace_id="ws-1",
        request=ResearchWorkspaceOutputSubmitRequest(
            artifact_type="infographic",
            source_ids=["src-1"],
        ),
        workspace_db=fake_workspace_db,
        job_manager=fake_job_manager,
        user_id="42",
    )

    assert result.artifact_type == "infographic"
    assert result.status == "queued"
    assert fake_workspace_db.added_artifacts[0]["status"] == "pending"
    assert fake_job_manager.created_jobs[0]["job_type"] == "research_workspace_output"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py -v
```

Expected: FAIL because schemas and submission helper do not exist.

- [ ] **Step 3: Add Pydantic schemas**

Create `research_workspace_outputs.py`:

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from tldw_Server_API.app.api.v1.schemas.workspace_schemas import WorkspaceArtifactResponse

ResearchWorkspaceOutputArtifactType = Literal["video_overview", "infographic"]
ResearchWorkspaceOutputStatus = Literal["queued", "processing", "completed", "failed", "cancelled"]


class ResearchWorkspaceOutputSettings(BaseModel):
    provider: str | None = Field(default=None, max_length=128)
    model: str | None = Field(default=None, max_length=256)
    title_hint: str | None = Field(default=None, max_length=256)
    slides_visual_style_id: str | None = Field(default=None, max_length=128)
    tts_provider: str | None = Field(default=None, max_length=64)
    tts_model: str | None = Field(default=None, max_length=128)
    tts_voice: str | None = Field(default=None, max_length=128)
    image_backend: str | None = Field(default=None, max_length=128)
    image_width: int | None = Field(default=None, ge=256, le=2048)
    image_height: int | None = Field(default=None, ge=256, le=2048)


class ResearchWorkspaceOutputSubmitRequest(BaseModel):
    artifact_type: ResearchWorkspaceOutputArtifactType
    source_ids: list[str] = Field(..., min_length=1, max_length=50)
    settings: ResearchWorkspaceOutputSettings = Field(default_factory=ResearchWorkspaceOutputSettings)

    @field_validator("source_ids")
    @classmethod
    def _source_ids_must_be_non_empty_strings(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not normalized:
            raise ValueError("source_ids must include at least one source id")
        return list(dict.fromkeys(normalized))


class ResearchWorkspaceOutputSubmitResponse(BaseModel):
    job_id: int
    status: ResearchWorkspaceOutputStatus
    workspace_id: str
    artifact_id: str
    artifact_type: ResearchWorkspaceOutputArtifactType


class ResearchWorkspaceOutputStatusResponse(BaseModel):
    job_id: int
    status: ResearchWorkspaceOutputStatus
    progress_percent: float | None = None
    progress_message: str | None = None
    workspace_id: str
    artifact_id: str
    artifact_type: ResearchWorkspaceOutputArtifactType
    artifact: WorkspaceArtifactResponse | None = None
    error: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Add submission/status helpers**

Create constants and minimal helpers in `output_jobs.py`:

```python
RESEARCH_WORKSPACE_OUTPUT_JOB_DOMAIN = "research_workspace"
RESEARCH_WORKSPACE_OUTPUT_JOB_QUEUE = "default"
RESEARCH_WORKSPACE_OUTPUT_JOBS_QUEUE_ENV = "RESEARCH_WORKSPACE_OUTPUT_JOBS_QUEUE"
RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE = "research_workspace_output"


_PUBLIC_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")


def _safe_public_error_code(value: str) -> str:
    raw = str(value or "").strip().lower()
    if _PUBLIC_ERROR_CODE_RE.fullmatch(raw):
        return raw
    return "research_workspace_output_failed"


class ResearchWorkspaceOutputJobError(RuntimeError):
    def __init__(
        self,
        public_code: str,
        *,
        status_code: int = 400,
        retryable: bool = False,
        backoff_seconds: int | None = None,
    ) -> None:
        super().__init__(public_code)
        self.public_code = _safe_public_error_code(public_code)
        self.status_code = status_code
        self.retryable = retryable
        self.backoff_seconds = backoff_seconds


def research_workspace_output_jobs_queue() -> str:
    raw = (os.getenv(RESEARCH_WORKSPACE_OUTPUT_JOBS_QUEUE_ENV) or "").strip().lower()
    if raw in {"default", "high", "low"}:
        return raw
    return RESEARCH_WORKSPACE_OUTPUT_JOB_QUEUE


def submit_research_workspace_output_job(
    *,
    workspace_id: str,
    request: ResearchWorkspaceOutputSubmitRequest,
    workspace_db: CharactersRAGDB,
    job_manager: JobManager,
    user_id: int | str,
) -> ResearchWorkspaceOutputSubmitResponse:
    _validate_workspace_sources(workspace_db, workspace_id, request.source_ids)
    artifact_id = f"{request.artifact_type}-{uuid.uuid4().hex}"
    workspace_db.add_workspace_artifact(
        workspace_id,
        _pending_artifact_payload(
            artifact_id=artifact_id,
            artifact_type=request.artifact_type,
            source_ids=request.source_ids,
            user_id=str(user_id),
            settings=request.settings,
        ),
    )
    job = job_manager.create_job(
        domain=RESEARCH_WORKSPACE_OUTPUT_JOB_DOMAIN,
        queue=research_workspace_output_jobs_queue(),
        job_type=RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
        owner_user_id=str(user_id),
        payload={
            "workspace_id": workspace_id,
            "artifact_id": artifact_id,
            "artifact_type": request.artifact_type,
            "source_ids": request.source_ids,
            "settings": request.settings.model_dump(exclude_none=True),
            "user_id": str(user_id),
        },
        max_retries=1,
    )
    return ResearchWorkspaceOutputSubmitResponse(
        job_id=int(job["id"]),
        status=_public_job_status(job.get("status")),
        workspace_id=workspace_id,
        artifact_id=artifact_id,
        artifact_type=request.artifact_type,
    )
```

Keep `_validate_workspace_sources()` small: call `workspace_db.list_workspace_sources(workspace_id)`, confirm every requested source exists, and fail before creating the artifact if none are usable.
All API and worker code imports `ResearchWorkspaceOutputJobError` from `output_jobs.py`; do not define a second worker-local error class with different fields.

- [ ] **Step 5: Add FastAPI routes**

In `workspaces.py`, import schemas/helpers and add routes near the existing artifacts routes:

```python
@router.post(
    "/{workspace_id}/outputs",
    response_model=ResearchWorkspaceOutputSubmitResponse,
    status_code=202,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Generate a Research Workspace media output",
)
async def submit_workspace_output(
    workspace_id: str,
    body: ResearchWorkspaceOutputSubmitRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    jm: JobManager | None = Depends(try_get_workspace_job_manager),
    current_user: User = Depends(get_request_user),
) -> ResearchWorkspaceOutputSubmitResponse:
    _require_workspace(db, workspace_id)
    if jm is None:
        raise HTTPException(status_code=503, detail="jobs_unavailable")
    try:
        return submit_research_workspace_output_job(
            workspace_id=workspace_id,
            request=body,
            workspace_db=db,
            job_manager=jm,
            user_id=current_user.id,
        )
    except ResearchWorkspaceOutputJobError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_code) from exc
```

Add a GET route that calls `get_research_workspace_output_job_status(...)`, checks workspace ownership through `_require_workspace`, and returns `ResearchWorkspaceOutputStatusResponse`.
Add a status-route test that rejects a job whose payload `workspace_id` or `owner_user_id` does not match the path/current user before returning artifact details.

- [ ] **Step 6: Run tests to verify pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/research_workspace_outputs.py \
  tldw_Server_API/app/core/Research_Workspace/output_jobs.py \
  tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py
git commit -m "feat: add research workspace output job API"
```

## Task 3: Worker Registration And Jobs Worker Skeleton

**Files:**
- Create: `tldw_Server_API/app/services/research_workspace_output_jobs_worker.py`
- Modify: `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
- Create: `tldw_Server_API/tests/Research_Workspace/test_output_jobs_startup.py`
- Modify: `tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py`

- [ ] **Step 1: Write failing startup/worker tests**

```python
def test_content_job_specs_include_research_workspace_output_worker():
    specs = provide_content_jobs_worker_specs()
    names = {spec.name for spec in specs}

    assert "research_workspace_output_jobs_task" in names


@pytest.mark.asyncio
async def test_worker_rejects_unrelated_job_type(fake_job_manager):
    with pytest.raises(ResearchWorkspaceOutputJobError) as excinfo:
        await process_research_workspace_output_job(
            {"id": 1, "job_type": "other", "payload": {}},
            job_manager=fake_job_manager,
        )

    assert excinfo.value.retryable is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_startup.py \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v
```

Expected: FAIL because worker module and startup registration do not exist.

- [ ] **Step 3: Add worker module**

Model it on `presentation_render_jobs_worker.py`:

```python
from tldw_Server_API.app.core.Research_Workspace.output_jobs import (
    RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
    ResearchWorkspaceOutputJobError,
    normalize_research_workspace_output_payload,
    process_research_workspace_output_payload,
    research_workspace_output_jobs_queue,
)


@dataclass
class _ProgressState:
    percent: float | None = None
    message: str | None = None


async def process_research_workspace_output_job(
    job: dict[str, Any],
    *,
    job_manager: JobManager,
    worker_id: str = "research-workspace-output-worker",
    progress: _ProgressState | None = None,
) -> dict[str, Any]:
    payload = normalize_research_workspace_output_payload(job.get("payload"))
    if str(job.get("job_type") or "").lower() != RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE:
        raise ResearchWorkspaceOutputJobError("unsupported_job_type", retryable=False)
    user_id = resolve_research_workspace_output_job_user_id(job, payload)
    workspace_db = await open_research_workspace_output_notes_db(user_id)
    try:
        with managed_media_database(
            "research_workspace_output_worker",
            db_path=str(DatabasePaths.get_media_db_path(user_id)),
            initialize=False,
        ) as media_db:
            return await process_research_workspace_output_payload(
                job=job,
                payload=payload,
                workspace_db=workspace_db,
                media_db=media_db,
                user_id=user_id,
                job_manager=job_manager,
                progress=progress,
            )
    finally:
        close_research_workspace_output_notes_db(workspace_db)


def resolve_research_workspace_output_job_user_id(job: dict[str, Any], payload: dict[str, Any]) -> int:
    owner = payload.get("user_id") or job.get("owner_user_id")
    try:
        user_id = int(owner)
    except (TypeError, ValueError) as exc:
        raise ResearchWorkspaceOutputJobError("missing_owner_user_id", retryable=False) from exc
    if user_id <= 0:
        raise ResearchWorkspaceOutputJobError("missing_owner_user_id", retryable=False)
    return user_id


async def open_research_workspace_output_notes_db(user_id: int) -> CharactersRAGDB:
    return await get_chacha_db_for_user_id(
        user_id,
        client_id=f"research-workspace-output-worker-{user_id}",
    )


def close_research_workspace_output_notes_db(db: Any) -> None:
    if hasattr(db, "release_context_connection"):
        db.release_context_connection()
    elif hasattr(db, "close_connection"):
        db.close_connection()
```

`process_research_workspace_output_payload(...)` lives in `output_jobs.py` and receives already-open `workspace_db` and `media_db` handles. The worker must not depend on FastAPI request dependencies.

Add `run_research_workspace_output_jobs_worker(stop_event=None)` with `WorkerSDK`, a progress callback, queue resolution through `research_workspace_output_jobs_queue()`, and a worker ID env var `RESEARCH_WORKSPACE_OUTPUT_JOBS_WORKER_ID`.

- [ ] **Step 4: Register startup worker**

In `ContentJobsPollerHandles`, add:

```python
research_workspace_output_jobs_stop_event: Any | None = None
research_workspace_output_jobs_task: Any | None = None
```

In `provide_content_jobs_worker_specs()`, add:

```python
stop_event_worker_spec(
    name="research_workspace_output_jobs_task",
    worker_service=_run_research_workspace_output_jobs_worker_service,
    category="jobs",
    phase=ShutdownPhase.JOB_POLLER_QUIESCE,
    enabled=route_enabled_predicate(
        "RESEARCH_WORKSPACE_OUTPUT_JOBS_WORKER_ENABLED",
        "research-workspace-output-jobs",
        default_stable=True,
    ),
),
```

Add `_start_research_workspace_output_jobs_worker(...)` mirroring `_start_presentation_render_jobs_worker(...)`, call it from `start_content_jobs_pollers()`, and wire the handles into the returned dataclass.

At the bottom:

```python
def _run_research_workspace_output_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.research_workspace_output_jobs_worker import (
        run_research_workspace_output_jobs_worker as _run_research_workspace_output_jobs_worker,
    )

    return _run_research_workspace_output_jobs_worker(stop_event)
```

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_startup.py \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/services/research_workspace_output_jobs_worker.py \
  tldw_Server_API/app/services/startup_content_jobs_pollers.py \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_startup.py \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py
git commit -m "feat: start research workspace output worker"
```

## Task 4: Shared Output Context And Durable Artifact Persistence

**Files:**
- Modify: `tldw_Server_API/app/core/Research_Workspace/output_jobs.py`
- Modify: `tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py`

- [ ] **Step 1: Write failing tests for context and persistence**

```python
def test_build_source_context_uses_selected_ready_media(fake_workspace_db, fake_media_db):
    context = build_research_workspace_output_source_context(
        workspace_db=fake_workspace_db,
        media_db=fake_media_db,
        workspace_id="ws-1",
        source_ids=["src-1"],
        max_chars=10_000,
    )

    assert "# Source One" in context.text
    assert context.source_lineage["selected_source_ids"] == ["src-1"]


def test_persist_output_bytes_creates_durable_output_artifact(tmp_path, fake_collections_db):
    artifact = persist_research_workspace_output_bytes(
        collections_db=fake_collections_db,
        user_id=42,
        job_id=7,
        artifact_type="infographic",
        title="Infographic",
        content=b"png-bytes",
        format_="png",
        content_type="image/png",
        workspace_id="ws-1",
        workspace_artifact_id="infographic-1",
    )

    assert artifact.download_url == "/api/v1/outputs/123/download"
    assert fake_collections_db.created[0]["type_"] == "research_workspace_infographic"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v
```

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Add source context helper**

Create a small dataclass:

```python
@dataclass(frozen=True)
class ResearchWorkspaceOutputSourceContext:
    text: str
    source_lineage: dict[str, Any]
    preview_text: str
```

Implementation guidance:

- Call `workspace_db.list_workspace_sources(workspace_id)`.
- Match requested `source_ids`.
- For each source, use `media_id` and the media DB content path already used by `_source_preview_payload`: prefer `media["content"]` if present. If implementation needs transcript/document fallback, reuse `get_latest_transcription()` and `get_document_version()` as `slides.py` does.
- Cap total source text with existing Studio defaults where practical. Do not add unbounded prompt payloads.
- Raise a non-retryable `ResearchWorkspaceOutputJobError("source_context_empty")` when no selected source has usable content.

Expected text shape:

```python
parts.append(f"# {title}\\n\\n{content_excerpt}")
```

- [ ] **Step 4: Add durable output byte persistence**

Use output artifact storage only:

```python
def persist_research_workspace_output_bytes(
    *,
    collections_db: CollectionsDatabase,
    user_id: int,
    job_id: int,
    artifact_type: str,
    title: str,
    content: bytes,
    format_: str,
    content_type: str,
    workspace_id: str,
    workspace_artifact_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> ResearchWorkspacePersistedOutput:
    if not content:
        raise ResearchWorkspaceOutputJobError("empty_output", retryable=False)

    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"research-workspace-{workspace_artifact_id}-{uuid.uuid4().hex}.{format_}"
    path = outputs_dir / filename
    path.write_bytes(content)

    row = collections_db.create_output_artifact(
        job_id=job_id,
        type_=f"research_workspace_{artifact_type}",
        title=title,
        format_=format_,
        storage_path=filename,
        workspace_tag=f"workspace:{workspace_id}",
        metadata_json=json.dumps(
            {
                "origin": "research_workspace",
                "workspace_id": workspace_id,
                "workspace_artifact_id": workspace_artifact_id,
                "content_type": content_type,
                "byte_size": len(content),
                **dict(metadata or {}),
            },
            ensure_ascii=False,
        ),
    )
    return ResearchWorkspacePersistedOutput(
        output_id=int(row.id),
        download_url=f"/api/v1/outputs/{row.id}/download",
        format=format_,
        content_type=content_type,
        byte_size=len(content),
    )
```

If a storage-path normalizer already exists in touched code, use it instead of trusting `filename`. Do not store absolute paths in metadata or export refs.

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Research_Workspace/output_jobs.py \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py
git commit -m "feat: persist research workspace output artifacts"
```

## Task 5: Infographic Worker Implementation

**Files:**
- Modify: `tldw_Server_API/app/core/Research_Workspace/output_jobs.py`
- Modify: `tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py`

- [ ] **Step 1: Write failing infographic worker tests**

```python
@pytest.mark.asyncio
async def test_infographic_worker_generates_image_and_updates_workspace_artifact(monkeypatch, fake_workspace_db):
    monkeypatch.setattr(output_jobs, "build_research_workspace_output_source_context", fake_context)
    monkeypatch.setattr(output_jobs, "generate_infographic_prompt", lambda **_: "Create a concise infographic")
    monkeypatch.setattr(output_jobs, "ImageAdapter", FakeImageAdapter)
    monkeypatch.setattr(output_jobs, "CollectionsDatabase", FakeCollectionsDatabaseFactory)
    fake_job_manager = FakeJobManager()

    result = await process_research_workspace_output_payload(
        job={"id": 7, "owner_user_id": "42"},
        payload={
            "workspace_id": "ws-1",
            "artifact_id": "infographic-abc",
            "artifact_type": "infographic",
            "source_ids": ["src-1"],
            "settings": {"image_backend": "local", "image_width": 1024, "image_height": 1024},
            "user_id": "42",
        },
        workspace_db=fake_workspace_db,
        media_db=FakeMediaDb(),
        user_id=42,
        job_manager=fake_job_manager,
    )

    assert result["output_id"] == 123
    update = fake_workspace_db.updated_artifacts[-1]
    assert update["status"] == "complete"
    assert update["content_type"] == "image/png"
    assert update["export_refs"][0]["url"] == "/api/v1/outputs/123/download"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py::test_infographic_worker_generates_image_and_updates_workspace_artifact -v
```

Expected: FAIL because infographic processing is not implemented.

- [ ] **Step 3: Add prompt generation and image adapter call**

Keep prompt generation small and source-grounded. Use existing LLM chat helper if available, with deterministic test-mode fallback. The first implementation only needs one PNG.

```python
def generate_infographic_prompt(
    *,
    source_context: ResearchWorkspaceOutputSourceContext,
    title_hint: str | None,
    provider: str | None,
    model: str | None,
) -> str:
    prompt = (
        "Create one clean, information-dense infographic image. "
        "Use only the facts in the source context. "
        "Prefer labeled sections, numbers, timelines, and comparisons where relevant.\\n\\n"
        f"{source_context.text[:8000]}"
    )
    return call_llm_for_prompt_or_test_fallback(prompt, provider=provider, model=model)
```

Call `ImageAdapter` directly and persist returned bytes as an output artifact:

```python
adapter = ImageAdapter()
structured = adapter.normalize(
    {
        "backend": settings.image_backend,
        "prompt": infographic_prompt,
        "width": settings.image_width or 1024,
        "height": settings.image_height or 1024,
    }
)
issues = adapter.validate(structured)
if issues:
    raise ResearchWorkspaceOutputJobError(issues[0].code, retryable=False)
export = await asyncio.to_thread(adapter.export, structured, format="png")
persisted = persist_research_workspace_output_bytes(
    collections_db=collections_db,
    user_id=user_id,
    job_id=job_id,
    artifact_type="infographic",
    title=title,
    content=export.content,
    format_="png",
    content_type=export.content_type or "image/png",
    workspace_id=workspace_id,
    workspace_artifact_id=artifact_id,
    metadata={"image_backend": structured.get("backend")},
)
```

- [ ] **Step 4: Update workspace artifact on success/failure**

Success patch:

```python
workspace_db.update_workspace_artifact(
    workspace_id,
    artifact_id,
    {
        "status": "complete",
        "content_type": "image/png",
        "preview_text": source_context.preview_text,
        "summary": "Generated infographic",
        "producer_metadata": {
            "producer_type": "research_workspace_output_job",
            "job_id": job_id,
            "artifact_type": "infographic",
            "image_backend": structured.get("backend"),
        },
        "source_lineage": source_context.source_lineage,
        "version_metadata": {"version_label": "v1"},
        "export_refs": [
            {
                "id": persisted.output_id,
                "fileId": persisted.output_id,
                "format": "png",
                "url": persisted.download_url,
                "status": "ready",
                "content_type": "image/png",
                "bytes": persisted.byte_size,
            }
        ],
        "completed_at": _utc_now_iso(),
    },
)
```

Failure patch should set `status="failed"`, keep `content_type`, and write a sanitized public reason into `producer_metadata["error"]`; do not expose traceback text.

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Research_Workspace/output_jobs.py \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py
git commit -m "feat: generate research workspace infographics"
```

## Task 6: Video Overview Worker Implementation

**Files:**
- Modify: `tldw_Server_API/app/core/Research_Workspace/output_jobs.py`
- Modify: `tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py`

- [ ] **Step 1: Write failing video worker tests**

Test direct render invocation, per-slide audio output refs, final MP4 output artifact, and completed workspace artifact.

```python
@pytest.mark.asyncio
async def test_video_worker_renders_narrated_slideshow_without_nested_job(monkeypatch, fake_workspace_db):
    render_calls = []
    fake_job_manager = FakeJobManager()
    monkeypatch.setattr(output_jobs, "SlidesGenerator", FakeSlidesGenerator)
    monkeypatch.setattr(output_jobs, "generate_tts_audio_bytes", fake_tts_audio)
    monkeypatch.setattr(output_jobs, "render_presentation_video", lambda **kwargs: render_calls.append(kwargs) or fake_render_result())

    result = await process_research_workspace_output_payload(
        job={"id": 7, "owner_user_id": "42"},
        payload={
            "workspace_id": "ws-1",
            "artifact_id": "video-overview-abc",
            "artifact_type": "video_overview",
            "source_ids": ["src-1"],
            "settings": {"tts_provider": "kitten_tts"},
            "user_id": "42",
        },
        workspace_db=fake_workspace_db,
        media_db=FakeMediaDb(),
        user_id=42,
        job_manager=fake_job_manager,
    )

    assert result["format"] == "mp4"
    assert render_calls
    assert all(
        slide["metadata"]["studio"]["audio"]["asset_ref"].startswith("output:")
        for slide in render_calls[0]["slides"]
    )
    assert not fake_job_manager.created_jobs
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py::test_video_worker_renders_narrated_slideshow_without_nested_job -v
```

Expected: FAIL because video processing is not implemented.

- [ ] **Step 3: Generate and persist slides**

Use `SlidesGenerator.generate_from_text(...)` directly. Keep WP2 to 5 to 8 slides by prompt/title hints and post-limit if needed.

```python
provider = normalize_research_workspace_llm_provider(settings.provider)
model = normalize_research_workspace_llm_model(settings.model)
generator = SlidesGenerator()
generated = generator.generate_from_text(
    source_text=source_context.text,
    title_hint=settings.title_hint or "Video Overview",
    provider=provider,
    model=model,
    api_key=None,
    temperature=0.4,
    max_tokens=2400,
    max_source_tokens=50_000,
    max_source_chars=200_000,
    enable_chunking=True,
    chunk_size_tokens=4_000,
    summary_tokens=900,
    visual_style_snapshot=None,
)
slides = normalize_research_workspace_video_slides(generated["slides"], max_slides=8)
```

Create the presentation through `SlidesDatabase.create_presentation(...)`:

```python
row = slides_db.create_presentation(
    presentation_id=None,
    title=generated.get("title") or "Video Overview",
    description=None,
    theme="default",
    marp_theme=None,
    settings=json.dumps({"origin": "research_workspace_video_overview"}),
    studio_data=None,
    template_id=None,
    visual_style_id=settings.slides_visual_style_id,
    visual_style_scope=None,
    visual_style_name=None,
    visual_style_version=None,
    visual_style_snapshot=None,
    slides=json.dumps(slides),
    slides_text=flatten_research_workspace_slides_text(slides),
    source_type="research_workspace",
    source_ref=json.dumps({"workspace_id": workspace_id, "source_ids": source_ids}),
    source_query=None,
    custom_css=None,
)
```

- [ ] **Step 4: Synthesize narration as output artifacts**

Add a wrapper around TTS service v2. It can mirror `audiobook_jobs_worker._generate_tts_audio(...)`, but keep it local to `output_jobs.py` unless a public helper already exists.

```python
async def generate_tts_audio_bytes(
    *,
    text: str,
    provider: str | None,
    model: str | None,
    voice: str | None,
    speed: float | None,
    response_format: str,
    user_id: int,
) -> bytes:
    request = OpenAISpeechRequest(
        model=model or "tts-1",
        input=text,
        voice=voice or "alloy",
        response_format=response_format,
        speed=float(speed or 1.0),
        stream=False,
    )
    service = await get_tts_service_v2()
    chunks = bytearray()
    async for chunk in service.generate_speech(request, provider=provider, fallback=True, user_id=user_id):
        chunks.extend(chunk)
    if not chunks:
        raise ResearchWorkspaceOutputJobError("tts_empty_audio", retryable=False)
    return bytes(chunks)
```

For each slide:

```python
notes = str(slide.get("speaker_notes") or slide.get("content") or slide.get("title") or "").strip()
audio_bytes = await generate_tts_audio_bytes(..., text=notes, response_format="mp3")
audio_output = persist_research_workspace_output_bytes(
    collections_db=collections_db,
    user_id=user_id,
    job_id=job_id,
    artifact_type="video_overview_narration",
    title=f"{title} narration {index + 1}",
    content=audio_bytes,
    format_="mp3",
    content_type="audio/mpeg",
    workspace_id=workspace_id,
    workspace_artifact_id=artifact_id,
    metadata={"slide_index": index},
)
slide.setdefault("metadata", {}).setdefault("studio", {})["audio"] = {
    "asset_ref": f"output:{audio_output.output_id}",
}
```

- [ ] **Step 5: Call `render_presentation_video` directly**

Do not enqueue a Presentation Render job here.

```python
render_result = await asyncio.to_thread(
    render_presentation_video,
    presentation_id=row.id,
    presentation_version=int(row.version),
    title=row.title,
    slides=slides,
    output_format="mp4",
    output_dir=DatabasePaths.get_user_outputs_dir(user_id),
    user_id=user_id,
)
```

If `render_presentation_video` already writes the MP4 under the user output directory and returns `storage_path`, create the final output artifact with that path instead of rewriting bytes:

```python
final_row = collections_db.create_output_artifact(
    job_id=job_id,
    type_="research_workspace_video_overview",
    title=f"{row.title} (MP4)",
    format_="mp4",
    storage_path=render_result.storage_path,
    workspace_tag=f"workspace:{workspace_id}",
    metadata_json=json.dumps({...}, ensure_ascii=False),
)
```

- [ ] **Step 6: Update completed workspace artifact**

Patch the workspace artifact with:

- `status`: `complete`
- `content_type`: `video/mp4`
- `preview_text`: short source-grounded summary
- `producer_metadata`: job ID, presentation ID/version, TTS provider/model/voice, render settings
- `source_lineage`: selected source IDs and context metadata
- `export_refs`: output ID, file ID, format `mp4`, URL `/api/v1/outputs/{id}/download`, byte size/content type
- `serverId` is frontend-only; backend exposes the output ID via `export_refs`

- [ ] **Step 7: Run tests to verify pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Research_Workspace/output_jobs.py \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py
git commit -m "feat: generate research workspace video overviews"
```

## Task 7: Frontend Types, Capability Mapping, And API Client

**Files:**
- Modify: `apps/packages/ui/src/types/workspace.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/research-workspace-capabilities.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/research-workspace-capabilities.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts`

- [ ] **Step 1: Write failing frontend tests**

```ts
it("maps media output artifact types to media capability boundaries", () => {
  expect(getArtifactCapabilityId("video_overview")).toBe("video_overview_generation")
  expect(getArtifactCapabilityId("infographic")).toBe("infographic_generation")
})

it("submits and polls research workspace output jobs", async () => {
  vi.mocked(bgRequest).mockResolvedValueOnce({
    job_id: 7,
    status: "queued",
    workspace_id: "ws-1",
    artifact_id: "infographic-1",
    artifact_type: "infographic"
  })

  await workspaceApiMethods.submitWorkspaceOutput("ws-1", {
    artifact_type: "infographic",
    source_ids: ["src-1"]
  })

  expect(bgRequest).toHaveBeenCalledWith({
    path: "/api/v1/workspaces/ws-1/outputs",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: { artifact_type: "infographic", source_ids: ["src-1"] }
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/research-workspace-capabilities.test.ts \
  apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts
```

Expected: FAIL because types/API methods do not exist.

- [ ] **Step 3: Add artifact types and output config**

In `workspace.ts`:

```ts
export type ArtifactType =
  | "summary"
  | "audio_overview"
  | "video_overview"
  | "infographic"
  | "mindmap"
  | "report"
  | "compare_sources"
  | "flashcards"
  | "quiz"
  | "timeline"
  | "slides"
  | "data_table"
```

Add `OUTPUT_TYPES` entries:

```ts
{
  type: "video_overview",
  label: "Video Overview",
  icon: "Video",
  description: "Generate a narrated slideshow video from your sources",
  requiresSelectedSources: true
},
{
  type: "infographic",
  label: "Infographic",
  icon: "Image",
  description: "Generate an infographic image from your sources",
  requiresSelectedSources: true
}
```

- [ ] **Step 4: Add capability IDs and mapping**

In `research-workspace-capabilities.ts`:

```ts
export const RESEARCH_WORKSPACE_CAPABILITY_IDS = [
  "source_browse",
  "chat",
  "artifact_text_generation",
  "slides_generation",
  "audio_summary",
  "video_overview_generation",
  "image_generation",
  "infographic_generation",
  "export_download",
  "sync_share"
] as const

export function getArtifactCapabilityId(type: ArtifactType): ResearchWorkspaceCapabilityId {
  if (type === "slides") return "slides_generation"
  if (type === "audio_overview") return "audio_summary"
  if (type === "video_overview") return "video_overview_generation"
  if (type === "infographic") return "infographic_generation"
  return "artifact_text_generation"
}
```

- [ ] **Step 5: Add API client types and methods**

In `workspace-api.ts`:

```ts
export type ResearchWorkspaceOutputArtifactType = "video_overview" | "infographic"

export interface ResearchWorkspaceOutputSubmitRequest {
  artifact_type: ResearchWorkspaceOutputArtifactType
  source_ids: string[]
  settings?: Record<string, unknown>
}

export interface ResearchWorkspaceOutputSubmitResponse {
  job_id: number
  status: string
  workspace_id: string
  artifact_id: string
  artifact_type: ResearchWorkspaceOutputArtifactType
}

export interface ResearchWorkspaceOutputStatusResponse extends ResearchWorkspaceOutputSubmitResponse {
  progress_percent?: number | null
  progress_message?: string | null
  artifact?: WorkspaceArtifactApiResponse | null
  error?: string | null
  result?: Record<string, unknown>
}
```

Methods:

```ts
async submitWorkspaceOutput(
  workspaceId: string,
  data: ResearchWorkspaceOutputSubmitRequest
): Promise<ResearchWorkspaceOutputSubmitResponse> {
  return await bgRequest<ResearchWorkspaceOutputSubmitResponse>({
    path: workspacePath(workspaceId, "/outputs"),
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: data
  })
},

async getWorkspaceOutputStatus(
  workspaceId: string,
  jobId: number | string
): Promise<ResearchWorkspaceOutputStatusResponse> {
  return await bgRequest<ResearchWorkspaceOutputStatusResponse>({
    path: workspacePath(workspaceId, `/outputs/${encodeWorkspacePathSegment(String(jobId), "jobId")}`),
    method: "GET"
  })
},
```

- [ ] **Step 6: Run tests to verify pass**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/research-workspace-capabilities.test.ts \
  apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/types/workspace.ts \
  apps/packages/ui/src/components/Option/ResearchWorkspace/research-workspace-capabilities.ts \
  apps/packages/ui/src/services/tldw/domains/workspace-api.ts \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/research-workspace-capabilities.test.ts \
  apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts
git commit -m "feat: add research workspace media output client contract"
```

## Task 8: Studio Pane Submission, Polling, Preview, And Download

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/ArtifactModalContent.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactExport.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx` or create `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.media-outputs.test.tsx`

- [ ] **Step 1: Write failing Studio tests**

Tests should assert buttons render, capability blocks disable them, completed artifacts render the correct media element, and download uses output artifact ID.

```tsx
it("renders media output buttons", () => {
  render(<StudioPane {...propsWithSourcesAndReadyCapabilities} />)

  expect(screen.getByRole("button", { name: /video overview/i })).toBeInTheDocument()
  expect(screen.getByRole("button", { name: /infographic/i })).toBeInTheDocument()
})

it("renders completed media previews from export refs", () => {
  render(
    <ArtifactModalContent
      artifact={{
        ...baseArtifact,
        type: "infographic",
        exportRefs: [{ format: "png", url: "/api/v1/outputs/123/download", status: "ready" }]
      }}
    />
  )

  expect(screen.getByRole("img", { name: /infographic/i })).toHaveAttribute(
    "src",
    "/api/v1/outputs/123/download"
  )
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx
```

Expected: FAIL because buttons/viewers do not exist.

- [ ] **Step 3: Add Studio buttons and icons**

In `index.tsx`, import `Video` and `Image` from `lucide-react`.

```ts
const ARTIFACT_TYPE_ICONS: Record<ArtifactType, React.ElementType> = {
  audio_overview: Headphones,
  video_overview: Video,
  infographic: Image,
  ...
}
```

Add output buttons:

```ts
{
  type: "video_overview",
  label: "Video Overview",
  description: OUTPUT_TYPES.find((config) => config.type === "video_overview")?.description || "Generate a narrated slideshow video",
  icon: Video
},
{
  type: "infographic",
  label: "Infographic",
  description: OUTPUT_TYPES.find((config) => config.type === "infographic")?.description || "Generate an infographic image",
  icon: Image
}
```

Put them in `OUTPUT_GROUPS`. Keep the page dense and operational; do not add instructional copy.

- [ ] **Step 4: Add backend media generation path**

In `useArtifactGeneration.tsx`, add helpers:

```ts
const getReadyExportRef = (artifact: GeneratedArtifact, formats: string[]) =>
  artifact.exportRefs?.find(
    (ref) => formats.includes(String(ref.format).toLowerCase()) && typeof ref.url === "string"
  )

const toGeneratedArtifactFromWorkspaceArtifact = (
  response: WorkspaceArtifactApiResponse,
  fallbackType: ArtifactType
): Partial<GeneratedArtifact> => {
  const exportRef = response.export_refs?.[0]
  return {
    id: response.id,
    type: (response.artifact_type as ArtifactType) || fallbackType,
    title: response.title,
    status: response.status === "complete" ? "completed" : response.status === "failed" ? "failed" : "generating",
    contentType: response.content_type,
    previewText: response.preview_text ?? undefined,
    summary: response.summary ?? undefined,
    exportRefs: response.export_refs as GeneratedArtifact["exportRefs"],
    serverId: exportRef?.fileId ?? exportRef?.id,
    producerMetadata: response.producer_metadata,
    sourceLineage: Array.isArray(response.source_lineage) ? response.source_lineage : undefined,
    version: response.version,
    completedAt: response.completed_at ? new Date(response.completed_at) : undefined
  }
}
```

Add `generateBackendMediaOutput(type, options)`:

```ts
const submit = await tldwClient.submitWorkspaceOutput(workspaceId, {
  artifact_type: type,
  source_ids: selectedSources.map((source) => source.id),
  settings: {
    provider: normalizedApiProvider,
    model: selectedModel || undefined,
    title_hint: workspaceName || undefined,
    slides_visual_style_id: type === "video_overview" ? slidesVisualStyleValue || undefined : undefined,
    tts_provider: type === "video_overview" ? audioSettings.provider : undefined,
    tts_model: type === "video_overview" ? audioSettings.model : undefined,
    tts_voice: type === "video_overview" ? audioSettings.voice : undefined
  }
})
```

Polling guidance:

- Poll `getWorkspaceOutputStatus(workspaceId, submit.job_id)` every 1500 to 2500 ms.
- Update the pending local artifact with `progressMessage` in `producerMetadata`, not visible tutorial text.
- Stop after completed, failed, cancelled, or a reasonable timeout.
- On completion, replace/update the pending artifact using the returned `artifact`.

- [ ] **Step 5: Add media viewers**

In `ArtifactModalContent.tsx`:

```tsx
const getArtifactPreviewUrl = (artifact: GeneratedArtifact, formats: string[]) => {
  const ref = artifact.exportRefs?.find((entry) =>
    formats.includes(String(entry.format || "").toLowerCase()) &&
    typeof entry.url === "string" &&
    entry.url.trim()
  )
  return typeof ref?.url === "string" ? ref.url : null
}

export const VideoOverviewArtifactViewer: React.FC<{ artifact: GeneratedArtifact }> = ({ artifact }) => {
  const src = getArtifactPreviewUrl(artifact, ["mp4"])
  if (!src) return <MarkdownPreview content={artifact.previewText || artifact.summary || ""} />
  return (
    <div className="max-h-[70vh] overflow-y-auto">
      <video className="max-h-[62vh] w-full rounded border border-border bg-black" controls preload="metadata">
        <source src={src} type={artifact.contentType || "video/mp4"} />
      </video>
    </div>
  )
}

export const InfographicArtifactViewer: React.FC<{ artifact: GeneratedArtifact }> = ({ artifact }) => {
  const src = getArtifactPreviewUrl(artifact, ["png"])
  if (!src) return <MarkdownPreview content={artifact.previewText || artifact.summary || ""} />
  return (
    <div className="max-h-[70vh] overflow-auto">
      <img src={src} alt={artifact.title || "Infographic"} className="mx-auto max-h-[64vh] max-w-full rounded border border-border object-contain" />
    </div>
  )
}
```

Wire these into `renderArtifactModalContent(...)` in `index.tsx`.

- [ ] **Step 6: Add download handling**

In `useArtifactExport.tsx`:

```ts
case "video_overview":
  return "mp4"
case "infographic":
  return "png"
```

Before generic text fallback:

```ts
if ((artifact.type === "video_overview" || artifact.type === "infographic") && artifact.serverId) {
  const blob = await tldwClient.downloadOutput(String(artifact.serverId))
  downloadBlobFile(blob, `${artifact.title}.${getFileExtension(artifact.type)}`)
  return
}
```

If `serverId` is missing but `exportRefs[0].url` exists, use that URL as a fallback link. Prefer `serverId` when available.

- [ ] **Step 7: Run tests to verify pass**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/research-workspace-capabilities.test.ts \
  apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/ArtifactModalContent.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactExport.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx
git commit -m "feat: render research workspace media outputs"
```

## Task 9: End-To-End Verification, Security Scan, And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-12160 - Implement-Research-Workspace-NotebookLM-media-outputs-WP2.md`

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py \
  tldw_Server_API/tests/Research_Workspace/test_capability_endpoint.py \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py \
  tldw_Server_API/tests/Research_Workspace/test_output_jobs_startup.py \
  -v
```

Expected: all selected tests PASS.

- [ ] **Step 2: Run focused frontend tests**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/research-workspace-capabilities.test.ts \
  apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx
```

Expected: all selected tests PASS.

- [ ] **Step 3: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Research_Workspace \
  tldw_Server_API/app/services/research_workspace_output_jobs_worker.py \
  tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  -f json -o /tmp/bandit_research_workspace_wp2.json
```

Expected: no new findings in touched code. If Bandit reports existing unrelated findings in touched broad files, document exact finding IDs and why they are pre-existing; fix new findings before continuing.

- [ ] **Step 4: Optional local smoke with real backends**

Only run this when local TTS, image generation, and ffmpeg are configured. The user has a local llama.cpp API at `127.0.0.1:9099`, but do not assume image/TTS are available.

Run server and frontend as usual, then:

- Create/open a Research Workspace.
- Select one ready source with extracted text.
- Generate `Infographic`; verify the completed card opens an image preview and downloads PNG.
- Generate `Video Overview`; verify the completed card opens a video player and downloads MP4.
- Confirm `/api/v1/outputs/{id}/download` works for both.

- [ ] **Step 5: Update Backlog task**

Use Backlog CLI or MCP:

```bash
backlog task edit TASK-12160 \
  --append-notes "Implemented Research Workspace WP2 media outputs. Verification: backend tests PASS, frontend tests PASS, Bandit PASS. See Docs/superpowers/plans/2026-07-05-research-workspace-notebooklm-media-outputs-wp2-plan.md."
```

If the CLI schema differs, use the available Backlog command that appends notes/status without manually editing task files.

- [ ] **Step 6: Final status and commit**

```bash
git status --short
git add backlog/tasks/task-12160\ -\ Implement-Research-Workspace-NotebookLM-media-outputs-WP2.md
git commit -m "chore: close research workspace media outputs task"
```

Expected: working tree clean except intentionally untracked local artifacts, if any.

## Final Implementation Notes

- Keep `status="complete"` in backend workspace artifact rows. The frontend maps backend `complete` to UI `completed`.
- Use `completed_at` only after final output artifact persistence succeeds.
- Use `progress_message` values like `build_context`, `generate_infographic`, `generate_slides`, `synthesize_narration`, `render_video`, and `persist_artifact`; do not expose stack traces or provider errors directly.
- Prefer small helper functions in `output_jobs.py`. If the file grows beyond easy review, split only the video-specific implementation into `output_video.py` and infographic-specific implementation into `output_infographic.py`.
- Do not update broad generated OpenAPI files unless the repository already requires it for touched routes.
