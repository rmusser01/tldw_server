# Audio Studio MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a server-backed `/audio-studio` page that promotes Narration, Podcast, and Briefing to first-class workflows, keeps Music as the fourth MVP workflow, migrates the existing Audiobook Studio UI into the Narration workflow, and keeps `/audiobook-studio` as a compatibility route until the new page is stable.

**Architecture:** Canonical project state lives in per-user Collections DB tables exposed by `/api/v1/audio-studio`. Generation and rendering are asynchronous Jobs in the `audio_studio` domain. Provider integrations use an adapter registry with strict outbound endpoint allowlisting and secret redaction. The shared UI package owns the studio route/components/store, and the Next.js app exposes `/audio-studio` plus `/audiobook-studio` compatibility.

**Tech Stack:** FastAPI, Pydantic, Collections DB SQLite/PostgreSQL abstractions, Jobs `WorkerSDK`, existing TTS service, HTTPX for external provider adapters, ffmpeg through existing audio conversion helpers, React, Ant Design, Zustand, React Query, Dexie migration helpers, Vitest, Playwright, pytest, Bandit.

---

## Current Code Context

- Existing backend audiobook API: `tldw_Server_API/app/api/v1/endpoints/audio/audiobooks.py`
- Existing backend audiobook schemas: `tldw_Server_API/app/api/v1/schemas/audiobook_schemas.py`
- Existing audiobook worker: `tldw_Server_API/app/services/audiobook_jobs_worker.py`
- Existing content router registration: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Existing Collections DB audiobook tables and methods: `tldw_Server_API/app/core/DB_Management/Collections_DB.py`
- Existing shared UI route: `apps/packages/ui/src/routes/option-audiobook-studio.tsx`
- Existing shared UI component tree: `apps/packages/ui/src/components/Option/AudiobookStudio/`
- Existing shared UI store and Dexie persistence: `apps/packages/ui/src/store/audiobook-studio.tsx`, `apps/packages/ui/src/db/dexie/audiobook-projects.ts`
- Existing frontend page: `apps/tldw-frontend/pages/audiobook-studio.tsx`
- Existing E2E coverage: `apps/tldw-frontend/e2e/workflows/tier-2-features/audiobook-studio.spec.ts`

## Scope Boundaries

- MVP includes project CRUD, section/script editing, provider catalog, speech generation for Narration/Podcast/Briefing, ACE-Step prompt-based music generation, artifact listing, render creation, export creation, and local Audiobook/Dexie migration.
- MVP does not include a multitrack waveform timeline editor. Create a follow-up Backlog task titled `Add Audio Studio timeline editor slice` before implementing timeline editing.
- MVP does not include local ACE-Step model execution. ACE-Step support is external HTTP only.
- Existing `/api/v1/audiobooks` remains operational during the MVP and is not removed in this plan.

## Backlog Staging

Before implementation edits begin, create implementation Backlog tasks for reviewable slices:

```bash
backlog task create "Implement Audio Studio backend foundation" --label audio --label backend --priority high --doc Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md --doc Docs/superpowers/specs/2026-06-23-audio-studio-design.md
backlog task create "Implement Audio Studio jobs and providers" --label audio --label jobs --priority high --doc Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md --doc Docs/superpowers/specs/2026-06-23-audio-studio-design.md
backlog task create "Implement Audio Studio frontend route" --label audio --label webui --priority high --doc Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md --doc Docs/superpowers/specs/2026-06-23-audio-studio-design.md
backlog task create "Implement Audio Studio migration and compatibility" --label audio --label migration --priority high --doc Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md --doc Docs/superpowers/specs/2026-06-23-audio-studio-design.md
backlog task create "Add Audio Studio timeline editor slice" --label audio --label backlog --priority medium --doc Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md --doc Docs/superpowers/specs/2026-06-23-audio-studio-design.md
```

Each implementation task must reference this plan and `Docs/superpowers/specs/2026-06-23-audio-studio-design.md`.

---

## Stage 1: Backend Data Model And API Contracts

**Goal:** Establish server-owned Audio Studio models, persistence, and endpoint contracts without running generation.

**Success Criteria:** Audio Studio schemas validate workflows/resources, Collections DB can create/list/update/archive projects and revisions, and API endpoints return deterministic responses with owner isolation.

**Tests:** Pydantic schema tests, Collections DB repository tests, FastAPI integration tests.

**Status:** In Progress (TASK-2350)

### Task 1.1: Add Audio Studio Pydantic Schemas

- [ ] Create `tldw_Server_API/app/api/v1/schemas/audio_studio_schemas.py`.
- [ ] Add unit tests in `tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_schemas.py`.
- [ ] Use string enums for workflows and resources so API validation is stable:

```python
from enum import Enum

class AudioStudioWorkflow(str, Enum):
    NARRATION = "narration"
    PODCAST = "podcast"
    BRIEFING = "briefing"
    MUSIC = "music"

class AudioStudioResourceKind(str, Enum):
    SECTION = "section"
    TRACK = "track"
    CLIP = "clip"
    ARTIFACT = "artifact"
    RENDER = "render"
    EXPORT = "export"
```

- [ ] Include request/response models for:
  - `AudioStudioProjectCreate`
  - `AudioStudioProjectUpdate`
  - `AudioStudioProjectResponse`
  - `AudioStudioSectionUpsert`
  - `AudioStudioTrackUpsert`
  - `AudioStudioClipUpsert`
  - `AudioStudioGenerationCreate`
  - `AudioStudioRenderCreate`
  - `AudioStudioExportCreate`
  - `AudioStudioMigrationPreview`
  - `AudioStudioMigrationCommit`
- [ ] Enforce:
  - `workflow` in `narration|podcast|briefing|music`
  - `base_revision_id` required for mutating existing projects
  - `idempotency_key` length 16 to 200 chars for generation/render/export requests
  - `provider` not carrying secrets
  - `external_url` absent from client requests
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_schemas.py -v
```

Expected red result before implementation: imports or validation assertions fail. Expected green result after implementation: all schema tests pass.

### Task 1.2: Add Collections DB Tables And Repository Methods

- [ ] Extend `tldw_Server_API/app/core/DB_Management/Collections_DB.py`.
- [ ] Add rows near existing audiobook row dataclasses:
  - `AudioStudioProjectRow`
  - `AudioStudioRevisionRow`
  - `AudioStudioSectionRow`
  - `AudioStudioTrackRow`
  - `AudioStudioClipRow`
  - `AudioStudioArtifactRow`
  - `AudioStudioGenerationJobRow`
- [ ] Add SQLite DDL blocks for:

```sql
CREATE TABLE IF NOT EXISTS audio_studio_projects (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    title TEXT NOT NULL,
    workflow TEXT NOT NULL,
    status TEXT NOT NULL,
    settings_json TEXT NOT NULL DEFAULT '{}',
    current_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived_at TEXT,
    deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at TEXT,
    retention_until TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_audio_studio_projects_user_project_id
    ON audio_studio_projects(user_id, project_id);
CREATE INDEX IF NOT EXISTS idx_audio_studio_projects_user_updated
    ON audio_studio_projects(user_id, updated_at DESC);
```

```sql
CREATE TABLE IF NOT EXISTS audio_studio_project_revisions (
    revision_id TEXT PRIMARY KEY,
    project_row_id BIGINT NOT NULL,
    user_id TEXT NOT NULL,
    parent_revision_id TEXT,
    mutation_kind TEXT NOT NULL,
    resource_kind TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audio_studio_revisions_project
    ON audio_studio_project_revisions(project_row_id, created_at DESC);
```

```sql
CREATE TABLE IF NOT EXISTS audio_studio_sections (
    id INTEGER PRIMARY KEY,
    project_row_id BIGINT NOT NULL,
    section_id TEXT NOT NULL,
    workflow TEXT NOT NULL,
    title TEXT,
    body_text TEXT,
    speaker_id TEXT,
    order_index INTEGER NOT NULL,
    settings_json TEXT NOT NULL DEFAULT '{}',
    current_revision_id TEXT,
    archived_at TEXT,
    deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_audio_studio_sections_project_section
    ON audio_studio_sections(project_row_id, section_id);
```

```sql
CREATE TABLE IF NOT EXISTS audio_studio_artifacts (
    id INTEGER PRIMARY KEY,
    project_row_id BIGINT NOT NULL,
    artifact_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    provider TEXT,
    output_id BIGINT,
    storage_path TEXT,
    mime_type TEXT,
    size_bytes BIGINT,
    source_resource_kind TEXT,
    source_resource_id TEXT,
    source_revision_id TEXT,
    content_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    archived_at TEXT,
    deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_audio_studio_artifacts_project_artifact
    ON audio_studio_artifacts(project_row_id, artifact_id);
```

- [ ] Add PostgreSQL DDL in the same schema branch with `BIGSERIAL PRIMARY KEY` for numeric primary keys and `BIGINT` for numeric foreign-key references.
- [ ] Add matching tables for `audio_studio_tracks`, `audio_studio_clips`, `audio_studio_generation_jobs`, and `audio_studio_idempotency_keys` using the same owner/project/revision fields.
- [ ] Add repository-style methods on `CollectionsDatabase`:
  - `create_audio_studio_project`
  - `get_audio_studio_project`
  - `get_audio_studio_project_by_project_id`
  - `list_audio_studio_projects`
  - `update_audio_studio_project`
  - `archive_audio_studio_project`
  - `create_audio_studio_revision`
  - `upsert_audio_studio_section`
  - `upsert_audio_studio_track`
  - `upsert_audio_studio_clip`
  - `create_audio_studio_artifact`
  - `list_audio_studio_artifacts`
  - `record_audio_studio_generation_job`
  - `get_audio_studio_idempotency_record`
  - `put_audio_studio_idempotency_record`
- [ ] Add tests in `tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_collections_db.py` covering owner isolation, unique IDs per user, archive vs delete, revision creation, stale base revision rejection, and idempotency record lookup.
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_collections_db.py -v
```

Expected green result: repository tests pass for SQLite. PostgreSQL coverage uses existing backend abstraction tests when a Postgres fixture is available.

### Task 1.3: Add API Endpoint Skeleton

- [ ] Create `tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py`.
- [ ] Register it in `tldw_Server_API/app/api/v1/router_groups/content.py` with:

```python
ImportedRouterSpec(
    import_path="tldw_Server_API.app.api.v1.endpoints.audio.audio_studio",
    log_name="audio_studio",
    prefix=f"{API_V1_PREFIX}",
    tags=("audio-studio",),
    route_key="audio-studio",
    default_stable=False,
)
```

- [ ] Implement endpoints:
  - `GET /api/v1/audio-studio/workflows`
  - `POST /api/v1/audio-studio/projects`
  - `GET /api/v1/audio-studio/projects`
  - `GET /api/v1/audio-studio/projects/{project_id}`
  - `PATCH /api/v1/audio-studio/projects/{project_id}`
  - `DELETE /api/v1/audio-studio/projects/{project_id}`
  - `PUT /api/v1/audio-studio/projects/{project_id}/sections/{section_id}`
  - `PUT /api/v1/audio-studio/projects/{project_id}/tracks/{track_id}`
  - `PUT /api/v1/audio-studio/projects/{project_id}/clips/{clip_id}`
- [ ] Use `get_request_user`, `get_collections_db_for_user`, and the same 404 owner-isolation pattern as `audiobooks.py`.
- [ ] Add integration tests in `tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_projects_api.py`:

```python
def test_create_project_supports_first_class_workflows(client_audio_studio):
    for workflow in ("narration", "podcast", "briefing", "music"):
        response = client_audio_studio.post(
            "/api/v1/audio-studio/projects",
            json={"title": f"{workflow} project", "workflow": workflow},
        )
        assert response.status_code == 200
        assert response.json()["workflow"] == workflow
```

- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_projects_api.py -v
```

Expected green result: all project and owner-isolation tests pass.

---

## Stage 2: Providers, Security, And Jobs

**Goal:** Add generation adapters and Jobs orchestration for speech and music generation.

**Success Criteria:** Provider registry exposes configured providers, external HTTP adapters fail closed unless allowlisted, generation jobs are idempotent, and worker handlers record artifacts without logging secrets.

**Tests:** Provider unit tests, HTTP mock tests, Jobs handler tests, endpoint integration tests.

**Status:** Not Started

### Task 2.1: Add Provider Adapter Interfaces

- [ ] Create package `tldw_Server_API/app/core/Audio_Studio/`.
- [ ] Add:
  - `tldw_Server_API/app/core/Audio_Studio/models.py`
  - `tldw_Server_API/app/core/Audio_Studio/providers/base.py`
  - `tldw_Server_API/app/core/Audio_Studio/providers/registry.py`
  - `tldw_Server_API/app/core/Audio_Studio/providers/speech.py`
  - `tldw_Server_API/app/core/Audio_Studio/providers/ace_step.py`
  - `tldw_Server_API/app/core/Audio_Studio/security.py`
- [ ] Use this base contract:

```python
from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True)
class AudioGenerationRequest:
    workflow: str
    kind: str
    prompt: str | None
    text: str | None
    provider_options: dict[str, Any]
    target_resource_kind: str
    target_resource_id: str
    target_revision_id: str

@dataclass(frozen=True)
class AudioGenerationResult:
    mime_type: str
    content_bytes: bytes
    provider: str
    metadata: dict[str, Any]

class AudioStudioProviderAdapter(Protocol):
    provider_id: str
    supported_kinds: frozenset[str]

    async def generate(self, request: AudioGenerationRequest) -> AudioGenerationResult:
        ...
```

- [ ] Implement `SpeechTtsAdapter` by wrapping the existing TTS service paths used by `audiobook_jobs_worker.py`.
- [ ] Implement `AceStepHttpAdapter` as an external HTTP adapter only.
- [ ] Add tests in `tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_provider_registry.py`.
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_provider_registry.py -v
```

Expected green result: registry returns speech providers, includes ACE-Step only when explicitly configured, and rejects unsupported generation kinds.

### Task 2.2: Enforce External Endpoint Allowlisting And Secret Handling

- [ ] Implement `validate_external_audio_endpoint` in `security.py`.
- [ ] Read external config from environment variables:
  - `AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST`
  - `AUDIO_STUDIO_ACE_STEP_BASE_URL`
  - `AUDIO_STUDIO_ACE_STEP_API_KEY`
  - `AUDIO_STUDIO_ACE_STEP_TIMEOUT_SECONDS`
- [ ] Enforce:
  - Exact scheme, host, and port match against the allowlist.
  - Only `https` is allowed unless `AUDIO_STUDIO_ALLOW_HTTP_ENDPOINTS=1`.
  - Redirect targets are validated before following.
  - Provider API keys are read at call time and never stored in DB payloads.
  - Logs use `redact_audio_studio_secret(value)` for request metadata.
- [ ] Implement tests in `tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_external_security.py`:

```python
def test_ace_step_requires_allowlisted_origin(monkeypatch):
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_BASE_URL", "https://ace.localhost.invalid")
    monkeypatch.setenv("AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST", "https://other.localhost.invalid")
    with pytest.raises(ValueError, match="external_audio_endpoint_not_allowlisted"):
        build_ace_step_adapter_from_env()
```

- [ ] Add HTTP mock tests with `respx` if already available; otherwise use `httpx.MockTransport` without adding dependencies.
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_external_security.py -v
```

Expected green result: unallowlisted endpoints fail, secrets are redacted, and configured allowlisted ACE-Step calls return artifacts.

### Task 2.3: Add Audio Studio Jobs Helpers

- [ ] Create `tldw_Server_API/app/core/Audio_Studio/jobs.py`.
- [ ] Define:

```python
AUDIO_STUDIO_DOMAIN = "audio_studio"
AUDIO_STUDIO_QUEUE = "default"
JOB_TYPE_GENERATE = "audio_studio_generate"
JOB_TYPE_RENDER = "audio_studio_render"
JOB_TYPE_EXPORT = "audio_studio_export"
JOB_TYPE_MIGRATE = "audio_studio_migrate"
```

- [ ] Add enqueue helpers:
  - `enqueue_audio_studio_generation_job`
  - `enqueue_audio_studio_render_job`
  - `enqueue_audio_studio_export_job`
  - `enqueue_audio_studio_migration_job`
- [ ] Build idempotency keys from `user_id`, `project_id`, `job_type`, `target_resource_kind`, `target_resource_id`, `target_revision_id`, and caller-provided `idempotency_key`.
- [ ] Add tests in `tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_jobs.py`.
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_jobs.py -v
```

Expected green result: duplicate idempotency requests return the existing enqueueable job, and stale target revision pins are rejected before a job is created.

### Task 2.4: Add Worker Handler And Startup Registration

- [ ] Create `tldw_Server_API/app/core/Audio_Studio/jobs_worker.py`.
- [ ] Add a WorkerSDK loop matching `tldw_Server_API/app/core/Explainer/jobs_worker.py`:

```python
async def run_audio_studio_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    cfg = WorkerConfig(
        domain=AUDIO_STUDIO_DOMAIN,
        queue=AUDIO_STUDIO_QUEUE,
        worker_id=os.getenv("AUDIO_STUDIO_JOBS_WORKER_ID") or f"audio-studio-jobs-{os.getpid()}",
        lease_seconds=coerce_int(os.getenv("AUDIO_STUDIO_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"), 60),
    )
    sdk = WorkerSDK(jobs_manager_from_env(), cfg)
    await sdk.run(handler=build_audio_studio_job_handler())
```

- [ ] Add startup integration in `tldw_Server_API/app/services/startup_content_jobs_pollers.py`:
  - `_start_audio_studio_jobs_worker`
  - `_run_audio_studio_jobs_worker_service`
  - shutdown handle fields and tests aligned with existing audiobook worker tests
- [ ] Add unit tests in:
  - `tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_jobs_worker.py`
  - `tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py`
  - `tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py`
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_jobs_worker.py tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py -v
```

Expected green result: worker starts only when `AUDIO_STUDIO_JOBS_WORKER_ENABLED` allows it, stops cleanly, and records job progress.

### Task 2.5: Add Generation Endpoints

- [ ] Extend `audio_studio.py` with:
  - `GET /api/v1/audio-studio/providers`
  - `POST /api/v1/audio-studio/projects/{project_id}/generations`
  - `GET /api/v1/audio-studio/projects/{project_id}/generations/{job_id}`
  - `GET /api/v1/audio-studio/projects/{project_id}/artifacts`
- [ ] Generation request shape:

```json
{
  "kind": "speech",
  "provider": "tts",
  "target_resource_kind": "section",
  "target_resource_id": "sec_001",
  "target_revision_id": "rev_001",
  "idempotency_key": "client-uuid-or-content-key",
  "options": {
    "voice": "af_heart",
    "format": "mp3"
  }
}
```

- [ ] Add integration tests in `tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_generation_api.py`.
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_generation_api.py -v
```

Expected green result: endpoint creates Jobs rows in `audio_studio`, owner isolation works, idempotent duplicate calls return the first job, and provider keys do not appear in stored payload JSON.

---

## Stage 3: Render, Export, And Migration Services

**Goal:** Separate generation artifacts from render/export artifacts, and provide a safe migration path from local Audiobook Studio projects.

**Success Criteria:** Render jobs compose approved project artifacts, export jobs package artifacts, and Dexie projects can be previewed and committed into server projects.

**Tests:** Render service unit tests, export service tests, migration API tests, worker integration tests.

**Status:** Not Started

### Task 3.1: Add Render Service

- [ ] Create `tldw_Server_API/app/core/Audio_Studio/render.py`.
- [ ] Use existing `AudioConverter` from `tldw_Server_API/app/core/TTS/audio_converter.py` for format conversion.
- [ ] Add `AudioStudioRender` output records separate from generation outputs:

```python
@dataclass(frozen=True)
class AudioStudioRenderPlan:
    project_id: str
    render_id: str
    target_revision_id: str
    clip_artifact_ids: list[str]
    output_format: str
    loudness_normalize: bool
```

- [ ] Implement:
  - `build_render_plan`
  - `render_audio_studio_mix`
  - `record_audio_studio_render_artifact`
- [ ] Render validates artifact owner, project, and `source_revision_id` before reading files.
- [ ] Add tests in `tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_render.py` using small generated WAV bytes.
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_render.py -v
```

Expected green result: render rejects cross-project artifacts, rejects stale revision pins, and records render artifacts independently from generation artifacts.

### Task 3.2: Add Export Service

- [ ] Create `tldw_Server_API/app/core/Audio_Studio/export.py`.
- [ ] Implement:
  - `create_audio_studio_export_manifest`
  - `package_audio_studio_export`
  - `record_audio_studio_export_artifact`
- [ ] Export formats for MVP:
  - single audio file
  - zip package with manifest JSON
  - narration-compatible audiobook package when workflow is `narration`
- [ ] Extend `audio_studio.py` with:
  - `POST /api/v1/audio-studio/projects/{project_id}/renders`
  - `GET /api/v1/audio-studio/projects/{project_id}/renders/{job_id}`
  - `POST /api/v1/audio-studio/projects/{project_id}/exports`
  - `GET /api/v1/audio-studio/projects/{project_id}/exports/{job_id}`
- [ ] Add tests in:
  - `tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_export.py`
  - `tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_render_export_api.py`
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_export.py tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_render_export_api.py -v
```

Expected green result: render/export endpoints create distinct Jobs, output manifests include source artifact hashes, and repeated idempotency keys return existing jobs.

### Task 3.3: Add Audiobook Dexie Migration

- [ ] Create `tldw_Server_API/app/core/Audio_Studio/migration.py`.
- [ ] Add migration endpoints:
  - `POST /api/v1/audio-studio/migrations/audiobook/preview`
  - `POST /api/v1/audio-studio/migrations/audiobook/commit`
- [ ] Client sends a sanitized Dexie payload with project metadata, chapters, and optional audio blobs as upload references.
- [ ] Backend converts legacy projects to:
  - workflow `narration`
  - sections for each chapter
  - voice settings in `settings_json`
  - existing chapter audio as generation artifacts when present
- [ ] Add migration tests in `tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_audiobook_migration.py`.
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_audiobook_migration.py -v
```

Expected green result: preview reports import counts without writing projects, commit creates narration projects, duplicate commit idempotency returns the prior project, and malformed Dexie payloads return 422.

---

## Stage 4: Shared UI And Frontend Route

**Goal:** Replace the Audiobook-first UI with an Audio Studio route that keeps Narration, Podcast, and Briefing prominent and uses the existing Audiobook Studio controls as the Narration workflow base.

**Success Criteria:** `/audio-studio` renders with first-class workflow navigation, Narration preserves the existing audiobook editing flow, Podcast/Briefing use persisted section workflows, Music can submit ACE-Step generation when configured, and `/audiobook-studio` routes users into Narration compatibility.

**Tests:** Shared UI unit tests, service tests, route metadata tests, Next.js route tests, Playwright E2E.

**Status:** Not Started

### Task 4.1: Add Audio Studio Client Service

- [ ] Create `apps/packages/ui/src/services/audio-studio.ts`.
- [ ] Add typed functions:
  - `listAudioStudioWorkflows`
  - `listAudioStudioProjects`
  - `createAudioStudioProject`
  - `updateAudioStudioProject`
  - `upsertAudioStudioSection`
  - `upsertAudioStudioTrack`
  - `upsertAudioStudioClip`
  - `createAudioStudioGeneration`
  - `createAudioStudioRender`
  - `createAudioStudioExport`
  - `previewAudiobookMigration`
  - `commitAudiobookMigration`
- [ ] Use the existing `tldwClient` request helpers rather than direct `fetch`.
- [ ] Add service tests in `apps/packages/ui/src/services/__tests__/audio-studio.test.ts`.
- [ ] Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/services/__tests__/audio-studio.test.ts
```

Expected green result: service builds `/api/v1/audio-studio` URLs, sends auth through the shared client, and normalizes backend errors into existing UI error patterns.

### Task 4.2: Add Shared Store And Hooks

- [ ] Create `apps/packages/ui/src/store/audio-studio.tsx`.
- [ ] Create hooks:
  - `apps/packages/ui/src/hooks/useAudioStudioProjects.ts`
  - `apps/packages/ui/src/hooks/useAudioStudioGeneration.tsx`
  - `apps/packages/ui/src/hooks/useAudioStudioMigration.ts`
- [ ] Retain `apps/packages/ui/src/store/audiobook-studio.tsx` as a compatibility adapter during migration.
- [ ] Store shape:

```ts
export type AudioStudioWorkflow = "narration" | "podcast" | "briefing" | "music"

export type AudioStudioSection = {
  id: string
  workflow: AudioStudioWorkflow
  title: string
  bodyText: string
  speakerId?: string
  order: number
  revisionId?: string
}
```

- [ ] First-class workflow rules:
  - Narration uses chapter-oriented labels and audiobook import.
  - Podcast shows speakers and turn-based script sections.
  - Briefing shows itemized brief sections.
  - Music shows prompt, lyrics, style, duration, and provider controls.
- [ ] Add tests in:
  - `apps/packages/ui/src/store/__tests__/audio-studio.test.tsx`
  - `apps/packages/ui/src/hooks/__tests__/useAudioStudioProjects.test.tsx`
- [ ] Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/store/__tests__/audio-studio.test.tsx ../packages/ui/src/hooks/__tests__/useAudioStudioProjects.test.tsx
```

Expected green result: store supports all four workflows and refuses to overwrite local changes when the server returns a newer revision.

### Task 4.3: Build Audio Studio Components

- [ ] Create component directory `apps/packages/ui/src/components/Option/AudioStudio/`.
- [ ] Add:
  - `AudioStudioPage.tsx`
  - `WorkflowSwitcher.tsx`
  - `ProjectSidebar.tsx`
  - `ProjectHeader.tsx`
  - `NarrationWorkflow.tsx`
  - `PodcastWorkflow.tsx`
  - `BriefingWorkflow.tsx`
  - `MusicWorkflow.tsx`
  - `GenerationPanel.tsx`
  - `RenderExportPanel.tsx`
  - `MigrationBanner.tsx`
  - `CompatibilityRedirect.tsx`
- [ ] Move reusable controls from `apps/packages/ui/src/components/Option/AudiobookStudio/` into Audio Studio components:
  - text editor
  - chapter/section list
  - voice settings
  - output controls
- [ ] Keep wrappers in the old Audiobook component path during compatibility so imports do not break.
- [ ] Avoid nested UI cards and keep the studio as a dense tool surface:

```tsx
<PageShell maxWidthClassName="max-w-7xl" className="py-4">
  <AudioStudioProjectHeader />
  <WorkflowSwitcher activeWorkflow={workflow} onChange={setWorkflow} />
  <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)_320px]">
    <ProjectSidebar />
    <WorkflowEditor />
    <GenerationPanel />
  </div>
</PageShell>
```

- [ ] Add tests in `apps/packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx`.
- [ ] Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx
```

Expected green result: page renders workflow switcher, Narration/Podcast/Briefing/Music labels are visible, Narration shows imported audiobook controls, and no beta-only Audiobook heading remains on `/audio-studio`.

### Task 4.4: Add Routes And Navigation Metadata

- [ ] Create `apps/packages/ui/src/routes/option-audio-studio.tsx`.
- [ ] Modify `apps/packages/ui/src/routes/option-audiobook-studio.tsx` to render `CompatibilityRedirect`.
- [ ] Modify:
  - `apps/packages/ui/src/routes/route-registry.tsx`
  - `apps/packages/ui/src/routes/route-metadata.ts`
  - `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
  - `apps/packages/ui/src/components/Layouts/ModeSelector.tsx`
  - `apps/packages/ui/src/assets/locale/en/option.json`
  - `apps/packages/ui/src/assets/locale/en/audiobook.json`
- [ ] Add Next.js pages:
  - `apps/tldw-frontend/pages/audio-studio.tsx`
  - keep `apps/tldw-frontend/pages/audiobook-studio.tsx` as a dynamic import of compatibility route
- [ ] Route behavior:
  - `/audio-studio` opens Audio Studio.
  - `/audio-studio?workflow=narration` opens Narration.
  - `/audio-studio?workflow=podcast` opens Podcast.
  - `/audio-studio?workflow=briefing` opens Briefing.
  - `/audio-studio?workflow=music` opens Music.
  - `/audiobook-studio` opens compatibility UI and routes to Narration after migration check.
- [ ] Add route metadata tests in `apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts`.
- [ ] Add frontend route tests in `apps/tldw-frontend/__tests__/pages/audio-studio-route.test.tsx`.
- [ ] Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts __tests__/pages/audio-studio-route.test.tsx
```

Expected green result: `/audio-studio` is audited and visible, `/audiobook-studio` is marked legacy alias, and both Next pages import the intended route modules.

### Task 4.5: Wire Audiobook Migration In UI

- [ ] Extend `apps/packages/ui/src/db/dexie/audiobook-projects.ts` with read-only export helpers:
  - `listLegacyAudiobookProjectsForMigration`
  - `serializeLegacyAudiobookProjectForMigration`
  - `markLegacyAudiobookProjectMigrated`
- [ ] The compatibility route must:
  - detect local Dexie projects
  - show a migration banner when local projects exist
  - preview migration counts
  - commit selected migrations
  - redirect to `/audio-studio?workflow=narration&project=<project_id>` after commit
  - redirect to `/audio-studio?workflow=narration` when no local projects exist
- [ ] Add tests in:
  - `apps/packages/ui/src/db/dexie/__tests__/audiobook-migration.test.ts`
  - `apps/packages/ui/src/components/Option/AudioStudio/__tests__/CompatibilityRedirect.test.tsx`
- [ ] Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/db/dexie/__tests__/audiobook-migration.test.ts ../packages/ui/src/components/Option/AudioStudio/__tests__/CompatibilityRedirect.test.tsx
```

Expected green result: compatibility route preserves local projects until commit succeeds and never deletes Dexie data as part of preview.

---

## Stage 5: E2E, Documentation, And Verification

**Goal:** Prove the new route, API, Jobs, security, and compatibility behavior hold together.

**Success Criteria:** Focused backend/frontend tests pass, E2E covers all first-class workflows and compatibility route, docs document provider setup and security constraints, Bandit reports no new findings in touched backend code.

**Tests:** Focused pytest, Vitest, Playwright, Bandit.

**Status:** Not Started

### Task 5.1: Add E2E Page Objects And Workflows

- [ ] Add `apps/tldw-frontend/e2e/utils/page-objects/AudioStudioPage.ts`.
- [ ] Keep `AudiobookStudioPage.ts` as a compatibility wrapper that navigates to `/audiobook-studio`.
- [ ] Add `apps/tldw-frontend/e2e/workflows/tier-2-features/audio-studio.spec.ts`.
- [ ] Update:
  - `apps/tldw-frontend/e2e/utils/page-objects/index.ts`
  - `apps/tldw-frontend/e2e/page-mapping.ts`
  - `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
  - `apps/tldw-frontend/e2e/ux-audit/audit-v3.spec.ts`
- [ ] E2E checks:
  - `/audio-studio` renders.
  - Narration, Podcast, Briefing, and Music workflow controls are visible.
  - Narration contains existing audiobook content/chapter/generate/output flow.
  - `/audiobook-studio` reaches Narration compatibility.
  - Provider list failures show a recoverable state.
- [ ] Run:

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/tier-2-features/audio-studio.spec.ts --reporter=line
```

Expected green result: all Audio Studio browser workflow tests pass with no critical console errors.

### Task 5.2: Update Docs And Configuration Examples

- [ ] Create `Docs/Audio_Studio.md`.
- [ ] Update:
  - `Docs/API-related/TTS_API.md`
  - `Docs/API-related/Audio_Jobs_API.md`
  - `Docs/Code_Documentation/Jobs_Module.md`
  - `tldw_Server_API/Config_Files/config.txt` comments for non-secret provider settings only
  - `tldw_Server_API/Config_Files/.env.example` for secret env names
- [ ] Document:
  - `/audio-studio` route
  - `/audiobook-studio` compatibility route
  - first-class workflows
  - provider adapter model
  - ACE-Step external HTTP setup
  - endpoint allowlisting
  - idempotency and revision requirements
  - render/export separation
  - migration process
- [ ] Do not document real API keys or test secrets.
- [ ] Run:

```bash
rg -n "AUDIO_STUDIO_ACE_STEP_API_KEY=.*[A-Za-z0-9_-]{16,}" Docs tldw_Server_API/Config_Files
```

Expected output: no matches.

### Task 5.3: Run Focused Verification

- [ ] Backend tests:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio -v
```

Expected output: all Audio Studio backend tests pass.

- [ ] Frontend tests:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/services/__tests__/audio-studio.test.ts ../packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx __tests__/pages/audio-studio-route.test.tsx
```

Expected output: all targeted UI tests pass.

- [ ] Existing audiobook regression tests:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audiobooks -v
```

Expected output: existing audiobook backend tests still pass.

- [ ] Existing audiobook E2E compatibility test:

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/tier-2-features/audiobook-studio.spec.ts --reporter=line
```

Expected output: compatibility route tests pass or are updated to assert redirect into Narration.

- [ ] Bandit for touched backend code:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py tldw_Server_API/app/core/Audio_Studio tldw_Server_API/app/core/DB_Management/Collections_DB.py -f json -o /tmp/bandit_audio_studio.json
```

Expected output: no new high or medium findings in touched Audio Studio code.

- [ ] Diff hygiene:

```bash
git diff --check -- tldw_Server_API/app/api/v1/schemas/audio_studio_schemas.py tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py tldw_Server_API/app/api/v1/router_groups/content.py tldw_Server_API/app/core/Audio_Studio tldw_Server_API/app/core/DB_Management/Collections_DB.py tldw_Server_API/app/services/startup_content_jobs_pollers.py tldw_Server_API/tests/Audio_Studio apps/packages/ui/src/components/Option/AudioStudio apps/packages/ui/src/routes apps/packages/ui/src/store/audio-studio.tsx apps/packages/ui/src/hooks/useAudioStudioProjects.ts apps/packages/ui/src/hooks/useAudioStudioGeneration.tsx apps/packages/ui/src/hooks/useAudioStudioMigration.ts apps/packages/ui/src/services/audio-studio.ts apps/packages/ui/src/db/dexie/audiobook-projects.ts apps/tldw-frontend/pages/audio-studio.tsx apps/tldw-frontend/pages/audiobook-studio.tsx apps/tldw-frontend/e2e apps/tldw-frontend/__tests__/pages/audio-studio-route.test.tsx Docs/Audio_Studio.md Docs/API-related/TTS_API.md Docs/API-related/Audio_Jobs_API.md Docs/Code_Documentation/Jobs_Module.md tldw_Server_API/Config_Files/.env.example
```

Expected output: no whitespace errors in touched Audio Studio paths. If a full-worktree `git diff --check` is run, pre-existing unrelated whitespace in other dirty files must be recorded separately instead of blocking this feature.

### Task 5.4: Commit Sequence

- [ ] Commit backend schemas and persistence:

```bash
git add tldw_Server_API/app/api/v1/schemas/audio_studio_schemas.py tldw_Server_API/app/core/DB_Management/Collections_DB.py tldw_Server_API/tests/Audio_Studio/unit
git commit -m "feat(audio-studio): add project persistence model"
```

- [ ] Commit API and provider jobs:

```bash
git add tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py tldw_Server_API/app/api/v1/router_groups/content.py tldw_Server_API/app/core/Audio_Studio tldw_Server_API/app/services/startup_content_jobs_pollers.py tldw_Server_API/tests/Audio_Studio tldw_Server_API/tests/Services
git commit -m "feat(audio-studio): add generation jobs and providers"
```

- [ ] Commit frontend route:

```bash
git add apps/packages/ui/src/components/Option/AudioStudio apps/packages/ui/src/routes apps/packages/ui/src/store/audio-studio.tsx apps/packages/ui/src/hooks/useAudioStudioProjects.ts apps/packages/ui/src/hooks/useAudioStudioGeneration.tsx apps/packages/ui/src/services/audio-studio.ts apps/tldw-frontend/pages/audio-studio.tsx apps/tldw-frontend/pages/audiobook-studio.tsx
git commit -m "feat(webui): add Audio Studio route"
```

- [ ] Commit migration, docs, and E2E:

```bash
git add Docs/Audio_Studio.md Docs/API-related/TTS_API.md Docs/API-related/Audio_Jobs_API.md Docs/Code_Documentation/Jobs_Module.md tldw_Server_API/Config_Files/.env.example apps/packages/ui/src/db/dexie/audiobook-projects.ts apps/packages/ui/src/db/dexie/__tests__/audiobook-migration.test.ts apps/packages/ui/src/components/Option/AudioStudio/__tests__/CompatibilityRedirect.test.tsx apps/tldw-frontend/e2e/utils/page-objects/AudioStudioPage.ts apps/tldw-frontend/e2e/utils/page-objects/AudiobookStudioPage.ts apps/tldw-frontend/e2e/utils/page-objects/index.ts apps/tldw-frontend/e2e/workflows/tier-2-features/audio-studio.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/audiobook-studio.spec.ts apps/tldw-frontend/e2e/page-mapping.ts apps/tldw-frontend/e2e/smoke/page-inventory.ts apps/tldw-frontend/e2e/ux-audit/audit-v3.spec.ts
git commit -m "feat(audio-studio): add migration and compatibility coverage"
```

---

## Review Checkpoints

- [ ] After Stage 1, review the schema for overbroad tables, missing indexes, and revision/idempotency gaps.
- [ ] After Stage 2, review the external endpoint allowlist, redirects, secret redaction, and provider payload storage.
- [ ] After Stage 3, review render/export separation and artifact provenance.
- [ ] After Stage 4, review whether Narration, Podcast, and Briefing are visually first-class and not hidden behind Music.
- [ ] After Stage 5, review test coverage gaps and create follow-up Backlog tasks before merge.

## Non-MVP Follow-Up Tasks

- [ ] `Add Audio Studio timeline editor slice`: waveform timeline, clip dragging, trim/fade controls, and live preview.
- [ ] `Add Audio Studio advanced ACE-Step operations`: lyric repainting, style transfer, continuation, and stem-level controls if supported by configured services.
- [ ] `Add Audio Studio collaboration history`: richer revision browser and conflict resolution UI.
- [ ] `Retire Audiobook Studio compatibility route`: only after migration telemetry-free local checks and user-visible release notes are complete.
