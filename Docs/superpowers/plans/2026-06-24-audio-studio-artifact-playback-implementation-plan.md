# Audio Studio Artifact Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first Audio Studio artifact playback/download slice so generated narration/podcast/briefing artifacts can be fetched safely from `/audio-studio`, previewed in the existing timeline UI, and downloaded without exposing filesystem paths or weakening auth.

**Architecture:** Use an authenticated backend media endpoint as the artifact access strategy. Audio Studio artifact metadata remains JSON-only; media bytes are served from a scoped endpoint that validates project ownership, artifact ownership, MIME type, storage path, file existence, and optional byte ranges. The frontend uses service helpers to build the media path and, for the current authenticated WebUI flow, fetches bytes through the existing background request layer into a short-lived Blob URL for the selected timeline clip preview.

**Tech Stack:** FastAPI, Starlette `FileResponse`/`StreamingResponse`, `CollectionsDatabase`, pytest/httpx `TestClient`, Next/React, TypeScript, Vitest, existing `bgRequest` service layer.

---

## Scope Decisions

- Use `GET /api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/media` as the MVP media endpoint.
- Do not use signed URLs in this slice. Keeping auth on every request avoids token persistence, logging, and revocation problems while the page is still stabilizing.
- Support `Range: bytes=...` for direct browser/media clients and return `206 Partial Content` for valid single ranges.
- Default to inline playback; `?download=true` sets `Content-Disposition: attachment`.
- Keep provider capability discovery out of this slice. It remains a separate follow-up so endpoint security and UI playback can stabilize first.
- Limit MVP playback to audio artifact MIME types. Non-audio export/package downloads can be added in a later export-focused slice.
- Do not add waveform rendering or multi-track mixing in this slice. The UI should preview the selected clip's artifact and keep the existing timeline structure.
- Do not change `/audiobook-studio` compatibility routing in this slice.
- Keep Narration, Podcast, and Briefing workflows first-class in the UI. Playback should enhance all speech-first workflows, not shift priority toward Music.

## Chosen Access Strategy

Use a backend streaming endpoint instead of signed URLs:

- Auth stays inside normal FastAPI dependencies.
- Project and artifact ownership can be checked on every request.
- No temporary URL secrets need to be generated, logged, stored in job output, or passed through frontend state.
- The pattern already exists for Workflows artifact downloads in `tldw_Server_API/app/api/v1/endpoints/workflows.py`, including range handling and content headers.

Endpoint contract:

```http
GET /api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/media
Range: bytes=0-1023
```

Success headers:

```http
Content-Type: audio/wav
Content-Length: 1024
Accept-Ranges: bytes
Content-Range: bytes 0-1023/4096
Content-Disposition: inline; filename="clip.wav"
Cache-Control: private, no-store
```

Failure behavior:

- `404` when the project, artifact, or backing file does not exist for the current user.
- `400` when the stored path is malformed, points to a URL, or cannot be resolved safely.
- `415` when the MIME type is not allowed for the media endpoint.
- `416` when the requested range is syntactically valid but not satisfiable.

## Relevant Files

- Backend endpoint: `tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py`
- Collections DB support: `tldw_Server_API/app/core/DB_Management/Collections_DB.py`
- Storage path helpers: `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- Backend tests: `tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_artifact_media_api.py`
- Existing render/export tests: `tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_render_export_api.py`
- Frontend service: `apps/packages/ui/src/services/audio-studio.ts`
- Frontend background transport: `apps/packages/ui/src/services/background-proxy.ts`
- Frontend service tests: `apps/packages/ui/src/services/__tests__/audio-studio.test.ts`
- Frontend query hooks: `apps/packages/ui/src/hooks/useAudioStudioProjects.ts`
- Frontend store exports: `apps/packages/ui/src/store/audio-studio.tsx`
- Timeline UI: `apps/packages/ui/src/components/Option/AudioStudio/TimelineEditor.tsx`
- Audio Studio page tests: `apps/packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx`
- Public docs: `Docs/Audio_Studio.md`
- Backlog task: `TASK-2357`

## Stage 1: Backend Failing Tests

**Goal:** Lock the endpoint security and range behavior before implementation.

**Success Criteria:**

- Tests cover full-file playback, range playback, download disposition, cross-user/project isolation, MIME rejection, path rejection, missing file behavior, and unsatisfiable ranges.
- Tests cover strict storage-root containment, including absolute paths outside the user artifact roots and symlink escape.
- Tests cover auth behavior enough to prove the endpoint is not only protected by router-local dependency overrides.
- Tests fail before the endpoint exists.

**Tests:**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_artifact_media_api.py -v
```

**Implementation Steps:**

- [ ] Create `tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_artifact_media_api.py`.
- [ ] Reuse the existing Audio Studio router-only `FastAPI` fixture pattern from `test_audio_studio_projects_api.py`.
- [ ] Add a helper that creates a project through the API and inserts an artifact through `CollectionsDatabase.for_user(user_id=1)`.
- [ ] Store normal test artifacts under the configured per-user output root, not arbitrary absolute temp paths:

```python
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

outputs_dir = DatabasePaths.get_user_base_directory(1) / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
wav_path = outputs_dir / "clip.wav"
wav_path.write_bytes(audio_bytes)
storage_path = db.resolve_output_storage_path("clip.wav")
```

- [ ] Insert normal artifacts with `storage_path=storage_path`, so the endpoint serves relative, per-user output filenames.
- [ ] Test full media response:

```python
response = client.get(
    f"/api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/media"
)

assert response.status_code == 200
assert response.content == audio_bytes
assert response.headers["accept-ranges"] == "bytes"
assert response.headers["content-type"].startswith("audio/wav")
assert "storage_path" not in response.text
```

- [ ] Test range media response:

```python
response = client.get(
    f"/api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/media",
    headers={"Range": "bytes=0-9"},
)

assert response.status_code == 206
assert response.content == audio_bytes[:10]
assert response.headers["content-range"] == f"bytes 0-9/{len(audio_bytes)}"
assert response.headers["accept-ranges"] == "bytes"
```

- [ ] Test a valid suffix range:

```python
response = client.get(
    f"/api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/media",
    headers={"Range": "bytes=-10"},
)

assert response.status_code == 206
assert response.content == audio_bytes[-10:]
```

- [ ] Test `?download=true` returns `Content-Disposition: attachment`.
- [ ] Test an artifact owned by another user or attached to another project returns `404`.
- [ ] Test unsupported MIME, for example `text/html` or `application/x-msdownload`, returns `415`.
- [ ] Test URL-like storage paths such as `https://example.invalid/audio.wav` return `400`.
- [ ] Test arbitrary absolute paths outside the per-user output/temp-output roots return `400`.
- [ ] Test relative traversal values such as `../clip.wav` return `400`.
- [ ] Test a symlink inside the output root that resolves outside the output root returns `400`.
- [ ] Test a relative filename that exists in both `outputs` and temp outputs. Use SHA-256 hex for `content_hash`; if both candidates exist, the endpoint must choose the single file whose SHA-256 hex digest matches `row.content_hash` or reject ambiguity.
- [ ] Test extension/MIME mismatch, for example `storage_path="clip.html"` with `mime_type="audio/wav"`, returns `415`.
- [ ] Test actual file size mismatch against `row.size_bytes` returns `409` or `400`; choose one status in implementation and keep it consistent.
- [ ] Test a missing backing file returns `404`.
- [ ] Test `Range: bytes=999999-999999` returns `416`.
- [ ] Test malformed ranges return `416`:
  - `items=0-10`
  - `bytes=10-1`
  - `bytes=0-1,2-3`
  - `bytes=`
  - `bytes=-0`
- [ ] Add an auth smoke test without overriding `get_request_user`:
  - Configure `AUTH_MODE=single_user` and `SINGLE_USER_API_KEY`.
  - Assert a request with no credentials is rejected.
  - Assert a request with the correct `X-API-KEY` can reach the endpoint.
- [ ] Keep the router-local user override tests for multi-user-style user isolation by switching the overridden `User(id=...)` between user `1` and user `2`.
- [ ] If a full JWT fixture is too expensive for this slice, document that the endpoint relies on the existing shared `get_request_user` JWT tests and include the dependency-level user-isolation tests above.

## Stage 2: Backend Media Endpoint

**Goal:** Implement secure artifact media serving with ownership checks and byte-range support.

**Success Criteria:**

- Endpoint passes Stage 1 tests.
- No raw filesystem paths are exposed in response bodies or headers.
- Stored paths cannot escape the current user's Audio Studio artifact roots, even if the DB row is malformed.
- MIME, extension, and size metadata must agree before bytes are served.
- The implementation is narrowly scoped to Audio Studio and follows the existing Workflows artifact range behavior where practical.

**Tests:**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_artifact_media_api.py -v
python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_render_export_api.py -v
```

**Implementation Steps:**

- [ ] In `audio_studio.py`, add imports for `mimetypes`, `re` if needed, `Iterator`, `Request`, `FileResponse`, `StreamingResponse`, `DatabasePaths`, storage exceptions if needed, and filesystem path handling via an alias such as `from pathlib import Path as FileSystemPath`.
- [ ] Add a conservative MIME allowlist helper:

```python
_AUDIO_STUDIO_PLAYBACK_MIME_TYPES = {
    "audio/aac",
    "audio/flac",
    "audio/m4a",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/wave",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
}


def _is_playable_audio_mime(mime_type: str | None) -> bool:
    normalized = (mime_type or "").split(";", 1)[0].strip().lower()
    return normalized in _AUDIO_STUDIO_PLAYBACK_MIME_TYPES
```

- [ ] Add an extension allowlist that maps the same supported audio formats to suffixes, for example `.wav`, `.mp3`, `.m4a`, `.mp4`, `.aac`, `.flac`, `.ogg`, and `.webm`.
- [ ] Validate MIME and extension together:
  - Normalize `row.mime_type` first.
  - If MIME is missing, infer with `mimetypes.guess_type(path.name)` and still enforce the allowlist.
  - Reject an allowed MIME with a dangerous or mismatched extension.
  - Do not serve `text/html`, SVG, executable, or generic unknown files from this endpoint.
- [ ] Add `_load_audio_studio_artifact_or_404(collections_db, project, artifact_id)` using `collections_db.list_audio_studio_artifacts(project_row_id=project.id, artifact_id=artifact_id, limit=1)`.
- [ ] Add safe filename generation for `Content-Disposition`. Use the artifact id plus an extension derived from the MIME type or file suffix; strip CR/LF, quotes, slashes, and path separators.
- [ ] Add storage path resolution:
  - Reject empty storage paths.
  - Reject URL-like values containing `://`.
  - Prefer relative filenames normalized by `collections_db.resolve_output_storage_path(...)`.
  - Reject relative paths with separators or traversal, matching the output storage helper behavior.
  - Build explicit candidate roots for the current user:

```python
output_roots = [
    DatabasePaths.get_user_base_directory(current_user.id) / "outputs",
    DatabasePaths.get_user_temp_outputs_dir(current_user.id),
]
```

  - For relative storage paths, join the normalized filename to those roots and collect existing regular-file candidates.
  - If exactly one candidate exists, use it.
  - If multiple candidates exist for the same relative filename, verify `content_hash` as a SHA-256 hex digest against candidates and use the single hash match.
  - If multiple candidates match or none match, reject with `409 Conflict` so same-name output/temp-output collisions cannot serve the wrong file.
  - Reject absolute paths by default. If compatibility with existing absolute rows is required, allow an absolute path only when its strict resolved path is inside one of the explicit current-user roots.
  - Use strict realpath containment after following symlinks:

```python
resolved_file = candidate.resolve(strict=True)
resolved_root = root.resolve(strict=True)
resolved_file.relative_to(resolved_root)
```

  - Reject directories and missing files.
  - Reject symlinks that resolve outside the configured output roots.
  - Keep all path-related error details generic so host paths are not exposed.
- [ ] Validate `row.size_bytes` before serving:
  - Compare it to `path.stat().st_size` when `row.size_bytes` is not `None`.
  - Reject mismatches with a deterministic status, preferably `409 Conflict` for stale artifact metadata.
- [ ] Treat `content_hash` as metadata in this slice unless the file is small enough for cheap verification. Document that full checksum verification is a future hardening option rather than silently implying it is checked.
- [ ] Exception to the previous item: when root collision resolution finds multiple same-name candidates, verify `content_hash` to disambiguate or reject the request.
- [ ] Add single-range parsing compatible with Workflows:

```python
def _parse_single_byte_range(range_header: str | None, file_size: int) -> tuple[int, int] | None:
    if not range_header:
        return None
    if not range_header.startswith("bytes="):
        raise HTTPException(status_code=416, detail="Unsupported range unit")
    # Support "bytes=start-end", "bytes=start-", and "bytes=-suffix".
```

- [ ] Reject multi-range requests instead of trying to assemble multipart byte responses.
- [ ] Reject empty ranges, suffix length zero, and `start > end`.
- [ ] Add an iterator helper for partial responses:

```python
def _iter_file_range(path: FileSystemPath, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as file_obj:
        file_obj.seek(start)
        remaining = length
        while remaining > 0:
            chunk = file_obj.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
```

- [ ] Add the endpoint after `list_audio_studio_artifacts`:

```python
@router.get("/projects/{project_id}/artifacts/{artifact_id}/media")
async def get_audio_studio_artifact_media(
    request: Request,
    project_id: AudioStudioIdPath,
    artifact_id: AudioStudioIdPath,
    download: bool = Query(False),
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
):
    project = _load_project_or_404(collections_db, project_id)
    artifact = _load_audio_studio_artifact_or_404(collections_db, project, artifact_id)
    ...
```

- [ ] Return `FileResponse` for full responses with `Accept-Ranges: bytes`, `Cache-Control: private, no-store`, and safe `Content-Disposition`.
- [ ] Return `StreamingResponse` with `206`, `Content-Range`, `Content-Length`, `Accept-Ranges`, `Cache-Control`, and safe `Content-Disposition` for range responses.
- [ ] Keep error messages generic enough to avoid leaking host paths.

## Stage 3: Backend Contract Documentation

**Goal:** Document the new artifact access strategy and endpoint behavior for future slices.

**Success Criteria:**

- `Docs/Audio_Studio.md` describes the backend-streaming decision, endpoint path, supported MIME scope, range behavior, and the reason signed URLs are deferred.

**Tests:**

```bash
git diff --check -- Docs/Audio_Studio.md
```

**Implementation Steps:**

- [ ] Add an "Artifact Media Access" subsection near the existing Audio Studio implementation/roadmap section.
- [ ] Document:
  - Authenticated endpoint strategy.
  - `GET /api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/media`.
  - `Range` support for browser playback.
  - `download=true` behavior.
  - No raw storage paths in API responses.
  - Files are served only from current-user output/temp-output roots after realpath containment checks.
  - Auth-mode coverage: single-user API key smoke coverage plus user-isolation coverage for multi-user-style dependency results.
  - Signed URLs deferred until there is a concrete external-client requirement.

## Stage 4: Frontend Service Helpers

**Goal:** Add typed service helpers, artifact metadata listing, background binary transport support, and authenticated Blob fetching.

**Success Criteria:**

- Service tests cover artifact listing, path encoding, `download=true`, and `bgRequest` array-buffer use.
- Background transport tests cover array-buffer direct bypass for `/api/v1/audio-studio/` media paths.
- UI code does not hand-build media endpoint strings.

**Tests:**

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/services/__tests__/audio-studio.test.ts
bunx vitest run ../packages/ui/src/services/__tests__/background-proxy.test.ts
```

**Implementation Steps:**

- [ ] Add and export an `AudioStudioArtifact` type matching the backend artifact response:

```ts
export type AudioStudioArtifact = {
  artifact_id: string
  artifact_type: string
  provider?: string | null
  mime_type?: string | null
  size_bytes?: number | null
  source_resource_kind?: string | null
  source_resource_id?: string | null
  source_revision_id?: string | null
  metadata?: Record<string, unknown>
  created_at?: string
}

export type AudioStudioArtifactListResponse = {
  artifacts: AudioStudioArtifact[]
  limit: number
  offset: number
  total?: number
}
```

- [ ] Add `listAudioStudioArtifacts(projectId)`:

```ts
export const listAudioStudioArtifacts = async (
  projectId: string,
): Promise<AudioStudioArtifact[]> => {
  const response = await bgRequest<AudioStudioArtifactListResponse>({
    path: apiPath(`${projectPath(projectId)}/artifacts`),
    method: "GET",
  })
  return response.artifacts
}
```

- [ ] Add media path construction using the existing `API_BASE`/`projectPath` convention. The path must include `/api/v1`:

```ts
export function getAudioStudioArtifactMediaPath(
  projectId: string,
  artifactId: string,
  options: { download?: boolean } = {},
): string {
  const query = options.download ? "?download=true" : "";
  return apiPath(
    `${projectPath(projectId)}/artifacts/${encodeURIComponent(artifactId)}/media${query}`,
  );
}
```

- [ ] Add authenticated Blob fetch helper:

```ts
export async function fetchAudioStudioArtifactBlob(
  projectId: string,
  artifact: Pick<AudioStudioArtifact, "artifact_id" | "mime_type">,
): Promise<Blob> {
  const data = await bgRequest<ArrayBuffer>({
    path: getAudioStudioArtifactMediaPath(projectId, artifact.artifact_id),
    method: "GET",
    responseType: "arrayBuffer",
  });

  return new Blob([data], { type: artifact.mime_type || "application/octet-stream" });
}
```

- [ ] Update `apps/packages/ui/src/services/background-proxy.ts` so the existing array-buffer direct-bypass covers Audio Studio media:

```ts
const shouldBypassBackground =
  responseType === "arrayBuffer" &&
  typeof path === "string" &&
  (path.includes("/api/v1/audio/") || path.includes("/api/v1/audio-studio/"))
```

- [ ] Add tests asserting project/artifact IDs are URL-encoded.
- [ ] Add tests asserting `download=true` is appended only when requested.
- [ ] Add a test asserting `getAudioStudioArtifactMediaPath("p 1", "a/1")` returns an `/api/v1/audio-studio/...` path with encoded segments.
- [ ] Add tests asserting `listAudioStudioArtifacts` calls `bgRequest` with `GET /api/v1/audio-studio/projects/{projectId}/artifacts`.
- [ ] Add tests asserting `fetchAudioStudioArtifactBlob` uses `responseType: "arrayBuffer"` and returns a `Blob` with the artifact MIME type.
- [ ] Add background-proxy tests or extend existing ones to assert `/api/v1/audio-studio/.../media` array-buffer requests use the same direct binary path as `/api/v1/audio/...`.

## Stage 5: Timeline Playback UI

**Goal:** Make generated clip artifacts playable from the Audio Studio timeline without redesigning the editor.

**Success Criteria:**

- Selecting a timeline clip with `artifact_id` fetches the artifact through the service helper and renders a native audio control.
- Artifact metadata is loaded explicitly through `listAudioStudioArtifacts(projectId)` and looked up by `artifact_id`; clips are not assumed to contain MIME or size metadata.
- Blob URLs are revoked when selection changes or the component unmounts.
- Clips without artifacts show a compact unavailable state rather than a broken player.
- Large artifacts are not fetched eagerly into memory.
- User-facing download in this MVP is small-file-only and uses the authenticated Blob already fetched through `bgRequest`; large-file WebUI download is deferred to `TASK-2358`.
- Existing Narration, Podcast, and Briefing workflows remain prominent and unchanged.

**Tests:**

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx
```

**Implementation Steps:**

- [ ] Add query keys and a hook in `useAudioStudioProjects.ts`:

```ts
artifacts: (projectId: string | null) =>
  [...audioStudioProjectQueryKeys.all, "projects", projectId, "artifacts"] as const
```

```ts
export const useAudioStudioArtifacts = (projectId: string | null) =>
  useQuery({
    queryKey: audioStudioProjectQueryKeys.artifacts(projectId),
    queryFn: () => {
      if (!projectId) return Promise.resolve([])
      return listAudioStudioArtifacts(projectId)
    },
    enabled: Boolean(projectId),
  })
```

- [ ] In `AudioStudioPage.tsx`, call `useAudioStudioArtifacts(activeProject?.project_id ?? null)` and pass `artifactsQuery.data ?? []` into `TimelineEditor`.
- [ ] Update `TimelineEditor` props:

```ts
type TimelineEditorProps = {
  artifacts?: AudioStudioArtifact[]
}

export const TimelineEditor: React.FC<TimelineEditorProps> = ({ artifacts = [] }) => {
  ...
}
```

- [ ] Build a memoized artifact lookup:

```ts
const artifactById = useMemo(
  () => new Map(artifacts.map((artifact) => [artifact.artifact_id, artifact])),
  [artifacts],
)
```

- [ ] In `TimelineEditor.tsx`, derive the selected clip's artifact id from the existing selected clip state.
- [ ] Resolve `selectedArtifact` with `selectedClip?.artifact_id ? artifactById.get(selectedClip.artifact_id) : null`.
- [ ] Define a frontend preview fetch threshold, for example:

```ts
const MAX_BLOB_PREVIEW_BYTES = 25 * 1024 * 1024
```

- [ ] If `selectedArtifact.size_bytes` exceeds the threshold, do not call `fetchAudioStudioArtifactBlob` and do not render a native `<a href>` or `<audio src>` pointing at the media endpoint. Native media/link requests cannot attach `X-API-KEY` in single-user mode, and query-string secrets are not allowed.
- [ ] For over-threshold artifacts in this MVP UI, render a compact disabled preview/download state. The backend endpoint remains available to authenticated API clients that can send headers; WebUI large-file streaming/download needs a later auth-aware transport slice.
- [ ] Keep large-artifact WebUI transport in follow-up `TASK-2358`, choosing between a short-lived signed URL, service-worker/header-injection route, or streamed authenticated frontend fetch.
- [ ] Add local state:

```ts
const [previewUrl, setPreviewUrl] = useState<string | null>(null);
const [previewError, setPreviewError] = useState<string | null>(null);
const [isPreviewLoading, setIsPreviewLoading] = useState(false);
```

- [ ] Fetch and revoke Blob URLs in an effect:

```ts
useEffect(() => {
  let cancelled = false;
  let objectUrl: string | null = null;

  async function loadPreview() {
    if (!projectId || !selectedArtifact) {
      setPreviewUrl(null);
      return;
    }
    if ((selectedArtifact.size_bytes ?? 0) > MAX_BLOB_PREVIEW_BYTES) {
      setPreviewUrl(null);
      return;
    }
    setIsPreviewLoading(true);
    setPreviewError(null);
    try {
      const blob = await fetchAudioStudioArtifactBlob(projectId, selectedArtifact);
      if (cancelled) return;
      objectUrl = URL.createObjectURL(blob);
      setPreviewUrl(objectUrl);
    } catch {
      if (!cancelled) setPreviewError("Preview unavailable");
    } finally {
      if (!cancelled) setIsPreviewLoading(false);
    }
  }

  void loadPreview();

  return () => {
    cancelled = true;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  };
}, [projectId, selectedArtifact?.artifact_id]);
```

- [ ] Render the preview in the selected clip inspector:

```tsx
{previewUrl ? (
  <>
    <audio aria-label="Selected clip audio preview" controls src={previewUrl} />
    <a href={previewUrl} download={`${selectedArtifact.artifact_id}.wav`}>
      Download clip audio
    </a>
  </>
) : (
  <p className="...">No playable artifact</p>
)}
```

- [ ] Use a Blob URL for small-file downloads; never put API keys or tokens into a media endpoint URL.
- [ ] Keep layout dimensions stable so the inspector does not jump when preview loads.
- [ ] Do not replace the existing timeline playhead simulation with a full audio transport in this slice.

## Stage 6: Frontend UI Tests

**Goal:** Verify the UI path without depending on real backend media bytes.

**Success Criteria:**

- Tests cover playable clip, missing artifact, and failed preview fetch states.
- Tests cover artifact metadata lookup and large-artifact no-fetch behavior.
- Object URL creation and revocation are mocked and asserted.

**Tests:**

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx
```

**Implementation Steps:**

- [ ] Mock `fetchAudioStudioArtifactBlob` to return a small audio `Blob`.
- [ ] Mock `listAudioStudioArtifacts` or `useAudioStudioArtifacts` so the active clip's `artifact_id` resolves to metadata with `mime_type` and `size_bytes`.
- [ ] Mock `URL.createObjectURL` and `URL.revokeObjectURL`.
- [ ] Select a clip with an artifact and assert `audio[aria-label="Selected clip audio preview"]` is rendered with the mock blob URL.
- [ ] Assert the small-file download link uses the Blob URL, not `/api/v1/audio-studio/.../media`, and has a safe `download` filename.
- [ ] Select or render a clip without `artifact_id` and assert the unavailable state is rendered.
- [ ] Render a clip whose `artifact_id` is not present in the artifact list and assert the missing-artifact state is rendered without calling `fetchAudioStudioArtifactBlob`.
- [ ] Render a clip whose artifact exceeds `MAX_BLOB_PREVIEW_BYTES` and assert no Blob fetch occurs, no media endpoint link or Blob download link is rendered, and the disabled large-artifact state is shown.
- [ ] Make the service helper reject and assert the compact error state is rendered.

## Stage 7: Focused Verification

**Goal:** Confirm the backend, frontend, docs, and security gates pass for the touched scope.

**Success Criteria:**

- All focused backend and frontend tests pass.
- Bandit reports no new findings in touched backend endpoint code.
- Diff has no whitespace errors.
- Backlog task records verification results.

**Commands:**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_artifact_media_api.py -v
python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_render_export_api.py -v
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py -f json -o /tmp/bandit_audio_studio_artifacts.json
```

```bash
cd apps/tldw-frontend
bunx vitest run \
  ../packages/ui/src/services/__tests__/audio-studio.test.ts \
  ../packages/ui/src/services/__tests__/background-proxy.test.ts \
  ../packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx
```

```bash
git diff --check -- \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py \
  tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_artifact_media_api.py \
  apps/packages/ui/src/services/audio-studio.ts \
  apps/packages/ui/src/services/background-proxy.ts \
  apps/packages/ui/src/services/__tests__/audio-studio.test.ts \
  apps/packages/ui/src/services/__tests__/background-proxy.test.ts \
  apps/packages/ui/src/hooks/useAudioStudioProjects.ts \
  apps/packages/ui/src/store/audio-studio.tsx \
  apps/packages/ui/src/components/Option/AudioStudio/TimelineEditor.tsx \
  apps/packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx \
  Docs/Audio_Studio.md
```

## Stage 8: Task Finalization

**Goal:** Close the implementation slice cleanly.

**Success Criteria:**

- `TASK-2357` is updated with the plan result if this plan is the only deliverable, or a new implementation task is created before code edits begin.
- If implementation proceeds under this task, it records touched files, verification commands, and final summary.
- Commit includes the Backlog task changes and implementation/doc/test changes.

**Implementation Steps:**

- [ ] Update Backlog task status and notes before starting implementation.
- [ ] If the implementation grows beyond the endpoint plus preview UI, split follow-up work into separate Backlog tasks.
- [ ] Keep provider capability discovery as a separate task.
- [ ] Keep waveform/editor improvements as the next editing slice.

## Rollback Plan

- Backend: remove the media endpoint and helpers from `audio_studio.py`; artifact metadata endpoints continue to work.
- Frontend: remove the Blob preview helper usage; the timeline returns to metadata-only previews.
- Docs: remove the "Artifact Media Access" subsection if the endpoint is reverted.

## Known Follow-Ups Not Included

- Provider capability registry and `/audio-studio/providers/capabilities`.
- Signed URL support for external clients, if a real external-client requirement appears.
- Large-artifact WebUI playback/download transport that preserves auth without query-string secrets (`TASK-2358`).
- Waveform extraction/rendering.
- Multi-track timeline playback and export composition.
- Non-audio package artifact download UI.
