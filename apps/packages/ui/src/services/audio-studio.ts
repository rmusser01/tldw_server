import { bgRequest } from "@/services/background-proxy"
import { buildQuery } from "@/services/resource-client"

export type AudioStudioWorkflow = "narration" | "podcast" | "briefing" | "music"
export type AudioStudioProjectStatus = "draft" | "active" | "archived" | string
export type AudioStudioTrackKind =
  | "speech"
  | "music"
  | "sfx"
  | "ambience"
  | "mixed"
  | string
export type AudioStudioClipType =
  | "speech"
  | "music"
  | "sfx"
  | "ambience"
  | "imported"
  | "render"
  | string

export type AudioStudioWorkflowSummary = {
  id: AudioStudioWorkflow
  label: string
  description?: string
  priority?: number
}

export type AudioStudioWorkflowListResponse = {
  workflows: AudioStudioWorkflowSummary[]
}

export type AudioStudioSection = {
  section_id: string
  workflow: AudioStudioWorkflow
  title: string
  body_text?: string
  speaker_id?: string
  order: number
  revision_id?: string
  settings?: Record<string, unknown>
}

export type AudioStudioTrack = {
  track_id: string
  name: string
  kind: AudioStudioTrackKind
  order: number
  muted?: boolean
  solo?: boolean
  revision_id?: string
  settings?: Record<string, unknown>
}

export type AudioStudioClip = {
  clip_id: string
  track_id: string
  section_id?: string
  artifact_id?: string
  start_ms: number
  duration_ms?: number
  volume?: number
  revision_id?: string
  settings?: Record<string, unknown>
}

export type AudioStudioProject = {
  project_id: string
  title: string
  workflow: AudioStudioWorkflow
  status: AudioStudioProjectStatus
  description?: string
  revision_id?: string
  current_revision_id?: string
  updated_at?: string
  created_at?: string
  settings?: Record<string, unknown>
  sections?: AudioStudioSection[]
  tracks?: AudioStudioTrack[]
  clips?: AudioStudioClip[]
}

export type AudioStudioProjectListResponse = {
  projects: AudioStudioProject[]
  limit: number
  offset: number
  total?: number
}

export type ListAudioStudioProjectsParams = {
  workflow?: AudioStudioWorkflow
  includeArchived?: boolean
}

export type CreateAudioStudioProjectRequest = {
  title: string
  workflow: AudioStudioWorkflow
  description?: string
  settings?: Record<string, unknown>
}

export type UpdateAudioStudioProjectRequest = Partial<
  Pick<CreateAudioStudioProjectRequest, "title" | "description" | "settings">
> & {
  base_revision_id: string
}

export type AudioStudioSectionUpsertRequest = {
  base_revision_id: string
  title?: string
  body_text?: string
  speaker_id?: string
  order_index?: number
  settings?: Record<string, unknown>
  metadata?: Record<string, unknown>
}

export type AudioStudioSectionResponse = {
  section_id: string
  workflow: AudioStudioWorkflow
  title?: string | null
  body_text?: string | null
  speaker_id?: string | null
  order_index: number
  settings?: Record<string, unknown>
  current_revision_id?: string | null
  archived_at?: string | null
}

export type AudioStudioTrackUpsertRequest = {
  base_revision_id: string
  name: string
  kind: AudioStudioTrackKind
  order_index?: number
  muted?: boolean
  solo?: boolean
  volume?: number
  settings?: Record<string, unknown>
  metadata?: Record<string, unknown>
}

export type AudioStudioTrackResponse = {
  track_id: string
  name: string
  kind: AudioStudioTrackKind
  order_index: number
  muted: boolean
  solo: boolean
  volume: number
  settings?: Record<string, unknown>
  current_revision_id?: string | null
  archived_at?: string | null
}

export type AudioStudioClipUpsertRequest = {
  base_revision_id: string
  track_id: string
  section_id?: string
  title?: string
  clip_type: AudioStudioClipType
  artifact_id?: string
  start_ms: number
  duration_ms?: number
  volume?: number
  fade_in_ms?: number
  fade_out_ms?: number
  muted?: boolean
  settings?: Record<string, unknown>
  metadata?: Record<string, unknown>
}

export type AudioStudioClipResponse = {
  clip_id: string
  section_id?: string | null
  track_id: string
  title?: string | null
  clip_type: AudioStudioClipType
  start_ms: number
  duration_ms?: number | null
  volume: number
  fade_in_ms: number
  fade_out_ms: number
  muted: boolean
  artifact_id?: string | null
  settings?: Record<string, unknown>
  current_revision_id?: string | null
  archived_at?: string | null
}

export type AudioStudioGenerationCreateRequest = {
  kind: "speech" | "music" | "script" | string
  provider: string | Record<string, unknown>
  idempotency_key: string
  target_resource_kind:
    | "section"
    | "track"
    | "clip"
    | "artifact"
    | "render"
    | "export"
    | string
  target_resource_id: string
  target_revision_id: string
  options?: Record<string, unknown>
}

export type AudioStudioRenderCreateRequest = {
  idempotency_key: string
  timeline_revision_id?: string
  settings?: Record<string, unknown>
}

export type AudioStudioExportCreateRequest = {
  idempotency_key: string
  format: "zip" | "wav" | "mp3" | string
  render_id?: string
  settings?: Record<string, unknown>
}

export type AudioStudioJobResponse = {
  job_id: string
  status: string
  project_id?: string
}

export type AudiobookMigrationPreviewRequest = {
  legacy_project_ids?: string[]
}

export type AudiobookMigrationCommitRequest = {
  legacy_project_ids?: string[]
  idempotency_key: string
}

export type AudiobookMigrationResponse = {
  migration_id?: string
  status: string
  projects?: Array<{ legacy_project_id: string; project_id?: string; status: string }>
}

const JSON_HEADERS = { "Content-Type": "application/json" }
const API_BASE = "/api/v1/audio-studio"

const apiPath = (path: string) => path as any
const projectPath = (projectId: string) =>
  apiPath(`${API_BASE}/projects/${encodeURIComponent(projectId)}`)
const resourcePath = (projectId: string, resource: string, resourceId: string) =>
  apiPath(
    `${API_BASE}/projects/${encodeURIComponent(projectId)}/${resource}/${encodeURIComponent(resourceId)}`
  )

export const listAudioStudioWorkflows = async (): Promise<
  AudioStudioWorkflowSummary[]
> => {
  const response = await bgRequest<AudioStudioWorkflowListResponse>({
    path: apiPath(`${API_BASE}/workflows`),
    method: "GET"
  })
  return response.workflows
}

export const listAudioStudioProjects = async (
  params: ListAudioStudioProjectsParams = {}
): Promise<AudioStudioProject[]> => {
  const query = buildQuery({
    workflow: params.workflow,
    include_archived: params.includeArchived
  })
  const response = await bgRequest<AudioStudioProjectListResponse>({
    path: apiPath(`${API_BASE}/projects${query}`),
    method: "GET"
  })
  return response.projects
}

export const createAudioStudioProject = async (
  body: CreateAudioStudioProjectRequest
): Promise<AudioStudioProject> =>
  bgRequest<AudioStudioProject>({
    path: apiPath(`${API_BASE}/projects`),
    method: "POST",
    headers: JSON_HEADERS,
    body
  })

export const updateAudioStudioProject = async (
  projectId: string,
  body: UpdateAudioStudioProjectRequest
): Promise<AudioStudioProject> =>
  bgRequest<AudioStudioProject>({
    path: projectPath(projectId),
    method: "PATCH",
    headers: JSON_HEADERS,
    body
  })

export const upsertAudioStudioSection = async (
  projectId: string,
  sectionId: string,
  body: AudioStudioSectionUpsertRequest
): Promise<AudioStudioSectionResponse> =>
  bgRequest<AudioStudioSectionResponse>({
    path: resourcePath(projectId, "sections", sectionId),
    method: "PUT",
    headers: JSON_HEADERS,
    body
  })

export const upsertAudioStudioTrack = async (
  projectId: string,
  trackId: string,
  body: AudioStudioTrackUpsertRequest
): Promise<AudioStudioTrackResponse> =>
  bgRequest<AudioStudioTrackResponse>({
    path: resourcePath(projectId, "tracks", trackId),
    method: "PUT",
    headers: JSON_HEADERS,
    body
  })

export const upsertAudioStudioClip = async (
  projectId: string,
  clipId: string,
  body: AudioStudioClipUpsertRequest
): Promise<AudioStudioClipResponse> =>
  bgRequest<AudioStudioClipResponse>({
    path: resourcePath(projectId, "clips", clipId),
    method: "PUT",
    headers: JSON_HEADERS,
    body
  })

export const createAudioStudioGeneration = async (
  projectId: string,
  body: AudioStudioGenerationCreateRequest
): Promise<AudioStudioJobResponse> =>
  bgRequest<AudioStudioJobResponse>({
    path: apiPath(`${projectPath(projectId)}/generations`),
    method: "POST",
    headers: JSON_HEADERS,
    body
  })

export const createAudioStudioRender = async (
  projectId: string,
  body: AudioStudioRenderCreateRequest
): Promise<AudioStudioJobResponse> =>
  bgRequest<AudioStudioJobResponse>({
    path: apiPath(`${projectPath(projectId)}/renders`),
    method: "POST",
    headers: JSON_HEADERS,
    body
  })

export const createAudioStudioExport = async (
  projectId: string,
  body: AudioStudioExportCreateRequest
): Promise<AudioStudioJobResponse> =>
  bgRequest<AudioStudioJobResponse>({
    path: apiPath(`${projectPath(projectId)}/exports`),
    method: "POST",
    headers: JSON_HEADERS,
    body
  })

export const previewAudiobookMigration = async (
  body: AudiobookMigrationPreviewRequest
): Promise<AudiobookMigrationResponse> =>
  bgRequest<AudiobookMigrationResponse>({
    path: apiPath(`${API_BASE}/migrations/audiobook/preview`),
    method: "POST",
    headers: JSON_HEADERS,
    body
  })

export const commitAudiobookMigration = async (
  body: AudiobookMigrationCommitRequest
): Promise<AudiobookMigrationResponse> =>
  bgRequest<AudiobookMigrationResponse>({
    path: apiPath(`${API_BASE}/migrations/audiobook/commit`),
    method: "POST",
    headers: JSON_HEADERS,
    body
  })
