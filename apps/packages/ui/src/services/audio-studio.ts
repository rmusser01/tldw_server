import { bgRequest, type BgRequestInit } from "@/services/background-proxy"
import { buildQuery } from "@/services/resource-client"
import { resolveBrowserRequestTransport } from "@/services/tldw/request-core"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import type { LegacyAudiobookProjectMigrationPayload } from "@/db/dexie/audiobook-projects"

import { normalizeTtsProvidersResponse } from "@/services/tldw/audio-providers"
import { extractVoices } from "@/services/tldw/audio-voices"
import { normalizeTtsProviderKey } from "@/services/tldw/tts-provider-keys"
import { requestScopeFields, type ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"

export type AudioStudioRequestOptions = {
  requestScope?: ServicePromptRequestScope
  signal?: AbortSignal
  assertCurrent?: () => void
  assertCanWrite?: () => void
}

const request = async <T>(init: BgRequestInit, options?: AudioStudioRequestOptions): Promise<T> => {
  options?.assertCurrent?.()
  if (!["GET", "HEAD"].includes(String(init.method ?? "GET").toUpperCase())) options?.assertCanWrite?.()
  const fields = options?.requestScope ? requestScopeFields(options.requestScope) : {}
  const result = await bgRequest<T>({ ...init, ...fields,
    ...((fields.headers || init.headers) ? { headers: { ...fields.headers, ...init.headers } } : {}),
    ...(options ? { abortSignal: options.signal, timeoutMs: 15000 } : {})
  })
  options?.assertCurrent?.()
  return result
}

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
  order_index?: number
  revision_id?: string
  settings?: Record<string, unknown>
}

export type AudioStudioTrack = {
  track_id: string
  name: string
  kind: AudioStudioTrackKind
  order: number
  order_index?: number
  muted?: boolean
  solo?: boolean
  revision_id?: string
  settings?: Record<string, unknown>
}

export type AudioStudioClip = {
  clip_id: string
  track_id: string
  section_id?: string
  title?: string
  clip_type?: AudioStudioClipType
  artifact_id?: string
  start_ms: number
  duration_ms?: number
  volume?: number
  fade_in_ms?: number
  fade_out_ms?: number
  muted?: boolean
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

export type AudioStudioArtifact = {
  artifact_id: string
  artifact_type: string
  provider?: string | null
  mime_type?: string | null
  size_bytes?: number | null
  source_resource_kind?: string | null
  source_resource_id?: string | null
  source_revision_id?: string | null
  metadata: Record<string, unknown>
  created_at: string
}

export type AudioStudioArtifactListResponse = {
  artifacts: AudioStudioArtifact[]
  limit: number
  offset: number
}

export type AudioStudioMediaTicketPurpose = "playback" | "download"

export type AudioStudioMediaTicketResponse = {
  ticket_path: string
  ticket_url?: string | null
  expires_at: string
  purpose: AudioStudioMediaTicketPurpose,
  options?: AudioStudioRequestOptions
  artifact_id: string
}

export type AudioStudioResolvedMediaTicket = AudioStudioMediaTicketResponse & {
  ticket_url: string
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

type AudioStudioOutputCreateRequest = Pick<AudioStudioGenerationCreateRequest,
  "idempotency_key" | "target_resource_kind" | "target_resource_id" | "target_revision_id" | "options"> & { settings?: Record<string, unknown> }
export type AudioStudioRenderCreateRequest = AudioStudioOutputCreateRequest & { render_type: "preview_mix" | "final_mix" }
export type AudioStudioExportCreateRequest = AudioStudioOutputCreateRequest & {
  export_type: "single_audio" | "zip_package" | "narration_package" | "package"
  source_render_id?: string
}

export type AudioStudioJobResponse = {
  job_id: string
  status: string
  project_id?: string
}

export type AudiobookMigrationPreviewRequest = {
  legacy_project_id?: string
  project_payload?: LegacyAudiobookProjectMigrationPayload | Record<string, unknown>
  options?: Record<string, unknown>
}

export type AudiobookMigrationCommitRequest = {
  legacy_project_id?: string
  preview_id?: string
  project_payload?: LegacyAudiobookProjectMigrationPayload | Record<string, unknown>
  options?: Record<string, unknown>
  idempotency_key: string
}

export type AudiobookMigrationCounts = {
  projects?: number
  chapters?: number
  audio_assets?: number
  audioAssets?: number
  [key: string]: unknown
}

export type AudiobookMigrationResponse = {
  preview_id?: string
  fingerprint?: string
  workflow?: AudioStudioWorkflow
  project_count?: number
  section_count?: number
  audio_reference_count?: number
  needs_regeneration_count?: number
  warnings?: string[]
  project?: AudioStudioProject
  imported_section_count?: number
  replayed?: boolean
  counts?: AudiobookMigrationCounts
}

const JSON_HEADERS = { "Content-Type": "application/json" }
const API_BASE = "/api/v1/audio-studio"

const apiPath = (path: string) => path as AllowedPath
const projectPath = (projectId: string) =>
  apiPath(`${API_BASE}/projects/${encodeURIComponent(projectId)}`)
const resourcePath = (projectId: string, resource: string, resourceId: string) =>
  apiPath(
    `${API_BASE}/projects/${encodeURIComponent(projectId)}/${resource}/${encodeURIComponent(resourceId)}`
  )

export const listAudioStudioWorkflows = async (options?: AudioStudioRequestOptions): Promise<
  AudioStudioWorkflowSummary[]
> => {
  const response = await request<AudioStudioWorkflowListResponse>({
    path: apiPath(`${API_BASE}/workflows`),
    method: "GET"
  }, options)
  return response.workflows
}

export const listAudioStudioProjects = async (
  params: ListAudioStudioProjectsParams = {},
  options?: AudioStudioRequestOptions
): Promise<AudioStudioProject[]> => {
  const query = buildQuery({
    workflow: params.workflow,
    include_archived: params.includeArchived
  })
  const response = await request<AudioStudioProjectListResponse>({
    path: apiPath(`${API_BASE}/projects${query}`),
    method: "GET"
  }, options)
  return response.projects
}

export const listAudioStudioArtifacts = async (
  projectId: string,
  options?: AudioStudioRequestOptions
): Promise<AudioStudioArtifact[]> => {
  const artifacts: AudioStudioArtifact[] = []
  let offset = 0
  while (true) {
    const response = await request<AudioStudioArtifactListResponse>({
      path: apiPath(`${projectPath(projectId)}/artifacts${buildQuery({ limit: 100, offset })}`), method: "GET"
    }, options)
    artifacts.push(...response.artifacts)
    if (response.artifacts.length < 100) return artifacts
    offset += response.artifacts.length
  }
}

export const getAudioStudioArtifactMediaPath = (
  projectId: string,
  artifactId: string,
  options: { download?: boolean } = {}
): string => {
  const path = `${projectPath(projectId)}/artifacts/${encodeURIComponent(artifactId)}/media`
  return options.download ? `${path}?download=true` : path
}

const isAbsoluteHttpUrl = (value: string | null | undefined): value is string =>
  typeof value === "string" && /^https?:\/\//i.test(value)

const resolveAudioStudioTicketBrowserUrl = async (
  ticket: AudioStudioMediaTicketResponse, options?: AudioStudioRequestOptions
): Promise<string> => {
  if (isAbsoluteHttpUrl(ticket.ticket_url)) {
    return ticket.ticket_url
  }
  const config = options?.requestScope?.config ?? await tldwClient.getConfig().catch(() => null)
  return resolveBrowserRequestTransport({
    config,
    path: ticket.ticket_path
  }).url
}

export const mintAudioStudioArtifactMediaTicket = async (
  projectId: string,
  artifactId: string,
  purpose: AudioStudioMediaTicketPurpose,
  options?: AudioStudioRequestOptions
): Promise<AudioStudioResolvedMediaTicket> => {
  const response = await request<AudioStudioMediaTicketResponse>({
    path: apiPath(
      `${projectPath(projectId)}/artifacts/${encodeURIComponent(artifactId)}/tickets`
    ),
    method: "POST",
    headers: JSON_HEADERS,
    body: { purpose }
  }, options)
  return {
    ...response,
    ticket_url: await resolveAudioStudioTicketBrowserUrl(response, options)
  }
}

export const fetchAudioStudioArtifactBlob = async (
  projectId: string,
  artifact: Pick<AudioStudioArtifact, "artifact_id" | "mime_type">,
  options?: AudioStudioRequestOptions
): Promise<Blob> => {
  const response = await request<{
    ok: boolean
    status: number
    data?: ArrayBuffer
    error?: string
    headers?: Record<string, string>
  }>({
    path: apiPath(getAudioStudioArtifactMediaPath(projectId, artifact.artifact_id)),
    method: "GET",
    responseType: "arrayBuffer",
    returnResponse: true
  }, options)
  if (!response?.ok) {
    throw new Error(response?.error || `Audio Studio artifact fetch failed: ${response?.status ?? "unknown"}`)
  }
  const headers = new Headers(response.headers || {})
  return new Blob([response.data ?? new Uint8Array()], {
    type: headers.get("content-type") || artifact.mime_type || "application/octet-stream"
  })
}

export const createAudioStudioProject = async (
  body: CreateAudioStudioProjectRequest,
  options?: AudioStudioRequestOptions
): Promise<AudioStudioProject> =>
  request<AudioStudioProject>({
    path: apiPath(`${API_BASE}/projects`),
    method: "POST",
    headers: JSON_HEADERS,
    body
  }, options)

export const updateAudioStudioProject = async (
  projectId: string,
  body: UpdateAudioStudioProjectRequest,
  options?: AudioStudioRequestOptions
): Promise<AudioStudioProject> =>
  request<AudioStudioProject>({
    path: projectPath(projectId),
    method: "PATCH",
    headers: JSON_HEADERS,
    body
  }, options)

export const upsertAudioStudioSection = async (
  projectId: string,
  sectionId: string,
  body: AudioStudioSectionUpsertRequest,
  options?: AudioStudioRequestOptions
): Promise<AudioStudioSectionResponse> =>
  request<AudioStudioSectionResponse>({
    path: resourcePath(projectId, "sections", sectionId),
    method: "PUT",
    headers: JSON_HEADERS,
    body
  }, options)

export const upsertAudioStudioTrack = async (
  projectId: string,
  trackId: string,
  body: AudioStudioTrackUpsertRequest,
  options?: AudioStudioRequestOptions
): Promise<AudioStudioTrackResponse> =>
  request<AudioStudioTrackResponse>({
    path: resourcePath(projectId, "tracks", trackId),
    method: "PUT",
    headers: JSON_HEADERS,
    body
  }, options)

export const upsertAudioStudioClip = async (
  projectId: string,
  clipId: string,
  body: AudioStudioClipUpsertRequest,
  options?: AudioStudioRequestOptions
): Promise<AudioStudioClipResponse> =>
  request<AudioStudioClipResponse>({
    path: resourcePath(projectId, "clips", clipId),
    method: "PUT",
    headers: JSON_HEADERS,
    body
  }, options)

export const createAudioStudioGeneration = async (
  projectId: string,
  body: AudioStudioGenerationCreateRequest,
  options?: AudioStudioRequestOptions
): Promise<AudioStudioJobResponse> =>
  request<AudioStudioJobResponse>({
    path: apiPath(`${projectPath(projectId)}/generations`),
    method: "POST",
    headers: JSON_HEADERS,
    body
  }, options)

export const createAudioStudioRender = async (
  projectId: string,
  body: AudioStudioRenderCreateRequest,
  options?: AudioStudioRequestOptions
): Promise<AudioStudioJobResponse> =>
  request<AudioStudioJobResponse>({
    path: apiPath(`${projectPath(projectId)}/renders`),
    method: "POST",
    headers: JSON_HEADERS,
    body
  }, options)

export const createAudioStudioExport = async (
  projectId: string,
  body: AudioStudioExportCreateRequest,
  options?: AudioStudioRequestOptions
): Promise<AudioStudioJobResponse> =>
  request<AudioStudioJobResponse>({
    path: apiPath(`${projectPath(projectId)}/exports`),
    method: "POST",
    headers: JSON_HEADERS,
    body
  }, options)

export const previewAudiobookMigration = async (
  body: AudiobookMigrationPreviewRequest,
  options?: AudioStudioRequestOptions
): Promise<AudiobookMigrationResponse> =>
  request<AudiobookMigrationResponse>({
    path: apiPath(`${API_BASE}/migrations/audiobook/preview`),
    method: "POST",
    headers: JSON_HEADERS,
    body
  }, options)

export const commitAudiobookMigration = async (
  body: AudiobookMigrationCommitRequest,
  options?: AudioStudioRequestOptions
): Promise<AudiobookMigrationResponse> =>
  request<AudiobookMigrationResponse>({
    path: apiPath(`${API_BASE}/migrations/audiobook/commit`),
    method: "POST",
    headers: JSON_HEADERS,
    body
  }, options)

export const getAudioStudioProject = (projectId: string, options?: AudioStudioRequestOptions) =>
  request<AudioStudioProject>({ path: projectPath(projectId), method: "GET" }, options)

export type AudioStudioJob = AudioStudioJobResponse & {
  job_type: string
  target_resource_id: string
  target_revision_id: string
  retry_count: number
  max_retries: number
  result?: Record<string, unknown> | null
  error?: string | null
}
export const listAudioStudioJobs = async (projectId: string, options?: AudioStudioRequestOptions): Promise<AudioStudioJob[]> => {
  const jobs: AudioStudioJob[] = []
  let cursor: { before_id: number; created_before: string } | undefined
  do {
    const page = await request<{ jobs: AudioStudioJob[]; next_before_id: number | null; next_created_before: string | null }>({
      path: apiPath(`${projectPath(projectId)}/jobs${buildQuery({ limit: 100, ...cursor })}`), method: "GET"
    }, options)
    jobs.push(...page.jobs)
    cursor = page.next_before_id != null && page.next_created_before ? { before_id: page.next_before_id, created_before: page.next_created_before } : undefined
  } while (cursor)
  return jobs
}
export const cancelAudioStudioJob = (projectId: string, jobId: string, options?: AudioStudioRequestOptions) =>
  request<AudioStudioJob>({ path: apiPath(`${projectPath(projectId)}/jobs/${encodeURIComponent(jobId)}/cancel`), method: "POST" }, options)

export const getAudioStudioSpeechProviders = async (options?: AudioStudioRequestOptions) =>
  normalizeTtsProvidersResponse(await request<unknown>({ path: "/api/v1/audio/providers", method: "GET" }, options))
export const getAudioStudioVoices = async (provider: string, model: string, options?: AudioStudioRequestOptions) => {
  const custom = extractVoices(await request<unknown>({ path: "/api/v1/audio/voices", method: "GET" }, options))
    .filter((voice) => normalizeTtsProviderKey(voice.provider) === normalizeTtsProviderKey(provider))
    .flatMap((voice) => {
      const id = voice.voice_id ?? voice.id
      return typeof id === "string" && id.trim() ? [{ ...voice, voice_id: id.startsWith("custom:") ? id : `custom:${id}`, name: `Custom: ${voice.name ?? id}` }] : []
    })
  try {
    const builtin = extractVoices(await request<unknown>({ path: apiPath(`/api/v1/audio/voices/catalog${buildQuery({ provider, model })}`), method: "GET" }, options))
    return [...builtin, ...custom]
  } catch (error) {
    // Providers with only user samples have no built-in catalog entry.
    if (typeof error === "object" && error !== null && "status" in error && error.status === 404) {
      options?.assertCurrent?.()
      return custom
    }
    throw error
  }
}
export const getAudioStudioProviders = async (options?: AudioStudioRequestOptions) =>
  request<{ providers: { provider_id: string; supported_kinds: string[] }[] }>({ path: apiPath(`${API_BASE}/providers`), method: "GET" }, options)

export type AudioStudioAdmissionKind = "generation" | "render" | "export"
export type AudioStudioAdmissionBody = AudioStudioGenerationCreateRequest | AudioStudioRenderCreateRequest | AudioStudioExportCreateRequest
export type AudioStudioAdmissionIntent = { kind: AudioStudioAdmissionKind; body: AudioStudioAdmissionBody; job_id?: string }
