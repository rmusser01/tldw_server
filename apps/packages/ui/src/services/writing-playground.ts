import { bgRequest } from "@/services/background-proxy"
import { buildQuery, createResourceClient } from "@/services/resource-client"
import type { AllowedPath } from "@/services/tldw/openapi-guard"

export type ManuscriptSceneResponse = {
  id: string
  chapter_id: string
  project_id: string
  title: string
  sort_order: number
  content: Record<string, unknown>
  content_plain: string
  synopsis?: string | null
  word_count: number
  pov_character_id?: string | null
  status: string
  created_at: string
  last_modified: string
  deleted: boolean
  client_id: string
  version: number
}

export type ManuscriptCharacterResponse = {
  id: string
  project_id: string
  name: string
  role: string
  cast_group?: string | null
  full_name?: string | null
  age?: string | null
  gender?: string | null
  appearance?: string | null
  personality?: string | null
  backstory?: string | null
  motivation?: string | null
  arc_summary?: string | null
  notes?: string | null
  custom_fields: Record<string, unknown>
  sort_order: number
  created_at: string
  last_modified: string
  deleted: boolean
  client_id: string
  version: number
}

export type ManuscriptRelationshipResponse = {
  id: string
  project_id: string
  from_character_id: string
  to_character_id: string
  relationship_type: string
  description?: string | null
  bidirectional: boolean
  created_at: string
  last_modified: string
  deleted: boolean
  client_id: string
  version: number
}

export type ManuscriptWorldInfoResponse = {
  id: string
  project_id: string
  kind: string
  name: string
  description?: string | null
  parent_id?: string | null
  properties: Record<string, unknown>
  tags: string[]
  sort_order: number
  created_at: string
  last_modified: string
  deleted: boolean
  client_id: string
  version: number
}

export type ManuscriptPlotLineResponse = {
  id: string
  project_id: string
  title: string
  description?: string | null
  status: string
  color?: string | null
  sort_order: number
  created_at: string
  last_modified: string
  deleted: boolean
  client_id: string
  version: number
}

export type ManuscriptPlotHoleResponse = {
  id: string
  project_id: string
  title: string
  description?: string | null
  severity: string
  status: string
  resolution?: string | null
  scene_id?: string | null
  chapter_id?: string | null
  plot_line_id?: string | null
  detected_by: string
  created_at: string
  last_modified: string
  deleted: boolean
  client_id: string
  version: number
}

export type SceneCharacterLinkResponse = {
  scene_id: string
  character_id: string
  is_pov: boolean
  name: string
  role: string
}

export type ManuscriptCitationResponse = {
  id: string
  project_id: string
  scene_id: string
  source_type: string
  source_id?: string | null
  source_title?: string | null
  excerpt?: string | null
  query_used?: string | null
  anchor_offset?: number | null
  created_at: string
  last_modified: string
  deleted: boolean
  client_id: string
  version: number
}

export type ManuscriptResearchResult = {
  id?: string
  title?: string
  source_title?: string
  snippet?: string
  excerpt?: string
  source_type?: string
  [key: string]: unknown
}

export type ManuscriptResearchResponse = {
  query: string
  results: ManuscriptResearchResult[]
}

const sessionsClient = createResourceClient({
  basePath: "/api/v1/writing/sessions" as AllowedPath
})

const templatesClient = createResourceClient({
  basePath: "/api/v1/writing/templates" as AllowedPath,
  detailPath: (name) =>
    `/api/v1/writing/templates/${encodeURIComponent(String(name))}` as AllowedPath
})

const themesClient = createResourceClient({
  basePath: "/api/v1/writing/themes" as AllowedPath,
  detailPath: (name) =>
    `/api/v1/writing/themes/${encodeURIComponent(String(name))}` as AllowedPath
})

const wordcloudClient = createResourceClient({
  basePath: "/api/v1/writing/wordclouds" as AllowedPath,
  detailPath: (id) =>
    `/api/v1/writing/wordclouds/${encodeURIComponent(String(id))}` as AllowedPath
})

export type WritingVersionResponse = {
  version: number
}

export type WritingServerCapabilities = {
  sessions: boolean
  templates: boolean
  themes: boolean
  defaults_catalog?: boolean
  snapshots?: boolean
  tokenize: boolean
  detokenize?: boolean
  token_count: boolean
  wordclouds?: boolean
  token_probabilities?: {
    inline_reroll?: boolean
  }
  context?: {
    author_note_depth_mode?: "insertion" | "annotation" | string
    context_order?: boolean
    context_budget?: boolean
  }
}

export type WritingTokenizerSupport = {
  available: boolean
  tokenizer?: string | null
  kind?: string | null
  source?: string | null
  detokenize?: boolean
  error?: string | null
}

export type WritingExtraBodyCompat = {
  supported: boolean
  effective_reason?: string | null
  known_params: string[]
  param_groups: string[]
  notes?: string | null
  example?: Record<string, unknown>
  source?: string
}

export type WritingProviderCapabilities = {
  name: string
  models: string[]
  capabilities: Record<string, unknown>
  supported_fields: string[]
  features: Record<string, boolean>
  tokenizers?: Record<string, WritingTokenizerSupport>
  extra_body_compat?: WritingExtraBodyCompat
  model_extra_body_compat?: Record<string, WritingExtraBodyCompat>
}

export type WritingRequestedCapabilities = {
  provider: string
  model?: string | null
  supported_fields: string[]
  features: Record<string, boolean>
  tokenizer_available: boolean
  tokenizer?: string | null
  tokenizer_kind?: string | null
  tokenizer_source?: string | null
  detokenize_available?: boolean
  tokenization_error?: string | null
  extra_body_compat?: WritingExtraBodyCompat
}

export type WritingCapabilitiesResponse = {
  version: number
  server: WritingServerCapabilities
  default_provider?: string | null
  providers?: WritingProviderCapabilities[]
  requested?: WritingRequestedCapabilities
}

export type WritingSessionListItem = {
  id: string
  name: string
  last_modified: string
  version: number
}

export type WritingSessionListResponse = {
  sessions: WritingSessionListItem[]
  total: number
}

export type WritingSessionResponse = {
  id: string
  name: string
  payload: Record<string, unknown>
  schema_version: number
  version_parent_id?: string | null
  created_at: string
  last_modified: string
  deleted: boolean
  client_id: string
  version: number
}

export type WritingSessionCreate = {
  name: string
  payload: Record<string, unknown>
  schema_version?: number | null
  id?: string | null
  version_parent_id?: string | null
}

export type WritingSessionUpdate = {
  name?: string | null
  payload?: Record<string, unknown> | null
  schema_version?: number | null
  version_parent_id?: string | null
}

export type WritingTemplateResponse = {
  id: number
  name: string
  payload: Record<string, unknown>
  schema_version: number
  version_parent_id?: string | null
  is_default: boolean
  created_at: string
  last_modified: string
  deleted: boolean
  client_id: string
  version: number
}

export type WritingTemplateListResponse = {
  templates: WritingTemplateResponse[]
  total: number
}

export type WritingTemplateCreate = {
  name: string
  payload: Record<string, unknown>
  schema_version?: number | null
  version_parent_id?: string | null
  is_default?: boolean | null
}

export type WritingTemplateUpdate = {
  name?: string | null
  payload?: Record<string, unknown> | null
  schema_version?: number | null
  version_parent_id?: string | null
  is_default?: boolean | null
}

export type WritingThemeResponse = {
  id: number
  name: string
  class_name?: string | null
  css?: string | null
  schema_version: number
  version_parent_id?: string | null
  is_default: boolean
  order: number
  created_at: string
  last_modified: string
  deleted: boolean
  client_id: string
  version: number
}

export type WritingThemeListResponse = {
  themes: WritingThemeResponse[]
  total: number
}

export type WritingThemeCreate = {
  name: string
  class_name?: string | null
  css?: string | null
  schema_version?: number | null
  version_parent_id?: string | null
  is_default?: boolean | null
  order?: number | null
}

export type WritingThemeUpdate = {
  name?: string | null
  class_name?: string | null
  css?: string | null
  schema_version?: number | null
  version_parent_id?: string | null
  is_default?: boolean | null
  order?: number | null
}

export type WritingDefaultTemplate = {
  name: string
  payload: Record<string, unknown>
  schema_version: number
  is_default: boolean
}

export type WritingDefaultTheme = {
  name: string
  class_name?: string | null
  css?: string | null
  schema_version: number
  is_default: boolean
  order: number
}

export type WritingDefaultsResponse = {
  version: number
  templates: WritingDefaultTemplate[]
  themes: WritingDefaultTheme[]
}

export type ManuscriptCharacter = Pick<
  ManuscriptCharacterResponse,
  "id" | "project_id" | "name" | "role"
>

export type ManuscriptCharacterListResponse = {
  characters: ManuscriptCharacter[]
  total: number
}

export type ManuscriptRelationship = Pick<
  ManuscriptRelationshipResponse,
  "id" | "relationship_type" | "from_character_id" | "to_character_id"
> & {
  character_a_id?: string
  character_b_id?: string
}

export type ManuscriptRelationshipListResponse = {
  relationships: ManuscriptRelationship[]
  total: number
}

export type ManuscriptWorldInfoItem = Pick<
  ManuscriptWorldInfoResponse,
  "id" | "project_id" | "name" | "kind"
>

export type ManuscriptWorldInfoListResponse = {
  items: ManuscriptWorldInfoItem[]
  total: number
}

export type ManuscriptPlotLine = Pick<
  ManuscriptPlotLineResponse,
  "id" | "title" | "status"
>

export type ManuscriptPlotLineListResponse = {
  plot_lines: ManuscriptPlotLine[]
  total: number
}

export type ManuscriptPlotHole = Pick<
  ManuscriptPlotHoleResponse,
  "id" | "title" | "severity"
>

export type ManuscriptPlotHoleListResponse = {
  plot_holes: ManuscriptPlotHole[]
  total: number
}

export type WritingSnapshotCounts = {
  sessions: number
  templates: number
  themes: number
}

export type WritingSnapshotSessionItem = {
  id?: string | null
  name: string
  payload: Record<string, unknown>
  schema_version: number
  version_parent_id?: string | null
}

export type WritingSnapshotTemplateItem = {
  name: string
  payload: Record<string, unknown>
  schema_version: number
  version_parent_id?: string | null
  is_default?: boolean
}

export type WritingSnapshotThemeItem = {
  name: string
  class_name?: string | null
  css?: string | null
  schema_version: number
  version_parent_id?: string | null
  is_default?: boolean
  order?: number
}

export type WritingSnapshotPayload = {
  sessions: WritingSnapshotSessionItem[]
  templates: WritingSnapshotTemplateItem[]
  themes: WritingSnapshotThemeItem[]
}

export type WritingSnapshotExportResponse = {
  version: number
  counts: WritingSnapshotCounts
  sessions: WritingSnapshotSessionItem[]
  templates: WritingSnapshotTemplateItem[]
  themes: WritingSnapshotThemeItem[]
}

export type WritingSnapshotImportRequest = {
  mode?: "merge" | "replace"
  snapshot: WritingSnapshotPayload
}

export type WritingSnapshotImportResponse = {
  mode: "merge" | "replace"
  imported: WritingSnapshotCounts
}

export type WritingTokenizeOptions = {
  include_strings?: boolean
}

export type WritingTokenizeRequest = {
  provider: string
  model: string
  text: string
  options?: WritingTokenizeOptions
}

export type WritingTokenizeMeta = {
  provider: string
  model: string
  tokenizer: string
  tokenizer_kind?: string | null
  tokenizer_source?: string | null
  detokenize_available?: boolean
  input_chars: number
  token_count: number
  warnings: string[]
}

export type WritingTokenizeResponse = {
  ids: number[]
  strings?: string[]
  meta: WritingTokenizeMeta
}

export type WritingTokenCountRequest = {
  provider: string
  model: string
  text: string
}

export type WritingTokenCountResponse = {
  count: number
  meta: WritingTokenizeMeta
}

export type WritingDetokenizeRequest = {
  provider: string
  model: string
  ids: number[]
}

export type WritingDetokenizeResponse = {
  text: string
  strings?: string[]
  meta: WritingTokenizeMeta
}

export type WritingWordcloudOptions = {
  max_words?: number
  min_word_length?: number
  keep_numbers?: boolean
  stopwords?: string[] | null
}

export type WritingWordcloudRequest = {
  text: string
  options?: WritingWordcloudOptions
}

export type WritingWordcloudWord = {
  text: string
  weight: number
}

export type WritingWordcloudMeta = {
  input_chars: number
  total_tokens: number
  top_n: number
}

export type WritingWordcloudResult = {
  words: WritingWordcloudWord[]
  meta: WritingWordcloudMeta
}

export type WritingWordcloudResponse = {
  id: string
  status: "queued" | "running" | "ready" | "failed" | string
  cached?: boolean
  result?: WritingWordcloudResult | null
  error?: string | null
}

type WritingCapabilitiesQuery = {
  provider?: string
  model?: string
  includeProviders?: boolean
  includeDeprecated?: boolean
}

const buildExpectedVersionHeaders = (expectedVersion: number) => ({
  "expected-version": String(expectedVersion)
})

export async function getWritingVersion(): Promise<WritingVersionResponse> {
  return await bgRequest<WritingVersionResponse>({
    path: "/api/v1/writing/version",
    method: "GET"
  })
}

export async function getWritingCapabilities(
  options: WritingCapabilitiesQuery = {}
): Promise<WritingCapabilitiesResponse> {
  const query = buildQuery({
    provider: options.provider,
    model: options.model,
    include_providers: options.includeProviders ?? false,
    include_deprecated: options.includeDeprecated ?? false
  })
  const path = `/api/v1/writing/capabilities${query}` as AllowedPath
  return await bgRequest<WritingCapabilitiesResponse>({
    path,
    method: "GET"
  })
}

export async function getWritingDefaults(): Promise<WritingDefaultsResponse> {
  return await bgRequest<WritingDefaultsResponse>({
    path: "/api/v1/writing/defaults",
    method: "GET"
  })
}

export async function listWritingSessions(params?: {
  limit?: number
  offset?: number
}): Promise<WritingSessionListResponse> {
  return await sessionsClient.list<WritingSessionListResponse>({
    limit: params?.limit,
    offset: params?.offset
  })
}

export async function createWritingSession(
  input: WritingSessionCreate
): Promise<WritingSessionResponse> {
  return await sessionsClient.create<WritingSessionResponse>(input)
}

export async function getWritingSession(id: string): Promise<WritingSessionResponse> {
  return await sessionsClient.get<WritingSessionResponse>(id)
}

export async function updateWritingSession(
  id: string,
  input: WritingSessionUpdate,
  expectedVersion: number
): Promise<WritingSessionResponse> {
  return await sessionsClient.update<WritingSessionResponse>(id, input, {
    headers: buildExpectedVersionHeaders(expectedVersion)
  })
}

export async function deleteWritingSession(
  id: string,
  expectedVersion: number
): Promise<void> {
  await sessionsClient.remove<void>(id, undefined, {
    headers: buildExpectedVersionHeaders(expectedVersion)
  })
}

export async function cloneWritingSession(
  id: string,
  name?: string
): Promise<WritingSessionResponse> {
  return await bgRequest<WritingSessionResponse>({
    path: `/api/v1/writing/sessions/${encodeURIComponent(id)}/clone` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: name ? { name } : {}
  })
}

export async function listWritingTemplates(params?: {
  limit?: number
  offset?: number
}): Promise<WritingTemplateListResponse> {
  return await templatesClient.list<WritingTemplateListResponse>({
    limit: params?.limit,
    offset: params?.offset
  })
}

export async function createWritingTemplate(
  input: WritingTemplateCreate
): Promise<WritingTemplateResponse> {
  return await templatesClient.create<WritingTemplateResponse>(input)
}

export async function getWritingTemplate(
  name: string
): Promise<WritingTemplateResponse> {
  return await templatesClient.get<WritingTemplateResponse>(name)
}

export async function updateWritingTemplate(
  name: string,
  input: WritingTemplateUpdate,
  expectedVersion: number
): Promise<WritingTemplateResponse> {
  return await templatesClient.update<WritingTemplateResponse>(name, input, {
    headers: buildExpectedVersionHeaders(expectedVersion)
  })
}

export async function deleteWritingTemplate(
  name: string,
  expectedVersion: number
): Promise<void> {
  await templatesClient.remove<void>(name, undefined, {
    headers: buildExpectedVersionHeaders(expectedVersion)
  })
}

export async function listWritingThemes(params?: {
  limit?: number
  offset?: number
}): Promise<WritingThemeListResponse> {
  return await themesClient.list<WritingThemeListResponse>({
    limit: params?.limit,
    offset: params?.offset
  })
}

export async function createWritingTheme(
  input: WritingThemeCreate
): Promise<WritingThemeResponse> {
  return await themesClient.create<WritingThemeResponse>(input)
}

export async function getWritingTheme(name: string): Promise<WritingThemeResponse> {
  return await themesClient.get<WritingThemeResponse>(name)
}

export async function updateWritingTheme(
  name: string,
  input: WritingThemeUpdate,
  expectedVersion: number
): Promise<WritingThemeResponse> {
  return await themesClient.update<WritingThemeResponse>(name, input, {
    headers: buildExpectedVersionHeaders(expectedVersion)
  })
}

export async function deleteWritingTheme(
  name: string,
  expectedVersion: number
): Promise<void> {
  await themesClient.remove<void>(name, undefined, {
    headers: buildExpectedVersionHeaders(expectedVersion)
  })
}

export async function exportWritingSnapshot(): Promise<WritingSnapshotExportResponse> {
  return await bgRequest<WritingSnapshotExportResponse>({
    path: "/api/v1/writing/snapshot/export",
    method: "GET"
  })
}

export async function importWritingSnapshot(
  input: WritingSnapshotImportRequest
): Promise<WritingSnapshotImportResponse> {
  return await bgRequest<WritingSnapshotImportResponse>({
    path: "/api/v1/writing/snapshot/import",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: input
  })
}

export async function tokenizeWritingText(
  input: WritingTokenizeRequest
): Promise<WritingTokenizeResponse> {
  return await bgRequest<WritingTokenizeResponse>({
    path: "/api/v1/writing/tokenize",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: input
  })
}

export async function countWritingTokens(
  input: WritingTokenCountRequest
): Promise<WritingTokenCountResponse> {
  return await bgRequest<WritingTokenCountResponse>({
    path: "/api/v1/writing/token-count",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: input
  })
}

export async function detokenizeWritingTokens(
  input: WritingDetokenizeRequest
): Promise<WritingDetokenizeResponse> {
  return await bgRequest<WritingDetokenizeResponse>({
    path: "/api/v1/writing/detokenize",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: input
  })
}

export async function createWritingWordcloud(
  input: WritingWordcloudRequest
): Promise<WritingWordcloudResponse> {
  return await wordcloudClient.create<WritingWordcloudResponse>(input)
}

export async function getWritingWordcloud(
  id: string
): Promise<WritingWordcloudResponse> {
  return await wordcloudClient.get<WritingWordcloudResponse>(id)
}

// ── Manuscript API ─────────────────────────────────────

const manuscriptProjectsClient = createResourceClient({
  basePath: "/api/v1/writing/manuscripts/projects" as AllowedPath,
  detailPath: (id) =>
    `/api/v1/writing/manuscripts/projects/${encodeURIComponent(String(id))}` as AllowedPath,
})

export async function listManuscriptProjects(params?: {
  status?: string
  limit?: number
  offset?: number
}) {
  return manuscriptProjectsClient.list(params)
}

export async function getManuscriptProject(id: string) {
  return manuscriptProjectsClient.get(id)
}

export async function createManuscriptProject(data: Record<string, unknown>) {
  return manuscriptProjectsClient.create(data)
}

export async function updateManuscriptProject(
  id: string,
  data: Record<string, unknown>,
  version: number,
) {
  return manuscriptProjectsClient.update(id, data, {
    headers: buildExpectedVersionHeaders(version),
  })
}

export async function deleteManuscriptProject(id: string, version: number) {
  return manuscriptProjectsClient.remove(id, undefined, {
    headers: buildExpectedVersionHeaders(version),
  })
}

export async function getManuscriptStructure(projectId: string) {
  return await bgRequest({
    path: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/structure` as AllowedPath,
    method: "GET",
  })
}

export async function searchManuscriptScenes(
  projectId: string,
  query: string,
  limit = 20,
) {
  const qs = buildQuery({ q: query, limit })
  return await bgRequest({
    path: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/search${qs}` as AllowedPath,
    method: "GET",
  })
}

export async function reorderManuscriptItems(
  projectId: string,
  entityType: "parts" | "chapters" | "scenes",
  items: { id: string; sort_order: number; version: number; new_parent_id?: string | null }[],
) {
  return await bgRequest({
    path: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/reorder` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: { entity_type: entityType, items },
  })
}

// Scene-level CRUD (needs chapter context for create)
export async function createManuscriptScene(
  chapterId: string,
  data: Record<string, unknown>,
) {
  return await bgRequest({
    path: `/api/v1/writing/manuscripts/chapters/${encodeURIComponent(chapterId)}/scenes` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: data,
  })
}

export async function getManuscriptScene(sceneId: string): Promise<ManuscriptSceneResponse> {
  return await bgRequest<ManuscriptSceneResponse>({
    path: `/api/v1/writing/manuscripts/scenes/${encodeURIComponent(sceneId)}` as AllowedPath,
    method: "GET",
  })
}

export async function updateManuscriptScene(
  sceneId: string,
  data: Record<string, unknown>,
  version: number,
) {
  return await bgRequest({
    path: `/api/v1/writing/manuscripts/scenes/${encodeURIComponent(sceneId)}` as AllowedPath,
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      "expected-version": String(version),
    },
    body: data,
  })
}

// ── Characters ──────────────────────────────────────────

export async function listManuscriptCharacters(
  projectId: string,
  params?: { role?: string; cast_group?: string },
): Promise<ManuscriptCharacterListResponse> {
  const query = new URLSearchParams()
  if (params?.role) query.set("role", params.role)
  if (params?.cast_group) query.set("cast_group", params.cast_group)
  const qs = query.toString()
  const path = `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/characters${qs ? `?${qs}` : ""}`
  return await bgRequest<ManuscriptCharacterListResponse>({
    path: path as AllowedPath,
    method: "GET",
  })
}

export async function createManuscriptCharacter(
  projectId: string,
  data: Record<string, unknown>,
): Promise<ManuscriptCharacterResponse> {
  return bgRequest<ManuscriptCharacterResponse>({
    path: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/characters` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: data,
  })
}

export async function deleteManuscriptCharacter(characterId: string, version: number) {
  return bgRequest({
    path: `/api/v1/writing/manuscripts/characters/${encodeURIComponent(characterId)}` as AllowedPath,
    method: "DELETE",
    headers: buildExpectedVersionHeaders(version),
  })
}

// ── Relationships ───────────────────────────────────────

export async function listManuscriptRelationships(
  projectId: string,
): Promise<ManuscriptRelationshipListResponse> {
  return await bgRequest<ManuscriptRelationshipListResponse>({
    path: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/characters/relationships` as AllowedPath,
    method: "GET",
  })
}

export async function createManuscriptRelationship(
  projectId: string,
  data: Record<string, unknown>,
): Promise<ManuscriptRelationshipResponse> {
  return bgRequest<ManuscriptRelationshipResponse>({
    path: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/characters/relationships` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: data,
  })
}

// ── World Info ──────────────────────────────────────────

export async function listManuscriptWorldInfo(
  projectId: string,
  params?: { kind?: string },
): Promise<ManuscriptWorldInfoListResponse> {
  const query = new URLSearchParams()
  if (params?.kind) query.set("kind", params.kind)
  const qs = query.toString()
  const path = `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/world-info${qs ? `?${qs}` : ""}`
  return await bgRequest<ManuscriptWorldInfoListResponse>({
    path: path as AllowedPath,
    method: "GET",
  })
}

export async function createManuscriptWorldInfo(
  projectId: string,
  data: Record<string, unknown>,
): Promise<ManuscriptWorldInfoResponse> {
  return bgRequest<ManuscriptWorldInfoResponse>({
    path: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/world-info` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: data,
  })
}

// ── Plot Lines ──────────────────────────────────────────

export async function listManuscriptPlotLines(
  projectId: string,
): Promise<ManuscriptPlotLineListResponse> {
  return await bgRequest<ManuscriptPlotLineListResponse>({
    path: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/plot-lines` as AllowedPath,
    method: "GET",
  })
}

export async function createManuscriptPlotLine(
  projectId: string,
  data: Record<string, unknown>,
): Promise<ManuscriptPlotLineResponse> {
  return bgRequest<ManuscriptPlotLineResponse>({
    path: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/plot-lines` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: data,
  })
}

// ── Plot Holes ──────────────────────────────────────────

export async function listManuscriptPlotHoles(
  projectId: string,
): Promise<ManuscriptPlotHoleListResponse> {
  return await bgRequest<ManuscriptPlotHoleListResponse>({
    path: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/plot-holes` as AllowedPath,
    method: "GET",
  })
}

// ── Scene Linking ───────────────────────────────────────

export async function linkSceneCharacter(sceneId: string, characterId: string, isPov = false) {
  return bgRequest<SceneCharacterLinkResponse>({
    path: `/api/v1/writing/manuscripts/scenes/${encodeURIComponent(sceneId)}/characters` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: { character_id: characterId, is_pov: isPov },
  })
}

export async function unlinkSceneCharacter(sceneId: string, characterId: string) {
  return bgRequest({
    path: `/api/v1/writing/manuscripts/scenes/${encodeURIComponent(sceneId)}/characters/${encodeURIComponent(characterId)}` as AllowedPath,
    method: "DELETE",
  })
}

export async function listSceneCharacters(
  sceneId: string,
): Promise<SceneCharacterLinkResponse[]> {
  return await bgRequest<SceneCharacterLinkResponse[]>({
    path: `/api/v1/writing/manuscripts/scenes/${encodeURIComponent(sceneId)}/characters` as AllowedPath,
    method: "GET",
  })
}

// ── Citations ───────────────────────────────────────────

export async function listManuscriptCitations(
  sceneId: string,
): Promise<ManuscriptCitationResponse[]> {
  return await bgRequest<ManuscriptCitationResponse[]>({
    path: `/api/v1/writing/manuscripts/scenes/${encodeURIComponent(sceneId)}/citations` as AllowedPath,
    method: "GET",
  })
}

export type ManuscriptCitationCreatePayload = {
  source_type: string
  source_id?: string | null
  source_title?: string | null
  excerpt?: string | null
  query_used?: string | null
  anchor_offset?: number | null
  id?: string | null
}

export async function createManuscriptCitation(
  sceneId: string,
  data: ManuscriptCitationCreatePayload,
): Promise<ManuscriptCitationResponse> {
  return bgRequest<ManuscriptCitationResponse>({
    path: `/api/v1/writing/manuscripts/scenes/${encodeURIComponent(sceneId)}/citations` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: data,
  })
}

// ── Analysis ────────────────────────────────────────────

export async function analyzeScene(
  sceneId: string,
  data?: { analysis_types?: string[]; provider?: string; model?: string },
) {
  return bgRequest({
    path: `/api/v1/writing/manuscripts/scenes/${encodeURIComponent(sceneId)}/analyze` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: data || { analysis_types: ["pacing"] },
  })
}

export async function analyzeChapter(
  chapterId: string,
  data?: { analysis_types?: string[]; provider?: string; model?: string },
) {
  return bgRequest({
    path: `/api/v1/writing/manuscripts/chapters/${encodeURIComponent(chapterId)}/analyze` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: data || { analysis_types: ["pacing"] },
  })
}

export async function analyzeProjectPlotHoles(
  projectId: string,
  data?: { provider?: string; model?: string },
) {
  return bgRequest({
    path: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/analyze/plot-holes` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: { analysis_types: ["plot_holes"], ...data },
  })
}

export async function analyzeProjectConsistency(
  projectId: string,
  data?: { provider?: string; model?: string },
) {
  return bgRequest({
    path: `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/analyze/consistency` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: { analysis_types: ["consistency"], ...data },
  })
}

export async function listManuscriptAnalyses(
  projectId: string,
  params?: { scope_type?: string; analysis_type?: string; include_stale?: boolean },
) {
  const query = new URLSearchParams()
  if (params?.scope_type) query.set("scope_type", params.scope_type)
  if (params?.analysis_type) query.set("analysis_type", params.analysis_type)
  if (params?.include_stale) query.set("include_stale", "true")
  const qs = query.toString()
  const path = `/api/v1/writing/manuscripts/projects/${encodeURIComponent(projectId)}/analyses${qs ? `?${qs}` : ""}`
  return await bgRequest({
    path: path as AllowedPath,
    method: "GET",
  })
}

// ── Research ────────────────────────────────────────────

export async function searchManuscriptResearch(
  sceneId: string,
  query: string,
  topK = 5,
): Promise<ManuscriptResearchResponse> {
  return await bgRequest<ManuscriptResearchResponse>({
    path: `/api/v1/writing/manuscripts/scenes/${encodeURIComponent(sceneId)}/research` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: { query, top_k: topK },
  })
}
