import type { ChatScope } from "@/types/chat-scope"
import { toChatScopeParams } from "@/types/chat-scope"
import { Storage } from "@plasmohq/storage"
import { createSafeStorage, safeStorageSerde } from "@/utils/safe-storage"
import { bgRequest, bgStream, bgUpload } from "@/services/background-proxy"
import { isPlaceholderApiKey } from "@/utils/api-key"
import { normalizeChatRole } from "@/utils/normalize-chat-role"
import { createJsonResponseLike } from "@/services/tldw/json-response-like"
import type { AllowedPath, PathOrUrl } from "@/services/tldw/openapi-guard"
import { tldwRequest } from "@/services/tldw/request-core"
import { appendPathQuery } from "@/services/tldw/path-utils"
import { inferUploadMediaTypeFromUrl } from "@/services/tldw/media-routing"
import { captureChatRequestDebugSnapshot } from "@/services/tldw/chat-request-debug"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import { toTrimmedStringArray } from "@/services/tldw/client-utils"
import { getTldwTTSModel, getTldwTTSVoice } from "@/services/tts"
import {
  DEFAULT_CHARACTER_PROFILE_PREFERENCE_KEY,
  normalizeDefaultCharacterPreferenceId
} from "@/utils/default-character-preference"
import type { PersonaBuddySummary } from "@/types/persona-buddy"
import {
  resolveWebUiQuickstartServerUrl,
  type BrowserSurface
} from "@/services/tldw/browser-networking"
import {
  buildContentPayload,
  mapApiDetailToUi,
  mapApiListToUi,
  mapUiSourceToApi,
  type ApiDataTableGenerateResponse,
  type ApiDataTableJobStatus
} from "@/services/tldw/data-tables"
import type { DataTableColumn } from "@/types/data-tables"
import type {
  CreateReadingSavedSearchRequest,
  CreateReadingDigestScheduleRequest,
  ImportSource,
  ReadingNoteLink,
  ReadingListResponse,
  ReadingSavedSearch,
  ReadingSavedSearchListResponse,
  ReadingDigestSchedule,
  ReadingImportJobResponse,
  ReadingImportJobStatus,
  ReadingImportJobsListResponse,
  UpdateReadingSavedSearchRequest,
  UpdateReadingDigestScheduleRequest
} from "@/types/collections"
import {
  normalizeIngestionSource,
  normalizeIngestionSourceItem,
  normalizeIngestionSourceItemsListResponse,
  normalizeIngestionSourceListResponse,
  normalizeIngestionSourceSyncSummary,
  normalizeIngestionSourceSyncTrigger,
  normalizeIngestionSourceType,
  normalizeReadingDigestSchedule,
  toFiniteNumber,
  toOptionalString,
  toRecord
} from "./collections-normalizers"
import {
  buildPresentationVisualStyleSnapshot,
  clonePresentationVisualStyleSnapshot
} from "./presentation-style"
import {
  normalizePersonaExemplar,
  normalizePersonaProfile
} from "./persona-normalizers"

export {
  normalizeIngestionSource,
  normalizeIngestionSourceItem,
  normalizeIngestionSourceItemsListResponse,
  normalizeIngestionSourceListResponse,
  normalizeIngestionSourceSyncSummary,
  normalizeIngestionSourceSyncTrigger,
  normalizeIngestionSourceType,
  normalizeReadingDigestSchedule,
  toFiniteNumber,
  toOptionalString,
  toRecord
} from "./collections-normalizers"
export {
  buildPresentationVisualStyleSnapshot,
  clonePresentationVisualStyleSnapshot
} from "./presentation-style"
export {
  normalizePersonaExemplar,
  normalizePersonaProfile
} from "./persona-normalizers"

const DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
const CHARACTER_CACHE_TTL_MS = 5 * 60 * 1000
const CHAT_MESSAGES_CACHE_TTL_MS = 60 * 1000
const RAG_QUERY_MAX_LENGTH = 20000
const CHAT_COMPLETION_ERROR_MESSAGE = "Chat completion failed."
const CHAT_COMPLETION_ERRORS_MESSAGE =
  "One or more internal errors were suppressed."

const toRecordOrNull = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null

const isSavedDegradedCharacterPersistError = (error: unknown): boolean => {
  const candidate = error as
    | {
        detail?: unknown
        details?: { detail?: unknown; code?: unknown; saved?: unknown }
      }
    | null
  const detail =
    toRecordOrNull(candidate?.detail)
    ?? toRecordOrNull(candidate?.details?.detail)
    ?? toRecordOrNull(candidate?.details)
  return detail?.code === "persist_validation_degraded" && detail?.saved === true
}

const isSuspiciousChatCompletionString = (value: string): boolean =>
  /traceback|stack(?:\s*trace)?|exception|error|\/Users\/|[A-Za-z]:\\|\.py:\d+/i.test(
    value
  )

const normalizeChatCompletionResponseBody = (
  value: unknown
): Record<string, unknown> | unknown[] => {
  if (typeof value === "string") {
    if (isSuspiciousChatCompletionString(value)) {
      return {
        error: CHAT_COMPLETION_ERROR_MESSAGE,
        errors: [CHAT_COMPLETION_ERRORS_MESSAGE]
      }
    }
    return { content: value }
  }
  const sanitized = sanitizeChatCompletionPayload(value)
  if (Array.isArray(sanitized)) {
    return sanitized
  }
  if (sanitized && typeof sanitized === "object") {
    return sanitized as Record<string, unknown>
  }
  return { content: sanitized ?? "" }
}

const sanitizeChatCompletionPayload = (value: unknown): unknown => {
  if (typeof value === "string") {
    return isSuspiciousChatCompletionString(value)
      ? CHAT_COMPLETION_ERROR_MESSAGE
      : value
  }
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeChatCompletionPayload(item))
  }
  if (value && typeof value === "object") {
    const sanitized: Record<string, unknown> = {}
    for (const [key, item] of Object.entries(value)) {
      if (
        key === "details" ||
        key === "exception" ||
        key === "traceback" ||
        key === "stack" ||
        key === "stack_trace"
      ) {
        continue
      }
      if (key === "error" && item) {
        sanitized[key] = CHAT_COMPLETION_ERROR_MESSAGE
        continue
      }
      if (key === "errors" && item) {
        sanitized[key] = [CHAT_COMPLETION_ERRORS_MESSAGE]
        continue
      }
      sanitized[key] = sanitizeChatCompletionPayload(item)
    }
    return sanitized
  }
  return value
}

const toOptionalNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return null
}

export interface TldwConfig {
  serverUrl: string
  apiKey?: string
  accessToken?: string
  refreshToken?: string
  orgId?: number
  authMode: 'single-user' | 'multi-user'
}

export interface CurrentUserStorageQuotaResponse {
  user_id: number
  storage_used_mb: number
  storage_quota_mb: number
  available_mb: number
  usage_percentage: number
}

export interface OpenAIOAuthAuthorizeRequest {
  credential_fields?: Record<string, unknown>
  return_path?: string
}

export interface OpenAIOAuthAuthorizeResponse {
  provider: "openai"
  auth_url: string
  auth_session_id: string
  expires_at: string
}

export interface OpenAIOAuthStatusResponse {
  provider: "openai"
  connected: boolean
  auth_source: "api_key" | "oauth" | "none"
  updated_at?: string | null
  last_used_at?: string | null
  expires_at?: string | null
  scope?: string | null
}

export interface OpenAIOAuthRefreshResponse {
  provider: "openai"
  status: string
  updated_at?: string | null
  expires_at?: string | null
}

export interface OpenAICredentialSourceSwitchResponse {
  provider: "openai"
  auth_source: "api_key" | "oauth"
  updated_at?: string | null
}

const getCurrentBrowserSurface = (): BrowserSurface => {
  if (typeof window === "undefined") {
    return "extension"
  }

  try {
    const protocol = String(window.location?.protocol || "").trim().toLowerCase()
    if (protocol === "chrome-extension:" || protocol === "moz-extension:") {
      return "extension"
    }
    if (protocol === "http:" || protocol === "https:") {
      return "webui-page"
    }
  } catch {
    // Fall through to the browser-app default.
  }

  return "browser-app"
}

const getQuickstartWebUiServerUrl = (
): string | null => {
  try {
    return resolveWebUiQuickstartServerUrl({
      surface: getCurrentBrowserSurface(),
      deploymentMode: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
      pageOrigin:
        typeof window === "undefined" ? null : String(window.location?.origin || "").trim(),
      apiOrigin: process.env.NEXT_PUBLIC_API_URL
    })
  } catch {
    return null
  }
}

export type PresentationStudioSlide = {
  order: number
  layout: string
  title?: string | null
  content: string
  speaker_notes?: string | null
  metadata: Record<string, any>
}

export type PresentationVisualStyleSnapshot = {
  id: string
  scope: string
  name: string
  description?: string | null
  category?: string | null
  guide_number?: number | null
  tags?: string[]
  best_for?: string[]
  generation_rules?: Record<string, any>
  artifact_preferences?: string[]
  appearance_defaults?: Record<string, any>
  fallback_policy?: Record<string, any>
  version?: number | null
}

export type VisualStyleRecord = {
  id: string
  name: string
  scope: string
  description?: string | null
  category?: string | null
  guide_number?: number | null
  tags: string[]
  best_for: string[]
  generation_rules: Record<string, any>
  artifact_preferences: string[]
  appearance_defaults: Record<string, any>
  fallback_policy: Record<string, any>
  version?: number | null
  created_at?: string | null
  updated_at?: string | null
}

export type VisualStyleCreateInput = {
  name: string
  description?: string | null
  generation_rules?: Record<string, any>
  artifact_preferences?: string[]
  appearance_defaults?: Record<string, any>
  fallback_policy?: Record<string, any>
}

export type VisualStylePatchInput = {
  name?: string | null
  description?: string | null
  generation_rules?: Record<string, any> | null
  artifact_preferences?: string[] | null
  appearance_defaults?: Record<string, any> | null
  fallback_policy?: Record<string, any> | null
}

export type PresentationStudioRecord = {
  id: string
  title: string
  description?: string | null
  theme: string
  marp_theme?: string | null
  template_id?: string | null
  visual_style_id?: string | null
  visual_style_scope?: string | null
  visual_style_name?: string | null
  visual_style_version?: number | null
  visual_style_snapshot?: PresentationVisualStyleSnapshot | null
  settings?: Record<string, any> | null
  studio_data?: Record<string, any> | null
  slides: PresentationStudioSlide[]
  custom_css?: string | null
  source_type?: string | null
  source_ref?: unknown
  source_query?: string | null
  created_at: string
  last_modified: string
  deleted?: boolean
  client_id?: string
  version: number
}

export type PresentationRenderFormat = "mp4" | "webm"

export type PresentationRenderJob = {
  job_id: number
  status: string
  job_type: string
  presentation_id?: string | null
  presentation_version?: number | null
  format?: PresentationRenderFormat | null
  output_id?: number | null
  download_url?: string | null
  error?: string | null
}

export type PresentationRenderArtifact = {
  output_id: number
  format: PresentationRenderFormat
  title?: string | null
  download_url: string
  presentation_version?: number | null
  created_at?: string | null
}

export type PresentationRenderArtifactList = {
  presentation_id: string
  artifacts: PresentationRenderArtifact[]
}

const normalizeVisualStyleSnapshot = (
  value: unknown
): PresentationVisualStyleSnapshot | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null
  }
  const snapshot = value as Record<string, unknown>
  const id = String(snapshot.id ?? "").trim()
  const scope = String(snapshot.scope ?? "").trim()
  const name = String(snapshot.name ?? "").trim()
  if (!id || !scope || !name) {
    return null
  }
  return clonePresentationVisualStyleSnapshot({
    id,
    scope,
    name,
    description: toOptionalString(snapshot.description),
    category: toOptionalString(snapshot.category),
    guide_number: toOptionalNumber(snapshot.guide_number),
    tags: toTrimmedStringArray(snapshot.tags),
    best_for: toTrimmedStringArray(snapshot.best_for),
    generation_rules: toRecord(snapshot.generation_rules),
    artifact_preferences: toTrimmedStringArray(snapshot.artifact_preferences),
    appearance_defaults: toRecord(snapshot.appearance_defaults),
    fallback_policy: toRecord(snapshot.fallback_policy),
    version: toOptionalNumber(snapshot.version)
  })
}

const normalizeVisualStyleRecord = (style: unknown): VisualStyleRecord => {
  const record = style && typeof style === "object" && !Array.isArray(style)
    ? (style as Record<string, unknown>)
    : {}
  return {
    id: String(record.id ?? ""),
    name: String(record.name ?? ""),
    scope: String(record.scope ?? ""),
    description: toOptionalString(record.description),
    category: toOptionalString(record.category),
    guide_number: toOptionalNumber(record.guide_number),
    tags: toTrimmedStringArray(record.tags),
    best_for: toTrimmedStringArray(record.best_for),
    generation_rules: toRecord(record.generation_rules),
    artifact_preferences: toTrimmedStringArray(record.artifact_preferences),
    appearance_defaults: toRecord(record.appearance_defaults),
    fallback_policy: toRecord(record.fallback_policy),
    version: toOptionalNumber(record.version),
    created_at: toOptionalString(record.created_at),
    updated_at: toOptionalString(record.updated_at)
  }
}

const normalizePresentationStudioRecord = (presentation: unknown): PresentationStudioRecord => {
  const record =
    presentation && typeof presentation === "object" && !Array.isArray(presentation)
      ? (presentation as Record<string, unknown>)
      : {}
  const slides = Array.isArray(record.slides)
    ? (record.slides as PresentationStudioSlide[])
    : []
  return {
    id: String(record.id ?? ""),
    title: String(record.title ?? ""),
    description: toOptionalString(record.description),
    theme: String(record.theme ?? "black"),
    marp_theme: toOptionalString(record.marp_theme),
    template_id: toOptionalString(record.template_id),
    visual_style_id: toOptionalString(record.visual_style_id),
    visual_style_scope: toOptionalString(record.visual_style_scope),
    visual_style_name: toOptionalString(record.visual_style_name),
    visual_style_version: toOptionalNumber(record.visual_style_version),
    visual_style_snapshot: normalizeVisualStyleSnapshot(record.visual_style_snapshot),
    settings: Object.keys(toRecord(record.settings)).length > 0 ? toRecord(record.settings) : null,
    studio_data:
      Object.keys(toRecord(record.studio_data)).length > 0 ? toRecord(record.studio_data) : null,
    slides,
    custom_css: toOptionalString(record.custom_css),
    source_type: toOptionalString(record.source_type),
    source_ref: record.source_ref ?? null,
    source_query: toOptionalString(record.source_query),
    created_at: String(record.created_at ?? ""),
    last_modified: String(record.last_modified ?? ""),
    deleted: Boolean(record.deleted),
    client_id: toOptionalString(record.client_id) ?? undefined,
    version: toFiniteNumber(record.version, 0)
  }
}

export type UserProfileUpdateEntry = {
  key: string
  value: unknown | null
}

export type UserProfileUpdateResponse = {
  profile_version?: string
  applied: string[]
  skipped: Array<{ key: string; message: string }>
}

export interface TldwModel {
  id: string
  name: string
  provider: string
  description?: string
  capabilities?: string[] | Record<string, unknown>
  context_length?: number
  vision?: boolean
  function_calling?: boolean
  json_output?: boolean
  type?: string
  modalities?: {
    input?: string[]
    output?: string[]
  }
}

export type ChatCompletionContentPartText = {
  type: "text"
  text: string
}

export type ChatCompletionContentPartImage = {
  type: "image_url"
  image_url: {
    url: string
    detail?: "auto" | "low" | "high" | null
  }
}

export type ChatCompletionContentPart =
  | ChatCompletionContentPartText
  | ChatCompletionContentPartImage

export type ChatCompletionUserContent = string | ChatCompletionContentPart[]

export type ChatCompletionAssistantContent = string | null

export type ChatCompletionToolCall = {
  id: string
  type: "function"
  function: {
    name: string
    arguments?: string | null
    parameters?: Record<string, unknown> | null
    description?: string | null
  }
}

export type FunctionCall = {
  name: string
  arguments: string
}

export type ChatMessage =
  | {
      role: "system"
      content: string
      name?: string | null
    }
  | {
      role: "user"
      content: ChatCompletionUserContent
      name?: string | null
    }
  | {
      role: "assistant"
      content: ChatCompletionAssistantContent
      name?: string | null
      tool_calls?: ChatCompletionToolCall[] | null
      function_call?: FunctionCall | null
    }
  | {
      role: "tool"
      content: string
      tool_call_id: string
      name?: string | null
    }

export interface ChatResearchContextOutlineSection {
  title: string
}

export interface ChatResearchContextClaim {
  text: string
}

export interface ChatResearchContextVerificationSummary {
  unsupported_claim_count?: number
}

export interface ChatResearchContextSourceTrustSummary {
  high_trust_count?: number
}

export interface ChatResearchContext {
  run_id: string
  query: string
  question: string
  outline: ChatResearchContextOutlineSection[]
  key_claims: ChatResearchContextClaim[]
  unresolved_questions: string[]
  verification_summary?: ChatResearchContextVerificationSummary
  source_trust_summary?: ChatResearchContextSourceTrustSummary
  research_url: string
}

export interface ResearchRunFollowUpOutlineItem {
  title: string
  focus_area?: string | null
}

export interface ResearchRunFollowUpClaimItem {
  claim_id: string
  text: string
}

export interface ResearchRunFollowUpVerificationSummary {
  supported_claim_count: number
  unsupported_claim_count: number
}

export interface ResearchRunFollowUpSourceTrustSummary {
  high_trust_count: number
  low_trust_count: number
}

export interface ResearchRunFollowUpBackground {
  question: string
  outline: ResearchRunFollowUpOutlineItem[]
  key_claims: ResearchRunFollowUpClaimItem[]
  unresolved_questions: string[]
  verification_summary: ResearchRunFollowUpVerificationSummary
  source_trust_summary: ResearchRunFollowUpSourceTrustSummary
}

export interface ResearchRunFollowUp {
  question: string
  background?: ResearchRunFollowUpBackground | null
}

export interface ResearchChatHandoff {
  chat_id: string
  launch_message_id?: string | null
}

export interface ResearchRunCreateRequest {
  query: string
  source_policy?: string
  autonomy_mode?: string
  limits_json?: Record<string, unknown> | null
  provider_overrides?: Record<string, unknown> | null
  chat_handoff?: ResearchChatHandoff | null
  follow_up?: ResearchRunFollowUp | null
}

export interface ResearchRunResponse {
  id: string
  status: string
  phase: string
  control_state: string
  progress_percent?: number | null
  progress_message?: string | null
  active_job_id?: string | null
  latest_checkpoint_id?: string | null
  completed_at?: string | null
  chat_id?: string | null
}

export interface ChatCompletionRequest {
  messages: ChatMessage[]
  model: string
  routing?: {
    strategy?: "llm_router" | "rules_router"
    objective?: "highest_quality" | "lowest_cost" | "lowest_latency" | "balanced"
    mode?: "per_turn" | "sticky_session"
    cross_provider?: boolean
    failure_mode?: "fallback_then_error" | "error"
  }
  stream?: boolean
  temperature?: number
  logprobs?: boolean
  top_logprobs?: number
  max_tokens?: number
  top_p?: number
  frequency_penalty?: number
  presence_penalty?: number
  reasoning_effort?: "low" | "medium" | "high"
  tool_choice?: "auto" | "none" | "required"
  tools?: Record<string, unknown>[]
  save_to_db?: boolean
  conversation_id?: string
  history_message_limit?: number
  history_message_order?: string
  slash_command_injection_mode?: string
  api_provider?: string
  extra_headers?: Record<string, unknown>
  extra_body?: Record<string, unknown>
  thinking_budget_tokens?: number
  grammar_mode?: "none" | "library" | "inline"
  grammar_id?: string
  grammar_inline?: string
  grammar_override?: string
  response_format?: { type: "json_object" | "text" }
  research_context?: ChatResearchContext
}

export interface ServerChatSummary {
  id: string
  title: string
  created_at: string
  updated_at?: string | null
  last_active?: string | null
  message_count?: number | null
  source?: string | null
  state?: ConversationState | string | null
  topic_label?: string | null
  cluster_id?: string | null
  external_ref?: string | null
  bm25_norm?: number | null
  character_id?: string | number | null
  parent_conversation_id?: string | null
  root_id?: string | null
  forked_from_message_id?: string | null
  version?: number | null
  scope_type?: "global" | "workspace" | null
  workspace_id?: string | null
}

export interface ChatLinkedResearchRun {
  run_id: string
  query: string
  status: string
  phase: string
  control_state: string
  latest_checkpoint_id: string | null
  updated_at: string
}

export interface ChatLinkedResearchRunsResponse {
  runs: ChatLinkedResearchRun[]
}

export type ResearchBundleResponse = Record<string, unknown>

export interface PersonaProfileSummary {
  id: string
  name?: string | null
  character_card_id?: number | null
  origin_character_id?: number | null
  buddy_summary?: PersonaBuddySummary | null
  metadata?: Record<string, unknown> | null
}

export interface PersonaProfile extends PersonaProfileSummary {
  mode?: string | null
  system_prompt?: string | null
  use_persona_state_context_default?: boolean
}

export interface PersonaExemplar {
  id: string
  persona_id: string
  kind: string
  content: string
  tone?: string | null
  scenario_tags: string[]
  capability_tags: string[]
  priority: number
  enabled: boolean
  source_type?: string | null
  source_ref?: string | null
  notes?: string | null
  created_at?: string | null
  last_modified?: string | null
}

export type PersonaExemplarInput = {
  kind: string
  content: string
  tone?: string | null
  scenario_tags?: string[]
  capability_tags?: string[]
  priority?: number
  enabled?: boolean
  source_type?: string | null
  source_ref?: string | null
  notes?: string | null
}

export type PersonaExemplarListOptions = {
  includeDisabled?: boolean
  includeDeleted?: boolean
  includeDeletedPersonas?: boolean
}

export type PersonaExemplarImportInput = {
  transcript: string
  source_ref?: string | null
  notes?: string | null
  max_candidates?: number
}

export type PersonaExemplarReviewInput = {
  action: "approve" | "reject"
  notes?: string | null
}

export type ConversationSharePermission = "view"

export interface ConversationShareLinkSummary {
  id: string
  permission: ConversationSharePermission
  created_at: string
  expires_at: string
  revoked_at?: string | null
  share_path?: string | null
  token?: string | null
}

export interface ConversationShareLinkCreateResponse {
  share_id: string
  permission: ConversationSharePermission
  created_at: string
  expires_at: string
  token: string
  share_path: string
}

export interface ConversationShareLinksListResponse {
  conversation_id: string
  links: ConversationShareLinkSummary[]
}

export interface ConversationShareLinkResolveResponse {
  conversation_id: string
  title?: string | null
  source?: string | null
  permission: ConversationSharePermission
  shared_by_user_id: string
  expires_at: string
  messages: any[]
}

export type ConversationState =
  | "in-progress"
  | "resolved"
  | "backlog"
  | "non-viable"

export interface ServerChatMessage {
  id: string
  role: "system" | "user" | "assistant"
  sender?: string
  content: string
  created_at: string
  version?: number
  metadata_extra?: Record<string, unknown>
  pinned?: boolean
}

export type ChatSettingsResponse = {
  conversation_id: string
  settings: Record<string, unknown>
  last_modified: string
}

export type WorldBookProcessDiagnostic = {
  entry_id: number | null
  world_book_id: number | null
  activation_reason: "keyword_match" | "regex_match" | "depth" | string
  keyword?: string | null
  token_cost: number
  priority: number
  regex_match: boolean
  content_preview: string
  depth_level?: number | null
}

export type WorldBookProcessResponse = {
  injected_content: string
  entries_matched: number
  tokens_used: number
  books_used: number
  entry_ids: number[]
  token_budget?: number | null
  budget_exhausted?: boolean | null
  skipped_entries_due_to_budget?: number | null
  diagnostics: WorldBookProcessDiagnostic[]
}

export type LorebookDiagnosticTurn = {
  message_id: string
  timestamp?: string | null
  turn_number: number
  message_preview: string
  diagnostics: Record<string, unknown>[]
}

export type LorebookDiagnosticExportResponse = {
  chat_id: string
  character_id?: string | null
  total_turns_with_diagnostics: number
  turns: LorebookDiagnosticTurn[]
  page: number
  size: number
}

export type PromptPayload = {
  name?: string
  title?: string
  author?: string
  details?: string
  system_prompt?: string | null
  user_prompt?: string | null
  keywords?: string[]
  content?: string
  is_system?: boolean
}

export type FileArtifactExport = {
  status?: "none" | "ready" | "pending"
  format?: "ics" | "md" | "html" | "xlsx" | "csv" | "json" | "png" | "jpg" | "webp"
  url?: string | null
  content_type?: string | null
  bytes?: number | null
  job_id?: string | null
  content_b64?: string | null
  expires_at?: string | null
}

export type FileArtifact = {
  file_id: number
  file_type: string
  title: string
  structured?: Record<string, unknown>
  export?: FileArtifactExport
  created_at?: string
  updated_at?: string
}

export type FileCreateResponse = {
  artifact: FileArtifact
}

export type ReferenceImageCandidate = {
  file_id: number
  title: string
  mime_type: string
  width?: number | null
  height?: number | null
  created_at: string
}

export type ReferenceImageListResponse = {
  items: ReferenceImageCandidate[]
}

export type ImageArtifactRequest = {
  backend: string
  prompt: string
  negativePrompt?: string
  referenceFileId?: number
  width?: number
  height?: number
  steps?: number
  cfgScale?: number
  seed?: number
  sampler?: string
  model?: string
  extraParams?: Record<string, unknown>
  format?: "png" | "jpg" | "webp"
  title?: string
  persist?: boolean
  timeoutMs?: number
}

export interface ImageBackend {
  id: string
  name: string
  is_configured: boolean
  supported_formats?: string[]
}

export interface TldwEmbeddingModel {
  provider: string
  model: string
  allowed?: boolean
  default?: boolean
}

export interface TldwEmbeddingModelsResponse {
  data?: TldwEmbeddingModel[]
  allowed_providers?: string[] | null
  allowed_models?: string[] | null
}

export interface TldwEmbeddingProvidersConfig {
  default_provider: string
  default_model: string
  providers: {
    name: string
    models: string[]
  }[]
}

export type CharacterListSortBy =
  | "name"
  | "creator"
  | "created_at"
  | "updated_at"
  | "last_used_at"
  | "conversation_count"

export type CharacterListSortOrder = "asc" | "desc"

export interface CharacterListQueryParams {
  page?: number
  page_size?: number
  query?: string
  tags?: string[]
  match_all_tags?: boolean
  creator?: string
  has_conversations?: boolean
  favorite_only?: boolean
  created_from?: string
  created_to?: string
  updated_from?: string
  updated_to?: string
  sort_by?: CharacterListSortBy
  sort_order?: CharacterListSortOrder
  include_image_base64?: boolean
  include_deleted?: boolean
  deleted_only?: boolean
}

export interface CharacterListQueryResponse {
  items: any[]
  total: number
  page: number
  page_size: number
  has_more: boolean
}

export interface CharacterVersionEntry {
  change_id: number
  version: number
  operation: string
  timestamp?: string | null
  client_id?: string | null
  payload: Record<string, unknown>
}

export interface CharacterVersionListResponse {
  items: CharacterVersionEntry[]
  total: number
}

export interface CharacterVersionDiffField {
  field: string
  old_value: unknown
  new_value: unknown
}

export interface CharacterVersionDiffResponse {
  character_id: number
  from_entry: CharacterVersionEntry
  to_entry: CharacterVersionEntry
  changed_fields: CharacterVersionDiffField[]
  changed_count: number
}

// Admin / RBAC types
export interface AdminUserSummary {
  id: number
  uuid: string
  username: string
  email: string
  role: string
  is_active: boolean
  is_verified: boolean
  created_at: string
  last_login?: string | null
  storage_quota_mb: number
  storage_used_mb: number
}

export interface AdminUserListResponse {
  users: AdminUserSummary[]
  total: number
  page: number
  limit: number
  pages: number
}

export interface AdminUserUpdateRequest {
  email?: string
  role?: string
  is_active?: boolean
  is_verified?: boolean
  is_locked?: boolean
  storage_quota_mb?: number
}

export interface AdminRole {
  id: number
  name: string
  description?: string | null
  is_system?: boolean
}

// MLX admin types
export interface MlxStatusConfig {
  device?: string | null
  dtype?: string | null
  compile?: boolean
  warmup?: boolean
  max_seq_len?: number | null
  max_batch_size?: number | null
}

export interface MlxStatus {
  active: boolean
  model: string | null
  loaded_at: number | string | null
  supports_embeddings: boolean
  warmup_completed: boolean
  max_concurrent: number
  config?: MlxStatusConfig
}

export interface MlxLoadRequest {
  model_path?: string
  max_seq_len?: number
  max_batch_size?: number
  device?: string
  dtype?: string
  quantization?: string
  compile?: boolean
  warmup?: boolean
  prompt_template?: string
  revision?: string
  trust_remote_code?: boolean
  tokenizer?: string
  adapter?: string
  adapter_weights?: string
  max_kv_cache_size?: number
  max_concurrent?: number
}

export interface MlxUnloadRequest {
  reason?: string
}

export interface MediaIngestionBudgetDiagnostics {
  status: string
  entity?: string | null
  policy_id?: string | null
  limits?: {
    jobs_max_concurrent?: number | null
    ingestion_bytes_daily_cap?: number | null
  } | null
  usage?: {
    jobs_active?: number | null
    jobs_remaining?: number | null
    ingestion_bytes_daily_used?: number | null
    ingestion_bytes_daily_remaining?: number | null
  } | null
  retry_after?: number | null
  reason?: string | null
  error?: string | null
}

export class TldwApiClientBase {
  private storage: Storage
  private config: TldwConfig | null = null
  private baseUrl: string = ''
  private headers: HeadersInit = {}
  characterCache = new Map<string, { value: any; expiresAt: number }>()
  characterInFlight = new Map<string, Promise<any>>()
  chatMessagesCache = new Map<
    string,
    { value: ServerChatMessage[]; expiresAt: number }
  >()
  chatMessagesInFlight = new Map<string, Promise<ServerChatMessage[]>>()
  private openApiPathSet: Set<string> | null = null
  private openApiPathSetPromise: Promise<Set<string> | null> | null = null
  private resolvedPathCache = new Map<string, string>()

  constructor() {
    this.storage = createSafeStorage({
      serde: safeStorageSerde
    })
  }

  private getEnvApiKey(): string | null {
    try {
      const env: any = (import.meta as any)?.env || {}
      const raw =
        (env?.VITE_TLDW_API_KEY as string | undefined) ??
        (env?.VITE_TLDW_DEFAULT_API_KEY as string | undefined)
      const key = (raw || "").trim()
      return key || null
    } catch {
      return null
    }
  }

  private isDevMode(): boolean {
    try {
      const env: any = (import.meta as any)?.env || {}
      return Boolean(env?.DEV) || env?.MODE === "development"
    } catch {
      return false
    }
  }

  private getMissingApiKeyMessage(): string {
    return "tldw server API key is missing. Open Settings → tldw server and configure an API key before continuing."
  }

  normalizeRagQuery(rawQuery: string): string {
    const normalized =
      typeof rawQuery === "string" ? rawQuery : String(rawQuery ?? "")
    if (normalized.length <= RAG_QUERY_MAX_LENGTH) {
      return normalized
    }

    // Keep client behavior resilient to backend schema limits.
    const truncated = normalized.slice(0, RAG_QUERY_MAX_LENGTH).trimEnd()
    console.warn(
      `[tldw:rag] query exceeded ${RAG_QUERY_MAX_LENGTH} characters; truncating before request`,
      { originalLength: normalized.length, truncatedLength: truncated.length }
    )
    return truncated
  }

  getChatMessagesCacheKey(chatId: string, query: string): string {
    return `${chatId}${query || ""}`
  }

  invalidateChatMessagesCache(chatId?: string | number): void {
    const cid = chatId != null ? String(chatId) : null
    if (!cid) {
      this.chatMessagesCache.clear()
      return
    }
    for (const key of this.chatMessagesCache.keys()) {
      if (key.startsWith(cid)) {
        this.chatMessagesCache.delete(key)
      }
    }
  }

  private getPlaceholderApiKeyMessage(): string {
    return "tldw server API key is still set to a placeholder value. Replace it with your real API key in Settings → tldw server before continuing."
  }

  async ensureConfigForRequest(requireAuth: boolean): Promise<TldwConfig> {
    const cfg = (await this.getConfig()) || null
    const hostedMode = isHostedTldwDeployment()
    if ((!cfg || !cfg.serverUrl) && !hostedMode) {
      const msg =
        "tldw server is not configured. Open Settings → tldw server in the extension and set the server URL and API key."
      // eslint-disable-next-line no-console
      console.warn(msg)
      throw new Error(msg)
    }

    if (hostedMode) {
      return {
        serverUrl: String(cfg?.serverUrl || ""),
        apiKey: cfg?.apiKey,
        accessToken: cfg?.accessToken,
        refreshToken: cfg?.refreshToken,
        orgId: cfg?.orgId,
        authMode: cfg?.authMode || "multi-user"
      }
    }

    if (!requireAuth) {
      return cfg
    }

    if (cfg.authMode === "multi-user") {
      const token = (cfg.accessToken || "").trim()
      if (!token) {
        const msg =
          "Not authenticated. Please log in under Settings → tldw server before continuing."
        // eslint-disable-next-line no-console
        console.warn(msg)
        throw new Error(msg)
      }
      return cfg
    }

    // single-user auth
    const key = (cfg.apiKey || "").trim()
    if (!key) {
      const msg = this.getMissingApiKeyMessage()
      // eslint-disable-next-line no-console
      console.warn(msg)
      throw new Error(msg)
    }
    if (isPlaceholderApiKey(key)) {
      const msg = this.getPlaceholderApiKeyMessage()
      // eslint-disable-next-line no-console
      console.warn(msg)
      throw new Error(msg)
    }
    return cfg
  }

  async request<T>(init: any, requireAuth = true): Promise<T> {
    await this.ensureConfigForRequest(requireAuth && !init?.noAuth)
    return await bgRequest<T>(init)
  }

  async fetchWithAuth(
    path: PathOrUrl,
    init?: {
      method?: string
      headers?: Record<string, string>
      body?: any
      timeoutMs?: number
      signal?: AbortSignal
      responseType?: "json" | "text" | "arrayBuffer"
    }
  ): Promise<{
    ok: boolean
    status?: number
    error?: string
    data?: any
    json: () => Promise<any>
    text: () => Promise<string>
  }> {
    await this.ensureConfigForRequest(true)
    const response = await bgRequest<any, PathOrUrl>({
      path,
      method: (init?.method || "GET") as any,
      headers: init?.headers,
      body: init?.body,
      timeoutMs: init?.timeoutMs,
      abortSignal: init?.signal,
      responseType: init?.responseType,
      returnResponse: true
    })
    const data = response?.data
    const text = () => {
      if (typeof data === "string") return Promise.resolve(data)
      if (data == null) return Promise.resolve("")
      try {
        return Promise.resolve(JSON.stringify(data))
      } catch {
        return Promise.resolve(String(data))
      }
    }
    return {
      ok: Boolean(response?.ok),
      status: response?.status,
      error: response?.error,
      data,
      json: () => Promise.resolve(data),
      text
    }
  }

  async upload<T>(init: any, requireAuth = true): Promise<T> {
    await this.ensureConfigForRequest(requireAuth)
    return await bgUpload<T>(init)
  }

  async *stream(init: any, requireAuth = true): AsyncGenerator<string> {
    await this.ensureConfigForRequest(requireAuth)
    for await (const line of bgStream(init)) {
      yield line as string
    }
  }

  async initialize(): Promise<void> {
    let stored = await this.storage.get<TldwConfig>("tldwConfig")
    if (!stored) {
      try {
        const localStore = createSafeStorage({
          area: "local",
          serde: safeStorageSerde
        })
        const localConfig = await localStore.get<TldwConfig>("tldwConfig")
        if (localConfig) {
          stored = localConfig
          await this.storage.set("tldwConfig", localConfig)
        }
      } catch {
        // ignore migration failures
      }
    }
    const envApiKey = this.getEnvApiKey()
    const quickstartWebUiServerUrl = getQuickstartWebUiServerUrl()

    if (!stored) {
      // True first-run: leave config null so callers (like the connection
      // store) can distinguish an unconfigured state from a misconfigured
      // or unreachable server.
      this.config = null
    } else {
      const hydrated: TldwConfig = {
        ...stored,
        // Default authMode but do not silently inject a server URL if none
        // has been configured yet.
        authMode: stored.authMode || "single-user",
        serverUrl: quickstartWebUiServerUrl || stored.serverUrl || ""
      }
      if (!hydrated.apiKey && envApiKey) {
        hydrated.apiKey = envApiKey
      }
      this.config = hydrated
      await this.storage.set("tldwConfig", hydrated)
    }

    const config = this.config
    const hostedMode = isHostedTldwDeployment()
    const nextBaseUrl = hostedMode
      ? String(config?.serverUrl || "").replace(/\/$/, "")
      : (quickstartWebUiServerUrl || config?.serverUrl || DEFAULT_SERVER_URL).replace(/\/$/, "")
    if (this.baseUrl && this.baseUrl !== nextBaseUrl) {
      this.openApiPathSet = null
      this.openApiPathSetPromise = null
      this.resolvedPathCache.clear()
    }
    this.baseUrl = nextBaseUrl

    // Set up headers based on auth mode
    this.headers = {
      "Content-Type": "application/json"
    }

    if (!hostedMode && config?.authMode === "single-user" && config.apiKey) {
      const key = String(config.apiKey || "").trim()
      if (key) {
        this.headers["X-API-KEY"] = key
      }
    } else if (!hostedMode && config?.authMode === "multi-user" && config.accessToken) {
      this.headers["Authorization"] = `Bearer ${config.accessToken}`
    }
    if (config?.orgId) {
      this.headers["X-TLDW-Org-Id"] = String(config.orgId)
    }
  }

  async getConfig(): Promise<TldwConfig | null> {
    if (this.config === null) {
      await this.initialize().catch(() => null)
    }
    return this.config
  }

  async updateConfig(config: Partial<TldwConfig>): Promise<void> {
    const currentConfig = (await this.getConfig()) || {}
    const newConfig = { ...(currentConfig as any), ...config } as TldwConfig
    await this.storage.set('tldwConfig', newConfig)
    this.config = newConfig
    await this.initialize().catch(() => null)
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("tldw:config-updated"))
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      await bgRequest<{ status?: string; [k: string]: any }>({
        path: '/api/v1/health',
        method: 'GET'
      })
      return true
    } catch {
      // Swallow errors to avoid noisy console during first-run
      return false
    }
  }

  async getServerInfo(): Promise<any> {
    return await bgRequest<any>({ path: '/', method: 'GET' })
  }

  async getCurrentUserProfile(params?: {
    sections?: string | string[]
    includeSources?: boolean
    includeRaw?: boolean
    maskSecrets?: boolean
  }): Promise<any> {
    const sections = Array.isArray(params?.sections)
      ? params?.sections.join(",")
      : params?.sections
    const query = this.buildQuery({
      sections,
      include_sources: params?.includeSources,
      include_raw: params?.includeRaw,
      mask_secrets: params?.maskSecrets
    })
    return await this.request<any>({
      path: `/api/v1/users/me/profile${query}`,
      method: "GET"
    })
  }

  async updateCurrentUserProfile(payload: {
    updates: UserProfileUpdateEntry[]
    profile_version?: string
    dry_run?: boolean
  }): Promise<UserProfileUpdateResponse> {
    return await this.request<UserProfileUpdateResponse>({
      path: "/api/v1/users/me/profile",
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async getCurrentUserStorageQuota(): Promise<CurrentUserStorageQuotaResponse> {
    return await this.request<CurrentUserStorageQuotaResponse>({
      path: "/api/v1/users/storage",
      method: "GET"
    })
  }

  async startOpenAIOAuthAuthorize(
    payload: OpenAIOAuthAuthorizeRequest = {}
  ): Promise<OpenAIOAuthAuthorizeResponse> {
    return await this.request<OpenAIOAuthAuthorizeResponse>({
      path: "/api/v1/users/keys/openai/oauth/authorize",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async getOpenAIOAuthStatus(): Promise<OpenAIOAuthStatusResponse> {
    return await this.request<OpenAIOAuthStatusResponse>({
      path: "/api/v1/users/keys/openai/oauth/status",
      method: "GET"
    })
  }

  async refreshOpenAIOAuth(): Promise<OpenAIOAuthRefreshResponse> {
    return await this.request<OpenAIOAuthRefreshResponse>({
      path: "/api/v1/users/keys/openai/oauth/refresh",
      method: "POST"
    })
  }

  async disconnectOpenAIOAuth(): Promise<void> {
    await this.request<void>({
      path: "/api/v1/users/keys/openai/oauth",
      method: "DELETE"
    })
  }

  async switchOpenAICredentialSource(
    authSource: "api_key" | "oauth"
  ): Promise<OpenAICredentialSourceSwitchResponse> {
    return await this.request<OpenAICredentialSourceSwitchResponse>({
      path: "/api/v1/users/keys/openai/source",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { auth_source: authSource }
    })
  }

  // ── BYOK Provider Keys ──────────────────────────────────────────────

  async listUserProviderKeys(): Promise<{
    items: Array<{
      provider: string
      has_key: boolean
      source: string
      key_hint: string | null
      auth_source: string | null
      last_used_at: string | null
    }>
  }> {
    return await this.request<{
      items: Array<{
        provider: string
        has_key: boolean
        source: string
        key_hint: string | null
        auth_source: string | null
        last_used_at: string | null
      }>
    }>({
      path: "/api/v1/users/keys",
      method: "GET"
    })
  }

  async upsertUserProviderKey(
    provider: string,
    apiKey: string,
    opts?: { credential_fields?: Record<string, unknown>; metadata?: Record<string, unknown> }
  ): Promise<{
    provider: string
    status: string
    key_hint: string
    updated_at: string
  }> {
    return await this.request<{
      provider: string
      status: string
      key_hint: string
      updated_at: string
    }>({
      path: "/api/v1/users/keys",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        provider,
        api_key: apiKey,
        ...opts,
      }
    })
  }

  async testUserProviderKey(
    provider: string,
    model?: string
  ): Promise<{
    provider: string
    status: string
    model: string | null
  }> {
    return await this.request<{
      provider: string
      status: string
      model: string | null
    }>({
      path: "/api/v1/users/keys/test",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { provider, model }
    })
  }

  async deleteUserProviderKey(provider: string): Promise<void> {
    await this.request<void>({
      path: `/api/v1/users/keys/${encodeURIComponent(provider)}`,
      method: "DELETE"
    })
  }

  async getDefaultCharacterPreference(): Promise<string | null> {
    const profile = await this.getCurrentUserProfile({
      sections: "preferences"
    })
    const raw = profile?.preferences?.[DEFAULT_CHARACTER_PROFILE_PREFERENCE_KEY]
    if (
      raw &&
      typeof raw === "object" &&
      !Array.isArray(raw) &&
      "value" in raw
    ) {
      return normalizeDefaultCharacterPreferenceId(
        (raw as { value?: unknown }).value
      )
    }
    return normalizeDefaultCharacterPreferenceId(raw)
  }

  async setDefaultCharacterPreference(
    characterId: string | null
  ): Promise<UserProfileUpdateResponse> {
    const normalizedCharacterId = normalizeDefaultCharacterPreferenceId(characterId)
    return await this.updateCurrentUserProfile({
      updates: [
        {
          key: DEFAULT_CHARACTER_PROFILE_PREFERENCE_KEY,
          value: normalizedCharacterId
        }
      ]
    })
  }

  buildQuery(params?: Record<string, any>): string {
    if (!params || Object.keys(params).length === 0) {
      return ''
    }
    const search = new URLSearchParams()
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined || value === null) continue
      if (Array.isArray(value)) {
        value.forEach((entry) => search.append(key, String(entry)))
        continue
      }
      search.append(key, String(value))
    }
    const query = search.toString()
    return query ? `?${query}` : ''
  }

  async getOpenAPISpec(): Promise<any | null> {
    try {
      if (!this.baseUrl) await this.initialize()
      if (!this.baseUrl) return null
      return await bgRequest<any>({
        path: `${this.baseUrl.replace(/\/$/, '')}/openapi.json` as any,
        method: 'GET' as any
      })
    } catch {
      return null
    }
  }

  normalizePathShape(path: string): string {
    return path.replace(/\{[^}]+\}/g, "{}")
  }

  async getOpenApiPathSet(): Promise<Set<string> | null> {
    if (this.openApiPathSet) return this.openApiPathSet
    if (!this.openApiPathSetPromise) {
      this.openApiPathSetPromise = (async () => {
        const spec = await this.getOpenAPISpec()
        if (!spec?.paths || typeof spec.paths !== "object") {
          this.openApiPathSet = null
          this.openApiPathSetPromise = null
          return null
        }
        const paths = new Set(Object.keys(spec.paths))
        this.openApiPathSet = paths
        this.resolvedPathCache.clear()
        return paths
      })()
    }
    return this.openApiPathSetPromise
  }

  async resolveApiPath(
    key: string,
    candidates: string[]
  ): Promise<AllowedPath> {
    const cached = this.resolvedPathCache.get(key)
    if (cached) return cached as AllowedPath
    const fallback = candidates[0] as AllowedPath
    if (!fallback) {
      throw new Error(`No path candidates provided for ${key}`)
    }
    const specPaths = await this.getOpenApiPathSet().catch(() => null)
    if (!specPaths || specPaths.size === 0) {
      this.resolvedPathCache.set(key, fallback)
      return fallback
    }

    const specShapes = new Set(
      Array.from(specPaths, (path) => this.normalizePathShape(String(path)))
    )

    const resolved =
      candidates.find((candidate) => {
        if (specPaths.has(candidate)) return true
        return specShapes.has(this.normalizePathShape(candidate))
      }) || fallback

    this.resolvedPathCache.set(key, resolved)
    return resolved as AllowedPath
  }

  fillPathParams(
    template: AllowedPath,
    values: string | string[]
  ): AllowedPath {
    if (!template.includes("{")) return template
    if (Array.isArray(values)) {
      let index = 0
      return template.replace(/\{[^}]+\}/g, () => {
        const value = values[index] ?? ""
        index += 1
        return encodeURIComponent(value)
      }) as AllowedPath
    }
    const encoded = encodeURIComponent(values)
    return template.replace(/\{[^}]+\}/g, encoded) as AllowedPath
  }

  async postChatMetric(payload: Record<string, unknown>): Promise<any> {
    return await this.request<any>({
      path: "/api/v1/metrics/chat",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async getModels(options?: {
    refreshOpenRouter?: boolean
  }): Promise<TldwModel[]> {
    const meta = await this.getModelsMetadata(options)
    const list =
      Array.isArray(meta) && meta.length > 0
        ? meta
        : meta && typeof meta === "object" && Array.isArray((meta as any).models)
          ? (meta as any).models
          : []

    const toNonEmptyString = (value: unknown): string | null => {
      if (typeof value !== "string") return null
      const trimmed = value.trim()
      return trimmed.length > 0 ? trimmed : null
    }
    const isLikelyModelId = (value: string): boolean => {
      if (/\s/.test(value)) return false
      return /[/:._-]/.test(value)
    }

    return list.map((m: any) => {
      const rawModel =
        toNonEmptyString(m.model) || toNonEmptyString(m.model_id)
      const rawName = toNonEmptyString(m.name)
      const rawId = toNonEmptyString(m.id)
      const canonicalModelId =
        rawModel ||
        (rawName && isLikelyModelId(rawName) ? rawName : null) ||
        rawId ||
        rawName ||
        "unknown-model"
      const displayName =
        rawName && !isLikelyModelId(rawName) && rawName !== canonicalModelId
          ? `${rawName} (${canonicalModelId})`
          : canonicalModelId

      return {
        id: canonicalModelId,
        name: displayName,
        provider: String(m.provider || "default"),
        description: m.description,
        capabilities: Array.isArray(m.capabilities)
          ? m.capabilities
          : Array.isArray(m.features)
            ? m.features
            : typeof m.capabilities === "object"
              ? m.capabilities
              : undefined,
        context_length:
          typeof m.context_length === "number"
            ? m.context_length
            : typeof m.context_window === "number"
              ? m.context_window
              : typeof m.contextLength === "number"
                ? m.contextLength
                : undefined,
        vision: Boolean(
          (m.capabilities && m.capabilities.vision) ?? m.vision
        ),
        function_calling: Boolean(
          (m.capabilities &&
            (m.capabilities.function_calling || m.capabilities.tool_use)) ??
            m.function_calling
        ),
        json_output: Boolean(
          (m.capabilities && m.capabilities.json_mode) ?? m.json_output
        ),
        type: typeof m.type === "string" ? m.type : undefined,
        modalities:
          m.modalities && typeof m.modalities === "object"
            ? {
                input: Array.isArray(m.modalities.input)
                  ? m.modalities.input.map((v: any) => String(v))
                  : undefined,
                output: Array.isArray(m.modalities.output)
                  ? m.modalities.output.map((v: any) => String(v))
                  : undefined
              }
            : {
                input: Array.isArray(m.input_modality)
                  ? m.input_modality.map((v: any) => String(v))
                  : Array.isArray(m.input_modalities)
                    ? m.input_modalities.map((v: any) => String(v))
                    : typeof m.input_modality === "string"
                      ? [String(m.input_modality)]
                      : undefined,
                output: Array.isArray(m.output_modality)
                  ? m.output_modality.map((v: any) => String(v))
                  : Array.isArray(m.output_modalities)
                    ? m.output_modalities.map((v: any) => String(v))
                    : typeof m.output_modality === "string"
                      ? [String(m.output_modality)]
                      : undefined
              }
      }
    })
  }

  async getProviders(): Promise<any> {
    return await bgRequest<any>({ path: '/api/v1/llm/providers', method: 'GET' })
  }

  /**
   * Check which LLM providers are configured on the server.
   * Returns `{ providers: [...], any_configured: boolean }`.
   */
  async getProvidersStatus(): Promise<{
    providers: Array<{
      name: string
      configured: boolean
      requires_api_key: boolean
      key_hint?: string | null
      key_source?: string | null
    }>
    any_configured: boolean
  }> {
    return await bgRequest<any>({ path: '/api/v1/config/providers', method: 'GET' })
  }

  async getModelsMetadata(options?: {
    refreshOpenRouter?: boolean
  }): Promise<any> {
    // tldw_server returns either an array or an object
    // of the form { models: [...], total: N }.
    const query = options?.refreshOpenRouter ? "?refresh_openrouter=true" : ""
    const path = appendPathQuery("/api/v1/llm/models/metadata", query)
    return await bgRequest<any>({ path, method: 'GET' })
  }

  async getImageBackends(): Promise<ImageBackend[]> {
    try {
      const meta = await this.getModelsMetadata()
      const list: any[] =
        Array.isArray(meta) && meta.length > 0
          ? meta
          : meta && typeof meta === "object" && Array.isArray((meta as any).models)
            ? (meta as any).models
            : []

      return list
        .filter((m: any) => m.type === "image")
        .map((m: any) => ({
          id: String(m.name || m.id || "").replace(/^image\//, ""),
          name: String(m.name || m.id || ""),
          is_configured: Boolean(m.is_configured),
          supported_formats: Array.isArray(m.supported_formats) ? m.supported_formats : undefined
        }))
        .filter((b) => b.id.length > 0)
    } catch (e) {
      if (import.meta.env?.DEV) {
        console.warn("tldw_server: getImageBackends failed", e)
      }
      return []
    }
  }

  async generateImage(payload: {
    backend: string
    prompt: string
    negative_prompt?: string
    width?: number
    height?: number
    steps?: number
    cfg_scale?: number
    format?: "png" | "jpg" | "webp"
    persist?: boolean
    timeoutMs?: number
  }): Promise<{ content_b64: string; content_type: string }> {
    const response = await this.createImageArtifact({
      backend: payload.backend,
      prompt: payload.prompt,
      negativePrompt: payload.negative_prompt,
      width: payload.width,
      height: payload.height,
      steps: payload.steps,
      cfgScale: payload.cfg_scale,
      format: payload.format,
      persist: payload.persist,
      timeoutMs: payload.timeoutMs
    })
    const exportInfo = response?.artifact?.export
    const content_b64 = exportInfo?.content_b64
    if (!content_b64) {
      throw new Error("Image generation returned no data.")
    }
    const content_type =
      exportInfo?.content_type ||
      (exportInfo?.format ? `image/${exportInfo.format}` : "image/png")
    return { content_b64, content_type }
  }

  // Embeddings - Models & Providers
  async getEmbeddingModelsList(): Promise<TldwEmbeddingModel[]> {
    try {
      const data = await bgRequest<TldwEmbeddingModelsResponse | TldwEmbeddingModel[]>({
        path: "/api/v1/embeddings/models",
        method: "GET"
      })

      const list: any[] = Array.isArray(data)
        ? data
        : Array.isArray((data as TldwEmbeddingModelsResponse)?.data)
          ? (data as TldwEmbeddingModelsResponse).data!
          : []

      return list
        .map((item) => ({
          provider: String((item as any).provider || "unknown"),
          model: String((item as any).model || ""),
          allowed:
            typeof (item as any).allowed === "boolean"
              ? Boolean((item as any).allowed)
              : true,
          default: Boolean((item as any).default)
        }))
        .filter((m) => m.model.length > 0)
    } catch (e) {
      if (import.meta.env?.DEV) {
        console.warn("tldw_server: GET /api/v1/embeddings/models failed", e)
      }
      return []
    }
  }

  async getEmbeddingProvidersConfig(): Promise<TldwEmbeddingProvidersConfig | null> {
    try {
      const cfg = await bgRequest<TldwEmbeddingProvidersConfig>({
        path: "/api/v1/embeddings/providers-config",
        method: "GET"
      })
      return cfg
    } catch (e) {
      if (import.meta.env?.DEV) {
        console.warn(
          "tldw_server: GET /api/v1/embeddings/providers-config failed",
          e
        )
      }
      return null
    }
  }

  // Admin / diagnostics helpers
  async getSystemStats(options?: { timeoutMs?: number }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/stats",
      method: "GET",
      timeoutMs: options?.timeoutMs
    })
  }

  async getMediaIngestionBudgetDiagnostics(params: {
    userId: number
    policyId?: string
  }): Promise<MediaIngestionBudgetDiagnostics> {
    const query = this.buildQuery({
      user_id: params.userId,
      policy_id: params.policyId || "media.default"
    })
    return await bgRequest<MediaIngestionBudgetDiagnostics>({
      path: `/api/v1/resource-governor/diag/media-budget${query}`,
      method: "GET"
    })
  }

  async getLlamacppStatus(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/llamacpp/status",
      method: "GET"
    })
  }

  async listLlamacppModels(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/llamacpp/models",
      method: "GET"
    })
  }

  async startLlamacppServer(
    modelFilename: string,
    serverArgs?: Record<string, any>
  ): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/llamacpp/start_server",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        model_filename: modelFilename,
        server_args: serverArgs || {}
      }
    })
  }

  async stopLlamacppServer(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/llamacpp/stop_server",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  }

  async getLlmProviders(
    includeDeprecated = false
  ): Promise<any> {
    const query = this.buildQuery(includeDeprecated ? { include_deprecated: true } : {})
    return await bgRequest<any>({
      path: `/api/v1/llm/providers${query}`,
      method: "GET"
    })
  }

  // MLX admin helpers
  async getMlxStatus(): Promise<MlxStatus> {
    return await bgRequest<MlxStatus>({
      path: "/api/v1/llm/providers/mlx/status",
      method: "GET"
    })
  }

  async loadMlxModel(payload: MlxLoadRequest): Promise<MlxStatus> {
    return await bgRequest<MlxStatus>({
      path: "/api/v1/llm/providers/mlx/load",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async unloadMlxModel(payload?: MlxUnloadRequest): Promise<{ message?: string }> {
    return await bgRequest<{ message?: string }>({
      path: "/api/v1/llm/providers/mlx/unload",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload || {}
    })
  }

  async createChatCompletion(request: ChatCompletionRequest): Promise<Response> {
    // Non-stream request via background
    captureChatRequestDebugSnapshot({
      endpoint: "/api/v1/chat/completions",
      method: "POST",
      mode: "non-stream",
      body: request
    })
    const res = await bgRequest<Response>({ path: '/api/v1/chat/completions', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: request })
    // bgRequest returns parsed data; for non-streaming chat we expect a JSON structure or text. To keep existing consumers happy, wrap as Response-like
    // For simplicity, return a minimal object with json() and text()
    const data = res as any
    const safeData = normalizeChatCompletionResponseBody(data)
    return createJsonResponseLike(safeData, { status: 200 })
  }

  async *streamChatCompletion(request: ChatCompletionRequest, options?: { signal?: AbortSignal; streamIdleTimeoutMs?: number }): AsyncGenerator<any, void, unknown> {
    request.stream = true
    captureChatRequestDebugSnapshot({
      endpoint: "/api/v1/chat/completions",
      method: "POST",
      mode: "stream",
      body: request
    })
    for await (const line of bgStream({ path: '/api/v1/chat/completions', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: request, abortSignal: options?.signal, streamIdleTimeoutMs: options?.streamIdleTimeoutMs })) {
      try {
        const parsed = JSON.parse(line)
        yield parsed
      } catch (e) {
        // Ignore non-JSON lines
      }
    }
  }

  // RAG Methods
  async ragHealth(): Promise<any> {
    return await this.request<any>({ path: '/api/v1/rag/health', method: 'GET' })
  }

  async ragSearch(query: string, options?: any): Promise<any> {
    const { timeoutMs, signal, ...rest } = options || {}
    const normalizedQuery = this.normalizeRagQuery(query)
    try {
      return await bgRequest<any>({
        path: '/api/v1/rag/search',
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: { query: normalizedQuery, ...rest },
        timeoutMs,
        abortSignal: signal
      })
    } catch (error) {
      const status = (error as { status?: number } | null)?.status
      const message = error instanceof Error ? error.message : String(error ?? '')
      const aborted =
        (error as { name?: string } | null)?.name === 'AbortError' ||
        /abort|cancel/i.test(message)
      if (aborted) {
        throw error
      }
      const shouldRetryWithoutRerank =
        status === 500 &&
        rest?.enable_reranking !== false &&
        rest?.reranking_strategy !== 'none'

      if (!shouldRetryWithoutRerank) {
        throw error
      }

      // Some local/dev servers fail hard when FlashRank assets are missing.
      // Retry once with reranking disabled so retrieval still works.
      console.warn(
        '[tldw:rag] /api/v1/rag/search failed; retrying once without reranking',
        { status, message }
      )
      return await bgRequest<any>({
        path: '/api/v1/rag/search',
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: {
          query: normalizedQuery,
          ...rest,
          enable_reranking: false,
          reranking_strategy: 'none'
        },
        timeoutMs,
        abortSignal: signal
      })
    }
  }

  async *ragSearchStream(
    query: string,
    options?: any
  ): AsyncGenerator<any, void, unknown> {
    const { timeoutMs, signal, ...rest } = options || {}
    const normalizedQuery = this.normalizeRagQuery(query)
    for await (const line of bgStream({
      path: '/api/v1/rag/search/stream',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: { query: normalizedQuery, ...rest },
      abortSignal: signal,
      streamIdleTimeoutMs: timeoutMs
    })) {
      try {
        yield JSON.parse(line)
      } catch {
        // Ignore malformed stream chunks
      }
    }
  }

  async ragSimple(query: string, options?: any): Promise<any> {
    const { timeoutMs, ...rest } = options || {}
    const normalizedQuery = this.normalizeRagQuery(query)
    return await bgRequest<any>({ path: '/api/v1/rag/simple', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: { query: normalizedQuery, ...rest }, timeoutMs })
  }

  // Research / Web search
  async webSearch(options: any): Promise<any> {
    const { timeoutMs, signal, ...rest } = options || {}
    return await bgRequest<any>({
      path: "/api/v1/research/websearch",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: rest,
      timeoutMs,
      abortSignal: signal
    })
  }

  // Media Methods
  async addMedia(url: string, metadata?: any): Promise<any> {
    const sourceUrl = String(url || "").trim()
    const {
      timeoutMs,
      media_type,
      urls: rawUrls,
      ...rest
    } = metadata || {}
    const urls = Array.isArray(rawUrls)
      ? rawUrls
          .map((item: unknown) => String(item || "").trim())
          .filter((item: string) => item.length > 0)
      : typeof rawUrls === "string" && rawUrls.trim()
        ? [rawUrls.trim()]
        : sourceUrl
          ? [sourceUrl]
          : []
    if (urls.length === 0) {
      throw new Error("addMedia requires a URL")
    }
    const resolvedMediaType =
      typeof media_type === "string" && media_type.trim()
        ? media_type.trim()
        : inferUploadMediaTypeFromUrl(urls[0])

    return await bgUpload<any>({
      path: "/api/v1/media/add",
      method: "POST",
      fields: {
        ...rest,
        media_type: resolvedMediaType,
        urls
      },
      timeoutMs
    })
  }

  async submitMediaIngestJobs(fields?: Record<string, any>): Promise<any> {
    const { timeoutMs, ...rest } = fields || {}
    const normalized: Record<string, any> = {}
    for (const [k, v] of Object.entries(rest || {})) {
      if (typeof v === "undefined" || v === null) continue
      normalized[k] = v
    }
    return await bgUpload<any>({
      path: "/api/v1/media/ingest/jobs",
      method: "POST",
      fields: normalized,
      timeoutMs
    })
  }

  async getMediaIngestJob(
    jobId: number | string,
    options?: { timeoutMs?: number }
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/media/ingest/jobs/${encodeURIComponent(String(jobId))}`,
      method: "GET",
      timeoutMs: options?.timeoutMs
    })
  }

  async listMediaIngestJobs(
    params: {
      batch_id: string
      limit?: number
    },
    options?: { timeoutMs?: number }
  ): Promise<any> {
    const query = this.buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/media/ingest/jobs${query}`,
      method: "GET",
      timeoutMs: options?.timeoutMs
    })
  }

  async addMediaForm(fields: Record<string, any>): Promise<any> {
    // Multipart form for rich ingest parameters
    // Accepts a flat fields map; callers may pass booleans/strings and they will be converted
    const normalized: Record<string, any> = {}
    for (const [k, v] of Object.entries(fields || {})) {
      if (typeof v === 'undefined' || v === null) continue
      if (typeof v === 'boolean') normalized[k] = v ? 'true' : 'false'
      else normalized[k] = v
    }
    return await bgUpload<any>({ path: '/api/v1/media/add', method: 'POST', fields: normalized })
  }

  async uploadMedia(file: File, fields?: Record<string, any>): Promise<any> {
    const data = await file.arrayBuffer()
    const name = file.name || 'upload'
    const type = file.type || 'application/octet-stream'
    const normalized: Record<string, any> = {}
    for (const [k, v] of Object.entries(fields || {})) {
      if (typeof v === 'undefined' || v === null) continue
      if (typeof v === 'boolean') normalized[k] = v ? 'true' : 'false'
      else normalized[k] = v
    }
    let uploadTimeoutMs = 60000
    const cfg = await this.getConfig().catch(() => null)
    if (cfg && typeof (cfg as any).uploadRequestTimeoutMs === "number") {
      const cfgTimeout = Number((cfg as any).uploadRequestTimeoutMs)
      if (cfgTimeout > 0) {
        uploadTimeoutMs = cfgTimeout
      }
    }
    uploadTimeoutMs = Math.max(uploadTimeoutMs, 5000)
    return await bgUpload<any>({
      path: '/api/v1/media/add',
      method: 'POST',
      fields: normalized,
      file: { name, type, data },
      fileFieldName: 'files',
      timeoutMs: uploadTimeoutMs
    })
  }

  async listMedia(
    params?: {
      page?: number
      results_per_page?: number
      include_keywords?: boolean
    },
    options?: { signal?: AbortSignal }
  ): Promise<any> {
    const query = this.buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/media${query}`,
      method: "GET",
      abortSignal: options?.signal
    })
  }

  async searchMedia(
    payload: {
      query?: string
      fields?: string[]
      exact_phrase?: string
      media_types?: string[]
      date_range?: Record<string, any>
      must_have?: string[]
      must_not_have?: string[]
      sort_by?: string
      boost_fields?: Record<string, number>
    },
    params?: { page?: number; results_per_page?: number },
    options?: { signal?: AbortSignal }
  ): Promise<any> {
    const query = this.buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/media/search${query}`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      abortSignal: options?.signal
    })
  }

  async updateMediaKeywords(
    mediaId: string | number,
    payload: { keywords: string[]; mode?: "add" | "remove" | "set" }
  ): Promise<{ media_id: number; keywords: string[] }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{ media_id: number; keywords: string[] }>({
      path: `/api/v1/media/${id}/keywords`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async bulkUpdateMediaKeywords(payload: {
    media_ids: number[]
    keywords: string[]
    mode?: "add" | "remove" | "set"
  }): Promise<{
    endpoint: "bulk" | "fallback"
    updated: number
    failed: number
    results: Array<{
      media_id: number
      success: boolean
      keywords: string[] | null
      error: string | null
    }>
  }> {
    const rawIds = Array.isArray(payload.media_ids) ? payload.media_ids : []
    const mediaIds = Array.from(
      new Set(
        rawIds
          .map((id) => Number(id))
          .filter((id) => Number.isFinite(id) && id > 0)
          .map((id) => Math.trunc(id))
      )
    )
    if (mediaIds.length === 0) {
      throw new Error("media_ids_required")
    }

    const keywords = Array.isArray(payload.keywords)
      ? payload.keywords
          .map((keyword) => String(keyword ?? "").trim())
          .filter((keyword) => keyword.length > 0)
      : []
    const mode = payload.mode ?? "add"

    const requestPayload = {
      media_ids: mediaIds,
      keywords,
      mode
    } as const

    try {
      const response = await bgRequest<any>({
        path: "/api/v1/media/bulk/keyword-update",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestPayload
      })
      const results = Array.isArray(response?.results)
        ? response.results.map((entry: any) => ({
            media_id: Number(entry?.media_id ?? entry?.id ?? 0),
            success: Boolean(entry?.success ?? true),
            keywords: Array.isArray(entry?.keywords) ? entry.keywords.map(String) : null,
            error:
              typeof entry?.error === "string"
                ? entry.error
                : typeof entry?.detail === "string"
                  ? entry.detail
                  : null
          }))
        : mediaIds.map((mediaId) => ({
            media_id: mediaId,
            success: true,
            keywords: null,
            error: null
          }))
      const updatedCount =
        typeof response?.updated === "number"
          ? Math.max(0, Math.trunc(response.updated))
          : results.filter((entry) => entry.success).length
      const failedCount =
        typeof response?.failed === "number"
          ? Math.max(0, Math.trunc(response.failed))
          : Math.max(0, results.length - updatedCount)

      return {
        endpoint: "bulk",
        updated: updatedCount,
        failed: failedCount,
        results
      }
    } catch (error) {
      const candidate = error as
        | { status?: number; response?: { status?: number }; statusCode?: number }
        | undefined
      const statusCode = Number(
        candidate?.status ?? candidate?.response?.status ?? candidate?.statusCode
      )
      if (!Number.isFinite(statusCode) || (statusCode !== 404 && statusCode !== 405)) {
        throw error
      }
    }

    const settled = await Promise.allSettled(
      mediaIds.map(async (mediaId) => {
        const updated = await this.updateMediaKeywords(mediaId, {
          keywords,
          mode
        })
        return {
          media_id: mediaId,
          success: true,
          keywords: Array.isArray(updated?.keywords) ? updated.keywords : [],
          error: null as string | null
        }
      })
    )

    const results = settled.map((entry, index) => {
      const mediaId = mediaIds[index]
      if (entry.status === "fulfilled") {
        return entry.value
      }
      const reason = entry.reason
      const detail =
        typeof reason?.message === "string"
          ? reason.message
          : typeof reason === "string"
            ? reason
            : "keyword_update_failed"
      return {
        media_id: mediaId,
        success: false,
        keywords: null,
        error: detail
      }
    })
    const updated = results.filter((entry) => entry.success).length

    return {
      endpoint: "fallback",
      updated,
      failed: results.length - updated,
      results
    }
  }

  async deleteMedia(mediaId: string | number): Promise<void> {
    const id = encodeURIComponent(String(mediaId))
    await bgRequest<void>({
      path: `/api/v1/media/${id}`,
      method: "DELETE"
    })
  }

  async restoreMedia(mediaId: string | number): Promise<any> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<any>({
      path: `/api/v1/media/${id}/restore`,
      method: "POST"
    })
  }

  async permanentlyDeleteMedia(mediaId: string | number): Promise<void> {
    const id = encodeURIComponent(String(mediaId))
    await bgRequest<void>({
      path: `/api/v1/media/${id}/permanent`,
      method: "DELETE"
    })
  }

  async reprocessMedia(
    mediaId: string | number,
    options?: Record<string, unknown>
  ): Promise<any> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<any>({
      path: `/api/v1/media/${id}/reprocess`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: options || {}
    })
  }

  async getMediaStatistics(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/media/statistics",
      method: "GET"
    })
  }

  async getMediaDetails(
    mediaId: string | number,
    options?: {
      include_content?: boolean
      include_versions?: boolean
      include_version_content?: boolean
      signal?: AbortSignal
    }
  ): Promise<any> {
    const id = encodeURIComponent(String(mediaId))
    const query = this.buildQuery({
      include_content: options?.include_content ?? true,
      include_versions: options?.include_versions ?? false,
      include_version_content: options?.include_version_content ?? false
    })
    return await bgRequest<any>({
      path: `/api/v1/media/${id}${query}`,
      method: "GET",
      abortSignal: options?.signal
    })
  }

  async getDocumentOutline(
    mediaId: string | number,
    options?: { signal?: AbortSignal }
  ): Promise<{
    media_id: number
    has_outline: boolean
    entries: Array<{ level: number; title: string; page: number }>
    total_pages: number
  }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{
      media_id: number
      has_outline: boolean
      entries: Array<{ level: number; title: string; page: number }>
      total_pages: number
    }>({
      path: `/api/v1/media/${id}/outline`,
      method: "GET",
      abortSignal: options?.signal
    })
  }

  async generateDocumentInsights(
    mediaId: string | number,
    options?: {
      categories?: string[]
      model?: string
      max_content_length?: number
      force?: boolean
      signal?: AbortSignal
    }
  ): Promise<{
    media_id: number
    insights: Array<{
      category: string
      title: string
      content: string
      confidence?: number
    }>
    model_used: string
    cached: boolean
  }> {
    const id = encodeURIComponent(String(mediaId))
    const body: Record<string, unknown> = {}
    if (options?.categories) body.categories = options.categories
    if (options?.model) body.model = options.model
    if (options?.max_content_length) body.max_content_length = options.max_content_length
    if (options?.force) body.force = options.force

    return await bgRequest<{
      media_id: number
      insights: Array<{
        category: string
        title: string
        content: string
        confidence?: number
      }>
      model_used: string
      cached: boolean
    }>({
      path: `/api/v1/media/${id}/insights`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      abortSignal: options?.signal
    })
  }

  async getDocumentFigures(
    mediaId: string | number,
    options?: {
      minSize?: number
      signal?: AbortSignal
    }
  ): Promise<{
    media_id: number
    has_figures: boolean
    figures: Array<{
      id: string
      page: number
      width: number
      height: number
      format: string
      data_url?: string
      caption?: string
    }>
    total_count: number
  }> {
    const id = encodeURIComponent(String(mediaId))
    const minSize = options?.minSize ?? 50
    return await bgRequest<{
      media_id: number
      has_figures: boolean
      figures: Array<{
        id: string
        page: number
        width: number
        height: number
        format: string
        data_url?: string
        caption?: string
      }>
      total_count: number
    }>({
      path: `/api/v1/media/${id}/figures?min_size=${minSize}`,
      method: "GET",
      abortSignal: options?.signal
    })
  }

  async getDocumentReferences(
    mediaId: string | number,
    options?: {
      enrich?: boolean
      referenceIndex?: number
      offset?: number
      limit?: number
      parseCap?: number
      search?: string
      signal?: AbortSignal
    }
  ): Promise<{
    media_id: number
    has_references: boolean
    references: Array<{
      raw_text: string
      title?: string
      authors?: string
      year?: number
      venue?: string
      doi?: string
      arxiv_id?: string
      url?: string
      citation_count?: number
      semantic_scholar_id?: string
      open_access_pdf?: string
    }>
    enrichment_source?: string
    enriched_count?: number
    enrichment_limited?: boolean
    total_detected?: number
    truncated?: boolean
    offset?: number
    limit?: number
    returned_count?: number
    total_available?: number
    has_more?: boolean
    next_offset?: number | null
  }> {
    const id = encodeURIComponent(String(mediaId))
    const enrich = options?.enrich !== false
    const referenceIndex =
      typeof options?.referenceIndex === "number"
        ? `&reference_index=${options.referenceIndex}`
        : ""
    const offset =
      typeof options?.offset === "number" ? `&offset=${Math.max(0, options.offset)}` : ""
    const limit =
      typeof options?.limit === "number" ? `&limit=${Math.max(1, options.limit)}` : ""
    const parseCap =
      typeof options?.parseCap === "number" ? `&parse_cap=${Math.max(1, options.parseCap)}` : ""
    const search =
      typeof options?.search === "string" && options.search.trim().length > 0
        ? `&search=${encodeURIComponent(options.search.trim())}`
        : ""
    return await bgRequest<{
      media_id: number
      has_references: boolean
      references: Array<{
        raw_text: string
        title?: string
        authors?: string
        year?: number
        venue?: string
        doi?: string
        arxiv_id?: string
        url?: string
        citation_count?: number
        semantic_scholar_id?: string
        open_access_pdf?: string
      }>
      enrichment_source?: string
      enriched_count?: number
      enrichment_limited?: boolean
      total_detected?: number
      truncated?: boolean
      offset?: number
      limit?: number
      returned_count?: number
      total_available?: number
      has_more?: boolean
      next_offset?: number | null
    }>({
      path: `/api/v1/media/${id}/references?enrich=${enrich}${referenceIndex}${offset}${limit}${parseCap}${search}`,
      method: "GET",
      abortSignal: options?.signal,
      timeoutMs: 45000
    })
  }

  // Translation Methods
  async translate(
    text: string,
    targetLanguage: string = "English",
    options?: { model?: string; provider?: string }
  ): Promise<{
    translated_text: string
    target_language: string
    model_used: string
    detected_source_language?: string
  }> {
    return await bgRequest<{
      translated_text: string
      target_language: string
      model_used: string
      detected_source_language?: string
    }>({
      path: "/api/v1/translate",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        text,
        target_language: targetLanguage,
        ...(options?.model && { model: options.model }),
        ...(options?.provider && { provider: options.provider })
      }
    })
  }

  // Document Annotations Methods
  async listAnnotations(
    mediaId: string | number,
    options?: { signal?: AbortSignal }
  ): Promise<{
    media_id: number
    annotations: Array<{
      id: string
      media_id: number
      location: string
      text: string
      color: "yellow" | "green" | "blue" | "pink"
      note?: string
      annotation_type: "highlight" | "page_note"
      created_at: string
      updated_at: string
    }>
    total_count: number
  }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{
      media_id: number
      annotations: Array<{
        id: string
        media_id: number
        location: string
        text: string
        color: "yellow" | "green" | "blue" | "pink"
        note?: string
        annotation_type: "highlight" | "page_note"
        created_at: string
        updated_at: string
      }>
      total_count: number
    }>({
      path: `/api/v1/media/${id}/annotations`,
      method: "GET",
      abortSignal: options?.signal
    })
  }

  async createAnnotation(
    mediaId: string | number,
    annotation: {
      location: string
      text: string
      color?: "yellow" | "green" | "blue" | "pink"
      note?: string
      annotation_type?: "highlight" | "page_note"
    },
    options?: { signal?: AbortSignal }
  ): Promise<{
    id: string
    media_id: number
    location: string
    text: string
    color: "yellow" | "green" | "blue" | "pink"
    note?: string
    annotation_type: "highlight" | "page_note"
    created_at: string
    updated_at: string
  }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{
      id: string
      media_id: number
      location: string
      text: string
      color: "yellow" | "green" | "blue" | "pink"
      note?: string
      annotation_type: "highlight" | "page_note"
      created_at: string
      updated_at: string
    }>({
      path: `/api/v1/media/${id}/annotations`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: annotation,
      abortSignal: options?.signal
    })
  }

  async updateAnnotation(
    mediaId: string | number,
    annotationId: string,
    updates: {
      text?: string
      color?: "yellow" | "green" | "blue" | "pink"
      note?: string
    },
    options?: { signal?: AbortSignal }
  ): Promise<{
    id: string
    media_id: number
    location: string
    text: string
    color: "yellow" | "green" | "blue" | "pink"
    note?: string
    annotation_type: "highlight" | "page_note"
    created_at: string
    updated_at: string
  }> {
    const id = encodeURIComponent(String(mediaId))
    const annId = encodeURIComponent(annotationId)
    return await bgRequest<{
      id: string
      media_id: number
      location: string
      text: string
      color: "yellow" | "green" | "blue" | "pink"
      note?: string
      annotation_type: "highlight" | "page_note"
      created_at: string
      updated_at: string
    }>({
      path: `/api/v1/media/${id}/annotations/${annId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: updates,
      abortSignal: options?.signal
    })
  }

  async deleteAnnotation(
    mediaId: string | number,
    annotationId: string,
    options?: { signal?: AbortSignal }
  ): Promise<void> {
    const id = encodeURIComponent(String(mediaId))
    const annId = encodeURIComponent(annotationId)
    await bgRequest<void>({
      path: `/api/v1/media/${id}/annotations/${annId}`,
      method: "DELETE",
      abortSignal: options?.signal
    })
  }

  async syncAnnotations(
    mediaId: string | number,
    annotations: Array<{
      location: string
      text: string
      color?: "yellow" | "green" | "blue" | "pink"
      note?: string
      annotation_type?: "highlight" | "page_note"
    }>,
    clientIds?: string[],
    options?: { signal?: AbortSignal }
  ): Promise<{
    media_id: number
    synced_count: number
    annotations: Array<{
      id: string
      media_id: number
      location: string
      text: string
      color: "yellow" | "green" | "blue" | "pink"
      note?: string
      annotation_type: "highlight" | "page_note"
      created_at: string
      updated_at: string
    }>
    id_mapping?: Record<string, string>
  }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{
      media_id: number
      synced_count: number
      annotations: Array<{
        id: string
        media_id: number
        location: string
        text: string
        color: "yellow" | "green" | "blue" | "pink"
        note?: string
        annotation_type: "highlight" | "page_note"
        created_at: string
        updated_at: string
      }>
      id_mapping?: Record<string, string>
    }>({
      path: `/api/v1/media/${id}/annotations/sync`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        annotations,
        ...(clientIds && { client_ids: clientIds })
      },
      abortSignal: options?.signal
    })
  }

  // Reading Progress Methods
  async getReadingProgress(
    mediaId: string | number,
    options?: { signal?: AbortSignal }
  ): Promise<{
    media_id: number
    has_progress?: boolean
    current_page?: number
    total_pages?: number
    zoom_level?: number
    view_mode?: "single" | "continuous" | "thumbnails"
    percent_complete?: number
    cfi?: string
    last_read_at?: string
  }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{
      media_id: number
      has_progress?: boolean
      current_page?: number
      total_pages?: number
      zoom_level?: number
      view_mode?: "single" | "continuous" | "thumbnails"
      percent_complete?: number
      cfi?: string
      last_read_at?: string
    }>({
      path: `/api/v1/media/${id}/progress`,
      method: "GET",
      abortSignal: options?.signal
    })
  }

  async updateReadingProgress(
    mediaId: string | number,
    progress: {
      current_page: number
      total_pages: number
      zoom_level?: number
      view_mode?: "single" | "continuous" | "thumbnails"
      cfi?: string
      percentage?: number
    },
    options?: { signal?: AbortSignal }
  ): Promise<{
    media_id: number
    current_page: number
    total_pages: number
    zoom_level: number
    view_mode: "single" | "continuous" | "thumbnails"
    percent_complete: number
    cfi?: string
    last_read_at: string
  }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{
      media_id: number
      current_page: number
      total_pages: number
      zoom_level: number
      view_mode: "single" | "continuous" | "thumbnails"
      percent_complete: number
      cfi?: string
      last_read_at: string
    }>({
      path: `/api/v1/media/${id}/progress`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: progress,
      abortSignal: options?.signal
    })
  }

  async deleteReadingProgress(
    mediaId: string | number,
    options?: { signal?: AbortSignal }
  ): Promise<void> {
    const id = encodeURIComponent(String(mediaId))
    await bgRequest<void>({
      path: `/api/v1/media/${id}/progress`,
      method: "DELETE",
      abortSignal: options?.signal
    })
  }

  // Notes Methods
  async createNote(content: string, metadata?: any): Promise<any> {
    return await bgRequest<any>({ path: '/api/v1/notes/', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: { content, ...metadata } })
  }

  async listNotes(
    params?: {
      page?: number
      results_per_page?: number
      limit?: number
      offset?: number
      include_keywords?: boolean
    },
    options?: { signal?: AbortSignal }
  ): Promise<any> {
    const limit = params?.limit ?? params?.results_per_page
    const offset = params?.offset ?? (
      params?.page != null && limit != null
        ? Math.max(0, (params.page - 1) * limit)
        : undefined
    )
    const query = this.buildQuery({
      limit,
      offset,
      include_keywords: params?.include_keywords
    } as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/notes/${query}`,
      method: "GET",
      abortSignal: options?.signal
    })
  }

  async searchNotes(query: string): Promise<any> {
    const normalized = query.trim()
    if (!normalized) {
      return await this.listNotes()
    }
    const queryString = this.buildQuery({
      query: normalized
    })
    return await bgRequest<any>({
      path: `/api/v1/notes/search/${queryString}`,
      method: "GET"
    })
  }
  // Prompts Methods
  async getPrompts(): Promise<any> {
    const path = await this.resolveApiPath("prompts.list", [
      "/api/v1/prompts",
      "/api/v1/prompts/"
    ])
    return await bgRequest<any>({ path, method: 'GET' })
  }

  async searchPrompts(query: string): Promise<any> {
    // TODO: confirm trailing slash per OpenAPI (`/api/v1/prompts/search` exists without slash)
    return await bgRequest<any>({ path: '/api/v1/prompts/search', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: { query } })
  }

  async createPrompt(payload: PromptPayload): Promise<any> {
    const name = payload.name || payload.title || 'Untitled'
    const system_prompt = payload.system_prompt ?? (payload.is_system ? payload.content : undefined)
    const user_prompt = payload.user_prompt ?? (!payload.is_system ? payload.content : undefined)
    const keywords = payload.keywords
    const normalized: Record<string, any> = {
      name,
      author: payload.author,
      details: payload.details,
      system_prompt,
      user_prompt,
      keywords
    }

    Object.keys(normalized).forEach((key) => {
      if (typeof normalized[key] === 'undefined') delete normalized[key]
    })

    const path = await this.resolveApiPath("prompts.create", [
      "/api/v1/prompts",
      "/api/v1/prompts/"
    ])
    return await bgRequest<any>({
      path,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: normalized
    })
  }

  async updatePrompt(id: string | number, payload: PromptPayload): Promise<any> {
    const pid = String(id)
    const name = payload.name || payload.title || 'Untitled'
    const system_prompt = payload.system_prompt ?? (payload.is_system ? payload.content : undefined)
    const user_prompt = payload.user_prompt ?? (!payload.is_system ? payload.content : undefined)
    const keywords = payload.keywords

    const normalized: Record<string, any> = {
      name,
      author: payload.author,
      details: payload.details,
      system_prompt,
      user_prompt,
      keywords
    }

    Object.keys(normalized).forEach((key) => {
      if (typeof normalized[key] === 'undefined') delete normalized[key]
    })

    // Path per OpenAPI: /api/v1/prompts/{prompt_identifier}
    return await bgRequest<any>({ path: `/api/v1/prompts/${pid}`, method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: normalized })
  }

  // Characters API
  private normalizeCharacterListResponse(payload: unknown): any[] {
    if (Array.isArray(payload)) {
      return payload
    }
    if (!payload || typeof payload !== "object") {
      return []
    }

    const objectPayload = payload as Record<string, unknown>
    const candidateLists = [
      objectPayload.items,
      objectPayload.characters,
      objectPayload.results,
      objectPayload.data
    ]

    for (const candidate of candidateLists) {
      if (Array.isArray(candidate)) {
        return candidate
      }
    }

    return []
  }

  async listCharacters(params?: Record<string, any>): Promise<any[]> {
    const query = this.buildQuery(params)
    const listPathCandidates = ["/api/v1/characters", "/api/v1/characters/"] as const
    const base = await this.resolveApiPath("characters.list", [...listPathCandidates])
    const requestList = async (path: string) =>
      this.normalizeCharacterListResponse(
        await bgRequest<any>({
          path: appendPathQuery(path as AllowedPath, query),
          method: "GET"
        })
      )

    try {
      return await requestList(base)
    } catch (error) {
      const candidate = error as
        | {
            status?: unknown
            response?: { status?: unknown }
            message?: unknown
            details?: unknown
          }
        | null
        | undefined
      const rawStatus = candidate?.status ?? candidate?.response?.status
      const statusCodeFromNumberLike =
        typeof rawStatus === "number"
          ? rawStatus
          : typeof rawStatus === "string"
            ? Number(rawStatus)
            : Number.NaN
      const statusCodeFromMessage = String(candidate?.message || "").match(
        /\b(301|302|307|308|404|405|422)\b/
      )
      const statusCode = Number.isFinite(statusCodeFromNumberLike)
        ? statusCodeFromNumberLike
        : statusCodeFromMessage
          ? Number(statusCodeFromMessage[1])
          : Number.NaN
      const normalizedMessage = String(candidate?.message || "").toLowerCase()
      const normalizedDetails = (() => {
        const details = candidate?.details
        if (typeof details === "string") return details.toLowerCase()
        if (details == null) return ""
        try {
          return JSON.stringify(details).toLowerCase()
        } catch {
          return String(details).toLowerCase()
        }
      })()
      const shouldTryAlternatePath =
        statusCode === 301 ||
        statusCode === 302 ||
        statusCode === 307 ||
        statusCode === 308 ||
        statusCode === 404 ||
        statusCode === 405 ||
        statusCode === 422 ||
        normalizedMessage.includes("path.character_id") ||
        normalizedMessage.includes("unable to parse string as an integer") ||
        normalizedMessage.includes('input":"query"') ||
        normalizedMessage.includes("/api/v1/characters/query") ||
        normalizedDetails.includes("path.character_id") ||
        normalizedDetails.includes("unable to parse string as an integer") ||
        normalizedDetails.includes('input":"query"') ||
        normalizedDetails.includes("/api/v1/characters/query")

      if (!shouldTryAlternatePath) {
        throw error
      }

      const alternatePath = listPathCandidates.find((path) => path !== base)
      if (!alternatePath) {
        throw error
      }

      try {
        return await requestList(alternatePath)
      } catch {
        throw error
      }
    }
  }

  async listCharactersPage(
    params?: CharacterListQueryParams
  ): Promise<CharacterListQueryResponse> {
    const query = this.buildQuery(params as Record<string, any> | undefined)
    const base = await this.resolveApiPath("characters.query", [
      "/api/v1/characters/query",
      "/api/v1/characters/query/"
    ])
    const requestedPage =
      typeof params?.page === "number" && Number.isFinite(params.page)
        ? Math.max(1, Math.floor(params.page))
        : 1
    const requestedPageSize =
      typeof params?.page_size === "number" && Number.isFinite(params.page_size)
        ? Math.max(1, Math.floor(params.page_size))
        : 25

    const buildLegacyListFallback = async (): Promise<CharacterListQueryResponse> => {
      const offset = (requestedPage - 1) * requestedPageSize
      const legacyResponse = await this.listCharacters({
        limit: requestedPageSize,
        offset,
        query: params?.query,
        tags: params?.tags,
        match_all_tags: params?.match_all_tags,
        creator: params?.creator,
        has_conversations: params?.has_conversations,
        favorite_only: params?.favorite_only,
        include_deleted: params?.include_deleted,
        deleted_only: params?.deleted_only,
        sort_by: params?.sort_by,
        sort_order: params?.sort_order,
        include_image_base64: params?.include_image_base64
      })
      const legacyCandidate = legacyResponse as
        | {
            items?: unknown
            total?: unknown
            has_more?: unknown
          }
        | null
        | undefined
      const legacyItems = Array.isArray(legacyCandidate?.items)
        ? legacyCandidate.items
        : Array.isArray(legacyResponse)
          ? legacyResponse
          : []
      const legacyHasMore =
        typeof legacyCandidate?.has_more === "boolean"
          ? legacyCandidate.has_more
          : legacyItems.length >= requestedPageSize
      const legacyTotal =
        typeof legacyCandidate?.total === "number" &&
        Number.isFinite(legacyCandidate.total)
          ? legacyCandidate.total
          : legacyHasMore
            ? offset + legacyItems.length + 1
            : offset + legacyItems.length

      return {
        items: legacyItems,
        total: legacyTotal,
        page: requestedPage,
        page_size: requestedPageSize,
        has_more: legacyHasMore
      }
    }

    const isQueryRouteConflict = (error: unknown): boolean => {
      const candidate = error as
        | {
            status?: unknown
            response?: { status?: unknown }
            message?: unknown
            details?: unknown
          }
        | null
        | undefined
      const rawStatus = candidate?.status ?? candidate?.response?.status
      const statusCodeFromNumberLike =
        typeof rawStatus === "number"
          ? rawStatus
          : typeof rawStatus === "string"
            ? Number(rawStatus)
            : Number.NaN
      const statusCodeFromMessage = String(candidate?.message || "").match(
        /\b(404|405|422)\b/
      )
      const statusCode = Number.isFinite(statusCodeFromNumberLike)
        ? statusCodeFromNumberLike
        : statusCodeFromMessage
          ? Number(statusCodeFromMessage[1])
          : Number.NaN
      const normalizedMessage = String(candidate?.message || "").toLowerCase()
      const normalizedDetails = (() => {
        const details = candidate?.details
        if (typeof details === "string") return details.toLowerCase()
        if (details == null) return ""
        try {
          return JSON.stringify(details).toLowerCase()
        } catch {
          return String(details).toLowerCase()
        }
      })()
      return (
        statusCode === 404 ||
        statusCode === 405 ||
        statusCode === 422 ||
        normalizedMessage.includes("path.character_id") ||
        normalizedMessage.includes("unable to parse string as an integer") ||
        normalizedMessage.includes('input":"query"') ||
        normalizedMessage.includes("/api/v1/characters/query") ||
        normalizedDetails.includes("path.character_id") ||
        normalizedDetails.includes("unable to parse string as an integer") ||
        normalizedDetails.includes('input":"query"') ||
        normalizedDetails.includes("/api/v1/characters/query")
      )
    }

    let response: any
    try {
      response = await bgRequest<any>({
        path: appendPathQuery(base, query),
        method: "GET"
      })
    } catch (error) {
      if (!isQueryRouteConflict(error)) {
        throw error
      }
      return await buildLegacyListFallback()
    }

    if (Array.isArray(response)) {
      return {
        items: response,
        total: response.length,
        page: Number(params?.page || 1),
        page_size: Number(params?.page_size || response.length || 25),
        has_more: false
      }
    }

    const responseLooksLikeRouteConflict =
      response &&
      typeof response === "object" &&
      !Array.isArray(response) &&
      !Array.isArray((response as any).items) &&
      (() => {
        try {
          const payload = JSON.stringify(response).toLowerCase()
          return (
            payload.includes("path.character_id") ||
            payload.includes("unable to parse string as an integer") ||
            payload.includes('input":"query"') ||
            payload.includes("/api/v1/characters/query")
          )
        } catch {
          return false
        }
      })()

    if (responseLooksLikeRouteConflict) {
      return await buildLegacyListFallback()
    }

    const items = Array.isArray(response?.items) ? response.items : []
    const total =
      typeof response?.total === "number" && Number.isFinite(response.total)
        ? response.total
        : items.length
    const page =
      typeof response?.page === "number" && Number.isFinite(response.page)
        ? response.page
        : Number(params?.page || 1)
    const pageSize =
      typeof response?.page_size === "number" &&
      Number.isFinite(response.page_size)
        ? response.page_size
        : Number(params?.page_size || 25)

    return {
      items,
      total,
      page,
      page_size: pageSize,
      has_more: Boolean(response?.has_more)
    }
  }

  private getCharacterListIdentity(character: any, fallbackIndex: number): string {
    const id = character?.id ?? character?.character_id ?? character?.characterId
    if (id !== undefined && id !== null && String(id).trim().length > 0) {
      return `id:${String(id)}`
    }

    const slug = character?.slug
    if (typeof slug === "string" && slug.trim().length > 0) {
      return `slug:${slug}`
    }

    const name = character?.name ?? character?.title
    if (typeof name === "string" && name.trim().length > 0) {
      return `name:${name}`
    }

    return `idx:${fallbackIndex}`
  }

  async listAllCharacters(options?: {
    pageSize?: number
    maxPages?: number
  }): Promise<any[]> {
    const requestedPageSize =
      typeof options?.pageSize === "number" && Number.isFinite(options.pageSize)
        ? Math.floor(options.pageSize)
        : 1000
    const requestedMaxPages =
      typeof options?.maxPages === "number" && Number.isFinite(options.maxPages)
        ? Math.floor(options.maxPages)
        : 20
    const pageSize = Math.min(1000, Math.max(1, requestedPageSize))
    const maxPages = Math.min(200, Math.max(1, requestedMaxPages))

    const characters: any[] = []
    const seen = new Set<string>()

    for (let pageIndex = 0; pageIndex < maxPages; pageIndex += 1) {
      const offset = pageIndex * pageSize
      const page = await this.listCharacters({ limit: pageSize, offset })
      const pageList = Array.isArray(page) ? page : []
      if (pageList.length === 0) {
        break
      }

      let addedFromPage = 0
      for (const character of pageList) {
        const identity = this.getCharacterListIdentity(
          character,
          characters.length + addedFromPage
        )
        if (seen.has(identity)) continue
        seen.add(identity)
        characters.push(character)
        addedFromPage += 1
      }

      // Stop if we reached the final partial page, or the backend ignored offset
      // and returned only already-seen entries.
      if (pageList.length < pageSize || addedFromPage === 0) {
        break
      }
    }

    return characters
  }

   async searchCharacters(query: string, params?: Record<string, any>): Promise<any[]> {
    const qp = this.buildQuery({ query, ...(params || {}) })
    const base = await this.resolveApiPath("characters.search", [
      "/api/v1/characters/search",
      "/api/v1/characters/search/"
    ])
    return await bgRequest<any[]>({
      path: appendPathQuery(base, qp),
      method: 'GET'
    })
  }

  async filterCharactersByTags(
    tags: string[],
    options?: { match_all?: boolean; limit?: number; offset?: number }
  ): Promise<any[]> {
    const qp = this.buildQuery({
      tags,
      ...(options || {})
    })
    const base = await this.resolveApiPath("characters.filter", [
      "/api/v1/characters/filter",
      "/api/v1/characters/filter/"
    ])
    return await bgRequest<any[]>({
      path: appendPathQuery(base, qp),
      method: 'GET'
    })
  }

  async getCharacter(id: string | number, options?: { forceRefresh?: boolean }): Promise<any> {
    const cid = String(id)
    const forceRefresh = options?.forceRefresh === true
    if (!forceRefresh) {
      const cached = this.characterCache.get(cid)
      if (cached && cached.expiresAt > Date.now()) {
        return cached.value
      }
    }
    const inFlight = this.characterInFlight.get(cid)
    if (inFlight) return inFlight

    const request = (async () => {
      try {
        const template = await this.resolveApiPath("characters.get", [
          "/api/v1/characters/{id}",
          "/api/v1/characters/{id}/"
        ])
        const path = this.fillPathParams(template, cid)
        const value = await bgRequest<any>({
          path,
          method: 'GET'
        })
        this.characterCache.set(cid, {
          value,
          expiresAt: Date.now() + CHARACTER_CACHE_TTL_MS
        })
        return value
      } finally {
        this.characterInFlight.delete(cid)
      }
    })()

    this.characterInFlight.set(cid, request)
    return request
  }

  async listCharacterVersions(
    id: string | number,
    options?: { limit?: number }
  ): Promise<CharacterVersionListResponse> {
    const cid = String(id)
    const query = this.buildQuery({
      limit: options?.limit ?? 50
    })
    const template = await this.resolveApiPath("characters.versions", [
      "/api/v1/characters/{id}/versions",
      "/api/v1/characters/{id}/versions/"
    ])
    const path = appendPathQuery(this.fillPathParams(template, cid), query)
    const response = await bgRequest<any>({
      path,
      method: "GET"
    })

    const items = Array.isArray(response?.items)
      ? response.items
      : Array.isArray(response)
        ? response
        : []
    return {
      items: items.map((item: any) => ({
        change_id:
          typeof item?.change_id === "number" && Number.isFinite(item.change_id)
            ? item.change_id
            : Number(item?.change_id || 0),
        version:
          typeof item?.version === "number" && Number.isFinite(item.version)
            ? item.version
            : Number(item?.version || 0),
        operation: String(item?.operation || "update"),
        timestamp: item?.timestamp ?? null,
        client_id: item?.client_id ?? null,
        payload:
          item?.payload && typeof item.payload === "object" && !Array.isArray(item.payload)
            ? item.payload
            : {}
      })),
      total:
        typeof response?.total === "number" && Number.isFinite(response.total)
          ? response.total
          : items.length
    }
  }

  async diffCharacterVersions(
    id: string | number,
    fromVersion: number,
    toVersion: number
  ): Promise<CharacterVersionDiffResponse> {
    const cid = String(id)
    const query = this.buildQuery({
      from_version: fromVersion,
      to_version: toVersion
    })
    const template = await this.resolveApiPath("characters.versionDiff", [
      "/api/v1/characters/{id}/versions/diff",
      "/api/v1/characters/{id}/versions/diff/"
    ])
    const path = appendPathQuery(this.fillPathParams(template, cid), query)
    const response = await bgRequest<any>({
      path,
      method: "GET"
    })

    const normalizeVersionEntry = (entry: any): CharacterVersionEntry => ({
      change_id:
        typeof entry?.change_id === "number" && Number.isFinite(entry.change_id)
          ? entry.change_id
          : Number(entry?.change_id || 0),
      version:
        typeof entry?.version === "number" && Number.isFinite(entry.version)
          ? entry.version
          : Number(entry?.version || 0),
      operation: String(entry?.operation || "update"),
      timestamp: entry?.timestamp ?? null,
      client_id: entry?.client_id ?? null,
      payload:
        entry?.payload && typeof entry.payload === "object" && !Array.isArray(entry.payload)
          ? entry.payload
          : {}
    })

    const changedFields = Array.isArray(response?.changed_fields)
      ? response.changed_fields
      : []

    return {
      character_id:
        typeof response?.character_id === "number" && Number.isFinite(response.character_id)
          ? response.character_id
          : Number(response?.character_id || 0),
      from_entry: normalizeVersionEntry(response?.from_entry),
      to_entry: normalizeVersionEntry(response?.to_entry),
      changed_fields: changedFields.map((field: any) => ({
        field: String(field?.field || ""),
        old_value: field?.old_value,
        new_value: field?.new_value
      })),
      changed_count:
        typeof response?.changed_count === "number" && Number.isFinite(response.changed_count)
          ? response.changed_count
          : changedFields.length
    }
  }

  async revertCharacter(
    id: string | number,
    targetVersion: number
  ): Promise<any> {
    const cid = String(id)
    const template = await this.resolveApiPath("characters.revert", [
      "/api/v1/characters/{id}/revert",
      "/api/v1/characters/{id}/revert/"
    ])
    const path = this.fillPathParams(template, cid)
    const response = await bgRequest<any>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        target_version: targetVersion
      }
    })
    this.characterCache.delete(cid)
    return response
  }

  async createCharacter(payload: Record<string, any>): Promise<any> {
    const pathCandidates = [
      "/api/v1/characters/",
      "/api/v1/characters"
    ] as const
    const path = await this.resolveApiPath("characters.create", [...pathCandidates])
    await this.ensureConfigForRequest(true)

    const requestCreate = async (requestPath: string) =>
      await bgRequest<any>({
        path: requestPath as AllowedPath,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload
      })

    const requestCreateDirect = async (
      requestPath: string
    ): Promise<any> => {
      const storage = createSafeStorage()
      const response = await tldwRequest(
        {
          path: requestPath as AllowedPath,
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload
        },
        {
          getConfig: () =>
            storage.get<TldwConfig>("tldwConfig").catch(() => null)
        }
      )
      if (response?.ok) {
        return response.data
      }

      const error = new Error(
        typeof response?.error === "string" && response.error.trim().length > 0
          ? response.error
          : `Request failed: ${response?.status ?? 0}`
      ) as Error & {
        status?: number
        details?: unknown
      }
      error.status = response?.status
      if (typeof response?.data !== "undefined") {
        error.details = response.data
      }
      throw error
    }

    const readErrorText = (error: unknown): string => {
      const candidate = error as
        | {
            message?: unknown
            details?: unknown
          }
        | null
        | undefined
      const message = String(candidate?.message || "")
      const details = (() => {
        const value = candidate?.details
        if (typeof value === "string") return value
        if (value == null) return ""
        try {
          return JSON.stringify(value)
        } catch {
          return String(value)
        }
      })()
      return `${message} ${details}`.toLowerCase()
    }

    const getErrorStatusCode = (error: unknown): number | null => {
      const candidate = error as
        | {
            status?: unknown
            response?: { status?: unknown }
            message?: unknown
            details?: unknown
          }
        | null
        | undefined
      const rawStatus = candidate?.status ?? candidate?.response?.status
      const statusCodeFromNumberLike =
        typeof rawStatus === "number"
          ? rawStatus
          : typeof rawStatus === "string"
            ? Number(rawStatus)
            : Number.NaN
      if (Number.isFinite(statusCodeFromNumberLike)) {
        return statusCodeFromNumberLike
      }
      const statusFromText = readErrorText(error).match(
        /\b(301|302|307|308|404|405|422)\b/
      )
      if (!statusFromText) return null
      const parsedStatus = Number(statusFromText[1])
      return Number.isFinite(parsedStatus) ? parsedStatus : null
    }

    const isExtensionTimeoutError = (error: unknown): boolean => {
      return Boolean(
        (error as { __tldwExtensionTimeout?: boolean } | null)
          ?.__tldwExtensionTimeout
      ) || readErrorText(error).includes("extension messaging timeout")
    }

    const shouldTryAlternatePath = (error: unknown): boolean => {
      const statusCode = getErrorStatusCode(error)
      if (
        statusCode === 301 ||
        statusCode === 302 ||
        statusCode === 307 ||
        statusCode === 308 ||
        statusCode === 404 ||
        statusCode === 405 ||
        statusCode === 422
      ) {
        return true
      }
      const normalizedText = readErrorText(error)
      return (
        normalizedText.includes("path.character_id") ||
        normalizedText.includes("unable to parse string as an integer") ||
        normalizedText.includes("/api/v1/characters/query")
      )
    }

    const runCreateWithTimeoutRetry = async (
      requestPath: string
    ): Promise<any> => {
      try {
        return await requestCreate(requestPath)
      } catch (error) {
        if (!isExtensionTimeoutError(error)) {
          throw error
        }
        try {
          return await requestCreate(requestPath)
        } catch (retryError) {
          if (!isExtensionTimeoutError(retryError)) {
            throw retryError
          }
          return await requestCreateDirect(requestPath)
        }
      }
    }

    try {
      return await runCreateWithTimeoutRetry(path)
    } catch (error) {
      if (!shouldTryAlternatePath(error)) {
        throw error
      }

      const alternatePath = pathCandidates.find(
        (candidate) => candidate !== path
      )
      if (!alternatePath) {
        throw error
      }

      return await runCreateWithTimeoutRetry(alternatePath)
    }
  }

  async importCharacterFile(
    file: File,
    options?: { allowImageOnly?: boolean }
  ): Promise<any> {
    const data = await file.arrayBuffer()
    const name = file.name || "character-card"
    const type = file.type || "application/octet-stream"
    const path = await this.resolveApiPath("characters.import", [
      "/api/v1/characters/import",
      "/api/v1/characters/import/"
    ])
    const fields = options?.allowImageOnly
      ? { allow_image_only: true }
      : undefined
    return await this.upload<any>({
      path,
      method: "POST",
      fileFieldName: "character_file",
      file: { name, type, data },
      fields
    })
  }

  async exportCharacter(
    id: string | number,
    options?: { format?: 'v3' | 'v2' | 'json'; includeWorldBooks?: boolean }
  ): Promise<any> {
    const cid = String(id)
    const params = new URLSearchParams()
    if (options?.format) {
      params.set('format', options.format)
    }
    if (options?.includeWorldBooks) {
      params.set('include_world_books', 'true')
    }
    const qp = params.toString() ? `?${params.toString()}` : ''
    const template = await this.resolveApiPath("characters.export", [
      "/api/v1/characters/{id}/export",
      "/api/v1/characters/{id}/export/"
    ])
    const path = appendPathQuery(this.fillPathParams(template, cid), qp)
    return await bgRequest<any>({
      path,
      method: 'GET'
    })
  }

  async updateCharacter(id: string | number, payload: Record<string, any>, expectedVersion?: number): Promise<any> {
    const cid = String(id)
    const qp = expectedVersion != null ? `?expected_version=${encodeURIComponent(String(expectedVersion))}` : ''
    const template = await this.resolveApiPath("characters.update", [
      "/api/v1/characters/{id}",
      "/api/v1/characters/{id}/"
    ])
    const path = appendPathQuery(this.fillPathParams(template, cid), qp)
    const res = await bgRequest<any>({
      path,
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: payload
    })
    this.characterCache.delete(cid)
    return res
  }

  async deleteCharacter(id: string | number, expectedVersion?: number): Promise<void> {
    const cid = String(id)
    let resolvedVersion = Number(expectedVersion)
    if (!Number.isInteger(resolvedVersion) || resolvedVersion < 0) {
      const character = await this.getCharacter(cid, { forceRefresh: true })
      const fetchedVersion = Number(character?.version)
      if (!Number.isInteger(fetchedVersion) || fetchedVersion < 0) {
        throw new Error("Character delete failed: missing expected version")
      }
      resolvedVersion = fetchedVersion
    }
    const template = await this.resolveApiPath("characters.delete", [
      "/api/v1/characters/{id}",
      "/api/v1/characters/{id}/"
    ])
    const path = appendPathQuery(
      this.fillPathParams(template, cid),
      `?expected_version=${encodeURIComponent(String(resolvedVersion))}`
    )
    await bgRequest<void>({ path, method: 'DELETE' })
    this.characterCache.delete(cid)
  }

  async restoreCharacter(id: string | number, expectedVersion: number): Promise<any> {
    const cid = String(id)
    const template = await this.resolveApiPath("characters.restore", [
      "/api/v1/characters/{id}/restore",
      "/api/v1/characters/{id}/restore/"
    ])
    const path = appendPathQuery(
      this.fillPathParams(template, cid),
      `?expected_version=${expectedVersion}`
    )
    const res = await bgRequest<any>({ path, method: 'POST' })
    this.characterCache.delete(cid)
    return res
  }

  // Character chat sessions
  async listCharacterChatSessions(): Promise<any[]> {
    const path = await this.resolveApiPath("characterChatSessions.list", [
      "/api/v1/character-chat/sessions",
      "/api/v1/character_chat_sessions",
      "/api/v1/character_chat_sessions/"
    ])
    return await bgRequest<any[]>({ path, method: 'GET' })
  }

  async createCharacterChatSession(character_id: string): Promise<any> {
    const body = { character_id }
    const path = await this.resolveApiPath("characterChatSessions.create", [
      "/api/v1/character-chat/sessions",
      "/api/v1/character_chat_sessions",
      "/api/v1/character_chat_sessions/"
    ])
    return await bgRequest<any>({
      path,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    })
  }

  async deleteCharacterChatSession(session_id: string | number): Promise<void> {
    const sid = String(session_id)
    const template = await this.resolveApiPath("characterChatSessions.delete", [
      "/api/v1/character-chat/sessions/{session_id}",
      "/api/v1/character_chat_sessions/{session_id}",
      "/api/v1/character_chat_sessions/{session_id}/"
    ])
    const path = this.fillPathParams(template, sid)
    await bgRequest<void>({ path, method: 'DELETE' })
  }

  // Character messages
  async listCharacterMessages(session_id: string | number): Promise<any[]> {
    const sid = String(session_id)
    const query = this.buildQuery({ session_id: sid })
    const template = await this.resolveApiPath("characterChatMessages.list", [
      "/api/v1/character-chat/sessions/{session_id}/messages",
      "/api/v1/character-messages",
      "/api/v1/character_messages"
    ])
    const path = template.includes("{")
      ? this.fillPathParams(template, sid)
      : appendPathQuery(template, query)
    return await bgRequest<any[]>({ path, method: 'GET' })
  }

  async sendCharacterMessage(session_id: string | number, content: string, options?: { extra?: Record<string, any> }): Promise<any> {
    const sid = String(session_id)
    const body = { content, session_id: sid, ...(options?.extra || {}) }
    const template = await this.resolveApiPath("characterChatMessages.send", [
      "/api/v1/character-chat/sessions/{session_id}/messages",
      "/api/v1/character_messages",
      "/api/v1/character-messages"
    ])
    const path = template.includes("{")
      ? this.fillPathParams(template, sid)
      : template
    return await bgRequest<any>({
      path,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    })
  }

  async * streamCharacterMessage(session_id: string | number, content: string, options?: { extra?: Record<string, any> }): AsyncGenerator<any> {
    const sid = String(session_id)
    const body = { content, session_id: sid, ...(options?.extra || {}) }
    const template = await this.resolveApiPath("characterChatMessages.stream", [
      "/api/v1/character-chat/sessions/{session_id}/messages/stream",
      "/api/v1/character_messages/stream",
      "/api/v1/character-messages/stream"
    ])
    const path = this.fillPathParams(template, sid)
    for await (const line of bgStream({ path, method: 'POST', headers: { 'Content-Type': 'application/json' }, body })) {
      try { yield JSON.parse(line) } catch {}
    }
  }

  private normalizeChatSummary(input: any): ServerChatSummary {
    const created_at = String(input?.created_at || input?.createdAt || "")
    const updated_at =
      input?.updated_at ??
      input?.updatedAt ??
      input?.last_modified ??
      input?.lastModified ??
      null
    const state = input?.state ?? input?.conversation_state ?? null
    const last_active =
      input?.last_active ??
      input?.lastActive ??
      updated_at ??
      created_at ??
      null
    const messageCountRaw = input?.message_count ?? input?.messageCount
    const message_count =
      typeof messageCountRaw === "number"
        ? messageCountRaw
        : typeof messageCountRaw === "string" && messageCountRaw.trim().length > 0
          ? Number.parseFloat(messageCountRaw)
          : null
    return {
      id: String(input?.id ?? ""),
      title: String(input?.title || ""),
      created_at,
      updated_at: updated_at ? String(updated_at) : null,
      last_active: last_active ? String(last_active) : null,
      message_count: Number.isFinite(message_count as number)
        ? (message_count as number)
        : null,
      source: input?.source ?? null,
      state: state ? String(state) : null,
      topic_label: input?.topic_label ?? input?.topicLabel ?? null,
      cluster_id: input?.cluster_id ?? input?.clusterId ?? null,
      external_ref: input?.external_ref ?? input?.externalRef ?? null,
      bm25_norm:
        typeof input?.bm25_norm === "number"
          ? input?.bm25_norm
          : typeof input?.relevance === "number"
            ? input?.relevance
            : null,
      character_id: input?.character_id ?? input?.characterId ?? null,
      parent_conversation_id:
        input?.parent_conversation_id ?? input?.parentConversationId ?? null,
      root_id: input?.root_id ?? input?.rootId ?? null,
      forked_from_message_id:
        input?.forked_from_message_id ?? input?.forkedFromMessageId ?? null,
      version:
        typeof input?.version === "number"
          ? input.version
          : typeof input?.expected_version === "number"
            ? input.expected_version
            : null
    }
  }

  // Chats API (resource-based)
  async listChatCommands(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/commands",
      method: "GET"
    })
  }

  async listChats(
    params?: Record<string, any>,
    options?: { signal?: AbortSignal }
  ): Promise<ServerChatSummary[]> {
    const query = this.buildQuery(params)
    const data = await bgRequest<any>({
      path: `/api/v1/chats/${query}`,
      method: "GET",
      abortSignal: options?.signal
    })

    let list: any[] = []

    if (Array.isArray(data)) {
      list = data
    } else if (data && typeof data === "object") {
      const obj: any = data
      if (Array.isArray(obj.chats)) {
        list = obj.chats
      } else if (Array.isArray(obj.items)) {
        list = obj.items
      } else if (Array.isArray(obj.results)) {
        list = obj.results
      } else if (Array.isArray(obj.data)) {
        list = obj.data
      }
    }

    return list.map((c) => this.normalizeChatSummary(c))
  }

  async listChatsWithMeta(
    params?: Record<string, any>,
    options?: { signal?: AbortSignal }
  ): Promise<{ chats: ServerChatSummary[]; total: number }> {
    const query = this.buildQuery(params)
    const data = await bgRequest<any>({
      path: `/api/v1/chats/${query}`,
      method: "GET",
      abortSignal: options?.signal
    })

    let list: any[] = []
    let total: number | null = null

    if (Array.isArray(data)) {
      list = data
    } else if (data && typeof data === "object") {
      const obj: any = data
      if (typeof obj.total === "number") {
        total = obj.total
      } else if (typeof obj.count === "number") {
        total = obj.count
      }
      if (Array.isArray(obj.chats)) {
        list = obj.chats
      } else if (Array.isArray(obj.items)) {
        list = obj.items
      } else if (Array.isArray(obj.results)) {
        list = obj.results
      } else if (Array.isArray(obj.data)) {
        list = obj.data
      }
    }

    const chats = list.map((c) => this.normalizeChatSummary(c))
    return {
      chats,
      total: typeof total === "number" ? total : chats.length
    }
  }

  async createChat(payload: Record<string, any>): Promise<ServerChatSummary> {
    const res = await bgRequest<any>({
      path: "/api/v1/chats/",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
    return this.normalizeChatSummary(res)
  }

  async completeCharacterChatTurn(
    chat_id: string | number,
    payload: Record<string, any>
  ): Promise<any> {
    const cid = String(chat_id)
    return await bgRequest<any>({
      path: `/api/v1/chats/${cid}/complete-v2`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async getChat(chat_id: string | number): Promise<ServerChatSummary> {
    const cid = String(chat_id)
    const res = await bgRequest<any>({
      path: `/api/v1/chats/${cid}`,
      method: "GET"
    })
    return this.normalizeChatSummary(res)
  }

  async listChatResearchRuns(
    chat_id: string | number
  ): Promise<ChatLinkedResearchRunsResponse> {
    const cid = String(chat_id)
    const response = await bgRequest<any>({
      path: `/api/v1/chats/${cid}/research-runs`,
      method: "GET"
    })
    const runs = Array.isArray(response?.runs) ? response.runs : []
    return {
      runs: runs.map((run: any) => ({
        run_id: String(run?.run_id ?? ""),
        query: String(run?.query ?? ""),
        status: String(run?.status ?? ""),
        phase: String(run?.phase ?? ""),
        control_state: String(run?.control_state ?? "running"),
        latest_checkpoint_id:
          typeof run?.latest_checkpoint_id === "string" && run.latest_checkpoint_id.trim()
            ? run.latest_checkpoint_id
            : null,
        updated_at: String(run?.updated_at ?? "")
      }))
    }
  }

  async createResearchRun(
    payload: ResearchRunCreateRequest
  ): Promise<ResearchRunResponse> {
    return await bgRequest<ResearchRunResponse>({
      path: "/api/v1/research/runs",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async getResearchBundle(
    run_id: string | number
  ): Promise<ResearchBundleResponse> {
    const rid = String(run_id)
    return await bgRequest<ResearchBundleResponse>({
      path: `/api/v1/research/runs/${encodeURIComponent(rid)}/bundle`,
      method: "GET"
    })
  }

  async getChatSettings(chat_id: string | number): Promise<ChatSettingsResponse> {
    const cid = String(chat_id)
    return await bgRequest<ChatSettingsResponse>({
      path: `/api/v1/chats/${cid}/settings`,
      method: "GET"
    })
  }

  async updateChatSettings(
    chat_id: string | number,
    settings: Record<string, unknown>
  ): Promise<ChatSettingsResponse> {
    const cid = String(chat_id)
    return await bgRequest<ChatSettingsResponse>({
      path: `/api/v1/chats/${cid}/settings`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: { settings }
    })
  }

  async getChatLorebookDiagnostics(
    chat_id: string | number,
    params?: Record<string, any>
  ): Promise<LorebookDiagnosticExportResponse> {
    const cid = String(chat_id)
    const query = this.buildQuery(params)
    return await bgRequest<LorebookDiagnosticExportResponse>({
      path: `/api/v1/chats/${cid}/diagnostics/lorebook${query}`,
      method: "GET"
    })
  }

  async updateChat(
    chat_id: string | number,
    payload: Record<string, any>,
    options?: { expectedVersion?: number }
  ): Promise<ServerChatSummary> {
    const cid = String(chat_id)
    let expectedVersion = options?.expectedVersion
    if (expectedVersion == null) {
      try {
        const current = await this.getChat(cid)
        if (typeof current?.version === "number") {
          expectedVersion = current.version
        }
      } catch {
        // ignore and fall back to unversioned update
      }
    }
    const qp =
      typeof expectedVersion === "number"
        ? `?expected_version=${encodeURIComponent(String(expectedVersion))}`
        : ""
    const res = await bgRequest<any>({
      path: `/api/v1/chats/${cid}${qp}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
    return this.normalizeChatSummary(res)
  }

  async deleteChat(
    chat_id: string | number,
    options?: {
      expectedVersion?: number
      hardDelete?: boolean
    }
  ): Promise<void> {
    const cid = String(chat_id)
    const query = this.buildQuery({
      ...(typeof options?.expectedVersion === "number"
        ? { expected_version: options.expectedVersion }
        : {}),
      ...(options?.hardDelete ? { hard_delete: true } : {})
    })
    await bgRequest<void>({
      path: `/api/v1/chats/${cid}${query}`,
      method: "DELETE"
    })
  }

  async restoreChat(
    chat_id: string | number,
    options?: { expectedVersion?: number }
  ): Promise<ServerChatSummary> {
    const cid = String(chat_id)
    const query = this.buildQuery(
      typeof options?.expectedVersion === "number"
        ? { expected_version: options.expectedVersion }
        : {}
    )
    const res = await bgRequest<any>({
      path: `/api/v1/chats/${cid}/restore${query}`,
      method: "POST"
    })
    return this.normalizeChatSummary(res)
  }

  async createConversationShareLink(
    chat_id: string | number,
    payload?: {
      permission?: ConversationSharePermission
      ttl_seconds?: number
      label?: string
    }
  ): Promise<ConversationShareLinkCreateResponse> {
    const cid = String(chat_id)
    return await bgRequest<ConversationShareLinkCreateResponse>({
      path: `/api/v1/chat/conversations/${encodeURIComponent(cid)}/share-links`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload || {},
    })
  }

  async listConversationShareLinks(
    chat_id: string | number
  ): Promise<ConversationShareLinksListResponse> {
    const cid = String(chat_id)
    return await bgRequest<ConversationShareLinksListResponse>({
      path: `/api/v1/chat/conversations/${encodeURIComponent(cid)}/share-links`,
      method: "GET",
    })
  }

  async revokeConversationShareLink(
    chat_id: string | number,
    shareId: string
  ): Promise<{ success: boolean; share_id: string }> {
    const cid = encodeURIComponent(String(chat_id))
    const sid = encodeURIComponent(String(shareId))
    return await bgRequest<{ success: boolean; share_id: string }>({
      path: `/api/v1/chat/conversations/${cid}/share-links/${sid}`,
      method: "DELETE",
    })
  }

  async resolveConversationShareLink(
    token: string
  ): Promise<ConversationShareLinkResolveResponse> {
    const encodedToken = encodeURIComponent(token)
    return await bgRequest<ConversationShareLinkResolveResponse>({
      path: `/api/v1/chat/shared/conversations/${encodedToken}`,
      method: "GET",
      noAuth: true,
    })
  }

  async listChatMessages(
    chat_id: string | number,
    params?: Record<string, any>,
    options?: { signal?: AbortSignal }
  ): Promise<ServerChatMessage[]> {
    const cid = String(chat_id)
    const query = this.buildQuery(params)
    const cacheKey = this.getChatMessagesCacheKey(cid, query)
    const cached = this.chatMessagesCache.get(cacheKey)
    if (cached && cached.expiresAt > Date.now()) {
      return cached.value
    }
    if (cached) {
      this.chatMessagesCache.delete(cacheKey)
    }

    const inFlight = this.chatMessagesInFlight.get(cacheKey)
    if (inFlight) {
      return inFlight
    }

    const request = (async () => {
      const data = await bgRequest<any>({
        path: `/api/v1/chats/${cid}/messages${query}`,
        method: "GET",
        abortSignal: options?.signal
      })

      let list: any[] = []

      if (Array.isArray(data)) {
        list = data
      } else if (data && typeof data === "object") {
        const obj: any = data
        if (Array.isArray(obj.messages)) {
          list = obj.messages
        } else if (Array.isArray(obj.items)) {
          list = obj.items
        } else if (Array.isArray(obj.results)) {
          list = obj.results
        } else if (Array.isArray(obj.data)) {
          list = obj.data
        }
      }

      const normalized = list.map((m) => {
        const senderCandidate =
          typeof m.sender === "string"
            ? m.sender
            : typeof m.author === "string"
              ? m.author
              : typeof (m as any)?.message?.sender === "string"
                ? (m as any).message.sender
                : typeof (m as any)?.message?.author === "string"
                  ? (m as any).message.author
                  : undefined
        const roleCandidate =
          typeof m.role === "string"
            ? m.role
            : typeof senderCandidate === "string"
              ? senderCandidate
                : typeof (m as any)?.message?.role === "string"
                  ? (m as any).message.role
                  : undefined
        const senderLower =
          typeof senderCandidate === "string"
            ? senderCandidate.trim().toLowerCase()
            : ""
        const senderLooksLikeUser =
          senderLower === "user" ||
          senderLower === "human" ||
          senderLower.startsWith("user")
        const senderLooksLikeSystem =
          senderLower === "system" || senderLower.startsWith("system")
        const senderLooksLikeTool =
          senderLower === "tool" ||
          senderLower.startsWith("tool") ||
          senderLower === "function"
        const fallbackRole =
          senderLower &&
          !senderLooksLikeUser &&
          !senderLooksLikeSystem &&
          !senderLooksLikeTool
            ? "assistant"
            : "user"
        const role =
          typeof (m as any)?.is_bot === "boolean" ||
          typeof (m as any)?.isBot === "boolean"
            ? (m as any).is_bot || (m as any).isBot
              ? "assistant"
              : "user"
            : normalizeChatRole(roleCandidate, fallbackRole)
        const created_at = String(
          m.created_at || m.createdAt || m.timestamp || ""
        )
        const metadataExtraCandidate =
          (m as any).metadata_extra ?? (m as any).metadataExtra
        const metadataExtra =
          metadataExtraCandidate &&
          typeof metadataExtraCandidate === "object" &&
          !Array.isArray(metadataExtraCandidate)
            ? (metadataExtraCandidate as Record<string, unknown>)
            : undefined
        const rawPinned =
          (metadataExtra?.pinned as unknown) ?? (m as any).pinned
        const pinned =
          typeof rawPinned === "boolean"
            ? rawPinned
            : typeof rawPinned === "string"
              ? ["1", "true", "yes", "on"].includes(rawPinned.trim().toLowerCase())
              : undefined
        return {
          id: String(m.id),
          role,
          sender:
            typeof senderCandidate === "string" && senderCandidate.trim().length > 0
              ? senderCandidate
              : undefined,
          content: String(m.content ?? ""),
          created_at,
          version:
            typeof m.version === "number"
              ? m.version
              : typeof m.expected_version === "number"
                ? m.expected_version
                : undefined,
          metadata_extra: metadataExtra,
          pinned
        } as ServerChatMessage
      })
      this.chatMessagesCache.set(cacheKey, {
        value: normalized,
        expiresAt: Date.now() + CHAT_MESSAGES_CACHE_TTL_MS
      })
      return normalized
    })()

    this.chatMessagesInFlight.set(cacheKey, request)
    try {
      return await request
    } finally {
      this.chatMessagesInFlight.delete(cacheKey)
    }
  }

  async addChatMessage(
    chat_id: string | number,
    payload: Record<string, any>
  ): Promise<ServerChatMessage> {
    const cid = String(chat_id)
    const res = await bgRequest<ServerChatMessage>({
      path: `/api/v1/chats/${cid}/messages`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
    this.invalidateChatMessagesCache(cid)
    return res
  }

  async prepareCharacterCompletion(
    chat_id: string | number,
    payload?: Record<string, any>
  ): Promise<any> {
    const cid = String(chat_id)
    const body = payload || {}
    captureChatRequestDebugSnapshot({
      endpoint: `/api/v1/chats/${cid}/completions`,
      method: "POST",
      mode: "non-stream",
      body
    })
    return await bgRequest<any>({
      path: `/api/v1/chats/${cid}/completions`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body
    })
  }

  async getCharacterPromptPreview(
    chat_id: string | number,
    payload?: Record<string, any>
  ): Promise<any> {
    const cid = String(chat_id)
    return await bgRequest<any>({
      path: `/api/v1/chats/${cid}/prompt-preview`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload || {}
    })
  }

  async persistCharacterCompletion(
    chat_id: string | number,
    payload: Record<string, any>
  ): Promise<any> {
    const cid = String(chat_id)
    try {
      const res = await bgRequest<any>({
        path: `/api/v1/chats/${cid}/completions/persist`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload
      })
      this.invalidateChatMessagesCache(cid)
      return res
    } catch (error) {
      if (isSavedDegradedCharacterPersistError(error)) {
        this.invalidateChatMessagesCache(cid)
      }
      throw error
    }
  }

  async *streamCharacterChatCompletion(
    chat_id: string | number,
    payload?: Record<string, any>,
    options?: { signal?: AbortSignal; streamIdleTimeoutMs?: number }
  ): AsyncGenerator<any> {
    const cid = String(chat_id)
    const body = { ...(payload || {}), stream: true }
    captureChatRequestDebugSnapshot({
      endpoint: `/api/v1/chats/${cid}/complete-v2`,
      method: "POST",
      mode: "stream",
      body
    })
    for await (const line of bgStream({
      path: `/api/v1/chats/${cid}/complete-v2`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      abortSignal: options?.signal,
      streamIdleTimeoutMs: options?.streamIdleTimeoutMs
    })) {
      if (!line) continue
      try {
        const parsed = JSON.parse(line)
        yield parsed
      } catch {
        yield line
      }
    }
  }

  async searchChatMessages(chat_id: string | number, query: string, limit?: number): Promise<any> {
    const cid = String(chat_id)
    const qp = `?query=${encodeURIComponent(query)}${typeof limit === 'number' ? `&limit=${encodeURIComponent(String(limit))}` : ''}`
    return await bgRequest<any>({ path: `/api/v1/chats/${cid}/messages/search${qp}`, method: 'GET' })
  }

  async completeChat(chat_id: string | number, payload?: Record<string, any>): Promise<any> {
    const cid = String(chat_id)
    const body = payload || {}
    captureChatRequestDebugSnapshot({
      endpoint: `/api/v1/chats/${cid}/complete`,
      method: "POST",
      mode: "non-stream",
      body
    })
    return await bgRequest<any>({ path: `/api/v1/chats/${cid}/complete`, method: 'POST', headers: { 'Content-Type': 'application/json' }, body })
  }

  async * streamCompleteChat(chat_id: string | number, payload?: Record<string, any>): AsyncGenerator<any> {
    const cid = String(chat_id)
    const body = payload || {}
    captureChatRequestDebugSnapshot({
      endpoint: `/api/v1/chats/${cid}/complete`,
      method: "POST",
      mode: "stream",
      body
    })
    for await (const line of bgStream({ path: `/api/v1/chats/${cid}/complete`, method: 'POST', headers: { 'Content-Type': 'application/json' }, body })) {
      try { yield JSON.parse(line) } catch {}
    }
  }

  // Message (single) APIs
  async getMessage(message_id: string | number): Promise<any> {
    const mid = String(message_id)
    return await bgRequest<any>({ path: `/api/v1/messages/${mid}`, method: 'GET' })
  }

  async editMessage(
    message_id: string | number,
    content: string,
    expectedVersion: number,
    chatId?: string | number,
    options?: { pinned?: boolean }
  ): Promise<any> {
    const mid = String(message_id)
    const qp = `?expected_version=${encodeURIComponent(String(expectedVersion))}`
    const body: Record<string, unknown> = { content }
    if (typeof options?.pinned === "boolean") {
      body.pinned = options.pinned
    }
    const res = await bgRequest<any>({
      path: `/api/v1/messages/${mid}${qp}`,
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body
    })
    if (chatId != null) {
      this.invalidateChatMessagesCache(chatId)
    }
    return res
  }

  async deleteMessage(
    message_id: string | number,
    expectedVersion: number,
    chatId?: string | number
  ): Promise<void> {
    const mid = String(message_id)
    const qp = `?expected_version=${encodeURIComponent(String(expectedVersion))}`
    await bgRequest<void>({
      path: `/api/v1/messages/${mid}${qp}`,
      method: 'DELETE'
    })
    if (chatId != null) {
      this.invalidateChatMessagesCache(chatId)
    }
  }

  async saveChatKnowledge(payload: {
    conversation_id: string | number
    message_id: string | number
    snippet: string
    tags?: string[]
    make_flashcard?: boolean
  }): Promise<any> {
    const body = {
      ...payload,
      conversation_id: String(payload.conversation_id),
      message_id: String(payload.message_id)
    }
    return await bgRequest<any>({
      path: "/api/v1/chat/knowledge/save",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body
    })
  }

  // World Books
  async listWorldBooks(include_disabled?: boolean): Promise<any> {
    const qp = include_disabled ? `?include_disabled=true` : ''
    return await bgRequest<any>({ path: `/api/v1/characters/world-books${qp}`, method: 'GET' })
  }

  async getWorldBookRuntimeConfig(): Promise<{ max_recursive_depth: number }> {
    return await bgRequest<{ max_recursive_depth: number }>({
      path: "/api/v1/characters/world-books/config",
      method: "GET"
    })
  }

  async createWorldBook(payload: Record<string, any>): Promise<any> {
    return await bgRequest<any>({ path: '/api/v1/characters/world-books', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload })
  }

  async updateWorldBook(
    world_book_id: number | string,
    payload: Record<string, any>,
    options?: { expectedVersion?: number }
  ): Promise<any> {
    const wid = String(world_book_id)
    const query = this.buildQuery(
      typeof options?.expectedVersion === "number"
        ? { expected_version: options.expectedVersion }
        : {}
    )
    return await bgRequest<any>({
      path: `/api/v1/characters/world-books/${wid}${query}`,
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: payload
    })
  }

  async deleteWorldBook(world_book_id: number | string): Promise<any> {
    const wid = String(world_book_id)
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/${wid}`, method: 'DELETE' })
  }

  async listWorldBookEntries(world_book_id: number | string, enabled_only?: boolean): Promise<any> {
    const wid = String(world_book_id)
    const qp = enabled_only ? `?enabled_only=true` : ''
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/${wid}/entries${qp}`, method: 'GET' })
  }

  async addWorldBookEntry(world_book_id: number | string, payload: Record<string, any>): Promise<any> {
    const wid = String(world_book_id)
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/${wid}/entries`, method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload })
  }

  async updateWorldBookEntry(entry_id: number | string, payload: Record<string, any>): Promise<any> {
    const eid = String(entry_id)
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/entries/${eid}`, method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: payload })
  }

  async deleteWorldBookEntry(entry_id: number | string): Promise<any> {
    const eid = String(entry_id)
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/entries/${eid}`, method: 'DELETE' })
  }

  async bulkWorldBookEntries(payload: { entry_ids: number[]; operation: string; priority?: number }): Promise<any> {
    return await bgRequest<any>({
      path: '/api/v1/characters/world-books/entries/bulk',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload
    })
  }

  async attachWorldBookToCharacter(
    character_id: number | string,
    world_book_id: number | string,
    options?: { enabled?: boolean; priority?: number }
  ): Promise<any> {
    const cid = String(character_id)
    const body: Record<string, any> = { world_book_id: Number(world_book_id) }
    if (typeof options?.enabled === "boolean") {
      body.enabled = options.enabled
    }
    if (typeof options?.priority === "number" && Number.isFinite(options.priority)) {
      body.priority = options.priority
    }
    return await bgRequest<any>({
      path: `/api/v1/characters/${cid}/world-books`,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    })
  }

  async detachWorldBookFromCharacter(character_id: number | string, world_book_id: number | string): Promise<any> {
    const cid = String(character_id)
    const wid = String(world_book_id)
    return await bgRequest<any>({ path: `/api/v1/characters/${cid}/world-books/${wid}`, method: 'DELETE' })
  }

  async listCharacterWorldBooks(character_id: number | string): Promise<any> {
    const cid = String(character_id)
    return await bgRequest<any>({ path: `/api/v1/characters/${cid}/world-books`, method: 'GET' })
  }

  async processWorldBookContext(payload: {
    text: string
    world_book_ids?: number[]
    character_id?: number
    scan_depth?: number
    token_budget?: number
    recursive_scanning?: boolean
  }): Promise<WorldBookProcessResponse> {
    return await bgRequest<WorldBookProcessResponse>({
      path: "/api/v1/characters/world-books/process",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async exportWorldBook(world_book_id: number | string): Promise<any> {
    const wid = String(world_book_id)
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/${wid}/export`, method: 'GET' })
  }

  async importWorldBook(request: { world_book: Record<string, any>; entries?: any[]; merge_on_conflict?: boolean }): Promise<any> {
    return await bgRequest<any>({ path: '/api/v1/characters/world-books/import', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: request })
  }

  async worldBookStatistics(world_book_id: number | string): Promise<any> {
    const wid = String(world_book_id)
    return await bgRequest<any>({ path: `/api/v1/characters/world-books/${wid}/statistics`, method: 'GET' })
  }

  // Chat Dictionaries
  async createDictionary(payload: Record<string, any>): Promise<any> {
    return await bgRequest<any>({ path: '/api/v1/chat/dictionaries', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload })
  }

  async listDictionaries(include_inactive?: boolean, include_usage?: boolean): Promise<any> {
    const params = new URLSearchParams()
    if (include_inactive) params.set('include_inactive', 'true')
    if (include_usage) params.set('include_usage', 'true')
    const qp = params.toString()
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries${qp ? `?${qp}` : ''}`, method: 'GET' })
  }

  async getDictionary(dictionary_id: number | string): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}`, method: 'GET' })
  }

  async updateDictionary(dictionary_id: number | string, payload: Record<string, any>): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}`, method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: payload })
  }

  async deleteDictionary(dictionary_id: number | string, hard_delete?: boolean): Promise<any> {
    const id = String(dictionary_id)
    const qp = hard_delete ? `?hard_delete=true` : ''
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}${qp}`, method: 'DELETE' })
  }

  async listDictionaryEntries(dictionary_id: number | string, group?: string): Promise<any> {
    const id = String(dictionary_id)
    const qp = group ? `?group=${encodeURIComponent(group)}` : ''
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}/entries${qp}`, method: 'GET' })
    }

  async addDictionaryEntry(dictionary_id: number | string, payload: Record<string, any>): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}/entries`, method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload })
  }

  async updateDictionaryEntry(entry_id: number | string, payload: Record<string, any>): Promise<any> {
    const eid = String(entry_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/entries/${eid}`, method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: payload })
  }

  async deleteDictionaryEntry(entry_id: number | string): Promise<any> {
    const eid = String(entry_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/entries/${eid}`, method: 'DELETE' })
  }

  async bulkDictionaryEntries(payload: {
    entry_ids: number[]
    operation: "delete" | "activate" | "deactivate" | "group"
    group_name?: string
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/dictionaries/entries/bulk",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async reorderDictionaryEntries(
    dictionary_id: number | string,
    payload: {
      entry_ids: number[]
    }
  ): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({
      path: `/api/v1/chat/dictionaries/${id}/entries/reorder`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async exportDictionaryMarkdown(dictionary_id: number | string): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}/export`, method: 'GET' })
  }

  async exportDictionaryJSON(dictionary_id: number | string): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}/export/json`, method: 'GET' })
  }

  async importDictionaryJSON(data: any, activate?: boolean): Promise<any> {
    return await bgRequest<any>({ path: '/api/v1/chat/dictionaries/import/json', method: 'POST', headers: { 'Content-Type': 'application/json' }, body: { data, activate: !!activate } })
  }

  async importDictionaryMarkdown(
    name: string,
    content: string,
    activate?: boolean
  ): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/dictionaries/import",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        name,
        content,
        activate: !!activate
      }
    })
  }

  async validateDictionary(payload: {
    data: Record<string, any>
    schema_version?: number
    strict?: boolean
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/dictionaries/validate",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async processDictionary(payload: {
    text: string
    token_budget?: number
    dictionary_id?: number | string
    max_iterations?: number
    chat_id?: string
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/dictionaries/process",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async dictionaryActivity(
    dictionary_id: number | string,
    params?: {
      limit?: number
      offset?: number
    }
  ): Promise<any> {
    const id = String(dictionary_id)
    const query = new URLSearchParams()
    if (typeof params?.limit === "number" && Number.isFinite(params.limit)) {
      query.set("limit", String(Math.max(1, Math.floor(params.limit))))
    }
    if (typeof params?.offset === "number" && Number.isFinite(params.offset)) {
      query.set("offset", String(Math.max(0, Math.floor(params.offset))))
    }
    const qp = query.toString()
    return await bgRequest<any>({
      path: `/api/v1/chat/dictionaries/${id}/activity${qp ? `?${qp}` : ""}`,
      method: "GET"
    })
  }

  async dictionaryStatistics(dictionary_id: number | string): Promise<any> {
    const id = String(dictionary_id)
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}/statistics`, method: 'GET' })
  }

  async dictionaryVersions(
    dictionary_id: number | string,
    params?: {
      limit?: number
      offset?: number
    }
  ): Promise<any> {
    const id = String(dictionary_id)
    const query = new URLSearchParams()
    if (typeof params?.limit === "number" && Number.isFinite(params.limit)) {
      query.set("limit", String(Math.max(1, Math.floor(params.limit))))
    }
    if (typeof params?.offset === "number" && Number.isFinite(params.offset)) {
      query.set("offset", String(Math.max(0, Math.floor(params.offset))))
    }
    const qp = query.toString()
    return await bgRequest<any>({
      path: `/api/v1/chat/dictionaries/${id}/versions${qp ? `?${qp}` : ""}`,
      method: "GET"
    })
  }

  async dictionaryVersionSnapshot(
    dictionary_id: number | string,
    revision: number | string
  ): Promise<any> {
    const id = String(dictionary_id)
    const rev = String(revision)
    return await bgRequest<any>({
      path: `/api/v1/chat/dictionaries/${id}/versions/${rev}`,
      method: "GET"
    })
  }

  async revertDictionaryVersion(
    dictionary_id: number | string,
    revision: number | string
  ): Promise<any> {
    const id = String(dictionary_id)
    const rev = String(revision)
    return await bgRequest<any>({
      path: `/api/v1/chat/dictionaries/${id}/versions/${rev}/revert`,
      method: "POST"
    })
  }

  // Chat Documents
  async generateChatDocument(payload: {
    conversation_id: string | number
    document_type: string
    provider: string
    model: string
    specific_message?: string | null
    custom_prompt?: string | null
    stream?: boolean
    async_generation?: boolean
  }): Promise<any> {
    const body = {
      ...payload,
      conversation_id: String(payload.conversation_id)
    }
    return await bgRequest<any>({
      path: "/api/v1/chat/documents/generate",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body
    })
  }

  async listChatDocuments(params?: {
    conversation_id?: string | number
    document_type?: string
    limit?: number
  }): Promise<any> {
    const query = this.buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/chat/documents${query}`,
      method: "GET"
    })
  }

  async getChatDocument(document_id: number | string): Promise<any> {
    const id = String(document_id)
    return await bgRequest<any>({
      path: `/api/v1/chat/documents/${id}`,
      method: "GET"
    })
  }

  async deleteChatDocument(document_id: number | string): Promise<any> {
    const id = String(document_id)
    return await bgRequest<any>({
      path: `/api/v1/chat/documents/${id}`,
      method: "DELETE"
    })
  }

  async getChatDocumentJob(job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chat/documents/jobs/${id}`,
      method: "GET"
    })
  }

  async cancelChatDocumentJob(job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chat/documents/jobs/${id}`,
      method: "DELETE"
    })
  }

  async saveChatDocumentPrompt(payload: {
    document_type: string
    system_prompt: string
    user_prompt: string
    temperature?: number
    max_tokens?: number
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/documents/prompts",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async getChatDocumentPrompt(document_type: string): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/chat/documents/prompts/${encodeURIComponent(document_type)}`,
      method: "GET"
    })
  }

  async chatDocumentStatistics(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/documents/statistics",
      method: "GET"
    })
  }

  // Chatbooks
  async exportChatbook(payload: {
    name: string
    description: string
    content_selections: Record<string, string[]>
    author?: string
    include_media?: boolean
    media_quality?: string
    include_embeddings?: boolean
    include_generated_content?: boolean
    tags?: string[]
    categories?: string[]
    async_mode?: boolean
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chatbooks/export",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async previewChatbook(file: File): Promise<any> {
    const data = await file.arrayBuffer()
    const name = file.name || "chatbook.zip"
    const type = file.type || "application/zip"
    return await bgUpload<any>({
      path: "/api/v1/chatbooks/preview",
      method: "POST",
      file: { name, type, data }
    })
  }

  async importChatbook(
    file: File,
    options?: {
      conflict_resolution?: string
      prefix_imported?: boolean
      import_media?: boolean
      import_embeddings?: boolean
      async_mode?: boolean
      content_selections?: Record<string, string[]>
    }
  ): Promise<any> {
    const data = await file.arrayBuffer()
    const name = file.name || "chatbook.zip"
    const type = file.type || "application/zip"
    const normalized: Record<string, any> = {}
    for (const [k, v] of Object.entries(options || {})) {
      if (typeof v === "undefined" || v === null) continue
      normalized[k] = typeof v === "boolean" ? (v ? "true" : "false") : v
    }
    return await bgUpload<any>({
      path: "/api/v1/chatbooks/import",
      method: "POST",
      fields: normalized,
      file: { name, type, data }
    })
  }

  async listChatbookExportJobs(params?: { limit?: number; offset?: number }): Promise<any> {
    const query = this.buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/export/jobs${query}`,
      method: "GET"
    })
  }

  async listChatbookImportJobs(params?: { limit?: number; offset?: number }): Promise<any> {
    const query = this.buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/import/jobs${query}`,
      method: "GET"
    })
  }

  async getChatbookExportJob(job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/export/jobs/${id}`,
      method: "GET"
    })
  }

  async getChatbookImportJob(job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/import/jobs/${id}`,
      method: "GET"
    })
  }

  async cancelChatbookExportJob(job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/export/jobs/${id}`,
      method: "DELETE"
    })
  }

  async cancelChatbookImportJob(job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/import/jobs/${id}`,
      method: "DELETE"
    })
  }

  async removeChatbookExportJob(job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/export/jobs/${id}/remove`,
      method: "DELETE"
    })
  }

  async removeChatbookImportJob(job_id: string): Promise<any> {
    const id = String(job_id)
    return await bgRequest<any>({
      path: `/api/v1/chatbooks/import/jobs/${id}/remove`,
      method: "DELETE"
    })
  }

  async cleanupChatbooks(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chatbooks/cleanup",
      method: "POST"
    })
  }

  async chatbooksHealth(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chatbooks/health",
      method: "GET"
    })
  }

  async downloadChatbookExport(job_id: string): Promise<{ blob: Blob; filename: string }> {
    await this.ensureConfigForRequest(true)
    const response = await this.request<{
      ok: boolean
      status: number
      data?: ArrayBuffer
      error?: string
      headers?: Record<string, string>
    }>({
      path: `/api/v1/chatbooks/download/${encodeURIComponent(job_id)}`,
      method: "GET",
      headers: { Accept: "application/octet-stream" },
      responseType: "arrayBuffer",
      returnResponse: true
    })
    if (!response) {
      throw new Error("Download failed")
    }
    if (!response.ok) {
      throw new Error(response.error || `Download failed: ${response.status}`)
    }
    const headers = new Headers(response.headers || {})
    const blob = new Blob([response.data ?? new Uint8Array()], {
      type: headers.get("content-type") || "application/octet-stream"
    })
    const disposition = headers.get("content-disposition")
    let filename = `chatbook-${job_id}.zip`
    if (disposition) {
      const utfMatch = disposition.match(/filename\*=UTF-8''([^;]+)/i)
      const plainMatch = disposition.match(/filename="?([^\";]+)"?/i)
      const raw = utfMatch?.[1] || plainMatch?.[1]
      if (raw) {
        try {
          filename = decodeURIComponent(raw)
        } catch {
          filename = raw
        }
      }
    }
    return { blob, filename }
  }

  async chatQueueStatus(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/chat/queue/status",
      method: "GET"
    })
  }

  async chatQueueActivity(limit?: number): Promise<any> {
    const query = this.buildQuery(
      typeof limit === "number" ? { limit } : undefined
    )
    return await bgRequest<any>({
      path: `/api/v1/chat/queue/activity${query}`,
      method: "GET"
    })
  }

  // STT Methods
  async getTranscriptionModels(options?: { timeoutMs?: number }): Promise<any> {
    await this.ensureConfigForRequest(true)
    return await bgRequest<any>({
      path: "/api/v1/media/transcription-models",
      method: "GET",
      timeoutMs: options?.timeoutMs
    })
  }

  async getTranscriptionModelHealth(model: string): Promise<any> {
    await this.ensureConfigForRequest(true)
    const query = this.buildQuery({ model })
    return await bgRequest<any>({
      path: `/api/v1/audio/transcriptions/health${query}`,
      method: "GET"
    })
  }

  async transcribeAudio(audioFile: File | Blob, options?: any): Promise<any> {
    await this.ensureConfigForRequest(true)
    const fields: Record<string, any> = {}
    if (options) {
      if (options.model != null) fields.model = options.model
      if (options.language != null) fields.language = options.language
      if (options.prompt != null) fields.prompt = options.prompt
      if (options.response_format != null) fields.response_format = options.response_format
      if (options.temperature != null) fields.temperature = options.temperature
      if (options.task != null) fields.task = options.task
      if (options.timestamp_granularities != null) {
        fields.timestamp_granularities = options.timestamp_granularities
      }
      if (options.segment != null) fields.segment = options.segment
      if (options.seg_K != null) fields.seg_K = options.seg_K
      if (options.seg_min_segment_size != null) {
        fields.seg_min_segment_size = options.seg_min_segment_size
      }
      if (options.seg_lambda_balance != null) {
        fields.seg_lambda_balance = options.seg_lambda_balance
      }
      if (options.seg_utterance_expansion_width != null) {
        fields.seg_utterance_expansion_width = options.seg_utterance_expansion_width
      }
      if (options.seg_embeddings_provider != null) {
        fields.seg_embeddings_provider = options.seg_embeddings_provider
      }
      if (options.seg_embeddings_model != null) {
        fields.seg_embeddings_model = options.seg_embeddings_model
      }
    }
    const data = await audioFile.arrayBuffer()
    const name = (typeof File !== 'undefined' && audioFile instanceof File && (audioFile as File).name) ? (audioFile as File).name : 'audio'
    const type = (audioFile as any)?.type || 'application/octet-stream'
    return await this.upload<any>({ path: '/api/v1/audio/transcriptions', method: 'POST', fields, file: { name, type, data } })
  }

  async synthesizeSpeech(
    text: string,
    options?: {
      voice?: string
      model?: string
      responseFormat?: string
      speed?: number
      language?: string
      normalizationOptions?: Record<string, any>
      extraParams?: Record<string, any>
      stream?: boolean
      signal?: AbortSignal
    }
  ): Promise<ArrayBuffer> {
    await this.ensureConfigForRequest(true)
    const body: Record<string, any> = { input: text, text }
    if (options?.voice) body.voice = options.voice
    if (options?.model) body.model = options.model
    if (options?.responseFormat) body.response_format = options.responseFormat
    if (options?.speed != null) body.speed = options.speed
    if (options?.language) body.lang_code = options.language
    if (options?.normalizationOptions) {
      body.normalization_options = options.normalizationOptions
    }
    if (options?.extraParams) body.extra_params = options.extraParams
    if (options?.stream != null) body.stream = options.stream
    const accept = (() => {
      switch ((options?.responseFormat || "").trim().toLowerCase()) {
        case "wav":
          return "audio/wav"
        case "opus":
          return "audio/opus"
        case "aac":
          return "audio/aac"
        case "flac":
          return "audio/flac"
        case "ogg":
          return "audio/ogg"
        case "webm":
          return "audio/webm"
        case "ulaw":
          return "audio/basic"
        case "pcm":
          return "audio/L16; rate=24000; channels=1"
        case "mp3":
        default:
          return "audio/mpeg"
      }
    })()
    const data = await this.request<any>({
      path: "/api/v1/audio/speech",
      method: "POST",
      headers: { Accept: accept },
      body,
      responseType: "arrayBuffer",
      abortSignal: options?.signal
    })

    const normalizeArrayBuffer = async (value: unknown): Promise<ArrayBuffer | null> => {
      if (!value) return null
      if (value instanceof ArrayBuffer) return value
      if (typeof SharedArrayBuffer !== "undefined" && value instanceof SharedArrayBuffer) {
        return new Uint8Array(value).slice(0).buffer
      }
      if (ArrayBuffer.isView(value)) {
        const view = value as ArrayBufferView
        if (
          typeof SharedArrayBuffer !== "undefined" &&
          view.buffer instanceof SharedArrayBuffer
        ) {
          const copy = new Uint8Array(view.byteLength)
          copy.set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength))
          return copy.buffer
        }
        if (view.buffer instanceof ArrayBuffer) {
          return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength)
        }
      }
      if (typeof Blob !== "undefined" && value instanceof Blob) {
        return await value.arrayBuffer()
      }
      const tag = Object.prototype.toString.call(value)
      if (tag === "[object ArrayBuffer]" && typeof (value as any).slice === "function") {
        return (value as any).slice(0)
      }
      if (Array.isArray(value) && value.every((entry) => typeof entry === "number")) {
        return new Uint8Array(value).buffer
      }
      if (typeof value === "object") {
        const record = value as Record<string, any>
        if (
          typeof record.type === "string" &&
          record.type.toLowerCase() === "buffer" &&
          Array.isArray(record.data)
        ) {
          return new Uint8Array(record.data).buffer
        }
        if (
          typeof record.ok === "boolean" &&
          Object.prototype.hasOwnProperty.call(record, "data")
        ) {
          const nested = await normalizeArrayBuffer(record.data)
          if (nested) return nested
        }
        if (
          typeof record.byteLength === "number" &&
          typeof record.slice === "function"
        ) {
          try {
            const sliced = record.slice(0)
            if (
              typeof SharedArrayBuffer !== "undefined" &&
              sliced instanceof SharedArrayBuffer
            ) {
              return new Uint8Array(sliced).slice(0).buffer
            }
            return sliced
          } catch {
            // ignore and continue
          }
        }
        if (typeof record.arrayBuffer === "function") {
          return await record.arrayBuffer()
        }
        if (record.data !== undefined) {
          const nested = await normalizeArrayBuffer(record.data)
          if (nested) return nested
        }
        if (record.buffer !== undefined) {
          const nested = await normalizeArrayBuffer(record.buffer)
          if (nested) return nested
        }
        if (typeof record.length === "number") {
          const maybeArray = Array.from(record as ArrayLike<unknown>)
          if (maybeArray.length > 0 && maybeArray.every((entry) => typeof entry === "number")) {
            return new Uint8Array(maybeArray).buffer
          }
        }
      }
      return null
    }

    const normalized = await normalizeArrayBuffer(data)
    if (!normalized) {
      // eslint-disable-next-line no-console
      try {
        // eslint-disable-next-line no-console
        console.error("[tldw][tts] Invalid audio buffer from /api/v1/audio/speech", {
          type: typeof data,
          tag: Object.prototype.toString.call(data),
          constructor:
            typeof data === "object" && data ? (data as any).constructor?.name : undefined,
          keys:
            typeof data === "object" && data
              ? Object.keys(data as object).slice(0, 10)
              : [],
          dataType: typeof (data as any)?.data,
          dataTag:
            typeof (data as any)?.data !== "undefined"
              ? Object.prototype.toString.call((data as any).data)
              : undefined,
          dataKeys:
            (data as any)?.data && typeof (data as any).data === "object"
              ? Object.keys((data as any).data).slice(0, 10)
              : undefined
        })
        if (typeof data === "object" && data) {
          // eslint-disable-next-line no-console
          console.error(
            "[tldw][tts] Invalid audio buffer payload sample",
            JSON.stringify(data, null, 2).slice(0, 2000)
          )
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error("[tldw][tts] Failed to log invalid audio buffer payload", e)
      }
      throw new Error("TTS returned an invalid audio buffer.")
    }
    return normalized
  }

  async createTtsJob(payload: {
    input: string
    model?: string
    voice?: string
    response_format?: string
    speed?: number
    lang_code?: string
    normalization_options?: Record<string, any>
    extra_params?: Record<string, any>
  }): Promise<{ job_id: number; status: string }> {
    return await bgRequest<{ job_id: number; status: string }>({
      path: "/api/v1/audio/speech/jobs",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async getTtsJobArtifacts(jobId: number): Promise<{
    job_id: number
    artifacts: Array<{
      output_id: number
      format: string
      type: string
      title: string
      download_url: string
      metadata?: Record<string, any>
    }>
  }> {
    const id = encodeURIComponent(String(jobId))
    return await bgRequest({
      path: `/api/v1/audio/speech/jobs/${id}/artifacts`,
      method: "GET"
    })
  }

  async *streamAudioJobProgress(
    jobId: number,
    options?: { signal?: AbortSignal; afterId?: number; streamIdleTimeoutMs?: number }
  ): AsyncGenerator<any> {
    const id = encodeURIComponent(String(jobId))
    const query = options?.afterId
      ? `?after_id=${encodeURIComponent(String(options.afterId))}`
      : ""
    const path = `/api/v1/audio/jobs/${id}/progress/stream${query}` as const
    for await (const line of bgStream({
      path,
      method: "GET",
      headers: { Accept: "text/event-stream" },
      abortSignal: options?.signal,
      streamIdleTimeoutMs: options?.streamIdleTimeoutMs
    })) {
      try {
        yield JSON.parse(line as string)
      } catch {
        yield { event: "raw", data: line }
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Data Tables API
  // ─────────────────────────────────────────────────────────────────────────

  async listDataTables(params?: {
    page?: number
    page_size?: number
    limit?: number
    offset?: number
    search?: string
    status?: string
    workspace_tag?: string
  }): Promise<{ tables: any[]; total: number }> {
    const limit = params?.limit ?? params?.page_size ?? 20
    const page = params?.page ?? 1
    const offset = params?.offset ?? Math.max(0, (page - 1) * limit)
    const query = this.buildQuery({
      limit,
      offset,
      search: params?.search,
      status_filter: params?.status,
      workspace_tag: params?.workspace_tag
    } as Record<string, any>)
    const response = await bgRequest<any>({
      path: `/api/v1/data-tables${query}`,
      method: "GET"
    })
    return mapApiListToUi(response)
  }

  async getDataTable(
    tableId: string,
    params?: {
      rows_limit?: number
      rows_offset?: number
      include_rows?: boolean
      include_sources?: boolean
    }
  ): Promise<any> {
    const id = encodeURIComponent(tableId)
    const query = this.buildQuery({
      rows_limit: params?.rows_limit,
      rows_offset: params?.rows_offset,
      include_rows: params?.include_rows,
      include_sources: params?.include_sources
    } as Record<string, any>)
    const response = await bgRequest<any>({
      path: `/api/v1/data-tables/${id}${query}`,
      method: "GET"
    })
    return response?.table ? mapApiDetailToUi(response) : response
  }

  async generateDataTable(payload: {
    name: string
    prompt: string
    workspace_tag?: string
    sources: Array<{ type: string; id: string; title: string; snippet?: string }>
    column_hints?: Array<{ name?: string; type?: string; description?: string; format?: string }>
    model?: string
    max_rows?: number
  }): Promise<ApiDataTableGenerateResponse> {
    const body = {
      name: payload.name,
      prompt: payload.prompt,
      workspace_tag: payload.workspace_tag,
      sources: payload.sources.map(mapUiSourceToApi),
      column_hints: payload.column_hints,
      model: payload.model,
      max_rows: payload.max_rows
    }
    return await bgRequest<ApiDataTableGenerateResponse>({
      path: "/api/v1/data-tables/generate",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body
    })
  }

  async updateDataTable(
    tableId: string,
    payload: { name?: string; description?: string }
  ): Promise<any> {
    const id = encodeURIComponent(tableId)
    return await bgRequest<any>({
      path: `/api/v1/data-tables/${id}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async saveDataTableContent(
    tableId: string,
    payload: {
      columns: DataTableColumn[]
      rows: Record<string, any>[]
    }
  ): Promise<any> {
    const id = encodeURIComponent(tableId)
    const body = buildContentPayload(payload.columns, payload.rows)
    const response = await bgRequest<any>({
      path: `/api/v1/data-tables/${id}/content`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body
    })
    return response?.table ? mapApiDetailToUi(response) : response
  }

  async deleteDataTable(tableId: string): Promise<void> {
    const id = encodeURIComponent(tableId)
    await bgRequest<void>({
      path: `/api/v1/data-tables/${id}`,
      method: "DELETE"
    })
  }

  async getDataTableJob(jobId: number): Promise<ApiDataTableJobStatus> {
    return await bgRequest<ApiDataTableJobStatus>({
      path: `/api/v1/data-tables/jobs/${encodeURIComponent(String(jobId))}`,
      method: "GET"
    })
  }

  async regenerateDataTable(
    tableId: string,
    payload?: { prompt?: string; model?: string; max_rows?: number }
  ): Promise<ApiDataTableGenerateResponse> {
    const id = encodeURIComponent(tableId)
    return await bgRequest<ApiDataTableGenerateResponse>({
      path: `/api/v1/data-tables/${id}/regenerate`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload || {}
    })
  }

  async exportDataTable(
    tableId: string,
    format: "csv" | "xlsx" | "json"
  ): Promise<{ blob: Blob; filename: string }> {
    await this.ensureConfigForRequest(true)

    const fallbackFilename = `data-table-${tableId}.${format}`
    const resolveFilename = (res: Response) => {
      const disposition = res.headers.get("content-disposition")
      if (!disposition) return fallbackFilename
      const utfMatch = disposition.match(/filename\*=UTF-8''([^;]+)/i)
      const plainMatch = disposition.match(/filename="?([^\";]+)"?/i)
      const raw = utfMatch?.[1] || plainMatch?.[1]
      if (!raw) return fallbackFilename
      try {
        return decodeURIComponent(raw)
      } catch {
        return raw
      }
    }
    const readErrorDetail = async (res: Response) => {
      try {
        const data = await res.json()
        return data?.detail || data?.error || data?.message
      } catch {
        return undefined
      }
    }
    const bytesToArrayBuffer = (bytes: Uint8Array): ArrayBuffer => {
      if (bytes.buffer instanceof ArrayBuffer) {
        return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
      }
      return new Uint8Array(bytes).buffer as ArrayBuffer
    }
    const requestWithAuth = async (
      path: PathOrUrl,
      options?: {
        method?: "GET" | "POST" | "PUT" | "DELETE"
        body?: unknown
        signal?: AbortSignal
      }
    ) => {
      const response = await bgRequest<{
        ok: boolean
        status: number
        data?: unknown
        error?: string
        headers?: Record<string, string>
      }, PathOrUrl>({
        path,
        method: options?.method ?? "GET",
        body: options?.body,
        abortSignal: options?.signal,
        responseType: "arrayBuffer",
        returnResponse: true
      })
      if (!response) {
        throw new Error(`Request failed (${options?.method ?? "GET"} ${path})`)
      }
      if (!response.ok && response.status === 0) {
        throw new Error(response.error || "Network error")
      }
      const headers = new Headers(response.headers || {})
      let body: BodyInit | null = null
      if (response.data instanceof ArrayBuffer) {
        body = response.data
      } else if (response.data instanceof Uint8Array) {
        body = bytesToArrayBuffer(response.data)
      } else if (response.data instanceof Blob) {
        body = response.data
      } else if (typeof response.data === "string") {
        body = response.data
      } else if (response.data != null) {
        body = JSON.stringify(response.data)
        if (!headers.has("content-type")) {
          headers.set("content-type", "application/json")
        }
      }
      return new Response(body, { status: response.status, headers })
    }
    const readBlobResponse = async (res: Response) => {
      const blob = await res.blob()
      return { blob, filename: resolveFilename(res) }
    }
    const decodeBase64Blob = (data: string, contentType?: string | null) => {
      const binary = atob(data)
      const bytes = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i)
      }
      return new Blob([bytes], {
        type: contentType || "application/octet-stream"
      })
    }
    const waitForExportReady = async (fileId: number) => {
      const timeoutMs = 5 * 60 * 1000
      const intervalMs = 1500
      const start = Date.now()
      while (Date.now() - start < timeoutMs) {
        const statusRes = await requestWithAuth(`/api/v1/files/${fileId}`)
        if (!statusRes.ok) {
          const detail = await readErrorDetail(statusRes)
          throw new Error(detail || `Export status failed: ${statusRes.status}`)
        }
        const payload = await statusRes.json()
        const exportInfo = payload?.artifact?.export || payload?.export
        if (exportInfo?.status === "ready") {
          return exportInfo
        }
        if (exportInfo?.status && exportInfo.status !== "pending") {
          throw new Error("Export failed")
        }
        await new Promise((resolve) => setTimeout(resolve, intervalMs))
      }
      throw new Error("Export timed out")
    }
    const downloadFromUrl = async (url: string) => {
      const resolved = url.startsWith("http")
        ? (url as PathOrUrl)
        : ((url.startsWith("/") ? url : `/${url}`) as PathOrUrl)
      const fileRes = await requestWithAuth(resolved)
      if (!fileRes.ok) {
        const detail = await readErrorDetail(fileRes)
        throw new Error(detail || `Export download failed: ${fileRes.status}`)
      }
      return await readBlobResponse(fileRes)
    }
    const exportViaArtifact = async () => {
      const exportUrl =
        `/api/v1/data-tables/${encodeURIComponent(tableId)}/export?format=${encodeURIComponent(
          format
        )}&async_mode=auto&mode=url` as AllowedPath
      const exportRes = await requestWithAuth(exportUrl)
      if (!exportRes.ok) {
        const detail = await readErrorDetail(exportRes)
        throw new Error(detail || `Export failed: ${exportRes.status}`)
      }
      const contentType = exportRes.headers.get("content-type") || ""
      if (!contentType.includes("application/json")) {
        return await readBlobResponse(exportRes)
      }
      const payload = await exportRes.json()
      const exportInfo = payload?.export || payload?.artifact?.export
      const fileId = payload?.file_id || payload?.artifact?.file_id
      if (exportInfo?.content_b64) {
        const blob = decodeBase64Blob(
          exportInfo.content_b64,
          exportInfo.content_type
        )
        return { blob, filename: resolveFilename(exportRes) }
      }
      if (!fileId) {
        throw new Error("Export response missing file id")
      }
      const resolvedExport =
        exportInfo?.status === "pending" ? await waitForExportReady(fileId) : exportInfo
      if (!resolvedExport?.url) {
        throw new Error("Export URL missing")
      }
      return await downloadFromUrl(resolvedExport.url)
    }

    const url =
      `/api/v1/data-tables/${encodeURIComponent(tableId)}/export?format=${encodeURIComponent(
        format
      )}&download=true` as AllowedPath
    const res = await requestWithAuth(url)

    if (!res.ok) {
      const detail = await readErrorDetail(res)
      if (res.status === 422 && detail === "export_size_exceeded") {
        return await exportViaArtifact()
      }
      throw new Error(detail || `Export failed: ${res.status}`)
    }

    return await readBlobResponse(res)
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // File Artifacts
  // ─────────────────────────────────────────────────────────────────────────────

  async createImageArtifact(request: ImageArtifactRequest): Promise<FileCreateResponse> {
    const payload: Record<string, unknown> = {
      backend: request.backend,
      prompt: request.prompt
    }
    if (request.negativePrompt) payload.negative_prompt = request.negativePrompt
    if (typeof request.referenceFileId === "number") {
      payload.reference_file_id = request.referenceFileId
    }
    if (typeof request.width === "number") payload.width = request.width
    if (typeof request.height === "number") payload.height = request.height
    if (typeof request.steps === "number") payload.steps = request.steps
    if (typeof request.cfgScale === "number") payload.cfg_scale = request.cfgScale
    if (typeof request.seed === "number") payload.seed = request.seed
    if (request.sampler) payload.sampler = request.sampler
    if (request.model) payload.model = request.model
    if (request.extraParams) payload.extra_params = request.extraParams

    const body: Record<string, unknown> = {
      file_type: "image",
      payload,
      export: {
        format: request.format || "png",
        mode: "inline",
        async_mode: "sync"
      },
      options: {
        persist: typeof request.persist === "boolean" ? request.persist : true
      }
    }
    if (request.title) {
      body.title = request.title
    }

    return await this.request<FileCreateResponse>({
      path: "/api/v1/files/create",
      method: "POST",
      body,
      timeoutMs: request.timeoutMs
    })
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Collections / Reading List API
  // ─────────────────────────────────────────────────────────────────────────────

  async getItems(params?: {
    page?: number
    size?: number
    q?: string
    status_filter?: string | string[]
    tags?: string[]
    favorite?: boolean
    domain?: string
    date_from?: string
    date_to?: string
    origin?: string
    job_id?: number
    run_id?: number
  }): Promise<any> {
    const query = new URLSearchParams()
    if (params?.page) query.set("page", String(params.page))
    if (params?.size) query.set("size", String(params.size))
    if (params?.q) query.set("q", params.q)
    if (params?.status_filter) {
      const statuses = Array.isArray(params.status_filter)
        ? params.status_filter
        : [params.status_filter]
      statuses.filter(Boolean).forEach((status) => query.append("status_filter", status))
    }
    if (params?.tags?.length) params.tags.forEach((tag) => query.append("tags", tag))
    if (params?.favorite !== undefined) query.set("favorite", String(params.favorite))
    if (params?.domain) query.set("domain", params.domain)
    if (params?.date_from) query.set("date_from", params.date_from)
    if (params?.date_to) query.set("date_to", params.date_to)
    if (params?.origin) query.set("origin", params.origin)
    if (params?.job_id !== undefined) query.set("job_id", String(params.job_id))
    if (params?.run_id !== undefined) query.set("run_id", String(params.run_id))
    const qs = query.toString()
    const path = `/api/v1/items${qs ? `?${qs}` : ""}` as const
    const data = await bgRequest<any>({ path, method: "GET" })
    const items = Array.isArray(data?.items)
      ? data.items.map((item: any) => ({
          ...item,
          id: String(item?.id),
          content_item_id:
            item?.content_item_id === null || typeof item?.content_item_id === "undefined"
              ? undefined
              : String(item.content_item_id),
          media_id:
            item?.media_id === null || typeof item?.media_id === "undefined"
              ? undefined
              : String(item.media_id),
          title: item?.title || item?.url || "Untitled",
          tags: Array.isArray(item?.tags) ? item.tags : []
        }))
      : []
    return {
      ...data,
      items,
      total: data?.total ?? items.length,
      page: data?.page ?? params?.page ?? 1,
      size: data?.size ?? params?.size ?? items.length
    }
  }

  async bulkUpdateItems(data: {
    item_ids: string[]
    action: "set_status" | "set_favorite" | "add_tags" | "remove_tags" | "replace_tags" | "delete"
    status?: string
    favorite?: boolean
    tags?: string[]
    hard?: boolean
  }): Promise<{
    total: number
    succeeded: number
    failed: number
    results: Array<{ item_id: string; success: boolean; error?: string | null }>
  }> {
    const itemIds = (data.item_ids || [])
      .map((id) => Number(id))
      .filter((id) => Number.isFinite(id) && id > 0)
      .map((id) => Math.floor(id))
    if (itemIds.length === 0) {
      throw new Error("item_ids_required")
    }

    const response = await bgRequest<any>({
      path: "/api/v1/items/bulk",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        item_ids: itemIds,
        action: data.action,
        status: data.status,
        favorite: data.favorite,
        tags: data.tags,
        hard: data.hard
      }
    })

    const results = Array.isArray(response?.results)
      ? response.results.map((entry: any) => ({
          item_id: String(entry?.item_id),
          success: Boolean(entry?.success),
          error: entry?.error ?? null
        }))
      : []

    return {
      total: response?.total ?? itemIds.length,
      succeeded: response?.succeeded ?? results.filter((entry) => entry.success).length,
      failed: response?.failed ?? results.filter((entry) => !entry.success).length,
      results
    }
  }

  async getReadingList(params?: {
    page?: number
    size?: number
    q?: string
    status?: string | string[]
    tags?: string[]
    favorite?: boolean
    sort?: string
    domain?: string
    date_from?: string
    date_to?: string
  }): Promise<ReadingListResponse> {
    const query = new URLSearchParams()
    if (params?.page) query.set("page", String(params.page))
    if (params?.size) query.set("size", String(params.size))
    if (params?.q) query.set("q", params.q)
    if (params?.status) {
      const statuses = Array.isArray(params.status) ? params.status : [params.status]
      statuses.filter(Boolean).forEach((status) => query.append("status", status))
    }
    if (params?.tags?.length) params.tags.forEach((tag) => query.append("tags", tag))
    if (params?.favorite !== undefined) query.set("favorite", String(params.favorite))
    if (params?.sort) query.set("sort", params.sort)
    if (params?.domain) query.set("domain", params.domain)
    if (params?.date_from) query.set("date_from", params.date_from)
    if (params?.date_to) query.set("date_to", params.date_to)
    const qs = query.toString()
    const path = `/api/v1/reading/items${qs ? `?${qs}` : ""}` as const
    const data = await bgRequest<any>({ path, method: "GET" })
    const items = Array.isArray(data?.items)
      ? data.items.map((item: any) => ({
          id: String(item.id),
          title: item.title || item.url || "Untitled",
          url: item.url,
          canonical_url: item.canonical_url,
          domain: item.domain,
          summary: item.summary ?? undefined,
          notes: item.notes ?? undefined,
          status: item.status ?? "saved",
          favorite: Boolean(item.favorite),
          tags: Array.isArray(item.tags) ? item.tags : [],
          reading_time_minutes: item.reading_time_minutes,
          created_at: item.created_at,
          updated_at: item.updated_at,
          published_at: item.published_at
        }))
      : []
    return {
      ...data,
      items,
      total: data?.total ?? items.length,
      page: data?.page ?? params?.page ?? 1,
      size: data?.size ?? params?.size ?? items.length
    }
  }

  async getReadingItem(itemId: string): Promise<any> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}` as const
    const item = await bgRequest<any>({ path, method: "GET" })
    return {
      ...item,
      id: String(item?.id),
      media_id: item?.media_id ? String(item.media_id) : undefined,
      favorite: Boolean(item?.favorite),
      tags: Array.isArray(item?.tags) ? item.tags : []
    }
  }

  async addReadingItem(data: {
    url: string
    title?: string
    tags?: string[]
    notes?: string
    status?: string
    favorite?: boolean
    summary?: string
    content?: string
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/reading/save",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  }

  async updateReadingItem(
    itemId: string,
    data: {
      status?: string
      favorite?: boolean
      tags?: string[]
      notes?: string
      title?: string
    }
  ): Promise<any> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}` as const
    return await bgRequest<any>({
      path,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  }

  async bulkUpdateReadingItems(data: {
    item_ids: string[]
    action: "set_status" | "set_favorite" | "add_tags" | "remove_tags" | "replace_tags" | "delete"
    status?: string
    favorite?: boolean
    tags?: string[]
    hard?: boolean
  }): Promise<{
    total: number
    succeeded: number
    failed: number
    results: Array<{ item_id: string; success: boolean; error?: string | null }>
  }> {
    const itemIds = (data.item_ids || [])
      .map((id) => Number(id))
      .filter((id) => Number.isFinite(id) && id > 0)
      .map((id) => Math.floor(id))
    if (itemIds.length === 0) {
      throw new Error("item_ids_required")
    }

    const response = await bgRequest<any>({
      path: "/api/v1/reading/items/bulk",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        item_ids: itemIds,
        action: data.action,
        status: data.status,
        favorite: data.favorite,
        tags: data.tags,
        hard: data.hard
      }
    })

    const results = Array.isArray(response?.results)
      ? response.results.map((entry: any) => ({
          item_id: String(entry?.item_id),
          success: Boolean(entry?.success),
          error: entry?.error ?? null
        }))
      : []

    return {
      total: response?.total ?? itemIds.length,
      succeeded: response?.succeeded ?? results.filter((entry) => entry.success).length,
      failed: response?.failed ?? results.filter((entry) => !entry.success).length,
      results
    }
  }

  async deleteReadingItem(itemId: string, options?: { hard?: boolean }): Promise<void> {
    const query = new URLSearchParams()
    if (options?.hard !== undefined) query.set("hard", String(options.hard))
    const qs = query.toString()
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}${qs ? `?${qs}` : ""}` as const
    await bgRequest<void>({ path, method: "DELETE" })
  }

  async createReadingSavedSearch(
    data: CreateReadingSavedSearchRequest
  ): Promise<ReadingSavedSearch> {
    const row = await bgRequest<any>({
      path: "/api/v1/reading/saved-searches",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
    return {
      id: String(row?.id ?? ""),
      name: String(row?.name ?? ""),
      query:
        row?.query && typeof row.query === "object" && !Array.isArray(row.query)
          ? row.query
          : {},
      sort: row?.sort ?? undefined,
      created_at: row?.created_at ?? undefined,
      updated_at: row?.updated_at ?? undefined
    }
  }

  async listReadingSavedSearches(
    params?: { limit?: number; offset?: number }
  ): Promise<ReadingSavedSearchListResponse> {
    const query = new URLSearchParams()
    if (typeof params?.limit === "number" && Number.isFinite(params.limit)) {
      query.set("limit", String(Math.max(1, Math.floor(params.limit))))
    }
    if (typeof params?.offset === "number" && Number.isFinite(params.offset)) {
      query.set("offset", String(Math.max(0, Math.floor(params.offset))))
    }
    const qs = query.toString()
    const path = `/api/v1/reading/saved-searches${qs ? `?${qs}` : ""}` as const
    const data = await bgRequest<any>({ path, method: "GET" })
    const items: ReadingSavedSearch[] = Array.isArray(data?.items)
      ? data.items.map((row: any) => ({
          id: String(row?.id ?? ""),
          name: String(row?.name ?? ""),
          query:
            row?.query && typeof row.query === "object" && !Array.isArray(row.query)
              ? row.query
              : {},
          sort: row?.sort ?? undefined,
          created_at: row?.created_at ?? undefined,
          updated_at: row?.updated_at ?? undefined
        }))
      : []
    return {
      items,
      total: Number.isFinite(data?.total) ? Number(data.total) : items.length,
      limit:
        Number.isFinite(data?.limit) && Number(data.limit) > 0
          ? Number(data.limit)
          : params?.limit ?? 50,
      offset:
        Number.isFinite(data?.offset) && Number(data.offset) >= 0
          ? Number(data.offset)
          : params?.offset ?? 0
    }
  }

  async updateReadingSavedSearch(
    searchId: string,
    data: UpdateReadingSavedSearchRequest
  ): Promise<ReadingSavedSearch> {
    const path = `/api/v1/reading/saved-searches/${encodeURIComponent(searchId)}` as const
    const row = await bgRequest<any>({
      path,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: data
    })
    return {
      id: String(row?.id ?? ""),
      name: String(row?.name ?? ""),
      query:
        row?.query && typeof row.query === "object" && !Array.isArray(row.query)
          ? row.query
          : {},
      sort: row?.sort ?? undefined,
      created_at: row?.created_at ?? undefined,
      updated_at: row?.updated_at ?? undefined
    }
  }

  async deleteReadingSavedSearch(searchId: string): Promise<{ ok: boolean }> {
    const path = `/api/v1/reading/saved-searches/${encodeURIComponent(searchId)}` as const
    const response = await bgRequest<{ ok?: boolean }>({ path, method: "DELETE" })
    return { ok: Boolean(response?.ok) }
  }

  async linkReadingItemToNote(itemId: string, noteId: string): Promise<ReadingNoteLink> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}/links/note` as const
    const row = await bgRequest<any>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { note_id: noteId }
    })
    return {
      item_id: String(row?.item_id ?? itemId),
      note_id: String(row?.note_id ?? noteId),
      created_at: row?.created_at ?? undefined
    }
  }

  async listReadingItemNoteLinks(itemId: string): Promise<ReadingNoteLink[]> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}/links` as const
    const data = await bgRequest<any>({ path, method: "GET" })
    if (!Array.isArray(data?.links)) {
      return []
    }
    return data.links.map((row: any) => ({
      item_id: String(row?.item_id ?? itemId),
      note_id: String(row?.note_id ?? ""),
      created_at: row?.created_at ?? undefined
    }))
  }

  async unlinkReadingItemNote(itemId: string, noteId: string): Promise<{ ok: boolean }> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}/links/note/${encodeURIComponent(noteId)}` as const
    const response = await bgRequest<{ ok?: boolean }>({ path, method: "DELETE" })
    return { ok: Boolean(response?.ok) }
  }

  async summarizeReadingItem(
    itemId: string,
    options?: {
      provider?: string
      model?: string
      prompt?: string
      system_prompt?: string
      temperature?: number
      recursive?: boolean
      chunked?: boolean
    }
  ): Promise<{ summary: string; provider: string; model?: string }> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}/summarize` as const
    return await bgRequest<any>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: options || {}
    })
  }

  async generateReadingItemTts(
    itemId: string,
    options?: { model?: string; voice?: string }
  ): Promise<{ audio_url: string }> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}/tts` as const
    const model = options?.model || (await getTldwTTSModel())
    const voice = options?.voice || (await getTldwTTSVoice())
    const data = await bgRequest<ArrayBuffer>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        model,
        voice,
        response_format: "mp3",
        stream: false,
      },
      responseType: "arrayBuffer"
    })
    const blob = new Blob([data], { type: "audio/mpeg" })
    return { audio_url: URL.createObjectURL(blob) }
  }

  // Highlights
  async getHighlights(itemId: string): Promise<any[]> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}/highlights` as const
    const data = await bgRequest<any>({ path, method: "GET" })
    return Array.isArray(data)
      ? data.map((highlight) => ({
          ...highlight,
          id: String(highlight.id),
          item_id: String(highlight.item_id),
          color: highlight.color || "yellow",
          state: highlight.state || "active",
          anchor_strategy: highlight.anchor_strategy || "fuzzy_quote"
        }))
      : []
  }

  async createHighlight(data: {
    item_id: string
    quote: string
    note?: string
    color?: string
    start_offset?: number
    end_offset?: number
    anchor_strategy?: string
  }): Promise<any> {
    const path = `/api/v1/reading/items/${encodeURIComponent(data.item_id)}/highlight` as const
    const payload = {
      item_id: Number(data.item_id),
      quote: data.quote,
      note: data.note,
      color: data.color,
      start_offset: data.start_offset,
      end_offset: data.end_offset,
      anchor_strategy: data.anchor_strategy
    }
    const highlight = await bgRequest<any>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
    return {
      ...highlight,
      id: String(highlight.id),
      item_id: String(highlight.item_id),
      color: highlight.color || "yellow",
      state: highlight.state || "active",
      anchor_strategy: highlight.anchor_strategy || "fuzzy_quote"
    }
  }

  async updateHighlight(
    highlightId: string,
    data: { note?: string; color?: string; state?: string }
  ): Promise<any> {
    const path = `/api/v1/reading/highlights/${encodeURIComponent(highlightId)}` as const
    const highlight = await bgRequest<any>({
      path,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: data
    })
    return {
      ...highlight,
      id: String(highlight.id),
      item_id: String(highlight.item_id),
      color: highlight.color || "yellow",
      state: highlight.state || "active",
      anchor_strategy: highlight.anchor_strategy || "fuzzy_quote"
    }
  }

  async deleteHighlight(highlightId: string): Promise<void> {
    const path = `/api/v1/reading/highlights/${encodeURIComponent(highlightId)}` as const
    await bgRequest<void>({ path, method: "DELETE" })
  }

  // Output Templates
  async getOutputTemplates(params?: {
    q?: string
    limit?: number
    offset?: number
  }): Promise<any> {
    const query = new URLSearchParams()
    if (params?.q) query.set("q", params.q)
    if (params?.limit) query.set("limit", String(params.limit))
    if (params?.offset !== undefined) query.set("offset", String(params.offset))
    const qs = query.toString()
    const path = `/api/v1/outputs/templates${qs ? `?${qs}` : ""}` as const
    const data = await bgRequest<any>({ path, method: "GET" })
    const items = Array.isArray(data?.items)
      ? data.items.map((template: any) => ({
          ...template,
          id: String(template.id)
        }))
      : []
    return {
      ...data,
      items,
      total: data?.total ?? items.length
    }
  }

  async createOutputTemplate(data: {
    name: string
    description?: string
    type: string
    format: string
    body: string
    is_default?: boolean
  }): Promise<any> {
    const template = await bgRequest<any>({
      path: "/api/v1/outputs/templates",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
    return { ...template, id: String(template.id) }
  }

  async updateOutputTemplate(
    templateId: string,
    data: {
      name?: string
      description?: string
      body?: string
      is_default?: boolean
      type?: string
      format?: string
    }
  ): Promise<any> {
    const path = `/api/v1/outputs/templates/${encodeURIComponent(templateId)}` as const
    const template = await bgRequest<any>({
      path,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: data
    })
    return { ...template, id: String(template.id) }
  }

  async deleteOutputTemplate(templateId: string): Promise<void> {
    const path = `/api/v1/outputs/templates/${encodeURIComponent(templateId)}` as const
    await bgRequest<void>({ path, method: "DELETE" })
  }

  async previewTemplate(data: {
    template_id: string
    item_ids?: string[]
    run_id?: string
    limit?: number
    data?: Record<string, unknown>
  }): Promise<{ rendered: string; format: string }> {
    const path = `/api/v1/outputs/templates/${encodeURIComponent(data.template_id)}/preview` as const
    return await bgRequest<any>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        template_id: Number(data.template_id),
        item_ids: data.item_ids?.map((id) => Number(id)),
        run_id: data.run_id ? Number(data.run_id) : undefined,
        limit: data.limit,
        data: data.data
      }
    })
  }

  async listOutputs(params?: {
    page?: number
    size?: number
    job_id?: number
    run_id?: number
    type?: string
    workspace_tag?: string
    include_deleted?: boolean
  }): Promise<{ items: any[]; total: number; page?: number; size?: number }> {
    const query = this.buildQuery(params as Record<string, any>)
    const response = await bgRequest<any>({
      path: `/api/v1/outputs${query}`,
      method: "GET"
    })
    const items = Array.isArray(response?.items)
      ? response.items.map((item: any) => ({
          ...item,
          id: String(item.id),
          media_item_id:
            item.media_item_id === null || typeof item.media_item_id === "undefined"
              ? undefined
              : String(item.media_item_id)
        }))
      : []
    return {
      ...response,
      items,
      total: response?.total ?? items.length
    }
  }

  async generateOutput(data: {
    template_id: string
    item_ids?: string[]
    run_id?: string
    title?: string
    workspace_tag?: string
    data?: Record<string, unknown>
  }): Promise<any> {
    const output = await bgRequest<any>({
      path: "/api/v1/outputs",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        template_id: Number(data.template_id),
        item_ids: data.item_ids?.map((id) => Number(id)),
        run_id: data.run_id ? Number(data.run_id) : undefined,
        title: data.title,
        workspace_tag: data.workspace_tag,
        data: data.data
      }
    })
    return { ...output, id: String(output.id) }
  }

  async downloadOutput(outputId: string, format?: string): Promise<Blob> {
    const path = `/api/v1/outputs/${encodeURIComponent(outputId)}/download` as const
    const data = await bgRequest<ArrayBuffer>({
      path,
      method: "GET",
      responseType: "arrayBuffer"
    })
    const mime =
      format === "html"
        ? "text/html"
        : format === "md"
          ? "text/markdown"
          : format === "mp3"
            ? "audio/mpeg"
            : "application/octet-stream"
    return new Blob([data], { type: mime })
  }

  // Import/Export
  async importReadingList(data: {
    source: ImportSource
    file: File
    merge_tags?: boolean
  }): Promise<ReadingImportJobResponse> {
    const buffer = await data.file.arrayBuffer()
    const fileData = Array.from(new Uint8Array(buffer))
    return await this.upload<ReadingImportJobResponse>({
      path: "/api/v1/reading/import",
      method: "POST",
      fileFieldName: "file",
      file: {
        name: data.file.name,
        type: data.file.type || "application/octet-stream",
        data: fileData
      },
      fields: {
        source: data.source,
        merge_tags: data.merge_tags ?? true
      }
    })
  }

  async listReadingImportJobs(params?: {
    status?: string
    limit?: number
    offset?: number
  }): Promise<ReadingImportJobsListResponse> {
    const query = this.buildQuery(params as Record<string, any>)
    return await bgRequest<ReadingImportJobsListResponse>({
      path: `/api/v1/reading/import/jobs${query}`,
      method: "GET"
    })
  }

  async getReadingImportJob(job_id: number | string): Promise<ReadingImportJobStatus> {
    const id = String(job_id)
    return await bgRequest<ReadingImportJobStatus>({
      path: `/api/v1/reading/import/jobs/${id}`,
      method: "GET"
    })
  }

  async exportReadingList(params: {
    format: string
    status?: string[]
    tags?: string[]
    favorite?: boolean
    q?: string
    domain?: string
    page?: number
    size?: number
    include_highlights?: boolean
    include_notes?: boolean
  }): Promise<{ blob: Blob; filename: string }> {
    const query = new URLSearchParams()
    query.set("format", params.format)
    if (params?.status?.length) params.status.forEach((status) => query.append("status", status))
    if (params?.tags?.length) params.tags.forEach((tag) => query.append("tags", tag))
    if (params?.favorite !== undefined) query.set("favorite", String(params.favorite))
    if (params?.q) query.set("q", params.q)
    if (params?.domain) query.set("domain", params.domain)
    if (params?.page) query.set("page", String(params.page))
    if (params?.size) query.set("size", String(params.size))
    if (params?.include_highlights !== undefined) {
      query.set("include_highlights", String(params.include_highlights))
    }
    if (params?.include_notes !== undefined) {
      query.set("include_notes", String(params.include_notes))
    }
    const qs = query.toString()
    const path = `/api/v1/reading/export${qs ? `?${qs}` : ""}` as const
    const response = await bgRequest<any>({
      path,
      method: "GET",
      responseType: "arrayBuffer",
      returnResponse: true
    })
    if (!response) {
      throw new Error("Export failed")
    }
    if (!response.ok) {
      const msg = response.error || `Export failed: ${response.status}`
      throw new Error(msg)
    }
    const headers = new Headers(response.headers || {})
    const contentDisposition = headers.get("content-disposition") || ""
    const filenameMatch = /filename="?([^"]+)"?/i.exec(contentDisposition)
    const filename = filenameMatch?.[1] || "reading_export.jsonl"
    const blob = new Blob([response.data], { type: headers.get("content-type") || "application/octet-stream" })
    return { blob, filename }
  }

  async createReadingDigestSchedule(
    data: CreateReadingDigestScheduleRequest
  ): Promise<{ id: string }> {
    const payload: Record<string, unknown> = { ...data }
    if (!payload.format) payload.format = "md"
    if (typeof payload.enabled !== "boolean") payload.enabled = true
    if (typeof payload.require_online !== "boolean") payload.require_online = false
    const response = await bgRequest<{ id: string }>({
      path: "/api/v1/reading/digests/schedules",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
    return { id: String(response?.id ?? "") }
  }

  async listReadingDigestSchedules(params?: {
    limit?: number
    offset?: number
  }): Promise<ReadingDigestSchedule[]> {
    const query = new URLSearchParams()
    if (typeof params?.limit === "number" && Number.isFinite(params.limit)) {
      query.set("limit", String(Math.max(1, Math.floor(params.limit))))
    }
    if (typeof params?.offset === "number" && Number.isFinite(params.offset)) {
      query.set("offset", String(Math.max(0, Math.floor(params.offset))))
    }
    const qs = query.toString()
    const path = `/api/v1/reading/digests/schedules${qs ? `?${qs}` : ""}` as const
    const rows = await bgRequest<any>({ path, method: "GET" })
    if (!Array.isArray(rows)) {
      return []
    }
    return rows.map((row) => normalizeReadingDigestSchedule(row))
  }

  async getReadingDigestSchedule(scheduleId: string): Promise<ReadingDigestSchedule> {
    const path = `/api/v1/reading/digests/schedules/${encodeURIComponent(scheduleId)}` as const
    const schedule = await bgRequest<any>({ path, method: "GET" })
    return normalizeReadingDigestSchedule(schedule)
  }

  async updateReadingDigestSchedule(
    scheduleId: string,
    data: UpdateReadingDigestScheduleRequest
  ): Promise<ReadingDigestSchedule> {
    const path = `/api/v1/reading/digests/schedules/${encodeURIComponent(scheduleId)}` as const
    const schedule = await bgRequest<any>({
      path,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: data
    })
    return normalizeReadingDigestSchedule(schedule)
  }

  async deleteReadingDigestSchedule(scheduleId: string): Promise<{ ok: boolean }> {
    const path = `/api/v1/reading/digests/schedules/${encodeURIComponent(scheduleId)}` as const
    const response = await bgRequest<{ ok?: boolean }>({ path, method: "DELETE" })
    return { ok: Boolean(response?.ok) }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Slides / Presentations API
  // ─────────────────────────────────────────────────────────────────────────────

}

// ─────────────────────────────────────────────────────────────────────────────
// Domain Method Mixins
// ─────────────────────────────────────────────────────────────────────────────

import { adminMethods } from "./domains/admin"
import { mediaMethods } from "./domains/media"
import { characterMethods } from "./domains/characters"
import { chatRagMethods } from "./domains/chat-rag"
import { collectionsMethods } from "./domains/collections"
import { modelsAudioMethods } from "./domains/models-audio"
import { presentationsMethods } from "./domains/presentations"
import { workspaceApiMethods } from "./domains/workspace-api"
import { webClipperMethods } from "./domains/web-clipper"

export class TldwApiClient extends TldwApiClientBase {}

// Declaration merging: extend the class type with all domain methods
export interface TldwApiClient
  extends
    Omit<typeof adminMethods, never>,
    Omit<typeof mediaMethods, never>,
    Omit<typeof characterMethods, never>,
    Omit<typeof chatRagMethods, never>,
    Omit<typeof collectionsMethods, never>,
    Omit<typeof modelsAudioMethods, never>,
    Omit<typeof presentationsMethods, never>,
    Omit<typeof workspaceApiMethods, never>,
    Omit<typeof webClipperMethods, never> {}

// Apply domain methods to the prototype
Object.assign(
  TldwApiClient.prototype,
  adminMethods,
  mediaMethods,
  characterMethods,
  chatRagMethods,
  collectionsMethods,
  modelsAudioMethods,
  presentationsMethods,
  workspaceApiMethods,
  webClipperMethods
)

// Also expose core helpers that domain files reference via `this`
export type TldwApiClientCore = TldwApiClient

// Singleton instance
export const tldwClient = new TldwApiClient()
