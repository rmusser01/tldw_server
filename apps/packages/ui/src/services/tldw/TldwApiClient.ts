import type {
  RoleplayResumeMetadata,
  RoleplayResumeUserAppendRequest,
  RoleplayResumeUserAppendReceipt
} from "./roleplay-resume"
import type { InitialChatSetupAttempt } from "@/utils/settings-return"
import type { LlamacppRequestOptions } from "./domains/models-audio"
import { knowledgeNoteWriteFields, retainKnowledgeNoteProvenance } from "@/utils/knowledge-note-provenance"
import { BoundedTtlCache } from "./bounded-ttl-cache"
import type { HistoryAdmissionV1 } from "@/types/history-selection"
import type { ChatScope } from "@/types/chat-scope"
import { clearFlashcardsGenerateHandoffs } from "@/services/tldw/flashcards-generate-handoff"
import { toChatScopeParams } from "@/types/chat-scope"
import type {
  LlamacppSnapshotSlotsResponse,
  LlamacppSnapshotCatalogResponse,
  LlamacppSnapshotOperationResponse,
  LlamacppSnapshotRequest,
  LlamacppAsset,
  LlamacppAssetsResponse,
  LlamacppConfigResponse,
  LlamacppConfigUpdateRequest,
  LlamacppHardwareSnapshotResponse,
  LlamacppInventoryItem,
  LlamacppInventoryResponse,
  LlamacppLifecycleActionResponse,
  LlamacppLogTailResponse,
  LlamacppProfile,
  LlamacppProfileCreateRequest,
  LlamacppProfileListResponse,
  LlamacppProfileUpdateRequest,
  LlamacppRuntimeListResponse,
  LlamacppUseInChatResponse,
  LlamacppValidationRequest,
  LlamacppValidationResponse
} from "@/types/llamacpp-admin"
import { Storage } from "@plasmohq/storage"
import { createSafeStorage, safeStorageSerde } from "@/utils/safe-storage"
import { bgRequest, bgStream, bgUpload } from "@/services/background-proxy"
import { requestOwnedPrompt } from "@/services/prompt-owner-request"
import { isPlaceholderApiKey } from "@/utils/api-key"
import { normalizeChatRole } from "@/utils/normalize-chat-role"
import { createJsonResponseLike } from "@/services/tldw/json-response-like"
import type { AllowedPath, PathOrUrl } from "@/services/tldw/openapi-guard"
import { tldwRequest } from "@/services/tldw/request-core"
import { createServicePromptScopeChangedError, isRequestConfigScopeChangedError, servicePromptTargetsMatch } from "@/services/tldw/service-prompt-scope-error"
import { connectionAuthoritiesMatch } from "@/services/chat-surface-scope"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { appendPathQuery } from "@/services/tldw/path-utils"
import { inferUploadMediaTypeFromUrl } from "@/services/tldw/media-routing"
import {
  captureChatRequestDebugSnapshot,
  type ChatRequestDebugMetadata
} from "@/services/tldw/chat-request-debug"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import {
  cancelRuntimeAuthBootstrapRecovery,
  recoverRuntimeAuthBootstrap
} from "./runtime-bootstrap-recovery"

import { getNormalizedTldwModels } from "@/services/tldw/model-normalization"
import { getTldwTTSModel, getTldwTTSVoice } from "@/services/tts"
import {
  DEFAULT_CHARACTER_PROFILE_PREFERENCE_KEY,
  normalizeDefaultCharacterPreferenceId
} from "@/utils/default-character-preference"
import type { PersonaBuddySummary } from "@/types/persona-buddy"
import {
  COOKIE_SESSION_CONFIG_KEY,
  isCookieSessionBrowserTransport,
  resolveWebUiQuickstartServerUrl,
  type BrowserSurface
} from "@/services/tldw/browser-networking"
import {
  getRuntimeSingleUserApiKeyOverride,
  invalidateCookieSessionConfig,
  isCookieSessionConfigInvalidated
} from "@/services/tldw/runtime-auth-override"
import type {
  ApiKeyPersistence,
  CredentialSource,
  ManualSessionCredential
} from "@/services/tldw/single-user-credential"
import {
  clearManualCredentials,
  getBuildTimeApiKey,
  hasNewerCurrentAccessToken,
  hasInvalidatedRefreshSession,
  invalidateRefreshSessionIfCurrent,
  isCompleteDeviceCredential,
  MANUAL_SESSION_KEY,
  REFRESH_ROTATION_KEY,
  normalizeServerOrigin,
  resolveEffectiveTldwConfig,
  resolveManualCredential,
  storeRefreshRotationIfCurrent,
  toPersistedTldwConfig
} from "@/services/tldw/single-user-credential"
import {
  type TldwProvidersResponse
} from "@/services/tldw/model-provider-availability"
import {
  chatModelOutcomeTarget,
  trackChatModelRequestOutcome
} from "@/services/tldw/chat-model-outcome"
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
  AudioPreset,
  AudioPresetCreatePayload,
  AudioPresetKind,
  AudioPresetListResponse,
  AudioPresetUpdatePayload,
  AudioPresetValidationResponse
} from "@/types/audio-presets"
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
// Speech synthesis can take much longer than the 10s default request timeout
// (e.g. long passages on local Kokoro), so give it a generous default that
// callers can still override.
const TTS_REQUEST_TIMEOUT_MS = 120000
const MANUAL_SESSION_CREDENTIAL_LOCK = "tldw:manual-session-credential"

const hasManualSessionCredentialLock = (): boolean =>
  typeof navigator !== "undefined" && Boolean(navigator.locks?.request)

const withManualSessionCredentialLock = async <T>(
  operation: () => Promise<T>
): Promise<T> => {
  if (!hasManualSessionCredentialLock()) return await operation()
  return await navigator.locks.request(MANUAL_SESSION_CREDENTIAL_LOCK, operation)
}

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
  apiBearer?: string
  accessToken?: string
  refreshToken?: string
  orgId?: number
  authMode: 'single-user' | 'multi-user'
  authSource?: "manual" | "cookie-session"
  credentialSource?: CredentialSource
  apiKeyPersistence?: ApiKeyPersistence
  apiKeyServerOrigin?: string
}

export type ServicePromptTargetConfig = Readonly<
  Pick<
    TldwConfig,
    "serverUrl" | "authMode" | "authSource" | "orgId"
  > & {
    expectedUserId?: string | number | null
    expectedRefreshToken?: string
    expectedSingleUserApiKeyScope?: string
  }
>

export type ExplainerMode = "goal" | "sources"
export type ExplainerOutputIntent = "explain" | "plan" | "both"
export type ExplainerGrounding = "source_only" | "source_led" | "open"
export type ExplainerDepthPreset = "quick" | "standard" | "deep"
export type ExplainerNodeKind = "question" | "answer" | "explanation" | "step" | "summary"
export type ExplainerNodeStatus = "idle" | "queued" | "generating" | "error" | "complete"
export type ExplainerEvidenceState =
  | "supported"
  | "partially_supported"
  | "uncited"
  | "insufficient"

export interface ExplainerSelectedSourcePayload {
  sourceId: string
  sourceType: string
  title: string
  snapshotVersion?: string | null
  metadata?: Record<string, unknown> | null
}

export interface ExplainerSessionCreatePayload {
  title: string
  mode: ExplainerMode
  outputIntent: ExplainerOutputIntent
  grounding: ExplainerGrounding
  depthPreset: ExplainerDepthPreset
  selectedSources: ExplainerSelectedSourcePayload[]
  rootPrompt: string
}

export interface ExplainerSessionPatchPayload {
  title?: string
  outputIntent?: ExplainerOutputIntent
  grounding?: ExplainerGrounding
  depthPreset?: ExplainerDepthPreset
  selectedSources?: ExplainerSelectedSourcePayload[]
}

export interface ExplainerNodeCreatePayload {
  parentId?: string | null
  title: string
  body?: string | null
  kind?: ExplainerNodeKind
  intent?: ExplainerOutputIntent
  status?: ExplainerNodeStatus
  evidenceState?: ExplainerEvidenceState
  outsideKnowledgeUsed?: boolean
  citations?: Record<string, unknown>[]
}

export interface ExplainerNodePatchPayload {
  title?: string | null
  body?: string | null
  status?: ExplainerNodeStatus
  evidenceState?: ExplainerEvidenceState
  outsideKnowledgeUsed?: boolean
  selectedOptionId?: string | null
  selectedCustomAnswer?: string | null
  questionOptions?: Record<string, unknown>[] | null
  generationMetadata?: Record<string, unknown> | null
  citations?: Record<string, unknown>[] | null
}

export interface ExplainerNodeExpandPayload {
  intent?: ExplainerOutputIntent | null
}

export interface ExplainerQuestionAnswerPayload {
  selectedOptionId?: string | null
  selectedCustomAnswer?: string | null
}

export interface ExplainerChatbookExportPayload {
  name?: string | null
  description?: string | null
  asyncMode?: boolean
}

export interface ChatbookAccountScopeCategory {
  category: string
  label: string
  count: number
  restore_status: "restorable" | "pointer_only" | "non_restorable"
  sensitivity: "public" | "personal" | "sensitive" | "secret"
  warning?: string | null
}

export interface ChatbookAccountScopeResponse {
  mode: "full_account"
  categories: ChatbookAccountScopeCategory[]
  total_items: number
  pointer_only_count: number
  sensitive_category_count: number
  warning_count: number
  estimated_size_bytes?: number | null
}

export interface CurrentUserStorageQuotaResponse {
  user_id: number
  storage_used_mb: number
  storage_quota_mb: number | null
  available_mb: number | null
  usage_percentage: number | null
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

export type OpenWebUIHydrationScopeRequest = {
  import_scope_id?: string | null
  conversation_ids?: string[]
  source_user_id?: string | null
}

export type OpenWebUIHydrationRequest = {
  openwebui_data_root: string
  scope?: OpenWebUIHydrationScopeRequest
  process_supported_files?: boolean
}

export type OpenWebUIImportScopeConversationSummary = {
  source_conversation_id?: string | null
  conversation_id: string
  title: string
  attachment_reference_count: number
}

export type OpenWebUIImportScopeSummary = {
  scope_id: string
  source_format: string
  source_user_id?: string | null
  source_user_label?: string | null
  conversation_count: number
  attachment_reference_count: number
  created_at?: string | null
  conversation_ids?: string[]
  conversations?: OpenWebUIImportScopeConversationSummary[]
}

export type OpenWebUIImportScopesResponse = {
  scopes: OpenWebUIImportScopeSummary[]
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

export const isQuickstartWebUiSameOriginServerUrl = (serverUrl: string): boolean => {
  const quickstartUrl = getQuickstartWebUiServerUrl()
  if (!quickstartUrl) return false
  try {
    const server = new URL(serverUrl)
    const quickstart = new URL(quickstartUrl)
    const serverPath = server.pathname.replace(/\/+$/, "")
    return server.origin === quickstart.origin && serverPath === ""
  } catch {
    return false
  }
}

export const isActiveCookieSessionConfig = (
  config: TldwConfig | null | undefined,
  quickstartWebUiServerUrl = getQuickstartWebUiServerUrl()
): boolean =>
  Boolean(
    quickstartWebUiServerUrl &&
      config?.serverUrl === quickstartWebUiServerUrl &&
      isCookieSessionBrowserTransport({
        authMode: config?.authMode,
        authSource: config?.authSource,
        transportMode: "quickstart",
        transportKind: "same-origin",
        pageOrigin: quickstartWebUiServerUrl
      })
  )

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

export type PresentationStudioRecordBase = {
  id: string
  title: string
  description?: string | null
  theme: string
  source_type?: string | null
  source_ref?: unknown
  source_query?: string | null
  created_at: string
  last_modified: string
  deleted?: boolean
  client_id?: string
  version: number
}

export type StructuredPresentationStudioRecord = PresentationStudioRecordBase & {
  content_kind: "structured_slides"
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
}

export type StandaloneHtmlPresentationStudioRecord = PresentationStudioRecordBase & {
  content_kind: "standalone_html"
  html_document: string
  html_sha256: string
  html_bytes: number
  html_slide_count: number
  generation_provenance: Record<string, unknown>
}

export type UnsupportedPresentationStudioRecord = PresentationStudioRecordBase & {
  content_kind: "unsupported"
  unsupported_content_kind: string | null
  read_only: true
}

export type PresentationStudioRecord =
  | StructuredPresentationStudioRecord
  | StandaloneHtmlPresentationStudioRecord
  | UnsupportedPresentationStudioRecord

export type PresentationDetailResult = {
  record: PresentationStudioRecord
  etag: string | null
}

export type StandalonePresentationDetailResult = {
  record: StandaloneHtmlPresentationStudioRecord
  etag: string | null
}

export type PresentationProvenanceSummary = {
  source_kind: string | null
  provider: string | null
  model: string | null
}

export type PresentationSummaryBase = {
  id: string
  title: string
  description: string | null
  theme: string
  created_at: string
  last_modified: string
  deleted: boolean
  version: number
  provenance: PresentationProvenanceSummary
}

export type StructuredPresentationSummary = PresentationSummaryBase & {
  content_kind: "structured_slides"
  slide_count: number
}

export type StandaloneHtmlPresentationSummary = PresentationSummaryBase & {
  content_kind: "standalone_html"
  html_slide_count: number
  html_bytes: number
}

export type UnsupportedPresentationSummary = PresentationSummaryBase & {
  content_kind: "unsupported"
  unsupported_content_kind: string | null
  read_only: true
}

export type PresentationSummary =
  StructuredPresentationSummary | StandaloneHtmlPresentationSummary | UnsupportedPresentationSummary

export type PresentationListResponse = {
  presentations: PresentationSummary[]
  total: number
  limit: number
  offset: number
  pagination: {
    mode: "offset"
    limit: number
    offset: number
    total: number
    has_more: boolean
    next_offset: number | null
  }
  has_more: boolean | null
  next_offset: number | null
}

export type PresentationMetadataResult = {
  record: PresentationSummary
  etag: string | null
}

export type StandaloneHtmlContentCapabilityReason = "validator_unavailable"
export type StandaloneHtmlGenerationCapabilityReason =
  | "feature_disabled"
  | "egress_disabled"
  | "default_model_not_configured"
  | "default_model_not_allowed"
  | "default_endpoint_not_allowed"
  | "prompt_asset_unavailable"
  | "digest_key_unavailable"
  | "generation_worker_unavailable"
  | "generation_reconciler_overloaded"
  | "validator_unavailable"

type StandaloneHtmlContentCapability = {
  read: true
  draft_attachment: true
  limits: {
    max_document_bytes: number
    max_source_write_bytes: number
    max_draft_attachment_bytes: number
    max_slides: number
    max_nesting_depth: number
  }
} & (
  | {
      edit: true
      export_attachment: true
      reason: null
    }
  | {
      edit: false
      export_attachment: false
      reason: StandaloneHtmlContentCapabilityReason
    }
)

type StandaloneHtmlGenerationCapability = {
  transport: "slides_generation_job"
  source_kinds: ["prompt", "chat", "media", "notes", "rag"]
  input_limits: {
    max_request_bytes: number
    max_source_chars: number
    max_source_tokens: number
    max_audience_chars: number
    max_source_identifier_bytes: number
    max_note_ids: number
    max_rag_query_chars: number
    max_rag_top_k: number
  }
  output_limits: {
    max_provider_response_bytes: number
    max_document_bytes: number
  }
} & (
  | {
      enabled: true
      reason: null
      provider: string
      model: string
      adapter_id: string
      endpoint_identity: string
      generation_config_revision: `sha256:${string}`
    }
  | {
      enabled: false
      reason: StandaloneHtmlGenerationCapabilityReason
      provider: null
      model: null
      adapter_id: null
      endpoint_identity: null
      generation_config_revision: null
    }
)

export type SlidesCapabilities = {
  schema_version: 1
  content_kind_request_header: "X-Slides-Accept-Content-Kinds"
  content_kinds: {
    structured_slides: { read: true; edit: true }
    standalone_html: StandaloneHtmlContentCapability
  }
  generation_modes: {
    structured_slides: {
      enabled: true
      transport: "existing_source_endpoints"
    }
    standalone_html: StandaloneHtmlGenerationCapability
  }
}

export type PresentationGenerationSource =
  | { kind: "prompt"; prompt: string }
  | { kind: "chat"; conversation_id: string }
  | { kind: "media"; media_id: number }
  | { kind: "notes"; note_ids: string[] }
  | { kind: "rag"; query: string; top_k?: number }

export type PresentationGenerationRequest = {
  generation_mode: "standalone_html"
  generation_config_revision: string
  source: PresentationGenerationSource
  html_options: {
    presentation_type:
      | "pitch-deck"
      | "tech-sharing"
      | "product-launch"
      | "weekly-report"
      | "course-module"
      | "keynote"
      | "data-report"
      | "training"
      | "social-media"
      | "case-study"
      | "comparison"
      | "roadmap"
    audience: string
    slide_count: number
    visual_direction:
      | "auto"
      | "dark-technical"
      | "minimal-light"
      | "editorial"
      | "corporate"
      | "soft-pastel"
      | "bold-creative"
      | "neo-brutalist"
    delivery_style: "speaker-led" | "self-guided"
  }
}

type PresentationGenerationReceiptBase = {
  generation_id: string
  status_url: string
}

export type PresentationGenerationReceipt =
  | (PresentationGenerationReceiptBase & {
      status: "queued" | "running"
      presentation_id: null
      progress_text?: string | null
    })
  | (PresentationGenerationReceiptBase & {
      status: "completed"
      presentation_id: string | null
      content_kind: "standalone_html"
    })
  | (PresentationGenerationReceiptBase & {
      status: "failed"
      presentation_id: null
      error_code: string
      error_message: string
    })
  | (PresentationGenerationReceiptBase & {
      status: "cancelled"
      presentation_id: null
      error_code: "generation_cancelled"
    })

export type PresentationGenerationStatusResult = {
  receipt: PresentationGenerationReceipt
  retryAfterMs: number | null
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
  is_server_default?: boolean
  name: string
  provider: string
  description?: string
  capabilities?: string[] | Record<string, unknown>
  context_length?: number
  vision?: boolean
  function_calling?: boolean
  json_output?: boolean
  type?: string
  is_configured?: boolean
  provider_is_configured?: boolean
  provider_enabled?: boolean
  availability?: string
  readiness_reason_code?: string
  readiness_message?: string
  chat_provider?: string
  /** Server-observed provider reachability: "working", "unreachable" or "unknown". */
  provider_health?: string
  /**
   * Optional sampling fields the provider accepts ("top_k", "min_p"). Absent
   * when the server doesn't say; then neither is sent (#3283).
   */
  sampling_fields?: string[]
  catalog_only?: boolean
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
  tldw_history_selection_v1?: import("@/types/history-selection").HistorySelectionV1
  tldw_history_branch?: boolean
  tldw_history_edit_v1?: import("@/types/history-selection").HistoryEditTargetV1
  tldw_history_server_existing_input_v1?: import("@/types/history-selection").HistorySelectionV1
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
  /** Top-K sampling, sent only to providers that accept it (#3283). */
  topk?: number
  /** Min-P sampling, sent only to providers that accept it (#3283). */
  minp?: number
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

export type ScopedRequestOptions = {
  idempotencyKey?: string
  signal?: AbortSignal
  requestScope?: ServicePromptRequestScope
}

export type ChatCompletionRequestOptions = {
  scope?: ChatScope
  signal?: AbortSignal
  timeoutMs?: number
  debugMetadata?: ChatRequestDebugMetadata
  requestScope?: ServicePromptRequestScope
}

export type ChatCompletionStreamOptions = ChatCompletionRequestOptions & {
  onDispatch?: () => void
  streamIdleTimeoutMs?: number
}

export interface ServerChatSummary extends RoleplayResumeMetadata {
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
  character_name?: string | null
  assistant_kind?: "character" | "persona" | null
  assistant_id?: string | number | null
  assistant_name?: string | null
  persona_memory_mode?: "read_only" | "read_write" | null
  parent_conversation_id?: string | null
  root_id?: string | null
  forked_from_message_id?: string | null
  version?: number | null
  history_version?: number | null
  scope_type?: "global" | "workspace" | null
  workspace_id?: string | null
  /** Where the search query matched. Set only by a search that includes message content. */
  matched_in?: Array<"title" | "content"> | null
  /** Plain-text excerpt of the message that matched the search query. */
  match_snippet?: string | null
  /** ID of the message that match_snippet is taken from. */
  match_message_id?: string | null
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
  tldw_history_admission_v1?: HistoryAdmissionV1
  parent_message_id?: string | null
  images?: string[]
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
  storage_quota_mb: number | null
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
  reason?: string
  admin_password?: string
  admin_reauth_token?: string
  email?: string
  role?: string
  is_active?: boolean
  is_verified?: boolean
  is_locked?: boolean
  storage_quota_mb?: number | null
}

export interface AdminUserCreateRequest {
  username: string
  email: string
  password: string
  role: 'user' | 'admin'
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

export interface TtsProviderUnloadResponse {
  provider: string
  unloaded: boolean
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

export interface SandboxAdminRuntimeDiagnosticsSummary {
  total: number
  ready: number
  unavailable: number
  host_gated: number
  scaffold: number
  host_local_warning_runtimes?: string[]
  repair_supported_runtimes?: string[]
}

export interface SandboxAdminRuntimeDiagnosticsItem {
  name: string
  available: boolean
  implementation_state?: string | null
  readiness: string
  reasons?: string[]
  normalized_reasons?: string[]
  boundary_class?: string | null
  vm_grade_isolation?: boolean
  untrusted_eligible?: boolean
  isolation_warnings?: string[]
  strict_deny_all_supported?: boolean
  strict_allowlist_supported?: boolean
  session_reuse_model?: string | null
  requires_live_health_check?: boolean
  repair_supported?: boolean
  recommended_action?: string
}

export interface SandboxAdminStartupWarningSummary {
  present: boolean
  blocking: boolean
  codes?: string[]
}

export interface SandboxAdminRuntimeDiagnosticsResponse {
  source: "feature_discovery"
  summary: SandboxAdminRuntimeDiagnosticsSummary
  runtimes: SandboxAdminRuntimeDiagnosticsItem[]
  startup_warning_summary?: SandboxAdminStartupWarningSummary | null
}

export type SandboxWorkspaceDiagnosticStateValue =
  | "available"
  | "not_configured"
  | "unavailable"
  | "blocked"
  | "unknown"

export interface SandboxWorkspaceDiagnosticState {
  state: SandboxWorkspaceDiagnosticStateValue
  reason_code?: string | null
  message: string
  management_surface?: string | null
}

export interface SandboxWorkspaceDiagnosticsRunSummary {
  id: string
  runtime?: string | null
  runtime_version?: string | null
  base_image?: string | null
  phase: string
  status_reason_code?: string | null
  status_reason_details?: Record<string, unknown> | null
  exit_code?: number | null
  started_at?: string | null
  finished_at?: string | null
  message?: string | null
  session_id?: string | null
  persona_id?: string | null
  workspace_id?: string | null
  workspace_group_id?: string | null
  scope_snapshot_id?: string | null
}

export interface SandboxWorkspaceDiagnosticsRunList {
  total: number
  limit: number
  has_more: boolean
  items: SandboxWorkspaceDiagnosticsRunSummary[]
}

export interface SandboxWorkspaceDiagnosticsLinks {
  runtime_config?: string | null
  admin_runs?: string | null
}

export interface SandboxWorkspaceDiagnosticsResponse {
  workspace_id: string
  source_label: "research_workspace"
  runtime: SandboxWorkspaceDiagnosticState
  admission: SandboxWorkspaceDiagnosticState
  runs: SandboxWorkspaceDiagnosticsRunList
  links: SandboxWorkspaceDiagnosticsLinks
}

// Cookie-session owners can change without changing the connection config.
// One watcher fences all client instances without retaining each instance.
let domainCacheAccountRevision = 0
const stopWatchingDomainCacheAccount = watchChatAccountChanges(invalidated => {
  if (invalidated) domainCacheAccountRevision += 1
})
const domainCacheHot = (import.meta as { hot?: { dispose: (callback: () => void) => void } }).hot
domainCacheHot?.dispose(stopWatchingDomainCacheAccount)

type TldwProvidersStatus = {
  providers: Array<{
    name: string
    configured: boolean
    requires_api_key: boolean
    key_hint?: string | null
    key_source?: string | null
  }>
  any_configured: boolean
}

// getProvidersStatus is mounted-on by 12+ components independently; this
// module-level TTL cache + in-flight dedup (mirroring the apiSend pattern)
// collapses the resulting burst into one request per TTL window.
const PROVIDERS_STATUS_CACHE_TTL_MS = 60_000
let providersStatusCache: { revision: number; at: number; value: TldwProvidersStatus } | null = null
let providersStatusInFlight: { revision: number; request: Promise<TldwProvidersStatus> } | null = null

const getProvidersStatusCached = async (): Promise<TldwProvidersStatus> => {
  const revision = domainCacheAccountRevision
  const now = Date.now()
  if (
    providersStatusCache &&
    providersStatusCache.revision === revision &&
    now - providersStatusCache.at < PROVIDERS_STATUS_CACHE_TTL_MS
  ) {
    return providersStatusCache.value
  }
  if (providersStatusInFlight?.revision === revision) {
    return providersStatusInFlight.request
  }
  const request = (async () => {
    try {
      const value = (await bgRequest<any>({
        path: '/api/v1/config/providers',
        method: 'GET'
      })) as TldwProvidersStatus
      if (revision !== domainCacheAccountRevision)
        throw createServicePromptScopeChangedError()
      providersStatusCache = { revision, at: Date.now(), value }
      return value
    } finally {
      if (providersStatusInFlight?.request === request) {
        providersStatusInFlight = null
      }
    }
  })()
  providersStatusInFlight = { revision, request }
  return request
}

export class TldwApiClientBase {
  private storage: Storage
  private sessionStorage: Storage
  private config: TldwConfig | null = null
  private configInitialization = 0
  private completedConfigInitialization = 0
  private baseUrl: string = ''
  private headers: HeadersInit = {}
  characterCache = new BoundedTtlCache<any>()
  characterInFlight = new Map<string, Promise<any>>()
  // Keys are `${chatId}${query}` (see getChatMessagesCacheKey), so the query
  // shape's chat is everything before the first "?". ~5 shapes per chat
  // (WP-31 / TASK-13523) on top of the cache-wide entry/byte caps.
  chatMessagesCache = new BoundedTtlCache<ServerChatMessage[]>({
    maxPerGroup: 5,
    groupOf: (key) => key.split("?")[0]!
  })
  chatMessagesInFlight = new Map<string, Promise<ServerChatMessage[]>>()
  private domainCacheConfig: TldwConfig | null = null
  private domainCacheAccountRevision = -1
  private domainCacheRevision = 0
  private openApiPathSet: Set<string> | null = null
  private openApiPathSetPromise: Promise<Set<string> | null> | null = null
  private resolvedPathCache = new Map<string, string>()

  constructor() {
    this.storage = createSafeStorage({
      area: "local",
      serde: safeStorageSerde
    })
    this.sessionStorage = createSafeStorage({
      area: "session",
      serde: safeStorageSerde
    })
  }

  private applyConfigState(): void {
    const config = this.config
    const quickstartWebUiServerUrl = getQuickstartWebUiServerUrl()
    const hostedMode = isHostedTldwDeployment()
    const nextBaseUrl = hostedMode
      ? String(config?.serverUrl || "").replace(/\/$/, "")
      : (isActiveCookieSessionConfig(config, quickstartWebUiServerUrl)
          ? quickstartWebUiServerUrl
          : config?.serverUrl || DEFAULT_SERVER_URL
        ).replace(/\/$/, "")
    if (this.baseUrl && this.baseUrl !== nextBaseUrl) {
      this.openApiPathSet = null
      this.openApiPathSetPromise = null
      this.resolvedPathCache.clear()
    }
    this.baseUrl = nextBaseUrl

    this.headers = { "Content-Type": "application/json" }
    const runtimeApiKey = !hostedMode
      ? getRuntimeSingleUserApiKeyOverride()
      : null
    const cookieSession = isActiveCookieSessionConfig(
      config,
      quickstartWebUiServerUrl
    )
    if (!cookieSession) {
      if (runtimeApiKey) {
        this.headers["X-API-KEY"] = runtimeApiKey
      } else if (
        !hostedMode &&
        config?.authMode === "single-user" &&
        config.apiKey
      ) {
        const key = String(config.apiKey).trim()
        if (key) this.headers["X-API-KEY"] = key
      } else if (
        !hostedMode &&
        config?.authMode === "multi-user" &&
        config.accessToken
      ) {
        this.headers["Authorization"] = `Bearer ${config.accessToken}`
      }
    }
    if (config?.orgId) {
      this.headers["X-TLDW-Org-Id"] = String(config.orgId)
    }
  }

  private publishConfigUpdated(previousConfig: TldwConfig | null, onlyIfAuthorityChanged = false, initialSetupToken?: object): void {
    const authorityChanged = !connectionAuthoritiesMatch(this.config, previousConfig)
    if (onlyIfAuthorityChanged && !authorityChanged) return
    if (authorityChanged) {
      if (typeof window !== "undefined") {
        try {
          window.sessionStorage.removeItem("tldw-h1-playground-reference")
        } catch {
          // Owner validation still fences old references when storage is blocked.
          console.warn("Could not retire the automatic Chat handoff after the account or server changed.")
        }
      }
      void clearFlashcardsGenerateHandoffs().catch(() => console.warn("Could not clear private Flashcards transfers after the account or server changed."))
    }
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged, ...(initialSetupToken ? { initialSetupToken } : {}) }
      }))
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

  private async syncConnectionServerUrl(serverUrl: unknown): Promise<void> {
    const normalized =
      typeof serverUrl === "string" ? serverUrl.trim().replace(/\/$/, "") : ""
    if (!normalized) {
      await this.storage.remove("tldwServerUrl").catch(() => undefined)
      if (typeof window === "undefined") return
      try {
        window.localStorage?.removeItem("tldw-api-host")
      } catch {
        // Best-effort compatibility mirror for the WebUI bootstrap.
      }
      return
    }

    await this.storage.set("tldwServerUrl", normalized).catch(() => undefined)

    if (typeof window === "undefined") return
    try {
      window.localStorage?.setItem("tldw-api-host", normalized)
    } catch {
      // Best-effort compatibility mirror for the WebUI bootstrap.
    }
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

  private invalidateDomainCaches(): void {
    this.domainCacheRevision += 1
    this.characterCache.clear()
    this.characterInFlight.clear()
    this.chatMessagesCache.clear()
    this.chatMessagesInFlight.clear()
  }

  async getDomainCacheRevision(): Promise<number> {
    const accountRevision = domainCacheAccountRevision
    const startedRevision = this.domainCacheRevision
    try {
      // getConfig alone may retain a previous server or API-key connection.
      await this.initialize()
      const initializedConfig = this.config
      const config = await this.ensureConfigForRequest(true, initializedConfig)
      if (accountRevision !== domainCacheAccountRevision ||
          !connectionAuthoritiesMatch(initializedConfig, this.config)) {
        throw createServicePromptScopeChangedError()
      }
      if (this.domainCacheAccountRevision !== accountRevision ||
          !connectionAuthoritiesMatch(config, this.domainCacheConfig)) {
        this.invalidateDomainCaches()
      }
      this.domainCacheConfig = { ...config }
      this.domainCacheAccountRevision = accountRevision
      return this.domainCacheRevision
    } catch (error) {
      if (this.domainCacheRevision === startedRevision) {
        this.invalidateDomainCaches()
        this.domainCacheConfig = null
      }
      throw error
    }
  }

  assertDomainCacheRevision(revision: number): void {
    // Keep the final check synchronous with cache reads and publication.
    if (revision !== this.domainCacheRevision ||
        this.domainCacheAccountRevision !== domainCacheAccountRevision) {
      throw createServicePromptScopeChangedError()
    }
  }

  getDomainCacheConfigSnapshot(revision: number): TldwConfig {
    this.assertDomainCacheRevision(revision)
    if (!this.domainCacheConfig) throw createServicePromptScopeChangedError()
    // Domain single-flight owns these reads; do not join a transport-only key.
    return { ...this.domainCacheConfig }
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

  async ensureConfigForRequest(
    requireAuth: boolean,
    resolvedConfig?: TldwConfig | null
  ): Promise<TldwConfig> {
    const cfg = (resolvedConfig === undefined ? await this.getConfig() : resolvedConfig) || null
    const hostedMode = isHostedTldwDeployment()
    const runtimeApiKey = !hostedMode
      ? getRuntimeSingleUserApiKeyOverride()
      : null
    const cookieSession = isActiveCookieSessionConfig(cfg)
    if ((!cfg || !cfg.serverUrl) && !hostedMode && !runtimeApiKey) {
      const location = getCurrentBrowserSurface() === "extension"
        ? "Settings → tldw server in the extension"
        : "Settings → tldw server"
      const msg = `tldw server is not configured. Open ${location} and set the server URL and API key.`
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
        authMode: cfg?.authMode || "multi-user",
        authSource: cfg?.authSource
      }
    }

    if (!requireAuth) {
      return cfg
    }

    if (cookieSession) {
      return cfg
    }

    if (runtimeApiKey) {
      return {
        ...(cfg || {}),
        serverUrl: String(
          cfg?.serverUrl || getQuickstartWebUiServerUrl() || DEFAULT_SERVER_URL
        ),
        authMode: "single-user",
        apiKey: runtimeApiKey
      }
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

  async request<T = any>(init: any, requireAuth = true): Promise<T> {
    await this.ensureConfigForRequest(requireAuth && !init?.noAuth)
    return await bgRequest<T>(init)
  }

  async requestWithCurrentConfig<T = any>(
    init: any | ((config: TldwConfig) => any),
    requireAuth = true
  ): Promise<T> {
    const usesConfigFactory = typeof init === "function"
    const staticInit = usesConfigFactory ? null : init
    const useDirectWebRequest = getCurrentBrowserSurface() === "webui-page"
    if (useDirectWebRequest) await this.initialize()
    const cfg = await this.ensureConfigForRequest(
      requireAuth && !staticInit?.noAuth
    )
    const resolvedInit = typeof init === "function" ? init(cfg) : init
    if (!useDirectWebRequest) {
      return await bgRequest<T>(
        usesConfigFactory
          ? { ...resolvedInit, configSnapshot: cfg }
          : resolvedInit
      )
    }

    const response = await tldwRequest(resolvedInit, {
      getConfig: async () => cfg
    })
    if (!response?.ok && !resolvedInit?.returnResponse) {
      const message =
        typeof response?.error === "string" && response.error.trim()
          ? response.error
          : `Request failed: ${response?.status ?? 0}`
      const error = new Error(message) as Error & {
        status?: number
        details?: unknown
      }
      error.status = response?.status
      error.details = response?.data
      throw error
    }
    return (resolvedInit?.returnResponse ? response : response.data) as T
  }

  async fetchWithAuth(
    path: PathOrUrl,
    init?: {
      requestScope?: ServicePromptRequestScope
      method?: string
      headers?: Record<string, string>
      body?: any
      timeoutMs?: number
      signal?: AbortSignal
      responseType?: "json" | "text" | "arrayBuffer"
      suppressBackendUnavailableEvent?: boolean
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
    const scopeFields = requestScopeFields(init?.requestScope)
    const response = await bgRequest<any, PathOrUrl>({
      ...scopeFields,
      path,
      method: (init?.method || "GET") as any,
      headers: scopeFields.headers ? { ...init?.headers, ...scopeFields.headers } : init?.headers,
      body: init?.body,
      timeoutMs: init?.timeoutMs,
      abortSignal: init?.signal,
      responseType: init?.responseType,
      suppressBackendUnavailableEvent: init?.suppressBackendUnavailableEvent,
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
    const throwIfAborted = () => {
      if (!init?.abortSignal?.aborted) return
      const error = new Error("Upload was cancelled")
      error.name = "AbortError"
      throw error
    }
    throwIfAborted()
    await this.ensureConfigForRequest(requireAuth)
    throwIfAborted()
    const result = await bgUpload<T>(init)
    throwIfAborted()
    return result
  }

  async *stream(init: any, requireAuth = true): AsyncGenerator<string> {
    await this.ensureConfigForRequest(requireAuth)
    for await (const line of bgStream(init)) {
      yield line as string
    }
  }

  async initialize(): Promise<void> {
    await recoverRuntimeAuthBootstrap()
    const initialization = ++this.configInitialization
    let storedManual = await this.storage.get<TldwConfig>("tldwConfig")
    if (!storedManual) {
      try {
        const legacySyncStore = createSafeStorage({
          area: "sync",
          serde: safeStorageSerde
        })
        const legacySyncConfig = await legacySyncStore.get<TldwConfig>("tldwConfig")
        if (legacySyncConfig) {
          storedManual = legacySyncConfig
          await this.storage.set("tldwConfig", legacySyncConfig)
          await legacySyncStore.remove("tldwConfig").catch(() => undefined)
        }
      } catch {
        // ignore migration failures
      }
    }
    const quickstartWebUiServerUrl = getQuickstartWebUiServerUrl()
    const envApiKey = quickstartWebUiServerUrl ? null : getBuildTimeApiKey()
    const storedCookieSession =
      quickstartWebUiServerUrl && !isCookieSessionConfigInvalidated()
        ? await this.storage
            .get<TldwConfig>(COOKIE_SESSION_CONFIG_KEY)
            .catch(() => null)
        : null
    const activeCookieSession = isActiveCookieSessionConfig(
      storedCookieSession,
      quickstartWebUiServerUrl
    )
      ? storedCookieSession
      : null
    if (storedManual) {
      const origin = normalizeServerOrigin(storedManual.serverUrl)
      const hasOwn = (key: keyof TldwConfig) =>
        Object.prototype.hasOwnProperty.call(storedManual, key)
      const hasNoCredentialMetadata =
        !hasOwn("credentialSource") &&
        !hasOwn("apiKeyPersistence") &&
        !hasOwn("apiKeyServerOrigin")
      const hasLegacyAuthSource =
        !hasOwn("authSource") || storedManual.authSource === "manual"
      const hasLegacyManualKey =
        !isHostedTldwDeployment() &&
        !quickstartWebUiServerUrl &&
        storedManual.authMode === "single-user" &&
        hasLegacyAuthSource &&
        typeof storedManual.apiKey === "string" &&
        Boolean(storedManual.apiKey.trim()) &&
        !isPlaceholderApiKey(storedManual.apiKey) &&
        hasNoCredentialMetadata &&
        Boolean(origin) &&
        !activeCookieSession &&
        !envApiKey &&
        !getRuntimeSingleUserApiKeyOverride()

      let persistedManual = storedManual
      if (hasLegacyManualKey && origin) {
        persistedManual = {
          ...storedManual,
          authSource: "manual",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: origin
        }
      } else if (
        storedManual.apiKey &&
        !isCompleteDeviceCredential(storedManual)
      ) {
        persistedManual = toPersistedTldwConfig(storedManual)
      }

      if (persistedManual !== storedManual) {
        await this.storage
          .set("tldwConfig", persistedManual)
          .catch(() => undefined)
      }
      storedManual = persistedManual
    }
    if (storedManual?.apiKeyPersistence === "session") {
      const manualApiKey = await resolveManualCredential(storedManual, {
        session: this.sessionStorage
      })
      if (!manualApiKey && hasManualSessionCredentialLock()) {
        await withManualSessionCredentialLock(async () => {
          const currentConfig = await this.storage
            .get<TldwConfig>("tldwConfig")
            .catch(() => null)
          const currentManualApiKey = currentConfig
            ? await resolveManualCredential(currentConfig, {
                session: this.sessionStorage
              })
            : null
          if (!currentManualApiKey) {
            await this.sessionStorage
              .remove(MANUAL_SESSION_KEY)
              .catch(() => undefined)
          }
        })
      }
    }
    const stored = await resolveEffectiveTldwConfig(
      {
        persistent: {
          get: async <T>(key: string) =>
            key === "tldwConfig"
              ? (storedManual as T | null | undefined)
              : await this.storage.get<T>(key),
          set: <T>(key: string, value: T) => this.storage.set(key, value),
          remove: (key: string) => this.storage.remove(key)
        },
        session: this.sessionStorage
      },
      activeCookieSession
        ? {
            cookieSession: activeCookieSession,
            expectedCookieOrigin: quickstartWebUiServerUrl
          }
        : undefined
    )
    let config: TldwConfig | null
    if (!stored) {
      // True first-run without quickstart auth material: leave config null so
      // callers can distinguish unconfigured from misconfigured/unreachable.
      config = null
    } else {
      const hydrated: TldwConfig = {
        ...stored,
        // Default authMode but do not silently inject a server URL if none
        // has been configured yet.
        authMode: stored.authMode || "single-user",
        serverUrl: activeCookieSession
          ? quickstartWebUiServerUrl || ""
          : stored.serverUrl || ""
      }
      if (activeCookieSession) {
        delete hydrated.apiKey
      } else if (!hydrated.apiKey && envApiKey) {
        hydrated.apiKey = envApiKey
      }
      config = hydrated
    }
    // An older storage read must not overwrite a newer owner's config or token.
    if (initialization < this.completedConfigInitialization) {
      if (!connectionAuthoritiesMatch(config, this.config)) {
        throw createServicePromptScopeChangedError()
      }
      return
    }
    this.completedConfigInitialization = initialization
    this.config = config
    this.applyConfigState()
  }

  private async activateManualConfig(config: TldwConfig, previousConfig: TldwConfig | null, initialSetup?: InitialChatSetupAttempt): Promise<void> {
    this.config = config
    this.applyConfigState()
    this.publishConfigUpdated(previousConfig, true, initialSetup?.token)
    initialSetup?.afterBoundary(config)
    await this.syncConnectionServerUrl(config.serverUrl)
    this.publishConfigUpdated(config, false, initialSetup?.token)
  }

  private async writeSessionOrMemoryUnlocked(
    input: { serverUrl: string; apiKey: string },
    serverOrigin: string,
    assertCurrent?: () => void
  ): Promise<{ persistence: "session" | "memory"; config: TldwConfig }> {
    const persisted: TldwConfig = {
      authMode: "single-user",
      authSource: "manual",
      serverUrl: input.serverUrl,
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: serverOrigin
    }
    const record: ManualSessionCredential = {
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: serverOrigin,
      apiKey: input.apiKey
    }
    const hydrated = { ...persisted, apiKey: input.apiKey }

    try {
      assertCurrent?.()
      await this.storage.set("tldwConfig", persisted)
      assertCurrent?.()
      await this.sessionStorage.set(MANUAL_SESSION_KEY, record)
      assertCurrent?.()
      return { persistence: "session", config: hydrated }
    } catch (error) {
      if (isRequestConfigScopeChangedError(error)) throw error
      await clearManualCredentials(this.storage, this.sessionStorage).catch(
        async () => {
          await this.sessionStorage
            .remove(MANUAL_SESSION_KEY)
            .catch(() => undefined)
        }
      )
      return { persistence: "memory", config: hydrated }
    }
  }

  private async clearManualSingleUserCredentialsUnlocked(): Promise<TldwConfig | null> {
    await clearManualCredentials(this.storage, this.sessionStorage)
    return (
      (await this.storage.get<TldwConfig>("tldwConfig").catch(() => null)) ||
      null
    )
  }

  async saveManualSingleUserCredential(input: {
    serverUrl: string
    apiKey: string
    persistence: ApiKeyPersistence
    assertCurrent?: (current: TldwConfig | null) => void
    initialSetup?: InitialChatSetupAttempt
  }): Promise<"device" | "session" | "memory"> {
    const serverUrl = String(input.serverUrl || "").trim()
    const apiKey = String(input.apiKey || "").trim()
    const serverOrigin = normalizeServerOrigin(serverUrl)
    if (!serverOrigin) throw new Error("Invalid server URL")
    if (!apiKey) throw new Error("API key is required")

    cancelRuntimeAuthBootstrapRecovery()
    const initialSetup = input.initialSetup
    if (initialSetup) {
      initialSetup.assertCurrent()
      if (!hasManualSessionCredentialLock() || !(await this.hasNoWebUiConnectionConfiguration())) throw createServicePromptScopeChangedError()
      initialSetup.assertCurrent()
    }
    const previousConfig = await this.getConfig()
    input.assertCurrent?.(previousConfig)
    let qualifiedInitialSetup = initialSetup
    const saved = await withManualSessionCredentialLock(async () => {
      if (initialSetup) {
        initialSetup.assertCurrent()
        if (!(await this.hasNoWebUiConnectionConfiguration())) throw createServicePromptScopeChangedError()
        initialSetup.assertCurrent()
      }
      if (input.assertCurrent) {
        // Initialization can acquire this same credential lock for session cleanup.
        const current = await resolveEffectiveTldwConfig({ persistent: this.storage, session: this.sessionStorage },
          this.config?.authSource === "cookie-session"
            ? { cookieSession: this.config, expectedCookieOrigin: getQuickstartWebUiServerUrl() }
            : undefined)
        input.assertCurrent(current)
      }
      // Strict first configuration has no previous credentials to clear.
      if (!initialSetup) await this.clearManualSingleUserCredentialsUnlocked()
      let saved: { persistence: ApiKeyPersistence | "memory"; config: TldwConfig }
      if (input.persistence !== "device") {
        saved = await this.writeSessionOrMemoryUnlocked(
          { serverUrl, apiKey }, serverOrigin, initialSetup?.assertCurrent
        )
      } else {
        const deviceConfig: TldwConfig = {
          authMode: "single-user",
          authSource: "manual",
          serverUrl,
          apiKey,
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: serverOrigin
        }
        try {
          initialSetup?.assertCurrent()
          await this.storage.set("tldwConfig", deviceConfig)
          initialSetup?.assertCurrent()
          await this.sessionStorage
            .remove(MANUAL_SESSION_KEY)
            .catch(() => undefined)
          initialSetup?.assertCurrent()
          saved = { persistence: "device", config: deviceConfig }
        } catch (error) {
          if (isRequestConfigScopeChangedError(error)) throw error
          qualifiedInitialSetup?.cancel()
          qualifiedInitialSetup = undefined
          saved = await this.writeSessionOrMemoryUnlocked({ serverUrl, apiKey }, serverOrigin)
        }
      }
      if (qualifiedInitialSetup) {
        if (saved.persistence === "memory") {
          qualifiedInitialSetup.cancel()
          qualifiedInitialSetup = undefined
        } else {
          qualifiedInitialSetup.assertCurrent()
          if (!(await this.verifyInitialWebUiConfiguration(saved.config))) throw createServicePromptScopeChangedError()
          qualifiedInitialSetup.assertCurrent()
          await this.activateManualConfig(saved.config, previousConfig, qualifiedInitialSetup)
        }
      }
      return saved
    })
    if (!qualifiedInitialSetup) await this.activateManualConfig(saved.config, previousConfig)
    return saved.persistence
  }

  async hydrateManualSingleUserCredential(): Promise<TldwConfig | null> {
    await this.initialize()
    return this.config?.credentialSource === "manual" ? this.config : null
  }

  async clearManualSingleUserCredentials(): Promise<void> {
    cancelRuntimeAuthBootstrapRecovery()
    const stored = await withManualSessionCredentialLock(async () =>
      await this.clearManualSingleUserCredentialsUnlocked()
    )
    if (this.config?.authSource !== "cookie-session") {
      this.config = stored
      this.applyConfigState()
    }
  }

  async clearCookieSingleUserSession(): Promise<void> {
    cancelRuntimeAuthBootstrapRecovery()
    invalidateCookieSessionConfig()
    const previousConfig = this.config
    let failure: unknown
    try {
      await this.storage.remove(COOKIE_SESSION_CONFIG_KEY)
    } catch (error) {
      failure = error
    }
    this.config = null
    this.applyConfigState()
    this.publishConfigUpdated(previousConfig, true)
    try {
      await this.initialize()
    } catch (error) {
      failure ??= error
    }
    this.publishConfigUpdated(null)
    if (failure) throw failure
  }

  /** Strict admission for an explicit initial WebUI Save; never initializes or migrates. */
  async hasNoWebUiConnectionConfiguration(): Promise<boolean> {
    return this.verifyInitialWebUiConfiguration()
  }

  private async verifyInitialWebUiConfiguration(expected?: TldwConfig): Promise<boolean> {
    try {
      const persisted = expected ? toPersistedTldwConfig(expected) : undefined
      const session = expected?.apiKeyPersistence === "session" ? {
        credentialSource: "manual", apiKeyPersistence: "session", apiKeyServerOrigin: expected.apiKeyServerOrigin, apiKey: expected.apiKey
      } : undefined
      const sync = createSafeStorage({ area: "sync", serde: safeStorageSerde })
      const rawAbsent = () => {
        const env = (import.meta as unknown as { env?: Record<string, unknown> }).env || {}
        const processEnv: Record<string, string | undefined> = typeof process === "undefined" ? {} : process.env
        if (getCurrentBrowserSurface() !== "webui-page" ||
          (this.config !== null && (!expected || !connectionAuthoritiesMatch(this.config, expected))) ||
          isHostedTldwDeployment() || getRuntimeSingleUserApiKeyOverride() ||
          getQuickstartWebUiServerUrl() ||
          [env.VITE_TLDW_API_KEY, env.VITE_TLDW_DEFAULT_API_KEY,
            processEnv.VITE_TLDW_API_KEY, processEnv.VITE_TLDW_DEFAULT_API_KEY,
            processEnv.NEXT_PUBLIC_X_API_KEY, processEnv.NEXT_PUBLIC_API_BEARER]
            .some(value => value !== undefined && value !== "") ||
          (window as Window & { __tldwRuntimeApiKey?: unknown }).__tldwRuntimeApiKey != null) return false
        // The WebUI shim may fall back to memory; that cannot prove disk absence.
        if ((this.storage as Storage & { hasPersistentBackend?: boolean }).hasPersistentBackend === false ||
          (sync as Storage & { hasPersistentBackend?: boolean }).hasPersistentBackend === false) return false
        return window.localStorage.getItem("tldwConfig") === (persisted ? JSON.stringify(persisted) : null) &&
          [COOKIE_SESSION_CONFIG_KEY, REFRESH_ROTATION_KEY,
          "plasmo-sync:tldwConfig", "apiKey", "accessToken", "access_token",
          "tldwRuntimeSessionSingleUserApiKey", "tldwRuntimeAuthMetadata"]
          .every(key => window.localStorage.getItem(key) === null) &&
          window.sessionStorage.getItem(MANUAL_SESSION_KEY) === (session ? JSON.stringify(session) : null)
      }
      if (!rawAbsent()) return false
      const values = await Promise.all([
        this.storage.get("tldwConfig"), sync.get("tldwConfig"),
        this.storage.get(COOKIE_SESSION_CONFIG_KEY), this.storage.get(REFRESH_ROTATION_KEY),
        this.sessionStorage.get(MANUAL_SESSION_KEY)
      ])
      return JSON.stringify(values[0]) === JSON.stringify(persisted) &&
        values.slice(1, 4).every(value => value === undefined) &&
        JSON.stringify(values[4]) === JSON.stringify(session) && rawAbsent()
    } catch {
      // Failure or malformed authority retires only the initial-input exception.
      return false
    }
  }

  async getConfig(): Promise<TldwConfig | null> {
    if (
      isCookieSessionConfigInvalidated() &&
      this.config?.authSource === "cookie-session"
    ) {
      this.config = null
    }
    const cachedConfig = this.config
    const hasNewerAccessToken = Boolean(
      cachedConfig?.authMode === "multi-user" &&
      cachedConfig.accessToken &&
      await hasNewerCurrentAccessToken(
        this.storage,
        cachedConfig,
        cachedConfig.accessToken
      ).catch(() => false)
    )
    const sessionInvalidated = Boolean(cachedConfig &&
      await hasInvalidatedRefreshSession(this.storage, cachedConfig).catch(() => false))
    if (this.config === null || hasNewerAccessToken || sessionInvalidated) {
      await this.initialize().catch(() => null)
    }
    return this.config
  }

  async invalidateRefreshSession(checked: TldwConfig): Promise<boolean> {
    const invalidated = await invalidateRefreshSessionIfCurrent(this.storage, checked)
    await this.getConfig()
    return invalidated
  }

  async commitTokenRefresh(
    checked: TldwConfig,
    expectedRefreshToken: string,
    tokens: Readonly<{ accessToken: string; refreshToken: string }>
  ): Promise<boolean> {
    const stored = await storeRefreshRotationIfCurrent(
      this.storage,
      checked,
      expectedRefreshToken,
      tokens
    )
    this.config = null
    await this.initialize().catch(() => null)
    const current = this.config
    return Boolean(
      stored &&
      current &&
      current.authMode === "multi-user" &&
      servicePromptTargetsMatch(current, checked) &&
      String(current.accessToken || "").trim() === tokens.accessToken &&
      String(current.refreshToken || "").trim() === tokens.refreshToken
    )
  }

  async updateConfig(
    config: Partial<TldwConfig>,
    assertCurrent?: (current: Partial<TldwConfig>) => void,
    assertCommitted?: (current: Partial<TldwConfig>) => void,
    initialSetupToken?: object
  ): Promise<void> {
    cancelRuntimeAuthBootstrapRecovery()
    await this.initialize()
    let currentConfig = (await this.getConfig()) || ({} as TldwConfig)
    assertCurrent?.(currentConfig)
    const previousConfig = currentConfig
    const targetAuthMode = config.authMode || currentConfig.authMode
    const submittedApiKey = Object.prototype.hasOwnProperty.call(config, "apiKey")
      ? String(config.apiKey || "").trim()
      : null
    if (
      submittedApiKey &&
      targetAuthMode === "single-user" &&
      (config.serverUrl || currentConfig.serverUrl)
    ) {
      await this.saveManualSingleUserCredential({
        serverUrl: config.serverUrl || currentConfig.serverUrl,
        apiKey: submittedApiKey,
        persistence: config.apiKeyPersistence || "device",
        ...(assertCurrent ? { assertCurrent: (current: TldwConfig | null) => assertCurrent(current || {}) } : {})
      })
      assertCommitted?.((await this.getConfig()) || {})
      return
    }

    const authModeChanged = Boolean(
      config.authMode && config.authMode !== currentConfig.authMode
    )
    const originChanged = Object.prototype.hasOwnProperty.call(
      config,
      "serverUrl"
    )
      ? normalizeServerOrigin(config.serverUrl || "") !==
        normalizeServerOrigin(currentConfig.serverUrl || "")
      : false
    if (authModeChanged || originChanged || submittedApiKey === "") {
      await this.clearManualSingleUserCredentials()
      currentConfig =
        (await this.storage.get<TldwConfig>("tldwConfig").catch(() => null)) ||
        ({} as TldwConfig)
    }

    // The auth attempt may have expired while configuration reads were pending.
    assertCurrent?.(currentConfig)
    const newConfig = { ...currentConfig, ...config } as TldwConfig
    const persisted = toPersistedTldwConfig(newConfig)
    if (assertCurrent) {
      // Observe the write as well as its verification: either await can cross
      // an account boundary, including a switch away and back to this account.
      let invalidated = false
      let revision = 0
      const unwatch = watchChatAccountChanges((changed, current) => {
        if (current) {
          revision += 1
          if (!connectionAuthoritiesMatch(current, persisted)) invalidated = true
        } else if (changed) invalidated = true
      })
      let committed: TldwConfig | null | undefined
      try {
        await this.storage.set("tldwConfig", persisted)
        let readRevision: number
        do {
          readRevision = revision
          committed = await this.storage.get<TldwConfig>("tldwConfig")
          if (invalidated) throw createServicePromptScopeChangedError()
          if (!committed || !connectionAuthoritiesMatch(committed, persisted)) {
            this.config = committed || null
            this.applyConfigState()
            throw createServicePromptScopeChangedError()
          }
          // A delayed own notification may be older or newer than this read.
          // Read again instead of treating the notification as authoritative.
        } while (readRevision !== revision)
        const hydrated = committed.apiKeyPersistence === "session" && committed.credentialSource === "manual"
          ? { ...committed, apiKey: await resolveManualCredential(committed, { session: this.sessionStorage }) || undefined }
          : committed
        if (invalidated || !connectionAuthoritiesMatch(hydrated, newConfig)) throw createServicePromptScopeChangedError()
        try {
          assertCommitted?.(hydrated)
        } catch (error) {
          this.config = null
          throw error
        }
        this.config = hydrated
        this.applyConfigState()
      } finally {
        unwatch()
      }
      this.publishConfigUpdated(previousConfig, true, initialSetupToken)
      this.publishConfigUpdated(this.config, false, initialSetupToken)
      return
    }
    await this.storage.set("tldwConfig", persisted)
    this.config = newConfig
    this.applyConfigState()
    this.publishConfigUpdated(previousConfig, true)
    if (Object.prototype.hasOwnProperty.call(config, "serverUrl")) {
      await this.syncConnectionServerUrl(config.serverUrl)
    }
    await this.initialize().catch(() => null)
    this.publishConfigUpdated(newConfig)
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

  async getDefaultCharacterPreference(options?: ScopedRequestOptions): Promise<string | null> {
    const profile = options?.requestScope ? await bgRequest<{ preferences?: Record<string, unknown> }>({
      path: "/api/v1/users/me/profile?sections=preferences",
      method: "GET",
      ...requestScopeFields(options.requestScope),
      abortSignal: options.signal
    }) : await this.getCurrentUserProfile({
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
    characterId: string | null,
    options?: ScopedRequestOptions
  ): Promise<UserProfileUpdateResponse> {
    const normalizedCharacterId = normalizeDefaultCharacterPreferenceId(characterId)
    if (options?.requestScope) {
      return bgRequest<UserProfileUpdateResponse>({
        path: "/api/v1/users/me/profile", method: "PATCH",
        ...requestScopeFields(options.requestScope),
        abortSignal: options.signal,
        body: { updates: [{ key: DEFAULT_CHARACTER_PROFILE_PREFERENCE_KEY, value: normalizedCharacterId }] }
      })
    }
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

  async getOpenAPISpec(
    options?: ScopedRequestOptions & { configSnapshot?: TldwConfig | null }
  ): Promise<any | null> {
    try {
      if (options?.signal?.aborted) return null
      if (options?.requestScope || options?.configSnapshot !== undefined) {
        if (!options.requestScope && !options.configSnapshot?.serverUrl)
          return null
        return await bgRequest<unknown>({
          path: "/openapi.json",
          method: "GET",
          ...(options.requestScope
            ? requestScopeFields(options.requestScope)
            : { configSnapshot: options.configSnapshot }),
          abortSignal: options.signal
        })
      }
      return await this.requestWithCurrentConfig<any>(
        (config) => ({
          path: `${config.serverUrl.replace(/\/$/, "")}/openapi.json`,
          method: "GET",
          abortSignal: options?.signal
        }),
        false
      )
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



  /**
   * Check which LLM providers are configured on the server.
   * Returns `{ providers: [...], any_configured: boolean }`.
   * Served from a short module-level TTL cache with in-flight dedup, since
   * a dozen-plus components request it independently on mount.
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
    return await getProvidersStatusCached()
  }








  async getSandboxRuntimeDiagnostics(): Promise<SandboxAdminRuntimeDiagnosticsResponse> {
    return await bgRequest<SandboxAdminRuntimeDiagnosticsResponse>({
      path: "/api/v1/sandbox/admin/runtime-diagnostics",
      method: "GET"
    })
  }

  async getSandboxWorkspaceDiagnostics(
    workspaceId: string,
    options?: { sourceLabel?: "research_workspace"; limit?: number }
  ): Promise<SandboxWorkspaceDiagnosticsResponse> {
    const query = this.buildQuery({
      source_label: options?.sourceLabel || "research_workspace",
      limit: options?.limit
    })
    const encodedWorkspaceId = encodeURIComponent(workspaceId)
    return await bgRequest<SandboxWorkspaceDiagnosticsResponse>({
      path: `/api/v1/sandbox/workspaces/${encodedWorkspaceId}/diagnostics${query}`,
      method: "GET"
    })
  }

















  async getLlamacppSnapshotSlots(
    profileId: string,
    signal?: AbortSignal, options?: LlamacppRequestOptions
  ): Promise<LlamacppSnapshotSlotsResponse> {
    return await bgRequest<LlamacppSnapshotSlotsResponse>({
      ...options,
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}/slots`,
      method: "GET",
      abortSignal: signal && options?.abortSignal ? AbortSignal.any([signal, options.abortSignal]) : signal ?? options?.abortSignal
    })
  }

  async listLlamacppSnapshots(
    profileId: string,
    offset = 0,
    signal?: AbortSignal, options?: LlamacppRequestOptions
  ): Promise<LlamacppSnapshotCatalogResponse> {
    return await bgRequest<LlamacppSnapshotCatalogResponse>({
      ...options,
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}/snapshots?offset=${offset}&limit=50`,
      method: "GET",
      abortSignal: signal && options?.abortSignal ? AbortSignal.any([signal, options.abortSignal]) : signal ?? options?.abortSignal
    })
  }

  async saveLlamacppSnapshot(
    profileId: string,
    payload: LlamacppSnapshotRequest, options?: LlamacppRequestOptions
  ): Promise<LlamacppSnapshotOperationResponse> {
    return await bgRequest<LlamacppSnapshotOperationResponse>({
      ...options,
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}/snapshots`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async restoreLlamacppSnapshot(
    profileId: string,
    snapshotId: string,
    payload: LlamacppSnapshotRequest & { replace_confirmed: true }, options?: LlamacppRequestOptions
  ): Promise<LlamacppSnapshotOperationResponse> {
    return await bgRequest<LlamacppSnapshotOperationResponse>({
      ...options,
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}/snapshots/${encodeURIComponent(snapshotId)}/restore`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async deleteLlamacppSnapshot(
    profileId: string,
    snapshotId: string, options?: LlamacppRequestOptions
  ): Promise<{ deleted: boolean }> {
    return await bgRequest<{ deleted: boolean }>({
      ...options,
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}/snapshots/${encodeURIComponent(snapshotId)}`,
      method: "DELETE"
    })
  }

  async getLlamacppSnapshotOperation(
    profileId: string,
    operationId: string,
    signal?: AbortSignal, options?: LlamacppRequestOptions
  ): Promise<LlamacppSnapshotOperationResponse> {
    return await bgRequest<LlamacppSnapshotOperationResponse>({
      ...options,
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}/snapshot-operations/${encodeURIComponent(operationId)}`,
      method: "GET",
      abortSignal: signal && options?.abortSignal ? AbortSignal.any([signal, options.abortSignal]) : signal ?? options?.abortSignal
    })
  }

















  async createChatCompletion(
    request: ChatCompletionRequest,
    options?: ChatCompletionRequestOptions
  ): Promise<Response> {
    // Non-stream request via background
    captureChatRequestDebugSnapshot({
      endpoint: "/api/v1/chat/completions",
      method: "POST",
      mode: "non-stream",
      body: request,
      metadata: options?.debugMetadata
    })
    const scopeFields = requestScopeFields(options?.requestScope)
    // Feeds the model chip's readiness (CC-01).
    const res = await trackChatModelRequestOutcome(chatModelOutcomeTarget(request), () => bgRequest<Response>({
      path: '/api/v1/chat/completions',
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...scopeFields.headers },
      body: request,
      timeoutMs: options?.timeoutMs,
      abortSignal: options?.signal,
      ...(scopeFields.servicePromptConfig
        ? { servicePromptConfig: scopeFields.servicePromptConfig }
        : {})
    }))
    // bgRequest throws on any non-2xx response, so a resolved value here is
    // always a successful completion. Return the parsed body unmodified — the
    // streaming path does not scrub content, and scrubbing successful replies
    // that merely mention "error"/"exception" or include a file path was
    // silently corrupting legitimate assistant content. Wrap as Response-like
    // to keep existing consumers happy.
    const data = res as any
    return createJsonResponseLike(data, { status: 200 })
  }

  async *streamChatCompletion(request: ChatCompletionRequest, options?: ChatCompletionStreamOptions): AsyncGenerator<any, void, unknown> {
    request = { ...request, stream: true }
    captureChatRequestDebugSnapshot({
      endpoint: "/api/v1/chat/completions",
      method: "POST",
      mode: "stream",
      body: request,
      metadata: options?.debugMetadata
    })
    const scopeFields = requestScopeFields(options?.requestScope)
    for await (const line of bgStream({
      path: '/api/v1/chat/completions',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...scopeFields.headers
      },
      body: request,
      abortSignal: options?.signal,
      onDispatch: options?.onDispatch,
      streamIdleTimeoutMs: options?.streamIdleTimeoutMs,
      ...(scopeFields.servicePromptConfig
        ? { servicePromptConfig: scopeFields.servicePromptConfig }
        : {})
    })) {
      try {
        const parsed = JSON.parse(line)
        yield parsed
      } catch (e) {
        // Ignore non-JSON lines
      }
    }
  }

  // RAG Methods
  async getResearchWorkspaceCapabilities(): Promise<unknown> {
    return await this.request<unknown>({
      path: '/api/v1/research-workspace/capabilities',
      method: 'GET'
    })
  }



  async *ragSearchStream(
    query: string,
    options?: any
  ): AsyncGenerator<any, void, unknown> {
    yield* chatRagMethods.ragSearchStream.call(this as any, query, options)
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
      offset?: number
    },
    options?: {
      timeoutMs?: number
      signal?: AbortSignal
      requestScope?: ServicePromptRequestScope
    }
  ): Promise<any> {
    const query = this.buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/media/ingest/jobs${query}`,
      method: "GET",
      ...requestScopeFields(options?.requestScope),
      abortSignal: options?.signal,
      timeoutMs: options?.timeoutMs
    })
  }



  async listMedia(
    params?: {
      page?: number
      results_per_page?: number
      include_keywords?: boolean
    },
    options?: { signal?: AbortSignal; requestScope?: ServicePromptRequestScope }
  ): Promise<any> {
    const query = this.buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      ...requestScopeFields(options?.requestScope),
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
    options?: { signal?: AbortSignal; requestScope?: ServicePromptRequestScope }
  ): Promise<any> {
    const query = this.buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/media/search${query}`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      ...requestScopeFields(options?.requestScope),
      abortSignal: options?.signal
    })
  }

  async updateMediaKeywords(
    mediaId: string | number,
    payload: { keywords: string[]; mode?: "add" | "remove" | "set" },
    options?: { suppressBackendUnavailableEvent?: boolean; signal?: AbortSignal; requestScope?: ServicePromptRequestScope }
  ): Promise<{ media_id: number; keywords: string[] }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{ media_id: number; keywords: string[] }>({
      path: `/api/v1/media/${id}/keywords`,
      method: "PATCH",
      body: payload,
      ...requestScopeFields(options?.requestScope),
      headers: { ...requestScopeFields(options?.requestScope).headers, "Content-Type": "application/json" },
      abortSignal: options?.signal,
      suppressBackendUnavailableEvent: options?.suppressBackendUnavailableEvent
    })
  }





  async reprocessMedia(
    mediaId: string | number,
    options?: Record<string, unknown>,
    requestOptions?: { signal?: AbortSignal; requestScope?: ServicePromptRequestScope }
  ): Promise<any> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<any>({
      path: `/api/v1/media/${id}/reprocess`,
      method: "POST",
      body: options || {},
      ...requestScopeFields(requestOptions?.requestScope),
      headers: { ...requestScopeFields(requestOptions?.requestScope).headers, "Content-Type": "application/json" },
      abortSignal: requestOptions?.signal
    })
  }


  async getMediaDetails(
    mediaId: string | number,
    options?: {
      include_content?: boolean
      include_versions?: boolean
      include_version_content?: boolean
      requestScope?: ServicePromptRequestScope
      signal?: AbortSignal
      suppressBackendUnavailableEvent?: boolean
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
      ...requestScopeFields(options?.requestScope),
      abortSignal: options?.signal,
      suppressBackendUnavailableEvent: options?.suppressBackendUnavailableEvent
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


  async listNoteFolders(): Promise<{
    items: Array<{
      id: number
      name: string
      path: string
      parent_id: number | null
    }>
    folders?: Array<{
      id: number
      name: string
      path: string
      parent_id: number | null
    }>
    count: number
  }> {
    return await bgRequest({
      path: "/api/v1/notes/folders/",
      method: "GET"
    })
  }

  async createNoteFolder(path: string): Promise<{
    id: number
    name: string
    path: string
    parent_id: number | null
  }> {
    return await bgRequest({
      path: "/api/v1/notes/folders/",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { path }
    })
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

  async searchNotes(
    query: string,
    params?: { limit?: number; offset?: number },
    options?: ScopedRequestOptions,
  ): Promise<any> {
    return collectionsMethods.searchNotes.call(this, query, params, options)
  }









   async searchCharacters(query: string, params?: Record<string, any>, options?: ScopedRequestOptions): Promise<any[]> {
    const scopeFields = requestScopeFields(options?.requestScope)
    const qp = this.buildQuery({ query, ...(params || {}) })
    const base = await this.resolveApiPath("characters.search", [
      "/api/v1/characters/search",
      "/api/v1/characters/search/"
    ])
    return await bgRequest<any[]>({
      ...scopeFields,
      ...(options?.signal ? { abortSignal: options.signal } : {}),
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

  async getCharacter(id: string | number, options?: ScopedRequestOptions & { forceRefresh?: boolean }): Promise<any> {
    const cid = String(id)
    if (options?.requestScope) {
      const template = await this.resolveApiPath("characters.get", ["/api/v1/characters/{id}", "/api/v1/characters/{id}/"])
      return bgRequest({
        path: this.fillPathParams(template, cid), method: "GET",
        ...requestScopeFields(options.requestScope), abortSignal: options.signal
      })
    }
    const cacheRevision = await this.getDomainCacheRevision()
    this.assertDomainCacheRevision(cacheRevision)
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
        this.assertDomainCacheRevision(cacheRevision)
        const value = await bgRequest<any>({
          path,
          method: 'GET',
          configSnapshot: this.getDomainCacheConfigSnapshot(cacheRevision)
        })
        await this.getDomainCacheRevision()
        this.assertDomainCacheRevision(cacheRevision)
        this.characterCache.set(cid, {
          value,
          expiresAt: Date.now() + CHARACTER_CACHE_TTL_MS
        })
        return value
      } finally {
        if (this.characterInFlight.get(cid) === request) {
          this.characterInFlight.delete(cid)
        }
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



  async listChats(
    params?: Record<string, any>,
    options?: { signal?: AbortSignal; scope?: ChatScope }
  ): Promise<ServerChatSummary[]> {
    const query = this.buildQuery({ ...params, ...toChatScopeParams(options?.scope) })
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

    // Runtime implementation is the chat-rag mixin (ownership-guarded).
    return list.map((c) => (this as unknown as TldwApiClientCore).normalizeChatSummary(c))
  }

  async listChatsWithMeta(
    params?: Record<string, any>,
    options?: { signal?: AbortSignal; scope?: ChatScope; requestScope?: ServicePromptRequestScope }
  ): Promise<{ chats: ServerChatSummary[]; total: number }> {
    return chatRagMethods.listChatsWithMeta.call(this as unknown as TldwApiClientCore, params, options)
  }

  async createChat(
    payload: Record<string, any>,
    options?: {
      scope?: ChatScope
      signal?: AbortSignal
      requestScope?: ServicePromptRequestScope
      seedFirstMessage?: boolean
      greetingStrategy?: "default" | "alternate_random" | "alternate_index"
      alternateIndex?: number
    }
  ): Promise<ServerChatSummary> {
    return chatRagMethods.createChat.call(this as unknown as TldwApiClientCore, payload, options)
  }

  async appendRoleplayResumeInput(
    chat_id: string | number,
    payload: RoleplayResumeUserAppendRequest,
    options: ScopedRequestOptions & {onDispatch?: () => void}
  ): Promise<RoleplayResumeUserAppendReceipt> {
    return chatRagMethods.appendRoleplayResumeInput.call(
      this as unknown as TldwApiClientCore,
      chat_id,
      payload,
      options
    )
  }

  async completeCharacterChatTurn(
    chat_id: string | number,
    payload: Record<string, unknown>,
    options?: ScopedRequestOptions & {onDispatch?: () => void}
  ): Promise<Record<string, unknown>> {
    return chatRagMethods.completeCharacterChatTurn.call(
      this as unknown as TldwApiClientCore,
      chat_id,
      payload,
      options
    )
  }

  async getChat(
    chat_id: string | number,
    options?: { scope?: ChatScope; requiredMetadata?: readonly string[]; requireHistoryVersion?: boolean; requireRoleplayResume?: boolean } & ScopedRequestOptions
  ): Promise<ServerChatSummary> {
    return chatRagMethods.getChat.call(this as unknown as TldwApiClientCore, chat_id, options)
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

  async getChatSettings(
    chat_id: string | number,
    options?: { scope?: ChatScope; signal?: AbortSignal; requestScope?: ServicePromptRequestScope }
  ): Promise<ChatSettingsResponse> {
    const scopeFields = requestScopeFields(options?.requestScope)
    const cid = String(chat_id)
    const query = this.buildQuery(toChatScopeParams(options?.scope))
    return await bgRequest<ChatSettingsResponse>({
      path: appendPathQuery(`/api/v1/chats/${cid}/settings`, query),
      method: "GET",
      headers: scopeFields.headers,
      abortSignal: options?.signal,
      ...(scopeFields.servicePromptConfig ? {servicePromptConfig: scopeFields.servicePromptConfig} : {}),
      expectedStatuses: [404]
    })
  }

  async updateChatSettings(
    chat_id: string | number,
    settings: Record<string, unknown>,
    options?: { scope?: ChatScope; signal?: AbortSignal; requestScope?: ServicePromptRequestScope }
  ): Promise<ChatSettingsResponse> {
    const scopeFields = requestScopeFields(options?.requestScope)
    const cid = String(chat_id)
    const query = this.buildQuery(toChatScopeParams(options?.scope))
    return await bgRequest<ChatSettingsResponse>({
      path: appendPathQuery(`/api/v1/chats/${cid}/settings`, query),
      method: "PUT",
      headers: { "Content-Type": "application/json", ...scopeFields.headers },
      abortSignal: options?.signal,
      ...(scopeFields.servicePromptConfig ? {servicePromptConfig: scopeFields.servicePromptConfig} : {}),
      body: { settings }
    })
  }

  async getChatLorebookDiagnostics(
    chat_id: string | number,
    params?: Record<string, any>,
    options?: { scope?: ChatScope }
  ): Promise<LorebookDiagnosticExportResponse> {
    const cid = String(chat_id)
    const query = this.buildQuery({ ...params, ...toChatScopeParams(options?.scope) })
    return await bgRequest<LorebookDiagnosticExportResponse>({
      path: `/api/v1/chats/${cid}/diagnostics/lorebook${query}`,
      method: "GET"
    })
  }

  async updateChat(
    chat_id: string | number,
    payload: Record<string, any>,
    options?: { expectedVersion?: number; scope?: ChatScope } & ScopedRequestOptions
  ): Promise<ServerChatSummary> {
    const scopeFields = requestScopeFields(options?.requestScope)
    const cid = String(chat_id)
    let expectedVersion = options?.expectedVersion
    if (expectedVersion == null) {
      try {
        const current = await this.getChat(cid, options)
        if (typeof current?.version === "number") {
          expectedVersion = current.version
        }
      } catch {
        // ignore and fall back to unversioned update
      }
    }
    const qp = this.buildQuery({
      ...toChatScopeParams(options?.scope),
      ...(typeof expectedVersion === "number" ? {expected_version:expectedVersion} : {})
    })
    const res = await bgRequest<any>({
      ...scopeFields,
      ...(options?.signal ? { abortSignal: options.signal } : {}),
      path: `/api/v1/chats/${cid}${qp}`,
      method: "PUT",
      headers: { ...scopeFields.headers, "Content-Type": "application/json" },
      body: payload
    })
    // Runtime implementation is the chat-rag mixin (ownership-guarded).
    return (this as unknown as TldwApiClientCore).normalizeChatSummary(res)
  }

  async deleteChat(
    chat_id: string | number,
    options?: {
      signal?: AbortSignal
      requestScope?: ServicePromptRequestScope
      expectedVersion?: number
      hardDelete?: boolean
      scope?: ChatScope
    }
  ): Promise<void> {
    const scopeFields = requestScopeFields(options?.requestScope)
    const cid = String(chat_id)
    const query = this.buildQuery({
      ...toChatScopeParams(options?.scope),
      ...(typeof options?.expectedVersion === "number"
        ? { expected_version: options.expectedVersion }
        : {}),
      ...(options?.hardDelete ? { hard_delete: true } : {})
    })
    await bgRequest<void>({
      ...scopeFields,
      ...(options?.signal ? { abortSignal: options.signal } : {}),
      path: `/api/v1/chats/${cid}${query}`,
      method: "DELETE"
    })
  }

  async restoreChat(
    chat_id: string | number,
    options?: { expectedVersion?: number; scope?: ChatScope } & ScopedRequestOptions
  ): Promise<ServerChatSummary> {
    const scopeFields = requestScopeFields(options?.requestScope)
    const cid = String(chat_id)
    const query = this.buildQuery({
      ...toChatScopeParams(options?.scope),
      ...(typeof options?.expectedVersion === "number" ? {expected_version:options.expectedVersion} : {})
    })
    const res = await bgRequest<any>({
      ...scopeFields,
      ...(options?.signal ? {abortSignal:options.signal} : {}),
      path: `/api/v1/chats/${cid}/restore${query}`,
      method: "POST"
    })
    // normalizeChatSummary's runtime implementation is the chat-rag mixin
    // (pinned by the prototype-ownership guard); typed via the merged core.
    return (this as unknown as import("./TldwApiClient").TldwApiClientCore).normalizeChatSummary(res)
  }

  async createConversationShareLink(
    chat_id: string | number,
    payload?: {
      permission?: ConversationSharePermission
      ttl_seconds?: number
      label?: string
    },
    options?: { scope?: ChatScope } & ScopedRequestOptions
  ): Promise<ConversationShareLinkCreateResponse> {
    const scopeFields = requestScopeFields(options?.requestScope)
    const cid = String(chat_id)
    const query = this.buildQuery(toChatScopeParams(options?.scope))
    return await bgRequest<ConversationShareLinkCreateResponse>({
      ...scopeFields,
      ...(options?.signal ? { abortSignal: options.signal } : {}),
      path: appendPathQuery(
        `/api/v1/chat/conversations/${encodeURIComponent(cid)}/share-links`,
        query
      ),
      method: "POST",
      headers: { "Content-Type": "application/json", ...scopeFields.headers },
      body: payload || {},
    })
  }

  async listConversationShareLinks(
    chat_id: string | number,
    options?: { scope?: ChatScope }
  ): Promise<ConversationShareLinksListResponse> {
    const cid = String(chat_id)
    const query = this.buildQuery(toChatScopeParams(options?.scope))
    return await bgRequest<ConversationShareLinksListResponse>({
      path: appendPathQuery(
        `/api/v1/chat/conversations/${encodeURIComponent(cid)}/share-links`,
        query
      ),
      method: "GET",
    })
  }

  async revokeConversationShareLink(
    chat_id: string | number,
    shareId: string,
    options?: { scope?: ChatScope } & ScopedRequestOptions
  ): Promise<{ success: boolean; share_id: string }> {
    const scopeFields = requestScopeFields(options?.requestScope)
    const cid = encodeURIComponent(String(chat_id))
    const sid = encodeURIComponent(String(shareId))
    const query = this.buildQuery(toChatScopeParams(options?.scope))
    return await bgRequest<{ success: boolean; share_id: string }>({
      ...scopeFields,
      ...(options?.signal ? { abortSignal: options.signal } : {}),
      path: appendPathQuery(
        `/api/v1/chat/conversations/${cid}/share-links/${sid}`,
        query
      ),
      method: "DELETE",
    })
  }


  async listChatMessages(
    chat_id: string | number,
    params?: Record<string, any>,
    options?: { signal?: AbortSignal; scope?: ChatScope; requestScope?: ServicePromptRequestScope; fresh?: boolean }
  ): Promise<ServerChatMessage[]> {
    const cid = String(chat_id)
    const query = this.buildQuery({
      ...toChatScopeParams(options?.scope),
      ...(params || {})
    })
    const cacheKey = this.getChatMessagesCacheKey(cid, query)
    const useSharedCache = !options?.fresh && !options?.requestScope
    const cacheRevision = useSharedCache ? await this.getDomainCacheRevision() : 0
    if (useSharedCache) this.assertDomainCacheRevision(cacheRevision)
    const cached = useSharedCache ? this.chatMessagesCache.get(cacheKey) : undefined
    if (cached && cached.expiresAt > Date.now()) {
      return cached.value
    }
    if (cached) {
      this.chatMessagesCache.delete(cacheKey)
    }

    const inFlight = useSharedCache ? this.chatMessagesInFlight.get(cacheKey) : undefined
    if (inFlight) {
      return inFlight
    }

    const request = (async () => {
      const scopeFields = requestScopeFields(options?.requestScope)
      const data = await bgRequest<any>({
        path: `/api/v1/chats/${cid}/messages${query}`,
        method: "GET",
        ...(useSharedCache ? { configSnapshot: this.getDomainCacheConfigSnapshot(cacheRevision) } : {}),
        abortSignal: options?.signal,
        ...(scopeFields.servicePromptConfig ? {
          headers: scopeFields.headers,
          servicePromptConfig: scopeFields.servicePromptConfig
        } : {})
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
      if (useSharedCache) {
        await this.getDomainCacheRevision()
        this.assertDomainCacheRevision(cacheRevision)
        this.chatMessagesCache.set(cacheKey, {
          value: normalized,
          expiresAt: Date.now() + CHAT_MESSAGES_CACHE_TTL_MS
        })
      }
      return normalized
    })()

    if (useSharedCache) this.chatMessagesInFlight.set(cacheKey, request)
    try {
      return await request
    } finally {
      if (useSharedCache && this.chatMessagesInFlight.get(cacheKey) === request) {
        this.chatMessagesInFlight.delete(cacheKey)
      }
    }
  }

  async addChatMessage(
    chat_id: string | number,
    payload: Record<string, any>,
    options?: {
      scope?: ChatScope
      signal?: AbortSignal
      requestScope?: ServicePromptRequestScope
    }
  ): Promise<ServerChatMessage> {
    const cid = String(chat_id)
    const query = this.buildQuery(toChatScopeParams(options?.scope))
    const scopeFields = requestScopeFields(options?.requestScope)
    const res = await bgRequest<ServerChatMessage>({
      path: appendPathQuery(`/api/v1/chats/${cid}/messages`, query),
      method: "POST",
      headers: { "Content-Type": "application/json", ...scopeFields.headers },
      body: payload,
      abortSignal: options?.signal,
      ...(scopeFields.servicePromptConfig
        ? { servicePromptConfig: scopeFields.servicePromptConfig }
        : {})
    })
    this.invalidateChatMessagesCache(cid)
    return res
  }



  async persistCharacterCompletion(
    chat_id: string | number,
    payload: Record<string, any>,
    options?: { scope?: ChatScope; signal?: AbortSignal; requestScope?: ServicePromptRequestScope }
  ): Promise<any> {
    return chatRagMethods.persistCharacterCompletion.call(this as any, chat_id, payload, options)
  }

  async *streamCharacterChatCompletion(
    chat_id: string | number,
    payload?: Record<string, any>,
    options?: {
      signal?: AbortSignal
      streamIdleTimeoutMs?: number
      scope?: ChatScope
      requestScope?: ServicePromptRequestScope
      onDispatch?: () => void
    }
  ): AsyncGenerator<any> {
    yield* chatRagMethods.streamCharacterChatCompletion(
      chat_id,
      payload,
      options
    )
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
  async getMessage(
    message_id: string | number,
    options?: { scope?: ChatScope } & ScopedRequestOptions
  ): Promise<any> {
    return chatRagMethods.getMessage.call(this as unknown as TldwApiClientCore, message_id, options)
  }

  async editMessage(
    message_id: string | number,
    content: string,
    expectedVersion: number,
    chatId?: string | number,
    options?: { pinned?: boolean; scope?: ChatScope } & ScopedRequestOptions
  ): Promise<any> {
    return chatRagMethods.editMessage.call(this as unknown as TldwApiClientCore, message_id, content, expectedVersion, chatId, options)
  }

  async deleteMessage(
    message_id: string | number,
    expectedVersion: number,
    chatId?: string | number,
    options?: { scope?: ChatScope; requireLeaf?: boolean; expectedRevision?: string } & ScopedRequestOptions
  ): Promise<void> {
    return chatRagMethods.deleteMessage.call(this as unknown as TldwApiClientCore, message_id, expectedVersion, chatId, options)
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












  async listDictionaryEntries(dictionary_id: number | string, group?: string): Promise<any> {
    const id = String(dictionary_id)
    const qp = group ? `?group=${encodeURIComponent(group)}` : ''
    return await bgRequest<any>({ path: `/api/v1/chat/dictionaries/${id}/entries${qp}`, method: 'GET' })
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












  // Explainer Workspace
  async createExplainerSession(payload: ExplainerSessionCreatePayload): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/explainer/sessions",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async listExplainerSessions(params?: { limit?: number; offset?: number }): Promise<any> {
    const query = this.buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/explainer/sessions${query}`,
      method: "GET"
    })
  }

  async getExplainerSession(sessionId: string): Promise<any> {
    const id = encodeURIComponent(String(sessionId))
    return await bgRequest<any>({
      path: `/api/v1/explainer/sessions/${id}`,
      method: "GET"
    })
  }

  async updateExplainerSession(
    sessionId: string,
    payload: ExplainerSessionPatchPayload
  ): Promise<any> {
    const id = encodeURIComponent(String(sessionId))
    return await bgRequest<any>({
      path: `/api/v1/explainer/sessions/${id}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async deleteExplainerSession(sessionId: string): Promise<any> {
    const id = encodeURIComponent(String(sessionId))
    return await bgRequest<any>({
      path: `/api/v1/explainer/sessions/${id}`,
      method: "DELETE"
    })
  }

  async createExplainerNode(
    sessionId: string,
    payload: ExplainerNodeCreatePayload
  ): Promise<any> {
    const id = encodeURIComponent(String(sessionId))
    return await bgRequest<any>({
      path: `/api/v1/explainer/sessions/${id}/nodes`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async updateExplainerNode(
    sessionId: string,
    nodeId: string,
    payload: ExplainerNodePatchPayload
  ): Promise<any> {
    const sid = encodeURIComponent(String(sessionId))
    const nid = encodeURIComponent(String(nodeId))
    return await bgRequest<any>({
      path: `/api/v1/explainer/sessions/${sid}/nodes/${nid}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async deleteExplainerNode(sessionId: string, nodeId: string): Promise<any> {
    const sid = encodeURIComponent(String(sessionId))
    const nid = encodeURIComponent(String(nodeId))
    return await bgRequest<any>({
      path: `/api/v1/explainer/sessions/${sid}/nodes/${nid}`,
      method: "DELETE"
    })
  }

  async expandExplainerNode(
    sessionId: string,
    nodeId: string,
    payload?: ExplainerNodeExpandPayload
  ): Promise<any> {
    const sid = encodeURIComponent(String(sessionId))
    const nid = encodeURIComponent(String(nodeId))
    return await bgRequest<any>({
      path: `/api/v1/explainer/sessions/${sid}/nodes/${nid}/expand`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload ?? {}
    })
  }

  async answerExplainerQuestion(
    sessionId: string,
    nodeId: string,
    payload: ExplainerQuestionAnswerPayload
  ): Promise<any> {
    const sid = encodeURIComponent(String(sessionId))
    const nid = encodeURIComponent(String(nodeId))
    return await bgRequest<any>({
      path: `/api/v1/explainer/sessions/${sid}/nodes/${nid}/answer-question`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }

  async getExplainerJob(jobId: string): Promise<any> {
    const id = encodeURIComponent(String(jobId))
    return await bgRequest<any>({
      path: `/api/v1/explainer/jobs/${id}`,
      method: "GET"
    })
  }

  async exportExplainerChatbook(
    sessionId: string,
    payload?: ExplainerChatbookExportPayload
  ): Promise<any> {
    const id = encodeURIComponent(String(sessionId))
    return await bgRequest<any>({
      path: `/api/v1/explainer/sessions/${id}/export-chatbook`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload ?? {}
    })
  }


  async getChatbookExportScope(): Promise<ChatbookAccountScopeResponse> {
    return await bgRequest<ChatbookAccountScopeResponse>({
      path: "/api/v1/chatbooks/export/scope",
      method: "GET"
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
      source_format?: string
      selected_openwebui_user_id?: string
    }
  ): Promise<any> {
    const data = await file.arrayBuffer()
    const name = file.name || "chatbook.zip"
    const type = file.type || "application/zip"
    const normalized: Record<string, any> = {}
    for (const [k, v] of Object.entries(options || {})) {
      if (typeof v === "undefined" || v === null) continue
      if (typeof v === "boolean") {
        normalized[k] = v ? "true" : "false"
      } else if (typeof v === "object") {
        normalized[k] = JSON.stringify(v)
      } else {
        normalized[k] = v
      }
    }
    return await bgUpload<any>({
      path: "/api/v1/chatbooks/import",
      method: "POST",
      fields: normalized,
      file: { name, type, data }
    })
  }


  async listOpenWebUIImportScopes(): Promise<OpenWebUIImportScopesResponse> {
    return await bgRequest<OpenWebUIImportScopesResponse>({
      path: "/api/v1/chatbooks/openwebui/import-scopes",
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


  // ─────────────────────────────────────────────────────────────────────────────
  // File Artifacts
  // ─────────────────────────────────────────────────────────────────────────────

  async createImageArtifact(
    request: ImageArtifactRequest,
    options?: ScopedRequestOptions & { onDispatch?: () => void }
  ): Promise<FileCreateResponse> {
    return await mediaMethods.createImageArtifact(request, options)
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Collections / Reading List API
  // ─────────────────────────────────────────────────────────────────────────────






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
import {
  chatRagMethods,
} from "./domains/chat-rag"
import { collectionsMethods } from "./domains/collections"
import { modelsAudioMethods } from "./domains/models-audio"
import { presentationsMethods } from "./domains/presentations"
import { prototypeWorkspaceMethods } from "./domains/prototype-workspaces"
import { setupOnboardingMethods } from "./domains/setup-onboarding"
import { workspaceApiMethods } from "./domains/workspace-api"
import { webClipperMethods } from "./domains/web-clipper"
import { visualIdentityMethods } from "./domains/visual-identities"
import {
  requestScopeFields,
  servicePromptMethods,
  type ServicePromptRequestScope
} from "./domains/service-prompts"

export type {
  TldwSpeechDetailedResult,
  TldwSpeechOptions
} from "./domains/models-audio"

// eslint-disable-next-line @typescript-eslint/no-unsafe-declaration-merging
export class TldwApiClient extends TldwApiClientBase {}

// Declaration merging: extend the class type with all domain methods
type TldwDomainMethodOverride =
  | keyof TldwApiClientBase
  | "normalizeCharacterListResponse"
  | "getCharacterListIdentity"
  | "normalizeChatSummary"
type TldwDomainMethods<T> = Omit<T, TldwDomainMethodOverride>

// Shared helpers that live in domain mixins but are referenced through the
// merged client type by sibling domains and the base.
export interface TldwApiClientCharacterHelpers {
  normalizeCharacterListResponse: (typeof characterMethods)["normalizeCharacterListResponse"]
  getCharacterListIdentity: (typeof characterMethods)["getCharacterListIdentity"]
  normalizeChatSummary: (typeof chatRagMethods)["normalizeChatSummary"]
}

// eslint-disable-next-line @typescript-eslint/no-unsafe-declaration-merging, no-redeclare
export interface TldwApiClient
  extends
    TldwApiClientCharacterHelpers,
    TldwDomainMethods<typeof adminMethods>,
    TldwDomainMethods<typeof mediaMethods>,
    TldwDomainMethods<typeof characterMethods>,
    TldwDomainMethods<typeof chatRagMethods>,
    TldwDomainMethods<typeof collectionsMethods>,
    TldwDomainMethods<typeof modelsAudioMethods>,
    TldwDomainMethods<typeof presentationsMethods>,
    TldwDomainMethods<typeof prototypeWorkspaceMethods>,
    TldwDomainMethods<typeof setupOnboardingMethods>,
    TldwDomainMethods<typeof workspaceApiMethods>,
    TldwDomainMethods<typeof webClipperMethods>,
    TldwDomainMethods<typeof visualIdentityMethods>,
    TldwDomainMethods<typeof servicePromptMethods> {}

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
  prototypeWorkspaceMethods,
  setupOnboardingMethods,
  workspaceApiMethods,
  webClipperMethods,
  visualIdentityMethods,
  servicePromptMethods
)

// createChatCompletion remains canonical on TldwApiClientBase. TTS synthesis is
// canonical in modelsAudioMethods so explicit-backend negotiation and response
// provenance stay available while preserving the configured request timeout.
Object.assign(TldwApiClient.prototype, {
  createChatCompletion: TldwApiClientBase.prototype.createChatCompletion
})

// Also expose core helpers that domain files reference via `this`
export type TldwApiClientCore = TldwApiClient

// Singleton instance
export const tldwClient = new TldwApiClient()
