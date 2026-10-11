import {
  isActiveCookieSessionConfig,
  tldwClient,
  type TldwConfig
} from "./TldwApiClient"
import { bgRequest } from "@/services/background-proxy"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import {
  getRuntimeSingleUserApiKeyOverride
} from "@/services/tldw/runtime-auth-override"
import { isPlaceholderApiKey } from "@/utils/api-key"
import { createSafeStorage } from "@/utils/safe-storage"
import { requestScopeFields, type ServicePromptRequestScope } from "./domains/service-prompts"

export type ServerCapabilities = {
  hasChat: boolean
  hasRag: boolean
  hasMedia: boolean
  hasMediaPlaylistPreflight?: boolean
  hasMediaIngestJobs?: boolean
  hasMediaIngestJobEvents?: boolean
  hasMediaIngestWorker?: boolean
  hasDurableMediaCollections?: boolean
  hasKnowledgeQaMediaScope?: boolean
  hasNotes: boolean
  hasSlides: boolean
  hasPresentationStudio: boolean
  hasPresentationRender: boolean
  hasIngestionSources: boolean
  canCreateLocalDirectoryIngestionSource: boolean | null
  hasPrompts: boolean
  hasFlashcards: boolean
  hasQuizzes: boolean
  hasCharacters: boolean
  hasWorldBooks: boolean
  hasChatDictionaries: boolean
  hasChatKnowledgeSave: boolean
  hasChatDocuments: boolean
  hasChatbooks: boolean
  hasChatQueue: boolean
  hasChatSaveToDb: boolean
  hasChatTurnIdentity?: boolean
  hasWebClipper: boolean
  hasStt: boolean
  hasTts: boolean
  hasVoiceChat: boolean
  hasVoiceConversationTransport: boolean
  hasAudio: boolean
  hasEmbeddings: boolean
  hasMetrics: boolean
  hasMcp: boolean
  hasReading: boolean
  hasWriting: boolean
  hasWebSearch: boolean
  hasFeedbackExplicit: boolean
  hasFeedbackImplicit: boolean
  hasSkills: boolean
  hasPersona: boolean
  hasPersonaLiveControl: boolean
  hasPersonalization: boolean
  hasGuardian: boolean
  hasSelfMonitoring: boolean
  ffmpegAvailable: boolean | null
  specVersion: string | null
  specSource: "authoritative" | "fallback"
}

const defaultCapabilities: ServerCapabilities = {
  hasChat: false,
  hasRag: false,
  hasMedia: false,
  hasMediaPlaylistPreflight: false,
  hasMediaIngestJobs: false,
  hasMediaIngestJobEvents: false,
  hasMediaIngestWorker: false,
  hasDurableMediaCollections: false,
  hasKnowledgeQaMediaScope: false,
  hasNotes: false,
  hasSlides: false,
  hasPresentationStudio: false,
  hasPresentationRender: false,
  hasIngestionSources: false,
  canCreateLocalDirectoryIngestionSource: false,
  hasPrompts: false,
  hasFlashcards: false,
  hasQuizzes: false,
  hasCharacters: false,
  hasWorldBooks: false,
  hasChatDictionaries: false,
  hasChatKnowledgeSave: false,
  hasChatDocuments: false,
  hasChatbooks: false,
  hasChatQueue: false,
  hasChatSaveToDb: false,
  hasChatTurnIdentity: false,
  hasWebClipper: false,
  hasStt: false,
  hasTts: false,
  hasVoiceChat: false,
  hasVoiceConversationTransport: false,
  hasAudio: false,
  hasEmbeddings: false,
  hasMetrics: false,
  hasMcp: false,
  hasReading: false,
  hasWriting: false,
  hasWebSearch: false,
  hasFeedbackExplicit: false,
  hasFeedbackImplicit: false,
  hasSkills: false,
  hasPersona: false,
  hasPersonaLiveControl: false,
  hasPersonalization: false,
  hasGuardian: false,
  hasSelfMonitoring: false,
  ffmpegAvailable: null,
  specVersion: null,
  specSource: "authoritative"
}

const fallbackSpec = {
  info: { version: "local-fallback" },
  paths: Object.fromEntries(
    [
      "/api/v1/chat/completions",
      "/api/v1/feedback/explicit",
      "/api/v1/rag/search",
      "/api/v1/rag/health",
      "/api/v1/rag/",
      "/api/v1/rag/feedback/implicit",
      "/api/v1/media/playlists/preflight",
      "/api/v1/media/ingest/jobs",
      "/api/v1/media/ingest/jobs/events/stream",
      "/api/v1/media/collections",
      "/api/v1/media/add",
      "/api/v1/media/",
      "/api/v1/media/process-videos",
      "/api/v1/media/process-documents",
      "/api/v1/media/process-pdfs",
      "/api/v1/media/process-ebooks",
      "/api/v1/media/process-audios",
      "/api/v1/notes/",
      "/api/v1/slides/generate/from-media",
      "/api/v1/slides/presentations",
      "/api/v1/slides/presentations/{presentation_id}",
      "/api/v1/slides/presentations/{presentation_id}/export",
      "/api/v1/slides/presentations/{presentation_id}/render-jobs",
      "/api/v1/slides/render-jobs/{job_id}",
      "/api/v1/slides/presentations/{presentation_id}/render-artifacts",
      "/api/v1/ingestion-sources",
      "/api/v1/prompts",
      "/api/v1/flashcards",
      "/api/v1/flashcards/decks",
      "/api/v1/quizzes",
      "/api/v1/quizzes/generate",
      "/api/v1/characters",
      "/api/v1/characters/world-books",
      "/api/v1/chat/dictionaries",
      "/api/v1/chat/dictionaries/validate",
      "/api/v1/chat/dictionaries/process",
      "/api/v1/chat/knowledge/save",
      "/api/v1/chat/documents",
      "/api/v1/chat/documents/generate",
      "/api/v1/chat/documents/bulk",
      "/api/v1/chat/documents/jobs",
      "/api/v1/chat/documents/prompts",
      "/api/v1/chat/documents/statistics",
      "/api/v1/chat/queue/status",
      "/api/v1/chat/queue/activity",
      "/api/v1/chatbooks/export",
      "/api/v1/chatbooks/preview",
      "/api/v1/chatbooks/import",
      "/api/v1/chatbooks/export/jobs",
      "/api/v1/chatbooks/import/jobs",
      "/api/v1/chatbooks/download",
      "/api/v1/chatbooks/cleanup",
      "/api/v1/chatbooks/health",
      "/api/v1/audio/transcriptions",
      "/api/v1/audio/transcriptions/health",
      "/api/v1/audio/speech",
      "/api/v1/audio/voice-conversion",
      "/api/v1/audio/tts/providers/{provider}/unload",
      "/api/v1/audio/voices/catalog",
      "/api/v1/audio/health",
      "/api/v1/audio/stream/transcribe",
      "/api/v1/audio/chat/stream",
      "/api/v1/embeddings/models",
      "/api/v1/embeddings/providers-config",
      "/api/v1/embeddings/health",
      "/api/v1/metrics/health",
      "/api/v1/metrics",
      "/api/v1/mcp/health",
      "/api/v1/reading/save",
      "/api/v1/reading/items",
      "/api/v1/writing/version",
      "/api/v1/writing/capabilities",
      "/api/v1/writing/sessions",
      "/api/v1/writing/templates",
      "/api/v1/writing/themes",
      "/api/v1/writing/tokenize",
      "/api/v1/writing/token-count",
      "/api/v1/research/websearch",
      "/api/v1/skills/",
      "/api/v1/skills/context",
      "/api/v1/persona/catalog",
      "/api/v1/persona/session",
      "/api/v1/persona/live/sessions",
      "/api/v1/persona/stream",
      "/api/v1/personalization/profile",
      "/api/v1/personalization/opt-in",
      "/api/v1/personalization/memories"
    ].map((p) => [p, {}])
  )
}

type DocsInfoResponse = {
  capabilities?: Record<string, unknown> | null
  supported_features?: Record<string, unknown> | null
  ffmpeg_available?: boolean | null
}

type IngestionSourceCapabilitiesResponse = {
  can_create_local_directory?: unknown
}

const CAPABILITIES_CACHE_TTL_MS = 5 * 60 * 1000
const CAPABILITIES_STORAGE_KEY = "__tldwServerCapabilitiesCacheV5"

type CapabilitiesCachePayload = {
  key: string
  fetchedAt: number
  capabilities: ServerCapabilities
}

export type ServerCapabilitiesCacheDiagnostics = {
  calls: number
  forceRefreshCalls: number
  inMemoryHits: number
  persistedHits: number
  inFlightHits: number
  staleMemoryMisses: number
  stalePersistedMisses: number
  networkFetches: number
  networkErrors: number
  fallbackSpecUses: number
  lastSource: "in-memory" | "persisted" | "in-flight" | "network" | "fallback" | null
  lastCacheKey: string | null
  lastFetchAt: number | null
  lastFetchDurationMs: number | null
  lastError: string | null
  inMemoryCacheEntries: number
  inFlightRequests: number
}

const DIAGNOSTICS_LOG_INTERVAL_MS = 30_000

const createEmptyDiagnostics = (): Omit<
  ServerCapabilitiesCacheDiagnostics,
  "inMemoryCacheEntries" | "inFlightRequests"
> => ({
  calls: 0,
  forceRefreshCalls: 0,
  inMemoryHits: 0,
  persistedHits: 0,
  inFlightHits: 0,
  staleMemoryMisses: 0,
  stalePersistedMisses: 0,
  networkFetches: 0,
  networkErrors: 0,
  fallbackSpecUses: 0,
  lastSource: null,
  lastCacheKey: null,
  lastFetchAt: null,
  lastFetchDurationMs: null,
  lastError: null
})

const capabilitiesDiagnostics = createEmptyDiagnostics()
let lastDiagnosticsLogAt = 0

const normalizePaths = (raw: any): Record<string, any> => {
  const out: Record<string, any> = {}
  if (!raw || typeof raw !== "object") return out
  for (const key of Object.keys(raw)) {
    const k = key.trim()
    out[k] = raw[key]
    if (k.endsWith("/")) {
      out[k.slice(0, -1)] = raw[key]
    } else {
      out[`${k}/`] = raw[key]
    }
  }
  return out
}

const resolveSchemaRef = (schema: any, spec: any): any => {
  if (!schema || typeof schema !== "object") return schema
  const ref = schema.$ref
  if (typeof ref !== "string") return schema
  const prefix = "#/components/schemas/"
  if (!ref.startsWith(prefix)) return schema
  const name = ref.slice(prefix.length)
  const resolved = spec?.components?.schemas?.[name]
  return resolved || schema
}

const schemaHasProperty = (
  schema: any,
  property: string,
  spec: any,
  seen: Set<string> = new Set()
): boolean => {
  if (!schema || typeof schema !== "object") return false
  const ref = typeof schema.$ref === "string" ? schema.$ref : null
  if (ref) {
    if (seen.has(ref)) return false
    seen.add(ref)
    return schemaHasProperty(resolveSchemaRef(schema, spec), property, spec, seen)
  }
  if (schema.properties && schema.properties[property]) return true

  const combos = [
    ...(Array.isArray(schema.allOf) ? schema.allOf : []),
    ...(Array.isArray(schema.anyOf) ? schema.anyOf : []),
    ...(Array.isArray(schema.oneOf) ? schema.oneOf : [])
  ]
  for (const entry of combos) {
    if (schemaHasProperty(entry, property, spec, seen)) return true
  }
  return false
}

const detectChatSaveToDb = (spec: any): boolean => {
  const post = spec?.paths?.["/api/v1/chat/completions"]?.post
  const schema =
    post?.requestBody?.content?.["application/json"]?.schema ??
    post?.requestBody?.content?.["application/json;charset=utf-8"]?.schema
  return schemaHasProperty(schema, "save_to_db", spec)
}

const parseBooleanish = (raw: unknown): boolean | null => {
  if (typeof raw === "boolean") return raw
  if (typeof raw === "number") return raw !== 0
  if (typeof raw !== "string") return null
  const normalized = raw.trim().toLowerCase()
  if (!normalized) return null
  if (["true", "1", "yes", "on", "enabled"].includes(normalized)) {
    return true
  }
  if (["false", "0", "no", "off", "disabled"].includes(normalized)) {
    return false
  }
  return null
}

const extractFeatureFlag = (
  docsInfo: DocsInfoResponse | null | undefined,
  key: string
): boolean | null => {
  const maps: Array<Record<string, unknown> | null | undefined> = [
    docsInfo?.capabilities,
    docsInfo?.supported_features
  ]
  for (const map of maps) {
    if (!map || typeof map !== "object" || !(key in map)) {
      continue
    }
    const parsed = parseBooleanish(map[key])
    if (parsed !== null) {
      return parsed
    }
  }
  return null
}

const applyDocsInfoFeatureGates = (
  capabilities: ServerCapabilities,
  docsInfo: DocsInfoResponse | null | undefined
): ServerCapabilities => {
  const audioFeatureEnabled = extractFeatureFlag(docsInfo, "hasAudio")
  const sttFeatureEnabled = extractFeatureFlag(docsInfo, "hasStt")
  const ttsFeatureEnabled = extractFeatureFlag(docsInfo, "hasTts")
  const voiceChatFeatureEnabled = extractFeatureFlag(docsInfo, "hasVoiceChat")
  const voiceConversationTransportFeatureEnabled = extractFeatureFlag(
    docsInfo,
    "hasVoiceConversationTransport"
  )
  const slidesFeatureEnabled = extractFeatureFlag(docsInfo, "hasSlides")
  const presentationStudioFeatureEnabled = extractFeatureFlag(
    docsInfo,
    "hasPresentationStudio"
  )
  const presentationRenderFeatureEnabled = extractFeatureFlag(
    docsInfo,
    "hasPresentationRender"
  )
  const mediaPlaylistPreflightEnabled = extractFeatureFlag(
    docsInfo,
    "hasMediaPlaylistPreflight"
  )
  const mediaIngestJobsEnabled = extractFeatureFlag(docsInfo, "hasMediaIngestJobs")
  const mediaIngestJobEventsEnabled = extractFeatureFlag(
    docsInfo,
    "hasMediaIngestJobEvents"
  )
  const mediaIngestWorkerEnabled = extractFeatureFlag(
    docsInfo,
    "hasMediaIngestWorker"
  )
  const durableMediaCollectionsEnabled = extractFeatureFlag(
    docsInfo,
    "hasDurableMediaCollections"
  )
  const knowledgeQaMediaScopeEnabled = extractFeatureFlag(
    docsInfo,
    "hasKnowledgeQaMediaScope"
  )
  const personaFeatureEnabled = extractFeatureFlag(docsInfo, "persona")
  const personaLiveControlFeatureEnabled = extractFeatureFlag(
    docsInfo,
    "hasPersonaLiveControl"
  )
  const personalizationFeatureEnabled = extractFeatureFlag(
    docsInfo,
    "personalization"
  )
  const mergeFeatureFlag = (
    computed: boolean | undefined,
    explicit: boolean | null
  ): boolean => (explicit === null ? Boolean(computed) : explicit)
  const hasStt = mergeFeatureFlag(capabilities.hasStt, sttFeatureEnabled)
  const hasTts = mergeFeatureFlag(capabilities.hasTts, ttsFeatureEnabled)
  const hasVoiceChat = mergeFeatureFlag(
    capabilities.hasVoiceChat,
    voiceChatFeatureEnabled
  )
  const hasVoiceConversationTransport = mergeFeatureFlag(
    capabilities.hasVoiceConversationTransport,
    voiceConversationTransportFeatureEnabled
  )
  const hasAudio =
    mergeFeatureFlag(capabilities.hasAudio, audioFeatureEnabled) ||
    hasStt ||
    hasTts ||
    hasVoiceChat
  const hasSlides = mergeFeatureFlag(capabilities.hasSlides, slidesFeatureEnabled)
  const hasPresentationStudio =
    mergeFeatureFlag(
      capabilities.hasPresentationStudio,
      presentationStudioFeatureEnabled
    ) && hasSlides
  const hasPresentationRender =
    mergeFeatureFlag(
      capabilities.hasPresentationRender,
      presentationRenderFeatureEnabled
    ) && hasPresentationStudio

  return {
    ...capabilities,
    hasAudio,
    hasStt,
    hasTts,
    hasVoiceChat,
    hasVoiceConversationTransport,
    hasSlides,
    hasPresentationStudio,
    hasPresentationRender,
    hasMediaPlaylistPreflight: mergeFeatureFlag(
      capabilities.hasMediaPlaylistPreflight,
      mediaPlaylistPreflightEnabled
    ),
    hasMediaIngestJobs: mergeFeatureFlag(
      capabilities.hasMediaIngestJobs,
      mediaIngestJobsEnabled
    ),
    hasMediaIngestJobEvents: mergeFeatureFlag(
      capabilities.hasMediaIngestJobEvents,
      mediaIngestJobEventsEnabled
    ),
    hasMediaIngestWorker: mergeFeatureFlag(
      capabilities.hasMediaIngestWorker,
      mediaIngestWorkerEnabled
    ),
    hasDurableMediaCollections: mergeFeatureFlag(
      capabilities.hasDurableMediaCollections,
      durableMediaCollectionsEnabled
    ),
    hasKnowledgeQaMediaScope: mergeFeatureFlag(
      capabilities.hasKnowledgeQaMediaScope,
      knowledgeQaMediaScopeEnabled
    ),
    hasPersona:
      personaFeatureEnabled === null
        ? capabilities.hasPersona
        : capabilities.hasPersona && personaFeatureEnabled,
    hasPersonaLiveControl:
      personaLiveControlFeatureEnabled === null
        ? capabilities.hasPersonaLiveControl
        : capabilities.hasPersonaLiveControl && personaLiveControlFeatureEnabled,
    hasPersonalization:
      personalizationFeatureEnabled === null
        ? capabilities.hasPersonalization
        : capabilities.hasPersonalization && personalizationFeatureEnabled,
    ffmpegAvailable:
      docsInfo?.ffmpeg_available != null
        ? Boolean(docsInfo.ffmpeg_available)
        : capabilities.ffmpegAvailable
  }
}

const applyIngestionSourceCapabilityGates = (
  capabilities: ServerCapabilities,
  sourceCapabilities: IngestionSourceCapabilitiesResponse | null | undefined
): ServerCapabilities => {
  const canCreateLocalDirectory = parseBooleanish(
    sourceCapabilities?.can_create_local_directory
  )
  if (canCreateLocalDirectory === null) {
    return capabilities
  }
  return {
    ...capabilities,
    canCreateLocalDirectoryIngestionSource: canCreateLocalDirectory
  }
}

const computeCapabilities = (
  spec: any | null | undefined,
  specSource: "authoritative" | "fallback" = "authoritative"
): ServerCapabilities => {
  if (!spec || typeof spec !== "object") {
    return { ...defaultCapabilities, specSource }
  }
  const paths = normalizePaths(spec.paths || {})
  const has = (p: string) => Boolean(paths[p])
  const hasChatSaveToDb = detectChatSaveToDb(spec)
  const chatRequestContent =
    spec?.paths?.["/api/v1/chat/completions"]?.post?.requestBody?.content
  const hasChatTurnIdentity = schemaHasProperty(
    chatRequestContent?.["application/json"]?.schema ??
      chatRequestContent?.["application/json;charset=utf-8"]?.schema,
    "tldw_turn",
    spec
  )
  const hasSlidesRoutes =
    has("/api/v1/slides/generate/from-media") ||
    has("/api/v1/slides/presentations") ||
    has("/api/v1/slides/presentations/{presentation_id}") ||
    has("/api/v1/slides/presentations/{presentation_id}/export")
  const hasPresentationRender =
    has("/api/v1/slides/presentations/{presentation_id}/render-jobs") ||
    has("/api/v1/slides/render-jobs/{job_id}") ||
    has("/api/v1/slides/presentations/{presentation_id}/render-artifacts")
  const hasSlides = hasSlidesRoutes || hasPresentationRender
  const hasPresentationStudio = hasSlides
  const hasStt =
    has("/api/v1/audio/transcriptions") ||
    has("/api/v1/audio/transcriptions/health") ||
    has("/api/v1/audio/stream/transcribe") ||
    has("/api/v1/audio/chat/stream")
  const hasTts =
    has("/api/v1/audio/speech") ||
    has("/api/v1/audio/health") ||
    has("/api/v1/audio/voices/catalog") ||
    has("/api/v1/audio/chat/stream")
  const hasVoiceChat =
    has("/api/v1/audio/chat/stream") || (hasStt && hasTts)
  const hasVoiceConversationTransport =
    specSource === "fallback" ? false : has("/api/v1/audio/chat/stream")
  const hasMediaPlaylistPreflight = has("/api/v1/media/playlists/preflight")
  const hasMediaIngestJobs = has("/api/v1/media/ingest/jobs")
  const hasMediaIngestJobEvents = has("/api/v1/media/ingest/jobs/events/stream")
  const hasDurableMediaCollections = has("/api/v1/media/collections")
  const hasIngestionSources =
    has("/api/v1/ingestion-sources") ||
    has("/api/v1/ingestion-sources/{source_id}") ||
    has("/api/v1/ingestion-sources/{source_id}/items")
  const hasPersonaLiveControl = has("/api/v1/persona/live/sessions")

  return {
    hasChat: has("/api/v1/chat/completions"),
    hasRag: has("/api/v1/rag/search") || has("/api/v1/rag/health") || has("/api/v1/rag/"),
    hasMedia:
      hasMediaPlaylistPreflight ||
      hasMediaIngestJobs ||
      hasDurableMediaCollections ||
      has("/api/v1/media/add") ||
      has("/api/v1/media/") ||
      has("/api/v1/media/process-videos") ||
      has("/api/v1/media/process-documents"),
    hasMediaPlaylistPreflight,
    hasMediaIngestJobs,
    hasMediaIngestJobEvents,
    hasMediaIngestWorker: false,
    hasDurableMediaCollections,
    hasKnowledgeQaMediaScope: false,
    hasNotes: has("/api/v1/notes/"),
    hasSlides,
    hasPresentationStudio,
    hasPresentationRender,
    hasIngestionSources,
    canCreateLocalDirectoryIngestionSource: hasIngestionSources ? null : false,
    hasPrompts: has("/api/v1/prompts") || has("/api/v1/prompts/"),
    hasFlashcards:
      has("/api/v1/flashcards") ||
      has("/api/v1/flashcards/") ||
      has("/api/v1/flashcards/decks"),
    hasQuizzes:
      has("/api/v1/quizzes") ||
      has("/api/v1/quizzes/") ||
      has("/api/v1/quizzes/generate"),
    hasCharacters: has("/api/v1/characters") || has("/api/v1/characters/"),
    hasWorldBooks: has("/api/v1/characters/world-books"),
    hasChatDictionaries: has("/api/v1/chat/dictionaries"),
    hasChatKnowledgeSave: has("/api/v1/chat/knowledge/save"),
    hasChatDocuments: has("/api/v1/chat/documents") || has("/api/v1/chat/documents/generate"),
    hasChatbooks: has("/api/v1/chatbooks/export") || has("/api/v1/chatbooks/health"),
    hasChatQueue: has("/api/v1/chat/queue/status") || has("/api/v1/chat/queue/activity"),
    hasChatSaveToDb,
    hasChatTurnIdentity,
    hasWebClipper: has("/api/v1/web-clipper/save"),
    hasStt,
    hasTts,
    hasVoiceChat,
    hasVoiceConversationTransport,
    hasAudio: hasStt || hasTts || hasVoiceChat,
    hasEmbeddings:
      has("/api/v1/embeddings/models") ||
      has("/api/v1/embeddings/providers-config") ||
      has("/api/v1/embeddings/health"),
    hasMetrics: has("/api/v1/metrics/health") || has("/api/v1/metrics"),
    hasMcp: has("/api/v1/mcp/health"),
    hasReading: has("/api/v1/reading/save") && has("/api/v1/reading/items"),
    hasWriting:
      has("/api/v1/writing/sessions") ||
      has("/api/v1/writing/version") ||
      has("/api/v1/writing/capabilities"),
    hasWebSearch: has("/api/v1/research/websearch"),
    hasSkills: has("/api/v1/skills/") || has("/api/v1/skills/context"),
    hasPersona:
      has("/api/v1/persona/catalog") ||
      has("/api/v1/persona/session") ||
      has("/api/v1/persona/stream") ||
      hasPersonaLiveControl,
    hasPersonaLiveControl,
    hasPersonalization:
      has("/api/v1/personalization/profile") ||
      has("/api/v1/personalization/opt-in") ||
      has("/api/v1/personalization/memories"),
    hasFeedbackExplicit: has("/api/v1/feedback/explicit"),
    hasFeedbackImplicit: has("/api/v1/rag/feedback/implicit"),
    hasGuardian:
      has("/api/v1/guardian/relationships") ||
      has("/api/v1/guardian/policies") ||
      has("/api/v1/guardian/audit/{relationship_id}"),
    hasSelfMonitoring:
      has("/api/v1/self-monitoring/rules") ||
      has("/api/v1/self-monitoring/alerts") ||
      has("/api/v1/self-monitoring/crisis-resources"),
    ffmpegAvailable: null,
    specVersion: spec?.info?.version ?? null,
    specSource
  }
}

const inMemoryCapabilitiesCache = new Map<string, CapabilitiesCachePayload>()
const inFlightByCacheKey = new Map<string, Promise<ServerCapabilities>>()
let capabilitiesStorage: ReturnType<typeof createSafeStorage> | null = null

const isDevRuntime = (): boolean => {
  try {
    const env: any = (import.meta as any)?.env || {}
    return Boolean(env?.DEV) || env?.MODE === "development"
  } catch {
    return false
  }
}

const toErrorString = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message || String(error)
  }
  return typeof error === "string" ? error : "unknown-error"
}

export const getServerCapabilitiesCacheDiagnostics =
  (): ServerCapabilitiesCacheDiagnostics => ({
    ...capabilitiesDiagnostics,
    inMemoryCacheEntries: inMemoryCapabilitiesCache.size,
    inFlightRequests: inFlightByCacheKey.size
  })

const publishCapabilitiesDiagnostics = (): void => {
  if (typeof globalThis === "undefined") return
  const root = globalThis as typeof globalThis & {
    __tldwDiagnostics?: Record<string, unknown>
  }
  if (!root.__tldwDiagnostics) {
    root.__tldwDiagnostics = {}
  }
  root.__tldwDiagnostics.getServerCapabilitiesCacheDiagnostics =
    getServerCapabilitiesCacheDiagnostics
  root.__tldwDiagnostics.serverCapabilitiesCache =
    getServerCapabilitiesCacheDiagnostics()
}

const maybeLogDiagnostics = (reason: string): void => {
  publishCapabilitiesDiagnostics()
  if (!isDevRuntime()) return

  const now = Date.now()
  const shouldLog =
    capabilitiesDiagnostics.calls <= 5 ||
    now - lastDiagnosticsLogAt >= DIAGNOSTICS_LOG_INTERVAL_MS
  if (!shouldLog) return

  lastDiagnosticsLogAt = now
  // Keep this lightweight and periodic to avoid noisy logs.
  console.debug(
    "[tldw:capabilities-cache]",
    reason,
    getServerCapabilitiesCacheDiagnostics()
  )
}

const getCapabilitiesStorage = () => {
  if (capabilitiesStorage) return capabilitiesStorage
  capabilitiesStorage = createSafeStorage({ area: "local" })
  return capabilitiesStorage
}

const isFreshCache = (fetchedAt: number, now: number): boolean =>
  Number.isFinite(fetchedAt) && now - fetchedAt < CAPABILITIES_CACHE_TTL_MS

const isCapabilitiesCachePayload = (
  raw: unknown
): raw is CapabilitiesCachePayload => {
  if (!raw || typeof raw !== "object") return false
  const payload = raw as Partial<CapabilitiesCachePayload>
  return (
    typeof payload.key === "string" &&
    typeof payload.fetchedAt === "number" &&
    !!payload.capabilities &&
    typeof payload.capabilities === "object"
  )
}

const hasUsableApiKey = (value: unknown): boolean => {
  const key = String(value || "").trim()
  return Boolean(key && !isPlaceholderApiKey(key))
}

const shouldProbeIngestionSourceCapabilities = (
  config: TldwConfig | null
): boolean => {
  if (isHostedTldwDeployment()) return true
  if (isActiveCookieSessionConfig(config)) return true
  if (hasUsableApiKey(getRuntimeSingleUserApiKeyOverride())) {
    return true
  }
  if (config?.authMode === "multi-user") {
    return Boolean(String(config.accessToken || "").trim())
  }
  return hasUsableApiKey(config?.apiKey)
}

type CapabilitiesFetchContext = {
  cacheKey: string
  shouldProbeIngestionSourceCapabilities: boolean
}

const getCapabilitiesFetchContext = async (): Promise<CapabilitiesFetchContext> => {
  const config = await tldwClient.getConfig().catch(() => null)
  const runtimeApiKey = isHostedTldwDeployment()
    ? null
    : getRuntimeSingleUserApiKeyOverride()
  const cacheConfig = runtimeApiKey
    ? {
        ...(config || {}),
        authMode: "single-user" as const,
        apiKey: runtimeApiKey
      }
    : config

  return {
    cacheKey: buildChatSurfaceScopeKeyFromConfig(cacheConfig),
    shouldProbeIngestionSourceCapabilities:
      shouldProbeIngestionSourceCapabilities(config)
  }
}

const readPersistedCapabilities = async (
  cacheKey: string,
  now: number
): Promise<CapabilitiesCachePayload | null> => {
  try {
    const storage = getCapabilitiesStorage()
    const raw = await storage.get<unknown>(CAPABILITIES_STORAGE_KEY)
    if (!isCapabilitiesCachePayload(raw)) return null
    if (raw.key !== cacheKey) return null
    if (!isFreshCache(raw.fetchedAt, now)) {
      capabilitiesDiagnostics.stalePersistedMisses += 1
      return null
    }
    return {
      ...raw,
      capabilities: {
        ...defaultCapabilities,
        ...raw.capabilities
      }
    }
  } catch {
    return null
  }
}

const persistCapabilities = async (
  payload: CapabilitiesCachePayload
): Promise<void> => {
  try {
    const storage = getCapabilitiesStorage()
    await storage.set(CAPABILITIES_STORAGE_KEY, payload)
  } catch {
    // best-effort cache write only
  }
}

const fetchCapabilitiesFromServer = async (
  shouldProbeSourceCapabilities: boolean
): Promise<ServerCapabilities> => {
  const startedAt = Date.now()
  capabilitiesDiagnostics.networkFetches += 1
  let spec: any | null = null
  let docsInfo: DocsInfoResponse | null = null
  try {
    const [openApiSpec, docsInfoResponse] = await Promise.all([
      tldwClient.getOpenAPISpec(),
      bgRequest<DocsInfoResponse, any>({
        path: "/api/v1/config/docs-info" as any,
        method: "GET" as any,
        noAuth: true
      }).catch(() => null)
    ])
    spec = openApiSpec
    docsInfo = docsInfoResponse
    capabilitiesDiagnostics.lastError = null
  } catch (error) {
    capabilitiesDiagnostics.networkErrors += 1
    capabilitiesDiagnostics.lastError = toErrorString(error)
    // ignore, fall back to bundled spec
  }
  let diagnosticsSource: ServerCapabilitiesCacheDiagnostics["lastSource"] = "network"
  let specSource: "authoritative" | "fallback" = "authoritative"
  if (!spec) {
    spec = fallbackSpec
    diagnosticsSource = "fallback"
    specSource = "fallback"
    capabilitiesDiagnostics.fallbackSpecUses += 1
  }
  capabilitiesDiagnostics.lastFetchAt = Date.now()
  capabilitiesDiagnostics.lastFetchDurationMs =
    capabilitiesDiagnostics.lastFetchAt - startedAt
  capabilitiesDiagnostics.lastSource = diagnosticsSource

  maybeLogDiagnostics(
    diagnosticsSource === "fallback" ? "fallback-spec" : "network-fetch"
  )
  let capabilities = applyDocsInfoFeatureGates(
    computeCapabilities(spec, specSource),
    docsInfo
  )

  if (capabilities.hasIngestionSources && shouldProbeSourceCapabilities) {
    try {
      const sourceCapabilities =
        await bgRequest<IngestionSourceCapabilitiesResponse, any>({
          path: "/api/v1/ingestion-sources/capabilities" as any,
          method: "GET" as any
        })
      capabilities = applyIngestionSourceCapabilityGates(
        capabilities,
        sourceCapabilities
      )
    } catch {
      // Source entitlements are user-scoped. Failure should not hide generic source support.
    }
  }

  return capabilities
}

export const getChatTurnIdentitySupport = async (
  requestScope: ServicePromptRequestScope,
  signal?: AbortSignal
): Promise<boolean> => {
  // Retry safety must use the same pinned target as the completion, not UI cache state.
  const spec = await bgRequest({
    path: "/openapi.json",
    method: "GET",
    abortSignal: signal,
    ...requestScopeFields(requestScope)
  })
  return computeCapabilities(spec, "authoritative").hasChatTurnIdentity === true
}

const selectedDurableMarker = {
  version: 1, history: "h1_single_input_v1", result: "rag_source_v1", request_digest: "history_context_wire_v1",
  recovery_read: "protected_live_v1", inference_guarantee: "multiple_results_possible"
}
const sourceBounds = {
  sources: 20, excerpt_scalars: 1000, excerpt_utf8_bytes: 4000, aggregate_excerpt_scalars: 16000,
  name_utf8_bytes: 1000, metadata_text_utf8_bytes: 1000, chunk_id_utf8_bytes: 512, compact_label_utf8_bytes: 128,
  url_utf8_bytes: 2048, result_canonical_utf8_bytes: 65536, distinct_excerpts: true,
  media_id_utf8_bytes: 512, paired_character_ranges: true, chunk_index_within_total: true
}
type SchemaObject = Record<string, unknown>
const schemaObject = (value: unknown): SchemaObject =>
  value && typeof value === "object" && !Array.isArray(value) ? value as SchemaObject : {}
const exactFields = (value: unknown, expected: SchemaObject): boolean => {
  const object = schemaObject(value)
  return Object.keys(object).length === Object.keys(expected).length &&
    Object.entries(expected).every(([key, item]) => object[key] === item)
}

/** A marker is insufficient: follow operation references to the strict bounded wire DTOs. */
const selectedDurableContract = (spec: unknown): boolean => {
  const resolve = (value: unknown) => schemaObject(resolveSchemaRef(value, spec))
  const variants = (value: unknown, allowNull = false, depth = 0): SchemaObject[] => {
    if (depth > 8) return []
    const node = resolve(value)
    const union = node.oneOf ?? node.anyOf
    if (!Array.isArray(union)) return [node]
    return union.flatMap(item => variants(item, allowNull, depth + 1)).filter(item => !allowNull || item.type !== "null")
  }
  const single = (value: unknown) => {
    const nodes = variants(value, true)
    return nodes.length === 1 ? nodes[0] : {}
  }
  const props = (value: unknown) => schemaObject(resolve(value).properties)
  const strict = (value: unknown, keys: string[], required = keys): SchemaObject | null => {
    const node = resolve(value)
    const properties = props(node)
    const mandatory = node.required ?? []
    if (node.type !== "object" || node.additionalProperties !== false ||
        Object.keys(properties).length !== keys.length || !keys.every(key => key in properties) ||
        !Array.isArray(mandatory) || mandatory.length !== required.length || !required.every(key => mandatory.includes(key))) return null
    return properties
  }
  const literal = (value: unknown, constant: string | number) => {
    const node = resolve(value)
    return node.type === (typeof constant === "number" ? "integer" : "string") &&
      (node.const === constant || (Array.isArray(node.enum) && node.enum.length === 1 && node.enum[0] === constant))
  }
  const string = (value: unknown) => resolve(value).type === "string"
  const hex = (value: unknown) => {
    const node = resolve(value)
    return string(node) && node.pattern === "^[0-9a-f]{64}$" && node.minLength === 64 && node.maxLength === 64
  }
  const safeInt = (value: unknown) => {
    const node = resolve(value)
    return node.type === "integer" && node.minimum === 0 && node.maximum === Number.MAX_SAFE_INTEGER
  }
  const nonnegative = (value: unknown) => resolve(value).type === "integer" && resolve(value).minimum === 0
  const referenceKeys = ["version", "owner_key", "conversation_id", "input_message_id", "input_message_revision", "selection_digest"]
  const reference = (value: unknown, full = false) => {
    const fields = strict(value, full ? [...referenceKeys, "messages", "originating_selection_revision"] : referenceKeys)
    return !!fields && literal(fields.version, 1) && referenceKeys.slice(1).every(key => string(fields[key])) &&
      (!full || (manifest(fields.messages) && nonnegative(fields.originating_selection_revision)))
  }
  const manifest = (value: unknown) => {
    const array = resolve(value)
    const fields = strict(array.items, ["id", "revision"])
    return array.type === "array" && !!fields && string(fields.id) && string(fields.revision)
  }
  const selection = (value: unknown) => {
    const fields = strict(value, ["version", "owner_key", "conversation_id", "interpretation", "cursor", "selection_revision",
      "purpose", "messages", "fences", "storage_context_digest", "request_context_digest", "selection_digest"])
    if (!fields || !literal(fields.version, 1) || !["owner_key", "conversation_id", "storage_context_digest", "request_context_digest", "selection_digest"].every(key => string(fields[key])) ||
        !manifest(fields.messages) || !nonnegative(fields.selection_revision) || !Array.isArray(resolve(fields.purpose).enum) ||
        !(resolve(fields.purpose).enum as unknown[]).includes("send")) return false
    const fences = strict(fields.fences, ["conversation", "history", "settings"])
    if (!fences || !Object.values(fences).every(string)) return false
    const interpretations = variants(fields.interpretation)
    const cursors = variants(fields.cursor)
    return interpretations.length === 2 && ["parent_graph_v1", "legacy_linear_v1"].every(kind => interpretations.some(node => literal(props(node).kind, kind))) && interpretations.every(node => {
      const kind = props(node).kind
      return literal(kind, "parent_graph_v1") ? !!strict(node, ["kind"]) :
        literal(kind, "legacy_linear_v1") && !!strict(node, ["kind", "projection_id"]) && string(props(node).projection_id)
    }) && cursors.length === 3 && ["empty", "before_message", "after_message"].every(kind => cursors.some(node => literal(props(node).kind, kind))) && cursors.every(node => literal(props(node).kind, "empty") ? !!strict(node, ["kind"]) :
      (literal(props(node).kind, "before_message") || literal(props(node).kind, "after_message")) &&
      !!strict(node, ["kind", "message_id"]) && string(props(node).message_id))
  }
  const source = (value: unknown) => {
    const fields = strict(value, ["name", "type", "mode", "url", "pageContent", "metadata"])
    if (!fields || !["name", "type", "url", "pageContent"].every(key => string(fields[key])) ||
        !literal(fields.mode, "rag") || resolve(fields.pageContent).maxLength !== 1000) return false
    const metadata = strict(fields.metadata, ["source", "title", "chunk_id", "retrieval_strategy", "source_type", "selection_reason", "score", "page", "loc",
      "media_id", "author", "chunk_index", "total_chunks", "start_char", "end_char", "chunk_start", "chunk_end"], [])
    if (!metadata || !["source", "title", "chunk_id", "retrieval_strategy", "source_type", "selection_reason", "media_id", "author"].every(key => string(metadata[key])) ||
        !["page", "chunk_index", "start_char", "end_char", "chunk_start", "chunk_end"].every(key => safeInt(metadata[key])) ||
        resolve(metadata.total_chunks).type !== "integer" || resolve(metadata.total_chunks).minimum !== 1 ||
        resolve(metadata.total_chunks).maximum !== Number.MAX_SAFE_INTEGER) return false
    const score = variants(metadata.score)
    if (!score.length || score.some(node => node.type !== "integer" && node.type !== "number")) return false
    const loc = strict(metadata.loc, ["lines"])
    const lines = loc && strict(loc.lines, ["from", "to"])
    return !!lines && safeInt(lines.from) && safeInt(lines.to)
  }
  const payload = (value: unknown, receipt = false) => {
    const fields = strict(value, receipt ? ["version", "sources", "result_message_id", "result_message_revision", "admission", "request_context_digest"] : ["version", "sources"])
    if (!fields || !literal(fields.version, 1)) return false
    const sources = resolve(fields.sources)
    if (sources.type !== "array" || sources.maxItems !== 20 || (sources.minItems !== undefined && sources.minItems !== 0) || !source(sources.items)) return false
    return receipt ? string(fields.result_message_id) && resolve(fields.result_message_id).format === "uuid" &&
      literal(fields.result_message_revision, "1") && reference(fields.admission) && hex(fields.request_context_digest)
      : exactFields(resolve(value)["x-tldw-source-bounds"], sourceBounds)
  }
  const scope = (value: unknown) => {
    const nodes = variants(value)
    return nodes.length === 2 && ["global", "workspace"].every(kind => nodes.some(node => literal(props(node).scope_type, kind))) && nodes.every(node => {
      const fields = strict(node, ["scope_type", "workspace_id"])
      return !!fields && (literal(fields.scope_type, "global") ? resolve(fields.workspace_id).type === "null" :
        literal(fields.scope_type, "workspace") && string(fields.workspace_id) && resolve(fields.workspace_id).minLength === 1)
    })
  }
  const recovery = (value: unknown) => {
    const nodes = variants(value, true)
    const statuses = nodes.map(node => resolve(props(node).status).const)
    if (nodes.length !== 3 || new Set(statuses).size !== 3) return false
    return nodes.every(node => {
      const fields = props(node)
      if (literal(fields.status, "input_verified")) return !!strict(node, ["version", "status", "scope", "admission"]) && literal(fields.version, 1) && scope(fields.scope) && reference(fields.admission, true)
      if (literal(fields.status, "result_verified")) return !!strict(node, ["version", "status", "scope", "result"]) && literal(fields.version, 1) && scope(fields.scope) && payload(fields.result, true)
      const codes = resolve(fields.code).enum
      return literal(fields.status, "unverified") && !!strict(node, ["version", "status", "code"]) && literal(fields.version, 1) &&
        Array.isArray(codes) && codes.length === 3 && ["no_protected_binding", "live_state_mismatch", "unsupported_projection"].every(code => codes.includes(code))
    })
  }
  const requiredFields = (value: unknown, keys: string[]) => {
    const node = resolve(value)
    const required = node.required
    return node.type === "object" && Array.isArray(required) && keys.every(key => required.includes(key)) ? props(node) : null
  }
  const messageList = (value: unknown) => {
    const fields = requiredFields(value, ["messages", "total", "limit", "offset", "pagination"])
    if (!fields || !["total", "limit", "offset"].every(key => resolve(fields[key]).type === "integer")) return false
    const messages = resolve(fields.messages)
    if (messages.type !== "array" || !recovery(props(resolve(messages.items)).tldw_history_recovery_v1) ||
        single(fields.has_more).type !== "boolean" || !nonnegative(single(fields.next_offset))) return false
    const pagination = requiredFields(fields.pagination, ["limit", "offset", "has_more"])
    return !!pagination && literal(pagination.mode, "offset") && resolve(pagination.limit).type === "integer" &&
      resolve(pagination.limit).minimum === 1 && nonnegative(pagination.offset) &&
      nonnegative(single(pagination.total)) && resolve(pagination.has_more).type === "boolean" &&
      nonnegative(single(pagination.next_offset))
  }
  const paths = schemaObject(schemaObject(spec).paths)
  const post = schemaObject(schemaObject(paths["/api/v1/chat/completions"]).post)
  if (!exactFields(post["x-tldw-selected-durable-turn"], selectedDurableMarker)) return false
  const jsonSchema = (value: unknown) => schemaObject(schemaObject(schemaObject(value).content)["application/json"]).schema
  const turn = single(props(single(jsonSchema(post.requestBody))).tldw_turn)
  const turnFields = strict(turn, ["user_message_id", "history_v1", "result_v1"], ["user_message_id"])
  if (!turnFields || !string(turnFields.user_message_id) || resolve(turnFields.user_message_id).format !== "uuid" || !payload(single(turnFields.result_v1))) return false
  const histories = variants(turnFields.history_v1, true).flatMap(node => variants(node))
  if (histories.length !== 2 || !histories.some(node => literal(props(node).kind, "selection")) || !histories.some(node => literal(props(node).kind, "admission")) ||
      !histories.every(node => {
        const fields = props(node)
        return literal(fields.kind, "selection") ? !!strict(node, ["version", "kind", "selection"]) && literal(fields.version, 1) && selection(fields.selection) :
          literal(fields.kind, "admission") && !!strict(node, ["version", "kind", "admission", "request_context_digest"]) && literal(fields.version, 1) && reference(fields.admission) && hex(fields.request_context_digest)
      })) return false
  const readMarker = { version: 1, projection: "protected_live_v1" }
  for (const [path, list] of [["/api/v1/messages/{message_id}", false], ["/api/v1/chats/{chat_id}/messages", true]] as const) {
    const route = schemaObject(paths[path])
    const operation = schemaObject(route.get)
    if (!exactFields(operation["x-tldw-history-recovery-read"], readMarker)) return false
    const parameters = [...(Array.isArray(route.parameters) ? route.parameters : []), ...(Array.isArray(operation.parameters) ? operation.parameters : [])].map(resolve)
    const query = (name: string) => {
      const matches = parameters.filter(param => param.in === "query" && param.name === name)
      return matches.length === 1 ? resolve(matches[0].schema) : {}
    }
    const optIn = query("include_history_recovery_v1")
    const scopeQuery = single(query("scope_type"))
    const scopes = scopeQuery.enum
    if (optIn.type !== "boolean" || optIn.default !== false || scopeQuery.type !== "string" || !Array.isArray(scopes) || scopes.length !== 2 || !scopes.includes("global") || !scopes.includes("workspace") ||
        !string(single(query("workspace_id"))) ||
        (list && !["format_for_completions", "include_character_context", "include_deleted", "render_placeholders"].every(name => query(name).type === "boolean"))) return false
    const response = schemaObject(schemaObject(operation.responses)["200"])
    if (!list) {
      if (!recovery(props(single(jsonSchema(response))).tldw_history_recovery_v1)) return false
    } else {
      const limit = query("limit")
      const offset = query("offset")
      if (limit.type !== "integer" || limit.default !== 50 || limit.minimum !== 1 || limit.maximum !== 200 ||
          offset.type !== "integer" || offset.default !== 0 || offset.minimum !== 0 || offset.maximum !== undefined ||
          !messageList(jsonSchema(response))) return false
    }
  }
  return true
}

export const getSelectedDurableTurnSupport = async (
  requestScope: ServicePromptRequestScope, signal?: AbortSignal
): Promise<boolean> => {
  signal?.throwIfAborted()
  const spec = await bgRequest({ path: "/openapi.json", method: "GET", abortSignal: signal, ...requestScopeFields(requestScope) })
  signal?.throwIfAborted()
  return selectedDurableContract(spec)
}

export const getServerCapabilities = async (
  options?: { forceRefresh?: boolean }
): Promise<ServerCapabilities> => {
  const { cacheKey, shouldProbeIngestionSourceCapabilities } =
    await getCapabilitiesFetchContext()
  const now = Date.now()
  const forceRefresh = options?.forceRefresh === true
  capabilitiesDiagnostics.calls += 1
  capabilitiesDiagnostics.lastCacheKey = cacheKey

  if (forceRefresh) {
    capabilitiesDiagnostics.forceRefreshCalls += 1
  }

  if (!forceRefresh) {
    const inMemory = inMemoryCapabilitiesCache.get(cacheKey)
    if (inMemory) {
      if (!isFreshCache(inMemory.fetchedAt, now)) {
        capabilitiesDiagnostics.staleMemoryMisses += 1
      } else {
        capabilitiesDiagnostics.inMemoryHits += 1
        capabilitiesDiagnostics.lastSource = "in-memory"
        maybeLogDiagnostics("in-memory-hit")
        return inMemory.capabilities
      }
    }

    const persisted = await readPersistedCapabilities(cacheKey, now)
    if (persisted) {
      capabilitiesDiagnostics.persistedHits += 1
      capabilitiesDiagnostics.lastSource = "persisted"
      inMemoryCapabilitiesCache.set(cacheKey, persisted)
      maybeLogDiagnostics("persisted-hit")
      return persisted.capabilities
    }
  }

  const existing = inFlightByCacheKey.get(cacheKey)
  if (existing) {
    capabilitiesDiagnostics.inFlightHits += 1
    capabilitiesDiagnostics.lastSource = "in-flight"
    maybeLogDiagnostics("in-flight-hit")
    return existing
  }

  const request = (async () => {
    const capabilities = await fetchCapabilitiesFromServer(
      shouldProbeIngestionSourceCapabilities
    )
    const payload: CapabilitiesCachePayload = {
      key: cacheKey,
      fetchedAt: Date.now(),
      capabilities
    }
    inMemoryCapabilitiesCache.set(cacheKey, payload)
    void persistCapabilities(payload)
    return capabilities
  })()

  inFlightByCacheKey.set(cacheKey, request)
  try {
    return await request
  } finally {
    inFlightByCacheKey.delete(cacheKey)
    publishCapabilitiesDiagnostics()
  }
}
