import { tldwClient } from "./TldwApiClient"
import { bgRequest } from "@/services/background-proxy"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { createSafeStorage } from "@/utils/safe-storage"

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
      has("/api/v1/persona/stream"),
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

const getCapabilitiesCacheKey = async (): Promise<string> => {
  try {
    const cfg = await tldwClient.getConfig()
    return buildChatSurfaceScopeKeyFromConfig(cfg)
  } catch {
    return buildChatSurfaceScopeKeyFromConfig(null)
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

const fetchCapabilitiesFromServer = async (): Promise<ServerCapabilities> => {
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

  if (capabilities.hasIngestionSources) {
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

export const getServerCapabilities = async (
  options?: { forceRefresh?: boolean }
): Promise<ServerCapabilities> => {
  const cacheKey = await getCapabilitiesCacheKey()
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
    const capabilities = await fetchCapabilitiesFromServer()
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
