import { tldwClient, TldwModel, type TldwConfig } from "./TldwApiClient"
import { createSafeStorage } from "@/utils/safe-storage"
import { isPlaceholderApiKey } from "@/utils/api-key"
import { getRuntimeSingleUserApiKeyOverride } from "@/services/tldw/runtime-auth-override"
import {
  getProviderDisplayName,
  inferProviderFromModel
} from "@/utils/provider-registry"

const IMAGE_MODEL_HINTS = [
  "image",
  "dall-e",
  "dalle",
  "flux",
  "stable-diffusion",
  "sdxl",
  "midjourney",
  "recraft",
  "pixart",
  "playground",
  "kolors",
  "imagen"
]

const VIDEO_MODEL_HINTS = ["video", "veo", "sora", "kling", "hunyuan-video"]

const AUDIO_MODEL_HINTS = [
  "whisper",
  "transcribe",
  "asr",
  "tts",
  "speech",
  "audio",
  "voice"
]

const NON_CHAT_MODEL_HINTS = ["rerank", "moderation", "safety"]

const UNSELECTABLE_CHAT_MODEL_AVAILABILITY = new Set([
  "disabled",
  "failed",
  "unavailable",
  "not-configured"
])

const hasAnyHint = (value: string, hints: string[]): boolean =>
  hints.some((hint) => value.includes(hint))

const normalizeAvailabilityStatus = (value: string | undefined): string | null => {
  if (typeof value !== "string") return null
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[\s_-]+/g, "-")
  return normalized.length > 0 ? normalized : null
}

const isAbortLikeModelFetchError = (error: unknown): boolean => {
  const candidate = error as
    | (Error & {
        code?: unknown
        status?: unknown
      })
    | null
    | undefined
  const message =
    candidate instanceof Error ? candidate.message.toLowerCase() : ""
  return (
    candidate?.name === "AbortError" ||
    candidate?.code === "REQUEST_ABORTED" ||
    message.includes("abort")
  )
}

const hasUsableApiKey = (value: unknown): boolean => {
  const key = String(value || "").trim()
  return Boolean(key && !isPlaceholderApiKey(key))
}

export interface ModelInfo {
  id: string
  name: string
  provider: string
  type: 'chat' | 'embedding' | 'image' | 'other'
  capabilities?: string[]
  contextLength?: number
  description?: string
  isConfigured?: boolean
  providerIsConfigured?: boolean
  providerEnabled?: boolean
  availability?: string
  readinessReasonCode?: string
  readinessMessage?: string
  chatProvider?: string
  catalogOnly?: boolean
  modalities?: {
    input?: string[]
    output?: string[]
  }
}

type ModelCacheRecord = {
  version: number
  models: ModelInfo[] | null
  timestamp: number
  scope: string | null
  invalidationToken?: string
}

type InvalidationListener = (token: string) => void

export class TldwModelsService {
  private cachedModels: ModelInfo[] | null = null
  private lastFetchTime: number = 0
  private lastForcedFetchTime: number = 0
  private readonly CACHE_DURATION = 15 * 60 * 1000 // 15 minutes
  private readonly FORCE_REFRESH_COOLDOWN = 30 * 1000
  private readonly CACHE_KEY = "tldwModelsCache"
  private readonly CACHE_SCHEMA_VERSION = 4
  private readonly INVALIDATION_TOKEN_HISTORY_LIMIT = 64
  private storage = createSafeStorage({ area: "local" })
  private storageLoaded = false
  private storageInitPromise: Promise<void> | null = null
  private storageWritePromise: Promise<void> = Promise.resolve()
  private inFlightFetch: Promise<ModelInfo[]> | null = null
  private cacheScopeKey: string | null = null
  private invalidationGeneration = 0
  private lastAppliedInvalidationToken: string | null = null
  private seenInvalidationTokens = new Set<string>()
  private pendingLocalInvalidationTokens = new Set<string>()
  private invalidationListeners = new Set<InvalidationListener>()

  constructor() {
    this.getModels = this.getModels.bind(this)
    this.getChatModels = this.getChatModels.bind(this)
    this.getCachedChatModels = this.getCachedChatModels.bind(this)
    this.getEmbeddingModels = this.getEmbeddingModels.bind(this)
    this.getImageModels = this.getImageModels.bind(this)
    this.storage.watch?.({
      [this.CACHE_KEY]: (change) => {
        this.applyInvalidationRecord(change.newValue)
      }
    })
  }

  private async ensureStorageLoaded() {
    if (this.storageLoaded) return
    if (!this.storageInitPromise) {
      const loadGeneration = this.invalidationGeneration
      this.storageInitPromise = (async () => {
        try {
          const cached =
            (await this.storage.get<ModelCacheRecord>(this.CACHE_KEY)) || null
          const cacheVersion =
            typeof cached?.version === "number" ? cached.version : 0
          if (this.applyInvalidationRecord(cached)) {
            return
          }
          if (
            cacheVersion === this.CACHE_SCHEMA_VERSION &&
            cached?.models &&
            Array.isArray(cached.models) &&
            loadGeneration === this.invalidationGeneration
          ) {
            this.cachedModels = cached.models as ModelInfo[]
            this.lastFetchTime = Number(cached.timestamp || 0)
            this.cacheScopeKey =
              typeof cached.scope === "string" ? cached.scope : null
          }
        } catch {
          // ignore storage read failures
        } finally {
          this.storageLoaded = true
        }
      })()
    }
    await this.storageInitPromise
  }

  private async persistCache(expectedGeneration = this.invalidationGeneration) {
    if (expectedGeneration !== this.invalidationGeneration) return
    const value = {
      version: this.CACHE_SCHEMA_VERSION,
      models: this.cachedModels,
      timestamp: this.lastFetchTime,
      scope: this.cacheScopeKey
    }
    const write = this.storageWritePromise.then(async () => {
      if (expectedGeneration !== this.invalidationGeneration) return
      try {
        await this.storage.set(this.CACHE_KEY, value)
      } catch {
        // Best-effort persistence; ignore errors
      }
    })
    this.storageWritePromise = write
    await write
  }

  private async persistInvalidationTombstone(
    token: string,
    expectedGeneration: number
  ): Promise<void> {
    const value: ModelCacheRecord = {
      version: this.CACHE_SCHEMA_VERSION,
      models: null,
      timestamp: 0,
      scope: null,
      invalidationToken: token
    }
    const write = this.storageWritePromise.then(async () => {
      if (
        expectedGeneration !== this.invalidationGeneration ||
        token !== this.lastAppliedInvalidationToken
      ) {
        this.pendingLocalInvalidationTokens.delete(token)
        return
      }
      try {
        await this.storage.set(this.CACHE_KEY, value)
      } catch {
        this.pendingLocalInvalidationTokens.delete(token)
        // Best-effort persistence; ignore errors
      }
    })
    this.storageWritePromise = write
    await write
  }

  private invalidateCacheState() {
    this.invalidationGeneration += 1
    this.cachedModels = null
    this.lastFetchTime = 0
    this.lastForcedFetchTime = 0
    this.inFlightFetch = null
    this.cacheScopeKey = null
  }

  private applyInvalidationRecord(value: unknown): boolean {
    const record = value as Partial<ModelCacheRecord> | null
    if (
      record?.version !== this.CACHE_SCHEMA_VERSION ||
      record.models !== null ||
      typeof record.invalidationToken !== "string"
    ) {
      return false
    }
    const token = record.invalidationToken.trim()
    const applied = this.applyInvalidationToken(token)
    this.pendingLocalInvalidationTokens.delete(token)
    return applied
  }

  private applyInvalidationToken(token: string): boolean {
    const normalizedToken = token.trim()
    if (
      !normalizedToken ||
      normalizedToken === this.lastAppliedInvalidationToken ||
      this.seenInvalidationTokens.has(normalizedToken) ||
      this.pendingLocalInvalidationTokens.has(normalizedToken)
    ) {
      return false
    }
    this.rememberInvalidationToken(
      this.seenInvalidationTokens,
      normalizedToken
    )
    this.lastAppliedInvalidationToken = normalizedToken
    if (!this.storageLoaded && !this.storageInitPromise) {
      this.storageLoaded = true
    }
    this.invalidateCacheState()
    this.invalidationListeners.forEach((listener) => {
      try {
        listener(normalizedToken)
      } catch (error) {
        console.error("Model cache invalidation listener failed:", error)
      }
    })
    return true
  }

  private rememberInvalidationToken(tokens: Set<string>, token: string): void {
    if (tokens.has(token)) return
    tokens.add(token)
    if (tokens.size <= this.INVALIDATION_TOKEN_HISTORY_LIMIT) return
    const oldestToken = tokens.values().next().value
    if (oldestToken) tokens.delete(oldestToken)
  }

  private createInvalidationToken(): string {
    if (typeof globalThis.crypto?.randomUUID === "function") {
      return globalThis.crypto.randomUUID()
    }
    return `${Date.now()}-${Math.random().toString(36).slice(2)}`
  }

  private reconcileCacheScope(
    scopeKey: string,
    requestGeneration: number
  ): number {
    if (requestGeneration !== this.invalidationGeneration) {
      return requestGeneration
    }
    if (this.cacheScopeKey && this.cacheScopeKey !== scopeKey) {
      this.invalidateCacheState()
      requestGeneration = this.invalidationGeneration
    }
    this.cacheScopeKey = scopeKey
    return requestGeneration
  }

  private isConfiguredForModels(config: TldwConfig | null): boolean {
    if (!config) return false
    const serverUrl = String(config.serverUrl || "").trim()
    if (!serverUrl) return false

    if (config.authMode === "multi-user") {
      return Boolean(String(config.accessToken || "").trim())
    }

    return hasUsableApiKey(getRuntimeSingleUserApiKeyOverride()) || hasUsableApiKey(config.apiKey)
  }

  private buildCacheScope(config: TldwConfig | null): string {
    if (!config) return "none"
    const serverUrl = String(config.serverUrl || "").trim().toLowerCase()
    const authMode = String(config.authMode || "single-user")
    const hasAccessToken = Boolean(String(config.accessToken || "").trim())
    const hasApiKey =
      hasUsableApiKey(config.apiKey) ||
      hasUsableApiKey(getRuntimeSingleUserApiKeyOverride())
    const orgId = config.orgId != null ? String(config.orgId) : "none"
    return `${serverUrl}|${authMode}|${hasAccessToken ? "token" : hasApiKey ? "key" : "none"}|${orgId}`
  }

  /**
   * Get available models from tldw server
   * Uses cache to avoid frequent API calls
   */
  async getModels(
    forceRefresh: boolean = false,
    options?: { refreshOpenRouter?: boolean }
  ): Promise<ModelInfo[]> {
    await this.ensureStorageLoaded()
    let fetchGeneration = this.invalidationGeneration
    const config = await tldwClient.getConfig().catch(() => null)
    const scopeKey = this.buildCacheScope(config)
    fetchGeneration = this.reconcileCacheScope(scopeKey, fetchGeneration)

    const now = Date.now()

    // Return cached models if available and not expired
    if (!forceRefresh && this.cachedModels && (now - this.lastFetchTime) < this.CACHE_DURATION) {
      return this.cachedModels
    }
    if (
      forceRefresh &&
      this.cachedModels &&
      (now - this.lastForcedFetchTime) < this.FORCE_REFRESH_COOLDOWN
    ) {
      return this.cachedModels
    }
    if (this.inFlightFetch) {
      return await this.inFlightFetch
    }

    if (!this.isConfiguredForModels(config)) {
      return this.cachedModels || []
    }

    const fetchFromServer = async () => {
      await tldwClient.initialize()
      const models = await tldwClient.getModels({
        refreshOpenRouter: options?.refreshOpenRouter === true
      })

      // Transform tldw models to our format
      const transformedModels = models.map(model => this.transformModel(model))
      if (fetchGeneration === this.invalidationGeneration) {
        this.cachedModels = transformedModels
        this.lastFetchTime = Date.now()
        if (forceRefresh) {
          this.lastForcedFetchTime = this.lastFetchTime
        }
        await this.persistCache(fetchGeneration)
      }

      return transformedModels
    }

    const fetchPromise = fetchFromServer().catch(async (error) => {
      if (isAbortLikeModelFetchError(error)) {
        return this.cachedModels || []
      }

      if (!import.meta.env?.DEV) {
        console.error('Failed to fetch models from tldw:', error)
      }

      // Return cached models if available, even if expired
      if (this.cachedModels) {
        return this.cachedModels
      }

      // Return empty array as fallback
      return []
    })

    if (fetchGeneration === this.invalidationGeneration) {
      this.inFlightFetch = fetchPromise
    }
    try {
      return await fetchPromise
    } finally {
      if (this.inFlightFetch === fetchPromise) {
        this.inFlightFetch = null
      }
    }
  }

  /**
   * Get chat models only
   */
  async getChatModels(
    forceRefresh: boolean = false,
    options?: { refreshOpenRouter?: boolean }
  ): Promise<ModelInfo[]> {
    const models = await this.getModels(forceRefresh, options)
    return models.filter((model) => this.isSelectableChatModel(model))
  }

  async getCachedChatModels(): Promise<ModelInfo[]> {
    await this.ensureStorageLoaded()
    const requestGeneration = this.invalidationGeneration
    const config = await tldwClient.getConfig().catch(() => null)
    const scopeKey = this.buildCacheScope(config)
    this.reconcileCacheScope(scopeKey, requestGeneration)
    return (this.cachedModels || []).filter((model) =>
      this.isSelectableChatModel(model)
    )
  }

  /**
   * Get embedding models only
   */
  async getEmbeddingModels(
    forceRefresh: boolean = false,
    options?: { refreshOpenRouter?: boolean }
  ): Promise<ModelInfo[]> {
    const models = await this.getModels(forceRefresh, options)
    return models.filter(m => m.type === 'embedding')
  }

  /**
   * Get image models only
   */
  async getImageModels(
    forceRefresh: boolean = false,
    options?: { refreshOpenRouter?: boolean }
  ): Promise<ModelInfo[]> {
    const models = await this.getModels(forceRefresh, options)
    return models.filter(m => m.type === 'image')
  }

  /**
   * Get a specific model by ID
   */
  async getModel(modelId: string): Promise<ModelInfo | null> {
    const models = await this.getModels()
    return models.find(m => m.id === modelId) || null
  }

  /**
   * Check if a model exists
   */
  async modelExists(modelId: string): Promise<boolean> {
    const model = await this.getModel(modelId)
    return model !== null
  }

  /**
   * Get models grouped by provider
   */
  async getModelsByProvider(): Promise<Map<string, ModelInfo[]>> {
    const models = await this.getModels()
    const grouped = new Map<string, ModelInfo[]>()

    for (const model of models) {
      const provider = model.provider
      if (!grouped.has(provider)) {
        grouped.set(provider, [])
      }
      grouped.get(provider)!.push(model)
    }

    return grouped
  }

  /**
   * Transform tldw model to our format
   */
  private transformModel(tldwModel: TldwModel): ModelInfo {
    const nameLower = tldwModel.name.toLowerCase()
    const declaredType = (tldwModel.type || "").trim().toLowerCase()
    const normalizeMods = (mods?: string[]) =>
      Array.isArray(mods)
        ? mods.map((v) => String(v).trim().toLowerCase()).filter(Boolean)
        : []
    const inputMods = normalizeMods(tldwModel.modalities?.input)
    const outputMods = normalizeMods(tldwModel.modalities?.output)

    const caps: string[] = []
    if (Array.isArray(tldwModel.capabilities)) {
      caps.push(...tldwModel.capabilities)
    } else if (tldwModel.capabilities && typeof tldwModel.capabilities === "object") {
      Object.entries(tldwModel.capabilities).forEach(([key, value]) => {
        if (value) caps.push(key)
      })
    }
    if (tldwModel.vision) caps.push('vision')
    if (tldwModel.function_calling) caps.push('tools')
    // Heuristic: flag some models as "fast" based on name
    if (
      nameLower.includes('mini') ||
      nameLower.includes('flash') ||
      nameLower.includes('small') ||
      nameLower.includes('haiku')
    ) {
      caps.push('fast')
    }
    const capsNormalized = caps.map((cap) => cap.toLowerCase())

    // Determine model type based on declared metadata, modalities, or heuristics.
    let type: 'chat' | 'embedding' | 'image' | 'other' = 'chat'
    if (declaredType === 'image') {
      type = 'image'
    } else if (declaredType === 'embedding') {
      type = 'embedding'
    } else if (declaredType === 'video' || declaredType === 'audio') {
      type = 'other'
    } else if (outputMods.includes('image')) {
      type = 'image'
    } else if (outputMods.includes('embedding')) {
      type = 'embedding'
    } else if (capsNormalized.includes('image') || capsNormalized.includes('image_generation')) {
      type = 'image'
    } else if (
      hasAnyHint(nameLower, IMAGE_MODEL_HINTS) ||
      hasAnyHint(tldwModel.id.toLowerCase(), IMAGE_MODEL_HINTS)
    ) {
      type = 'image'
    } else if (nameLower.includes('embed') || nameLower.includes('embedding')) {
      type = 'embedding'
    } else if (
      hasAnyHint(nameLower, VIDEO_MODEL_HINTS) ||
      hasAnyHint(tldwModel.id.toLowerCase(), VIDEO_MODEL_HINTS) ||
      hasAnyHint(nameLower, AUDIO_MODEL_HINTS) ||
      hasAnyHint(tldwModel.id.toLowerCase(), AUDIO_MODEL_HINTS) ||
      hasAnyHint(nameLower, NON_CHAT_MODEL_HINTS) ||
      hasAnyHint(tldwModel.id.toLowerCase(), NON_CHAT_MODEL_HINTS)
    ) {
      type = 'other'
    }

    // Extract provider from model ID or name if not provided
    const inferred =
      inferProviderFromModel(tldwModel.id, "llm") ||
      inferProviderFromModel(tldwModel.name, "llm")
    const provider = tldwModel.provider || inferred || "unknown"
    const toOptionalBoolean = (value: unknown): boolean | undefined =>
      typeof value === "boolean" ? value : undefined
    const toOptionalString = (value: unknown): string | undefined => {
      if (typeof value !== "string") return undefined
      const trimmed = value.trim()
      return trimmed.length > 0 ? trimmed : undefined
    }
    const rawModel = tldwModel as TldwModel & Record<string, unknown>

    return {
      id: tldwModel.id,
      name: tldwModel.name || tldwModel.id,
      provider: provider,
      type: type,
      capabilities: caps.length ? Array.from(new Set(caps)) : undefined,
      contextLength: tldwModel.context_length,
      description: tldwModel.description,
      modalities: tldwModel.modalities,
      isConfigured: toOptionalBoolean(tldwModel.is_configured),
      providerIsConfigured: toOptionalBoolean(
        rawModel.provider_is_configured ?? rawModel.providerIsConfigured
      ),
      providerEnabled: toOptionalBoolean(tldwModel.provider_enabled),
      availability:
        typeof tldwModel.availability === "string"
          ? tldwModel.availability
          : undefined,
      readinessReasonCode: toOptionalString(
        rawModel.readiness_reason_code ?? rawModel.readinessReasonCode
      ),
      readinessMessage: toOptionalString(
        rawModel.readiness_message ?? rawModel.readinessMessage
      ),
      chatProvider: toOptionalString(
        rawModel.chat_provider ?? rawModel.chatProvider
      )
    }
  }

  private isSelectableChatModel(model: ModelInfo): boolean {
    if (model.type !== "chat") return false
    if (model.isConfigured === false) return false
    if (model.providerIsConfigured === false) return false
    if (model.providerEnabled === false) return false
    const availability = normalizeAvailabilityStatus(model.availability)
    if (
      availability &&
      UNSELECTABLE_CHAT_MODEL_AVAILABILITY.has(availability)
    ) {
      return false
    }
    return true
  }

  /**
   * Clear the model cache
   */
  async clearCache(): Promise<void> {
    const token = this.createInvalidationToken()
    this.applyInvalidationToken(token)
    this.rememberInvalidationToken(
      this.pendingLocalInvalidationTokens,
      token
    )
    await this.persistInvalidationTombstone(
      token,
      this.invalidationGeneration
    )
  }

  subscribeInvalidation(listener: InvalidationListener): () => void {
    this.invalidationListeners.add(listener)
    return () => this.invalidationListeners.delete(listener)
  }

  /**
   * Get provider display name
   */
  getProviderDisplayName(provider: string): string {
    return getProviderDisplayName(provider)
  }

  /**
   * Warm the cache and return the latest models.
   */
  async warmCache(
    force: boolean = false,
    options?: { refreshOpenRouter?: boolean }
  ): Promise<ModelInfo[]> {
    return await this.getModels(force, options)
  }
}

// Singleton instance
export const tldwModels = new TldwModelsService()
