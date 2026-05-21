import { bgRequest } from "@/services/background-proxy"
import { buildQuery } from "../client-utils"
import { appendPathQuery } from "../path-utils"
import type {
  LlamacppAsset,
  LlamacppAcquisitionJobListResponse,
  LlamacppAcquisitionJobResponse,
  LlamacppAssetDownloadRequest,
  LlamacppAssetImportPreviewResponse,
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
import type {
  TldwModel,
  ImageBackend,
  TldwEmbeddingModel,
  TldwEmbeddingModelsResponse,
  TldwEmbeddingProvidersConfig,
  MediaIngestionBudgetDiagnostics,
  MlxStatus,
  MlxLoadRequest,
  MlxUnloadRequest,
} from "../TldwApiClient"
import type {
  AudioPresetCreatePayload,
  AudioPresetKind,
  AudioPresetListResponse,
  AudioPresetUpdatePayload,
  AudioPresetValidationResponse,
  AudioPreset
} from "@/types/audio-presets"

/**
 * Minimal interface for the TldwApiClient methods referenced via `this`.
 */
export interface TldwApiClientCore {
  getModelsMetadata(options?: { refreshOpenRouter?: boolean }): Promise<any>
  createImageArtifact(request: any): Promise<any>
  ensureConfigForRequest(requireAuth: boolean): Promise<any>
  request<T>(init: any, requireAuth?: boolean): Promise<T>
  requestWithCurrentConfig<T>(init: any, requireAuth?: boolean): Promise<T>
  upload<T>(init: any, requireAuth?: boolean): Promise<T>
}

export const modelsAudioMethods = {
  // ── LLM Models & Providers ──

  async getModels(
    this: TldwApiClientCore,
    options?: {
      refreshOpenRouter?: boolean
    }
  ): Promise<TldwModel[]> {
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
    const toOptionalBoolean = (value: unknown): boolean | undefined =>
      typeof value === "boolean" ? value : undefined
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
        is_configured: toOptionalBoolean(m.is_configured),
        provider_is_configured: toOptionalBoolean(m.provider_is_configured),
        catalog_only: toOptionalBoolean(m.catalog_only),
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
  },

  async getProviders(): Promise<any> {
    return await bgRequest<any>({ path: '/api/v1/llm/providers', method: 'GET' })
  },

  async getModelsMetadata(options?: {
    refreshOpenRouter?: boolean
  }): Promise<any> {
    // tldw_server returns either an array or an object
    // of the form { models: [...], total: N }.
    const query = options?.refreshOpenRouter ? "?refresh_openrouter=true" : ""
    const path = appendPathQuery("/api/v1/llm/models/metadata", query)
    return await bgRequest<any>({ path, method: 'GET' })
  },

  async getImageBackends(
    this: TldwApiClientCore
  ): Promise<ImageBackend[]> {
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
  },

  async generateImage(
    this: TldwApiClientCore,
    payload: {
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
    }
  ): Promise<{ content_b64: string; content_type: string }> {
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
  },

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
  },

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
  },

  // Admin / diagnostics helpers
  async getSystemStats(options?: { timeoutMs?: number }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/stats",
      method: "GET",
      timeoutMs: options?.timeoutMs
    })
  },

  async getMediaIngestionBudgetDiagnostics(params: {
    userId: number
    policyId?: string
  }): Promise<MediaIngestionBudgetDiagnostics> {
    const query = buildQuery({
      user_id: params.userId,
      policy_id: params.policyId || "media.default"
    })
    return await bgRequest<MediaIngestionBudgetDiagnostics>({
      path: `/api/v1/resource-governor/diag/media-budget${query}`,
      method: "GET"
    })
  },

  async getLlamacppStatus(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/llamacpp/status",
      method: "GET"
    })
  },

  async listLlamacppModels(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/llamacpp/models",
      method: "GET"
    })
  },

  async getLlamacppConfig(): Promise<LlamacppConfigResponse> {
    return await bgRequest<LlamacppConfigResponse>({
      path: "/api/v1/llamacpp/config",
      method: "GET"
    })
  },

  async updateLlamacppConfig(
    payload: LlamacppConfigUpdateRequest
  ): Promise<LlamacppConfigResponse> {
    return await bgRequest<LlamacppConfigResponse>({
      path: "/api/v1/llamacpp/config",
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async validateLlamacpp(
    payload: LlamacppValidationRequest
  ): Promise<LlamacppValidationResponse> {
    return await bgRequest<LlamacppValidationResponse>({
      path: "/api/v1/llamacpp/validate",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async getLlamacppInventory(): Promise<LlamacppInventoryResponse> {
    return await bgRequest<LlamacppInventoryResponse>({
      path: "/api/v1/llamacpp/inventory",
      method: "GET"
    })
  },

  async getLlamacppAssets(): Promise<LlamacppAssetsResponse> {
    return await bgRequest<LlamacppAssetsResponse>({
      path: "/api/v1/llamacpp/assets",
      method: "GET"
    })
  },

  async registerLlamacppModelPath(path: string): Promise<LlamacppInventoryItem> {
    return await bgRequest<LlamacppInventoryItem>({
      path: "/api/v1/llamacpp/models/register-path",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { path }
    })
  },

  async registerLlamacppAssetPath(path: string): Promise<LlamacppAsset> {
    return await bgRequest<LlamacppAsset>({
      path: "/api/v1/llamacpp/assets/register-path",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { path }
    })
  },

  async importLlamacppAssetFolder(path: string): Promise<LlamacppAsset> {
    return await bgRequest<LlamacppAsset>({
      path: "/api/v1/llamacpp/assets/import-folder",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { path }
    })
  },

  async previewLlamacppAssetFolder(
    path: string
  ): Promise<LlamacppAssetImportPreviewResponse> {
    return await bgRequest<LlamacppAssetImportPreviewResponse>({
      path: "/api/v1/llamacpp/assets/import-folder/preview",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { path }
    })
  },

  async startLlamacppAssetDownload(
    payload: LlamacppAssetDownloadRequest
  ): Promise<LlamacppAcquisitionJobResponse> {
    return await bgRequest<LlamacppAcquisitionJobResponse>({
      path: "/api/v1/llamacpp/assets/downloads",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async listLlamacppAssetDownloads(): Promise<LlamacppAcquisitionJobListResponse> {
    return await bgRequest<LlamacppAcquisitionJobListResponse>({
      path: "/api/v1/llamacpp/assets/downloads",
      method: "GET"
    })
  },

  async getLlamacppAssetDownload(
    jobId: string | number
  ): Promise<LlamacppAcquisitionJobResponse> {
    return await bgRequest<LlamacppAcquisitionJobResponse>({
      path: `/api/v1/llamacpp/assets/downloads/${encodeURIComponent(String(jobId))}`,
      method: "GET"
    })
  },

  async cancelLlamacppAssetDownload(
    jobId: string | number
  ): Promise<LlamacppAcquisitionJobResponse> {
    return await bgRequest<LlamacppAcquisitionJobResponse>({
      path: `/api/v1/llamacpp/assets/downloads/${encodeURIComponent(String(jobId))}`,
      method: "DELETE"
    })
  },

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
  },

  async startLlamacppModel(
    modelId: string,
    serverArgs?: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    return await bgRequest<Record<string, unknown>>({
      path: "/api/v1/llamacpp/start-by-model",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        model_id: modelId,
        server_args: serverArgs || {}
      }
    })
  },

  async stopLlamacppServer(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/llamacpp/stop_server",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async useLlamacppInChat(): Promise<LlamacppUseInChatResponse> {
    return await bgRequest<LlamacppUseInChatResponse>({
      path: "/api/v1/llamacpp/use-in-chat",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async tailLlamacppLogs(lines?: number): Promise<LlamacppLogTailResponse> {
    const query = buildQuery(lines === undefined ? {} : { lines })
    return await bgRequest<LlamacppLogTailResponse>({
      path: `/api/v1/llamacpp/logs/tail${query}`,
      method: "GET"
    })
  },

  async listLlamacppProfiles(): Promise<LlamacppProfileListResponse> {
    return await bgRequest<LlamacppProfileListResponse>({
      path: "/api/v1/llamacpp/profiles",
      method: "GET"
    })
  },

  async createLlamacppProfile(
    payload: LlamacppProfileCreateRequest
  ): Promise<LlamacppProfile> {
    return await bgRequest<LlamacppProfile>({
      path: "/api/v1/llamacpp/profiles",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async updateLlamacppProfile(
    profileId: string,
    payload: LlamacppProfileUpdateRequest
  ): Promise<LlamacppProfile> {
    return await bgRequest<LlamacppProfile>({
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async deleteLlamacppProfile(
    profileId: string
  ): Promise<{ profile_id: string; deleted: boolean }> {
    return await bgRequest<{ profile_id: string; deleted: boolean }>({
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}`,
      method: "DELETE"
    })
  },

  async startLlamacppProfile(
    profileId: string
  ): Promise<LlamacppLifecycleActionResponse> {
    return await bgRequest<LlamacppLifecycleActionResponse>({
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}/start`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async stopLlamacppProfile(
    profileId: string
  ): Promise<LlamacppLifecycleActionResponse> {
    return await bgRequest<LlamacppLifecycleActionResponse>({
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}/stop`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async pauseLlamacppProfile(
    profileId: string
  ): Promise<LlamacppLifecycleActionResponse> {
    return await bgRequest<LlamacppLifecycleActionResponse>({
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}/pause`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async resumeLlamacppProfile(
    profileId: string
  ): Promise<LlamacppLifecycleActionResponse> {
    return await bgRequest<LlamacppLifecycleActionResponse>({
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}/resume`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async useLlamacppProfileInChat(
    profileId: string
  ): Promise<LlamacppUseInChatResponse> {
    return await bgRequest<LlamacppUseInChatResponse>({
      path: `/api/v1/llamacpp/profiles/${encodeURIComponent(profileId)}/use-in-chat`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async listLlamacppInstances(): Promise<LlamacppRuntimeListResponse> {
    return await bgRequest<LlamacppRuntimeListResponse>({
      path: "/api/v1/llamacpp/instances",
      method: "GET"
    })
  },

  async tailLlamacppInstanceLogs(
    profileId: string,
    lines?: number
  ): Promise<LlamacppLogTailResponse> {
    const query = buildQuery(lines === undefined ? {} : { lines })
    return await bgRequest<LlamacppLogTailResponse>({
      path: `/api/v1/llamacpp/instances/${encodeURIComponent(profileId)}/logs/tail${query}`,
      method: "GET"
    })
  },

  async getLlamacppHardware(): Promise<LlamacppHardwareSnapshotResponse> {
    return await bgRequest<LlamacppHardwareSnapshotResponse>({
      path: "/api/v1/llamacpp/hardware",
      method: "GET"
    })
  },

  async getLlmProviders(
    includeDeprecated = false
  ): Promise<any> {
    const query = buildQuery(includeDeprecated ? { include_deprecated: true } : {})
    return await bgRequest<any>({
      path: `/api/v1/llm/providers${query}`,
      method: "GET"
    })
  },

  // MLX admin helpers
  async getMlxStatus(): Promise<MlxStatus> {
    return await bgRequest<MlxStatus>({
      path: "/api/v1/llm/providers/mlx/status",
      method: "GET"
    })
  },

  async loadMlxModel(payload: MlxLoadRequest): Promise<MlxStatus> {
    return await bgRequest<MlxStatus>({
      path: "/api/v1/llm/providers/mlx/load",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async unloadMlxModel(payload?: MlxUnloadRequest): Promise<{ message?: string }> {
    return await bgRequest<{ message?: string }>({
      path: "/api/v1/llm/providers/mlx/unload",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload || {}
    })
  },

  // ── Audio: Transcription & TTS ──

  async getTranscriptionModels(
    this: TldwApiClientCore,
    options?: { timeoutMs?: number }
  ): Promise<any> {
    await this.ensureConfigForRequest(true)
    return await bgRequest<any>({
      path: "/api/v1/media/transcription-models",
      method: "GET",
      timeoutMs: options?.timeoutMs
    })
  },

  async getTranscriptionModelHealth(
    this: TldwApiClientCore,
    model: string
  ): Promise<any> {
    await this.ensureConfigForRequest(true)
    const query = buildQuery({ model })
    return await bgRequest<any>({
      path: `/api/v1/audio/transcriptions/health${query}`,
      method: "GET"
    })
  },

  async getTranscriptionCapabilities(
    this: TldwApiClientCore,
    options?: { timeoutMs?: number }
  ): Promise<any> {
    await this.ensureConfigForRequest(true)
    return await bgRequest<any>({
      path: "/api/v1/audio/transcriptions/capabilities",
      method: "GET",
      timeoutMs: options?.timeoutMs
    })
  },

  async listAudioPresets(
    this: TldwApiClientCore,
    options?: {
      kind?: AudioPresetKind
      favorite?: boolean
      is_default?: boolean
      limit?: number
      offset?: number
      timeoutMs?: number
    }
  ): Promise<AudioPresetListResponse> {
    await this.ensureConfigForRequest(true)
    const query = buildQuery({
      kind: options?.kind,
      favorite: options?.favorite,
      is_default: options?.is_default,
      limit: options?.limit,
      offset: options?.offset
    })
    return await this.requestWithCurrentConfig<AudioPresetListResponse>({
      path: `/api/v1/audio/presets${query}`,
      method: "GET",
      timeoutMs: options?.timeoutMs
    })
  },

  async createAudioPreset(
    this: TldwApiClientCore,
    payload: AudioPresetCreatePayload
  ): Promise<AudioPreset> {
    await this.ensureConfigForRequest(true)
    return await this.requestWithCurrentConfig<AudioPreset>({
      path: "/api/v1/audio/presets",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async updateAudioPreset(
    this: TldwApiClientCore,
    presetId: string,
    payload: AudioPresetUpdatePayload
  ): Promise<AudioPreset> {
    await this.ensureConfigForRequest(true)
    return await this.requestWithCurrentConfig<AudioPreset>({
      path: `/api/v1/audio/presets/${encodeURIComponent(presetId)}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async deleteAudioPreset(
    this: TldwApiClientCore,
    presetId: string
  ): Promise<void> {
    await this.ensureConfigForRequest(true)
    await this.requestWithCurrentConfig<void>({
      path: `/api/v1/audio/presets/${encodeURIComponent(presetId)}`,
      method: "DELETE"
    })
  },

  async validateAudioPreset(
    this: TldwApiClientCore,
    presetId: string
  ): Promise<AudioPresetValidationResponse> {
    await this.ensureConfigForRequest(true)
    return await this.requestWithCurrentConfig<AudioPresetValidationResponse>({
      path: `/api/v1/audio/presets/${encodeURIComponent(presetId)}/validate`,
      method: "POST"
    })
  },

  async transcribeAudio(
    this: TldwApiClientCore,
    audioFile: File | Blob,
    options?: any
  ): Promise<any> {
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
  },

  async synthesizeSpeech(
    this: TldwApiClientCore,
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
  },

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
  },

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
  },
}

export type ModelsAudioMethods = typeof modelsAudioMethods
