import {
  systemPromptForNonRagOption,
  getWebSearchPrompt
} from "~/services/tldw-server"
import {
  type ChatHistory,
  type Message,
  type MessageMetadataExtra,
  type ToolChoice
} from "~/store/option"
import { getPromptById } from "@/db/dexie/helpers"
import { generateHistory } from "@/utils/generate-history"
import { humanMessageFormatter } from "@/utils/human-message"
import { systemPromptFormatter } from "@/utils/system-message"
import type { ActorSettings } from "@/types/actor"
import { maybeInjectActorMessage } from "@/utils/actor"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { getSearchSettings } from "@/services/search"
import { resolveImageBackendCandidates } from "@/utils/image-backends"
import {
  getImageBackendConfigs,
  normalizeImageBackendConfig,
  parseExtraParams,
  resolveImageBackendConfig
} from "@/services/image-generation"
import type {
  ImageGenerationRefineMetadata,
  ImageGenerationPromptMode,
  ImageGenerationRequestSnapshot
} from "@/utils/image-generation-chat"
import type { DynamicUIRequest } from "@/types/dynamic-ui"
import type { SaveMessageData, SaveMessageErrorData } from "@/types/chat-modes"
import {
  runChatPipeline,
  type ChatModeDefinition
} from "./chatModePipeline"
import { appendSystemPromptSuffix } from "@/utils/output-formatting-guide"
import type { ChatSubmitResult } from "@/hooks/chat/chat-action-utils"

interface WebSearchPayload {
  query: string
  aggregate: boolean
  engine?: string
  result_count?: number
  searx_url?: string
  searx_json_mode?: boolean
  google_domain?: string
}

const MAX_WEBSEARCH_SNIPPET_LENGTH = 600

const truncateText = (value: string, max = MAX_WEBSEARCH_SNIPPET_LENGTH) => {
  const trimmed = value.trim()
  if (trimmed.length <= max) return trimmed
  return `${trimmed.slice(0, max - 1).trimEnd()}...`
}

const normalizeWebSearchResult = (result: any) => {
  const title =
    result?.title || result?.name || result?.metadata?.title || ""
  const url = result?.url || result?.link || ""
  const snippet =
    result?.content ||
    result?.snippet ||
    result?.text ||
    result?.metadata?.snippet ||
    ""
  const published =
    result?.metadata?.date_published ||
    result?.publishedDate ||
    result?.published ||
    null

  return {
    title: String(title || ""),
    url: String(url || ""),
    snippet: truncateText(String(snippet || "")),
    published
  }
}

const buildWebSearchPrompt = async (results: any[]) => {
  if (!Array.isArray(results) || results.length === 0) return null
  const prompt = await getWebSearchPrompt()
  const now = new Date().toISOString()
  const formattedResults = results
    .map((result, index) => {
      const normalized = normalizeWebSearchResult(result)
      const lines = [
        `Result ${index + 1}:`,
        `Title: ${normalized.title || "Untitled"}`,
        normalized.url ? `URL: ${normalized.url}` : null,
        normalized.snippet ? `Snippet: ${normalized.snippet}` : null,
        normalized.published ? `Published: ${normalized.published}` : null
      ].filter(Boolean)
      return lines.join("\n")
    })
    .join("\n\n")

  return prompt
    .replace("{current_date_time}", now)
    .replace("{search_results}", formattedResults)
}

const buildWebSearchSources = (results: any[]) => {
  if (!Array.isArray(results)) return []
  return results.map((result) => {
    const normalized = normalizeWebSearchResult(result)
    return {
      name: normalized.title || normalized.url || "Source",
      url: normalized.url || undefined,
      content: normalized.snippet,
      metadata: {
        title: normalized.title || undefined,
        source: normalized.title || normalized.url || undefined,
        date_published: normalized.published || undefined
      },
      mode: "web_search"
    }
  })
}

type NormalChatModeParams = {
  selectedModel: string
  useOCR: boolean
  selectedSystemPrompt: string
  currentChatModelSettings: any
  assistantIdentity?: {
    name: string
    avatarUrl?: string | null
  }
  imageBackendOverride?: string
  imageGenerationRequest?: Partial<ImageGenerationRequestSnapshot>
  imageGenerationRefine?: ImageGenerationRefineMetadata
  imageGenerationPromptMode?: ImageGenerationPromptMode
  imageGenerationSource?: "slash-command" | "generate-modal" | "message-regen"
  dynamicUIRequest?: DynamicUIRequest
  userMetadataExtra?: MessageMetadataExtra
  toolChoice?: ToolChoice
  setMessages: (messages: Message[] | ((prev: Message[]) => Message[])) => void
  saveMessageOnSuccess: (data: SaveMessageData) => Promise<string | null>
  saveMessageOnError: (data: SaveMessageErrorData) => Promise<string | null>
  setHistory: (history: ChatHistory) => void
  setIsProcessing: (value: boolean) => void
  setStreaming: (value: boolean) => void
  setAbortController: (controller: AbortController | null) => void
  historyId: string | null
  serverChatId?: string | null
  setHistoryId: (id: string) => void
  uploadedFiles?: any[]
  actorSettings?: ActorSettings
  systemPromptAppendix?: string
  overlaySystemPrompt?: string
  webSearch?: boolean
  setIsSearchingInternet?: (value: boolean) => void
  clusterId?: string
  userMessageType?: string
  assistantMessageType?: string
  modelIdOverride?: string
  userMessageId?: string
  assistantMessageId?: string
  userParentMessageId?: string | null
  assistantParentMessageId?: string | null
  historyForModel?: ChatHistory
  messageForModel?: string
  regenerateFromMessage?: Message
}

const isBackendUnavailableError = (error: unknown): boolean => {
  if (!error) return false
  const message = error instanceof Error ? error.message : String(error)
  return message.toLowerCase().includes("image_backend_unavailable")
}

const normalizeImageBackendOverride = (
  value?: string | null
): string | null => {
  if (!value) return null
  const trimmed = value.trim()
  return trimmed || null
}

const normalChatModeDefinition: ChatModeDefinition<NormalChatModeParams> = {
  id: "normal",
  buildUserMessage: (ctx) => ({
    isBot: false,
    name: "You",
    message: ctx.message,
    sources: [],
    images: ctx.image ? [ctx.image] : [],
    createdAt: ctx.createdAt,
    id: ctx.resolvedUserMessageId,
    modelImage: ctx.modelInfo?.model_avatar,
    modelName: ctx.modelInfo?.model_name || ctx.selectedModel,
    documents:
      ctx.uploadedFiles?.map((file) => ({
        type: "file",
        filename: file.filename,
        fileSize: file.size,
        processed: file.processed
      })) || [],
    messageType: ctx.userMessageType,
    clusterId: ctx.clusterId,
    modelId: ctx.userModelId,
    parentMessageId: ctx.userParentMessageId ?? null
  }),
  buildAssistantMessage: (ctx) => ({
    isBot: true,
    name:
      ctx.assistantIdentity?.name ||
      ctx.modelInfo?.model_name ||
      ctx.selectedModel,
    message: "▋",
    sources: [],
    createdAt: ctx.createdAt,
    id: ctx.resolvedAssistantMessageId,
    modelImage:
      ctx.assistantIdentity?.avatarUrl || ctx.modelInfo?.model_avatar,
    modelName:
      ctx.assistantIdentity?.name ||
      ctx.modelInfo?.model_name ||
      ctx.selectedModel,
    messageType: ctx.assistantMessageType,
    clusterId: ctx.clusterId,
    modelId: ctx.resolvedModelId,
    parentMessageId: ctx.resolvedAssistantParentMessageId ?? null
  }),
  preflight: async (ctx) => {
    const requestOverride = ctx.imageGenerationRequest || {}
    const requestBackend = normalizeImageBackendOverride(requestOverride.backend)
    const overrideBackend =
      normalizeImageBackendOverride(ctx.imageBackendOverride) || requestBackend
    const overrideCandidates = overrideBackend
      ? [overrideBackend, ...resolveImageBackendCandidates(overrideBackend)]
      : []
    let imageBackendCandidates =
      overrideCandidates.length > 0
        ? Array.from(new Set(overrideCandidates))
        : resolveImageBackendCandidates(
            ctx.currentChatModelSettings?.apiProvider,
            ctx.selectedModel
          )
    if (imageBackendCandidates.length > 0) {
      const promptSource =
        typeof requestOverride.prompt === "string"
          ? requestOverride.prompt
          : ctx.message
      const prompt = promptSource.trim()
      ctx.setIsProcessing(true)
      let lastError: unknown = null

      const backendConfigs = await getImageBackendConfigs().catch(() => ({}))

      await tldwClient.initialize()
      for (const backend of imageBackendCandidates) {
        if (ctx.signal?.aborted) break
        try {
          if (!prompt) {
            throw new Error("Image prompt is required.")
          }
          const rawConfig = resolveImageBackendConfig(backend, backendConfigs)
          const config = normalizeImageBackendConfig(rawConfig)
          const overrideExtraParams =
            requestOverride.extraParams &&
            typeof requestOverride.extraParams === "object" &&
            !Array.isArray(requestOverride.extraParams)
              ? (requestOverride.extraParams as Record<string, unknown>)
              : undefined
          const extraParams =
            overrideExtraParams ?? parseExtraParams(config.extraParams)
          const format = requestOverride.format || config.format || "png"
          const negativePrompt =
            requestOverride.negativePrompt || config.negativePrompt
          const width =
            typeof requestOverride.width === "number"
              ? requestOverride.width
              : config.width
          const height =
            typeof requestOverride.height === "number"
              ? requestOverride.height
              : config.height
          const steps =
            typeof requestOverride.steps === "number"
              ? requestOverride.steps
              : config.steps
          const cfgScale =
            typeof requestOverride.cfgScale === "number"
              ? requestOverride.cfgScale
              : config.cfgScale
          const seed =
            typeof requestOverride.seed === "number"
              ? requestOverride.seed
              : config.seed
          const sampler = requestOverride.sampler || config.sampler
          const model = requestOverride.model || config.model

          const requestSnapshot: ImageGenerationRequestSnapshot = {
            prompt,
            backend,
            format,
            negativePrompt,
            referenceFileId: requestOverride.referenceFileId,
            width,
            height,
            steps,
            cfgScale,
            seed,
            sampler,
            model,
            extraParams
          }
          const response = await tldwClient.createImageArtifact({
            backend,
            prompt,
            format,
            negativePrompt,
            referenceFileId: requestOverride.referenceFileId,
            width,
            height,
            steps,
            cfgScale,
            seed,
            sampler,
            model,
            extraParams
          })
          const exportInfo = response?.artifact?.export
          const contentB64 = exportInfo?.content_b64
          const contentType = exportInfo?.content_type || "image/png"
          if (!contentB64) {
            throw new Error("Image generation returned no data.")
          }
          const imageUrl = `data:${contentType};base64,${contentB64}`
          return {
            handled: true,
            fullText: "",
            images: [imageUrl],
            generationInfo: {
              file_id: response?.artifact?.file_id,
              content_type: contentType,
              bytes: exportInfo?.bytes,
              format: exportInfo?.format,
              image_generation: {
                request: requestSnapshot,
                promptMode: ctx.imageGenerationPromptMode,
                source:
                  ctx.imageGenerationSource ||
                  (ctx.imageGenerationRequest ? "generate-modal" : "slash-command"),
                createdAt: Date.now(),
                refine: ctx.imageGenerationRefine,
                refine_model: ctx.imageGenerationRefine?.model,
                refine_latency_ms: ctx.imageGenerationRefine?.latencyMs,
                diff_stats: ctx.imageGenerationRefine?.diffStats
              }
            },
            skipHistoryAppend: true
          }
        } catch (error) {
          if (isBackendUnavailableError(error)) {
            lastError = error
            continue
          }
          throw error
        }
      }

    if (lastError) {
      throw lastError
    }
    return null
  }
    return null
  },
  preparePrompt: async (ctx) => {
    const prompt = await systemPromptForNonRagOption()
    const selectedPrompt = await getPromptById(ctx.selectedSystemPrompt)
    const promptId = ctx.selectedSystemPrompt
    let promptContent: string | undefined = undefined
    let webSearchSources: any[] = []
    let webSearchSystemMessage: any | null = null
    const messageForModel = ctx.messageForModel ?? ctx.message

    let humanMessage = await humanMessageFormatter({
      content: [
        {
          text: messageForModel,
          type: "text"
        }
      ],
      model: ctx.selectedModel,
      useOCR: ctx.useOCR
    })
    if (ctx.image.length > 0) {
      humanMessage = await humanMessageFormatter({
        content: [
          {
            text: messageForModel,
            type: "text"
          },
          {
            image_url: ctx.image,
            type: "image_url"
          }
        ],
        model: ctx.selectedModel,
        useOCR: ctx.useOCR
      })
    }

    if (ctx.webSearch) {
      ctx.setIsProcessing(true)
      if (ctx.setIsSearchingInternet) {
        ctx.setIsSearchingInternet(true)
      }
      try {
        await tldwClient.initialize()
        const {
          searchProvider,
          totalSearchResults,
          searxngURL,
          searxngJSONMode,
          googleDomain
        } = await getSearchSettings()

        const engineMap: Record<string, string> = {
          google: "google",
          duckduckgo: "duckduckgo",
          brave: "brave",
          "brave-api": "brave",
          searxng: "searx",
          "tavily-api": "tavily",
          exa: "exa",
          firecrawl: "firecrawl",
          sogou: "sogou",
          baidu: "baidu",
          bing: "bing",
          stract: "stract",
          startpage: "startpage"
        }
        const provider = (searchProvider || "").toLowerCase()
        const engine = engineMap[provider]

        const payload: WebSearchPayload = {
          query: ctx.message,
          aggregate: false
        }
        if (engine) {
          payload.engine = engine
        }
        if (typeof totalSearchResults === "number" && totalSearchResults > 0) {
          payload.result_count = totalSearchResults
        }
        if (provider === "searxng" && searxngURL) {
          payload.searx_url = searxngURL
        }
        if (provider === "searxng" && searxngJSONMode) {
          payload.searx_json_mode = true
        }
        if (provider === "google" && googleDomain) {
          payload.google_domain = googleDomain
        }

        const res = await tldwClient.webSearch({
          ...payload,
          signal: ctx.signal
        })

        if (res?.error) {
          throw new Error(
            typeof res.error === "string"
              ? res.error
              : res.error?.message || "Web search failed"
          )
        }

        const results = res?.web_search_results_dict?.results || []
        webSearchSources = buildWebSearchSources(results)
        const webSearchPrompt = await buildWebSearchPrompt(results)
        if (webSearchPrompt) {
          webSearchSystemMessage = await systemPromptFormatter({
            content: webSearchPrompt
          })
        }
      } catch (error) {
        console.error("Web search failed, continuing without context", error)
      } finally {
        if (ctx.setIsSearchingInternet) {
          ctx.setIsSearchingInternet(false)
        }
      }
    }

    let applicationChatHistory = generateHistory(
      ctx.historyForModel ?? ctx.history,
      ctx.selectedModel
    )
    const baseSystemPrompts: string[] = []
    const overlayPrompt = ctx.overlaySystemPrompt?.trim() || ""
    const resolvePromptWithAppendix = (content?: string | null) =>
      appendSystemPromptSuffix(content || "", ctx.systemPromptAppendix)

    if (!selectedPrompt) {
      const resolvedDefaultPrompt = resolvePromptWithAppendix(prompt)
      if (resolvedDefaultPrompt) {
        baseSystemPrompts.push(resolvedDefaultPrompt)
      }
    }

    const isTempSystemprompt =
      ctx.currentChatModelSettings.systemPrompt &&
      ctx.currentChatModelSettings.systemPrompt?.trim().length > 0

    if (!isTempSystemprompt && selectedPrompt) {
      const selectedPromptContent =
        selectedPrompt.system_prompt ?? selectedPrompt.content
      const resolvedSelectedPrompt =
        resolvePromptWithAppendix(selectedPromptContent)
      if (resolvedSelectedPrompt) {
        baseSystemPrompts.push(resolvedSelectedPrompt)
      }
      promptContent = resolvedSelectedPrompt
    }

    if (isTempSystemprompt) {
      const resolvedTemporaryPrompt = resolvePromptWithAppendix(
        ctx.currentChatModelSettings.systemPrompt
      )
      if (resolvedTemporaryPrompt) {
        baseSystemPrompts.push(resolvedTemporaryPrompt)
      }
      promptContent = resolvedTemporaryPrompt
    }

    if (overlayPrompt) {
      baseSystemPrompts.push(overlayPrompt)
    }

    if (baseSystemPrompts.length > 0) {
      const promptMessages = await Promise.all(
        baseSystemPrompts.map((content) =>
          systemPromptFormatter({
            content
          })
        )
      )
      applicationChatHistory = [...promptMessages, ...applicationChatHistory]
    }

    const templatesActive = !!ctx.selectedSystemPrompt
    applicationChatHistory = await maybeInjectActorMessage(
      applicationChatHistory,
      ctx.actorSettings || null,
      templatesActive
    )

    if (webSearchSystemMessage) {
      applicationChatHistory.push(webSearchSystemMessage)
    }

    return {
      chatHistory: applicationChatHistory,
      humanMessage,
      sources: webSearchSources,
      promptId,
      promptContent
    }
  }
}

export const normalChatMode = async (
  message: string,
  image: string,
  isRegenerate: boolean,
  messages: Message[],
  history: ChatHistory,
  signal: AbortSignal,
  params: NormalChatModeParams
): Promise<ChatSubmitResult> => {
  console.log("Using normalChatMode")
  const resolvedImage =
    image.length > 0 ? `data:image/jpeg;base64,${image.split(",")[1]}` : ""

  return runChatPipeline(
    normalChatModeDefinition,
    message,
    resolvedImage,
    isRegenerate,
    messages,
    history,
    signal,
    params
  )
}
