import { captureChatDispatchSource } from "@/hooks/chat/chat-dispatch-source"
import type { ActorSceneSend } from "@/hooks/chat/actor-scene-source"
import type {
  HistoryLoadReceipt,
  HistorySelectionController
} from "@/hooks/chat/useHistorySelection"
import { captureChatDraftLoad, captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { NativeHistoryBranchChoice, NativeHistoryBranchRefusal, NativeHistoryEditRefusal } from "@/hooks/chat/native-character-preflight"
import type {
  HistorySendTurn,
  NativeHistoryEditResend
} from "@/types/chat-modes"
import type { HistoryAdmissionReferenceV1 } from "@/types/history-selection"
import {
  assertHistoryConversationTarget,
  captureHistorySnapshot,
  nativeHistoryNeedsBranchChoice,
  requireNativeHistoryEditAdmission,
  historyAdmissionReference,
  isDefinitiveNativeBranchRejection,
  isDefinitiveNativeEditRejection,
  nativeEditResendView
} from "@/services/chat-history-selection"
import {
  saveHistoryTurnRecovery,
  dismissHistoryTurnRecovery
} from "@/db/dexie/history-selection"
import type { HistoryTurnRecovery } from "@/db/dexie/types"
import {
  keepHistoryTurnRecovery,
  markHistoryTurnLive,
  releaseHistoryTurn
} from "@/services/history-turn-keep"
import { beginServerChatWrite } from "@/store/server-chat-save-status"
import {
  beginChatTurnSettlement,
  wasChatTurnStoppedByUser
} from "@/hooks/chat/abort-turn-cleanup"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { resolveChatToolRequest } from "@/utils/chat-tools"
import { systemPromptForNonRagOption } from "~/services/tldw-server"
import {
  useStoreMessageOption,
  type ChatHistory,
  type Message,
  type MessageMetadataExtra,
  type ToolChoice
} from "~/store/option"
import {
  getPromptById,
  saveHistory,
  formatSelectedHistory
} from "@/db/dexie/helpers"
import { generateHistory } from "@/utils/generate-history"
import { excludeLocalRagDiagnostics } from "@/utils/local-rag-diagnostic"
import { createImageDataUrl } from "@/utils/image-utils"
import { humanMessageFormatter } from "@/utils/human-message"
import { systemPromptFormatter } from "@/utils/system-message"
import type { ActorSettings } from "@/types/actor"
import type { ActorSceneRead } from "@/services/actor-settings"
import { bindActorSceneAfterLoad } from "@/hooks/chat/actor-scene-source"
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
  getRequiredServicePrompt,
  runChatPipeline,
  type ChatModeDefinition,
  type ChatModeParamsBase
} from "./chatModePipeline"
import { appendSystemPromptSuffix } from "@/utils/output-formatting-guide"
import { chatSubmitFailed, type ChatSubmitResult } from "@/hooks/chat/chat-action-utils"
import { createChatWithRecovery, markChatCreationBound, markChatCreationSettled } from "@/services/pending-chat-promotion"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import type { ChatDraftAddress, ChatDraftHandle } from "@/utils/chat-draft-address"
import {
  loadServicePromptSnapshot,
  renderServicePromptPart,
  type ServicePromptSnapshot
} from "@/services/service-prompts"
import { isRequestConfigScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import type { ChatScope } from "@/types/chat-scope"
import { WEBUI_CHAT_SOURCE } from "@/utils/character-chat-session"

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
  const title = result?.title || result?.name || result?.metadata?.title || ""
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

const buildWebSearchPrompt = (
  results: any[],
  prompt: ReturnType<typeof getRequiredServicePrompt>
) => {
  if (!Array.isArray(results) || results.length === 0) return null
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

  return renderServicePromptPart(
    prompt.definition,
    "template",
    prompt.parts.template,
    { current_date_time: now, search_results: formattedResults }
  )
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
  surface?: "playground" | "sidepanel"
  historyPreparation?: {
    dispatchIsCurrent: () => boolean
    draftIsCurrent: () => boolean
    acceptDraftBind: ReturnType<typeof captureChatDraftLoad> | null
    sourceDraft: ChatDraftAddress | null
  }
  historySelection?: {
    controller: HistorySelectionController
    originIsCurrent: () => boolean
    sourceIsCurrent?: () => boolean
    onResultFollow?: () => void
    temporary?: boolean
    createServerChat?: boolean
    /**
     * Retry of a reply that ended early (CS-04): the admission the question
     * already has. The view must stand just before that question.
     */
    retryAdmission?: HistoryAdmissionReferenceV1
    editResend?: NativeHistoryEditResend
  }
  historyTurn?: HistorySendTurn

  selectedModel: string
  /** Resolved selector values still follow their global source unless explicitly overridden. */
  selectionSource?: {
    model: "global" | "explicit"
    toolChoice: "global" | "explicit"
  }
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
  ownsAbortController?: (signal: AbortSignal) => boolean
  releaseAbortControllerIfOwned?: (signal: AbortSignal) => boolean
  discardCurrentTurnOnAbort?: () => boolean
  wasStoppedByUser?: () => boolean
  historyId: string | null
  serverChatId?: string | null
  setServerChatId?: (id: string) => void
  scope?: ChatScope
  setHistoryId: (id: string) => void
  uploadedFiles?: any[]
  actorSceneSend?: ActorSceneSend
  actorSettings?: ActorSettings
  actorScene?: ActorSceneRead
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
  servicePromptSnapshot?: ServicePromptSnapshot
  onResponseDispatch?: () => void
}

/** Existing H1 ownership and preparation shared by the local context modes. */
export type HistoryChatModeParams = Pick<NormalChatModeParams,
  | "surface" | "historyPreparation" | "historySelection" | "historyTurn"
  | "selectionSource" | "serverChatId" | "setServerChatId" | "scope"
  | "actorScene" | "webSearch" | "uploadedFiles" | "onResponseDispatch"
>

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
    modelImage: ctx.assistantIdentity?.avatarUrl || ctx.modelInfo?.model_avatar,
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
    const requestBackend = normalizeImageBackendOverride(
      requestOverride.backend
    )
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
      const assertImagePreparationCurrent = () => {
        ctx.assertPreparationCurrent?.()
        if (ctx.actorSceneSend?.isCurrent() === false)
          throw new Error("actor_settings_scope_changed")
        ctx.signal?.throwIfAborted()
      }
      assertImagePreparationCurrent()
      const promptSource =
        typeof requestOverride.prompt === "string"
          ? requestOverride.prompt
          : ctx.message
      const prompt = promptSource.trim()
      ctx.setIsProcessing(true)
      let lastError: unknown = null

      const backendConfigs = await getImageBackendConfigs().catch(() => ({}))
      assertImagePreparationCurrent()

      await tldwClient.initialize()
      assertImagePreparationCurrent()
      for (const backend of imageBackendCandidates) {
        assertImagePreparationCurrent()
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
          assertImagePreparationCurrent()
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
          }, {
            requestScope: ctx.servicePromptSnapshot?.requestScope ?? ctx.actorSceneSend?.requestScope,
            signal: ctx.signal,
            onDispatch: ctx.onResponseDispatch
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
                  (ctx.imageGenerationRequest
                    ? "generate-modal"
                    : "slash-command"),
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
    ctx.assertPreparationCurrent?.()
    const webSearchPrompt = ctx.webSearch
      ? getRequiredServicePrompt(
          ctx.servicePromptSnapshot,
          "chat.web_search.answer"
        )
      : null
    const prompt = await systemPromptForNonRagOption()
    ctx.assertPreparationCurrent?.()
    const selectedPrompt = await getPromptById(ctx.selectedSystemPrompt)
    ctx.assertPreparationCurrent?.()
    const promptId = ctx.selectedSystemPrompt
    let promptContent: string | undefined = undefined
    let webSearchSources: any[] = []
    let webSearchSystemMessage: any | null = null
    let webSearchResults: any[] = []
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
    ctx.assertPreparationCurrent?.()
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
      ctx.assertPreparationCurrent?.()
    }

    if (ctx.webSearch) {
      ctx.setIsProcessing(true)
      if (ctx.setIsSearchingInternet) {
        ctx.setIsSearchingInternet(true)
      }
      try {
        ctx.assertPreparationCurrent?.()
        await tldwClient.initialize()
        ctx.assertPreparationCurrent?.()
        const {
          searchProvider,
          totalSearchResults,
          searxngURL,
          searxngJSONMode,
          googleDomain
        } = await getSearchSettings()
        ctx.assertPreparationCurrent?.()

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

        ctx.assertPreparationCurrent?.()
        const res = await tldwClient.webSearch({
          ...payload,
          signal: ctx.signal,
          requestScope: ctx.servicePromptSnapshot?.requestScope
        })
        ctx.assertPreparationCurrent?.()

        if (res?.error) {
          throw new Error(
            typeof res.error === "string"
              ? res.error
              : res.error?.message || "Web search failed"
          )
        }

        webSearchResults = res?.web_search_results_dict?.results || []
        webSearchSources = buildWebSearchSources(webSearchResults)
      } catch (error) {
        ctx.assertPreparationCurrent?.()
        if (ctx.signal.aborted || isRequestConfigScopeChangedError(error)) {
          throw error
        }
        console.error("Web search failed, continuing without context", error)
      } finally {
        if (ctx.setIsSearchingInternet) {
          ctx.setIsSearchingInternet(false)
        }
      }
    }

    if (webSearchPrompt) {
      const renderedWebSearchPrompt = buildWebSearchPrompt(
        webSearchResults,
        webSearchPrompt
      )
      if (renderedWebSearchPrompt) {
        webSearchSystemMessage = await systemPromptFormatter({
          content: renderedWebSearchPrompt
        })
        ctx.assertPreparationCurrent?.()
      }
    }

    let applicationChatHistory = generateHistory(
      ctx.historyForModel ?? ctx.history,
      ctx.selectedModel,
      ctx.historyTurn ? { versioned: true } : undefined
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
      const resolvedSelectedPrompt = resolvePromptWithAppendix(
        selectedPromptContent
      )
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
      ctx.assertPreparationCurrent?.()
      applicationChatHistory = [...promptMessages, ...applicationChatHistory]
    }

    const templatesActive = !!ctx.selectedSystemPrompt
    applicationChatHistory = await maybeInjectActorMessage(
      applicationChatHistory,
      ctx.actorSettings || null,
      templatesActive
    )
    ctx.assertPreparationCurrent?.()

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

/** How often a streaming reply is checkpointed so a reload can keep it (CS-04). */
const HISTORY_TURN_CHECKPOINT_INTERVAL_MS = 1_000

/** Capture once before caller setup waits; explicit overrides retain their own values. */
export const captureNormalChatPreparation = (
  params: Pick<
    NormalChatModeParams,
    | "selectedModel"
    | "selectionSource"
    | "toolChoice"
    | "historyId"
    | "serverChatId"
  >,
  nativeSourceIsCurrent?: () => boolean
) => {
  const dispatchIsCurrent = captureChatDispatchSource(params)
  const sourceAddress = useStoreMessageOption.getState().chatDraftAddress
  const sourceDraft = sourceAddress ? structuredClone(sourceAddress) : null
  const acceptDraftBind =
    !params.historyId && !params.serverChatId && sourceAddress?.kind === "fresh"
      ? captureChatDraftLoad("bind")
      : null
  const draftIsCurrent =
    acceptDraftBind?.isCurrent ?? (sourceAddress?.kind === "native" && nativeSourceIsCurrent ? nativeSourceIsCurrent : captureChatDraftSource().isCurrent)
  return { dispatchIsCurrent, draftIsCurrent, acceptDraftBind, sourceDraft }
}

/** Establish one operation-lived adapter; display leases never authorize its later writes. */
const captureNormalHistoryTurn = async (
  params: ChatModeParamsBase & HistoryChatModeParams,
  message: string,
  snapshot: ServicePromptSnapshot,
  signal: AbortSignal,
  dispatchIsCurrent: () => boolean,
  draftIsCurrent: () => boolean,
  acceptDraftBind: ReturnType<typeof captureChatDraftLoad> | null,
  sourceDraft: ChatDraftAddress | null,
  stopSignal: AbortSignal
): Promise<HistorySendTurn> => {
  const { controller, originIsCurrent, temporary } = params.historySelection!
  let creationHandle: ChatDraftHandle | undefined
  if (temporary || params.historyId === "temp")
    throw new Error("temporary_history_unavailable")
  if (!originIsCurrent()) throw new Error("stale_selection")
  let current = controller.getCurrent()
  let selectionCurrent = originIsCurrent
  let targetHistoryId = params.historyId
  let targetServerChatId = params.serverChatId
  const preparationIsCurrent = () => draftIsCurrent() && acceptDraftBind?.isCurrent() !== false &&
    !signal.aborted && !snapshot.scopeInvalidatedSignal.aborted && dispatchIsCurrent()
  if (!preparationIsCurrent()) throw new Error("stale_selection")
  if (
    params.actorScene &&
    (!params.actorScene.target.isCurrent() ||
      (params.actorScene.target.localProfileId === undefined &&
        (!params.actorScene.target.account?.isCurrent() ||
          params.actorScene.target.accountKey !==
            chatDraftOwnerKey(snapshot.requestScope))))
  )
    throw new Error("actor_settings_scope_changed")
  if (!current.owner && current.status === "idle") {
    const scope = params.scope ? structuredClone(params.scope) : undefined
    let localId = params.historyId
    let serverId = params.serverChatId
    const toolsState = useMcpToolsStore.getState()
    const toolRequest = resolveChatToolRequest({
      tools: toolsState.chatTools ?? toolsState.tools,
      toolChoice: params.toolChoice ?? useStoreMessageOption.getState().toolChoice,
      mcpHealthState: toolsState.healthState,
      hasMcp: toolsState.healthState !== "unavailable"
    })
    if (!localId && !serverId && !current.view &&
      params.historySelection!.createServerChat && !toolRequest.tools?.length) {
      if (!sourceDraft || sourceDraft.kind !== "fresh" || !params.surface)
        throw new Error("chat_creation_source_invalid")
      creationHandle = { address: sourceDraft, ownerHash: chatDraftOwnerKey(snapshot.requestScope) }
      const created = await createChatWithRecovery({
        snapshot,
        handle: creationHandle,
        scope,
        surface: params.surface,
        inputText: message,
        setup: {},
        isCurrent: () => originIsCurrent() && preparationIsCurrent(),
        create: (id) =>
          tldwClient.createChat(
            {
              id,
              title: message.trim().slice(0, 80) || "Untitled Chat",
              source:
                scope?.type === "workspace" ? undefined : WEBUI_CHAT_SOURCE
            },
            { scope, signal, requestScope: snapshot.requestScope }
          )
      })

      // The recovery record keeps a late ACK even when this view can no longer bind.
      if (!originIsCurrent() || !preparationIsCurrent()) throw new Error("stale_selection")
      serverId = created.id
    }
    if (!localId && !serverId) {
      const created = await saveHistory(
        message.trim().slice(0, 80) || "Untitled Chat",
        false,
        "web-ui",
        undefined,
        undefined,
        snapshot.requestScope
      )
      // Creation may have succeeded after navigation. It remains its own local history.
      if (!originIsCurrent() || acceptDraftBind?.isCurrent() === false) throw new Error("stale_selection")
      localId = created.id
      params.setHistoryId(localId)
    }
    if (!originIsCurrent()) throw new Error("stale_selection")
    targetHistoryId = localId
    targetServerChatId = serverId
    let loadPhase: "preparing" | "accepted" | "rejected" = "preparing"
    const target = {
      historyId: localId, serverChatId: serverId, scope,
      // Accepted saved ownership is independent of the originating turn's Stop.
      isCurrent: () => acceptDraftBind?.isCurrent() !== false &&
        (loadPhase === "accepted" || (loadPhase === "preparing" && preparationIsCurrent()))
    }
    let receipt: HistoryLoadReceipt | undefined
    let adoptionIsCurrent: (() => boolean) | undefined
    try {
      const loaded = await controller.loadConversation(target, null, value => {
        loadPhase = "rejected"
        const now = controller.getCurrent()
        if (!preparationIsCurrent() || now.owner !== value.owner || now.view !== value.view) return
        const expectedId = target.serverChatId || (!params.historyId ? localId : null)
        const expectedKind = target.serverChatId ? "native" : "local"
        if (expectedId && (value.owner.kind !== expectedKind || value.owner.conversation_id !== expectedId)) return
        // Draft binding validates this owner's lease through the same guard.
        loadPhase = "preparing"
        if (acceptDraftBind && !acceptDraftBind(value)) {
          loadPhase = "rejected"
          return
        }
        receipt = value
        loadPhase = "accepted"
        adoptionIsCurrent = controller.fence()
        if (serverId) params.setServerChatId?.(serverId)
        if (acceptDraftBind) {
          const bindingIsCurrent = params.historyPreparation!.draftIsCurrent
          let metadataRetired = false
          // Raw display IDs guard this command, independently of the accepted H1 load.
          params.historyPreparation!.draftIsCurrent = () => {
            const state = useStoreMessageOption.getState()
            const valid = !metadataRetired && bindingIsCurrent() &&
              state.historyId === (localId ?? null) && state.serverChatId === (serverId ?? null)
            if (!valid) metadataRetired = true
            return valid
          }
        }
      })
      current = controller.getCurrent()
      if (!loaded || !receipt || current.owner !== receipt.owner || current.view !== receipt.view ||
        !adoptionIsCurrent?.() || !preparationIsCurrent() || receipt.owner.kind === "unavailable" || receipt.owner.validate_lease?.() === false)
        throw new Error("stale_selection")
      selectionCurrent = adoptionIsCurrent
      if (creationHandle && serverId) await markChatCreationBound(creationHandle, serverId)
      if (!selectionCurrent() || !preparationIsCurrent()) throw new Error("stale_selection")
      if (acceptDraftBind && params.actorScene) {
        const acceptedReceipt = receipt
        const boundIsCurrent = () =>
          acceptDraftBind.isCurrent() &&
          controller.getCurrent().owner === acceptedReceipt.owner &&
          controller.getCurrent().view?.owner_key ===
            acceptedReceipt.view.owner_key &&
          controller.getCurrent().view?.conversation_id ===
            acceptedReceipt.view.conversation_id
        const scenePreparationIsCurrent = () =>
          preparationIsCurrent() &&
          adoptionIsCurrent?.() === true &&
          controller.getCurrent().owner === acceptedReceipt.owner &&
          controller.getCurrent().view === acceptedReceipt.view
        if (
          !(await bindActorSceneAfterLoad(
            params.actorScene,
            acceptedReceipt,
            boundIsCurrent,
            scenePreparationIsCurrent
          ))
        )
          throw new Error("actor_settings_bind_not_committed")
      }
      if (!selectionCurrent() || !preparationIsCurrent() || controller.getCurrent().owner !== receipt.owner || controller.getCurrent().view !== receipt.view)
        throw new Error("stale_selection")
    } finally {
      if (!receipt || !adoptionIsCurrent?.()) loadPhase = "rejected"
    }
  }
  if (
    current.status !== "ready" ||
    !current.owner ||
    current.owner.kind === "unavailable" ||
    !current.view ||
    !current.bookmarkScope ||
    current.capture?.status !== "captured"
  ) {
    throw new Error(current.error || "history_selection_not_ready")
  }
  if (current.owner.validate_lease?.() === false) throw new Error("stale_selection")
  await assertHistoryConversationTarget(current.owner, { historyId: targetHistoryId, serverChatId: targetServerChatId })
  if (!selectionCurrent() || controller.getCurrent().owner !== current.owner || controller.getCurrent().view !== current.view || current.owner.validate_lease?.() === false)
    throw new Error("stale_selection")
  const originalOwner = current.owner
  const originalCapture = current.capture
  const originView = structuredClone(current.view)
  const editResend = params.historySelection!.editResend
  const branchIntent = current.owner.kind === "native" && !editResend &&
    controller.takeBranchIntent?.(current.view) === true
  const view = editResend
    ? nativeEditResendView(
        current.owner,
        originView,
        current.capture,
        editResend
      )
    : structuredClone(originView)
  const retryAdmission = params.historySelection!.retryAdmission
  if (retryAdmission &&
    (retryAdmission.owner_key !== view.owner_key || retryAdmission.conversation_id !== view.conversation_id ||
      view.cursor.kind !== "before_message" || view.cursor.message_id !== retryAdmission.input_message_id))
    throw new Error("invalid_history_retry")
  const bookmarkScope = { ...current.bookmarkScope }
  const authValid = () => !snapshot.scopeInvalidatedSignal.aborted
  const owner =
    current.owner.kind === "native"
      ? {
          ...current.owner,
          owner_key: view.owner_key,
          request_scope: snapshot.requestScope,
          validate_lease: authValid
        }
      : {
          ...current.owner,
          // The accepted local question settles under its original account,
          // independently of the controller's retired display/load lease.
          validate_lease: authValid
        }
  const validateLease = () =>
    authValid() &&
    dispatchIsCurrent() && (owner.kind !== "native" || !!turn.admission || editPreparationIsCurrent())
  const canUpdateView = () => {
    const now = controller.getCurrent().view
    return (
      draftIsCurrent() && acceptDraftBind?.isCurrent() !== false && selectionCurrent() && controller.getCurrent().owner === originalOwner &&
      !!now &&
      now.owner_key === originView.owner_key &&
      now.conversation_id === originView.conversation_id &&
      now.view_session_id === originView.view_session_id &&
      now.selection_revision === originView.selection_revision
    )
  }
  const editOriginIsCurrent = () =>
    selectionCurrent() &&
    preparationIsCurrent() &&
    controller.getCurrent().owner === current.owner &&
    controller.getCurrent().view === current.view &&
    originalOwner.validate_lease?.() !== false &&
    canUpdateView() &&
    editResend?.isOriginCurrent() !== false
  const editPreparationIsCurrent = () =>
    editOriginIsCurrent() && editResend?.isCurrent() !== false
  const capture = await captureHistorySnapshot(owner, view, "send", signal)
  if (capture.status !== "captured") throw new Error(capture.code)
  if (!canUpdateView()) throw new Error("stale_selection")
  if (editResend) {
    if (!editPreparationIsCurrent()) throw new Error("stale_selection")
    nativeEditResendView(owner, originView, capture, editResend, originalCapture)
  }
  const operationId = crypto.randomUUID()
  // Until this turn ends, no view may keep its record: the reply is still
  // being written (CS-04). Released by normalChatMode when the turn ends.
  markHistoryTurnLive(operationId)
  // Record writes run in order, so a late checkpoint never replaces the final
  // record of the turn.
  let recoveryWrites: Promise<void> = Promise.resolve()
  let lastRecord: HistoryTurnRecovery | null = null
  const saveRecord = (next: HistoryTurnRecovery) => {
    const write = recoveryWrites.then(async () => {
      lastRecord = next
      await saveHistoryTurnRecovery(bookmarkScope, view, next)
    })
    recoveryWrites = write.catch(() => undefined)
    return write
  }
  const record = (
    turn: HistorySendTurn,
    content: string,
    state: HistoryTurnRecovery["state"],
    extra: Pick<
      Partial<HistoryTurnRecovery>,
      "outcome" | "interruption_reason" | "settled_message_id"
    > = {}
  ): HistoryTurnRecovery => ({
    operation_id: operationId,
    ...(editResend ? { action: "edit_resend" as const } : {}),
    origin_view: view,
    // A retry's input was admitted under the selection of its first send.
    selection_digest:
      turn.retryAdmission?.selection_digest ?? turn.selection!.selection_digest,
    request_context_digest: turn.selection!.request_context_digest,
    owner_key: view.owner_key,
    conversation_id: view.conversation_id,
    input_id: turn.input!.id,
    assistant_id: turn.assistantId!,
    created_at: turn.createdAt!,
    input_text: turn.input!.content,
    input_images: turn.input!.images ?? [],
    result_text: content,
    state,
    ...(turn.replyModel?.name ? { model_name: turn.replyModel.name } : {}),
    ...(turn.replyModel?.id ? { model_id: turn.replyModel.id } : {}),
    ...(turn.admission
      ? { admission: historyAdmissionReference(turn.admission) }
      : {}),
    ...extra
  })
  // A server chat holds the question once it is admitted; until the reply is
  // settled there too, its latest write is still in flight (CS-04).
  let endReplyWrite: ((outcome: "saved" | "failed" | "unknown") => void) | null =
    null
  // Throttled checkpoints of the reply so far: a reload keeps what arrived.
  let checkpointsStopped = false
  let checkpointTimer: ReturnType<typeof setTimeout> | null = null
  let lastCheckpointAt = 0
  let pendingCheckpoint: string | null = null
  const stopCheckpoints = () => {
    checkpointsStopped = true
    pendingCheckpoint = null
    if (checkpointTimer) clearTimeout(checkpointTimer)
    checkpointTimer = null
  }
  const flushCheckpoint = () => {
    checkpointTimer = null
    const content = pendingCheckpoint
    pendingCheckpoint = null
    if (checkpointsStopped || content == null || !turn.admission) return
    lastCheckpointAt = Date.now()
    // If the page goes away now, this partial reply was interrupted.
    void saveRecord(
      record(turn, content, "generated_unsaved", { outcome: "interrupted" })
    ).catch(() => undefined)
  }
  const turn: HistorySendTurn = {
    owner,
    capture,
    ...(retryAdmission ? { retryAdmission } : {}),
    ...(owner.kind === "native" ? { historyBranch: Boolean(editResend || branchIntent) } : {}),
    ...(editResend ? { historyEdit: Object.freeze({ version: 1 as const, message_id: editResend.message_id, message_revision: editResend.message_revision }) } : {}),
    currentView: () =>
      editResend
        ? canUpdateView()
          ? view
          : null
        : controller.getCurrent().view,
    validateLease,
    canUpdateView,
    preparationIsCurrent: editPreparationIsCurrent,
    beforeDispatch: async () => {
      if ((editResend || branchIntent) && !editPreparationIsCurrent())
        throw new Error("stale_selection")
      if (owner.kind === "native" && turn.historyEdit)
        await requireNativeHistoryEditAdmission(owner, capture, "messages", {
          signal, validate_lease: editPreparationIsCurrent
        })
      if (
        owner.kind === "native" &&
        turn.historyBranch === false &&
        !turn.retryAdmission &&
        (await nativeHistoryNeedsBranchChoice(owner, capture, "messages", {
          signal,
          validate_lease: editPreparationIsCurrent
        }))
      )
        throw new NativeHistoryBranchChoice(
          () => canUpdateView() && originalOwner.validate_lease?.() !== false
        )

      await saveRecord(record(turn, "", "dispatching"))
      if ((editResend || branchIntent) && !editPreparationIsCurrent())
        throw new Error("stale_selection")
    },
    afterAdmission: async () => {
      if (turn.owner.kind === "native")
        endReplyWrite = beginServerChatWrite(turn.owner.conversation_id)
      await saveRecord(record(turn, "", "accepted_unsent"))
    },
    checkpoint: (content) => {
      if (checkpointsStopped || !turn.admission || !content.trim()) return
      pendingCheckpoint = content
      if (checkpointTimer) return
      const wait = Math.max(
        0,
        HISTORY_TURN_CHECKPOINT_INTERVAL_MS - (Date.now() - lastCheckpointAt)
      )
      checkpointTimer = setTimeout(flushCheckpoint, wait)
    },
    recover: async (data, error) => {
      stopCheckpoints()
      if (
        owner.kind === "native" &&
        !turn.retryAdmission &&
        !turn.admission &&
        !turn.resultId &&
        data.content === "" &&
        turn.selection &&
        ((turn.historyBranch === false && (error instanceof NativeHistoryBranchChoice ||
          isDefinitiveNativeBranchRejection(error, turn.selection))) ||
          (turn.dispatched && turn.historyEdit && isDefinitiveNativeEditRejection(error, turn.selection, turn.historyEdit)))
      ) {
        const refusal =
          error instanceof NativeHistoryBranchChoice
            ? error
            : new (turn.historyEdit ? NativeHistoryEditRefusal : NativeHistoryBranchRefusal)(
                error,
                () => canUpdateView() && originalOwner.validate_lease?.() !== false
              )
        try {
          await saveRecord(record(turn, "", "refused"))
        } catch {
          console.warn("Could not retain a native input that was not admitted.")
        }
        if (refusal.isCurrent()) {
          if (!turn.historyEdit) controller.holdStaleSend?.(current.view!, draftIsCurrent)
          try {
            await controller.refreshRecovery?.()
          } catch {
            console.warn("Could not display retained native input.")
          }
        }
        throw refusal
      }

      if (turn.resultId) {
        // Keep the acknowledged receipt until this view or a reopened view follows it.
        await recoveryWrites
        await saveRecord(record(turn, data.content, "generated_unsaved", {
          outcome: "complete",
          settled_message_id: turn.resultId
        }))
        return
      }
      // A reply the server never received leaves the server copy incomplete.
      endReplyWrite?.(data.content.trim() ? "failed" : "unknown")
      await saveRecord(
        record(
          turn,
          data.content,
          data.content
            ? "generated_unsaved"
            : turn.admission
              ? "accepted_unsent"
              : "unknown",
          data.outcome
            ? {
                outcome: data.outcome,
                ...(data.interruptionReason
                  ? { interruption_reason: data.interruptionReason }
                  : {})
              }
            : {}
        )
      )
      await controller.refreshRecovery?.()
    },
    cancelPreparation: async () => {
      stopCheckpoints()
      await recoveryWrites
      if (lastRecord) {
        try { await saveRecord({ ...lastRecord, state: "refused" }) }
        catch { console.warn("Could not record retained input from cancelled preparation.") }
        if (canUpdateView()) await controller.refreshRecovery?.()
      } else await dismissHistoryTurnRecovery(bookmarkScope, view, operationId)
    },
    complete: async (content) => {
      // The reply is settled. Its record stays until the view follows it.
      stopCheckpoints()
      endReplyWrite?.("saved")
      await recoveryWrites
      await saveRecord(
        record(
          turn,
          content ?? lastRecord?.result_text ?? "",
          "generated_unsaved",
          {
            outcome: "complete",
            settled_message_id: turn.resultId
          }
        )
      )
      if (creationHandle && owner.kind === "native")
        await markChatCreationSettled(creationHandle, owner.conversation_id)
    },
    followResult: async (id) => {
      if (!canUpdateView() || (editResend && !editOriginIsCurrent())) return false
      // Following owns its H1 advancement, while draft publication stays fenced.
      const commandIsCurrent = captureChatDraftSource(authValid).isCurrent
      const followed = await controller.followResult(
        originView,
        id,
        () => commandIsCurrent() && editResend?.isCurrent() !== false
      )
      if (
        followed &&
        commandIsCurrent() &&
        params.historySelection?.sourceIsCurrent?.() !== false
      )
        params.historySelection?.onResultFollow?.()
      return followed
    },
    finish: async (followed) => {
      await recoveryWrites
      if (followed || !turn.resultId) {
        await dismissHistoryTurnRecovery(
          bookmarkScope,
          view,
          operationId,
          "completed"
        )
        return
      }
      // The view went away before it could follow the settled reply (the
      // user left /chat): the next view of this chat follows it instead.
      await saveRecord(
        record(turn, lastRecord?.result_text ?? "", "generated_unsaved", {
          outcome: "complete",
          settled_message_id: turn.resultId
        })
      )
      if (editResend && editOriginIsCurrent())
        await controller.refreshRecovery?.()
    },
    keep: async () => {
      await recoveryWrites
      if (!lastRecord?.outcome) return
      releaseHistoryTurn(operationId)
      await keepHistoryTurnRecovery(
        controller,
        {
          scope: bookmarkScope,
          turn: lastRecord
        },
        stopSignal
      )
    },
    release: () => {
      stopCheckpoints()
      endReplyWrite?.("unknown")
      releaseHistoryTurn(operationId)
    }
  }
  return turn
}

/** Reuse the accepted H1 bind and operation-lived settlement for each real mode. */
export const runChatModeWithHistory = async <TParams extends ChatModeParamsBase & HistoryChatModeParams>(
  mode: ChatModeDefinition<TParams>,
  message: string, image: string, isRegenerate: boolean,
  messages: Message[], history: ChatHistory, signal: AbortSignal, params: TParams
): Promise<ChatSubmitResult> => {
  const preparation = params.historySelection
    ? params.historyPreparation ?? captureNormalChatPreparation(params)
    : undefined
  const servicePromptSnapshot = params.servicePromptSnapshot
  const executionSignal = servicePromptSnapshot
    ? servicePromptSnapshot.scopeSignal
    : signal
  let createdTurn: HistorySendTurn | undefined
  const settlement = params.historySelection
    ? beginChatTurnSettlement(signal)
    : undefined
  try {
    if (servicePromptSnapshot?.scopeInvalidatedSignal.aborted || executionSignal.aborted)
      return chatSubmitFailed("Request cancelled")
    if (params.historySelection) {
      if (isRegenerate) throw new Error("unsupported_history_action")
      const originalOwner = params.historySelection.controller.getCurrent().owner
      if (originalOwner?.kind === "native" && mode.id !== "normal")
        throw new Error("unsupported_history_action_context")
      if (params.uploadedFiles?.length && mode.id === "normal")
        throw new Error("unsupported_history_message_assets")
      if (!servicePromptSnapshot) throw new Error("history_scope_unavailable")
      const historyTurn = await captureNormalHistoryTurn(
        { ...params, historyPreparation: preparation },
        message,
        servicePromptSnapshot!,
        executionSignal,
        preparation!.dispatchIsCurrent,
        () => params.historySelection?.sourceIsCurrent?.() !== false && preparation!.draftIsCurrent(),
        preparation!.acceptDraftBind,
        preparation!.sourceDraft,
        signal
      )
      createdTurn = historyTurn
      // Canonical roles/content come from the captured owner, not display normalization.
      const displayMessages = formatSelectedHistory(historyTurn.capture).messages
      const inferenceIds = historyTurn.owner.kind === "local"
        ? new Set(excludeLocalRagDiagnostics(displayMessages).map(row => row.id))
        : undefined
      const selected = historyTurn.capture.rows.map((node, index) => {
        const content = historyTurn.capture.selected_content[index]
        const metadata = content.extra_metadata ?? {}
        const local = (
          historyTurn.owner.kind === "local"
            ? metadata.local_history
            : undefined
        ) as Record<string, unknown> | undefined

        return {
          id: node.id,
          role: node.role,
          content: content.message,
          images: [...content.images],
          tool_calls: content.tool_calls ? [...content.tool_calls] : undefined,
          tool_call_id: metadata.tool_call_id as string | undefined,
          function_call: metadata.function_call as
            | Record<string, unknown>
            | undefined,
          messageType: local?.messageType as string | undefined
        }
      }).filter(row => !inferenceIds || inferenceIds.has(row.id))
      generateHistory(selected, params.selectedModel, { versioned: true })
      if (
        historyTurn.owner.kind === "native" &&
        (params.webSearch || params.dynamicUIRequest)
      )
        throw new Error("unsupported_history_native_result_metadata")
      const selectedParent = historyTurn.capture.rows.at(-1)?.id ?? null
      if (
        params.userParentMessageId &&
        params.userParentMessageId !== selectedParent
      )
        throw new Error("unsupported_history_reply_target")
      messages = displayMessages
      params = {
        ...params,
        historyTurn,
        ...(params.historySelection?.editResend
          ? { setMessages: () => {}, setHistory: () => {} }
          : {}),
        // A retry shows and settles under the question it reasks.
        ...(historyTurn.retryAdmission
          ? { userMessageId: historyTurn.retryAdmission.input_message_id }
          : {}),
        userParentMessageId: selectedParent,
        historyForModel: selected as ChatHistory,
        historyId:
          historyTurn.owner.kind === "local"
            ? historyTurn.owner.conversation_id
            : params.historyId
      }
      history = selected as ChatHistory
    }
    const resolvedImage =
      image.length > 0
        ? (image.startsWith("data:") ? createImageDataUrl(image) : null) ??
          `data:image/jpeg;base64,${image.includes(",") ? image.split(",")[1] : image}`
        : ""

    const result = await runChatPipeline(
      mode,
      message,
      resolvedImage,
      isRegenerate,
      messages,
      history,
      executionSignal,
      {
        ...params,
        servicePromptSnapshot,
        discardCurrentTurnOnAbort: servicePromptSnapshot
          ? () => servicePromptSnapshot.scopeInvalidatedSignal.aborted
          : undefined,
        ownsAbortController: params.ownsAbortController
          ? () => params.ownsAbortController!(signal)
          : undefined,
        wasStoppedByUser: () => wasChatTurnStoppedByUser(signal),
        releaseAbortControllerIfOwned: params.releaseAbortControllerIfOwned
          ? () => params.releaseAbortControllerIfOwned!(signal)
          : undefined
      }
    )
    const historyTurn = params.historyTurn
    if (result.status === "submitted" && historyTurn?.resultId) {
      const followed = await historyTurn.followResult(historyTurn.resultId)
      await historyTurn.finish?.(followed !== false)
    } else if (historyTurn?.outcome && !params.historySelection?.editResend) {
      // Partial replies stay in the transcript; unknown save receipts stay in review.
      await historyTurn.keep?.()
    }
    return result
  } finally {
    createdTurn?.release?.()
    settlement?.finish()
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
  const preparation = params.historySelection
    ? params.historyPreparation ?? captureNormalChatPreparation(params)
    : undefined
  const ownsServicePromptSnapshot =
    !params.servicePromptSnapshot &&
    (params.webSearch || !!params.historySelection || !!params.actorSceneSend)
  const servicePromptSnapshot =
    params.servicePromptSnapshot ??
    (params.webSearch || params.historySelection || params.actorSceneSend
      ? await loadServicePromptSnapshot(
          params.webSearch ? ["chat.web_search.answer"] : [],
          {
            signal,
            requestScope:
              params.actorSceneSend?.requestScope ?? params.actorScene?.target.account?.requestScope ??
              (params.historySelection?.controller.getCurrent().owner?.kind ===
              "native"
                ? (
                    params.historySelection.controller.getCurrent()
                      .owner as import("@/services/chat-history-selection").NativeHistoryOwnerV1
                  ).request_scope
                : undefined)
          }
        )
      : undefined)
  try {
    if (params.webSearch) getRequiredServicePrompt(servicePromptSnapshot, "chat.web_search.answer")
    return await runChatModeWithHistory(normalChatModeDefinition, message, image,
      isRegenerate, messages, history, signal,
      { ...params, historyPreparation: preparation, servicePromptSnapshot })
  } finally {
    if (ownsServicePromptSnapshot && servicePromptSnapshot) servicePromptSnapshot.release()
  }
}
