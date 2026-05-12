import { startTransition } from "react"
import { generateID } from "@/db/dexie/helpers"
import { getModelNicknameByID } from "@/db/dexie/nickname"
import { isReasoningEnded, isReasoningStarted } from "@/libs/reasoning"
import { pageAssistModel } from "@/models"
import type { ActorSettings } from "@/types/actor"
import type { ChatDocuments } from "@/models/ChatTypes"
import type { SaveMessageData, SaveMessageErrorData } from "@/types/chat-modes"
import { buildAssistantErrorContent } from "@/utils/chat-error-message"
import { applyMcpModuleDisclosureFromToolCalls } from "@/utils/mcp-disclosure"
import {
  buildMessageVariant,
  getLastUserMessageId,
  normalizeMessageVariants,
  applyVariantToMessage,
  type MessageVariant,
  updateActiveVariant
} from "@/utils/message-variants"
import {
  consumeStreamingChunk,
  type StreamingChunk
} from "@/utils/streaming-chunks"
import { parseProviderQualifiedModelSelection } from "@/utils/resolve-api-provider"
import { buildMessageSteeringSnippet } from "@/utils/message-steering"
import { useStoreMessageOption } from "@/store/option"
import type { ChatHistory, Message, ToolChoice } from "~/store/option"
import type { ToolCall } from "@/types/tool-calls"
import type { ChatResearchContext } from "@/services/tldw/TldwApiClient"
import type {
  MessageSteeringFlags,
  MessageSteeringPromptTemplates
} from "@/types/message-steering"
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  IMAGE_GENERATION_USER_MESSAGE_TYPE,
  normalizeImageGenerationVariantBundle,
  type ImageGenerationEventSyncPolicy
} from "@/utils/image-generation-chat"
import {
  chatSubmitFailed,
  chatSubmitSkipped,
  chatSubmitSubmitted,
  type ChatSubmitResult
} from "@/hooks/chat/chat-action-utils"
import { isAbortLikeError } from "@/hooks/chat/abort-turn-cleanup"

const STREAMING_UPDATE_INTERVAL_MS = 80
let didLogPipelineSetHistoryMissing = false

export type ChatModeParamsBase = {
  selectedModel: string
  useOCR: boolean
  toolChoice?: ToolChoice
  conversationId?: string
  setMessages: (messages: Message[] | ((prev: Message[]) => Message[])) => void
  saveMessageOnSuccess: (data: SaveMessageData) => Promise<string | null>
  saveMessageOnError: (data: SaveMessageErrorData) => Promise<string | null>
  setHistory: (history: ChatHistory) => void
  setIsProcessing: (value: boolean) => void
  setStreaming: (value: boolean) => void
  setAbortController: (controller: AbortController | null) => void
  historyId: string | null
  setHistoryId: (id: string) => void
  actorSettings?: ActorSettings
  documents?: ChatDocuments
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
  messageSteering?: MessageSteeringFlags
  messageSteeringPrompts?: MessageSteeringPromptTemplates
  imageEventSyncPolicy?: ImageGenerationEventSyncPolicy
  researchContext?: ChatResearchContext
}

export type ChatModeContext<TParams extends ChatModeParamsBase> = TParams & {
  message: string
  image: string
  isRegenerate: boolean
  messages: Message[]
  history: ChatHistory
  signal: AbortSignal
  createdAt: number
  generateMessageId: string
  resolvedUserMessageId?: string
  resolvedAssistantMessageId: string
  resolvedAssistantParentMessageId: string | null
  resolvedModelId: string
  userModelId?: string
  modelInfo: { model_name: string; model_avatar?: string } | null
  regenerateVariants: MessageVariant[]
}

export type ChatModePrompt = {
  chatHistory: ChatHistory
  humanMessage?: any
  sources?: unknown[]
  promptId?: string
  promptContent?: string
}

export type ChatModePreflightResult = {
  handled: true
  fullText: string
  sources?: unknown[]
  images?: string[]
  generationInfo?: unknown
  promptId?: string
  promptContent?: string
  saveToDb?: boolean
  conversationId?: string
  skipHistoryAppend?: boolean
}

export type ChatModeMessageSetup = {
  targetMessageId: string
  initialFullText?: string
}

export type ChatModeDefinition<TParams extends ChatModeParamsBase> = {
  id: string
  buildUserMessage?: (ctx: ChatModeContext<TParams>) => Message
  buildAssistantMessage?: (ctx: ChatModeContext<TParams>) => Message
  setupMessages?: (ctx: ChatModeContext<TParams>) => ChatModeMessageSetup
  preparePrompt: (ctx: ChatModeContext<TParams>) => Promise<ChatModePrompt>
  preflight?: (
    ctx: ChatModeContext<TParams>
  ) => Promise<ChatModePreflightResult | null>
  updateHistory?: (ctx: ChatModeContext<TParams>, fullText: string) => ChatHistory
  isContinue?: boolean
  extractGenerationInfo?: (output: unknown) => unknown
}

const defaultExtractGenerationInfo = (output: any) =>
  output?.generations?.[0][0]?.generationInfo

const extractToolCalls = (generationInfo: unknown): ToolCall[] | undefined => {
  if (!generationInfo || typeof generationInfo !== "object") return undefined
  const candidate =
    (generationInfo as any).tool_calls ?? (generationInfo as any).toolCalls
  return Array.isArray(candidate) ? (candidate as ToolCall[]) : undefined
}

export const runChatPipeline = async <TParams extends ChatModeParamsBase>(
  mode: ChatModeDefinition<TParams>,
  message: string,
  image: string,
  isRegenerate: boolean,
  messages: Message[],
  history: ChatHistory,
  signal: AbortSignal,
  params: TParams
): Promise<ChatSubmitResult> => {
  const {
    selectedModel: rawSelectedModel,
    toolChoice,
    setMessages,
    saveMessageOnSuccess,
    saveMessageOnError,
    setHistory,
    setIsProcessing,
    setStreaming,
    setAbortController,
    historyId,
    setHistoryId,
    clusterId,
    userMessageType,
    assistantMessageType,
    modelIdOverride: rawModelIdOverride,
    userMessageId,
    assistantMessageId,
    userParentMessageId,
    assistantParentMessageId,
    documents,
    regenerateFromMessage,
    imageEventSyncPolicy,
    conversationId
  } = params
  const selectedModelSelection =
    parseProviderQualifiedModelSelection(rawSelectedModel)
  const selectedModel =
    selectedModelSelection.modelId || rawSelectedModel
  const modelIdOverride = rawModelIdOverride
    ? parseProviderQualifiedModelSelection(rawModelIdOverride).modelId
    : undefined

  const resolvedAssistantMessageId = assistantMessageId ?? generateID()
  const resolvedUserMessageId =
    !isRegenerate ? userMessageId ?? generateID() : undefined
  const createdAt = Date.now()
  let generateMessageId = resolvedAssistantMessageId
  const modelInfo = await getModelNicknameByID(selectedModel)

  const isSharedCompareUser = userMessageType === "compare:user"
  const isImageGenerationTurn =
    userMessageType === IMAGE_GENERATION_USER_MESSAGE_TYPE ||
    assistantMessageType === IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
  const resolvedModelId = modelIdOverride || selectedModel
  const userModelId = isSharedCompareUser ? undefined : resolvedModelId
  const fallbackParentMessageId = getLastUserMessageId(messages)
  const resolvedAssistantParentMessageId =
    assistantParentMessageId ??
    (isRegenerate
      ? regenerateFromMessage?.parentMessageId ?? fallbackParentMessageId
      : resolvedUserMessageId ?? null)
  const regenerateVariants =
    isRegenerate && regenerateFromMessage
      ? normalizeMessageVariants(regenerateFromMessage)
      : []

  const context: ChatModeContext<TParams> = {
    ...params,
    selectedModel,
    modelIdOverride,
    message,
    image,
    isRegenerate,
    messages,
    history,
    signal,
    createdAt,
    generateMessageId,
    resolvedUserMessageId,
    resolvedAssistantMessageId,
    resolvedAssistantParentMessageId,
    resolvedModelId,
    userModelId,
    modelInfo,
    regenerateVariants
  }

  let fullText = ""
  let contentToSave = ""
  let timetaken = 0
  let promptContent: string | undefined = undefined
  let promptId: string | undefined = undefined
  let streamingTimer: ReturnType<typeof setTimeout> | null = null
  let lastStreamingUpdateAt = 0
  let pendingStreamingText = ""
  let pendingReasoningTime = 0
  const setMessagesWithTransition = (
    messagesOrUpdater: Message[] | ((prev: Message[]) => Message[])
  ) => {
    startTransition(() => {
      setMessages(messagesOrUpdater)
    })
  }
  const setHistorySafely = (nextHistory: ChatHistory) => {
    if (typeof setHistory === "function") {
      setHistory(nextHistory)
      return
    }
    const fallback = useStoreMessageOption.getState().setHistory
    if (typeof fallback === "function") {
      fallback(nextHistory)
      return
    }
    if (!didLogPipelineSetHistoryMissing) {
      didLogPipelineSetHistoryMissing = true
      console.error("[chat] runChatPipeline could not resolve setHistory setter", {
        setHistoryType: typeof setHistory
      })
    }
  }

  const normalizeImageVariantsForMessage = (messageEntry: Message): Message => {
    if (!isImageGenerationTurn) {
      return messageEntry
    }

    const variants = Array.isArray(messageEntry.variants)
      ? (messageEntry.variants as MessageVariant[])
      : []
    const normalized = normalizeImageGenerationVariantBundle({
      messageId: messageEntry.id,
      messageGenerationInfo: messageEntry.generationInfo,
      variants,
      activeVariantIndex:
        typeof messageEntry.activeVariantIndex === "number"
          ? messageEntry.activeVariantIndex
          : undefined,
      fallbackCreatedAt: createdAt,
      hasVisibleVariant:
        variants.length > 0 ||
        (Array.isArray(messageEntry.images) && messageEntry.images.length > 0)
    })

    if (variants.length === 0) {
      return {
        ...messageEntry,
        generationInfo: normalized.generationInfo ?? messageEntry.generationInfo
      }
    }

    const activeVariant = normalized.variants[
      normalized.activeVariantIndex
    ] as MessageVariant | undefined
    const withNormalizedVariants: Message = {
      ...messageEntry,
      variants: normalized.variants as MessageVariant[],
      activeVariantIndex: normalized.activeVariantIndex,
      generationInfo: normalized.generationInfo ?? messageEntry.generationInfo
    }
    if (!activeVariant) {
      return withNormalizedVariants
    }

    return applyVariantToMessage(
      withNormalizedVariants,
      activeVariant,
      normalized.activeVariantIndex
    )
  }

  const flushStreamingUpdate = () => {
    streamingTimer = null
    lastStreamingUpdateAt = Date.now()
    setMessagesWithTransition((prev) =>
      prev.map((msg) =>
        msg.id === generateMessageId
          ? updateActiveVariant(msg, {
              message: pendingStreamingText,
              reasoning_time_taken: pendingReasoningTime
            })
          : msg
      )
    )
  }

  // Throttle streaming UI updates to keep the input responsive.
  const scheduleStreamingUpdate = (text: string, reasoningTime: number) => {
    pendingStreamingText = text
    pendingReasoningTime = reasoningTime
    if (streamingTimer != null) return
    const now = Date.now()
    const elapsed = now - lastStreamingUpdateAt
    const delay = Math.max(0, STREAMING_UPDATE_INTERVAL_MS - elapsed)
    streamingTimer = setTimeout(flushStreamingUpdate, delay)
  }

  const cancelStreamingUpdate = () => {
    if (streamingTimer == null) return
    clearTimeout(streamingTimer)
    streamingTimer = null
  }

  const abortCancelStreamingUpdate = () => cancelStreamingUpdate()
  signal.addEventListener("abort", abortCancelStreamingUpdate)

  if (mode.setupMessages) {
    const setup = mode.setupMessages(context)
    generateMessageId = setup.targetMessageId
    context.generateMessageId = generateMessageId
    context.resolvedAssistantMessageId = generateMessageId
    if (typeof setup.initialFullText === "string") {
      fullText = setup.initialFullText
      contentToSave = setup.initialFullText
    }
  } else {
    if (!mode.buildAssistantMessage || !mode.buildUserMessage) {
      throw new Error(`Chat mode "${mode.id}" is missing message builders.`)
    }
    setMessagesWithTransition((prev) => {
      const assistantStub = mode.buildAssistantMessage!(context)
      if (!isRegenerate) {
        const userMessageEntry = mode.buildUserMessage!(context)
        return [...prev, userMessageEntry, assistantStub]
      }
      return [...prev, assistantStub]
    })

    if (regenerateVariants.length > 0) {
      setMessagesWithTransition((prev) => {
        const next = [...prev]
        const lastIndex = next.findLastIndex(
          (msg) => msg.id === resolvedAssistantMessageId
        )
        if (lastIndex >= 0) {
          const stub = next[lastIndex]
          const variants = [
            ...regenerateVariants,
            buildMessageVariant(stub)
          ]
          next[lastIndex] = normalizeImageVariantsForMessage({
            ...stub,
            variants,
            activeVariantIndex: variants.length - 1
          })
        }
        return next
      })
    }
  }

  try {
    const preflight = mode.preflight ? await mode.preflight(context) : null
    if (preflight?.handled) {
      fullText = preflight.fullText
      const sources = preflight.sources ?? []
      const images = preflight.images ?? []
      const toolCalls = extractToolCalls(preflight.generationInfo)
      applyMcpModuleDisclosureFromToolCalls(toolCalls)
      const nextHistory = mode.updateHistory
        ? mode.updateHistory(context, fullText)
        : preflight.skipHistoryAppend
          ? history
        : ([
            ...history,
            { role: "user", content: message, image },
            { role: "assistant", content: fullText }
          ] as ChatHistory)
      const preflightGenerationInfoForSave = (() => {
        if (!isImageGenerationTurn) {
          return preflight.generationInfo
        }

        const normalizedBundle = normalizeImageGenerationVariantBundle({
          messageId: resolvedAssistantMessageId,
          messageGenerationInfo: preflight.generationInfo,
          variants:
            regenerateVariants.length > 0
              ? [
                  ...regenerateVariants,
                  {
                    id: resolvedAssistantMessageId,
                    generationInfo: preflight.generationInfo
                  }
                ]
              : [],
          activeVariantIndex:
            regenerateVariants.length > 0 ? regenerateVariants.length : 0,
          fallbackCreatedAt: createdAt,
          hasVisibleVariant: images.length > 0
        })
        return normalizedBundle.generationInfo ?? preflight.generationInfo
      })()

      setMessagesWithTransition((prev) =>
        prev.map((msg) =>
          msg.id === generateMessageId
            ? normalizeImageVariantsForMessage(
                updateActiveVariant(msg, {
                  message: fullText,
                  sources,
                  images,
                  generationInfo: preflightGenerationInfoForSave,
                  toolCalls,
                  reasoning_time_taken: timetaken
                })
              )
            : msg
        )
      )
      setHistorySafely(nextHistory)

      await saveMessageOnSuccess({
        historyId,
        setHistoryId,
        isRegenerate,
        selectedModel,
        message,
        image,
        fullText,
        source: sources,
        assistantImages: images,
        userMessageType,
        assistantMessageType,
        clusterId,
        modelId: resolvedModelId,
        userModelId,
        userMessageId: resolvedUserMessageId,
        assistantMessageId: resolvedAssistantMessageId,
        userParentMessageId: userParentMessageId ?? null,
        assistantParentMessageId: resolvedAssistantParentMessageId ?? null,
        documents,
        isContinue: mode.isContinue,
        generationInfo: preflightGenerationInfoForSave as any,
        prompt_content: preflight.promptContent,
        prompt_id: preflight.promptId,
        reasoning_time_taken: timetaken,
        saveToDb: preflight.saveToDb ?? false,
        conversationId: preflight.conversationId,
        imageEventSyncPolicy
      })
      return chatSubmitSubmitted()
    }

    const promptData = await mode.preparePrompt(context)
    const steeringSnippet = buildMessageSteeringSnippet(
      context.messageSteering || {
        continueAsUser: false,
        impersonateUser: false,
        forceNarrate: false
      },
      context.messageSteeringPrompts
    )
    if (steeringSnippet) {
      promptData.chatHistory = [
        ...promptData.chatHistory,
        { role: "system", content: steeringSnippet }
      ]
    }
    promptContent = promptData.promptContent
    promptId = promptData.promptId
    const sources = promptData.sources ?? []
    const humanMessage = promptData.humanMessage

    const modelClient = await pageAssistModel({
      model: selectedModel,
      toolChoice,
      conversationId,
      researchContext: context.researchContext
    })

    let generationInfo: unknown = undefined
    const chunks = await modelClient.stream(
      humanMessage
        ? [...promptData.chatHistory, humanMessage]
        : [...promptData.chatHistory],
      {
        signal,
        callbacks: [
          {
            handleLLMEnd(output: unknown): void {
              const extractor =
                mode.extractGenerationInfo ?? defaultExtractGenerationInfo
              generationInfo = extractor(output)
            }
          }
        ]
      }
    )

    let count = 0
    let reasoningStartTime: Date | null = null
    let reasoningEndTime: Date | null = null
    let apiReasoning = false
    let streamTransportInterrupted = false
    let streamTransportInterruptionReason: string | null = null

    for await (const chunk of chunks) {
      if (signal.aborted) break

      const interruptionEvent =
        chunk && typeof chunk === "object" && !Array.isArray(chunk)
          ? (chunk as Record<string, unknown>)
          : null
      if (
        typeof interruptionEvent?.event === "string" &&
        interruptionEvent.event.toLowerCase() ===
          "stream_transport_interrupted"
      ) {
        streamTransportInterrupted = true
        const detail =
          typeof interruptionEvent.detail === "string"
            ? interruptionEvent.detail.trim()
            : ""
        streamTransportInterruptionReason =
          detail.length > 0 ? detail : streamTransportInterruptionReason
        continue
      }

      const chunkState = consumeStreamingChunk(
        { fullText, contentToSave, apiReasoning },
        chunk as StreamingChunk
      )
      fullText = chunkState.fullText
      contentToSave = chunkState.contentToSave
      apiReasoning = chunkState.apiReasoning

      if (isReasoningStarted(fullText) && !reasoningStartTime) {
        reasoningStartTime = new Date()
      }

      if (
        reasoningStartTime &&
        !reasoningEndTime &&
        isReasoningEnded(fullText)
      ) {
        reasoningEndTime = new Date()
        const reasoningTime =
          reasoningEndTime.getTime() - reasoningStartTime.getTime()
        timetaken = reasoningTime
      }

      if (count === 0) {
        setIsProcessing(true)
      }

      scheduleStreamingUpdate(`${fullText}▋`, timetaken)
      count++
    }

    cancelStreamingUpdate()
    signal.removeEventListener("abort", abortCancelStreamingUpdate)
    const toolCalls = extractToolCalls(generationInfo)
    applyMcpModuleDisclosureFromToolCalls(toolCalls)
    const finalGenerationInfo = streamTransportInterrupted
      ? {
          ...(generationInfo && typeof generationInfo === "object" && !Array.isArray(generationInfo)
            ? generationInfo
            : {}),
          interrupted: true,
          partialResponseSaved: true,
          streamTransportInterrupted: true,
          streamTransportInterruptionReason:
            streamTransportInterruptionReason ||
            "Stream transport interrupted; partial response saved."
        }
      : generationInfo
    setMessagesWithTransition((prev) =>
      prev.map((msg) =>
        msg.id === generateMessageId
          ? normalizeImageVariantsForMessage(
              updateActiveVariant(msg, {
                message: fullText,
                sources,
                generationInfo: finalGenerationInfo,
                toolCalls,
                reasoning_time_taken: timetaken
              })
            )
          : msg
      )
    )

    setHistorySafely(
      mode.updateHistory
        ? mode.updateHistory(context, fullText)
        : ([
            ...history,
            { role: "user", content: message, image },
            { role: "assistant", content: fullText }
          ] as ChatHistory)
    )

    await saveMessageOnSuccess({
      historyId,
      setHistoryId,
      isRegenerate,
      selectedModel,
      message,
      image,
      fullText,
      source: sources,
      userMessageType,
      assistantMessageType,
      clusterId,
      modelId: resolvedModelId,
      userModelId,
      userMessageId: resolvedUserMessageId,
      assistantMessageId: resolvedAssistantMessageId,
      userParentMessageId: userParentMessageId ?? null,
      assistantParentMessageId: resolvedAssistantParentMessageId ?? null,
      documents,
      isContinue: mode.isContinue,
      generationInfo: finalGenerationInfo as any,
      prompt_content: promptContent,
      prompt_id: promptId,
      reasoning_time_taken: timetaken,
      saveToDb: Boolean(modelClient.saveToDb),
      conversationId: modelClient.conversationId,
      imageEventSyncPolicy
    })
    return chatSubmitSubmitted()
  } catch (e) {
    cancelStreamingUpdate()
    signal.removeEventListener("abort", abortCancelStreamingUpdate)
    const isAbort = signal.aborted || isAbortLikeError(e)
    const assistantContent = isAbort
      ? fullText
      : buildAssistantErrorContent(fullText, e)
    const interruptionReason = isAbort
      ? "Request cancelled"
      : e instanceof Error && e.message.trim().length > 0
        ? e.message
        : "Something went wrong."
    setMessagesWithTransition((prev) =>
      prev.map((msg) =>
        msg.id === generateMessageId
          ? normalizeImageVariantsForMessage(
              updateActiveVariant(msg, {
                message: assistantContent,
                generationInfo: {
                  ...(msg.generationInfo || {}),
                  interrupted: true,
                  interruptionReason,
                  interruptedAt: Date.now()
                }
              })
            )
          : msg
      )
    )

    const errorSave = await saveMessageOnError({
      e,
      botMessage: assistantContent,
      history,
      historyId,
      image,
      selectedModel,
      setHistory: setHistorySafely,
      setHistoryId,
      userMessage: message,
      isRegenerating: isRegenerate,
      userMessageType,
      assistantMessageType,
      clusterId,
      modelId: resolvedModelId,
      userModelId,
      userMessageId: resolvedUserMessageId,
      assistantMessageId: resolvedAssistantMessageId,
      userParentMessageId: userParentMessageId ?? null,
      assistantParentMessageId: assistantParentMessageId ?? null,
      documents,
      isContinue: mode.isContinue,
      prompt_content: promptContent,
      prompt_id: promptId
    })

    if (isAbort) {
      return chatSubmitSkipped(interruptionReason)
    }

    if (!errorSave) {
      throw e
    }
    return chatSubmitFailed(interruptionReason)
  } finally {
    signal.removeEventListener("abort", abortCancelStreamingUpdate)
    setIsProcessing(false)
    setStreaming(false)
    setAbortController(null)
  }
}
