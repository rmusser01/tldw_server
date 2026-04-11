import React from "react"
import type { NotificationInstance } from "antd/es/notification/interface"
import type { TFunction } from "i18next"
import {
  generateID,
  saveHistory,
  saveMessage,
  updateHistory,
  updateMessage,
  updateMessageMedia,
  removeMessageByIndex,
  formatToChatHistory,
  formatToMessage,
  getSessionFiles,
  getPromptById
} from "@/db/dexie/helpers"
import { getModelNicknameByID } from "@/db/dexie/nickname"
import { isReasoningEnded, isReasoningStarted } from "@/libs/reasoning"
import type { ChatDocuments } from "@/models/ChatTypes"
import { normalChatMode } from "@/hooks/chat-modes/normalChatMode"
import { continueChatMode } from "@/hooks/chat-modes/continueChatMode"
import { ragMode } from "@/hooks/chat-modes/ragMode"
import { tabChatMode } from "@/hooks/chat-modes/tabChatMode"
import { documentChatMode } from "@/hooks/chat-modes/documentChatMode"
import {
  validateBeforeSubmit,
  createSaveMessageOnSuccess,
  createSaveMessageOnError
} from "@/hooks/utils/messageHelpers"
import {
  createRegenerateLastMessage,
  createEditMessage,
  createBranchMessage
} from "@/hooks/handlers/messageHandlers"
import { generateBranchFromMessageIds } from "@/db/dexie/branch"
import { type UploadedFile } from "@/db/dexie/types"
import { buildAssistantErrorContent } from "@/utils/chat-error-message"
import { detectCharacterMood } from "@/utils/character-mood"
import {
  buildMessageVariant,
  getLastUserMessageId,
  normalizeMessageVariants,
  updateActiveVariant
} from "@/utils/message-variants"
import { resolveImageBackendCandidates } from "@/utils/image-backends"
import {
  buildImageGenerationEventMirrorContent,
  isImageGenerationMessageType,
  PLAYGROUND_IMAGE_EVENT_SYNC_DEFAULT_STORAGE_KEY,
  resolveImageGenerationEventSyncMode,
  normalizeImageGenerationEventSyncMode,
  normalizeImageGenerationEventSyncPolicy,
  type ImageGenerationEventSyncPolicy,
  type ImageGenerationEventSyncMode,
  type ImageGenerationRefineMetadata,
  type ImageGenerationPromptMode,
  type ImageGenerationRequestSnapshot
} from "@/utils/image-generation-chat"
import {
  resolveApiProviderForModel,
  resolveExplicitProviderForSelectedModel
} from "@/utils/resolve-api-provider"
import { consumeStreamingChunk } from "@/utils/streaming-chunks"
import { updatePageTitle } from "@/utils/update-page-title"
import { normalizeConversationState } from "@/utils/conversation-state"
import {
  DEFAULT_MESSAGE_STEERING_PROMPTS,
  hasActiveMessageSteering,
  normalizeMessageSteeringPrompts,
  resolveMessageSteering,
  toMessageSteeringPromptPayload
} from "@/utils/message-steering"
import {
  SELECTED_CHARACTER_STORAGE_KEY,
  selectedCharacterStorage,
  selectedCharacterSyncStorage,
  parseSelectedCharacterValue
} from "@/utils/selected-character-storage"
import {
  tldwClient,
  type ChatResearchContext,
  type ConversationState
} from "@/services/tldw/TldwApiClient"
import { getServerCapabilities } from "@/services/tldw/server-capabilities"
import { generateTitle } from "@/services/title"
import { trackCompareMetric } from "@/utils/compare-metrics"
import { MAX_COMPARE_MODELS } from "@/hooks/chat/compare-constants"
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"
import {
  discardAbortedTurnIfRequested,
  isAbortLikeError
} from "@/hooks/chat/abort-turn-cleanup"
import { resolveSavedDegradedCharacterPersist } from "@/hooks/chat/characterPersistOutcome"
import { ensurePersonaServerChat } from "@/hooks/chat/personaServerChat"
import {
  isPersonaAssistantSelection,
  type AssistantSelection
} from "@/types/assistant-selection"
import type { Character } from "@/types/character"
import type {
  MessageSteeringMode,
  MessageSteeringPromptTemplates,
  MessageSteeringState
} from "@/types/message-steering"
import {
  type ChatHistory,
  type Message,
  useStoreMessageOption,
  type Knowledge,
  type ReplyTarget,
  type ToolChoice
} from "@/store/option"
import type { ChatModelSettings } from "@/store/model"
import type { SaveMessageData } from "@/types/chat-modes"
import {
  buildGreetingOptionsFromEntries,
  collectGreetingEntries,
  collectGreetings,
  isGreetingMessageType,
  resolveGreetingSelection
} from "@/utils/character-greetings"
import { useStorage } from "@plasmohq/storage/hook"
import {
  PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY,
  resolveOutputFormattingGuideSuffix
} from "@/utils/output-formatting-guide"

type ChatModelSettingsStore = ChatModelSettings & {
  setSystemPrompt?: (prompt: string) => void
}

type ChatModeOverrides = {
  historyId?: string | null
  serverChatId?: string | null
  selectedModel?: string | null
  selectedSystemPrompt?: string | null
  toolChoice?: ToolChoice | null
  useOCR?: boolean
  webSearch?: boolean
  imageEventSyncPolicy?: ImageGenerationEventSyncPolicy
  researchContext?: ChatResearchContext
} & Record<string, unknown>

const loadActorSettings = () => import("@/services/actor-settings")
const STREAMING_UPDATE_INTERVAL_MS = 80

type SaveMessagePayload = Omit<SaveMessageData, "setHistoryId"> & {
  setHistoryId?: SaveMessageData["setHistoryId"]
  conversationId?: string | number | null
  message_source?: "copilot" | "web-ui" | "server" | "branch"
  message_type?: string
  serverMessagesAlreadyPersisted?: boolean
}

type TldwChatMeta =
  | {
      id?: string | number
      chat_id?: string | number
      version?: number
      state?: string | null
      conversation_state?: string | null
      topic_label?: string | null
      cluster_id?: string | null
      source?: string | null
      external_ref?: string | null
      title?: string | null
      character_id?: string | number | null
      assistant_kind?: "character" | "persona" | null
      assistant_id?: string | number | null
      persona_memory_mode?: "read_only" | "read_write" | null
    }
  | string
  | number
  | null
  | undefined

const attemptCharacterStreamRecoveryPersist = async ({
  chatId,
  temporaryChat,
  assistantContent,
  alreadyPersisted,
  error,
  persist
}: {
  chatId: string | null
  temporaryChat: boolean
  assistantContent: string
  alreadyPersisted: boolean
  error: unknown
  persist: (content: string) => Promise<boolean>
}): Promise<boolean> => {
  if (alreadyPersisted || temporaryChat) return false
  if (!chatId || isAbortLikeError(error)) return false
  const trimmedContent = assistantContent.trim()
  if (!trimmedContent) return false
  try {
    return await persist(trimmedContent)
  } catch {
    return false
  }
}

type UseChatActionsOptions = {
  t: TFunction
  notification: NotificationInstance
  abortController: AbortController | null
  setAbortController: (controller: AbortController | null) => void
  messages: Message[]
  setMessages: (
    messagesOrUpdater: Message[] | ((prev: Message[]) => Message[])
  ) => void
  history: ChatHistory
  setHistory: (
    historyOrUpdater: ChatHistory | ((prev: ChatHistory) => ChatHistory)
  ) => void
  historyId: string | null
  setHistoryId: (
    historyId: string | null,
    options?: { preserveServerChatId?: boolean }
  ) => void
  temporaryChat: boolean
  selectedModel: string | null
  useOCR: boolean
  selectedSystemPrompt: string | null
  selectedKnowledge: Knowledge | null
  toolChoice: ToolChoice
  webSearch: boolean
  currentChatModelSettings: ChatModelSettingsStore
  setIsSearchingInternet: (isSearchingInternet: boolean) => void
  setIsProcessing: (isProcessing: boolean) => void
  setStreaming: (streaming: boolean) => void
  setActionInfo: (actionInfo: string) => void
  fileRetrievalEnabled: boolean
  ragMediaIds: number[] | null
  ragSearchMode: "hybrid" | "vector" | "fts"
  ragTopK: number | null
  ragEnableGeneration: boolean
  ragEnableCitations: boolean
  ragSources: string[]
  ragAdvancedOptions: Record<string, unknown>
  serverChatId: string | null
  serverChatTitle: string | null
  serverChatCharacterId: string | number | null
  serverChatAssistantKind: "character" | "persona" | null
  serverChatAssistantId: string | null
  serverChatPersonaMemoryMode: "read_only" | "read_write" | null
  serverChatState: ConversationState | null
  serverChatTopic: string | null
  serverChatClusterId: string | null
  serverChatSource: string | null
  serverChatExternalRef: string | null
  setServerChatId: (id: string | null) => void
  setServerChatTitle: (title: string | null) => void
  setServerChatCharacterId: (id: string | number | null) => void
  setServerChatAssistantKind: (
    kind: "character" | "persona" | null
  ) => void
  setServerChatAssistantId: (id: string | null) => void
  setServerChatPersonaMemoryMode: (
    mode: "read_only" | "read_write" | null
  ) => void
  setServerChatMetaLoaded: (loaded: boolean) => void
  setServerChatState: (state: ConversationState | null) => void
  setServerChatVersion: (version: number | null) => void
  setServerChatTopic: (topic: string | null) => void
  setServerChatClusterId: (clusterId: string | null) => void
  setServerChatSource: (source: string | null) => void
  setServerChatExternalRef: (ref: string | null) => void
  ensureServerChatHistoryId: (
    chatId: string,
    title?: string
  ) => Promise<string | null>
  contextFiles: UploadedFile[]
  setContextFiles: (files: UploadedFile[]) => void
  documentContext: ChatDocuments | null
  setDocumentContext: (docs: ChatDocuments) => void
  uploadedFiles: UploadedFile[]
  compareModeActive: boolean
  compareSelectedModels: string[]
  compareMaxModels: number
  compareFeatureEnabled: boolean
  markCompareHistoryCreated: (historyId: string) => void
  replyTarget: ReplyTarget | null
  clearReplyTarget: () => void
  messageSteeringPrompts: MessageSteeringPromptTemplates | null
  setSelectedQuickPrompt: (prompt: string | null) => void
  setSelectedSystemPrompt: (prompt: string) => void
  invalidateServerChatHistory: () => void
  selectedCharacter: Character | null
  selectedAssistant: AssistantSelection | null
  messageSteeringMode: MessageSteeringMode
  messageSteeringForceNarrate: boolean
  clearMessageSteering: () => void
}

export const useChatActions = ({
  t,
  notification,
  abortController,
  setAbortController,
  messages,
  setMessages,
  history,
  setHistory,
  historyId,
  setHistoryId,
  temporaryChat,
  selectedModel,
  useOCR,
  selectedSystemPrompt,
  selectedKnowledge,
  toolChoice,
  webSearch,
  currentChatModelSettings,
  setIsSearchingInternet,
  setIsProcessing,
  setStreaming,
  setActionInfo,
  fileRetrievalEnabled,
  ragMediaIds,
  ragSearchMode,
  ragTopK,
  ragEnableGeneration,
  ragEnableCitations,
  ragSources,
  ragAdvancedOptions,
  serverChatId,
  serverChatTitle,
  serverChatCharacterId,
  serverChatAssistantKind,
  serverChatAssistantId,
  serverChatPersonaMemoryMode,
  serverChatState,
  serverChatTopic,
  serverChatClusterId,
  serverChatSource,
  serverChatExternalRef,
  setServerChatId,
  setServerChatTitle,
  setServerChatCharacterId,
  setServerChatAssistantKind,
  setServerChatAssistantId,
  setServerChatPersonaMemoryMode,
  setServerChatMetaLoaded,
  setServerChatState,
  setServerChatVersion,
  setServerChatTopic,
  setServerChatClusterId,
  setServerChatSource,
  setServerChatExternalRef,
  ensureServerChatHistoryId,
  contextFiles,
  setContextFiles,
  documentContext,
  setDocumentContext,
  uploadedFiles,
  compareModeActive,
  compareSelectedModels,
  compareMaxModels,
  compareFeatureEnabled,
  markCompareHistoryCreated,
  replyTarget,
  clearReplyTarget,
  messageSteeringPrompts,
  setSelectedQuickPrompt,
  setSelectedSystemPrompt,
  invalidateServerChatHistory,
  selectedCharacter,
  selectedAssistant,
  messageSteeringMode,
  messageSteeringForceNarrate,
  clearMessageSteering
}: UseChatActionsOptions) => {
  const [appendFormattingGuidePrompt] = useStorage(
    PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY,
    false
  )
  const [imageEventSyncGlobalDefault] = useStorage<ImageGenerationEventSyncMode>(
    PLAYGROUND_IMAGE_EVENT_SYNC_DEFAULT_STORAGE_KEY,
    "off"
  )
  const normalizeSelectedModel = React.useCallback(
    (value: string | null | undefined): string | null => {
      if (typeof value !== "string") return null
      const trimmed = value.trim()
      return trimmed.length > 0 ? trimmed : null
    },
    []
  )

  const getEffectiveSelectedModel = React.useCallback(
    (preferred?: string | null): string | null => {
      const fromPreferred = normalizeSelectedModel(preferred)
      if (fromPreferred) return fromPreferred

      const fromHookState = normalizeSelectedModel(selectedModel)
      if (fromHookState) return fromHookState

      try {
        const fromStore = normalizeSelectedModel(
          useStoreMessageOption.getState().selectedModel
        )
        if (fromStore) return fromStore
      } catch {
        // Best-effort fallback only.
      }

      return null
    },
    [normalizeSelectedModel, selectedModel]
  )

  const { settings: chatSettings } = useChatSettingsRecord({
    historyId,
    serverChatId
  })
  const greetingEnabled = chatSettings?.greetingEnabled ?? true
  const greetingSelectionId =
    typeof chatSettings?.greetingSelectionId === "string"
      ? chatSettings.greetingSelectionId
      : null
  const greetingsChecksum =
    typeof chatSettings?.greetingsChecksum === "string"
      ? chatSettings.greetingsChecksum
      : null
  const useCharacterDefault = Boolean(chatSettings?.useCharacterDefault)
  const directedCharacterId = React.useMemo(() => {
    const raw = chatSettings?.directedCharacterId
    const parsed = Number.parseInt(String(raw ?? ""), 10)
    if (!Number.isFinite(parsed) || parsed <= 0) return null
    return parsed
  }, [chatSettings?.directedCharacterId])
  const resolvedMessageSteering = React.useMemo(
    () =>
      resolveMessageSteering({
        mode: messageSteeringMode,
        forceNarrate: messageSteeringForceNarrate
      }),
    [messageSteeringForceNarrate, messageSteeringMode]
  )
  const resolvedMessageSteeringPrompts = React.useMemo(
    () =>
      normalizeMessageSteeringPrompts(
        messageSteeringPrompts ?? DEFAULT_MESSAGE_STEERING_PROMPTS
      ),
    [messageSteeringPrompts]
  )
  const systemPromptAppendix = React.useMemo(
    () => resolveOutputFormattingGuideSuffix(Boolean(appendFormattingGuidePrompt)),
    [appendFormattingGuidePrompt]
  )
  const messagesRef = React.useRef(messages)
  const discardCurrentTurnOnAbortRef = React.useRef(false)

  React.useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  const resolveSelectedCharacter = React.useCallback(async () => {
    try {
      const storedRaw = await selectedCharacterStorage.get(
        SELECTED_CHARACTER_STORAGE_KEY
      )
      const stored = parseSelectedCharacterValue<Character>(storedRaw)
      if (stored?.id) {
        if (
          !selectedCharacter?.id ||
          String(stored.id) !== String(selectedCharacter.id)
        ) {
          return stored
        }
      }
      const storedSyncRaw = await selectedCharacterSyncStorage.get(
        SELECTED_CHARACTER_STORAGE_KEY
      )
      const storedSync = parseSelectedCharacterValue<Character>(storedSyncRaw)
      if (storedSync?.id) {
        await selectedCharacterStorage
          .set(SELECTED_CHARACTER_STORAGE_KEY, storedSync)
          .catch(() => {})
        if (
          !selectedCharacter?.id ||
          String(storedSync.id) !== String(selectedCharacter.id)
        ) {
          return storedSync
        }
      }
    } catch {
      // best-effort only
    }
    return selectedCharacter
  }, [selectedCharacter])

  const baseSaveMessageOnSuccess = createSaveMessageOnSuccess(
    temporaryChat,
    setHistoryId as (
      id: string,
      options?: { preserveServerChatId?: boolean }
    ) => void
  )
  const saveMessageOnError = createSaveMessageOnError(
    temporaryChat,
    history,
    setHistory,
    setHistoryId as (
      id: string,
      options?: { preserveServerChatId?: boolean }
    ) => void
  )

  const resolveImageEventSyncModeForPayload = React.useCallback(
    (payload?: SaveMessagePayload): ImageGenerationEventSyncMode => {
      const requestPolicy = normalizeImageGenerationEventSyncPolicy(
        payload?.imageEventSyncPolicy,
        "inherit"
      )
      const chatMode = normalizeImageGenerationEventSyncMode(
        chatSettings?.imageEventSyncMode,
        "off"
      )
      const globalMode = normalizeImageGenerationEventSyncMode(
        imageEventSyncGlobalDefault,
        "off"
      )
      return resolveImageGenerationEventSyncMode({
        requestPolicy,
        chatMode,
        globalMode
      })
    },
    [chatSettings?.imageEventSyncMode, imageEventSyncGlobalDefault]
  )

  const updateImageEventSyncMetadata = React.useCallback(
    async (
      payload: SaveMessagePayload,
      update: {
        status: "pending" | "synced" | "failed"
        policy: ImageGenerationEventSyncPolicy
        mode: ImageGenerationEventSyncMode
        serverMessageId?: string
        error?: string
      }
    ) => {
      const targetMessageId = payload.assistantMessageId
      if (!targetMessageId) return
      const now = Date.now()

      let nextGenerationInfo: Record<string, unknown> | null = null
      let nextImages: string[] = []

      setMessages((prev) =>
        prev.map((entry) => {
          if (entry.id !== targetMessageId) return entry
          const currentGenerationInfo =
            entry.generationInfo &&
            typeof entry.generationInfo === "object" &&
            !Array.isArray(entry.generationInfo)
              ? (entry.generationInfo as Record<string, unknown>)
              : {}
          const currentImageGeneration =
            currentGenerationInfo.image_generation &&
            typeof currentGenerationInfo.image_generation === "object" &&
            !Array.isArray(currentGenerationInfo.image_generation)
              ? (currentGenerationInfo.image_generation as Record<string, unknown>)
              : {}

          nextGenerationInfo = {
            ...currentGenerationInfo,
            image_generation: {
              ...currentImageGeneration,
              sync: {
                mode: update.mode,
                policy: update.policy,
                status: update.status,
                serverMessageId: update.serverMessageId,
                error: update.error,
                lastAttemptAt: now,
                mirroredAt: update.status === "synced" ? now : undefined
              }
            }
          }
          nextImages = Array.isArray(entry.images)
            ? entry.images.filter(
                (image): image is string =>
                  typeof image === "string" && image.length > 0
              )
            : []
          return {
            ...entry,
            generationInfo: nextGenerationInfo
          }
        })
      )

      if (!nextGenerationInfo) return
      await updateMessageMedia(targetMessageId, {
        images: nextImages,
        generationInfo: nextGenerationInfo
      }).catch(() => null)
    },
    [setMessages]
  )

  const saveMessageOnSuccess = async (
    payload?: SaveMessagePayload
  ): Promise<string | null> => {
    const payloadWithHistory = payload
      ? {
          ...payload,
          setHistoryId:
            payload.setHistoryId ??
            ((id: string) => {
              setHistoryId(id)
            })
        }
      : undefined
    const historyKey = await baseSaveMessageOnSuccess(payloadWithHistory)

    if (!payload?.historyId && historyKey) {
      markCompareHistoryCreated(historyKey)
    }

    if (temporaryChat) {
      return historyKey
    }

    let skipServerWrite = false
    const payloadConversationId =
      typeof payload?.conversationId === "string"
        ? payload.conversationId
        : payload?.conversationId != null
          ? String(payload.conversationId)
          : null
    const resolvedServerConversationId =
      payloadConversationId ??
      (serverChatId != null ? String(serverChatId) : null)
    const serverConversationMatches = payloadConversationId
      ? serverChatId != null
        ? payloadConversationId === String(serverChatId)
        : true
      : true
    const isServerConversation = Boolean(
      resolvedServerConversationId && serverConversationMatches
    )
    const isImageGenerationNoOp =
      isImageGenerationMessageType(payload?.userMessageType) ||
      isImageGenerationMessageType(payload?.assistantMessageType)
    const imageEventSyncPolicy = normalizeImageGenerationEventSyncPolicy(
      payload?.imageEventSyncPolicy,
      "inherit"
    )
    const imageEventSyncMode = resolveImageEventSyncModeForPayload(payload)

    if (isServerConversation && payload?.saveToDb) {
      try {
        const caps = await getServerCapabilities()
        skipServerWrite = Boolean(caps?.hasChatSaveToDb)
      } catch {
        skipServerWrite = false
      }
    }

    if (payload?.serverMessagesAlreadyPersisted) {
      skipServerWrite = true
    }

    // When resuming a server-backed chat, mirror new turns to /api/v1/chats.
    if (
      resolvedServerConversationId &&
      serverConversationMatches &&
      !skipServerWrite
    ) {
      if (
        isImageGenerationNoOp &&
        imageEventSyncMode === "on" &&
        payload?.assistantMessageId
      ) {
        await updateImageEventSyncMetadata(payload, {
          status: "pending",
          mode: imageEventSyncMode,
          policy: imageEventSyncPolicy
        })

        try {
          const generationInfo =
            payload.generationInfo &&
            typeof payload.generationInfo === "object" &&
            !Array.isArray(payload.generationInfo)
              ? (payload.generationInfo as Record<string, unknown>)
              : {}
          const imageGeneration =
            generationInfo.image_generation &&
            typeof generationInfo.image_generation === "object" &&
            !Array.isArray(generationInfo.image_generation)
              ? (generationInfo.image_generation as Record<string, unknown>)
              : {}
          const request =
            imageGeneration.request &&
            typeof imageGeneration.request === "object" &&
            !Array.isArray(imageGeneration.request)
              ? imageGeneration.request
              : null
          if (!request) {
            throw new Error("Image event sync skipped: missing request metadata.")
          }

          const mirroredImages = Array.isArray(payload.assistantImages)
            ? payload.assistantImages.filter(
                (value): value is string =>
                  typeof value === "string" && value.startsWith("data:image/")
              )
            : []
          const latestPreview = mirroredImages[mirroredImages.length - 1]
          const variantCount =
            typeof imageGeneration.variant_count === "number" &&
            Number.isFinite(imageGeneration.variant_count)
              ? Math.max(1, Math.round(imageGeneration.variant_count))
              : undefined
          const activeVariantIndex =
            typeof imageGeneration.active_variant_index === "number" &&
            Number.isFinite(imageGeneration.active_variant_index)
              ? Math.max(0, Math.round(imageGeneration.active_variant_index))
              : undefined
          const eventId =
            typeof imageGeneration.event_id === "string" &&
            imageGeneration.event_id.trim().length > 0
              ? imageGeneration.event_id.trim()
              : payload.assistantMessageId
          const mirroredContent = buildImageGenerationEventMirrorContent({
            kind: "image_generation_event",
            version: 1,
            eventId,
            createdAt:
              typeof imageGeneration.createdAt === "number" &&
              Number.isFinite(imageGeneration.createdAt)
                ? imageGeneration.createdAt
                : Date.now(),
            fileId:
              typeof generationInfo.file_id === "string" &&
              generationInfo.file_id.trim().length > 0
                ? generationInfo.file_id.trim()
                : undefined,
            request: request as ImageGenerationRequestSnapshot,
            promptMode:
              imageGeneration.promptMode === "scene" ||
              imageGeneration.promptMode === "expression" ||
              imageGeneration.promptMode === "selfie" ||
              imageGeneration.promptMode === "camera-angle" ||
              imageGeneration.promptMode === "outfit" ||
              imageGeneration.promptMode === "custom"
                ? imageGeneration.promptMode
                : undefined,
            source:
              imageGeneration.source === "slash-command" ||
              imageGeneration.source === "generate-modal" ||
              imageGeneration.source === "message-regen"
                ? imageGeneration.source
                : undefined,
            refine:
              imageGeneration.refine &&
              typeof imageGeneration.refine === "object" &&
              !Array.isArray(imageGeneration.refine)
                ? (imageGeneration.refine as ImageGenerationRefineMetadata)
                : undefined,
            variantCount,
            activeVariantIndex,
            imageDataUrl: latestPreview
          })

          const mirroredMessage = await tldwClient.addChatMessage(
            resolvedServerConversationId,
            {
            role: "assistant",
            content: mirroredContent
            }
          )

          await updateImageEventSyncMetadata(payload, {
            status: "synced",
            mode: imageEventSyncMode,
            policy: imageEventSyncPolicy,
            serverMessageId:
              mirroredMessage?.id != null ? String(mirroredMessage.id) : undefined
          })
        } catch (error) {
          const reason =
            error instanceof Error && error.message.trim().length > 0
              ? error.message
              : "Failed to mirror image event to server history."
          await updateImageEventSyncMetadata(payload, {
            status: "failed",
            mode: imageEventSyncMode,
            policy: imageEventSyncPolicy,
            error: reason
          })
        }
      } else if (
        !isImageGenerationNoOp &&
        !payload?.isRegenerate &&
        !payload?.isContinue &&
        typeof payload?.message === "string" &&
        typeof payload?.fullText === "string"
      ) {
        try {
          const cid = resolvedServerConversationId
          const userContent = payload.message.trim()
          const assistantContent = payload.fullText.trim()

          if (userContent.length > 0) {
            await tldwClient.addChatMessage(cid, {
              role: "user",
              content: userContent
            })
          }

          if (assistantContent.length > 0) {
            await tldwClient.addChatMessage(cid, {
              role: "assistant",
              content: assistantContent
            })
          }
        } catch {
          // Ignore sync errors; local history is still saved.
        }
      }
    }

    return historyKey
  }

  const buildChatModeParams = async (overrides: ChatModeOverrides = {}) => {
    const hasHistoryOverride = Object.prototype.hasOwnProperty.call(
      overrides,
      "historyId"
    )
    const resolvedServerChatId =
      overrides.serverChatId === undefined ? serverChatId : overrides.serverChatId
    const resolvedHistoryId = hasHistoryOverride
      ? overrides.historyId
      : resolvedServerChatId && !temporaryChat
        ? await ensureServerChatHistoryId(
            resolvedServerChatId,
            serverChatTitle || undefined
          )
        : historyId

    const { getActorSettingsForChat } = await loadActorSettings()
    const actorSettings = await getActorSettingsForChat({
      historyId: resolvedHistoryId ?? historyId,
      serverChatId: resolvedServerChatId
    })

    const effectiveSelectedModel = getEffectiveSelectedModel(
      overrides.selectedModel
    )
    const resolvedSelectedSystemPrompt = Object.prototype.hasOwnProperty.call(
      overrides,
      "selectedSystemPrompt"
    )
      ? (overrides.selectedSystemPrompt as string | null)
      : selectedSystemPrompt
    const resolvedToolChoice =
      overrides.toolChoice === "auto" ||
      overrides.toolChoice === "required" ||
      overrides.toolChoice === "none"
        ? overrides.toolChoice
        : toolChoice
    const resolvedUseOCR =
      typeof overrides.useOCR === "boolean" ? overrides.useOCR : useOCR
    const resolvedWebSearch =
      typeof overrides.webSearch === "boolean"
        ? overrides.webSearch
        : webSearch

    return {
      selectedModel: effectiveSelectedModel || "",
      useOCR: resolvedUseOCR,
      selectedSystemPrompt: resolvedSelectedSystemPrompt,
      selectedKnowledge,
      toolChoice: resolvedToolChoice,
      currentChatModelSettings,
      setMessages,
      setIsSearchingInternet,
      saveMessageOnSuccess,
      saveMessageOnError,
      setHistory,
      setIsProcessing,
      setStreaming,
      setAbortController,
      historyId: resolvedHistoryId ?? historyId,
      setHistoryId,
      fileRetrievalEnabled,
      ragMediaIds,
      ragSearchMode,
      ragTopK,
      ragEnableGeneration,
      ragEnableCitations,
      ragSources,
      ragAdvancedOptions,
      setActionInfo,
      webSearch: resolvedWebSearch,
      actorSettings,
      systemPromptAppendix,
      messageSteeringPrompts: resolvedMessageSteeringPrompts,
      ...overrides
    }
  }

  const ensurePersonaServerChatWithState = React.useCallback(
    async ({
      assistant,
      serverChatIdOverride
    }: {
      assistant: AssistantSelection & { kind: "persona" }
      serverChatIdOverride?: string | null
    }): Promise<{
      chatId: string
      historyId: string | null
      personaMemoryMode: "read_only" | "read_write"
    }> =>
      ensurePersonaServerChat({
        assistant,
        serverChatIdOverride,
        serverChatId,
        serverChatTitle,
        serverChatAssistantKind,
        serverChatAssistantId,
        serverChatPersonaMemoryMode,
        serverChatState,
        serverChatTopic,
        serverChatClusterId,
        serverChatSource,
        serverChatExternalRef,
        historyId,
        temporaryChat,
        createChat: (payload) => tldwClient.createChat(payload),
        ensureServerChatHistoryId,
        invalidateServerChatHistory,
        setServerChatId,
        setServerChatTitle,
        setServerChatCharacterId,
        setServerChatAssistantKind,
        setServerChatAssistantId,
        setServerChatPersonaMemoryMode,
        setServerChatMetaLoaded,
        setServerChatState,
        setServerChatVersion,
        setServerChatTopic,
        setServerChatClusterId,
        setServerChatSource,
        setServerChatExternalRef
      }),
    [
      ensureServerChatHistoryId,
      historyId,
      invalidateServerChatHistory,
      serverChatAssistantId,
      serverChatAssistantKind,
      serverChatClusterId,
      serverChatExternalRef,
      serverChatId,
      serverChatPersonaMemoryMode,
      serverChatSource,
      serverChatState,
      serverChatTitle,
      serverChatTopic,
      setServerChatAssistantId,
      setServerChatAssistantKind,
      setServerChatCharacterId,
      setServerChatClusterId,
      setServerChatExternalRef,
      setServerChatId,
      setServerChatMetaLoaded,
      setServerChatPersonaMemoryMode,
      setServerChatSource,
      setServerChatState,
      setServerChatTitle,
      setServerChatTopic,
      setServerChatVersion,
      temporaryChat
    ]
  )

  const characterChatMode = async ({
    message,
    image,
    isRegenerate,
    messages: chatHistory,
    history: chatMemory,
    signal,
    model,
    regenerateFromMessage,
    character,
    messageSteering,
    serverChatIdOverride
  }: {
    message: string
    image: string
    isRegenerate: boolean
    messages: Message[]
    history: ChatHistory
    signal: AbortSignal
    model: string
    regenerateFromMessage?: Message
    character?: Character | null
    serverChatIdOverride?: string | null
    messageSteering: {
      continueAsUser: boolean
      impersonateUser: boolean
      forceNarrate: boolean
    }
  }) => {
    const activeCharacter = character ?? selectedCharacter
    if (!activeCharacter?.id) {
      throw new Error("No character selected")
    }

    const resolveGreetingText = (): string => {
      if (!greetingEnabled) return ""

      const hasUserTurns =
        chatHistory.some((entry) => !entry.isBot) ||
        chatMemory.some((entry) => entry.role === "user")
      const greetingEntries = collectGreetingEntries(activeCharacter as any)
      const greetingOptions = buildGreetingOptionsFromEntries(greetingEntries)
      const selectedGreeting =
        resolveGreetingSelection({
          options: greetingOptions,
          greetingSelectionId,
          greetingsChecksum,
          useCharacterDefault,
          fallback: "first"
        }).option?.text?.trim() ?? ""
      if (!hasUserTurns && selectedGreeting.length > 0) {
        return selectedGreeting
      }

      const fromMessages = chatHistory.find(
        (entry) =>
          entry.isBot &&
          isGreetingMessageType(entry.messageType) &&
          typeof entry.message === "string" &&
          entry.message.trim().length > 0
      )
      if (fromMessages?.message) {
        return fromMessages.message.trim()
      }

      const fromHistory = chatMemory.find(
        (entry) =>
          entry.role === "assistant" &&
          isGreetingMessageType(entry.messageType) &&
          typeof entry.content === "string" &&
          entry.content.trim().length > 0
      )
      if (fromHistory?.content) {
        return fromHistory.content.trim()
      }

      if (selectedGreeting.length > 0) {
        return selectedGreeting
      }

      const fromCharacter = collectGreetings(activeCharacter as any).find(
        (candidate) =>
          typeof candidate === "string" && candidate.trim().length > 0
      )
      if (fromCharacter) {
        return fromCharacter.trim()
      }

      return ""
    }

    const greetingText = resolveGreetingText()
    const hasGreetingInHistory =
      greetingText.length > 0 &&
      chatMemory.some(
        (entry) =>
          entry.role === "assistant" &&
          typeof entry.content === "string" &&
          entry.content.trim() === greetingText
      )
    const historyBase: ChatHistory =
      greetingText.length > 0 && !hasGreetingInHistory
        ? [
            {
              role: "assistant",
              content: greetingText,
              messageType: "character:greeting"
            },
            ...chatMemory
          ]
        : chatMemory

    let fullText = ""
    let contentToSave = ""
    const resolvedAssistantMessageId = generateID()
    const resolvedUserMessageId = !isRegenerate ? generateID() : undefined
    let persistedUserServerMessageId: string | undefined
    let activeChatId: string | null = null
    let assistantPersistedToServer = false
    let generateMessageId = resolvedAssistantMessageId
    const fallbackParentMessageId = getLastUserMessageId(chatHistory)
    const resolvedAssistantParentMessageId = isRegenerate
      ? regenerateFromMessage?.parentMessageId ?? fallbackParentMessageId
      : resolvedUserMessageId ?? null
    const regenerateVariants =
      isRegenerate && regenerateFromMessage
        ? normalizeMessageVariants(regenerateFromMessage)
        : []
    const resolvedModel = model?.trim()
    let streamingTimer: ReturnType<typeof setTimeout> | null = null
    let pendingStreamingText = ""
    let pendingReasoningTime = 0
    let lastStreamingUpdateAt = 0

    const flushStreamingUpdate = () => {
      if (pendingStreamingText.length === 0) return
      setMessages((prev) =>
        prev.map((m) =>
          m.id === generateMessageId
            ? updateActiveVariant(m, {
                message: pendingStreamingText,
                reasoning_time_taken: pendingReasoningTime
              })
            : m
        )
      )
      pendingStreamingText = ""
      pendingReasoningTime = 0
      lastStreamingUpdateAt = Date.now()
    }

    const scheduleStreamingUpdate = (text: string, reasoningTime: number) => {
      pendingStreamingText = text
      pendingReasoningTime = reasoningTime
      if (streamingTimer !== null) return
      const elapsed = Date.now() - lastStreamingUpdateAt
      const delay = Math.max(0, STREAMING_UPDATE_INTERVAL_MS - elapsed)
      streamingTimer = setTimeout(() => {
        streamingTimer = null
        flushStreamingUpdate()
      }, delay)
    }

    const cancelStreamingUpdate = () => {
      if (streamingTimer === null) return
      clearTimeout(streamingTimer)
      streamingTimer = null
    }

    try {
      if (!resolvedModel) {
        notification.error({
          message: t("error"),
          description: t("validationSelectModel")
        })
        setIsProcessing(false)
        setStreaming(false)
        return
      }

      const hasImageInput =
        typeof image === "string" && image.trim().length > 0
      if (!isRegenerate && message.trim().length === 0 && !hasImageInput) {
        notification.error({
          message: t("error"),
          description: t(
            "playground:composer.validationMessageRequired",
            "Type a message before sending."
          )
        })
        setIsProcessing(false)
        setStreaming(false)
        return
      }

      await tldwClient.initialize().catch(() => null)

      const modelInfo = await getModelNicknameByID(resolvedModel)
      const characterName =
        activeCharacter?.name || modelInfo?.model_name || resolvedModel
      const characterAvatar =
        activeCharacter?.avatar_url || modelInfo?.model_avatar
      const createdAt = Date.now()
      const hasGreetingInMessages = chatHistory.some((entry) => {
        if (!entry?.isBot) return false
        if (isGreetingMessageType(entry?.messageType)) return true
        if (!greetingText) return false
        return (
          typeof entry.message === "string" &&
          entry.message.trim() === greetingText
        )
      })
      const greetingSeedMessage: Message | null =
        greetingText.length > 0 && !hasGreetingInMessages
          ? {
              isBot: true,
              role: "assistant",
              name: characterName,
              message: greetingText,
              messageType: "character:greeting",
              sources: [],
              createdAt,
              id: generateID(),
              modelImage: characterAvatar,
              modelName: characterName
            }
          : null
      const chatMessagesBase = greetingSeedMessage
        ? [greetingSeedMessage, ...chatHistory]
        : chatHistory
      const assistantStub: Message = {
        isBot: true,
        role: "assistant",
        name: characterName,
        message: "▋",
        sources: [],
        createdAt,
        id: generateMessageId,
        modelImage: characterAvatar,
        modelName: characterName,
        parentMessageId: resolvedAssistantParentMessageId ?? null
      }
      if (regenerateVariants.length > 0) {
        const variants = [
          ...regenerateVariants,
          buildMessageVariant(assistantStub)
        ]
        assistantStub.variants = variants
        assistantStub.activeVariantIndex = variants.length - 1
      }

      const newMessageList: Message[] = !isRegenerate
        ? [
            ...chatMessagesBase,
            {
              isBot: false,
              role: "user",
              name: "You",
              message,
              sources: [],
              images: [],
              createdAt,
              id: resolvedUserMessageId,
              parentMessageId: null
            },
            assistantStub
          ]
        : [...chatMessagesBase, assistantStub]
      setMessages(newMessageList)

      const activeCharacterId = String(activeCharacter.id)
      const serverCharacterId =
        serverChatCharacterId != null ? String(serverChatCharacterId) : null
      const overrideChatId =
        typeof serverChatIdOverride === "string" &&
        serverChatIdOverride.trim().length > 0
          ? serverChatIdOverride.trim()
          : null
      const resolvedServerChatId = overrideChatId || serverChatId
      const shouldResetServerChat =
        Boolean(resolvedServerChatId) &&
        (!serverCharacterId || serverCharacterId !== activeCharacterId)

      if (shouldResetServerChat) {
        setServerChatId(null)
        setServerChatCharacterId(null)
        setServerChatMetaLoaded(false)
        setServerChatTitle(null)
        setServerChatState("in-progress")
        setServerChatVersion(null)
        setServerChatTopic(null)
        setServerChatClusterId(null)
        setServerChatSource(null)
        setServerChatExternalRef(null)
      }

      let chatId = shouldResetServerChat ? null : resolvedServerChatId
      let createdNewChat = false
      if (!chatId) {
        const created = await tldwClient.createChat({
          character_id: activeCharacter.id,
          state: serverChatState || "in-progress",
          topic_label: serverChatTopic || undefined,
          cluster_id: serverChatClusterId || undefined,
          source: serverChatSource || undefined,
          external_ref: serverChatExternalRef || undefined
        }) as TldwChatMeta

        let rawId: string | number | undefined
        if (created && typeof created === "object") {
          const {
            id,
            chat_id,
            version,
            state,
            conversation_state,
            topic_label,
            cluster_id,
            source,
            external_ref
          } = created
          rawId = id ?? chat_id
          const normalizedState = normalizeConversationState(
            state ?? conversation_state ?? null
          )
          setServerChatState(normalizedState)
          setServerChatVersion(typeof version === "number" ? version : null)
          setServerChatTopic(topic_label ?? null)
          setServerChatClusterId(cluster_id ?? null)
          setServerChatSource(source ?? null)
          setServerChatExternalRef(external_ref ?? null)
        } else if (typeof created === "string" || typeof created === "number") {
          rawId = created
        }

        const normalizedId = rawId != null ? String(rawId) : ""
        if (!normalizedId) {
          throw new Error("Failed to create character chat session")
        }
        chatId = normalizedId
        createdNewChat = true
        setServerChatId(normalizedId)
        const createdTitle =
          created && typeof created === "object"
            ? String(created.title ?? "")
            : ""
        const createdCharacterId =
          created && typeof created === "object"
            ? created.character_id ?? activeCharacter?.id ?? null
            : activeCharacter?.id ?? null
        setServerChatTitle(createdTitle)
        setServerChatCharacterId(createdCharacterId)
        setServerChatMetaLoaded(true)
        invalidateServerChatHistory()
      }
      activeChatId = chatId

      if (createdNewChat && !isRegenerate && greetingText.length > 0) {
        try {
          const createdGreeting = (await tldwClient.addChatMessage(chatId, {
            role: "assistant",
            content: greetingText
          })) as { id?: string | number; version?: number } | null
          if (createdGreeting?.id != null) {
            setMessages((prev) => {
              const updated = [...prev] as (Message & {
                serverMessageId?: string
                serverMessageVersion?: number
              })[]
              const serverMessageId = String(createdGreeting.id)
              const serverMessageVersion = createdGreeting.version
              for (let i = 0; i < updated.length; i += 1) {
                if (
                  updated[i]?.isBot &&
                  isGreetingMessageType(updated[i]?.messageType) &&
                  !updated[i]?.serverMessageId
                ) {
                  updated[i] = {
                    ...updated[i],
                    serverMessageId,
                    serverMessageVersion
                  }
                  break
                }
              }
              return updated as Message[]
            })
          }
        } catch (greetingPersistError) {
          console.warn(
            "Failed to persist character greeting for new chat:",
            greetingPersistError
          )
        }
      }

      if (!isRegenerate) {
        type TldwChatMessage = {
          id?: string | number
          version?: number
          role?: string
          content?: string
          image_base64?: string
        }

        const payload: TldwChatMessage = { role: "user" }
        const trimmedUserMessage = message.trim()
        if (trimmedUserMessage.length > 0) {
          payload.content = message
        }
        let normalizedImage = image
        if (normalizedImage.length > 0 && !normalizedImage.startsWith("data:")) {
          const payloadValue = normalizedImage.includes(",")
            ? normalizedImage.split(",")[1]
            : normalizedImage
          if (payloadValue !== undefined && payloadValue.length > 0) {
            normalizedImage = `data:image/jpeg;base64,${payloadValue}`
          }
        }
        if (normalizedImage && normalizedImage.startsWith("data:")) {
          const b64 = normalizedImage.includes(",")
            ? normalizedImage.split(",")[1]
            : normalizedImage
          if (b64 !== undefined && b64.length > 0) {
            payload.image_base64 = b64
          }
        }
        if (payload.content || payload.image_base64) {
          const createdUser = (await tldwClient.addChatMessage(
            chatId,
            payload
          )) as TldwChatMessage | null
          persistedUserServerMessageId =
            createdUser?.id != null ? String(createdUser.id) : undefined
          setMessages((prev) => {
            const updated = [...prev] as (Message & {
              serverMessageId?: string
              serverMessageVersion?: number
            })[]
            const serverMessageId =
              createdUser?.id != null ? String(createdUser.id) : undefined
            const serverMessageVersion = createdUser?.version
            for (let i = updated.length - 1; i >= 0; i--) {
              if (!updated[i].isBot) {
                updated[i] = {
                  ...updated[i],
                  serverMessageId,
                  serverMessageVersion
                }
                break
              }
            }
            return updated as Message[]
          })
        }
      }

      let count = 0
      let reasoningStartTime: Date | null = null
      let reasoningEndTime: Date | null = null
      let timetaken = 0
      let apiReasoning = false
      let streamTransportInterrupted = false
      let streamTransportInterruptionReason: string | null = null

      const explicitProvider = resolveExplicitProviderForSelectedModel({
        currentSelectedModel: getEffectiveSelectedModel(),
        requestedSelectedModel: resolvedModel,
        explicitProvider: currentChatModelSettings.apiProvider
      })
      const resolvedApiProvider = await resolveApiProviderForModel({
        modelId: resolvedModel,
        explicitProvider
      })
      const normalizedModel = resolvedModel.replace(/^tldw:/, "").trim()
      const streamModel =
        normalizedModel.length > 0 ? normalizedModel : resolvedModel

      const shouldPersistToServer = !temporaryChat
      for await (const chunk of tldwClient.streamCharacterChatCompletion(
        chatId,
        {
          include_character_context: true,
          model: streamModel,
          provider: resolvedApiProvider,
          save_to_db: shouldPersistToServer,
          directed_character_id: directedCharacterId ?? undefined,
          continue_as_user: messageSteering.continueAsUser,
          impersonate_user: messageSteering.impersonateUser,
          force_narrate: messageSteering.forceNarrate,
          message_steering_prompts: toMessageSteeringPromptPayload(
            resolvedMessageSteeringPrompts
          )
        },
        { signal }
      )) {
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
          chunk
        )
        fullText = chunkState.fullText
        contentToSave = chunkState.contentToSave
        apiReasoning = chunkState.apiReasoning

        if (chunkState.token) {
          scheduleStreamingUpdate(`${fullText}▋`, timetaken)
        }
        if (count === 0) {
          setIsProcessing(true)
        }

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

        count++
        if (signal?.aborted) break
      }
      cancelStreamingUpdate()
      flushStreamingUpdate()

      if (signal?.aborted) {
        const abortError = new Error("AbortError")
        ;(abortError as any).name = "AbortError"
        throw abortError
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === generateMessageId
            ? (() => {
                const nextVariantPayload: Record<string, unknown> = {
                  message: fullText,
                  reasoning_time_taken: timetaken
                }
                if (streamTransportInterrupted) {
                  const existingGenerationInfo =
                    m.generationInfo &&
                    typeof m.generationInfo === "object" &&
                    !Array.isArray(m.generationInfo)
                      ? (m.generationInfo as Record<string, unknown>)
                      : {}
                  nextVariantPayload.generationInfo = {
                    ...existingGenerationInfo,
                    streamTransportInterrupted: true,
                    partialResponseSaved: true,
                    streamTransportInterruptionReason:
                      streamTransportInterruptionReason ||
                      "Stream transport interrupted; partial response saved."
                  }
                }
                return updateActiveVariant(m, nextVariantPayload)
              })()
            : m
        )
      )

      const finalContent = contentToSave || fullText
      const finalPersistedContent = finalContent.trim()

      if (finalPersistedContent.length > 0) {
        let fallbackSpeakerId: number | undefined
        let speakerCharacterId: number | undefined
        let detectedMood: ReturnType<typeof detectCharacterMood> | undefined
        let resolvedMoodLabel: string | undefined
        let resolvedMoodConfidence: number | undefined
        let resolvedMoodTopic: string | undefined
        let metadataExtra: Record<string, unknown> | undefined
        try {
          fallbackSpeakerId = Number.parseInt(
            String(activeCharacter.id),
            10
          )
          speakerCharacterId =
            Number.isFinite(directedCharacterId ?? NaN) &&
            (directedCharacterId ?? 0) > 0
              ? directedCharacterId
              : Number.isFinite(fallbackSpeakerId) && fallbackSpeakerId > 0
                ? fallbackSpeakerId
                : undefined
          detectedMood = detectCharacterMood({
            assistantText: finalPersistedContent,
            userText: message
          })
          resolvedMoodLabel = detectedMood.label
          resolvedMoodConfidence =
            typeof detectedMood.confidence === "number" &&
            Number.isFinite(detectedMood.confidence)
              ? detectedMood.confidence
              : undefined
          resolvedMoodTopic =
            typeof detectedMood.topic === "string" && detectedMood.topic.trim()
              ? detectedMood.topic.trim()
              : undefined

          const persistPayload: Record<string, unknown> = {
            assistant_content: finalPersistedContent,
            assistant_message_id: generateMessageId,
            speaker_character_id: speakerCharacterId,
            speaker_character_name: characterName
          }
          if (resolvedMoodLabel) {
            persistPayload.mood_label = resolvedMoodLabel
          }
          if (typeof resolvedMoodConfidence === "number") {
            persistPayload.mood_confidence = resolvedMoodConfidence
          }
          if (resolvedMoodTopic) {
            persistPayload.mood_topic = resolvedMoodTopic
          }
          if (persistedUserServerMessageId) {
            persistPayload.user_message_id = persistedUserServerMessageId
          }
          metadataExtra = {
            speaker_character_id: speakerCharacterId ?? null,
            speaker_character_name: characterName,
            mood_label: resolvedMoodLabel,
            mood_confidence: resolvedMoodConfidence ?? null,
            mood_topic: resolvedMoodTopic ?? null
          }

          const persisted = (await tldwClient.persistCharacterCompletion(
            chatId,
            persistPayload
          )) as
            | {
                assistant_message_id?: string | number
                message_id?: string | number
                id?: string | number
                version?: number
              }
            | null
          const createdAsstServerId =
            persisted?.assistant_message_id ??
            persisted?.message_id ??
            persisted?.id
          const createdAsstVersion = persisted?.version
          assistantPersistedToServer = createdAsstServerId != null
          setMessages((prev) =>
            ((prev as any[]).map((m) => {
              if (m.id !== generateMessageId) return m
              const serverMessageId =
                createdAsstServerId != null
                  ? String(createdAsstServerId)
                  : undefined
              return updateActiveVariant(m, {
                serverMessageId,
                serverMessageVersion: createdAsstVersion,
                metadataExtra,
                speakerCharacterId: speakerCharacterId ?? null,
                speakerCharacterName: characterName,
                moodLabel: resolvedMoodLabel,
                moodConfidence: resolvedMoodConfidence ?? null,
                moodTopic: resolvedMoodTopic ?? null
              })
            }) as Message[])
          )
        } catch (e) {
          console.error(
            "Failed to persist assistant message via completions/persist:",
            e
          )
          const savedOutcome = resolveSavedDegradedCharacterPersist(e)
          if (savedOutcome?.saved) {
            assistantPersistedToServer = true
            setMessages((prev) =>
              ((prev as any[]).map((m) => {
                if (m.id !== generateMessageId) return m
                const serverMessageId =
                  savedOutcome.assistantMessageId != null
                    ? String(savedOutcome.assistantMessageId)
                    : undefined
                return updateActiveVariant(m, {
                  serverMessageId,
                  serverMessageVersion: savedOutcome.version,
                  metadataExtra,
                  speakerCharacterId: speakerCharacterId ?? null,
                  speakerCharacterName: characterName,
                  moodLabel: resolvedMoodLabel,
                  moodConfidence: resolvedMoodConfidence ?? null,
                  moodTopic: resolvedMoodTopic ?? null
                })
              }) as Message[])
            )
          } else {
            try {
              const createdAsst = (await tldwClient.addChatMessage(chatId, {
                role: "assistant",
                content: finalPersistedContent
              })) as { id?: string | number; version?: number } | null
              assistantPersistedToServer = createdAsst?.id != null
              setMessages((prev) =>
                ((prev as any[]).map((m) => {
                  if (m.id !== generateMessageId) return m
                  const serverMessageId =
                    createdAsst?.id != null ? String(createdAsst.id) : undefined
                return updateActiveVariant(m, {
                  serverMessageId,
                  serverMessageVersion: createdAsst?.version,
                  metadataExtra,
                  speakerCharacterId: speakerCharacterId ?? null,
                  speakerCharacterName: characterName,
                  moodLabel: resolvedMoodLabel,
                  moodConfidence: resolvedMoodConfidence ?? null,
                  moodTopic: resolvedMoodTopic ?? null
                })
                }) as Message[])
              )
            } catch (fallbackError) {
              console.error(
                "Failed fallback assistant persistence with addChatMessage:",
                fallbackError
              )
            }
          }
        }
      } else {
        console.warn(
          "Skipping assistant persistence because completion content is empty."
        )
      }

      const lastEntry = historyBase[historyBase.length - 1]
      const prevEntry = historyBase[historyBase.length - 2]
      const endsWithUser =
        lastEntry?.role === "user" && lastEntry.content === message
      const endsWithUserAssistant =
        lastEntry?.role === "assistant" &&
        prevEntry?.role === "user" &&
        prevEntry.content === message

      if (isRegenerate) {
        if (endsWithUser) {
          setHistory([
            ...historyBase,
            { role: "assistant", content: finalContent }
          ])
        } else if (endsWithUserAssistant) {
          setHistory(
            historyBase.map((entry, index) =>
              index === historyBase.length - 1 && entry.role === "assistant"
                ? { ...entry, content: finalContent }
                : entry
            )
          )
        } else {
          setHistory([
            ...historyBase,
            { role: "user", content: message, image },
            { role: "assistant", content: finalContent }
          ])
        }
      } else {
        setHistory([
          ...historyBase,
          { role: "user", content: message, image },
          { role: "assistant", content: finalContent }
        ])
      }

      await saveMessageOnSuccess({
        historyId,
        isRegenerate,
        selectedModel: resolvedModel,
        modelId: resolvedModel,
        message,
        image,
        fullText: finalContent,
        source: [],
        message_source: "web-ui",
        reasoning_time_taken: timetaken,
        userMessageId: resolvedUserMessageId,
        assistantMessageId: resolvedAssistantMessageId,
        assistantParentMessageId: resolvedAssistantParentMessageId ?? null,
        conversationId: activeChatId,
        saveToDb: true,
        serverMessagesAlreadyPersisted: Boolean(activeChatId) && !temporaryChat
      })

      setIsProcessing(false)
      setStreaming(false)
    } catch (e) {
      if (
        discardAbortedTurnIfRequested({
          discardRequested: discardCurrentTurnOnAbortRef.current,
          error: e,
          previousMessages: chatHistory,
          previousHistory: chatMemory,
          setMessages,
          setHistory
        })
      ) {
        setIsProcessing(false)
        setStreaming(false)
        return
      }

      cancelStreamingUpdate()
      await attemptCharacterStreamRecoveryPersist({
        chatId: activeChatId,
        temporaryChat,
        assistantContent: contentToSave || fullText,
        alreadyPersisted: assistantPersistedToServer,
        error: e,
        persist: async (assistantContent) => {
          if (!activeChatId) return false
          const createdAsst = (await tldwClient.addChatMessage(activeChatId, {
            role: "assistant",
            content: assistantContent
          })) as { id?: string | number; version?: number } | null
          if (createdAsst?.id == null) return false
          assistantPersistedToServer = true
          setMessages((prev) =>
            ((prev as any[]).map((m) => {
              if (m.id !== generateMessageId) return m
              return updateActiveVariant(m, {
                serverMessageId: String(createdAsst.id),
                serverMessageVersion: createdAsst?.version
              })
            }) as Message[])
          )
          return true
        }
      })
      const assistantContent = buildAssistantErrorContent(fullText, e)
      const interruptionReason =
        e instanceof Error ? e.message : t("somethingWentWrong")
      if (generateMessageId) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === generateMessageId
              ? updateActiveVariant(msg, {
                  message: assistantContent,
                  generationInfo: {
                    ...(msg.generationInfo || {}),
                    interrupted: true,
                    interruptionReason,
                    interruptedAt: Date.now()
                  }
                })
              : msg
          )
        )
      }
      const errorSave = await saveMessageOnError({
        e,
        botMessage: assistantContent,
        history: historyBase,
        historyId,
        image,
        selectedModel: resolvedModel,
        modelId: resolvedModel,
        userMessage: message,
        isRegenerating: isRegenerate,
        message_source: "web-ui",
        userMessageId: resolvedUserMessageId,
        assistantMessageId: resolvedAssistantMessageId,
        assistantParentMessageId: resolvedAssistantParentMessageId ?? null
      })

      if (!errorSave) {
        notification.error({
          message: t("error"),
          description: e instanceof Error ? e.message : t("somethingWentWrong")
        })
      }
      setIsProcessing(false)
      setStreaming(false)
    } finally {
      discardCurrentTurnOnAbortRef.current = false
      cancelStreamingUpdate()
      setAbortController(null)
    }
  }

  const buildCompareHistoryTitle = React.useCallback(
    (title: string) => {
      const trimmed =
        title?.trim() ||
        t("common:untitled", { defaultValue: "Untitled" })
      return t(
        "playground:composer.compareHistoryPrefix",
        "Compare: {{title}}",
        { title: trimmed }
      )
    },
    [t]
  )

  const buildCompareSplitTitle = React.useCallback(
    (title: string) => {
      const trimmed =
        title?.trim() ||
        t("common:untitled", { defaultValue: "Untitled" })
      const suffix = t(
        "playground:composer.compareHistorySuffix",
        "(from compare)"
      )
      if (trimmed.includes(suffix)) {
        return trimmed
      }
      return `${trimmed} ${suffix}`.trim()
    },
    [t]
  )

  const getMessageModelKey = (message: Message) =>
    message.modelId || message.modelName || message.name

  const shouldIncludeMessageForModel = (
    message: Message,
    modelId: string
  ) => {
    if (!message.isBot) {
      if (message.messageType === "compare:perModelUser") {
        return message.modelId === modelId
      }
      return true
    }
    const messageModel = getMessageModelKey(message)
    if (!messageModel) {
      return false
    }
    return messageModel === modelId
  }

  const buildHistoryFromMessages = React.useCallback(
    (items: Message[]): ChatHistory =>
      items
        .filter((message) =>
          !isImageGenerationMessageType(message.messageType) &&
          (greetingEnabled ? true : !isGreetingMessageType(message.messageType))
        )
        .map((message) => ({
          role: message.isBot ? "assistant" : "user",
          content: message.message,
          image: message.images?.[0],
          messageType: message.messageType
        })),
    [greetingEnabled]
  )

  const buildHistoryForModel = (
    items: Message[],
    modelId: string
  ): ChatHistory =>
    buildHistoryFromMessages(
      items.filter((message) => shouldIncludeMessageForModel(message, modelId))
    )

  const getCompareUserMessageId = (items: Message[], clusterId: string) =>
    items.find(
      (message) =>
        message.messageType === "compare:user" &&
        message.clusterId === clusterId
    )?.id || null

  const getLastThreadMessageId = (
    items: Message[],
    clusterId: string,
    modelId: string
  ) => {
    const threadMessages = items.filter(
      (message) =>
        message.clusterId === clusterId &&
        getMessageModelKey(message) === modelId
    )
    const lastThreadMessage = threadMessages[threadMessages.length - 1]
    return lastThreadMessage?.id || getCompareUserMessageId(items, clusterId)
  }

  const refreshHistoryFromMessages = React.useCallback(() => {
    const next = buildHistoryFromMessages(messagesRef.current)
    setHistory(next)
  }, [buildHistoryFromMessages, setHistory])

  const extractContinuationDraft = React.useCallback(
    (fullText: string, priorText: string): string => {
      const trimmedFull = fullText.trim()
      if (!trimmedFull) return ""
      const trimmedPrior = priorText.trim()
      if (!trimmedPrior) return trimmedFull
      if (!fullText.startsWith(priorText)) return trimmedFull
      const appended = fullText.slice(priorText.length).trim()
      return appended || trimmedFull
    },
    []
  )

  React.useEffect(() => {
    refreshHistoryFromMessages()
  }, [greetingEnabled, refreshHistoryFromMessages])

  const getCompareBranchMessageIds = (
    items: Message[],
    clusterId: string,
    modelId: string
  ) => {
    const userIndex = items.findIndex(
      (message) =>
        message.messageType === "compare:user" &&
        message.clusterId === clusterId
    )
    if (userIndex === -1) {
      return []
    }

    const messageIds = new Set<string>()
    items.forEach((message, index) => {
      if (!message.id) {
        return
      }
      if (index < userIndex) {
        if (shouldIncludeMessageForModel(message, modelId)) {
          messageIds.add(message.id)
        }
        return
      }
      if (message.clusterId !== clusterId) {
        return
      }
      if (message.messageType === "compare:user") {
        messageIds.add(message.id)
        return
      }
      if (shouldIncludeMessageForModel(message, modelId)) {
        messageIds.add(message.id)
      }
    })

    return Array.from(messageIds)
  }

  const validateBeforeSubmitFn = () => {
    const effectiveSelectedModel = getEffectiveSelectedModel()
    if (compareModeActive) {
      const maxModels =
        typeof compareMaxModels === "number" && compareMaxModels > 0
          ? compareMaxModels
          : MAX_COMPARE_MODELS

      if (!compareSelectedModels || compareSelectedModels.length === 0) {
        notification.error({
          message: t("error"),
          description: t(
            "playground:composer.validationCompareSelectModels",
            "Select at least one model to use in Compare mode."
          )
        })
        return false
      }
      if (compareSelectedModels.length > maxModels) {
        notification.error({
          message: t("error"),
          description: t(
            "playground:composer.compareMaxModels",
            "You can compare up to {{limit}} models per turn.",
            { limit: maxModels }
          )
        })
        return false
      }
      return true
    }
    return validateBeforeSubmit(effectiveSelectedModel || "", t, notification)
  }

  const onSubmit = async ({
    message,
    image,
    isRegenerate = false,
    messages: chatHistory,
    memory,
    controller,
    isContinue,
    docs,
    regenerateFromMessage,
    imageBackendOverride,
    userMessageType,
    assistantMessageType,
    imageGenerationRequest,
    imageGenerationRefine,
    imageGenerationPromptMode,
    imageGenerationSource,
    imageEventSyncPolicy,
    messageSteeringOverride,
    requestOverrides,
    continueOutputTarget = "chat",
    serverChatIdOverride,
    researchContext
  }: {
    message: string
    image: string
    isRegenerate?: boolean
    isContinue?: boolean
    messages?: Message[]
    memory?: ChatHistory
    controller?: AbortController
    docs?: ChatDocuments
    regenerateFromMessage?: Message
    imageBackendOverride?: string
    userMessageType?: string
    assistantMessageType?: string
    imageGenerationRequest?: Partial<ImageGenerationRequestSnapshot>
    imageGenerationRefine?: ImageGenerationRefineMetadata
    imageGenerationPromptMode?: ImageGenerationPromptMode
    imageGenerationSource?: "slash-command" | "generate-modal" | "message-regen"
    imageEventSyncPolicy?: ImageGenerationEventSyncPolicy
    messageSteeringOverride?: Partial<MessageSteeringState> | null
    requestOverrides?: ChatModeOverrides
    continueOutputTarget?: "chat" | "composer_input"
    serverChatIdOverride?: string | null
    researchContext?: ChatResearchContext
  }) => {
    const effectiveSelectedModel = getEffectiveSelectedModel(
      requestOverrides?.selectedModel
    )
    setStreaming(true)
    const trimmedImageBackendOverride =
      typeof imageBackendOverride === "string"
        ? imageBackendOverride.trim()
        : ""
    let signal: AbortSignal
    if (!controller) {
      const newController = new AbortController()
      signal = newController.signal
      setAbortController(newController)
    } else {
      setAbortController(controller)
      signal = controller.signal
    }

    const messageSteeringForTurn = messageSteeringOverride
      ? resolveMessageSteering({
          mode: messageSteeringOverride.mode ?? messageSteeringMode,
          forceNarrate:
            messageSteeringOverride.forceNarrate ?? messageSteeringForceNarrate
        })
      : resolvedMessageSteering
    if (messageSteeringForTurn.hadConflict) {
      notification.warning({
        message: t("warning", { defaultValue: "Warning" }),
        description: t(
          "playground:composer.steering.conflictResolved",
          "Impersonate user overrides Continue as user for this response."
        )
      })
    }
    const shouldConsumeSteering = hasActiveMessageSteering(
      messageSteeringForTurn
    )
    let steeringApplied = false
    const markSteeringApplied = () => {
      if (shouldConsumeSteering) {
        steeringApplied = true
      }
    }

    const chatModeParams = await buildChatModeParams({
      ...(requestOverrides ?? {}),
      selectedModel: effectiveSelectedModel,
      messageSteering: messageSteeringForTurn,
      userMessageType,
      assistantMessageType,
      imageGenerationRequest,
      imageGenerationRefine,
      imageGenerationPromptMode,
      imageGenerationSource,
      imageEventSyncPolicy,
      researchContext
    })
    const baseMessages = chatHistory || messages
    const baseHistory = memory || history
    const capturedReplyTargetId = replyTarget?.id ?? null
    const replyActive =
      Boolean(replyTarget) &&
      !compareModeActive &&
      !isRegenerate &&
      !isContinue &&
      !selectedCharacter?.id
    const replyOverrides = replyActive
      ? (() => {
          const userMessageId = generateID()
          const assistantMessageId = generateID()
          return {
            userMessageId,
            assistantMessageId,
            userParentMessageId: replyTarget?.id ?? null,
            assistantParentMessageId: userMessageId
          }
        })()
      : {}
    const chatModeParamsWithReply = replyActive
      ? { ...chatModeParams, ...replyOverrides }
      : chatModeParams
    const chatModeParamsWithRegen = {
      ...chatModeParamsWithReply,
      regenerateFromMessage: isRegenerate ? regenerateFromMessage : undefined
    }

    try {
      if (isContinue) {
        const continueMessages = chatHistory || messages
        const continueHistory = memory || history
        const continueTargetMessage =
          continueMessages[continueMessages.length - 1]
        const priorAssistantText = continueTargetMessage?.message || ""
        const priorHistorySnapshot = continueHistory.map((entry) => ({
          ...entry
        }))

        markSteeringApplied()
        await continueChatMode(
          continueMessages,
          continueHistory,
          signal,
          chatModeParams
        )

        if (continueOutputTarget === "composer_input") {
          const currentMessages = messagesRef.current
          const continuedMessage = continueTargetMessage?.id
            ? currentMessages.find((entry) => entry.id === continueTargetMessage.id)
            : currentMessages[currentMessages.length - 1]
          const continuedText = continuedMessage?.message || ""
          const continuationDraft = extractContinuationDraft(
            continuedText,
            priorAssistantText
          )

          setSelectedQuickPrompt(continuationDraft)

          if (continueTargetMessage?.id) {
            const targetId = continueTargetMessage.id
            setMessages((prev) =>
              prev.map((entry) =>
                entry.id === targetId
                  ? updateActiveVariant(entry, { message: priorAssistantText })
                  : entry
              )
            )
          } else {
            setMessages((prev) => {
              if (prev.length === 0) return prev
              const next = [...prev]
              const lastIndex = next.length - 1
              next[lastIndex] = updateActiveVariant(next[lastIndex], {
                message: priorAssistantText
              })
              return next
            })
          }

          setHistory(priorHistorySnapshot)

          const resolvedHistoryId =
            typeof chatModeParams.historyId === "string" &&
            chatModeParams.historyId.length > 0
              ? chatModeParams.historyId
              : historyId
          if (
            resolvedHistoryId &&
            resolvedHistoryId !== "temp" &&
            continueTargetMessage?.id
          ) {
            await updateMessage(
              resolvedHistoryId,
              continueTargetMessage.id,
              priorAssistantText
            ).catch(() => null)
          }
        }

        return
      }

      const hasExplicitImageBackend = trimmedImageBackendOverride.length > 0
      const imageBackendCandidates = hasExplicitImageBackend
        ? [trimmedImageBackendOverride]
        : resolveImageBackendCandidates(
            currentChatModelSettings?.apiProvider,
            effectiveSelectedModel
          )
      if (hasExplicitImageBackend || imageBackendCandidates.length > 0) {
        const resolvedImageModelLabel = hasExplicitImageBackend
          ? trimmedImageBackendOverride ||
            (effectiveSelectedModel || "").trim() ||
            currentChatModelSettings?.apiProvider ||
            "image-generation"
          : (effectiveSelectedModel || "").trim() ||
            currentChatModelSettings?.apiProvider ||
            "image-generation"
        const enhancedChatModeParams = {
          ...chatModeParamsWithRegen,
          selectedModel: resolvedImageModelLabel,
          uploadedFiles: hasExplicitImageBackend ? [] : uploadedFiles,
          imageBackendOverride: hasExplicitImageBackend
            ? trimmedImageBackendOverride
            : undefined
        }
        await normalChatMode(
          message,
          image,
          isRegenerate,
          baseMessages,
          baseHistory,
          signal,
          enhancedChatModeParams
        )
        return
      }
      // console.log("contextFiles", contextFiles)
      if (contextFiles.length > 0) {
        markSteeringApplied()
        await documentChatMode(
          message,
          image,
          isRegenerate,
          chatHistory || messages,
          memory || history,
          signal,
          contextFiles,
          chatModeParamsWithRegen
        )
        // setFileRetrievalEnabled(false)
        return
      }

      if (docs?.length > 0 || documentContext?.length > 0) {
        const processingTabs = docs || documentContext || []

        if (docs?.length > 0) {
          setDocumentContext(
            Array.from(new Set([...(documentContext || []), ...docs]))
          )
        }
        markSteeringApplied()
        await tabChatMode(
          message,
          image,
          processingTabs,
          isRegenerate,
          chatHistory || messages,
          memory || history,
          signal,
          chatModeParamsWithRegen
        )
        return
      }

      const hasScopedRagMediaIds =
        Array.isArray(ragMediaIds) && ragMediaIds.length > 0
      const shouldUseRag =
        Boolean(selectedKnowledge) ||
        (fileRetrievalEnabled && hasScopedRagMediaIds)
      if (shouldUseRag) {
        markSteeringApplied()
        await ragMode(
          message,
          image,
          isRegenerate,
          chatHistory || messages,
          memory || history,
          signal,
          chatModeParamsWithRegen
        )
      } else {
        // Include uploaded files info even in normal mode
        const enhancedChatModeParams = {
          ...chatModeParamsWithRegen,
          uploadedFiles: uploadedFiles
        }
        const baseMessages = chatHistory || messages
        const baseHistory = memory || history

        if (!compareModeActive) {
          const resolvedSelectedCharacter = await resolveSelectedCharacter()
          if (resolvedSelectedCharacter?.id) {
            const resolvedModel = effectiveSelectedModel?.trim()
            if (!resolvedModel) {
              notification.error({
                message: t("error"),
                description: t("validationSelectModel")
              })
              setIsProcessing(false)
              setStreaming(false)
              setAbortController(null)
              return
            }
            markSteeringApplied()
            await characterChatMode({
              message,
              image,
              isRegenerate,
              messages: baseMessages,
              history: baseHistory,
              signal,
              model: resolvedModel,
              regenerateFromMessage,
              character: resolvedSelectedCharacter,
              messageSteering: messageSteeringForTurn,
              serverChatIdOverride
            })
            return
          }

          if (isPersonaAssistantSelection(selectedAssistant)) {
            const resolvedModel = effectiveSelectedModel?.trim()
            if (!resolvedModel) {
              notification.error({
                message: t("error"),
                description: t("validationSelectModel")
              })
              setIsProcessing(false)
              setStreaming(false)
              setAbortController(null)
              return
            }

            const personaServerChat = await ensurePersonaServerChatWithState({
              assistant: selectedAssistant,
              serverChatIdOverride
            })
            const assistantIdentity = {
              name: selectedAssistant.name,
              avatarUrl:
                typeof selectedAssistant.avatar_url === "string"
                  ? selectedAssistant.avatar_url
                  : undefined
            }
            markSteeringApplied()
            await normalChatMode(
              message,
              image,
              isRegenerate,
              baseMessages,
              baseHistory,
              signal,
              {
                ...enhancedChatModeParams,
                assistantIdentity,
                historyId: personaServerChat.historyId,
                serverChatId: personaServerChat.chatId,
                saveMessageOnSuccess: (data: SaveMessageData) =>
                  saveMessageOnSuccess({
                    ...data,
                    conversationId: personaServerChat.chatId
                  })
              }
            )
            return
          }
        }

        if (!compareModeActive) {
          markSteeringApplied()
          await normalChatMode(
            message,
            image,
            isRegenerate,
            baseMessages,
            baseHistory,
            signal,
            enhancedChatModeParams
          )
        } else {
          const maxModels =
            typeof compareMaxModels === "number" && compareMaxModels > 0
              ? compareMaxModels
              : MAX_COMPARE_MODELS

          const modelsRaw =
            compareSelectedModels && compareSelectedModels.length > 0
              ? compareSelectedModels
              : effectiveSelectedModel
                ? [effectiveSelectedModel]
                : []
          if (modelsRaw.length === 0) {
            throw new Error("No models selected for Compare mode")
          }
          const uniqueModels = Array.from(new Set(modelsRaw))
          const models =
            uniqueModels.length > maxModels
              ? uniqueModels.slice(0, maxModels)
              : uniqueModels

          if (uniqueModels.length > maxModels) {
            notification.warning({
              message: t("error"),
              description: t(
                "playground:composer.compareMaxModelsTrimmed",
                "Compare is limited to {{limit}} models per turn. Using the first {{limit}} selected models.",
                { count: maxModels, limit: maxModels }
              )
            })
          }
          const clusterId = generateID()
          const compareUserMessageId = generateID()
          const lastMessage = baseMessages[baseMessages.length - 1]
          const compareUserParentMessageId = lastMessage?.id || null
          const resolvedImage =
            image.length > 0
              ? image.startsWith("data:")
                ? image
                : image.includes(",")
                  ? `data:image/jpeg;base64,${image.split(",")[1]}`
                  : `data:image/jpeg;base64,${image}`
              : ""
          const compareUserMessage: Message = {
            isBot: false,
            name: "You",
            message,
            sources: [],
            images: resolvedImage ? [resolvedImage] : [],
            createdAt: Date.now(),
            id: compareUserMessageId,
            messageType: "compare:user",
            clusterId,
            parentMessageId: compareUserParentMessageId,
            documents:
              uploadedFiles?.map((file) => ({
                type: "file",
                filename: file.filename,
                fileSize: file.size,
                processed: file.processed
              })) || []
          }

          setMessages((prev) => [...prev, compareUserMessage])
          setHistory((prev) => [
            ...prev,
            {
              role: "user" as const,
              content: compareUserMessage.message,
              image: compareUserMessage.images?.[0],
              messageType: compareUserMessage.messageType
            }
          ])

          let activeHistoryId = historyId
          if (temporaryChat) {
            if (historyId !== "temp") {
              setHistoryId("temp")
            }
            activeHistoryId = "temp"
          } else if (!activeHistoryId) {
            const title = await generateTitle(
              uniqueModels[0] || effectiveSelectedModel || "",
              message,
              message
            )
            const compareTitle = buildCompareHistoryTitle(title)
            const newHistory = await saveHistory(compareTitle, false, "web-ui")
            updatePageTitle(compareTitle)
            activeHistoryId = newHistory.id
            setHistoryId(newHistory.id)
            markCompareHistoryCreated(newHistory.id)
          }

          if (!temporaryChat && activeHistoryId) {
            await saveMessage({
              id: compareUserMessageId,
              history_id: activeHistoryId,
              name: effectiveSelectedModel || uniqueModels[0] || "You",
              role: "user",
              content: message,
              images: resolvedImage ? [resolvedImage] : [],
              time: 1,
              message_type: "compare:user",
              clusterId,
              parent_message_id: compareUserParentMessageId,
              documents:
                uploadedFiles?.map((file) => ({
                  type: "file",
                  filename: file.filename,
                  fileSize: file.size,
                  processed: file.processed
                })) || []
            })
          }

          setIsProcessing(true)

          const compareChatModeParams = await buildChatModeParams({
            historyId: activeHistoryId,
            setHistory: () => {},
            setStreaming: () => {},
            setIsProcessing: () => {},
            setAbortController: () => {},
            messageSteering: messageSteeringForTurn
          })
          const compareEnhancedParams = {
            ...compareChatModeParams,
            uploadedFiles: uploadedFiles
          }

          const comparePromises = models.map((modelId) => {
            const historyForModel = buildHistoryForModel(baseMessages, modelId)
            return normalChatMode(
              message,
              image,
              true,
              baseMessages,
              baseHistory,
              signal,
              {
                ...compareEnhancedParams,
                selectedModel: modelId,
                clusterId,
                assistantMessageType: "compare:reply",
                modelIdOverride: modelId,
                assistantParentMessageId: compareUserMessageId,
                historyForModel
              }
            ).catch((e) => {
              const errorMessage =
                e instanceof Error
                  ? e.message
                  : t("somethingWentWrong")
              notification.error({
                message: t("error"),
                description: errorMessage
              })
            })
          })

          markSteeringApplied()
          await Promise.allSettled(comparePromises)
          refreshHistoryFromMessages()
          setIsProcessing(false)
          setStreaming(false)
          setAbortController(null)
        }
      }
    } catch (e) {
      const errorMessage =
        e instanceof Error ? e.message : t("somethingWentWrong")
      notification.error({
        message: t("error"),
        description: errorMessage
      })
      setIsProcessing(false)
      setStreaming(false)
    } finally {
      if (replyActive && capturedReplyTargetId != null) {
        const currentReplyTarget = useStoreMessageOption.getState().replyTarget
        if (currentReplyTarget?.id === capturedReplyTargetId) {
          clearReplyTarget()
        }
      }
      if (steeringApplied) {
        clearMessageSteering()
      }
    }
  }

  const sendPerModelReply = async ({
    clusterId,
    modelId,
    message
  }: {
    clusterId: string
    modelId: string
    message: string
  }) => {
    const trimmed = message.trim()
    if (!trimmed) {
      return
    }

    if (!compareFeatureEnabled) {
      notification.error({
        message: t("error"),
        description: t(
          "playground:composer.compareDisabled",
          "Compare mode is disabled in settings."
        )
      })
      return
    }

    const messageSteeringForTurn = resolvedMessageSteering
    const shouldConsumeSteering = hasActiveMessageSteering(
      messageSteeringForTurn
    )

    setStreaming(true)
    const newController = new AbortController()
    setAbortController(newController)
    const signal = newController.signal

    const baseMessages = messages
    const baseHistory = history
    const userMessageId = generateID()
    const assistantMessageId = generateID()
    const userParentMessageId = getLastThreadMessageId(
      baseMessages,
      clusterId,
      modelId
    )

    try {
      const chatModeParams = await buildChatModeParams({
        messageSteering: messageSteeringForTurn
      })
      const enhancedChatModeParams = {
        ...chatModeParams,
        uploadedFiles: uploadedFiles
      }
      const historyForModel = buildHistoryForModel(baseMessages, modelId)
      const perModelOverrides = {
        selectedModel: modelId,
        clusterId,
        userMessageType: "compare:perModelUser",
        assistantMessageType: "compare:reply",
        modelIdOverride: modelId,
        userMessageId,
        assistantMessageId,
        userParentMessageId,
        assistantParentMessageId: userMessageId,
        historyForModel
      }

      if (contextFiles.length > 0) {
        await documentChatMode(
          trimmed,
          "",
          false,
          baseMessages,
          baseHistory,
          signal,
          contextFiles,
          {
            ...chatModeParams,
            ...perModelOverrides
          }
        )
        return
      }

      if (documentContext && documentContext.length > 0) {
        await tabChatMode(
          trimmed,
          "",
          documentContext,
          false,
          baseMessages,
          baseHistory,
          signal,
          {
            ...chatModeParams,
            ...perModelOverrides
          }
        )
        return
      }

      const hasScopedRagMediaIds =
        Array.isArray(ragMediaIds) && ragMediaIds.length > 0
      const shouldUseRag =
        Boolean(selectedKnowledge) ||
        (fileRetrievalEnabled && hasScopedRagMediaIds)
      if (shouldUseRag) {
        await ragMode(
          trimmed,
          "",
          false,
          baseMessages,
          baseHistory,
          signal,
          {
            ...chatModeParams,
            ...perModelOverrides
          }
        )
        return
      }

      await normalChatMode(
        trimmed,
        "",
        false,
        baseMessages,
        baseHistory,
        signal,
        {
          ...enhancedChatModeParams,
          ...perModelOverrides
        }
      )
    } catch (e) {
      const errorMessage =
        e instanceof Error ? e.message : t("somethingWentWrong")
      notification.error({
        message: t("error"),
        description: errorMessage
      })
    } finally {
      setStreaming(false)
      setIsProcessing(false)
      setAbortController(null)
      if (shouldConsumeSteering) {
        clearMessageSteering()
      }
    }
  }

  const createChatBranch = createBranchMessage({
    notification,
    historyId,
    setHistory,
    setHistoryId: setHistoryId as (id: string | null) => void,
    setMessages,
    setContext: setContextFiles,
    setSelectedSystemPrompt,
    setSystemPrompt: currentChatModelSettings.setSystemPrompt,
    serverChatId,
    setServerChatId,
    setServerChatTitle,
    setServerChatCharacterId,
    setServerChatMetaLoaded,
    serverChatState,
    setServerChatState,
    setServerChatVersion,
    serverChatTopic,
    setServerChatTopic,
    serverChatClusterId,
    setServerChatClusterId,
    serverChatSource,
    setServerChatSource,
    serverChatExternalRef,
    setServerChatExternalRef,
    onServerChatMutated: invalidateServerChatHistory,
    characterId: serverChatCharacterId ?? null,
    chatTitle: serverChatTitle ?? null,
    messages,
    history
  })

  const createServerOnlyChatBranch = createBranchMessage({
    notification,
    historyId,
    setHistory,
    setHistoryId: setHistoryId as (id: string | null) => void,
    setMessages,
    setContext: setContextFiles,
    setSelectedSystemPrompt,
    setSystemPrompt: currentChatModelSettings.setSystemPrompt,
    serverChatId,
    setServerChatId,
    setServerChatTitle,
    setServerChatCharacterId,
    setServerChatMetaLoaded,
    serverChatState,
    setServerChatState,
    setServerChatVersion,
    serverChatTopic,
    setServerChatTopic,
    serverChatClusterId,
    setServerChatClusterId,
    serverChatSource,
    setServerChatSource,
    serverChatExternalRef,
    setServerChatExternalRef,
    onServerChatMutated: invalidateServerChatHistory,
    characterId: serverChatCharacterId ?? null,
    chatTitle: serverChatTitle ?? null,
    messages,
    history,
    serverOnly: true
  })

  const regenerateLastMessage = createRegenerateLastMessage({
    validateBeforeSubmitFn,
    history,
    messages,
    setHistory,
    setMessages,
    onSubmit,
    beforeSubmit: async ({ nextMessages }) => {
      if (!serverChatId) return
      if (selectedCharacter?.id == null && serverChatCharacterId == null) return

      const branchIndex = nextMessages.length - 1
      if (branchIndex < 0) return

      const branchedChatId = await createServerOnlyChatBranch(branchIndex)
      if (!branchedChatId) {
        throw new Error("Failed to create branch for regeneration")
      }

      return {
        submitExtras: {
          serverChatIdOverride: branchedChatId
        }
      }
    }
  })

  const stopStreamingRequest = React.useCallback(
    (options?: unknown) => {
      if (!abortController) {
        return
      }

      const discardTurn =
        typeof options === "object" &&
        options !== null &&
        "discardTurn" in options &&
        options.discardTurn === true

      if (discardTurn) {
        discardCurrentTurnOnAbortRef.current = true
      }

      abortController.abort()
      setAbortController(null)
    },
    [abortController, setAbortController]
  )

  const editMessage = createEditMessage({
    messages,
    history,
    setMessages,
    setHistory,
    historyId,
    validateBeforeSubmitFn,
    onSubmit
  })

  const deleteMessage = React.useCallback(
    async (index: number) => {
      const target = messages[index]
      if (!target) return

      // Capture values synchronously before any awaits
      const targetId = target.serverMessageId ?? target.id
      const serverMessageId = target.serverMessageId
      const serverMessageVersion = target.serverMessageVersion
      const historyRole = target.role ?? (target.isBot ? "assistant" : "user")
      const historyContent = target.message ?? ""

      if (replyTarget?.id && targetId && replyTarget.id === targetId) {
        clearReplyTarget()
      }

      try {
        if (serverMessageId) {
          await tldwClient.initialize().catch(() => null)
          let expectedVersion = serverMessageVersion
          if (expectedVersion == null) {
            const serverMessage = await tldwClient.getMessage(serverMessageId)
            expectedVersion = serverMessage?.version
          }
          if (expectedVersion == null) {
            throw new Error("Missing server message version")
          }
          await tldwClient.deleteMessage(
            serverMessageId,
            Number(expectedVersion),
            serverChatId ?? undefined
          )
          invalidateServerChatHistory()
        }

        if (historyId) {
          await removeMessageByIndex(historyId, index)
        }
      } catch (err) {
        console.error("[deleteMessage] Failed to delete message", err)
        return
      }

      setMessages((prev) => prev.filter((m) => m.id !== targetId))
      setHistory((prev) => {
        let removed = false
        return prev.filter((h) => {
          if (!removed && h.role === historyRole && h.content === historyContent) {
            removed = true
            return false
          }
          return true
        })
      })
    },
    [
      clearReplyTarget,
      historyId,
      invalidateServerChatHistory,
      messages,
      replyTarget?.id,
      serverChatId,
      setHistory,
      setMessages
    ]
  )

  const toggleMessagePinned = React.useCallback(
    async (index: number) => {
      const target = messages[index]
      if (!target) return

      // Capture values synchronously before any awaits
      const targetId = target.id
      const nextPinned = !Boolean(target.pinned)
      const serverMessageId = target.serverMessageId
      const serverMessageVersion = target.serverMessageVersion
      const messageText = String(target.message || "")

      try {
        if (serverMessageId) {
          await tldwClient.initialize().catch(() => null)
          let expectedVersion = serverMessageVersion
          if (expectedVersion == null) {
            const serverMessage = await tldwClient.getMessage(serverMessageId)
            expectedVersion = serverMessage?.version
          }
          if (expectedVersion == null) {
            throw new Error("Missing server message version")
          }
          await tldwClient.editMessage(
            serverMessageId,
            messageText,
            Number(expectedVersion),
            serverChatId ?? undefined,
            { pinned: nextPinned }
          )
          invalidateServerChatHistory()
        }
      } catch (err) {
        console.error("[toggleMessagePinned] Failed to toggle pin", err)
        return
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === targetId ? { ...m, pinned: nextPinned } : m
        )
      )
    },
    [invalidateServerChatHistory, messages, serverChatId, setMessages]
  )

  const createCompareBranch = async ({
    clusterId,
    modelId,
    open = true
  }: {
    clusterId: string
    modelId: string
    open?: boolean
  }): Promise<string | null> => {
    if (!historyId || historyId === "temp") {
      return null
    }

    const messageIds = getCompareBranchMessageIds(messages, clusterId, modelId)
    if (messageIds.length === 0) {
      return null
    }

    try {
      const newBranch = await generateBranchFromMessageIds(
        historyId,
        messageIds
      )
      if (!newBranch) {
        return null
      }

      const splitTitle = buildCompareSplitTitle(newBranch.history.title || "")
      await updateHistory(newBranch.history.id, splitTitle)

      void trackCompareMetric({ type: "split_single" })

      if (open) {
        setHistory(formatToChatHistory(newBranch.messages))
        setMessages(formatToMessage(newBranch.messages))
        setHistoryId(newBranch.history.id)
        const systemFiles = await getSessionFiles(newBranch.history.id)
        setContextFiles(systemFiles)

        const lastUsedPrompt = newBranch?.history?.last_used_prompt
        if (lastUsedPrompt) {
          if (lastUsedPrompt.prompt_id) {
            const prompt = await getPromptById(lastUsedPrompt.prompt_id)
            if (prompt) {
              setSelectedSystemPrompt(lastUsedPrompt.prompt_id)
            }
          }
          if (currentChatModelSettings?.setSystemPrompt) {
            currentChatModelSettings.setSystemPrompt(
              lastUsedPrompt.prompt_content
            )
          }
        }
      }

      return newBranch.history.id
    } catch (e) {
      console.log("[compare-branch] failed", e)
      return null
    }
  }

  return {
    onSubmit,
    sendPerModelReply,
    regenerateLastMessage,
    stopStreamingRequest,
    editMessage,
    deleteMessage,
    toggleMessagePinned,
    createChatBranch,
    createCompareBranch
  }
}
