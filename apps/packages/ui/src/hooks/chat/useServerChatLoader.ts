import { excludeLocalRagDiagnostics } from "@/utils/local-rag-diagnostic"
import { waitForChatPromotion } from "@/services/pending-chat-promotion"
import React from "react"
import { formatToMessage } from "@/db/dexie/helpers"
import { shallow } from "zustand/shallow"
import type { TFunction } from "i18next"
import { useChatBaseState } from "@/hooks/chat/useChatBaseState"
import { useStoreMessageOption } from "@/store/option"
import type { Message } from "@/store/option"
import {
  tldwClient,
  type ServerChatMessage
} from "@/services/tldw/TldwApiClient"
import { reconcileServerChatMessages, reconcileServerChatMirror, serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { watchServerChatLoadAuthority } from "@/services/server-chat-load-authority"
import { getSelectedAssistantOperationRevision, useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { syncChatSettingsForServerChat } from "@/services/chat-settings"
import { validateCachedServerChatId } from "@/store/workspace-sync-contract"
import type { ChatScope } from "@/types/chat-scope"
import { normalizeConversationState } from "@/utils/conversation-state"
import { normalizeChatRole } from "@/utils/normalize-chat-role"
import { updatePageTitle } from "@/utils/update-page-title"
import {
  effectiveAssistantStateToSelection,
  resolveEffectiveAssistantState
} from "@/hooks/chat/effective-assistant-state"
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  parseImageGenerationEventMirrorContent
} from "@/utils/image-generation-chat"
import { normalizeMessageMetadataExtra } from "@/utils/dynamic-ui"
import {
  characterToAssistantSelection,
  personaToAssistantSelection
} from "@/types/assistant-selection"
import { isTrackedCharacterChatSource } from "@/utils/character-chat-session"

type NotificationApi = {
  error: (payload: { message: string; description?: string }) => void
}

type UseServerChatLoaderOptions = {
  ensureServerChatHistoryId: (
    chatId: string,
    title?: string,
    scopeInvalidatedSignal?: AbortSignal,
    snapshot?: ServicePromptSnapshot
  ) => Promise<string | null>
  notification: NotificationApi
  t: TFunction
  scope?: ChatScope
}

type PreserveLocalMessagesArgs = {
  currentMessages: Message[]
  serverMessages: Message[]
  isStreaming: boolean
  isProcessing: boolean
}

type ShouldSkipLoadedServerChatReloadArgs = {
  activeServerChatId: string | null
  loadedChatId: string | null
  loaded: boolean
  currentMessages: Message[]
}

type FetchServerChatMessagesPageArgs = {
  limit: number
  offset: number
  signal?: AbortSignal
}

type FetchServerChatMessagesPage = (
  params: FetchServerChatMessagesPageArgs
) => Promise<ServerChatMessage[]>

const SERVER_CHAT_MESSAGES_FETCH_LIMIT = 200
const SERVER_CHAT_MESSAGES_FETCH_MAX_PAGES = 100
const SERVER_CHAT_IMAGES_MAX_ENCODED_CHARS = 64 * 1024 * 1024

const toServerMessageId = (message: Message): string | null => {
  if (typeof message.serverMessageId !== "string") return null
  const trimmed = message.serverMessageId.trim()
  return trimmed.length > 0 ? trimmed : null
}

const toServerMessageCreatedAtMs = (message: ServerChatMessage): number | null => {
  if (typeof message?.created_at !== "string") return null
  const parsed = Date.parse(message.created_at)
  return Number.isNaN(parsed) ? null : parsed
}

const isSyntheticGreetingPlaceholder = (message: Message): boolean => {
  const messageType = message.messageType
  if (messageType !== "character:greeting" && messageType !== "greeting") {
    return false
  }
  return (
    !toServerMessageId(message) &&
    Boolean(message.isBot || message.role === "assistant") &&
    typeof message.message === "string" &&
    message.message.trim().length > 0
  )
}

export const shouldPreserveLocalMessagesForServerLoad = ({
  currentMessages,
  serverMessages,
  isStreaming,
  isProcessing
}: PreserveLocalMessagesArgs): boolean => {
  if (isStreaming || isProcessing) return true
  if (!Array.isArray(currentMessages) || currentMessages.length === 0) return false

  const serverMessageIds = new Set(
    serverMessages.map((message) => toServerMessageId(message)).filter(Boolean)
  )

  const hasUnsyncedMessages = currentMessages.some(
    (message) =>
      !toServerMessageId(message) &&
      !serverMessageIds.has(String(message.id || "")) &&
      !isSyntheticGreetingPlaceholder(message) &&
      typeof message.message === "string" &&
      message.message.trim().length > 0
  )
  if (hasUnsyncedMessages) return true

  return currentMessages.some((message) => {
    const serverMessageId = toServerMessageId(message)
    return Boolean(serverMessageId) && !serverMessageIds.has(serverMessageId)
  })
}

export const shouldSkipLoadedServerChatReload = ({
  activeServerChatId,
  loadedChatId,
  loaded,
  currentMessages
}: ShouldSkipLoadedServerChatReloadArgs): boolean => {
  if (!activeServerChatId || !loaded) return false
  if (loadedChatId !== activeServerChatId) return false
  return Array.isArray(currentMessages) && currentMessages.length > 0
}

export const shouldCommitServerChatLoadResult = ({
  requestedChatId,
  activeServerChatId,
  requestController,
  activeController
}: {
  requestedChatId: string | null
  activeServerChatId: string | null
  requestController: AbortController | null
  activeController: AbortController | null
}): boolean => {
  if (!requestedChatId || !activeServerChatId) return false
  if (requestedChatId !== activeServerChatId) return false
  if (requestController?.signal.aborted) return false
  return requestController != null && requestController === activeController
}

export const fetchAllServerChatMessages = async (
  fetchPage: FetchServerChatMessagesPage,
  options?: {
    limit?: number
    maxPages?: number
    signal?: AbortSignal
  }
): Promise<ServerChatMessage[]> => {
  const limit = Math.max(
    1,
    Math.min(200, options?.limit ?? SERVER_CHAT_MESSAGES_FETCH_LIMIT)
  )
  const maxPages = Math.max(
    1,
    options?.maxPages ?? SERVER_CHAT_MESSAGES_FETCH_MAX_PAGES
  )
  const messages: ServerChatMessage[] = []
  let offset = 0
  let imageCharacters = 0

  for (let page = 0; page < maxPages; page += 1) {
    options?.signal?.throwIfAborted()
    const batch = await fetchPage({
      limit,
      offset,
      signal: options?.signal
    })
    options?.signal?.throwIfAborted()
    if (!Array.isArray(batch) || batch.length === 0) {
      break
    }
    imageCharacters += batch.reduce((total, message) => total + (message.images || []).reduce((size, image) => size + image.length, 0), 0)
    if (imageCharacters > SERVER_CHAT_IMAGES_MAX_ENCODED_CHARS) {
      throw new Error("Saved chat attachments exceed the image history limit. Local work has been retained.")
    }
    messages.push(...batch)
    offset += batch.length
    if (batch.length < limit) {
      break
    }
  }

  if (messages.length <= 1) {
    return messages
  }

  const deduped: ServerChatMessage[] = []
  const seenIds = new Set<string>()
  for (const message of messages) {
    const normalizedId = String(message?.id ?? "").trim()
    if (normalizedId.length > 0) {
      if (seenIds.has(normalizedId)) {
        continue
      }
      seenIds.add(normalizedId)
    }
    deduped.push(message)
  }

  return deduped
    .map((message, index) => ({
      index,
      message,
      createdAtMs: toServerMessageCreatedAtMs(message)
    }))
    .sort((left, right) => {
      const leftCreatedAt = left.createdAtMs
      const rightCreatedAt = right.createdAtMs
      if (leftCreatedAt != null && rightCreatedAt != null) {
        if (leftCreatedAt !== rightCreatedAt) {
          return leftCreatedAt - rightCreatedAt
        }
      } else if (leftCreatedAt != null) {
        return -1
      } else if (rightCreatedAt != null) {
        return 1
      }
      return left.index - right.index
    })
    .map((entry) => entry.message)
}

const resolveAssistantId = (value: unknown): string | null => {
  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed.length > 0 ? trimmed : null
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value)
  }
  return null
}

const isMissingServerChatReferenceError = (error: unknown): boolean => {
  const candidate = error as
    | {
        status?: unknown
        response?: { status?: unknown }
        message?: unknown
      }
    | null
    | undefined
  const rawStatus = candidate?.status ?? candidate?.response?.status
  const statusCode =
    typeof rawStatus === "number"
      ? rawStatus
      : typeof rawStatus === "string"
        ? Number.parseInt(rawStatus, 10)
        : Number.NaN

  if (statusCode === 404) return true

  const message = String(candidate?.message || "").toLowerCase()
  return message.includes("404") || message.includes("not found")
}

const isDeniedServerChatError = (error: unknown): boolean => {
  const candidate = error as { status?: unknown; response?: { status?: unknown } } | null
  const status = Number(candidate?.status ?? candidate?.response?.status)
  return status === 401 || status === 403
}

export const resolveServerChatAssistantIdentity = (
  chat: Record<string, unknown> | null | undefined
): {
  assistantKind: "character" | "persona" | null
  assistantId: string | null
  characterId: string | number | null
  personaMemoryMode: "read_only" | "read_write" | null
} => {
  const candidate = chat && typeof chat === "object" ? chat : null
  const assistantKind =
    candidate?.assistant_kind === "character" || candidate?.assistant_kind === "persona"
      ? candidate.assistant_kind
      : null
  const source =
    typeof candidate?.source === "string" ? candidate.source.trim() : null
  const assistantId = resolveAssistantId(candidate?.assistant_id)
  const rawCharacterId =
    candidate?.character_id ??
    candidate?.characterId ??
    null
  const characterId =
    typeof rawCharacterId === "number" && Number.isFinite(rawCharacterId)
      ? rawCharacterId
      : typeof rawCharacterId === "string" && rawCharacterId.trim().length > 0
        ? rawCharacterId
        : null
  const personaMemoryMode =
    candidate?.persona_memory_mode === "read_only" ||
    candidate?.persona_memory_mode === "read_write"
      ? candidate.persona_memory_mode
      : null

  if (assistantKind === "persona" && assistantId) {
    return {
      assistantKind,
      assistantId,
      characterId: null,
      personaMemoryMode
    }
  }

  if (assistantKind === "character" && assistantId) {
    return {
      assistantKind,
      assistantId,
      characterId: characterId ?? assistantId,
      personaMemoryMode
    }
  }

  if (characterId != null && isTrackedCharacterChatSource(source)) {
    return {
      assistantKind: "character",
      assistantId: String(characterId),
      characterId,
      personaMemoryMode: null
    }
  }

  return {
    assistantKind: null,
    assistantId: null,
    characterId: null,
    personaMemoryMode
  }
}

type MapServerMessagesArgs = {
  serverMessages: ServerChatMessage[]
  assistantName: string
  characterId: string | number | null
}

export const mapServerChatMessagesToPlaygroundMessages = ({
  serverMessages,
  assistantName,
  characterId
}: MapServerMessagesArgs): Message[] => {
  let encounteredUserMessage = false
  return serverMessages.map((m) => {
    const meta = m as unknown as Record<string, unknown>
    const createdAt = Date.parse(m.created_at)
    const metadataExtraCandidate =
      (m as unknown as { metadata_extra?: unknown }).metadata_extra ??
      (meta?.metadata_extra as unknown)
    const metadataExtra = normalizeMessageMetadataExtra(metadataExtraCandidate)
    const speakerCharacterIdRaw = metadataExtra?.speaker_character_id
    const speakerCharacterId =
      typeof speakerCharacterIdRaw === "number" &&
      Number.isFinite(speakerCharacterIdRaw)
        ? speakerCharacterIdRaw
        : typeof speakerCharacterIdRaw === "string" &&
            speakerCharacterIdRaw.trim().length > 0 &&
            Number.isFinite(Number(speakerCharacterIdRaw))
          ? Number(speakerCharacterIdRaw)
          : null
    const moodConfidenceRaw = metadataExtra?.mood_confidence
    const moodConfidence =
      typeof moodConfidenceRaw === "number" && Number.isFinite(moodConfidenceRaw)
        ? moodConfidenceRaw
        : typeof moodConfidenceRaw === "string" &&
            moodConfidenceRaw.trim().length > 0 &&
            Number.isFinite(Number(moodConfidenceRaw))
          ? Number(moodConfidenceRaw)
          : null
    const senderName =
      typeof (m as any).sender === "string" ? String((m as any).sender).trim() : ""
    const explicitMessageType =
      (meta?.message_type as string | undefined) ??
      (meta?.messageType as string | undefined)
    const inferredGreetingMessageType =
      !explicitMessageType &&
      characterId != null &&
      m.role === "assistant" &&
      !encounteredUserMessage
        ? "character:greeting"
        : undefined

    const mirroredImageEvent = parseImageGenerationEventMirrorContent(m.content)
    const normalizedMessageType = mirroredImageEvent
      ? IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
      : explicitMessageType ?? inferredGreetingMessageType
    if (m.role === "user") {
      encounteredUserMessage = true
    }

    const mirroredImageDataUrl =
      typeof mirroredImageEvent?.imageDataUrl === "string" &&
      mirroredImageEvent.imageDataUrl.startsWith("data:image/")
        ? mirroredImageEvent.imageDataUrl
        : undefined
    const generationInfo = mirroredImageEvent
      ? {
          file_id: mirroredImageEvent.fileId,
          image_generation: {
            request: mirroredImageEvent.request,
            promptMode: mirroredImageEvent.promptMode,
            source: mirroredImageEvent.source,
            createdAt:
              mirroredImageEvent.createdAt ??
              (Number.isNaN(createdAt) ? Date.now() : createdAt),
            refine: mirroredImageEvent.refine,
            variant_count: mirroredImageEvent.variantCount,
            active_variant_index: mirroredImageEvent.activeVariantIndex,
            event_id: mirroredImageEvent.eventId,
            sync: {
              mode: "on",
              policy: "on",
              status: "synced",
              serverMessageId: String(m.id),
              mirroredAt: Number.isNaN(createdAt) ? Date.now() : createdAt,
              lastAttemptAt: Number.isNaN(createdAt) ? Date.now() : createdAt
            }
          }
        }
      : undefined

    return {
      createdAt: Number.isNaN(createdAt) ? undefined : createdAt,
      isBot: m.role === "assistant",
      role: normalizeChatRole(m.role),
      name:
        m.role === "assistant"
          ? senderName || assistantName
          : m.role === "system"
            ? "System"
            : "You",
      message: mirroredImageEvent || (m.role === "user" && m.version === 1 &&
        metadataExtra?.content_placeholder_reason === "image_attachment" && m.images?.length &&
        m.content === `<Image attachment x${m.images.length}>`) ? "" : m.content,
      sources: [],
      images: mirroredImageDataUrl ? [mirroredImageDataUrl] : m.images || [],
      generationInfo,
      id: String(m.id),
      serverMessageId: String(m.id),
      serverMessageVersion: m.version,
      parentMessageId:
        (meta?.parent_message_id as string | null | undefined) ??
        (meta?.parentMessageId as string | null | undefined) ??
        null,
      messageType: normalizedMessageType,
      clusterId:
        (meta?.cluster_id as string | undefined) ??
        (meta?.clusterId as string | undefined),
      modelId:
        (meta?.model_id as string | undefined) ??
        (meta?.modelId as string | undefined),
      modelName:
        (meta?.model_name as string | undefined) ??
        (meta?.modelName as string | undefined) ??
        assistantName,
      modelImage:
        (meta?.model_image as string | undefined) ??
        (meta?.modelImage as string | undefined),
      metadataExtra,
      speakerCharacterId,
      speakerCharacterName:
        typeof metadataExtra?.speaker_character_name === "string"
          ? metadataExtra.speaker_character_name
          : undefined,
      moodLabel:
        typeof metadataExtra?.mood_label === "string"
          ? metadataExtra.mood_label
          : undefined,
      moodConfidence,
      moodTopic:
        typeof metadataExtra?.mood_topic === "string"
          ? metadataExtra.mood_topic
          : null,
      visualActorKind:
        metadataExtra?.visual_actor_kind === "character" ||
        metadataExtra?.visual_actor_kind === "persona"
          ? metadataExtra.visual_actor_kind
          : null,
      visualActorId:
        typeof metadataExtra?.visual_actor_id === "string" ||
        typeof metadataExtra?.visual_actor_id === "number"
          ? metadataExtra.visual_actor_id
          : null,
      visualPackId:
        typeof metadataExtra?.visual_pack_id === "number"
          ? metadataExtra.visual_pack_id
          : null,
      visualPackVersionId:
        typeof metadataExtra?.visual_pack_version_id === "number"
          ? metadataExtra.visual_pack_version_id
          : null,
      visualExpressionKey:
        typeof metadataExtra?.visual_expression_key === "string"
          ? metadataExtra.visual_expression_key
          : null,
      visualAssetId:
        typeof metadataExtra?.visual_asset_id === "number"
          ? metadataExtra.visual_asset_id
          : null,
      visualAssetUrl:
        typeof metadataExtra?.visual_pack_id === "number" &&
        typeof metadataExtra?.visual_asset_id === "number"
          ? tldwClient.getVisualIdentityAssetContentPath(
              metadataExtra.visual_pack_id,
              metadataExtra.visual_asset_id
            )
          : null,
      visualPreviewUrl:
        typeof metadataExtra?.visual_preview_url === "string"
          ? metadataExtra.visual_preview_url
          : null,
      visualFallbackReason:
        typeof metadataExtra?.visual_fallback_reason === "string"
          ? metadataExtra.visual_fallback_reason
          : null,
      visualIsAnimated:
        typeof metadataExtra?.visual_is_animated === "boolean"
          ? metadataExtra.visual_is_animated
          : null,
      pinned: Boolean(
        (meta?.pinned as boolean | undefined) ??
          (metadataExtra?.pinned as boolean | undefined)
      )
    } satisfies Message
  })
}

type ApplyAssistantPresentationArgs = {
  messages: Message[]
  assistantName: string
  assistantAvatarUrl?: string | null
}

export const applyAssistantPresentationToMessages = ({
  messages,
  assistantName,
  assistantAvatarUrl
}: ApplyAssistantPresentationArgs): Message[] =>
  messages.map((message) => {
    if (!message.isBot || message.role !== "assistant") {
      return message
    }

    const hasExplicitName =
      typeof message.name === "string" &&
      message.name.trim().length > 0 &&
      message.name !== "Assistant"
    const hasExplicitModelName =
      typeof message.modelName === "string" &&
      message.modelName.trim().length > 0 &&
      message.modelName !== "Assistant"

    return {
      ...message,
      name: hasExplicitName ? message.name : assistantName,
      modelName: hasExplicitModelName ? message.modelName : assistantName,
      modelImage:
        message.modelImage ??
        (typeof assistantAvatarUrl === "string" &&
        assistantAvatarUrl.trim().length > 0
          ? assistantAvatarUrl
          : undefined)
    }
  })

export const reportDeferredAssistantPresentationError = ({
  stage,
  assistantKind,
  assistantId,
  characterId,
  error
}: {
  stage: "persona-profile" | "character-profile" | "presentation-apply"
  assistantKind: string | null
  assistantId: string | null
  characterId: number | null
  error: unknown
}): void => {
  console.warn("[useServerChatLoader] Deferred assistant presentation failed", {
    stage,
    assistantKind,
    assistantId,
    characterId,
    error
  })
}

const resolveDeferredCharacterId = (value: string | number | null): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

export const useServerChatLoader = ({
  ensureServerChatHistoryId,
  notification,
  t,
  scope
}: UseServerChatLoaderOptions) => {
  const [selectedAssistant, setSelectedAssistant, assistantMeta] = useSelectedAssistant(null)
  const {
    messages,
    streaming,
    isProcessing,
    setHistory,
    setMessages,
    setIsLoading
  } = useChatBaseState(useStoreMessageOption)
  const messagesRef = React.useRef(messages)
  const streamingRef = React.useRef(streaming)
  const processingRef = React.useRef(isProcessing)
  const selectedAssistantRef = React.useRef(selectedAssistant)
  const {
    serverChatId,
    serverChatTitle,
    serverChatCharacterId,
    serverChatAssistantKind,
    serverChatAssistantId,
    serverChatPersonaMemoryMode,
    serverChatMetaLoaded,
    temporaryChat,
    setServerChatId,
    setServerChatLoadState,
    setServerChatLoadError,
    setServerChatTitle,
    setServerChatCharacterId,
    setServerChatAssistantKind,
    setServerChatAssistantId,
    setServerChatPersonaMemoryMode,
    setServerChatState,
    setServerChatVersion,
    setServerChatTopic,
    setServerChatClusterId,
    setServerChatSource,
    setServerChatExternalRef,
    setServerChatMetaLoaded
  } = useStoreMessageOption(
    (state) => ({
      serverChatId: state.serverChatId,
      serverChatTitle: state.serverChatTitle,
      serverChatCharacterId: state.serverChatCharacterId,
      serverChatAssistantKind: state.serverChatAssistantKind,
      serverChatAssistantId: state.serverChatAssistantId,
      serverChatPersonaMemoryMode: state.serverChatPersonaMemoryMode,
      serverChatMetaLoaded: state.serverChatMetaLoaded,
      temporaryChat: state.temporaryChat,
      setServerChatId: state.setServerChatId,
      setServerChatLoadState: state.setServerChatLoadState,
      setServerChatLoadError: state.setServerChatLoadError,
      setServerChatTitle: state.setServerChatTitle,
      setServerChatCharacterId: state.setServerChatCharacterId,
      setServerChatAssistantKind: state.setServerChatAssistantKind,
      setServerChatAssistantId: state.setServerChatAssistantId,
      setServerChatPersonaMemoryMode: state.setServerChatPersonaMemoryMode,
      setServerChatState: state.setServerChatState,
      setServerChatVersion: state.setServerChatVersion,
      setServerChatTopic: state.setServerChatTopic,
      setServerChatClusterId: state.setServerChatClusterId,
      setServerChatSource: state.setServerChatSource,
      setServerChatExternalRef: state.setServerChatExternalRef,
      setServerChatMetaLoaded: state.setServerChatMetaLoaded
    }),
    shallow
  )

  const serverChatLoadRef = React.useRef<{
    chatId: string | null
    controller: AbortController | null
    inFlight: boolean
    loaded: boolean
  }>({ chatId: null, controller: null, inFlight: false, loaded: false })
  const serverChatDebounceRef = React.useRef<{
    chatId: string | null
    timer: ReturnType<typeof setTimeout> | null
  }>({ chatId: null, timer: null })

  messagesRef.current = messages
  streamingRef.current = streaming
  processingRef.current = isProcessing
  selectedAssistantRef.current = selectedAssistant

  React.useEffect(() => {
    return () => {
      if (serverChatDebounceRef.current.timer) {
        clearTimeout(serverChatDebounceRef.current.timer)
      }
      if (serverChatLoadRef.current.controller) {
        serverChatLoadRef.current.controller.abort()
      }
    }
  }, [])

  React.useEffect(() => {
    if (!serverChatId || assistantMeta?.isLoading) return
    if (
      shouldSkipLoadedServerChatReload({
        activeServerChatId: serverChatId,
        loadedChatId: serverChatLoadRef.current.chatId,
        loaded: serverChatLoadRef.current.loaded,
        currentMessages: messagesRef.current
      })
    ) {
      return
    }
    if (serverChatLoadRef.current.inFlight) {
      if (serverChatLoadRef.current.chatId === serverChatId) {
        return
      }
      if (serverChatLoadRef.current.controller) {
        serverChatLoadRef.current.controller.abort()
      }
      serverChatLoadRef.current.inFlight = false
    }

    if (serverChatDebounceRef.current.timer) {
      clearTimeout(serverChatDebounceRef.current.timer)
    }

    serverChatDebounceRef.current.chatId = serverChatId
    serverChatDebounceRef.current.timer = setTimeout(() => {
      const controller = new AbortController()
      let ownedSelectionRevision = getSelectedAssistantOperationRevision()
      const canCommitCurrentLoad = () =>
        useStoreMessageOption.getState().serverChatId === serverChatId &&
        shouldCommitServerChatLoadResult({
          requestedChatId: serverChatId,
          activeServerChatId: serverChatLoadRef.current.chatId,
          requestController: controller,
          activeController: serverChatLoadRef.current.controller
        })
      const applyOwnedAssistantSelection = async (selection: Parameters<typeof setSelectedAssistant>[0]) => {
        const before = ownedSelectionRevision
        if (!canCommitCurrentLoad() || getSelectedAssistantOperationRevision() !== before) return false
        await setSelectedAssistant(selection, { isCurrent: () => {
          const revision = getSelectedAssistantOperationRevision()
          // This operation increments the existing revision synchronously. A
          // later picker operation must win even before React rerenders.
          return canCommitCurrentLoad() && (revision === before || revision === before + 1)
        } })
        ownedSelectionRevision = before + 1
        return canCommitCurrentLoad() && getSelectedAssistantOperationRevision() === ownedSelectionRevision
      }
      serverChatLoadRef.current = {
        chatId: serverChatId,
        controller,
        inFlight: true,
        loaded: false
      }

      const loadServerChat = async () => {
        let didLoadSuccessfully = false
        let snapshot: ServicePromptSnapshot | undefined
        let stopWatchingAuthority: (() => void) | undefined
        let pendingAssistantPresentation: Promise<unknown> | undefined
        try {
          setIsLoading(true)
          setServerChatLoadState("loading")
          setServerChatLoadError(null)
          snapshot = await loadServicePromptSnapshot([], { signal: controller.signal })
          snapshot.scopeInvalidatedSignal.addEventListener("abort", () => controller.abort(), { once: true })
          if (snapshot.scopeSignal.aborted || !canCommitCurrentLoad()) return
          stopWatchingAuthority = watchServerChatLoadAuthority(snapshot, controller)
          await waitForChatPromotion(setServerChatId, useStoreMessageOption.getState().historyId, snapshot, { waitUntilSaved: true })
          if (!canCommitCurrentLoad()) return

          let assistantName = "Assistant"
          let chatTitle = serverChatTitle || ""
          let characterId = serverChatCharacterId ?? null
          let assistantKind = serverChatAssistantKind
          let assistantId = serverChatAssistantId
          let personaMemoryMode = serverChatPersonaMemoryMode
          let restoreAssistantSelection = true

          if (!serverChatMetaLoaded) {
            try {
              const chat = await tldwClient.getChat(
                serverChatId,
                { scope, signal: snapshot.scopeSignal, requestScope: snapshot.requestScope }
              )
              if (!canCommitCurrentLoad()) {
                return
              }
              const validatedServerChatId = validateCachedServerChatId({
                cachedId: serverChatId,
                serverScope: {
                  scope_type: String((chat as any)?.scope_type || ""),
                  workspace_id:
                    typeof (chat as any)?.workspace_id === "string"
                      ? (chat as any).workspace_id
                      : null
                },
                expectedScope: scope || { type: "global" }
              })
              if (!validatedServerChatId) {
                setMessages([])
                setHistory([])
                setServerChatTitle(null)
                setIsLoading(false)
                setServerChatId(null)
                return
              }
              const meta = chat as unknown as Record<string, unknown>
              chatTitle = String(meta?.title || chatTitle || "")
              const resolvedAssistantIdentity =
                resolveServerChatAssistantIdentity(meta)
              assistantKind = resolvedAssistantIdentity.assistantKind
              assistantId = resolvedAssistantIdentity.assistantId
              characterId = resolvedAssistantIdentity.characterId
              personaMemoryMode = resolvedAssistantIdentity.personaMemoryMode
              restoreAssistantSelection = getSelectedAssistantOperationRevision() === ownedSelectionRevision
              setServerChatTitle(chatTitle || "")
              setServerChatCharacterId(characterId)
              setServerChatAssistantKind(assistantKind)
              setServerChatAssistantId(assistantId)
              setServerChatPersonaMemoryMode(personaMemoryMode)
              setServerChatState(
                normalizeConversationState(
                  (meta?.state as string | null | undefined) ??
                    (meta?.conversation_state as string | null | undefined)
                )
              )
              setServerChatVersion(
                typeof meta?.version === "number" ? meta.version : null
              )
              setServerChatTopic(
                typeof meta?.topic_label === "string"
                  ? meta.topic_label
                  : null
              )
              setServerChatClusterId(
                typeof meta?.cluster_id === "string" ? meta.cluster_id : null
              )
              setServerChatSource(
                typeof meta?.source === "string" ? meta.source : null
              )
              setServerChatExternalRef(
                typeof meta?.external_ref === "string"
                  ? meta.external_ref
                  : null
              )
              setServerChatMetaLoaded(true)
            } catch (error) {
              if (isMissingServerChatReferenceError(error) || isDeniedServerChatError(error)) throw error
              // ignore metadata failures; still try to load messages
            }
          }

          const deferredAssistantPresentationPromise = (async () => {
            if (!restoreAssistantSelection) return null
            let syncedSettings = null
            if (assistantKind == null && characterId == null) {
              try {
                syncedSettings = await syncChatSettingsForServerChat({
                  historyId: null,
                  serverChatId,
                  allowScratchFallback: false
                })
              } catch {
                syncedSettings = null
              }
            }

            if (assistantKind === "persona" && assistantId) {
              try {
                const persona = await tldwClient.getPersonaProfile(assistantId, {
                  signal: snapshot!.scopeSignal, requestScope: snapshot!.requestScope
                })
                if (persona) {
                  if (!canCommitCurrentLoad()) {
                    return null
                  }
                  const nextAssistantName = persona.name || "Persona"
                  const selection = personaToAssistantSelection({
                    ...persona,
                    id: assistantId,
                    name: nextAssistantName
                  })
                  if (!await applyOwnedAssistantSelection(selection)) return null
                  return {
                    assistantName: nextAssistantName,
                    assistantAvatarUrl: selection?.avatar_url ?? null
                  }
                }
              } catch (error) {
                reportDeferredAssistantPresentationError({
                  stage: "persona-profile",
                  assistantKind,
                  assistantId,
                  characterId: resolveDeferredCharacterId(characterId),
                  error
                })
                if (!canCommitCurrentLoad()) {
                  return null
                }
                const selection = personaToAssistantSelection({
                  id: assistantId,
                  name: "Persona"
                })
                if (!await applyOwnedAssistantSelection(selection)) return null
                return {
                  assistantName: selection?.name || "Persona",
                  assistantAvatarUrl: selection?.avatar_url ?? null
                }
              }
            } else if (characterId != null) {
              try {
                const character = await tldwClient.getCharacter(characterId, {
                  signal: snapshot!.scopeSignal, requestScope: snapshot!.requestScope
                })
                if (character) {
                  if (!canCommitCurrentLoad()) {
                    return null
                  }
                  const selection = characterToAssistantSelection({
                    ...character,
                    id: String(character.id ?? characterId)
                  })
                  if (!await applyOwnedAssistantSelection(selection)) return null
                  return {
                    assistantName:
                      selection?.name ||
                      character.name ||
                      character.title ||
                      assistantName,
                    assistantAvatarUrl: selection?.avatar_url ?? null
                  }
                }
              } catch (error) {
                reportDeferredAssistantPresentationError({
                  stage: "character-profile",
                  assistantKind,
                  assistantId,
                  characterId: resolveDeferredCharacterId(characterId),
                  error
                })
              }
              if (!canCommitCurrentLoad()) {
                return null
              }
              const fallbackState = resolveEffectiveAssistantState({
                tracked: {
                  assistantKind,
                  assistantId,
                  characterId
                },
                draftSelection: selectedAssistantRef.current
              })
              const selection =
                effectiveAssistantStateToSelection(fallbackState)
              if (selection) {
                if (!await applyOwnedAssistantSelection(selection)) return null
                return {
                  assistantName: selection.name,
                  assistantAvatarUrl: selection.avatar_url ?? null
                }
              }
              if (!await applyOwnedAssistantSelection(null)) return null
              return null
            }

            if (!canCommitCurrentLoad()) {
              return null
            }
            const effectiveAssistantState = resolveEffectiveAssistantState({
              tracked: {
                assistantKind,
                assistantId,
                characterId
              },
              settings: syncedSettings
            })
            const selection =
              effectiveAssistantStateToSelection(effectiveAssistantState)
            if (selection) {
              if (!await applyOwnedAssistantSelection(selection)) return null
              return {
                assistantName: selection.name,
                assistantAvatarUrl: selection.avatar_url ?? null
              }
            }
            if (!await applyOwnedAssistantSelection(null)) return null
            return null
          })()

          pendingAssistantPresentation = deferredAssistantPresentationPromise

          const list = await fetchAllServerChatMessages(
            ({ limit, offset, signal }) =>
              tldwClient.listChatMessages(
                serverChatId,
                {
                  include_deleted: "false",
                  include_metadata: "true",
                  include_images: "true",
                  render_placeholders: assistantKind === "character" || characterId != null ? "true" : "false",
                  limit,
                  offset
                },
                { signal, scope, requestScope: snapshot!.requestScope }
              ),
            {
              signal: controller.signal
            }
          )

          if (list.some(message => message.role === "user" && (message.images?.length ?? 0) > 1)) {
            throw new Error("This conversation has multiple images in one user message. Chat currently supports one image per turn. Your local work has been kept; open a new conversation to continue.")
          }
          const mappedMessages = mapServerChatMessagesToPlaygroundMessages({
            serverMessages: list,
            assistantName,
            characterId
          })
          if (!canCommitCurrentLoad()) {
            return
          }
          const active = streamingRef.current || processingRef.current
          if (!active) {
            const merged = reconcileServerChatMessages(useStoreMessageOption.getState().messages, mappedMessages)
            setHistory(excludeLocalRagDiagnostics(merged).map(message => ({ role: message.role, content: message.message, image: message.images?.[0], messageType: message.messageType })))
            setMessages(merged)
          }
          const shouldApplyDeferredAssistantPresentation =
            !active
          if (shouldApplyDeferredAssistantPresentation) {
            void deferredAssistantPresentationPromise
              .then((presentation) => {
                if (!presentation || !canCommitCurrentLoad()) {
                  return
                }
                setMessages((currentMessages) =>
                  applyAssistantPresentationToMessages({
                    messages: currentMessages,
                    assistantName: presentation.assistantName,
                    assistantAvatarUrl: presentation.assistantAvatarUrl
                  })
                )
              })
              .catch((error) => {
                reportDeferredAssistantPresentationError({
                  stage: "presentation-apply",
                  assistantKind,
                  assistantId,
                  characterId: resolveDeferredCharacterId(characterId),
                  error
                })
              })
          }
          if (!temporaryChat && !active) {
            if (!canCommitCurrentLoad()) {
              return
            }
            const beforeMirror = useStoreMessageOption.getState().messages
            try {
              const localHistoryId = await ensureServerChatHistoryId(
                serverChatId,
                chatTitle || undefined,
                snapshot.scopeSignal,
                snapshot
              )
              if (!canCommitCurrentLoad()) {
                return
              }
              if (localHistoryId) {
                try {
                  await syncChatSettingsForServerChat({
                    historyId: localHistoryId,
                    serverChatId,
                    allowScratchFallback: false
                  })
                } catch {
                  // Best-effort settings sync.
                }
                if (!canCommitCurrentLoad() || streamingRef.current || processingRef.current) return
                const mirror = await reconcileServerChatMirror({
                  historyId: localHistoryId, chatId: serverChatId,
                  ownerKey: serverChatMirrorOwnerKey(snapshot), messages: mappedMessages,
                  localMessages: useStoreMessageOption.getState().messages,
                  signal: snapshot.scopeSignal
                })
                if (!canCommitCurrentLoad() || streamingRef.current || processingRef.current ||
                  useStoreMessageOption.getState().historyId !== localHistoryId) return
                const merged = reconcileServerChatMessages(
                  useStoreMessageOption.getState().messages,
                  formatToMessage(mirror.rows),
                  beforeMirror
                ).map(message => {
                  const id = message.serverMessageId ? mirror.localIds.get(message.serverMessageId) : undefined
                  return id && message.id !== id ? { ...message, id } : message
                })
                setHistory(excludeLocalRagDiagnostics(merged).map(message => ({ role: message.role, content: message.message, image: message.images?.[0], messageType: message.messageType })))
                setMessages(merged)
              }
            } catch {
              if (!canCommitCurrentLoad()) {
                return
              }
              // Local mirror is best-effort for server chats.
            }
          }
          if (!canCommitCurrentLoad()) {
            return
          }
          if (chatTitle) {
            updatePageTitle(chatTitle)
          }
          didLoadSuccessfully = true
          setServerChatLoadError(null)
          setServerChatLoadState("loaded")
        } catch (e: unknown) {
          const message = e instanceof Error ? e.message : String(e || "")
          const isAbort =
            e instanceof Error && e.name === "AbortError"
              ? true
              : message.toLowerCase().includes("abort")
          if (!isAbort && canCommitCurrentLoad() &&
            (isMissingServerChatReferenceError(e) || isDeniedServerChatError(e))) {
            setMessages([])
            setHistory([])
            setServerChatTitle(null)
            updatePageTitle()
          }
          if (!isAbort && isMissingServerChatReferenceError(e) && canCommitCurrentLoad()) {
            setIsLoading(false)
            setServerChatId(null)
            return
          }
          if (!isAbort && canCommitCurrentLoad()) {
            const description =
              message ||
              t("common:serverChatLoadError", {
                defaultValue:
                  "Failed to load server chat. Check your connection and try again."
              })
            setServerChatLoadState("failed")
            setServerChatLoadError(description)
            notification.error({
              message: t("error", { defaultValue: "Error" }),
              description
            })
          }
        } finally {
          if (useStoreMessageOption.getState().serverChatId === serverChatId) setIsLoading(false)
          // Messages are ready independently of optional profile enrichment.
          // Keep this load's authority alive until that guarded work settles.
          await pendingAssistantPresentation?.catch(() => undefined)
          stopWatchingAuthority?.()
          snapshot?.release()
          if (serverChatLoadRef.current.controller === controller) {
            serverChatLoadRef.current = {
              chatId: serverChatId,
              controller: null,
              inFlight: false,
              loaded: didLoadSuccessfully
            }
          }
        }
      }

      void loadServerChat()
    }, 200)

    return () => {
      if (serverChatDebounceRef.current.timer) {
        clearTimeout(serverChatDebounceRef.current.timer)
        serverChatDebounceRef.current.timer = null
      }
    }
  }, [
    assistantMeta?.isLoading,
    ensureServerChatHistoryId,
    notification,
    serverChatAssistantId,
    serverChatAssistantKind,
    serverChatCharacterId,
    serverChatId,
    serverChatMetaLoaded,
    serverChatPersonaMemoryMode,
    serverChatTitle,
    setHistory,
    setIsLoading,
    setMessages,
    setSelectedAssistant,
    setServerChatAssistantId,
    setServerChatAssistantKind,
    setServerChatCharacterId,
    setServerChatClusterId,
    setServerChatExternalRef,
    setServerChatId,
    setServerChatLoadError,
    setServerChatLoadState,
    setServerChatMetaLoaded,
    setServerChatPersonaMemoryMode,
    setServerChatSource,
    setServerChatState,
    setServerChatTitle,
    setServerChatTopic,
    setServerChatVersion,
    scope,
    t,
    temporaryChat
  ])
}
