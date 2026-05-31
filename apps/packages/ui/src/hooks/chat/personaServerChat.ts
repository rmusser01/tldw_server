import type { AssistantSelection } from "@/types/assistant-selection"
import type { ChatScope } from "@/types/chat-scope"
import { normalizeConversationState } from "@/utils/conversation-state"

export const DEFAULT_PERSONA_MEMORY_MODE = "read_only" as const

type PersonaAssistant = AssistantSelection & { kind: "persona" }

type EnsurePersonaServerChatArgs = {
  assistant: PersonaAssistant
  serverChatIdOverride?: string | null
  serverChatId: string | null
  serverChatTitle: string | null
  serverChatAssistantKind: "character" | "persona" | null
  serverChatAssistantId: string | null
  serverChatPersonaMemoryMode: "read_only" | "read_write" | null
  serverChatMetaLoaded?: boolean
  serverChatState: string | null
  serverChatTopic: string | null
  serverChatClusterId: string | null
  serverChatSource: string | null
  serverChatExternalRef: string | null
  historyId: string | null
  temporaryChat: boolean
  scope?: ChatScope
  createChat: (
    payload: Record<string, unknown>,
    options?: { scope?: ChatScope }
  ) => Promise<any>
  ensureServerChatHistoryId: (
    chatId: string,
    title?: string
  ) => Promise<string | null>
  invalidateServerChatHistory: () => void
  setServerChatId: (value: string | null) => void
  setServerChatTitle: (value: string | null) => void
  setServerChatCharacterId: (value: string | number | null) => void
  setServerChatAssistantKind: (value: "character" | "persona" | null) => void
  setServerChatAssistantId: (value: string | null) => void
  setServerChatPersonaMemoryMode: (
    value: "read_only" | "read_write" | null
  ) => void
  setServerChatMetaLoaded: (value: boolean) => void
  setServerChatState: (value: string | null) => void
  setServerChatVersion: (value: number | null) => void
  setServerChatTopic: (value: string | null) => void
  setServerChatClusterId: (value: string | null) => void
  setServerChatSource: (value: string | null) => void
  setServerChatExternalRef: (value: string | null) => void
}

export const resetAssistantServerChatState = ({
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
}: Pick<
  EnsurePersonaServerChatArgs,
  | "setServerChatId"
  | "setServerChatTitle"
  | "setServerChatCharacterId"
  | "setServerChatAssistantKind"
  | "setServerChatAssistantId"
  | "setServerChatPersonaMemoryMode"
  | "setServerChatMetaLoaded"
  | "setServerChatState"
  | "setServerChatVersion"
  | "setServerChatTopic"
  | "setServerChatClusterId"
  | "setServerChatSource"
  | "setServerChatExternalRef"
>) => {
  setServerChatId(null)
  setServerChatTitle(null)
  setServerChatCharacterId(null)
  setServerChatAssistantKind(null)
  setServerChatAssistantId(null)
  setServerChatPersonaMemoryMode(null)
  setServerChatMetaLoaded(false)
  setServerChatState("in-progress")
  setServerChatVersion(null)
  setServerChatTopic(null)
  setServerChatClusterId(null)
  setServerChatSource(null)
  setServerChatExternalRef(null)
}

export const ensurePersonaServerChat = async ({
  assistant,
  serverChatIdOverride,
  serverChatId,
  serverChatTitle,
  serverChatAssistantKind,
  serverChatAssistantId,
  serverChatPersonaMemoryMode,
  serverChatMetaLoaded = false,
  serverChatState,
  serverChatTopic,
  serverChatClusterId,
  serverChatSource,
  serverChatExternalRef,
  historyId,
  temporaryChat,
  scope,
  createChat,
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
}: EnsurePersonaServerChatArgs): Promise<{
  chatId: string
  historyId: string | null
  personaMemoryMode: "read_only" | "read_write"
}> => {
  const overrideChatId =
    typeof serverChatIdOverride === "string" &&
    serverChatIdOverride.trim().length > 0
      ? serverChatIdOverride.trim()
      : null
  const resolvedServerChatId = overrideChatId || serverChatId
  const assistantId = String(assistant.id)
  const isMatchingPersonaChat =
    Boolean(resolvedServerChatId) &&
    serverChatMetaLoaded &&
    serverChatAssistantKind === "persona" &&
    Boolean(serverChatAssistantId) &&
    String(serverChatAssistantId) === assistantId
  const personaMemoryMode = isMatchingPersonaChat
    ? serverChatPersonaMemoryMode ?? DEFAULT_PERSONA_MEMORY_MODE
    : DEFAULT_PERSONA_MEMORY_MODE
  const shouldResetServerChat =
    Boolean(resolvedServerChatId) &&
    serverChatMetaLoaded &&
    !isMatchingPersonaChat

  if (shouldResetServerChat) {
    resetAssistantServerChatState({
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
    })
  }

  let chatId = shouldResetServerChat ? null : resolvedServerChatId
  if (!chatId) {
    const created = await createChat({
      assistant_kind: "persona",
      assistant_id: assistantId,
      persona_memory_mode: personaMemoryMode,
      state: "in-progress",
      topic_label: undefined,
      cluster_id: undefined,
      source: undefined,
      external_ref: undefined
    }, scope ? { scope } : undefined)

    let rawId: string | number | undefined
    const createdMeta =
      created && typeof created === "object"
        ? (created as Record<string, unknown>)
        : null
    if (created && typeof created === "object") {
      rawId = created.id ?? created.chat_id
    } else if (typeof created === "string" || typeof created === "number") {
      rawId = created
    }

    const normalizedId = rawId != null ? String(rawId) : ""
    if (!normalizedId) {
      throw new Error("Failed to create persona-backed chat session")
    }
    chatId = normalizedId
    setServerChatId(normalizedId)
    const createdState =
      typeof createdMeta?.state === "string"
        ? createdMeta.state
        : typeof createdMeta?.conversation_state === "string"
          ? createdMeta.conversation_state
          : "in-progress"
    const createdVersion =
      typeof createdMeta?.version === "number" ? createdMeta.version : null
    const createdTopic =
      typeof createdMeta?.topic_label === "string"
        ? createdMeta.topic_label
        : null
    const createdClusterId =
      typeof createdMeta?.cluster_id === "string" ? createdMeta.cluster_id : null
    const createdSource =
      typeof createdMeta?.source === "string" ? createdMeta.source : null
    const createdExternalRef =
      typeof createdMeta?.external_ref === "string"
        ? createdMeta.external_ref
        : null
    const createdTitle =
      typeof createdMeta?.title === "string" ? createdMeta.title : ""
    const createdCharacterId =
      typeof createdMeta?.character_id === "string" ||
      typeof createdMeta?.character_id === "number"
        ? createdMeta.character_id
        : null
    const createdAssistantKind =
      createdMeta?.assistant_kind === "character" ||
      createdMeta?.assistant_kind === "persona"
        ? createdMeta.assistant_kind
        : "persona"
    const createdAssistantId =
      typeof createdMeta?.assistant_id === "string" ||
      typeof createdMeta?.assistant_id === "number"
        ? String(createdMeta.assistant_id)
        : assistantId
    const createdPersonaMemoryMode =
      createdMeta?.persona_memory_mode === "read_only" ||
      createdMeta?.persona_memory_mode === "read_write"
        ? createdMeta.persona_memory_mode
        : personaMemoryMode

    setServerChatState(
      normalizeConversationState(createdState)
    )
    setServerChatVersion(createdVersion)
    setServerChatTopic(createdTopic)
    setServerChatClusterId(createdClusterId)
    setServerChatSource(createdSource)
    setServerChatExternalRef(createdExternalRef)
    setServerChatTitle(createdTitle)
    setServerChatCharacterId(createdCharacterId)
    setServerChatAssistantKind(createdAssistantKind)
    setServerChatAssistantId(createdAssistantId)
    setServerChatPersonaMemoryMode(createdPersonaMemoryMode)
    setServerChatMetaLoaded(true)
    invalidateServerChatHistory()
  } else {
    setServerChatAssistantKind("persona")
    setServerChatAssistantId(assistantId)
    setServerChatPersonaMemoryMode(personaMemoryMode)
    setServerChatCharacterId(null)
  }

  const resolvedHistoryId =
    temporaryChat || !chatId
      ? historyId
      : await ensureServerChatHistoryId(chatId, serverChatTitle || undefined)

  return {
    chatId,
    historyId: resolvedHistoryId ?? historyId,
    personaMemoryMode
  }
}
