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
    serverChatAssistantKind === "persona" &&
    Boolean(serverChatAssistantId) &&
    String(serverChatAssistantId) === assistantId
  const personaMemoryMode = isMatchingPersonaChat
    ? serverChatPersonaMemoryMode ?? DEFAULT_PERSONA_MEMORY_MODE
    : DEFAULT_PERSONA_MEMORY_MODE
  const shouldResetServerChat =
    Boolean(resolvedServerChatId) && !isMatchingPersonaChat

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
      state: shouldResetServerChat
        ? "in-progress"
        : serverChatState || "in-progress",
      topic_label: shouldResetServerChat
        ? undefined
        : serverChatTopic || undefined,
      cluster_id: shouldResetServerChat
        ? undefined
        : serverChatClusterId || undefined,
      source: shouldResetServerChat ? undefined : serverChatSource || undefined,
      external_ref: shouldResetServerChat
        ? undefined
        : serverChatExternalRef || undefined
    }, scope ? { scope } : undefined)

    let rawId: string | number | undefined
    if (created && typeof created === "object") {
      rawId = created.id ?? created.chat_id
      setServerChatState(
        normalizeConversationState(
          created.state ?? created.conversation_state ?? null
        )
      )
      setServerChatVersion(
        typeof created.version === "number" ? created.version : null
      )
      setServerChatTopic(created.topic_label ?? null)
      setServerChatClusterId(created.cluster_id ?? null)
      setServerChatSource(created.source ?? null)
      setServerChatExternalRef(created.external_ref ?? null)
      setServerChatTitle(String(created.title ?? ""))
      setServerChatCharacterId(created.character_id ?? null)
      setServerChatAssistantKind(created.assistant_kind ?? "persona")
      setServerChatAssistantId(
        created.assistant_id != null ? String(created.assistant_id) : assistantId
      )
      setServerChatPersonaMemoryMode(
        created.persona_memory_mode ?? personaMemoryMode
      )
    } else if (typeof created === "string" || typeof created === "number") {
      rawId = created
    }

    const normalizedId = rawId != null ? String(rawId) : ""
    if (!normalizedId) {
      throw new Error("Failed to create persona-backed chat session")
    }
    chatId = normalizedId
    setServerChatId(normalizedId)
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
