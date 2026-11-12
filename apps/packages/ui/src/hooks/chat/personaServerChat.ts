import type { AssistantSelection } from "@/types/assistant-selection"
import type { ChatScope } from "@/types/chat-scope"
import { normalizeConversationState } from "@/utils/conversation-state"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"

export const DEFAULT_PERSONA_MEMORY_MODE = "read_only" as const

type PersonaAssistant = AssistantSelection & { kind: "persona" }
type PersonaMemoryMode = "read_only" | "read_write"

const normalizePersonaMemoryMode = (value: unknown): PersonaMemoryMode | null =>
  value === "read_only" || value === "read_write" ? value : null

type EnsurePersonaServerChatArgs = {
  assistant: PersonaAssistant
  serverChatIdOverride?: string | null
  requestedPersonaMemoryMode?: PersonaMemoryMode | null
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
  requestScope?: ServicePromptRequestScope
  signal?: AbortSignal
  scopeInvalidatedSignal?: AbortSignal
  isCurrent?: () => boolean
  createChat: (
    payload: Record<string, unknown>,
    options?: {
      scope?: ChatScope
      requestScope?: ServicePromptRequestScope
      signal?: AbortSignal
    }
  ) => Promise<any>
  ensureServerChatHistoryId: (
    chatId: string,
    title?: string,
    scopeInvalidatedSignal?: AbortSignal
  ) => Promise<string | null>
  invalidateServerChatHistory: () => void
  setServerChatId: import("@/store/option/types").State["setServerChatId"]
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

export const resetAssistantServerChatState = (
  {
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
  >,
  assertCurrent: () => void = () => {}
) => {
  const publish = (update: () => void) => {
    assertCurrent()
    update()
    assertCurrent()
  }
  publish(() => setServerChatId(null))
  publish(() => setServerChatTitle(null))
  publish(() => setServerChatCharacterId(null))
  publish(() => setServerChatAssistantKind(null))
  publish(() => setServerChatAssistantId(null))
  publish(() => setServerChatPersonaMemoryMode(null))
  publish(() => setServerChatMetaLoaded(false))
  publish(() => setServerChatState("in-progress"))
  publish(() => setServerChatVersion(null))
  publish(() => setServerChatTopic(null))
  publish(() => setServerChatClusterId(null))
  publish(() => setServerChatSource(null))
  publish(() => setServerChatExternalRef(null))
}

export const ensurePersonaServerChat = async ({
  assistant,
  serverChatIdOverride,
  requestedPersonaMemoryMode,
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
  requestScope,
  signal,
  scopeInvalidatedSignal,
  isCurrent,
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
  const capturedScope = scope ? structuredClone(scope) : undefined
  const capturedRequestScope = requestScope
    ? structuredClone(requestScope)
    : undefined
  const throwIfScopeInvalidated = () => {
    if (scopeInvalidatedSignal?.aborted || isCurrent?.() === false) {
      throw createServicePromptScopeChangedError()
    }
    signal?.throwIfAborted()
  }
  const publish = (update: () => void) => {
    throwIfScopeInvalidated()
    update()
    throwIfScopeInvalidated()
  }
  throwIfScopeInvalidated()
  const overrideChatId =
    typeof serverChatIdOverride === "string" &&
    serverChatIdOverride.trim().length > 0
      ? serverChatIdOverride.trim()
      : null
  const resolvedServerChatId = overrideChatId || serverChatId
  const assistantId = String(assistant.id)
  const newChatPersonaMemoryMode =
    normalizePersonaMemoryMode(requestedPersonaMemoryMode) ??
    DEFAULT_PERSONA_MEMORY_MODE
  const isReusingUnhydratedChat =
    Boolean(resolvedServerChatId) && !serverChatMetaLoaded
  const isMatchingPersonaChat =
    Boolean(resolvedServerChatId) &&
    serverChatMetaLoaded &&
    serverChatAssistantKind === "persona" &&
    Boolean(serverChatAssistantId) &&
    String(serverChatAssistantId) === assistantId
  const personaMemoryMode =
    isMatchingPersonaChat || isReusingUnhydratedChat
      ? normalizePersonaMemoryMode(serverChatPersonaMemoryMode) ??
        newChatPersonaMemoryMode
      : newChatPersonaMemoryMode
  const shouldResetServerChat =
    Boolean(resolvedServerChatId) &&
    serverChatMetaLoaded &&
    !isMatchingPersonaChat

  if (shouldResetServerChat) {
    throwIfScopeInvalidated()
    resetAssistantServerChatState(
      {
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
      },
      throwIfScopeInvalidated
    )
  }

  let chatId = shouldResetServerChat ? null : resolvedServerChatId
  let publishServerChatState: (() => void) | null = null
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
    }, capturedScope || capturedRequestScope || signal
      ? { scope: capturedScope, requestScope: capturedRequestScope, signal }
      : undefined)

    throwIfScopeInvalidated()

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

    publishServerChatState = () => {
      publish(() => setServerChatId(normalizedId, { preserveTemporaryChat: temporaryChat }))
      publish(() => setServerChatState(normalizeConversationState(createdState)))
      publish(() => setServerChatVersion(createdVersion))
      publish(() => setServerChatTopic(createdTopic))
      publish(() => setServerChatClusterId(createdClusterId))
      publish(() => setServerChatSource(createdSource))
      publish(() => setServerChatExternalRef(createdExternalRef))
      publish(() => setServerChatTitle(createdTitle))
      publish(() => setServerChatCharacterId(createdCharacterId))
      publish(() => setServerChatAssistantKind(createdAssistantKind))
      publish(() => setServerChatAssistantId(createdAssistantId))
      publish(() => setServerChatPersonaMemoryMode(createdPersonaMemoryMode))
      publish(() => setServerChatMetaLoaded(true))
      publish(invalidateServerChatHistory)
    }
  } else {
    throwIfScopeInvalidated()
    publishServerChatState = () => {
      publish(() => setServerChatAssistantKind("persona"))
      publish(() => setServerChatAssistantId(assistantId))
      publish(() => setServerChatPersonaMemoryMode(personaMemoryMode))
      publish(() => setServerChatCharacterId(null))
    }
  }

  let resolvedHistoryId = historyId
  if (!temporaryChat && chatId) {
    throwIfScopeInvalidated()
    resolvedHistoryId = await ensureServerChatHistoryId(
      chatId,
      serverChatTitle || undefined,
      scopeInvalidatedSignal
    )
    throwIfScopeInvalidated()
  }
  throwIfScopeInvalidated()
  publishServerChatState?.()

  return {
    chatId,
    historyId: resolvedHistoryId ?? historyId,
    personaMemoryMode
  }
}
