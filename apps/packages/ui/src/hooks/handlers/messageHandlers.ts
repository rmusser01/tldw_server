import { type ChatHistory, type Message } from "~/store/option"
import {
  deleteChatForEdit,
  formatToChatHistory,
  formatToMessage,
  saveHistory,
  saveMessage,
  updateMessageByIndex
} from "@/db/dexie/helpers"
import { generateBranchMessage } from "@/db/dexie/branch"
import { getPromptById, getSessionFiles, UploadedFile } from "@/db"
import { tldwClient, type ConversationState } from "@/services/tldw/TldwApiClient"
import { normalizeConversationState } from "@/utils/conversation-state"
import type { NotificationInstance } from "antd/es/notification/interface"
import type { ChatScope } from "@/types/chat-scope"

export const createRegenerateLastMessage = ({
  validateBeforeSubmitFn,
  history,
  messages,
  setHistory,
  setMessages,
  onSubmit,
  beforeSubmit
}: {
  validateBeforeSubmitFn: () => boolean
  history: ChatHistory
  messages: Message[]
  setHistory: (history: ChatHistory) => void
  setMessages: (messages: Message[]) => void
  onSubmit: (params: any) => Promise<unknown>
  beforeSubmit?: (params: {
    lastAssistant: Message
    lastAssistantIndex: number
    userContent: string
    userImage: string
    userMessageType: Message["messageType"] | undefined
    newHistory: ChatHistory
    nextMessages: Message[]
  }) => Promise<
    | {
        memory?: ChatHistory
        messages?: Message[]
        submitExtras?: Record<string, unknown>
      }
    | void
  >
}) => {
  return async () => {
    if (typeof setHistory !== "function") {
      console.error("[chat] regenerate aborted: setHistory is not callable", {
        setHistoryType: typeof setHistory
      })
      return
    }
    const isOk = validateBeforeSubmitFn()

    if (!isOk) {
      return
    }
    const lastAssistantIndex = (() => {
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i]?.isBot) return i
      }
      return -1
    })()
    if (lastAssistantIndex < 0) {
      return
    }

    const lastAssistant = messages[lastAssistantIndex]
    const historyUser = (() => {
      for (let i = history.length - 1; i >= 0; i--) {
        if (history[i]?.role === "user") {
          return { index: i, entry: history[i] }
        }
      }
      return null
    })()
    const messageUser = (() => {
      for (let i = lastAssistantIndex - 1; i >= 0; i--) {
        if (!messages[i]?.isBot) return messages[i]
      }
      return null
    })()

    const userContent =
      (historyUser?.entry?.content ?? messageUser?.message ?? "").trim()
    if (!userContent) {
      return
    }

    const userImage =
      historyUser?.entry?.image || messageUser?.images?.[0] || ""
    const userMessageType =
      historyUser?.entry?.messageType || messageUser?.messageType

    const newHistory = historyUser
      ? history.slice(0, historyUser.index)
      : history.slice(0, Math.max(history.length - 2, 0))
    const nextMessages = messages.filter((_, idx) => idx !== lastAssistantIndex)

    const beforeSubmitResult =
      (await beforeSubmit?.({
        lastAssistant,
        lastAssistantIndex,
        userContent,
        userImage,
        userMessageType,
        newHistory,
        nextMessages
      })) || {}
    const submitHistory = beforeSubmitResult.memory ?? newHistory
    const submitMessages = beforeSubmitResult.messages ?? nextMessages
    const submitExtras = beforeSubmitResult.submitExtras ?? {}

    setHistory(submitHistory)
    setMessages(submitMessages)

    const newController = new AbortController()
    await onSubmit({
      message: userContent,
      image: userImage,
      isRegenerate: true,
      memory: submitHistory,
      messages: submitMessages,
      controller: newController,
      messageType: userMessageType,
      regenerateFromMessage: lastAssistant,
      ...submitExtras
    })
  }
}

export const createEditMessage = ({
  messages,
  history,
  setMessages,
  setHistory,
  historyId,
  validateBeforeSubmitFn,
  onSubmit
}: {
  messages: Message[]
  history: ChatHistory
  setMessages: (messages: Message[]) => void
  setHistory: (history: ChatHistory) => void
  historyId: string | null
  validateBeforeSubmitFn: () => boolean
  onSubmit: (params: any) => Promise<unknown>
}) => {
  return async (
    index: number,
    message: string,
    isHuman: boolean,
    isSend: boolean
  ) => {
    const newHistory = history

    // if human message and send then only trigger the submit
    if (isHuman && isSend) {
      const isOk = validateBeforeSubmitFn()

      if (!isOk) {
        return
      }

      const currentHumanMessage = messages[index]
      const updatedMessages = messages.map((msg, idx) =>
        idx === index ? { ...msg, message } : msg
      )
      const previousMessages = updatedMessages.slice(0, index + 1)
      setMessages(previousMessages)
      const previousHistory = newHistory.slice(0, index)
      setHistory(previousHistory)
      await updateMessageByIndex(historyId, index, message)
      await deleteChatForEdit(historyId, index)
      const abortController = new AbortController()
      await onSubmit({
        message: message,
        image: currentHumanMessage.images[0] || "",
        isRegenerate: true,
        messages: previousMessages,
        memory: previousHistory,
        controller: abortController
      })
      return
    }
    const updatedMessages = messages.map((msg, idx) =>
      idx === index ? { ...msg, message } : msg
    )
    setMessages(updatedMessages)
    const updatedHistory = newHistory.map((item, idx) =>
      idx === index ? { ...item, content: message } : item
    )
    setHistory(updatedHistory)
    await updateMessageByIndex(historyId, index, message)
  }
}

export const createBranchMessage = ({
  notification,
  setMessages,
  setHistory,
  historyId,
  setHistoryId,
  setContext,
  setSelectedSystemPrompt,
  setSystemPrompt,
  serverChatId,
  scope,
  setServerChatId,
  setServerChatState,
  setServerChatVersion,
  setServerChatTitle,
  setServerChatCharacterId,
  setServerChatMetaLoaded,
  setServerChatTopic,
  setServerChatClusterId,
  setServerChatSource,
  setServerChatExternalRef,
  characterId,
  chatTitle,
  serverChatState,
  serverChatTopic,
  serverChatClusterId,
  serverChatSource,
  serverChatExternalRef,
  messages,
  history,
  onServerChatMutated,
  serverOnly = false
}: {
  setMessages: (messages: Message[]) => void
  setHistory: (history: ChatHistory) => void
  historyId: string | null
  setHistoryId: (id: string | null) => void
  setSelectedSystemPrompt?: (prompt: string) => void
  setSystemPrompt?: (prompt: string) => void
  setContext?: (context: UploadedFile[]) => void
  serverChatId?: string | null
  scope?: ChatScope
  setServerChatId?: (id: string | null) => void
  setServerChatState?: (state: ConversationState | null) => void
  setServerChatVersion?: (version: number | null) => void
  setServerChatTitle?: (title: string | null) => void
  setServerChatCharacterId?: (id: string | number | null) => void
  setServerChatMetaLoaded?: (loaded: boolean) => void
  setServerChatTopic?: (topic: string | null) => void
  setServerChatClusterId?: (clusterId: string | null) => void
  setServerChatSource?: (source: string | null) => void
  setServerChatExternalRef?: (ref: string | null) => void
  characterId?: string | number | null
  chatTitle?: string | null
  serverChatState?: ConversationState | null
  serverChatTopic?: string | null
  serverChatClusterId?: string | null
  serverChatSource?: string | null
  serverChatExternalRef?: string | null
  messages?: Message[]
  history?: ChatHistory
  onServerChatMutated?: () => void
  serverOnly?: boolean
  notification: NotificationInstance
}) => {
  const createLocalBranch = async (index: number): Promise<string | null> => {
    if (!historyId) {
      // No persisted history; nothing to branch from.
      return null
    }

    try {
      const newBranch = await generateBranchMessage(historyId, index)
      setHistory(formatToChatHistory(newBranch.messages))
      setMessages(formatToMessage(newBranch.messages))
      setHistoryId(newBranch.history.id)
      const systemFiles = await getSessionFiles(newBranch.history.id)
      if (setContext) {
        setContext(systemFiles)
      }

      const lastUsedPrompt = newBranch?.history?.last_used_prompt
      if (lastUsedPrompt) {
        if (lastUsedPrompt.prompt_id) {
          const prompt = await getPromptById(lastUsedPrompt.prompt_id)
          if (prompt && setSelectedSystemPrompt) {
            setSelectedSystemPrompt(lastUsedPrompt.prompt_id)
          }
        }
        if (setSystemPrompt) {
          setSystemPrompt(lastUsedPrompt.prompt_content)
        }
      }
      return newBranch.history.id
    } catch (e) {
      return null
    }
  }

  const createLocalBranchFromSnapshot = async (
    index: number,
    branchTitle: string
  ): Promise<string | null> => {
    if (!messages || messages.length === 0) {
      return null
    }

    const snapshot = messages.slice(0, index + 1)
    if (snapshot.length === 0) {
      return null
    }

    try {
      const newHistory = await saveHistory(branchTitle, false, "branch")
      const savedMessages: any[] = []

      for (let i = 0; i < snapshot.length; i++) {
        const msg = snapshot[i]
        const role =
          msg.name === "System"
            ? "system"
            : msg.isBot
              ? "assistant"
              : "user"
        const name =
          msg.name ||
          (role === "assistant"
            ? "Assistant"
            : role === "system"
              ? "System"
              : "You")
        const saved = await saveMessage({
          history_id: newHistory.id,
          name,
          role,
          content: String(msg.message ?? ""),
          images: msg.images || [],
          source: msg.sources || [],
          time: i,
          message_type: msg.messageType,
          clusterId: msg.clusterId,
          modelId: msg.modelId,
          modelImage: msg.modelImage,
          modelName: msg.modelName,
          parent_message_id: msg.parentMessageId ?? null,
          documents: msg.documents
        })
        savedMessages.push(saved)
      }

      setHistory(formatToChatHistory(savedMessages))
      setMessages(formatToMessage(savedMessages))
      setHistoryId(newHistory.id)
      if (setContext) {
        setContext([])
      }
      return newHistory.id
    } catch (e) {
      return null
    }
  }

  return async (index: number): Promise<string | null> => {
    // When a server-backed character chat is active, create a new server chat
    // branched from the current context and mirror the prefix messages.
    if (serverChatId) {
      try {
        await tldwClient.initialize().catch(() => null)

        let resolvedTitle = (chatTitle || "").trim()
        let resolvedCharacterId = characterId ?? null
        let resolvedState = normalizeConversationState(
          serverChatState || "in-progress"
        )
        try {
          const chat = await tldwClient.getChat(
            serverChatId,
            scope ? { scope } : undefined
          )
          if (!resolvedTitle) {
            resolvedTitle = (chat?.title || "").trim()
          }
          const chatCharacterId =
            (chat as any)?.character_id ?? (chat as any)?.characterId ?? null
          if (chatCharacterId != null) {
            resolvedCharacterId = chatCharacterId
          }
          resolvedState = normalizeConversationState(
            (chat as any)?.state ??
              (chat as any)?.conversation_state ??
              resolvedState
          )
        } catch (e) {
          // server metadata fetch failed; continue with resolved defaults
        }

        const originalTitle =
          resolvedTitle || (serverChatTopic || "").trim() || "Extension chat"
        const shortId = String(serverChatId).slice(0, 8)
        const base =
          originalTitle.length > 60
            ? `${originalTitle.slice(0, 57)}…`
            : originalTitle
        const branchTitle = `${base} [${shortId}] · msg #${index + 1}`

        if (resolvedCharacterId == null) {
          throw new Error("Cannot branch server chat without character_id")
        }

        const payload: Record<string, any> = {
          title: branchTitle,
          parent_conversation_id: serverChatId,
          state: resolvedState,
          topic_label: serverChatTopic || undefined,
          cluster_id: serverChatClusterId || undefined,
          source: serverChatSource || undefined,
          external_ref: serverChatExternalRef || undefined
        }
        if (resolvedCharacterId != null) {
          payload.character_id = resolvedCharacterId
        }

        const created = await tldwClient.createChat(
          payload,
          scope ? { scope } : undefined
        )
        const rawId =
          (created as any)?.id ?? (created as any)?.chat_id ?? created
        const newChatId = rawId != null ? String(rawId) : ""
        if (!newChatId) {
          throw new Error("Failed to create server branch chat")
        }
        onServerChatMutated?.()

        const snapshot: ChatHistory =
          (history && Array.isArray(history) ? history : []).slice(
            0,
            index + 1
          )

        for (const msg of snapshot) {
          const content = (msg.content || "").trim()
          if (!content) continue
          const role =
            msg.role === "system" ||
            msg.role === "assistant" ||
            msg.role === "user"
              ? msg.role
              : "user"
          await tldwClient.addChatMessage(
            newChatId,
            {
              role,
              content
            },
            scope ? { scope } : undefined
          )
        }

        if (setServerChatId) {
          setServerChatId(newChatId)
        }
        if (setServerChatState) {
          setServerChatState(
            (created as any)?.state ??
              (created as any)?.conversation_state ??
              "in-progress"
          )
        }
        if (setServerChatVersion) {
          setServerChatVersion((created as any)?.version ?? null)
        }
        if (setServerChatTopic) {
          setServerChatTopic((created as any)?.topic_label ?? null)
        }
        if (setServerChatClusterId) {
          setServerChatClusterId((created as any)?.cluster_id ?? null)
        }
        if (setServerChatSource) {
          setServerChatSource((created as any)?.source ?? null)
        }
        if (setServerChatExternalRef) {
          setServerChatExternalRef((created as any)?.external_ref ?? null)
        }
        if (setServerChatTitle) {
          setServerChatTitle(
            String((created as any)?.title ?? chatTitle ?? "")
          )
        }
        if (setServerChatCharacterId) {
          setServerChatCharacterId(
            (created as any)?.character_id ?? characterId ?? null
          )
        }
        if (setServerChatMetaLoaded) {
          setServerChatMetaLoaded(true)
        }

        if (messages && messages.length > 0) {
          const slicedMessages = messages.slice(0, index + 1)
          setMessages(slicedMessages)
          if (history && history.length > 0) {
            setHistory(snapshot)
          }
        }

        return newChatId
      } catch (e) {
        // server branch failed; attempt local fallback
        if (serverOnly) {
          notification.error({
            message: "Branch failed",
            description:
              "Unable to create a branched server chat. Check your server connection and try again."
          })
          return null
        }
        const fallbackTitle = `${String(serverChatId).slice(0, 8)} · msg #${
          index + 1
        }`
        const fallbackId =
          (await createLocalBranch(index)) ??
          (await createLocalBranchFromSnapshot(index, fallbackTitle))
        if (fallbackId) {
          notification.warning({
            message: "Branch fallback",
            description:
              "Server branch failed. Created a local branch instead."
          })
          return fallbackId
        }
        notification.error({
          message: "Branch failed",
          description:
            "Unable to create a branched server chat. Check your server connection and try again."
        })
        return null
      }
    }

    // Local Dexie-backed branch (existing behavior)
    return (
      (await createLocalBranch(index)) ??
      (await createLocalBranchFromSnapshot(index, `Branch · msg #${index + 1}`))
    )
  }
}

export const createStopStreamingRequest = (
  abortController: AbortController | null,
  setAbortController: (controller: AbortController | null) => void
) => {
  return () => {
    if (abortController) {
      abortController.abort()
      setAbortController(null)
    }
  }
}
