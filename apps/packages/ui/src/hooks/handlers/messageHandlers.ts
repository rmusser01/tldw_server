import { type ChatHistory, type Message } from "~/store/option"
import {
  formatToChatHistory,
  formatToMessage,
  updateMessageById
} from "@/db/dexie/helpers"
import { prepareLocalFork, commitLocalFork } from "@/db/dexie/branch"
import type { UploadedFile } from "@/db"
import type { ConversationState } from "@/services/tldw/TldwApiClient"
import type { NotificationInstance } from "antd/es/notification/interface"
import type { ChatScope } from "@/types/chat-scope"
import type { ForkRequestV1, ForkResultV1 } from "@/types/history-selection"
import { isGreetingMessageType } from "@/utils/character-greetings"
import { isImageGenerationMessageType } from "@/utils/image-generation-chat"

/** Derive provider display history from the same visible identities after a successful mutation. */
export const historyFromVisibleMessages = (messages: Message[]): ChatHistory =>
  messages
    .filter(
      (message) =>
        !isGreetingMessageType(message.messageType) &&
        !isImageGenerationMessageType(message.messageType)
    )
    .map((message) => ({
      role: message.role ?? (message.isBot ? "assistant" : "user"),
      content: message.message,
      images: message.images,
      image: message.images?.[0],
      messageType: message.messageType
    }))

/** Same-parent regeneration requires an admitted-parent protocol; resubmitting a user is not regeneration. */
export const createRegenerateLastMessage =
  (options: {
    validateBeforeSubmitFn: () => boolean
    history: ChatHistory
    messages: Message[]
    setHistory: (history: ChatHistory) => void
    setMessages: (messages: Message[]) => void
    onSubmit: (params: any) => Promise<unknown>
    beforeSubmit?: (params: any) => Promise<any>
    notification?: NotificationInstance
  }) =>
  async () => {
    const target = [...options.messages]
      .reverse()
      .find((row) => row.isBot && row.id)
    const error = Object.assign(new Error("unsupported_history_regeneration"), {
      boundary: target
        ? { kind: "before_message", message_id: target.id }
        : null
    })
    options.notification?.error({
      message: "Message action unavailable",
      description: error.message
    })
    throw error
  }

export const createEditMessage =
  (options: {
    messages: Message[]
    history: ChatHistory
    setMessages: (messages: Message[]) => void
    setHistory: (history: ChatHistory) => void
    historyId: string | null
    validateBeforeSubmitFn: () => boolean
    onSubmit: (params: any) => Promise<unknown>
    captureViewFence?: () => () => boolean
    mutate?: (target: Message, content: string) => Promise<void>
    notification?: NotificationInstance
  }) =>
  async (index: number, message: string, isHuman: boolean, isSend: boolean) => {
    const target = options.messages[index]
    if (!target?.id) throw new Error("missing_message")
    // Capture the stable action boundary synchronously. Unsupported forms must not choose/truncate.
    const boundary = { kind: "before_message", message_id: target.id } as const
    if (isHuman && isSend) {
      const error = new Error(
        `unsupported_history_edit_and_send:${boundary.message_id}`
      )
      options.notification?.error({
        message: "Message action unavailable",
        description: error.message
      })
      throw error
    }
    if (!options.historyId && !options.mutate)
      throw new Error("temporary_history_unavailable")
    const current = options.captureViewFence?.() ?? (() => true)
    try {
      if (options.mutate) await options.mutate(target, message)
      else await updateMessageById(options.historyId!, target.id, message)
    } catch (error) {
      options.notification?.error({
        message: "Message edit failed",
        description:
          error instanceof Error ? error.message : "message_edit_failed"
      })
      throw error
    }
    if (!current()) return
    const updated = options.messages.map((row) =>
      row.id === target.id ? { ...row, message } : row
    )
    options.setMessages(updated)
    options.setHistory(historyFromVisibleMessages(updated))
  }

export const createBranchMessage =
  (options: {
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

    captureViewFence?: () => () => boolean
  }) =>
  async (request: ForkRequestV1): Promise<ForkResultV1> => {
    const binding = {
      operation_id: request.operation_id,
      owner_key: request.owner_key
    }
    const current = options.captureViewFence?.() ?? (() => true)
    if (options.serverChatId || options.serverOnly)
      return {
        ...binding,
        state: "blocked",
        code: "native_fork_projection_unavailable"
      }
    if (!options.historyId || options.historyId === "temp")
      return {
        ...binding,
        state: "blocked",
        code: "temporary_history_unavailable"
      }
    if (request.input.selection.conversation_id !== options.historyId)
      return {
        ...binding,
        state: "rejected",
        code: "owner_conversation_mismatch"
      }
    let observed: ForkResultV1 | undefined
    try {
      const prepared = await prepareLocalFork(request, {
        validate_lease: current
      })
      if (!current())
        return { ...binding, state: "rejected", code: "stale_selection" }
      // Dispatch has begun: a later view change only suppresses application of the actual owner result.
      const result = await commitLocalFork(prepared)
      observed = result
      if (result.state !== "committed" || !current()) return result
      options.setHistory(formatToChatHistory(prepared.messages))
      options.setMessages(formatToMessage(prepared.messages))
      options.setContext?.(prepared.files?.files ?? [])
      options.setSelectedSystemPrompt?.("")
      options.setSystemPrompt?.(
        prepared.history.last_used_prompt?.prompt_content ?? ""
      )
      options.setHistoryId(result.child_id)
      return result
    } catch (error) {
      const code = error instanceof Error ? error.message : "local_fork_failed"
      options.notification.error({
        message: "Branch failed",
        description: code
      })
      return observed ?? { ...binding, state: "rejected", code }
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
