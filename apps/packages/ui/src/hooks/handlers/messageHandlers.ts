import i18n from "i18next"
import { getLocalRagDiagnosticUser } from "@/utils/local-rag-diagnostic"
import type { UploadedFile } from "@/db"
import { commitLocalFork, prepareLocalFork } from "@/db/dexie/branch"
import {
  type ForkDispatchClaim,
  claimForkOperation,
  finishForkOperation,
  forkOperationResult,
  prepareForkOperation,
  recordForkCandidate
} from "@/db/dexie/fork-operations"
import {
  formatToChatHistory,
  formatToMessage,
  updateMessageById
} from "@/db/dexie/helpers"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import {
  commitNativeFork,
  prepareNativeFork
} from "@/services/chat-history-selection"
import type { ConversationState } from "@/services/tldw/TldwApiClient"
import type { ChatScope } from "@/types/chat-scope"
import type { ForkRequestV1, ForkResultV1 } from "@/types/history-selection"
import { isGreetingMessageType } from "@/utils/character-greetings"
import { isImageGenerationMessageType } from "@/utils/image-generation-chat"
import type { NotificationInstance } from "antd/es/notification/interface"
import type { ChatSubmitResult } from "@/hooks/chat/chat-action-utils"

import { type ChatHistory, type Message } from "~/store/option"

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

export const createRegenerateLastMessage = ({
  validateBeforeSubmitFn,
  history,
  messages,
  setHistory,
  setMessages,
  onSubmit,
  beforeSubmit,
  historySelection,
  notification,
  allowOrdinaryRetry = false
}: {
  validateBeforeSubmitFn: () => boolean
  history: ChatHistory
  messages: Message[]
  setHistory: (history: ChatHistory) => void
  setMessages: (messages: Message[]) => void
  onSubmit: (params: any) => Promise<ChatSubmitResult | void>
  historySelection?: HistorySelectionController | null
  notification?: NotificationInstance
  allowOrdinaryRetry?: boolean
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
    if (!allowOrdinaryRetry || historySelection) {
      const target = [...messages].reverse().find((row) => row.isBot && row.id)
      const error = Object.assign(new Error("unsupported_history_regeneration"), {
        boundary: target ? { kind: "before_message", message_id: target.id } : null
      })
      notification?.error({ message: "Message action unavailable", description: error.message })
      throw error
    }
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
    const diagnosticUser = getLocalRagDiagnosticUser(messages, lastAssistant)
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
      diagnosticUser?.message ?? historyUser?.entry?.content ?? messageUser?.message ?? ""
    const userImage = diagnosticUser
      ? diagnosticUser.images?.[0] || ""
      : historyUser?.entry?.image || messageUser?.images?.[0] || ""
    if (!userContent.trim() && !userImage) {
      return
    }
    const userMessageType = diagnosticUser
      ? diagnosticUser.messageType
      : historyUser?.entry?.messageType || messageUser?.messageType

    const newHistory = diagnosticUser ? history : historyUser
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
    return await onSubmit({
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
    historySelection?: HistorySelectionController | null
    onOpened?: (childId: string) => void
    notification: NotificationInstance

    captureViewFence?: () => () => boolean
  }) =>
  async (request: ForkRequestV1): Promise<ForkResultV1> => {
    const binding = {
      operation_id: request.operation_id,
      owner_key: request.owner_key
    }
    const current = options.captureViewFence?.() ?? (() => true)
    const origin = options.historySelection?.getCurrent?.()
    const native = origin?.owner?.kind === "native" ? origin.owner : null
    if ((options.serverChatId || options.serverOnly) && !native)
      return {
        ...binding,
        state: "blocked",
        code: "native_fork_projection_unavailable"
      }
    if (!native && (!options.historyId || options.historyId === "temp"))
      return {
        ...binding,
        state: "blocked",
        code: "temporary_history_unavailable"
      }
    if (
      request.input.selection.conversation_id !==
      (native?.conversation_id ?? options.historyId)
    )
      return {
        ...binding,
        state: "rejected",
        code: "owner_conversation_mismatch"
      }
    let observed: ForkResultV1 | undefined
    let outcomeRecorded = false
    let claim: ForkDispatchClaim | null = null
    try {
      const operation = await prepareForkOperation(request, {
        kind: native ? "native" : "local",
        scope: native?.scope ?? { type: "global" }
      })
      if (
        operation.operation_id !== request.operation_id ||
        operation.state !== "prepared"
      )
        return forkOperationResult(operation)
      claim = await claimForkOperation(operation)
      if (!claim) return forkOperationResult(operation)
      const prepared = native
        ? await prepareNativeFork(native, request, { validate_lease: current })
        : await prepareLocalFork(request, { validate_lease: current })
      if (!current()) throw new Error("stale_selection")
      // Dispatch has begun: a later view change only suppresses application of the actual owner result.
      const result = native
        ? await commitNativeFork(
            prepared as Awaited<ReturnType<typeof prepareNativeFork>>,
            (childId) => recordForkCandidate(claim!, childId)
          )
        : await commitLocalFork(
            prepared as Awaited<ReturnType<typeof prepareLocalFork>>
          )
      observed = result
      await finishForkOperation(claim, result)
      outcomeRecorded = true
      if (
        (result.state !== "committed" && result.state !== "legacy_completed") ||
        !current()
      )
        return result
      const controller = options.historySelection
      if (!controller) return result
      await controller.loadConversation(
        native
          ? { serverChatId: result.child_id, scope: native.scope }
          : { historyId: result.child_id },
        undefined,
        (receipt) => {
          const live = controller.getCurrent()
          if (
            live.status !== "ready" ||
            live.owner !== receipt.owner ||
            live.view !== receipt.view ||
            receipt.owner.kind !== (native ? "native" : "local") ||
            receipt.owner.conversation_id !== result.child_id ||
            receipt.view.conversation_id !== result.child_id ||
            receipt.view.owner_key !== result.owner_key
          )
            return
          if (native) {
            if (
              receipt.owner.kind !== "native" ||
              !receipt.owner.validate_lease()
            )
              return
            const expected = native.scope ?? { type: "global" }
            const actual = receipt.owner.scope ?? { type: "global" }
            if (
              actual.type !== expected.type ||
              (actual.type === "workspace" &&
                expected.type === "workspace" &&
                actual.workspaceId !== expected.workspaceId)
            )
              return
            options.setHistoryId(null)
            options.setContext?.([])
            options.setSelectedSystemPrompt?.("")
            options.setSystemPrompt?.("")
            options.setServerChatMetaLoaded?.(false)
            options.setServerChatId?.(result.child_id)
            options.onOpened?.(result.child_id)
            return
          }
          const local = prepared as Awaited<ReturnType<typeof prepareLocalFork>>
          options.setHistory(formatToChatHistory(local.messages))
          options.setMessages(formatToMessage(local.messages))
          options.setContext?.(local.files?.files ?? [])
          options.setSelectedSystemPrompt?.("")
          options.setSystemPrompt?.(
            local.history.last_used_prompt?.prompt_content ?? ""
          )
          options.setHistoryId(result.child_id)
          options.onOpened?.(result.child_id)
        }
      )
      return result
    } catch (error) {
      const code = error instanceof Error ? error.message : "local_fork_failed"
      if (
        observed?.state === "committed" ||
        observed?.state === "legacy_completed"
      ) {
        if (
          !outcomeRecorded &&
          (!current() || (native && !native.validate_lease()))
        )
          return observed
        options.notification.warning({
          message: outcomeRecorded
            ? i18n.t("playground:historySelection.copyOpenFailed", {
                defaultValue: "Copy saved; opening failed"
              })
            : i18n.t("playground:historySelection.copyRecordFailed", {
                defaultValue: "Copy saved; recovery record update failed"
              }),
          description: outcomeRecorded
            ? code
            : i18n.t("playground:historySelection.copyRecordFailedDetail", {
                defaultValue:
                  "Saved copy: {{childId}}. Its recovery record could not be updated. Reason: {{code}}",
                childId: observed.child_id,
                code
              })
        })
      } else
        options.notification.error({
          message: "Branch failed",
          description: code
        })
      const result = observed ?? {
        ...binding,
        state: "rejected" as const,
        code
      }
      if (!observed && claim)
        await finishForkOperation(claim, result).catch(() => {})
      return result
    } finally {
      await Promise.resolve(
        options.historySelection?.refreshForkOperations?.()
      ).catch(() => {})
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
