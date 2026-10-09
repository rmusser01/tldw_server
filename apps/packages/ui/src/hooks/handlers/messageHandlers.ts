import { getMessageDeleteFeedback } from "@/utils/message-delete-feedback"
import { removeAcknowledgedServerMirrorMessage, serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { nativeRegenerationFeedback } from "@/hooks/chat/native-character-preflight"
import i18n from "i18next"
import { getMessageEditFeedback } from "@/utils/message-edit-feedback"

import { captureChatDraftLoad } from "@/hooks/chat/chat-draft-transition"
import { excludeLocalRagDiagnostics, getLocalRagDiagnosticUser } from "@/utils/local-rag-diagnostic"
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
  nativeHistoryMessagePayload,
  nativeRegenerateView,
  saveNativeHistoryMessage,
  deleteNativeHistoryMessage,
  assertHistoryConversationTarget,
  commitNativeFork,
  prepareNativeFork
} from "@/services/chat-history-selection"
import type { ConversationState } from "@/services/tldw/TldwApiClient"
import type { ChatScope } from "@/types/chat-scope"
import type { ForkRequestV1, ForkResultV1 } from "@/types/history-selection"
import { isGreetingMessageType } from "@/utils/character-greetings"
import { isImageGenerationMessageType } from "@/utils/image-generation-chat"
import type { NotificationInstance } from "antd/es/notification/interface"
import { chatSubmitSkipped, chatSubmitSubmitted, type ChatSubmitResult } from "@/hooks/chat/chat-action-utils"

import { useStoreMessageOption, type ChatHistory, type Message } from "~/store/option"

const editIssueDescription = (issue: unknown) => {
  const copy = getMessageEditFeedback(issue)
  const translated = i18n.t(`common:${copy.key}`, copy.fallback)
  return typeof translated === "string" && translated.trim()
    ? translated
    : copy.fallback
}

/** Derive provider display history from the same visible identities after a successful mutation. */
export const historyFromVisibleMessages = (messages: Message[]): ChatHistory =>
  excludeLocalRagDiagnostics(messages)
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
  getMessages,
  captureViewFence,
  captureSourceFence,
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
  getMessages?: () => Message[]
  captureViewFence?: () => () => boolean
  captureSourceFence?: () => () => boolean
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
  const current = captureViewFence?.() ?? historySelection?.fence?.() ?? (() => true)
  const sourceCurrent = captureSourceFence?.() ?? (() => true)
  return async (dispatch?: { controller?: AbortController; assertCurrent?: () => void }) => {
    dispatch?.controller?.signal.throwIfAborted()
    dispatch?.assertCurrent?.()
    if (!current() || !sourceCurrent()) return chatSubmitSkipped("stale_selection")
    const busy = useStoreMessageOption.getState()
    if (busy.streaming || busy.isProcessing) return chatSubmitSkipped("response_in_progress")
    const origin = historySelection?.getCurrent()
    if (origin?.status === "ready" && origin.owner?.kind === "native" &&
        origin.view && origin.capture?.status === "captured") {
      const reply = origin.capture.rows.at(-1)
      const input = origin.capture.rows.at(-2)
      const before = getMessages?.() ?? messages
      const target = before.at(-1)
      const question = before.at(-2)
      const inputContent = origin.capture.selected_content.at(-2)
      const replyContent = origin.capture.selected_content.at(-1)
      if (!reply || !input || !target?.isBot || question?.isBot ||
          (target.serverMessageId ?? target.id) !== reply.id || (question?.serverMessageId ?? question?.id) !== input.id ||
          before.filter(row => (row.serverMessageId ?? row.id) === reply.id).length !== 1 ||
          before.filter(row => (row.serverMessageId ?? row.id) === input.id).length !== 1 ||
          target.message !== replyContent?.message || question?.message !== inputContent?.message ||
          JSON.stringify(question.images ?? []) !== JSON.stringify(inputContent?.images ?? []))
        return chatSubmitSkipped("message_changed")
      const isCurrent = () => {
        try {
          dispatch?.assertCurrent?.()
          return sourceCurrent() && !dispatch?.controller?.signal.aborted
        } catch { return false }
      }
      const intent = {
        owner_key: origin.view.owner_key, conversation_id: origin.view.conversation_id,
        message_id: reply.id, message_revision: reply.revision,
        input_message_id: input.id, input_message_revision: input.revision,
        isCurrent, isOriginCurrent: () => current() && isCurrent()
      }
      nativeRegenerateView(origin.owner, origin.view, origin.capture, intent)
      if (!validateBeforeSubmitFn()) return chatSubmitSkipped("validation_failed")
      return await onSubmit({ message: inputContent!.message, image: inputContent!.images[0] ?? "", isRegenerate: true,
        historyRegenerate: intent, controller: dispatch?.controller ?? new AbortController(),
        assertCurrent: () => { if (!intent.isOriginCurrent()) throw new Error("stale_selection") } })
    }
    if (!allowOrdinaryRetry || historySelection) {
      const target = [...messages].reverse().find((row) => row.isBot && row.id)
      const error = Object.assign(new Error("unsupported_history_regeneration"), {
        boundary: target ? { kind: "before_message", message_id: target.id } : null
      })
      notification?.error({ message: "Message action unavailable", description: nativeRegenerationFeedback(error, (key, fallback) => String(i18n.t(key, fallback))) })
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
    dispatch?.controller?.signal.throwIfAborted()
    dispatch?.assertCurrent?.()
    const submitHistory = beforeSubmitResult.memory ?? newHistory
    const submitMessages = beforeSubmitResult.messages ?? nextMessages
    const submitExtras = beforeSubmitResult.submitExtras ?? {}

    setHistory(submitHistory)
    setMessages(submitMessages)

    const newController = dispatch?.controller ?? new AbortController()
    return await onSubmit({
      message: userContent,
      image: userImage,
      isRegenerate: true,
      memory: submitHistory,
      messages: submitMessages,
      controller: newController,
      messageType: userMessageType,
      regenerateFromMessage: submitExtras.serverChatIdOverride ? undefined : lastAssistant,
      ...submitExtras,
      assertCurrent: dispatch?.assertCurrent
    })
  }
}

/** A retained confirmation belongs to its rendered H1 and draft source. */
export const createNativeDeleteMessage = (options: {
  messages: Message[]
  getMessages: () => Message[]
  historySelection?: HistorySelectionController | null
  historyId: string | null
  serverChatId?: string | null
  captureSourceFence: () => () => boolean
  notification: NotificationInstance
  invalidateServerChatHistory: () => void
}) => {
  const selection = options.historySelection
  const origin = selection?.getCurrent()
  const viewCurrent = selection?.fence() ?? (() => false)
  const sourceCurrent = options.captureSourceFence()
  const canMutate = () => {
    const state = useStoreMessageOption.getState()
    return !state.streaming && !state.isProcessing
  }
  return async (index: number): Promise<boolean> => {
    if (!viewCurrent() || !sourceCurrent()) return false
    if (
      !selection ||
      origin?.status !== "ready" ||
      origin.owner?.kind !== "native" ||
      origin.capture?.status !== "captured"
    )
      throw new Error("native_history_mutation_unavailable")
    if (!canMutate()) throw new Error("response_in_progress")
    const target = options.messages[index]
    if (!target?.id) throw new Error("missing_message")
    const currentTargets = options
      .getMessages()
      .filter((row) => row.id === target.id)
    if (
      currentTargets.length !== 1 ||
      currentTargets[0].message !== target.message
    )
      throw new Error("message_changed")
    const current = () => sourceCurrent() && canMutate()
    let acknowledged = false
    try {
      await assertHistoryConversationTarget(origin.owner, {
        historyId: options.historyId,
        serverChatId: options.serverChatId
      })
      const receipt = await deleteNativeHistoryMessage(
        origin.owner,
        origin.capture,
        {
          ...target,
          id: target.serverMessageId ?? target.id
        },
        {
          signal: selection.getSignal(),
          validate_lease: () => viewCurrent() && current()
        }
      )
      acknowledged = true
      if (!current() || !viewCurrent()) return false
      options.invalidateServerChatHistory()
      let mirrorFailed = false
      if (options.historyId) {
        try {
          await removeAcknowledgedServerMirrorMessage({
            historyId: options.historyId,
            chatId: origin.owner.conversation_id,
            serverMessageId: receipt.message_id,
            ownerKey: serverChatMirrorOwnerKey({
              requestScope: origin.owner.request_scope
            }),
            scope: origin.owner.scope ?? { type: "global" },
            signal: selection.getSignal(),
            validate_lease: () =>
              viewCurrent() &&
              current() &&
              origin.owner!.kind === "native" &&
              origin.owner.validate_lease()
          })
        } catch {
          mirrorFailed = true
        }
      }
      if (!viewCurrent() || !current()) return false
      const refreshed = await selection.refreshDeletedMessage(
        origin.capture,
        receipt,
        current
      )
      if (!refreshed?.isCurrent()) return false
      if (refreshed.installed && !mirrorFailed) {
        const reply = useStoreMessageOption.getState().replyTarget
        if (reply?.id === target.id || reply?.id === receipt.message_id)
          useStoreMessageOption.getState().clearReplyTarget()
        return refreshed.isCurrent()
      }
      const copy = getMessageDeleteFeedback("message_delete_refresh_failed")
      options.notification.warning({
        message: i18n.t("common:deleted", "Deleted"),
        description: i18n.t(`common:${copy.key}`, copy.fallback)
      })
      return false
    } catch (error) {
      if (!sourceCurrent() || !viewCurrent()) return false
      const copy = getMessageDeleteFeedback(
        acknowledged ? "message_delete_refresh_failed" : error
      )
      const description = i18n.t(`common:${copy.key}`, copy.fallback)
      throw Object.assign(new Error(description), { cause: error })
    }
  }
}

export const createEditMessage = (options: {
  messages: Message[]
  getMessages?: () => Message[]
  historySelection?: HistorySelectionController | null
  history: ChatHistory
  setMessages: (messages: Message[]) => void
  setHistory: (history: ChatHistory) => void
  historyId: string | null
  validateBeforeSubmitFn: () => boolean
  onSubmit: (params: any) => Promise<ChatSubmitResult | void>
  serverChatId?: string | null
  captureViewFence?: () => () => boolean
  captureSourceFence?: () => () => boolean
  mutate?: (target: Message, content: string) => Promise<void>
  notification?: NotificationInstance
}) => {
  const current = options.captureViewFence?.() ?? (() => true)
  const sourceCurrent = options.captureSourceFence?.() ?? (() => true)
  const canMutate = () => {
    const state = useStoreMessageOption.getState()
    return !state.streaming && !state.isProcessing
  }
  return async (
    index: number,
    message: string,
    isHuman: boolean,
    isSend: boolean,
    isEditorCurrent?: () => boolean,
    isEditorOriginCurrent?: () => boolean
  ): Promise<ChatSubmitResult> => {
    if (!current() || !sourceCurrent())
      return chatSubmitSkipped("stale_selection")
    if (!canMutate()) return chatSubmitSkipped("response_in_progress")
    const target = options.messages[index]
    if (!target?.id) throw new Error("missing_message")
    // Unsupported forms must not choose/truncate or mutate the captured stable target.
    if (isHuman && isSend) {
      const origin = options.historySelection?.getCurrent()
      if (
        origin?.status === "ready" &&
        origin.owner?.kind === "native" &&
        origin.view &&
        origin.capture?.status === "captured" &&
        isEditorCurrent?.()
      ) {
        const before = options.getMessages?.() ?? options.messages
        const candidates = before.filter((row) => row.id === target.id)
        const messageId = target.serverMessageId ?? target.id
        const node = origin.capture.snapshot.nodes.find(
          (row) => row.id === messageId
        )
        if (
          candidates.length !== 1 ||
          candidates[0].message !== target.message ||
          node?.role !== "user"
        )
          return chatSubmitSkipped("message_changed")
        nativeHistoryMessagePayload({
          ...target,
          id: messageId,
          history_id: origin.view.conversation_id,
          role: "user",
          content: message,
          createdAt: target.createdAt ?? Date.now()
        })
        if (!options.validateBeforeSubmitFn())
          return chatSubmitSkipped("validation_failed")
        const result = await options.onSubmit({
          message,
          image: target.images?.[0] ?? "",
          historyEditResend: {
            owner_key: origin.view.owner_key,
            conversation_id: origin.view.conversation_id,
            message_id: messageId,
            message_revision: node.revision,
            isCurrent: () => sourceCurrent() && isEditorCurrent(),
            isOriginCurrent: () => current() && sourceCurrent()
          },
          assertCurrent: () => {
            if (!current()) throw new Error("stale_selection")
          }
        })
        if (result) return result
        return chatSubmitSkipped("submission_unacknowledged")
      }
      const error = new Error(`unsupported_history_edit_and_send:${target.id}`)
      options.notification?.error({
        message: "Message action unavailable",
        description: editIssueDescription(error)
      })
      throw error
    }
    const nativeOrigin = options.historySelection?.getCurrent()
    if (nativeOrigin?.owner?.kind === "native") {
      const selection = options.historySelection!
      if (
        nativeOrigin.status !== "ready" ||
        nativeOrigin.capture?.status !== "captured" ||
        !isEditorCurrent?.()
      )
        return chatSubmitSkipped("stale_selection")
      const visible = options.getMessages?.() ?? options.messages
      const candidates = visible.filter((row) => row.id === target.id)
      if (candidates.length !== 1 || candidates[0].message !== target.message)
        return chatSubmitSkipped("message_changed")
      const editorCurrent = () =>
        sourceCurrent() && canMutate() && isEditorCurrent()
      const editorOriginCurrent = () =>
        sourceCurrent() &&
        canMutate() &&
        (isEditorOriginCurrent?.() ?? isEditorCurrent())
      let receipt: Awaited<ReturnType<typeof saveNativeHistoryMessage>>
      try {
        await assertHistoryConversationTarget(nativeOrigin.owner, {
          historyId: options.historyId,
          serverChatId: options.serverChatId
        })
        receipt = await saveNativeHistoryMessage(
          nativeOrigin.owner,
          nativeOrigin.capture,
          { ...target, id: target.serverMessageId ?? target.id },
          message,
          {
            signal: selection.getSignal(),
            validate_lease: () => current() && editorCurrent(),
            acknowledgement_current: () => current() && editorOriginCurrent()
          }
        )
      } catch (error) {
        if (!current() || !editorOriginCurrent())
          return chatSubmitSkipped("stale_selection")
        options.notification?.error({
          message: "Message edit failed",
          description: editIssueDescription(error)
        })
        throw error
      }
      if (!current() || !editorOriginCurrent())
        return chatSubmitSkipped("stale_selection")
      const installed = await selection.refreshEditedMessage(
        nativeOrigin.capture,
        receipt,
        editorOriginCurrent
      )
      return installed && editorOriginCurrent()
        ? chatSubmitSubmitted()
        : chatSubmitSkipped("submission_unacknowledged")
    }
    if (!options.historyId && !options.mutate)
      throw new Error("temporary_history_unavailable")
    const before = options.getMessages?.() ?? options.messages
    const candidates = before.filter((row) => row.id === target.id)
    if (candidates.length !== 1 || candidates[0].message !== target.message)
      return chatSubmitSkipped("message_changed")
    try {
      if (options.mutate) await options.mutate(target, message)
      else await updateMessageById(options.historyId!, target.id, message)
    } catch (error) {
      if (current())
        options.notification?.error({
          message: "Message edit failed",
          description: editIssueDescription(error)
        })
      throw error
    }
    if (!current()) return chatSubmitSkipped("stale_selection")
    const visible = options.getMessages?.() ?? options.messages
    const currentTargets = visible.filter((row) => row.id === target.id)
    if (
      currentTargets.length !== 1 ||
      (currentTargets[0].message !== target.message &&
        currentTargets[0].message !== message)
    )
      return chatSubmitSkipped("message_changed")
    const updated = visible.map((row) =>
      row.id === target.id ? { ...row, message } : row
    )
    options.setMessages(updated)
    if (!current()) return chatSubmitSkipped("stale_selection")
    options.setHistory(historyFromVisibleMessages(updated))
    return current()
      ? chatSubmitSubmitted()
      : chatSubmitSkipped("stale_selection")
  }
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
    const acceptDraft = captureChatDraftLoad("switch")
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
          ? { serverChatId: result.child_id, scope: native.scope, isCurrent: acceptDraft.isCurrent }
          : { historyId: result.child_id, isCurrent: acceptDraft.isCurrent },
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
            if (!acceptDraft(receipt)) return
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
          if (!acceptDraft(receipt)) return
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
