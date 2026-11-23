import React from "react"
import { removeMessageById } from "@/db/dexie/helpers"
import { historyFromVisibleMessages } from "@/hooks/handlers/messageHandlers"
import { useStoreMessageOption, type ChatHistory, type Message } from "@/store/option"
import { captureLocalMutationOwner } from "./chat-action-utils"
import { captureChatDraftSource } from "./chat-draft-transition"
import type { HistorySelectionController } from "./useHistorySelection"

type DeleteChatMessageOptions = {
  isMounted: () => boolean
  historySelection: HistorySelectionController | null
  nativeDeleteOrigin: boolean
  nativeDeleteMessage: (index: number) => Promise<boolean>
  historyId: string | null
  serverChatId: string | null
  temporaryChat: boolean
  messages: Message[]
  ordinaryLocalMutation: boolean
  clearReplyTarget: () => void
  setMessages: (value: Message[] | ((previous: Message[]) => Message[])) => void
  setHistory: (value: ChatHistory | ((previous: ChatHistory) => ChatHistory)) => void
  buildHistoryFromMessages: (messages: Message[]) => ChatHistory
}

export const useDeleteChatMessage = ({
  isMounted,
  historySelection,
  nativeDeleteOrigin,
  nativeDeleteMessage,
  historyId,
  serverChatId,
  temporaryChat,
  messages,
  ordinaryLocalMutation,
  clearReplyTarget,
  setMessages,
  setHistory,
  buildHistoryFromMessages
}: DeleteChatMessageOptions) => {
  const deleteViewCurrent = historySelection?.fence()
  const deleteDraftCurrent = captureChatDraftSource(
    isMounted
  ).isCurrent
  return React.useCallback(
    async (index: number) => {
      const deleteSourceCurrent = () =>
        deleteDraftCurrent() && (deleteViewCurrent?.() ?? true)
      const deleteCurrent = () => {
        const state = useStoreMessageOption.getState()
        return deleteSourceCurrent() && !state.streaming && !state.isProcessing
      }
      if (!deleteSourceCurrent()) return false
      if (!deleteCurrent()) throw new Error("response_in_progress")
      const target = messages[index]
      if (!target?.id) throw new Error("missing_message")
      if (nativeDeleteOrigin) return nativeDeleteMessage(index)
      if (serverChatId) throw new Error("native_history_mutation_unavailable")
      const targetCurrent = () => {
        const liveTargets = useStoreMessageOption
          .getState()
          .messages.filter((row) => row.id === target.id)
        return (
          liveTargets.length === 1 &&
          liveTargets[0].message === target.message &&
          liveTargets[0].isBot === target.isBot &&
          liveTargets[0].role === target.role &&
          JSON.stringify(liveTargets[0].images ?? []) ===
            JSON.stringify(target.images ?? [])
        )
      }
      if (!targetCurrent()) throw new Error("message_changed")
      const deletionCurrent = () => deleteCurrent() && targetCurrent()
      const selectedHistory = ordinaryLocalMutation ? null : historySelection
      if (
        !selectedHistory &&
        !historySelection &&
        !temporaryChat &&
        !serverChatId
      )
        throw new Error("history_selection_unavailable")
      if (!selectedHistory) {
        const serverMessageId = target.serverMessageId
        try {
          if (historyId && historyId !== "temp")
            await removeMessageById(historyId, target.id, undefined, {
              validate_lease: deletionCurrent
            })
          else if (!temporaryChat)
            throw new Error("history_selection_unavailable")
        } catch (error) {
          if (!deletionCurrent()) return false
          throw error
        }
        if (!deletionCurrent()) return false
        const reply = useStoreMessageOption.getState().replyTarget
        if (reply?.id === target.id || reply?.id === serverMessageId)
          clearReplyTarget()
        if (!deleteCurrent()) return false
        setMessages((prev) =>
          deleteCurrent() ? prev.filter((row) => row.id !== target.id) : prev
        )
        if (!deleteCurrent()) return false
        setHistory((prev) =>
          deleteCurrent()
            ? buildHistoryFromMessages(
                useStoreMessageOption
                  .getState()
                  .messages.filter((row) => row.id !== target.id)
              )
            : prev
        )
        return deleteCurrent()
      }
      const origin = selectedHistory.getCurrent()
      const current = deleteCurrent
      try {
        const owner = captureLocalMutationOwner(
          selectedHistory,
          historyId,
          serverChatId
        )
        const removed = await removeMessageById(
          owner.conversation_id,
          target.id,
          owner,
          { validate_lease: deletionCurrent }
        )
        if (!deletionCurrent()) return false
        const reply = useStoreMessageOption.getState().replyTarget
        if (reply?.id === target.id) clearReplyTarget()
        const remaining = useStoreMessageOption
          .getState()
          .messages.filter((row) => row.id !== target.id)
        if (!current()) return false
        setMessages(remaining)
        if (!current()) return false
        setHistory((previous) =>
          current()
            ? historyFromVisibleMessages(
                useStoreMessageOption
                  .getState()
                  .messages.filter((row) => row.id !== target.id)
              )
            : previous
        )
        const view = origin?.view
        if (
          removed &&
          view?.interpretation.kind === "parent_graph_v1" &&
          view.cursor.kind !== "empty" &&
          view.cursor.message_id === target.id
        ) {
          if (!current()) return false
          const following = selectedHistory.choose(
            removed.parent_message_id
              ? { kind: "after_message", message_id: removed.parent_message_id }
              : { kind: "empty" }
          )
          const followingCurrent = selectedHistory.fence() ?? (() => false)
          const followed = await following
          const state = useStoreMessageOption.getState()
          return (
            followed === true &&
            followingCurrent() &&
            deleteDraftCurrent() &&
            !state.streaming &&
            !state.isProcessing
          )
        }
        return current()
      } catch (error) {
        if (!deletionCurrent()) return false
        throw error
      }
    },
    [
      nativeDeleteMessage,
      nativeDeleteOrigin,
      deleteViewCurrent,
      deleteDraftCurrent,
      historyId,
      serverChatId,
      temporaryChat,
      messages,
      historySelection,
      ordinaryLocalMutation,
      clearReplyTarget,
      setMessages,
      setHistory,
      buildHistoryFromMessages
    ]
  )
}
