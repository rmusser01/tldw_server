import React from "react"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import type { Message } from "@/store/option"
import {
  isHistoryTurnLive,
  isRetainedPartialReply,
  keepHistoryTurnRecovery,
  onHistoryTurnReleased,
  retainedReplyMessage,
  type HistoryTurnRecoveryEntry
} from "@/services/history-turn-keep"

type SetMessages = (
  messages: Message[] | ((previous: Message[]) => Message[])
) => void

const inputIdOf = (entry: HistoryTurnRecoveryEntry) =>
  entry.turn.admission?.input_message_id ?? entry.turn.input_id

/**
 * Keeps the open chat's interrupted and orphaned turns in its transcript
 * (CS-04, #3104).
 *
 * When a chat opens (after a reload, or on return to /chat) or a turn of this
 * page ends off-screen, turns retained with an `outcome` are put back into the
 * selected history instead of waiting in review. A partial reply of a server
 * chat, which only this device holds, is shown after its question as an
 * interrupted reply until the user retries or moves on.
 *
 * Returns whether a retained turn is shown in the transcript or handled here,
 * so the review panel can leave it out.
 */
export function useKeepInterruptedHistoryTurns({
  selection,
  messages,
  setMessages
}: {
  selection: HistorySelectionController | null | undefined
  messages: Message[]
  setMessages: SetMessages
}) {
  const selectionRef = React.useRef(selection)
  selectionRef.current = selection
  // Retained partial server replies this view already kept (it followed
  // their question); their records stay for the transcript bubble.
  const keptRef = React.useRef(new Set<string>())
  const recoveries = selection?.recoveries
  const status = selection?.status
  const ownerKind =
    selection?.owner?.kind === "local" || selection?.owner?.kind === "native"
      ? selection.owner.kind
      : undefined

  React.useEffect(() => {
    const controller = selectionRef.current
    if (!controller || status !== "ready" || !recoveries?.length) return
    const pending = recoveries
      .filter(
        (entry) =>
          entry.turn.outcome &&
          !isHistoryTurnLive(entry.turn.operation_id) &&
          !keptRef.current.has(entry.turn.operation_id)
      )
      .sort((a, b) => a.turn.created_at - b.turn.created_at)
    if (!pending.length) return
    let cancelled = false
    void (async () => {
      for (const entry of pending) {
        if (cancelled) return
        const result = await keepHistoryTurnRecovery(controller, entry)
        if (result.status === "kept" && result.retained)
          keptRef.current.add(entry.turn.operation_id)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [recoveries, status])

  // A turn of this page that ended off-screen (e.g. it finished after the
  // user left and came back) leaves a record for this view to keep.
  React.useEffect(
    () =>
      onHistoryTurnReleased(() => {
        void selectionRef.current?.refreshRecovery?.()
      }),
    []
  )

  const retainedReplies = React.useMemo(
    () =>
      ownerKind === "native"
        ? (recoveries ?? []).filter((entry) =>
            isRetainedPartialReply(entry.turn, ownerKind)
          )
        : [],
    [ownerKind, recoveries]
  )

  React.useEffect(() => {
    if (!retainedReplies.length) return
    const last = messages[messages.length - 1]
    for (const entry of retainedReplies) {
      const inputId = inputIdOf(entry)
      const index = messages.findIndex((message) => message.id === inputId)
      if (index < 0) continue
      const next = messages[index + 1]
      if (index === messages.length - 1 && last && !last.isBot) {
        // The question is the last message: show the kept partial reply.
        setMessages((previous) =>
          previous[previous.length - 1]?.id !== inputId ||
          previous.some((message) => message.id === entry.turn.assistant_id)
            ? previous
            : [...previous, retainedReplyMessage(entry.turn)]
        )
      } else if (next && next.id !== entry.turn.assistant_id) {
        // The user continued after it: the cut-off reply is no longer needed.
        void selectionRef.current?.dismissRecovery(entry)
      }
    }
  }, [messages, retainedReplies, setMessages])

  return React.useCallback(
    (entry: HistoryTurnRecoveryEntry) =>
      isHistoryTurnLive(entry.turn.operation_id) ||
      (Boolean(entry.turn.outcome) &&
        (!isRetainedPartialReply(entry.turn, ownerKind) ||
          messages.some((message) => message.id === entry.turn.assistant_id))),
    [messages, ownerKind]
  )
}
