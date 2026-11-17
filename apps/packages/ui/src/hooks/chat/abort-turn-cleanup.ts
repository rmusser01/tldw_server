import type { ChatHistory, Message } from "@/store/option"
import type { HistorySelectionController } from "./useHistorySelection"
import type { HistoryCursorV1 } from "@/types/history-selection"

export const isAbortLikeError = (error: unknown): boolean => {
  if (error instanceof Error) {
    if (error.name === "AbortError") return true
    return error.message.toLowerCase().includes("abort")
  }

  return String(error || "")
    .toLowerCase()
    .includes("abort")
}

export const discardAbortedTurnIfRequested = ({
  discardRequested,
  error,
  previousMessages,
  previousHistory,
  setMessages,
  setHistory
}: {
  discardRequested: boolean
  error: unknown
  previousMessages: Message[]
  previousHistory: ChatHistory
  setMessages: (
    messagesOrUpdater: Message[] | ((prev: Message[]) => Message[])
  ) => void
  setHistory: (
    historyOrUpdater: ChatHistory | ((prev: ChatHistory) => ChatHistory)
  ) => void
}): boolean => {
  if (!discardRequested || !isAbortLikeError(error)) {
    return false
  }

  setMessages(previousMessages)
  setHistory(previousHistory)
  return true
}

// Turns the user stopped on purpose, so the kept reply reads "Stopped" rather
// than "Interrupted" (CS-04, #3104). Keyed by the turn's own abort signal.
const userStoppedSignals = new WeakSet<AbortSignal>()

export const markChatTurnStoppedByUser = (signal: AbortSignal) => {
  userStoppedSignals.add(signal)
}

export const wasChatTurnStoppedByUser = (
  signal: AbortSignal | null | undefined
): boolean => Boolean(signal && userStoppedSignals.has(signal))

// The response may become idle before its stopped history installation finishes.
type ChatTurnSettlement = {
  pending: boolean
  focusReturn: {
    history: Pick<HistorySelectionController, "getCurrent"> | null
    isCurrent: () => boolean
    acceptHistory: (isCurrent: () => boolean) => void
    finish: () => void
  } | null
  finish: () => void
}
const turnSettlements = new WeakMap<AbortSignal, ChatTurnSettlement>()

export const getChatTurnSettlement = (signal: AbortSignal) =>
  turnSettlements.get(signal)

export const beginChatTurnSettlement = (
  signal: AbortSignal
): ChatTurnSettlement => {
  const settlement: ChatTurnSettlement = {
    pending: true,
    focusReturn: null,
    finish: () => {
      if (!settlement.pending) return
      settlement.pending = false
      if (turnSettlements.get(signal) === settlement)
        settlement.focusReturn?.finish()
    }
  }
  turnSettlements.set(signal, settlement)
  return settlement
}

/** Adopt only the stopped response's own admitted and accepted H1 installation. */
export const followStoppedChatTurnSelection = async (
  signal: AbortSignal | undefined,
  controller: HistorySelectionController,
  cursor: HistoryCursorV1
): Promise<boolean> => {
  const settlement = signal ? turnSettlements.get(signal) : undefined
  const focus = settlement?.focusReturn
  const origin = controller.getCurrent()
  const owner = origin.owner
  const matchesResultView = (view: typeof origin.view) =>
    Boolean(
      origin.view &&
        view &&
        view.owner_key === origin.view.owner_key &&
        view.conversation_id === origin.view.conversation_id &&
        view.view_session_id === origin.view.view_session_id &&
        view.selection_revision === origin.view.selection_revision + 1 &&
        JSON.stringify(view.interpretation) ===
          JSON.stringify(origin.view.interpretation) &&
        JSON.stringify(view.cursor) === JSON.stringify(cursor)
    )
  const admitted =
    signal &&
    wasChatTurnStoppedByUser(signal) &&
    settlement?.pending &&
    focus?.history?.getCurrent === controller.getCurrent &&
    focus.isCurrent()
  const following = controller.choose(cursor)
  const installedIsCurrent = controller.fence()
  const admittedChoice = matchesResultView(controller.getCurrent().view)
  let pending = true,
    accepted = false
  let installed: ReturnType<HistorySelectionController["getCurrent"]> | null =
    null
  if (admitted && admittedChoice)
    focus.acceptHistory(() => {
      const current = controller.getCurrent()
      return (
        turnSettlements.get(signal) === settlement &&
        installedIsCurrent() &&
        current.owner === owner &&
        (pending ||
          (accepted &&
            current.status === "ready" &&
            current.view === installed?.view &&
            current.capture === installed?.capture))
      )
    })
  try {
    const followed = await following
    installed = controller.getCurrent()
    accepted =
      followed &&
      installedIsCurrent() &&
      installed.owner === owner &&
      admittedChoice &&
      installed.status === "ready" &&
      matchesResultView(installed.view) &&
      Boolean(installed.capture)
    return followed
  } finally {
    pending = false
  }
}
