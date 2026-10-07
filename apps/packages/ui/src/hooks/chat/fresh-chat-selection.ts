/**
 * Fresh-chat invariant for the history-selection controller (CS-01, CS-05; #3106).
 *
 * New chat and Clear leave the chat surface addressing no conversation, and the
 * Playground then resets its HistorySelection controller. A controller that is
 * not idle while no conversation is addressed still holds the owner, view or
 * capture of a conversation the user left (or one still being restored). A send
 * built from it would carry that conversation's turns, or be appended to it.
 */
import type { HistorySelectionController } from "./useHistorySelection"

export type ChatConversationTarget = {
  historyId?: string | null
  serverChatId?: string | null
}

type SelectionStatus = Pick<HistorySelectionController, "status">
type ResettableSelection = {
  getCurrent: () => SelectionStatus
  reset: () => void
}

/** No saved local history and no server chat is addressed ("temp" marks a temporary chat). */
export const addressesNoConversation = ({
  historyId,
  serverChatId
}: ChatConversationTarget): boolean =>
  (!historyId || historyId === "temp") && !serverChatId

/**
 * True while a chat that addresses no conversation still has a non-idle
 * controller. Send stays disabled until this is false.
 */
export const isFreshChatSelectionPending = (
  selection: SelectionStatus | null | undefined,
  target: ChatConversationTarget
): boolean =>
  selection != null &&
  selection.status !== "idle" &&
  addressesNoConversation(target)

/**
 * Defensive check before a turn is built: a turn that addresses no conversation
 * must never use another conversation's settled selection, so reset it and let
 * the turn start a new conversation. A selection that is still loading (a
 * restore or handoff in progress) is left alone; the send path refuses it as
 * not ready. Returns true when the controller was reset.
 */
export const resetStaleSelectionForFreshTurn = (
  controller: ResettableSelection | null | undefined,
  target: ChatConversationTarget
): boolean => {
  const current = controller?.getCurrent()
  if (
    !controller ||
    !current ||
    current.status === "loading" ||
    !isFreshChatSelectionPending(current, target)
  )
    return false
  controller.reset()
  return true
}
