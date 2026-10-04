import type { ChatHistory, Message } from "@/store/option"

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
