import { createStore } from "zustand/vanilla"

/**
 * Acknowledged save status of server chats, keyed by server chat id.
 *
 * Chat persistence labels ("Saved on server", "Saved on this device", ...)
 * must report state the server has acknowledged, never intent or connectivity
 * (CS-03 / XS-05, #3104). Every server chat write records its outcome here:
 * - "saving": at least one write for the chat is still in flight;
 * - "saved": the most recent settled write was acknowledged by the server;
 * - "failed": the most recent settled write was not acknowledged;
 * - "unknown": no write was recorded in this session. A chat id is only ever
 *   issued by the server, so an unknown chat is one the server already holds.
 *
 * Recording an outcome never changes how or whether a chat is saved. The store
 * is framework-free so the API client can record writes; React components read
 * it through `useServerChatSaveStatus` (hooks/chat/useServerChatSaveStatus).
 */
export type ServerChatSaveStatus = "unknown" | "saving" | "saved" | "failed"

/** "unknown" ends a write without changing the last acknowledged outcome. */
export type ServerChatWriteOutcome = "saved" | "failed" | "unknown"

export type ServerChatSaveEntry = {
  inFlight: number
  lastOutcome: "saved" | "failed" | null
}

type ServerChatSaveStatusState = {
  entries: Record<string, ServerChatSaveEntry>
}

const EMPTY_ENTRY: ServerChatSaveEntry = { inFlight: 0, lastOutcome: null }

export const serverChatSaveStatusStore =
  createStore<ServerChatSaveStatusState>(() => ({ entries: {} }))

export const normalizeServerChatId = (chatId: unknown): string | null => {
  if (typeof chatId !== "string" && typeof chatId !== "number") return null
  const normalized = String(chatId).trim()
  return normalized.length > 0 ? normalized : null
}

const updateEntry = (
  chatId: string,
  update: (entry: ServerChatSaveEntry) => ServerChatSaveEntry
) => {
  serverChatSaveStatusStore.setState((state) => ({
    entries: {
      ...state.entries,
      [chatId]: update(state.entries[chatId] ?? EMPTY_ENTRY)
    }
  }))
}

export const toServerChatSaveStatus = (
  entry: ServerChatSaveEntry | undefined
): ServerChatSaveStatus => {
  if (!entry) return "unknown"
  if (entry.inFlight > 0) return "saving"
  return entry.lastOutcome ?? "unknown"
}

/**
 * Marks a server write for `chatId` as in flight. Call the returned function
 * exactly once with the outcome; later calls are ignored.
 */
export const beginServerChatWrite = (
  chatId: unknown
): ((outcome: ServerChatWriteOutcome) => void) => {
  const id = normalizeServerChatId(chatId)
  if (!id) return () => undefined
  updateEntry(id, (entry) => ({ ...entry, inFlight: entry.inFlight + 1 }))
  let ended = false
  return (outcome) => {
    if (ended) return
    ended = true
    updateEntry(id, (entry) => ({
      inFlight: Math.max(0, entry.inFlight - 1),
      lastOutcome: outcome === "unknown" ? entry.lastOutcome : outcome
    }))
  }
}

/**
 * Runs a server write for `chatId` and records whether the server acknowledged
 * it. The write's result or error is passed through unchanged.
 */
export const trackServerChatWrite = async <T>(
  chatId: unknown,
  write: () => Promise<T>,
  options?: { isAcknowledgedError?: (error: unknown) => boolean }
): Promise<T> => {
  const endWrite = beginServerChatWrite(chatId)
  try {
    const result = await write()
    endWrite("saved")
    return result
  } catch (error) {
    endWrite(options?.isAcknowledgedError?.(error) ? "saved" : "failed")
    throw error
  }
}

export const getServerChatSaveStatus = (chatId: unknown): ServerChatSaveStatus => {
  const id = normalizeServerChatId(chatId)
  return id
    ? toServerChatSaveStatus(serverChatSaveStatusStore.getState().entries[id])
    : "unknown"
}

export const resetServerChatSaveStatus = () => {
  serverChatSaveStatusStore.setState({ entries: {} })
}
