import { db } from "./schema"

/**
 * Why the local chat `historyId` must not be promoted to a new server chat,
 * or null when it may be (CS-N3, #3104).
 *
 * - "history_selection_owned": the selected-history controller owns the chat
 *   and writes each turn to it itself (the question before the reply
 *   streams). Promoting it from the Playground's in-memory transcript, which
 *   can be stale (for example on return to /chat after an interrupted reply),
 *   created empty or partial server chats. Saving such chats to the server by
 *   default is Stage 2 (D1).
 * - "server_linked": the chat already mirrors a server chat.
 *
 * A draft with no local history yet keeps today's promotion of its in-memory
 * transcript (e.g. a greeting-only chat).
 */
export type ChatPromotionBlocker = "history_selection_owned" | "server_linked"

export const getChatPromotionBlocker = async (
  historyId: string | null | undefined
): Promise<ChatPromotionBlocker | null> => {
  if (!historyId || historyId === "temp") return null
  const history = await db.chatHistories.get(historyId)
  if (!history) return null
  if (history.server_chat_id || history.message_source === "server")
    return "server_linked"
  if (history.local_owner_key) return "history_selection_owned"
  return null
}
