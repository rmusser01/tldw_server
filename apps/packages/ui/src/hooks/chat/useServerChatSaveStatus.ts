import { useStore } from "zustand"
import {
  normalizeServerChatId,
  serverChatSaveStatusStore,
  toServerChatSaveStatus,
  type ServerChatSaveStatus
} from "@/store/server-chat-save-status"

/** Acknowledged save status of a server chat; re-renders when it changes. */
export const useServerChatSaveStatus = (
  chatId: unknown
): ServerChatSaveStatus => {
  const id = normalizeServerChatId(chatId)
  return useStore(serverChatSaveStatusStore, (state) =>
    id ? toServerChatSaveStatus(state.entries[id]) : "unknown"
  )
}
