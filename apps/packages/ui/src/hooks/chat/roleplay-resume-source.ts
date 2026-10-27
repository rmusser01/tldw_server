import { useStoreMessageOption } from "@/store/option"
import type { HistorySelectionController } from "./useHistorySelection"

/** A saved character routes to its protected contract before live setup preparation. */
export const isSavedRoleplayConversation = (
  controller?: HistorySelectionController | null
): boolean => {
  const state = useStoreMessageOption.getState()
  const owner = controller?.getCurrent().owner
  return (
    owner?.kind === "native" &&
    owner.scope?.type !== "workspace" &&
    state.serverChatId === owner.conversation_id &&
    !state.temporaryChat &&
    (state.serverChatAssistantKind === "character" ||
      (state.serverChatAssistantKind == null &&
        state.serverChatCharacterId != null))
  )
}
