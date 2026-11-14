import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat"
import type { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import { loadServicePromptSnapshot } from "@/services/service-prompts"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import React from "react"
import { shallow } from "zustand/shallow"

import { resolveDictionaryChatReferenceId } from "../listUtils"

export function useDictionaryChatContextNavigation(
  owner: Pick<
    ReturnType<typeof useChatDraftOwner>,
    "requestScope" | "isCurrent"
  >,
  notification: {
    warning: (config: { message: string; description?: string }) => void
  }
) {
  const selectServerChat = useSelectServerChat()
  const selectionFence = useHistorySelectionContext()?.fence
  useStoreMessageOption(
    (state) => [
      state.chatDraftRevision,
      state.chatDraftAddress,
      state.historyId,
      state.serverChatId
    ],
    shallow
  )
  usePlaygroundSessionStore(
    (state) => [state.restoreRevision, state.sourceSelectionRevision],
    shallow
  )
  const source = captureChatDraftSource(selectionFence?.())
  const mounted = React.useRef(true)
  React.useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])
  const openChatContextFromDictionary = React.useCallback(
    async (chatRef: unknown, validateModal: () => boolean = () => true) => {
      const chatId = resolveDictionaryChatReferenceId(chatRef)
      const isCurrent = () =>
        mounted.current &&
        owner.isCurrent() &&
        source.isCurrent() &&
        validateModal()
      if (!chatId || !isCurrent() || !owner.requestScope) return false
      let snapshot:
        | Awaited<ReturnType<typeof loadServicePromptSnapshot>>
        | undefined
      try {
        snapshot = await loadServicePromptSnapshot([], {
          requestScope: owner.requestScope
        })
        if (!isCurrent() || snapshot.scopeSignal.aborted) return false
        await tldwClient.initialize()
        if (!isCurrent() || snapshot.scopeSignal.aborted) return false
        const chat = await tldwClient.getChat(chatId, {
          requestScope: owner.requestScope,
          signal: snapshot.scopeSignal,
          scope: { type: "global" },
          requiredMetadata: ["title", "state"]
        })
        if (!isCurrent() || snapshot.scopeSignal.aborted) return false
        if (
          chat.id !== chatId ||
          chat.scope_type !== "global" ||
          chat.workspace_id
        )
          throw new Error("The linked chat scope could not be verified.")
        return selectServerChat(chat, {
          temporaryChat: false,
          isCurrent: () =>
            mounted.current && owner.isCurrent() && validateModal()
        })
      } catch {
        if (isCurrent())
          notification.warning({
            message: "Unable to open linked chat",
            description:
              "Reload the dictionary library and try again. Your current chat is kept."
          })
        return false
      } finally {
        snapshot?.release()
      }
    },
    [owner, source, selectServerChat, notification]
  )
  return { openChatContextFromDictionary }
}
