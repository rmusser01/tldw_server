import React from "react"
import { tldwClient } from "@/services/tldw/TldwApiClient"

export type ForkParentChat = { id: string; title: string }

/**
 * The saved chat the open chat was branched from (CM-12), or `null` for a chat
 * that isn't a branch or whose parent can't be read (for example a deleted
 * one). The chat page's header uses it to lead back to the parent.
 */
export const useForkParentChat = (serverChatId: string | null | undefined): ForkParentChat | null => {
  const [found, setFound] = React.useState<{ chatId: string; parent: ForkParentChat | null } | null>(null)
  React.useEffect(() => {
    if (!serverChatId) return
    const controller = new AbortController()
    const settle = (parent: ForkParentChat | null) => {
      if (!controller.signal.aborted) setFound({ chatId: serverChatId, parent })
    }
    void (async () => {
      try {
        const chat = await tldwClient.getChat(serverChatId, { signal: controller.signal })
        const parentId = typeof chat?.parent_conversation_id === "string" ? chat.parent_conversation_id.trim() : ""
        if (controller.signal.aborted) return
        if (!parentId || parentId === serverChatId) return settle(null)
        const parent = await tldwClient.getChat(parentId, { signal: controller.signal })
        settle({ id: parentId, title: typeof parent?.title === "string" ? parent.title : "" })
      } catch {
        // Lineage is a convenience: a chat whose parent can't be read shows none.
        settle(null)
      }
    })()
    return () => controller.abort()
  }, [serverChatId])
  return serverChatId && found?.chatId === serverChatId ? found.parent : null
}
