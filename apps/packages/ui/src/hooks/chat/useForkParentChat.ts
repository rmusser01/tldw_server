import { useQuery, type QueryClient } from "@tanstack/react-query"
import type { ServerChatHistoryItem } from "@/hooks/useServerChatHistory"
import { getQueryClient } from "@/services/query-client"
import { tldwClient } from "@/services/tldw/TldwApiClient"

export type ForkParentChat = { id: string; title: string }

const parentIdOf = (chat: { parent_conversation_id?: string | null } | null | undefined, chatId: string) => {
  const parentId = typeof chat?.parent_conversation_id === "string" ? chat.parent_conversation_id.trim() : ""
  return parentId && parentId !== chatId ? parentId : null
}

/**
 * The parent from a chat list the sidebar already read, so opening a listed
 * chat costs no extra read. `undefined` when no cached list has the chat.
 */
const listedParentId = (client: QueryClient, chatId: string): string | null | undefined => {
  for (const [, data] of client.getQueriesData<{ items?: ServerChatHistoryItem[] }>({ queryKey: ["serverChatHistory"] })) {
    const listed = data?.items?.find((chat) => chat.id === chatId)
    if (listed) return parentIdOf(listed, chatId)
  }
  return undefined
}

/**
 * The saved chat the open chat was branched from (CM-12), or `null` for a chat
 * that isn't a branch or whose parent can't be read (for example a deleted
 * one). The chat page's header uses it to lead back to the parent.
 *
 * A chat's parent never changes, so it is read once per chat (from the
 * sidebar's list when it has the chat), and the parent's title once per
 * parent, through the app's shared query cache.
 */
export const useForkParentChat = (serverChatId: string | null | undefined): ForkParentChat | null => {
  // The app's provider holds this same client; passing it keeps the hook
  // usable where no provider is mounted.
  const client = getQueryClient()
  const lineage = useQuery(
    {
      queryKey: ["chat-lineage", serverChatId],
      enabled: Boolean(serverChatId),
      staleTime: Infinity,
      retry: false,
      queryFn: async ({ signal }) => {
        const listed = listedParentId(client, serverChatId!)
        if (listed !== undefined) return listed
        return parentIdOf(await tldwClient.getChat(serverChatId!, { signal }), serverChatId!)
      }
    },
    client
  )
  const parentId = serverChatId ? lineage.data ?? null : null
  const parent = useQuery(
    {
      queryKey: ["chat-title", parentId],
      enabled: Boolean(parentId),
      staleTime: 5 * 60 * 1000,
      retry: false,
      queryFn: async ({ signal }) => {
        const chat = await tldwClient.getChat(parentId!, { signal })
        return typeof chat?.title === "string" ? chat.title : ""
      }
    },
    client
  )
  return parentId && parent.isSuccess ? { id: parentId, title: parent.data } : null
}
