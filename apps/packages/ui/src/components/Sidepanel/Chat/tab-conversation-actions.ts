/**
 * The real conversation behind a side-panel tab, for the tab menu's Rename and
 * Delete (XS-07). A tab bound to a server chat is a view of that chat, so these
 * act on the server first and keep the local copy in step; a tab with only a
 * local history renames that history.
 */
import { removeServerChatMirror } from "@/db/dexie/server-chat-mirror"
import { updateHistory } from "@/db/dexie/helpers"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { SidepanelChatTab } from "@/store/sidepanel-chat-tabs"

/**
 * Rename the conversation a tab shows: its server chat when it has one, then
 * its local history. Returns the title to show. Throws, changing nothing, if
 * the server rename fails.
 */
export async function renameTabConversation(
  tab: Pick<SidepanelChatTab, "historyId" | "serverChatId">,
  title: string,
  requestScope?: ServicePromptRequestScope
): Promise<string> {
  let resolvedTitle = title
  if (tab.serverChatId) {
    if (!requestScope) throw new Error("A server chat is renamed only for a verified account")
    const updated = await tldwClient.updateChat(tab.serverChatId, { title }, { requestScope })
    resolvedTitle = updated?.title || title
  }
  if (tab.historyId) {
    try {
      await updateHistory(tab.historyId, resolvedTitle)
    } catch (error) {
      // The server holds the name; the local copy takes it when next linked.
      console.warn("[sidepanel] Could not rename the local copy of a chat", error)
    }
  }
  return resolvedTitle
}

/** Move a server chat to Trash: a soft delete that can be undone or restored from Trash. */
export const moveServerChatToTrash = (
  chatId: string,
  requestScope: ServicePromptRequestScope
): Promise<void> => tldwClient.deleteChat(chatId, { requestScope })

/**
 * Drop this owner's local copy of a chat that is now in Trash, so local
 * history search stops listing it. Reopening the chat after a restore links a
 * fresh copy.
 */
export async function removeLocalCopyOfServerChat(chatId: string, ownerKey: string): Promise<void> {
  try {
    await removeServerChatMirror({ chatId, ownerKey })
  } catch (error) {
    console.warn("[sidepanel] Could not remove the local copy of a deleted chat", error)
  }
}

/** Bring a chat back from Trash. */
export const restoreServerChatFromTrash = (chatId: string) => tldwClient.restoreChat(chatId)
