/**
 * Per-chat export read from the server (CS-09).
 *
 * Since D7 most chats exist only on the server, so an export that reads the
 * device's copy fails for them. This reads the saved chat page by page, on the
 * path the chat shows by default (its newest branch, CM-N2), and saves it as
 * Markdown or JSON.
 */
import { fetchAllServerChatMessages } from "@/hooks/chat/useServerChatLoader"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import type { ChatScope } from "@/types/chat-scope"
import { downloadTranscript, type ChatTranscript, type TranscriptFormat } from "@/utils/chat-transcript"
import { serverChatActivePath } from "@/utils/newest-leaf-path"

export type ServerChatExportOptions = {
  /** The title the caller already shows, used when the chat's details can't be read. */
  title?: string | null
  scope?: ChatScope
  requestScope?: ServicePromptRequestScope
  signal?: AbortSignal
}

export const readServerChatTranscript = async (
  chatId: string,
  { title, scope, requestScope, signal }: ServerChatExportOptions = {}
): Promise<ChatTranscript> => {
  const options = { scope, requestScope, signal }
  const [chat, listed] = await Promise.all([
    // The title and dates are optional; the messages are not.
    tldwClient.getChat(chatId, options).catch(() => null),
    fetchAllServerChatMessages(
      (page) =>
        tldwClient.listChatMessages(
          chatId,
          { limit: page.limit, offset: page.offset, include_deleted: "false" },
          { ...options, signal: page.signal }
        ),
      { signal }
    )
  ])
  return {
    id: chatId,
    title: chat?.title?.trim() || title?.trim() || "Untitled chat",
    createdAt: chat?.created_at ?? null,
    updatedAt: chat?.updated_at ?? null,
    messages: serverChatActivePath(listed).messages.map((message) => ({
      role: message.role,
      content: message.content,
      timestamp: message.created_at || null
    }))
  }
}

export const exportServerChat = async (
  chatId: string,
  format: TranscriptFormat,
  options?: ServerChatExportOptions
) => {
  downloadTranscript(await readServerChatTranscript(chatId, options), format)
}
