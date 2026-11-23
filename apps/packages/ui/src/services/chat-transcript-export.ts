/**
 * Per-chat export read from the server (CS-09).
 *
 * Since D7 most chats exist only on the server, so an export that reads the
 * device's copy fails for them. This reads every saved message page by page,
 * alternative replies included, in creation order: the JSON export keeps them
 * all, and the Markdown export shows the chat's newest path (CM-N2).
 */
import { fetchAllServerChatMessages } from "@/hooks/chat/useServerChatLoader"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import type { ChatScope } from "@/types/chat-scope"
import type { ServerChatMessage } from "@/services/tldw/TldwApiClient"
import {
  downloadTranscript,
  type ChatTranscript,
  type TranscriptFormat,
  type TranscriptMessage
} from "@/utils/chat-transcript"

/**
 * The chat has more messages than one export can read (the loader's
 * 20,000-message page limit), so it can't be exported here.
 */
export class ChatExportTooLargeError extends Error {
  constructor() {
    super("chat_export_too_large")
    this.name = "ChatExportTooLargeError"
  }
}

export type ServerChatExportOptions = {
  /** The title the caller already shows, used when the chat's details can't be read. */
  title?: string | null
  scope?: ChatScope
  requestScope?: ServicePromptRequestScope
  signal?: AbortSignal
}

const text = (value: unknown) => (typeof value === "string" && value.trim() ? value : undefined)

/** One saved message with the fields the server returns, leaving out absent ones. */
const transcriptMessage = (message: ServerChatMessage): TranscriptMessage => {
  const metadata = message.metadata_extra
  const entry: TranscriptMessage = {
    id: String(message.id),
    parentId: message.parent_message_id ?? null,
    role: message.role,
    content: message.content,
    timestamp: message.created_at || null
  }
  if (text(message.sender) && message.sender !== message.role) entry.sender = message.sender
  if (typeof message.version === "number") entry.version = message.version
  if (text(metadata?.model_id)) entry.model = metadata!.model_id as string
  if (text(metadata?.provider)) entry.provider = metadata!.provider as string
  if (metadata && Object.keys(metadata).length) entry.metadata = metadata
  return entry
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
          { limit: page.limit, offset: page.offset, include_deleted: "false", include_metadata: "true" },
          { ...options, signal: page.signal }
        ),
      { signal }
    ).catch((error: unknown) => {
      if (error instanceof Error && /pagination limit/i.test(error.message)) throw new ChatExportTooLargeError()
      throw error
    })
  ])
  // Image data isn't read for an export; JSON says when the chat has some.
  const imagesOmitted = listed.some((message) =>
    message.has_image === true ||
    (message.images?.length ?? 0) > 0 ||
    message.metadata_extra?.content_placeholder_reason === "image_attachment")
  return {
    id: chatId,
    title: chat?.title?.trim() || title?.trim() || "Untitled chat",
    createdAt: chat?.created_at ?? null,
    updatedAt: chat?.updated_at ?? null,
    // fetchAllServerChatMessages returns them in stable creation order.
    messages: listed.map(transcriptMessage),
    ...(imagesOmitted ? { imagesOmitted } : {})
  }
}

export const exportServerChat = async (
  chatId: string,
  format: TranscriptFormat,
  options?: ServerChatExportOptions
) => {
  downloadTranscript(await readServerChatTranscript(chatId, options), format)
}
