/**
 * Conversation Export Utilities
 * Export a conversation saved on this device to JSON or Markdown, in the shared
 * transcript format (utils/chat-transcript.ts). A chat saved on the server is
 * exported from the server instead (services/chat-transcript-export.ts).
 */

import { db } from "@/db/dexie/schema"
import type { Message } from "@/db/dexie/types"
import {
  downloadTranscript,
  type ChatTranscript,
  type TranscriptMessage
} from "@/utils/chat-transcript"

export type ExportedConversation = ChatTranscript
export type ExportedMessage = TranscriptMessage

/**
 * Get conversation data from IndexedDB
 */
async function getConversationData(
  historyId: string
): Promise<ExportedConversation | null> {
  try {
    const history = await db.chatHistories.get(historyId)
    if (!history) {
      console.warn(`Conversation not found: ${historyId}`)
      return null
    }

    const messages = await db.messages
      .where("history_id")
      .equals(historyId)
      .toArray()

    // Every saved message, alternatives included, in stable creation order;
    // ids and parents let the Markdown export follow the newest path.
    const exportedMessages: ExportedMessage[] = messages
      .map((msg: Message, index) => ({ msg, index }))
      .sort((left, right) =>
        (left.msg.createdAt ?? 0) - (right.msg.createdAt ?? 0) || left.index - right.index)
      .map(({ msg }) => {
        const role = msg.role as "user" | "assistant" | "system"
        const model = role === "assistant" ? msg.modelId || msg.modelName || msg.name : undefined
        return {
          id: msg.id,
          parentId: msg.parent_message_id ?? null,
          role,
          content: msg.content,
          timestamp: msg.createdAt ? new Date(msg.createdAt).toISOString() : undefined,
          ...(model ? { model } : {})
        }
      })

    return {
      id: historyId,
      title: history.title || "Untitled Conversation",
      createdAt: history.createdAt
        ? new Date(history.createdAt).toISOString()
        : new Date().toISOString(),
      model: history.model_id || undefined,
      messages: exportedMessages
    }
  } catch (error) {
    console.error("Error getting conversation data:", error)
    return null
  }
}

/**
 * Export conversation to JSON format
 */
export async function exportToJSON(historyId: string): Promise<boolean> {
  const data = await getConversationData(historyId)
  if (!data) {
    return false
  }
  downloadTranscript(data, "json")
  return true
}

/**
 * Export conversation to Markdown format
 */
export async function exportToMarkdown(historyId: string): Promise<boolean> {
  const data = await getConversationData(historyId)
  if (!data) {
    return false
  }
  downloadTranscript(data, "markdown")
  return true
}

/**
 * Export conversation by tab ID (convenience wrapper)
 * Looks up historyId from the tab and exports
 */
export async function exportTabToJSON(
  historyId: string | null
): Promise<boolean> {
  if (!historyId) {
    console.warn("No historyId provided for export")
    return false
  }
  return exportToJSON(historyId)
}

export async function exportTabToMarkdown(
  historyId: string | null
): Promise<boolean> {
  if (!historyId) {
    console.warn("No historyId provided for export")
    return false
  }
  return exportToMarkdown(historyId)
}
