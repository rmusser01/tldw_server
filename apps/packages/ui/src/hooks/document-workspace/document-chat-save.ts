import {
  acknowledgeDocumentChatImport, reserveDocumentChatImport, verifyDocumentChatImport
} from "@/db/dexie/document-chat-import"
import type { HistoryInfo, Message } from "@/db/dexie/types"
import { DOCUMENT_CHAT_SOURCES, projectDocumentChatCitations, readDocumentChatCitations } from "@/utils/document-chat-citations"
import { tldwClient } from "@/services/tldw"
import type { ServerChatMessage } from "@/services/tldw/TldwApiClient"
import type { DocumentChatRequestOptions, DocumentChatTarget } from "@/services/tldw/domains/media"

export async function saveBrowserDocumentChat(
  historyId: string, target: DocumentChatTarget, options: DocumentChatRequestOptions,
  confirmedLegacy: boolean, reviewedOrder?: readonly string[]
) {
  const { requestScope, signal } = options
  const reservation = await reserveDocumentChatImport(historyId, target, requestScope, confirmedLegacy, reviewedOrder)
  let record = reservation.documentChatRecord
  if (!record) {
    signal?.throwIfAborted()
    const imported = await tldwClient.importDocumentChat(target, reservation.payload, options)
    if (imported.id !== reservation.chatId || !imported.document_chat_record)
      throw new Error("The server did not confirm this save. Retry to check the same conversation.")
    record = imported.document_chat_record
    await acknowledgeDocumentChatImport(historyId, target, requestScope, imported.id, record)
  }
  signal?.throwIfAborted()
  await tldwClient.attachDocumentChat(target, reservation.chatId, record, options)
  await acknowledgeDocumentChatImport(historyId, target, requestScope, reservation.chatId, record, true)
  const fullySaved = await verifyDocumentChatImport(historyId, target, requestScope, reservation)
  return { id: reservation.chatId, fullySaved }
}

/** Only the originating account's matching persisted message can identify an older mirror. */
export async function linkExistingDocumentChat(
  history: HistoryInfo, rows: Message[], target: DocumentChatTarget,
  options: DocumentChatRequestOptions, confirmedLegacy = false
) {
  if (!history.server_chat_id || (history.document_media_uuid
    ? history.document_media_uuid !== target.mediaUuid
    : !confirmedLegacy)) return null
  const chatId = history.server_chat_id
  const chat = await tldwClient.getChat(chatId, { ...options, documentTarget: target })
  if (chat.id !== chatId || !chat.document_chat_record || chat.scope_type === "workspace") return null
  const remote = new Map<string, ServerChatMessage>()
  const pageSize = 200
  for (let offset = 0; ; offset += pageSize) {
    const page = await tldwClient.listChatMessages(chatId, {
      limit: pageSize, offset, include_metadata: true, include_images: true, render_placeholders: false
    }, { ...options, fresh: true })
    page.forEach(row => remote.set(row.id, row))
    if (page.length < pageSize) break
  }
  const ids = new Map(rows.map(row => [row.id, row.serverMessageId || row.id]))
  const matching = rows.filter(row => {
    const match = remote.get(ids.get(row.id)!)
    return match?.content === row.content && match.role === row.role
  })
  if (!matching.length) return null
  const fullySaved = matching.length === rows.length && matching.every(row => {
    const match = remote.get(ids.get(row.id)!)!
    const parent = row.parent_message_id
    const sources = row.metadataExtra?.[DOCUMENT_CHAT_SOURCES]
    const citations = sources ? readDocumentChatCitations(sources, target)
      : row.sources?.length ? projectDocumentChatCitations(row.sources, target) : null
    return (!row.sources?.length && !sources || Boolean(citations)) &&
      (match.parent_message_id ?? null) === (parent ? ids.get(parent) ?? parent : null) &&
      JSON.stringify(row.images?.length ? row.images : row.image ? [row.image] : []) === JSON.stringify(match.images ?? []) &&
      JSON.stringify(citations) === JSON.stringify(readDocumentChatCitations(match.metadata_extra?.[DOCUMENT_CHAT_SOURCES], target))
  })
  await tldwClient.attachDocumentChat(target, chatId, chat.document_chat_record, options)
  return { id: chatId, fullySaved }
}
