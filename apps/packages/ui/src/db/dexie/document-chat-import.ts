import { db } from "./schema"
import { historyDigest } from "./history-selection"
import { serverChatMirrorOwnerKey } from "./server-chat-mirror"
import type { HistoryInfo, Message } from "./types"
import { resolveParentPath } from "@/utils/history-selection"
import {
  DOCUMENT_CHAT_SOURCES, projectDocumentChatCitations, readDocumentChatCitations, type DocumentChatCitation
} from "@/utils/document-chat-citations"
import type { DocumentChatTarget } from "@/services/tldw/domains/media"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"

export type DocumentChatImportPayload = {
  id: string
  title: string
  created_at: string
  messages: Array<{
    id: string
    parent_message_id: string | null
    role: "user" | "assistant" | "system"
    content: string
    timestamp: string
    images: string[]
    metadata?: Record<string, unknown>
    document_chat_sources_v1?: DocumentChatCitation[]
  }>
}

export type DocumentChatImportReservation = {
  chatId: string
  mediaUuid: string
  payload: DocumentChatImportPayload
  source: Record<string, string>
  documentChatRecord?: string
  attached?: boolean
}

export class DocumentChatReviewRequired extends Error {
  constructor() {
    super("Review the message order before saving this conversation.")
  }
}

function requireHistory(
  history: HistoryInfo | undefined, target: DocumentChatTarget, requestScope: ServicePromptRequestScope
): HistoryInfo {
  if (!history || history.doc_id !== `document:${target.mediaId}` ||
      history.server_scope_key !== serverChatMirrorOwnerKey({ requestScope }) ||
      (history.document_media_uuid && history.document_media_uuid !== target.mediaUuid)) {
    throw new Error("This conversation does not belong to the current document and account.")
  }
  return history
}

function frozenSource(rows: Message[]) {
  return Object.fromEntries(rows.map(row => [row.id, historyDigest(row)]))
}

function convertMessage(
  row: Message, target: DocumentChatTarget, parents: ReadonlyMap<string, string | null>
): DocumentChatImportPayload["messages"][number] {
  if (row.role !== "assistant" && row.role !== "user" && row.role !== "system")
    throw new Error("This chat has a message type that can only be kept on this browser.")
  if (row.documents?.length || row.search || row.discoSkillComment || row.clusterId || row.serverMessageId) {
    throw new Error("This chat has attachments or server history that need to be reviewed before saving.")
  }
  const generation = row.generationInfo ?? {}
  const stored = row.metadataExtra ?? {}
  const generationKeys = ["model_id", "provider", "generation_status", "finish_reason", "usage"]
  const acceptedKeys = [...generationKeys, "sender_role", "workspace_rag_sources_v1", DOCUMENT_CHAT_SOURCES]
  if (Object.keys(stored).some(key => !acceptedKeys.includes(key)) ||
      Object.keys(generation).some(key => !generationKeys.includes(key))) {
    throw new Error("Some message details can only be kept on this browser.")
  }
  const metadata = Object.fromEntries(generationKeys.flatMap(key => {
    const value = stored[key] ?? generation[key] ?? (key === "model_id" ? row.modelId : undefined)
    return value == null ? [] : [[key, value]]
  }))
  const canonical = stored[DOCUMENT_CHAT_SOURCES]
  const citations = canonical !== undefined ? readDocumentChatCitations(canonical, target)
    : row.sources?.length ? projectDocumentChatCitations(row.sources, target) : null
  if ((canonical !== undefined || row.sources?.length || stored.workspace_rag_sources_v1) && !citations)
    throw new Error("Some source references could not be saved. The original chat is still on this browser.")
  return {
    id: row.id, parent_message_id: parents.get(row.id) ?? null, role: row.role,
    content: row.content, timestamp: new Date(row.createdAt).toISOString(),
    images: row.images?.length ? row.images : row.image ? [row.image] : [],
    ...(row.role === "assistant" && Object.keys(metadata).length ? { metadata } : {}),
    ...(citations ? { document_chat_sources_v1: citations } : {})
  }
}

/** Reserve the exact native request once across tabs before any server write. */
export async function reserveDocumentChatImport(
  historyId: string, target: DocumentChatTarget, requestScope: ServicePromptRequestScope,
  confirmedLegacy = false, reviewedOrder?: readonly string[]
): Promise<DocumentChatImportReservation> {
  return db.transaction("rw", db.chatHistories, db.messages, async () => {
    const history = requireHistory(await db.chatHistories.get(historyId), target, requestScope)
    if (!history.document_media_uuid && !confirmedLegacy) throw new Error("Confirm this document before saving the chat.")
    if (history.server_chat_id) throw new Error("This conversation already has a server record.")
    if (history.document_chat_import) {
      if (history.document_chat_import.mediaUuid !== target.mediaUuid)
        throw new Error("This save was reserved for a different document.")
      return history.document_chat_import
    }
    const rows = await db.messages.where("history_id").equals(historyId).toArray()
    if (!rows.length) throw new Error("Send a message before saving this conversation.")
    const parents = new Map(rows.map(row => [row.id, row.parent_message_id ?? null]))
    if (reviewedOrder?.length === rows.length && new Set(reviewedOrder).size === rows.length &&
        reviewedOrder.every(id => parents.has(id))) {
      reviewedOrder.forEach((id, index) => parents.set(id, reviewedOrder[index - 1] ?? null))
    }
    try {
      if ([...parents.values()].filter(parent => parent === null).length !== 1) throw new Error("Multiple roots")
      resolveParentPath(rows.map(row => ({
        id: row.id, parent_id: parents.get(row.id) ?? null, revision: "1", role: row.role, settled: true
      })), { kind: "empty" })
    } catch {
      throw new DocumentChatReviewRequired()
    }
    const chatId = crypto.randomUUID()
    const reservation = {
      chatId, mediaUuid: target.mediaUuid, source: frozenSource(rows),
      payload: {
        id: chatId, title: history.title, created_at: new Date(history.createdAt).toISOString(),
        messages: rows.map(row => convertMessage(row, target, parents))
      }
    }
    await db.chatHistories.update(historyId, {
      document_chat_import: reservation, document_media_uuid: target.mediaUuid
    })
    return reservation
  })
}

export async function verifyDocumentChatImport(
  historyId: string, target: DocumentChatTarget, requestScope: ServicePromptRequestScope,
  reservation: DocumentChatImportReservation
): Promise<boolean> {
  return db.transaction("r", db.chatHistories, db.messages, async () => {
    requireHistory(await db.chatHistories.get(historyId), target, requestScope)
    const rows = await db.messages.where("history_id").equals(historyId).toArray()
    return rows.length === Object.keys(reservation.source).length &&
      rows.every(row => reservation.source[row.id] === historyDigest(row))
  })
}

export async function acknowledgeDocumentChatImport(
  historyId: string, target: DocumentChatTarget, requestScope: ServicePromptRequestScope,
  chatId: string, record: string, attached = false
) {
  return db.transaction("rw", db.chatHistories, async () => {
    const history = requireHistory(await db.chatHistories.get(historyId), target, requestScope)
    const reservation = history.document_chat_import
    if (!reservation || reservation.chatId !== chatId || (reservation.documentChatRecord && reservation.documentChatRecord !== record)) {
      throw new Error("This chat save changed in another browser tab.")
    }
    const acknowledged = { ...reservation, documentChatRecord: record, attached: Boolean(reservation.attached || attached) }
    await db.chatHistories.update(historyId, { document_chat_import: acknowledged })
    return acknowledged
  })
}
