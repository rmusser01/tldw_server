import {
  tldwClient,
  type ExplainerChatbookExportPayload,
  type ExplainerNodeCreatePayload,
  type ExplainerNodeExpandPayload,
  type ExplainerNodePatchPayload,
  type ExplainerQuestionAnswerPayload,
  type ExplainerSessionCreatePayload,
  type ExplainerSessionPatchPayload
} from "@/services/tldw/TldwApiClient"
import type {
  ExplainerExportResponse,
  ExplainerJobStatus,
  ExplainerSession,
  ExplainerSessionListResponse,
  ExplainerSourceCandidate
} from "./types"

const toArray = (value: unknown): Record<string, unknown>[] => {
  if (Array.isArray(value)) return value.filter(isRecord)
  if (!isRecord(value)) return []
  for (const key of ["items", "results", "media", "notes"]) {
    const nested = value[key]
    if (Array.isArray(nested)) return nested.filter(isRecord)
  }
  return []
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const text = (value: unknown, fallback = ""): string =>
  typeof value === "string" && value.trim() ? value : fallback

const optionalText = (value: unknown): string | null =>
  typeof value === "string" && value.trim() ? value : null

const idText = (value: unknown): string | null => {
  if (typeof value === "string" && value.trim()) return value
  if (typeof value === "number" && Number.isFinite(value)) return String(value)
  return null
}

const normalizeMediaCandidate = (item: Record<string, unknown>): ExplainerSourceCandidate | null => {
  const id = idText(item.media_id ?? item.mediaId ?? item.id ?? item.uuid)
  if (!id) return null
  const title = text(item.title ?? item.name ?? item.url, `Media ${id}`)
  return {
    sourceId: id,
    sourceType: "media",
    title,
    description: optionalText(item.media_type ?? item.type ?? item.url),
    metadata: {
      mediaId: item.media_id ?? item.mediaId ?? item.id,
      url: item.url,
      type: item.media_type ?? item.type
    }
  }
}

const normalizeNoteCandidate = (item: Record<string, unknown>): ExplainerSourceCandidate | null => {
  const id = idText(item.id ?? item.note_id ?? item.noteId)
  if (!id) return null
  const title = text(item.title ?? item.name, `Note ${id}`)
  return {
    sourceId: id,
    sourceType: "note",
    title,
    description: optionalText(item.content ?? item.preview ?? item.summary),
    metadata: {
      noteId: item.id ?? item.note_id ?? item.noteId,
      version: item.version
    }
  }
}

export const explainerApi = {
  listSessions: (params?: { limit?: number; offset?: number }) =>
    tldwClient.listExplainerSessions(params) as Promise<ExplainerSessionListResponse>,

  getSession: (sessionId: string) =>
    tldwClient.getExplainerSession(sessionId) as Promise<ExplainerSession>,

  createSession: (payload: ExplainerSessionCreatePayload) =>
    tldwClient.createExplainerSession(payload) as Promise<ExplainerSession>,

  updateSession: (sessionId: string, payload: ExplainerSessionPatchPayload) =>
    tldwClient.updateExplainerSession(sessionId, payload) as Promise<ExplainerSession>,

  deleteSession: (sessionId: string) => tldwClient.deleteExplainerSession(sessionId),

  createNode: (sessionId: string, payload: ExplainerNodeCreatePayload) =>
    tldwClient.createExplainerNode(sessionId, payload),

  updateNode: (sessionId: string, nodeId: string, payload: ExplainerNodePatchPayload) =>
    tldwClient.updateExplainerNode(sessionId, nodeId, payload),

  deleteNode: (sessionId: string, nodeId: string) =>
    tldwClient.deleteExplainerNode(sessionId, nodeId),

  expandNode: (sessionId: string, nodeId: string, payload?: ExplainerNodeExpandPayload) =>
    tldwClient.expandExplainerNode(sessionId, nodeId, payload),

  answerQuestion: (
    sessionId: string,
    nodeId: string,
    payload: ExplainerQuestionAnswerPayload
  ) => tldwClient.answerExplainerQuestion(sessionId, nodeId, payload),

  getJob: (jobId: string) =>
    tldwClient.getExplainerJob(jobId) as Promise<ExplainerJobStatus>,

  exportChatbook: (sessionId: string, payload?: ExplainerChatbookExportPayload) =>
    tldwClient.exportExplainerChatbook(
      sessionId,
      payload
    ) as Promise<ExplainerExportResponse>,

  searchSources: async (query: string): Promise<ExplainerSourceCandidate[]> => {
    const normalized = query.trim()
    if (!normalized) return []
    const [mediaResponse, notesResponse] = await Promise.all([
      tldwClient.searchMedia({ query: normalized }, { page: 1, results_per_page: 8 }),
      tldwClient.searchNotes(normalized)
    ])
    const media = toArray(mediaResponse)
      .map(normalizeMediaCandidate)
      .filter((item): item is ExplainerSourceCandidate => Boolean(item))
    const notes = toArray(notesResponse)
      .map(normalizeNoteCandidate)
      .filter((item): item is ExplainerSourceCandidate => Boolean(item))
    return [...media, ...notes].slice(0, 16)
  }
}
