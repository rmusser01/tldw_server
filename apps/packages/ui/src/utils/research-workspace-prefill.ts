import type { KnowledgeNoteSource } from "./knowledge-note-provenance"
import { createSafeStorage } from "@/utils/safe-storage"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { deriveScopedUserId } from "@/utils/media-navigation-scope"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import type { WebClipperSaveRequest } from "@/services/web-clipper/types"
import type { WebArticleCapturePin, WorkspaceSource } from "@/types/workspace"
import type { WorkspaceSourceType } from "@/types/workspace"

const PREFILL_KEY = "__tldw_research_workspace_prefill"
const storage = createSafeStorage({ area: "local" })
let pendingPrefillWrite: Promise<void> = Promise.resolve()
// Keep readiness checkpoints ordered across receiver unmount/remount and new handoffs.
const persistPrefill = (write: () => Promise<void>): Promise<void> => {
  const next = pendingPrefillWrite.catch(() => {}).then(write)
  pendingPrefillWrite = next
  return next
}

type KnowledgeQaResultLike = {
  id?: string
  sourceId?: string | number
  content?: string
  text?: string
  metadata?: {
    title?: unknown
    source?: unknown
    source_type?: unknown
    url?: unknown
    page_number?: unknown
    media_id?: unknown
    mediaId?: unknown
    document_id?: unknown
    doc_id?: unknown
    [key: string]: unknown
  }
}

export type WorkspaceKnowledgeQaPrefillSource = KnowledgeNoteSource & { importError?: string }

export type ResearchWorkspacePrefill = {
  kind: "knowledge_qa_thread"
  id: string
  ownerScope?: string
  workspaceId?: string
  pendingNoteWrite?: {
    idempotencyKey: string
    method: "POST" | "PUT"
    expectedVersion?: number
    body: Record<string, unknown>
  }
  canonicalNoteId?: string
  legacyNoteId?: number
  draftRetained?: boolean
  completed?: boolean
  selectionIntent?: {
    mediaIds: number[]
    selectedSourceIds: string[]
  } | null
  answerTrustState?: string | null
  answerEvidenceOrigin?: string | null
  answerTrustReasonCodes?: string[]
  scope?: KnowledgeQaScope
  createdAt: string
  threadId: string | null
  query: string
  answer: string | null
  citations: number[]
  sources: WorkspaceKnowledgeQaPrefillSource[]
}

export type KnowledgeQaScope = {
  sources?: string[]
  include_media_ids?: number[]
  include_note_ids?: string[]
  collection_id?: string | number | null
  keyword_filter?: string | string[] | null
  enable_web_fallback?: boolean
}

export type BuildKnowledgeQaWorkspacePrefillInput = {
  answerTrustState?: string | null
  answerEvidenceOrigin?: string | null
  answerTrustReasonCodes?: string[]
  scope?: KnowledgeQaScope
  threadId: string | null
  query: string
  answer: string | null
  citations: number[]
  results: KnowledgeQaResultLike[]
}

const normalizeString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const normalized = value.trim()
  return normalized.length > 0 ? normalized : null
}

const parseNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isSafeInteger(value) && value > 0) {
    return value
  }
  if (typeof value === "string" && /^\d+$/.test(value.trim())) {
    const parsed = Number(value)
    return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null
  }
  return null
}

const toWorkspaceSourceType = (
  sourceTypeRaw: string | null,
  url: string | null,
): WorkspaceSourceType => {
  const sourceType = (sourceTypeRaw || "").toLowerCase()
  if (sourceType.includes("pdf")) return "pdf"
  if (sourceType.includes("video")) return "video"
  if (sourceType.includes("audio")) return "audio"
  if (
    sourceType.includes("website") ||
    sourceType.includes("web") ||
    sourceType.includes("url")
  ) {
    return "website"
  }
  if (sourceType.includes("text")) return "text"
  if (!sourceType && url) return "website"
  return "document"
}

const resolveMediaId = (result: KnowledgeQaResultLike): number | null => {
  const metadata = result.metadata || {}
  if (/note/i.test(String(metadata.source_type || ""))) return null
  // Stored web articles carry explicit media identity; external result IDs do not.
  for (const candidate of [metadata.media_id, metadata.mediaId]) {
    const parsed = parseNumber(candidate)
    if (parsed != null) return parsed
  }
  if (/web|url/i.test(String(metadata.source_type || ""))) return null
  const candidates = [
    metadata.document_id,
    metadata.doc_id,
    ...(metadata.source_type === "media_db"
      ? [metadata.source_id, result.sourceId]
      : []),
    result.id,
  ]
  for (const candidate of candidates) {
    const parsed = parseNumber(candidate)
    if (parsed != null) return parsed
  }
  return null
}

export const toPrefillSource = (
  result: KnowledgeQaResultLike,
  index: number,
  citedIndices: Set<number>,
): WorkspaceKnowledgeQaPrefillSource => {
  const metadata = result.metadata || {}
  const sourceType = normalizeString(metadata.source_type)
  let url = normalizeString(metadata.url)
  const pageNumber = parseNumber(metadata.page_number)
  const fallbackTitle =
    normalizeString(metadata.source) || `Source ${index + 1}`
  const title = normalizeString(metadata.title) || fallbackTitle
  const citationIndex = citedIndices.has(index + 1) ? index + 1 : undefined

  const originalReference =
    metadata.note_id ??
    metadata.media_id ??
    metadata.mediaId ??
    metadata.document_id ??
    metadata.doc_id ??
    metadata.source_id ??
    result.sourceId ??
    result.id ??
    url
  const originalId =
    typeof originalReference === "string" ||
    (typeof originalReference === "number" &&
      Number.isFinite(originalReference))
      ? originalReference
      : null

  if (!url && /note/i.test(sourceType || "") && originalId != null) {
    url = `/notes?source_ref_id=${encodeURIComponent(String(originalId))}`
  }

  return {
    originalId,
    excerpt: result.content || result.text || "",
    mediaId: resolveMediaId(result),
    title,
    type: toWorkspaceSourceType(sourceType, url),
    sourceType,
    ...(url ? { url } : {}),
    ...(pageNumber != null ? { pageNumber } : {}),
    ...(citationIndex != null ? { citationIndex } : {}),
  }
}

export const buildKnowledgeQaWorkspacePrefill = (
  input: BuildKnowledgeQaWorkspacePrefillInput,
): ResearchWorkspacePrefill => {
  const citedIndices = new Set(input.citations)
  const sources = input.results.map((result, index) =>
    toPrefillSource(result, index, citedIndices),
  )

  return {
    kind: "knowledge_qa_thread",
    id: crypto.randomUUID(),
    answerTrustState: input.answerTrustState,
    answerEvidenceOrigin: input.answerEvidenceOrigin,
    answerTrustReasonCodes: input.answerTrustReasonCodes,
    scope: input.scope
      ? {
          sources: input.scope.sources?.slice(),
          include_media_ids: input.scope.include_media_ids?.slice(),
          include_note_ids: input.scope.include_note_ids?.slice(),
          collection_id: input.scope.collection_id,
          keyword_filter: Array.isArray(input.scope.keyword_filter)
            ? input.scope.keyword_filter.slice()
            : input.scope.keyword_filter,
          enable_web_fallback: input.scope.enable_web_fallback,
        }
      : undefined,
    createdAt: new Date().toISOString(),
    threadId: input.threadId,
    query: input.query.trim(),
    answer: input.answer,
    citations: [...new Set(input.citations)].filter(
      (index) => Number.isFinite(index) && index > 0,
    ),
    sources,
  }
}

/** Public account/server identity only; credentials never enter storage keys. */
export const getResearchWorkspaceOwner = async (): Promise<string> => {
  const config = await tldwClient.getConfig()
  if (!config?.serverUrl || deriveScopedUserId(config) === "user:anonymous") {
    throw new Error("Sign in before continuing in Research Workspace.")
  }
  return buildChatSurfaceScopeKeyFromConfig({ ...config, apiKey: undefined })
}

const assertPersistentStorage = () => {
  if (
    (storage as typeof storage & { hasPersistentBackend?: boolean })
      .hasPersistentBackend === false
  ) {
    throw new Error(
      "Research Workspace requires persistent storage. Enable browser storage and retry.",
    )
  }
}

const prefillKey = (owner: string) => `${PREFILL_KEY}:${owner}`

export const queueResearchWorkspacePrefill = async (
  payload: ResearchWorkspacePrefill,
  expectedOwner?: string,
): Promise<void> => {
  assertPersistentStorage()
  let invalidated = false
  const stop = watchChatAccountChanges((changed) => {
    invalidated ||= changed
  })
  try {
    const owner = await getResearchWorkspaceOwner()
    if (invalidated || (expectedOwner && owner !== expectedOwner))
      throw new Error("Account changed. Try again.")
    await persistPrefill(async () => {
      if (invalidated) throw new Error("Account changed. Try again.")
      await storage.set(prefillKey(owner), { ...payload, ownerScope: owner })
    })
    if (invalidated) throw new Error("Account changed. Try again.")
  } finally {
    stop()
  }
}

/** Read without deleting: progress remains available after failures or navigation. */
export const consumeResearchWorkspacePrefill = async (
  owner?: string,
  includeSelectionIntent = false,
): Promise<ResearchWorkspacePrefill | null> => {
  // Writes report failures to their callers; a later read may still recover evidence.
  await pendingPrefillWrite.catch(() => {})
  const expectedOwner = owner ?? (await getResearchWorkspaceOwner())
  const payload = await storage.get<ResearchWorkspacePrefill | null>(
    prefillKey(expectedOwner),
  )
  return payload?.ownerScope === expectedOwner &&
    (!payload.completed || (includeSelectionIntent && payload.selectionIntent))
    ? payload
    : null
}

export const saveResearchWorkspacePrefill = async (
  payload: ResearchWorkspacePrefill,
): Promise<void> => {
  assertPersistentStorage()
  if (!payload.ownerScope) throw new Error("Missing import owner")
  const owner = payload.ownerScope
  const checkpoint = structuredClone(payload)
  await persistPrefill(async () => {
    const current = await storage.get<ResearchWorkspacePrefill | null>(
      prefillKey(owner),
    )
    if (current && current.id !== checkpoint.id)
      throw new Error("A newer Knowledge handoff is waiting.")
    await storage.set(prefillKey(owner), checkpoint)
  })
}

/** Capture checkpoints share the owner-bound recovery backend, never the replaceable handoff. */
export type ResearchWebCapture = {
  ownerScope: string
  workspaceId: string
  sourceId: string
  body: WebClipperSaveRequest
  pin?: WebArticleCapturePin
  attached?: boolean
}

export const readResearchWebCaptures = async (
  owner: string,
  workspaceId: string
): Promise<ResearchWebCapture[]> => {
  await pendingPrefillWrite.catch(() => {})
  const records = await storage.get<Record<string, ResearchWebCapture>>(
    `${prefillKey(owner)}:web-captures`
  )
  return Object.values(records || {})
    .filter(
      (record) =>
        record.ownerScope === owner && record.workspaceId === workspaceId
    )
    .map((record) => {
      // These are the closed, credential-free acceptance objects produced by prepareWebCaptureAcceptance.
      Object.freeze(record.body.content)
      Object.freeze(record.body.workspace)
      Object.freeze(record.body.enhancements)
      Object.freeze(record.body.capture_metadata?.web_capture_v1)
      Object.freeze(record.body.capture_metadata)
      Object.freeze(record.body)
      return record
    })
}

export const saveResearchWebCapture = async (
  record: ResearchWebCapture
): Promise<void> => {
  assertPersistentStorage()
  if (
    !record.ownerScope ||
    record.body.workspace?.workspace_id !== record.workspaceId
  )
    throw new Error("Missing capture owner or destination")
  const checkpoint = structuredClone(record)
  await persistPrefill(async () => {
    const key = `${prefillKey(checkpoint.ownerScope)}:web-captures`
    const records =
      (await storage.get<Record<string, ResearchWebCapture>>(key)) || {}
    const previous = records[checkpoint.body.clip_id]
    if (
      previous &&
      (previous.workspaceId !== checkpoint.workspaceId ||
        JSON.stringify(previous.body) !== JSON.stringify(checkpoint.body))
    )
      throw new Error("Accepted capture cannot change; retry the original body")
    await storage.set(key, {
      ...records,
      [checkpoint.body.clip_id]: checkpoint
    })
  })
}

/** Only decorate authoritative owned membership; a checkpoint never restores a removed source. */
export const retainResearchWebCapturePins = (
  sources: WorkspaceSource[],
  records: ResearchWebCapture[]
): WorkspaceSource[] =>
  sources.map((source) => {
    const record = records.find(
      (item) =>
        item.pin &&
        source.id === `web-clipper:${item.pin.clipId}` &&
        source.mediaId === item.pin.mediaId &&
        source.url === item.pin.requestedUrl
    )
    return record?.pin ? { ...source, webCapture: record.pin } : source
  })

const truncate = (value: string, max: number): string =>
  value.length > max ? `${value.slice(0, max - 1)}...` : value

export const buildKnowledgeQaSeedNote = (
  payload: Extract<ResearchWorkspacePrefill, { kind: "knowledge_qa_thread" }>,
): string => {
  const question = payload.query.trim()
  const answer = (payload.answer || "").trim()

  const lines: string[] = []
  lines.push(
    payload.threadId || question || answer
      ? "Imported from Knowledge QA"
      : "Imported reviewed sources",
  )
  lines.push(`Import reference: ${payload.id}`)
  if (payload.threadId) lines.push(`Session: ${payload.threadId}`)
  if (payload.answerTrustState)
    lines.push(`Answer status: ${payload.answerTrustState}`)
  if (payload.answerEvidenceOrigin)
    lines.push(`Evidence origin: ${payload.answerEvidenceOrigin}`)
  if (payload.answerTrustReasonCodes?.length)
    lines.push(`Qualifications: ${payload.answerTrustReasonCodes.join(", ")}`)
  if (payload.scope)
    lines.push(`Selected scope: ${JSON.stringify(payload.scope)}`)

  if (question) {
    lines.push(`Question: ${question}`)
  }

  if (answer) {
    lines.push("")
    lines.push("Answer:")
    lines.push(answer)
  }

  if (payload.sources.length > 0) {
    lines.push("")
    lines.push("Sources:")
    for (const source of payload.sources) {
      const citationPrefix =
        source.citationIndex != null ? `[${source.citationIndex}] ` : ""
      const pageSuffix =
        source.pageNumber != null ? ` (p. ${source.pageNumber})` : ""
      const urlSuffix = source.url ? ` - ${source.url}` : ""
      lines.push(
        `- ${citationPrefix}${truncate(source.title, 140)}${pageSuffix}${urlSuffix}`,
      )
      lines.push(
        `  Original reference: ${source.sourceType || "media"} / ${source.originalId ?? source.mediaId ?? "unknown"}`,
      )
      if (source.mediaId == null)
        lines.push(
          source.originalVersion != null
            ? `  Full note snapshot (version ${source.originalVersion}); original retrieved evidence below. Refresh by importing the note again.`
            : "  Retrieved-excerpt snapshot; not a live or complete copy of the original.",
        )
      if (source.excerpt) lines.push(source.excerpt)
    }
  }

  return lines.join("\n")
}
