import type { WorkspaceSource } from "@/types/workspace"
import { isKnowledgeAnswerTrustState } from "@/components/Option/KnowledgeQA/trustState"

export type KnowledgeNoteProvenance = {
  origin: "knowledge_qa" | "reviewed_sources"
  trust_state?: string
  evidence_origin?: string
  thread_id?: string
  research?: {
    workspace_id: string
    import_id: string
    sources: Array<{
      mediaId: number
      evidence: NonNullable<WorkspaceSource["knowledgeQaEvidence"]>
    }>
  }
}

const MAX_MARKER_LENGTH = 1_000_000
const markerPattern = /^<!-- tldw-knowledge:v1:([^\r\n]+) -->$/gm
const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
const shortString = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0 && value.length <= 512

const positiveId = (value: unknown): value is number =>
  Number.isSafeInteger(value) && Number(value) > 0
const stringList = (value: unknown) =>
  Array.isArray(value) && value.length <= 100 && value.every(shortString)
const validateResearch = (
  value: unknown,
): KnowledgeNoteProvenance["research"] | null => {
  if (
    !isRecord(value) ||
    !shortString(value.workspace_id) ||
    !shortString(value.import_id) ||
    !Array.isArray(value.sources) ||
    value.sources.length > 100
  )
    return null
  const sources: NonNullable<KnowledgeNoteProvenance["research"]>["sources"] =
    []
  for (const entry of value.sources) {
    if (
      !isRecord(entry) ||
      !positiveId(entry.mediaId) ||
      !isRecord(entry.evidence)
    )
      return null
    const evidence = entry.evidence
    if (
      !shortString(evidence.importId) ||
      (evidence.threadId !== null && !shortString(evidence.threadId)) ||
      typeof evidence.snapshot !== "boolean" ||
      !Array.isArray(evidence.sources) ||
      evidence.sources.length > 100
    )
      return null
    if (
      evidence.trustState != null &&
      !isKnowledgeAnswerTrustState(evidence.trustState)
    )
      return null
    if (
      evidence.trustReasonCodes != null &&
      !stringList(evidence.trustReasonCodes)
    )
      return null
    if (
      evidence.evidenceOrigin != null &&
      !["local_library", "web_fallback", "mixed", "unknown_origin"].includes(
        String(evidence.evidenceOrigin),
      )
    )
      return null
    const scope = evidence.scope
    if (
      scope != null &&
      (!isRecord(scope) ||
        Object.entries(scope).some(([key, item]) => {
          if (item === undefined) return false
          if (key === "sources" || key === "include_note_ids")
            return !stringList(item)
          if (key === "include_media_ids")
            return (
              !Array.isArray(item) ||
              item.length > 100 ||
              !item.every(positiveId)
            )
          if (key === "enable_web_fallback") return typeof item !== "boolean"
          if (key === "collection_id")
            return item !== null && !shortString(item) && !positiveId(item)
          if (key === "keyword_filter")
            return (
              item !== null &&
              item !== "" &&
              !shortString(item) &&
              !stringList(item)
            )
          return true
        }))
    )
      return null
    const retained = []
    for (const source of evidence.sources) {
      if (
        !isRecord(source) ||
        !shortString(source.title) ||
        typeof source.excerpt !== "string" ||
        source.excerpt.length > 100_000 ||
        !["pdf", "video", "audio", "website", "text", "document"].includes(
          String(source.type),
        ) ||
        (source.originalId !== null &&
          !shortString(source.originalId) &&
          !positiveId(source.originalId)) ||
        (source.mediaId !== null && !positiveId(source.mediaId)) ||
        (source.sourceType !== null && !shortString(source.sourceType))
      )
        return null
      if (
        (source.url != null &&
          (typeof source.url !== "string" || source.url.length > 4096)) ||
        [source.snapshotMediaId, source.pageNumber, source.citationIndex].some(
          (item) => item != null && !positiveId(item),
        )
      )
        return null
      retained.push({
        originalId: source.originalId,
        excerpt: source.excerpt,
        mediaId: source.mediaId,
        title: source.title,
        type: source.type,
        sourceType: source.sourceType,
        ...(source.url ? { url: source.url } : {}),
        ...(source.snapshotMediaId
          ? { snapshotMediaId: source.snapshotMediaId }
          : {}),
        ...(source.pageNumber ? { pageNumber: source.pageNumber } : {}),
        ...(source.citationIndex
          ? { citationIndex: source.citationIndex }
          : {}),
      })
    }
    sources.push({
      mediaId: entry.mediaId,
      evidence: {
        importId: evidence.importId,
        threadId: evidence.threadId,
        snapshot: evidence.snapshot,
        sources: retained,
        ...(evidence.trustState ? { trustState: evidence.trustState } : {}),
        ...(evidence.trustReasonCodes
          ? { trustReasonCodes: evidence.trustReasonCodes }
          : {}),
        ...(evidence.evidenceOrigin
          ? { evidenceOrigin: evidence.evidenceOrigin }
          : {}),
        ...(scope ? { scope } : {}),
      },
    } as NonNullable<KnowledgeNoteProvenance["research"]>["sources"][number])
  }
  return {
    workspace_id: value.workspace_id,
    import_id: value.import_id,
    sources,
  }
}

export const validateKnowledgeNoteProvenance = (
  value: unknown,
): KnowledgeNoteProvenance | null => {
  if (
    !isRecord(value) ||
    !["knowledge_qa", "reviewed_sources"].includes(String(value.origin))
  )
    return null
  if (
    value.trust_state != null &&
    !isKnowledgeAnswerTrustState(value.trust_state)
  )
    return null
  if (
    value.evidence_origin != null &&
    !["local_library", "web_fallback", "mixed", "unknown_origin"].includes(
      String(value.evidence_origin),
    )
  )
    return null
  if (value.thread_id != null && !shortString(value.thread_id)) return null
  const research =
    value.research == null ? undefined : validateResearch(value.research)
  if (research === null) return null
  return {
    ...(research ? { research } : {}),
    origin: value.origin as KnowledgeNoteProvenance["origin"],
    ...(value.trust_state ? { trust_state: value.trust_state as string } : {}),
    ...(value.evidence_origin
      ? { evidence_origin: value.evidence_origin as string }
      : {}),
    ...(value.thread_id ? { thread_id: value.thread_id as string } : {}),
  }
}

/** Content travels through canonical Notes, rich-text editing and file sync. */
export const readKnowledgeNoteProvenance = (
  content: string,
): KnowledgeNoteProvenance | null => {
  for (const match of content.matchAll(new RegExp(markerPattern))) {
    if (match[1].length > MAX_MARKER_LENGTH) continue
    try {
      const parsed = validateKnowledgeNoteProvenance(
        JSON.parse(decodeURIComponent(match[1])),
      )
      if (parsed) return parsed
    } catch {
      /* Malformed content grants no provenance. */
    }
  }
  return null
}

export const retainKnowledgeNoteProvenance = (
  content: string,
  metadata?: unknown,
): string => {
  const original = isRecord(metadata)
    ? validateKnowledgeNoteProvenance(metadata.knowledge_provenance) ||
      validateKnowledgeNoteProvenance(metadata)
    : null
  const provenance = original || readKnowledgeNoteProvenance(content)
  if (!provenance) return content
  const encoded = encodeURIComponent(JSON.stringify(provenance))
  if (encoded.length > MAX_MARKER_LENGTH)
    throw new Error("Knowledge provenance is too large to save")
  const body = stripKnowledgeNoteProvenance(content).trimEnd()
  return `${body}\n\n<!-- tldw-knowledge:v1:${encoded} -->`
}

/** Hide only validated application provenance; ordinary comments remain user text. */
export const stripKnowledgeNoteProvenance = (content: string): string => {
  let removed = false
  const body = content.replace(new RegExp(markerPattern), (marker) => {
    if (!readKnowledgeNoteProvenance(marker)) return marker
    removed = true
    return ""
  })
  return removed ? body.trimEnd() : content
}
