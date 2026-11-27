import { z } from "zod"
import type { DocumentChatTarget } from "@/services/tldw/domains/media"

export const DOCUMENT_CHAT_SOURCES = "document_chat_sources_v1"

const citationsSchema = z.array(z.object({
  media_uuid: z.string().uuid(),
  title: z.string().trim().min(1).max(512),
  excerpt: z.string().trim().min(1).max(4096),
  page: z.number().int().positive().optional(),
  epub_cfi: z.string().max(1024).regex(/^epubcfi\(.+\)$/).optional()
}).strict()).min(1).max(64)

export type DocumentChatCitation = z.infer<typeof citationsSchema>[number]

export function readDocumentChatCitations(value: unknown, target?: DocumentChatTarget) {
  const parsed = citationsSchema.safeParse(value)
  if (!parsed.success || (target && parsed.data.some(source => source.media_uuid !== target.mediaUuid)) ||
      encodeURIComponent(JSON.stringify(parsed.data)).length > 262_144) return null
  return parsed.data
}

/** Convert the document-only RAG results to stable references without retaining an external URL. */
export function projectDocumentChatCitations(sources: unknown[], target: DocumentChatTarget) {
  const citations = sources.map(value => {
    if (!value || typeof value !== "object") return null
    const source = value as Record<string, unknown>
    const metadata = (source.metadata ?? {}) as Record<string, unknown>
    if ((metadata.media_id != null && Number(metadata.media_id) !== target.mediaId) ||
        (metadata.media_uuid != null && metadata.media_uuid !== target.mediaUuid)) return null
    const page = metadata.page ?? metadata.page_number
    const epubCfi = metadata.epub_cfi ?? metadata.cfi
    return {
      media_uuid: target.mediaUuid,
      title: source.name ?? metadata.title,
      excerpt: source.pageContent ?? source.content ?? source.excerpt,
      ...(page != null ? { page: Number(page) } : {}),
      ...(epubCfi != null ? { epub_cfi: epubCfi } : {})
    }
  })
  return readDocumentChatCitations(citations, target)
}

export function restoreDocumentChatSources(value: unknown) {
  return readDocumentChatCitations(value)?.map(citation => ({
    name: citation.title,
    pageContent: citation.excerpt,
    mode: "rag",
    documentCitation: citation,
    metadata: { page: citation.page, epub_cfi: citation.epub_cfi }
  }))
}
