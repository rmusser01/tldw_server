type SourceIds = {
  documentIds: string[]
  chunkIds: string[]
  corpus?: string
}

const normalizeId = (value: unknown): string | null => {
  if (value == null) return null
  const asString = String(value).trim()
  return asString.length > 0 ? asString : null
}

export const extractSourceFeedbackIds = (source: any): SourceIds => {
  const metadata = source?.metadata || {}
  const docId =
    normalizeId(metadata.document_id) ||
    normalizeId(metadata.doc_id) ||
    normalizeId(metadata.documentId) ||
    normalizeId(metadata.docId) ||
    normalizeId(source?.document_id) ||
    normalizeId(source?.doc_id) ||
    normalizeId(source?.documentId) ||
    normalizeId(source?.docId)
  const chunkId =
    normalizeId(metadata.chunk_id) ||
    normalizeId(metadata.chunkId) ||
    normalizeId(source?.chunk_id) ||
    normalizeId(source?.chunkId)
  const corpus =
    normalizeId(metadata.corpus) ||
    normalizeId(metadata.namespace) ||
    normalizeId(metadata.index) ||
    normalizeId(source?.corpus) ||
    normalizeId(source?.namespace) ||
    undefined

  return {
    documentIds: docId ? [docId] : [],
    chunkIds: chunkId ? [chunkId] : [],
    corpus
  }
}

export const getSourceImpressionId = (
  source: any,
  index?: number
): string => {
  const { documentIds, chunkIds } = extractSourceFeedbackIds(source)
  if (chunkIds.length > 0) return chunkIds[0]
  if (documentIds.length > 0) return documentIds[0]
  const url = normalizeId(source?.url)
  if (url) return `url:${url}`
  const name = normalizeId(source?.name)
  if (name) return `name:${name}`
  return typeof index === "number" ? `index:${index}` : "unknown"
}

export const getSourceFeedbackKey = (source: any, index?: number): string => {
  const { documentIds, chunkIds } = extractSourceFeedbackIds(source)
  if (chunkIds.length > 0) return `chunk:${chunkIds[0]}`
  const identity = documentIds.length > 0 ? `doc:${documentIds[0]}` : getSourceImpressionId(source, index)
  // Older saved citations have no chunk ID; keep separate passages independently reviewable.
  const citation = source?.metadata?.citation_index ?? index
  return citation != null ? `${identity}:citation:${citation}` : identity
}

export const collectImpressionList = (sources: any[] = []): string[] => {
  return sources.map((source, index) => getSourceImpressionId(source, index))
}
