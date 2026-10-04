import { z } from "zod"
import { HistorySelectionError } from "./history-selection"
import type { HistoryAdmissionReferenceV1 } from "@/types/history-selection"
import type { HistoryDurableResultV1, HistoryDurableSourceV1, HistoryDurableResultReceiptV1 } from "@/types/history-durable-turn"

const fail = (code = "invalid_history_durable_result"): never => { throw new HistorySelectionError(code) }
const freeze = <T>(value: T): T => {
  if (value && typeof value === "object") {
    Object.values(value).forEach(freeze)
    Object.freeze(value)
  }
  return value
}
const record = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) return fail()
  return value as Record<string, unknown>
}
const onlyKeys = (value: Record<string, unknown>, keys: readonly string[]) => {
  if (Object.keys(value).some(key => !keys.includes(key))) fail()
}
const plainObjects = (value: unknown): void => {
  if (Array.isArray(value)) value.forEach(plainObjects)
  else if (value && typeof value === "object") Object.values(record(value)).forEach(plainObjects)
}
const bytes = (value: string) => new TextEncoder().encode(value).length
const scalarString = (value: string) => !/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value)
const text = (limit: number, allowEmpty = false) => z.string().refine(value =>
  scalarString(value) && bytes(value) <= limit && (allowEmpty || value.trim().length > 0))
const safeInt = z.number().int().min(0).max(Number.MAX_SAFE_INTEGER)
const metadataSchema = z.object({
  source: text(1000).optional(), title: text(1000).optional(), chunk_id: text(512).optional(),
  retrieval_strategy: text(128).optional(), source_type: text(128).optional(),
  selection_reason: text(1000).optional(), score: z.number().finite().optional(),
  media_id: text(512).optional(), author: text(1000).optional(),
  chunk_index: safeInt.optional(), total_chunks: safeInt.min(1).optional(),
  start_char: safeInt.optional(), end_char: safeInt.optional(),
  chunk_start: safeInt.optional(), chunk_end: safeInt.optional(),
  page: safeInt.optional(), loc: z.object({
    lines: z.object({ from: safeInt, to: safeInt }).strict().refine(lines => lines.from <= lines.to)
  }).strict().optional()
}).strict().refine(metadata =>
  (metadata.chunk_index === undefined || metadata.total_chunks === undefined || metadata.chunk_index < metadata.total_chunks) &&
  [[metadata.start_char, metadata.end_char], [metadata.chunk_start, metadata.chunk_end]].every(([start, end]) =>
    (start === undefined && end === undefined) || (start !== undefined && end !== undefined && start <= end)))
const sourceSchema = z.object({
  name: text(1000), type: text(128), mode: z.literal("rag"), url: text(2048, true),
  pageContent: text(4000).refine(value => Array.from(value).length <= 1000), metadata: metadataSchema
}).strict()
const resultSchema = z.object({ version: z.literal(1), sources: z.array(sourceSchema).max(20) }).strict()
const hex64 = z.string().regex(/^[0-9a-f]{64}$/)

/** No truncation, normalization, credential fields or evidence-marker deduplication. */
export const parseHistoryDurableResult = (value: unknown, mode: "plain" | "rag"): HistoryDurableResultV1 => {
  if (mode !== "plain" && mode !== "rag") return fail()
  plainObjects(value)
  const parsed = resultSchema.safeParse(value)
  if (!parsed.success) return fail()
  const result = parsed.data
  if ((mode === "plain" && result.sources.length !== 0) || (mode === "rag" && result.sources.length === 0) ||
      result.sources.reduce((sum, source) => sum + Array.from(source.pageContent).length, 0) > 16000 ||
      new Set(result.sources.map(source => source.pageContent)).size !== result.sources.length ||
      bytes(JSON.stringify(result)) > 65536) return fail()
  // Key ordering changes no byte count; hashing stays in the canonical body helper.
  return freeze(JSON.parse(JSON.stringify(result)))
}
const rawSourceKeys = ["name", "type", "mode", "url", "pageContent", "metadata", "content", "text", "snippet",
  "score", "relevance", "chunk_id", "chunkId", "strategy", "retrieval_strategy", "search_mode",
  "source_type", "rationale", "reason", "why_selected"]
const rawMetadataKeys = ["source", "title", "chunk_id", "chunkId", "retrieval_strategy", "reranking_strategy",
  "search_mode", "source_type", "type", "selection_reason", "rationale", "reason", "why_selected",
  "score", "relevance", "rerank_score", "bm25_norm", "page", "loc", "url", "media_type", "chunk_type",
  "created_at", "last_modified", "transcription_model", "retrieval_mode", "paragraph_kind", "highlighted",
  "match_count", "snippets", "ancestry_titles", "embedding_model", "embedding_provider",
  "source_id", "evidence_origin", "section_path",
  "media_id", "note_id", "record_id", "start", "end", "author", "chunk_index", "total_chunks", "start_char", "end_char", "chunk_start", "chunk_end"]
const rawIdentifier = z.union([text(512), safeInt])
const identifier = (value: unknown) => typeof value === "number" ? String(value) : value
const first = (...values: unknown[]) => values.find(value => Boolean(value))
const firstScore = (...values: unknown[]) => values.find(value => typeof value === "number" && Number.isFinite(value))

/** Raw RagSourceEntry only; already projected wire sources use parseHistoryDurableResult. */
export const projectHistoryDurableSources = (value: unknown): readonly HistoryDurableSourceV1[] => {
  if (!Array.isArray(value)) return fail()
  const sources = value.map(item => {
    const source = record(item)
    onlyKeys(source, rawSourceKeys)
    const metadata = source.metadata === undefined ? {} : record(source.metadata)
    onlyKeys(metadata, rawMetadataKeys)
    if (Object.entries(source).some(([key, item]) => !["metadata", "score", "relevance"].includes(key) &&
        item !== undefined && typeof item !== "string") ||
        Object.entries(metadata).some(([key, item]) => !["score", "relevance", "rerank_score", "bm25_norm", "page", "loc", "match_count", "snippets", "ancestry_titles",
          "media_id", "note_id", "record_id", "chunk_id", "chunkId", "start", "end",
          "chunk_index", "total_chunks", "start_char", "end_char", "chunk_start", "chunk_end"].includes(key) &&
        item !== undefined && typeof item !== "string")) fail()
    // Search decoration/bookkeeping is not consumed by citation display or navigation.
    if (["media_id", "note_id", "record_id", "chunk_id", "chunkId"].some(key =>
        metadata[key] !== undefined && !rawIdentifier.safeParse(metadata[key]).success) ||
        ["start", "end"].some(key => metadata[key] !== undefined && !safeInt.safeParse(metadata[key]).success) ||
        (typeof metadata.start === "number" && typeof metadata.end === "number" && metadata.start > metadata.end) ||
        (metadata.media_type !== undefined && !text(128).safeParse(metadata.media_type).success) ||
        (metadata.match_count !== undefined && !safeInt.safeParse(metadata.match_count).success) ||
        (metadata.snippets !== undefined && !z.array(z.string()).safeParse(metadata.snippets).success) ||
        (metadata.source_id !== undefined && !text(512).safeParse(metadata.source_id).success) ||
        (metadata.evidence_origin !== undefined && !text(128).safeParse(metadata.evidence_origin).success) ||
        (metadata.section_path !== undefined && !text(1000).safeParse(metadata.section_path).success) ||
        (metadata.ancestry_titles !== undefined && !z.array(text(1000)).max(20).safeParse(metadata.ancestry_titles).success)) fail()
    if (metadata.url !== undefined && metadata.url !== source.url) fail()
    const projected: Record<string, unknown> = {}
    for (const key of ["source", "title", "page", "loc", "media_id", "author", "chunk_index", "total_chunks", "start_char", "end_char", "chunk_start", "chunk_end"]) {
      if (metadata[key] !== undefined) projected[key] = key === "media_id" ? identifier(metadata[key]) : metadata[key]
    }
    const aliases = {
      score: firstScore(source.score, source.relevance, metadata.score, metadata.relevance, metadata.rerank_score, metadata.bm25_norm),
      chunk_id: first(source.chunk_id, source.chunkId, identifier(metadata.chunk_id), identifier(metadata.chunkId)),
      retrieval_strategy: first(source.strategy, source.retrieval_strategy, source.search_mode,
        metadata.retrieval_strategy, metadata.reranking_strategy, metadata.search_mode),
      source_type: first(source.source_type, source.type, metadata.source_type, metadata.type),
      selection_reason: first(source.rationale, source.reason, source.why_selected,
        metadata.selection_reason, metadata.rationale, metadata.reason, metadata.why_selected)
    }
    for (const [key, alias] of Object.entries(aliases)) if (alias !== undefined) projected[key] = alias
    return { name: source.name, type: source.type, mode: source.mode, url: source.url,
      pageContent: first(source.pageContent, source.content, source.text, source.snippet) ?? "", metadata: projected }
  })
  return parseHistoryDurableResult({ version: 1, sources }, "rag").sources
}

/** Display-only public metadata; never admission or recovery authority. */
export const historyResultV1ToMessageSources = (value: unknown): readonly (HistoryDurableSourceV1 & { readonly source_type?: string })[] => {
  const metadata = record(value)
  onlyKeys(metadata, ["version", "request_context_digest", "sources"])
  if (!hex64.safeParse(metadata.request_context_digest).success) return fail()
  const sources = parseHistoryDurableResult({ version: metadata.version, sources: metadata.sources },
    Array.isArray(metadata.sources) && metadata.sources.length === 0 ? "plain" : "rag").sources
  // MessageSource reads this alias before type; it is presentation, never wire data.
  return freeze(sources.map(source => source.metadata.source_type === undefined ? source :
    { ...source, source_type: source.metadata.source_type }))
}
const referenceSchema = z.object({ version: z.literal(1), owner_key: z.string().min(1),
  conversation_id: z.string().min(1), input_message_id: z.guid(), input_message_revision: z.string().min(1),
  selection_digest: hex64 }).strict()

/** Shape and observation binding only; the caller supplies a previously validated admission. */
export const validateHistoryDurableResultReceipt = (
  owner: Pick<HistoryAdmissionReferenceV1, "owner_key" | "conversation_id">,
  admission: HistoryAdmissionReferenceV1, requestDigest: string, sources: readonly HistoryDurableSourceV1[], value: unknown
): HistoryDurableResultReceiptV1 => {
  const receipt = record(value)
  plainObjects(receipt)
  onlyKeys(receipt, ["version", "result_message_id", "result_message_revision", "admission", "request_context_digest", "sources"])
  const parsed = referenceSchema.safeParse(receipt.admission)
  const expected = referenceSchema.safeParse(admission)
  if (!parsed.success || !expected.success || receipt.version !== 1 || receipt.result_message_revision !== "1" ||
      !z.guid().safeParse(receipt.result_message_id).success || receipt.result_message_id === expected.data.input_message_id ||
      owner.owner_key !== expected.data.owner_key || owner.conversation_id !== expected.data.conversation_id ||
      Object.keys(expected.data).some(key => parsed.data[key as keyof HistoryAdmissionReferenceV1] !== expected.data[key as keyof HistoryAdmissionReferenceV1]) ||
      !hex64.safeParse(requestDigest).success || receipt.request_context_digest !== requestDigest)
    return fail("invalid_history_durable_receipt")
  const mode = sources.length ? "rag" : "plain"
  const result = parseHistoryDurableResult({ version: receipt.version, sources: receipt.sources }, mode)
  const expectedResult = parseHistoryDurableResult({ version: 1, sources }, mode)
  if (JSON.stringify(result.sources) !== JSON.stringify(expectedResult.sources)) return fail("invalid_history_durable_receipt")
  return freeze({ ...result, result_message_id: receipt.result_message_id as string,
    result_message_revision: "1", admission: parsed.data, request_context_digest: requestDigest })
}
