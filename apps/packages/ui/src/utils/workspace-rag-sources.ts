import {
  validateKnowledgeNoteProvenance,
  projectPdfPageSpans,
  type KnowledgeNoteProvenance,
  type KnowledgeNoteSource,
} from "./knowledge-note-provenance";

export const WORKSPACE_RAG_SOURCES_KEY = "workspace_rag_sources_v1";
const MAX_SOURCES = 64;
const MAX_ENCODED_LENGTH = 262_144;
const sourceTypes = new Set([
  "pdf",
  "video",
  "audio",
  "website",
  "text",
  "document",
]);
const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

const positiveId = (value: unknown): number | null => {
  if (
    typeof value !== "number" &&
    !(typeof value === "string" && /^[1-9][0-9]*$/.test(value))
  )
    return null;
  const id = Number(value);
  return Number.isSafeInteger(id) && id > 0 ? id : null;
};

const safeUrl = (value: unknown): boolean => {
  if (value == null || value === "") return true;
  if (typeof value !== "string" || value.length > 4096) return false;
  try {
    return ["http:", "https:"].includes(new URL(value).protocol);
  } catch {
    return false;
  }
};

/** Recorded references never authorize access to the underlying Media. */
export const readWorkspaceRagSourceProvenance = (
  value: unknown,
): KnowledgeNoteProvenance | null => {
  if (
    !record(value) ||
    Object.keys(value).some((key) => !["origin", "sources"].includes(key)) ||
    value.origin !== "knowledge_qa" ||
    !Array.isArray(value.sources) ||
    !value.sources.length ||
    value.sources.length > MAX_SOURCES
  )
    return null;
  const parsed = validateKnowledgeNoteProvenance(value);
  if (
    !parsed?.sources ||
    parsed.sources.some(
      (source) => positiveId(source.mediaId) === null || !safeUrl(source.url),
    )
  )
    return null;
  return encodeURIComponent(JSON.stringify(parsed)).length <= MAX_ENCODED_LENGTH
    ? parsed
    : null;
};

/** Project only the displayed excerpt and stable reference fields, never arbitrary metadata. */
export const projectWorkspaceRagSources = (
  value: unknown,
): KnowledgeNoteProvenance | null => {
  if (!Array.isArray(value) || !value.length || value.length > MAX_SOURCES)
    return null;
  const sources: KnowledgeNoteSource[] = [];
  for (const row of value) {
    if (!record(row)) return null;
    const metadata = record(row.metadata) ? row.metadata : {};
    const mediaId = positiveId(
      row.mediaId ?? row.media_id ?? metadata.mediaId ?? metadata.media_id,
    );
    const excerpt = row.pageContent ?? row.content ?? row.text ?? row.chunk;
    const title = row.name ?? metadata.title ?? metadata.source;
    const sourceType = row.type ?? metadata.type ?? metadata.media_type;
    if (
      mediaId === null ||
      typeof excerpt !== "string" ||
      !excerpt.trim() ||
      excerpt.length > 100_000 ||
      typeof title !== "string" ||
      !title.trim() ||
      title.length > 4096 ||
      (sourceType != null &&
        (typeof sourceType !== "string" || sourceType.length > 4096))
    )
      return null;
    const url = row.url ?? metadata.url;
    const pageNumber = positiveId(row.pageNumber ?? row.page_number ?? metadata.page_number ?? metadata.pageNumber);
    const citationIndex = positiveId(row.citationIndex ?? row.citation_index ?? metadata.citation_index ?? metadata.citationIndex);
    const chunkId = row.chunkId ?? row.chunk_id ?? metadata.chunk_id ?? metadata.chunkId;
    const pageSpan = Array.isArray(metadata.page_spans)
      ? metadata.page_spans.find(span => record(span) && span.page_number === pageNumber)
      : null;
    const printedPageLabel = metadata.printed_page_label ?? pageSpan?.printed_page_label;
    const pageSpans = projectPdfPageSpans(metadata.page_spans);
    sources.push({
      originalId: mediaId,
      excerpt,
      mediaId,
      title,
      type: (typeof sourceType === "string" && sourceTypes.has(sourceType)
        ? sourceType
        : "text") as KnowledgeNoteSource["type"],
      sourceType: typeof sourceType === "string" ? sourceType : null,
      ...(typeof url === "string" && url && safeUrl(url) ? { url } : {}),
      ...(pageNumber !== null ? {pageNumber} : {}),
      ...(citationIndex !== null ? {citationIndex} : {}),
      ...(typeof chunkId === "string" && chunkId.length > 0 && chunkId.length <= 512 ? {chunkId} : {}),
      ...(pageNumber !== null && typeof printedPageLabel === "string" && printedPageLabel.length > 0 && printedPageLabel.length <= 512 ? {printedPageLabel} : {}),
      ...(pageSpans ? { pageSpans } : {}),
    });
  }
  return readWorkspaceRagSourceProvenance({ origin: "knowledge_qa", sources });
};

export const restoreWorkspaceRagSources = (value: unknown) =>
  readWorkspaceRagSourceProvenance(value)?.sources?.map((source) => ({
    name: source.title,
    type: source.sourceType || source.type,
    mode: "rag" as const,
    url: source.url || "",
    pageContent: source.excerpt,
    metadata: {
      media_id: source.mediaId,
      title: source.title,
      type: source.sourceType || source.type,
      ...(source.pageNumber != null ? {page_number: source.pageNumber} : {}),
      ...(source.citationIndex != null ? {citation_index: source.citationIndex} : {}),
      ...(source.chunkId != null ? {chunk_id: source.chunkId} : {}),
      ...(source.printedPageLabel != null ? {printed_page_label: source.printedPageLabel} : {}),
      ...(source.pageSpans ? { page_spans: source.pageSpans.map(span => ({
        page_number: span.pageNumber,
        start_char: span.startChar,
        end_char: span.endChar,
        ...(span.printedPageLabel != null ? { printed_page_label: span.printedPageLabel } : {}),
      })) } : {}),
    },
  })) || [];
