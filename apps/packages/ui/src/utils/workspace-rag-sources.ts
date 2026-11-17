import {
  validateKnowledgeNoteProvenance,
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
    },
  })) || [];
