import type { WorkspaceSource } from "@/types/workspace";
import type { KnowledgeQaScope } from "./research-workspace-prefill";
import { isKnowledgeAnswerTrustState } from "@/components/Option/KnowledgeQA/trustState";

export type KnowledgeNoteSource = {
  originalId: string | number | null;
  excerpt: string;
  mediaId: number | null;
  title: string;
  type: "pdf" | "video" | "audio" | "website" | "text" | "document";
  sourceType: string | null;
  url?: string | null;
  snapshotMediaId?: number | null;
  originalVersion?: number | null;
  pageNumber?: number | null;
  citationIndex?: number | null;
};
export type KnowledgeNoteEvidence = {
  importId: string;
  threadId: string | null;
  snapshot: boolean;
  sources: KnowledgeNoteSource[];
  trustState?: string | null;
  trustReasonCodes?: string[] | null;
  evidenceOrigin?: string | null;
  scope?: KnowledgeQaScope | null;
};

export type KnowledgeNoteProvenance = {
  origin: "knowledge_qa" | "reviewed_sources";
  trust_state?: string | null;
  evidence_origin?: string | null;
  thread_id?: string | null;
  question?: string | null;
  scope?: KnowledgeQaScope | null;
  trust_reason_codes?: string[] | null;
  sources?: KnowledgeNoteSource[] | null;
  research?: {
    workspace_id: string;
    import_id: string;
    sources: Array<{
      mediaId: number;
      evidence: KnowledgeNoteEvidence;
    }>;
  } | null;
};

/** Independent server fields remain optional for older servers and local drafts. */
export type KnowledgeNoteHead = {
  knowledge_provenance_state?: "unsupported" | "absent" | "active" | "deleted";
  knowledge_provenance_version?: number | null;
  knowledge_provenance_hash?: string | null;
  knowledge_provenance?: KnowledgeNoteProvenance | null;
  knowledge_provenance_reconciliation?: "canonical_wins" | null;
};

const MAX_MARKER_LENGTH = 1_000_000;
const markerPattern = /^<!-- tldw-knowledge:v1:([^\r\n]+) -->$/gm;
const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const keys = (
  value: unknown,
  allowed: string[],
): value is Record<string, unknown> =>
  isRecord(value) && Object.keys(value).every((key) => allowed.includes(key));
const shortString = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0 && value.length <= 512;
const positiveId = (value: unknown): value is number =>
  Number.isSafeInteger(value) && Number(value) > 0;
const list = (value: unknown, valid: (item: unknown) => boolean): boolean =>
  Array.isArray(value) && value.length <= 100 && value.every(valid);
const stringList = (value: unknown) => list(value, shortString);
const optional = (value: unknown, valid: (item: unknown) => boolean) =>
  value == null || valid(value);
const origin = (value: unknown) =>
  typeof value === "string" && ["local_library", "web_fallback", "mixed", "unknown_origin"].includes(
    value,
  );
const excerpt = (value: unknown) =>
  typeof value === "string" && value.length <= 100_000;
const scope = (value: unknown): boolean =>
  keys(value, [
    "sources",
    "include_note_ids",
    "include_media_ids",
    "enable_web_fallback",
    "collection_id",
    "keyword_filter",
  ]) &&
  Object.entries(value).every(([key, item]) => {
    if (item === undefined) return true;
    if (key === "sources" || key === "include_note_ids")
      return stringList(item);
    if (key === "include_media_ids") return list(item, positiveId);
    if (key === "enable_web_fallback") return typeof item === "boolean";
    if (key === "collection_id")
      return item === null || shortString(item) || positiveId(item);
    return (
      item === null || item === "" || shortString(item) || stringList(item)
    );
  });
const source = (value: unknown): boolean =>
  keys(value, [
    "originalId",
    "excerpt",
    "mediaId",
    "title",
    "type",
    "sourceType",
    "url",
    "snapshotMediaId",
    "originalVersion",
    "pageNumber",
    "citationIndex",
  ]) &&
  (value.originalId === null ||
    shortString(value.originalId) ||
    positiveId(value.originalId)) &&
  excerpt(value.excerpt) &&
  (value.mediaId === null || positiveId(value.mediaId)) &&
  shortString(value.title) &&
  typeof value.type === "string" &&
  ["pdf", "video", "audio", "website", "text", "document"].includes(
    value.type,
  ) &&
  (value.sourceType === null || shortString(value.sourceType)) &&
  optional(
    value.url,
    (item) => typeof item === "string" && item.length <= 4096,
  ) &&
  [
    value.snapshotMediaId,
    value.originalVersion,
    value.pageNumber,
    value.citationIndex,
  ].every((item) => optional(item, positiveId));
const evidence = (value: unknown): boolean =>
  keys(value, [
    "importId",
    "threadId",
    "snapshot",
    "sources",
    "trustState",
    "trustReasonCodes",
    "evidenceOrigin",
    "scope",
  ]) &&
  shortString(value.importId) &&
  (value.threadId === null || shortString(value.threadId)) &&
  typeof value.snapshot === "boolean" &&
  list(value.sources, source) &&
  optional(value.trustState, isKnowledgeAnswerTrustState) &&
  optional(value.trustReasonCodes, stringList) &&
  optional(value.evidenceOrigin, origin) &&
  optional(value.scope, scope);
const research = (value: unknown): boolean =>
  keys(value, ["workspace_id", "import_id", "sources"]) &&
  shortString(value.workspace_id) &&
  shortString(value.import_id) &&
  list(
    value.sources,
    (item) =>
      keys(item, ["mediaId", "evidence"]) &&
      positiveId(item.mediaId) &&
      evidence(item.evidence),
  );

/** Match the strict server contract; never salvage arbitrary metadata as evidence. */
export const validateKnowledgeNoteProvenance = (
  value: unknown,
): KnowledgeNoteProvenance | null => {
  if (
    !keys(value, [
      "origin",
      "trust_state",
      "evidence_origin",
      "thread_id",
      "research",
      "question",
      "scope",
      "trust_reason_codes",
      "sources",
    ]) ||
    (value.origin !== "knowledge_qa" && value.origin !== "reviewed_sources") ||
    !optional(value.trust_state, isKnowledgeAnswerTrustState) ||
    !optional(value.evidence_origin, origin) ||
    !optional(value.thread_id, shortString) ||
    !optional(value.research, research) ||
    !optional(value.question, excerpt) ||
    !optional(value.scope, scope) ||
    !optional(value.trust_reason_codes, stringList) ||
    !optional(value.sources, (item) => list(item, source))
  )
    return null;
  try {
    const serialized = JSON.stringify(value);
    // JSON escapes lone surrogates, but neither server UTF-8 nor portable URI encoding accepts them.
    const stringsValid = (item: unknown): boolean =>
      typeof item === "string"
        ? !/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(
            item,
          )
        : item === null ||
          typeof item !== "object" ||
          Object.values(item).every(stringsValid);
    if (
      !stringsValid(value) ||
      encodeURIComponent(serialized).length > MAX_MARKER_LENGTH
    )
      return null;
    return JSON.parse(serialized);
  } catch {
    return null;
  }
};

export const readKnowledgeNoteProvenance = (
  content: string,
): KnowledgeNoteProvenance | null => {
  for (const match of content.matchAll(new RegExp(markerPattern))) {
    if (match[1].length > MAX_MARKER_LENGTH) continue;
    try {
      const parsed = validateKnowledgeNoteProvenance(
        JSON.parse(decodeURIComponent(match[1])),
      );
      if (parsed) return parsed;
    } catch {
      /* Malformed comments grant no provenance. */
    }
  }
  return null;
};

const canonicalJson = (value: unknown): string =>
  JSON.stringify(value, (_, item) =>
    isRecord(item)
      ? Object.fromEntries(
          Object.entries(item).sort(([a], [b]) => a.localeCompare(b)),
        )
      : item,
  );

export const knowledgeNoteProvenanceMatches = (
  left: unknown,
  right: unknown,
): boolean => canonicalJson(left) === canonicalJson(right);

/** Resolve at every read/save/export boundary. A retained tombstone never falls back. */
export const resolveKnowledgeNoteProvenance = (
  note?: unknown,
  content = "",
) => {
  const record = isRecord(note) ? note : {};
  const marker =
    readKnowledgeNoteProvenance(content) ||
    readKnowledgeNoteProvenance(
      typeof record.content === "string" ? record.content : "",
    );
  const state = record.knowledge_provenance_state;
  const metadata = isRecord(record.metadata) ? record.metadata : {};
  const canonical = state === "active" || state === "deleted";
  const provenance =
    state === "deleted"
      ? null
      : state === "active"
        ? validateKnowledgeNoteProvenance(record.knowledge_provenance)
        : validateKnowledgeNoteProvenance(record.knowledge_provenance) ||
          validateKnowledgeNoteProvenance(record) ||
          validateKnowledgeNoteProvenance(
            metadata.knowledge_provenance,
          ) ||
          validateKnowledgeNoteProvenance(metadata) ||
          marker;
  const reconciliation =
    record.knowledge_provenance_reconciliation === "canonical_wins" ||
    (canonical &&
      marker !== null &&
      !knowledgeNoteProvenanceMatches(marker, provenance));
  return { provenance, state, reconciliation };
};

/** Copy only independent head fields, while retaining legacy portable evidence. */
export const knowledgeNoteHead = (
  note?: unknown,
  content = "",
): KnowledgeNoteHead => {
  const record = isRecord(note) ? note : {};
  const resolved = resolveKnowledgeNoteProvenance(record, content);
  if (record.knowledge_provenance_state === undefined && !resolved.provenance)
    return {};
  return {
    ...(record.knowledge_provenance_state !== undefined
      ? {
          knowledge_provenance_state: record.knowledge_provenance_state as KnowledgeNoteHead["knowledge_provenance_state"],
          knowledge_provenance_version: record.knowledge_provenance_version as KnowledgeNoteHead["knowledge_provenance_version"],
          knowledge_provenance_hash: record.knowledge_provenance_hash as KnowledgeNoteHead["knowledge_provenance_hash"],
        }
      : {}),
    knowledge_provenance: resolved.provenance,
    knowledge_provenance_reconciliation: resolved.reconciliation
      ? "canonical_wins"
      : null,
  };
};

export const retainKnowledgeNoteProvenance = (
  content: string,
  metadata?: unknown,
): string => {
  const { provenance, state } = resolveKnowledgeNoteProvenance(
    metadata,
    content,
  );
  if (!provenance)
    return state === "deleted" || state === "active"
      ? stripKnowledgeNoteProvenance(content)
      : content;
  const body = stripKnowledgeNoteProvenance(content).trimEnd();
  return `${body}\n\n<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(provenance))} -->`;
};

/** New history and absent-head backfill use explicit bases; ordinary edits omit unchanged history. */
export const knowledgeNoteWriteFields = (
  content: string,
  head?: unknown,
  options: { create?: boolean; replacement?: KnowledgeNoteProvenance } = {},
) => {
  const record = isRecord(head) ? head : {};
  const { provenance, state } = resolveKnowledgeNoteProvenance(record, content);
  if (state === "deleted" || state === "unsupported") return {};
  const replacement = options.replacement
    ? validateKnowledgeNoteProvenance(options.replacement)
    : null;
  if (options.replacement && !replacement)
    throw new Error("Source history is invalid or too large to save");
  const value = replacement || provenance;
  if (!value) return {};
  if (
    options.create ||
    (state === "absent" && record.knowledge_provenance_version === 0)
  )
    return { knowledge_provenance: value, expected_provenance_version: 0 };
  if (
    replacement &&
    state === "active" &&
    positiveId(record.knowledge_provenance_version) &&
    !knowledgeNoteProvenanceMatches(replacement, provenance)
  )
    return {
      knowledge_provenance: replacement,
      expected_provenance_version: record.knowledge_provenance_version,
    };
  return {};
};

export const stripKnowledgeNoteProvenance = (content: string): string => {
  let removed = false;
  const body = content.replace(new RegExp(markerPattern), (marker) => {
    if (!readKnowledgeNoteProvenance(marker)) return marker;
    removed = true;
    return "";
  });
  return removed ? body.trimEnd() : content;
};

/** Explicit cited-message capture merges evidence without changing the saved canonical head. */
export const appendCapturedNoteProvenance = (
  head: KnowledgeNoteHead & {
    pendingKnowledgeProvenance?: KnowledgeNoteProvenance;
    content?: string;
  },
  cited: WorkspaceSource[],
): KnowledgeNoteProvenance | null => {
  if (
    head.knowledge_provenance_state === "deleted" ||
    head.knowledge_provenance_state === "unsupported"
  )
    return null;
  const captures = cited.filter((source) => source.webCapture);
  if (!captures.length) return null;
  const original =
    head.pendingKnowledgeProvenance ||
    resolveKnowledgeNoteProvenance(head).provenance;
  const sources = [...(original?.sources || [])];
  for (const source of captures) {
    const pin = source.webCapture!;
    const refs: KnowledgeNoteSource[] = [
      ...(source.knowledgeQaEvidence?.sources || []),
      {
        originalId: pin.clipId,
        excerpt: "",
        mediaId: pin.mediaId,
        title: source.title,
        type: source.type,
        sourceType: "server_article",
        url: pin.requestedUrl,
        snapshotMediaId: pin.mediaId,
        originalVersion: pin.versionNumber,
      },
    ];
    for (const reference of refs)
      if (
        !sources.some((existing) =>
          knowledgeNoteProvenanceMatches(existing, reference),
        )
      )
        sources.push(reference);
  }
  const result = validateKnowledgeNoteProvenance({
    ...original,
    origin: original?.origin || "reviewed_sources",
    sources,
  });
  if (!result)
    throw new Error("Source history is invalid or too large to save");
  return result;
};
