import { browser } from "wxt/browser";
import type { BgUploadInit } from "@/services/background-proxy";
import { assertQuickIngestRequestCurrent, scopedQuickIngestRequest, scopedQuickIngestUpload, type QuickIngestRequestContext } from "./quick-ingest-request-scope";
import {
  getProcessPathForType,
  resolveIngestUrlType,
  inferUploadMediaTypeFromFile,
  normalizeMediaType,
  shouldKeepOriginalFile,
} from "@/services/tldw/media-routing";
import {
  applyQuickIngestChunkingFields,
  shouldSubmitQuickIngestAdvancedField,
  type QuickIngestAutoChunkingGoal,
  type QuickIngestChunkingMode,
} from "@/services/tldw/quick-ingest-chunking";
import {
  ACTIVE_INGEST_JOB_STATUSES,
  INGEST_JOB_STATUS_INTERRUPTED,
  TERMINAL_INGEST_JOB_STATUSES,
  getQuickIngestJobPollInterval,
  isTransientIngestStatusRead,
  pollSingleIngestJob,
  readIngestJobProgress,
  type IngestJobProgress,
  type IngestJobStatusResponse,
  requireSubmittedIngestJobs,
} from "@/services/tldw/ingest-jobs-orchestrator";
import {
  completedIngestJobIndicatesFailure,
  completedIngestJobIndicatesSkipped,
  extractCompletedIngestJobError,
  extractCompletedIngestJobMediaId,
} from "@/services/tldw/ingest-job-results";
import {
  normalizePersistentAddResponse,
  shouldFallbackToPersistentAdd,
} from "@/services/tldw/quick-ingest-fallback";
import type {
  ConferenceBatchMetadata,
  ConferenceItemMetadataOverride,
  PersistedQuickIngestTracking,
  PlaylistQueueMetadata,
} from "@/components/Common/QuickIngest/types";
import {
  DUPLICATE_SKIP_MESSAGE,
  getQuickIngestFileSizeError,
  isDbMessageDuplicate,
} from "@/components/Common/QuickIngest/constants";
import {
  buildConferenceCollectionCreatePayload,
  buildConferenceCollectionItemPayload,
  normalizeMediaCollectionItem,
  normalizeMediaCollectionResponse,
  resolveConferenceDuplicatePolicy,
  type ApiMediaCollection,
  type ApiMediaCollectionItem,
  type ConferenceRetryRequestItem,
  type MediaCollectionItemStatus,
} from "@/services/tldw/conference-collections";

type TypeDefaults = {
  audio?: { language?: string; diarize?: boolean };
  document?: { ocr?: boolean };
  video?: { captions?: boolean };
};

type QuickIngestEntry = {
  id: string;
  url: string;
  type: "auto" | "html" | "pdf" | "document" | "audio" | "video";
  defaults?: TypeDefaults;
  keywords?: string;
  playlist?: PlaylistQueueMetadata;
  processAgain?: boolean;
  sourceUrlEdited?: boolean;
  conferenceOverride?: ConferenceItemMetadataOverride;
  audio?: { language?: string; diarize?: boolean };
  document?: { ocr?: boolean };
  video?: { captions?: boolean };
};

type QuickIngestFilePayload = {
  id?: string;
  name?: string;
  type?: string;
  data?: number[] | Uint8Array | ArrayBuffer;
  directFile?: File;
  defaults?: TypeDefaults;
  conferenceOverride?: ConferenceItemMetadataOverride;
};

type QuickIngestBatchInput = QuickIngestRequestContext & {
  entries: QuickIngestEntry[];
  files: QuickIngestFilePayload[];
  storeRemote: boolean;
  processOnly: boolean;
  common?: {
    perform_analysis?: boolean;
    perform_chunking?: boolean;
    overwrite_existing?: boolean;
    chunking_mode?: QuickIngestChunkingMode;
    auto_chunking_goal?: QuickIngestAutoChunkingGoal;
    auto_chunking_use_llm?: boolean;
  };
  advancedValues?: Record<string, any>;
  fileDefaults?: TypeDefaults;
  conferenceBatchMetadata?: ConferenceBatchMetadata | null;
  conferenceRetry?: { collectionId: string; items: ConferenceRetryRequestItem[] };
  chunkingTemplateName?: string;
  autoApplyTemplate?: boolean;
  __quickIngestSessionId?: string;
  onTrackingMetadata?: (tracking: PersistedQuickIngestTracking) => void;
  onJobProgress?: (itemId: string, progress: IngestJobProgress) => void;
  onJobStatusUnavailable?: (unavailable: boolean) => void;
  onStopUnconfirmed?: () => void;
  onItemResult?: (result: QuickIngestBatchResult) => void;
  isJobCancellationConfirmed?: (itemId: string) => boolean;
};

type QuickIngestBatchResult = {
  id: string;
  status: "ok" | "error";
  outcome?: "skipped" | "submit_failed" | "failed" | "cancelled";
  url?: string;
  fileName?: string;
  type: string;
  data?: unknown;
  error?: string;
  message?: string;
  persisted?: boolean;
  mediaId?: string | number | null;
  collectionItemId?: string | number | null;
  retryAttempt?: number | null;
  idempotencyKey?: string | null;
};

type QuickIngestBatchResponse = {
  ok: boolean;
  error?: string;
  results?: QuickIngestBatchResult[];
  interruptedJobIds?: number[];
  interruptedItemIds?: string[];
};

export const QUICK_INGEST_ANALYSIS_PROVIDER_WARNING =
  "Choose an analysis provider before running ingest analysis.";

const hasAnalysisProviderValue = (value: unknown): boolean => {
  const normalized = String(value || "").trim();
  return Boolean(normalized) && normalized.toLowerCase() !== "none";
};

export const getQuickIngestAnalysisProviderWarning = (
  input: Pick<QuickIngestBatchInput, "common" | "advancedValues">,
): string | null => {
  if (!input?.common?.perform_analysis) return null;
  const advancedValues = input.advancedValues || {};
  if (hasAnalysisProviderValue(advancedValues.api_name)) {
    return null;
  }
  return QUICK_INGEST_ANALYSIS_PROVIDER_WARNING;
};

export type QuickIngestStartAck = {
  ok: boolean;
  sessionId?: string;
  error?: string;
};

export type QuickIngestCancelInput = QuickIngestRequestContext & {
  sessionId: string;
  reason?: string;
  batchIds?: string[];
  tracking?: PersistedQuickIngestTracking;
};

export type QuickIngestCancelResponse = {
  ok: boolean;
  error?: string;
};

const EXTENSION_TIMEOUT_MS = 10_000;
const QUICK_INGEST_RUNTIME_PING_TIMEOUT_MS = 400;
const QUICK_INGEST_RUNTIME_HEALTH_TTL_MS = 30_000;
const DIRECT_INGEST_TIMEOUT_MS = 5 * 60 * 1000;
const LARGE_NATIVE_FILE_TIMEOUT_MS = 20 * 60_000;
const LARGE_NATIVE_UPLOAD_THRESHOLD_BYTES = 50 * 1024 ** 2;
const DIRECT_REMOTE_POLL_INTERVAL_MS = 1_200;
/** Stop new submissions and abort outstanding uploads without abandoning accepted jobs. */
const directRuns = new Map<string, { uploads: AbortController; batchIds: Set<string> }>();
let lastQuickIngestRuntimeHealthCheckAt = 0;
let quickIngestRuntimeMessagingUsable: boolean | null = null;

const DIRECT_QUICK_INGEST_SESSION_PREFIX = "qi-direct-";
const DIRECT_QUICK_INGEST_TRANSPORT = { preferDirect: true } as const;

const buildDirectSessionSuffix = (): string => {
  try {
    if (
      typeof globalThis !== "undefined" &&
      typeof globalThis.crypto?.randomUUID === "function"
    ) {
      return globalThis.crypto.randomUUID().replace(/-/g, "").slice(0, 8);
    }
    if (
      typeof globalThis !== "undefined" &&
      typeof globalThis.crypto?.getRandomValues === "function"
    ) {
      const bytes = new Uint8Array(4);
      globalThis.crypto.getRandomValues(bytes);
      return Array.from(bytes, (byte) =>
        byte.toString(16).padStart(2, "0"),
      ).join("");
    }
  } catch {
    // Fall through to timestamp suffix below.
  }
  return Date.now().toString(36).slice(-8);
};

const normalizeBatchIds = (batchIds?: string[]): string[] =>
  Array.from(
    new Set(
      Array.isArray(batchIds)
        ? batchIds
            .map((batchId) => String(batchId || "").trim())
            .filter(Boolean)
        : [],
    ),
  );

const buildJobIdToItemId = (
  jobIds: number[],
  sourceItemId: string,
): Record<string, string> =>
  Object.fromEntries(jobIds.map((jobId) => [String(jobId), sourceItemId]));

const buildJobIdToCollectionItemId = (
  jobIds: number[],
  planned: PlannedConferenceCollectionItem | undefined,
): Record<string, string> | undefined => {
  if (!planned) return undefined;
  return Object.fromEntries(
    jobIds.map((jobId) => [String(jobId), String(planned.itemId)]),
  );
};

const requestBatchStop = async (
  input: QuickIngestRequestContext,
  batchId: string,
  reason = "user_cancelled",
): Promise<boolean> => {
  try {
    const response = await scopedQuickIngestRequest<{
      ok?: boolean;
      data?: { success?: boolean; failed?: number };
    }>(input, {
      path: `/api/v1/media/ingest/jobs/cancel?batch_id=${encodeURIComponent(batchId)}&reason=${encodeURIComponent(reason)}`,
      method: "POST",
      timeoutMs: 10_000,
      returnResponse: true,
      ...DIRECT_QUICK_INGEST_TRANSPORT,
    });
    return response?.ok === true && response.data?.success === true && !response.data.failed;
  } catch (error) {
    assertQuickIngestRequestCurrent(input, error);
    return false;
  }
};

const hasExtensionMessagingRuntime = (): boolean =>
  Boolean(browser?.runtime?.sendMessage && browser?.runtime?.id);

const getRuntimeManifestVersion = (): number | null => {
  try {
    const manifest = browser?.runtime?.getManifest?.() as
      | { manifest_version?: number }
      | undefined;
    const manifestVersion = Number(manifest?.manifest_version);
    return Number.isFinite(manifestVersion) && manifestVersion > 0
      ? Math.trunc(manifestVersion)
      : null;
  } catch {
    return null;
  }
};

const shouldPreferDirectQuickIngestSession = (): boolean =>
  (getRuntimeManifestVersion() || 0) >= 3;

const isDirectQuickIngestSessionId = (sessionId: string | undefined): boolean =>
  String(sessionId || "").trim().startsWith(DIRECT_QUICK_INGEST_SESSION_PREFIX);

const invalidateQuickIngestRuntimeHealth = (): void => {
  quickIngestRuntimeMessagingUsable = false;
  lastQuickIngestRuntimeHealthCheckAt = Date.now();
};

const canUseExtensionMessagingRuntime = async (): Promise<boolean> => {
  if (!hasExtensionMessagingRuntime()) return false;

  const now = Date.now();
  if (
    quickIngestRuntimeMessagingUsable !== null &&
    now - lastQuickIngestRuntimeHealthCheckAt <
      QUICK_INGEST_RUNTIME_HEALTH_TTL_MS
  ) {
    return quickIngestRuntimeMessagingUsable;
  }

  try {
    const pingResult = await Promise.race([
      browser.runtime.sendMessage({ type: "tldw:ping" }),
      new Promise<null>((resolve) =>
        setTimeout(() => resolve(null), QUICK_INGEST_RUNTIME_PING_TIMEOUT_MS),
      ),
    ]);
    quickIngestRuntimeMessagingUsable =
      Boolean(pingResult) && Boolean((pingResult as { ok?: unknown }).ok);
  } catch {
    quickIngestRuntimeMessagingUsable = false;
  }
  lastQuickIngestRuntimeHealthCheckAt = now;
  return Boolean(quickIngestRuntimeMessagingUsable);
};

const sendExtensionMessageWithTimeout = async <T>(
  message: Record<string, unknown>,
  timeoutMs: number = EXTENSION_TIMEOUT_MS,
): Promise<T> => {
  const extensionPromise = browser.runtime.sendMessage(message);
  const timeoutPromise = new Promise<null>((resolve) => {
    setTimeout(() => resolve(null), timeoutMs);
  });
  const result = await Promise.race([extensionPromise, timeoutPromise]);
  if (result === null) {
    invalidateQuickIngestRuntimeHealth();
    throw new Error(
      "Extension messaging timed out. Please try again or reload the page.",
    );
  }
  try {
    return result as T;
  } catch (error) {
    invalidateQuickIngestRuntimeHealth();
    throw error;
  }
};

const assignPath = (obj: Record<string, any>, path: string[], val: any) => {
  let cur: Record<string, any> = obj;
  for (let i = 0; i < path.length; i += 1) {
    const seg = path[i];
    if (!seg) continue;
    if (i === path.length - 1) {
      cur[seg] = val;
      return;
    }
    const existing = cur[seg];
    if (!existing || typeof existing !== "object" || Array.isArray(existing)) {
      cur[seg] = {};
    }
    cur = cur[seg];
  }
};

const normalizeJsonField = (value: unknown) => {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  if (!trimmed) return value;
  const looksJson =
    (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
    (trimmed.startsWith("[") && trimmed.endsWith("]"));
  if (!looksJson) return value;
  try {
    return JSON.parse(trimmed);
  } catch {
    return value;
  }
};

const serializeUploadFields = (
  fields: Record<string, any>,
): Record<string, any> => {
  const serialized: Record<string, any> = {};
  for (const [key, value] of Object.entries(fields || {})) {
    if (value == null) continue;
    if (Array.isArray(value)) {
      serialized[key] = value.map((entry) =>
        typeof entry === "string" ? entry : JSON.stringify(entry),
      );
      continue;
    }
    if (typeof value === "object") {
      serialized[key] = JSON.stringify(value);
      continue;
    }
    serialized[key] = value;
  }
  return serialized;
};

const hasNativeFile = (files: QuickIngestFilePayload[]): boolean =>
  Array.isArray(files) && files.some((file) => file?.directFile);

const getNativeFileTimeout = (file?: File): number =>
  file && file.size > LARGE_NATIVE_UPLOAD_THRESHOLD_BYTES
    ? LARGE_NATIVE_FILE_TIMEOUT_MS
    : DIRECT_INGEST_TIMEOUT_MS;

const submitPersistentAdd = async (input: QuickIngestRequestContext, {
  fields,
  file,
  directFile,
}: Pick<BgUploadInit, "file" | "directFile"> & { fields: Record<string, any> }): Promise<any> =>
  normalizePersistentAddResponse(
    await scopedQuickIngestUpload<any>(input, {
      path: "/api/v1/media/add",
      method: "POST",
      fields: serializeUploadFields(fields),
      file,
      ...(directFile ? { directFile } : {}),
      fileFieldName: file || directFile ? "files" : undefined,
      timeoutMs: getNativeFileTimeout(directFile),
      ...DIRECT_QUICK_INGEST_TRANSPORT,
    }),
  );

const buildFields = ({
  rawType,
  entry,
  defaults,
  common,
  advancedValues,
  chunkingTemplateName,
  autoApplyTemplate,
  persist = true,
}: {
  rawType: string;
  entry?: QuickIngestEntry;
  defaults?: TypeDefaults;
  common?: QuickIngestBatchInput["common"];
  advancedValues?: Record<string, any>;
  chunkingTemplateName?: string;
  autoApplyTemplate?: boolean;
  persist?: boolean;
}): Record<string, any> => {
  const mediaType = normalizeMediaType(rawType);
  const fields: Record<string, any> = {
    media_type: mediaType,
    perform_analysis: Boolean(common?.perform_analysis),
    overwrite_existing: Boolean(common?.overwrite_existing),
    keep_original_file: persist && shouldKeepOriginalFile(rawType),
  };

  const nested: Record<string, any> = {};
  for (const [key, value] of Object.entries(advancedValues || {})) {
    if (!shouldSubmitQuickIngestAdvancedField(key, common)) continue;
    if (key.includes(".")) assignPath(nested, key.split("."), value);
    else fields[key] = value;
  }
  for (const [key, value] of Object.entries(nested)) {
    fields[key] = value;
  }

  if (typeof entry?.keywords === "string") {
    const trimmed = entry.keywords.trim();
    if (trimmed) {
      fields.keywords = trimmed;
    }
  }

  const resolvedDefaults: TypeDefaults = (() => {
    if (!defaults || typeof defaults !== "object") return {};
    if (mediaType === "audio") return { audio: defaults.audio };
    if (mediaType === "video") {
      return { audio: defaults.audio, video: defaults.video };
    }
    if (
      mediaType === "document" ||
      mediaType === "pdf" ||
      mediaType === "ebook"
    ) {
      return { document: defaults.document };
    }
    return {};
  })();

  const audio = { ...(resolvedDefaults.audio || {}), ...(entry?.audio || {}) };
  const video = { ...(resolvedDefaults.video || {}), ...(entry?.video || {}) };
  const document = {
    ...(resolvedDefaults.document || {}),
    ...(entry?.document || {}),
  };

  if (audio.language && fields.transcription_language == null) {
    fields.transcription_language = audio.language;
  }
  if (typeof audio.diarize === "boolean" && fields.diarize == null) {
    fields.diarize = audio.diarize;
  }
  if (typeof video.captions === "boolean" && fields.timestamp_option == null) {
    fields.timestamp_option = video.captions;
  }
  if (
    mediaType === "pdf" &&
    typeof document.ocr === "boolean" &&
    fields.enable_ocr == null
  ) {
    fields.enable_ocr = document.ocr;
  }
  if (typeof document.ocr === "boolean" && fields.pdf_parsing_engine == null) {
    fields.pdf_parsing_engine = document.ocr ? "pymupdf4llm" : "";
  }

  applyQuickIngestChunkingFields(fields, {
    common,
    chunkingTemplateName,
    autoApplyTemplate,
  });

  return fields;
};

const processWebScrape = async (input: QuickIngestRequestContext, {
  url,
  entry,
  common,
  advancedValues,
  chunkingTemplateName,
  autoApplyTemplate,
  persist = false,
}: {
  url: string;
  entry?: QuickIngestEntry;
  common?: QuickIngestBatchInput["common"];
  advancedValues?: Record<string, any>;
  chunkingTemplateName?: string;
  autoApplyTemplate?: boolean;
  persist?: boolean;
}): Promise<any> => {
  const nestedBody: Record<string, any> = {};
  for (const [key, value] of Object.entries(advancedValues || {})) {
    if (!shouldSubmitQuickIngestAdvancedField(key, common)) continue;
    if (key.includes(".")) assignPath(nestedBody, key.split("."), value);
    else nestedBody[key] = value;
  }

  const normalizedBody: Record<string, any> = { ...nestedBody };
  for (const key of ["custom_headers", "custom_cookies", "custom_titles"]) {
    if (key in normalizedBody) {
      normalizedBody[key] = normalizeJsonField(normalizedBody[key]);
    }
  }

  const body: Record<string, any> = {
    scrape_method: "Individual URLs",
    url_input: url,
    summarize_checkbox: Boolean(common?.perform_analysis),
    ...normalizedBody,
    mode: persist ? "persist" : "ephemeral",
    include_extracted_content: !persist,
  };
  if (persist && body.summarize_checkbox) {
    body.skip_existing_analysis = !common?.overwrite_existing;
  }
  applyQuickIngestChunkingFields(body, {
    common,
    chunkingTemplateName,
    autoApplyTemplate,
  });

  if (typeof entry?.keywords === "string") {
    const trimmed = entry.keywords.trim();
    if (trimmed) {
      body.keywords = trimmed;
    }
  }

  return await scopedQuickIngestRequest<any>(input, {
    path: "/api/v1/media/process-web-scraping",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
    ...DIRECT_QUICK_INGEST_TRANSPORT,
  });
};

const extractWebScrapeMediaId = (value: unknown): string | number | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const mediaIds = Array.isArray(record.media_ids) ? record.media_ids : [];
  const candidate = mediaIds.length > 0 ? mediaIds[0] : record.media_id;
  return typeof candidate === "string" || typeof candidate === "number"
    ? candidate
    : null;
};

const webScrapeResponseIndicatesPersisted = (
  value: unknown,
  mediaId: string | number | null,
): boolean => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return mediaId != null;
  }
  const status = String((value as Record<string, unknown>).status || "")
    .trim()
    .toLowerCase();
  return mediaId != null || status === "persist-ok";
};

type PlannedConferenceCollectionItem = {
  collectionId: number;
  itemId: number;
  retryAttempt?: number;
  idempotencyKey?: string | null;
};

type PlannedConferenceCollection = {
  collectionId: number;
  itemsByEntryId: Map<string, PlannedConferenceCollectionItem>;
};

const hasUsableConferenceMetadata = (
  metadata: QuickIngestBatchInput["conferenceBatchMetadata"],
): metadata is ConferenceBatchMetadata =>
  Boolean(
    metadata &&
      typeof metadata === "object" &&
      (
        String(metadata.collectionName || "").trim() ||
        String(metadata.conferenceName || "").trim() ||
        String(metadata.sourcePlaylistUrl || "").trim() ||
        (Array.isArray(metadata.sharedTags) && metadata.sharedTags.length > 0)
      ),
  );

const getConferenceFallbackName = (entries: QuickIngestEntry[]): string | null => {
  for (const entry of entries) {
    const title = String(entry?.playlist?.playlistTitle || "").trim();
    if (title) return title;
  }
  return null;
};

const createPlannedConferenceCollection = async (
  input: QuickIngestBatchInput,
  entries: QuickIngestEntry[],
): Promise<PlannedConferenceCollection | null> => {
  if (input.conferenceRetry) {
    const collectionId = Number(input.conferenceRetry.collectionId);
    if (!Number.isSafeInteger(collectionId) || collectionId <= 0) throw new Error("Invalid retry collection identity");
    const requests = new Map(input.conferenceRetry.items.map(item => [item.resultId, item]));
    const itemsByEntryId = new Map<string, PlannedConferenceCollectionItem>();
    for (const entry of entries) {
      const request = requests.get(entry.id);
      const itemId = Number(request?.collectionItemId);
      if (!request || !Number.isSafeInteger(itemId) || itemId <= 0 || !Number.isSafeInteger(request.retryAttempt) || request.retryAttempt <= 0 || !request.idempotencyKey) {
        throw new Error("Invalid retry item identity");
      }
      if (entry.sourceUrlEdited && entry.url) {
        const receipt = await scopedQuickIngestRequest<ApiMediaCollectionItem>(input, {
          path: `/api/v1/media/collections/${collectionId}/items/${itemId}`,
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: { source_url: entry.url, retry_count: request.retryAttempt, idempotency_key: request.idempotencyKey },
          timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
          ...DIRECT_QUICK_INGEST_TRANSPORT,
        });
        if (Number(receipt?.id) !== itemId || receipt.source_url !== entry.url) {
          throw new Error("The collection did not confirm the corrected source URL.");
        }
      }
      itemsByEntryId.set(entry.id, { collectionId, itemId, retryAttempt: request.retryAttempt, idempotencyKey: request.idempotencyKey });
    }
    return { collectionId, itemsByEntryId };
  }
  if (!hasUsableConferenceMetadata(input.conferenceBatchMetadata)) return null;
  const selectedEntries = entries.filter(
    (entry) =>
      String(entry?.url || "").trim() &&
      entry?.conferenceOverride?.selected !== false,
  );
  if (selectedEntries.length === 0) return null;

  const collection = normalizeMediaCollectionResponse(
    await scopedQuickIngestRequest<ApiMediaCollection>(input, {
      path: "/api/v1/media/collections",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: buildConferenceCollectionCreatePayload(
        input.conferenceBatchMetadata,
        getConferenceFallbackName(selectedEntries),
      ),
      timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
      ...DIRECT_QUICK_INGEST_TRANSPORT,
    }),
  );

  const itemsByEntryId = new Map<string, PlannedConferenceCollectionItem>();
  for (const entry of selectedEntries) {
    const item = normalizeMediaCollectionItem(
      await scopedQuickIngestRequest<ApiMediaCollectionItem>(input, {
        path: `/api/v1/media/collections/${encodeURIComponent(String(collection.id))}/items`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: buildConferenceCollectionItemPayload(input.conferenceBatchMetadata, {
          id: entry.id,
          url: entry.url,
          playlist: entry.playlist,
          processAgain: entry.processAgain,
          conferenceOverride: entry.conferenceOverride,
        }),
        timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
        ...DIRECT_QUICK_INGEST_TRANSPORT,
      }),
    );
    itemsByEntryId.set(entry.id, {
      collectionId: collection.id,
      itemId: item.id,
      idempotencyKey: item.idempotencyKey,
    });
  }

  return {
    collectionId: collection.id,
    itemsByEntryId,
  };
};

const patchConferenceCollectionItem = async (
  input: QuickIngestRequestContext,
  planned: PlannedConferenceCollectionItem | undefined,
  payload: Record<string, unknown>,
): Promise<void> => {
  if (!planned) return;
  await scopedQuickIngestRequest<ApiMediaCollectionItem>(input, {
    path: `/api/v1/media/collections/${encodeURIComponent(
      String(planned.collectionId),
    )}/items/${encodeURIComponent(String(planned.itemId))}`,
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: {
      ...payload,
      ...(planned.retryAttempt
        ? {
            retry_count: planned.retryAttempt,
            idempotency_key: planned.idempotencyKey
          }
        : {})
    },
    timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
    ...DIRECT_QUICK_INGEST_TRANSPORT,
  }).catch(() => {
    // Collection status updates should not hide the ingest result itself.
  });
};

const getConferenceTerminalStatus = (
  duplicate: boolean,
  failed: boolean,
): MediaCollectionItemStatus =>
  duplicate ? "skipped_existing" : failed ? "failed" : "completed";

const applyPlannedConferenceFields = (
  fields: Record<string, any>,
  planned: PlannedConferenceCollectionItem | undefined,
) => {
  if (!planned) return;
  fields.media_collection_id = planned.collectionId;
  fields.media_collection_item_id = planned.itemId;
  fields.planned_item_ids = [String(planned.itemId)];
  if (planned.idempotencyKey) {
    fields.idempotency_key = planned.idempotencyKey;
    fields.idempotency_keys = [planned.idempotencyKey];
  }
};

const runDirectQuickIngestBatch = async (
  input: QuickIngestBatchInput,
): Promise<QuickIngestBatchResponse> => {
  const entries = Array.isArray(input.entries)
    ? input.entries.filter((entry) => entry?.conferenceOverride?.selected !== false)
    : [];
  const files = Array.isArray(input.files)
    ? input.files.filter((file) => file?.conferenceOverride?.selected !== false)
    : [];
  const fileDefaults =
    input.fileDefaults && typeof input.fileDefaults === "object"
      ? input.fileDefaults
      : {};
  const shouldStoreRemote =
    Boolean(input.storeRemote) && !input.processOnly;
  const directSessionId =
    String(input.__quickIngestSessionId || "").trim() || undefined;
  const uploads = new AbortController();
  const batchIds = new Set<string>();
  const runContext = {
    ...input,
    signal: input.signal
      ? AbortSignal.any([input.signal, uploads.signal])
      : uploads.signal,
  };

  const out: QuickIngestBatchResult[] = [];
  const interruptedJobIds = new Set<number>();
  const interruptedItemIds: string[] = [];
  const unavailableJobIds = new Set<number>();
  const inputOrder = new Map([...entries, ...files].map((item, index) => [item.id, index]));
  const acceptedMonitors: Promise<{ error: unknown } | void>[] = [];
  const monitorAcceptedJob = (monitor: Promise<void>) => {
    // Observe failures immediately while later POSTs are still in flight.
    acceptedMonitors.push(monitor.then(() => undefined, (error) => ({ error })));
  };
  const recordResult = (result: QuickIngestBatchResult) => {
    out.push(result);
    input.onItemResult?.(result);
  };
  let conferencePlan: PlannedConferenceCollection | null = null;

  const trackAcceptedBatch = async (tracking: PersistedQuickIngestTracking & { batchId: string }) => {
    const stopAlreadyRequested = uploads.signal.aborted;
    batchIds.add(tracking.batchId);
    input.onTrackingMetadata?.(tracking);
    // A URL submission already on the wire can return a job after the stop request.
    if (stopAlreadyRequested && !(await requestBatchStop(input, tracking.batchId))) {
      input.onStopUnconfirmed?.();
    }
  };

  const pollIngestJobStatus = async (jobId: number, itemId: string) => {
    const startedAt = Date.now();
    let statusUnavailable = false;
    const reportStatusUnavailable = (unavailable: boolean) => {
      statusUnavailable = unavailable;
      if (unavailable) unavailableJobIds.add(jobId);
      else unavailableJobIds.delete(jobId);
      input.onJobStatusUnavailable?.(unavailableJobIds.size > 0);
    };
    const invalidStatusResponse = { ok: false };
    const result = await pollSingleIngestJob({
      jobId,
      timeoutMs: Infinity,
      pollIntervalMs: () => getQuickIngestJobPollInterval(startedAt, DIRECT_REMOTE_POLL_INTERVAL_MS, statusUnavailable),
      signal: input.signal,
      fetchJob: async (trackedJobId) => {
        try {
          const response = await scopedQuickIngestRequest<IngestJobStatusResponse | undefined>(input, {
            path: `/api/v1/media/ingest/jobs/${trackedJobId}`,
            method: "GET",
            timeoutMs: DIRECT_REMOTE_POLL_INTERVAL_MS + 3000,
            returnResponse: true,
            ...DIRECT_QUICK_INGEST_TRANSPORT,
          });
          if (response?.ok) {
            const { status } = readIngestJobProgress(response.data);
            if (!ACTIVE_INGEST_JOB_STATUSES.has(status) && !TERMINAL_INGEST_JOB_STATUSES.has(status)) {
              reportStatusUnavailable(true);
              return invalidStatusResponse;
            }
          }
          reportStatusUnavailable(!response?.ok);
          return response;
        } catch (error) {
          assertQuickIngestRequestCurrent(input, error);
          reportStatusUnavailable(true);
          const status = (error as { status?: number } | null)?.status;
          return status === undefined ? undefined : { ok: false, status };
        }
      },
      onRequestError: (response) => response === invalidStatusResponse ||
        (response?.status !== undefined && !isTransientIngestStatusRead(response.status))
          ? { terminalStatus: "interrupted", error: INGEST_JOB_STATUS_INTERRUPTED }
          : undefined,
      onProgress: (progress) => input.onJobProgress?.(itemId, progress),
      isCancelled: () => input.isJobCancellationConfirmed?.(itemId) === true,
      onCancel: () => undefined,
    });
    if (result.terminalStatus === "interrupted") interruptedJobIds.add(jobId);
    else if (statusUnavailable) reportStatusUnavailable(false);
    return result;
  };

  try {
    if (directSessionId) directRuns.set(directSessionId, { uploads, batchIds });

    if (shouldStoreRemote) {
      try {
        conferencePlan = await createPlannedConferenceCollection(input, entries);
      } catch (error) {
        assertQuickIngestRequestCurrent(input, error);
        if (input.conferenceRetry) throw error;
        console.warn("[tldw] Conference collection planning failed", error);
      }
    }

    for (const entry of entries) {
      assertQuickIngestRequestCurrent(input);
      const url = String(entry?.url || "").trim();
      if (!url) continue;

      const resolvedType = resolveIngestUrlType(url, entry.type);

      const plannedConferenceItem =
        conferencePlan?.itemsByEntryId.get(entry.id);
      const duplicatePolicyResolution = resolveConferenceDuplicatePolicy(
        entry.playlist?.duplicateStatus,
        entry.conferenceOverride?.duplicatePolicy,
        entry.processAgain,
      );
      let jobSubmitted = false;
      let localProcessingAttempted = false;
      let latestJobId: number | undefined;
      let data: unknown;
      let resultOutcome: QuickIngestBatchResult["outcome"];
      let resultMessage: string | undefined;
      let resultError: string | undefined;
      let resultMediaId: string | number | null = null;
      let resultPersisted = false;
      const recordEntryResult = () => {
        recordResult({
          id: entry.id,
          status: resultError ? "error" : "ok",
          outcome: resultError ? "failed" : resultOutcome,
          url,
          type: resolvedType,
          data,
          error: resultError,
          message: resultMessage,
          persisted: resultPersisted,
          mediaId: resultMediaId,
          collectionItemId: plannedConferenceItem?.itemId ?? null,
          retryAttempt: plannedConferenceItem ? plannedConferenceItem.retryAttempt ?? 0 : null,
          idempotencyKey: plannedConferenceItem?.idempotencyKey ?? null,
        });
      };
      const recordEntryFailure = async (error: unknown) => {
        assertQuickIngestRequestCurrent(input, error);
        const outcome =
          jobSubmitted || localProcessingAttempted ? "failed" : "submit_failed";
        await patchConferenceCollectionItem(input, plannedConferenceItem, {
          status: outcome,
          latest_job_id:
            typeof latestJobId === "number" ? String(latestJobId) : undefined,
          error_summary:
            error instanceof Error
              ? error.message
              : String(error || "Request failed"),
        });
        recordResult({
          id: entry.id,
          status: "error",
          outcome,
          url,
          type: resolvedType,
          error:
            error instanceof Error
              ? error.message
              : String(error || "Request failed"),
          collectionItemId: plannedConferenceItem?.itemId ?? null,
          retryAttempt: plannedConferenceItem ? plannedConferenceItem.retryAttempt ?? 0 : null,
          idempotencyKey: plannedConferenceItem?.idempotencyKey ?? null,
        });
      };
      try {
        if (shouldStoreRemote && !duplicatePolicyResolution.shouldSubmitJob) {
          if (!plannedConferenceItem && hasUsableConferenceMetadata(input.conferenceBatchMetadata)) {
            throw new Error("Could not add this source to the collection. Its existing Media record was preserved.");
          }
          await patchConferenceCollectionItem(input, plannedConferenceItem, {
            status: duplicatePolicyResolution.plannedStatus,
            retry_count: 0,
          });
          recordResult({
            id: entry.id,
            status: "ok",
            outcome: "skipped",
            url,
            type: resolvedType,
            collectionItemId: plannedConferenceItem?.itemId,
            retryAttempt: plannedConferenceItem?.retryAttempt ?? 0,
            idempotencyKey: plannedConferenceItem?.idempotencyKey ?? null,
            message: DUPLICATE_SKIP_MESSAGE,
            persisted: false,
          });
          continue;
        }

        if (uploads.signal.aborted) {
          const reason = "Stopped before submission.";
          await patchConferenceCollectionItem(input, plannedConferenceItem, {
            status: "cancelled", error_summary: reason,
          });
          recordResult({
            id: entry.id, url, type: resolvedType, status: "error",
            outcome: "cancelled", error: reason,
            collectionItemId: plannedConferenceItem?.itemId,
            retryAttempt: plannedConferenceItem?.retryAttempt,
            idempotencyKey: plannedConferenceItem?.idempotencyKey,
          });
          continue;
        }

        if (resolvedType === "html") {
          localProcessingAttempted = true;
          data = await processWebScrape(input, {
            url,
            entry,
            common: input.common,
            advancedValues: input.advancedValues,
            chunkingTemplateName: input.chunkingTemplateName,
            autoApplyTemplate: input.autoApplyTemplate,
            persist: shouldStoreRemote,
          });
          const htmlDuplicate = completedIngestJobIndicatesSkipped(data);
          const htmlFailure = completedIngestJobIndicatesFailure(data);
          if (htmlDuplicate) {
            resultOutcome = "skipped";
            resultMessage = DUPLICATE_SKIP_MESSAGE;
          } else if (htmlFailure) {
            resultError =
              extractCompletedIngestJobError(data) || "Web scraping failed";
          }
          resultMediaId = shouldStoreRemote ? extractWebScrapeMediaId(data) : null;
          resultPersisted =
            shouldStoreRemote &&
            !resultError &&
            webScrapeResponseIndicatesPersisted(data, resultMediaId);
          await patchConferenceCollectionItem(input, plannedConferenceItem, {
            status: getConferenceTerminalStatus(htmlDuplicate, Boolean(resultError)),
            media_id: resultMediaId,
            error_summary: resultError,
          });
        } else if (shouldStoreRemote) {
          const fields = buildFields({
            rawType: resolvedType,
            entry,
            defaults:
              entry?.defaults && typeof entry.defaults === "object"
                ? entry.defaults
                : fileDefaults,
            common: input.common,
            advancedValues: input.advancedValues,
            chunkingTemplateName: input.chunkingTemplateName,
            autoApplyTemplate: input.autoApplyTemplate,
          });
          if (duplicatePolicyResolution.forceOverwrite) {
            fields.overwrite_existing = true;
          }
          applyPlannedConferenceFields(fields, plannedConferenceItem);
          fields.urls = [url];
          try {
            const submitData = await scopedQuickIngestUpload<any>(input, {
              path: "/api/v1/media/ingest/jobs",
              method: "POST",
              fields: serializeUploadFields(fields),
              timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
              ...DIRECT_QUICK_INGEST_TRANSPORT,
            });
            const { batchId, jobIds } = requireSubmittedIngestJobs(submitData);
            const firstJobId = jobIds[0];
            jobSubmitted = true;
            latestJobId = firstJobId;
            await trackAcceptedBatch({
              mode: "webui-direct",
              sessionId: directSessionId,
              batchId,
              batchIds: [batchId],
              collectionId: plannedConferenceItem
                ? String(plannedConferenceItem.collectionId)
                : undefined,
              plannedItemIds: plannedConferenceItem
                ? [String(plannedConferenceItem.itemId)]
                : undefined,
              jobIds,
              submittedItemIds: [entry.id],
              itemIds: [entry.id],
              jobIdToItemId: buildJobIdToItemId(jobIds, entry.id),
              jobIdToCollectionItemId: buildJobIdToCollectionItemId(
                jobIds,
                plannedConferenceItem,
              ),
              durableMode: plannedConferenceItem ? "durable_collection" : undefined,
              startedAt: Date.now(),
            });
            await patchConferenceCollectionItem(input, plannedConferenceItem, {
              status: "processing",
              retry_count: plannedConferenceItem?.retryAttempt ?? 0,
              latest_job_id: String(firstJobId),
            });
            const pollEntry = async () => {
              const pollResult = await pollIngestJobStatus(firstJobId, entry.id);
              if (pollResult.terminalStatus === "interrupted") return false;
              if (pollResult.terminalStatus === "cancelled") {
                await patchConferenceCollectionItem(input, plannedConferenceItem, {
                  status: "cancelled",
                  latest_job_id: String(firstJobId),
                  error_summary: pollResult.error,
                });
                recordResult({
                  id: entry.id,
                  status: "error",
                  outcome: "cancelled",
                  url,
                  type: resolvedType,
                  error: pollResult.error,
                  collectionItemId: plannedConferenceItem?.itemId ?? null,
                  retryAttempt: plannedConferenceItem?.retryAttempt ?? null,
                  idempotencyKey: plannedConferenceItem?.idempotencyKey ?? null,
                });
                return false;
              }
              if (pollResult.terminalStatus !== "completed") {
                throw new Error(String(pollResult.error || "Ingest failed"));
              }
              data = pollResult.data;
              resultMediaId = extractCompletedIngestJobMediaId(pollResult.data);
              const completedDuplicate = completedIngestJobIndicatesSkipped(
                pollResult.data,
              );
              if (completedDuplicate) {
                resultOutcome = "skipped";
                resultMessage = DUPLICATE_SKIP_MESSAGE;
              } else if (completedIngestJobIndicatesFailure(pollResult.data)) {
                resultError =
                  extractCompletedIngestJobError(pollResult.data) || "Ingest failed";
              }
              await patchConferenceCollectionItem(input, plannedConferenceItem, {
                status: getConferenceTerminalStatus(
                  completedDuplicate,
                  Boolean(resultError),
                ),
                latest_job_id: String(latestJobId),
                media_id: resultMediaId,
                error_summary: resultError,
              });
              return true;
            };
            monitorAcceptedJob(pollEntry().then((completed) => {
              if (completed) recordEntryResult();
            }).catch(recordEntryFailure));
            continue;
          } catch (error) {
      assertQuickIngestRequestCurrent(input, error);
            if (uploads.signal.aborted || !shouldFallbackToPersistentAdd(error)) {
              throw error;
            }
            await patchConferenceCollectionItem(input, plannedConferenceItem, {
              status: jobSubmitted ? "failed" : "submit_failed",
              latest_job_id:
                typeof latestJobId === "number" ? String(latestJobId) : undefined,
              error_summary:
                error instanceof Error ? error.message : String(error || "Submit failed"),
            });
            if (typeof latestJobId === "number") {
              fields.media_ingest_job_id = String(latestJobId);
            }
            localProcessingAttempted = true;
            data = await submitPersistentAdd(input, { fields });
            resultMediaId = extractCompletedIngestJobMediaId(data);
            const fallbackDuplicate = completedIngestJobIndicatesSkipped(data);
            if (fallbackDuplicate) {
              resultOutcome = "skipped";
              resultMessage = DUPLICATE_SKIP_MESSAGE;
            } else if (completedIngestJobIndicatesFailure(data)) {
              resultError =
                extractCompletedIngestJobError(data) || "Ingest failed";
            }
            await patchConferenceCollectionItem(input, plannedConferenceItem, {
              status: getConferenceTerminalStatus(
                fallbackDuplicate,
                Boolean(resultError),
              ),
              latest_job_id:
                typeof latestJobId === "number" ? String(latestJobId) : undefined,
              media_id: resultMediaId,
              error_summary: resultError,
            });
          }
        } else {
          const fields = buildFields({
            rawType: resolvedType,
            entry,
            defaults:
              entry?.defaults && typeof entry.defaults === "object"
                ? entry.defaults
                : fileDefaults,
            common: input.common,
            advancedValues: input.advancedValues,
            chunkingTemplateName: input.chunkingTemplateName,
            autoApplyTemplate: input.autoApplyTemplate,
            persist: false,
          });
          fields.urls = [url];
          localProcessingAttempted = true;
          data = await scopedQuickIngestUpload<any>(input, {
            path: getProcessPathForType(resolvedType),
            method: "POST",
            fields: serializeUploadFields(fields),
            timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
            ...DIRECT_QUICK_INGEST_TRANSPORT,
          });
          if (completedIngestJobIndicatesFailure(data)) {
            resultError =
              extractCompletedIngestJobError(data) || "Ingest failed";
          } else if (completedIngestJobIndicatesSkipped(data)) {
            resultOutcome = "skipped";
            resultMessage = DUPLICATE_SKIP_MESSAGE;
          }
        }

        recordEntryResult();
      } catch (error) {
        await recordEntryFailure(error);
      }
    }

    for (const file of files) {
      assertQuickIngestRequestCurrent(input);
      const id = String(file?.id || crypto.randomUUID());
      const directFile = file?.directFile;
      const fileName = String(directFile?.name || file?.name || "upload");
      const mimeType = directFile ? directFile.type : file?.type;
      const mediaType = inferUploadMediaTypeFromFile(fileName, mimeType);
      const bytes = file?.data;
      const requestContext = directFile ? runContext : input;
      const timeoutMs = getNativeFileTimeout(directFile);
      if (uploads.signal.aborted) {
        recordResult({
          id, fileName, type: mediaType, status: "error",
          outcome: "cancelled", error: "Stopped before submission.",
        });
        continue;
      }
      let jobAcknowledged = false;
      const recordFileFailure = (error: unknown) => {
        assertQuickIngestRequestCurrent(input, error);
        recordResult({
          id,
          status: "error",
          fileName,
          type: "file",
          error: error instanceof Error ? error.message : String(error || "Upload failed"),
        });
      };

      try {
        const size = directFile?.size ?? (Array.isArray(bytes) ? bytes.length : bytes?.byteLength ?? 0);
        const sizeError = getQuickIngestFileSizeError({ name: fileName, type: mimeType, size });
        if (sizeError) throw new Error(sizeError);

        const fields = buildFields({
          rawType: mediaType,
          defaults:
            file?.defaults && typeof file.defaults === "object"
              ? file.defaults
              : fileDefaults,
          common: input.common,
          advancedValues: input.advancedValues,
          chunkingTemplateName: input.chunkingTemplateName,
          autoApplyTemplate: input.autoApplyTemplate,
          persist: shouldStoreRemote,
        });
        const upload = directFile
          ? { directFile }
          : { file: { name: fileName, type: mimeType || "application/octet-stream", data: bytes || [] } };
        if (shouldStoreRemote) {
          try {
            const submitData = await scopedQuickIngestUpload<any>(requestContext, {
              path: "/api/v1/media/ingest/jobs",
              method: "POST",
              fields: serializeUploadFields(fields),
              ...upload,
              fileFieldName: "files",
              timeoutMs,
              ...DIRECT_QUICK_INGEST_TRANSPORT,
            });
            const { batchId, jobIds } = requireSubmittedIngestJobs(submitData);
            jobAcknowledged = true;
            await trackAcceptedBatch({
              mode: "webui-direct",
              sessionId: directSessionId,
              batchId,
              batchIds: [batchId],
              jobIds,
              submittedItemIds: [id],
              itemIds: [id],
              jobIdToItemId: buildJobIdToItemId(jobIds, id),
              startedAt: Date.now(),
            });
            const firstJobId = jobIds[0];
            const pollFile = async () => {
              const pollResult = await pollIngestJobStatus(firstJobId, id);
              if (pollResult.terminalStatus === "interrupted") return;
              if (pollResult.terminalStatus === "cancelled") {
                recordResult({
                  id,
                  status: "error",
                  outcome: "cancelled",
                  fileName,
                  type: mediaType,
                  error: pollResult.error,
                });
                return;
              }
              if (pollResult.terminalStatus !== "completed") {
                throw new Error(String(pollResult.error || "Upload failed"));
              }
              const isDuplicate = completedIngestJobIndicatesSkipped(pollResult.data);
              recordResult({
                id,
                status: "ok",
                outcome: isDuplicate ? ("skipped" as const) : undefined,
                fileName,
                type: mediaType,
                data: pollResult.data,
                message: isDuplicate ? DUPLICATE_SKIP_MESSAGE : undefined,
                mediaId: extractCompletedIngestJobMediaId(pollResult.data),
                persisted: shouldStoreRemote && shouldKeepOriginalFile(mediaType),
              });
            };
            monitorAcceptedJob(pollFile().catch(recordFileFailure));
          } catch (error) {
      assertQuickIngestRequestCurrent(input, error);
            if (uploads.signal.aborted || !shouldFallbackToPersistentAdd(error)) {
              throw error;
            }
            const data = await submitPersistentAdd(requestContext, {
              fields,
              ...upload,
            });
            const fallbackDuplicate = completedIngestJobIndicatesSkipped(data);
            recordResult({
              id,
              status: "ok",
              outcome: fallbackDuplicate ? ("skipped" as const) : undefined,
              fileName,
              type: mediaType,
              data,
              message: fallbackDuplicate ? DUPLICATE_SKIP_MESSAGE : undefined,
              mediaId: extractCompletedIngestJobMediaId(data),
              persisted: shouldStoreRemote && shouldKeepOriginalFile(mediaType),
            });
          }
          continue;
        }

        const data = await scopedQuickIngestUpload<any>(requestContext, {
          path: getProcessPathForType(mediaType),
          method: "POST",
          fields: serializeUploadFields(fields),
          ...upload,
          fileFieldName: "files",
          timeoutMs,
          ...DIRECT_QUICK_INGEST_TRANSPORT,
        });

        const directDuplicate = isDbMessageDuplicate(data);
        recordResult({
          id,
          status: "ok",
          outcome: directDuplicate ? ("skipped" as const) : undefined,
          fileName,
          type: mediaType,
          data,
          message: directDuplicate ? DUPLICATE_SKIP_MESSAGE : undefined,
          mediaId: null,
          persisted: false,
        });
      } catch (error) {
        assertQuickIngestRequestCurrent(input, error);
        if (directFile && !jobAcknowledged && uploads.signal.aborted) {
          // A broken upload connection cannot establish whether the server accepted it.
          interruptedItemIds.push(id);
          continue;
        }
        recordFileFailure(error);
      }
    }

    for (const failure of await Promise.all(acceptedMonitors)) {
      if (failure) throw failure.error;
    }
    assertQuickIngestRequestCurrent(input);
    return {
      ok: true,
      results: out.sort((a, b) => (inputOrder.get(a.id) ?? inputOrder.size) - (inputOrder.get(b.id) ?? inputOrder.size)),
      ...(interruptedItemIds.length ? { interruptedItemIds } : {}),
      ...(interruptedJobIds.size ? { interruptedJobIds: [...interruptedJobIds] } : {}),
    };
  } finally {
    uploads.abort();
    if (directSessionId) directRuns.delete(directSessionId);
  }
};

const runtimePayload = (input: QuickIngestBatchInput) => {
  const {
    signal: _signal,
    assertCurrent: _assertCurrent,
    onTrackingMetadata: _tracking,
    onJobProgress: _progress,
    onJobStatusUnavailable: _statusUnavailable,
    onStopUnconfirmed: _stopUnconfirmed,
    onItemResult: _result,
    isJobCancellationConfirmed: _cancelled,
    ...payload
  } = input;
  return payload;
};

export const submitQuickIngestBatch = async (
  input: QuickIngestBatchInput,
): Promise<QuickIngestBatchResponse> => {
  assertQuickIngestRequestCurrent(input);
  if (
    !input.conferenceRetry &&
    !isDirectQuickIngestSessionId(input?.__quickIngestSessionId) &&
    !hasNativeFile(input.files) &&
    (await canUseExtensionMessagingRuntime())
  ) {
    try {
      const result =
        await (assertQuickIngestRequestCurrent(input), sendExtensionMessageWithTimeout<QuickIngestBatchResponse>({
          type: "tldw:quick-ingest-batch",
          payload: runtimePayload(input),
        }));
      return result;
    } catch (error) {
      assertQuickIngestRequestCurrent(input, error);
      // Fall through to the direct path when runtime messaging is unavailable
      // even though the extension context still exists.
    }
  }

  return await runDirectQuickIngestBatch(input);
};

export const startQuickIngestSession = async (
  input: QuickIngestBatchInput,
): Promise<QuickIngestStartAck> => {
  assertQuickIngestRequestCurrent(input);
  // Native Files and collection retries require the direct executor.
  if (
    !input.conferenceRetry &&
    !hasNativeFile(input.files) &&
    !shouldPreferDirectQuickIngestSession() &&
    (await canUseExtensionMessagingRuntime())
  ) {
    try {
      assertQuickIngestRequestCurrent(input);
      return await sendExtensionMessageWithTimeout<QuickIngestStartAck>({
        type: "tldw:quick-ingest/start",
        payload: runtimePayload(input),
      });
    } catch (error) {
      assertQuickIngestRequestCurrent(input, error);
      // Fall through to the direct session ack when the runtime exists
      // but message delivery is unhealthy.
    }
  }

  // Direct runtimes currently run ingest synchronously. Return a local ack
  // so session-native callers can still establish a run identity.
  return {
    ok: true,
    sessionId: `${DIRECT_QUICK_INGEST_SESSION_PREFIX}${Date.now()}-${buildDirectSessionSuffix()}`,
  };
};

export const cancelQuickIngestSession = async (
  input: QuickIngestCancelInput,
): Promise<QuickIngestCancelResponse> => {
  assertQuickIngestRequestCurrent(input);
  const sessionId = String(input?.sessionId || "").trim();
  const tracking = input?.tracking;
  if (!sessionId) {
    return { ok: false, error: "Missing session id." };
  }

  if (
    !isDirectQuickIngestSessionId(sessionId) &&
    (await canUseExtensionMessagingRuntime())
  ) {
    try {
      assertQuickIngestRequestCurrent(input);
      return await sendExtensionMessageWithTimeout<QuickIngestCancelResponse>({
        type: "tldw:quick-ingest/cancel",
        payload: {
          requestScope: input.requestScope,
          sessionId,
          reason: input?.reason,
        },
      });
    } catch (error) {
      assertQuickIngestRequestCurrent(input, error);
      // Fall through to the direct cancellation path when runtime messaging
      // stops responding in packaged extension contexts.
    }
  }

  const run = directRuns.get(sessionId);
  run?.uploads.abort();
  const batchIds = normalizeBatchIds([
    ...(run?.batchIds || []),
    ...(input.batchIds || []),
    tracking?.batchId || "",
    ...(tracking?.batchIds || []),
  ]);
  const acknowledgements = await Promise.all(
    batchIds.map((batchId) => requestBatchStop(input, batchId, input.reason)),
  );
  // A batch acknowledgement carries counts, never individual job outcomes.
  return { ok: acknowledgements.length > 0 && acknowledgements.every(Boolean) };
};

export const __resetQuickIngestRuntimeHealthForTests = (): void => {
  quickIngestRuntimeMessagingUsable = null;
  lastQuickIngestRuntimeHealthCheckAt = 0;
};
