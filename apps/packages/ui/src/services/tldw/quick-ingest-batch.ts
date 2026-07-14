import { browser } from "wxt/browser";
import { bgRequest, bgUpload } from "@/services/background-proxy";
import {
  getProcessPathForType,
  inferIngestTypeFromUrl,
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
  createIngestJobsTracker,
  pollSingleIngestJob,
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
  type MediaCollectionItemStatus,
} from "@/services/tldw/conference-collections";
import { mediaMethods } from "@/services/tldw/domains/media";
import {
  PlaylistIngestPublicError,
  cancelRun,
  createRun,
  submitPendingChunks,
  type PlaylistIngestRunCreateRequest,
  type PlaylistIngestRunCreateResult,
  type PlaylistReviewRequiredRecoveryItem,
} from "@/services/tldw/playlist-ingest";

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
  defaults?: TypeDefaults;
  conferenceOverride?: ConferenceItemMetadataOverride;
};

type QuickIngestBatchInput = {
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
  conferenceItemMetadata?: Record<
    string,
    {
      playlist?: PlaylistQueueMetadata;
      conferenceOverride?: ConferenceItemMetadataOverride;
    }
  >;
  chunkingTemplateName?: string;
  autoApplyTemplate?: boolean;
  pendingRunRequest?: PlaylistIngestRunCreateRequest | null;
  __quickIngestSessionId?: string;
  __quickIngestShouldStop?: () => boolean;
  onTrackingMetadata?: (
    tracking: PersistedQuickIngestTracking,
  ) => void | Promise<void>;
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
  accepted?: boolean;
  submissionBlocked?: boolean;
  submissionCleanupFailed?: boolean;
  runId?: string;
  retryAfterMs?: number | null;
  unsentOccurrenceIds?: string[];
  error?: string;
  results?: QuickIngestBatchResult[];
  reviewRequired?: PlaylistReviewRequiredRecoveryItem[];
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
  indeterminate?: boolean;
  error?: string;
};

export type QuickIngestCancelInput = {
  sessionId: string;
  reason?: string;
  batchIds?: string[];
  tracking?: PersistedQuickIngestTracking;
};

export type QuickIngestCancelResponse = {
  ok: boolean;
  error?: string;
};

export type QuickIngestSessionReplayResponse = {
  ok: boolean;
  active?: boolean;
  event?: {
    type: string;
    payload: Record<string, unknown>;
  } | null;
  replayAck?: {
    runId: string;
    generation: string;
  };
  error?: string;
};

export type QuickIngestReplayAckInput = {
  sessionId: string;
  runId: string;
  generation: string;
};

const EXTENSION_TIMEOUT_MS = 10_000;
const QUICK_INGEST_RUNTIME_PING_TIMEOUT_MS = 400;
const QUICK_INGEST_RUNTIME_HEALTH_TTL_MS = 30_000;
const DIRECT_INGEST_TIMEOUT_MS = 5 * 60 * 1000;
const DIRECT_REMOTE_POLL_INTERVAL_MS = 1_200;
type DirectQuickIngestTracker = ReturnType<
  typeof createIngestJobsTracker<{ sourceId: string }>
>;

const directQuickIngestSessionTrackers = new Map<
  string,
  DirectQuickIngestTracker
>();
const directQuickIngestCancelledSessions = new Set<string>();
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

const createOpaqueExtensionIdentity = (prefix: "qi" | "qia"): string => {
  try {
    if (typeof globalThis.crypto?.randomUUID === "function") {
      return `${prefix}-${globalThis.crypto.randomUUID()}`;
    }
  } catch {
    // Fall back to the existing bounded local suffix below.
  }
  return `${prefix}-${Date.now()}-${buildDirectSessionSuffix()}`;
};

const createExtensionQuickIngestSessionId = (): string =>
  createOpaqueExtensionIdentity("qi");

const createExtensionQuickIngestAttemptToken = (): string =>
  createOpaqueExtensionIdentity("qia");

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

const ensureDirectSessionTracker = (
  sessionId: string | undefined,
): DirectQuickIngestTracker | undefined => {
  const normalizedSessionId = String(sessionId || "").trim();
  if (!normalizedSessionId) return undefined;
  const existing = directQuickIngestSessionTrackers.get(normalizedSessionId);
  if (existing) return existing;
  const created = createIngestJobsTracker<{ sourceId: string }>();
  directQuickIngestSessionTrackers.set(normalizedSessionId, created);
  return created;
};

const clearDirectSessionTracking = (sessionId: string | undefined) => {
  const normalizedSessionId = String(sessionId || "").trim();
  if (!normalizedSessionId) return;
  directQuickIngestSessionTrackers.delete(normalizedSessionId);
  directQuickIngestCancelledSessions.delete(normalizedSessionId);
};

const isDirectSessionCancelled = (sessionId: string | undefined) => {
  const normalizedSessionId = String(sessionId || "").trim();
  if (!normalizedSessionId) return false;
  return directQuickIngestCancelledSessions.has(normalizedSessionId);
};

const cancelDirectSessionBatches = async (
  sessionId: string | undefined,
  reason: string,
): Promise<void> => {
  const normalizedSessionId = String(sessionId || "").trim();
  if (!normalizedSessionId) return;
  const tracker = directQuickIngestSessionTrackers.get(normalizedSessionId);
  if (!tracker) return;

  await tracker.cancelTrackedBatches(async (batchId) => {
    await bgRequest<any>({
      path: `/api/v1/media/ingest/jobs/cancel?batch_id=${encodeURIComponent(
        batchId,
      )}&reason=${encodeURIComponent(reason || "user_cancelled")}`,
      method: "POST",
      timeoutMs: 10_000,
      returnResponse: true,
      ...DIRECT_QUICK_INGEST_TRANSPORT,
    }).catch(() => {
      // best effort cancellation
    });
  });
};

const hasExtensionMessagingRuntime = (): boolean =>
  Boolean(browser?.runtime?.sendMessage && browser?.runtime?.id);

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

const submitPersistentAdd = async ({
  fields,
  file,
}: {
  fields: Record<string, any>;
  file?: {
    name: string;
    type: string;
    data: number[] | Uint8Array | ArrayBuffer;
  };
}): Promise<any> =>
  normalizePersistentAddResponse(
    await bgUpload<any>({
      path: "/api/v1/media/add",
      method: "POST",
      fields: serializeUploadFields(fields),
      file,
      fileFieldName: file ? "files" : undefined,
      timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
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

const mediaTypeForProcessingSource = (sourceKind: string | null): string => {
  const normalized = String(sourceKind || "").toLowerCase();
  if (normalized.includes("audio")) return "audio";
  if (normalized.includes("pdf")) return "pdf";
  if (normalized.includes("document") || normalized.includes("web")) {
    return "document";
  }
  return "video";
};

const compactRunMetadata = (
  value: Record<string, unknown>,
): Record<string, unknown> =>
  Object.fromEntries(
    Object.entries(value).filter(([, entry]) => {
      if (entry === undefined || entry === null || entry === "") return false;
      return !Array.isArray(entry) || entry.length > 0;
    }),
  );

const buildVersion2RunRequest = (
  input: QuickIngestBatchInput,
  pendingRunRequest: PlaylistIngestRunCreateRequest,
): PlaylistIngestRunCreateRequest => {
  const normalizedProcessingOptions = buildFields({
    rawType: "auto",
    common: input.common,
    advancedValues: input.advancedValues,
    chunkingTemplateName: input.chunkingTemplateName,
    autoApplyTemplate: input.autoApplyTemplate,
    persist: input.storeRemote && !input.processOnly,
  });
  delete normalizedProcessingOptions.media_type;
  delete normalizedProcessingOptions.keep_original_file;

  const pendingOccurrenceIds = new Set(
    pendingRunRequest.inputs.map((runInput) => runInput.occurrenceId),
  );
  const conferenceSummaries = Object.entries(
    input.conferenceItemMetadata || {},
  ).flatMap(([occurrenceId, metadata]) => {
    if (!pendingOccurrenceIds.has(occurrenceId)) return [];
    const playlist = metadata.playlist;
    const override = metadata.conferenceOverride;
    if (!playlist && !override) return [];
    return [
      compactRunMetadata({
        occurrence_id: occurrenceId,
        playlist: playlist
          ? compactRunMetadata({
              playlist_id: playlist.playlistId,
              playlist_title: playlist.playlistTitle,
              ordinal: playlist.ordinal,
              title: playlist.title,
              channel_or_uploader: playlist.channelOrUploader,
              duration_seconds: playlist.durationSeconds,
              normalized_source_id: playlist.normalizedSourceId,
              duplicate_status: playlist.duplicateStatus,
            })
          : undefined,
        conference_override: override
          ? compactRunMetadata({
              title: override.title,
              speaker: override.speaker,
              talk_date: override.talkDate,
              track: override.track,
              tags: Array.from(
                new Set(
                  (override.tags || [])
                    .map((tag) => String(tag).trim())
                    .filter(Boolean),
                ),
              ),
              duplicate_policy: override.duplicatePolicy,
              selected: override.selected,
            })
          : undefined,
      }),
    ];
  });

  const collection = input.conferenceBatchMetadata
    ? buildConferenceCollectionCreatePayload(input.conferenceBatchMetadata)
    : null;
  const processingOptions = {
    ...(pendingRunRequest.processingOptions || {}),
    ...normalizedProcessingOptions,
  };
  const playlistSummaries = [
    ...(pendingRunRequest.playlistSummaries || []),
    ...conferenceSummaries,
  ];

  return {
    ...pendingRunRequest,
    ...(Object.keys(processingOptions).length > 0 ? { processingOptions } : {}),
    ...(playlistSummaries.length > 0 ? { playlistSummaries } : {}),
    ...(collection
      ? {
          newCollection: {
            name: collection.name,
            ...(collection.source_url
              ? { sourceUrl: collection.source_url }
              : {}),
            defaultTags: collection.default_tags,
          },
        }
      : {}),
  };
};

const runVersion2QuickIngestBatch = async (
  input: QuickIngestBatchInput,
): Promise<QuickIngestBatchResponse> => {
  const pendingRunRequest = input.pendingRunRequest;
  if (!pendingRunRequest) {
    return { ok: false, error: "Missing playlist ingest run request." };
  }

  const sessionId = String(input.__quickIngestSessionId || "").trim();
  const submittedItemIds = pendingRunRequest.inputs.map(
    (runInput) => runInput.occurrenceId,
  );
  const startedAt = Date.now();
  const isCancelled = () =>
    Boolean(
      input.__quickIngestShouldStop?.() || isDirectSessionCancelled(sessionId),
    );
  const publishTracking = async (
    submissionState: NonNullable<
      PersistedQuickIngestTracking["submissionState"]
    >,
    patch: Partial<PersistedQuickIngestTracking> = {},
  ): Promise<void> => {
    await input.onTrackingMetadata?.({
      mode: isDirectQuickIngestSessionId(sessionId)
        ? "webui-direct"
        : "extension-runtime",
      submissionState,
      sessionId: sessionId || undefined,
      submissionOccurrenceIds: submittedItemIds,
      ...(submissionState === "creating_run"
        ? {}
        : { submittedItemIds, itemIds: submittedItemIds }),
      startedAt,
      ...patch,
    });
  };

  await publishTracking("creating_run");

  let run: PlaylistIngestRunCreateResult;
  try {
    run = await createRun(
      mediaMethods,
      buildVersion2RunRequest(input, pendingRunRequest),
      DIRECT_QUICK_INGEST_TRANSPORT,
    );
  } catch (error) {
    if (
      error instanceof PlaylistIngestPublicError &&
      error.recovery?.kind === "reviewRequired"
    ) {
      return {
        ok: false,
        error: error.message,
        reviewRequired: error.recovery.items,
      };
    }
    throw error;
  }

  await publishTracking("run_created", { runId: run.runId });

  if (isCancelled()) {
    let cleanupFailed = false;
    try {
      await cancelRun(
        mediaMethods,
        run.runId,
        { reason: "user_cancelled" },
        DIRECT_QUICK_INGEST_TRANSPORT,
      );
    } catch {
      cleanupFailed = true;
    }
    await publishTracking(cleanupFailed ? "cleanup_required" : "acknowledged", {
      runId: run.runId,
    });
    clearDirectSessionTracking(sessionId);
    return {
      ok: false,
      accepted: false,
      submissionBlocked: true,
      ...(cleanupFailed ? { submissionCleanupFailed: true } : {}),
      runId: run.runId,
      retryAfterMs: null,
      unsentOccurrenceIds: run.processingOccurrences.map(
        (occurrence) => occurrence.occurrenceId,
      ),
      error: cleanupFailed
        ? "Cancellation was requested, but the server did not confirm run cancellation."
        : "Cancelled by user.",
    };
  }

  const entriesByOccurrenceId = new Map(
    input.entries.map((entry) => [entry.id, entry] as const),
  );
  const filePayloadsByOccurrenceId = new Map(
    input.files.flatMap((file) => {
      const occurrenceId = String(file.id || "").trim();
      return occurrenceId ? [[occurrenceId, file] as const] : [];
    }),
  );
  const filesByOccurrenceId = Object.fromEntries(
    input.files.flatMap((file) => {
      const occurrenceId = String(file.id || "").trim();
      if (!occurrenceId) return [];
      return [
        [
          occurrenceId,
          {
            name: String(file.name || "upload"),
            type: String(file.type || "application/octet-stream"),
            data: file.data || [],
          },
        ] as const,
      ];
    }),
  );
  const baseFieldsByOccurrenceId = Object.fromEntries(
    run.processingOccurrences.map((occurrence) => {
      const entry = entriesByOccurrenceId.get(occurrence.occurrenceId);
      const file = filePayloadsByOccurrenceId.get(occurrence.occurrenceId);
      const rawType = entry?.type
        ? entry.type
        : file
          ? inferUploadMediaTypeFromFile(file.name, file.type)
          : mediaTypeForProcessingSource(occurrence.sourceKind);
      return [
        occurrence.occurrenceId,
        buildFields({
          rawType,
          entry,
          defaults:
            entry?.defaults ||
            (file?.defaults && typeof file.defaults === "object"
              ? file.defaults
              : input.fileDefaults),
          common: input.common,
          advancedValues: input.advancedValues,
          chunkingTemplateName: input.chunkingTemplateName,
          autoApplyTemplate: input.autoApplyTemplate,
          persist: input.storeRemote && !input.processOnly,
        }),
      ] as const;
    }),
  );
  const publishSubmittedTracking = async (
    submitted: Awaited<ReturnType<typeof submitPendingChunks>>,
    submissionState: NonNullable<
      PersistedQuickIngestTracking["submissionState"]
    >,
  ): Promise<void> => {
    const accepted = submitted.submissions.filter(
      (submission) => submission.accepted,
    );
    const jobIds = accepted.flatMap((submission) =>
      submission.jobId === null ? [] : [submission.jobId],
    );
    await publishTracking(submissionState, {
      runId: run.runId,
      batchId: submitted.batchIds.at(-1),
      batchIds: submitted.batchIds,
      jobIds,
      jobIdToItemId: Object.fromEntries(
        accepted.flatMap((submission) =>
          submission.jobId === null
            ? []
            : [[String(submission.jobId), submission.occurrenceId]],
        ),
      ),
    });
  };

  const submitted = await submitPendingChunks({
    run,
    baseFields: {},
    baseFieldsByOccurrenceId,
    filesByOccurrenceId,
    shouldStop: isCancelled,
    onProgress: (progress) => publishSubmittedTracking(progress, "submitting"),
    submitChunk: (request) =>
      bgUpload({
        ...request,
        timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
        ...DIRECT_QUICK_INGEST_TRANSPORT,
      }),
  });

  const accepted = submitted.submissions.filter(
    (submission) => submission.accepted,
  );

  let submissionCleanupError: string | null = null;
  if (submitted.stopped && submitted.unsentOccurrenceIds.length > 0) {
    try {
      const userCancelled = isCancelled();
      await cancelRun(
        mediaMethods,
        run.runId,
        userCancelled
          ? { reason: "user_cancelled" }
          : {
              occurrenceIds: submitted.unsentOccurrenceIds,
              reason: "submission_stopped",
            },
        DIRECT_QUICK_INGEST_TRANSPORT,
      );
    } catch (error) {
      submissionCleanupError =
        error instanceof Error
          ? error.message
          : "The server did not confirm occurrence cancellation.";
    }
  }

  await publishSubmittedTracking(
    submitted,
    submissionCleanupError ? "cleanup_required" : "acknowledged",
  );
  clearDirectSessionTracking(sessionId);

  return {
    ok: !submitted.stopped,
    accepted: !submitted.stopped || accepted.length > 0,
    runId: run.runId,
    ...(submitted.stopped
      ? {
          submissionBlocked: true,
          ...(submissionCleanupError
            ? { submissionCleanupFailed: true }
            : {}),
          error:
            submissionCleanupError !== null
              ? `Submission stopped, but the server could not cancel the unsent occurrences. ${submissionCleanupError} Retry cancellation before reconnecting.`
              : submitted.retryAfterMs === null
              ? "Playlist ingest submission stopped before every item was accepted. Try again."
              : `Playlist ingest submission was rate limited. Try again in ${Math.ceil(submitted.retryAfterMs / 1000)} seconds.`,
          retryAfterMs: submitted.retryAfterMs,
          unsentOccurrenceIds: submitted.unsentOccurrenceIds,
        }
      : {}),
  };
};

const processWebScrape = async ({
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
    mode: persist ? "persist" : "ephemeral",
    summarize_checkbox: Boolean(common?.perform_analysis),
    ...normalizedBody,
  };
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

  return await bgRequest<any>({
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
  if (!hasUsableConferenceMetadata(input.conferenceBatchMetadata)) return null;
  const selectedEntries = entries.filter(
    (entry) =>
      String(entry?.url || "").trim() &&
      entry?.conferenceOverride?.selected !== false,
  );
  if (selectedEntries.length === 0) return null;

  const collection = normalizeMediaCollectionResponse(
    await bgRequest<ApiMediaCollection>({
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
      await bgRequest<ApiMediaCollectionItem>({
        path: `/api/v1/media/collections/${encodeURIComponent(String(collection.id))}/items`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: buildConferenceCollectionItemPayload(input.conferenceBatchMetadata, {
          id: entry.id,
          url: entry.url,
          playlist: entry.playlist,
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
  planned: PlannedConferenceCollectionItem | undefined,
  payload: Record<string, unknown>,
): Promise<void> => {
  if (!planned) return;
  await bgRequest<ApiMediaCollectionItem>({
    path: `/api/v1/media/collections/${encodeURIComponent(
      String(planned.collectionId),
    )}/items/${encodeURIComponent(String(planned.itemId))}`,
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: payload,
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
  const shouldStoreRemote = input.storeRemote && !input.processOnly;
  const directSessionId =
    String(input.__quickIngestSessionId || "").trim() || undefined;

  const out: QuickIngestBatchResult[] = [];
  let conferencePlan: PlannedConferenceCollection | null = null;
  if (shouldStoreRemote) {
    try {
      conferencePlan = await createPlannedConferenceCollection(input, entries);
    } catch (error) {
      console.warn("[tldw] Conference collection planning failed", error);
    }
  }

  const pollIngestJobStatus = async (
    jobId: number,
    timeoutMs: number,
  ): Promise<{ ok: boolean; data?: any; error?: string }> => {
    const pollResult = await pollSingleIngestJob({
      jobId,
      timeoutMs,
      pollIntervalMs: DIRECT_REMOTE_POLL_INTERVAL_MS,
      fetchJob: async (trackedJobId) =>
        (await bgRequest<any>({
          path: `/api/v1/media/ingest/jobs/${trackedJobId}`,
          method: "GET",
          timeoutMs: DIRECT_REMOTE_POLL_INTERVAL_MS + 3000,
          returnResponse: true,
          ...DIRECT_QUICK_INGEST_TRANSPORT,
        })) as
          | { ok: boolean; status?: number; data?: any; error?: string }
          | undefined,
      isCancelled: () => isDirectSessionCancelled(directSessionId),
      onCancel: async () => {
        await cancelDirectSessionBatches(directSessionId, "user_cancelled");
      },
    });

    if (pollResult.terminalStatus === "completed") {
      return { ok: true, data: pollResult.data };
    }
    return {
      ok: false,
      error: String(pollResult.error || "Ingest failed"),
      data: pollResult.data,
    };
  };

  try {
    if (directSessionId) {
      directQuickIngestCancelledSessions.delete(directSessionId);
      directQuickIngestSessionTrackers.set(
        directSessionId,
        createIngestJobsTracker<{ sourceId: string }>(),
      );
    }

    for (const entry of entries) {
      if (isDirectSessionCancelled(directSessionId)) {
        break;
      }
      const url = String(entry?.url || "").trim();
      if (!url) continue;

      const explicitType =
        entry?.type && typeof entry.type === "string" ? entry.type : "auto";
      const resolvedType =
        explicitType === "auto" ? inferIngestTypeFromUrl(url) : explicitType;

      const plannedConferenceItem =
        conferencePlan?.itemsByEntryId.get(entry.id);
      const duplicatePolicyResolution = resolveConferenceDuplicatePolicy(
        entry.playlist?.duplicateStatus,
        entry.conferenceOverride?.duplicatePolicy,
      );
      let jobSubmitted = false;
      let localProcessingAttempted = false;
      let latestJobId: number | undefined;
      try {
        if (
          shouldStoreRemote &&
          plannedConferenceItem &&
          !duplicatePolicyResolution.shouldSubmitJob
        ) {
          await patchConferenceCollectionItem(plannedConferenceItem, {
            status: duplicatePolicyResolution.plannedStatus,
            retry_count: 0,
          });
          out.push({
            id: entry.id,
            status: "ok",
            outcome: "skipped",
            url,
            type: resolvedType,
            collectionItemId: plannedConferenceItem.itemId,
            retryAttempt: 0,
            idempotencyKey: plannedConferenceItem.idempotencyKey ?? null,
            message: DUPLICATE_SKIP_MESSAGE,
            persisted: false,
          });
          continue;
        }

        let data: unknown;
        let resultOutcome: QuickIngestBatchResult["outcome"];
        let resultMessage: string | undefined;
        let resultError: string | undefined;
        let resultMediaId: string | number | null = null;
        let resultPersisted = false;
        if (resolvedType === "html") {
          localProcessingAttempted = true;
          data = await processWebScrape({
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
          await patchConferenceCollectionItem(plannedConferenceItem, {
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
            const submitData = await bgUpload<any>({
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
            await patchConferenceCollectionItem(plannedConferenceItem, {
              status: "processing",
              latest_job_id: String(firstJobId),
            });
            const directTracker = ensureDirectSessionTracker(directSessionId);
            directTracker?.trackJobs(batchId, jobIds, { sourceId: entry.id });
            input.onTrackingMetadata?.({
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
            const pollResult = await pollIngestJobStatus(
              firstJobId,
              DIRECT_INGEST_TIMEOUT_MS,
            );
            if (!pollResult.ok) {
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
            await patchConferenceCollectionItem(plannedConferenceItem, {
              status: getConferenceTerminalStatus(
                completedDuplicate,
                Boolean(resultError),
              ),
              latest_job_id: String(latestJobId),
              media_id: resultMediaId,
              error_summary: resultError,
            });
          } catch (error) {
            if (!shouldFallbackToPersistentAdd(error)) {
              throw error;
            }
            await patchConferenceCollectionItem(plannedConferenceItem, {
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
            data = await submitPersistentAdd({ fields });
            resultMediaId = extractCompletedIngestJobMediaId(data);
            const fallbackDuplicate = completedIngestJobIndicatesSkipped(data);
            if (fallbackDuplicate) {
              resultOutcome = "skipped";
              resultMessage = DUPLICATE_SKIP_MESSAGE;
            } else if (completedIngestJobIndicatesFailure(data)) {
              resultError =
                extractCompletedIngestJobError(data) || "Ingest failed";
            }
            await patchConferenceCollectionItem(plannedConferenceItem, {
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
          data = await bgUpload<any>({
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

        out.push({
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
          retryAttempt: plannedConferenceItem ? 0 : null,
          idempotencyKey: plannedConferenceItem?.idempotencyKey ?? null,
        });
      } catch (error) {
        const outcome =
          jobSubmitted || localProcessingAttempted ? "failed" : "submit_failed";
        await patchConferenceCollectionItem(plannedConferenceItem, {
          status: outcome,
          latest_job_id:
            typeof latestJobId === "number" ? String(latestJobId) : undefined,
          error_summary:
            error instanceof Error
              ? error.message
              : String(error || "Request failed"),
        });
        out.push({
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
          retryAttempt: plannedConferenceItem ? 0 : null,
          idempotencyKey: plannedConferenceItem?.idempotencyKey ?? null,
        });
      }
    }

    for (const file of files) {
      if (isDirectSessionCancelled(directSessionId)) {
        break;
      }
      const id = String(file?.id || crypto.randomUUID());
      const fileName = String(file?.name || "upload");
      const mediaType = inferUploadMediaTypeFromFile(fileName, file?.type);

      try {
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
        const uploadFile = {
          name: fileName,
          type: file?.type || "application/octet-stream",
          data:
            (file?.data as number[] | Uint8Array | ArrayBuffer | undefined) ||
            [],
        };
        if (shouldStoreRemote) {
          try {
            const submitData = await bgUpload<any>({
              path: "/api/v1/media/ingest/jobs",
              method: "POST",
              fields: serializeUploadFields(fields),
              file: uploadFile,
              fileFieldName: "files",
              timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
              ...DIRECT_QUICK_INGEST_TRANSPORT,
            });
            const { batchId, jobIds } = requireSubmittedIngestJobs(submitData);
            const directTracker = ensureDirectSessionTracker(directSessionId);
            directTracker?.trackJobs(batchId, jobIds, { sourceId: id });
            input.onTrackingMetadata?.({
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
            const pollResult = await pollIngestJobStatus(
              firstJobId,
              DIRECT_INGEST_TIMEOUT_MS,
            );
            if (!pollResult.ok) {
              throw new Error(String(pollResult.error || "Upload failed"));
            }
            const isDuplicate = isDbMessageDuplicate(pollResult.data);
            out.push({
              id,
              status: "ok",
              outcome: isDuplicate ? ("skipped" as const) : undefined,
              fileName,
              type: mediaType,
              data: pollResult.data,
              message: isDuplicate ? DUPLICATE_SKIP_MESSAGE : undefined,
              persisted: shouldStoreRemote && shouldKeepOriginalFile(mediaType),
            });
          } catch (error) {
            if (!shouldFallbackToPersistentAdd(error)) {
              throw error;
            }
            const data = await submitPersistentAdd({
              fields,
              file: uploadFile,
            });
            const fallbackDuplicate = isDbMessageDuplicate(data);
            out.push({
              id,
              status: "ok",
              outcome: fallbackDuplicate ? ("skipped" as const) : undefined,
              fileName,
              type: mediaType,
              data,
              message: fallbackDuplicate ? DUPLICATE_SKIP_MESSAGE : undefined,
              persisted: shouldStoreRemote && shouldKeepOriginalFile(mediaType),
            });
          }
          continue;
        }

        const data = await bgUpload<any>({
          path: getProcessPathForType(mediaType),
          method: "POST",
          fields: serializeUploadFields(fields),
          file: uploadFile,
          timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
          ...DIRECT_QUICK_INGEST_TRANSPORT,
        });

        const directDuplicate = isDbMessageDuplicate(data);
        out.push({
          id,
          status: "ok",
          outcome: directDuplicate ? ("skipped" as const) : undefined,
          fileName,
          type: mediaType,
          data,
          message: directDuplicate ? DUPLICATE_SKIP_MESSAGE : undefined,
          persisted: false,
        });
      } catch (error) {
        out.push({
          id,
          status: "error",
          fileName,
          type: "file",
          error:
            error instanceof Error
              ? error.message
              : String(error || "Upload failed"),
        });
      }
    }

    return { ok: true, results: out };
  } finally {
    clearDirectSessionTracking(directSessionId);
  }
};

export const submitQuickIngestBatch = async (
  input: QuickIngestBatchInput,
): Promise<QuickIngestBatchResponse> => {
  if (input?.pendingRunRequest) {
    return await runVersion2QuickIngestBatch(input);
  }
  if (
    !isDirectQuickIngestSessionId(input?.__quickIngestSessionId) &&
    (await canUseExtensionMessagingRuntime())
  ) {
    try {
      const result =
        await sendExtensionMessageWithTimeout<QuickIngestBatchResponse>({
          type: "tldw:quick-ingest-batch",
          payload: input,
        });
      return result;
    } catch {
      // Fall through to the direct path when runtime messaging is unavailable
      // even though the extension context still exists.
    }
  }

  return await runDirectQuickIngestBatch(input);
};

export const startQuickIngestSession = async (
  input: QuickIngestBatchInput,
): Promise<QuickIngestStartAck> => {
  if (await canUseExtensionMessagingRuntime()) {
    const sessionId = createExtensionQuickIngestSessionId();
    const attemptToken = createExtensionQuickIngestAttemptToken();
    try {
      const ack = await sendExtensionMessageWithTimeout<QuickIngestStartAck>({
        type: "tldw:quick-ingest/start",
        sessionId,
        attemptToken,
        payload: input,
      });
      if (!ack?.ok || ack.sessionId !== sessionId) {
        return {
          ok: false,
          sessionId,
          error:
            ack?.error ||
            "Extension quick ingest returned a conflicting session identity.",
        };
      }
      return ack;
    } catch {
      try {
        const replay =
          await sendExtensionMessageWithTimeout<QuickIngestSessionReplayResponse>(
            {
              type: "tldw:quick-ingest/replay",
              payload: { sessionId },
            },
          );
        if (replay?.ok) return { ok: true, sessionId };
        return {
          ok: false,
          sessionId,
          error:
            replay?.error ||
            "Quick ingest start delivery was interrupted and no retained session was found.",
        };
      } catch (error) {
        return {
          ok: false,
          indeterminate: true,
          sessionId,
          error:
            error instanceof Error
              ? error.message
              : "Quick ingest start delivery was interrupted.",
        };
      }
    }
  }

  // Direct runtimes currently run ingest synchronously. Return a local ack
  // so session-native callers can still establish a run identity.
  return {
    ok: true,
    sessionId: `${DIRECT_QUICK_INGEST_SESSION_PREFIX}${Date.now()}-${buildDirectSessionSuffix()}`,
  };
};

const replayQuickIngestSessionMessage = async (
  sessionId: string,
): Promise<QuickIngestSessionReplayResponse> =>
  sendExtensionMessageWithTimeout<QuickIngestSessionReplayResponse>({
    type: "tldw:quick-ingest/replay",
    payload: { sessionId },
  });

export const queryQuickIngestSession = async (
  sessionId: string,
): Promise<QuickIngestSessionReplayResponse> => {
  const normalizedSessionId = String(sessionId || "").trim();
  if (!normalizedSessionId) return { ok: false, error: "Missing session id." };
  if (!(await canUseExtensionMessagingRuntime())) {
    return { ok: false, error: "Extension runtime is unavailable." };
  }
  try {
    return await replayQuickIngestSessionMessage(normalizedSessionId);
  } catch (error) {
    return {
      ok: false,
      error:
        error instanceof Error
          ? error.message
          : "Quick ingest session replay failed.",
    };
  }
};

export const acknowledgeQuickIngestSessionReplay = async (
  input: QuickIngestReplayAckInput,
): Promise<QuickIngestCancelResponse> => {
  const sessionId = String(input?.sessionId || "").trim();
  const runId = String(input?.runId || "").trim();
  const generation = String(input?.generation || "").trim();
  if (!sessionId || !runId || !generation) {
    return { ok: false, error: "Replay acknowledgement is incomplete." };
  }
  if (!(await canUseExtensionMessagingRuntime())) {
    return { ok: false, error: "Extension runtime is unavailable." };
  }
  try {
    return await sendExtensionMessageWithTimeout<QuickIngestCancelResponse>({
      type: "tldw:quick-ingest/replay-ack",
      payload: { sessionId, runId, generation },
    });
  } catch (error) {
    return {
      ok: false,
      error:
        error instanceof Error
          ? error.message
          : "Quick ingest replay acknowledgement failed.",
    };
  }
};

export const cancelQuickIngestSession = async (
  input: QuickIngestCancelInput,
): Promise<QuickIngestCancelResponse> => {
  const sessionId = String(input?.sessionId || "").trim();
  const tracking = input?.tracking;
  const reason = input?.reason || "user_cancelled";
  if (!sessionId) {
    return { ok: false, error: "Missing session id." };
  }

  const directSession =
    isDirectQuickIngestSessionId(sessionId) ||
    tracking?.mode === "webui-direct" ||
    directQuickIngestSessionTrackers.has(sessionId);

  if (directSession) {
    directQuickIngestCancelledSessions.add(sessionId);
  }

  if (tracking?.runId) {
    try {
      await cancelRun(mediaMethods, tracking.runId, {
        reason,
      });
      return { ok: true };
    } catch (error) {
      const status =
        error instanceof PlaylistIngestPublicError ? error.status : null;
      if (status !== 404 && status !== 405 && status !== 501) {
        return {
          ok: false,
          error:
            error instanceof Error
              ? error.message
              : "The ingest run could not be cancelled.",
        };
      }
      // Older servers may not support run cancellation; try tracked batches.
    }
  }

  if (!directSession) {
    if (!(await canUseExtensionMessagingRuntime())) {
      return {
        ok: false,
        error:
          "The extension runtime is unavailable; cancellation remains unconfirmed and will be reconciled when it reconnects.",
      };
    }
    try {
      return await sendExtensionMessageWithTimeout<QuickIngestCancelResponse>({
        type: "tldw:quick-ingest/cancel",
        payload: {
          sessionId,
          reason: input?.reason,
        },
      });
    } catch (error) {
      return {
        ok: false,
        error:
          error instanceof Error
            ? error.message
            : "The extension runtime did not confirm cancellation.",
      };
    }
  }

  directQuickIngestCancelledSessions.add(sessionId);
  await cancelDirectSessionBatches(sessionId, reason);
  const batchIds = normalizeBatchIds([
    ...(input?.batchIds || []),
    tracking?.batchId || "",
    ...(tracking?.batchIds || []),
  ]);
  let cancelledBatchCount = 0;
  let lastBatchError: unknown = null;
  for (const batchId of batchIds) {
    try {
      await bgRequest<any>({
        path: `/api/v1/media/ingest/jobs/cancel?batch_id=${encodeURIComponent(
          batchId,
        )}&reason=${encodeURIComponent(reason)}`,
        method: "POST",
        timeoutMs: 10_000,
        returnResponse: true,
        ...DIRECT_QUICK_INGEST_TRANSPORT,
      });
      cancelledBatchCount += 1;
    } catch (error) {
      lastBatchError = error;
    }
  }
  if (batchIds.length > 0 && cancelledBatchCount === 0) {
    return {
      ok: false,
      error:
        lastBatchError instanceof Error
          ? lastBatchError.message
          : "The tracked ingest batches could not be cancelled.",
    };
  }
  if (tracking?.runId && batchIds.length === 0) {
    return {
      ok: false,
      error: "This server does not support run cancellation, and no tracked batches were available.",
    };
  }
  return { ok: true };
};

export const __resetQuickIngestRuntimeHealthForTests = (): void => {
  quickIngestRuntimeMessagingUsable = null;
  lastQuickIngestRuntimeHealthCheckAt = 0;
};
