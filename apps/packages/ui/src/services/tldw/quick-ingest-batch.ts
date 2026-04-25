import { browser } from "wxt/browser";
import { bgRequest, bgUpload } from "@/services/background-proxy";
import {
  getProcessPathForType,
  inferIngestTypeFromUrl,
  inferUploadMediaTypeFromFile,
  normalizeMediaType,
  shouldKeepOriginalFile,
} from "@/services/tldw/media-routing";
import { resolvePerformChunking } from "@/services/tldw/ingest-defaults";
import {
  createIngestJobsTracker,
  extractIngestJobIds,
  pollSingleIngestJob,
} from "@/services/tldw/ingest-jobs-orchestrator";
import {
  normalizePersistentAddResponse,
  shouldFallbackToPersistentAdd,
} from "@/services/tldw/quick-ingest-fallback";
import type { PersistedQuickIngestTracking } from "@/components/Common/QuickIngest/types";
import {
  DUPLICATE_SKIP_MESSAGE,
  isDbMessageDuplicate,
} from "@/components/Common/QuickIngest/constants";

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
  };
  advancedValues?: Record<string, any>;
  fileDefaults?: TypeDefaults;
  chunkingTemplateName?: string;
  autoApplyTemplate?: boolean;
  __quickIngestSessionId?: string;
  onTrackingMetadata?: (tracking: PersistedQuickIngestTracking) => void;
};

type QuickIngestBatchResult = {
  id: string;
  status: "ok" | "error";
  outcome?: "skipped";
  url?: string;
  fileName?: string;
  type: string;
  data?: unknown;
  error?: string;
  message?: string;
  persisted?: boolean;
};

type QuickIngestBatchResponse = {
  ok: boolean;
  error?: string;
  results?: QuickIngestBatchResult[];
};

export type QuickIngestStartAck = {
  ok: boolean;
  sessionId?: string;
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

const getRuntimeManifestVersion = (): number | null => {
  try {
    const manifestVersion = Number(browser?.runtime?.getManifest?.()?.manifest_version);
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
    perform_chunking: resolvePerformChunking(common?.perform_chunking),
    overwrite_existing: Boolean(common?.overwrite_existing),
    keep_original_file: persist && shouldKeepOriginalFile(rawType),
  };

  const nested: Record<string, any> = {};
  for (const [key, value] of Object.entries(advancedValues || {})) {
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

  if (chunkingTemplateName) {
    fields.chunking_template_name = chunkingTemplateName;
  }
  if (autoApplyTemplate) {
    fields.auto_apply_template = true;
  }

  return fields;
};

const processWebScrape = async ({
  url,
  entry,
  common,
  advancedValues,
}: {
  url: string;
  entry?: QuickIngestEntry;
  common?: QuickIngestBatchInput["common"];
  advancedValues?: Record<string, any>;
}): Promise<any> => {
  const nestedBody: Record<string, any> = {};
  for (const [key, value] of Object.entries(advancedValues || {})) {
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
    mode: "ephemeral",
    summarize_checkbox: Boolean(common?.perform_analysis),
    ...normalizedBody,
  };

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

const runDirectQuickIngestBatch = async (
  input: QuickIngestBatchInput,
): Promise<QuickIngestBatchResponse> => {
  const entries = Array.isArray(input.entries) ? input.entries : [];
  const files = Array.isArray(input.files) ? input.files : [];
  const fileDefaults =
    input.fileDefaults && typeof input.fileDefaults === "object"
      ? input.fileDefaults
      : {};
  const shouldStoreRemote =
    Boolean(input.storeRemote) && !Boolean(input.processOnly);
  const directSessionId =
    String(input.__quickIngestSessionId || "").trim() || undefined;

  const out: QuickIngestBatchResult[] = [];

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

      try {
        let data: unknown;
        if (shouldStoreRemote) {
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
          fields.urls = [url];
          try {
            const submitData = await bgUpload<any>({
              path: "/api/v1/media/ingest/jobs",
              method: "POST",
              fields: serializeUploadFields(fields),
              timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
              ...DIRECT_QUICK_INGEST_TRANSPORT,
            });
            const batchId = String(submitData?.batch_id || "").trim();
            const jobIds = extractIngestJobIds(submitData);
            if (!batchId || jobIds.length === 0) {
              throw new Error("Ingest job submission returned no job IDs.");
            }
            const directTracker = ensureDirectSessionTracker(directSessionId);
            directTracker?.trackJobs(batchId, jobIds, { sourceId: entry.id });
            input.onTrackingMetadata?.({
              mode: "webui-direct",
              sessionId: directSessionId,
              batchId,
              batchIds: [batchId],
              jobIds,
              submittedItemIds: [entry.id],
              itemIds: [entry.id],
              jobIdToItemId: buildJobIdToItemId(jobIds, entry.id),
              startedAt: Date.now(),
            });
            const firstJobId = jobIds[0];
            const pollResult = await pollIngestJobStatus(
              firstJobId,
              DIRECT_INGEST_TIMEOUT_MS,
            );
            if (!pollResult.ok) {
              throw new Error(String(pollResult.error || "Ingest failed"));
            }
            data = pollResult.data;
          } catch (error) {
            if (!shouldFallbackToPersistentAdd(error)) {
              throw error;
            }
            data = await submitPersistentAdd({ fields });
          }
        } else if (resolvedType === "html") {
          data = await processWebScrape({
            url,
            entry,
            common: input.common,
            advancedValues: input.advancedValues,
          });
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
          data = await bgUpload<any>({
            path: getProcessPathForType(resolvedType),
            method: "POST",
            fields: serializeUploadFields(fields),
            timeoutMs: DIRECT_INGEST_TIMEOUT_MS,
            ...DIRECT_QUICK_INGEST_TRANSPORT,
          });
        }

        out.push({
          id: entry.id,
          status: "ok",
          url,
          type: resolvedType,
          data,
          persisted: false,
        });
      } catch (error) {
        out.push({
          id: entry.id,
          status: "error",
          url,
          type: resolvedType,
          error:
            error instanceof Error
              ? error.message
              : String(error || "Request failed"),
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
            const batchId = String(submitData?.batch_id || "").trim();
            const jobIds = extractIngestJobIds(submitData);
            if (!batchId || jobIds.length === 0) {
              throw new Error("Ingest job submission returned no job IDs.");
            }
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
  if (!shouldPreferDirectQuickIngestSession() && (await canUseExtensionMessagingRuntime())) {
    try {
      return await sendExtensionMessageWithTimeout<QuickIngestStartAck>({
        type: "tldw:quick-ingest/start",
        payload: input,
      });
    } catch {
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
      return await sendExtensionMessageWithTimeout<QuickIngestCancelResponse>({
        type: "tldw:quick-ingest/cancel",
        payload: {
          sessionId,
          reason: input?.reason,
        },
      });
    } catch {
      // Fall through to the direct cancellation path when runtime messaging
      // stops responding in packaged extension contexts.
    }
  }

  directQuickIngestCancelledSessions.add(sessionId);
  await cancelDirectSessionBatches(
    sessionId,
    input?.reason || "user_cancelled",
  );
  for (const batchId of normalizeBatchIds([
    ...(input?.batchIds || []),
    tracking?.batchId || "",
    ...(tracking?.batchIds || []),
  ])) {
    try {
      await bgRequest<any>({
        path: `/api/v1/media/ingest/jobs/cancel?batch_id=${encodeURIComponent(
          batchId,
        )}&reason=${encodeURIComponent(input?.reason || "user_cancelled")}`,
        method: "POST",
        timeoutMs: 10_000,
        returnResponse: true,
        ...DIRECT_QUICK_INGEST_TRANSPORT,
      });
    } catch {
      // best effort cancellation for resumed sessions without in-memory trackers
    }
  }
  return { ok: true };
};

export const __resetQuickIngestRuntimeHealthForTests = (): void => {
  quickIngestRuntimeMessagingUsable = null;
  lastQuickIngestRuntimeHealthCheckAt = 0;
};
