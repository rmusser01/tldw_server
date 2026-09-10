import { browser } from "wxt/browser";
import { createSafeStorage } from "@/utils/safe-storage";
import { formatErrorMessage } from "@/utils/format-error-message";
import { sanitizeRagProviderFailure } from "@/services/rag/provider-error-contract";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import { tldwAuth } from "@/services/tldw/TldwAuth";
import { tldwModels } from "@/services/tldw";
import { apiSend } from "@/services/api-send";
import { tldwRequest } from "@/services/tldw/request-core";
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode";
import {
  hasNewerCurrentAccessToken,
  resolveEffectiveTldwConfig,
  storeRefreshRotationIfCurrent,
  waitForNewerCurrentAccessToken,
} from "@/services/tldw/single-user-credential";
import {
  getProcessPathForType,
  getProcessPathForUrl,
  inferMediaTypeFromUrl,
  inferUploadMediaTypeFromFile,
  normalizeMediaType,
} from "@/services/tldw/media-routing";
import {
  normalizePersistentAddResponse,
  shouldFallbackToPersistentAdd,
} from "@/services/tldw/quick-ingest-fallback";
import {
  applyQuickIngestChunkingFields,
  shouldSubmitQuickIngestAdvancedField,
} from "@/services/tldw/quick-ingest-chunking";
import {
  ensureSidepanelOpen,
  pickFirstString,
  extractTranscriptionPieces,
  clampText,
  notify,
} from "@/services/background-helpers";
import { buildClipDraft } from "@/services/web-clipper/draft-builder";
import {
  CLIPPER_CAPTURE_MESSAGE_TYPE,
  normalizePendingClipDraft,
} from "@/services/web-clipper/pending-draft";
import { isRestrictedClipperPage } from "@/services/web-clipper/content-extract";
import { ModelDb } from "@/db/models";
import { generateID } from "@/db";
import {
  initBackground,
  MODEL_WARM_ALARM_NAME,
} from "@/entries/shared/background-init";
import { startNotificationSubscription } from "@/entries/shared/notification-subscription";
import {
  buildContextMenuAddPayload,
  buildContextMenuProcessPayload,
  extractYouTubeTimestampSeconds,
  normalizeUrlForDedupe,
  resolveContextMenuTargetUrl,
} from "@/entries/shared/ingest-payloads";
import {
  createQuickIngestSessionRuntime,
  type QuickIngestSessionRunContext,
} from "@/entries/shared/quick-ingest-session-runtime";
import {
  createIngestJobsTracker,
  pollTrackedIngestJobs,
  requireSubmittedIngestJobs,
} from "@/services/tldw/ingest-jobs-orchestrator";
import {
  completedIngestJobIndicatesFailure,
  extractCompletedIngestJobError,
  extractCompletedIngestJobMediaId,
} from "@/services/tldw/ingest-job-results";
import {
  buildConferenceCollectionCreatePayload,
  buildConferenceCollectionItemPayload,
} from "@/services/tldw/conference-collections";
import {
  ABSOLUTE_URL_BLOCK_ERROR,
  evaluateAbsoluteUrlAccess,
} from "@/utils/absolute-url-guard";
import {
  createServicePromptScopeChangedError,
  isRequestConfigScopeChangedError,
  isServicePromptRequestPath,
  servicePromptPrincipalMatches,
  servicePromptRefreshLineageMatches,
  servicePromptSingleUserApiKeyScopeMatches,
  servicePromptTargetsMatch,
} from "@/services/tldw/service-prompt-scope-error";
import { arrayBufferToBase64 } from "@/utils/compress";
import {
  createSerializedSessionStateWriter,
  getSessionStorageArea,
  readPersistedSessionState,
  selectInterruptedIngestFunnelIds,
  serializeIngestSessions,
  serializePendingReplay,
  serializeQuickIngestBatches,
  serializeQuickIngestSessions,
  type PersistedQuickIngestBatch,
} from "@/entries/background-session-store";

type BackgroundDiagnostics = {
  startedAt: number;
  modelWarmCount: number;
  lastModelWarmAt: number | null;
  lastModelWarmError: string | null;
  runtimeMessageCount: number;
  runtimePingCount: number;
  lastRuntimeMessageType: string | null;
  lastRuntimeSenderUrl: string | null;
  alarmFires: number;
  lastAlarmAt: number | null;
  ports: {
    stream: number;
    stt: number;
    copilot: number;
  };
  lastStreamAt: number | null;
  lastSttAt: number | null;
  lastCopilotAt: number | null;
};

const backgroundDiagnostics: BackgroundDiagnostics = {
  startedAt: Date.now(),
  modelWarmCount: 0,
  lastModelWarmAt: null,
  lastModelWarmError: null,
  runtimeMessageCount: 0,
  runtimePingCount: 0,
  lastRuntimeMessageType: null,
  lastRuntimeSenderUrl: null,
  alarmFires: 0,
  lastAlarmAt: null,
  ports: {
    stream: 0,
    stt: 0,
    copilot: 0,
  },
  lastStreamAt: null,
  lastSttAt: null,
  lastCopilotAt: null,
};

const logBackgroundError = (label: string, error: unknown) => {
  console.debug(`[tldw] background ${label} failed`, error);
};

type ServicePromptTargetLock = Readonly<{
  serverUrl?: unknown;
  authMode?: unknown;
  authSource?: unknown;
  orgId?: unknown;
  expectedUserId?: unknown;
  expectedRefreshToken?: unknown;
  expectedSingleUserApiKeyScope?: unknown;
}>;

const readServicePromptTargetLock = (
  value: unknown,
): ServicePromptTargetLock | null =>
  value && typeof value === "object" && !Array.isArray(value)
    ? Object.freeze({
        serverUrl: (value as Record<string, unknown>).serverUrl,
        authMode: (value as Record<string, unknown>).authMode,
        authSource: (value as Record<string, unknown>).authSource,
        orgId: (value as Record<string, unknown>).orgId,
        expectedUserId: (value as Record<string, unknown>).expectedUserId,
        expectedRefreshToken: (value as Record<string, unknown>).expectedRefreshToken,
        expectedSingleUserApiKeyScope:
          (value as Record<string, unknown>).expectedSingleUserApiKeyScope,
      })
    : null;

const waitFor = (delayMs: number): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, delayMs);
  });

const sendBackgroundRuntimeMessage = async (
  message: {
    from: "background";
    type: string;
    text?: string;
    payload?: unknown;
  },
  options?: {
    retryDelayMs?: number;
    maxAttempts?: number;
    requireHandled?: boolean;
  },
): Promise<void> => {
  const retryDelayMs = options?.retryDelayMs ?? 500;
  const maxAttempts = Math.max(1, options?.maxAttempts ?? 4);
  const requireHandled = options?.requireHandled ?? false;
  let lastError: unknown = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const response = await browser.runtime.sendMessage(message);
      if (
        !requireHandled ||
        (response as { handled?: boolean } | undefined)?.handled
      ) {
        return;
      }
      lastError = new Error(`No receiver acknowledged ${message.type}`);
    } catch (error) {
      lastError = error;
      logBackgroundError(`send ${message.type}`, error);
    }

    if (attempt < maxAttempts) {
      await waitFor(retryDelayMs);
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error(`Failed to deliver ${message.type}`);
};

type WebClipperContextMenuClickInfo = {
  pageUrl?: string | null;
  selectionText?: string | null;
};

type WebClipperContextMenuTab =
  | {
      id?: number | null;
      url?: string | null;
      title?: string | null;
    }
  | null
  | undefined;

export const launchWebClipperFromContextMenu = async (
  info: WebClipperContextMenuClickInfo,
  tab?: WebClipperContextMenuTab,
): Promise<void> => {
  const pageUrl = String(info?.pageUrl || tab?.url || "").trim();
  const pageTitle = String(tab?.title || "").trim();
  const selectionText = String(info?.selectionText || "").trim();

  if (isRestrictedClipperPage(pageUrl)) {
    notify(
      browser.i18n.getMessage("contextSaveToClipper") || "Save to Clipper",
      browser.i18n.getMessage("contextSaveToClipperRestrictedPage") ||
        "This page is restricted, so the clipper cannot capture it.",
    );
    return;
  }

  const requestedType = selectionText ? "selection" : "article";
  const fallbackDraft = buildClipDraft({
    requestedType,
    pageUrl,
    pageTitle: pageTitle || pageUrl,
    extracted: {
      selectionText: selectionText || undefined,
      articleText: selectionText || undefined,
      fullPageText: selectionText || undefined,
    },
  });
  const tabsApi = (browser as any)?.tabs;
  let clipDraft = fallbackDraft;

  if (tab?.id != null && typeof tabsApi?.sendMessage === "function") {
    try {
      const response = await tabsApi.sendMessage(tab.id, {
        type: "capture-web-clipper",
        requestedType,
        pageUrl,
        pageTitle: pageTitle || pageUrl,
        selectionText: selectionText || undefined,
      });
      const normalized = normalizePendingClipDraft(response);
      if (normalized) {
        clipDraft = normalized;
      }
    } catch (error) {
      logBackgroundError("request web clipper capture", error);
    }
  }

  const title =
    browser.i18n.getMessage("contextSaveToClipper") || "Save to Clipper";
  try {
    await ensureSidepanelOpen(tab?.id ?? undefined);
    await sendBackgroundRuntimeMessage(
      {
        from: "background",
        type: CLIPPER_CAPTURE_MESSAGE_TYPE,
        text: clipDraft.visibleBody,
        payload: clipDraft,
      },
      {
        retryDelayMs: 500,
        maxAttempts: 4,
        requireHandled: true,
      },
    );
  } catch (error) {
    logBackgroundError("launch web clipper", error);
    notify(
      title,
      browser.i18n.getMessage("contextSaveToClipperDeliveryFailed") ||
        "Could not open the sidebar to save this clip. Check that the tldw Assistant sidebar is allowed on this site and try again.",
    );
  }
};

type IngestLifecycleStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "auth_required";

type IngestFunnelEvent =
  | "context_click"
  | "job_queued"
  | "media_completed"
  | "first_chat_message";

type IngestSession = {
  funnelId: string;
  url: string;
  normalizedUrl: string;
  tabId?: number;
  status: IngestLifecycleStatus;
  batchId?: string;
  jobIds: number[];
  createdAt: number;
  retryCount: number;
  awaitingAuth: boolean;
  mediaId?: number;
  lastError?: string;
  timestampSeconds?: number | null;
};

type QuickIngestModalSession = {
  sessionId: string;
  cancelled: boolean;
  abortControllers: Set<AbortController>;
};

// Metadata carried alongside each remotely-queued ingest job so results can be
// mapped back to their originating entry/file. Defined at module scope so both
// the live quick-ingest run and the post-restart resume path share one type.
type QuickIngestRemoteResultMeta = {
  id: string;
  type: string;
  url?: string;
  fileName?: string;
};

const warmModels = async (
  force = false,
  throwOnError = false,
): Promise<any[] | null> => {
  backgroundDiagnostics.modelWarmCount += 1;
  backgroundDiagnostics.lastModelWarmAt = Date.now();
  backgroundDiagnostics.lastModelWarmError = null;
  try {
    const models = await tldwModels.warmCache(Boolean(force), {
      refreshOpenRouter: Boolean(force),
    });

    // Sync models to local database
    if (models && models.length > 0) {
      const db = new ModelDb();
      const existing = await db.getAll();
      const existingLookups = new Set(
        existing.map((m: any) => m?.lookup).filter(Boolean),
      );

      for (const model of models) {
        try {
          const lookup = `${model.id}_tldw_${model.provider}`;
          if (existingLookups.has(lookup)) continue;

          // Transform ModelInfo to Model format
          const dbModel = {
            id: `${model.id}_${generateID()}`,
            model_id: model.id,
            name: model.name,
            provider_id: `tldw_${model.provider}`,
            lookup,
            model_type: model.type || "chat",
            db_type: "openai_model",
          };

          await db.create(dbModel);
          existingLookups.add(lookup);
        } catch (err) {
          // Log but don't fail the entire sync if one model fails
          console.debug("[tldw] Failed to sync model to DB:", model.id, err);
        }
      }
    }

    return models;
  } catch (e) {
    console.debug("[tldw] model warmup failed", e);
    backgroundDiagnostics.lastModelWarmError =
      (e as any)?.message || "model warmup failed";
    if (throwOnError) {
      throw e;
    }
    return null;
  }
};

export default defineBackground({
  main() {
    const storage = createSafeStorage({ area: "local" });
    const sessionStorage = createSafeStorage({ area: "session" });
    const getEffectiveConfig = () =>
      resolveEffectiveTldwConfig({
        persistent: storage,
        session: sessionStorage,
      });
    const resolveCurrentServicePromptConfig = async (
      checked: ServicePromptTargetLock,
    ) => {
      const current = await getEffectiveConfig();
      const singleUserApiKeyScopeMatches = current
        ? servicePromptSingleUserApiKeyScopeMatches(
            current,
            checked.expectedSingleUserApiKeyScope,
          )
        : true;
      if (
        (!current && !isHostedTldwDeployment()) ||
        (current && !servicePromptTargetsMatch(current, checked)) ||
        (current &&
          checked.expectedUserId !== null &&
          checked.expectedUserId !== undefined &&
          current.authMode === "multi-user" &&
          !isHostedTldwDeployment() &&
          current.authSource !== "cookie-session" &&
          !servicePromptPrincipalMatches(current, checked.expectedUserId)) ||
        (current &&
          !servicePromptRefreshLineageMatches(
            current,
            checked.expectedRefreshToken,
          )) ||
        !singleUserApiKeyScopeMatches ||
        (current?.authMode === "multi-user" &&
          !isHostedTldwDeployment() &&
          current.authSource !== "cookie-session" &&
          !String(current.accessToken || "").trim())
      ) {
        throw createServicePromptScopeChangedError();
      }
      const effective = current || checked;
      return {
        ...effective,
        serverUrl: checked.serverUrl,
        authMode: checked.authMode,
        authSource: checked.authSource,
        orgId: checked.orgId,
        apiKey: current?.apiKey,
        accessToken: current?.accessToken,
        refreshToken: current?.refreshToken,
      };
    };
    let handleRuntimeMessageRef:
      | ((message: any, sender: any) => Promise<any>)
      | null = null;
    let initializePromise: Promise<void> | null = null;
    let isCopilotRunning: boolean = false;
    let actionIconClick: string = "webui";
    let contextMenuClick: string = "sidePanel";
    const contextMenuId = {
      webui: "open-web-ui-pa",
      sidePanel: "open-side-panel-pa",
    };
    const transcribeMenuId = {
      transcribe: "transcribe-media-pa",
      transcribeAndSummarize: "transcribe-and-summarize-media-pa",
    };
    const saveToClipperMenuId = "save-to-clipper-pa";
    const saveToCompanionMenuId = "save-to-companion-pa";
    const saveToNotesMenuId = "save-to-notes-pa";
    const narrateSelectionMenuId = "narrate-selection-pa";
    const getActionApi = () => {
      const anyBrowser = browser as any;
      const anyChrome = (globalThis as any).chrome;
      return (
        anyBrowser?.action ||
        anyBrowser?.browserAction ||
        anyChrome?.action ||
        anyChrome?.browserAction
      );
    };

    const initialize = async () => {
      try {
        await initBackground({
          storage,
          contextMenuId,
          saveToCompanionMenuId,
          saveToClipperMenuId,
          saveToNotesMenuId,
          narrateSelectionMenuId,
          transcribeMenuId,
          warmModels,
          capabilities: {
            sendToTldw: true,
            processLocal: true,
            transcribe: true,
            openApiCheck: true,
          },
          onActionIconClickChange: (value) => {
            actionIconClick = value;
          },
          onContextMenuClickChange: (value) => {
            contextMenuClick = value;
          },
        });
        // Start notification subscription after init
        void startNotificationSubscription().catch((err) => {
          console.debug(
            "[background] Notification subscription deferred:",
            err,
          );
        });
        // Rehydrate persisted ingest/auth-replay/quick-ingest session state and
        // resume any polling interrupted by a worker suspension (TASK-12094).
        void rehydrateSessionState();
      } catch (error) {
        console.error("Error in initLogic:", error);
      }
    };

    const buildBackgroundDiagnostics = () => {
      const memory = (globalThis as any)?.performance?.memory;
      return {
        ...backgroundDiagnostics,
        chatQueueDepth: chatQueue.length,
        chatActiveCount,
        chatBackoffMs,
        chatBackoffUntil,
        memory:
          memory && typeof memory.usedJSHeapSize === "number"
            ? {
                usedJSHeapSize: memory.usedJSHeapSize,
                totalJSHeapSize: memory.totalJSHeapSize,
                jsHeapSizeLimit: memory.jsHeapSizeLimit,
              }
            : null,
      };
    };

    (
      globalThis as typeof globalThis & {
        __tldwBackgroundDiagnostics?: () => ReturnType<
          typeof buildBackgroundDiagnostics
        >;
      }
    ).__tldwBackgroundDiagnostics = buildBackgroundDiagnostics;

    let refreshInFlight: Promise<any> | null = null;
    const scopedRefreshes = new Map<string, Promise<void>>();
    const refreshScopedAuth = async (
      checked: ServicePromptTargetLock,
      originalConfig?: Awaited<
        ReturnType<typeof resolveCurrentServicePromptConfig>
      >,
    ): Promise<void> => {
      const cfg = originalConfig ??
        await resolveCurrentServicePromptConfig(checked);
      const refreshToken = String(cfg.refreshToken || "").trim();
      const capturedAccessToken = String(cfg.accessToken || "").trim();
      if (!refreshToken) {
        throw new Error("Token refresh failed: no refresh token available");
      }
      if (
        originalConfig &&
        await hasNewerCurrentAccessToken(
          storage,
          checked,
          capturedAccessToken,
        )
      ) {
        return;
      }
      const current = originalConfig
        ? await resolveCurrentServicePromptConfig(checked)
        : cfg;
      if (String(current.refreshToken || "").trim() !== refreshToken) {
        throw createServicePromptScopeChangedError();
      }
      const key = JSON.stringify([
        checked.serverUrl ?? null,
        checked.authMode ?? null,
        checked.authSource ?? null,
        checked.orgId ?? null,
        refreshToken,
      ]);
      let refresh = scopedRefreshes.get(key);
      if (!refresh) {
        refresh = (async () => {
          try {
            const response = await tldwRequest(
              {
                path: "/api/v1/auth/refresh",
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: { refresh_token: refreshToken },
                noAuth: true,
              },
              {
                getConfig: async () => cfg,
                useRuntimeAuthOverride: false,
              },
            );
            const tokens = response.ok
              ? (response.data as {
                  access_token?: string;
                  refresh_token?: string;
                } | null)
              : null;
            if (!tokens?.access_token) {
              throw new Error(
                `Token refresh failed: ${response.error || `no access token in refresh response (status ${response.status ?? "unknown"})`}`,
              );
            }
            const stored = await storeRefreshRotationIfCurrent(
              storage,
              { ...checked, accessToken: capturedAccessToken },
              refreshToken,
              {
                accessToken: tokens.access_token,
                refreshToken: tokens.refresh_token || refreshToken,
              },
            );
            if (!stored) {
              throw createServicePromptScopeChangedError();
            }
          } catch (error) {
            if (isRequestConfigScopeChangedError(error)) throw error;
            if (
              originalConfig &&
              await waitForNewerCurrentAccessToken(
                storage,
                checked,
                capturedAccessToken,
              )
            ) {
              return;
            }
            throw error;
          } finally {
            scopedRefreshes.delete(key);
          }
        })();
        scopedRefreshes.set(key, refresh);
      }
      await refresh;
    };
    let streamDebugEnabled = false;
    const ingestSessions = new Map<string, IngestSession>();
    const pendingAuthReplay = new Set<string>();
    const quickIngestModalSessions = new Map<string, QuickIngestModalSession>();
    // Durable per-session record of a quick-ingest batch's remote-job polling
    // phase (job ids + progress cursor + already-collected results). Persisted so
    // the polling phase can be resumed/finalized after a worker restart even
    // though the multipart UPLOAD that produced the job ids cannot be resumed.
    const quickIngestBatchRecords = new Map<string, PersistedQuickIngestBatch>();

    // MV3 worker-survivability (TASK-12094): the maps above are wiped when
    // Chrome suspends an idle service worker (~30s). We mirror them into
    // chrome.storage.session (fallback local) on mutation and rehydrate them on
    // worker restart, and drive ingest polling from a chrome.alarms backstop so
    // completed ingests still notify / cancel + retry still find the session.
    const INGEST_POLL_ALARM_NAME = "tldw:ingest-poll";
    const INGEST_POLL_ALARM_PERIOD_MINUTES = 0.5;
    // Backstop alarm that resumes an interrupted quick-ingest remote-job poll
    // after the worker is suspended mid-batch.
    const QUICK_INGEST_POLL_ALARM_NAME = "tldw:quick-ingest-poll";
    const QUICK_INGEST_POLL_ALARM_PERIOD_MINUTES = 0.5;
    // Funnel ids currently being polled inside this live worker; used to avoid
    // starting a duplicate poll from the alarm/rehydrate backstop.
    const activeIngestPollFunnelIds = new Set<string>();
    // Quick-ingest session ids whose remote-job poll is running (live or resumed)
    // in this worker; avoids a duplicate resume from the alarm/rehydrate backstop.
    const activeQuickIngestBatchSessionIds = new Set<string>();
    let sessionStateHydrated = false;

    // Cancellation is meaningful only while this worker and its fetch are
    // alive, so these controllers intentionally remain in memory rather than
    // joining the durable ingest-session state above.
    const runtimeRequestControllers = new Map<string, AbortController>();
    const runCancelableRuntimeRequest = async (
      rawRequestId: unknown,
      run: (signal?: AbortSignal) => Promise<any>,
    ): Promise<any> => {
      const requestId =
        typeof rawRequestId === "string" ? rawRequestId.trim() : "";
      if (!requestId) return await run();

      const controller = new AbortController();
      runtimeRequestControllers.get(requestId)?.abort();
      runtimeRequestControllers.set(requestId, controller);
      try {
        const response = await run(controller.signal);
        if (controller.signal.aborted) {
          return {
            ok: false,
            status: 499,
            error: "Cancelled by user.",
          };
        }
        return response;
      } finally {
        if (runtimeRequestControllers.get(requestId) === controller) {
          runtimeRequestControllers.delete(requestId);
        }
      }
    };

    // Serialize the actual storage writes so that the many rapid, fire-and-forget
    // persist calls below cannot land out of order (an older Map snapshot
    // clobbering a newer one). The snapshot is taken synchronously at call time;
    // the writer coalesces overlapping writes so only the latest snapshot wins.
    const writeSessionStateSerialized = createSerializedSessionStateWriter(
      getSessionStorageArea(),
      (error) => logBackgroundError("persist session state", error),
    );

    const persistSessionState = (): void => {
      try {
        writeSessionStateSerialized({
          ingestSessions: serializeIngestSessions(ingestSessions),
          pendingAuthReplay: serializePendingReplay(pendingAuthReplay),
          quickIngestSessions:
            serializeQuickIngestSessions(quickIngestModalSessions),
          quickIngestBatches:
            serializeQuickIngestBatches(quickIngestBatchRecords),
        });
      } catch (error) {
        logBackgroundError("persist session state", error);
      }
    };

    const hasActiveIngestPollSessions = (): boolean => {
      for (const session of ingestSessions.values()) {
        if (
          (session.status === "queued" || session.status === "running") &&
          !session.awaitingAuth &&
          Array.isArray(session.jobIds) &&
          session.jobIds.length > 0
        ) {
          return true;
        }
      }
      return false;
    };

    // Keep a chrome.alarms backstop alive whenever there is an ingest still
    // being polled. If the worker is suspended mid-poll, the alarm wakes it so
    // polling resumes and the completed ingest still notifies the UI.
    const scheduleIngestPollAlarm = async (): Promise<void> => {
      try {
        if (!browser?.alarms) return;
        if (!hasActiveIngestPollSessions()) {
          await browser.alarms.clear(INGEST_POLL_ALARM_NAME);
          return;
        }
        const existing = await browser.alarms
          .get(INGEST_POLL_ALARM_NAME)
          .catch(() => null);
        if (existing) return;
        await browser.alarms.create(INGEST_POLL_ALARM_NAME, {
          periodInMinutes: INGEST_POLL_ALARM_PERIOD_MINUTES,
        });
      } catch (error) {
        logBackgroundError("schedule ingest poll alarm", error);
      }
    };

    // Keep a chrome.alarms backstop alive whenever a quick-ingest batch has a
    // persisted remote-job polling phase outstanding, so a suspended worker is
    // woken to resume and finalize it.
    const scheduleQuickIngestBatchAlarm = async (): Promise<void> => {
      try {
        if (!browser?.alarms) return;
        if (quickIngestBatchRecords.size === 0) {
          await browser.alarms.clear(QUICK_INGEST_POLL_ALARM_NAME);
          return;
        }
        const existing = await browser.alarms
          .get(QUICK_INGEST_POLL_ALARM_NAME)
          .catch(() => null);
        if (existing) return;
        await browser.alarms.create(QUICK_INGEST_POLL_ALARM_NAME, {
          periodInMinutes: QUICK_INGEST_POLL_ALARM_PERIOD_MINUTES,
        });
      } catch (error) {
        logBackgroundError("schedule quick ingest poll alarm", error);
      }
    };

    const INGEST_FUNNEL_METRICS_KEY = "tldw:ingestFunnelMetrics";
    const INGEST_FUNNEL_METRICS_LIMIT = 200;
    const METADATA_DEDUPE_FIELDS = [
      "url",
      "source_url",
      "source",
      "input_ref",
      "canonical_url",
      "webpage_url",
    ];
    const TERMINAL_INGEST_JOB_STATUSES = new Set([
      "completed",
      "failed",
      "cancelled",
      "quarantined",
    ]);

    const createFunnelId = (): string =>
      `ingest-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

    const createQuickIngestSessionId = (): string =>
      `qi-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

    const getQuickIngestModalSession = (
      sessionId: string | null | undefined,
    ): QuickIngestModalSession | null => {
      const normalized = String(sessionId || "").trim();
      if (!normalized) return null;
      return quickIngestModalSessions.get(normalized) || null;
    };

    const isQuickIngestCancelled = (
      sessionId: string | null | undefined,
    ): boolean => {
      const session = getQuickIngestModalSession(sessionId);
      return Boolean(session?.cancelled);
    };

    const registerQuickIngestAbortController = (
      sessionId: string | null | undefined,
      controller: AbortController,
    ) => {
      const session = getQuickIngestModalSession(sessionId);
      if (!session) return;
      session.abortControllers.add(controller);
    };

    const unregisterQuickIngestAbortController = (
      sessionId: string | null | undefined,
      controller: AbortController,
    ) => {
      const session = getQuickIngestModalSession(sessionId);
      if (!session) return;
      session.abortControllers.delete(controller);
    };

    const isLikelyAuthError = (status: number, error?: string): boolean => {
      if (status === 401) return true;
      const text = String(error || "").toLowerCase();
      if (!text) return false;
      return (
        text.includes("not authenticated") ||
        text.includes("api key") ||
        text.includes("login") ||
        text.includes("unauthorized")
      );
    };

    const hasUsableAuthConfig = (cfg: any): boolean => {
      if (!cfg || typeof cfg !== "object") return false;
      const authMode = String(cfg.authMode || "single-user");
      if (authMode === "multi-user") {
        return Boolean(String(cfg.accessToken || "").trim());
      }
      return Boolean(String(cfg.apiKey || "").trim());
    };

    const toPositiveInt = (value: unknown): number | null => {
      const parsed = Number(value);
      if (!Number.isFinite(parsed) || parsed <= 0) return null;
      return Math.trunc(parsed);
    };

    const pickMediaIdFromAny = (value: unknown): number | null => {
      if (!value || typeof value !== "object") return null;
      const row = value as Record<string, unknown>;
      return (
        toPositiveInt(row.media_id) ||
        toPositiveInt(row.db_id) ||
        toPositiveInt(row.id)
      );
    };

    const pickMetadataSearchMediaId = (data: unknown): number | null => {
      if (!data || typeof data !== "object") return null;
      const root = data as Record<string, unknown>;
      const rows = Array.isArray(root.results) ? root.results : [];
      for (const row of rows) {
        const mediaId = pickMediaIdFromAny(row);
        if (mediaId != null) return mediaId;
      }
      return null;
    };

    const buildMetadataSearchPath = (urlValue: string): string => {
      const filters = METADATA_DEDUPE_FIELDS.map((field) => ({
        field,
        op: "eq",
        value: urlValue,
      }));
      const query = new URLSearchParams({
        filters: JSON.stringify(filters),
        match_mode: "any",
        group_by_media: "true",
        page: "1",
        per_page: "1",
      });
      return `/api/v1/media/metadata-search?${query.toString()}`;
    };

    const appendIngestFunnelMetric = async (
      event: IngestFunnelEvent,
      funnelId: string,
      metadata?: Record<string, unknown>,
    ) => {
      try {
        const current = await storage.get<any>(INGEST_FUNNEL_METRICS_KEY);
        const entries = Array.isArray(current) ? current : [];
        entries.push({
          event,
          funnelId,
          timestamp: new Date().toISOString(),
          metadata: metadata || {},
        });
        if (entries.length > INGEST_FUNNEL_METRICS_LIMIT) {
          entries.splice(0, entries.length - INGEST_FUNNEL_METRICS_LIMIT);
        }
        await storage.set(INGEST_FUNNEL_METRICS_KEY, entries);
      } catch (error) {
        logBackgroundError("append ingest funnel metric", error);
      }
    };

    const emitBackgroundMessage = async (
      tabId: number | undefined,
      type: string,
      payload?: Record<string, unknown>,
    ) => {
      ensureSidepanelOpen(tabId);
      try {
        await browser.runtime.sendMessage({
          from: "background",
          type,
          payload,
        });
      } catch (error) {
        logBackgroundError(`send ${type}`, error);
        setTimeout(() => {
          try {
            browser.runtime.sendMessage({
              from: "background",
              type,
              payload,
            });
          } catch (retryError) {
            logBackgroundError(`send ${type} (retry)`, retryError);
          }
        }, 500);
      }
    };

    const emitIngestStatus = async (
      session: IngestSession,
      update: Partial<{
        status: IngestLifecycleStatus;
        progressPercent: number;
        progressMessage: string;
        error: string;
        mediaId: number;
        canCancel: boolean;
        canRetry: boolean;
      }>,
    ) => {
      if (update.status) {
        session.status = update.status;
      }
      if (typeof update.mediaId === "number" && update.mediaId > 0) {
        session.mediaId = Math.trunc(update.mediaId);
      }
      if (update.error) {
        session.lastError = update.error;
      }
      const canCancel =
        typeof update.canCancel === "boolean"
          ? update.canCancel
          : session.status === "queued" || session.status === "running";
      const canRetry =
        typeof update.canRetry === "boolean"
          ? update.canRetry
          : session.status === "failed" ||
            session.status === "cancelled" ||
            session.status === "auth_required";
      await emitBackgroundMessage(session.tabId, "media-ingest-status", {
        funnelId: session.funnelId,
        url: session.url,
        status: session.status,
        progressPercent: update.progressPercent,
        progressMessage: update.progressMessage,
        error: update.error,
        mediaId: session.mediaId,
        jobIds: session.jobIds,
        canCancel,
        canRetry,
        timestampSeconds: session.timestampSeconds ?? undefined,
      });
      // Persist after every status change so the session survives suspension
      // and cancel/retry can find it after a worker restart.
      void persistSessionState();
    };

    const sendIngestReadyMessage = async (
      tabId: number | undefined,
      payload: {
        funnelId: string;
        mediaId: string;
        url?: string;
        mode: "rag_media";
        timestampSeconds?: number;
      },
    ) => {
      await emitBackgroundMessage(tabId, "media-ingest-ready", payload);
    };

    const openAuthSettings = async () => {
      try {
        const settingsUrl = browser.runtime.getURL(
          "options.html#/settings/tldw",
        );
        await browser.tabs.create({ url: settingsUrl });
      } catch (error) {
        logBackgroundError("open auth settings", error);
      }
    };

    const queueAuthRecovery = async (
      session: IngestSession,
      errorText: string,
    ) => {
      session.awaitingAuth = true;
      session.lastError = errorText;
      pendingAuthReplay.add(session.funnelId);
      await emitIngestStatus(session, {
        status: "auth_required",
        error: errorText,
        canCancel: false,
        canRetry: true,
      });
      notify(
        "tldw_server",
        "Authentication required. Opened settings to update credentials; ingest will retry automatically.",
      );
      await openAuthSettings();
    };

    const findExistingMediaForUrl = async (
      rawUrl: string,
      normalizedUrl: string,
    ): Promise<number | null> => {
      const candidates = Array.from(
        new Set(
          [
            String(rawUrl || "").trim(),
            String(normalizedUrl || "").trim(),
          ].filter((value) => value.length > 0),
        ),
      );
      for (const candidate of candidates) {
        const path = buildMetadataSearchPath(candidate);
        const resp = (await handleTldwRequest({
          path,
          method: "GET",
          timeoutMs: 15000,
        })) as
          | { ok: boolean; status?: number; data?: any; error?: string }
          | undefined;
        if (!resp?.ok) {
          if (isLikelyAuthError(Number(resp?.status || 0), resp?.error)) {
            throw new Error(resp?.error || "Authentication required.");
          }
          continue;
        }
        const mediaId = pickMetadataSearchMediaId(resp.data);
        if (mediaId != null) return mediaId;
      }
      return null;
    };

    const handleTranscribeClick = async (
      info: any,
      tab: any,
      mode: "transcribe" | "transcribe+summary",
    ) => {
      const pageUrl = info.pageUrl || (tab && tab.url) || "";
      const targetUrl =
        info.linkUrl && /^https?:/i.test(info.linkUrl) ? info.linkUrl : pageUrl;
      if (!targetUrl) {
        notify("tldw_server", "No URL found to transcribe.");
        return;
      }
      const path = getProcessPathForUrl(targetUrl);
      if (
        path !== "/api/v1/media/process-audios" &&
        path !== "/api/v1/media/process-videos"
      ) {
        notify(
          "tldw_server",
          "Transcription is available for audio or video URLs only.",
        );
        return;
      }

      try {
        const resp = await apiSend({
          path,
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: {
            urls: [targetUrl],
            perform_analysis: mode === "transcribe+summary",
            perform_chunking: false,
            summarize_recursively: mode === "transcribe+summary",
            timestamp_option: true,
          },
          timeoutMs: 180000,
        });
        if (!resp?.ok) {
          notify(
            "tldw_server",
            resp?.error ||
              "Transcription failed. Check your connection and server config.",
          );
          return;
        }
        const { transcript, summary } = extractTranscriptionPieces(resp.data);
        const safeTranscript = clampText(transcript);
        const safeSummary = clampText(summary);
        const label =
          mode === "transcribe+summary"
            ? "Transcription + summary"
            : "Transcription";
        const bodyParts = [];
        if (safeTranscript) bodyParts.push(`Transcript:\n${safeTranscript}`);
        if (safeSummary) bodyParts.push(`Summary:\n${safeSummary}`);
        const combinedText =
          bodyParts.join("\n\n") ||
          "Request completed. Open Media or the sidebar to view results.";

        ensureSidepanelOpen(tab?.id);
        try {
          await browser.runtime.sendMessage({
            from: "background",
            type:
              mode === "transcribe+summary"
                ? "transcription+summary"
                : "transcription",
            text: combinedText,
            payload: {
              url: targetUrl,
              transcript: safeTranscript,
              summary: safeSummary,
              mode,
            },
          });
        } catch (error) {
          logBackgroundError("send transcription result", error);
          setTimeout(() => {
            try {
              browser.runtime.sendMessage({
                from: "background",
                type:
                  mode === "transcribe+summary"
                    ? "transcription+summary"
                    : "transcription",
                text: combinedText,
                payload: {
                  url: targetUrl,
                  transcript: safeTranscript,
                  summary: safeSummary,
                  mode,
                },
              });
            } catch (fallbackError) {
              logBackgroundError(
                "send transcription result (retry)",
                fallbackError,
              );
            }
          }, 500);
        }
        notify(
          "tldw_server",
          `${label} sent to sidebar. You can also review it under Media in the Web UI.`,
        );
      } catch (e: any) {
        notify("tldw_server", e?.message || "Transcription request failed.");
      }
    };

    const deriveStreamIdleTimeout = (
      cfg: any,
      path: string,
      override?: number,
    ) => {
      if (override && override > 0) return override;
      const p = String(path || "");
      const defaultIdle = 45000; // bump default idle timeout to 45s to tolerate slow providers
      if (p.includes("/api/v1/chat/completions")) {
        return Number(cfg?.chatStreamIdleTimeoutMs) > 0
          ? Number(cfg.chatStreamIdleTimeoutMs)
          : Number(cfg?.streamIdleTimeoutMs) > 0
            ? Number(cfg.streamIdleTimeoutMs)
            : defaultIdle;
      }
      return Number(cfg?.streamIdleTimeoutMs) > 0
        ? Number(cfg.streamIdleTimeoutMs)
        : defaultIdle;
    };

    const CHAT_QUEUE_CONCURRENCY = 2;
    const CHAT_BACKOFF_BASE_MS = 1000;
    const CHAT_BACKOFF_MAX_MS = 30_000;
    let chatBackoffUntil = 0;
    let chatBackoffMs = CHAT_BACKOFF_BASE_MS;
    let chatQueueTimer: ReturnType<typeof setTimeout> | null = null;
    let chatActiveCount = 0;
    const chatQueue: Array<{
      run: () => Promise<any>;
      resolve: (value: any) => void;
      reject: (reason?: any) => void;
    }> = [];

    const isChatEndpoint = (path: string): boolean => {
      const raw = String(path || "");
      let pathname = raw;
      if (/^https?:/i.test(raw)) {
        try {
          pathname = new URL(raw).pathname;
        } catch (error) {
          logBackgroundError("parse chat endpoint url", error);
          pathname = raw;
        }
      }
      const normalized = pathname.toLowerCase();
      return (
        normalized.startsWith("/api/v1/chat/") ||
        normalized.startsWith("/api/v1/chats/")
      );
    };

    const updateChatBackoff = (resp: any) => {
      if (!resp || typeof resp.status !== "number") return;
      if (resp.status === 429) {
        const retryDelay =
          typeof resp.retryAfterMs === "number" && resp.retryAfterMs > 0
            ? resp.retryAfterMs
            : chatBackoffMs;
        chatBackoffUntil = Math.max(chatBackoffUntil, Date.now() + retryDelay);
        chatBackoffMs = Math.min(chatBackoffMs * 2, CHAT_BACKOFF_MAX_MS);
        return;
      }
      if (resp.ok) {
        chatBackoffUntil = 0;
        chatBackoffMs = CHAT_BACKOFF_BASE_MS;
      }
    };

    const scheduleChatDrain = () => {
      if (chatQueueTimer) return;
      const delay = Math.max(0, chatBackoffUntil - Date.now());
      chatQueueTimer = setTimeout(() => {
        chatQueueTimer = null;
        drainChatQueue();
      }, delay);
    };

    const drainChatQueue = () => {
      if (chatActiveCount >= CHAT_QUEUE_CONCURRENCY) return;
      if (chatQueue.length === 0) return;
      const now = Date.now();
      if (chatBackoffUntil > now) {
        scheduleChatDrain();
        return;
      }
      const task = chatQueue.shift();
      if (!task) return;
      chatActiveCount += 1;
      task
        .run()
        .then((resp) => {
          chatActiveCount -= 1;
          updateChatBackoff(resp);
          task.resolve(resp);
          drainChatQueue();
        })
        .catch((err) => {
          chatActiveCount -= 1;
          task.reject(err);
          drainChatQueue();
        });
    };

    const enqueueChatRequest = <T>(run: () => Promise<T>): Promise<T> =>
      new Promise((resolve, reject) => {
        chatQueue.push({ run, resolve, reject });
        drainChatQueue();
      });

    const normalizeFileData = (input: any): Uint8Array | null => {
      if (!input) return null;
      if (input instanceof ArrayBuffer) return new Uint8Array(input);
      if (
        typeof SharedArrayBuffer !== "undefined" &&
        input instanceof SharedArrayBuffer
      ) {
        return new Uint8Array(input);
      }
      if (ArrayBuffer.isView(input)) {
        return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
      }
      if (
        typeof input === "object" &&
        input !== null &&
        typeof (input as { byteLength?: number }).byteLength === "number" &&
        typeof (input as { slice?: unknown }).slice === "function" &&
        Object.prototype.toString.call(input) === "[object ArrayBuffer]"
      ) {
        try {
          return new Uint8Array(input as ArrayBuffer);
        } catch (error) {
          logBackgroundError("normalize arraybuffer-like input", error);
          return null;
        }
      }
      // Accept common structured-clone shapes (e.g., { data: [...] })
      if (Array.isArray((input as any)?.data))
        return new Uint8Array((input as any).data);
      if (Array.isArray(input)) return new Uint8Array(input);
      if (typeof input === "string" && input.startsWith("data:")) {
        try {
          const [meta, payload = ""] = input.split(",", 2);
          const isBase64 = /;base64/i.test(meta);
          if (isBase64) {
            const binary = atob(payload);
            const out = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i += 1) {
              out[i] = binary.charCodeAt(i);
            }
            return out;
          }
          const text = decodeURIComponent(payload);
          return new TextEncoder().encode(text);
        } catch (error) {
          logBackgroundError("decode file data url", error);
          return null;
        }
      }
      return null;
    };

    const handleUpload = async (payload: {
      path?: string;
      method?: string;
      headers?: Record<string, string>;
      fields?: Record<string, any>;
      file?: {
        fieldName?: string;
        name?: string;
        type?: string;
        data?:
          | ArrayBuffer
          | Uint8Array
          | { data?: number[] }
          | number[]
          | string;
      };
      files?: Array<{
        fieldName?: string;
        name?: string;
        type?: string;
        data?:
          | ArrayBuffer
          | Uint8Array
          | { data?: number[] }
          | number[]
          | string;
      }>;
      fileFieldName?: string;
      timeoutMs?: number;
      responseType?: "json" | "text" | "arrayBuffer";
      quickIngestSessionId?: string;
      servicePromptConfig?: ServicePromptTargetLock;
      requestId?: string;
    }, requestAbortSignal?: AbortSignal) => {
      const {
        path,
        method = "POST",
        headers: requestHeaders = {},
        fields = {},
        file,
        files,
        fileFieldName,
        responseType,
      } = payload || {};
      const servicePromptConfig = readServicePromptTargetLock(
        payload?.servicePromptConfig,
      );
      if (
        servicePromptConfig &&
        !isServicePromptRequestPath(path, method)
      ) {
        return {
          ok: false,
          status: 400,
          error:
            "A Service Prompt config can only be used with Service Prompt requests.",
        };
      }
      let cfg: any;
      try {
        cfg = servicePromptConfig
          ? await resolveCurrentServicePromptConfig(servicePromptConfig)
          : await getEffectiveConfig();
      } catch (error) {
        if (
          servicePromptConfig &&
          (error as { status?: unknown })?.status === 412
        ) {
          const scopedError = createServicePromptScopeChangedError();
          return {
            ok: false,
            status: 412,
            error: scopedError.message,
            data: scopedError.details,
          };
        }
        throw error;
      }
      const absoluteAccess = evaluateAbsoluteUrlAccess(path, cfg);
      const isAbsolute = absoluteAccess.isAbsolute;
      // Mirror the request-path guard: refuse cross-origin absolute URLs that
      // are not explicitly allowlisted before any fetch or credential use.
      if (absoluteAccess.blocked) {
        return { ok: false, status: 400, error: ABSOLUTE_URL_BLOCK_ERROR };
      }
      const toArrayBuffer = (bytes: Uint8Array): ArrayBuffer => {
        if (bytes.buffer instanceof ArrayBuffer) {
          return bytes.buffer.slice(
            bytes.byteOffset,
            bytes.byteOffset + bytes.byteLength,
          );
        }
        return new Uint8Array(bytes).buffer as ArrayBuffer;
      };
      if (!cfg?.serverUrl && !isAbsolute) {
        return { ok: false, status: 400, error: "tldw server not configured" };
      }
      const baseUrl = cfg?.serverUrl
        ? String(cfg.serverUrl).replace(/\/$/, "")
        : "";
      const url = isAbsolute
        ? path
        : `${baseUrl}${path?.startsWith("/") ? "" : "/"}${path}`;
      try {
        const form = new FormData();
        for (const [k, v] of Object.entries(fields || {})) {
          // Preserve arrays (e.g., urls) instead of stringifying them into JSON blobs
          if (Array.isArray(v)) {
            v.forEach((item) =>
              form.append(
                k,
                typeof item === "string" ? item : JSON.stringify(item),
              ),
            );
          } else {
            form.append(k, typeof v === "string" ? v : JSON.stringify(v));
          }
        }
        const appendUploadFile = (
          item: {
            fieldName?: string;
            name?: string;
            type?: string;
            data?:
              | ArrayBuffer
              | Uint8Array
              | { data?: number[] }
              | number[]
              | string;
          },
          fieldName: string,
          { appendLegacyFileAlias = false }: { appendLegacyFileAlias?: boolean } = {},
        ): { ok: true } | { ok: false; status: number; error: string } => {
          if (item?.data === undefined || item?.data === null) return { ok: true };
          const bytes = normalizeFileData(item.data);
          if (!bytes || bytes.byteLength === 0) {
            return {
              ok: false,
              status: 400,
              error:
                "File data missing or unreadable. Please re-select the file and try again.",
            };
          }
          const blob = new Blob([toArrayBuffer(bytes)], {
            type: item.type || "application/octet-stream",
          });
          const filename = item.name || "file";
          form.append(fieldName, blob, filename);
          if (appendLegacyFileAlias && fieldName !== "file") {
            form.append("file", blob, filename);
          }
          return { ok: true };
        };

        if (Array.isArray(files) && files.length > 0) {
          for (const item of files) {
            const result = appendUploadFile(item, item.fieldName || "files");
            if (!result.ok) return result;
          }
        } else if (file?.data !== undefined && file?.data !== null) {
          const trimmedFieldName =
            typeof fileFieldName === "string" ? fileFieldName.trim() : "";
          if (trimmedFieldName) {
            const result = appendUploadFile(file, trimmedFieldName);
            if (!result.ok) return result;
          } else {
            const result = appendUploadFile(file, "files", {
              appendLegacyFileAlias: true,
            });
            if (!result.ok) return result;
          }
        }
        const headers: Record<string, string> = { ...requestHeaders };
        for (const key of Object.keys(headers)) {
          const normalized = key.toLowerCase();
          if (
            normalized === "authorization" ||
            normalized === "content-type" ||
            normalized === "x-api-key"
          ) {
            delete headers[key];
          }
        }
        // Skip credentials for allowlisted-but-cross-origin absolute URLs,
        // matching request-core's `shouldSkipAuth` behavior.
        if (!absoluteAccess.skipAuth) {
          if (cfg?.authMode === "single-user") {
            const key = (cfg?.apiKey || "").trim();
            if (!key) {
              return {
                ok: false,
                status: 401,
                error:
                  "Add or update your API key in Settings → tldw server, then try again.",
              };
            }
            headers["X-API-KEY"] = key;
          }
          if (cfg?.authMode === "multi-user") {
            const token = (cfg?.accessToken || "").trim();
            if (!token)
              return {
                ok: false,
                status: 401,
                error: "Not authenticated. Please login under Settings > tldw.",
              };
            headers["Authorization"] = `Bearer ${token}`;
          }
          if (cfg?.orgId) {
            headers["X-TLDW-Org-Id"] = String(cfg.orgId);
          }
        }
        const controller = new AbortController();
        const onRequestAbort = () => controller.abort();
        if (requestAbortSignal?.aborted) {
          onRequestAbort();
        } else {
          requestAbortSignal?.addEventListener("abort", onRequestAbort, {
            once: true,
          });
        }
        const quickIngestSessionId = String(
          payload?.quickIngestSessionId || "",
        ).trim();
        registerQuickIngestAbortController(quickIngestSessionId, controller);
        const timeoutMs =
          Number(payload?.timeoutMs) > 0
            ? Number(payload?.timeoutMs)
            : Number(cfg?.uploadRequestTimeoutMs) > 0
              ? Number(cfg.uploadRequestTimeoutMs)
              : Number(cfg?.mediaRequestTimeoutMs) > 0
                ? Number(cfg.mediaRequestTimeoutMs)
                : Number(cfg?.requestTimeoutMs) > 0
                  ? Number(cfg.requestTimeoutMs)
                  : 60000;
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        let resp: Response;
        try {
          resp = await fetch(url, {
            method,
            headers,
            body: form,
            signal: controller.signal,
          });
        } finally {
          clearTimeout(timeout);
          requestAbortSignal?.removeEventListener("abort", onRequestAbort);
          unregisterQuickIngestAbortController(
            quickIngestSessionId,
            controller,
          );
        }
        const contentType = resp.headers.get("content-type") || "";
        let data: any = null;
        const readDefaultBody = async () => {
          if (contentType.includes("application/json"))
            return await resp.json().catch(() => null);
          return await resp.text().catch(() => null);
        };
        if (responseType === "arrayBuffer") {
          data = resp.ok ? await resp.arrayBuffer().catch(() => null) : await readDefaultBody();
        } else if (responseType === "json") {
          data = await resp.json().catch(() => null);
        } else if (responseType === "text") {
          data = await resp.text().catch(() => null);
        } else {
          data = await readDefaultBody();
        }
        const error = resp.ok
          ? undefined
          : formatErrorMessage(data, `Upload failed: ${resp.status}`);
        return { ok: resp.ok, status: resp.status, data, error };
      } catch (e: any) {
        const raw = String(e?.message || "");
        const isAbort = raw.toLowerCase().includes("abort");
        if (
          isAbort &&
          (requestAbortSignal?.aborted ||
            isQuickIngestCancelled(payload?.quickIngestSessionId))
        ) {
          return {
            ok: false,
            status: 499,
            error: "Cancelled by user.",
          };
        }
        return {
          ok: false,
          status: 0,
          error: isAbort
            ? "Upload timed out waiting for the server response. The ingest may still complete."
            : raw || "Upload failed",
        };
      }
    };

    const runTldwRequest = async (
      payload: any,
      requestAbortSignal?: AbortSignal,
    ) => {
      const rawServicePromptConfig = payload?.servicePromptConfig;
      const requestPayload = { ...(payload || {}) };
      delete requestPayload.servicePromptConfig;
      delete requestPayload.requestId;
      delete requestPayload.abortSignal;
      if (requestAbortSignal) {
        requestPayload.abortSignal = requestAbortSignal;
      }
      const servicePromptConfig = readServicePromptTargetLock(
        rawServicePromptConfig,
      );
      if (servicePromptConfig) {
        if (!isServicePromptRequestPath(requestPayload.path, requestPayload.method)) {
          return {
            ok: false,
            status: 400,
            error: "A Service Prompt config can only be used with Service Prompt requests.",
          };
        }
      }
      let originalScopedConfig: Awaited<
        ReturnType<typeof resolveCurrentServicePromptConfig>
      > | undefined;
      try {
        return await tldwRequest(requestPayload, {
          // IMPORTANT: getConfig must fetch fresh config each time it's called
          // (not pre-fetch once), because the config may not be seeded yet when
          // runTldwRequest is first invoked, but may be available on retry. A
          // scoped request freezes only the checked target and resolves current
          // credentials at dispatch, while the expected-user header binds them.
          getConfig: servicePromptConfig
            ? async () => {
                const current = await resolveCurrentServicePromptConfig(
                  servicePromptConfig,
                );
                originalScopedConfig ??= current;
                return current;
              }
            : async () => await getEffectiveConfig(),
          ...(servicePromptConfig ? { useRuntimeAuthOverride: false } : {}),
          refreshAuth: servicePromptConfig
            ? () => refreshScopedAuth(
                servicePromptConfig,
                originalScopedConfig,
              )
            : async () => {
                if (!refreshInFlight) {
                  refreshInFlight = (async () => {
                    try {
                      await tldwAuth.refreshToken();
                    } finally {
                      refreshInFlight = null;
                    }
                  })();
                }
                try {
                  await refreshInFlight;
                } catch (error) {
                  logBackgroundError("refresh auth", error);
                }
              },
        });
      } catch (error) {
        if (servicePromptConfig && (error as { status?: unknown })?.status === 412) {
          const message =
            "The server or authenticated account changed before the request was sent.";
          return {
            ok: false,
            status: 412,
            error: message,
            data: {
              detail: { code: "request_config_scope_changed", message },
            },
          };
        }
        throw error;
      }
    };

    const handleTldwRequest = async (
      payload: any,
      requestAbortSignal?: AbortSignal,
    ) => {
      const path = payload?.path;
      if (isChatEndpoint(String(path || ""))) {
        return enqueueChatRequest(() =>
          runTldwRequest(payload, requestAbortSignal),
        );
      }
      return await runTldwRequest(payload, requestAbortSignal);
    };

    // Best-effort PATCH of a planned conference-collection item. Shared by the
    // live quick-ingest run and the post-restart resume path.
    const patchConferenceCollectionItem = async (
      planned:
        | { collectionId: number; itemId: number }
        | null
        | undefined,
      body: Record<string, unknown>,
      timeoutMs: number,
    ) => {
      if (!planned) return;
      await handleTldwRequest({
        path: `/api/v1/media/collections/${encodeURIComponent(
          String(planned.collectionId),
        )}/items/${encodeURIComponent(String(planned.itemId))}`,
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body,
        timeoutMs,
      }).catch((error) => {
        logBackgroundError("quick ingest collection item patch", error);
      });
    };

    // Cancel the outstanding remote ingest batches tracked by `tracker`.
    const cancelRemoteIngestBatches = async (
      tracker: ReturnType<
        typeof createIngestJobsTracker<QuickIngestRemoteResultMeta>
      >,
      reason: string,
    ) => {
      await tracker.cancelTrackedBatches(async (batchId) => {
        try {
          await handleTldwRequest({
            path: `/api/v1/media/ingest/jobs/cancel?batch_id=${encodeURIComponent(
              batchId,
            )}&reason=${encodeURIComponent(reason || "user_cancelled")}`,
            method: "POST",
            timeoutMs: 10_000,
          });
        } catch (error) {
          logBackgroundError(`cancel quick ingest batch ${batchId}`, error);
        }
      });
    };

    // Poll the remote ingest jobs held by `tracker` to terminal state. This is
    // the single implementation used by both the live quick-ingest run and the
    // resume-after-restart path, so both map results identically.
    const pollRemoteIngestJobs = (opts: {
      tracker: ReturnType<
        typeof createIngestJobsTracker<QuickIngestRemoteResultMeta>
      >;
      ingestTimeoutMs: number;
      isCancelled: () => boolean;
      onPendingJobIds?: (jobIds: number[]) => void;
    }): Promise<any[]> => {
      return pollTrackedIngestJobs({
        tracker: opts.tracker,
        timeoutMs: opts.ingestTimeoutMs,
        pollIntervalMs: 1200,
        isCancelled: opts.isCancelled,
        onCancel: async () => {
          await cancelRemoteIngestBatches(opts.tracker, "user_cancelled");
        },
        onPendingJobIds: opts.onPendingJobIds,
        fetchJob: async (jobId) =>
          (await handleTldwRequest({
            path: `/api/v1/media/ingest/jobs/${jobId}`,
            method: "GET",
            timeoutMs: 4200,
          })) as
            | { ok: boolean; status?: number; data?: any; error?: string }
            | undefined,
        mapRequestError: (item, response) => {
          if (
            isLikelyAuthError(Number(response?.status || 0), response?.error)
          ) {
            return {
              id: item.meta.id,
              status: "error",
              url: item.meta.url,
              fileName: item.meta.fileName,
              type: item.meta.type,
              error: response?.error || "Authentication required.",
              data: undefined,
            };
          }
          return undefined;
        },
        mapCompleted: (item, data) => ({
          id: item.meta.id,
          status: "ok",
          url: item.meta.url,
          fileName: item.meta.fileName,
          type: item.meta.type,
          data,
        }),
        mapCancelled: (item) => ({
          id: item.meta.id,
          status: "error",
          url: item.meta.url,
          fileName: item.meta.fileName,
          type: item.meta.type,
          error: "Cancelled by user.",
          data: undefined,
        }),
        mapFailure: (item, details) => ({
          id: item.meta.id,
          status: "error",
          url: item.meta.url,
          fileName: item.meta.fileName,
          type: item.meta.type,
          error: String(details.error || `Ingest ${details.status || "failed"}`),
          data: details.data,
        }),
      });
    };

    // Resume + finalize a quick-ingest batch's remote-job polling phase from its
    // persisted record after a worker restart. Re-polls ALL originally-submitted
    // jobs (server job status is idempotent, so already-completed jobs return
    // immediately) and re-emits progress + a terminal message so the sidepanel is
    // not left stuck and a post-restart cancel is honored.
    const resumeQuickIngestBatch = async (
      record: PersistedQuickIngestBatch,
    ): Promise<void> => {
      const sessionId = String(record.sessionId || "").trim();
      if (!sessionId) return;
      if (activeQuickIngestBatchSessionIds.has(sessionId)) return;
      activeQuickIngestBatchSessionIds.add(sessionId);

      const isCancelled = () => isQuickIngestCancelled(sessionId);
      const totalCount = record.totalCount;
      const ingestTimeoutMs = Math.max(record.ingestTimeoutMs || 0, 60_000);
      const out: any[] = Array.isArray(record.collectedResults)
        ? record.collectedResults.slice()
        : [];
      let processedCount = record.processedCount;
      const plannedItems = new Map<
        string,
        { collectionId: number; itemId: number; idempotencyKey?: string | null }
      >();
      for (const planned of record.plannedConferenceItems || []) {
        plannedItems.set(planned.key, {
          collectionId: planned.collectionId,
          itemId: planned.itemId,
          idempotencyKey: planned.idempotencyKey ?? null,
        });
      }

      const emitProgress = (result: any) => {
        processedCount += 1;
        void emitBackgroundMessage(undefined, "tldw:quick-ingest/progress", {
          sessionId,
          result,
          processedCount,
          totalCount,
        });
      };

      try {
        const tracker =
          createIngestJobsTracker<QuickIngestRemoteResultMeta>();
        for (const job of record.remoteJobs || []) {
          try {
            tracker.trackJobs(job.batchId, [job.jobId], {
              id: String((job.meta as any)?.id || ""),
              type: String((job.meta as any)?.type || ""),
              url:
                typeof (job.meta as any)?.url === "string"
                  ? (job.meta as any).url
                  : undefined,
              fileName:
                typeof (job.meta as any)?.fileName === "string"
                  ? (job.meta as any).fileName
                  : undefined,
            });
          } catch (error) {
            logBackgroundError(
              `resume quick ingest track job ${job.jobId}`,
              error,
            );
          }
        }

        if (tracker.hasItems()) {
          const remoteResults = await pollRemoteIngestJobs({
            tracker,
            ingestTimeoutMs,
            isCancelled,
          });
          for (const result of remoteResults) {
            await patchConferenceCollectionItem(
              plannedItems.get(String(result?.id || "")),
              {
                status: result?.status === "ok" ? "completed" : "failed",
                media_id:
                  result?.status === "ok"
                    ? extractCompletedIngestJobMediaId(result?.data)
                    : undefined,
                error_summary:
                  result?.status === "ok"
                    ? undefined
                    : String(result?.error || "Ingest failed"),
              },
              ingestTimeoutMs,
            );
            out.push(result);
            emitProgress(result);
          }
        }

        if (isCancelled()) {
          await emitBackgroundMessage(undefined, "tldw:quick-ingest/cancelled", {
            sessionId,
            reason: "user_cancelled",
            jobIds: (record.remoteJobs || []).map((job) => job.jobId),
          });
        } else {
          await emitBackgroundMessage(undefined, "tldw:quick-ingest/completed", {
            sessionId,
            results: out,
            summary: { resumedAfterRestart: true },
          });
        }
      } catch (error) {
        logBackgroundError(`resume quick ingest ${sessionId}`, error);
        await emitBackgroundMessage(undefined, "tldw:quick-ingest/failed", {
          sessionId,
          error:
            error instanceof Error
              ? error.message
              : String(error || "Quick ingest failed."),
        });
      } finally {
        quickIngestBatchRecords.delete(sessionId);
        quickIngestModalSessions.delete(sessionId);
        activeQuickIngestBatchSessionIds.delete(sessionId);
        void persistSessionState();
        void scheduleQuickIngestBatchAlarm();
      }
    };

    // On worker restart / alarm fire: resume any persisted remote-job polling
    // batch. Safe to call repeatedly — the active-session guard prevents a live
    // or in-flight resume from being started twice.
    const resumeQuickIngestBatches = async (): Promise<void> => {
      for (const record of Array.from(quickIngestBatchRecords.values())) {
        if (activeQuickIngestBatchSessionIds.has(record.sessionId)) continue;
        void resumeQuickIngestBatch(record);
      }
      await scheduleQuickIngestBatchAlarm();
    };

    // Report-as-interrupted the specified quick-ingest modal sessions that were
    // rehydrated from persisted state but have NO resumable remote-job batch —
    // their multipart upload phase was killed mid-flight (no live fetch to
    // resume), so we emit a terminal message to unblock the sidepanel. Only ever
    // called from rehydrate (once per worker) and only for previous-worker
    // sessions, so it never races a freshly-started or completing live batch.
    const reportInterruptedQuickIngestUploads = async (
      sessionIds: string[],
    ): Promise<void> => {
      for (const sessionId of sessionIds) {
        if (quickIngestBatchRecords.has(sessionId)) continue;
        if (activeQuickIngestBatchSessionIds.has(sessionId)) continue;
        const session = quickIngestModalSessions.get(sessionId);
        if (!session) continue;
        activeQuickIngestBatchSessionIds.add(sessionId);
        try {
          if (session.cancelled) {
            await emitBackgroundMessage(
              undefined,
              "tldw:quick-ingest/cancelled",
              { sessionId, reason: "user_cancelled", jobIds: [] },
            );
          } else {
            await emitBackgroundMessage(undefined, "tldw:quick-ingest/failed", {
              sessionId,
              error:
                "Quick ingest was interrupted by a browser/service-worker restart before it finished.",
            });
          }
        } finally {
          quickIngestModalSessions.delete(sessionId);
          activeQuickIngestBatchSessionIds.delete(sessionId);
          void persistSessionState();
        }
      }
    };

    // Regular ingest sessions restored from a previous worker that are still in a
    // pre-submission state (queued/running, no jobIds, not awaiting auth) were
    // interrupted before their non-resumable job submission finished.
    // resumeActiveIngestSessions() cannot resume them (there are no jobIds to
    // poll), so report each once as failed/interrupted and clear it — otherwise
    // the sidepanel is left stuck on "Checking for existing media…" forever.
    // Mirrors reportInterruptedQuickIngestUploads for the no-remote-batch case.
    const reportInterruptedIngestSessions = async (
      funnelIds: string[],
    ): Promise<void> => {
      for (const funnelId of funnelIds) {
        const session = ingestSessions.get(funnelId);
        if (!session) continue;
        if (activeIngestPollFunnelIds.has(funnelId)) continue;
        // Re-check the live session in case it advanced since the snapshot.
        if (session.status !== "queued" && session.status !== "running") {
          continue;
        }
        if (session.awaitingAuth) continue;
        if (Array.isArray(session.jobIds) && session.jobIds.length > 0) continue;
        const msg =
          "Ingest was interrupted by a browser/service-worker restart before it finished.";
        session.lastError = msg;
        await emitIngestStatus(session, {
          status: "failed",
          error: msg,
          progressMessage: msg,
          canCancel: false,
          canRetry: false,
        });
        ingestSessions.delete(funnelId);
        void persistSessionState();
      }
    };

    const extractMediaIdFromJobStatus = (data: any): number | null => {
      const direct = Number(data?.media_id);
      if (Number.isFinite(direct) && direct > 0) {
        return Math.trunc(direct);
      }
      const nested = Number(data?.result?.media_id);
      if (Number.isFinite(nested) && nested > 0) {
        return Math.trunc(nested);
      }
      const completedMediaId = Number(extractCompletedIngestJobMediaId(data));
      if (Number.isFinite(completedMediaId) && completedMediaId > 0) {
        return Math.trunc(completedMediaId);
      }
      return null;
    };

    const extractMediaIdFromAddResponse = (data: any): number | null => {
      if (!data || typeof data !== "object") return null;
      const root = data as Record<string, unknown>;
      const fromRoot = pickMediaIdFromAny(root);
      if (fromRoot != null) return fromRoot;
      const rows = Array.isArray(root.results) ? root.results : [];
      for (const row of rows) {
        const mediaId = pickMediaIdFromAny(row);
        if (mediaId != null) return mediaId;
      }
      return pickMediaIdFromAny(root.result);
    };

    const isQueuedStatus = (value: string): boolean =>
      value === "queued" || value === "pending";

    const isRunningStatus = (value: string): boolean =>
      value === "running" || value === "in_progress" || value === "processing";

    const pollIngestJobsForSession = async (
      session: IngestSession,
      opts?: { timeoutMs?: number; intervalMs?: number },
    ): Promise<{
      mediaId: number | null;
      finalStatus:
        | "completed"
        | "failed"
        | "cancelled"
        | "auth_required"
        | "timeout";
      error?: string;
    }> => {
      const timeoutMs = Math.max(
        10_000,
        Number(opts?.timeoutMs) || 5 * 60 * 1000,
      );
      const intervalMs = Math.max(500, Number(opts?.intervalMs) || 1200);
      const deadline = Date.now() + timeoutMs;
      const unresolved = new Set(session.jobIds.map((id) => Math.trunc(id)));
      let lastStatus: IngestLifecycleStatus | null = null;
      let lastProgressPercent: number | null = null;
      let lastProgressMessage = "";
      let finalStatus: "failed" | "cancelled" = "failed";
      let finalError = "";

      while (unresolved.size > 0 && Date.now() < deadline) {
        const activeSession = ingestSessions.get(session.funnelId);
        if (!activeSession || activeSession.status === "cancelled") {
          return { mediaId: null, finalStatus: "cancelled" };
        }

        let sawPending = false;
        let anyQueued = false;
        let anyRunning = false;
        let anyFailed = false;
        let anyCancelled = false;
        let maxProgressPercent = 0;
        let progressMessage = "";
        for (const jobId of Array.from(unresolved)) {
          const resp = (await handleTldwRequest({
            path: `/api/v1/media/ingest/jobs/${jobId}`,
            method: "GET",
            timeoutMs: intervalMs + 3000,
          })) as
            | { ok: boolean; status?: number; data?: any; error?: string }
            | undefined;
          if (!resp?.ok) {
            if (isLikelyAuthError(Number(resp?.status || 0), resp?.error)) {
              return {
                mediaId: null,
                finalStatus: "auth_required",
                error: resp?.error || "Authentication required.",
              };
            }
            finalError = resp?.error || finalError;
            sawPending = true;
            continue;
          }
          const status = String(resp.data?.status || "").toLowerCase();
          const completedWithFailure =
            status === "completed" && completedIngestJobIndicatesFailure(resp.data);
          if (!completedWithFailure) {
            const mediaId = extractMediaIdFromJobStatus(resp.data);
            if (mediaId != null) {
              return { mediaId, finalStatus: "completed" };
            }
          }
          if (isQueuedStatus(status)) anyQueued = true;
          else if (isRunningStatus(status)) anyRunning = true;
          else if (
            completedWithFailure ||
            status === "failed" ||
            status === "quarantined"
          )
            anyFailed = true;
          else if (status === "cancelled") anyCancelled = true;
          const progressPercent = Number(resp.data?.progress_percent);
          if (
            Number.isFinite(progressPercent) &&
            progressPercent > maxProgressPercent
          ) {
            maxProgressPercent = Math.min(100, Math.max(0, progressPercent));
          }
          if (!progressMessage) {
            const candidateMessage = String(
              (completedWithFailure
                ? extractCompletedIngestJobError(resp.data)
                : undefined) ||
                resp.data?.progress_message ||
                "",
            ).trim();
            if (candidateMessage) progressMessage = candidateMessage;
          }
          if (completedWithFailure) {
            finalError =
              extractCompletedIngestJobError(resp.data) || finalError;
          }

          if (TERMINAL_INGEST_JOB_STATUSES.has(status)) {
            unresolved.delete(jobId);
          } else {
            sawPending = true;
          }
        }

        let statusUpdate: IngestLifecycleStatus = "queued";
        if (anyRunning) {
          statusUpdate = "running";
        } else if (!anyRunning && !anyQueued && anyFailed) {
          statusUpdate = "failed";
        } else if (!anyRunning && !anyQueued && anyCancelled && !anyFailed) {
          statusUpdate = "cancelled";
        } else if (anyQueued || sawPending) {
          statusUpdate = "queued";
        }

        if (
          statusUpdate !== lastStatus ||
          maxProgressPercent !== lastProgressPercent ||
          progressMessage !== lastProgressMessage
        ) {
          await emitIngestStatus(session, {
            status: statusUpdate,
            progressPercent: Number.isFinite(maxProgressPercent)
              ? maxProgressPercent
              : undefined,
            progressMessage: progressMessage || undefined,
          });
          lastStatus = statusUpdate;
          lastProgressPercent = Number.isFinite(maxProgressPercent)
            ? maxProgressPercent
            : null;
          lastProgressMessage = progressMessage;
        }

        if (!sawPending && unresolved.size === 0) {
          if (anyFailed) {
            finalStatus = "failed";
          } else if (anyCancelled && !anyFailed) {
            finalStatus = "cancelled";
          }
          break;
        }
        if (!sawPending) break;
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
      }

      if (Date.now() >= deadline && unresolved.size > 0) {
        return {
          mediaId: null,
          finalStatus: "timeout",
          error: "Timed out while waiting for media ingest jobs.",
        };
      }

      return {
        mediaId: null,
        finalStatus,
        error: finalError || undefined,
      };
    };

    const cancelIngestSessionById = async (
      funnelId: string,
      reason?: string,
    ): Promise<{ ok: boolean; error?: string }> => {
      const session = ingestSessions.get(funnelId);
      if (!session) return { ok: false, error: "Ingest session not found." };
      pendingAuthReplay.delete(funnelId);
      session.awaitingAuth = false;
      session.status = "cancelled";
      await emitIngestStatus(session, {
        status: "cancelled",
        progressMessage: "Cancelling ingest...",
        canCancel: false,
        canRetry: true,
      });
      if (session.batchId) {
        await handleTldwRequest({
          path: `/api/v1/media/ingest/jobs/cancel?batch_id=${encodeURIComponent(
            session.batchId,
          )}&reason=${encodeURIComponent(reason || "user_cancelled")}`,
          method: "POST",
          timeoutMs: 10_000,
        }).catch((error) => {
          logBackgroundError(`cancel ingest batch ${session.batchId}`, error);
        });
      } else {
        for (const jobId of session.jobIds) {
          await handleTldwRequest({
            path: `/api/v1/media/ingest/jobs/${jobId}?reason=${encodeURIComponent(
              reason || "user_cancelled",
            )}`,
            method: "DELETE",
            timeoutMs: 8000,
          }).catch((error) => {
            logBackgroundError(`cancel ingest job ${jobId}`, error);
          });
        }
      }
      await emitIngestStatus(session, {
        status: "cancelled",
        progressMessage: "Ingest cancelled.",
        canCancel: false,
        canRetry: true,
      });
      return { ok: true };
    };

    // Terminal-state handling for an ingest poll result. Extracted so both the
    // inline (live-worker) poll and the alarm/rehydrate resume path share the
    // exact same completion / auth / cancel / failure behavior.
    const finalizeIngestSession = async (
      session: IngestSession,
      pollResult: Awaited<ReturnType<typeof pollIngestJobsForSession>>,
    ): Promise<void> => {
      if (pollResult.finalStatus === "completed" && pollResult.mediaId != null) {
        session.mediaId = pollResult.mediaId;
        await appendIngestFunnelMetric("media_completed", session.funnelId, {
          mediaId: pollResult.mediaId,
          deduped: false,
        });
        await emitIngestStatus(session, {
          status: "completed",
          mediaId: pollResult.mediaId,
          progressPercent: 100,
          progressMessage: "Ingest complete. Opening media-scoped chat.",
          canCancel: false,
          canRetry: false,
        });
        await sendIngestReadyMessage(session.tabId, {
          funnelId: session.funnelId,
          mediaId: String(pollResult.mediaId),
          url: session.url,
          mode: "rag_media",
          timestampSeconds:
            typeof session.timestampSeconds === "number" &&
            session.timestampSeconds >= 0
              ? session.timestampSeconds
              : undefined,
        });
        notify("tldw_server", "Ready. Opened media-scoped chat in sidebar.");
      } else if (pollResult.finalStatus === "auth_required") {
        await queueAuthRecovery(
          session,
          pollResult.error || "Authentication required.",
        );
      } else if (pollResult.finalStatus === "cancelled") {
        await emitIngestStatus(session, {
          status: "cancelled",
          progressMessage: "Ingest cancelled.",
          canCancel: false,
          canRetry: true,
        });
        notify("tldw_server", "Ingest was cancelled.");
      } else {
        const failureMessage =
          pollResult.error ||
          (pollResult.finalStatus === "timeout"
            ? "Ingest timed out. Retry or open Media to inspect job status."
            : "No completed media yet. Open Media to check job status.");
        session.lastError = failureMessage;
        await emitIngestStatus(session, {
          status: "failed",
          error: failureMessage,
          progressMessage: failureMessage,
          canCancel: false,
          canRetry: true,
        });
        notify("tldw_server", failureMessage);
      }
      // Reconcile the poll alarm now that this session has reached a terminal
      // state (clears the alarm once no active ingest remains).
      await scheduleIngestPollAlarm();
    };

    const startContextMenuIngest = async (
      session: IngestSession,
      options?: {
        trackContextClick?: boolean;
        reason?: "initial" | "manual_retry" | "auth_replay";
      },
    ) => {
      ingestSessions.set(session.funnelId, session);
      session.awaitingAuth = false;
      session.lastError = undefined;
      session.mediaId = undefined;
      session.jobIds = [];
      session.batchId = undefined;

      if (options?.trackContextClick) {
        await appendIngestFunnelMetric("context_click", session.funnelId, {
          url: session.url,
        });
      }

      await emitIngestStatus(session, {
        status: "queued",
        progressMessage: "Checking for existing media...",
        canCancel: false,
        canRetry: false,
      });

      let existingMediaId: number | null = null;
      try {
        existingMediaId = await findExistingMediaForUrl(
          session.url,
          session.normalizedUrl,
        );
      } catch (error) {
        const msg = formatErrorMessage(error, "Authentication required.");
        await queueAuthRecovery(session, msg);
        return;
      }

      if (existingMediaId != null) {
        session.mediaId = existingMediaId;
        await appendIngestFunnelMetric("media_completed", session.funnelId, {
          mediaId: existingMediaId,
          deduped: true,
        });
        await emitIngestStatus(session, {
          status: "completed",
          mediaId: existingMediaId,
          progressPercent: 100,
          progressMessage:
            "Already in your library. Opening media-scoped chat.",
          canCancel: false,
          canRetry: false,
        });
        await sendIngestReadyMessage(session.tabId, {
          funnelId: session.funnelId,
          mediaId: String(existingMediaId),
          url: session.url,
          mode: "rag_media",
          timestampSeconds:
            typeof session.timestampSeconds === "number" &&
            session.timestampSeconds >= 0
              ? session.timestampSeconds
              : undefined,
        });
        notify(
          "tldw_server",
          "Already ingested. Opened media-scoped chat in sidebar.",
        );
        return;
      }

      const addPayload = buildContextMenuAddPayload(session.url);
      const jobsResp = await handleUpload({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        fields: addPayload.fields,
        timeoutMs: 180000,
      });
      if (!jobsResp?.ok) {
        if (isLikelyAuthError(Number(jobsResp?.status || 0), jobsResp?.error)) {
          await queueAuthRecovery(
            session,
            jobsResp?.error || "Authentication required.",
          );
          return;
        }
        const addResp = await handleUpload(addPayload);
        if (!addResp?.ok) {
          if (isLikelyAuthError(Number(addResp?.status || 0), addResp?.error)) {
            await queueAuthRecovery(
              session,
              addResp?.error || "Authentication required.",
            );
            return;
          }
          const msg = addResp?.error || jobsResp?.error || "Ingest failed";
          session.lastError = msg;
          await emitIngestStatus(session, {
            status: "failed",
            error: msg,
            progressMessage: msg,
            canCancel: false,
            canRetry: true,
          });
          notify("tldw_server", msg);
          return;
        }
        const mediaId = extractMediaIdFromAddResponse(addResp.data);
        if (mediaId == null) {
          const msg = "Ingest completed but media id was not returned.";
          session.lastError = msg;
          await emitIngestStatus(session, {
            status: "failed",
            error: msg,
            canCancel: false,
            canRetry: true,
          });
          notify("tldw_server", msg);
          return;
        }
        session.mediaId = mediaId;
        await appendIngestFunnelMetric("media_completed", session.funnelId, {
          mediaId,
          fallback: true,
        });
        await emitIngestStatus(session, {
          status: "completed",
          mediaId,
          progressPercent: 100,
          progressMessage: "Ingest complete. Opening media-scoped chat.",
          canCancel: false,
          canRetry: false,
        });
        await sendIngestReadyMessage(session.tabId, {
          funnelId: session.funnelId,
          mediaId: String(mediaId),
          url: session.url,
          mode: "rag_media",
          timestampSeconds:
            typeof session.timestampSeconds === "number" &&
            session.timestampSeconds >= 0
              ? session.timestampSeconds
              : undefined,
        });
        notify("tldw_server", "Ready. Opened media-scoped chat in sidebar.");
        return;
      }

      try {
        const submittedJobs = requireSubmittedIngestJobs(jobsResp.data);
        session.jobIds = submittedJobs.jobIds;
        session.batchId = submittedJobs.batchId;
      } catch (error) {
        const msg =
          error instanceof Error
            ? error.message
            : String(error || "Ingest job submission returned no job IDs.");
        session.lastError = msg;
        await emitIngestStatus(session, {
          status: "failed",
          error: msg,
          canCancel: false,
          canRetry: true,
        });
        notify("tldw_server", msg);
        return;
      }

      await appendIngestFunnelMetric("job_queued", session.funnelId, {
        jobIds: session.jobIds,
        url: session.url,
      });

      await emitIngestStatus(session, {
        status: "queued",
        progressPercent: 0,
        progressMessage: "Queued for processing.",
        canCancel: true,
        canRetry: false,
      });
      notify(
        "tldw_server",
        "Queued for processing. Preparing chat when ready…",
      );

      // Register an alarm backstop before polling so that, if Chrome suspends
      // the worker mid-poll, the alarm resumes polling on wake (TASK-12094).
      activeIngestPollFunnelIds.add(session.funnelId);
      void scheduleIngestPollAlarm();
      let pollResult: Awaited<ReturnType<typeof pollIngestJobsForSession>>;
      try {
        pollResult = await pollIngestJobsForSession(session, {
          timeoutMs: 10 * 60 * 1000,
          intervalMs: 1500,
        });
      } finally {
        activeIngestPollFunnelIds.delete(session.funnelId);
      }
      await finalizeIngestSession(session, pollResult);
    };

    const retryIngestSessionById = async (
      funnelId: string,
      reason: "manual_retry" | "auth_replay",
    ): Promise<{ ok: boolean; error?: string }> => {
      const session = ingestSessions.get(funnelId);
      if (!session) return { ok: false, error: "Ingest session not found." };
      if (session.status === "queued" || session.status === "running") {
        return { ok: false, error: "Ingest is already in progress." };
      }
      pendingAuthReplay.delete(funnelId);
      void persistSessionState();
      session.retryCount += 1;
      void startContextMenuIngest(session, {
        trackContextClick: false,
        reason,
      });
      return { ok: true };
    };

    const replayPendingAuthSessions = async () => {
      if (pendingAuthReplay.size === 0) return;
      const cfg = await getEffectiveConfig();
      if (!hasUsableAuthConfig(cfg)) return;
      for (const funnelId of Array.from(pendingAuthReplay)) {
        pendingAuthReplay.delete(funnelId);
        const session = ingestSessions.get(funnelId);
        if (!session) continue;
        session.awaitingAuth = false;
        void startContextMenuIngest(session, {
          trackContextClick: false,
          reason: "auth_replay",
        });
      }
      void persistSessionState();
    };

    // Re-poll a rehydrated ingest session (after a worker restart / alarm wake)
    // using its persisted jobIds, then run the shared terminal handling.
    const resumeIngestSessionPoll = async (
      session: IngestSession,
    ): Promise<void> => {
      if (activeIngestPollFunnelIds.has(session.funnelId)) return;
      if (session.status !== "queued" && session.status !== "running") return;
      if (session.awaitingAuth) return;
      if (!Array.isArray(session.jobIds) || session.jobIds.length === 0) return;
      activeIngestPollFunnelIds.add(session.funnelId);
      try {
        const pollResult = await pollIngestJobsForSession(session, {
          timeoutMs: 10 * 60 * 1000,
          intervalMs: 1500,
        });
        await finalizeIngestSession(session, pollResult);
      } catch (error) {
        logBackgroundError(`resume ingest ${session.funnelId}`, error);
      } finally {
        activeIngestPollFunnelIds.delete(session.funnelId);
      }
    };

    const resumeActiveIngestSessions = async (): Promise<void> => {
      const candidates = Array.from(ingestSessions.values()).filter(
        (session) =>
          (session.status === "queued" || session.status === "running") &&
          !session.awaitingAuth &&
          Array.isArray(session.jobIds) &&
          session.jobIds.length > 0 &&
          !activeIngestPollFunnelIds.has(session.funnelId),
      );
      for (const session of candidates) {
        void resumeIngestSessionPoll(session);
      }
      await scheduleIngestPollAlarm();
    };

    // Rehydrate persisted session state after a worker restart, then resume any
    // interrupted ingest polling and replay pending 401 auth-recoveries. Runs at
    // most once per worker lifetime.
    const rehydrateSessionState = async (): Promise<void> => {
      if (sessionStateHydrated) return;
      sessionStateHydrated = true;
      let state;
      try {
        state = await readPersistedSessionState();
      } catch (error) {
        logBackgroundError("read session state", error);
        return;
      }
      for (const [funnelId, value] of Object.entries(state.ingestSessions)) {
        if (!ingestSessions.has(funnelId)) {
          ingestSessions.set(funnelId, value as unknown as IngestSession);
        }
      }
      for (const funnelId of state.pendingAuthReplay) {
        pendingAuthReplay.add(funnelId);
      }
      for (const entry of state.quickIngestSessions) {
        if (!quickIngestModalSessions.has(entry.sessionId)) {
          quickIngestModalSessions.set(entry.sessionId, {
            sessionId: entry.sessionId,
            cancelled: entry.cancelled,
            abortControllers: new Set(),
          });
        }
      }
      for (const batch of state.quickIngestBatches) {
        if (!quickIngestBatchRecords.has(batch.sessionId)) {
          quickIngestBatchRecords.set(batch.sessionId, batch);
        }
      }
      // Quick-ingest modal sessions restored from a previous worker that have no
      // resumable remote-job batch were interrupted during their (non-resumable)
      // upload phase — report those once so the sidepanel is not left stuck.
      const interruptedUploadSessionIds = state.quickIngestSessions
        .map((entry) => entry.sessionId)
        .filter((sessionId) => !quickIngestBatchRecords.has(sessionId));
      // Snapshot the ingest sessions that were interrupted mid pre-submission
      // (queued/running with no jobIds and not awaiting auth) up front, before
      // the auth-replay path can begin re-driving any of them.
      const interruptedIngestFunnelIds = selectInterruptedIngestFunnelIds(
        state.ingestSessions,
        state.pendingAuthReplay,
      );
      await resumeActiveIngestSessions();
      void replayPendingAuthSessions();
      await resumeQuickIngestBatches();
      await reportInterruptedQuickIngestUploads(interruptedUploadSessionIds);
      await reportInterruptedIngestSessions(interruptedIngestFunnelIds);
    };

    const runQuickIngestBatch = async (
      payload: any,
      runtimeContext?: QuickIngestSessionRunContext,
    ): Promise<{ ok: boolean; results: any[] }> => {
      const entries = Array.isArray(payload?.entries)
        ? payload.entries.filter(
            (entry: any) => entry?.conferenceOverride?.selected !== false,
          )
        : [];
      const files = Array.isArray(payload?.files)
        ? payload.files.filter(
            (file: any) => file?.conferenceOverride?.selected !== false,
          )
        : [];
      const storeRemote = Boolean(payload?.storeRemote);
      const processOnly = Boolean(payload?.processOnly);
      const common = payload?.common || {};
      const advancedValues =
        payload?.advancedValues && typeof payload.advancedValues === "object"
          ? payload.advancedValues
          : {};
      const fileDefaults =
        payload?.fileDefaults && typeof payload.fileDefaults === "object"
          ? payload.fileDefaults
          : {};
      const chunkingTemplateName =
        typeof payload?.chunkingTemplateName === "string"
          ? payload.chunkingTemplateName
          : undefined;
      const autoApplyTemplate = Boolean(payload?.autoApplyTemplate);
      const shouldStoreRemote = storeRemote && !processOnly;
      const sessionId =
        String(
          runtimeContext?.sessionId || payload?.__quickIngestSessionId || "",
        ).trim() || undefined;

      const cfg = await storage.get<any>("tldwConfig");
      const ingestTimeoutMs = Math.max(
        Number(cfg?.uploadRequestTimeoutMs) || 0,
        Number(cfg?.mediaRequestTimeoutMs) || 0,
        Number(cfg?.requestTimeoutMs) || 0,
        5 * 60 * 1000,
      );
      const totalCount = entries.length + files.length;
      let processedCount = 0;
      const out: any[] = [];
      const queuedRemoteJobs =
        createIngestJobsTracker<QuickIngestRemoteResultMeta>();
      const conferenceBatchMetadata =
        payload?.conferenceBatchMetadata &&
        typeof payload.conferenceBatchMetadata === "object"
          ? payload.conferenceBatchMetadata
          : null;
      type PlannedConferenceCollectionItem = {
        collectionId: number;
        itemId: number;
        idempotencyKey?: string | null;
      };
      const plannedConferenceItems = new Map<
        string,
        PlannedConferenceCollectionItem
      >();

      const isCancelled = () =>
        Boolean(
          runtimeContext?.isCancelled?.() || isQuickIngestCancelled(sessionId),
        );

      const assignPath = (obj: any, path: string[], val: any) => {
        let cur = obj;
        for (let i = 0; i < path.length; i++) {
          const seg = path[i];
          if (i === path.length - 1) cur[seg] = val;
          else cur = cur[seg] = cur[seg] || {};
        }
      };

      const buildFields = (rawType: string, entry?: any, defaults?: any) => {
        const mediaType = normalizeMediaType(rawType);
        const fields: Record<string, any> = {
          media_type: mediaType,
          perform_analysis: Boolean(common.perform_analysis),
          overwrite_existing: Boolean(common.overwrite_existing),
        };
        const resolvedDefaults: {
          audio?: { language?: string; diarize?: boolean };
          document?: { ocr?: boolean };
          video?: { captions?: boolean };
        } = (() => {
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
        const nested: Record<string, any> = {};
        for (const [k, v] of Object.entries(
          advancedValues as Record<string, any>,
        )) {
          if (!shouldSubmitQuickIngestAdvancedField(k, common)) continue;
          if (k.includes(".")) assignPath(nested, k.split("."), v);
          else fields[k] = v;
        }
        for (const [k, v] of Object.entries(nested)) fields[k] = v;
        if (typeof entry?.keywords === "string") {
          const trimmed = entry.keywords.trim();
          if (trimmed) {
            fields.keywords = trimmed;
          }
        }
        const audio = {
          ...(resolvedDefaults.audio || {}),
          ...(entry?.audio || {}),
        };
        const video = {
          ...(resolvedDefaults.video || {}),
          ...(entry?.video || {}),
        };
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
        if (
          typeof video.captions === "boolean" &&
          fields.timestamp_option == null
        ) {
          fields.timestamp_option = video.captions;
        }
        if (
          typeof document.ocr === "boolean" &&
          fields.pdf_parsing_engine == null
        ) {
          fields.pdf_parsing_engine = document.ocr ? "pymupdf4llm" : "";
        }
        applyQuickIngestChunkingFields(fields, {
          common,
          chunkingTemplateName,
          autoApplyTemplate,
        });
        return fields;
      };

      const hasConferenceMetadata = (): boolean =>
        Boolean(
          conferenceBatchMetadata &&
            (
              String(conferenceBatchMetadata.collectionName || "").trim() ||
              String(conferenceBatchMetadata.conferenceName || "").trim() ||
              String(conferenceBatchMetadata.sourcePlaylistUrl || "").trim() ||
              (Array.isArray(conferenceBatchMetadata.sharedTags) &&
                conferenceBatchMetadata.sharedTags.length > 0)
            ),
        );

      const getConferenceFallbackName = (): string | null => {
        for (const entry of entries) {
          const title = String(entry?.playlist?.playlistTitle || "").trim();
          if (title) return title;
        }
        return null;
      };

      const createPlannedConferenceItems = async () => {
        if (!shouldStoreRemote || !hasConferenceMetadata()) return;
        const selectedEntries = entries.filter((entry: any) =>
          String(entry?.url || "").trim(),
        );
        if (selectedEntries.length === 0) return;

        let collectionId: number | null = null;
        try {
          const collectionResp = (await handleTldwRequest({
            path: "/api/v1/media/collections",
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: buildConferenceCollectionCreatePayload(
              conferenceBatchMetadata,
              getConferenceFallbackName(),
            ),
            timeoutMs: ingestTimeoutMs,
          })) as
            | { ok: boolean; error?: string; status?: number; data?: any }
            | undefined;
          if (!collectionResp?.ok) {
            throw new Error(
              collectionResp?.error ||
                `Collection creation failed: ${collectionResp?.status}`,
            );
          }
          const parsedCollectionId = Number(collectionResp.data?.id);
          if (!Number.isFinite(parsedCollectionId) || parsedCollectionId <= 0) {
            throw new Error("Collection creation returned no collection id.");
          }
          collectionId = parsedCollectionId;
        } catch (error) {
          console.debug("[tldw] conference collection planning failed", error);
          return;
        }
        if (collectionId === null) return;

        for (const entry of selectedEntries) {
          try {
            const itemResp = (await handleTldwRequest({
              path: `/api/v1/media/collections/${encodeURIComponent(
                String(collectionId),
              )}/items`,
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: buildConferenceCollectionItemPayload(conferenceBatchMetadata, {
                id: String(entry.id || ""),
                url: String(entry.url || ""),
                playlist: entry.playlist,
                conferenceOverride: entry.conferenceOverride,
              }),
              timeoutMs: ingestTimeoutMs,
            })) as
              | { ok: boolean; error?: string; status?: number; data?: any }
              | undefined;
            if (!itemResp?.ok) {
              throw new Error(
                itemResp?.error || `Collection item failed: ${itemResp?.status}`,
              );
            }
            const itemId = Number(itemResp.data?.id);
            if (Number.isFinite(itemId) && itemId > 0) {
              plannedConferenceItems.set(String(entry.id || ""), {
                collectionId,
                itemId,
                idempotencyKey:
                  typeof itemResp.data?.idempotency_key === "string"
                    ? itemResp.data.idempotency_key
                    : null,
              });
            }
          } catch (error) {
            console.debug(
              "[tldw] conference collection item planning failed",
              { collectionId, entryId: entry?.id },
              error,
            );
          }
        }
      };

      const patchPlannedConferenceItem = async (
        planned: PlannedConferenceCollectionItem | undefined,
        body: Record<string, unknown>,
      ) => {
        await patchConferenceCollectionItem(planned, body, ingestTimeoutMs);
      };

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

      const processWebScrape = async (url: string, entry?: any) => {
        const nestedBody: Record<string, any> = {};
        for (const [k, v] of Object.entries(
          advancedValues as Record<string, any>,
        )) {
          if (!shouldSubmitQuickIngestAdvancedField(k, common)) continue;
          if (k.includes(".")) assignPath(nestedBody, k.split("."), v);
          else nestedBody[k] = v;
        }
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
        const normalizedBody: Record<string, any> = { ...nestedBody };
        for (const key of [
          "custom_headers",
          "custom_cookies",
          "custom_titles",
        ]) {
          if (key in normalizedBody) {
            normalizedBody[key] = normalizeJsonField(normalizedBody[key]);
          }
        }
        const body: any = {
          scrape_method: "Individual URLs",
          url_input: url,
          mode: "ephemeral",
          summarize_checkbox: Boolean(common.perform_analysis),
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
        const controller = new AbortController();
        registerQuickIngestAbortController(sessionId, controller);
        runtimeContext?.registerAbortController(controller);
        try {
          const resp = (await handleTldwRequest({
            path: "/api/v1/media/process-web-scraping",
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body,
            timeoutMs: ingestTimeoutMs,
            abortSignal: controller.signal,
          })) as
            | { ok: boolean; error?: string; status?: number; data?: any }
            | undefined;
          if (!resp?.ok) {
            const msg = resp?.error || `Request failed: ${resp?.status}`;
            throw new Error(msg);
          }
          return resp.data;
        } finally {
          unregisterQuickIngestAbortController(sessionId, controller);
        }
      };

      const emitProgress = (result: any) => {
        processedCount += 1;
        const progressPayload = {
          result,
          processedCount,
          totalCount,
        };
        if (runtimeContext) {
          void runtimeContext.emitProgress(progressPayload);
          return;
        }

        try {
          void browser.runtime
            .sendMessage({
              type: "tldw:quick-ingest-progress",
              payload: {
                ...progressPayload,
                sessionId,
              },
            })
            .catch((error) => {
              logBackgroundError("quick ingest progress message", error);
            });
        } catch (error) {
          logBackgroundError("quick ingest progress message", error);
        }
      };

      const toFallbackCandidate = (response: {
        status?: number;
        error?: unknown;
        data?: unknown;
      }) => ({
        status: response?.status,
        error: response?.error,
        details: response?.data,
      });

      const submitPersistentAddFallback = async ({
        fields,
        file,
      }: {
        fields: Record<string, any>;
        file?: {
          name?: string;
          type?: string;
          data?:
            | ArrayBuffer
            | Uint8Array
            | { data?: number[] }
            | number[]
            | string;
        };
      }) => {
        const fallbackResp = await handleUpload({
          path: "/api/v1/media/add",
          method: "POST",
          fields,
          file,
          timeoutMs: ingestTimeoutMs,
          quickIngestSessionId: sessionId,
        });
        if (!fallbackResp?.ok) {
          throw new Error(fallbackResp?.error || "Upload failed");
        }
        return normalizePersistentAddResponse(fallbackResp.data);
      };

      const trackRemoteJobs = (
        submitData: any,
        resultTemplate: QuickIngestRemoteResultMeta,
      ): number => {
        const trackedJobIds = queuedRemoteJobs.trackSubmit(
          submitData,
          resultTemplate,
        );
        runtimeContext?.setJobIds(queuedRemoteJobs.getJobIds());
        return trackedJobIds.length;
      };

      // Poll the queued remote jobs to completion using the shared implementation
      // (also used by the post-restart resume path) so both map results the same.
      const pollQueuedRemoteJobs = async (): Promise<any[]> =>
        pollRemoteIngestJobs({
          tracker: queuedRemoteJobs,
          ingestTimeoutMs,
          isCancelled,
          onPendingJobIds: (jobIds) => {
            runtimeContext?.setJobIds(jobIds);
          },
        });

      // Snapshot the resumable remote-job phase so a worker restart can re-poll
      // and finalize it. Only the queued/remote-job phase is durable — the raw
      // multipart uploads that produced these job ids cannot be resumed.
      const persistQuickIngestBatchRecord = () => {
        if (!sessionId) return;
        quickIngestBatchRecords.set(sessionId, {
          sessionId,
          totalCount,
          processedCount,
          ingestTimeoutMs,
          remoteJobs: queuedRemoteJobs.getItems().map((item) => ({
            jobId: item.jobId,
            batchId: item.batchId,
            meta: item.meta as unknown as Record<string, unknown>,
          })),
          collectedResults: out.slice(),
          plannedConferenceItems: Array.from(
            plannedConferenceItems.entries(),
          ).map(([key, value]) => ({
            key,
            collectionId: value.collectionId,
            itemId: value.itemId,
            idempotencyKey: value.idempotencyKey ?? null,
          })),
        });
        activeQuickIngestBatchSessionIds.add(sessionId);
        void persistSessionState();
        void scheduleQuickIngestBatchAlarm();
      };

      const clearQuickIngestBatchRecord = () => {
        if (!sessionId) return;
        quickIngestBatchRecords.delete(sessionId);
        activeQuickIngestBatchSessionIds.delete(sessionId);
        void persistSessionState();
        void scheduleQuickIngestBatchAlarm();
      };

      await createPlannedConferenceItems();

      for (const r of entries) {
        if (isCancelled()) break;
        const url = String(r?.url || "").trim();
        if (!url) continue;
        const explicitType =
          r?.type && typeof r.type === "string" ? r.type : "auto";
        const t =
          explicitType === "auto" ? inferMediaTypeFromUrl(url) : explicitType;
        const plannedConferenceItem = plannedConferenceItems.get(
          String(r.id || ""),
        );
        let jobSubmitted = false;
        let latestJobId: number | undefined;
        try {
          let data: any;
          if (shouldStoreRemote) {
            const resolvedDefaults =
              r?.defaults && typeof r.defaults === "object"
                ? r.defaults
                : fileDefaults;
            const fields: Record<string, any> = buildFields(
              t,
              r,
              resolvedDefaults,
            );
            applyPlannedConferenceFields(fields, plannedConferenceItem);
            fields.urls = [url];
            const resp = await handleUpload({
              path: "/api/v1/media/ingest/jobs",
              method: "POST",
              fields,
              timeoutMs: ingestTimeoutMs,
              quickIngestSessionId: sessionId,
            });
            if (!resp?.ok) {
              if (!shouldFallbackToPersistentAdd(toFallbackCandidate(resp))) {
                const msg = resp?.error || `Upload failed: ${resp?.status}`;
                throw new Error(msg);
              }
              if (typeof latestJobId === "number") {
                fields.media_ingest_job_id = String(latestJobId);
              }
              data = await submitPersistentAddFallback({ fields });
            } else {
              const trackedCount = trackRemoteJobs(resp.data, {
                id: String(r.id || crypto.randomUUID()),
                url,
                type: t,
              });
              const jobIds = queuedRemoteJobs.getJobIds();
              latestJobId = jobIds[jobIds.length - trackedCount];
              jobSubmitted = trackedCount > 0;
              await patchPlannedConferenceItem(plannedConferenceItem, {
                status: "processing",
                latest_job_id:
                  typeof latestJobId === "number"
                    ? String(latestJobId)
                    : undefined,
              });
              continue;
            }
          } else if (t === "html") {
            data = await processWebScrape(url, r);
          } else {
            const resolvedDefaults =
              r?.defaults && typeof r.defaults === "object"
                ? r.defaults
                : fileDefaults;
            const fields = buildFields(t, r, resolvedDefaults);
            fields.urls = [url];
            const resp = await handleUpload({
              path: getProcessPathForType(t),
              method: "POST",
              fields,
              timeoutMs: ingestTimeoutMs,
              quickIngestSessionId: sessionId,
            });
            if (!resp?.ok) {
              const msg = resp?.error || `Upload failed: ${resp?.status}`;
              throw new Error(msg);
            }
            data = resp.data;
          }
          const result = { id: r.id, status: "ok", url, type: t, data };
          out.push(result);
          emitProgress(result);
        } catch (e: any) {
          if (isCancelled()) break;
          await patchPlannedConferenceItem(plannedConferenceItem, {
            status: jobSubmitted ? "failed" : "submit_failed",
            latest_job_id:
              typeof latestJobId === "number" ? String(latestJobId) : undefined,
            error_summary: e?.message || "Request failed",
          });
          const result = {
            id: r.id,
            status: "error",
            url,
            type: t,
            error: e?.message || "Request failed",
          };
          out.push(result);
          emitProgress(result);
        }
      }

      for (const f of files) {
        if (isCancelled()) break;
        const id = f?.id || crypto.randomUUID();
        const name = f?.name || "upload";
        const mediaType = inferUploadMediaTypeFromFile(name, f?.type);
        const resolvedFileDefaults =
          f?.defaults && typeof f.defaults === "object"
            ? f.defaults
            : fileDefaults;
        try {
          let data: any;
          if (shouldStoreRemote) {
            const fields: Record<string, any> = buildFields(
              mediaType,
              undefined,
              resolvedFileDefaults,
            );
            const resp = await handleUpload({
              path: "/api/v1/media/ingest/jobs",
              method: "POST",
              fields,
              file: {
                name,
                type: f?.type || "application/octet-stream",
                data: f?.data,
              },
              timeoutMs: ingestTimeoutMs,
              quickIngestSessionId: sessionId,
            });
            if (!resp?.ok) {
              if (!shouldFallbackToPersistentAdd(toFallbackCandidate(resp))) {
                const msg = resp?.error || `Upload failed: ${resp?.status}`;
                throw new Error(msg);
              }
              data = await submitPersistentAddFallback({
                fields,
                file: {
                  name,
                  type: f?.type || "application/octet-stream",
                  data: f?.data,
                },
              });
            } else {
              trackRemoteJobs(resp.data, {
                id: String(id),
                fileName: name,
                type: mediaType,
              });
              continue;
            }
          } else {
            const fields: Record<string, any> = buildFields(
              mediaType,
              undefined,
              resolvedFileDefaults,
            );
            const resp = await handleUpload({
              path: getProcessPathForType(mediaType),
              method: "POST",
              fields,
              file: {
                name,
                type: f?.type || "application/octet-stream",
                data: f?.data,
              },
              timeoutMs: ingestTimeoutMs,
              quickIngestSessionId: sessionId,
            });
            if (!resp?.ok) {
              const msg = resp?.error || `Upload failed: ${resp?.status}`;
              throw new Error(msg);
            }
            data = resp.data;
          }
          const result = {
            id,
            status: "ok",
            fileName: name,
            type: mediaType,
            data,
          };
          out.push(result);
          emitProgress(result);
        } catch (e: any) {
          if (isCancelled()) break;
          const result = {
            id,
            status: "error",
            fileName: name,
            type: "file",
            error: e?.message || "Upload failed",
          };
          out.push(result);
          emitProgress(result);
        }
      }

      if (shouldStoreRemote && queuedRemoteJobs.hasItems()) {
        // Persist the resumable remote-job phase before we start polling so a
        // worker suspension mid-poll can be resumed from the alarm/startup path.
        persistQuickIngestBatchRecord();
        try {
          const remoteResults = await pollQueuedRemoteJobs();
          for (const result of remoteResults) {
            await patchPlannedConferenceItem(
              plannedConferenceItems.get(String(result?.id || "")),
              {
                status: result?.status === "ok" ? "completed" : "failed",
                media_id:
                  result?.status === "ok"
                    ? extractCompletedIngestJobMediaId(result?.data)
                    : undefined,
                error_summary:
                  result?.status === "ok"
                    ? undefined
                    : String(result?.error || "Ingest failed"),
              },
            );
            out.push(result);
            emitProgress(result);
          }
        } finally {
          // The live poll finished (or threw) in this worker; drop the durable
          // record so the alarm/startup path does not re-poll it.
          clearQuickIngestBatchRecord();
        }
      }

      return { ok: true, results: out };
    };

    const quickIngestSessionRuntime = createQuickIngestSessionRuntime({
      run: async (payload, context) => {
        const result = await runQuickIngestBatch(payload, context);
        return {
          results: result.results,
        };
      },
      emit: async (type, payload) => {
        await emitBackgroundMessage(undefined, type, payload);
        if (
          type === "tldw:quick-ingest/completed" ||
          type === "tldw:quick-ingest/failed" ||
          type === "tldw:quick-ingest/cancelled"
        ) {
          const sessionId = String((payload as any)?.sessionId || "").trim();
          if (sessionId) {
            quickIngestModalSessions.delete(sessionId);
            void persistSessionState();
          }
        }
      },
      createSessionId: createQuickIngestSessionId,
    });

    handleRuntimeMessageRef = async (message: any, sender: any) => {
      // Simple ping for E2E tests - verifies message handler is working
      if (message.type === "tldw:ping") {
        return { ok: true, pong: true, timestamp: Date.now() };
      }
      if (message.type === "tldw:diagnostics") {
        return { ok: true, data: buildBackgroundDiagnostics() };
      }
      if (message.type === "tldw:debug") {
        streamDebugEnabled = Boolean(message?.enable);
        return { ok: true };
      }
      if (message.type === "tldw:models:refresh") {
        try {
          const models = await warmModels(true, true);
          const count = Array.isArray(models) ? models.length : 0;
          return { ok: true, count };
        } catch (e: any) {
          return { ok: false, error: e?.message || "Model refresh failed" };
        }
      }
      if (message.type === "tldw:get-tab-id") {
        const tabId = sender?.tab?.id ?? null;
        return { ok: tabId != null, tabId };
      }
      if (message.type === "tldw:quick-ingest/start") {
        const startAck = quickIngestSessionRuntime.start(
          (message.payload || {}) as Record<string, unknown>,
        );
        if (startAck?.ok && startAck.sessionId) {
          quickIngestModalSessions.set(startAck.sessionId, {
            sessionId: startAck.sessionId,
            cancelled: false,
            abortControllers: new Set(),
          });
          void persistSessionState();
        }
        return startAck;
      }
      if (message.type === "tldw:quick-ingest/cancel") {
        const sessionId = String(message?.payload?.sessionId || "").trim();
        const reason = String(message?.payload?.reason || "user_cancelled");
        if (!sessionId) {
          return { ok: false, error: "Missing sessionId." };
        }

        const session = getQuickIngestModalSession(sessionId);
        if (session) {
          session.cancelled = true;
          void persistSessionState();
          for (const controller of Array.from(session.abortControllers)) {
            try {
              controller.abort();
            } catch {
              // best effort
            }
          }
          session.abortControllers.clear();
        }

        const runtimeCancel = quickIngestSessionRuntime.cancel(
          sessionId,
          reason,
        );
        if (!runtimeCancel.ok && !session) {
          return runtimeCancel;
        }
        return { ok: true };
      }
      if (message.type === "tldw:quick-ingest-batch") {
        return await runQuickIngestBatch(message.payload || {});
      }
      if (message.type === "tldw:cancel-request") {
        const requestId = String(message?.payload?.requestId || "").trim();
        if (!requestId) {
          return { ok: false, cancelled: false, error: "Missing requestId." };
        }
        const controller = runtimeRequestControllers.get(requestId);
        if (controller) {
          runtimeRequestControllers.delete(requestId);
          controller.abort();
        }
        return { ok: true, cancelled: Boolean(controller) };
      }
      if (message.type === "sidepanel") {
        try {
          const tabId = sender?.tab?.id ?? undefined;
          ensureSidepanelOpen(tabId);
        } catch (error) {
          logBackgroundError("ensure sidepanel open", error);
        }
        return undefined;
      }
      if (message.type === "tldw:media-ingest/cancel") {
        const funnelId = String(message?.payload?.funnelId || "").trim();
        if (!funnelId) {
          return { ok: false, error: "Missing funnelId" };
        }
        return await cancelIngestSessionById(
          funnelId,
          String(message?.payload?.reason || "user_cancelled"),
        );
      }
      if (message.type === "tldw:media-ingest/retry") {
        const funnelId = String(message?.payload?.funnelId || "").trim();
        if (!funnelId) {
          return { ok: false, error: "Missing funnelId" };
        }
        return await retryIngestSessionById(funnelId, "manual_retry");
      }
      if (message.type === "tldw:media-ingest/open-auth-settings") {
        await openAuthSettings();
        return { ok: true };
      }
      if (message.type === "tldw:media-ingest/funnel-event") {
        const funnelId = String(message?.payload?.funnelId || "").trim();
        const event = String(message?.payload?.event || "").trim();
        if (!funnelId || !event) {
          return { ok: false, error: "Missing funnel metric payload" };
        }
        if (
          event === "context_click" ||
          event === "job_queued" ||
          event === "media_completed" ||
          event === "first_chat_message"
        ) {
          await appendIngestFunnelMetric(event as IngestFunnelEvent, funnelId, {
            ...(message?.payload?.metadata &&
            typeof message.payload.metadata === "object"
              ? message.payload.metadata
              : {}),
          });
          return { ok: true };
        }
        return { ok: false, error: "Unsupported funnel event" };
      }
      if (message.type === "tldw:upload") {
        return runCancelableRuntimeRequest(
          message?.payload?.requestId,
          (signal) => handleUpload(message.payload || {}, signal),
        );
      }
      if (message.type === "tldw:request") {
        return runCancelableRuntimeRequest(
          message?.payload?.requestId,
          (signal) => handleTldwRequest(message.payload || {}, signal),
        );
      }
      if (message.type === "tldw:ingest") {
        try {
          const tabs = await browser.tabs.query({
            active: true,
            currentWindow: true,
          });
          const tab = tabs[0];
          const pageUrl = resolveContextMenuTargetUrl(
            { pageUrl: tab?.url || "" },
            tab,
          );
          if (!pageUrl)
            return { ok: false, status: 400, error: "No active tab URL" };
          if (message.mode === "process") {
            return await handleTldwRequest(
              buildContextMenuProcessPayload(pageUrl),
            );
          }
          return await handleUpload(buildContextMenuAddPayload(pageUrl));
        } catch (e: any) {
          return { ok: false, status: 0, error: e?.message || "Ingest failed" };
        }
      }
      return undefined;
    };

    // Defense-in-depth: only honor privileged runtime messages/ports that
    // originate from this extension's own contexts. A compromised content
    // script still shares our runtime id, but this rejects any sender whose id
    // differs from ours (e.g. another extension). When the id cannot be
    // determined (non-extension test/web contexts), we do not block.
    const isTrustedRuntimeSender = (
      sender: { id?: string } | null | undefined,
    ): boolean => {
      try {
        const selfId = (browser as any)?.runtime?.id;
        const senderId = sender?.id;
        if (selfId && senderId && senderId !== selfId) return false;
      } catch {
        // Fall through: identity indeterminate, do not block.
      }
      return true;
    };

    browser.storage.onChanged.addListener((changes, areaName) => {
      if (areaName !== "local") return;
      if (
        !changes ||
        !Object.prototype.hasOwnProperty.call(changes, "tldwConfig")
      ) {
        return;
      }
      void replayPendingAuthSessions();
    });

    browser.runtime.onConnect.addListener((port) => {
      if (!isTrustedRuntimeSender(port.sender)) {
        try {
          port.disconnect();
        } catch (error) {
          logBackgroundError("reject untrusted port", error);
        }
        return;
      }
      if (port.name === "pgCopilot") {
        isCopilotRunning = true;
        backgroundDiagnostics.ports.copilot += 1;
        backgroundDiagnostics.lastCopilotAt = Date.now();
        port.onDisconnect.addListener(() => {
          isCopilotRunning = false;
          backgroundDiagnostics.ports.copilot = Math.max(
            0,
            backgroundDiagnostics.ports.copilot - 1,
          );
        });
      } else if (port.name === "tldw:stt") {
        backgroundDiagnostics.ports.stt += 1;
        backgroundDiagnostics.lastSttAt = Date.now();
        let ws: WebSocket | null = null;
        let disconnected = false;
        let connectTimer: ReturnType<typeof setTimeout> | null = null;
        const safePost = (msg: any) => {
          if (disconnected) return;
          try {
            port.postMessage(msg);
          } catch (error) {
            logBackgroundError("stt port postMessage", error);
          }
        };
        const onMsg = async (msg: any) => {
          try {
            if (msg?.action === "connect") {
              const cfg = await getEffectiveConfig();
              if (!cfg?.serverUrl)
                throw new Error("tldw server not configured");
              const base = cfg.serverUrl
                .replace(/^http/, "ws")
                .replace(/\/$/, "");
              const rawToken =
                cfg.authMode === "single-user" ? cfg.apiKey : cfg.accessToken;
              const token = String(rawToken || "").trim();
              if (!token) {
                throw new Error(
                  "Not authenticated. Configure tldw credentials in Settings > tldw.",
                );
              }
              // TASK-12106: keep the auth token OUT of the URL (URLs leak into
              // access/proxy logs and history). The transcribe WS authenticates
              // from an {type:"auth", token} first frame sent below on open
              // (streaming_service.py:641-647 multi-user / 720-723 single-user).
              const url = `${base}/api/v1/audio/stream/transcribe`;
              ws = new WebSocket(url);
              ws.binaryType = "arraybuffer";
              connectTimer = setTimeout(() => {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                  safePost({
                    event: "error",
                    message:
                      "STT connection timeout. Check tldw server health.",
                  });
                  try {
                    ws?.close();
                  } catch (error) {
                    logBackgroundError("stt websocket close (timeout)", error);
                  }
                  ws = null;
                }
              }, 10000);
              ws.onopen = () => {
                if (connectTimer) {
                  clearTimeout(connectTimer);
                  connectTimer = null;
                }
                // Send auth as the first frame, before the content script starts
                // streaming audio (it only sends audio after the "open" event).
                try {
                  ws?.send(JSON.stringify({ type: "auth", token }));
                  ws?.send(
                    JSON.stringify({
                      type: "config",
                      protocol_version: 1,
                      mode: "captions",
                      audio_format: "pcm16",
                      sample_rate: 16000,
                      channels: 1,
                    }),
                  );
                } catch (error) {
                  logBackgroundError("stt websocket send auth/config", error);
                }
                safePost({ event: "open" });
              };
              ws.onmessage = (ev) => safePost({ event: "data", data: ev.data });
              ws.onerror = () =>
                safePost({ event: "error", message: "STT websocket error" });
              ws.onclose = () => {
                if (connectTimer) {
                  clearTimeout(connectTimer);
                  connectTimer = null;
                }
                safePost({ event: "close" });
              };
            } else if (
              msg?.action === "audio" &&
              ws &&
              ws.readyState === WebSocket.OPEN
            ) {
              const normalized = normalizeFileData(msg.data);
              const data = normalized
                ? new Uint8Array(normalized).buffer
                : null;
              if (data) {
                ws.send(
                  JSON.stringify({
                    type: "audio",
                    data: arrayBufferToBase64(data),
                  }),
                );
              } else {
                logBackgroundError(
                  "stt websocket audio payload",
                  new Error("Unsupported audio payload"),
                );
              }
            } else if (msg?.action === "close") {
              try {
                ws?.close();
              } catch (error) {
                logBackgroundError("stt websocket close", error);
              }
              ws = null;
            }
          } catch (e: any) {
            safePost({ event: "error", message: e?.message || "ws error" });
          }
        };
        port.onMessage.addListener(onMsg);
        port.onDisconnect.addListener(() => {
          disconnected = true;
          backgroundDiagnostics.ports.stt = Math.max(
            0,
            backgroundDiagnostics.ports.stt - 1,
          );
          try {
            port.onMessage.removeListener(onMsg);
          } catch (error) {
            logBackgroundError("stt port removeListener", error);
          }
          if (connectTimer) {
            clearTimeout(connectTimer);
            connectTimer = null;
          }
          try {
            ws?.close();
          } catch (error) {
            logBackgroundError("stt websocket close (disconnect)", error);
          }
        });
      }
    });

    const actionApi = getActionApi();
    const openOptionsTab = () => {
      const url = browser.runtime.getURL("/options.html#/");
      browser.tabs.create({ url });
    };
    actionApi?.onClicked?.addListener((tab: any) => {
      if (actionIconClick === "webui") {
        openOptionsTab();
      } else {
        ensureSidepanelOpen(tab?.id);
      }
    });

    browser.contextMenus.onClicked.addListener(async (info, tab) => {
      if (info.menuItemId === "open-side-panel-pa") {
        ensureSidepanelOpen(tab?.id);
      } else if (info.menuItemId === "open-web-ui-pa") {
        openOptionsTab();
      } else if (info.menuItemId === transcribeMenuId.transcribe) {
        await handleTranscribeClick(info, tab, "transcribe");
      } else if (info.menuItemId === transcribeMenuId.transcribeAndSummarize) {
        await handleTranscribeClick(info, tab, "transcribe+summary");
      } else if (info.menuItemId === saveToClipperMenuId) {
        await launchWebClipperFromContextMenu(info, tab);
      } else if (info.menuItemId === saveToNotesMenuId) {
        const selection = String(info.selectionText || "").trim();
        if (!selection) {
          notify(
            browser.i18n.getMessage("contextSaveToNotes"),
            browser.i18n.getMessage("contextSaveToNotesNoSelection"),
          );
          return;
        }
        const title =
          browser.i18n.getMessage("contextSaveToNotes") || "Save to Notes";
        const openingMessage =
          browser.i18n.getMessage("contextSaveToNotesOpeningSidebar") ||
          "Opening sidebar to save note…";
        notify(title, openingMessage);
        setTimeout(
          async () => {
            try {
              await ensureSidepanelOpen(tab.id!);
              await browser.runtime.sendMessage({
                from: "background",
                type: "save-to-notes",
                text: selection,
                payload: {
                  selectionText: selection,
                  pageUrl: info.pageUrl || (tab && tab.url) || "",
                  pageTitle: tab?.title || "",
                },
              });
            } catch (e: any) {
              const failureMessage =
                browser.i18n.getMessage("contextSaveToNotesDeliveryFailed") ||
                "Could not open the sidebar to save this note. Check that the tldw Assistant sidepanel is allowed on this site and try again.";
              notify(title, failureMessage);
            }
          },
          isCopilotRunning ? 0 : 5000,
        );
      } else if (info.menuItemId === saveToCompanionMenuId) {
        const selection = String(info.selectionText || "").trim();
        if (!selection) {
          notify(
            browser.i18n.getMessage("contextSaveToCompanion") ||
              "Save to Companion",
            browser.i18n.getMessage("contextSaveToCompanionNoSelection") ||
              "Select text first to save to Companion.",
          );
          return;
        }
        const title =
          browser.i18n.getMessage("contextSaveToCompanion") ||
          "Save to Companion";
        const openingMessage =
          browser.i18n.getMessage("contextSaveToCompanionOpeningSidebar") ||
          "Opening sidebar to save selection to companion...";
        notify(title, openingMessage);
        setTimeout(
          async () => {
            try {
              await ensureSidepanelOpen(tab?.id);
              const captureId =
                typeof globalThis.crypto?.randomUUID === "function"
                  ? globalThis.crypto.randomUUID()
                  : `capture-${Date.now()}`;
              await browser.runtime.sendMessage({
                from: "background",
                type: "save-to-companion",
                text: selection,
                payload: {
                  captureId,
                  selectionText: selection,
                  pageUrl: info.pageUrl || (tab && tab.url) || "",
                  pageTitle: tab?.title || "",
                  action: "save_selection",
                },
              });
            } catch (_error) {
              const failureMessage =
                browser.i18n.getMessage(
                  "contextSaveToCompanionDeliveryFailed",
                ) ||
                "Could not open the sidebar to save this selection. Check that the tldw Assistant sidepanel is allowed on this site and try again.";
              notify(title, failureMessage);
            }
          },
          isCopilotRunning ? 0 : 5000,
        );
      } else if (info.menuItemId === narrateSelectionMenuId) {
        const selection = String(info.selectionText || "").trim();
        if (!selection) {
          notify(
            browser.i18n.getMessage("contextNarrateSelection"),
            browser.i18n.getMessage("contextNarrateSelectionNoSelection"),
          );
          return;
        }
        const title =
          browser.i18n.getMessage("contextNarrateSelection") ||
          "Narrate selection";
        const openingMessage =
          browser.i18n.getMessage("contextSidebarOpening") ||
          "Opening sidebar...";
        notify(title, openingMessage);
        setTimeout(
          async () => {
            try {
              await ensureSidepanelOpen(tab?.id);
              await browser.runtime.sendMessage({
                from: "background",
                type: "narrate-selection",
                text: selection,
                payload: {
                  selectionText: selection,
                  pageUrl: info.pageUrl || (tab && tab.url) || "",
                  pageTitle: tab?.title || "",
                },
              });
            } catch (e) {
              console.error("[tldw] narrate selection failed:", e);
            }
          },
          isCopilotRunning ? 0 : 5000,
        );
      } else if (info.menuItemId === "send-to-tldw") {
        try {
          const targetUrl = resolveContextMenuTargetUrl(info, tab);
          if (!targetUrl) return;
          const addPayload = buildContextMenuAddPayload(targetUrl);
          const session: IngestSession = {
            funnelId: createFunnelId(),
            url: targetUrl,
            normalizedUrl: normalizeUrlForDedupe(targetUrl),
            tabId: tab?.id,
            status: "queued",
            jobIds: [],
            createdAt: Date.now(),
            retryCount: 0,
            awaitingAuth: false,
            timestampSeconds: extractYouTubeTimestampSeconds(targetUrl),
          };
          notify("tldw_server", "Starting ingest…");
          void startContextMenuIngest(session, {
            trackContextClick: true,
            reason: "initial",
          });
        } catch (e) {
          console.error("Failed to send to tldw_server:", e);
          notify("tldw_server", "Failed to send item to tldw_server");
        }
      } else if (info.menuItemId === "process-local-tldw") {
        try {
          const targetUrl = resolveContextMenuTargetUrl(info, tab);
          if (!targetUrl) return;
          const resp = await handleTldwRequest(
            buildContextMenuProcessPayload(targetUrl),
          );
          if (!resp?.ok) {
            throw new Error(resp?.error || "Processing failed");
          }
          notify("tldw_server", "Processed page (not saved to server)");
        } catch (e) {
          console.error("Failed to process locally:", e);
          notify("tldw_server", "Failed to process page");
        }
      } else if (info.menuItemId === "summarize-pa") {
        ensureSidepanelOpen(tab?.id);
        // this is a bad method hope somone can fix it :)
        setTimeout(
          async () => {
            await browser.runtime.sendMessage({
              from: "background",
              type: "summary",
              text: info.selectionText,
            });
          },
          isCopilotRunning ? 0 : 5000,
        );
      } else if (info.menuItemId === "rephrase-pa") {
        ensureSidepanelOpen(tab?.id);
        setTimeout(
          async () => {
            await browser.runtime.sendMessage({
              type: "rephrase",
              from: "background",
              text: info.selectionText,
            });
          },
          isCopilotRunning ? 0 : 5000,
        );
      } else if (info.menuItemId === "translate-pg") {
        ensureSidepanelOpen(tab?.id);

        setTimeout(
          async () => {
            await browser.runtime.sendMessage({
              type: "translate",
              from: "background",
              text: info.selectionText,
            });
          },
          isCopilotRunning ? 0 : 5000,
        );
      } else if (info.menuItemId === "explain-pa") {
        ensureSidepanelOpen(tab?.id);

        setTimeout(
          async () => {
            await browser.runtime.sendMessage({
              type: "explain",
              from: "background",
              text: info.selectionText,
            });
          },
          isCopilotRunning ? 0 : 5000,
        );
      } else if (info.menuItemId === "custom-pg") {
        ensureSidepanelOpen(tab?.id);

        setTimeout(
          async () => {
            await browser.runtime.sendMessage({
              type: "custom",
              from: "background",
              text: info.selectionText,
            });
          },
          isCopilotRunning ? 0 : 5000,
        );
      } else if (info.menuItemId === "contextual-popup-pa") {
        const selection = String(info.selectionText || "").trim();
        if (!selection) {
          notify(
            browser.i18n.getMessage("contextCopilotPopup"),
            browser.i18n.getMessage("contextCopilotPopupNoSelection"),
          );
          return;
        }
        const tabId = tab?.id;
        if (!tabId) {
          notify(
            browser.i18n.getMessage("contextCopilotPopup"),
            browser.i18n.getMessage("contextCopilotPopupNoTab"),
          );
          return;
        }
        try {
          await browser.tabs.sendMessage(
            tabId,
            {
              type: "tldw:popup:open",
              payload: {
                selectionText: selection,
                pageUrl: info.pageUrl || tab?.url || "",
                pageTitle: tab?.title || "",
                frameId: info.frameId,
              },
            },
            typeof info.frameId === "number"
              ? { frameId: info.frameId }
              : undefined,
          );
        } catch (error) {
          logBackgroundError("contextual popup sendMessage", error);
          notify(
            browser.i18n.getMessage("contextCopilotPopup"),
            browser.i18n.getMessage("contextCopilotPopupDeliveryFailed"),
          );
        }
      }
    });

    // Stream handler via Port API
    browser.runtime.onConnect.addListener((port) => {
      if (!isTrustedRuntimeSender(port.sender)) {
        try {
          port.disconnect();
        } catch (error) {
          logBackgroundError("reject untrusted stream port", error);
        }
        return;
      }
      if (port.name === "tldw:stream") {
        backgroundDiagnostics.ports.stream += 1;
        backgroundDiagnostics.lastStreamAt = Date.now();
        let abort: AbortController | null = null;
        let idleTimer: any = null;
        let closed = false;
        let disconnected = false;
        const safePost = (msg: any) => {
          if (disconnected) return;
          try {
            port.postMessage(msg);
          } catch (error) {
            logBackgroundError("stream port postMessage", error);
          }
        };
        const onMsg = async (msg: any) => {
          const sanitizeRagProviderStreamError =
            msg?.sanitizeRagProviderStreamError === true;
          const postStreamError = (payload: {
            event: "error";
            message: unknown;
            status?: unknown;
            code?: unknown;
            details?: unknown;
            retryAfter?: unknown;
          }) => {
            if (!sanitizeRagProviderStreamError) {
              safePost(payload);
              return;
            }
            const sanitized = sanitizeRagProviderFailure({
              status: payload.status,
              error: formatErrorMessage(payload.message, "Stream error"),
              data:
                payload.details ??
                (payload.code
                  ? {
                      detail: {
                        error_code: payload.code,
                        message: payload.message,
                      },
                    }
                  : undefined),
            });
            const safePayload: Record<string, unknown> = {
              event: "error",
              message: sanitized.message,
            };
            if (typeof sanitized.status === "number") {
              safePayload.status = sanitized.status;
            }
            if (sanitized.code) {
              safePayload.code = sanitized.code;
            }
            if (sanitized.details) {
              safePayload.details = sanitized.details;
            }
            safePost(safePayload);
          };
          try {
            const servicePromptConfig = readServicePromptTargetLock(
              msg?.servicePromptConfig,
            );
            const path = msg.path as string;
            if (servicePromptConfig && !isServicePromptRequestPath(path, msg.method)) {
              const error = new Error(
                "A Service Prompt config can only be used with Service Prompt requests.",
              ) as Error & { status: number };
              error.status = 400;
              throw error;
            }
            const scopedCfg = servicePromptConfig
              ? await resolveCurrentServicePromptConfig(servicePromptConfig)
              : null;
            const cfg = scopedCfg ?? (await getEffectiveConfig());
            if (!cfg?.serverUrl) throw new Error("tldw server not configured");
            const baseUrl = String(cfg.serverUrl).replace(/\/$/, "");
            const streamAccess = evaluateAbsoluteUrlAccess(path, cfg);
            // Mirror the request-path guard: refuse cross-origin absolute URLs
            // that are not explicitly allowlisted before opening the stream.
            if (streamAccess.blocked) {
              postStreamError({
                event: "error",
                message: ABSOLUTE_URL_BLOCK_ERROR,
              });
              return;
            }
            const url = streamAccess.isAbsolute
              ? path
              : `${baseUrl}${path.startsWith("/") ? "" : "/"}${path}`;
            const headers: Record<string, string> = { ...(msg.headers || {}) };
            for (const k of Object.keys(headers)) {
              const kl = k.toLowerCase();
              if (kl === "x-api-key" || kl === "authorization")
                delete headers[k];
            }
            // Skip credentials for allowlisted-but-cross-origin absolute URLs,
            // matching request-core's `shouldSkipAuth` behavior.
            if (!streamAccess.skipAuth) {
              if (cfg.authMode === "single-user") {
                const key = (cfg.apiKey || "").trim();
                if (!key) {
                  postStreamError({
                    event: "error",
                    message:
                      "Add or update your API key in Settings → tldw server, then try again.",
                  });
                  return;
                }
                headers["X-API-KEY"] = key;
              } else if (cfg.authMode === "multi-user") {
                const token = (cfg.accessToken || "").trim();
                if (token) headers["Authorization"] = `Bearer ${token}`;
                else {
                  postStreamError({
                    event: "error",
                    message:
                      "Not authenticated. Please login under Settings > tldw.",
                  });
                  return;
                }
              }
            }
            headers["Accept"] = "text/event-stream";
            headers["Cache-Control"] = headers["Cache-Control"] || "no-cache";
            headers["Connection"] = headers["Connection"] || "keep-alive";
            const requestAbort = new AbortController();
            abort = requestAbort;
            const idleMs = deriveStreamIdleTimeout(
              cfg,
              path,
              Number(msg?.streamIdleTimeoutMs),
            );
            const resetIdle = () => {
              if (idleTimer) clearTimeout(idleTimer);
              idleTimer = setTimeout(() => {
                if (!closed) {
                  try {
                    abort?.abort();
                  } catch (error) {
                    logBackgroundError("stream abort", error);
                  }
                  postStreamError({
                    event: "error",
                    message: "Stream timeout: no updates received",
                  });
                }
              }, idleMs);
            };
            // Ensure SSE-friendly headers
            headers["Accept"] = headers["Accept"] || "text/event-stream";
            headers["Cache-Control"] = headers["Cache-Control"] || "no-cache";
            headers["Connection"] = headers["Connection"] || "keep-alive";

            let resp = await fetch(url, {
              method: msg.method || "POST",
              headers,
              body:
                typeof msg.body === "string"
                  ? msg.body
                  : JSON.stringify(msg.body),
              signal: requestAbort.signal,
            });
            if (
              !streamAccess.skipAuth &&
              resp.status === 401 &&
              cfg.authMode === "multi-user" &&
              cfg.refreshToken
            ) {
              try {
                if (servicePromptConfig) {
                  await refreshScopedAuth(
                    servicePromptConfig,
                    scopedCfg ?? undefined,
                  );
                } else {
                  if (!refreshInFlight) {
                    refreshInFlight = (async () => {
                      try {
                        await tldwAuth.refreshToken();
                      } finally {
                        refreshInFlight = null;
                      }
                    })();
                  }
                  await refreshInFlight;
                }
              } catch (error) {
                if (
                  servicePromptConfig &&
                  (error as { status?: unknown })?.status === 412
                ) {
                  throw error;
                }
                logBackgroundError("refresh auth (stream)", error);
              }
              if (disconnected || closed || requestAbort.signal.aborted) {
                return;
              }
              const updated = servicePromptConfig
                ? await resolveCurrentServicePromptConfig(servicePromptConfig)
                : await getEffectiveConfig();
              if (updated?.accessToken)
                headers["Authorization"] = `Bearer ${updated.accessToken}`;
              const retryController = new AbortController();
              if (disconnected || closed || requestAbort.signal.aborted) {
                retryController.abort();
                return;
              }
              abort = retryController;
              resp = await fetch(url, {
                method: msg.method || "POST",
                headers,
                body:
                  typeof msg.body === "string"
                    ? msg.body
                    : JSON.stringify(msg.body),
                signal: retryController.signal,
              });
            }
            if (!resp.ok) {
              const ct = resp.headers.get("content-type") || "";
              let errMsg: any = resp.statusText;
              let errDetails: any = undefined;
              if (ct.includes("application/json")) {
                const j = await resp.json().catch(() => null);
                if (j && (j.detail || j.error || j.message))
                  errMsg = j.detail || j.error || j.message;
                errDetails = j;
              } else {
                const t = await resp.text().catch(() => null);
                if (t) errMsg = t;
              }
              postStreamError({
                event: "error",
                status: resp.status,
                message: formatErrorMessage(errMsg, `HTTP ${resp.status}`),
                details: errDetails,
                retryAfter: resp.headers.get("retry-after"),
              });
              return;
            }
            if (!resp.body) throw new Error("No response body");
            safePost({ event: "open" });
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            resetIdle();
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              resetIdle();
              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split("\n");
              buffer = lines.pop() || "";
              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                // Any SSE activity resets idle timer
                resetIdle();
                if (trimmed.startsWith("event:")) {
                  const name = trimmed.slice(6).trim();
                  if (streamDebugEnabled) {
                    try {
                      await browser.runtime.sendMessage({
                        type: "tldw:stream-debug",
                        payload: { kind: "event", name, time: Date.now() },
                      });
                    } catch (error) {
                      logBackgroundError("stream debug event", error);
                    }
                  }
                } else if (trimmed.startsWith("data:")) {
                  const data = trimmed.slice(5).trim();
                  if (streamDebugEnabled) {
                    try {
                      await browser.runtime.sendMessage({
                        type: "tldw:stream-debug",
                        payload: { kind: "data", data, time: Date.now() },
                      });
                    } catch (error) {
                      logBackgroundError("stream debug data", error);
                    }
                  }
                  if (data === "[DONE]") {
                    closed = true;
                    if (idleTimer) clearTimeout(idleTimer);
                    safePost({ event: "done" });
                    return;
                  }
                  safePost({ event: "data", data });
                } else if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
                  // Some servers may omit the 'data:' prefix; treat JSON lines as data
                  const data = trimmed;
                  if (streamDebugEnabled) {
                    try {
                      await browser.runtime.sendMessage({
                        type: "tldw:stream-debug",
                        payload: { kind: "data", data, time: Date.now() },
                      });
                    } catch (error) {
                      logBackgroundError("stream debug json", error);
                    }
                  }
                  safePost({ event: "data", data });
                }
              }
            }
            closed = true;
            if (idleTimer) clearTimeout(idleTimer);
            safePost({ event: "done" });
          } catch (e: any) {
            if (idleTimer) clearTimeout(idleTimer);
            postStreamError({
              event: "error",
              ...(typeof e?.status === "number" ? { status: e.status } : {}),
              ...(typeof e?.details !== "undefined"
                ? { details: e.details }
                : {}),
              message: formatErrorMessage(e, "Stream error"),
            });
          }
        };
        port.onMessage.addListener(onMsg);
        port.onDisconnect.addListener(() => {
          disconnected = true;
          backgroundDiagnostics.ports.stream = Math.max(
            0,
            backgroundDiagnostics.ports.stream - 1,
          );
          try {
            port.onMessage.removeListener(onMsg);
          } catch (error) {
            logBackgroundError("stream port removeListener", error);
          }
          try {
            abort?.abort();
          } catch (error) {
            logBackgroundError("stream abort (disconnect)", error);
          }
        });
      }
    });

    const ensureInitialized = () => {
      if (!initializePromise) {
        initializePromise = initialize().catch((error) => {
          initializePromise = null;
          handleRuntimeMessageRef = null;
          throw error;
        });
      }
      return initializePromise;
    };

    const runtimeOnMessage =
      (globalThis as any).chrome?.runtime?.onMessage ||
      browser.runtime.onMessage;

    runtimeOnMessage.addListener(
      (message: any, sender: any, sendResponse: any) => {
        if (!isTrustedRuntimeSender(sender)) {
          return false;
        }
        backgroundDiagnostics.runtimeMessageCount += 1;
        backgroundDiagnostics.lastRuntimeMessageType =
          typeof message?.type === "string" ? message.type : null;
        backgroundDiagnostics.lastRuntimeSenderUrl =
          typeof sender?.url === "string"
            ? sender.url
            : typeof sender?.tab?.url === "string"
              ? sender.tab.url
              : null;
        if (message?.type === "tldw:ping") {
          backgroundDiagnostics.runtimePingCount += 1;
          sendResponse({ ok: true, pong: true, timestamp: Date.now() });
          return;
        }

        void ensureInitialized()
          .then(async () => {
            if (!handleRuntimeMessageRef) {
              return undefined;
            }
            return await handleRuntimeMessageRef(message, sender);
          })
          .then((response) => {
            sendResponse(response);
          })
          .catch((error) => {
            logBackgroundError("runtime message", error);
            sendResponse({
              ok: false,
              status: 0,
              error: (error as Error)?.message || "Background error",
            });
          });
        return true;
      },
    );

    if (browser?.alarms) {
      browser.alarms.onAlarm.addListener((alarm) => {
        if (alarm.name === INGEST_POLL_ALARM_NAME) {
          backgroundDiagnostics.alarmFires += 1;
          backgroundDiagnostics.lastAlarmAt = Date.now();
          // Ensure state is hydrated (worker may have just woken), then resume
          // any ingest polling interrupted by suspension (TASK-12094).
          void (async () => {
            await rehydrateSessionState();
            await resumeActiveIngestSessions();
          })();
          return;
        }
        if (alarm.name === QUICK_INGEST_POLL_ALARM_NAME) {
          backgroundDiagnostics.alarmFires += 1;
          backgroundDiagnostics.lastAlarmAt = Date.now();
          // Ensure state is hydrated (worker may have just woken), then resume
          // any interrupted quick-ingest remote-job polling (TASK-12094).
          void (async () => {
            await rehydrateSessionState();
            await resumeQuickIngestBatches();
          })();
          return;
        }
        if (alarm.name !== MODEL_WARM_ALARM_NAME) return;
        backgroundDiagnostics.alarmFires += 1;
        backgroundDiagnostics.lastAlarmAt = Date.now();
        void warmModels(true);
      });
    }

    if (browser?.runtime?.onStartup) {
      browser.runtime.onStartup.addListener(() => {
        void rehydrateSessionState();
      });
    }

    void ensureInitialized();
  },
  persistent: false,
});
