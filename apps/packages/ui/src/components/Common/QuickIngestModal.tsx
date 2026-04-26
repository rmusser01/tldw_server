import React, { useCallback } from 'react'
import { Modal, Button, Input, Select, Space, Switch, Typography, Tag, message, Collapse, InputNumber, Tooltip as AntTooltip, Spin } from 'antd'
import { useQuery } from "@tanstack/react-query"
import { useTranslation } from 'react-i18next'
import { browser } from "wxt/browser"
import { tldwClient } from '@/services/tldw/TldwApiClient'
import {
  cancelQuickIngestSession,
  startQuickIngestSession,
  submitQuickIngestBatch
} from "@/services/tldw/quick-ingest-batch"
import { QuickIngestTabs } from "./QuickIngest/QuickIngestTabs"
import { QueueTab } from "./QuickIngest/QueueTab/QueueTab"
import { FileDropZone } from "./QuickIngest/QueueTab/FileDropZone"
import { OptionsTab } from "./QuickIngest/OptionsTab/OptionsTab"
import { ResultsTab } from "./QuickIngest/ResultsTab/ResultsTab"
import { ProcessButton } from "./QuickIngest/shared/ProcessButton"
import { QUICK_INGEST_ACCEPT_STRING } from "./QuickIngest/constants"
import { DEFAULT_PRESET } from "./QuickIngest/presets"
import { HelpCircle, Headphones, Layers, Database, FileText, Film, Cookie, Info, Clock, Grid, BookText, Link2, File as FileIcon, AlertTriangle, Star, X } from 'lucide-react'
import { useStorage } from '@plasmohq/storage/hook'
import { QuickIngestInspectorDrawer } from "@/components/Common/QuickIngestInspectorDrawer"
import {
  fetchChatModels,
  getEmbeddingModels
} from '@/services/tldw-server'
import {
  ensureSelectOption,
  getAdvancedFieldSelectOptions
} from "@/components/Common/QuickIngest/advanced-field-options"
import {
  inferIngestTypeFromUrl,
} from "@/services/tldw/media-routing"
import { DRAFT_STORAGE_CAP_BYTES } from "@/db/dexie/drafts"
import {
  useIngestWizardFlow,
  useIngestOptions,
  useIngestPresets,
  useIngestQueue,
  useIngestResults,
  getFileInstanceId,
  normalizeResultItem,
  normalizeResultStatus,
  mediaIdFromPayload,
  titleFromPayload,
  MAX_LOCAL_FILE_BYTES,
  RESULT_FILTERS,
} from './hooks'
import type { ResultItem, PlannedRunContext, ResultsFilter } from './hooks'

type Props = {
  open: boolean
  onClose: () => void
  autoProcessQueued?: boolean
}

const ProcessingIndicator = ({ label }: { label: string }) => (
  <div className="flex items-center gap-2 text-[11px] text-text-subtle">
    <span className="relative flex h-2.5 w-2.5">
      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60 opacity-75" />
      <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-primary" />
    </span>
    <span>{label}</span>
  </div>
)

const QueuedItemRow = React.lazy(() =>
  import("./QuickIngest/QueuedItemRow").then((m) => ({
    default: m.QueuedItemRow
  }))
)
const QueuedFileRow = React.lazy(() =>
  import("./QuickIngest/QueuedFileRow").then((m) => ({
    default: m.QueuedFileRow
  }))
)

const INLINE_FILE_WARN_BYTES = 100 * 1024 * 1024
const MAX_RECOMMENDED_FIELDS = 12

const RECOMMENDED_FIELD_NAMES = new Set<string>([
  "cookies", "cookie", "headers", "custom_headers", "custom_cookies",
  "user_agent", "authorization", "auth_header", "embedding_model",
  "default_embedding_model", "context_strategy", "perform_chunking",
  "perform_analysis", "overwrite_existing", "system_prompt", "custom_prompt",
  "scrape_method", "crawl_strategy", "include_external", "score_threshold",
  "max_pages", "max_depth", "url_level"
])

const logicalGroupForField = (name: string): string => {
  const n = name.toLowerCase()
  if (n.startsWith('transcription_') || ['diarize', 'vad_use', 'chunk_language'].includes(n)) return 'Transcription'
  if (n.startsWith('chunk_') || ['use_adaptive_chunking', 'enable_contextual_chunking', 'use_multi_level_chunking', 'perform_chunking', 'contextual_llm_model'].includes(n)) return 'Chunking'
  if (n.includes('embedding')) return 'Embeddings'
  if (n.startsWith('context_')) return 'Context'
  if (n.includes('summarization') || n.includes('analysis') || n === 'system_prompt' || n === 'custom_prompt') return 'Analysis/Summarization'
  if (n.includes('pdf') || n.includes('ocr')) return 'Document/PDF'
  if (n.includes('video')) return 'Video'
  if (n.includes('cookie') || n === 'cookies' || n === 'headers' || n === 'authorization' || n === 'auth_header' || n.includes('user_agent')) return 'Cookies/Auth'
  if (['author', 'title', 'keywords', 'api_name'].includes(n)) return 'Metadata'
  if (['start_time', 'end_time'].includes(n)) return 'Timing'
  return 'Other'
}

const isRecommendedField = (name: string, logicalGroup: string): boolean => {
  const n = name.toLowerCase()
  if (RECOMMENDED_FIELD_NAMES.has(n)) return true
  if (n.includes('embedding')) return true
  if (logicalGroup === 'Analysis/Summarization' && (n.includes('summary') || n.includes('summarization') || n.includes('analysis'))) return true
  return false
}

const iconForGroup = (group: string) => {
  const cls = 'w-4 h-4 mr-1 text-text-subtle'
  switch (group) {
    case 'Recommended': return <Star className={cls} />
    case 'Transcription': return <Headphones className={cls} />
    case 'Chunking': return <Layers className={cls} />
    case 'Embeddings': return <Database className={cls} />
    case 'Context': return <Layers className={cls} />
    case 'Analysis/Summarization': return <BookText className={cls} />
    case 'Document/PDF': return <FileText className={cls} />
    case 'Video': return <Film className={cls} />
    case 'Cookies/Auth': return <Cookie className={cls} />
    case 'Metadata': return <Info className={cls} />
    case 'Timing': return <Clock className={cls} />
    default: return <Grid className={cls} />
  }
}

export const QuickIngestModal: React.FC<Props> = ({
  open,
  onClose,
  autoProcessQueued = false
}) => {
  const { t } = useTranslation(['option', 'settings'])
  const qi = React.useCallback(
    (key: string, defaultValue: string, options?: Record<string, any>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t]
  )
  const [messageApi, contextHolder] = message.useMessage({
    top: 12,
    getContainer: () =>
      (document.querySelector('.quick-ingest-modal .ant-modal-content') as HTMLElement) || document.body
  })

  // ---- Local state not covered by hooks ----
  const [running, setRunning] = React.useState<boolean>(false)
  const [inspectorIntroDismissed, setInspectorIntroDismissed] = useStorage<boolean>('quickIngestInspectorIntroDismissed', false)
  const [onboardingDismissed, setOnboardingDismissed] = useStorage<boolean>('quickIngestOnboardingDismissed', false)
  const introToast = React.useRef(false)
  const autoProcessedRef = React.useRef(false)
  const activeSessionIdRef = React.useRef<string | null>(null)
  const terminalSessionIdsRef = React.useRef<Set<string>>(new Set())
  const plannedRunContextRef = React.useRef<PlannedRunContext | null>(null)

  // ---- Hook: Options ----
  const options = useIngestOptions({
    open,
    running,
    ingestConnectionStatus: "online", // placeholder, overridden below
    messageApi,
    qi,
  })

  // ---- Hook: Wizard Flow (connection, tabs, badges, inspector) ----
  const wizard = useIngestWizardFlow({
    open,
    running,
    plannedCount: 0, // placeholder, overridden below after queue is available
    common: options.common,
    advancedValues: options.advancedValues,
    hasTypeDefaultChanges: options.hasTypeDefaultChanges,
    lastRunError: null, // placeholder
    reviewBeforeStorage: options.reviewBeforeStorage,
    storeRemote: options.storeRemote,
  })

  // Re-call options with real connection status
  const optionsReal = useIngestOptions({
    open,
    running,
    ingestConnectionStatus: wizard.ingestConnectionStatus,
    messageApi,
    qi,
  })

  // ---- Hook: Queue ----
  const queue = useIngestQueue({
    open,
    running,
    ingestBlocked: wizard.ingestBlocked,
    messageApi,
    qi,
    normalizedTypeDefaults: optionsReal.normalizedTypeDefaults,
  })

  // ---- Hook: Presets ----
  const presets = useIngestPresets({
    open,
    common: optionsReal.common,
    setCommon: optionsReal.setCommon,
    storeRemote: optionsReal.storeRemote,
    setStoreRemote: optionsReal.setStoreRemote,
    reviewBeforeStorage: optionsReal.reviewBeforeStorage,
    setReviewBeforeStorage: optionsReal.setReviewBeforeStorage,
    normalizedTypeDefaults: optionsReal.normalizedTypeDefaults,
    setTypeDefaults: optionsReal.setTypeDefaults,
    advancedValues: optionsReal.advancedValues,
    setAdvancedValues: optionsReal.setAdvancedValues,
  })

  // ---- Hook: Results ----
  const results = useIngestResults({
    open,
    running,
    messageApi,
    qi,
    t,
    onClose,
    reviewBeforeStorage: optionsReal.reviewBeforeStorage,
    storeRemote: optionsReal.storeRemote,
    processOnly: optionsReal.processOnly,
    common: optionsReal.common,
    advancedValues: optionsReal.advancedValues,
    rows: queue.rows,
    formatBytes: queue.formatBytes,
    plannedRunContextRef,
    setRows: queue.setRows,
    setQueuedFiles: queue.setQueuedFiles,
    setLocalFiles: queue.setLocalFiles,
    buildRowEntry: queue.buildRowEntry,
    createDefaultsSnapshot: queue.createDefaultsSnapshot,
    setReviewBeforeStorage: optionsReal.setReviewBeforeStorage,
    setStoreRemote: optionsReal.setStoreRemote,
  })

  // Re-call wizard with real values now that queue and results are available
  const wizardReal = useIngestWizardFlow({
    open,
    running,
    plannedCount: queue.plannedCount,
    common: optionsReal.common,
    advancedValues: optionsReal.advancedValues,
    hasTypeDefaultChanges: optionsReal.hasTypeDefaultChanges,
    lastRunError: results.lastRunError,
    reviewBeforeStorage: optionsReal.reviewBeforeStorage,
    storeRemote: optionsReal.storeRemote,
  })

  // ---- Queries ----
  const { data: chatModels = [], isLoading: chatModelsLoading } = useQuery({
    queryKey: ["playground:chatModels"],
    queryFn: () => fetchChatModels({ returnEmpty: true }),
    enabled: open && wizardReal.ingestConnectionStatus === "online"
  })
  const { data: embeddingModels = [], isLoading: embeddingModelsLoading } =
    useQuery({
      queryKey: ["embedding-models"],
      queryFn: () => getEmbeddingModels(),
      enabled: open && wizardReal.ingestConnectionStatus === "online"
    })

  // ---- Derived convenience aliases ----
  const {
    activeTab, setActiveTab,
    ingestConnectionStatus, ingestBlocked, isOnlineForIngest, isConfiguring,
    tabBadges,
    inspectorOpen, setInspectorOpen,
    hasOpenedInspector, setHasOpenedInspector,
    showInspectorIntro, setShowInspectorIntro,
    handleCloseInspector,
    modalReady,
    checkOnce,
  } = wizardReal

  const {
    storeRemote, setStoreRemote,
    reviewBeforeStorage,
    common,
    advancedValues, setAdvancedValues,
    advancedOpen, setAdvancedOpen,
    advSearch, setAdvSearch,
    fieldDetailsOpen, setFieldDetailsOpen,
    savedAdvValues, setSavedAdvValues,
    uiPrefs, setUiPrefs,
    specPrefs,
    specSource,
    normalizedTypeDefaults,
    processOnly,
    shouldStoreRemote,
    modifiedAdvancedCount,
    advancedDefaultsDirty,
    lastRefreshedLabel,
    specSourceLabel,
    resolvedAdvSchema,
    transcriptionModelChoices,
    transcriptionModelValue,
    transcriptionModelsLoading,
    storageLabel,
    storageHintSeen, setStorageHintSeen,
    ragEmbeddingLabel,
    setAdvancedValue,
    handleTranscriptionModelChange,
    handleReviewToggle,
    persistSpecPrefs,
    loadSpec,
    confirmDanger,
    chunkingTemplateName, setChunkingTemplateName,
    autoApplyTemplate, setAutoApplyTemplate,
    setTypeDefaults,
  } = optionsReal

  const {
    rows, setRows,
    queuedFiles, setQueuedFiles,
    localFiles, setLocalFiles,
    selectedRowId, setSelectedRowId,
    selectedFileId, setSelectedFileId,
    pendingUrlInput, setPendingUrlInput,
    reattachInputRef,
    queuedFileStubs,
    attachedFileStubs,
    missingFileStubs,
    attachedFiles,
    fileForStubId,
    hasMissingFiles,
    plannedCount,
    hasAudioItems,
    hasDocumentItems,
    hasVideoItems,
    selectedRow,
    selectedFileStub,
    selectedFile,
    fileTypeFromName,
    formatBytes,
    mergeDefaults,
    statusForUrlRow,
    statusForFile,
    addRow,
    removeRow,
    updateRow,
    addUrlsFromInput,
    clearAllQueues,
    pasteFromClipboard,
    addLocalFiles,
    handleReattachChange,
    requestFileReattach,
    handleReattachSelectedFile,
    handleFileDrop,
    createDefaultsSnapshot,
    buildRowEntry,
  } = queue

  const {
    results: resultItems,
    setResults,
    totalPlanned, setTotalPlanned,
    processedCount, setProcessedCount,
    liveTotalCount, setLiveTotalCount,
    runStartedAt, setRunStartedAt,
    lastRunError, setLastRunError,
    lastRunCancelled, setLastRunCancelled,
    draftCreationError,
    draftCreationRetrying,
    reviewNavigationError,
    reviewBatchId,
    lastRunProcessOnly, setLastRunProcessOnly,
    resultsFilter, setResultsFilter,
    lastFileLookupRef,
    lastFileIdByInstanceIdRef,
    pendingStoreWithoutReviewRef,
    unmountedRef,
    doneCount,
    totalCount,
    progressMeta,
    resultsWithOutcome,
    resultSummary,
    visibleResults,
    hasReviewableResults,
    firstResultWithMedia,
    resultById,
    isCancelledMessage,
    appendMissingResultsFromPlan,
    getResultForFile,
    createDraftsFromResults,
    openHealthDiagnostics,
    openModelSettings,
    tryOpenContentReview,
    handleReviewBatchReady,
    downloadJson,
    downloadResultsJson,
    openInMediaViewer,
    discussInChat,
    retryFailedUrls,
    exportFailedList,
    retryDraftCreation,
    proceedWithoutReview,
    markFailure,
    clearFailure,
    recordRunSuccess,
    recordRunFailure,
    recordRunCancelled,
    summarizeResultOutcomes,
    deriveResultOutcome,
  } = results

  // ---- Sync review + draft state ----
  React.useEffect(() => {
    if (!reviewBeforeStorage) {
      results.setDraftCreationError(null)
      results.setReviewNavigationError(null)
    }
  }, [reviewBeforeStorage])

  // ---- Inspector intro / dismiss ----
  const handleDismissInspectorIntro = React.useCallback(() => {
    setShowInspectorIntro(false)
    try { setInspectorIntroDismissed(true) } catch {}
    setInspectorOpen(false)
    if (!introToast.current) {
      messageApi.success(
        qi("inspectorIntroDismissed", "Intro dismissed — reset anytime in Settings > Quick Ingest (Reset Intro).")
      )
      introToast.current = true
    }
  }, [messageApi, qi, setInspectorIntroDismissed, setInspectorOpen, setShowInspectorIntro])

  // Keep intro hidden if user dismissed previously
  React.useEffect(() => {
    if (inspectorIntroDismissed) setShowInspectorIntro(false)
  }, [inspectorIntroDismissed, setShowInspectorIntro])

  // Track inspector usage
  React.useEffect(() => {
    if ((selectedRow || selectedFileStub) && inspectorOpen && !hasOpenedInspector) {
      setHasOpenedInspector(true)
    }
  }, [hasOpenedInspector, inspectorOpen, selectedFileStub, selectedRow, setHasOpenedInspector])

  // ---- Type icon helper ----
  const typeIcon = React.useCallback((type: string) => {
    const cls = 'w-4 h-4 text-text-subtle'
    switch (type) {
      case 'audio': return <Headphones className={cls} />
      case 'video': return <Film className={cls} />
      case 'pdf':
      case 'document': return <FileText className={cls} />
      case 'html': return <Link2 className={cls} />
      default: return <FileIcon className={cls} />
    }
  }, [])

  // ---- Pending label ----
  const pendingLabel = React.useMemo(() => {
    if (!ingestBlocked) return ""
    if (ingestConnectionStatus === "unconfigured") {
      return t("quickIngest.pendingUnconfigured", "Not connected — configure your server to run.")
    }
    return t("quickIngest.pendingLabel", "Not connected — reconnect to run.")
  }, [ingestBlocked, ingestConnectionStatus, t])

  // ---- Session management ----
  const setActiveSessionId = React.useCallback((sessionId: string | null) => {
    activeSessionIdRef.current = sessionId
  }, [])

  React.useEffect(() => {
    if (open) return
    setActiveSessionId(null)
    plannedRunContextRef.current = null
    terminalSessionIdsRef.current.clear()
  }, [open, setActiveSessionId])

  const completeRunningState = React.useCallback(() => {
    setRunning(false)
    setRunStartedAt(null)
    setActiveSessionId(null)
  }, [setActiveSessionId, setRunStartedAt])

  // ---- Run handlers ----
  const handleRunFailure = React.useCallback(
    (errorMessage: string) => {
      const msg = String(errorMessage || "").trim() || "Quick ingest failed."
      setResults((prev) =>
        appendMissingResultsFromPlan(prev, msg, { status: "error", outcome: "failed" })
      )
      completeRunningState()
      setLastRunCancelled(false)
      setLastRunError(msg)
      const total = plannedRunContextRef.current?.total || 0
      recordRunFailure({ totalCount: total, failedCount: total > 0 ? total : undefined, errorMessage: msg })
      markFailure()
      plannedRunContextRef.current = null
    },
    [appendMissingResultsFromPlan, completeRunningState, markFailure, recordRunFailure, setLastRunCancelled, setLastRunError, setResults]
  )

  const handleRunCancelled = React.useCallback(
    (messageText?: string) => {
      const msg = String(messageText || "").trim() || qi("cancelledByUser", "Cancelled by user.")
      let nextResultsSnapshot: ResultItem[] = []
      setResults((prev) => {
        const next = appendMissingResultsFromPlan(prev, msg, { status: "error", outcome: "cancelled" })
        nextResultsSnapshot = next
        return next
      })
      completeRunningState()
      setLastRunCancelled(true)
      setLastRunError(null)
      clearFailure()
      const counts = summarizeResultOutcomes(nextResultsSnapshot)
      recordRunCancelled({
        totalCount: plannedRunContextRef.current?.total || nextResultsSnapshot.length,
        successCount: counts.successCount,
        failedCount: counts.failCount,
        cancelledCount: counts.cancelledCount,
        errorMessage: msg
      })
      plannedRunContextRef.current = null
    },
    [appendMissingResultsFromPlan, clearFailure, completeRunningState, qi, recordRunCancelled, setLastRunCancelled, setLastRunError, setResults, summarizeResultOutcomes]
  )

  const handleRunCompleted = React.useCallback(
    async (normalizedResults: ResultItem[]) => {
      const out = appendMissingResultsFromPlan(
        normalizedResults,
        qi("missingResultItems", "No result was returned for this item."),
        { status: "error", outcome: "failed" }
      )
      if (unmountedRef.current) return
      setResults(out)
      completeRunningState()
      setLastRunCancelled(false)
      const { successCount, failCount, cancelledCount } = summarizeResultOutcomes(out)
      const hasOkResults = successCount > 0
      const firstSuccessfulItem = out.find((r) => r.status === "ok") || null
      const firstMediaId = firstSuccessfulItem ? mediaIdFromPayload(firstSuccessfulItem.data) : null
      const primarySourceLabel = firstSuccessfulItem?.url || firstSuccessfulItem?.fileName || null

      if (hasOkResults) {
        recordRunSuccess({
          totalCount: out.length, successCount, failedCount: failCount,
          firstMediaId: firstMediaId === null || typeof firstMediaId === "undefined" ? null : String(firstMediaId),
          primarySourceLabel
        })
      } else if (cancelledCount > 0) {
        recordRunCancelled({ totalCount: out.length, successCount, failedCount: failCount, cancelledCount })
      } else {
        const firstError = out.find((item) => item.status === "error")?.error || null
        recordRunFailure({ totalCount: out.length, failedCount: out.length, errorMessage: firstError })
      }

      const fileLookup = plannedRunContextRef.current?.fileLookup || new Map<string, File>()
      let createdDraftBatch: { batchId: string; draftIds: string[]; skippedAssets: number } | null = null
      if (reviewBeforeStorage && hasOkResults) {
        let draftErrorMessage: string | null = null
        try {
          createdDraftBatch = await createDraftsFromResults(out, fileLookup)
          if (createdDraftBatch?.batchId) results.setReviewBatchId(createdDraftBatch.batchId)
        } catch (err) {
          console.error("[quickIngest] Failed to create review drafts", err)
          draftErrorMessage = qi("reviewDraftsFailedFallback", "Failed to create review drafts.")
        }
        if (!createdDraftBatch?.batchId) {
          const msg = draftErrorMessage || qi("reviewDraftsFailedFallback", "Failed to create review drafts.")
          messageApi.error(msg)
          results.setDraftCreationError(msg)
          plannedRunContextRef.current = null
          return
        }
      }

      if (lastRunProcessOnly && !reviewBeforeStorage && out.length > 0) {
        messageApi.info(qi("processingComplete", "Processing complete. Use \"Download JSON\" below to save results locally."))
      }
      if (out.length > 0) {
        const summary = cancelledCount > 0
          ? `${successCount} succeeded · ${failCount} failed · ${cancelledCount} cancelled`
          : `${successCount} succeeded · ${failCount} failed`
        if (failCount > 0) messageApi.warning(summary)
        else if (cancelledCount > 0) messageApi.info(summary)
        else messageApi.success(summary)
      }
      if (createdDraftBatch?.batchId) await handleReviewBatchReady(createdDraftBatch)
      clearFailure()
      setLastRunError(null)
      plannedRunContextRef.current = null
    },
    [appendMissingResultsFromPlan, clearFailure, completeRunningState, createDraftsFromResults,
     handleReviewBatchReady, lastRunProcessOnly, messageApi, qi, recordRunCancelled,
     recordRunFailure, recordRunSuccess, reviewBeforeStorage, setLastRunCancelled, setLastRunError,
     setResults, summarizeResultOutcomes, unmountedRef, results]
  )

  // ---- Main run function ----
  const run = React.useCallback(async () => {
    setLastRunError(null)
    setLastRunCancelled(false)
    results.setDraftCreationError(null)
    results.setReviewNavigationError(null)
    lastFileLookupRef.current = null
    lastFileIdByInstanceIdRef.current = null
    clearFailure()
    terminalSessionIdsRef.current.clear()
    setActiveSessionId(null)

    if (missingFileStubs.length > 0) {
      messageApi.error(qi("missingFilesBlock", "Reattach {{count}} local file(s) to run ingest.", { count: missingFileStubs.length }))
      return
    }
    if (ingestBlocked) {
      const blockedMessage =
        ingestConnectionStatus === "unconfigured"
          ? t("quickIngest.unavailableUnconfigured", "Ingest unavailable \u2014 server not configured")
          : ingestConnectionStatus === "unknown"
            ? t("quickIngest.checkingTitle", "Checking server connection\u2026")
            : t("quickIngest.unavailableOffline", "Ingest unavailable \u2014 not connected")
      messageApi.warning(blockedMessage)
      return
    }
    const valid = rows.filter((r) => r.url.trim().length > 0)
    if (valid.length === 0 && attachedFiles.length === 0) {
      messageApi.error('Please add at least one URL or file')
      return
    }
    const plannedUrlEntries = valid.map((row) => ({
      id: row.id, url: row.url,
      type: row.type === "auto" ? inferIngestTypeFromUrl(row.url) : row.type
    }))
    const plannedFiles = attachedFiles.map((file) => ({ file, type: fileTypeFromName(file) }))
    const oversizedFiles = attachedFiles.filter((f) => f.size && f.size > MAX_LOCAL_FILE_BYTES)
    if (oversizedFiles.length > 0) {
      const maxLabel = formatBytes(MAX_LOCAL_FILE_BYTES)
      const names = oversizedFiles.map((f) => f.name).slice(0, 3).join(', ')
      const suffix = oversizedFiles.length > 3 ? '\u2026' : ''
      const msg = names
        ? `File too large: ${names}${suffix}. Each file must be smaller than ${maxLabel}.`
        : `One or more files are too large. Each file must be smaller than ${maxLabel}.`
      messageApi.error(msg)
      setLastRunError(msg)
      return
    }
    const total = valid.length + attachedFiles.length
    wizardReal.setRunNonce((n) => n + 1)
    setTotalPlanned(total)
    setProcessedCount(0)
    setLiveTotalCount(total)
    setRunStartedAt(Date.now())
    setLastRunProcessOnly(processOnly)
    setRunning(true)
    setResults([])
    results.setReviewBatchId(null)
    const fileLookup = new Map<string, File>()
    const fileIdByInstanceId = new Map<string, string>()
    for (const file of attachedFiles) {
      const instanceId = getFileInstanceId(file)
      if (!fileIdByInstanceId.has(instanceId)) fileIdByInstanceId.set(instanceId, crypto.randomUUID())
      const fileId = fileIdByInstanceId.get(instanceId)
      if (fileId) fileLookup.set(fileId, file)
    }
    lastFileLookupRef.current = fileLookup
    lastFileIdByInstanceIdRef.current = fileIdByInstanceId
    plannedRunContextRef.current = { total, plannedUrlEntries, plannedFiles, fileLookup, fileIdByInstanceId }

    let sessionIdForRun: string | null = null
    const shouldIgnoreRunResolution = (sessionId: string | null) => {
      const normalizedSessionId = String(sessionId || "").trim()
      if (!normalizedSessionId) return false
      if (terminalSessionIdsRef.current.has(normalizedSessionId)) return true
      const activeSessionId = String(activeSessionIdRef.current || "").trim()
      return activeSessionId !== normalizedSessionId
    }

    try {
      try { await tldwClient.initialize() } catch {}

      const entries = valid.map((r) => {
        const inferredType = r.type === "auto" ? inferIngestTypeFromUrl(r.url) : r.type
        const rowDefaults = r.defaults || normalizedTypeDefaults
        const defaultsForType = {
          audio: inferredType === "audio" ? rowDefaults.audio : undefined,
          document: inferredType === "document" || inferredType === "pdf" ? rowDefaults.document : undefined,
          video: inferredType === "video" ? rowDefaults.video : undefined
        }
        return {
          id: r.id, url: r.url, type: r.type, keywords: r.keywords,
          audio: mergeDefaults(defaultsForType.audio, r.audio),
          document: mergeDefaults(defaultsForType.document, r.document),
          video: mergeDefaults(defaultsForType.video, r.video)
        }
      })

      const fileDefaults = {
        audio: mergeDefaults(normalizedTypeDefaults.audio),
        document: mergeDefaults(normalizedTypeDefaults.document),
        video: mergeDefaults(normalizedTypeDefaults.video)
      }
      const fileDefaultsByInstanceId = new Map<string, any>()
      for (const stub of attachedFileStubs) {
        const file = fileForStubId.get(stub.id)
        if (!file) continue
        fileDefaultsByInstanceId.set(getFileInstanceId(file), stub.defaults || normalizedTypeDefaults)
      }

      const filesPayload = await Promise.all(
        attachedFiles.map(async (f) => {
          const instanceId = getFileInstanceId(f)
          const defaultsForFile = fileDefaultsByInstanceId.get(instanceId) || normalizedTypeDefaults
          if (f.size && f.size > INLINE_FILE_WARN_BYTES) {
            const msg = `File "${f.name}" is too large for inline transfer (over ${formatBytes(INLINE_FILE_WARN_BYTES)}). Please upload a smaller file or process directly on the server.`
            messageApi.error(msg)
            throw new Error(msg)
          }
          if (f.size && f.size > MAX_LOCAL_FILE_BYTES) {
            throw new Error(`File "${f.name}" is too large to ingest (over ${formatBytes(MAX_LOCAL_FILE_BYTES)}).`)
          }
          const id = fileIdByInstanceId.get(instanceId) || crypto.randomUUID()
          fileIdByInstanceId.set(instanceId, id)
          fileLookup.set(id, f)
          const data = Array.from(new Uint8Array(await f.arrayBuffer()))
          return { id, name: f.name, type: f.type, data, defaults: defaultsForFile }
        })
      )

      const requestPayload = {
        entries, files: filesPayload, storeRemote, processOnly, common, advancedValues,
        fileDefaults, chunkingTemplateName, autoApplyTemplate
      }

      const startAck = await startQuickIngestSession(requestPayload)
      if (!startAck?.ok || !startAck?.sessionId) {
        const msg = startAck?.error || qi("quickIngestStartFailed", "Quick ingest failed to start. Check tldw server settings and try again.")
        messageApi.error(msg)
        if (!unmountedRef.current) handleRunFailure(msg)
        return
      }
      const sessionId = String(startAck.sessionId).trim()
      sessionIdForRun = sessionId
      terminalSessionIdsRef.current.delete(sessionId)
      setActiveSessionId(sessionId)

      if (sessionId && !sessionId.startsWith("qi-direct-")) return

      let resp: { ok: boolean; error?: string; results?: Array<Partial<ResultItem>> } | undefined
      try {
        resp = (await submitQuickIngestBatch({
          ...requestPayload,
          __quickIngestSessionId: sessionId,
        })) as any
      } catch (sendErr: any) {
        if (shouldIgnoreRunResolution(sessionIdForRun)) return
        throw sendErr
      }

      if (unmountedRef.current || shouldIgnoreRunResolution(sessionIdForRun)) return

      if (!resp?.ok) {
        if (shouldIgnoreRunResolution(sessionIdForRun)) return
        const msg = resp?.error || "Quick ingest failed. Check tldw server settings and try again."
        messageApi.error(msg)
        if (!unmountedRef.current) handleRunFailure(msg)
        return
      }

      const normalizedResults = (resp.results || [])
        .map((item) => normalizeResultItem(item))
        .filter((item): item is ResultItem => Boolean(item))
      if (total > 0 && normalizedResults.length === 0) {
        if (shouldIgnoreRunResolution(sessionIdForRun)) return
        handleRunFailure(qi("noResultItemsReturned", "Ingest request finished without item results."))
        return
      }
      if (shouldIgnoreRunResolution(sessionIdForRun)) return
      await handleRunCompleted(normalizedResults)
    } catch (e: any) {
      if (shouldIgnoreRunResolution(sessionIdForRun)) return
      const msg = e?.message || "Quick ingest failed."
      messageApi.error(msg)
      if (!unmountedRef.current) handleRunFailure(msg)
    }
  }, [
    advancedValues, autoApplyTemplate, chunkingTemplateName, clearFailure, common,
    handleRunCompleted, handleRunFailure, ingestBlocked, ingestConnectionStatus,
    attachedFiles, attachedFileStubs, fileForStubId, formatBytes, markFailure,
    messageApi, mergeDefaults, processOnly, qi, fileTypeFromName,
    reviewBeforeStorage, rows, storeRemote, t, normalizedTypeDefaults,
    missingFileStubs.length, setActiveSessionId, setLastRunCancelled, setLastRunError,
    setTotalPlanned, setProcessedCount, setLiveTotalCount, setRunStartedAt,
    setLastRunProcessOnly, setResults, lastFileLookupRef, lastFileIdByInstanceIdRef,
    unmountedRef, results, wizardReal
  ])

  // ---- Cancel ----
  const requestCancelActiveRun = React.useCallback(async () => {
    if (!running) return
    const sessionId = String(activeSessionIdRef.current || "").trim()
    if (!sessionId) return
    const confirmed = await confirmDanger({
      title: qi("cancelRunConfirmTitle", "Cancel current ingest run?"),
      content: qi("cancelRunConfirmBody", "This stops remaining items in the current run. Completed items stay in results."),
      okText: qi("cancelRunConfirmAction", "Cancel run"),
      cancelText: qi("cancelRunKeep", "Keep running"),
      danger: true,
      autoFocusButton: "cancel"
    })
    if (!confirmed) return
    terminalSessionIdsRef.current.add(sessionId)
    handleRunCancelled(qi("cancelledByUser", "Cancelled by user."))
    try {
      const response = await cancelQuickIngestSession({ sessionId, reason: "user_cancelled" })
      if (!response?.ok && response?.error) messageApi.warning(response.error)
    } catch (error) {
      messageApi.warning(error instanceof Error ? error.message : String(error || "Cancel request failed."))
    }
  }, [confirmDanger, handleRunCancelled, messageApi, qi, running])

  const handleModalCancel = React.useCallback(() => {
    if (running) { void requestCancelActiveRun(); return }
    onClose()
  }, [onClose, requestCancelActiveRun, running])

  // ---- Auto-process ----
  React.useEffect(() => {
    if (!open) {
      autoProcessedRef.current = false
      return
    }
    if (autoProcessedRef.current) return
    if (!autoProcessQueued) return
    if (running || ingestBlocked || plannedCount <= 0) return
    autoProcessedRef.current = true
    void run()
  }, [autoProcessQueued, open, run, running, ingestBlocked, plannedCount])

  // Auto-run after "store without review" is triggered.
  React.useEffect(() => {
    if (!pendingStoreWithoutReviewRef.current) return
    if (reviewBeforeStorage) return
    pendingStoreWithoutReviewRef.current = false
    void run()
  }, [reviewBeforeStorage, run, pendingStoreWithoutReviewRef])

  // ---- Live progress from background runtime ----
  React.useEffect(() => {
    const handler = (message: any) => {
      if (!message || typeof message.type !== "string") return
      const type = String(message.type)
      if (
        type !== "tldw:quick-ingest/progress" &&
        type !== "tldw:quick-ingest/completed" &&
        type !== "tldw:quick-ingest/failed" &&
        type !== "tldw:quick-ingest/cancelled" &&
        type !== "tldw:quick-ingest-progress"
      ) return

      const payload = message.payload || {}
      const sessionId = String(payload.sessionId || "").trim()
      const activeSessionId = String(activeSessionIdRef.current || "").trim()
      if (!sessionId || !activeSessionId || sessionId !== activeSessionId) return
      if (
        terminalSessionIdsRef.current.has(sessionId) &&
        type !== "tldw:quick-ingest/progress" &&
        type !== "tldw:quick-ingest-progress"
      ) return

      if (type === "tldw:quick-ingest/progress" || type === "tldw:quick-ingest-progress") {
        const result = normalizeResultItem(payload.result as Partial<ResultItem> | undefined)
        if (typeof payload.processedCount === "number") setProcessedCount(payload.processedCount)
        if (typeof payload.totalCount === "number") {
          setLiveTotalCount(payload.totalCount)
          setTotalPlanned(payload.totalCount)
        }
        if (!result || !result.id) return
        setResults((prev) => {
          const map = new Map<string, ResultItem>()
          for (const r of prev) if (r.id) map.set(r.id, r)
          const existing = map.get(result.id)
          map.set(result.id, {
            ...(existing || {}), ...result,
            status: normalizeResultStatus(result.status),
            outcome: isCancelledMessage(result.error) ? "cancelled" : existing?.outcome
          })
          return Array.from(map.values())
        })
        return
      }

      terminalSessionIdsRef.current.add(sessionId)
      if (type === "tldw:quick-ingest/completed") {
        const normalizedResults = (payload.results || [])
          .map((item: Partial<ResultItem>) => normalizeResultItem(item))
          .filter((item: ResultItem | null): item is ResultItem => Boolean(item))
        if ((plannedRunContextRef.current?.total || 0) > 0 && normalizedResults.length === 0) {
          handleRunFailure(qi("noResultItemsReturned", "Ingest request finished without item results."))
          return
        }
        void handleRunCompleted(normalizedResults)
        return
      }
      if (type === "tldw:quick-ingest/failed") {
        handleRunFailure(String(payload.error || "").trim() || qi("statusFailed", "Quick ingest failed."))
        return
      }
      if (type === "tldw:quick-ingest/cancelled") {
        handleRunCancelled(String(payload.reason || "").trim() || qi("cancelledByUser", "Cancelled by user."))
      }
    }
    try { if (browser?.runtime?.onMessage?.addListener) browser.runtime.onMessage.addListener(handler) } catch {}
    return () => {
      try { if (browser?.runtime?.onMessage?.removeListener) browser.runtime.onMessage.removeListener(handler) } catch {}
    }
  }, [handleRunCancelled, handleRunCompleted, handleRunFailure, isCancelledMessage, qi, setProcessedCount, setLiveTotalCount, setTotalPlanned, setResults])

  // ---- Inspector intro force-reset event ----
  React.useEffect(() => {
    const forceIntro = () => {
      setShowInspectorIntro(true)
      setInspectorOpen(true)
      try { setInspectorIntroDismissed(false) } catch {}
    }
    window.addEventListener('tldw:quick-ingest-force-intro', forceIntro)
    return () => window.removeEventListener('tldw:quick-ingest-force-intro', forceIntro)
  }, [setInspectorIntroDismissed, setInspectorOpen, setShowInspectorIntro])

  // ---- Retry helpers ----
  const requeueFailed = React.useCallback(() => {
    const failedItems = resultItems.filter((r) => r.status === "error")
    const rowsById = new Map(rows.map((row) => [row.id, row]))
    const rowsByUrl = new Map(rows.map((row) => [row.url.trim(), row]))
    const fileIdByInstanceId = lastFileIdByInstanceIdRef.current
    const fileIdToStub = new Map<string, any>()
    if (fileIdByInstanceId) {
      for (const stub of queuedFileStubs) {
        if (!stub.instanceId) continue
        const fileId = fileIdByInstanceId.get(stub.instanceId)
        if (fileId) fileIdToStub.set(fileId, stub)
      }
    }
    const stubsByName = new Map<string, any[]>()
    for (const stub of queuedFileStubs) {
      const name = stub.name || ""
      if (!name) continue
      const list = stubsByName.get(name) || []
      list.push(stub)
      stubsByName.set(name, list)
    }

    const failedUrls: typeof rows = []
    const failedFileStubs: typeof queuedFileStubs = []
    let failedFileCount = 0

    for (const item of failedItems) {
      if (item.url) {
        const key = (item.url || "").trim()
        const existing = (item.id && rowsById.get(item.id)) || (key ? rowsByUrl.get(key) : undefined)
        if (existing) {
          failedUrls.push({ ...existing, id: crypto.randomUUID(), defaults: existing.defaults || createDefaultsSnapshot() })
        } else {
          failedUrls.push(buildRowEntry(item.url || "", "auto"))
        }
        continue
      }
      if (item.fileName) {
        failedFileCount += 1
        const byId = item.id ? fileIdToStub.get(item.id) : undefined
        const byName = stubsByName.get(item.fileName) || []
        const existingStub = byId || (byName.length === 1 ? byName[0] : undefined)
        if (existingStub) {
          failedFileStubs.push({ ...existingStub, id: crypto.randomUUID(), defaults: existingStub.defaults || createDefaultsSnapshot() })
        }
      }
    }

    if (failedUrls.length === 0 && failedFileCount === 0) {
      messageApi.info(qi("noFailedToRequeue", "No failed items to requeue."))
      return
    }
    if (failedUrls.length > 0) setRows((prev) => [...prev, ...failedUrls])
    if (failedFileStubs.length > 0) setQueuedFiles((prev: any) => [...(prev || []), ...failedFileStubs])
    setResults([])
    const msg = failedFileCount > 0
      ? qi("requeuedFailedWithFiles", "Requeued {{urlCount}} URL(s). Re-upload {{fileCount}} file(s) if needed.", { urlCount: failedUrls.length, fileCount: failedFileCount })
      : qi("requeuedFailed", "Requeued {{count}} failed item(s).", { count: failedUrls.length })
    messageApi.success(msg)
  }, [buildRowEntry, createDefaultsSnapshot, messageApi, qi, queuedFileStubs, resultItems, rows, setQueuedFiles, setRows, setResults, lastFileIdByInstanceIdRef])

  const confirmReplaceQueue = React.useCallback(
    async (otherCount: number) => {
      if (otherCount <= 0) return true
      return await confirmDanger({
        title: qi("retryReplaceQueueTitle", "Replace current queue?"),
        content: qi("retryReplaceQueueBody", "Retrying this item will replace the current queue and remove {{count}} other item(s). Continue?", { count: otherCount }),
        okText: qi("retryReplaceQueueConfirm", "Replace and retry"),
        cancelText: qi("cancel", "Cancel"),
        danger: false
      })
    },
    [confirmDanger, qi]
  )

  const resetQueueForRetry = React.useCallback(
    (nextRows: typeof rows, nextFiles: File[], msg: string) => {
      const defaultsSnapshot = createDefaultsSnapshot()
      const { buildQueuedFileStub } = require('./hooks')
      const nextFileStubs = nextFiles.map((file: File) => buildQueuedFileStub(file, defaultsSnapshot))
      setRows(nextRows)
      setQueuedFiles(nextFileStubs)
      setLocalFiles(nextFiles)
      setSelectedRowId(nextRows[0]?.id ?? null)
      setSelectedFileId(nextFileStubs[0]?.id ?? null)
      setResults([])
      setProcessedCount(0)
      const total = nextRows.filter((r) => r.url.trim().length > 0).length + nextFiles.length
      setTotalPlanned(total)
      setLiveTotalCount(total)
      setRunStartedAt(null)
      if (msg) messageApi.info(msg)
    },
    [createDefaultsSnapshot, messageApi, setQueuedFiles, setRows, setLocalFiles, setSelectedRowId, setSelectedFileId, setResults, setProcessedCount, setTotalPlanned, setLiveTotalCount, setRunStartedAt]
  )

  const retrySingleRow = React.useCallback(
    async (row: typeof rows[0]) => {
      if (running) return
      const totalItems = rows.filter((r) => r.url.trim().length > 0).length + queuedFileStubs.length
      const otherCount = totalItems - (row.url.trim() ? 1 : 0)
      const ok = await confirmReplaceQueue(otherCount)
      if (!ok) return
      const nextRow = { ...row, id: crypto.randomUUID() }
      resetQueueForRetry([nextRow], [], qi("queuedRetrySingle", "Queued 1 item to retry. Click Run to start."))
    },
    [confirmReplaceQueue, qi, queuedFileStubs.length, resetQueueForRetry, rows, running]
  )

  const retrySingleFile = React.useCallback(
    async (file: File) => {
      if (running) return
      const totalItems = rows.filter((r) => r.url.trim().length > 0).length + queuedFileStubs.length
      const otherCount = Math.max(0, totalItems - 1)
      const ok = await confirmReplaceQueue(otherCount)
      if (!ok) return
      resetQueueForRetry([], [file], qi("queuedRetryFile", "Queued {{name}} for retry. Click Run to start.", { name: file.name || "file" }))
    },
    [confirmReplaceQueue, qi, queuedFileStubs.length, resetQueueForRetry, rows, running]
  )

  // ---- Status with run state overlays ----
  const statusForUrlRowWithRunState = React.useCallback(
    (row: typeof rows[0]) => {
      const base = statusForUrlRow(row)
      const result = resultById.get(row.id)
      if (result?.status === "error") {
        return { label: qi("statusFailed", "Failed"), color: "red", reason: result.error || base.reason }
      }
      return base
    },
    [qi, resultById, statusForUrlRow]
  )

  const statusForFileWithRunState = React.useCallback(
    (fileLike: { size: number }, attached: boolean) => {
      const base = statusForFile(fileLike, attached)
      if (attached && typeof (fileLike as File).name === "string" && typeof (fileLike as File).lastModified === "number") {
        const match = getResultForFile(fileLike as File)
        if (match?.status === "error") {
          return { label: qi("statusFailed", "Failed"), color: "red", reason: match.error || base.reason }
        }
      }
      return base
    },
    [getResultForFile, qi, statusForFile]
  )

  // ---- Modal drag handlers ----
  const handleModalDragOver = React.useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault()
      if (activeTab !== "queue" && event.dataTransfer) event.dataTransfer.dropEffect = "none"
    },
    [activeTab]
  )

  const handleModalDrop = React.useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault()
      if (activeTab !== "queue") event.stopPropagation()
    },
    [activeTab]
  )

  // ---- Convenience ----
  const missingFileCount = missingFileStubs.length
  const draftStorageCapLabel = formatBytes(DRAFT_STORAGE_CAP_BYTES)

  // Connection banner
  let connectionBannerTitle: string | null = null
  let connectionBannerBody: string | null = null
  if (!isOnlineForIngest) {
    if (isConfiguring) {
      connectionBannerTitle = t("quickIngest.connectingTitle", "Checking connection...")
      connectionBannerBody = t("quickIngest.connectingBody", "Verifying your tldw server is reachable.")
    } else {
      connectionBannerTitle = t("quickIngest.notConnectedTitle", "Not connected to server")
      connectionBannerBody = t("quickIngest.notConnectedBody", "Configure your server to process content. Inputs are disabled until connected.")
    }
  }

  return (
    <Modal
      title={
        <div className="flex items-center gap-2">
          <span>
            {t('quickIngest.title', 'Quick Ingest')}
            {plannedCount > 0 && (
              <span className="ml-1.5 text-text-muted font-normal text-sm">
                ({plannedCount})
              </span>
            )}
          </span>
          <Button
            size="small"
            type="text"
            icon={<HelpCircle className="w-4 h-4" />}
            aria-label={qi('openInspectorIntro', 'Open Inspector intro')}
            title={qi('openInspectorIntro', 'Open Inspector intro')}
            onClick={() => {
              setShowInspectorIntro(true)
              try { setInspectorIntroDismissed(false) } catch {}
              setInspectorOpen(true)
            }}
          />
        </div>
      }
      open={open}
      onCancel={handleModalCancel}
      footer={null}
      width={760}
      style={{ maxWidth: "calc(100vw - 32px)" }}
      styles={{
        body: {
          maxHeight: "calc(100vh - 160px)",
          overflowY: "auto"
        }
      }}
      destroyOnHidden
      rootClassName="quick-ingest-modal"
      maskClosable={!running}
    >
      {contextHolder}
      <div
        className="relative"
        data-state={modalReady ? 'ready' : 'loading'}
        onDragOver={handleModalDragOver}
        onDrop={handleModalDrop}
      >
      {/* Tab Navigation */}
      <QuickIngestTabs
        activeTab={activeTab}
        onTabChange={setActiveTab}
        badges={tabBadges}
      />

      <Space orientation="vertical" className="w-full">
        {/* Connection/Onboarding banners shown on all tabs */}
        {!isOnlineForIngest && connectionBannerTitle && (
          <div className="rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
            <div className="font-medium">{connectionBannerTitle}</div>
            {connectionBannerBody && (
              <div className="mt-0.5">{connectionBannerBody}</div>
            )}
            {!isConfiguring && (
              <Button size="small" className="mt-2" onClick={openHealthDiagnostics}>
                {t("quickIngest.configureServer", "Configure server")}
              </Button>
            )}
          </div>
        )}
        {!onboardingDismissed && isOnlineForIngest && (
          <div className="rounded-md border border-primary/30 bg-primary/5 px-3 py-2 text-xs text-text">
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="font-medium mb-1">
                  {qi("onboardingTitle", "Add content to your knowledge base")}
                </div>
                <ul className="list-disc list-inside space-y-0.5 text-text-muted">
                  <li>{qi("onboardingStep1", "Paste URLs or drop files above")}</li>
                  <li>{qi("onboardingStep2", "Configure options (or use defaults)")}</li>
                  <li>{qi("onboardingStep3", "Click Process to start ingestion")}</li>
                </ul>
              </div>
              <Button
                type="text"
                size="small"
                icon={<X className="w-3.5 h-3.5" />}
                onClick={() => { try { setOnboardingDismissed(true) } catch {} }}
                aria-label={qi("dismissOnboarding", "Dismiss onboarding")}
              />
            </div>
          </div>
        )}
        {lastRunError && (
          <div className="mt-2 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
            <div className="font-medium">
              {t("quickIngest.errorSummary", "We couldn't process ingest items right now.")}
            </div>
            <div className="mt-1">
              {t("quickIngest.errorHint", "Try again after checking your tldw server. Health & diagnostics can help troubleshoot ingest issues.")}
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Button size="small" type="primary" onClick={openHealthDiagnostics} data-testid="quick-ingest-open-health">
                {t("settings:healthSummary.diagnostics", "Health & diagnostics")}
              </Button>
              <Typography.Text className="text-[11px] text-danger">{lastRunError}</Typography.Text>
            </div>
          </div>
        )}
        {draftCreationError && (
          <div className="mt-2 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
            <div className="font-medium">{qi("reviewDraftsFailedTitle", "Review drafts couldn't be created.")}</div>
            <div className="mt-1">{draftCreationError}</div>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Button size="small" type="primary" onClick={retryDraftCreation} loading={draftCreationRetrying} disabled={draftCreationRetrying || running || !hasReviewableResults}>
                {qi("reviewDraftsRetry", "Retry draft creation")}
              </Button>
              <Button size="small" onClick={downloadResultsJson} disabled={resultItems.length === 0}>
                {t("quickIngest.downloadJson") || "Download JSON"}
              </Button>
              <Button size="small" onClick={proceedWithoutReview} disabled={running}>
                {qi("reviewDraftsStoreWithoutReview", "Store without review")}
              </Button>
            </div>
          </div>
        )}
        {reviewNavigationError && (
          <div className="mt-2 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
            <div className="font-medium">{reviewNavigationError}</div>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Button
                size="small"
                type="primary"
                onClick={() => { if (!reviewBatchId) return; void tryOpenContentReview(reviewBatchId, { closeOnSuccess: true, closeDelayMs: 250 }) }}
                disabled={!reviewBatchId}
              >
                {qi("reviewNavigationRetry", "Retry opening Content Review")}
              </Button>
            </div>
          </div>
        )}

        {/* QUEUE TAB CONTENT */}
        <QueueTab isActive={activeTab === "queue"}>
        <div className="flex flex-col gap-1">
          <Typography.Text strong>{t('quickIngest.howItWorks', 'How this works')}</Typography.Text>
          <Typography.Paragraph type="secondary" className="!mb-1 text-sm text-text-muted">
            {t('quickIngest.howItWorksDesc', 'Add URLs or files, pick processing mode (store vs process-only), tweak options, then run Ingest/Process.')}
          </Typography.Paragraph>
        </div>
        <div className="rounded-md border border-border bg-surface2 px-3 py-2 text-xs text-text">
          <div className="font-medium mb-1">{qi('tipsTitle', 'Tips')}</div>
          <ul className="list-disc list-inside space-y-1">
            <li>{qi('tipsHybrid', 'Hybrid input: drop files or paste URLs (one URL per line; commas also supported) to build the queue.')}</li>
            <li>{qi('tipsPerType', 'Per-type settings (Audio/PDF/Video) apply to new items of that type.')}</li>
            <li>{qi('tipsInspector', 'Use the Inspector to see status, type, and quick checks before ingesting.')}</li>
          </ul>
        </div>
        <div className="space-y-3">
          <div className="rounded-md border border-border bg-surface p-3 text-text">
            <div className="flex items-start justify-between gap-2">
              <div>
                <Typography.Title level={5} className="!mb-1 !text-text">
                  {t('quickIngest.sourceHeading') || 'Input'}
                </Typography.Title>
                <Typography.Text className="!text-text-muted">
                  {t('quickIngest.subtitle') || 'Drop files or paste URLs; items immediately join the queue.'}
                </Typography.Text>
                <div className="text-xs text-text-subtle mt-1">
                  {qi('supportedFormats', 'Supported: docs, PDFs, audio, video, and web URLs.')}
                </div>
              </div>
              <Tag color="blue">
                {qi('itemsReady', '{{count}} item(s) ready', { count: plannedCount || 0 })}
              </Tag>
            </div>
            <input type="file" style={{ display: 'none' }} ref={reattachInputRef} onChange={handleReattachChange} accept={QUICK_INGEST_ACCEPT_STRING} />
            <FileDropZone
              onFilesAdded={addLocalFiles}
              onFilesRejected={(errors) => {
                messageApi.error(
                  errors.length === 1 ? errors[0] : qi('filesRejected', '{{count}} files rejected', { count: errors.length }),
                  errors.length > 1 ? 5 : 3
                )
              }}
              running={running}
              isOnlineForIngest={isOnlineForIngest}
            />
            <div className="mt-2 flex justify-center">
              <Button onClick={pasteFromClipboard} disabled={running || !isOnlineForIngest} aria-label={qi('pasteFromClipboard', 'Paste URLs from clipboard')} title={qi('pasteFromClipboard', 'Paste URLs from clipboard')}>
                {qi('pasteFromClipboard', 'Paste URLs from clipboard')}
              </Button>
            </div>
            {queuedFileStubs.length > 0 && (
              <div className="mt-2 flex items-start gap-2 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
                <AlertTriangle className="mt-0.5 h-4 w-4" />
                <span>{qi("localFilesWarning", "Local files stay attached only while this modal is open. Keep it open until you click Run, or reattach files after reopening.")}</span>
              </div>
            )}
            <div className="mt-3 space-y-2">
              <div className="flex items-center justify-between">
                <Typography.Text strong>{qi('pasteUrlsTitle', 'Paste URLs')}</Typography.Text>
                <Typography.Text className="text-xs text-text-subtle">{qi('pasteUrlsHint', 'One URL per line (commas also supported)')}</Typography.Text>
              </div>
              <label htmlFor="quick-ingest-url-input" className="text-xs font-medium text-text">{qi('urlsLabel', 'URLs to ingest')}</label>
              <div className="flex w-full items-start gap-2">
                <Input.TextArea
                  id="quick-ingest-url-input"
                  autoSize={{ minRows: 3, maxRows: 8 }}
                  placeholder={qi('urlsPlaceholder', 'https://example.com\nhttps://example.org')}
                  value={pendingUrlInput}
                  onChange={(e) => setPendingUrlInput(e.target.value)}
                  disabled={running || !isOnlineForIngest}
                  aria-label={qi('urlsInputAria', 'Paste URLs input')}
                  title={qi('urlsInputAria', 'Paste URLs input')}
                />
                <Button type="primary" className="shrink-0" onClick={() => void addUrlsFromInput(pendingUrlInput)} disabled={running || !isOnlineForIngest} aria-label={qi('addUrlsAria', 'Add URLs to queue')} title={qi('addUrlsAria', 'Add URLs to queue')}>
                  {qi('addUrls', 'Add URLs')}
                </Button>
              </div>
              <div className="flex items-center gap-2 text-xs text-text-muted">
                <AlertTriangle className="w-4 h-4 text-text-subtle" />
                <span>{qi('authRequiredHint', 'Authentication-required pages may need cookies set in Advanced.')}</span>
              </div>
            </div>
          </div>

          <div className="rounded-md border border-border bg-surface p-3 text-text">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Typography.Title level={5} className="!mb-0 !text-text">{qi('queueTitle', 'Queue')}</Typography.Title>
              <div className="flex items-center gap-2">
                <Button size="small" onClick={clearAllQueues} disabled={running && plannedCount > 0} aria-label={qi('clearAllAria', 'Clear all queued items')} title={qi('clearAllAria', 'Clear all queued items')}>{qi('clearAll', 'Clear all')}</Button>
                <Button size="small" onClick={addRow} disabled={running || !isOnlineForIngest} aria-label={qi('addBlankRowAria', 'Add blank URL row')} title={qi('addBlankRowAria', 'Add blank URL row')}>{qi('addBlankRow', 'Add blank row')}</Button>
                <Button size="small" aria-label={qi('openInspector', 'Open Inspector')} title={qi('openInspector', 'Open Inspector')} onClick={() => setInspectorOpen(true)} disabled={!(selectedRow || selectedFileStub)}>{qi('openInspector', 'Open Inspector')}</Button>
              </div>
            </div>
            <div className="text-xs text-text-subtle mb-2">
              {qi('queueDescription', 'Queued items appear here. Click a row to open the Inspector; badges show defaults, custom edits, or items needing attention.')}
            </div>
            {missingFileStubs.length > 0 && (
              <div className="mb-2 flex items-start gap-2 rounded-md border border-warn/30 bg-warn/10 px-2 py-1 text-xs text-warn">
                <AlertTriangle className="mt-0.5 h-4 w-4" />
                <span>{qi("missingFilesQueueNotice", "{{count}} local file(s) need reattachment. Add the files again or remove them before running ingest.", { count: missingFileStubs.length })}</span>
              </div>
            )}
            <div className="max-h-[360px] space-y-2 overflow-auto pr-1">
              <React.Suspense fallback={null}>
                {rows.map((row) => {
                  const baseStatus = statusForUrlRow(row)
                  const isSelected = selectedRowId === row.id
                  const detected = row.type === "auto" ? inferIngestTypeFromUrl(row.url) : row.type
                  const res = resultById.get(row.id)
                  const status = res?.status === "error"
                    ? { label: qi("statusFailed", "Failed"), color: "red", reason: res.error || baseStatus.reason }
                    : baseStatus
                  const isProcessing = running && !res?.status
                  let runTag: React.ReactNode = null
                  if (res?.status === "ok") runTag = <Tag color="green">{qi("statusDone", "Done")}</Tag>
                  else if (res?.status === "error") runTag = <AntTooltip title={res.error || qi("statusFailed", "Failed")}><Tag color="red">{qi("statusFailed", "Failed")}</Tag></AntTooltip>
                  else if (running) runTag = <Tag icon={<Spin size="small" />} color="blue">{qi("statusRunning", "Running")}</Tag>
                  const pendingTag = ingestBlocked && !running && row.url.trim().length > 0 && (!res || !res.status) ? <Tag>{pendingLabel}</Tag> : null
                  const processingIndicator = isProcessing ? <div className="mt-1"><ProcessingIndicator label={qi("processingItem", "Processing...")} /></div> : null

                  return (
                    <QueuedItemRow
                      key={row.id} row={row} isSelected={isSelected} detectedType={detected} status={status}
                      runTag={runTag} pendingTag={pendingTag} processingIndicator={processingIndicator}
                      running={running} queueDisabled={ingestBlocked} canRetry={res?.status === "error"} qi={qi} typeIcon={typeIcon}
                      onSelect={() => { setSelectedRowId(row.id); setSelectedFileId(null); setInspectorOpen(true) }}
                      onOpenInspector={() => { setSelectedRowId(row.id); setSelectedFileId(null); setInspectorOpen(true) }}
                      onUpdateRow={(updates) => updateRow(row.id, updates)}
                      onRetry={() => { void retrySingleRow(row) }}
                      onRemove={() => removeRow(row.id)}
                    />
                  )
                })}

                {queuedFileStubs.map((stub) => {
                  const attachedFile = fileForStubId.get(stub.id)
                  const baseStatus = statusForFile(attachedFile || stub, Boolean(attachedFile))
                  const isSelected = selectedFileId === stub.id
                  const type = fileTypeFromName(stub)
                  const match = attachedFile ? getResultForFile(attachedFile) : null
                  const runStatus = match?.status
                  const status = runStatus === "error"
                    ? { label: qi("statusFailed", "Failed"), color: "red", reason: match?.error || baseStatus.reason }
                    : baseStatus
                  const isProcessing = !!attachedFile && running && !runStatus
                  let runTag: React.ReactNode = null
                  if (runStatus === "ok") runTag = <Tag color="green">{qi("statusDone", "Done")}</Tag>
                  else if (runStatus === "error") runTag = <AntTooltip title={match?.error || qi("statusFailed", "Failed")}><Tag color="red">{qi("statusFailed", "Failed")}</Tag></AntTooltip>
                  else if (running && attachedFile) runTag = <Tag icon={<Spin size="small" />} color="blue">{qi("statusRunning", "Running")}</Tag>
                  const pendingTag = ingestBlocked && !running && attachedFile && !runStatus ? <Tag>{pendingLabel}</Tag> : null
                  const processingIndicator = isProcessing ? <div className="mt-1"><ProcessingIndicator label={qi("processingItem", "Processing...")} /></div> : null
                  const handleRemove = () => {
                    setQueuedFiles((prev: any) => (prev || []).filter((file: any) => file.id !== stub.id))
                    if (attachedFile) {
                      const instanceId = getFileInstanceId(attachedFile)
                      setLocalFiles((prev) => prev.filter((file) => getFileInstanceId(file) !== instanceId))
                    }
                    if (selectedFileId === stub.id) setSelectedFileId(null)
                  }

                  return (
                    <QueuedFileRow
                      key={stub.id} stub={stub} isSelected={isSelected} status={status} fileType={type}
                      sizeLabel={formatBytes(stub.size)} runTag={runTag} pendingTag={pendingTag}
                      processingIndicator={processingIndicator} running={running} queueDisabled={ingestBlocked}
                      showReattach={!attachedFile} canRetry={runStatus === "error" && Boolean(attachedFile)} qi={qi} typeIcon={typeIcon}
                      onSelect={() => { setSelectedFileId(stub.id); setSelectedRowId(null); setInspectorOpen(true) }}
                      onOpenInspector={() => { setSelectedFileId(stub.id); setSelectedRowId(null); setInspectorOpen(true) }}
                      onReattach={() => requestFileReattach(stub.id)}
                      onRetry={() => { if (attachedFile) void retrySingleFile(attachedFile) }}
                      onRemove={handleRemove}
                    />
                  )
                })}
              </React.Suspense>

              {rows.length === 0 && queuedFileStubs.length === 0 && (
                <div className="rounded-md border border-dashed border-border p-4 text-center text-sm text-text-muted space-y-1.5">
                  <div>{qi("emptyQueue", "No items queued yet.")}</div>
                  <div className="text-xs">{qi("emptyQueueHint", "Paste URLs (one per line) or drop files here.")}</div>
                  <div className="text-xs text-text-subtle">{qi("emptyQueueExample", "Try a YouTube URL, PDF, or article link.")}</div>
                  <div className="text-[11px] text-text-subtle pt-1">{qi("emptyQueueShortcut", "Tip: Use Ctrl+V to paste from clipboard")}</div>
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="mt-3 flex justify-end">
          <ProcessButton
            plannedCount={plannedCount} running={running} ingestBlocked={ingestBlocked}
            hasMissingFiles={hasMissingFiles} missingFileCount={missingFileCount}
            onRun={run} onCancel={requestCancelActiveRun} storeRemote={storeRemote} reviewBeforeStorage={reviewBeforeStorage}
          />
        </div>
        </QueueTab>

        {/* OPTIONS TAB CONTENT */}
        <OptionsTab
          isActive={activeTab === "options"}
          qi={qi} t={t}
          activePreset={presets.activePreset}
          onPresetChange={presets.handlePresetChange}
          onPresetReset={presets.handlePresetReset}
          hasAudioItems={hasAudioItems} hasDocumentItems={hasDocumentItems} hasVideoItems={hasVideoItems}
          running={running} ingestBlocked={ingestBlocked}
          common={common} setCommon={optionsReal.setCommon}
          normalizedTypeDefaults={normalizedTypeDefaults} setTypeDefaults={setTypeDefaults}
          transcriptionModelOptions={transcriptionModelChoices} transcriptionModelsLoading={transcriptionModelsLoading}
          transcriptionModelValue={transcriptionModelValue} onTranscriptionModelChange={handleTranscriptionModelChange}
          ragEmbeddingLabel={ragEmbeddingLabel} openModelSettings={openModelSettings}
          storeRemote={storeRemote} setStoreRemote={setStoreRemote}
          reviewBeforeStorage={reviewBeforeStorage} handleReviewToggle={handleReviewToggle}
          storageLabel={storageLabel} storageHintSeen={storageHintSeen} setStorageHintSeen={setStorageHintSeen}
          draftStorageCapLabel={draftStorageCapLabel}
          doneCount={doneCount} totalCount={totalCount} plannedCount={plannedCount}
          progressMeta={progressMeta} lastRunError={lastRunError}
          run={run} hasMissingFiles={hasMissingFiles} missingFileCount={missingFileCount}
          ingestConnectionStatus={ingestConnectionStatus} checkOnce={checkOnce}
          onClose={onClose}
          chunkingTemplateName={chunkingTemplateName} setChunkingTemplateName={setChunkingTemplateName}
          autoApplyTemplate={autoApplyTemplate} setAutoApplyTemplate={setAutoApplyTemplate}
        />

        {/* Inspector drawer glow and drawer - visible when queue tab active */}
        {activeTab === 'queue' && (
        <React.Fragment>
        <div
          aria-hidden
          className={`pointer-events-none absolute inset-0 transition-opacity duration-300 ease-out ${inspectorOpen && (selectedRow || selectedFileStub) ? 'opacity-100' : 'opacity-0'}`}
        >
          <div className="absolute right-0 top-0 h-full w-40 bg-gradient-to-l from-primary/30 via-primary/10 to-transparent blur-md" />
        </div>

        <QuickIngestInspectorDrawer
          open={inspectorOpen && (!!selectedRow || !!selectedFileStub)}
          onClose={handleCloseInspector}
          showIntro={showInspectorIntro}
          onDismissIntro={handleDismissInspectorIntro}
          qi={qi}
          selectedRow={selectedRow}
          selectedFile={selectedFile || selectedFileStub}
          selectedFileAttached={Boolean(selectedFile)}
          typeIcon={typeIcon}
          inferIngestTypeFromUrl={inferIngestTypeFromUrl}
          fileTypeFromName={fileTypeFromName}
          statusForUrlRow={statusForUrlRowWithRunState}
          statusForFile={statusForFileWithRunState}
          formatBytes={formatBytes}
          onReattachFile={handleReattachSelectedFile}
        />
        </React.Fragment>
        )}

        {/* Advanced options - visible when options tab active */}
        {activeTab === 'options' && (
        <Collapse
          className="mt-3"
          activeKey={advancedOpen ? ['adv'] : []}
          onChange={(k) => setAdvancedOpen(Array.isArray(k) ? k.includes('adv') : Boolean(k))}
          items={[{
          key: 'adv',
          label: (
            <div className="flex flex-col gap-1 w-full">
              <div className="flex items-center gap-2">
                <span>{qi('advancedOptionsTitle', 'Advanced options')}</span>
                <Tag color="blue">{t('quickIngest.advancedSummary', '{{count}} advanced fields loaded', { count: resolvedAdvSchema.length })}</Tag>
                {modifiedAdvancedCount > 0 && <Tag color="gold">{t('quickIngest.modifiedCount', '{{count}} modified', { count: modifiedAdvancedCount })}</Tag>}
                {specSourceLabel && <Tag color="geekblue">{specSourceLabel}</Tag>}
                {lastRefreshedLabel && <Typography.Text className="text-[11px] text-text-subtle">{t('quickIngest.advancedRefreshed', 'Refreshed {{time}}', { time: lastRefreshedLabel })}</Typography.Text>}
                {specSource !== 'none' && (
                  <AntTooltip title={<div className="max-w-80 text-xs">{specSource === 'server' ? qi('specTooltipLive', 'Using live server OpenAPI spec') : qi('specTooltipFallback', 'No spec detected; using fallback fields')}</div>}>
                    <Info className="w-4 h-4 text-text-subtle" />
                  </AntTooltip>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs text-text-subtle ml-auto">
                {ragEmbeddingLabel && <Typography.Text className="text-[11px] text-text-subtle">{t('quickIngest.ragEmbeddingHint', 'RAG embedding model: {{label}}', { label: ragEmbeddingLabel })}</Typography.Text>}
                <Button size="small" type="default" aria-label={qi('resetInspectorIntro', 'Reset Inspector intro helper')} title={qi('resetInspectorIntro', 'Reset Inspector intro helper')}
                  onClick={(e) => { e.stopPropagation(); setInspectorIntroDismissed(false); setShowInspectorIntro(true); setInspectorOpen(true) }}>
                  {qi('resetInspectorIntro', 'Reset Inspector Intro')}
                </Button>
                <Space size="small" align="center">
                  <span className="text-xs text-text-subtle">{qi('preferServerLabel', 'Prefer server')}</span>
                  <Switch size="small" aria-label={qi('preferServerAria', 'Advanced options \u2013 prefer server OpenAPI spec')} title={qi('preferServerTitle', 'Prefer server OpenAPI spec')}
                    checked={!!specPrefs?.preferServer}
                    onChange={async (v) => { persistSpecPrefs({ ...(specPrefs || {}), preferServer: v }); await loadSpec(v, { reportDiff: true, persist: true, forceFetch: v }) }}
                  />
                </Space>
                <Button size="small" aria-label={qi('reloadSpecAria', 'Reload advanced spec from server')} title={qi('reloadSpecAria', 'Reload advanced spec from server')}
                  onClick={(e) => { e.stopPropagation(); void loadSpec(true, { reportDiff: true, persist: true, forceFetch: true }) }}>
                  {qi('reloadFromServer', 'Reload from server')}
                </Button>
                <span className="h-4 border-l border-border" aria-hidden />
                <Button size="small" aria-label={qi('saveAdvancedDefaultsAria', 'Save current advanced options as defaults')} title={qi('saveAdvancedDefaultsAria', 'Save current advanced options as defaults')}
                  disabled={!advancedDefaultsDirty}
                  onClick={(e) => {
                    e.stopPropagation()
                    try {
                      setSavedAdvValues(advancedValues)
                      messageApi.success(qi('advancedSaved', 'Advanced options saved as defaults for future sessions.'))
                    } catch {
                      messageApi.error(qi('advancedSaveFailed', 'Could not save advanced defaults \u2014 storage quota may be limited.'))
                    }
                  }}>
                  {qi('saveAdvancedDefaults', 'Save as default')}
                </Button>
                <span className="h-4 border-l border-border" aria-hidden />
                <Button size="small" danger aria-label={qi('resetAdvancedAria', 'Reset advanced options and UI state')} title={qi('resetAdvancedAria', 'Reset advanced options and UI state')}
                  onClick={async (e) => {
                    e.stopPropagation()
                    const ok = await confirmDanger({ title: qi('confirmResetTitle', 'Please confirm'), content: qi('confirmResetContent', 'Reset all advanced options and UI state?'), okText: qi('reset', 'Reset'), cancelText: qi('cancel', 'Cancel') })
                    if (!ok) return
                    setAdvancedValues({})
                    setSavedAdvValues({})
                    setFieldDetailsOpen({})
                    setUiPrefs({ advancedOpen: false, fieldDetailsOpen: {} })
                    setAdvSearch('')
                    setAdvancedOpen(false)
                    messageApi.success(qi('advancedReset', 'Advanced options reset'))
                  }}>
                  {qi('resetAdvanced', 'Reset Advanced')}
                </Button>
              </div>
            </div>
          ),
          children: (
        <Space orientation="vertical" className="w-full">
              <div className="flex items-center gap-2">
                <Input allowClear placeholder={qi('searchAdvanced', 'Search advanced fields...')} value={advSearch} onChange={(e) => setAdvSearch(e.target.value)} className="max-w-80" aria-label={qi('searchAdvanced', 'Search advanced fields...')} title={qi('searchAdvanced', 'Search advanced fields...')} />
                {modifiedAdvancedCount > 0 && <Tag color="gold">{t('quickIngest.modifiedCount', '{{count}} modified', { count: modifiedAdvancedCount })}</Tag>}
              </div>
              {resolvedAdvSchema.length === 0 ? (
                <Typography.Text type="secondary">{t('quickIngest.advancedEmpty', 'No advanced options detected \u2014 try reloading the spec.')}</Typography.Text>
              ) : (
                (() => {
                  const grouped: Record<string, typeof resolvedAdvSchema> = {}
                  const recommended: typeof resolvedAdvSchema = []
                  const q = advSearch.trim().toLowerCase()
                  const match = (f: { name: string; title?: string; description?: string }) => {
                    if (!q) return true
                    return f.name.toLowerCase().includes(q) || (f.title || '').toLowerCase().includes(q) || (f.description || '').toLowerCase().includes(q)
                  }
                  const allMatched = resolvedAdvSchema.filter(match).filter((field) => field.name !== "transcription_model")

                  for (const f of allMatched) {
                    const logical = logicalGroupForField(f.name)
                    const isRec = isRecommendedField(f.name, logical)
                    if (isRec && recommended.length < MAX_RECOMMENDED_FIELDS) recommended.push(f)
                    if (!grouped[logical]) grouped[logical] = []
                    grouped[logical].push(f)
                  }

                  const recommendedNameSet = new Set(recommended.map((f) => f.name))
                  const order: string[] = []
                  if (recommended.length > 0) order.push('Recommended')
                  order.push(...Object.keys(grouped).filter((g) => g !== 'Recommended').sort())

                  return order.map((g) => (
                    <div key={g} className="mb-2">
                      <Typography.Title level={5} className="!mb-2 flex items-center">
                        {iconForGroup(g)}
                        {g === 'Recommended' ? t('quickIngest.recommendedGroup', 'Recommended fields') : g}
                      </Typography.Title>
                      <Space orientation="vertical" className="w-full">
                        {(g === "Recommended" ? recommended : grouped[g]).map((f) => {
                          const v = advancedValues[f.name]
                          const setV = (nv: any) => setAdvancedValue(f.name, nv)
                          const isOpen = fieldDetailsOpen[f.name]
                          const setOpen = (open: boolean) => setFieldDetailsOpen((prev) => ({ ...prev, [f.name]: open }))
                          const isTranscriptionModel = f.name === "transcription_model"
                          const isContextualModel = f.name === "contextual_llm_model"
                          const isEmbeddingModel = f.name === "embedding_model"
                          const ariaLabel = `${g} \u2013 ${f.title || f.name}`
                          const isAlsoRecommended = g !== "Recommended" && recommendedNameSet.has(f.name)
                          const canShowDetailsHere = !!f.description && (g === "Recommended" || !recommendedNameSet.has(f.name))
                          const selectOptions = getAdvancedFieldSelectOptions({ fieldName: f.name, fieldEnum: f.enum, t, chatModels, embeddingModels })
                          const fallbackEnumOptions = f.enum && f.enum.length > 0 ? f.enum.map((entry) => ({ value: String(entry), label: String(entry) })) : null
                          const selectValue = v === undefined || v === null || v === "" ? undefined : String(v)
                          const resolvedSelectOptions = selectOptions ? ensureSelectOption(selectOptions, selectValue) : fallbackEnumOptions ? ensureSelectOption(fallbackEnumOptions, selectValue) : null
                          const Label = (
                            <div className="flex items-center gap-1">
                              <span className="min-w-60 text-sm">{f.title || f.name}</span>
                              {isAlsoRecommended && <Tag color="blue" className="border-0 text-[10px] leading-none px-1 py-0">{qi('recommendedBadge', 'Recommended')}</Tag>}
                              {f.description ? <AntTooltip placement="right" trigger={["hover","click"]} title={<div className="max-w-96 text-xs">{f.description}</div>}><HelpCircle className="w-3.5 h-3.5 text-text-subtle cursor-help" /></AntTooltip> : null}
                            </div>
                          )
                          if (resolvedSelectOptions) {
                            return (
                              <div key={f.name} className="flex items-center gap-2">
                                {Label}
                                <Select className="w-72" allowClear showSearch={isTranscriptionModel || isContextualModel || isEmbeddingModel || resolvedSelectOptions.length > 6}
                                  loading={(isTranscriptionModel && transcriptionModelsLoading) || (isContextualModel && chatModelsLoading) || (isEmbeddingModel && embeddingModelsLoading)}
                                  aria-label={ariaLabel} value={selectValue} onChange={setV as any} options={resolvedSelectOptions} />
                                {canShowDetailsHere && <button className="text-xs underline text-text-subtle" onClick={() => setOpen(!isOpen)}>{isOpen ? qi('hideDetails', 'Hide details') : qi('showDetails', 'Show details')}</button>}
                              </div>
                            )
                          }
                          if (f.type === 'boolean') {
                            const boolState = v === true || v === 'true' ? 'true' : v === false || v === 'false' ? 'false' : 'unset'
                            return (
                              <div key={f.name} className="flex items-center gap-2">
                                {Label}
                                <Switch checked={boolState === 'true'} onChange={(checked) => setAdvancedValue(f.name, checked)} aria-label={ariaLabel} />
                                <Button size="small" onClick={() => setAdvancedValue(f.name, undefined)} disabled={boolState === 'unset'}>{qi('unset', 'Unset')}</Button>
                                <Typography.Text type="secondary" className="text-[11px] text-text-subtle">
                                  {boolState === 'unset' ? qi('unsetLabel', 'Currently unset (server defaults)') : boolState === 'true' ? qi('onLabel', 'On') : qi('offLabel', 'Off')}
                                </Typography.Text>
                                {canShowDetailsHere && <button className="text-xs underline text-text-subtle" onClick={() => setOpen(!isOpen)}>{isOpen ? qi('hideDetails', 'Hide details') : qi('showDetails', 'Show details')}</button>}
                              </div>
                            )
                          }
                          if (f.type === 'integer' || f.type === 'number') {
                            return (
                              <div key={f.name} className="flex items-center gap-2">
                                {Label}
                                <InputNumber className="w-40" aria-label={ariaLabel} value={v} onChange={setV as any} />
                                {canShowDetailsHere && <button className="text-xs underline text-text-subtle" onClick={() => setOpen(!isOpen)}>{isOpen ? qi('hideDetails', 'Hide details') : qi('showDetails', 'Show details')}</button>}
                              </div>
                            )
                          }
                          return (
                            <div key={f.name} className="flex items-center gap-2">
                              {Label}
                              <Input className="w-96" aria-label={ariaLabel} value={v} onChange={(e) => setV(e.target.value)} />
                              {canShowDetailsHere && <button className="text-xs underline text-text-subtle" onClick={() => setOpen(!isOpen)}>{isOpen ? qi('hideDetails', 'Hide details') : qi('showDetails', 'Show details')}</button>}
                            </div>
                          )
                        })}
                      {/* details sections */}
                      {(g === "Recommended" ? recommended : grouped[g]).map((f) => {
                        const showHere = !!f.description && fieldDetailsOpen[f.name] && (g === "Recommended" || !recommendedNameSet.has(f.name))
                        return showHere ? <div key={`${f.name}-details`} className="ml-4 mt-1 p-2 rounded bg-surface2 text-xs text-text-muted max-w-[48rem]">{f.description}</div> : null
                      })}
                      </Space>
                    </div>
                  ))
                })()
              )}
            </Space>
          )
        }]} />
        )}

        {/* RESULTS TAB CONTENT */}
        <ResultsTab
          isActive={activeTab === "results"}
          processButton={
            <ProcessButton
              plannedCount={plannedCount} running={running} ingestBlocked={ingestBlocked}
              hasMissingFiles={hasMissingFiles} missingFileCount={missingFileCount}
              onRun={run} onCancel={requestCancelActiveRun} storeRemote={storeRemote} reviewBeforeStorage={reviewBeforeStorage}
            />
          }
          data={{
            results: resultsWithOutcome,
            visibleResults,
            resultSummary,
            running,
            progressMeta,
            filters: {
              value: resultsFilter,
              options: RESULT_FILTERS,
              onChange: (value) => setResultsFilter(value as ResultsFilter)
            }
          }}
          context={{
            shouldStoreRemote,
            firstResultWithMedia,
            reviewBatchId,
            processOnly,
            mediaIdFromPayload,
            titleFromPayload
          }}
          actions={{
            retryFailedUrls,
            requeueFailed,
            exportFailedList,
            tryOpenContentReview: (batchId) => { void tryOpenContentReview(batchId) },
            openInMediaViewer,
            discussInChat,
            downloadJson,
            openHealthDiagnostics
          }}
          i18n={{ qi, t }}
        />
      </Space>
      </div>
    </Modal>
  )
}

export default QuickIngestModal
