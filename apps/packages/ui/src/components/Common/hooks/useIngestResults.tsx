import React from 'react'
import type { MessageInstance } from 'antd/es/message/interface'
import { useNavigate } from "react-router-dom"
import { browser } from "wxt/browser"
import type { ResultOutcome } from "../QuickIngest/types"
import {
  DRAFT_STORAGE_CAP_BYTES,
  storeDraftAsset,
  upsertContentDraft,
  upsertDraftBatch
} from "@/db/dexie/drafts"
import type { ContentDraft, DraftBatch } from "@/db/dexie/types"
import { setSetting } from "@/services/settings/registry"
import {
  DISCUSS_MEDIA_PROMPT_SETTING,
  LAST_MEDIA_ID_SETTING
} from "@/services/settings/ui-settings"
import { resolvePerformChunking } from "@/services/tldw/ingest-defaults"
import { useQuickIngestStore } from "@/store/quick-ingest"
import { detectSections } from "@/utils/content-review"
import {
  coerceDraftMediaType,
} from "@/services/tldw/media-routing"
import type { Entry, TypeDefaults } from './useIngestQueue'
import { getFileInstanceId, snapshotTypeDefaults, buildQueuedFileStub } from './useIngestQueue'

// ---------------------------------------------------------------------------
// Processing / result types (mirrored from QuickIngestModal)
// ---------------------------------------------------------------------------

export type ProcessingItem = {
  id?: string | number
  media_id?: string | number
  pk?: string | number
  uuid?: string | number
  media?: ProcessingItem
  status?: string
  url?: string
  input_ref?: string
  media_type?: string
  content?: string | Array<string | number>
  text?: string
  transcript?: string
  transcription?: string
  summary?: string
  analysis_content?: string | null
  analysis?: string | null
  analysis_status?: string
  analysis_error?: string
  warnings?: string[]
  prompt?: string
  custom_prompt?: string
  title?: string
  metadata?: Record<string, any>
  keywords?: string[] | string
  segments?: Record<string, any>[]
}

export type ProcessingResultPayload =
  | ProcessingItem[]
  | ProcessingItem
  | {
      results?: ProcessingItem[]
      articles?: ProcessingItem[]
      result?: ProcessingItem | ProcessingItem[]
    }
  | null
  | undefined

export type ResultItem = {
  id: string
  status: 'ok' | 'error'
  outcome?: ResultOutcome
  url?: string
  fileName?: string
  type: string
  data?: ProcessingResultPayload
  error?: string
  warning?: string
}

type ProcessingOptions = {
  perform_analysis: boolean
  perform_chunking: boolean
  overwrite_existing: boolean
  advancedValues: Record<string, any>
}

type OptionsHash = `#${string}`

type PlannedRunContext = {
  total: number
  plannedUrlEntries: Array<{
    id: string
    url: string
    type: Entry["type"]
  }>
  plannedFiles: Array<{
    file: File
    type: Entry["type"]
  }>
  fileLookup: Map<string, File>
  fileIdByInstanceId: Map<string, string>
}

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

const SKIPPED_STATUS_TOKENS = [
  "skip", "skipped", "duplicate", "exists", "already", "cached", "unchanged"
]

const normalizeStatusLabel = (value: unknown) =>
  String(value || "").trim().toLowerCase()

const isSkippedStatus = (status: string) =>
  SKIPPED_STATUS_TOKENS.some((token) => status.includes(token))

const RESULT_SUCCESS_STATUS_TOKENS = [
  "ok", "success", "succeeded", "completed", "complete", "done", "ingested", "processed", "ready"
]

const RESULT_FAILURE_STATUS_TOKENS = [
  "error", "failed", "failure", "cancelled", "canceled", "timeout", "auth_required", "quarantined"
]

export const normalizeResultStatus = (status: unknown): ResultItem["status"] => {
  const normalized = normalizeStatusLabel(status)
  if (normalized && RESULT_SUCCESS_STATUS_TOKENS.includes(normalized)) return "ok"
  if (normalized && RESULT_FAILURE_STATUS_TOKENS.includes(normalized)) return "error"
  return "error"
}

export function mediaIdFromPayload(
  data: ProcessingResultPayload,
  visited?: WeakSet<object>
): string | number | null {
  if (!data || typeof data !== "object") return null
  if (Array.isArray(data)) return data.length > 0 ? mediaIdFromPayload(data[0], visited) : null
  if (!visited) visited = new WeakSet<object>()
  if (visited.has(data as object)) return null
  visited.add(data as object)

  if ("results" in data && Array.isArray(data.results) && data.results.length > 0)
    return mediaIdFromPayload(data.results[0], visited)
  if ("articles" in data && Array.isArray(data.articles) && data.articles.length > 0)
    return mediaIdFromPayload(data.articles[0], visited)
  if ("result" in data && data.result)
    return mediaIdFromPayload(Array.isArray(data.result) ? data.result[0] : data.result, visited)

  if (isProcessingItem(data)) {
    const direct = data.id ?? data.media_id ?? data.pk ?? data.uuid
    if (direct !== undefined && direct !== null) return direct
    if (data.media && typeof data.media === "object") return mediaIdFromPayload(data.media, visited)
  }
  return null
}

export function titleFromPayload(data: ProcessingResultPayload): string | null {
  const items = extractProcessingItems(data)
  if (items.length === 0) return null
  const title = items[0]?.title || items[0]?.metadata?.title
  return typeof title === "string" && title.trim() ? title.trim() : null
}

const normalizeKeywords = (value: ProcessingItem['keywords']): string[] => {
  if (Array.isArray(value)) return value.map((v) => String(v || "").trim()).filter(Boolean)
  if (typeof value === "string") return value.split(",").map((v) => v.trim()).filter(Boolean)
  return []
}

const resolveContent = (item: ProcessingItem): string => {
  if (Array.isArray(item?.content)) {
    return item.content
      .filter((v: unknown) => typeof v === "string" || typeof v === "number")
      .map((v: string | number) => String(v))
      .join("\n")
  }
  const candidates = [item?.content, item?.text, item?.transcript, item?.transcription, item?.summary, item?.analysis_content]
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) return candidate
  }
  return ""
}

const resolveTitle = (item: ProcessingItem, fallback: string): string => {
  const title = item?.title || item?.metadata?.title || item?.input_ref || fallback
  return String(title || "").trim() || fallback
}

const resolveAnalysis = (item: ProcessingItem): string | undefined => {
  if (typeof item?.analysis === "string" && !analysisWarningFromText(item.analysis)) return item.analysis
  if (typeof item?.analysis_content === "string" && !analysisWarningFromText(item.analysis_content)) return item.analysis_content
  return undefined
}

const resolvePrompt = (item: ProcessingItem): string | undefined => {
  if (typeof item?.prompt === "string") return item.prompt
  if (typeof item?.custom_prompt === "string") return item.custom_prompt
  return undefined
}

const inferContentFormat = (content: string): "plain" | "markdown" => {
  const text = String(content || "")
  if (/(^|\n)#{1,6}\s+\S/.test(text)) return "markdown"
  if (/```/.test(text)) return "markdown"
  if (/(^|\n)\s*[-*]\s+\S/.test(text)) return "markdown"
  return "plain"
}

const isResultsWrapper = (value: unknown): value is { results: ProcessingItem[] } => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  return Array.isArray((value as { results?: unknown }).results)
}

const isArticlesWrapper = (value: unknown): value is { articles: ProcessingItem[] } => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  return Array.isArray((value as { articles?: unknown }).articles)
}

const isResultWrapper = (value: unknown): value is { result: ProcessingItem | ProcessingItem[] } => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const result = (value as { result?: unknown }).result
  return Array.isArray(result) || (typeof result === "object" && result !== null)
}

const isProcessingItem = (value: unknown): value is ProcessingItem => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  if (isResultsWrapper(value) || isArticlesWrapper(value) || isResultWrapper(value)) return false
  return (
    "id" in value || "media_id" in value || "pk" in value || "uuid" in value ||
    "content" in value || "text" in value || "transcript" in value ||
    "analysis" in value || "analysis_content" in value
  )
}

const extractProcessingItems = (data: ProcessingResultPayload): ProcessingItem[] => {
  if (!data) return []
  if (Array.isArray(data)) return data
  if (isResultsWrapper(data)) return data.results
  if (isArticlesWrapper(data)) return data.articles
  if (isResultWrapper(data)) return Array.isArray(data.result) ? data.result : [data.result]
  if (isProcessingItem(data)) return [data]
  return []
}

const getProcessingStatusLabels = (data: ProcessingResultPayload): string[] =>
  extractProcessingItems(data).map((item) => normalizeStatusLabel(item.status)).filter(Boolean)

const analysisWarningFromText = (value: unknown): string | null => {
  const text = String(value || "").trim()
  if (!text) return null
  const normalized = text.toLowerCase()
  const reason = text.replace(/^error:\s*/i, "").replace(/^\[|\]$/g, "").trim()
  if (
    normalized.includes("analysis api provider is required") ||
    normalized.includes("no api specified") ||
    normalized.includes("choose an analysis provider")
  ) {
    return "Analysis skipped: choose an analysis provider, then retry analysis."
  }
  if (normalized.startsWith("error:")) return `Analysis failed: ${reason || "API error"}`
  if (normalized.startsWith("[analysis failed")) return `Analysis failed: ${reason || "API error"}`
  if (normalized.startsWith("[analysis skipped")) return `Analysis skipped: ${reason || "not available"}`
  return null
}

export const analysisWarningFromPayload = (data: ProcessingResultPayload): string | undefined => {
  for (const item of extractProcessingItems(data)) {
    const status = normalizeStatusLabel(item.analysis_status)
    const detail = item.analysis_error || item.warnings?.find((warning) => analysisWarningFromText(warning))
    if (status === "skipped") {
      return analysisWarningFromText(detail) || `Analysis skipped: ${String(detail || "not available").trim()}`
    }
    if (status === "failed") {
      return analysisWarningFromText(detail) || `Analysis failed: ${String(detail || "API error").trim()}`
    }
    const analysisWarning =
      analysisWarningFromText(item.analysis) ||
      analysisWarningFromText(item.analysis_content)
    if (analysisWarning) return analysisWarning
  }
  return undefined
}

export const normalizeResultItem = (
  item: Partial<ResultItem> | null | undefined
): ResultItem | null => {
  if (!item) return null
  if (item.id === undefined || item.id === null) return null
  const id = String(item.id).trim()
  if (!id) return null
  return {
    id,
    status: normalizeResultStatus(item.status),
    url: item.url,
    fileName: item.fileName,
    type: String(item.type || "item"),
    data: item.data as ProcessingResultPayload,
    error: item.error,
    warning: item.warning || analysisWarningFromPayload(item.data as ProcessingResultPayload)
  }
}

const cloneObject = <T extends Record<string, any>>(value: T): T | null => {
  try { return structuredClone(value) } catch {
    try { return JSON.parse(JSON.stringify(value)) } catch {
      console.warn("[cloneObject] Failed to clone object, returning null", value)
      return null
    }
  }
}

const RESULT_FILTERS = {
  ALL: "all",
  SUCCESS: "success",
  ERROR: "error"
} as const

type ResultsFilter = (typeof RESULT_FILTERS)[keyof typeof RESULT_FILTERS]

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UseIngestResultsDeps {
  open: boolean
  running: boolean
  messageApi: MessageInstance
  qi: (key: string, defaultValue: string, options?: Record<string, any>) => string
  t: (key: string, opts?: Record<string, any>) => string
  onClose: () => void
  /** From options hook */
  reviewBeforeStorage: boolean
  storeRemote: boolean
  processOnly: boolean
  common: { perform_analysis: boolean; perform_chunking: boolean; overwrite_existing: boolean }
  advancedValues: Record<string, any>
  rows: Entry[]
  /** From queue hook */
  formatBytes: (bytes?: number) => string
  plannedRunContextRef: React.MutableRefObject<PlannedRunContext | null>
  /** Queue mutation setters needed for retry operations */
  setRows: React.Dispatch<React.SetStateAction<Entry[]>>
  setQueuedFiles: (v: any) => void
  setLocalFiles: React.Dispatch<React.SetStateAction<File[]>>
  buildRowEntry: (url?: string, type?: Entry["type"]) => Entry
  createDefaultsSnapshot: () => TypeDefaults | undefined
  /** Options setters needed for proceedWithoutReview */
  setReviewBeforeStorage: (v: boolean) => void
  setStoreRemote: (v: boolean) => void
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useIngestResults(deps: UseIngestResultsDeps) {
  const {
    open,
    running: runningProp,
    messageApi,
    qi,
    t,
    onClose,
    reviewBeforeStorage,
    storeRemote,
    processOnly,
    common,
    advancedValues,
    rows,
    formatBytes,
    plannedRunContextRef,
    setRows: setRowsFn,
    setQueuedFiles: setQueuedFilesFn,
    setLocalFiles: setLocalFilesFn,
    buildRowEntry,
    createDefaultsSnapshot,
    setReviewBeforeStorage: setReviewBeforeStorageFn,
    setStoreRemote: setStoreRemoteFn,
  } = deps

  const navigate = useNavigate()

  const {
    markFailure,
    clearFailure,
    recordRunSuccess,
    recordRunFailure,
    recordRunCancelled
  } = useQuickIngestStore((s) => ({
    markFailure: s.markFailure,
    clearFailure: s.clearFailure,
    recordRunSuccess: s.recordRunSuccess,
    recordRunFailure: s.recordRunFailure,
    recordRunCancelled: s.recordRunCancelled
  }))

  // ---- state ----
  const [results, setResults] = React.useState<ResultItem[]>([])
  const [totalPlanned, setTotalPlanned] = React.useState<number>(0)
  const [processedCount, setProcessedCount] = React.useState<number>(0)
  const [liveTotalCount, setLiveTotalCount] = React.useState<number>(0)
  const [runStartedAt, setRunStartedAt] = React.useState<number | null>(null)
  const [lastRunError, setLastRunError] = React.useState<string | null>(null)
  const [lastRunCancelled, setLastRunCancelled] = React.useState<boolean>(false)
  const [draftCreationError, setDraftCreationError] = React.useState<string | null>(null)
  const [draftCreationRetrying, setDraftCreationRetrying] = React.useState(false)
  const [reviewNavigationError, setReviewNavigationError] = React.useState<string | null>(null)
  const [reviewBatchId, setReviewBatchId] = React.useState<string | null>(null)
  const [progressTick, setProgressTick] = React.useState<number>(0)
  const [lastRunProcessOnly, setLastRunProcessOnly] = React.useState(processOnly)
  const [resultsFilter, setResultsFilter] = React.useState<ResultsFilter>(RESULT_FILTERS.ALL)

  const lastFileLookupRef = React.useRef<Map<string, File> | null>(null)
  const lastFileIdByInstanceIdRef = React.useRef<Map<string, string> | null>(null)
  const pendingStoreWithoutReviewRef = React.useRef(false)
  const unmountedRef = React.useRef(false)

  React.useEffect(() => {
    unmountedRef.current = false
    return () => { unmountedRef.current = true }
  }, [])

  // Progress tick timer
  React.useEffect(() => {
    if (!runningProp) return
    const id = window.setInterval(() => setProgressTick((t) => t + 1), 1000)
    return () => window.clearInterval(id)
  }, [runningProp])

  // Clean up on modal close
  React.useEffect(() => {
    if (!open) {
      setReviewBatchId(null)
      setDraftCreationError(null)
      setDraftCreationRetrying(false)
      setReviewNavigationError(null)
      lastFileLookupRef.current = null
      lastFileIdByInstanceIdRef.current = null
      pendingStoreWithoutReviewRef.current = false
    }
  }, [open])

  // ---- cancelled message check ----
  const isCancelledMessage = React.useCallback((value: unknown): boolean => {
    const text = String(value || "").toLowerCase()
    return text.includes("cancelled") || text.includes("canceled") || text.includes("abort")
  }, [])

  // ---- outcome derivation ----
  const deriveResultOutcome = React.useCallback(
    (item: ResultItem): ResultOutcome => {
      if (item.outcome === "cancelled") return "cancelled"
      if (item.status === "error" && isCancelledMessage(item.error)) return "cancelled"
      if (item.status === "error") return "failed"
      const statuses = getProcessingStatusLabels(item.data)
      const isSkipped = statuses.length > 0 && statuses.every((status) => isSkippedStatus(status))
      if (isSkipped) return "skipped"
      return lastRunProcessOnly ? "processed" : "ingested"
    },
    [isCancelledMessage, lastRunProcessOnly]
  )

  const summarizeResultOutcomes = React.useCallback(
    (items: ResultItem[]) => {
      let successCount = 0
      let failCount = 0
      let cancelledCount = 0
      for (const item of items) {
        if (item.status === "ok") { successCount += 1; continue }
        const outcome = deriveResultOutcome(item)
        if (outcome === "cancelled") cancelledCount += 1
        else failCount += 1
      }
      return { successCount, failCount, cancelledCount }
    },
    [deriveResultOutcome]
  )

  // ---- append missing results ----
  const appendMissingResultsFromPlan = React.useCallback(
    (
      existingResults: ResultItem[],
      errorMessage: string,
      options?: { status?: ResultItem["status"]; outcome?: ResultOutcome }
    ): ResultItem[] => {
      const planned = plannedRunContextRef.current
      if (!planned) return existingResults
      const message = String(errorMessage || "").trim() || qi("statusFailed", "Failed")
      const status = options?.status || "error"
      const outcome = options?.outcome
      const byId = new Map<string, ResultItem>()

      for (const existing of existingResults) {
        const normalized = normalizeResultItem(existing)
        if (normalized) byId.set(normalized.id, normalized)
      }

      for (const row of planned.plannedUrlEntries) {
        if (byId.has(row.id)) continue
        byId.set(row.id, { id: row.id, status, outcome, url: row.url, type: row.type, error: message })
      }
      for (const entry of planned.plannedFiles) {
        const fileId = planned.fileIdByInstanceId.get(getFileInstanceId(entry.file))
        if (!fileId || byId.has(fileId)) continue
        byId.set(fileId, { id: fileId, status, outcome, fileName: entry.file.name, type: entry.type, error: message })
      }
      return Array.from(byId.values())
    },
    [qi, plannedRunContextRef]
  )

  // ---- result lookup ----
  const resultById = React.useMemo(() => {
    const map = new Map<string, ResultItem>()
    for (const r of results) map.set(r.id, r)
    return map
  }, [results])

  const getResultForFile = React.useCallback(
    (file: File) => {
      const fileIdByInstanceId = lastFileIdByInstanceIdRef.current
      if (fileIdByInstanceId) {
        const id = fileIdByInstanceId.get(getFileInstanceId(file))
        if (id) return resultById.get(id) || null
      }
      const fallbackMatches = results.filter((r) => r.fileName === file.name)
      return fallbackMatches.length === 1 ? fallbackMatches[0] : null
    },
    [resultById, results]
  )

  // ---- progress ----
  const doneCount = processedCount || results.length || 0
  const totalCount = liveTotalCount || totalPlanned || 0

  const progressMeta = React.useMemo(() => {
    const total = liveTotalCount || totalPlanned || 0
    const done = processedCount || results.length || 0
    const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0
    const elapsedMs = runStartedAt ? Date.now() - runStartedAt : 0
    const elapsedLabel =
      elapsedMs > 0
        ? `${Math.floor(elapsedMs / 60000)}:${String(Math.floor((elapsedMs % 60000) / 1000)).padStart(2, '0')}`
        : null
    const state: "running" | "failed" | "complete" | "cancelled" | "ready" =
      runningProp
        ? "running"
        : lastRunCancelled
          ? "cancelled"
          : lastRunError
            ? "failed"
            : total > 0 && done >= total
              ? "complete"
              : "ready"
    return { total, done, pct, elapsedLabel, state, error: lastRunError }
  }, [lastRunCancelled, lastRunError, liveTotalCount, processedCount, progressTick, results.length, runStartedAt, runningProp, totalPlanned])

  // ---- results with outcome ----
  const resultsWithOutcome = React.useMemo(() => {
    if (!results || results.length === 0) return results
    return results.map((item) => ({ ...item, outcome: deriveResultOutcome(item) }))
  }, [deriveResultOutcome, results])

  const resultSummary = React.useMemo(() => {
    if (!results || results.length === 0) return null
    const { successCount, failCount, cancelledCount } = summarizeResultOutcomes(results)
    return { successCount, failCount, cancelledCount }
  }, [results, summarizeResultOutcomes])

  const visibleResults = React.useMemo(() => {
    let items = resultsWithOutcome || []
    if (resultsFilter === RESULT_FILTERS.SUCCESS) {
      items = items.filter((r) => r.status === "ok")
    } else if (resultsFilter === RESULT_FILTERS.ERROR) {
      items = items.filter((r) => r.status === "error")
    }
    const ranked = [...items].sort((a, b) => {
      const rank = (s: ResultItem["status"]) => s === "error" ? 0 : s === "ok" ? 1 : 2
      return rank(a.status) - rank(b.status)
    })
    return ranked
  }, [resultsFilter, resultsWithOutcome])

  const hasReviewableResults = React.useMemo(
    () => results.some((r) => r.status === "ok"),
    [results]
  )

  const firstResultWithMedia = React.useMemo(
    () => results.find((r) => r.status === "ok" && mediaIdFromPayload(r.data)),
    [results]
  )

  // ---- draft creation ----
  const createDraftBatchMetadata = React.useCallback(
    (okResults: ResultItem[]) => {
      const now = Date.now()
      const batchId = crypto.randomUUID()
      const batch: DraftBatch = {
        id: batchId,
        source: "quick_ingest",
        sourceDetails: { total: okResults.length },
        createdAt: now,
        updatedAt: now
      }
      const processingOptions: ProcessingOptions = {
        perform_analysis: Boolean(common.perform_analysis),
        perform_chunking: resolvePerformChunking(common.perform_chunking),
        overwrite_existing: Boolean(common.overwrite_existing),
        advancedValues: { ...(advancedValues || {}) }
      }
      const expiresAt = now + 30 * 24 * 60 * 60 * 1000
      const rowMap = new Map(rows.map((row) => [row.id, row]))
      return { now, batchId, batch, processingOptions, expiresAt, rowMap }
    },
    [advancedValues, common, rows]
  )

  const extractMetadataForDraft = React.useCallback((processed: ProcessingItem) => {
    const metadata = processed?.metadata && typeof processed.metadata === "object" ? processed.metadata : {}
    const metadataCopy = cloneObject(metadata)
    const originalMetadata = cloneObject(metadata)
    if (!metadataCopy || !originalMetadata) {
      console.warn("[createDraftsFromResults] Unable to clone metadata, using empty metadata", metadata)
    }
    const keywords = normalizeKeywords(processed?.keywords || metadata?.keywords)
    return { metadataCopy: metadataCopy ?? {}, originalMetadata: originalMetadata ?? {}, keywords }
  }, [])

  const storeDraftAssetIfPresent = React.useCallback(
    async ({ draftId, localFile, item, sourceRow, processed }: {
      draftId: string; localFile?: File; item: ResultItem; sourceRow?: Entry; processed: ProcessingItem
    }) => {
      let sourceAssetId: string | undefined
      let source: ContentDraft["source"] = {
        kind: "url",
        url: sourceRow?.url || item.url || processed?.url || processed?.input_ref
      }
      let skippedAssetsDelta = 0
      if (localFile) {
        const stored = await storeDraftAsset(draftId, localFile)
        const fileSource: ContentDraft["source"] = {
          kind: "file",
          fileName: localFile.name,
          mimeType: localFile.type,
          sizeBytes: localFile.size,
          lastModified: localFile.lastModified
        }
        if (stored.asset) sourceAssetId = stored.asset.id
        else skippedAssetsDelta += 1
        source = fileSource
      } else if (item.fileName) {
        source = { kind: "file", fileName: item.fileName }
      }
      return { source, sourceAssetId, skippedAssetsDelta }
    },
    []
  )

  const buildDraftFromProcessedItem = React.useCallback(
    async ({ item, processed, sourceRow, localFile, batchId, now, expiresAt, processingOptions }: {
      item: ResultItem; processed: ProcessingItem; sourceRow?: Entry; localFile?: File;
      batchId: string; now: number; expiresAt: number; processingOptions: ProcessingOptions
    }): Promise<{ draftId: string; skippedAssetsDelta: number } | null> => {
      const statusLabel = String(processed?.status || "").toLowerCase()
      if (statusLabel === "error" || statusLabel === "failed") return null

      const draftId = crypto.randomUUID()
      const content = resolveContent(processed)
      const { metadataCopy, originalMetadata, keywords } = extractMetadataForDraft(processed)
      const mediaTypeRaw = String(processed?.media_type || item.type || sourceRow?.type || "document").toLowerCase()
      const mediaType: ContentDraft["mediaType"] = coerceDraftMediaType(mediaTypeRaw)
      const sourceLabel = sourceRow?.url || item.url || localFile?.name || item.fileName || processed?.input_ref || "Untitled source"
      const title = resolveTitle(processed, sourceLabel)
      const contentFormat = inferContentFormat(content)
      const { sections, strategy } = detectSections(content, processed?.segments)
      const processingSnapshot = { ...processingOptions, advancedValues: { ...(processingOptions.advancedValues || {}) } }
      const analysis = resolveAnalysis(processed)
      const prompt = resolvePrompt(processed)

      const { source, sourceAssetId, skippedAssetsDelta } =
        await storeDraftAssetIfPresent({ draftId, localFile, item, sourceRow, processed })

      const draft: ContentDraft = {
        id: draftId, batchId, source, sourceAssetId, mediaType, title,
        originalTitle: title, content, originalContent: content,
        contentFormat, originalContentFormat: contentFormat,
        metadata: metadataCopy, originalMetadata, keywords,
        sections: sections.length > 0 ? sections : undefined,
        excludedSectionIds: [],
        sectionStrategy: strategy || undefined,
        revisions: [],
        processingOptions: processingSnapshot,
        status: "pending",
        createdAt: now, updatedAt: now, expiresAt,
        analysis, prompt, originalAnalysis: analysis, originalPrompt: prompt
      }
      await upsertContentDraft(draft)
      return { draftId, skippedAssetsDelta }
    },
    [extractMetadataForDraft, storeDraftAssetIfPresent]
  )

  const createDraftsFromResults = React.useCallback(
    async (out: ResultItem[], fileLookup: Map<string, File>): Promise<{
      batchId: string; draftIds: string[]; skippedAssets: number
    } | null> => {
      const okResults = out.filter((item) => item.status === "ok")
      if (okResults.length === 0) return null
      const { now, batchId, batch, processingOptions, expiresAt, rowMap } = createDraftBatchMetadata(okResults)
      const draftIds: string[] = []
      let skippedAssets = 0
      await upsertDraftBatch(batch)
      const draftPromises = okResults.map(async (item) => {
        const sourceRow = rowMap.get(item.id)
        const localFile = fileLookup.get(item.id)
        const processingItems = extractProcessingItems(item.data)
        if (processingItems.length === 0) return []
        const itemDrafts = await Promise.all(
          processingItems.map(async (processed) =>
            buildDraftFromProcessedItem({ item, processed, sourceRow, localFile, batchId, now, expiresAt, processingOptions })
          )
        )
        return itemDrafts.filter((draft): draft is { draftId: string; skippedAssetsDelta: number } => Boolean(draft))
      })
      const allDrafts = (await Promise.all(draftPromises)).flat()
      for (const draft of allDrafts) {
        skippedAssets += draft.skippedAssetsDelta
        draftIds.push(draft.draftId)
      }
      return { batchId, draftIds, skippedAssets }
    },
    [buildDraftFromProcessedItem, createDraftBatchMetadata]
  )

  // ---- navigation helpers ----
  const openOptionsRoute = React.useCallback((hash: OptionsHash) => {
    try {
      const path = window.location.pathname || ""
      if (path.includes("options.html")) { window.location.hash = hash; return }
      try {
        const url = browser.runtime.getURL(`/options.html${hash}`)
        if (browser?.tabs?.create) browser.tabs.create({ url })
        else window.open(url, "_blank")
        return
      } catch {}
      window.open(`/options.html${hash}`, "_blank")
    } catch {}
  }, [])

  const openHealthDiagnostics = React.useCallback(() => {
    openOptionsRoute("#/settings/health")
  }, [openOptionsRoute])

  const openModelSettings = React.useCallback(() => {
    openOptionsRoute("#/settings/model")
  }, [openOptionsRoute])

  const openContentReview = React.useCallback(
    async (batchId: string): Promise<boolean> => {
      const hash = `#/content-review?batch=${batchId}`
      const isOptionsContext = window.location.pathname.includes("options.html")
      if (isOptionsContext) {
        try { navigate(`/content-review?batch=${batchId}`); return true } catch {
          messageApi.error(qi("reviewNavigationFailed", "Couldn't open Content Review. Please try again."))
          return false
        }
      }
      try {
        const url = browser.runtime.getURL(`/options.html${hash}`)
        await browser.tabs.create({ url })
        return true
      } catch {
        try {
          const fallbackUrl = browser.runtime.getURL(`/options.html${hash}`)
          const win = window.open(fallbackUrl, "_blank")
          if (!win) { messageApi.error(qi("reviewNavigationFailed", "Couldn't open Content Review. Please try again.")); return false }
          return Boolean(win)
        } catch {
          messageApi.error(qi("reviewNavigationFailed", "Couldn't open Content Review. Please try again."))
          return false
        }
      }
    },
    [messageApi, navigate, qi]
  )

  const tryOpenContentReview = React.useCallback(
    async (batchId: string, options?: { closeOnSuccess?: boolean; closeDelayMs?: number }) => {
      setReviewNavigationError(null)
      const ok = await openContentReview(batchId)
      if (!ok) {
        const msg = qi("reviewNavigationFailed", "Couldn't open Content Review. Please try again.")
        setReviewNavigationError(msg)
        return false
      }
      if (options?.closeOnSuccess) {
        const delayMs = typeof options.closeDelayMs === "number" ? options.closeDelayMs : 250
        if (delayMs > 0) window.setTimeout(() => { onClose() }, delayMs)
        else onClose()
      }
      return true
    },
    [onClose, openContentReview, qi]
  )

  const handleReviewBatchReady = React.useCallback(
    async (batch: { batchId: string; skippedAssets: number }) => {
      if (!batch?.batchId) return
      if (batch.skippedAssets > 0) {
        const reviewStorageCapDefault =
          batch.skippedAssets === 1
            ? "{{count}} file exceeds the local draft cap ({{cap}}). Attach sources before committing audio/video."
            : "{{count}} files exceed the local draft cap ({{cap}}). Attach sources before committing audio/video."
        messageApi.warning(qi("reviewStorageCapWarning", reviewStorageCapDefault, { count: batch.skippedAssets, cap: formatBytes(DRAFT_STORAGE_CAP_BYTES) }))
      } else {
        messageApi.success(qi("reviewDraftsCreated", "Drafts ready for review."))
      }
      await tryOpenContentReview(batch.batchId, { closeOnSuccess: true, closeDelayMs: 250 })
    },
    [formatBytes, messageApi, qi, tryOpenContentReview]
  )

  // ---- download helpers ----
  const downloadBlobAsJson = React.useCallback((data: unknown, filename: string) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
    const a = document.createElement("a")
    a.href = URL.createObjectURL(blob)
    a.download = filename
    a.click()
    URL.revokeObjectURL(a.href)
  }, [])

  const downloadJson = React.useCallback((item: ResultItem) => {
    downloadBlobAsJson(item.data ?? {}, "processed.json")
  }, [downloadBlobAsJson])

  const downloadResultsJson = React.useCallback(() => {
    if (!results.length) return
    downloadBlobAsJson(results, "quick-ingest-results.json")
  }, [downloadBlobAsJson, results])

  // ---- media viewer / discuss ----
  const openInMediaViewer = React.useCallback((item: ResultItem) => {
    try {
      const id = mediaIdFromPayload(item.data)
      if (id == null) return
      void setSetting(LAST_MEDIA_ID_SETTING, String(id))
      openOptionsRoute("#/media-multi")
    } catch {}
  }, [openOptionsRoute])

  const discussInChat = React.useCallback((item: ResultItem) => {
    try {
      const id = mediaIdFromPayload(item.data)
      if (id == null) return
      let sourceUrl: string | undefined
      if (item.data && typeof item.data === "object" && !Array.isArray(item.data)) {
        const payload = item.data as Record<string, unknown>
        sourceUrl = typeof payload.url === "string" ? payload.url : typeof payload.source_url === "string" ? payload.source_url : undefined
      }
      const payload = { mediaId: String(id), url: item.url || sourceUrl, mode: "rag_media" as const }
      void setSetting(DISCUSS_MEDIA_PROMPT_SETTING, payload)
      try { window.dispatchEvent(new CustomEvent("tldw:discuss-media", { detail: payload })) } catch {}
      openOptionsRoute("#/")
    } catch {}
  }, [openOptionsRoute])

  // ---- retry / requeue ----
  const retryFailedUrls = React.useCallback(
    () => {
      const failedByUrl = results.filter((r) => r.status === "error" && r.url)
      const rowsById = new Map(rows.map((row) => [row.id, row]))
      const rowsByUrl = new Map(rows.map((row) => [row.url.trim(), row]))
      const failedUrls = failedByUrl.map((r) => {
        const key = (r.url || "").trim()
        const existing = (r.id && rowsById.get(r.id)) || (key ? rowsByUrl.get(key) : undefined)
        if (existing) return { ...existing, id: crypto.randomUUID(), defaults: existing.defaults || createDefaultsSnapshot() }
        return buildRowEntry(r.url || "", "auto")
      })
      if (failedUrls.length === 0) {
        messageApi.info(qi("noFailedUrlToRetry", "No failed URL items to retry."))
        return
      }
      setRowsFn(failedUrls)
      setQueuedFilesFn([])
      setLocalFilesFn([])
      setResults([])
      setProcessedCount(0)
      setTotalPlanned(failedUrls.length)
      setLiveTotalCount(failedUrls.length)
      setRunStartedAt(null)
      messageApi.info(qi("queuedFailedUrls", "Queued {{count}} failed URL(s) to retry.", { count: failedUrls.length }))
    },
    [buildRowEntry, createDefaultsSnapshot, messageApi, qi, results, rows, setRowsFn, setQueuedFilesFn, setLocalFilesFn]
  )

  const exportFailedList = React.useCallback(() => {
    const failedItems = results.filter((r) => r.status === "error")
    const lines = failedItems.map((item) => {
      if (item.url) return `URL: ${item.url}`
      if (item.fileName) return `File: ${item.fileName}`
      return `Unknown: ${item.id}`
    })
    if (lines.length === 0) {
      messageApi.info(qi("noFailedToExport", "No failed items to export."))
      return
    }
    const text = lines.join('\n')
    navigator.clipboard.writeText(text).then(() => {
      messageApi.success(qi("exportedToClipboard", "Copied {{count}} failed item(s) to clipboard.", { count: lines.length }))
    }).catch(() => {
      const blob = new Blob([text], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'failed-items.txt'
      a.click()
      URL.revokeObjectURL(url)
      messageApi.success(qi("exportedToFile", "Downloaded {{count}} failed item(s) as file.", { count: lines.length }))
    })
  }, [messageApi, qi, results])

  // ---- draft retry / proceed ----
  const retryDraftCreation = React.useCallback(async () => {
    if (draftCreationRetrying || runningProp || !hasReviewableResults) return
    const fileLookup = lastFileLookupRef.current || new Map<string, File>()
    setDraftCreationRetrying(true)
    setDraftCreationError(null)
    try {
      const created = await createDraftsFromResults(results, fileLookup)
      if (unmountedRef.current) return
      if (!created?.batchId) {
        const msg = qi("reviewDraftsFailedFallback", "Failed to create review drafts.")
        messageApi.error(msg)
        setDraftCreationError(msg)
        return
      }
      setReviewBatchId(created.batchId)
      await handleReviewBatchReady(created)
    } catch (err) {
      console.error("[quickIngest] Failed to retry review draft creation", err)
      const msg = qi("reviewDraftsFailedFallback", "Failed to create review drafts.")
      messageApi.error(msg)
      if (!unmountedRef.current) setDraftCreationError(msg)
    } finally {
      if (!unmountedRef.current) setDraftCreationRetrying(false)
    }
  }, [createDraftsFromResults, draftCreationRetrying, handleReviewBatchReady, hasReviewableResults, messageApi, qi, results, runningProp])

  const proceedWithoutReview = React.useCallback(() => {
    if (runningProp) return
    pendingStoreWithoutReviewRef.current = true
    setDraftCreationError(null)
    setReviewBeforeStorageFn(false)
    setStoreRemoteFn(true)
  }, [runningProp, setReviewBeforeStorageFn, setStoreRemoteFn])

  return {
    // state
    results, setResults,
    totalPlanned, setTotalPlanned,
    processedCount, setProcessedCount,
    liveTotalCount, setLiveTotalCount,
    runStartedAt, setRunStartedAt,
    lastRunError, setLastRunError,
    lastRunCancelled, setLastRunCancelled,
    draftCreationError, setDraftCreationError,
    draftCreationRetrying,
    reviewNavigationError, setReviewNavigationError,
    reviewBatchId, setReviewBatchId,
    lastRunProcessOnly, setLastRunProcessOnly,
    resultsFilter, setResultsFilter,
    // refs
    lastFileLookupRef,
    lastFileIdByInstanceIdRef,
    pendingStoreWithoutReviewRef,
    unmountedRef,
    // derived
    doneCount,
    totalCount,
    progressMeta,
    resultsWithOutcome,
    resultSummary,
    visibleResults,
    hasReviewableResults,
    firstResultWithMedia,
    resultById,
    RESULT_FILTERS,
    // callbacks
    isCancelledMessage,
    deriveResultOutcome,
    summarizeResultOutcomes,
    appendMissingResultsFromPlan,
    getResultForFile,
    createDraftsFromResults,
    openOptionsRoute,
    openHealthDiagnostics,
    openModelSettings,
    openContentReview,
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
    // store actions
    markFailure,
    clearFailure,
    recordRunSuccess,
    recordRunFailure,
    recordRunCancelled,
  }
}

export { extractProcessingItems, getProcessingStatusLabels, RESULT_FILTERS }
export type { PlannedRunContext, ResultsFilter }
