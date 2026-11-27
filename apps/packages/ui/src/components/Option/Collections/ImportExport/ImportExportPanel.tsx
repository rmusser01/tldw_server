import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Button,
  Card,
  Checkbox,
  Empty,
  Input,
  List,
  message,
  Radio,
  Spin,
  Steps,
  Upload
} from "antd"
import {
  Upload as UploadIcon,
  Download,
  FileText,
  Bookmark,
  ArrowRight,
  AlertCircle
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives/Alert"
import { StatePanel } from "@/components/ui/state"
import { useCollectionsStore } from "@/store/collections"
import { useTldwApiClient } from "@/hooks/useTldwApiClient"
import { useSelectionKeyboard } from "@/hooks/useSelectionKeyboard"
import type {
  ImportSource,
  ExportFormat as CollectionExportFormat,
  ReadingItem,
  ReadingItemSummary,
  ReadingImportJobState,
  ReadingImportJobStatus
} from "@/types/collections"

const IMPORT_SOURCES: {
  value: ImportSource
  labelKey: string
  icon: React.ReactNode
  descriptionKey: string
}[] = [
  {
    value: "auto",
    labelKey: "collections:import.sources.auto.label",
    icon: <FileText className="h-6 w-6" />,
    descriptionKey: "collections:import.sources.auto.description"
  },
  {
    value: "pocket",
    labelKey: "collections:import.sources.pocket.label",
    icon: <Bookmark className="h-6 w-6" />,
    descriptionKey: "collections:import.sources.pocket.description"
  },
  {
    value: "instapaper",
    labelKey: "collections:import.sources.instapaper.label",
    icon: <FileText className="h-6 w-6" />,
    descriptionKey: "collections:import.sources.instapaper.description"
  }
]

const EXPORT_FORMATS: { value: CollectionExportFormat; labelKey: string }[] = [
  { value: "jsonl", labelKey: "collections:export.formats.jsonl" },
  { value: "zip", labelKey: "collections:export.formats.zip" }
]

const MAX_IMPORT_FILE_BYTES = 10 * 1024 * 1024
const MAX_IMPORT_ERRORS_TO_SHOW = 5
const IMPORT_POLL_INTERVAL_MS = 1500
const IMPORT_TERMINAL_STATES = new Set<ReadingImportJobState>([
  "completed",
  "failed",
  "cancelled",
  "quarantined"
])
const IMPORT_SOURCE_EXTENSIONS: Record<ImportSource, string[]> = {
  auto: [".json", ".csv"],
  pocket: [".json"],
  instapaper: [".csv"]
}

export const ImportExportPanel: React.FC = () => {
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <ImportSection />
      <ExportSection />
    </div>
  )
}

// Import Section
const ImportSection: React.FC = () => {
  const { t } = useTranslation(["collections", "common"])
  const api = useTldwApiClient()

  const importSource = useCollectionsStore((s) => s.importSource)
  const importInProgress = useCollectionsStore((s) => s.importInProgress)
  const importError = useCollectionsStore((s) => s.importError)
  const importResult = useCollectionsStore((s) => s.importResult)
  const importWizardStep = useCollectionsStore((s) => s.importWizardStep)

  const setImportSource = useCollectionsStore((s) => s.setImportSource)
  const setImportFile = useCollectionsStore((s) => s.setImportFile)
  const setImportInProgress = useCollectionsStore((s) => s.setImportInProgress)
  const setImportError = useCollectionsStore((s) => s.setImportError)
  const setImportResult = useCollectionsStore((s) => s.setImportResult)
  const setImportWizardStep = useCollectionsStore((s) => s.setImportWizardStep)
  const resetImportWizard = useCollectionsStore((s) => s.resetImportWizard)
  const [mergeTags, setMergeTags] = useState(true)
  const [importJob, setImportJob] = useState<ReadingImportJobStatus | null>(null)
  const importPollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearImportPoll = useCallback(() => {
    if (importPollTimerRef.current) {
      clearTimeout(importPollTimerRef.current)
      importPollTimerRef.current = null
    }
  }, [])

  useEffect(() => {
    return () => {
      clearImportPoll()
    }
  }, [clearImportPoll])

  const describeExpectedFormats = useCallback(
    (source: ImportSource | null) => {
      if (!source || source === "auto") {
        return t("collections:import.formats", "Supports Pocket JSON and Instapaper CSV")
      }
      if (source === "pocket") {
        return t("collections:import.formatsPocket", "Pocket imports require JSON exports")
      }
      return t("collections:import.formatsInstapaper", "Instapaper imports require CSV exports")
    },
    [t]
  )

  const validateImportFile = useCallback(
    (file: File, source: ImportSource) => {
      const filename = file.name.toLowerCase()
      const expectedExtensions = IMPORT_SOURCE_EXTENSIONS[source]
      const extensionAllowed = expectedExtensions.some((ext) => filename.endsWith(ext))
      if (!extensionAllowed) {
        return t(
          "collections:import.invalidType",
          "File type does not match selected source. Expected: {{expected}}",
          { expected: expectedExtensions.join(", ") }
        )
      }
      if (file.size > MAX_IMPORT_FILE_BYTES) {
        return t("collections:import.fileTooLarge", "File exceeds the {{maxMb}} MB limit.", {
          maxMb: Math.floor(MAX_IMPORT_FILE_BYTES / (1024 * 1024))
        })
      }
      return null
    },
    [t]
  )

  const mapImportError = useCallback(
    (error: unknown) => {
      const maybeError = error as Error & { status?: number; details?: unknown }
      if (typeof maybeError?.status === "number") {
        if (maybeError.status === 400) {
          return t(
            "collections:import.invalidPayload",
            "Import failed due to invalid file content or source format."
          )
        }
        if (maybeError.status === 413) {
          return t("collections:import.fileTooLarge", "File exceeds the {{maxMb}} MB limit.", {
            maxMb: Math.floor(MAX_IMPORT_FILE_BYTES / (1024 * 1024))
          })
        }
      }
      return maybeError?.message || t("collections:import.failed", "Import failed")
    },
    [t]
  )

  const importStatusLabel = useCallback(
    (status: ReadingImportJobState) =>
      t(
        `collections:import.jobStatus.${status}`,
        status.charAt(0).toUpperCase() + status.slice(1)
      ),
    [t]
  )

  const finalizeImport = useCallback(
    (job: ReadingImportJobStatus) => {
      clearImportPoll()
      setImportInProgress(false)
      if (job.status === "completed" && job.result) {
        setImportResult(job.result)
        setImportError(null)
        setImportWizardStep("result")
        message.success(
          t("collections:import.success", "Imported {{count}} items", {
            count: job.result.imported
          })
        )
        return
      }
      if (job.result) {
        setImportResult(job.result)
      }
      const err =
        job.error_message ||
        t("collections:import.jobFailed", "Import job {{status}}", {
          status: importStatusLabel(job.status)
        })
      setImportError(err)
      setImportWizardStep("result")
    },
    [
      clearImportPoll,
      importStatusLabel,
      setImportError,
      setImportInProgress,
      setImportResult,
      setImportWizardStep,
      t
    ]
  )

  const pollImportJob = useCallback(
    async (jobId: number) => {
      try {
        const status = await api.getReadingImportJob(jobId)
        setImportJob(status)
        if (IMPORT_TERMINAL_STATES.has(status.status)) {
          finalizeImport(status)
          return
        }
        importPollTimerRef.current = setTimeout(() => {
          void pollImportJob(jobId)
        }, IMPORT_POLL_INTERVAL_MS)
      } catch (error: unknown) {
        clearImportPoll()
        setImportInProgress(false)
        const msg = mapImportError(error)
        setImportError(msg)
        message.error(msg)
      }
    },
    [api, clearImportPoll, finalizeImport, mapImportError, setImportError, setImportInProgress]
  )

  const handleSourceSelect = useCallback((source: ImportSource) => {
    clearImportPoll()
    setImportSource(source)
    setImportFile(null)
    setImportError(null)
    setImportResult(null)
    setImportInProgress(false)
    setImportWizardStep("upload")
    setImportJob(null)
    setMergeTags(true)
  }, [
    clearImportPoll,
    setImportError,
    setImportFile,
    setImportInProgress,
    setImportResult,
    setImportSource,
    setImportWizardStep
  ])

  const handleFileUpload = useCallback(async (file: File) => {
    setImportFile(file)
    if (!importSource) {
      message.error(t("collections:import.sourceRequired", "Select a source first"))
      return
    }
    const validationError = validateImportFile(file, importSource)
    if (validationError) {
      setImportError(validationError)
      message.error(validationError)
      return
    }

    clearImportPoll()
    setImportInProgress(true)
    setImportError(null)
    setImportResult(null)
    setImportJob(null)

    try {
      const created = await api.importReadingList({
        source: importSource,
        file,
        merge_tags: mergeTags
      })
      if (typeof created?.job_id !== "number") {
        setImportInProgress(false)
        setImportError(
          t(
            "collections:import.missingJobId",
            "Import request was accepted but no job id was returned."
          )
        )
        return
      }
      const initialJob: ReadingImportJobStatus = {
        job_id: created.job_id,
        job_uuid: created.job_uuid,
        status: created.status
      }
      setImportJob(initialJob)
      if (IMPORT_TERMINAL_STATES.has(created.status)) {
        void pollImportJob(created.job_id)
        return
      }
      importPollTimerRef.current = setTimeout(() => {
        void pollImportJob(created.job_id)
      }, 250)
    } catch (error: unknown) {
      clearImportPoll()
      setImportInProgress(false)
      const msg = mapImportError(error)
      setImportError(msg)
      message.error(msg)
    }
  }, [
    api,
    clearImportPoll,
    importSource,
    mapImportError,
    mergeTags,
    pollImportJob,
    setImportFile,
    setImportInProgress,
    setImportError,
    setImportJob,
    setImportResult,
    t,
    validateImportFile
  ])

  const stepItems = [
    { title: t("collections:import.steps.source", "Source") },
    { title: t("collections:import.steps.upload", "Upload") },
    { title: t("collections:import.steps.result", "Result") }
  ]

  const currentStep =
    importWizardStep === "source"
      ? 0
      : importWizardStep === "upload"
        ? 1
        : 2

  const importTitle = t("collections:import.title", "Import")
  const processingMessage = t("collections:import.processing", "Processing file...")
  const progressTitle = importJob
    ? t("collections:import.currentStatus", "Current status: {{status}}", {
        status: importStatusLabel(importJob.status)
      })
    : null
  const progressMessage = importJob?.progress_message || processingMessage
  const resultTitle = importResult
    ? t("collections:import.complete", "Import Complete")
    : t("collections:import.failedTitle", "Import Failed")
  const resultMessage = importResult
    ? t(
        "collections:import.resultSummary",
        "Imported: {{imported}}, Updated: {{updated}}, Skipped: {{skipped}}",
        {
          imported: importResult.imported,
          updated: importResult.updated,
          skipped: importResult.skipped
        }
      )
    : importError ||
      t("collections:import.failed", "The import job did not complete successfully.")
  const resultErrorCount = importResult?.errors.length ?? 0
  const resultErrors =
    resultErrorCount > 0
      ? t("collections:import.errorsSummary", "Import errors: {{count}}", { count: resultErrorCount })
      : ""
  const completedState = resultErrorCount > 0 ? "degraded" : "ready"
  const importAnnouncement =
    importWizardStep === "result"
      ? [resultTitle, resultMessage, resultErrors].filter(Boolean).join(". ")
      : importInProgress
        ? [progressTitle, progressMessage].filter(Boolean).join(". ")
        : ""

  return (
    <Card
      title={
        <span className="flex items-center gap-2">
          <UploadIcon className="h-5 w-5" />
          {importTitle}
        </span>
      }
    >
      <Steps current={currentStep} items={stepItems} size="small" className="mb-6" />
      <p role="status" aria-label={importTitle} aria-atomic="true" className="sr-only">
        {importAnnouncement}
      </p>

      {importWizardStep === "source" && (
        <div className="space-y-3">
          <p className="text-sm text-text-muted">
            {t("collections:import.selectSource", "Select import source:")}
          </p>
          {IMPORT_SOURCES.map((source) => (
            <button
              key={source.value}
              onClick={() => handleSourceSelect(source.value)}
              className="flex w-full items-center gap-4 rounded-lg border border-border p-4 text-left transition-colors hover:border-primary hover:bg-primary/10"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-surface">
                {source.icon}
              </div>
              <div>
                <div className="font-medium">{t(source.labelKey)}</div>
                <div className="text-sm text-text-muted">{t(source.descriptionKey)}</div>
              </div>
              <ArrowRight className="ml-auto h-5 w-5 text-text-subtle" />
            </button>
          ))}
        </div>
      )}

      {importWizardStep === "upload" && (
        <div className="space-y-4">
          <Alert
            aria-live="off"
            title={
              <span className="text-text">
                {t("collections:import.selectedSource", "Source: {{source}}", {
                  source: importSource
                    ? t(`collections:import.sources.${importSource}.label`, importSource)
                    : t("collections:import.sources.auto.label", "Auto-detect")
                })}
              </span>
            }
          >
            <span className="text-text">{describeExpectedFormats(importSource)}</span>
          </Alert>

          <Checkbox
            checked={mergeTags}
            disabled={importInProgress}
            onChange={(e) => setMergeTags(e.target.checked)}
          >
            {t("collections:import.mergeTags", "Merge imported tags with existing tags")}
          </Checkbox>

          <Upload.Dragger
            accept={
              importSource ? IMPORT_SOURCE_EXTENSIONS[importSource].join(",") : ".json,.csv"
            }
            maxCount={1}
            disabled={importInProgress}
            beforeUpload={(file) => {
              void handleFileUpload(file as File)
              return false
            }}
            showUploadList={false}
          >
            {importInProgress ? (
              <div className="py-8">
                <Spin size="large" />
                <p className="mt-4 text-text-muted">
                  {processingMessage}
                </p>
              </div>
            ) : (
              <div className="py-8">
                <UploadIcon className="mx-auto h-10 w-10 text-text-subtle" />
                <p className="mt-4 text-text-muted">
                  {t("collections:import.dropzone", "Click or drag file to upload")}
                </p>
                <p className="mt-2 text-sm text-text-subtle">
                  {describeExpectedFormats(importSource)}
                </p>
                <p className="mt-1 text-xs text-text-subtle">
                  {t("collections:import.maxSize", "Max file size: {{size}} MB", {
                    size: Math.floor(MAX_IMPORT_FILE_BYTES / (1024 * 1024))
                  })}
                </p>
              </div>
            )}
          </Upload.Dragger>

          {importInProgress && importJob && (
            <Alert
              aria-live="off"
              title={<span className="text-text">{progressTitle}</span>}
            >
              <span className="text-text">{progressMessage}</span>
            </Alert>
          )}

          {importError && (
            <div className="flex items-center gap-2 text-danger">
              <AlertCircle className="h-4 w-4" />
              <span className="text-sm">{importError}</span>
            </div>
          )}

          <Button onClick={resetImportWizard} disabled={importInProgress}>
            {t("common:back", "Back")}
          </Button>
        </div>
      )}

      {importWizardStep === "result" && (
        <div className="space-y-3 pt-4">
          <StatePanel
            state={importResult ? completedState : "error"}
            title={resultTitle}
            message={resultMessage}
            className="border-border"
            primaryAction={{
              label: importResult
                ? t("collections:import.importMore", "Import More")
                : t("collections:import.tryAgain", "Try Again"),
              onClick: resetImportWizard
            }}
          />
          {importResult && resultErrorCount > 0 && (
            <Alert
              variant="warning"
              role="status"
              aria-live="off"
              title={
                <span className="text-text">
                  {t("collections:import.topErrorsTitle", "Top import errors")}
                </span>
              }
            >
              <ul className="mb-0 list-disc pl-5 text-text">
                {importResult.errors.slice(0, MAX_IMPORT_ERRORS_TO_SHOW).map((err, index) => (
                  <li key={`${err}-${index}`}>{err}</li>
                ))}
                {resultErrorCount > MAX_IMPORT_ERRORS_TO_SHOW && (
                  <li>
                    {t("collections:import.moreErrors", "+{{count}} more errors", {
                      count: resultErrorCount - MAX_IMPORT_ERRORS_TO_SHOW
                    })}
                  </li>
                )}
              </ul>
            </Alert>
          )}
        </div>
      )}
    </Card>
  )
}

// Export Section
const ExportSection: React.FC = () => {
  const { t } = useTranslation(["collections", "common"])
  const api = useTldwApiClient()
  const MAX_EXPORT_PAGES = 200

  const storedItems = useCollectionsStore((s) => s.items)
  const exportFormat = useCollectionsStore((s) => s.exportFormat)
  const exportInProgress = useCollectionsStore((s) => s.exportInProgress)
  const itemsSearch = useCollectionsStore((s) => s.itemsSearch)
  const filterStatus = useCollectionsStore((s) => s.filterStatus)
  const filterTags = useCollectionsStore((s) => s.filterTags)
  const filterFavorite = useCollectionsStore((s) => s.filterFavorite)
  const filterDomain = useCollectionsStore((s) => s.filterDomain)
  const filterDateFrom = useCollectionsStore((s) => s.filterDateFrom)
  const filterDateTo = useCollectionsStore((s) => s.filterDateTo)

  const setExportFormat = useCollectionsStore((s) => s.setExportFormat)
  const setExportInProgress = useCollectionsStore((s) => s.setExportInProgress)

  const [exportItems, setExportItems] = useState<ReadingItemSummary[]>([])
  const [exportItemsLoading, setExportItemsLoading] = useState(false)
  const [exportItemsError, setExportItemsError] = useState<string | null>(null)
  const [exportSearch, setExportSearch] = useState("")
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [copying, setCopying] = useState(false)
  const [applyReadingFilters, setApplyReadingFilters] = useState(true)
  const [includeHighlights, setIncludeHighlights] = useState(false)
  const [includeNotes, setIncludeNotes] = useState(true)
  const [loadProgress, setLoadProgress] = useState<{ loaded: number; total: number | null }>({
    loaded: 0,
    total: null
  })

  const listFilters = useMemo(() => {
    if (!applyReadingFilters) {
      return {}
    }
    return {
      q: itemsSearch || undefined,
      status: filterStatus !== "all" ? filterStatus : undefined,
      tags: filterTags.length > 0 ? filterTags : undefined,
      favorite: filterFavorite ?? undefined,
      domain: filterDomain || undefined,
      date_from: filterDateFrom || undefined,
      date_to: filterDateTo || undefined
    }
  }, [
    applyReadingFilters,
    filterDateFrom,
    filterDateTo,
    filterDomain,
    filterFavorite,
    filterStatus,
    filterTags,
    itemsSearch
  ])

  const serverExportFilters = useMemo(() => {
    if (!applyReadingFilters) {
      return {}
    }
    return {
      q: itemsSearch || undefined,
      status: filterStatus !== "all" ? [filterStatus] : undefined,
      tags: filterTags.length > 0 ? filterTags : undefined,
      favorite: filterFavorite ?? undefined,
      domain: filterDomain || undefined
    }
  }, [applyReadingFilters, filterDomain, filterFavorite, filterStatus, filterTags, itemsSearch])

  const hasDateFilter = Boolean(filterDateFrom || filterDateTo)

  const exportFilenameHint = useMemo(() => {
    if (selectedIds.length > 0) {
      return "reading_export_selection.jsonl"
    }
    if (applyReadingFilters && hasDateFilter) {
      return "reading_export_filtered.jsonl"
    }
    return exportFormat === "zip"
      ? "reading_export_<timestamp>.zip"
      : "reading_export_<timestamp>.jsonl"
  }, [applyReadingFilters, exportFormat, hasDateFilter, selectedIds.length])

  const exportFilenameHintContext = useMemo(() => {
    if (selectedIds.length > 0) {
      return t("collections:export.filenameHintContext.selection", "selected items")
    }
    if (applyReadingFilters) {
      return t("collections:export.filenameHintContext.filtered", "filtered list")
    }
    return t("collections:export.filenameHintContext.all", "all items")
  }, [applyReadingFilters, selectedIds.length, t])

  const activeFilterLabels = useMemo(() => {
    const labels: string[] = []
    if (itemsSearch.trim()) labels.push(t("collections:export.filter.search", "search"))
    if (filterStatus !== "all") labels.push(t("collections:export.filter.status", "status"))
    if (filterTags.length > 0) labels.push(t("collections:export.filter.tags", "tags"))
    if (filterFavorite !== null) labels.push(t("collections:export.filter.favorite", "favorite"))
    if (filterDomain.trim()) labels.push(t("collections:export.filter.domain", "domain"))
    if (hasDateFilter) labels.push(t("collections:export.filter.date", "date range"))
    return labels
  }, [
    filterDomain,
    filterFavorite,
    filterStatus,
    filterTags.length,
    hasDateFilter,
    itemsSearch,
    t
  ])

  useEffect(() => {
    if (typeof window === "undefined") return
    if (!window.location.search.includes("e2e=1")) return
    const win = window as unknown as {
      __tldw_exportSelectByTitle?: (title: string) => number
      __tldw_exportSelectAll?: () => number
      __tldw_exportClearSelection?: () => number
      __tldw_exportSetFormat?: (format: string) => void
      __tldw_exportFormat?: string
    }
    win.__tldw_exportSelectByTitle = (title: string) => {
      const next = exportItems
        .filter((item) => item.title === title)
        .map((item) => item.id)
      setSelectedIds(next)
      return next.length
    }
    win.__tldw_exportSelectAll = () => {
      const next = exportItems.map((item) => item.id)
      setSelectedIds(next)
      return next.length
    }
    win.__tldw_exportClearSelection = () => {
      setSelectedIds([])
      return 0
    }
    win.__tldw_exportSetFormat = (format: string) => {
      if (format === "jsonl" || format === "zip") {
        setExportFormat(format)
      } else {
        console.warn(`Invalid export format: ${format}`)
      }
    }
    win.__tldw_exportFormat = exportFormat
    return () => {
      delete win.__tldw_exportSelectByTitle
      delete win.__tldw_exportSelectAll
      delete win.__tldw_exportClearSelection
      delete win.__tldw_exportSetFormat
      delete win.__tldw_exportFormat
    }
  }, [exportFormat, exportItems, setExportFormat])

  useEffect(() => {
    if (typeof window === "undefined") return
    if (!window.location.search.includes("e2e=1")) return
    const win = window as unknown as { __tldw_exportSelectedCount?: number }
    win.__tldw_exportSelectedCount = selectedIds.length
  }, [selectedIds.length])

  useEffect(() => {
    if (typeof window === "undefined") return
    if (!window.location.search.includes("e2e=1")) return
    const win = window as unknown as { __tldw_exportFormat?: string }
    win.__tldw_exportFormat = exportFormat
  }, [exportFormat])

  useEffect(() => {
    if (typeof window === "undefined") return
    if (!window.location.search.includes("e2e=1")) return
    const win = window as unknown as {
      __tldw_exportSelectedCount?: number
      __tldw_lastDownload?: { filename: string; type: string; size: number }
    }
    return () => {
      delete win.__tldw_exportSelectedCount
      delete win.__tldw_lastDownload
    }
  }, [])

  useEffect(() => {
    if (exportItems.length === 0 && storedItems.length > 0) {
      setExportItems(storedItems)
    }
  }, [exportItems.length, storedItems])

  useEffect(() => {
    let active = true
    const loadItems = async () => {
      setExportItemsLoading(true)
      setExportItemsError(null)
      setLoadProgress({ loaded: 0, total: null })
      try {
        const allItems: ReadingItemSummary[] = []
        const pageSize = 200
        let page = 1
        let total: number | null = null
        while (page <= MAX_EXPORT_PAGES) {
          const response = await api.getReadingList({
            page,
            size: pageSize,
            ...(listFilters || {})
          })
          const pageItems = Array.isArray(response?.items) ? response.items : []
          allItems.push(...pageItems)
          if (total === null && typeof response?.total === "number") {
            total = response.total
          }
          if (active) {
            setLoadProgress({ loaded: allItems.length, total })
          }
          if (pageItems.length === 0) break
          if (total !== null && allItems.length >= total) break
          if (pageItems.length < pageSize) break
          page += 1
        }
        if (!active) return
        setExportItems(allItems)
      } catch (error: unknown) {
        if (!active) return
        const msg = error instanceof Error ? error.message : "Failed to load reading items"
        setExportItemsError(msg)
      } finally {
        if (active) {
          setExportItemsLoading(false)
          setLoadProgress({ loaded: 0, total: null })
        }
      }
    }
    void loadItems()
    return () => {
      active = false
    }
  }, [api, listFilters])

  useEffect(() => {
    setSelectedIds((previous) =>
      previous.filter((id) => exportItems.some((item) => item.id === id))
    )
  }, [exportItems])

  const filteredItems = useMemo(() => {
    const q = exportSearch.trim().toLowerCase()
    if (!q) return exportItems
    return exportItems.filter((item) => {
      const haystack = `${item.title} ${item.url || ""} ${item.domain || ""}`.toLowerCase()
      return haystack.includes(q)
    })
  }, [exportItems, exportSearch])

  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds])
  const allFilteredSelected =
    filteredItems.length > 0 && filteredItems.every((item) => selectedSet.has(item.id))
  const someFilteredSelected =
    filteredItems.length > 0 && filteredItems.some((item) => selectedSet.has(item.id))

  const handleSelectAll = useCallback(
    (checked: boolean) => {
      if (!checked) {
        setSelectedIds((prev) => prev.filter((id) => !filteredItems.some((item) => item.id === id)))
        return
      }
      setSelectedIds((prev) => {
        const next = new Set(prev)
        filteredItems.forEach((item) => next.add(item.id))
        return Array.from(next)
      })
    },
    [filteredItems]
  )

  const handleClearSelection = useCallback(() => {
    setSelectedIds([])
  }, [])

  const {
    focusedIndex,
    handleItemClick,
    handleItemToggle,
    handleKeyDown,
    listRef
  } = useSelectionKeyboard({
    items: filteredItems,
    selectedIds,
    getItemId: (item) => item.id,
    onSelectionChange: setSelectedIds
  })
  const lastShiftKeyRef = useRef(false)

  const resolveItemsWithDetail = useCallback(async (items: ReadingItemSummary[]): Promise<ReadingItem[]> => {
    const detailed = await Promise.all(
      items.map(async (item) => {
        try {
          return await api.getReadingItem(item.id)
        } catch (error) {
          console.warn(`Failed to fetch full item ${item.id}, using summary:`, error)
          return item as ReadingItem
        }
      })
    )
    return detailed as ReadingItem[]
  }, [api])

  const resolveSelectedItems = useCallback(async (): Promise<ReadingItem[]> => {
    const selected = exportItems.filter((item) => selectedSet.has(item.id))
    return await resolveItemsWithDetail(selected)
  }, [exportItems, resolveItemsWithDetail, selectedSet])

  const buildJsonlPayload = useCallback(async (items: ReadingItem[]) => {
    const lines = await Promise.all(
      items.map(async (item) => {
        const payload: Record<string, unknown> = {
          id: item.id,
          title: item.title,
          url: item.url || item.canonical_url || "",
          canonical_url: item.canonical_url,
          domain: item.domain,
          summary: item.summary,
          status: item.status,
          favorite: item.favorite,
          tags: item.tags,
          created_at: item.created_at,
          updated_at: item.updated_at,
          published_at: item.published_at
        }
        if (includeNotes) {
          payload.notes = item.notes
        }
        if (includeHighlights) {
          try {
            payload.highlights = await api.getHighlights(item.id)
          } catch (error) {
            console.debug(`Failed to fetch highlights for ${item.id}:`, error)
            payload.highlights = []
          }
        }
        return JSON.stringify(payload)
      })
    )
    return lines.join("\n")
  }, [api, includeHighlights, includeNotes])

  const triggerDownload = (blob: Blob, filename: string) => {
    if (typeof window !== "undefined" && window.location.search.includes("e2e=1")) {
      const win = window as unknown as {
        __tldw_lastDownload?: { filename: string; type: string; size: number }
      }
      win.__tldw_lastDownload = { filename, type: blob.type, size: blob.size }
    }
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleExport = useCallback(async () => {
    setExportInProgress(true)
    try {
      if (selectedIds.length > 0 && exportFormat === "zip") {
        message.warning(
          t(
            "collections:export.zipSelectionUnsupported",
            "ZIP export doesn't support item selection yet."
          )
        )
        setExportInProgress(false)
        return
      }
      if (selectedIds.length > 0) {
        const items = await resolveSelectedItems()
        const payload = await buildJsonlPayload(items)
        const blob = new Blob([payload], { type: "application/x-ndjson" })
        triggerDownload(blob, "reading_export_selection.jsonl")
        message.success(t("collections:export.success", "Export ready for download"))
        return
      }

      if (applyReadingFilters && hasDateFilter) {
        if (exportFormat === "zip") {
          message.warning(
            t(
              "collections:export.dateZipUnsupported",
              "Date range filter export is available in JSONL mode only."
            )
          )
          return
        }
        const detailedItems = await resolveItemsWithDetail(exportItems)
        const payload = await buildJsonlPayload(detailedItems)
        const blob = new Blob([payload], { type: "application/x-ndjson" })
        triggerDownload(blob, "reading_export_filtered.jsonl")
        message.success(t("collections:export.success", "Export ready for download"))
        return
      }

      const response = await api.exportReadingList({
        format: exportFormat,
        ...(serverExportFilters || {}),
        include_highlights: includeHighlights,
        include_notes: includeNotes
      })
      triggerDownload(response.blob, response.filename)
      message.success(t("collections:export.success", "Export ready for download"))
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : "Export failed"
      message.error(msg)
    } finally {
      setExportInProgress(false)
    }
  }, [
    api,
    applyReadingFilters,
    buildJsonlPayload,
    exportFormat,
    exportItems,
    hasDateFilter,
    includeHighlights,
    includeNotes,
    resolveItemsWithDetail,
    resolveSelectedItems,
    selectedIds.length,
    serverExportFilters,
    setExportInProgress,
    t
  ])

  const handleCopy = useCallback(async () => {
    if (selectedIds.length === 0) {
      message.warning(
        t("collections:export.selectItems", "Select items to copy first")
      )
      return
    }
    setCopying(true)
    try {
      const items = await resolveSelectedItems()
      const payload = await buildJsonlPayload(items)
      await navigator.clipboard.writeText(payload)
      message.success(t("collections:export.copied", "Copied to clipboard"))
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : "Copy failed"
      message.error(msg)
    } finally {
      setCopying(false)
    }
  }, [buildJsonlPayload, resolveSelectedItems, selectedIds.length, t])

  return (
    <Card
      title={
        <span className="flex items-center gap-2">
          <Download className="h-5 w-5" />
          {t("collections:export.title", "Export")}
        </span>
      }
    >
      <div className="space-y-4">
        <div>
          <label className="mb-2 block text-sm font-medium">
            {t("collections:export.format", "Format")}
          </label>
          <Radio.Group
            value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value)}
          >
            {EXPORT_FORMATS.map((fmt) => (
              <Radio.Button key={fmt.value} value={fmt.value}>
                {t(fmt.labelKey)}
              </Radio.Button>
            ))}
          </Radio.Group>
        </div>

        <div className="space-y-2 rounded-md border border-border p-3">
          <Checkbox
            checked={applyReadingFilters}
            onChange={(e) => setApplyReadingFilters(e.target.checked)}
          >
            {t("collections:export.useReadingFilters", "Use current Reading filters")}
          </Checkbox>
          {applyReadingFilters && (
            <p className="text-xs text-text-muted">
              {activeFilterLabels.length > 0
                ? t(
                    "collections:export.activeFilters",
                    "Active filters: {{filters}}",
                    { filters: activeFilterLabels.join(", ") }
                  )
                : t(
                    "collections:export.noActiveFilters",
                    "No active filters. Export will include all items."
                  )}
            </p>
          )}
          <div className="grid gap-2 sm:grid-cols-2">
            <Checkbox
              checked={includeHighlights}
              onChange={(e) => setIncludeHighlights(e.target.checked)}
            >
              {t("collections:export.includeHighlights", "Include highlights")}
            </Checkbox>
            <Checkbox
              checked={includeNotes}
              onChange={(e) => setIncludeNotes(e.target.checked)}
            >
              {t("collections:export.includeNotes", "Include notes")}
            </Checkbox>
          </div>
          <p
            className="text-xs text-text-muted"
            data-testid="export-filename-hint"
          >
            {t("collections:export.filenameHint", "Filename hint: {{filename}}", {
              filename: exportFilenameHint
            })}
            {" · "}
            {t("collections:export.filenameHintContext", "Scope: {{scope}}", {
              scope: exportFilenameHintContext
            })}
          </p>
        </div>

        <div>
          <label className="mb-2 block text-sm font-medium">
            {t("collections:export.items", "Items to Export")}
          </label>
          <Input
            value={exportSearch}
            onChange={(e) => setExportSearch(e.target.value)}
            placeholder={t("collections:export.searchPlaceholder", "Search items...")}
            size="small"
            allowClear
          />
          <div className="mt-2 flex items-center justify-between text-xs text-text-muted">
            <Checkbox
              indeterminate={someFilteredSelected && !allFilteredSelected}
              checked={allFilteredSelected}
              onChange={(e) => handleSelectAll(e.target.checked)}
            >
              {t("collections:export.selectAll", "Select all")}
            </Checkbox>
            <div className="flex items-center gap-2">
              <span aria-live="polite" aria-atomic="true">
                {t("collections:export.selectedCount", "{{count}} selected", {
                  count: selectedIds.length
                })}
              </span>
              {selectedIds.length > 0 && (
                <Button type="link" size="small" onClick={handleClearSelection}>
                  {t("collections:export.clearSelection", "Clear")}
                </Button>
              )}
            </div>
          </div>
          <div
            ref={listRef as React.RefObject<HTMLDivElement>}
            tabIndex={0}
            onKeyDownCapture={handleKeyDown}
            className="mt-2 max-h-48 overflow-auto rounded-md border border-border focus:outline-none focus:ring-2 focus:ring-primary focus:ring-inset"
            role="listbox"
            aria-label={t("collections:export.itemList", "Export items list")}
          >
            {exportItemsLoading ? (
              <div className="flex flex-col items-center justify-center gap-2 py-6">
                <Spin size="small" />
                {loadProgress.total !== null && (
                  <span className="text-xs text-text-muted">
                    {t("collections:export.loadingProgress", "Loading {{loaded}} / {{total}} items...", {
                      loaded: loadProgress.loaded,
                      total: loadProgress.total
                    })}
                  </span>
                )}
              </div>
            ) : exportItemsError ? (
              <Empty description={exportItemsError} image={Empty.PRESENTED_IMAGE_SIMPLE} />
            ) : filteredItems.length === 0 ? (
              <Empty
                description={t("collections:export.noItems", "No items found")}
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              />
            ) : (
              <List
                size="small"
                dataSource={filteredItems}
                renderItem={(item, index) => {
                  const isSelected = selectedSet.has(item.id)
                  const isFocused = index === focusedIndex
                  return (
                    <List.Item
                      data-selection-item
                      className={`cursor-pointer py-2 hover:bg-surface ${
                        isFocused ? "ring-2 ring-inset ring-primary" : ""
                      }`}
                      onClick={(e) => handleItemClick(index, e)}
                      role="option"
                      aria-selected={isSelected}
                    >
                      <Checkbox
                        checked={isSelected}
                        onClick={(e) => {
                          lastShiftKeyRef.current = e.shiftKey
                          e.stopPropagation()
                        }}
                        onKeyDown={(e) => {
                          lastShiftKeyRef.current = e.shiftKey
                        }}
                        onChange={() => {
                          const shiftKey = lastShiftKeyRef.current
                          lastShiftKeyRef.current = false
                          handleItemToggle(index, { shiftKey })
                        }}
                      >
                        <span className="text-sm">{item.title}</span>
                      </Checkbox>
                    </List.Item>
                  )
                }}
              />
            )}
          </div>
          <p className="mt-2 text-xs text-text-muted">
            {t(
              "collections:export.selectionHint",
              "Select items to export, or leave empty to export everything."
            )}
          </p>
        </div>

        {exportFormat === "zip" && selectedIds.length > 0 && (
          <Alert
            variant="warning"
            role="status"
            aria-live="off"
            title={
              <span role="alert" className="text-text">
                {t(
                  "collections:export.zipSelectionWarning",
                  "ZIP export doesn't support item selection"
                )}
              </span>
            }
            secondaryAction={{
              label: t("collections:export.switchToJsonl", "Switch to JSONL"),
              onClick: () => setExportFormat("jsonl")
            }}
          >
            <span className="text-text">
              {t(
                "collections:export.zipSelectionHint",
                "To export selected items, switch to JSONL format or clear your selection to export all items as ZIP."
              )}
            </span>
          </Alert>
        )}

        {exportFormat === "zip" && applyReadingFilters && hasDateFilter && selectedIds.length === 0 && (
          <Alert
            variant="warning"
            title={
              <span className="text-text">
                {t(
                  "collections:export.dateZipUnsupported",
                  "Date range filter export is available in JSONL mode only."
                )}
              </span>
            }
          />
        )}

        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <Button onClick={handleCopy} disabled={selectedIds.length === 0} loading={copying}>
            {t("collections:export.copy", "Copy JSONL")}
          </Button>
          <Button
            type="primary"
            icon={<Download className="h-4 w-4" />}
            onClick={handleExport}
            loading={exportInProgress}
            block
          >
            {t("collections:export.download", "Download Export")}
          </Button>
        </div>
      </div>
    </Card>
  )
}
