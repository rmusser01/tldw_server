import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Alert,
  Button,
  Dropdown,
  Progress,
  Select,
  Space,
  Table,
  Tooltip,
  message
} from "antd"
import type { ColumnsType } from "antd/es/table"
import { ChevronDown, Eye, RefreshCw } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useWatchlistsStore } from "@/store/watchlists"
import {
  fetchJobRuns,
  cancelWatchlistRun,
  exportRunsCsv,
  fetchWatchlistJobs,
  fetchWatchlistRuns
} from "@/services/watchlists"
import type { WatchlistJob, WatchlistRun } from "@/types/watchlists"
import { formatRelativeTime } from "@/utils/dateFormatters"
import { StatusTag } from "../shared"
import { RunDetailDrawer } from "./RunDetailDrawer"
import { Download, Square } from "lucide-react"
import { mapWatchlistsError } from "../shared/watchlists-error"
import {
  getRunFailureHint,
  resolveStalledRunNotification
} from "./run-notifications"
import { fetchFilteredJobRuns, RUNS_CLIENT_FILTER_PAGE_SIZE } from "./runs-filter-fetch"
import { hasActiveWatchlistRuns } from "./polling-utils"

const POLL_INTERVAL_MS = 5000
const DEFAULT_RUNS_CSV_SERVER_THRESHOLD = 2000
const RUNS_API_PAGE_SIZE = RUNS_CLIENT_FILTER_PAGE_SIZE
const RUNS_CSV_SERVER_PAGE_SIZE = 1000
const RUNS_ADVANCED_FILTERS_STORAGE_KEY = "watchlists:runs:advanced-filters:v1"
const RUN_STALLED_THRESHOLD_MS = 45 * 60_000
type RunsCsvTalliesMode = "none" | "per_run" | "aggregate"

const normalizeRunStatus = (status: unknown): string =>
  String(status || "").trim().toLowerCase()

const toStatusDefaultLabel = (status: string): string =>
  status
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase())

const resolveRunsCsvServerThreshold = (): number => {
  const raw = process.env.NEXT_PUBLIC_RUNS_CSV_SERVER_THRESHOLD
  if (!raw || !raw.trim()) return DEFAULT_RUNS_CSV_SERVER_THRESHOLD
  const parsed = Number(raw)
  if (!Number.isFinite(parsed) || parsed < 0) return DEFAULT_RUNS_CSV_SERVER_THRESHOLD
  return Math.floor(parsed)
}

const escapeCsvCell = (value: unknown): string => {
  const text = value == null ? "" : String(value)
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, "\"\"")}"`
  }
  return text
}

const downloadCsv = (content: string, filename: string): void => {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  URL.revokeObjectURL(url)
}

const readStoredDisclosureState = (key: string): boolean | null => {
  if (typeof window === "undefined") return null
  try {
    const raw = window.localStorage.getItem(key)
    if (raw === "1") return true
    if (raw === "0") return false
  } catch {
    // Ignore storage access errors and fall back to defaults.
  }
  return null
}

const persistDisclosureState = (key: string, value: boolean): void => {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(key, value ? "1" : "0")
  } catch {
    // Ignore storage access errors; disclosure defaults still work.
  }
}

export const RunsTab: React.FC = () => {
  const { t } = useTranslation(["watchlists", "common"])

  // Store state
  const runs = useWatchlistsStore((s) => s.runs)
  const runsLoading = useWatchlistsStore((s) => s.runsLoading)
  const runsTotal = useWatchlistsStore((s) => s.runsTotal)
  const activeTab = useWatchlistsStore((s) => s.activeTab)
  const runsPage = useWatchlistsStore((s) => s.runsPage)
  const runsPageSize = useWatchlistsStore((s) => s.runsPageSize)
  const runsJobFilter = useWatchlistsStore((s) => s.runsJobFilter)
  const runsStatusFilter = useWatchlistsStore((s) => s.runsStatusFilter)
  const pollingActive = useWatchlistsStore((s) => s.pollingActive)
  const runDetailOpen = useWatchlistsStore((s) => s.runDetailOpen)
  const selectedRunId = useWatchlistsStore((s) => s.selectedRunId)
  const selectedWatchlistId = useWatchlistsStore((s) => s.selectedWatchlistId)
  const [jobs, setJobs] = useState<WatchlistJob[]>([])
  const [exportingRunsCsv, setExportingRunsCsv] = useState(false)
  const [runsCsvTalliesMode, setRunsCsvTalliesMode] = useState<RunsCsvTalliesMode>("none")
  const [lastRefreshedAt, setLastRefreshedAt] = useState<string | null>(null)
  const [cancellingRunIds, setCancellingRunIds] = useState<number[]>([])
  const [failedCancelRunIds, setFailedCancelRunIds] = useState<number[]>([])
  const [runsLoadError, setRunsLoadError] = useState<ReturnType<typeof mapWatchlistsError> | null>(null)
  const [runsLiveAnnouncement, setRunsLiveAnnouncement] = useState("")
  const hasActiveRunsFilters = Boolean(runsJobFilter || runsStatusFilter)
  const [showAdvancedFilters, setShowAdvancedFilters] = useState<boolean>(() => {
    const stored = readStoredDisclosureState(RUNS_ADVANCED_FILTERS_STORAGE_KEY)
    return stored ?? hasActiveRunsFilters
  })
  const [documentVisible, setDocumentVisible] = useState<boolean>(() => {
    if (typeof document === "undefined") return true
    return document.visibilityState !== "hidden"
  })

  // Store actions
  const setRuns = useWatchlistsStore((s) => s.setRuns)
  const setRunsLoading = useWatchlistsStore((s) => s.setRunsLoading)
  const setRunsPage = useWatchlistsStore((s) => s.setRunsPage)
  const setRunsPageSize = useWatchlistsStore((s) => s.setRunsPageSize)
  const setRunsJobFilter = useWatchlistsStore((s) => s.setRunsJobFilter)
  const setRunsStatusFilter = useWatchlistsStore((s) => s.setRunsStatusFilter)
  const setOutputsJobFilter = useWatchlistsStore((s) => s.setOutputsJobFilter)
  const setOutputsRunFilter = useWatchlistsStore((s) => s.setOutputsRunFilter)
  const setActiveTab = useWatchlistsStore((s) => s.setActiveTab)
  const setPollingActive = useWatchlistsStore((s) => s.setPollingActive)
  const openRunDetail = useWatchlistsStore((s) => s.openRunDetail)
  const closeRunDetail = useWatchlistsStore((s) => s.closeRunDetail)
  const updateRunInList = useWatchlistsStore((s) => s.updateRunInList)

  // Refs for polling
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const previousVisibleRunsStatusRef = useRef<Map<number, string>>(new Map())
  const hasRunsAnnouncementBaselineRef = useRef(false)

  // Fetch runs
  const loadRuns = useCallback(async (showLoading = true) => {
    if (showLoading) setRunsLoading(true)
    try {
      let items: WatchlistRun[] = []
      let total = 0
      let pagedItems: WatchlistRun[] = []

      if (runsJobFilter && runsStatusFilter) {
        const filtered = await fetchFilteredJobRuns({
          jobId: runsJobFilter,
          statusFilter: runsStatusFilter,
          currentPage: runsPage,
          pageSize: runsPageSize,
          fetchPage: fetchJobRuns
        })

        const start = (runsPage - 1) * runsPageSize
        const end = runsPage * runsPageSize
        items = filtered.filteredItems
        total = filtered.exactTotal
          ? items.length
          : Math.max(items.length, end + (filtered.hasMoreInSource ? 1 : 0))
        pagedItems = items.slice(start, end)
      } else if (runsJobFilter) {
        const result = await fetchJobRuns(runsJobFilter, {
          page: runsPage,
          size: runsPageSize
        })
        items = result.items || []
        total = result.total || items.length
        pagedItems = items
      } else {
        const result = await fetchWatchlistRuns({
          watchlist_id: selectedWatchlistId ?? undefined,
          q: runsStatusFilter || undefined,
          page: runsPage,
          size: runsPageSize
        })
        items = result.items || []
        total = result.total || items.length
        pagedItems = items
      }

      setRuns(pagedItems, total)
      setRunsLoadError(null)

      // Check if any runs are still running
      setPollingActive(hasActiveWatchlistRuns(items))
      setLastRefreshedAt(new Date().toISOString())
    } catch (err) {
      console.error("Failed to fetch runs:", err)
      setRunsLoadError(
        mapWatchlistsError(err, {
          t,
          context: t("watchlists:runs.title", "Activity"),
          fallbackMessage: t("watchlists:runs.fetchError", "Failed to load runs")
        })
      )
    } finally {
      if (showLoading) setRunsLoading(false)
    }
  }, [
    runsJobFilter,
    runsStatusFilter,
    selectedWatchlistId,
    runsPage,
    runsPageSize,
    setRuns,
    setRunsLoading,
    setPollingActive,
    t
  ])

  // Load jobs for filter dropdown
  const loadJobs = useCallback(async () => {
    try {
      const result = await fetchWatchlistJobs({
        watchlist_id: selectedWatchlistId ?? undefined,
        page: 1,
        size: 200
      })
      setJobs(result.items || [])
    } catch (err) {
      console.error("Failed to fetch jobs:", err)
    }
  }, [selectedWatchlistId])

  // Initial load
  useEffect(() => {
    loadRuns()
    loadJobs()
  }, [loadRuns, loadJobs])

  useEffect(() => {
    if (runsJobFilter && runsCsvTalliesMode === "aggregate") {
      setRunsCsvTalliesMode("per_run")
    }
  }, [runsJobFilter, runsCsvTalliesMode])

  useEffect(() => {
    if (hasActiveRunsFilters && !showAdvancedFilters) {
      setShowAdvancedFilters(true)
    }
  }, [hasActiveRunsFilters, showAdvancedFilters])

  useEffect(() => {
    persistDisclosureState(RUNS_ADVANCED_FILTERS_STORAGE_KEY, showAdvancedFilters)
  }, [showAdvancedFilters])

  useEffect(() => {
    if (typeof document === "undefined") return
    const handleVisibilityChange = () => {
      setDocumentVisible(document.visibilityState !== "hidden")
    }
    document.addEventListener("visibilitychange", handleVisibilityChange)
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange)
    }
  }, [])

  const getRunStatusLabel = useCallback((status: string) => {
    const normalized = normalizeRunStatus(status)
    if (!normalized) {
      return t("watchlists:runs.statusLabels.unknown", "Unknown")
    }
    return t(
      `watchlists:runs.statusLabels.${normalized}`,
      toStatusDefaultLabel(normalized)
    )
  }, [t])

  const shouldAutoRefreshRuns = pollingActive && activeTab === "runs" && documentVisible

  // Polling for active runs
  useEffect(() => {
    if (shouldAutoRefreshRuns) {
      pollIntervalRef.current = setInterval(() => {
        loadRuns(false)
      }, POLL_INTERVAL_MS)
    } else if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = null
    }

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [loadRuns, shouldAutoRefreshRuns])

  useEffect(() => {
    const visibleRuns = Array.isArray(runs) ? runs : []
    const nextStatuses = new Map<number, string>()
    visibleRuns.forEach((run) => {
      nextStatuses.set(run.id, normalizeRunStatus(run.status))
    })

    if (!hasRunsAnnouncementBaselineRef.current) {
      hasRunsAnnouncementBaselineRef.current = true
      previousVisibleRunsStatusRef.current = nextStatuses
      return
    }

    const changedRuns = visibleRuns.filter((run) => {
      const previousStatus = previousVisibleRunsStatusRef.current.get(run.id)
      const currentStatus = normalizeRunStatus(run.status)
      return Boolean(previousStatus) && previousStatus !== currentStatus
    })

    if (changedRuns.length === 1) {
      const changedRun = changedRuns[0]
      setRunsLiveAnnouncement(
        t("watchlists:runs.live.statusChangedSingle", "Run #{{id}} status changed to {{status}}.", {
          id: changedRun.id,
          status: getRunStatusLabel(changedRun.status)
        })
      )
    } else if (changedRuns.length > 1) {
      setRunsLiveAnnouncement(
        t("watchlists:runs.live.statusChangedMultiple", "{{count}} runs changed status.", {
          count: changedRuns.length
        })
      )
    }

    previousVisibleRunsStatusRef.current = nextStatuses
  }, [getRunStatusLabel, runs, t])

  useEffect(() => {
    if (!runsLoadError) return
    setRunsLiveAnnouncement(
      t("watchlists:runs.live.loadError", "Activity refresh failed.")
    )
  }, [runsLoadError, t])

  // Get job name by ID
  const getJobName = useCallback(
    (jobId: number) => {
      const job = jobs.find((j) => j.id === jobId)
      return job?.name || `Monitor #${jobId}`
    },
    [jobs]
  )

  // Calculate duration
  const calculateDuration = (run: WatchlistRun): string => {
    if (!run.started_at) return "-"
    const start = new Date(run.started_at)
    const end = run.finished_at ? new Date(run.finished_at) : new Date()
    const durationMs = end.getTime() - start.getTime()

    if (durationMs < 1000) return "<1s"
    if (durationMs < 60000) return `${Math.round(durationMs / 1000)}s`
    if (durationMs < 3600000) return `${Math.round(durationMs / 60000)}m`
    return `${Math.round(durationMs / 3600000)}h`
  }

  const fetchAllRunsForClientCsv = useCallback(async (): Promise<WatchlistRun[]> => {
    const useClientFilter = Boolean(runsJobFilter && runsStatusFilter)
    let page = 1
    const collected: WatchlistRun[] = []
    let hasMore = true
    while (hasMore) {
      const result = runsJobFilter
        ? await fetchJobRuns(runsJobFilter, { page, size: RUNS_API_PAGE_SIZE })
        : await fetchWatchlistRuns({
            watchlist_id: selectedWatchlistId ?? undefined,
            q: runsStatusFilter || undefined,
            page,
            size: RUNS_API_PAGE_SIZE
          })
      const items = Array.isArray(result.items) ? result.items : []
      if (!items.length) break
      collected.push(...items)
      if (typeof result.has_more === "boolean") {
        hasMore = result.has_more
      } else {
        hasMore = items.length >= RUNS_API_PAGE_SIZE
      }
      page += 1
    }
    if (useClientFilter && runsStatusFilter) {
      return collected.filter((run) => run.status === runsStatusFilter)
    }
    return collected
  }, [runsJobFilter, runsStatusFilter, selectedWatchlistId])

  const toRunsCsv = useCallback((items: WatchlistRun[]): string => {
    const rows = [
      [
        "id",
        "job_id",
        "status",
        "started_at",
        "finished_at",
        "items_found",
        "items_ingested",
        "items_filtered",
        "items_duplicates",
        "items_errored",
        "filters_include",
        "filters_exclude",
        "filters_flag"
      ].join(",")
    ]
    items.forEach((run) => {
      const stats = run.stats || {}
      const filtersActionsRaw = (stats as Record<string, unknown>)?.filters_actions
      const filtersActions =
        filtersActionsRaw && typeof filtersActionsRaw === "object"
          ? (filtersActionsRaw as Record<string, number>)
          : {}
      rows.push(
        [
          run.id,
          run.job_id,
          run.status,
          run.started_at || "",
          run.finished_at || "",
          stats?.items_found ?? 0,
          stats?.items_ingested ?? 0,
          stats?.items_filtered ?? 0,
          stats?.items_duplicates ?? stats?.items_duplicate ?? 0,
          stats?.items_errored ?? 0,
          filtersActions.include ?? 0,
          filtersActions.exclude ?? 0,
          filtersActions.flag ?? 0
        ]
          .map(escapeCsvCell)
          .join(",")
      )
    })
    return `${rows.join("\n")}\n`
  }, [])

  const fetchServerRunsCsvMerged = useCallback(async (mode: RunsCsvTalliesMode): Promise<string> => {
    const scope: "global" | "job" = runsJobFilter ? "job" : "global"
    const q = !runsJobFilter && runsStatusFilter ? runsStatusFilter : undefined
    const talliesMode: RunsCsvTalliesMode =
      runsJobFilter && mode === "aggregate"
        ? "per_run"
        : mode
    if (scope === "global" && talliesMode === "aggregate") {
      return exportRunsCsv({
        scope,
        watchlist_id: selectedWatchlistId ?? undefined,
        q,
        include_tallies: true,
        tallies_mode: "aggregate"
      })
    }
    let page = 1
    let header = ""
    const dataRows: string[] = []
    const includeTallies = talliesMode === "per_run"

    // Merge paginated CSV chunks into a single downloadable CSV.
    while (true) {
      const csvChunk = await exportRunsCsv({
        scope,
        watchlist_id: selectedWatchlistId ?? undefined,
        job_id: runsJobFilter || undefined,
        q,
        page,
        size: RUNS_CSV_SERVER_PAGE_SIZE,
        include_tallies: includeTallies,
        tallies_mode: includeTallies ? "per_run" : undefined
      })
      const lines = csvChunk.split("\n")
      if (!header) {
        header = lines[0] || ""
      }
      const rows = lines.slice(1).filter((line) => line.trim().length > 0)
      if (!rows.length) break
      const filteredRows =
        runsJobFilter && runsStatusFilter
          ? rows.filter((line) => {
              const cols = line.split(",")
              return (cols[2] || "").trim().toLowerCase() === runsStatusFilter.toLowerCase()
            })
          : rows
      dataRows.push(...filteredRows)
      if (rows.length < RUNS_CSV_SERVER_PAGE_SIZE) break
      page += 1
      if (page > 200) break
    }
    if (!header) return ""
    return `${[header, ...dataRows].join("\n")}\n`
  }, [runsJobFilter, runsStatusFilter, selectedWatchlistId])

  const handleExportRunsCsv = useCallback(async (modeOverride?: RunsCsvTalliesMode) => {
    try {
      setExportingRunsCsv(true)
      const threshold = resolveRunsCsvServerThreshold()
      const rowEstimate = Math.max(Number(runsTotal || 0), Array.isArray(runs) ? runs.length : 0)
      const selectedMode = modeOverride ?? runsCsvTalliesMode
      const talliesMode: RunsCsvTalliesMode =
        runsJobFilter && selectedMode === "aggregate"
          ? "per_run"
          : selectedMode
      const preferServerCsv = talliesMode !== "none" || rowEstimate >= threshold
      const csv = preferServerCsv
        ? await fetchServerRunsCsvMerged(selectedMode)
        : toRunsCsv(await fetchAllRunsForClientCsv())
      if (!csv || !csv.trim()) {
        message.warning(t("watchlists:runs.exportEmpty", "No runs available to export"))
        return
      }
      const filenameSuffix = runsJobFilter ? `job_${runsJobFilter}` : "global"
      const filename =
        talliesMode === "aggregate"
          ? `watchlists_runs_global_tallies_${Date.now()}.csv`
          : talliesMode === "per_run"
            ? `watchlists_runs_${filenameSuffix}_with_tallies_${Date.now()}.csv`
            : `watchlists_runs_${filenameSuffix}_${Date.now()}.csv`
      downloadCsv(csv, filename)
      message.success(t("watchlists:runs.exported", "Runs CSV exported"))
    } catch (err) {
      console.error("Failed to export runs CSV:", err)
      message.error(t("watchlists:runs.exportError", "Failed to export runs CSV"))
    } finally {
      setExportingRunsCsv(false)
    }
  }, [
    fetchAllRunsForClientCsv,
    fetchServerRunsCsvMerged,
    runs,
    runsCsvTalliesMode,
    runsJobFilter,
    runsTotal,
    t,
    toRunsCsv
  ])

  const exportModeOptions: Array<{
    value: RunsCsvTalliesMode
    label: string
    disabled?: boolean
  }> = [
    { value: "none", label: t("watchlists:runs.exportMode.standard", "Standard CSV") },
    { value: "per_run", label: t("watchlists:runs.exportMode.perRun", "Per-run tallies") },
    {
      value: "aggregate",
      label: t("watchlists:runs.exportMode.aggregate", "Global tallies summary"),
      disabled: Boolean(runsJobFilter)
    }
  ]

  const activeExportModeLabel = (
    exportModeOptions.find((option) => option.value === runsCsvTalliesMode && !option.disabled)?.label ||
    exportModeOptions[0].label
  )

  // Table columns
  const allColumns: ColumnsType<WatchlistRun> = [
    {
      title: t("watchlists:runs.columns.job", "Monitor"),
      key: "job",
      width: 200,
      ellipsis: true,
      render: (_, record) => (
        <span className="font-medium">{getJobName(record.job_id)}</span>
      )
    },
    {
      title: t("watchlists:runs.columns.status", "Status"),
      dataIndex: "status",
      key: "status",
      width: 120,
      render: (status: string, record) => {
        const statusNormalized = normalizeRunStatus(status)
        const ingestedCount = Number(record.stats?.items_ingested || 0)
        const hasBriefingOutputSignal =
          statusNormalized === "completed" &&
          Number.isFinite(ingestedCount) &&
          ingestedCount > 0
        const progressPercent = Math.round(
          ((record.stats?.items_ingested || 0) /
            Math.max(record.stats?.items_found || 1, 1)) *
            100
        )
        return (
          <div className="flex items-center gap-2">
            <StatusTag status={status} />
            {status === "running" && (
              <Progress
                percent={progressPercent}
                size="small"
                showInfo={false}
                className="w-16"
                aria-label={t("watchlists:runs.progressAria", "Run ingestion progress {{percent}} percent", {
                  percent: progressPercent
                })}
              />
            )}
            {hasBriefingOutputSignal && (
              <span
                className="text-xs text-text-muted"
                data-testid={`watchlists-run-briefing-included-${record.id}`}
              >
                {t("watchlists:runs.includedInBriefing", "Included in briefing")}
              </span>
            )}
          </div>
        )
      }
    },
    {
      title: t("watchlists:runs.columns.started", "Started"),
      dataIndex: "started_at",
      key: "started_at",
      width: 140,
      render: (date: string | null) =>
        date ? (
          <span className="text-sm text-text-muted">
            {formatRelativeTime(date, t)}
          </span>
        ) : (
          <span className="text-sm text-text-subtle">-</span>
        )
    },
    {
      title: t("watchlists:runs.columns.duration", "Duration"),
      key: "duration",
      width: 80,
      render: (_, record) => (
        <span className="text-sm text-text-muted">
          {calculateDuration(record)}
        </span>
      )
    },
    {
      title: t("watchlists:runs.columns.itemsFound", "Found"),
      key: "items_found",
      width: 80,
      align: "center",
      render: (_, record) => (
        <span className="text-sm">
          {record.stats?.items_found ?? "-"}
        </span>
      )
    },
    {
      title: t("watchlists:runs.columns.itemsProcessed", "Processed"),
      key: "items_processed",
      width: 100,
      align: "center",
      render: (_, record) => (
        <span className="text-sm">
          {record.stats?.items_ingested ?? "-"}
        </span>
      )
    },
    {
      title: t("watchlists:runs.columns.itemsFiltered", "Filtered"),
      key: "items_filtered",
      width: 100,
      align: "center",
      render: (_, record) => (
        <span className="text-sm">
          {record.stats?.items_filtered ?? "-"}
        </span>
      )
    },
    {
      title: t("watchlists:runs.columns.itemsErrored", "Errors"),
      key: "items_errored",
      width: 90,
      align: "center",
      render: (_, record) => (
        <span className="text-sm">
          {record.stats?.items_errored ?? "-"}
        </span>
      )
    },
    {
      title: t("watchlists:runs.columns.actions", "Actions"),
      key: "actions",
      width: 140,
      align: "center",
      render: (_, record) => {
        const status = String(record.status || "").toLowerCase()
        const isCancellable = status === "running" || status === "pending" || status === "queued"
        const cancelling = cancellingRunIds.includes(record.id)
        const cancelFailed = failedCancelRunIds.includes(record.id)
        return (
          <Space size={4}>
            <Tooltip title={t("watchlists:runs.viewDetails", "View Details")}>
              <Button
                type="text"
                size="small"
                aria-label={t("watchlists:runs.viewDetails", "View Details")}
                icon={<Eye className="h-4 w-4" />}
                onClick={() => openRunDetail(record.id)}
              />
            </Tooltip>
            <Tooltip title={t("watchlists:runs.openReports", "Open Reports")}>
              <Button
                type="text"
                size="small"
                aria-label={t("watchlists:runs.openReports", "Open Reports")}
                icon={<Download className="h-4 w-4" />}
                data-testid={`watchlists-run-open-outputs-${record.id}`}
                onClick={() => openRunOutputs(record.id, record.job_id)}
              />
            </Tooltip>
            {isCancellable && (
              <Tooltip
                title={
                  cancelling
                    ? t("watchlists:runs.cancelling", "Cancelling...")
                    : cancelFailed
                      ? t("watchlists:runs.cancelFailedRetry", "Cancel failed. Retry.")
                      : t("watchlists:runs.cancelRun", "Cancel run")
                }
              >
                <Button
                  type="text"
                  size="small"
                  danger
                  loading={cancelling}
                  aria-label={
                    cancelling
                      ? t("watchlists:runs.cancelling", "Cancelling...")
                      : cancelFailed
                        ? t("watchlists:runs.cancelFailedRetry", "Cancel failed. Retry.")
                        : t("watchlists:runs.cancelRun", "Cancel run")
                  }
                  icon={!cancelling ? <Square className="h-4 w-4" /> : undefined}
                  data-testid={`watchlists-run-cancel-${record.id}`}
                  onClick={async () => {
                    if (cancelling) return
                    setCancellingRunIds((prev) => (prev.includes(record.id) ? prev : [...prev, record.id]))
                    setFailedCancelRunIds((prev) => prev.filter((id) => id !== record.id))
                    try {
                      const result = await cancelWatchlistRun(record.id)
                      if (!result?.cancelled) {
                        setFailedCancelRunIds((prev) => (prev.includes(record.id) ? prev : [...prev, record.id]))
                        message.error(
                          t("watchlists:runs.cancelRunError", "Failed to cancel run")
                        )
                        return
                      }
                      updateRunInList(record.id, {
                        status: "cancelled" as any,
                        finished_at: new Date().toISOString(),
                        error_msg: "cancelled_by_user"
                      })
                      message.success(
                        t("watchlists:runs.cancelRunSuccess", "Run cancelled")
                      )
                      setRunsLiveAnnouncement(
                        t("watchlists:runs.live.cancelRunSuccess", "Run #{{id}} cancelled.", {
                          id: record.id
                        })
                      )
                    } catch (err) {
                      console.error("Failed to cancel run:", err)
                      setFailedCancelRunIds((prev) => (prev.includes(record.id) ? prev : [...prev, record.id]))
                      message.error(
                        t("watchlists:runs.cancelRunError", "Failed to cancel run")
                      )
                      setRunsLiveAnnouncement(
                        t("watchlists:runs.live.cancelRunError", "Failed to cancel run #{{id}}.", {
                          id: record.id
                        })
                      )
                    } finally {
                      setCancellingRunIds((prev) => prev.filter((id) => id !== record.id))
                    }
                  }}
                />
              </Tooltip>
            )}
          </Space>
        )
      }
    }
  ]
  const resolveColumnKey = (column: ColumnsType<WatchlistRun>[number]): string => {
    if (column.key != null) return String(column.key)
    if ("dataIndex" in column && column.dataIndex != null) {
      return Array.isArray(column.dataIndex)
        ? column.dataIndex.map((entry) => String(entry)).join(".")
        : String(column.dataIndex)
    }
    return ""
  }
  const defaultColumnKeys = new Set(["job", "status", "started_at", "actions"])
  const columns = showAdvancedFilters
    ? allColumns
    : allColumns.filter((column) => defaultColumnKeys.has(resolveColumnKey(column)))

  // Status options for filter
  const statusOptions = useMemo(() => ([
    { value: "pending", label: getRunStatusLabel("pending") },
    { value: "running", label: getRunStatusLabel("running") },
    { value: "completed", label: getRunStatusLabel("completed") },
    { value: "failed", label: getRunStatusLabel("failed") },
    { value: "cancelled", label: getRunStatusLabel("cancelled") }
  ]), [getRunStatusLabel])
  const activeStatusLabel = statusOptions.find((option) => option.value === runsStatusFilter)?.label
  const activeRunsFilterSummary = [
    runsJobFilter
      ? t("watchlists:runs.activeFiltersJob", "Monitor: {{name}}", {
          name: getJobName(Number(runsJobFilter))
        })
      : null,
    activeStatusLabel
      ? t("watchlists:runs.activeFiltersStatus", "Status: {{status}}", {
          status: activeStatusLabel
        })
      : null
  ]
    .filter(Boolean)
    .join(" • ")

  const runsAttention = useMemo(() => {
    const runItems = Array.isArray(runs) ? runs : []
    const failedRuns = runItems.filter(
      (run) => String(run.status || "").toLowerCase() === "failed"
    )
    const stalledEvents = runItems
      .map((run) =>
        resolveStalledRunNotification(
          run,
          Date.now(),
          RUN_STALLED_THRESHOLD_MS,
          t
        )
      )
      .filter((event): event is NonNullable<typeof event> => Boolean(event))

    if (!failedRuns.length && !stalledEvents.length) return null

    const newestFailedRun = [...failedRuns].sort((a, b) => b.id - a.id)[0] || null
    const failedHint = newestFailedRun
      ? getRunFailureHint(newestFailedRun.error_msg, t)
      : null
    const fallbackHint = stalledEvents[0]?.hint || null

    return {
      failedCount: failedRuns.length,
      stalledCount: stalledEvents.length,
      newestFailedRunId: newestFailedRun?.id ?? null,
      description: t(
        "watchlists:runs.attention.description",
        "{{failed}} failed run{{failedPlural}} and {{stalled}} stalled run{{stalledPlural}} need review.",
        {
          failed: failedRuns.length,
          failedPlural: failedRuns.length === 1 ? "" : "s",
          stalled: stalledEvents.length,
          stalledPlural: stalledEvents.length === 1 ? "" : "s"
        }
      ),
      hint: failedHint || fallbackHint
    }
  }, [runs, t])

  const openRunOutputs = useCallback((runId: number, jobId: number) => {
    setOutputsJobFilter(jobId)
    setOutputsRunFilter(runId)
    setActiveTab("outputs")
  }, [setActiveTab, setOutputsJobFilter, setOutputsRunFilter])

  return (
    <div className="space-y-4">
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
        data-testid="watchlists-runs-live-region"
      >
        {runsLiveAnnouncement}
      </div>

      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <Button
            type={showAdvancedFilters ? "default" : "dashed"}
            size="small"
            data-testid="watchlists-runs-advanced-toggle"
            onClick={() => setShowAdvancedFilters((prev) => !prev)}
          >
            {showAdvancedFilters
              ? t("watchlists:runs.hideAdvancedFilters", "Hide advanced filters")
              : t("watchlists:runs.showAdvancedFilters", "Show advanced filters")}
          </Button>
          {!showAdvancedFilters && hasActiveRunsFilters && (
            <>
              <span className="text-sm text-text-muted" data-testid="watchlists-runs-active-filters-summary">
                {t("watchlists:runs.activeFilters", "Active filters")}: {activeRunsFilterSummary}
              </span>
              <Button
                size="small"
                type="text"
                onClick={() => {
                  setRunsJobFilter(null)
                  setRunsStatusFilter(null)
                }}
              >
                {t("common:clear", "Clear")}
              </Button>
            </>
          )}
          {showAdvancedFilters && (
            <>
              <Select
                data-testid="watchlists-runs-job-filter"
                placeholder={t("watchlists:runs.filterByJob", "Filter by monitor")}
                value={runsJobFilter}
                onChange={setRunsJobFilter}
                allowClear
                className="w-48"
                options={jobs.map((j) => ({
                  label: j.name,
                  value: j.id
                }))}
              />
              <Select
                data-testid="watchlists-runs-status-filter"
                placeholder={t("watchlists:runs.filterByStatus", "Filter by status")}
                value={runsStatusFilter}
                onChange={setRunsStatusFilter}
                allowClear
                className="w-36"
                options={statusOptions}
              />
            </>
          )}
          {!showAdvancedFilters && (
            <span className="text-xs text-text-subtle">
              {t("watchlists:runs.metricsHint", "Showing core columns. Use advanced mode for run metrics.")}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {lastRefreshedAt && (
            <Tooltip title={new Date(lastRefreshedAt).toLocaleString()}>
              <span className="text-sm text-text-muted">
                {t("watchlists:runs.lastRefreshed", "Last refreshed {{time}}", {
                  time: formatRelativeTime(lastRefreshedAt, t, { compact: true })
                })}
              </span>
            </Tooltip>
          )}
          <span
            role="status"
            aria-live="polite"
            className="text-sm text-primary flex items-center gap-1"
          >
            {pollingActive && (
              <>
                <span className="relative flex h-2 w-2" aria-hidden="true">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
                </span>
                {t("watchlists:runs.autoRefreshing", "Auto-refreshing")}
              </>
            )}
          </span>
          <Button
            icon={<RefreshCw className="h-4 w-4" />}
            onClick={() => loadRuns()}
            loading={runsLoading}
          >
            {t("common:refresh", "Refresh")}
          </Button>
          <div className="inline-flex items-center">
            <Button
              icon={<Download className="h-4 w-4" />}
              onClick={() => void handleExportRunsCsv()}
              loading={exportingRunsCsv}
              data-testid="runs-csv-export-button"
            >
              {t("watchlists:runs.exportCsv", "Export CSV")} ({activeExportModeLabel})
            </Button>
            <Dropdown
              menu={{
                items: exportModeOptions.map((option) => ({
                  key: option.value,
                  label: option.label,
                  disabled: option.disabled
                })),
                onClick: ({ key }) => {
                  const nextMode = String(key) as RunsCsvTalliesMode
                  setRunsCsvTalliesMode(nextMode)
                  void handleExportRunsCsv(nextMode)
                }
              }}
              trigger={["click"]}
            >
              <Button
                aria-label={t("watchlists:runs.exportOptions", "Export options")}
                icon={<ChevronDown className="h-4 w-4" />}
                loading={exportingRunsCsv}
                data-testid="runs-csv-export-options"
              />
            </Dropdown>
          </div>
        </div>
      </div>

      {/* Description */}
      <div className="text-sm text-text-muted">
        {t("watchlists:runs.description", "View execution history and logs for your watchlist monitors.")}
      </div>

      {runsAttention && (
        <Alert
          type="warning"
          showIcon
          title={t("watchlists:runs.attention.title", "Reliability attention required")}
          description={
            `${runsAttention.description}${runsAttention.hint ? ` ${runsAttention.hint}` : ""}`.trim()
          }
          action={(
            <div className="flex flex-wrap gap-2">
              {runsAttention.newestFailedRunId != null && (
                <Button
                  size="small"
                  onClick={() => openRunDetail(runsAttention.newestFailedRunId as number)}
                >
                  {t("watchlists:runs.attention.viewFailedRun", "View newest failed run")}
                </Button>
              )}
              {runsAttention.failedCount > 0 && runsStatusFilter !== "failed" && (
                <Button
                  size="small"
                  onClick={() => setRunsStatusFilter("failed")}
                >
                  {t("watchlists:runs.attention.filterFailed", "Show failed runs")}
                </Button>
              )}
            </div>
          )}
        />
      )}

      {runsLoadError && (
        <Alert
          type={runsLoadError.severity}
          showIcon
          title={runsLoadError.title}
          description={runsLoadError.description}
          action={(
            <Button
              size="small"
              onClick={() => void loadRuns()}
              loading={runsLoading}
            >
              {t("watchlists:errors.retry", "Retry")}
            </Button>
          )}
        />
      )}

      {/* Table */}
      <Table
        dataSource={Array.isArray(runs) ? runs : []}
        columns={columns}
        rowKey="id"
        aria-label={t("watchlists:runs.tableAria", "Activity runs table")}
        loading={runsLoading}
        pagination={{
          current: runsPage,
          pageSize: runsPageSize,
          total: runsTotal,
          showSizeChanger: true,
          showTotal: (total) =>
            t("watchlists:runs.totalItems", "{{total}} runs", { total }),
          onChange: (page, pageSize) => {
            setRunsPage(page)
            if (pageSize !== runsPageSize) {
              setRunsPageSize(pageSize)
            }
          }
        }}
        size="middle"
        scroll={{ x: 800 }}
      />

      {/* Run Detail Drawer */}
      <RunDetailDrawer
        runId={selectedRunId}
        open={runDetailOpen}
        onClose={closeRunDetail}
      />
    </div>
  )
}
