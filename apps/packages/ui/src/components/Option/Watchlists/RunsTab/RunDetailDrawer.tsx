import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import {
  Button,
  Descriptions,
  Drawer,
  Empty,
  Pagination,
  Spin,
  Switch,
  Table,
  Tabs,
  Tag,
  Tooltip,
  message
} from "antd"
import type { ColumnsType } from "antd/es/table"
import { Download } from "lucide-react"
import { useTranslation } from "react-i18next"
import { LOADING_STATE_LABEL } from "@/design-system"
import { Alert } from "@/components/ui"
import { RecoveryCallout, type RecoveryState } from "@/components/ui/state"
import {
  useCancelWatchlistsQuery,
  useInvalidateWatchlists,
  useIsCurrentWatchlistsAccount,
  useRefreshWatchlistsQuery,
  useSetWatchlistsQueryData,
  useWatchlistsQuery
} from "@/hooks/useWatchlistsQuery"
import { useWatchlistsStore } from "@/store/watchlists"
import {
  cancelWatchlistRun,
  exportRunTalliesCsv,
  fetchWatchlistOutputs,
  fetchWatchlistSources,
  fetchScrapedItems,
  getWatchlistRunDiagnostics,
  getWatchlistRunAudio,
  getRunDetails,
  retryWatchlistRunAudio,
  retryWatchlistRunDelivery,
  triggerWatchlistRun,
  updateScrapedItem
} from "@/services/watchlists"
import {
  buildWatchlistsRunWebSocketUrl,
  parseWatchlistsRunStreamPayload
} from "@/services/watchlists-stream"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { PaginatedResponse, RunDetailResponse, RunStatus, ScrapedItem, WatchlistRunAudioStatus } from "@/types/watchlists"
import { formatRelativeTime } from "@/utils/dateFormatters"
import { safeExternalUrl } from "@/utils/safe-external-url"
import { StatusTag } from "../shared"
import { isWatchlistRunActive, isWatchlistRunTerminal } from "../shared/runStatus"
import { mapWatchlistsError } from "../shared/watchlists-error"
import { classifyRunFailure, getRunFailureHint } from "./run-notifications"
import { usePatchRunLists } from "./usePatchRunLists"
import {
  type AudioArtifactSummary,
  createOutputMetadataLabels,
  getAudioStatusSummary
} from "../OutputsTab/outputMetadata"
import { useWatchlistsViewport } from "../shared/useWatchlistsViewport"
import {
  getFocusableActiveElement,
  restoreFocusToElement
} from "../shared/focus-management"

interface RunDetailDrawerProps {
  runId: number | null
  open: boolean
  onClose: () => void
}

type StreamConnectionState =
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected"
  | "error"

const SOURCE_LOOKUP_LIMIT = 200
const RUN_DETAILS_POLL_INTERVAL_MS = 5_000

/** Maps failure kind → [i18n key, fallback] pairs for the "Common causes" section */
const COMMON_CAUSES_BY_KIND: Record<string, [string, string][]> = {
  auth: [
    ["watchlists:runs.detail.commonCauses.auth1", "Source requires authentication or API key"],
    ["watchlists:runs.detail.commonCauses.auth2", "Credentials expired or permissions changed"]
  ],
  rate_limit: [
    ["watchlists:runs.detail.commonCauses.rateLimit1", "Monitor schedule is too frequent for this source"],
    ["watchlists:runs.detail.commonCauses.rateLimit2", "Source has per-IP rate limits"]
  ],
  timeout: [
    ["watchlists:runs.detail.commonCauses.timeout1", "Source is slow to respond or temporarily unavailable"],
    ["watchlists:runs.detail.commonCauses.timeout2", "Too many concurrent requests — reduce concurrency"]
  ],
  dns: [
    ["watchlists:runs.detail.commonCauses.network1", "Source host is unreachable or has changed"],
    ["watchlists:runs.detail.commonCauses.network2", "Local network or DNS configuration issue"]
  ],
  network: [
    ["watchlists:runs.detail.commonCauses.network1", "Source host is unreachable or has changed"],
    ["watchlists:runs.detail.commonCauses.network2", "Local network or DNS configuration issue"]
  ],
  tls: [
    ["watchlists:runs.detail.commonCauses.tls1", "SSL certificate expired or self-signed"],
    ["watchlists:runs.detail.commonCauses.tls2", "Certificate chain is incomplete"]
  ]
}

const getRecoveryStateForSeverity = (
  severity: ReturnType<typeof mapWatchlistsError>["severity"] | string | null | undefined
): RecoveryState => {
  if (severity === "error") return "error"
  return "degraded"
}

const formatRunStatCount = (value: RunDetailResponse["stats"] extends infer Stats
  ? Stats extends Record<string, infer StatValue>
    ? StatValue
    : unknown
  : unknown): React.ReactNode => {
  if (typeof value === "number" || typeof value === "string") return value
  if (typeof value === "boolean") return value ? "Yes" : "No"
  return 0
}

export const RunDetailDrawer: React.FC<RunDetailDrawerProps> = ({
  runId,
  open,
  onClose
}) => {
  const { t } = useTranslation(["watchlists", "common"])
  const { isConstrained } = useWatchlistsViewport()
  const hostedMode = isHostedTldwDeployment()
  const outputMetadataLabels = useMemo(() => createOutputMetadataLabels(t), [t])
  const [itemsPagination, setItemsPagination] = useState({ runId, page: 1, size: 20 })
  const itemsPage = itemsPagination.runId === runId ? itemsPagination.page : 1
  const itemsPageSize = itemsPagination.size
  const [updatingItemIds, setUpdatingItemIds] = useState<number[]>([])
  const [exportingTalliesCsv, setExportingTalliesCsv] = useState(false)
  const [streamState, setStreamState] = useState<StreamConnectionState>("disconnected")
  const [streamError, setStreamError] = useState<string | null>(null)
  const [lastStreamEventAt, setLastStreamEventAt] = useState<string | null>(null)
  const [streamingEnabled, setStreamingEnabled] = useState(true)
  const [cancelState, setCancelState] = useState<"idle" | "cancelling" | "failed-to-cancel">("idle")
  const [retryingRun, setRetryingRun] = useState(false)
  const [retryingDelivery, setRetryingDelivery] = useState(false)
  const [retryingAudio, setRetryingAudio] = useState(false)
  const [exportingDiagnostics, setExportingDiagnostics] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const translationRef = useRef(t)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectAttemptRef = useRef(0)
  const manuallyClosedRef = useRef(false)
  const currentStatusRef = useRef<string | null>(null)
  const runViewRef = useRef<symbol | null>(null)
  const pollingRunIdRef = useRef<number | null>(null)
  const restoreFocusTargetRef = useRef<HTMLElement | null>(null)
  const wasOpenRef = useRef(false)
  const setActiveTab = useWatchlistsStore((s) => s.setActiveTab)
  const setOutputsJobFilter = useWatchlistsStore((s) => s.setOutputsJobFilter)
  const setOutputsRunFilter = useWatchlistsStore((s) => s.setOutputsRunFilter)
  const openJobForm = useWatchlistsStore((s) => s.openJobForm)

  const invalidateWatchlists = useInvalidateWatchlists()
  const cancelWatchlistsQuery = useCancelWatchlistsQuery()
  const refreshWatchlistsQuery = useRefreshWatchlistsQuery()
  const isCurrentWatchlistsAccount = useIsCurrentWatchlistsAccount()
  const setQueryData = useSetWatchlistsQueryData()
  const patchRunLists = usePatchRunLists()
  const runOpen = open && runId != null
  const hostedLiveUpdates =
    hostedMode && runOpen && streamingEnabled && cancelState !== "cancelling"
  const detailsQuery = useWatchlistsQuery("run-detail", runId, ({ signal }) => getRunDetails(runId!, signal), {
    enabled: runOpen,
    refetchInterval: (query) =>
      hostedLiveUpdates && isWatchlistRunActive(query.state.data?.status)
        ? RUN_DETAILS_POLL_INTERVAL_MS
        : false
  })
  const data = runOpen ? detailsQuery.data : undefined
  const loading = runOpen && detailsQuery.isLoading
  const detailsError = detailsQuery.error ? mapWatchlistsError(detailsQuery.error, {
    t,
    context: t("watchlists:runs.detail.context", "run details"),
    fallbackMessage: t("watchlists:runs.detail.loadError", "Failed to load details"),
    operationLabel: t("watchlists:errors.operation.load", "load")
  }) : null
  const error = hostedMode && data ? null : detailsError

  const sourceParams = { page: 1, size: SOURCE_LOOKUP_LIMIT }
  const { data: sources } = useWatchlistsQuery("sources", sourceParams, ({ signal }) => fetchWatchlistSources(sourceParams, signal), {
    enabled: open
  })
  const sourceNamesById = useMemo(
    () => Object.fromEntries((sources?.items ?? []).map((source) => [source.id, source.name])),
    [sources]
  )
  const outputParams = { run_id: runId ?? undefined, page: 1, size: 1 }
  const linkedOutputsQuery = useWatchlistsQuery("outputs", outputParams, ({ signal }) => fetchWatchlistOutputs(outputParams, signal), {
    enabled: runOpen
  })
  const linkedOutputsLoading = linkedOutputsQuery.isFetching
  const linkedOutputs = linkedOutputsQuery.data
  const linkedOutputCount = linkedOutputs
    ? Math.max(0, linkedOutputs.total ?? linkedOutputs.items?.length ?? 0)
    : null

  const itemsParams = { run_id: runId ?? undefined, page: itemsPage, size: itemsPageSize }
  const itemsQuery = useWatchlistsQuery("items", itemsParams, ({ signal }) => fetchScrapedItems(itemsParams, signal), {
    enabled: runOpen
  })
  const items = itemsQuery.data?.items ?? []
  const itemsTotal = itemsQuery.data?.total ?? 0
  const itemsLoading = itemsQuery.isFetching

  const setData = useCallback((update: (previous: RunDetailResponse | undefined) => RunDetailResponse | undefined) => {
    setQueryData<RunDetailResponse>("run-detail", runId, update)
  }, [runId, setQueryData])

  const audioTaskId = useMemo(() => {
    const stats = data?.stats as Record<string, unknown> | null | undefined
    const value = stats?.audio_briefing_task_id ?? stats?.audio_task_id
    if (typeof value === "string" && value.trim()) return value.trim()
    if (typeof value === "number" && Number.isFinite(value)) return String(value)
    return null
  }, [data?.stats])

  const audioParams = { runId, taskId: audioTaskId }
  const audioQuery = useWatchlistsQuery("run-audio", audioParams, ({ signal }) => getWatchlistRunAudio(runId!, signal), {
    enabled: runOpen && Boolean(audioTaskId)
  })
  const audioStatus = audioQuery.data
  const audioStatusLoading = audioQuery.isLoading
  const audioStatusError = audioQuery.error
    ? audioQuery.error.message || t("watchlists:runs.detail.audioStatusError", "Failed to load audio status")
    : null

  const audioSummary = useMemo(() => {
    const liveSummary = getAudioStatusSummary(audioStatus, outputMetadataLabels)
    if (liveSummary.requested) return liveSummary
    return getAudioStatusSummary(data?.stats, outputMetadataLabels)
  }, [audioStatus, data?.stats, outputMetadataLabels])

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

  const downloadJson = (content: unknown, filename: string): void => {
    const blob = new Blob([JSON.stringify(content, null, 2)], {
      type: "application/json;charset=utf-8"
    })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = filename
    document.body.appendChild(anchor)
    anchor.click()
    document.body.removeChild(anchor)
    URL.revokeObjectURL(url)
  }

  const clearReconnectTimer = () => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
  }

  const closeStream = () => {
    clearReconnectTimer()
    if (wsRef.current && wsRef.current.readyState < WebSocket.CLOSING) {
      wsRef.current.close()
    }
    wsRef.current = null
    reconnectAttemptRef.current = 0
  }

  const appendLogText = (incoming: string) => {
    const MAX_LOG_CHARS = 200_000
    setData((prev) => {
      if (!prev) return prev
      const current = prev.log_text || ""
      const merged = `${current}${incoming}`
      if (merged.length <= MAX_LOG_CHARS) {
        return { ...prev, log_text: merged }
      }
      return {
        ...prev,
        log_text: merged.slice(-MAX_LOG_CHARS),
        truncated: true
      }
    })
  }

  const applyRunSnapshot = (
    run: {
      id: number
      job_id: number
      status: string
      started_at?: string | null
      finished_at?: string | null
    },
    stats: Record<string, number>,
    errorMsg?: string | null
  ) => {
    setData((prev) => {
      if (!prev) {
        return {
          id: run.id,
          job_id: run.job_id,
          status: run.status,
          started_at: run.started_at ?? null,
          finished_at: run.finished_at ?? null,
          stats,
          filter_tallies: null,
          error_msg: errorMsg ?? null,
          log_text: null,
          log_path: null,
          truncated: false,
          filtered_sample: null
        }
      }
      return {
        ...prev,
        id: run.id,
        job_id: run.job_id,
        status: run.status,
        started_at: run.started_at ?? prev.started_at,
        finished_at: run.finished_at ?? prev.finished_at,
        stats,
        error_msg: errorMsg ?? prev.error_msg
      }
    })

    patchRunLists(run.id, {
      status: run.status as RunStatus,
      started_at: run.started_at ?? null,
      finished_at: run.finished_at ?? null,
      stats,
      error_msg: errorMsg ?? null
    })
  }

  useEffect(() => {
    currentStatusRef.current = data?.status || null
  }, [data?.status])

  useEffect(() => {
    translationRef.current = t
  }, [t])

  useLayoutEffect(() => {
    if (open) {
      if (!wasOpenRef.current) {
        restoreFocusTargetRef.current = getFocusableActiveElement()
      }
      wasOpenRef.current = true
      return
    }

    if (wasOpenRef.current) {
      wasOpenRef.current = false
      restoreFocusToElement(restoreFocusTargetRef.current)
    }
  }, [open])

  useEffect(() => {
    runViewRef.current = open && runId != null ? Symbol() : null
    setCancelState("idle")
    setStreamError(null)
    setLastStreamEventAt(null)
    if (!open) {
      setItemsPagination({ runId, page: 1, size: 20 })
      setStreamState("disconnected")
    }
    return () => {
      runViewRef.current = null
    }
  }, [open, runId])

  const pollingRunDetails = hostedLiveUpdates && isWatchlistRunActive(data?.status)
  useEffect(() => {
    if (!pollingRunDetails || runId == null) return
    return () => {
      void cancelWatchlistsQuery("run-detail", runId)
    }
  }, [pollingRunDetails, runId, cancelWatchlistsQuery])

  useEffect(() => {
    const previousPollingRunId = pollingRunIdRef.current
    pollingRunIdRef.current = pollingRunDetails ? runId : null
    if (!hostedLiveUpdates || !data || data.id !== runId) return
    patchRunLists(data.id, {
      status: data.status as RunStatus,
      started_at: data.started_at,
      finished_at: data.finished_at,
      stats: data.stats,
      error_msg: data.error_msg,
      log_path: data.log_path
    })
    if (previousPollingRunId === runId && isWatchlistRunTerminal(data.status)) {
      void refreshWatchlistsQuery("items", { run_id: runId, page: itemsPage, size: itemsPageSize })
    }
  }, [data, hostedLiveUpdates, itemsPage, itemsPageSize, patchRunLists, pollingRunDetails, refreshWatchlistsQuery, runId])

  const itemsError = itemsQuery.error
  useEffect(() => {
    if (itemsError && runOpen) {
      console.error("Failed to fetch run items:", itemsError)
      message.error(translationRef.current("watchlists:runs.detail.itemsError", "Failed to load items"))
    }
  }, [itemsError, runOpen])

  useEffect(() => {
    if (!open || !runId || !streamingEnabled || hostedMode) {
      manuallyClosedRef.current = true
      closeStream()
      return
    }

    let disposed = false

    const connectStream = async () => {
      const attempt = reconnectAttemptRef.current
      setStreamState(attempt > 0 ? "reconnecting" : "connecting")
      setStreamError(null)

      try {
        const config = await tldwClient.getConfig()
        if (disposed) return
        if (!config) {
          setStreamState("error")
          setStreamError(t("watchlists:runs.detail.streamSetupError", "Failed to connect live stream"))
          return
        }
        const wsUrl = buildWatchlistsRunWebSocketUrl(config, runId)
        const ws = new WebSocket(wsUrl)
        wsRef.current = ws

        ws.onopen = () => {
          if (disposed || wsRef.current !== ws) return
          reconnectAttemptRef.current = 0
          setStreamState("connected")
          setStreamError(null)
        }

        ws.onmessage = (event) => {
          if (disposed || wsRef.current !== ws || typeof event.data !== "string") return
          let parsedJson: unknown = null
          try {
            parsedJson = JSON.parse(event.data)
          } catch {
            return
          }

          const streamEvent = parseWatchlistsRunStreamPayload(parsedJson)
          if (!streamEvent) return
          setLastStreamEventAt(new Date().toISOString())

          if (streamEvent.type === "snapshot") {
            applyRunSnapshot(streamEvent.run, streamEvent.stats, streamEvent.error_msg)
            setData((prev) => {
              if (!prev) {
                return {
                  id: streamEvent.run.id,
                  job_id: streamEvent.run.job_id,
                  status: streamEvent.run.status,
                  started_at: streamEvent.run.started_at ?? null,
                  finished_at: streamEvent.run.finished_at ?? null,
                  stats: streamEvent.stats,
                  filter_tallies: null,
                  error_msg: streamEvent.error_msg ?? null,
                  log_text: typeof streamEvent.log_tail === "string" ? streamEvent.log_tail : null,
                  log_path: null,
                  truncated: Boolean(streamEvent.log_truncated),
                  filtered_sample: null
                }
              }
              return {
                ...prev,
                log_text: typeof streamEvent.log_tail === "string" ? streamEvent.log_tail : prev.log_text,
                truncated: Boolean(streamEvent.log_truncated || prev.truncated)
              }
            })
            return
          }

          if (streamEvent.type === "run_update") {
            applyRunSnapshot(streamEvent.run, streamEvent.stats, streamEvent.error_msg)
            return
          }

          if (streamEvent.type === "log") {
            appendLogText(streamEvent.text)
            return
          }

          if (streamEvent.type === "complete") {
            const finalStatus = streamEvent.status || currentStatusRef.current || "completed"
            setData((prev) => (prev ? { ...prev, status: finalStatus } : prev))
            patchRunLists(runId, { status: finalStatus as RunStatus })
            setStreamState("disconnected")
            manuallyClosedRef.current = true
            closeStream()
          }
        }

        ws.onerror = () => {
          if (disposed || wsRef.current !== ws) return
          setStreamState("error")
          setStreamError(t("watchlists:runs.detail.streamError", "Live stream error"))
        }

        ws.onclose = () => {
          if (disposed || wsRef.current !== ws) return
          wsRef.current = null

          if (manuallyClosedRef.current) {
            setStreamState("disconnected")
            return
          }

          if (isWatchlistRunTerminal(currentStatusRef.current)) {
            setStreamState("disconnected")
            return
          }

          const nextAttempt = reconnectAttemptRef.current + 1
          reconnectAttemptRef.current = nextAttempt
          setStreamState("reconnecting")
          const delayMs = Math.min(1000 * 2 ** Math.max(nextAttempt - 1, 0), 10000)
          clearReconnectTimer()
          reconnectTimerRef.current = setTimeout(() => {
            if (!disposed) void connectStream()
          }, delayMs)
        }
      } catch (err) {
        if (disposed) return
        setStreamState("error")
        setStreamError(
          err instanceof Error
            ? err.message
            : t("watchlists:runs.detail.streamSetupError", "Failed to connect live stream")
        )
      }
    }

    manuallyClosedRef.current = false
    void connectStream()

    return () => {
      disposed = true
      manuallyClosedRef.current = true
      closeStream()
      setStreamState("disconnected")
    }
  }, [open, runId, streamingEnabled, hostedMode, t, patchRunLists, setData])

  // Calculate duration
  const calculateDuration = (): string => {
    if (!data?.started_at) return "-"
    const start = new Date(data.started_at)
    const end = data.finished_at ? new Date(data.finished_at) : new Date()
    const durationMs = end.getTime() - start.getTime()

    if (durationMs < 1000) {
      return t(
        "watchlists:runs.detail.duration.lessThanSecond",
        "<1 second"
      )
    }

    if (durationMs < 60000) {
      return t("watchlists:runs.detail.duration.seconds", "{{count}} seconds", {
        count: Math.max(1, Math.round(durationMs / 1000))
      })
    }

    if (durationMs < 3600000) {
      return t("watchlists:runs.detail.duration.minutes", "{{count}} minutes", {
        count: Math.max(1, Math.round(durationMs / 60000))
      })
    }

    return t("watchlists:runs.detail.duration.hours", "{{count}} hours", {
      count: Number((durationMs / 3600000).toFixed(1))
    })
  }

  const handleToggleReviewed = async (item: ScrapedItem, reviewed: boolean) => {
    if (updatingItemIds.includes(item.id)) return
    setUpdatingItemIds((prev) => [...prev, item.id])
    try {
      const updated = await updateScrapedItem(item.id, { reviewed })
      setQueryData<PaginatedResponse<ScrapedItem>>("items", itemsParams, (page) => page && {
        ...page,
        items: page.items.map((entry) => entry.id === item.id ? { ...entry, reviewed: updated.reviewed } : entry)
      })
      void invalidateWatchlists("items", "overview")
    } catch (err) {
      console.error("Failed to update item:", err)
      message.error(t("watchlists:runs.detail.itemsUpdateError", "Failed to update item"))
    } finally {
      setUpdatingItemIds((prev) => prev.filter((id) => id !== item.id))
    }
  }

  const handleExportTalliesCsv = async () => {
    if (!runId) return
    try {
      setExportingTalliesCsv(true)
      const csv = await exportRunTalliesCsv(runId)
      if (!csv || !csv.trim()) {
        message.warning(t("watchlists:runs.detail.talliesEmpty", "No tallies available to export"))
        return
      }
      downloadCsv(csv, `watchlists_run_${runId}_tallies_${Date.now()}.csv`)
      message.success(t("watchlists:runs.detail.talliesExported", "Tallies CSV exported"))
    } catch (err) {
      console.error("Failed to export run tallies CSV:", err)
      message.error(t("watchlists:runs.detail.talliesExportError", "Failed to export tallies CSV"))
    } finally {
      setExportingTalliesCsv(false)
    }
  }

  const handleCancelRun = async () => {
    const runView = runViewRef.current
    if (!runView || !runId || data?.id !== runId || cancelState === "cancelling") return
    const isCurrentRun = () => runViewRef.current === runView && isCurrentWatchlistsAccount()
    setCancelState("cancelling")
    try {
      const result = await cancelWatchlistRun(runId)
      if (!result?.cancelled) {
        if (!isCurrentRun()) return
        setCancelState("failed-to-cancel")
        message.error(t("watchlists:runs.cancelRunError", "Failed to cancel run"))
        return
      }
      const finishedAt = new Date().toISOString()
      if (hostedMode) await cancelWatchlistsQuery("run-detail", runId)
      setData((prev) =>
        prev
          ? {
              ...prev,
              status: "cancelled",
              finished_at: prev.finished_at || finishedAt,
              error_msg: prev.error_msg || "cancelled_by_user"
            }
          : prev
      )
      patchRunLists(runId, {
        status: "cancelled",
        finished_at: finishedAt,
        error_msg: "cancelled_by_user"
      })
      void invalidateWatchlists("runs", "jobs", "overview")
      if (!isCurrentRun()) return
      if (hostedMode) {
        void refreshWatchlistsQuery("run-detail", runId)
        void refreshWatchlistsQuery("items", itemsParams)
      }
      setCancelState("idle")
      message.success(t("watchlists:runs.cancelRunSuccess", "Run cancelled"))
      manuallyClosedRef.current = true
      closeStream()
      setStreamState("disconnected")
    } catch (err) {
      console.error("Failed to cancel run:", err)
      if (!isCurrentRun()) return
      setCancelState("failed-to-cancel")
      message.error(t("watchlists:runs.cancelRunError", "Failed to cancel run"))
    }
  }

  const itemColumns: ColumnsType<ScrapedItem> = [
    {
      title: t("watchlists:runs.detail.itemsColumns.title", "Title"),
      dataIndex: "title",
      key: "title",
      ellipsis: true,
      render: (title: string | null, record) => {
        const safeUrl = safeExternalUrl(record.url)
        return (
        <div className="space-y-1">
          <div className="font-medium">
            {safeUrl ? (
              <a
                href={safeUrl}
                target="_blank"
                rel="noreferrer"
                className="text-primary hover:underline"
              >
                {title || safeUrl}
              </a>
            ) : (
              title || t("watchlists:runs.detail.itemsUntitled", "Untitled")
            )}
          </div>
          {record.summary && (
            <div className="text-xs text-text-muted line-clamp-2">{record.summary}</div>
          )}
        </div>
        )
      }
    },
    {
      title: t("watchlists:runs.detail.itemsColumns.status", "Status"),
      dataIndex: "status",
      key: "status",
      width: 110,
      render: (status: string) => {
        const normalized = String(status || "").toLowerCase()
        const label =
          normalized === "ingested"
            ? t("watchlists:runs.detail.itemsStatusInBriefing", "Included in briefing")
            : normalized === "filtered"
              ? t("watchlists:runs.detail.itemsStatusFilteredOut", "Excluded from briefing")
              : status
        return (
          <Tag color={normalized === "ingested" ? "green" : "default"}>
            {label}
          </Tag>
        )
      }
    },
    {
      title: t("watchlists:runs.detail.itemsColumns.reviewed", "Reviewed"),
      dataIndex: "reviewed",
      key: "reviewed",
      width: 110,
      render: (_: boolean, record) => (
        <Switch
          checked={record.reviewed}
          onChange={(checked) => handleToggleReviewed(record, checked)}
          loading={updatingItemIds.includes(record.id)}
          size="small"
        />
      )
    },
    {
      title: t("watchlists:runs.detail.itemsColumns.source", "Source"),
      dataIndex: "source_id",
      key: "source_id",
      width: 180,
      render: (sourceId: number) => {
        const sourceReference = t(
          "watchlists:runs.detail.itemsSourceReference",
          "#{{id}}",
          { id: sourceId }
        )
        const sourceName = sourceNamesById[sourceId]
        if (!sourceName) return sourceReference

        return (
          <Tooltip title={sourceReference}>
            <span className="text-sm text-text-muted">{sourceName}</span>
          </Tooltip>
        )
      }
    },
    {
      title: t("watchlists:runs.detail.itemsColumns.published", "Published"),
      dataIndex: "published_at",
      key: "published_at",
      width: 150,
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
      title: t("watchlists:runs.detail.itemsColumns.created", "Ingested"),
      dataIndex: "created_at",
      key: "created_at",
      width: 150,
      render: (date: string) => (
        <span className="text-sm text-text-muted">
          {formatRelativeTime(date, t)}
        </span>
      )
    }
  ]

  const getItemTitle = (item: ScrapedItem): string =>
    item.title || item.url || t("watchlists:runs.detail.itemsUntitled", "Untitled")

  const renderItemSource = (sourceId: number) => {
    const sourceReference = t(
      "watchlists:runs.detail.itemsSourceReference",
      "#{{id}}",
      { id: sourceId }
    )
    const sourceName = sourceNamesById[sourceId]
    if (!sourceName) return sourceReference
    return sourceName
  }

  const renderItemStatus = (item: ScrapedItem) => {
    const normalized = String(item.status || "").toLowerCase()
    const label =
      normalized === "ingested"
        ? t("watchlists:runs.detail.itemsStatusInBriefing", "Included in briefing")
        : normalized === "filtered"
          ? t("watchlists:runs.detail.itemsStatusFilteredOut", "Excluded from briefing")
          : item.status
    return (
      <Tag color={normalized === "ingested" ? "green" : "default"}>
        {label}
      </Tag>
    )
  }

  const renderConstrainedItems = () => (
    <div className="space-y-3" data-testid="watchlists-run-items-constrained-list">
      {items.map((item) => {
        const title = getItemTitle(item)
        const itemSafeUrl = safeExternalUrl(item.url)
        return (
          <article
            key={item.id}
            className="rounded-lg border border-border bg-surface p-3"
            data-testid={`watchlists-run-item-card-${item.id}`}
          >
            <div className="space-y-1">
              <div className="font-medium text-text">{title}</div>
              {item.summary && (
                <div className="text-xs text-text-muted line-clamp-2">{item.summary}</div>
              )}
            </div>

            <div className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:runs.detail.itemsColumns.status", "Status")}
                </div>
                {renderItemStatus(item)}
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:runs.detail.itemsColumns.source", "Source")}
                </div>
                <span className="text-text-muted">{renderItemSource(item.source_id)}</span>
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:runs.detail.itemsColumns.published", "Published")}
                </div>
                <span className="text-text-muted">
                  {item.published_at ? formatRelativeTime(item.published_at, t) : "-"}
                </span>
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:runs.detail.itemsColumns.created", "Ingested")}
                </div>
                <span className="text-text-muted">
                  {item.created_at ? formatRelativeTime(item.created_at, t) : "-"}
                </span>
              </div>
            </div>

            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <span className="inline-flex items-center gap-2">
                <Switch
                  checked={item.reviewed}
                  onChange={(checked) => handleToggleReviewed(item, checked)}
                  loading={updatingItemIds.includes(item.id)}
                  size="small"
                  aria-label={t("watchlists:runs.detail.toggleReviewedAria", "Toggle reviewed for {{title}}", {
                    title
                  })}
                />
                <span className="text-xs text-text-muted">
                  {item.reviewed
                    ? t("watchlists:runs.detail.reviewed", "Reviewed")
                    : t("watchlists:runs.detail.needsReview", "Needs review")}
                </span>
              </span>
              {itemSafeUrl ? (
                <a href={itemSafeUrl} target="_blank" rel="noreferrer" className="text-sm text-primary hover:underline">
                  {t("watchlists:runs.detail.openSource", "Open source")}
                </a>
              ) : null}
            </div>
          </article>
        )
      })}
      <div className="text-xs text-text-subtle">
        {t("watchlists:runs.detail.itemsTotal", "{{total}} items", { total: itemsTotal })}
      </div>
      {itemsTotal > itemsPageSize && (
        <Pagination
          current={itemsPage}
          pageSize={itemsPageSize}
          total={itemsTotal}
          showSizeChanger
          showTotal={(total) => t("watchlists:runs.detail.itemsTotal", "{{total}} items", { total })}
          onChange={(page, size) => setItemsPagination({ runId, page, size })}
        />
      )}
    </div>
  )

  const streamStateColorMap: Record<StreamConnectionState, string> = {
    connecting: "blue",
    connected: "green",
    reconnecting: "gold",
    disconnected: "default",
    error: "red"
  }

  const streamStateLabelMap: Record<StreamConnectionState, string> = {
    connecting: t("watchlists:runs.detail.streamState.connecting", "Connecting"),
    connected: t("watchlists:runs.detail.streamState.connected", "Connected"),
    reconnecting: t("watchlists:runs.detail.streamState.reconnecting", "Reconnecting"),
    disconnected: t("watchlists:runs.detail.streamState.disconnected", "Disconnected"),
    error: t("watchlists:runs.detail.streamState.error", "Error")
  }

  const failureKind = useMemo(
    () => classifyRunFailure(data?.error_msg),
    [data?.error_msg]
  )

  const remediationHint = useMemo(
    () => getRunFailureHint(data?.error_msg, t),
    [data?.error_msg, t]
  )

  const showSourceRecoveryAction = failureKind === "auth" ||
    failureKind === "timeout" ||
    failureKind === "dns" ||
    failureKind === "tls" ||
    failureKind === "network"

  const handleRetryRun = async () => {
    if (!data?.job_id || retryingRun) return
    setRetryingRun(true)
    try {
      const rerun = await triggerWatchlistRun(data.job_id)
      void invalidateWatchlists("runs", "jobs", "overview")
      message.success(
        t("watchlists:runs.detail.retryTriggered", "Retry started as run #{{id}}.", {
          id: rerun.id
        })
      )
      setActiveTab("runs")
      onClose()
    } catch (err) {
      console.error("Failed to trigger retry run:", err)
      message.error(t("watchlists:runs.detail.retryError", "Failed to retry run"))
    } finally {
      setRetryingRun(false)
    }
  }

  const handleRetryDelivery = async () => {
    if (!runId || retryingDelivery) return
    setRetryingDelivery(true)
    try {
      const result = await retryWatchlistRunDelivery(runId)
      if (!result.retried) {
        message.warning(
          result.message || t("watchlists:runs.detail.deliveryRetrySkipped", "Delivery retry was not started.")
        )
        return
      }
      message.success(
        t("watchlists:runs.detail.deliveryRetrySuccess", "Delivery retry completed for output #{{id}}.", {
          id: result.output_id ?? "-"
        })
      )
      void invalidateWatchlists("run-detail", "outputs", "overview")
    } catch (err) {
      console.error("Failed to retry delivery:", err)
      message.error(t("watchlists:runs.detail.deliveryRetryError", "Failed to retry delivery"))
    } finally {
      setRetryingDelivery(false)
    }
  }

  const handleRetryAudio = async () => {
    if (!runId || retryingAudio) return
    setRetryingAudio(true)
    try {
      const result = await retryWatchlistRunAudio(runId)
      if (!result.retried) {
        message.warning(
          result.message || t("watchlists:runs.detail.audioRetrySkipped", "Audio retry was not started.")
        )
        return
      }
      if (result.task_id) {
        const taskId = result.task_id
        const pending: WatchlistRunAudioStatus = {
          run_id: runId,
          task_id: taskId,
          status: "pending",
          audio_uri: null,
          download_url: null,
          error: null
        }
        setQueryData<WatchlistRunAudioStatus>("run-audio", audioParams, () => pending)
        setQueryData<WatchlistRunAudioStatus>("run-audio", { runId, taskId }, () => pending)
      } else {
        void invalidateWatchlists("run-audio")
      }
      message.success(
        t("watchlists:runs.detail.audioRetrySuccess", "Audio retry queued.")
      )
      void invalidateWatchlists("run-detail", "outputs", "overview")
    } catch (err) {
      console.error("Failed to retry audio:", err)
      message.error(t("watchlists:runs.detail.audioRetryError", "Failed to retry audio"))
    } finally {
      setRetryingAudio(false)
    }
  }

  const handleDownloadDiagnostics = async () => {
    if (!runId || exportingDiagnostics) return
    setExportingDiagnostics(true)
    try {
      const diagnostics = await getWatchlistRunDiagnostics(runId)
      downloadJson(diagnostics, `watchlists_run_${runId}_diagnostics_${Date.now()}.json`)
      message.success(t("watchlists:runs.detail.diagnosticsExported", "Diagnostics downloaded"))
    } catch (err) {
      console.error("Failed to download run diagnostics:", err)
      message.error(t("watchlists:runs.detail.diagnosticsExportError", "Failed to download diagnostics"))
    } finally {
      setExportingDiagnostics(false)
    }
  }

  const handleEditMonitor = () => {
    if (!data?.job_id) return
    setActiveTab("jobs")
    openJobForm(data.job_id)
    onClose()
  }

  const handleOpenSources = () => {
    setActiveTab("sources")
    onClose()
  }

  const handleOpenOutputs = () => {
    if (!data) return
    setOutputsJobFilter(data.job_id)
    setOutputsRunFilter(data.id)
    setActiveTab("outputs")
    onClose()
  }

  const renderAudioStatusPanel = () => {
    const displayAudioTaskId = audioStatus?.task_id || audioTaskId
    if (!displayAudioTaskId && !audioSummary.requested) return null

    const hasFinalAudio = Boolean(audioSummary.downloadUrl || audioSummary.finalArtifact)
    const hasAudioArtifacts = Boolean(
      audioSummary.scriptArtifact ||
        audioSummary.speakerArtifacts.length > 0 ||
        audioSummary.finalArtifact
    )
    const statusText = audioStatusLoading
      ? t("watchlists:runs.detail.audioStatusLoading", LOADING_STATE_LABEL)
      : audioStatusError
        ? t("watchlists:runs.detail.audioStatusUnavailable", "Unknown")
        : audioSummary.statusLabel
    const renderArtifact = (artifact: AudioArtifactSummary, key: string) => (
      <div key={key}>
        <div className="font-medium text-text">{artifact.label}</div>
        {artifact.displayName ? <div>{artifact.displayName}</div> : null}
        {artifact.downloadUrl ? (
          <a
            href={artifact.downloadUrl}
            target="_blank"
            rel="noreferrer"
            className="text-primary hover:underline"
            aria-label={t("watchlists:runs.detail.openAudioArtifactAria", "Open {{label}}", {
              label: artifact.label
            })}
          >
            {t("watchlists:runs.detail.openAudioArtifact", "Open")}
          </a>
        ) : null}
      </div>
    )

    return (
      <div
        className="rounded-lg border border-border bg-surface p-3 space-y-2"
        data-testid="watchlists-run-audio-status"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="text-sm font-medium text-text">
              {t("watchlists:runs.detail.audioStatusTitle", "Audio briefing")}
            </div>
            <div className="text-xs text-text-muted">
              {displayAudioTaskId
                ? t("watchlists:runs.detail.audioTask", "Task {{taskId}}", {
                    taskId: displayAudioTaskId
                  })
                : t("watchlists:runs.detail.audioNoTask", "No audio task created")}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Tag color={audioStatusError ? "red" : audioSummary.statusColor}>
              {statusText}
            </Tag>
            {audioSummary.stale && (
              <Tag color="gold">{t("watchlists:runs.detail.audioStale", "Stale")}</Tag>
            )}
          </div>
        </div>

        {audioSummary.supersededBy && (
          <div className="text-xs text-warning">
            {t("watchlists:runs.detail.audioSupersededBy", "Superseded by {{requestId}}", {
              requestId: audioSummary.supersededBy
            })}
          </div>
        )}

        {audioStatusError ? (
          <div className="text-xs text-danger">{audioStatusError}</div>
        ) : audioSummary.fallbackReason ? (
          audioSummary.status === "fallback" ? (
            <div className="text-xs text-warning">
              {t("watchlists:runs.detail.audioFallback", "Fallback: {{reason}}", {
                reason: audioSummary.fallbackReason
              })}
            </div>
          ) : (
            <div className="text-xs text-warning">
              {t("watchlists:runs.detail.audioReason", "Reason: {{reason}}", {
                reason: audioSummary.fallbackReason
              })}
            </div>
          )
        ) : null}

        {hasFinalAudio && (
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="font-medium text-text">
              {t("watchlists:runs.detail.audioFinalAvailable", "Final audio available")}
            </span>
            {audioSummary.downloadUrl && (
              <a
                href={audioSummary.downloadUrl}
                target="_blank"
                rel="noreferrer"
                className="text-primary hover:underline"
              >
                {t("watchlists:runs.detail.audioDownloadFinal", "Download final audio")}
              </a>
            )}
          </div>
        )}

        {hasAudioArtifacts && (
          <div className="space-y-2" data-testid="watchlists-run-audio-artifacts">
            <div className="text-sm font-medium text-text">
              {t("watchlists:runs.detail.audioArtifactsLabel", "Audio artifacts")}
            </div>
            <div className="grid gap-2 text-xs text-text-muted sm:grid-cols-2">
              {audioSummary.scriptArtifact && (
                renderArtifact(audioSummary.scriptArtifact, "script")
              )}
              {audioSummary.speakerArtifacts.map((artifact, index) => (
                renderArtifact(artifact, `${artifact.speakerId || artifact.label}-${index}`)
              ))}
              {audioSummary.finalArtifact && (
                renderArtifact(audioSummary.finalArtifact, "final")
              )}
            </div>
          </div>
        )}

        {(displayAudioTaskId ||
          audioSummary.status === "queue_unavailable" ||
          audioSummary.status === "enqueue_failed" ||
          audioSummary.status === "failed") && (
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="small"
              onClick={handleRetryAudio}
              loading={retryingAudio}
            >
              {t("watchlists:runs.detail.retryAudio", "Retry audio")}
            </Button>
          </div>
        )}
      </div>
    )
  }

  const tabItems = [
    {
      key: "stats",
      label: t("watchlists:runs.detail.stats", "Statistics"),
      children: (
        <div className="space-y-4">
          {data && (
            <Alert
              variant="info"
              title={t("watchlists:runs.detail.linkageTitle", "Run linkage")}
            >
              <div className="space-y-2">
                <p>
                  {t(
                    "watchlists:runs.detail.linkageDescription",
                    "Monitor #{{jobId}} produced {{count}} report{{plural}} for this run.",
                    {
                      jobId: data.job_id,
                      count: linkedOutputCount ?? 0,
                      plural: linkedOutputCount === 1 ? "" : "s"
                    }
                  )}
                </p>
                <div className="flex flex-wrap gap-2">
                  <Button size="small" onClick={handleEditMonitor}>
                    {t("watchlists:runs.detail.openMonitor", "Open monitor")}
                  </Button>
                  <Button
                    size="small"
                    type="primary"
                    loading={linkedOutputsLoading}
                    onClick={handleOpenOutputs}
                  >
                    {t("watchlists:runs.detail.openRunOutputs", "Open reports for this run")}
                  </Button>
                  <Button
                    size="small"
                    loading={retryingDelivery}
                    disabled={(linkedOutputCount ?? 0) <= 0}
                    onClick={handleRetryDelivery}
                  >
                    {t("watchlists:runs.detail.retryDelivery", "Retry delivery")}
                  </Button>
                  <Button
                    size="small"
                    icon={<Download className="h-3.5 w-3.5" />}
                    loading={exportingDiagnostics}
                    onClick={handleDownloadDiagnostics}
                  >
                    {t("watchlists:runs.detail.downloadDiagnostics", "Download diagnostics")}
                  </Button>
                </div>
              </div>
            </Alert>
          )}

          {renderAudioStatusPanel()}

          {data && (
            <Descriptions column={2} size="small" bordered>
              <Descriptions.Item label={t("watchlists:runs.detail.statsLabels.status", "Status")}>
                <StatusTag status={data.status} />
              </Descriptions.Item>
              <Descriptions.Item label={t("watchlists:runs.detail.statsLabels.duration", "Duration")}>
                {calculateDuration()}
              </Descriptions.Item>
              <Descriptions.Item label={t("watchlists:runs.detail.statsLabels.started", "Started")}>
                {data.started_at ? formatRelativeTime(data.started_at, t) : "-"}
              </Descriptions.Item>
              <Descriptions.Item label={t("watchlists:runs.detail.statsLabels.finished", "Finished")}>
                {data.finished_at ? formatRelativeTime(data.finished_at, t) : "-"}
              </Descriptions.Item>
              <Descriptions.Item label={t("watchlists:runs.detail.statsLabels.itemsFound", "Items Found")}>
                {formatRunStatCount(data.stats?.items_found)}
              </Descriptions.Item>
              <Descriptions.Item label={t("watchlists:runs.detail.statsLabels.itemsIngested", "Items Ingested")}>
                {formatRunStatCount(data.stats?.items_ingested)}
              </Descriptions.Item>
              <Descriptions.Item label={t("watchlists:runs.detail.statsLabels.itemsFiltered", "Items Filtered")}>
                {formatRunStatCount(data.stats?.items_filtered)}
              </Descriptions.Item>
              <Descriptions.Item label={t("watchlists:runs.detail.statsLabels.errors", "Errors")}>
                {formatRunStatCount(data.stats?.items_errored)}
              </Descriptions.Item>
              <Descriptions.Item label={t("watchlists:runs.detail.statsLabels.monitor", "Monitor")}>
                <Button size="small" type="link" className="px-0" onClick={handleEditMonitor}>
                  {t("watchlists:runs.detail.openMonitor", "Open monitor settings")}
                </Button>
              </Descriptions.Item>
            </Descriptions>
          )}

          {data?.error_msg && (
            <div className="mt-4 space-y-3">
              <RecoveryCallout
                state="degraded"
                title={t("watchlists:runs.detail.remediationTitle", "Suggested recovery steps")}
                message={remediationHint || t(
                  "watchlists:runs.detail.remediationFallback",
                  "Open logs and adjust source or monitor settings, then retry."
                )}
                primaryAction={{
                  label: t("watchlists:runs.detail.retryRun", "Retry run"),
                  onClick: handleRetryRun,
                  loading: retryingRun
                }}
                secondaryActions={[
                  {
                    label: t("watchlists:runs.detail.editMonitor", "Edit monitor schedule"),
                    onClick: handleEditMonitor
                  },
                  ...(showSourceRecoveryAction
                    ? [
                        {
                          label: t("watchlists:runs.detail.openSources", "Review source settings"),
                          onClick: handleOpenSources
                        }
                      ]
                    : [])
                ]}
              />
              <div className="p-3 bg-danger/10 border border-danger/30 rounded text-sm text-danger font-mono">
                {data.error_msg}
              </div>
              {/* Common causes by failure kind */}
              {failureKind && failureKind !== "unknown" && (
                <Alert
                  variant="info"
                  title={t("watchlists:runs.detail.commonCausesTitle", "Common causes")}
                >
                  <ul className="list-disc pl-4 text-sm space-y-1">
                    {(COMMON_CAUSES_BY_KIND[failureKind] ?? []).map(([key, fallback]) => (
                      <li key={key}>{t(key, fallback)}</li>
                    ))}
                  </ul>
                </Alert>
              )}
            </div>
          )}

          {Array.isArray(data?.filtered_sample) && data.filtered_sample.length > 0 && (
            <div className="mt-4 space-y-2">
              <div className="text-sm font-medium">
                {t("watchlists:runs.detail.filteredSampleTitle", "Filtered item sample")}
              </div>
              <Alert
                variant="info"
                title={t(
                  "watchlists:runs.detail.filteredSampleSummary",
                  "Showing {{count}} recently filtered item{{plural}} for quick diagnosis.",
                  {
                    count: data.filtered_sample.length,
                    plural: data.filtered_sample.length === 1 ? "" : "s"
                  }
                )}
              >
                {t(
                  "watchlists:runs.detail.filteredSampleHelp",
                  "Use this sample with filter tallies below to tune include/exclude rules."
                )}
              </Alert>
              <div className="space-y-2">
                {data.filtered_sample.map((sample, index) => {
                  const record = typeof sample === "object" && sample !== null
                    ? (sample as Record<string, unknown>)
                    : {}
                  const title =
                    typeof record.title === "string" && record.title.trim().length > 0
                      ? record.title
                      : typeof record.url === "string" && record.url.trim().length > 0
                        ? record.url
                        : t("watchlists:runs.detail.itemsUntitled", "Untitled")
                  const statusText =
                    typeof record.status === "string" && record.status.trim().length > 0
                      ? record.status
                      : "filtered"
                  return (
                    <div
                      key={String(record.id ?? `${index}`)}
                      className="rounded border border-border px-3 py-2 text-sm"
                    >
                      <div className="font-medium">{title}</div>
                      <div className="text-xs text-text-muted">
                        {t("watchlists:runs.detail.filteredSampleStatus", "Status: {{status}}", {
                          status: statusText
                        })}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {data?.filter_tallies && Object.keys(data.filter_tallies).length > 0 && (
            <div className="mt-4">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-sm font-medium">
                  {t("watchlists:runs.detail.filterMatches", "Filter Matches")}
                </div>
                <Button
                  size="small"
                  icon={<Download className="h-3.5 w-3.5" />}
                  onClick={handleExportTalliesCsv}
                  loading={exportingTalliesCsv}
                >
                  {t("watchlists:runs.detail.downloadTalliesCsv", "Download Tallies CSV")}
                </Button>
              </div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(data.filter_tallies).map(([filter, count]) => (
                  <Tag key={filter}>
                    {filter}: {count}
                  </Tag>
                ))}
              </div>
            </div>
          )}
        </div>
      )
    },
    {
      key: "logs",
      label: t("watchlists:runs.detail.logs", "Logs"),
      children: (
        <div>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              {hostedMode ? (
                hostedLiveUpdates && !detailsError && isWatchlistRunActive(data?.status) && (
                  <Tag color="green">{t("watchlists:runs.autoRefreshing", "Auto-refreshing")}</Tag>
                )
              ) : (
                <Tag color={streamStateColorMap[streamState]}>
                  {t("watchlists:runs.detail.stream", "Live stream")}: {streamStateLabelMap[streamState]}
                </Tag>
              )}
              {lastStreamEventAt && (
                <span className="text-xs text-text-muted">
                  {t("watchlists:runs.detail.lastStreamEvent", "Last event {{time}}", {
                    time: formatRelativeTime(lastStreamEventAt, t, { compact: true })
                  })}
                </span>
              )}
            </div>
            <Switch
              checked={streamingEnabled}
              size="small"
              onChange={setStreamingEnabled}
              checkedChildren={t("watchlists:runs.detail.liveOn", "Live")}
              unCheckedChildren={t("watchlists:runs.detail.liveOff", "Off")}
            />
          </div>
          {streamError && (
            <Alert
              variant="error"
              className="mb-3"
              title={t("watchlists:runs.detail.streamErrorTitle", "Stream error")}
            >
              {streamError}
            </Alert>
          )}
          {data?.truncated && (
            <Alert
              variant="warning"
              className="mb-3"
              title={t("watchlists:runs.detail.logsTruncated", "Logs truncated")}
            >
              {t("watchlists:runs.detail.logsTruncatedDesc", "Showing the most recent log output.")}
            </Alert>
          )}
          {data?.log_text ? (
            <pre className="bg-bg text-text p-4 rounded-lg font-mono text-xs max-h-96 overflow-auto whitespace-pre-wrap border border-border">
              {data.log_text}
            </pre>
          ) : data?.log_path ? (
            <div className="text-sm text-text-muted">
              {t("watchlists:runs.detail.logsPath", "Logs stored at {{path}}", { path: data.log_path })}
            </div>
          ) : (
            <Empty
              description={t("watchlists:runs.detail.noLogs", "No logs available")}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          )}
        </div>
      )
    },
    {
      key: "items",
      label: t("watchlists:runs.detail.items", "Scraped Items"),
      children: (
        <div>
          {items.length === 0 && !itemsLoading ? (
            <Empty
              description={t("watchlists:runs.detail.noItems", "No items found")}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          ) : (
            isConstrained ? (
              renderConstrainedItems()
            ) : (
              <Table
                dataSource={items}
                columns={itemColumns}
                rowKey="id"
                aria-label={t("watchlists:runs.detail.itemsTableAria", "Run items table")}
                loading={itemsLoading}
                pagination={{
                  current: itemsPage,
                  pageSize: itemsPageSize,
                  total: itemsTotal,
                  showSizeChanger: true,
                  onChange: (page, size) => setItemsPagination({ runId, page, size })
                }}
                size="small"
              />
            )
          )}
        </div>
      )
    }
  ]

  const runStatus = data ? String(data.status || "").toLowerCase() : ""

  return (
    <Drawer
      title={t("watchlists:runs.detail.title", "Run Details")}
      extra={
        data && ["running", "pending", "queued"].includes(runStatus) ? (
          <Button
            danger
            size="small"
            loading={cancelState === "cancelling"}
            aria-label={
              cancelState === "cancelling"
                ? t("watchlists:runs.cancelling", "Cancelling...")
                : cancelState === "failed-to-cancel"
                  ? t("watchlists:runs.cancelFailedRetry", "Cancel failed. Retry.")
                  : t("watchlists:runs.cancelRun", "Cancel run")
            }
            onClick={handleCancelRun}
          >
            {cancelState === "cancelling"
              ? t("watchlists:runs.cancelling", "Cancelling...")
              : cancelState === "failed-to-cancel"
                ? t("watchlists:runs.cancelFailedRetry", "Cancel failed. Retry.")
                : t("watchlists:runs.cancelRun", "Cancel run")}
          </Button>
        ) : data && runStatus === "failed" && !data.error_msg ? (
          <Button
            size="small"
            type="primary"
            loading={retryingRun}
            aria-label={t("watchlists:runs.detail.retryRun", "Retry run")}
            onClick={handleRetryRun}
          >
            {t("watchlists:runs.detail.retryRun", "Retry run")}
          </Button>
        ) : undefined
      }
      placement="right"
      onClose={onClose}
      open={open}
      styles={{ wrapper: { width: isConstrained ? "100vw" : 600 } }}
    >
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Spin size="large" />
        </div>
      ) : error ? (
        <RecoveryCallout
          state={getRecoveryStateForSeverity(error.severity)}
          title={error.title}
          message={error.description}
          primaryAction={{
            label: t("watchlists:errors.retry", "Retry"),
            onClick: () => void detailsQuery.refetch()
          }}
        />
      ) : data ? (
        <>
          {hostedMode && detailsError && (
            <Alert variant="warning" className="mb-3" title={detailsError.title}>
              {detailsError.description}
            </Alert>
          )}
          <Tabs items={tabItems} />
        </>
      ) : null}
    </Drawer>
  )
}
