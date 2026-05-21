import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Alert,
  Button,
  InputNumber,
  Input,
  Modal,
  Select,
  Space,
  Table,
  Tag,
  Tooltip,
  message
} from "antd"
import type { ColumnsType } from "antd/es/table"
import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Download,
  Eye,
  FileText,
  PlusCircle,
  RefreshCw,
  RotateCcw,
  XCircle
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useWatchlistsStore } from "@/store/watchlists"
import {
  createWatchlistOutput,
  fetchWatchlistJobs,
  fetchWatchlistOutputs,
  fetchWatchlistTemplates,
  downloadWatchlistOutput,
  downloadWatchlistOutputBinary
} from "@/services/watchlists"
import type { WatchlistJob, WatchlistOutput, WatchlistTemplate } from "@/types/watchlists"
import { formatRelativeTime } from "@/utils/dateFormatters"
import { trackWatchlistsOnboardingTelemetry } from "@/utils/watchlists-onboarding-telemetry"
import { OutputPreviewDrawer } from "./OutputPreviewDrawer"
import {
  buildDeliveryDisclosureSummary,
  buildRegenerateOutputRequest,
  createOutputMetadataLabels,
  getDeliveryStatusColor,
  getDeliveryStatusLabel,
  getOutputArtifactLabel,
  getOutputArtifactTagColor,
  getOutputDeliveryStatuses,
  getOutputFileExtension,
  getOutputMimeType,
  getOutputReportReadiness,
  getOutputReportSnapshotAvailable,
  getOutputTemplateName,
  getOutputTemplateVersion,
  getReadinessLabel,
  getReadinessTagColor,
  getAlertCount,
  getSourceCount,
  getWeakEvidenceWarningCount,
  isAudioOutput
} from "./outputMetadata"
import { ReportBuilderDrawer } from "./ReportBuilderDrawer"
import { useWatchlistsViewport } from "../shared/useWatchlistsViewport"
import {
  getFocusableActiveElement,
  restoreFocusToElement
} from "../shared/focus-management"

const OUTPUTS_ADVANCED_FILTERS_STORAGE_KEY = "watchlists:outputs:advanced-filters:v1"

const getSafeLogErrorMessage = (err: unknown): string => {
  if (err instanceof Error) return err.message
  if (typeof err === "string") return err
  try {
    return JSON.stringify(err) ?? String(err)
  } catch {
    return String(err)
  }
}

const readStoredDisclosureState = (key: string): boolean | null => {
  if (typeof window === "undefined") return null
  try {
    const raw = window.localStorage.getItem(key)
    if (raw === "1") return true
    if (raw === "0") return false
  } catch {
    // Ignore storage errors and use default fallback.
  }
  return null
}

const persistDisclosureState = (key: string, value: boolean): void => {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(key, value ? "1" : "0")
  } catch {
    // Ignore storage errors and keep UI functional.
  }
}

const normalizeDeliverySnapshot = (metadata: unknown): string => {
  const deliveries = getOutputDeliveryStatuses(metadata)
  if (!deliveries.length) return ""
  return deliveries
    .map((delivery) => `${delivery.channel}:${delivery.status}`.toLowerCase())
    .sort()
    .join("|")
}

const normalizeDeliveryStatusValue = (value: unknown): string =>
  String(value || "").trim().toLowerCase()

const DELIVERY_ISSUE_STATUSES = new Set(["failed", "error", "partial", "warning"])

const hasOutputDeliveryIssue = (output: WatchlistOutput): boolean =>
  getOutputDeliveryStatuses(output.metadata).some((delivery) =>
    DELIVERY_ISSUE_STATUSES.has(normalizeDeliveryStatusValue(delivery.status))
  )

export const OutputsTab: React.FC = () => {
  const { t } = useTranslation(["watchlists", "common"])
  const { isConstrained } = useWatchlistsViewport()
  const outputMetadataLabels = useMemo(() => createOutputMetadataLabels(t), [t])

  // Store state
  const outputs = useWatchlistsStore((s) => s.outputs)
  const outputsLoading = useWatchlistsStore((s) => s.outputsLoading)
  const outputsTotal = useWatchlistsStore((s) => s.outputsTotal)
  const outputsPage = useWatchlistsStore((s) => s.outputsPage)
  const outputsPageSize = useWatchlistsStore((s) => s.outputsPageSize)
  const outputsJobFilter = useWatchlistsStore((s) => s.outputsJobFilter)
  const outputsRunFilter = useWatchlistsStore((s) => s.outputsRunFilter)
  const outputPreviewOpen = useWatchlistsStore((s) => s.outputPreviewOpen)
  const selectedOutputId = useWatchlistsStore((s) => s.selectedOutputId)
  const selectedWatchlistId = useWatchlistsStore((s) => s.selectedWatchlistId)
  const watchlists = useWatchlistsStore((s) => Array.isArray(s.watchlists) ? s.watchlists : [])

  // Store actions
  const setOutputs = useWatchlistsStore((s) => s.setOutputs)
  const setOutputsLoading = useWatchlistsStore((s) => s.setOutputsLoading)
  const setOutputsPage = useWatchlistsStore((s) => s.setOutputsPage)
  const setOutputsPageSize = useWatchlistsStore((s) => s.setOutputsPageSize)
  const setOutputsJobFilter = useWatchlistsStore((s) => s.setOutputsJobFilter)
  const setOutputsRunFilter = useWatchlistsStore((s) => s.setOutputsRunFilter)
  const setRunsJobFilter = useWatchlistsStore((s) => s.setRunsJobFilter)
  const openRunDetail = useWatchlistsStore((s) => s.openRunDetail)
  const setActiveTab = useWatchlistsStore((s) => s.setActiveTab)
  const openJobForm = useWatchlistsStore((s) => s.openJobForm)
  const openOutputPreview = useWatchlistsStore((s) => s.openOutputPreview)
  const closeOutputPreview = useWatchlistsStore((s) => s.closeOutputPreview)
  const setRunsStatusFilter = useWatchlistsStore((s) => s.setRunsStatusFilter)

  const [jobs, setJobs] = useState<WatchlistJob[]>([])
  const [regenOpen, setRegenOpen] = useState(false)
  const [regenOutput, setRegenOutput] = useState<WatchlistOutput | null>(null)
  const [templates, setTemplates] = useState<WatchlistTemplate[]>([])
  const [templatesLoading, setTemplatesLoading] = useState(false)
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null)
  const [selectedTemplateVersion, setSelectedTemplateVersion] = useState<number | null>(null)
  const [customTitle, setCustomTitle] = useState("")
  const [regenLoading, setRegenLoading] = useState(false)
  const [outputsLiveAnnouncement, setOutputsLiveAnnouncement] = useState("")
  const [reportBuilderOpen, setReportBuilderOpen] = useState(false)
  const regenOutputIsAudio = useMemo(() => isAudioOutput(regenOutput), [regenOutput])
  const [deliveryStatusFilter, setDeliveryStatusFilter] = useState<string | null>(null)
  const normalizedDeliveryStatusFilter = normalizeDeliveryStatusValue(deliveryStatusFilter)
  const hasActiveOutputFilters = Boolean(
    outputsJobFilter || outputsRunFilter || normalizedDeliveryStatusFilter
  )
  const [showAdvancedFilters, setShowAdvancedFilters] = useState<boolean>(() => {
    const stored = readStoredDisclosureState(OUTPUTS_ADVANCED_FILTERS_STORAGE_KEY)
    return stored ?? hasActiveOutputFilters
  })
  const previousDeliverySnapshotRef = useRef<Map<number, string>>(new Map())
  const hasOutputAnnouncementBaselineRef = useRef(false)
  const regenTriggerRef = useRef<HTMLElement | null>(null)
  const wasRegenOpenRef = useRef(false)
  const regenTemplateFieldId = "outputs-regenerate-template-field"
  const regenTemplateVersionFieldId = "outputs-regenerate-template-version-field"
  const regenTitleFieldId = "outputs-regenerate-title-field"

  const selectedTemplateVersionOptions = useMemo(() => {
    if (!selectedTemplate) return []
    const template = templates.find((entry) => entry.name === selectedTemplate)
    if (!template?.available_versions || !Array.isArray(template.available_versions)) return []
    return [...template.available_versions]
      .filter((value): value is number => Number.isInteger(value) && value > 0)
      .sort((a, b) => b - a)
      .map((value) => ({ label: `v${value}`, value }))
  }, [selectedTemplate, templates])

  // Fetch outputs
  const loadOutputs = useCallback(async () => {
    setOutputsLoading(true)
    try {
      const result = await fetchWatchlistOutputs({
        watchlist_id: selectedWatchlistId ?? undefined,
        job_id: outputsJobFilter || undefined,
        run_id: outputsRunFilter || undefined,
        page: outputsPage,
        size: outputsPageSize
      })
      setOutputs(result.items, result.total)
      const firstOutput = Array.isArray(result.items) ? result.items[0] : null
      if (firstOutput) {
        void trackWatchlistsOnboardingTelemetry({
          type: "first_output_succeeded",
          outputId: firstOutput.id,
          format: typeof firstOutput.format === "string" ? firstOutput.format : null
        })
      }
    } catch (err) {
      console.error("Failed to fetch outputs:", err)
      message.error(t("watchlists:outputs.fetchError", "Failed to load outputs"))
      setOutputsLiveAnnouncement(
        t("watchlists:outputs.live.loadError", "Reports refresh failed.")
      )
    } finally {
      setOutputsLoading(false)
    }
  }, [
    outputsJobFilter,
    outputsRunFilter,
    selectedWatchlistId,
    outputsPage,
    outputsPageSize,
    setOutputs,
    setOutputsLoading,
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
    loadOutputs()
    loadJobs()
  }, [loadOutputs, loadJobs])

  const loadTemplates = useCallback(async () => {
    setTemplatesLoading(true)
    try {
      const result = await fetchWatchlistTemplates()
      setTemplates(Array.isArray(result.items) ? result.items : [])
    } catch (err) {
      console.error("Failed to fetch templates:", err)
      setTemplates([])
    } finally {
      setTemplatesLoading(false)
    }
  }, [])

  useEffect(() => {
    if (regenOpen) {
      loadTemplates()
    }
  }, [regenOpen, loadTemplates])

  useEffect(() => {
    if (regenOpen) {
      wasRegenOpenRef.current = true
      return
    }

    if (wasRegenOpenRef.current) {
      wasRegenOpenRef.current = false
      restoreFocusToElement(regenTriggerRef.current)
    }
  }, [regenOpen])

  useEffect(() => {
    if (hasActiveOutputFilters && !showAdvancedFilters) {
      setShowAdvancedFilters(true)
    }
  }, [hasActiveOutputFilters, showAdvancedFilters])

  useEffect(() => {
    persistDisclosureState(OUTPUTS_ADVANCED_FILTERS_STORAGE_KEY, showAdvancedFilters)
  }, [showAdvancedFilters])

  const getOutputAccessibleTitle = useCallback((output: WatchlistOutput) => {
    const title = typeof output.title === "string" ? output.title.trim() : ""
    return title.length > 0 ? title : `Output #${output.id}`
  }, [])

  useEffect(() => {
    const outputItems = Array.isArray(outputs) ? outputs : []
    const nextSnapshot = new Map<number, string>()
    outputItems.forEach((output) => {
      nextSnapshot.set(output.id, normalizeDeliverySnapshot(output.metadata))
    })

    if (!hasOutputAnnouncementBaselineRef.current) {
      hasOutputAnnouncementBaselineRef.current = true
      previousDeliverySnapshotRef.current = nextSnapshot
      return
    }

    const changedOutputs = outputItems.filter((output) => {
      const previousSnapshot = previousDeliverySnapshotRef.current.get(output.id)
      const currentSnapshot = nextSnapshot.get(output.id)
      return Boolean(previousSnapshot) && previousSnapshot !== currentSnapshot
    })

    if (changedOutputs.length === 1) {
      setOutputsLiveAnnouncement(
        t("watchlists:outputs.live.deliveryChangedSingle", "Delivery status updated for {{title}}.", {
          title: getOutputAccessibleTitle(changedOutputs[0])
        })
      )
    } else if (changedOutputs.length > 1) {
      setOutputsLiveAnnouncement(
        t("watchlists:outputs.live.deliveryChangedMultiple", "Delivery status updated for {{count}} reports.", {
          count: changedOutputs.length
        })
      )
    }

    previousDeliverySnapshotRef.current = nextSnapshot
  }, [getOutputAccessibleTitle, outputs, t])

  // Get job name by ID
  const getJobName = useCallback(
    (jobId: number) => {
      const job = jobs.find((j) => j.id === jobId)
      return job?.name || `Monitor #${jobId}`
    },
    [jobs]
  )

  const activeOutputFilterSummary = useMemo(() => {
    const parts: string[] = []
    if (outputsJobFilter) {
      parts.push(
        t("watchlists:outputs.activeFiltersJob", "Monitor: {{name}}", {
          name: getJobName(Number(outputsJobFilter))
        })
      )
    }
    if (outputsRunFilter) {
      parts.push(
        t("watchlists:outputs.activeFiltersRun", "Run: #{{id}}", {
          id: outputsRunFilter
        })
      )
    }
    if (normalizedDeliveryStatusFilter) {
      parts.push(
        t("watchlists:outputs.activeFiltersDelivery", "Delivery: {{status}}", {
          status: getDeliveryStatusLabel(normalizedDeliveryStatusFilter)
        })
      )
    }
    return parts.join(" • ")
  }, [getJobName, normalizedDeliveryStatusFilter, outputsJobFilter, outputsRunFilter, t])

  const filteredOutputs = useMemo(() => {
    const outputItems = Array.isArray(outputs) ? outputs : []
    if (!normalizedDeliveryStatusFilter) return outputItems
    return outputItems.filter((output) =>
      getOutputDeliveryStatuses(output.metadata).some(
        (delivery) =>
          normalizeDeliveryStatusValue(delivery.status) === normalizedDeliveryStatusFilter
      )
    )
  }, [normalizedDeliveryStatusFilter, outputs])

  const outputsWithDeliveryIssues = useMemo(
    () => filteredOutputs.filter((output) => hasOutputDeliveryIssue(output)),
    [filteredOutputs]
  )

  const openFailedRuns = useCallback((jobId: number | null = null) => {
    setRunsStatusFilter("failed")
    setRunsJobFilter(jobId)
    setActiveTab("runs")
  }, [setActiveTab, setRunsJobFilter, setRunsStatusFilter])

  const openOutputMonitor = useCallback((jobId: number) => {
    setActiveTab("jobs")
    openJobForm(jobId)
  }, [openJobForm, setActiveTab])

  const openOutputRun = useCallback((output: WatchlistOutput) => {
    setRunsJobFilter(output.job_id)
    setRunsStatusFilter(null)
    setActiveTab("runs")
    openRunDetail(output.run_id)
  }, [openRunDetail, setActiveTab, setRunsJobFilter, setRunsStatusFilter])

  const deliveryStatusFilterOptions = useMemo(
    () => [
      { label: t("watchlists:outputs.deliveryStatus.failed", "Failed"), value: "failed" },
      { label: t("watchlists:outputs.deliveryStatus.partial", "Partial"), value: "partial" },
      { label: t("watchlists:outputs.deliveryStatus.pending", "Pending"), value: "pending" },
      { label: t("watchlists:outputs.deliveryStatus.sent", "Sent"), value: "sent" },
      { label: t("watchlists:outputs.deliveryStatus.stored", "Stored"), value: "stored" }
    ],
    [t]
  )

  const handleJumpToMonitor = useCallback((jobId: number) => {
    setActiveTab("jobs")
    openJobForm(jobId)
  }, [openJobForm, setActiveTab])

  const handleJumpToRun = useCallback((runId: number, jobId: number) => {
    setRunsJobFilter(jobId)
    setActiveTab("runs")
    openRunDetail(runId)
  }, [openRunDetail, setActiveTab, setRunsJobFilter])

  // Handle download
  const handleDownload = async (output: WatchlistOutput) => {
    try {
      const mimeType = getOutputMimeType(output.format)
      const blob = isAudioOutput(output)
        ? new Blob([await downloadWatchlistOutputBinary(output.id)], { type: mimeType })
        : new Blob([await downloadWatchlistOutput(output.id)], { type: mimeType })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${output.title || `output-${output.id}`}.${getOutputFileExtension(output)}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      message.success(t("watchlists:outputs.downloaded", "Output downloaded"))
      setOutputsLiveAnnouncement(
        t("watchlists:outputs.live.downloaded", "Downloaded {{title}}.", {
          title: getOutputAccessibleTitle(output)
        })
      )
    } catch (err) {
      console.error("Failed to download output:", err)
      message.error(t("watchlists:outputs.downloadError", "Failed to download output"))
      setOutputsLiveAnnouncement(
        t("watchlists:outputs.live.downloadError", "Failed to download {{title}}.", {
          title: getOutputAccessibleTitle(output)
        })
      )
    }
  }

  const openRegenerate = (output: WatchlistOutput) => {
    regenTriggerRef.current = getFocusableActiveElement()
    const outputIsAudio = isAudioOutput(output)
    setRegenOutput(output)
    setSelectedTemplate(outputIsAudio ? null : (getOutputTemplateName(output.metadata) || null))
    setSelectedTemplateVersion(outputIsAudio ? null : (getOutputTemplateVersion(output.metadata) || null))
    setCustomTitle(output.title || "")
    setRegenOpen(true)
  }

  const handleRegenerate = async () => {
    if (!regenOutput) return
    setRegenLoading(true)
    try {
      const regeneratePayload = buildRegenerateOutputRequest(regenOutput, {
        title: customTitle,
        templateName: selectedTemplate,
        templateVersion: selectedTemplateVersion,
        allowTemplateOverrides: !regenOutputIsAudio
      })
      await createWatchlistOutput(regeneratePayload)
      message.success(t("watchlists:outputs.regenerated", "Output regenerated"))
      setOutputsLiveAnnouncement(
        t("watchlists:outputs.live.regenerated", "Regenerated {{title}}.", {
          title: getOutputAccessibleTitle(regenOutput)
        })
      )
      setRegenOpen(false)
      loadOutputs()
    } catch (err) {
      console.error("Failed to regenerate output:", getSafeLogErrorMessage(err))
      message.error(t("watchlists:outputs.regenerateError", "Failed to regenerate output"))
      setOutputsLiveAnnouncement(
        t("watchlists:outputs.live.regenerateError", "Failed to regenerate {{title}}.", {
          title: getOutputAccessibleTitle(regenOutput)
        })
      )
    } finally {
      setRegenLoading(false)
    }
  }

  const renderDeliveryStatusIcon = (status: string) => {
    const normalized = status.trim().toLowerCase()
    if (normalized === "sent" || normalized === "stored" || normalized === "success") {
      return <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />
    }
    if (normalized === "partial" || normalized === "warning") {
      return <AlertTriangle className="h-3.5 w-3.5" aria-hidden />
    }
    if (normalized === "queued" || normalized === "pending" || normalized === "in_progress") {
      return <Clock3 className="h-3.5 w-3.5" aria-hidden />
    }
    if (normalized === "failed" || normalized === "error") {
      return <XCircle className="h-3.5 w-3.5" aria-hidden />
    }
    return <Clock3 className="h-3.5 w-3.5" aria-hidden />
  }

  // Get selected output for preview
  const selectedOutput = selectedOutputId
    ? outputs.find((o) => o.id === selectedOutputId)
    : null
  const selectedWatchlist = useMemo(
    () => watchlists.find((watchlist) => watchlist.id === selectedWatchlistId) || null,
    [selectedWatchlistId, watchlists]
  )

  // Table columns
  const allColumns: ColumnsType<WatchlistOutput> = [
    {
      title: t("watchlists:outputs.columns.title", "Title"),
      dataIndex: "title",
      key: "title",
      ellipsis: true,
      render: (title: string | null, record) => (
        <div className="space-y-1">
          <span className="inline-flex flex-wrap items-center gap-2">
            <span className="font-medium">{title || `Output #${record.id}`}</span>
            <Tag color={getOutputArtifactTagColor(record)}>
              {getOutputArtifactLabel(record)}
            </Tag>
            {getOutputReportSnapshotAvailable(record.metadata) ? (
              <Tag color="blue">
                {t("watchlists:reports.evidence.title", "Evidence snapshot")}
              </Tag>
            ) : null}
          </span>
          <span className="inline-flex flex-wrap gap-2 text-xs text-text-muted">
            <span>
              {t("watchlists:reports.table.sourceCount", "{{count}} sources", {
                count: getSourceCount(record.metadata)
              })}
            </span>
            <span>
              {t("watchlists:reports.table.alertCount", "{{count}} alerts", {
                count: getAlertCount(record.metadata)
              })}
            </span>
            {getWeakEvidenceWarningCount(record.metadata) > 0 && (
              <span>
                {t("watchlists:reports.table.weakWarningCount", "{{count}} weak evidence warnings", {
                  count: getWeakEvidenceWarningCount(record.metadata)
                })}
              </span>
            )}
          </span>
        </div>
      )
    },
    {
      title: t("watchlists:outputs.columns.job", "Monitor"),
      key: "job",
      width: 180,
      ellipsis: true,
      render: (_, record) => (
        <Button
          type="link"
          size="small"
          className="px-0"
          onClick={() => openOutputMonitor(record.job_id)}
          data-testid={`watchlists-output-open-job-${record.id}`}
        >
          {getJobName(record.job_id)}
        </Button>
      )
    },
    {
      title: t("watchlists:outputs.columns.run", "Run"),
      dataIndex: "run_id",
      key: "run_id",
      width: 100,
      render: (runId: number, record) => (
        <Button
          type="link"
          size="small"
          className="px-0"
          onClick={() => openOutputRun(record)}
          data-testid={`watchlists-output-open-run-${record.id}`}
        >
          #{runId}
        </Button>
      )
    },
    {
      title: t("watchlists:outputs.columns.format", "Format"),
      dataIndex: "format",
      key: "format",
      width: 100,
      render: (_format: string, record) => (
        <Tag color={getOutputArtifactTagColor(record)}>
          {getOutputArtifactLabel(record)}
        </Tag>
      )
    },
    {
      title: t("watchlists:outputs.columns.created", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      width: 150,
      render: (date: string) => (
        <span className="text-sm text-text-muted">
          {formatRelativeTime(date, t)}
        </span>
      )
    },
    {
      title: t("watchlists:outputs.columns.delivery", "Delivery"),
      key: "delivery",
      width: 220,
      render: (_, record) => {
        const deliveries = getOutputDeliveryStatuses(record.metadata)
        if (deliveries.length === 0) {
          return <span className="text-text-subtle">-</span>
        }
        const disclosure = buildDeliveryDisclosureSummary(deliveries, {
          maxVisible: showAdvancedFilters ? deliveries.length : 1
        })
        return (
          <Space size={[4, 4]} wrap>
            {disclosure.visible.map((delivery, index) => (
              <Tooltip key={`${delivery.channel}-${delivery.status}-${index}`} title={delivery.detail}>
                <Tag color={getDeliveryStatusColor(delivery.status)}>
                  <span className="inline-flex items-center gap-1">
                    {renderDeliveryStatusIcon(delivery.status)}
                    <span>
                      {delivery.channel} {getDeliveryStatusLabel(delivery.status)}
                    </span>
                  </span>
                </Tag>
              </Tooltip>
            ))}
            {disclosure.hidden.length > 0 && (
              <Tooltip
                title={(
                  <div className="space-y-1">
                    <div className="text-xs font-medium">
                      {t("watchlists:outputs.deliveryOverflowTitle", "Additional delivery statuses")}
                    </div>
                    {disclosure.hidden.map((delivery, index) => (
                      <div
                        key={`${delivery.channel}-${delivery.status}-hidden-${index}`}
                        className="text-xs"
                      >
                        {delivery.channel} {getDeliveryStatusLabel(delivery.status)}
                        {delivery.detail ? `: ${delivery.detail}` : ""}
                      </div>
                    ))}
                  </div>
                )}
              >
                <Button
                  type="link"
                  size="small"
                  className="px-0"
                  aria-label={t("watchlists:outputs.deliveryOverflowAria", "Show additional delivery statuses")}
                >
                  {t("watchlists:outputs.deliveryOverflowCount", "+{{count}} more", {
                    count: disclosure.hidden.length
                  })}
                </Button>
              </Tooltip>
            )}
          </Space>
        )
      }
    },
    {
      title: t("watchlists:outputs.columns.readiness", "Readiness"),
      key: "readiness",
      width: 180,
      render: (_, record) => {
        const readiness = getOutputReportReadiness(record.metadata)
        return (
          <Space size={[4, 4]} wrap>
            <Tag color={getReadinessTagColor(readiness.state)}>
              {getReadinessLabel(readiness.state, outputMetadataLabels)}
            </Tag>
            {getOutputReportSnapshotAvailable(record.metadata) ? (
              <Tooltip title={t("watchlists:reports.evidence.snapshotAvailable", "Immutable evidence snapshot is available")}>
                <Tag color="blue">
                  <span className="inline-flex items-center gap-1">
                    <FileText className="h-3.5 w-3.5" aria-hidden />
                    {t("watchlists:reports.evidence.snapshotShort", "Evidence")}
                  </span>
                </Tag>
              </Tooltip>
            ) : null}
          </Space>
        )
      }
    },
    {
      title: t("watchlists:outputs.columns.expires", "Expires"),
      dataIndex: "expires_at",
      key: "expires_at",
      width: 150,
      render: (date: string | null, record) => {
        if (record.expired) {
          return <Tag color="red">Expired</Tag>
        }
        if (!date) {
          return <span className="text-text-subtle">Never</span>
        }
        return (
          <span className="text-sm text-text-muted">
            {formatRelativeTime(date, t)}
          </span>
        )
      }
    },
    {
      title: t("watchlists:outputs.columns.actions", "Actions"),
      key: "actions",
      width: 140,
      align: "center",
      render: (_, record) => (
        <Space size="small">
          <Tooltip title={t("watchlists:outputs.preview", "Preview")}>
            <Button
              type="text"
              size="small"
              aria-label={t("watchlists:outputs.preview", "Preview")}
              icon={<Eye className="h-4 w-4" />}
              onClick={() => openOutputPreview(record.id)}
            />
          </Tooltip>
          <Tooltip title={t("watchlists:outputs.download", "Download")}>
            <Button
              type="text"
              size="small"
              aria-label={t("watchlists:outputs.download", "Download")}
              icon={<Download className="h-4 w-4" />}
              onClick={() => handleDownload(record)}
            />
          </Tooltip>
          <Tooltip title={t("watchlists:outputs.regenerate", "Regenerate")}>
            <Button
              type="text"
              size="small"
              aria-label={t("watchlists:outputs.regenerate", "Regenerate")}
              icon={<RotateCcw className="h-4 w-4" />}
              onClick={() => openRegenerate(record)}
            />
          </Tooltip>
        </Space>
      )
    }
  ]
  const resolveColumnKey = (column: ColumnsType<WatchlistOutput>[number]): string => {
    if (column.key != null) return String(column.key)
    if ("dataIndex" in column && column.dataIndex != null) {
      return Array.isArray(column.dataIndex)
        ? column.dataIndex.map((entry) => String(entry)).join(".")
        : String(column.dataIndex)
    }
    return ""
  }
  const defaultColumnKeys = new Set(["title", "job", "run_id", "created_at", "delivery", "readiness", "actions"])
  const columns = showAdvancedFilters
    ? allColumns
    : allColumns.filter((column) => defaultColumnKeys.has(resolveColumnKey(column)))

  const renderConstrainedOutputList = () => (
    <div className="space-y-3" data-testid="watchlists-outputs-constrained-list">
      {filteredOutputs.map((output) => {
        const deliveries = getOutputDeliveryStatuses(output.metadata)
        const deliveryDisclosure = buildDeliveryDisclosureSummary(deliveries, {
          maxVisible: showAdvancedFilters ? deliveries.length : 2
        })
        const readiness = getOutputReportReadiness(output.metadata)
        return (
          <article
            key={output.id}
            className="rounded-lg border border-border bg-surface p-3"
            data-testid={`watchlists-output-card-${output.id}`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium text-text">{output.title || `Output #${output.id}`}</span>
                  <Tag color={getOutputArtifactTagColor(output)}>
                    {getOutputArtifactLabel(output)}
                  </Tag>
                  {getOutputReportSnapshotAvailable(output.metadata) ? (
                    <Tag color="blue">
                      {t("watchlists:reports.evidence.title", "Evidence snapshot")}
                    </Tag>
                  ) : null}
                </div>
                <div className="flex flex-wrap gap-2 text-xs text-text-muted">
                  <span>
                    {t("watchlists:reports.table.sourceCount", "{{count}} sources", {
                      count: getSourceCount(output.metadata)
                    })}
                  </span>
                  <span>
                    {t("watchlists:reports.table.alertCount", "{{count}} alerts", {
                      count: getAlertCount(output.metadata)
                    })}
                  </span>
                  {getWeakEvidenceWarningCount(output.metadata) > 0 && (
                    <span>
                      {t("watchlists:reports.table.weakWarningCount", "{{count}} weak evidence warnings", {
                        count: getWeakEvidenceWarningCount(output.metadata)
                      })}
                    </span>
                  )}
                </div>
              </div>
              <span className="shrink-0 text-xs text-text-muted">
                #{output.id}
              </span>
            </div>

            <div className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:outputs.columns.job", "Monitor")}
                </div>
                <Button
                  type="link"
                  size="small"
                  className="px-0"
                  onClick={() => openOutputMonitor(output.job_id)}
                  data-testid={`watchlists-output-open-job-${output.id}`}
                >
                  {getJobName(output.job_id)}
                </Button>
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:outputs.columns.run", "Run")}
                </div>
                <Button
                  type="link"
                  size="small"
                  className="px-0"
                  onClick={() => openOutputRun(output)}
                  data-testid={`watchlists-output-open-run-${output.id}`}
                >
                  #{output.run_id}
                </Button>
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:outputs.columns.created", "Created")}
                </div>
                <span className="text-text-muted">{formatRelativeTime(output.created_at, t)}</span>
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:outputs.columns.expires", "Expires")}
                </div>
                {output.expired ? (
                  <Tag color="red">{t("watchlists:outputs.expired", "Expired")}</Tag>
                ) : output.expires_at ? (
                  <span className="text-text-muted">{formatRelativeTime(output.expires_at, t)}</span>
                ) : (
                  <span className="text-text-muted">{t("watchlists:outputs.never", "Never")}</span>
                )}
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:outputs.columns.delivery", "Delivery")}
                </div>
                {deliveries.length === 0 ? (
                  <span className="text-text-subtle">-</span>
                ) : (
                  <div className="flex flex-wrap gap-1">
                    {deliveryDisclosure.visible.map((delivery, index) => (
                      <Tag key={`${delivery.channel}-${delivery.status}-${index}`} color={getDeliveryStatusColor(delivery.status)}>
                        <span className="inline-flex items-center gap-1">
                          {renderDeliveryStatusIcon(delivery.status)}
                          <span>
                            {delivery.channel} {getDeliveryStatusLabel(delivery.status)}
                          </span>
                        </span>
                      </Tag>
                    ))}
                    {deliveryDisclosure.hidden.length > 0 && (
                      <Tag>
                        {t("watchlists:outputs.deliveryOverflowCount", "+{{count}} more", {
                          count: deliveryDisclosure.hidden.length
                        })}
                      </Tag>
                    )}
                  </div>
                )}
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:outputs.columns.readiness", "Readiness")}
                </div>
                <div className="flex flex-wrap gap-1">
                  <Tag color={getReadinessTagColor(readiness.state)}>
                    {getReadinessLabel(readiness.state, outputMetadataLabels)}
                  </Tag>
                  {getOutputReportSnapshotAvailable(output.metadata) ? (
                    <Tag color="blue">
                      <span className="inline-flex items-center gap-1">
                        <FileText className="h-3.5 w-3.5" aria-hidden />
                        {t("watchlists:reports.evidence.snapshotShort", "Evidence")}
                      </span>
                    </Tag>
                  ) : null}
                </div>
              </div>
            </div>

            <div className="mt-3 flex flex-wrap justify-end gap-2">
              <Button
                type="text"
                size="small"
                aria-label={t("watchlists:outputs.preview", "Preview")}
                icon={<Eye className="h-4 w-4" />}
                onClick={() => openOutputPreview(output.id)}
              />
              <Button
                type="text"
                size="small"
                aria-label={t("watchlists:outputs.download", "Download")}
                icon={<Download className="h-4 w-4" />}
                onClick={() => handleDownload(output)}
              />
              <Button
                type="text"
                size="small"
                aria-label={t("watchlists:outputs.regenerate", "Regenerate")}
                icon={<RotateCcw className="h-4 w-4" />}
                onClick={() => openRegenerate(output)}
              />
            </div>
          </article>
        )
      })}
      <div className="text-xs text-text-subtle">
        {t("watchlists:outputs.totalItems", "{{total}} outputs", {
          total: normalizedDeliveryStatusFilter ? filteredOutputs.length : outputsTotal
        })}
      </div>
    </div>
  )

  return (
    <div className="space-y-4">
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
        data-testid="watchlists-outputs-live-region"
      >
        {outputsLiveAnnouncement}
      </div>

      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <Button
            type={showAdvancedFilters ? "default" : "dashed"}
            size="small"
            data-testid="watchlists-outputs-advanced-toggle"
            onClick={() => setShowAdvancedFilters((prev) => !prev)}
          >
            {showAdvancedFilters
              ? t("watchlists:outputs.hideAdvancedFilters", "Hide advanced filters")
              : t("watchlists:outputs.showAdvancedFilters", "Show advanced filters")}
          </Button>
          {!showAdvancedFilters && hasActiveOutputFilters && (
            <>
              <span className="text-sm text-text-muted" data-testid="watchlists-outputs-active-filters-summary">
                {t("watchlists:outputs.activeFilters", "Active filters")}: {activeOutputFilterSummary}
              </span>
              <Button
                size="small"
                type="text"
                onClick={() => {
                  setOutputsJobFilter(null)
                  setOutputsRunFilter(null)
                  setDeliveryStatusFilter(null)
                }}
              >
                {t("common:clear", "Clear")}
              </Button>
            </>
          )}
          {showAdvancedFilters && (
            <Select
              data-testid="watchlists-outputs-job-filter"
              placeholder={t("watchlists:outputs.filterByJob", "Filter by monitor")}
              value={outputsJobFilter}
              onChange={setOutputsJobFilter}
              allowClear
              className="w-48"
              options={jobs.map((j) => ({
                label: j.name,
                value: j.id
              }))}
            />
          )}
          {showAdvancedFilters && (
            <InputNumber
              data-testid="watchlists-outputs-run-filter"
              placeholder={t("watchlists:outputs.filterByRun", "Filter by run ID")}
              value={outputsRunFilter ?? undefined}
              onChange={(value) =>
                setOutputsRunFilter(
                  typeof value === "number" && Number.isInteger(value) && value > 0
                    ? value
                    : null
                )
              }
              min={1}
              precision={0}
              className="w-40"
            />
          )}
          {showAdvancedFilters && (
            <Select
              data-testid="watchlists-outputs-delivery-filter"
              placeholder={t("watchlists:outputs.filterByDelivery", "Filter by delivery status")}
              value={normalizedDeliveryStatusFilter || null}
              onChange={(value) => {
                setDeliveryStatusFilter(typeof value === "string" ? value : null)
                setOutputsPage(1)
              }}
              allowClear
              className="w-44"
              options={deliveryStatusFilterOptions}
            />
          )}
          {!showAdvancedFilters && (
            <span className="text-xs text-text-subtle">
              {t("watchlists:outputs.metricsHint", "Showing core columns. Use advanced mode for format/run details.")}
            </span>
          )}
        </div>
        <Button
          icon={<RefreshCw className="h-4 w-4" />}
          onClick={loadOutputs}
          loading={outputsLoading}
        >
          {t("common:refresh", "Refresh")}
        </Button>
        <Button
          type="primary"
          icon={<PlusCircle className="h-4 w-4" />}
          onClick={() => setReportBuilderOpen(true)}
        >
          {t("watchlists:reports.builder.createReport", "Create report")}
        </Button>
      </div>

      {/* Description */}
      <div className="text-sm text-text-muted">
        {t("watchlists:outputs.description", "Generated briefings and reports from your watchlist monitors.")}
      </div>

      {outputsWithDeliveryIssues.length > 0 && (
        <Alert
          type="warning"
          showIcon
          data-testid="watchlists-outputs-delivery-issues-banner"
          title={t(
            "watchlists:outputs.deliveryIssuesBannerTitle",
            "Delivery issues detected in {{count}} report{{plural}}.",
            {
              count: outputsWithDeliveryIssues.length,
              plural: outputsWithDeliveryIssues.length === 1 ? "" : "s"
            }
          )}
          description={t(
            "watchlists:outputs.deliveryIssuesBannerDescription",
            "Review failed or partial deliveries and open Activity to investigate monitor/run failures."
          )}
          action={(
            <Space size={8} wrap>
              <Button
                size="small"
                data-testid="watchlists-outputs-banner-show-failed"
                onClick={() => {
                  setShowAdvancedFilters(true)
                  setDeliveryStatusFilter("failed")
                  setOutputsPage(1)
                }}
              >
                {t("watchlists:outputs.deliveryIssuesShowFailed", "Show failed only")}
              </Button>
              <Button
                size="small"
                type="link"
                data-testid="watchlists-outputs-banner-open-runs"
                onClick={() => openFailedRuns(null)}
              >
                {t("watchlists:outputs.deliveryIssuesOpenRuns", "Open failed runs")}
              </Button>
            </Space>
          )}
        />
      )}

      {isConstrained ? (
        renderConstrainedOutputList()
      ) : (
        <Table
          dataSource={filteredOutputs}
          columns={columns}
          rowKey="id"
          aria-label={t("watchlists:outputs.tableAria", "Reports table")}
          loading={outputsLoading}
          pagination={{
            current: outputsPage,
            pageSize: outputsPageSize,
            total: normalizedDeliveryStatusFilter ? filteredOutputs.length : outputsTotal,
            showSizeChanger: true,
            showTotal: (total) =>
              t("watchlists:outputs.totalItems", "{{total}} outputs", { total }),
            onChange: (page, pageSize) => {
              setOutputsPage(page)
              if (pageSize !== outputsPageSize) {
                setOutputsPageSize(pageSize)
              }
            }
          }}
          size="middle"
          scroll={{ x: 800 }}
        />
      )}

      {/* Output Preview Drawer */}
      <OutputPreviewDrawer
        output={selectedOutput}
        open={outputPreviewOpen}
        onClose={closeOutputPreview}
      />

      {reportBuilderOpen && (
        <ReportBuilderDrawer
          open={reportBuilderOpen}
          selectedWatchlist={selectedWatchlist}
          defaultRunId={outputsRunFilter}
          onClose={() => setReportBuilderOpen(false)}
          onCreated={() => {
            setReportBuilderOpen(false)
            loadOutputs()
          }}
        />
      )}

      <Modal
        title={t("watchlists:outputs.regenerateTitle", "Regenerate Output")}
        open={regenOpen}
        onCancel={() => setRegenOpen(false)}
        onOk={handleRegenerate}
        okText={t("watchlists:outputs.regenerate", "Regenerate")}
        cancelText={t("common:cancel", "Cancel")}
        confirmLoading={regenLoading}
      >
        <div className="space-y-3">
          {!regenOutputIsAudio ? (
            <>
              <div>
                <label className="text-xs font-medium text-text-muted mb-1 block" htmlFor={regenTemplateFieldId}>
                  {t("watchlists:outputs.templateLabel", "Template")}
                </label>
                <Select
                  id={regenTemplateFieldId}
                  data-testid="outputs-regenerate-template"
                  value={selectedTemplate ?? undefined}
                  onChange={(value) => {
                    const nextTemplate = value ?? null
                    if (nextTemplate !== selectedTemplate) {
                      setSelectedTemplateVersion(null)
                    }
                    setSelectedTemplate(nextTemplate)
                  }}
                  placeholder={t("watchlists:outputs.templatePlaceholder", "Select a template")}
                  options={templates.map((template) => ({
                    label: template.name,
                    value: template.name
                  }))}
                  loading={templatesLoading}
                  allowClear
                  className="w-full"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-text-muted mb-1 block" htmlFor={regenTemplateVersionFieldId}>
                  {t("watchlists:outputs.templateVersionLabel", "Template version")}
                </label>
                {selectedTemplateVersionOptions.length > 0 ? (
                  <Select
                    id={regenTemplateVersionFieldId}
                    data-testid="outputs-regenerate-template-version-select"
                    value={selectedTemplateVersion ?? undefined}
                    onChange={(value) =>
                      setSelectedTemplateVersion(typeof value === "number" ? value : null)
                    }
                    placeholder={t("watchlists:outputs.templateVersionPlaceholder", "Latest/default")}
                    options={selectedTemplateVersionOptions}
                    disabled={!selectedTemplate}
                    allowClear
                    className="w-full"
                  />
                ) : (
                  <InputNumber
                    id={regenTemplateVersionFieldId}
                    data-testid="outputs-regenerate-template-version-input"
                    value={selectedTemplateVersion ?? undefined}
                    min={1}
                    precision={0}
                    onChange={(value) =>
                      setSelectedTemplateVersion(
                        typeof value === "number" && Number.isInteger(value) && value > 0 ? value : null
                      )
                    }
                    disabled={!selectedTemplate}
                    placeholder={t("watchlists:outputs.templateVersionPlaceholder", "Latest/default")}
                    className="w-full"
                  />
                )}
              </div>
            </>
          ) : (
            <div className="rounded-lg border border-border bg-surface p-3 text-xs text-text-muted">
              {t(
                "watchlists:outputs.regenerateAudioTemplateHint",
                "Audio outputs regenerate using run audio settings. Template overrides are unavailable."
              )}
            </div>
          )}
          <div>
            <label className="text-xs font-medium text-text-muted mb-1 block" htmlFor={regenTitleFieldId}>
              {t("watchlists:outputs.titleLabel", "Title")}
            </label>
            <Input
              id={regenTitleFieldId}
              value={customTitle}
              onChange={(e) => setCustomTitle(e.target.value)}
              placeholder={t("watchlists:outputs.titlePlaceholder", "Optional title override")}
            />
          </div>
        </div>
      </Modal>
    </div>
  )
}
