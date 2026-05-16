import React, { useCallback, useEffect, useState } from "react"
import {
  Alert,
  Button,
  Modal,
  Space,
  Switch,
  Table,
  Tag,
  Tooltip,
  message
} from "antd"
import type { ColumnsType } from "antd/es/table"
import { Edit2, Eye, Play, Plus, RefreshCw, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useUndoNotification } from "@/hooks/useUndoNotification"
import { useWatchlistsStore } from "@/store/watchlists"
import {
  deleteWatchlistJob,
  fetchJobRuns,
  fetchWatchlistGroups,
  fetchWatchlistJobs,
  fetchWatchlistSources,
  restoreWatchlistJob,
  triggerWatchlistRun,
  updateWatchlistJob
} from "@/services/watchlists"
import type { WatchlistJob } from "@/types/watchlists"
import { formatRelativeTime } from "@/utils/dateFormatters"
import { CronDisplay } from "../shared"
import { JobFormModal } from "./JobFormModal"
import {
  JOB_DELETE_UNDO_WINDOW_SECONDS,
  resolveJobUndoWindowSeconds,
  toJobRestoreId
} from "./job-undo"
import {
  buildScopeTooltipLines,
  summarizeFilters,
  summarizeOutputLinkage,
  summarizeScopeCounts
} from "./job-summaries"
import { JobPreviewModal } from "./JobPreviewModal"
import { mapWatchlistsError } from "../shared/watchlists-error"
import { useWatchlistsViewport } from "../shared/useWatchlistsViewport"

const SCOPE_CATALOG_LIMIT = 200
const JOBS_ADVANCED_COLUMNS_STORAGE_KEY = "watchlists:jobs:advanced-columns:v1"

const readStoredDisclosureState = (key: string): boolean | null => {
  if (typeof window === "undefined") return null
  try {
    const raw = window.localStorage.getItem(key)
    if (raw === "1") return true
    if (raw === "0") return false
  } catch {
    // Ignore storage access errors and keep defaults.
  }
  return null
}

const persistDisclosureState = (key: string, value: boolean): void => {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(key, value ? "1" : "0")
  } catch {
    // Ignore storage access errors and keep UI functional.
  }
}

const isAudioBriefingEnabled = (outputPrefs: unknown): boolean => {
  if (!outputPrefs || typeof outputPrefs !== "object" || Array.isArray(outputPrefs)) {
    return false
  }
  const value = (outputPrefs as Record<string, unknown>).generate_audio
  return value === true
}

export const JobsTab: React.FC = () => {
  const { t } = useTranslation(["watchlists", "common"])
  const { showUndoNotification } = useUndoNotification()
  const { isConstrained } = useWatchlistsViewport()

  // Store state
  const jobs = useWatchlistsStore((s) => s.jobs)
  const jobsLoading = useWatchlistsStore((s) => s.jobsLoading)
  const jobsTotal = useWatchlistsStore((s) => s.jobsTotal)
  const jobsPage = useWatchlistsStore((s) => s.jobsPage)
  const jobsPageSize = useWatchlistsStore((s) => s.jobsPageSize)
  const jobFormOpen = useWatchlistsStore((s) => s.jobFormOpen)
  const jobFormEditId = useWatchlistsStore((s) => s.jobFormEditId)
  const selectedWatchlistId = useWatchlistsStore((s) => s.selectedWatchlistId)

  // Store actions
  const setJobs = useWatchlistsStore((s) => s.setJobs)
  const setJobsLoading = useWatchlistsStore((s) => s.setJobsLoading)
  const setJobsPage = useWatchlistsStore((s) => s.setJobsPage)
  const setJobsPageSize = useWatchlistsStore((s) => s.setJobsPageSize)
  const openJobForm = useWatchlistsStore((s) => s.openJobForm)
  const closeJobForm = useWatchlistsStore((s) => s.closeJobForm)
  const addJob = useWatchlistsStore((s) => s.addJob)
  const updateJobInList = useWatchlistsStore((s) => s.updateJobInList)
  const removeJob = useWatchlistsStore((s) => s.removeJob)
  const addRun = useWatchlistsStore((s) => s.addRun)

  // Local state
  const [triggeringJobId, setTriggeringJobId] = useState<number | null>(null)
  const [previewJob, setPreviewJob] = useState<WatchlistJob | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const [sourceNamesById, setSourceNamesById] = useState<Record<number, string>>({})
  const [groupNamesById, setGroupNamesById] = useState<Record<number, string>>({})
  const [jobsLoadError, setJobsLoadError] = useState<ReturnType<typeof mapWatchlistsError> | null>(null)
  const [showAdvancedColumns, setShowAdvancedColumns] = useState<boolean>(() => {
    const stored = readStoredDisclosureState(JOBS_ADVANCED_COLUMNS_STORAGE_KEY)
    return stored ?? false
  })

  // Fetch jobs
  const loadJobs = useCallback(async () => {
    setJobsLoading(true)
    try {
      const result = await fetchWatchlistJobs({
        watchlist_id: selectedWatchlistId ?? undefined,
        page: jobsPage,
        size: jobsPageSize
      })
      setJobs(result.items, result.total)
      setJobsLoadError(null)
    } catch (err) {
      console.error("Failed to fetch jobs:", err)
      setJobsLoadError(
        mapWatchlistsError(err, {
          t,
          context: t("watchlists:jobs.title", "Monitors"),
          fallbackMessage: t("watchlists:jobs.fetchError", "Failed to load monitors")
        })
      )
    } finally {
      setJobsLoading(false)
    }
  }, [jobsPage, jobsPageSize, selectedWatchlistId, setJobs, setJobsLoading, t])

  // Initial load
  useEffect(() => {
    loadJobs()
  }, [loadJobs])

  const loadScopeCatalog = useCallback(async () => {
    try {
      const [sourcesResult, groupsResult] = await Promise.all([
        fetchWatchlistSources({
          watchlist_id: selectedWatchlistId ?? undefined,
          page: 1,
          size: SCOPE_CATALOG_LIMIT
        }),
        fetchWatchlistGroups({ page: 1, size: SCOPE_CATALOG_LIMIT })
      ])

      const nextSourceNames: Record<number, string> = {}
      const nextGroupNames: Record<number, string> = {}

      for (const source of sourcesResult.items || []) {
        nextSourceNames[source.id] = source.name
      }
      for (const group of groupsResult.items || []) {
        nextGroupNames[group.id] = group.name
      }

      setSourceNamesById(nextSourceNames)
      setGroupNamesById(nextGroupNames)
    } catch (err) {
      console.warn("Failed to fetch monitor scope catalog:", err)
    }
  }, [selectedWatchlistId])

  useEffect(() => {
    void loadScopeCatalog()
  }, [loadScopeCatalog])

  useEffect(() => {
    persistDisclosureState(JOBS_ADVANCED_COLUMNS_STORAGE_KEY, showAdvancedColumns)
  }, [showAdvancedColumns])

  // Handle toggle active
  const handleToggleActive = async (job: WatchlistJob) => {
    try {
      const updated = await updateWatchlistJob(job.id, { active: !job.active })
      updateJobInList(job.id, updated)
      message.success(
        job.active
          ? t("watchlists:jobs.disabled", "Monitor disabled")
          : t("watchlists:jobs.enabled", "Monitor enabled")
      )
    } catch (err) {
      console.error("Failed to toggle job:", err)
      message.error(t("watchlists:jobs.toggleError", "Failed to update monitor"))
    }
  }

  // Handle delete with impact assessment
  const executeDelete = async (jobId: number) => {
    const deletedJob = jobs.find((job) => job.id === jobId)
    try {
      const deleteResult = await deleteWatchlistJob(jobId)
      removeJob(jobId)
      if (!deletedJob) {
        message.success(t("watchlists:jobs.deleted", "Monitor deleted"))
        return
      }
      const undoWindowSeconds = resolveJobUndoWindowSeconds(
        deleteResult?.restore_window_seconds,
        JOB_DELETE_UNDO_WINDOW_SECONDS
      )

      showUndoNotification({
        title: t("watchlists:jobs.undoDeleteTitle", "Monitor deleted"),
        description: t(
          "watchlists:jobs.undoDeleteDescription",
          "Undo restores this monitor's schedule, feed scope, and delivery settings for {{seconds}} seconds.",
          { seconds: undoWindowSeconds }
        ),
        duration: undoWindowSeconds,
        onDismiss: () => {
          void loadJobs()
        },
        onUndo: async () => {
          try {
            await restoreWatchlistJob(toJobRestoreId(deletedJob))
          } catch (restoreErr) {
            const restoreError = mapWatchlistsError(restoreErr, {
              t,
              context: t("watchlists:jobs.title", "Monitors"),
              operationLabel: t("watchlists:errors.operation.retry", "retry"),
              fallbackMessage: t(
                "watchlists:jobs.undoRestoreError",
                "Could not restore this monitor. Refresh Monitors and retry while the undo timer is active."
              )
            })
            throw new Error(restoreError.description)
          } finally {
            await loadJobs()
          }
        }
      })
    } catch (err) {
      console.error("Failed to delete job:", err)
      message.error(t("watchlists:jobs.deleteError", "Failed to delete monitor"))
    }
  }

  const requestDeleteConfirmation = async (job: WatchlistJob) => {
    const warnings: string[] = []

    try {
      const runsResult = await fetchJobRuns(job.id, { page: 1, size: 10 })
      const activeRuns = (runsResult.items || []).filter(
        (run) => {
          const normalized = String(run.status || "").toLowerCase()
          return normalized === "running" || normalized === "pending" || normalized === "queued"
        }
      )
      if (activeRuns.length > 0) {
        warnings.push(
          t(
            "watchlists:jobs.deleteConfirmActiveRunsWarning",
            "This monitor has {{count}} active or pending run{{plural}}. Deleting it will not cancel in-progress runs.",
            { count: activeRuns.length, plural: activeRuns.length === 1 ? "" : "s" }
          )
        )
      }
    } catch {
      // Non-critical: proceed with delete confirmation without run info.
    }

    const outputPrefs = job.output_prefs as Record<string, unknown> | undefined
    const hasDelivery =
      outputPrefs &&
      (Array.isArray(outputPrefs.email_recipients) && (outputPrefs.email_recipients as unknown[]).length > 0 ||
        outputPrefs.chatbook_path)
    if (hasDelivery) {
      warnings.push(
        t(
          "watchlists:jobs.deleteConfirmDeliveryWarning",
          "This monitor has configured delivery settings (email, chatbook). Deleting it removes scheduled deliveries."
        )
      )
    }

    Modal.confirm({
      title: t("watchlists:jobs.deleteConfirm", "Delete this monitor?"),
      content: warnings.length > 0 ? (
        <div className="space-y-2">
          {warnings.map((warning, index) => (
            <p key={index} className="text-sm text-warning">
              {warning}
            </p>
          ))}
          <p className="text-sm text-text-muted">
            {t(
              "watchlists:jobs.deleteConfirmDescription",
              "You can undo this deletion for {{seconds}} seconds.",
              { seconds: JOB_DELETE_UNDO_WINDOW_SECONDS }
            )}
          </p>
        </div>
      ) : (
        <p className="text-sm text-text-muted">
          {t(
            "watchlists:jobs.deleteConfirmDescription",
            "You can undo this deletion for {{seconds}} seconds.",
            { seconds: JOB_DELETE_UNDO_WINDOW_SECONDS }
          )}
        </p>
      ),
      okText: t("common:delete", "Delete"),
      okButtonProps: { danger: true },
      cancelText: t("common:cancel", "Cancel"),
      onCancel: () => undefined,
      onOk: () => executeDelete(job.id)
    })
  }

  // Handle manual run trigger
  const handleTriggerRun = async (jobId: number) => {
    setTriggeringJobId(jobId)
    try {
      const run = await triggerWatchlistRun(jobId)
      addRun(run)
      updateJobInList(jobId, { last_run_at: run.started_at || new Date().toISOString() })
      message.success(t("watchlists:jobs.runTriggered", "Run triggered"))
    } catch (err) {
      console.error("Failed to trigger run:", err)
      message.error(t("watchlists:jobs.runError", "Failed to trigger run"))
    } finally {
      setTriggeringJobId(null)
    }
  }

  // Get job for editing
  const editingJob = jobFormEditId
    ? jobs.find((j) => j.id === jobFormEditId)
    : undefined

  // Table columns
  const allColumns: ColumnsType<WatchlistJob> = [
    {
      title: t("watchlists:jobs.columns.name", "Name"),
      dataIndex: "name",
      key: "name",
      ellipsis: true,
      render: (name: string, record) => (
        <div>
          <span className="inline-flex items-center gap-2">
            <span className="font-medium">{name}</span>
            {isAudioBriefingEnabled(record.output_prefs) && (
              <Tag color="purple" data-testid={`job-audio-enabled-chip-${record.id}`}>
                {t("watchlists:jobs.audioEnabledChip", "Audio on")}
              </Tag>
            )}
          </span>
          {record.description && (
            <div className="text-xs text-text-muted truncate">
              {record.description}
            </div>
          )}
          {!showAdvancedColumns && (
            <div className="text-xs text-text-muted mt-1" data-testid={`job-compact-summary-${record.id}`}>
              {t("watchlists:jobs.compactSummary", "{{scope}} • {{filters}} filters", {
                scope: summarizeScopeCounts(record.scope, t),
                filters: summarizeFilters(record.job_filters?.filters, t).count
              })}
            </div>
          )}
          <div
            className="text-xs text-text-muted mt-1"
            data-testid={`job-output-linkage-${record.id}`}
          >
            {t("watchlists:jobs.outputLinkage.label", "Output linkage")}:{" "}
            {summarizeOutputLinkage(record.output_prefs, t)}
          </div>
        </div>
      )
    },
    {
      title: t("watchlists:jobs.columns.schedule", "Schedule"),
      dataIndex: "schedule_expr",
      key: "schedule_expr",
      width: 180,
      render: (schedule: string | null) => <CronDisplay expression={schedule} />
    },
    {
      title: t("watchlists:jobs.columns.scope", "Feeds"),
      key: "scope",
      width: 220,
      render: (_, record) => {
        const scopeSummary = summarizeScopeCounts(record.scope, t)
        const tooltipLines = buildScopeTooltipLines(
          record.scope,
          { sources: sourceNamesById, groups: groupNamesById },
          t
        )
        return (
          <Tooltip
            title={
              <div className="max-w-[340px]">
                {tooltipLines.map((line, index) => (
                  <div key={`${record.id}-scope-line-${index}`}>{line}</div>
                ))}
              </div>
            }
          >
            <span
              className="text-sm text-text-muted cursor-help"
              data-testid={`job-scope-summary-${record.id}`}
            >
              {scopeSummary}
            </span>
          </Tooltip>
        )
      }
    },
    {
      title: t("watchlists:jobs.columns.filters", "Filters"),
      key: "filters",
      width: 260,
      render: (_, record) => {
        const summary = summarizeFilters(record.job_filters?.filters, t)
        if (summary.count === 0) {
          return <span className="text-text-subtle">-</span>
        }

        return (
          <Tooltip
            title={
              <div className="max-w-[340px]">
                {summary.tooltipLines.map((line, index) => (
                  <div key={`${record.id}-filter-line-${index}`}>{line}</div>
                ))}
              </div>
            }
          >
            <div
              className="flex items-center gap-2"
              data-testid={`job-filters-summary-${record.id}`}
            >
              <Tag>{summary.count}</Tag>
              <span className="text-sm text-text-muted truncate">
                {summary.preview}
              </span>
            </div>
          </Tooltip>
        )
      }
    },
    {
      title: t("watchlists:jobs.columns.lastRun", "Last Run"),
      dataIndex: "last_run_at",
      key: "last_run_at",
      width: 140,
      render: (date: string | null) =>
        date ? (
          <span className="text-sm text-text-muted">
            {formatRelativeTime(date, t)}
          </span>
        ) : (
          <span className="text-sm text-text-subtle">
            {t("watchlists:jobs.never", "Never")}
          </span>
        )
    },
    {
      title: t("watchlists:jobs.columns.nextRun", "Next Run"),
      dataIndex: "next_run_at",
      key: "next_run_at",
      width: 150,
      render: (date: string | null, record) => {
        if (!date) {
          return (
            <span className="text-sm text-text-subtle">
              {record.schedule_expr
                ? t("watchlists:jobs.pending", "Pending")
                : t("watchlists:jobs.notScheduled", "Not scheduled")}
            </span>
          )
        }

        return (
          <Tooltip title={new Date(date).toLocaleString()}>
            <span className="text-sm text-text-muted">
              {formatRelativeTime(date, t)}
            </span>
          </Tooltip>
        )
      }
    },
    {
      title: t("watchlists:jobs.columns.active", "Active"),
      dataIndex: "active",
      key: "active",
      width: 140,
      align: "center",
      render: (active: boolean, record) => (
        <span className="inline-flex items-center gap-2">
          <Switch
            checked={active}
            size="small"
            aria-label={t("watchlists:jobs.toggleActiveAria", "Toggle active for {{name}}", { name: record.name })}
            onChange={() => handleToggleActive(record)}
          />
          <span className="text-xs text-text-muted">
            {active
              ? t("common:enabled", "Enabled")
              : t("common:disabled", "Disabled")}
          </span>
        </span>
      )
    },
    {
      title: t("watchlists:jobs.columns.actions", "Actions"),
      key: "actions",
      width: 140,
      align: "center",
      render: (_, record) => {
        const runNowTooltip = record.active
          ? t("watchlists:jobs.runNow", "Run Now")
          : t("watchlists:jobs.runNowDisabledHint", "Activate this monitor to run it manually")
        return (
        <Space size="small">
          <Tooltip title={runNowTooltip}>
            <Button
              type="text"
              size="small"
              aria-label={runNowTooltip}
              icon={<Play className="h-4 w-4" />}
              onClick={() => handleTriggerRun(record.id)}
              loading={triggeringJobId === record.id}
              disabled={!record.active}
            />
          </Tooltip>
          <Tooltip title={t("watchlists:jobs.preview.button", "Preview")}>
            <Button
              type="text"
              size="small"
              aria-label={t("watchlists:jobs.preview.button", "Preview")}
              icon={<Eye className="h-4 w-4" />}
              onClick={() => {
                setPreviewJob(record)
                setPreviewOpen(true)
              }}
            />
          </Tooltip>
          <Tooltip title={t("common:edit", "Edit")}>
            <Button
              type="text"
              size="small"
              aria-label={t("common:edit", "Edit")}
              icon={<Edit2 className="h-4 w-4" />}
              onClick={() => openJobForm(record.id)}
            />
          </Tooltip>
          <Tooltip title={t("common:delete", "Delete")}>
            <Button
              type="text"
              size="small"
              danger
              aria-label={t("common:delete", "Delete")}
              icon={<Trash2 className="h-4 w-4" />}
              onClick={() => requestDeleteConfirmation(record)}
            />
          </Tooltip>
        </Space>
      )}
    }
  ]
  const resolveColumnKey = (column: ColumnsType<WatchlistJob>[number]): string => {
    if (column.key != null) return String(column.key)
    if ("dataIndex" in column && column.dataIndex != null) {
      return Array.isArray(column.dataIndex)
        ? column.dataIndex.map((entry) => String(entry)).join(".")
        : String(column.dataIndex)
    }
    return ""
  }
  const defaultColumnKeys = new Set(["name", "schedule_expr", "active", "actions"])
  const columns = showAdvancedColumns
    ? allColumns
    : allColumns.filter((column) => defaultColumnKeys.has(resolveColumnKey(column)))

  const renderFilterSummary = (job: WatchlistJob) => {
    const summary = summarizeFilters(job.job_filters?.filters, t)
    if (summary.count === 0) {
      return (
        <span className="text-text-subtle" data-testid={`job-filters-summary-${job.id}`}>
          {t("watchlists:jobs.noFilters", "No filters")}
        </span>
      )
    }
    return (
      <div className="flex items-center gap-2" data-testid={`job-filters-summary-${job.id}`}>
        <Tag>{summary.count}</Tag>
        <span className="min-w-0 truncate text-sm text-text-muted">{summary.preview}</span>
      </div>
    )
  }

  const renderConstrainedJobList = () => (
    <div className="space-y-3" data-testid="watchlists-jobs-constrained-list">
      {(Array.isArray(jobs) ? jobs : []).map((job) => {
        const runNowLabel = job.active
          ? t("watchlists:jobs.runNow", "Run Now")
          : t("watchlists:jobs.runNowDisabledHint", "Activate this monitor to run it manually")
        return (
          <article
            key={job.id}
            className="rounded-lg border border-border bg-surface p-3"
            data-testid={`watchlists-job-card-${job.id}`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium text-text">{job.name}</span>
                  {isAudioBriefingEnabled(job.output_prefs) && (
                    <Tag color="purple" data-testid={`job-audio-enabled-chip-${job.id}`}>
                      {t("watchlists:jobs.audioEnabledChip", "Audio on")}
                    </Tag>
                  )}
                </div>
                {job.description ? (
                  <div className="text-xs text-text-muted">{job.description}</div>
                ) : null}
              </div>
              <span className="inline-flex shrink-0 items-center gap-2">
                <Switch
                  checked={job.active}
                  size="small"
                  aria-label={t("watchlists:jobs.toggleActiveAria", "Toggle active for {{name}}", { name: job.name })}
                  onChange={() => handleToggleActive(job)}
                />
                <span className="text-xs text-text-muted">
                  {job.active ? t("common:enabled", "Enabled") : t("common:disabled", "Disabled")}
                </span>
              </span>
            </div>

            <div className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:jobs.columns.schedule", "Schedule")}
                </div>
                <CronDisplay expression={job.schedule_expr} />
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:jobs.columns.scope", "Feeds")}
                </div>
                <span className="text-sm text-text-muted" data-testid={`job-scope-summary-${job.id}`}>
                  {summarizeScopeCounts(job.scope, t)}
                </span>
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:jobs.columns.filters", "Filters")}
                </div>
                {renderFilterSummary(job)}
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:jobs.outputLinkage.label", "Output linkage")}
                </div>
                <div className="text-sm text-text-muted" data-testid={`job-output-linkage-${job.id}`}>
                  {summarizeOutputLinkage(job.output_prefs, t)}
                </div>
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:jobs.columns.lastRun", "Last run")}
                </div>
                <span className="text-sm text-text-muted">
                  {job.last_run_at ? formatRelativeTime(job.last_run_at, t) : t("watchlists:jobs.never", "Never")}
                </span>
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:jobs.columns.nextRun", "Next run")}
                </div>
                <span className="text-sm text-text-muted">
                  {job.next_run_at
                    ? formatRelativeTime(job.next_run_at, t)
                    : job.schedule_expr
                      ? t("watchlists:jobs.pending", "Pending")
                      : t("watchlists:jobs.notScheduled", "Not scheduled")}
                </span>
              </div>
            </div>

            <div className="mt-3 flex flex-wrap justify-end gap-2">
              <Button
                type="text"
                size="small"
                aria-label={runNowLabel}
                icon={<Play className="h-4 w-4" />}
                onClick={() => handleTriggerRun(job.id)}
                loading={triggeringJobId === job.id}
                disabled={!job.active}
              />
              <Button
                type="text"
                size="small"
                aria-label={t("watchlists:jobs.previewForMonitorAria", "Preview {{name}}", { name: job.name })}
                icon={<Eye className="h-4 w-4" />}
                onClick={() => {
                  setPreviewJob(job)
                  setPreviewOpen(true)
                }}
              />
              <Button
                type="text"
                size="small"
                aria-label={t("watchlists:jobs.editMonitorAria", "Edit {{name}}", { name: job.name })}
                icon={<Edit2 className="h-4 w-4" />}
                onClick={() => openJobForm(job.id)}
              />
              <Button
                type="text"
                size="small"
                danger
                aria-label={t("watchlists:jobs.deleteMonitorAria", "Delete {{name}}", { name: job.name })}
                icon={<Trash2 className="h-4 w-4" />}
                onClick={() => requestDeleteConfirmation(job)}
              />
            </div>
          </article>
        )
      })}
      <div className="text-xs text-text-subtle">
        {t("watchlists:jobs.totalItems", "{{total}} monitors", { total: jobsTotal })}
      </div>
    </div>
  )

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="text-sm text-text-muted">
          {t("watchlists:jobs.description", "Create monitors that automatically fetch and process updates from your feeds.")}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="small"
            type={showAdvancedColumns ? "default" : "dashed"}
            data-testid="watchlists-jobs-advanced-toggle"
            onClick={() => setShowAdvancedColumns((previous) => !previous)}
          >
            {showAdvancedColumns
              ? t("watchlists:jobs.hideAdvancedDetails", "Hide advanced details")
              : t("watchlists:jobs.showAdvancedDetails", "Show advanced details")}
          </Button>
          <Button
            icon={<RefreshCw className="h-4 w-4" />}
            onClick={loadJobs}
            loading={jobsLoading}
          >
            {t("common:refresh", "Refresh")}
          </Button>
          <Button
            type="primary"
            icon={<Plus className="h-4 w-4" />}
            onClick={() => openJobForm()}
          >
            {t("watchlists:jobs.addJob", "Add Monitor")}
          </Button>
        </div>
      </div>
      {!showAdvancedColumns && (
        <div className="text-xs text-text-subtle">
          {t("watchlists:jobs.metricsHint", "Showing core columns. Use advanced mode for scope, filters, and run timing.")}
        </div>
      )}

      {jobsLoadError && (
        <Alert
          type={jobsLoadError.severity}
          showIcon
          title={jobsLoadError.title}
          description={jobsLoadError.description}
          action={(
            <Button
              size="small"
              onClick={() => void loadJobs()}
              loading={jobsLoading}
            >
              {t("watchlists:errors.retry", "Retry")}
            </Button>
          )}
        />
      )}

      {isConstrained ? (
        renderConstrainedJobList()
      ) : (
        <Table
          dataSource={Array.isArray(jobs) ? jobs : []}
          columns={columns}
          rowKey="id"
          aria-label={t("watchlists:jobs.tableAria", "Monitors table")}
          loading={jobsLoading}
          pagination={{
            current: jobsPage,
            pageSize: jobsPageSize,
            total: jobsTotal,
            showSizeChanger: true,
            showTotal: (total) =>
              t("watchlists:jobs.totalItems", "{{total}} monitors", { total }),
            onChange: (page, pageSize) => {
              setJobsPage(page)
              if (pageSize !== jobsPageSize) {
                setJobsPageSize(pageSize)
              }
            }
          }}
          size="middle"
          scroll={{ x: 900 }}
        />
      )}

      {/* Job Form Modal */}
      <JobFormModal
        open={jobFormOpen}
        onClose={closeJobForm}
        initialValues={editingJob}
        watchlistId={selectedWatchlistId}
        onSuccess={() => {
          closeJobForm()
          loadJobs()
        }}
      />

      <JobPreviewModal
        job={previewJob}
        open={previewOpen}
        onClose={() => setPreviewOpen(false)}
      />
    </div>
  )
}
