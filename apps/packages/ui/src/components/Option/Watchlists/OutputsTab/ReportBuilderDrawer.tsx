import React, { useCallback, useEffect, useMemo, useState } from "react"
import { Alert, Button, Checkbox, Drawer, Empty, Input, Select, Spin, Tag, message } from "antd"
import { useTranslation } from "react-i18next"
import {
  createWatchlistOutput,
  fetchScrapedItems,
  fetchWatchlistRuns,
  fetchWatchlistTemplates
} from "@/services/watchlists"
import type {
  ScrapedItem,
  WatchlistContainer,
  WatchlistOutput,
  WatchlistReportPreset,
  WatchlistRun,
  WatchlistTemplate
} from "@/types/watchlists"

interface ReportBuilderDrawerProps {
  open: boolean
  selectedWatchlist: WatchlistContainer | null | undefined
  defaultRunId?: number | null
  onClose: () => void
  onCreated: (output: WatchlistOutput) => void
}

interface PreflightWarning {
  code: string
  label: string
  severity: "warning" | "blocking"
}

const DEFAULT_PAGE_SIZE = 200

const formatCountLabel = (count: number, singular: string, plural = `${singular}s`) =>
  `${count} ${count === 1 ? singular : plural}`

const presetForWatchlist = (
  watchlist: WatchlistContainer | null | undefined
): WatchlistReportPreset => {
  if (watchlist?.domain === "cti_osint") return "cti_osint"
  if (watchlist?.domain === "news") return "news_briefing"
  return "general_research"
}

const buildPreflightWarnings = (
  preset: WatchlistReportPreset,
  queuedItems: ScrapedItem[]
): PreflightWarning[] => {
  const warnings: PreflightWarning[] = []
  if (queuedItems.length === 0) {
    return [
      {
        code: "no_included_updates",
        label: "No included updates",
        severity: "blocking"
      }
    ]
  }

  const uniqueSources = new Set(queuedItems.map((item) => item.source_id).filter(Boolean))
  if (uniqueSources.size <= 1) {
    warnings.push({
      code: "single_source",
      label: "Only one source is represented.",
      severity: "warning"
    })
  }

  if (preset === "cti_osint" && queuedItems.every((item) => !item.alert_summary?.total)) {
    warnings.push({
      code: "no_alert_evidence",
      label: "No alert evidence",
      severity: "warning"
    })
  }

  if (queuedItems.some((item) => !item.reviewed)) {
    warnings.push({
      code: "unreviewed_updates",
      label: "One or more queued updates have not been reviewed.",
      severity: "warning"
    })
  }

  return warnings
}

export const ReportBuilderDrawer: React.FC<ReportBuilderDrawerProps> = ({
  open,
  selectedWatchlist,
  defaultRunId,
  onClose,
  onCreated
}) => {
  const { t } = useTranslation(["watchlists", "common"])
  const [preset, setPreset] = useState<WatchlistReportPreset>(() => presetForWatchlist(selectedWatchlist))
  const [runs, setRuns] = useState<WatchlistRun[]>([])
  const [selectedRunId, setSelectedRunId] = useState<number | null>(defaultRunId ?? null)
  const [queuedItems, setQueuedItems] = useState<ScrapedItem[]>([])
  const [allRunItems, setAllRunItems] = useState<ScrapedItem[]>([])
  const [templates, setTemplates] = useState<WatchlistTemplate[]>([])
  const [templateName, setTemplateName] = useState<string | null>(null)
  const [title, setTitle] = useState("")
  const [format, setFormat] = useState<"md" | "html">("md")
  const [includeEvidenceTable, setIncludeEvidenceTable] = useState(true)
  const [includeExcludedItems, setIncludeExcludedItems] = useState(true)
  const [loadingRuns, setLoadingRuns] = useState(false)
  const [loadingQueue, setLoadingQueue] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [isConstrained, setIsConstrained] = useState(false)

  useEffect(() => {
    if (!open) return
    setPreset(presetForWatchlist(selectedWatchlist))
    setTitle("")
    setFormat("md")
    setIncludeEvidenceTable(true)
    setIncludeExcludedItems(true)
  }, [open, selectedWatchlist])

  useEffect(() => {
    if (!open) return
    const updateViewport = () => {
      setIsConstrained(typeof window !== "undefined" && window.innerWidth <= 480)
    }
    updateViewport()
    window.addEventListener("resize", updateViewport)
    return () => window.removeEventListener("resize", updateViewport)
  }, [open])

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setLoadingRuns(true)
    fetchWatchlistRuns({
      watchlist_id: selectedWatchlist?.id,
      page: 1,
      size: 100
    })
      .then((result) => {
        if (cancelled) return
        const nextRuns = Array.isArray(result.items) ? result.items : []
        setRuns(nextRuns)
        const nextRunId = defaultRunId ?? nextRuns[0]?.id ?? null
        setSelectedRunId(nextRunId)
      })
      .catch((err) => {
        console.error("Failed to load runs for report builder:", err)
        setRuns([])
        setSelectedRunId(null)
      })
      .finally(() => {
        if (!cancelled) setLoadingRuns(false)
      })

    fetchWatchlistTemplates()
      .then((result) => {
        if (!cancelled) setTemplates(Array.isArray(result.items) ? result.items : [])
      })
      .catch((err) => {
        console.error("Failed to load report templates:", err)
        if (!cancelled) setTemplates([])
      })

    return () => {
      cancelled = true
    }
  }, [defaultRunId, open, selectedWatchlist?.id])

  useEffect(() => {
    if (!open || selectedRunId == null) {
      setQueuedItems([])
      setAllRunItems([])
      return
    }
    let cancelled = false
    setLoadingQueue(true)
    Promise.all([
      fetchScrapedItems({
        watchlist_id: selectedWatchlist?.id,
        run_id: selectedRunId,
        queued_for_briefing: true,
        include_alert_summary: true,
        page: 1,
        size: DEFAULT_PAGE_SIZE
      }),
      fetchScrapedItems({
        watchlist_id: selectedWatchlist?.id,
        run_id: selectedRunId,
        include_alert_summary: true,
        page: 1,
        size: DEFAULT_PAGE_SIZE
      })
    ])
      .then(([queuedResult, allResult]) => {
        if (cancelled) return
        setQueuedItems(Array.isArray(queuedResult.items) ? queuedResult.items : [])
        setAllRunItems(Array.isArray(allResult.items) ? allResult.items : [])
      })
      .catch((err) => {
        console.error("Failed to load queued report updates:", err)
        if (!cancelled) {
          setQueuedItems([])
          setAllRunItems([])
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingQueue(false)
      })
    return () => {
      cancelled = true
    }
  }, [open, selectedRunId, selectedWatchlist?.id])

  const preflightWarnings = useMemo(
    () => buildPreflightWarnings(preset, queuedItems),
    [preset, queuedItems]
  )
  const hasBlockingWarning = preflightWarnings.some((warning) => warning.severity === "blocking")
  const hasNonBlockingWarnings = preflightWarnings.some((warning) => warning.severity === "warning")
  const excludedItems = useMemo(
    () => allRunItems.filter((item) => !queuedItems.some((queued) => queued.id === item.id)),
    [allRunItems, queuedItems]
  )
  const sourceCount = useMemo(
    () => new Set(queuedItems.map((item) => item.source_id).filter(Boolean)).size,
    [queuedItems]
  )

  const handleGenerate = useCallback(async () => {
    if (selectedRunId == null) {
      message.error(t("watchlists:reports.builder.runRequired", "Select a run to generate a report."))
      return
    }
    if (queuedItems.length === 0) return
    setGenerating(true)
    try {
      const created = await createWatchlistOutput({
        run_id: selectedRunId,
        item_ids: queuedItems.map((item) => item.id),
        title: title.trim() || undefined,
        format,
        report_preset: preset,
        include_evidence_table: includeEvidenceTable,
        include_excluded_items: includeExcludedItems,
        allow_weak_evidence: true,
        require_reviewed_items: false,
        template_name: templateName || undefined
      })
      message.success(t("watchlists:reports.builder.created", "Generated defensible report."))
      onCreated(created)
    } catch (err) {
      console.error("Failed to create Watchlist report:", err)
      message.error(t("watchlists:reports.builder.createError", "Failed to generate report."))
    } finally {
      setGenerating(false)
    }
  }, [
    format,
    includeEvidenceTable,
    includeExcludedItems,
    onCreated,
    preset,
    queuedItems,
    selectedRunId,
    t,
    templateName,
    title
  ])

  const generateLabel = hasNonBlockingWarnings
    ? t("watchlists:reports.builder.proceedWithWarnings", "Proceed with warnings")
    : t("watchlists:reports.builder.generate", "Generate defensible report")

  return (
    <Drawer
      title={t("watchlists:reports.builder.title", "Create report")}
      placement="right"
      open={open}
      onClose={onClose}
      styles={{ wrapper: { width: 720 } }}
      extra={(
        <Button
          type="primary"
          onClick={handleGenerate}
          loading={generating}
          disabled={Boolean(selectedRunId != null && (hasBlockingWarning || loadingQueue))}
        >
          {generateLabel}
        </Button>
      )}
    >
      <div
        className={isConstrained ? "space-y-4" : "grid grid-cols-[minmax(0,1fr)_260px] gap-4"}
        data-testid="report-builder-layout"
        data-layout={isConstrained ? "stacked" : "wide"}
      >
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="space-y-1">
              <span className="text-xs font-medium text-text-muted">
                {t("watchlists:reports.builder.preset", "Report preset")}
              </span>
              <Select
                data-testid="report-builder-preset"
                value={preset}
                onChange={(value) => setPreset(value as WatchlistReportPreset)}
                className="w-full"
                options={[
                  { value: "cti_osint", label: t("watchlists:reports.presets.cti", "CTI / OSINT") },
                  { value: "news_briefing", label: t("watchlists:reports.presets.news", "News briefing") },
                  { value: "general_research", label: t("watchlists:reports.presets.general", "General research") }
                ]}
              />
            </label>
            <label className="space-y-1">
              <span className="text-xs font-medium text-text-muted">
                {t("watchlists:reports.builder.run", "Run")}
              </span>
              <Select
                data-testid="report-builder-run"
                value={selectedRunId ?? undefined}
                onChange={(value) => setSelectedRunId(typeof value === "number" ? value : Number(value) || null)}
                loading={loadingRuns}
                className="w-full"
                options={runs.map((run) => ({
                  value: run.id,
                  label: `Run #${run.id}`
                }))}
              />
            </label>
          </div>

          <label className="space-y-1 block">
            <span className="text-xs font-medium text-text-muted">
              {t("watchlists:reports.builder.reportTitle", "Report title")}
            </span>
            <Input
              data-testid="report-builder-title"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              placeholder={t("watchlists:reports.builder.titlePlaceholder", "Optional title")}
            />
          </label>

          <div className="grid gap-3 sm:grid-cols-2">
            <label className="space-y-1">
              <span className="text-xs font-medium text-text-muted">
                {t("watchlists:reports.builder.format", "Format")}
              </span>
              <Select
                data-testid="report-builder-format"
                value={format}
                onChange={(value) => setFormat(value === "html" ? "html" : "md")}
                options={[
                  { value: "md", label: "Markdown" },
                  { value: "html", label: "HTML" }
                ]}
              />
            </label>
            <label className="space-y-1">
              <span className="text-xs font-medium text-text-muted">
                {t("watchlists:reports.builder.template", "Template")}
              </span>
              <Select
                data-testid="report-builder-template"
                value={templateName ?? undefined}
                onChange={(value) => setTemplateName(typeof value === "string" && value ? value : null)}
                allowClear
                options={templates.map((template) => ({
                  value: template.name,
                  label: template.name
                }))}
              />
            </label>
          </div>

          <div className="flex flex-wrap gap-3">
            <Checkbox
              checked={includeEvidenceTable}
              onChange={(event) => setIncludeEvidenceTable(Boolean(event.target.checked))}
            >
              {t("watchlists:reports.builder.includeEvidence", "Include evidence table")}
            </Checkbox>
            <Checkbox
              checked={includeExcludedItems}
              onChange={(event) => setIncludeExcludedItems(Boolean(event.target.checked))}
            >
              {t("watchlists:reports.builder.includeExcluded", "Include excluded trail")}
            </Checkbox>
          </div>

          {loadingQueue && queuedItems.length > 0 ? (
            <div className="py-6 text-center"><Spin /></div>
          ) : queuedItems.length > 0 ? (
            <div className="rounded-lg border border-border bg-surface p-3 space-y-2">
              <div className="font-medium text-text">
                {t("watchlists:reports.builder.includedUpdates", "Included updates")}
              </div>
              <div className="space-y-2">
                {queuedItems.slice(0, 8).map((item) => (
                  <div key={item.id} className="flex flex-wrap items-center justify-between gap-2 text-sm">
                    <span>{item.title || `Update #${item.id}`}</span>
                    <div className="flex flex-wrap gap-1">
                      {item.reviewed ? <Tag color="green">Reviewed</Tag> : <Tag color="gold">Needs review</Tag>}
                      {item.alert_summary?.total ? <Tag color="red">Alert evidence</Tag> : null}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={t("watchlists:reports.builder.emptyQueue", "No queued updates found for this run.")}
            />
          )}
        </div>

        <aside className="space-y-3">
          <div className="rounded-lg border border-border bg-surface p-3 space-y-2">
            <div className="font-medium text-text">
              {t("watchlists:reports.builder.readiness", "Report readiness")}
            </div>
            {selectedRunId == null && (
              <Alert
                type="warning"
                showIcon
                title={t("watchlists:reports.builder.runRequired", "Select a run to generate a report.")}
              />
            )}
            <div className="text-sm text-text-muted">
              {formatCountLabel(queuedItems.length, "queued update")}
            </div>
            <div className="text-sm text-text-muted">
              {formatCountLabel(sourceCount, "source")}
            </div>
            {preflightWarnings.length > 0 ? (
              <div className="space-y-2">
                {preflightWarnings.map((warning) => (
                  <Alert
                    key={warning.code}
                    type={warning.severity === "blocking" ? "error" : "warning"}
                    showIcon
                    title={warning.label}
                  />
                ))}
              </div>
            ) : (
              <Tag color="green">{t("watchlists:reports.builder.ready", "Ready")}</Tag>
            )}
          </div>

          <div className="rounded-lg border border-border bg-surface p-3 space-y-2">
            <div className="font-medium text-text">
              {t("watchlists:reports.builder.excludedTrail", "Excluded trail")}
            </div>
            {excludedItems.length > 0 ? (
              <div className="text-sm text-text-muted">
                {formatCountLabel(excludedItems.length, "update")} not queued
              </div>
            ) : (
              <div className="text-sm text-text-muted">
                {t("watchlists:reports.builder.noExcluded", "No excluded updates in this run.")}
              </div>
            )}
          </div>
        </aside>
      </div>
    </Drawer>
  )
}
