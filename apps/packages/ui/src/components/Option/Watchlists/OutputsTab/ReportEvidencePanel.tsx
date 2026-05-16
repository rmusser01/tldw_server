import React, { useCallback, useEffect, useMemo, useState } from "react"
import { Alert, Button, Empty, Spin, Table, Tag, Tooltip } from "antd"
import type { ColumnsType } from "antd/es/table"
import { useTranslation } from "react-i18next"
import { getWatchlistOutputEvidence } from "@/services/watchlists"
import type {
  WatchlistOutputEvidenceResponse,
  WatchlistReportEvidenceItem,
  WatchlistReportExcludedItem
} from "@/types/watchlists"
import {
  getReadinessLabel,
  getReadinessTagColor
} from "./outputMetadata"
import { useWatchlistsViewport } from "../shared/useWatchlistsViewport"

interface ReportEvidencePanelProps {
  outputId: number
  evidenceResponse?: WatchlistOutputEvidenceResponse | null
  compact?: boolean
}

const formatExcludedReason = (reason: string): string => {
  if (reason === "not_queued_for_report") return "Not queued for report"
  if (reason === "filtered_or_error") return "Filtered or errored"
  if (reason === "excluded_from_report") return "Excluded from report"
  return reason
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

export const ReportEvidencePanel: React.FC<ReportEvidencePanelProps> = ({
  outputId,
  evidenceResponse,
  compact = false
}) => {
  const { t } = useTranslation(["watchlists", "common"])
  const { isConstrained } = useWatchlistsViewport()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [response, setResponse] = useState<WatchlistOutputEvidenceResponse | null>(
    evidenceResponse ?? null
  )

  const loadEvidence = useCallback(async () => {
    if (evidenceResponse) {
      setResponse(evidenceResponse)
      setError(null)
      return
    }
    setLoading(true)
    setError(null)
    try {
      const next = await getWatchlistOutputEvidence(outputId)
      setResponse(next)
    } catch (err) {
      console.error("Failed to load Watchlist report evidence:", err)
      setError(err instanceof Error ? err.message : "evidence_unavailable")
    } finally {
      setLoading(false)
    }
  }, [evidenceResponse, outputId])

  useEffect(() => {
    loadEvidence()
  }, [loadEvidence])

  const snapshot = response?.snapshot ?? null
  const readiness = response?.readiness ?? snapshot?.readiness ?? null
  const readinessWarnings = Array.isArray(readiness?.warnings) ? readiness.warnings : []

  const includedColumns: ColumnsType<WatchlistReportEvidenceItem> = useMemo(
    () => [
      {
        title: t("watchlists:reports.evidence.columns.update", "Update"),
        key: "title",
        render: (_, item) => (
          <div className="space-y-1">
            <div className="font-medium text-text">{item.title || `Update #${item.id}`}</div>
            {item.summary && (
              <div className="text-xs text-text-muted line-clamp-2">{item.summary}</div>
            )}
          </div>
        )
      },
      {
        title: t("watchlists:reports.evidence.columns.source", "Source"),
        key: "source",
        render: (_, item) => item.source_name || `Source #${item.source_id ?? "-"}`
      },
      {
        title: t("watchlists:reports.evidence.columns.published", "Published"),
        dataIndex: "published_at",
        key: "published_at",
        render: (value) => value || "-"
      },
      {
        title: t("watchlists:reports.evidence.columns.alerts", "Alert evidence"),
        key: "alerts",
        render: (_, item) => (
          <div className="flex flex-wrap gap-1">
            {item.alerts.length > 0 ? (
              item.alerts.map((alert) => (
                <Tooltip key={alert.id} title={alert.matched_text || alert.snippet}>
                  <Tag color={alert.severity === "critical" ? "red" : "gold"}>
                    {alert.severity}
                  </Tag>
                </Tooltip>
              ))
            ) : (
              <span className="text-text-subtle">-</span>
            )}
          </div>
        )
      },
      {
        title: t("watchlists:reports.evidence.columns.state", "Review/queue state"),
        key: "state",
        render: (_, item) => (
          <div className="flex flex-wrap gap-1">
            <Tag color={item.reviewed ? "green" : "gold"}>
              {item.reviewed
                ? t("watchlists:reports.evidence.reviewed", "Reviewed")
                : t("watchlists:reports.evidence.needsReview", "Needs review")}
            </Tag>
            {item.queued_for_briefing && (
              <Tag color="blue">{t("watchlists:reports.evidence.queued", "Queued")}</Tag>
            )}
          </div>
        )
      },
      {
        title: t("watchlists:reports.evidence.columns.link", "Link"),
        key: "link",
        render: (_, item) => item.url ? (
          <a href={item.url} target="_blank" rel="noreferrer">
            {t("watchlists:reports.evidence.openSource", "Open source")}
          </a>
        ) : "-"
      }
    ],
    [t]
  )

  const renderAlertEvidence = (item: WatchlistReportEvidenceItem) => (
    <div className="flex flex-wrap gap-1">
      {item.alerts.length > 0 ? (
        item.alerts.map((alert) => (
          <Tooltip key={alert.id} title={alert.matched_text || alert.snippet}>
            <Tag color={alert.severity === "critical" ? "red" : "gold"}>
              {alert.severity}
            </Tag>
          </Tooltip>
        ))
      ) : (
        <span className="text-text-subtle">-</span>
      )}
    </div>
  )

  const renderReviewState = (item: WatchlistReportEvidenceItem) => (
    <div className="flex flex-wrap gap-1">
      <Tag color={item.reviewed ? "green" : "gold"}>
        {item.reviewed
          ? t("watchlists:reports.evidence.reviewed", "Reviewed")
          : t("watchlists:reports.evidence.needsReview", "Needs review")}
      </Tag>
      {item.queued_for_briefing && (
        <Tag color="blue">{t("watchlists:reports.evidence.queued", "Queued")}</Tag>
      )}
    </div>
  )

  const renderConstrainedIncludedEvidence = (items: WatchlistReportEvidenceItem[]) => (
    <div className="space-y-3" data-testid="report-evidence-included-constrained-list">
      {items.map((item) => (
        <article
          key={item.id}
          className="rounded-lg border border-border bg-surface p-3"
          data-testid={`report-evidence-included-card-${item.id}`}
        >
          <div className="space-y-1">
            <div className="font-medium text-text">{item.title || `Update #${item.id}`}</div>
            {item.summary && (
              <div className="text-xs text-text-muted line-clamp-2">{item.summary}</div>
            )}
          </div>

          <div className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
            <div>
              <div className="text-xs font-medium text-text-subtle">
                {t("watchlists:reports.evidence.columns.source", "Source")}
              </div>
              <span className="text-text-muted">
                {item.source_name || `Source #${item.source_id ?? "-"}`}
              </span>
            </div>
            <div>
              <div className="text-xs font-medium text-text-subtle">
                {t("watchlists:reports.evidence.columns.published", "Published")}
              </div>
              <span className="text-text-muted">{item.published_at || "-"}</span>
            </div>
            <div>
              <div className="text-xs font-medium text-text-subtle">
                {t("watchlists:reports.evidence.columns.alerts", "Alert evidence")}
              </div>
              {renderAlertEvidence(item)}
            </div>
            <div>
              <div className="text-xs font-medium text-text-subtle">
                {t("watchlists:reports.evidence.columns.state", "Review/queue state")}
              </div>
              {renderReviewState(item)}
            </div>
          </div>

          {item.url ? (
            <div className="mt-3">
              <a href={item.url} target="_blank" rel="noreferrer" className="text-sm text-primary hover:underline">
                {t("watchlists:reports.evidence.openSource", "Open source")}
              </a>
            </div>
          ) : null}
        </article>
      ))}
    </div>
  )

  if (loading && !response) {
    return (
      <div className="flex items-center justify-center py-6">
        <Spin />
      </div>
    )
  }

  if (error) {
    return (
      <Alert
        type="error"
        showIcon
        title={t("watchlists:reports.evidence.errorTitle", "Evidence snapshot unavailable")}
        description={error}
        action={(
          <Button size="small" onClick={loadEvidence}>
            {t("common:retry", "Retry")}
          </Button>
        )}
      />
    )
  }

  if (!response) {
    return (
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description={t("watchlists:reports.evidence.empty", "No report evidence available")}
      />
    )
  }

  if (!response.immutable_snapshot || !snapshot) {
    const warning = readinessWarnings[0]
    return (
      <Alert
        type="info"
        showIcon
        title={t("watchlists:reports.evidence.legacyTitle", "Live provenance only")}
        description={
          warning?.message ||
          t(
            "watchlists:reports.evidence.legacyDescription",
            "This older report was created before evidence snapshots were available."
          )
        }
      />
    )
  }

  const uniqueSourceCount = Number(snapshot.source_summary?.unique_source_count ?? 0)
  const missingSourceCount = Number(snapshot.source_summary?.missing_source_count ?? 0)

  return (
    <div
      className={compact ? "space-y-3 text-sm" : "space-y-4"}
      data-testid="report-evidence-panel"
    >
      <div className="rounded-lg border border-border bg-surface p-3 space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-medium text-text">
            {t("watchlists:reports.evidence.title", "Evidence snapshot")}
          </span>
          {readiness && (
            <Tag color={getReadinessTagColor(readiness.state)}>
              {getReadinessLabel(readiness.state)}
            </Tag>
          )}
        </div>
        <div className="text-xs text-text-muted">
          {t(
            "watchlists:reports.evidence.capturedAt",
            "Immutable snapshot captured at {{time}}",
            { time: snapshot.generated_at }
          )}
        </div>
        <div className="flex flex-wrap gap-2 text-xs text-text-muted">
          <span>
            {t("watchlists:reports.evidence.uniqueSources", "Unique sources: {{count}}", {
              count: uniqueSourceCount
            })}
          </span>
          <span>
            {t("watchlists:reports.evidence.alertCount", "Alerts: {{count}}", {
              count: snapshot.alert_count
            })}
          </span>
          {missingSourceCount > 0 && (
            <span>
              {t("watchlists:reports.evidence.missingSources", "Missing provenance: {{count}}", {
                count: missingSourceCount
              })}
            </span>
          )}
        </div>
      </div>

      {readinessWarnings.length ? (
        <div className="space-y-2">
          {readinessWarnings.map((warning) => (
            <Alert
              key={`${warning.code}-${warning.message}`}
              type={warning.severity === "blocking" ? "error" : "warning"}
              showIcon
              title={warning.message}
            />
          ))}
        </div>
      ) : null}

      {isConstrained ? (
        renderConstrainedIncludedEvidence(snapshot.included_items)
      ) : (
        <Table
          dataSource={snapshot.included_items}
          columns={includedColumns}
          rowKey="id"
          aria-label={t("watchlists:reports.evidence.tableAria", "Evidence table")}
          size="small"
          pagination={false}
        />
      )}

      <div className="rounded-lg border border-border bg-surface p-3 space-y-2">
        <div className="font-medium text-text">
          {t("watchlists:reports.evidence.excludedTrail", "Excluded trail")}
        </div>
        {snapshot.excluded_items.length > 0 ? (
          <div className="space-y-2">
            {snapshot.excluded_items.map((item: WatchlistReportExcludedItem) => (
              <div
                key={item.id}
                className="flex flex-wrap items-center justify-between gap-2 text-sm"
              >
                <span>{item.title || `Update #${item.id}`}</span>
                <Tag>{formatExcludedReason(item.reason)}</Tag>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-sm text-text-muted">
            {t("watchlists:reports.evidence.noExcluded", "No excluded updates captured.")}
          </div>
        )}
      </div>
    </div>
  )
}
