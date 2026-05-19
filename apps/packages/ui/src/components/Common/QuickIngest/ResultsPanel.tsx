import React from "react"
import { Button, List, Progress, Select, Tag, Typography } from "antd"
import type { TFunction } from "i18next"
import type {
  ResultFilters,
  ResultItem,
  ResultItemWithMediaId,
  ResultSummary,
  ResultsFilter
} from "@/components/Common/QuickIngest/types"
import { ResultsListItem } from "@/components/Common/QuickIngest/ResultsListItem"

type ResultsPanelProps = {
  data: {
    results: ResultItem[]
    visibleResults: ResultItem[]
    resultSummary: ResultSummary | null
    running: boolean
    progressMeta: {
      total: number
      done: number
      pct: number
      elapsedLabel?: string | null
      state?: "running" | "failed" | "complete" | "cancelled" | "ready"
      error?: string | null
    }
    filters: {
      value: ResultsFilter
      options: ResultFilters
      onChange: (value: ResultsFilter) => void
    }
  }
  context: {
    shouldStoreRemote: boolean
    firstResultWithMedia?: ResultItem | null
    reviewBatchId?: string | null
    processOnly: boolean
    mediaIdFromPayload: (payload: unknown) => string | number | null
    titleFromPayload: (payload: unknown) => string | null
  }
  actions: {
    retryFailedUrls: () => void
    requeueFailed: () => void
    exportFailedList: () => void
    tryOpenContentReview: (batchId: string) => void | Promise<void>
    openInMediaViewer: (item: ResultItem) => void
    discussInChat: (item: ResultItem) => void
    downloadJson: (item: ResultItem) => void
    openHealthDiagnostics: () => void
  }
  i18n: {
    qi: (key: string, defaultValue: string, options?: Record<string, any>) => string
    t: TFunction
  }
}

export const ResultsPanel: React.FC<ResultsPanelProps> = ({
  data,
  context,
  actions,
  i18n
}) => {
  const { results, visibleResults, resultSummary, running, filters } = data
  const { progressMeta } = data
  const {
    shouldStoreRemote,
    firstResultWithMedia,
    reviewBatchId,
    processOnly,
    mediaIdFromPayload,
    titleFromPayload
  } = context
  const {
    retryFailedUrls,
    requeueFailed,
    exportFailedList,
    tryOpenContentReview,
    openInMediaViewer,
    discussInChat,
    downloadJson,
    openHealthDiagnostics
  } = actions
  const { qi, t } = i18n

  const resultsWithMediaIds = React.useMemo<ResultItemWithMediaId[]>(() => {
    return results.map((item) => {
      const mediaId =
        item.status === "ok" && shouldStoreRemote
          ? mediaIdFromPayload(item.data)
          : null
      const title =
        item.status === "ok" ? titleFromPayload(item.data) : null

      return {
        ...item,
        mediaId,
        title
      }
    })
  }, [results, shouldStoreRemote, mediaIdFromPayload, titleFromPayload])

  const enrichmentCache = React.useMemo(() => {
    return new Map<string, { mediaId: ResultItemWithMediaId["mediaId"]; title: string | null | undefined }>(
      resultsWithMediaIds.map((item) => [item.id, { mediaId: item.mediaId, title: item.title }])
    )
  }, [resultsWithMediaIds])

  const visibleResultsWithMediaIds = React.useMemo<ResultItemWithMediaId[]>(() => {
    return visibleResults.map((item) => {
      const cached = enrichmentCache.get(item.id)
      return {
        ...item,
        mediaId: cached?.mediaId ?? null,
        title: cached?.title ?? null
      }
    })
  }, [enrichmentCache, visibleResults])
  // Dispatch a custom event when ingest completes so other panels can react
  const ingestCompleteDispatched = React.useRef(false)
  React.useEffect(() => {
    if (resultSummary && !running && !ingestCompleteDispatched.current) {
      ingestCompleteDispatched.current = true
      window.dispatchEvent(new CustomEvent("tldw:quick-ingest-complete", {
        detail: {
          batchId: reviewBatchId ?? null,
          successCount: resultSummary.successCount,
          failCount: resultSummary.failCount
        }
      }))
    }
  }, [resultSummary, running, reviewBatchId])

  const hasErrors = React.useMemo(
    () =>
      results.some((item) => {
        if (item.status !== "error") return false
        if (item.outcome === "cancelled") return false
        const text = String(item.error || "").toLowerCase()
        return !(text.includes("cancelled") || text.includes("canceled"))
      }),
    [results]
  )
  const handleRetryClick = React.useCallback(() => {
    retryFailedUrls()
  }, [retryFailedUrls])
  const handleFilterChange = React.useCallback(
    (value: ResultsFilter | string) => {
      filters.onChange(value as ResultsFilter)
    },
    [filters]
  )
  const handleOpenFirstInMedia = React.useCallback(() => {
    if (!firstResultWithMedia) return
    openInMediaViewer(firstResultWithMedia)
  }, [firstResultWithMedia, openInMediaViewer])
  const handleOpenContentReview = React.useCallback(() => {
    if (!reviewBatchId) return
    void tryOpenContentReview(reviewBatchId)
  }, [reviewBatchId, tryOpenContentReview])
  const renderResultItem = React.useCallback(
    (item: ResultItemWithMediaId) => (
      <ResultsListItem
        item={item}
        processOnly={processOnly}
        onDownloadJson={downloadJson}
        onOpenMedia={openInMediaViewer}
        onDiscussInChat={discussInChat}
        t={t}
      />
    ),
    [processOnly, downloadJson, openInMediaViewer, discussInChat, t]
  )

  const hasResults = results.length > 0
  const hasProgress = progressMeta.total > 0
  const progressState =
    progressMeta.state || (running ? "running" : "ready")
  const progressTitle =
    progressState === "running"
      ? qi("ingestProgressTitle", "Current ingest progress")
      : progressState === "failed"
        ? qi("ingestFailedTitle", "Last ingest run failed")
        : progressState === "cancelled"
          ? qi("ingestCancelledTitle", "Ingest run cancelled")
        : progressState === "complete"
          ? qi("ingestCompleteTitle", "Ingest run completed")
          : qi("itemsReadyTitle", "Items ready to ingest")
  const progressContainerClassName =
    progressState === "failed"
      ? "mb-3 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger"
      : progressState === "cancelled"
        ? "mb-3 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn"
      : "mb-3 rounded-md border border-border bg-surface2 px-3 py-2 text-xs text-text"

  if (!hasResults && !hasProgress) return null

  return (
    <div className="mt-4">
      {running && hasProgress ? (
        <div className="sr-only" aria-live="polite" role="status">
          {t(
            "quickIngest.progress",
            "Processing {{done}} / {{total}} items…",
            {
              done: progressMeta.done,
              total: progressMeta.total
            }
          )}
        </div>
      ) : null}
      {hasProgress && (
        <div className={progressContainerClassName}>
          <div className="flex items-center justify-between">
            <span className="font-medium">{progressTitle}</span>
            {progressMeta.elapsedLabel ? (
              <span>
                {qi("elapsedLabel", "Elapsed {{time}}", {
                  time: progressMeta.elapsedLabel
                })}
              </span>
            ) : null}
          </div>
          {progressState === "failed" && progressMeta.error ? (
            <div className="mt-1">{progressMeta.error}</div>
          ) : null}
          <Progress percent={progressMeta.pct} showInfo={false} size="small" />
          <div className="flex justify-between text-xs text-text-muted mt-1">
            <span>
              {qi("processedCount", "{{done}}/{{total}} processed", {
                done: progressMeta.done,
                total: progressMeta.total
              })}
            </span>
          </div>
        </div>
      )}
      {hasResults ? (
        <>
          <div className="flex items-center justify-between">
            <Typography.Title level={5} className="!mb-0">
              {t("quickIngest.results") || "Results"}
            </Typography.Title>
            <div className="flex items-center gap-2 text-xs">
              <Tag color="blue">
                {qi("resultsCount", "{{count}} item(s)", {
                  count: results.length
                })}
              </Tag>
              <Button
                size="small"
                onClick={handleRetryClick}
                disabled={!hasErrors}
              >
                {qi("retryFailedUrls", "Retry failed URLs")}
              </Button>
              <Select
                size="small"
                className="w-32"
                aria-label={t(
                  "quickIngest.resultsFilterAria",
                  "Filter results by status"
                )}
                value={filters.value}
                onChange={handleFilterChange}
                options={[
                  {
                    value: filters.options.ALL,
                    label: t("quickIngest.resultsFilterAll", "All")
                  },
                  {
                    value: filters.options.ERROR,
                    label: t("quickIngest.resultsFilterFailed", "Failed only")
                  },
                  {
                    value: filters.options.SUCCESS,
                    label: t("quickIngest.resultsFilterSucceeded", "Succeeded only")
                  }
                ]}
              />
            </div>
          </div>
          {resultSummary && !running && (
        <div
          className="mt-2 rounded-md border border-border bg-surface2 px-3 py-2 text-xs text-text"
          data-testid="quick-ingest-complete"
        >
          <div className="font-medium">
            {resultSummary.cancelledCount > 0
              ? t(
                  "quickIngest.summaryCancelled",
                  "Quick ingest was cancelled."
                )
              : resultSummary.failCount === 0
              ? t(
                  "quickIngest.summaryAllSucceeded",
                  "Quick ingest completed successfully."
                )
              : t(
                  "quickIngest.summarySomeFailed",
                  "Quick ingest completed with some errors."
                )}
          </div>
          <div className="mt-1">
            {resultSummary.cancelledCount > 0
              ? t(
                  "quickIngest.summaryCountsWithCancelled",
                  "{{success}} succeeded \u00b7 {{failed}} failed \u00b7 {{cancelled}} cancelled",
                  {
                    success: resultSummary.successCount,
                    failed: resultSummary.failCount,
                    cancelled: resultSummary.cancelledCount
                  }
                )
              : t("quickIngest.summaryCounts", "{{success}} succeeded \u00b7 {{failed}} failed", {
                  success: resultSummary.successCount,
                  failed: resultSummary.failCount
                })}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            {shouldStoreRemote && firstResultWithMedia && (
              <Button
                size="small"
                type="primary"
                data-testid="quick-ingest-open-media-primary"
                aria-label={t(
                  "quickIngest.openFirstInMediaAria",
                  "Open first result in media viewer"
                )}
                onClick={handleOpenFirstInMedia}
              >
                {t("quickIngest.openFirstInMedia", "Open in Media viewer")}
              </Button>
            )}
            {reviewBatchId ? (
              <Button
                size="small"
                type="primary"
                aria-label={qi(
                  "openContentReviewAria",
                  "Open Content Review for batch"
                )}
                onClick={handleOpenContentReview}
              >
                {qi("openContentReview", "Open Content Review")}
              </Button>
            ) : null}
            {resultSummary.failCount > 0 && (
              <>
                <Button
                  size="small"
                  onClick={handleRetryClick}
                  aria-label={qi("retryFailedUrlsAria", "Retry all failed URLs")}
                >
                  {qi("retryFailedUrls", "Retry failed URLs")}
                </Button>
                <Button
                  size="small"
                  onClick={requeueFailed}
                  aria-label={qi("requeueFailedAria", "Requeue failed items")}
                >
                  {qi("requeueFailed", "Requeue failed")}
                </Button>
                <Button
                  size="small"
                  onClick={exportFailedList}
                  aria-label={qi("exportFailedListAria", "Export failed items list")}
                >
                  {qi("exportFailedList", "Export failed list")}
                </Button>
              </>
            )}
            <Button
              size="small"
              type="default"
              onClick={openHealthDiagnostics}
              aria-label={t(
                "settings:healthSummary.diagnosticsAria",
                "Open health and diagnostics"
              )}
            >
              {t("settings:healthSummary.diagnostics", "Health & diagnostics")}
            </Button>
          </div>
        </div>
          )}
          <List
            size="small"
            dataSource={visibleResultsWithMediaIds}
            renderItem={renderResultItem}
          />
        </>
      ) : null}
    </div>
  )
}

export default ResultsPanel
