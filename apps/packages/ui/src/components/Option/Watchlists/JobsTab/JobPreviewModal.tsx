import React, { useEffect, useLayoutEffect, useRef, useState } from "react"
import { Modal, Spin, Table, Tag } from "antd"
import type { ColumnsType } from "antd/es/table"
import { useTranslation } from "react-i18next"
import { previewWatchlistJob } from "@/services/watchlists"
import type { JobPreviewResult, PreviewItem, WatchlistJob } from "@/types/watchlists"
import { buildWatchlistsModalChrome, useWatchlistsViewport } from "../shared"
import {
  getFocusableActiveElement,
  restoreFocusToElement
} from "../shared/focus-management"

interface JobPreviewModalProps {
  job: WatchlistJob | null
  open: boolean
  onClose: () => void
}

export const JobPreviewModal: React.FC<JobPreviewModalProps> = ({
  job,
  open,
  onClose
}) => {
  const { t } = useTranslation(["watchlists", "common"])
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState<JobPreviewResult | null>(null)
  const restoreFocusTargetRef = useRef<HTMLElement | null>(null)
  const wasOpenRef = useRef(false)
  const { isConstrained } = useWatchlistsViewport()
  const modalChrome = buildWatchlistsModalChrome(isConstrained, 800)

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
    if (!open || !job) {
      setPreview(null)
      return
    }
    setLoading(true)
    previewWatchlistJob(job.id, { limit: 50, per_source: 10 })
      .then((result) => setPreview(result))
      .catch((err) => {
        console.error("Failed to preview job:", err)
      })
      .finally(() => setLoading(false))
  }, [open, job])

  const columns: ColumnsType<PreviewItem> = [
    {
      title: t("watchlists:jobs.preview.columns.title", "Title"),
      dataIndex: "title",
      key: "title",
      ellipsis: true,
      render: (title: string | null, record) => title || record.url || "-"
    },
    {
      title: t("watchlists:jobs.preview.columns.decision", "Decision"),
      dataIndex: "decision",
      key: "decision",
      width: 120,
      render: (decision: string) => (
        <Tag color={decision === "ingest" ? "green" : "red"}>{decision}</Tag>
      )
    },
    {
      title: t("watchlists:jobs.preview.columns.action", "Action"),
      dataIndex: "matched_action",
      key: "matched_action",
      width: 120,
      render: (action: string | null | undefined) => action ? <Tag>{action}</Tag> : "-"
    },
    {
      title: t("watchlists:jobs.preview.columns.reason", "Reason"),
      key: "reason",
      width: 220,
      render: (_, record) => {
        if (!record.matched_filter_key && !record.matched_filter_type && record.matched_filter_id == null) {
          return "-"
        }
        const reasonType = record.matched_filter_type ? String(record.matched_filter_type) : "filter"
        const reasonId = record.matched_filter_id != null ? `#${record.matched_filter_id}` : ""
        const reasonKey = record.matched_filter_key ?? ""
        return [reasonType, reasonId, reasonKey].filter(Boolean).join(" ")
      }
    },
    {
      title: t("watchlists:jobs.preview.columns.source", "Source"),
      dataIndex: "source_id",
      key: "source_id",
      width: 100,
      render: (sourceId: number) => `#${sourceId}`
    }
  ]

  const renderPreviewItems = () => {
    const items = preview?.items || []
    if (isConstrained) {
      return (
        <div className="space-y-2" data-testid="job-preview-constrained-list">
          {items.map((item, index) => {
            const title = item.title || item.url || "-"
            const reasonType = item.matched_filter_type
              ? String(item.matched_filter_type)
              : "filter"
            const reasonId = item.matched_filter_id != null ? `#${item.matched_filter_id}` : ""
            const reasonKey = item.matched_filter_key ?? ""
            const reason = [reasonType, reasonId, reasonKey].filter(Boolean).join(" ")
            return (
              <div
                key={`${item.source_id}-${item.url ?? index}`}
                className="rounded-md border border-border bg-surface p-3"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="line-clamp-2 text-sm font-medium">{title}</div>
                    {item.url ? (
                      <div className="mt-1 break-all text-xs text-text-muted">
                        {item.url}
                      </div>
                    ) : null}
                  </div>
                  <Tag color={item.decision === "ingest" ? "green" : "red"}>
                    {item.decision}
                  </Tag>
                </div>
                <div className="mt-2 flex flex-wrap gap-2 text-xs text-text-muted">
                  <span>{t("watchlists:jobs.preview.columns.source", "Source")}: #{item.source_id}</span>
                  {item.matched_action ? <Tag>{item.matched_action}</Tag> : null}
                  {reason ? <span>{reason}</span> : null}
                </div>
              </div>
            )
          })}
        </div>
      )
    }

    return (
      <Table
        dataSource={items}
        columns={columns}
        rowKey={(item) => `${item.source_id}-${item.url ?? ""}`}
        pagination={false}
        size="small"
      />
    )
  }

  return (
    <Modal
      title={t("watchlists:jobs.preview.title", "Monitor Preview")}
      open={open}
      onCancel={onClose}
      footer={null}
      data-testid="job-preview-modal"
      width={modalChrome.width}
      style={modalChrome.style}
      styles={modalChrome.styles}
    >
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Spin size="large" />
        </div>
      ) : preview ? (
        <div className="space-y-4">
          <div className="text-sm text-text-muted">
            {t(
              "watchlists:jobs.preview.summary",
              "{{total}} candidates: {{ingestable}} ingestable, {{filtered}} filtered",
              {
                total: preview.total,
                ingestable: preview.ingestable,
                filtered: preview.filtered
              }
            )}
          </div>
          {renderPreviewItems()}
        </div>
      ) : (
        <div className="text-center text-sm text-text-muted py-8">
          {t("watchlists:jobs.preview.empty", "No preview data available")}
        </div>
      )}
    </Modal>
  )
}
