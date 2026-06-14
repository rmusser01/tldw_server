import React from "react"
import { Input, Button, Select } from "antd"
import type { MediaReviewState, MediaReviewActions } from "@/components/Review/media-review-types"
import type { MediaMultiBatchExportFormat } from "@/components/Review/media-multi-batch-actions"

interface MediaReviewBatchBarProps {
  state: MediaReviewState
  actions: MediaReviewActions
}

export const MediaReviewBatchBar: React.FC<MediaReviewBatchBarProps> = ({ state, actions }) => {
  const {
    t,
    selectedIds,
    batchKeywordsDraft, setBatchKeywordsDraft,
    batchExportFormat, setBatchExportFormat,
    batchActionLoading
  } = state

  const {
    handleBatchAddTags,
    handleBatchExport,
    handleBatchReprocess,
    handleBatchMoveToTrash
  } = actions

  if (selectedIds.length === 0) return null

  return (
    <div
      className="mb-2 flex flex-wrap items-center gap-2 rounded border border-border bg-surface2/40 px-2 py-2"
      data-testid="media-multi-batch-toolbar"
    >
      <span className="text-xs font-medium text-text-muted">
        {t("mediaPage.batchToolbarSelected", "{{count}} selected", {
          count: selectedIds.length
        })}
      </span>
      <Input
        data-testid="media-multi-batch-keywords"
        value={batchKeywordsDraft}
        onChange={(event) => setBatchKeywordsDraft(event.target.value)}
        placeholder={t(
          "mediaPage.batchKeywordsPlaceholder",
          "Batch keywords (comma-separated)"
        )}
        aria-label={
          t(
            "mediaPage.batchKeywordsLabel",
            "Batch keywords"
          ) as string
        }
        className="min-w-[14rem] max-w-[22rem]"
      />
      <Button
        data-testid="media-multi-batch-add-tags"
        size="small"
        onClick={() => { void handleBatchAddTags() }}
        disabled={batchActionLoading != null && batchActionLoading !== "keywords"}
      >
        {t("mediaPage.batchAddTags", "Add tags")}
      </Button>
      <Select
        data-testid="media-multi-batch-export-format"
        value={batchExportFormat}
        aria-label={t("mediaPage.batchExportFormat", "Export format") as string}
        className="min-w-[10rem]"
        onChange={(value) => setBatchExportFormat(value as MediaMultiBatchExportFormat)}
        options={[
          { value: "json", label: t("mediaPage.batchExportJson", "JSON") },
          { value: "markdown", label: t("mediaPage.batchExportMarkdown", "Markdown") },
          { value: "text", label: t("mediaPage.batchExportText", "Text") }
        ]}
      />
      <Button
        data-testid="media-multi-batch-export"
        size="small"
        onClick={handleBatchExport}
        disabled={batchActionLoading != null && batchActionLoading !== "export"}
      >
        {t("mediaPage.batchExportSelected", "Export selected")}
      </Button>
      <Button
        data-testid="media-multi-batch-reprocess"
        size="small"
        onClick={() => { void handleBatchReprocess() }}
        disabled={batchActionLoading != null && batchActionLoading !== "reprocess"}
      >
        {t("mediaPage.batchReprocess", "Reprocess")}
      </Button>
      <Button
        data-testid="media-multi-batch-trash"
        size="small"
        danger
        className="ml-auto"
        onClick={() => { void handleBatchMoveToTrash() }}
        disabled={batchActionLoading != null && batchActionLoading !== "trash"}
      >
        {t("mediaPage.batchTrashAction", "Move to trash")}
      </Button>
    </div>
  )
}
