import React from "react"
import { Alert, Button, Checkbox, List, Radio, Tag, Typography } from "antd"
import { ListVideo, Plus, RefreshCw } from "lucide-react"
import type { PlaylistPreflightResult } from "@/services/tldw/playlist-preflight"
import type { ConferenceDuplicatePolicy } from "./types"

const isDuplicatePreflightStatus = (status: string | undefined): boolean =>
  status === "duplicate_existing" || status === "duplicate_in_batch"

type PlaylistPreflightPanelProps = {
  candidateUrl: string
  loading?: boolean
  error?: string | null
  result?: PlaylistPreflightResult | null
  duplicatePolicy?: ConferenceDuplicatePolicy
  onPreview: () => void
  onAddItems: () => void
  onItemSelectionChange?: (ordinal: number, selected: boolean) => void
  onDuplicatePolicyChange?: (policy: ConferenceDuplicatePolicy) => void
}

export const PlaylistPreflightPanel: React.FC<PlaylistPreflightPanelProps> = ({
  candidateUrl,
  loading = false,
  error = null,
  result = null,
  duplicatePolicy = "skip",
  onPreview,
  onAddItems,
  onItemSelectionChange,
  onDuplicatePolicyChange
}) => {
  const selectedCount =
    result?.items.filter((item) => item.selected && item.sourceUrl).length ?? 0
  const duplicateCount =
    result?.duplicateCount ??
    result?.items.filter((item) => isDuplicatePreflightStatus(item.duplicateStatus)).length ??
    0

  return (
    <div className="rounded-md border border-border bg-surface px-3 py-2">
      <div className="flex items-start gap-2">
        <ListVideo className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <Typography.Text className="block text-sm font-medium">
            {result?.playlistTitle || "Playlist detected"}
          </Typography.Text>
          <Typography.Text className="block truncate text-[11px] text-text-muted">
            {candidateUrl}
          </Typography.Text>
        </div>
        <Button
          size="small"
          type={result ? "default" : "primary"}
          loading={loading}
          onClick={onPreview}
          icon={<RefreshCw className="h-3.5 w-3.5" />}
        >
          {result ? "Refresh" : "Preview"}
        </Button>
      </div>

      {error && (
        <Alert
          className="mt-2"
          type="warning"
          showIcon
          message={error}
        />
      )}

      {result && (
        <div className="mt-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <Tag color="blue">{result.itemCount} items</Tag>
            <Tag color="green">{selectedCount} selected</Tag>
            {duplicateCount > 0 && <Tag color="warning">{duplicateCount} duplicates</Tag>}
          </div>

          {duplicateCount > 0 && (
            <div className="mt-2 rounded border border-amber-500/20 bg-amber-500/5 px-2 py-1.5">
              <Typography.Text className="block text-[11px] font-medium text-text-muted">
                Duplicate policy
              </Typography.Text>
              <Radio.Group
                className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs"
                value={duplicatePolicy}
                onChange={(event) =>
                  onDuplicatePolicyChange?.(
                    event.target.value as ConferenceDuplicatePolicy
                  )
                }
              >
                <Radio value="skip">Skip duplicates</Radio>
                <Radio value="overwrite">Overwrite</Radio>
                <Radio value="update_metadata_only">Update metadata only</Radio>
                <Radio value="include_existing">Include existing</Radio>
              </Radio.Group>
            </div>
          )}

          <List
            className="mt-2 max-h-36 overflow-y-auto rounded border border-border"
            size="small"
            dataSource={result.items}
            renderItem={(item) => (
              <List.Item className="!px-2 !py-1">
                <Checkbox
                  className="mr-2"
                  checked={item.selected}
                  disabled={!item.sourceUrl}
                  aria-label={`Select ${item.title}`}
                  onChange={(event) =>
                    onItemSelectionChange?.(item.ordinal, event.target.checked)
                  }
                />
                <div className="min-w-0 flex-1">
                  <Typography.Text className="block truncate text-xs">
                    {item.ordinal}. {item.title}
                  </Typography.Text>
                  <Typography.Text className="block truncate text-[11px] text-text-muted">
                    {item.sourceUrl}
                  </Typography.Text>
                </div>
                {isDuplicatePreflightStatus(item.duplicateStatus) && (
                  <Tag color="warning" className="!mr-0">
                    duplicate
                  </Tag>
                )}
              </List.Item>
            )}
          />

          <div className="mt-2 flex justify-end">
            <Button
              size="small"
              type="primary"
              onClick={onAddItems}
              disabled={selectedCount === 0}
              icon={<Plus className="h-3.5 w-3.5" />}
            >
              Add {selectedCount}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

export default PlaylistPreflightPanel
