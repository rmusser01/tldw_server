import React from "react"
import { Tag, Tooltip } from "antd"
import type { ColumnsType } from "antd/es/table"
import { MessageCircle, Star } from "lucide-react"
import { SyncStatusBadge } from "./SyncStatusBadge"
import type { PromptListSortKey, PromptListSortOrder, PromptRowVM } from "./prompt-workspace-types"

export type PromptTableColumnLabels = {
  title: string
  preview: string
  tags: string
  updated: string
  lastUsed: string
  status: string
  actions: string
  author: string
  system: string
  user: string
  unknown: string
  offlineStatus: string
  edit: string
}

export type PromptTableColumnOptions = {
  isOnline: boolean
  isCompactViewport: boolean
  sortKey: PromptListSortKey
  sortOrder: PromptListSortOrder
  onToggleFavorite?: (row: PromptRowVM, nextFavorite: boolean) => void
  onEdit?: (row: PromptRowVM) => void
  onOpenConflictResolution?: (row: PromptRowVM) => void
  canRetrySync?: (row: PromptRowVM) => boolean
  onRetrySync?: (row: PromptRowVM) => void
  renderActions?: (row: PromptRowVM) => React.ReactNode
  renderTitleMeta?: (row: PromptRowVM) => React.ReactNode
  favoriteButtonTestId?: (row: PromptRowVM) => string
  labels?: Partial<PromptTableColumnLabels>
  formatRelativeTime?: (timestamp?: number) => string
}

const defaultFormatRelativeTime = (timestamp?: number) => {
  if (!timestamp) return "Unknown"
  const now = Date.now()
  const diffMs = Math.max(0, now - timestamp)
  const diffMinutes = Math.floor(diffMs / (1000 * 60))
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))
  if (diffMinutes < 1) return "Just now"
  if (diffMinutes < 60) return `${diffMinutes}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 30) return `${diffDays}d ago`
  return new Date(timestamp).toLocaleDateString()
}

const renderPreview = (
  record: PromptRowVM,
  isCompactViewport: boolean,
  labels: Pick<PromptTableColumnLabels, "system" | "user">
) => {
  const clampClass = isCompactViewport ? "line-clamp-1" : "line-clamp-2"
  const labelClassName =
    "inline-flex items-center rounded-full border border-border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-text-muted"
  return (
    <div className={`flex flex-col gap-1 ${isCompactViewport ? "max-w-[14rem]" : "max-w-[24rem]"}`}>
      {record.previewSystem ? (
        <div className="flex items-start gap-2">
          <span className={labelClassName}>{labels.system}</span>
          <span className={`${clampClass} text-sm text-text-muted`}>
            {record.previewSystem}
          </span>
        </div>
      ) : null}
      {record.previewUser ? (
        <div className="flex items-start gap-2">
          <span className={labelClassName}>{labels.user}</span>
          <span className={`${clampClass} text-sm text-text-muted`}>
            {record.previewUser}
          </span>
        </div>
      ) : null}
    </div>
  )
}

const MAX_VISIBLE_KEYWORDS = 2

const renderKeywordChips = (
  keywords: string[],
  promptId: string
) => {
  if (keywords.length === 0) return null

  const visible = keywords.slice(0, MAX_VISIBLE_KEYWORDS)
  const hidden = keywords.slice(MAX_VISIBLE_KEYWORDS)

  return (
    <div className="flex min-w-0 flex-wrap items-center gap-1">
      {visible.map((keyword) => (
        <Tag key={`${promptId}-${keyword}`}>{keyword}</Tag>
      ))}
      {hidden.length > 0 && (
        <Tooltip title={hidden.join(", ")}>
          <span className="cursor-help rounded-sm text-xs text-text-subtle underline-offset-2">
            +{hidden.length}
          </span>
        </Tooltip>
      )}
    </div>
  )
}

export const buildPromptTableColumns = (
  options: PromptTableColumnOptions
): ColumnsType<PromptRowVM> => {
  const {
    isOnline,
    isCompactViewport,
    sortKey,
    sortOrder,
    onToggleFavorite,
    onEdit,
    onOpenConflictResolution,
    canRetrySync,
    onRetrySync,
    renderActions,
    renderTitleMeta,
    favoriteButtonTestId,
    labels,
    formatRelativeTime = defaultFormatRelativeTime
  } = options

  const resolvedLabels: PromptTableColumnLabels = {
    title: labels?.title || "Title",
    preview: labels?.preview || "Preview",
    tags: labels?.tags || "Tags",
    updated: labels?.updated || "Updated",
    lastUsed: labels?.lastUsed || "Last used",
    status: labels?.status || "Status",
    actions: labels?.actions || "Actions",
    author: labels?.author || "Author",
    system: labels?.system || "System",
    user: labels?.user || "User",
    unknown: labels?.unknown || "Unknown",
    offlineStatus:
      labels?.offlineStatus ||
      "Sync unavailable while offline. Showing last known state.",
    edit: labels?.edit || "Edit"
  }

  const columns: ColumnsType<PromptRowVM> = [
    {
      title: "",
      key: "favorite",
      dataIndex: "favorite",
      width: 48,
      render: (_value, record) => (
        <button
          type="button"
          aria-label={record.favorite ? "Unfavorite prompt" : "Favorite prompt"}
          aria-pressed={record.favorite}
          data-testid={favoriteButtonTestId?.(record)}
          onClick={(event) => {
            event.preventDefault()
            event.stopPropagation()
            onToggleFavorite?.(record, !record.favorite)
          }}
          className={`inline-flex min-h-10 min-w-10 items-center justify-center rounded-md border border-transparent p-1.5 transition motion-reduce:transition-none focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-1 focus:ring-offset-bg ${
            record.favorite
              ? "text-primary hover:border-primary/30 hover:bg-primary/10"
              : "text-text-muted hover:border-border hover:bg-surface2"
          }`}
        >
          <Star className={`size-4 ${record.favorite ? "fill-current" : ""}`} />
        </button>
      )
    },
    {
      title: resolvedLabels.title,
      dataIndex: "title",
      key: "title",
      width: 360,
      sorter: true,
      sortOrder: sortKey === "title" ? sortOrder || undefined : undefined,
      render: (_value, record) => (
        <div className="flex max-w-72 flex-col gap-1">
          <div className="flex min-w-0 items-start gap-2">
            <span className="line-clamp-1 min-w-0 flex-1 font-medium text-text">
              {record.title}
            </span>
            {record.usageCount > 0 && (
              <Tooltip
                title={`${record.usageCount} ${
                  record.usageCount === 1 ? "use" : "uses"
                }`}
              >
                <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                  <MessageCircle className="size-3" />
                  {record.usageCount > 99 ? "99+" : record.usageCount}
                </span>
              </Tooltip>
            )}
          </div>
          {record.author ? (
            <span className="line-clamp-1 text-xs text-text-muted">
              {resolvedLabels.author}: {record.author}
            </span>
          ) : null}
          {record.details ? (
            <span className="line-clamp-2 text-xs text-text-muted">{record.details}</span>
          ) : null}
          {renderTitleMeta?.(record)}
        </div>
      )
    },
    {
      title: resolvedLabels.preview,
      key: "preview",
      width: 320,
      render: (_value, record) => renderPreview(record, isCompactViewport, resolvedLabels)
    },
    ...(!isCompactViewport
      ? [
          {
            title: resolvedLabels.tags,
            key: "keywords",
            width: 220,
            render: (_value: unknown, record: PromptRowVM) => (
              <div className="max-w-64">
                {renderKeywordChips(record.keywords || [], record.id)}
              </div>
            )
          },
          {
            title: resolvedLabels.updated,
            key: "modifiedAt",
            width: 180,
            sorter: true,
            sortOrder: sortKey === "modifiedAt" ? sortOrder || undefined : undefined,
            render: (_value: unknown, record: PromptRowVM) => (
              <Tooltip
                title={
                  record.updatedAt
                    ? new Date(record.updatedAt).toLocaleString()
                    : resolvedLabels.unknown
                }
              >
                <div className="min-w-0">
                  <div className="text-xs font-medium text-text">
                    {formatRelativeTime(record.updatedAt)}
                  </div>
                  <div className="line-clamp-1 text-[11px] text-text-subtle">
                    {resolvedLabels.lastUsed}:{" "}
                    {record.lastUsedAt
                      ? formatRelativeTime(record.lastUsedAt)
                      : resolvedLabels.unknown}
                  </div>
                </div>
              </Tooltip>
            )
          },
          {
            title: resolvedLabels.status,
            key: "syncStatus",
            width: 124,
            render: (_value: unknown, record: PromptRowVM) => (
              <Tooltip
                title={
                  !isOnline
                    ? resolvedLabels.offlineStatus
                    : undefined
                }
              >
                <div className={!isOnline ? "opacity-60" : undefined}>
                  <SyncStatusBadge
                    syncStatus={record.syncStatus}
                    sourceSystem={record.sourceSystem}
                    serverId={record.serverId}
                    compact={false}
                    onClick={
                      record.syncStatus === "conflict"
                        ? () => onOpenConflictResolution?.(record)
                        : undefined
                    }
                    onRetry={
                      isOnline &&
                      onRetrySync &&
                      (canRetrySync?.(record) ?? false)
                        ? () => onRetrySync(record)
                        : undefined
                    }
                  />
                </div>
              </Tooltip>
            )
          }
        ]
      : []),
    {
      title: resolvedLabels.actions,
      key: "actions",
      width: 210,
      render: (_value, record) =>
        renderActions ? (
          renderActions(record)
        ) : (
          <button
            type="button"
            className="rounded border border-border px-2 py-1 text-xs text-text hover:bg-surface2"
            onClick={(event) => {
              event.preventDefault()
              event.stopPropagation()
              onEdit?.(record)
            }}
          >
            {resolvedLabels.edit}
          </button>
        )
    }
  ]

  return columns
}
