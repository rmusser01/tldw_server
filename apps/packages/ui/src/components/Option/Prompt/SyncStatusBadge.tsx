import React from "react"
import { Tag, Tooltip } from "antd"
import { Cloud, CloudOff, AlertTriangle, RefreshCw, HardDrive } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { PromptSyncStatus, PromptSourceSystem } from "@/db/dexie/types"

interface SyncStatusBadgeProps {
  syncStatus?: PromptSyncStatus
  sourceSystem?: PromptSourceSystem
  serverId?: number | null
  lastSyncedAt?: number | null
  compact?: boolean
  onClick?: () => void
  onRetry?: () => void
}

export const SyncStatusBadge: React.FC<SyncStatusBadgeProps> = ({
  syncStatus = "local",
  sourceSystem = "workspace",
  serverId,
  lastSyncedAt,
  compact = false,
  onClick,
  onRetry
}) => {
  const { t } = useTranslation(["settings", "common"])
  const isInteractive = typeof onClick === "function"

  const formatLastSync = (timestamp: number | null | undefined) => {
    if (!timestamp) return null
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / (1000 * 60))
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

    if (diffMins < 1) return t("common:justNow", "Just now")
    if (diffMins < 60) return t("common:minutesAgo", "{{count}}m ago", { count: diffMins })
    if (diffHours < 24) return t("common:hoursAgo", "{{count}}h ago", { count: diffHours })
    return t("common:daysAgo", "{{count}}d ago", { count: diffDays })
  }

  const getStatusConfig = () => {
    switch (syncStatus) {
      case "synced":
        return {
          icon: <Cloud className="size-3" />,
          color: "green",
          label: t("settings:managePrompts.sync.synced", "Synced"),
          tooltip: t("settings:managePrompts.sync.syncedTooltip", "Synced with server{{time}}", {
            time: lastSyncedAt ? ` (${formatLastSync(lastSyncedAt)})` : ""
          })
        }
      case "pending":
        return {
          icon: <RefreshCw className="size-3" />,
          color: "gold",
          label: t("settings:managePrompts.sync.pending", "Pending"),
          tooltip: t("settings:managePrompts.sync.pendingTooltip", "Local changes not yet synced")
        }
      case "conflict":
        return {
          icon: <AlertTriangle className="size-3" />,
          color: "red",
          label: t("settings:managePrompts.sync.conflict", "Conflict"),
          tooltip: t("settings:managePrompts.sync.conflictTooltip", "Local and server versions differ")
        }
      case "local":
      default:
        return {
          icon: <HardDrive className="size-3" />,
          color: "default",
          label: t("settings:managePrompts.sync.local", "Local"),
          tooltip: t("settings:managePrompts.sync.localTooltip", "Stored locally only")
        }
    }
  }

  const getSourceConfig = () => {
    switch (sourceSystem) {
      case "studio":
        return {
          label: t("settings:managePrompts.source.studio", "Studio"),
          tooltip: t("settings:managePrompts.source.studioTooltip", "Created in Prompt Studio")
        }
      case "copilot":
        return {
          label: t("settings:managePrompts.source.copilot", "Copilot"),
          tooltip: t("settings:managePrompts.source.copilotTooltip", "From server Copilot prompts")
        }
      case "workspace":
      default:
        return {
          label: t("settings:managePrompts.source.workspace", "Workspace"),
          tooltip: t("settings:managePrompts.source.workspaceTooltip", "Created in Prompts Workspace")
        }
    }
  }

  const statusConfig = getStatusConfig()
  const sourceConfig = getSourceConfig()
  const showRetry = syncStatus === "pending" && typeof onRetry === "function"

  const retryButton = showRetry ? (
    <Tooltip title={t("settings:managePrompts.sync.retryTooltip", { defaultValue: "Retry sync" })}>
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation()
          onRetry?.()
        }}
        className="inline-flex items-center text-primary hover:underline focus:outline-none"
        aria-label={t("settings:managePrompts.sync.retry", { defaultValue: "Retry sync" })}
        data-testid="sync-retry-button"
      >
        <RefreshCw className="size-3" />
      </button>
    </Tooltip>
  ) : null

  if (compact) {
    return (
      <Tooltip title={`${statusConfig.tooltip} | ${sourceConfig.tooltip}`}>
        <span className="inline-flex items-center gap-1">
          {isInteractive ? (
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                onClick?.()
              }}
              className="inline-flex items-center gap-1 rounded p-0.5 text-text-muted hover:text-text focus:outline-none focus:ring-2 focus:ring-primary"
              aria-label={t("settings:managePrompts.sync.resolveConflict", {
                defaultValue: "Resolve conflict"
              })}
            >
              {statusConfig.icon}
            </button>
          ) : (
            <span className="inline-flex items-center gap-1 text-text-muted">
              {statusConfig.icon}
            </span>
          )}
          {retryButton}
        </span>
      </Tooltip>
    )
  }

  return (
    <div className="inline-flex items-center gap-1">
      <Tooltip title={statusConfig.tooltip}>
        {isInteractive ? (
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              onClick?.()
            }}
            className="rounded focus:outline-none focus:ring-2 focus:ring-primary"
            aria-label={t("settings:managePrompts.sync.resolveConflict", {
              defaultValue: "Resolve conflict"
            })}
          >
            <Tag
              color={statusConfig.color}
              className="inline-flex items-center gap-1 text-xs cursor-pointer"
            >
              {statusConfig.icon}
              {statusConfig.label}
            </Tag>
          </button>
        ) : (
          <Tag
            color={statusConfig.color}
            className="inline-flex items-center gap-1 text-xs"
          >
            {statusConfig.icon}
            {statusConfig.label}
          </Tag>
        )}
      </Tooltip>
      {retryButton}
      {serverId && (
        <Tooltip title={`Server ID: ${serverId}`}>
          <span className="text-xs text-text-muted">
            #{serverId}
          </span>
        </Tooltip>
      )}
    </div>
  )
}

export default SyncStatusBadge
