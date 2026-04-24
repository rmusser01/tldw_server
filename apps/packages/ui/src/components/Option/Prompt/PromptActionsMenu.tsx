import { Dropdown, MenuProps, Tooltip } from "antd"
import {
  MoreHorizontal,
  Pen,
  MessageCircle,
  CopyIcon,
  Trash2,
  CloudUpload,
  CloudDownload,
  RefreshCw,
  Unlink,
  AlertTriangle,
  Link,
  Play,
  History
} from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import type { PromptSyncStatus } from "@/db/dexie/types"

interface PromptActionsMenuProps {
  promptId: string
  disabled?: boolean
  syncStatus?: PromptSyncStatus
  serverId?: number | null
  inlineUseInChat?: boolean
  onEdit: () => void
  onDuplicate: () => void
  onUseInChat: () => void
  onQuickTest?: () => void
  onDelete: () => void
  onShareLink?: () => void
  onRetrySync?: () => void
  onPushToServer?: () => void
  onPullFromServer?: () => void
  onUnlink?: () => void
  onResolveConflict?: () => void
  onViewHistory?: () => void
}

export const PromptActionsMenu: React.FC<PromptActionsMenuProps> = ({
  promptId,
  disabled = false,
  syncStatus,
  serverId,
  inlineUseInChat = true,
  onEdit,
  onDuplicate,
  onUseInChat,
  onQuickTest,
  onDelete,
  onShareLink,
  onRetrySync,
  onPushToServer,
  onPullFromServer,
  onUnlink,
  onResolveConflict,
  onViewHistory
}) => {
  const { t } = useTranslation(["settings", "common", "option"])

  const isSynced = !!serverId || syncStatus === "synced"
  const isConflict = syncStatus === "conflict"
  const canSync = !disabled && (onPushToServer || onPullFromServer)

  const syncItems: MenuProps["items"] = canSync ? [
    ...(onResolveConflict && isConflict ? [{
      key: "resolveConflict",
      label: t("managePrompts.sync.resolveConflict", {
        defaultValue: "Resolve conflict"
      }),
      icon: <AlertTriangle className="size-4" />,
      onClick: onResolveConflict
    }] : []),
    ...(onResolveConflict && isConflict ? [{ type: "divider" as const }] : []),
    // Retry sync option (for pending prompts)
    ...(onRetrySync ? [{
      key: "retrySync",
      label: (
        <span
          data-testid="menu-item-retrySync"
          className="inline-flex items-center gap-1.5"
        >
          {t("managePrompts.sync.retrySync", { defaultValue: "Retry sync" })}
        </span>
      ),
      icon: <RefreshCw className="size-4" />,
      onClick: onRetrySync
    }] : []),
    // Push to server option (for local or pending prompts)
    ...(onPushToServer && !isSynced ? [{
      key: "push",
      label: t("managePrompts.sync.pushToServer", { defaultValue: "Push to Server" }),
      icon: <CloudUpload className="size-4" />,
      onClick: onPushToServer
    }] : []),
    // Pull from server (for synced prompts)
    ...(onPullFromServer && isSynced ? [{
      key: "pull",
      label: t("managePrompts.sync.pullFromServer", { defaultValue: "Pull from Server" }),
      icon: <CloudDownload className="size-4" />,
      onClick: onPullFromServer
    }] : []),
    // Unlink option (for synced prompts)
    ...(onUnlink && isSynced ? [{
      key: "unlink",
      label: t("managePrompts.sync.unlink", { defaultValue: "Unlink from Server" }),
      icon: <Unlink className="size-4" />,
      onClick: onUnlink
    }] : []),
    ...(canSync ? [{ type: "divider" as const }] : [])
  ] : []

  const overflowItems: MenuProps["items"] = [
    ...syncItems,
    ...(!inlineUseInChat
      ? [
          {
            key: "useInChat",
            label: t("option:promptInsert.useInChat", {
              defaultValue: "Use in chat"
            }),
            icon: <MessageCircle className="size-4" />,
            disabled,
            onClick: onUseInChat
          },
          {
            type: "divider" as const
          }
        ]
      : []),
    ...(onQuickTest
      ? [
          {
            key: "quickTest",
            label: t("managePrompts.quickTest.action", {
              defaultValue: "Quick test"
            }),
            icon: <Play className="size-4" />,
            disabled,
            onClick: onQuickTest
          },
          {
            type: "divider" as const
          }
        ]
      : []),
    ...(onShareLink && isSynced
      ? [
          {
            key: "shareLink",
            label: t("managePrompts.share.copyLinkAction", {
              defaultValue: "Copy share link"
            }),
            icon: <Link className="size-4" />,
            disabled,
            onClick: onShareLink
          },
          {
            type: "divider" as const
          }
        ]
      : []),
    {
      key: "duplicate",
      label: t("managePrompts.tooltip.duplicate", { defaultValue: "Duplicate" }),
      icon: <CopyIcon className="size-4" />,
      disabled,
      onClick: onDuplicate
    },
    ...(onViewHistory
      ? [
          {
            key: "viewHistory",
            label: t("managePrompts.versionHistory", { defaultValue: "Version history" }),
            icon: <History className="size-4" />,
            disabled,
            onClick: onViewHistory
          }
        ]
      : []),
    {
      type: "divider"
    },
    {
      key: "delete",
      label: t("common:delete", { defaultValue: "Delete" }),
      icon: <Trash2 className="size-4" />,
      danger: true,
      disabled,
      onClick: onDelete
    }
  ]

  return (
    <div className="flex items-center gap-1 whitespace-nowrap">
      <Tooltip title={t("managePrompts.tooltip.edit")}>
        <button
          type="button"
          aria-label={t("managePrompts.tooltip.edit")}
          data-testid={`prompt-edit-${promptId}`}
          onClick={onEdit}
          disabled={disabled}
          className="inline-flex items-center rounded-md border border-transparent p-1.5 text-text-muted transition motion-reduce:transition-none hover:border-border hover:bg-surface2 focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-1 focus:ring-offset-bg disabled:cursor-not-allowed disabled:opacity-60"
        >
          <Pen className="size-4" />
        </button>
      </Tooltip>

      {inlineUseInChat && (
        <Tooltip
          title={t("option:promptInsert.useInChatTooltip", {
            defaultValue: "Open chat and insert this prompt into the composer."
          })}
        >
          <button
            type="button"
            aria-label={t("option:promptInsert.useInChat", {
              defaultValue: "Use in chat"
            })}
            data-testid={`prompt-use-${promptId}`}
            onClick={onUseInChat}
            disabled={disabled}
            className="inline-flex items-center rounded-md border border-transparent p-1.5 text-primary transition motion-reduce:transition-none hover:border-primary/30 hover:bg-primary/10 focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-1 focus:ring-offset-bg disabled:cursor-not-allowed disabled:opacity-60"
          >
            <MessageCircle className="size-4" />
          </button>
        </Tooltip>
      )}

      <Dropdown
        menu={{ items: overflowItems }}
        trigger={["click"]}
        placement="bottomRight"
      >
        <button
          type="button"
          aria-label={t("common:moreActions", { defaultValue: "More actions" })}
          data-testid={`prompt-more-${promptId}`}
          disabled={disabled}
          className="inline-flex items-center rounded-md border border-transparent p-1.5 text-text-muted transition motion-reduce:transition-none hover:border-border hover:bg-surface2 focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-1 focus:ring-offset-bg disabled:cursor-not-allowed disabled:opacity-60"
        >
          <MoreHorizontal className="size-4" />
        </button>
      </Dropdown>
    </div>
  )
}
