import React from "react"
import type { TFunction } from "i18next"
import { Dropdown, Tooltip } from "antd"
import {
  CheckCircle2,
  Circle,
  Clock,
  GitBranch,
  MoreHorizontal,
  Pencil,
  Pin,
  PinOff,
  RotateCcw,
  Table2,
  Settings2,
  Tag,
  Trash2,
  XCircle
} from "lucide-react"

import type { ServerChatHistoryItem } from "@/hooks/useServerChatHistory"
import type { ConversationState } from "@/services/tldw/TldwApiClient"
import { CONVERSATION_STATE_OPTIONS } from "@/utils/conversation-state"
import { formatRelativeTime } from "@/utils/dateFormatters"
import { cn } from "@/libs/utils"
import { ChatStateBadge } from "./ChatStateBadge"

const STATE_ICON_BY_VALUE: Record<ConversationState, React.ReactElement> = {
  "in-progress": <Circle className="size-3 text-primary" />,
  resolved: <CheckCircle2 className="size-3 text-success" />,
  backlog: <Clock className="size-3 text-text-subtle" />,
  "non-viable": <XCircle className="size-3 text-danger" />
}

type ServerChatRowCommonProps = {
  chat: ServerChatHistoryItem
  isTrashView: boolean
  isPinned: boolean
  isActive: boolean
  /** Temporary mode: the row opens the chat but hides pin and management actions. */
  readOnly?: boolean
  openMenuFor: string | null
  setOpenMenuFor: (value: string | null) => void
  onSelectChat: (chat: ServerChatHistoryItem) => void
  onTogglePinned: (chatId: string) => void
  onOpenSettings: (chat: ServerChatHistoryItem) => void
  onRenameChat: (chat: ServerChatHistoryItem) => void
  onCreateTable: (chat: ServerChatHistoryItem) => void
  onEditTopic: (chat: ServerChatHistoryItem) => void
  onDeleteChat: (chat: ServerChatHistoryItem) => void | Promise<void>
  onRestoreChat: (chat: ServerChatHistoryItem) => void | Promise<void>
  onUpdateState?: (chat: ServerChatHistoryItem, state: ConversationState) => void
  t: TFunction
}

type ServerChatRowProps = ServerChatRowCommonProps & {
  selectionMode: false
  isSelected?: undefined
  onToggleSelected?: undefined
}

type ServerChatRowSelectionProps = ServerChatRowCommonProps & {
  selectionMode: true
  isSelected: boolean
  onToggleSelected: (chatId: string) => void
}

type ServerChatRowAllProps = ServerChatRowProps | ServerChatRowSelectionProps

export const ServerChatRow = React.memo((props: ServerChatRowAllProps) => {
  const {
    chat,
    isTrashView,
    readOnly = false,
    isPinned,
    isActive,
    openMenuFor,
    setOpenMenuFor,
    onSelectChat,
    onTogglePinned,
    onOpenSettings,
    onRenameChat,
    onCreateTable,
    onEditTopic,
    onDeleteChat,
    onRestoreChat,
    onUpdateState,
    t
  } = props
  const selectionMode = props.selectionMode
  const isSelected = props.selectionMode ? props.isSelected : false
  const onToggleSelected = props.selectionMode ? props.onToggleSelected : undefined
  const lastModifiedMs = chat.updatedAtMs ?? chat.createdAtMs
  const lastModifiedLabel = Number.isNaN(lastModifiedMs)
    ? null
    : formatRelativeTime(new Date(lastModifiedMs).toISOString(), t)
  const lastModifiedTitle = Number.isNaN(lastModifiedMs)
    ? undefined
    : new Date(lastModifiedMs).toLocaleString()
  const stateMenuItems = CONVERSATION_STATE_OPTIONS.map((option) => ({
    key: `state-${option.value}`,
    icon: STATE_ICON_BY_VALUE[option.value],
    label: t(option.labelToken, option.defaultLabel),
    onClick: () => {
      setOpenMenuFor(null)
      onUpdateState?.(chat, option.value)
    }
  }))
  const activeMenuItems = [
    {
      key: "settings",
      icon: <Settings2 className="size-3" />,
      label: t("playground:composer.openModelSettings", {
        defaultValue: "Open current chat settings"
      }),
      onClick: () => {
        setOpenMenuFor(null)
        onOpenSettings(chat)
      }
    },
    {
      key: "rename",
      icon: <Pencil className="size-3" />,
      label: t("common:rename", { defaultValue: "Rename" }),
      onClick: () => {
        setOpenMenuFor(null)
        onRenameChat(chat)
      }
    },
    {
      key: "create-table",
      icon: <Table2 className="size-3" />,
      label: t("dataTables:createFromChat", "Create Table from Chat"),
      onClick: () => {
        setOpenMenuFor(null)
        onCreateTable(chat)
      }
    },
    {
      key: "state",
      label: t("sidepanel:contextMenu.status", "Status"),
      children: stateMenuItems
    },
    {
      key: "topic",
      icon: <Tag className="size-3" />,
      label: t("playground:composer.topicPlaceholder", "Topic label (optional)"),
      onClick: () => {
        setOpenMenuFor(null)
        onEditTopic(chat)
      }
    },
    { type: "divider" as const },
    {
      key: "delete",
      icon: <Trash2 className="size-3" />,
      label: t("common:delete", { defaultValue: "Delete" }),
      danger: true,
      onClick: () => {
        setOpenMenuFor(null)
        void onDeleteChat(chat)
      }
    }
  ]
  const trashMenuItems = [
    {
      key: "restore",
      icon: <RotateCcw className="size-3" />,
      label: t("common:restore", { defaultValue: "Restore" }),
      onClick: () => {
        setOpenMenuFor(null)
        void onRestoreChat(chat)
      }
    },
    { type: "divider" as const },
    {
      key: "hard-delete",
      icon: <Trash2 className="size-3" />,
      label: t("common:deletePermanently", {
        defaultValue: "Delete permanently"
      }),
      danger: true,
      onClick: () => {
        setOpenMenuFor(null)
        void onDeleteChat(chat)
      }
    }
  ]
  const menuItems = isTrashView ? trashMenuItems : activeMenuItems

  const renderSourceInfo = () => {
    if (chat.parent_conversation_id) {
      return (
        <Tooltip
          title={t("common:serverChatForkedTooltip", {
            chatId: String(chat.parent_conversation_id).slice(0, 8),
            defaultValue: "Forked from chat {{chatId}}"
          })}
        >
          <span className="inline-flex items-center gap-1">
            <GitBranch className="size-3" />
            <span>
              {t("common:serverChatForkedLabel", {
                defaultValue: "Forked chat"
              })}
            </span>
          </span>
        </Tooltip>
      )
    }
    return (
      <span>
        {t("common:serverChatSourceLabel", {
          defaultValue: "Server"
        })}
      </span>
    )
  }

  const handleRowClick = React.useCallback(() => {
    if (selectionMode && onToggleSelected) {
      onToggleSelected(chat.id)
      return
    }
    onSelectChat(chat)
  }, [selectionMode, onToggleSelected, chat, onSelectChat])

  return (
    <div
      className={cn(
        "flex py-2 px-2 items-center gap-3 relative rounded-md truncate group transition-opacity duration-300 ease-in-out border",
        isActive
          ? "bg-surface2 border-border-strong text-text"
          : "bg-surface text-text border-border hover:bg-surface2",
        selectionMode && isSelected && "border-primary/40"
      )}
    >
      {selectionMode ? (
        <input
          type="checkbox"
          checked={isSelected}
          onChange={() => onToggleSelected?.(chat.id)}
          onClick={(event) => event.stopPropagation()}
          className="size-3 rounded border-border text-primary accent-primary"
          aria-label={t("sidepanel:multiSelect.toggle", "Toggle selection")}
        />
      ) : null}
      <button
        className="flex flex-col overflow-hidden flex-1 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-1 rounded"
        onClick={handleRowClick}
      >
        <span className="truncate text-sm" title={chat.title}>
          {chat.title}
        </span>
        {chat.match_snippet && (
          // A search matched what was said in this chat: show it, since the title alone does not explain the hit.
          <span
            className="truncate text-xs text-text-subtle"
            title={chat.match_snippet}
            data-testid="server-chat-match-snippet"
          >
            {chat.match_snippet}
          </span>
        )}
        <div className="flex flex-wrap items-center gap-2 text-xs text-text-subtle mt-1">
          <ChatStateBadge state={chat.state} />
          {lastModifiedLabel && (
            <span
              className="text-[11px] text-text-subtle"
              title={lastModifiedTitle}
            >
              {t("common:updated", { defaultValue: "Updated" })}{" "}
              {lastModifiedLabel}
            </span>
          )}
          {chat.topic_label && (
            <span
              className="truncate max-w-[12rem]"
              title={String(chat.topic_label)}
            >
              {String(chat.topic_label)}
            </span>
          )}
        </div>
        <span className="flex items-center gap-1 text-xs text-text-subtle">
          {renderSourceInfo()}
        </span>
      </button>
      {!selectionMode && !readOnly && (
        <div className="flex flex-col items-center gap-1">
          {!isTrashView && (
            <Tooltip title={isPinned ? t("common:unpin") : t("common:pin")}>
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  onTogglePinned(chat.id)
                }}
                className="rounded p-1 text-text-subtle hover:bg-surface2 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--color-focus)] h-7 w-7 sm:min-w-0 sm:min-h-0"
                aria-label={isPinned ? t("common:unpin") : t("common:pin")}
                aria-pressed={isPinned}
              >
                {isPinned ? (
                  <PinOff className="size-3" />
                ) : (
                  <Pin className="size-3" />
                )}
              </button>
            </Tooltip>
          )}
          <Dropdown
            menu={{ items: menuItems, id: `server-chat-actions-${chat.id}` }}
            trigger={["click"]}
            placement="bottomRight"
            open={openMenuFor === chat.id}
            onOpenChange={(open) => setOpenMenuFor(open ? chat.id : null)}
          >
            <button
              type="button"
              onClick={(event) => event.stopPropagation()}
              className="rounded p-1 text-text-subtle hover:bg-surface2 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--color-focus)] h-7 w-7 sm:min-w-0 sm:min-h-0"
              aria-label={`${t("option:header.moreActions", {
                defaultValue: "More actions"
              })}: ${chat.title}`}
              aria-haspopup="menu"
              aria-expanded={openMenuFor === chat.id}
              aria-controls={`server-chat-actions-${chat.id}`}
            >
              <MoreHorizontal className="size-3.5" />
            </button>
          </Dropdown>
        </div>
      )}
    </div>
  )
})

ServerChatRow.displayName = "ServerChatRow"
