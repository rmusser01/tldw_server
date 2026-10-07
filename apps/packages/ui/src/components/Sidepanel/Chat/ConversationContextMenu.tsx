import React from "react"
import { Dropdown, Input, Modal } from "antd"
import type { InputRef, MenuProps } from "antd"
import {
  Pencil,
  Pin,
  PinOff,
  FolderPlus,
  Circle,
  CheckCircle2,
  Clock,
  XCircle,
  Download,
  FileJson,
  FileText,
  Trash2,
  X
} from "lucide-react"
import { useTranslation } from "react-i18next"
import type { SidepanelChatTab } from "@/store/sidepanel-chat-tabs"

export type ConversationStatus =
  | "in_progress"
  | "resolved"
  | "backlog"
  | "non_viable"
  | null

/** Resolve `false` to keep a dialog open, e.g. after the action failed. */
type DialogAction = void | boolean | Promise<void | boolean>

export type ConversationContextMenuProps = {
  tab: SidepanelChatTab
  children: React.ReactNode
  onRename: (tabId: string, newLabel: string) => DialogAction
  /**
   * The conversation's full title for the Rename dialog to start from. The
   * tab label is truncated for display, so it is used only when this is
   * missing or resolves to null.
   */
  loadRenameTitle?: (tabId: string) => Promise<string | null>
  onTogglePin: (tabId: string) => void
  onSetStatus: (tabId: string, status: ConversationStatus) => void
  onAddToFolder: (tabId: string) => void
  onExportJSON: (tabId: string) => void
  onExportMarkdown: (tabId: string) => void
  /** Delete the tab's server chat; offered only for a tab bound to one. */
  onDelete: (tabId: string) => DialogAction
  /** Close the tab. A tab with no server chat offers this instead of Delete. */
  onCloseTab: (tabId: string) => void
  currentStatus?: ConversationStatus
}

export const ConversationContextMenu: React.FC<
  ConversationContextMenuProps
> = ({
  tab,
  children,
  onRename,
  loadRenameTitle,
  onTogglePin,
  onSetStatus,
  onAddToFolder,
  onExportJSON,
  onExportMarkdown,
  onDelete,
  onCloseTab,
  currentStatus
}) => {
  const { t } = useTranslation(["common", "sidepanel"])
  const [renameModalOpen, setRenameModalOpen] = React.useState(false)
  const [renameValue, setRenameValue] = React.useState(tab.label)
  // The title the dialog started from; saving it unchanged renames nothing.
  const [renameBaseline, setRenameBaseline] = React.useState(tab.label)
  const [renameTitleLoading, setRenameTitleLoading] = React.useState(false)
  const renameRequestRef = React.useRef(0)
  const renameInputRef = React.useRef<InputRef>(null)
  const [renaming, setRenaming] = React.useState(false)
  const [deleteConfirmOpen, setDeleteConfirmOpen] = React.useState(false)
  const [deleting, setDeleting] = React.useState(false)
  const deletesServerChat = Boolean(tab.serverChatId)

  const openRenameDialog = () => {
    const request = ++renameRequestRef.current
    setRenameValue(tab.label)
    setRenameBaseline(tab.label)
    setRenameModalOpen(true)
    if (!loadRenameTitle) return
    setRenameTitleLoading(true)
    loadRenameTitle(tab.id)
      .then((title) => {
        if (request !== renameRequestRef.current || !title?.trim()) return
        setRenameValue(title)
        setRenameBaseline(title)
      })
      .catch((error) => {
        console.warn("[sidepanel] Could not load the chat's title to rename", error)
      })
      .finally(() => {
        if (request === renameRequestRef.current) setRenameTitleLoading(false)
      })
  }

  const closeRenameDialog = () => {
    renameRequestRef.current++
    setRenameTitleLoading(false)
    setRenameModalOpen(false)
  }

  React.useEffect(() => {
    if (renameModalOpen && !renameTitleLoading) renameInputRef.current?.focus()
  }, [renameModalOpen, renameTitleLoading])

  const handleRenameSubmit = async () => {
    if (renaming || renameTitleLoading) return
    const nextLabel = renameValue.trim()
    if (!nextLabel || nextLabel === renameBaseline.trim()) {
      setRenameModalOpen(false)
      return
    }
    setRenaming(true)
    try {
      if ((await onRename(tab.id, nextLabel)) !== false) setRenameModalOpen(false)
    } finally {
      setRenaming(false)
    }
  }

  const handleDeleteConfirm = async () => {
    if (deleting) return
    setDeleting(true)
    try {
      if ((await onDelete(tab.id)) !== false) setDeleteConfirmOpen(false)
    } finally {
      setDeleting(false)
    }
  }

  const statusItems: MenuProps["items"] = [
    {
      key: "status-in_progress",
      icon: <Circle className="size-3 text-primary" />,
      label: t("sidepanel:contextMenu.statusInProgress", "In Progress"),
      onClick: () => onSetStatus(tab.id, "in_progress")
    },
    {
      key: "status-resolved",
      icon: <CheckCircle2 className="size-3 text-success" />,
      label: t("sidepanel:contextMenu.statusResolved", "Resolved"),
      onClick: () => onSetStatus(tab.id, "resolved")
    },
    {
      key: "status-backlog",
      icon: <Clock className="size-3 text-text-muted" />,
      label: t("sidepanel:contextMenu.statusBacklog", "Backlog"),
      onClick: () => onSetStatus(tab.id, "backlog")
    },
    {
      key: "status-non_viable",
      icon: <XCircle className="size-3 text-danger" />,
      label: t("sidepanel:contextMenu.statusNonViable", "Non-viable"),
      onClick: () => onSetStatus(tab.id, "non_viable")
    },
    { type: "divider" },
    {
      key: "status-clear",
      label: t("sidepanel:contextMenu.statusClear", "Clear status"),
      onClick: () => onSetStatus(tab.id, null)
    }
  ]

  const exportItems: MenuProps["items"] = [
    {
      key: "export-json",
      icon: <FileJson className="size-3" />,
      label: t("sidepanel:contextMenu.exportJSON", "Export as JSON"),
      onClick: () => onExportJSON(tab.id)
    },
    {
      key: "export-md",
      icon: <FileText className="size-3" />,
      label: t("sidepanel:contextMenu.exportMarkdown", "Export as Markdown"),
      onClick: () => onExportMarkdown(tab.id)
    }
  ]

  const menuItems: MenuProps["items"] = [
    {
      key: "rename",
      icon: <Pencil className="size-3" />,
      label: t("sidepanel:contextMenu.rename", "Rename"),
      onClick: openRenameDialog
    },
    {
      key: "pin",
      icon: tab.pinned ? (
        <PinOff className="size-3" />
      ) : (
        <Pin className="size-3" />
      ),
      label: tab.pinned
        ? t("common:unpin", "Unpin")
        : t("common:pin", "Pin"),
      onClick: () => onTogglePin(tab.id)
    },
    {
      key: "folder",
      icon: <FolderPlus className="size-3" />,
      label: t("sidepanel:contextMenu.addToFolder", "Add to folder..."),
      onClick: () => onAddToFolder(tab.id)
    },
    { type: "divider" },
    {
      key: "status",
      label: t("sidepanel:contextMenu.status", "Status"),
      children: statusItems
    },
    { type: "divider" },
    {
      key: "export",
      icon: <Download className="size-3" />,
      label: t("sidepanel:contextMenu.export", "Export"),
      children: exportItems
    },
    { type: "divider" },
    deletesServerChat
      ? {
          key: "delete",
          icon: <Trash2 className="size-3" />,
          label: t("common:delete", "Delete"),
          danger: true,
          onClick: () => setDeleteConfirmOpen(true)
        }
      : {
          // Nothing is deleted: the chat has no server copy to remove.
          key: "close",
          icon: <X className="size-3" />,
          label: t("sidepanel:contextMenu.closeTab", "Close tab"),
          onClick: () => onCloseTab(tab.id)
        }
  ]

  return (
    <>
      <Dropdown
        menu={{ items: menuItems }}
        trigger={["contextMenu"]}
        destroyPopupOnHide
      >
        {children}
      </Dropdown>

      {/* Rename Modal */}
      <Modal
        open={renameModalOpen}
        title={t("sidepanel:contextMenu.renameTitle", "Rename conversation")}
        onOk={() => void handleRenameSubmit()}
        onCancel={closeRenameDialog}
        okText={t("common:save", "Save")}
        cancelText={t("common:cancel", "Cancel")}
        confirmLoading={renaming}
        okButtonProps={{ disabled: renameTitleLoading }}
        destroyOnHidden
      >
        <Input
          ref={renameInputRef}
          value={renameValue}
          onChange={(e) => setRenameValue(e.target.value)}
          onPressEnter={() => void handleRenameSubmit()}
          disabled={renameTitleLoading}
          placeholder={t(
            "sidepanel:contextMenu.renamePlaceholder",
            "Enter conversation name"
          )}
        />
      </Modal>

      {/* Delete Confirmation Modal */}
      <Modal
        open={deleteConfirmOpen}
        title={t("sidepanel:contextMenu.deleteTitle", "Delete conversation")}
        onOk={() => void handleDeleteConfirm()}
        onCancel={() => setDeleteConfirmOpen(false)}
        okText={t("sidepanel:contextMenu.moveToTrash", "Move to Trash")}
        cancelText={t("common:cancel", "Cancel")}
        okButtonProps={{ danger: true, loading: deleting }}
        destroyOnHidden
      >
        <p>
          {t("sidepanel:contextMenu.moveToTrashConfirm", {
            defaultValue:
              "\"{{title}}\" moves to Trash and its tab closes. You can undo this right away, or restore it later from Trash in the full chat page.",
            title: tab.label
          })}
        </p>
      </Modal>
    </>
  )
}

export default ConversationContextMenu
