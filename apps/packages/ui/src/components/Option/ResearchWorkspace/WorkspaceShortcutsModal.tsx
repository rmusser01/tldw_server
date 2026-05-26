import React from "react"
import { Modal } from "antd"
import { useTranslation } from "react-i18next"
import { isMac } from "@/hooks/useKeyboardShortcuts"

type WorkspaceShortcutsModalProps = {
  open: boolean
  onClose: () => void
  includeShowShortcutsShortcut?: boolean
}

export const WorkspaceShortcutsModal: React.FC<WorkspaceShortcutsModalProps> = ({
  open,
  onClose,
  includeShowShortcutsShortcut = false,
}) => {
  const { t } = useTranslation(["playground"])

  const shortcuts = React.useMemo(() => {
    const modifierLabel = isMac ? "Cmd" : "Ctrl"
    const rows = [
      {
        action: t("playground:workspace.shortcutSearch", "Search workspace"),
        combo: "Alt+K",
      },
      {
        action: t("playground:workspace.shortcutFocusSources", "Focus sources pane"),
        combo: "Alt+1",
      },
      {
        action: t("playground:workspace.shortcutFocusChat", "Focus chat pane"),
        combo: "Alt+2",
      },
      {
        action: t("playground:workspace.shortcutFocusStudio", "Focus studio pane"),
        combo: "Alt+3",
      },
      {
        action: t("playground:workspace.shortcutNewNote", "New note"),
        combo: "Alt+N",
      },
      {
        action: t("playground:workspace.shortcutNewWorkspace", "New workspace"),
        combo: "Alt+Shift+N",
      },
      {
        action: t("playground:workspace.shortcutUndo", "Undo"),
        combo: `${modifierLabel}+Z`,
      },
    ]

    if (includeShowShortcutsShortcut) {
      rows.push({
        action: t("playground:workspace.shortcutShowShortcuts", "Show shortcuts"),
        combo: "?",
      })
    }

    return rows
  }, [includeShowShortcutsShortcut, t])

  return (
    <Modal
      title={t("playground:workspace.keyboardShortcuts", "Keyboard Shortcuts")}
      open={open}
      onCancel={onClose}
      footer={null}
      width={520}
      destroyOnHidden
    >
      <div className="space-y-2">
        {shortcuts.map((shortcut) => (
          <div
            key={`${shortcut.action}-${shortcut.combo}`}
            className="flex items-center justify-between rounded border border-border px-3 py-2"
          >
            <span className="text-sm text-text">{shortcut.action}</span>
            <code className="rounded bg-surface2 px-2 py-0.5 text-xs font-semibold text-text">
              {shortcut.combo}
            </code>
          </div>
        ))}
      </div>
    </Modal>
  )
}
