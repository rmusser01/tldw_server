import { useCallback, useMemo, useState } from "react"
import {
  useKeyboardShortcuts,
  type KeyboardShortcutConfig
} from "@/hooks/keyboard/useKeyboardShortcuts"

interface KanbanShortcutActions {
  onNewCard?: () => void
  onNewBoard?: () => void
  onNewList?: () => void
  onClosePanel?: () => void
}

export const useKanbanShortcuts = (actions: KanbanShortcutActions) => {
  const [helpOpen, setHelpOpen] = useState(false)

  const showHelp = useCallback(() => setHelpOpen(true), [])

  const shortcuts: KeyboardShortcutConfig[] = useMemo(() => [
    ...(actions.onNewCard
      ? [{
          shortcut: { key: "n", preventDefault: true, stopPropagation: false },
          action: actions.onNewCard,
          ignoreIfEditable: true,
          description: "New card in active list"
        }]
      : []),
    ...(actions.onNewBoard
      ? [{
          shortcut: { key: "b", preventDefault: true, stopPropagation: false },
          action: actions.onNewBoard,
          ignoreIfEditable: true,
          description: "New board"
        }]
      : []),
    ...(actions.onNewList
      ? [{
          shortcut: { key: "l", preventDefault: true, stopPropagation: false },
          action: actions.onNewList,
          ignoreIfEditable: true,
          description: "New list"
        }]
      : []),
    ...(actions.onClosePanel
      ? [{
          shortcut: { key: "Escape", preventDefault: true, stopPropagation: false },
          action: actions.onClosePanel,
          ignoreIfEditable: true,
          description: "Close panel"
        }]
      : []),
    {
      shortcut: { key: "?", shiftKey: true, preventDefault: true, stopPropagation: false },
      action: showHelp,
      ignoreIfEditable: true,
      description: "Show keyboard shortcuts"
    }
  ], [actions.onNewCard, actions.onNewBoard, actions.onNewList, actions.onClosePanel, showHelp])

  useKeyboardShortcuts(shortcuts)

  return { helpOpen, setHelpOpen, shortcuts }
}
