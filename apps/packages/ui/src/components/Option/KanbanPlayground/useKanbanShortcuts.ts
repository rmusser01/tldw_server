import { useCallback, useMemo, useState } from "react"
import {
  useKeyboardShortcuts,
  type KeyboardShortcutConfig
} from "@/hooks/keyboard/useKeyboardShortcuts"
import { isEditableTarget } from "@/utils/editable-target"

interface KanbanShortcutActions {
  onNewCard?: () => void
  onNewBoard?: () => void
  onNewList?: () => void
  onClosePanel?: () => void
}

export const useKanbanShortcuts = (actions: KanbanShortcutActions) => {
  const [helpOpen, setHelpOpen] = useState(false)

  const guard = useCallback(
    (fn?: () => void) => () => {
      if (isEditableTarget(document.activeElement)) return
      fn?.()
    },
    []
  )

  const showHelp = useCallback(() => setHelpOpen(true), [])

  const shortcuts: KeyboardShortcutConfig[] = useMemo(() => [
    ...(actions.onNewCard
      ? [{
          shortcut: { key: "n", preventDefault: true, stopPropagation: false },
          action: guard(actions.onNewCard),
          description: "New card in active list"
        }]
      : []),
    ...(actions.onNewBoard
      ? [{
          shortcut: { key: "b", preventDefault: true, stopPropagation: false },
          action: guard(actions.onNewBoard),
          description: "New board"
        }]
      : []),
    ...(actions.onNewList
      ? [{
          shortcut: { key: "l", preventDefault: true, stopPropagation: false },
          action: guard(actions.onNewList),
          description: "New list"
        }]
      : []),
    ...(actions.onClosePanel
      ? [{
          shortcut: { key: "Escape", preventDefault: true, stopPropagation: false },
          action: guard(actions.onClosePanel),
          description: "Close panel"
        }]
      : []),
    {
      shortcut: { key: "?", shiftKey: true, preventDefault: true, stopPropagation: false },
      action: guard(showHelp),
      description: "Show keyboard shortcuts"
    }
  ], [actions.onNewCard, actions.onNewBoard, actions.onNewList, actions.onClosePanel, guard, showHelp])

  useKeyboardShortcuts(shortcuts)

  return { helpOpen, setHelpOpen, shortcuts }
}
