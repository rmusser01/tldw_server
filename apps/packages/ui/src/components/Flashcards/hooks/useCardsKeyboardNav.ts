import { useEffect, useCallback } from "react"
import { isEditableTarget } from "@/utils/editable-target"

export interface CardsKeyboardNavOptions {
  /** Whether navigation is enabled */
  enabled?: boolean
  /** Total number of items in the list */
  itemCount: number
  /** Current focused index (-1 if none) */
  focusedIndex: number
  /** Callback to set the focused index */
  onFocusChange: (index: number) => void
  /** Callback when Enter is pressed on focused item */
  onEdit?: (index: number) => void
  /** Callback when Space is pressed on focused item */
  onToggleSelect?: (index: number) => void
  /** Callback when Delete is pressed on focused item */
  onDelete?: (index: number) => void
}

type CardsKeyboardNavAction =
  | { type: "focus"; index: number }
  | { type: "edit"; index: number }
  | { type: "toggle"; index: number }
  | { type: "delete"; index: number }
  | { type: "clear" }

export type CardsKeyboardNavResult = {
  preventDefault: boolean
  action: CardsKeyboardNavAction
}

export function getCardsKeyboardNavResult({
  key,
  itemCount,
  focusedIndex
}: {
  key: string
  itemCount: number
  focusedIndex: number
}): CardsKeyboardNavResult | null {
  switch (key) {
    case "j":
    case "ArrowDown": {
      if (itemCount <= 0) return null
      const nextIndex =
        focusedIndex < 0 ? 0 : Math.min(focusedIndex + 1, itemCount - 1)
      return {
        preventDefault: true,
        action: { type: "focus", index: nextIndex }
      }
    }
    case "k":
    case "ArrowUp": {
      if (itemCount <= 0) return null
      const prevIndex =
        focusedIndex < 0 ? itemCount - 1 : Math.max(focusedIndex - 1, 0)
      return {
        preventDefault: true,
        action: { type: "focus", index: prevIndex }
      }
    }
    case "Enter":
      if (focusedIndex >= 0 && focusedIndex < itemCount) {
        return {
          preventDefault: true,
          action: { type: "edit", index: focusedIndex }
        }
      }
      return null
    case " ":
      if (focusedIndex >= 0 && focusedIndex < itemCount) {
        return {
          preventDefault: true,
          action: { type: "toggle", index: focusedIndex }
        }
      }
      return null
    case "Delete":
    case "Backspace":
      if (focusedIndex >= 0 && focusedIndex < itemCount) {
        return {
          preventDefault: true,
          action: { type: "delete", index: focusedIndex }
        }
      }
      return null
    case "Escape":
      return {
        preventDefault: false,
        action: { type: "clear" }
      }
    default:
      return null
  }
}

/**
 * Hook for keyboard navigation in the Cards tab list.
 *
 * Shortcuts:
 * - j / ArrowDown: Move focus down
 * - k / ArrowUp: Move focus up
 * - Enter: Edit focused card
 * - Space: Toggle selection of focused card
 * - Delete / Backspace: Delete focused card
 */
export function useCardsKeyboardNav({
  enabled = true,
  itemCount,
  focusedIndex,
  onFocusChange,
  onEdit,
  onToggleSelect,
  onDelete
}: CardsKeyboardNavOptions) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (isEditableTarget(e.target)) {
        return
      }

      const result = getCardsKeyboardNavResult({
        key: e.key,
        itemCount,
        focusedIndex
      })
      if (!result) return
      if (result.preventDefault) {
        e.preventDefault()
      }
      switch (result.action.type) {
        case "focus":
          onFocusChange(result.action.index)
          break
        case "edit":
          if (onEdit) onEdit(result.action.index)
          break
        case "toggle":
          if (onToggleSelect) onToggleSelect(result.action.index)
          break
        case "delete":
          if (onDelete) onDelete(result.action.index)
          break
        case "clear":
          onFocusChange(-1)
          break
      }
    },
    [itemCount, focusedIndex, onFocusChange, onEdit, onToggleSelect, onDelete]
  )

  useEffect(() => {
    if (!enabled) return

    window.addEventListener("keydown", handleKeyDown)
    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [enabled, handleKeyDown])
}

export default useCardsKeyboardNav
