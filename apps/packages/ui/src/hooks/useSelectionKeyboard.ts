import { useCallback, useRef, useState } from "react"
import { isEditableTarget } from "@/utils/editable-target"

interface UseSelectionKeyboardOptions<T> {
  items: T[]
  selectedIds: string[]
  getItemId: (item: T) => string
  onSelectionChange: (ids: string[]) => void
}

interface UseSelectionKeyboardReturn {
  focusedIndex: number
  setFocusedIndex: (index: number) => void
  lastSelectedIndex: number | null
  handleItemClick: (index: number, event: React.MouseEvent) => void
  handleItemToggle: (index: number, options?: { shiftKey?: boolean }) => void
  handleKeyDown: (event: React.KeyboardEvent) => void
  listRef: React.RefObject<HTMLDivElement | null>
}

/**
 * Hook for keyboard navigation (j/k/Space) and Shift+click range selection
 * in list components.
 *
 * Usage:
 * - j/ArrowDown: Move focus down
 * - k/ArrowUp: Move focus up
 * - Space/Enter: Toggle selection of focused item
 * - Shift+Click: Select range from last selected to clicked item
 */
export function useSelectionKeyboard<T>({
  items,
  selectedIds,
  getItemId,
  onSelectionChange
}: UseSelectionKeyboardOptions<T>): UseSelectionKeyboardReturn {
  const [focusedIndex, setFocusedIndexState] = useState(-1)
  const [lastSelectedIndex, setLastSelectedIndexState] = useState<number | null>(null)
  const focusedIndexRef = useRef(-1)
  const lastSelectedIndexRef = useRef<number | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)

  const setFocusedIndex = useCallback((index: number) => {
    focusedIndexRef.current = index
    setFocusedIndexState(index)
  }, [])

  const setLastSelectedIndex = useCallback((index: number | null) => {
    lastSelectedIndexRef.current = index
    setLastSelectedIndexState(index)
  }, [])

  const toggleSelection = useCallback(
    (index: number) => {
      if (index < 0 || index >= items.length) return

      const item = items[index]
      const itemId = getItemId(item)
      const selectedSet = new Set(selectedIds)

      if (selectedSet.has(itemId)) {
        selectedSet.delete(itemId)
      } else {
        selectedSet.add(itemId)
      }

      setLastSelectedIndex(index)
      onSelectionChange(Array.from(selectedSet))
    },
    [items, selectedIds, getItemId, onSelectionChange]
  )

  const selectRange = useCallback(
    (fromIndex: number, toIndex: number) => {
      const start = Math.min(fromIndex, toIndex)
      const end = Math.max(fromIndex, toIndex)
      const selectedSet = new Set(selectedIds)

      for (let i = start; i <= end; i++) {
        if (i >= 0 && i < items.length) {
          const itemId = getItemId(items[i])
          selectedSet.add(itemId)
        }
      }

      onSelectionChange(Array.from(selectedSet))
    },
    [items, selectedIds, getItemId, onSelectionChange]
  )

  const handleItemToggle = useCallback(
    (index: number, options?: { shiftKey?: boolean }) => {
      if (index < 0 || index >= items.length) return

      const anchorIndex = lastSelectedIndexRef.current
      if (options?.shiftKey && anchorIndex !== null) {
        selectRange(anchorIndex, index)
        setLastSelectedIndex(index)
        setFocusedIndex(index)
        return
      }

      toggleSelection(index)
      setFocusedIndex(index)
    },
    [items.length, selectRange, setFocusedIndex, setLastSelectedIndex, toggleSelection]
  )

  const handleItemClick = useCallback(
    (index: number, event: React.MouseEvent) => {
      handleItemToggle(index, { shiftKey: event.shiftKey })
    },
    [handleItemToggle]
  )

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      // Checkboxes stay live so j/k/space keep working while a row checkbox has focus.
      const isCheckbox = target instanceof HTMLInputElement && target.type === "checkbox"
      if (!isCheckbox && isEditableTarget(target)) return

      const key = event.key.toLowerCase()
      const currentFocusedIndex = focusedIndexRef.current

      // j or ArrowDown: move down
      if (key === "j" || key === "arrowdown") {
        event.preventDefault()
        const nextIndex = Math.min(currentFocusedIndex + 1, items.length - 1)
        setFocusedIndex(nextIndex)
        listRef.current?.focus()

        // Scroll focused item into view
        const listElement = listRef.current
        if (listElement) {
          const itemElements = listElement.querySelectorAll('[data-selection-item]')
          itemElements[nextIndex]?.scrollIntoView({ block: "nearest" })
        }
        return
      }

      // k or ArrowUp: move up
      if (key === "k" || key === "arrowup") {
        event.preventDefault()
        const prevIndex = Math.max(currentFocusedIndex - 1, 0)
        setFocusedIndex(prevIndex)
        listRef.current?.focus()

        // Scroll focused item into view
        const listElement = listRef.current
        if (listElement) {
          const itemElements = listElement.querySelectorAll('[data-selection-item]')
          itemElements[prevIndex]?.scrollIntoView({ block: "nearest" })
        }
        return
      }

      // Space or Enter: toggle selection
      if (key === " " || key === "space" || key === "spacebar" || key === "enter") {
        event.preventDefault()
        if (currentFocusedIndex >= 0 && currentFocusedIndex < items.length) {
          const anchorIndex = lastSelectedIndexRef.current
          if (event.shiftKey && anchorIndex !== null) {
            selectRange(anchorIndex, currentFocusedIndex)
            setLastSelectedIndex(currentFocusedIndex)
          } else {
            toggleSelection(currentFocusedIndex)
          }
        }
        listRef.current?.focus()
        return
      }

      // Escape: clear focus
      if (key === "escape") {
        event.preventDefault()
        setFocusedIndex(-1)
        listRef.current?.focus()
        return
      }
    },
    [items.length, selectRange, setFocusedIndex, setLastSelectedIndex, toggleSelection]
  )

  return {
    focusedIndex,
    setFocusedIndex,
    lastSelectedIndex,
    handleItemClick,
    handleItemToggle,
    handleKeyDown,
    listRef
  }
}
