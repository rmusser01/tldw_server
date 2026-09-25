import { useEffect, useCallback } from "react"
import { isEditableTarget } from "@/utils/editable-target"

const RATING_MAP: Record<string, number> = {
  "1": 0, // Again
  "2": 2, // Hard
  "3": 3, // Good
  "4": 5 // Easy
}

type FlashcardShortcutAction =
  | { type: "flip" }
  | { type: "rate"; rating: number }
  | { type: "edit" }
  | { type: "undo" }

export type FlashcardShortcutResult = {
  preventDefault: boolean
  action?: FlashcardShortcutAction
}

export function getFlashcardShortcutResult(
  key: string,
  showingAnswer: boolean,
  ctrlOrMeta: boolean
): FlashcardShortcutResult | null {
  // Ctrl/Cmd+Z for undo (platform standard)
  if (ctrlOrMeta && key.toLowerCase() === "z") {
    return {
      preventDefault: true,
      action: { type: "undo" }
    }
  }

  if (key === " ") {
    return {
      preventDefault: true,
      action: showingAnswer ? undefined : { type: "flip" }
    }
  }

  if (key.toLowerCase() === "e") {
    return {
      preventDefault: true,
      action: { type: "edit" }
    }
  }

  if (showingAnswer && key in RATING_MAP) {
    return {
      preventDefault: true,
      action: { type: "rate", rating: RATING_MAP[key] }
    }
  }

  return null
}

interface FlashcardShortcutsOptions {
  /** Whether shortcuts are enabled */
  enabled?: boolean
  /** Whether the answer is currently shown */
  showingAnswer: boolean
  /** Callback to flip the card (show answer) */
  onFlip: () => void
  /** Callback to submit a rating (0=Again, 2=Hard, 3=Good, 5=Easy) */
  onRate: (rating: number) => void
  /** Callback to open edit drawer for the current card */
  onEdit?: () => void
  /** Callback to undo last rating (Ctrl/Cmd+Z) */
  onUndo?: () => void
}

/**
 * Hook for keyboard shortcuts in flashcard review.
 *
 * Shortcuts:
 * - Space: Flip card (show answer)
 * - 1: Rate Again (0)
 * - 2: Rate Hard (2)
 * - 3: Rate Good (3)
 * - 4: Rate Easy (5)
 * - E: Edit current card
 * - Ctrl/Cmd+Z: Undo last rating
 */
export function useFlashcardShortcuts({
  enabled = true,
  showingAnswer,
  onFlip,
  onRate,
  onEdit,
  onUndo
}: FlashcardShortcutsOptions) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (isEditableTarget(e.target)) {
        return
      }

      const ctrlOrMeta = e.ctrlKey || e.metaKey
      const result = getFlashcardShortcutResult(e.key, showingAnswer, ctrlOrMeta)
      if (!result) return
      if (result.preventDefault) {
        e.preventDefault()
      }
      if (!result.action) return
      if (result.action.type === "flip") {
        onFlip()
        return
      }
      if (result.action.type === "undo") {
        onUndo?.()
        return
      }
      if (result.action.type === "edit") {
        onEdit?.()
        return
      }
      onRate(result.action.rating)
    },
    [showingAnswer, onFlip, onRate, onEdit, onUndo]
  )

  useEffect(() => {
    if (!enabled) return

    window.addEventListener("keydown", handleKeyDown)
    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [enabled, handleKeyDown])
}

export default useFlashcardShortcuts
