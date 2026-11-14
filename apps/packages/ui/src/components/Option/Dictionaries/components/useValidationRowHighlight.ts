import React from "react"
import { smoothScrollBehavior } from "@/utils/reduced-motion"

type UseValidationRowHighlightResult = {
  highlightedValidationEntryId: number | null
  jumpToValidationEntry: (field: unknown) => void
}

export function useValidationRowHighlight(entries: any[]): UseValidationRowHighlightResult {
  const [highlightedValidationEntryId, setHighlightedValidationEntryId] = React.useState<number | null>(null)
  const validationRowHighlightTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  React.useEffect(() => {
    return () => {
      if (validationRowHighlightTimerRef.current) {
        clearTimeout(validationRowHighlightTimerRef.current)
      }
    }
  }, [])

  const jumpToValidationEntry = React.useCallback(
    (field: unknown) => {
      if (typeof field !== "string") return
      const match = field.match(/^entries\[(\d+)\]/)
      if (!match) return
      const fieldIndex = Number(match[1])
      if (!Number.isFinite(fieldIndex) || fieldIndex < 0 || fieldIndex >= entries.length) {
        return
      }

      const entryId = Number(entries[fieldIndex]?.id)
      if (!Number.isFinite(entryId) || entryId <= 0) return

      if (validationRowHighlightTimerRef.current) {
        clearTimeout(validationRowHighlightTimerRef.current)
      }
      setHighlightedValidationEntryId(entryId)
      validationRowHighlightTimerRef.current = setTimeout(() => {
        setHighlightedValidationEntryId(null)
      }, 2200)

      if (typeof document !== "undefined") {
        const rowElement = document.querySelector(`tr[data-row-key="${entryId}"]`)
        if (rowElement instanceof HTMLElement) {
          try {
            rowElement.scrollIntoView({ behavior: smoothScrollBehavior(), block: "center" })
          } catch {
            rowElement.scrollIntoView()
          }
        }
      }
    },
    [entries]
  )

  return {
    highlightedValidationEntryId,
    jumpToValidationEntry,
  }
}
