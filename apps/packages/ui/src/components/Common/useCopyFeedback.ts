import { useCallback, useEffect, useRef, useState } from "react"

/** How long a "Copied" confirmation stays visible after a successful copy. */
export const COPY_FEEDBACK_MS = 4000

/**
 * Copies text to the clipboard and keeps a short-lived `copied` flag so the
 * caller can confirm the copy (check icon, "Copied" text). A failed clipboard
 * write leaves the flag unset and resolves to `false`.
 */
export const useCopyFeedback = (resetAfterMs: number = COPY_FEEDBACK_MS) => {
  const [copied, setCopied] = useState(false)
  const resetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(
    () => () => {
      if (resetTimerRef.current) {
        clearTimeout(resetTimerRef.current)
        resetTimerRef.current = null
      }
    },
    []
  )

  const copy = useCallback(
    async (text: string): Promise<boolean> => {
      try {
        await navigator.clipboard.writeText(text)
      } catch {
        return false
      }
      setCopied(true)
      if (resetTimerRef.current) {
        clearTimeout(resetTimerRef.current)
      }
      resetTimerRef.current = setTimeout(() => {
        setCopied(false)
        resetTimerRef.current = null
      }, resetAfterMs)
      return true
    },
    [resetAfterMs]
  )

  return { copied, copy }
}
