import { useCallback, useEffect, useRef, useState } from "react"

/** How long a "Copied" or "Couldn't copy" message stays visible after a copy. */
export const COPY_FEEDBACK_MS = 4000

export type CopyFeedbackStatus = "idle" | "copied" | "failed"

/**
 * Runs a clipboard write and keeps a short-lived status so the caller can
 * confirm the copy (check icon, "Copied") or say it failed. `copy` writes
 * plain text; `run` wraps any other write (for example formatted text).
 * Both resolve to whether the write succeeded.
 */
export const useCopyFeedback = (resetAfterMs: number = COPY_FEEDBACK_MS) => {
  const [status, setStatus] = useState<CopyFeedbackStatus>("idle")
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

  const run = useCallback(
    async (write: () => Promise<unknown>): Promise<boolean> => {
      let copied = true
      try {
        await write()
      } catch {
        copied = false
      }
      setStatus(copied ? "copied" : "failed")
      if (resetTimerRef.current) {
        clearTimeout(resetTimerRef.current)
      }
      resetTimerRef.current = setTimeout(() => {
        setStatus("idle")
        resetTimerRef.current = null
      }, resetAfterMs)
      return copied
    },
    [resetAfterMs]
  )

  const copy = useCallback(
    (text: string) => run(() => navigator.clipboard.writeText(text)),
    [run]
  )

  return {
    status,
    copied: status === "copied",
    failed: status === "failed",
    copy,
    run
  }
}
