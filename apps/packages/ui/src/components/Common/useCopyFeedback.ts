import { useCallback, useEffect, useRef, useState } from "react"

/** How long a "Copied" confirmation stays visible after a successful copy. */
export const COPY_FEEDBACK_MS = 4000
/**
 * How long "Couldn't copy" and its hint stay visible, wherever the hook is
 * used: long enough to read, even where a success clears sooner.
 */
export const COPY_FAILURE_MS = 4000

export type CopyFeedbackStatus = "idle" | "copied" | "failed"

/**
 * Runs a clipboard write and keeps a short-lived status so the caller can
 * confirm the copy (check icon, "Copied") or say it failed. `copy` writes
 * plain text; `run` wraps any other write (for example formatted text).
 * Both resolve to whether the write succeeded. `attempt` changes on every
 * copy, so CopyStatus can announce a repeated result again.
 */
export const useCopyFeedback = (resetAfterMs: number = COPY_FEEDBACK_MS) => {
  const [result, setResult] = useState<{ status: CopyFeedbackStatus; attempt: number }>({
    status: "idle",
    attempt: 0
  })
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
      setResult((current) => ({
        status: copied ? "copied" : "failed",
        attempt: current.attempt + 1
      }))
      if (resetTimerRef.current) {
        clearTimeout(resetTimerRef.current)
      }
      resetTimerRef.current = setTimeout(() => {
        setResult((current) => ({ ...current, status: "idle" }))
        resetTimerRef.current = null
      }, copied ? resetAfterMs : COPY_FAILURE_MS)
      return copied
    },
    [resetAfterMs]
  )

  const copy = useCallback(
    (text: string) => run(() => navigator.clipboard.writeText(text)),
    [run]
  )

  return {
    status: result.status,
    attempt: result.attempt,
    copied: result.status === "copied",
    failed: result.status === "failed",
    copy,
    run
  }
}
