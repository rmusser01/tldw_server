import React from "react"

import type { ConnectionUxState } from "@/types/connection"

type ServerErrorUxState = Extract<
  ConnectionUxState,
  "error_unreachable" | "error_auth"
>

const isServerErrorUxState = (
  uxState: ConnectionUxState
): uxState is ServerErrorUxState =>
  uxState === "error_unreachable" || uxState === "error_auth"

/**
 * Keeps the last server error on screen while a re-check runs (XS-11).
 *
 * Every connection check, manual or from the background poll, passes through
 * the "testing" state. Without this the offline card would flip to the setup
 * copy and unmount its Retry and Diagnostics buttons for the length of each
 * check, pulling keyboard focus away every few seconds.
 */
export const useStickyServerErrorUxState = (
  uxState: ConnectionUxState
): ConnectionUxState => {
  const [lastError, setLastError] = React.useState<ServerErrorUxState | null>(
    isServerErrorUxState(uxState) ? uxState : null
  )
  if (isServerErrorUxState(uxState)) {
    if (lastError !== uxState) setLastError(uxState)
  } else if (uxState !== "testing" && lastError !== null) {
    setLastError(null)
  }
  if (uxState === "testing" && lastError) return lastError
  return uxState
}

/**
 * "Retry now" for the offline card (XS-11).
 *
 * The button shows as busy only for a retry the user asked for, not during the
 * background poll. A busy button is disabled, and some browsers drop focus
 * from a disabled element, so focus goes back to the button once the retry
 * settles if it had focus when the user activated it.
 */
export const useConnectionRetry = (checkOnce: () => Promise<void> | void) => {
  const [isRetrying, setIsRetrying] = React.useState(false)
  const retryButtonRef = React.useRef<HTMLButtonElement | null>(null)
  const restoreFocusRef = React.useRef(false)
  const mountedRef = React.useRef(true)

  React.useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  const retry = React.useCallback(() => {
    if (typeof document !== "undefined") {
      restoreFocusRef.current =
        document.activeElement === retryButtonRef.current
    }
    setIsRetrying(true)
    let pending: Promise<void> | void
    try {
      pending = checkOnce()
    } catch (error) {
      pending = Promise.reject(error)
    }
    void Promise.resolve(pending)
      .catch((error: unknown) => {
        console.debug("[EmptySidePanel] connection retry failed:", error)
      })
      .finally(() => {
        if (mountedRef.current) setIsRetrying(false)
      })
  }, [checkOnce])

  React.useEffect(() => {
    if (isRetrying || !restoreFocusRef.current) return
    restoreFocusRef.current = false
    const button = retryButtonRef.current
    if (!button || !button.isConnected) return
    const active = document.activeElement
    if (!active || active === document.body || active === button) {
      button.focus()
    }
  }, [isRetrying])

  return { retry, isRetrying, retryButtonRef }
}
