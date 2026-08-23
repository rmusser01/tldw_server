import React from "react"
import {
  UNSAFE_DataRouterContext,
  unstable_usePrompt,
  useBlocker
} from "react-router-dom"

export type RouteLeavePromptProps = {
  when: boolean
  message: string
}

const ShimRouteLeavePrompt: React.FC<RouteLeavePromptProps> = (props) => {
  unstable_usePrompt(props)
  return null
}

export const RouteLeavePrompt: React.FC<RouteLeavePromptProps> = ({
  when,
  message
}) => {
  const dataRouterContext = React.useContext(UNSAFE_DataRouterContext)
  const blocker = useBlocker(Boolean(dataRouterContext) && when)
  const blockerRef = React.useRef(blocker)
  const proceedTimerRef = React.useRef<number | null>(null)
  blockerRef.current = blocker

  const cancelProceed = React.useCallback(() => {
    if (proceedTimerRef.current !== null) {
      window.clearTimeout(proceedTimerRef.current)
      proceedTimerRef.current = null
    }
  }, [])

  React.useEffect(() => {
    cancelProceed()
    if (blocker.state !== "blocked") return
    if (!when || !window.confirm(message)) {
      try {
        blocker.reset?.()
      } catch {
        // A stale router blocker is already safely unblocked.
      }
      return
    }
    const ownedBlocker = blocker
    const timer = window.setTimeout(() => {
      if (proceedTimerRef.current !== timer) return
      proceedTimerRef.current = null
      if (blockerRef.current !== ownedBlocker || blockerRef.current.state !== "blocked") return
      try {
        ownedBlocker.proceed?.()
      } catch {
        // Cleanup or a newer navigation already retired this blocker.
      }
    }, 0)
    proceedTimerRef.current = timer
    return cancelProceed
  }, [blocker, cancelProceed, message, when])

  React.useEffect(() => () => {
    cancelProceed()
    const current = blockerRef.current
    if (current.state === "blocked") {
      try {
        current.reset?.()
      } catch {
        // The owning router may already be disposed.
      }
    }
  }, [cancelProceed])

  return dataRouterContext ? null : <ShimRouteLeavePrompt when={when} message={message} />
}
