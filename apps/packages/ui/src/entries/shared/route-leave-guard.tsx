import React from "react"
import { UNSAFE_DataRouterContext, useBlocker } from "react-router-dom"

export type RouteLeaveGuardProps = {
  /** Hold in-app navigation to another page while true. */
  when: boolean
  /**
   * Finish the page's pending work. Resolve true to continue the navigation,
   * false (or throw) to stay on the page.
   */
  onLeave: () => Promise<boolean>
}

type BlockerArgs = {
  currentLocation: { pathname: string }
  nextLocation: { pathname: string }
}

// The web build maps react-router-dom to a Next router shim whose useBlocker
// works without a data router; react-router's own needs one (the extension).
const nextRouterShimCanBlock =
  (useBlocker as unknown as { tldwNextRouterShim?: boolean }).tldwNextRouterShim === true

const ActiveRouteLeaveGuard: React.FC<RouteLeaveGuardProps> = ({ when, onLeave }) => {
  const whenRef = React.useRef(when)
  whenRef.current = when
  const onLeaveRef = React.useRef(onLeave)
  onLeaveRef.current = onLeave
  const shouldBlock = React.useCallback(
    ({ currentLocation, nextLocation }: BlockerArgs) =>
      whenRef.current && currentLocation.pathname !== nextLocation.pathname,
    []
  )
  const blocker = useBlocker(shouldBlock)
  const blockerRef = React.useRef(blocker)
  blockerRef.current = blocker
  const settlingRef = React.useRef(false)

  React.useEffect(() => {
    if (blocker.state !== "blocked" || settlingRef.current) return
    settlingRef.current = true
    void (async () => {
      let leave = false
      try {
        leave = await onLeaveRef.current()
      } catch {
        leave = false
      } finally {
        settlingRef.current = false
      }
      const current = blockerRef.current
      if (current.state !== "blocked") return
      try {
        if (leave) current.proceed?.()
        else current.reset?.()
      } catch {
        // A newer navigation or router disposal already retired this blocker.
      }
    })()
  }, [blocker])

  React.useEffect(
    () => () => {
      const current = blockerRef.current
      if (current.state !== "blocked") return
      try {
        current.reset?.()
      } catch {
        // The owning router may already be disposed.
      }
    },
    []
  )

  return null
}

/**
 * Async counterpart of RouteLeavePrompt: instead of a synchronous confirm it
 * holds the navigation, awaits `onLeave`, then continues or stays. Only page
 * changes are held; query and hash updates on the same page pass through.
 */
export const RouteLeaveGuard: React.FC<RouteLeaveGuardProps> = (props) => {
  const dataRouterContext = React.useContext(UNSAFE_DataRouterContext)
  if (!dataRouterContext && !nextRouterShimCanBlock) return null
  return <ActiveRouteLeaveGuard {...props} />
}
