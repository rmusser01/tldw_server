import React from "react"

/**
 * True once `active` has stayed true for `delayMs`, so a short wait shows no
 * indicator (CM-14). Each new active period waits again, and the timer is
 * cleared when `active` turns false or the component unmounts.
 */
export const useShowAfterDelay = (active: boolean, delayMs: number): boolean => {
  const [shown, setShown] = React.useState(false)
  React.useEffect(() => {
    setShown(false)
    if (!active) return
    const timer = setTimeout(() => setShown(true), delayMs)
    return () => clearTimeout(timer)
  }, [active, delayMs])
  return active && shown
}
