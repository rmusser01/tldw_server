import { useEffect, useState } from "react"

export const WATCHLISTS_CONSTRAINED_BREAKPOINT = 768

export const isWatchlistsConstrainedViewport = (width?: number): boolean =>
  Number.isFinite(width) && Number(width) >= 0 && Number(width) < WATCHLISTS_CONSTRAINED_BREAKPOINT

const getInitialConstrainedViewport = (): boolean => {
  if (typeof window === "undefined") return false
  return isWatchlistsConstrainedViewport(window.innerWidth)
}

export const useWatchlistsViewport = () => {
  const [isConstrained, setIsConstrained] = useState(getInitialConstrainedViewport)

  useEffect(() => {
    if (typeof window === "undefined") return

    const query = `(max-width: ${WATCHLISTS_CONSTRAINED_BREAKPOINT - 1}px)`
    if (typeof window.matchMedia === "function") {
      const mediaQuery = window.matchMedia(query)
      const handleChange = (event: MediaQueryListEvent) => setIsConstrained(event.matches)
      setIsConstrained(mediaQuery.matches)
      if (typeof mediaQuery.addEventListener === "function") {
        mediaQuery.addEventListener("change", handleChange)
        return () => mediaQuery.removeEventListener("change", handleChange)
      }
      if (typeof mediaQuery.addListener === "function") {
        mediaQuery.addListener(handleChange)
        return () => mediaQuery.removeListener(handleChange)
      }
    }

    const handleResize = () => setIsConstrained(isWatchlistsConstrainedViewport(window.innerWidth))
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  return {
    breakpoint: WATCHLISTS_CONSTRAINED_BREAKPOINT,
    isConstrained
  }
}
