import React from "react"

/** The transcript's error bubble carries this attribute (Message.tsx). */
export const CHAT_ERROR_BUBBLE_ATTRIBUTE = "data-chat-error-bubble"

/**
 * Whether the transcript's latest error bubble is on screen, so the composer
 * banner doesn't repeat it (CC-N2). It watches the bubble itself rather than
 * the scroll position, because the banner's own height moves the composer and
 * so the "at bottom" threshold.
 *
 * Returns null until known, false when the bubble is off screen or not in the
 * page, and true when it is visible (or when the browser can't tell).
 */
export const useChatErrorBubbleInView = (errorKey: string | null): boolean | null => {
  const [inView, setInView] = React.useState<boolean | null>(null)
  // A layout effect runs after the transcript in the same commit has rendered
  // the bubble, so a missing bubble is known before paint.
  React.useLayoutEffect(() => {
    if (!errorKey || typeof document === "undefined") {
      setInView(null)
      return
    }
    const bubbles = document.querySelectorAll<HTMLElement>(`[${CHAT_ERROR_BUBBLE_ATTRIBUTE}]`)
    const latest = bubbles[bubbles.length - 1]
    if (!latest) {
      setInView(false)
      return
    }
    if (typeof IntersectionObserver === "undefined") {
      setInView(true)
      return
    }
    setInView(null)
    const observer = new IntersectionObserver(
      (entries) => setInView(entries.some((entry) => entry.isIntersecting)),
      { threshold: 0.25 }
    )
    observer.observe(latest)
    return () => observer.disconnect()
  }, [errorKey])
  return inView
}
