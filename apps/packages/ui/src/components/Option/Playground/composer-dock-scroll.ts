import React from "react"

/**
 * Keep the focused composer control inside the dock's scroll area (AX-04,
 * #3124).
 *
 * /chat focuses the textarea on load while the composer dock is still small,
 * so the dock scrolls the textarea's bottom into view. When the dock then grows
 * the stale offset can leave the textarea above the visible area; at 320x256
 * (400% zoom) nothing showed it. Revealing the focused control again whenever
 * the dock resizes fixes that, without touching the page's other scrollers.
 */
export const revealInScroller = (
  scroller: HTMLElement,
  element: HTMLElement
) => {
  const area = scroller.getBoundingClientRect()
  const box = element.getBoundingClientRect()
  if (box.top < area.top) {
    scroller.scrollTop -= area.top - box.top
  } else if (box.bottom > area.bottom) {
    // Scroll just enough to show it, but never past its top.
    scroller.scrollTop += Math.min(box.bottom - area.bottom, box.top - area.top)
  }
}

/**
 * Re-reveal the focused control in `scrollerRef` when `dockHeight` changes.
 * When it sits in the composer row (`data-composer-row`: the textarea and
 * Send) and the row fits, the whole row is revealed. A scroll the user made
 * inside the dock since the last resize is left alone (AX-04 review).
 */
export const useKeepFocusedComposerControlInView = (
  scrollerRef: React.RefObject<HTMLElement | null>,
  dockHeight: number | null | undefined
) => {
  const userScrolledRef = React.useRef(false)
  // The position our own reveal set, so its scroll event isn't the user's.
  const revealedScrollTopRef = React.useRef<number | null>(null)
  React.useEffect(() => {
    const scroller = scrollerRef.current
    if (!scroller) return
    const onScroll = () => {
      if (scroller.scrollTop !== revealedScrollTopRef.current)
        userScrolledRef.current = true
    }
    scroller.addEventListener("scroll", onScroll, { passive: true })
    return () => scroller.removeEventListener("scroll", onScroll)
  }, [scrollerRef])
  React.useLayoutEffect(() => {
    const scroller = scrollerRef.current
    const userScrolled = userScrolledRef.current
    userScrolledRef.current = false
    if (!scroller || userScrolled) return
    const focused = typeof document === "undefined" ? null : document.activeElement
    if (!(focused instanceof HTMLElement) || !scroller.contains(focused)) return
    const row = focused.closest<HTMLElement>("[data-composer-row]")
    const target =
      row &&
      scroller.contains(row) &&
      row.getBoundingClientRect().height <= scroller.clientHeight
        ? row
        : focused
    revealInScroller(scroller, target)
    revealedScrollTopRef.current = scroller.scrollTop
  }, [scrollerRef, dockHeight])
}
