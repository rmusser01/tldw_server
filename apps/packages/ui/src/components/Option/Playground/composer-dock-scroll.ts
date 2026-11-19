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

/** Re-reveal the focused control in `scrollerRef` when `dockHeight` changes. */
export const useKeepFocusedComposerControlInView = (
  scrollerRef: React.RefObject<HTMLElement | null>,
  dockHeight: number | null | undefined
) => {
  React.useLayoutEffect(() => {
    const scroller = scrollerRef.current
    const focused = typeof document === "undefined" ? null : document.activeElement
    if (!scroller || !(focused instanceof HTMLElement)) return
    if (!scroller.contains(focused)) return
    revealInScroller(scroller, focused)
  }, [scrollerRef, dockHeight])
}
