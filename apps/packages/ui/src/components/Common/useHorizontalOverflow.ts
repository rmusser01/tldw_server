import React from "react"

/**
 * Tracks whether a horizontally scrollable box has content hidden past its
 * edges (XS-14). Used to tell readers that a wide table continues to the
 * right instead of silently clipping its last columns.
 */
export type HorizontalOverflow = {
  /** The content is wider than the box. */
  overflows: boolean
  /** Some content is still hidden to the right of the visible area. */
  hasMoreRight: boolean
}

const NONE: HorizontalOverflow = { overflows: false, hasMoreRight: false }

// Sub-pixel rounding can leave a 1px difference on a box that fits.
const TOLERANCE_PX = 1

const measure = (element: HTMLElement | null): HorizontalOverflow => {
  if (!element) return NONE
  const hidden = element.scrollWidth - element.clientWidth
  if (hidden <= TOLERANCE_PX) return NONE
  return {
    overflows: true,
    hasMoreRight: hidden - element.scrollLeft > TOLERANCE_PX
  }
}

export const useHorizontalOverflow = (
  ref: React.RefObject<HTMLElement>,
  /** Re-measure when this changes, for example the rendered content. */
  contentKey?: unknown
): HorizontalOverflow => {
  const [state, setState] = React.useState<HorizontalOverflow>(NONE)

  React.useLayoutEffect(() => {
    const element = ref.current
    if (!element) return
    const update = () => {
      const next = measure(element)
      setState((previous) =>
        previous.overflows === next.overflows &&
        previous.hasMoreRight === next.hasMoreRight
          ? previous
          : next
      )
    }
    update()
    element.addEventListener("scroll", update, { passive: true })
    let observer: ResizeObserver | null = null
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(update)
      observer.observe(element)
      Array.from(element.children).forEach((child) => observer?.observe(child))
    }
    return () => {
      element.removeEventListener("scroll", update)
      observer?.disconnect()
    }
  }, [ref, contentKey])

  return state
}
