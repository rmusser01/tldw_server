import React from 'react'

/** Pixels of hidden content we ignore, so sub-pixel rounding never shows a cue. */
const CONTENT_BELOW_TOLERANCE_PX = 4

/**
 * Reports whether a scroll container has content hidden below its visible
 * area. Used to show a "more below" cue instead of clipping controls silently.
 *
 * Re-checks on scroll, window resize, size changes and DOM changes inside the
 * container (sections expanding or collapsing).
 */
export const useHasContentBelow = (
  ref: React.RefObject<HTMLElement | null>
): boolean => {
  const [hasContentBelow, setHasContentBelow] = React.useState(false)

  React.useLayoutEffect(() => {
    const node = ref.current
    if (!node) return
    const update = () => {
      const hidden = node.scrollHeight - node.clientHeight - node.scrollTop
      setHasContentBelow(hidden > CONTENT_BELOW_TOLERANCE_PX)
    }
    update()
    node.addEventListener('scroll', update, { passive: true })
    window.addEventListener('resize', update)
    const resizeObserver =
      typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(update)
    resizeObserver?.observe(node)
    const mutationObserver =
      typeof MutationObserver === 'undefined' ? null : new MutationObserver(update)
    mutationObserver?.observe(node, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['hidden', 'aria-expanded']
    })
    return () => {
      node.removeEventListener('scroll', update)
      window.removeEventListener('resize', update)
      resizeObserver?.disconnect()
      mutationObserver?.disconnect()
    }
  }, [ref])

  return hasContentBelow
}
