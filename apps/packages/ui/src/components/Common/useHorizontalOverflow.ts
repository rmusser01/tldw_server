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

/**
 * The box's own size, its direct children and any table inside it are
 * watched, plus later DOM changes inside it, so the measurement stays current
 * without being rebuilt on every render of streamed content.
 */
export const useHorizontalOverflow = (
  ref: React.RefObject<HTMLElement>
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

    const watched = new Set<Element>()
    const resizeObserver =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update)
    const watch = (target: Element) => {
      if (!resizeObserver || watched.has(target)) return
      watched.add(target)
      resizeObserver.observe(target)
    }
    const watchContent = () => {
      Array.from(element.children).forEach(watch)
      element.querySelectorAll("table").forEach(watch)
    }
    watch(element)
    watchContent()

    const mutationObserver =
      typeof MutationObserver === "undefined"
        ? null
        : new MutationObserver(() => {
            watchContent()
            update()
          })
    mutationObserver?.observe(element, { childList: true, subtree: true })

    return () => {
      element.removeEventListener("scroll", update)
      mutationObserver?.disconnect()
      resizeObserver?.disconnect()
    }
  }, [ref])

  return state
}

const MAX_TABLE_NAME_LENGTH = 60

const readTableName = (element: HTMLElement | null): string | null => {
  const source =
    element?.querySelector("caption") ?? element?.querySelector("th")
  const text = source?.textContent?.replace(/\s+/g, " ").trim()
  if (!text) return null
  return text.length > MAX_TABLE_NAME_LENGTH
    ? `${text.slice(0, MAX_TABLE_NAME_LENGTH - 1)}…`
    : text
}

/**
 * A short name for a scrollable table, from its caption or else its first
 * header cell (XS-14), so several wide tables in one reply are not all
 * announced as the same "Table" region. Read only while `active`.
 */
export const useTableName = (
  ref: React.RefObject<HTMLElement>,
  active: boolean
): string | null => {
  const [name, setName] = React.useState<string | null>(null)
  React.useLayoutEffect(() => {
    if (!active) return
    setName(readTableName(ref.current))
  }, [active, ref])
  return name
}
