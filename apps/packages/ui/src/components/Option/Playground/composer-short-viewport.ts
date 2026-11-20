import React from "react"

/**
 * Short-viewport composer (AX-04, #3124; owner decision [14]).
 *
 * At 200-400% zoom the window is 225-400 CSS px tall and the composer filled
 * it: at 320x256 the transcript got 0-54px. When the viewport is at most 500px
 * tall, the composer's toolbar row collapses first, behind one "Composer
 * options" button, and Send moves up beside the textarea. The transcript keeps
 * min(42dvh, 45% of the chat column) and the dock at least 4.5rem
 * (Playground.tsx); the dock reveals the whole composer row
 * (composer-dock-scroll.ts). Opening "Composer options" brings the row back on
 * its own line. In the cockpit, a mobile rail panel overlays the chat.
 *
 * Literal class strings, so Tailwind's content scan finds them.
 */

/** Hide a toolbar part while the short-viewport row is collapsed. */
export const SHORT_VIEWPORT_COLLAPSE = "[@media(max-height:500px)]:hidden"

/** The collapsed toolbar takes only its own width, beside the textarea. */
export const SHORT_VIEWPORT_INLINE = "[@media(max-height:500px)]:w-auto"

/** Send's group keeps its own width instead of filling the row. */
export const SHORT_VIEWPORT_INLINE_GROUP =
  "[@media(max-height:500px)]:w-auto [@media(max-height:500px)]:flex-none"

/** PlaygroundForm: the textarea and the toolbar share one wrapping row. */
export const SHORT_VIEWPORT_COMPOSER_ROW =
  "[@media(max-height:500px)]:flex-row [@media(max-height:500px)]:flex-wrap [@media(max-height:500px)]:items-end"

/** PlaygroundForm: the textarea fills the row, but never below 6rem. */
export const SHORT_VIEWPORT_INPUT =
  "[@media(max-height:500px)]:min-w-[6rem] [@media(max-height:500px)]:flex-1"

/** PlaygroundForm: the toolbar's slot sizes to its content in that row. */
export const SHORT_VIEWPORT_TOOLBAR_SLOT =
  "[@media(max-height:500px)]:w-auto [@media(max-height:500px)]:max-w-full"

/** Cockpit mobile rail panel: overlays the chat instead of taking its space. */
export const SHORT_VIEWPORT_OVERLAY_PANEL =
  "[@media(max-height:500px)]:absolute [@media(max-height:500px)]:inset-x-2 [@media(max-height:500px)]:top-full [@media(max-height:500px)]:z-40 [@media(max-height:500px)]:max-h-[min(40dvh,calc(100dvh-13rem))] [@media(max-height:500px)]:rounded-b-md [@media(max-height:500px)]:border [@media(max-height:500px)]:shadow-xl"

const SHORT_VIEWPORT_QUERY = "(max-height: 500px)"

/** Whether the viewport is short (the same query as the classes above). */
export const useIsShortViewport = (): boolean => {
  const read = () =>
    typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? window.matchMedia(SHORT_VIEWPORT_QUERY).matches
      : false
  const [short, setShort] = React.useState(read)
  React.useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function")
      return
    const query = window.matchMedia(SHORT_VIEWPORT_QUERY)
    const update = () => setShort(query.matches)
    update()
    query.addEventListener?.("change", update)
    return () => query.removeEventListener?.("change", update)
  }, [])
  return short
}
