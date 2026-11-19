import React from "react"

/**
 * Width-aware cockpit rails (CO-04, #3113).
 *
 * The Context and Runtime rails were fixed grid columns from `lg`, so at
 * 1024px with both open (and the app sidebar) the transcript got a few
 * hundred pixels. The shell now measures its own width:
 * - wide enough for both rails and a usable transcript: both inline;
 * - otherwise one rail at a time, the one the user opened last;
 * - and when even one inline rail would squeeze the transcript, that rail
 *   overlays the transcript instead.
 * The saved rail preferences are inputs only; this never changes them.
 */
export type CockpitRail = "context" | "runtime"
export type CockpitRailPlacement = "inline" | "overlay" | "hidden"

/** The narrowest transcript the cockpit keeps beside inline rails. */
export const COCKPIT_TRANSCRIPT_MIN_WIDTH = 560
/** Upper bounds of the rails' grid columns (PlaygroundCockpitShell). */
export const COCKPIT_CONTEXT_RAIL_MAX_WIDTH = 280
export const COCKPIT_RUNTIME_RAIL_MAX_WIDTH = 300
export const COCKPIT_BOTH_RAILS_MIN_WIDTH =
  COCKPIT_TRANSCRIPT_MIN_WIDTH +
  COCKPIT_CONTEXT_RAIL_MAX_WIDTH +
  COCKPIT_RUNTIME_RAIL_MAX_WIDTH
export const COCKPIT_ONE_RAIL_MIN_WIDTH =
  COCKPIT_TRANSCRIPT_MIN_WIDTH + COCKPIT_RUNTIME_RAIL_MAX_WIDTH

const railMaxWidth = (rail: CockpitRail) =>
  rail === "context"
    ? COCKPIT_CONTEXT_RAIL_MAX_WIDTH
    : COCKPIT_RUNTIME_RAIL_MAX_WIDTH

export const resolveCockpitRailLayout = ({
  width,
  contextVisible,
  runtimeVisible,
  preferred
}: {
  /** The shell body's width in px, or null before it has been measured. */
  width: number | null
  contextVisible: boolean
  runtimeVisible: boolean
  /** The rail opened most recently; it wins when only one fits. */
  preferred: CockpitRail
}): Record<CockpitRail, CockpitRailPlacement> => {
  const layout: Record<CockpitRail, CockpitRailPlacement> = {
    context: contextVisible ? "inline" : "hidden",
    runtime: runtimeVisible ? "inline" : "hidden"
  }
  if (width === null) return layout
  if (contextVisible && runtimeVisible) {
    if (width >= COCKPIT_BOTH_RAILS_MIN_WIDTH) return layout
    layout[preferred === "context" ? "runtime" : "context"] = "hidden"
  }
  for (const rail of ["context", "runtime"] as const) {
    if (
      layout[rail] === "inline" &&
      width < COCKPIT_TRANSCRIPT_MIN_WIDTH + railMaxWidth(rail)
    ) {
      layout[rail] = "overlay"
    }
  }
  return layout
}

/** Below `lg` the rails are tab panels above the chat, not columns. */
const COCKPIT_RAIL_COLUMNS_MIN_VIEWPORT = 1024

/**
 * The cockpit body's content width from ResizeObserver, or null until it is
 * measured and while the window is narrower than `lg`.
 */
export const useCockpitBodyWidth = (
  ref: React.RefObject<HTMLElement | null>
): number | null => {
  const [width, setWidth] = React.useState<number | null>(null)
  React.useLayoutEffect(() => {
    const element = ref.current
    if (!element || typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver((entries) => {
      const next = entries[entries.length - 1]?.contentRect.width
      if (window.innerWidth < COCKPIT_RAIL_COLUMNS_MIN_VIEWPORT) setWidth(null)
      else if (typeof next === "number" && next > 0) setWidth(Math.round(next))
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [ref])
  return width
}
