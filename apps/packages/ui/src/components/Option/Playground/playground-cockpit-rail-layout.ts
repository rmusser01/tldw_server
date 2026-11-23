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
  preferred,
  overlayDismissed = false
}: {
  /** The shell body's width in px, or null before it has been measured. */
  width: number | null
  contextVisible: boolean
  runtimeVisible: boolean
  /** The rail opened most recently; it wins when only one fits. */
  preferred: CockpitRail
  /**
   * The user closed an overlay this session: no rail overlays the chat until
   * they open one again, so closing one doesn't pop up the other.
   */
  overlayDismissed?: boolean
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
      layout[rail] = overlayDismissed ? "hidden" : "overlay"
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
    if (!element) return
    const apply = (next: number | undefined) => {
      if (window.innerWidth < COCKPIT_RAIL_COLUMNS_MIN_VIEWPORT) setWidth(null)
      else if (typeof next === "number" && next > 0) setWidth(Math.round(next))
    }
    // Measure now, before the first paint, so the layout doesn't flash
    // (AX-04 review); the observer follows later changes.
    apply(element.getBoundingClientRect().width)
    if (typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver((entries) => {
      apply(entries[entries.length - 1]?.contentRect.width)
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [ref])
  return width
}

const PREFERRED_RAIL_SESSION_KEY = "tldw:playgroundCockpitPreferredRail"

const readSessionRail = (): CockpitRail | null => {
  try {
    const value = window.sessionStorage.getItem(PREFERRED_RAIL_SESSION_KEY)
    return value === "context" || value === "runtime" ? value : null
  } catch {
    return null
  }
}

const writeSessionRail = (rail: CockpitRail) => {
  try {
    window.sessionStorage.setItem(PREFERRED_RAIL_SESSION_KEY, rail)
  } catch {
    // Storage blocked: the choice just doesn't survive a remount.
  }
}

const railElementId = (rail: CockpitRail) =>
  rail === "context"
    ? "playground-cockpit-left-rail"
    : "playground-cockpit-right-rail"
const restoreTestId = (rail: CockpitRail) => `${railElementId(rail)}-restore`

/**
 * The cockpit's rail state (CO-04, AX-04 review): the measured width, the
 * rail opened last (kept for the session, derived during render so a newly
 * opened rail shows in the same frame), overlay dismissal, and focus: an
 * overlay the user opens takes focus, and closing it returns focus to its
 * tab. The saved open/closed preferences change only through the callbacks.
 */
export const useCockpitRails = ({
  bodyRef,
  contextVisible,
  runtimeVisible,
  onContextVisibleChange,
  onRuntimeVisibleChange
}: {
  bodyRef: React.RefObject<HTMLElement | null>
  contextVisible: boolean
  runtimeVisible: boolean
  onContextVisibleChange?: (visible: boolean) => void
  onRuntimeVisibleChange?: (visible: boolean) => void
}) => {
  const width = useCockpitBodyWidth(bodyRef)
  const [preferred, setPreferred] = React.useState<CockpitRail>(
    () => readSessionRail() ?? "context"
  )
  const [seen, setSeen] = React.useState({
    context: contextVisible,
    runtime: runtimeVisible
  })
  if (seen.context !== contextVisible || seen.runtime !== runtimeVisible) {
    // Only one rail opening is the user's choice; both turning on together
    // is the saved preferences loading, which keeps the preferred rail.
    const contextOpened = contextVisible && !seen.context
    const runtimeOpened = runtimeVisible && !seen.runtime
    const opened =
      runtimeOpened && !contextOpened
        ? "runtime"
        : contextOpened && !runtimeOpened
          ? "context"
          : null
    setSeen({ context: contextVisible, runtime: runtimeVisible })
    if (opened && opened !== preferred) setPreferred(opened)
  }
  React.useEffect(() => {
    writeSessionRail(preferred)
  }, [preferred])
  const [overlayDismissed, setOverlayDismissed] = React.useState(false)
  const layout = resolveCockpitRailLayout({
    width,
    contextVisible,
    runtimeVisible,
    preferred,
    overlayDismissed
  })
  const focusTargetRef = React.useRef<{
    kind: "rail" | "restore"
    rail: CockpitRail
  } | null>(null)
  React.useEffect(() => {
    const target = focusTargetRef.current
    if (!target) return
    focusTargetRef.current = null
    if (target.kind === "rail") {
      if (layout[target.rail] !== "overlay") return
      document
        .getElementById(railElementId(target.rail))
        ?.querySelector<HTMLElement>("button")
        ?.focus()
      return
    }
    document
      .querySelector<HTMLElement>(`[data-testid="${restoreTestId(target.rail)}"]`)
      ?.focus()
  })
  const setVisible = (rail: CockpitRail, visible: boolean) =>
    rail === "context"
      ? onContextVisibleChange?.(visible)
      : onRuntimeVisibleChange?.(visible)
  const openRail = (rail: CockpitRail) => {
    setPreferred(rail)
    setOverlayDismissed(false)
    focusTargetRef.current = { kind: "rail", rail }
    setVisible(rail, true)
  }
  const closeRail = (rail: CockpitRail) => {
    if (layout[rail] === "overlay") {
      setOverlayDismissed(true)
      focusTargetRef.current = { kind: "restore", rail }
    }
    setVisible(rail, false)
  }
  /** Escape inside an overlay closes it, and only it. */
  const overlayKeyDown =
    (rail: CockpitRail) => (event: React.KeyboardEvent<HTMLElement>) => {
      if (event.key !== "Escape" || event.defaultPrevented) return
      event.preventDefault()
      event.stopPropagation()
      closeRail(rail)
    }
  return { layout, openRail, closeRail, overlayKeyDown }
}
