/** Media query for the OS "reduce motion" accessibility setting. */
export const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)"

/**
 * Whether the user asked the OS to reduce motion. False when matchMedia is
 * unavailable (server rendering, old test environments).
 */
export function prefersReducedMotion(): boolean {
  try {
    return (
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia(REDUCED_MOTION_QUERY).matches
    )
  } catch {
    return false
  }
}

/**
 * Scroll behaviour for scripted jumps (scrollIntoView, scrollTo): "smooth"
 * unless the user asked to reduce motion (AX-19). CSS scroll-behavior does not
 * override a behaviour passed from script, so callers must ask here.
 */
export function smoothScrollBehavior(): ScrollBehavior {
  return prefersReducedMotion() ? "auto" : "smooth"
}
