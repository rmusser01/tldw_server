import React from "react"

/**
 * Returns `true` from the first render where `active` is true, and keeps
 * returning `true` afterwards.
 *
 * Used to keep a lazily loaded overlay host mounted once it has been needed,
 * so its dialogs can finish their close animation and hand focus back to the
 * control that opened them instead of dropping it on `<body>` (AX-09).
 */
export function useLatchedFlag(active: boolean): boolean {
  const [latched, setLatched] = React.useState(active)
  if (active && !latched) {
    // Adjusting state while rendering is React's documented pattern for
    // deriving state from a changed prop without an extra effect pass.
    setLatched(true)
  }
  return latched || active
}
