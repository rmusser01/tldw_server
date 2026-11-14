import React from "react"

/**
 * Remembers which element had focus before a transient surface (such as the
 * in-thread search bar) took it, and gives it back when the surface closes.
 *
 * When the remembered element is gone, `fallback` runs instead (for example,
 * focusing the composer), so focus never drops to `<body>` (AX-09).
 */
export function useFocusReturn(fallback: () => void) {
  const returnTargetRef = React.useRef<HTMLElement | null>(null)

  const rememberFocus = React.useCallback(() => {
    if (typeof document === "undefined") return
    const active = document.activeElement
    returnTargetRef.current =
      active instanceof HTMLElement && active !== document.body ? active : null
  }, [])

  const restoreFocus = React.useCallback(() => {
    const target = returnTargetRef.current
    returnTargetRef.current = null
    // Wait a frame so the closing surface has unmounted before focus moves.
    requestAnimationFrame(() => {
      if (target?.isConnected) {
        target.focus()
        return
      }
      fallback()
    })
  }, [fallback])

  return { rememberFocus, restoreFocus }
}
