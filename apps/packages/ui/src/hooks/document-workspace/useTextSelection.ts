import { useState, useEffect, useCallback } from "react"

export interface TextSelection {
  text: string
  rect: DOMRect
  page?: number
}

/** Keep the captured PDF passage when focus moves from the viewer to an action. */
export function useTextSelection(containerRef: React.RefObject<HTMLElement | null>) {
  const [selection, setSelection] = useState<TextSelection | null>(null)

  const clearSelection = useCallback(() => {
    setSelection(null)
    const native = window.getSelection()
    if (native?.anchorNode && containerRef.current?.contains(native.anchorNode)) {
      native.removeAllRanges()
    }
  }, [containerRef])

  useEffect(() => {
    let selectingWithMouse = false

    const readSelection = () => {
      const native = window.getSelection()
      const viewer = containerRef.current
      if (selectingWithMouse || !viewer || !native || native.isCollapsed || !native.rangeCount ||
          !viewer.contains(native.anchorNode) || !viewer.contains(native.focusNode)) return

      const range = native.getRangeAt(0)
      const start = range.startContainer
      const element = start instanceof Element ? start : start.parentElement
      if (element?.closest("[data-selection-popover]")) return

      const text = native.toString().trim()
      if (!text) return
      const page = Number(element?.closest("[data-page-number]")?.getAttribute("data-page-number"))
      setSelection({ text, rect: range.getBoundingClientRect(), page: page > 0 ? page : undefined })
    }

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target
      selectingWithMouse = event.pointerType === "mouse" && target instanceof Element &&
        Boolean(containerRef.current?.contains(target)) && !target.closest("[data-selection-popover]")
    }
    const handlePointerUp = () => {
      if (!selectingWithMouse) return
      selectingWithMouse = false
      readSelection()
    }
    const handlePointerCancel = () => { selectingWithMouse = false }

    document.addEventListener("selectionchange", readSelection)
    document.addEventListener("pointerdown", handlePointerDown)
    document.addEventListener("pointerup", handlePointerUp)
    document.addEventListener("pointercancel", handlePointerCancel)

    return () => {
      document.removeEventListener("selectionchange", readSelection)
      document.removeEventListener("pointerdown", handlePointerDown)
      document.removeEventListener("pointerup", handlePointerUp)
      document.removeEventListener("pointercancel", handlePointerCancel)
    }
  }, [containerRef])

  return { selection, clearSelection }
}
