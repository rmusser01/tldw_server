import { useCallback, useEffect, useRef, useState } from "react"

const OPTIONAL_REGION_FLOOR_PX = 56

// Observe the occupied dock, including CSS caps and border/padding, not its Form.
// It also records the height of everything but the optional region
// (`[data-sidepanel-dock-optional]`) as --sp-dock-fixed-h, so a dock capped at
// 40% grows to fit the textarea and Send (XS-15).
export function useComposerDockHeight() {
  const [height, setHeight] = useState(0)
  const nodeRef = useRef<HTMLDivElement | null>(null)
  const observerRef = useRef<ResizeObserver | null>(null)
  const ref = useCallback((node: HTMLDivElement | null) => {
    observerRef.current?.disconnect()
    observerRef.current = null
    nodeRef.current = node
    if (!node) {
      setHeight(0)
      return
    }
    const measure = () => {
      if (nodeRef.current !== node) return
      setHeight(Math.max(0, Math.ceil(node.getBoundingClientRect().height)))
      const content = node.firstElementChild as HTMLElement | null
      const optional = node.querySelector<HTMLElement>("[data-sidepanel-dock-optional]")
      if (content && optional) {
        // Intrinsic heights (scrollHeight counts rows the cap clips; the optional
        // region scrolls, so only its visible box is in it), and a
        // little of the optional region kept in view so a hint stays reachable.
        const optionalFloor = Math.min(optional.scrollHeight, OPTIONAL_REGION_FLOOR_PX)
        const fixed = Math.max(0, Math.ceil(content.scrollHeight - optional.clientHeight + optionalFloor))
        node.style.setProperty("--sp-dock-fixed-h", `${fixed}px`)
      }
    }
    measure()
    if (typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver(() => {
        if (observerRef.current === observer) measure()
      })
      observerRef.current = observer
      observer.observe(node, { box: "border-box" })
      if (node.firstElementChild) observer.observe(node.firstElementChild)
      const essential = node.querySelector("[data-sidepanel-dock-essential]")
      if (essential) observer.observe(essential)
    }
  }, [])
  useEffect(() => {
    // Strict Mode restores effects while the same DOM ref stays attached.
    if (nodeRef.current) ref(nodeRef.current)
    return () => {
      observerRef.current?.disconnect()
      observerRef.current = null
    }
  }, [ref])
  return { ref, height }
}
