import { useCallback, useEffect, useRef, useState } from "react"

// Observe the occupied dock, including CSS caps and border/padding, not its Form.
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
      if (nodeRef.current === node) {
        setHeight(Math.max(0, Math.ceil(node.getBoundingClientRect().height)))
      }
    }
    measure()
    if (typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver(() => {
        if (observerRef.current === observer) measure()
      })
      observerRef.current = observer
      observer.observe(node, { box: "border-box" })
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
