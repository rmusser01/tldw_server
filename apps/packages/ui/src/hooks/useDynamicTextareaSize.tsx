import { useCallback, useLayoutEffect, useState } from "react"

type TextareaStyleLike = {
  height: string
  maxHeight: string
  overflowY: string
}

export type DynamicTextareaLike = {
  scrollHeight: number
  style: TextareaStyleLike
}

export type DynamicTextareaApplyResult = {
  heightPx: number
  changed: boolean
}

/**
 * Applies textarea height/max-height styles while avoiding redundant writes.
 */
export const applyDynamicTextareaSize = (
  textarea: DynamicTextareaLike,
  maxHeight?: number,
  lastAppliedHeightPx?: number | null
): DynamicTextareaApplyResult => {
  // Reset before measuring so textarea can shrink after larger drafts.
  if (textarea.style.height !== "auto") {
    textarea.style.height = "auto"
  }
  const contentHeight = Math.max(0, Math.ceil(textarea.scrollHeight || 0))
  const hasMax = typeof maxHeight === "number" && Number.isFinite(maxHeight)
  const clampedMax = hasMax ? Math.max(0, Math.ceil(maxHeight as number)) : null
  const nextHeight = clampedMax != null ? Math.min(contentHeight, clampedMax) : contentHeight
  const nextOverflowY =
    clampedMax != null && contentHeight > clampedMax ? "scroll" : "hidden"

  if (clampedMax != null) {
    const nextMaxHeight = `${clampedMax}px`
    if (textarea.style.maxHeight !== nextMaxHeight) {
      textarea.style.maxHeight = nextMaxHeight
    }
  } else if (textarea.style.maxHeight !== "") {
    textarea.style.maxHeight = ""
  }

  if (textarea.style.overflowY !== nextOverflowY) {
    textarea.style.overflowY = nextOverflowY
  }

  const changed = lastAppliedHeightPx == null || lastAppliedHeightPx !== nextHeight
  const nextHeightCss = `${nextHeight}px`
  if (changed || textarea.style.height !== nextHeightCss) {
    textarea.style.height = nextHeightCss
  }

  return { heightPx: nextHeight, changed }
}

/**
 * Fits the mounted textarea to content and width, within its existing cap.
 * Return the DOM ref callback; the supplied object remains available for focus.
 */
const useDynamicTextareaSize = (
  textareaRef: React.MutableRefObject<HTMLTextAreaElement | null>,
  textContent: string,
  maxHeight?: number
): React.RefCallback<HTMLTextAreaElement> => {
  const [textarea, setTextarea] = useState<HTMLTextAreaElement | null>(null)
  const attachTextarea = useCallback(
    (node: HTMLTextAreaElement | null) => {
      textareaRef.current = node
      setTextarea(node)
    },
    [textareaRef]
  )

  useLayoutEffect(() => {
    if (typeof window === "undefined" || !textarea) return
    let active = true
    let raf: number | null = null
    let lastAppliedHeight: number | null = null
    let width: number | null = null
    const schedule = () => {
      if (!active || textareaRef.current !== textarea || raf != null) return
      raf = window.requestAnimationFrame(() => {
        raf = null
        if (!active || textareaRef.current !== textarea) return
        lastAppliedHeight = applyDynamicTextareaSize(
          textarea,
          maxHeight,
          lastAppliedHeight
        ).heightPx
      })
    }
    const observer =
      typeof window.ResizeObserver === "function"
        ? new window.ResizeObserver((entries) => {
            for (const entry of entries) {
              if (
                entry.target !== textarea ||
                entry.contentRect.width === width
              )
                continue
              width = entry.contentRect.width
              schedule()
            }
          })
        : null
    observer?.observe(textarea)
    if (!observer) window.addEventListener("resize", schedule)
    schedule()

    return () => {
      active = false
      observer?.disconnect()
      if (!observer) window.removeEventListener("resize", schedule)
      if (raf != null) window.cancelAnimationFrame(raf)
    }
  }, [textarea, textareaRef, textContent, maxHeight])

  return attachTextarea
}

export default useDynamicTextareaSize
