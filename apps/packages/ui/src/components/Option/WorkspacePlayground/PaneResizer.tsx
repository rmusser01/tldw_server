import React from "react"

const MIN_LEFT_WIDTH = 200
const MIN_RIGHT_WIDTH = 240
const DEFAULT_LEFT_WIDTH = 288
const DEFAULT_RIGHT_WIDTH = 320

interface PaneResizerProps {
  /** Which pane this resizer controls */
  pane: "left" | "right"
  /** Current width in px */
  width: number
  /** Called with the new width during drag */
  onResize: (width: number) => void
  /** Called on double-click to reset to default */
  onReset?: () => void
}

export { DEFAULT_LEFT_WIDTH, DEFAULT_RIGHT_WIDTH, MIN_LEFT_WIDTH, MIN_RIGHT_WIDTH }

export const PaneResizer: React.FC<PaneResizerProps> = ({
  pane,
  width,
  onResize,
  onReset
}) => {
  const isDragging = React.useRef(false)
  const startX = React.useRef(0)
  const startWidth = React.useRef(0)
  const [hovering, setHovering] = React.useState(false)

  const minWidth = pane === "left" ? MIN_LEFT_WIDTH : MIN_RIGHT_WIDTH

  const handlePointerDown = React.useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault()
      isDragging.current = true
      startX.current = e.clientX
      startWidth.current = width

      const target = e.currentTarget as HTMLElement
      target.setPointerCapture(e.pointerId)

      const handlePointerMove = (moveEvent: PointerEvent) => {
        if (!isDragging.current) return
        const delta = pane === "left"
          ? moveEvent.clientX - startX.current
          : startX.current - moveEvent.clientX
        const newWidth = Math.max(minWidth, startWidth.current + delta)
        onResize(newWidth)
      }

      const handlePointerUp = () => {
        isDragging.current = false
        target.releasePointerCapture(e.pointerId)
        target.removeEventListener("pointermove", handlePointerMove)
        target.removeEventListener("pointerup", handlePointerUp)
      }

      target.addEventListener("pointermove", handlePointerMove)
      target.addEventListener("pointerup", handlePointerUp)
    },
    [pane, width, minWidth, onResize]
  )

  const handleDoubleClick = React.useCallback(() => {
    onReset?.()
  }, [onReset])

  return (
    <div
      data-testid={`pane-resizer-${pane}`}
      role="separator"
      aria-orientation="vertical"
      aria-valuenow={width}
      aria-valuemin={minWidth}
      title="Double-click to reset width"
      className={`hidden cursor-col-resize touch-none select-none items-center justify-center lg:flex ${
        hovering || isDragging.current ? "w-2" : "w-1"
      } transition-all`}
      onPointerDown={handlePointerDown}
      onDoubleClick={handleDoubleClick}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
    >
      <div
        className={`h-8 rounded-full transition-all ${
          hovering || isDragging.current
            ? "w-1 bg-primary/50"
            : "w-0.5 bg-border/40"
        }`}
      />
    </div>
  )
}
