import React from "react"

interface ResizablePanelsProps {
  /** Left panel (filter sidebar) */
  left: React.ReactNode
  /** Center panel (results list) */
  center: React.ReactNode
  /** Right panel (reading pane) */
  right: React.ReactNode
  /** Whether to collapse to single-panel mode (mobile) */
  collapsed?: boolean
  /** Minimum width in px for each panel */
  minLeft?: number
  minCenter?: number
  /** Default width in px for left and center panels */
  defaultLeft?: number
  defaultCenter?: number
  /** Labels for mobile panel switcher tabs */
  mobileTab?: 0 | 1 | 2
  onMobileTabChange?: (tab: 0 | 1 | 2) => void
  tabLabels?: [string, string, string]
}

/**
 * Three-panel resizable layout using CSS grid with drag handles.
 * Left panel: filter sidebar, Center: results list, Right: reading pane (fills remaining space).
 */
export const ResizablePanels: React.FC<ResizablePanelsProps> = ({
  left,
  center,
  right,
  collapsed = false,
  minLeft = 180,
  minCenter = 280,
  defaultLeft = 220,
  defaultCenter = 320,
  mobileTab: controlledMobileTab,
  onMobileTabChange,
  tabLabels = ["Filters", "Results", "Content"]
}) => {
  const [leftWidth, setLeftWidth] = React.useState(defaultLeft)
  const [centerWidth, setCenterWidth] = React.useState(defaultCenter)
  const [localMobileTab, setMobileTab] = React.useState<0 | 1 | 2>(1)
  const mobileTab = controlledMobileTab ?? localMobileTab
  const draggingRef = React.useRef<"left" | "center" | null>(null)
  const startXRef = React.useRef(0)
  const startWidthRef = React.useRef(0)

  const handleMouseDown = React.useCallback(
    (handle: "left" | "center", e: React.MouseEvent) => {
      e.preventDefault()
      draggingRef.current = handle
      startXRef.current = e.clientX
      startWidthRef.current = handle === "left" ? leftWidth : centerWidth
    },
    [leftWidth, centerWidth]
  )

  React.useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!draggingRef.current) return
      const delta = e.clientX - startXRef.current
      if (draggingRef.current === "left") {
        setLeftWidth(Math.max(minLeft, startWidthRef.current + delta))
      } else {
        setCenterWidth(Math.max(minCenter, startWidthRef.current + delta))
      }
    }
    const handleMouseUp = () => {
      draggingRef.current = null
    }
    document.addEventListener("mousemove", handleMouseMove)
    document.addEventListener("mouseup", handleMouseUp)
    return () => {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
    }
  }, [minLeft, minCenter])

  if (collapsed) {
    const panels = [left, center, right]
    return (
      <div className="flex flex-col h-full min-h-0 w-full" data-testid="resizable-panels-collapsed">
        {/* Mobile tab bar */}
        <div className="flex border-b border-border bg-surface" data-testid="mobile-tab-bar">
          {tabLabels.map((label, idx) => (
            <button
              key={idx}
              type="button"
              className={`flex-1 px-3 py-2 text-xs font-medium min-h-[44px] transition-colors ${
                mobileTab === idx
                  ? "text-primary border-b-2 border-primary bg-primary/5"
                  : "text-text-muted hover:text-text"
              }`}
              onClick={() => { setMobileTab(idx as 0 | 1 | 2); onMobileTabChange?.(idx as 0 | 1 | 2) }}
              aria-pressed={mobileTab === idx}
              data-testid={`mobile-tab-${idx}`}
            >
              {label}
            </button>
          ))}
        </div>
        {/* Active panel */}
        <div className="flex-1 min-h-0 overflow-auto">
          {panels[mobileTab]}
        </div>
      </div>
    )
  }

  return (
    <div
      className="flex h-full min-h-0 w-full"
      data-testid="resizable-panels"
      style={{
        display: "grid",
        gridTemplateColumns: `${leftWidth}px 6px ${centerWidth}px 6px 1fr`
      }}
    >
      {/* Left panel: Filter sidebar */}
      <div className="overflow-auto min-h-0 h-full" data-testid="panel-left">
        {left}
      </div>

      {/* Left drag handle */}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize filter panel"
        className="cursor-col-resize bg-border/30 hover:bg-primary/40 active:bg-primary/60 transition-colors w-[6px] flex items-center justify-center"
        onMouseDown={(e) => handleMouseDown("left", e)}
      >
        <div className="w-[2px] h-8 rounded bg-border" />
      </div>

      {/* Center panel: Results list */}
      <div className="overflow-auto min-h-0 h-full flex flex-col" data-testid="panel-center">
        {center}
      </div>

      {/* Center drag handle */}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize results panel"
        className="cursor-col-resize bg-border/30 hover:bg-primary/40 active:bg-primary/60 transition-colors w-[6px] flex items-center justify-center"
        onMouseDown={(e) => handleMouseDown("center", e)}
      >
        <div className="w-[2px] h-8 rounded bg-border" />
      </div>

      {/* Right panel: Reading pane */}
      <div className="overflow-auto min-h-0 h-full" data-testid="panel-right">
        {right}
      </div>
    </div>
  )
}
