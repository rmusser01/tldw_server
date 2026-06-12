import { useRef } from "react"
import type { ExplainerMode } from "./types"

const MODES = ["goal", "sources"] as const

type ExplainerModeTabsProps = {
  activeMode: ExplainerMode
  onModeChange: (mode: ExplainerMode) => void
}

export const ExplainerModeTabs = ({
  activeMode,
  onModeChange
}: ExplainerModeTabsProps) => {
  const tabRefs = useRef<Record<ExplainerMode, HTMLButtonElement | null>>({
    goal: null,
    sources: null
  })

  const selectMode = (mode: ExplainerMode) => {
    onModeChange(mode)
    tabRefs.current[mode]?.focus()
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>, mode: ExplainerMode) => {
    const index = MODES.indexOf(mode)
    switch (event.key) {
      case "ArrowRight":
      case "ArrowDown":
        event.preventDefault()
        selectMode(MODES[(index + 1) % MODES.length])
        break
      case "ArrowLeft":
      case "ArrowUp":
        event.preventDefault()
        selectMode(MODES[(index - 1 + MODES.length) % MODES.length])
        break
      case "Home":
        event.preventDefault()
        selectMode(MODES[0])
        break
      case "End":
        event.preventDefault()
        selectMode(MODES[MODES.length - 1])
        break
      default:
        break
    }
  }

  return (
    <div
      role="tablist"
      aria-label="Explainer mode"
      className="inline-flex rounded-md border border-border bg-surface2 p-1"
    >
      {MODES.map((mode) => (
        <button
          key={mode}
          ref={(element) => {
            tabRefs.current[mode] = element
          }}
          type="button"
          role="tab"
          aria-selected={activeMode === mode}
          tabIndex={activeMode === mode ? 0 : -1}
          className={[
            "rounded px-4 py-2 text-sm font-medium transition-colors",
            activeMode === mode
              ? "bg-surface text-text shadow-sm"
              : "text-text-muted hover:text-text"
          ].join(" ")}
          onClick={() => onModeChange(mode)}
          onKeyDown={(event) => handleKeyDown(event, mode)}
        >
          {mode === "goal" ? "Goal" : "Sources"}
        </button>
      ))}
    </div>
  )
}
