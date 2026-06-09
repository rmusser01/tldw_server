import type { ExplainerMode } from "./types"

type ExplainerModeTabsProps = {
  activeMode: ExplainerMode
  onModeChange: (mode: ExplainerMode) => void
}

export const ExplainerModeTabs = ({
  activeMode,
  onModeChange
}: ExplainerModeTabsProps) => (
  <div
    role="tablist"
    aria-label="Explainer mode"
    className="inline-flex rounded-md border border-border bg-surface2 p-1"
  >
    {(["goal", "sources"] as const).map((mode) => (
      <button
        key={mode}
        type="button"
        role="tab"
        aria-selected={activeMode === mode}
        className={[
          "rounded px-4 py-2 text-sm font-medium transition-colors",
          activeMode === mode
            ? "bg-surface text-text shadow-sm"
            : "text-text-muted hover:text-text"
        ].join(" ")}
        onClick={() => onModeChange(mode)}
      >
        {mode === "goal" ? "Goal" : "Sources"}
      </button>
    ))}
  </div>
)
