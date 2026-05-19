import React from "react"

type MCPModuleToggleGridProps = {
  enabledModules: string[]
  onToggle: (moduleId: string, enabled: boolean) => void
}

const BUILT_IN_MODULES: Array<{ id: string; label: string }> = [
  { id: "media", label: "Media" },
  { id: "notes", label: "Notes" },
  { id: "filesystem", label: "Filesystem" },
  { id: "knowledge", label: "Knowledge" },
  { id: "kanban", label: "Kanban" },
  { id: "quizzes", label: "Quizzes" },
  { id: "flashcards", label: "Flashcards" },
  { id: "slides", label: "Slides" }
]

export const MCPModuleToggleGrid: React.FC<MCPModuleToggleGridProps> = ({
  enabledModules,
  onToggle
}) => {
  return (
    <div
      data-testid="mcp-module-toggle-grid"
      className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4"
    >
      {BUILT_IN_MODULES.map((mod) => {
        const enabled = enabledModules.includes(mod.id)
        return (
          <button
            key={mod.id}
            type="button"
            role="switch"
            aria-checked={enabled}
            aria-label={`Toggle ${mod.label}`}
            className={
              "flex items-center justify-between rounded-lg border px-3 py-2 text-left text-sm font-medium transition-colors " +
              (enabled
                ? "border-blue-500/60 bg-blue-500/10 text-text"
                : "border-border bg-surface2 text-text-muted")
            }
            onClick={() => onToggle(mod.id, !enabled)}
          >
            <span>{mod.label}</span>
            <span
              aria-hidden="true"
              className={
                "ml-2 inline-block h-3 w-3 rounded-full transition-colors " +
                (enabled ? "bg-blue-400" : "bg-surface")
              }
            />
          </button>
        )
      })}
    </div>
  )
}
