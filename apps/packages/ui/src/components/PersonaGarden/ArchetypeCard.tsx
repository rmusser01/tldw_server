import React from "react"

import type { ArchetypeSummary } from "@/types/archetype"

type ArchetypeCardProps = {
  archetype: ArchetypeSummary
  selected: boolean
  onSelect: (key: string) => void
}

export const ArchetypeCard: React.FC<ArchetypeCardProps> = ({
  archetype,
  selected,
  onSelect
}) => {
  const isBlankCanvas = archetype.key === "blank_canvas"

  const handleClick = React.useCallback(() => {
    onSelect(archetype.key)
  }, [archetype.key, onSelect])

  const handleKeyDown = React.useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault()
        onSelect(archetype.key)
      }
    },
    [archetype.key, onSelect]
  )

  return (
    <button
      type="button"
      data-testid={`archetype-card-${archetype.key}`}
      aria-label={archetype.label}
      aria-pressed={selected}
      data-selected={selected ? "true" : "false"}
      className={[
        "flex w-full flex-col items-start gap-2 rounded-lg px-4 py-4 text-left transition-colors",
        isBlankCanvas
          ? "border-2 border-dashed border-border bg-surface2/50"
          : "border border-border bg-surface2",
        selected && !isBlankCanvas
          ? "border-blue-500 bg-blue-500/10"
          : "",
        selected && isBlankCanvas
          ? "border-blue-500 bg-blue-500/5"
          : "",
        "hover:bg-surface2/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/60"
      ]
        .filter(Boolean)
        .join(" ")}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
    >
      <div
        className={[
          "flex h-10 w-10 items-center justify-center rounded-lg text-xl",
          isBlankCanvas
            ? "bg-surface text-text-muted"
            : "bg-surface text-text"
        ].join(" ")}
      >
        {archetype.icon}
      </div>
      <div>
        <div
          className={[
            "text-sm font-medium",
            isBlankCanvas ? "text-text-muted" : "text-text"
          ].join(" ")}
        >
          {archetype.label}
        </div>
        <div className="mt-0.5 text-xs text-text-muted">
          {archetype.tagline}
        </div>
      </div>
    </button>
  )
}
