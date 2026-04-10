import React from "react"
import { useTranslation } from "react-i18next"
import type { RagSource } from "@/services/rag/unified-rag"

type SourceChipsProps = {
  selectedSources: RagSource[]
  onSourcesChange: (sources: RagSource[]) => void
  disabled?: boolean
}

const ALL_SOURCES: RagSource[] = [
  "media_db",
  "notes",
  "characters",
  "chats",
  "kanban",
]

/**
 * Source filter chips - multi-select with "All" behavior
 *
 * - [All] is default and deselects when any specific source is selected
 * - Selecting multiple specific sources (e.g., [Notes] + [Media]) searches both
 * - Clicking a selected source deselects it; if none remain, reverts to All
 */
export const SourceChips: React.FC<SourceChipsProps> = ({
  selectedSources,
  onSourcesChange,
  disabled = false
}) => {
  const { t } = useTranslation(["sidepanel"])

  const selectedSourceSet = new Set(selectedSources)
  const normalizedSources = ALL_SOURCES.filter((source) =>
    selectedSourceSet.has(source)
  )

  // Check if "All" is effectively selected (all sources or empty array)
  const isAllSelected =
    normalizedSources.length === 0 ||
    normalizedSources.length === ALL_SOURCES.length

  const sourceLabels: Record<RagSource, string> = {
    media_db: t("sidepanel:rag.sources.media", "Media"),
    notes: t("sidepanel:rag.sources.notes", "Notes"),
    characters: t("sidepanel:rag.sources.characters", "Story Characters"),
    chats: t("sidepanel:rag.sources.chats", "Conversations"),
    kanban: t("sidepanel:rag.sources.kanban", "Task Boards"),
  }

  const handleAllClick = () => {
    // Clicking "All" selects all sources (or clears to default)
    onSourcesChange([])
  }

  const handleSourceClick = (source: RagSource) => {
    if (isAllSelected) {
      // If "All" is selected, clicking a specific source selects only that source
      onSourcesChange([source])
    } else if (selectedSourceSet.has(source)) {
      // Deselect this source
      const nextSources = normalizedSources.filter((s) => s !== source)
      // If no sources remain, revert to "All"
      onSourcesChange(nextSources.length === 0 ? [] : nextSources)
    } else {
      // Add this source to selection
      const nextSourceSet = new Set(normalizedSources)
      nextSourceSet.add(source)
      const nextSources = ALL_SOURCES.filter((s) => nextSourceSet.has(s))
      // If all sources are now selected, treat as "All"
      onSourcesChange(
        nextSources.length === ALL_SOURCES.length ? [] : nextSources
      )
    }
  }

  const chipClass = (isSelected: boolean) =>
    `px-3 py-1 text-xs font-medium rounded-full transition-colors ${
      disabled ? "cursor-not-allowed" : "cursor-pointer"
    }
     ${
       isSelected
         ? "bg-accent text-white"
         : "bg-surface2 text-text-muted hover:bg-surface3 hover:text-text"
     }
     ${disabled ? "opacity-50" : ""}
    `

  return (
    <div
      className="flex flex-wrap gap-2"
      role="group"
      aria-label={t("sidepanel:rag.sourceFilter", "Filter by source")}
    >
      <button
        type="button"
        onClick={handleAllClick}
        disabled={disabled}
        className={chipClass(isAllSelected)}
        aria-pressed={isAllSelected}
      >
        {t("sidepanel:rag.sources.all", "All")}
      </button>

      {ALL_SOURCES.map((source) => {
        const isSelected = !isAllSelected && selectedSourceSet.has(source)
        return (
          <button
            key={source}
            type="button"
            onClick={() => handleSourceClick(source)}
            disabled={disabled}
            className={chipClass(isSelected)}
            aria-pressed={isSelected}
          >
            {sourceLabels[source]}
          </button>
        )
      })}
    </div>
  )
}
