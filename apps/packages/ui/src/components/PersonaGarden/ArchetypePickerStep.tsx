import React from "react"

import { useArchetypeCatalog } from "@/hooks/useArchetypeCatalog"

import { ArchetypeCard } from "./ArchetypeCard"

type ArchetypePickerStepProps = {
  selectedKey: string | null
  onSelect: (key: string) => void
}

const SKELETON_COUNT = 6

function SkeletonCard(): React.ReactElement {
  return (
    <div className="flex w-full animate-pulse flex-col items-start gap-2 rounded-lg border border-border bg-surface2 px-4 py-4">
      <div className="h-10 w-10 rounded-lg bg-surface" />
      <div className="w-full space-y-1.5">
        <div className="h-4 w-24 rounded bg-surface" />
        <div className="h-3 w-36 rounded bg-surface" />
      </div>
    </div>
  )
}

export const ArchetypePickerStep: React.FC<ArchetypePickerStepProps> = ({
  selectedKey,
  onSelect
}) => {
  const { archetypes, loading, error } = useArchetypeCatalog()

  const sorted = React.useMemo(() => {
    const nonBlank = archetypes.filter((a) => a.key !== "blank_canvas")
    const blank = archetypes.filter((a) => a.key === "blank_canvas")
    return [...nonBlank, ...blank]
  }, [archetypes])

  return (
    <div className="space-y-3">
      <div>
        <div className="text-sm font-semibold text-text">
          Pick a starting point
        </div>
        <div className="text-xs text-text-muted">
          Choose an archetype that best matches how you want to use your
          assistant. You can customize everything later.
        </div>
      </div>

      {error ? (
        <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
          <div>{error}</div>
          <div className="mt-1 text-xs text-red-100">
            Check your connection and reload the page to try again.
          </div>
        </div>
      ) : loading ? (
        <div
          data-testid="archetype-picker-loading"
          className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3"
        >
          {Array.from({ length: SKELETON_COUNT }, (_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : sorted.length === 0 ? (
        <div className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text-muted">
          No archetypes available. Reload the page to try again.
        </div>
      ) : (
        <div
          data-testid="archetype-picker-grid"
          className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3"
        >
          {sorted.map((archetype) => (
            <ArchetypeCard
              key={archetype.key}
              archetype={archetype}
              selected={selectedKey === archetype.key}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  )
}
