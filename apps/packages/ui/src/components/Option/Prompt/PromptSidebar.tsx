import React from "react"
import {
  PanelLeftClose,
  PanelLeftOpen,
  NotebookPen,
  Bot,
  FlaskConical,
  Trash2,
} from "lucide-react"
import { SmartCollections } from "./SmartCollections"
import { FacetedFilters } from "./FacetedFilters"
import { FilterPresets } from "./FilterPresets"
import type { PromptSavedView } from "./prompt-workspace-types"
import type { TagMatchMode } from "./custom-prompts-utils"
import type { FilterPreset } from "./useFilterPresets"

type SegmentType = "custom" | "copilot" | "studio" | "trash"

type WorkspaceItem = {
  id: SegmentType
  label: string
  icon: React.ReactNode
  badge?: number | null
}

type PromptSidebarProps = {
  collapsed: boolean
  onToggleCollapsed: () => void
  // Workspace segment
  selectedSegment: SegmentType
  onSegmentChange: (s: SegmentType) => void
  trashCount?: number
  // Smart collections (only for custom segment)
  savedView: PromptSavedView
  onSavedViewChange: (v: PromptSavedView) => void
  smartCounts: Partial<Record<PromptSavedView, number>>
  // Faceted filters
  typeFilter: string
  onTypeFilterChange: (v: string) => void
  typeCounts: Record<string, number>
  syncFilter: string
  onSyncFilterChange: (v: string) => void
  syncCounts: Record<string, number>
  tagFilter: string[]
  onTagFilterChange: (v: string[]) => void
  tagMatchMode: TagMatchMode
  onTagMatchModeChange: (v: TagMatchMode) => void
  tagCounts: Record<string, number>
  // Filter presets
  presets: FilterPreset[]
  onLoadPreset: (preset: FilterPreset) => void
  onSavePreset: (name: string) => void
  onDeletePreset: (id: string) => void
}

export const PromptSidebar: React.FC<PromptSidebarProps> = ({
  collapsed,
  onToggleCollapsed,
  selectedSegment,
  onSegmentChange,
  trashCount,
  savedView,
  onSavedViewChange,
  smartCounts,
  typeFilter,
  onTypeFilterChange,
  typeCounts,
  syncFilter,
  onSyncFilterChange,
  syncCounts,
  tagFilter,
  onTagFilterChange,
  tagMatchMode,
  onTagMatchModeChange,
  tagCounts,
  presets,
  onLoadPreset,
  onSavePreset,
  onDeletePreset,
}) => {
  const workspaces: WorkspaceItem[] = [
    { id: "custom", label: "Custom", icon: <NotebookPen className="size-4" /> },
    { id: "copilot", label: "Copilot", icon: <Bot className="size-4" /> },
    { id: "studio", label: "Studio", icon: <FlaskConical className="size-4" /> },
    {
      id: "trash",
      label: "Trash",
      icon: <Trash2 className="size-4" />,
      badge: trashCount,
    },
  ]

  if (collapsed) {
    return (
      <div
        className="flex w-12 shrink-0 flex-col items-center gap-2 border-r border-border bg-surface py-3"
        data-testid="prompt-sidebar-collapsed"
      >
        <button
          type="button"
          onClick={onToggleCollapsed}
          className="rounded p-1.5 text-text-muted hover:bg-surface2 hover:text-text"
          aria-label="Expand sidebar"
          data-testid="prompt-sidebar-expand"
        >
          <PanelLeftOpen className="size-4" />
        </button>
        <div className="my-1 h-px w-6 bg-border" />
        {workspaces.map((ws) => (
          <button
            key={ws.id}
            type="button"
            onClick={() => onSegmentChange(ws.id)}
            className={`relative rounded p-1.5 transition-colors ${
              selectedSegment === ws.id
                ? "bg-primary/10 text-primary"
                : "text-text-muted hover:bg-surface2 hover:text-text"
            }`}
            title={ws.label}
            data-testid={`sidebar-ws-icon-${ws.id}`}
          >
            {ws.icon}
            {ws.badge != null && ws.badge > 0 && (
              <span className="absolute -right-0.5 -top-0.5 flex size-3.5 items-center justify-center rounded-full bg-danger text-[9px] text-white">
                {ws.badge > 9 ? "9+" : ws.badge}
              </span>
            )}
          </button>
        ))}
      </div>
    )
  }

  return (
    <div
      className="flex w-60 shrink-0 flex-col border-r border-border bg-surface"
      data-testid="prompt-sidebar"
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
          Prompts
        </span>
        <button
          type="button"
          onClick={onToggleCollapsed}
          className="rounded p-1 text-text-muted hover:bg-surface2 hover:text-text"
          aria-label="Collapse sidebar"
          data-testid="prompt-sidebar-collapse"
        >
          <PanelLeftClose className="size-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-5">
        {/* Workspaces */}
        <div>
          <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-text-muted">
            Workspaces
          </h4>
          <div className="space-y-0.5">
            {workspaces.map((ws) => (
              <button
                key={ws.id}
                type="button"
                data-testid={`sidebar-ws-${ws.id}`}
                onClick={() => onSegmentChange(ws.id)}
                className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors ${
                  selectedSegment === ws.id
                    ? "bg-primary/10 text-primary font-medium"
                    : "text-text-muted hover:bg-surface2 hover:text-text"
                }`}
              >
                {ws.icon}
                <span className="flex-1 text-left">{ws.label}</span>
                {ws.badge != null && ws.badge > 0 && (
                  <span className="text-xs text-text-muted tabular-nums">
                    {ws.badge}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Smart collections and filters only for custom segment */}
        {selectedSegment === "custom" && (
          <>
            <div>
              <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-text-muted">
                Collections
              </h4>
              <SmartCollections
                activeView={savedView}
                onViewChange={onSavedViewChange}
                counts={smartCounts}
              />
            </div>

            <FacetedFilters
              typeFilter={typeFilter}
              onTypeFilterChange={onTypeFilterChange}
              typeCounts={typeCounts}
              syncFilter={syncFilter}
              onSyncFilterChange={onSyncFilterChange}
              syncCounts={syncCounts}
              tagFilter={tagFilter}
              onTagFilterChange={onTagFilterChange}
              tagMatchMode={tagMatchMode}
              onTagMatchModeChange={onTagMatchModeChange}
              tagCounts={tagCounts}
            />

            <FilterPresets
              presets={presets}
              onLoad={onLoadPreset}
              onSave={onSavePreset}
              onDelete={onDeletePreset}
            />
          </>
        )}
      </div>
    </div>
  )
}
