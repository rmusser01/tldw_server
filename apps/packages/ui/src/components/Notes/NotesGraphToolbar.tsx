import IconButton from "@/components/Common/IconButton"
import type { NotesGraphEdgeType } from "@/services/note-graph-suggestions"
import {
  Focus,
  ListTree,
  Maximize2,
  Network,
  Plus,
  RefreshCw,
  SlidersHorizontal,
  ZoomIn,
  ZoomOut
} from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"

import type { NotesGraphLayout } from "./hooks/useNotesGraphWorkspace"

type GraphSearchResult = { id: string; label: string }

type NotesGraphToolbarProps = {
  viewMode: "canvas" | "relationships"
  suggestionsAuthorized: boolean
  search: string
  searchResults: GraphSearchResult[]
  radius: 1 | 2
  maxNodes: number
  maxNodeCap: number
  layout: NotesGraphLayout
  scope: "focused" | "all"
  allNotes: {
    activeNoteCount: number
    effectiveNoteCap: number
    eligible: boolean
  }
  visibleEdgeTypes: ReadonlySet<NotesGraphEdgeType>
  showProvisional: boolean
  canExpand: boolean
  isRefreshing: boolean
  onSearchChange: (value: string) => void
  onViewModeChange: (mode: "canvas" | "relationships") => void
  onSelectSearchResult: (nodeId: string) => void
  onRadiusChange: (radius: 1 | 2) => void
  onMaxNodesChange: (maxNodes: number) => void
  onLayoutChange: (layout: NotesGraphLayout) => void
  onShowFocused: () => void
  onShowAllNotes: () => void
  onToggleEdgeType: (edgeType: NotesGraphEdgeType) => void
  onToggleProvisional: () => void
  onFocusCurrent: () => void
  onExpand: () => void
  onRefresh: () => void
  onZoomIn: () => void
  onZoomOut: () => void
  onFit: () => void
}

const EDGE_OPTIONS: Array<{ type: NotesGraphEdgeType; label: string }> = [
  { type: "manual", label: "Manual links" },
  { type: "wikilink", label: "Note links" },
  { type: "backlink", label: "Backlinks" },
  { type: "tag_membership", label: "Tag membership" },
  { type: "source_membership", label: "Source membership" }
]

const EDGE_MENU_ID = "notes-graph-edge-menu"
const iconButtonClassName =
  "flex-none border border-border bg-surface text-text hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"

const NotesGraphToolbar: React.FC<NotesGraphToolbarProps> = ({
  viewMode,
  suggestionsAuthorized,
  search,
  searchResults,
  radius,
  maxNodes,
  maxNodeCap,
  layout,
  scope,
  allNotes,
  visibleEdgeTypes,
  showProvisional,
  canExpand,
  isRefreshing,
  onSearchChange,
  onViewModeChange,
  onSelectSearchResult,
  onRadiusChange,
  onMaxNodesChange,
  onLayoutChange,
  onShowFocused,
  onShowAllNotes,
  onToggleEdgeType,
  onToggleProvisional,
  onFocusCurrent,
  onExpand,
  onRefresh,
  onZoomIn,
  onZoomOut,
  onFit
}) => {
  const { t } = useTranslation(["option", "common"])
  const [edgeMenuOpen, setEdgeMenuOpen] = React.useState(false)
  const iconSize = 16

  return (
    <div
      className="border-b border-border bg-surface px-3 py-2"
      data-testid="notes-graph-toolbar">
      <div className="flex flex-wrap items-end gap-2">
        <div
          className="inline-flex h-11 flex-none border border-border"
          role="group"
          aria-label={t("option:notesSearch.graphViewMode")}>
          <button
            type="button"
            className={`inline-flex min-w-[112px] items-center justify-center gap-2 px-3 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus ${viewMode === "canvas" ? "bg-primary text-primary-foreground" : "bg-surface text-text"}`}
            aria-pressed={viewMode === "canvas"}
            onClick={() => onViewModeChange("canvas")}>
            <Network size={16} aria-hidden="true" />
            {t("option:notesSearch.graphCanvas")}
          </button>
          <button
            type="button"
            className={`inline-flex min-w-[132px] items-center justify-center gap-2 border-l border-border px-3 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus ${viewMode === "relationships" ? "bg-primary text-primary-foreground" : "bg-surface text-text"}`}
            aria-pressed={viewMode === "relationships"}
            onClick={() => onViewModeChange("relationships")}>
            <ListTree size={16} aria-hidden="true" />
            {t("option:notesSearch.graphRelationships")}
          </button>
        </div>
        <div className="relative w-[180px] flex-none sm:w-[200px] 2xl:w-[280px]">
          <label className="sr-only" htmlFor="notes-graph-search">
            {t("option:notesSearch.graphSearchLoaded", {
              defaultValue: "Search loaded nodes"
            })}
          </label>
          <input
            id="notes-graph-search"
            type="search"
            role="searchbox"
            className="h-9 w-full border border-border bg-bg px-3 text-sm text-text outline-none placeholder:text-text-muted focus:ring-2 focus:ring-focus"
            aria-label="Search loaded nodes"
            placeholder={t("option:notesSearch.graphSearchLoaded", {
              defaultValue: "Search loaded nodes"
            })}
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
          />
          {search.trim() && searchResults.length > 0 ? (
            <div className="absolute left-0 top-10 z-20 max-h-48 w-full overflow-auto border border-border bg-elevated shadow-lg">
              {searchResults.map((node) => (
                <button
                  type="button"
                  key={node.id}
                  className="block w-full px-3 py-2 text-left text-sm text-text hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus"
                  onClick={() => onSelectSearchResult(node.id)}>
                  {node.label}
                </button>
              ))}
            </div>
          ) : null}
        </div>

        <label className="flex flex-col gap-1 text-xs text-text-muted">
          {t("option:notesSearch.graphRadiusLabel", { defaultValue: "Radius" })}
          <select
            className="h-9 min-w-[76px] border border-border bg-bg px-2 text-sm text-text focus:ring-2 focus:ring-focus"
            aria-label="Graph radius"
            value={radius}
            onChange={(event) =>
              onRadiusChange(Number(event.target.value) as 1 | 2)
            }>
            <option value={1}>1</option>
            <option value={2}>2</option>
          </select>
        </label>

        <label className="flex flex-col gap-1 text-xs text-text-muted">
          {t("option:notesSearch.graphMaxNodesLabel", {
            defaultValue: "Max nodes"
          })}
          <input
            type="number"
            className="h-9 w-[92px] border border-border bg-bg px-2 text-sm text-text focus:ring-2 focus:ring-focus"
            aria-label="Maximum graph nodes"
            min={20}
            max={maxNodeCap}
            value={maxNodes}
            onChange={(event) => onMaxNodesChange(Number(event.target.value))}
          />
        </label>

        <label className="flex flex-col gap-1 text-xs text-text-muted">
          {t("option:notesSearch.graphLayoutLabel", { defaultValue: "Layout" })}
          <select
            className="h-9 min-w-[112px] border border-border bg-bg px-2 text-sm text-text focus:ring-2 focus:ring-focus"
            aria-label="Graph layout"
            value={layout}
            onChange={(event) =>
              onLayoutChange(event.target.value as NotesGraphLayout)
            }>
            <option value="dagre">Dagre</option>
            <option value="circle">Circle</option>
            <option value="grid">Grid</option>
            <option value="concentric">Concentric</option>
          </select>
        </label>

        <div
          className="inline-flex h-9 border border-border"
          role="group"
          aria-label="Graph scope">
          <button
            type="button"
            className={`px-3 text-sm ${scope === "focused" ? "bg-primary text-primary-foreground" : "bg-surface text-text"}`}
            aria-pressed={scope === "focused"}
            onClick={onShowFocused}>
            Focused
          </button>
          <button
            type="button"
            className={`border-l border-border px-3 text-sm ${scope === "all" ? "bg-primary text-primary-foreground" : "bg-surface text-text"}`}
            aria-pressed={scope === "all"}
            disabled={!allNotes.eligible}
            onClick={onShowAllNotes}>
            All notes
          </button>
        </div>

        <IconButton
          ariaLabel="Focus current note"
          className={iconButtonClassName}
          onClick={onFocusCurrent}>
          <Focus size={iconSize} aria-hidden="true" />
        </IconButton>
        <IconButton
          ariaLabel="Expand graph"
          className={iconButtonClassName}
          disabled={!canExpand}
          onClick={onExpand}>
          <Plus size={iconSize} aria-hidden="true" />
        </IconButton>
        <IconButton
          ariaLabel="Refresh graph"
          className={iconButtonClassName}
          disabled={isRefreshing}
          onClick={onRefresh}>
          <RefreshCw size={iconSize} aria-hidden="true" />
        </IconButton>
        <IconButton
          ariaLabel="Zoom in"
          className={iconButtonClassName}
          disabled={viewMode !== "canvas"}
          onClick={onZoomIn}>
          <ZoomIn size={iconSize} aria-hidden="true" />
        </IconButton>
        <IconButton
          ariaLabel="Zoom out"
          className={iconButtonClassName}
          disabled={viewMode !== "canvas"}
          onClick={onZoomOut}>
          <ZoomOut size={iconSize} aria-hidden="true" />
        </IconButton>
        <IconButton
          ariaLabel="Fit graph to view"
          className={iconButtonClassName}
          disabled={viewMode !== "canvas"}
          onClick={onFit}>
          <Maximize2 size={iconSize} aria-hidden="true" />
        </IconButton>

        <div className="relative">
          <IconButton
            ariaLabel="Edge visibility"
            className={iconButtonClassName}
            ariaExpanded={edgeMenuOpen}
            ariaControls={EDGE_MENU_ID}
            onClick={() => setEdgeMenuOpen((open) => !open)}>
            <SlidersHorizontal size={iconSize} aria-hidden="true" />
          </IconButton>
          {edgeMenuOpen ? (
            <div
              id={EDGE_MENU_ID}
              role="group"
              aria-label="Edge visibility filters"
              className="absolute right-0 top-12 z-20 min-w-[190px] border border-border bg-elevated p-3 shadow-lg">
              {EDGE_OPTIONS.map(({ type, label }) => (
                <label
                  key={type}
                  className="flex min-h-8 items-center gap-2 text-sm text-text">
                  <input
                    type="checkbox"
                    checked={visibleEdgeTypes.has(type)}
                    onChange={() => onToggleEdgeType(type)}
                  />
                  {label}
                </label>
              ))}
              {suggestionsAuthorized ? (
                <label className="flex min-h-8 items-center gap-2 border-t border-border pt-2 text-sm text-text">
                  <input
                    type="checkbox"
                    checked={showProvisional}
                    onChange={onToggleProvisional}
                  />
                  {t("option:notesSearch.graphSuggestions")}
                </label>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}

export default NotesGraphToolbar
