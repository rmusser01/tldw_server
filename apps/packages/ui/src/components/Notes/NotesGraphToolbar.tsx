import IconButton from "@/components/Common/IconButton"
import type { NotesGraphEdgeType } from "@/services/note-graph-suggestions"
import {
  Focus,
  ListTree,
  Maximize2,
  Network,
  Plus,
  RefreshCw,
  RotateCcw,
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
  semanticAvailable: boolean
  semanticEnabled: boolean
  semanticFocusRequired: boolean
  semanticTopK: number
  semanticMaxTopK: number
  semanticThreshold: number
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
  onSemanticEnabledChange: (enabled: boolean) => void
  onSemanticTopKChange: (value: number) => void
  onSemanticThresholdChange: (value: number) => void
  onSemanticReset: () => void
  onFocusCurrent: () => void
  onExpand: () => void
  onRefresh: () => void
  onZoomIn: () => void
  onZoomOut: () => void
  onFit: () => void
}

const EDGE_OPTIONS: Array<{
  type: NotesGraphEdgeType
  labelKey: string
  defaultLabel: string
}> = [
  {
    type: "manual",
    labelKey: "option:notesSearch.graphEdgeType.manual",
    defaultLabel: "Manual links"
  },
  {
    type: "wikilink",
    labelKey: "option:notesSearch.graphEdgeType.wikilink",
    defaultLabel: "Note links"
  },
  {
    type: "backlink",
    labelKey: "option:notesSearch.graphEdgeType.backlink",
    defaultLabel: "Backlinks"
  },
  {
    type: "tag_membership",
    labelKey: "option:notesSearch.graphEdgeType.tag_membership",
    defaultLabel: "Tag membership"
  },
  {
    type: "source_membership",
    labelKey: "option:notesSearch.graphEdgeType.source_membership",
    defaultLabel: "Source membership"
  }
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
  semanticAvailable,
  semanticEnabled,
  semanticFocusRequired,
  semanticTopK,
  semanticMaxTopK,
  semanticThreshold,
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
  onSemanticEnabledChange,
  onSemanticTopKChange,
  onSemanticThresholdChange,
  onSemanticReset,
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
          aria-label={t("option:notesSearch.graphViewMode", {
            defaultValue: "Graph view"
          })}>
          <button
            type="button"
            className={`inline-flex min-w-[112px] items-center justify-center gap-2 px-3 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus ${viewMode === "canvas" ? "bg-primary text-primary-foreground" : "bg-surface text-text"}`}
            aria-pressed={viewMode === "canvas"}
            onClick={() => onViewModeChange("canvas")}>
            <Network size={16} aria-hidden="true" />
            {t("option:notesSearch.graphCanvas", { defaultValue: "Canvas" })}
          </button>
          <button
            type="button"
            className={`inline-flex min-w-[132px] items-center justify-center gap-2 border-l border-border px-3 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus ${viewMode === "relationships" ? "bg-primary text-primary-foreground" : "bg-surface text-text"}`}
            aria-pressed={viewMode === "relationships"}
            onClick={() => onViewModeChange("relationships")}>
            <ListTree size={16} aria-hidden="true" />
            {t("option:notesSearch.graphRelationships", {
              defaultValue: "Relationships"
            })}
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
            aria-label={t("option:notesSearch.graphSearchLoaded", {
              defaultValue: "Search loaded nodes"
            })}
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
            aria-label={t("option:notesSearch.graphRadiusAria", {
              defaultValue: "Graph radius"
            })}
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
            aria-label={t("option:notesSearch.graphMaxNodesAria", {
              defaultValue: "Maximum graph nodes"
            })}
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
            aria-label={t("option:notesSearch.graphLayoutAria", {
              defaultValue: "Graph layout"
            })}
            value={layout}
            onChange={(event) =>
              onLayoutChange(event.target.value as NotesGraphLayout)
            }>
            <option value="dagre">
              {t("option:notesSearch.graphLayoutOption.dagre", {
                defaultValue: "Dagre"
              })}
            </option>
            <option value="circle">
              {t("option:notesSearch.graphLayoutOption.circle", {
                defaultValue: "Circle"
              })}
            </option>
            <option value="grid">
              {t("option:notesSearch.graphLayoutOption.grid", {
                defaultValue: "Grid"
              })}
            </option>
            <option value="concentric">
              {t("option:notesSearch.graphLayoutOption.concentric", {
                defaultValue: "Concentric"
              })}
            </option>
          </select>
        </label>

        <div
          className="inline-flex h-9 border border-border"
          role="group"
          aria-label={t("option:notesSearch.graphScope", {
            defaultValue: "Graph scope"
          })}>
          <button
            type="button"
            className={`px-3 text-sm ${scope === "focused" ? "bg-primary text-primary-foreground" : "bg-surface text-text"}`}
            aria-pressed={scope === "focused"}
            onClick={onShowFocused}>
            {t("option:notesSearch.graphScopeFocused", {
              defaultValue: "Focused"
            })}
          </button>
          <button
            type="button"
            className={`border-l border-border px-3 text-sm ${scope === "all" ? "bg-primary text-primary-foreground" : "bg-surface text-text"}`}
            aria-pressed={scope === "all"}
            disabled={!allNotes.eligible}
            onClick={onShowAllNotes}>
            {t("option:notesSearch.graphScopeAll", {
              defaultValue: "All notes"
            })}
          </button>
        </div>

        {semanticAvailable || semanticEnabled ? (
          <fieldset className="flex min-h-11 min-w-0 flex-wrap items-end gap-2 border border-border bg-surface px-2 py-1">
            <legend className="sr-only">
              {t("option:notesSearch.graphSimilarContent", {
                defaultValue: "Similar content"
              })}
            </legend>
            <label className="flex h-9 items-center gap-2 whitespace-nowrap text-sm text-text">
              <input
                type="checkbox"
                checked={semanticEnabled}
                onChange={() => onSemanticEnabledChange(!semanticEnabled)}
              />
              {t("option:notesSearch.graphSimilarContent", {
                defaultValue: "Similar content"
              })}
            </label>
            {semanticEnabled ? (
              <>
                <label className="flex flex-col gap-0.5 text-xs text-text-muted">
                  {t("option:notesSearch.graphSemanticNeighbors", {
                    defaultValue: "Neighbors"
                  })}
                  <input
                    type="number"
                    className="h-8 w-[72px] border border-border bg-bg px-2 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                    aria-label={t("option:notesSearch.graphSemanticNeighbors", {
                      defaultValue: "Neighbors"
                    })}
                    min={1}
                    max={semanticMaxTopK}
                    value={semanticTopK}
                    onChange={(event) =>
                      onSemanticTopKChange(Number(event.target.value))
                    }
                  />
                </label>
                <label className="flex min-w-[168px] flex-1 flex-col gap-0.5 text-xs text-text-muted">
                  <span className="flex items-center justify-between gap-2">
                    {t("option:notesSearch.graphSemanticThreshold", {
                      defaultValue: "Minimum passage similarity"
                    })}
                    <output className="font-mono text-text">
                      {semanticThreshold.toFixed(2)}
                    </output>
                  </span>
                  <input
                    type="range"
                    className="h-8 w-full min-w-[144px] accent-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                    aria-label={t("option:notesSearch.graphSemanticThreshold", {
                      defaultValue: "Minimum passage similarity"
                    })}
                    min={0}
                    max={1}
                    step={0.01}
                    value={semanticThreshold}
                    onChange={(event) =>
                      onSemanticThresholdChange(Number(event.target.value))
                    }
                  />
                </label>
                <IconButton
                  ariaLabel={t("option:notesSearch.graphSemanticReset", {
                    defaultValue: "Reset Similar content controls"
                  })}
                  className={iconButtonClassName}
                  onClick={onSemanticReset}>
                  <RotateCcw size={iconSize} aria-hidden="true" />
                </IconButton>
              </>
            ) : null}
          </fieldset>
        ) : null}

        {semanticFocusRequired ? (
          <p
            className="min-h-9 min-w-0 flex-1 break-words px-2 py-2 text-xs text-warn"
            role="status">
            {t("option:notesSearch.graphSemanticFocusRequired", {
              defaultValue: "Focus a Note to load similar content."
            })}
          </p>
        ) : null}

        <IconButton
          ariaLabel={t("option:notesSearch.graphFocusCurrent", {
            defaultValue: "Focus current note"
          })}
          className={iconButtonClassName}
          onClick={onFocusCurrent}>
          <Focus size={iconSize} aria-hidden="true" />
        </IconButton>
        <IconButton
          ariaLabel={t("option:notesSearch.graphExpand", {
            defaultValue: "Expand graph"
          })}
          className={iconButtonClassName}
          disabled={!canExpand}
          onClick={onExpand}>
          <Plus size={iconSize} aria-hidden="true" />
        </IconButton>
        <IconButton
          ariaLabel={t("option:notesSearch.graphRefresh", {
            defaultValue: "Refresh graph"
          })}
          className={iconButtonClassName}
          disabled={isRefreshing}
          onClick={onRefresh}>
          <RefreshCw size={iconSize} aria-hidden="true" />
        </IconButton>
        <IconButton
          ariaLabel={t("option:notesSearch.graphZoomIn", {
            defaultValue: "Zoom in"
          })}
          className={iconButtonClassName}
          disabled={viewMode !== "canvas"}
          onClick={onZoomIn}>
          <ZoomIn size={iconSize} aria-hidden="true" />
        </IconButton>
        <IconButton
          ariaLabel={t("option:notesSearch.graphZoomOut", {
            defaultValue: "Zoom out"
          })}
          className={iconButtonClassName}
          disabled={viewMode !== "canvas"}
          onClick={onZoomOut}>
          <ZoomOut size={iconSize} aria-hidden="true" />
        </IconButton>
        <IconButton
          ariaLabel={t("option:notesSearch.graphFit", {
            defaultValue: "Fit graph to view"
          })}
          className={iconButtonClassName}
          disabled={viewMode !== "canvas"}
          onClick={onFit}>
          <Maximize2 size={iconSize} aria-hidden="true" />
        </IconButton>

        <div className="relative">
          <IconButton
            ariaLabel={t("option:notesSearch.graphEdgeVisibility", {
              defaultValue: "Edge visibility"
            })}
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
              aria-label={t("option:notesSearch.graphEdgeVisibilityFilters", {
                defaultValue: "Edge visibility filters"
              })}
              className="absolute right-0 top-12 z-20 min-w-[190px] border border-border bg-elevated p-3 shadow-lg">
              {EDGE_OPTIONS.map(({ type, labelKey, defaultLabel }) => (
                <label
                  key={type}
                  className="flex min-h-8 items-center gap-2 text-sm text-text">
                  <input
                    type="checkbox"
                    checked={visibleEdgeTypes.has(type)}
                    onChange={() => onToggleEdgeType(type)}
                  />
                  {t(labelKey, { defaultValue: defaultLabel })}
                </label>
              ))}
              {suggestionsAuthorized ? (
                <label className="flex min-h-8 items-center gap-2 border-t border-border pt-2 text-sm text-text">
                  <input
                    type="checkbox"
                    checked={showProvisional}
                    onChange={onToggleProvisional}
                  />
                  {t("option:notesSearch.graphSuggestions", {
                    defaultValue: "Suggestions"
                  })}
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
