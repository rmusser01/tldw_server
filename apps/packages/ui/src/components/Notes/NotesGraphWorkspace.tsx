import { Tooltip } from "antd"
import { PanelLeftOpen } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"

import { NotesEditorEmptyState } from "./NotesEditorPane"
import NotesGraphCanvas, {
  type NotesGraphCanvasHandle
} from "./NotesGraphCanvas"
import NotesGraphToolbar from "./NotesGraphToolbar"
import { useNotesGraphSuggestions } from "./hooks/useNotesGraphSuggestions"
import { useNotesGraphWorkspace } from "./hooks/useNotesGraphWorkspace"
import {
  NOTES_EDITOR_REGION_ID,
  normalizeGraphNoteId
} from "./notes-manager-utils"

type NotesGraphWorkspaceProps = {
  authorityScope: string | null
  isOnline: boolean
  initialFocusNoteId: string | null
  selectedNoteId: string | number | null
  hasActiveNotes: boolean
  onSelectNote: (noteId: string) => void
  onCreateNote: () => void
  isMobileViewport?: boolean
  onOpenSidebar?: () => void
}

const NotesGraphWorkspace: React.FC<NotesGraphWorkspaceProps> = ({
  authorityScope,
  isOnline,
  initialFocusNoteId,
  selectedNoteId,
  hasActiveNotes,
  onSelectNote,
  onCreateNote,
  isMobileViewport = false,
  onOpenSidebar
}) => {
  const { t } = useTranslation(["option", "common"])
  const rootRef = React.useRef<HTMLElement | null>(null)
  const canvasRef = React.useRef<NotesGraphCanvasHandle | null>(null)
  const [mountedFocusNoteId] = React.useState(initialFocusNoteId)
  const [radius, setRadius] = React.useState<1 | 2>(1)
  const [maxNodesInput, setMaxNodesInput] = React.useState(120)
  const [selectedNodeId, setSelectedNodeId] = React.useState<string | null>(
    null
  )
  const [showProvisional, setShowProvisional] = React.useState(true)
  const maxNodeCap = radius === 2 ? 200 : 300
  const maxNodes = Math.min(
    Math.max(20, Number(maxNodesInput) || 120),
    maxNodeCap
  )
  const maxEdges = Math.min(radius === 2 ? 800 : 1200, maxNodes * 4)

  React.useEffect(() => {
    rootRef.current?.focus()
  }, [])
  React.useEffect(() => {
    setMaxNodesInput((current) => Math.min(current, maxNodeCap))
  }, [maxNodeCap])

  const workspace = useNotesGraphWorkspace({
    authorityScope,
    enabled: hasActiveNotes,
    isOnline,
    initialFocusNoteId: mountedFocusNoteId,
    radius,
    maxNodes,
    maxEdges
  })
  const loadedNodeIds = React.useMemo(
    () => new Set(workspace.graph?.nodes.map((node) => node.id) ?? []),
    [workspace.graph]
  )
  const suggestions = useNotesGraphSuggestions({
    authorityScope,
    enabled: Boolean(
      hasActiveNotes && workspace.graph && workspace.focusNoteId
    ),
    isOnline,
    noteId: workspace.focusNoteId,
    loadedNodeIds
  })
  const provisionalOverlays = React.useMemo(
    () => Object.values(suggestions.provisionalBySuggestionId),
    [suggestions.provisionalBySuggestionId]
  )
  const normalizedSelectedId = React.useMemo(() => {
    const normalized = normalizeGraphNoteId(selectedNoteId)
    return normalized ? `note:${normalized}` : null
  }, [selectedNoteId])
  React.useEffect(() => {
    setSelectedNodeId(normalizedSelectedId)
  }, [normalizedSelectedId])

  const handleSelectNode = React.useCallback(
    (nodeId: string) => {
      setSelectedNodeId(nodeId)
      if (nodeId.startsWith("note:")) onSelectNote(normalizeGraphNoteId(nodeId))
    },
    [onSelectNote]
  )
  const handleSelectSearchResult = React.useCallback(
    (nodeId: string) => {
      handleSelectNode(nodeId)
      canvasRef.current?.focusNode(nodeId)
    },
    [handleSelectNode]
  )
  const focusCurrent = React.useCallback(() => {
    const current = normalizeGraphNoteId(selectedNoteId)
    if (current) workspace.focus(current)
  }, [selectedNoteId, workspace])
  const showFocused = React.useCallback(() => {
    const focusId =
      normalizeGraphNoteId(selectedNoteId) ||
      workspace.focusNoteId ||
      mountedFocusNoteId
    if (focusId) workspace.focus(focusId)
  }, [mountedFocusNoteId, selectedNoteId, workspace])

  if (!hasActiveNotes || !mountedFocusNoteId) {
    return (
      <section
        ref={rootRef}
        id={NOTES_EDITOR_REGION_ID}
        tabIndex={-1}
        role="region"
        aria-label="Notes graph"
        data-testid="notes-graph-workspace"
        className={`${isMobileViewport ? "ml-0" : "ml-4"} flex min-h-[520px] flex-1 items-center justify-center bg-bg focus:outline-none focus:ring-2 focus:ring-focus`}>
        <NotesEditorEmptyState disabled={false} onCreateNote={onCreateNote} />
      </section>
    )
  }

  const disabledReason =
    workspace.graph && !workspace.allNotes.eligible ? (
      <p
        className="border-b border-border bg-surface px-3 py-2 text-xs text-text-muted"
        data-testid="notes-graph-all-disabled-reason">
        {t("option:notesSearch.graphAllNotesUnavailable", {
          defaultValue:
            "All notes is available for up to {{cap}} active notes. This library has {{count}}.",
          cap: workspace.allNotes.effectiveNoteCap,
          count: workspace.allNotes.activeNoteCount
        })}
      </p>
    ) : null

  return (
    <section
      ref={rootRef}
      id={NOTES_EDITOR_REGION_ID}
      tabIndex={-1}
      role="region"
      aria-label="Notes graph"
      data-testid="notes-graph-workspace"
      className={`${isMobileViewport ? "ml-0" : "ml-4"} flex min-h-[520px] min-w-0 flex-1 flex-col overflow-hidden bg-bg focus:outline-none focus:ring-2 focus:ring-focus`}>
      {isMobileViewport ? (
        <div className="flex h-11 flex-none items-center border-b border-border bg-surface px-3">
          <Tooltip title="Open notes list">
            <button
              type="button"
              className="inline-flex h-9 w-9 items-center justify-center border border-border bg-surface text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              aria-label="Open notes list"
              onClick={onOpenSidebar}>
              <PanelLeftOpen size={16} aria-hidden="true" />
            </button>
          </Tooltip>
        </div>
      ) : null}
      <NotesGraphToolbar
        search={workspace.search}
        searchResults={workspace.searchResults}
        radius={radius}
        maxNodes={maxNodesInput}
        maxNodeCap={maxNodeCap}
        layout={workspace.layout}
        scope={workspace.scope}
        allNotes={workspace.allNotes}
        visibleEdgeTypes={workspace.visibleEdgeTypes}
        showProvisional={showProvisional}
        canExpand={workspace.canExpand}
        isRefreshing={workspace.graphQuery.isFetching}
        onSearchChange={workspace.setSearch}
        onSelectSearchResult={handleSelectSearchResult}
        onRadiusChange={setRadius}
        onMaxNodesChange={setMaxNodesInput}
        onLayoutChange={workspace.setLayout}
        onShowFocused={showFocused}
        onShowAllNotes={workspace.showAllNotes}
        onToggleEdgeType={workspace.toggleEdgeType}
        onToggleProvisional={() => setShowProvisional((visible) => !visible)}
        onFocusCurrent={focusCurrent}
        onExpand={() => {
          void workspace.expand()
        }}
        onRefresh={() => {
          void workspace.refresh()
        }}
        onZoomIn={() => canvasRef.current?.zoomIn()}
        onZoomOut={() => canvasRef.current?.zoomOut()}
        onFit={() => canvasRef.current?.fit()}
      />
      {disabledReason}
      {workspace.isOffline ? (
        <p
          className="border-b border-border bg-surface px-3 py-2 text-xs text-text-muted"
          data-testid="notes-graph-offline-state"
          role="status">
          {t("option:notesSearch.graphOffline", {
            defaultValue: "Offline: showing the last available graph."
          })}
        </p>
      ) : null}
      {workspace.error && workspace.graph ? (
        <p
          className="border-b border-border bg-surface px-3 py-2 text-xs text-error"
          data-testid="notes-graph-degraded-state"
          role="status">
          {t("option:notesSearch.graphDegraded", {
            defaultValue: "Refresh failed. Showing the last available graph."
          })}
        </p>
      ) : null}
      {workspace.graph?.truncated ? (
        <p
          className="border-b border-border bg-surface px-3 py-2 text-xs text-warn"
          data-testid="notes-graph-truncated-warning"
          role="status">
          {t("option:notesSearch.graphTruncated", {
            defaultValue: "This graph was truncated by server limits."
          })}
        </p>
      ) : null}
      {workspace.isLoading && !workspace.graph ? (
        <div
          className="flex flex-1 items-center justify-center text-sm text-text-muted"
          role="status">
          {t("common:loading", { defaultValue: "Loading..." })}
        </div>
      ) : workspace.error && !workspace.graph ? (
        <div
          className="flex flex-1 items-center justify-center px-6 text-sm text-error"
          role="alert">
          {t("option:notesSearch.graphLoadFailed", {
            defaultValue: "Could not load the notes graph."
          })}
        </div>
      ) : workspace.graph ? (
        <div className="min-h-0 flex-1">
          <NotesGraphCanvas
            ref={canvasRef}
            graph={workspace.graph}
            layout={workspace.layout}
            focusNoteId={workspace.focusNoteId}
            selectedNodeId={selectedNodeId ?? normalizedSelectedId}
            visibleEdgeTypes={workspace.visibleEdgeTypes}
            provisionalOverlays={provisionalOverlays}
            showProvisional={showProvisional}
            onSelectNode={handleSelectNode}
          />
        </div>
      ) : null}
    </section>
  )
}

export default NotesGraphWorkspace
