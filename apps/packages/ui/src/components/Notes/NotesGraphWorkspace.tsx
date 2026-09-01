import { Tooltip } from "antd"
import { PanelLeftOpen } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"

import { NotesEditorEmptyState } from "./NotesEditorPane"
import NotesGraphCanvas, {
  type NotesGraphCanvasHandle
} from "./NotesGraphCanvas"
import NotesGraphInspector from "./NotesGraphInspector"
import NotesGraphRelationshipsView from "./NotesGraphRelationshipsView"
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
  const pendingCanvasFocusRef = React.useRef<{
    authorityScope: string | null
    nodeId: string
  } | null>(null)
  const mountedFocusRef = React.useRef({
    authorityScope,
    noteId: authorityScope ? initialFocusNoteId : null
  })
  if (mountedFocusRef.current.authorityScope !== authorityScope) {
    mountedFocusRef.current = {
      authorityScope,
      noteId: authorityScope ? initialFocusNoteId : null
    }
  } else if (
    authorityScope &&
    mountedFocusRef.current.noteId == null &&
    initialFocusNoteId
  ) {
    mountedFocusRef.current.noteId = initialFocusNoteId
  }
  const mountedFocusNoteId = mountedFocusRef.current.noteId
  const [radius, setRadius] = React.useState<1 | 2>(1)
  const [maxNodesInput, setMaxNodesInput] = React.useState(120)
  const normalizedSelectedId = React.useMemo(() => {
    const normalized = normalizeGraphNoteId(selectedNoteId)
    return normalized ? `note:${normalized}` : null
  }, [selectedNoteId])
  const [storedSelection, setStoredSelection] = React.useState(() => ({
    authorityScope,
    controlledNodeId: normalizedSelectedId,
    nodeId: normalizedSelectedId
  }))
  let selection = storedSelection
  if (storedSelection.authorityScope !== authorityScope) {
    selection = {
      authorityScope,
      controlledNodeId: normalizedSelectedId,
      nodeId: null
    }
    setStoredSelection(selection)
  } else if (storedSelection.controlledNodeId !== normalizedSelectedId) {
    selection = {
      authorityScope,
      controlledNodeId: normalizedSelectedId,
      nodeId: normalizedSelectedId
    }
    setStoredSelection(selection)
  }
  const [showProvisional, setShowProvisional] = React.useState(true)
  const [viewMode, setViewMode] = React.useState<"canvas" | "relationships">(
    "canvas"
  )
  const [announcement, setAnnouncement] = React.useState({
    id: 0,
    message: ""
  })
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
    maxEdges,
    semanticManagementEnabled: true
  })
  const loadedNodeIds = React.useMemo(
    () => new Set(workspace.graph?.nodes.map((node) => node.id) ?? []),
    [workspace.graph]
  )
  const selectedNodeCandidate =
    selection.nodeId ??
    (workspace.focusNoteId ? `note:${workspace.focusNoteId}` : null)
  const currentSelectedNodeId =
    selectedNodeCandidate && loadedNodeIds.has(selectedNodeCandidate)
      ? selectedNodeCandidate
      : null
  const selectedSuggestionNoteId = React.useMemo(() => {
    const selectedNode = workspace.graph?.nodes.find(
      (node) => node.id === currentSelectedNodeId
    )
    if (selectedNode?.type !== "note" || !selectedNode.id.startsWith("note:"))
      return null
    return normalizeGraphNoteId(selectedNode.id) || null
  }, [currentSelectedNodeId, workspace.graph])
  const suggestionsAuthorized = Boolean(
    workspace.graph?.suggestions_authorized === true && selectedSuggestionNoteId
  )
  const suggestions = useNotesGraphSuggestions({
    authorityScope,
    enabled: hasActiveNotes && suggestionsAuthorized,
    isOnline,
    noteId: selectedSuggestionNoteId,
    loadedNodeIds,
    fallbackTargetLabel: t("option:notesSearch.graphSuggestedNote")
  })
  const provisionalOverlays = React.useMemo(
    () => Object.values(suggestions.provisionalBySuggestionId),
    [suggestions.provisionalBySuggestionId]
  )
  const isDecisionPending = Boolean(
    suggestions.mutations?.acceptance?.isPending ||
      suggestions.mutations?.rejection?.isPending
  )
  const announce = React.useCallback((message: string) => {
    setAnnouncement((current) => ({ id: current.id + 1, message }))
  }, [])

  const handleSelectNode = React.useCallback(
    (nodeId: string) => {
      setStoredSelection({
        authorityScope,
        controlledNodeId: normalizedSelectedId,
        nodeId
      })
      if (nodeId.startsWith("note:")) onSelectNote(normalizeGraphNoteId(nodeId))
    },
    [authorityScope, normalizedSelectedId, onSelectNote]
  )
  const handleFocusNode = React.useCallback(
    (nodeId: string) => {
      handleSelectNode(nodeId)
      if (canvasRef.current) {
        canvasRef.current.focusNode(nodeId)
      } else {
        pendingCanvasFocusRef.current = { authorityScope, nodeId }
      }
    },
    [authorityScope, handleSelectNode]
  )
  const handleSelectSearchResult = React.useCallback(
    (nodeId: string) => {
      handleFocusNode(nodeId)
    },
    [handleFocusNode]
  )
  React.useEffect(() => {
    const pendingFocus = pendingCanvasFocusRef.current
    if (!pendingFocus) return
    if (pendingFocus.authorityScope !== authorityScope) {
      pendingCanvasFocusRef.current = null
      return
    }
    if (viewMode !== "canvas") return
    canvasRef.current?.focusNode(pendingFocus.nodeId)
    pendingCanvasFocusRef.current = null
  }, [authorityScope, viewMode])
  const handleSuggestionDecision = React.useCallback(
    async (action: "accept" | "reject", suggestionId: string) => {
      const item = suggestions.suggestions?.find(
        (entry) => entry.id === suggestionId
      )
      if (!item) {
        announce(t("option:notesSearch.graphSuggestionDecisionFailed"))
        return false
      }
      try {
        if (action === "accept") await suggestions.accept(item)
        else await suggestions.reject(item)
        announce(
          t(
            action === "accept"
              ? "option:notesSearch.graphSuggestionAccepted"
              : "option:notesSearch.graphSuggestionRejected"
          )
        )
        return true
      } catch {
        announce(t("option:notesSearch.graphSuggestionDecisionFailed"))
        return false
      }
    },
    [announce, suggestions, t]
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
      className={`${isMobileViewport ? "ml-0" : "ml-4"} flex min-h-0 min-w-0 flex-1 flex-col overflow-x-hidden overflow-y-auto bg-bg focus:outline-none focus:ring-2 focus:ring-focus`}>
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
        viewMode={viewMode}
        suggestionsAuthorized={suggestionsAuthorized}
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
        onViewModeChange={setViewMode}
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
          {t("common:loading.title", { defaultValue: "Loading..." })}
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
        <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
          <div
            className="min-h-[420px] min-w-0 flex-1 sm:min-h-[520px]"
            data-testid="notes-graph-primary-view">
            {viewMode === "canvas" ? (
              <div className="h-full" data-testid="notes-graph-canvas-slot">
                <NotesGraphCanvas
                  ref={canvasRef}
                  graph={workspace.graph}
                  layout={workspace.layout}
                  focusNoteId={workspace.focusNoteId}
                  selectedNodeId={currentSelectedNodeId}
                  visibleEdgeTypes={workspace.visibleEdgeTypes}
                  provisionalOverlays={
                    suggestionsAuthorized ? provisionalOverlays : []
                  }
                  showProvisional={suggestionsAuthorized && showProvisional}
                  onSelectNode={handleSelectNode}
                />
              </div>
            ) : (
              <NotesGraphRelationshipsView
                graph={workspace.graph}
                selectedNodeId={currentSelectedNodeId}
                visibleEdgeTypes={workspace.visibleEdgeTypes}
                provisionalOverlays={
                  suggestionsAuthorized ? provisionalOverlays : []
                }
                suggestions={suggestions.suggestions ?? []}
                suggestionsAuthorized={suggestionsAuthorized}
                isOnline={isOnline}
                canAccept={Boolean(
                  suggestions.capabilities?.allowed_actions.includes(
                    "accept"
                  ) && !isDecisionPending
                )}
                canReject={Boolean(
                  suggestions.capabilities?.allowed_actions.includes(
                    "reject"
                  ) && !isDecisionPending
                )}
                onSelectNode={handleFocusNode}
                onDecideSuggestion={handleSuggestionDecision}
              />
            )}
          </div>
          <div
            className="max-h-[min(420px,45vh)] min-h-[280px] flex-none overflow-y-auto scroll-pb-28 border-t border-border pb-28 lg:h-auto lg:max-h-none lg:min-h-0 lg:w-[360px] lg:border-l lg:border-t-0"
            data-testid="notes-graph-inspector-region">
            <NotesGraphInspector
              graph={workspace.graph}
              selectedNodeId={currentSelectedNodeId}
              suggestionsAuthorized={suggestionsAuthorized}
              isOnline={isOnline}
              controller={suggestions}
              semanticController={workspace.semanticIndex}
              semanticEnabled={workspace.semantic?.enabled ?? false}
              onSemanticEnabledChange={workspace.semantic?.setEnabled}
              onSelectNode={handleFocusNode}
              onAnnounce={announce}
              onDecideSuggestion={handleSuggestionDecision}
            />
          </div>
        </div>
      ) : null}
      <p className="sr-only" aria-live="polite" aria-atomic="true">
        {announcement.message ? (
          <span key={announcement.id}>{announcement.message}</span>
        ) : null}
      </p>
    </section>
  )
}

export default NotesGraphWorkspace
