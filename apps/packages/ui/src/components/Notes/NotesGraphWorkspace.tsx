import type { NotesGraphEdge } from "@/services/note-graph-suggestions"
import { Tooltip } from "antd"
import { PanelLeftOpen } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"

import { NotesEditorEmptyState } from "./NotesEditorPane"
import NotesGraphCanvas, {
  type NotesGraphCanvasHandle
} from "./NotesGraphCanvas"
import NotesGraphFilters from "./NotesGraphFilters"
import NotesGraphInspector from "./NotesGraphInspector"
import NotesGraphNotePreview, {
  type NotesGraphNotePreviewData
} from "./NotesGraphNotePreview"
import NotesGraphRelationshipsView from "./NotesGraphRelationshipsView"
import NotesGraphToolbar from "./NotesGraphToolbar"
import { useNotesGraphSuggestions } from "./hooks/useNotesGraphSuggestions"
import { useNotesGraphWorkspace } from "./hooks/useNotesGraphWorkspace"
import {
  NOTES_GRAPH_TREE_LAYOUT_LIMIT,
  countVisibleNotesGraphEdges,
  keepLinkedNotes,
  overlaysWithDrawnEnds,
  resolveNotesGraphCanvasLayout
} from "./notes-graph-view"
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
  /** Resolves false when the pick was cancelled (for example, unsaved changes). */
  onSelectNote: (noteId: string) => void | Promise<boolean | void>
  onCreateNote: () => void
  isMobileViewport?: boolean
  onOpenSidebar?: () => void
  /** The open note's title and text, previewed beside the graph (NL-11). */
  notePreview?: NotesGraphNotePreviewData | null
  /** Leaves Graph for the editor, with the note still open. */
  onOpenInEditor?: (noteId: string) => void
  /** Tags offered by the graph's tag filter. */
  tagOptions?: string[]
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
  onOpenSidebar,
  notePreview = null,
  onOpenInEditor,
  tagOptions
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
  const [linkedOnly, setLinkedOnly] = React.useState(false)
  // Tree is the default; only a reader's own choice of it is explained when a
  // large graph falls back to Rings (NL-11 review).
  const [layoutChosen, setLayoutChosen] = React.useState(false)
  // A list pick in All notes: panned to once the graph has loaded, or
  // reported when the note is not in this view (NL-11 review).
  const pendingPickRef = React.useRef<{
    authorityScope: string | null
    noteId: string
  } | null>(null)
  const [pickOutsideView, setPickOutsideView] = React.useState<string | null>(
    null
  )
  // The note last picked in the graph itself: a selection change it causes is
  // not a list pick, so it must not re-centre the graph (NL-11).
  const graphPickRef = React.useRef<string | null>(null)
  const lastSelectionRef = React.useRef({
    authorityScope,
    noteId: normalizeGraphNoteId(selectedNoteId) || null
  })
  const [viewMode, setViewMode] = React.useState<"canvas" | "relationships">(
    "canvas"
  )
  const [selectedEdgeId, setSelectedEdgeId] = React.useState<string | null>(
    null
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
    // preventScroll: focusing the region must not scroll the page (NO-01).
    rootRef.current?.focus({ preventScroll: true })
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
  React.useEffect(() => {
    if (
      selectedEdgeId &&
      !workspace.graph?.edges.some((edge) => edge.id === selectedEdgeId)
    ) {
      setSelectedEdgeId(null)
    }
  }, [selectedEdgeId, workspace.graph])
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
  const displayGraph = React.useMemo(() => {
    if (!workspace.graph || !linkedOnly) return workspace.graph
    const focusNodeId =
      workspace.scope === "focused" && workspace.focusNoteId
        ? `note:${workspace.focusNoteId}`
        : null
    return keepLinkedNotes(workspace.graph, workspace.visibleEdgeTypes, [
      focusNodeId,
      currentSelectedNodeId
    ])
  }, [
    currentSelectedNodeId,
    linkedOnly,
    workspace.focusNoteId,
    workspace.graph,
    workspace.scope,
    workspace.visibleEdgeTypes
  ])
  const focusOnCanvas = React.useCallback(
    (nodeId: string) => {
      if (canvasRef.current && viewMode === "canvas") {
        canvasRef.current.focusNode(nodeId)
      } else {
        pendingCanvasFocusRef.current = { authorityScope, nodeId }
      }
    },
    [authorityScope, viewMode]
  )
  // In All notes a picked note is panned to once the graph has it; one the
  // view does not hold (a tag filter, a truncated graph) is reported, with a
  // way to show it, instead of silently dropping the filter (NL-11 review).
  const workspaceScope = workspace.scope
  const deliverPendingPick = React.useCallback(() => {
    const pick = pendingPickRef.current
    if (!pick) return
    if (pick.authorityScope !== authorityScope || workspaceScope !== "all") {
      pendingPickRef.current = null
      return
    }
    if (!displayGraph) return
    pendingPickRef.current = null
    const nodeId = `note:${pick.noteId}`
    if (displayGraph.nodes.some((node) => node.id === nodeId)) {
      focusOnCanvas(nodeId)
    } else {
      setPickOutsideView(pick.noteId)
    }
  }, [authorityScope, displayGraph, focusOnCanvas, workspaceScope])
  React.useEffect(() => {
    deliverPendingPick()
  }, [deliverPendingPick])
  React.useEffect(() => {
    setPickOutsideView(null)
  }, [workspaceScope, workspace.tagFilter])
  // A note picked in the notes list while in Graph re-centres the graph on it
  // (NL-11): All notes pans to it, a focused graph refocuses on it.
  const focusWorkspaceNote = workspace.focus
  React.useEffect(() => {
    const noteId = normalizeGraphNoteId(selectedNoteId) || null
    const previous = lastSelectionRef.current
    lastSelectionRef.current = { authorityScope, noteId }
    if (
      previous.authorityScope !== authorityScope ||
      !noteId ||
      previous.noteId === noteId
    )
      return
    const pickedInGraph = graphPickRef.current === noteId
    graphPickRef.current = null
    if (pickedInGraph) return
    setPickOutsideView(null)
    if (workspaceScope === "all") {
      pendingPickRef.current = { authorityScope, noteId }
      deliverPendingPick()
    } else {
      focusWorkspaceNote(noteId)
    }
  }, [
    authorityScope,
    deliverPendingPick,
    focusWorkspaceNote,
    selectedNoteId,
    workspaceScope
  ])
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
  // Only overlays whose ends the canvas draws: "Linked notes only" can hide
  // a suggestion's note (NL-11 review).
  const provisionalOverlays = React.useMemo(
    () =>
      displayGraph
        ? overlaysWithDrawnEnds(
            Object.values(suggestions.provisionalBySuggestionId),
            displayGraph
          )
        : [],
    [displayGraph, suggestions.provisionalBySuggestionId]
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
      setSelectedEdgeId(null)
      setStoredSelection({
        authorityScope,
        controlledNodeId: normalizedSelectedId,
        nodeId
      })
      if (nodeId.startsWith("note:")) {
        const noteId = normalizeGraphNoteId(nodeId)
        graphPickRef.current = noteId
        // A cancelled pick never changes the selection, so forget it: a later
        // list pick of the same note must still re-centre (NL-11 review).
        void Promise.resolve(onSelectNote(noteId)).then((opened) => {
          if (opened === false && graphPickRef.current === noteId) {
            graphPickRef.current = null
          }
        })
      }
    },
    [authorityScope, normalizedSelectedId, onSelectNote]
  )
  const displayedNodeIds = React.useMemo(
    () => new Set(displayGraph?.nodes.map((node) => node.id) ?? []),
    [displayGraph]
  )
  const handleFocusNode = React.useCallback(
    (nodeId: string) => {
      handleSelectNode(nodeId)
      // A note "Linked notes only" hides is shown once selected; centre on it
      // after the canvas draws it (NL-11 review).
      if (canvasRef.current && displayedNodeIds.has(nodeId)) {
        canvasRef.current.focusNode(nodeId)
      } else {
        pendingCanvasFocusRef.current = { authorityScope, nodeId }
      }
    },
    [authorityScope, displayedNodeIds, handleSelectNode]
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
    if (viewMode !== "canvas" || !canvasRef.current) return
    if (displayGraph && !displayedNodeIds.has(pendingFocus.nodeId)) return
    canvasRef.current.focusNode(pendingFocus.nodeId)
    pendingCanvasFocusRef.current = null
  }, [authorityScope, displayGraph, displayedNodeIds, viewMode])
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
  const handleCreateManualLink = React.useCallback(
    async (edge: NotesGraphEdge) => {
      try {
        const succeeded = await workspace.createManualLink(edge)
        if (!succeeded) throw new Error("manual link conversion unavailable")
        announce(t("option:notesSearch.graphManualLinkCreated"))
        return true
      } catch {
        announce(t("option:notesSearch.graphManualLinkFailed"))
        return false
      }
    },
    [announce, t, workspace]
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

  const tagFilter = workspace.tagFilter ?? null
  const disabledReason =
    workspace.graph && !workspace.allNotes.eligible && !tagFilter ? (
      <p
        className="border-b border-border bg-surface px-3 py-2 text-xs text-text-muted"
        data-testid="notes-graph-all-disabled-reason">
        {t("option:notesSearch.graphAllNotesUnavailable", {
          defaultValue:
            "All notes is available for up to {{cap}} active notes. This library has {{count}}.",
          cap: workspace.allNotes.effectiveNoteCap,
          count: workspace.allNotes.activeNoteCount
        })}{" "}
        {t("option:notesSearch.graphAllNotesFilterHint", {
          defaultValue: "Pick a tag to see part of it."
        })}
      </p>
    ) : null
  const canvasLayout = displayGraph
    ? resolveNotesGraphCanvasLayout(
        workspace.layout,
        displayGraph.nodes.length,
        countVisibleNotesGraphEdges(displayGraph, workspace.visibleEdgeTypes)
      )
    : workspace.layout
  const layoutFallbackNotice =
    viewMode === "canvas" && layoutChosen && canvasLayout !== workspace.layout ? (
      <p
        className="border-b border-border bg-surface px-3 py-2 text-xs text-text-muted"
        data-testid="notes-graph-layout-fallback"
        role="status">
        {t("option:notesSearch.graphLayoutFallback", {
          defaultValue:
            "Tree is slow for a graph this large, so it uses the Rings layout. Tree works up to {{nodes}} items and {{links}} links.",
          nodes: NOTES_GRAPH_TREE_LAYOUT_LIMIT.nodes,
          links: NOTES_GRAPH_TREE_LAYOUT_LIMIT.edges
        })}
      </p>
    ) : null
  const pickOutsideViewNotice = pickOutsideView ? (
    <p
      className="flex flex-wrap items-center gap-2 border-b border-border bg-surface px-3 py-2 text-xs text-text-muted"
      data-testid="notes-graph-pick-outside-view"
      role="status">
      {t("option:notesSearch.graphPickOutsideView", {
        defaultValue: "The note you picked isn't in this view."
      })}
      <button
        type="button"
        className="min-h-8 border border-border bg-surface px-2 text-xs text-text hover:bg-surface2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
        onClick={() => {
          setPickOutsideView(null)
          workspace.focus(pickOutsideView)
        }}>
        {t("option:notesSearch.graphPickShowIt", {
          defaultValue: "Show it in the graph"
        })}
      </button>
    </p>
  ) : null
  const selectedNoteKey = normalizeGraphNoteId(selectedNoteId) || null
  const preview =
    notePreview &&
    selectedNoteKey &&
    normalizeGraphNoteId(notePreview.noteId) === selectedNoteKey
      ? notePreview
      : null

  // With no note open, the focus is a fallback (the most recent or first
  // visible note); say so rather than centring on it silently (NL-14).
  const focusFallbackLabel =
    workspace.scope === "focused" &&
    !normalizeGraphNoteId(selectedNoteId) &&
    workspace.focusNoteId
      ? workspace.graph?.nodes.find(
          (node) => node.id === `note:${workspace.focusNoteId}`
        )?.label ?? null
      : null
  const focusFallbackHint = focusFallbackLabel ? (
    <p
      className="border-b border-border bg-surface px-3 py-2 text-xs text-text-muted"
      data-testid="notes-graph-focus-fallback-hint">
      {t("option:notesSearch.graphFocusFallbackHint", {
        defaultValue:
          "No note is open, so the graph is centered on “{{title}}”. Pick a note to focus the graph.",
        title: focusFallbackLabel
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
      {workspace.isPermissionDenied ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-sm">
          <p role="alert">
            {t("option:notesSearch.graphPermissionUnavailable", {
              defaultValue:
                "Notes graph is unavailable for this account. Ask an administrator for access."
            })}
          </p>
          <button
            type="button"
            className="border border-border bg-surface px-3 py-2 text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:opacity-50"
            disabled={!isOnline || workspace.graphQuery.isFetching}
            onClick={() => {
              void workspace.refresh()
            }}>
            {t("option:notesSearch.graphRefresh", {
              defaultValue: "Refresh graph"
            })}
          </button>
        </div>
      ) : (
        <NotesGraphToolbar
          viewMode={viewMode}
          suggestionsAuthorized={suggestionsAuthorized}
          search={workspace.search}
          searchResults={workspace.searchResults}
          radius={radius}
          maxNodes={maxNodesInput}
          maxNodeCap={maxNodeCap}
          layout={layoutChosen ? workspace.layout : canvasLayout}
          scope={tagFilter ? "tag" : workspace.scope}
          allNotes={workspace.allNotes}
          visibleEdgeTypes={workspace.visibleEdgeTypes}
          showProvisional={showProvisional}
          semanticAvailable={Boolean(workspace.semanticIndex?.status?.active_generation_usable)}
          semanticEnabled={workspace.semantic.enabled}
          semanticFocusRequired={workspace.semantic.focusRequired}
          semanticTopK={workspace.semantic.topK}
          semanticMaxTopK={workspace.semantic.maxTopK}
          semanticThreshold={workspace.semantic.threshold}
          canExpand={workspace.canExpand}
          isRefreshing={workspace.graphQuery.isFetching}
          filterControls={
            <NotesGraphFilters
              tagOptions={tagOptions ?? []}
              tag={tagFilter}
              linkedOnly={linkedOnly}
              onTagChange={(tag) => workspace.filterByTag?.(tag)}
              onLinkedOnlyChange={setLinkedOnly}
            />
          }
          onSearchChange={workspace.setSearch}
          onViewModeChange={setViewMode}
          onSelectSearchResult={handleSelectSearchResult}
          onRadiusChange={setRadius}
          onMaxNodesChange={setMaxNodesInput}
          onLayoutChange={(nextLayout) => {
            setLayoutChosen(true)
            workspace.setLayout(nextLayout)
          }}
          onShowFocused={showFocused}
          onShowAllNotes={workspace.showAllNotes}
          onToggleEdgeType={workspace.toggleEdgeType}
          onToggleProvisional={() => setShowProvisional((visible) => !visible)}
          onSemanticEnabledChange={workspace.semantic.setEnabled}
          onSemanticTopKChange={workspace.semantic.setTopK}
          onSemanticThresholdChange={workspace.semantic.setThreshold}
          onSemanticReset={workspace.semantic.reset}
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
      )}
      {disabledReason}
      {focusFallbackHint}
      {pickOutsideViewNotice}
      {layoutFallbackNotice}
      {workspace.isOffline && !workspace.isPermissionDenied ? (
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
      {workspace.isPermissionDenied ? null : workspace.isLoading &&
        !workspace.graph ? (
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
      ) : workspace.graph && displayGraph ? (
        <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
          <div
            className="min-h-[420px] min-w-0 flex-1 sm:min-h-[520px]"
            data-testid="notes-graph-primary-view">
            {viewMode === "canvas" ? (
              <div className="h-full" data-testid="notes-graph-canvas-slot">
                <NotesGraphCanvas
                  ref={canvasRef}
                  graph={displayGraph}
                  layout={canvasLayout}
                  focusNoteId={workspace.focusNoteId}
                  selectedNodeId={currentSelectedNodeId}
                  visibleEdgeTypes={workspace.visibleEdgeTypes}
                  provisionalOverlays={
                    suggestionsAuthorized ? provisionalOverlays : []
                  }
                  showProvisional={suggestionsAuthorized && showProvisional}
                  onSelectNode={handleSelectNode}
                  onSelectEdge={setSelectedEdgeId}
                />
              </div>
            ) : (
              <NotesGraphRelationshipsView
                graph={displayGraph}
                selectedNodeId={currentSelectedNodeId}
                visibleEdgeTypes={workspace.visibleEdgeTypes}
                provisionalOverlays={
                  suggestionsAuthorized ? provisionalOverlays : []
                }
                suggestions={suggestions.suggestions ?? []}
                suggestionsAuthorized={suggestionsAuthorized}
                manualLinkAuthorized={
                  workspace.graph.manual_link_authorized === true
                }
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
                onSelectEdge={setSelectedEdgeId}
                onCreateManualLink={handleCreateManualLink}
                onDecideSuggestion={handleSuggestionDecision}
                manualLinkPendingEdgeIds={workspace.manualLinkPendingEdgeIds}
                queryIdentity={workspace.queryIdentity}
              />
            )}
          </div>
          <div
            className="max-h-[min(420px,45vh)] min-h-[280px] flex-none overflow-y-auto scroll-pb-28 border-t border-border pb-28 lg:h-auto lg:max-h-none lg:min-h-0 lg:w-[360px] lg:border-l lg:border-t-0"
            data-testid="notes-graph-inspector-region">
            {preview ? (
              <NotesGraphNotePreview
                preview={preview}
                onOpenInEditor={onOpenInEditor}
              />
            ) : null}
            <NotesGraphInspector
              graph={workspace.graph}
              selectedNodeId={currentSelectedNodeId}
              selectedEdgeId={selectedEdgeId}
              suggestionsAuthorized={suggestionsAuthorized}
              manualLinkAuthorized={
                workspace.graph.manual_link_authorized === true
              }
              isOnline={isOnline}
              controller={suggestions}
              semanticController={workspace.semanticIndex}
              semanticEnabled={workspace.semantic?.enabled ?? false}
              onSemanticEnabledChange={workspace.semantic?.setEnabled}
              onSelectNode={handleFocusNode}
              onSelectEdge={setSelectedEdgeId}
              onCreateManualLink={handleCreateManualLink}
              manualLinkPendingEdgeIds={workspace.manualLinkPendingEdgeIds}
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
