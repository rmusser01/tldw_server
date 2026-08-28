import {
  type NotesGraphEdge,
  type NotesGraphEdgeType,
  type NotesGraphNode,
  type NotesGraphResponse,
  fetchNotesGraph
} from "@/services/note-graph-suggestions"
import { useQuery } from "@tanstack/react-query"
import * as React from "react"

const DEFAULT_EDGE_TYPES: NotesGraphEdgeType[] = [
  "manual",
  "wikilink",
  "backlink",
  "tag_membership",
  "source_membership"
]

export const notesGraphWorkspaceQueryKey = ["notes-graph-workspace"] as const

export type NotesGraphLayout = "dagre" | "circle" | "grid" | "concentric"

export type UseNotesGraphWorkspaceOptions = {
  enabled: boolean
  isOnline: boolean
  initialFocusNoteId: string | null
  datasetId?: string
  radius?: 1 | 2
  maxNodes?: number
  maxEdges?: number
}

const mergeById = <T extends { id: string }>(current: T[], next: T[]): T[] => {
  const merged = new Map(current.map((item) => [item.id, item]))
  next.forEach((item) => merged.set(item.id, item))
  return Array.from(merged.values())
}

const mergeGraph = (
  current: NotesGraphResponse,
  next: NotesGraphResponse
): NotesGraphResponse => ({
  ...next,
  nodes: mergeById<NotesGraphNode>(current.nodes, next.nodes),
  edges: mergeById<NotesGraphEdge>(current.edges, next.edges),
  truncated: current.truncated || next.truncated,
  truncated_by: Array.from(
    new Set([...current.truncated_by, ...next.truncated_by])
  )
})

export function useNotesGraphWorkspace(options: UseNotesGraphWorkspaceOptions) {
  const [focusNoteId, setFocusNoteId] = React.useState(
    options.initialFocusNoteId
  )
  const [scope, setScope] = React.useState<"focused" | "all">("focused")
  const [layout, setLayout] = React.useState<NotesGraphLayout>("dagre")
  const [search, setSearch] = React.useState("")
  const [visibleEdgeTypes, setVisibleEdgeTypes] = React.useState(
    () => new Set<NotesGraphEdgeType>(DEFAULT_EDGE_TYPES)
  )
  const [lastAuthoritativeGraph, setLastAuthoritativeGraph] =
    React.useState<NotesGraphResponse | null>(null)
  const expansionIdentity = React.useRef(0)

  React.useEffect(() => {
    setFocusNoteId(options.initialFocusNoteId)
    setScope("focused")
    expansionIdentity.current += 1
  }, [options.initialFocusNoteId])

  React.useEffect(() => {
    setLastAuthoritativeGraph(null)
    setScope("focused")
    expansionIdentity.current += 1
  }, [options.datasetId])

  const centerNoteId =
    scope === "focused" ? focusNoteId ?? undefined : undefined
  const enabled =
    options.enabled &&
    options.isOnline &&
    (scope === "all" || Boolean(centerNoteId))

  const graphQuery = useQuery({
    queryKey: [
      ...notesGraphWorkspaceQueryKey,
      options.datasetId ?? null,
      scope,
      centerNoteId ?? null,
      options.radius ?? 1,
      options.maxNodes ?? 120,
      options.maxEdges ?? 480
    ],
    enabled,
    retry: false,
    queryFn: () =>
      fetchNotesGraph({
        centerNoteId,
        datasetId: options.datasetId,
        radius: options.radius ?? 1,
        maxNodes: options.maxNodes ?? 120,
        maxEdges: options.maxEdges ?? 480,
        cursor: undefined
      })
  })

  React.useEffect(() => {
    if (graphQuery.data) setLastAuthoritativeGraph(graphQuery.data)
  }, [graphQuery.data])

  const allNotes = React.useMemo(() => {
    if (!lastAuthoritativeGraph) {
      return { activeNoteCount: 0, effectiveNoteCap: 0, eligible: false }
    }
    const effectiveNoteCap = Math.min(
      lastAuthoritativeGraph.all_notes_note_cap,
      lastAuthoritativeGraph.limits.max_nodes
    )
    return {
      activeNoteCount: lastAuthoritativeGraph.active_note_count,
      effectiveNoteCap,
      eligible:
        lastAuthoritativeGraph.all_notes_eligible &&
        lastAuthoritativeGraph.active_note_count <= effectiveNoteCap
    }
  }, [lastAuthoritativeGraph])

  const searchResults = React.useMemo(() => {
    const needle = search.trim().toLocaleLowerCase()
    if (!needle || !lastAuthoritativeGraph) return []
    return lastAuthoritativeGraph.nodes.filter((node) =>
      node.label.toLocaleLowerCase().includes(needle)
    )
  }, [lastAuthoritativeGraph, search])

  const expand = React.useCallback(async () => {
    const current = lastAuthoritativeGraph
    if (!options.isOnline || !current?.has_more || !current.cursor)
      return current
    const identity = expansionIdentity.current
    const next = await fetchNotesGraph({
      centerNoteId,
      datasetId: options.datasetId,
      radius: options.radius ?? 1,
      maxNodes: options.maxNodes ?? 120,
      maxEdges: options.maxEdges ?? 480,
      cursor: current.cursor
    })
    if (identity !== expansionIdentity.current) return lastAuthoritativeGraph
    const merged = mergeGraph(current, next)
    setLastAuthoritativeGraph(merged)
    return merged
  }, [
    centerNoteId,
    lastAuthoritativeGraph,
    options.datasetId,
    options.isOnline,
    options.maxEdges,
    options.maxNodes,
    options.radius
  ])

  const focus = React.useCallback((noteId: string) => {
    expansionIdentity.current += 1
    setFocusNoteId(noteId)
    setScope("focused")
  }, [])

  const showAllNotes = React.useCallback(() => {
    if (!allNotes.eligible) return false
    expansionIdentity.current += 1
    setScope("all")
    return true
  }, [allNotes.eligible])

  const toggleEdgeType = React.useCallback((edgeType: NotesGraphEdgeType) => {
    setVisibleEdgeTypes((current) => {
      const next = new Set(current)
      if (next.has(edgeType)) next.delete(edgeType)
      else next.add(edgeType)
      return next
    })
  }, [])

  return {
    graph: lastAuthoritativeGraph,
    graphQuery,
    focusNoteId,
    scope,
    layout,
    setLayout,
    search,
    setSearch,
    searchResults,
    visibleEdgeTypes,
    toggleEdgeType,
    allNotes,
    canExpand: Boolean(
      options.isOnline &&
        lastAuthoritativeGraph?.has_more &&
        lastAuthoritativeGraph.cursor
    ),
    expand,
    focus,
    showAllNotes,
    refresh: graphQuery.refetch,
    isOffline: !options.isOnline,
    isLoading: graphQuery.isLoading && !lastAuthoritativeGraph,
    error: graphQuery.error
  }
}
