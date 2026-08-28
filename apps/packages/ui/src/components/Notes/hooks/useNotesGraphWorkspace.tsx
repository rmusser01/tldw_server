import {
  type NotesGraphEdge,
  type NotesGraphEdgeType,
  type NotesGraphNode,
  type NotesGraphResponse,
  fetchNotesGraph
} from "@/services/note-graph-suggestions"
import { useInfiniteQuery } from "@tanstack/react-query"
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
  authorityScope: string | null
  enabled: boolean
  isOnline: boolean
  initialFocusNoteId: string | null
  datasetId?: string
  radius?: 1 | 2
  maxNodes?: number
  maxEdges?: number
}

type NavigationState = {
  inputIdentity: string
  focusNoteId: string | null
  scope: "focused" | "all"
}

type GraphPage = {
  scopeIdentity: string
  graph: NotesGraphResponse
}

const mergeById = <T extends { id: string }>(
  pages: T[][],
  limit: number
): T[] => {
  const merged = new Map<string, T>()
  pages.forEach((items) => {
    items.forEach((item) => merged.set(item.id, item))
  })
  return Array.from(merged.values()).slice(0, limit)
}

const aggregatePages = (
  data: { pages: GraphPage[] } | undefined,
  scopeIdentity: string
): NotesGraphResponse | null => {
  const pages = data?.pages.filter(
    (page) => page.scopeIdentity === scopeIdentity
  )
  if (!pages?.length || pages.length !== data?.pages.length) return null
  const first = pages[0].graph
  const last = pages[pages.length - 1].graph
  return {
    ...first,
    nodes: mergeById<NotesGraphNode>(
      pages.map((page) => page.graph.nodes),
      first.limits.max_nodes
    ),
    edges: mergeById<NotesGraphEdge>(
      pages.map((page) => page.graph.edges),
      first.limits.max_edges
    ),
    truncated: pages.some((page) => page.graph.truncated),
    truncated_by: Array.from(
      new Set(pages.flatMap((page) => page.graph.truncated_by))
    ),
    has_more: last.has_more,
    cursor: last.cursor
  }
}

const authority = (value: string | null | undefined): string | null =>
  typeof value === "string" && value.length > 0 ? value : null

export function useNotesGraphWorkspace(options: UseNotesGraphWorkspaceOptions) {
  const authorityScope = authority(options.authorityScope)
  const inputIdentity = JSON.stringify([
    authorityScope,
    options.datasetId ?? null,
    options.initialFocusNoteId
  ])
  const initialNavigation = React.useMemo<NavigationState>(
    () => ({
      inputIdentity,
      focusNoteId: options.initialFocusNoteId,
      scope: "focused"
    }),
    [inputIdentity, options.initialFocusNoteId]
  )
  const [navigation, setNavigation] =
    React.useState<NavigationState>(initialNavigation)
  const effectiveNavigation =
    navigation.inputIdentity === inputIdentity ? navigation : initialNavigation
  const [layout, setLayout] = React.useState<NotesGraphLayout>("dagre")
  const [search, setSearch] = React.useState("")
  const [visibleEdgeTypes, setVisibleEdgeTypes] = React.useState(
    () => new Set<NotesGraphEdgeType>(DEFAULT_EDGE_TYPES)
  )
  React.useEffect(() => {
    setNavigation((current) =>
      current.inputIdentity === inputIdentity ? current : initialNavigation
    )
  }, [initialNavigation, inputIdentity])

  const centerNoteId =
    effectiveNavigation.scope === "focused"
      ? effectiveNavigation.focusNoteId ?? undefined
      : undefined
  const radius = options.radius ?? 1
  const maxNodes = options.maxNodes ?? 120
  const maxEdges = options.maxEdges ?? 480
  const scopeIdentity = JSON.stringify([
    authorityScope,
    options.datasetId ?? null,
    effectiveNavigation.scope,
    centerNoteId ?? null,
    radius,
    maxNodes,
    maxEdges
  ])
  const enabled = Boolean(
    authorityScope &&
      options.enabled &&
      options.isOnline &&
      (effectiveNavigation.scope === "all" || centerNoteId)
  )

  const graphQuery = useInfiniteQuery({
    queryKey: [
      ...notesGraphWorkspaceQueryKey,
      authorityScope,
      options.datasetId ?? null,
      effectiveNavigation.scope,
      centerNoteId ?? null,
      radius,
      maxNodes,
      maxEdges
    ],
    initialPageParam: undefined as string | undefined,
    enabled,
    retry: false,
    refetchOnWindowFocus: false,
    queryFn: async ({ pageParam }) => ({
      scopeIdentity,
      graph: await fetchNotesGraph({
        centerNoteId,
        datasetId: options.datasetId,
        radius,
        maxNodes,
        maxEdges,
        cursor: pageParam
      })
    }),
    getNextPageParam: (lastPage) =>
      lastPage.graph.has_more ? lastPage.graph.cursor ?? undefined : undefined
  })

  const graph = React.useMemo(
    () => aggregatePages(graphQuery.data, scopeIdentity),
    [graphQuery.data, scopeIdentity]
  )

  const allNotes = React.useMemo(() => {
    if (!graph) {
      return { activeNoteCount: 0, effectiveNoteCap: 0, eligible: false }
    }
    const effectiveNoteCap = Math.min(
      graph.all_notes_note_cap,
      graph.limits.max_nodes
    )
    return {
      activeNoteCount: graph.active_note_count,
      effectiveNoteCap,
      eligible:
        graph.all_notes_eligible && graph.active_note_count <= effectiveNoteCap
    }
  }, [graph])

  const searchResults = React.useMemo(() => {
    const needle = search.trim().toLocaleLowerCase()
    if (!needle || !graph) return []
    return graph.nodes.filter((node) =>
      node.label.toLocaleLowerCase().includes(needle)
    )
  }, [graph, search])

  const expand = React.useCallback(async () => {
    if (!enabled || !graph?.has_more || !graph.cursor) return graph
    const result = await graphQuery.fetchNextPage()
    return aggregatePages(result.data, scopeIdentity)
  }, [enabled, graph, graphQuery, scopeIdentity])

  const focus = React.useCallback(
    (noteId: string) => {
      setNavigation({ inputIdentity, focusNoteId: noteId, scope: "focused" })
    },
    [inputIdentity]
  )

  const showAllNotes = React.useCallback(() => {
    if (!allNotes.eligible) return false
    setNavigation((current) => ({
      inputIdentity,
      focusNoteId:
        current.inputIdentity === inputIdentity
          ? current.focusNoteId
          : options.initialFocusNoteId,
      scope: "all"
    }))
    return true
  }, [allNotes.eligible, inputIdentity, options.initialFocusNoteId])

  const toggleEdgeType = React.useCallback((edgeType: NotesGraphEdgeType) => {
    setVisibleEdgeTypes((current) => {
      const next = new Set(current)
      if (next.has(edgeType)) next.delete(edgeType)
      else next.add(edgeType)
      return next
    })
  }, [])

  const refresh = React.useCallback(async () => {
    if (!enabled) return graph
    const result = await graphQuery.refetch({ cancelRefetch: true })
    return aggregatePages(result.data, scopeIdentity)
  }, [enabled, graph, graphQuery, scopeIdentity])

  return {
    graph,
    graphQuery,
    focusNoteId: effectiveNavigation.focusNoteId,
    scope: effectiveNavigation.scope,
    layout,
    setLayout,
    search,
    setSearch,
    searchResults,
    visibleEdgeTypes,
    toggleEdgeType,
    allNotes,
    canExpand: Boolean(enabled && graph?.has_more && graph.cursor),
    expand,
    focus,
    showAllNotes,
    refresh,
    isOffline: !options.isOnline,
    isLoading: Boolean(authorityScope && graphQuery.isLoading && !graph),
    error: graphQuery.error
  }
}
