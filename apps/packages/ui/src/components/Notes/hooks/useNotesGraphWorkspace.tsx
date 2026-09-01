import {
  NOTES_GRAPH_SEMANTIC_MAX_TOP_K,
  type NotesGraphEdge,
  type NotesGraphEdgeType,
  type NotesGraphNode,
  type NotesGraphResponse,
  fetchNotesGraph
} from "@/services/note-graph-suggestions"
import {
  type InfiniteData,
  useInfiniteQuery,
  useQueryClient
} from "@tanstack/react-query"
import * as React from "react"

import { useNotesSemanticIndex } from "./useNotesSemanticIndex"

const ORDINARY_EDGE_TYPES: NotesGraphEdgeType[] = [
  "manual",
  "wikilink",
  "backlink",
  "tag_membership",
  "source_membership"
]
const DEFAULT_SEMANTIC_THRESHOLD = 0.75
const DEFAULT_SEMANTIC_TOP_K = 10

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
  semanticManagementEnabled?: boolean
}

type NavigationState = {
  inputIdentity: string
  focusNoteId: string | null
  scope: "focused" | "all"
}

type GraphInfiniteData = InfiniteData<NotesGraphResponse, string | undefined>

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
  data: GraphInfiniteData | undefined
): NotesGraphResponse | null => {
  const pages = data?.pages
  if (!pages?.length) return null
  const first = pages[0]
  const last = pages[pages.length - 1]
  const nodes = mergeById<NotesGraphNode>(
    pages.map((page) => page.nodes),
    first.limits.max_nodes
  )
  const nodeIds = new Set(nodes.map((node) => node.id))
  return {
    ...first,
    nodes,
    edges: mergeById<NotesGraphEdge>(
      pages.map((page) =>
        page.edges.filter(
          (edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target)
        )
      ),
      first.limits.max_edges
    ),
    truncated: pages.some((page) => page.truncated),
    truncated_by: Array.from(
      new Set(pages.flatMap((page) => page.truncated_by))
    ),
    has_more: last.has_more,
    cursor: last.cursor
  }
}

const boundGraphData = (data: GraphInfiniteData): GraphInfiniteData => {
  const first = data.pages[0]
  if (!first) return data
  const nodeIds = new Set<string>()
  const edgeIds = new Set<string>()
  const responseCursors = new Set<string>()
  const pages: NotesGraphResponse[] = []
  const pageParams: Array<string | undefined> = []

  for (let index = 0; index < data.pages.length; index += 1) {
    const page = data.pages[index]
    const requestCursor = data.pageParams[index]
    const nodes = page.nodes.filter((node) => {
      if (nodeIds.has(node.id) || nodeIds.size >= first.limits.max_nodes)
        return false
      nodeIds.add(node.id)
      return true
    })
    const edges = page.edges.filter((edge) => {
      if (edgeIds.has(edge.id) || edgeIds.size >= first.limits.max_edges)
        return false
      if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) return false
      edgeIds.add(edge.id)
      return true
    })
    const madeProgress = nodes.length > 0 || edges.length > 0
    const repeatedCursor = Boolean(
      page.cursor &&
        (page.cursor === requestCursor || responseCursors.has(page.cursor))
    )
    const reachedLimit =
      nodeIds.size >= first.limits.max_nodes ||
      edgeIds.size >= first.limits.max_edges
    const mustStop =
      page.has_more && (!madeProgress || repeatedCursor || reachedLimit)

    pages.push(
      mustStop
        ? { ...page, nodes, edges, has_more: false, cursor: null }
        : {
            ...page,
            nodes,
            edges
          }
    )
    pageParams.push(requestCursor)
    if (page.cursor) responseCursors.add(page.cursor)
    if (mustStop) break
  }

  return { pages, pageParams }
}

const authority = (value: string | null | undefined): string | null =>
  typeof value === "string" && value.length > 0 ? value : null

export function useNotesGraphWorkspace(options: UseNotesGraphWorkspaceOptions) {
  const queryClient = useQueryClient()
  const authorityScope = authority(options.authorityScope)
  const semanticIndex = useNotesSemanticIndex({
    authorityScope,
    enabled: Boolean(options.enabled && options.semanticManagementEnabled),
    isOnline: options.isOnline,
    datasetId: options.datasetId
  })
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
    () => new Set<NotesGraphEdgeType>(ORDINARY_EDGE_TYPES)
  )
  const [semanticThreshold, setSemanticThreshold] = React.useState(
    DEFAULT_SEMANTIC_THRESHOLD
  )
  const [semanticTopK, setSemanticTopK] = React.useState(DEFAULT_SEMANTIC_TOP_K)
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
  const semanticEnabled = visibleEdgeTypes.has("semantic")
  const requestedEdgeTypes = React.useMemo<NotesGraphEdgeType[]>(
    () =>
      semanticEnabled
        ? [...ORDINARY_EDGE_TYPES, "semantic"]
        : [...ORDINARY_EDGE_TYPES],
    [semanticEnabled]
  )
  const enabled = Boolean(
    authorityScope &&
      options.enabled &&
      options.isOnline &&
      (effectiveNavigation.scope === "all" || centerNoteId)
  )

  const queryKey = React.useMemo(
    () =>
      [
        ...notesGraphWorkspaceQueryKey,
        authorityScope,
        options.datasetId ?? null,
        effectiveNavigation.scope,
        effectiveNavigation.focusNoteId,
        centerNoteId ?? null,
        radius,
        maxNodes,
        maxEdges,
        semanticEnabled,
        requestedEdgeTypes,
        semanticThreshold,
        semanticTopK
      ] as const,
    [
      authorityScope,
      centerNoteId,
      effectiveNavigation.focusNoteId,
      effectiveNavigation.scope,
      maxEdges,
      maxNodes,
      options.datasetId,
      radius,
      requestedEdgeTypes,
      semanticEnabled,
      semanticThreshold,
      semanticTopK
    ]
  )
  const queryIdentity = JSON.stringify(queryKey)
  const fetchPage = React.useCallback(
    (pageParam: string | undefined) => {
      return fetchNotesGraph({
        centerNoteId,
        datasetId: options.datasetId,
        radius,
        edgeTypes: requestedEdgeTypes,
        maxNodes,
        maxEdges,
        cursor: pageParam,
        semanticThreshold: semanticEnabled ? semanticThreshold : undefined,
        semanticTopK: semanticEnabled ? semanticTopK : undefined
      })
    },
    [
      centerNoteId,
      maxEdges,
      maxNodes,
      options.datasetId,
      radius,
      requestedEdgeTypes,
      semanticEnabled,
      semanticThreshold,
      semanticTopK
    ]
  )
  const graphQuery = useInfiniteQuery<
    NotesGraphResponse,
    Error,
    GraphInfiniteData,
    typeof queryKey,
    string | undefined
  >({
    queryKey,
    initialPageParam: undefined as string | undefined,
    enabled,
    retry: false,
    refetchOnWindowFocus: false,
    structuralSharing: (_previous, next: GraphInfiniteData) =>
      boundGraphData(next),
    queryFn: async ({ pageParam }) => {
      const page = await fetchPage(pageParam)
      return pageParam === undefined
        ? boundGraphData({ pages: [page], pageParams: [pageParam] }).pages[0]
        : page
    },
    getNextPageParam: (_lastPage, allPages, _lastPageParam, allPageParams) => {
      const bounded = boundGraphData({
        pages: allPages,
        pageParams: allPageParams
      })
      if (bounded.pages.length !== allPages.length) return undefined
      const last = bounded.pages.at(-1)
      return last?.has_more ? last.cursor ?? undefined : undefined
    }
  })

  const currentGraph = React.useMemo(
    () => aggregatePages(graphQuery.data),
    [graphQuery.data]
  )
  const fallbackIdentity = JSON.stringify([
    authorityScope,
    options.datasetId ?? null,
    effectiveNavigation.scope,
    effectiveNavigation.focusNoteId,
    centerNoteId ?? null,
    radius,
    maxNodes,
    maxEdges
  ])
  const lastGoodGraph = React.useRef<{
    identity: string
    graph: NotesGraphResponse
  } | null>(null)
  if (currentGraph) {
    lastGoodGraph.current = { identity: fallbackIdentity, graph: currentGraph }
  }
  const graph =
    currentGraph ??
    (lastGoodGraph.current?.identity === fallbackIdentity
      ? lastGoodGraph.current.graph
      : null)

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

  const expansionInFlight = React.useRef(
    new Map<string, Promise<NotesGraphResponse | null>>()
  )
  const expand = React.useCallback((): Promise<NotesGraphResponse | null> => {
    const pending = expansionInFlight.current.get(queryIdentity)
    if (pending) return pending
    if (
      !enabled ||
      graphQuery.isFetchingNextPage ||
      !graph?.has_more ||
      !graph.cursor
    )
      return Promise.resolve(graph)

    const requestCursor = graph.cursor
    const expansionPageKey = [
      ...queryKey,
      "cursor-page",
      requestCursor
    ] as const
    const expansion = queryClient
      .fetchQuery({
        queryKey: expansionPageKey,
        queryFn: () => fetchPage(requestCursor),
        retry: false,
        staleTime: Infinity,
        gcTime: 0
      })
      .then((page) => {
        queryClient.setQueryData<GraphInfiniteData>(queryKey, (current) => {
          const currentGraph = aggregatePages(current)
          if (
            !current ||
            !currentGraph?.has_more ||
            currentGraph.cursor !== requestCursor
          ) {
            return current
          }
          return boundGraphData({
            pages: [...current.pages, page],
            pageParams: [...current.pageParams, requestCursor]
          })
        })
        return aggregatePages(
          queryClient.getQueryData<GraphInfiniteData>(queryKey)
        )
      })
    expansionInFlight.current.set(queryIdentity, expansion)
    const clear = () => {
      if (expansionInFlight.current.get(queryIdentity) === expansion) {
        expansionInFlight.current.delete(queryIdentity)
      }
      queryClient.removeQueries({ queryKey: expansionPageKey, exact: true })
    }
    void expansion.then(clear, clear)
    return expansion
  }, [
    enabled,
    fetchPage,
    graph,
    graphQuery,
    queryClient,
    queryIdentity,
    queryKey
  ])

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
    await graphQuery.refetch({ cancelRefetch: true })
    return aggregatePages(queryClient.getQueryData<GraphInfiniteData>(queryKey))
  }, [enabled, graph, graphQuery, queryClient, queryKey])

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
    semantic: {
      enabled: semanticEnabled,
      setEnabled: (next: boolean) => {
        setVisibleEdgeTypes((current) => {
          const enabledNow = current.has("semantic")
          if (enabledNow === next) return current
          const updated = new Set(current)
          if (next) updated.add("semantic")
          else updated.delete("semantic")
          return updated
        })
      },
      threshold: semanticThreshold,
      setThreshold: (next: number) =>
        setSemanticThreshold(Math.min(1, Math.max(0, next))),
      topK: semanticTopK,
      setTopK: (next: number) =>
        setSemanticTopK(
          Math.min(
            NOTES_GRAPH_SEMANTIC_MAX_TOP_K,
            Math.max(1, Math.trunc(next))
          )
        )
    },
    semanticIndex,
    allNotes,
    canExpand: Boolean(
      enabled &&
        !graphQuery.isFetchingNextPage &&
        graph?.has_more &&
        graph.cursor
    ),
    expand,
    focus,
    showAllNotes,
    refresh,
    isOffline: !options.isOnline,
    isLoading: Boolean(authorityScope && graphQuery.isLoading && !graph),
    error: graphQuery.error
  }
}
