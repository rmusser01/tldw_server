import type {
  NotesGraphEdgeType,
  NotesGraphResponse
} from "@/services/note-graph-suggestions"

/**
 * What the Notes graph shows, apart from what it fetches (NL-11, #3101).
 *
 * All notes now holds up to the server's cap (500), so the client keeps a
 * large graph usable: a "Linked notes only" filter, and a layout that stays
 * fast for the size.
 */

type NotesGraphCanvasLayout = "dagre" | "circle" | "grid" | "concentric"

/**
 * What a graph with no focus note (All notes, or a tag filter) asks for: the
 * server's projection ceiling. The server clamps it to its All-notes budget,
 * so the All-notes cap stays one server setting (NOTES_GRAPH_ALL_NOTES_NOTE_CAP).
 */
export const NOTES_GRAPH_LIBRARY_REQUEST = {
  maxNodes: 1_000,
  maxEdges: 4_000
} as const

/**
 * The Tree (dagre) layout runs on the main thread. Measured 2026-11-19 with
 * cytoscape-dagre: 200 items and 600 links took about 1 s, 300 items and
 * 1,200 links about 8 s, 500 items and 1,200 links about 10 s. Rings took
 * under 35 ms for each.
 */
export const NOTES_GRAPH_TREE_LAYOUT_LIMIT = { nodes: 200, edges: 600 } as const

/** Above this many links, link labels show only on a hovered or selected link. */
export const NOTES_GRAPH_EDGE_LABEL_LIMIT = 150

const NOTE_TO_NOTE_EDGE_TYPES: ReadonlySet<NotesGraphEdgeType> = new Set([
  "manual",
  "wikilink",
  "backlink",
  "semantic"
])

const isNoteNode = (id: string): boolean => id.startsWith("note:")

/** The layout the canvas runs: Tree falls back to Rings when it would stall. */
export const resolveNotesGraphCanvasLayout = <T extends NotesGraphCanvasLayout>(
  layout: T,
  nodeCount: number,
  edgeCount: number
): T | "concentric" =>
  layout === "dagre" &&
  (nodeCount > NOTES_GRAPH_TREE_LAYOUT_LIMIT.nodes ||
    edgeCount > NOTES_GRAPH_TREE_LAYOUT_LIMIT.edges)
    ? "concentric"
    : layout

/** How many of the graph's links the canvas draws. */
export const countVisibleNotesGraphEdges = (
  graph: NotesGraphResponse,
  visibleEdgeTypes: ReadonlySet<NotesGraphEdgeType>
): number => graph.edges.filter((edge) => visibleEdgeTypes.has(edge.type)).length

/**
 * "Linked notes only": keeps the notes that a shown link joins to another
 * note, the tags and sources of those notes, and `keepNodeId` (the focus
 * note), so an ego graph never loses its centre.
 */
export const keepLinkedNotes = (
  graph: NotesGraphResponse,
  visibleEdgeTypes: ReadonlySet<NotesGraphEdgeType>,
  keepNodeId: string | null = null
): NotesGraphResponse => {
  const shownEdges = graph.edges.filter((edge) => visibleEdgeTypes.has(edge.type))
  const keptNotes = new Set<string>(keepNodeId ? [keepNodeId] : [])
  for (const edge of shownEdges) {
    if (
      NOTE_TO_NOTE_EDGE_TYPES.has(edge.type) &&
      edge.source !== edge.target &&
      isNoteNode(edge.source) &&
      isNoteNode(edge.target)
    ) {
      keptNotes.add(edge.source)
      keptNotes.add(edge.target)
    }
  }
  const kept = new Set(keptNotes)
  for (const edge of shownEdges) {
    if (keptNotes.has(edge.source) && !isNoteNode(edge.target)) kept.add(edge.target)
    if (keptNotes.has(edge.target) && !isNoteNode(edge.source)) kept.add(edge.source)
  }
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => kept.has(node.id)),
    edges: graph.edges.filter(
      (edge) => kept.has(edge.source) && kept.has(edge.target)
    )
  }
}
