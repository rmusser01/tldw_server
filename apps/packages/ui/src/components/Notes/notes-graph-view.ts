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

/** Real links: semantic similarity is not one (NL-11 review). */
const NOTE_TO_NOTE_EDGE_TYPES: ReadonlySet<NotesGraphEdgeType> = new Set([
  "manual",
  "wikilink",
  "backlink"
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
 * "Linked notes only": keeps the linked notes, the tags and sources of those
 * notes, and `keepNodeIds` (the focus and selected notes), so the graph
 * never loses what the reader is on. A note is linked when the server says
 * so (`has_note_links`, counted before the degree cap prunes edges); from an
 * older server, when a shown real link joins it to another note.
 */
export const keepLinkedNotes = (
  graph: NotesGraphResponse,
  visibleEdgeTypes: ReadonlySet<NotesGraphEdgeType>,
  keepNodeIds: ReadonlyArray<string | null> = []
): NotesGraphResponse => {
  const shownEdges = graph.edges.filter((edge) => visibleEdgeTypes.has(edge.type))
  const keptNotes = new Set<string>(
    keepNodeIds.filter((id): id is string => Boolean(id))
  )
  const serverStatus = new Map<string, boolean>()
  for (const node of graph.nodes) {
    if (typeof node.has_note_links !== "boolean") continue
    serverStatus.set(node.id, node.has_note_links)
    if (node.has_note_links) keptNotes.add(node.id)
  }
  for (const edge of shownEdges) {
    if (
      NOTE_TO_NOTE_EDGE_TYPES.has(edge.type) &&
      edge.source !== edge.target &&
      isNoteNode(edge.source) &&
      isNoteNode(edge.target)
    ) {
      if (!serverStatus.has(edge.source)) keptNotes.add(edge.source)
      if (!serverStatus.has(edge.target)) keptNotes.add(edge.target)
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

type NotesGraphOverlayEnds = {
  edge: { source: string; target: string }
  node: { id: string } | null
}

/**
 * The suggestion overlays the canvas can draw: both ends must be a drawn
 * node, or the target the overlay's own ephemeral node. Cytoscape throws on
 * an edge with a missing end, which takes the whole page down.
 */
export const overlaysWithDrawnEnds = <T extends NotesGraphOverlayEnds>(
  overlays: T[],
  graph: Pick<NotesGraphResponse, "nodes">
): T[] => {
  const drawn = new Set(graph.nodes.map((node) => node.id))
  return overlays.filter(
    (overlay) =>
      drawn.has(overlay.edge.source) &&
      (drawn.has(overlay.edge.target) || overlay.node?.id === overlay.edge.target)
  )
}
