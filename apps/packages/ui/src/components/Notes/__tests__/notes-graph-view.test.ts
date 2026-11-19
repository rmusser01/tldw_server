import type {
  NotesGraphEdge,
  NotesGraphEdgeType,
  NotesGraphNode,
  NotesGraphResponse
} from "@/services/note-graph-suggestions"
import { describe, expect, it } from "vitest"

import {
  NOTES_GRAPH_TREE_LAYOUT_LIMIT,
  keepLinkedNotes,
  resolveNotesGraphCanvasLayout
} from "../notes-graph-view"

const node = (id: string): NotesGraphNode => ({
  id,
  type: id.startsWith("note:") ? "note" : id.startsWith("tag:") ? "tag" : "source",
  label: id,
  created_at: null,
  deleted: false,
  degree: null,
  tag_count: null,
  primary_source_id: null
})

const edge = (
  source: string,
  target: string,
  type: NotesGraphEdgeType
): NotesGraphEdge => ({
  id: `${type}:${source}:${target}`,
  source,
  target,
  type,
  directed: false,
  weight: null,
  label: null
})

const graph = (nodes: string[], edges: NotesGraphEdge[]): NotesGraphResponse => ({
  nodes: nodes.map(node),
  edges,
  truncated: false,
  truncated_by: [],
  has_more: false,
  cursor: null,
  limits: { max_nodes: 625, max_edges: 1200, max_degree: 40 },
  radius_cap_applied: false,
  active_note_count: nodes.length,
  all_notes_note_cap: 500,
  all_notes_eligible: true,
  manual_link_authorized: false
})

const ALL_EDGES = new Set<NotesGraphEdgeType>([
  "manual",
  "wikilink",
  "backlink",
  "tag_membership",
  "source_membership"
])

describe("keepLinkedNotes (NL-11 'Linked notes only')", () => {
  const library = graph(
    ["note:a", "note:b", "note:lonely", "tag:reading", "tag:orphan-tag", "source:web"],
    [
      edge("note:a", "note:b", "wikilink"),
      edge("note:a", "tag:reading", "tag_membership"),
      edge("note:lonely", "tag:orphan-tag", "tag_membership"),
      edge("note:b", "source:web", "source_membership")
    ]
  )

  it("keeps notes joined to another note, with their tags and sources", () => {
    const filtered = keepLinkedNotes(library, ALL_EDGES)

    expect(filtered.nodes.map((entry) => entry.id)).toEqual([
      "note:a",
      "note:b",
      "tag:reading",
      "source:web"
    ])
    expect(filtered.edges.map((entry) => entry.id)).toEqual([
      "wikilink:note:a:note:b",
      "tag_membership:note:a:tag:reading",
      "source_membership:note:b:source:web"
    ])
  })

  it("does not count a hidden link type or a shared tag as a link", () => {
    const filtered = keepLinkedNotes(
      library,
      new Set<NotesGraphEdgeType>(["tag_membership"])
    )

    expect(filtered.nodes).toEqual([])
  })

  it("keeps the focus note even when it has no links", () => {
    const filtered = keepLinkedNotes(library, ALL_EDGES, "note:lonely")

    expect(filtered.nodes.map((entry) => entry.id)).toContain("note:lonely")
    expect(filtered.nodes.map((entry) => entry.id)).toContain("tag:orphan-tag")
  })
})

describe("resolveNotesGraphCanvasLayout", () => {
  it("keeps Tree for a graph it lays out quickly", () => {
    expect(resolveNotesGraphCanvasLayout("dagre", 120, 300)).toBe("dagre")
  })

  it("uses Rings instead of Tree above the measured size limits", () => {
    expect(
      resolveNotesGraphCanvasLayout("dagre", NOTES_GRAPH_TREE_LAYOUT_LIMIT.nodes + 1, 10)
    ).toBe("concentric")
    expect(
      resolveNotesGraphCanvasLayout("dagre", 10, NOTES_GRAPH_TREE_LAYOUT_LIMIT.edges + 1)
    ).toBe("concentric")
  })

  it("leaves the other layouts alone at any size", () => {
    expect(resolveNotesGraphCanvasLayout("grid", 625, 1200)).toBe("grid")
    expect(resolveNotesGraphCanvasLayout("circle", 625, 1200)).toBe("circle")
  })
})
