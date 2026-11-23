/**
 * NL-11 review fixes (#3101): "Linked notes only" must never hand the canvas
 * a suggestion overlay whose ends it does not draw (Cytoscape throws and the
 * route error boundary replaces /notes), and keeps the selected note.
 */
import type {
  NotesGraphNode,
  NotesGraphResponse
} from "@/services/note-graph-suggestions"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import NotesGraphWorkspace from "../NotesGraphWorkspace"

const { mockCanvas, mockFocusNode, mockSuggestions, mockUseNotesGraphWorkspace } = vi.hoisted(() => ({
  mockCanvas: vi.fn(),
  mockFocusNode: vi.fn(),
  mockSuggestions: vi.fn(),
  mockUseNotesGraphWorkspace: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      options?: string | { defaultValue?: string; [key: string]: unknown }
    ) => {
      const fallback =
        typeof options === "string" ? options : options?.defaultValue ?? key
      if (typeof options !== "object") return fallback
      return Object.entries(options).reduce(
        (value, [name, replacement]) =>
          value.replace(`{{${name}}}`, String(replacement)),
        fallback
      )
    }
  })
}))

vi.mock("../hooks/useNotesGraphWorkspace", () => ({
  useNotesGraphWorkspace: mockUseNotesGraphWorkspace
}))

vi.mock("../hooks/useNotesGraphSuggestions", () => ({
  useNotesGraphSuggestions: mockSuggestions
}))

vi.mock("../NotesGraphInspector", () => ({
  default: () => <div data-testid="notes-graph-inspector" />
}))

vi.mock("../NotesGraphCanvas", async () => {
  const ReactModule = await import("react")
  return {
    default: ReactModule.forwardRef((props: Record<string, unknown>, ref) => {
      mockCanvas(props)
      ReactModule.useImperativeHandle(ref, () => ({
        zoomIn: vi.fn(),
        zoomOut: vi.fn(),
        fit: vi.fn(),
        focusNode: mockFocusNode
      }))
      return <div data-testid="notes-graph-canvas" />
    })
  }
})

const noteNode = (id: string, label = id): NotesGraphNode => ({
  id,
  type: "note",
  label,
  created_at: null,
  deleted: false,
  degree: 0,
  tag_count: 0,
  primary_source_id: null
})

const baseGraph: NotesGraphResponse = {
  nodes: [noteNode("note:a", "Alpha note"), noteNode("note:b", "Beta note"), noteNode("note:c", "Lonely note")],
  edges: [
    {
      id: "wikilink:a:b",
      source: "note:a",
      target: "note:b",
      type: "wikilink",
      directed: true,
      weight: null,
      label: null
    }
  ],
  truncated: false,
  truncated_by: [],
  has_more: false,
  cursor: null,
  limits: { max_nodes: 120, max_edges: 480, max_degree: 40 },
  radius_cap_applied: false,
  active_note_count: 3,
  all_notes_note_cap: 500,
  all_notes_eligible: true,
  manual_link_authorized: false
}

const workspaceState = (overrides: Record<string, unknown> = {}) => ({
  graph: baseGraph,
  graphQuery: { isFetching: false },
  focusNoteId: "a",
  scope: "focused" as const,
  tagFilter: null,
  filterByTag: vi.fn(),
  layout: "dagre" as const,
  setLayout: vi.fn(),
  search: "",
  setSearch: vi.fn(),
  searchResults: [],
  visibleEdgeTypes: new Set(["manual", "wikilink", "backlink", "tag_membership", "source_membership"]),
  toggleEdgeType: vi.fn(),
  semantic: {
    enabled: false,
    setEnabled: vi.fn(),
    threshold: 0.75,
    setThreshold: vi.fn(),
    topK: 10,
    maxTopK: 50,
    setTopK: vi.fn(),
    focusRequired: false,
    reset: vi.fn()
  },
  semanticIndex: undefined,
  allNotes: { activeNoteCount: 3, effectiveNoteCap: 500, eligible: true },
  canExpand: false,
  expand: vi.fn(),
  focus: vi.fn(),
  showAllNotes: vi.fn(() => true),
  refresh: vi.fn(),
  isPermissionDenied: false,
  createManualLink: vi.fn(),
  manualLinkPendingEdgeIds: new Set<string>(),
  queryIdentity: "q",
  isOffline: false,
  isLoading: false,
  error: null,
  ...overrides
})

type WorkspaceProps = React.ComponentProps<typeof NotesGraphWorkspace>

const renderWorkspace = (props: Partial<WorkspaceProps> = {}) => {
  const client = new QueryClient()
  const view = (next: Partial<WorkspaceProps>) => (
    <QueryClientProvider client={client}>
      <NotesGraphWorkspace
        authorityScope="authority-a"
        isOnline
        initialFocusNoteId="a"
        selectedNoteId="a"
        hasActiveNotes
        onSelectNote={vi.fn()}
        onCreateNote={vi.fn()}
        {...props}
        {...next}
      />
    </QueryClientProvider>
  )
  const rendered = render(view({}))
  return { ...rendered, rerenderWith: (next: Partial<WorkspaceProps>) => rendered.rerender(view(next)) }
}

const lastCanvasProps = () =>
  mockCanvas.mock.calls.at(-1)?.[0] as {
    graph: NotesGraphResponse
    layout: string
    onSelectNode: (nodeId: string) => void
  }

const overlay = (id: string, source: string, target: string, ephemeral: boolean) => ({
  edge: {
    id: `suggestion-edge:${id}`,
    suggestionId: id,
    source,
    target,
    type: "provisional_suggestion" as const,
    directed: false as const
  },
  node: ephemeral
    ? {
        id: target,
        suggestionId: id,
        type: "provisional_note" as const,
        label: "Suggested note" as const
      }
    : null
})

describe("NotesGraphWorkspace Linked notes only with suggestions (NL-11 review)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => cleanup())

  it("keeps the selected note and drops overlays with an end the canvas does not draw", () => {
    const graph = {
      ...baseGraph,
      nodes: [...baseGraph.nodes, noteNode("note:d", "Other lonely note")],
      suggestions_authorized: true
    }
    mockUseNotesGraphWorkspace.mockReturnValue(workspaceState({ graph, scope: "all" }))
    mockSuggestions.mockReturnValue({
      provisionalBySuggestionId: {
        toHidden: overlay("toHidden", "note:c", "note:d", false),
        toNew: overlay("toNew", "note:c", "suggestion-node:toNew", true)
      }
    })
    renderWorkspace({ selectedNoteId: "c" })

    fireEvent.click(screen.getByRole("checkbox", { name: "Linked notes only" }))

    const props = lastCanvasProps() as ReturnType<typeof lastCanvasProps> & {
      provisionalOverlays: Array<ReturnType<typeof overlay>>
    }
    const drawn = new Set([
      ...props.graph.nodes.map((node) => node.id),
      ...props.provisionalOverlays.flatMap((entry) => (entry.node ? [entry.node.id] : []))
    ])
    expect(drawn.has("note:c")).toBe(true)
    expect(drawn.has("note:d")).toBe(false)
    expect(props.provisionalOverlays.map((entry) => entry.edge.suggestionId)).toEqual(["toNew"])
    for (const entry of props.provisionalOverlays) {
      expect(drawn.has(entry.edge.source)).toBe(true)
      expect(drawn.has(entry.edge.target)).toBe(true)
    }
  })
})
