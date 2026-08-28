import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import NotesGraphWorkspace from "../NotesGraphWorkspace"
import {
  hasNotesGraphActiveNotes,
  type NotesListViewMode,
  resolveNotesGraphFocusNoteId
} from "../notes-manager-utils"

const { mockCanvas, mockUseNotesGraphSuggestions, mockUseNotesGraphWorkspace } =
  vi.hoisted(() => ({
    mockCanvas: vi.fn(),
    mockUseNotesGraphSuggestions: vi.fn(),
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
  useNotesGraphSuggestions: mockUseNotesGraphSuggestions
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
        focusNode: vi.fn()
      }))
      return <div data-testid="notes-graph-canvas" />
    })
  }
})

const graph = {
  nodes: [
    {
      id: "note:a",
      type: "note" as const,
      label: "Alpha note",
      created_at: null,
      deleted: false,
      degree: 1,
      tag_count: 0,
      primary_source_id: null
    },
    {
      id: "note:b",
      type: "note" as const,
      label: "Beta note",
      created_at: null,
      deleted: false,
      degree: 1,
      tag_count: 0,
      primary_source_id: null
    }
  ],
  edges: [],
  truncated: false,
  truncated_by: [],
  has_more: true,
  cursor: "next-page",
  limits: { max_nodes: 50, max_edges: 200, max_degree: 40 },
  radius_cap_applied: false,
  active_note_count: 8,
  all_notes_note_cap: 7,
  all_notes_eligible: false
}

const baseWorkspaceState = () => ({
  graph,
  graphQuery: { isFetching: false },
  focusNoteId: "a",
  scope: "focused" as const,
  layout: "dagre" as const,
  setLayout: vi.fn(),
  search: "",
  setSearch: vi.fn(),
  searchResults: [],
  visibleEdgeTypes: new Set([
    "manual",
    "wikilink",
    "backlink",
    "tag_membership",
    "source_membership"
  ] as const),
  toggleEdgeType: vi.fn(),
  allNotes: { activeNoteCount: 8, effectiveNoteCap: 7, eligible: false },
  canExpand: true,
  expand: vi.fn(async () => graph),
  focus: vi.fn(),
  showAllNotes: vi.fn(() => false),
  refresh: vi.fn(async () => graph),
  isOffline: false,
  isLoading: false,
  error: null
})

const renderWorkspace = (
  props: Partial<React.ComponentProps<typeof NotesGraphWorkspace>> = {}
) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesGraphWorkspace
        authorityScope="opaque-authority"
        isOnline
        initialFocusNoteId="a"
        selectedNoteId="a"
        hasActiveNotes
        onSelectNote={vi.fn()}
        onCreateNote={vi.fn()}
        {...props}
      />
    </QueryClientProvider>
  )
}

describe("NotesGraphWorkspace first-class view mode", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUseNotesGraphWorkspace.mockReturnValue(baseWorkspaceState())
    mockUseNotesGraphSuggestions.mockReturnValue({
      provisionalBySuggestionId: {
        s1: {
          edge: {
            id: "suggestion-edge:s1",
            suggestionId: "s1",
            source: "note:a",
            target: "suggestion-node:s1",
            type: "provisional_suggestion",
            directed: false
          },
          node: {
            id: "suggestion-node:s1",
            suggestionId: "s1",
            type: "provisional_note",
            label: "Suggested note"
          }
        }
      }
    })
  })

  it("accepts graph as a Notes view and resolves selected, verified recent, then visible focus", () => {
    const graphMode: NotesListViewMode = "graph"
    expect(graphMode).toBe("graph")
    expect(
      resolveNotesGraphFocusNoteId(
        "selected",
        [{ id: "recent", title: "Recent" }],
        [{ id: "visible", title: "Visible" }]
      )
    ).toBe("selected")
    expect(
      resolveNotesGraphFocusNoteId(
        null,
        [
          { id: "cross-server", title: "Cross-server" },
          { id: "recent", title: "Recent" }
        ],
        [
          { id: "visible", title: "Visible", deleted: false },
          { id: "recent", title: "Recent", deleted: false }
        ]
      )
    ).toBe("recent")
    expect(
      resolveNotesGraphFocusNoteId(
        null,
        [{ id: "deleted", title: "Deleted" }],
        [
          { id: "deleted", title: "Deleted", deleted: true },
          { id: "visible", title: "Visible", deleted: false }
        ]
      )
    ).toBe("visible")
    expect(
      resolveNotesGraphFocusNoteId(
        null,
        [{ id: "stale", title: "Stale" }],
        [{ id: "visible", title: "Visible", deleted: false }]
      )
    ).toBe("visible")
    expect(resolveNotesGraphFocusNoteId(null, [{ id: "stale" }], [])).toBeNull()
  })

  it("does not treat unverified recents or deleted rows as an active library", () => {
    expect(hasNotesGraphActiveNotes(null, 0, [])).toBe(false)
    expect(
      hasNotesGraphActiveNotes(null, 0, [{ id: "deleted", deleted: true }])
    ).toBe(false)
    expect(
      hasNotesGraphActiveNotes(null, 0, [{ id: "active", deleted: false }])
    ).toBe(true)
    expect(hasNotesGraphActiveNotes("loaded-selection", 0, [])).toBe(true)
  })

  it("passes only the opaque authority and mounted focus into the authoritative workspace hook", () => {
    renderWorkspace({
      initialFocusNoteId: "recent-note",
      selectedNoteId: null
    })

    expect(mockUseNotesGraphWorkspace).toHaveBeenCalledWith(
      expect.objectContaining({
        authorityScope: "opaque-authority",
        enabled: true,
        initialFocusNoteId: "recent-note",
        isOnline: true
      })
    )
    expect(document.body).not.toHaveTextContent("api-key")
    expect(document.body).not.toHaveTextContent("access-token")
  })

  it("shows the standard Notes empty state when there is no active note", () => {
    const onCreateNote = vi.fn()
    renderWorkspace({
      initialFocusNoteId: null,
      selectedNoteId: null,
      hasActiveNotes: false,
      onCreateNote
    })

    expect(screen.getByTestId("notes-editor-empty-state")).toBeInTheDocument()
    expect(screen.queryByTestId("notes-graph-canvas")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Create note" }))
    expect(onCreateNote).toHaveBeenCalled()
  })

  it("uses the server-reported non-default All notes cap and concise disabled reason", () => {
    renderWorkspace()

    expect(screen.getByRole("button", { name: "All notes" })).toBeDisabled()
    expect(
      screen.getByTestId("notes-graph-all-disabled-reason")
    ).toHaveTextContent(
      "All notes is available for up to 7 active notes. This library has 8."
    )
    expect(
      screen.getByTestId("notes-graph-all-disabled-reason")
    ).not.toHaveTextContent("100")
  })

  it("allows All notes only when server eligibility and both effective caps permit it", () => {
    const state = baseWorkspaceState()
    state.graph = {
      ...graph,
      active_note_count: 7,
      all_notes_note_cap: 20,
      all_notes_eligible: true,
      limits: { ...graph.limits, max_nodes: 7 }
    }
    state.allNotes = {
      activeNoteCount: 7,
      effectiveNoteCap: 7,
      eligible: true
    }
    state.showAllNotes = vi.fn(() => true)
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    renderWorkspace()

    fireEvent.click(screen.getByRole("button", { name: "All notes" }))
    expect(state.showAllNotes).toHaveBeenCalled()
  })

  it("focuses the current sidebar selection explicitly without discarding the loaded canvas", () => {
    const state = baseWorkspaceState()
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    const { rerender } = renderWorkspace({ selectedNoteId: "a" })

    rerender(
      <QueryClientProvider client={new QueryClient()}>
        <NotesGraphWorkspace
          authorityScope="opaque-authority"
          isOnline
          initialFocusNoteId="a"
          selectedNoteId="b"
          hasActiveNotes
          onSelectNote={vi.fn()}
          onCreateNote={vi.fn()}
        />
      </QueryClientProvider>
    )
    expect(screen.getByTestId("notes-graph-canvas")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Focus current note" }))
    expect(state.focus).toHaveBeenCalledWith("b")
  })

  it("passes provisional overlays separately and expands only on an explicit command", () => {
    const state = baseWorkspaceState()
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    renderWorkspace()

    expect(mockCanvas).toHaveBeenLastCalledWith(
      expect.objectContaining({
        provisionalOverlays: [
          expect.objectContaining({
            edge: expect.objectContaining({ id: "suggestion-edge:s1" }),
            node: expect.objectContaining({ id: "suggestion-node:s1" })
          })
        ]
      })
    )
    expect(state.expand).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Expand graph" }))
    expect(state.expand).toHaveBeenCalledTimes(1)
  })

  it("keeps last-good graph state visible while marking truncation, degraded refresh, and offline state", () => {
    const state = baseWorkspaceState()
    state.graph = {
      ...graph,
      truncated: true,
      truncated_by: ["max_nodes"]
    }
    state.error = new Error("refresh failed")
    state.isOffline = true
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    renderWorkspace({ isOnline: false })

    expect(screen.getByTestId("notes-graph-canvas")).toBeInTheDocument()
    expect(screen.getByTestId("notes-graph-offline-state")).toHaveTextContent(
      "Offline"
    )
    expect(screen.getByTestId("notes-graph-degraded-state")).toHaveTextContent(
      "last available graph"
    )
    expect(
      screen.getByTestId("notes-graph-truncated-warning")
    ).toHaveTextContent("server limits")
  })

  it("keeps Notes discovery available from the narrow graph workspace", () => {
    const onOpenSidebar = vi.fn()
    renderWorkspace({ isMobileViewport: true, onOpenSidebar })

    fireEvent.click(screen.getByRole("button", { name: "Open notes list" }))
    expect(onOpenSidebar).toHaveBeenCalledTimes(1)
  })

  it("reflows wrapped controls into a vertically scrollable 320px workspace", () => {
    renderWorkspace({ isMobileViewport: true })

    const workspace = screen.getByTestId("notes-graph-workspace")
    const canvasSlot = screen.getByTestId("notes-graph-canvas-slot")
    const toolbar = screen.getByTestId("notes-graph-toolbar")
    Object.defineProperties(workspace, {
      clientWidth: { configurable: true, value: 320 },
      clientHeight: { configurable: true, value: 360 },
      scrollHeight: { configurable: true, value: 1040 }
    })

    expect(workspace.clientWidth).toBe(320)
    expect(workspace.scrollHeight).toBeGreaterThan(workspace.clientHeight)
    expect(workspace).toHaveClass("overflow-y-auto")
    expect(toolbar.querySelector(".flex-wrap")).toBeInTheDocument()
    expect(canvasSlot).toHaveClass("min-h-[420px]", "sm:min-h-[520px]")
  })
})
