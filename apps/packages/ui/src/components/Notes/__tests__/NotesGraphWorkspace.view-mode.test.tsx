import type { NotesGraphResponse } from "@/services/note-graph-suggestions"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import React from "react"
import { createRoot } from "react-dom/client"
import { beforeEach, describe, expect, it, vi } from "vitest"

import NotesGraphWorkspace from "../NotesGraphWorkspace"
import {
  type NotesListViewMode,
  hasNotesGraphActiveNotes,
  resolveNotesGraphFocusNoteId
} from "../notes-manager-utils"

const {
  mockCanvas,
  mockCommittedCanvas,
  mockCommittedSuggestionInput,
  mockFocusNode,
  mockNestedSuggestionRequest,
  mockUseNotesGraphSuggestions,
  mockUseNotesGraphWorkspace
} = vi.hoisted(() => ({
  mockCanvas: vi.fn(),
  mockCommittedCanvas: vi.fn(),
  mockCommittedSuggestionInput: vi.fn(),
  mockFocusNode: vi.fn(),
  mockNestedSuggestionRequest: vi.fn(),
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
      ReactModule.useLayoutEffect(() => {
        mockCommittedCanvas(props)
      })
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

const graph: NotesGraphResponse = {
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
  all_notes_eligible: false,
  suggestions_authorized: true
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

const workspaceStateWithCommonNote = (focusNoteId: string | null) => ({
  ...baseWorkspaceState(),
  graph: {
    ...graph,
    nodes: [{ ...graph.nodes[0], id: "note:common" }, graph.nodes[1]]
  },
  focusNoteId
})

type SuggestionHookInput = {
  authorityScope: string | null
  enabled: boolean
  noteId: string | null
}

const useTrackedSuggestionController = (options: SuggestionHookInput) => {
  React.useLayoutEffect(() => {
    const committedInput = {
      authorityScope: options.authorityScope,
      enabled: options.enabled,
      noteId: options.noteId
    }
    mockCommittedSuggestionInput(committedInput)
    if (options.enabled) mockNestedSuggestionRequest(committedInput)
  })

  const sourceNoteId = options.noteId ?? "a"
  return {
    provisionalBySuggestionId: {
      s1: {
        edge: {
          id: "suggestion-edge:s1",
          suggestionId: "s1",
          source: `note:${sourceNoteId}`,
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
  }
}

const renderWorkspace = (
  props: Partial<React.ComponentProps<typeof NotesGraphWorkspace>> = {}
) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  const renderCurrent = (
    currentProps: Partial<React.ComponentProps<typeof NotesGraphWorkspace>>
  ) => (
    <QueryClientProvider client={queryClient}>
      <NotesGraphWorkspace
        authorityScope="opaque-authority"
        isOnline
        initialFocusNoteId="a"
        selectedNoteId="a"
        hasActiveNotes
        onSelectNote={vi.fn()}
        onCreateNote={vi.fn()}
        {...currentProps}
      />
    </QueryClientProvider>
  )
  const rendered = render(renderCurrent(props))
  return {
    ...rendered,
    rerenderWorkspace: (
      nextProps: Partial<React.ComponentProps<typeof NotesGraphWorkspace>>
    ) => rendered.rerender(renderCurrent(nextProps))
  }
}

describe("NotesGraphWorkspace first-class view mode", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUseNotesGraphWorkspace.mockReturnValue(baseWorkspaceState())
    mockUseNotesGraphSuggestions.mockImplementation(
      useTrackedSuggestionController
    )
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

  it("freezes initial focus per verified authority instead of component lifetime", () => {
    const queryClient = new QueryClient()
    const view = render(
      <QueryClientProvider client={queryClient}>
        <NotesGraphWorkspace
          authorityScope="scope-a"
          isOnline
          initialFocusNoteId="account-a-note"
          selectedNoteId="account-a-note"
          hasActiveNotes
          onSelectNote={vi.fn()}
          onCreateNote={vi.fn()}
        />
      </QueryClientProvider>
    )
    expect(mockUseNotesGraphWorkspace).toHaveBeenLastCalledWith(
      expect.objectContaining({
        authorityScope: "scope-a",
        initialFocusNoteId: "account-a-note"
      })
    )

    view.rerender(
      <QueryClientProvider client={queryClient}>
        <NotesGraphWorkspace
          authorityScope="scope-b"
          isOnline
          initialFocusNoteId="account-b-note"
          selectedNoteId="account-b-note"
          hasActiveNotes
          onSelectNote={vi.fn()}
          onCreateNote={vi.fn()}
        />
      </QueryClientProvider>
    )
    expect(mockUseNotesGraphWorkspace).toHaveBeenLastCalledWith(
      expect.objectContaining({
        authorityScope: "scope-b",
        initialFocusNoteId: "account-b-note"
      })
    )

    view.rerender(
      <QueryClientProvider client={queryClient}>
        <NotesGraphWorkspace
          authorityScope="scope-b"
          isOnline
          initialFocusNoteId="later-account-b-note"
          selectedNoteId="later-account-b-note"
          hasActiveNotes
          onSelectNote={vi.fn()}
          onCreateNote={vi.fn()}
        />
      </QueryClientProvider>
    )
    expect(mockUseNotesGraphWorkspace).toHaveBeenLastCalledWith(
      expect.objectContaining({ initialFocusNoteId: "account-b-note" })
    )
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

  it("keys suggestions to a selected loaded note without refocusing the graph neighborhood", () => {
    const state = baseWorkspaceState()
    const onSelectNote = vi.fn()
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    renderWorkspace({ onSelectNote })

    const canvasProps = mockCanvas.mock.calls.at(-1)?.[0] as {
      onSelectNode: (nodeId: string) => void
    }
    act(() => canvasProps.onSelectNode("note:b"))

    expect(onSelectNote).toHaveBeenCalledWith("b")
    expect(state.focus).not.toHaveBeenCalled()
    expect(mockUseNotesGraphSuggestions).toHaveBeenLastCalledWith(
      expect.objectContaining({ enabled: true, noteId: "b" })
    )
    expect(mockCanvas).toHaveBeenLastCalledWith(
      expect.objectContaining({ selectedNodeId: "note:b" })
    )
  })

  it.each([
    ["b", "b", true],
    ["missing", null, false]
  ] as const)(
    "fails closed in the same commit when controlled selection changes to %s",
    (controlledNoteId, expectedSuggestionNoteId, expectedEnabled) => {
      const state = baseWorkspaceState()
      mockUseNotesGraphWorkspace.mockReturnValue(state)
      const { rerenderWorkspace } = renderWorkspace({ selectedNoteId: "a" })

      mockCommittedCanvas.mockClear()
      mockCommittedSuggestionInput.mockClear()
      rerenderWorkspace({ selectedNoteId: controlledNoteId })

      expect(mockCommittedSuggestionInput).toHaveBeenCalled()
      for (const [input] of mockCommittedSuggestionInput.mock.calls) {
        expect(input).toEqual({
          authorityScope: "opaque-authority",
          enabled: expectedEnabled,
          noteId: expectedSuggestionNoteId
        })
      }
      for (const [canvasProps] of mockCommittedCanvas.mock.calls) {
        expect(canvasProps.selectedNodeId).toBe(
          expectedSuggestionNoteId ? `note:${expectedSuggestionNoteId}` : null
        )
        if (expectedEnabled) {
          expect(canvasProps.provisionalOverlays).toEqual([
            expect.objectContaining({
              edge: expect.objectContaining({ source: "note:b" })
            })
          ])
        } else {
          expect(canvasProps).toEqual(
            expect.objectContaining({
              provisionalOverlays: [],
              showProvisional: false
            })
          )
        }
      }
      if (!expectedEnabled) {
        expect(
          screen.queryByRole("tab", { name: "Suggestions" })
        ).not.toBeInTheDocument()
      }
    }
  )

  it("invalidates an unchanged controlled note across authority until current-authority focus owns selection", () => {
    const authorityA = workspaceStateWithCommonNote("common")
    const authorityB = workspaceStateWithCommonNote(null)
    mockUseNotesGraphWorkspace.mockImplementation(({ authorityScope }) =>
      authorityScope === "scope-b" ? authorityB : authorityA
    )
    const { rerenderWorkspace } = renderWorkspace({
      authorityScope: "scope-a",
      initialFocusNoteId: "common",
      selectedNoteId: "common"
    })

    mockCommittedCanvas.mockClear()
    mockCommittedSuggestionInput.mockClear()
    mockNestedSuggestionRequest.mockClear()
    rerenderWorkspace({
      authorityScope: "scope-b",
      initialFocusNoteId: "common",
      selectedNoteId: "common"
    })

    expect(mockCommittedSuggestionInput).toHaveBeenCalled()
    for (const [input] of mockCommittedSuggestionInput.mock.calls) {
      expect(input).toEqual({
        authorityScope: "scope-b",
        enabled: false,
        noteId: null
      })
    }
    expect(mockNestedSuggestionRequest).not.toHaveBeenCalled()
    expect(
      screen.queryByRole("tab", { name: "Suggestions" })
    ).not.toBeInTheDocument()
    for (const [canvasProps] of mockCommittedCanvas.mock.calls) {
      expect(canvasProps).toEqual(
        expect.objectContaining({
          selectedNodeId: null,
          provisionalOverlays: [],
          showProvisional: false
        })
      )
    }

    authorityB.focusNoteId = "b"
    mockCommittedSuggestionInput.mockClear()
    mockNestedSuggestionRequest.mockClear()
    rerenderWorkspace({
      authorityScope: "scope-b",
      initialFocusNoteId: "common",
      selectedNoteId: "common"
    })

    expect(mockCommittedSuggestionInput).toHaveBeenLastCalledWith({
      authorityScope: "scope-b",
      enabled: true,
      noteId: "b"
    })
    expect(mockNestedSuggestionRequest).toHaveBeenLastCalledWith({
      authorityScope: "scope-b",
      enabled: true,
      noteId: "b"
    })
  })

  it("does not publish selection ownership from an abandoned authority transition", async () => {
    const suspendedTransition = new Promise<never>(() => {})
    let suspendedBRenderCount = 0
    let shouldSuspendB = true
    const authorityA = workspaceStateWithCommonNote("common")
    const authorityB = workspaceStateWithCommonNote(null)
    mockUseNotesGraphWorkspace.mockImplementation(({ authorityScope }) =>
      authorityScope === "scope-b" ? authorityB : authorityA
    )

    const Suspender = ({ authorityScope }: { authorityScope: string }) => {
      if (authorityScope === "scope-b" && shouldSuspendB) {
        suspendedBRenderCount += 1
        throw suspendedTransition
      }
      return null
    }
    const queryClient = new QueryClient()
    const renderScope = (authorityScope: string) => (
      <React.StrictMode>
        <QueryClientProvider client={queryClient}>
          <React.Suspense fallback={<span>Suspended</span>}>
            <NotesGraphWorkspace
              authorityScope={authorityScope}
              isOnline
              initialFocusNoteId="common"
              selectedNoteId="common"
              hasActiveNotes
              onSelectNote={vi.fn()}
              onCreateNote={vi.fn()}
            />
            <Suspender authorityScope={authorityScope} />
          </React.Suspense>
        </QueryClientProvider>
      </React.StrictMode>
    )
    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)

    try {
      await act(async () => {
        root.render(renderScope("scope-a"))
      })
      mockCommittedSuggestionInput.mockClear()

      act(() => {
        React.startTransition(() => {
          root.render(renderScope("scope-b"))
        })
      })
      await waitFor(() => expect(suspendedBRenderCount).toBeGreaterThan(0))
      expect(mockCommittedSuggestionInput).not.toHaveBeenCalled()

      await act(async () => {
        root.render(renderScope("scope-a"))
      })
      expect(mockCommittedSuggestionInput).toHaveBeenLastCalledWith({
        authorityScope: "scope-a",
        enabled: true,
        noteId: "common"
      })

      shouldSuspendB = false
      mockCommittedSuggestionInput.mockClear()
      await act(async () => {
        React.startTransition(() => {
          root.render(renderScope("scope-b"))
        })
      })
      expect(mockCommittedSuggestionInput).toHaveBeenLastCalledWith({
        authorityScope: "scope-b",
        enabled: false,
        noteId: null
      })
    } finally {
      act(() => root.unmount())
      container.remove()
    }
  })

  it.each([
    ["tag:topic", "tag"],
    ["source:book", "source"]
  ] as const)(
    "disables suggestion calls and controls for a selected %s node",
    (nodeId, type) => {
      const state = baseWorkspaceState()
      state.graph = {
        ...graph,
        nodes: [
          ...graph.nodes,
          {
            id: nodeId,
            type,
            label: `${type} node`,
            created_at: null,
            deleted: null,
            degree: 1,
            tag_count: null,
            primary_source_id: null
          }
        ]
      }
      mockUseNotesGraphWorkspace.mockReturnValue(state)
      renderWorkspace()

      const canvasProps = mockCanvas.mock.calls.at(-1)?.[0] as {
        onSelectNode: (selectedId: string) => void
      }
      act(() => canvasProps.onSelectNode(nodeId))

      expect(state.focus).not.toHaveBeenCalled()
      expect(mockUseNotesGraphSuggestions).toHaveBeenLastCalledWith(
        expect.objectContaining({ enabled: false, noteId: null })
      )
      expect(
        screen.queryByRole("tab", { name: "Suggestions" })
      ).not.toBeInTheDocument()
      fireEvent.click(screen.getByRole("button", { name: "Edge visibility" }))
      expect(
        screen.queryByRole("checkbox", {
          name: "option:notesSearch.graphSuggestions"
        })
      ).not.toBeInTheDocument()
      expect(mockCanvas).toHaveBeenLastCalledWith(
        expect.objectContaining({
          provisionalOverlays: [],
          showProvisional: false
        })
      )
    }
  )

  it("disables suggestion calls and controls when no canonical note is selected", () => {
    const state = baseWorkspaceState()
    state.graph = { ...graph, nodes: [] }
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    renderWorkspace({ selectedNoteId: null })

    expect(mockUseNotesGraphSuggestions).toHaveBeenLastCalledWith(
      expect.objectContaining({ enabled: false, noteId: null })
    )
    expect(
      screen.queryByRole("tab", { name: "Suggestions" })
    ).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Edge visibility" }))
    expect(
      screen.queryByRole("checkbox", {
        name: "option:notesSearch.graphSuggestions"
      })
    ).not.toBeInTheDocument()
    expect(mockCanvas).toHaveBeenLastCalledWith(
      expect.objectContaining({
        provisionalOverlays: [],
        showProvisional: false
      })
    )
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

  it("fails closed without suggestion authorization and keeps the loaded graph readable", () => {
    const state = baseWorkspaceState()
    state.graph = { ...graph, suggestions_authorized: false }
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    renderWorkspace()

    expect(mockUseNotesGraphSuggestions).toHaveBeenLastCalledWith(
      expect.objectContaining({ enabled: false })
    )
    expect(screen.getByTestId("notes-graph-canvas")).toBeInTheDocument()
    expect(
      screen.queryByRole("tab", { name: "Suggestions" })
    ).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Edge visibility" }))
    expect(screen.queryByText("Suggestions")).not.toBeInTheDocument()
  })

  it.each(["acceptance", "rejection"] as const)(
    "disables both relationship decisions while %s is pending",
    (pendingMutation) => {
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
        },
        capabilities: { allowed_actions: ["accept", "reject"] },
        suggestions: [
          {
            id: "s1",
            kind: "related_note",
            target_title: "Suggested note",
            target_note_id: "target",
            match_strength: "possible",
            rationale: "Grounded",
            evidence: []
          }
        ],
        mutations: {
          acceptance: { isPending: pendingMutation === "acceptance" },
          rejection: { isPending: pendingMutation === "rejection" }
        }
      })
      renderWorkspace()
      fireEvent.click(
        screen.getByRole("button", {
          name: "option:notesSearch.graphRelationships"
        })
      )

      expect(
        screen.getByRole("button", {
          name: "notesSearch.graphAcceptSuggestion"
        })
      ).toBeDisabled()
      expect(
        screen.getByRole("button", {
          name: "notesSearch.graphRejectSuggestion"
        })
      ).toBeDisabled()
    }
  )

  it("remounts repeated relationship failure announcements and restores focus", async () => {
    const accept = vi.fn().mockRejectedValue(new Error("conflict"))
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
      },
      capabilities: { allowed_actions: ["accept", "reject"] },
      suggestions: [
        {
          id: "s1",
          kind: "related_note",
          target_title: "Suggested note",
          target_note_id: "target",
          match_strength: "possible",
          rationale: "Grounded",
          evidence: []
        }
      ],
      accept,
      reject: vi.fn(),
      mutations: {
        acceptance: { isPending: false },
        rejection: { isPending: false }
      }
    })
    renderWorkspace()
    fireEvent.click(
      screen.getByRole("button", {
        name: "option:notesSearch.graphRelationships"
      })
    )
    const acceptButton = screen.getByRole("button", {
      name: "notesSearch.graphAcceptSuggestion"
    })
    acceptButton.focus()
    fireEvent.click(acceptButton)

    await waitFor(() => expect(accept).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(acceptButton).toHaveFocus())
    const liveRegion = document.querySelector('[aria-live="polite"]')
    const firstAnnouncement = liveRegion?.firstElementChild
    expect(firstAnnouncement).not.toBeNull()
    expect(document.querySelectorAll('[aria-live="polite"]')).toHaveLength(1)
    expect(liveRegion).toHaveTextContent(
      "option:notesSearch.graphSuggestionDecisionFailed"
    )

    fireEvent.click(acceptButton)
    await waitFor(() => expect(accept).toHaveBeenCalledTimes(2))
    await waitFor(() =>
      expect(liveRegion?.firstElementChild).not.toBe(firstAnnouncement)
    )
    await waitFor(() => expect(acceptButton).toHaveFocus())
    expect(document.querySelectorAll('[aria-live="polite"]')).toHaveLength(1)
  })

  it("delivers a queued same-authority relationship focus when Canvas remounts", async () => {
    const state = baseWorkspaceState()
    state.graph = {
      ...graph,
      edges: [
        {
          id: "edge:a-b",
          source: "note:a",
          target: "note:b",
          type: "manual",
          directed: true,
          weight: null,
          label: null
        }
      ]
    }
    mockUseNotesGraphWorkspace.mockReturnValue(state)
    renderWorkspace()

    fireEvent.click(screen.getByRole("button", { name: "Beta note" }))
    expect(mockFocusNode).toHaveBeenCalledWith("note:b")
    mockFocusNode.mockClear()

    fireEvent.click(
      screen.getByRole("button", {
        name: "option:notesSearch.graphRelationships"
      })
    )
    fireEvent.click(
      within(screen.getByTestId("notes-graph-relationships-view")).getByRole(
        "button",
        { name: "Alpha note" }
      )
    )
    expect(mockFocusNode).not.toHaveBeenCalled()
    fireEvent.click(
      screen.getByRole("button", { name: "option:notesSearch.graphCanvas" })
    )

    await waitFor(() => expect(mockFocusNode).toHaveBeenCalledWith("note:a"))
    expect(mockFocusNode).toHaveBeenCalledTimes(1)
  })

  it("discards authority A queued focus before Canvas remounts for authority B", () => {
    const authorityA = workspaceStateWithCommonNote("common")
    authorityA.graph = {
      ...authorityA.graph,
      edges: [
        {
          id: "edge:common-b",
          source: "note:common",
          target: "note:b",
          type: "manual",
          directed: true,
          weight: null,
          label: null
        }
      ]
    }
    const authorityB = workspaceStateWithCommonNote(null)
    mockUseNotesGraphWorkspace.mockImplementation(({ authorityScope }) =>
      authorityScope === "scope-b" ? authorityB : authorityA
    )
    const { rerenderWorkspace } = renderWorkspace({
      authorityScope: "scope-a",
      initialFocusNoteId: "common",
      selectedNoteId: "common"
    })

    fireEvent.click(
      screen.getByRole("button", {
        name: "option:notesSearch.graphRelationships"
      })
    )
    fireEvent.click(
      within(screen.getByTestId("notes-graph-relationships-view")).getByRole(
        "button",
        { name: "Beta note" }
      )
    )
    expect(mockFocusNode).not.toHaveBeenCalled()

    rerenderWorkspace({
      authorityScope: "scope-b",
      initialFocusNoteId: "common",
      selectedNoteId: "common"
    })
    fireEvent.click(
      screen.getByRole("button", { name: "option:notesSearch.graphCanvas" })
    )

    expect(mockFocusNode).not.toHaveBeenCalled()
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
    const primaryView = screen.getByTestId("notes-graph-primary-view")
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
    expect(primaryView).toHaveClass("min-h-[420px]", "sm:min-h-[520px]")
  })
})
