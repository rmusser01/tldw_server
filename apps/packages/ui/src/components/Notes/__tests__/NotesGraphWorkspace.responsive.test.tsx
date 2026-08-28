// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen } from "@testing-library/react"
import React from "react"
import { describe, expect, it, vi } from "vitest"

import NotesGraphWorkspace from "../NotesGraphWorkspace"

const mocks = vi.hoisted(() => ({ workspace: vi.fn(), suggestions: vi.fn() }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) =>
      options?.defaultValue ??
      (
        {
          "option:notesSearch.graphCanvas": "Canvas",
          "option:notesSearch.graphRelationships": "Relationships",
          "option:notesSearch.graphViewMode": "Graph view",
          "option:notesSearch.graphDetails": "Details"
        } as Record<string, string>
      )[key] ??
      key
  })
}))
vi.mock("../hooks/useNotesGraphWorkspace", () => ({
  useNotesGraphWorkspace: mocks.workspace
}))
vi.mock("../hooks/useNotesGraphSuggestions", () => ({
  useNotesGraphSuggestions: mocks.suggestions
}))
vi.mock("../NotesGraphCanvas", async () => {
  const ReactModule = await import("react")
  return {
    default: ReactModule.forwardRef(() => <div data-testid="canvas" />)
  }
})

const renderWorkspace = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <NotesGraphWorkspace
        authorityScope="scope"
        isOnline
        initialFocusNoteId="a"
        selectedNoteId="a"
        hasActiveNotes
        onSelectNote={vi.fn()}
        onCreateNote={vi.fn()}
      />
    </QueryClientProvider>
  )

describe("NotesGraphWorkspace responsive inspector", () => {
  it("uses one loaded graph for Canvas and Relationships with an unframed responsive inspector", () => {
    const graph = {
      nodes: [
        {
          id: "note:a",
          type: "note",
          label: "A",
          created_at: null,
          deleted: false,
          degree: 0,
          tag_count: 0,
          primary_source_id: null
        }
      ],
      edges: [],
      truncated: false,
      truncated_by: [],
      has_more: false,
      cursor: null,
      limits: { max_nodes: 20, max_edges: 20, max_degree: 20 },
      radius_cap_applied: false,
      active_note_count: 1,
      all_notes_note_cap: 20,
      all_notes_eligible: true,
      suggestions_authorized: false
    }
    const workspaceState = {
      graph,
      graphQuery: { isFetching: false },
      focusNoteId: "a",
      scope: "focused",
      layout: "dagre",
      setLayout: vi.fn(),
      search: "",
      setSearch: vi.fn(),
      searchResults: [],
      visibleEdgeTypes: new Set(),
      toggleEdgeType: vi.fn(),
      allNotes: { activeNoteCount: 1, effectiveNoteCap: 20, eligible: true },
      canExpand: false,
      expand: vi.fn(),
      focus: vi.fn(),
      showAllNotes: vi.fn(),
      refresh: vi.fn(),
      isOffline: false,
      isLoading: false,
      error: null
    }
    mocks.workspace.mockReturnValue(workspaceState)
    mocks.suggestions.mockReturnValue({
      suggestions: [],
      provisionalBySuggestionId: {},
      capabilities: null,
      activeRun: null,
      lastTerminalRun: null,
      mutations: {}
    })
    renderWorkspace()

    const primary = screen.getByTestId("notes-graph-primary-view")
    const inspector = screen.getByTestId("notes-graph-inspector-region")
    expect(primary).toHaveClass("min-h-[420px]", "sm:min-h-[520px]")
    expect(inspector).toHaveClass(
      "overflow-y-auto",
      "border-t",
      "lg:border-l",
      "lg:border-t-0"
    )
    expect(inspector).not.toHaveAttribute("role", "dialog")
    expect(inspector.querySelector('[data-ui="card"]')).toBeNull()

    fireEvent.click(screen.getByRole("button", { name: "Relationships" }))
    expect(
      screen.getByTestId("notes-graph-relationships-view")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("canvas")).not.toBeInTheDocument()
    expect(workspaceState.refresh).not.toHaveBeenCalled()
    expect(
      mocks.workspace.mock.results.every(
        (result) => result.value.graph === graph
      )
    ).toBe(true)
    expect(screen.getByRole("button", { name: "Zoom in" })).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Fit graph to view" })
    ).toBeDisabled()
  })
})
