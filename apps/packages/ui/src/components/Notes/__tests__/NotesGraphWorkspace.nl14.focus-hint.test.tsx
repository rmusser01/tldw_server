/**
 * NL-14: with no note open, the graph centres on a recent note without
 * saying so. It now names that note and says how to pick another.
 */
import type { NotesGraphResponse } from "@/services/note-graph-suggestions"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, render, screen } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import NotesGraphWorkspace from "../NotesGraphWorkspace"

const { mockUseNotesGraphWorkspace } = vi.hoisted(() => ({
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
  useNotesGraphSuggestions: () => ({ provisionalBySuggestionId: {} })
}))

vi.mock("../NotesGraphCanvas", async () => {
  const ReactModule = await import("react")
  return {
    default: ReactModule.forwardRef(() => <div data-testid="notes-graph-canvas" />)
  }
})

vi.mock("../NotesGraphInspector", () => ({
  default: () => <div data-testid="notes-graph-inspector" />
}))

const graph: NotesGraphResponse = {
  nodes: [
    {
      id: "note:recent",
      type: "note" as const,
      label: "Sourdough log",
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
  limits: { max_nodes: 120, max_edges: 480, max_degree: 40 },
  radius_cap_applied: false,
  active_note_count: 1,
  all_notes_note_cap: 100,
  all_notes_eligible: true,
  manual_link_authorized: false
}

const workspaceState = (overrides: Record<string, unknown> = {}) => ({
  graph,
  graphQuery: { isFetching: false },
  focusNoteId: "recent",
  scope: "focused" as const,
  layout: "dagre" as const,
  setLayout: vi.fn(),
  search: "",
  setSearch: vi.fn(),
  searchResults: [],
  visibleEdgeTypes: new Set(["manual", "wikilink"] as const),
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
  allNotes: { activeNoteCount: 1, effectiveNoteCap: 100, eligible: true },
  canExpand: false,
  expand: vi.fn(),
  focus: vi.fn(),
  showAllNotes: vi.fn(),
  refresh: vi.fn(),
  isOffline: false,
  isLoading: false,
  error: null,
  ...overrides
})

const renderWorkspace = (selectedNoteId: string | null) =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <NotesGraphWorkspace
        authorityScope="scope"
        isOnline
        initialFocusNoteId="recent"
        selectedNoteId={selectedNoteId}
        hasActiveNotes
        onSelectNote={vi.fn()}
        onCreateNote={vi.fn()}
      />
    </QueryClientProvider>
  )

describe("NotesGraphWorkspace focus fallback hint (NL-14)", () => {
  beforeEach(() => {
    mockUseNotesGraphWorkspace.mockReturnValue(workspaceState())
  })
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("says which note the graph fell back to and how to pick another", () => {
    renderWorkspace(null)

    expect(screen.getByTestId("notes-graph-focus-fallback-hint")).toHaveTextContent(
      "No note is open, so the graph is centered on “Sourdough log”. Pick a note to focus the graph."
    )
  })

  it("says nothing when the user picked the focus note", () => {
    renderWorkspace("recent")

    expect(screen.queryByTestId("notes-graph-focus-fallback-hint")).not.toBeInTheDocument()
  })

  it("says nothing in the All notes view", () => {
    mockUseNotesGraphWorkspace.mockReturnValue(workspaceState({ scope: "all" }))
    renderWorkspace(null)

    expect(screen.queryByTestId("notes-graph-focus-fallback-hint")).not.toBeInTheDocument()
  })
})
