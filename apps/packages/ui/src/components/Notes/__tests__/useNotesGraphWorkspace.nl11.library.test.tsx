// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, renderHook } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useNotesGraphWorkspace } from "../hooks/useNotesGraphWorkspace"
import { NOTES_GRAPH_LIBRARY_REQUEST } from "../notes-graph-view"

/**
 * NL-11 (decision 15): All notes holds up to the server's cap (500), and a
 * tag filter shows part of a library that is over it. The cap comes from the
 * server alone, not from the focused graph's own node limit.
 */
const mocks = vi.hoisted(() => ({
  createSemanticManualLink: vi.fn(),
  fetchNotesGraph: vi.fn()
}))

vi.mock("@/services/note-graph-suggestions", () => ({
  NOTES_GRAPH_SEMANTIC_MAX_TOP_K: 50,
  NotesGraphSuggestionClientError: class extends Error {},
  createSemanticManualLink: mocks.createSemanticManualLink,
  fetchNotesGraph: mocks.fetchNotesGraph
}))

const graph = (overrides: Record<string, unknown> = {}) => ({
  nodes: [
    { id: "note:a", type: "note", label: "Alpha note" },
    { id: "note:b", type: "note", label: "Beta note" }
  ],
  edges: [],
  truncated: false,
  truncated_by: [],
  has_more: false,
  cursor: null,
  limits: { max_nodes: 120, max_edges: 480, max_degree: 40 },
  radius_cap_applied: false,
  active_note_count: 300,
  all_notes_note_cap: 500,
  all_notes_eligible: true,
  ...overrides
})

const wrapper =
  (client: QueryClient) =>
  ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )

const flush = async () => {
  for (let step = 0; step < 4; step += 1) {
    await act(async () => {
      await Promise.resolve()
      await vi.runAllTimersAsync()
    })
  }
}

const renderWorkspace = (radius: 1 | 2 = 1) =>
  renderHook(
    () =>
      useNotesGraphWorkspace({
        authorityScope: "authority-a",
        enabled: true,
        isOnline: true,
        initialFocusNoteId: "a",
        radius,
        maxNodes: 120,
        maxEdges: 480
      }),
    {
      wrapper: wrapper(
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      )
    }
  )

describe("useNotesGraphWorkspace library graphs (NL-11)", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.resetAllMocks()
  })

  afterEach(() => {
    cleanup()
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
  })

  it("offers All notes up to the server's cap even when the focused graph asks for fewer nodes", async () => {
    mocks.fetchNotesGraph.mockResolvedValue(graph())
    const { result } = renderWorkspace()
    await flush()

    expect(result.current.allNotes).toEqual({
      activeNoteCount: 300,
      effectiveNoteCap: 500,
      eligible: true
    })
  })

  it("asks for a whole-library budget at radius 1 when showing All notes", async () => {
    mocks.fetchNotesGraph.mockResolvedValue(graph())
    const { result } = renderWorkspace(2)
    await flush()

    act(() => {
      expect(result.current.showAllNotes()).toBe(true)
    })
    await flush()

    expect(mocks.fetchNotesGraph).toHaveBeenLastCalledWith(
      expect.objectContaining({
        centerNoteId: undefined,
        radius: 1,
        maxNodes: NOTES_GRAPH_LIBRARY_REQUEST.maxNodes,
        maxEdges: NOTES_GRAPH_LIBRARY_REQUEST.maxEdges,
        tag: undefined
      })
    )
    expect(NOTES_GRAPH_LIBRARY_REQUEST.maxNodes).toBeGreaterThanOrEqual(500)
  })

  it("asks the server which notes have links (NL-11 review)", async () => {
    mocks.fetchNotesGraph.mockResolvedValue(graph())
    renderWorkspace()
    await flush()

    expect(mocks.fetchNotesGraph).toHaveBeenLastCalledWith(
      expect.objectContaining({ includeLinkStatus: true })
    )
  })

  it("filters a library over the cap by tag", async () => {
    mocks.fetchNotesGraph.mockResolvedValue(
      graph({ active_note_count: 2_000, all_notes_eligible: false })
    )
    const { result } = renderWorkspace()
    await flush()
    expect(result.current.allNotes.eligible).toBe(false)

    act(() => result.current.filterByTag("reading"))
    await flush()

    expect(result.current.scope).toBe("all")
    expect(result.current.tagFilter).toBe("reading")
    expect(mocks.fetchNotesGraph).toHaveBeenLastCalledWith(
      expect.objectContaining({
        centerNoteId: undefined,
        radius: 1,
        maxNodes: NOTES_GRAPH_LIBRARY_REQUEST.maxNodes,
        tag: "reading"
      })
    )
  })

  it("clears the tag filter back to the focus note when All notes is over the cap", async () => {
    mocks.fetchNotesGraph.mockResolvedValue(
      graph({ active_note_count: 2_000, all_notes_eligible: false })
    )
    const { result } = renderWorkspace()
    await flush()
    act(() => result.current.filterByTag("reading"))
    await flush()

    act(() => result.current.filterByTag(null))
    await flush()

    expect(result.current.scope).toBe("focused")
    expect(result.current.tagFilter).toBeNull()
    expect(mocks.fetchNotesGraph).toHaveBeenLastCalledWith(
      expect.objectContaining({ centerNoteId: "a", tag: undefined })
    )
  })

  it("drops the tag filter when a note is focused", async () => {
    mocks.fetchNotesGraph.mockResolvedValue(graph())
    const { result } = renderWorkspace()
    await flush()
    act(() => result.current.filterByTag("reading"))
    await flush()

    act(() => result.current.focus("b"))
    await flush()

    expect(result.current.scope).toBe("focused")
    expect(result.current.tagFilter).toBeNull()
    expect(mocks.fetchNotesGraph).toHaveBeenLastCalledWith(
      expect.objectContaining({ centerNoteId: "b", tag: undefined })
    )
  })
})
