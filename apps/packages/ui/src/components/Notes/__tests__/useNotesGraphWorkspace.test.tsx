// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, renderHook } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useNotesGraphWorkspace } from "../hooks/useNotesGraphWorkspace"

const mocks = vi.hoisted(() => ({
  fetchNotesGraph: vi.fn()
}))

vi.mock("@/services/note-graph-suggestions", () => ({
  fetchNotesGraph: mocks.fetchNotesGraph
}))

const graph = (overrides: Record<string, unknown> = {}) => ({
  nodes: [
    { id: "note:a", type: "note", label: "Alpha note" },
    { id: "note:b", type: "note", label: "Beta note" },
    { id: "tag:research", type: "tag", label: "Research" }
  ],
  edges: [
    {
      id: "edge:one",
      source: "note:a",
      target: "note:b",
      type: "manual",
      directed: false,
      weight: 1,
      label: null
    }
  ],
  truncated: false,
  truncated_by: [],
  has_more: false,
  cursor: null,
  limits: { max_nodes: 120, max_edges: 480, max_degree: 40 },
  radius_cap_applied: false,
  active_note_count: 2,
  all_notes_note_cap: 100,
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

describe("useNotesGraphWorkspace", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.clearAllMocks()
  })

  afterEach(() => {
    cleanup()
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
  })

  it("loads a focused neighborhood and expands only through an explicit cursor command", async () => {
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(
        graph({
          has_more: true,
          cursor: "cursor one",
          truncated: true,
          truncated_by: ["max_nodes"]
        })
      )
      .mockResolvedValueOnce(
        graph({
          nodes: [{ id: "note:c", type: "note", label: "Gamma note" }],
          edges: [
            {
              id: "edge:two",
              source: "note:b",
              target: "note:c",
              type: "wikilink",
              directed: true,
              weight: 1,
              label: null
            }
          ],
          cursor: null,
          has_more: false
        })
      )
      .mockResolvedValueOnce(
        graph({
          nodes: [{ id: "note:d", type: "note", label: "Delta note" }]
        })
      )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphWorkspace({
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      { wrapper: wrapper(client) }
    )

    await flush()
    expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(1)
    expect(mocks.fetchNotesGraph).toHaveBeenCalledWith(
      expect.objectContaining({ centerNoteId: "note:a", cursor: undefined })
    )
    expect(result.current.graph?.nodes.map((node) => node.id)).toEqual([
      "note:a",
      "note:b",
      "tag:research"
    ])

    await act(async () => {
      await result.current.expand()
    })

    expect(mocks.fetchNotesGraph).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ centerNoteId: "note:a", cursor: "cursor one" })
    )
    expect(result.current.graph?.nodes.map((node) => node.id)).toEqual([
      "note:a",
      "note:b",
      "tag:research",
      "note:c"
    ])
    expect(result.current.canExpand).toBe(false)

    act(() => result.current.focus("note:d"))
    await flush()
    expect(mocks.fetchNotesGraph).toHaveBeenNthCalledWith(
      3,
      expect.objectContaining({ centerNoteId: "note:d", cursor: undefined })
    )
    expect(result.current.graph?.nodes[0].id).toBe("note:d")
  })

  it("derives All-notes eligibility only from authoritative count, note cap, and max_nodes", async () => {
    mocks.fetchNotesGraph.mockResolvedValueOnce(
      graph({
        active_note_count: 81,
        all_notes_note_cap: 100,
        all_notes_eligible: true,
        limits: { max_nodes: 80, max_edges: 480, max_degree: 40 }
      })
    )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphWorkspace({
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      { wrapper: wrapper(client) }
    )

    await flush()

    expect(result.current.allNotes).toEqual({
      activeNoteCount: 81,
      effectiveNoteCap: 80,
      eligible: false
    })
  })

  it("searches loaded nodes only and keeps filters and layout session-local", async () => {
    mocks.fetchNotesGraph.mockResolvedValue(graph())
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const first = renderHook(
      () =>
        useNotesGraphWorkspace({
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    act(() => {
      first.result.current.setSearch("beta")
      first.result.current.setLayout("circle")
      first.result.current.toggleEdgeType("manual")
    })

    await flush()

    expect(first.result.current.searchResults.map((node) => node.id)).toEqual([
      "note:b"
    ])
    expect(first.result.current.layout).toBe("circle")
    expect(first.result.current.visibleEdgeTypes.has("manual")).toBe(false)
    expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(1)
    first.unmount()

    const second = renderHook(
      () =>
        useNotesGraphWorkspace({
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      {
        wrapper: wrapper(
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        )
      }
    )
    await flush()

    expect(second.result.current.search).toBe("")
    expect(second.result.current.layout).toBe("dagre")
    expect(second.result.current.visibleEdgeTypes.has("manual")).toBe(true)
  })

  it("enters All-notes mode only when the authoritative response allows it", async () => {
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(graph())
      .mockResolvedValueOnce(graph({ active_note_count: 2 }))
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphWorkspace({
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    expect(result.current.showAllNotes()).toBe(true)
    await flush()
    expect(mocks.fetchNotesGraph).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ centerNoteId: undefined, cursor: undefined })
    )
    expect(result.current.scope).toBe("all")
  })

  it("preserves the last authoritative graph through refresh failure and offline transition", async () => {
    mocks.fetchNotesGraph.mockResolvedValueOnce(graph())
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const { result, rerender } = renderHook(
      ({ online, datasetId }) =>
        useNotesGraphWorkspace({
          enabled: true,
          isOnline: online,
          initialFocusNoteId: "note:a",
          datasetId
        }),
      {
        initialProps: { online: true, datasetId: "dataset-a" },
        wrapper: wrapper(client)
      }
    )
    await flush()
    expect(result.current.graph?.nodes[0].id).toBe("note:a")

    mocks.fetchNotesGraph.mockRejectedValueOnce(new Error("network failed"))
    await act(async () => {
      await result.current.refresh()
    })
    expect(result.current.graph?.nodes[0].id).toBe("note:a")

    rerender({ online: false, datasetId: "dataset-a" })
    await flush()
    expect(result.current.graph?.nodes[0].id).toBe("note:a")
    expect(result.current.isOffline).toBe(true)
    expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(2)

    rerender({ online: false, datasetId: "dataset-b" })
    await flush()
    expect(result.current.graph).toBeNull()
    expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(2)
  })
})
