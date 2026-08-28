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

const cachedGraphPages = (client: QueryClient) => {
  const cached = client
    .getQueryCache()
    .findAll()
    .find((query) => query.queryKey[0] === "notes-graph-workspace")?.state
    .data as
    | {
        pages?: Array<
          ReturnType<typeof graph> | { graph: ReturnType<typeof graph> }
        >
      }
    | undefined
  return (cached?.pages ?? []).map((page) =>
    "graph" in page ? page.graph : page
  )
}

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
    vi.resetAllMocks()
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
          truncated_by: ["max_nodes"],
          limits: { max_nodes: 4, max_edges: 2, max_degree: 40 }
        })
      )
      .mockResolvedValueOnce(
        graph({
          nodes: [
            { id: "note:c", type: "note", label: "Gamma note" },
            { id: "note:overflow", type: "note", label: "Overflow note" }
          ],
          edges: [
            {
              id: "edge:two",
              source: "note:b",
              target: "note:c",
              type: "wikilink",
              directed: true,
              weight: 1,
              label: null
            },
            {
              id: "edge:overflow",
              source: "note:c",
              target: "note:overflow",
              type: "manual",
              directed: false,
              weight: 1,
              label: null
            }
          ],
          cursor: null,
          has_more: false,
          limits: { max_nodes: 4, max_edges: 2, max_degree: 40 }
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
          authorityScope: "authority-a",
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
    await flush()

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
    expect(result.current.graph?.edges.map((edge) => edge.id)).toEqual([
      "edge:one",
      "edge:two"
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

  it("single-flights expansion and stops when a continuing cursor adds no progress", async () => {
    let resolveExpansion:
      | ((value: ReturnType<typeof graph>) => void)
      | undefined
    const expansionResponse = new Promise<ReturnType<typeof graph>>(
      (resolve) => {
        resolveExpansion = resolve
      }
    )
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(
        graph({
          has_more: true,
          cursor: "repeat-cursor",
          limits: { max_nodes: 10, max_edges: 10, max_degree: 40 }
        })
      )
      .mockImplementation(() => expansionResponse)
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphWorkspace({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    let firstExpansion: Promise<unknown>
    let concurrentExpansion: Promise<unknown>
    act(() => {
      firstExpansion = result.current.expand()
      concurrentExpansion = result.current.expand()
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(2)

    await act(async () => {
      resolveExpansion?.(
        graph({
          has_more: true,
          cursor: "next-cursor",
          limits: { max_nodes: 10, max_edges: 10, max_degree: 40 }
        })
      )
      await Promise.all([firstExpansion!, concurrentExpansion!])
    })
    await flush()

    expect(result.current.canExpand).toBe(false)
    expect(cachedGraphPages(client)).toHaveLength(2)
    await act(async () => {
      await result.current.expand()
    })
    expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(2)
  })

  it("stops a repeated cursor even when the page adds a new node", async () => {
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(
        graph({
          nodes: [{ id: "note:a", type: "note", label: "Alpha" }],
          edges: [],
          has_more: true,
          cursor: "same-cursor",
          limits: { max_nodes: 10, max_edges: 10, max_degree: 40 }
        })
      )
      .mockResolvedValueOnce(
        graph({
          nodes: [{ id: "note:b", type: "note", label: "Beta" }],
          edges: [],
          has_more: true,
          cursor: "same-cursor",
          limits: { max_nodes: 10, max_edges: 10, max_degree: 40 }
        })
      )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphWorkspace({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    await act(async () => {
      await result.current.expand()
    })
    await flush()

    expect(result.current.graph?.nodes.map((node) => node.id)).toEqual([
      "note:a",
      "note:b"
    ])
    expect(result.current.canExpand).toBe(false)
    await act(async () => {
      await result.current.expand()
    })
    expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(2)
  })

  it("bounds cached pages at authoritative limits and removes dangling edges", async () => {
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(
        graph({
          nodes: [
            { id: "note:a", type: "note", label: "Alpha" },
            { id: "note:b", type: "note", label: "Beta" }
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
          has_more: true,
          cursor: "page-2",
          limits: { max_nodes: 3, max_edges: 10, max_degree: 40 }
        })
      )
      .mockResolvedValueOnce(
        graph({
          nodes: [
            { id: "note:c", type: "note", label: "Gamma" },
            { id: "note:d", type: "note", label: "Delta" }
          ],
          edges: [
            {
              id: "edge:two",
              source: "note:b",
              target: "note:c",
              type: "manual",
              directed: false,
              weight: 1,
              label: null
            },
            {
              id: "edge:dangling",
              source: "note:c",
              target: "note:d",
              type: "manual",
              directed: false,
              weight: 1,
              label: null
            }
          ],
          has_more: true,
          cursor: "page-3",
          limits: { max_nodes: 3, max_edges: 10, max_degree: 40 }
        })
      )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphWorkspace({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    await act(async () => {
      await result.current.expand()
    })
    await flush()

    const pages = cachedGraphPages(client)
    const cachedNodes = pages.flatMap((page) => page.nodes)
    const cachedEdges = pages.flatMap((page) => page.edges)
    const nodeIds = new Set(cachedNodes.map((node) => node.id))
    expect(
      new Set(cachedNodes.map((node) => node.id)).size
    ).toBeLessThanOrEqual(3)
    expect(
      new Set(cachedEdges.map((edge) => edge.id)).size
    ).toBeLessThanOrEqual(2)
    expect(
      cachedEdges.every(
        (edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target)
      )
    ).toBe(true)
    expect(result.current.graph?.nodes.map((node) => node.id)).toEqual([
      "note:a",
      "note:b",
      "note:c"
    ])
    expect(result.current.graph?.edges.map((edge) => edge.id)).toEqual([
      "edge:one",
      "edge:two"
    ])
    expect(result.current.canExpand).toBe(false)
  })

  it("stops page growth when the authoritative edge limit is reached", async () => {
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(
        graph({
          nodes: [
            { id: "note:a", type: "note", label: "Alpha" },
            { id: "note:b", type: "note", label: "Beta" }
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
          has_more: true,
          cursor: "edge-page-2",
          limits: { max_nodes: 10, max_edges: 2, max_degree: 40 }
        })
      )
      .mockResolvedValueOnce(
        graph({
          nodes: [{ id: "note:c", type: "note", label: "Gamma" }],
          edges: [
            {
              id: "edge:two",
              source: "note:b",
              target: "note:c",
              type: "manual",
              directed: false,
              weight: 1,
              label: null
            }
          ],
          has_more: true,
          cursor: "edge-page-3",
          limits: { max_nodes: 10, max_edges: 2, max_degree: 40 }
        })
      )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphWorkspace({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    await act(async () => {
      await result.current.expand()
    })
    await flush()

    expect(result.current.graph?.nodes).toHaveLength(3)
    expect(result.current.graph?.edges).toHaveLength(2)
    expect(result.current.canExpand).toBe(false)
    await act(async () => {
      await result.current.expand()
    })
    expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(2)
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
          authorityScope: "authority-a",
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
          authorityScope: "authority-a",
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
          authorityScope: "authority-a",
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
          authorityScope: "authority-a",
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
          authorityScope: "authority-a",
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

  it("exposes no prior graph or request while authority is absent or switching", async () => {
    let resolveAuthorityB:
      | ((value: ReturnType<typeof graph>) => void)
      | undefined
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(
        graph({ nodes: [{ id: "note:a", type: "note", label: "Account A" }] })
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveAuthorityB = resolve
          })
      )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const { result, rerender } = renderHook(
      ({ authorityScope }: { authorityScope: string | null }) =>
        useNotesGraphWorkspace({
          authorityScope,
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      {
        initialProps: { authorityScope: "account-a@server-a" },
        wrapper: wrapper(client)
      }
    )
    await flush()
    expect(result.current.graph?.nodes[0].label).toBe("Account A")

    rerender({ authorityScope: null })
    expect(result.current.graph).toBeNull()
    expect(result.current.isLoading).toBe(false)
    expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(1)

    rerender({ authorityScope: "account-b@server-b" })
    expect(result.current.graph).toBeNull()
    expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(2)

    await act(async () => {
      resolveAuthorityB?.(
        graph({ nodes: [{ id: "note:b", type: "note", label: "Account B" }] })
      )
      await Promise.resolve()
    })
    await flush()
    expect(result.current.graph?.nodes[0].label).toBe("Account B")
    expect(
      client
        .getQueryCache()
        .findAll()
        .every((query) => query.queryKey[1] !== undefined)
    ).toBe(true)
  })

  it("changes note, dataset, and All-notes scopes without one-frame prior state", async () => {
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(graph())
      .mockResolvedValueOnce(
        graph({ nodes: [{ id: "note:b", type: "note", label: "Beta" }] })
      )
      .mockResolvedValueOnce(
        graph({ nodes: [{ id: "note:all", type: "note", label: "All" }] })
      )
      .mockResolvedValueOnce(
        graph({ nodes: [{ id: "note:c", type: "note", label: "Dataset B" }] })
      )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const { result, rerender } = renderHook(
      ({ noteId, datasetId }) =>
        useNotesGraphWorkspace({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          initialFocusNoteId: noteId,
          datasetId
        }),
      {
        initialProps: { noteId: "note:a", datasetId: "dataset-a" },
        wrapper: wrapper(client)
      }
    )
    await flush()

    act(() => result.current.focus("note:b"))
    expect(result.current.graph).toBeNull()
    await flush()
    expect(result.current.graph?.nodes[0].id).toBe("note:b")

    act(() => {
      expect(result.current.showAllNotes()).toBe(true)
    })
    expect(result.current.graph).toBeNull()
    await flush()
    expect(result.current.graph?.nodes[0].id).toBe("note:all")

    rerender({ noteId: "note:c", datasetId: "dataset-b" })
    expect(result.current.graph).toBeNull()
    await flush()
    expect(result.current.graph?.nodes[0].id).toBe("note:c")
  })

  it("bounds cursor pages and keeps a newer refresh ahead of stale expansion", async () => {
    let resolveExpansion:
      | ((value: ReturnType<typeof graph>) => void)
      | undefined
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(
        graph({
          nodes: [
            { id: "note:a", type: "note", label: "Alpha" },
            { id: "note:b", type: "note", label: "Beta" }
          ],
          edges: [],
          has_more: true,
          cursor: "page-2",
          limits: { max_nodes: 3, max_edges: 1, max_degree: 40 }
        })
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveExpansion = resolve
          })
      )
      .mockResolvedValueOnce(
        graph({
          nodes: [{ id: "note:fresh", type: "note", label: "Fresh" }],
          edges: [],
          limits: { max_nodes: 3, max_edges: 1, max_degree: 40 }
        })
      )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphWorkspace({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    let expansion: Promise<unknown>
    act(() => {
      expansion = result.current.expand()
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(mocks.fetchNotesGraph).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ cursor: "page-2" })
    )
    await act(async () => {
      await result.current.refresh()
    })
    await flush()
    expect(result.current.graph?.nodes.map((node) => node.id)).toEqual([
      "note:fresh"
    ])

    await act(async () => {
      resolveExpansion?.(
        graph({
          nodes: [
            { id: "note:c", type: "note", label: "Gamma" },
            { id: "note:d", type: "note", label: "Delta" }
          ],
          edges: [
            {
              id: "edge:two",
              source: "note:a",
              target: "note:c",
              type: "manual",
              directed: false,
              weight: 1,
              label: null
            },
            {
              id: "edge:three",
              source: "note:a",
              target: "note:d",
              type: "manual",
              directed: false,
              weight: 1,
              label: null
            }
          ],
          has_more: false,
          cursor: null,
          limits: { max_nodes: 3, max_edges: 1, max_degree: 40 }
        })
      )
      await expansion!
    })
    expect(result.current.graph?.nodes.map((node) => node.id)).toEqual([
      "note:fresh"
    ])
  })

  it("does not perform component state updates when expansion settles after unmount", async () => {
    let resolveExpansion:
      | ((value: ReturnType<typeof graph>) => void)
      | undefined
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(graph({ has_more: true, cursor: "page-2" }))
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveExpansion = resolve
          })
      )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined)
    const hook = renderHook(
      () =>
        useNotesGraphWorkspace({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      { wrapper: wrapper(client) }
    )
    await flush()
    act(() => {
      void hook.result.current.expand()
    })
    hook.unmount()

    await act(async () => {
      resolveExpansion?.(graph({ has_more: false, cursor: null }))
      await Promise.resolve()
    })

    expect(consoleError).not.toHaveBeenCalled()
    consoleError.mockRestore()
  })
})
