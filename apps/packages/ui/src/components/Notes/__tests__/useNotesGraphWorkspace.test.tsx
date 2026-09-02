// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, renderHook } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useNotesGraphWorkspace } from "../hooks/useNotesGraphWorkspace"

const mocks = vi.hoisted(() => {
  class ClientError extends Error {
    constructor(
      readonly status: number,
      readonly code: string
    ) {
      super(code)
    }
  }
  return {
    ClientError,
    createSemanticManualLink: vi.fn(),
    fetchNotesGraph: vi.fn()
  }
})

vi.mock("@/services/note-graph-suggestions", () => ({
  NotesGraphSuggestionClientError: mocks.ClientError,
  createSemanticManualLink: mocks.createSemanticManualLink,
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
    mocks.createSemanticManualLink.mockResolvedValue({
      status: "created",
      edge: {
        edge_id: "manual:a:b",
        from_note_id: "a",
        to_note_id: "b"
      }
    })
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

  it.each(["authority", "note", "dataset", "mode"] as const)(
    "isolates a pending expansion across a %s query-key transition",
    async (transition) => {
      let resolveA: ((value: ReturnType<typeof graph>) => void) | undefined
      let resolveB: ((value: ReturnType<typeof graph>) => void) | undefined
      const expansionA = new Promise<ReturnType<typeof graph>>((resolve) => {
        resolveA = resolve
      })
      const expansionB = new Promise<ReturnType<typeof graph>>((resolve) => {
        resolveB = resolve
      })
      mocks.fetchNotesGraph
        .mockResolvedValueOnce(
          graph({
            nodes: [{ id: "note:a", type: "note", label: "Scope A" }],
            edges: [],
            has_more: true,
            cursor: "cursor-a",
            limits: { max_nodes: 10, max_edges: 10, max_degree: 40 }
          })
        )
        .mockImplementationOnce(() => expansionA)
        .mockResolvedValueOnce(
          graph({
            nodes: [{ id: "note:b", type: "note", label: "Scope B" }],
            edges: [],
            has_more: true,
            cursor: "cursor-b",
            limits: { max_nodes: 10, max_edges: 10, max_degree: 40 }
          })
        )
        .mockImplementationOnce(() => expansionB)
      const client = new QueryClient({
        defaultOptions: { queries: { retry: false, gcTime: Infinity } }
      })
      const initialProps = {
        authorityScope: "authority-a",
        datasetId: "dataset-a",
        noteId: "note:a"
      }
      const { result, rerender } = renderHook(
        ({ authorityScope, datasetId, noteId }) =>
          useNotesGraphWorkspace({
            authorityScope,
            enabled: true,
            isOnline: true,
            initialFocusNoteId: noteId,
            datasetId
          }),
        { initialProps, wrapper: wrapper(client) }
      )
      await flush()

      let pendingA: Promise<unknown>
      act(() => {
        pendingA = result.current.expand()
      })
      await act(async () => {
        await Promise.resolve()
      })
      expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(2)

      if (transition === "mode") {
        act(() => {
          expect(result.current.showAllNotes()).toBe(true)
        })
      } else {
        rerender({
          authorityScope:
            transition === "authority" ? "authority-b" : "authority-a",
          datasetId: transition === "dataset" ? "dataset-b" : "dataset-a",
          noteId: transition === "note" ? "note:b" : "note:a"
        })
      }
      expect(result.current.graph).toBeNull()
      await flush()
      expect(result.current.graph?.nodes.map((node) => node.id)).toEqual([
        "note:b"
      ])
      if (transition === "mode") {
        expect(
          client
            .getQueryCache()
            .findAll()
            .find((query) => query.queryKey[3] === "all")?.queryKey
        ).toEqual([
          "notes-graph-workspace",
          "authority-a",
          "dataset-a",
          "all",
          "note:a",
          null,
          1,
          120,
          480,
          false,
          [
            "manual",
            "wikilink",
            "backlink",
            "tag_membership",
            "source_membership"
          ],
          null,
          null
        ])
      }

      let pendingB: Promise<unknown>
      act(() => {
        pendingB = result.current.expand()
      })
      await act(async () => {
        await Promise.resolve()
      })
      expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(4)

      await act(async () => {
        resolveB?.(
          graph({
            nodes: [
              { id: "note:b-expanded", type: "note", label: "Scope B page" }
            ],
            edges: [],
            has_more: false,
            cursor: null,
            limits: { max_nodes: 10, max_edges: 10, max_degree: 40 }
          })
        )
        const value = (await pendingB!) as {
          nodes: Array<{ id: string }>
        } | null
        expect(value?.nodes.map((node) => node.id)).toEqual([
          "note:b",
          "note:b-expanded"
        ])
      })
      await flush()

      await act(async () => {
        resolveA?.(
          graph({
            nodes: [
              { id: "note:a-expanded", type: "note", label: "Scope A page" }
            ],
            edges: [],
            has_more: false,
            cursor: null,
            limits: { max_nodes: 10, max_edges: 10, max_degree: 40 }
          })
        )
        await pendingA!
      })
      await flush()
      const scopeAData = client.getQueryData<{
        pages: Array<{ nodes: Array<{ id: string }> }>
      }>([
        "notes-graph-workspace",
        "authority-a",
        "dataset-a",
        "focused",
        "note:a",
        "note:a",
        1,
        120,
        480,
        false,
        [
          "manual",
          "wikilink",
          "backlink",
          "tag_membership",
          "source_membership"
        ],
        null,
        null
      ])
      expect(
        scopeAData?.pages.flatMap((page) => page.nodes.map((node) => node.id))
      ).toEqual(["note:a", "note:a-expanded"])
      expect(result.current.graph?.nodes.map((node) => node.id)).toEqual([
        "note:b",
        "note:b-expanded"
      ])
    }
  )

  it("reuses an exact-key expansion after another scope starts expanding", async () => {
    let resolveA: ((value: ReturnType<typeof graph>) => void) | undefined
    let resolveB: ((value: ReturnType<typeof graph>) => void) | undefined
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(
        graph({
          nodes: [{ id: "note:a", type: "note", label: "Scope A" }],
          edges: [],
          has_more: true,
          cursor: "cursor-a"
        })
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveA = resolve
          })
      )
      .mockResolvedValueOnce(
        graph({
          nodes: [{ id: "note:b", type: "note", label: "Scope B" }],
          edges: [],
          has_more: true,
          cursor: "cursor-b"
        })
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveB = resolve
          })
      )
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false, gcTime: Infinity, staleTime: Infinity }
      }
    })
    const { result, rerender } = renderHook(
      ({ authorityScope }) =>
        useNotesGraphWorkspace({
          authorityScope,
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a"
        }),
      {
        initialProps: { authorityScope: "authority-a" },
        wrapper: wrapper(client)
      }
    )
    await flush()

    const pendingA = result.current.expand()
    await act(async () => {
      await Promise.resolve()
    })
    rerender({ authorityScope: "authority-b" })
    await flush()
    const pendingB = result.current.expand()
    await act(async () => {
      await Promise.resolve()
    })

    rerender({ authorityScope: "authority-a" })
    await flush()
    expect(result.current.expand()).toBe(pendingA)
    expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(4)

    await act(async () => {
      resolveA?.(
        graph({
          nodes: [{ id: "note:a-page", type: "note", label: "A page" }],
          edges: [],
          has_more: false,
          cursor: null
        })
      )
      await pendingA
      rerender({ authorityScope: "authority-b" })
      resolveB?.(
        graph({
          nodes: [{ id: "note:b-page", type: "note", label: "B page" }],
          edges: [],
          has_more: false,
          cursor: null
        })
      )
      await pendingB
    })
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

  it("keeps semantic off by default and binds the complete edge set and controls into query identity", async () => {
    mocks.fetchNotesGraph.mockResolvedValue(graph())
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: Infinity } }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphWorkspace({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a",
          datasetId: "dataset-a"
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    const ordinaryEdgeTypes = [
      "manual",
      "wikilink",
      "backlink",
      "tag_membership",
      "source_membership"
    ]
    expect(result.current.semantic.enabled).toBe(false)
    expect(mocks.fetchNotesGraph).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        edgeTypes: ordinaryEdgeTypes,
        semanticThreshold: undefined,
        semanticTopK: undefined
      })
    )

    act(() => result.current.semantic.setEnabled(true))
    await flush()

    expect(mocks.fetchNotesGraph).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        edgeTypes: [...ordinaryEdgeTypes, "semantic"],
        semanticThreshold: 0.75,
        semanticTopK: 10
      })
    )
    expect(
      client
        .getQueryCache()
        .findAll()
        .some(
          (query) =>
            JSON.stringify(query.queryKey) ===
            JSON.stringify([
              "notes-graph-workspace",
              "authority-a",
              "dataset-a",
              "focused",
              "note:a",
              "note:a",
              1,
              120,
              480,
              true,
              [...ordinaryEdgeTypes, "semantic"],
              0.75,
              10
            ])
        )
    ).toBe(true)
  })

  it("preserves semantic cursor bindings while merging ordinary continuation pages", async () => {
    const semanticEdge = {
      id: "semantic:one",
      source: "note:a",
      target: "note:b",
      type: "semantic",
      directed: false,
      weight: 0.9,
      label: null
    }
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(graph())
      .mockResolvedValueOnce(
        graph({
          edges: [semanticEdge],
          has_more: true,
          cursor: "ordinary-page-2",
          limits: { max_nodes: 5, max_edges: 5, max_degree: 40 }
        })
      )
      .mockResolvedValueOnce(
        graph({
          nodes: [
            { id: "note:a", type: "note", label: "Alpha note" },
            { id: "note:b", type: "note", label: "Beta note" },
            { id: "note:c", type: "note", label: "Gamma note" }
          ],
          edges: [
            {
              id: "edge:ordinary-two",
              source: "note:b",
              target: "note:c",
              type: "wikilink",
              directed: true,
              weight: 1,
              label: null
            }
          ],
          has_more: false,
          cursor: null,
          limits: { max_nodes: 5, max_edges: 5, max_degree: 40 }
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
    act(() => result.current.semantic.setEnabled(true))
    await flush()

    await act(async () => {
      await result.current.expand()
    })
    await flush()

    expect(mocks.fetchNotesGraph).toHaveBeenNthCalledWith(
      3,
      expect.objectContaining({
        cursor: "ordinary-page-2",
        edgeTypes: [
          "manual",
          "wikilink",
          "backlink",
          "tag_membership",
          "source_membership"
        ],
        semanticThreshold: undefined,
        semanticTopK: undefined
      })
    )
    expect(result.current.graph?.nodes.map((node) => node.id)).toEqual([
      "note:a",
      "note:b",
      "tag:research",
      "note:c"
    ])
    expect(result.current.graph?.edges.map((edge) => edge.id)).toEqual([
      "semantic:one",
      "edge:ordinary-two"
    ])
  })

  it("preserves the last ordinary graph offline when a semantic request has no result", async () => {
    let resolveSemantic: ((value: ReturnType<typeof graph>) => void) | undefined
    mocks.fetchNotesGraph.mockResolvedValueOnce(graph()).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveSemantic = resolve
        })
    )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const { result, rerender } = renderHook(
      ({ isOnline }) =>
        useNotesGraphWorkspace({
          authorityScope: "authority-a",
          enabled: true,
          isOnline,
          initialFocusNoteId: "note:a"
        }),
      {
        initialProps: { isOnline: true },
        wrapper: wrapper(client)
      }
    )
    await flush()
    expect(result.current.graph?.nodes[0].id).toBe("note:a")

    act(() => result.current.semantic.setEnabled(true))
    await act(async () => Promise.resolve())
    rerender({ isOnline: false })

    expect(result.current.isOffline).toBe(true)
    expect(result.current.graph?.nodes[0].id).toBe("note:a")
    await act(async () => {
      resolveSemantic?.(graph())
    })
  })

  it("preserves Similar content in All notes but requests only ordinary first-page edges", async () => {
    mocks.fetchNotesGraph.mockResolvedValue(graph())
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
    act(() => result.current.semantic.setEnabled(true))
    await flush()
    act(() => {
      result.current.showAllNotes()
    })
    await flush()

    expect(result.current.semantic.enabled).toBe(true)
    expect(result.current.semantic.focusRequired).toBe(true)
    expect(mocks.fetchNotesGraph).toHaveBeenLastCalledWith({
      centerNoteId: undefined,
      datasetId: undefined,
      radius: 1,
      edgeTypes: [
        "manual",
        "wikilink",
        "backlink",
        "tag_membership",
        "source_membership"
      ],
      maxNodes: 120,
      maxEdges: 480,
      cursor: undefined,
      semanticThreshold: undefined,
      semanticTopK: undefined
    })
  })

  it("retains All notes without refetching when only semantic preferences change", async () => {
    const allNotesGraph = graph({
      nodes: [{ id: "note:all", type: "note", label: "All notes result" }],
      edges: []
    })
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(graph())
      .mockResolvedValueOnce(graph())
      .mockResolvedValueOnce(allNotesGraph)
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
    act(() => result.current.semantic.setEnabled(true))
    await flush()
    act(() => {
      result.current.showAllNotes()
    })
    await flush()
    expect(result.current.graph?.nodes[0].id).toBe("note:all")

    act(() => result.current.semantic.setThreshold(0.85))
    await act(async () => Promise.resolve())

    expect(result.current.semantic.enabled).toBe(true)
    expect(result.current.semantic.focusRequired).toBe(true)
    expect(result.current.graph?.nodes[0].id).toBe("note:all")
    expect(mocks.fetchNotesGraph).toHaveBeenCalledTimes(3)
  })

  it("creates a canonical manual link and invalidates the graph", async () => {
    const semanticEdge = {
      id: "semantic:a:b",
      source: "note:a",
      target: "note:b",
      type: "semantic",
      directed: false,
      weight: 0.9,
      label: null,
      evidence: { generation_id: "generation-a" }
    }
    mocks.fetchNotesGraph.mockResolvedValue(
      graph({ manual_link_authorized: true, edges: [semanticEdge] })
    )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const invalidate = vi
      .spyOn(client, "invalidateQueries")
      .mockResolvedValue(undefined)
    const { result } = renderHook(
      () =>
        useNotesGraphWorkspace({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          initialFocusNoteId: "note:a",
          datasetId: "dataset-a"
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    await expect(
      result.current.createManualLink(semanticEdge as never)
    ).resolves.toBe(true)
    expect(mocks.createSemanticManualLink).toHaveBeenCalledWith({
      sourceNoteId: "a",
      targetNoteId: "b",
      datasetId: "dataset-a",
      generationId: "generation-a",
      idempotencyKey: expect.any(String)
    })
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: ["notes-graph-workspace"]
    })
  })

  it("does not report stale semantic conversion conflicts as success", async () => {
    const semanticEdge = {
      id: "semantic:a:b",
      source: "note:a",
      target: "note:b",
      type: "semantic",
      directed: false,
      weight: 0.9,
      label: null,
      evidence: { generation_id: "generation-a" }
    }
    const conflict = new mocks.ClientError(
      409,
      "notes_semantic_conversion_generation_stale"
    )
    mocks.createSemanticManualLink.mockRejectedValue(conflict)
    mocks.fetchNotesGraph.mockResolvedValue(
      graph({ manual_link_authorized: true, edges: [semanticEdge] })
    )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const invalidate = vi
      .spyOn(client, "invalidateQueries")
      .mockResolvedValue(undefined)
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

    await expect(
      result.current.createManualLink(semanticEdge as never)
    ).rejects.toBe(conflict)
    expect(invalidate).not.toHaveBeenCalled()
  })

  it("does not expose a completed semantic graph while changed threshold controls are unresolved", async () => {
    const semanticEdge = {
      id: "semantic:stale-threshold",
      source: "note:a",
      target: "note:b",
      type: "semantic",
      directed: false,
      weight: 0.8,
      label: null
    }
    let rejectChangedThreshold: ((reason?: unknown) => void) | undefined
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(graph())
      .mockResolvedValueOnce(graph({ edges: [graph().edges[0], semanticEdge] }))
      .mockImplementationOnce(
        () =>
          new Promise((_resolve, reject) => {
            rejectChangedThreshold = reject
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
    act(() => result.current.semantic.setEnabled(true))
    await flush()
    expect(result.current.graph?.edges.map((edge) => edge.id)).toContain(
      "semantic:stale-threshold"
    )

    act(() => result.current.semantic.setThreshold(0.85))
    await act(async () => Promise.resolve())

    expect(result.current.semantic.threshold).toBe(0.85)
    expect(mocks.fetchNotesGraph).toHaveBeenNthCalledWith(
      3,
      expect.objectContaining({ cursor: undefined, semanticThreshold: 0.85 })
    )
    expect(result.current.graph?.edges.map((edge) => edge.id)).toEqual([
      "edge:one"
    ])
    await act(async () => {
      rejectChangedThreshold?.(new Error("threshold request failed"))
      await Promise.resolve()
    })
    expect(result.current.graph?.edges.map((edge) => edge.id)).toEqual([
      "edge:one"
    ])
  })

  it("removes semantic fallback edges immediately when the edge toggle is turned off", async () => {
    const semanticEdge = {
      id: "semantic:toggle-off",
      source: "note:a",
      target: "note:b",
      type: "semantic",
      directed: false,
      weight: 0.9,
      label: null
    }
    let resolveOrdinary: ((value: ReturnType<typeof graph>) => void) | undefined
    mocks.fetchNotesGraph
      .mockResolvedValueOnce(graph())
      .mockResolvedValueOnce(graph({ edges: [graph().edges[0], semanticEdge] }))
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveOrdinary = resolve
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
    act(() => result.current.semantic.setEnabled(true))
    await flush()
    expect(result.current.graph?.edges.map((edge) => edge.id)).toContain(
      "semantic:toggle-off"
    )

    act(() => result.current.semantic.setEnabled(false))
    await act(async () => Promise.resolve())

    expect(result.current.graph?.edges.map((edge) => edge.id)).toEqual([
      "edge:one"
    ])
    await act(async () => {
      resolveOrdinary?.(graph())
      await Promise.resolve()
    })
  })
})
