// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, renderHook } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useNotesGraphSuggestions } from "../hooks/useNotesGraphSuggestions"

const mocks = vi.hoisted(() => ({
  accept: vi.fn(),
  cancel: vi.fn(),
  createCommand: vi.fn(),
  createRun: vi.fn(),
  getCapabilities: vi.fn(),
  getRun: vi.fn(),
  listRuns: vi.fn(),
  listSuggestions: vi.fn(),
  reject: vi.fn(),
  reset: vi.fn()
}))

vi.mock("@/services/note-graph-suggestions", async () => {
  class NotesGraphSuggestionClientError extends Error {
    status: number
    code: string
    constructor(status: number, code: string, message: string) {
      super(message)
      this.status = status
      this.code = code
    }
  }
  return {
    NotesGraphSuggestionClientError,
    acceptNotesGraphSuggestion: mocks.accept,
    cancelNotesGraphSuggestionRun: mocks.cancel,
    createNotesGraphSuggestionCommand: mocks.createCommand,
    createNotesGraphSuggestionRun: mocks.createRun,
    createNotesGraphOfflineError: () =>
      new NotesGraphSuggestionClientError(
        0,
        "notes_graph_offline",
        "Notes graph changes are unavailable while offline."
      ),
    getNotesGraphSuggestionCapabilities: mocks.getCapabilities,
    getNotesGraphSuggestionRun: mocks.getRun,
    isNotesGraphCapabilitiesChangedError: (error: unknown) =>
      (error as { status?: number; code?: string })?.status === 412 &&
      (error as { code?: string })?.code === "notes_graph_capabilities_changed",
    listNotesGraphSuggestionRuns: mocks.listRuns,
    listNotesGraphSuggestions: mocks.listSuggestions,
    rejectNotesGraphSuggestion: mocks.reject,
    resetNotesGraphSuggestionRejections: mocks.reset
  }
})

const fingerprint = (value: string) => `sha256:${value.repeat(64).slice(0, 64)}`

const capability = (revision = fingerprint("a")) => ({
  provider: "provider-one",
  model: "model-one",
  endpoint_origin_revision: fingerprint("b"),
  data_boundary: "remote",
  disclosure_external: true,
  outbound_data_categories: ["selected_note_excerpt"],
  generation_available: true,
  unavailable_reason: null,
  limits: {
    max_candidates: 30,
    max_relationships: 5,
    max_tags: 5,
    max_new_tags: 2,
    max_tag_catalog: 100,
    max_estimated_input_tokens: 24000,
    max_output_tokens: 2000,
    provider_timeout_seconds: 120,
    response_candidates: 1
  },
  allowed_actions: ["generate", "accept", "reject"],
  revision,
  etag: `"${revision}"`
})

const run = (id: string, state: string, createdAt: string, revision = 1) => ({
  id,
  provider: "provider-one",
  model: "model-one",
  state,
  revision,
  created_at: createdAt,
  started_at: null,
  completed_at: null,
  suggestion_count: 0,
  related_note_count: 0,
  tag_count: 0,
  invalid_item_count: 0,
  cancellation_available: ["admitting", "queued", "running"].includes(state),
  error_code: null,
  guidance_key: null
})

const suggestion = (overrides: Record<string, unknown> = {}) => ({
  id: "suggestion-one",
  run_id: "run-new",
  kind: "related_note",
  state: "pending",
  revision: 2,
  source_note_id: "source-note",
  source_fingerprint: fingerprint("c"),
  target_note_id: "target-note",
  target_fingerprint: fingerprint("d"),
  target_title: "Target title",
  normalized_tag: null,
  display_tag: null,
  existing_tag: false,
  match_strength: "strong",
  rationale: "Related concept",
  evidence: [],
  updated_at: "2026-08-27T12:00:00Z",
  ...overrides
})

const suggestionPage = (items = [suggestion()]) => ({
  items,
  next_cursor: null,
  current_source_fingerprint: fingerprint("c"),
  rejection_set_revision: 3,
  rejection_count: 1
})

const cachedSuggestionIds = (client: QueryClient): string[] | undefined => {
  const cached = client
    .getQueryCache()
    .findAll()
    .find((query) => query.queryKey.includes("items"))?.state.data as
    | { items?: Array<{ id: string }> }
    | { data?: { items?: Array<{ id: string }> } }
    | undefined
  return "data" in (cached ?? {})
    ? (cached as { data?: { items?: Array<{ id: string }> } }).data?.items?.map(
        (item) => item.id
      )
    : (cached as { items?: Array<{ id: string }> } | undefined)?.items?.map(
        (item) => item.id
      )
}

const wrapper =
  (client: QueryClient) =>
  ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )

const flush = async () => {
  for (let step = 0; step < 5; step += 1) {
    await act(async () => {
      await Promise.resolve()
      await vi.runAllTimersAsync()
    })
  }
}

const settleQueries = async () => {
  for (let step = 0; step < 8; step += 1) {
    await act(async () => {
      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(1)
    })
  }
}

describe("useNotesGraphSuggestions", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.resetAllMocks()
    mocks.createCommand.mockImplementation((input) => ({
      ...input,
      idempotencyKey: "uuid-once"
    }))
    mocks.getCapabilities.mockResolvedValue(capability())
    mocks.getRun.mockImplementation(({ runId }) =>
      Promise.resolve(run(runId, "queued", "2026-08-27T12:00:00Z"))
    )
    mocks.listRuns.mockResolvedValue({ items: [], next_cursor: null })
    mocks.listSuggestions.mockResolvedValue(suggestionPage([]))
  })

  afterEach(() => {
    cleanup()
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
  })

  it("makes zero nested suggestion calls when the graph response is unauthorized", async () => {
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: false,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set(["note:source-note"])
        }),
      { wrapper: wrapper(client) }
    )
    await settleQueries()

    expect(mocks.getCapabilities).not.toHaveBeenCalled()
    expect(mocks.listRuns).not.toHaveBeenCalled()
    expect(mocks.getRun).not.toHaveBeenCalled()
    expect(mocks.listSuggestions).not.toHaveBeenCalled()
    expect(result.current.suggestions).toEqual([])
    expect(result.current.provisionalBySuggestionId).toEqual({})
  })

  it("clears a reconciled terminal owner, fences its stale list row, and adopts fresh recovery", async () => {
    let resolveRecoveryRuns:
      | ((page: { items: ReturnType<typeof run>[]; next_cursor: null }) => void)
      | undefined
    const recoveryRuns = new Promise<{
      items: ReturnType<typeof run>[]
      next_cursor: null
    }>((resolve) => {
      resolveRecoveryRuns = resolve
    })
    mocks.listRuns
      .mockResolvedValueOnce({
        items: [
          run("run-terminal", "succeeded", "2026-08-27T13:00:00Z"),
          {
            ...run("run-other-model", "running", "2026-08-27T14:00:00Z"),
            model: "other"
          },
          run("run-new", "running", "2026-08-27T12:00:00Z")
        ],
        next_cursor: null
      })
      .mockImplementationOnce(() => recoveryRuns)
    mocks.getRun
      .mockResolvedValueOnce(
        run("run-new", "running", "2026-08-27T12:00:00Z", 2)
      )
      .mockResolvedValueOnce(
        run("run-new", "succeeded", "2026-08-27T12:00:00Z", 3)
      )
      .mockResolvedValueOnce(
        run("run-recovered", "running", "2026-08-27T15:00:00Z", 1)
      )
    mocks.listSuggestions.mockResolvedValue(suggestionPage())
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })

    const first = renderHook(
      ({ authorityScope, enabled }) =>
        useNotesGraphSuggestions({
          authorityScope,
          enabled,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set(["note:source-note"]),
          pollIntervalMs: 1000
        }),
      {
        initialProps: { authorityScope: "authority-a", enabled: true },
        wrapper: wrapper(client)
      }
    )
    await settleQueries()

    expect(first.result.current.activeRun?.id).toBe("run-new")
    expect(mocks.getRun).toHaveBeenCalledWith(
      expect.objectContaining({ noteId: "source-note", runId: "run-new" })
    )
    expect(mocks.createRun).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    await settleQueries()

    expect(first.result.current.activeRun).toBeNull()
    expect(first.result.current.lastTerminalRun).toMatchObject({
      id: "run-new",
      state: "succeeded"
    })
    expect(mocks.getRun).toHaveBeenCalledTimes(2)
    expect(mocks.listSuggestions).toHaveBeenCalledTimes(2)
    expect(mocks.listRuns).toHaveBeenCalledTimes(2)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000)
    })
    expect(mocks.getRun).toHaveBeenCalledTimes(2)
    expect(mocks.listSuggestions).toHaveBeenCalledTimes(2)
    expect(mocks.createRun).not.toHaveBeenCalled()

    await act(async () => {
      resolveRecoveryRuns?.({
        items: [run("run-recovered", "running", "2026-08-27T15:00:00Z", 1)],
        next_cursor: null
      })
    })
    await settleQueries()

    expect(first.result.current.activeRun).toMatchObject({
      id: "run-recovered",
      state: "running"
    })
    expect(mocks.getRun).toHaveBeenCalledTimes(3)
    expect(mocks.getRun).toHaveBeenLastCalledWith(
      expect.objectContaining({ runId: "run-recovered" })
    )
    expect(mocks.createRun).not.toHaveBeenCalled()

    first.rerender({ authorityScope: "authority-b", enabled: false })
    expect(first.result.current.lastTerminalRun).toBeNull()
    first.rerender({ authorityScope: "authority-a", enabled: false })
    expect(first.result.current.lastTerminalRun).toBeNull()
  })

  it.each(["failed", "cancelled", "stale"] as const)(
    "resolves an exact %s owner without suggestion invalidation and adopts fresh recovery",
    async (terminalState) => {
      let resolveRecoveryRuns:
        | ((page: {
            items: ReturnType<typeof run>[]
            next_cursor: null
          }) => void)
        | undefined
      const recoveryRuns = new Promise<{
        items: ReturnType<typeof run>[]
        next_cursor: null
      }>((resolve) => {
        resolveRecoveryRuns = resolve
      })
      const oldRunId = `run-${terminalState}`
      const recoveredRunId = `run-${terminalState}-recovered`
      mocks.listRuns
        .mockResolvedValueOnce({
          items: [run(oldRunId, "running", "2026-08-27T12:00:00Z")],
          next_cursor: null
        })
        .mockImplementationOnce(() => recoveryRuns)
      mocks.getRun
        .mockResolvedValueOnce(
          run(oldRunId, terminalState, "2026-08-27T12:00:00Z", 2)
        )
        .mockResolvedValueOnce(
          run(recoveredRunId, "running", "2026-08-27T13:00:00Z", 1)
        )
      mocks.listSuggestions.mockResolvedValue(suggestionPage())
      const client = new QueryClient({
        defaultOptions: {
          queries: { retry: false },
          mutations: { retry: false }
        }
      })
      const { result } = renderHook(
        () =>
          useNotesGraphSuggestions({
            authorityScope: "authority-a",
            enabled: true,
            isOnline: true,
            noteId: "source-note",
            datasetId: "dataset-a",
            provider: "provider-one",
            model: "model-one",
            loadedNodeIds: new Set(),
            pollIntervalMs: 500
          }),
        { wrapper: wrapper(client) }
      )
      await settleQueries()

      expect(result.current.activeRun).toBeNull()
      expect(result.current.lastTerminalRun).toMatchObject({
        id: oldRunId,
        state: terminalState
      })
      expect(mocks.getRun).toHaveBeenCalledTimes(1)
      expect(mocks.getRun).toHaveBeenLastCalledWith(
        expect.objectContaining({
          noteId: "source-note",
          datasetId: "dataset-a",
          runId: oldRunId
        })
      )
      expect(mocks.listRuns).toHaveBeenCalledTimes(2)
      expect(mocks.listSuggestions).toHaveBeenCalledTimes(1)

      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000)
      })
      expect(mocks.getRun).toHaveBeenCalledTimes(1)

      await act(async () => {
        resolveRecoveryRuns?.({
          items: [run(recoveredRunId, "running", "2026-08-27T13:00:00Z", 1)],
          next_cursor: null
        })
      })
      await settleQueries()

      expect(result.current.activeRun).toMatchObject({
        id: recoveredRunId,
        provider: "provider-one",
        model: "model-one",
        state: "running"
      })
      expect(mocks.getRun).toHaveBeenCalledTimes(2)
      expect(mocks.getRun).toHaveBeenLastCalledWith(
        expect.objectContaining({ runId: recoveredRunId })
      )
      expect(mocks.listSuggestions).toHaveBeenCalledTimes(1)
    }
  )

  it("delegates the single 412 retry to the service while retaining one command UUID", async () => {
    const nextCapability = capability(fingerprint("e"))
    mocks.createRun.mockImplementationOnce(
      async (_command, _capability, config) => {
        config.onCapabilitiesChanged(nextCapability)
        return run("run-created", "queued", "2026-08-27T12:00:00Z")
      }
    )
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set()
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    await act(async () => {
      await result.current.generate()
    })
    await settleQueries()

    expect(mocks.createCommand).toHaveBeenCalledTimes(1)
    expect(mocks.createRun).toHaveBeenCalledTimes(1)
    expect(mocks.createRun.mock.calls[0][0].idempotencyKey).toBe("uuid-once")
    expect(mocks.createRun.mock.calls[0][1].etag).toBe(capability().etag)
    expect(mocks.getCapabilities).toHaveBeenCalledTimes(1)
    expect(result.current.capabilities?.etag).toBe(nextCapability.etag)
    expect(result.current.activeRun?.id).toBe("run-created")
  })

  it("does not coerce malformed mutation error statuses into network retries", async () => {
    mocks.createRun.mockRejectedValue({ status: "0" })
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set()
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    let failurePromise: Promise<unknown>
    act(() => {
      failurePromise = result.current.generate().then(
        () => null,
        (error) => error
      )
    })
    await act(async () => {
      await vi.runAllTimersAsync()
    })
    const failure = await failurePromise!

    expect(failure).toMatchObject({ status: "0" })
    expect(mocks.createRun).toHaveBeenCalledTimes(1)
  })

  it("revokes service-owned retry authority on an account or server switch", async () => {
    let generationConfig: { canRetry?: () => boolean } | undefined
    let rejectGeneration: ((error: unknown) => void) | undefined
    mocks.createRun.mockImplementationOnce(
      (_command, _capability, config) =>
        new Promise((_resolve, reject) => {
          generationConfig = config
          rejectGeneration = reject
        })
    )
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result, rerender } = renderHook(
      ({ authorityScope }: { authorityScope: string | null }) =>
        useNotesGraphSuggestions({
          authorityScope,
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set()
        }),
      {
        initialProps: { authorityScope: "account-a@server-a" },
        wrapper: wrapper(client)
      }
    )
    await flush()

    let failurePromise: Promise<unknown>
    act(() => {
      failurePromise = result.current.generate().then(
        () => null,
        (error) => error
      )
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(generationConfig?.canRetry?.()).toBe(true)

    rerender({ authorityScope: "account-b@server-b" })
    expect(generationConfig?.canRetry?.()).toBe(false)

    rerender({ authorityScope: "account-a@server-a" })
    expect(generationConfig?.canRetry?.()).toBe(false)

    await act(async () => {
      rejectGeneration?.({ status: 412 })
      await vi.runAllTimersAsync()
    })
    expect(await failurePromise!).toMatchObject({ status: 412 })
    expect(mocks.createRun).toHaveBeenCalledTimes(1)
  })

  it.each(["provider", "model"] as const)(
    "permanently revokes generation 412 recovery on a %s switch, including ABA",
    async (transition) => {
      let generationConfig: { canRetry?: () => boolean } | undefined
      let rejectGeneration: ((error: unknown) => void) | undefined
      mocks.getCapabilities.mockImplementation(({ provider, model }) =>
        Promise.resolve({
          ...capability(),
          provider: provider ?? "provider-one",
          model: model ?? "model-one"
        })
      )
      mocks.createRun.mockImplementationOnce(
        (_command, _capability, config) =>
          new Promise((_resolve, reject) => {
            generationConfig = config
            rejectGeneration = reject
          })
      )
      const client = new QueryClient({
        defaultOptions: {
          queries: { retry: false },
          mutations: { retry: false }
        }
      })
      const initialProps = {
        provider: "provider-one",
        model: "model-one"
      }
      const { result, rerender } = renderHook(
        ({ provider, model }) =>
          useNotesGraphSuggestions({
            authorityScope: "authority-a",
            enabled: true,
            isOnline: true,
            noteId: "source-note",
            provider,
            model,
            loadedNodeIds: new Set()
          }),
        { initialProps, wrapper: wrapper(client) }
      )
      await flush()

      let failurePromise: Promise<unknown>
      act(() => {
        failurePromise = result.current.generate().then(
          () => null,
          (error) => error
        )
      })
      await act(async () => {
        await Promise.resolve()
      })
      expect(generationConfig?.canRetry?.()).toBe(true)

      rerender({
        provider: transition === "provider" ? "provider-two" : "provider-one",
        model: transition === "model" ? "model-two" : "model-one"
      })
      expect(generationConfig?.canRetry?.()).toBe(false)
      rerender(initialProps)
      expect(generationConfig?.canRetry?.()).toBe(false)

      await act(async () => {
        rejectGeneration?.({ status: 412 })
        await vi.runAllTimersAsync()
      })
      expect(await failurePromise!).toMatchObject({ status: 412 })
      expect(mocks.createRun).toHaveBeenCalledTimes(1)
    }
  )

  it("cancels the adopted active run with one retained command key", async () => {
    mocks.listRuns.mockResolvedValue({
      items: [run("run-new", "running", "2026-08-27T12:00:00Z", 4)],
      next_cursor: null
    })
    mocks.getRun.mockResolvedValue(
      run("run-new", "running", "2026-08-27T12:00:00Z", 4)
    )
    mocks.cancel.mockResolvedValue({
      resource_id: "run-new",
      state: "cancelling",
      revision: 5
    })
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set()
        }),
      { wrapper: wrapper(client) }
    )
    await settleQueries()

    await act(async () => {
      await result.current.cancel()
    })

    expect(mocks.createCommand).toHaveBeenCalledTimes(1)
    expect(mocks.cancel).toHaveBeenCalledWith({
      noteId: "source-note",
      datasetId: undefined,
      runId: "run-new",
      expectedRevision: 4,
      idempotencyKey: "uuid-once"
    })
  })

  it("keeps provisional overlays separate and invalidates decisions without mutating offline", async () => {
    mocks.listSuggestions.mockResolvedValue(
      suggestionPage([
        suggestion(),
        suggestion({
          id: "suggestion-two",
          target_note_id: "unloaded-note",
          target_fingerprint: fingerprint("e")
        }),
        suggestion({
          id: "tag-one",
          kind: "tag",
          target_note_id: null,
          target_fingerprint: null,
          normalized_tag: "cardiology",
          display_tag: "Cardiology"
        })
      ])
    )
    mocks.accept.mockResolvedValue({
      resource_id: "suggestion-one",
      state: "accepted",
      revision: 3
    })
    mocks.cancel.mockResolvedValue({
      resource_id: "run-new",
      state: "cancelling",
      revision: 3
    })
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const invalidate = vi.spyOn(client, "invalidateQueries")
    const { result, rerender } = renderHook(
      ({ online, noteId }) =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: online,
          noteId,
          loadedNodeIds: new Set(["source-note", "target-note"])
        }),
      {
        initialProps: { online: true, noteId: "source-note" },
        wrapper: wrapper(client)
      }
    )
    await flush()

    expect(Object.keys(result.current.provisionalBySuggestionId)).toEqual([
      "suggestion-one",
      "suggestion-two"
    ])
    expect(
      result.current.provisionalBySuggestionId["suggestion-one"]
    ).toMatchObject({
      edge: {
        suggestionId: "suggestion-one",
        source: "source-note",
        target: "target-note"
      },
      node: null
    })
    expect(
      result.current.provisionalBySuggestionId["suggestion-two"]
    ).toMatchObject({
      edge: {
        suggestionId: "suggestion-two",
        source: "source-note",
        target: "suggestion-node:suggestion-two"
      },
      node: {
        id: "suggestion-node:suggestion-two",
        suggestionId: "suggestion-two",
        label: "Target title"
      }
    })
    expect(result.current.suggestions).toHaveLength(3)

    await act(async () => {
      await result.current.accept(result.current.suggestions[0])
    })
    expect(cachedSuggestionIds(client)).toEqual(["suggestion-two", "tag-one"])
    await settleQueries()
    expect(mocks.accept).toHaveBeenCalledWith(
      expect.objectContaining({
        suggestionId: "suggestion-one",
        idempotencyKey: "uuid-once"
      })
    )
    expect(
      result.current.provisionalBySuggestionId["suggestion-two"].node
    ).toMatchObject({ label: "Target title" })
    expect(invalidate).toHaveBeenCalledWith(
      expect.objectContaining({
        queryKey: ["notes-graph-workspace", "authority-a"]
      })
    )
    expect(result.current.suggestions.map((item) => item.id)).toEqual([
      "suggestion-two",
      "tag-one"
    ])
    expect(
      client
        .getMutationCache()
        .getAll()
        .every(
          (mutation) => mutation.options.mutationKey?.[1] === "authority-a"
        )
    ).toBe(true)

    rerender({ online: false, noteId: "source-note" })
    await flush()
    expect(result.current.suggestions).toHaveLength(2)
    await expect(result.current.cancel()).rejects.toMatchObject({
      code: "notes_graph_offline"
    })
    await expect(
      result.current.reject(result.current.suggestions[0])
    ).rejects.toMatchObject({
      code: "notes_graph_offline"
    })
    await expect(result.current.resetRejections()).rejects.toMatchObject({
      code: "notes_graph_offline"
    })
    expect(mocks.cancel).not.toHaveBeenCalled()
    expect(mocks.reject).not.toHaveBeenCalled()
    expect(mocks.reset).not.toHaveBeenCalled()

    const listCalls = mocks.listSuggestions.mock.calls.length
    rerender({ online: false, noteId: "other-note" })
    expect(result.current.suggestions).toEqual([])
    expect(result.current.provisionalBySuggestionId).toEqual({})
    await flush()
    expect(mocks.listSuggestions).toHaveBeenCalledTimes(listCalls)
  })

  it("omits provisional relationships whose authoritative source is not loaded", async () => {
    mocks.listSuggestions.mockResolvedValue(
      suggestionPage([
        suggestion({
          id: "loaded-source",
          source_note_id: "source-note",
          target_note_id: "target-note"
        }),
        suggestion({
          id: "unloaded-source",
          source_note_id: "other-source",
          target_note_id: "target-note"
        })
      ])
    )
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set(["note:source-note", "note:target-note"])
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    expect(result.current.provisionalBySuggestionId).toEqual({
      "loaded-source": {
        edge: {
          id: "suggestion-edge:loaded-source",
          suggestionId: "loaded-source",
          source: "note:source-note",
          target: "note:target-note",
          type: "provisional_suggestion",
          directed: false
        },
        node: null
      }
    })
  })

  it("keeps decided suggestions removed when the provider changes", async () => {
    mocks.getCapabilities.mockImplementation(({ provider, model }) =>
      Promise.resolve({
        ...capability(),
        provider: provider ?? "provider-one",
        model: model ?? "model-one"
      })
    )
    mocks.listSuggestions.mockResolvedValue(
      suggestionPage([
        suggestion(),
        suggestion({ id: "suggestion-two", target_note_id: "target-two" })
      ])
    )
    mocks.reject.mockResolvedValue({
      resource_id: "suggestion-one",
      state: "rejected",
      revision: 3,
      cleared_count: null
    })
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result, rerender } = renderHook(
      ({ provider, model }) =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          provider,
          model,
          loadedNodeIds: new Set(["source-note"])
        }),
      {
        initialProps: { provider: "provider-one", model: "model-one" },
        wrapper: wrapper(client)
      }
    )
    await flush()

    await act(async () => {
      await result.current.reject(result.current.suggestions[0])
    })
    await settleQueries()
    expect(result.current.suggestions.map((item) => item.id)).toEqual([
      "suggestion-two"
    ])
    expect(mocks.listSuggestions).toHaveBeenCalledTimes(1)

    rerender({ provider: "provider-two", model: "model-two" })
    expect(result.current.suggestions.map((item) => item.id)).toEqual([
      "suggestion-two"
    ])
    await settleQueries()
    expect(mocks.listSuggestions).toHaveBeenCalledTimes(1)
    expect(result.current.suggestions.map((item) => item.id)).toEqual([
      "suggestion-two"
    ])
  })

  it("reconciles one suggestion cache on terminal success across provider changes", async () => {
    mocks.getCapabilities.mockImplementation(({ provider, model }) =>
      Promise.resolve({
        ...capability(),
        provider: provider ?? "provider-one",
        model: model ?? "model-one"
      })
    )
    mocks.listRuns
      .mockResolvedValueOnce({
        items: [run("run-one", "running", "2026-08-27T12:00:00Z")],
        next_cursor: null
      })
      .mockResolvedValue({ items: [], next_cursor: null })
    mocks.getRun.mockResolvedValue(
      run("run-one", "succeeded", "2026-08-27T12:00:00Z", 2)
    )
    mocks.listSuggestions
      .mockResolvedValueOnce(
        suggestionPage([suggestion({ id: "before-publication" })])
      )
      .mockResolvedValueOnce(
        suggestionPage([suggestion({ id: "published-suggestion" })])
      )
      .mockResolvedValue(
        suggestionPage([suggestion({ id: "before-publication" })])
      )
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result, rerender } = renderHook(
      ({ provider, model }) =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          provider,
          model,
          loadedNodeIds: new Set(["source-note"])
        }),
      {
        initialProps: { provider: "provider-one", model: "model-one" },
        wrapper: wrapper(client)
      }
    )
    await flush()

    expect(result.current.suggestions.map((item) => item.id)).toEqual([
      "published-suggestion"
    ])
    expect(mocks.listSuggestions).toHaveBeenCalledTimes(2)

    rerender({ provider: "provider-two", model: "model-two" })
    expect(result.current.suggestions.map((item) => item.id)).toEqual([
      "published-suggestion"
    ])
    await settleQueries()
    expect(mocks.listSuggestions).toHaveBeenCalledTimes(2)
    expect(result.current.suggestions.map((item) => item.id)).toEqual([
      "published-suggestion"
    ])
  })

  it("retains one adopted provider run across a switch and reconciles its terminal publication", async () => {
    const oldProviderRun = run(
      "run-provider-one",
      "running",
      "2026-08-27T12:00:00Z"
    )
    const unrelatedRun = {
      ...run("run-provider-two", "running", "2026-08-27T13:00:00Z"),
      provider: "provider-two",
      model: "model-two"
    }
    mocks.getCapabilities.mockImplementation(({ provider, model }) =>
      Promise.resolve({
        ...capability(),
        provider: provider ?? "provider-one",
        model: model ?? "model-one"
      })
    )
    mocks.listRuns.mockImplementation(({ states: _states }) =>
      Promise.resolve({
        items:
          mocks.getCapabilities.mock.calls.at(-1)?.[0]?.provider ===
          "provider-two"
            ? [unrelatedRun]
            : [oldProviderRun],
        next_cursor: null
      })
    )
    mocks.getRun.mockResolvedValueOnce(oldProviderRun).mockResolvedValueOnce({
      ...oldProviderRun,
      state: "succeeded",
      revision: 2,
      cancellation_available: false
    })
    mocks.listSuggestions
      .mockResolvedValueOnce(
        suggestionPage([suggestion({ id: "before-publication" })])
      )
      .mockResolvedValueOnce(
        suggestionPage([suggestion({ id: "published-suggestion" })])
      )
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result, rerender } = renderHook(
      ({ provider, model }) =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          provider,
          model,
          loadedNodeIds: new Set(),
          pollIntervalMs: 500
        }),
      {
        initialProps: { provider: "provider-one", model: "model-one" },
        wrapper: wrapper(client)
      }
    )
    await settleQueries()

    expect(result.current.activeRun?.id).toBe("run-provider-one")
    expect(mocks.listRuns).toHaveBeenCalledTimes(1)

    rerender({ provider: "provider-two", model: "model-two" })
    expect(result.current.activeRun).toBeNull()
    await settleQueries()
    expect(mocks.listRuns).toHaveBeenCalledTimes(1)
    await expect(result.current.generate()).rejects.toMatchObject({
      status: 409,
      code: "notes_graph_owner_active_run_conflict"
    })
    expect(mocks.createRun).not.toHaveBeenCalled()
    await expect(result.current.cancel()).rejects.toThrow(
      "No active suggestion run"
    )
    expect(mocks.cancel).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })
    await settleQueries()
    expect(mocks.getRun).toHaveBeenCalledTimes(2)
    expect(mocks.getRun).toHaveBeenLastCalledWith(
      expect.objectContaining({ runId: "run-provider-one" })
    )
    expect(mocks.listSuggestions).toHaveBeenCalledTimes(2)
    expect(result.current.suggestions.map((item) => item.id)).toEqual([
      "published-suggestion"
    ])
    expect(result.current.activeRun).toBeNull()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000)
    })
    expect(mocks.getRun).toHaveBeenCalledTimes(2)
    expect(mocks.listSuggestions).toHaveBeenCalledTimes(2)

    rerender({ provider: "provider-one", model: "model-one" })
    await settleQueries()
    expect(result.current.activeRun).toBeNull()
    expect(mocks.getRun).toHaveBeenCalledTimes(2)
    expect(
      client
        .getQueryCache()
        .findAll()
        .filter((query) => query.queryKey.includes("run-provider-one"))
    ).toHaveLength(1)
  })

  it("exposes no suggestion, evidence, rationale, or run from another authority", async () => {
    mocks.listRuns
      .mockResolvedValueOnce({
        items: [run("run-a", "running", "2026-08-27T12:00:00Z")],
        next_cursor: null
      })
      .mockResolvedValueOnce({ items: [], next_cursor: null })
    mocks.getRun.mockResolvedValue(
      run("run-a", "running", "2026-08-27T12:00:00Z")
    )
    mocks.listSuggestions
      .mockResolvedValueOnce(
        suggestionPage([
          suggestion({
            rationale: "Account A rationale",
            evidence: [{ text: "Account A evidence" }]
          })
        ])
      )
      .mockResolvedValueOnce(suggestionPage([]))
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result, rerender } = renderHook(
      ({ authorityScope }: { authorityScope: string | null }) =>
        useNotesGraphSuggestions({
          authorityScope,
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set(["source-note"]),
          pollIntervalMs: 1000
        }),
      {
        initialProps: { authorityScope: "account-a@server-a" },
        wrapper: wrapper(client)
      }
    )
    await settleQueries()
    expect(result.current.suggestions[0].rationale).toBe("Account A rationale")
    expect(result.current.activeRun?.id).toBe("run-a")
    const callsBeforeSwitch = {
      capabilities: mocks.getCapabilities.mock.calls.length,
      runs: mocks.listRuns.mock.calls.length,
      suggestions: mocks.listSuggestions.mock.calls.length
    }

    rerender({ authorityScope: null })
    expect(result.current.capabilities).toBeNull()
    expect(result.current.activeRun).toBeNull()
    expect(result.current.lastTerminalRun).toBeNull()
    expect(result.current.suggestions).toEqual([])
    expect(result.current.provisionalBySuggestionId).toEqual({})
    await expect(result.current.generate()).rejects.toMatchObject({
      code: "notes_graph_invalid_request"
    })
    expect(mocks.getCapabilities).toHaveBeenCalledTimes(
      callsBeforeSwitch.capabilities
    )
    expect(mocks.listRuns).toHaveBeenCalledTimes(callsBeforeSwitch.runs)
    expect(mocks.listSuggestions).toHaveBeenCalledTimes(
      callsBeforeSwitch.suggestions
    )
    expect(mocks.createRun).not.toHaveBeenCalled()

    rerender({ authorityScope: "account-b@server-b" })
    expect(result.current.capabilities).toBeNull()
    expect(result.current.activeRun).toBeNull()
    expect(result.current.suggestions).toEqual([])
    await settleQueries()
    expect(result.current.suggestions).toEqual([])
    expect(
      client
        .getQueryCache()
        .findAll()
        .every((query) => query.queryKey[1] !== undefined)
    ).toBe(true)
    expect(
      client
        .getMutationCache()
        .getAll()
        .every((mutation) => mutation.options.mutationKey?.[1] !== undefined)
    ).toBe(true)
  })

  it("changes note and dataset without one-frame suggestion or current-run state", async () => {
    mocks.listRuns
      .mockResolvedValueOnce({
        items: [run("run-a", "queued", "2026-08-27T12:00:00Z")],
        next_cursor: null
      })
      .mockResolvedValueOnce({
        items: [run("run-b", "queued", "2026-08-27T12:00:00Z")],
        next_cursor: null
      })
      .mockResolvedValueOnce({
        items: [run("run-c", "queued", "2026-08-27T12:00:00Z")],
        next_cursor: null
      })
    mocks.getRun.mockImplementation(({ runId }) =>
      Promise.resolve(run(runId, "queued", "2026-08-27T12:00:00Z"))
    )
    mocks.listSuggestions
      .mockResolvedValueOnce(
        suggestionPage([suggestion({ id: "suggestion-a" })])
      )
      .mockResolvedValueOnce(
        suggestionPage([suggestion({ id: "suggestion-b" })])
      )
      .mockResolvedValueOnce(
        suggestionPage([suggestion({ id: "suggestion-c" })])
      )
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result, rerender } = renderHook(
      ({ noteId, datasetId }) =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          noteId,
          datasetId,
          loadedNodeIds: new Set()
        }),
      {
        initialProps: { noteId: "note-a", datasetId: "dataset-a" },
        wrapper: wrapper(client)
      }
    )
    await settleQueries()
    expect(result.current.suggestions[0].id).toBe("suggestion-a")
    expect(result.current.activeRun?.id).toBe("run-a")

    rerender({ noteId: "note-b", datasetId: "dataset-a" })
    expect(result.current.suggestions).toEqual([])
    expect(result.current.activeRun).toBeNull()
    await settleQueries()
    expect(result.current.suggestions[0].id).toBe("suggestion-b")
    expect(result.current.activeRun?.id).toBe("run-b")

    rerender({ noteId: "note-c", datasetId: "dataset-b" })
    expect(result.current.suggestions).toEqual([])
    expect(result.current.activeRun).toBeNull()
    await settleQueries()
    expect(result.current.suggestions[0].id).toBe("suggestion-c")
    expect(result.current.activeRun?.id).toBe("run-c")
  })

  it("does not let stale terminal reconciliation clear a newer scope owner", async () => {
    let resolveReconciliation: (() => void) | undefined
    const reconciliation = new Promise<void>((resolve) => {
      resolveReconciliation = resolve
    })
    mocks.listRuns
      .mockResolvedValueOnce({
        items: [run("run-a", "running", "2026-08-27T12:00:00Z")],
        next_cursor: null
      })
      .mockResolvedValueOnce({
        items: [run("run-b", "running", "2026-08-27T13:00:00Z")],
        next_cursor: null
      })
    mocks.getRun.mockImplementation(({ runId }) =>
      Promise.resolve(
        run(
          runId,
          runId === "run-a" ? "succeeded" : "running",
          "2026-08-27T12:00:00Z"
        )
      )
    )
    mocks.listSuggestions
      .mockResolvedValueOnce(
        suggestionPage([suggestion({ id: "suggestion-a" })])
      )
      .mockResolvedValueOnce(
        suggestionPage([suggestion({ id: "suggestion-b" })])
      )
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    vi.spyOn(client, "invalidateQueries").mockImplementationOnce(
      () => reconciliation
    )
    const { result, rerender } = renderHook(
      ({ noteId }) =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          noteId,
          loadedNodeIds: new Set()
        }),
      {
        initialProps: { noteId: "note-a" },
        wrapper: wrapper(client)
      }
    )
    await settleQueries()
    expect(result.current.activeRun).toBeNull()

    rerender({ noteId: "note-b" })
    expect(result.current.activeRun).toBeNull()
    await settleQueries()
    expect(result.current.activeRun?.id).toBe("run-b")

    await act(async () => {
      resolveReconciliation?.()
      await reconciliation
    })
    await settleQueries()
    expect(result.current.activeRun?.id).toBe("run-b")
  })

  it("does not adopt a late mutation result after an authority switch", async () => {
    let resolveAccept:
      | ((value: {
          resource_id: string
          state: string
          revision: number
        }) => void)
      | undefined
    mocks.listSuggestions
      .mockResolvedValueOnce(
        suggestionPage([suggestion({ id: "suggestion-a" })])
      )
      .mockResolvedValueOnce(
        suggestionPage([suggestion({ id: "suggestion-b" })])
      )
    mocks.accept.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveAccept = resolve
        })
    )
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const invalidate = vi.spyOn(client, "invalidateQueries")
    const { result, rerender } = renderHook(
      ({ authorityScope }) =>
        useNotesGraphSuggestions({
          authorityScope,
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set()
        }),
      {
        initialProps: { authorityScope: "authority-a" },
        wrapper: wrapper(client)
      }
    )
    await flush()

    let pendingAccept: Promise<unknown>
    act(() => {
      pendingAccept = result.current.accept(result.current.suggestions[0])
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(mocks.accept).toHaveBeenCalledTimes(1)

    rerender({ authorityScope: "authority-b" })
    expect(result.current.suggestions).toEqual([])
    await settleQueries()
    expect(result.current.suggestions.map((item) => item.id)).toEqual([
      "suggestion-b"
    ])

    await act(async () => {
      resolveAccept?.({
        resource_id: "suggestion-a",
        state: "accepted",
        revision: 3
      })
      await pendingAccept!
    })
    await settleQueries()

    expect(result.current.suggestions.map((item) => item.id)).toEqual([
      "suggestion-b"
    ])
    expect(invalidate).not.toHaveBeenCalledWith(
      expect.objectContaining({
        queryKey: ["notes-graph-workspace", "authority-b"]
      })
    )
  })

  it("clears a detail-error owner, fences its stale list row, and adopts fresh recovery", async () => {
    let resolveRecoveryRuns:
      | ((page: { items: ReturnType<typeof run>[]; next_cursor: null }) => void)
      | undefined
    const recoveryRuns = new Promise<{
      items: ReturnType<typeof run>[]
      next_cursor: null
    }>((resolve) => {
      resolveRecoveryRuns = resolve
    })
    mocks.listRuns
      .mockResolvedValueOnce({
        items: [run("run-error", "running", "2026-08-27T12:00:00Z")],
        next_cursor: null
      })
      .mockImplementationOnce(() => recoveryRuns)
    mocks.getRun
      .mockRejectedValueOnce(new Error("permission denied"))
      .mockResolvedValueOnce(
        run("run-recovered", "running", "2026-08-27T13:00:00Z", 1)
      )
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set(),
          pollIntervalMs: 500
        }),
      {
        wrapper: wrapper(client)
      }
    )
    await settleQueries()
    expect(result.current.activeRun).toBeNull()
    expect(mocks.getRun).toHaveBeenCalledTimes(1)
    expect(mocks.listRuns).toHaveBeenCalledTimes(2)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000)
    })
    expect(mocks.getRun).toHaveBeenCalledTimes(1)

    await act(async () => {
      resolveRecoveryRuns?.({
        items: [run("run-recovered", "running", "2026-08-27T13:00:00Z", 1)],
        next_cursor: null
      })
    })
    await settleQueries()

    expect(mocks.getRun).toHaveBeenCalledTimes(2)
    expect(result.current.activeRun).toMatchObject({
      id: "run-recovered",
      state: "running"
    })
  })

  it("removes a rejected suggestion from the exact scoped cache immediately", async () => {
    mocks.listSuggestions.mockResolvedValue(
      suggestionPage([
        suggestion(),
        suggestion({ id: "suggestion-two", target_note_id: "target-two" })
      ])
    )
    mocks.reject.mockResolvedValue({
      resource_id: "suggestion-one",
      state: "rejected",
      revision: 3,
      cleared_count: null
    })
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphSuggestions({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set()
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    await act(async () => {
      await result.current.reject(result.current.suggestions[0])
    })
    expect(cachedSuggestionIds(client)).toEqual(["suggestion-two"])
    await settleQueries()

    expect(result.current.suggestions.map((item) => item.id)).toEqual([
      "suggestion-two"
    ])
    expect(mocks.listSuggestions).toHaveBeenCalledTimes(1)
  })
})
