// @vitest-environment jsdom
import { NotesGraphSuggestionClientError } from "@/services/note-graph-suggestions"
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
    vi.clearAllMocks()
    mocks.createCommand.mockImplementation((input) => ({
      ...input,
      idempotencyKey: "uuid-once"
    }))
    mocks.getCapabilities.mockResolvedValue(capability())
    mocks.listRuns.mockResolvedValue({ items: [], next_cursor: null })
    mocks.listSuggestions.mockResolvedValue(suggestionPage([]))
  })

  afterEach(() => {
    cleanup()
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
  })

  it("adopts the newest matching nonterminal run after reload, polls it, and stops at terminal", async () => {
    mocks.listRuns.mockResolvedValue({
      items: [
        run("run-terminal", "succeeded", "2026-08-27T13:00:00Z"),
        run("run-old", "queued", "2026-08-27T10:00:00Z"),
        {
          ...run("run-other-model", "running", "2026-08-27T14:00:00Z"),
          model: "other"
        },
        run("run-new", "running", "2026-08-27T12:00:00Z")
      ],
      next_cursor: null
    })
    mocks.getRun
      .mockResolvedValueOnce(
        run("run-new", "running", "2026-08-27T12:00:00Z", 2)
      )
      .mockResolvedValueOnce(
        run("run-new", "succeeded", "2026-08-27T12:00:00Z", 3)
      )
    mocks.listSuggestions.mockResolvedValue(suggestionPage())
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })

    const first = renderHook(
      () =>
        useNotesGraphSuggestions({
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set(["note:source-note"]),
          pollIntervalMs: 1000
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    expect(first.result.current.activeRun?.id).toBe("run-new")
    expect(mocks.getRun).toHaveBeenCalledWith(
      expect.objectContaining({ noteId: "source-note", runId: "run-new" })
    )
    expect(mocks.createRun).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    await flush()

    expect(first.result.current.activeRun?.state).toBe("succeeded")
    expect(mocks.getRun).toHaveBeenCalledTimes(2)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000)
    })
    expect(mocks.getRun).toHaveBeenCalledTimes(2)
    expect(mocks.createRun).not.toHaveBeenCalled()

    first.unmount()
    const reloadedClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })
    mocks.getRun
      .mockReset()
      .mockResolvedValue(run("run-new", "succeeded", "2026-08-27T12:00:00Z", 3))
    renderHook(
      () =>
        useNotesGraphSuggestions({
          enabled: true,
          isOnline: true,
          noteId: "source-note",
          loadedNodeIds: new Set(["note:source-note"]),
          pollIntervalMs: 1000
        }),
      { wrapper: wrapper(reloadedClient) }
    )
    await flush()
    expect(mocks.createRun).not.toHaveBeenCalled()
  })

  it("retains one UUID while capability 412 refreshes disclosure and retries generation", async () => {
    const nextCapability = capability(fingerprint("e"))
    mocks.getCapabilities
      .mockResolvedValueOnce(capability())
      .mockResolvedValueOnce(nextCapability)
    mocks.createRun
      .mockRejectedValueOnce(
        new NotesGraphSuggestionClientError(
          412,
          "notes_graph_capabilities_changed",
          "Suggestion capabilities changed; refresh and retry."
        )
      )
      .mockResolvedValueOnce(
        run("run-created", "queued", "2026-08-27T12:00:00Z")
      )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphSuggestions({
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

    expect(mocks.createCommand).toHaveBeenCalledTimes(1)
    expect(mocks.createRun).toHaveBeenCalledTimes(2)
    expect(mocks.createRun.mock.calls[0][0].idempotencyKey).toBe("uuid-once")
    expect(mocks.createRun.mock.calls[1][0].idempotencyKey).toBe("uuid-once")
    expect(mocks.createRun.mock.calls[0][1].etag).toBe(capability().etag)
    expect(mocks.createRun.mock.calls[1][1].etag).toBe(nextCapability.etag)
    expect(result.current.activeRun?.id).toBe("run-created")
  })

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
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })
    const { result } = renderHook(
      () =>
        useNotesGraphSuggestions({
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
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })
    const invalidate = vi.spyOn(client, "invalidateQueries")
    const { result, rerender } = renderHook(
      ({ online, noteId }) =>
        useNotesGraphSuggestions({
          enabled: true,
          isOnline: online,
          noteId,
          loadedNodeIds: new Set(["note:source-note"])
        }),
      {
        initialProps: { online: true, noteId: "source-note" },
        wrapper: wrapper(client)
      }
    )
    await flush()

    expect(Object.keys(result.current.provisionalBySuggestionId)).toEqual([
      "suggestion-one"
    ])
    expect(
      result.current.provisionalBySuggestionId["suggestion-one"]
    ).toMatchObject({
      edge: { suggestionId: "suggestion-one" },
      node: { suggestionId: "suggestion-one", label: "Suggested note" }
    })
    expect(result.current.suggestions).toHaveLength(2)

    await act(async () => {
      await result.current.accept(result.current.suggestions[0])
    })
    expect(mocks.accept).toHaveBeenCalledWith(
      expect.objectContaining({
        suggestionId: "suggestion-one",
        idempotencyKey: "uuid-once"
      })
    )
    expect(invalidate).toHaveBeenCalledWith(
      expect.objectContaining({ queryKey: ["notes-graph-workspace"] })
    )
    expect(invalidate).toHaveBeenCalledWith(
      expect.objectContaining({
        queryKey: ["notes-graph-suggestions", "source-note"]
      })
    )

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
    await flush()
    expect(result.current.suggestions).toEqual([])
    expect(result.current.provisionalBySuggestionId).toEqual({})
    expect(mocks.listSuggestions).toHaveBeenCalledTimes(listCalls)
  })
})
