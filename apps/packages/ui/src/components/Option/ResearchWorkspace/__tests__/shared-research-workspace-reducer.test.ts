import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type {
  SharedChatRequest,
  SharedWorkspaceBootstrap
} from "@/types/shared-workspace"
import {
  createInitialSharedResearchWorkspaceState,
  sharedResearchWorkspaceReducer
} from "../SharedResearchWorkspace/shared-research-workspace-reducer"
import { useSharedResearchWorkspace } from "../SharedResearchWorkspace/useSharedResearchWorkspace"

const api = vi.hoisted(() => ({
  bootstrap: vi.fn(),
  listSources: vi.fn(),
  previewSource: vi.fn(),
  listMessages: vi.fn(),
  ask: vi.fn()
}))

vi.mock("@/services/tldw/domains/shared-workspaces", () => ({
  sharedWorkspacesApi: api
}))

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

const source = (
  sourceId: string,
  retrievalReady = true
): SharedWorkspaceBootstrap["sources"]["items"][number] => ({
  source_id: sourceId,
  title: sourceId,
  source_type: "document",
  origin_url: null,
  origin_host: null,
  state: retrievalReady ? "queryable" : "processing",
  reason_code: null,
  citation_ready: retrievalReady,
  retrieval_ready: retrievalReady,
  position: 0,
  added_at: null
})

const bootstrap = (
  shareId: number,
  items = [source("source-1"), source("source-2", false)],
  generationDefault: SharedWorkspaceBootstrap["generation_default"] = {
    provider: "openai",
    model: "gpt-5-mini",
    ready: true,
    reason_code: null
  }
): SharedWorkspaceBootstrap => ({
  schema_version: 1,
  generated_at: "2026-08-22T00:00:00Z",
  share: {
    share_id: shareId,
    access_level: "view_chat",
    allow_clone: false,
    owner_display_name: "Owner",
    shared_at: null
  },
  workspace: {
    workspace_id: `workspace-${shareId}`,
    name: `Workspace ${shareId}`,
    description: ""
  },
  allowed_actions: {
    inspect_sources: { allowed: true, reason_code: null },
    ask_grounded_questions: { allowed: true, reason_code: null },
    add_sources: { allowed: false, reason_code: "recipient_read_only" },
    edit_workspace: { allowed: false, reason_code: "recipient_read_only" },
    clone_workspace: { allowed: false, reason_code: "clone_not_allowed" }
  },
  generation_default: generationDefault,
  source_summary: { total: items.length, queryable: 1, processing: 1, failed: 0 },
  sources: {
    items,
    pagination: { offset: 0, limit: 50, total: items.length, has_more: false }
  },
  conversation: { conversation_id: null, messages: [], next_before: null },
  partial_errors: []
})

const apiError = (
  code: string,
  retryable: boolean,
  retryAfterMs?: number
) => ({
  status: code === "shared_workspace_not_found" ? 404 : 422,
  detail: {
    code,
    message: code,
    retryable,
    retry_after_ms: retryAfterMs
  }
})

describe("sharedResearchWorkspaceReducer", () => {
  it("starts fail closed and selects only queryable bootstrap sources", () => {
    const initial = createInitialSharedResearchWorkspaceState(4, 1)
    expect(
      Object.values(initial.allowedActions).every((action) => !action.allowed)
    ).toBe(true)

    const state = sharedResearchWorkspaceReducer(initial, {
      type: "bootstrapSucceeded",
      generation: 1,
      bootstrap: bootstrap(4)
    })

    expect(state.selectedSourceIds).toEqual(["source-1"])
    expect(state.provider).toBe("openai")
    expect(state.model).toBe("gpt-5-mini")
  })

  it("fails closed for an internally inconsistent generation default", () => {
    const state = sharedResearchWorkspaceReducer(
      createInitialSharedResearchWorkspaceState(4, 1),
      {
        type: "bootstrapSucceeded",
        generation: 1,
        bootstrap: bootstrap(4, [source("source-1")], {
          provider: null,
          model: null,
          ready: true,
          reason_code: null
        })
      }
    )

    expect(state.provider).toBeNull()
    expect(state.model).toBeNull()
    expect(state.allowedActions.ask_grounded_questions.allowed).toBe(false)
  })

  it("reconciles removed and nonqueryable selections on source refresh", () => {
    let state = sharedResearchWorkspaceReducer(
      createInitialSharedResearchWorkspaceState(4, 1),
      { type: "bootstrapSucceeded", generation: 1, bootstrap: bootstrap(4) }
    )
    state = {
      ...state,
      selectedSourceIds: ["source-1", "source-2", "removed"]
    }

    state = sharedResearchWorkspaceReducer(state, {
      type: "sourcesSucceeded",
      generation: 1,
      page: {
        items: [source("source-1"), source("source-2", false)],
        pagination: { offset: 0, limit: 50, total: 2, has_more: false },
        summary: { total: 2, queryable: 1, processing: 1, failed: 0 },
        partial_errors: []
      }
    })

    expect(state.selectedSourceIds).toEqual(["source-1"])
  })

  it("prepends older history without duplicating message IDs", () => {
    const initial = {
      ...createInitialSharedResearchWorkspaceState(4, 1),
      messages: [
        {
          message_id: "m2",
          role: "assistant" as const,
          content: "new",
          created_at: "2026-08-22T00:00:02Z",
          citations: []
        }
      ]
    }

    const state = sharedResearchWorkspaceReducer(initial, {
      type: "historySucceeded",
      generation: 1,
      page: {
        conversation_id: "conversation-1",
        messages: [
          {
            message_id: "m1",
            role: "user",
            content: "old",
            created_at: "2026-08-22T00:00:01Z",
            citations: []
          },
          initial.messages[0]
        ],
        next_before: null
      }
    })

    expect(state.messages.map((message) => message.message_id)).toEqual([
      "m1",
      "m2"
    ])
  })

  it("ignores stale responses from an older share generation", () => {
    const state = createInitialSharedResearchWorkspaceState(5, 2)
    const next = sharedResearchWorkspaceReducer(state, {
      type: "bootstrapSucceeded",
      generation: 1,
      bootstrap: bootstrap(4)
    })

    expect(next).toBe(state)
  })
})

describe("useSharedResearchWorkspace", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("clears the previous share synchronously, aborts it, and ignores its response", async () => {
    const first = deferred<SharedWorkspaceBootstrap>()
    api.bootstrap
      .mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce(bootstrap(5))
    const { result, rerender } = renderHook(
      ({ shareId }) => useSharedResearchWorkspace(shareId),
      { initialProps: { shareId: 4 } }
    )

    await waitFor(() => expect(api.bootstrap).toHaveBeenCalledTimes(1))
    const firstSignal = api.bootstrap.mock.calls[0][1] as AbortSignal

    rerender({ shareId: 5 })

    expect(firstSignal.aborted).toBe(true)
    expect(result.current.state.bootstrap).toBeNull()
    await waitFor(() => expect(result.current.state.bootstrap?.share.share_id).toBe(5))

    first.resolve(bootstrap(4))
    await act(async () => first.promise)
    expect(result.current.state.bootstrap?.share.share_id).toBe(5)
  })

  it("keeps the draft until success and retries the immutable payload with the same UUID", async () => {
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.ask
      .mockRejectedValueOnce(new TypeError("connection reset"))
      .mockResolvedValueOnce({
        schema_version: 1,
        request_id: "request-1",
        conversation_id: "conversation-1",
        turn: {
          user_message: {
            message_id: "user-1",
            role: "user",
            content: "Question",
            created_at: "2026-08-22T00:00:01Z"
          },
          assistant_message: {
            message_id: "assistant-1",
            role: "assistant",
            content: "Answer",
            created_at: "2026-08-22T00:00:02Z"
          }
        },
        citations: [],
        generation: { provider: "openai", model: "gpt-5-mini" },
        source_scope: { mode: "include", effective_source_count: 1 },
        replay: { replayed: false }
      })
    const uuid = vi.fn().mockReturnValue("request-1")
    const { result } = renderHook(() =>
      useSharedResearchWorkspace(4, { createRequestId: uuid })
    )
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    act(() => result.current.setDraft("  Question  "))
    await act(async () => result.current.submitDraft())

    const frozen = api.ask.mock.calls[0][1] as SharedChatRequest
    expect(frozen).toEqual({
      request_id: "request-1",
      query: "Question",
      source_scope: { mode: "include", source_ids: ["source-1"] },
      provider: "openai",
      model: "gpt-5-mini"
    })
    expect(result.current.state.draft).toBe("  Question  ")

    await act(async () => result.current.retryPending())

    expect(api.ask.mock.calls[1][1]).toBe(frozen)
    expect(uuid).toHaveBeenCalledTimes(1)
    expect(result.current.state.draft).toBe("")
  })

  it("invalidates a failed receipt after edits and allocates a new UUID", async () => {
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.ask.mockRejectedValue(new TypeError("connection reset"))
    const uuid = vi.fn().mockReturnValueOnce("request-1").mockReturnValueOnce("request-2")
    const { result } = renderHook(() =>
      useSharedResearchWorkspace(4, { createRequestId: uuid })
    )
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    act(() => result.current.setDraft("Question"))
    await act(async () => result.current.submitDraft())
    expect(result.current.state.pendingSubmission?.request.request_id).toBe("request-1")

    act(() => result.current.setDraft("Edited"))
    expect(result.current.state.pendingSubmission).toBeNull()
    await act(async () => result.current.submitDraft())

    expect(api.ask.mock.calls[1][1].request_id).toBe("request-2")
  })

  it.each([
    ["source scope", (controller: ReturnType<typeof useSharedResearchWorkspace>) =>
      controller.setSelectedSourceIds(["source-1", "source-2"])],
    ["provider", (controller: ReturnType<typeof useSharedResearchWorkspace>) =>
      controller.setProvider("anthropic")],
    ["model", (controller: ReturnType<typeof useSharedResearchWorkspace>) =>
      controller.setModel("claude-test")]
  ])("invalidates an ambiguous receipt after a %s edit", async (_name, edit) => {
    api.bootstrap.mockResolvedValue(bootstrap(4, [source("source-1"), source("source-2")]))
    api.ask.mockRejectedValue(new TypeError("connection reset"))
    const { result } = renderHook(() =>
      useSharedResearchWorkspace(4, { createRequestId: () => "request-1" })
    )
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))
    act(() => {
      result.current.setSelectedSourceIds(["source-1"])
      result.current.setDraft("Question")
    })
    await act(async () => result.current.submitDraft())

    act(() => edit(result.current))

    expect(result.current.state.pendingSubmission).toBeNull()
  })

  it("does not retain a reusable UUID for a typed retryable error", async () => {
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.ask.mockRejectedValue(apiError("generation_failed", true))
    const { result } = renderHook(() =>
      useSharedResearchWorkspace(4, { createRequestId: () => "request-1" })
    )
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))
    act(() => result.current.setDraft("Question"))

    await act(async () => result.current.submitDraft())

    expect(result.current.state.pendingSubmission).toBeNull()
    expect(result.current.state.draft).toBe("Question")
  })

  it("sorts and deduplicates source IDs before freezing a submission", async () => {
    api.bootstrap.mockResolvedValue(
      bootstrap(4, [source("source-1"), source("source-2")])
    )
    api.ask.mockRejectedValue(new TypeError("connection reset"))
    const { result } = renderHook(() =>
      useSharedResearchWorkspace(4, { createRequestId: () => "request-1" })
    )
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))
    act(() => {
      result.current.setSelectedSourceIds([
        "source-2",
        "source-1",
        "source-2"
      ])
      result.current.setDraft("Question")
    })

    await act(async () => result.current.submitDraft())

    expect(api.ask.mock.calls[0][1].source_scope.source_ids).toEqual([
      "source-1",
      "source-2"
    ])
  })

  it("does not refresh sources from a stale aborted source-conflict response", async () => {
    const staleSubmission = deferred<never>()
    api.bootstrap
      .mockResolvedValueOnce(bootstrap(4))
      .mockResolvedValueOnce(bootstrap(5))
    api.ask.mockReturnValue(staleSubmission.promise)
    const { result, rerender } = renderHook(
      ({ shareId }) =>
        useSharedResearchWorkspace(shareId, {
          createRequestId: () => "request-1"
        }),
      { initialProps: { shareId: 4 } }
    )
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))
    act(() => result.current.setDraft("Question"))
    act(() => {
      void result.current.submitDraft()
    })
    await waitFor(() => expect(api.ask).toHaveBeenCalledTimes(1))

    rerender({ shareId: 5 })
    await act(async () => {
      staleSubmission.reject(apiError("shared_source_changed", false))
      await staleSubmission.promise.catch(() => undefined)
    })

    expect(api.listSources).not.toHaveBeenCalled()
    expect(result.current.state.bootstrap?.share.share_id).toBe(5)
  })

  it("aborts every active operation and clears all old share data on replacement", async () => {
    const sourceRequest = deferred<Awaited<ReturnType<typeof api.listSources>>>()
    const previewRequest = deferred<Awaited<ReturnType<typeof api.previewSource>>>()
    const historyRequest = deferred<Awaited<ReturnType<typeof api.listMessages>>>()
    const chatRequest = deferred<Awaited<ReturnType<typeof api.ask>>>()
    const initial = bootstrap(4)
    initial.conversation.next_before = "older"
    api.bootstrap.mockResolvedValueOnce(initial).mockResolvedValueOnce(bootstrap(5))
    api.listSources.mockReturnValue(sourceRequest.promise)
    api.previewSource.mockReturnValue(previewRequest.promise)
    api.listMessages.mockReturnValue(historyRequest.promise)
    api.ask.mockReturnValue(chatRequest.promise)
    const { result, rerender } = renderHook(
      ({ shareId }) =>
        useSharedResearchWorkspace(shareId, {
          createRequestId: () => "request-1"
        }),
      { initialProps: { shareId: 4 } }
    )
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))
    act(() => result.current.setDraft("Question"))
    act(() => {
      void result.current.refreshSources()
      void result.current.previewSource("source-1")
      void result.current.loadOlderHistory()
      void result.current.submitDraft()
    })
    const sourceSignal = api.listSources.mock.calls[0][2] as AbortSignal
    const previewSignal = api.previewSource.mock.calls[0][3] as AbortSignal
    const historySignal = api.listMessages.mock.calls[0][2] as AbortSignal
    const chatSignal = api.ask.mock.calls[0][2] as AbortSignal

    rerender({ shareId: 5 })

    expect(sourceSignal.aborted).toBe(true)
    expect(previewSignal.aborted).toBe(true)
    expect(historySignal.aborted).toBe(true)
    expect(chatSignal.aborted).toBe(true)
    expect(result.current.state.bootstrap).toBeNull()
    expect(result.current.state.messages).toEqual([])
    expect(result.current.state.draft).toBe("")
    expect(result.current.state.preview).toBeNull()
  })

  it("refreshes changed sources and forces a new request UUID", async () => {
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.ask
      .mockRejectedValueOnce(apiError("shared_source_changed", false))
      .mockRejectedValueOnce(new TypeError("still offline"))
    api.listSources.mockResolvedValue({
      items: [source("source-1")],
      pagination: { offset: 0, limit: 50, total: 1, has_more: false },
      summary: { total: 1, queryable: 1, processing: 0, failed: 0 },
      partial_errors: []
    })
    const uuid = vi.fn().mockReturnValueOnce("request-1").mockReturnValueOnce("request-2")
    const { result } = renderHook(() =>
      useSharedResearchWorkspace(4, { createRequestId: uuid })
    )
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    act(() => result.current.setDraft("Question"))
    await act(async () => result.current.submitDraft())

    expect(api.listSources).toHaveBeenCalledTimes(1)
    expect(result.current.state.pendingSubmission).toBeNull()
    await act(async () => result.current.submitDraft())
    expect(api.ask.mock.calls[1][1].request_id).toBe("request-2")
  })

  it("tracks a bounded rate-limit countdown and preserves context-budget drafts", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-08-22T00:00:00Z"))
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.ask
      .mockRejectedValueOnce(apiError("shared_chat_rate_limited", true, 1500))
      .mockRejectedValueOnce(apiError("shared_chat_context_too_large", false))
    const { result } = renderHook(() => useSharedResearchWorkspace(4))
    await act(async () => Promise.resolve())
    await act(async () => Promise.resolve())

    act(() => result.current.setDraft("Keep this exact draft"))
    await act(async () => result.current.submitDraft())
    expect(result.current.state.rateLimitUntil).toBe(Date.now() + 1500)
    expect(result.current.state.draft).toBe("Keep this exact draft")

    act(() => result.current.setDraft("Context draft"))
    await act(async () => result.current.submitDraft())
    expect(result.current.state.errors.submission?.code).toBe(
      "shared_chat_context_too_large"
    )
    expect(result.current.state.draft).toBe("Context draft")
    vi.useRealTimers()
  })
})
