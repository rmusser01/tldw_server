import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SharedWorkspacePostCommitResponseError } from "@/services/tldw/domains/shared-workspaces"
import type {
  SharedChatRequest,
  SharedChatResponse,
  SharedMessagePage,
  SharedSourcePage,
  SharedSourcePreview,
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

vi.mock(
  "@/services/tldw/domains/shared-workspaces",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("@/services/tldw/domains/shared-workspaces")
      >()
    return { ...actual, sharedWorkspacesApi: api }
  }
)

const deferred = <T>() => {
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
): SharedWorkspaceBootstrap => {
  const queryable = items.filter((item) => item.retrieval_ready).length
  const processing = items.filter((item) => item.state === "processing").length
  const failed = items.filter((item) => item.state === "failed").length
  return {
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
    source_summary: { total: items.length, queryable, processing, failed },
    sources: {
      items,
      pagination: { offset: 0, limit: 50, total: items.length, has_more: false }
    },
    conversation: { conversation_id: null, messages: [], next_before: null },
    partial_errors: []
  }
}

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

const sourcePage = (
  items: SharedSourcePage["items"],
  pagination: SharedSourcePage["pagination"] = {
    offset: 0,
    limit: 50,
    total: items.length,
    has_more: false
  },
  summary: SharedSourcePage["summary"] = {
    total: pagination.total,
    queryable: items.filter((item) => item.retrieval_ready).length,
    processing: items.filter((item) => item.state === "processing").length,
    failed: items.filter((item) => item.state === "failed").length
  }
): SharedSourcePage => ({
  items,
  pagination,
  summary,
  partial_errors: []
})

const chatResponse = (
  requestId: string,
  suffix = "1",
  mode: "all" | "include" = "include"
): SharedChatResponse => ({
  schema_version: 1,
  request_id: requestId,
  conversation_id: "conversation-1",
  turn: {
    user_message: {
      message_id: `user-${suffix}`,
      role: "user",
      content: `Question ${suffix}`,
      created_at: "2026-08-22T00:00:01Z"
    },
    assistant_message: {
      message_id: `assistant-${suffix}`,
      role: "assistant",
      content: `Answer ${suffix}`,
      created_at: "2026-08-22T00:00:02Z"
    }
  },
  citations: [],
  generation: { provider: "openai", model: "gpt-5-mini" },
  source_scope: { mode, effective_source_count: 1 },
  replay: { replayed: false }
})

const historyPage = (messageId: string): SharedMessagePage => ({
  conversation_id: "conversation-1",
  messages: [
    {
      message_id: messageId,
      role: "user",
      content: messageId,
      created_at: "2026-08-22T00:00:00Z",
      citations: []
    }
  ],
  next_before: null
})

const preview = (sourceId: string): SharedSourcePreview => ({
  source_id: sourceId,
  title: sourceId,
  source_type: "document",
  origin_url: null,
  origin_host: null,
  state: "queryable",
  reason_code: null,
  content_available: true,
  preview_mode: "content_excerpt",
  unavailable_reason: null,
  text_preview: sourceId,
  text_total_chars: sourceId.length,
  text_truncated: false,
  snippets: [],
  generated_at: "2026-08-22T00:00:00Z"
})

describe("sharedResearchWorkspaceReducer", () => {
  it("clears old preview evidence and exposes the next target at preview start", () => {
    const loaded = sharedResearchWorkspaceReducer(
      createInitialSharedResearchWorkspaceState(4, 1),
      {
        type: "previewSucceeded",
        generation: 1,
        preview: preview("source-old")
      }
    )

    const loading = sharedResearchWorkspaceReducer(loaded, {
      type: "previewStarted",
      generation: 1,
      sourceId: "source-new",
      chunkIndex: 7
    })

    expect(loading.preview).toBeNull()
    expect(loading.previewLoading).toBe(true)
    expect(loading.previewTarget).toEqual({
      sourceId: "source-new",
      chunkIndex: 7
    })
  })
  it("starts fail closed and represents the bootstrap selection as all queryable sources", () => {
    const initial = createInitialSharedResearchWorkspaceState(4, 1)
    expect(
      Object.values(initial.allowedActions).every((action) => !action.allowed)
    ).toBe(true)

    const state = sharedResearchWorkspaceReducer(initial, {
      type: "bootstrapSucceeded",
      generation: 1,
      bootstrap: bootstrap(4)
    })

    expect(state.sourceScopeMode).toBe("all")
    expect(state.selectedSourceIds).toEqual([])
    expect(state.lastCompletedAssistantMessageId).toBeNull()
    expect(state.provider).toBe("openai")
    expect(state.model).toBe("gpt-5-mini")
  })

  it("rejects an inconsistent generation default without rewriting server actions", () => {
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
    expect(state.allowedActions.ask_grounded_questions).toEqual({
      allowed: true,
      reason_code: null
    })
  })

  it("reconciles removed and nonqueryable selections from a complete source refresh", () => {
    let state = sharedResearchWorkspaceReducer(
      createInitialSharedResearchWorkspaceState(4, 1),
      { type: "bootstrapSucceeded", generation: 1, bootstrap: bootstrap(4) }
    )
    state = {
      ...state,
      sourceScopeMode: "include",
      selectedSourceIds: ["source-1", "source-2", "removed"]
    }

    state = sharedResearchWorkspaceReducer(state, {
      type: "sourcesSucceeded",
      generation: 1,
      query: { offset: 0, limit: 50 },
      page: sourcePage([source("source-1"), source("source-2", false)])
    })

    expect(state.selectedSourceIds).toEqual(["source-1"])
  })

  it("keeps include selections across filtered and paginated source pages", () => {
    let state = sharedResearchWorkspaceReducer(
      createInitialSharedResearchWorkspaceState(4, 1),
      { type: "bootstrapSucceeded", generation: 1, bootstrap: bootstrap(4) }
    )
    state = sharedResearchWorkspaceReducer(state, {
      type: "selectedSourcesChanged",
      sourceIds: ["source-1", "off-page", "not-ready"]
    })

    state = sharedResearchWorkspaceReducer(state, {
      type: "sourcesSucceeded",
      generation: 1,
      query: { offset: 50, limit: 50, q: "filtered" },
      page: sourcePage(
        [source("source-1"), source("not-ready", false)],
        { offset: 50, limit: 50, total: 120, has_more: true }
      )
    })

    expect(state.sourceScopeMode).toBe("include")
    expect(state.selectedSourceIds).toEqual(["source-1", "off-page"])
  })

  it("keeps all-mode selection implicit across source pages", () => {
    let state = sharedResearchWorkspaceReducer(
      createInitialSharedResearchWorkspaceState(4, 1),
      { type: "bootstrapSucceeded", generation: 1, bootstrap: bootstrap(4) }
    )

    state = sharedResearchWorkspaceReducer(state, {
      type: "sourcesSucceeded",
      generation: 1,
      query: { offset: 50, limit: 50 },
      page: sourcePage([source("source-51")], {
        offset: 50,
        limit: 50,
        total: 75,
        has_more: false
      })
    })

    expect(state.sourceScopeMode).toBe("all")
    expect(state.selectedSourceIds).toEqual([])
  })

  it("records only the exact assistant message completed by a submission", () => {
    let state = sharedResearchWorkspaceReducer(createInitialSharedResearchWorkspaceState(4, 1), {
      type: "bootstrapSucceeded",
      generation: 1,
      bootstrap: bootstrap(4)
    })
    const request: SharedChatRequest = {
      request_id: "request-1",
      query: "Question",
      source_scope: { mode: "all", source_ids: [] },
      provider: "openai",
      model: "gpt-5-mini"
    }
    state = sharedResearchWorkspaceReducer(state, {
      type: "submissionStarted",
      generation: 1,
      request,
      submittedDraft: "Question",
      draftRevision: 0
    })

    state = sharedResearchWorkspaceReducer(state, {
      type: "submissionSucceeded",
      generation: 1,
      response: chatResponse("request-1", "exact", "all")
    })

    expect(state.lastCompletedAssistantMessageId).toBe("assistant-exact")
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
    vi.resetAllMocks()
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
      .mockResolvedValueOnce(chatResponse("request-1", "1", "all"))
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
      source_scope: { mode: "all", source_ids: [] },
      provider: "openai",
      model: "gpt-5-mini"
    })
    expect(result.current.state.draft).toBe("  Question  ")

    await act(async () => result.current.retryPending())

    expect(api.ask.mock.calls[1][1]).toBe(frozen)
    expect(uuid).toHaveBeenCalledTimes(1)
    expect(result.current.state.draft).toBe("")
  })

  it("materializes every unfiltered queryable source before deselecting from all mode", async () => {
    const items = Array.from({ length: 75 }, (_, index) => source(`source-${index + 1}`))
    const initial = bootstrap(4, items.slice(0, 50))
    initial.source_summary = {
      total: 75,
      queryable: 75,
      processing: 0,
      failed: 0
    }
    initial.sources.pagination = {
      offset: 0,
      limit: 50,
      total: 75,
      has_more: true
    }
    const summary = initial.source_summary
    api.bootstrap.mockResolvedValue(initial)
    api.listSources
      .mockResolvedValueOnce(
        sourcePage(items.slice(0, 50), { offset: 0, limit: 50, total: 75, has_more: true }, summary)
      )
      .mockResolvedValueOnce(
        sourcePage(items.slice(50), { offset: 50, limit: 50, total: 75, has_more: false }, summary)
      )
    const { result } = renderHook(() => useSharedResearchWorkspace(4))
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))
    act(() =>
      result.current.setSourceQuery({
        offset: 0,
        limit: 50,
        q: "visible-filter"
      })
    )

    await act(async () => result.current.toggleSource(items[0], false))

    expect(api.listSources.mock.calls.map((call) => call[1])).toEqual([
      { offset: 0, limit: 200 },
      { offset: 50, limit: 200 }
    ])
    expect(result.current.state.sourceQuery).toEqual({
      offset: 0,
      limit: 50,
      q: "visible-filter"
    })
    expect(result.current.state.sourceScopeMode).toBe("include")
    expect(result.current.state.selectedSourceIds).toHaveLength(74)
    expect(result.current.state.selectedSourceIds).not.toContain("source-1")
    expect(result.current.state.selectedSourceIds).toContain("source-75")
    expect(result.current.state.errors.selection).toBeNull()
  })

  it("leaves all mode unchanged when source materialization fails between pages", async () => {
    const items = Array.from({ length: 75 }, (_, index) => source(`source-${index + 1}`))
    const initial = bootstrap(4, items.slice(0, 50))
    initial.source_summary = {
      total: 75,
      queryable: 75,
      processing: 0,
      failed: 0
    }
    initial.sources.pagination = {
      offset: 0,
      limit: 50,
      total: 75,
      has_more: true
    }
    api.bootstrap.mockResolvedValue(initial)
    api.listSources
      .mockResolvedValueOnce(
        sourcePage(
          items.slice(0, 50),
          { offset: 0, limit: 50, total: 75, has_more: true },
          initial.source_summary
        )
      )
      .mockRejectedValueOnce(new TypeError("connection reset"))
    const { result } = renderHook(() => useSharedResearchWorkspace(4))
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    await act(async () => result.current.toggleSource(items[0], false))

    expect(result.current.state.sourceScopeMode).toBe("all")
    expect(result.current.state.selectedSourceIds).toEqual([])
    expect(result.current.state.selectionMaterializing).toBe(false)
    expect(result.current.state.errors.selection?.code).toBe("shared_source_selection_unavailable")
  })

  it("leaves all mode unchanged when paginated materialization is inconsistent", async () => {
    const items = Array.from({ length: 75 }, (_, index) => source(`source-${index + 1}`))
    const initial = bootstrap(4, items.slice(0, 50))
    initial.source_summary = {
      total: 75,
      queryable: 75,
      processing: 0,
      failed: 0
    }
    initial.sources.pagination = {
      offset: 0,
      limit: 50,
      total: 75,
      has_more: true
    }
    api.bootstrap.mockResolvedValue(initial)
    api.listSources
      .mockResolvedValueOnce(
        sourcePage(
          items.slice(0, 50),
          { offset: 0, limit: 50, total: 75, has_more: true },
          initial.source_summary
        )
      )
      .mockResolvedValueOnce(
        sourcePage(
          [source("source-50"), ...items.slice(51)],
          { offset: 50, limit: 50, total: 75, has_more: false },
          initial.source_summary
        )
      )
    const { result } = renderHook(() => useSharedResearchWorkspace(4))
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    await act(async () => result.current.toggleSource(items[0], false))

    expect(result.current.state.sourceScopeMode).toBe("all")
    expect(result.current.state.selectedSourceIds).toEqual([])
    expect(result.current.state.errors.selection?.code).toBe("shared_source_selection_unavailable")
  })

  it("retains an immutable receipt for an invalid successful chat response", async () => {
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.ask
      .mockRejectedValueOnce(new SharedWorkspacePostCommitResponseError())
      .mockResolvedValueOnce(chatResponse("request-1", "1", "all"))
    const uuid = vi.fn().mockReturnValue("request-1")
    const { result } = renderHook(() => useSharedResearchWorkspace(4, { createRequestId: uuid }))
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))
    act(() => result.current.setDraft("Question"))

    await act(async () => result.current.submitDraft())

    const frozen = api.ask.mock.calls[0][1] as SharedChatRequest
    expect(result.current.state.pendingSubmission?.status).toBe("retryable")
    expect(result.current.state.errors.submission?.code).toBe("shared_chat_response_unconfirmed")

    await act(async () => result.current.retryPending())

    expect(api.ask.mock.calls[1][1]).toBe(frozen)
    expect(uuid).toHaveBeenCalledTimes(1)
    expect(result.current.state.lastCompletedAssistantMessageId).toBe("assistant-1")
  })

  it("preserves an exact raw draft edit made while a request is in flight", async () => {
    const pending = deferred<SharedChatResponse>()
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.ask.mockReturnValue(pending.promise)
    const { result } = renderHook(() =>
      useSharedResearchWorkspace(4, { createRequestId: () => "request-1" })
    )
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    act(() => result.current.setDraft("Question"))
    act(() => {
      void result.current.submitDraft()
    })
    await waitFor(() => expect(api.ask).toHaveBeenCalledTimes(1))
    act(() => result.current.setDraft("  Question  "))

    await act(async () => {
      pending.resolve(chatResponse("request-1", "1", "all"))
      await pending.promise
    })

    await waitFor(() =>
      expect(result.current.state.pendingSubmission).toBeNull()
    )
    expect(result.current.state.messages).toHaveLength(2)
    expect(result.current.state.draft).toBe("  Question  ")
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

  it("reuses a UUID only for transport ambiguity, never an untyped HTTP error", async () => {
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.ask
      .mockRejectedValueOnce({
        status: 400,
        detail: { message: "Malformed request" }
      })
      .mockRejectedValueOnce(new TypeError("connection reset"))
      .mockResolvedValueOnce(chatResponse("request-2", "2", "all"))
    const uuid = vi
      .fn()
      .mockReturnValueOnce("request-1")
      .mockReturnValueOnce("request-2")
    const { result } = renderHook(() =>
      useSharedResearchWorkspace(4, { createRequestId: uuid })
    )
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))
    act(() => result.current.setDraft("Question"))

    await act(async () => result.current.submitDraft())
    expect(result.current.state.pendingSubmission).toBeNull()
    await act(async () => result.current.retryPending())
    expect(api.ask).toHaveBeenCalledTimes(1)

    await act(async () => result.current.submitDraft())
    expect(result.current.state.pendingSubmission?.request.request_id).toBe(
      "request-2"
    )
    await act(async () => result.current.retryPending())

    expect(api.ask.mock.calls.map((call) => call[1].request_id)).toEqual([
      "request-1",
      "request-2",
      "request-2"
    ])
    expect(uuid).toHaveBeenCalledTimes(2)
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

  it("ignores a reordered same-share source response after a newer refresh", async () => {
    const older = deferred<SharedSourcePage>()
    const newer = deferred<SharedSourcePage>()
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.listSources
      .mockReturnValueOnce(older.promise)
      .mockReturnValueOnce(newer.promise)
    const { result } = renderHook(() => useSharedResearchWorkspace(4))
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    act(() => {
      void result.current.refreshSources({ offset: 0, limit: 50, q: "older" })
      void result.current.refreshSources({ offset: 0, limit: 50, q: "newer" })
    })
    const olderSignal = api.listSources.mock.calls[0][2] as AbortSignal
    expect(olderSignal.aborted).toBe(true)

    await act(async () => {
      newer.resolve(sourcePage([source("newer")]))
      await newer.promise
    })
    await act(async () => {
      older.resolve(sourcePage([source("older")]))
      await older.promise
    })

    expect(result.current.state.sources?.items[0]?.source_id).toBe("newer")
  })

  it("ignores reordered same-share preview and history responses", async () => {
    const olderPreview = deferred<SharedSourcePreview>()
    const newerPreview = deferred<SharedSourcePreview>()
    const olderHistory = deferred<SharedMessagePage>()
    const newerHistory = deferred<SharedMessagePage>()
    const initial = bootstrap(4)
    initial.conversation.next_before = "older"
    api.bootstrap.mockResolvedValue(initial)
    api.previewSource
      .mockReturnValueOnce(olderPreview.promise)
      .mockReturnValueOnce(newerPreview.promise)
    api.listMessages
      .mockReturnValueOnce(olderHistory.promise)
      .mockReturnValueOnce(newerHistory.promise)
    const { result } = renderHook(() => useSharedResearchWorkspace(4))
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    act(() => {
      void result.current.previewSource("older")
      void result.current.previewSource("newer")
      void result.current.loadOlderHistory()
      void result.current.loadOlderHistory()
    })

    await act(async () => {
      newerPreview.resolve(preview("newer"))
      newerHistory.resolve(historyPage("newer-message"))
      await Promise.all([newerPreview.promise, newerHistory.promise])
    })
    await act(async () => {
      olderPreview.resolve(preview("older"))
      olderHistory.resolve(historyPage("older-message"))
      await Promise.all([olderPreview.promise, olderHistory.promise])
    })

    expect(result.current.state.preview?.source_id).toBe("newer")
    expect(result.current.state.messages.map((message) => message.message_id)).toEqual([
      "newer-message"
    ])
  })

  it("ignores a reordered submission and reconciles a mismatched response with the same UUID", async () => {
    const older = deferred<SharedChatResponse>()
    const newer = deferred<SharedChatResponse>()
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.ask.mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise)
      .mockResolvedValueOnce(chatResponse("request-2", "3", "all"))
    const uuid = vi
      .fn()
      .mockReturnValueOnce("request-1")
      .mockReturnValueOnce("request-2")
    const { result } = renderHook(() =>
      useSharedResearchWorkspace(4, { createRequestId: uuid })
    )
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    act(() => result.current.setDraft("First"))
    act(() => {
      void result.current.submitDraft()
    })
    await waitFor(() => expect(api.ask).toHaveBeenCalledTimes(1))
    act(() => result.current.setDraft("Second"))
    act(() => {
      void result.current.submitDraft()
    })
    await waitFor(() => expect(api.ask).toHaveBeenCalledTimes(2))

    await act(async () => {
      newer.resolve(chatResponse("wrong-request", "2", "all"))
      await newer.promise
    })
    await act(async () => {
      older.resolve(chatResponse("request-1", "1", "all"))
      await older.promise
    })

    expect(result.current.state.messages).toEqual([])
    expect(result.current.state.draft).toBe("Second")
    expect(result.current.state.pendingSubmission?.status).toBe("retryable")
    expect(result.current.state.pendingSubmission?.request.request_id).toBe("request-2")
    expect(result.current.state.errors.submission?.code).toBe(
      "shared_chat_response_mismatch"
    )

    const frozen = api.ask.mock.calls[1][1]
    await act(async () => result.current.retryPending())

    expect(api.ask.mock.calls[2][1]).toBe(frozen)
    expect(uuid).toHaveBeenCalledTimes(2)
    expect(result.current.state.lastCompletedAssistantMessageId).toBe("assistant-3")
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

  it("blocks an over-cap all scope until an explicit include subset is chosen", async () => {
    const initial = bootstrap(4)
    initial.source_summary = {
      total: 501,
      queryable: 501,
      processing: 0,
      failed: 0
    }
    initial.sources.pagination = {
      offset: 0,
      limit: 50,
      total: 501,
      has_more: true
    }
    api.bootstrap.mockResolvedValue(initial)
    api.ask.mockRejectedValue(new TypeError("connection reset"))
    const { result } = renderHook(() =>
      useSharedResearchWorkspace(4, { createRequestId: () => "request-1" })
    )
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    act(() => result.current.setDraft("Question"))
    await act(async () => result.current.submitDraft())
    expect(api.ask).not.toHaveBeenCalled()

    act(() => result.current.setSelectedSourceIds(["source-1"]))
    await act(async () => result.current.submitDraft())

    expect(api.ask.mock.calls[0][1].source_scope).toEqual({
      mode: "include",
      source_ids: ["source-1"]
    })
  })

  it("supports explicit select-all and clear scope transitions", async () => {
    api.bootstrap.mockResolvedValue(bootstrap(4))
    const { result } = renderHook(() => useSharedResearchWorkspace(4))
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    act(() => result.current.setSelectedSourceIds(["source-1"]))
    expect(result.current.state.sourceScopeMode).toBe("include")

    act(() => result.current.selectAllSources())
    expect(result.current.state.sourceScopeMode).toBe("all")
    expect(result.current.state.selectedSourceIds).toEqual([])

    act(() => result.current.clearSelectedSources())
    expect(result.current.state.sourceScopeMode).toBe("include")
    expect(result.current.state.selectedSourceIds).toEqual([])
  })

  it("makes every inspect operation a no-op when inspect_sources is denied", async () => {
    const denied = bootstrap(4)
    denied.allowed_actions.inspect_sources = {
      allowed: false,
      reason_code: "workspace_inspection_disabled"
    }
    api.bootstrap.mockResolvedValue(denied)
    const { result } = renderHook(() => useSharedResearchWorkspace(4))
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))
    const initialQuery = result.current.state.sourceQuery
    const initialSelection = result.current.state.selectedSourceIds

    act(() => {
      result.current.setSourceQuery({ offset: 50, limit: 50, q: "blocked" })
      result.current.setSelectedSourceIds([])
      result.current.selectAllSources()
      result.current.clearSelectedSources()
    })
    await act(async () => {
      await result.current.refreshSources({ offset: 50, limit: 50 })
      await result.current.previewSource("source-1")
    })

    expect(result.current.state.sourceQuery).toEqual(initialQuery)
    expect(result.current.state.selectedSourceIds).toEqual(initialSelection)
    expect(api.listSources).not.toHaveBeenCalled()
    expect(api.previewSource).not.toHaveBeenCalled()
  })

  it("blocks retries during a rate limit and updates the bounded countdown", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-08-22T00:00:00Z"))
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.ask
      .mockRejectedValueOnce(apiError("shared_chat_rate_limited", true, 1500))
      .mockRejectedValueOnce(new TypeError("connection reset"))
    const uuid = vi
      .fn()
      .mockReturnValueOnce("request-1")
      .mockReturnValueOnce("request-2")
    const { result, unmount } = renderHook(() =>
      useSharedResearchWorkspace(4, { createRequestId: uuid })
    )
    await act(async () => Promise.resolve())
    await act(async () => Promise.resolve())

    act(() => result.current.setDraft("Keep this exact draft"))
    await act(async () => result.current.submitDraft())
    expect(result.current.state.rateLimitUntil).toBe(Date.now() + 1500)
    expect(result.current.state.rateLimitRemainingMs).toBe(1500)
    expect(result.current.state.draft).toBe("Keep this exact draft")

    await act(async () => result.current.submitDraft())
    expect(api.ask).toHaveBeenCalledTimes(1)

    await act(async () => vi.advanceTimersByTimeAsync(1000))
    expect(result.current.state.rateLimitRemainingMs).toBeGreaterThan(0)
    expect(result.current.state.rateLimitRemainingMs).toBeLessThanOrEqual(500)
    await act(async () => result.current.retryPending())
    expect(api.ask).toHaveBeenCalledTimes(1)

    await act(async () => vi.advanceTimersByTimeAsync(500))
    expect(result.current.state.rateLimitUntil).toBeNull()
    expect(result.current.state.rateLimitRemainingMs).toBe(0)
    await act(async () => result.current.submitDraft())
    expect(api.ask).toHaveBeenCalledTimes(2)
    expect(api.ask.mock.calls[1][1].request_id).toBe("request-2")

    unmount()
    vi.useRealTimers()
  })

  it("preserves the exact draft for a context-budget error", async () => {
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.ask.mockRejectedValue(
      apiError("shared_chat_context_too_large", false)
    )
    const { result } = renderHook(() => useSharedResearchWorkspace(4))
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    act(() => result.current.setDraft("Context draft"))
    await act(async () => result.current.submitDraft())

    expect(result.current.state.errors.submission?.code).toBe(
      "shared_chat_context_too_large"
    )
    expect(result.current.state.draft).toBe("Context draft")
  })

  it("never exposes stale evidence under a newer deferred preview target", async () => {
    const first = deferred<SharedSourcePreview>()
    const second = deferred<SharedSourcePreview>()
    api.bootstrap.mockResolvedValue(bootstrap(4))
    api.previewSource
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise)
    const { result } = renderHook(() => useSharedResearchWorkspace(4))
    await waitFor(() => expect(result.current.state.status).toBe("loaded"))

    act(() => {
      void result.current.previewSource("source-1")
    })
    expect(result.current.state.preview).toBeNull()
    expect(result.current.state.previewLoading).toBe(true)
    expect(result.current.state.previewTarget).toEqual({
      sourceId: "source-1",
      chunkIndex: null
    })

    act(() => {
      void result.current.previewSource("source-2", 9)
    })
    expect(result.current.state.preview).toBeNull()
    expect(result.current.state.previewTarget).toEqual({
      sourceId: "source-2",
      chunkIndex: 9
    })

    await act(async () => first.resolve(preview("source-1")))
    expect(result.current.state.preview).toBeNull()
    expect(result.current.state.previewTarget?.sourceId).toBe("source-2")

    await act(async () => second.resolve(preview("source-2")))
    expect(result.current.state.preview?.source_id).toBe("source-2")
    expect(result.current.state.previewLoading).toBe(false)
  })
})
