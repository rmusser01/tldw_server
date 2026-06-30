import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  deriveServerChatHistoryViewState,
  fetchAllServerChatPages,
  filterServerChatHistoryItems,
  isRecoverableServerChatHistoryError,
  mapServerChatHistoryItems,
  useServerChatHistory
} from "@/hooks/useServerChatHistory"
import type { ServerChatSummary } from "@/services/tldw/TldwApiClient"

const {
  initializeMock,
  listChatsWithMetaMock,
  searchConversationsWithMetaMock,
  checkOnceMock
} = vi.hoisted(() => ({
  initializeMock: vi.fn(async () => undefined),
  listChatsWithMetaMock: vi.fn(),
  searchConversationsWithMetaMock: vi.fn(),
  checkOnceMock: vi.fn(async () => undefined)
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({ isConnected: true })
}))

vi.mock("@/store/connection", () => ({
  useConnectionStore: (selector: (state: { checkOnce: typeof checkOnceMock }) => unknown) =>
    selector({ checkOnce: checkOnceMock })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: () => initializeMock(),
    listChatsWithMeta: (params?: unknown, options?: unknown) =>
      listChatsWithMetaMock(params, options),
    searchConversationsWithMeta: (params?: unknown, options?: unknown) =>
      searchConversationsWithMetaMock(params, options)
  }
}))

const createChat = (
  id: number,
  overrides: Partial<ServerChatSummary> = {}
): ServerChatSummary => ({
  id: `chat-${id}`,
  title: `Chat ${id}`,
  created_at: `2026-01-${String(id).padStart(2, "0")}T00:00:00Z`,
  updated_at: `2026-01-${String(id).padStart(2, "0")}T01:00:00Z`,
  ...overrides
})

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0
      }
    }
  })

  return {
    queryClient,
    wrapper: ({ children }: { children: React.ReactNode }) =>
      React.createElement(QueryClientProvider, { client: queryClient }, children)
  }
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe("fetchAllServerChatPages", () => {
  it("fetches pages until the reported total is reached", async () => {
    const fetchPage = vi
      .fn()
      .mockResolvedValueOnce({ chats: [createChat(1), createChat(2)], total: 5 })
      .mockResolvedValueOnce({ chats: [createChat(3), createChat(4)], total: 5 })
      .mockResolvedValueOnce({ chats: [createChat(5)], total: 5 })

    const result = await fetchAllServerChatPages(fetchPage, {
      limit: 2,
      maxPages: 10
    })

    expect(result.map((chat) => chat.id)).toEqual([
      "chat-1",
      "chat-2",
      "chat-3",
      "chat-4",
      "chat-5"
    ])
    expect(fetchPage).toHaveBeenCalledTimes(3)
    expect(fetchPage).toHaveBeenNthCalledWith(1, { limit: 2, offset: 0, signal: undefined })
    expect(fetchPage).toHaveBeenNthCalledWith(2, { limit: 2, offset: 2, signal: undefined })
    expect(fetchPage).toHaveBeenNthCalledWith(3, { limit: 2, offset: 4, signal: undefined })
  })

  it("stops when maxPages is reached", async () => {
    const fetchPage = vi.fn().mockResolvedValue({
      chats: [createChat(1), createChat(2)],
      total: 999
    })

    const result = await fetchAllServerChatPages(fetchPage, {
      limit: 2,
      maxPages: 2
    })

    expect(result).toHaveLength(4)
    expect(fetchPage).toHaveBeenCalledTimes(2)
  })

  it("returns already-fetched chats when a later page is rate-limited", async () => {
    const rateLimitedError = new Error("rate_limited (GET /api/v1/chats/)")
    ;(rateLimitedError as Error & { status?: number }).status = 429
    const fetchPage = vi
      .fn()
      .mockResolvedValueOnce({ chats: [createChat(1), createChat(2)], total: 6 })
      .mockRejectedValueOnce(rateLimitedError)

    const result = await fetchAllServerChatPages(fetchPage, {
      limit: 2,
      maxPages: 10
    })

    expect(result.map((chat) => chat.id)).toEqual(["chat-1", "chat-2"])
    expect(fetchPage).toHaveBeenCalledTimes(2)
  })
})

describe("deriveServerChatHistoryViewState", () => {
  it("derives recoverable-error state when prior chat data is still usable", () => {
    const previous = mapServerChatHistoryItems([createChat(1), createChat(2)])

    const result = deriveServerChatHistoryViewState({
      previousData: previous,
      error: Object.assign(new Error("rate_limited"), { status: 429 })
    })

    expect(result.data).toEqual(previous)
    expect(result.sidebarRefreshState).toBe("recoverable-error")
    expect(result.hasUsableData).toBe(true)
    expect(result.isShowingStaleData).toBe(true)
  })

  it("returns unavailable recoverable state when no data was previously loaded", () => {
    const result = deriveServerChatHistoryViewState({
      previousData: [],
      error: new Error("The operation was aborted.")
    })

    expect(result.data).toEqual([])
    expect(result.sidebarRefreshState).toBe("recoverable-error")
    expect(result.hasUsableData).toBe(false)
    expect(result.isShowingStaleData).toBe(false)
  })
})

describe("useServerChatHistory", () => {
  it("does not fetch server chat history when overview mode is disabled", async () => {
    const { queryClient, wrapper } = createWrapper()

    renderHook(
      () =>
        useServerChatHistory("", {
          enabled: false,
          mode: "overview"
        }),
      { wrapper }
    )

    await waitFor(() => {
      expect(initializeMock).not.toHaveBeenCalled()
      expect(listChatsWithMetaMock).not.toHaveBeenCalled()
      expect(searchConversationsWithMetaMock).not.toHaveBeenCalled()
    })

    queryClient.clear()
  })

  it("uses a search-specific fetch path instead of overview pagination when search is active", async () => {
    searchConversationsWithMetaMock.mockResolvedValueOnce({
      chats: [createChat(7, { title: "Quota review" })],
      total: 1
    })

    const { queryClient, wrapper } = createWrapper()
    const { result } = renderHook(
      () =>
        useServerChatHistory("quota", {
          enabled: true,
          mode: "search"
        }),
      { wrapper }
    )

    await waitFor(() =>
      expect(result.current.data.map((chat) => chat.id)).toEqual(["chat-7"])
    )

    expect(searchConversationsWithMetaMock).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "quota",
        limit: 50,
        offset: 0,
        order_by: "recency"
      }),
      expect.anything()
    )
    expect(listChatsWithMetaMock).not.toHaveBeenCalled()

    queryClient.clear()
  })

  it("fetches only the requested overview page when server pagination is supported", async () => {
    listChatsWithMetaMock.mockResolvedValueOnce({
      chats: [createChat(26), createChat(27)],
      total: 60
    })

    const { queryClient, wrapper } = createWrapper()
    const { result } = renderHook(
      () =>
        useServerChatHistory("", {
          enabled: true,
          mode: "overview",
          page: 2,
          limit: 25,
          filterMode: "all"
        }),
      { wrapper }
    )

    await waitFor(() =>
      expect(result.current.data.map((chat) => chat.id)).toEqual([
        "chat-26",
        "chat-27"
      ])
    )

    expect(listChatsWithMetaMock).toHaveBeenCalledTimes(1)
    expect(listChatsWithMetaMock).toHaveBeenCalledWith(
      expect.objectContaining({
        limit: 25,
        offset: 25,
        ordering: "-updated_at",
        include_message_counts: false
      }),
      expect.anything()
    )
    expect(result.current.total).toBe(60)

    queryClient.clear()
  })

  it("uses server-side paging for character overview when character scope is supported", async () => {
    listChatsWithMetaMock.mockResolvedValueOnce({
      chats: [createChat(1, { character_id: 7 })],
      total: 10
    })

    const { queryClient, wrapper } = createWrapper()
    const { result } = renderHook(
      () =>
        useServerChatHistory("", {
          enabled: true,
          mode: "overview",
          page: 1,
          limit: 25,
          filterMode: "character"
        }),
      { wrapper }
    )

    await waitFor(() =>
      expect(result.current.data.map((chat) => chat.id)).toEqual(["chat-1"])
    )

    expect(listChatsWithMetaMock).toHaveBeenCalledTimes(1)
    expect(listChatsWithMetaMock).toHaveBeenCalledWith(
      expect.objectContaining({
        limit: 25,
        offset: 0,
        ordering: "-updated_at",
        include_message_counts: false,
        character_scope: "character"
      }),
      expect.anything()
    )
    expect(result.current.total).toBe(10)

    queryClient.clear()
  })

  it("passes character scope through the server-side search path", async () => {
    searchConversationsWithMetaMock.mockResolvedValueOnce({
      chats: [createChat(12, { character_id: null, title: "Quota plain" })],
      total: 1
    })

    const { queryClient, wrapper } = createWrapper()
    const { result } = renderHook(
      () =>
        useServerChatHistory("quota", {
          enabled: true,
          mode: "search",
          filterMode: "non_character"
        }),
      { wrapper }
    )

    await waitFor(() =>
      expect(result.current.data.map((chat) => chat.id)).toEqual(["chat-12"])
    )

    expect(searchConversationsWithMetaMock).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "quota",
        limit: 50,
        offset: 0,
        order_by: "recency",
        character_scope: "non_character"
      }),
      expect.anything()
    )

    queryClient.clear()
  })

  it("passes character filter mode through server-side search as character scope", async () => {
    searchConversationsWithMetaMock.mockResolvedValueOnce({
      chats: [createChat(17, { character_id: 7, title: "Mira replay" })],
      total: 1
    })

    const { queryClient, wrapper } = createWrapper()
    const { result } = renderHook(
      () =>
        useServerChatHistory("mira", {
          enabled: true,
          mode: "search",
          filterMode: "character"
        }),
      { wrapper }
    )

    await waitFor(() =>
      expect(result.current.data.map((chat) => chat.id)).toEqual(["chat-17"])
    )

    expect(searchConversationsWithMetaMock).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "mira",
        limit: 50,
        offset: 0,
        order_by: "recency",
        character_scope: "character"
      }),
      expect.anything()
    )

    queryClient.clear()
  })

  it("pages server-side search results when search mode receives page and limit", async () => {
    searchConversationsWithMetaMock.mockResolvedValueOnce({
      chats: [createChat(26, { title: "Quota page 2" })],
      total: 60
    })

    const { queryClient, wrapper } = createWrapper()
    const { result } = renderHook(
      () =>
        useServerChatHistory("quota", {
          enabled: true,
          mode: "search",
          page: 2,
          limit: 25
        }),
      { wrapper }
    )

    await waitFor(() =>
      expect(result.current.data.map((chat) => chat.id)).toEqual(["chat-26"])
    )

    expect(searchConversationsWithMetaMock).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "quota",
        limit: 25,
        offset: 25,
        order_by: "recency"
      }),
      expect.anything()
    )
    expect(result.current.total).toBe(60)

    queryClient.clear()
  })

  it("uses server-side search when search mode targets trash chats", async () => {
    searchConversationsWithMetaMock.mockResolvedValueOnce({
      chats: [createChat(9, { title: "Quota cleanup" })],
      total: 1
    })

    const { queryClient, wrapper } = createWrapper()
    const { result } = renderHook(
      () =>
        useServerChatHistory("quota", {
          enabled: true,
          mode: "search",
          deletedOnly: true
        }),
      { wrapper }
    )

    await waitFor(() =>
      expect(result.current.data.map((chat) => chat.id)).toEqual(["chat-9"])
    )

    expect(searchConversationsWithMetaMock).toHaveBeenCalledWith(
      expect.objectContaining({
        deleted_only: true,
        include_deleted: true,
        query: "quota",
        limit: 50,
        offset: 0,
        order_by: "recency"
      }),
      expect.anything()
    )
    expect(listChatsWithMetaMock).not.toHaveBeenCalled()

    queryClient.clear()
  })

  it("keeps previously rendered chat rows visible after a recoverable refresh failure", async () => {
    listChatsWithMetaMock
      .mockResolvedValueOnce({
        chats: [createChat(1), createChat(2)],
        total: 2
      })
      .mockRejectedValueOnce(
        Object.assign(new Error("rate_limited (GET /api/v1/chats/)"), { status: 429 })
      )

    const { queryClient, wrapper } = createWrapper()
    const { result } = renderHook(() => useServerChatHistory(""), { wrapper })

    await waitFor(() => expect(result.current.data.map((chat) => chat.id)).toEqual(["chat-1", "chat-2"]))
    expect(result.current.sidebarRefreshState).toBe("ready")
    expect(result.current.hasUsableData).toBe(true)
    expect(result.current.isShowingStaleData).toBe(false)

    await result.current.refetch()

    await waitFor(() => {
      expect(result.current.data.map((chat) => chat.id)).toEqual(["chat-1", "chat-2"])
      expect(result.current.sidebarRefreshState).toBe("recoverable-error")
      expect(result.current.hasUsableData).toBe(true)
      expect(result.current.isShowingStaleData).toBe(true)
    })

    queryClient.clear()
  })
})

describe("useServerChatHistory scope forwarding", () => {
  it("passes workspace scope through to listChatsWithMeta", async () => {
    listChatsWithMetaMock.mockResolvedValueOnce({
      chats: [createChat(1)],
      total: 1
    })

    const { queryClient, wrapper } = createWrapper()
    const { result } = renderHook(
      () =>
        useServerChatHistory("", {
          scope: { type: "workspace", workspaceId: "workspace-a" }
        }),
      { wrapper }
    )

    await waitFor(() => expect(result.current.data).toHaveLength(1))

    expect(listChatsWithMetaMock).toHaveBeenCalledWith(
      expect.objectContaining({
        limit: expect.any(Number),
        offset: expect.any(Number)
      }),
      expect.objectContaining({
        scope: { type: "workspace", workspaceId: "workspace-a" }
      })
    )

    queryClient.clear()
  })

  it("passes workspace scope through to searchConversationsWithMeta", async () => {
    searchConversationsWithMetaMock.mockResolvedValueOnce({
      chats: [createChat(2, { title: "Scoped quota" })],
      total: 1
    })

    const { queryClient, wrapper } = createWrapper()
    const { result } = renderHook(
      () =>
        useServerChatHistory("quota", {
          mode: "search",
          scope: { type: "workspace", workspaceId: "workspace-a" }
        }),
      { wrapper }
    )

    await waitFor(() => expect(result.current.data).toHaveLength(1))

    expect(searchConversationsWithMetaMock).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "quota",
        order_by: "recency"
      }),
      expect.objectContaining({
        scope: { type: "workspace", workspaceId: "workspace-a" }
      })
    )

    queryClient.clear()
  })
})

describe("filterServerChatHistoryItems", () => {
  it("filters case-insensitively by title, topic, and state", () => {
    const mapped = mapServerChatHistoryItems([
      createChat(1, { title: "Roadmap Review", topic_label: "planning", state: "resolved" }),
      createChat(2, { title: "Standup", topic_label: "status", state: "in-progress" }),
      createChat(3, { title: "Architecture", topic_label: "Design", state: "backlog" })
    ])

    expect(filterServerChatHistoryItems(mapped, "review").map((item) => item.id)).toEqual([
      "chat-1"
    ])
    expect(filterServerChatHistoryItems(mapped, "DESIGN").map((item) => item.id)).toEqual([
      "chat-3"
    ])
    expect(filterServerChatHistoryItems(mapped, "in-progress").map((item) => item.id)).toEqual(
      ["chat-2"]
    )
  })
})

describe("isRecoverableServerChatHistoryError", () => {
  it("returns true for auth status errors", () => {
    expect(
      isRecoverableServerChatHistoryError({
        message: "Invalid API key (GET /api/v1/chats/)",
        status: 401
      })
    ).toBe(true)

    expect(
      isRecoverableServerChatHistoryError({
        message: "Forbidden (GET /api/v1/chats/)",
        status: 403
      })
    ).toBe(true)
  })

  it("returns true for auth/config messages without explicit status", () => {
    expect(
      isRecoverableServerChatHistoryError(
        new Error("server not configured (GET /api/v1/chats/)")
      )
    ).toBe(true)

    expect(
      isRecoverableServerChatHistoryError(
        new Error("Unauthorized (GET /api/v1/chats/)")
      )
    ).toBe(true)

    expect(
      isRecoverableServerChatHistoryError(
        new Error("The operation was aborted. (GET /api/v1/chats/)")
      )
    ).toBe(true)

    expect(
      isRecoverableServerChatHistoryError(
        new Error("rate_limited (GET /api/v1/chats/)")
      )
    ).toBe(true)
  })

  it("returns true for HTTP 429 rate-limit status", () => {
    expect(
      isRecoverableServerChatHistoryError({
        message: "Too Many Requests (GET /api/v1/chats/)",
        status: 429
      })
    ).toBe(true)
  })

  it("returns false for non-auth server failures", () => {
    expect(
      isRecoverableServerChatHistoryError(
        new Error("Internal server error (GET /api/v1/chats/)")
      )
    ).toBe(false)
  })
})
