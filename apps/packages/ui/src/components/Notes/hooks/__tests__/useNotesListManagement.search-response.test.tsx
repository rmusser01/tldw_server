import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useNotesListManagement } from "../useNotesListManagement"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  getSetting: vi.fn(),
  setSetting: vi.fn(),
  confirmDanger: vi.fn(),
  setKeywordTokens: vi.fn(),
  message: {
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  }
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mocks.bgRequest
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    getSetting: mocks.getSetting,
    setSetting: mocks.setSetting
  }
})

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })

  return function Wrapper({ children }: { children: React.ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  }
}

describe("useNotesListManagement search response normalization", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getSetting.mockResolvedValue(null)
    mocks.setSetting.mockResolvedValue(undefined)
    mocks.confirmDanger.mockResolvedValue(true)
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/search/?")) {
        return {
          notes: [{ id: "n-1", title: "Alpha", content: "Body", last_modified: "2026-01-01T00:00:00Z" }],
          items: [{ id: "n-1", title: "Alpha", content: "Body", last_modified: "2026-01-01T00:00:00Z" }],
          results: [{ id: "n-1", title: "Alpha", content: "Body", last_modified: "2026-01-01T00:00:00Z" }],
          count: 1,
          limit: 1,
          offset: 0,
          total: 42,
          pagination: {
            mode: "offset",
            limit: 1,
            offset: 0,
            total: 42,
            has_more: true,
            next_offset: 1
          }
        }
      }
      if (path.startsWith("/api/v1/notes/collections")) {
        return { items: [], total: 0, pagination: { total: 0 } }
      }
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], total: 0, pagination: { total: 0 } }
      }
      return {}
    })
  })

  it("preserves canonical search totals from backend envelopes", async () => {
    const { result } = renderHook(
      () =>
        useNotesListManagement({
          isOnline: true,
          message: mocks.message as any,
          confirmDanger: mocks.confirmDanger,
          queryClient: new QueryClient(),
          t: (key: string) => key,
          keywordTokens: [],
          setKeywordTokens: mocks.setKeywordTokens,
          notebookKeywordTokens: []
        }),
      { wrapper: createWrapper() }
    )

    let response: Awaited<ReturnType<typeof result.current.fetchFilteredNotesRaw>> | undefined
    await act(async () => {
      response = await result.current.fetchFilteredNotesRaw("alpha", [], 1, 1)
    })

    expect(response?.items).toHaveLength(1)
    expect(response?.total).toBe(42)
  })
})
