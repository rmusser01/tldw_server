import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useNotesListManagement } from "../useNotesListManagement"

/**
 * Decision 19 (#3101): a saved filter always matches all of its tags. The
 * reader's any/all toggle applies to the tags the reader picked on top.
 */
const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  getSetting: vi.fn(),
  setSetting: vi.fn(),
  confirmDanger: vi.fn(),
  setKeywordTokens: vi.fn(),
  message: { success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() }
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mocks.bgRequest
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/settings/registry")>()
  return { ...actual, getSetting: mocks.getSetting, setSetting: mocks.setSetting }
})

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider
    client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    {children}
  </QueryClientProvider>
)

const renderList = (keywordTokens: string[], notebookKeywordTokens: string[]) =>
  renderHook(
    () =>
      useNotesListManagement({
        isOnline: true,
        message: mocks.message as any,
        confirmDanger: mocks.confirmDanger,
        queryClient: new QueryClient(),
        t: (key: string) => key,
        keywordTokens,
        setKeywordTokens: mocks.setKeywordTokens,
        notebookKeywordTokens
      }),
    { wrapper }
  )

const lastSearch = (): URLSearchParams => {
  const path = mocks.bgRequest.mock.calls
    .map(([request]) => String(request?.path || ""))
    .filter((candidate) => candidate.startsWith("/api/v1/notes/search/?"))
    .at(-1)
  return new URLSearchParams(String(path).split("?")[1])
}

describe("useNotesListManagement saved filters (decision 19)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getSetting.mockResolvedValue(null)
    mocks.setSetting.mockResolvedValue(undefined)
    mocks.bgRequest.mockResolvedValue({ items: [], total: 0, pagination: { total: 0 } })
  })

  it("matches every tag of a saved filter even when the toggle says any", async () => {
    const { result } = renderList([], ["rag", "baking"])
    act(() => result.current.setTagMatch("any"))

    await act(async () => {
      await result.current.fetchFilteredNotesRaw("", ["rag", "baking"], 1, 20)
    })

    expect(lastSearch().getAll("tokens")).toEqual(["rag", "baking"])
    expect(lastSearch().get("tokens_match")).toBe("all")
    expect(result.current.tagMatchApplies).toBe(false)
  })

  it("applies the toggle to picked tags and keeps the saved filter's tags required", async () => {
    const { result } = renderList(["eval", "thesis"], ["rag", "baking"])
    act(() => result.current.setTagMatch("any"))

    await act(async () => {
      await result.current.fetchFilteredNotesRaw("", ["eval", "thesis", "rag", "baking"], 1, 20)
    })

    expect(lastSearch().getAll("tokens")).toEqual(["eval", "thesis"])
    expect(lastSearch().get("tokens_match")).toBe("any")
    expect(lastSearch().getAll("required_tokens")).toEqual(["rag", "baking"])
    expect(result.current.tagMatchApplies).toBe(true)
  })

  it("keeps the toggle for two picked tags when one is in the saved filter, and any is met by it", async () => {
    const { result } = renderList(["a", "c"], ["a", "b"])
    expect(result.current.tagMatchApplies).toBe(true)
    act(() => result.current.setTagMatch("any"))

    await act(async () => {
      await result.current.fetchFilteredNotesRaw("", ["a", "c", "b"], 1, 20)
    })

    expect(lastSearch().getAll("tokens")).toEqual(["a", "b"])
    expect(lastSearch().get("tokens_match")).toBe("all")
    expect(lastSearch().has("required_tokens")).toBe(false)
  })

  it("does not offer the toggle for one picked tag on top of a saved filter", () => {
    const { result } = renderList(["eval"], ["rag", "baking"])

    expect(result.current.tagMatchApplies).toBe(false)
  })
})
