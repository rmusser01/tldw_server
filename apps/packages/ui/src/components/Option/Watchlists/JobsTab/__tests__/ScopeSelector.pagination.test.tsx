import { describe, expect, it, vi, beforeEach } from "vitest"
import { waitFor } from "@testing-library/react"
import { render } from "../../shared/__tests__/render-with-watchlists-query"
import { ScopeSelector } from "../ScopeSelector"

const servicesMock = vi.hoisted(() => ({
  fetchWatchlistSources: vi.fn(),
  fetchWatchlistGroups: vi.fn(),
  fetchWatchlistTags: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        if (typeof fallbackOrOptions.defaultValue === "string") {
          return fallbackOrOptions.defaultValue
        }
      }
      return key
    }
  })
}))

vi.mock("antd", () => ({
  Select: () => <div data-testid="select" />,
  Tabs: ({ items }: { items?: Array<{ key: string; children?: unknown }> }) => (
    <div>{items?.find((item) => item.key === "sources")?.children as any}</div>
  ),
  Tag: ({ children }: { children: unknown }) => <span>{children as any}</span>
}))

vi.mock("@/services/watchlists", () => ({
  fetchWatchlistSources: (...args: unknown[]) => servicesMock.fetchWatchlistSources(...args),
  fetchWatchlistGroups: (...args: unknown[]) => servicesMock.fetchWatchlistGroups(...args),
  fetchWatchlistTags: (...args: unknown[]) => servicesMock.fetchWatchlistTags(...args)
}))

const makeSources = (count: number, start = 1) =>
  Array.from({ length: count }, (_entry, index) => {
    const id = start + index
    return {
      id,
      name: `Source ${id}`,
      url: `https://example.com/${id}`,
      source_type: "rss",
      active: true,
      tags: [],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: null,
      last_scraped_at: null,
      status: null
    }
  })

describe("ScopeSelector source pagination", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    servicesMock.fetchWatchlistGroups.mockResolvedValue({ items: [], total: 0 })
    servicesMock.fetchWatchlistTags.mockResolvedValue({ items: [], total: 0 })
  })

  it("loads sources with backend-safe page size", async () => {
    servicesMock.fetchWatchlistSources
      .mockResolvedValueOnce({ items: makeSources(200, 1), total: 250 })
      .mockResolvedValueOnce({ items: makeSources(50, 201), total: 250 })

    render(<ScopeSelector value={{}} onChange={vi.fn()} />)

    await waitFor(() => {
      expect(servicesMock.fetchWatchlistSources).toHaveBeenCalledTimes(2)
    })

    expect(servicesMock.fetchWatchlistSources).toHaveBeenNthCalledWith(1, {
      page: 1,
      size: 200
    }, expect.any(AbortSignal))
    expect(servicesMock.fetchWatchlistSources).toHaveBeenNthCalledWith(2, {
      page: 2,
      size: 200
    }, expect.any(AbortSignal))
  })

  it("shares a scope catalog request between selectors for the same watchlist", async () => {
    servicesMock.fetchWatchlistSources.mockResolvedValue({ items: makeSources(1), total: 1 })

    render(<>
      <ScopeSelector value={{}} onChange={vi.fn()} watchlistId={4} />
      <ScopeSelector value={{}} onChange={vi.fn()} watchlistId={4} />
    </>)

    await waitFor(() => expect(servicesMock.fetchWatchlistSources).toHaveBeenCalledTimes(1))
    expect(servicesMock.fetchWatchlistSources).toHaveBeenCalledWith(
      { watchlist_id: 4, page: 1, size: 200 },
      expect.any(AbortSignal)
    )
  })
})
