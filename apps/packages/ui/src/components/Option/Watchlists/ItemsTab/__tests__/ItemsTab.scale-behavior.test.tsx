// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, afterEach, describe, expect, it, vi, type Mock } from "vitest"
import { ItemsTab } from "../ItemsTab"
import { useWatchlistsStore } from "@/store/watchlists"

const serviceMocks = vi.hoisted(() => ({
  createWatchlistOutput: vi.fn(),
  fetchScrapedItemSmartCounts: vi.fn(),
  fetchWatchlistSources: vi.fn(),
  fetchWatchlistRuns: vi.fn(),
  fetchScrapedItems: vi.fn(),
  updateScrapedItem: vi.fn()
}))

const uiMocks = vi.hoisted(() => ({
  messageSuccess: vi.fn(),
  messageInfo: vi.fn(),
  messageWarning: vi.fn(),
  messageError: vi.fn()
}))

const interpolate = (template: string, values?: Record<string, unknown>) => {
  if (!values) return template
  return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
    const value = values[token]
    return value == null ? "" : String(value)
  })
}

const stableTranslate = (
  key: string,
  fallbackOrOptions?: string | { defaultValue?: string },
  maybeOptions?: Record<string, unknown>
) => {
  if (typeof fallbackOrOptions === "string") {
    return interpolate(fallbackOrOptions, maybeOptions)
  }
  if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
    const maybeDefault = fallbackOrOptions.defaultValue
    if (typeof maybeDefault === "string") {
      return interpolate(maybeDefault, maybeOptions)
    }
  }
  return key
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: stableTranslate
  })
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    message: {
      success: (...args: unknown[]) => uiMocks.messageSuccess(...args),
      info: (...args: unknown[]) => uiMocks.messageInfo(...args),
      warning: (...args: unknown[]) => uiMocks.messageWarning(...args),
      error: (...args: unknown[]) => uiMocks.messageError(...args)
    }
  }
})

vi.mock("@/services/watchlists", () => ({
  createWatchlistOutput: (...args: unknown[]) => serviceMocks.createWatchlistOutput(...args),
  fetchScrapedItemSmartCounts: (...args: unknown[]) =>
    serviceMocks.fetchScrapedItemSmartCounts(...args),
  fetchWatchlistSources: (...args: unknown[]) => serviceMocks.fetchWatchlistSources(...args),
  fetchWatchlistRuns: (...args: unknown[]) => serviceMocks.fetchWatchlistRuns(...args),
  fetchScrapedItems: (...args: unknown[]) => serviceMocks.fetchScrapedItems(...args),
  updateScrapedItem: (...args: unknown[]) => serviceMocks.updateScrapedItem(...args)
}))

const buildSources = (count: number) =>
  Array.from({ length: count }, (_value, index) => {
    const id = index + 1
    return {
      id,
      name: `Source ${id}`,
      url: `https://example.com/source-${id}.xml`,
      source_type: id % 2 === 0 ? "rss" : "site",
      active: true,
      tags: id % 3 === 0 ? ["tech"] : ["news"],
      created_at: "2026-02-20T07:00:00Z",
      updated_at: "2026-02-20T07:00:00Z",
      last_scraped_at: "2026-02-20T08:00:00Z",
      status: "healthy"
    }
  })

const itemsFixture = [
  {
    id: 101,
    run_id: 1,
    job_id: 1,
    source_id: 1,
    url: "https://example.com/one",
    title: "Item One",
    summary: "Summary one",
    tags: ["tech"],
    status: "ingested",
    reviewed: false,
    queued_for_briefing: false,
    created_at: "2026-02-18T08:00:00Z",
    published_at: "2026-02-18T08:00:00Z"
  },
  {
    id: 102,
    run_id: 1,
    job_id: 1,
    source_id: 2,
    url: "https://example.com/two",
    title: "Item Two",
    summary: "Summary two",
    tags: ["tech"],
    status: "ingested",
    reviewed: false,
    queued_for_briefing: false,
    created_at: "2026-02-18T08:10:00Z",
    published_at: "2026-02-18T08:10:00Z"
  }
]

describe("ItemsTab scale behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    useWatchlistsStore.getState().resetStore()
    ;(serviceMocks.fetchScrapedItemSmartCounts as Mock).mockResolvedValue({
      all: 2,
      today: 2,
      today_unread: 2,
      unread: 2,
      reviewed: 0,
      queued: 0
    })

    ;(serviceMocks.fetchScrapedItems as Mock).mockImplementation(async (params?: Record<string, unknown>) => {
      if (params?.size === 1) {
        if (params?.reviewed === false) return { items: [], total: 2, page: 1, size: 1, has_more: false }
        if (params?.reviewed === true) return { items: [], total: 0, page: 1, size: 1, has_more: false }
        return { items: [], total: 2, page: 1, size: 1, has_more: false }
      }
      return {
        items: itemsFixture,
        total: itemsFixture.length,
        page: Number(params?.page || 1),
        size: Number(params?.size || 25),
        has_more: false
      }
    })

    ;(serviceMocks.updateScrapedItem as Mock).mockImplementation(async (itemId: number) => ({
      id: itemId,
      reviewed: true
    }))

    ;(serviceMocks.fetchWatchlistRuns as Mock).mockResolvedValue({
      items: [{ id: 1, job_id: 1, status: "completed" }],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })
    ;(serviceMocks.createWatchlistOutput as Mock).mockResolvedValue({ id: 1, run_id: 1 })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("renders source rows incrementally for large source catalogs and expands on scroll", async () => {
    const sources = buildSources(260)
    ;(serviceMocks.fetchWatchlistSources as Mock).mockImplementation(async (params?: Record<string, unknown>) => {
      const page = Number(params?.page || 1)
      const size = Number(params?.size || 200)
      const start = (page - 1) * size
      const end = start + size
      return {
        items: sources.slice(start, end),
        total: sources.length,
        page,
        size,
        has_more: end < sources.length
      }
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-items-source-row-1")).toBeInTheDocument()
    })

    expect(screen.getAllByTestId(/watchlists-items-source-row-\d+/)).toHaveLength(120)
    expect(screen.getByTestId("watchlists-items-source-window-hint")).toHaveTextContent(
      "Showing 120 of 260 feeds. Scroll to load more."
    )

    const list = screen.getByTestId("watchlists-items-source-list")
    let scrollTop = 0
    Object.defineProperty(list, "scrollHeight", { configurable: true, get: () => 2600 })
    Object.defineProperty(list, "clientHeight", { configurable: true, get: () => 420 })
    Object.defineProperty(list, "scrollTop", { configurable: true, get: () => scrollTop })

    scrollTop = 2400
    fireEvent.scroll(list)
    await waitFor(() => {
      expect(screen.getAllByTestId(/watchlists-items-source-row-\d+/).length).toBeGreaterThan(120)
    })

    fireEvent.scroll(list)
    await waitFor(() => {
      expect(screen.getAllByTestId(/watchlists-items-source-row-\d+/)).toHaveLength(260)
    })
  })

  it("shows a source-catalog cap notice when source volume exceeds the load limit", async () => {
    const sources = buildSources(1205)
    ;(serviceMocks.fetchWatchlistSources as Mock).mockImplementation(async (params?: Record<string, unknown>) => {
      const page = Number(params?.page || 1)
      const size = Number(params?.size || 200)
      const start = (page - 1) * size
      const end = start + size
      return {
        items: sources.slice(start, end),
        total: sources.length,
        page,
        size,
        has_more: end < sources.length
      }
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-items-source-cap-hint")).toHaveTextContent(
        "Showing first 1000 feeds. Use Feeds tab filters to narrow your source list."
      )
    })

    expect(serviceMocks.fetchWatchlistSources).toHaveBeenCalledTimes(5)
  })

  it("debounces search-driven reloads and keeps row selection local without extra fetches", async () => {
    const sources = buildSources(15)
    ;(serviceMocks.fetchWatchlistSources as Mock).mockResolvedValue({
      items: sources,
      total: sources.length,
      page: 1,
      size: 200,
      has_more: false
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    ;(serviceMocks.fetchScrapedItems as Mock).mockClear()

    const search = screen.getByPlaceholderText("Search updates...")
    fireEvent.change(search, { target: { value: "a" } })
    fireEvent.change(search, { target: { value: "ai" } })
    fireEvent.change(search, { target: { value: "ai chips" } })
    expect(serviceMocks.fetchScrapedItems).toHaveBeenCalledTimes(0)

    await waitFor(() => {
      expect(serviceMocks.fetchScrapedItems).toHaveBeenCalledTimes(1)
    })

    ;(serviceMocks.fetchScrapedItems as Mock).mockClear()
    fireEvent.click(screen.getByTestId("watchlists-item-row-102"))
    expect(screen.getByTestId("watchlists-item-reader")).toHaveTextContent("Item Two")
    expect(serviceMocks.fetchScrapedItems).toHaveBeenCalledTimes(0)
  })
})
