// @vitest-environment jsdom

import React from "react"
import { beforeEach, describe, expect, it, vi, type Mock } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { ItemsTab } from "../ItemsTab"
import { useWatchlistsStore } from "@/store/watchlists"

const serviceMocks = vi.hoisted(() => ({
  fetchScrapedItemSmartCounts: vi.fn(),
  fetchWatchlistSources: vi.fn(),
  fetchWatchlistRuns: vi.fn(),
  fetchScrapedItems: vi.fn(),
  updateScrapedItem: vi.fn()
}))

const uiMocks = vi.hoisted(() => ({
  modalConfirm: vi.fn(),
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
  const mockedModal = Object.assign(actual.Modal, {
    confirm: (config: Record<string, unknown>) => uiMocks.modalConfirm(config)
  })
  return {
    ...actual,
    Modal: mockedModal,
    message: {
      success: (...args: unknown[]) => uiMocks.messageSuccess(...args),
      info: (...args: unknown[]) => uiMocks.messageInfo(...args),
      warning: (...args: unknown[]) => uiMocks.messageWarning(...args),
      error: (...args: unknown[]) => uiMocks.messageError(...args)
    }
  }
})

vi.mock("@/services/watchlists", () => ({
  fetchScrapedItemSmartCounts: (...args: unknown[]) =>
    serviceMocks.fetchScrapedItemSmartCounts(...args),
  fetchWatchlistSources: (...args: unknown[]) => serviceMocks.fetchWatchlistSources(...args),
  fetchWatchlistRuns: (...args: unknown[]) => serviceMocks.fetchWatchlistRuns(...args),
  fetchScrapedItems: (...args: unknown[]) => serviceMocks.fetchScrapedItems(...args),
  updateScrapedItem: (...args: unknown[]) => serviceMocks.updateScrapedItem(...args)
}))

const makeSources = (count: number) =>
  Array.from({ length: count }, (_, index) => {
    const sourceId = index + 1
    return {
      id: sourceId,
      name: `Source ${sourceId}`,
      url: `https://source${sourceId}.example.com/rss.xml`,
      source_type: "rss",
      active: true,
      tags: ["news"],
      created_at: "2026-02-18T07:00:00Z",
      updated_at: "2026-02-18T07:00:00Z",
      last_scraped_at: `2026-02-18T08:${String(index % 60).padStart(2, "0")}:00Z`,
      status: "healthy"
    }
  })

const listItemsFixture = [
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
    created_at: "2026-02-18T08:00:00Z",
    published_at: "2026-02-18T08:00:00Z"
  },
  {
    id: 102,
    run_id: 1,
    job_id: 1,
    source_id: 1,
    url: "https://example.com/two",
    title: "Item Two",
    summary: "Summary two",
    tags: ["tech"],
    status: "ingested",
    reviewed: false,
    created_at: "2026-02-18T08:10:00Z",
    published_at: "2026-02-18T08:10:00Z"
  }
]

describe("ItemsTab scale and responsive behavior", () => {
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
    ;(serviceMocks.fetchWatchlistRuns as Mock).mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 200,
      has_more: false
    })

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: query.includes("max-width: 1024px"),
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }

    ;(serviceMocks.fetchScrapedItems as Mock).mockImplementation(async (params?: Record<string, unknown>) => {
      if (params?.size === 1) {
        if (params?.reviewed === false) return { items: [], total: 2, page: 1, size: 1, has_more: false }
        if (params?.reviewed === true) return { items: [], total: 0, page: 1, size: 1, has_more: false }
        return { items: [], total: 2, page: 1, size: 1, has_more: false }
      }
      return {
        items: listItemsFixture,
        total: listItemsFixture.length,
        page: Number(params?.page || 1),
        size: Number(params?.size || 25),
        has_more: false
      }
    })

    ;(serviceMocks.updateScrapedItem as Mock).mockImplementation(async (itemId: number) => ({
      id: itemId,
      reviewed: true
    }))

    uiMocks.modalConfirm.mockImplementation((config: Record<string, unknown>) => {
      const onOk = config?.onOk
      if (typeof onOk === "function") {
        return Promise.resolve(onOk())
      }
      return undefined
    })
  })

  it("renders all source rows for 5-source profile without virtualized hint", async () => {
    serviceMocks.fetchWatchlistSources.mockResolvedValue({
      items: makeSources(5),
      total: 5,
      page: 1,
      size: 200,
      has_more: false
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getAllByTestId(/watchlists-items-source-row-\d+/)).toHaveLength(5)
    })

    expect(screen.queryByTestId("watchlists-items-source-window-hint")).not.toBeInTheDocument()
  })

  it("caps source rows at 120 for 200-source profile and expands on scroll", async () => {
    serviceMocks.fetchWatchlistSources.mockResolvedValue({
      items: makeSources(200),
      total: 200,
      page: 1,
      size: 200,
      has_more: false
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getAllByTestId(/watchlists-items-source-row-\d+/)).toHaveLength(120)
    })

    expect(screen.queryByTestId("watchlists-items-source-row-200")).not.toBeInTheDocument()
    expect(screen.getByTestId("watchlists-items-source-window-hint")).toHaveTextContent(
      "Showing 120 of 200 feeds. Scroll to load more."
    )

    const sourceList = screen.getByTestId("watchlists-items-source-list")
    Object.defineProperty(sourceList, "scrollTop", { configurable: true, value: 950 })
    Object.defineProperty(sourceList, "scrollHeight", { configurable: true, value: 1200 })
    Object.defineProperty(sourceList, "clientHeight", { configurable: true, value: 240 })

    fireEvent.scroll(sourceList)

    await waitFor(() => {
      expect(screen.getAllByTestId(/watchlists-items-source-row-\d+/)).toHaveLength(200)
    })
    expect(screen.getByTestId("watchlists-items-source-row-200")).toBeInTheDocument()
    expect(screen.queryByTestId("watchlists-items-source-window-hint")).not.toBeInTheDocument()
  })

  it("keeps triage panes and actions available on narrow viewport", async () => {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 390
    })
    window.dispatchEvent(new Event("resize"))

    serviceMocks.fetchWatchlistSources.mockResolvedValue({
      items: makeSources(50),
      total: 50,
      page: 1,
      size: 200,
      has_more: false
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    expect(screen.getByTestId("watchlists-items-left-pane")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-items-list-pane")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-items-reader-pane")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("watchlists-item-row-101"))
    expect(screen.getByTestId("watchlists-item-include-briefing")).toBeInTheDocument()
  })

  it("reuses smart-count cache on immediate refresh to avoid repeated count fan-out", async () => {
    serviceMocks.fetchWatchlistSources.mockResolvedValue({
      items: makeSources(50),
      total: 50,
      page: 1,
      size: 200,
      has_more: false
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    const initialFetchCallCount = (serviceMocks.fetchScrapedItems as Mock).mock.calls.length

    fireEvent.click(screen.getByRole("button", { name: "Refresh" }))

    await waitFor(() => {
      expect((serviceMocks.fetchScrapedItems as Mock).mock.calls.length).toBeGreaterThan(
        initialFetchCallCount
      )
    })

    const additionalCalls =
      (serviceMocks.fetchScrapedItems as Mock).mock.calls.length - initialFetchCallCount
    expect(additionalCalls).toBeLessThanOrEqual(2)
  }, 15_000)
})
