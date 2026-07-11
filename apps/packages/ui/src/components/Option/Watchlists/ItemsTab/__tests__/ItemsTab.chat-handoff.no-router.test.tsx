// @vitest-environment jsdom

import React from "react"
import { beforeEach, describe, expect, it, vi, type Mock } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
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

const settingsMocks = vi.hoisted(() => ({
  setSetting: vi.fn()
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

vi.mock("@/services/watchlists", () => ({
  createWatchlistOutput: (...args: unknown[]) => serviceMocks.createWatchlistOutput(...args),
  fetchScrapedItemSmartCounts: (...args: unknown[]) =>
    serviceMocks.fetchScrapedItemSmartCounts(...args),
  fetchWatchlistSources: (...args: unknown[]) => serviceMocks.fetchWatchlistSources(...args),
  fetchWatchlistRuns: (...args: unknown[]) => serviceMocks.fetchWatchlistRuns(...args),
  fetchScrapedItems: (...args: unknown[]) => serviceMocks.fetchScrapedItems(...args),
  updateScrapedItem: (...args: unknown[]) => serviceMocks.updateScrapedItem(...args)
}))

vi.mock("@/services/settings", () => ({
  setSetting: (...args: unknown[]) => settingsMocks.setSetting(...args)
}))

describe("ItemsTab chat handoff without router context", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    useWatchlistsStore.getState().resetStore()
    window.location.hash = "#/watchlists"

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
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

    ;(serviceMocks.fetchScrapedItemSmartCounts as Mock).mockResolvedValue({
      all: 1,
      today: 1,
      today_unread: 1,
      unread: 1,
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

    ;(serviceMocks.fetchWatchlistSources as Mock).mockResolvedValue({
      items: [
        {
          id: 1,
          name: "Tech Daily",
          url: "https://example.com/rss.xml",
          source_type: "rss",
          active: true,
          tags: ["tech"],
          created_at: "2026-02-18T07:00:00Z",
          updated_at: "2026-02-18T07:00:00Z",
          last_scraped_at: "2026-02-18T08:00:00Z",
          status: "healthy"
        }
      ],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })

    ;(serviceMocks.fetchScrapedItems as Mock).mockImplementation(async (params?: Record<string, unknown>) => {
      const items = [
        {
          id: 101,
          run_id: 1,
          job_id: 1,
          source_id: 1,
          url: "https://example.com/one",
          title: "Item One",
          summary: "Summary one",
          content: "Full content of item one",
          tags: ["tech"],
          status: "ingested",
          reviewed: false,
          queued_for_briefing: false,
          created_at: "2026-02-18T08:00:00Z",
          published_at: "2026-02-18T08:00:00Z"
        }
      ]

      if (params?.size === 1) {
        return { items: [], total: items.length, page: 1, size: 1, has_more: false }
      }

      return {
        items,
        total: items.length,
        page: Number(params?.page || 1),
        size: Number(params?.size || 25),
        has_more: false
      }
    })

    ;(serviceMocks.updateScrapedItem as Mock).mockResolvedValue({})
    ;(settingsMocks.setSetting as Mock).mockResolvedValue(undefined)
  })

  it("falls back to hash navigation for chat handoff when no router is mounted", async () => {
    const dispatchSpy = vi.spyOn(window, "dispatchEvent")

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Open update: Item One" }))
    fireEvent.click(screen.getByTestId("watchlists-item-chat-about"))

    await waitFor(() => {
      expect(settingsMocks.setSetting).toHaveBeenCalled()
    })

    expect(window.location.hash).toBe("#/")
    expect(
      dispatchSpy.mock.calls.some(
        ([event]) => event instanceof CustomEvent && event.type === "tldw:discuss-watchlist"
      )
    ).toBe(true)

    dispatchSpy.mockRestore()
  })
})
