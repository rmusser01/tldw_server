// @vitest-environment jsdom

import React from "react"
import { beforeEach, describe, expect, it, vi, type Mock } from "vitest"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { ItemsTab } from "../ItemsTab/ItemsTab"
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

const navigationMocks = vi.hoisted(() => ({
  navigate: vi.fn()
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

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => navigationMocks.navigate
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

vi.mock("@/services/settings", () => ({
  setSetting: (...args: unknown[]) => settingsMocks.setSetting(...args)
}))

const makeItems = () => [
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
  },
  {
    id: 102,
    run_id: 1,
    job_id: 1,
    source_id: 1,
    url: "https://example.com/two",
    title: "Item Two",
    summary: "Summary two",
    content: "Full content of item two",
    tags: ["tech"],
    status: "ingested",
    reviewed: false,
    queued_for_briefing: false,
    created_at: "2026-02-18T08:10:00Z",
    published_at: "2026-02-18T08:10:00Z"
  },
  {
    id: 103,
    run_id: 1,
    job_id: 1,
    source_id: 1,
    url: "https://example.com/three",
    title: null,
    summary: null,
    content: null,
    tags: ["tech"],
    status: "filtered",
    reviewed: true,
    queued_for_briefing: false,
    created_at: "2026-02-18T08:20:00Z",
    published_at: "2026-02-18T08:20:00Z"
  }
]

const setupFetchScrapedItemsMock = (listItems = makeItems()) => {
  ;(serviceMocks.fetchScrapedItems as Mock).mockImplementation(
    async (params?: Record<string, unknown>) => {
      const filteredItems = listItems.filter((item) => {
        if (
          typeof params?.run_id === "number" &&
          Number.isInteger(params.run_id) &&
          item.run_id !== Number(params.run_id)
        ) {
          return false
        }
        if (
          typeof params?.reviewed === "boolean" &&
          item.reviewed !== params.reviewed
        ) {
          return false
        }
        if (
          typeof params?.status === "string" &&
          params.status.length > 0 &&
          item.status !== params.status
        ) {
          return false
        }
        if (
          typeof params?.queued_for_briefing === "boolean" &&
          Boolean(item.queued_for_briefing) !== params.queued_for_briefing
        ) {
          return false
        }
        return true
      })

      const page = Number(params?.page || 1)
      const pageSize = Number(params?.size || 25)
      if (pageSize === 1) {
        return {
          items: [],
          total: filteredItems.length,
          page,
          size: 1,
          has_more: false
        }
      }

      const start = (page - 1) * pageSize
      const pageItems = filteredItems.slice(start, start + pageSize)
      return {
        items: pageItems,
        total: filteredItems.length,
        page,
        size: pageSize,
        has_more: start + pageSize < filteredItems.length
      }
    }
  )
}

describe("ItemsTab chat handoff", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    useWatchlistsStore.getState().resetStore()
    ;(serviceMocks.fetchScrapedItemSmartCounts as Mock).mockResolvedValue({
      all: 3,
      today: 3,
      today_unread: 2,
      unread: 2,
      reviewed: 1,
      queued: 0
    })

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

    serviceMocks.fetchWatchlistSources.mockResolvedValue({
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

    serviceMocks.fetchWatchlistRuns.mockResolvedValue({
      items: [
        {
          id: 1,
          job_id: 1,
          status: "completed",
          started_at: "2026-02-18T08:00:00Z",
          finished_at: "2026-02-18T08:10:00Z",
          stats: { items_found: 3, items_ingested: 2, items_filtered: 1 }
        }
      ],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })

    setupFetchScrapedItemsMock()
    serviceMocks.updateScrapedItem.mockImplementation(async (itemId: number) => ({
      id: itemId,
      reviewed: true
    }))
    serviceMocks.createWatchlistOutput.mockResolvedValue({ id: 9001, run_id: 1 })

    settingsMocks.setSetting.mockResolvedValue(undefined)

    uiMocks.modalConfirm.mockImplementation((config: Record<string, unknown>) => {
      const onOk = config?.onOk
      if (typeof onOk === "function") {
        return Promise.resolve(onOk())
      }
      return undefined
    })
  })

  it("shows Chat button in action bar for selected item", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    // Select an item with content
    fireEvent.click(screen.getByTestId("watchlists-item-row-101"))

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-reader")).toHaveTextContent("Item One")
    })

    // The Chat button should be visible in the action bar
    const chatButton = screen.getByTestId("watchlists-item-chat-about")
    expect(chatButton).toBeInTheDocument()
    expect(chatButton).not.toBeDisabled()
  })

  it("disables Chat button when item has no usable content", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-103")).toBeInTheDocument()
    })

    // Select the item with no title, summary, or content. Selection lives on
    // the row's inner "Open update" button, not the wrapper div.
    fireEvent.click(
      within(screen.getByTestId("watchlists-item-row-103")).getByRole("button", {
        name: /open update/i
      })
    )

    await waitFor(() => {
      const chatButton = screen.getByTestId("watchlists-item-chat-about")
      expect(chatButton).toBeDisabled()
    })
  })

  it("stores handoff payload and dispatches event on Chat click", async () => {
    const dispatchSpy = vi.spyOn(window, "dispatchEvent")

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    // Ensure item 101 is selected in the reader
    fireEvent.click(screen.getByTestId("watchlists-item-row-101"))

    const chatButton = screen.getByTestId("watchlists-item-chat-about")
    fireEvent.click(chatButton)

    await waitFor(() => {
      expect(settingsMocks.setSetting).toHaveBeenCalled()
    })

    // Verify setSetting was called with the correct payload structure
    const setSettingCall = settingsMocks.setSetting.mock.calls[0]
    const payload = setSettingCall[1]
    expect(payload).toHaveProperty("articles")
    expect(payload.articles).toHaveLength(1)
    expect(payload.articles[0]).toMatchObject({
      title: "Item One",
      url: "https://example.com/one",
      sourceType: "item"
    })

    // Verify the custom event was dispatched
    const customEvents = dispatchSpy.mock.calls.filter(
      ([event]) => event instanceof CustomEvent && event.type === "tldw:discuss-watchlist"
    )
    expect(customEvents).toHaveLength(1)
    const eventDetail = (customEvents[0][0] as CustomEvent).detail
    expect(eventDetail.articles).toHaveLength(1)
    expect(eventDetail.articles[0].title).toBe("Item One")

    dispatchSpy.mockRestore()
  })

  it("navigates to root on Chat click", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-item-row-101"))

    const chatButton = screen.getByTestId("watchlists-item-chat-about")
    fireEvent.click(chatButton)

    await waitFor(() => {
      expect(navigationMocks.navigate).toHaveBeenCalledWith("/")
    })
  })

  it("shows Chat about selected with count when items are checked", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    // Check two items via their checkboxes
    fireEvent.click(screen.getByTestId("watchlists-item-select-101"))
    fireEvent.click(screen.getByTestId("watchlists-item-select-102"))

    const chatSelectedButton = screen.getByTestId("watchlists-items-chat-selected")
    expect(chatSelectedButton).not.toBeDisabled()
    expect(chatSelectedButton).toHaveTextContent("Chat about selected (2)")
  })

  it("disables Chat about selected when no items are checked", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    const chatSelectedButton = screen.getByTestId("watchlists-items-chat-selected")
    expect(chatSelectedButton).toBeDisabled()
    expect(chatSelectedButton).toHaveTextContent("Chat about selected (0)")
  })
})
