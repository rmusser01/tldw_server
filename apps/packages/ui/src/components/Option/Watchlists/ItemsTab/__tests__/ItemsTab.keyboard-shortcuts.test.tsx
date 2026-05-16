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
    created_at: "2026-02-18T08:20:00Z",
    published_at: "2026-02-18T08:20:00Z"
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
    queued_for_briefing: false,
    created_at: "2026-02-18T08:10:00Z",
    published_at: "2026-02-18T08:10:00Z"
  }
]

describe("ItemsTab keyboard shortcuts", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    useWatchlistsStore.getState().resetStore()
    serviceMocks.fetchScrapedItemSmartCounts.mockResolvedValue({
      all: 2,
      today: 2,
      today_unread: 2,
      unread: 2,
      reviewed: 0,
      queued: 0
    })

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
      items: [{ id: 1, job_id: 1, status: "completed" }],
      total: 1,
      page: 1,
      size: 200,
      has_more: false
    })
    serviceMocks.createWatchlistOutput.mockResolvedValue({ id: 1, run_id: 1 })

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
        size: Number(params?.size || 20),
        has_more: false
      }
    })

    ;(serviceMocks.updateScrapedItem as Mock).mockImplementation(async (itemId: number, payload: { reviewed?: boolean }) => {
      const existing = itemsFixture.find((item) => item.id === itemId)
      return {
        ...(existing || { id: itemId }),
        reviewed: Boolean(payload?.reviewed)
      }
    })
  })

  it("supports j/k/space/o/r/n shortcuts while ignoring typing targets", async () => {
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null)
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })
    expect(screen.getByTestId("watchlists-item-reader")).toHaveTextContent("Item One")

    fireEvent.keyDown(document, { key: "j" })
    expect(screen.getByTestId("watchlists-item-reader")).toHaveTextContent("Item Two")

    fireEvent.keyDown(document, { key: "k" })
    expect(screen.getByTestId("watchlists-item-reader")).toHaveTextContent("Item One")

    const searchInput = screen.getByPlaceholderText("Search updates...")
    searchInput.focus()
    fireEvent.keyDown(searchInput, { key: "j" })
    expect(screen.getByTestId("watchlists-item-reader")).toHaveTextContent("Item One")
    searchInput.blur()

    fireEvent.keyDown(document, { key: " " })
    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(101, { reviewed: true })
    })

    fireEvent.keyDown(document, { key: "o" })
    expect(openSpy).toHaveBeenCalledWith(
      "https://example.com/one",
      "_blank",
      "noopener,noreferrer"
    )

    const refreshCallCount = (serviceMocks.fetchScrapedItems as Mock).mock.calls.length
    fireEvent.keyDown(document, { key: "r" })
    await waitFor(() => {
      expect((serviceMocks.fetchScrapedItems as Mock).mock.calls.length).toBeGreaterThan(
        refreshCallCount
      )
    })

    fireEvent.keyDown(document, { key: "n" })
    expect(useWatchlistsStore.getState().activeTab).toBe("sources")
    expect(useWatchlistsStore.getState().sourceFormOpen).toBe(true)
  })

  it("opens the selected article only once after reader focus and review toggle interactions", async () => {
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null)
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-item-reader"))
    fireEvent.keyDown(document, { key: " " })
    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(101, { reviewed: true })
    })

    fireEvent.keyDown(document, { key: "o" })
    expect(openSpy).toHaveBeenCalledTimes(1)
    expect(openSpy).toHaveBeenCalledWith(
      "https://example.com/one",
      "_blank",
      "noopener,noreferrer"
    )
  })

  it("ignores navigation shortcuts when focus is inside contenteditable regions", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    const editable = document.createElement("div")
    editable.setAttribute("contenteditable", "true")
    editable.textContent = "Draft note"
    document.body.appendChild(editable)
    editable.focus()

    fireEvent.keyDown(editable, { key: "j" })
    expect(screen.getByTestId("watchlists-item-reader")).toHaveTextContent("Item One")

    editable.remove()
  })

  it("announces reader selection changes in a screen-reader live region", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    const liveRegion = screen.getByTestId("watchlists-items-live-region")
    expect(liveRegion).toHaveTextContent("")

    fireEvent.keyDown(document, { key: "j" })
    await waitFor(() => {
      expect(liveRegion).toHaveTextContent("Selected Item Two from Tech Daily (Unread).")
    })
  })

  it("provides row-level aria labels for list navigation context", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    expect(screen.getByTestId("watchlists-item-row-101")).toHaveAttribute(
      "aria-label",
      "Item One from Tech Daily. Unread."
    )
  })

  it("opens shortcut help panel from keyboard", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.keyDown(document, { key: "?", shiftKey: true })
    expect(await screen.findByTestId("watchlists-items-shortcuts-modal")).toBeInTheDocument()
    expect(screen.getByText("j / k")).toBeInTheDocument()
  })

  it("opens shortcut help panel from button", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-items-shortcuts-help"))
    expect(await screen.findByTestId("watchlists-items-shortcuts-modal")).toBeInTheDocument()
  })

  it("supports shortcut hint strip dismiss and restore lifecycle", async () => {
    const { unmount } = render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    expect(screen.getByTestId("watchlists-items-shortcuts-hint-strip")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("watchlists-items-shortcuts-hint-dismiss"))
    expect(screen.queryByTestId("watchlists-items-shortcuts-hint-strip")).not.toBeInTheDocument()
    expect(screen.getByTestId("watchlists-items-shortcuts-hint-restore")).toBeInTheDocument()
    expect(window.localStorage.getItem("watchlists:items:shortcuts-hint-dismissed")).toBe("1")

    unmount()
    render(<ItemsTab />)
    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })
    expect(screen.queryByTestId("watchlists-items-shortcuts-hint-strip")).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId("watchlists-items-shortcuts-hint-restore"))
    expect(screen.getByTestId("watchlists-items-shortcuts-hint-strip")).toBeInTheDocument()
    expect(window.localStorage.getItem("watchlists:items:shortcuts-hint-dismissed")).toBe("0")
  })

  it("restores focus to shortcuts button after dismissing help opened from keyboard", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    const shortcutsButton = screen.getByTestId("watchlists-items-shortcuts-help")

    fireEvent.keyDown(document, { key: "?", shiftKey: true })
    expect(await screen.findByTestId("watchlists-items-shortcuts-modal")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("watchlists-items-shortcuts-close"))

    await waitFor(() => {
      expect(shortcutsButton).toHaveFocus()
    })
  })

  it("blocks navigation shortcuts while help is open", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    const shortcutsButton = screen.getByTestId("watchlists-items-shortcuts-help")
    fireEvent.click(shortcutsButton)
    expect(await screen.findByTestId("watchlists-items-shortcuts-modal")).toBeInTheDocument()

    fireEvent.keyDown(document, { key: "j" })
    expect(screen.getByTestId("watchlists-item-reader")).toHaveTextContent("Item One")
  })

  it("keeps reader panes and triage controls available in narrow viewport", async () => {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 390
    })
    window.dispatchEvent(new Event("resize"))

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    expect(screen.getByTestId("watchlists-items-left-pane")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-items-list-pane")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-items-reader-pane")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-items-shortcuts-help")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-items-mark-page")).toBeInTheDocument()
  })
})
