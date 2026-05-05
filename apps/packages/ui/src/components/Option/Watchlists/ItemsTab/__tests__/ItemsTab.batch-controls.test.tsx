// @vitest-environment jsdom

import React from "react"
import { beforeEach, describe, expect, it, vi, type Mock } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { ItemsTab } from "../ItemsTab"
import { useWatchlistsStore } from "@/store/watchlists"
import {
  ITEMS_PAGE_SIZE_STORAGE_KEY,
  ITEMS_SORT_MODE_STORAGE_KEY,
  ITEMS_VIEW_PRESETS_STORAGE_KEY
} from "../items-utils"

const serviceMocks = vi.hoisted(() => ({
  createWatchlistOutput: vi.fn(),
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
  createWatchlistOutput: (...args: unknown[]) => serviceMocks.createWatchlistOutput(...args),
  fetchScrapedItemSmartCounts: (...args: unknown[]) =>
    serviceMocks.fetchScrapedItemSmartCounts(...args),
  fetchWatchlistSources: (...args: unknown[]) => serviceMocks.fetchWatchlistSources(...args),
  fetchWatchlistRuns: (...args: unknown[]) => serviceMocks.fetchWatchlistRuns(...args),
  fetchScrapedItems: (...args: unknown[]) => serviceMocks.fetchScrapedItems(...args),
  updateScrapedItem: (...args: unknown[]) => serviceMocks.updateScrapedItem(...args)
}))

const makeItems = () => ([
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
  },
  {
    id: 103,
    run_id: 1,
    job_id: 1,
    source_id: 1,
    url: "https://example.com/three",
    title: "Item Three",
    summary: "Summary three",
    tags: ["tech"],
    status: "filtered",
    reviewed: true,
    queued_for_briefing: false,
    created_at: "2026-02-18T08:20:00Z",
    published_at: "2026-02-18T08:20:00Z"
  }
])

const createDeferred = <TValue,>() => {
  let resolve!: (value: TValue | PromiseLike<TValue>) => void
  const promise = new Promise<TValue>((res) => {
    resolve = res
  })
  return { promise, resolve }
}

const setupFetchScrapedItemsMock = (listItems = makeItems()) => {
  ;(serviceMocks.fetchScrapedItems as Mock).mockImplementation(async (params?: Record<string, unknown>) => {
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
  })
}

describe("ItemsTab batch throughput controls", () => {
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

    uiMocks.modalConfirm.mockImplementation((config: Record<string, unknown>) => {
      const onOk = config?.onOk
      if (typeof onOk === "function") {
        return Promise.resolve(onOk())
      }
      return undefined
    })
  })

  it("enables mark-selected only when selection exists and confirms before applying", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    expect(screen.getByTestId("watchlists-item-row-review-state-101")).toHaveTextContent("Unread")
    expect(screen.getByTestId("watchlists-item-row-review-state-103")).toHaveTextContent("Reviewed")

    expect(screen.getByTestId("watchlists-items-batch-scope-summary")).toHaveTextContent(
      "Selected: 0 unread"
    )

    const markSelectedButton = screen.getByTestId("watchlists-items-mark-selected")
    expect(markSelectedButton).toBeDisabled()
    expect(screen.getByTestId("watchlists-items-selected-count")).toHaveTextContent("0 selected")

    fireEvent.click(screen.getByTestId("watchlists-item-select-101"))
    expect(markSelectedButton).not.toBeDisabled()
    expect(screen.getByTestId("watchlists-items-selected-count")).toHaveTextContent("1 selected")

    fireEvent.click(markSelectedButton)

    await waitFor(() => {
      expect(uiMocks.modalConfirm).toHaveBeenCalled()
    })

    const confirmConfig = uiMocks.modalConfirm.mock.calls[0]?.[0] as Record<string, unknown>
    expect(confirmConfig?.title).toBe("Mark selected items as reviewed?")
    expect(confirmConfig?.content).toBe(
      "Scope: selected item. This will mark 1 item as reviewed."
    )

    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(101, { reviewed: true })
    })
    expect(uiMocks.messageSuccess).toHaveBeenCalledWith("Marked 1 selected item as reviewed.")
  })

  it("respects persisted page-size, supports mark-page, and persists changed size", async () => {
    window.localStorage.setItem(ITEMS_PAGE_SIZE_STORAGE_KEY, "50")
    render(<ItemsTab />)

    await waitFor(() => {
      expect(serviceMocks.fetchScrapedItems).toHaveBeenCalled()
    })

    expect(serviceMocks.fetchScrapedItems).toHaveBeenCalledWith(
      expect.objectContaining({
        page: 1,
        size: 50
      })
    )

    fireEvent.click(screen.getByTestId("watchlists-items-mark-page"))
    await waitFor(() => {
      expect(uiMocks.modalConfirm).toHaveBeenCalled()
    })

    const confirmConfig = uiMocks.modalConfirm.mock.calls[0]?.[0] as Record<string, unknown>
    expect(confirmConfig?.title).toBe("Mark this page as reviewed?")
    expect(confirmConfig?.content).toBe(
      "Scope: items on this page. This will mark 2 items as reviewed."
    )

    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(101, { reviewed: true })
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(102, { reviewed: true })
    })
    expect(uiMocks.messageSuccess).toHaveBeenCalledWith(
      "Marked 2 items on this page as reviewed."
    )

    const pageSizeSelect = screen.getByTestId("watchlists-items-page-size-select")
    fireEvent.mouseDown(pageSizeSelect)
    fireEvent.click(await screen.findByText("20 / page"))

    await waitFor(() => {
      expect(serviceMocks.fetchScrapedItems).toHaveBeenCalledWith(
        expect.objectContaining({
          page: 1,
          size: 20
        })
      )
    })

    expect(window.localStorage.getItem(ITEMS_PAGE_SIZE_STORAGE_KEY)).toBe("20")
  })

  it("supports all-filtered batch confirmation scope and applies unread-only updates", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-items-mark-all-filtered"))

    await waitFor(() => {
      expect(uiMocks.modalConfirm).toHaveBeenCalled()
    })

    const confirmConfig = uiMocks.modalConfirm.mock.calls[0]?.[0] as Record<string, unknown>
    expect(confirmConfig?.title).toBe("Mark all filtered items as reviewed?")
    expect(confirmConfig?.content).toBe(
      "Scope: all filtered items. This will mark 2 items as reviewed."
    )

    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(101, { reviewed: true })
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(102, { reviewed: true })
    })
    expect(serviceMocks.updateScrapedItem).not.toHaveBeenCalledWith(103, { reviewed: true })
    expect(uiMocks.messageSuccess).toHaveBeenCalledWith(
      "Marked 2 all filtered items as reviewed."
    )
  })

  it("shows batch progress state while updates are running and keeps terminal summary visible", async () => {
    const pendingById = new Map<number, ReturnType<typeof createDeferred<{ id: number; reviewed: boolean }>>>()
    pendingById.set(101, createDeferred<{ id: number; reviewed: boolean }>())
    pendingById.set(102, createDeferred<{ id: number; reviewed: boolean }>())

    serviceMocks.updateScrapedItem.mockImplementation((itemId: number) => {
      const pending = pendingById.get(itemId)
      if (!pending) throw new Error(`missing deferred for ${itemId}`)
      return pending.promise
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-items-mark-page"))
    await waitFor(() => {
      expect(uiMocks.modalConfirm).toHaveBeenCalled()
    })

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-items-batch-progress")).toBeInTheDocument()
      expect(screen.getByTestId("watchlists-items-batch-progress-count")).toHaveTextContent("0 / 2")
    })

    pendingById.get(101)?.resolve({ id: 101, reviewed: true })
    pendingById.get(102)?.resolve({ id: 102, reviewed: true })

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-items-batch-progress-count")).toHaveTextContent("2 / 2")
      expect(screen.getByTestId("watchlists-items-batch-progress-summary")).toHaveTextContent(
        "Batch review complete: 2 succeeded, 0 failed."
      )
    })
  })

  it("supports saved view preset save, apply, and delete", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-items-smart-feed-reviewed"))
    await waitFor(() => {
      expect(useWatchlistsStore.getState().itemsSmartFilter).toBe("reviewed")
    })

    const sortSelect = screen.getByTestId("watchlists-items-sort-select")
    fireEvent.mouseDown(sortSelect)
    fireEvent.click(await screen.findByText("Unread first"))

    const searchInput = screen.getByPlaceholderText("Search feed items...")
    fireEvent.change(searchInput, { target: { value: "alpha" } })

    fireEvent.click(screen.getByTestId("watchlists-items-view-save"))
    const nameInput = await screen.findByTestId("watchlists-items-view-name-input")
    fireEvent.change(nameInput, { target: { value: "Triage Alpha" } })
    const saveButtons = screen.getAllByRole("button", { name: "Save view" })
    fireEvent.click(saveButtons[saveButtons.length - 1])

    await waitFor(() => {
      const raw = window.localStorage.getItem(ITEMS_VIEW_PRESETS_STORAGE_KEY)
      expect(raw).toContain("Triage Alpha")
      expect(raw).toContain("\"sortMode\":\"unreadFirst\"")
    })

    fireEvent.click(screen.getByTestId("watchlists-items-smart-feed-all"))
    await waitFor(() => {
      expect(useWatchlistsStore.getState().itemsSmartFilter).toBe("all")
    })

    fireEvent.change(searchInput, { target: { value: "beta" } })
    fireEvent.mouseDown(sortSelect)
    fireEvent.click(await screen.findByText("Newest first"))

    const presetsSelect = screen.getByTestId("watchlists-items-view-presets-select")
    fireEvent.mouseDown(presetsSelect)
    fireEvent.click(await screen.findByText("Triage Alpha"))
    expect(searchInput).toHaveValue("alpha")
    expect(sortSelect).toHaveTextContent("Unread first")
    expect(useWatchlistsStore.getState().itemsSmartFilter).toBe("reviewed")

    fireEvent.click(screen.getByTestId("watchlists-items-view-delete"))
    await waitFor(() => {
      const raw = window.localStorage.getItem(ITEMS_VIEW_PRESETS_STORAGE_KEY)
      expect(raw).toContain("system-unread-today")
      expect(raw).toContain("system-high-priority")
      expect(raw).toContain("system-needs-review")
      expect(raw).not.toContain("Triage Alpha")
    })
  }, 20_000)

  it("transitions smart feed filters and requests matching reviewed states", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-items-smart-feed-reviewed"))
    expect(useWatchlistsStore.getState().itemsSmartFilter).toBe("reviewed")

    await waitFor(() => {
      expect(
        (serviceMocks.fetchScrapedItems as Mock).mock.calls.some(([params]) =>
          params?.reviewed === true && params?.size === 25
        )
      ).toBe(true)
    })

    fireEvent.click(screen.getByTestId("watchlists-items-smart-feed-unread"))
    expect(useWatchlistsStore.getState().itemsSmartFilter).toBe("unread")

    await waitFor(() => {
      expect(
        (serviceMocks.fetchScrapedItems as Mock).mock.calls.some(([params]) =>
          params?.reviewed === false && params?.size === 25
        )
      ).toBe(true)
    })
  })

  it("supports triage sort changes and persists sort preference", async () => {
    setupFetchScrapedItemsMock([
      {
        id: 201,
        run_id: 1,
        job_id: 1,
        source_id: 1,
        url: "https://example.com/reviewed-newest",
        title: "Reviewed newest",
        summary: "summary",
        tags: ["tech"],
        status: "ingested",
        reviewed: true,
        queued_for_briefing: false,
        created_at: "2026-02-18T08:20:00Z",
        published_at: "2026-02-18T08:20:00Z"
      },
      {
        id: 202,
        run_id: 1,
        job_id: 1,
        source_id: 1,
        url: "https://example.com/unread-mid",
        title: "Unread mid",
        summary: "summary",
        tags: ["tech"],
        status: "ingested",
        reviewed: false,
        queued_for_briefing: false,
        created_at: "2026-02-18T08:10:00Z",
        published_at: "2026-02-18T08:10:00Z"
      },
      {
        id: 203,
        run_id: 1,
        job_id: 1,
        source_id: 1,
        url: "https://example.com/unread-oldest",
        title: "Unread oldest",
        summary: "summary",
        tags: ["tech"],
        status: "ingested",
        reviewed: false,
        queued_for_briefing: false,
        created_at: "2026-02-18T08:00:00Z",
        published_at: "2026-02-18T08:00:00Z"
      }
    ])

    render(<ItemsTab />)

    const orderedRowIds = () =>
      screen.getAllByTestId(/^watchlists-item-row-\d+$/).map((node) =>
        Number(String(node.getAttribute("data-testid") || "").replace("watchlists-item-row-", ""))
      )

    await waitFor(() => {
      expect(orderedRowIds()).toEqual([201, 202, 203])
    })

    const sortSelect = screen.getByTestId("watchlists-items-sort-select")
    fireEvent.mouseDown(sortSelect)
    fireEvent.click(await screen.findByText("Unread first"))

    await waitFor(() => {
      expect(orderedRowIds()).toEqual([202, 203, 201])
    })
    expect(window.localStorage.getItem(ITEMS_SORT_MODE_STORAGE_KEY)).toBe("unreadFirst")

    fireEvent.mouseDown(sortSelect)
    fireEvent.click(await screen.findByText("Oldest first"))

    await waitFor(() => {
      expect(orderedRowIds()).toEqual([203, 202, 201])
    })
    expect(window.localStorage.getItem(ITEMS_SORT_MODE_STORAGE_KEY)).toBe("oldest")
  })

  it("reconciles partial batch-review failures without losing successful row updates", async () => {
    serviceMocks.updateScrapedItem.mockImplementation(async (itemId: number) => {
      if (itemId === 102) {
        throw new Error("transient failure")
      }
      return { id: itemId, reviewed: true }
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-items-mark-page"))

    await waitFor(() => {
      expect(uiMocks.modalConfirm).toHaveBeenCalled()
    })

    await waitFor(() => {
      expect(uiMocks.messageWarning).toHaveBeenCalledWith(
        "Marked 1 items on this page as reviewed; 1 failed."
      )
    })

    expect(screen.getByTestId("watchlists-item-row-review-state-101")).toHaveTextContent("Reviewed")
    expect(screen.getByTestId("watchlists-item-row-review-state-102")).toHaveTextContent("Unread")
    expect(screen.getByTestId("watchlists-items-batch-retry-failed")).toHaveTextContent(
      "Retry 1 failed"
    )
  })

  it("retries failed batch updates from the recovery entrypoint", async () => {
    const failOnce = new Set([102])
    serviceMocks.updateScrapedItem.mockImplementation(async (itemId: number) => {
      if (failOnce.has(itemId)) {
        failOnce.delete(itemId)
        throw new Error("transient failure")
      }
      return { id: itemId, reviewed: true }
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-items-mark-page"))
    await waitFor(() => {
      expect(screen.getByTestId("watchlists-items-batch-retry-failed")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-items-batch-retry-failed"))

    await waitFor(() => {
      const callsFor102 = serviceMocks.updateScrapedItem.mock.calls.filter(
        ([itemId]) => Number(itemId) === 102
      )
      expect(callsFor102).toHaveLength(2)
      expect(screen.getByTestId("watchlists-item-row-review-state-102")).toHaveTextContent("Reviewed")
      expect(screen.queryByTestId("watchlists-items-batch-retry-failed")).not.toBeInTheDocument()
    })
  })

  it("paginates all-filtered review lookups for thousand-item datasets", async () => {
    const highVolumeItems = Array.from({ length: 1200 }, (_item, index) => {
      const id = index + 1
      const minute = String(index % 60).padStart(2, "0")
      return {
        id,
        run_id: 1,
        job_id: 1,
        source_id: 1,
        url: `https://example.com/high-volume-${id}`,
        title: `High Volume Item ${id}`,
        summary: "summary",
        tags: ["tech"],
        status: "ingested",
        reviewed: false,
        created_at: `2026-02-18T08:${minute}:00Z`,
        published_at: `2026-02-18T08:${minute}:00Z`
      }
    })

    ;(serviceMocks.fetchScrapedItems as Mock).mockImplementation(async (params?: Record<string, unknown>) => {
      const page = Number(params?.page || 1)
      const size = Number(params?.size || 25)

      if (size === 1) {
        if (params?.reviewed === false) return { items: [], total: highVolumeItems.length, page: 1, size: 1, has_more: false }
        if (params?.reviewed === true) return { items: [], total: 0, page: 1, size: 1, has_more: false }
        return { items: [], total: highVolumeItems.length, page: 1, size: 1, has_more: false }
      }

      if (params?.reviewed === false && size === 200) {
        const start = (page - 1) * size
        return {
          items: highVolumeItems.slice(start, start + size),
          total: highVolumeItems.length,
          page,
          size,
          has_more: start + size < highVolumeItems.length
        }
      }

      const start = (page - 1) * size
      return {
        items: highVolumeItems.slice(start, start + size),
        total: highVolumeItems.length,
        page,
        size,
        has_more: start + size < highVolumeItems.length
      }
    })

    ;(serviceMocks.updateScrapedItem as Mock).mockImplementation(async (itemId: number) => ({
      id: itemId,
      reviewed: true
    }))

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-1")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-items-mark-all-filtered"))

    await waitFor(() => {
      expect(uiMocks.modalConfirm).toHaveBeenCalled()
    })

    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledTimes(1200)
    }, { timeout: 15000 })

    const lookupCalls = (serviceMocks.fetchScrapedItems as Mock).mock.calls
      .map((entry) => entry[0] as Record<string, unknown>)
      .filter((params) => params?.reviewed === false && Number(params?.size) === 200)

    expect(lookupCalls).toHaveLength(6)
    expect(lookupCalls.map((params) => Number(params.page))).toEqual([1, 2, 3, 4, 5, 6])
    expect(uiMocks.messageSuccess).toHaveBeenCalledWith("Marked 1200 all filtered items as reviewed.")
  })

  it("supports item handoff to monitor/run/reports and briefing queue action", async () => {
    serviceMocks.updateScrapedItem.mockImplementation(async (itemId: number, updates?: Record<string, unknown>) => {
      const item = makeItems().find((entry) => entry.id === itemId)
      return {
        ...(item || { id: itemId }),
        reviewed: Boolean(updates?.reviewed ?? item?.reviewed),
        status: typeof updates?.status === "string" ? updates.status : item?.status,
        queued_for_briefing:
          typeof updates?.queued_for_briefing === "boolean"
            ? updates.queued_for_briefing
            : item?.queued_for_briefing
      }
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-103")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-item-row-103"))
    fireEvent.click(screen.getByTestId("watchlists-item-include-briefing"))

    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(103, { queued_for_briefing: true })
    })
    expect(uiMocks.messageSuccess).toHaveBeenCalledWith("Added to the next briefing queue.")

    fireEvent.click(screen.getByTestId("watchlists-item-jump-monitor"))
    expect(useWatchlistsStore.getState().activeTab).toBe("jobs")
    expect(useWatchlistsStore.getState().jobFormOpen).toBe(true)
    expect(useWatchlistsStore.getState().jobFormEditId).toBe(1)

    fireEvent.click(screen.getByTestId("watchlists-item-jump-run"))
    expect(useWatchlistsStore.getState().activeTab).toBe("runs")
    expect(useWatchlistsStore.getState().runsJobFilter).toBe(1)
    expect(useWatchlistsStore.getState().runDetailOpen).toBe(true)
    expect(useWatchlistsStore.getState().selectedRunId).toBe(1)

    fireEvent.click(screen.getByTestId("watchlists-item-jump-outputs"))
    expect(useWatchlistsStore.getState().activeTab).toBe("outputs")
    expect(useWatchlistsStore.getState().outputsJobFilter).toBe(1)
    expect(useWatchlistsStore.getState().outputsRunFilter).toBe(1)
  })

  it("keeps queue action stateful as items move in and out of briefing queue", async () => {
    serviceMocks.updateScrapedItem.mockImplementation(async (itemId: number, updates?: Record<string, unknown>) => {
      const item = makeItems().find((entry) => entry.id === itemId)
      return {
        ...(item || { id: itemId }),
        reviewed: Boolean(updates?.reviewed ?? item?.reviewed),
        status: typeof updates?.status === "string" ? updates.status : item?.status,
        queued_for_briefing:
          typeof updates?.queued_for_briefing === "boolean"
            ? updates.queued_for_briefing
            : item?.queued_for_briefing
      }
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-item-row-101"))
    expect(screen.getByTestId("watchlists-item-include-briefing")).toHaveTextContent(
      "Include in next briefing"
    )

    fireEvent.click(screen.getByTestId("watchlists-item-row-103"))
    const includeButton = screen.getByTestId("watchlists-item-include-briefing")
    fireEvent.click(includeButton)

    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(103, { queued_for_briefing: true })
    })
    expect(uiMocks.messageSuccess).toHaveBeenCalledWith("Added to the next briefing queue.")
    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-include-briefing")).toHaveTextContent(
        "Remove from briefing queue"
      )
    })

    fireEvent.click(screen.getByTestId("watchlists-item-include-briefing"))
    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(103, { queued_for_briefing: false })
    })
    expect(uiMocks.messageSuccess).toHaveBeenCalledWith("Removed from the briefing queue.")
  })

  it("supports explicit queued view and run-scoped report generation", async () => {
    const queueItems = [
      {
        ...makeItems()[0],
        id: 101,
        run_id: 1,
        queued_for_briefing: true
      },
      {
        ...makeItems()[1],
        id: 102,
        run_id: 1,
        queued_for_briefing: false
      },
      {
        ...makeItems()[2],
        id: 201,
        run_id: 2,
        queued_for_briefing: true
      }
    ]
    setupFetchScrapedItemsMock(queueItems)
    serviceMocks.fetchWatchlistRuns.mockResolvedValue({
      items: [
        { id: 1, job_id: 1, status: "completed" },
        { id: 2, job_id: 1, status: "completed" }
      ],
      total: 2,
      page: 1,
      size: 200,
      has_more: false
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-items-smart-feed-queued"))

    await waitFor(() => {
      const queuedRequests = (serviceMocks.fetchScrapedItems as Mock).mock.calls
        .map((call) => call[0] as Record<string, unknown>)
        .filter((params) => params?.queued_for_briefing === true && params?.size !== 1)
      expect(
        queuedRequests.some((params) => params.run_id === 1)
      ).toBe(true)
    })

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })
    expect(screen.queryByTestId("watchlists-item-row-201")).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId("watchlists-items-queue-generate-report"))

    await waitFor(() => {
      expect(serviceMocks.createWatchlistOutput).toHaveBeenCalledWith({
        run_id: 1,
        item_ids: [101]
      })
    })
    expect(uiMocks.messageSuccess).toHaveBeenCalledWith("Created report from 1 queued item.")
    expect(useWatchlistsStore.getState().activeTab).toBe("outputs")
    expect(useWatchlistsStore.getState().outputsRunFilter).toBe(1)
  })

  it("marks all filtered items as reviewed with scoped confirmation messaging", async () => {
    const largeUnreadSet = Array.from({ length: 45 }, (_value, index) => ({
      id: 500 + index,
      run_id: 9,
      job_id: 3,
      source_id: 1,
      url: `https://example.com/${500 + index}`,
      title: `Bulk Item ${500 + index}`,
      summary: `Summary ${500 + index}`,
      tags: ["ops"],
      status: "ingested",
      reviewed: false,
      queued_for_briefing: false,
      created_at: "2026-02-18T10:00:00Z",
      published_at: "2026-02-18T10:00:00Z"
    }))
    setupFetchScrapedItemsMock(largeUnreadSet)
    serviceMocks.updateScrapedItem.mockImplementation(async (itemId: number) => ({
      id: itemId,
      reviewed: true
    }))

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-500")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-items-mark-all-filtered"))

    await waitFor(() => {
      expect(uiMocks.modalConfirm).toHaveBeenCalled()
    })
    const confirmConfig = uiMocks.modalConfirm.mock.calls[0]?.[0] as Record<string, unknown>
    expect(confirmConfig?.title).toBe("Mark all filtered items as reviewed?")
    expect(confirmConfig?.content).toBe(
      "Scope: all filtered items. This will mark 45 items as reviewed."
    )

    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledTimes(45)
    })
    expect(uiMocks.messageSuccess).toHaveBeenCalledWith(
      "Marked 45 all filtered items as reviewed."
    )
  })

  it("shows partial-failure feedback and keeps failed selections for retry", async () => {
    serviceMocks.updateScrapedItem.mockImplementation(async (itemId: number) => {
      if (itemId === 102) {
        throw new Error("update failed")
      }
      return { id: itemId, reviewed: true }
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-item-select-101"))
    fireEvent.click(screen.getByTestId("watchlists-item-select-102"))
    fireEvent.click(screen.getByTestId("watchlists-items-mark-selected"))

    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(101, { reviewed: true })
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(102, { reviewed: true })
    })

    expect(uiMocks.messageWarning).toHaveBeenCalledWith(
      "Marked 1 selected item as reviewed; 1 failed."
    )
    expect(screen.getByTestId("watchlists-item-row-review-state-101")).toHaveTextContent("Reviewed")
    expect(screen.getByTestId("watchlists-item-row-review-state-102")).toHaveTextContent("Unread")
    expect(screen.getByTestId("watchlists-items-selected-count")).toHaveTextContent("1 selected")
  })

  it("handles high-volume all-filtered review operations", async () => {
    const largeUnreadSet = Array.from({ length: 240 }, (_value, index) => ({
      id: 800 + index,
      run_id: 9,
      job_id: 3,
      source_id: 1,
      url: `https://example.com/${800 + index}`,
      title: `Large Item ${800 + index}`,
      summary: `Summary ${800 + index}`,
      tags: ["ops"],
      status: "ingested",
      reviewed: false,
      queued_for_briefing: false,
      created_at: "2026-02-18T10:00:00Z",
      published_at: "2026-02-18T10:00:00Z"
    }))
    setupFetchScrapedItemsMock(largeUnreadSet)
    serviceMocks.updateScrapedItem.mockImplementation(async (itemId: number) => ({
      id: itemId,
      reviewed: true
    }))

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-800")).toBeInTheDocument()
    })

    const start = performance.now()
    fireEvent.click(screen.getByTestId("watchlists-items-mark-all-filtered"))

    await waitFor(
      () => {
        expect(serviceMocks.updateScrapedItem).toHaveBeenCalledTimes(240)
      },
      { timeout: 15000 }
    )
    expect(uiMocks.messageSuccess).toHaveBeenCalledWith(
      "Marked 240 all filtered items as reviewed."
    )
    expect(performance.now() - start).toBeLessThan(15000)
  })

  it("surfaces in-progress batch status and completion summary for high-volume operations", async () => {
    const largeUnreadSet = Array.from({ length: 45 }, (_value, index) => ({
      id: 900 + index,
      run_id: 9,
      job_id: 3,
      source_id: 1,
      url: `https://example.com/${900 + index}`,
      title: `Progress Item ${900 + index}`,
      summary: `Summary ${900 + index}`,
      tags: ["ops"],
      status: "ingested",
      reviewed: false,
      queued_for_briefing: false,
      created_at: "2026-02-18T10:00:00Z",
      published_at: "2026-02-18T10:00:00Z"
    }))
    setupFetchScrapedItemsMock(largeUnreadSet)

    let callCount = 0
    const firstChunkResolvers: Array<() => void> = []
    serviceMocks.updateScrapedItem.mockImplementation((itemId: number) => {
      callCount += 1
      if (callCount <= 20) {
        return new Promise((resolve) => {
          firstChunkResolvers.push(() => resolve({ id: itemId, reviewed: true }))
        })
      }
      return Promise.resolve({ id: itemId, reviewed: true })
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-900")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-items-mark-all-filtered"))

    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledTimes(20)
    })

    expect(screen.getByTestId("watchlists-items-batch-progress-panel")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-items-batch-progress-count")).toHaveTextContent("0 / 45")
    expect(screen.getByTestId("watchlists-items-batch-progress-summary")).toHaveTextContent(
      "Processing 0 of 45 items"
    )

    firstChunkResolvers.forEach((resolve) => resolve())

    await waitFor(
      () => {
        expect(serviceMocks.updateScrapedItem).toHaveBeenCalledTimes(45)
      },
      { timeout: 15000 }
    )
    await waitFor(() => {
      expect(screen.getByTestId("watchlists-items-batch-progress-summary")).toHaveTextContent(
        "Batch review complete: 45 succeeded, 0 failed."
      )
    })
  })

  it("offers retry for failed batch items and reconciles selection after successful retry", async () => {
    let shouldFail102 = true
    serviceMocks.updateScrapedItem.mockImplementation(async (itemId: number) => {
      if (itemId === 102 && shouldFail102) {
        shouldFail102 = false
        throw new Error("update failed")
      }
      return { id: itemId, reviewed: true }
    })

    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-item-select-101"))
    fireEvent.click(screen.getByTestId("watchlists-item-select-102"))
    fireEvent.click(screen.getByTestId("watchlists-items-mark-selected"))

    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(101, { reviewed: true })
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(102, { reviewed: true })
    })

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-items-batch-progress-summary")).toHaveTextContent(
        "Batch review complete: 1 succeeded, 1 failed."
      )
    })

    const retryButton = screen.getByTestId("watchlists-items-batch-retry-failed")
    fireEvent.click(retryButton)

    await waitFor(() => {
      expect(serviceMocks.updateScrapedItem).toHaveBeenCalledWith(102, { reviewed: true })
    })
    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-review-state-102")).toHaveTextContent(
        "Reviewed"
      )
    })

    expect(screen.getByTestId("watchlists-items-selected-count")).toHaveTextContent("0 selected")
    expect(screen.getByTestId("watchlists-items-batch-progress-summary")).toHaveTextContent(
      "Batch review complete: 1 succeeded, 0 failed."
    )
  })
})
