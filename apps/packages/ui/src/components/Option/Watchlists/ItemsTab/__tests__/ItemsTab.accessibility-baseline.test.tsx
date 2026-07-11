// @vitest-environment jsdom

import React from "react"
import i18next from "i18next"
import { beforeAll, beforeEach, describe, expect, it, vi, type Mock } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ItemsTab } from "../ItemsTab"
import { useWatchlistsStore } from "@/store/watchlists"
import commonEn from "@/assets/locale/en/common.json"
import watchlistsEn from "@/assets/locale/en/watchlists.json"

const serviceMocks = vi.hoisted(() => ({
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
  messageError: vi.fn(),
  i18nRef: { current: null as ReturnType<typeof i18next.createInstance> | null },
  translate: (
    key: string,
    fallbackOrOptions?: string | { defaultValue?: string },
    maybeOptions?: Record<string, unknown>
  ) => {
    const fallback = typeof fallbackOrOptions === "string"
      ? fallbackOrOptions
      : fallbackOrOptions?.defaultValue
    return uiMocks.i18nRef.current?.t(key, { defaultValue: fallback, ...maybeOptions }) ?? key
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: uiMocks.translate
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
  fetchScrapedItemSmartCounts: (...args: unknown[]) =>
    serviceMocks.fetchScrapedItemSmartCounts(...args),
  fetchWatchlistSources: (...args: unknown[]) => serviceMocks.fetchWatchlistSources(...args),
  fetchWatchlistRuns: (...args: unknown[]) => serviceMocks.fetchWatchlistRuns(...args),
  fetchScrapedItems: (...args: unknown[]) => serviceMocks.fetchScrapedItems(...args),
  updateScrapedItem: (...args: unknown[]) => serviceMocks.updateScrapedItem(...args)
}))

describe("ItemsTab accessibility baseline", () => {
  beforeAll(async () => {
    const instance = i18next.createInstance()
    await instance.init({
      lng: "en",
      fallbackLng: false,
      resources: { en: { common: commonEn, watchlists: watchlistsEn } },
      ns: ["watchlists", "common"],
      defaultNS: "watchlists",
      interpolation: { escapeValue: false }
    })
    uiMocks.i18nRef.current = instance
  })

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
          name: "BBC",
          url: "https://bbc.example/rss.xml",
          source_type: "rss",
          active: true,
          tags: ["tech"],
          created_at: "2026-02-18T07:00:00Z",
          updated_at: "2026-02-18T07:00:00Z",
          last_scraped_at: "2026-02-18T08:00:00Z",
          status: "healthy"
        },
        {
          id: 2,
          name: "NPR",
          url: "https://npr.example/rss.xml",
          source_type: "rss",
          active: true,
          tags: ["news"],
          created_at: "2026-02-18T07:00:00Z",
          updated_at: "2026-02-18T07:00:00Z",
          last_scraped_at: "2026-02-18T08:00:00Z",
          status: "healthy"
        },
        {
          id: 3,
          name: "The Guardian",
          url: "https://guardian.example/rss.xml",
          source_type: "rss",
          active: true,
          tags: ["news"],
          created_at: "2026-02-18T07:00:00Z",
          updated_at: "2026-02-18T07:00:00Z",
          last_scraped_at: "2026-02-18T08:00:00Z",
          status: "healthy"
        }
      ],
      total: 3,
      page: 1,
      size: 200,
      has_more: false
    })

    ;(serviceMocks.fetchScrapedItems as Mock).mockImplementation(async (params?: Record<string, unknown>) => {
      if (params?.size === 1) {
        if (params?.reviewed === false) return { items: [], total: 2, page: 1, size: 1, has_more: false }
        if (params?.reviewed === true) return { items: [], total: 0, page: 1, size: 1, has_more: false }
        return { items: [], total: 2, page: 1, size: 1, has_more: false }
      }
      return {
        items: [
          {
            id: 101,
            run_id: 1,
            job_id: 1,
            source_id: 1,
            url: "https://example.com/one",
            title: "BBC title",
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
            source_id: 2,
            url: "https://example.com/two",
            title: "NPR title",
            summary: "Summary two",
            tags: ["tech"],
            status: "ingested",
            reviewed: false,
            created_at: "2026-02-18T08:10:00Z",
            published_at: "2026-02-18T08:10:00Z"
          },
          {
            id: 103,
            run_id: 1,
            job_id: 1,
            source_id: 3,
            url: "https://example.com/three",
            title: "Guardian title",
            summary: "Summary three",
            tags: ["news"],
            status: "ingested",
            reviewed: false,
            created_at: "2026-02-18T08:20:00Z",
            published_at: "2026-02-18T08:20:00Z"
          }
        ],
        total: 3,
        page: Number(params?.page || 1),
        size: Number(params?.size || 25),
        has_more: false
      }
    })

    ;(serviceMocks.updateScrapedItem as Mock).mockImplementation(async (itemId: number) => ({
      id: itemId,
      reviewed: true
    }))
  })

  it("exposes explicit text status labels for row state changes", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    expect(screen.getByTestId("watchlists-item-row-review-state-101")).toHaveTextContent("Unread")
    expect(screen.getByTestId("watchlists-item-row-review-state-102")).toHaveTextContent("Unread")

    fireEvent.click(screen.getByRole("button", { name: "Open update: BBC title" }))
    fireEvent.click(screen.getByRole("button", { name: "Mark as reviewed" }))

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-review-state-101")).toHaveTextContent(
        "Reviewed"
      )
    })
  }, 15_000)

  it("keeps primary triage controls keyboard-discoverable by accessible name", async () => {
    render(<ItemsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-item-row-101")).toBeInTheDocument()
    })

    expect(
      screen.getByRole("button", { name: "Mark selected as reviewed" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Mark page as reviewed" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Mark all filtered updates" })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Shortcuts" })).toBeInTheDocument()

    expect(screen.getByRole("complementary", { name: "Feed filters" })).toBeInTheDocument()
    expect(screen.getByRole("region", { name: "Feeds list" })).toBeInTheDocument()
    expect(
      screen.getByRole("region", { name: "Updates list and triage controls" })
    ).toBeInTheDocument()
    expect(screen.getByRole("region", { name: "Updates list" })).toBeInTheDocument()
    expect(screen.getByRole("region", { name: "Update reader" })).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Open update: BBC title" })
    ).toBeInTheDocument()
  }, 15_000)

  it("binds update actions and selections to each record title", async () => {
    render(<ItemsTab />)

    await screen.findByTestId("watchlists-item-row-103")

    for (const title of ["BBC title", "NPR title", "Guardian title"]) {
      expect(screen.getByRole("button", { name: `Open update: ${title}` })).toBeVisible()
      expect(screen.getByRole("checkbox", { name: `Select update: ${title}` })).toBeInTheDocument()
    }
  }, 15_000)

  it("separates multi-record selection from native click and keyboard open actions", async () => {
    const user = userEvent.setup()
    render(<ItemsTab />)

    await screen.findByTestId("watchlists-item-row-103")

    for (const title of ["BBC title", "NPR title", "Guardian title"]) {
      const row = screen.getByTestId(
        `watchlists-item-row-${title === "BBC title" ? 101 : title === "NPR title" ? 102 : 103}`
      )
      const checkbox = screen.getByRole("checkbox", { name: `Select update: ${title}` })
      const openButton = screen.getByRole("button", { name: `Open update: ${title}` })

      expect(row.tagName).toBe("DIV")
      expect(checkbox.closest("button")).toBeNull()
      expect(openButton).toHaveAttribute("type", "button")
    }

    await user.click(screen.getByRole("button", { name: "Open update: BBC title" }))
    expect(screen.getByRole("button", { name: "Open update: BBC title" })).toHaveAttribute(
      "aria-pressed",
      "true"
    )

    const nprOpen = screen.getByRole("button", { name: "Open update: NPR title" })
    nprOpen.focus()
    await user.keyboard("{Enter}")
    await waitFor(() => expect(nprOpen).toHaveAttribute("aria-pressed", "true"))

    const guardianOpen = screen.getByRole("button", { name: "Open update: Guardian title" })
    expect(guardianOpen.tagName).toBe("BUTTON")
    expect(guardianOpen).not.toHaveAttribute("role")
    expect(guardianOpen).not.toHaveAttribute("tabindex")
  }, 15_000)
})
