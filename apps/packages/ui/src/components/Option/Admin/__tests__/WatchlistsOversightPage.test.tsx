// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const apiMock = vi.hoisted(() => ({
  listAdminUsers: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

const watchlistsMock = vi.hoisted(() => ({
  fetchWatchlistSources: vi.fn(),
  fetchScrapedItems: vi.fn(),
  fetchWatchlistRuns: vi.fn(),
  fetchScrapedItemSmartCounts: vi.fn()
}))

vi.mock("@/services/watchlists", () => watchlistsMock)

// The page's data-loading callbacks list `t` as a useCallback dependency, so
// the mock must hand back a STABLE t reference - a fresh closure per render
// retriggers the load effect forever.
const stableT = vi.hoisted(
  () =>
    (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return fallbackOrOptions
      }
      if (
        fallbackOrOptions &&
        typeof fallbackOrOptions === "object" &&
        typeof fallbackOrOptions.defaultValue === "string"
      ) {
        return fallbackOrOptions.defaultValue
      }
      return (maybeOptions?.defaultValue as string) || key
    }
)

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: stableT })
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  const Select = ({
    options = [],
    value,
    onChange,
    placeholder,
    loading
  }: {
    options?: Array<{ value: number; label: string }>
    value?: number | null
    onChange?: (value: number) => void
    placeholder?: string
    loading?: boolean
  }) => (
    <select
      aria-label="Select User"
      disabled={loading}
      value={value ?? ""}
      onChange={(event) => onChange?.(Number(event.currentTarget.value))}
    >
      <option value="">{placeholder ?? "Select"}</option>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  )
  return { ...actual, Select }
})

import WatchlistsOversightPage from "../WatchlistsOversightPage"

describe("WatchlistsOversightPage (#2922)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.listAdminUsers.mockResolvedValue({
      users: [
        { id: 1, username: "audit-admin", email: "admin@example.local" },
        { id: 2, username: "alice", email: "alice@example.local" }
      ]
    })
    watchlistsMock.fetchWatchlistSources.mockResolvedValue({
      items: [
        {
          id: 11,
          name: "Battery Tech News",
          url: "https://example.com/feed.xml",
          source_type: "rss",
          active: true,
          tags: [],
          created_at: "2026-09-06T00:00:00Z",
          last_scraped_at: null
        }
      ],
      total: 1
    })
    watchlistsMock.fetchScrapedItems.mockResolvedValue({
      items: [
        {
          id: 101,
          run_id: 1,
          job_id: 1,
          source_id: 11,
          title: "Recycling supply chain shifts",
          published_at: "2026-09-05T12:00:00Z",
          tags: [],
          status: "ingested",
          reviewed: false,
          created_at: "2026-09-05T12:00:00Z"
        }
      ],
      total: 1
    })
    watchlistsMock.fetchWatchlistRuns.mockResolvedValue({
      items: [
        {
          id: 7,
          job_id: 1,
          status: "success",
          started_at: "2026-09-05T11:00:00Z",
          finished_at: "2026-09-05T11:01:00Z"
        }
      ],
      total: 1
    })
    watchlistsMock.fetchScrapedItemSmartCounts.mockResolvedValue({
      all: 1,
      today: 0,
      today_unread: 0,
      unread: 1,
      reviewed: 0,
      queued: 0
    })
  })

  const selectAlice = async () => {
    render(<WatchlistsOversightPage />)
    const select = await screen.findByLabelText("Select User")
    fireEvent.change(select, { target: { value: "2" } })
  }

  it("scopes every fleet read to the selected user via target_user_id", async () => {
    await selectAlice()

    await waitFor(() => {
      expect(watchlistsMock.fetchWatchlistSources).toHaveBeenCalledWith(
        expect.objectContaining({ target_user_id: 2 })
      )
    })
    expect(watchlistsMock.fetchScrapedItems).toHaveBeenCalledWith(
      expect.objectContaining({ target_user_id: 2 })
    )
    expect(watchlistsMock.fetchWatchlistRuns).toHaveBeenCalledWith(
      expect.objectContaining({ target_user_id: 2 })
    )
    expect(watchlistsMock.fetchScrapedItemSmartCounts).toHaveBeenCalledWith(
      expect.objectContaining({ target_user_id: 2 })
    )
  })

  it("renders the selected user's feeds, items, and runs", async () => {
    await selectAlice()

    expect(await screen.findByText("Battery Tech News")).toBeInTheDocument()
    expect(screen.getByText("Recycling supply chain shifts")).toBeInTheDocument()
    const summary = screen.getByTestId("oversight-summary")
    expect(within(summary).getByText("Unread")).toBeInTheDocument()
  })

  it("does not load any data before a user is selected", async () => {
    render(<WatchlistsOversightPage />)
    await screen.findByLabelText("Select User")

    expect(
      screen.getByText("Select a user above to inspect their watchlist activity.")
    ).toBeInTheDocument()
    expect(watchlistsMock.fetchScrapedItems).not.toHaveBeenCalled()
  })

  it("renders private-only sharing mode as a designed state, not an error", async () => {
    watchlistsMock.fetchWatchlistSources.mockRejectedValue(
      new Error("403 watchlists_private_only_mode")
    )
    watchlistsMock.fetchScrapedItems.mockRejectedValue(
      new Error("403 watchlists_private_only_mode")
    )
    watchlistsMock.fetchWatchlistRuns.mockRejectedValue(
      new Error("403 watchlists_private_only_mode")
    )
    watchlistsMock.fetchScrapedItemSmartCounts.mockRejectedValue(
      new Error("403 watchlists_private_only_mode")
    )

    await selectAlice()

    expect(
      await screen.findByText("Watchlist sharing is disabled on this server")
    ).toBeInTheDocument()
  })
})
