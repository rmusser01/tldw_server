// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import WatchlistsPage from "../WatchlistsPage"

const apiMock = vi.hoisted(() => ({
  listWatchlists: vi.fn(),
  listMonitoringAlerts: vi.fn(),
  createWatchlist: vi.fn(),
  deleteWatchlist: vi.fn(),
  acknowledgeAlert: vi.fn(),
  dismissAlert: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

describe("WatchlistsPage admin guard alerts", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.listWatchlists.mockResolvedValue([])
    apiMock.listMonitoringAlerts.mockResolvedValue({ items: [] })
  })

  it("renders forbidden access guard with the design-system Alert primitive", async () => {
    apiMock.listWatchlists.mockRejectedValueOnce({ status: 403 })

    render(<WatchlistsPage />)

    const title = await screen.findByText("Access Denied")

    expect(title.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
    expect(
      screen.getByText(
        "You don't have permission to access watchlists administration."
      )
    ).toBeInTheDocument()
  })

  it("renders unavailable guard with the design-system Alert primitive", async () => {
    apiMock.listWatchlists.mockRejectedValueOnce({ status: 404 })

    render(<WatchlistsPage />)

    const title = await screen.findByText("Not Available")

    expect(title.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
    expect(
      screen.getByText("Watchlists administration is not available on this server.")
    ).toBeInTheDocument()
  })
})
