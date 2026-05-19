// @vitest-environment jsdom

import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import OptionWatchlists from "../option-watchlists"

const routeStateMocks = vi.hoisted(() => ({
  setActiveTab: vi.fn(),
  setItemsSelectedSourceId: vi.fn(),
  setItemsSmartFilter: vi.fn(),
  setItemsStatusFilter: vi.fn(),
  setItemsSearchQuery: vi.fn(),
  setRunsJobFilter: vi.fn(),
  setRunsStatusFilter: vi.fn(),
  openRunDetail: vi.fn(),
  setOutputsJobFilter: vi.fn(),
  setOutputsRunFilter: vi.fn(),
  openOutputPreview: vi.fn(),
  stateRef: { current: {} as Record<string, unknown> }
}))

vi.mock("@/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({
    children,
    routeId,
    routeLabel
  }: {
    children: React.ReactNode
    routeId: string
    routeLabel: string
  }) => (
    <div data-route-boundary={routeId} data-route-label={routeLabel}>
      {children}
    </div>
  )
}))

vi.mock("@/components/Option/Watchlists/WatchlistsPlaygroundPage", () => ({
  WatchlistsPlaygroundPage: () => <div data-testid="watchlists-playground" />
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector(routeStateMocks.stateRef.current)
}))

const createStoreState = (): Record<string, unknown> => ({
  setActiveTab: routeStateMocks.setActiveTab,
  setItemsSelectedSourceId: routeStateMocks.setItemsSelectedSourceId,
  setItemsSmartFilter: routeStateMocks.setItemsSmartFilter,
  setItemsStatusFilter: routeStateMocks.setItemsStatusFilter,
  setItemsSearchQuery: routeStateMocks.setItemsSearchQuery,
  setRunsJobFilter: routeStateMocks.setRunsJobFilter,
  setRunsStatusFilter: routeStateMocks.setRunsStatusFilter,
  openRunDetail: routeStateMocks.openRunDetail,
  setOutputsJobFilter: routeStateMocks.setOutputsJobFilter,
  setOutputsRunFilter: routeStateMocks.setOutputsRunFilter,
  openOutputPreview: routeStateMocks.openOutputPreview
})

const renderAt = (path: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/watchlists" element={<OptionWatchlists />} />
      </Routes>
    </MemoryRouter>
  )

describe("option-watchlists route query handoff", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    routeStateMocks.stateRef.current = createStoreState()
  })

  it("applies items deep-link filters from query params", async () => {
    renderAt("/watchlists?tab=items&source_id=12&item_smart=todayUnread&item_status=filtered&item_q=chip")

    await waitFor(() => {
      expect(routeStateMocks.setActiveTab).toHaveBeenCalledWith("items")
    })
    expect(routeStateMocks.setItemsSelectedSourceId).toHaveBeenCalledWith(12)
    expect(routeStateMocks.setItemsSmartFilter).toHaveBeenCalledWith("todayUnread")
    expect(routeStateMocks.setItemsStatusFilter).toHaveBeenCalledWith("filtered")
    expect(routeStateMocks.setItemsSearchQuery).toHaveBeenCalledWith("chip")
  })

  it("applies run deep-link context and opens run detail when tab is runs", async () => {
    renderAt("/watchlists?tab=runs&job_id=44&run_id=55&run_status=failed")

    await waitFor(() => {
      expect(routeStateMocks.setActiveTab).toHaveBeenCalledWith("runs")
    })
    expect(routeStateMocks.setRunsJobFilter).toHaveBeenCalledWith(44)
    expect(routeStateMocks.setOutputsJobFilter).toHaveBeenCalledWith(44)
    expect(routeStateMocks.setOutputsRunFilter).toHaveBeenCalledWith(55)
    expect(routeStateMocks.setRunsStatusFilter).toHaveBeenCalledWith("failed")
    expect(routeStateMocks.openRunDetail).toHaveBeenCalledWith(55)
  })

  it("applies output deep-link context without forcing run drawer", async () => {
    renderAt("/watchlists?tab=outputs&job_id=8&run_id=13&output_id=377")

    await waitFor(() => {
      expect(routeStateMocks.setActiveTab).toHaveBeenCalledWith("outputs")
    })
    expect(routeStateMocks.setOutputsJobFilter).toHaveBeenCalledWith(8)
    expect(routeStateMocks.setOutputsRunFilter).toHaveBeenCalledWith(13)
    expect(routeStateMocks.openOutputPreview).toHaveBeenCalledWith(377)
    expect(routeStateMocks.openRunDetail).not.toHaveBeenCalled()
  })

  it("wraps the watchlists route in the shared route error boundary", () => {
    const { container } = renderAt("/watchlists")

    expect(container.querySelector('[data-route-boundary="watchlists"]')).toHaveAttribute(
      "data-route-label",
      "Watchlists"
    )
    expect(screen.getByTestId("option-layout")).toContainElement(
      screen.getByTestId("watchlists-playground")
    )
  })

  it("accepts user-facing tab aliases and camelCase deep-link params", async () => {
    renderAt(
      "/watchlists?tab=activity&sourceId=12&itemSmart=todayUnread&itemStatus=ingested&itemQuery=chips&jobId=44&runId=55&open_run=1"
    )

    await waitFor(() => {
      expect(routeStateMocks.setActiveTab).toHaveBeenCalledWith("runs")
    })
    expect(routeStateMocks.setItemsSelectedSourceId).toHaveBeenCalledWith(12)
    expect(routeStateMocks.setItemsSmartFilter).toHaveBeenCalledWith("todayUnread")
    expect(routeStateMocks.setItemsStatusFilter).toHaveBeenCalledWith("ingested")
    expect(routeStateMocks.setItemsSearchQuery).toHaveBeenCalledWith("chips")
    expect(routeStateMocks.setRunsJobFilter).toHaveBeenCalledWith(44)
    expect(routeStateMocks.setOutputsJobFilter).toHaveBeenCalledWith(44)
    expect(routeStateMocks.setOutputsRunFilter).toHaveBeenCalledWith(55)
    expect(routeStateMocks.openRunDetail).toHaveBeenCalledWith(55)
  })
})
