// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { WatchlistSetupWizard } from "../WatchlistSetupWizard"

const services = vi.hoisted(() => ({
  triggerRun: vi.fn(),
  createOutput: vi.fn(),
  testSource: vi.fn()
}))

vi.mock("@/services/watchlists", () => ({
  triggerWatchlistRun: (...args: unknown[]) => services.triggerRun(...args),
  createWatchlistOutput: (...args: unknown[]) => services.createOutput(...args),
  testWatchlistSourceDraft: (...args: unknown[]) => services.testSource(...args)
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: Record<string, unknown>) =>
      (fallback || _key).replace(/\{\{(\w+)\}\}/g, (_match, token) => String(options?.[token] ?? ""))
  })
}))

const watchlist = {
  id: 501,
  name: "Lakers weekly",
  description: null,
  objective: null,
  domain: "general" as const,
  status: "active" as const,
  priority: "medium" as const,
  tags: [],
  created_at: "2026-07-10T00:00:00Z",
  updated_at: "2026-07-10T00:00:00Z"
}

const inactiveJob = {
  id: 77,
  name: "Lakers weekly monitor",
  watchlist_id: 501,
  scope: { sources: [301] },
  active: false,
  created_at: "2026-07-10T00:00:00Z"
}

const renderWizard = (overrides: Record<string, unknown> = {}) => {
  const props = {
    onCancel: vi.fn(),
    onCreateWatchlist: vi.fn().mockResolvedValue(watchlist),
    onCreateSources: vi.fn().mockResolvedValue([301]),
    onCreateJob: vi.fn().mockResolvedValue(inactiveJob),
    onUpdateJob: vi.fn().mockResolvedValue({ ...inactiveJob, active: true }),
    onComplete: vi.fn(),
    ...overrides
  }
  render(<WatchlistSetupWizard open {...props} />)
  return props
}

beforeEach(() => {
  vi.clearAllMocks()
  services.triggerRun.mockResolvedValue({ id: 88 })
  services.createOutput.mockResolvedValue({ id: 99 })
  services.testSource.mockResolvedValue({ total: 1, ingestable: 1, filtered: 0, items: [] })
})

describe("WatchlistSetupWizard", () => {
  it("creates the container, then continues in the canonical wizard at Sources", async () => {
    const props = renderWizard()

    fireEvent.change(screen.getByLabelText("Watchlist name"), {
      target: { value: "Lakers weekly" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Continue to Sources" }))

    await waitFor(() => expect(props.onCreateWatchlist).toHaveBeenCalledTimes(1))
    const pipeline = within(await screen.findByRole("dialog", { name: "Set up briefing" }))
    expect(pipeline.getByLabelText("Add a new source")).toBeChecked()
    expect(pipeline.getAllByRole("listitem").map((item) => item.textContent)).toEqual([
      expect.stringContaining("Sources"),
      expect.stringContaining("Cadence"),
      expect.stringContaining("Briefing"),
      expect.stringContaining("Delivery"),
      expect.stringContaining("Test")
    ])
  })

  it("keeps container input after creation failure", async () => {
    renderWizard({ onCreateWatchlist: vi.fn().mockRejectedValue(new Error("Server unavailable")) })

    fireEvent.change(screen.getByLabelText("Watchlist name"), {
      target: { value: "Research watchlist" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Continue to Sources" }))

    expect(await screen.findByText("Server unavailable")).toBeInTheDocument()
    expect(screen.getByLabelText("Watchlist name")).toHaveValue("Research watchlist")
  })

  it("tests an inactive canonical monitor and activates the same id", async () => {
    const props = renderWizard()
    fireEvent.change(screen.getByLabelText("Watchlist name"), { target: { value: "Lakers weekly" } })
    fireEvent.click(screen.getByRole("button", { name: "Continue to Sources" }))
    const pipeline = within(await screen.findByRole("dialog", { name: "Set up briefing" }))

    fireEvent.change(pipeline.getByLabelText("Source name"), { target: { value: "Lakers feed" } })
    fireEvent.change(pipeline.getByLabelText("Source URL"), { target: { value: "https://example.com/lakers.xml" } })
    fireEvent.click(pipeline.getByRole("button", { name: "Next: Cadence" }))
    fireEvent.change(pipeline.getByLabelText("Monitor name"), { target: { value: "Lakers weekly monitor" } })
    fireEvent.click(pipeline.getByRole("button", { name: "Next: Briefing" }))
    fireEvent.click(pipeline.getByRole("button", { name: "Next: Delivery" }))
    fireEvent.click(pipeline.getByRole("button", { name: "Next: Test" }))
    fireEvent.click(pipeline.getByRole("button", { name: "Generate 60-second sample" }))

    await waitFor(() => {
      expect(props.onCreateJob).toHaveBeenCalledWith(
        501,
        expect.objectContaining({ active: false, watchlist_id: 501 })
      )
    })
    expect(services.triggerRun).toHaveBeenCalledWith(77)

    fireEvent.click(pipeline.getByRole("button", { name: "Activate schedule" }))
    await waitFor(() => expect(props.onUpdateJob).toHaveBeenCalledWith(77, { active: true }))
    expect(props.onCreateJob).toHaveBeenCalledTimes(1)
    expect(props.onComplete).toHaveBeenCalledWith(expect.objectContaining({
      watchlist,
      sourceIds: [301],
      job: expect.objectContaining({ id: 77, active: true })
    }))
  }, 20_000)
})
