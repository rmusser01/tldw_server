// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import UsageAnalyticsPage from "../UsageAnalyticsPage"

const apiMock = vi.hoisted(() => ({
  getDailyUsage: vi.fn(),
  getTopUsage: vi.fn(),
  getLlmUsage: vi.fn(),
  getLlmUsageSummary: vi.fn(),
  getLlmTopSpenders: vi.fn(),
  getRouterAnalyticsProviders: vi.fn(),
  exportDailyUsageCsv: vi.fn(),
  exportTopUsageCsv: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
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
      return maybeOptions?.defaultValue || key
    }
  })
}))

const DAY_MS = 24 * 60 * 60 * 1000

describe("UsageAnalyticsPage date-range pass-through", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.getDailyUsage.mockResolvedValue([])
    apiMock.getTopUsage.mockResolvedValue([])
    apiMock.getLlmUsage.mockResolvedValue([])
    apiMock.getLlmUsageSummary.mockResolvedValue({})
    apiMock.getLlmTopSpenders.mockResolvedValue([])
    apiMock.getRouterAnalyticsProviders.mockResolvedValue([])
    apiMock.exportDailyUsageCsv.mockResolvedValue("")
    apiMock.exportTopUsageCsv.mockResolvedValue("")
  })

  it("loads the default 7-day window once on mount without a duplicate reload", async () => {
    render(<UsageAnalyticsPage />)

    await waitFor(() => {
      expect(apiMock.getDailyUsage).toHaveBeenCalledTimes(1)
      expect(apiMock.getTopUsage).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlmUsage).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlmUsageSummary).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlmTopSpenders).toHaveBeenCalledTimes(1)
      expect(apiMock.getRouterAnalyticsProviders).toHaveBeenCalledTimes(1)
    })

    // The initial load already carries the selected range; switching ranges
    // (covered below) must not stack a second mount-style fan-out.
    expect(apiMock.getDailyUsage).toHaveBeenCalledWith(
      expect.objectContaining({
        start_date: expect.any(String),
        end_date: expect.any(String)
      })
    )
  })

  it("test_all_datasets_receive_range", async () => {
    render(<UsageAnalyticsPage />)

    // Let the initial 7d load settle, then isolate the range-switch calls.
    await waitFor(() => {
      expect(apiMock.getDailyUsage).toHaveBeenCalledTimes(1)
    })
    apiMock.getDailyUsage.mockClear()
    apiMock.getTopUsage.mockClear()
    apiMock.getLlmUsage.mockClear()
    apiMock.getLlmUsageSummary.mockClear()
    apiMock.getLlmTopSpenders.mockClear()

    // antd v6 keeps the mousedown handler on the `.ant-select` root.
    const rangeSelect = screen
      .getByRole("combobox")
      .closest(".ant-select") as HTMLElement
    fireEvent.mouseDown(rangeSelect)
    fireEvent.click(await screen.findByText("Last 30 days"))

    // Selecting a range reloads every dataset exactly once.
    await waitFor(() => {
      expect(apiMock.getDailyUsage).toHaveBeenCalledTimes(1)
      expect(apiMock.getTopUsage).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlmUsage).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlmUsageSummary).toHaveBeenCalledTimes(1)
      expect(apiMock.getLlmTopSpenders).toHaveBeenCalledTimes(1)
    })

    const [topUsageArgs] = apiMock.getTopUsage.mock.calls[0]
    // /usage/top documents start/end as YYYY-MM-DD inclusive.
    expect(topUsageArgs.start).toMatch(/^\d{4}-\d{2}-\d{2}$/)
    expect(topUsageArgs.end).toMatch(/^\d{4}-\d{2}-\d{2}$/)
    expect(
      Date.parse(topUsageArgs.end) - Date.parse(topUsageArgs.start)
    ).toBeGreaterThanOrEqual(28 * DAY_MS)

    // The llm-usage* endpoints document start/end as ISO timestamps.
    for (const llmMock of [
      apiMock.getLlmUsage,
      apiMock.getLlmUsageSummary,
      apiMock.getLlmTopSpenders
    ]) {
      const [args] = llmMock.mock.calls[0]
      expect(typeof args.start).toBe("string")
      expect(args.start).toMatch(/T/)
      expect(typeof args.end).toBe("string")
      expect(args.end).toMatch(/T/)
      expect(Number.isNaN(Date.parse(args.start))).toBe(false)
      expect(Date.parse(args.end) - Date.parse(args.start)).toBeGreaterThanOrEqual(
        28 * DAY_MS
      )
    }
  })
})
