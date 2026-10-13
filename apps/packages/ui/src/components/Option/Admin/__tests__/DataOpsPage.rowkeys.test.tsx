// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import DataOpsPage from "../DataOpsPage"
import { formatAdminDateTime } from "../admin-format"

const apiMock = vi.hoisted(() => ({
  listBackups: vi.fn(),
  listBackupSchedules: vi.fn(),
  listBundles: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

const tableRowKeys = () =>
  Array.from(document.querySelectorAll("tr.ant-table-row")).map((row) =>
    row.getAttribute("data-row-key")
  )

const waitForRows = async (count: number) =>
  waitFor(() => {
    const keys = tableRowKeys()
    expect(keys).toHaveLength(count)
    return keys
  })

describe("DataOpsPage stable row keys (C-S5)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.listBackups.mockResolvedValue([])
    apiMock.listBackupSchedules.mockResolvedValue([])
    apiMock.listBundles.mockResolvedValue([])
  })

  it("keys backups by the deterministic composite instead of JSON.stringify", async () => {
    apiMock.listBackups.mockResolvedValue([
      { dataset: "media", created_at: "2026-01-01T00:00:00Z", filename: "media-1.db", status: "completed", size: 10 },
      { dataset: "media", created_at: "2026-01-01T00:00:00Z", filename: "media-2.db", status: "completed", size: 20 }
    ])

    const view = render(<DataOpsPage />)

    // Two backup rows (schedules are mocked empty in this test).
    let keys = await waitForRows(2)
    expect(keys).toEqual([
      "media|2026-01-01T00:00:00Z|media-1.db",
      "media|2026-01-01T00:00:00Z|media-2.db"
    ])

    // Same data re-rendered: identical key set, no JSON serialization.
    view.rerender(<DataOpsPage />)
    expect(tableRowKeys()).toEqual(keys)
    for (const key of keys) {
      expect(key).not.toContain("{")
      expect(key).not.toContain('"')
    }

    // The created cell renders through the shared admin formatter.
    expect(
      screen.getAllByText(formatAdminDateTime("2026-01-01T00:00:00Z")).length
    ).toBeGreaterThan(0)
  })

  it("suffixes duplicate schedule composites at fetch time instead of colliding", async () => {
    // Schedules have no id/schedule_id here and no created_at, so both rows
    // collapse onto the same composite; the loader must disambiguate.
    apiMock.listBackupSchedules.mockResolvedValue([
      { dataset: "media", frequency: "daily", time_of_day: "02:00", retention_count: 14 },
      { dataset: "media", frequency: "weekly", time_of_day: "03:00", retention_count: 8 }
    ])

    render(<DataOpsPage />)

    const keys = await waitForRows(2)
    expect(new Set(keys).size).toBe(keys.length)
    expect(keys).toEqual(["media||", "media||#2"])
  })

  it("keys bundles by the composite with a fetch-time suffix for duplicates", async () => {
    apiMock.listBundles.mockResolvedValue([
      { datasets: ["media"], created_at: "2026-03-01T00:00:00Z" },
      { datasets: ["media", "users"], created_at: "2026-03-01T00:00:00Z" }
    ])

    render(<DataOpsPage />)

    const user = userEvent.setup()
    await user.click(screen.getByRole("tab", { name: "Bundles" }))

    // Two bundle rows each carry a "media" dataset tag.
    await waitFor(() => {
      expect(screen.getAllByText("media", { exact: true })).toHaveLength(2)
    })
    expect(tableRowKeys()).toEqual([
      "ds|2026-03-01T00:00:00Z|",
      "ds|2026-03-01T00:00:00Z|#2"
    ])
  })
})
