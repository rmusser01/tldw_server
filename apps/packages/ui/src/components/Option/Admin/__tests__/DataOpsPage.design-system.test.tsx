// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import DataOpsPage from "../DataOpsPage"

const apiMock = vi.hoisted(() => ({
  listBackups: vi.fn(),
  listBackupSchedules: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

const expectDesignSystemAlertForTitle = async (titleText: string) => {
  const title = await screen.findByText(titleText)
  const alert = title.closest('[data-ds-component="Alert"]')

  expect(alert).not.toBeNull()
  const alertEl = alert as HTMLElement
  expect(alertEl).toHaveAttribute("role", "alert")
  return alertEl
}

describe("DataOpsPage design-system states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.listBackups.mockResolvedValue([])
    apiMock.listBackupSchedules.mockResolvedValue([])
  })

  it("renders forbidden guard feedback through the design-system Alert primitive", async () => {
    apiMock.listBackups.mockRejectedValueOnce({ status: 403 })

    render(<DataOpsPage />)

    const alert = await expectDesignSystemAlertForTitle("Access Denied")
    expect(alert).toHaveTextContent(
      "You don't have permission to access data operations."
    )
  })

  it("renders missing-endpoint guard feedback through the design-system Alert primitive", async () => {
    apiMock.listBackups.mockRejectedValueOnce({ status: 404 })

    render(<DataOpsPage />)

    const alert = await expectDesignSystemAlertForTitle("Not Available")
    expect(alert).toHaveTextContent(
      "Data operations are not available on this server."
    )
  })
})
