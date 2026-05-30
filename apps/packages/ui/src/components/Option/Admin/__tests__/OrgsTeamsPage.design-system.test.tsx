// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import OrgsTeamsPage from "../OrgsTeamsPage"

const apiMock = vi.hoisted(() => ({
  listOrgs: vi.fn()
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

describe("OrgsTeamsPage design-system states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.listOrgs.mockResolvedValue([])
  })

  it("renders forbidden guard feedback through the design-system Alert primitive", async () => {
    apiMock.listOrgs.mockRejectedValueOnce({ status: 403 })

    render(<OrgsTeamsPage />)

    const alert = await expectDesignSystemAlertForTitle("Access Denied")
    expect(alert).toHaveTextContent(
      "You don't have permission to manage organizations."
    )
  })

  it("renders missing-endpoint guard feedback through the design-system Alert primitive", async () => {
    apiMock.listOrgs.mockRejectedValueOnce({ status: 404 })

    render(<OrgsTeamsPage />)

    const alert = await expectDesignSystemAlertForTitle("Not Available")
    expect(alert).toHaveTextContent(
      "Organization management is not available on this server."
    )
  })
})
