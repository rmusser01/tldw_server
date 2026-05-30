// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import RbacEditorPage from "../RbacEditorPage"

const apiMock = vi.hoisted(() => ({
  getRolePermissionMatrix: vi.fn(),
  listPermissionCategories: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

describe("RbacEditorPage design-system states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    apiMock.getRolePermissionMatrix.mockResolvedValue({
      roles: [],
      permissions: [],
      grid: {}
    })
    apiMock.listPermissionCategories.mockResolvedValue([])
  })

  it("renders admin guard feedback through the design-system Alert primitive", async () => {
    apiMock.getRolePermissionMatrix.mockRejectedValueOnce({ status: 403 })

    render(<RbacEditorPage />)

    const title = await screen.findByText("Access Restricted")
    const alert = title.closest('[data-ds-component="Alert"]')

    expect(alert).not.toBeNull()
    const alertEl = alert as HTMLElement
    expect(alertEl).toHaveAttribute("role", "alert")
    expect(alertEl).toHaveTextContent("Access Restricted")
    expect(alertEl).toHaveTextContent("forbidden")
  })
})
