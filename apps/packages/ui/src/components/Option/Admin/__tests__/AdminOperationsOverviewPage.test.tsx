// @vitest-environment jsdom
import React from "react"
import { describe, expect, it } from "vitest"
import { render, screen, within } from "@testing-library/react"

import { AdminOperationsOverviewPage } from "../AdminOperationsOverviewPage"

describe("AdminOperationsOverviewPage", () => {
  it("renders admin operations as an overview with module drill-down routes", () => {
    render(<AdminOperationsOverviewPage />)

    expect(
      screen.getByRole("heading", { name: "Admin Operations" })
    ).toBeInTheDocument()
    expect(screen.queryByTestId("route-redirect-panel")).not.toBeInTheDocument()

    const modules = screen.getByTestId("admin-operations-modules")

    for (const [label, href] of [
      ["Server Admin", "/admin/server"],
      ["Workspace Integrations", "/admin/integrations"],
      ["Admin Sources", "/admin/sources"],
      ["Monitoring", "/admin/monitoring"]
    ] as const) {
      const link = within(modules).getByRole("link", { name: label })
      expect(link).toHaveAttribute("href", href)
    }
  })

  it("keeps module status visible and diagnostics behind disclosure", () => {
    render(<AdminOperationsOverviewPage />)

    const serverCard = screen.getByTestId("admin-module-/admin/server")

    expect(within(serverCard).getByText("Route ready")).toBeInTheDocument()
    expect(
      within(serverCard).getByText(
        "Open the module to load live server health and user data."
      )
    ).toBeInTheDocument()

    expect(screen.getAllByText("Diagnostics")).toHaveLength(4)
    expect(
      screen.queryByText("implementationOwner", { exact: false })
    ).not.toBeInTheDocument()
  })
})
