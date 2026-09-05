// @vitest-environment jsdom
import React from "react"
import { describe, expect, it } from "vitest"
import { render, screen, within } from "@testing-library/react"

import { AdminOperationsOverviewPage } from "../AdminOperationsOverviewPage"
import { ADMIN_MODULES } from "../admin-modules"

describe("AdminOperationsOverviewPage", () => {
  it("links every registered admin module (the overview is the complete map)", () => {
    render(<AdminOperationsOverviewPage />)

    expect(
      screen.getByRole("heading", { name: "Admin Operations" })
    ).toBeInTheDocument()

    const modules = screen.getByTestId("admin-operations-modules")
    for (const module of ADMIN_MODULES) {
      const link = within(modules).getByRole("link", { name: module.label })
      expect(link).toHaveAttribute("href", module.route)
    }
    // Guard against regressing to a partial list (2026-09 audit finding S1).
    expect(ADMIN_MODULES.length).toBeGreaterThanOrEqual(17)
  })

  it("speaks operator language, not implementation status", () => {
    render(<AdminOperationsOverviewPage />)

    expect(screen.queryByText("Route ready")).not.toBeInTheDocument()
    expect(screen.queryByText("Diagnostics")).not.toBeInTheDocument()
    expect(
      screen.queryByText("frontend_state", { exact: false })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByText("Needs module configuration")
    ).not.toBeInTheDocument()
  })
})
