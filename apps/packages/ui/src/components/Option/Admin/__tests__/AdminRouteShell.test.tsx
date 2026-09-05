// @vitest-environment jsdom
import React from "react"
import { describe, expect, it } from "vitest"
import { render, screen, within } from "@testing-library/react"

import { AdminRouteShell } from "../AdminRouteShell"
import { ADMIN_MODULES, adminModuleForRoute, isAdminRoute } from "../admin-modules"

describe("AdminRouteShell", () => {
  it("renders a nav landmark linking every admin module", () => {
    render(
      <AdminRouteShell path="/admin/server">
        <div>content</div>
      </AdminRouteShell>
    )

    const nav = screen.getByRole("navigation", { name: "Admin modules" })
    for (const module of ADMIN_MODULES) {
      const link = within(nav).getByRole("link", { name: module.label })
      expect(link).toHaveAttribute("href", module.route)
    }
  })

  it("marks the current module and titles the document after it", () => {
    render(
      <AdminRouteShell path="/admin/server">
        <div>content</div>
      </AdminRouteShell>
    )

    const nav = screen.getByRole("navigation", { name: "Admin modules" })
    expect(
      within(nav).getByRole("link", { name: "Server Admin" })
    ).toHaveAttribute("aria-current", "page")
    expect(document.title).toBe("Server Admin · Admin · tldw")
  })

  it("provides a skip link to the admin content region", () => {
    render(
      <AdminRouteShell path="/admin/monitoring">
        <div>content</div>
      </AdminRouteShell>
    )

    const skip = screen.getByRole("link", { name: "Skip to admin content" })
    expect(skip).toHaveAttribute("href", "#admin-content")
    expect(document.getElementById("admin-content")).toBeTruthy()
  })

  it("resolves admin route helpers consistently", () => {
    expect(isAdminRoute("/admin")).toBe(true)
    expect(isAdminRoute("/admin/server/")).toBe(true)
    expect(isAdminRoute("/administration")).toBe(false)
    expect(isAdminRoute("/chat")).toBe(false)
    expect(adminModuleForRoute("/admin/server?tab=users")?.label).toBe(
      "Server Admin"
    )
    expect(adminModuleForRoute("/admin")).toBeUndefined()
  })
})
