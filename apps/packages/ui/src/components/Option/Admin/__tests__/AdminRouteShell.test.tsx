// @vitest-environment jsdom
import React from "react"
import { describe, expect, it } from "vitest"
import { render, screen, within } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

import { AdminRouteShell } from "../AdminRouteShell"
import { ADMIN_MODULES, adminModuleForRoute, isAdminRoute } from "../admin-modules"

describe("AdminRouteShell", () => {
  it("renders a nav landmark linking every admin module", () => {
    render(
      <MemoryRouter>
        <AdminRouteShell path="/admin/server">
          <div>content</div>
        </AdminRouteShell>
      </MemoryRouter>
    )

    const nav = screen.getByRole("navigation", { name: "Admin modules" })
    for (const module of ADMIN_MODULES) {
      // Coming-soon modules carry a "Soon" badge inside the link (#2897),
      // so match on the label prefix rather than the exact accessible name.
      const link = within(nav).getByRole("link", {
        name: new RegExp(
          `^${module.label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`
        )
      })
      expect(link).toHaveAttribute("href", module.route)
    }
  })

  it("wraps the module nav instead of clipping modules past the viewport (#2888)", () => {
    render(
      <MemoryRouter>
        <AdminRouteShell path="/admin/server">
          <div>content</div>
        </AdminRouteShell>
      </MemoryRouter>
    )

    const nav = screen.getByRole("navigation", { name: "Admin modules" })
    const row = nav.firstElementChild as HTMLElement
    expect(row.className).toContain("flex-wrap")
    expect(row.className).not.toContain("whitespace-nowrap")
    expect(row.className).not.toContain("overflow-x-auto")
  })

  it("marks the current module and titles the document after it", () => {
    render(
      <MemoryRouter>
        <AdminRouteShell path="/admin/server">
          <div>content</div>
        </AdminRouteShell>
      </MemoryRouter>
    )

    const nav = screen.getByRole("navigation", { name: "Admin modules" })
    expect(
      within(nav).getByRole("link", { name: "Server Admin" })
    ).toHaveAttribute("aria-current", "page")
    expect(document.title).toBe("Server Admin · Admin · tldw")
  })

  it("provides a skip link to the admin content region", () => {
    render(
      <MemoryRouter>
        <AdminRouteShell path="/admin/monitoring">
          <div>content</div>
        </AdminRouteShell>
      </MemoryRouter>
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
