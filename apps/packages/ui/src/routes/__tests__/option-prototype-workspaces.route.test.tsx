import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MemoryRouter, Route, Routes } from "react-router-dom"

import OptionPrototypeWorkspaces from "../option-prototype-workspaces"

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Option/PrototypeWorkspace", () => ({
  PrototypeWorkspacePage: () => (
    <div data-testid="prototype-workspace-page">Prototype Workspace</div>
  )
}))

vi.mock("../option-index", () => ({
  __esModule: true,
  default: () => <div data-testid="option-index" />
}))

vi.mock("../settings-route", () => ({
  __esModule: true,
  createSettingsRoute: () => () => <div data-testid="settings-route-stub" />
}))

import { ROUTE_DEFINITIONS } from "../route-registry"

describe("option prototype workspaces route", () => {
  it("uses the standard option layout with a flex shell for the prototype workspace page", () => {
    render(<OptionPrototypeWorkspaces />)

    const shell = screen.getByTestId("prototype-workspaces-route-shell")
    expect(shell.className).toContain("flex")
    expect(shell.className).toContain("flex-1")
    expect(shell.className).toContain("min-h-0")
    expect(shell.className).toContain("overflow-hidden")
    expect(screen.getByTestId("option-layout")).toBeInTheDocument()
    expect(screen.getByTestId("prototype-workspace-page")).toBeVisible()
  })

  it("registers the prototype workspace route in the main route registry", async () => {
    const route = ROUTE_DEFINITIONS.find(
      (candidate) => candidate.path === "/prototype-workspaces"
    )

    expect(route).toBeDefined()

    render(
      <MemoryRouter initialEntries={["/prototype-workspaces"]}>
        <Routes>
          <Route path="*" element={route!.element} />
        </Routes>
      </MemoryRouter>
    )

    expect(await screen.findByTestId("prototype-workspace-page")).toBeVisible()
  })
})
