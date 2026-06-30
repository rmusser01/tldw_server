import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MemoryRouter, Route, Routes } from "react-router-dom"

import { WORKSPACES_PATH } from "../route-paths"
import { getRouteMetadata, isAuditedRootRoute } from "../route-metadata"
import OptionWorkspaces from "../option-workspaces"

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Option/Workspaces/WorkspacesManagerPage", () => ({
  WorkspacesManagerPage: () => (
    <div data-testid="workspaces-manager-page">Workspaces Manager</div>
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

describe("option workspaces route", () => {
  it("owns the /workspaces route constant and metadata without aliases", () => {
    expect(WORKSPACES_PATH).toBe("/workspaces")
    expect(isAuditedRootRoute(WORKSPACES_PATH)).toBe(true)
    expect(getRouteMetadata(WORKSPACES_PATH)).toMatchObject({
      path: WORKSPACES_PATH,
      label: "Workspaces",
      group: "workspace",
      nav: "secondary",
      requiresBackend: true
    })
    expect(getRouteMetadata(WORKSPACES_PATH)?.aliases).toBeUndefined()
    expect(getRouteMetadata(WORKSPACES_PATH)?.redirectsTo).toBeUndefined()
  })

  it("uses the standard option layout with a flex shell", () => {
    render(<OptionWorkspaces />)

    const shell = screen.getByTestId("workspaces-route-shell")
    expect(shell.className).toContain("flex")
    expect(shell.className).toContain("flex-1")
    expect(shell.className).toContain("min-h-0")
    expect(shell.className).toContain("overflow-hidden")
    expect(screen.getByTestId("option-layout")).toBeInTheDocument()
    expect(screen.getByTestId("workspaces-manager-page")).toBeVisible()
  })

  it("registers the canonical manager route in the main registry", async () => {
    const route = ROUTE_DEFINITIONS.find(
      (candidate) => candidate.path === WORKSPACES_PATH
    )

    expect(route).toBeDefined()

    render(
      <MemoryRouter initialEntries={[WORKSPACES_PATH]}>
        <Routes>
          <Route path="*" element={route!.element} />
        </Routes>
      </MemoryRouter>
    )

    expect(await screen.findByTestId("workspaces-manager-page")).toBeVisible()
  })
})
