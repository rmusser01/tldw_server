// @vitest-environment jsdom

import React from "react"
import { existsSync, readFileSync } from "node:fs"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

vi.mock("@/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/components/Option/CompanionHome", () => ({
  CompanionHomeShell: ({ surface }: { surface: "options" | "sidepanel" }) => (
    <div data-testid="companion-home-shell">{surface}</div>
  )
}))

import OptionCompanion from "../option-companion"
import { getRouteMetadata } from "../route-metadata"

const routeRegistryPathCandidates = [
  "src/routes/route-registry.tsx",
  "../packages/ui/src/routes/route-registry.tsx",
  "apps/packages/ui/src/routes/route-registry.tsx"
]

const routeRegistryPath = routeRegistryPathCandidates.find((candidate) =>
  existsSync(candidate)
)

if (!routeRegistryPath) {
  throw new Error("Unable to locate route-registry.tsx for companion route test")
}

const routeRegistrySource = readFileSync(routeRegistryPath, "utf8")

const renderRoute = () =>
  render(
    <MemoryRouter>
      <OptionCompanion />
    </MemoryRouter>
  )

describe("option companion route", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders the shared Companion Home shell inside the option layout", async () => {
    renderRoute()

    expect(screen.getByTestId("option-layout")).toBeInTheDocument()
    expect(await screen.findByTestId("companion-home-shell")).toBeInTheDocument()
    expect(screen.getByText("options")).toBeInTheDocument()
  })

  it("registers the companion workspace route in the route registry", () => {
    expect(routeRegistrySource).toMatch(/path:\s*"\/companion"/)
    const metadata = getRouteMetadata("/companion")
    expect(metadata?.label).toBe("Companion")
    expect(metadata?.group).toBe("chat")
    expect(metadata?.availability).toContain("web")
  })
})
