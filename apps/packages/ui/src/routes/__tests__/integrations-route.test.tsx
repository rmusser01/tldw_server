import React from "react"
import { existsSync, readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  integrationPage: vi.fn(),
  adminIntegrationPage: vi.fn()
}))

vi.mock("@/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="route-error-boundary">{children}</div>
  )
}))

vi.mock("@/components/Option/Integrations/IntegrationManagementPage", () => ({
  IntegrationManagementPage: ({ scope }: { scope: string }) => {
    mocks.integrationPage(scope)
    return <div data-testid="integration-management-page" data-scope={scope} />
  }
}))

import OptionIntegrations from "../option-integrations"
import OptionAdminIntegrations from "../option-admin-integrations"

const testFileDirectory = dirname(fileURLToPath(import.meta.url))
const webRouteRegistryRelativePath = "apps/packages/ui/src/routes/route-registry.tsx"
const extensionRouteRegistryRelativePath =
  "apps/tldw-frontend/extension/routes/route-registry.tsx"
const extensionPersonalRouteRelativePath =
  "apps/tldw-frontend/extension/routes/option-integrations.tsx"
const extensionAdminRouteRelativePath =
  "apps/tldw-frontend/extension/routes/option-admin-integrations.tsx"

const resolveWorkspaceRoot = (startDirectory: string): string => {
  let currentDirectory = startDirectory
  while (true) {
    const webPath = resolve(currentDirectory, webRouteRegistryRelativePath)
    const extensionPath = resolve(currentDirectory, extensionRouteRegistryRelativePath)
    if (existsSync(webPath) && existsSync(extensionPath)) {
      return currentDirectory
    }

    const parentDirectory = dirname(currentDirectory)
    if (parentDirectory === currentDirectory) {
      throw new Error("Unable to locate workspace root for integrations route tests")
    }
    currentDirectory = parentDirectory
  }
}

const workspaceRoot = resolveWorkspaceRoot(testFileDirectory)
const webRouteRegistrySource = readFileSync(
  resolve(workspaceRoot, webRouteRegistryRelativePath),
  "utf8"
)
const extensionRouteRegistrySource = readFileSync(
  resolve(workspaceRoot, extensionRouteRegistryRelativePath),
  "utf8"
)
const extensionPersonalRoutePath = resolve(workspaceRoot, extensionPersonalRouteRelativePath)
const extensionAdminRoutePath = resolve(workspaceRoot, extensionAdminRouteRelativePath)

describe("integration routes", () => {
  beforeEach(() => {
    mocks.integrationPage.mockReset()
    mocks.adminIntegrationPage.mockReset()
  })

  it("renders the personal integrations page in the shared option layout", () => {
    render(
      <MemoryRouter>
        <OptionIntegrations />
      </MemoryRouter>
    )

    expect(screen.getByTestId("option-layout")).toBeInTheDocument()
    expect(screen.getByTestId("integration-management-page")).toHaveAttribute("data-scope", "personal")
  })

  it("renders the admin integrations page in the route error boundary", () => {
    render(
      <MemoryRouter>
        <OptionAdminIntegrations />
      </MemoryRouter>
    )

    expect(screen.getByTestId("route-error-boundary")).toBeInTheDocument()
    expect(screen.getByTestId("option-layout")).toBeInTheDocument()
    expect(screen.getByTestId("integration-management-page")).toHaveAttribute("data-scope", "workspace")
  })

  it("registers personal and admin integrations routes in both web and extension registries", () => {
    expect(webRouteRegistrySource).toContain('path: "/integrations"')
    expect(webRouteRegistrySource).toContain('path: "/admin/integrations"')
    expect(extensionRouteRegistrySource).toContain('path: "/integrations"')
    expect(extensionRouteRegistrySource).toContain('path: "/admin/integrations"')
  })

  it("uses dedicated extension route shells for both integrations surfaces", () => {
    expect(existsSync(extensionPersonalRoutePath)).toBe(true)
    expect(existsSync(extensionAdminRoutePath)).toBe(true)

    const extensionPersonalRouteSource = readFileSync(extensionPersonalRoutePath, "utf8")
    const extensionAdminRouteSource = readFileSync(extensionAdminRoutePath, "utf8")

    expect(extensionPersonalRouteSource).toContain("IntegrationManagementPage")
    expect(extensionPersonalRouteSource).toContain('scope="personal"')
    expect(extensionAdminRouteSource).toContain("RouteErrorBoundary")
    expect(extensionAdminRouteSource).toContain("IntegrationManagementPage")
    expect(extensionAdminRouteSource).toContain('scope="workspace"')
  })
})
