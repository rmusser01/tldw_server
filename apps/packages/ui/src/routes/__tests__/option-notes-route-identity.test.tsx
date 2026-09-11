import { existsSync, readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import OptionNotes from "../option-notes"

const testDirectory = dirname(fileURLToPath(import.meta.url))
const sharedRegistryRelative = "apps/packages/ui/src/routes/route-registry.tsx"
const extensionRegistryRelative =
  "apps/tldw-frontend/extension/routes/route-registry.tsx"

const workspaceRoot = (() => {
  let current = testDirectory
  while (true) {
    if (
      existsSync(resolve(current, sharedRegistryRelative)) &&
      existsSync(resolve(current, extensionRegistryRelative))
    ) {
      return current
    }
    const parent = dirname(current)
    if (parent === current) {
      throw new Error("Unable to locate workspace root for Notes route identity")
    }
    current = parent
  }
})()

const source = (relativePath: string) =>
  readFileSync(resolve(workspaceRoot, relativePath), "utf8")

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({
    children,
    routeId,
    routeLabel
  }: {
    children: React.ReactNode
    routeId: string
    routeLabel: string
  }) => (
    <div data-testid="route-boundary" data-route-id={routeId} data-route-label={routeLabel}>
      {children}
    </div>
  )
}))

vi.mock("@/components/Notes/NotesManagerPage", () => ({
  __esModule: true,
  default: () => <div data-testid="notes-manager-page">Notes manager</div>
}))

describe("notes option route identity", () => {
  it("wraps /notes in the Notes route boundary and page", () => {
    render(<OptionNotes />)

    const boundary = screen.getByTestId("route-boundary")

    expect(screen.getByTestId("option-layout")).toBeVisible()
    expect(boundary).toHaveAttribute("data-route-id", "notes")
    expect(boundary).toHaveAttribute("data-route-label", "Notes")
    expect(screen.getByTestId("notes-manager-page")).toBeVisible()
  })

  it("keeps hosted and extension aliases on the shared Notes implementation", () => {
    const sharedRegistry = source(sharedRegistryRelative)
    const extensionRegistry = source(extensionRegistryRelative)
    const sharedOption = source("apps/packages/ui/src/routes/option-notes.tsx")
    const extensionOption = source(
      "apps/tldw-frontend/extension/routes/option-notes.tsx"
    )
    const hostedPage = source("apps/tldw-frontend/pages/notes.tsx")

    for (const registry of [sharedRegistry, extensionRegistry]) {
      expect(registry).toMatch(
        /const OptionNotes = lazy\(\(\) => import\("\.\/option-notes"\)\)/
      )
      expect(registry).toMatch(/path: "\/notes",\s*element: <OptionNotes \/>/)
    }
    expect(hostedPage).toContain('import("@/routes/option-notes")')
    for (const optionModule of [sharedOption, extensionOption]) {
      expect(optionModule).toContain(
        'import NotesManagerPage from "@/components/Notes/NotesManagerPage"'
      )
      expect(optionModule).toContain("<NotesManagerPage />")
      expect(optionModule).not.toMatch(/components\/Notes\/(?!NotesManagerPage)/)
    }
  })
})
