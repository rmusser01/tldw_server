import { existsSync, readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testFileDirectory = dirname(fileURLToPath(import.meta.url))
const webRouteRegistryRelativePath = "apps/packages/ui/src/routes/route-registry.tsx"
const extensionRouteRegistryRelativePath =
  "apps/tldw-frontend/extension/routes/route-registry.tsx"
const webRouteRelativePath = "apps/packages/ui/src/routes/option-scheduled-tasks.tsx"
const extensionRouteRelativePath =
  "apps/tldw-frontend/extension/routes/option-scheduled-tasks.tsx"
const hostedResultsAliasRelativePath = "apps/tldw-frontend/pages/scheduled-tasks/results.tsx"

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
      throw new Error("Unable to locate workspace root for scheduled tasks route test")
    }
    currentDirectory = parentDirectory
  }
}

const workspaceRoot = resolveWorkspaceRoot(testFileDirectory)
const webRouteSource = readFileSync(resolve(workspaceRoot, webRouteRelativePath), "utf8")
const webRouteRegistrySource = readFileSync(
  resolve(workspaceRoot, webRouteRegistryRelativePath),
  "utf8"
)
const extensionRouteRegistrySource = readFileSync(
  resolve(workspaceRoot, extensionRouteRegistryRelativePath),
  "utf8"
)
const extensionRoutePath = resolve(workspaceRoot, extensionRouteRelativePath)
const hostedResultsAliasPath = resolve(workspaceRoot, hostedResultsAliasRelativePath)

describe("scheduled tasks route wiring", () => {
  it("registers the scheduled tasks route shell", () => {
    expect(webRouteSource).toContain("ScheduledTasksPage")
    expect(webRouteSource).toContain("RouteErrorBoundary")
    expect(webRouteSource).toContain("OptionLayout")
  })

  it("registers the scheduled tasks page in both web and extension route registries", () => {
    expect(webRouteRegistrySource).toContain('path: "/scheduled-tasks"')
    expect(extensionRouteRegistrySource).toContain('path: "/scheduled-tasks"')
  })

  it("registers scheduled tasks Results aliases across route surfaces", () => {
    expect(webRouteRegistrySource).toContain('path: "/scheduled-tasks/results"')
    expect(extensionRouteRegistrySource).toContain('path: "/scheduled-tasks/results"')
    expect(existsSync(hostedResultsAliasPath)).toBe(true)
    const hostedResultsAliasSource = readFileSync(hostedResultsAliasPath, "utf8")
    expect(hostedResultsAliasSource).toContain("option-scheduled-tasks")
  })

  it("uses a dedicated extension scheduled tasks route shell", () => {
    expect(existsSync(extensionRoutePath)).toBe(true)

    const extensionRouteSource = readFileSync(extensionRoutePath, "utf8")

    expect(extensionRouteSource).toContain("ScheduledTasksPage")
    expect(extensionRouteSource).toContain("RouteErrorBoundary")
  })
})
