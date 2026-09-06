import { existsSync, readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testFileDirectory = dirname(fileURLToPath(import.meta.url))
const webRouteRegistryRelativePath = "apps/packages/ui/src/routes/route-registry.tsx"
const extensionRouteRegistryRelativePath =
  "apps/tldw-frontend/extension/routes/route-registry.tsx"

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
      throw new Error("Unable to locate workspace root for family guardrails route parity test")
    }
    currentDirectory = parentDirectory
  }
}

const workspaceRoot = resolveWorkspaceRoot(testFileDirectory)
const webRouteRegistryPath = resolve(workspaceRoot, webRouteRegistryRelativePath)
const extensionRouteRegistryPath = resolve(
  workspaceRoot,
  extensionRouteRegistryRelativePath
)

const webRouteRegistrySource = readFileSync(webRouteRegistryPath, "utf8")
const extensionRouteRegistrySource = readFileSync(extensionRouteRegistryPath, "utf8")
// The web UI keeps nav metadata in the shared settings nav config, while the
// extension registry still declares nav blocks inline on route definitions.
const webNavConfigPath = resolve(
  workspaceRoot,
  "apps/packages/ui/src/components/Layouts/settings-nav-config.ts"
)
const webNavConfigSource = readFileSync(webNavConfigPath, "utf8")

const extractFamilyWizardNavEntry = (source: string): string => {
  const startIndex = source.indexOf('path: "/settings/family-guardrails"')
  if (startIndex === -1) {
    throw new Error("family wizard nav entry not found")
  }
  const remainder = source.slice(startIndex + 1)
  const nextEntryOffset = remainder.search(/path:\s*"/)
  return source.slice(
    startIndex,
    nextEntryOffset === -1 ? undefined : startIndex + 1 + nextEntryOffset
  )
}
const webRouteModulePath = resolve(
  workspaceRoot,
  "apps/packages/ui/src/routes/option-family-guardrails-wizard.tsx"
)
const extensionRouteModulePath = resolve(
  workspaceRoot,
  "apps/tldw-frontend/extension/routes/option-family-guardrails-wizard.tsx"
)
const webRouteModuleSource = readFileSync(webRouteModulePath, "utf8")
const extensionRouteModuleSource = readFileSync(extensionRouteModulePath, "utf8")

const normalizeSource = (source: string): string =>
  source
    .replace(/\r\n/g, "\n")
    .trim()

describe("family guardrails route parity", () => {
  it("registers the same family wizard settings path in web and extension registries", () => {
    expect(webRouteRegistrySource).toContain('path: "/settings/family-guardrails"')
    expect(extensionRouteRegistrySource).toContain('path: "/settings/family-guardrails"')
  })

  it("uses dedicated family wizard option route modules in both surfaces", () => {
    expect(webRouteRegistrySource).toMatch(
      /const OptionFamilyGuardrailsWizard = lazy\(\s*\(\) => import\("\.\/option-family-guardrails-wizard"\)\s*\)/
    )
    expect(extensionRouteRegistrySource).toMatch(
      /const OptionFamilyGuardrailsWizard = lazy\(\s*\(\) => import\("\.\/option-family-guardrails-wizard"\)\s*\)/
    )
  })

  it("keeps family wizard navigation metadata aligned", () => {
    const webNavEntry = extractFamilyWizardNavEntry(webNavConfigSource)
    const extensionNavEntry = extractFamilyWizardNavEntry(
      extensionRouteRegistrySource
    )

    // Both surfaces label the wizard nav entry via the same translation key.
    expect(webNavEntry).toContain('labelToken: "settings:familyGuardrailsWizardNav"')
    expect(extensionNavEntry).toContain(
      'labelToken: "settings:familyGuardrailsWizardNav"'
    )
    // Both surfaces pin an explicit nav group and order for the entry.
    expect(webNavEntry).toMatch(/group:\s*"[a-zA-Z]+"/)
    expect(extensionNavEntry).toMatch(/group:\s*"[a-zA-Z]+"/)
    expect(webNavEntry).toMatch(/order:\s*\d/)
    expect(extensionNavEntry).toMatch(/order:\s*\d/)
  })

  it("keeps the dedicated family wizard route modules in sync", () => {
    expect(normalizeSource(extensionRouteModuleSource)).toBe(normalizeSource(webRouteModuleSource))
  })
})
