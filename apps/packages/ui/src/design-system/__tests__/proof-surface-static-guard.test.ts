import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const resolveAppsRoot = () => {
  const here = path.dirname(fileURLToPath(import.meta.url))
  const appsRoot = path.resolve(here, "../../../../..")
  if (
    fs.existsSync(path.resolve(appsRoot, "packages/ui")) &&
    fs.existsSync(path.resolve(appsRoot, "tldw-frontend"))
  ) {
    return appsRoot
  }
  throw new Error(`Unable to resolve apps root from ${here}; computed ${appsRoot}`)
}

const appsRoot = resolveAppsRoot()

const readAppFile = (relativePath: string) =>
  fs.readFileSync(path.resolve(appsRoot, relativePath), "utf8")

const proofSurfaceFiles = [
  "packages/ui/src/components/Common/BackendUnavailableRecovery.tsx",
  "tldw-frontend/components/networking/ConfigurationErrorScreen.tsx",
  "tldw-frontend/components/networking/ServerReadinessGate.tsx",
  "packages/ui/src/routes/option-setup.tsx",
  "packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx",
  "packages/ui/src/components/Option/Settings/health-status.tsx",
  "packages/ui/src/components/Option/Admin/ServerAdminPage.tsx"
]

describe("design-system proof surface static guard", () => {
  it("keeps the v1 proof surface pinned to recovery, setup, health, and admin entry points", () => {
    expect(proofSurfaceFiles).toEqual([
      "packages/ui/src/components/Common/BackendUnavailableRecovery.tsx",
      "tldw-frontend/components/networking/ConfigurationErrorScreen.tsx",
      "tldw-frontend/components/networking/ServerReadinessGate.tsx",
      "packages/ui/src/routes/option-setup.tsx",
      "packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx",
      "packages/ui/src/components/Option/Settings/health-status.tsx",
      "packages/ui/src/components/Option/Admin/ServerAdminPage.tsx"
    ])

    for (const file of proofSurfaceFiles) {
      expect(fs.existsSync(path.resolve(appsRoot, file)), file).toBe(true)
    }
  })

  it("keeps backend recovery and readiness gates on canonical state primitives", () => {
    const backendUnavailable = readAppFile(
      "packages/ui/src/components/Common/BackendUnavailableRecovery.tsx"
    )
    const readinessGate = readAppFile(
      "tldw-frontend/components/networking/ServerReadinessGate.tsx"
    )
    const configurationError = readAppFile(
      "tldw-frontend/components/networking/ConfigurationErrorScreen.tsx"
    )

    expect(backendUnavailable).toContain("RecoveryCallout")
    expect(backendUnavailable).toContain('state="unavailable"')
    expect(readinessGate).toContain("StatePanel")
    expect(readinessGate).toContain('isRetrying ? "retrying" : "loading"')
    expect(readinessGate).not.toContain("style={{")
    expect(configurationError).toContain("SetupRequiredPanel")
    expect(configurationError).not.toMatch(/background:\s*["'`]?#/i)
  })

  it("keeps setup, health, and admin screens on canonical state labels and actions", () => {
    const setupRoute = readAppFile("packages/ui/src/routes/option-setup.tsx")
    const onboarding = readAppFile(
      "packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx"
    )
    const healthStatus = readAppFile(
      "packages/ui/src/components/Option/Settings/health-status.tsx"
    )
    const adminPage = readAppFile(
      "packages/ui/src/components/Option/Admin/ServerAdminPage.tsx"
    )

    expect(setupRoute).toContain("SetupRequiredPanel")
    expect(onboarding).toContain('getDesignSystemState("setup_required")')
    expect(onboarding).toContain('getDesignSystemState("auth_required")')
    expect(onboarding).toContain('getDesignSystemState("retrying")')
    expect(onboarding).toContain("progressHeaderState")
    expect(onboarding).not.toContain("<span>{retryingState.label}</span>")

    expect(healthStatus).toContain("StatePanel")
    expect(healthStatus).toContain("RecoveryCallout")
    expect(healthStatus).toContain("SetupRequiredPanel")
    expect(healthStatus).toContain("'Ready'")
    expect(healthStatus).toContain("'Degraded'")
    expect(healthStatus).toContain("'Loading'")

    expect(adminPage).toContain("PermissionNotice")
    expect(adminPage).toContain("RecoveryCallout")
    expect(adminPage).toContain("StatePanel")
    expect(adminPage).toContain("openAdminDocumentation")
  })
})
