import { readFileSync } from "node:fs"
import { resolve } from "node:path"

import { describe, expect, it } from "vitest"

describe("settings route split", () => {
  it("registers setup and preferences as separate settings routes", () => {
    const source = readFileSync(
      resolve(
        process.cwd(),
        "../packages/ui/src/routes/option-settings-route-registry.tsx"
      ),
      "utf8"
    )

    expect(source).toContain('path: "/settings"')
    expect(source).toContain('path: "/settings/preferences"')
    expect(source).toContain("setup-recovery-settings")
    expect(source).toContain("preferences-settings")
  })

  it("keeps health and processed settings routes inside the shared settings layout", () => {
    const healthSource = readFileSync(
      resolve(process.cwd(), "../packages/ui/src/routes/option-settings-health.tsx"),
      "utf8"
    )
    const processedSource = readFileSync(
      resolve(process.cwd(), "../packages/ui/src/routes/option-settings-processed.tsx"),
      "utf8"
    )

    expect(healthSource).toContain("SettingsRoute")
    expect(processedSource).toContain("SettingsRoute")
  })

  it("exposes a hosted /settings/data page", () => {
    const dataPageSource = readFileSync(
      resolve(process.cwd(), "pages/settings/data.tsx"),
      "utf8"
    )

    expect(dataPageSource).toContain("DataManagementSettings")
    expect(dataPageSource).toContain("SettingsRoute")
  })

  it("lists setup, preferences, and data settings in hosted smoke inventories", () => {
    const pageMapping = readFileSync(
      resolve(process.cwd(), "e2e/page-mapping.ts"),
      "utf8"
    )
    const pageInventory = readFileSync(
      resolve(process.cwd(), "e2e/smoke/page-inventory.ts"),
      "utf8"
    )

    expect(pageMapping).toContain('webuiPath: "/settings"')
    expect(pageMapping).toContain('webuiPath: "/settings/preferences"')
    expect(pageMapping).toContain('webuiPath: "/settings/data"')
    expect(pageInventory).toContain('path: "/settings/preferences"')
    expect(pageInventory).toContain('path: "/settings/data"')
  })
})
