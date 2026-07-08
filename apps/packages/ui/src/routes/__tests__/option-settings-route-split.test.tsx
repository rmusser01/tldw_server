import { readFile } from "node:fs/promises"
import { resolve } from "node:path"

import { describe, expect, it } from "vitest"

describe("settings route split", () => {
  it("registers setup and preferences as separate settings routes", async () => {
    const source = await readFile(
      resolve(
        __dirname,
        "../option-settings-route-registry.tsx"
      ),
      "utf8"
    )

    expect(source).toContain('path: "/settings"')
    expect(source).toContain('path: "/settings/preferences"')
    expect(source).toContain("setup-recovery-settings")
    expect(source).toContain("preferences-settings")
  })

  it("keeps health and processed settings routes inside the shared settings layout", async () => {
    const healthSource = await readFile(
      resolve(__dirname, "../option-settings-health.tsx"),
      "utf8"
    )
    const processedSource = await readFile(
      resolve(__dirname, "../option-settings-processed.tsx"),
      "utf8"
    )

    expect(healthSource).toContain("SettingsRoute")
    expect(processedSource).toContain("SettingsRoute")
  })

  it("exposes a hosted /settings/data page", async () => {
    const dataPageSource = await readFile(
      resolve(
        __dirname,
        "../../../../../tldw-frontend/pages/settings/data.tsx"
      ),
      "utf8"
    )

    expect(dataPageSource).toContain("DataManagementSettings")
    expect(dataPageSource).toContain("SettingsRoute")
  })

  it("sets document titles on the hosted settings pages", async () => {
    const pagePaths = [
      "../../../../../tldw-frontend/pages/settings/index.tsx",
      "../../../../../tldw-frontend/pages/settings/preferences.tsx",
      "../../../../../tldw-frontend/pages/settings/ui.tsx",
      "../../../../../tldw-frontend/pages/settings/data.tsx"
    ]

    for (const pagePath of pagePaths) {
      const source = await readFile(resolve(__dirname, pagePath), "utf8")
      expect(source).toContain("next/head")
      expect(source).toContain("<title>")
    }
  })

  it("lists setup, preferences, and data settings in hosted smoke inventories", async () => {
    const pageMapping = await readFile(
      resolve(__dirname, "../../../../../tldw-frontend/e2e/page-mapping.ts"),
      "utf8"
    )
    const pageInventory = await readFile(
      resolve(
        __dirname,
        "../../../../../tldw-frontend/e2e/smoke/page-inventory.ts"
      ),
      "utf8"
    )

    expect(pageMapping).toContain('webuiPath: "/settings"')
    expect(pageMapping).toContain('webuiPath: "/settings/preferences"')
    expect(pageMapping).toContain('webuiPath: "/settings/data"')
    expect(pageInventory).toContain('path: "/settings/preferences"')
    expect(pageInventory).toContain('path: "/settings/data"')
  })
})
