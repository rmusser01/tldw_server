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

  it("registers the shared Workflow prompts editor in every settings host", async () => {
    const sources = await Promise.all([
      readFile(resolve(__dirname, "../option-settings-route-registry.tsx"), "utf8"),
      readFile(resolve(__dirname, "../route-registry.tsx"), "utf8"),
      readFile(
        resolve(
          __dirname,
          "../../../../../tldw-frontend/extension/routes/route-registry.tsx"
        ),
        "utf8"
      )
    ])

    for (const source of sources) {
      expect(source).toContain('path: "/settings/prompt"')
      expect(source).toContain("ServicePromptsSettings")
      expect(source).not.toContain('"PromptWorkspaceSettings"')
    }
  })

  it("keeps reusable Prompt Library navigation separate from Workflow prompts", async () => {
    const [omniSource, promptSearchSource] = await Promise.all([
      readFile(resolve(__dirname, "../../hooks/useOmniSearchDeps.tsx"), "utf8"),
      readFile(resolve(__dirname, "../../components/Common/PromptSearch.tsx"), "utf8")
    ])

    expect(omniSource).toContain('id: "prompts"')
    expect(omniSource).toContain('route: "/prompts"')
    expect(omniSource).toContain('id: "settings-workflow-prompts"')
    expect(omniSource).toContain('route: "/settings/prompt"')
    expect(omniSource).toContain('labelKey: "settings:servicePrompts.title"')
    expect(omniSource).toContain('description: "Edit instructions used by supported workflows"')
    expect(promptSearchSource).toContain('to="/prompts"')
    expect(promptSearchSource).not.toContain('to="/settings/prompt"')
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
    const pageTitles = [
      [
        "../../../../../tldw-frontend/pages/settings/index.tsx",
        "Setup &amp; Recovery | Settings | tldw"
      ],
      [
        "../../../../../tldw-frontend/pages/settings/preferences.tsx",
        "Preferences | Settings | tldw"
      ],
      [
        "../../../../../tldw-frontend/pages/settings/ui.tsx",
        "UI Customization | Settings | tldw"
      ],
      [
        "../../../../../tldw-frontend/pages/settings/data.tsx",
        "Data Management | Settings | tldw"
      ]
    ]

    for (const [pagePath, title] of pageTitles) {
      const source = await readFile(resolve(__dirname, pagePath), "utf8")
      expect(source).toContain("next/head")
      expect(source).toContain(`<title>${title}</title>`)
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

  it("names Workflow prompts in both host inventories", async () => {
    const [webInventory, extensionInventory] = await Promise.all([
      readFile(
        resolve(
          __dirname,
          "../../../../../tldw-frontend/e2e/smoke/page-inventory.ts"
        ),
        "utf8"
      ),
      readFile(
        resolve(__dirname, "../../../../../extension/tests/e2e/page-inventory.ts"),
        "utf8"
      )
    ])

    expect(webInventory).toContain(
      '{ path: "/settings/prompt", name: "Workflow prompts", category: "settings" }'
    )
    expect(extensionInventory).toContain(
      '{ kind: "options", path: "/settings/prompt", name: "Workflow prompts" }'
    )
  })
})
