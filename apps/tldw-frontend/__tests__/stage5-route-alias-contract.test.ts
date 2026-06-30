import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readSource = (relativePath: string) =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8")

describe("stage 5 route alias contract", () => {
  it("allows the claims-review alias route to satisfy the gate via the redirect panel", () => {
    const source = readSource("e2e/smoke/stage5-release-gate.spec.ts")

    expect(source).toContain("allowRedirectPanel?: boolean")
    expect(source).toContain('path: "/claims-review"')
    expect(source).toContain('name: "Claims Review"')
    expect(source).toContain('expectedPath: "/content-review"')
    expect(source).toContain("allowRedirectPanel: true")
    expect(source).toContain('const redirectPanel = page.getByTestId("route-redirect-panel")')
    expect(source).toContain("let resolvedViaRedirectPanel = false")
    expect(source).toContain("resolvedViaRedirectPanel =")
    expect(source).toContain("await redirectPanel.isVisible().catch(() => false)")
  })

  it("preserves moderation playground as a redirect alias to content rules", () => {
    const releaseGateSource = readSource("e2e/smoke/stage5-release-gate.spec.ts")
    const nextPageSource = readSource("pages/moderation-playground.tsx")

    expect(releaseGateSource).toContain('path: "/moderation-playground"')
    expect(releaseGateSource).toContain('name: "Moderation Playground"')
    expect(releaseGateSource).toContain('expectedPath: "/moderation/rules"')
    expect(releaseGateSource).toContain("allowRedirectPanel: true")
    expect(nextPageSource).toContain("RouteRedirect")
    expect(nextPageSource).toContain('to="/moderation/rules"')
    expect(nextPageSource).toContain("Moderation Playground has moved")
    expect(nextPageSource).toContain("Content Rules")
  })

  it("lists canonical moderation review and content rules routes in smoke inventory", () => {
    const pageInventorySource = readSource("e2e/smoke/page-inventory.ts")
    const pageMappingSource = readSource("e2e/page-mapping.ts")

    expect(pageInventorySource).toContain('path: "/moderation"')
    expect(pageInventorySource).toContain('name: "Moderation Review"')
    expect(pageInventorySource).toContain('path: "/moderation/rules"')
    expect(pageInventorySource).toContain('name: "Content Rules"')
    expect(pageMappingSource).toContain('name: "Moderation Review"')
    expect(pageMappingSource).toContain('webuiPath: "/moderation"')
    expect(pageMappingSource).toContain('name: "Content Rules"')
    expect(pageMappingSource).toContain('webuiPath: "/moderation/rules"')
  })
})
