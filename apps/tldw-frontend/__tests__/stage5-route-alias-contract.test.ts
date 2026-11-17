import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"
import ModerationPlaygroundRedirectPage from "@web/pages/moderation-playground"
import { RouteRedirect } from "@web/components/navigation/RouteRedirect"

const readSource = (relativePath: string) =>
  readFileSync(path.resolve(__dirname, "..", relativePath), "utf8")

describe("stage 5 route alias contract", () => {
  it("requires the Claims decision workspace at its own route", () => {
    const source = readSource("e2e/smoke/stage5-release-gate.spec.ts")
    const claimsRoute = source.match(/path: "\/claims-review",[\s\S]*?\}/)?.[0]

    expect(claimsRoute).toContain('name: "Claims Review"')
    expect(claimsRoute).not.toContain("expectedPath")
    expect(claimsRoute).not.toContain("allowRedirectPanel")
    expect(readSource("e2e/smoke/page-inventory.ts")).toMatch(/path: "\/claims-review",\s*name: "Claims Review",\s*category: "workspace"\s*\}/)
    expect(readSource("e2e/workflows/tier-5-specialized/claims-review.spec.ts")).toContain('name: "Approve selected claims"')
  })

  it("preserves moderation playground as a redirect alias to content rules", () => {
    // This legacy alias is covered here; it is no longer in Stage 5's
    // critical-route subset. Exercise the real page's redirect configuration.
    const page = ModerationPlaygroundRedirectPage()

    expect(page.type).toBe(RouteRedirect)
    expect(page.props).toMatchObject({
      to: "/moderation/rules",
      title: "Moderation Playground has moved",
      description: expect.stringContaining("Content Rules"),
    })
    expect(readSource("e2e/smoke/page-inventory.ts")).toMatch(
      /path: "\/moderation-playground",\s*name: "Moderation Playground Legacy Redirect"/,
    )
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
