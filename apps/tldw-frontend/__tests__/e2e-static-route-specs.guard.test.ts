import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const STATIC_ROUTE_SPECS = [
  "e2e/workflows/hosted-placeholder-routes.spec.ts",
  "e2e/workflows/route-placeholder-settings.spec.ts",
  "e2e/workflows/tier-2-features/documentation.spec.ts",
  "e2e/workflows/tier-4-admin/privileges.spec.ts",
  "e2e/workflows/tier-4-admin/profile-companion.spec.ts",
  "e2e/workflows/tier-4-admin/settings-full.spec.ts",
  "e2e/workflows/tier-1-critical/settings-core.spec.ts",
  "e2e/workflows/tier-5-specialized/model-playground.spec.ts",
  "e2e/workflows/tier-5-specialized/skills.spec.ts",
  "e2e/workflows/tier-5-specialized/claims-review.spec.ts",
  "e2e/workflows/tier-5-specialized/repo2txt.spec.ts",
  "e2e/workflows/tier-5-specialized/osint.spec.ts",
  "e2e/workflows/tier-5-specialized/researchers.spec.ts",
  "e2e/workflows/tier-5-specialized/journalists.spec.ts",
  "e2e/workflows/tier-5-specialized/moderation-routes.spec.ts",
  "e2e/workflows/tier-5-specialized/chunking-playground.spec.ts",
]

describe("static route specs", () => {
  it("avoid brittle networkidle waits", () => {
    for (const relativePath of STATIC_ROUTE_SPECS) {
      const source = readFileSync(path.join(process.cwd(), relativePath), "utf8")
      expect(source).not.toContain('waitForLoadState("networkidle")')
    }
  })
})
