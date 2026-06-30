import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("Workspaces Next page wrapper", () => {
  it("routes /workspaces to the shared Workspaces manager without SSR", () => {
    const pagePath = path.resolve(process.cwd(), "pages/workspaces.tsx")

    expect(existsSync(pagePath)).toBe(true)

    const source = readFileSync(pagePath, "utf8")
    expect(source).toContain("@/routes/option-workspaces")
    expect(source).toMatch(/ssr:\s*false/)
  })
})
