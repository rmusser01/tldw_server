import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const loadSource = (...candidates: string[]) => {
  const path = candidates.find((candidate) => existsSync(candidate))
  if (!path) {
    throw new Error(`Missing settings MCP Hub page shim: ${candidates.join(" | ")}`)
  }
  return readFileSync(path, "utf8")
}

describe("settings MCP Hub Next.js page shim", () => {
  it("loads the settings-shell MCP Hub route wrapper", () => {
    const source = loadSource(
      "pages/settings/mcp-hub.tsx",
      "tldw-frontend/pages/settings/mcp-hub.tsx",
      "apps/tldw-frontend/pages/settings/mcp-hub.tsx"
    )

    expect(source).toContain('import("@/routes/option-settings-mcp-hub")')
    expect(source).not.toContain('import("@/routes/option-mcp-hub")')
  })
})
