import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

type ProductStateBaselineEntry = {
  id: string
  path: string
  rule: string
  subject: string
}

const testDir = path.dirname(fileURLToPath(import.meta.url))
const packageRoot = path.resolve(testDir, "../../..")
const baselinePath = path.resolve(
  packageRoot,
  "scripts/design-system-product-state-baseline.json"
)
const scopedPrefixes = [
  "src/components/Option/MCPHub/",
  "src/components/Option/ACPPlayground/",
  "src/components/Option/WorkspacePlayground/",
]

const readBaseline = (): ProductStateBaselineEntry[] =>
  JSON.parse(readFileSync(baselinePath, "utf8"))

describe("MCP/ACP/Workspace product-state baseline ownership", () => {
  it("keeps owned MCP, ACP, and Workspace paths out of the legacy baseline", () => {
    const scopedEntries = readBaseline().filter((entry) =>
      scopedPrefixes.some((prefix) => entry.path.startsWith(prefix))
    )

    expect(scopedEntries).toEqual([])
  })
})
