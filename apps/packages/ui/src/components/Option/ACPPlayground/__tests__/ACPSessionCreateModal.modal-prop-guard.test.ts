import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import path from "node:path"
import { describe, expect, it } from "vitest"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const sourcePath = path.resolve(
  testDir,
  "../ACPSessionCreateModal.tsx"
)

describe("ACPSessionCreateModal modal prop guard", () => {
  it("uses destroyOnHidden instead of the deprecated destroyOnClose prop", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain("destroyOnHidden")
    expect(source).not.toContain("destroyOnClose")
  })

  it("uses the design-system registry for the ready step fallback label", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain('getDesignSystemState("ready").label')
    expect(source).toContain('t("acp.create.steps.ready"')
  })

  it("uses the shared product-state Alert adapter for creation errors", () => {
    const source = readFileSync(sourcePath, "utf8")
    const antdImport = source.match(/import\s*\{[\s\S]*?\}\s*from\s*"antd"/)?.[0] ?? ""

    expect(source).toContain(
      'import { ProductStateAlert as Alert } from "@/components/Option/productStatePrimitives"'
    )
    expect(antdImport).not.toMatch(/\bAlert\b/)
    expect(source).toContain("<Alert")
  })
})
