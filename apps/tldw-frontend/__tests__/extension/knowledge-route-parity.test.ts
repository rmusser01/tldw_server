import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const readFirstExisting = (candidates: string[], label: string) => {
  const path = candidates.find((candidate) => existsSync(candidate))
  if (!path) {
    throw new Error(`Unable to locate ${label}`)
  }
  return readFileSync(path, "utf8")
}

const activeExtensionOptionsEntry = readFirstExisting(
  [
    "apps/extension/entrypoints/options/main.tsx",
    "../extension/entrypoints/options/main.tsx",
    "extension/entrypoints/options/main.tsx"
  ],
  "active extension options entrypoint"
)

const activeSharedRouteRegistry = readFirstExisting(
  [
    "apps/packages/ui/src/routes/route-registry.tsx",
    "../packages/ui/src/routes/route-registry.tsx",
    "packages/ui/src/routes/route-registry.tsx"
  ],
  "shared route registry"
)

const activeKnowledgeRoute = readFirstExisting(
  [
    "apps/packages/ui/src/routes/option-knowledge.tsx",
    "../packages/ui/src/routes/option-knowledge.tsx",
    "packages/ui/src/routes/option-knowledge.tsx"
  ],
  "shared knowledge route"
)

const legacyExtensionRouteRegistry = readFirstExisting(
  [
    "apps/tldw-frontend/extension/routes/route-registry.tsx",
    "extension/routes/route-registry.tsx",
    "tldw-frontend/extension/routes/route-registry.tsx"
  ],
  "legacy extension route registry"
)

const legacyKnowledgeRoute = readFirstExisting(
  [
    "apps/tldw-frontend/extension/routes/option-knowledge.tsx",
    "extension/routes/option-knowledge.tsx",
    "tldw-frontend/extension/routes/option-knowledge.tsx"
  ],
  "legacy extension knowledge route"
)

const extensionPageInventory = readFirstExisting(
  [
    "apps/extension/tests/e2e/page-inventory.ts",
    "../extension/tests/e2e/page-inventory.ts",
    "extension/tests/e2e/page-inventory.ts"
  ],
  "extension page inventory"
)

const webPageMapping = readFirstExisting(
  [
    "apps/tldw-frontend/e2e/page-mapping.ts",
    "e2e/page-mapping.ts"
  ],
  "web page mapping"
)

const webSmokePageInventory = readFirstExisting(
  [
    "apps/tldw-frontend/e2e/smoke/page-inventory.ts",
    "e2e/smoke/page-inventory.ts"
  ],
  "web smoke page inventory"
)

describe("extension /knowledge route parity", () => {
  it("uses the shared UI options app as the shipped extension route graph", () => {
    expect(activeExtensionOptionsEntry).toMatch(
      /import\s+["']@tldw\/ui\/entries\/options\/main["']/
    )
    expect(activeExtensionOptionsEntry).not.toContain(
      "tldw-frontend/extension/routes"
    )
  })

  it("routes active extension options knowledge paths to the shared KnowledgeQA workspace", () => {
    expect(activeSharedRouteRegistry).toMatch(/path:\s*"\/settings\/knowledge"/)
    expect(activeSharedRouteRegistry).toMatch(/path:\s*"\/knowledge"/)
    expect(activeSharedRouteRegistry).toMatch(
      /path:\s*"\/knowledge\/thread\/:threadId"/
    )
    expect(activeSharedRouteRegistry).toMatch(
      /path:\s*"\/knowledge\/shared\/:shareToken"/
    )
    expect(activeKnowledgeRoute).toContain("KnowledgeQA")
    expect(activeKnowledgeRoute).not.toContain("KnowledgeSettings")
  })

  it("keeps the legacy route mirror safe if it is imported by old tests or builds", () => {
    expect(legacyExtensionRouteRegistry).toMatch(/path:\s*"\/knowledge"/)
    expect(legacyExtensionRouteRegistry).toMatch(
      /path:\s*"\/knowledge\/thread\/:threadId"/
    )
    expect(legacyExtensionRouteRegistry).toMatch(
      /path:\s*"\/knowledge\/shared\/:shareToken"/
    )
    expect(legacyKnowledgeRoute).toContain("KnowledgeQA")
    expect(legacyKnowledgeRoute).not.toContain("KnowledgeSettings")
  })

  it("describes extension /knowledge as Knowledge QA rather than settings-only knowledge", () => {
    expect(extensionPageInventory).toMatch(
      /path:\s*"\/knowledge",\s*name:\s*"Knowledge QA"/
    )
    expect(extensionPageInventory).toMatch(
      /path:\s*"\/knowledge\/thread\/page-inventory-thread"/
    )
    expect(extensionPageInventory).toMatch(
      /path:\s*"\/knowledge\/shared\/page-inventory-share-token"/
    )
    const knowledgeQaBlock = webPageMapping.match(
      /\{[\s\S]*?name:\s*"Knowledge QA Workspace"[\s\S]*?\n\s*\},/
    )?.[0]
    expect(knowledgeQaBlock).toBeDefined()
    expect(knowledgeQaBlock).toContain('webuiPath: "/knowledge"')
    expect(knowledgeQaBlock).toContain('sharedComponent: "KnowledgeQA"')
    expect(webSmokePageInventory).toMatch(
      /path:\s*"\/knowledge",\s*name:\s*"Knowledge QA"/
    )
  })
})
