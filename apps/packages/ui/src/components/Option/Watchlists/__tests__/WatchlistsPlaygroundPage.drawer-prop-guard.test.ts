import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import path from "node:path"
import { describe, expect, it } from "vitest"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const sourcePath = path.resolve(
  testDir,
  "../WatchlistsPlaygroundPage.tsx"
)

describe("WatchlistsPlaygroundPage drawer prop guard", () => {
  it("uses Drawer size instead of the deprecated width prop", () => {
    const source = readFileSync(sourcePath, "utf8")
    const drawerMarkup = source.match(/<Drawer[\s\S]*?<\/Drawer>/)?.[0]

    expect(drawerMarkup).toContain("size={isConstrained ? \"100%\" : 520}")
    expect(drawerMarkup).not.toContain("width=")
  })
})
