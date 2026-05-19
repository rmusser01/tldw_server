import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readShellSource = () =>
  fs.readFileSync(
    path.resolve(__dirname, "..", "ModerationPlaygroundShell.tsx"),
    "utf8"
  )

describe("ModerationPlaygroundShell skeleton loaders", () => {
  it("uses Skeleton component instead of plain Loading text in Suspense fallbacks", () => {
    const source = readShellSource()
    expect(source).toContain("Skeleton")
    // Should NOT have bare "Loading..." text in fallbacks
    const fallbackMatches = source.match(/fallback=\{<div[^>]*>Loading\.\.\.<\/div>\}/g)
    expect(fallbackMatches).toBeNull()
  })
})
