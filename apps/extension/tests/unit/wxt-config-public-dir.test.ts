import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

import { describe, expect, test } from "bun:test"

import config from "../../wxt.config.ts"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const extensionRoot = path.resolve(testDir, "../..")

describe("wxt config publicDir", () => {
  test("points directly at shared ui public assets", () => {
    const expectedSharedPublicDir = path.resolve(extensionRoot, "../packages/ui/src/public")
    const legacySymlinkPath = path.join(extensionRoot, "public")

    expect(config.publicDir).toBe(expectedSharedPublicDir)
    expect(config.publicDir).not.toBe(legacySymlinkPath)
  })

  test("contains every root-relative font referenced by shared CSS", () => {
    const sharedPublicDir = path.resolve(extensionRoot, "../packages/ui/src/public")
    const sharedCssPath = path.resolve(
      extensionRoot,
      "../packages/ui/src/assets/tailwind-shared.css"
    )
    const sharedCss = readFileSync(sharedCssPath, "utf8")
    const referencedFonts = [
      ...sharedCss.matchAll(/url\(["']?\/fonts\/([^"')]+)["']?\)/g),
    ].map((match) => match[1])

    expect(referencedFonts.length).toBeGreaterThan(0)
    for (const fontFile of referencedFonts) {
      expect(existsSync(path.join(sharedPublicDir, "fonts", fontFile))).toBe(true)
    }
  })
})
