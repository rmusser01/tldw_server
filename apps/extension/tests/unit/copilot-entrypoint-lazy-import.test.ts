import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { resolve } from "node:path"

const entrypointPath = resolve(
  import.meta.dir,
  "../../entrypoints/copilot-popup.content.tsx"
)

describe("copilot content entrypoint", () => {
  test("keeps shared popup implementation behind runtime import", () => {
    const source = readFileSync(entrypointPath, "utf8")

    expect(source).toMatch(
      /import\s*\{\s*defineContentScript\s*\}\s*from\s*["']wxt\/utils\/define-content-script["']/
    )
    expect(source).toContain("defineContentScript")
    expect(source).toMatch(
      /import\(\s*["']@tldw\/ui\/entries\/copilot-popup\.content["']\s*\)/
    )
    expect(source).toMatch(/\btry\s*\{/)
    expect(source).toMatch(/\bcatch\s*\(\s*error\s*\)/)
    expect(source).toContain("Failed to load copilot popup content entrypoint")
    expect(source).toContain("cause: error")
    expect(source).not.toContain(
      'export { default } from "@tldw/ui/entries/copilot-popup.content"'
    )
  })
})
