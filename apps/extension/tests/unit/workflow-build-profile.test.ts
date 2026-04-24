import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const appDir = path.resolve(__dirname, "..", "..")
const repoRoot = path.resolve(appDir, "..", "..")

describe("extension workflow build profile contract", () => {
  it("forces production extension builds in the watchlists required workflow", () => {
    const workflow = readFileSync(
      path.join(repoRoot, ".github", "workflows", "ui-watchlists-extension-e2e.yml"),
      "utf8"
    )

    expect(workflow).toContain("run: bun run build:chrome:prod")
  })
})
