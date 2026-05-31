import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const sourcePath = path.resolve(__dirname, "..", "ChatPane", "index.tsx")

describe("ChatPane input availability guard", () => {
  it("disables the textarea when chat is unavailable and blocks submit when sending is unavailable", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain(
      "disabled={isLoading || isPreparingContext || isChatUnavailable}"
    )
    expect(source).toMatch(
      /disabled=\{\s*!value\.trim\(\)\s*\|\|\s*isPreparingContext\s*\|\|\s*isChatUnavailable\s*\|\|\s*isSendBlocked\s*\}/
    )
  })
})
