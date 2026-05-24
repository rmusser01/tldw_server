import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const sourcePath = path.resolve(__dirname, "..", "ChatPane", "index.tsx")

describe("ChatPane input availability guard", () => {
  it("disables the textarea when chat is unavailable", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain(
      "disabled={isLoading || isPreparingContext || isChatUnavailable}"
    )
    expect(source).toContain(
      "disabled={!value.trim() || isPreparingContext || isChatUnavailable}"
    )
  })
})
