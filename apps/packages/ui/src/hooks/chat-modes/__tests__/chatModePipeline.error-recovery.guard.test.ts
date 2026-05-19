import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("chatModePipeline interruption metadata guard", () => {
  it("marks interrupted generations with fallback-friendly metadata", () => {
    const sourcePath = path.resolve(__dirname, "../chatModePipeline.ts")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain("interrupted: true")
    expect(source).toContain("interruptionReason")
    expect(source).toContain("interruptedAt: Date.now()")
  })

  it("returns a skipped submit result for user-initiated aborts", () => {
    const sourcePath = path.resolve(__dirname, "../chatModePipeline.ts")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain("isAbortLikeError")
    expect(source).toContain("return chatSubmitSkipped(")
    expect(source).toMatch(/if \(isAbort\)[\s\S]*return chatSubmitSkipped/)
  })
})
