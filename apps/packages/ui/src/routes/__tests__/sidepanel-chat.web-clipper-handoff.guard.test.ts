import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const sourcePath = path.resolve(testDir, "../sidepanel-chat.tsx")

describe("sidepanel-chat web clipper handoff", () => {
  it("consumes pending web clipper analyze requests through onSubmit", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain("readPendingWebClipAnalyzeRequest")
    expect(source).toContain("WEB_CLIPPER_ANALYZE_MESSAGE_TYPE")
    expect(source).toContain("requestOverrides: pendingAnalyze.requestOverrides")
    expect(source).toContain("clearPendingWebClipAnalyzeRequest(pendingAnalyze.id)")
    expect(source).toContain("hasSubmittedWebClipAnalyzeMessage")
  })
})
