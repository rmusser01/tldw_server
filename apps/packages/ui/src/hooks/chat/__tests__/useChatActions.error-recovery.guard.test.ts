import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

/**
 * Guard test: ensures critical recovery patterns exist in the
 * useChatActions module (facade + sub-modules).
 */
const readChatActionsSources = () => {
  const dir = path.resolve(__dirname, "..")
  const files = [
    "useChatActions.ts",
    "useCharacterChatMode.ts",
    "chat-action-utils.ts",
  ]
  return files
    .map((f) => {
      try {
        return fs.readFileSync(path.join(dir, f), "utf8")
      } catch {
        return ""
      }
    })
    .join("\n")
}

describe("useChatActions interruption recovery guard", () => {
  it("marks interrupted assistant variants with recovery metadata", () => {
    const source = readChatActionsSources()

    expect(source).toContain("interrupted: true")
    expect(source).toContain("interruptionReason")
    expect(source).toContain("interruptedAt: Date.now()")
    expect(source).toContain("imageGenerationRefine")
    expect(source).toContain("attemptCharacterStreamRecoveryPersist")
    expect(source).toContain("streamTransportInterrupted")
    expect(source).toContain("partialResponseSaved")
    expect(source).toContain("resolveSavedDegradedCharacterPersist")
  })
})
