import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

/**
 * Guard test: ensures the stream-inactivity watchdog actually lives in the two
 * shipped (LIVE) character-streaming paths, not just the extracted (formerly
 * unused) copy. Task-12108 / round-2 audit finding R5.
 */
const readSource = (relativePath: string) => {
  const file = path.resolve(__dirname, relativePath)
  try {
    return fs.readFileSync(file, "utf8")
  } catch {
    return ""
  }
}

const LIVE_SOURCES: Array<{ name: string; source: string }> = [
  {
    // Playground / Option path
    name: "useChatActions.ts",
    source: readSource("../useChatActions.ts")
  },
  {
    // Sidepanel path
    name: "useMessage.tsx",
    source: readSource("../../useMessage.tsx")
  }
]

describe("live characterChatMode stream-inactivity watchdog guard", () => {
  it.each(LIVE_SOURCES)(
    "$name arms a 60s inactivity watchdog that aborts a stalled stream",
    ({ source }) => {
      expect(source.length).toBeGreaterThan(0)
      // Watchdog wiring
      expect(source).toContain("STREAM_INACTIVITY_TIMEOUT_MS = 60_000")
      expect(source).toContain("resetInactivityTimer")
      expect(source).toContain("inactivityAborted = true")
      // The watchdog must reset on each received chunk and re-throw on timeout.
      expect(source).toContain("StreamInactivityTimeout")
      // The abort must be surfaced/recovered, not swallowed.
      expect(source).toContain("buildCharacterChatAssistantErrorContent")
    }
  )

  it("keeps the watchdog value aligned with the extracted reference copy", () => {
    const extracted = readSource("../useCharacterChatMode.ts")
    expect(extracted).toContain("STREAM_INACTIVITY_TIMEOUT_MS = 60_000")
  })
})
