import { describe, expect, it } from "vitest"
import {
  composerMentionsAvailable,
  composerPlaceholder
} from "../composer-placeholder"

const t = (_key: string, defaultValue: string) => defaultValue

describe("composer placeholder (CC-11)", () => {
  it("mentions '@' only in the extension with tab mentions turned on", () => {
    expect(composerMentionsAvailable(true, true)).toBe(true)
    expect(composerMentionsAvailable(false, true)).toBe(false)
    expect(composerMentionsAvailable(undefined, true)).toBe(false)
    expect(composerMentionsAvailable(true, false)).toBe(false)
  })

  it("advertises only slash commands when mentions are unavailable", () => {
    expect(composerPlaceholder(t, false)).toBe("Type a message... (/ for commands)")
    expect(composerPlaceholder(t, true)).toBe(
      "Type a message... (/ commands, @ mentions)"
    )
  })
})
