import { describe, expect, it } from "vitest"
import {
  composerMentionsAvailable,
  composerShortcutHint
} from "../composer-shortcut-hint"

const t = (_key: string, defaultValue: string) => defaultValue

describe("composer shortcut hint (CC-11)", () => {
  it("mentions '@' only in the extension with tab mentions turned on", () => {
    expect(composerMentionsAvailable(true, true)).toBe(true)
    expect(composerMentionsAvailable(false, true)).toBe(false)
    expect(composerMentionsAvailable(undefined, true)).toBe(false)
    expect(composerMentionsAvailable(true, false)).toBe(false)
  })

  it("advertises only slash commands when mentions are unavailable", () => {
    expect(composerShortcutHint(t, false)).toBe("/ commands")
    expect(composerShortcutHint(t, true)).toBe("/ commands · @ mentions")
  })
})
