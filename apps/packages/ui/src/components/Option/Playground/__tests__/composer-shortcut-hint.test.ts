import { describe, expect, it } from "vitest"
import {
  composerMentionsAvailable,
  composerShortcutHint
} from "../composer-shortcut-hint"

const t = (_key: string, defaultValue: string) => defaultValue

describe("composer shortcut hint (CC-11)", () => {
  it("mentions '@' in the extension with tab mentions turned on", () => {
    expect(composerMentionsAvailable(true, true)).toBe(true)
    expect(composerMentionsAvailable(false, true)).toBe(false)
    expect(composerMentionsAvailable(undefined, true)).toBe(false)
  })

  // Decision [4]: on the web '@' searches notes, so it is named only when Notes is there.
  it("mentions '@' on the web only when the server has Notes", () => {
    expect(composerMentionsAvailable(true, false, true)).toBe(true)
    expect(composerMentionsAvailable(false, false, true)).toBe(true)
    expect(composerMentionsAvailable(true, false, false)).toBe(false)
    expect(composerMentionsAvailable(false, false)).toBe(false)
  })

  it("advertises only slash commands when mentions are unavailable", () => {
    expect(composerShortcutHint(t, false)).toBe("/ commands")
    expect(composerShortcutHint(t, true)).toBe("/ commands · @ mentions")
  })
})
