import { describe, expect, it } from "vitest"
import { CHAT_MESSAGE_VIEW_SETTING } from "@/services/settings/ui-settings"
import { normalizeSettingValue } from "@/services/settings/registry"

describe("chat message view preference", () => {
  it("starts new readers in Transcript", () => {
    expect(CHAT_MESSAGE_VIEW_SETTING.defaultValue).toBe("transcript")
  })
  it.each(["transcript", "conversation", "research"])("retains an explicit %s choice", value => {
    expect(normalizeSettingValue(CHAT_MESSAGE_VIEW_SETTING, value)).toBe(value)
  })
  it.each([undefined, null, "", "compact", {}, 3])("recovers an invalid preference without changing chat data: %s", value => {
    expect(normalizeSettingValue(CHAT_MESSAGE_VIEW_SETTING, value)).toBe("transcript")
  })
})
