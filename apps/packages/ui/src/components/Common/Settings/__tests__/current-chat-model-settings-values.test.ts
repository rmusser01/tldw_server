import { describe, expect, it } from "vitest"

import { normalizeCurrentChatModelSettingValue } from "../current-chat-model-settings-values"

describe("current chat model setting values", () => {
  it("coerces numeric form strings before saving scoped settings", () => {
    expect(normalizeCurrentChatModelSettingValue("temperature", "0.31")).toBe(
      0.31
    )
    expect(normalizeCurrentChatModelSettingValue("numCtx", "8192")).toBe(8192)
  })

  it("keeps non-numeric setting values unchanged", () => {
    expect(normalizeCurrentChatModelSettingValue("apiProvider", "openai")).toBe(
      "openai"
    )
    expect(normalizeCurrentChatModelSettingValue("jsonMode", true)).toBe(true)
  })

  it("stores blank numeric inputs as unset values", () => {
    expect(normalizeCurrentChatModelSettingValue("temperature", "")).toBeUndefined()
    expect(normalizeCurrentChatModelSettingValue("topP", "   ")).toBeUndefined()
  })
})
