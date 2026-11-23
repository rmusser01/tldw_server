import { beforeEach, describe, expect, it, vi } from "vitest"

const backing = new Map<string, unknown>()

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: ({ area = "local" }: { area?: string } = {}) => ({
    area,
    watch: () => () => {},
    get: async (key: string) => backing.get(`${area}::${key}`),
    set: async (key: string, value: unknown) => {
      backing.set(`${area}::${key}`, value)
    },
    remove: async (key: string) => {
      backing.delete(`${area}::${key}`)
    }
  })
}))

import { getSetting, setSetting } from "@/services/settings/registry"
import {
  VOICE_CHAT_AUTO_RESUME_SETTING,
  VOICE_CHAT_ENABLED_SETTING
} from "@/services/settings/ui-settings"

describe("voiceChatEnabled does not survive a page load", () => {
  beforeEach(() => backing.clear())

  it("ignores a stale true stored in the local area", async () => {
    backing.set("local::voiceChatEnabled", true)
    expect(await getSetting(VOICE_CHAT_ENABLED_SETTING)).toBe(false)
  })

  it("keeps a value set earlier in the same session", async () => {
    await setSetting(VOICE_CHAT_ENABLED_SETTING, true)
    expect(await getSetting(VOICE_CHAT_ENABLED_SETTING)).toBe(true)
    expect(backing.get("session::voiceChatEnabled")).toBe(true)
    expect(backing.has("local::voiceChatEnabled")).toBe(false)
  })

  it("leaves voiceChatAutoResume in the local area", () => {
    expect(VOICE_CHAT_AUTO_RESUME_SETTING.area).toBe("local")
  })
})
