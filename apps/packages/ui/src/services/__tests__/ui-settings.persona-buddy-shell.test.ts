import { beforeEach, describe, expect, it } from "vitest"

import { SETTINGS_INDEX } from "@/data/settings-index"
import {
  clearSetting,
  getSetting,
  normalizeSettingValue,
  setSetting
} from "@/services/settings/registry"
import {
  PERSONA_BUDDY_SHELL_ENABLED_SETTING
} from "@/services/settings/ui-settings"

describe("persona buddy shell setting", () => {
  beforeEach(async () => {
    localStorage.clear()
    await clearSetting(PERSONA_BUDDY_SHELL_ENABLED_SETTING)
  })

  it("defaults to enabled", () => {
    expect(PERSONA_BUDDY_SHELL_ENABLED_SETTING.defaultValue).toBe(true)
    expect(
      normalizeSettingValue(PERSONA_BUDDY_SHELL_ENABLED_SETTING, undefined)
    ).toBe(true)
  })

  it("coerces boolean-like values and mirrors persisted state to localStorage", async () => {
    expect(
      normalizeSettingValue(PERSONA_BUDDY_SHELL_ENABLED_SETTING, "false")
    ).toBe(false)
    expect(
      normalizeSettingValue(PERSONA_BUDDY_SHELL_ENABLED_SETTING, "true")
    ).toBe(true)

    await setSetting(
      PERSONA_BUDDY_SHELL_ENABLED_SETTING,
      "false" as unknown as boolean
    )

    expect(await getSetting(PERSONA_BUDDY_SHELL_ENABLED_SETTING)).toBe(false)
    expect(localStorage.getItem("tldw:personaBuddyShellEnabled")).toBe("false")
  })

  it("indexes the setting on the UI customization page", () => {
    const entry = SETTINGS_INDEX.find(
      (item) => item.id === "setting-persona-buddy-shell"
    )

    expect(entry).toMatchObject({
      route: "/settings/ui",
      section: "UI",
      storageKey: PERSONA_BUDDY_SHELL_ENABLED_SETTING.key,
      controlType: "switch"
    })
  })
})
