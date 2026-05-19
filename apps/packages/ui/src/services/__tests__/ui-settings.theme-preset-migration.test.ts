import { beforeEach, describe, expect, it } from "vitest"

import { clearSetting, getSetting } from "@/services/settings/registry"
import {
  THEME_PRESET_SETTING
} from "@/services/settings/ui-settings"
import {
  CURRENT_USER_PREFERENCE_MIGRATION,
  THEME_MIGRATION_VERSION_KEY,
  THEME_PRESET_KEY,
} from "@/themes/user-preference-migration"

describe("theme preset setting hydration", () => {
  beforeEach(async () => {
    localStorage.clear()
    await clearSetting(THEME_PRESET_SETTING)
  })

  it("migrates fresh installs to primer before theme preset hydration", async () => {
    expect(await getSetting(THEME_PRESET_SETTING)).toBe("primer")
    expect(localStorage.getItem(THEME_PRESET_KEY)).toBe("primer")
    expect(localStorage.getItem(THEME_MIGRATION_VERSION_KEY)).toBe(
      String(CURRENT_USER_PREFERENCE_MIGRATION)
    )
  })

  it("keeps the legacy default experience for existing users without a preset", async () => {
    localStorage.setItem("theme", "dark")

    expect(await getSetting(THEME_PRESET_SETTING)).toBe("default")
    expect(localStorage.getItem(THEME_PRESET_KEY)).toBeNull()
    expect(localStorage.getItem(THEME_MIGRATION_VERSION_KEY)).toBe(
      String(CURRENT_USER_PREFERENCE_MIGRATION)
    )
  })
})
