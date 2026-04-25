import { beforeEach, describe, expect, it } from "vitest"
import {
  CURRENT_USER_PREFERENCE_MIGRATION,
  migrateThemeUserPreferences,
  THEME_MIGRATION_VERSION_KEY,
  THEME_PRESET_KEY,
} from "../user-preference-migration"

describe("migrateThemeUserPreferences", () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it("sets Primer as the default preset for a fresh install", () => {
    migrateThemeUserPreferences()

    expect(window.localStorage.getItem(THEME_PRESET_KEY)).toBe("primer")
    expect(window.localStorage.getItem(THEME_MIGRATION_VERSION_KEY)).toBe(
      String(CURRENT_USER_PREFERENCE_MIGRATION)
    )
  })

  it("preserves an existing user's stored preset", () => {
    window.localStorage.setItem(THEME_PRESET_KEY, "solarized")

    migrateThemeUserPreferences()

    expect(window.localStorage.getItem(THEME_PRESET_KEY)).toBe("solarized")
    expect(window.localStorage.getItem(THEME_MIGRATION_VERSION_KEY)).toBe(
      String(CURRENT_USER_PREFERENCE_MIGRATION)
    )
  })

  it("does not switch existing users who already have other persisted UI state", () => {
    window.localStorage.setItem("theme", "dark")

    migrateThemeUserPreferences()

    expect(window.localStorage.getItem(THEME_PRESET_KEY)).toBeNull()
    expect(window.localStorage.getItem(THEME_MIGRATION_VERSION_KEY)).toBe(
      String(CURRENT_USER_PREFERENCE_MIGRATION)
    )
  })

  it("does not re-migrate once the version marker is set", () => {
    window.localStorage.setItem(
      THEME_MIGRATION_VERSION_KEY,
      String(CURRENT_USER_PREFERENCE_MIGRATION)
    )

    migrateThemeUserPreferences()

    // Fresh install indicators should not have been written a second time
    expect(window.localStorage.getItem(THEME_PRESET_KEY)).toBeNull()
  })

  it("preserves preset even if only the migration marker was unset (existing explicit default)", () => {
    window.localStorage.setItem(THEME_PRESET_KEY, "default")

    migrateThemeUserPreferences()

    expect(window.localStorage.getItem(THEME_PRESET_KEY)).toBe("default")
    expect(window.localStorage.getItem(THEME_MIGRATION_VERSION_KEY)).toBe(
      String(CURRENT_USER_PREFERENCE_MIGRATION)
    )
  })

  it("treats an empty stored preset as corrupt and repairs it to primer", () => {
    window.localStorage.setItem(THEME_PRESET_KEY, "   ")
    window.localStorage.setItem("theme", "dark")

    migrateThemeUserPreferences()

    expect(window.localStorage.getItem(THEME_PRESET_KEY)).toBe("primer")
    expect(window.localStorage.getItem(THEME_MIGRATION_VERSION_KEY)).toBe(
      String(CURRENT_USER_PREFERENCE_MIGRATION)
    )
  })

  it("is idempotent — calling twice has the same effect as once", () => {
    migrateThemeUserPreferences()
    const afterFirst = {
      preset: window.localStorage.getItem(THEME_PRESET_KEY),
      version: window.localStorage.getItem(THEME_MIGRATION_VERSION_KEY),
    }

    migrateThemeUserPreferences()
    const afterSecond = {
      preset: window.localStorage.getItem(THEME_PRESET_KEY),
      version: window.localStorage.getItem(THEME_MIGRATION_VERSION_KEY),
    }

    expect(afterSecond).toEqual(afterFirst)
  })

  it("treats a corrupted version marker as unmigrated and migrates again", () => {
    window.localStorage.setItem(THEME_MIGRATION_VERSION_KEY, "not-a-number")

    migrateThemeUserPreferences()

    expect(window.localStorage.getItem(THEME_PRESET_KEY)).toBe("primer")
    expect(window.localStorage.getItem(THEME_MIGRATION_VERSION_KEY)).toBe(
      String(CURRENT_USER_PREFERENCE_MIGRATION)
    )
  })

  it("treats partially numeric version markers as corrupt and migrates again", () => {
    window.localStorage.setItem(THEME_MIGRATION_VERSION_KEY, "1foo")

    migrateThemeUserPreferences()

    expect(window.localStorage.getItem(THEME_PRESET_KEY)).toBe("primer")
    expect(window.localStorage.getItem(THEME_MIGRATION_VERSION_KEY)).toBe(
      String(CURRENT_USER_PREFERENCE_MIGRATION)
    )
  })

  it("does not throw if localStorage.setItem rejects (quota / Safari private)", () => {
    const originalSetItem = window.localStorage.setItem
    try {
      window.localStorage.setItem = () => {
        throw new Error("QuotaExceededError")
      }

      expect(() => migrateThemeUserPreferences()).not.toThrow()
    } finally {
      window.localStorage.setItem = originalSetItem
    }
  })
})
