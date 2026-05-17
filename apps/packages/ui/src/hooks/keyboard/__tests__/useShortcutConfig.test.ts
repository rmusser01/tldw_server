import { describe, expect, it } from "vitest"
import {
  defaultShortcuts,
  mergeShortcutConfig,
  type ShortcutConfig
} from "../useShortcutConfig"

describe("shortcut config defaults", () => {
  it("binds Sources mode navigation to Alt+2", () => {
    expect(defaultShortcuts.modeSources).toEqual({
      key: "2",
      altKey: true,
      preventDefault: true,
      stopPropagation: true
    })
  })

  it("merges persisted overrides over complete defaults", () => {
    const custom = {
      key: "m",
      ctrlKey: true,
      preventDefault: false,
      stopPropagation: false
    }

    const merged = mergeShortcutConfig({ modeMedia: custom })

    expect(merged.modeSources).toEqual(defaultShortcuts.modeSources)
    expect(merged.modeNotes).toEqual(defaultShortcuts.modeNotes)
    expect(merged.modeMedia).toEqual(custom)
  })

  it("keeps modeSources available for legacy persisted configs", () => {
    const legacyShortcuts = {
      ...defaultShortcuts
    } as Partial<ShortcutConfig>
    delete legacyShortcuts.modeSources

    expect(mergeShortcutConfig(legacyShortcuts).modeSources).toEqual(
      defaultShortcuts.modeSources
    )
  })

  it("ignores nullish or malformed persisted shortcut configs", () => {
    expect(mergeShortcutConfig(null)).toEqual(defaultShortcuts)
    expect(mergeShortcutConfig("bad-storage-value")).toEqual(defaultShortcuts)
    expect(mergeShortcutConfig(["bad-storage-value"])).toEqual(defaultShortcuts)
  })
})
