import { describe, expect, it } from "vitest"

import {
  DEFAULT_HEADER_SHORTCUT_SELECTION,
  HEADER_SHORTCUT_SELECTION_SETTING
} from "@/services/settings/ui-settings"
import { normalizeSettingValue } from "@/services/settings/registry"

const LEGACY_SELECTION = [
  "chat",
  "prompts",
  "watchlists",
  "workflows",
  "acp-playground",
  "settings"
]

describe("header shortcut defaults", () => {
  it("includes Sources in the default selection", () => {
    expect(DEFAULT_HEADER_SHORTCUT_SELECTION).toContain("sources")
  })

  it("includes Companion Home first in the default selection", () => {
    expect(DEFAULT_HEADER_SHORTCUT_SELECTION[0]).toBe("companion-home")
  })

  it("includes the integrations control-plane shortcuts in the default selection", () => {
    expect(DEFAULT_HEADER_SHORTCUT_SELECTION).toContain("integrations")
    expect(DEFAULT_HEADER_SHORTCUT_SELECTION).toContain("scheduled-tasks")
    expect(DEFAULT_HEADER_SHORTCUT_SELECTION).toContain("admin-integrations")
  })

  it("adds Sources to legacy full-default persisted selections missing only Sources", () => {
    const legacyFullDefaultSelection = DEFAULT_HEADER_SHORTCUT_SELECTION.filter(
      (id) => id !== "sources"
    )

    const normalized = normalizeSettingValue(
      HEADER_SHORTCUT_SELECTION_SETTING,
      legacyFullDefaultSelection
    )

    expect(normalized).toContain("sources")
  })

  it("adds Sources to legacy full-default selections missing Companion Home and Sources", () => {
    const legacyFullDefaultSelection = DEFAULT_HEADER_SHORTCUT_SELECTION.filter(
      (id) => id !== "companion-home" && id !== "sources"
    )

    const normalized = normalizeSettingValue(
      HEADER_SHORTCUT_SELECTION_SETTING,
      legacyFullDefaultSelection
    )

    expect(normalized).toContain("companion-home")
    expect(normalized).toContain("sources")
  })

  it("does not add Sources to custom persisted selections", () => {
    const normalized = normalizeSettingValue(
      HEADER_SHORTCUT_SELECTION_SETTING,
      ["chat", "notes", "settings"]
    )

    expect(normalized).not.toContain("sources")
  })

  it("adds Companion Home to persisted filtered selections", () => {
    const normalized = normalizeSettingValue(
      HEADER_SHORTCUT_SELECTION_SETTING,
      ["chat", "settings"]
    )

    expect(normalized).toEqual(
      expect.arrayContaining(["companion-home", "chat", "settings"])
    )
  })

  it("adds the integrations control-plane shortcuts to persisted legacy selections", () => {
    const normalized = normalizeSettingValue(
      HEADER_SHORTCUT_SELECTION_SETTING,
      LEGACY_SELECTION
    )

    expect(normalized).toEqual(
      expect.arrayContaining([
        "chat",
        "prompts",
        "watchlists",
        "workflows",
        "acp-playground",
        "settings",
        "integrations",
        "scheduled-tasks",
        "admin-integrations"
      ])
    )
  })

  it("keeps hosted account and billing shortcuts when they appear in persisted selections", () => {
    const normalized = normalizeSettingValue(
      HEADER_SHORTCUT_SELECTION_SETTING,
      ["chat", "account", "billing", "settings"]
    )

    expect(normalized).toEqual(
      expect.arrayContaining([
        "chat",
        "account",
        "billing",
        "settings",
      ])
    )
  })

  it("migrates stored moderation playground shortcuts to review and rules", () => {
    const normalized = normalizeSettingValue(
      HEADER_SHORTCUT_SELECTION_SETTING,
      ["chat", "moderation-playground", "settings"]
    )

    expect(normalized).toEqual(
      expect.arrayContaining([
        "chat",
        "moderation-review",
        "moderation-rules",
        "settings"
      ])
    )
    expect(normalized).not.toContain("moderation-playground")
  })
})
