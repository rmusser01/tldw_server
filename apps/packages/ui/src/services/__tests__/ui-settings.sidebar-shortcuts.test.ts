import { describe, expect, it } from "vitest"

import {
  DEFAULT_SIDEBAR_SHORTCUT_SELECTION,
  SIDEBAR_SHORTCUT_SELECTION_SETTING
} from "@/services/settings/ui-settings"
import { normalizeSettingValue } from "@/services/settings/registry"

const LEGACY_DEFAULT_SELECTION = [
  "quick-ingest",
  "chat",
  "prompts",
  "prompt-studio",
  "characters",
  "chat-dictionaries",
  "world-books",
  "knowledge-qa",
  "media",
  "document-workspace"
]

describe("sidebar shortcut defaults", () => {
  it("includes deep research and moderation review in the default selection", () => {
    expect(DEFAULT_SIDEBAR_SHORTCUT_SELECTION).toContain("deep-research")
    expect(DEFAULT_SIDEBAR_SHORTCUT_SELECTION).toContain("moderation-review")
    expect(DEFAULT_SIDEBAR_SHORTCUT_SELECTION).not.toContain("moderation-playground")
    expect(DEFAULT_SIDEBAR_SHORTCUT_SELECTION).not.toContain("chat-dictionaries")
    expect(DEFAULT_SIDEBAR_SHORTCUT_SELECTION).toHaveLength(15)
  })

  it("migrates legacy default selection to include deep research and moderation review", () => {
    const normalized = normalizeSettingValue(
      SIDEBAR_SHORTCUT_SELECTION_SETTING,
      LEGACY_DEFAULT_SELECTION
    )

    expect(normalized).toContain("deep-research")
    expect(normalized).toContain("moderation-review")
    expect(normalized).not.toContain("moderation-playground")
    expect(normalized).not.toContain("prompt-studio")
    expect(normalized).not.toContain("chat-dictionaries")
    expect(normalized).toHaveLength(15)
  })

  it("maps stored moderation playground shortcuts to content rules", () => {
    const normalized = normalizeSettingValue(SIDEBAR_SHORTCUT_SELECTION_SETTING, [
      "quick-ingest",
      "moderation-playground"
    ])

    expect(normalized).toEqual(["quick-ingest", "moderation-rules"])
  })

  it("keeps custom user selections unchanged", () => {
    const customSelection = ["quick-ingest", "chat", "prompts"]

    const normalized = normalizeSettingValue(
      SIDEBAR_SHORTCUT_SELECTION_SETTING,
      customSelection
    )

    expect(normalized).toEqual(customSelection)
  })
})
