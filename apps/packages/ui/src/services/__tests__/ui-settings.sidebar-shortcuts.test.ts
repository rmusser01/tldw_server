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
  const REQUESTED_DEFAULT_SELECTION = [
    "quick-ingest",
    "chat",
    "prompts",
    "characters",
    "chat-dictionaries",
    "world-books",
    "notes",
    "knowledge-qa",
    "media",
    "document-workspace",
    "research-workspace",
    "kanban-playground",
    "watchlists"
  ]

  it("uses the requested default sidepanel shortcut order", () => {
    expect(DEFAULT_SIDEBAR_SHORTCUT_SELECTION).toEqual(
      REQUESTED_DEFAULT_SELECTION
    )
    expect(DEFAULT_SIDEBAR_SHORTCUT_SELECTION).toContain("chat-dictionaries")
    expect(DEFAULT_SIDEBAR_SHORTCUT_SELECTION).not.toContain("moderation-playground")
    expect(DEFAULT_SIDEBAR_SHORTCUT_SELECTION).toHaveLength(13)
  })

  it("migrates legacy default selection to the requested sidepanel order", () => {
    const normalized = normalizeSettingValue(
      SIDEBAR_SHORTCUT_SELECTION_SETTING,
      LEGACY_DEFAULT_SELECTION
    )

    expect(normalized).toEqual(REQUESTED_DEFAULT_SELECTION)
    expect(normalized).not.toContain("moderation-playground")
    expect(normalized).not.toContain("prompt-studio")
  })

  it("migrates the previous full default selection to the requested sidepanel order", () => {
    const normalized = normalizeSettingValue(
      SIDEBAR_SHORTCUT_SELECTION_SETTING,
      [
        "quick-ingest",
        "chat",
        "chat-workspace",
        "prompts",
        "characters",
        "deep-research",
        "world-books",
        "knowledge-qa",
        "media",
        "watchlists",
        "document-workspace",
        "flashcards",
        "moderation-review",
        "tts-playground",
        "stt-playground"
      ]
    )

    expect(normalized).toEqual(REQUESTED_DEFAULT_SELECTION)
  })

  it("maps stored moderation playground shortcuts to content rules", () => {
    const normalized = normalizeSettingValue(SIDEBAR_SHORTCUT_SELECTION_SETTING, [
      "quick-ingest",
      "moderation-playground"
    ])

    expect(normalized).toEqual(["quick-ingest", "moderation-rules"])
  })

  it("keeps custom user selections unchanged", () => {
    const customSelection = ["prompts", "quick-ingest", "chat"]

    const normalized = normalizeSettingValue(
      SIDEBAR_SHORTCUT_SELECTION_SETTING,
      customSelection
    )

    expect(normalized).toEqual(customSelection)
  })
})
