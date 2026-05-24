import { describe, expect, it } from "vitest"
import { getHeaderShortcutGroups } from "../header-shortcut-items"

describe("header shortcut descriptions", () => {
  const groups = getHeaderShortcutGroups()
  const allItems = groups.flatMap((g) => g.items)

  const JARGON_IDS = [
    "stt-playground",
    "tts-playground",
    "knowledge-qa",
    "chunking-playground",
    "moderation-playground",
    "acp-playground",
    "chatbooks-playground",
    "world-books",
    "deep-research",
    "research-workspace",
    "prompt-studio",
    "model-playground",
    "mcp-hub",
    "chat-dictionaries",
    "evaluations",
    "repo2txt"
  ]

  for (const id of JARGON_IDS) {
    it(`item "${id}" has a descriptionDefault`, () => {
      const item = allItems.find((i) => i.id === id)
      expect(item).toBeDefined()
      expect(item!.descriptionDefault).toBeTruthy()
      expect(item!.descriptionDefault!.length).toBeGreaterThan(5)
    })
  }
})
