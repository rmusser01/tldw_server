import { describe, expect, it } from "vitest"

import { buildSidepanelFullAppChatPath } from "../sidepanel-full-app-route"
import type { AssistantSelection } from "@/types/assistant-selection"

describe("buildSidepanelFullAppChatPath", () => {
  it("opens normal chat when no role-play assistant is selected", () => {
    expect(buildSidepanelFullAppChatPath()).toBe("/chat")
  })

  it("carries the selected sidepanel character id into the full app", () => {
    expect(
      buildSidepanelFullAppChatPath({
        selectedCharacterId: " char-42 "
      })
    ).toBe("/chat?mode=character&characterId=char-42")
  })

  it("prefers the selected character assistant over a stale legacy character id", () => {
    const selectedAssistant: AssistantSelection = {
      kind: "character",
      id: "assistant-character",
      name: "Mira Vale"
    }

    expect(
      buildSidepanelFullAppChatPath({
        selectedAssistant,
        selectedCharacterId: "legacy-character"
      })
    ).toBe("/chat?mode=character&characterId=assistant-character")
  })

  it("preserves persona role-play intent without inventing a character id", () => {
    const selectedAssistant: AssistantSelection = {
      kind: "persona",
      id: "persona-1",
      name: "Research Persona"
    }

    expect(
      buildSidepanelFullAppChatPath({
        selectedAssistant,
        selectedCharacterId: "legacy-character"
      })
    ).toBe("/chat?mode=character")
  })
})
