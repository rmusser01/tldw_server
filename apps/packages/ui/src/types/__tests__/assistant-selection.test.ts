import { describe, expect, it } from "vitest"

import {
  assistantSelectionToCharacter,
  characterToAssistantSelection,
  isPersonaAssistantSelection,
  normalizeAssistantSelection,
  personaToAssistantSelection
} from "../assistant-selection"

describe("normalizeAssistantSelection", () => {
  it("normalizes persona selections with numeric ids", () => {
    expect(
      normalizeAssistantSelection({
        kind: "persona",
        id: 42,
        title: "Research Persona",
        avatar_url: "https://example.com/avatar.png"
      })
    ).toEqual(
      expect.objectContaining({
        kind: "persona",
        id: "42",
        name: "Research Persona",
        avatar_url: "https://example.com/avatar.png"
      })
    )
  })

  it.each([
    ["blank string id", { kind: "persona", id: "   ", name: "Blank" }],
    ["missing id", { kind: "persona", name: "Missing" }],
    ["invalid kind", { kind: "workspace", id: "persona-1", name: "Invalid" }],
    ["array payload", [{ kind: "persona", id: "persona-1" }]],
    ["null payload", null]
  ])("rejects %s", (_label, payload) => {
    expect(normalizeAssistantSelection(payload)).toBeNull()
    expect(isPersonaAssistantSelection(payload)).toBe(false)
  })

  it("uses stable persona fallback values for optional invalid text fields", () => {
    expect(
      personaToAssistantSelection({
        id: "persona-1",
        name: "",
        slug: 7,
        greeting: false,
        system_prompt: { prompt: "invalid" },
        extensions: ["invalid"]
      })
    ).toEqual(
      expect.objectContaining({
        kind: "persona",
        id: "persona-1",
        name: "Persona",
        slug: null,
        greeting: null,
        system_prompt: null,
        extensions: null
      })
    )
  })

  it("normalizes legacy character-like values without treating them as personas", () => {
    const selection = characterToAssistantSelection({
      id: 7,
      title: "Legacy Character",
      character_id: 7
    })

    expect(selection).toEqual(
      expect.objectContaining({
        kind: "character",
        id: "7",
        name: "Legacy Character"
      })
    )
    expect(isPersonaAssistantSelection(selection)).toBe(false)
    expect(assistantSelectionToCharacter(selection)).toEqual(
      expect.objectContaining({
        id: "7",
        title: "Legacy Character",
        character_id: 7
      })
    )
  })
})
