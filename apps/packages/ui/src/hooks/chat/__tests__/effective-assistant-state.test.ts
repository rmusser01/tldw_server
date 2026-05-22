import { describe, expect, it } from "vitest"

import { resolveEffectiveAssistantState } from "@/hooks/chat/effective-assistant-state"

describe("resolveEffectiveAssistantState", () => {
  it("resolves a tracked character from assistant_kind and character_id", () => {
    expect(
      resolveEffectiveAssistantState({
        tracked: {
          assistantKind: "character",
          characterId: "char-17",
          displayName: "Rin",
          avatarUrl: "https://cdn.example.test/rin.png",
          systemPromptSnapshot: "Stay in character."
        },
        settings: null
      })
    ).toEqual({
      mode: "tracked_character",
      kind: "character",
      id: "char-17",
      displayName: "Rin",
      avatarUrl: "https://cdn.example.test/rin.png",
      systemPromptSnapshot: "Stay in character.",
      source: "tracked"
    })
  })

  it("resolves a tracked persona from assistant_kind and assistant_id", () => {
    expect(
      resolveEffectiveAssistantState({
        tracked: {
          assistantKind: "persona",
          assistantId: "persona-9",
          displayName: "Analyst",
          avatarUrl: "https://cdn.example.test/analyst.png",
          systemPromptSnapshot: "Be concise."
        },
        settings: null
      })
    ).toEqual({
      mode: "tracked_persona",
      kind: "persona",
      id: "persona-9",
      displayName: "Analyst",
      avatarUrl: "https://cdn.example.test/analyst.png",
      systemPromptSnapshot: "Be concise.",
      source: "tracked"
    })
  })

  it("resolves an overlay from settings.assistantOverlay", () => {
    expect(
      resolveEffectiveAssistantState({
        tracked: null,
        settings: {
          assistantOverlay: {
            kind: "persona",
            id: "overlay-22",
            name: "Overlay Persona",
            avatar_url: "https://cdn.example.test/overlay.png",
            system_prompt_snapshot: "Use the overlay snapshot.",
            updatedAt: "2026-05-22T12:00:00.000Z"
          }
        }
      })
    ).toEqual({
      mode: "overlay",
      kind: "persona",
      id: "overlay-22",
      displayName: "Overlay Persona",
      avatarUrl: "https://cdn.example.test/overlay.png",
      systemPromptSnapshot: "Use the overlay snapshot.",
      source: "overlay"
    })
  })

  it("resolves plain mode when neither tracked metadata nor overlay exists", () => {
    expect(
      resolveEffectiveAssistantState({
        tracked: null,
        settings: null
      })
    ).toEqual({
      mode: "plain",
      kind: null,
      id: null,
      displayName: null,
      avatarUrl: null,
      systemPromptSnapshot: null,
      source: "none"
    })
  })

  it("lets tracked modes win over overlay when mixed data appears", () => {
    expect(
      resolveEffectiveAssistantState({
        tracked: {
          assistantKind: "character",
          characterId: "char-99",
          displayName: "Tracked Character",
          avatarUrl: "https://cdn.example.test/tracked.png",
          systemPromptSnapshot: "Tracked snapshot."
        },
        settings: {
          assistantOverlay: {
            kind: "persona",
            id: "overlay-99",
            name: "Overlay Persona",
            avatar_url: "https://cdn.example.test/overlay.png",
            system_prompt_snapshot: "Overlay snapshot.",
            updatedAt: "2026-05-22T12:00:00.000Z"
          }
        }
      })
    ).toEqual({
      mode: "tracked_character",
      kind: "character",
      id: "char-99",
      displayName: "Tracked Character",
      avatarUrl: "https://cdn.example.test/tracked.png",
      systemPromptSnapshot: "Tracked snapshot.",
      source: "tracked"
    })
  })

  it("falls back to draft metadata when tracked presentation fields are sparse", () => {
    expect(
      resolveEffectiveAssistantState({
        tracked: {
          assistantKind: "character",
          characterId: "char-5",
          displayName: "   ",
          avatarUrl: null,
          systemPromptSnapshot: null
        },
        settings: null,
        draftSelection: {
          kind: "character",
          id: "char-5",
          name: "Draft Character",
          avatar_url: "https://cdn.example.test/draft-character.png",
          system_prompt: "Draft character prompt"
        }
      })
    ).toEqual({
      mode: "tracked_character",
      kind: "character",
      id: "char-5",
      displayName: "Draft Character",
      avatarUrl: "https://cdn.example.test/draft-character.png",
      systemPromptSnapshot: "Draft character prompt",
      source: "tracked"
    })
  })

  it("falls back to draft metadata when overlay presentation fields are sparse", () => {
    expect(
      resolveEffectiveAssistantState({
        tracked: null,
        settings: {
          assistantOverlay: {
            kind: "persona",
            id: "overlay-7",
            name: "   ",
            avatar_url: null,
            system_prompt_snapshot: null,
            updatedAt: "2026-05-22T12:00:00.000Z"
          }
        },
        draftSelection: {
          kind: "persona",
          id: "overlay-7",
          name: "Draft Overlay",
          avatar_url: "https://cdn.example.test/draft-overlay.png",
          system_prompt: "Draft overlay prompt"
        }
      })
    ).toEqual({
      mode: "overlay",
      kind: "persona",
      id: "overlay-7",
      displayName: "Draft Overlay",
      avatarUrl: "https://cdn.example.test/draft-overlay.png",
      systemPromptSnapshot: "Draft overlay prompt",
      source: "overlay"
    })
  })
})
