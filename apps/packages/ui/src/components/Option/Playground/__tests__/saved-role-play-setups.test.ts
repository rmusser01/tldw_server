import { describe, expect, it } from "vitest"

import {
  createStartupTemplateBundle,
  describeRolePlaySetupPreview,
  isRolePlayRelevantBundle
} from "../startup-template-bundles"

const pinnedResults = [
  {
    id: "source-1",
    snippet: "Pinned evidence",
    title: "Dataset"
  },
  {
    id: "source-2",
    snippet: "Second pinned source"
  }
]

const scene = {
  version: 3,
  isEnabled: true,
  aspects: [],
  notes: "The room smells like ozone.",
  chatPosition: "before",
  chatDepth: 0,
  chatRole: "system"
} as any

describe("saved role-play setup helpers", () => {
  it("treats bundles saved from Role-play setup as role-play relevant", () => {
    const bundle = createStartupTemplateBundle({
      name: "Mira detective scene",
      selectedModel: "openai:gpt-4.1",
      systemPrompt: "Observe everything.",
      presetKey: "creative",
      source: "role-play-setup",
      character: {
        id: "char-mira",
        name: "Mira"
      } as any,
      ragPinnedResults: pinnedResults,
      rolePlay: {
        source: "role-play-setup",
        identity: {
          kind: "character",
          id: "char-mira",
          name: "Mira"
        },
        behavior: {
          source: "template",
          templateId: "detective",
          templateTitle: "Detective",
          templateCategory: "roleplay",
          systemPrompt: "Observe everything.",
          modified: false
        },
        scene,
        generation: {
          presetKey: "creative",
          settings: {
            temperature: 1.2,
            topP: 0.95
          }
        },
        context: {
          ragPinnedCount: 2,
          ragPinnedResultIds: ["source-1", "source-2"]
        }
      }
    })

    expect(isRolePlayRelevantBundle(bundle)).toBe(true)
  })

  it("treats character and persona identities as role-play relevant", () => {
    const characterBundle = createStartupTemplateBundle({
      name: "Character setup",
      selectedModel: null,
      systemPrompt: "",
      character: {
        id: "char-mira",
        name: "Mira"
      } as any
    })
    const personaBundle = createStartupTemplateBundle({
      name: "Persona setup",
      selectedModel: null,
      systemPrompt: "",
      rolePlay: {
        source: "role-play-setup",
        identity: {
          kind: "persona",
          id: "persona-guide",
          name: "Patient Guide"
        },
        behavior: null,
        scene: null,
        generation: null,
        context: null
      }
    })

    expect(isRolePlayRelevantBundle(characterBundle)).toBe(true)
    expect(isRolePlayRelevantBundle(personaBundle)).toBe(true)
  })

  it("treats role-play behavior templates as role-play relevant", () => {
    const bundle = createStartupTemplateBundle({
      name: "Behavior setup",
      selectedModel: null,
      systemPrompt: "Stay in character as a careful detective.",
      rolePlay: {
        source: "role-play-setup",
        identity: null,
        behavior: {
          source: "template",
          templateId: "detective",
          templateTitle: "Detective",
          templateCategory: "roleplay",
          systemPrompt: "Stay in character as a careful detective.",
          modified: false
        },
        scene: null,
        generation: null,
        context: null
      }
    })

    expect(isRolePlayRelevantBundle(bundle)).toBe(true)
  })

  it("does not treat generation style alone as role-play relevant", () => {
    const bundle = createStartupTemplateBundle({
      name: "Creative writing launch",
      selectedModel: "openai:gpt-4.1",
      systemPrompt: "",
      presetKey: "creative"
    })

    expect(isRolePlayRelevantBundle(bundle)).toBe(false)
  })

  it("does not treat template name substrings as role-play relevance", () => {
    const bundle = createStartupTemplateBundle({
      name: "Role-play detective mode",
      selectedModel: null,
      systemPrompt: "",
      presetKey: "balanced"
    })

    expect(isRolePlayRelevantBundle(bundle)).toBe(false)
  })

  it("describes saved role-play setup identity, behavior, generation, and context", () => {
    const bundle = createStartupTemplateBundle({
      name: "Mira detective scene",
      selectedModel: "openai:gpt-4.1",
      systemPrompt: "Observe everything.",
      presetKey: "creative",
      source: "role-play-setup",
      character: {
        id: "char-mira",
        name: "Mira"
      } as any,
      ragPinnedResults: pinnedResults,
      rolePlay: {
        source: "role-play-setup",
        identity: {
          kind: "character",
          id: "char-mira",
          name: "Mira"
        },
        behavior: {
          source: "template",
          templateId: "detective",
          templateTitle: "Detective",
          templateCategory: "roleplay",
          systemPrompt: "Observe everything.",
          modified: false
        },
        scene,
        generation: {
          presetKey: "creative",
          settings: {
            temperature: 1.2,
            topP: 0.95
          }
        },
        context: {
          ragPinnedCount: 2,
          ragPinnedResultIds: ["source-1", "source-2"]
        }
      }
    })

    expect(describeRolePlaySetupPreview(bundle)).toEqual({
      identity: "Character: Mira",
      behavior: "Detective",
      scene: "Scene: active",
      generation: "Creative",
      context: "2 pinned sources"
    })
  })
})
