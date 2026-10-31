import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { usePromptTemplates, type UsePromptTemplatesDeps } from "../hooks"
import { createStartupTemplateBundle } from "../startup-template-bundles"

const createDeps = (
  overrides: Partial<UsePromptTemplatesDeps> = {}
): UsePromptTemplatesDeps => ({
  startupTemplatesRaw: "[]",
  setStartupTemplatesRaw: vi.fn(),
  promptLibrary: [],
  selectedModel: null,
  systemPrompt: "",
  selectedSystemPrompt: null,
  selectedQuickPrompt: null,
  selectedCharacter: null,
  ragPinnedResults: [],
  currentChatModelSettings: {},
  setSelectedModel: vi.fn(),
  setSelectedSystemPrompt: vi.fn(),
  setSelectedQuickPrompt: vi.fn(),
  setSystemPrompt: vi.fn(),
  applyAssistantSelection: vi.fn(async () => undefined),
  setRagPinnedResults: vi.fn(),
  updateChatModelSettings: vi.fn(),
  compareModeActive: false,
  setCompareSelectedModels: vi.fn(),
  setModeAnnouncement: vi.fn(),
  t: (_key: string, fallback?: string) => fallback || _key,
  ...overrides
})

describe("usePromptTemplates role-play setup apply", () => {
  it("preserves behavior template identity when applying a saved setup", async () => {
    const deps = createDeps()
    const setup = createStartupTemplateBundle({
      name: "Detective setup",
      selectedModel: "openai:gpt-4.1",
      systemPrompt: "Observe everything.",
      presetKey: "creative",
      source: "role-play-setup",
      character: {
        id: "char-mira",
        name: "Mira"
      } as any,
      rolePlay: {
        source: "role-play-setup",
        identity: {
          kind: "character",
          id: "char-mira",
          name: "Mira"
        },
        behavior: {
          source: "template",
          templateId: "detective-template",
          templateTitle: "Detective",
          templateCategory: "roleplay",
          systemPrompt: "Observe everything.",
          modified: false
        },
        scene: null,
        generation: {
          presetKey: "creative",
          settings: {}
        },
        context: null
      }
    })

    const { result } = renderHook(() => usePromptTemplates(deps))

    await act(async () => {
      await result.current.handleApplySavedRolePlaySetup(setup)
    })

    expect(deps.updateChatModelSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        systemPromptTemplateId: "detective-template"
      })
    )
    expect(deps.applyAssistantSelection).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "character",
        id: "char-mira",
        name: "Mira"
      })
    )
    expect(deps.applyAssistantSelection).toHaveBeenCalledTimes(1)
  })

  it("restores persona identity through selected assistant state", async () => {
    const applyAssistantSelection = vi.fn(async () => undefined)
    const deps = createDeps({
      applyAssistantSelection
    })
    const setup = createStartupTemplateBundle({
      name: "Persona setup",
      selectedModel: null,
      systemPrompt: "",
      source: "role-play-setup",
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

    const { result } = renderHook(() => usePromptTemplates(deps))

    await act(async () => {
      await result.current.handleApplySavedRolePlaySetup(setup)
    })

    expect(applyAssistantSelection).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "persona",
        id: "persona-guide",
        name: "Patient Guide"
      })
    )
    expect(applyAssistantSelection).toHaveBeenCalledTimes(1)
  })
})
