import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { Prompt } from "@/db/dexie/types"
import type { Character } from "@/types/character"
import {
  usePromptTemplates,
  type UsePromptTemplatesDeps
} from "../hooks/usePromptTemplates"
import {
  createStartupTemplateBundle,
  describeRolePlaySetupPreview,
  describeStartupTemplatePrompt,
  parseStartupTemplateBundles,
  resolveStartupTemplatePrompt,
  serializeStartupTemplateBundles,
  type StartupTemplateRolePlayMetadata
} from "../startup-template-bundles"

const previousCharacter = {
  id: "mira",
  name: "Mira",
  greeting: "Mira's observatory.",
  system_prompt: "You are Mira.",
  extensions: { previousCard: true }
} as Character
const previousPrompt: Prompt = {
  id: "previous-prompt",
  title: "Previous behavior",
  content: "Use the previous behavior.",
  is_system: true,
  createdAt: 1,
  updatedAt: 1,
  sourceSystem: "workspace"
}
const snapshot = (
  overrides: Partial<StartupTemplateRolePlayMetadata> = {}
): StartupTemplateRolePlayMetadata => ({
  source: "role-play-setup",
  identity: null,
  behavior: null,
  scene: null,
  generation: { presetKey: "balanced", settings: {} },
  context: null,
  ...overrides
})
const createDeps = (
  overrides: Partial<UsePromptTemplatesDeps> = {}
): UsePromptTemplatesDeps => ({
  startupTemplatesRaw: "[]",
  setStartupTemplatesRaw: vi.fn(),
  promptLibrary: [previousPrompt],
  selectedModel: null,
  systemPrompt: previousPrompt.content,
  selectedSystemPrompt: previousPrompt.id,
  selectedQuickPrompt: null,
  selectedCharacter: previousCharacter,
  ragPinnedResults: [],
  currentChatModelSettings: {},
  setSelectedModel: vi.fn(),
  setSelectedSystemPrompt: vi.fn(),
  setSelectedQuickPrompt: vi.fn(),
  setSystemPrompt: vi.fn(),
  applyAssistantSelection: vi.fn(async () => true),
  setRagPinnedResults: vi.fn(),
  updateChatModelSettings: vi.fn(),
  compareModeActive: false,
  setCompareSelectedModels: vi.fn(),
  setModeAnnouncement: vi.fn(),
  t: (key, fallback) => fallback || key,
  ...overrides
})
const conflictingBundle = (rolePlay: StartupTemplateRolePlayMetadata) =>
  createStartupTemplateBundle({
    name: "Imported setup",
    source: "role-play-setup",
    selectedModel: null,
    character: previousCharacter,
    systemPrompt: previousPrompt.content,
    selectedSystemPromptId: previousPrompt.id,
    promptTitle: previousPrompt.title,
    promptSource: "prompt-library",
    rolePlay
  })
const saveSnapshot = (
  rolePlay: StartupTemplateRolePlayMetadata,
  deps: UsePromptTemplatesDeps
) => {
  const hook = renderHook(() => usePromptTemplates(deps))
  act(() =>
    hook.result.current.handleSaveRolePlaySetup({
      name: "Saved setup",
      rolePlay
    })
  )
  const raw = vi.mocked(deps.setStartupTemplatesRaw).mock.calls[0]![0]
  return { ...hook, saved: parseStartupTemplateBundles(raw)[0]! }
}

describe("saved roleplay snapshot authority", () => {
  it("keeps explicit identity and behavior clears through Save, storage and Apply", async () => {
    const deps = createDeps()
    const { result, saved } = saveSnapshot(snapshot(), deps)
    await act(async () => {
      await result.current.applyStartupTemplateBundle(saved)
    })
    expect({
      storedCharacter: saved.character,
      storedPrompt: saved.systemPrompt,
      storedPromptId: saved.selectedSystemPromptId,
      storedPromptSource: saved.promptSource,
      appliedIdentity: vi.mocked(deps.applyAssistantSelection).mock
        .calls[0]?.[0],
      appliedPrompt: vi.mocked(deps.setSystemPrompt).mock.calls[0]?.[0]
    }).toEqual({
      storedCharacter: null,
      storedPrompt: "",
      storedPromptId: null,
      storedPromptSource: "none",
      appliedIdentity: null,
      appliedPrompt: ""
    })
  })
  it("roundtrips a fully cleared valid snapshot without losing its explicit clears", () => {
    const cleared = snapshot({ generation: null })
    const { saved } = saveSnapshot(cleared, createDeps())
    const reparsed = parseStartupTemplateBundles(
      serializeStartupTemplateBundles([saved])
    )[0]!
    expect({ source: reparsed.source, rolePlay: reparsed.rolePlay }).toEqual({
      source: "role-play-setup",
      rolePlay: cleared
    })
  })
  it("saves a persona without retaining the currently selected character card", async () => {
    const deps = createDeps()
    const { result, saved } = saveSnapshot(
      snapshot({
        identity: { kind: "persona", id: "guide", name: "Patient Guide" }
      }),
      deps
    )
    await act(async () => {
      await result.current.applyStartupTemplateBundle(saved)
    })
    expect({
      character: saved.character,
      kind: vi.mocked(deps.applyAssistantSelection).mock.calls[0]?.[0]?.kind,
      id: vi.mocked(deps.applyAssistantSelection).mock.calls[0]?.[0]?.id
    }).toEqual({ character: null, kind: "persona", id: "guide" })
  })
  it("saves a changed character without copying the previous card's greeting or behavior", async () => {
    const deps = createDeps()
    const { result, saved } = saveSnapshot(
      snapshot({ identity: { kind: "character", id: "nova", name: "Nova" } }),
      deps
    )
    await act(async () => {
      await result.current.applyStartupTemplateBundle(saved)
    })
    const selection = vi.mocked(deps.applyAssistantSelection).mock.calls[0]?.[0]
    expect({
      id: selection?.id,
      name: selection?.name,
      greeting: selection?.greeting,
      systemPrompt: selection?.system_prompt,
      extensions: selection?.extensions
    }).toEqual({
      id: "nova",
      name: "Nova",
      greeting: undefined,
      systemPrompt: undefined,
      extensions: null
    })
  })
  it("preserves matching character card details and the snapshot's explicit name", async () => {
    const deps = createDeps()
    const { result, saved } = saveSnapshot(
      snapshot({
        identity: { kind: "character", id: "mira", name: "Mira the Guide" }
      }),
      deps
    )
    await act(async () => {
      await result.current.applyStartupTemplateBundle(saved)
    })
    expect(vi.mocked(deps.applyAssistantSelection).mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        id: "mira",
        name: "Mira the Guide",
        greeting: previousCharacter.greeting,
        system_prompt: previousCharacter.system_prompt,
        extensions: previousCharacter.extensions
      })
    )
  })
  it("applies explicit identity clear before a conflicting legacy character", async () => {
    const deps = createDeps()
    const { result } = renderHook(() => usePromptTemplates(deps))
    await act(async () => {
      await result.current.applyStartupTemplateBundle(
        conflictingBundle(snapshot())
      )
    })
    expect(
      vi.mocked(deps.applyAssistantSelection).mock.calls[0]?.[0]
    ).toBeNull()
  })
  it("applies a changed snapshot character before conflicting legacy card details", async () => {
    const deps = createDeps()
    const { result } = renderHook(() => usePromptTemplates(deps))
    await act(async () => {
      await result.current.applyStartupTemplateBundle(
        conflictingBundle(
          snapshot({
            identity: { kind: "character", id: "nova", name: "Nova" }
          })
        )
      )
    })
    const selection = vi.mocked(deps.applyAssistantSelection).mock.calls[0]?.[0]
    expect({
      id: selection?.id,
      name: selection?.name,
      greeting: selection?.greeting
    }).toEqual({
      id: "nova",
      name: "Nova",
      greeting: undefined
    })
  })
  it("applies exact saved behavior before a changed or conflicting prompt-library link", async () => {
    const deps = createDeps()
    const { result } = renderHook(() => usePromptTemplates(deps))
    await act(async () => {
      await result.current.applyStartupTemplateBundle(
        conflictingBundle(
          snapshot({
            behavior: {
              source: "modified-template",
              templateId: "detective",
              templateTitle: "Detective",
              templateCategory: "roleplay",
              systemPrompt: "Observe everything, quietly.",
              modified: true
            }
          })
        )
      )
    })
    expect({
      prompt: vi.mocked(deps.setSystemPrompt).mock.calls[0]?.[0],
      promptId: vi.mocked(deps.setSelectedSystemPrompt).mock.calls[0]?.[0],
      templateId: vi.mocked(deps.updateChatModelSettings).mock.calls[0]?.[0]
        .systemPromptTemplateId
    }).toEqual({
      prompt: "Observe everything, quietly.",
      promptId: undefined,
      templateId: "detective"
    })
  })
  it("shows the same explicit clears in the saved preview and prompt summary", () => {
    const bundle = conflictingBundle(snapshot())
    const preview = describeRolePlaySetupPreview(bundle)
    expect({
      identity: preview.identity,
      behavior: preview.behavior,
      prompt: describeStartupTemplatePrompt(bundle, [previousPrompt]),
      resolvedPrompt: resolveStartupTemplatePrompt(bundle, [previousPrompt])
        .prompt
    }).toEqual({
      identity: "None",
      behavior: "No behavior",
      prompt: "No prompt",
      resolvedPrompt: null
    })
  })
  it.each([
    [
      "missing identity",
      (record: Record<string, unknown>) => {
        delete record.identity
      }
    ],
    [
      "missing behavior",
      (record: Record<string, unknown>) => {
        delete record.behavior
      }
    ],
    [
      "malformed identity",
      (record: Record<string, unknown>) => {
        record.identity = { kind: "invalid", id: "wrong", name: "Wrong" }
      }
    ],
    [
      "malformed behavior",
      (record: Record<string, unknown>) => {
        record.behavior = 42
      }
    ],
    [
      "non-text behavior",
      (record: Record<string, unknown>) => {
        record.behavior = {
          source: "custom",
          templateTitle: "Invalid",
          systemPrompt: 42
        }
      }
    ],
    [
      "malformed generation",
      (record: Record<string, unknown>) => {
        record.generation = { presetKey: "balanced", settings: "invalid" }
      }
    ]
  ] as const)(
    "does not grant snapshot-clear authority to %s beside valid metadata",
    async (_name, corrupt) => {
      const raw = {
        ...conflictingBundle(snapshot()),
        rolePlay: { ...snapshot() } as Record<string, unknown>
      }
      corrupt(raw.rolePlay)
      const parsed = parseStartupTemplateBundles(JSON.stringify([raw]))[0]!
      const deps = createDeps()
      const { result } = renderHook(() => usePromptTemplates(deps))
      await act(async () => {
        await result.current.applyStartupTemplateBundle(parsed)
      })
      expect({
        rolePlay: parsed.rolePlay,
        identity: vi.mocked(deps.applyAssistantSelection).mock.calls[0]?.[0]
          ?.id,
        prompt: vi.mocked(deps.setSystemPrompt).mock.calls[0]?.[0]
      }).toEqual({
        rolePlay: null,
        identity: "mira",
        prompt: previousPrompt.content
      })
    }
  )
  it("keeps legacy character and library-prompt restoration when no snapshot was saved", async () => {
    const deps = createDeps()
    const legacy = createStartupTemplateBundle({
      name: "Legacy",
      selectedModel: null,
      character: previousCharacter,
      systemPrompt: "Older text",
      selectedSystemPromptId: previousPrompt.id,
      promptSource: "prompt-library"
    })
    const { result } = renderHook(() => usePromptTemplates(deps))
    await act(async () => {
      await result.current.applyStartupTemplateBundle(legacy)
    })
    expect({
      identity: vi.mocked(deps.applyAssistantSelection).mock.calls[0]?.[0]?.id,
      prompt: vi.mocked(deps.setSystemPrompt).mock.calls[0]?.[0],
      promptId: vi.mocked(deps.setSelectedSystemPrompt).mock.calls[0]?.[0]
    }).toEqual({
      identity: "mira",
      prompt: previousPrompt.content,
      promptId: previousPrompt.id
    })
  })
  it("accepts a complete saved snapshot with an explicitly empty context", () => {
    const emptyContext = { ragPinnedCount: 0, ragPinnedResultIds: [] }
    const bundle = conflictingBundle(snapshot({ context: emptyContext }))
    const reparsed = parseStartupTemplateBundles(
      serializeStartupTemplateBundles([bundle])
    )[0]!
    expect(reparsed.rolePlay).toEqual(snapshot({ context: emptyContext }))
  })
  it.each([
    { ragPinnedCount: 1 },
    { ragPinnedCount: "one", ragPinnedResultIds: ["source-1"] },
    { ragPinnedCount: 1, ragPinnedResultIds: [42] }
  ])(
    "refuses malformed context alongside valid snapshot components: %j",
    (context) => {
      const bundle = {
        ...conflictingBundle(snapshot()),
        rolePlay: { ...snapshot(), context }
      }
      expect(
        parseStartupTemplateBundles(JSON.stringify([bundle]))[0]?.rolePlay
      ).toBeNull()
    }
  )
})
