import { beforeEach, describe, expect, it, vi } from "vitest"
import type {
  StructuredPromptDefinition,
  StructuredPromptPreviewResponse
} from "@/services/prompt-studio"

const state = vi.hoisted(() => ({
  prompts: new Map<string, any>()
}))

const mocks = vi.hoisted(() => ({
  createPrompt: vi.fn(),
  updatePrompt: vi.fn(),
  getPrompt: vi.fn(),
  promptAdd: vi.fn(),
  promptGet: vi.fn(),
  promptUpdate: vi.fn(),
  promptWhere: vi.fn()
}))

vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: async () => ({
    defaultProjectId: null,
    autoSyncWorkspacePrompts: true
  }),
  setPromptStudioDefaults: async () => undefined
}))

vi.mock("@/services/prompt-studio", () => ({
  listProjects: vi.fn(async () => ({ data: { data: [] } })),
  createProject: vi.fn(async () => ({ data: { data: { id: 42 } } })),
  createPrompt: (...args: unknown[]) =>
    (mocks.createPrompt as (...args: unknown[]) => unknown)(...args),
  updatePrompt: (...args: unknown[]) =>
    (mocks.updatePrompt as (...args: unknown[]) => unknown)(...args),
  getPrompt: (...args: unknown[]) =>
    (mocks.getPrompt as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/db/dexie/schema", () => ({
  db: {
    prompts: {
      add: (...args: unknown[]) =>
        (mocks.promptAdd as (...args: unknown[]) => unknown)(...args),
      get: (...args: unknown[]) =>
        (mocks.promptGet as (...args: unknown[]) => unknown)(...args),
      update: (...args: unknown[]) =>
        (mocks.promptUpdate as (...args: unknown[]) => unknown)(...args),
      where: (...args: unknown[]) =>
        (mocks.promptWhere as (...args: unknown[]) => unknown)(...args)
    }
  }
}))

vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: class {}
}))

vi.mock("@/db/dexie/helpers", () => ({
  generateID: () => "generated-local-id"
}))

const importPromptSync = async () => import("@/services/prompt-sync")

const makeDefinition = (content: string) => ({
  schema_version: 1,
  format: "structured",
  variables: [
    {
      name: "topic",
      required: true,
      input_type: "text"
    }
  ],
  blocks: [
    {
      id: "task",
      name: "Task",
      role: "user",
      content,
      enabled: true,
      order: 10,
      is_template: true
    }
  ]
})

const makeRecipeDefinition = (content = "Explain {{topic}}.") => ({
  schema_version: 2 as const,
  format: "structured" as const,
  definition_kind: "single_text_recipe" as const,
  variables: [
    {
      name: "topic",
      label: "Topic",
      required: true,
      default_value: null,
      input_type: "text"
    }
  ],
  blocks: [
    {
      id: "objective",
      name: "Objective",
      section_key: "objective",
      role: "user" as const,
      kind: "objective",
      content,
      enabled: true,
      order: 10,
      is_template: true
    }
  ],
  assembly_config: {
    assembly_mode: "single_text" as const,
    target_role: "user" as const,
    render_format: "markdown" as const,
    block_separator: "\n\n"
  }
})

describe("prompt-sync structured prompt support", () => {
  beforeEach(() => {
    state.prompts.clear()

    mocks.createPrompt.mockReset()
    mocks.updatePrompt.mockReset()
    mocks.getPrompt.mockReset()
    mocks.promptAdd.mockReset()
    mocks.promptGet.mockReset()
    mocks.promptUpdate.mockReset()
    mocks.promptWhere.mockReset()

    mocks.promptAdd.mockImplementation(async (prompt: any) => {
      state.prompts.set(prompt.id, prompt)
    })
    mocks.promptGet.mockImplementation(async (id: string) =>
      state.prompts.get(id)
    )
    mocks.promptUpdate.mockImplementation(
      async (id: string, updates: Record<string, unknown>) => {
        const current = state.prompts.get(id)
        if (!current) return
        state.prompts.set(id, { ...current, ...updates })
      }
    )
    mocks.promptWhere.mockImplementation((field: string) => ({
      equals: (value: unknown) => ({
        first: async () => {
          for (const prompt of state.prompts.values()) {
            if (prompt?.[field] === value) return prompt
          }
          return undefined
        }
      })
    }))
  })

  it("pushes structured prompt fields to Prompt Studio create payloads", async () => {
    state.prompts.set("local-structured", {
      id: "local-structured",
      title: "Structured Prompt",
      name: "Structured Prompt",
      content: "Explain {{topic}}",
      is_system: false,
      system_prompt: "Legacy system snapshot",
      user_prompt: "Explain {{topic}}",
      promptFormat: "structured",
      promptSchemaVersion: 1,
      structuredPromptDefinition: makeDefinition("Explain {{topic}}"),
      fewShotExamples: [
        {
          inputs: { topic: "Indexes" },
          outputs: { answer: "Use the covering index." }
        }
      ],
      modulesConfig: [
        {
          type: "style_rules",
          enabled: true,
          config: { tone: "concise" }
        }
      ],
      createdAt: 1,
      updatedAt: 1,
      syncStatus: "local"
    })

    mocks.createPrompt.mockResolvedValue({
      data: {
        data: {
          id: 101,
          project_id: 42,
          name: "Structured Prompt",
          system_prompt: "Legacy system snapshot",
          user_prompt: "Explain {{topic}}",
          prompt_format: "structured",
          prompt_schema_version: 1,
          prompt_definition: makeDefinition("Explain {{topic}}"),
          version_number: 1,
          updated_at: "2026-03-10T00:00:00Z"
        }
      }
    })

    const { pushToStudio } = await importPromptSync()
    await pushToStudio("local-structured", 42)

    expect(mocks.createPrompt).toHaveBeenCalledWith(
      expect.objectContaining({
        project_id: 42,
        prompt_format: "structured",
        prompt_schema_version: 1,
        prompt_definition: makeDefinition("Explain {{topic}}"),
        few_shot_examples: [
          {
            inputs: { topic: "Indexes" },
            outputs: { answer: "Use the covering index." }
          }
        ],
        modules_config: [
          {
            type: "style_rules",
            enabled: true,
            config: { tone: "concise" }
          }
        ]
      })
    )
  })

  it("exposes the recipe union and single-text preview response contract", () => {
    const definition: StructuredPromptDefinition = makeRecipeDefinition()
    const response = {} as StructuredPromptPreviewResponse
    const renderedText: string | undefined = response.rendered_text

    expect(definition.schema_version).toBe(2)
    expect(renderedText).toBeUndefined()
  })

  it("pulls structured prompt fields into the local prompt record", async () => {
    mocks.getPrompt.mockResolvedValue({
      data: {
        data: {
          id: 501,
          project_id: 9,
          name: "Server Structured Prompt",
          system_prompt: "Legacy system snapshot",
          user_prompt: "Explain {{topic}}",
          prompt_format: "structured",
          prompt_schema_version: 1,
          prompt_definition: makeDefinition("Explain {{topic}}"),
          few_shot_examples: [
            {
              inputs: { topic: "SQLite" },
              outputs: { answer: "Use FTS5." }
            }
          ],
          modules_config: [
            {
              type: "style_rules",
              enabled: true,
              config: { tone: "formal" }
            }
          ],
          version_number: 2,
          updated_at: "2026-03-10T00:00:00Z"
        }
      }
    })

    const { pullFromStudio } = await importPromptSync()
    await pullFromStudio(501)

    expect(state.prompts.get("generated-local-id")).toEqual(
      expect.objectContaining({
        promptFormat: "structured",
        promptSchemaVersion: 1,
        structuredPromptDefinition: makeDefinition("Explain {{topic}}"),
        fewShotExamples: [
          {
            inputs: { topic: "SQLite" },
            outputs: { answer: "Use FTS5." }
          }
        ],
        modulesConfig: [
          {
            type: "style_rules",
            enabled: true,
            config: { tone: "formal" }
          }
        ]
      })
    )
  })

  it("treats structured definition changes as sync conflicts even when legacy text matches", async () => {
    state.prompts.set("local-conflict-structured", {
      id: "local-conflict-structured",
      title: "Structured Prompt",
      name: "Structured Prompt",
      content: "Explain {{topic}}",
      is_system: false,
      system_prompt: "Legacy system snapshot",
      user_prompt: "Explain {{topic}}",
      promptFormat: "structured",
      promptSchemaVersion: 1,
      structuredPromptDefinition: makeDefinition("Explain {{topic}}"),
      createdAt: 1,
      updatedAt: 20,
      serverId: 77,
      syncStatus: "synced",
      serverUpdatedAt: "2026-03-10T00:00:00Z",
      lastSyncedAt: 10
    })

    mocks.getPrompt.mockResolvedValue({
      data: {
        data: {
          id: 77,
          project_id: 9,
          name: "Structured Prompt",
          system_prompt: "Legacy system snapshot",
          user_prompt: "Explain {{topic}}",
          prompt_format: "structured",
          prompt_schema_version: 1,
          prompt_definition: makeDefinition("Summarize {{topic}}"),
          version_number: 2,
          updated_at: "2026-03-10T01:00:00Z"
        }
      }
    })

    const { getSyncStatus } = await importPromptSync()
    await expect(getSyncStatus("local-conflict-structured")).resolves.toEqual(
      expect.objectContaining({
        status: "conflict",
        hasConflict: true
      })
    )
  })

  it("treats few-shot or module changes as sync conflicts for legacy prompts", async () => {
    state.prompts.set("local-conflict-legacy", {
      id: "local-conflict-legacy",
      title: "Legacy Prompt",
      name: "Legacy Prompt",
      content: "Explain topic",
      is_system: false,
      system_prompt: "Legacy system snapshot",
      user_prompt: "Explain topic",
      fewShotExamples: [
        {
          inputs: { topic: "SQLite" },
          outputs: { answer: "Use indexes." }
        }
      ],
      modulesConfig: [
        {
          type: "style_rules",
          enabled: true,
          config: { tone: "concise" }
        }
      ],
      createdAt: 1,
      updatedAt: 20,
      serverId: 88,
      syncStatus: "synced",
      serverUpdatedAt: "2026-03-10T00:00:00Z",
      lastSyncedAt: 10
    })

    mocks.getPrompt.mockResolvedValue({
      data: {
        data: {
          id: 88,
          project_id: 9,
          name: "Legacy Prompt",
          system_prompt: "Legacy system snapshot",
          user_prompt: "Explain topic",
          few_shot_examples: [
            {
              inputs: { topic: "SQLite" },
              outputs: { answer: "Use FTS5." }
            }
          ],
          modules_config: [
            {
              type: "style_rules",
              enabled: true,
              config: { tone: "formal" }
            }
          ],
          version_number: 2,
          updated_at: "2026-03-10T01:00:00Z"
        }
      }
    })

    const { getSyncStatus } = await importPromptSync()
    await expect(getSyncStatus("local-conflict-legacy")).resolves.toEqual(
      expect.objectContaining({
        status: "conflict",
        hasConflict: true
      })
    )
  })

  it("pushes a validated recipe without runtime values and preserves server identity", async () => {
    const definition = makeRecipeDefinition()
    state.prompts.set("local-recipe", {
      id: "local-recipe",
      title: "Recipe",
      name: "Recipe",
      content: "## Objective\n\nExplain {{topic}}.",
      is_system: false,
      system_prompt: "",
      user_prompt: "## Objective\n\nExplain {{topic}}.",
      promptFormat: "structured",
      promptSchemaVersion: 2,
      structuredPromptDefinition: definition,
      createdAt: 1,
      updatedAt: 1,
      syncStatus: "local"
    })
    mocks.createPrompt.mockResolvedValue({
      data: {
        data: {
          id: 601,
          project_id: 42,
          name: "Recipe",
          system_prompt: "",
          user_prompt: "## Objective\n\nExplain {{topic}}.",
          prompt_format: "structured",
          prompt_schema_version: 2,
          prompt_definition: definition,
          version_number: 1,
          updated_at: "2026-03-10T00:00:00Z"
        }
      }
    })

    const { pushToStudio } = await importPromptSync()
    await expect(pushToStudio("local-recipe", 42)).resolves.toEqual(
      expect.objectContaining({ success: true, syncStatus: "synced" })
    )

    const payload = mocks.createPrompt.mock.calls[0][0]
    expect(payload.prompt_schema_version).toBe(2)
    expect(payload.prompt_definition).toEqual(definition)
    expect(JSON.stringify(payload)).not.toContain("runtime_values")
    expect(state.prompts.get("local-recipe")).toEqual(
      expect.objectContaining({
        promptSchemaVersion: 2,
        structuredPromptDefinition: definition,
        syncPayloadVersion: 1
      })
    )
  })

  it("rejects an unsafe local recipe before calling the server", async () => {
    const unsafeDefinition = {
      ...makeRecipeDefinition(),
      runtime_values: { topic: "PRIVATE_LOCAL_VALUE" }
    }
    const original = {
      id: "unsafe-local-recipe",
      title: "Unsafe Recipe",
      name: "Unsafe Recipe",
      content: "safe snapshot",
      is_system: false,
      user_prompt: "safe snapshot",
      promptFormat: "structured",
      promptSchemaVersion: 2,
      structuredPromptDefinition: unsafeDefinition,
      createdAt: 1,
      updatedAt: 1,
      syncStatus: "local"
    }
    state.prompts.set(original.id, original)

    const { pushToStudio } = await importPromptSync()
    const result = await pushToStudio(original.id, 42)

    expect(result).toEqual(
      expect.objectContaining({
        success: false,
        error: "invalid_recipe_runtime_values"
      })
    )
    expect(mocks.createPrompt).not.toHaveBeenCalled()
    expect(state.prompts.get(original.id)).toEqual(original)
  })

  it("rejects a malicious server recipe without overwriting a good local record", async () => {
    const goodDefinition = makeRecipeDefinition("Keep {{topic}} safe.")
    const original = {
      id: "good-local-recipe",
      title: "Good Recipe",
      name: "Good Recipe",
      content: "## Objective\n\nKeep {{topic}} safe.",
      is_system: false,
      user_prompt: "## Objective\n\nKeep {{topic}} safe.",
      promptFormat: "structured",
      promptSchemaVersion: 2,
      structuredPromptDefinition: goodDefinition,
      createdAt: 1,
      updatedAt: 10,
      serverId: 701,
      syncStatus: "synced"
    }
    state.prompts.set(original.id, original)
    mocks.getPrompt.mockResolvedValue({
      data: {
        data: {
          id: 701,
          project_id: 42,
          name: "Poisoned Recipe",
          system_prompt: "",
          user_prompt: "PRIVATE_SERVER_VALUE",
          prompt_format: "structured",
          prompt_schema_version: 2,
          prompt_definition: {
            ...makeRecipeDefinition(),
            resolved_values: { topic: "PRIVATE_SERVER_VALUE" }
          },
          version_number: 2,
          updated_at: "2026-03-10T01:00:00Z"
        }
      }
    })

    const { pullFromStudio } = await importPromptSync()
    const result = await pullFromStudio(701, original.id)

    expect(result).toEqual(
      expect.objectContaining({
        success: false,
        error: "invalid_recipe_runtime_values"
      })
    )
    expect(mocks.promptUpdate).not.toHaveBeenCalled()
    expect(state.prompts.get(original.id)).toEqual(original)
  })

  it("hashes recipe identity and compiled snapshots into conflict detection", async () => {
    state.prompts.set("recipe-identity-conflict", {
      id: "recipe-identity-conflict",
      title: "Same text",
      name: "Same text",
      content: "same text",
      is_system: false,
      system_prompt: "",
      user_prompt: "same text",
      promptFormat: "structured",
      promptSchemaVersion: 1,
      structuredPromptDefinition: makeDefinition("same text"),
      createdAt: 1,
      updatedAt: 20,
      serverId: 702,
      syncStatus: "synced",
      serverUpdatedAt: "2026-03-10T00:00:00Z",
      lastSyncedAt: 10
    })
    mocks.getPrompt.mockResolvedValue({
      data: {
        data: {
          id: 702,
          project_id: 42,
          name: "Same text",
          system_prompt: "",
          user_prompt: "same text",
          prompt_format: "structured",
          prompt_schema_version: 2,
          prompt_definition: makeRecipeDefinition("same text"),
          version_number: 2,
          updated_at: "2026-03-10T01:00:00Z"
        }
      }
    })

    const { getSyncStatus } = await importPromptSync()
    await expect(getSyncStatus("recipe-identity-conflict")).resolves.toEqual(
      expect.objectContaining({ status: "conflict", hasConflict: true })
    )

    const recipe = makeRecipeDefinition("same text")
    state.prompts.set("recipe-snapshot-conflict", {
      id: "recipe-snapshot-conflict",
      title: "Recipe",
      name: "Recipe",
      content: "local snapshot",
      is_system: false,
      system_prompt: "",
      user_prompt: "local snapshot",
      promptFormat: "structured",
      promptSchemaVersion: 2,
      structuredPromptDefinition: recipe,
      createdAt: 1,
      updatedAt: 20,
      serverId: 703,
      syncStatus: "synced",
      serverUpdatedAt: "2026-03-10T00:00:00Z",
      lastSyncedAt: 10
    })
    mocks.getPrompt.mockResolvedValue({
      data: {
        data: {
          id: 703,
          project_id: 42,
          name: "Recipe",
          system_prompt: "",
          user_prompt: "server snapshot",
          prompt_format: "structured",
          prompt_schema_version: 2,
          prompt_definition: recipe,
          version_number: 2,
          updated_at: "2026-03-10T01:00:00Z"
        }
      }
    })
    await expect(getSyncStatus("recipe-snapshot-conflict")).resolves.toEqual(
      expect.objectContaining({ status: "conflict", hasConflict: true })
    )
  })
})
