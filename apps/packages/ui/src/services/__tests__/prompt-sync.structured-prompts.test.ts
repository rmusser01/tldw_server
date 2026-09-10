import { beforeEach, describe, expect, it, vi } from "vitest"
import { existsSync, readFileSync } from "node:fs"
import { resolve } from "node:path"
import type {
  StructuredPromptDefinition,
  StructuredPromptPreviewResponse
} from "@/services/prompt-studio"

const state = vi.hoisted(() => ({
  prompts: new Map<string, any>()
}))

const mocks = vi.hoisted(() => ({
  getDefaults: vi.fn(),
  setDefaults: vi.fn(),
  listProjects: vi.fn(),
  createProject: vi.fn(),
  createPrompt: vi.fn(),
  updatePrompt: vi.fn(),
  getPrompt: vi.fn(),
  promptAdd: vi.fn(),
  promptGet: vi.fn(),
  promptUpdate: vi.fn(),
  promptWhere: vi.fn()
}))

vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: (...args: unknown[]) => mocks.getDefaults(...args),
  setPromptStudioDefaults: (...args: unknown[]) => mocks.setDefaults(...args)
}))

vi.mock("@/services/prompt-studio", () => ({
  listProjects: (...args: unknown[]) => mocks.listProjects(...args),
  createProject: (...args: unknown[]) => mocks.createProject(...args),
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
      label: null,
      description: null,
      required: true,
      default_value: null,
      input_type: "text",
      options: null,
      max_length: null
    }
  ],
  blocks: [
    {
      id: "task",
      name: "Task",
      role: "user",
      kind: null,
      content,
      enabled: true,
      order: 10,
      is_template: true
    }
  ],
  assembly_config: {
    legacy_system_roles: ["system", "developer"],
    legacy_user_roles: ["user"],
    block_separator: "\n\n"
  }
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

type V1IntegerSyncCase = {
  name: string
  field: "order" | "max_length" | "schema_version"
  input_value: number | string | null
  python_value: number | null
  sync_eligible: boolean
}
const integerFixtureRelativePath =
  "Docs/fixtures/single-text-recipes/v1-integer-sync-cases.json"
const integerFixturePath = ["", "..", "../..", "../../..", "../../../.."]
  .map((prefix) => resolve(process.cwd(), prefix, integerFixtureRelativePath))
  .find(existsSync)
if (!integerFixturePath) throw new Error("v1 integer sync fixture not found")
const v1IntegerSyncCases = JSON.parse(
  readFileSync(integerFixturePath, "utf8")
) as V1IntegerSyncCase[]

const makeV1IntegerDefinition = (
  testCase: Pick<V1IntegerSyncCase, "field" | "input_value">
) => {
  const definition = makeDefinition("Explain {{topic}}.")
  if (testCase.field === "schema_version") {
    definition.schema_version = testCase.input_value as number
  } else if (testCase.field === "order") {
    definition.blocks[0].order = testCase.input_value as number
  } else {
    definition.variables[0].max_length = testCase.input_value as number
  }
  return definition
}

const makeCanonicalV1IntegerDefinition = (
  testCase: Pick<V1IntegerSyncCase, "field" | "python_value">
) =>
  makeV1IntegerDefinition({
    field: testCase.field,
    input_value: testCase.python_value
  })

const getV1IntegerValue = (
  definition: ReturnType<typeof makeDefinition>,
  field: V1IntegerSyncCase["field"]
) =>
  field === "schema_version"
    ? definition.schema_version
    : field === "order"
      ? definition.blocks[0].order
      : definition.variables[0].max_length

describe("prompt-sync structured prompt support", () => {
  beforeEach(() => {
    state.prompts.clear()

    mocks.getDefaults.mockReset().mockResolvedValue({
      defaultProjectId: null,
      autoSyncWorkspacePrompts: true
    })
    mocks.setDefaults.mockReset().mockResolvedValue(undefined)
    mocks.listProjects.mockReset().mockResolvedValue({ data: { data: [] } })
    mocks.createProject
      .mockReset()
      .mockResolvedValue({ data: { data: { id: 42 } } })
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

  it("pushes a Python-valid sparse v1 definition in canonical wire form", async () => {
    const sparseDefinition = {
      schema_version: 1,
      variables: [],
      blocks: []
    }
    const canonicalDefinition = {
      schema_version: 1,
      format: "structured",
      variables: [],
      blocks: [],
      assembly_config: {
        legacy_system_roles: ["system", "developer"],
        legacy_user_roles: ["user"],
        block_separator: "\n\n"
      }
    }
    state.prompts.set("local-sparse-v1", {
      id: "local-sparse-v1",
      title: "Sparse v1",
      name: "Sparse v1",
      content: "legacy snapshot",
      is_system: false,
      user_prompt: "legacy snapshot",
      promptFormat: "structured",
      promptSchemaVersion: 1,
      structuredPromptDefinition: sparseDefinition,
      createdAt: 1,
      updatedAt: 1,
      syncStatus: "local"
    })
    mocks.createPrompt.mockResolvedValue({
      data: {
        data: {
          id: 102,
          project_id: 42,
          name: "Sparse v1",
          system_prompt: "",
          user_prompt: "legacy snapshot",
          prompt_format: "structured",
          prompt_schema_version: 1,
          prompt_definition: canonicalDefinition,
          version_number: 1,
          updated_at: "2026-03-10T00:00:00Z"
        }
      }
    })

    const { pushToStudio } = await importPromptSync()
    const result = await pushToStudio("local-sparse-v1", 42)

    expect(result).toEqual(
      expect.objectContaining({ success: true, syncStatus: "synced" })
    )
    expect(mocks.createPrompt).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt_format: "structured",
        prompt_schema_version: 1,
        prompt_definition: canonicalDefinition
      })
    )
  })

  it.each(v1IntegerSyncCases)(
    "push enforces the exact v1 integer sync contract: $name",
    async (testCase) => {
      const definition = makeV1IntegerDefinition(testCase)
      const definitionBefore = JSON.stringify(definition)
      const original = {
        id: `push-${testCase.name}`,
        title: "V1 integer prompt",
        name: "V1 integer prompt",
        content: "Explain {{topic}}.",
        is_system: false,
        user_prompt: "Explain {{topic}}.",
        promptFormat: "structured" as const,
        promptSchemaVersion: 1,
        structuredPromptDefinition: definition,
        createdAt: 1,
        updatedAt: 1,
        syncStatus: "local" as const
      }
      state.prompts.set(original.id, structuredClone(original))
      const before = JSON.stringify(state.prompts.get(original.id))
      if (testCase.sync_eligible) {
        const canonicalDefinition = makeCanonicalV1IntegerDefinition(testCase)
        mocks.createPrompt.mockResolvedValueOnce({
          data: {
            data: {
              id: 810,
              project_id: 42,
              name: "V1 integer prompt",
              system_prompt: "",
              user_prompt: "Explain {{topic}}.",
              prompt_format: "structured",
              prompt_schema_version: 1,
              prompt_definition: canonicalDefinition,
              version_number: 1,
              updated_at: "2026-03-10T00:00:00Z"
            }
          }
        })
      }

      const { pushToStudio } = await importPromptSync()
      const result = await pushToStudio(original.id, 42)

      if (testCase.sync_eligible) {
        expect(result).toEqual(
          expect.objectContaining({ success: true, syncStatus: "synced" })
        )
        const payload = mocks.createPrompt.mock.calls[0][0]
        expect(
          getV1IntegerValue(payload.prompt_definition, testCase.field)
        ).toBe(testCase.python_value)
        expect(mocks.promptUpdate).toHaveBeenCalledTimes(1)
        expect(payload.prompt_definition).toEqual(
          makeCanonicalV1IntegerDefinition(testCase)
        )
        expect(
          state.prompts.get(original.id).structuredPromptDefinition
        ).toEqual(makeCanonicalV1IntegerDefinition(testCase))
      } else {
        expect(result).toEqual(
          expect.objectContaining({
            success: false,
            error: "invalid_prompt_definition",
            failureKind: "validation",
            syncStatus: "local"
          })
        )
        expect(mocks.createPrompt).not.toHaveBeenCalled()
        expect(mocks.updatePrompt).not.toHaveBeenCalled()
        expect(mocks.promptUpdate).not.toHaveBeenCalled()
        expect(JSON.stringify(state.prompts.get(original.id))).toBe(before)
      }
      expect(JSON.stringify(definition)).toBe(definitionBefore)
    }
  )

  it.each(v1IntegerSyncCases.filter((testCase) => !testCase.sync_eligible))(
    "auto-sync rejects v1 integers before defaults, requests, or writes: $name",
    async (testCase) => {
      const original = {
        id: `auto-${testCase.name}`,
        title: "V1 integer prompt",
        content: "Explain {{topic}}.",
        user_prompt: "Explain {{topic}}.",
        is_system: false,
        promptFormat: "structured",
        promptSchemaVersion: 1,
        structuredPromptDefinition: makeV1IntegerDefinition(testCase),
        syncStatus: "conflict",
        serverId: 814,
        studioPromptId: 814,
        studioProjectId: null,
        updatedAt: 20,
        createdAt: 1,
        lastSyncedAt: 10,
        serverUpdatedAt: "2026-03-10T00:00:00Z"
      }
      state.prompts.set(original.id, structuredClone(original))
      const { autoSyncPrompt } = await importPromptSync()

      expect(await autoSyncPrompt(original.id)).toEqual({
        success: false,
        localId: original.id,
        serverId: 814,
        error: "invalid_prompt_definition",
        syncStatus: "conflict",
        failureKind: "validation"
      })
      expect(mocks.getDefaults).not.toHaveBeenCalled()
      expect(mocks.setDefaults).not.toHaveBeenCalled()
      expect(mocks.listProjects).not.toHaveBeenCalled()
      expect(mocks.createProject).not.toHaveBeenCalled()
      expect(mocks.createPrompt).not.toHaveBeenCalled()
      expect(mocks.updatePrompt).not.toHaveBeenCalled()
      expect(mocks.getPrompt).not.toHaveBeenCalled()
      expect(mocks.promptAdd).not.toHaveBeenCalled()
      expect(mocks.promptUpdate).not.toHaveBeenCalled()
      expect(state.prompts.get(original.id)).toEqual(original)
    }
  )

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

  it.each(
    v1IntegerSyncCases.flatMap((testCase) =>
      [false, true].map((existing) => ({ ...testCase, existing }))
    )
  )(
    "pull enforces the exact v1 integer sync contract (existing=$existing): $name",
    async (testCase) => {
      const definition = makeV1IntegerDefinition(testCase)
      const definitionBefore = JSON.stringify(definition)
      const original = {
        id: "existing-local-id",
        title: "Good local record",
        content: "Explain {{topic}}.",
        user_prompt: "Explain {{topic}}.",
        is_system: false,
        promptFormat: "structured",
        promptSchemaVersion: 1,
        structuredPromptDefinition: makeDefinition("Explain {{topic}}."),
        syncStatus: "conflict",
        serverId: 811,
        studioPromptId: 811,
        studioProjectId: 42,
        updatedAt: 20,
        createdAt: 1,
        lastSyncedAt: 10,
        serverUpdatedAt: "2026-03-10T00:00:00Z"
      }
      if (testCase.existing) {
        state.prompts.set(original.id, structuredClone(original))
      }
      mocks.getPrompt.mockResolvedValueOnce({
        data: {
          data: {
            id: 811,
            project_id: 42,
            name: "V1 integer prompt",
            system_prompt: "",
            user_prompt: "Explain {{topic}}.",
            prompt_format: "structured",
            prompt_schema_version: 1,
            prompt_definition: definition,
            version_number: 1,
            updated_at: "2026-03-10T00:00:00Z"
          }
        }
      })

      const { pullFromStudio } = await importPromptSync()
      const result = await pullFromStudio(
        811,
        testCase.existing ? original.id : undefined
      )

      if (testCase.sync_eligible) {
        expect(result).toEqual(
          expect.objectContaining({ success: true, syncStatus: "synced" })
        )
        const saved = state.prompts.get(
          testCase.existing ? original.id : "generated-local-id"
        )
        expect(
          getV1IntegerValue(saved.structuredPromptDefinition, testCase.field)
        ).toBe(testCase.python_value)
        expect(mocks.promptAdd).toHaveBeenCalledTimes(testCase.existing ? 0 : 1)
        expect(mocks.promptUpdate).toHaveBeenCalledTimes(
          testCase.existing ? 1 : 0
        )
        expect(saved.structuredPromptDefinition).toEqual(
          makeCanonicalV1IntegerDefinition(testCase)
        )
      } else {
        expect(result).toEqual(
          expect.objectContaining({
            success: false,
            error: "invalid_prompt_definition"
          })
        )
        expect(mocks.promptAdd).not.toHaveBeenCalled()
        expect(mocks.promptUpdate).not.toHaveBeenCalled()
        expect(state.prompts.size).toBe(testCase.existing ? 1 : 0)
        if (testCase.existing) {
          expect(state.prompts.get(original.id)).toEqual(original)
        }
      }
      expect(JSON.stringify(definition)).toBe(definitionBefore)
    }
  )

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

  it.each(v1IntegerSyncCases)(
    "local conflict validation enforces the exact v1 integer sync contract: $name",
    async (testCase) => {
      const definition = makeV1IntegerDefinition(testCase)
      const original = {
        id: `local-conflict-${testCase.name}`,
        title: "V1 integer prompt",
        name: "V1 integer prompt",
        content: "Explain {{topic}}.",
        is_system: false,
        user_prompt: "Explain {{topic}}.",
        promptFormat: "structured" as const,
        promptSchemaVersion: 1,
        structuredPromptDefinition: definition,
        createdAt: 1,
        updatedAt: 20,
        serverId: 812,
        syncStatus: "synced" as const,
        serverUpdatedAt: "2026-03-10T00:00:00Z",
        lastSyncedAt: 10
      }
      state.prompts.set(original.id, structuredClone(original))
      mocks.getPrompt.mockResolvedValueOnce({
        data: {
          data: {
            id: 812,
            project_id: 42,
            name: "V1 integer prompt",
            system_prompt: "",
            user_prompt: "Explain {{topic}}.",
            prompt_format: "structured",
            prompt_schema_version: 1,
            prompt_definition: makeCanonicalV1IntegerDefinition(testCase),
            version_number: 2,
            updated_at: "2026-03-10T01:00:00Z"
          }
        }
      })

      const { getSyncStatus } = await importPromptSync()
      const result = await getSyncStatus(original.id)

      expect(result).toEqual(
        expect.objectContaining({ status: "synced", hasConflict: false })
      )
      expect(mocks.getPrompt).toHaveBeenCalledTimes(
        testCase.sync_eligible ? 1 : 0
      )
      expect(mocks.promptUpdate).not.toHaveBeenCalled()
      expect(state.prompts.get(original.id)).toEqual(original)
    }
  )

  it.each(v1IntegerSyncCases)(
    "server conflict validation enforces the exact v1 integer sync contract: $name",
    async (testCase) => {
      const localDefinition = makeDefinition("Explain {{topic}}.")
      if (testCase.field === "order") localDefinition.blocks[0].order = 42
      const original = {
        id: `server-conflict-${testCase.name}`,
        title: "V1 integer prompt",
        name: "V1 integer prompt",
        content: "Explain {{topic}}.",
        is_system: false,
        user_prompt: "Explain {{topic}}.",
        promptFormat: "structured" as const,
        promptSchemaVersion: 1,
        structuredPromptDefinition: localDefinition,
        createdAt: 1,
        updatedAt: 20,
        serverId: 813,
        syncStatus: "synced" as const,
        serverUpdatedAt: "2026-03-10T00:00:00Z",
        lastSyncedAt: 10
      }
      state.prompts.set(original.id, structuredClone(original))
      mocks.getPrompt.mockResolvedValueOnce({
        data: {
          data: {
            id: 813,
            project_id: 42,
            name: "V1 integer prompt",
            system_prompt: "",
            user_prompt: "Explain {{topic}}.",
            prompt_format: "structured",
            prompt_schema_version: 1,
            prompt_definition: makeV1IntegerDefinition(testCase),
            version_number: 2,
            updated_at: "2026-03-10T01:00:00Z"
          }
        }
      })

      const { getSyncStatus } = await importPromptSync()
      const result = await getSyncStatus(original.id)

      const hasConflict =
        testCase.sync_eligible &&
        testCase.field !== "schema_version" &&
        !(testCase.field === "max_length" && testCase.python_value === null)
      expect(result).toEqual(
        expect.objectContaining({
          status: hasConflict ? "conflict" : "synced",
          hasConflict
        })
      )
      expect(mocks.getPrompt).toHaveBeenCalledTimes(1)
      expect(mocks.promptUpdate).not.toHaveBeenCalled()
      expect(state.prompts.get(original.id)).toEqual(original)
    }
  )

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

  it.each([
    ["outer version mismatch", makeRecipeDefinition(), 1],
    [
      "malformed recipe block",
      (() => {
        const value = makeRecipeDefinition()
        value.blocks[0].id = "   "
        return value
      })(),
      2
    ],
    ["future outer version", makeRecipeDefinition(), 99]
  ])(
    "rejects invalid local transport identity (%s) before any request",
    async (_caseName, definition, outerVersion) => {
      const original = {
        id: `invalid-local-${outerVersion}-${_caseName}`,
        title: "Invalid Local Recipe",
        name: "Invalid Local Recipe",
        content: "safe snapshot",
        is_system: false,
        user_prompt: "safe snapshot",
        promptFormat: "structured" as const,
        promptSchemaVersion: outerVersion,
        structuredPromptDefinition: definition,
        createdAt: 1,
        updatedAt: 1,
        syncStatus: "local" as const
      }
      state.prompts.set(original.id, structuredClone(original))
      const before = JSON.stringify(state.prompts.get(original.id))

      const { pushToStudio } = await importPromptSync()
      const result = await pushToStudio(original.id, 42)

      expect(result.success).toBe(false)
      expect(mocks.createPrompt).not.toHaveBeenCalled()
      expect(mocks.updatePrompt).not.toHaveBeenCalled()
      expect(mocks.promptUpdate).not.toHaveBeenCalled()
      expect(JSON.stringify(state.prompts.get(original.id))).toBe(before)
    }
  )

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

  it.each(["pull", "link"])(
    "%s rejects mismatched server transport identity without local or link mutation",
    async (operation) => {
      const goodDefinition = makeRecipeDefinition("Keep {{topic}} safe.")
      const original = {
        id: `server-mismatch-${operation}`,
        title: "Good Recipe",
        name: "Good Recipe",
        content: "safe snapshot",
        is_system: false,
        user_prompt: "safe snapshot",
        promptFormat: "structured" as const,
        promptSchemaVersion: 2,
        structuredPromptDefinition: goodDefinition,
        createdAt: 1,
        updatedAt: 10,
        serverId: operation === "pull" ? 704 : undefined,
        syncStatus: "synced" as const
      }
      state.prompts.set(original.id, structuredClone(original))
      const before = JSON.stringify(state.prompts.get(original.id))
      mocks.getPrompt.mockResolvedValue({
        data: {
          data: {
            id: 704,
            project_id: 42,
            name: "Mismatched Recipe",
            system_prompt: "",
            user_prompt: "server snapshot",
            prompt_format: "structured",
            prompt_schema_version: 1,
            prompt_definition: goodDefinition,
            version_number: 2,
            updated_at: "2026-03-10T01:00:00Z"
          }
        }
      })

      const { linkPrompts, pullFromStudio } = await importPromptSync()
      const result =
        operation === "pull"
          ? await pullFromStudio(704, original.id)
          : await linkPrompts(original.id, 704)

      expect(result.success).toBe(false)
      expect(mocks.promptUpdate).not.toHaveBeenCalled()
      expect(JSON.stringify(state.prompts.get(original.id))).toBe(before)
    }
  )

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
