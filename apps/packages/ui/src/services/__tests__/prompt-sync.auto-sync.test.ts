import { beforeEach, describe, expect, it, vi } from "vitest"

const state = vi.hoisted(() => ({
  defaults: {
    defaultProjectId: null as number | null,
    autoSyncWorkspacePrompts: true
  },
  prompts: new Map<string, any>()
}))

const mocks = vi.hoisted(() => ({
  getPromptStudioDefaults: vi.fn(),
  setPromptStudioDefaults: vi.fn(),
  listProjects: vi.fn(),
  createProject: vi.fn(),
  createPrompt: vi.fn(),
  updatePrompt: vi.fn(),
  getPrompt: vi.fn(),
  promptGet: vi.fn(),
  promptUpdate: vi.fn(),
  promptWhere: vi.fn()
}))

vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: (...args: unknown[]) =>
    (mocks.getPromptStudioDefaults as (...args: unknown[]) => unknown)(...args),
  setPromptStudioDefaults: (...args: unknown[]) =>
    (mocks.setPromptStudioDefaults as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/prompt-studio", () => ({
  listProjects: (...args: unknown[]) =>
    (mocks.listProjects as (...args: unknown[]) => unknown)(...args),
  createProject: (...args: unknown[]) =>
    (mocks.createProject as (...args: unknown[]) => unknown)(...args),
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
  PageAssistDatabase: class {
    async getAllPrompts() {
      return Array.from(state.prompts.values())
    }
  }
}))

vi.mock("@/db/dexie/helpers", () => ({
  generateID: () => "generated-local-id"
}))

const importPromptSync = async () => import("@/services/prompt-sync")

const makeRecipeDefinition = () => ({
  schema_version: 2 as const,
  format: "structured" as const,
  definition_kind: "single_text_recipe" as const,
  variables: [],
  blocks: [
    {
      id: "objective",
      name: "Objective",
      section_key: "objective",
      role: "user" as const,
      kind: "objective",
      content: "Explain the task.",
      enabled: true,
      order: 10,
      is_template: false
    }
  ],
  assembly_config: {
    assembly_mode: "single_text" as const,
    target_role: "user" as const,
    render_format: "freeform" as const,
    block_separator: "\n\n"
  }
})

const invalidServerIdentityCases = [
  {
    name: "runtime map",
    outerVersion: 2,
    definition: {
      ...makeRecipeDefinition(),
      resolved_values: { topic: "PRIVATE_AUTO_SYNC_SENTINEL" }
    }
  },
  {
    name: "malformed block",
    outerVersion: 2,
    definition: {
      ...makeRecipeDefinition(),
      blocks: [
        {
          ...makeRecipeDefinition().blocks[0],
          id: "",
          content: "PRIVATE_AUTO_SYNC_SENTINEL"
        }
      ]
    }
  },
  {
    name: "future version",
    outerVersion: 99,
    definition: {
      ...makeRecipeDefinition(),
      schema_version: 99
    }
  },
  {
    name: "outer mismatch",
    outerVersion: 1,
    definition: makeRecipeDefinition()
  }
]

const invalidAutoSyncCases = (["create", "update"] as const).flatMap(
  (operation) =>
    invalidServerIdentityCases.map((testCase) => ({ operation, ...testCase }))
)

describe("prompt-sync auto-sync defaults", () => {
  beforeEach(() => {
    state.defaults = {
      defaultProjectId: null,
      autoSyncWorkspacePrompts: true
    }
    state.prompts.clear()

    mocks.getPromptStudioDefaults.mockReset()
    mocks.setPromptStudioDefaults.mockReset()
    mocks.listProjects.mockReset()
    mocks.createProject.mockReset()
    mocks.createPrompt.mockReset()
    mocks.updatePrompt.mockReset()
    mocks.getPrompt.mockReset()
    mocks.promptGet.mockReset()
    mocks.promptUpdate.mockReset()
    mocks.promptWhere.mockReset()

    mocks.getPromptStudioDefaults.mockImplementation(async () => ({
      ...state.defaults
    }))
    mocks.setPromptStudioDefaults.mockImplementation(
      async (updates: Record<string, unknown>) => {
        state.defaults = { ...state.defaults, ...updates }
        return { ...state.defaults }
      }
    )
    mocks.listProjects.mockResolvedValue({ data: { data: [] } })
    mocks.createProject.mockResolvedValue({
      data: { data: { id: 17, name: "Workspace Prompts" } }
    })
    mocks.createPrompt.mockResolvedValue({
      data: {
        data: {
          id: 101,
          project_id: 17,
          name: "Prompt",
          system_prompt: null,
          user_prompt: "hello",
          version_number: 1,
          updated_at: "2026-02-17T00:00:00Z"
        }
      }
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

  it("enables auto-sync by default and can be disabled", async () => {
    const { shouldAutoSyncWorkspacePrompts } = await importPromptSync()

    await expect(shouldAutoSyncWorkspacePrompts()).resolves.toBe(true)

    state.defaults.autoSyncWorkspacePrompts = false
    await expect(shouldAutoSyncWorkspacePrompts()).resolves.toBe(false)
  })

  it("uses configured default project when available", async () => {
    state.defaults.defaultProjectId = 42
    const { resolveAutoSyncProjectId } = await importPromptSync()

    await expect(resolveAutoSyncProjectId()).resolves.toBe(42)
    expect(mocks.listProjects).not.toHaveBeenCalled()
  })

  it("chooses and persists first available project when default is missing", async () => {
    mocks.listProjects.mockResolvedValue({
      data: { data: [{ id: 7, name: "Team prompts" }] }
    })
    const { resolveAutoSyncProjectId } = await importPromptSync()

    await expect(resolveAutoSyncProjectId()).resolves.toBe(7)
    expect(mocks.setPromptStudioDefaults).toHaveBeenCalledWith(
      expect.objectContaining({ defaultProjectId: 7 })
    )
  })

  it("auto-creates a sync project and syncs prompt when none exists", async () => {
    state.prompts.set("local-1", {
      id: "local-1",
      title: "Prompt",
      name: "Prompt",
      content: "hello",
      user_prompt: "hello",
      is_system: false,
      createdAt: 1,
      updatedAt: 1,
      syncStatus: "local"
    })
    const { autoSyncPrompt } = await importPromptSync()

    const result = await autoSyncPrompt("local-1")
    expect(result.success).toBe(true)
    expect(result.syncStatus).toBe("synced")
    expect(mocks.createProject).toHaveBeenCalledTimes(1)
    expect(mocks.createPrompt).toHaveBeenCalledWith(
      expect.objectContaining({ project_id: 17, name: "Prompt" })
    )
    expect(state.prompts.get("local-1")).toEqual(
      expect.objectContaining({
        serverId: 101,
        studioProjectId: 17,
        syncStatus: "synced"
      })
    )
  })

  it("marks prompt pending when project resolution fails", async () => {
    state.prompts.set("local-2", {
      id: "local-2",
      title: "Pending prompt",
      name: "Pending prompt",
      content: "test",
      is_system: false,
      createdAt: 1,
      updatedAt: 1,
      syncStatus: "local"
    })
    mocks.createProject.mockRejectedValueOnce(new Error("forbidden"))
    const { autoSyncPrompt } = await importPromptSync()

    const result = await autoSyncPrompt("local-2")
    expect(result.success).toBe(false)
    expect(result.syncStatus).toBe("pending")
    expect(state.prompts.get("local-2")).toEqual(
      expect.objectContaining({
        syncStatus: "pending"
      })
    )
  })

  it("keeps the legacy pending result when an unlinked server create is empty", async () => {
    state.prompts.set("local-empty-create", {
      id: "local-empty-create",
      title: "Pending prompt",
      name: "Pending prompt",
      content: "test",
      is_system: false,
      createdAt: 1,
      updatedAt: 1,
      syncStatus: "local"
    })
    mocks.createPrompt.mockResolvedValueOnce({ data: { data: null } })
    const { pushToStudio } = await importPromptSync()

    const result = await pushToStudio("local-empty-create", 17)

    expect(result.success).toBe(false)
    expect(result.syncStatus).toBe("pending")
    expect(mocks.promptUpdate).not.toHaveBeenCalled()
  })

  it.each(invalidAutoSyncCases)(
    "quarantines invalid $operation response identity: $name",
    async ({ operation, outerVersion, definition }) => {
      const sentinel = "PRIVATE_AUTO_SYNC_SENTINEL"
      const original = {
        id: `invalid-auto-${operation}-${outerVersion}`,
        title: "Good local recipe",
        name: "Good local recipe",
        content: "Safe local snapshot",
        is_system: false,
        system_prompt: "",
        user_prompt: "Safe local snapshot",
        promptFormat: "structured" as const,
        promptSchemaVersion: 2,
        structuredPromptDefinition: makeRecipeDefinition(),
        createdAt: 1,
        updatedAt: 25,
        studioProjectId: 17,
        syncStatus:
          operation === "create" ? ("local" as const) : ("synced" as const),
        ...(operation === "update"
          ? {
              serverId: 701,
              studioPromptId: 701,
              lastSyncedAt: 20,
              serverUpdatedAt: "2026-03-10T00:00:00Z"
            }
          : {})
      }
      state.prompts.set(original.id, structuredClone(original))
      const serverResponse = {
        data: {
          data: {
            id: 801,
            project_id: 17,
            name: sentinel,
            system_prompt: "",
            user_prompt: sentinel,
            prompt_format: "structured",
            prompt_schema_version: outerVersion,
            prompt_definition: definition,
            version_number: 2,
            updated_at: "2026-03-10T01:00:00Z"
          }
        }
      }
      if (operation === "create") {
        mocks.createPrompt.mockResolvedValueOnce(serverResponse)
      } else {
        mocks.updatePrompt.mockResolvedValueOnce(serverResponse)
      }
      const captured: string[] = []
      const warn = vi.spyOn(console, "warn").mockImplementation((...items) => {
        captured.push(items.join(" "))
      })
      const error = vi
        .spyOn(console, "error")
        .mockImplementation((...items) => {
          captured.push(items.join(" "))
        })
      try {
        const { autoSyncPrompt } = await importPromptSync()
        const result = await autoSyncPrompt(original.id, 17)

        expect(result).toEqual(
          expect.objectContaining({
            success: false,
            syncStatus: original.syncStatus
          })
        )
        expect(result.error).not.toContain(sentinel)
      } finally {
        warn.mockRestore()
        error.mockRestore()
      }

      expect(state.prompts.get(original.id)).toEqual(original)
      expect(mocks.promptUpdate).not.toHaveBeenCalled()
      expect(captured.join("\n")).not.toContain(sentinel)
    }
  )

  it.each(["create", "update"] as const)(
    "retains pending-state behavior for transient $operation failures",
    async (operation) => {
      const original = {
        id: `transient-auto-${operation}`,
        title: "Transient prompt",
        name: "Transient prompt",
        content: "Safe local snapshot",
        is_system: false,
        user_prompt: "Safe local snapshot",
        createdAt: 1,
        updatedAt: 25,
        studioProjectId: 17,
        syncStatus:
          operation === "create" ? ("local" as const) : ("synced" as const),
        ...(operation === "update"
          ? { serverId: 702, studioPromptId: 702 }
          : {})
      }
      state.prompts.set(original.id, structuredClone(original))
      if (operation === "create") {
        mocks.createPrompt.mockRejectedValueOnce(new Error("offline"))
      } else {
        mocks.updatePrompt.mockRejectedValueOnce(new Error("offline"))
      }

      const { autoSyncPrompt } = await importPromptSync()
      const result = await autoSyncPrompt(original.id, 17)

      expect(result).toEqual(
        expect.objectContaining({ success: false, syncStatus: "pending" })
      )
      expect(state.prompts.get(original.id)).toEqual(
        expect.objectContaining({
          syncStatus: "pending",
          studioProjectId: 17,
          updatedAt: expect.any(Number)
        })
      )
      expect(mocks.promptUpdate).toHaveBeenCalledTimes(1)
    }
  )

  it("returns conflict details with both local and server prompts", async () => {
    state.prompts.set("local-conflict-info", {
      id: "local-conflict-info",
      title: "Local Prompt",
      name: "Local Prompt",
      content: "local",
      user_prompt: "local",
      createdAt: 1,
      updatedAt: 10,
      serverId: 321,
      syncStatus: "conflict"
    })
    mocks.getPrompt.mockResolvedValue({
      data: {
        data: {
          id: 321,
          project_id: 9,
          name: "Server Prompt",
          system_prompt: "server system",
          user_prompt: "server user",
          version_number: 2,
          updated_at: "2026-02-17T10:00:00Z"
        }
      }
    })

    const { getConflictInfo } = await importPromptSync()
    const info = await getConflictInfo("local-conflict-info")

    expect(info).toEqual(
      expect.objectContaining({
        localPrompt: expect.objectContaining({ id: "local-conflict-info" }),
        serverPrompt: expect.objectContaining({ id: 321 })
      })
    )
  })

  it("resolveConflict keep_local pushes the local version to server", async () => {
    state.prompts.set("local-keep-local", {
      id: "local-keep-local",
      title: "Prompt Local",
      name: "Prompt Local",
      content: "local content",
      system_prompt: "local system",
      user_prompt: "local user",
      createdAt: 1,
      updatedAt: 10,
      serverId: 77,
      studioProjectId: 17,
      syncStatus: "conflict"
    })
    mocks.updatePrompt.mockResolvedValue({
      data: {
        data: {
          id: 77,
          project_id: 17,
          name: "Prompt Local",
          system_prompt: "local system",
          user_prompt: "local user",
          version_number: 3,
          updated_at: "2026-02-17T10:05:00Z"
        }
      }
    })

    const { resolveConflict } = await importPromptSync()
    const result = await resolveConflict("local-keep-local", "keep_local")

    expect(result.success).toBe(true)
    expect(result.syncStatus).toBe("synced")
    expect(mocks.updatePrompt).toHaveBeenCalledWith(
      77,
      expect.objectContaining({
        name: "Prompt Local"
      })
    )
    expect(state.prompts.get("local-keep-local")).toEqual(
      expect.objectContaining({
        serverId: 77,
        syncStatus: "synced"
      })
    )
  })

  it("resolveConflict keep_server pulls server version into local", async () => {
    state.prompts.set("local-keep-server", {
      id: "local-keep-server",
      title: "Local stale",
      name: "Local stale",
      content: "stale",
      user_prompt: "stale",
      createdAt: 1,
      updatedAt: 11,
      serverId: 88,
      syncStatus: "conflict"
    })
    mocks.getPrompt.mockResolvedValue({
      data: {
        data: {
          id: 88,
          project_id: 17,
          name: "Server fresh",
          system_prompt: "fresh system",
          user_prompt: "fresh user",
          version_number: 4,
          updated_at: "2026-02-17T10:10:00Z"
        }
      }
    })

    const { resolveConflict } = await importPromptSync()
    const result = await resolveConflict("local-keep-server", "keep_server")

    expect(result.success).toBe(true)
    expect(result.syncStatus).toBe("synced")
    expect(state.prompts.get("local-keep-server")).toEqual(
      expect.objectContaining({
        name: "Server fresh",
        system_prompt: "fresh system",
        user_prompt: "fresh user",
        syncStatus: "synced"
      })
    )
  })

  it("resolveConflict keep_both creates the server copy before one atomic local update", async () => {
    state.prompts.set("local-keep-both", {
      id: "local-keep-both",
      title: "Prompt Keep Both",
      name: "Prompt Keep Both",
      content: "keep both content",
      user_prompt: "keep both user",
      createdAt: 1,
      updatedAt: 12,
      serverId: 99,
      studioProjectId: 17,
      syncStatus: "conflict"
    })
    mocks.createPrompt.mockResolvedValue({
      data: {
        data: {
          id: 199,
          project_id: 17,
          name: "Prompt Keep Both",
          system_prompt: null,
          user_prompt: "keep both user",
          version_number: 1,
          updated_at: "2026-02-17T10:15:00Z"
        }
      }
    })

    const { resolveConflict } = await importPromptSync()
    const result = await resolveConflict("local-keep-both", "keep_both")

    expect(result.success).toBe(true)
    expect(result.syncStatus).toBe("synced")
    expect(mocks.promptUpdate).toHaveBeenCalledTimes(1)
    expect(mocks.promptUpdate).not.toHaveBeenCalledWith(
      "local-keep-both",
      expect.objectContaining({ serverId: null })
    )
    expect(mocks.createPrompt).toHaveBeenCalledWith(
      expect.objectContaining({
        project_id: 17,
        name: "Prompt Keep Both"
      })
    )
    expect(state.prompts.get("local-keep-both")).toEqual(
      expect.objectContaining({
        serverId: 199,
        syncStatus: "synced"
      })
    )
  })

  it.each([
    ["missing", undefined],
    ["null", null],
    ["zero", 0],
    ["negative", -1],
    ["fractional", 1.5],
    ["non-finite", Number.NaN],
    ["string", "17"]
  ])(
    "resolveConflict keep_both fails without mutation for an %s project id",
    async (_name, projectId) => {
      const original: Record<string, unknown> = {
        id: `keep-both-invalid-project-${_name}`,
        title: "Prompt Keep Both",
        name: "Prompt Keep Both",
        content: "keep both content",
        user_prompt: "keep both user",
        createdAt: 1,
        updatedAt: 12,
        serverId: 99,
        studioPromptId: 99,
        syncStatus: "conflict"
      }
      if (projectId !== undefined) original.studioProjectId = projectId
      state.prompts.set(String(original.id), structuredClone(original))

      const { resolveConflict } = await importPromptSync()
      const result = await resolveConflict(String(original.id), "keep_both")

      expect(result).toEqual(
        expect.objectContaining({
          success: false,
          syncStatus: "conflict"
        })
      )
      expect(state.prompts.get(String(original.id))).toEqual(original)
      expect(mocks.createPrompt).not.toHaveBeenCalled()
      expect(mocks.promptUpdate).not.toHaveBeenCalled()
    }
  )

  it("resolveConflict keep_both preserves recipe identity in the new server copy", async () => {
    const definition = makeRecipeDefinition()
    state.prompts.set("recipe-keep-both", {
      id: "recipe-keep-both",
      title: "Recipe Keep Both",
      name: "Recipe Keep Both",
      content: "Explain the task.",
      is_system: false,
      system_prompt: "",
      user_prompt: "Explain the task.",
      promptFormat: "structured",
      promptSchemaVersion: 2,
      structuredPromptDefinition: definition,
      createdAt: 1,
      updatedAt: 12,
      serverId: 299,
      studioProjectId: 17,
      syncStatus: "conflict"
    })
    mocks.createPrompt.mockResolvedValue({
      data: {
        data: {
          id: 399,
          project_id: 17,
          name: "Recipe Keep Both",
          system_prompt: "",
          user_prompt: "Explain the task.",
          prompt_format: "structured",
          prompt_schema_version: 2,
          prompt_definition: definition,
          version_number: 1,
          updated_at: "2026-02-17T10:15:00Z"
        }
      }
    })

    const { resolveConflict } = await importPromptSync()
    const result = await resolveConflict("recipe-keep-both", "keep_both")

    expect(result).toEqual(
      expect.objectContaining({
        success: true,
        serverId: 399,
        syncStatus: "synced"
      })
    )
    expect(mocks.createPrompt).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt_format: "structured",
        prompt_schema_version: 2,
        prompt_definition: definition
      })
    )
    expect(state.prompts.get("recipe-keep-both")).toEqual(
      expect.objectContaining({
        serverId: 399,
        promptSchemaVersion: 2,
        structuredPromptDefinition: definition,
        syncPayloadVersion: 1
      })
    )
    expect(mocks.promptUpdate).toHaveBeenCalledTimes(1)
    expect(JSON.stringify(mocks.createPrompt.mock.calls[0][0])).not.toMatch(
      /runtime_values|variable_values|resolved_values/
    )
  })

  it.each(["validation", "network", "empty response", "malicious response"])(
    "resolveConflict keep_both is failure-atomic on %s failure",
    async (failure) => {
      const definition = makeRecipeDefinition()
      if (failure === "validation") {
        definition.schema_version = 1 as 2
      }
      const original = {
        id: `keep-both-${failure}`,
        title: "Atomic Recipe",
        name: "Atomic Recipe",
        content: "Explain the task.",
        is_system: false,
        system_prompt: "",
        user_prompt: "Explain the task.",
        promptFormat: "structured" as const,
        promptSchemaVersion: 2,
        structuredPromptDefinition: definition,
        createdAt: 1,
        updatedAt: 12,
        serverId: 499,
        studioProjectId: 17,
        studioPromptId: 499,
        sourceSystem: "studio",
        syncStatus: "conflict" as const,
        lastSyncedAt: 11,
        serverUpdatedAt: "2026-02-17T10:00:00Z"
      }
      state.prompts.set(original.id, structuredClone(original))
      const before = JSON.stringify(state.prompts.get(original.id))

      if (failure === "network") {
        mocks.createPrompt.mockRejectedValueOnce(new Error("network failed"))
      } else if (failure === "empty response") {
        mocks.createPrompt.mockResolvedValueOnce({ data: { data: null } })
      } else if (failure === "malicious response") {
        mocks.createPrompt.mockResolvedValueOnce({
          data: {
            data: {
              id: 599,
              project_id: 17,
              name: "Atomic Recipe",
              system_prompt: "",
              user_prompt: "PRIVATE_SERVER_VALUE",
              prompt_format: "structured",
              prompt_schema_version: 2,
              prompt_definition: {
                ...makeRecipeDefinition(),
                resolved_values: { topic: "PRIVATE_SERVER_VALUE" }
              },
              version_number: 1,
              updated_at: "2026-02-17T10:15:00Z"
            }
          }
        })
      }

      const { resolveConflict } = await importPromptSync()
      const result = await resolveConflict(original.id, "keep_both")

      expect(result.success).toBe(false)
      expect(JSON.stringify(state.prompts.get(original.id))).toBe(before)
      expect(mocks.promptUpdate).not.toHaveBeenCalled()
      if (failure === "validation") {
        expect(mocks.createPrompt).not.toHaveBeenCalled()
      }
    }
  )

  it("getSyncStatus does not flag metadata-only server updates as conflicts", async () => {
    state.prompts.set("status-metadata-only", {
      id: "status-metadata-only",
      title: "Prompt",
      name: "Prompt",
      content: "same user text",
      user_prompt: "same user text",
      is_system: false,
      createdAt: 1,
      updatedAt: 50,
      lastSyncedAt: 10,
      serverUpdatedAt: "2026-02-17T10:00:00Z",
      serverId: 701,
      syncStatus: "pending"
    })
    mocks.getPrompt.mockResolvedValue({
      data: {
        data: {
          id: 701,
          project_id: 17,
          name: "Prompt",
          system_prompt: null,
          user_prompt: "same user text",
          version_number: 3,
          updated_at: "2026-02-17T11:00:00Z"
        }
      }
    })

    const { getSyncStatus } = await importPromptSync()
    const status = await getSyncStatus("status-metadata-only")

    expect(status.hasConflict).toBe(false)
    expect(status.status).toBe("pending")
  })

  it("getSyncStatus flags conflict when both timestamp and content differ", async () => {
    state.prompts.set("status-content-conflict", {
      id: "status-content-conflict",
      title: "Prompt",
      name: "Prompt",
      content: "local user text",
      user_prompt: "local user text",
      is_system: false,
      createdAt: 1,
      updatedAt: 60,
      lastSyncedAt: 10,
      serverUpdatedAt: "2026-02-17T10:00:00Z",
      serverId: 702,
      syncStatus: "pending"
    })
    mocks.getPrompt.mockResolvedValue({
      data: {
        data: {
          id: 702,
          project_id: 17,
          name: "Prompt",
          system_prompt: null,
          user_prompt: "server user text",
          version_number: 4,
          updated_at: "2026-02-17T11:05:00Z"
        }
      }
    })

    const { getSyncStatus } = await importPromptSync()
    const status = await getSyncStatus("status-content-conflict")

    expect(status.hasConflict).toBe(true)
    expect(status.status).toBe("conflict")
  })

  it("getSyncStatus supports legacy content-only local prompts", async () => {
    state.prompts.set("status-legacy-content", {
      id: "status-legacy-content",
      title: "Legacy Prompt",
      name: "Legacy Prompt",
      content: "legacy system text",
      is_system: true,
      createdAt: 1,
      updatedAt: 70,
      lastSyncedAt: 10,
      serverUpdatedAt: "2026-02-17T10:00:00Z",
      serverId: 703,
      syncStatus: "pending"
    })
    mocks.getPrompt.mockResolvedValue({
      data: {
        data: {
          id: 703,
          project_id: 17,
          name: "Legacy Prompt",
          system_prompt: "legacy system text",
          user_prompt: null,
          version_number: 2,
          updated_at: "2026-02-17T11:10:00Z"
        }
      }
    })

    const { getSyncStatus } = await importPromptSync()
    const status = await getSyncStatus("status-legacy-content")

    expect(status.hasConflict).toBe(false)
    expect(status.status).toBe("pending")
  })
})
