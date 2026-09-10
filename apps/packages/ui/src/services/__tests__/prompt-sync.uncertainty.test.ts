import { beforeEach, describe, expect, it, vi } from "vitest"
import { buildChatSurfaceScopeKeyFromConfig } from "../chat-surface-scope"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  create: vi.fn(),
  update: vi.fn(),
  get: vi.fn(),
  rows: new Map<string, Record<string, unknown>>()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getConfig: mocks.getConfig }
}))
vi.mock("@/services/prompt-studio", () => ({
  createPrompt: mocks.create,
  updatePrompt: mocks.update,
  getPrompt: mocks.get,
  listProjects: async () => ({ data: { data: [{ id: 42 }] } }),
  createProject: vi.fn()
}))
vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: async () => ({
    defaultProjectId: 42,
    autoSyncWorkspacePrompts: true
  }),
  setPromptStudioDefaults: vi.fn()
}))
vi.mock("@/db/dexie/helpers", () => ({ generateID: () => "generated-id" }))
vi.mock("@/db/dexie/chat", () => ({ PageAssistDatabase: class {} }))
vi.mock("@/db/dexie/schema", () => ({
  db: {
    prompts: {
      get: async (id: string) => mocks.rows.get(id),
      update: async (id: string, fields: object) => {
        mocks.rows.set(id, { ...mocks.rows.get(id), ...fields })
      },
      add: async (row: { id: string }) => mocks.rows.set(row.id, row),
      where: (field: string) => ({
        equals: (value: unknown) => ({
          first: async () =>
            [...mocks.rows.values()].find((row) => row[field] === value)
        })
      })
    }
  }
}))

const config = (serverUrl: string, sub: string, exp = 1) => ({
  serverUrl,
  authMode: "multi-user" as const,
  accessToken: `header.${btoa(JSON.stringify({ sub, exp }))}.signature`
})
const owner = config("https://a.test", "alice")
const ownerScope = buildChatSurfaceScopeKeyFromConfig(owner)
const otherConnections = [
  ["backend", config("https://b.test", "alice")],
  ["principal", config("https://a.test", "bob")]
] as const
const response = {
  data: {
    data: {
      id: 101,
      project_id: 42,
      name: "Reconciled",
      system_prompt: "",
      user_prompt: "Task",
      prompt_format: "legacy",
      prompt_schema_version: null,
      prompt_definition: null,
      version_number: 2,
      updated_at: "2026-09-10T00:00:00Z"
    }
  }
}

const transportResponse = async () => {
  const dispatchConfig = await mocks.getConfig().catch(() => null)
  return {
    ...response,
    persistenceScope: dispatchConfig
      ? buildChatSurfaceScopeKeyFromConfig(dispatchConfig)
      : null
  }
}

describe("authoritative prompt sync uncertainty cleanup", () => {
  beforeEach(() => {
    vi.resetModules()
    vi.clearAllMocks()
    mocks.rows.clear()
    mocks.getConfig.mockResolvedValue(owner)
    mocks.create.mockImplementation(transportResponse)
    mocks.update.mockImplementation(transportResponse)
    mocks.get.mockImplementation(transportResponse)
  })

  it.each([
    "manual create",
    "manual update",
    "auto create",
    "auto update",
    "pull exact",
    "pull by server ID",
    "pull new"
  ])(
    "%s clears only its matching backend/principal and exact local ID",
    async (operation) => {
      const registry = await import("../recipe-persistence-uncertainty")
      const sync = await import("../prompt-sync")
      for (const [, other] of otherConnections) {
        const otherScope = buildChatSurfaceScopeKeyFromConfig(other)
        const id = operation === "pull new" ? "generated-id" : "same-id"
        mocks.rows.clear()
        if (operation !== "pull new")
          mocks.rows.set(id, {
            id,
            title: "Task",
            name: "Task",
            content: "Task",
            is_system: false,
            createdAt: 1,
            syncStatus: "local",
            ...(operation.includes("update") ||
            operation === "pull by server ID"
              ? { serverId: 101 }
              : {})
          })
        registry.markRecipePersistenceUncertain(id, ownerScope)
        registry.markRecipePersistenceUncertain(id, otherScope)
        registry.markRecipePersistenceUncertain("unrelated-id", otherScope)
        mocks.getConfig.mockResolvedValue(other)
        const run = () =>
          operation.startsWith("manual")
            ? sync.pushToStudio(id, 42)
            : operation.startsWith("auto")
              ? sync.autoSyncPrompt(id, 42)
              : sync.pullFromStudio(
                  101,
                  operation === "pull exact" ? id : undefined
                )
        expect(await run()).toMatchObject({
          success: true,
          persistenceScope: otherScope
        })
        expect(registry.isRecipePersistenceUncertain(id, ownerScope)).toBe(true)
        expect(registry.isRecipePersistenceUncertain(id, otherScope)).toBe(
          false
        )
        expect(
          registry.isRecipePersistenceUncertain("unrelated-id", otherScope)
        ).toBe(true)

        // Credential refresh belongs to A, not a new uncertainty namespace.
        mocks.getConfig.mockResolvedValue(config("https://a.test", "alice", 2))
        expect(await run()).toMatchObject({
          success: true,
          persistenceScope: ownerScope
        })
        expect(registry.isRecipePersistenceUncertain(id, ownerScope)).toBe(
          false
        )
      }
    }
  )

  it("does not clear any owned marker when the transport cannot report its scope", async () => {
    const registry = await import("../recipe-persistence-uncertainty")
    const { pullFromStudio } = await import("../prompt-sync")
    mocks.rows.set("same-id", { id: "same-id" })
    registry.markRecipePersistenceUncertain("same-id", ownerScope)
    mocks.getConfig.mockRejectedValue(new Error("config unavailable"))
    expect(await pullFromStudio(101, "same-id")).toMatchObject({
      success: true,
      persistenceScope: null
    })
    expect(registry.isRecipePersistenceUncertain("same-id", ownerScope)).toBe(
      true
    )
  })

  it("clears the dispatch owner, not the connection selected while the response is pending", async () => {
    const registry = await import("../recipe-persistence-uncertainty")
    const { pullFromStudio } = await import("../prompt-sync")
    const other = otherConnections[0][1]
    const otherScope = buildChatSurfaceScopeKeyFromConfig(other)
    mocks.rows.set("same-id", { id: "same-id" })
    registry.markRecipePersistenceUncertain("same-id", ownerScope)
    registry.markRecipePersistenceUncertain("same-id", otherScope)
    mocks.get.mockImplementationOnce(async () => {
      const capturedResponse = await transportResponse()
      mocks.getConfig.mockResolvedValue(other)
      return capturedResponse
    })
    expect(await pullFromStudio(101, "same-id")).toMatchObject({
      success: true,
      persistenceScope: ownerScope
    })
    expect(registry.isRecipePersistenceUncertain("same-id", ownerScope)).toBe(
      false
    )
    expect(registry.isRecipePersistenceUncertain("same-id", otherScope)).toBe(
      true
    )
  })
})
