import { CLEAR_TASK_RECIPE } from "@/components/Common/PromptAssist/recipes/built-in-recipes"
import { beforeEach, describe, expect, it, vi } from "vitest"

import * as sync from "../prompt-sync"
import * as registry from "../recipe-persistence-uncertainty"

const ownerA = "recipe-owner:sha256:" + "a".repeat(64)
const ownerB = "recipe-owner:sha256:" + "b".repeat(64)
const mocks = vi.hoisted(() => ({
  defaults: vi.fn(),
  projects: vi.fn(),
  createProject: vi.fn(),
  create: vi.fn(),
  update: vi.fn(),
  get: vi.fn(),
  reconcile: vi.fn(),
  rows: new Map<string, Record<string, unknown>>()
}))
vi.mock("@/services/prompt-studio", () => ({
  createPrompt: (...args: unknown[]) => mocks.create(...args),
  updatePrompt: (...args: unknown[]) => mocks.update(...args),
  getPrompt: (...args: unknown[]) => mocks.get(...args),
  listProjects: (...args: unknown[]) => mocks.projects(...args),
  createProject: (...args: unknown[]) => mocks.createProject(...args)
}))
vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: (...args: unknown[]) => mocks.defaults(...args),
  setPromptStudioDefaults: vi.fn()
}))
vi.mock("@/db/dexie/helpers", () => ({ generateID: () => "new-id" }))
vi.mock("@/db/dexie/chat", () => ({ PageAssistDatabase: class {} }))
vi.mock("@/db/dexie/schema", () => ({
  db: {
    prompts: {
      get: async (id: string) => mocks.rows.get(id),
      update: async (
        id: string,
        fields: object | ((row: Record<string, unknown>) => void | boolean)
      ) => {
        await mocks.reconcile()
        if (!mocks.rows.has(id)) return 0
        const row = { ...mocks.rows.get(id) }
        if (typeof fields === "function") {
          if (fields(row) === false) return 1
        } else Object.assign(row, fields)
        mocks.rows.set(id, row)
        return 1
      },
      add: async (row: { id: string }) => {
        await mocks.reconcile()
        mocks.rows.set(row.id, row)
      },
      where: (field: string) => ({
        equals: (value: unknown) => ({
          first: async () =>
            [...mocks.rows.values()].find((row) => row[field] === value)
        })
      })
    }
  }
}))

const server = {
  id: 101,
  project_id: 42,
  name: "Reconciled",
  version_number: 2,
  updated_at: "2026-09-10T00:00:00Z",
  prompt_format: "structured",
  prompt_schema_version: 2,
  prompt_definition: CLEAR_TASK_RECIPE.definition
}
const response = () => ({
  ok: true,
  status: 200,
  data: { success: true, data: server },
  recipePersistence: { state: "dispatched", actualOwnerId: ownerB }
})
const seed = (linked = false) =>
  mocks.rows.set("exact-id", {
    id: "exact-id",
    title: "Recipe",
    syncStatus: linked ? "conflict" : "local",
    promptFormat: "structured",
    promptSchemaVersion: 2,
    structuredPromptDefinition: CLEAR_TASK_RECIPE.definition,
    studioProjectId: 42,
    ...(linked ? { serverId: 101 } : {})
  })

describe("owner-aware sync and exact reconciliation", () => {
  it("linking cannot replace a concurrent durable error with pending or new references", async () => {
    seed(true)
    let release!: () => void
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    mocks.get.mockImplementationOnce(async () => {
      await gate
      return response()
    })
    const linking = sync.linkPrompts("exact-id", 101)
    await vi.waitFor(() => expect(mocks.get).toHaveBeenCalled())
    mocks.update.mockImplementationOnce(async () => {
      await registry.markRecipePersistenceScoped("exact-id", ownerB)
      return { ...response(), ok: false, status: 0, data: undefined }
    })
    await sync.pushToStudio("exact-id", 42, { expectedOwnerId: ownerB })
    const locked = structuredClone(mocks.rows.get("exact-id"))
    expect(locked.syncStatus).toBe("error")
    release()
    expect.soft(await linking).toMatchObject({
      success: false,
      syncStatus: "error",
      recipeWriteBlocked: true,
      failureKind: "validation"
    })
    expect(mocks.rows.get("exact-id")).toEqual(locked)
  })

  it.each(["manual", "auto"])(
    "%s refuses a durable error after the registry restarts clear",
    async (mode) => {
      seed(true)
      mocks.rows.get("exact-id").syncStatus = "error"
      const result =
        mode === "manual"
          ? await sync.pushToStudio("exact-id", 42, { expectedOwnerId: ownerB })
          : await sync.autoSyncPrompt("exact-id", 42, {
              expectedOwnerId: ownerB
            })
      expect(result).toMatchObject({
        success: false,
        syncStatus: "error",
        recipeOwnership: { dispatch: { state: "not_dispatched" } }
      })
      expect(mocks.update).not.toHaveBeenCalled()
      expect(mocks.rows.get("exact-id").syncStatus).toBe("error")
    }
  )

  it("keeps v2 local-pending without remotely discovering a project", async () => {
    seed()
    mocks.rows.get("exact-id").studioProjectId = null
    mocks.defaults.mockResolvedValue({ defaultProjectId: null })
    const result = await sync.autoSyncPrompt("exact-id", undefined, {
      expectedOwnerId: ownerB
    })
    expect(result).toMatchObject({
      success: false,
      syncStatus: "pending",
      recipeOwnership: { dispatch: { state: "not_dispatched" } }
    })
    expect(mocks.create).not.toHaveBeenCalled()
    expect(mocks.projects).not.toHaveBeenCalled()
    expect(mocks.createProject).not.toHaveBeenCalled()
    expect(mocks.rows.get("exact-id").studioProjectId).toBeNull()
  })

  it.each([undefined, 1, 99])(
    "rejects inconsistent local v2 outer version %s before project or prompt requests",
    async (outerVersion) => {
      seed()
      mocks.rows.get("exact-id").promptSchemaVersion = outerVersion
      mocks.rows.get("exact-id").studioProjectId = null
      const result = await sync.autoSyncPrompt("exact-id", undefined, {
        expectedOwnerId: ownerB
      })
      expect(result).toMatchObject({
        success: false,
        failureKind: "validation",
        recipeOwnership: { dispatch: { state: "not_dispatched" } }
      })
      expect(mocks.defaults).not.toHaveBeenCalled()
      expect(mocks.projects).not.toHaveBeenCalled()
      expect(mocks.createProject).not.toHaveBeenCalled()
      expect(mocks.create).not.toHaveBeenCalled()
    }
  )

  it.each(["exact", "by server ID"])(
    "pull %s retains scoped state if the local row disappears during reconciliation",
    async (mode) => {
      seed(true)
      await registry.markRecipePersistenceScoped("exact-id", ownerB)
      mocks.reconcile.mockImplementationOnce(() =>
        mocks.rows.delete("exact-id")
      )
      expect(
        await sync.pullFromStudio(
          101,
          mode === "exact" ? "exact-id" : undefined
        )
      ).toMatchObject({ success: false })
      expect(
        await registry.readRecipePersistenceUncertainty("exact-id", ownerB)
      ).toBe("scoped")
    }
  )
  it("blocks unresolved auto-sync before project discovery and leaves no new remote attempt", async () => {
    seed()
    mocks.rows.get("exact-id").studioProjectId = null
    await registry.markRecipePersistenceScoped("exact-id", ownerB)
    expect(
      await sync.autoSyncPrompt("exact-id", undefined, {
        expectedOwnerId: ownerB
      })
    ).toMatchObject({ success: false })
    expect(mocks.defaults).not.toHaveBeenCalled()
    expect(mocks.create).not.toHaveBeenCalled()
  })
  beforeEach(async () => {
    vi.clearAllMocks()
    mocks.rows.clear()
    mocks.defaults.mockResolvedValue({
      defaultProjectId: 42,
      autoSyncWorkspacePrompts: true
    })
    mocks.projects.mockResolvedValue({ data: { data: [{ id: 42 }] } })
    mocks.reconcile.mockResolvedValue(undefined)
    for (const id of ["exact-id", "other-id", "new-id"]) {
      await registry.forgetRecipePersistenceUnknown(id)
      for (const owner of [ownerA, ownerB])
        await registry.clearRecipePersistenceScoped(id, owner)
    }
    mocks.create.mockImplementation(async () => response())
    mocks.update.mockImplementation(async () => response())
    mocks.get.mockImplementation(async () => response())
  })

  it.each([
    [{ state: "not_dispatched", actualOwnerId: null }, "known_rejection"],
    [{ state: "dispatched", actualOwnerId: ownerB }, "scoped_uncertain"],
    [{ state: "dispatched", actualOwnerId: null }, "unknown_owner"],
    [{ state: "unknown", actualOwnerId: null }, "unknown_owner"]
  ] as const)("classifies %j as %s", (dispatch, expected) => {
    expect(sync.classifyRecipeDispatch(dispatch)).toBe(expected)
  })

  it.each([
    "manual create",
    "manual update",
    "auto create",
    "auto update",
    "keep_local",
    "keep_both",
    "keep_server",
    "pull exact",
    "pull by ID",
    "pull new"
  ])("%s clears only the reconciled owner and ID", async (operation) => {
    const id = operation === "pull new" ? "new-id" : "exact-id"
    const isPull = operation.startsWith("pull") || operation === "keep_server"
    if (operation !== "pull new") seed(!operation.includes("create"))
    if (isPull) await registry.markRecipePersistenceScoped(id, ownerB)
    else {
      await registry.markRecipePersistenceScoped(id, ownerA)
      const dispatch = async () => {
        await registry.markRecipePersistenceScoped(id, ownerB)
        return response()
      }
      mocks.create.mockImplementation(dispatch)
      mocks.update.mockImplementation(dispatch)
    }
    await registry.markRecipePersistenceScoped("other-id", ownerB)
    const input = { expectedOwnerId: ownerB }
    const result = operation.startsWith("manual")
      ? await sync.pushToStudio(id, 42, input)
      : operation.startsWith("auto")
        ? await sync.autoSyncPrompt(id, 42, input)
        : operation.startsWith("keep_")
          ? await sync.resolveConflict(
              id,
              operation as sync.ConflictResolution,
              input
            )
          : await sync.pullFromStudio(
              101,
              operation === "pull exact" ? id : undefined
            )
    expect(result).toMatchObject({
      success: true,
      localId: id,
      recipeOwnership: {
        localId: id,
        dispatch: { state: "dispatched", actualOwnerId: ownerB }
      }
    })
    expect(await registry.readRecipePersistenceUncertainty(id, ownerA)).toBe(
      isPull ? "clear" : "scoped"
    )
    expect(await registry.readRecipePersistenceUncertainty(id, ownerB)).toBe(
      "clear"
    )
    expect(
      await registry.readRecipePersistenceUncertainty("other-id", ownerB)
    ).toBe("scoped")
  })

  it.each(["create", "update"])(
    "threads exact owner and ID through %s and refuses missing v2 ownership",
    async (operation) => {
      seed(operation === "update")
      const transport = operation === "create" ? mocks.create : mocks.update
      expect(await sync.pushToStudio("exact-id", 42)).toMatchObject({
        success: false,
        failureKind: "validation",
        recipeOwnership: {
          dispatch: { state: "not_dispatched", actualOwnerId: null }
        }
      })
      expect(transport).not.toHaveBeenCalled()
      await sync.autoSyncPrompt("exact-id", 42, { expectedOwnerId: ownerB })
      expect(transport.mock.calls[0].at(-1)).toEqual({
        recipePersistence: {
          mode: "require",
          expectedOwnerId: ownerB,
          localId: "exact-id"
        }
      })
    }
  )

  it.each([
    [
      "malformed 2xx",
      {
        ok: true,
        status: 200,
        data: { data: { ...server, prompt_schema_version: 999 } }
      }
    ],
    ["connection loss", { ok: false, status: 0, error: "connection lost" }],
    ["timeout", { ok: false, status: 0, error: "Timeout" }],
    [
      "unclassified 5xx",
      { ok: false, status: 500, data: { detail: "Internal error" } }
    ],
    ["untyped 409", { ok: false, status: 409 }]
  ])(
    "retains scoped uncertainty after %s without making the row retryable pending",
    async (_label, failure) => {
      seed()
      mocks.create.mockImplementation(async () => {
        await registry.markRecipePersistenceScoped("exact-id", ownerB)
        return { ...response(), ...failure }
      })
      const result = await sync.autoSyncPrompt("exact-id", 42, {
        expectedOwnerId: ownerB
      })
      expect(result.success).toBe(false)
      expect(result.recipeOwnership?.dispatch).toEqual({
        state: "dispatched",
        actualOwnerId: ownerB
      })
      expect(
        await registry.readRecipePersistenceUncertainty("exact-id", ownerB)
      ).toBe("scoped")
      expect(mocks.rows.get("exact-id").syncStatus).toBe("error")
    }
  )

  it.each([
    [
      422,
      {
        detail: [
          { loc: ["body", "name"], msg: "Field required", type: "missing" }
        ]
      }
    ],
    [403, { detail: "Access denied to this prompt" }],
    [409, { detail: "Prompt with this name already exists in the project" }]
  ])(
    "clears a typed no-mutation %s rejection before rollback",
    async (status, data) => {
      seed()
      mocks.create.mockImplementation(async () => {
        await registry.markRecipePersistenceScoped("exact-id", ownerB)
        return { ...response(), ok: false, status, data }
      })
      expect(
        await sync.pushToStudio("exact-id", 42, { expectedOwnerId: ownerB })
      ).toMatchObject({ success: false, failureKind: "validation" })
      expect(
        await registry.readRecipePersistenceUncertainty("exact-id", ownerB)
      ).toBe("clear")
      expect(mocks.rows.get("exact-id").title).toBe("Recipe")
    }
  )

  it.each(["write fails", "row disappears"])(
    "retains the scoped marker when local reconciliation %s",
    async (mode) => {
      seed()
      mocks.create.mockImplementation(async () => {
        await registry.markRecipePersistenceScoped("exact-id", ownerB)
        if (mode === "row disappears") mocks.rows.delete("exact-id")
        else mocks.reconcile.mockRejectedValue(new Error("disk unavailable"))
        return response()
      })
      expect(
        await sync.pushToStudio("exact-id", 42, { expectedOwnerId: ownerB })
      ).toMatchObject({ success: false })
      expect(
        await registry.readRecipePersistenceUncertainty("exact-id", ownerB)
      ).toBe("scoped")
    }
  )

  it.each(["scoped", "unknown_owner"])(
    "blocks repeated v2 writes with %s state before dispatch",
    async (state) => {
      seed()
      if (state === "scoped")
        await registry.markRecipePersistenceScoped("exact-id", ownerB)
      else await registry.markRecipePersistenceUnknown("exact-id")
      expect(
        await sync.autoSyncPrompt("exact-id", 42, { expectedOwnerId: ownerB })
      ).toMatchObject({
        success: false,
        recipeOwnership: {
          dispatch: { state: "not_dispatched", actualOwnerId: null }
        }
      })
      expect(mocks.create).not.toHaveBeenCalled()
    }
  )
  it("does not automatically retry after a lost connection even if the row had been pending", async () => {
    seed()
    mocks.rows.get("exact-id").syncStatus = "pending"
    mocks.create.mockImplementation(async () => {
      await registry.markRecipePersistenceScoped("exact-id", ownerB)
      return { ...response(), ok: false, status: 0, data: undefined }
    })
    await sync.autoSyncPrompt("exact-id", 42, { expectedOwnerId: ownerB })
    expect(mocks.rows.get("exact-id").syncStatus).toBe("error")
    await sync.autoSyncPrompt("exact-id", 42, { expectedOwnerId: ownerB })
    expect(mocks.create).toHaveBeenCalledTimes(1)
  })
  it("rejects unowned v2 auto sync before even resolving or creating a project", async () => {
    seed()
    mocks.rows.get("exact-id").studioProjectId = null
    expect(await sync.autoSyncPrompt("exact-id")).toMatchObject({
      success: false,
      failureKind: "validation"
    })
    expect(mocks.defaults).not.toHaveBeenCalled()
    expect(mocks.create).not.toHaveBeenCalled()
  })
  it("fails closed when the registry read is unavailable", async () => {
    seed()
    const read = vi
      .spyOn(registry, "readRecipePersistenceUncertainty")
      .mockRejectedValueOnce(new Error("background unavailable"))
    try {
      expect(
        await sync.pushToStudio("exact-id", 42, { expectedOwnerId: ownerB })
      ).toMatchObject({
        success: false,
        recipeOwnership: { dispatch: { state: "not_dispatched" } }
      })
      expect(mocks.create).not.toHaveBeenCalled()
    } finally {
      read.mockRestore()
    }
  })
  it.each([
    { ok: true, status: 200, data: { data: server } },
    {
      ok: false,
      status: 403,
      data: { detail: "Access denied to this prompt" }
    }
  ])(
    "unknown dispatch quarantines even an apparently conclusive body",
    async (body) => {
      seed()
      mocks.create.mockResolvedValue({
        ...body,
        recipePersistence: { state: "unknown", actualOwnerId: null }
      })
      expect(
        await sync.pushToStudio("exact-id", 42, { expectedOwnerId: ownerB })
      ).toMatchObject({ success: false, syncStatus: "error" })
      expect(
        await registry.readRecipePersistenceUncertainty("exact-id", ownerA)
      ).toBe("unknown_owner")
    }
  )
  it("pull with no reported owner cannot clear scoped or unknown markers", async () => {
    seed(true)
    mocks.rows.get("exact-id").syncStatus = "error"
    await registry.markRecipePersistenceScoped("exact-id", ownerA)
    await registry.markRecipePersistenceUnknown("exact-id")
    mocks.get.mockResolvedValue({
      ...response(),
      recipePersistence: { state: "dispatched", actualOwnerId: null }
    })
    expect(await sync.pullFromStudio(101, "exact-id")).toMatchObject({
      success: false,
      syncStatus: "error",
      recipeWriteBlocked: true
    })
    expect(mocks.rows.get("exact-id")).toMatchObject({
      name: "Reconciled",
      syncStatus: "error",
      serverId: 101,
      studioProjectId: 42
    })
    expect(
      await registry.readRecipePersistenceUncertainty("exact-id", ownerA)
    ).toBe("unknown_owner")
    await registry.forgetRecipePersistenceUnknown("exact-id")
    expect(
      await registry.readRecipePersistenceUncertainty("exact-id", ownerA)
    ).toBe("scoped")
  })

  it("a different pull owner cannot erase another owner's durable lock", async () => {
    seed(true)
    mocks.rows.get("exact-id").syncStatus = "error"
    await registry.markRecipePersistenceScoped("exact-id", ownerA)

    expect(await sync.pullFromStudio(101, "exact-id")).toMatchObject({
      success: false,
      syncStatus: "error",
      recipeWriteBlocked: true
    })
    expect(mocks.rows.get("exact-id")).toMatchObject({
      name: "Reconciled",
      syncStatus: "error",
      serverId: 101,
      studioProjectId: 42
    })
    expect(
      await registry.readRecipePersistenceUncertainty("exact-id", ownerA)
    ).toBe("scoped")
  })

  it("same-owner pull cannot erase a coexisting unknown quarantine", async () => {
    seed(true)
    mocks.rows.get("exact-id").syncStatus = "error"
    await registry.markRecipePersistenceScoped("exact-id", ownerB)
    await registry.markRecipePersistenceUnknown("exact-id")

    expect(await sync.pullFromStudio(101, "exact-id")).toMatchObject({
      success: false,
      syncStatus: "error",
      recipeWriteBlocked: true
    })
    expect(mocks.rows.get("exact-id").syncStatus).toBe("error")
    await registry.forgetRecipePersistenceUnknown("exact-id")
    expect(
      await registry.readRecipePersistenceUncertainty("exact-id", ownerB)
    ).toBe("scoped")
  })

  it("keep-server preserves the durable lock when exact reconciliation is blocked", async () => {
    seed(true)
    mocks.rows.get("exact-id").syncStatus = "error"
    await registry.markRecipePersistenceScoped("exact-id", ownerA)

    expect(
      await sync.resolveConflict("exact-id", "keep_server", {
        expectedOwnerId: ownerB
      })
    ).toMatchObject({
      success: false,
      syncStatus: "error",
      recipeWriteBlocked: true
    })
    expect(mocks.rows.get("exact-id")).toMatchObject({
      name: "Reconciled",
      syncStatus: "error",
      serverId: 101
    })
  })

  it("pull preserves copied content and durable error when reconciliation authority fails", async () => {
    seed(true)
    mocks.rows.get("exact-id").syncStatus = "error"
    const reconcile = vi
      .spyOn(registry, "reconcileRecipePersistenceExact")
      .mockRejectedValueOnce(new Error("background unavailable"))
    try {
      expect(await sync.pullFromStudio(101, "exact-id")).toMatchObject({
        success: false,
        syncStatus: "error",
        recipeWriteBlocked: true
      })
      expect(mocks.rows.get("exact-id")).toMatchObject({
        name: "Reconciled",
        syncStatus: "error",
        serverId: 101
      })
    } finally {
      reconcile.mockRestore()
    }
  })

  it("pull reports the retained durable error when its final local reconciliation write fails", async () => {
    seed(true)
    mocks.rows.get("exact-id").syncStatus = "error"
    await registry.markRecipePersistenceScoped("exact-id", ownerB)
    mocks.reconcile
      .mockResolvedValueOnce(undefined)
      .mockRejectedValueOnce(new Error("disk unavailable"))

    expect(await sync.pullFromStudio(101, "exact-id")).toMatchObject({
      success: false,
      syncStatus: "error",
      serverId: 101,
      recipeWriteBlocked: true
    })
    expect(mocks.rows.get("exact-id")).toMatchObject({
      name: "Reconciled",
      syncStatus: "error",
      serverId: 101
    })
  })

  it("a trustworthy pull reconciles a restarted durable-only lock", async () => {
    seed(true)
    mocks.rows.get("exact-id").syncStatus = "error"

    expect(await sync.pullFromStudio(101, "exact-id")).toMatchObject({
      success: true,
      syncStatus: "synced"
    })
    expect(mocks.rows.get("exact-id").syncStatus).toBe("synced")
  })

  it.each(["scoped", "unknown_owner"])(
    "unlink preserves a linked v2 row while %s uncertainty exists",
    async (state) => {
      seed(true)
      if (state === "scoped")
        await registry.markRecipePersistenceScoped("exact-id", ownerA)
      else await registry.markRecipePersistenceUnknown("exact-id")

      expect(await sync.unlinkPrompt("exact-id")).toMatchObject({
        success: false,
        syncStatus: "conflict",
        serverId: 101,
        recipeWriteBlocked: true
      })
      expect(mocks.rows.get("exact-id")).toMatchObject({
        syncStatus: "conflict",
        serverId: 101,
        studioProjectId: 42
      })
    }
  )

  it("unlink preserves a restarted durable-only v2 lock and linkage", async () => {
    seed(true)
    mocks.rows.get("exact-id").syncStatus = "error"

    expect(await sync.unlinkPrompt("exact-id")).toMatchObject({
      success: false,
      syncStatus: "error",
      serverId: 101,
      recipeWriteBlocked: true
    })
    expect(mocks.rows.get("exact-id")).toMatchObject({
      syncStatus: "error",
      serverId: 101,
      studioProjectId: 42
    })
  })

  it("unlink preserves v2 linkage when its authority is unavailable", async () => {
    seed(true)
    mocks.rows.get("exact-id").syncStatus = "synced"
    const begin = vi
      .spyOn(registry, "beginRecipePersistenceUnlink")
      .mockRejectedValueOnce(new Error("background unavailable"))
    try {
      expect(await sync.unlinkPrompt("exact-id")).toMatchObject({
        success: false,
        syncStatus: "synced",
        serverId: 101,
        recipeWriteBlocked: true
      })
      expect(mocks.rows.get("exact-id")).toMatchObject({
        syncStatus: "synced",
        serverId: 101,
        studioProjectId: 42
      })
    } finally {
      begin.mockRestore()
    }
  })

  it("a clean v2 unlink lease blocks a mutation until the linkage write finishes", async () => {
    seed(true)
    mocks.rows.get("exact-id").syncStatus = "synced"
    mocks.reconcile.mockImplementationOnce(() => {
      expect(() =>
        registry.directRecipeRequestAuthority.dispatchAuthority.markDispatched(
          "exact-id",
          ownerB
        )
      ).toThrow("Recipe has an unresolved operation")
    })

    expect(await sync.unlinkPrompt("exact-id")).toMatchObject({
      success: true,
      syncStatus: "local"
    })
    expect(mocks.rows.get("exact-id")).toMatchObject({
      syncStatus: "local",
      serverId: null,
      studioProjectId: null
    })
    expect(
      await registry.readRecipePersistenceUncertainty("exact-id", ownerB)
    ).toBe("clear")
  })

  it("retains v1 unlink behavior even from an error status", async () => {
    seed(true)
    Object.assign(mocks.rows.get("exact-id"), {
      syncStatus: "error",
      promptFormat: "legacy",
      promptSchemaVersion: null,
      structuredPromptDefinition: null
    })

    expect(await sync.unlinkPrompt("exact-id")).toMatchObject({
      success: true,
      syncStatus: "local"
    })
    expect(mocks.rows.get("exact-id")).toMatchObject({
      syncStatus: "local",
      serverId: null,
      studioProjectId: null
    })
  })
})
