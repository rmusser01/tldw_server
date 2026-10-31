import { beforeEach, expect, it, vi } from "vitest"
const state = vi.hoisted(() => ({ owner: "recipe-owner:sha256:" + "a".repeat(64), owners: [] as string[], rows: new Map<string, any>() }))
const mocks = vi.hoisted(() => ({ createPrompt: vi.fn(), updatePrompt: vi.fn(), getPrompt: vi.fn(), defaults: vi.fn(), listProjects: vi.fn() }))
vi.mock("@/services/recipe-persistence-uncertainty", async original => ({ ...await original(), resolveRecipePersistenceOwnerView: async () => ({ ownerId: state.owners.shift() ?? state.owner, authorizationRevision: "revision" }) }))
vi.mock("@/services/prompt-studio", () => ({ createPrompt: mocks.createPrompt, updatePrompt: mocks.updatePrompt, getPrompt: mocks.getPrompt, listProjects: mocks.listProjects, createProject: vi.fn() }))
vi.mock("@/services/prompt-studio-settings", () => ({ getPromptStudioDefaults: mocks.defaults, getPromptStudioDefaultsForOwner: mocks.defaults, setPromptStudioDefaults: vi.fn() }))
vi.mock("@/db/dexie/helpers", () => ({ generateID: () => "new-owned" }))
vi.mock("@/db/dexie/schema", () => ({ db: { prompts: {
  get: async (id: string) => state.rows.get(id),
  update: async (id: string, fields: any) => { const row = state.rows.get(id); if (!row) return 0; if (typeof fields === "function") { if (fields(row) === false) return 0 } else Object.assign(row, fields); return 1 },
  add: async (row: any) => state.rows.set(row.id, row),
  where: (field: string) => ({ equals: (value: any) => ({ first: async () => [...state.rows.values()].find(row => row[field] === value), filter: (fn: (row: any) => boolean) => ({ first: async () => [...state.rows.values()].find(row => row[field] === value && fn(row)), toArray: async () => [...state.rows.values()].filter(row => row[field] === value && fn(row)) }), toArray: async () => [...state.rows.values()].filter(row => row[field] === value) }) })
} } }))
import { pushToStudio, pullFromStudio, unlinkPrompt, autoSyncPrompt, getConflictInfo, getSyncStatus } from "@/services/prompt-sync"
const alice = "recipe-owner:sha256:" + "a".repeat(64), bob = "recipe-owner:sha256:" + "b".repeat(64)
const row = (id: string, ownerId?: string) => ({ id, ownerId, title: id, name: id, content: id, is_system: false, createdAt: 1, serverId: 1, studioProjectId: 1, syncStatus: "synced" })
it.each(["status", "conflict"])("captures the first owner before the optional %s compatibility read", async mode => {
  state.rows.set("alice", row("alice", alice))
  state.owners = [alice, bob]
  try { await (mode === "status" ? getSyncStatus("alice") : getConflictInfo("alice")) } catch { /* Ownership retirement may reject the read. */ }
  expect(mocks.getPrompt).not.toHaveBeenCalled()
})
beforeEach(() => { vi.clearAllMocks(); state.owner = alice; state.rows.clear(); state.rows.set("bob", row("bob", bob)); mocks.defaults.mockResolvedValue({ defaultProjectId: 1 }); mocks.getPrompt.mockResolvedValue({ ok: true, data: { data: { id: 1, project_id: 1, name: "Alice", user_prompt: "Alice owned", version_number: 1 } }, recipePersistence: { state: "dispatched", actualOwnerId: alice } }); mocks.updatePrompt.mockResolvedValue({ ok: true, data: { data: { id: 1, project_id: 1, name: "Changed", version_number: 1 } } }) })
it.each(["push", "auto", "unlink"])("refuses foreign %s without dispatch or modification", async action => {
  const original = structuredClone(state.rows.get("bob"))
  const result = action === "push" ? await pushToStudio("bob", 1, { expectedOwnerId: alice }) : action === "auto" ? await autoSyncPrompt("bob", 1, { expectedOwnerId: alice }) : await unlinkPrompt("bob")
  expect(result.success).toBe(false); expect(mocks.updatePrompt).not.toHaveBeenCalled(); expect(mocks.createPrompt).not.toHaveBeenCalled(); expect(state.rows.get("bob")).toEqual(original)
})
it("keeps foreign same-number links intact while pulling an owned canonical copy", async () => {
  const original = structuredClone(state.rows.get("bob"))
  const result = await pullFromStudio(1)
  expect(result.localId).toBe("new-owned"); expect(state.rows.get("new-owned").ownerId).toBe(alice); expect(state.rows.get("bob")).toEqual(original)
})
it("rejects a caller-provided foreign local target before pulling", async () => {
  const result = await pullFromStudio(1, "bob")
  expect(result.success).toBe(false); expect(mocks.getPrompt).not.toHaveBeenCalled()
})
it("binds an explicitly chosen unlinked legacy target to the canonical response owner", async () => {
  state.rows.set("legacy", { id: "legacy", title: "Legacy", content: "Shared", createdAt: 1 })
  const result = await pullFromStudio(1, "legacy")
  expect(result.success).toBe(true)
  expect(state.rows.get("legacy")).toMatchObject({ ownerId: alice, user_prompt: "Alice owned", serverId: 1 })
})
it("rejects a captured pull after the requesting account changes", async () => {
  state.owner = bob
  const result = await pullFromStudio(1, undefined, alice)
  expect(result.success).toBe(false)
  expect(mocks.getPrompt).not.toHaveBeenCalled()
})
it("retains a durable recipe error without copying an ownerless response", async () => {
  state.rows.set("recipe", { ...row("recipe", alice), promptFormat: "structured", promptSchemaVersion: 2 })
  mocks.getPrompt.mockResolvedValueOnce({ ok: true, data: { data: { id: 1, name: "FOREIGN RESPONSE" } }, recipePersistence: { state: "dispatched", actualOwnerId: null } })
  const result = await pullFromStudio(1, "recipe", alice)
  expect(result).toMatchObject({ success: false, syncStatus: "error", recipeWriteBlocked: true })
  expect(state.rows.get("recipe")).toMatchObject({ title: "recipe", name: "recipe", ownerId: alice })
})
it("does not change an invalid unlinked local definition before rejecting it", async () => {
  const original = { id: "invalid", title: "Local", content: "Unchanged", createdAt: 1, promptFormat: "structured", promptSchemaVersion: 99, structuredPromptDefinition: { schema_version: 99 } }
  state.rows.set("invalid", structuredClone(original))
  const result = await pushToStudio("invalid", 1, { expectedOwnerId: alice })
  expect(result.success).toBe(false)
  expect(state.rows.get("invalid")).toEqual(original)
  expect(mocks.createPrompt).not.toHaveBeenCalled()
})

it.each(["create", "update"])("C01 %s acknowledgment preserves a newer local edit and the acknowledged server identity", async kind => {
  const original = { ...row("alice", alice), title: "Old prompt", name: "Old prompt", content: "Old instruction", user_prompt: "Old instruction", updatedAt: 100, ...(kind === "create" ? { serverId: undefined } : {}) }
  state.rows.set("alice", original)
  let finish!: (value: unknown) => void
  const transport = kind === "create" ? mocks.createPrompt : mocks.updatePrompt
  transport.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  const saving = pushToStudio("alice", 1, { expectedOwnerId: alice })
  await vi.waitFor(() => expect(transport).toHaveBeenCalledOnce())
  state.rows.set("alice", { ...original, title: "New prompt", name: "New prompt", content: "New instruction", user_prompt: "New instruction", updatedAt: 200, syncStatus: "pending" })
  finish({ ok: true, data: { data: { id: 1, project_id: 1, name: "Old prompt", user_prompt: "Old instruction", version_number: 2 } }, recipePersistence: { state: "dispatched", actualOwnerId: alice } })
  const result = await saving
  expect(state.rows.get("alice")).toMatchObject({ title: "New prompt", name: "New prompt", content: "New instruction", user_prompt: "New instruction", updatedAt: 200, serverId: 1, syncStatus: "pending" })
  expect(result).toMatchObject({ success: false, serverId: 1, syncStatus: "pending" })
})

it("C01 post-dispatch link refusal preserves dispatch evidence and the relinked row", async () => {
  state.rows.set("alice", row("alice", alice))
  let finish!: (value: unknown) => void
  mocks.updatePrompt.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  const saving = pushToStudio("alice", 1, { expectedOwnerId: alice })
  await vi.waitFor(() => expect(mocks.updatePrompt).toHaveBeenCalledOnce())
  const relinked = { ...row("alice", alice), serverId: 99, name: "Relinked prompt", updatedAt: 200 }
  state.rows.set("alice", structuredClone(relinked))
  finish({ ok: true, data: { data: { id: 1, project_id: 1, name: "Old response", version_number: 2 } }, recipePersistence: { state: "dispatched", actualOwnerId: alice } })
  const result = await saving
  expect(state.rows.get("alice")).toEqual(relinked)
  expect(result).toMatchObject({ success: false, recipeOwnership: { dispatch: { state: "dispatched", actualOwnerId: alice } } })
})

it("C01 auto-sync preserves the newer edit revision and retries by updating its acknowledged ID", async () => {
  const original = { ...row("alice", alice), serverId: undefined, name: "Old prompt", content: "Old instruction", user_prompt: "Old instruction", updatedAt: 100 }
  state.rows.set("alice", original)
  let finish!: (value: unknown) => void
  mocks.createPrompt.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  const saving = autoSyncPrompt("alice", 1, { expectedOwnerId: alice })
  await vi.waitFor(() => expect(mocks.createPrompt).toHaveBeenCalledOnce())
  state.rows.set("alice", { ...original, name: "New prompt", content: "New instruction", user_prompt: "New instruction", updatedAt: 200, syncStatus: "pending" })
  finish({ ok: true, data: { data: { id: 1, project_id: 1, name: "Old prompt", user_prompt: "Old instruction", version_number: 2 } }, recipePersistence: { state: "dispatched", actualOwnerId: alice } })
  expect(await saving).toMatchObject({ success: false, syncStatus: "pending", serverId: 1 })
  expect(state.rows.get("alice")).toMatchObject({ name: "New prompt", user_prompt: "New instruction", updatedAt: 200, serverId: 1, syncStatus: "pending" })
  mocks.updatePrompt.mockResolvedValueOnce({ ok: true, data: { data: { id: 1, project_id: 1, name: "New prompt", user_prompt: "New instruction", version_number: 3 } }, recipePersistence: { state: "dispatched", actualOwnerId: alice } })
  expect(await autoSyncPrompt("alice", 1, { expectedOwnerId: alice })).toMatchObject({ success: true, serverId: 1 })
  expect(mocks.createPrompt).toHaveBeenCalledOnce()
  expect(mocks.updatePrompt).toHaveBeenCalledOnce()
  expect(state.rows.get("alice")).toMatchObject({ name: "New prompt", user_prompt: "New instruction", syncStatus: "synced" })
})
