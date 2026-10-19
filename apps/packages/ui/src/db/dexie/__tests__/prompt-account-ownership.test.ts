import { beforeEach, describe, expect, it, vi } from "vitest"
import { PageAssistDatabase } from "../chat"
import { capturePromptOwnerScope } from "@/services/prompt-local-owner"

const state = vi.hoisted(() => {
  const rows = new Map<string, any>()
  const query = (predicate = (_p: any) => true) => ({
    toArray: async () => [...rows.values()].filter(predicate),
    first: async () => [...rows.values()].find(predicate),
    filter: (next: (p: any) => boolean) => query(p => predicate(p) && next(p)),
    reverse: () => query(predicate),
    sortBy: async (_key: string) => [...rows.values()].filter(predicate)
  })
  return { calls: 0, owner: "recipe-owner:sha256:" + "a".repeat(64), rows, table: {
    toArray: query().toArray,
    get: async (id: string) => rows.get(id),
    add: async (p: any) => { rows.set(p.id, structuredClone(p)) },
    put: async (p: any) => { rows.set(p.id, structuredClone(p)) },
    update: async (id: string, fields: any) => {
      const p = rows.get(id); if (!p) return 0
      if (typeof fields === "function") { if (fields(p) === false) return 0 }
      else Object.assign(p, fields)
      return 1
    },
    delete: async (id: string) => { rows.delete(id) },
    bulkDelete: async (ids: string[]) => { ids.forEach(id => rows.delete(id)) },
    clear: async () => { rows.clear() },
    bulkPut: async (prompts: any[]) => { prompts.forEach(prompt => rows.set(prompt.id, structuredClone(prompt))) },
    filter: query().filter,
    where: (key: string) => ({ equals: (value: unknown) => query(p => p[key] === value), anyOf: (values: unknown[]) => query(p => values.includes(p[key])) })
  } }
})
vi.mock("../schema", () => ({ db: { prompts: state.table, transaction: async (_mode: unknown, _table: unknown, fn: () => unknown) => fn() } }))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({ resolveRecipePersistenceOwnerView: async () => { state.calls++; return state.owner ? { ownerId: state.owner, authorizationRevision: "revision" } : null } }))
const alice = "recipe-owner:sha256:" + "a".repeat(64)
const bob = "recipe-owner:sha256:" + "b".repeat(64)
const record = (id: string, extra = {}) => ({ id, title: id, content: id, is_system: false, createdAt: 1, ...extra })
const database = new PageAssistDatabase()
beforeEach(() => {
  state.owner = alice; state.calls = 0; state.rows.clear()
  for (const p of [record("alice", { ownerId: alice }), record("bob", { ownerId: bob, serverId: 1 }), record("ambiguous", { serverId: 2, syncStatus: "synced" }), record("legacy"), record("bob-trash", { ownerId: bob, deletedAt: 1 }), record("alice-trash", { ownerId: alice, deletedAt: 1 })]) state.rows.set(p.id, p)
})
describe("prompt local account boundary", () => {
  it("offers only strictly shared unlinked legacy content without resolving an owner", async () => {
    expect((await database.getSharedLocalPrompts()).map(p => p.id)).toEqual(["legacy"])
    expect(state.calls).toBe(0)
  })
  it("uses a captured owner inside restore without another network owner resolution", async () => {
    const ownerScope = await capturePromptOwnerScope()
    try {
      await database.importPromptsV2([record("restored")] as any, { ownerScope })
      expect(state.calls).toBe(1)
      expect(state.rows.get("restored").ownerId).toBe(alice)
    } finally { ownerScope.release() }
  })
  it("migrates a legacy linked row without claiming or rewriting it", async () => {
    const linked = record("old-linked", { serverId: 17, syncStatus: "synced" })
    await database.migrateLegacyPrompt(linked as any)
    expect(state.rows.get("old-linked")).toEqual(linked)
    expect((await database.getAllPrompts()).map(p => p.id)).not.toContain("old-linked")
  })
  it("counts preserved foreign and ambiguous migration rows without exposing them", async () => {
    expect(await database.countStoredPromptsForMigration()).toBe(6)
  })
  it("lists current private and unlinked legacy rows without claiming ambiguous linked rows", async () => {
    expect((await database.getAllPrompts()).map(p => p.id)).toEqual(["alice", "legacy"])
    expect(state.rows.get("ambiguous")).not.toHaveProperty("ownerId")
  })
  it.each(["bob", "ambiguous"])("hides foreign direct ID %s", async id => {
    expect(await database.getPromptById(id)).toBeUndefined()
  })
  it.each(["update", "delete", "restore", "permanent", "usage"])("refuses %s of a foreign prompt without changing it", async action => {
    const before = structuredClone(state.rows.get("bob"))
    const operation = action === "update" ? database.updatePrompt("bob", { content: "stolen", ownerId: alice } as any)
      : action === "delete" ? database.deletePrompt("bob")
      : action === "restore" ? database.restorePrompt("bob")
      : action === "permanent" ? database.permanentlyDeletePrompt("bob")
      : database.incrementPromptUsage("bob")
    await expect(operation).rejects.toThrow(/owner|account/i)
    expect(state.rows.get("bob")).toEqual(before)
  })
  it("purges only current-owner trash and keeps foreign rows", async () => {
    expect(await database.emptyTrash()).toBe(1)
    expect(state.rows.has("bob-trash")).toBe(true)
  })
  it("binds new authenticated local saves and prevents owner reassignment", async () => {
    await database.addPrompt(record("new") as any)
    expect(state.rows.get("new").ownerId).toBe(alice)
    await database.updatePrompt("new", { ownerId: bob, content: "updated" } as any)
    expect(state.rows.get("new").ownerId).toBe(alice)
  })
  it("keeps unlinked legacy prompts usable offline without importing private rows", async () => {
    state.owner = ""
    expect((await database.getAllPrompts()).map(p => p.id)).toEqual(["legacy"])
    await database.addPrompt(record("offline") as any)
    expect(state.rows.get("offline").ownerId ?? null).toBeNull()
  })
  it("replaces only visible owned/local rows and cannot overwrite a foreign import ID", async () => {
    const foreign = structuredClone(state.rows.get("bob"))
    await database.importPromptsV2([record("new-import"), record("bob", { ownerId: alice, content: "forged" })] as any, { replaceExisting: true })
    expect(state.rows.get("bob")).toEqual(foreign)
    expect(state.rows.get("new-import").ownerId).toBe(alice)
    expect(state.rows.has("ambiguous")).toBe(true)
    expect(state.rows.has("alice")).toBe(false)
  })
  it("rejects an explicit retired save owner before creating a local row", async () => {
    await expect(database.addPrompt(record("stale") as any, bob)).rejects.toThrow(/account/i)
    expect(state.rows.has("stale")).toBe(false)
  })
})
