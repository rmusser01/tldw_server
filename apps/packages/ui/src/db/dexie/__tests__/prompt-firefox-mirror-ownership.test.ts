import { beforeEach, describe, expect, it, vi } from "vitest"
import { autoCleanupTrash, emptyTrash, markPromptSyncError, permanentlyDeletePrompt, restorePromptSnapshot, updatePrompt } from "../helpers"

const state = vi.hoisted(() => ({ dexie: new Map<string, any>(), firefox: new Map<string, any>(), owner: "recipe-owner:sha256:" + "a".repeat(64) }))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({ resolveRecipePersistenceOwnerView: async () => ({ ownerId: state.owner, authorizationRevision: "revision" }), clearRecipePersistenceScoped: vi.fn() }))
vi.mock("../schema", () => ({ db: { prompts: {
  get: async (id: string) => state.dexie.get(id),
  put: async (row: any) => state.dexie.set(row.id, structuredClone(row)),
  update: async (id: string, fields: any) => { Object.assign(state.dexie.get(id), fields); return 1 },
  delete: async (id: string) => state.dexie.delete(id),
  bulkDelete: async (ids: string[]) => ids.forEach(id => state.dexie.delete(id)),
  filter: (predicate: (row: any) => boolean) => ({ toArray: async () => [...state.dexie.values()].filter(predicate), reverse: () => ({ sortBy: async () => [...state.dexie.values()].filter(predicate) }) }),
  // WP-25: the trash queries read the numeric-deletedAt index range instead
  // of scanning the whole table; mirror the range semantics here.
  where: (key: string) => {
    const trashQuery = (test: (row: any) => boolean) => ({
      filter: (next: (row: any) => boolean) => trashQuery(row => test(row) && next(row)),
      toArray: async () => [...state.dexie.values()].filter(test),
      reverse: () => ({ sortBy: async () => [...state.dexie.values()].filter(test) })
    })
    return {
      above: (value: unknown) => trashQuery(row => row[key] > value),
      between: (lower: unknown, upper: unknown, includeLower = true, includeUpper = false) =>
        trashQuery(row =>
          (includeLower ? row[key] >= lower : row[key] > lower) &&
          (includeUpper ? row[key] <= upper : row[key] < upper))
    }
  }
} } }))
vi.mock("../..", () => ({
  getPromptByIdFB: async (id: string) => state.firefox.get(id),
  getAllPromptsFB: async () => [...state.firefox.values()],
  updatePromptFB: async ({ id, ...fields }: any) => { Object.assign(state.firefox.get(id), fields) },
  deletePromptByIdFB: async (id: string) => state.firefox.delete(id),
  restorePromptSnapshotFB: async (row: any) => state.firefox.set(row.id, structuredClone(row)),
  savePromptFB: vi.fn()
}))
const alice = "recipe-owner:sha256:" + "a".repeat(64)
const bob = "recipe-owner:sha256:" + "b".repeat(64)
const record = (extra = {}) => ({ id: "collision", title: "Original", content: "Original", is_system: false, createdAt: 1, ...extra })
const perform = async (action: string) => {
  if (action === "update") await updatePrompt({ id: "collision", content: "Alice edit", expectedOwnerId: alice })
  else if (action === "restore") await restorePromptSnapshot(record({ ownerId: alice, content: "Alice rollback" }) as any)
  else if (action === "error") await markPromptSyncError("collision")
  else if (action === "trash") await emptyTrash()
  else if (action === "cleanup") await autoCleanupTrash(1)
  else await permanentlyDeletePrompt("collision", alice)
}
beforeEach(() => { state.dexie.clear(); state.firefox.clear(); state.owner = alice })
describe("prompt mirror ownership", () => {
  it.each(["update", "restore", "error", "permanent", "trash", "cleanup"].flatMap(action => ["foreign", "ambiguous", "shared"].map(kind => [action, kind])))("preserves a distinct %s fallback row during %s", async (action, kind) => {
    state.dexie.set("collision", record({ ownerId: alice, deletedAt: 1 }))
    const fallback = record(kind === "foreign" ? { ownerId: bob, content: "Bob private" } : kind === "ambiguous" ? { serverId: 17, content: "Unclaimed private" } : { content: "Shared local" })
    state.firefox.set("collision", structuredClone(fallback))
    await perform(action)
    expect(state.firefox.get("collision")).toEqual(fallback)
  })
  it.each(["update", "restore", "error", "permanent", "trash", "cleanup"])("still mirrors %s for matching private ownership", async action => {
    state.dexie.set("collision", record({ ownerId: alice, deletedAt: 1 }))
    state.firefox.set("collision", record({ ownerId: alice, deletedAt: 1 }))
    await perform(action)
    if (["permanent", "trash", "cleanup"].includes(action)) expect(state.firefox.has("collision")).toBe(false)
    else expect(state.firefox.get("collision")).toMatchObject(action === "update" ? { ownerId: alice, content: "Alice edit" } : action === "restore" ? { ownerId: alice, content: "Alice rollback" } : { ownerId: alice, syncStatus: "error" })
  })
  it("still mirrors a strictly shared local edit without claiming ownership", async () => {
    state.dexie.set("collision", record())
    state.firefox.set("collision", record())
    await updatePrompt({ id: "collision", content: "Shared edit", expectedOwnerId: alice })
    expect(state.firefox.get("collision")).toMatchObject({ content: "Shared edit" })
    expect(state.firefox.get("collision").ownerId).toBeUndefined()
  })
})
