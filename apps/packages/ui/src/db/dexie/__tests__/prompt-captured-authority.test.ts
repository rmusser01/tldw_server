import { beforeEach, describe, expect, it, vi } from "vitest"
import { autoCleanupTrash, emptyTrash, getPromptById } from "../helpers"

const state = vi.hoisted(() => ({
  rows: new Map<string, any>(),
  mirrors: new Map<string, any>(),
  owner: "recipe-owner:sha256:" + "a".repeat(64),
  ownerSequence: [] as string[],
  switchAfterRead: false
}))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({
  resolveRecipePersistenceOwnerView: async () => ({
    ownerId: state.ownerSequence.shift() ?? state.owner,
    authorizationRevision: "revision"
  }),
  clearRecipePersistenceScoped: vi.fn()
}))
vi.mock("../schema", () => ({ db: { prompts: {
  get: async (id: string) => state.rows.get(id),
  bulkDelete: async (ids: string[]) => ids.forEach(id => state.rows.delete(id)),
  filter: (predicate: (row: any) => boolean) => ({
    toArray: async () => [...state.rows.values()].filter(predicate),
    reverse: () => ({
      sortBy: async () => {
        const rows = [...state.rows.values()].filter(predicate)
        if (state.switchAfterRead) state.owner = bob
        return rows
      }
    })
  }),
  // WP-25: the trash queries read the numeric-deletedAt index range instead
  // of scanning the whole table; mirror the range semantics here.
  where: (key: string) => {
    const trashQuery = (test: (row: any) => boolean) => ({
      filter: (next: (row: any) => boolean) => trashQuery(row => test(row) && next(row)),
      toArray: async () => [...state.rows.values()].filter(test),
      reverse: () => ({
        sortBy: async () => {
          const rows = [...state.rows.values()].filter(test)
          if (state.switchAfterRead) state.owner = bob
          return rows
        }
      })
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
  getPromptByIdFB: async (id: string) => state.mirrors.get(id),
  deletePromptByIdFB: async (id: string) => state.mirrors.delete(id)
}))
const alice = "recipe-owner:sha256:" + "a".repeat(64)
const bob = "recipe-owner:sha256:" + "b".repeat(64)
const record = (id: string, ownerId?: string) => ({
  id, ownerId, title: id, content: id, is_system: false, createdAt: 1, deletedAt: 1
})
beforeEach(() => {
  state.rows.clear()
  state.mirrors.clear()
  state.owner = alice
  state.ownerSequence = []
  state.switchAfterRead = false
})
describe("captured prompt helper authority", () => {
  it("does not return a successor's native row to the initiating owner", async () => {
    const originalBob = record("bob-only", bob)
    state.rows.set(originalBob.id, structuredClone(originalBob))
    state.ownerSequence = [alice, bob]
    expect(await getPromptById(originalBob.id)).toBeNull()
    expect(state.rows.get(originalBob.id)).toEqual(originalBob)
  })
  it.each(["trash", "cleanup"])("preserves both accounts across a retired %s selection", async action => {
    const originalAlice = record("alice-trash", alice)
    const originalBob = record("bob-trash", bob)
    state.rows.set(originalAlice.id, structuredClone(originalAlice))
    state.rows.set(originalBob.id, structuredClone(originalBob))
    state.mirrors.set(originalAlice.id, structuredClone(originalAlice))
    state.switchAfterRead = true
    const operation = action === "trash" ? emptyTrash() : autoCleanupTrash(1)
    await expect(operation).rejects.toThrow(/account changed/)
    expect(state.rows.get(originalBob.id)).toEqual(originalBob)
    expect(state.rows.get(originalAlice.id)).toEqual(originalAlice)
    expect(state.mirrors.get(originalAlice.id)).toEqual(originalAlice)
  })
  it.each([alice, undefined])("still reads matching private or shared local ownership %s", async ownerId => {
    const prompt = record("matching", ownerId)
    state.rows.set(prompt.id, prompt)
    expect(await getPromptById(prompt.id)).toEqual(prompt)
  })
})
