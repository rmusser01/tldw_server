import { beforeEach, describe, expect, it, vi } from "vitest"
import type { MemoryDexie } from "@/components/Option/Playground/__tests__/harness/memory-dexie"

/**
 * WP-25 / TASK-13523 — prompts trash queries must use the `deletedAt` index
 * (present in the schema since v9) instead of JS full-table scans.
 *
 * These tests characterize the *outcomes* of the trash query family on the
 * in-memory Dexie stand-in (jsdom has no IndexedDB; the repo ships no
 * fake-indexeddb). They were captured against the pre-rewrite `.filter()`
 * implementation and must stay green through the index rewrite.
 *
 * IndexedDB never indexes `null`: rows with `deletedAt: null` (or with the
 * field absent, as on pre-v9 legacy rows) have no entry in the `deletedAt`
 * index, and Dexie's `where(...).equals(null)` rejects the argument outright.
 * The index can therefore only ever select the *deleted* side (numeric
 * `deletedAt`); the non-deleted family keeps its JS filter by necessity.
 */
const memory = vi.hoisted(() => ({ db: null as unknown }))
const ownerState = vi.hoisted(() => ({ owner: "recipe-owner:sha256:" + "a".repeat(64) }))

vi.mock("@/db/dexie/schema", async () => {
  const { createMemoryDexie } = await import(
    "@/components/Option/Playground/__tests__/harness/memory-dexie"
  )
  memory.db = memory.db ?? createMemoryDexie()
  return { db: memory.db }
})
vi.mock("@/services/recipe-persistence-uncertainty", () => ({
  resolveRecipePersistenceOwnerView: async () => ({
    ownerId: ownerState.owner,
    authorizationRevision: "revision"
  }),
  clearRecipePersistenceScoped: vi.fn()
}))

import { PageAssistDatabase } from "../chat"
import type { Prompt } from "../types"

const db = () => memory.db as MemoryDexie
const alice = "recipe-owner:sha256:" + "a".repeat(64)
const bob = "recipe-owner:sha256:" + "b".repeat(64)

const record = (id: string, extra: Partial<Prompt> = {}): Prompt => ({
  id,
  title: id,
  name: id,
  content: `content of ${id}`,
  is_system: false,
  createdAt: 1_000,
  ...extra
} as Prompt)

const DAY_MS = 24 * 60 * 60 * 1000

/**
 * Seed every ownership/deletion shape the trash queries distinguish:
 * - alice-owned live / trashed (recent + older than the cleanup cutoff)
 * - bob-owned live / trashed (foreign — never visible to alice)
 * - shared unlinked legacy live / trashed (visible to any owner)
 * - a legacy row whose `deletedAt` key is entirely ABSENT (pre-v9 shape)
 */
const seed = () => {
  const now = Date.now()
  const rows: Prompt[] = [
    record("alice-live", { ownerId: alice }),
    record("alice-trash-recent", { ownerId: alice, deletedAt: now - 1_000 }),
    record("alice-trash-old", { ownerId: alice, deletedAt: now - 10 * DAY_MS }),
    record("bob-live", { ownerId: bob }),
    record("bob-trash", { ownerId: bob, deletedAt: now - 1_000 }),
    record("shared-live"),
    record("shared-trash", { deletedAt: now - 2_000 }),
    { id: "legacy-absent", title: "legacy-absent", content: "legacy", is_system: false, createdAt: 900 }
  ]
  db().prompts.bulkPut(rows as never[])
}

const database = new PageAssistDatabase()

describe("prompts trash queries (WP-25, deletedAt index)", () => {
  beforeEach(() => {
    db().resetAll()
    seed()
  })

  it("getDeletedPrompts lists trashed prompts visible to the owner, newest deletion first", async () => {
    // Shared unlinked trash is visible to every owner (isPromptOwnedBy);
    // bob's trashed row stays hidden. Distinct deletedAt values keep the
    // assertion independent of tie-ordering minutiae.
    const deleted = await database.getDeletedPrompts()
    expect(deleted.map(p => p.id)).toEqual(["alice-trash-recent", "shared-trash", "alice-trash-old"])
  })

  it("getSharedLocalPrompts(true) lists only shared trashed prompts without resolving an owner", async () => {
    const deleted = await database.getSharedLocalPrompts(true)
    expect(deleted.map(p => p.id)).toEqual(["shared-trash"])
  })

  it("getSharedLocalPrompts() keeps shared live prompts, including the pre-v9 row with no deletedAt key", async () => {
    const live = await database.getSharedLocalPrompts()
    // createdAt descending: shared-live (1_000) before legacy-absent (900).
    expect(live.map(p => p.id)).toEqual(["shared-live", "legacy-absent"])
  })

  it("getAllPrompts keeps returning live owned and shared rows, including absent-deletedAt legacy rows", async () => {
    const live = await database.getAllPrompts()
    expect(live.map(p => p.id).sort()).toEqual(["alice-live", "legacy-absent", "shared-live"])
  })

  it("emptyTrash permanently deletes the owner's and shared trashed rows, never foreign ones", async () => {
    // alice sees her two trashed rows plus the shared one.
    expect(await database.emptyTrash()).toBe(3)
    const remaining = (await db().prompts.toArray()).map(p => p.id).sort()
    expect(remaining).toEqual(["alice-live", "bob-live", "bob-trash", "legacy-absent", "shared-live"])
  })

  it("autoCleanupTrash purges only trashed rows older than the cutoff", async () => {
    // 1-day cutoff: alice-trash-old (10 days) is purged; 1-second-old trash survives.
    expect(await database.autoCleanupTrash(1)).toBe(1)
    const ids = (await db().prompts.toArray()).map(p => p.id)
    expect(ids).not.toContain("alice-trash-old")
    expect(ids).toContain("alice-trash-recent")
    expect(ids).toContain("bob-trash")
    expect(ids).toContain("shared-trash")
  })

  it("countStoredPromptsForMigration counts every stored row for any owner", async () => {
    expect(await database.countStoredPromptsForMigration()).toBe(8)
  })

  it("a prompt soft-deleted through deletePrompt becomes visible to the trash queries", async () => {
    await database.deletePrompt("alice-live")
    expect((await database.getDeletedPrompts()).map(p => p.id)).toContain("alice-live")
    expect((await database.getAllPrompts()).map(p => p.id)).not.toContain("alice-live")
  })

  describe("index usage (mechanism, post-rewrite)", () => {
    it("trash queries never materialize or JS-scan the whole prompts table", async () => {
      const toArray = vi.spyOn(db().prompts, "toArray")
      toArray.mockImplementation(async () => {
        throw new Error("full-table materialization: prompts.toArray() must not be used by trash queries")
      })
      const tableFilter = vi.spyOn(db().prompts, "filter")
      tableFilter.mockImplementation(() => {
        throw new Error("full-table JS scan: prompts.filter() must not be used by trash queries")
      })
      try {
        await database.getDeletedPrompts()
        await database.getSharedLocalPrompts(true)
        await database.emptyTrash()
        await database.autoCleanupTrash(1)
        await database.countStoredPromptsForMigration()
      } finally {
        toArray.mockRestore()
        tableFilter.mockRestore()
      }
    })
  })
})
