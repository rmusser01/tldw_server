import { beforeEach, describe, expect, it, vi } from "vitest"
import type { MemoryDexie } from "@/components/Option/Playground/__tests__/harness/memory-dexie"

/**
 * WP-28 / TASK-13523 — getAllModelNicknames() must not re-read the nickname
 * table on every call (it runs per history load and per model-list render).
 * A module-level in-memory cache serves repeats; nickname mutations in the
 * same module (saveModelNickname / importDataV2) and whole-database clears
 * invalidate it.
 *
 * Limitation (accepted per plan): the cache is per JS context. Another
 * browser context (popup + sidepanel) has its own module instance; there is
 * no cross-context invalidation. A same-context mutation that bypasses this
 * module would go unnoticed until reload.
 */
const memory = vi.hoisted(() => ({ db: null as unknown }))

vi.mock("@/db/dexie/schema", async () => {
  const { createMemoryDexie } = await import(
    "@/components/Option/Playground/__tests__/harness/memory-dexie"
  )
  memory.db = memory.db ?? createMemoryDexie()
  return { db: memory.db }
})

import { MemoryCollection } from "@/components/Option/Playground/__tests__/harness/memory-dexie"
import {
  getAllModelNicknames,
  getModelNicknameByID,
  invalidateModelNicknameCache,
  ModelNickname,
  saveModelNickname
} from "../nickname"
import type { ModelNickname as MNick, ModelNicknames } from "../types"

const db = () => memory.db as MemoryDexie

const row = (model_id: string, model_name: string): MNick => ({
  id: model_id,
  model_id,
  model_name,
  model_avatar: undefined
})

describe("model nickname cache (WP-28)", () => {
  beforeEach(() => {
    db().resetAll()
    invalidateModelNicknameCache()
    db().modelNickname.bulkPut([
      row("model-a", "Alpha"),
      row("model-b", "Beta")
    ] as never[])
  })

  it("returns a model_id-keyed record of every nickname", async () => {
    const nicknames = await getAllModelNicknames()
    expect(Object.keys(nicknames).sort()).toEqual(["model-a", "model-b"])
    expect(nicknames["model-a"].model_name).toBe("Alpha")
  })

  it("serves repeat reads from the cache without touching storage", async () => {
    const first = await getAllModelNicknames()
    // getAllModelNicknames reads through modelNickname.reverse().toArray(),
    // so watch the collection-level materialization, not just the table's.
    const toArray = vi.spyOn(MemoryCollection.prototype, "toArray")
    try {
      const second = await getAllModelNicknames()
      expect(second).toBe(first)
      expect(toArray).not.toHaveBeenCalled()
    } finally {
      toArray.mockRestore()
    }
  })

  it("a failed storage read is not cached", async () => {
    const toArray = vi.spyOn(MemoryCollection.prototype, "toArray")
    toArray.mockRejectedValueOnce(new Error("storage unavailable"))
    try {
      await expect(getAllModelNicknames()).resolves.toEqual({})
      const retry = await getAllModelNicknames()
      expect(Object.keys(retry).sort()).toEqual(["model-a", "model-b"])
    } finally {
      toArray.mockRestore()
    }
  })

  it("saveModelNickname invalidates the cache: the next read observes the mutation", async () => {
    await getAllModelNicknames()
    await saveModelNickname({ model_id: "model-a", model_name: "Alpha Renamed" })
    const nicknames = await getAllModelNicknames()
    expect(nicknames["model-a"].model_name).toBe("Alpha Renamed")
  })

  it("importDataV2 invalidates the cache: the next read observes imported rows", async () => {
    await getAllModelNicknames()
    const batch: ModelNicknames = [row("model-c", "Gamma")]
    // Settings import / migration write through ModelNickname.importDataV2.
    await new ModelNickname().importDataV2(batch)
    const nicknames = await getAllModelNicknames()
    expect(nicknames["model-c"]?.model_name).toBe("Gamma")
    expect(nicknames["model-a"]?.model_name).toBe("Alpha")
  })

  it("invalidateModelNicknameCache forces the next read back to storage", async () => {
    await getAllModelNicknames()
    db().modelNickname.clear()
    // A raw-table clear bypasses the module; the cache still serves stale
    // data until it is invalidated (documented limitation).
    expect((await getAllModelNicknames())["model-a"]).toBeDefined()
    invalidateModelNicknameCache()
    expect(await getAllModelNicknames()).toEqual({})
  })

  it("getModelNicknameByID reads straight through (uncached single lookups stay exact)", async () => {
    await getAllModelNicknames()
    db().modelNickname.put(row("model-d", "Delta") as never)
    expect((await getModelNicknameByID("model-d"))?.model_name).toBe("Delta")
  })
})
