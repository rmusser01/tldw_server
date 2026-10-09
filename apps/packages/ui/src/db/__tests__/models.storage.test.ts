import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

type StorageArea = chrome.storage.StorageArea

const createStorageArea = () => {
  const store = new Map<string, unknown>()
  const calls = {
    getNull: 0,
    setCalls: 0,
    removeCalls: 0,
    clearCalls: 0
  }
  const area: StorageArea & { __calls: typeof calls; __store: Map<string, unknown> } =
    (() => {
      const base: any = {
        __calls: calls,
        __store: store,
        get(keys: any, callback: (items: any) => void) {
          if (keys === null || keys === undefined) {
            calls.getNull += 1
            const all: Record<string, unknown> = {}
            store.forEach((value, key) => {
              all[key] = value
            })
            setTimeout(() => callback(all), 0)
            return
          }
          const out: Record<string, unknown> = {}
          for (const key of Array.isArray(keys) ? keys : [keys]) {
            if (store.has(key)) out[key] = store.get(key)
          }
          setTimeout(() => callback(out), 0)
        },
        set(items: Record<string, unknown>, callback: () => void) {
          calls.setCalls += 1
          Object.entries(items).forEach(([key, value]) => store.set(key, value))
          setTimeout(() => callback(), 0)
        },
        remove(keys: any, callback: () => void) {
          calls.removeCalls += 1
          for (const key of Array.isArray(keys) ? keys : [keys]) {
            store.delete(key)
          }
          setTimeout(() => callback(), 0)
        },
        clear(callback: () => void) {
          calls.clearCalls += 1
          store.clear()
          setTimeout(() => callback(), 0)
        }
      }
      return base
    })()
  return area
}

const model = (id: string, overrides: Record<string, unknown> = {}) => ({
  id,
  model_id: id.replace(/_model.*$/, ""),
  name: `Model ${id}`,
  provider_id: "tldw_openai",
  lookup: `${id}`,
  model_type: "chat",
  db_type: "openai_model",
  ...overrides
})

describe("ModelDb storage efficiency", () => {
  let area: ReturnType<typeof createStorageArea>
  const previousChrome = (globalThis as any).chrome

  beforeEach(() => {
    area = createStorageArea()
    const lastErrorGetter = vi.fn(() => undefined)
    ;(globalThis as any).chrome = {
      ...(previousChrome || {}),
      storage: { local: area },
      runtime: {
        lastError: undefined,
        getLastErrorMessage: lastErrorGetter
      }
    }
  })

  afterEach(() => {
    ;(globalThis as any).chrome = previousChrome
  })

  it("getAll() no longer dumps the entire storage after the one-time migration", async () => {
    area.__store.set("tldwConfig", { serverUrl: "https://tldw.example" })
    area.__store.set("unrelated", { not: "a model" })
    area.__store.set("legacy-model-1", model("legacy-model-1"))

    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()

    const first = await db.getAll()
    // Legacy record survived the migration and is still returned.
    expect(first).toHaveLength(1)
    expect(first[0]?.id).toBe("legacy-model-1")

    // The migration performed one full read; every later getAll must not.
    const getNullAfterMigration = area.__calls.getNull
    expect(getNullAfterMigration).toBeGreaterThanOrEqual(1)
    await db.getAll()
    await db.getAll()
    expect(area.__calls.getNull).toBe(getNullAfterMigration)

    // Non-model entries never leak into getAll().
    expect(first.some((m: any) => m?.not === "a model")).toBe(false)
  })

  it("createMany writes all new records plus the index in a single set()", async () => {
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    await db.getAll() // force migration

    area.__calls.setCalls = 0
    await db.createMany([
      model("a-model-1"),
      model("b-model-2"),
      model("c-model-3")
    ])

    expect(area.__calls.setCalls).toBe(1)
    const all = await db.getAll()
    expect(all.map((m: any) => m.id).sort()).toEqual([
      "a-model-1",
      "b-model-2",
      "c-model-3"
    ])
  })

  it("stores records under the model: prefix and keeps legacy data working", async () => {
    area.__store.set("legacy-model-1", model("legacy-model-1"))

    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    await db.getAll()

    // Legacy key migrated to the prefixed location.
    expect(area.__store.has("model:legacy-model-1")).toBe(true)
    expect(area.__store.has("legacy-model-1")).toBe(false)

    await db.createMany([model("new-model-2")])
    expect(area.__store.has("model:new-model-2")).toBe(true)

    const fetched = await db.getById("new-model-2")
    expect(fetched?.name).toBe("Model new-model-2")

    await db.update(model("new-model-2", { name: "Renamed" }))
    expect((await db.getById("new-model-2"))?.name).toBe("Renamed")

    await db.delete("new-model-2")
    expect(await db.getById("new-model-2")).toBeUndefined()
    expect((await db.getAll()).map((m: any) => m.id)).toEqual(["legacy-model-1"])
  })

  it("deleteAll only clears model records, leaving unrelated storage intact", async () => {
    area.__store.set("tldwConfig", { serverUrl: "https://tldw.example" })
    area.__store.set("legacy-model-1", model("legacy-model-1"))

    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    await db.getAll()
    await db.createMany([model("x-model-9")])

    await db.deleteAll()

    expect(area.__calls.clearCalls).toBe(0)
    expect(area.__store.has("tldwConfig")).toBe(true)
    expect(await db.getAll()).toEqual([])
  })

  it("createManyModels syncs a catalog with one batched write", async () => {
    const { createManyModels } = await import("@/db/models")
    await createManyModels([
      { model_id: "m1", name: "M1", provider_id: "tldw_openai", model_type: "chat" },
      { model_id: "m2", name: "M2", provider_id: "tldw_openai", model_type: "chat" }
    ])

    const db = new (await import("@/db/models")).ModelDb()
    const all = await db.getAll()
    expect(all).toHaveLength(2)

    area.__calls.setCalls = 0
    // Re-syncing the same catalog must not write anything.
    const { createManyModels: syncAgain } = await import("@/db/models")
    await syncAgain([
      { model_id: "m1", name: "M1", provider_id: "tldw_openai", model_type: "chat" },
      { model_id: "m2", name: "M2", provider_id: "tldw_openai", model_type: "chat" }
    ])
    expect(area.__calls.setCalls).toBe(0)
  })
})
