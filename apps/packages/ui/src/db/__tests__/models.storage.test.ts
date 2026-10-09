import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

type StorageArea = chrome.storage.StorageArea
type StorageOperation = "getKeys" | "get" | "set" | "remove"

vi.mock("@/db/nickname", () => ({
  getAllModelNicknames: async () => ({})
}))
vi.mock("@/libs/openai", () => ({
  getAllOpenAIModels: () => {
    throw new Error("Provider calls are not allowed in finite storage tests")
  }
}))

const createStorageArea = () => {
  const store = new Map<string, unknown>()
  const failNext = new Map<StorageOperation, string>()
  const calls = {
    getNull: 0,
    getKeys: 0,
    get: [] as (string | string[] | null)[],
    setCalls: 0,
    removeCalls: 0,
    clearCalls: 0
  }
  const respond = (operation: StorageOperation, callback: () => void) => {
    const error = failNext.get(operation)
    failNext.delete(operation)
    queueMicrotask(() => {
      globalThis.chrome.runtime.lastError = error
        ? { message: error }
        : undefined
      try {
        callback()
      } finally {
        globalThis.chrome.runtime.lastError = undefined
      }
    })
    return error
  }
  const area: StorageArea & {
    __calls: typeof calls
    __store: Map<string, unknown>
    __failNext: typeof failNext
  } =
    (() => {
      const base: any = {
        __calls: calls,
        __store: store,
        __failNext: failNext,
        getKeys(callback: (keys: string[]) => void) {
          if (this !== base) throw new Error("Storage API receiver was lost")
          calls.getKeys += 1
          const keys = [...store.keys()]
          respond("getKeys", () => callback(keys))
        },
        get(keys: any, callback: (items: any) => void) {
          calls.get.push(keys)
          if (keys === null || keys === undefined) {
            calls.getNull += 1
            const all: Record<string, unknown> = {}
            store.forEach((value, key) => {
              all[key] = value
            })
            respond("get", () => callback(all))
            return
          }
          const out: Record<string, unknown> = {}
          for (const key of Array.isArray(keys) ? keys : [keys]) {
            if (store.has(key)) out[key] = store.get(key)
          }
          respond("get", () => callback(out))
        },
        set(items: Record<string, unknown>, callback: () => void) {
          calls.setCalls += 1
          const error = respond("set", () => {
            if (!error) {
              Object.entries(items).forEach(([key, value]) => store.set(key, value))
            }
            callback()
          })
        },
        remove(keys: any, callback: () => void) {
          calls.removeCalls += 1
          const error = respond("remove", () => {
            if (!error) {
              for (const key of Array.isArray(keys) ? keys : [keys]) {
                store.delete(key)
              }
            }
            callback()
          })
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
    vi.stubGlobal("fetch", () => {
      throw new Error("Network calls are not allowed in finite storage tests")
    })
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
    vi.unstubAllGlobals()
  })

  it("getAll() discovers legacy records without storage writes or a full dump when getKeys exists", async () => {
    area.__store.set("tldwConfig", { serverUrl: "https://tldw.example" })
    area.__store.set("unrelated", { not: "a model" })
    area.__store.set("legacy-model-1", model("legacy-model-1"))

    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()

    const first = await db.getAll()
    expect(first).toHaveLength(1)
    expect(first[0]?.id).toBe("legacy-model-1")

    await db.getAll()
    await db.getAll()
    expect(area.__calls.getNull).toBe(0)
    expect(area.__calls.setCalls).toBe(0)
    expect(area.__calls.removeCalls).toBe(0)
    expect(area.__store.get("legacy-model-1")).toEqual(model("legacy-model-1"))

    // Non-model entries never leak into getAll().
    expect(first.some((m: any) => m?.not === "a model")).toBe(false)
  })

  it("createMany writes all new records in a single set()", async () => {
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    await db.getAll()

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

  it.each(["create", "update", "createMany"] as const)("%s rejects a destination occupied by a distinct prefix-shaped legacy model", async (operation) => {
    area.__store.set("model:x", model("model:x", { name: "Legacy preserved" }))
    const before = [...area.__store]
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    const write = operation === "createMany"
      ? db.createMany([model("x")])
      : db[operation](model("x"))
    await expect(write).rejects.toThrow(/occupied model storage key/)
    expect([...area.__store]).toEqual(before)
    expect(area.__calls.setCalls).toBe(0)
    expect(area.__calls.removeCalls).toBe(0)
  })

  it.each([
    { value: { setting: "retained" } },
    { value: { id: "x" } },
    { value: "retained" },
    { value: false },
    { value: 0 },
    { value: null },
    { value: undefined }
  ])("createMany rejects an unrelated occupied destination containing $value", async ({ value }) => {
    area.__store.set("model:x", value)
    const before = [...area.__store]
    const { ModelDb } = await import("@/db/models")
    await expect(new ModelDb().createMany([model("x")])).rejects.toThrow(/occupied model storage key/)
    expect([...area.__store]).toEqual(before)
    expect(area.__calls.setCalls).toBe(0)
    expect(area.__calls.removeCalls).toBe(0)
  })

  it.each([true, false])("createMany validates the entire batch before mutation when conflictFirst=%s", async (conflictFirst) => {
    area.__store.set("model:x", model("model:x"))
    area.__store.set("model:existing", model("existing", { name: "Existing preserved" }))
    const before = [...area.__store]
    const valid = [model("new"), model("existing", { name: "Changed" })]
    const records = conflictFirst ? [model("x"), ...valid] : [...valid, model("x")]
    const { ModelDb } = await import("@/db/models")
    await expect(new ModelDb().createMany(records)).rejects.toThrow(/occupied model storage key/)
    expect([...area.__store]).toEqual(before)
    expect(area.__calls.setCalls).toBe(0)
    expect(area.__calls.removeCalls).toBe(0)
  })

  it("createMany propagates destination-read failure before any mutation", async () => {
    area.__store.set("model:existing", model("existing"))
    area.__failNext.set("get", "destination read failed")
    const before = [...area.__store]
    const { ModelDb } = await import("@/db/models")
    await expect(new ModelDb().createMany([
      model("new"), model("existing", { name: "Changed" })
    ])).rejects.toEqual({ message: "destination read failed" })
    expect([...area.__store]).toEqual(before)
    expect(area.__calls.setCalls).toBe(0)
    expect(area.__calls.removeCalls).toBe(0)
  })

  it("createMany keeps ordinary legacy data and same-ID current put semantics with one set", async () => {
    area.__store.set("x", model("x", { name: "Legacy" }))
    area.__store.set("model:x", model("x", { name: "Current" }))
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    await db.createMany([model("x", { name: "Changed" }), model("new")])
    expect(area.__store.get("x")).toEqual(model("x", { name: "Legacy" }))
    expect(await db.getById("x")).toEqual(model("x", { name: "Changed" }))
    expect((await db.getAll()).map((entry) => entry.id).sort()).toEqual(["new", "x"])
    expect(area.__calls.setCalls).toBe(1)
  })

  it("createMany accepts missing-null keyed results without overwriting an occupied-null setting", async () => {
    const get = area.get.bind(area)
    area.get = ((keys: string[] | null, callback: (items: Record<string, unknown>) => void) => {
      get(keys, (items) => {
        const out = { ...items }
        if (Array.isArray(keys)) {
          for (const key of keys) {
            if (!Object.prototype.hasOwnProperty.call(out, key)) out[key] = null
          }
        }
        callback(out)
      })
    }) as StorageArea["get"]
    area.__store.set("model:occupied", null)
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    await db.createMany([model("new"), model("other")])
    expect(area.__store.get("model:new")).toEqual(model("new"))
    expect(area.__store.get("model:other")).toEqual(model("other"))
    expect(area.__calls.setCalls).toBe(1)
    const before = [...area.__store]
    await expect(db.create(model("occupied"))).rejects.toThrow(/occupied model storage key/)
    expect([...area.__store]).toEqual(before)
    expect(area.__calls.setCalls).toBe(1)
  })

  it("stores records under the model: prefix and keeps legacy data working", async () => {
    area.__store.set("legacy-model-1", model("legacy-model-1"))

    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    await db.getAll()

    // Reads never rewrite or delete legacy records.
    expect(area.__store.has("model:legacy-model-1")).toBe(false)
    expect(area.__store.has("legacy-model-1")).toBe(true)
    expect(await db.getById("legacy-model-1")).toEqual(model("legacy-model-1"))

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
    expect(area.__calls.setCalls).toBe(1)

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

  it("warmed instances and a fresh reader all see sequential distinct creates", async () => {
    area.__store.set("__tldwModelDbIndexV1", [])
    const { ModelDb } = await import("@/db/models")
    const first = new ModelDb()
    const second = new ModelDb()
    await first.getAll()
    await second.getAll()
    await first.create(model("first"))
    await second.create(model("second"))

    for (const reader of [first, second, new ModelDb()]) {
      expect((await reader.getAll()).map((entry) => entry.id).sort()).toEqual([
        "first", "second"
      ])
    }
  })

  it("simultaneous distinct creates from independent module contexts remain discoverable", async () => {
    area.__store.set("__tldwModelDbIndexV1", [])
    const firstModule = await import("@/db/models")
    const first = new firstModule.ModelDb()
    await first.getAll()
    vi.resetModules()
    const secondModule = await import("@/db/models")
    expect(firstModule.ModelDb).not.toBe(secondModule.ModelDb)
    const second = new secondModule.ModelDb()
    await second.getAll()
    await Promise.all([
      first.createMany([model("first"), model("first-batch")]),
      second.createMany([model("second"), model("second-batch")])
    ])

    for (const reader of [first, second, new secondModule.ModelDb()]) {
      expect((await reader.getAll()).map((entry) => entry.id).sort()).toEqual([
        "first", "first-batch", "second", "second-batch"
      ])
    }
  })

  it("a warmed delete does not hide a distinct model created by another instance", async () => {
    area.__store.set("__tldwModelDbIndexV1", ["old"])
    area.__store.set("model:old", model("old"))
    const { ModelDb } = await import("@/db/models")
    const deleting = new ModelDb()
    const creating = new ModelDb()
    await deleting.getAll()
    await creating.getAll()
    await creating.create(model("new"))
    await deleting.delete("old")

    for (const reader of [deleting, creating, new ModelDb()]) {
      expect((await reader.getAll()).map((entry) => entry.id)).toEqual(["new"])
    }
  })

  it.each([
    { index: [] },
    { index: ["missing"] },
    { index: ["first"] },
    { index: "malformed" }
  ])(
    "discovers persisted records despite incomplete index $index",
    async ({ index }) => {
      area.__store.set("__tldwModelDbIndexV1", index)
      area.__store.set("model:first", model("first"))
      area.__store.set("model:second", model("second"))
      const { ModelDb } = await import("@/db/models")
      expect((await new ModelDb().getAll()).map((entry) => entry.id).sort()).toEqual([
        "first", "second"
      ])
    }
  )

  it("bulk-reads the exact prefixed catalog and filters unrelated or malformed values", async () => {
    area.__store.set("model:first", model("first"))
    area.__store.set("model:second", model("second"))
    area.__store.set("model:wrong", model("different-id"))
    area.__store.set("model:invalid", { id: "invalid" })
    area.__store.set("config", { url: "https://finite.invalid" })
    area.__store.set("unrelated", { id: "not-the-key", model_id: "m", provider_id: "p" })
    const { ModelDb } = await import("@/db/models")

    expect((await new ModelDb().getAll()).map((entry) => entry.id).sort()).toEqual([
      "first", "second"
    ])
    expect(area.__calls.getKeys).toBe(1)
    expect(area.__calls.getNull).toBe(0)
    expect(area.__calls.get).toContainEqual([
      "model:first", "model:second", "model:wrong", "model:invalid"
    ])
  })

  it.each([true, false])("prefers current duplicates without mutating legacy data (getKeys=%s)", async (hasGetKeys) => {
    if (!hasGetKeys) (area as { getKeys?: unknown }).getKeys = undefined
    area.__store.set("shared", model("shared", { name: "Legacy" }))
    area.__store.set("model:shared", model("shared", { name: "Current" }))
    const before = [...area.__store]
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()

    expect(await db.getAll()).toEqual([model("shared", { name: "Current" })])
    expect(await db.getById("shared")).toEqual(model("shared", { name: "Current" }))
    expect([...area.__store]).toEqual(before)
    expect(area.__calls.setCalls).toBe(0)
    expect(area.__calls.removeCalls).toBe(0)
  })

  it("uses a read-only older-browser fallback and rediscovers late legacy/current records", async () => {
    ;(area as { getKeys?: unknown }).getKeys = undefined
    area.__store.set("legacy-id-without-generated-suffix", model("legacy-id-without-generated-suffix"))
    area.__store.set("__tldwModelDbIndexV1", [])
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    await db.getAll()
    area.__store.set("late-legacy", model("late-legacy"))
    area.__store.set("model:late-current", model("late-current"))

    expect((await db.getAll()).map((entry) => entry.id).sort()).toEqual([
      "late-current", "late-legacy", "legacy-id-without-generated-suffix"
    ])
    expect(area.__calls.getNull).toBe(2)
    expect(area.__calls.setCalls).toBe(0)
    expect(area.__calls.removeCalls).toBe(0)
  })

  it("getById reads current and legacy aliases directly without enumerating the catalog", async () => {
    area.__store.set("legacy", model("legacy"))
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    expect(await db.getById("legacy")).toEqual(model("legacy"))
    expect(await db.getById("missing")).toBeUndefined()
    expect(area.__calls.get).toEqual([["model:legacy", "legacy"], ["model:missing", "missing"]])
    expect(area.__calls.getKeys).toBe(0)
    expect(area.__calls.getNull).toBe(0)
    expect(area.__calls.setCalls).toBe(0)
  })

  it.each(["getKeys", "get"] as const)("propagates %s discovery failure without empty success or data mutation", async (operation) => {
    area.__store.set("model:existing", model("existing"))
    area.__store.set("legacy", model("legacy"))
    area.__failNext.set(operation, "discovery failed")
    const before = [...area.__store]
    const { ModelDb } = await import("@/db/models")
    await expect(new ModelDb().getAll()).rejects.toEqual({ message: "discovery failed" })
    expect([...area.__store]).toEqual(before)
    expect(area.__calls.setCalls).toBe(0)
    expect(area.__calls.removeCalls).toBe(0)
  })

  it("does not overwrite catalog data after a failed older-browser read", async () => {
    ;(area as { getKeys?: unknown }).getKeys = undefined
    area.__store.set("legacy", model("legacy"))
    area.__failNext.set("get", "legacy read failed")
    const before = [...area.__store]
    const { bulkAddModelsFB } = await import("@/db/models")
    await expect(bulkAddModelsFB([model("replacement")])).rejects.toEqual({ message: "legacy read failed" })
    expect([...area.__store]).toEqual(before)
    expect(area.__calls.setCalls).toBe(0)
    expect(area.__calls.removeCalls).toBe(0)
  })

  it("propagates a direct lookup failure rather than falling back to legacy success", async () => {
    area.__store.set("__tldwModelDbIndexV1", [])
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    await db.getAll()
    area.__store.set("legacy", model("legacy"))
    area.__failNext.set("get", "lookup failed")
    await expect(db.getById("legacy")).rejects.toEqual({ message: "lookup failed" })
  })

  it.each(["createMany", "update"] as const)("propagates %s write failure and preserves persisted values", async (operation) => {
    area.__store.set("model:existing", model("existing"))
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    await db.getAll()
    const before = [...area.__store]
    area.__failNext.set("set", "write failed")
    const write = operation === "createMany"
      ? db.createMany([model("new")])
      : db.update(model("existing", { name: "Changed" }))
    await expect(write).rejects.toEqual({ message: "write failed" })
    expect([...area.__store]).toEqual(before)
  })

  it("createModelFB keeps its failure signal for Dexie rollback callers", async () => {
    area.__store.set("__tldwModelDbIndexV1", [])
    area.__failNext.set("set", "write failed")
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined)
    const { createModelFB } = await import("@/db/models")
    expect(await createModelFB(model("new"))).toBe(false)
    expect(area.__store.has("model:new")).toBe(false)
    error.mockRestore()
  })

  it("propagates batched remove failure and preserves both aliases", async () => {
    area.__store.set("model:existing", model("existing"))
    area.__store.set("existing", model("existing", { name: "Legacy" }))
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    await db.getAll()
    const before = [...area.__store]
    area.__failNext.set("remove", "remove failed")
    await expect(db.deleteMany(["existing"])).rejects.toEqual({ message: "remove failed" })
    expect([...area.__store]).toEqual(before)
  })

  it("deleteAll rejects discovery failure instead of deleting from an empty index", async () => {
    area.__store.set("model:existing", model("existing"))
    area.__failNext.set("getKeys", "discovery failed")
    const before = [...area.__store]
    const { ModelDb } = await import("@/db/models")
    await expect(new ModelDb().deleteAll()).rejects.toEqual({ message: "discovery failed" })
    expect([...area.__store]).toEqual(before)
    expect(area.__calls.removeCalls).toBe(0)
  })

  it.each(["deleteMany", "deleteAll"] as const)("%s propagates alias-read failure without deletion", async (operation) => {
    area.__store.set("__tldwModelDbIndexV1", ["existing"])
    area.__store.set("model:existing", model("existing"))
    const get = area.get.bind(area)
    area.get = ((keys: string[], callback: (items: Record<string, unknown>) => void) => {
      if (keys.includes("existing")) area.__failNext.set("get", "alias read failed")
      get(keys, callback)
    }) as StorageArea["get"]
    const before = [...area.__store]
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    const deletion = operation === "deleteMany" ? db.deleteMany(["existing"]) : db.deleteAll()
    await expect(deletion).rejects.toEqual({ message: "alias read failed" })
    expect([...area.__store]).toEqual(before)
    expect(area.__calls.removeCalls).toBe(0)
  })

  it("an available getKeys synchronous failure propagates without a full-read fallback", async () => {
    const error = new Error("getKeys threw")
    ;(area as { getKeys?: unknown }).getKeys = () => { throw error }
    area.__store.set("model:existing", model("existing"))
    const before = [...area.__store]
    const { ModelDb } = await import("@/db/models")
    await expect(new ModelDb().getAll()).rejects.toBe(error)
    expect([...area.__store]).toEqual(before)
    expect(area.__calls.getNull).toBe(0)
    expect(area.__calls.setCalls).toBe(0)
  })

  it("deleteAll discovers unindexed current and legacy records without touching unrelated data", async () => {
    area.__store.set("__tldwModelDbIndexV1", [])
    area.__store.set("model:current", model("current"))
    area.__store.set("legacy", model("legacy"))
    area.__store.set("config", { token: "finite-placeholder" })
    const { ModelDb } = await import("@/db/models")
    await new ModelDb().deleteAll()
    expect([...area.__store]).toEqual([
      ["__tldwModelDbIndexV1", []],
      ["config", { token: "finite-placeholder" }]
    ])
    expect(await new ModelDb().getAll()).toEqual([])
    expect(area.__calls.removeCalls).toBe(1)
    expect(area.__calls.clearCalls).toBe(0)
  })

  it("deleteAll deletes an index-shaped model but preserves its unrelated unprefixed alias", async () => {
    const id = "__tldwModelDbIndexV1"
    area.__store.set(id, { setting: "retained" })
    area.__store.set(`model:${id}`, model(id))
    area.__store.set("model:ordinary", model("ordinary"))
    const { ModelDb } = await import("@/db/models")
    await new ModelDb().deleteAll()
    expect([...area.__store]).toEqual([[id, { setting: "retained" }]])
    expect(await new ModelDb().getAll()).toEqual([])
    expect(area.__calls.removeCalls).toBe(1)
    expect(area.__calls.clearCalls).toBe(0)
  })

  it("deleteAll leaves obsolete index metadata alone when no model exists", async () => {
    area.__store.set("__tldwModelDbIndexV1", ["stale-model"])
    const before = [...area.__store]
    const { ModelDb } = await import("@/db/models")
    await new ModelDb().deleteAll()
    expect([...area.__store]).toEqual(before)
    expect(area.__calls.removeCalls).toBe(0)
    expect(area.__calls.clearCalls).toBe(0)
  })

  it("deleteAll still deletes a genuine unprefixed index-shaped legacy model", async () => {
    const id = "__tldwModelDbIndexV1"
    area.__store.set(id, model(id))
    const { ModelDb } = await import("@/db/models")
    await new ModelDb().deleteAll()
    expect([...area.__store]).toEqual([])
    expect(area.__calls.removeCalls).toBe(1)
  })

  it("deleteMany preserves a non-model value sharing the unprefixed model id", async () => {
    area.__store.set("model:shared", model("shared"))
    area.__store.set("shared", { setting: "retained" })
    const { ModelDb } = await import("@/db/models")
    await new ModelDb().deleteMany(["shared"])
    expect([...area.__store]).toEqual([["shared", { setting: "retained" }]])
    expect(area.__calls.removeCalls).toBe(1)
  })

  it("deleteAll preserves a non-model legacy alias collision", async () => {
    area.__store.set("model:shared", model("shared"))
    area.__store.set("shared", { setting: "retained" })
    const { ModelDb } = await import("@/db/models")
    await new ModelDb().deleteAll()
    expect([...area.__store]).toEqual([["shared", { setting: "retained" }]])
    expect(area.__calls.removeCalls).toBe(1)
  })

  it.each([true, false])("keeps unrestricted prefix-shaped legacy ids discoverable and separately deletable (getKeys=%s)", async (hasGetKeys) => {
    if (!hasGetKeys) (area as { getKeys?: unknown }).getKeys = undefined
    area.__store.set("model:legacy", model("model:legacy"))
    area.__store.set("model:model:current", model("model:current"))
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    expect((await db.getAll()).map((entry) => entry.id).sort()).toEqual([
      "model:current", "model:legacy"
    ])
    await db.delete("legacy")
    expect(await db.getById("model:legacy")).toEqual(model("model:legacy"))
    await db.delete("model:legacy")
    expect((await db.getAll()).map((entry) => entry.id)).toEqual(["model:current"])
  })

  it.each([true, false])("does not hide a legacy model whose id matches the retired index key (getKeys=%s)", async (hasGetKeys) => {
    if (!hasGetKeys) (area as { getKeys?: unknown }).getKeys = undefined
    const id = "__tldwModelDbIndexV1"
    area.__store.set(id, model(id))
    const before = [...area.__store]
    const { ModelDb } = await import("@/db/models")
    const db = new ModelDb()
    expect(await db.getAll()).toEqual([model(id)])
    expect(await db.getById(id)).toEqual(model(id))
    expect([...area.__store]).toEqual(before)
  })

  it("provider deletion stays batched and preserves other providers and unrelated values", async () => {
    area.__store.set("__tldwModelDbIndexV1", [])
    area.__store.set("model:first", model("first"))
    area.__store.set("legacy", model("legacy"))
    area.__store.set("model:other", model("other", { provider_id: "other" }))
    area.__store.set("config", { retained: true })
    const { deleteAllModelsByProviderId, ModelDb } = await import("@/db/models")
    await deleteAllModelsByProviderId("tldw_openai")
    expect((await new ModelDb().getAll()).map((entry) => entry.id)).toEqual(["other"])
    expect(area.__store.get("config")).toEqual({ retained: true })
    expect(area.__calls.removeCalls).toBe(1)
  })

  it("bulk replacement keeps one remove and one write without deleting unrelated values", async () => {
    area.__store.set("model:old", model("old"))
    area.__store.set("config", { retained: true })
    const { bulkAddModelsFB, ModelDb } = await import("@/db/models")
    await bulkAddModelsFB([model("first"), model("second")])
    expect((await new ModelDb().getAll()).map((entry) => entry.id).sort()).toEqual(["first", "second"])
    expect(area.__store.get("config")).toEqual({ retained: true })
    expect(area.__calls.removeCalls).toBe(1)
    expect(area.__calls.setCalls).toBe(1)
  })

  it("keeps provider utility exports identical for existing callers", async () => {
    const catalog = await import("@/db/models")
    const utilities = await import("@/db/model-provider-utils")
    for (const name of [
      "removeModelSuffix", "isLMStudioModel", "isLlamafileModel", "isLLamaCppModel",
      "isVLLMModel", "getLMStudioModelId", "getLlamafileModelId", "getLLamaCppModelId",
      "getVLLMModelId", "isCustomModel", "dynamicFetchLMStudio", "dynamicFetchLLamaCpp",
      "dynamicFetchVLLM", "dynamicFetchLlamafile"
    ] as const) {
      expect(catalog[name]).toBe(utilities[name])
    }
  })
})
