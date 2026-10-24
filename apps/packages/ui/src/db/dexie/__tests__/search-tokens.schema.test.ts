import Dexie from "dexie"
import { describe, expect, it, vi } from "vitest"
import { createMemoryDexie } from "@/components/Option/Playground/__tests__/harness/memory-dexie"

/**
 * WP-23 / TASK-13523 — schema contract for the `searchTokens` side table.
 *
 * Fresh installs pick the version-17 declaration directly; existing databases
 * run the wired upgrade handler, whose backfill tokenizes every stored
 * message. Dexie cannot open under jsdom (no IndexedDB, no fake-indexeddb in
 * the repo), so the declaration is captured through the version() calls the
 * constructor makes, and the upgrade handler is executed against a scripted
 * transaction.
 */
import { tokenizeSearchText } from "../search-tokenizer"
import { PageAssistDexieDB } from "../schema"

type VersionRecord = {
  stores?: Record<string, string>
  upgrade?: unknown
}

const constructCapturing = () => {
  const records: Record<number, VersionRecord> = {}
  const versionCalls: number[] = []
  const original = Dexie.prototype.version
  const spy = vi.spyOn(Dexie.prototype, "version").mockImplementation(function (
    this: Dexie,
    versionNumber: number
  ) {
    versionCalls.push(versionNumber)
    const instance = original.call(this, versionNumber) as Dexie.Version & {
      stores(stores: Record<string, string>): unknown
      upgrade(handler: unknown): unknown
    }
    const record = (records[versionNumber] ??= {})
    // The wrapper must stay in the chain so both stores() and upgrade() are
    // captured when the constructor chains them.
    const wrapper = {
      stores(payload: Record<string, string>) {
        record.stores = payload
        instance.stores(payload)
        return wrapper
      },
      upgrade(handler: unknown) {
        record.upgrade = handler
        instance.upgrade(handler as never)
        return wrapper
      }
    }
    return wrapper as unknown as Dexie.Version
  })
  try {
    new PageAssistDexieDB()
  } finally {
    spy.mockRestore()
  }
  return { records, versionCalls }
}

describe("search tokenization", () => {
  it("splits content into unique lowercase ASCII word runs", () => {
    expect(tokenizeSearchText("Hello, HELLO world!")).toEqual(["hello", "world"])
    expect(tokenizeSearchText("one two one two")).toEqual(["one", "two"])
  })

  it("returns no tokens for empty or punctuation-only text", () => {
    expect(tokenizeSearchText("")).toEqual([])
    expect(tokenizeSearchText("+++ !!! ...")).toEqual([])
  })
})

describe("schema version 17 (fresh install declaration)", () => {
  it("declares version 17 as the newest version", () => {
    const { versionCalls } = constructCapturing()
    expect(Math.max(...versionCalls)).toBe(17)
  })

  it("adds only the additive searchTokens table", () => {
    const { records } = constructCapturing()
    const version17 = records[17]
    expect(version17).toBeDefined()
    // Additive-only: no pre-existing table's spec may change.
    expect(Object.keys(version17!.stores ?? {})).toEqual(["searchTokens"])
    expect(version17!.stores!.searchTokens).toBe("&[token+history_id], token")
  })
})

describe("schema version 17 (existing database upgrade)", () => {
  it("wires an upgrade handler", () => {
    const { records } = constructCapturing()
    expect(typeof records[17]?.upgrade).toBe("function")
  })

  it("backfills deduplicated token rows for every stored message", async () => {
    const { records } = constructCapturing()
    const upgrade = records[17]!.upgrade as (tx: unknown) => Promise<unknown>
    const messages = [
      { id: "m1", history_id: "h1", content: "hello world hello" },
      { id: "m2", history_id: "h1", content: "WORLD again" },
      { id: "m3", history_id: "h2", content: "" }
    ]
    const tokenPuts: Array<{ token: string; history_id: string }> = []
    let usedStreamingRead = false
    const tx = {
      table: (name: string) => {
        if (name === "messages")
          return {
            each: async (callback: (row: (typeof messages)[number]) => void) => {
              usedStreamingRead = true
              for (const row of messages) callback(row)
            }
          }
        if (name === "searchTokens")
          return {
            bulkPut: async (rows: Array<{ token: string; history_id: string }>) => {
              tokenPuts.push(...rows)
            }
          }
        throw new Error(`unexpected table ${name}`)
      }
    }
    await upgrade(tx)
    expect(usedStreamingRead).toBe(true)
    const pairs = tokenPuts.map((row) => `${row.token}|${row.history_id}`)
    // Case is folded and (token, history) pairs are deduplicated.
    expect(new Set(pairs).size).toBe(tokenPuts.length)
    expect(pairs).toContain("hello|h1")
    expect(pairs).toContain("world|h1")
    expect(pairs).toContain("again|h1")
    expect(tokenPuts.some((row) => row.history_id === "h2")).toBe(false)
  })
})

describe("in-memory harness mirrors the searchTokens schema", () => {
  it("keys rows uniquely by (token, history_id) and indexes token", async () => {
    const memory = createMemoryDexie()
    await memory.searchTokens.bulkPut([
      { token: "zebra", history_id: "a" },
      { token: "zebra", history_id: "a" },
      { token: "zebra", history_id: "b" }
    ])
    expect(await memory.searchTokens.count()).toBe(2)
    const keys = (await memory.searchTokens.where("token").anyOf(["zebra"]).primaryKeys()) as Array<
      [string, string]
    >
    expect(keys).toEqual([
      ["zebra", "a"],
      ["zebra", "b"]
    ])
  })
})
