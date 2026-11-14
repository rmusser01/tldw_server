import { beforeEach, describe, expect, it, vi } from "vitest"
import type { MemoryDexie } from "@/components/Option/Playground/__tests__/harness/memory-dexie"

/**
 * WP-26 / TASK-13523 — history metadata must come from index-only count
 * queries plus O(log n) newest-message lookups instead of materializing the
 * messages of every history. Opening the sidebar on 200 histories must not
 * load the messages table.
 *
 * The outcome tests below were captured against the pre-rewrite
 * `where('history_id').anyOf(ids).toArray()` implementation and must stay
 * green through the index rewrite. The mechanism test pins that no
 * messages-table materialization remains on the read path.
 *
 * Newest-message semantics: the previous code reduced over the messages in
 * primary-key scan order keeping strictly-greater `createdAt`, so a tie on
 * the maximum `createdAt` resolves to the FIRST row in primary-key order.
 * The rewrite preserves exactly that (compound-index `.last()` for the
 * maximum timestamp, then `.equals([history_id, createdAt]).first()` for the
 * lowest primary key inside the tie group).
 */
const memory = vi.hoisted(() => ({ db: null as unknown }))

vi.mock("@/db/dexie/schema", async () => {
  const { createMemoryDexie } = await import(
    "@/components/Option/Playground/__tests__/harness/memory-dexie"
  )
  memory.db = memory.db ?? createMemoryDexie()
  return { db: memory.db }
})

import { pageAssistDatabase } from "../chat"
import type { HistoryInfo, Message } from "../types"

const db = () => memory.db as MemoryDexie
const BASE_TS = 1_700_000_000_000

const history = (id: string, createdAt = BASE_TS): HistoryInfo => ({
  id,
  title: `Conversation ${id}`,
  is_rag: false,
  createdAt
})

const message = (
  id: string,
  history_id: string,
  createdAt: number,
  content = `content ${id}`
): Message => ({
  id,
  history_id,
  name: "Assistant",
  role: "assistant",
  content,
  createdAt
} as Message)

describe("history metadata (WP-26, index count + newest lookup)", () => {
  beforeEach(() => {
    db().resetAll()
  })

  it("getHistoryMetadata counts messages and reports the newest by createdAt, not by primary key", async () => {
    db().chatHistories.bulkPut([history("h1")])
    // Insertion/PK order deliberately disagrees with createdAt order.
    db().messages.bulkPut([
      message("m-a", "h1", BASE_TS + 300, "newest content"),
      message("m-b", "h1", BASE_TS + 100),
      message("m-c", "h1", BASE_TS + 200)
    ])

    const metadata = await pageAssistDatabase.getHistoryMetadata("h1")
    expect(metadata.messageCount).toBe(3)
    expect(metadata.lastMessage).toEqual({
      content: "newest content",
      role: "assistant",
      createdAt: BASE_TS + 300
    })
  })

  it("getHistoryMetadata resolves a createdAt tie to the first row in primary-key order", async () => {
    db().chatHistories.bulkPut([history("h1")])
    db().messages.bulkPut([
      message("m-x1", "h1", BASE_TS + 100, "tie one"),
      message("m-x2", "h1", BASE_TS + 100, "tie two"),
      message("m-older", "h1", BASE_TS + 50)
    ])

    const metadata = await pageAssistDatabase.getHistoryMetadata("h1")
    expect(metadata.messageCount).toBe(3)
    // Strictly-greater reduce over the PK-ordered scan keeps "m-x1".
    expect(metadata.lastMessage?.content).toBe("tie one")
  })

  it("getHistoryMetadata reports an empty history without a last message", async () => {
    db().chatHistories.bulkPut([history("empty")])
    expect(await pageAssistDatabase.getHistoryMetadata("empty")).toEqual({ messageCount: 0 })
  })

  it("getHistoryMetadata of an unknown history reports zero messages", async () => {
    expect(await pageAssistDatabase.getHistoryMetadata("missing")).toEqual({ messageCount: 0 })
  })

  it("getHistoriesWithMetadata returns an entry per requested id with per-history counts and newest messages", async () => {
    db().chatHistories.bulkPut([history("h1"), history("h2"), history("empty")])
    db().messages.bulkPut([
      message("h1-m1", "h1", BASE_TS + 10),
      message("h1-m2", "h1", BASE_TS + 20, "h1 newest"),
      message("h2-m1", "h2", BASE_TS + 5, "h2 newest")
    ])

    const metadata = await pageAssistDatabase.getHistoriesWithMetadata(["h1", "h2", "empty", "missing"])
    expect([...metadata.keys()].sort()).toEqual(["empty", "h1", "h2", "missing"])
    expect(metadata.get("h1")).toEqual({
      messageCount: 2,
      lastMessage: { content: "h1 newest", role: "assistant", createdAt: BASE_TS + 20 }
    })
    expect(metadata.get("h2")).toEqual({
      messageCount: 1,
      lastMessage: { content: "h2 newest", role: "assistant", createdAt: BASE_TS + 5 }
    })
    expect(metadata.get("empty")).toEqual({ messageCount: 0 })
    expect(metadata.get("missing")).toEqual({ messageCount: 0 })
  })

  it("getHistoriesWithMetadata mirrors the tie rule of getHistoryMetadata", async () => {
    db().chatHistories.bulkPut([history("h1")])
    db().messages.bulkPut([
      message("m-x1", "h1", BASE_TS + 100, "tie one"),
      message("m-x2", "h1", BASE_TS + 100, "tie two")
    ])

    const metadata = await pageAssistDatabase.getHistoriesWithMetadata(["h1"])
    expect(metadata.get("h1")?.lastMessage?.content).toBe("tie one")
  })

  it("handles an empty id list without touching storage", async () => {
    expect((await pageAssistDatabase.getHistoriesWithMetadata([])).size).toBe(0)
  })

  describe("index usage (mechanism, post-rewrite)", () => {
    it("never materializes the messages table while computing metadata for 200 histories", async () => {
      const histories = Array.from({ length: 200 }, (_, index) => history(`h-${String(index).padStart(3, "0")}`))
      db().chatHistories.bulkPut(histories)
      db().messages.bulkPut(
        histories.flatMap((h, index) =>
          Array.from({ length: 25 }, (_, m) => message(`${h.id}-m${m}`, h.id, BASE_TS + m))
        )
      )

      const toArray = vi.spyOn(db().messages, "toArray")
      toArray.mockImplementation(async () => {
        throw new Error("full-table materialization: messages.toArray() must not be used by metadata reads")
      })
      // The pre-rewrite shape batch-fetched every history's messages through
      // where('history_id').anyOf(...).toArray(); ban that batch fetch too.
      const originalWhere = db().messages.where.bind(db().messages)
      const where = vi.spyOn(db().messages, "where")
      where.mockImplementation(((index: unknown) => {
        const clause = originalWhere(index as string)
        if (index === "history_id") {
          const intercepted = Object.create(clause)
          intercepted.anyOf = () => {
            throw new Error("batch materialization: messages.where('history_id').anyOf() must not be used by metadata reads")
          }
          return intercepted
        }
        return clause
      }) as never)

      try {
        const ids = histories.map(h => h.id)
        const metadata = await pageAssistDatabase.getHistoriesWithMetadata(ids)
        expect(metadata.size).toBe(200)
        expect(metadata.get("h-000")).toEqual({
          messageCount: 25,
          lastMessage: { content: "content h-000-m24", role: "assistant", createdAt: BASE_TS + 24 }
        })
        const single = await pageAssistDatabase.getHistoryMetadata("h-199")
        expect(single.messageCount).toBe(25)
      } finally {
        toArray.mockRestore()
        where.mockRestore()
      }
    })
  })
})
