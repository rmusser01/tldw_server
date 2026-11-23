import { beforeEach, describe, expect, it, vi } from "vitest"
import type { MemoryDexie } from "@/components/Option/Playground/__tests__/harness/memory-dexie"

/**
 * TASK-15626: Settings > Data > Import Data restores chats through one
 * enclosing Dexie transaction (`importPageAssistData`), inside which
 * `importChatHistoryV2` opens its own nested transaction. Dexie rejects a
 * nested transaction that names a table its parent does not hold
 * ("Table X not included in parent transaction."), before any row is read
 * or written and whatever the row count.
 *
 * jsdom has no IndexedDB and the repo ships no fake-indexeddb, so the real
 * import, token-index and search code runs on the in-memory Dexie stand-in.
 * The stand-in lets a nested transaction join any parent, which is how the
 * v17 `searchTokens` gap went unnoticed; this file adds Dexie 4's
 * parent-scope check (`Dexie.prototype._transaction`) on top of it.
 */
const memory = vi.hoisted(() => ({ db: null as unknown }))

vi.mock("@/db/dexie/schema", async () => {
  const { AsyncLocalStorage } = await import("node:async_hooks")
  const { createMemoryDexie } = await import(
    "@/components/Option/Playground/__tests__/harness/memory-dexie"
  )
  const db = createMemoryDexie()
  const run = db.transaction
  const parentScope = new AsyncLocalStorage<Set<string>>()
  db.transaction = (...args: unknown[]) => {
    const operation = args.at(-1) as (tx: unknown) => Promise<unknown>
    const tables = args
      .slice(1, -1)
      .flat()
      .map((table) => (typeof table === "string" ? table : (table as { name: string }).name))
    const parent = parentScope.getStore()
    const missing = parent && tables.find((table) => !parent.has(table))
    if (missing) {
      const error = new Error(`Table ${missing} not included in parent transaction.`)
      error.name = "SubTransactionError"
      return Promise.reject(error)
    }
    return run(...args.slice(0, -1), (tx: unknown) =>
      parentScope.run(parent ?? new Set(tables), () => operation(tx))
    )
  }
  memory.db = db
  return { db }
})
// Only the deprecated pre-Dexie restore imports from the legacy barrel.
vi.mock("@/db", () => ({ importChatHistory: vi.fn(), importPrompts: vi.fn() }))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({
  resolveRecipePersistenceOwnerView: async () => ({ ownerId: "local-owner" }),
  clearRecipePersistenceScoped: vi.fn()
}))

import { importPageAssistData } from "../export-import"
import { pageAssistDatabase } from "@/db/dexie/chat"

const db = () => memory.db as MemoryDexie
const BASE_TS = 1_700_000_000_000

/** Same shape as e2e/local-chat-import.spec.ts. */
const chatBackup = (historyId: string, messages: Array<Record<string, unknown>>) => ({
  history: {
    id: historyId,
    title: "Local import transaction regression",
    is_rag: false,
    createdAt: BASE_TS,
    message_source: "web-ui"
  },
  messages
})

const message = (historyId: string, index: number, content: string) => ({
  id: `${historyId}-m${index}`,
  history_id: historyId,
  name: index % 2 === 0 ? "You" : "Assistant",
  role: index % 2 === 0 ? "user" : "assistant",
  content,
  createdAt: BASE_TS + index,
  parent_message_id: index === 0 ? null : `${historyId}-m${index - 1}`
})

/** 60 user/assistant pairs; assistant turns are multi-paragraph markdown. */
const longChat = (historyId: string) =>
  chatBackup(
    historyId,
    Array.from({ length: 120 }, (_, index) =>
      message(
        historyId,
        index,
        index % 2 === 0
          ? `Question ${index / 2}: how does step ${index / 2} work?`
          : [
              `## Step ${(index - 1) / 2}`,
              "",
              "Here is a **detailed** answer with a list:",
              "",
              "- first point",
              "- second point",
              "",
              index === 117 ? "The zephyrine detail appears only here." : "A closing paragraph."
            ].join("\n")
      )
    )
  )

const importFile = (data: unknown) =>
  importPageAssistData(
    new File([JSON.stringify(data)], "backup.json", { type: "application/json" })
  )

describe("Import Data restores local chats in one transaction", () => {
  beforeEach(() => db().resetAll())

  it("imports the one-message backup the e2e spec uses", async () => {
    await importFile({
      chat: [
        chatBackup("short", [
          message("short", 0, "Local import input; no answer generated.")
        ])
      ]
    })

    expect(await db().chatHistories.get("short")).toBeTruthy()
    expect(await db().messages.where("history_id").equals("short").count()).toBe(1)
  })

  it("imports a 120-message chat and indexes it for content search", async () => {
    await importFile({ chat: [longChat("long")] })

    expect(await db().messages.where("history_id").equals("long").count()).toBe(120)
    const found = await pageAssistDatabase.fullTextSearchChatHistories("zephyrine")
    expect(found.map((history) => history.id)).toEqual(["long"])
  })

  it("leaves no history, message or token behind when a row fails", async () => {
    const broken = chatBackup("broken", [{ ...message("broken", 0, "orphan words"), id: undefined }])

    await expect(importFile({ chat: [longChat("long"), broken] })).rejects.toThrow(/DataError/)

    expect(await db().chatHistories.count()).toBe(0)
    expect(await db().messages.count()).toBe(0)
    expect(await db().searchTokens.count()).toBe(0)
  })
})
