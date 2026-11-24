import { describe, expect, it } from "vitest"
import { createMemoryDexie } from "./memory-dexie"

// Dexie 4 rejects a nested transaction that does not fit its immediate parent
// (TASK-15626); the harness must too, or out-of-scope nesting passes unit tests.
describe("memory Dexie nested transaction scope", () => {
  it("rejects a child naming a table its parent lacks", async () => {
    const db = createMemoryDexie()
    const outcome = db.transaction("rw", [db.messages], () =>
      db.transaction("rw", [db.messages, db.searchTokens], async () => undefined)
    )
    await expect(outcome).rejects.toMatchObject({
      name: "SubTransactionError",
      message: "Table searchTokens not included in parent transaction."
    })
  })

  it("checks the immediate parent, not the outermost one", async () => {
    const db = createMemoryDexie()
    const outcome = db.transaction("rw", [db.messages, db.searchTokens], () =>
      db.transaction("rw", [db.messages], () =>
        db.transaction("rw", [db.searchTokens], async () => undefined)
      )
    )
    await expect(outcome).rejects.toThrow("Table searchTokens not included in parent transaction.")
  })

  it("lets a child use a subset of its parent's tables", async () => {
    const db = createMemoryDexie()
    await db.transaction("rw", [db.messages, db.searchTokens], () =>
      db.transaction("rw", ["messages"], () => db.messages.put({ id: "m", history_id: "h" }))
    )
    expect(await db.messages.count()).toBe(1)
  })

  it("rejects a read-write child of a read-only parent", async () => {
    const db = createMemoryDexie()
    const outcome = db.transaction("r", [db.messages], () =>
      db.transaction("rw", [db.messages], async () => undefined)
    )
    await expect(outcome).rejects.toMatchObject({
      name: "SubTransactionError",
      message: "Cannot enter a sub-transaction with READWRITE mode when parent transaction is READONLY"
    })
  })
})
