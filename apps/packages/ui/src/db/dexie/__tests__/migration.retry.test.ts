import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { db } from "../schema"
import { runAllMigrations } from "../migration"

vi.mock("../schema", async () => {
  const { createMemoryDexie } = await import("@/components/Option/Playground/__tests__/harness/memory-dexie")
  return { db: createMemoryDexie() }
})
vi.mock("@/services/model-settings", () => ({
  getLastUsedChatModel: async () => undefined,
  getLastUsedChatSystemPrompt: async () => undefined
}))

const history = { id: "old", title: "Original", createdAt: 1, is_rag: false }
const message = { id: "m", history_id: "old", name: "Me", content: "Original", role: "user", createdAt: 1 }
let source: Record<string, unknown>

beforeEach(async () => {
  await db.delete()
  source = {
    chatHistories: [history], old: [message], user_id: "old profile",
    prompts: [{ id: "p", title: "Original", serverId: 7 }],
    webshares: [{ id: "w", title: "Original", createdAt: 1 }],
    session_files_old: { sessionId: "old", files: [{ id: "f" }], retrievalEnabled: true },
  }
  vi.stubGlobal("chrome", { storage: { local: {
    get(key: string | null, callback: (value: Record<string, unknown>) => void) {
      callback(key === null ? source : { [key]: source[key] })
    },
    remove(key: string) { delete source[key] },
    set(values: Record<string, unknown>) { Object.assign(source, values) }
  } } })
})
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

it("rolls back a failed history, retains its source and retries without overwriting edits", async () => {
  source.old = [message, { ...message, id: "m2" }]
  const read = vi.spyOn(db.messages, "bulkPut").mockRejectedValueOnce(new Error("Quota exceeded"))
  await expect(runAllMigrations()).rejects.toThrow()
  expect(source.chatHistories).toEqual([history])
  expect(source.old).toHaveLength(2)
  expect(await db.chatHistories.get(history.id)).toBeUndefined()
  expect(await db.messages.count()).toBe(0)
  read.mockRestore()

  await db.chatHistories.put({ ...history, title: "Edited" })
  await db.messages.put({ ...message, content: "Edited" })
  await db.prompts.update("p", { title: "Edited" })
  await db.webshares.update("w", { title: "Edited" })
  await db.userSettings.put({ id: "main", user_id: "current profile" })
  await Promise.all([runAllMigrations(), runAllMigrations()])

  expect(await db.chatHistories.get("old")).toMatchObject({ title: "Edited" })
  expect(await db.messages.get("m")).toMatchObject({ content: "Edited" })
  expect(await db.messages.count()).toBe(2)
  expect(await db.prompts.get("p")).toMatchObject({ title: "Edited", serverId: 7 })
  expect((await db.prompts.get("p"))?.ownerId).toBeUndefined()
  expect(await db.webshares.get("w")).toMatchObject({ title: "Edited" })
  expect(await db.userSettings.get("main")).toMatchObject({ user_id: "current profile" })
  expect((await db.sessionFiles.get("old"))?.files).toHaveLength(1)
})

it("does not attach legacy messages to another destination owner", async () => {
  await db.chatHistories.put({ ...history, title: "Claimed", local_owner_key: "owner::x" })
  await expect(runAllMigrations()).rejects.toThrow()
  expect((await db.chatHistories.get("old"))?.title).toBe("Claimed")
  expect(await db.messages.count()).toBe(0)
  expect(source.old).toEqual([message])
})

it("rejects a message linked to a different history", async () => {
  source.old = [{ ...message, history_id: "other" }]
  await expect(runAllMigrations()).rejects.toThrow()
  expect(await db.messages.count()).toBe(0)
})

it("propagates a session-file write failure and can retry it", async () => {
  vi.spyOn(db.sessionFiles, "put").mockRejectedValueOnce(new Error("Quota exceeded"))
  await expect(runAllMigrations()).rejects.toThrow()
  await runAllMigrations()
  expect((await db.sessionFiles.get("old"))?.files).toHaveLength(1)
})

it("treats a browser storage read failure as a retryable error", async () => {
  // Chrome reports storage failures through lastError inside the callback.
  vi.stubGlobal("chrome", { runtime: {}, storage: { local: {
    get(key: string | null, callback: (data: Record<string, unknown>) => void) {
      if (key === history.id) {
        Object.assign(chrome.runtime, { lastError: { message: "Storage inaccessible" } })
        callback({})
        Reflect.deleteProperty(chrome.runtime, "lastError")
      } else {
        callback(key === null ? source : { [key]: source[key] })
      }
    }
  } } })
  await expect(runAllMigrations()).rejects.toThrow()
  expect(await db.chatHistories.get("old")).toBeUndefined()
})

it("does not attach session files to an existing claimed chat", async () => {
  source.chatHistories = []
  await db.chatHistories.put({ ...history, local_owner_key: "claimed" })
  await expect(runAllMigrations()).rejects.toThrow()
  expect(await db.sessionFiles.get("old")).toBeUndefined()
  expect(source.session_files_old).toBeDefined()
})
