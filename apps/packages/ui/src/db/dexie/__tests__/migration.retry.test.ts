import Dexie from "dexie"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { db } from "../schema"
import { PageAssistDatabase } from "../chat"
import { exportChatHistory } from "../helpers"
import { getLocalHistoryOwner } from "../history-selection"
import { withLegacyChatStorageLock } from "../legacy-chat-storage"
import { DatabaseMigration, getAllSessionIds, runAllMigrations } from "../migration"

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
  let lock = Promise.resolve<unknown>(undefined)
  vi.stubGlobal("navigator", Object.create(navigator, { locks: { value: {
    request<T>(_name: string, operation: () => Promise<T>) {
      const result = lock.then(operation)
      lock = result.catch(() => {})
      return result
    }
  } } }))
  vi.stubGlobal("chrome", { runtime: { id: "migration-test-extension" }, storage: { sync: { get: async () => ({}) }, local: {
    get(key: string | null, callback: (value: Record<string, unknown>) => void) {
      callback(key === null ? source : { [key]: source[key] })
    },
    clear(done?: () => void) {
      source = {}
      done?.()
    },
    remove(keys: string | string[], done?: () => void) {
      for (const key of Array.isArray(keys) ? keys : [keys]) delete source[key]
      done?.()
    },
    set(values: Record<string, unknown>, done?: () => void) {
      Object.assign(source, values)
      done?.()
    }
  } } })
})
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

it("rolls back a failed history, retains its source and retries without overwriting edits", async () => {
  source.old = [message, { ...message, id: "m2" }]
  const read = vi.spyOn(db.messages, "bulkPut").mockRejectedValueOnce(new Error("Quota exceeded"))
  expect((await new DatabaseMigration().migrateAllData()).errors).toEqual([
    "Failed to migrate chat history old: Error: Quota exceeded"
  ])
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
  expect((await new DatabaseMigration().migrateAllData()).errors).toEqual([
    "Failed to migrate chat history old: Error: Cannot add legacy messages to a claimed history"
  ])
  expect((await db.chatHistories.get("old"))?.title).toBe("Claimed")
  expect(await db.messages.count()).toBe(0)
  expect(source.old).toEqual([message])
})

it("rejects a message linked to a different history", async () => {
  source.old = [{ ...message, history_id: "other" }]
  expect((await new DatabaseMigration().migrateAllData()).errors).toEqual([
    "Failed to migrate chat history old: Error: Message belongs to a different history"
  ])
  expect(await db.messages.count()).toBe(0)
})

it("rolls back the selectable history when its attachments fail and can retry it", async () => {
  vi.spyOn(db.sessionFiles, "put").mockRejectedValueOnce(new Error("Quota exceeded"))
  await expect(runAllMigrations()).rejects.toThrow("Local database migration did not finish")
  await expect(getLocalHistoryOwner("old")).rejects.toThrow("missing_conversation")
  expect(await db.messages.get("m")).toBeUndefined()
  expect(source).toMatchObject({
    chatHistories: [history],
    old: [message],
    session_files_old: { sessionId: "old", files: [{ id: "f" }] }
  })

  await runAllMigrations()
  await expect(getLocalHistoryOwner("old")).resolves.toMatchObject({ conversation_id: "old" })
  expect((await db.sessionFiles.get("old"))?.files).toHaveLength(1)
})

it("propagates an orphan session-file write failure and can retry it", async () => {
  source.chatHistories = []
  vi.spyOn(db.sessionFiles, "put").mockRejectedValueOnce(new Error("Quota exceeded"))
  await expect(runAllMigrations()).rejects.toThrow("Session file migration did not finish")
  await runAllMigrations()
  expect((await db.sessionFiles.get("old"))?.files).toHaveLength(1)
})

it("treats a browser storage read failure as a retryable error", async () => {
  // Chrome reports storage failures through lastError inside the callback.
  vi.stubGlobal("chrome", { runtime: { id: "migration-test-extension" }, storage: { ...chrome.storage, local: {
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
  expect((await new DatabaseMigration().migrateAllData()).errors).toEqual([
    "Failed to migrate chat history old: Error: Storage inaccessible"
  ])
  await expect(runAllMigrations()).rejects.toThrow("Local database migration did not finish")
  expect(await db.chatHistories.get("old")).toBeUndefined()
})

it("does not attach session files to an existing claimed chat", async () => {
  source.chatHistories = []
  await db.chatHistories.put({ ...history, local_owner_key: "claimed" })
  await expect(runAllMigrations()).rejects.toThrow("Session file migration did not finish")
  expect(await db.sessionFiles.get("old")).toBeUndefined()
  expect(source.session_files_old).toBeDefined()
})

it("erases the retained source of a claimed chat when another migration record failed", async () => {
  const unrelated = { ...history, id: "bad" }
  source.chatHistories = [history, unrelated]
  source.bad = [message]
  await expect(runAllMigrations()).rejects.toThrow("Local database migration did not finish")
  await getLocalHistoryOwner("old")
  expect(await db.searchTokens.get(["original", "old"])).toBeDefined()
  const unrelatedToken = { token: "keep", history_id: "unrelated" }
  await db.searchTokens.bulkPut([unrelatedToken, { token: "stale", history_id: "old" }])

  await new PageAssistDatabase().deleteChatHistory("old")

  expect(await db.searchTokens.toArray()).toEqual([unrelatedToken])
  expect(source.chatHistories).toEqual([unrelated])
  expect(source.old).toBeUndefined()
  expect(source.session_files_old).toBeUndefined()
  expect(await db.sessionFiles.get("old")).toBeUndefined()
  await expect(runAllMigrations()).rejects.toThrow("Local database migration did not finish")
  expect(await db.chatHistories.get("old")).toBeUndefined()
  expect(await db.messages.get("m")).toBeUndefined()
})

it("does not erase a legacy chat that shares the ID of a scoped local row", async () => {
  await db.chatHistories.put({
    ...history, title: "Other account", local_owner_key: "local-history-v1:other-account"
  })
  await new PageAssistDatabase().deleteChatHistory("old")
  expect(source.chatHistories).toEqual([history])
  expect(source.old).toEqual([message])
  expect(source.session_files_old).toBeDefined()
  expect(await db.chatHistories.get("old")).toBeUndefined()
})

it("erases uncopied chats and orphan messages and files without removing other browser data", async () => {
  source.orphan = [{ ...message, history_id: "orphan" }]
  source.unrelated = [{ ...message, history_id: "different-key" }]
  source.session_files_orphan = { sessionId: "orphan", files: [{ id: "orphan-file" }] }
  await db.searchTokens.put({ token: "stale", history_id: "orphan" })
  await new PageAssistDatabase().deleteAllChatHistory()
  await runAllMigrations()

  expect(await db.searchTokens.count()).toBe(0)
  expect(source.old).toBeUndefined()
  expect(source.session_files_old).toBeUndefined()
  expect(source.session_files_orphan).toBeUndefined()
  expect(source.orphan).toBeUndefined()
  expect(source.unrelated).toEqual([{ ...message, history_id: "different-key" }])
  expect(source.prompts).toBeDefined()
  expect(source.webshares).toBeDefined()
  expect(await db.chatHistories.count()).toBe(0)
  expect(await db.messages.count()).toBe(0)
  expect(await db.sessionFiles.count()).toBe(0)
})

it("cannot reinsert a stale browser snapshot after the all-history erase", async () => {
  let release!: () => void
  let snapshotReady!: () => void
  const ready = new Promise<void>((resolve) => { snapshotReady = resolve })
  const get = chrome.storage.local.get
  vi.spyOn(chrome.storage.local, "get").mockImplementation(((key: string | null, callback: (data: Record<string, unknown>) => void) => {
    if (key === "old") {
      const snapshot = { old: source.old }
      release = () => callback(snapshot)
      snapshotReady()
    } else {
      Reflect.apply(get, chrome.storage.local, [key, callback])
    }
  }) as typeof get)

  const migrating = runAllMigrations()
  await ready
  const erasing = new PageAssistDatabase().deleteAllChatHistory()
  try {
    // A correctly serialized erase has to wait for the pending source read.
    await expect(Promise.race([erasing.then(() => true), new Promise((resolve) => setTimeout(() => resolve(false), 0))])).resolves.toBe(false)
  } finally {
    release()
    await Promise.all([migrating, erasing])
  }
  expect(await db.chatHistories.count()).toBe(0)
  expect(await db.sessionFiles.count()).toBe(0)
})

it("awaits Chrome erasure and keeps the copied chat when removal fails", async () => {
  await runAllMigrations()
  let reportFailure!: () => void
  let removeStarted!: () => void
  const started = new Promise<void>((resolve) => { removeStarted = resolve })
  const remove = vi.spyOn(chrome.storage.local, "remove").mockImplementationOnce(((_keys: string[], done: () => void) => {
    reportFailure = () => {
      Object.assign(chrome.runtime, { lastError: { message: "Storage inaccessible" } })
      done()
      Reflect.deleteProperty(chrome.runtime, "lastError")
    }
    removeStarted()
  }) as typeof chrome.storage.local.remove)

  const database = new PageAssistDatabase()
  let completed = false
  const erasing = database.deleteChatHistory("old").finally(() => { completed = true })
  const failure = expect(erasing).rejects.toThrow("Storage inaccessible")
  await started
  expect(completed).toBe(false)
  reportFailure()
  await failure
  expect(source.old).toEqual([message])
  expect(await db.chatHistories.get("old")).toBeDefined()
  expect(await db.messages.get("m")).toBeDefined()

  remove.mockRestore()
  await database.deleteChatHistory("old")
  expect(source.old).toBeUndefined()
  expect(await db.chatHistories.get("old")).toBeUndefined()
})

it("keeps replacement legacy content even when its ID and catalog metadata match a copy", async () => {
  await runAllMigrations()
  const replacement = { ...message, content: "Different account" }
  source.old = [replacement]
  await new PageAssistDatabase().deleteChatHistory("old")
  expect(source.old).toEqual([replacement])
  expect(source.session_files_old).toBeDefined()
})

it("strips migration provenance from archive exports and imports", async () => {
  await runAllMigrations()
  const copied = await db.chatHistories.get("old")
  expect(copied?.tldw_history_legacy_source).toBeDefined()
  expect((await exportChatHistory())[0].history).not.toHaveProperty("tldw_history_legacy_source")

  await db.chatHistories.delete("old")
  const database = new PageAssistDatabase()
  await database.importChatHistoryV2([{ history: copied, messages: [] }])
  expect((await db.chatHistories.get("old"))?.tldw_history_legacy_source).toBeUndefined()
  await database.deleteChatHistory("old")
  expect(source.old).toEqual([message])
})

it("preserves the first repeated legacy message", async () => {
  source.old = [message, { ...message, content: "Second copy" }]
  await runAllMigrations()
  expect(await db.messages.get("m")).toMatchObject({ content: "Original" })
})

it("preserves an empty non-history Chrome key that collides with a legacy chat ID", async () => {
  source.chatHistories = [{ ...history, id: "prompts" }]
  source.prompts = []
  await runAllMigrations()

  await new PageAssistDatabase().deleteChatHistory("prompts")
  expect(source.chatHistories).toEqual([])
  expect(source.prompts).toEqual([])
})

it("does not claim a different history's message payload as session files", async () => {
  const colliding = { ...history, id: "session_files_old" }
  source.chatHistories = [history, colliding]
  source.session_files_old = [{ ...message, id: "different", history_id: colliding.id }]

  await expect(runAllMigrations()).rejects.toThrow("Local database migration did not finish")
  expect(await db.chatHistories.get("old")).toBeUndefined()
  expect(source.session_files_old).toEqual([{ ...message, id: "different", history_id: colliding.id }])
  expect(await db.messages.get("different")).toBeDefined()
})

it("cannot restore a stale legacy webshare after the device reset", async () => {
  let release!: () => void
  let snapshotReady!: () => void
  const ready = new Promise<void>((resolve) => { snapshotReady = resolve })
  const get = chrome.storage.local.get
  vi.spyOn(chrome.storage.local, "get").mockImplementation(((key: string | null, callback: (data: Record<string, unknown>) => void) => {
    if (key === "webshares" && !release) {
      const snapshot = { webshares: source.webshares }
      release = () => callback(snapshot)
      snapshotReady()
    } else {
      Reflect.apply(get, chrome.storage.local, [key, callback])
    }
  }) as typeof get)

  const migrating = runAllMigrations()
  await ready
  const resetting = withLegacyChatStorageLock(async (legacy) => {
    await legacy?.clear()
    await new PageAssistDatabase().clearDB()
  })
  try {
    await expect(Promise.race([resetting.then(() => true), new Promise((resolve) => setTimeout(() => resolve(false), 0))])).resolves.toBe(false)
  } finally {
    release()
    await Promise.all([migrating, resetting])
  }
  expect(await db.webshares.count()).toBe(0)
  expect(await db.chatHistories.count()).toBe(0)
})

it.each(["messages", "files"])("erases the source after legacy %s are merged into an edited unclaimed chat", async (merged) => {
  await db.chatHistories.put({ ...history, title: "Edited" })
  if (merged === "files") await db.messages.put({ ...message, content: "Edited" })
  else delete source.session_files_old
  await runAllMigrations()
  await getLocalHistoryOwner("old")

  await new PageAssistDatabase().deleteChatHistory("old")
  await runAllMigrations()

  expect(source.old).toBeUndefined()
  expect(source.session_files_old).toBeUndefined()
  expect(await db.chatHistories.get("old")).toBeUndefined()
  expect(await db.sessionFiles.get("old")).toBeUndefined()
})

it("does not infer provenance when a legacy merge adds no data to an unclaimed same-ID chat", async () => {
  await db.chatHistories.put({ ...history, title: "Different chat" })
  await db.messages.put({ ...message, content: "Different message" })
  delete source.session_files_old
  await runAllMigrations()

  await new PageAssistDatabase().deleteChatHistory("old")
  expect(source.old).toEqual([message])
  expect(source.chatHistories).toEqual([history])
})

it("does not treat a prefixed catalog ID's messages as orphan files", async () => {
  const prefixed = { ...history, id: "session_files_other" }
  source.chatHistories = [prefixed]
  source.session_files_other = [{ ...message, history_id: prefixed.id }]
  delete source.session_files_old

  await runAllMigrations()
  expect(await db.messages.get("m")).toMatchObject({ history_id: prefixed.id })
  expect(await db.sessionFiles.count()).toBe(0)
})

it("retains conflicting catalog rows that share one legacy ID without publishing either", async () => {
  const duplicate = { ...history, title: "Different source" }
  source.chatHistories = [history, duplicate]
  await expect(runAllMigrations()).rejects.toThrow("Local database migration did not finish")
  expect(await db.chatHistories.get("old")).toBeUndefined()
  expect(source.chatHistories).toEqual([history, duplicate])
})

it("retains a legacy attachment envelope that names a different session", async () => {
  source.session_files_old = { sessionId: "other", files: [{ id: "f" }] }
  await expect(runAllMigrations()).rejects.toThrow("Local database migration did not finish")
  expect(await db.chatHistories.get("old")).toBeUndefined()
  expect(source.session_files_old).toEqual({ sessionId: "other", files: [{ id: "f" }] })
})

it("keeps the legacy lock until Chrome settles after Dexie aborts the erase", async () => {
  await runAllMigrations()
  let release!: () => void
  let removeStarted!: () => void
  const started = new Promise<void>((resolve) => { removeStarted = resolve })
  const remove = chrome.storage.local.remove
  vi.spyOn(chrome.storage.local, "remove").mockImplementationOnce(((keys: string[], done: () => void) => {
    release = () => Reflect.apply(remove, chrome.storage.local, [keys, done])
    removeStarted()
  }) as typeof remove)
  vi.spyOn(Dexie, "waitFor").mockRejectedValueOnce(new Error("Dexie transaction timed out"))

  const erasing = new PageAssistDatabase().deleteChatHistory("old")
  const failure = expect(erasing).rejects.toThrow("Dexie transaction timed out")
  await started
  const next = withLegacyChatStorageLock(async () => true)
  try {
    await expect(Promise.race([next, new Promise((resolve) => setTimeout(() => resolve(false), 0))])).resolves.toBe(false)
  } finally {
    release()
    await failure
    await next
  }
  expect(await db.chatHistories.get("old")).toBeDefined()
  expect(source.old).toBeUndefined()
})

it("settles a Chrome session scan when the catalog contains a corrupt entry", async () => {
  const corruptCatalog = { ...source, chatHistories: [null, history] }
  const callbackError = new Promise<unknown>((resolve) => {
    vi.spyOn(chrome.storage.local, "get").mockImplementation(((_key: null, callback: (data: Record<string, unknown>) => void) => {
      queueMicrotask(() => {
        try { callback(corruptCatalog) } catch (error) { resolve(error) }
      })
    }) as typeof chrome.storage.local.get)
  })
  await expect(Promise.race([getAllSessionIds(), callbackError])).resolves.toEqual(["old"])
})
