import { createRequire } from "node:module"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { WORKSPACE_STORAGE_KEY } from "@/store/workspace-events"

// Resolve the installed frontend test dependency from its owning package.
const { IDBFactory, IDBObjectStore, forceCloseDatabase } = createRequire(
  resolve(dirname(fileURLToPath(import.meta.url)), "../../../../../tldw-frontend/package.json")
)("fake-indexeddb") as {
  IDBFactory: new () => IDBFactory
  IDBObjectStore: { prototype: IDBObjectStore }
  forceCloseDatabase: (database: IDBDatabase) => void
}

const workspaceId = "offloaded"
const malformedRaw = '{ "state": { "notes": "retained draft" },\n'
const snapshotKey = `${WORKSPACE_STORAGE_KEY}:workspace:${workspaceId}:snapshot`
const chatKey = `${WORKSPACE_STORAGE_KEY}:workspace:${workspaceId}:chat`
const chatMessage = "Retained chat message ".repeat(1200)
const artifactContent = "Retained artifact content ".repeat(1200)
const chatSession = {
  messages: [{ isBot: false, name: "You", message: chatMessage, sources: [] }],
  historyId: "history-retained",
  serverChatId: "chat-retained"
}

const captureRawStorage = () =>
  Object.fromEntries(
    Array.from({ length: localStorage.length }, (_, index) => {
      const key = localStorage.key(index)!
      return [key, localStorage.getItem(key)]
    })
  )

const fixtureState = (kind: "chat" | "artifact") => ({
  workspaceId,
  savedWorkspaces: [],
  archivedWorkspaces: [],
  workspaceCollections: [],
  workspaceSnapshots: {
    [workspaceId]: {
      workspaceId,
      workspaceName: "Retained workspace",
      workspaceTag: "workspace:offloaded",
      workspaceCreatedAt: "2026-09-29T00:00:00.000Z",
      sources: [],
      selectedSourceIds: [],
      generatedArtifacts: kind === "artifact" ? [{
        id: "artifact-1",
        type: "summary",
        title: "Retained artifact",
        status: "completed",
        content: artifactContent,
        data: { retained: true },
        createdAt: "2026-09-29T00:00:00.000Z"
      }] : [],
      notes: "Retained notes"
    }
  },
  workspaceChatSessions: kind === "chat" ? {
    [workspaceId]: chatSession
  } : {}
})

const seedOffload = async (kind: "chat" | "artifact") => {
  const workspaceModule = await import("../workspace")
  const storage = workspaceModule.createWorkspaceStorage()
  await storage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify({
    state: fixtureState(kind), version: 1
  }))
  if (kind === "chat") {
    expect(JSON.parse(localStorage.getItem(chatKey)!).offloadType).toBe(
      "workspace_chat_session_v1"
    )
  } else {
    expect(JSON.parse(localStorage.getItem(snapshotKey)!).generatedArtifacts[0])
      .toHaveProperty("__tldwArtifactPayloadRef")
  }
  return { ...workspaceModule, storage }
}

const readOffloadRecord = (database: IDBDatabase, kind: "chat" | "artifact") =>
  new Promise<unknown>((resolve, reject) => {
    const storeName = kind === "chat" ? "workspace-chat-sessions" : "workspace-artifact-payloads"
    const recordKey = kind === "chat"
      ? `workspace:${workspaceId}:chat`
      : `workspace:${workspaceId}:artifact:artifact-1`
    const transaction = database.transaction(storeName)
    const request = transaction.objectStore(storeName).get(recordKey)
    transaction.oncomplete = () => resolve(request.result)
    transaction.onabort = () => reject(transaction.error)
    transaction.onerror = () => reject(transaction.error)
  })

describe("workspace hydration with the production IndexedDB adapter", () => {
  beforeEach(() => {
    vi.resetModules()
    vi.stubGlobal("indexedDB", new IDBFactory())
    localStorage.clear()
    localStorage.setItem("tldw:feature-rollout:workspace_split_storage_v1:enabled", "1")
    localStorage.setItem("tldw:feature-rollout:workspace_indexeddb_offload_v1:enabled", "1")
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    localStorage.clear()
  })

  it.each([false, true])("invalidates an admitted write awaiting database open (repaired retry: %s)", async (repair) => {
    const originalOpen = indexedDB.open.bind(indexedDB)
    let releaseOpen!: () => void
    const opened = new Promise<void>((resolveOpened) => {
      vi.spyOn(indexedDB, "open").mockImplementation((...args) => {
        const request = originalOpen(...args)
        Object.defineProperty(request, "onsuccess", {
          set(handler) {
            request.addEventListener("success", (event) => {
              releaseOpen = () => handler.call(request, event)
              resolveOpened()
            })
          }
        })
        return request
      })
    })
    const put = vi.spyOn(IDBObjectStore.prototype, "put")
    const { useWorkspaceStore: store } = await import("../workspace")
    const pendingWrite = store.setState({
      workspaceId,
      workspaceChatSessions: { [workspaceId]: { ...chatSession, history: [] } }
    })
    await opened
    localStorage.setItem(WORKSPACE_STORAGE_KEY, malformedRaw)
    const before = captureRawStorage()

    await store.persist.rehydrate()
    expect(store.getState().storeHydrationError).toEqual(expect.any(String))
    if (repair) {
      localStorage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify({
        version: 1,
        state: { workspaceId: "repaired", workspaceSnapshots: {}, workspaceChatSessions: {} }
      }))
      await store.persist.rehydrate()
      expect(store.getState().storeHydrated).toBe(true)
    }
    releaseOpen()
    await pendingWrite

    if (repair) {
      await store.setState({})
      expect(JSON.parse(localStorage.getItem(WORKSPACE_STORAGE_KEY)!).state.workspaceId)
        .toBe("repaired")
    } else {
      expect(captureRawStorage()).toEqual(before)
      expect(store.getState().storeHydrated).toBe(false)
    }
    expect(put).not.toHaveBeenCalled()
  })

  it("aborts an already queued IndexedDB mutation when a failing retry starts", async () => {
    const originalPut = IDBObjectStore.prototype.put
    let retry: ReturnType<typeof store.persist.rehydrate> | undefined
    let before: ReturnType<typeof captureRawStorage> | undefined
    const { useWorkspaceStore: store } = await import("../workspace")
    vi.spyOn(IDBObjectStore.prototype, "put").mockImplementation(function (...args) {
      const request = originalPut.apply(this, args)
      localStorage.setItem(WORKSPACE_STORAGE_KEY, malformedRaw)
      before = captureRawStorage()
      retry = store.persist.rehydrate()
      return request
    })

    await store.setState({
      workspaceId,
      workspaceChatSessions: { [workspaceId]: { ...chatSession, history: [] } }
    })
    await retry

    expect(before).toBeDefined()
    expect(captureRawStorage()).toEqual(before)
    const database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open("tldw-workspace-storage", 1)
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
    const record = await new Promise((resolve, reject) => {
      const request = database.transaction("workspace-chat-sessions")
        .objectStore("workspace-chat-sessions").get(`workspace:${workspaceId}:chat`)
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
    database.close()
    expect(record).toBeUndefined()
    expect(store.getState().storeHydrated).toBe(false)
  })

  describe.each(["chat", "artifact"] as const)("persisted %s references", (kind) => {
    it.each(["abnormal", "silent"] as const)("recovers the identical offload record after a closed connection (%s)", async (closeKind) => {
      const connections: IDBDatabase[] = []
      const originalOpen = indexedDB.open.bind(indexedDB)
      vi.spyOn(indexedDB, "open").mockImplementation((...args) => {
        const request = originalOpen(...args)
        request.addEventListener("success", () => connections.push(request.result))
        return request
      })
      const { useWorkspaceStore: store, storage } = await seedOffload(kind)
      const database = connections[0]
      const originalRecord = await readOffloadRecord(database, kind)
      const before = captureRawStorage()
      expect(originalRecord).toBeDefined()

      if (closeKind === "abnormal") {
        const closed = new Promise<void>((resolve) => {
          database.addEventListener("close", () => resolve(), { once: true })
        })
        forceCloseDatabase(database)
        await closed
      } else {
        database.close()
        await store.persist.rehydrate()
        expect(store.getState()).toMatchObject({
          storeHydrated: false, storeHydrationError: expect.any(String)
        })
        expect(captureRawStorage()).toEqual(before)
      }

      const reconstructed = JSON.parse((await storage.getItem(WORKSPACE_STORAGE_KEY))!).state
      expect(captureRawStorage()).toEqual(before)
      expect(connections).toHaveLength(2)
      expect(await readOffloadRecord(connections[1], kind)).toEqual(originalRecord)
      if (kind === "chat") {
        expect(reconstructed.workspaceChatSessions[workspaceId].messages[0].message).toBe(chatMessage)
      } else {
        expect(reconstructed.workspaceSnapshots[workspaceId].generatedArtifacts[0]).toMatchObject({
          content: artifactContent, data: { retained: true }
        })
      }

      await store.persist.rehydrate()
      expect(store.getState()).toMatchObject({ storeHydrated: true, storeHydrationError: null })
      expect(connections).toHaveLength(2)
      if (kind === "chat") {
        expect(store.getState().workspaceChatSessions[workspaceId].messages[0].message)
          .toBe(chatMessage)
      } else {
        expect(store.getState().generatedArtifacts[0]).toMatchObject({
          content: artifactContent, data: { retained: true }
        })
      }
    })

    it("does not let stale close handlers invalidate the replacement connection", async () => {
      const connections: IDBDatabase[] = []
      const originalOpen = indexedDB.open.bind(indexedDB)
      vi.spyOn(indexedDB, "open").mockImplementation((...args) => {
        const request = originalOpen(...args)
        request.addEventListener("success", () => connections.push(request.result))
        return request
      })
      const { useWorkspaceStore: store, storage } = await seedOffload(kind)
      const original = connections[0]
      const originalRecord = await readOffloadRecord(original, kind)
      const before = captureRawStorage()
      original.onversionchange!.call(original, new Event("versionchange") as IDBVersionChangeEvent)
      await storage.getItem(WORKSPACE_STORAGE_KEY)
      expect(connections).toHaveLength(2)

      original.onclose?.call(original, new Event("close"))
      original.onversionchange!.call(original, new Event("versionchange") as IDBVersionChangeEvent)
      await storage.getItem(WORKSPACE_STORAGE_KEY)

      expect(connections).toHaveLength(2)
      expect(await readOffloadRecord(connections[1], kind)).toEqual(originalRecord)
      expect(captureRawStorage()).toEqual(before)
      await store.persist.rehydrate()
      expect(store.getState()).toMatchObject({ storeHydrated: true, storeHydrationError: null })
      expect(connections).toHaveLength(2)
    })

    it.each([false, true])("bounds preservation of the existing offload record (already committed: %s)", async (committed) => {
      const { useWorkspaceStore: store, storage } = await seedOffload(kind)
      const storeName = kind === "chat" ? "workspace-chat-sessions" : "workspace-artifact-payloads"
      const recordKey = kind === "chat"
        ? `workspace:${workspaceId}:chat`
        : `workspace:${workspaceId}:artifact:artifact-1`
      const readRecord = async () => {
        const database = await new Promise<IDBDatabase>((resolve, reject) => {
          const request = indexedDB.open("tldw-workspace-storage", 1)
          request.onsuccess = () => resolve(request.result)
          request.onerror = () => reject(request.error)
        })
        try {
          return await new Promise((resolve, reject) => {
            const transaction = database.transaction(storeName)
            const request = transaction.objectStore(storeName).get(recordKey)
            transaction.oncomplete = () => resolve(request.result)
            transaction.onabort = () => reject(transaction.error)
            transaction.onerror = () => reject(transaction.error)
          })
        } finally {
          database.close()
        }
      }
      const originalRecord = await readRecord()
      expect(originalRecord).toBeDefined()
      const replacement = fixtureState(kind)
      if (kind === "chat") {
        replacement.workspaceChatSessions[workspaceId] = {
          ...chatSession,
          messages: [{ ...chatSession.messages[0], message: "Replacement chat ".repeat(1200) }]
        }
      } else {
        replacement.workspaceSnapshots[workspaceId].generatedArtifacts[0].content =
          "Replacement artifact ".repeat(1200)
      }
      const originalPut = IDBObjectStore.prototype.put
      let retry: ReturnType<typeof store.persist.rehydrate> | undefined
      let before: ReturnType<typeof captureRawStorage> | undefined
      const put = vi.spyOn(IDBObjectStore.prototype, "put").mockImplementation(function (...args) {
        expect(args[0].key).toBe(recordKey)
        const request = originalPut.apply(this, args)
        const failHydration = () => {
          localStorage.setItem(WORKSPACE_STORAGE_KEY, malformedRaw)
          before = captureRawStorage()
          retry = store.persist.rehydrate()
        }
        if (committed) {
          this.transaction.addEventListener("complete", failHydration)
        } else {
          request.addEventListener("success", failHydration)
        }
        return request
      })

      await storage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify({ state: replacement, version: 1 }))
      await retry

      expect(put).toHaveBeenCalledTimes(1)
      expect(before).toBeDefined()
      expect(store.getState()).toMatchObject({
        storeHydrated: false, storeHydrationError: expect.any(String)
      })
      expect(captureRawStorage()).toEqual(before)
      const recordAfterRetry = await readRecord()
      if (committed) {
        expect(recordAfterRetry).not.toEqual(originalRecord)
        expect(recordAfterRetry).toMatchObject(kind === "chat" ? {
          session: { messages: [{ message: "Replacement chat ".repeat(1200) }] }
        } : {
          payload: { content: "Replacement artifact ".repeat(1200) }
        })
      } else {
        expect(recordAfterRetry).toEqual(originalRecord)
      }
    })

    it("preserves offloaded data on startup read failure and recovers on retry", async () => {
      await seedOffload(kind)
      const before = captureRawStorage()
      vi.resetModules()
      const get = vi.spyOn(IDBObjectStore.prototype, "get").mockImplementationOnce(() => {
        throw new DOMException("Temporarily unavailable", "UnknownError")
      })

      const { useWorkspaceStore: store } = await import("../workspace")
      await vi.waitFor(() => {
        expect(store.getState().storeHydrationError).toEqual(expect.any(String))
      })
      expect(store.getState().storeHydrated).toBe(false)
      expect(captureRawStorage()).toEqual(before)
      await store.persist.rehydrate()
      await store.setState({})

      expect(get).toHaveBeenCalledTimes(2)
      expect(store.getState()).toMatchObject({ storeHydrated: true, storeHydrationError: null })
      if (kind === "chat") {
        expect(store.getState().workspaceChatSessions[workspaceId].messages[0].message)
          .toBe(chatMessage)
      } else {
        expect(store.getState().generatedArtifacts[0].content).toBe(artifactContent)
      }
    })

    it("fails closed on read errors, preserves pointers, and reads again after repair", async () => {
      const { useWorkspaceStore: store } = await seedOffload(kind)
      const before = captureRawStorage()
      const originalGet = IDBObjectStore.prototype.get
      let readsFail = true
      const get = vi.spyOn(IDBObjectStore.prototype, "get").mockImplementation(function (...args) {
        if (readsFail) throw new DOMException("Temporarily unavailable", "UnknownError")
        return originalGet.apply(this, args)
      })

      for (let attempt = 1; attempt <= 2; attempt += 1) {
        await store.persist.rehydrate()
        expect(store.getState()).toMatchObject({
          storeHydrated: false, storeHydrationError: expect.any(String)
        })
        expect(get).toHaveBeenCalledTimes(attempt)
        expect(captureRawStorage()).toEqual(before)
      }
      readsFail = false
      await store.persist.rehydrate()
      await store.setState({})

      expect(get).toHaveBeenCalledTimes(3)
      expect(store.getState()).toMatchObject({ storeHydrated: true, storeHydrationError: null })
      if (kind === "chat") {
        expect(store.getState().workspaceChatSessions[workspaceId].messages[0].message)
          .toBe(chatMessage)
      } else {
        expect(store.getState().generatedArtifacts[0]).toMatchObject({
          content: artifactContent, data: { retained: true }
        })
      }
    })

    it("rejects missing offloaded records instead of silently reconstructing empty data", async () => {
      const { useWorkspaceStore: store } = await seedOffload(kind)
      const before = captureRawStorage()
      const originalGet = IDBObjectStore.prototype.get
      vi.spyOn(IDBObjectStore.prototype, "get").mockImplementation(function () {
        return originalGet.call(this, "missing-record")
      })

      await store.persist.rehydrate()

      expect(store.getState()).toMatchObject({
        storeHydrated: false, storeHydrationError: expect.any(String)
      })
      expect(captureRawStorage()).toEqual(before)
    })

    it("still reads persisted offload references when new offloading is disabled", async () => {
      await seedOffload(kind)
      localStorage.setItem("tldw:feature-rollout:workspace_indexeddb_offload_v1:enabled", "0")
      const { createWorkspaceStorage } = await import("../workspace")
      const get = vi.spyOn(IDBObjectStore.prototype, "get")

      const raw = await createWorkspaceStorage().getItem(WORKSPACE_STORAGE_KEY)
      const state = JSON.parse(raw!).state

      expect(get).toHaveBeenCalledTimes(1)
      if (kind === "chat") {
        expect(state.workspaceChatSessions[workspaceId].messages[0].message).toBe(chatMessage)
      } else {
        expect(state.workspaceSnapshots[workspaceId].generatedArtifacts[0].content)
          .toBe(artifactContent)
      }
    })
  })

  it("keeps new payloads inline after a write failure and retries offloading after repair", async () => {
    const { createWorkspaceStorage } = await import("../workspace")
    const storage = createWorkspaceStorage()
    const originalPut = IDBObjectStore.prototype.put
    const put = vi.spyOn(IDBObjectStore.prototype, "put").mockImplementationOnce(() => {
      throw new DOMException("Temporarily unavailable", "UnknownError")
    })
    const envelope = JSON.stringify({ state: fixtureState("chat"), version: 1 })

    await storage.setItem(WORKSPACE_STORAGE_KEY, envelope)
    expect(JSON.parse(localStorage.getItem(chatKey)!).messages[0].message).toBe(chatMessage)
    put.mockImplementation(function (...args) { return originalPut.apply(this, args) })
    await storage.setItem(WORKSPACE_STORAGE_KEY, envelope)

    expect(put).toHaveBeenCalledTimes(2)
    expect(JSON.parse(localStorage.getItem(chatKey)!).offloadType)
      .toBe("workspace_chat_session_v1")
  })

  it("does not offload new payloads during monolithic migration when offloading is disabled", async () => {
    const { createWorkspaceStorage } = await import("../workspace")
    localStorage.setItem("tldw:feature-rollout:workspace_indexeddb_offload_v1:enabled", "0")
    localStorage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify({
      state: fixtureState("chat"), version: 1
    }))
    const put = vi.spyOn(IDBObjectStore.prototype, "put")

    await createWorkspaceStorage().getItem(WORKSPACE_STORAGE_KEY)
    await vi.waitFor(() => {
      expect(JSON.parse(localStorage.getItem(WORKSPACE_STORAGE_KEY)!).schema)
        .toBe("workspace_split_v1")
    })

    expect(put).not.toHaveBeenCalled()
    expect(JSON.parse(localStorage.getItem(chatKey)!).messages[0].message).toBe(chatMessage)
  })
})
