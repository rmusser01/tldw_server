import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { WORKSPACE_STORAGE_KEY } from "@/store/workspace-events"

const malformedRaw = '{\n  "state": {"notes":"private-draft"},\n  "version": 1,\n'
const retainedSnapshotKey = `${WORKSPACE_STORAGE_KEY}:workspace:retained:snapshot`
const retainedChatKey = `${WORKSPACE_STORAGE_KEY}:workspace:retained:chat`

const captureRawStorage = () =>
  Object.fromEntries(
    Array.from({ length: localStorage.length }, (_, index) => {
      const key = localStorage.key(index)!
      return [key, localStorage.getItem(key)]
    })
  )

const seedFailure = () => {
  localStorage.setItem(WORKSPACE_STORAGE_KEY, malformedRaw)
  localStorage.setItem(retainedSnapshotKey, '{ "notes" : "retained snapshot" }\n')
  localStorage.setItem(retainedChatKey, '{ "messages" : [] }\n')
}

describe.each([
  ["split", "1"],
  ["monolithic", "0"]
])("workspace hydration recovery (%s storage)", (_mode, splitFlag) => {
  beforeEach(() => {
    vi.resetModules()
    localStorage.clear()
    localStorage.setItem(
      "tldw:feature-rollout:workspace_split_storage_v1:enabled",
      splitFlag
    )
    localStorage.setItem(
      "tldw:feature-rollout:workspace_indexeddb_offload_v1:enabled",
      "0"
    )
  })

  afterEach(() => {
    vi.restoreAllMocks()
    localStorage.clear()
  })

  it("reports synchronous parser failure without treating corruption as empty", async () => {
    seedFailure()
    const before = captureRawStorage()

    const { useWorkspaceStore: store } = await import("../workspace")

    expect(store.getState()).toMatchObject({
      storeHydrated: false,
      storeHydrationError: expect.any(String),
      workspaceId: ""
    })
    expect(store.getState().storeHydrationError).not.toContain("private-draft")
    expect(store.getState().initializeWorkspace).toBeTypeOf("function")
    expect(store.persist.hasHydrated()).toBe(false)
    expect(captureRawStorage()).toEqual(before)
  })

  it("fences persist writes and removal while failed, including wrapped setters", async () => {
    seedFailure()
    const before = captureRawStorage()
    const { useWorkspaceStore: store } = await import("../workspace")

    await store.setState({ sourcesLoading: true })
    await store.getState().toggleLeftPane()
    expect(captureRawStorage()).toEqual(before)
    await store.persist.clearStorage()

    expect(captureRawStorage()).toEqual(before)
    expect(store.getState().storeHydrated).toBe(false)
  })

  it("keeps repeated retries fail-closed and preserves all original bytes", async () => {
    seedFailure()
    const before = captureRawStorage()
    const { useWorkspaceStore: store } = await import("../workspace")

    for (let attempt = 0; attempt < 2; attempt += 1) {
      await store.persist.rehydrate()
      expect(store.getState()).toMatchObject({
        storeHydrated: false,
        storeHydrationError: expect.any(String),
        workspaceId: ""
      })
      expect(captureRawStorage()).toEqual(before)
    }
  })

  it("notifies subscribers and revokes readiness when a later hydration fails", async () => {
    const { useWorkspaceStore: store } = await import("../workspace")
    expect(store.getState().storeHydrated).toBe(true)
    expect(store.getState().initializeWorkspace).toBeTypeOf("function")
    seedFailure()
    const before = captureRawStorage()
    const notifications: unknown[] = []
    const unsubscribe = store.subscribe((state) => {
      notifications.push([state.storeHydrated, state.storeHydrationError])
    })

    await store.persist.rehydrate()
    unsubscribe()

    expect(notifications).toContainEqual([false, expect.any(String)])
    expect(store.persist.hasHydrated()).toBe(false)
    expect(captureRawStorage()).toEqual(before)
  })

  it("recovers repaired data on retry and resumes normal persistence", async () => {
    seedFailure()
    const { useWorkspaceStore: store } = await import("../workspace")
    expect(store.getState().storeHydrationError).toEqual(expect.any(String))
    localStorage.setItem(
      WORKSPACE_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        state: {
          workspaceId: "repaired",
          savedWorkspaces: [],
          archivedWorkspaces: [],
          workspaceCollections: [],
          workspaceSnapshots: {
            repaired: {
              workspaceId: "repaired",
              workspaceName: "Repaired research",
              workspaceTag: "workspace:repaired",
              workspaceCreatedAt: "2026-09-29T00:00:00.000Z",
              sources: [],
              selectedSourceIds: [],
              generatedArtifacts: [],
              notes: "Recovered notes"
            }
          },
          workspaceChatSessions: {}
        }
      })
    )

    await store.persist.rehydrate()

    expect(store.getState()).toMatchObject({
      storeHydrated: true,
      storeHydrationError: null,
      workspaceId: "repaired",
      workspaceName: "Repaired research",
      notes: "Recovered notes"
    })
    expect(store.getState().workspaceCreatedAt).toBeInstanceOf(Date)
    expect(store.persist.hasHydrated()).toBe(true)
    await store.getState().setWorkspaceName("Writable after repair")
    const persisted = await store.persist
      .getOptions().storage!.getItem(WORKSPACE_STORAGE_KEY)
    expect(persisted?.state.workspaceSnapshots.repaired.workspaceName).toBe(
      "Writable after repair"
    )
    expect(persisted?.state).not.toHaveProperty("storeHydrationError")
  })

  it("reports storage access errors and retries without deleting data", async () => {
    seedFailure()
    const before = captureRawStorage()
    const getItem = vi.spyOn(Object.getPrototypeOf(localStorage), "getItem")
    getItem.mockImplementation(() => {
      throw new DOMException("Storage access denied", "SecurityError")
    })

    const { useWorkspaceStore: store } = await import("../workspace")

    expect(store.getState()).toMatchObject({
      storeHydrated: false,
      storeHydrationError: expect.any(String)
    })
    getItem.mockRestore()
    expect(captureRawStorage()).toEqual(before)
    await store.persist.rehydrate()
    expect(store.getState().storeHydrated).toBe(false)
    expect(captureRawStorage()).toEqual(before)
  })

  it("publishes retry-in-progress without writing over the failed payload", async () => {
    seedFailure()
    const before = captureRawStorage()
    const { useWorkspaceStore: store } = await import("../workspace")
    const storage = store.persist.getOptions().storage!
    let rejectRead!: (error: Error) => void
    store.persist.setOptions({
      storage: {
        ...storage,
        getItem: () => new Promise((_resolve, reject) => {
          rejectRead = reject
        })
      }
    })

    const retry = store.persist.rehydrate()

    expect(store.getState()).toMatchObject({
      storeHydrated: false,
      storeHydrationError: null
    })
    await store.getState().toggleRightPane()
    expect(captureRawStorage()).toEqual(before)
    rejectRead(new Error("Storage still unavailable"))
    await retry
    expect(store.getState().storeHydrationError).toEqual(expect.any(String))
    expect(captureRawStorage()).toEqual(before)
  })
})
