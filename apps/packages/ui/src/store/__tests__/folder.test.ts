import { beforeEach, describe, expect, it, vi } from "vitest"

// The folder store persists a small slice of state to localStorage. The tests
// below exercise the H11 fix: a single transient 404 must not permanently
// disable folder sync across sessions.

vi.mock("@/services/folder-api", () => ({
  fetchFolders: vi.fn(),
  fetchKeywords: vi.fn(),
  fetchFolderKeywordLinks: vi.fn(),
  fetchConversationKeywordLinks: vi.fn(),
  createFolder: vi.fn(),
  updateFolder: vi.fn(),
  deleteFolder: vi.fn(),
  createKeyword: vi.fn(),
  deleteKeyword: vi.fn(),
  linkKeywordToFolder: vi.fn(),
  unlinkKeywordFromFolder: vi.fn(),
  linkKeywordToConversation: vi.fn(),
  unlinkKeywordFromConversation: vi.fn()
}))

vi.mock("@/db/dexie/schema", () => {
  const table = () => ({
    clear: vi.fn(async () => undefined),
    bulkPut: vi.fn(async () => undefined),
    toArray: vi.fn(async () => []),
    put: vi.fn(async () => undefined),
    update: vi.fn(async () => undefined),
    where: vi.fn(() => ({ equals: vi.fn(() => ({ delete: vi.fn(async () => undefined) })) }))
  })
  return {
    db: {
      transaction: vi.fn(async (..._args: unknown[]) => {
        const cb = _args[_args.length - 1]
        return typeof cb === "function" ? await (cb as () => unknown)() : undefined
      }),
      folders: table(),
      keywords: table(),
      folderKeywordLinks: table(),
      conversationKeywordLinks: table()
    }
  }
})

import * as folderApi from "@/services/folder-api"
import { useFolderStore } from "../folder"

const FOLDER_STORAGE_KEY = "tldw-folder-store"

const notFound = () => ({ ok: false, status: 404, error: "Not Found" })
const ok = <T>(data: T) => ({ ok: true, status: 200, data })

const mockAllFetches = (result: unknown) => {
  vi.mocked(folderApi.fetchFolders).mockResolvedValue(result as never)
  vi.mocked(folderApi.fetchKeywords).mockResolvedValue(result as never)
  vi.mocked(folderApi.fetchFolderKeywordLinks).mockResolvedValue(result as never)
  vi.mocked(folderApi.fetchConversationKeywordLinks).mockResolvedValue(
    result as never
  )
}

describe("folder store sticky-failure recovery (H11)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    useFolderStore.setState({
      folders: [],
      keywords: [],
      folderKeywordLinks: [],
      conversationKeywordLinks: [],
      isLoading: false,
      lastSynced: null,
      error: null,
      folderApiAvailable: null
    })
  })

  it("disables sync only for the current session after a 404", async () => {
    mockAllFetches(notFound())

    await useFolderStore.getState().refreshFromServer()
    expect(useFolderStore.getState().folderApiAvailable).toBe(false)

    // Within the same session a subsequent refresh is skipped (avoids hammering
    // a server that lacks the folder API) — no additional fetch is issued.
    const callsAfterFirst = vi.mocked(folderApi.fetchFolders).mock.calls.length
    await useFolderStore.getState().refreshFromServer()
    expect(vi.mocked(folderApi.fetchFolders).mock.calls.length).toBe(
      callsAfterFirst
    )
  })

  it("does not rehydrate a persisted folderApiAvailable:false flag", async () => {
    // Simulate a user whose localStorage still carries a stale `false` written
    // by an earlier build (before the flag was removed from partialize).
    localStorage.setItem(
      FOLDER_STORAGE_KEY,
      JSON.stringify({
        state: {
          uiPrefs: {},
          viewMode: "folders",
          lastSynced: 123,
          folderApiAvailable: false
        },
        version: 0
      })
    )

    await useFolderStore.persist.rehydrate()

    // The unavailable flag resets to the retryable `null` default, while the
    // other persisted values still rehydrate normally.
    expect(useFolderStore.getState().folderApiAvailable).toBeNull()
    expect(useFolderStore.getState().viewMode).toBe("folders")
    expect(useFolderStore.getState().lastSynced).toBe(123)
  })

  it("re-probes and recovers after the flag resets for a new session", async () => {
    // A 404 disables sync this session.
    mockAllFetches(notFound())
    await useFolderStore.getState().refreshFromServer()
    expect(useFolderStore.getState().folderApiAvailable).toBe(false)

    // A new session starts with the flag reset (guaranteed by not persisting it
    // + the merge stripper). Folder sync must now succeed again.
    useFolderStore.setState({ folderApiAvailable: null })
    vi.mocked(folderApi.fetchFolders).mockResolvedValue(
      ok([{ id: 1, name: "Recovered", parent_id: null, deleted: false }]) as never
    )
    vi.mocked(folderApi.fetchKeywords).mockResolvedValue(ok([]) as never)
    vi.mocked(folderApi.fetchFolderKeywordLinks).mockResolvedValue(ok([]) as never)
    vi.mocked(folderApi.fetchConversationKeywordLinks).mockResolvedValue(
      ok([]) as never
    )

    await useFolderStore.getState().refreshFromServer()

    expect(useFolderStore.getState().folderApiAvailable).toBe(true)
    expect(useFolderStore.getState().folders).toHaveLength(1)
  })
})
