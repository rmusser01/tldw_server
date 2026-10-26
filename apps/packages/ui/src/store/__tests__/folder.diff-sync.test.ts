import { beforeEach, describe, expect, it, vi } from "vitest"

// WP-30 / TASK-13523 Stage 4: refreshFromServer must diff by last_modified so
// an unchanged refresh issues zero bulk writes, upsert only changed rows, and
// keep the legacy clear-and-rewrite for tables with vanished rows (the store
// cannot distinguish locally-created rows from server-deleted ones).

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

const dbMock = vi.hoisted(() => {
  const makeTable = (keyOf: (row: any) => string) => {
    const rows = new Map<string, any>()
    const table = {
      rows,
      clear: vi.fn(async () => {
        rows.clear()
      }),
      bulkPut: vi.fn(async (items: any[]) => {
        for (const item of items) rows.set(keyOf(item), item)
      }),
      toArray: vi.fn(async () => [...rows.values()]),
      put: vi.fn(async (item: any) => {
        rows.set(keyOf(item), item)
      }),
      update: vi.fn(async () => undefined),
      where: vi.fn(() => ({ equals: vi.fn(() => ({ delete: vi.fn(async () => undefined) })) }))
    }
    return table
  }
  const db = {
    transaction: vi.fn(async (...args: unknown[]) => {
      const cb = args[args.length - 1]
      return typeof cb === "function" ? await (cb as () => unknown)() : undefined
    }),
    folders: makeTable((row) => `id:${row.id}`),
    keywords: makeTable((row) => `id:${row.id}`),
    folderKeywordLinks: makeTable((row) => `${row.folder_id}::${row.keyword_id}`),
    conversationKeywordLinks: makeTable((row) => `${row.conversation_id}::${row.keyword_id}`)
  }
  return { db }
})

vi.mock("@/db/dexie/schema", () => dbMock)

import * as folderApi from "@/services/folder-api"
import { useFolderStore } from "../folder"

const ok = <T>(data: T) => ({ ok: true, status: 200, data })

const folder = (id: number, lastModified: string, name = `folder-${id}`) => ({
  id,
  name,
  parent_id: null,
  created_at: "2026-01-01T00:00:00Z",
  last_modified: lastModified,
  version: 1,
  deleted: false
})

const keyword = (id: number, lastModified: string) => ({
  id,
  keyword: `keyword-${id}`,
  created_at: "2026-01-01T00:00:00Z",
  last_modified: lastModified,
  version: 1,
  deleted: false
})

const folderLink = (folderId: number, keywordId: number) => ({
  folder_id: folderId,
  keyword_id: keywordId
})

const conversationLink = (conversationId: string, keywordId: number) => ({
  conversation_id: conversationId,
  keyword_id: keywordId
})

const seedLocal = (data: {
  folders?: ReturnType<typeof folder>[]
  keywords?: ReturnType<typeof keyword>[]
  folderKeywordLinks?: ReturnType<typeof folderLink>[]
  conversationKeywordLinks?: ReturnType<typeof conversationLink>[]
}) => {
  for (const row of data.folders ?? []) dbMock.db.folders.rows.set(`id:${row.id}`, row)
  for (const row of data.keywords ?? []) dbMock.db.keywords.rows.set(`id:${row.id}`, row)
  for (const row of data.folderKeywordLinks ?? []) {
    dbMock.db.folderKeywordLinks.rows.set(`${row.folder_id}::${row.keyword_id}`, row)
  }
  for (const row of data.conversationKeywordLinks ?? []) {
    dbMock.db.conversationKeywordLinks.rows.set(`${row.conversation_id}::${row.keyword_id}`, row)
  }
}

const mockFetches = (data: {
  folders: unknown[]
  keywords: unknown[]
  folderKeywordLinks: unknown[]
  conversationKeywordLinks: unknown[]
}) => {
  vi.mocked(folderApi.fetchFolders).mockResolvedValue(ok(data.folders) as never)
  vi.mocked(folderApi.fetchKeywords).mockResolvedValue(ok(data.keywords) as never)
  vi.mocked(folderApi.fetchFolderKeywordLinks).mockResolvedValue(ok(data.folderKeywordLinks) as never)
  vi.mocked(folderApi.fetchConversationKeywordLinks).mockResolvedValue(
    ok(data.conversationKeywordLinks) as never
  )
}

const tables = [
  dbMock.db.folders,
  dbMock.db.keywords,
  dbMock.db.folderKeywordLinks,
  dbMock.db.conversationKeywordLinks
]

describe("folder store diff-sync (WP-30)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    for (const table of tables) table.rows.clear()
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

  it("issues zero bulk writes when nothing changed since the last sync", async () => {
    const data = {
      folders: [folder(1, "2026-01-02T00:00:00Z")],
      keywords: [keyword(2, "2026-01-03T00:00:00Z")],
      folderKeywordLinks: [folderLink(1, 2)],
      conversationKeywordLinks: [conversationLink("chat-a", 2)]
    }
    seedLocal(data)
    mockFetches(data)

    await useFolderStore.getState().refreshFromServer()

    for (const table of tables) {
      expect(table.bulkPut).not.toHaveBeenCalled()
      expect(table.clear).not.toHaveBeenCalled()
    }
    expect(dbMock.db.transaction).not.toHaveBeenCalled()
  })

  it("upserts only changed and new rows, skipping unchanged ones", async () => {
    seedLocal({
      folders: [folder(1, "2026-01-02T00:00:00Z"), folder(2, "2026-01-02T00:00:00Z")],
      keywords: [keyword(10, "2026-01-03T00:00:00Z")],
      folderKeywordLinks: [folderLink(1, 10)],
      conversationKeywordLinks: [conversationLink("chat-a", 10)]
    })
    const changedFolder = folder(2, "2026-01-05T00:00:00Z", "renamed")
    const newFolder = folder(3, "2026-01-06T00:00:00Z")
    mockFetches({
      folders: [folder(1, "2026-01-02T00:00:00Z"), changedFolder, newFolder],
      keywords: [keyword(10, "2026-01-03T00:00:00Z")],
      folderKeywordLinks: [folderLink(1, 10)],
      conversationKeywordLinks: [conversationLink("chat-a", 10)]
    })

    await useFolderStore.getState().refreshFromServer()

    expect(dbMock.db.folders.clear).not.toHaveBeenCalled()
    expect(dbMock.db.folders.bulkPut).toHaveBeenCalledTimes(1)
    expect(dbMock.db.folders.bulkPut).toHaveBeenCalledWith([changedFolder, newFolder])
    // Untouched families must not be rewritten.
    expect(dbMock.db.keywords.bulkPut).not.toHaveBeenCalled()
    expect(dbMock.db.folderKeywordLinks.bulkPut).not.toHaveBeenCalled()
    expect(dbMock.db.conversationKeywordLinks.bulkPut).not.toHaveBeenCalled()

    const state = useFolderStore.getState()
    expect(state.folders).toHaveLength(3)
    expect(dbMock.db.folders.rows.size).toBe(3)
  })

  it("falls back to clear-and-rewrite only for tables with vanished rows", async () => {
    seedLocal({
      folders: [folder(1, "2026-01-02T00:00:00Z"), folder(2, "2026-01-02T00:00:00Z")],
      keywords: [keyword(10, "2026-01-03T00:00:00Z")],
      folderKeywordLinks: [folderLink(1, 10), folderLink(2, 10)],
      conversationKeywordLinks: []
    })
    const survivor = folder(1, "2026-01-02T00:00:00Z")
    mockFetches({
      folders: [survivor],
      keywords: [keyword(10, "2026-01-03T00:00:00Z")],
      folderKeywordLinks: [folderLink(1, 10)],
      conversationKeywordLinks: []
    })

    await useFolderStore.getState().refreshFromServer()

    // Folder 2 vanished remotely: the legacy full rewrite must run for that
    // family so deletion semantics stay identical to clear-and-rewrite.
    expect(dbMock.db.folders.clear).toHaveBeenCalledTimes(1)
    expect(dbMock.db.folders.bulkPut).toHaveBeenCalledTimes(1)
    expect(dbMock.db.folders.bulkPut).toHaveBeenCalledWith([survivor])
    expect([...dbMock.db.folders.rows.values()]).toEqual([survivor])

    // Links also lost a row: same per-table fallback.
    expect(dbMock.db.folderKeywordLinks.clear).toHaveBeenCalledTimes(1)
    expect(dbMock.db.folderKeywordLinks.bulkPut).toHaveBeenCalledWith([folderLink(1, 10)])

    // Keywords had no deletions: unchanged family must not be rewritten.
    expect(dbMock.db.keywords.clear).not.toHaveBeenCalled()
    expect(dbMock.db.keywords.bulkPut).not.toHaveBeenCalled()
  })

  it("treats rows missing last_modified as always-dirty", async () => {
    seedLocal({ folders: [folder(1, "2026-01-02T00:00:00Z")] })
    const unversioned = { ...folder(1, "2026-01-02T00:00:00Z"), last_modified: undefined }
    mockFetches({
      folders: [unversioned],
      keywords: [],
      folderKeywordLinks: [],
      conversationKeywordLinks: []
    })

    await useFolderStore.getState().refreshFromServer()

    expect(dbMock.db.folders.bulkPut).toHaveBeenCalledWith([unversioned])
  })

  it("diffs link tables by composite key and upserts only new links", async () => {
    seedLocal({ folderKeywordLinks: [folderLink(1, 10)] })
    mockFetches({
      folders: [],
      keywords: [],
      folderKeywordLinks: [folderLink(1, 10), folderLink(1, 20)],
      conversationKeywordLinks: [conversationLink("chat-a", 10)]
    })

    await useFolderStore.getState().refreshFromServer()

    expect(dbMock.db.folderKeywordLinks.clear).not.toHaveBeenCalled()
    expect(dbMock.db.folderKeywordLinks.bulkPut).toHaveBeenCalledWith([folderLink(1, 20)])
    expect(dbMock.db.conversationKeywordLinks.bulkPut).toHaveBeenCalledWith([
      conversationLink("chat-a", 10)
    ])
  })

  it("rewrites everything when the diff cannot be computed", async () => {
    seedLocal({ folders: [folder(1, "2026-01-02T00:00:00Z")] })
    dbMock.db.folders.toArray.mockRejectedValueOnce(new Error("dexie read failed") as never)
    const remoteFolder = folder(1, "2026-01-09T00:00:00Z")
    mockFetches({
      folders: [remoteFolder],
      keywords: [keyword(10, "2026-01-03T00:00:00Z")],
      folderKeywordLinks: [],
      conversationKeywordLinks: []
    })

    await useFolderStore.getState().refreshFromServer()

    // Unresolvable diff: full clear-and-rewrite fallback, exactly the legacy path.
    expect(dbMock.db.folders.clear).toHaveBeenCalledTimes(1)
    expect(dbMock.db.folders.bulkPut).toHaveBeenCalledWith([remoteFolder])
    expect(dbMock.db.keywords.clear).toHaveBeenCalledTimes(1)
    expect(useFolderStore.getState().folders).toEqual([remoteFolder])
  })
})
