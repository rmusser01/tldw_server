import { beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/services/folder-api", () => ({
  fetchFolders: vi.fn(), fetchKeywords: vi.fn(), fetchFolderKeywordLinks: vi.fn(),
  fetchConversationKeywordLinks: vi.fn(), createFolder: vi.fn(), updateFolder: vi.fn(),
  deleteFolder: vi.fn(), createKeyword: vi.fn(), deleteKeyword: vi.fn(),
  linkKeywordToFolder: vi.fn(), unlinkKeywordFromFolder: vi.fn(),
  linkKeywordToConversation: vi.fn(), unlinkKeywordFromConversation: vi.fn()
}))
vi.mock("@/db/dexie/schema", () => {
  const table = () => ({ clear: vi.fn(), bulkPut: vi.fn(), toArray: vi.fn(async () => []),
    put: vi.fn(), update: vi.fn(), where: vi.fn(() => ({ equals: () => ({ delete: vi.fn() }) })) })
  return { db: { transaction: vi.fn(async (...args: unknown[]) => (args.at(-1) as () => unknown)()),
    folders: table(), keywords: table(), folderKeywordLinks: table(), conversationKeywordLinks: table() } }
})
import * as api from "@/services/folder-api"
import { db } from "@/db/dexie/schema"
import { useFolderStore } from "../folder"

const ok = <T,>(data: T) => ({ ok: true, status: 200, data })
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(r => { resolve = r }); return { promise, resolve } }
const folder = { id: 1, name: "Adventure", parent_id: null, deleted: false }
const keyword = { id: 9, keyword: "Adventure", deleted: false }
const operation = (user = 7) => ({ requestScope: { config: { serverUrl: "https://folders.test", authMode: "multi-user" as const }, userId: user }, isCurrent: vi.fn(() => true) })
const seedReads = () => {
  vi.mocked(api.fetchFolders).mockResolvedValue(ok([folder]) as never)
  vi.mocked(api.fetchKeywords).mockResolvedValue(ok([keyword]) as never)
  vi.mocked(api.fetchFolderKeywordLinks).mockResolvedValue(ok([{ folder_id: 1, keyword_id: 9 }]) as never)
  vi.mocked(api.fetchConversationKeywordLinks).mockResolvedValue(ok([]) as never)
}
const refreshOwned = (op: ReturnType<typeof operation>) => useFolderStore.getState().refreshFromServer(op)

describe("history folder operations retain captured ownership", () => {
  beforeEach(() => {
    vi.resetAllMocks(); localStorage.clear()
    useFolderStore.setState({ folders: [], keywords: [], folderKeywordLinks: [], conversationKeywordLinks: [],
      isLoading: false, error: null, lastSynced: null, folderApiAvailable: null, serverDataScope: null } as never)
    seedReads()
    vi.mocked(api.linkKeywordToConversation).mockResolvedValue(ok(undefined) as never)
    vi.mocked(api.unlinkKeywordFromFolder).mockResolvedValue(ok(undefined) as never)
    vi.mocked(api.deleteKeyword).mockResolvedValue(ok(undefined) as never)
  })
  it("reads with the captured account, without turning unqualified cache into authority", async () => {
    const op = operation(); await refreshOwned(op)
    expect(api.fetchFolders).toHaveBeenCalledWith(expect.objectContaining({ requestScope: op.requestScope }))
    expect(db.folders.bulkPut).not.toHaveBeenCalled()
    expect(useFolderStore.getState().folders).toEqual([folder])
  })
  it("drops an obsolete created keyword ACK and its cache write", async () => {
    const op = operation(); await refreshOwned(op)
    const held = deferred<ReturnType<typeof ok<typeof keyword>>>()
    vi.mocked(api.createKeyword).mockReturnValue(held.promise as never)
    const pending = useFolderStore.getState().ensureKeyword("New tag", op)
    expect(api.createKeyword).toHaveBeenCalledTimes(1)
    op.isCurrent.mockReturnValue(false); held.resolve(ok({ ...keyword, id: 10, keyword: "New tag" }))
    expect(await pending).toBeNull()
    expect(useFolderStore.getState().keywords).toEqual([keyword])
    expect(db.keywords.put).not.toHaveBeenCalled()
  })
  it("cannot reuse cached IDs without a successful owned snapshot", async () => {
    const op = operation(); useFolderStore.setState({ keywords: [keyword] })
    expect(await useFolderStore.getState().ensureKeyword("Adventure", op)).toBeNull()
    expect(api.createKeyword).not.toHaveBeenCalled()
  })
  it("does not let an older refresh overwrite the replacement account", async () => {
    const old = operation(); const next = operation(8)
    const held = deferred<ReturnType<typeof ok<typeof folder[]>>>()
    vi.mocked(api.fetchFolders).mockReturnValueOnce(held.promise as never)
    const pending = refreshOwned(old); old.isCurrent.mockReturnValue(false)
    await refreshOwned(next); held.resolve(ok([{ ...folder, id: 42, name: "Old private folder" }]))
    await pending
    expect(useFolderStore.getState().folders).toEqual([folder])
    expect(db.folders.bulkPut).not.toHaveBeenCalled()
  })
  it("failed owned refresh does not load unqualified cache or allow an existing cached keyword", async () => {
    const op = operation(); useFolderStore.setState({ keywords: [keyword] })
    vi.mocked(api.fetchFolders).mockResolvedValue({ ok: false, error: "Offline", status: 503 } as never)
    await refreshOwned(op)
    expect(db.keywords.toArray).not.toHaveBeenCalled()
    expect(await useFolderStore.getState().ensureKeyword("Adventure", op)).toBeNull()
  })
  it("stops a compound folder operation before another link after its view retires", async () => {
    const op = operation(); vi.mocked(api.fetchFolderKeywordLinks).mockResolvedValue(ok([]) as never)
    await refreshOwned(op)
    vi.mocked(api.createKeyword).mockResolvedValue(ok(keyword) as never)
    const held = deferred<ReturnType<typeof ok<undefined>>>()
    vi.mocked(api.linkKeywordToFolder).mockReturnValue(held.promise as never)
    const pending = useFolderStore.getState().addConversationToFolder("owned-chat", 1, op)
    await vi.waitFor(() => expect(api.linkKeywordToFolder).toHaveBeenCalledTimes(1))
    op.isCurrent.mockReturnValue(false); held.resolve(ok(undefined)); await pending
    expect(api.linkKeywordToConversation).not.toHaveBeenCalled()
    expect(api.unlinkKeywordFromFolder).not.toHaveBeenCalled()
    expect(api.deleteKeyword).not.toHaveBeenCalled()
  })
  it("unqualified data publication revokes owned mutation authority", async () => {
    const op = operation(); await refreshOwned(op)
    useFolderStore.getState().setKeywords([keyword])
    expect(await useFolderStore.getState().ensureKeyword("Adventure", op)).toBeNull()
  })
  it("a current captured tag apply updates the qualified view and bypasses unqualified cache", async () => {
    const op = operation(); await refreshOwned(op)
    vi.mocked(api.linkKeywordToConversation).mockResolvedValue(ok(undefined) as never)
    expect(await useFolderStore.getState().addKeywordToConversation("owned-chat", 9, op)).toBe(true)
    expect(api.linkKeywordToConversation).toHaveBeenCalledWith("owned-chat", 9, expect.objectContaining({ requestScope: op.requestScope }))
    expect(useFolderStore.getState().conversationKeywordLinks).toEqual([{ conversation_id: "owned-chat", keyword_id: 9 }])
    expect(db.conversationKeywordLinks.put).not.toHaveBeenCalled()
  })
  it("a stale-account response disposes private organization data and stops further writes", async () => {
    const op = operation(); await refreshOwned(op)
    vi.mocked(api.linkKeywordToConversation).mockResolvedValue({ ok: false, status: 412, error: "Account changed" } as never)
    await useFolderStore.getState().addKeywordToConversation("owned-chat", 9, op)
    expect(useFolderStore.getState().folders).toEqual([])
    expect(useFolderStore.getState().keywords).toEqual([])
    expect(await useFolderStore.getState().ensureKeyword("Adventure", op)).toBeNull()
    expect(api.createKeyword).not.toHaveBeenCalled()
  })
  it("a delayed unscoped refresh cannot overwrite a newer verified picker read", async () => {
    const held = deferred<ReturnType<typeof ok<typeof folder[]>>>()
    vi.mocked(api.fetchFolders).mockReturnValueOnce(held.promise as never)
    const pending = useFolderStore.getState().refreshFromServer()
    const op = operation(); await refreshOwned(op)
    held.resolve(ok([{ ...folder, id: 42, name: "Old unqualified cache" }])); await pending
    expect(useFolderStore.getState().folders).toEqual([folder])
    expect(db.folders.bulkPut).not.toHaveBeenCalled()
  })
  it("closing a picker aborts its pending organization read even after unqualified publication", async () => {
    const op = operation()
    const held = deferred<ReturnType<typeof ok<typeof folder[]>>>()
    vi.mocked(api.fetchFolders).mockReturnValueOnce(held.promise as never)
    const pending = refreshOwned(op)
    const signal = vi.mocked(api.fetchFolders).mock.calls[0][0]!.abortSignal!
    useFolderStore.getState().setKeywords([keyword])
    useFolderStore.getState().releaseServerData(op)
    held.resolve(ok([folder])); await pending
    expect(signal.aborted).toBe(true)
    expect(useFolderStore.getState().isLoading).toBe(false)
    expect(useFolderStore.getState().folders).toEqual([])
  })
  it("a read that loses qualification settles loading and leaves a fresh retry usable", async () => {
    const op = operation()
    const held = deferred<ReturnType<typeof ok<typeof folder[]>>>()
    vi.mocked(api.fetchFolders).mockReturnValueOnce(held.promise as never)
    const pending = refreshOwned(op)
    useFolderStore.getState().setKeywords([keyword])
    held.resolve(ok([folder])); await pending
    expect(useFolderStore.getState().isLoading).toBe(false)
    expect(useFolderStore.getState().serverDataScope).toBeNull()
    await refreshOwned(op)
    expect(useFolderStore.getState().folders).toEqual([folder])
    expect(useFolderStore.getState().serverDataScope).toBe(op)
  })
  it("closing an old picker cannot clear a newer pending refresh's busy state", async () => {
    const old = operation(); await refreshOwned(old)
    const held = deferred<ReturnType<typeof ok<typeof folder[]>>>()
    vi.mocked(api.fetchFolders).mockReturnValueOnce(held.promise as never)
    const next = operation(8); const pending = refreshOwned(next)
    const signal = vi.mocked(api.fetchFolders).mock.calls.at(-1)![0]!.abortSignal!
    useFolderStore.getState().releaseServerData(old)
    expect(useFolderStore.getState().isLoading).toBe(true)
    expect(signal.aborted).toBe(false)
    held.resolve(ok([folder])); await pending
    expect(useFolderStore.getState().serverDataScope).toBe(next)
  })
  it("closing an old operation cannot dispose a newer account's qualified view", async () => {
    const old = operation(); await refreshOwned(old)
    const next = operation(8); await refreshOwned(next)
    useFolderStore.getState().releaseServerData(old)
    expect(useFolderStore.getState().serverDataScope).toBe(next)
    expect(useFolderStore.getState().folders).toEqual([folder])
  })

})
