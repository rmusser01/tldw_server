import type { TFunction } from "i18next"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const {
  getHistoryByServerChatIdMock,
  runChatPersistenceTransactionMock,
  saveHistoryMock,
  setHistoryServerChatIdMock,
  updateHistoryMock
} = vi.hoisted(() => ({
  getHistoryByServerChatIdMock: vi.fn(),
  runChatPersistenceTransactionMock: vi.fn(),
  saveHistoryMock: vi.fn(),
  setHistoryServerChatIdMock: vi.fn(),
  updateHistoryMock: vi.fn()
}))

vi.mock("@/db/dexie/helpers", () => ({
  getHistoryByServerChatId: getHistoryByServerChatIdMock,
  saveHistory: saveHistoryMock,
  setHistoryServerChatId: setHistoryServerChatIdMock,
  updateHistory: updateHistoryMock
}))

vi.mock("@/db/dexie/chat-persistence-transaction", () => ({
  runChatPersistenceTransaction: runChatPersistenceTransactionMock
}))

import { useServerChatHistoryId } from "../useServerChatHistoryId"
import { usePlaygroundSessionStore } from "@/store/playground-session"
const ownedLink = vi.hoisted(() => vi.fn())
vi.mock("@/db/dexie/server-chat-mirror", () => ({
  serverChatMirrorOwnerKey: (snapshot: ServicePromptSnapshot) => `owner:${snapshot.requestScope.userId}`,
  linkServerChatMirror: ownedLink
}))

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

const abortError = () => {
  const error = new Error("Request scope changed")
  error.name = "AbortError"
  return error
}

describe("useServerChatHistoryId", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    usePlaygroundSessionStore.getState().clearSession()
    ownedLink.mockImplementation(async ({ ownerKey }) => `history-${ownerKey}`)
    getHistoryByServerChatIdMock.mockResolvedValue(null)
    saveHistoryMock.mockResolvedValue({ id: "history-default" })
    setHistoryServerChatIdMock.mockResolvedValue(undefined)
    updateHistoryMock.mockResolvedValue(undefined)
    runChatPersistenceTransactionMock.mockImplementation(
      async (signal: AbortSignal | undefined, operation: () => Promise<unknown>) => {
        if (signal?.aborted) throw abortError()
        const result = await operation()
        if (signal?.aborted) throw abortError()
        return result
      }
    )
  })

  it("rejects an unowned eager link without reading or mutating local history", async () => {
    const { result } = renderHook(() => useServerChatHistoryId({ serverChatId: "chat", historyId: "legacy", setHistoryId: vi.fn(), temporaryChat: false, t: vi.fn() as unknown as TFunction }))
    await expect(result.current.ensureServerChatHistoryId("chat", "Cedar")).rejects.toMatchObject({ status: 412 })
    expect(getHistoryByServerChatIdMock).not.toHaveBeenCalled()
    expect(ownedLink).not.toHaveBeenCalled()
    expect(setHistoryServerChatIdMock).not.toHaveBeenCalled()
  })

  it("does not publish a delayed mirror over a newly selected history", async () => {
    const pending = deferred<string>()
    ownedLink.mockReturnValueOnce(pending.promise)
    const setHistoryId = vi.fn()
    const { result, rerender } = renderHook(({ historyId }) => useServerChatHistoryId({ serverChatId: "chat", historyId, setHistoryId, temporaryChat: false, t: vi.fn() as unknown as TFunction }), { initialProps: { historyId: "original" } })
    const request = result.current.ensureServerChatHistoryId("chat", "Cedar", undefined, { scopeKey: "A", requestScope: { userId: "A" } } as unknown as ServicePromptSnapshot)
    rerender({ historyId: "replacement" })
    pending.resolve("old-mirror")
    await expect(request).rejects.toMatchObject({ status: 412 })
    expect(setHistoryId).not.toHaveBeenCalled()
  })

  it.each([true, false])("requires a validated current persisted session for legacy adoption (%s)", async valid => {
    usePlaygroundSessionStore.getState().saveSession({ historyId: "legacy", serverChatId: "chat", scopeKey: valid ? "scope-A" : "scope-B" })
    const { result } = renderHook(() => useServerChatHistoryId({ serverChatId: "chat", historyId: "legacy", setHistoryId: vi.fn(), temporaryChat: false, t: ((_key: string) => "Cedar") as TFunction }))
    const snapshot = { scopeKey: "scope-A", requestScope: { userId: "A" } } as unknown as ServicePromptSnapshot
    await act(async () => { await result.current.ensureServerChatHistoryId("chat", "Cedar", undefined, snapshot) })
    expect(ownedLink).toHaveBeenCalledWith(expect.objectContaining({ ownerKey: "owner:A", legacyHistoryId: valid ? "legacy" : null }))
  })

  it("does not reuse a cached mapping for another verified owner of the same chat ID", async () => {
    const { result } = renderHook(() => useServerChatHistoryId({ serverChatId: "chat", historyId: null, setHistoryId: vi.fn(), temporaryChat: false, t: ((_key: string) => "Cedar") as TFunction }))
    const snapshot = (userId: string) => ({ scopeKey: `scope-${userId}`, requestScope: { userId } }) as unknown as ServicePromptSnapshot
    await act(async () => {
      expect(await result.current.ensureServerChatHistoryId("chat", "Cedar", undefined, snapshot("A"))).toBe("history-owner:A")
      expect(await result.current.ensureServerChatHistoryId("chat", "Cedar", undefined, snapshot("B"))).toBe("history-owner:B")
    })
  })

  it("does not publish or cache a new local history after its request scope changes", async () => {
    const staleSave = deferred<string>()
    ownedLink
      .mockImplementationOnce(() => staleSave.promise)
      .mockResolvedValueOnce("history-fresh")
    const setHistoryId = vi.fn()
    const scopeController = new AbortController()
    const { result } = renderHook(() =>
      useServerChatHistoryId({
        serverChatId: null,
        historyId: null,
        setHistoryId,
        temporaryChat: false,
        t: ((_key: string, options?: { defaultValue?: string }) =>
          options?.defaultValue ?? "Untitled") as TFunction
      })
    )

    const staleAttempt = result.current.ensureServerChatHistoryId(
      "server-chat-1",
      "Scoped title",
      scopeController.signal,
      { scopeKey: "scope-A", requestScope: { userId: "A" } } as unknown as ServicePromptSnapshot
    )
    await vi.waitFor(() => expect(ownedLink).toHaveBeenCalledTimes(1))
    scopeController.abort()
    staleSave.resolve("history-stale")

    await expect(staleAttempt).rejects.toMatchObject({
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })
    expect(setHistoryId).not.toHaveBeenCalled()

    await act(async () => {
      await expect(
        result.current.ensureServerChatHistoryId(
          "server-chat-1",
          "Fresh title", undefined, { scopeKey: "scope-A", requestScope: { userId: "A" } } as unknown as ServicePromptSnapshot
        )
      ).resolves.toBe("history-fresh")
    })
    expect(ownedLink).toHaveBeenCalledTimes(2)
    expect(setHistoryId).toHaveBeenCalledWith("history-fresh", {
      preserveServerChatId: true
    })
  })

  it("does not cache an existing local-history mapping when its scoped transaction aborts", async () => {
    const staleMapping = deferred<string>()
    ownedLink
      .mockImplementationOnce(() => staleMapping.promise)
      .mockResolvedValueOnce("history-local")
    const scopeController = new AbortController()
    const { result } = renderHook(() =>
      useServerChatHistoryId({
        serverChatId: null,
        historyId: "history-local",
        setHistoryId: vi.fn(),
        temporaryChat: false,
        t: ((_key: string, options?: { defaultValue?: string }) =>
          options?.defaultValue ?? "Untitled") as TFunction
      })
    )

    const staleAttempt = result.current.ensureServerChatHistoryId(
      "server-chat-2",
      undefined,
      scopeController.signal,
      { scopeKey: "scope-A", requestScope: { userId: "A" } } as unknown as ServicePromptSnapshot
    )
    await vi.waitFor(() =>
      expect(ownedLink).toHaveBeenCalledTimes(1)
    )
    scopeController.abort()
    staleMapping.resolve("history-local")

    await expect(staleAttempt).rejects.toMatchObject({ status: 412 })

    await act(async () => {
      await expect(
        result.current.ensureServerChatHistoryId("server-chat-2", undefined, undefined, { scopeKey: "scope-A", requestScope: { userId: "A" } } as unknown as ServicePromptSnapshot)
      ).resolves.toBe("history-local")
    })
    expect(ownedLink).toHaveBeenCalledTimes(2)
  })
})
