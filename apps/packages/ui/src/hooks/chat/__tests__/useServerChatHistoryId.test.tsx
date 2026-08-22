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

  it("does not publish or cache a new local history after its request scope changes", async () => {
    const staleSave = deferred<{ id: string }>()
    saveHistoryMock
      .mockImplementationOnce(() => staleSave.promise)
      .mockResolvedValueOnce({ id: "history-fresh" })
    const setHistoryId = vi.fn()
    const scopeController = new AbortController()
    const { result } = renderHook(() =>
      useServerChatHistoryId({
        serverChatId: null,
        historyId: null,
        setHistoryId,
        temporaryChat: false,
        t: ((_key: string, options?: { defaultValue?: string }) =>
          options?.defaultValue ?? "Untitled") as any
      })
    )

    const staleAttempt = result.current.ensureServerChatHistoryId(
      "server-chat-1",
      "Scoped title",
      scopeController.signal
    )
    await vi.waitFor(() => expect(saveHistoryMock).toHaveBeenCalledTimes(1))
    scopeController.abort()
    staleSave.resolve({ id: "history-stale" })

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
          "Fresh title"
        )
      ).resolves.toBe("history-fresh")
    })
    expect(saveHistoryMock).toHaveBeenCalledTimes(2)
    expect(setHistoryId).toHaveBeenCalledWith("history-fresh", {
      preserveServerChatId: true
    })
  })

  it("does not cache an existing local-history mapping when its scoped transaction aborts", async () => {
    const staleMapping = deferred<void>()
    setHistoryServerChatIdMock
      .mockImplementationOnce(() => staleMapping.promise)
      .mockResolvedValueOnce(undefined)
    const scopeController = new AbortController()
    const { result } = renderHook(() =>
      useServerChatHistoryId({
        serverChatId: null,
        historyId: "history-local",
        setHistoryId: vi.fn(),
        temporaryChat: false,
        t: ((_key: string, options?: { defaultValue?: string }) =>
          options?.defaultValue ?? "Untitled") as any
      })
    )

    const staleAttempt = result.current.ensureServerChatHistoryId(
      "server-chat-2",
      undefined,
      scopeController.signal
    )
    await vi.waitFor(() =>
      expect(setHistoryServerChatIdMock).toHaveBeenCalledTimes(1)
    )
    scopeController.abort()
    staleMapping.resolve()

    await expect(staleAttempt).rejects.toMatchObject({ status: 412 })

    await act(async () => {
      await expect(
        result.current.ensureServerChatHistoryId("server-chat-2")
      ).resolves.toBe("history-local")
    })
    expect(setHistoryServerChatIdMock).toHaveBeenCalledTimes(2)
  })
})
