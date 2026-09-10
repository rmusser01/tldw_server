import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  abort: vi.fn(),
  deleteHistory: vi.fn(),
  deleteMessages: vi.fn(),
  equals: vi.fn(),
  filterByHistory: vi.fn(),
  transaction: vi.fn()
}))

vi.mock("../schema", () => ({
  db: {
    chatHistories: {
      name: "chatHistories",
      delete: mocks.deleteHistory,
    },
    messages: {
      name: "messages",
      where: vi.fn(() => ({ equals: mocks.equals }))
    },
    modelNickname: { name: "modelNickname" },
    sessionFiles: { name: "sessionFiles" },
    transaction: (...args: unknown[]) => mocks.transaction(...args)
  }
}))

import {
  rollbackScopedComparePersistence,
  runChatPersistenceTransaction
} from "../chat-persistence-transaction"

describe("runChatPersistenceTransaction", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.transaction.mockImplementation(
      async (_mode: unknown, _tables: unknown, operation: (tx: unknown) => unknown) =>
        operation({ abort: mocks.abort })
    )
    mocks.equals.mockReturnValue({
      and: mocks.filterByHistory,
      delete: mocks.deleteMessages
    })
    mocks.filterByHistory.mockReturnValue({ delete: mocks.deleteMessages })
  })

  it("includes the nickname read used by continue saves in the transaction", async () => {
    const operation = vi.fn(async () => "history-1")

    await expect(
      runChatPersistenceTransaction(undefined, operation)
    ).resolves.toBe("history-1")

    expect(mocks.transaction).toHaveBeenCalledWith(
      "rw",
      [
        expect.objectContaining({ name: "chatHistories" }),
        expect.objectContaining({ name: "messages" }),
        expect.objectContaining({ name: "modelNickname" }),
        expect.objectContaining({ name: "sessionFiles" })
      ],
      expect.any(Function)
    )
    expect(operation).toHaveBeenCalledTimes(1)
  })

  it("aborts the transaction and rejects when the scope changes mid-write", async () => {
    let finishWrite!: () => void
    const pendingWrite = new Promise<void>((resolve) => {
      finishWrite = resolve
    })
    const controller = new AbortController()
    const persistence = runChatPersistenceTransaction(
      controller.signal,
      () => pendingWrite
    )

    await vi.waitFor(() => {
      expect(mocks.transaction).toHaveBeenCalledTimes(1)
    })
    controller.abort()
    expect(mocks.abort).toHaveBeenCalledTimes(1)
    finishWrite()

    await expect(persistence).rejects.toMatchObject({ name: "AbortError" })
  })

  it("does not open a transaction for an already-aborted scope", async () => {
    const controller = new AbortController()
    controller.abort()

    await expect(
      runChatPersistenceTransaction(controller.signal, vi.fn())
    ).rejects.toMatchObject({ name: "AbortError" })
    expect(mocks.transaction).not.toHaveBeenCalled()
  })

  it("keeps the abort guard active until the transaction commit resolves", async () => {
    let allowCommit!: () => void
    let operationFinished!: () => void
    const commitGate = new Promise<void>((resolve) => {
      allowCommit = resolve
    })
    const operationGate = new Promise<void>((resolve) => {
      operationFinished = resolve
    })
    mocks.transaction.mockImplementationOnce(
      async (_mode: unknown, _tables: unknown, operation: (tx: unknown) => unknown) => {
        const result = await operation({ abort: mocks.abort })
        operationFinished()
        await commitGate
        return result
      }
    )
    const controller = new AbortController()
    const persistence = runChatPersistenceTransaction(
      controller.signal,
      async () => "history-1"
    )

    await operationGate
    controller.abort()
    expect(mocks.abort).toHaveBeenCalledTimes(1)
    allowCommit()

    await expect(persistence).rejects.toMatchObject({ name: "AbortError" })
  })

  it("ignores a caller abort when the scope-only guard rejects it", async () => {
    let finishWrite!: () => void
    const pendingWrite = new Promise<void>((resolve) => {
      finishWrite = resolve
    })
    const controller = new AbortController()
    const persistence = runChatPersistenceTransaction(
      controller.signal,
      () => pendingWrite,
      () => false
    )

    await vi.waitFor(() => {
      expect(mocks.transaction).toHaveBeenCalledTimes(1)
    })
    controller.abort()
    expect(mocks.abort).not.toHaveBeenCalled()
    finishWrite()

    await expect(persistence).resolves.toBeUndefined()
  })

  it("deletes a rejected Compare cluster without deleting an existing history", async () => {
    await rollbackScopedComparePersistence({
      clusterId: "cluster-1",
      historyId: "history-1",
      removeHistory: false
    })

    expect(mocks.equals).toHaveBeenCalledWith("cluster-1")
    const filter = mocks.filterByHistory.mock.calls[0]?.[0]
    expect(filter({ history_id: "history-1" })).toBe(true)
    expect(filter({ history_id: "history-2" })).toBe(false)
    expect(mocks.deleteMessages).toHaveBeenCalledTimes(1)
    expect(mocks.deleteHistory).not.toHaveBeenCalled()
  })

  it("deletes all messages and the newly-created history for a rejected Compare turn", async () => {
    await rollbackScopedComparePersistence({
      clusterId: "cluster-1",
      historyId: "history-1",
      removeHistory: true
    })

    expect(mocks.equals).toHaveBeenCalledWith("history-1")
    expect(mocks.deleteMessages).toHaveBeenCalledTimes(1)
    expect(mocks.deleteHistory).toHaveBeenCalledWith("history-1")
  })
})
