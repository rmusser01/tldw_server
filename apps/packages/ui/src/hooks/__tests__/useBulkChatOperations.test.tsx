import { beforeEach, describe, expect, it, vi } from "vitest"
import type { TFunction } from "i18next"

import { runBulkDelete } from "../useBulkChatOperations"

const mocks = vi.hoisted(() => ({
  messageError: vi.fn(),
  messageSuccess: vi.fn(),
  messageWarning: vi.fn()
}))

vi.mock("antd", () => ({
  message: {
    error: mocks.messageError,
    success: mocks.messageSuccess,
    warning: mocks.messageWarning
  }
}))

const t = ((_key: string, fallback?: string) =>
  fallback || _key) as unknown as TFunction<"translation", undefined>

describe("runBulkDelete", () => {
  beforeEach(() => {
    mocks.messageError.mockClear()
    mocks.messageSuccess.mockClear()
    mocks.messageWarning.mockClear()
  })

  it("deletes all selected conversations and reports success", async () => {
    const deleteConversation = vi.fn().mockResolvedValue(undefined)
    const deleteResult = await runBulkDelete({
      selectedConversationIds: ["chat-1", "chat-2"],
      deleteConversation,
      t
    })

    expect(deleteConversation).toHaveBeenCalledTimes(2)
    expect(deleteConversation).toHaveBeenNthCalledWith(1, "chat-1")
    expect(deleteConversation).toHaveBeenNthCalledWith(2, "chat-2")
    expect(deleteResult).not.toBeNull()
    expect(Array.from(deleteResult?.deletedConversationIds || [])).toEqual([
      "chat-1",
      "chat-2"
    ])
    expect(Array.from(deleteResult?.failedConversationIds || [])).toEqual([])
    expect(mocks.messageSuccess).toHaveBeenCalledWith("Chats moved to trash.")
    expect(mocks.messageError).not.toHaveBeenCalled()
  })

  it("returns partial failures when some deletes fail", async () => {
    const deleteConversation = vi
      .fn()
      .mockResolvedValueOnce(undefined)
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValueOnce(undefined)

    const deleteResult = await runBulkDelete({
      selectedConversationIds: ["chat-1", "chat-2", "chat-3"],
      deleteConversation,
      t
    })

    expect(deleteConversation).toHaveBeenCalledTimes(3)
    expect(Array.from(deleteResult?.deletedConversationIds || [])).toEqual([
      "chat-1",
      "chat-3"
    ])
    expect(Array.from(deleteResult?.failedConversationIds || [])).toEqual([
      "chat-2"
    ])
    expect(mocks.messageError).toHaveBeenCalledWith(
      "Some chats could not be moved to trash."
    )
    expect(mocks.messageSuccess).not.toHaveBeenCalled()
  })

  it("returns null when no server conversations are selected", async () => {
    const deleteConversation = vi.fn()
    const deleteResult = await runBulkDelete({
      selectedConversationIds: [],
      deleteConversation,
      t
    })

    expect(deleteResult).toBeNull()
    expect(deleteConversation).not.toHaveBeenCalled()
    expect(mocks.messageError).not.toHaveBeenCalled()
    expect(mocks.messageSuccess).not.toHaveBeenCalled()
  })

  it("reports delete failure when all selected deletes fail", async () => {
    const deleteConversation = vi
      .fn()
      .mockRejectedValue(new Error("delete failed"))

    const deleteResult = await runBulkDelete({
      selectedConversationIds: ["chat-1", "chat-2"],
      deleteConversation,
      t
    })

    expect(deleteConversation).toHaveBeenCalledTimes(2)
    expect(Array.from(deleteResult?.deletedConversationIds || [])).toEqual([])
    expect(Array.from(deleteResult?.failedConversationIds || [])).toEqual([
      "chat-1",
      "chat-2"
    ])
    expect(mocks.messageError).toHaveBeenCalledWith(
      "Unable to move selected chats to trash."
    )
    expect(mocks.messageSuccess).not.toHaveBeenCalled()
  })
})

it("does not publish obsolete bulk feedback after its account or view is retired", async () => {
  mocks.messageError.mockClear()
  mocks.messageSuccess.mockClear()
  let current = true
  let acknowledge!: () => void
  const deleteConversation = vi.fn(
    () =>
      new Promise<void>((resolve) => {
        acknowledge = resolve
      })
  )
  const pending = runBulkDelete({
    selectedConversationIds: ["A"],
    deleteConversation,
    t,
    isCurrent: () => current
  })
  current = false
  acknowledge()
  expect(await pending).toBeNull()
  expect(mocks.messageSuccess).not.toHaveBeenCalled()
  expect(mocks.messageError).not.toHaveBeenCalled()
})
