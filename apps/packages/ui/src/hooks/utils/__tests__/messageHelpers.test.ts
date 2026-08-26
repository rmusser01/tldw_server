import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  createSaveMessageOnError,
  createSaveMessageOnSuccess
} from "../messageHelpers"

const mocks = vi.hoisted(() => ({
  saveError: vi.fn(async () => "history-error"),
  saveSuccess: vi.fn(async () => "history-success")
}))

vi.mock("../../chat-helper", () => ({
  saveMessageOnError: (...args: unknown[]) =>
    (mocks.saveError as (...args: unknown[]) => unknown)(...args),
  saveMessageOnSuccess: (...args: unknown[]) =>
    (mocks.saveSuccess as (...args: unknown[]) => unknown)(...args)
}))

describe("message helper wrappers", () => {
  beforeEach(() => {
    mocks.saveError.mockClear()
    mocks.saveSuccess.mockClear()
  })

  it("injects setHistory and setHistoryId defaults for saveMessageOnError", async () => {
    const setHistory = vi.fn()
    const setHistoryId = vi.fn()
    const wrapped = createSaveMessageOnError(false, [], setHistory, setHistoryId)

    await wrapped({ userMessage: "hello" })

    expect(mocks.saveError).toHaveBeenCalledTimes(1)
    const payload = (mocks.saveError.mock.calls[0] as unknown[] | undefined)?.[0] as
      | Record<string, unknown>
      | undefined
    expect(payload.setHistory).toBe(setHistory)
    expect(payload.setHistoryId).toBe(setHistoryId)
  })

  it("preserves explicit setters when provided", async () => {
    const defaultSetHistory = vi.fn()
    const defaultSetHistoryId = vi.fn()
    const explicitSetHistory = vi.fn()
    const explicitSetHistoryId = vi.fn()
    const wrapped = createSaveMessageOnError(
      false,
      [],
      defaultSetHistory,
      defaultSetHistoryId
    )

    await wrapped({
      setHistory: explicitSetHistory,
      setHistoryId: explicitSetHistoryId
    })

    const payload = (mocks.saveError.mock.calls[0] as unknown[] | undefined)?.[0] as
      | Record<string, unknown>
      | undefined
    expect(payload.setHistory).toBe(explicitSetHistory)
    expect(payload.setHistoryId).toBe(explicitSetHistoryId)
  })

  it("injects default setHistoryId for saveMessageOnSuccess", async () => {
    const setHistoryId = vi.fn()
    const wrapped = createSaveMessageOnSuccess(false, setHistoryId)

    await wrapped({})

    expect(mocks.saveSuccess).toHaveBeenCalledTimes(1)
    const payload = (mocks.saveSuccess.mock.calls[0] as unknown[] | undefined)?.[0] as
      | Record<string, unknown>
      | undefined
    expect(payload.setHistoryId).toBe(setHistoryId)
  })

  it("links a discovered server conversation id after a successful save", async () => {
    const setHistoryId = vi.fn()
    const onServerConversationLinked = vi.fn()
    const wrapped = createSaveMessageOnSuccess(false, setHistoryId, {
      onServerConversationLinked
    })

    await wrapped({
      conversationId: "server-chat-42"
    })

    expect(onServerConversationLinked).toHaveBeenCalledWith("server-chat-42")
  })

  it("preserves the active server chat while assigning its local mirror id", async () => {
    const setHistoryId = vi.fn()
    const wrapped = createSaveMessageOnSuccess(false, setHistoryId)

    await wrapped({
      conversationId: "server-chat-42"
    })

    const payload = (mocks.saveSuccess.mock.calls[0] as unknown[] | undefined)?.[0] as
      | { setHistoryId?: (id: string) => void }
      | undefined
    payload?.setHistoryId?.("local-history-7")

    expect(setHistoryId).toHaveBeenCalledWith("local-history-7", {
      preserveServerChatId: true
    })
  })

  it("preserves the active server chat through an explicit pipeline setter", async () => {
    const defaultSetHistoryId = vi.fn()
    const explicitSetHistoryId = vi.fn()
    const wrapped = createSaveMessageOnSuccess(false, defaultSetHistoryId)

    await wrapped({
      conversationId: "server-chat-42",
      setHistoryId: explicitSetHistoryId
    })

    const payload = (mocks.saveSuccess.mock.calls[0] as unknown[] | undefined)?.[0] as
      | { setHistoryId?: (id: string) => void }
      | undefined
    payload?.setHistoryId?.("local-history-7")

    expect(explicitSetHistoryId).toHaveBeenCalledWith("local-history-7", {
      preserveServerChatId: true
    })
    expect(defaultSetHistoryId).not.toHaveBeenCalled()
  })
})
