import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  streamMessage: vi.fn()
}))

vi.mock("@/services/tldw", async () => {
  const actual =
    await vi.importActual<typeof import("@/services/tldw")>("@/services/tldw")
  return {
    ...actual,
    tldwChat: {
      ...actual.tldwChat,
      streamMessage: mocks.streamMessage
    }
  }
})

import { ChatTldw } from "@/models/ChatTldw"
import {
  beginServerChatWrite,
  getServerChatSaveStatus,
  resetServerChatSaveStatus
} from "@/store/server-chat-save-status"
import { HumanMessage } from "@/types/messages"

const consume = async (stream: AsyncGenerator<unknown>) => {
  for await (const _token of stream) {
    // drain
  }
}

// CS-03 / XS-05 (#3104): a completion that the server persists into an
// existing server chat is a server write, so its outcome feeds the labels.
describe("ChatTldw server-persisted completions record the save status", () => {
  beforeEach(() => {
    mocks.streamMessage.mockReset()
    resetServerChatSaveStatus()
  })

  it("marks the server chat saving while streaming and saved when the stream completes", async () => {
    let statusDuringStream: string | null = null
    mocks.streamMessage.mockImplementationOnce(async function* () {
      statusDuringStream = getServerChatSaveStatus("server-chat")
      yield "Saved reply"
    })
    const model = new ChatTldw({
      model: "gpt-test",
      saveToDb: true,
      conversationId: "server-chat"
    })

    await consume(await model.stream([new HumanMessage("Hi")]))

    expect(statusDuringStream).toBe("saving")
    expect(getServerChatSaveStatus("server-chat")).toBe("saved")
  })

  it("marks the server chat failed when the persisted completion errors", async () => {
    mocks.streamMessage.mockImplementationOnce(async function* () {
      yield "Partial"
      throw new Error("provider failed")
    })
    const model = new ChatTldw({
      model: "gpt-test",
      saveToDb: true,
      conversationId: "server-chat"
    })

    await expect(
      consume(await model.stream([new HumanMessage("Hi")]))
    ).rejects.toThrow("provider failed")

    expect(getServerChatSaveStatus("server-chat")).toBe("failed")
  })

  it("keeps the previous outcome when the user stops the reply", async () => {
    beginServerChatWrite("server-chat")("saved")
    const controller = new AbortController()
    mocks.streamMessage.mockImplementationOnce(async function* () {
      yield "Partial"
      controller.abort()
      yield "ignored"
    })
    const model = new ChatTldw({
      model: "gpt-test",
      saveToDb: true,
      conversationId: "server-chat"
    })

    await consume(
      await model.stream([new HumanMessage("Hi")], { signal: controller.signal })
    )

    expect(getServerChatSaveStatus("server-chat")).toBe("saved")
  })

  it("does not track completions that are not saved to a server chat", async () => {
    mocks.streamMessage.mockImplementation(async function* () {
      yield "Local reply"
    })

    await consume(
      await new ChatTldw({
        model: "gpt-test",
        saveToDb: false,
        conversationId: "server-chat"
      }).stream([new HumanMessage("Hi")])
    )
    await consume(
      await new ChatTldw({
        model: "gpt-test",
        clientManagedHistory: true,
        saveToDb: true,
        conversationId: "server-chat"
      }).stream([new HumanMessage("Hi")])
    )

    expect(getServerChatSaveStatus("server-chat")).toBe("unknown")
  })
})
