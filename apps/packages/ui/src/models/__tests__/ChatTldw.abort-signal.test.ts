import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  sendMessage: vi.fn(),
  streamMessage: vi.fn()
}))

vi.mock("@/services/tldw", () => ({
  tldwChat: {
    sendMessage: mocks.sendMessage,
    streamMessage: mocks.streamMessage
  }
}))

import { ChatTldw } from "@/models/ChatTldw"
import { HumanMessage } from "@/types/messages"

describe("ChatTldw abort signal threading", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("threads the UI AbortSignal into tldwChat.streamMessage options", async () => {
    mocks.streamMessage.mockImplementation(async function* () {
      yield "hi"
    })

    const controller = new AbortController()
    const model = new ChatTldw({ model: "tldw:gpt-test", streaming: true })

    for await (const _token of await model.stream([new HumanMessage("Hi")], {
      signal: controller.signal
    })) {
      // consume the stream
    }

    expect(mocks.streamMessage).toHaveBeenCalledTimes(1)
    const options = mocks.streamMessage.mock.calls[0][1] as {
      signal?: AbortSignal
    }
    expect(options.signal).toBe(controller.signal)
  })

  it("threads an invoke AbortSignal into the non-streaming request", async () => {
    mocks.sendMessage.mockResolvedValue("rewritten")
    const controller = new AbortController()
    const model = new ChatTldw({ model: "tldw:gpt-test" })

    await expect(model.invoke([new HumanMessage("Hi")], {
      signal: controller.signal
    })).resolves.toEqual({ content: "rewritten" })

    expect(mocks.sendMessage.mock.calls[0]?.[1]).toMatchObject({
      signal: controller.signal
    })
  })

  it("threads the captured request scope into streaming and non-streaming chat", async () => {
    mocks.streamMessage.mockImplementation(async function* () {
      yield "hi"
    })
    mocks.sendMessage.mockResolvedValue("answer")
    const requestScope = {
      config: {
        serverUrl: "https://research-one.test",
        authMode: "multi-user" as const
      },
      userId: 42
    }
    const model = new ChatTldw({
      model: "tldw:gpt-test",
      streaming: true,
      requestScope
    })

    for await (const _token of await model.stream([new HumanMessage("Hi")])) {
      // consume the stream
    }
    await model.invoke([new HumanMessage("Hi")])

    expect(mocks.streamMessage.mock.calls[0]?.[1]).toMatchObject({
      requestScope
    })
    expect(mocks.sendMessage.mock.calls[0]?.[1]).toMatchObject({
      requestScope
    })
  })
})
