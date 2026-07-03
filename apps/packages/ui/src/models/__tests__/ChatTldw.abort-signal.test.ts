import { describe, expect, it, vi } from "vitest"

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
import { HumanMessage } from "@/types/messages"

describe("ChatTldw abort signal threading", () => {
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
})
