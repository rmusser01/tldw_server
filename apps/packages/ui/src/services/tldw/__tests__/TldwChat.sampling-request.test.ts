import { describe, expect, it } from "vitest"

import { prepareChatCompletionRequest } from "../TldwChat"

const messages = [{ role: "user" as const, content: "Hello" }]

// #3283: the chat's sampling settings must reach /chat/completions.
describe("sampling settings in the chat request", () => {
  it("sends both penalties, Top K and Min P under the server's field names", () => {
    const request = prepareChatCompletionRequest(messages, {
      model: "openrouter/model",
      frequencyPenalty: 0.4,
      presencePenalty: 0.2,
      topK: 40,
      minP: 0.05
    })
    expect(request).toMatchObject({
      frequency_penalty: 0.4,
      presence_penalty: 0.2,
      topk: 40,
      minp: 0.05
    })
  })

  it("leaves Top K and Min P out of the body when they are not set", () => {
    const body = JSON.parse(
      JSON.stringify(prepareChatCompletionRequest(messages, { model: "openai/gpt-4o" }))
    )
    expect(body).not.toHaveProperty("topk")
    expect(body).not.toHaveProperty("minp")
  })
})
