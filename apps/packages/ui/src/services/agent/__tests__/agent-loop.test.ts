import { expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgStream: vi.fn(),
  executeTool: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({ bgStream: mocks.bgStream }))
vi.mock("@/services/native/native-client", () => ({ executeTool: mocks.executeTool }))

import { AgentLoop } from "../agent-loop"

it("replays the signed Gemini function call before its result", async () => {
  const extraContent = { google: { thought_signature: "c2lnbmF0dXJl" } }
  mocks.executeTool.mockResolvedValue({ ok: true })
  mocks.bgStream
    .mockImplementationOnce(async function* () {
      yield `data: ${JSON.stringify({
        choices: [{ delta: { tool_calls: [{
          index: 0,
          id: "call-1",
          function: { name: "fs_read", arguments: "{}" },
          extra_content: extraContent
        }] } }]
      })}`
    })
    .mockImplementationOnce(async function* () {
      yield 'data: {"choices":[{"delta":{"content":"Done"}}]}'
    })

  const agent = new AgentLoop("workspace", "Read a file", {
    model: "gemini-3-flash-preview",
    maxSteps: 2
  }, vi.fn())

  await expect(agent.run()).resolves.toMatchObject({ status: "complete", response: "Done" })
  expect(mocks.bgStream.mock.lastCall?.[0].body.messages.slice(2)).toEqual([
    {
      role: "assistant",
      content: "",
      tool_calls: [{
        id: "call-1",
        type: "function",
        function: { name: "fs_read", arguments: "{}" },
        extra_content: extraContent
      }]
    },
    { role: "tool", content: '{"ok":true}', tool_call_id: "call-1" }
  ])
})

it.each([
  { model: "moonshot:kimi-k3.1-pro", chunks: ["  first ", "second\n"], expected: "  first second\n" },
  { model: "kimi-k3.1-pro", chunks: ["first ", "second"], expected: "first second" },
  { model: "openrouter:moonshotai/kimi-k2.6", chunks: ["provider-specific reasoning"], expected: undefined },
  { model: "kimi-k3.1-pro", chunks: ["a".repeat(100_000), "a"], expected: undefined }
])("replays bounded Moonshot tool reasoning for $model", async ({ model, chunks, expected }) => {
  mocks.executeTool.mockResolvedValue({ ok: true })
  mocks.bgStream
    .mockImplementationOnce(async function* () {
      for (const reasoning_content of chunks) {
        yield `data: ${JSON.stringify({ choices: [{ delta: { reasoning_content } }] })}`
      }
      yield `data: ${JSON.stringify({
        choices: [{ delta: { tool_calls: [{
          id: "call-1",
          function: { name: "fs_read", arguments: "{}" }
        }] } }]
      })}`
    })
    .mockImplementationOnce(async function* () {
      yield 'data: {"choices":[{"delta":{"content":"Done"}}]}'
    })

  const agent = new AgentLoop("workspace", "Read a file", { model, maxSteps: 2 }, vi.fn())
  await expect(agent.run()).resolves.toMatchObject({ status: "complete" })

  const assistant = mocks.bgStream.mock.lastCall?.[0].body.messages[2]
  expect(assistant).toMatchObject({ role: "assistant" })
  if (expected === undefined) {
    expect(assistant).not.toHaveProperty("reasoning_content")
  } else {
    expect(assistant.reasoning_content).toBe(expected)
  }
})
