import { beforeEach, describe, expect, it, vi } from "vitest"
const calls = vi.hoisted(() => ({ stream: vi.fn(), request: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgStream: (...args: unknown[]) => calls.stream(...args),
  bgRequest: (...args: unknown[]) => calls.request(...args), bgUpload: vi.fn() }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: vi.fn(async () => null), set: vi.fn(async () => undefined), remove: vi.fn(async () => undefined) }),
  safeStorageSerde: { serialize: (value: unknown) => value, deserialize: (value: unknown) => value }
}))
import { chatRagMethods } from "../chat-rag"
import { TldwApiClient, type ChatCompletionRequest } from "../../TldwApiClient"
const request = (stream: unknown = true) => ({ messages: [{ role: "user", content: "Original" }], model: "actual",
  stream, tldw_turn: { user_message_id: "12345678-1234-4321-8123-123456789abc", history_v1: { version: 1 } }
}) as unknown as ChatCompletionRequest
const consume = async (body: ChatCompletionRequest) => {
  const chunks = []
  for await (const chunk of chatRagMethods.streamChatCompletion.call({} as never, body)) chunks.push(chunk)
  return chunks
}
beforeEach(() => { calls.stream.mockReset(); calls.request.mockReset() })
describe("registered canonical completion scope", () => {
  it.each([
    [{ type: "workspace" as const, workspaceId: "space / &" }, "/api/v1/chat/completions?scope_type=workspace&workspace_id=space+%2F+%26"],
    [{ type: "global" as const }, "/api/v1/chat/completions?scope_type=global"],
    [undefined, "/api/v1/chat/completions"]
  ] as const)("pins nonstream scope without changing body or response: %j", async (scope, path) => {
    const body = Object.freeze({ messages: [{ role: "user" as const, content: "Exact original" }], model: "actual", stream: false })
    const completion = { choices: [{ message: { content: "An exception at /Users/example/file is legitimate content." } }] }
    calls.request.mockResolvedValue(completion)
    const signal = new AbortController().signal
    const requestScope = { config: { serverUrl: "https://pinned.example", authMode: "multi-user" as const }, userId: 7 }
    const response = await new TldwApiClient().createChatCompletion(body, { scope, requestScope, signal, timeoutMs: 1234 })
    expect(calls.request).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ path, method: "POST", body,
      headers: { "Content-Type": "application/json", "X-TLDW-Expected-User-ID": "7" }, abortSignal: signal, timeoutMs: 1234,
      servicePromptConfig: { ...requestScope.config, expectedUserId: 7 } }))
    expect(calls.request.mock.calls[0][0].body).toBe(body)
    expect(await response.json()).toEqual(completion)
  })
})
describe("nested selected durable stream", () => {
  it("preserves the exact prepared body instead of rewriting stream", async () => {
    const body = Object.freeze(request())
    calls.stream.mockImplementation(async function* () { yield '{"ok":true}' })
    expect(await consume(body)).toEqual([{ ok: true }])
    expect(calls.stream.mock.calls[0][0].body).toBe(body)
  })
  it.each([false, undefined, "true"])("rejects non-explicit stream=true (%s) before dispatch", async stream => {
    calls.stream.mockImplementation(async function* () { yield '{"ok":true}' })
    await expect(consume({ ...request(), stream } as ChatCompletionRequest)).rejects.toThrow()
    expect(calls.stream).not.toHaveBeenCalled()
  })
  it.each(["not-json", "[]", "null", "true", '"text"'])("fails closed on malformed nested SSE %s", async line => {
    calls.stream.mockImplementation(async function* () { yield line })
    await expect(consume(request())).rejects.toThrow()
  })
  it("ignores empty/comment/DONE control lines for nested mode", async () => {
    calls.stream.mockImplementation(async function* () { yield " "; yield ": keepalive"; yield "[DONE]" })
    expect(await consume(request())).toEqual([])
  })
  it("keeps the generic stream rewrite and tolerant parse behavior", async () => {
    calls.stream.mockImplementation(async function* () { yield "bad"; yield '{"ok":true}' })
    const bare = { messages: [], model: "legacy", stream: false }
    expect(await consume(bare)).toEqual([{ ok: true }])
    expect(calls.stream.mock.calls[0][0].body.stream).toBe(true)
  })
})
