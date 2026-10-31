import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgStream: vi.fn(),
  bgUpload: vi.fn()
}))

import { chatRagMethods } from "../chat-rag"

describe("chat RAG scoped errors", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("preserves a structured request-scope rejection", async () => {
    const scopeError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })
    mocks.bgRequest.mockRejectedValueOnce(scopeError)

    const request = chatRagMethods.ragSearch.call(
      {
        normalizeRagQuery: (query: string) => query
      } as any,
      "question",
      {
        requestScope: {
          config: {
            serverUrl: "https://server.test",
            authMode: "multi-user"
          },
          userId: 42
        }
      }
    )

    await expect(request).rejects.toBe(scopeError)
  })
})
it.each(["getChat", "getChatSettings", "updateChatSettings"] as const)("forwards scoped credentials, workspace and cancellation for %s", async method => {
  mocks.bgRequest.mockReset().mockResolvedValue({id: "child", conversation_id: "child", settings: {}})
  const signal = new AbortController().signal
  const options = {scope: {type: "workspace" as const, workspaceId: "original"}, signal,
    requestScope: {config: {serverUrl: "https://original.test", authMode: "multi-user" as const}, userId: "alice"}}
  const client = {normalizeChatSummary: (value: any) => value} as any
  if (method === "updateChatSettings") await chatRagMethods[method].call(client, "child", {authorNote: "explicit"}, options)
  else await chatRagMethods[method].call(client, "child", options)
  expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({
    path: expect.stringContaining("scope_type=workspace&workspace_id=original"), abortSignal: signal,
    headers: expect.objectContaining({"X-TLDW-Expected-User-ID": "alice"}),
    servicePromptConfig: expect.objectContaining({serverUrl: "https://original.test", expectedUserId: "alice"})
  }))
})
