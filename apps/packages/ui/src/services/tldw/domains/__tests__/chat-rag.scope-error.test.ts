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
import type { TldwApiClientCore } from "../../TldwApiClient"

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


describe("authoritative detail history versions", () => {
  const client = {
    normalizeChatSummary: chatRagMethods.normalizeChatSummary
  } as unknown as TldwApiClientCore
  it.each([0, 1, Number.MAX_SAFE_INTEGER])(
    "preserves safe history version %s",
    async (version) => {
      mocks.bgRequest.mockResolvedValueOnce({
        id: "child",
        history_version: version
      })
      const result = await chatRagMethods.getChat.call(client, "child", {
        requireHistoryVersion: true
      })
      expect(result.history_version).toBe(version)
    }
  )
  it.each([undefined, null, "4", -1, 1.5, NaN, Number.MAX_SAFE_INTEGER + 1])(
    "does not certify malformed history version %s",
    async (history_version) => {
      mocks.bgRequest.mockResolvedValueOnce({ id: "child", history_version })
      await expect(
        chatRagMethods.getChat.call(client, "child", {
          requireHistoryVersion: true
        })
      ).rejects.toThrow("unsupported_history_version")
    }
  )
  it.each([
    {},
    { id: "foreign", history_version: 3 },
    { chat_id: "child", history_version: 3 },
    { id: 1, history_version: 3 }
  ])("rejects unbound detail %j before normalization", async (body) => {
    mocks.bgRequest.mockResolvedValueOnce(body)
    await expect(
      chatRagMethods.getChat.call(client, "child", {
        requireHistoryVersion: true
      })
    ).rejects.toThrow("unsupported_history_version")
  })
  it("keeps an older detail readable without certifying a history version", async () => {
    mocks.bgRequest.mockResolvedValueOnce({ id: "child" })
    const result = await chatRagMethods.getChat.call(client, "child")
    expect(result.history_version).toBeNull()
  })
})

it.each([
  { scope_type: "global", workspace_id: null },
  { scope_type: "workspace", workspace_id: "foreign" },
  { workspace_id: "original" }
])(
  "cannot use a history version from foreign or absent workspace metadata %j",
  async (scope) => {
    mocks.bgRequest.mockResolvedValueOnce({
      id: "child",
      history_version: 3,
      ...scope
    })
    await expect(
      chatRagMethods.getChat.call(
        {
          normalizeChatSummary: chatRagMethods.normalizeChatSummary
        } as unknown as TldwApiClientCore,
        "child",
        {
          requireHistoryVersion: true,
          scope: { type: "workspace", workspaceId: "original" }
        }
      )
    ).rejects.toThrow("unsupported_history_version")
  }
)
