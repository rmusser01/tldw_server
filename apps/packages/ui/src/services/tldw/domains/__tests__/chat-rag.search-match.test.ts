import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgStream: vi.fn(),
  bgUpload: vi.fn()
}))

import type { TldwApiClientCore } from "../../TldwApiClient"
import { chatRagMethods } from "../chat-rag"

const client = {
  normalizeChatSummary: chatRagMethods.normalizeChatSummary
} as unknown as TldwApiClientCore

describe("conversation search match details (CS-02)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("sends search_in and keeps the server's match snippet on each chat", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      items: [
        {
          id: "chat-1",
          title: "Weekly sync",
          created_at: "2026-03-08T00:00:00Z",
          last_modified: "2026-03-08T00:01:00Z",
          matched_in: ["content"],
          match_snippet: "…the launch codeword is zebrafinch.",
          match_message_id: "msg-1"
        },
        {
          id: "chat-2",
          title: "Zebrafinch notes",
          created_at: "2026-03-07T00:00:00Z",
          matched_in: ["title"],
          match_snippet: null,
          match_message_id: null
        }
      ],
      pagination: { limit: 50, offset: 0, total: 2, has_more: false }
    })

    const result = await chatRagMethods.searchConversationsWithMeta.call(client, {
      query: "zebrafinch",
      search_in: "title,content"
    })

    const request = mocks.bgRequest.mock.calls[0][0] as { path: string }
    expect(request.path).toContain("/api/v1/chats/conversations?")
    expect(new URLSearchParams(request.path.split("?")[1]).get("search_in")).toBe(
      "title,content"
    )
    expect(result.total).toBe(2)
    expect(result.chats[0]).toMatchObject({
      id: "chat-1",
      matched_in: ["content"],
      match_snippet: "…the launch codeword is zebrafinch.",
      match_message_id: "msg-1"
    })
    expect(result.chats[1]).toMatchObject({
      id: "chat-2",
      matched_in: ["title"],
      match_snippet: null,
      match_message_id: null
    })
  })

  it("leaves match details empty for servers that do not report them", () => {
    const summary = chatRagMethods.normalizeChatSummary({
      id: "chat-3",
      title: "Lunch plans",
      created_at: "2026-03-06T00:00:00Z",
      matched_in: "content",
      match_snippet: "   "
    })

    expect(summary.matched_in).toBeNull()
    expect(summary.match_snippet).toBeNull()
    expect(summary.match_message_id).toBeNull()
  })
})
