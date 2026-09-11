import { beforeEach, describe, expect, it, vi } from "vitest"

import { resolveWebsiteChatContext } from "@/hooks/useMessage.website-context"

type WebsiteRagClient = Parameters<typeof resolveWebsiteChatContext>[0]["client"]

const makeInput = (client: WebsiteRagClient) => ({
  client,
  embedURL: "https://example.com/article",
  embedType: "html",
  embedHTML: "inline website content",
  embedPDF: [],
  maxWebsiteContext: 100,
  query: "What changed?",
})

describe("useMessage website RAG boundary", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => undefined)
  })

  it("scopes the actual media/add result envelope to its persisted media id", async () => {
    const client = {
      initialize: vi.fn().mockResolvedValue(undefined),
      addMedia: vi.fn().mockResolvedValue({
        results: [{ status: "Success", db_id: 321 }],
      }),
      ragSearch: vi.fn().mockResolvedValue({
        results: [
          {
            content: "persisted website chunk",
            metadata: {
              source: "Article",
              type: "html",
              url: "https://example.com/article",
            },
          },
        ],
      }),
    }

    const result = await resolveWebsiteChatContext(makeInput(client))

    expect(client.addMedia).toHaveBeenCalledWith("https://example.com/article")
    expect(client.ragSearch).toHaveBeenCalledWith("What changed?", {
      top_k: 4,
      sources: ["media_db"],
      include_media_ids: [321],
    })
    expect(result).toMatchObject({
      context: "<doc id='0'>persisted website chunk</doc>",
    })
  })

  it("binds website ingest and retrieval to the captured request scope", async () => {
    const signal = new AbortController().signal
    const requestScope = {
      config: {
        serverUrl: "https://api.example.test",
        authMode: "multi-user" as const,
      },
      userId: 42,
    }
    const client = {
      initialize: vi.fn().mockResolvedValue(undefined),
      addMedia: vi.fn().mockResolvedValue({
        results: [{ status: "Success", db_id: 321 }],
      }),
      ragSearch: vi.fn().mockResolvedValue({ results: [] }),
    }

    await resolveWebsiteChatContext({
      ...makeInput(client),
      signal,
      requestScope,
    })

    expect(client.addMedia).toHaveBeenCalledWith(
      "https://example.com/article",
      { signal, requestScope },
    )
    expect(client.ragSearch).toHaveBeenCalledWith("What changed?", {
      top_k: 4,
      sources: ["media_db"],
      include_media_ids: [321],
      signal,
      requestScope,
    })
  })

  it("does not turn a request-scope rejection into inline fallback", async () => {
    const scopeError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" },
      },
    })
    const client = {
      initialize: vi.fn().mockResolvedValue(undefined),
      addMedia: vi.fn().mockRejectedValue(scopeError),
      ragSearch: vi.fn(),
    }

    await expect(
      resolveWebsiteChatContext({
        ...makeInput(client),
        signal: new AbortController().signal,
        requestScope: {
          config: {
            serverUrl: "https://api.example.test",
            authMode: "multi-user" as const,
          },
          userId: 42,
        },
      }),
    ).rejects.toBe(scopeError)
    expect(client.ragSearch).not.toHaveBeenCalled()
  })

  it("uses inline content and never searches the whole corpus when media/add omits an id", async () => {
    const client = {
      initialize: vi.fn().mockResolvedValue(undefined),
      addMedia: vi.fn().mockResolvedValue({ results: [{ status: "Success" }] }),
      ragSearch: vi.fn(),
    }

    const result = await resolveWebsiteChatContext(makeInput(client))

    expect(client.ragSearch).not.toHaveBeenCalled()
    expect(result).toMatchObject({
      context: "inline website content",
    })
  })

  it("uses inline content and never searches the whole corpus when media/add fails", async () => {
    const client = {
      initialize: vi.fn().mockResolvedValue(undefined),
      addMedia: vi.fn().mockRejectedValue(new Error("ingest failed")),
      ragSearch: vi.fn(),
    }

    const result = await resolveWebsiteChatContext(makeInput(client))

    expect(client.ragSearch).not.toHaveBeenCalled()
    expect(result).toMatchObject({
      context: "inline website content",
    })
  })
})
