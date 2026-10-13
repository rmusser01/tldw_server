import { beforeEach, describe, expect, it, vi } from "vitest"
const mocks = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.request }))
vi.mock("@/db/dexie/schema", () => ({ db: {} }))
import { tldwMedia } from "../TldwMedia"

describe("explicit public article extraction", () => {
  beforeEach(() => mocks.request.mockReset())
  it("does no extraction until explicitly called and forwards a credential-free disabled-analysis profile", async () => {
    expect(mocks.request).not.toHaveBeenCalled()
    const options = {
      signal: new AbortController().signal,
      requestScope: {
        config: {
          serverUrl: "https://owner.example",
          authMode: "multi-user" as const
        },
        userId: 42
      }
    }
    const response = {
      status: "success",
      message: "Extracted",
      results: [
        {
          content: "  full 🐎 text  ",
          title: "Article",
          ingested_at: "2026-10-07T00:00:00+00:00"
        }
      ]
    }
    mocks.request.mockResolvedValue(response)
    expect(
      await tldwMedia.extractPublicArticle("https://EXAMPLE.com/%61", options)
    ).toBe(response)
    expect(mocks.request).toHaveBeenCalledWith({
      path: "/api/v1/media/ingest-web-content",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-TLDW-Expected-User-ID": "42"
      },
      abortSignal: options.signal,
      servicePromptConfig: {
        ...options.requestScope.config,
        expectedUserId: 42
      },
      body: {
        urls: ["https://EXAMPLE.com/%61"],
        scrape_method: "individual",
        credential_free: true,
        perform_analysis: false,
        perform_translation: false,
        perform_chunking: false,
        auto_chunking_use_llm: false,
        use_cookies: false,
        overwrite_existing: false,
        perform_rolling_summarization: false,
        perform_confabulation_check_of_analysis: false
      }
    })
  })
  it("retains oversized extraction verbatim for acceptance to reject", async () => {
    const response = {
      status: "error",
      results: [
        {
          content: "x".repeat(1_000_001),
          extraction_successful: false,
          error: "Article text exceeds the capture body limit"
        }
      ]
    }
    mocks.request.mockResolvedValue(response)
    expect(await tldwMedia.extractPublicArticle("https://example.com")).toBe(
      response
    )
  })
})
