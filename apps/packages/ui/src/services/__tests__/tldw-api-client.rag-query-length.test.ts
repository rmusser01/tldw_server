import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  bgStream: vi.fn(),
  tldwRequest: vi.fn(),
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args),
}))

vi.mock("@/services/tldw/request-core", () => ({
  tldwRequest: (...args: unknown[]) => mocks.tldwRequest(...args),
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async (key: string) =>
      key === "tldwConfig"
        ? {
            serverUrl: "http://127.0.0.1:8000",
            authMode: "single-user",
            apiKey: "test-key-not-placeholder",
          }
        : null
    ),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined),
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value,
  },
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"

const OVER_LIMIT_QUERY = `${"a".repeat(20005)}${" ".repeat(20)}`

describe("TldwApiClient RAG query length guard", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.spyOn(console, "warn").mockImplementation(() => undefined)
  })

  it("truncates ragSearch query payloads to backend-safe length", async () => {
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { results: [], answer: null },
    })

    const client = new TldwApiClient()
    await client.ragSearch(OVER_LIMIT_QUERY, { top_k: 5 })

    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
    expect(mocks.tldwRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/rag/search",
        body: expect.objectContaining({
          query: expect.any(String),
          top_k: 5,
        }),
      }),
      expect.any(Object)
    )
    const body = mocks.tldwRequest.mock.calls[0][0]?.body as Record<string, unknown>
    expect((body.query as string).length).toBeLessThanOrEqual(20000)
  })

  it("truncates ragSearchStream query payloads to backend-safe length", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      yield JSON.stringify({ type: "delta", text: "ok" })
    })

    const client = new TldwApiClient()
    const iterator = client.ragSearchStream(OVER_LIMIT_QUERY, { top_k: 3 })
    await iterator.next()

    expect(mocks.bgStream).toHaveBeenCalledTimes(1)
    const payload = mocks.bgStream.mock.calls[0][0] as Record<string, unknown>
    const body = payload.body as Record<string, unknown>
    expect(payload.path).toBe("/api/v1/rag/search/stream")
    expect(body.top_k).toBe(3)
    expect((body.query as string).length).toBeLessThanOrEqual(20000)
  })

  it("truncates ragSimple query payloads to backend-safe length", async () => {
    mocks.bgRequest.mockResolvedValue({ answer: null })

    const client = new TldwApiClient()
    await client.ragSimple(OVER_LIMIT_QUERY, { mode: "fast" })

    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/rag/simple",
        body: expect.objectContaining({
          query: expect.any(String),
          mode: "fast",
        }),
      })
    )
    const body = mocks.bgRequest.mock.calls[0][0]?.body as Record<string, unknown>
    expect((body.query as string).length).toBeLessThanOrEqual(20000)
  })

  it("sanitizes non-retryable ragSearch failures before surfacing them", async () => {
    mocks.tldwRequest.mockResolvedValue({
      ok: false,
      status: 403,
      error: "Request failed: 403 (POST /api/v1/rag/search) trace=/Users/private/dev.log",
      data: null,
    })

    const client = new TldwApiClient()

    await expect(client.ragSearch("blocked", { top_k: 5 })).rejects.toMatchObject({
      message: "RAG search failed. Access was denied.",
      status: 403,
    })
  })

  it("does not replay ragSearch POST requests after HTTP 500", async () => {
    mocks.tldwRequest.mockResolvedValue({
      ok: false,
      status: 500,
      error: "Request failed: 500 (POST /api/v1/rag/search) stacktrace",
      data: null,
    })

    const client = new TldwApiClient()

    await expect(
      client.ragSearch("retry me", { top_k: 5, enable_reranking: true })
    ).rejects.toMatchObject({
      message: "RAG search failed due to a server error.",
      status: 500,
    })

    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
    expect(mocks.tldwRequest.mock.calls[0][0]).toMatchObject({
      path: "/api/v1/rag/search",
      body: expect.objectContaining({
        query: "retry me",
        enable_reranking: true,
      }),
    })
  })
})
