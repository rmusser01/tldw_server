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

import {
  TldwApiClient,
  TldwApiClientBase,
} from "@/services/tldw/TldwApiClient"
import { chatRagMethods } from "@/services/tldw/domains/chat-rag"

const OVER_LIMIT_QUERY = `${"a".repeat(20005)}${" ".repeat(20)}`
const RAG_SIMPLE_MAX_PATH_LENGTH = 8000
const BROWSER_PROVIDER_SECRET =
  "sk-browser-provider-secret-must-not-reach-rag"
const PROTOTYPE_POLLUTION_MARKER =
  "browser-rag-options-must-not-alter-object-prototypes"

const nestedPrototypePollutionPayload = JSON.parse(`{
  "safe_value": "nested-safe-value",
  "__proto__": { "polluted": "${PROTOTYPE_POLLUTION_MARKER}" },
  "constructor": {
    "prototype": { "polluted": "${PROTOTYPE_POLLUTION_MARKER}" }
  },
  "prototype": { "polluted": "${PROTOTYPE_POLLUTION_MARKER}" }
}`) as Record<string, unknown>

class BrowserCredentialBox {
  role = "user"
  content = "unsafe custom history entry"
  api_key = BROWSER_PROVIDER_SECRET

  toJSON() {
    return {
      role: this.role,
      content: this.content,
      api_key: this.api_key,
    }
  }
}

const credentialShapedOptions = {
  generation_provider: "openai",
  generation_model: "gpt-4o-mini",
  max_generation_tokens: 512,
  workspace_id: "workspace-safe",
  api_key: BROWSER_PROVIDER_SECRET,
  provider_api_key: BROWSER_PROVIDER_SECRET,
  secret_key: BROWSER_PROVIDER_SECRET,
  authorization_header: BROWSER_PROVIDER_SECRET,
  bearer: BROWSER_PROVIDER_SECRET,
  cookie: BROWSER_PROVIDER_SECRET,
  HF_TOKEN: BROWSER_PROVIDER_SECRET,
  token: BROWSER_PROVIDER_SECRET,
  id_token: BROWSER_PROVIDER_SECRET,
  oauth2_token: BROWSER_PROVIDER_SECRET,
  base_url: BROWSER_PROVIDER_SECRET,
  api_base_url: BROWSER_PROVIDER_SECRET,
  api_url: BROWSER_PROVIDER_SECRET,
  endpoint: BROWSER_PROVIDER_SECRET,
  credential_fields: {
    access_token: BROWSER_PROVIDER_SECRET,
  },
  app_config: {
    openai_api: {
      api_key: BROWSER_PROVIDER_SECRET,
    },
  },
  chat_history: [
    {
      role: "user",
      content: "safe history entry",
      secret_key: BROWSER_PROVIDER_SECRET,
      authorization_header: BROWSER_PROVIDER_SECRET,
      bearer: BROWSER_PROVIDER_SECRET,
      cookie: BROWSER_PROVIDER_SECRET,
      nested: {
        HF_TOKEN: BROWSER_PROVIDER_SECRET,
        openai_api_key: BROWSER_PROVIDER_SECRET,
        anthropic_api_key: BROWSER_PROVIDER_SECRET,
        aws_access_key_id: BROWSER_PROVIDER_SECRET,
        aws_secret_access_key: BROWSER_PROVIDER_SECRET,
        aws_session_token: BROWSER_PROVIDER_SECRET,
        OPENAI_BASE_URL: BROWSER_PROVIDER_SECRET,
        x_api_key: BROWSER_PROVIDER_SECRET,
        max_tokens: 2048,
        token_count: 17,
        token_budget: 4096,
        ...nestedPrototypePollutionPayload,
      },
    },
    new BrowserCredentialBox(),
  ],
}

const expectServerOwnedProviderBody = (body: Record<string, unknown>) => {
  expect(body).toMatchObject({
    generation_provider: "openai",
    generation_model: "gpt-4o-mini",
    max_generation_tokens: 512,
    workspace_id: "workspace-safe",
  })
  expect(body.chat_history).toEqual([
    {
      role: "user",
      content: "safe history entry",
    },
  ])
  expect(body).not.toHaveProperty("api_key")
  expect(body).not.toHaveProperty("provider_api_key")
  expect(body).not.toHaveProperty("secret_key")
  expect(body).not.toHaveProperty("authorization_header")
  expect(body).not.toHaveProperty("bearer")
  expect(body).not.toHaveProperty("cookie")
  expect(body).not.toHaveProperty("credential_fields")
  expect(body).not.toHaveProperty("app_config")
  expect(JSON.stringify(body)).not.toContain(BROWSER_PROVIDER_SECRET)
  expect(JSON.stringify(body)).not.toContain(PROTOTYPE_POLLUTION_MARKER)
  expect(
    (Object.prototype as Record<string, unknown>).polluted
  ).toBeUndefined()
}

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

  it("keeps provider selection but strips browser credentials from ragSearch", async () => {
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { results: [], answer: null },
    })

    const client = new TldwApiClient()
    await client.ragSearch("server-owned credentials", credentialShapedOptions)

    const body = mocks.tldwRequest.mock.calls[0][0]?.body as Record<
      string,
      unknown
    >
    expectServerOwnedProviderBody(body)
  })

  it("keeps provider selection but strips browser credentials from ragSearchStream", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      yield JSON.stringify({ type: "delta", text: "ok" })
    })

    const client = new TldwApiClient()
    const iterator = client.ragSearchStream(
      "server-owned streaming credentials",
      credentialShapedOptions
    )
    await iterator.next()

    const payload = mocks.bgStream.mock.calls[0][0] as Record<string, unknown>
    const body = payload.body as Record<string, unknown>
    expectServerOwnedProviderBody(body)
  })

  it("positive-allowlists Pydantic chat_history fields at the direct mixin boundary", async () => {
    const requestWithCurrentConfig = vi.fn().mockResolvedValue({ results: [] })

    await chatRagMethods.ragSearch.call(
      {
        normalizeRagQuery: (query: string) => query,
        requestWithCurrentConfig,
      } as any,
      "direct mixin history boundary",
      credentialShapedOptions
    )

    expectServerOwnedProviderBody(
      requestWithCurrentConfig.mock.calls[0][0].body as Record<string, unknown>
    )
  })

  it.each([
    ["mixin", () => new TldwApiClient()],
    ["base", () => new TldwApiClientBase()],
  ])("allows a ragSimple GET path exactly at the documented limit for %s", async (_name, makeClient) => {
    mocks.bgRequest.mockResolvedValue({ answer: null })
    const fixedPathLength = "/api/v1/rag/simple?query=".length + "&top_k=5".length
    const exactQuery = "a".repeat(RAG_SIMPLE_MAX_PATH_LENGTH - fixedPathLength)

    await makeClient().ragSimple(exactQuery, { top_k: 5 })

    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    const request = mocks.bgRequest.mock.calls[0][0] as Record<string, unknown>
    expect(request.method).toBe("GET")
    expect(request).not.toHaveProperty("body")
    expect(String(request.path)).toHaveLength(RAG_SIMPLE_MAX_PATH_LENGTH)
  })

  it.each([
    ["mixin", () => new TldwApiClient()],
    ["base", () => new TldwApiClientBase()],
  ])("rejects an over-limit ragSimple GET path before transport for %s", async (_name, makeClient) => {
    const fixedPathLength = "/api/v1/rag/simple?query=".length + "&top_k=5".length
    const overLimitQuery = "a".repeat(
      RAG_SIMPLE_MAX_PATH_LENGTH - fixedPathLength + 1
    )

    await expect(
      makeClient().ragSimple(overLimitQuery, { top_k: 5 })
    ).rejects.toThrow("RAG simple request URL exceeds the 8,000-character transport limit.")

    expect(mocks.bgRequest).not.toHaveBeenCalled()
  })

  it("measures ragSimple limits after URL encoding and makes no request", async () => {
    const encodedOverLimitQuery = "😀".repeat(665)

    await expect(
      new TldwApiClient().ragSimple(encodedOverLimitQuery, { top_k: 5 })
    ).rejects.toThrow("RAG simple request URL exceeds the 8,000-character transport limit.")

    expect(encodedOverLimitQuery.length).toBeLessThan(RAG_SIMPLE_MAX_PATH_LENGTH)
    expect(mocks.bgRequest).not.toHaveBeenCalled()
  })

  it("strips credentials and prototype keys from ragSimple without dropping its options", async () => {
    mocks.bgRequest.mockResolvedValue({ answer: null })

    const client = new TldwApiClient()
    await client.ragSimple("simple boundary", {
      top_k: 3,
      sources: ["media_db", "notes"],
      mode: "unsupported",
      openai_api_key: BROWSER_PROVIDER_SECRET,
      aws_session_token: BROWSER_PROVIDER_SECRET,
      ...nestedPrototypePollutionPayload,
    })

    const request = mocks.bgRequest.mock.calls[0][0] as Record<string, unknown>
    expect(request.method).toBe("GET")
    expect(request.path).toBe(
      "/api/v1/rag/simple?query=simple+boundary&top_k=3&sources=media_db&sources=notes"
    )
    expect(request).not.toHaveProperty("body")
    expect(String(request.path)).not.toContain(BROWSER_PROVIDER_SECRET)
    expect(String(request.path)).not.toContain(PROTOTYPE_POLLUTION_MARKER)
    expect(String(request.path)).not.toContain("mode")
  })

  it("preserves query normalization at the direct ragSimple mixin boundary", async () => {
    mocks.bgRequest.mockResolvedValue({ answer: null })
    const normalizeRagQuery = vi.fn().mockReturnValue("normalized query")

    await chatRagMethods.ragSimple.call(
      { normalizeRagQuery } as any,
      "raw query",
      { top_k: 2 }
    )

    expect(normalizeRagQuery).toHaveBeenCalledWith("raw query")
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/rag/simple?query=normalized+query&top_k=2",
      method: "GET",
      timeoutMs: undefined,
    })
  })

  it("sanitizes direct TldwApiClientBase ragSearch requests", async () => {
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { results: [], answer: null },
    })

    const client = new TldwApiClientBase()
    await client.ragSearch("base search boundary", credentialShapedOptions)

    const body = mocks.tldwRequest.mock.calls[0][0]?.body as Record<
      string,
      unknown
    >
    expectServerOwnedProviderBody(body)
  })

  it("positive-allowlists Pydantic chat_history fields at the Base boundary", async () => {
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { results: [], answer: null },
    })

    await new TldwApiClientBase().ragSearch(
      "base history boundary",
      credentialShapedOptions
    )

    expectServerOwnedProviderBody(
      mocks.tldwRequest.mock.calls[0][0]?.body as Record<string, unknown>
    )
  })

  it("sanitizes direct TldwApiClientBase ragSearchStream requests", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      yield JSON.stringify({ type: "delta", text: "ok" })
    })

    const client = new TldwApiClientBase()
    const iterator = client.ragSearchStream(
      "base stream boundary",
      credentialShapedOptions
    )
    await iterator.next()

    const payload = mocks.bgStream.mock.calls[0][0] as Record<string, unknown>
    expectServerOwnedProviderBody(payload.body as Record<string, unknown>)
  })

  it("sanitizes direct TldwApiClientBase ragSimple requests", async () => {
    mocks.bgRequest.mockResolvedValue({ answer: null })

    const client = new TldwApiClientBase()
    await client.ragSimple("base simple boundary", {
      top_k: 2,
      sources: ["characters"],
      mode: "unsupported",
      openai_api_key: BROWSER_PROVIDER_SECRET,
      ...nestedPrototypePollutionPayload,
    })

    const request = mocks.bgRequest.mock.calls[0][0] as Record<string, unknown>
    expect(request.method).toBe("GET")
    expect(request.path).toBe(
      "/api/v1/rag/simple?query=base+simple+boundary&top_k=2&sources=characters"
    )
    expect(request).not.toHaveProperty("body")
    expect(String(request.path)).not.toContain(BROWSER_PROVIDER_SECRET)
    expect(String(request.path)).not.toContain(PROTOTYPE_POLLUTION_MARKER)
    expect(String(request.path)).not.toContain("mode")
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

  it("sanitizes direct Base ragSearch failures without replaying HTTP 500", async () => {
    mocks.tldwRequest.mockResolvedValue({
      ok: false,
      status: 500,
      error: `raw ${BROWSER_PROVIDER_SECRET} /Users/private/provider.log`,
      data: null,
    })

    await expect(
      new TldwApiClientBase().ragSearch("base failure", {
        enable_reranking: true,
      })
    ).rejects.toMatchObject({
      message: "RAG search failed due to a server error.",
      status: 500,
    })

    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
  })

  it("uses canonical stream parsing and sanitization at the Base boundary", async () => {
    mocks.bgStream.mockImplementation(async function* () {
      yield "not-json"
    })

    const next = new TldwApiClientBase()
      .ragSearchStream("base stream failure")
      .next()

    await expect(next).rejects.toMatchObject({
      name: "RagTerminalStreamError",
      message: "Invalid RAG stream event.",
    })
    expect(mocks.bgStream).toHaveBeenCalledWith(
      expect.objectContaining({
        sanitizeRagProviderStreamError: true,
      })
    )
  })
})
