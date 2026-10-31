import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tldwRequest, type TldwRequestPayload } from "@/services/tldw/request-core"
import { getServerCapabilities } from "@/services/tldw/server-capabilities"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  bgRequest: vi.fn(),
  fetch: vi.fn<typeof fetch>(),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: () => mocks.getConfig(),
    getOpenAPISpec: async () => ({ info: { version: "bootstrap" }, paths: {} }),
  },
  isActiveCookieSessionConfig: () => false,
}))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (request: TldwRequestPayload) => mocks.bgRequest(request),
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: async () => undefined, set: async () => undefined }),
}))

// The retired useConfig provider was removed in 661b76c144. Public docs discovery
// now lives in server-capabilities and uses the shared noAuth request contract.
describe("public docs-info bootstrap fetch mode", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", undefined)
    mocks.getConfig.mockReset()
    mocks.fetch.mockReset().mockResolvedValue(new Response(JSON.stringify({}), {
      status: 200,
      headers: { "content-type": "application/json" },
    }))
    mocks.bgRequest.mockReset().mockImplementation(async (request: TldwRequestPayload) => {
      const response = await tldwRequest(request, {
        getConfig: () => mocks.getConfig(),
        fetchFn: mocks.fetch,
      })
      if (!response.ok) throw new Error(response.error)
      return response.data
    })
  })

  afterEach(() => vi.unstubAllEnvs())

  it.each([
    { authMode: "single-user", apiKey: "private-api-key" },
    { authMode: "multi-user", accessToken: "private-access-token" },
  ])("does not send $authMode credentials to cross-origin docs discovery", async (auth) => {
    mocks.getConfig.mockResolvedValue({ serverUrl: "https://api.example.test", ...auth })

    await getServerCapabilities({ forceRefresh: true })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/config/docs-info",
      method: "GET",
      noAuth: true,
    })
    expect(mocks.fetch).toHaveBeenCalledTimes(1)
    const [url, init] = mocks.fetch.mock.calls[0]
    expect(url).toBe("https://api.example.test/api/v1/config/docs-info")
    // Fetch's default same-origin mode also omits cross-origin cookies.
    expect(init?.credentials ?? "same-origin").not.toBe("include")
    const headers = new Headers(init?.headers)
    expect(headers.has("authorization")).toBe(false)
    expect(headers.has("x-api-key")).toBe(false)
  })
})
