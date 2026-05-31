import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const ORIGINAL_ENV = { ...process.env }
const ORIGINAL_WINDOW_LOCATION = window.location

function setWindowLocation(href: string) {
  let current = new URL(href)
  Object.defineProperty(window, "location", {
    configurable: true,
    value: {
      get hash() {
        return current.hash
      },
      get href() {
        return current.href
      },
      set href(nextHref: string) {
        current = new URL(nextHref, current.href)
      },
      get origin() {
        return current.origin
      },
      get pathname() {
        return current.pathname
      },
      get search() {
        return current.search
      }
    }
  })
}

function jsonResponse(
  body: unknown,
  init: ResponseInit = {}
): Response {
  const headers = new Headers(init.headers)
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json")
  }
  return new Response(JSON.stringify(body), {
    ...init,
    headers
  })
}

function storedRequestHistory() {
  return JSON.parse(localStorage.getItem("tldw-request-history") || "[]")
}

function storedSessionId() {
  const raw = localStorage.getItem("tldw-session-id")
  return raw ? JSON.parse(raw).id : null
}

function getFetchCall(fetchMock: ReturnType<typeof vi.fn>, index = 0) {
  const [url, init] = fetchMock.mock.calls[index] as [string, RequestInit]
  return {
    url,
    init,
    headers: new Headers(init.headers)
  }
}

async function loadApiModule() {
  vi.resetModules()
  return import("@web/lib/api")
}

describe("fetch-backed WebUI api client", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.clearAllMocks()
    vi.unstubAllGlobals()
    localStorage.clear()
    document.cookie = "csrf_token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/"
    process.env = { ...ORIGINAL_ENV }
    delete process.env.NEXT_PUBLIC_API_URL
    delete process.env.NEXT_PUBLIC_API_VERSION
    delete process.env.NEXT_PUBLIC_API_BEARER
    delete process.env.NEXT_PUBLIC_X_API_KEY
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
    setWindowLocation("http://localhost:3000/chat")
    localStorage.setItem(
      "tldw-session-id",
      JSON.stringify({ id: "sess-existing", timestamp: Date.now() })
    )
  })

  afterEach(() => {
    vi.useRealTimers()
    process.env = { ...ORIGINAL_ENV }
  })

  afterAll(() => {
    Object.defineProperty(window, "location", {
      configurable: true,
      value: ORIGINAL_WINDOW_LOCATION
    })
  })

  it("uses fetch with mutable baseURL, JSON bodies, auth, CSRF, session headers, credentials, and history", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(
        { ok: true },
        {
          status: 200,
          headers: {
            "X-Session-ID": "sess-response"
          }
        }
      )
    )
    vi.stubGlobal("fetch", fetchMock)
    localStorage.setItem("access_token", "jwt-token")
    document.cookie = "csrf_token=csrf-123; path=/"

    const { apiClient, default: api } = await loadApiModule()
    api.defaults.baseURL = "https://api.example.test/api/v9"

    await expect(
      apiClient.post(
        "/items",
        { title: "Research note" },
        { headers: { "X-Custom": "custom-value" } }
      )
    ).resolves.toEqual({ ok: true })

    const { url, init, headers } = getFetchCall(fetchMock)
    expect(url).toBe("https://api.example.test/api/v9/items")
    expect(init).toEqual(
      expect.objectContaining({
        method: "POST",
        credentials: "include",
        body: JSON.stringify({ title: "Research note" })
      })
    )
    expect(headers.get("Content-Type")).toBe("application/json")
    expect(headers.get("Authorization")).toBe("Bearer jwt-token")
    expect(headers.get("X-CSRF-Token")).toBe("csrf-123")
    expect(headers.get("X-Session-ID")).toBe("sess-existing")
    expect(headers.get("X-Custom")).toBe("custom-value")
    expect(storedSessionId()).toBe("sess-response")
    expect(storedRequestHistory()[0]).toEqual(
      expect.objectContaining({
        method: "POST",
        url: "/items",
        baseURL: "https://api.example.test/api/v9",
        status: 200,
        ok: true,
        responseBody: { ok: true }
      })
    )
  })

  it("merges default headers before applying request-specific overrides", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ dismissed: true }))
    vi.stubGlobal("fetch", fetchMock)

    const { apiClient, default: api } = await loadApiModule()
    api.defaults.headers["X-Default"] = "default-value"
    api.defaults.headers["X-Override"] = "default-value"

    await expect(
      apiClient.post("/notifications/1/dismiss", undefined, {
        headers: {
          "X-Override": "request-value"
        }
      })
    ).resolves.toEqual({ dismissed: true })

    const { init, headers } = getFetchCall(fetchMock)
    expect(init.body).toBeUndefined()
    expect(headers.get("Content-Type")).toBe("application/json")
    expect(headers.get("X-Default")).toBe("default-value")
    expect(headers.get("X-Override")).toBe("request-value")
  })

  it("omits forced content type and cookie credentials for FormData with X-API-KEY auth", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ uploaded: true }))
    vi.stubGlobal("fetch", fetchMock)
    localStorage.setItem("apiKey", "single-user-key")
    document.cookie = "csrf_token=csrf-ignored; path=/"

    const { apiClient, default: api } = await loadApiModule()
    api.defaults.baseURL = "/api/v1"
    const formData = new FormData()
    formData.set("file", new Blob(["hello"]), "hello.txt")

    await expect(apiClient.post("/media/add", formData)).resolves.toEqual({
      uploaded: true
    })

    const { url, init, headers } = getFetchCall(fetchMock)
    expect(url).toBe("/api/v1/media/add")
    expect(init.body).toBe(formData)
    expect(init.credentials).toBe("omit")
    expect(headers.get("Content-Type")).toBeNull()
    expect(headers.get("X-API-KEY")).toBe("single-user-key")
    expect(headers.get("X-CSRF-Token")).toBeNull()
  })

  it("preserves caller-provided authorization and api key headers", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true }))
    vi.stubGlobal("fetch", fetchMock)
    localStorage.setItem("access_token", "jwt-token")
    localStorage.setItem("apiKey", "stored-api-key")

    const { apiClient } = await loadApiModule()

    await expect(
      apiClient.get("/delegated", {
        headers: {
          Authorization: "Bearer caller-token",
          "X-API-KEY": "caller-api-key"
        }
      })
    ).resolves.toEqual({ ok: true })

    const { headers } = getFetchCall(fetchMock)
    expect(headers.get("Authorization")).toBe("Bearer caller-token")
    expect(headers.get("X-API-KEY")).toBe("caller-api-key")
  })

  it("preserves protocol-relative URLs when appending query params", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true }))
    vi.stubGlobal("fetch", fetchMock)

    const { apiClient } = await loadApiModule()

    await expect(
      apiClient.get("//cdn.example.test/resource", {
        params: {
          q: "search",
          page: 2
        }
      })
    ).resolves.toEqual({ ok: true })

    const { url } = getFetchCall(fetchMock)
    expect(url).toBe("//cdn.example.test/resource?q=search&page=2")
  })

  it("parses explicit JSON response types even when content type is missing", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "Content-Type": "text/plain"
        }
      })
    )
    vi.stubGlobal("fetch", fetchMock)

    const { apiClient } = await loadApiModule()

    await expect(
      apiClient.get("/mislabelled-json", { responseType: "json" })
    ).resolves.toEqual({ ok: true })
  })

  it("maps API error payloads and retry-after headers to ApiError", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(
        { detail: "Rate limited" },
        {
          status: 429,
          headers: {
            "Retry-After": "17"
          }
        }
      )
    )
    vi.stubGlobal("fetch", fetchMock)

    const { apiClient } = await loadApiModule()

    await expect(apiClient.get("/limited")).rejects.toMatchObject({
      name: "ApiError",
      status: 429,
      statusCode: 429,
      detail: "Rate limited",
      retryAfter: 17,
      message: "Rate limited"
    })
    expect(storedRequestHistory()[0]).toEqual(
      expect.objectContaining({
        method: "GET",
        url: "/limited",
        status: 429,
        ok: false,
        errorMessage: "Rate limited"
      })
    )
  })

  it("keeps CSRF failures normalized to the existing refresh-page message", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ detail: "CSRF token missing" }, { status: 403 })
    )
    vi.stubGlobal("fetch", fetchMock)

    const { apiClient } = await loadApiModule()

    await expect(apiClient.post("/notes", { text: "x" })).rejects.toThrow(
      "CSRF validation failed. Refresh the page and try again."
    )
  })

  it("extracts JSON error details before applying success response types", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ detail: "Download denied" }, { status: 403 })
    )
    vi.stubGlobal("fetch", fetchMock)

    const { apiClient } = await loadApiModule()

    await expect(
      apiClient.get("/exports/archive", { responseType: "arraybuffer" })
    ).rejects.toMatchObject({
      name: "ApiError",
      status: 403,
      detail: "Download denied",
      message: "Download denied"
    })
  })

  it("clears auth state and redirects on unauthorized responses with malformed bodies", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("{not-json", {
        status: 401,
        statusText: "Unauthorized",
        headers: {
          "Content-Type": "application/json"
        }
      })
    )
    vi.stubGlobal("fetch", fetchMock)
    localStorage.setItem("access_token", "jwt-token")
    localStorage.setItem("user", JSON.stringify({ username: "alice" }))
    setWindowLocation("http://localhost:3000/chat")

    const { apiClient } = await loadApiModule()

    await expect(apiClient.get("/protected")).rejects.toMatchObject({
      name: "ApiError",
      status: 401
    })
    expect(localStorage.getItem("access_token")).toBeNull()
    expect(localStorage.getItem("user")).toBeNull()
    expect(window.location.href).toBe("http://localhost:3000/login")
    expect(storedRequestHistory()[0]).toEqual(
      expect.objectContaining({
        method: "GET",
        url: "/protected",
        status: 401,
        ok: false
      })
    )
  })

  it("supports timeout cancellation and caller-provided abort signals", async () => {
    vi.useFakeTimers()
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"))
        })
      })
    })
    vi.stubGlobal("fetch", fetchMock)

    const { apiClient } = await loadApiModule()
    const request = apiClient.get("/slow", { timeout: 5 })
    const requestAssertion = expect(request).rejects.toMatchObject({
      name: "ApiError"
    })

    await vi.advanceTimersByTimeAsync(5)

    await requestAssertion
    expect((fetchMock.mock.calls[0][1] as RequestInit).signal?.aborted).toBe(
      true
    )

    const controller = new AbortController()
    const abortedRequest = apiClient.get("/caller-abort", {
      signal: controller.signal
    })
    controller.abort()

    await expect(abortedRequest).rejects.toMatchObject({
      name: "ApiError"
    })
  })

  it("returns binary and empty responses using the request config response type", async () => {
    const bytes = new Uint8Array([1, 2, 3])
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(bytes, { status: 200 }))
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
    vi.stubGlobal("fetch", fetchMock)

    const { apiClient } = await loadApiModule()

    const binary = await apiClient.get<ArrayBuffer>("/binary", {
      responseType: "arraybuffer"
    })
    expect(Array.from(new Uint8Array(binary))).toEqual([1, 2, 3])

    await expect(apiClient.delete("/empty")).resolves.toBeUndefined()
  })
})
