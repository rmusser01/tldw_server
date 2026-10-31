import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  tldwRequest: vi.fn(),
  storedConfig: null as
    | {
        serverUrl: string
        authMode: "single-user" | "multi-user"
        apiKey?: string
        accessToken?: string
      }
    | null
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: vi.fn()
}))

vi.mock("@/services/tldw/request-core", () => ({
  tldwRequest: (...args: unknown[]) => mocks.tldwRequest(...args)
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async (key?: string) => {
      if (key === "tldwConfig") {
        return mocks.storedConfig
      }
      return null
    }),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"

const createConfiguredClient = (): TldwApiClient => {
  mocks.storedConfig = {
    serverUrl: "http://127.0.0.1:8000",
    authMode: "single-user",
    apiKey: "test-api-key"
  }
  return new TldwApiClient()
}

describe("TldwApiClient listAllCharacters", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("fetches all paginated character pages", async () => {
    const allCharacters = Array.from({ length: 2305 }, (_, index) => ({
      id: index + 1,
      name: `Character ${index + 1}`
    }))

    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.method !== "GET" || typeof request.path !== "string") return []
      if (!request.path.startsWith("/api/v1/characters")) return []

      const [, rawQuery = ""] = request.path.split("?")
      const query = new URLSearchParams(rawQuery)
      const limit = Number(query.get("limit") || "100")
      const offset = Number(query.get("offset") || "0")
      return allCharacters.slice(offset, offset + limit)
    })

    const client = new TldwApiClient()
    const list = await client.listAllCharacters()

    expect(list).toHaveLength(2305)
    expect(list[0]).toMatchObject({ id: 1 })
    expect(list[list.length - 1]).toMatchObject({ id: 2305 })

    const listCalls = mocks.bgRequest.mock.calls
      .map(([request]) => request as { path?: string; method?: string })
      .filter(
        (request) =>
          request.method === "GET" &&
          typeof request.path === "string" &&
          request.path.startsWith("/api/v1/characters")
      )
      .map((request) => request.path)

    expect(listCalls).toEqual([
      "/api/v1/characters/?limit=1000&offset=0",
      "/api/v1/characters/?limit=1000&offset=1000",
      "/api/v1/characters/?limit=1000&offset=2000"
    ])
  })

  it("stops when a page contains no new character identities", async () => {
    const repeatedPage = [
      { id: "a", name: "Alpha" },
      { id: "b", name: "Bravo" },
      { id: "c", name: "Charlie" }
    ]

    mocks.bgRequest.mockResolvedValue(repeatedPage)

    const client = new TldwApiClient()
    const list = await client.listAllCharacters({ pageSize: 3, maxPages: 10 })

    expect(list).toEqual(repeatedPage)

    const listCalls = mocks.bgRequest.mock.calls
      .map(([request]) => request as { path?: string; method?: string })
      .filter(
        (request) =>
          request.method === "GET" &&
          typeof request.path === "string" &&
          request.path.startsWith("/api/v1/characters")
      )
      .map((request) => request.path)

    expect(listCalls).toEqual([
      "/api/v1/characters/?limit=3&offset=0",
      "/api/v1/characters/?limit=3&offset=3"
    ])
  })

  it("normalizes object envelopes returned by character list endpoints", async () => {
    const payloadByOffset = new Map<number, unknown>([
      [
        0,
        {
          items: [
            { id: "a", name: "Alpha" },
            { id: "b", name: "Bravo" }
          ]
        }
      ],
      [
        2,
        {
          characters: [{ id: "c", name: "Charlie" }]
        }
      ]
    ])

    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.method !== "GET" || typeof request.path !== "string") return []
      if (!request.path.startsWith("/api/v1/characters")) return []

      const [, rawQuery = ""] = request.path.split("?")
      const query = new URLSearchParams(rawQuery)
      const offset = Number(query.get("offset") || "0")
      return payloadByOffset.get(offset) ?? { items: [] }
    })

    const client = new TldwApiClient()
    const list = await client.listAllCharacters({ pageSize: 2, maxPages: 10 })

    expect(list).toEqual([
      { id: "a", name: "Alpha" },
      { id: "b", name: "Bravo" },
      { id: "c", name: "Charlie" }
    ])

    const listCalls = mocks.bgRequest.mock.calls
      .map(([request]) => request as { path?: string; method?: string })
      .filter(
        (request) =>
          request.method === "GET" &&
          typeof request.path === "string" &&
          request.path.startsWith("/api/v1/characters")
      )
      .map((request) => request.path)

    expect(listCalls).toEqual([
      "/api/v1/characters/?limit=2&offset=0",
      "/api/v1/characters/?limit=2&offset=2"
    ])
  })
})

describe("TldwApiClient listCharactersPage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("returns normalized paged response", async () => {
    mocks.bgRequest.mockResolvedValue({
      items: [{ id: 1, name: "Alpha" }],
      total: 42,
      page: 2,
      page_size: 25,
      has_more: true
    })

    const client = new TldwApiClient()
    const response = await client.listCharactersPage({
      page: 2,
      page_size: 25,
      query: "alp",
      sort_by: "name",
      sort_order: "asc",
      include_image_base64: false
    })

    expect(response).toEqual({
      items: [{ id: 1, name: "Alpha" }],
      total: 42,
      page: 2,
      page_size: 25,
      has_more: true
    })

    const request = mocks.bgRequest.mock.calls
      .map(([call]) => call as { path?: string; method?: string })
      .find(
        (call) =>
          call.method === "GET" &&
          typeof call.path === "string" &&
          call.path.includes("/api/v1/characters/query?")
      )
    expect(request).toBeDefined()
    if (!request) return
    expect(request.method).toBe("GET")
    expect(request.path).toContain("/api/v1/characters/query?")
    expect(request.path).toContain("page=2")
    expect(request.path).toContain("page_size=25")
    expect(request.path).toContain("query=alp")
    expect(request.path).toContain("include_image_base64=false")
  })

  it("supports legacy array responses as a fallback", async () => {
    mocks.bgRequest.mockResolvedValue([{ id: 10, name: "Legacy" }])

    const client = new TldwApiClient()
    const response = await client.listCharactersPage({ page: 1, page_size: 10 })

    expect(response.items).toEqual([{ id: 10, name: "Legacy" }])
    expect(response.total).toBe(1)
    expect(response.page).toBe(1)
    expect(response.page_size).toBe(10)
    expect(response.has_more).toBe(false)
  })

  it("falls back to legacy /characters listing when /characters/query returns route-conflict 422", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.method !== "GET" || typeof request.path !== "string") {
        return {}
      }

      if (request.path.includes("/api/v1/characters/query?")) {
        const error = new Error(
          "Input should be a valid integer, unable to parse string as an integer (path.character_id)"
        ) as Error & { status?: number }
        error.status = 422
        throw error
      }

      if (request.path.startsWith("/api/v1/characters/?")) {
        return [{ id: 22, name: "Legacy fallback item" }]
      }

      return {}
    })

    const client = new TldwApiClient()
    const response = await client.listCharactersPage({
      page: 1,
      page_size: 10,
      sort_by: "name",
      sort_order: "asc",
      include_image_base64: false
    })

    expect(response).toEqual({
      items: [{ id: 22, name: "Legacy fallback item" }],
      total: 1,
      page: 1,
      page_size: 10,
      has_more: false
    })

    const paths = mocks.bgRequest.mock.calls
      .map(([call]) => (call as { path?: string })?.path)
      .filter((path): path is string => typeof path === "string")
    expect(paths.some((path) => path.includes("/api/v1/characters/query?"))).toBe(true)
    expect(paths.some((path) => path.startsWith("/api/v1/characters/?"))).toBe(true)
  })

  it("falls back when route-conflict details are only provided on error.details", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.method !== "GET" || typeof request.path !== "string") {
        return {}
      }

      if (request.path.includes("/api/v1/characters/query?")) {
        const error = Object.assign(new Error("Request failed"), {
          details: {
            detail:
              "Input should be a valid integer, unable to parse string as an integer (path.character_id)"
          }
        })
        throw error
      }

      if (request.path.startsWith("/api/v1/characters/?")) {
        return [{ id: 99, name: "Legacy from details fallback" }]
      }

      return {}
    })

    const client = new TldwApiClient()
    const response = await client.listCharactersPage({
      page: 1,
      page_size: 10
    })

    expect(response.items).toEqual([
      expect.objectContaining({ id: 99, name: "Legacy from details fallback" })
    ])
    expect(response.page).toBe(1)
    expect(response.page_size).toBe(10)
  })
})

describe("TldwApiClient createCharacter", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.tldwRequest.mockReset()
    mocks.storedConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key"
    }
  })

  it("uses trailing-slash create path when OpenAPI path discovery is unavailable", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.method === "POST" && request.path === "/api/v1/characters/") {
        return { id: 1, name: "Created" }
      }
      return {}
    })

    const client = createConfiguredClient()
    const created = await client.createCharacter({
      name: "Created",
      system_prompt: "Prompt"
    })

    expect(created).toEqual({ id: 1, name: "Created" })

    const postPaths = mocks.bgRequest.mock.calls
      .map(([request]) => request as { path?: string; method?: string })
      .filter(
        (request) => request.method === "POST" && typeof request.path === "string"
      )
      .map((request) => request.path)
    expect(postPaths).toEqual(["/api/v1/characters/"])
  })

  it("retries create with alternate trailing-slash path when initial create path redirects", async () => {
    mocks.tldwRequest.mockResolvedValueOnce({
      ok: true,
      status: 200,
      data: { paths: { "/api/v1/characters": { post: {} } } }
    })
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.method === "POST" && request.path === "/api/v1/characters") {
        const redirectError = new Error("Request failed: 307") as Error & {
          status?: number
        }
        redirectError.status = 307
        throw redirectError
      }

      if (request.method === "POST" && request.path === "/api/v1/characters/") {
        return { id: 2, name: "Recovered create" }
      }

      return {}
    })

    const client = createConfiguredClient()
    const created = await client.createCharacter({
      name: "Recovered create",
      system_prompt: "Prompt"
    })

    expect(created).toEqual({ id: 2, name: "Recovered create" })

    const postPaths = mocks.bgRequest.mock.calls
      .map(([request]) => request as { path?: string; method?: string })
      .filter(
        (request) => request.method === "POST" && typeof request.path === "string"
      )
      .map((request) => request.path)
    expect(postPaths).toEqual([
      "/api/v1/characters",
      "/api/v1/characters/"
    ])
  })

  it("retries once on extension messaging timeout before surfacing create errors", async () => {
    let createAttempts = 0
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.method === "POST" && request.path === "/api/v1/characters/") {
        createAttempts += 1
        if (createAttempts === 1) {
          const timeoutError = new Error("Extension messaging timeout") as Error & {
            __tldwExtensionTimeout?: boolean
          }
          timeoutError.__tldwExtensionTimeout = true
          throw timeoutError
        }
        return { id: 3, name: "Retry success" }
      }
      return {}
    })

    const client = createConfiguredClient()
    const created = await client.createCharacter({
      name: "Retry success",
      system_prompt: "Prompt"
    })

    expect(created).toEqual({ id: 3, name: "Retry success" })
    expect(createAttempts).toBe(2)
  })

  it("falls back to direct request after repeated extension messaging timeouts", async () => {
    let createAttempts = 0
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.method === "POST" && request.path === "/api/v1/characters/") {
        createAttempts += 1
        const timeoutError = new Error("Extension messaging timeout") as Error & {
          __tldwExtensionTimeout?: boolean
        }
        timeoutError.__tldwExtensionTimeout = true
        throw timeoutError
      }
      return {}
    })
    mocks.tldwRequest
      .mockResolvedValueOnce({ ok: true, status: 200, data: { paths: {} } })
      .mockResolvedValueOnce({
        ok: true,
        status: 201,
        data: { id: 4, name: "Direct fallback" }
      })

    const client = createConfiguredClient()
    const created = await client.createCharacter({
      name: "Direct fallback",
      system_prompt: "Prompt"
    })

    expect(created).toEqual({ id: 4, name: "Direct fallback" })
    expect(createAttempts).toBe(2)
    const directCreates = mocks.tldwRequest.mock.calls.filter(([request]) => request.method === "POST")
    expect(directCreates).toHaveLength(1)
    const directRequestInit = directCreates[0]?.[0] as
      | { path?: string; method?: string }
      | undefined
    expect(directRequestInit?.path).toBe("/api/v1/characters/")
    expect(directRequestInit?.method).toBe("POST")
  })
})
