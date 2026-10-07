import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: vi.fn(),
  bgStream: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"

describe("TldwApiClient notes methods", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.bgRequest.mockResolvedValue([])
  })

  it("maps page/results_per_page to limit/offset for listNotes", async () => {
    const client = new TldwApiClient()
    await client.listNotes({ page: 2, results_per_page: 200, include_keywords: false })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/notes/?limit=200&offset=200&include_keywords=false",
        method: "GET"
      })
    )
  })

  it("uses GET query params for searchNotes", async () => {
    const client = new TldwApiClient()
    await client.searchNotes("cell biology")

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/notes/search/?query=cell+biology",
        method: "GET"
      })
    )
  })

  it("falls back to listNotes when search query is empty", async () => {
    const client = new TldwApiClient()
    await client.searchNotes("   ")

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/notes/",
        method: "GET"
      })
    )
  })

  it("returns search results from the requested Notes page and carries cancellation", async () => {
    const client = new TldwApiClient()
    const controller = new AbortController()
    mocks.bgRequest.mockImplementation(
      async (request: { path: string; abortSignal?: AbortSignal }) =>
        request.path ===
          "/api/v1/notes/search/?query=biology&limit=50&offset=200" &&
        request.abortSignal === controller.signal
          ? { notes: [{ id: "note-207", title: "Page five note" }], total: 250 }
          : { notes: [] },
    )
    expect(
      await client.searchNotes(
        "biology",
        { limit: 50, offset: 200 },
        { signal: controller.signal },
      ),
    ).toEqual({
      notes: [{ id: "note-207", title: "Page five note" }],
      total: 250,
    })
  })

  it("loads note folders from the public notes folder endpoint", async () => {
    const client = new TldwApiClient()
    await client.listNoteFolders()

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/notes/folders/",
        method: "GET"
      })
    )
  })

  it("creates note folders by normalized path", async () => {
    const client = new TldwApiClient()
    await client.createNoteFolder("Inbox/Captured Articles")

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/notes/folders/",
        method: "POST",
        body: { path: "Inbox/Captured Articles" }
      })
    )
  })
})

describe("Notes language negotiation", () => {
  beforeEach(() => vi.resetAllMocks())
  it("requires an explicit advertised language capability", async () => {
    mocks.bgRequest.mockResolvedValueOnce({})
    const client = new TldwApiClient()
    expect(await client.getNotesLanguageCapability()).toEqual({ available: false, reason: "unsupported_server" })
    expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/notes/capabilities", method: "GET" }))
    mocks.bgRequest.mockResolvedValueOnce({ id: "n1", content: "{}" })
    await client.createNote("{}", { title: "Config" })
    expect(mocks.bgRequest).toHaveBeenLastCalledWith(expect.objectContaining({ body: { content: "{}", title: "Config" } }))
  })
  it("accepts explicit support and rechecks after reconnect", async () => {
    mocks.bgRequest.mockResolvedValueOnce({ content_language: true }).mockResolvedValueOnce({ content_language: false })
    const client = new TldwApiClient()
    expect(await client.getNotesLanguageCapability()).toEqual({ available: true, reason: null })
    expect(await client.getNotesLanguageCapability()).toEqual({ available: false, reason: "unsupported_server" })
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })
  it.each([[404, "unsupported_server"], [401, "authentication_required"], [503, "transport_unavailable"]])("bounds HTTP %s failures", async (status, reason) => {
    mocks.bgRequest.mockRejectedValueOnce({ status })
    expect(await new TldwApiClient().getNotesLanguageCapability()).toEqual({ available: false, reason })
  })
  it("keeps transport failures unavailable", async () => {
    mocks.bgRequest.mockRejectedValueOnce(new Error("private error"))
    expect(await new TldwApiClient().getNotesLanguageCapability()).toEqual({ available: false, reason: "transport_unavailable" })
  })
  it("pins capability and publication to the supplied profile", async () => {
    const requestScope = { config: { serverUrl: "https://a.example", authMode: "multi-user" as const, accessToken: "token-a" }, userId: "user-a" }
    mocks.bgRequest.mockResolvedValueOnce({ content_language: true }).mockResolvedValueOnce({ id: "n1", content_language: "Future/Mode" })
    await new TldwApiClient().createNote("{}", { title: "Config", content_language: "Future/Mode" }, { requestScope })
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    expect(mocks.bgRequest.mock.calls[0][0].servicePromptConfig).toEqual({ ...requestScope.config, expectedUserId: "user-a" })
    expect(mocks.bgRequest.mock.calls[1][0].servicePromptConfig).toEqual({ ...requestScope.config, expectedUserId: "user-a" })
    expect(mocks.bgRequest.mock.calls[1][0].body.content_language).toBe("Future/Mode")
  })
  it("does not publish language when the capability disappears", async () => {
    mocks.bgRequest.mockResolvedValue({})
    await expect(new TldwApiClient().createNote("{}", { title: "Config", content_language: null })).rejects.toMatchObject({ code: "unsupported_server" })
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })
})

it("rejects an in-flight profile switch before publishing metadata", async () => {
  vi.resetAllMocks()
  mocks.bgRequest.mockImplementationOnce(async () => {
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
    return { content_language: true }
  })
  await expect(new TldwApiClient().createNote("{}", { title: "Config", content_language: "json" })).rejects.toMatchObject({ code: "scope_changed" })
  expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
})

it("does not acknowledge a save after its account changes", async () => {
  vi.resetAllMocks()
  mocks.bgRequest.mockResolvedValueOnce({ content_language: true }).mockImplementationOnce(async () => {
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
    return { id: "n1", content_language: "json" }
  })
  await expect(new TldwApiClient().createNote("{}", { title: "Config", content_language: "json" })).rejects.toMatchObject({ code: "scope_changed" })
})

it("captures the ambient profile once for probe and publication", async () => {
  vi.resetAllMocks()
  const client = new TldwApiClient()
  vi.spyOn(client, "getConfig").mockResolvedValue({ serverUrl: "https://a.example", authMode: "multi-user", refreshToken: "refresh-a" })
  mocks.bgRequest.mockResolvedValueOnce({ content_language: true }).mockResolvedValueOnce({ id: "n1", content_language: "json" })
  await client.createNote("{}", { title: "Config", content_language: "json" })
  expect(mocks.bgRequest.mock.calls[0][0].servicePromptConfig).toMatchObject({ serverUrl: "https://a.example", expectedRefreshToken: "refresh-a" })
  expect(mocks.bgRequest.mock.calls[1][0].servicePromptConfig).toEqual(mocks.bgRequest.mock.calls[0][0].servicePromptConfig)
})
