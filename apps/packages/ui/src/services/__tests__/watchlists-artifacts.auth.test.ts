import { beforeEach, describe, expect, it, vi } from "vitest"

const runtime = vi.hoisted(() => ({
  config: {
    serverUrl: "https://api.example.test",
    authMode: "single-user",
    apiKey: "artifact-api-key",
    accessToken: ""
  } as Record<string, unknown>
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      id: "artifact-test-extension",
      sendMessage: vi.fn().mockRejectedValue(
        new Error("Could not establish connection. Receiving end does not exist.")
      )
    }
  }
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: async (key: string) => key === "tldwConfig" ? runtime.config : null,
    set: async () => undefined
  })
}))

import {
  WatchlistArtifactError,
  createWatchlistArtifactObjectUrl,
  fetchWatchlistArtifactBlob,
  fetchWatchlistArtifactText,
  revokeWatchlistArtifactObjectUrl
} from "../watchlists-artifacts"

describe("authenticated Watchlists artifacts", () => {
  beforeEach(() => {
    runtime.config = {
      serverUrl: "https://api.example.test",
      authMode: "single-user",
      apiKey: "artifact-api-key",
      accessToken: ""
    }
    vi.restoreAllMocks()
  })

  it("loads an audio Blob from the configured API origin with API-key auth in an extension", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(new Uint8Array([1, 2, 3]), {
        status: 200,
        headers: { "Content-Type": "audio/mpeg" }
      })
    )
    vi.stubGlobal("fetch", fetchMock)

    const blob = await fetchWatchlistArtifactBlob(
      "/api/v1/watchlists/runs/9/audio/final/download",
      { mimeType: "audio/mpeg" }
    )

    expect(blob).toBeInstanceOf(Blob)
    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.example.test/api/v1/watchlists/runs/9/audio/final/download",
      expect.objectContaining({
        headers: expect.objectContaining({ "X-API-KEY": "artifact-api-key" })
      })
    )
  })

  it("loads script text with Bearer auth and never resolves against the extension origin", async () => {
    runtime.config = {
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      accessToken: "artifact-access-token"
    }
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("# Private briefing script", { status: 200 })
    )
    vi.stubGlobal("fetch", fetchMock)

    await expect(
      fetchWatchlistArtifactText("/api/v1/watchlists/runs/9/audio/script/download")
    ).resolves.toBe("# Private briefing script")

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("https://api.example.test/api/v1/watchlists/runs/9/audio/script/download")
    expect(url).not.toContain("chrome-extension://")
    expect(new Headers(init.headers).get("Authorization")).toBe("Bearer artifact-access-token")
  })

  it("classifies missing artifacts separately from auth and network failures", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response("missing", { status: 404 }))
      .mockResolvedValueOnce(new Response("forbidden", { status: 403 }))
      .mockRejectedValueOnce(new TypeError("offline"))
    vi.stubGlobal("fetch", fetchMock)

    await expect(fetchWatchlistArtifactText("/api/v1/watchlists/runs/9/audio/script/download"))
      .rejects.toMatchObject<Partial<WatchlistArtifactError>>({ kind: "missing", status: 404 })
    await expect(fetchWatchlistArtifactText("/api/v1/watchlists/runs/9/audio/script/download"))
      .rejects.toMatchObject<Partial<WatchlistArtifactError>>({ kind: "auth", status: 403 })
    await expect(fetchWatchlistArtifactText("/api/v1/watchlists/runs/9/audio/script/download"))
      .rejects.toMatchObject<Partial<WatchlistArtifactError>>({ kind: "network" })
  })

  it("creates and revokes object URLs through one shared lifecycle", () => {
    const create = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:watchlist-artifact")
    const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined)
    const blob = new Blob(["audio"], { type: "audio/mpeg" })

    const url = createWatchlistArtifactObjectUrl(blob)
    revokeWatchlistArtifactObjectUrl(url)

    expect(create).toHaveBeenCalledWith(blob)
    expect(revoke).toHaveBeenCalledWith("blob:watchlist-artifact")
  })

  it("rejects non-API and cross-origin artifact locations", async () => {
    await expect(fetchWatchlistArtifactText("chrome-extension://id/private.mp3"))
      .rejects.toMatchObject({ kind: "unsafe" })
    await expect(fetchWatchlistArtifactText("https://attacker.example/private.mp3"))
      .rejects.toMatchObject({ kind: "unsafe" })
  })
})
