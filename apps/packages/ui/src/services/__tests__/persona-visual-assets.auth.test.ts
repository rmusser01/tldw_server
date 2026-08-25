import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  storage: new Map<string, unknown>(),
  createObjectURL: vi.fn(),
  revokeObjectURL: vi.fn()
}))

vi.mock("wxt/browser", () => ({
  browser: { runtime: {} }
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: async (key: string) => mocks.storage.get(key),
    set: async (key: string, value: unknown) => {
      mocks.storage.set(key, value)
    },
    remove: async (key: string) => {
      mocks.storage.delete(key)
    }
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

const pngBytes = new Uint8Array([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00
])

const sha256 = async (bytes: Uint8Array): Promise<string> => {
  const digest = await crypto.subtle.digest("SHA-256", bytes as BufferSource)
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("")
}

describe("persona visual asset authentication", () => {
  beforeEach(() => {
    vi.resetModules()
    mocks.storage.clear()
    mocks.createObjectURL.mockReset()
    mocks.revokeObjectURL.mockReset()
    mocks.createObjectURL.mockReturnValue("blob:authenticated-asset")
    class AssetUrl extends URL {
      static createObjectURL = mocks.createObjectURL
      static revokeObjectURL = mocks.revokeObjectURL
    }
    vi.stubGlobal("URL", AssetUrl)
  })

  afterEach(async () => {
    const { clearPersonaVisualAssetCache } = await import("../persona-visual-assets")
    clearPersonaVisualAssetCache()
    vi.unstubAllGlobals()
  })

  it.each([
    {
      name: "API key",
      config: {
        serverUrl: "https://api.example.test",
        authMode: "single-user" as const,
        apiKey: "test-api-key"
      },
      header: ["X-API-KEY", "test-api-key"] as const
    },
    {
      name: "bearer token",
      config: {
        serverUrl: "https://api.example.test",
        authMode: "multi-user" as const,
        accessToken: "test-access-token"
      },
      header: ["Authorization", "Bearer test-access-token"] as const
    }
  ])("uses real $name client transport headers for protected asset bytes", async ({ config, header }) => {
    const fetchSpy = vi.fn().mockResolvedValue(
      new Response(pngBytes, { status: 200, headers: { "Content-Type": "image/png" } })
    )
    vi.stubGlobal("fetch", fetchSpy)
    const [{ tldwClient }, { acquirePersonaVisualAsset }] = await Promise.all([
      import("@/services/tldw/TldwApiClient"),
      import("../persona-visual-assets")
    ])
    await tldwClient.updateConfig(config)
    const asset = {
      id: "asset-auth",
      url: "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/assets/asset-auth/content",
      checksum_sha256: await sha256(pngBytes),
      byte_size: pngBytes.byteLength,
      mime_type: "image/png"
    }

    const handle = await acquirePersonaVisualAsset(asset)
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit]

    expect(new Headers(init.headers).get(header[0])).toBe(header[1])
    handle.release()
  })
})
