import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { PersonaVisualAsset } from "@/types/persona-visuals"

const mocks = vi.hoisted(() => ({
  fetchWithAuth: vi.fn(),
  createObjectURL: vi.fn(),
  revokeObjectURL: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    fetchWithAuth: (...args: unknown[]) => mocks.fetchWithAuth(...args)
  }
}))

import {
  acquirePersonaVisualAsset,
  clearPersonaVisualAssetCache,
  PersonaVisualAssetError
} from "../persona-visual-assets"

const pngBytes = new Uint8Array([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00
])

const sha256 = async (bytes: Uint8Array): Promise<string> => {
  const digest = await crypto.subtle.digest("SHA-256", bytes)
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("")
}

const assetFor = async (
  bytes = pngBytes
): Promise<Pick<PersonaVisualAsset, "id" | "url" | "checksum_sha256" | "byte_size" | "mime_type">> => ({
  id: "asset-1",
  url: "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/assets/asset-1/content",
  checksum_sha256: await sha256(bytes),
  byte_size: bytes.byteLength,
  mime_type: "image/png"
})

const successfulByteResponse = (bytes: Uint8Array) => ({
  ok: true,
  status: 200,
  data: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  json: async () => null
})

describe("persona visual asset loader", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.createObjectURL.mockReturnValue("blob:persona-asset")
    vi.stubGlobal("URL", {
      createObjectURL: mocks.createObjectURL,
      revokeObjectURL: mocks.revokeObjectURL
    })
  })

  afterEach(() => {
    clearPersonaVisualAssetCache()
    vi.unstubAllGlobals()
  })

  it("loads protected bytes through the authenticated client", async () => {
    const asset = await assetFor()
    mocks.fetchWithAuth.mockResolvedValueOnce(successfulByteResponse(pngBytes))

    const handle = await acquirePersonaVisualAsset(asset, { maxBytes: 1024 })

    expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
      asset.url,
      expect.objectContaining({ responseType: "arrayBuffer" })
    )
    expect(handle.url).toBe("blob:persona-asset")
    expect(handle.mimeType).toBe("image/png")

    handle.release()
    expect(mocks.revokeObjectURL).toHaveBeenCalledTimes(1)
  })

  it("shares an immutable asset acquisition until every handle is released", async () => {
    const asset = await assetFor()
    mocks.fetchWithAuth.mockResolvedValueOnce(successfulByteResponse(pngBytes))

    const [first, second] = await Promise.all([
      acquirePersonaVisualAsset(asset),
      acquirePersonaVisualAsset(asset)
    ])

    expect(mocks.fetchWithAuth).toHaveBeenCalledTimes(1)
    expect(first.url).toBe(second.url)
    first.release()
    expect(mocks.revokeObjectURL).not.toHaveBeenCalled()
    second.release()
    second.release()
    expect(mocks.revokeObjectURL).toHaveBeenCalledTimes(1)
  })

  it("rejects declared and received size violations before creating an object URL", async () => {
    const asset = await assetFor()

    await expect(acquirePersonaVisualAsset(asset, { maxBytes: 1 })).rejects.toMatchObject({
      code: "asset_too_large"
    })
    expect(mocks.fetchWithAuth).not.toHaveBeenCalled()

    mocks.fetchWithAuth.mockResolvedValueOnce(successfulByteResponse(pngBytes))
    await expect(
      acquirePersonaVisualAsset({ ...asset, byte_size: pngBytes.byteLength - 1 })
    ).rejects.toMatchObject({ code: "asset_size_mismatch" })
    expect(mocks.createObjectURL).not.toHaveBeenCalled()
  })

  it("rejects checksum and MIME-signature mismatches before creating an object URL", async () => {
    const asset = await assetFor()
    mocks.fetchWithAuth
      .mockResolvedValueOnce(successfulByteResponse(pngBytes))
      .mockResolvedValueOnce(successfulByteResponse(pngBytes))

    await expect(
      acquirePersonaVisualAsset({ ...asset, checksum_sha256: "0".repeat(64) })
    ).rejects.toMatchObject({ code: "asset_checksum_mismatch" })
    await expect(
      acquirePersonaVisualAsset({ ...asset, mime_type: "image/gif" })
    ).rejects.toMatchObject({ code: "asset_mime_mismatch" })
    expect(mocks.createObjectURL).not.toHaveBeenCalled()
  })

  it("does not poison the cache when an authenticated acquisition fails", async () => {
    const asset = await assetFor()
    mocks.fetchWithAuth
      .mockRejectedValueOnce(new Error("unauthorized"))
      .mockResolvedValueOnce(successfulByteResponse(pngBytes))

    await expect(acquirePersonaVisualAsset(asset)).rejects.toThrow("unauthorized")
    const handle = await acquirePersonaVisualAsset(asset)

    expect(mocks.fetchWithAuth).toHaveBeenCalledTimes(2)
    handle.release()
  })

  it("rejects an aborted caller without revoking another retained handle", async () => {
    const asset = await assetFor()
    mocks.fetchWithAuth.mockResolvedValueOnce(successfulByteResponse(pngBytes))
    const controller = new AbortController()
    controller.abort()

    await expect(
      acquirePersonaVisualAsset(asset, { signal: controller.signal })
    ).rejects.toMatchObject({ code: "asset_aborted" })

    const handle = await acquirePersonaVisualAsset(asset)
    expect(mocks.fetchWithAuth).toHaveBeenCalledTimes(1)
    handle.release()
  })

  it("clears cached object URLs once without double-revoking released handles", async () => {
    const asset = await assetFor()
    mocks.fetchWithAuth.mockResolvedValueOnce(successfulByteResponse(pngBytes))
    const handle = await acquirePersonaVisualAsset(asset)

    clearPersonaVisualAssetCache()
    clearPersonaVisualAssetCache()
    handle.release()

    expect(mocks.revokeObjectURL).toHaveBeenCalledTimes(1)
  })

  it("reports malformed authenticated responses as asset boundary failures", async () => {
    const asset = await assetFor()
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      data: "not bytes",
      json: async () => null
    })

    await expect(acquirePersonaVisualAsset(asset)).rejects.toBeInstanceOf(
      PersonaVisualAssetError
    )
    expect(mocks.createObjectURL).not.toHaveBeenCalled()
  })
})
