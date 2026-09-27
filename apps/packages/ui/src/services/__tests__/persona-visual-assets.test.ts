import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
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

type AssetInput = {
  id: string
  url: string
  checksum_sha256: string
  byte_size: number
  mime_type: string
}

const sha256 = async (bytes: Uint8Array): Promise<string> => {
  const digest = await crypto.subtle.digest("SHA-256", bytes as BufferSource)
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("")
}

const assetFor = async (
  bytes = pngBytes
): Promise<AssetInput> => ({
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

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  return {
    promise: new Promise<T>((resolvePromise, rejectPromise) => {
      resolve = resolvePromise
      reject = rejectPromise
    }),
    resolve,
    reject
  }
}

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

  it("rejects malformed checksum declarations before fetching protected bytes", async () => {
    const asset = await assetFor()

    for (const checksum_sha256 of ["a".repeat(63), "g".repeat(64), "A".repeat(64)]) {
      await expect(
        acquirePersonaVisualAsset({ ...asset, checksum_sha256 })
      ).rejects.toMatchObject({ code: "asset_checksum_mismatch" })
    }

    expect(mocks.fetchWithAuth).not.toHaveBeenCalled()
    expect(mocks.createObjectURL).not.toHaveBeenCalled()
  })

  it("accepts only complete signatures for every supported raster MIME type", async () => {
    const validFixtures = [
      { mime_type: "image/png", bytes: pngBytes },
      { mime_type: "image/jpeg", bytes: new Uint8Array([0xff, 0xd8, 0xff, 0x00]) },
      { mime_type: "image/gif", bytes: new Uint8Array([0x47, 0x49, 0x46, 0x38, 0x39, 0x61]) },
      {
        mime_type: "image/webp",
        bytes: new Uint8Array([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50])
      }
    ] as const
    const invalidFixtures = [
      { mime_type: "image/png", bytes: new Uint8Array([0x89, 0x50, 0x4e]) },
      { mime_type: "image/jpeg", bytes: new Uint8Array([0xff, 0xd8, 0x00]) },
      { mime_type: "image/gif", bytes: new Uint8Array([0x47, 0x49, 0x46, 0x30, 0x30, 0x61]) },
      {
        mime_type: "image/webp",
        bytes: new Uint8Array([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x00])
      }
    ] as const
    mocks.createObjectURL.mockImplementation((_, index) => `blob:asset-${index}`)

    for (const fixture of validFixtures) {
      const asset = await assetFor(fixture.bytes)
      mocks.fetchWithAuth.mockResolvedValue(successfulByteResponse(fixture.bytes))
      const handle = await acquirePersonaVisualAsset({ ...asset, mime_type: fixture.mime_type })
      handle.release()
    }
    const createdAfterValidBytes = mocks.createObjectURL.mock.calls.length

    for (const fixture of invalidFixtures) {
      const asset = await assetFor(fixture.bytes)
      mocks.fetchWithAuth.mockResolvedValue(successfulByteResponse(fixture.bytes))
      await expect(
        acquirePersonaVisualAsset({ ...asset, mime_type: fixture.mime_type })
      ).rejects.toMatchObject({ code: "asset_mime_mismatch" })
    }

    expect(createdAfterValidBytes).toBe(4)
    expect(mocks.createObjectURL).toHaveBeenCalledTimes(4)
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

  it("starts a fresh authenticated acquisition immediately after a sole in-flight caller aborts", async () => {
    const asset = await assetFor()
    const firstLoad = deferred<ReturnType<typeof successfulByteResponse>>()
    mocks.fetchWithAuth
      .mockReturnValueOnce(firstLoad.promise)
      .mockResolvedValueOnce(successfulByteResponse(pngBytes))
    const controller = new AbortController()
    const aborted = acquirePersonaVisualAsset(asset, { signal: controller.signal })
    await Promise.resolve()
    controller.abort()
    const retry = acquirePersonaVisualAsset(asset)

    await expect(aborted).rejects.toMatchObject({ code: "asset_aborted" })
    const handle = await retry
    firstLoad.resolve(successfulByteResponse(pngBytes))
    await Promise.resolve()
    const retained = await acquirePersonaVisualAsset(asset)

    expect(mocks.fetchWithAuth).toHaveBeenCalledTimes(2)
    handle.release()
    retained.release()
  })

  it("does not abort a shared load when only one waiting caller aborts", async () => {
    const asset = await assetFor()
    const load = deferred<ReturnType<typeof successfulByteResponse>>()
    mocks.fetchWithAuth.mockReturnValueOnce(load.promise)
    const controller = new AbortController()
    const aborted = acquirePersonaVisualAsset(asset, { signal: controller.signal })
    const retained = acquirePersonaVisualAsset(asset)
    await Promise.resolve()

    controller.abort()
    await expect(aborted).rejects.toMatchObject({ code: "asset_aborted" })
    load.resolve(successfulByteResponse(pngBytes))

    const handle = await retained
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

  it("does not double-revoke a released asset when the cache is cleared before reacquisition", async () => {
    const asset = await assetFor()
    mocks.fetchWithAuth
      .mockResolvedValueOnce(successfulByteResponse(pngBytes))
      .mockResolvedValueOnce(successfulByteResponse(pngBytes))

    const released = await acquirePersonaVisualAsset(asset)
    released.release()
    clearPersonaVisualAssetCache()

    expect(mocks.revokeObjectURL).toHaveBeenCalledTimes(1)
    const reacquired = await acquirePersonaVisualAsset(asset)
    expect(mocks.fetchWithAuth).toHaveBeenCalledTimes(2)
    reacquired.release()
    expect(mocks.revokeObjectURL).toHaveBeenCalledTimes(2)
  })

  it("does not let a cleared in-flight load evict a newer acquisition", async () => {
    const asset = await assetFor()
    const firstLoad = deferred<ReturnType<typeof successfulByteResponse>>()
    mocks.fetchWithAuth
      .mockReturnValueOnce(firstLoad.promise)
      .mockResolvedValueOnce(successfulByteResponse(pngBytes))
    const cleared = acquirePersonaVisualAsset(asset)
    await Promise.resolve()
    clearPersonaVisualAssetCache()
    const retry = acquirePersonaVisualAsset(asset)

    firstLoad.resolve(successfulByteResponse(pngBytes))
    await expect(cleared).rejects.toMatchObject({ code: "asset_aborted" })
    const handle = await retry
    const retained = await acquirePersonaVisualAsset(asset)

    expect(mocks.fetchWithAuth).toHaveBeenCalledTimes(2)
    handle.release()
    retained.release()
  })

  it("uses checksum identity to keep immutable asset revisions distinct", async () => {
    const secondBytes = new Uint8Array([...pngBytes, 0x01])
    const first = await assetFor(pngBytes)
    const second = await assetFor(secondBytes)
    mocks.fetchWithAuth
      .mockResolvedValueOnce(successfulByteResponse(pngBytes))
      .mockResolvedValueOnce(successfulByteResponse(secondBytes))

    const firstHandle = await acquirePersonaVisualAsset(first)
    const secondHandle = await acquirePersonaVisualAsset(second)

    expect(mocks.fetchWithAuth).toHaveBeenCalledTimes(2)
    firstHandle.release()
    secondHandle.release()
    expect(mocks.revokeObjectURL).toHaveBeenCalledTimes(2)
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
