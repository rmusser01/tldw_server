import { beforeEach, describe, expect, it, vi } from "vitest"

const mockBgRequest = vi.hoisted(() => vi.fn())
const mockBgUpload = vi.hoisted(() => vi.fn())

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest,
  bgUpload: mockBgUpload
}))

import {
  acquireFlashcardAssetObjectUrl,
  clearFlashcardAssetObjectUrlCache,
  releaseFlashcardAssetObjectUrl,
  uploadFlashcardAsset
} from "@/services/flashcard-assets"

describe("flashcard asset service", () => {
  beforeEach(() => {
    mockBgRequest.mockReset()
    mockBgUpload.mockReset()
    clearFlashcardAssetObjectUrlCache()
  })

  it("uploads flashcard assets through the managed endpoint", async () => {
    mockBgUpload.mockResolvedValue({
      asset_uuid: "asset-1",
      reference: "flashcard-asset://asset-1",
      markdown_snippet: "![Slide](flashcard-asset://asset-1)"
    })

    const file = {
      name: "slide.png",
      type: "image/png",
      arrayBuffer: vi.fn(async () => Uint8Array.from([1, 2, 3]).buffer)
    } as unknown as File

    await uploadFlashcardAsset(file)

    expect(mockBgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/flashcards/assets",
        method: "POST",
        fileFieldName: "file",
        file: expect.objectContaining({
          name: "slide.png",
          type: "image/png",
          data: Uint8Array.from([1, 2, 3])
        })
      })
    )
  })

  it("does not upload after cancellation while file bytes are being read", async () => {
    const controller = new AbortController()
    let resolve!: (value: ArrayBuffer) => void
    const file = { name: "diagram.png", type: "image/png", arrayBuffer: () => new Promise<ArrayBuffer>(done => { resolve = done }) } as File
    const pending = uploadFlashcardAsset(file, { signal: controller.signal })
    controller.abort()
    resolve(new Uint8Array([1]).buffer)
    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(mockBgUpload).not.toHaveBeenCalled()
  })

  it("caches managed asset object URLs and revokes them after the last release", async () => {
    mockBgRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: Uint8Array.from([137, 80, 78, 71]).buffer,
      headers: { "content-type": "image/png" }
    })

    const createObjectUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:flashcard-asset-1")
    const revokeObjectUrl = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => undefined)

    const first = await acquireFlashcardAssetObjectUrl("asset-1")
    const second = await acquireFlashcardAssetObjectUrl("asset-1")

    expect(first).toBe("blob:flashcard-asset-1")
    expect(second).toBe("blob:flashcard-asset-1")
    expect(mockBgRequest).toHaveBeenCalledTimes(1)
    expect(createObjectUrl).toHaveBeenCalledTimes(1)

    releaseFlashcardAssetObjectUrl("asset-1")
    expect(revokeObjectUrl).not.toHaveBeenCalled()

    releaseFlashcardAssetObjectUrl("asset-1")
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:flashcard-asset-1")
  })
})
