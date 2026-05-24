import { describe, expect, it } from "vitest"
import {
  WORKSPACE_BANNER_MAX_BYTES,
  normalizeWorkspaceBannerImage,
  WorkspaceBannerImageNormalizationError
} from "../workspace-banner-image"

const createImageFile = (type: string, name = "banner"): File =>
  new File([new Uint8Array([1, 2, 3, 4])], `${name}.bin`, { type })

describe("normalizeWorkspaceBannerImage", () => {
  it("rejects unsupported mime types", async () => {
    const file = createImageFile("image/gif")

    await expect(normalizeWorkspaceBannerImage(file)).rejects.toMatchObject({
      name: WorkspaceBannerImageNormalizationError.name,
      code: "unsupported_mime_type"
    })
  })

  it("returns normalized payload under max byte cap", async () => {
    const file = createImageFile("image/png")
    const now = new Date("2026-02-25T08:00:00.000Z")

    const normalized = await normalizeWorkspaceBannerImage(file, {
      maxEdgePx: 1400,
      maxBytes: WORKSPACE_BANNER_MAX_BYTES,
      deps: {
        readFileAsDataUrl: async () => "data:image/png;base64,AAAA",
        decodeImageDimensions: async () => ({ width: 2800, height: 1400 }),
        encodeImageToWebp: async () => "data:image/webp;base64,AAAAAAAAAAAA",
        now: () => now
      }
    })

    expect(normalized.mimeType).toBe("image/webp")
    expect(normalized.width).toBe(1400)
    expect(normalized.height).toBe(700)
    expect(normalized.bytes).toBeGreaterThan(0)
    expect(normalized.updatedAt).toBe(now)
  })

  it("throws when normalized payload exceeds byte cap", async () => {
    const file = createImageFile("image/jpeg")

    await expect(
      normalizeWorkspaceBannerImage(file, {
        maxBytes: 1,
        deps: {
          readFileAsDataUrl: async () =>
            "data:image/jpeg;base64,AAAAAAAAAAAAAAAAAAAA",
          decodeImageDimensions: async () => ({ width: 1200, height: 400 }),
          encodeImageToWebp: async () =>
            "data:image/webp;base64,AAAAAAAAAAAAAAAAAAAA"
        }
      })
    ).rejects.toMatchObject({
      code: "image_too_large"
    })
  })
})
