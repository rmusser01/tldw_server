import { describe, expect, it } from "vitest"
import {
  createImageDataUrl,
  inferImageAttachmentMimeType,
  normalizeImageDataUrlMime,
  validateAndCreateImageDataUrl
} from "../image-utils"

const ONE_PIXEL_PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE/wH+J6m3XQAAAABJRU5ErkJggg=="
const WEBP_HEADER_BASE64 = "UklGRhAAAABXRUJQVlA4IAAAAAA="

describe("image utils", () => {
  it("builds a data URL from valid base64 image content", () => {
    expect(createImageDataUrl(ONE_PIXEL_PNG_BASE64)).toBe(
      `data:image/png;base64,${ONE_PIXEL_PNG_BASE64}`
    )
  })

  it("builds a data URL from webp base64 content", () => {
    expect(createImageDataUrl(WEBP_HEADER_BASE64)).toBe(
      `data:image/webp;base64,${WEBP_HEADER_BASE64}`
    )
  })

  it("rejects malformed base64 payloads", () => {
    expect(createImageDataUrl("AA=A")).toBeNull()
    expect(createImageDataUrl("abc$")).toBeNull()
  })

  it("returns null for very large malformed payloads without throwing", () => {
    const malformed = `${"A".repeat(199999)}!`
    expect(() => createImageDataUrl(malformed)).not.toThrow()
    expect(createImageDataUrl(malformed)).toBeNull()
  })

  it("returns an empty string when value is already a data URL", () => {
    expect(validateAndCreateImageDataUrl(`data:image/png;base64,${ONE_PIXEL_PNG_BASE64}`)).toBe(
      ""
    )
  })

  it("infers attachment image MIME type from filenames when browsers provide generic MIME", () => {
    expect(
      inferImageAttachmentMimeType({
        name: "scan.HEIC",
        type: "application/octet-stream"
      })
    ).toBe("image/heic")
    expect(inferImageAttachmentMimeType({ name: "photo.jpg", type: "" })).toBe(
      "image/jpeg"
    )
    expect(inferImageAttachmentMimeType({ name: "png", type: "" })).toBeNull()
  })

  it("normalizes generic image data URLs with the inferred attachment MIME", () => {
    expect(
      normalizeImageDataUrlMime(
        "data:application/octet-stream;base64,ZmFrZQ==",
        "image/jpeg"
      )
    ).toBe("data:image/jpeg;base64,ZmFrZQ==")
  })
})
