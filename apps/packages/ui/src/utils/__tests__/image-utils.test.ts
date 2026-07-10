import { afterEach, describe, expect, it, vi } from "vitest"
import {
  IMAGE_ATTACHMENT_MIME_TYPES,
  createImageDataUrl,
  inferImageAttachmentMimeType,
  normalizeImageDataUrlMime,
  safeImageUrl,
  validateAndCreateImageDataUrl
} from "../image-utils"

const ONE_PIXEL_PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE/wH+J6m3XQAAAABJRU5ErkJggg=="
const WEBP_HEADER_BASE64 = "UklGRhAAAABXRUJQVlA4IAAAAAA="

describe("image utils", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("normalizes absolute HTTP(S) image URLs to lowercase schemes", () => {
    expect(safeImageUrl("HTTPS://example.com/a.png")).toBe(
      "https://example.com/a.png"
    )
    expect(safeImageUrl("HTTP://example.com/a.png")).toBe(
      "http://example.com/a.png"
    )
  })

  it("preserves explicit relative image paths", () => {
    expect(safeImageUrl("/images/a.png")).toBe("/images/a.png")
    expect(safeImageUrl("./images/a.png")).toBe("./images/a.png")
    expect(safeImageUrl("../images/a.png")).toBe("../images/a.png")
  })

  it("makes bare relative image paths explicit", () => {
    expect(safeImageUrl("images/a.png")).toBe("./images/a.png")
  })

  it("accepts valid raster image data", () => {
    expect(safeImageUrl(ONE_PIXEL_PNG_BASE64)).toBe(
      `data:image/png;base64,${ONE_PIXEL_PNG_BASE64}`
    )
  })

  it("rejects non-image URL schemes", () => {
    expect(safeImageUrl("javascript:alert(1)")).toBeNull()
    expect(safeImageUrl("mailto:image@example.com")).toBeNull()
  })

  it("rejects SVG image data", () => {
    expect(safeImageUrl("data:image/svg+xml;base64,PHN2Zy8+")).toBeNull()
  })

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

  it("validates prebuilt data image URLs before returning them", () => {
    expect(createImageDataUrl(`data:image/png;base64,${ONE_PIXEL_PNG_BASE64}`)).toBe(
      `data:image/png;base64,${ONE_PIXEL_PNG_BASE64}`
    )
    expect(createImageDataUrl("data:image/svg+xml;base64,PHN2Zy8+")).toBeNull()
    expect(createImageDataUrl("data:image/png,not-base64")).toBeNull()
  })

  it("rejects data image URLs that mix standard and URL-safe base64 alphabets", () => {
    vi.stubGlobal(
      "atob",
      vi.fn(() => "\x89PNG\r\n\x1a\n")
    )

    expect(createImageDataUrl("data:image/png;base64,AAAA+___")).toBeNull()
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
    expect(inferImageAttachmentMimeType({ name: "favicon.ico", type: "" })).toBe(
      "image/ico"
    )
    expect(inferImageAttachmentMimeType({ name: "png", type: "" })).toBeNull()
    expect(
      inferImageAttachmentMimeType({
        name: "photo.png",
        type: "application/zip"
      })
    ).toBeNull()
  })

  it("normalizes generic image data URLs with the inferred attachment MIME", () => {
    expect(
      normalizeImageDataUrlMime(
        "data:application/octet-stream;base64,ZmFrZQ==",
        "image/jpeg"
      )
    ).toBe("data:image/jpeg;base64,ZmFrZQ==")
  })

  it("leaves generic data URLs unchanged when no attachment MIME is available", () => {
    const dataUrl = "data:application/octet-stream;base64,ZmFrZQ=="

    expect(normalizeImageDataUrlMime(dataUrl, null)).toBe(dataUrl)
    expect(normalizeImageDataUrlMime(dataUrl, undefined)).toBe(dataUrl)
  })

  it("keeps the shared unsupported image MIME policy aligned with legacy ico MIME", () => {
    expect(IMAGE_ATTACHMENT_MIME_TYPES.has("image/ico")).toBe(true)
    expect(IMAGE_ATTACHMENT_MIME_TYPES.has("image/x-icon")).toBe(false)
  })
})
