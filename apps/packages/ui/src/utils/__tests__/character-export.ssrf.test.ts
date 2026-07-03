import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  exportCharacterToPNG,
  isSafeAvatarFetchUrl
} from "../character-export"

// A minimal-but-valid PNG: 8-byte signature + IHDR chunk (length 13). Enough for
// embedMetadataInPNG's signature/IHDR checks so the export can complete.
const makeMinimalPng = (): Uint8Array => {
  const bytes = new Uint8Array(33)
  bytes.set([137, 80, 78, 71, 13, 10, 26, 10], 0) // PNG signature
  bytes[8] = 0
  bytes[9] = 0
  bytes[10] = 0
  bytes[11] = 13 // IHDR length = 13
  bytes.set([73, 72, 68, 82], 12) // "IHDR"
  // remaining 13 IHDR data bytes + 4 CRC bytes stay zeroed
  return bytes
}

const makeResponse = (
  body: Uint8Array,
  headers: Record<string, string> = {}
): any => ({
  ok: true,
  status: 200,
  statusText: "OK",
  headers: {
    get: (name: string) => headers[name.toLowerCase()] ?? null
  },
  arrayBuffer: async () => body.buffer
})

const originalCreateObjectURL = (URL as any).createObjectURL
const originalRevokeObjectURL = (URL as any).revokeObjectURL
let createObjectURLSpy: ReturnType<typeof vi.fn>

beforeEach(() => {
  createObjectURLSpy = vi.fn(() => "blob:mock")
  ;(URL as any).createObjectURL = createObjectURLSpy
  ;(URL as any).revokeObjectURL = vi.fn()
})

afterEach(() => {
  ;(URL as any).createObjectURL = originalCreateObjectURL
  ;(URL as any).revokeObjectURL = originalRevokeObjectURL
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe("isSafeAvatarFetchUrl", () => {
  it("allows same-origin http(s) URLs", () => {
    expect(isSafeAvatarFetchUrl(`${window.location.origin}/media/a.png`)).toBe(
      true
    )
  })

  it("rejects cross-origin URLs (SSRF / beacon guard)", () => {
    expect(isSafeAvatarFetchUrl("https://evil.example.com/beacon.png")).toBe(
      false
    )
    expect(isSafeAvatarFetchUrl("http://169.254.169.254/latest/meta-data")).toBe(
      false
    )
  })

  it("allows an explicitly allowlisted origin (e.g. the configured server)", () => {
    expect(
      isSafeAvatarFetchUrl("http://127.0.0.1:8000/media/a.png", [
        "http://127.0.0.1:8000"
      ])
    ).toBe(true)
  })

  it("rejects non-http(s) protocols", () => {
    expect(isSafeAvatarFetchUrl("file:///etc/passwd")).toBe(false)
    expect(isSafeAvatarFetchUrl("javascript:alert(1)")).toBe(false)
  })

  it("rejects empty input", () => {
    expect(isSafeAvatarFetchUrl("")).toBe(false)
    expect(isSafeAvatarFetchUrl("   ")).toBe(false)
  })
})

describe("exportCharacterToPNG avatar fetch hardening", () => {
  it("does NOT fetch a cross-origin avatar_url", async () => {
    const fetchFn = vi.fn(async () => makeResponse(makeMinimalPng()))
    vi.stubGlobal("fetch", fetchFn)

    // Cross-origin avatar is skipped; export falls back to a local placeholder
    // (canvas is unavailable in jsdom, so this may reject — that's incidental).
    await exportCharacterToPNG(
      { name: "Ada" },
      { avatarUrl: "https://evil.example.com/beacon.png" }
    ).catch(() => undefined)

    expect(fetchFn).not.toHaveBeenCalled()
  })

  it("fetches a same-origin avatar with an AbortSignal and no credentials", async () => {
    const png = makeMinimalPng()
    const fetchFn = vi.fn(async () =>
      makeResponse(png, { "content-length": String(png.byteLength) })
    )
    vi.stubGlobal("fetch", fetchFn)

    const sameOriginUrl = `${window.location.origin}/media/avatar.png`
    await exportCharacterToPNG({ name: "Ada" }, { avatarUrl: sameOriginUrl })

    expect(fetchFn).toHaveBeenCalledTimes(1)
    const [calledUrl, init] = fetchFn.mock.calls[0] as [string, RequestInit]
    expect(calledUrl).toBe(sameOriginUrl)
    expect(init?.credentials).toBe("omit")
    expect(init?.signal).toBeInstanceOf(AbortSignal)
    // Download path ran, so the fetched avatar was embedded.
    expect(createObjectURLSpy).toHaveBeenCalled()
  })

  it("bails before reading the body when content-length exceeds the size cap", async () => {
    const arrayBufferSpy = vi.fn(async () => makeMinimalPng().buffer)
    const oversized = 5 * 1024 * 1024 + 1
    const fetchFn = vi.fn(async () => ({
      ok: true,
      status: 200,
      statusText: "OK",
      headers: {
        get: (name: string) =>
          name.toLowerCase() === "content-length" ? String(oversized) : null
      },
      arrayBuffer: arrayBufferSpy
    }))
    vi.stubGlobal("fetch", fetchFn)

    const sameOriginUrl = `${window.location.origin}/media/huge.png`
    await exportCharacterToPNG({ name: "Ada" }, { avatarUrl: sameOriginUrl }).catch(
      () => undefined
    )

    expect(fetchFn).toHaveBeenCalledTimes(1)
    expect(arrayBufferSpy).not.toHaveBeenCalled()
  })

  it("decodes an inline data: avatar without any network request", async () => {
    const fetchFn = vi.fn(async () => makeResponse(makeMinimalPng()))
    vi.stubGlobal("fetch", fetchFn)

    const png = makeMinimalPng()
    let binary = ""
    png.forEach((byte) => {
      binary += String.fromCharCode(byte)
    })
    const dataUrl = `data:image/png;base64,${btoa(binary)}`

    await exportCharacterToPNG({ name: "Ada" }, { avatarUrl: dataUrl })

    expect(fetchFn).not.toHaveBeenCalled()
    expect(createObjectURLSpy).toHaveBeenCalled()
  })

  it("embeds an already-local base64 avatar without a network request", async () => {
    const fetchFn = vi.fn(async () => makeResponse(makeMinimalPng()))
    vi.stubGlobal("fetch", fetchFn)

    const png = makeMinimalPng()
    let binary = ""
    png.forEach((byte) => {
      binary += String.fromCharCode(byte)
    })

    await exportCharacterToPNG(
      { name: "Ada" },
      {
        avatarBase64: btoa(binary),
        // A hostile avatarUrl is ignored entirely when base64 is present.
        avatarUrl: "https://evil.example.com/beacon.png"
      }
    )

    expect(fetchFn).not.toHaveBeenCalled()
    expect(createObjectURLSpy).toHaveBeenCalled()
  })
})
