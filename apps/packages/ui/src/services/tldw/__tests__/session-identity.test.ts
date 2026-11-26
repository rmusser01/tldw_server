import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  clearSessionReplacedNotice,
  getInstallIdHeaders,
  hasSessionReplacedNotice,
  noteSessionReplaced
} from "@/services/tldw/session-identity"

describe("session identity", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("creates one install id and reuses it", () => {
    const first = getInstallIdHeaders()["X-TLDW-Device-Id"]
    expect(first).toMatch(/^[A-Za-z0-9_-]{8,64}$/)
    expect(getInstallIdHeaders()["X-TLDW-Device-Id"]).toBe(first)
    expect(localStorage.getItem("tldw:install-id")).toBe(first)
  })

  it("omits the header when storage is unavailable", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("denied")
    })
    expect(getInstallIdHeaders()).toEqual({})
  })

  it("falls back to getRandomValues when randomUUID is missing", () => {
    vi.stubGlobal("crypto", { getRandomValues: (a: Uint8Array) => a.fill(171) })
    expect(getInstallIdHeaders()["X-TLDW-Device-Id"]).toBe("ab".repeat(16))
  })

  it("omits the header when no crypto is available", () => {
    vi.stubGlobal("crypto", undefined)
    expect(getInstallIdHeaders()).toEqual({})
  })

  it("records the notice only for a 401 session_replaced", () => {
    noteSessionReplaced({ status: 401, details: { detail: "Invalid or expired refresh token" } })
    noteSessionReplaced({ status: 500, details: { detail: { code: "session_replaced" } } })
    expect(hasSessionReplacedNotice()).toBe(false)
    noteSessionReplaced({ status: 401, details: { detail: { code: "session_replaced" } } })
    expect(hasSessionReplacedNotice()).toBe(true)
    clearSessionReplacedNotice()
    expect(hasSessionReplacedNotice()).toBe(false)
  })
})
