import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getCurrentUser: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", async () => {
  const actual = await vi.importActual<Record<string, any>>("@/services/tldw/TldwApiClient")
  return {
    ...actual,
    tldwClient: { ...actual.tldwClient, getConfig: (...args: any[]) => mocks.getConfig(...args) }
  }
})

vi.mock("@/services/tldw/TldwAuth", async () => {
  const actual = await vi.importActual<Record<string, any>>("@/services/tldw/TldwAuth")
  return {
    ...actual,
    tldwAuth: { ...actual.tldwAuth, getCurrentUser: (...args: any[]) => mocks.getCurrentUser(...args) }
  }
})

const loadRecovery = () =>
  vi.importActual<Record<string, any>>(["..", "standalone-html-recovery"].join("/"))
const loadSource = () =>
  vi.importActual<Record<string, any>>(["..", "standalone-html-source"].join("/"))
const loadScopeHook = () =>
  vi.importActual<Record<string, any>>(
    ["..", "..", "..", "..", "hooks", "usePresentationPrincipalScope"].join("/")
  )

const SOURCE = "<!doctype html><title>Recovered 😀</title>"

describe("standalone HTML workspace recovery records", () => {
  beforeEach(() => {
    sessionStorage.clear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("canonicalizes the trusted origin/principal scope and stores only the closed 24-hour schema", async () => {
    const recovery = await loadRecovery()
    const source = await loadSource()
    const accepted = await source.validateStandaloneHtmlSource(SOURCE)
    const scope = recovery.createPresentationPrincipalScope(
      "HTTPS://TLDW.Example:443/a/path?ignored=1",
      " owner 42 "
    )

    expect(scope).toEqual({
      serverOrigin: "https://tldw.example",
      principalId: "owner 42",
      principalScope: "https://tldw.example|owner%2042"
    })
    const written = recovery.writeStandaloneHtmlRecovery(sessionStorage, scope, {
      presentationId: "html/1",
      baseEtag: '"opaque-v7"',
      baseDigest: "a".repeat(64),
      acceptedSource: accepted,
      updatedAt: 1_800_000_000_000
    })
    expect(written).toEqual(expect.objectContaining({ ok: true }))
    expect(sessionStorage.length).toBe(1)
    const key = sessionStorage.key(0)!
    expect(key).toContain("https%3A%2F%2Ftldw.example")
    expect(key).toContain("owner%2042")
    expect(key).toContain("html%2F1")

    const raw = JSON.parse(sessionStorage.getItem(key)!)
    expect(raw).toEqual({
      schemaVersion: 1,
      principalScope: "https://tldw.example|owner%2042",
      presentationId: "html/1",
      baseEtag: '"opaque-v7"',
      baseDigest: "a".repeat(64),
      source: SOURCE,
      updatedAt: 1_800_000_000_000
    })
    expect(Object.keys(raw)).toHaveLength(7)
  })

  it("reads only after a current trusted scope is supplied and revalidates exact bytes without autoapplying", async () => {
    const recovery = await loadRecovery()
    const source = await loadSource()
    const accepted = await source.validateStandaloneHtmlSource(SOURCE)
    const scope = recovery.createPresentationPrincipalScope("https://tldw.example", "42")
    recovery.writeStandaloneHtmlRecovery(sessionStorage, scope, {
      presentationId: "html-1",
      baseEtag: '"v7"',
      baseDigest: "b".repeat(64),
      acceptedSource: accepted,
      updatedAt: Date.now()
    })

    const result = await recovery.readStandaloneHtmlRecovery(
      sessionStorage,
      scope,
      "html-1",
      Date.now()
    )

    expect(result).toEqual({
      kind: "available",
      record: {
        schemaVersion: 1,
        principalScope: scope.principalScope,
        presentationId: "html-1",
        baseEtag: '"v7"',
        baseDigest: "b".repeat(64),
        source: SOURCE,
        updatedAt: expect.any(Number)
      },
      acceptedSource: expect.objectContaining({
        ok: true,
        source: SOURCE,
        bytes: new TextEncoder().encode(SOURCE)
      })
    })
    expect(result.kind).toBe("available")
    expect((result as any).applied).toBeUndefined()

    const otherScope = recovery.createPresentationPrincipalScope("https://tldw.example", "99")
    await expect(
      recovery.readStandaloneHtmlRecovery(sessionStorage, otherScope, "html-1", Date.now())
    ).resolves.toEqual({ kind: "none" })
  })

  it("expires and clears records older than exactly 24 hours before returning source", async () => {
    const recovery = await loadRecovery()
    const source = await loadSource()
    const accepted = await source.validateStandaloneHtmlSource(SOURCE)
    const scope = recovery.createPresentationPrincipalScope("https://tldw.example", "42")
    const now = 1_900_000_000_000
    recovery.writeStandaloneHtmlRecovery(sessionStorage, scope, {
      presentationId: "html-1",
      baseEtag: '"v7"',
      baseDigest: "c".repeat(64),
      acceptedSource: accepted,
      updatedAt: now - 86_400_001
    })

    await expect(
      recovery.readStandaloneHtmlRecovery(sessionStorage, scope, "html-1", now)
    ).resolves.toEqual({ kind: "none" })
    expect(sessionStorage.length).toBe(0)
  })

  it("refuses an oversized serialized record before JSON parsing or source encoding", async () => {
    const recovery = await loadRecovery()
    const scope = recovery.createPresentationPrincipalScope("https://tldw.example", "42")
    const raw = "x".repeat(6 * 1_048_576 + 8_193)
    const storage = {
      getItem: vi.fn(() => raw),
      setItem: vi.fn(),
      removeItem: vi.fn()
    }
    const parse = vi.spyOn(JSON, "parse")
    const RealEncoder = globalThis.TextEncoder
    const encoder = vi.fn(() => {
      throw new Error("oversized recovery must not be encoded")
    })
    Object.defineProperty(globalThis, "TextEncoder", {
      configurable: true,
      writable: true,
      value: encoder
    })

    try {
      await expect(
        recovery.readStandaloneHtmlRecovery(storage, scope, "html-1", Date.now())
      ).resolves.toEqual({ kind: "none" })
      expect(parse).not.toHaveBeenCalled()
      expect(encoder).not.toHaveBeenCalled()
      expect(storage.removeItem).toHaveBeenCalledTimes(1)
    } finally {
      Object.defineProperty(globalThis, "TextEncoder", {
        configurable: true,
        writable: true,
        value: RealEncoder
      })
    }
  })

  it.each([
    ["U+0000", "bad\u0000source"],
    ["lone high surrogate", "bad\ud800source"],
    ["lone low surrogate", "bad\udc00source"],
    ["more than 1 MiB", "x".repeat(1_048_577)]
  ])("rejects %s before encoding or sessionStorage persistence", async (_case, invalidSource) => {
    const recovery = await loadRecovery()
    const scope = recovery.createPresentationPrincipalScope("https://tldw.example", "42")
    const storage = { setItem: vi.fn(), getItem: vi.fn(), removeItem: vi.fn() }
    const RealEncoder = globalThis.TextEncoder
    const encoder = vi.fn(() => {
      throw new Error("encoding invalid source would violate the scalar boundary")
    })
    Object.defineProperty(globalThis, "TextEncoder", {
      configurable: true,
      writable: true,
      value: encoder
    })

    try {
      const result = recovery.writeStandaloneHtmlRecovery(storage, scope, {
        presentationId: "html-1",
        baseEtag: '"v7"',
        baseDigest: "d".repeat(64),
        acceptedSource: { source: invalidSource, digest: "e".repeat(64) },
        updatedAt: Date.now()
      })

      expect(result).toEqual(expect.objectContaining({ ok: false }))
      expect(storage.setItem).not.toHaveBeenCalled()
      expect(encoder).not.toHaveBeenCalled()
    } finally {
      Object.defineProperty(globalThis, "TextEncoder", {
        configurable: true,
        writable: true,
        value: RealEncoder
      })
    }
  })

  it("keeps memory usable and returns a persistent bounded warning when storage fails", async () => {
    const recovery = await loadRecovery()
    const source = await loadSource()
    const accepted = await source.validateStandaloneHtmlSource(SOURCE)
    const scope = recovery.createPresentationPrincipalScope("https://tldw.example", "42")
    const storage = {
      setItem: vi.fn(() => {
        throw new DOMException("quota", "QuotaExceededError")
      }),
      getItem: vi.fn(),
      removeItem: vi.fn()
    }

    const result = recovery.writeStandaloneHtmlRecovery(storage, scope, {
      presentationId: "html-1",
      baseEtag: '"v7"',
      baseDigest: "f".repeat(64),
      acceptedSource: accepted,
      updatedAt: Date.now()
    })

    expect(result).toEqual({
      ok: false,
      code: "recovery_unavailable",
      message: "Recovery unavailable. Keep this tab open or download your draft."
    })
    expect(accepted.source).toBe(SOURCE)
    expect(result.message.length).toBeLessThanOrEqual(100)
  })

  it("clears only the matching scoped record after a matching save or confirmed discard", async () => {
    const recovery = await loadRecovery()
    const source = await loadSource()
    const accepted = await source.validateStandaloneHtmlSource(SOURCE)
    const owner = recovery.createPresentationPrincipalScope("https://tldw.example", "42")
    const other = recovery.createPresentationPrincipalScope("https://tldw.example", "99")
    for (const scope of [owner, other]) {
      recovery.writeStandaloneHtmlRecovery(sessionStorage, scope, {
        presentationId: "html-1",
        baseEtag: '"v7"',
        baseDigest: accepted.digest,
        acceptedSource: accepted,
        updatedAt: Date.now()
      })
    }

    expect(recovery.clearStandaloneHtmlRecovery(sessionStorage, owner, "html-1")).toBe(true)
    expect(sessionStorage.length).toBe(1)
    expect(sessionStorage.key(0)).toContain("99")
  })
})

describe("usePresentationPrincipalScope", () => {
  beforeEach(() => {
    mocks.getConfig.mockReset().mockResolvedValue({ serverUrl: "https://TLDW.Example:443/path" })
    mocks.getCurrentUser.mockReset().mockResolvedValue({ id: 42, username: "owner", is_active: true })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("derives a canonical nonsecret scope and confirms it before reporting ready", async () => {
    const subject = await loadScopeHook()
    const { result } = renderHook(() => subject.usePresentationPrincipalScope())

    await waitFor(() => expect(result.current.status).toBe("ready"))
    expect(result.current.scope).toEqual({
      serverOrigin: "https://tldw.example",
      principalId: "42",
      principalScope: "https://tldw.example|42"
    })
  })

  it("fires synchronous boundary disposal before a logout or subject switch can reauthenticate", async () => {
    const subject = await loadScopeHook()
    const boundary = vi.fn()
    const { result } = renderHook(() =>
      subject.usePresentationPrincipalScope({ onBoundary: boundary })
    )
    await waitFor(() => expect(result.current.status).toBe("ready"))
    let resolveUser: ((value: unknown) => void) | null = null
    mocks.getCurrentUser.mockReturnValue(
      new Promise((resolve) => {
        resolveUser = resolve
      })
    )

    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "switch" } })
      )
    )

    expect(boundary).toHaveBeenCalledTimes(1)
    expect(result.current.scope).toBeNull()
    expect(result.current.status).toBe("loading")
    resolveUser?.({ id: 99, username: "other", is_active: true })
    await waitFor(() => expect(result.current.scope?.principalId).toBe("99"))

    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })
      )
    )
    expect(boundary).toHaveBeenCalledTimes(2)
    expect(result.current.status).toBe("guarded")
    expect(result.current.scope).toBeNull()
  })

  it("fences stale async scope resolutions and reauthenticates on pageshow, focus, and visibility restoration", async () => {
    const subject = await loadScopeHook()
    const { result } = renderHook(() => subject.usePresentationPrincipalScope())
    await waitFor(() => expect(result.current.status).toBe("ready"))
    const initialCalls = mocks.getCurrentUser.mock.calls.length

    act(() => window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true })))
    await waitFor(() => expect(mocks.getCurrentUser.mock.calls.length).toBeGreaterThan(initialCalls))

    const afterPageShow = mocks.getCurrentUser.mock.calls.length
    act(() => window.dispatchEvent(new Event("focus")))
    await waitFor(() => expect(mocks.getCurrentUser.mock.calls.length).toBeGreaterThan(afterPageShow))

    const descriptor = Object.getOwnPropertyDescriptor(document, "visibilityState")
    Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" })
    const afterFocus = mocks.getCurrentUser.mock.calls.length
    act(() => document.dispatchEvent(new Event("visibilitychange")))
    await waitFor(() => expect(mocks.getCurrentUser.mock.calls.length).toBeGreaterThan(afterFocus))
    if (descriptor) Object.defineProperty(document, "visibilityState", descriptor)
  })
})
