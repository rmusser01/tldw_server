import React from "react"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn(), watchers: new Set<Record<string, (change: unknown) => void>>(), nativeListeners: new Set<(changes: Record<string, unknown>, area: string) => void>() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null }, storage: { onChanged: {
  addListener: (listener: (changes: Record<string, unknown>, area: string) => void) => boundary.nativeListeners.add(listener),
  removeListener: (listener: (changes: Record<string, unknown>, area: string) => void) => boundary.nativeListeners.delete(listener)
} } } }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({ get: boundary.get, set: vi.fn(), remove: vi.fn(),
  watch: (handlers: Record<string, (change: unknown) => void>) => boundary.watchers.add(handlers),
  unwatch: (handlers: Record<string, (change: unknown) => void>) => boundary.watchers.delete(handlers)
}), safeStorageSerde: { serialize: JSON.stringify, deserialize: JSON.parse } }))
vi.mock("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false }))
import { useFlashcardReviewRun } from "../useFlashcardReviewRun"
import { useReviewFlashcardMutation, useEndFlashcardReviewSessionMutation } from "../useFlashcardQueries"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import { isServicePromptRequestPath } from "@/services/tldw/service-prompt-scope-error"
import { resolveDirectBrowserConfig } from "@/services/tldw/direct-browser-config"
import { invalidateRefreshSessionIfCurrent, refreshSessionInvalidationKey, storeRefreshRotationIfCurrent, REFRESH_ROTATION_KEY, type CredentialStorage } from "@/services/tldw/single-user-credential"
import type { FlashcardReviewContext } from "@/services/flashcards"

const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done }); return { promise, resolve } }
let config: TldwConfig
let userId: number
let sessionId: number
let queryClient: QueryClient
const closeError = vi.fn()
const globalContext: FlashcardReviewContext = { review_mode: "due", deck_id: null, tag_filter: null }
const ratingCalls = () => boundary.fetch.mock.calls.filter(([url]) => String(url).endsWith("/flashcards/review"))
const endCalls = () => boundary.fetch.mock.calls.filter(([url]) => String(url).endsWith("/flashcards/review-sessions/end"))
const token = (id: number) => `test.${btoa(JSON.stringify({ sub: String(id) }))}.signature`
const switchAccount = (id: number) => {
  userId = id
  config = { ...config, accessToken: token(id) }
  for (const handlers of [...boundary.watchers]) handlers.tldwConfig?.({ newValue: { ...config } })
  window.dispatchEvent(new CustomEvent("tldw:auth-credentials-changed"))
}
function setup(initial = globalContext) {
  return renderHook(({ context, enabled }) => {
    const review = useReviewFlashcardMutation()
    const end = useEndFlashcardReviewSessionMutation()
    return useFlashcardReviewRun({ context, enabled, review: review.mutateAsync, end: end.mutateAsync, onCloseError: closeError })
  }, { initialProps: { context: initial, enabled: true }, wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider> })
}
const submit = (result: ReturnType<typeof setup>["result"], cardUuid = "card-1") => act(async () => { await result.current.submit({ cardUuid, rating: 3 }) })

const canonicalCredentials = () => {
  config = { ...config, refreshToken: "synthetic-source-refresh" }
  const records = new Map<string, unknown>([["tldwConfig", { ...config }]])
  const storage: CredentialStorage = {
    get: async <T,>(key: string) => records.get(key) as T | undefined,
    set: async <T,>(key: string, value: T) => { records.set(key, value) },
    remove: async (key: string) => { records.delete(key) }
  }
  boundary.get.mockImplementation(storage.get)
  vi.mocked(tldwClient.getConfig).mockImplementation(() => resolveDirectBrowserConfig(storage))
  // Use the actual canonical projection and credential validation, including marker/rotation semantics.
  vi.mocked(tldwClient.ensureConfigForRequest).mockRestore()
  return { storage, records }
}

const holdOperation = async (view: ReturnType<typeof setup>, operation: string) => {
  if (operation === "end") await submit(view.result)
  const pending = deferred<Response>(); const base = boundary.fetch.getMockImplementation()!
  const path = operation === "end" ? "/review-sessions/end" : "/flashcards/review"
  let intercepted = false
  let signal: AbortSignal | undefined
  boundary.fetch.mockImplementation((url, init) => {
    if (!intercepted && String(url).endsWith(path)) { intercepted = true; signal = init.signal; return pending.promise }
    return base(url, init)
  })
  let result!: Promise<unknown>
  act(() => { result = operation === "end" ? view.result.current.complete() : view.result.current.submit({ cardUuid: "held", rating: 3 }) })
  await waitFor(() => expect(intercepted).toBe(true))
  return { pending, result, signal: signal! }
}

describe("review run with real scope lease, mutations and outbound transport", () => {
  beforeEach(() => {
    userId = 7; sessionId = 77
    config = { serverUrl: window.location.origin, authMode: "multi-user", accessToken: token(userId) }
    boundary.watchers.clear(); boundary.nativeListeners.clear(); closeError.mockReset()
    boundary.get.mockImplementation(async key => key === "tldwConfig" ? { ...config } : null)
    boundary.fetch.mockReset().mockImplementation(async (url: string, init: RequestInit) => {
      if (String(url).endsWith("/api/auth/session")) return response({ authenticated: true, user: { id: userId, is_active: true } })
      if (String(url).endsWith("/auth/me")) return response({ id: userId, is_active: true })
      if (String(url).endsWith("/flashcards/review")) return response({ review_session_id: sessionId, interval_days: 1 })
      if (String(url).endsWith("/review-sessions/end")) return response({ id: JSON.parse(String(init.body)).review_session_id, status: "completed", cards_reviewed: 7 })
      throw new Error(`Unexpected request ${url}`)
    })
    vi.stubGlobal("fetch", boundary.fetch)
    vi.spyOn(tldwClient, "initialize").mockResolvedValue()
    vi.spyOn(tldwClient, "getConfig").mockImplementation(async () => ({ ...config }))
    vi.spyOn(tldwClient, "ensureConfigForRequest").mockImplementation(async () => ({ ...config }))
    queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false }, queries: { retry: false } } })
  })
  afterEach(async () => { await act(async () => cleanup()); queryClient.clear(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.unstubAllEnvs() })

  it("sends seven explicit global ratings with one retained ID and ends that ID", async () => {
    const { result } = setup()
    for (let i = 0; i < 7; i++) await submit(result, `mixed-${i}`)
    expect(ratingCalls()).toHaveLength(7)
    ratingCalls().forEach(([, init], index) => {
      expect(JSON.parse(init.body)).toEqual({ card_uuid: `mixed-${index}`, rating: 3, review_context: globalContext,
        ...(index ? { review_session_id: 77 } : {}) })
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    })
    await act(async () => { expect(await result.current.complete()).toMatchObject({ id: 77, cards_reviewed: 7 }) })
    expect(endCalls()).toHaveLength(1)
    expect(new Headers(endCalls()[0][1].headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    expect(result.current.activeSessionId).toBeNull()
  })

  it("serializes the first rating and blocks repeated keyboard submissions until it settles", async () => {
    const pending = deferred<Response>(); const base = boundary.fetch.getMockImplementation()!
    boundary.fetch.mockImplementation((url, init) => String(url).endsWith("/flashcards/review") ? pending.promise : base(url, init))
    const { result } = setup()
    let first!: Promise<unknown>
    act(() => { first = result.current.submit({ cardUuid: "first", rating: 3 }) })
    await waitFor(() => expect(ratingCalls()).toHaveLength(1))
    await act(async () => { expect(await result.current.submit({ cardUuid: "duplicate", rating: 3 })).toBeNull() })
    await act(async () => { pending.resolve(response({ review_session_id: 77 })); await first })
    expect(ratingCalls()).toHaveLength(1)
  })

  it.each([
    { review_mode: "due", deck_id: 1, tag_filter: null },
    { review_mode: "cram", deck_id: null, tag_filter: "biology" }
  ] as FlashcardReviewContext[])("waits for a dispatched first rating before closing its old scope ($review_mode/$deck_id/$tag_filter)", async next => {
    const pending = deferred<Response>(); const base = boundary.fetch.getMockImplementation()!
    boundary.fetch.mockImplementation((url, init) => String(url).endsWith("/flashcards/review") && ratingCalls().length === 1 ? pending.promise : base(url, init))
    const view = setup()
    let first!: Promise<unknown>
    act(() => { first = view.result.current.submit({ cardUuid: "old", rating: 3 }) })
    await waitFor(() => expect(ratingCalls()).toHaveLength(1))
    view.rerender({ context: next, enabled: true })
    expect(endCalls()).toHaveLength(0)
    sessionId = 88
    await submit(view.result, "new")
    await act(async () => { pending.resolve(response({ review_session_id: 77 })); expect(await first).toBeNull() })
    await waitFor(() => expect(endCalls()).toHaveLength(1))
    expect(JSON.parse(endCalls()[0][1].body)).toEqual({ review_session_id: 77 })
    expect(view.result.current.activeSessionId).toBe(88)
    expect(JSON.parse(ratingCalls()[1][1].body).review_context).toEqual(next)
  })

  it.each(["rating", "end"])("does not install or clear a replacement run after a late %s in account A→B→A", async operation => {
    const view = setup()
    if (operation === "end") await submit(view.result)
    const pending = deferred<Response>(); const base = boundary.fetch.getMockImplementation()!
    const path = operation === "end" ? "/review-sessions/end" : "/flashcards/review"
    let intercepted = false
    boundary.fetch.mockImplementation((url, init) => {
      if (!intercepted && String(url).endsWith(path)) { intercepted = true; return pending.promise }
      return base(url, init)
    })
    let old!: Promise<unknown>
    act(() => { old = operation === "end" ? view.result.current.complete() : view.result.current.submit({ cardUuid: "old", rating: 3 }) })
    await waitFor(() => expect(intercepted).toBe(true))
    act(() => switchAccount(8))
    await submit(view.result, "bob")
    act(() => switchAccount(7))
    await submit(view.result, "alice-replacement")
    const invalidation = vi.spyOn(queryClient, "invalidateQueries")
    invalidation.mockClear()
    await act(async () => { pending.resolve(response({ review_session_id: 77, id: 77 })); expect(await old).toBeNull() })
    expect(view.result.current.activeSessionId).toBe(77)
    expect(invalidation).not.toHaveBeenCalled()
    expect(endCalls()).toHaveLength(operation === "end" ? 1 : 0)
  })

  it("blocks a target change before dispatch instead of sending current credentials to the old target", async () => {
    const view = setup()
    await waitFor(() => expect(boundary.fetch.mock.calls.some(([url]) => String(url).endsWith("/auth/me"))).toBe(true))
    await act(async () => {})
    config = { ...config, serverUrl: "https://other.test", accessToken: token(8) }
    await act(async () => { await expect(view.result.current.submit({ cardUuid: "old", rating: 3 })).rejects.toMatchObject({ status: 412 }) })
    expect(ratingCalls()).toHaveLength(0)
  })

  it("retains the same run through same-user credential rotation", async () => {
    const view = setup()
    await submit(view.result)
    config = { ...config, accessToken: `test.${btoa(JSON.stringify({ sub: "7", exp: 9999999999 }))}.signature` }
    act(() => { for (const handlers of [...boundary.watchers]) handlers.tldwConfig?.({ newValue: { ...config } }) })
    await submit(view.result, "rotated")
    expect(JSON.parse(ratingCalls()[1][1].body).review_session_id).toBe(77)
    expect(endCalls()).toHaveLength(0)
  })

  it("binds cookie-session mutations to independently verified owner without browser bearer credentials", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
    config = { serverUrl: window.location.origin, authMode: "multi-user", authSource: "cookie-session" }
    const view = setup()
    await submit(view.result)
    await act(async () => { await view.result.current.complete() })
    for (const [, init] of [...ratingCalls(), ...endCalls()]) {
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
      expect(new Headers(init.headers).get("Authorization")).toBeNull()
      expect(init.credentials ?? "same-origin").toBe("same-origin")
    }
    expect(ratingCalls()).toHaveLength(1)
    expect(endCalls()).toHaveLength(1)
  })

  it("does not let a delayed end clear a replacement after scope A→B→A", async () => {
    const view = setup()
    await submit(view.result)
    const pending = deferred<Response>(); const base = boundary.fetch.getMockImplementation()!
    let intercepted = false
    boundary.fetch.mockImplementation((url, init) => {
      if (!intercepted && String(url).endsWith("/review-sessions/end")) { intercepted = true; return pending.promise }
      return base(url, init)
    })
    view.rerender({ context: { review_mode: "cram", deck_id: 1, tag_filter: "bio" }, enabled: true })
    await waitFor(() => expect(intercepted).toBe(true))
    view.rerender({ context: globalContext, enabled: true })
    let replacement!: Promise<unknown>
    act(() => { replacement = view.result.current.submit({ cardUuid: "replacement", rating: 3 }) })
    await act(async () => {})
    // The server can reuse an active same-context session. Wait for its old end before a new rating.
    expect(ratingCalls()).toHaveLength(1)
    sessionId = 88
    await act(async () => { pending.resolve(response({ id: 77, status: "completed" })); await replacement })
    expect(view.result.current.activeSessionId).toBe(88)
    expect(endCalls()).toHaveLength(1)
  })

  it("ignores a late failed rating after replacement instead of installing its error", async () => {
    const pending = deferred<Response>(); const base = boundary.fetch.getMockImplementation()!
    let intercepted = false
    boundary.fetch.mockImplementation((url, init) => {
      if (!intercepted && String(url).endsWith("/flashcards/review")) { intercepted = true; return pending.promise }
      return base(url, init)
    })
    const view = setup()
    let old!: Promise<unknown>
    act(() => { old = view.result.current.submit({ cardUuid: "old", rating: 3 }) })
    await waitFor(() => expect(intercepted).toBe(true))
    act(() => switchAccount(8))
    await submit(view.result, "bob")
    await act(async () => { pending.resolve(response({ detail: "Old failure" }, 500)); expect(await old).toBeNull() })
    expect(view.result.current.activeSessionId).toBe(77)
    expect(closeError).not.toHaveBeenCalled()
  })

  it("invalidates a pending owner lookup on a storage-only account transition", async () => {
    const pending = deferred<Response>(); const base = boundary.fetch.getMockImplementation()!
    let intercepted = false
    boundary.fetch.mockImplementation((url, init) => {
      if (!intercepted && String(url).endsWith("/auth/me")) { intercepted = true; return pending.promise }
      return base(url, init)
    })
    const view = setup()
    let old!: Promise<unknown>
    act(() => { old = view.result.current.submit({ cardUuid: "old", rating: 3 }) })
    await waitFor(() => expect(intercepted).toBe(true))
    act(() => {
      userId = 8; config = { ...config, accessToken: token(8) }
      for (const handlers of [...boundary.watchers]) handlers.tldwConfig?.({ newValue: { ...config } })
    })
    expect(view.result.current.authorityRevision).toBeGreaterThan(0)
    await act(async () => { pending.resolve(response({ id: 7, is_active: true })); expect(await old).toBeNull() })
    expect(ratingCalls()).toHaveLength(0)
  })

  it("supports single-user key review and end with the same captured credential scope", async () => {
    config = { serverUrl: window.location.origin, authMode: "single-user", apiKey: "synthetic-study-key",
      credentialSource: "manual", apiKeyPersistence: "device", apiKeyServerOrigin: window.location.origin }
    const view = setup()
    await submit(view.result)
    await act(async () => { await view.result.current.complete() })
    expect(boundary.fetch.mock.calls.some(([url]) => String(url).endsWith("/auth/me"))).toBe(false)
    for (const [, init] of [...ratingCalls(), ...endCalls()]) {
      expect(new Headers(init.headers).get("X-API-KEY")).toBe("synthetic-study-key")
    }
    expect(ratingCalls()).toHaveLength(1)
    expect(endCalls()).toHaveLength(1)
  })

  it("does not replay a failed rating write and retains the acknowledged ID for an explicit retry", async () => {
    const view = setup()
    await submit(view.result)
    const base = boundary.fetch.getMockImplementation()!
    let failed = false
    boundary.fetch.mockImplementation((url, init) => {
      if (!failed && String(url).endsWith("/flashcards/review")) { failed = true; return Promise.reject(new TypeError("Network failed")) }
      return base(url, init)
    })
    await act(async () => { await expect(view.result.current.submit({ cardUuid: "next", rating: 3 })).rejects.toBeDefined() })
    expect(ratingCalls()).toHaveLength(2)
    expect(view.result.current.activeSessionId).toBe(77)
    await submit(view.result, "next")
    expect(JSON.parse(ratingCalls()[2][1].body).review_session_id).toBe(77)
  })

  it.each([
    ["/api/v1/flashcards/review", "POST", true],
    ["/api/v1/flashcards/review-sessions/end", "POST", true],
    ["/api/v1/flashcards/review", "DELETE", false],
    ["/api/v1/flashcards/review-sessions/end", "GET", false],
    ["/api/v1/flashcards/review-sessions/77", "POST", false],
    ["/api/v1/flashcards/review/", "POST", false],
    ["/api/v1/flashcards/../review", "POST", false]
  ])("retains the exact scoped method/path contract for %s %s", (path, method, allowed) => {
    expect(isServicePromptRequestPath(path, method)).toBe(allowed)
  })

  it("preserves the fixed quickstart single-user cookie scope without inventing an owner ID", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    config = { serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" }
    boundary.get.mockImplementation(async key => key === "tldwConfig" || key === "tldwCookieSessionConfig" ? { ...config } : null)
    const view = setup()
    await submit(view.result)
    await act(async () => { await view.result.current.complete() })
    for (const [, init] of [...ratingCalls(), ...endCalls()]) {
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBeNull()
      expect(new Headers(init.headers).get("Authorization")).toBeNull()
      expect(init.credentials ?? "same-origin").toBe("same-origin")
    }
    expect(ratingCalls()).toHaveLength(1)
    expect(endCalls()).toHaveLength(1)
  })

  it.each(["rating", "end"])("aborts a pending quickstart cookie %s on session-config invalidation", async operation => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    config = { serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" }
    boundary.get.mockImplementation(async key => key === "tldwConfig" || key === "tldwCookieSessionConfig" ? { ...config } : null)
    const view = setup()
    if (operation === "end") await submit(view.result)
    const pending = deferred<Response>(); const base = boundary.fetch.getMockImplementation()!
    const path = operation === "end" ? "/review-sessions/end" : "/flashcards/review"
    let intercepted = false
    let requestSignal: AbortSignal | undefined
    boundary.fetch.mockImplementation((url, init) => {
      if (!intercepted && String(url).endsWith(path)) {
        intercepted = true; requestSignal = init.signal; return pending.promise
      }
      return base(url, init)
    })
    let old!: Promise<unknown>
    act(() => { old = operation === "end" ? view.result.current.complete() : view.result.current.submit({ cardUuid: "old", rating: 3 }) })
    await waitFor(() => expect(intercepted).toBe(true))
    act(() => { for (const handlers of [...boundary.watchers]) handlers.tldwCookieSessionConfig?.({ newValue: null }) })
    expect(requestSignal?.aborted).toBe(true)
    await act(async () => { pending.resolve(response({ review_session_id: 77, id: 77 })); expect(await old).toBeNull() })
    expect(view.result.current.activeSessionId).toBeNull()
    expect(endCalls()).toHaveLength(operation === "end" ? 1 : 0)
  })

  it("closes the replacement run on scope change after an earlier manual End", async () => {
    const view = setup()
    await submit(view.result)
    await act(async () => { await view.result.current.complete() })
    sessionId = 88
    await submit(view.result, "next-run")
    view.rerender({ context: { ...globalContext, deck_id: 1 }, enabled: true })
    await waitFor(() => expect(endCalls()).toHaveLength(2))
    expect(JSON.parse(endCalls()[1][1].body)).toEqual({ review_session_id: 88 })
  })

  it("allows an explicit retry after a temporary owner-lookup failure", async () => {
    const base = boundary.fetch.getMockImplementation()!
    let recovered = false
    boundary.fetch.mockImplementation((url, init) => !recovered && String(url).endsWith("/auth/me")
      ? Promise.resolve(response({ detail: "Temporarily unavailable" }, 503)) : base(url, init))
    const view = setup()
    await act(async () => { await expect(view.result.current.submit({ cardUuid: "card", rating: 3 })).rejects.toBeDefined() })
    expect(ratingCalls()).toHaveLength(0)
    recovered = true
    await submit(view.result)
    expect(ratingCalls()).toHaveLength(1)
    expect(view.result.current.activeSessionId).toBe(77)
  })

  it("requires an explicit new session after the retained session expires, without replaying the rating", async () => {
    const view = setup()
    await submit(view.result)
    const base = boundary.fetch.getMockImplementation()!
    boundary.fetch.mockImplementation((url, init) => String(url).endsWith("/flashcards/review")
      ? Promise.resolve(response({ detail: "Flashcard review session is not active" }, 404)) : base(url, init))
    await act(async () => { await expect(view.result.current.submit({ cardUuid: "next", rating: 3 })).rejects.toMatchObject({ status: 404 }) })
    expect(view.result.current.canRestart).toBe(true)
    expect(view.result.current.activeSessionId).toBe(77)
    await act(async () => { expect(await view.result.current.submit({ cardUuid: "next", rating: 3 })).toBeNull() })
    expect(ratingCalls()).toHaveLength(2)
    act(() => { expect(view.result.current.restart()).toBe(true) })
    expect(ratingCalls()).toHaveLength(2)
    expect(endCalls()).toHaveLength(0)
    boundary.fetch.mockImplementation(base)
    sessionId = 88
    await submit(view.result, "next")
    expect(JSON.parse(ratingCalls()[2][1].body).review_session_id).toBeUndefined()
    expect(view.result.current.activeSessionId).toBe(88)
  })

  it("keeps a valid retained session when only the reviewed card is missing", async () => {
    const view = setup()
    await submit(view.result)
    const base = boundary.fetch.getMockImplementation()!
    boundary.fetch.mockImplementation((url, init) => String(url).endsWith("/flashcards/review")
      ? Promise.resolve(response({ detail: "Flashcard not found" }, 404)) : base(url, init))
    await act(async () => { await expect(view.result.current.submit({ cardUuid: "missing", rating: 3 })).rejects.toMatchObject({ status: 404 }) })
    expect(view.result.current.canRestart).toBe(false)
    expect(view.result.current.restart()).toBe(false)
    boundary.fetch.mockImplementation(base)
    await submit(view.result, "available")
    expect(JSON.parse(ratingCalls()[2][1].body).review_session_id).toBe(77)
  })

  it("performs no rating or end writes for practice-only Cram", async () => {
    const view = setup()
    view.rerender({ context: { review_mode: "cram", deck_id: null, tag_filter: null }, enabled: false })
    await submit(view.result)
    expect(ratingCalls()).toHaveLength(0)
    expect(endCalls()).toHaveLength(0)
  })

  it.each(["local rating", "local end", "native rating", "native end", "extension rating", "extension end"])(
    "retires canonical-invalidated work delivered through %s without changing raw credentials", async scenario => {
      const [delivery, operation] = scenario.split(" ")
      const { storage, records } = canonicalCredentials()
      const original = { ...config }
      const view = setup()
      const held = await holdOperation(view, operation)
      const invalidateCache = vi.spyOn(queryClient, "invalidateQueries")
      invalidateCache.mockClear()
      await act(async () => {
        // The helper executes in another context for native/extension delivery.
        const event = delivery !== "local" ? vi.spyOn(window, "dispatchEvent").mockReturnValue(true) : null
        expect(await invalidateRefreshSessionIfCurrent(storage, original)).toBe(true)
        event?.mockRestore()
        const marker = refreshSessionInvalidationKey(original)!
        if (delivery === "native") window.dispatchEvent(new StorageEvent("storage", { key: marker, newValue: "true" }))
        if (delivery === "extension") for (const listener of boundary.nativeListeners) listener({ [marker]: { newValue: true } }, "local")
      })
      const wasAborted = held.signal.aborted
      await act(async () => { held.pending.resolve(response({ review_session_id: 77, id: 77 })); await held.result })
      expect(wasAborted).toBe(true)
      expect(view.result.current.activeSessionId).toBeNull()
      expect(invalidateCache).not.toHaveBeenCalled()
      expect(records.get("tldwConfig")).toEqual(original)
    }
  )

  it("ignores an old session marker after a newer same-user login", async () => {
    const { storage, records } = canonicalCredentials()
    const old = { ...config }
    expect(await invalidateRefreshSessionIfCurrent(storage, old)).toBe(true)
    config = { ...config, accessToken: `test.${btoa(JSON.stringify({ sub: "7", login: 2 }))}.signature`, refreshToken: "new-login-refresh" }
    records.set("tldwConfig", { ...config })
    const view = setup()
    const held = await holdOperation(view, "rating")
    await act(async () => window.dispatchEvent(new StorageEvent("storage", { key: refreshSessionInvalidationKey(old)! })))
    expect(held.signal.aborted).toBe(false)
    await act(async () => { held.pending.resolve(response({ review_session_id: 77 })); expect(await held.result).not.toBeNull() })
    expect(view.result.current.activeSessionId).toBe(77)
  })

  it("keeps pending work and its retained ID across a valid canonical rotation and stale expiry hint", async () => {
    const { storage, records } = canonicalCredentials()
    const original = { ...config }
    const view = setup()
    const held = await holdOperation(view, "rating")
    const rotated = { accessToken: `test.${btoa(JSON.stringify({ sub: "7", rotation: 1 }))}.signature`, refreshToken: "rotated-refresh" }
    await act(async () => {
      expect(await storeRefreshRotationIfCurrent(storage, original, original.refreshToken!, rotated)).toBe(true)
      for (const watcher of [...boundary.watchers]) watcher[REFRESH_ROTATION_KEY]?.({ newValue: records.get(REFRESH_ROTATION_KEY) })
      expect(await invalidateRefreshSessionIfCurrent(storage, original)).toBe(false)
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { refreshSessionInvalidated: true } }))
    })
    expect(held.signal.aborted).toBe(false)
    await act(async () => { held.pending.resolve(response({ review_session_id: 77 })); await held.result })
    await submit(view.result, "rotated-card")
    expect(JSON.parse(ratingCalls()[1][1].body).review_session_id).toBe(77)
    expect(new Headers(ratingCalls()[1][1].headers).get("Authorization")).toBe(`Bearer ${rotated.accessToken}`)
    expect(records.get("tldwConfig")).toEqual(original)
  })

  it.each([
    ["changed subject", "rating", token(8)],
    ["opaque token", "rating", "opaque-rotated-access"],
    ["changed subject", "end", token(8)],
    ["opaque token", "end", "opaque-rotated-access"]
  ])("retires a pending %s %s when canonical rotation cannot match the verified run owner", async (_kind, operation, accessToken) => {
    const { storage, records } = canonicalCredentials()
    const original = { ...config }
    const view = setup()
    const held = await holdOperation(view, operation)
    const invalidateCache = vi.spyOn(queryClient, "invalidateQueries")
    invalidateCache.mockClear()
    await act(async () => {
      expect(await storeRefreshRotationIfCurrent(storage, original, original.refreshToken!, {
        accessToken, refreshToken: "different-authority-refresh"
      })).toBe(true)
      for (const watcher of [...boundary.watchers]) watcher[REFRESH_ROTATION_KEY]?.({ newValue: records.get(REFRESH_ROTATION_KEY) })
    })
    expect((await tldwClient.getConfig())?.accessToken).toBe(accessToken)
    expect(records.get("tldwConfig")).toEqual(original)
    const wasAborted = held.signal.aborted
    let accepted: unknown
    await act(async () => {
      held.pending.resolve(response({ review_session_id: 77, id: 77 }))
      accepted = await held.result
    })
    expect(wasAborted).toBe(true)
    expect(accepted).toBeNull()
    expect(view.result.current.activeSessionId).toBeNull()
    expect(invalidateCache).not.toHaveBeenCalled()
  })

  it("discards a delayed canonical invalidation check after login A→B→A", async () => {
    const { storage, records } = canonicalCredentials()
    const original = { ...config }
    const view = setup()
    const held = await holdOperation(view, "rating")
    const checked = deferred<TldwConfig>()
    const realEnsure = tldwClient.ensureConfigForRequest.bind(tldwClient)
    const ensure = vi.spyOn(tldwClient, "ensureConfigForRequest").mockImplementation(realEnsure).mockImplementationOnce(() => checked.promise)
    await act(async () => { expect(await invalidateRefreshSessionIfCurrent(storage, original)).toBe(true) })
    await waitFor(() => expect(ensure).toHaveBeenCalled())
    for (const id of [8, 7]) {
      act(() => {
        userId = id
        config = { ...original, accessToken: `test.${btoa(JSON.stringify({ sub: String(id), login: 2 }))}.signature`, refreshToken: `new-refresh-${id}` }
        records.set("tldwConfig", { ...config })
        for (const watcher of [...boundary.watchers]) watcher.tldwConfig?.({ newValue: { ...config } })
        window.dispatchEvent(new CustomEvent("tldw:auth-credentials-changed"))
      })
      await submit(view.result, `new-${id}`)
    }
    await act(async () => {
      checked.resolve({ ...original, accessToken: undefined, refreshToken: undefined })
      held.pending.resolve(response({ review_session_id: 90 }))
      expect(await held.result).toBeNull()
    })
    expect(view.result.current.activeSessionId).toBe(77)
    expect(endCalls()).toHaveLength(0)
  })

  it("discards an older masked-config check once a valid rotation has been verified for the same run", async () => {
    const { storage, records } = canonicalCredentials()
    const original = { ...config }
    const view = setup()
    const held = await holdOperation(view, "rating")
    const checked = deferred<TldwConfig>()
    const realEnsure = tldwClient.ensureConfigForRequest.bind(tldwClient)
    vi.spyOn(tldwClient, "ensureConfigForRequest").mockImplementation(realEnsure).mockImplementationOnce(() => checked.promise)
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { refreshSessionInvalidated: true } })))
    await act(async () => {
      expect(await storeRefreshRotationIfCurrent(storage, original, original.refreshToken!, {
        accessToken: `test.${btoa(JSON.stringify({ sub: "7", rotation: 2 }))}.signature`, refreshToken: "latest-refresh"
      })).toBe(true)
      for (const watcher of [...boundary.watchers]) watcher[REFRESH_ROTATION_KEY]?.({ newValue: records.get(REFRESH_ROTATION_KEY) })
    })
    await act(async () => checked.resolve({ ...original, accessToken: undefined, refreshToken: undefined }))
    expect(held.signal.aborted).toBe(false)
    await act(async () => { held.pending.resolve(response({ review_session_id: 77 })); expect(await held.result).not.toBeNull() })
    expect(view.result.current.activeSessionId).toBe(77)
  })

  it.each(["hosted", "quickstart"])("keeps the verified %s cookie run when a stale bearer invalidation hint arrives", async deployment => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", deployment)
    config = { serverUrl: window.location.origin, authMode: deployment === "hosted" ? "multi-user" : "single-user", authSource: "cookie-session" }
    vi.mocked(tldwClient.ensureConfigForRequest).mockRestore()
    const view = setup()
    const held = await holdOperation(view, "rating")
    await act(async () => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { refreshSessionInvalidated: true } })))
    expect(held.signal.aborted).toBe(false)
    await act(async () => { held.pending.resolve(response({ review_session_id: 77 })); expect(await held.result).not.toBeNull() })
    await act(async () => { await view.result.current.complete() })
    expect(JSON.parse(endCalls()[0][1].body).review_session_id).toBe(77)
  })
})
