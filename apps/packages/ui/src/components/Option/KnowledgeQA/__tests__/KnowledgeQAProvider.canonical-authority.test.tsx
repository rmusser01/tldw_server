import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"
import { InlineRecentSessions } from "../empty/InlineRecentSessions"
import { applyRefreshRotation, hasInvalidatedRefreshSession, invalidateRefreshSessionIfCurrent } from "@/services/tldw/single-user-credential"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

const boundary = vi.hoisted(() => ({
  storage: {} as Record<string, unknown>, connected: true, user: 1,
  config: vi.fn(), ensureConfig: vi.fn(), currentUser: vi.fn(), fetch: vi.fn(),
}))
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: (key: string, fallback: unknown) => [boundary.storage[key] ?? fallback] }))
vi.mock("@/store/connection", () => ({
  useConnectionStore: (select: (store: { state: { isConnected: boolean; mode: string } }) => unknown) => select({ state: { isConnected: boundary.connected, mode: "normal" } }),
}))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({ get: vi.fn(), watch: vi.fn(), unwatch: vi.fn() }) }))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: (...args: unknown[]) => boundary.currentUser(...args) } }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: vi.fn().mockResolvedValue(undefined),
  getConfig: (...args: unknown[]) => boundary.config(...args),
  ensureConfigForRequest: (...args: unknown[]) => boundary.ensureConfig(...args),
  fetchWithAuth: (...args: unknown[]) => boundary.fetch(...args),
  ragSourceHealth: vi.fn().mockResolvedValue({}),
} }))
vi.mock("@/hooks/useAntdMessage", () => ({ useAntdMessage: () => ({ open: vi.fn() }) }))
vi.mock("@/utils/knowledge-qa-search-metrics", () => ({ trackKnowledgeQaSearchMetric: vi.fn() }))

let current: ReturnType<typeof useKnowledgeQA>
function Recent() {
  const context = useKnowledgeQA()
  React.useLayoutEffect(() => { current = context }, [context])
  return <><InlineRecentSessions items={context.searchHistory} onRestore={context.restoreFromHistory} /><output aria-label="Draft">{context.query}</output></>
}
const view = () => <KnowledgeQAProvider><Recent /></KnowledgeQAProvider>
const token = (user: number, revision: number) => `test.${btoa(JSON.stringify({ sub: String(user), revision }))}.signature`
const config = (user = 1): TldwConfig => ({ serverUrl: "https://canonical-qa.test", authMode: "multi-user", accessToken: token(user, 1), refreshToken: `refresh-${user}-source` })
const rotation = (revision: number) => ({ version: 1, serverUrl: config().serverUrl, authMode: "multi-user", sourceAccessToken: token(1, 1), sourceRefreshToken: "refresh-1-source", accessToken: token(1, revision), refreshToken: `refresh-1-${revision}` })
const currentConfig = () => applyRefreshRotation(boundary.storage.tldwConfig as TldwConfig, boundary.storage.tldwRefreshRotation)
const credentialStorage = {
  get: async <T,>(key: string) => boundary.storage[key] as T | undefined,
  set: async <T,>(key: string, value: T) => { boundary.storage[key] = value },
  remove: async (key: string) => { delete boundary.storage[key] },
}
const effectiveConfig = async () => {
  const value = currentConfig()
  return await hasInvalidatedRefreshSession(credentialStorage, value)
    ? { ...value, accessToken: undefined, refreshToken: undefined } : value
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

describe("QA with real canonical configuration and verified scope lease", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    boundary.connected = true
    boundary.user = 1
    boundary.storage = { tldwConfig: config() }
    boundary.config.mockImplementation(async () => currentConfig())
    boundary.ensureConfig.mockImplementation(async (required = true) => {
      const value = await effectiveConfig()
      if (required && value.authMode === "multi-user" && value.authSource !== "cookie-session" && !value.accessToken) {
        throw Object.assign(new Error("Authentication required"), { status: 401 })
      }
      return value
    })
    boundary.currentUser.mockImplementation(async () => ({ id: boundary.user, is_active: true }))
    boundary.fetch.mockImplementation(async () => ({ ok: true, json: async () => [{
      id: `chat-${boundary.user}`, title: `Owner ${boundary.user} question`, keywords: ["__knowledge_QA__"],
      message_count: 2, last_modified: "2026-09-15T11:00:00Z",
    }] }))
  })

  it("keeps a draft and Recent rows during repeated valid token rotation while canonical reads are pending", async () => {
    const rendered = render(view())
    await screen.findByRole("button", { name: /Owner 1 question/ })
    act(() => current.setQuery("Own unfinished question"))
    for (const revision of [2, 3]) {
      const pending = deferred<TldwConfig>()
      boundary.config.mockReturnValueOnce(pending.promise)
      boundary.storage = { ...boundary.storage, tldwRefreshRotation: rotation(revision) }
      act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } })))
      rendered.rerender(view())
      expect(current.query).toBe("Own unfinished question")
      expect(screen.getByRole("button", { name: /Owner 1 question/ })).toBeInTheDocument()
      await act(async () => { pending.resolve(currentConfig()); await pending.promise })
      expect(current.query).toBe("Own unfinished question")
    }
  })

  it("preserves the valid rotated owner's draft after a stale expiry hint", async () => {
    const rendered = render(view())
    await screen.findByRole("button", { name: /Owner 1 question/ })
    act(() => current.setQuery("Owner draft before a stale expiry hint"))
    boundary.storage = { ...boundary.storage, tldwRefreshRotation: rotation(2) }
    rendered.rerender(view())
    await act(async () => { window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { refreshSessionInvalidated: true } })) })
    await screen.findByRole("button", { name: /Owner 1 question/ })
    expect(currentConfig().accessToken).toBe(token(1, 2))
    expect(current.query).toBe("Owner draft before a stale expiry hint")
  })

  it("ignores an expired check that resolves after a valid same-owner canonical rotation", async () => {
    const rendered = render(view())
    await screen.findByRole("button", { name: /Owner 1 question/ })
    act(() => current.setQuery("Draft retained by the newer rotation"))
    const pending = deferred<TldwConfig>()
    boundary.ensureConfig.mockReturnValueOnce(pending.promise)
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { refreshSessionInvalidated: true } })))
    boundary.storage = { ...boundary.storage, tldwRefreshRotation: rotation(2) }
    rendered.rerender(view())
    await act(async () => { await Promise.resolve() })
    await act(async () => { pending.resolve({ ...config(), accessToken: undefined, refreshToken: undefined }); await pending.promise })
    expect(current.query).toBe("Draft retained by the newer rotation")
    expect(screen.getByRole("button", { name: /Owner 1 question/ })).toBeInTheDocument()
  })

  it("rejects an older expiry check after a newer hint verifies the same owner", async () => {
    render(view())
    await screen.findByRole("button", { name: /Owner 1 question/ })
    act(() => current.setQuery("Draft retained by the newer check"))
    const pending = deferred<TldwConfig>()
    boundary.ensureConfig.mockReturnValueOnce(pending.promise)
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { refreshSessionInvalidated: true } })))
    await act(async () => { window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { refreshSessionInvalidated: true } })) })
    await act(async () => { pending.resolve({ ...config(), accessToken: undefined, refreshToken: undefined }); await pending.promise })
    expect(current.query).toBe("Draft retained by the newer check")
  })

  it("clears a currently invalidated session and rejects an older successful check", async () => {
    render(view())
    await screen.findByRole("button", { name: /Owner 1 question/ })
    act(() => current.setQuery("Expired session draft"))
    const pending = deferred<TldwConfig>()
    boundary.ensureConfig.mockReturnValueOnce(pending.promise)
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { refreshSessionInvalidated: true } })))
    await act(async () => { expect(await invalidateRefreshSessionIfCurrent(credentialStorage, currentConfig())).toBe(true) })
    await waitFor(() => expect(current.query).toBe(""))
    await act(async () => { pending.resolve(config()); await pending.promise })
    expect(current.searchHistory).toEqual([])
    expect(screen.queryByRole("button", { name: /Owner 1 question/ })).not.toBeInTheDocument()
  })

  it("does not let a delayed prior-owner expiry check clear the next owner's draft", async () => {
    const rendered = render(view())
    await screen.findByRole("button", { name: /Owner 1 question/ })
    const pending = deferred<TldwConfig>()
    boundary.ensureConfig.mockReturnValueOnce(pending.promise)
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { refreshSessionInvalidated: true } })))
    boundary.user = 2
    boundary.storage = { tldwConfig: config(2) }
    rendered.rerender(view())
    await screen.findByRole("button", { name: /Owner 2 question/ })
    act(() => current.setQuery("Owner 2 private draft"))
    await act(async () => { pending.resolve({ ...config(), accessToken: undefined, refreshToken: undefined }); await pending.promise })
    expect(current.query).toBe("Owner 2 private draft")
    expect(screen.queryByRole("button", { name: /Owner 1 question/ })).not.toBeInTheDocument()
  })

  it("masks an actual account change and ignores the older canonical read through A to B to A", async () => {
    const rendered = render(view())
    await screen.findByRole("button", { name: /Owner 1 question/ })
    act(() => current.setQuery("Old owner draft"))
    const pending = deferred<TldwConfig>()
    boundary.config.mockReturnValueOnce(pending.promise)
    boundary.user = 2
    boundary.storage = { tldwConfig: config(2) }
    rendered.rerender(view())
    expect(screen.queryByRole("button", { name: /Owner 1 question/ })).not.toBeInTheDocument()
    boundary.user = 1
    boundary.storage = { tldwConfig: config() }
    rendered.rerender(view())
    await screen.findByRole("button", { name: /Owner 1 question/ })
    await act(async () => { pending.resolve(config(2)); await pending.promise })
    expect(current.query).toBe("")
    expect(screen.queryByRole("button", { name: /Owner 2 question/ })).not.toBeInTheDocument()
  })

  it("does not trust a malformed rotation marker during canonical rehydration", async () => {
    const rendered = render(view())
    await screen.findByRole("button", { name: /Owner 1 question/ })
    act(() => current.setQuery("Private before unknown authority"))
    const pending = deferred<TldwConfig>()
    boundary.config.mockReturnValueOnce(pending.promise)
    boundary.storage = { ...boundary.storage, tldwRefreshRotation: token(1, 2) }
    rendered.rerender(view())
    expect(current.query).toBe("")
    expect(current.searchHistory).toEqual([])
    await act(async () => { pending.resolve(currentConfig()); await pending.promise })
  })

  it("does not render retained history before initial identity verification or after logout", async () => {
    const pending = deferred<{ id: number; is_active: boolean }>()
    boundary.currentUser.mockReturnValueOnce(pending.promise)
    const rendered = render(view())
    await waitFor(() => expect(boundary.currentUser).toHaveBeenCalled())
    expect(current.searchHistory).toEqual([])
    boundary.connected = false
    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })))
    rendered.rerender(view())
    await act(async () => { pending.resolve({ id: 1, is_active: true }); await pending.promise })
    expect(current.searchHistory).toEqual([])
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
})
