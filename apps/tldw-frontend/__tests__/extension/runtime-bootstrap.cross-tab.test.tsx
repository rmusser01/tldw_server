import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { useKnowledgeQA } from "@/components/Option/KnowledgeQA/KnowledgeQAProvider"

const notifications = vi.hoisted(() => ({ open: vi.fn() }))
vi.mock("@/hooks/useAntdMessage", () => ({ useAntdMessage: () => notifications }))
vi.mock("@/utils/knowledge-qa-search-metrics", () => ({ trackKnowledgeQaSearchMetric: vi.fn() }))

const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
const originalApiUrl = process.env.NEXT_PUBLIC_API_URL
const originalNavigatorLocks = Object.getOwnPropertyDescriptor(window.navigator, "locks")
let qa: ReturnType<typeof useKnowledgeQA>
let releaseSearch: (response: Response) => void
let queryStarted = false
let profileOk = true
let profileGate: Promise<void> | null = null
let profileProbeStarted = false
let peerStorageBridgeInstalled = false
const answer = "Mira Chen directs Rowan Observatory in Northbridge. Tours run Saturday at 10 [1]."
const result = {
  documents: [{ id: "source-1", content: answer, score: 0.95, source: "media_db", source_id: 1,
    metadata: { source_type: "media_db", source_id: 1, title: "Rowan Observatory" } }],
  generated_answer: answer, expanded_queries: [], metadata: {}, errors: [],
}
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { "Content-Type": "application/json" },
})

beforeEach(() => {
  vi.resetModules()
  vi.stubGlobal("localStorage", window.localStorage)
  localStorage.clear()
  sessionStorage.clear()
  process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
  delete process.env.NEXT_PUBLIC_API_URL
  localStorage.setItem("ff_knowledgeQaStreaming", "false")
  queryStarted = false
  profileOk = true
  profileGate = null
  profileProbeStarted = false
  peerStorageBridgeInstalled = false
  const lockTails = new Map<string, Promise<void>>()
  vi.stubGlobal("navigator", window.navigator)
  Object.defineProperty(window.navigator, "locks", {
    configurable: true,
    value: { request: async (name: string, work: () => unknown) => {
      const pending = (lockTails.get(name) || Promise.resolve()).then(work)
      lockTails.set(name, pending.then(() => undefined, () => undefined))
      return await pending
    } },
  })
  const pendingSearch = new Promise<Response>((resolve) => { releaseSearch = resolve })
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(input), window.location.origin).pathname
    const headers = new Headers(init?.headers)
    // The real cookie transport must not introduce an API key or Bearer header.
    expect(headers.has("X-API-KEY") || headers.has("Authorization")).toBe(false)
    if (path === "/api/_tldw-webui/runtime-config") return json({ runtimeAuth: {
      available: true, authMode: "single-user", transport: "cookie-session",
    } })
    if (path === "/api/_tldw-webui/session") return json({ ok: true })
    if (path === "/api/v1/users/me/profile") {
      profileProbeStarted = true
      const gate = profileGate
      const status = profileOk ? 200 : 401
      await gate
      return json({ id: 1, username: "single_user", is_active: true }, status)
    }
    if (path === "/api/v1/auth/sessions") return json({ sessions: [] })
    if (path === "/api/v1/health/live" || path === "/api/v1/rag/health") return json({ status: "ok" })
    if (path === "/api/v1/media") return json({ data: [], pagination: { total: 1 } })
    if (path === "/api/v1/notes/") return json({ notes: [], total: 0 })
    if (path === "/api/v1/rag/source-health") return json({ sources: [] })
    if (path === "/openapi.json") return json({ paths: {} })
    if (path === "/api/v1/characters/search") return json([{ id: 1, name: "Helpful AI Assistant" }])
    if (path === "/api/v1/chat/conversations") return json([])
    if (path === "/api/v1/chats/" || path === "/api/v1/chat/conversations/thread-qa") return json({ id: "thread-qa", version: 1, keywords: [] })
    if (path === "/api/v1/chats/thread-qa/messages") {
      const body = JSON.parse(String(init?.body))
      return json({ id: body.role === "user" ? "user-1" : "assistant-1", ...body })
    }
    if (path === "/api/v1/chat/messages/assistant-1/rag-context") return json({ ok: true })
    if (path === "/api/v1/rag/search") {
      const body = JSON.parse(String(init?.body))
      expect(body.include_media_ids).toEqual([1])
      expect(body.enable_web_fallback).toBe(false)
      queryStarted = true
      return pendingSearch
    }
    throw new Error(`Unexpected fixture request: ${init?.method || "GET"} ${path}`)
  }))
})

afterEach(() => {
  releaseSearch(json(result))
  vi.useRealTimers()
  if (originalNavigatorLocks) Object.defineProperty(window.navigator, "locks", originalNavigatorLocks)
  else Reflect.deleteProperty(window.navigator, "locks")
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  if (originalDeploymentMode === undefined) delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  else process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
  if (originalApiUrl === undefined) delete process.env.NEXT_PUBLIC_API_URL
  else process.env.NEXT_PUBLIC_API_URL = originalApiUrl
})

async function mountVerifiedQa() {
  // First tab completes the actual bootstrap before mounting its private page.
  const initialBootstrap = await import("@web/extension/shims/runtime-bootstrap")
  await initialBootstrap.runtimeBootstrapReady
  const { useConnectionStore } = await import("@/store/connection")
  await useConnectionStore.getState().checkOnce({ force: true })
  expect(useConnectionStore.getState().state.isConnected).toBe(true)
  const { KnowledgeQAProvider, useKnowledgeQA } = await import("@/components/Option/KnowledgeQA/KnowledgeQAProvider")
  function Probe() {
    const context = useKnowledgeQA()
    React.useLayoutEffect(() => { qa = context }, [context])
    return <><output aria-label="Question">{context.query}</output><output aria-label="Answer">{context.answer}</output></>
  }
  render(<MemoryRouter initialEntries={["/knowledge?media_ids=1"]}><KnowledgeQAProvider><Probe /></KnowledgeQAProvider></MemoryRouter>)
  await waitFor(() => expect(qa.settings.include_media_ids).toEqual([1]))
  return useConnectionStore
}

async function bootstrapPeerTab(beforeVerified?: () => Promise<void>) {
  // Each browser document has its own module state. Keep the first tab's actual
  // store/provider, and load a fresh bootstrap/module graph for the peer tab.
  vi.resetModules()
  if (!peerStorageBridgeInstalled) {
    peerStorageBridgeInstalled = true
    const backend = window.localStorage
    const prototype = Object.getPrototypeOf(backend) as Storage
    const remove = prototype.removeItem
    const set = prototype.setItem
    const deliver = (key: string, oldValue: string | null) => {
      const newValue = backend.getItem(key)
      if (oldValue === newValue) return
      window.dispatchEvent(new StorageEvent("storage", {
        key, oldValue, newValue,
        url: `${window.location.origin}/notes`,
      }))
    }
    // jsdom has one document: deliver precisely the native event another browser
    // tab receives for the peer's real persistent writes, without replacing any
    // auth, config resolver, connection, or QA authority implementation.
    vi.spyOn(prototype, "removeItem").mockImplementation(function (this: Storage, key) {
      const oldValue = this.getItem(key)
      remove.call(this, key)
      if (this === backend) deliver(key, oldValue)
    })
    vi.spyOn(prototype, "setItem").mockImplementation(function (this: Storage, key, value) {
      const oldValue = this.getItem(key)
      set.call(this, key, value)
      if (this === backend) deliver(key, oldValue)
    })
  }
  const peer = await import("@web/extension/shims/runtime-bootstrap")
  if (beforeVerified) await beforeVerified()
  await peer.runtimeBootstrapReady
}

describe("cookie bootstrap across WebUI tabs", () => {
  it.each(["in flight", "completed"])("keeps the first tab's %s QA result during a verified peer bootstrap", async (stage) => {
    const connection = await mountVerifiedQa()
    act(() => qa.setQuery("Who directs Rowan Observatory, where is it, and when are tours?"))
    let search!: Promise<void>
    act(() => { search = qa.search() })
    await waitFor(() => expect(queryStarted).toBe(true))
    if (stage === "completed") {
      await act(async () => { releaseSearch(json(result)); await search })
      expect(screen.getByLabelText("Answer")).toHaveTextContent(answer)
    }
    await act(async () => { await bootstrapPeerTab() })
    expect(connection.getState().state.isConnected).toBe(true)
    expect(screen.getByLabelText("Question")).toHaveTextContent("Who directs Rowan Observatory")
    if (stage === "in flight") {
      expect(qa.isSearching).toBe(true)
      await act(async () => { releaseSearch(json(result)); await search })
    }
    expect(screen.getByLabelText("Answer")).toHaveTextContent(answer)
    expect(qa.results).toHaveLength(1)
    expect(qa.citations).toHaveLength(1)
  })

  it("keeps the peer's canonical cookie config untrusted until its profile probe succeeds", async () => {
    const connection = await mountVerifiedQa()
    act(() => qa.setQuery("First tab private draft"))
    profileProbeStarted = false
    let releaseProfile!: () => void
    profileGate = new Promise<void>((resolve) => { releaseProfile = resolve })
    let pendingConfig!: ReturnType<typeof import("@/services/tldw/TldwApiClient").tldwClient.getConfig>
    await act(async () => {
      await bootstrapPeerTab(async () => {
        try {
          await waitFor(() => expect(profileProbeStarted).toBe(true))
          const { tldwClient } = await import("@/services/tldw/TldwApiClient")
          const { resolveDirectBrowserConfig } = await import("@/services/tldw/direct-browser-config")
          const { createSafeStorage } = await import("@/utils/safe-storage")
          pendingConfig = tldwClient.getConfig()
          expect(await resolveDirectBrowserConfig(createSafeStorage({ area: "local" }))).toBeNull()
          expect(connection.getState().state.isConnected).toBe(true)
          expect(screen.getByLabelText("Question")).toHaveTextContent("First tab private draft")
        } finally {
          releaseProfile()
        }
      })
    })
    expect(await pendingConfig).toMatchObject({ authSource: "cookie-session" })
  })


  it.each(["denied", "timed out"])("retains the newer identical verification after an older peer's profile is %s", async (failure) => {
    const connection = await mountVerifiedQa()
    act(() => qa.setQuery("Who directs Rowan Observatory?"))
    let search!: Promise<void>
    act(() => { search = qa.search() })
    await waitFor(() => expect(queryStarted).toBe(true))
    await act(async () => { releaseSearch(json(result)); await search })
    profileProbeStarted = false
    profileOk = false
    let releaseProfile!: () => void
    profileGate = new Promise<void>((resolve) => { releaseProfile = resolve })
    vi.useFakeTimers()
    let newerMarker: string | null = null
    let pendingOlderConfig!: ReturnType<typeof import("@/services/tldw/TldwApiClient").tldwClient.getConfig>
    try {
      await act(async () => {
        await bootstrapPeerTab(async () => {
          await vi.advanceTimersByTimeAsync(0)
          expect(profileProbeStarted).toBe(true)
          const { tldwClient: olderClient } = await import("@/services/tldw/TldwApiClient")
          const { resolveDirectBrowserConfig } = await import("@/services/tldw/direct-browser-config")
          const { createSafeStorage } = await import("@/utils/safe-storage")
          const olderStorage = createSafeStorage({ area: "local" })
          pendingOlderConfig = olderClient.getConfig()
          await vi.advanceTimersByTimeAsync(0)
          expect(await resolveDirectBrowserConfig(olderStorage)).toBeNull()
          profileOk = true
          profileGate = null
          await bootstrapPeerTab()
          newerMarker = localStorage.getItem("tldwCookieSessionConfig")
          expect(newerMarker).not.toBeNull()
          expect(await resolveDirectBrowserConfig(olderStorage)).toBeNull()
          if (failure === "timed out") await vi.advanceTimersByTimeAsync(8_000)
          else releaseProfile()
        })
      })
    } finally {
      releaseProfile()
      vi.useRealTimers()
    }
    expect(await pendingOlderConfig).toBeNull()
    expect(connection.getState().state.isConnected).toBe(true)
    expect(localStorage.getItem("tldwCookieSessionConfig")).toBe(newerMarker)
    expect(screen.getByLabelText("Answer")).toHaveTextContent(answer)
    expect(qa.results).toHaveLength(1)
    expect(qa.citations).toHaveLength(1)
  })

  it("verifies and publishes a peer cookie config when Web Locks are unavailable", async () => {
    const connection = await mountVerifiedQa()
    act(() => qa.setQuery("First tab private draft"))
    Object.defineProperty(window.navigator, "locks", { configurable: true, value: undefined })
    await act(async () => { await bootstrapPeerTab() })
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    expect(await tldwClient.getConfig()).toMatchObject({ authSource: "cookie-session" })
    expect(connection.getState().state.isConnected).toBe(true)
    expect(screen.getByLabelText("Question")).toHaveTextContent("First tab private draft")
  })

  it("keeps a denied peer locally untrusted without destructive shared cleanup when Web Locks are unavailable", async () => {
    const connection = await mountVerifiedQa()
    act(() => qa.setQuery("First tab private draft"))
    const verifiedMarker = localStorage.getItem("tldwCookieSessionConfig")
    Object.defineProperty(window.navigator, "locks", { configurable: true, value: undefined })
    profileOk = false
    await act(async () => { await bootstrapPeerTab() })
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    expect(await tldwClient.getConfig()).toBeNull()
    expect(connection.getState().state.isConnected).toBe(true)
    expect(localStorage.getItem("tldwCookieSessionConfig")).toBe(verifiedMarker)
    expect(screen.getByLabelText("Question")).toHaveTextContent("First tab private draft")
  })

  it("invalidates the first tab when the peer's authenticated profile probe is denied", async () => {
    const connection = await mountVerifiedQa()
    act(() => qa.setQuery("Private draft before denial"))
    profileOk = false
    await act(async () => { await bootstrapPeerTab() })
    expect(connection.getState().state.isConnected).toBe(false)
    expect(screen.getByLabelText("Question").textContent).toBe("")
    expect(localStorage.getItem("tldwCookieSessionConfig")).toBeNull()
  })
})
