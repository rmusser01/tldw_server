import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

const boundary = vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (options: unknown) => options
  })
  return {
    local: new Map<string, unknown>(),
    session: new Map<string, unknown>(),
    listeners: new Set<(message: unknown, sender: unknown, reply: (value: unknown) => void) => unknown>(),
    sent: [] as { type: string; payload?: Record<string, unknown> }[],
    beforeRequest: null as (() => void) | null,
    fetch: vi.fn(),
    connection: { config: null as TldwConfig | null, loading: false, authorityLoading: false }
  }
})
vi.mock("@/utils/safe-storage", async importOriginal => {
  const { safeStorageSerde } = await importOriginal<typeof import("@/utils/safe-storage")>()
  return {
    safeStorageSerde,
    createSafeStorage: ({ area = "local" } = {}) => {
      const values = area === "session" ? boundary.session : boundary.local
      return {
        get: async (key: string) => values.get(key),
        set: async (key: string, value: unknown) => { values.set(key, value) },
        remove: async (key: string) => { values.delete(key) }
      }
    }
  }
})
vi.mock("@/entries/shared/background-init", () => ({
  MODEL_WARM_ALARM_NAME: "warm",
  initBackground: async () => {}
}))
vi.mock("@/entries/shared/notification-subscription", () => ({
  startNotificationSubscription: async () => {}
}))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => boundary.connection
}))
vi.mock("wxt/browser", () => {
  const event = () => ({ addListener: vi.fn() })
  return {
    browser: {
      runtime: {
        id: "extension-fixture",
        getURL: (path: string) => `chrome-extension://extension-fixture${path}`,
        sendMessage: (message: { type: string; payload?: Record<string, unknown> }) => {
          boundary.sent.push(message)
          if (message.type === "tldw:request") boundary.beforeRequest?.()
          return new Promise(resolve => {
            const listener = [...boundary.listeners][0]
            if (!listener) throw new Error("Missing background listener")
            listener(message, { id: "extension-fixture" }, resolve)
          })
        },
        onConnect: event(),
        onStartup: event(),
        onMessage: { addListener: (listener: Parameters<typeof boundary.listeners.add>[0]) => boundary.listeners.add(listener) }
      },
      storage: {
        local: { get: async () => ({}), set: async () => {} },
        session: { get: async () => ({}), set: async () => {} },
        onChanged: event()
      },
      alarms: { get: async () => undefined, clear: async () => true, create: async () => {}, onAlarm: event() },
      tabs: { query: async () => [], create: vi.fn(), sendMessage: async () => {} },
      action: { onClicked: event() },
      contextMenus: { create: vi.fn(), removeAll: vi.fn(), onClicked: event() },
      i18n: { getMessage: (key: string) => key }
    }
  }
})

import background from "@/entries/background"
import { bgRequest } from "@/services/background-proxy"
import { tldwAuth } from "@/services/tldw/TldwAuth"
import { requestScopeFields } from "@/services/tldw/domains/service-prompts"
import { createNotesGraphAuthorityScope } from "@/components/Notes/hooks/useNotesGraphAuthorityScope"
import { useNotesWikilinks } from "@/components/Notes/hooks/useNotesWikilinks"

const config = (user = 7): TldwConfig => ({
  serverUrl: "https://notes.example.test",
  authMode: "multi-user",
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
})
const capturedConfig = config()
const scope = {
  config: { serverUrl: capturedConfig.serverUrl, authMode: capturedConfig.authMode },
  userId: 7
}
const noteId = "925871f6-30fa-470b-a2c4-2272051f2373"
const sourceId = "11111111-1111-4111-8111-111111111111"
const requests = [
  { path: "/api/v1/notes/search/?query=Sec&title_only=true", method: "GET" },
  { path: "/api/v1/notes/wikilinks/resolve", method: "POST", body: { titles: ["Secret"], ids: [], source_note_id: sourceId } }
] as const
const clients: QueryClient[] = []

function Harness() {
  const textarea = React.useRef<HTMLTextAreaElement | null>(null)
  const links = useNotesWikilinks({
    isOnline: true,
    authorityScope: createNotesGraphAuthorityScope(capturedConfig.serverUrl, 7),
    selectedId: sourceId,
    title: "Source",
    content: "[[Secret]] [[Sec",
    editorCursorIndex: 15,
    setEditorCursorIndex: () => {},
    editorDisabled: false,
    editorMode: "split",
    contentTextareaRef: textarea,
    resizeEditorTextarea: () => {},
    setContentDirty: () => {},
    data: [],
    noteRelations: { related: [], backlinks: [], manualLinks: [] }
  })
  return <>
    <output data-testid="suggestions">{links.wikilinkSuggestions.map(note => note.title).join(",")}</output>
    <output data-testid="preview">{links.previewContent}</output>
  </>
}

beforeEach(() => {
  vi.clearAllMocks()
  boundary.local.clear()
  boundary.session.clear()
  boundary.listeners.clear()
  boundary.sent.length = 0
  boundary.beforeRequest = null
  boundary.local.set("tldwConfig", config())
  boundary.connection = { config: config(), loading: false, authorityLoading: false }
  vi.spyOn(tldwAuth, "getCurrentUser").mockResolvedValue({ id: 7, is_active: true } as never)
  boundary.fetch.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    init?.signal?.throwIfAborted()
    const data = new URL(String(input)).pathname.endsWith("/resolve")
      ? { titles: [{ title: "Secret", note_id: noteId, note_title: "Secret owner 7", candidate_count: 1 }], ids: [] }
      : { items: [{ id: noteId, title: "Secret owner 7" }] }
    return new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } })
  })
  vi.stubGlobal("fetch", boundary.fetch)
  background.main()
})
afterEach(() => {
  cleanup()
  for (const client of clients.splice(0)) client.clear()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe("Stage26 Notes captured owner through actual proxy and worker", () => {
  it("delivers both mounted hook lookups with one coherent scoped transport contract", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
    clients.push(client)
    render(<QueryClientProvider client={client}><Harness /></QueryClientProvider>)
    await waitFor(() => expect(boundary.sent.filter(message => message.type === "tldw:request")).toHaveLength(2))
    await waitFor(() => expect(client.isFetching()).toBe(0))
    expect(client.getQueryCache().getAll().map(query => query.state.error)).toEqual([null, null])
    expect(boundary.fetch).toHaveBeenCalledTimes(2)
    await waitFor(() => expect(screen.getByTestId("suggestions")).toHaveTextContent("Secret owner 7"))
    await waitFor(() => expect(screen.getByTestId("preview")).toHaveTextContent(noteId))
    for (const message of boundary.sent.filter(message => message.type === "tldw:request")) {
      expect(message.payload?.servicePromptConfig).toMatchObject({ ...scope.config, expectedUserId: 7 })
      expect(message.payload).not.toHaveProperty("expectedConnectionAuthority")
      expect(message.payload).not.toHaveProperty("expectedConnectionEpoch")
    }
    for (const [, init] of boundary.fetch.mock.calls) {
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
      expect(new Headers(init.headers).get("Authorization")).toBe(`Bearer ${capturedConfig.accessToken}`)
    }
  })

  it.each(requests)("supports scoped-only $method $path through the real worker", async request => {
    await expect(bgRequest({ ...request, ...requestScopeFields(scope), abortSignal: new AbortController().signal })).resolves.toBeTruthy()
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
    expect(boundary.sent).toHaveLength(1)
    expect(boundary.sent[0].type).toBe("tldw:request")
  })

  it.each(requests)("retains worker rejection of mixed guards for $method $path", async request => {
    await expect(bgRequest({ ...request, ...requestScopeFields(scope), configSnapshot: capturedConfig, abortSignal: new AbortController().signal })).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })

  it.each(["principal", "server", "auth source", "organization"])("rejects a changed worker %s for both captured lookups", async change => {
    boundary.beforeRequest = () => boundary.local.set("tldwConfig", change === "principal" ? config(8)
      : change === "server" ? { ...config(), serverUrl: "https://other.example" }
      : change === "auth source" ? { ...config(), authSource: "cookie-session" }
      : { ...config(), orgId: "other-org" })
    for (const request of requests) {
      await expect(bgRequest({ ...request, ...requestScopeFields(scope), abortSignal: new AbortController().signal })).rejects.toMatchObject({ status: 412 })
    }
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
})
