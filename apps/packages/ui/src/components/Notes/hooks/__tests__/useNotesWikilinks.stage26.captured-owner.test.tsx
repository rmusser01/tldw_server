import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createNotesGraphAuthorityScope } from "../useNotesGraphAuthorityScope"
import { useNotesWikilinks } from "../useNotesWikilinks"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  getCurrentUser: vi.fn(),
  connection: { config: null as TldwConfig | null, loading: false, authorityLoading: false }
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.bgRequest }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: mocks.getCurrentUser } }))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => mocks.connection
}))

const SERVER = "https://notes.example.test"
const CONFIG: TldwConfig = { serverUrl: SERVER, authMode: "multi-user", accessToken: "owner-a" }
const SCOPE_A = createNotesGraphAuthorityScope(SERVER, 1)
const SCOPE_B = createNotesGraphAuthorityScope(SERVER, 2)
const NOTE_A = "11111111-1111-4111-8111-111111111111"
const NOTE_B = "22222222-2222-4222-8222-222222222222"
const STALE_NOTE = "33333333-3333-4333-8333-333333333333"

type Request = {
  path: string
  method: string
  body?: unknown
  headers?: Record<string, string>
  abortSignal?: AbortSignal
  configSnapshot?: TldwConfig | null
  servicePromptConfig?: {
    serverUrl: string
    authMode: string
    authSource?: string
    orgId?: number
    expectedUserId?: number | string | null
    expectedSingleUserApiKeyScope?: string | null
  }
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}
const response = (request: Request, id: string, title: string) =>
  request.method === "GET"
    ? { items: [{ id, title }] }
    : { titles: [{ title: "Secret", note_id: id, note_title: title, candidate_count: 1 }], ids: [] }

function Harness({ scope }: { scope: string | null }) {
  const textarea = React.useRef<HTMLTextAreaElement | null>(null)
  const links = useNotesWikilinks({
    isOnline: true,
    authorityScope: scope,
    selectedId: "source-note",
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

const clients: QueryClient[] = []
const mount = (scope: string | null = SCOPE_A) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
  clients.push(client)
  const tree = (owner: string | null) => <QueryClientProvider client={client}><Harness scope={owner} /></QueryClientProvider>
  const view = render(tree(scope))
  return { client, ...view, owner: (owner: string | null) => view.rerender(tree(owner)) }
}
const cachedData = (client: QueryClient) => client.getQueryCache().getAll().map(query => query.state.data)
const expectNoPrivateData = (client: QueryClient, id: string, title: string) => {
  for (const data of cachedData(client)) {
    if (Array.isArray(data)) expect(data).not.toContainEqual({ id, title })
    else if (data && typeof data === "object" && "titles" in data) {
      const titles = (data as { titles: Map<string, { noteId: string }> }).titles
      expect([...titles.values()].map(note => note.noteId)).not.toContain(id)
    }
  }
  expect(screen.queryByTestId("suggestions")?.textContent ?? "").not.toContain(title)
  expect(screen.queryByTestId("preview")?.textContent ?? "").not.toContain(id)
}

describe("Stage26 wikilink captured-owner transport and cache", () => {
  let userId: number
  let delivered: Request[]
  beforeEach(() => {
    vi.resetAllMocks()
    userId = 1
    delivered = []
    mocks.connection = { config: { ...CONFIG }, loading: false, authorityLoading: false }
    mocks.getCurrentUser.mockImplementation(async () => ({ id: userId, is_active: true }))
    mocks.bgRequest.mockImplementation(async (request: Request) => {
      delivered.push(request)
      return response(request, userId === 1 ? NOTE_A : NOTE_B, `Secret owner ${userId}`)
    })
  })
  afterEach(() => {
    cleanup()
    for (const client of clients.splice(0)) client.clear()
  })

  it("binds both lookups to the verified owner and preserves successful suggestions and preview", async () => {
    const { client } = mount()
    await waitFor(() => expect(screen.getByTestId("suggestions")).toHaveTextContent("Secret owner 1"))
    await waitFor(() => expect(screen.getByTestId("preview")).toHaveTextContent(NOTE_A))
    expect(delivered).toHaveLength(2)
    for (const request of delivered) {
      expect(request.headers?.["X-TLDW-Expected-User-ID"]).toBe("1")
      expect(request.configSnapshot).toBeUndefined()
      expect(request.servicePromptConfig).toMatchObject({ serverUrl: SERVER, authMode: "multi-user", expectedUserId: 1 })
      expect(request.abortSignal).toBeInstanceOf(AbortSignal)
    }
    expect(delivered.find(request => request.method === "POST")?.body).toEqual({ titles: ["Secret"], ids: [], source_note_id: "source-note" })
    expect(cachedData(client).filter(Boolean)).toHaveLength(2)
  })

  it("does not select B credentials or cache B notes after A query keys were captured", async () => {
    const credentials = deferred<void>()
    mocks.bgRequest.mockImplementation(async (request: Request) => {
      await credentials.promise
      const checked = request.servicePromptConfig
      if (checked && String(checked.expectedUserId) !== String(userId)) throw new Error("Owner changed before dispatch")
      delivered.push(request)
      return response(request, NOTE_B, "Secret private B")
    })
    const { client } = mount()
    await waitFor(() => expect(mocks.bgRequest).toHaveBeenCalledTimes(2))
    userId = 2
    await act(async () => { credentials.resolve(); await credentials.promise })
    await waitFor(() => expect(client.isFetching()).toBe(0))
    expectNoPrivateData(client, NOTE_B, "Secret private B")
    expect(delivered).toHaveLength(0)
  })

  it.each([
    ["a different account", { id: 2, is_active: true }],
    ["an inactive account", { id: 1, is_active: false }],
    ["no principal", { id: null, is_active: true }]
  ])("refuses both requests before dispatch for %s", async (_name, user) => {
    mocks.getCurrentUser.mockResolvedValue(user)
    const { client } = mount()
    await waitFor(() => expect(client.isFetching()).toBe(0))
    expect(delivered).toHaveLength(0)
    expect(cachedData(client).filter(Boolean)).toHaveLength(0)
  })

  it("fails closed when authentication fails", async () => {
    mocks.getCurrentUser.mockRejectedValue(new Error("Session expired"))
    const { client } = mount()
    await waitFor(() => expect(client.isFetching()).toBe(0))
    expect(delivered).toHaveLength(0)
    expect(cachedData(client).filter(Boolean)).toHaveLength(0)
  })

  it.each([
    ["principal A-B-A", "tldw:auth-principal-changed", { kind: "switch" }],
    ["config A-B-A", "tldw:config-updated", { authorityChanged: true }],
    ["session invalidation", "tldw:config-updated", { authorityChanged: false, refreshSessionInvalidated: true }]
  ])("does not cache stale responses after %s even when the owner key is unchanged", async (_name, event, detail) => {
    const pending = deferred<void>()
    let call = 0
    mocks.bgRequest.mockImplementation(async (request: Request) => {
      if (++call <= 2) {
        await pending.promise
        return response(request, STALE_NOTE, "Secret stale response")
      }
      return response(request, NOTE_A, "Secret current A")
    })
    const { client } = mount()
    await waitFor(() => expect(mocks.bgRequest).toHaveBeenCalledTimes(2))
    act(() => {
      window.dispatchEvent(new CustomEvent(event, { detail }))
      window.dispatchEvent(new CustomEvent(event, { detail }))
    })
    await act(async () => { pending.resolve(); await pending.promise })
    await waitFor(() => expect(client.isFetching()).toBe(0))
    expectNoPrivateData(client, STALE_NOTE, "Secret stale response")
  })

  it.each([
    ["organization", { orgId: 9 }],
    ["authentication source", { authSource: "cookie-session" as const }],
    ["credential", { accessToken: "replacement-credential" }]
  ])("fences both completions across a same-owner %s config A-B-A", async (_name, replacement) => {
    const pending = deferred<void>()
    let call = 0
    mocks.bgRequest.mockImplementation(async (request: Request) => {
      if (++call <= 2) {
        await pending.promise
        return response(request, STALE_NOTE, "Secret stale config")
      }
      return response(request, NOTE_A, "Secret current config")
    })
    const view = mount()
    await waitFor(() => expect(mocks.bgRequest).toHaveBeenCalledTimes(2))
    mocks.connection.config = { ...CONFIG, ...replacement }
    view.owner(SCOPE_A)
    mocks.connection.config = { ...CONFIG }
    view.owner(SCOPE_A)
    await act(async () => { pending.resolve(); await pending.promise })
    await waitFor(() => expect(view.client.isFetching()).toBe(0))
    expectNoPrivateData(view.client, STALE_NOTE, "Secret stale config")
  })

  it("fences A-B-A during the owner check before either stale request is sent", async () => {
    const pending = deferred<{ id: number; is_active: boolean }>()
    mocks.getCurrentUser.mockImplementationOnce(() => pending.promise).mockImplementationOnce(() => pending.promise)
    const view = mount()
    await waitFor(() => expect(mocks.getCurrentUser).toHaveBeenCalledTimes(2))
    userId = 2
    mocks.connection.config = { ...CONFIG, accessToken: "owner-b" }
    view.owner(SCOPE_B)
    await waitFor(() => expect(view.client.isFetching()).toBe(0))
    userId = 1
    mocks.connection.config = { ...CONFIG }
    view.owner(SCOPE_A)
    await waitFor(() => expect(view.client.isFetching()).toBe(0))
    const currentDeliveries = delivered.length
    await act(async () => { pending.resolve({ id: 1, is_active: true }); await pending.promise })
    await waitFor(() => expect(view.client.isFetching()).toBe(0))
    expect(delivered).toHaveLength(currentDeliveries)
  })

  it("does not publish cookie-session B responses if ownership changes without a render", async () => {
    mocks.connection.config = { ...CONFIG, authSource: "cookie-session", accessToken: undefined }
    mocks.bgRequest.mockImplementation(async (request: Request) => {
      userId = 2
      return response(request, NOTE_B, "Secret private cookie B")
    })
    const { client } = mount()
    await waitFor(() => expect(client.isFetching()).toBe(0))
    expectNoPrivateData(client, NOTE_B, "Secret private cookie B")
  })

  it("does not cache pending responses after unmount", async () => {
    const pending = deferred<void>()
    mocks.bgRequest.mockImplementation(async (request: Request) => {
      await pending.promise
      return response(request, STALE_NOTE, "Secret after unmount")
    })
    const view = mount()
    await waitFor(() => expect(mocks.bgRequest).toHaveBeenCalledTimes(2))
    view.unmount()
    await act(async () => { pending.resolve(); await pending.promise })
    await waitFor(() => expect(view.client.isFetching()).toBe(0))
    expectNoPrivateData(view.client, STALE_NOTE, "Secret after unmount")
  })

  it("keeps successful requests through benign config notifications", async () => {
    const pending = deferred<void>()
    mocks.bgRequest.mockImplementation(async (request: Request) => {
      await pending.promise
      return response(request, NOTE_A, "Secret same owner")
    })
    const { client } = mount()
    await waitFor(() => expect(mocks.bgRequest).toHaveBeenCalledTimes(2))
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } })))
    await act(async () => { pending.resolve(); await pending.promise })
    await waitFor(() => expect(client.isFetching()).toBe(0))
    expect(screen.getByTestId("suggestions")).toHaveTextContent("Secret same owner")
    expect(screen.getByTestId("preview")).toHaveTextContent(NOTE_A)
  })
})
