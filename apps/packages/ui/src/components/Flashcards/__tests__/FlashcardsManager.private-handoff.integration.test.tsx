import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { FlashcardsManager } from "../FlashcardsManager"
import { useFlashcardsGenerateTransfer } from "@/hooks/useFlashcardsGenerateTransfer"
import { loadServicePromptSnapshot } from "@/services/service-prompts"
import { flashcardsHandoffAuthority } from "@/services/tldw/flashcards-generate-transfer"
import {
  buildFlashcardsGenerateRoute,
  createFlashcardsGenerateHandoff
} from "@/services/tldw/flashcards-generate-handoff"

const mocks = vi.hoisted(() => ({ request: vi.fn(), user: 1, resolveUser: vi.fn(), messages: { success: vi.fn(), warning: vi.fn(), error: vi.fn() } }))
vi.mock("@plasmohq/storage", async () => import("../../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: async () => {},
  ensureConfigForRequest: async () => JSON.parse(window.localStorage.getItem("tldwConfig") || "null")
} }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: (...args: unknown[]) => mocks.resolveUser(...args) } }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: (...args: unknown[]) => mocks.request(...args), bgUpload: vi.fn() }))
vi.mock("@/services/prompt-studio", () => ({ getLlmProviders: async () => ({ providers: ["local"] }) }))
vi.mock("@/hooks/useAntdMessage", () => ({ useAntdMessage: () => mocks.messages }))
vi.mock("../hooks", async (original) => ({
  ...await original<typeof import("../hooks")>(),
  useDecksQuery: () => ({ data: [{ id: 7, name: "Owned deck" }] }),
  useImportLimitsQuery: () => ({ data: null })
}))
vi.mock("../tabs", async () => ({
  ReviewTab: () => null, ManageTab: () => null, SchedulerTab: () => null, TemplatesTab: () => null,
  ImportExportTab: (await import("../tabs/ImportExportTab")).ImportExportTab
}))
vi.mock("../components", () => ({ KeyboardShortcutsModal: () => null }))
vi.mock("../components/StudyPackCreateDrawer", () => ({ StudyPackCreateDrawer: () => null }))
vi.mock("../tabs/ImportExport/StudyPackPanel", () => ({ StudyPackPanel: () => null }))
vi.mock("../tabs/ImportExport/ImportPanel", () => ({ ImportPanel: () => null }))
vi.mock("../tabs/ImportExport/ExportPanel", () => ({ ExportPanel: () => null }))
vi.mock("../tabs/ImageOcclusionTransferPanel", () => ({ ImageOcclusionTransferPanel: () => null }))
vi.mock("wxt/browser", () => ({ browser: { runtime: {}, tabs: {} } }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({
  t: (key: string, options?: string | Record<string, unknown>) => typeof options === "string" ? options : String(options?.defaultValue || key).replace(/\{\{(\w+)\}\}/g, (_match, name: string) => String(options?.[name] ?? ""))
}) }))

const tokenFor = (user: number) => `eyJhbGciOiJub25lIn0.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
const setUser = (user: number) => {
  mocks.user = user
  window.localStorage.setItem("tldwConfig", JSON.stringify({
    serverUrl: "https://server.test", authMode: "multi-user", accessToken: tokenFor(user), refreshToken: `refresh-${user}`
  }))
  window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}
const privateIntent = { text: " \nAlice unsaved private note\n\t ", sourceType: "note" as const, sourceId: "alice-note", sourceTitle: "Alice private title" }
const createRoute = async () => {
  const snapshot = await loadServicePromptSnapshot([])
  try { return buildFlashcardsGenerateRoute(await createFlashcardsGenerateHandoff(privateIntent, flashcardsHandoffAuthority(snapshot))) }
  finally { snapshot.release() }
}
const Location = () => {
  const location = useLocation()
  const navigate = useNavigate()
  return <><output data-testid="location">{location.pathname + location.search + location.hash}</output><button onClick={() => navigate(-1)}>Browser Back</button></>
}
const mount = (route: string | string[], strict = false) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const tree = <QueryClientProvider client={client}><MemoryRouter initialEntries={Array.isArray(route) ? route : [route]}><Location /><FlashcardsManager /></MemoryRouter></QueryClientProvider>
  return render(strict ? <React.StrictMode>{tree}</React.StrictMode> : tree)
}

describe("actual Flashcards generation private handoff", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    setUser(1)
    mocks.resolveUser.mockImplementation(async () => ({ id: mocks.user, is_active: true }))
    mocks.request.mockImplementation(async ({ path }: { path: string }) => {
      if (path === "/api/v1/flashcards/generate") return { flashcards: [{ front: "Private front", back: "Private back", model_type: "basic" }] }
      if (path === "/api/v1/flashcards") return { id: 11 }
      if (path === "/api/v1/flashcards/decks") return { id: 7, name: "Owned deck" }
      return []
    })
    let tail = Promise.resolve()
    const locks = {
      request: (_name: string, work: () => unknown) => {
        const next = tail.then(work); tail = next.then(() => undefined, () => undefined); return next
      }
    }
    vi.stubGlobal("navigator", new Proxy(window.navigator, {
      get: (target, key) => key === "locks" ? locks : Reflect.get(target, key, target)
    }))
  })
  afterEach(() => vi.unstubAllGlobals())

  it("delivers through the actual router after its source unmounts", async () => {
    const failures = vi.fn()
    const Source = () => {
      const transfer = useFlashcardsGenerateTransfer()
      const navigate = useNavigate()
      return <button onClick={() => { void transfer(() => privateIntent, { navigate }).catch(failures) }}>Generate from source</button>
    }
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={["/source"]}><Location /><Routes>
      <Route path="/source" element={<Source />} />
      <Route path="/flashcards" element={<FlashcardsManager />} />
    </Routes></MemoryRouter></QueryClientProvider>)
    fireEvent.click(screen.getByRole("button", { name: "Generate from source" }))
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(privateIntent.text))
    expect(screen.queryByRole("button", { name: "Generate from source" })).not.toBeInTheDocument()
    expect(failures).not.toHaveBeenCalled()
    expect(screen.getByTestId("location")).not.toHaveTextContent("generate_handoff")
    expect(screen.getByTestId("location")).not.toHaveTextContent("Alice")
  })

  it("consumes once under StrictMode, removes the token, and preserves later editor changes", async () => {
    const route = await createRoute()
    mount(route, true)
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(privateIntent.text))
    const input = screen.getByTestId("flashcards-generate-text")
    await waitFor(() => expect(screen.getByTestId("location")).toHaveTextContent("/flashcards?tab=importExport"))
    expect(screen.getByTestId("location")).not.toHaveTextContent("generate_handoff")
    fireEvent.change(input, { target: { value: "My later edit" } })
    await act(async () => {})
    expect(input).toHaveValue("My later edit")
  })

  it("generates and saves the exact source with captured owner and provenance", async () => {
    mount(await createRoute())
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(privateIntent.text))
    fireEvent.click(screen.getByTestId("flashcards-generate-button"))
    await screen.findByDisplayValue("Private front")
    const generation = mocks.request.mock.calls.find(([request]) => request.path === "/api/v1/flashcards/generate")?.[0]
    expect(generation).toMatchObject({ body: { text: privateIntent.text }, servicePromptConfig: { expectedUserId: 1, serverUrl: "https://server.test" }, headers: { "X-TLDW-Expected-User-ID": "1" } })
    fireEvent.click(screen.getByTestId("flashcards-generate-save-button"))
    await waitFor(() => expect(mocks.request.mock.calls.some(([request]) => request.path === "/api/v1/flashcards")).toBe(true))
    const saved = mocks.request.mock.calls.find(([request]) => request.path === "/api/v1/flashcards")?.[0]
    expect(saved).toMatchObject({ body: { source_ref_type: "note", source_ref_id: "alice-note" }, servicePromptConfig: { expectedUserId: 1 } })
  })

  it("clears private source and ignores generation completing after A to B to A", async () => {
    const generation = deferred<unknown>()
    mocks.request.mockImplementation(({ path }: { path: string }) => path === "/api/v1/flashcards/generate" ? generation.promise : Promise.resolve([]))
    mount(await createRoute())
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(privateIntent.text))
    fireEvent.click(screen.getByTestId("flashcards-generate-button"))
    await waitFor(() => expect(mocks.request).toHaveBeenCalled())
    act(() => { setUser(2); setUser(1) })
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(""))
    await act(async () => generation.resolve({ flashcards: [{ front: "Late Alice front", back: "Late Alice back" }] }))
    expect(screen.queryByDisplayValue("Late Alice front")).not.toBeInTheDocument()
    expect(screen.queryByText("Alice private title")).not.toBeInTheDocument()
  })

  it("rejects legacy plaintext browser-history entry instead of importing Alice content into Bob", async () => {
    setUser(2)
    mount("/flashcards?tab=importExport&generate=1&generate_text=Alice%20private%20note&generate_source_id=alice-note&generate_source_title=Alice%20private%20title")
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(""))
    expect(await screen.findByText(/Reopen.*source/i)).toBeInTheDocument()
    await waitFor(() => expect(screen.getByTestId("location")).not.toHaveTextContent("Alice"))
    expect(mocks.request).not.toHaveBeenCalled()
  })

  it("keeps an unresolved target token until the intended account signs in", async () => {
    const route = await createRoute()
    mocks.resolveUser.mockResolvedValue(null)
    mount(route)
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(""))
    expect(screen.getByTestId("location")).toHaveTextContent("generate_handoff")
    mocks.resolveUser.mockImplementation(async () => ({ id: 1, is_active: true }))
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } })))
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(privateIntent.text))
  })
  it("does not continue saving cards after a delayed first save crosses A to B to A", async () => {
    const saving = deferred<unknown>()
    mocks.request.mockImplementation(({ path }: { path: string }) => {
      if (path.endsWith("/generate")) return Promise.resolve({ flashcards: [{ front: "First", back: "A" }, { front: "Second", back: "B" }] })
      if (path === "/api/v1/flashcards") return saving.promise
      return Promise.resolve([])
    })
    mount(await createRoute())
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(privateIntent.text))
    fireEvent.click(screen.getByTestId("flashcards-generate-button"))
    await screen.findByDisplayValue("First")
    fireEvent.click(screen.getByTestId("flashcards-generate-save-button"))
    await waitFor(() => expect(mocks.request.mock.calls.filter(([r]) => r.path === "/api/v1/flashcards")).toHaveLength(1))
    const first = mocks.request.mock.calls.find(([r]) => r.path === "/api/v1/flashcards")![0]
    act(() => { setUser(2); setUser(1) })
    expect(first.abortSignal.aborted).toBe(true)
    await act(async () => saving.resolve({ id: 3 }))
    expect(mocks.request.mock.calls.filter(([r]) => r.path === "/api/v1/flashcards")).toHaveLength(1)
    expect(screen.queryByDisplayValue("Second")).not.toBeInTheDocument()
  })

  it("rejects an opaque transfer from a foreign account", async () => {
    const route = await createRoute()
    setUser(2)
    mount(route)
    expect(await screen.findByText(/belongs to another account/)).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-generate-text")).toHaveValue("")
    expect(mocks.request).not.toHaveBeenCalled()
  })

  it("cleans an old private history entry when Bob goes Back", async () => {
    setUser(2)
    mount(["/flashcards?tab=importExport&generate_text=AlicePrivate&generate_source_title=AliceTitle", "/flashcards?tab=importExport"])
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(""))
    fireEvent.click(screen.getByRole("button", { name: "Browser Back" }))
    expect(await screen.findByText(/old link contains unbound/)).toBeInTheDocument()
    expect(screen.getByTestId("location")).not.toHaveTextContent("Alice")
    expect(screen.getByTestId("flashcards-generate-text")).toHaveValue("")
  })

  it("preserves manual direct-entry text during same-principal token rotation", async () => {
    mount("/flashcards?tab=importExport")
    const input = screen.getByTestId("flashcards-generate-text")
    fireEvent.change(input, { target: { value: "Manual source" } })
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } })))
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-button")).toBeEnabled())
    expect(input).toHaveValue("Manual source")
  })

})
