import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { HashRouter, MemoryRouter, Route, Routes, useLocation, useNavigate } from "react-router-dom"
import { Storage } from "@plasmohq/storage"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { FlashcardsManager } from "../FlashcardsManager"
import { useFlashcardsGenerateTransfer, useStudyPackTransfer } from "@/hooks/useFlashcardsGenerateTransfer"
import { loadServicePromptSnapshot } from "@/services/service-prompts"
import { flashcardsHandoffAuthority } from "@/services/tldw/flashcards-generate-transfer"
import {
  buildFlashcardsGenerateRoute,
  createFlashcardsGenerateHandoff,
  createStudyPackHandoff
} from "@/services/tldw/flashcards-generate-handoff"
import { buildStudyPackRoute } from "@/services/tldw/study-pack-handoff"

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
  useDecksQuery: () => ({ data: [{ id: 7, name: "Owned deck" }], isSuccess: true }),
  useImportLimitsQuery: () => ({ data: null })
}))
vi.mock("../tabs", async () => ({
  ReviewTab: ({ reviewDeckId, onReviewDeckChange }: { reviewDeckId?: number | null; onReviewDeckChange: (id: number | undefined) => void }) => <div>
    <output data-testid="review-deck">{reviewDeckId ?? "all"}</output>
    <button onClick={() => onReviewDeckChange(12)}>Select live deck</button>
    <button onClick={() => onReviewDeckChange(undefined)}>Clear live deck</button>
  </div>,
  ManageTab: () => null,
  SchedulerTab: ({ onDirtyChange, discardSignal }: { onDirtyChange: (dirty: boolean) => void; discardSignal: number }) => <div>
    <button onClick={() => onDirtyChange(true)}>Edit scheduler</button>
    <output data-testid="scheduler-discard">{discardSignal}</output>
  </div>,
  TemplatesTab: () => null,
  ImportExportTab: (await import("../tabs/ImportExportTab")).ImportExportTab
}))
vi.mock("../components", () => ({ KeyboardShortcutsModal: () => null }))
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
const studyIntent = { title: "Alice private study pack", sourceItems: [{ sourceType: "note" as const, sourceId: "alice-note", sourceTitle: "Alice private title" }] }
const createStudyRoute = async () => {
  const snapshot = await loadServicePromptSnapshot([])
  try { return buildStudyPackRoute(await createStudyPackHandoff(studyIntent, flashcardsHandoffAuthority(snapshot))) }
  finally { snapshot.release() }
}
const Location = () => {
  const location = useLocation()
  const navigate = useNavigate()
  return <><output data-testid="location">{location.pathname + location.search + location.hash}</output><button onClick={() => navigate(-1)}>Browser Back</button><button onClick={() => navigate("/flashcards?tab=review&deck_id=21")}>Open external deck</button></>
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

  it("delivers a Study Pack into the actual drawer after its source unmounts", async () => {
    const failures = vi.fn()
    const Source = () => {
      const transfer = useStudyPackTransfer()
      const navigate = useNavigate()
      return <button onClick={() => { void transfer(() => studyIntent, { navigate }).catch(failures) }}>Study this source</button>
    }
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={["/source"]}><Location /><Routes>
      <Route path="/source" element={<Source />} />
      <Route path="/flashcards" element={<FlashcardsManager />} />
    </Routes></MemoryRouter></QueryClientProvider>)
    fireEvent.click(screen.getByRole("button", { name: "Study this source" }))
    expect(await screen.findByDisplayValue(studyIntent.title)).toBeVisible()
    expect(screen.getByText("note · alice-note")).toBeVisible()
    expect(screen.queryByRole("button", { name: "Study this source" })).not.toBeInTheDocument()
    expect(screen.getByTestId("location")).not.toHaveTextContent(/Alice|alice-note|study_pack_handoff/)
    expect(failures).not.toHaveBeenCalled()
  })

  it("consumes a Study Pack once under StrictMode and cannot restore it on reload or replay", async () => {
    const route = await createStudyRoute()
    const view = mount(route, true)
    expect(await screen.findByDisplayValue(studyIntent.title)).toBeVisible()
    const cleanRoute = screen.getByTestId("location").textContent!
    expect(cleanRoute).toBe("/flashcards?tab=importExport")
    view.unmount()
    const reload = mount(cleanRoute)
    await act(async () => {})
    expect(screen.queryByDisplayValue(studyIntent.title)).not.toBeInTheDocument()
    reload.unmount()
    mount(route)
    expect(await screen.findByText(/missing, expired, or already consumed/)).toBeVisible()
    expect(screen.queryByText("Alice private title")).not.toBeInTheDocument()
  })

  it.each(["same-tab", "cross-tab", "A-B-A"])("clears an open Study Pack drawer at the account boundary (%s)", async boundary => {
    mount(await createStudyRoute())
    expect(await screen.findByDisplayValue(studyIntent.title)).toBeVisible()
    act(() => {
      if (boundary === "cross-tab") {
        const previous = window.localStorage.getItem("tldwConfig")!
        const next = JSON.stringify({ ...JSON.parse(previous), accessToken: tokenFor(2), refreshToken: "refresh-2" })
        mocks.user = 2
        window.localStorage.setItem("tldwConfig", next)
        window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: previous, newValue: next }))
      } else {
        setUser(2)
        if (boundary === "A-B-A") setUser(1)
      }
    })
    await waitFor(() => expect(screen.queryByDisplayValue(studyIntent.title)).not.toBeInTheDocument())
    expect(screen.queryByText("note · alice-note")).not.toBeInTheDocument()
    expect(screen.getByTestId("location")).not.toHaveTextContent(/Alice|alice-note/)
  })

  it("rejects another account's Study Pack token without rendering source details", async () => {
    const route = await createStudyRoute()
    setUser(2)
    mount(route)
    expect(await screen.findByText(/belongs to another account/)).toBeVisible()
    expect(screen.queryByDisplayValue(studyIntent.title)).not.toBeInTheDocument()
    expect(screen.queryByText("note · alice-note")).not.toBeInTheDocument()
  })

  it("scrubs an old Study Pack history entry when Bob goes Back and keeps reload empty", async () => {
    setUser(2)
    const legacy = `/flashcards?tab=importExport&study_pack=1&study_pack_title=${encodeURIComponent(studyIntent.title)}&study_pack_payload=${encodeURIComponent(JSON.stringify(studyIntent.sourceItems))}`
    const view = mount([legacy, "/flashcards?tab=importExport"])
    await act(async () => {})
    fireEvent.click(screen.getByRole("button", { name: "Browser Back" }))
    expect(await screen.findByText(/old link contains unbound/)).toBeVisible()
    const cleanRoute = screen.getByTestId("location").textContent!
    expect(cleanRoute).not.toMatch(/Alice|alice-note|study_pack/)
    expect(screen.queryByDisplayValue(studyIntent.title)).not.toBeInTheDocument()
    view.unmount()
    mount(cleanRoute)
    await act(async () => {})
    expect(screen.queryByText("note · alice-note")).not.toBeInTheDocument()
  })

  it("scrubs both legacy transfer types even when account verification fails", async () => {
    mocks.resolveUser.mockRejectedValue(new Error("Signed out"))
    mount("/flashcards?tab=importExport&generate_text=AliceGenerateSecret&study_pack=1&study_pack_title=AliceStudySecret&study_pack_payload=private")
    await waitFor(() => expect(screen.getByTestId("location").textContent).toBe("/flashcards?tab=importExport"))
    expect(screen.queryByDisplayValue("AliceGenerateSecret")).not.toBeInTheDocument()
    expect(screen.queryByDisplayValue("AliceStudySecret")).not.toBeInTheDocument()
  })

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

  it("keeps accepted tab routes on reload without losing the private editor or adding history", async () => {
    const view = mount(["/source", await createRoute()], true)
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(privateIntent.text))
    await waitFor(() => expect(screen.getByTestId("location")).not.toHaveTextContent("generate_handoff"))
    fireEvent.change(screen.getByTestId("flashcards-generate-text"), { target: { value: "Edited private source" } })
    fireEvent.click(screen.getByRole("tab", { name: "Study", exact: true }))
    expect.soft(screen.getByTestId("location")).toHaveTextContent("tab=review")
    const reloadRoute = screen.getByTestId("location").textContent!
    fireEvent.click(screen.getByRole("tab", { name: "Import / Export", exact: true }))
    expect(screen.getByTestId("flashcards-generate-text")).toHaveValue("Edited private source")
    expect(screen.getByTestId("location")).not.toHaveTextContent(/Alice|Edited|generate_handoff/)
    fireEvent.click(screen.getByRole("button", { name: "Browser Back" }))
    expect.soft(screen.getByTestId("location").textContent).toBe("/source")
    view.unmount()
    mount(reloadRoute)
    expect(screen.getByRole("tab", { name: "Study", exact: true })).toHaveAttribute("aria-selected", "true")
  })

  it("tab-only navigation preserves a live deck choice while a new external deck still applies", async () => {
    mount("/flashcards?tab=review&deck_id=9&quiz_id=3&attempt_id=4&include_workspace_items=1&other=kept#anchor")
    expect(screen.getByTestId("review-deck")).toHaveTextContent("9")
    fireEvent.click(screen.getByRole("button", { name: "Select live deck" }))
    fireEvent.click(screen.getByRole("tab", { name: "Manage", exact: true }))
    expect.soft(screen.getByTestId("location")).toHaveTextContent("tab=cards")
    expect(screen.getByTestId("location")).toHaveTextContent("deck_id=9&quiz_id=3&attempt_id=4&include_workspace_items=1&other=kept#anchor")
    fireEvent.click(screen.getByRole("tab", { name: "Study", exact: true }))
    expect.soft(screen.getByTestId("review-deck")).toHaveTextContent("12")
    fireEvent.click(screen.getByRole("button", { name: "Clear live deck" }))
    fireEvent.click(screen.getByRole("tab", { name: "Templates", exact: true }))
    fireEvent.click(screen.getByRole("tab", { name: "Study", exact: true }))
    expect.soft(screen.getByTestId("review-deck")).toHaveTextContent("all")
    fireEvent.click(screen.getByRole("button", { name: "Open external deck" }))
    expect(screen.getByTestId("review-deck")).toHaveTextContent("21")
  })

  it("changes a dirty Scheduler route only after discard is accepted", () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
    mount("/flashcards?tab=scheduler&deck_id=7#policy")
    fireEvent.click(screen.getByRole("button", { name: "Edit scheduler" }))
    fireEvent.click(screen.getByRole("tab", { name: "Study", exact: true }))
    expect(screen.getByTestId("location")).toHaveTextContent("tab=scheduler&deck_id=7#policy")
    expect(screen.getByRole("tab", { name: "Scheduler", exact: true })).toHaveAttribute("aria-selected", "true")
    expect(screen.getByTestId("scheduler-discard")).toHaveTextContent("0")
    confirm.mockReturnValue(true)
    fireEvent.click(screen.getByRole("tab", { name: "Study", exact: true }))
    expect.soft(screen.getByTestId("location")).toHaveTextContent("tab=review&deck_id=7#policy")
    expect(screen.getByTestId("scheduler-discard")).toHaveTextContent("1")
    expect(confirm).toHaveBeenCalledTimes(2)
  })

  it("finishes a pending private handoff without replacing the newly selected tab", async () => {
    const route = await createRoute()
    let removalStarted = false
    const release = deferred<void>()
    const originalRemove = Storage.prototype.remove
    vi.spyOn(Storage.prototype, "remove").mockImplementation(async function (key) {
      await originalRemove.call(this, key)
      if (key.startsWith("tldw:flashcards-generate-handoff:")) {
        removalStarted = true
        await release.promise
      }
    })
    mount(route)
    await waitFor(() => expect(removalStarted).toBe(true))
    fireEvent.click(screen.getByRole("tab", { name: "Study", exact: true }))
    await act(async () => { release.resolve() })
    await waitFor(() => expect(screen.getByTestId("location")).not.toHaveTextContent("generate_handoff"))
    expect(screen.getByRole("tab", { name: "Study", exact: true })).toHaveAttribute("aria-selected", "true")
    fireEvent.click(screen.getByRole("tab", { name: "Import / Export", exact: true }))
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(privateIntent.text))
    expect(screen.queryByText(/missing, expired, or already consumed/)).not.toBeInTheDocument()
  })

  it.each([
    ["Manage", "cards"], ["Templates", "templates"], ["Scheduler", "scheduler"]
  ])("restores the accepted %s tab on remount", (label, key) => {
    const view = mount("/flashcards?tab=review&other=kept#anchor")
    fireEvent.click(screen.getByRole("tab", { name: label, exact: true }))
    const route = screen.getByTestId("location").textContent!
    expect(route).toBe(`/flashcards?tab=${key}&other=kept#anchor`)
    view.unmount()
    mount(route)
    expect(screen.getByRole("tab", { name: label, exact: true })).toHaveAttribute("aria-selected", "true")
  })

  it("uses the real extension HashRouter for the accepted tab and reload", async () => {
    const previousUrl = window.location.href
    window.history.replaceState(null, "", "/#/flashcards?tab=importExport&other=kept")
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const tree = <QueryClientProvider client={client}><HashRouter><Location /><FlashcardsManager /></HashRouter></QueryClientProvider>
    const view = render(tree)
    try {
      fireEvent.click(screen.getByRole("tab", { name: "Study", exact: true }))
      await waitFor(() => expect(window.location.hash).toBe("#/flashcards?tab=review&other=kept"))
      view.unmount()
      const restored = render(tree)
      expect(screen.getByRole("tab", { name: "Study", exact: true })).toHaveAttribute("aria-selected", "true")
      restored.unmount()
    } finally {
      view.unmount()
      window.history.replaceState(null, "", previousUrl)
    }
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
