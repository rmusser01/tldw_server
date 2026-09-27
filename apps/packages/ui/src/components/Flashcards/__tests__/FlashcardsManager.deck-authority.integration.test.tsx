import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter, Route, Routes, useNavigate } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useUpdateDeckMutation } from "../hooks/useFlashcardQueries"
import { FlashcardsManager } from "../FlashcardsManager"

const mocks = vi.hoisted(() => ({ request: vi.fn(), user: 1, resolveUser: vi.fn(), decks: [{ id: 7, name: "Alice private deck" }] as Array<{id: number; name: string}>, messages: { success: vi.fn(), warning: vi.fn(), error: vi.fn() } }))
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
vi.mock("../components/StudyPackCreateDrawer", () => ({ StudyPackCreateDrawer: () => null }))
vi.mock("../tabs/ImportExport/StudyPackPanel", () => ({ StudyPackPanel: () => null }))
vi.mock("../tabs/ImportExport/ImportPanel", () => ({ ImportPanel: () => null }))
vi.mock("../tabs/ImportExport/ExportPanel", () => ({ ExportPanel: () => null }))
vi.mock("../tabs/ImageOcclusionPanel", () => ({ ImageOcclusionPanel: () => null }))
vi.mock("@/hooks/useUndoNotification", () => ({ useUndoNotification: () => ({ showUndoNotification: vi.fn() }) }))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/hooks/useServerCapabilities", () => ({ useServerCapabilities: () => ({ capabilities: { hasFlashcards: true }, loading: false }) }))
vi.mock("wxt/browser", () => ({ browser: { runtime: {}, tabs: {} } }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({
  t: (key: string, options?: string | Record<string, unknown>) => typeof options === "string" ? options : String(options?.defaultValue || key).replace(/\{\{(\w+)\}\}/g, (_match, name: string) => String(options?.[name] ?? ""))
}) }))

const tokenFor = (user: number) => `eyJhbGciOiJub25lIn0.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
const setUser = (user: number, serverUrl = "https://server.test") => {
  mocks.user = user
  window.localStorage.setItem("tldwConfig", JSON.stringify({
    serverUrl, authMode: "multi-user", accessToken: tokenFor(user), refreshToken: `refresh-${user}`
  }))
  window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
}

const clients: QueryClient[] = []
const Navigation = () => {
  const navigate = useNavigate()
  const update = useUpdateDeckMutation()
  return <><button onClick={() => update.mutate({ deckId: 7, update: { name: "Old Alice update", expected_version: 1 } })}>Legacy update</button><button onClick={() => navigate("/settings")}>Settings</button><button onClick={() => navigate(-1)}>Back</button></>
}
const mount = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  clients.push(client)
  render(<QueryClientProvider client={client}><MemoryRouter initialEntries={["/flashcards?tab=importExport"]}><Navigation /><Routes>
    <Route path="/settings" element={<div>Settings page</div>} />
    <Route path="/flashcards" element={<FlashcardsManager />} />
  </Routes></MemoryRouter></QueryClientProvider>)
  return client
}
const selectorIds = ["flashcards-generate-deck", "flashcards-occlusion-deck"]
const selectDeck = async (id: string, name = "Alice private deck") => {
  fireEvent.mouseDown((await screen.findByTestId(id)))
  await waitFor(() => {
    // AntD test mode reuses listbox IDs, and leaving popups remain mounted.
    const dropdowns = Array.from(document.querySelectorAll<HTMLElement>(".ant-select-dropdown:not(.ant-select-dropdown-hidden)"))
      .filter(element => element.style.pointerEvents !== "none")
    expect(dropdowns).toHaveLength(1)
    fireEvent.click(within(dropdowns[0]).getByText(name, { selector: ".ant-select-item-option-content" }))
  })
  await waitFor(() => expect(screen.getByTestId(id)).toHaveTextContent(name))
}

describe("Transfer deck authority with the actual QueryClient and selectors", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    setUser(1)
    mocks.decks = [{ id: 7, name: "Alice private deck" }]
    mocks.resolveUser.mockImplementation(async () => ({ id: mocks.user, is_active: true }))
    mocks.request.mockImplementation(async ({ path, method }: { path: string; method?: string }) => {
      if (path.split("?")[0] === "/api/v1/flashcards/decks" && (!method || method === "GET")) return mocks.decks
      if (path.split("?")[0] === "/api/v1/flashcards") return { items: [], total: 0 }
      return {}
    })
  })
  afterEach(() => { clients.splice(0).forEach(client => client.clear()) })

  it("does not restore Alice's deck label after Settings logout, Bob login and Back", async () => {
    mount()
    for (const id of selectorIds) await selectDeck(id)
    fireEvent.click(screen.getByRole("button", { name: "Settings", exact: true }))
    await act(async () => { mocks.decks = []; setUser(2) })
    fireEvent.click(screen.getByRole("button", { name: "Back", exact: true }))
    fireEvent.change(screen.getByTestId("flashcards-generate-text"), { target: { value: "Bob source" } })
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-button")).toBeEnabled())
    for (const id of selectorIds) expect(screen.getByTestId(id)).not.toHaveTextContent("Alice private deck")
  })

  it("waits for verified authority before either scoped panel catalogue request", async () => {
    let finish!: (user: unknown) => void
    mocks.resolveUser.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    mount()
    await act(async () => { await Promise.resolve() })
    expect(mocks.request.mock.calls.filter(([input]) => input.path.split("?")[0] === "/api/v1/flashcards/decks" && input.servicePromptConfig)).toHaveLength(0)
    for (const id of selectorIds) expect(screen.getByTestId(id)).not.toHaveTextContent("Alice private deck")
    await act(async () => { finish({ id: 1, is_active: true }) })
    for (const id of selectorIds) await waitFor(() => expect(screen.getByTestId(id)).toHaveTextContent("Alice private deck"))
  })

  it("uses Bob's current label for a colliding numeric deck ID", async () => {
    mount()
    for (const id of selectorIds) await selectDeck(id)
    fireEvent.click(screen.getByRole("button", { name: "Settings", exact: true }))
    await act(async () => { mocks.decks = [{ id: 7, name: "Bob current deck" }]; setUser(2) })
    fireEvent.click(screen.getByRole("button", { name: "Back", exact: true }))
    for (const id of selectorIds) await waitFor(() => {
      expect(screen.getByTestId(id)).toHaveTextContent("Bob current deck")
      expect(screen.getByTestId(id)).not.toHaveTextContent("Alice private deck")
    })
  })

  it("ignores a delayed Alice list through an A-to-B-to-A authority replacement", async () => {
    let finish!: (decks: unknown) => void
    const original = mocks.request.getMockImplementation()!
    let captured = false
    mocks.request.mockImplementation(input => {
      if (!captured && input.servicePromptConfig && input.path.split("?")[0] === "/api/v1/flashcards/decks") {
        captured = true
        return new Promise(resolve => { finish = resolve })
      }
      return original(input)
    })
    mount()
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    await act(async () => { mocks.decks = [{ id: 8, name: "Bob current deck" }]; setUser(2) })
    for (const id of selectorIds) await waitFor(() => expect(screen.getByTestId(id)).toHaveTextContent("Bob current deck"))
    mocks.request.mockImplementation(original)
    await act(async () => { mocks.decks = [{ id: 9, name: "Alice fresh deck" }]; setUser(1) })
    for (const id of selectorIds) await waitFor(() => expect(screen.getByTestId(id)).toHaveTextContent("Alice fresh deck"))
    await act(async () => { finish([{ id: 7, name: "Alice stale private deck" }]) })
    for (const id of selectorIds) expect(screen.getByTestId(id)).toHaveTextContent("Alice fresh deck")
  })

  it("isolates the same user ID on another server", async () => {
    mount()
    for (const id of selectorIds) await selectDeck(id)
    await act(async () => { mocks.decks = [{ id: 7, name: "Other server deck" }]; setUser(1, "https://other.test") })
    for (const id of selectorIds) await waitFor(() => {
      expect(screen.getByTestId(id)).toHaveTextContent("Other server deck")
      expect(screen.getByTestId(id)).not.toHaveTextContent("Alice private deck")
    })
    expect(mocks.request.mock.calls.some(([input]) => input.servicePromptConfig?.serverUrl === "https://other.test" && input.path.startsWith("/api/v1/flashcards/decks"))).toBe(true)
  })

  it("keeps scoped Bob data when an older unscoped deck mutation settles", async () => {
    let finish!: (deck: unknown) => void
    const original = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((input) => input.method === "PATCH" || input.method === "PUT"
      ? new Promise(resolve => { finish = resolve }) : original(input))
    const client = mount()
    for (const id of selectorIds) await selectDeck(id)
    fireEvent.click(screen.getByRole("button", { name: "Legacy update", exact: true }))
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    await act(async () => { mocks.decks = [{ id: 7, name: "Bob current deck" }]; setUser(2) })
    for (const id of selectorIds) await waitFor(() => expect(screen.getByTestId(id)).toHaveTextContent("Bob current deck"))
    mocks.request.mockImplementation(() => new Promise(() => {}))
    await act(async () => { finish({ id: 7, name: "Old Alice update" }) })
    const bobLists = client.getQueriesData<Array<{ name: string }>>({ predicate: query => String(query.queryKey[0]).startsWith("flashcards:decks") })
      .map(([, data]) => data).filter(data => data?.some(deck => deck.name === "Bob current deck"))
    expect(bobLists).toHaveLength(1)
    for (const id of selectorIds) expect(screen.getByTestId(id)).toHaveTextContent("Bob current deck")
  })

  it("preserves a selected deck and draft on a benign same-owner config rotation", async () => {
    mount()
    for (const id of selectorIds) await selectDeck(id)
    fireEvent.change(screen.getByTestId("flashcards-generate-text"), { target: { value: "Current source draft" } })
    await act(async () => { window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } })) })
    expect(screen.getByTestId("flashcards-generate-text")).toHaveValue("Current source draft")
    for (const id of selectorIds) expect(screen.getByTestId(id)).toHaveTextContent("Alice private deck")
  })

  it("drops a removed numeric selection after a successful same-owner empty catalogue", async () => {
    const client = mount()
    for (const id of selectorIds) await selectDeck(id)
    await act(async () => {
      mocks.decks = []
      await client.invalidateQueries({ predicate: query => String(query.queryKey[0]).startsWith("flashcards:decks") })
    })
    for (const id of selectorIds) await waitFor(() => expect(screen.getByTestId(id)).not.toHaveTextContent("Alice private deck"))
  })
  it("a fresh catalogue removes a previously acknowledged created deck", async () => {
    mocks.decks = []
    const original = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation(async input => {
      const path = input.path.split("?")[0]
      if (path === "/api/v1/flashcards/generate") return { flashcards: [{ front: "Question", back: "Answer", model_type: "basic" }], count: 1 }
      if (path === "/api/v1/flashcards/decks" && input.method === "POST") {
        const deck = { id: 12, name: "Generated Flashcards", version: 1 }
        mocks.decks = [deck]
        return deck
      }
      if (path === "/api/v1/flashcards" && input.method === "POST") return { uuid: "00000000-0000-0000-0000-000000000001", version: 1 }
      return original(input)
    })
    const client = mount()
    fireEvent.change(screen.getByTestId("flashcards-generate-text"), { target: { value: "Current owner source" } })
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-button")).toBeEnabled())
    fireEvent.click(screen.getByTestId("flashcards-generate-button"))
    fireEvent.click(await screen.findByTestId("flashcards-generate-save-button"))
    await waitFor(() => expect(mocks.messages.success).toHaveBeenCalled())
    expect(screen.getByTestId("flashcards-generate-deck")).toHaveTextContent("Generated Flashcards")
    await act(async () => {
      mocks.decks = []
      await client.invalidateQueries({ predicate: query => String(query.queryKey[0]).startsWith("flashcards:decks") })
    })
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-deck")).not.toHaveTextContent("Generated Flashcards"))
  })

})
