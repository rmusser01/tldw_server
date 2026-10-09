import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { ConfigProvider } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

const boundary = vi.hoisted(() => ({
  request: vi.fn(), ensureConfig: vi.fn(), user: vi.fn(), config: null as TldwConfig | null
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: boundary.request }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: async () => {}, ensureConfigForRequest: boundary.ensureConfig
} }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: boundary.user } }))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
vi.mock("@/services/tldw-server", () => ({
  LEGACY_SERVICE_PROMPT_DEFAULTS: {}, promptForRag: vi.fn(), getWebSearchPrompt: vi.fn()
}))
vi.mock("wxt/browser", () => ({ browser: { storage: { onChanged: { addListener: vi.fn(), removeListener: vi.fn() } } } }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({
  get: async () => null, set: async () => {}, remove: async () => {}, watch: () => {}, unwatch: () => {}
}), safeStorageSerde: { deserializer: (value: unknown) => value } }))
vi.mock("@/components/Common/MarkdownPreview", () => ({ MarkdownPreview: ({ content }: { content: string }) => <div>{content}</div> }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))

const config = (owner = "alice", serverUrl = "https://keywords.test", revision = "first"): TldwConfig => ({
  serverUrl, authMode: "multi-user", authSource: "manual", accessToken: `test.${btoa(JSON.stringify({ sub: owner }))}.${revision}`
})
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(yes => { resolve = yes })
  return { promise, resolve }
}
let QuickNotesSection: typeof import("../StudioPane/QuickNotesSection").QuickNotesSection
let store: typeof import("@/store/workspace").useWorkspaceStore
const mount = async () => {
  let view!: ReturnType<typeof render>
  await act(async () => { view = render(<ConfigProvider theme={{ token: { motion: false } }}><QuickNotesSection /></ConfigProvider>) })
  return view
}
const keywordCalls = () => boundary.request.mock.calls.filter(([request]) => request.path.startsWith("/api/v1/notes/keywords/"))
const enter = () => fireEvent.change(screen.getByRole("combobox", { name: "Note keywords" }), { target: { value: "private" } })

describe("mounted QuickNotes remount uses real owner-bound keyword reads", () => {
  beforeEach(async () => {
    vi.resetModules()
    vi.clearAllMocks()
    localStorage.clear()
    boundary.config = config()
    boundary.ensureConfig.mockImplementation(async () => boundary.config ? { ...boundary.config } : null)
    boundary.user.mockImplementation(async () => ({ id: JSON.parse(atob(boundary.config!.accessToken!.split(".")[1])).sub }))
    boundary.request.mockImplementation(async ({ path }: { path: string }) => path.startsWith("/api/v1/notes/keywords/") ? ["private-alice"] : [])
    vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Unexpected network operation") }))
    ;({ useWorkspaceStore: store } = await import("@/store/workspace"))
    ;({ QuickNotesSection } = await import("../StudioPane/QuickNotesSection"))
    store.getState().reset()
    store.getState().initializeWorkspace("Retained draft")
    store.getState().setCurrentNote({ title: "Local title", content: "Unsent local body", keywords: ["local-tag"], isDirty: true })
  })
  afterEach(() => vi.unstubAllGlobals())

  it.each(["owner", "server", "credentials", "ABA"])("does not display a retired cached keyword after %s remount", async kind => {
    const first = await mount()
    enter()
    expect(await screen.findByRole("option", { name: "private-alice" })).toBeInTheDocument()
    const draft = store.getState().currentNote
    act(() => {
      boundary.config = kind === "owner" ? config("bob") : kind === "server" ? config("alice", "https://other.test") : kind === "credentials" ? config("alice", "https://keywords.test", "second") : config("bob")
      window.dispatchEvent(new Event("tldw:auth-principal-changed"))
      if (kind === "ABA") {
        boundary.config = config()
        window.dispatchEvent(new Event("tldw:auth-principal-changed"))
      }
    })
    expect(screen.queryByRole("option", { name: "private-alice" })).toBeNull()
    expect(screen.getByRole("combobox", { name: "Note keywords" })).toHaveValue("private")
    expect(store.getState().currentNote).toBe(draft)
    first.unmount()
    boundary.request.mockImplementation(async ({ path }: { path: string }) => path.startsWith("/api/v1/notes/keywords/") ? ["private-current"] : [])
    await mount()
    expect(store.getState().currentNote).toBe(draft)
    enter()
    expect(await screen.findByRole("option", { name: "private-current" })).toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "private-alice" })).toBeNull()
    expect(keywordCalls()).toHaveLength(2)
    expect(keywordCalls()[1][0].configSnapshot).toEqual(boundary.config)
    expect(keywordCalls()[1][0].headers["X-TLDW-Expected-User-ID"]).toBe(kind === "owner" ? "bob" : "alice")
    expect(screen.getByRole("textbox", { name: "Note content" })).toHaveValue("Unsent local body")
    expect(fetch).not.toHaveBeenCalled()
  })

  it("never accepts or retains a late retired reply for a remounted owner", async () => {
    const gate = deferred<string[]>()
    boundary.request.mockImplementation(({ path }: { path: string }) => path.startsWith("/api/v1/notes/keywords/") ? gate.promise : [])
    const first = await mount()
    await waitFor(() => expect(keywordCalls()).toHaveLength(1))
    act(() => {
      boundary.config = config("bob")
      window.dispatchEvent(new Event("tldw:auth-principal-changed"))
    })
    first.unmount()
    boundary.request.mockImplementation(async ({ path }: { path: string }) => path.startsWith("/api/v1/notes/keywords/") ? ["private-bob"] : [])
    const second = await mount()
    enter()
    try {
      expect(await screen.findByRole("option", { name: "private-bob" })).toBeInTheDocument()
    } finally {
      await act(async () => { gate.resolve(["private-alice"]); await gate.promise })
    }
    expect(screen.queryByRole("option", { name: "private-alice" })).toBeNull()
    expect(keywordCalls()[0][0].abortSignal.aborted).toBe(true)
    second.unmount()
    await mount()
    enter()
    expect(await screen.findByRole("option", { name: "private-bob" })).toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "private-alice" })).toBeNull()
    expect(keywordCalls()).toHaveLength(3)
  })
})
