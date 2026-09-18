import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import SidepanelChat from "../sidepanel-chat"
import { loadFlashcardsTransferSnapshot, flashcardsHandoffAuthority } from "@/services/tldw/flashcards-generate-transfer"
import { consumeFlashcardsGenerateHandoff } from "@/services/tldw/flashcards-generate-handoff"

const mocks = vi.hoisted(() => {
  const noop = () => {}
  const message = new Proxy({ messages: [], history: [], queuedMessages: [], serverChatId: "conversation-41" }, {
    get: (target, key) => key in target ? Reflect.get(target, key) : String(key).startsWith("set") || ["clearChat", "stopStreamingRequest", "onSubmit"].includes(String(key)) ? noop : null
  })
  const tabs = new Proxy({ tabs: [{ id: "local-tab-99", label: "Local tab" }], activeTabId: "local-tab-99", snapshotsById: {} }, {
    get: (target, key) => key in target ? Reflect.get(target, key) : noop
  })
  return { noop, message, tabs, background: null as null | { type: string; text: string; payload: Record<string, unknown> }, initialize: vi.fn(), navigate: vi.fn() }
})
vi.mock("@/hooks/useBackgroundMessage", () => ({ default: () => mocks.background }))
vi.mock("@/hooks/useMessage", () => ({ useMessage: () => mocks.message }))
vi.mock("@/hooks/useMigration", () => ({ useMigration: () => {} }))
vi.mock("@/hooks/useSmartScroll", () => ({ useSmartScroll: () => ({ containerRef: { current: null }, autoScrollToBottom: mocks.noop }) }))
vi.mock("@/hooks/keyboard/useKeyboardShortcuts", () => ({ useChatShortcuts: () => {}, useSidebarShortcuts: () => {}, useChatModeShortcuts: () => {}, useWebSearchShortcuts: () => {} }))
vi.mock("@/hooks/useConnectionState", () => ({ useConnectionActions: () => ({ checkOnce: mocks.noop }) }))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => {} }))
vi.mock("@/hooks/useAntdNotification", () => ({ useAntdNotification: () => ({ warning: mocks.noop, error: mocks.noop }) }))
vi.mock("@/hooks/useCharacterGreeting", () => ({ useCharacterGreeting: () => {} }))
vi.mock("@/hooks/useTTS", () => ({ useTTS: () => ({ cancel: mocks.noop }) }))
vi.mock("@/hooks/useSelectedCharacter", () => ({ useSelectedCharacter: () => [null] }))
vi.mock("@/hooks/useSelectedAssistant", () => ({ useSelectedAssistant: () => [null] }))
vi.mock("@/hooks/useSetting", () => ({ useSetting: () => [100] }))
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: (_key: unknown, fallback: unknown) => [fallback] }))
vi.mock("@/store/sidepanel-chat-tabs", () => ({ useSidepanelChatTabsStore: Object.assign((selector: (state: unknown) => unknown) => selector(mocks.tabs), { getState: () => mocks.tabs }) }))
vi.mock("@/store/model", () => ({ useStoreChatModelSettings: Object.assign((selector: (state: unknown) => unknown) => selector({}), { getState: () => ({}) }) }))
vi.mock("@/store/option", () => ({ useStoreMessageOption: (selector: (state: unknown) => unknown) => selector({ setRagMediaIds: mocks.noop }) }))
vi.mock("@/store/ui-mode", () => ({ useUiModeStore: (selector: (state: unknown) => unknown) => selector({ mode: "simple" }) }))
vi.mock("@/store/artifacts", () => ({ useArtifactsStore: (selector: (state: unknown) => unknown) => selector({ isOpen: false, closeArtifact: mocks.noop }) }))
vi.mock("@/db/dexie/helpers", () => ({ generateID: () => "new-local-tab", getTitleById: async () => "", getRecentChatFromCopilot: async () => null }))
vi.mock("@/services/app", () => ({ copilotResumeLastChat: async () => false }))
vi.mock("@/services/web-clipper/enrichment", () => ({ readPendingWebClipAnalyzeRequest: () => null, clearPendingWebClipAnalyzeRequest: () => {} }))
vi.mock("@/components/Sidepanel/Chat/body", () => ({ SidePanelBody: () => null }))
vi.mock("@/components/Sidepanel/Chat/form", () => ({ SidepanelForm: () => null }))
vi.mock("@/components/Sidepanel/Chat/SidepanelHeaderSimple", () => ({ SidepanelHeaderSimple: () => null }))
vi.mock("@/components/Sidepanel/Chat/ConnectionBanner", () => ({ ConnectionBanner: () => null }))
vi.mock("@/components/Common/CommandPaletteHost", () => ({ CommandPaletteHost: () => null }))
vi.mock("@/components/Timeline", () => ({ TimelineModal: () => null }))
vi.mock("@/components/Sidepanel/Notes/NoteQuickSaveModal", () => ({ default: (props: {
  onGenerateFlashcards: () => void; onCancel: () => void; onContentChange: (value: string) => void; content: string; error?: string
}) => <section><textarea aria-label="Captured draft" value={props.content} onChange={event => props.onContentChange(event.target.value)} /><button onClick={props.onGenerateFlashcards}>Generate flashcards</button><button onClick={props.onCancel}>Cancel capture</button>{props.error && <p role="alert">{props.error}</p>}</section> }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, fallback?: string) => fallback || key }) }))
vi.mock("@plasmohq/storage", async () => import("../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
vi.mock("@/utils/browser-runtime", () => ({ isExtensionRuntime: () => false }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { initialize: mocks.initialize, ensureConfigForRequest: async () => JSON.parse(window.localStorage.getItem("tldwConfig") || "null") } }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: async () => ({ id: 1, is_active: true }) } }))
vi.mock("wxt/browser", () => ({ browser: { runtime: {} } }))

describe("actual sidepanel Chat private Flashcards producer", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.initialize.mockReset().mockResolvedValue(undefined)
    mocks.background = null
    mocks.message.serverChatId = "conversation-41"
    window.localStorage.clear()
    window.localStorage.setItem("tldwConfig", JSON.stringify({ serverUrl: "https://chat.test", authMode: "single-user", apiKey: "synthetic-key" }))
    let tail = Promise.resolve()
    const locks = { request: (_name: string, work: () => unknown) => { const next = tail.then(work); tail = next.then(() => undefined, () => undefined); return next } }
    vi.stubGlobal("navigator", new Proxy(window.navigator, { get: (target, key) => key === "locks" ? locks : Reflect.get(target, key, target) }))
    vi.spyOn(window, "open").mockReturnValue({ closed: false, opener: null, close: vi.fn(), location: { replace: mocks.navigate } } as unknown as Window)
  })
  afterEach(() => vi.unstubAllGlobals())

  it.each(["message-23", 23, undefined])("stores captured message provenance without the local tab ID (%s)", async messageId => {
    const source = render(<SidepanelChat />)
    mocks.background = { type: "save-to-notes", text: "Captured private text", payload: { messageId, pageTitle: "Private title" } }
    source.rerender(<SidepanelChat />)
    await screen.findByRole("button", { name: "Generate flashcards" })
    // The selection remains associated with its original conversation even if
    // the surrounding sidepanel has switched chats before the explicit action.
    mocks.message.serverChatId = "different-conversation"
    source.rerender(<SidepanelChat />)
    fireEvent.change(screen.getByLabelText("Captured draft"), { target: { value: " \nUnsaved exact text\t " } })
    fireEvent.click(screen.getByRole("button", { name: "Generate flashcards" }))
    await waitFor(() => expect(mocks.navigate).toHaveBeenCalledTimes(1))
    await act(async () => {})
    const route = new URL(mocks.navigate.mock.calls[0][0])
    expect(route.href).not.toMatch(/private|message-23|local-tab|conversation-41/i)
    const snapshot = await loadFlashcardsTransferSnapshot()
    try {
      const intent = await consumeFlashcardsGenerateHandoff(route.searchParams.get("generate_handoff")!, flashcardsHandoffAuthority(snapshot))
      expect(intent).toMatchObject({ text: " \nUnsaved exact text\t " })
      expect(intent.sourceId).toBe(messageId == null ? undefined : String(messageId))
      expect(intent.messageId).toBe(messageId == null ? undefined : String(messageId))
      expect(intent.sourceType).toBe(messageId == null ? "manual" : "message")
      expect(intent.conversationId).toBe(messageId == null ? undefined : "conversation-41")
    } finally { snapshot.release() }
    expect(screen.queryByLabelText("Captured draft")).not.toBeInTheDocument()
  })

  it("retains the captured draft when the destination is blocked", async () => {
    vi.spyOn(window, "open").mockReturnValue(null)
    mocks.background = { type: "save-to-notes", text: "Private source", payload: { messageId: "message-23" } }
    render(<SidepanelChat />)
    fireEvent.click(await screen.findByRole("button", { name: "Generate flashcards" }))
    expect(await screen.findByRole("alert")).toHaveTextContent(/popup was blocked/)
    expect(screen.getByLabelText("Captured draft")).toHaveValue("Private source")
    expect(mocks.navigate).not.toHaveBeenCalled()
  })

  it.each(["same-tab", "cross-tab", "A-B-A"]) ("clears a captured private draft before a later transfer under another owner (%s)", async boundary => {
    mocks.background = { type: "save-to-notes", text: "Owner A private source", payload: { messageId: "message-23" } }
    render(<SidepanelChat />)
    await screen.findByLabelText("Captured draft")
    const original = window.localStorage.getItem("tldwConfig")!
    const other = JSON.stringify({ serverUrl: "https://chat.test", authMode: "single-user", apiKey: "other-owner" })
    act(() => {
      window.localStorage.setItem("tldwConfig", other)
      if (boundary === "same-tab") window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
      else window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: original, newValue: other }))
      if (boundary === "A-B-A") {
        window.localStorage.setItem("tldwConfig", original)
        window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: other, newValue: original }))
      }
    })
    await waitFor(() => expect(screen.queryByLabelText("Captured draft")).not.toBeInTheDocument())
    expect(mocks.navigate).not.toHaveBeenCalled()
  })

  it("clears an unresolved capture immediately on a principal change and ignores its late resolution", async () => {
    let resolve!: () => void
    mocks.initialize.mockResolvedValueOnce(undefined).mockReturnValueOnce(new Promise<void>(done => { resolve = done }))
    mocks.background = { type: "save-to-notes", text: "Unresolved private source", payload: { messageId: "message-23" } }
    render(<SidepanelChat />)
    await screen.findByLabelText("Captured draft")
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")))
    expect(screen.queryByLabelText("Captured draft")).not.toBeInTheDocument()
    await act(async () => resolve?.())
    expect(screen.queryByLabelText("Captured draft")).not.toBeInTheDocument()
    expect(mocks.navigate).not.toHaveBeenCalled()
  })

  it("preserves a verified capture through same-principal refresh", async () => {
    const original = { serverUrl: "https://chat.test", authMode: "multi-user", accessToken: `test.${btoa(JSON.stringify({ sub: "1" }))}.old` }
    window.localStorage.setItem("tldwConfig", JSON.stringify(original))
    mocks.background = { type: "save-to-notes", text: "Same owner source", payload: { messageId: "message-23" } }
    render(<SidepanelChat />)
    await screen.findByLabelText("Captured draft")
    await act(async () => {})
    const rotated = { ...original, accessToken: `test.${btoa(JSON.stringify({ sub: "1" }))}.rotated` }
    act(() => {
      window.localStorage.setItem("tldwConfig", JSON.stringify(rotated))
      window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: JSON.stringify(original), newValue: JSON.stringify(rotated) }))
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } }))
    })
    expect(screen.getByLabelText("Captured draft")).toHaveValue("Same owner source")
    fireEvent.click(screen.getByRole("button", { name: "Generate flashcards" }))
    await waitFor(() => expect(mocks.navigate).toHaveBeenCalledTimes(1))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("retains an unverified draft but refuses to transfer it under later credentials", async () => {
    mocks.initialize.mockResolvedValueOnce(undefined).mockRejectedValueOnce(new Error("Synthetic connection failure"))
    mocks.background = { type: "save-to-notes", text: "Unverified source", payload: { messageId: "message-23" } }
    render(<SidepanelChat />)
    expect(await screen.findByRole("alert")).toHaveTextContent(/account could not be verified/)
    fireEvent.click(screen.getByRole("button", { name: "Generate flashcards" }))
    await act(async () => {})
    expect(screen.getByLabelText("Captured draft")).toHaveValue("Unverified source")
    expect(mocks.navigate).not.toHaveBeenCalled()
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } })))
    expect(screen.queryByLabelText("Captured draft")).not.toBeInTheDocument()
  })

  it("does not let a cancelled capture's delayed resolver replace a newer draft", async () => {
    let resolve!: () => void
    mocks.initialize.mockResolvedValueOnce(undefined).mockReturnValueOnce(new Promise<void>(done => { resolve = done }))
    mocks.background = { type: "save-to-notes", text: "Old source", payload: { messageId: "old-message" } }
    const source = render(<SidepanelChat />)
    fireEvent.click(await screen.findByRole("button", { name: "Cancel capture" }))
    mocks.background = { type: "save-to-notes", text: "New source", payload: { messageId: "new-message" } }
    source.rerender(<SidepanelChat />)
    await screen.findByDisplayValue("New source")
    await act(async () => resolve())
    expect(screen.getByLabelText("Captured draft")).toHaveValue("New source")
    fireEvent.click(screen.getByRole("button", { name: "Generate flashcards" }))
    await waitFor(() => expect(mocks.navigate).toHaveBeenCalledTimes(1))
    const snapshot = await loadFlashcardsTransferSnapshot()
    try {
      const token = new URL(mocks.navigate.mock.calls[0][0]).searchParams.get("generate_handoff")!
      expect(await consumeFlashcardsGenerateHandoff(token, flashcardsHandoffAuthority(snapshot))).toMatchObject({ text: "New source", sourceId: "new-message" })
    } finally { snapshot.release() }
  })
})
