// @vitest-environment jsdom
/**
 * XP-09 (#3101): the side panel has one "Open in full page" action. It takes
 * the open chat, the draft and page references to the extension's full page,
 * or to the WebUI when the user has configured one.
 */
import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  runtimeId: "extension-id" as string | undefined,
  tabsCreate: vi.fn(async (_options: { url: string }) => ({})),
  storage: {} as Record<string, unknown>,
  prepareExpansionPath: vi.fn(async (): Promise<string | null> => "/chat"),
  historySelection: true,
  selectedAssistant: null as null | { kind: "character"; id: string; name: string },
  createSidepanelChatHandoff: vi.fn(),
  consumeSidepanelChatHandoff: vi.fn(async () => null),
  notification: { error: vi.fn(), warning: vi.fn(), info: vi.fn() }
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() { return mocks.runtimeId },
      getURL: (path: string) => `chrome-extension://ext${path}`
    },
    tabs: { create: mocks.tabsCreate }
  }
}))

vi.mock("@/utils/safe-storage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: () => ({ get: async (key: string) => mocks.storage[key] })
}))

vi.mock("@/hooks/chat/useHistorySelection", () => ({
  useHistorySelectionContext: () =>
    mocks.historySelection ? { prepareExpansionPath: mocks.prepareExpansionPath } : null
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [mocks.selectedAssistant, vi.fn()] as const
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => mocks.notification
}))

vi.mock("@/services/sidepanel-chat-handoff", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/sidepanel-chat-handoff")>()
  return {
    ...actual,
    createSidepanelChatHandoff: mocks.createSidepanelChatHandoff,
    consumeSidepanelChatHandoff: mocks.consumeSidepanelChatHandoff
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback?: string) => fallback ?? _key })
}))

import {
  readConfiguredWebUiUrl,
  useSidepanelOpenFullPage,
  type SidepanelOpenFullPageDeps
} from "../useSidepanelOpenFullPage"

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

const baseDeps = (overrides: Partial<SidepanelOpenFullPageDeps> = {}): SidepanelOpenFullPageDeps => ({
  draftReady: true,
  draftText: "",
  handoffSourceKey: "source-1",
  captureHandoffGuard: () => () => true,
  hasVisiblePageContext: false,
  selectedCharacterId: null,
  ...overrides
})

const renderHookWith = (deps: SidepanelOpenFullPageDeps) =>
  renderHook(() => useSidepanelOpenFullPage(deps))

const openedUrl = () => mocks.tabsCreate.mock.calls[0]?.[0]?.url ?? ""

beforeEach(() => {
  vi.clearAllMocks()
  mocks.runtimeId = "extension-id"
  mocks.storage = {}
  mocks.historySelection = true
  mocks.selectedAssistant = null
  mocks.prepareExpansionPath.mockResolvedValue("/chat")
  mocks.createSidepanelChatHandoff.mockResolvedValue({
    id: "pkg-1",
    source: "sidepanel-chat",
    createdAt: "2026-11-19T00:00:00.000Z",
    expiresAt: "2026-11-19T00:10:00.000Z",
    draft: { text: "My draft" }
  })
})

describe("readConfiguredWebUiUrl", () => {
  it("is null when no WebUI URL is configured", async () => {
    mocks.storage = { tldwConfig: { serverUrl: "http://127.0.0.1:8000" } }
    await expect(readConfiguredWebUiUrl()).resolves.toBeNull()
  })

  it("reads the WebUI the user configured, and the legacy setting", async () => {
    mocks.storage = { tldwConfig: { webUiUrl: "https://notes.example.test/" } }
    await expect(readConfiguredWebUiUrl()).resolves.toBe("https://notes.example.test")
    mocks.storage = { webuiUrl: "http://localhost:8080" }
    await expect(readConfiguredWebUiUrl()).resolves.toBe("http://localhost:8080")
  })

  it("ignores an invalid one rather than guessing from the server URL", async () => {
    mocks.storage = { tldwConfig: { webUiUrl: "not a url", serverUrl: "http://127.0.0.1:8000" } }
    await expect(readConfiguredWebUiUrl()).resolves.toBeNull()
  })
})

describe("useSidepanelOpenFullPage", () => {
  it("names the one action 'Open in full page'", () => {
    const { result } = renderHookWith(baseDeps())
    expect(result.current.label).toBe("Open in full page")
  })

  it("carries the open chat and the draft to the extension's full page", async () => {
    mocks.prepareExpansionPath.mockResolvedValue("/chat?historySelection=REF")
    const { result } = renderHookWith(baseDeps({ draftText: "My draft  " }))
    let opened = false
    await act(async () => { opened = await result.current.open() })
    expect(opened).toBe(true)
    expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledWith(
      expect.objectContaining({ draftText: "My draft  ", routeIntent: { path: "/chat" } }),
      expect.any(Function)
    )
    expect(openedUrl()).toBe(
      "chrome-extension://ext/options.html#/chat?historySelection=REF&handoff=pkg-1"
    )
  })

  it("opens the open chat without a copy when there is no draft or page", async () => {
    mocks.prepareExpansionPath.mockResolvedValue("/chat?historySelection=REF")
    const { result } = renderHookWith(baseDeps())
    await act(async () => { await result.current.open() })
    expect(mocks.createSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(openedUrl()).toBe("chrome-extension://ext/options.html#/chat?historySelection=REF")
  })

  it("carries page references even without a draft", async () => {
    const pageContext = { title: "Docs page", url: "https://docs.test", snippets: [] }
    const { result } = renderHookWith(baseDeps({ hasVisiblePageContext: true, getVisiblePageContext: () => pageContext }))
    await act(async () => { await result.current.open() })
    expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledWith(
      expect.objectContaining({ draftText: "", pageContext }),
      expect.any(Function)
    )
    expect(openedUrl()).toBe("chrome-extension://ext/options.html#/chat?handoff=pkg-1")
  })

  it("opens a new Character chat on its character route", async () => {
    mocks.selectedAssistant = { kind: "character", id: "42", name: "Ada" }
    const { result } = renderHookWith(baseDeps({ draftText: "Hello Ada" }))
    await act(async () => { await result.current.open() })
    expect(openedUrl()).toContain("options.html#/chat?mode=character")
    expect(openedUrl()).toContain("characterId=42")
    // A copy would retire the character route on arrival, so none is made.
    expect(openedUrl()).not.toContain("handoff=")
    expect(mocks.createSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(mocks.notification.warning).toHaveBeenCalledOnce()
  })

  // Coordinator's decision: in the extension "Open in full page" always opens
  // the extension's page (the one target that carries the chat); a configured
  // WebUI is a separate "Open in WebUI" action.
  it("keeps the extension's page for Open in full page when a WebUI is configured", async () => {
    mocks.storage = { tldwConfig: { webUiUrl: "https://webui.example.test" } }
    const openInWebUi = vi.fn(async () => true)
    const { result } = renderHookWith(baseDeps({ draftText: "Draft", openInWebUi }))
    await waitFor(() => expect(result.current.webUi).not.toBeNull())
    await act(async () => { await result.current.open() })
    expect(openInWebUi).not.toHaveBeenCalled()
    expect(openedUrl()).toContain("chrome-extension://ext/options.html#/chat")
  })

  it("offers Open in WebUI, with the configured URL and accurate copy", async () => {
    mocks.storage = { tldwConfig: { webUiUrl: "https://webui.example.test/", serverUrl: "http://127.0.0.1:8000" } }
    const openInWebUi = vi.fn(async (_url?: string) => true)
    const { result } = renderHookWith(baseDeps({ draftText: "Draft", openInWebUi }))
    await waitFor(() => expect(result.current.webUi).not.toBeNull())
    expect(result.current.webUi?.label).toBe("Open in WebUI")
    expect(result.current.webUi?.description).toBe(
      "Opens the WebUI with your draft. You can bring this chat's settings along there. This chat and its attachments stay here."
    )
    await act(async () => { await result.current.webUi!.open() })
    expect(openInWebUi).toHaveBeenCalledWith("https://webui.example.test")
  })

  it("has no Open in WebUI when none is configured", async () => {
    const { result } = renderHookWith(baseDeps({ openInWebUi: vi.fn(async () => true) }))
    await act(async () => {})
    expect(result.current.webUi).toBeNull()
  })

  it("describes the web-hosted panel's hand-off accurately", () => {
    mocks.runtimeId = undefined
    const { result } = renderHookWith(baseDeps({ openInWebUi: vi.fn(async () => true) }))
    expect(result.current.description).toBe(
      "Opens the full page with your draft. You can bring this chat's settings along there. This chat and its attachments stay here."
    )
  })

  it("does not say \"this chat\" for a new or temporary chat", () => {
    const { result } = renderHookWith(baseDeps({ hasConversation: false }))
    expect(result.current.description).toBe(
      "Opens the chat page in a new tab, with your draft and page references. Your draft also stays here."
    )
  })

  it("warns that attachments stay in the side panel", async () => {
    const { result } = renderHookWith(baseDeps({ draftText: "Look", hasAttachments: true }))
    await act(async () => { await result.current.open() })
    expect(mocks.tabsCreate).toHaveBeenCalledOnce()
    expect(mocks.notification.warning).toHaveBeenCalledWith({
      message: "Attachments stay in the side panel. Add them again on the full page."
    })
  })

  it("names a Persona chat as one", async () => {
    mocks.selectedAssistant = { kind: "persona", id: "p1", name: "Guide" } as unknown as typeof mocks.selectedAssistant
    const { result } = renderHookWith(baseDeps({ draftText: "Hi" }))
    await act(async () => { await result.current.open() })
    expect(mocks.notification.warning).toHaveBeenCalledWith({
      message: "A new Persona chat opens without your draft. It stays here."
    })
  })

  it("explains why it can't open while the draft is being checked", () => {
    const { result } = renderHookWith(baseDeps({ draftReady: false, draftOwnerState: "pending" }))
    expect(result.current.disabledReason).toBe("Checking which account this draft belongs to…")
    const failed = renderHookWith(baseDeps({ draftReady: false, draftOwnerState: "error" }))
    expect(failed.result.current.disabledReason).toBe("This draft's account couldn't be checked. Retry from the draft notice.")
  })

  it("says so when the WebUI could not be opened", async () => {
    mocks.runtimeId = undefined
    const openInWebUi = vi.fn(async () => { throw new Error("Window popup was blocked") })
    const { result } = renderHookWith(baseDeps({ openInWebUi }))
    await act(async () => { await result.current.open() })
    expect(mocks.notification.error).toHaveBeenCalledOnce()
  })

  it("does nothing before the draft is restored", async () => {
    const { result } = renderHookWith(baseDeps({ draftReady: false }))
    expect(result.current.disabled).toBe(true)
    await act(async () => { await result.current.open() })
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
  })

  it("does not open a copy made for a source that has since changed", async () => {
    const held = deferred<unknown>()
    mocks.createSidepanelChatHandoff.mockReturnValueOnce(held.promise)
    let current = true
    const { result } = renderHookWith(baseDeps({ draftText: "Old", captureHandoffGuard: () => () => current }))
    let pending!: Promise<boolean>
    act(() => { pending = result.current.open() })
    await waitFor(() => expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledOnce())
    current = false
    await act(async () => { held.resolve({ id: "stale", draft: { text: "Old" } }); await pending })
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
    expect(mocks.consumeSidepanelChatHandoff).toHaveBeenCalledWith("stale")
  })

  it("ignores a second click while the first is still opening", async () => {
    const held = deferred<unknown>()
    mocks.createSidepanelChatHandoff.mockReturnValueOnce(held.promise)
    const { result } = renderHookWith(baseDeps({ draftText: "Once" }))
    let first!: Promise<boolean>
    act(() => { first = result.current.open() })
    await waitFor(() => expect(result.current.pending).toBe(true))
    await act(async () => { await result.current.open() })
    await act(async () => { held.resolve({ id: "pkg-1", draft: { text: "Once" } }); await first })
    expect(mocks.createSidepanelChatHandoff).toHaveBeenCalledOnce()
    expect(mocks.tabsCreate).toHaveBeenCalledOnce()
  })

  it("withdraws the copy and says so when the tab cannot open", async () => {
    mocks.tabsCreate.mockRejectedValueOnce(new Error("no tabs"))
    vi.spyOn(window, "open").mockReturnValueOnce(null)
    const { result } = renderHookWith(baseDeps({ draftText: "Keep me" }))
    let opened = true
    await act(async () => { opened = await result.current.open() })
    expect(opened).toBe(false)
    expect(mocks.consumeSidepanelChatHandoff).toHaveBeenCalledWith("pkg-1", expect.any(Function))
    expect(mocks.notification.error).toHaveBeenCalledOnce()
  })

  it("asks the user to wait while the chat is still restoring", async () => {
    mocks.prepareExpansionPath.mockResolvedValue(null)
    const { result } = renderHookWith(baseDeps({ draftText: "Draft" }))
    await act(async () => { await result.current.open() })
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
    expect(mocks.notification.error).toHaveBeenCalledOnce()
  })
})
