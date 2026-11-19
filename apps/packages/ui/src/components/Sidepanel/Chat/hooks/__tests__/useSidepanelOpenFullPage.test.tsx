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
  resolveSidepanelFullPageTarget,
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

describe("resolveSidepanelFullPageTarget", () => {
  it("opens the extension's own full page when no WebUI URL is configured", async () => {
    mocks.storage = { tldwConfig: { serverUrl: "http://127.0.0.1:8000" } }
    await expect(resolveSidepanelFullPageTarget()).resolves.toBe("extension")
  })

  it("opens the WebUI the user configured", async () => {
    mocks.storage = { tldwConfig: { webUiUrl: "https://notes.example.test" } }
    await expect(resolveSidepanelFullPageTarget()).resolves.toBe("webui")
  })

  it("reads the legacy WebUI URL setting too", async () => {
    mocks.storage = { webuiUrl: "http://localhost:8080" }
    await expect(resolveSidepanelFullPageTarget()).resolves.toBe("webui")
  })

  it("always opens the WebUI from the web-hosted panel", async () => {
    mocks.runtimeId = undefined
    await expect(resolveSidepanelFullPageTarget()).resolves.toBe("webui")
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

  it("opens a new character chat on its character route", async () => {
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

  it("opens a configured WebUI instead of the extension page", async () => {
    mocks.storage = { tldwConfig: { webUiUrl: "https://webui.example.test" } }
    const openInWebUi = vi.fn(async () => true)
    const { result } = renderHookWith(baseDeps({ draftText: "Draft", openInWebUi }))
    await waitFor(() => expect(result.current.target).toBe("webui"))
    await act(async () => { await result.current.open() })
    expect(openInWebUi).toHaveBeenCalledOnce()
    expect(mocks.tabsCreate).not.toHaveBeenCalled()
    expect(mocks.createSidepanelChatHandoff).not.toHaveBeenCalled()
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
