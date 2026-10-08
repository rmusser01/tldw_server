// @vitest-environment jsdom
import React from "react"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useClearChat } from "../useClearChat"
import {
  SETTINGS_NAVIGATION_REQUEST_EVENT,
  type SettingsNavigationRequestDetail
} from "@/utils/settings-return"

const navigateMock = vi.hoisted(() => vi.fn())
const destroyAllMock = vi.hoisted(() => vi.fn())
const cleanupOverlaysMock = vi.hoisted(() => vi.fn())
const updatePageTitleMock = vi.hoisted(() => vi.fn())
const focusTextAreaMock = vi.hoisted(() => vi.fn())
const resetModelSettingsMock = vi.hoisted(() => vi.fn())
const clearSessionMock = vi.hoisted(() => vi.fn())
const optionStoreSetStateMock = vi.hoisted(() => vi.fn())
const historySelectionResetMock = vi.hoisted(() => vi.fn())
const historySelectionContextMock = vi.hoisted(() => vi.fn())

const baseState = vi.hoisted(() => ({
  setMessages: vi.fn(),
  setHistory: vi.fn(),
  setHistoryId: vi.fn(),
  setIsFirstMessage: vi.fn(),
  setIsLoading: vi.fn(),
  setIsProcessing: vi.fn(),
  setStreaming: vi.fn()
}))

const optionState = vi.hoisted(() => ({
  temporaryChat: false,
  messages: [] as unknown[],
  history: [] as unknown[],
  serverChatId: "saved" as string | null,
  historyId: "local" as string | null,
  chatDraftAddress: null,
  chatDraftRevision: 1,
  startNewChatDraft: vi.fn(),
  setServerChatId: vi.fn(),
  setServerChatVersion: vi.fn(),
  setContextFiles: vi.fn(),
  setDocumentContext: vi.fn(),
  setUploadedFiles: vi.fn(),
  setFileRetrievalEnabled: vi.fn(),
  setActionInfo: vi.fn(),
  setRagMediaIds: vi.fn(),
  setRagSearchMode: vi.fn(),
  setRagTopK: vi.fn(),
  setRagEnableGeneration: vi.fn(),
  setRagEnableCitations: vi.fn(),
  setRagSources: vi.fn(),
  clearQueuedMessages: vi.fn(),
  setCompareMode: vi.fn(),
  setCompareSelectedModels: vi.fn(),
  clearReplyTarget: vi.fn(),
  setWebSearch: vi.fn()
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock
}))

vi.mock("antd", () => ({
  Modal: { destroyAll: destroyAllMock }
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [false]
}))

vi.mock("@/hooks/chat/useChatBaseState", () => ({
  useChatBaseState: () => baseState
}))

vi.mock("@/hooks/chat/useHistorySelection", () => ({
  useHistorySelectionContext: historySelectionContextMock
}))

vi.mock("@/hooks/utils/messageHelpers", () => ({
  focusTextArea: focusTextAreaMock
}))

vi.mock("@/store/option", () => {
  const useStoreMessageOption = Object.assign(
    (selector: (state: typeof optionState) => unknown) => selector(optionState),
    { setState: optionStoreSetStateMock, getState: () => ({ ...optionState }) }
  )
  return { useStoreMessageOption }
})

vi.mock("@/store", () => ({
  useStoreMessage: vi.fn()
}))

vi.mock("@/store/playground-session", () => ({
  usePlaygroundSessionStore: {
    getState: () => ({ clearSession: clearSessionMock, restoreRevision: 0 })
  }
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => ({ reset: resetModelSettingsMock })
}))

vi.mock("@/utils/cleanup-ant-overlays", () => ({
  cleanupAntOverlays: cleanupOverlaysMock
}))

vi.mock("@/utils/update-page-title", () => ({
  updatePageTitle: updatePageTitleMock
}))

describe("useClearChat settings navigation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    Object.assign(optionState, { temporaryChat: false, messages: [], history: [], serverChatId: "saved", historyId: "local", chatDraftAddress: null, chatDraftRevision: 1 })
    vi.spyOn(window, "confirm").mockReturnValue(false)
    historySelectionContextMock.mockReturnValue({ reset: historySelectionResetMock })
    window.history.replaceState({}, "", "/settings/prompt")
  })

  afterEach(() => vi.restoreAllMocks())

  it("does nothing when the mounted settings editor declines navigation", () => {
    const declineNavigation = vi.fn((event: Event) => event.preventDefault())
    window.addEventListener(
      SETTINGS_NAVIGATION_REQUEST_EVENT,
      declineNavigation,
      { once: true }
    )
    const { result } = renderHook(() => useClearChat())

    let accepted: boolean | undefined
    act(() => { accepted = result.current() })
    expect(accepted).toBe(false)

    expect(declineNavigation).toHaveBeenCalledOnce()
    expect((declineNavigation.mock.calls[0][0] as CustomEvent<
      SettingsNavigationRequestDetail
    >).detail).toEqual({ destination: "/chat" })
    expect(navigateMock).not.toHaveBeenCalled()
    expect(destroyAllMock).not.toHaveBeenCalled()
    expect(cleanupOverlaysMock).not.toHaveBeenCalled()
    expect(baseState.setMessages).not.toHaveBeenCalled()
    expect(optionState.setServerChatId).not.toHaveBeenCalled()
    expect(optionState.startNewChatDraft).not.toHaveBeenCalled()
    expect(resetModelSettingsMock).not.toHaveBeenCalled()
    expect(updatePageTitleMock).not.toHaveBeenCalled()
    expect(focusTextAreaMock).not.toHaveBeenCalled()
    expect(clearSessionMock).not.toHaveBeenCalled()
    expect(historySelectionResetMock).not.toHaveBeenCalled()
  })

  it("navigates and resets once when navigation is allowed", () => {
    const { result } = renderHook(() => useClearChat())

    act(() => expect(result.current()).toBe(true))

    expect(navigateMock).toHaveBeenCalledOnce()
    expect(navigateMock).toHaveBeenCalledWith("/chat")
    expect(destroyAllMock).toHaveBeenCalledOnce()
    expect(cleanupOverlaysMock).toHaveBeenCalledOnce()
    expect(optionState.setServerChatId).toHaveBeenCalledOnce()
    expect(optionState.startNewChatDraft).toHaveBeenCalledOnce()
    expect(resetModelSettingsMock).toHaveBeenCalledOnce()
    expect(updatePageTitleMock).toHaveBeenCalledOnce()
    expect(focusTextAreaMock).toHaveBeenCalledOnce()
    expect(clearSessionMock).toHaveBeenCalledOnce()
    expect(historySelectionResetMock).toHaveBeenCalledOnce()
    expect(historySelectionResetMock.mock.invocationCallOrder[0]).toBeLessThan(
      navigateMock.mock.invocationCallOrder[0]
    )
  })

  it("still clears standalone surfaces without a history selection provider", () => {
    historySelectionContextMock.mockReturnValue(null)
    const { result } = renderHook(() => useClearChat())

    act(() => expect(result.current()).toBe(true))

    expect(navigateMock).toHaveBeenCalledWith("/chat")
    expect(clearSessionMock).toHaveBeenCalledOnce()
    expect(historySelectionResetMock).not.toHaveBeenCalled()
  })
  const temporarySource = () => {
    Object.assign(optionState, {
      temporaryChat: true, serverChatId: null, historyId: "temp",
      messages: [{ id: "temporary-q", message: "Keep this exact question", images: ["data:image/png;base64,aA=="] }],
      history: [{ role: "user", content: "Keep this exact question" }]
    })
  }

  it.each(["both", "display", "history"])("keeps populated Temporary %s content and every reset side effect on Cancel", kind => {
    temporarySource()
    if (kind === "display") optionState.history = []
    if (kind === "history") optionState.messages = []
    const before = { ...optionState }
    const { result } = renderHook(() => useClearChat())
    act(() => expect(result.current()).toBe(false))
    expect(optionState).toEqual(before)
    expect(window.confirm).toHaveBeenCalledOnce()
    expect(navigateMock).not.toHaveBeenCalled()
    expect(historySelectionResetMock).not.toHaveBeenCalled()
    expect(optionState.startNewChatDraft).not.toHaveBeenCalled()
    expect(baseState.setMessages).not.toHaveBeenCalled()
    expect(clearSessionMock).not.toHaveBeenCalled()
    expect(destroyAllMock).not.toHaveBeenCalled()
    expect(updatePageTitleMock).not.toHaveBeenCalled()
  })

  it("does not ask a Temporary question after the settings editor refuses", () => {
    temporarySource()
    window.addEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, event => event.preventDefault(), { once: true })
    const { result } = renderHook(() => useClearChat())
    act(() => expect(result.current()).toBe(false))
    expect(window.confirm).not.toHaveBeenCalled()
    expect(optionState.startNewChatDraft).not.toHaveBeenCalled()
  })

  it("discards once when the Temporary choice is accepted", () => {
    temporarySource()
    vi.mocked(window.confirm).mockReturnValue(true)
    const { result } = renderHook(() => useClearChat())
    act(() => expect(result.current()).toBe(true))
    expect(window.confirm).toHaveBeenCalledOnce()
    expect(navigateMock).toHaveBeenCalledOnce()
    expect(optionState.startNewChatDraft).toHaveBeenCalledOnce()
    expect(historySelectionResetMock).toHaveBeenCalledOnce()
    expect(clearSessionMock).toHaveBeenCalledOnce()
  })

  it.each([false, true])("skips the question for an empty chat (Temporary=%s)", temporaryChat => {
    Object.assign(optionState, { temporaryChat, historyId: null, serverChatId: null })
    const { result } = renderHook(() => useClearChat())
    act(() => expect(result.current()).toBe(true))
    expect(window.confirm).not.toHaveBeenCalled()
  })

  it("does not clear a replacement conversation admitted while the browser question is open", () => {
    temporarySource()
    vi.mocked(window.confirm).mockImplementation(() => {
      optionState.chatDraftRevision++
      optionState.messages = [{ message: "Replacement conversation" }]
      return true
    })
    const { result } = renderHook(() => useClearChat())
    act(() => expect(result.current()).toBe(false))
    expect(optionState.messages).toEqual([{ message: "Replacement conversation" }])
    expect(optionState.startNewChatDraft).not.toHaveBeenCalled()
    expect(historySelectionResetMock).not.toHaveBeenCalled()
  })

  it("a React click event does not count as prior discard consent", () => {
    temporarySource()
    const { result } = renderHook(() => useClearChat())
    act(() => expect(result.current({ nativeEvent: new MouseEvent("click") } as React.MouseEvent)).toBe(false))
    expect(window.confirm).toHaveBeenCalledOnce()
    expect(optionState.startNewChatDraft).not.toHaveBeenCalled()
  })

  it("does not repeat an explicitly acknowledged all-data discard", () => {
    temporarySource()
    const { result } = renderHook(() => useClearChat())
    act(() => expect(result.current({ discardTemporaryConfirmed: true })).toBe(true))
    expect(window.confirm).not.toHaveBeenCalled()
    expect(optionState.startNewChatDraft).toHaveBeenCalledOnce()
  })

  it("keeps the same-ID replacement history view when the questioned H1 selection retires", () => {
    temporarySource()
    let current = true
    historySelectionContextMock.mockReturnValue({ reset: historySelectionResetMock, fence: () => () => current })
    vi.mocked(window.confirm).mockImplementation(() => { current = false; return true })
    const { result } = renderHook(() => useClearChat())
    act(() => expect(result.current()).toBe(false))
    expect(historySelectionResetMock).not.toHaveBeenCalled()
    expect(optionState.startNewChatDraft).not.toHaveBeenCalled()
  })

})
