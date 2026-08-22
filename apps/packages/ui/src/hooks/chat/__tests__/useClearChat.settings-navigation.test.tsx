// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

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

vi.mock("@/hooks/utils/messageHelpers", () => ({
  focusTextArea: focusTextAreaMock
}))

vi.mock("@/store/option", () => {
  const useStoreMessageOption = Object.assign(
    (selector: (state: typeof optionState) => unknown) => selector(optionState),
    { setState: optionStoreSetStateMock }
  )
  return { useStoreMessageOption }
})

vi.mock("@/store", () => ({
  useStoreMessage: vi.fn()
}))

vi.mock("@/store/playground-session", () => ({
  usePlaygroundSessionStore: {
    getState: () => ({ clearSession: clearSessionMock })
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
    window.history.replaceState({}, "", "/settings/prompt")
  })

  it("does nothing when the mounted settings editor declines navigation", () => {
    const declineNavigation = vi.fn((event: Event) => event.preventDefault())
    window.addEventListener(
      SETTINGS_NAVIGATION_REQUEST_EVENT,
      declineNavigation,
      { once: true }
    )
    const { result } = renderHook(() => useClearChat())

    act(() => result.current())

    expect(declineNavigation).toHaveBeenCalledOnce()
    expect((declineNavigation.mock.calls[0][0] as CustomEvent<
      SettingsNavigationRequestDetail
    >).detail).toEqual({ destination: "/chat" })
    expect(navigateMock).not.toHaveBeenCalled()
    expect(destroyAllMock).not.toHaveBeenCalled()
    expect(cleanupOverlaysMock).not.toHaveBeenCalled()
    expect(baseState.setMessages).not.toHaveBeenCalled()
    expect(optionState.setServerChatId).not.toHaveBeenCalled()
    expect(resetModelSettingsMock).not.toHaveBeenCalled()
    expect(updatePageTitleMock).not.toHaveBeenCalled()
    expect(focusTextAreaMock).not.toHaveBeenCalled()
    expect(clearSessionMock).not.toHaveBeenCalled()
  })

  it("navigates and resets once when navigation is allowed", () => {
    const { result } = renderHook(() => useClearChat())

    act(() => result.current())

    expect(navigateMock).toHaveBeenCalledOnce()
    expect(navigateMock).toHaveBeenCalledWith("/chat")
    expect(destroyAllMock).toHaveBeenCalledOnce()
    expect(cleanupOverlaysMock).toHaveBeenCalledOnce()
    expect(optionState.setServerChatId).toHaveBeenCalledOnce()
    expect(resetModelSettingsMock).toHaveBeenCalledOnce()
    expect(updatePageTitleMock).toHaveBeenCalledOnce()
    expect(focusTextAreaMock).toHaveBeenCalledOnce()
    expect(clearSessionMock).toHaveBeenCalledOnce()
  })
})
