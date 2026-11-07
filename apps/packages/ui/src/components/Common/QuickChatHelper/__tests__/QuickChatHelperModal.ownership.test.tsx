import React from "react"
import { ConfigProvider } from "antd"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { afterAll, afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest"
import { QuickChatHelperModal } from "../QuickChatHelperModal"
import { useQuickChatStore } from "@/store/quick-chat"

const authority = vi.hoisted(() => ({
  ready: true,
  ownerId: "Alice" as string | null
}))
vi.mock("@/hooks/usePromptLocalAuthority", () => ({
  usePromptLocalAuthority: () => authority
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback })
}))
vi.mock("@/hooks/useQuickChat", () => ({
  useQuickChat: () => ({
    messages: [
      {
        id: "visible",
        role: "user",
        content: "Existing question",
        timestamp: 1
      }
    ],
    sendMessage: vi.fn(),
    cancelStream: vi.fn(),
    isStreaming: false,
    hasModel: true,
    activeModel: "fixture",
    currentModel: "fixture",
    modelOverride: null,
    setModelOverride: vi.fn()
  })
}))
vi.mock("@/hooks/useChatModelCatalog", () => ({
  EMPTY_CHAT_MODELS: [],
  useChatModelCatalog: () => ({
    data: [],
    isLoading: false,
    isError: false,
    refetch: vi.fn()
  })
}))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionPhase: () => "connected",
  useIsConnected: () => true
}))
vi.mock("@/hooks/useStartTutorial", () => ({ useStartTutorial: () => vi.fn() }))
vi.mock("./../QuickChatMessage", () => ({ QuickChatMessage: () => <div /> }))
vi.mock("./../QuickChatInput", () => ({ QuickChatInput: () => <textarea /> }))
vi.mock("./../QuickChatGuidesPanel", () => ({
  QuickChatGuidesPanel: () => <div />
}))
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: { getURL: (path: string) => "chrome-extension://fixture" + path },
    storage: { onChanged: { addListener: vi.fn(), removeListener: vi.fn() } }
  }
}))

// jsdom has no layout scrolling; the actual modal lifecycle still runs.
const scroll = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollIntoView")
beforeAll(() => Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() }))
afterAll(() => {
  if (scroll) Object.defineProperty(HTMLElement.prototype, "scrollIntoView", scroll)
  else Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView")
})

const key = "quickchat_1760000000000"
const current = {
  messages: [
    {
      id: "current",
      role: "user" as const,
      content: "Current private question",
      timestamp: 2
    }
  ],
  modelOverride: "private model",
  assistantMode: "docs_rag" as const
}
function mount() {
  render(
    <ConfigProvider theme={{ token: { motion: false } }}>
      <MemoryRouter initialEntries={["/notes"]}>
        <QuickChatHelperModal open onClose={vi.fn()} />
      </MemoryRouter>
    </ConfigProvider>
  )
  return screen.getByRole("button", { name: "Pop out", exact: true })
}
beforeEach(() => {
  authority.ready = true
  authority.ownerId = "Alice"
  sessionStorage.clear()
  sessionStorage.setItem("unrelated", "keep")
  sessionStorage.setItem("quickchat_1760000000001", "another transfer")
  useQuickChatStore.setState({
    ...current,
    isOpen: true,
    isStreaming: false,
    popoutWindow: null
  })
  vi.spyOn(Date, "now").mockReturnValue(1760000000000)
})
afterEach(() => {
  vi.restoreAllMocks()
  sessionStorage.clear()
})

it("disables transfer until an owner is resolved", () => {
  authority.ready = false
  authority.ownerId = null
  const open = vi.spyOn(window, "open").mockReturnValue(null)
  const button = mount()
  expect(button).toBeDisabled()
  fireEvent.click(button)
  expect(open).not.toHaveBeenCalled()
  expect(sessionStorage.getItem(key)).toBeNull()
})
it("captures the current store and admitted owner, then removes only its blocked transfer", () => {
  const open = vi.spyOn(window, "open").mockImplementation(() => {
    expect(JSON.parse(sessionStorage.getItem(key)!)).toEqual({
      ...current,
      sourceRoute: "/notes",
      ownerId: "Alice"
    })
    return null
  })
  const button = mount()
  expect(button).toBeEnabled()
  fireEvent.click(button)
  expect(open).toHaveBeenCalledOnce()
  expect(sessionStorage.getItem(key)).toBeNull()
  expect(sessionStorage.getItem("unrelated")).toBe("keep")
  expect(sessionStorage.getItem("quickchat_1760000000001")).toBe(
    "another transfer"
  )
  expect(useQuickChatStore.getState().messages).toEqual(current.messages)
})
it("refuses a click after a principal boundary even before owner-hook rerender", () => {
  const open = vi.spyOn(window, "open").mockReturnValue(null)
  const button = mount()
  act(() =>
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
  )
  fireEvent.click(button)
  expect(authority.ready).toBe(true)
  expect(open).not.toHaveBeenCalled()
  expect(sessionStorage.getItem(key)).toBeNull()
})
