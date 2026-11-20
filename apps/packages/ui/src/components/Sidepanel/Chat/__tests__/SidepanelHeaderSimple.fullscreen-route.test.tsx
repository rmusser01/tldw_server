// @vitest-environment jsdom
/**
 * XP-09 (#3101): the header's full-page button is the side panel's one "Open
 * in full page" action. The composer owns it, because the draft lives there.
 */
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

const browserMocks = vi.hoisted(() => ({
  runtimeId: "header-extension" as string | undefined,
  storage: {} as Record<string, unknown>,
  createTab: vi.fn(() => Promise.resolve({ id: 1 })),
  getURL: vi.fn((path: string) => `chrome-extension://tldw${path}`)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: { get id() { return browserMocks.runtimeId }, getURL: browserMocks.getURL },
    tabs: { create: browserMocks.createTab }
  }
}))

vi.mock("@/utils/safe-storage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: () => ({ get: async (key: string) => browserMocks.storage[key] })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/hooks/useMessage", () => ({
  useMessage: () => ({ temporaryChat: false })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ error: vi.fn() })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: { hasPersona: false } })
}))

vi.mock("../TtsClipsDrawer", () => ({
  TtsClipsDrawer: () => <div data-testid="tts-clips-drawer" />
}))

import { SidepanelHeaderSimple } from "../SidepanelHeaderSimple"

const renderHeader = (props: React.ComponentProps<typeof SidepanelHeaderSimple> = {}) =>
  render(
    <MemoryRouter>
      <SidepanelHeaderSimple activeTitle="Sidepanel chat" {...props} />
    </MemoryRouter>
  )

describe("SidepanelHeaderSimple full-page action", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    browserMocks.runtimeId = "header-extension"
    browserMocks.storage = {}
  })

  it("names the one action 'Open in full page' and says what it carries", () => {
    renderHeader({ onOpenFullPage: vi.fn() })
    const open = screen.getByRole("button", { name: "Open in full page" })
    expect(open).toHaveAccessibleDescription(
      "Opens this chat in a new tab, with your draft and page references. Your draft also stays here."
    )
  })

  it("says when the action opens the WebUI the user configured", async () => {
    browserMocks.storage = { tldwConfig: { webUiUrl: "https://webui.example.test" } }
    renderHeader({ onOpenFullPage: vi.fn() })
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Open in full page" })).toHaveAccessibleDescription(
        "Opens this chat in the WebUI, with your draft and attachments. Chats kept only on this device stay here."
      )
    )
  })

  it("runs the composer's action rather than opening a page itself", async () => {
    const onOpenFullPage = vi.fn(async () => true)
    renderHeader({ onOpenFullPage })
    await act(async () => { fireEvent.click(screen.getByTestId("chat-open-full-screen")) })
    expect(onOpenFullPage).toHaveBeenCalledOnce()
    expect(browserMocks.createTab).not.toHaveBeenCalled()
  })

  it("is disabled until the conversation has restored", () => {
    const onOpenFullPage = vi.fn()
    renderHeader({ chatReady: false, onOpenFullPage })
    const open = screen.getByTestId("chat-open-full-screen")
    expect(open).toBeDisabled()
    expect(open).toHaveAccessibleDescription(
      "Restore or start a conversation before opening it in a full page."
    )
    fireEvent.click(open)
    expect(onOpenFullPage).not.toHaveBeenCalled()
  })

  // Review fix: Clipper, Companion and Persona have no composer to run the action.
  it("opens the extension's chat page on routes without a composer", async () => {
    const user = userEvent.setup()
    renderHeader()
    const open = screen.getByTestId("chat-open-full-screen")
    expect(open).toBeEnabled()
    expect(open).toHaveAccessibleDescription("Opens the chat page in a new tab. Drafts and page references stay here.")
    await user.click(open)
    expect(browserMocks.createTab).toHaveBeenCalledWith({ url: "chrome-extension://tldw/options.html#/chat" })
  })

  it("keeps the dashboard button on the dashboard route", async () => {
    const user = userEvent.setup()
    renderHeader({ onOpenFullPage: vi.fn() })
    await user.click(screen.getByTestId("chat-open-dashboard"))
    expect(browserMocks.getURL).toHaveBeenCalledWith("/options.html#/flashcards")
    expect(browserMocks.createTab).toHaveBeenCalledWith({
      url: "chrome-extension://tldw/options.html#/flashcards"
    })
  })
})
