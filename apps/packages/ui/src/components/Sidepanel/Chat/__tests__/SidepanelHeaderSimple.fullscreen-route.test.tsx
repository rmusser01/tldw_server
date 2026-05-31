// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

const browserMocks = vi.hoisted(() => ({
  createTab: vi.fn(() => Promise.resolve({ id: 1 })),
  getURL: vi.fn((path: string) => `chrome-extension://tldw${path}`)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: { getURL: browserMocks.getURL },
    tabs: { create: browserMocks.createTab }
  }
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

describe("SidepanelHeaderSimple full-screen route", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("states that full-screen opens /chat without promising draft or page state loss", () => {
    render(
      <MemoryRouter>
        <SidepanelHeaderSimple activeTitle="Sidepanel chat" />
      </MemoryRouter>
    )

    const openFullChat = screen.getByRole("button", {
      name: "Open full chat in WebUI"
    })

    expect(openFullChat).toHaveAccessibleDescription(
      "Opens /chat in a new tab. Use Continue in WebUI from the composer tools to carry a draft or page context."
    )
    expect(openFullChat).toHaveAttribute(
      "title",
      "Opens /chat in a new tab. Use Continue in WebUI from the composer tools to carry a draft or page context."
    )
  })

  it("opens the extension options carrier for the WebUI /chat route", async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <SidepanelHeaderSimple activeTitle="Sidepanel chat" />
      </MemoryRouter>
    )

    await user.click(screen.getByTestId("chat-open-full-screen"))

    expect(browserMocks.getURL).toHaveBeenCalledWith("/options.html#/chat")
    expect(browserMocks.createTab).toHaveBeenCalledWith({
      url: "chrome-extension://tldw/options.html#/chat"
    })
  })

  it("keeps the dashboard button on the dashboard route", async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <SidepanelHeaderSimple activeTitle="Sidepanel chat" />
      </MemoryRouter>
    )

    await user.click(screen.getByTestId("chat-open-dashboard"))

    expect(browserMocks.getURL).toHaveBeenCalledWith("/options.html#/flashcards")
    expect(browserMocks.createTab).toHaveBeenCalledWith({
      url: "chrome-extension://tldw/options.html#/flashcards"
    })
  })
})
