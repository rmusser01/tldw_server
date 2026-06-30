import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SidepanelHeaderSimple } from "../SidepanelHeaderSimple"

const { runtimeGetURLMock, tabsCreateMock } = vi.hoisted(() => ({
  runtimeGetURLMock: vi.fn((path: string) => `chrome-extension://test${path}`),
  tabsCreateMock: vi.fn().mockResolvedValue(undefined)
}))

vi.mock("@/assets/icon.png", () => ({
  default: "chrome-extension://test/icon.png"
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("react-router-dom", () => ({
  Link: ({
    children,
    to,
    ...props
  }: {
    children?: React.ReactNode
    to: string
  }) => (
    <a href={to} {...props}>
      {children}
    </a>
  )
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      getURL: runtimeGetURLMock
    },
    tabs: {
      create: tabsCreateMock
    }
  }
}))

vi.mock("@/hooks/useMessage", () => ({
  useMessage: () => ({
    temporaryChat: false
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: vi.fn()
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasPersona: false }
  })
}))

vi.mock("../StatusDot", () => ({
  StatusDot: () => <span data-testid="status-dot" />
}))

vi.mock("../TtsClipsDrawer", () => ({
  TtsClipsDrawer: () => <div data-testid="tts-clips-drawer" />
}))

describe("SidepanelHeaderSimple WebUI chat handoff", () => {
  beforeEach(() => {
    runtimeGetURLMock.mockClear()
    tabsCreateMock.mockClear()
  })

  it("delegates the visible full-screen chat action to the WebUI chat handoff", async () => {
    const onOpenChatInWebUi = vi.fn().mockResolvedValue(undefined)

    render(
      <SidepanelHeaderSimple onOpenChatInWebUi={onOpenChatInWebUi} />
    )

    fireEvent.click(screen.getByTestId("chat-open-full-screen"))

    await waitFor(() =>
      expect(onOpenChatInWebUi).toHaveBeenCalledTimes(1)
    )
    expect(runtimeGetURLMock).not.toHaveBeenCalledWith("/options.html#/")
    expect(tabsCreateMock).not.toHaveBeenCalled()
  })
})
