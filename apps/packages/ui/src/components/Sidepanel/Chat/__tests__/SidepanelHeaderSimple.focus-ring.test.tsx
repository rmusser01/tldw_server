// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: { getURL: vi.fn((path: string) => path) },
    tabs: { create: vi.fn(() => Promise.resolve({ id: 1 })) }
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
  useServerCapabilities: () => ({ capabilities: { hasPersona: true } })
}))

vi.mock("../TtsClipsDrawer", () => ({
  TtsClipsDrawer: () => <div data-testid="tts-clips-drawer" />
}))

import { SidepanelHeaderSimple } from "../SidepanelHeaderSimple"

// AX-08 (#3124): "Open Persona Garden" and "Open settings" are links, and the
// antd link reset removes their outline, so keyboard focus showed nothing.
// Every header icon control now draws a ring in the theme focus colour.
describe("SidepanelHeaderSimple focus indicators", () => {
  it("draws a focus ring on every header icon control", () => {
    render(
      <MemoryRouter>
        <SidepanelHeaderSimple activeTitle="Sidepanel chat" />
      </MemoryRouter>
    )

    const controls = [
      screen.getByTestId("chat-open-tts-clips"),
      screen.getByTestId("chat-open-persona"),
      screen.getByTestId("chat-open-dashboard"),
      screen.getByTestId("chat-open-full-screen"),
      screen.getByRole("link", { name: "sidepanel:header.openSettingsAria" })
    ]

    for (const control of controls) {
      expect(control).toHaveClass("focus-visible:ring-2", "focus-visible:ring-focus")
    }
  })
})
