// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SaveStatusIcon } from "../SaveStatusIcon"
import { StatusDot } from "../StatusDot"

const connectionState = vi.hoisted(() => ({
  ux: {
    uxState: "connected",
    mode: "full",
    isConnectedUx: true,
    isChecking: false,
    isConfigOrError: false
  },
  checkOnce: vi.fn()
}))

const notificationMock = vi.hoisted(() => ({
  warning: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => connectionState.ux,
  useConnectionActions: () => ({
    checkOnce: connectionState.checkOnce
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

describe("Chat status design-system badges", () => {
  beforeEach(() => {
    connectionState.ux = {
      uxState: "connected",
      mode: "full",
      isConnectedUx: true,
      isChecking: false,
      isConfigOrError: false
    }
    connectionState.checkOnce.mockReset()
    notificationMock.warning.mockReset()
  })

  it("renders the connection status through the shared Badge while preserving retry behavior", () => {
    connectionState.ux = {
      uxState: "failed",
      mode: "full",
      isConnectedUx: false,
      isChecking: false,
      isConfigOrError: false
    }

    render(<StatusDot />)

    const statusButton = screen.getByTestId("status-dot")
    const badge = screen.getByTestId("status-dot-badge")

    expect(statusButton).toHaveAccessibleName("Connection failed. Click to retry.")
    expect(badge).toHaveAttribute("data-ds-component", "Badge")
    expect(badge.querySelector(".sr-only")).toBeNull()
    expect(badge.querySelector("svg")).toHaveClass("text-current")

    fireEvent.click(statusButton)

    expect(connectionState.checkOnce).toHaveBeenCalledTimes(1)
  })

  it("renders the chat save status through the shared Badge while preserving the action", () => {
    const onClick = vi.fn()

    render(
      <SaveStatusIcon
        temporaryChat={false}
        serverChatId="server-chat-1"
        onClick={onClick}
      />
    )

    const statusButton = screen.getByTestId("chat-save-status")
    const badge = screen.getByTestId("chat-save-status-badge")

    expect(statusButton).toHaveAccessibleName("Saved to server")
    expect(badge).toHaveAttribute("data-ds-component", "Badge")
    expect(badge.querySelector(".sr-only")).toBeNull()

    fireEvent.click(statusButton)

    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it("keeps configured-but-unavailable status colors aligned with the Badge variant", () => {
    connectionState.ux = {
      uxState: "error_config",
      mode: "full",
      isConnectedUx: false,
      isChecking: false,
      isConfigOrError: true
    }

    render(<StatusDot />)

    const badge = screen.getByTestId("status-dot-badge")
    const icon = badge.querySelector("svg")

    expect(screen.getByTestId("status-dot")).toHaveAccessibleName(
      "Not connected. Open Settings to configure."
    )
    expect(badge).toHaveAttribute("data-ds-component", "Badge")
    expect(icon).toHaveClass("text-current")
    expect(icon).not.toHaveClass("text-danger")
  })
})
