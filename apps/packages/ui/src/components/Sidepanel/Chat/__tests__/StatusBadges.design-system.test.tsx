// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { getDesignSystemState } from "@/design-system"
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

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(actual.getDesignSystemState)
  }
})

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
    vi.clearAllMocks()
  })

  it.each([
    [
      "connected",
      {
        uxState: "connected",
        mode: "full",
        isConnectedUx: true,
        isChecking: false,
        isConfigOrError: false
      },
      "ready",
      "success",
      "Connected to your tldw server"
    ],
    [
      "checking",
      {
        uxState: "checking",
        mode: "full",
        isConnectedUx: false,
        isChecking: true,
        isConfigOrError: false
      },
      "retrying",
      "info",
      "Checking connection to your tldw server…"
    ],
    [
      "demo",
      {
        uxState: "connected",
        mode: "demo",
        isConnectedUx: true,
        isChecking: false,
        isConfigOrError: false
      },
      "ready",
      "demo",
      "Demo mode: explore with a sample workspace."
    ],
    [
      "setup issue",
      {
        uxState: "error_config",
        mode: "full",
        isConnectedUx: false,
        isChecking: false,
        isConfigOrError: true
      },
      "setup_required",
      "warning",
      "Not connected. Open Settings to configure."
    ],
    [
      "retryable failure",
      {
        uxState: "failed",
        mode: "full",
        isConnectedUx: false,
        isChecking: false,
        isConfigOrError: false
      },
      "unavailable",
      "danger",
      "Connection failed. Click to retry."
    ]
  ])(
    "renders %s connection status through the design-system state registry",
    (_label, uxState, stateKey, variant, accessibleName) => {
      connectionState.ux = uxState

      render(<StatusDot />)

      const statusButton = screen.getByTestId("status-dot")
      const badge = screen.getByTestId("status-dot-badge")

      expect(getDesignSystemState).toHaveBeenCalledWith(stateKey)
      expect(statusButton).toHaveAccessibleName(accessibleName)
      expect(badge).toHaveAttribute("data-ds-component", "Badge")
      expect(badge).toHaveAttribute("data-ds-variant", variant)
      expect(badge.querySelector(".sr-only")).toBeNull()
      expect(badge.querySelector("svg")).toHaveClass("text-current")
    }
  )

  it("preserves retry behavior for retryable connection failures", () => {
    connectionState.ux = {
      uxState: "failed",
      mode: "full",
      isConnectedUx: false,
      isChecking: false,
      isConfigOrError: false
    }

    render(<StatusDot />)

    const statusButton = screen.getByTestId("status-dot")

    expect(statusButton).toHaveAccessibleName("Connection failed. Click to retry.")

    fireEvent.click(statusButton)

    expect(connectionState.checkOnce).toHaveBeenCalledTimes(1)
  })

  it("disables retry while checking the connection", () => {
    connectionState.ux = {
      uxState: "checking",
      mode: "full",
      isConnectedUx: false,
      isChecking: true,
      isConfigOrError: false
    }

    render(<StatusDot />)

    const statusButton = screen.getByTestId("status-dot")

    expect(statusButton).toBeDisabled()

    fireEvent.click(statusButton)

    expect(connectionState.checkOnce).not.toHaveBeenCalled()
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
