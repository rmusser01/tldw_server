// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  getDesignSystemState,
  type DesignSystemStateKey
} from "@/design-system"
import type { ConnectionState, ConnectionUxState } from "@/types/connection"
import { SaveStatusIcon } from "../SaveStatusIcon"
import { StatusDot } from "../StatusDot"

type ConnectionUxHookState = {
  uxState: ConnectionUxState
  mode: ConnectionState["mode"]
  errorKind: ConnectionState["errorKind"]
  configStep: ConnectionState["configStep"]
  hasCompletedFirstRun: boolean
  isConnectedUx: boolean
  isChecking: boolean
  isConfigOrError: boolean
}

const connectionState = vi.hoisted(() => ({
  ux: {
    uxState: "connected_ok",
    mode: "normal",
    errorKind: "none",
    configStep: "none",
    hasCompletedFirstRun: true,
    isConnectedUx: true,
    isChecking: false,
    isConfigOrError: false
  } as ConnectionUxHookState,
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

const connectedUxStates: ConnectionUxState[] = [
  "connected_ok",
  "connected_degraded",
  "demo_mode"
]

const configOrErrorUxStates: ConnectionUxState[] = [
  "unconfigured",
  "configuring_url",
  "configuring_auth",
  "error_unreachable",
  "error_auth"
]

const makeConnectionUxState = ({
  uxState,
  mode = uxState === "demo_mode" ? "demo" : "normal",
  errorKind = uxState === "error_auth"
    ? "auth"
    : uxState === "error_unreachable"
      ? "unreachable"
      : uxState === "connected_degraded"
        ? "partial"
        : "none",
  configStep = uxState === "configuring_url"
    ? "url"
    : uxState === "configuring_auth"
      ? "auth"
      : "none",
  hasCompletedFirstRun = true
}: {
  uxState: ConnectionUxState
  mode?: ConnectionState["mode"]
  errorKind?: ConnectionState["errorKind"]
  configStep?: ConnectionState["configStep"]
  hasCompletedFirstRun?: boolean
}): ConnectionUxHookState => ({
  uxState,
  mode,
  errorKind,
  configStep,
  hasCompletedFirstRun,
  isConnectedUx: connectedUxStates.includes(uxState),
  isChecking: uxState === "testing",
  isConfigOrError: configOrErrorUxStates.includes(uxState)
})

const statusDotCases = [
  [
    "connected",
    makeConnectionUxState({ uxState: "connected_ok" }),
    "ready",
    "success",
    "Connected to your tldw server"
  ],
  [
    "degraded connection",
    makeConnectionUxState({ uxState: "connected_degraded" }),
    "degraded",
    "warning",
    "Connected to your tldw server"
  ],
  [
    "checking",
    makeConnectionUxState({ uxState: "testing" }),
    "retrying",
    "info",
    "Checking connection to your tldw server…"
  ],
  [
    "demo",
    makeConnectionUxState({ uxState: "demo_mode" }),
    "ready",
    "demo",
    "Demo mode: explore with a sample workspace."
  ],
  [
    "unconfigured setup",
    makeConnectionUxState({ uxState: "unconfigured" }),
    "setup_required",
    "warning",
    "Not connected. Open Settings to configure."
  ],
  [
    "configuring auth setup",
    makeConnectionUxState({ uxState: "configuring_auth" }),
    "setup_required",
    "warning",
    "Not connected. Open Settings to configure."
  ],
  [
    "auth error",
    makeConnectionUxState({ uxState: "error_auth" }),
    "auth_required",
    "warning",
    "Not connected. Open Settings to configure."
  ],
  [
    "unreachable error",
    makeConnectionUxState({ uxState: "error_unreachable" }),
    "unavailable",
    "danger",
    "Connection failed. Click to retry."
  ]
] satisfies Array<[
  string,
  ConnectionUxHookState,
  DesignSystemStateKey,
  string,
  string
]>

describe("Chat status design-system badges", () => {
  beforeEach(() => {
    connectionState.ux = makeConnectionUxState({ uxState: "connected_ok" })
    connectionState.checkOnce.mockReset()
    notificationMock.warning.mockReset()
    vi.clearAllMocks()
  })

  it.each(statusDotCases)(
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
    connectionState.ux = makeConnectionUxState({ uxState: "error_unreachable" })

    render(<StatusDot />)

    const statusButton = screen.getByTestId("status-dot")

    expect(statusButton).toHaveAccessibleName("Connection failed. Click to retry.")

    fireEvent.click(statusButton)

    expect(connectionState.checkOnce).toHaveBeenCalledTimes(1)
  })

  it("disables retry while checking the connection", () => {
    connectionState.ux = makeConnectionUxState({ uxState: "testing" })

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
    connectionState.ux = makeConnectionUxState({ uxState: "configuring_url" })

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
