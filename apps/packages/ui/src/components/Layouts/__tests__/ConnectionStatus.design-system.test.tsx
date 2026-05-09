import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ConnectionStatus } from "../ConnectionStatus"

type TranslationOptions = {
  defaultValue?: string
  [key: string]: unknown
}

const connectionState = vi.hoisted(() => ({
  state: {
    phase: "connected",
    isConnected: true
  }
}))

const navigateMock = vi.hoisted(() => vi.fn())

const translate = (
  key: string,
  fallback?: string | TranslationOptions,
  interpolation?: Record<string, unknown>
) => {
  const defaultValue =
    typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
  const values = typeof fallback === "object" ? fallback : interpolation

  if (!values) {
    return defaultValue
  }

  return Object.entries(values).reduce((copy, [name, value]) => {
    return copy.replaceAll(`{{${name}}}`, String(value))
  }, defaultValue)
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: translate
  })
}))

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return {
    ...actual,
    useNavigate: () => navigateMock
  }
})

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => connectionState.state
}))

describe("ConnectionStatus design-system badge", () => {
  beforeEach(() => {
    connectionState.state = {
      phase: "connected",
      isConnected: true
    }
    navigateMock.mockReset()
  })

  it.each([
    ["connected", "connected", true, "success", "Server: Online"],
    ["checking", "searching", false, "info", "Server: Checking..."],
    ["unconfigured", "unconfigured", false, "warning", "Server: Not configured"],
    ["offline", "error", false, "danger", "Server: Offline"]
  ])(
    "renders %s status through the shared Badge primitive",
    (_name, phase, isConnected, expectedVariant, label) => {
      connectionState.state = {
        phase,
        isConnected
      }

      render(<ConnectionStatus />)

      const badge = screen.getByTestId("connection-status-dot-badge")

      expect(badge).toHaveAttribute("data-ds-component", "Badge")
      expect(badge).toHaveAttribute("data-ds-variant", expectedVariant)
      expect(
        badge.querySelector('[data-testid="connection-status-dot"]')
      ).toBeInTheDocument()
      expect(screen.getByText(label)).toBeInTheDocument()
    }
  )

  it("preserves the custom click handler override", () => {
    const onClick = vi.fn()

    render(<ConnectionStatus onClick={onClick} />)

    fireEvent.click(screen.getByTestId("connection-status"))

    expect(onClick).toHaveBeenCalledTimes(1)
    expect(navigateMock).not.toHaveBeenCalled()
  })

  it("opens health diagnostics when no click handler is provided", () => {
    render(<ConnectionStatus />)

    fireEvent.click(screen.getByTestId("connection-status"))

    expect(navigateMock).toHaveBeenCalledWith("/settings/health")
  })
})
