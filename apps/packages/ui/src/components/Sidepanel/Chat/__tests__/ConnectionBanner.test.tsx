import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ConnectionBanner } from "../ConnectionBanner"
import { ConnectionPhase } from "@/types/connection"

const {
  setConfigPartialMock,
  checkOnceMock,
  setStoredApiKeyMock,
  messageSuccessMock,
  messageErrorMock,
  getConfigMock
} = vi.hoisted(() => ({
  setConfigPartialMock: vi.fn().mockResolvedValue(undefined),
  checkOnceMock: vi.fn().mockResolvedValue(undefined),
  setStoredApiKeyMock: vi.fn().mockResolvedValue(undefined),
  messageSuccessMock: vi.fn(),
  messageErrorMock: vi.fn(),
  getConfigMock: vi.fn().mockResolvedValue({
    authMode: "single-user"
  })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("antd", () => ({
  Alert: ({
    title,
    description
  }: {
    title?: React.ReactNode
    description?: React.ReactNode
  }) => (
    <div>
      <div>{title}</div>
      <div>{description}</div>
    </div>
  ),
  Button: ({
    children,
    onClick,
    disabled,
    loading
  }: {
    children?: React.ReactNode
    onClick?: () => void
    disabled?: boolean
    loading?: boolean
  }) => (
    <button type="button" onClick={onClick} disabled={disabled || loading}>
      {children}
    </button>
  ),
  Input: {
    Password: ({
      value,
      onChange,
      onPressEnter,
      placeholder
    }: {
      value?: string
      onChange?: (event: { target: { value: string } }) => void
      onPressEnter?: () => void
      placeholder?: string
    }) => (
      <input
        value={value}
        placeholder={placeholder}
        onChange={(event) =>
          onChange?.({ target: { value: event.currentTarget.value } })
        }
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            onPressEnter?.()
          }
        }}
      />
    )
  },
  message: {
    success: messageSuccessMock,
    error: messageErrorMock
  }
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    phase: ConnectionPhase.ERROR,
    isConnected: false,
    serverUrl: "http://127.0.0.1:8000"
  }),
  useConnectionUxState: () => ({
    uxState: "error_auth",
    isChecking: false,
    hasCompletedFirstRun: true
  }),
  useConnectionActions: () => ({
    checkOnce: checkOnceMock,
    setConfigPartial: setConfigPartialMock
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => ["", setStoredApiKeyMock]
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: getConfigMock
  }
}))

describe("ConnectionBanner", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    setConfigPartialMock.mockReset().mockResolvedValue(undefined)
    checkOnceMock.mockReset().mockResolvedValue(undefined)
    setStoredApiKeyMock.mockReset().mockResolvedValue(undefined)
    messageSuccessMock.mockReset()
    messageErrorMock.mockReset()
    getConfigMock.mockReset().mockResolvedValue({
      authMode: "single-user"
    })
  })

  it("repairs single-user auth through shared config instead of legacy api key storage", async () => {
    render(<ConnectionBanner />)

    const banner = screen.getByTestId("sidepanel-connection-banner")

    expect(banner).toHaveAttribute("data-ds-component", "RecoveryCallout")
    expect(screen.getByText("Sign in required")).toBeInTheDocument()

    fireEvent.click(
      screen.getByRole("button", { name: "Enter API Key" })
    )
    fireEvent.change(
      screen.getByPlaceholderText("Enter your API key"),
      {
        target: { value: "real-key" }
      }
    )
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    expect(setConfigPartialMock).toHaveBeenCalledWith({
      apiKey: "real-key"
    })
    expect(setStoredApiKeyMock).not.toHaveBeenCalled()

    await act(async () => {
      vi.runAllTimers()
    })

    expect(checkOnceMock).toHaveBeenCalledTimes(1)
  })
})
