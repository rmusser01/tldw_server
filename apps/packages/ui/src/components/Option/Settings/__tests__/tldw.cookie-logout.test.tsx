// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  logout: vi.fn(),
  probeServerHealth: vi.fn(),
  setFieldsValue: vi.fn(),
  messageSuccess: vi.fn()
}))

const formValues: Record<string, unknown> = {}
const form = {
  getFieldsValue: vi.fn(() => ({ ...formValues })),
  setFieldsValue: mocks.setFieldsValue,
  setFieldValue: vi.fn((key: string, value: unknown) => {
    formValues[key] = value
  }),
  validateFields: vi.fn().mockResolvedValue({})
}

vi.mock("antd", () => {
  const Form = Object.assign(
    ({ children }: { children?: React.ReactNode }) => <form>{children}</form>,
    { useForm: () => [form] }
  )

  return {
    Alert: ({ title }: { title?: React.ReactNode }) => <div>{title}</div>,
    Button: ({ children, onClick }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
      <button type="button" onClick={onClick}>{children}</button>
    ),
    Form,
    Modal: { confirm: vi.fn() },
    Space: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    Spin: ({ children }: { children?: React.ReactNode }) => <>{children}</>
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
  })
}))

vi.mock("react-router-dom", () => ({
  Link: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  useNavigate: () => vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: mocks.getConfig,
    initialize: vi.fn().mockResolvedValue(undefined),
    ragHealth: vi.fn().mockResolvedValue({ status: "ok" }),
    updateConfig: vi.fn().mockResolvedValue(undefined)
  }
}))

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    logout: mocks.logout,
    login: vi.fn(),
    requestMagicLink: vi.fn(),
    verifyMagicLink: vi.fn()
  }
}))

vi.mock("@/components/Common/Settings/SettingsSkeleton", () => ({
  SettingsSkeleton: () => <div>Loading</div>
}))
vi.mock("@/services/tldw-server", () => ({ DEFAULT_TLDW_API_KEY: "default-key" }))
vi.mock("@/services/api-send", () => ({ apiSend: vi.fn() }))
vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: mocks.messageSuccess,
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  })
}))
vi.mock("@/store/connection", () => ({
  useConnectionStore: { getState: () => ({ checkOnce: vi.fn() }) }
}))
vi.mock("@/services/auth-errors", () => ({
  mapMultiUserLoginErrorMessage: vi.fn(() => "Login failed")
}))
vi.mock("@/components/Option/Onboarding/validation", () => ({
  commitManualServerTransition: vi.fn()
}))
vi.mock("@/services/splash-auth", () => ({
  emitSplashAfterSingleUserAuthSuccess: vi.fn()
}))
vi.mock("@/components/Common/ServerOverviewHint", () => ({
  ServerOverviewHint: () => null
}))
vi.mock("@/utils/extension-permissions", () => ({
  requestOptionalHostPermission: vi.fn()
}))
vi.mock("../server-health-probe", () => ({
  probeServerHealth: mocks.probeServerHealth
}))
vi.mock("../tldw-settings-tabs", () => ({
  TldwSettingsTabs: () => null
}))
vi.mock("../TldwTimeoutSettings", () => ({
  TIMEOUT_PRESETS: {
    balanced: {
      request: 10,
      stream: 15,
      chatRequest: 10,
      chatStartup: 10,
      chatStream: 15,
      ragRequest: 10,
      media: 60,
      upload: 60
    }
  },
  determinePreset: vi.fn(() => "balanced"),
  TldwTimeoutSettings: () => null
}))
vi.mock("../TldwBillingSettings", () => ({
  TldwBillingSettings: () => null
}))
vi.mock("../TldwConnectionSettings", () => ({
  TldwConnectionSettings: (props: {
    configuredServerUrl: string
    authSource?: string
    rememberApiKey: boolean
    authMode: string
    connectionStatus: string | null
    connectionDetail: string
    coreStatus: string
    ragStatus: string
    onTestConnection: () => void
    onLogout: () => void
  }) => (
    <div>
      <span data-testid="server-url">{props.configuredServerUrl}</span>
      <span data-testid="auth-source">{props.authSource ?? "manual"}</span>
      <span data-testid="auth-mode">{props.authMode}</span>
      <span data-testid="remember-key">{String(props.rememberApiKey)}</span>
      <span data-testid="connection-status">{props.connectionStatus ?? "none"}</span>
      <span data-testid="connection-detail">{props.connectionDetail}</span>
      <span data-testid="core-status">{props.coreStatus}</span>
      <span data-testid="rag-status">{props.ragStatus}</span>
      {props.authSource !== "cookie-session" && <span>Manual key controls</span>}
      <button type="button" onClick={props.onTestConnection}>Test connection</button>
      <button type="button" onClick={props.onLogout}>Logout</button>
    </div>
  )
}))

import { TldwSettings } from "../tldw"

describe("TldwSettings cookie logout", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    Object.keys(formValues).forEach((key) => delete formValues[key])
    mocks.setFieldsValue.mockImplementation((values: Record<string, unknown>) => {
      Object.assign(formValues, values)
    })
    mocks.getConfig
      .mockResolvedValueOnce({
        serverUrl: "http://127.0.0.1:8000",
        apiKey: "stale-browser-key",
        authMode: "single-user",
        authSource: "cookie-session",
        apiKeyPersistence: "session"
      })
      .mockResolvedValueOnce(null)
    mocks.logout.mockResolvedValue(undefined)
    mocks.probeServerHealth.mockResolvedValue({ ok: true })
  })

  it("restores a clean manual single-user form after cookie-only logout", async () => {
    render(<TldwSettings />)

    expect(await screen.findByTestId("auth-source")).toHaveTextContent("cookie-session")
    expect(screen.queryByText("Manual key controls")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Test connection" }))
    await waitFor(() => {
      expect(screen.getByTestId("connection-status")).toHaveTextContent("success")
      expect(screen.getByTestId("core-status")).toHaveTextContent("connected")
      expect(screen.getByTestId("rag-status")).toHaveTextContent("healthy")
    })

    fireEvent.click(screen.getByRole("button", { name: "Logout" }))

    await waitFor(() => {
      expect(mocks.logout).toHaveBeenCalledTimes(1)
      expect(screen.getByText("Manual key controls")).toBeInTheDocument()
      expect(screen.getByTestId("auth-source")).toHaveTextContent("manual")
      expect(screen.getByTestId("auth-mode")).toHaveTextContent("single-user")
      expect(screen.getByTestId("server-url")).toBeEmptyDOMElement()
      expect(screen.getByTestId("remember-key")).toHaveTextContent("true")
      expect(screen.getByTestId("connection-status")).toHaveTextContent("none")
      expect(screen.getByTestId("connection-detail")).toBeEmptyDOMElement()
      expect(screen.getByTestId("core-status")).toHaveTextContent("unknown")
      expect(screen.getByTestId("rag-status")).toHaveTextContent("unknown")
    })
    expect(mocks.setFieldsValue).toHaveBeenLastCalledWith(
      expect.objectContaining({
        serverUrl: "",
        apiKey: "",
        authMode: "single-user",
        rememberApiKey: true
      })
    )
  })
})
