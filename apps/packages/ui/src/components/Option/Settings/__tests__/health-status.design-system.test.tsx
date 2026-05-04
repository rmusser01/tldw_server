// @vitest-environment jsdom
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import HealthStatus from "../health-status"

const apiSendMock = vi.hoisted(() => vi.fn())
const clientMock = vi.hoisted(() => ({
  getConfig: vi.fn(),
  initialize: vi.fn(),
  healthCheck: vi.fn(),
  chatQueueStatus: vi.fn(),
  chatQueueActivity: vi.fn()
}))
const connectionUxMock = vi.hoisted(() => ({
  uxState: "connected",
  errorKind: null as string | null
}))
const connectionStateMock = vi.hoisted(() => ({
  serverUrl: "http://127.0.0.1:8000",
  lastStatusCode: null as number | null,
  lastError: null as string | null
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      maybeOptions?: { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) return fallbackOrOptions.defaultValue
      if (maybeOptions?.defaultValue) return maybeOptions.defaultValue
      return _key
    }
  })
}))

vi.mock("react-router-dom", () => ({
  Link: ({ children, to }: { children: React.ReactNode; to: string }) => (
    <a href={to}>{children}</a>
  ),
  useNavigate: () => vi.fn()
}))

vi.mock("wxt/browser", () => ({
  browser: {
    tabs: {
      create: vi.fn()
    }
  }
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: clientMock
}))

vi.mock("@/services/api-send", () => ({
  apiSend: apiSendMock
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    warning: vi.fn(),
    error: vi.fn()
  })
}))

vi.mock("@/components/Common/ServerOverviewHint", () => ({
  ServerOverviewHint: () => <div>Server overview hint</div>
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => connectionStateMock,
  useConnectionUxState: () => connectionUxMock
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: {
      hasChatQueue: false
    }
  })
}))

const mockMatchMedia = () => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

const renderHealth = async () => {
  render(<HealthStatus />)
  await waitFor(() => {
    expect(apiSendMock).toHaveBeenCalledTimes(7)
  })
}

describe("HealthStatus design-system states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockMatchMedia()
    connectionUxMock.uxState = "connected"
    connectionUxMock.errorKind = null
    connectionStateMock.serverUrl = "http://127.0.0.1:8000"
    connectionStateMock.lastStatusCode = null
    connectionStateMock.lastError = null
    clientMock.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000"
    })
    clientMock.initialize.mockResolvedValue(undefined)
    clientMock.healthCheck.mockResolvedValue(true)
    apiSendMock.mockResolvedValue({
      ok: true,
      status: 200,
      data: { status: "ok" }
    })
  })

  it("renders Ready when every health check passes", async () => {
    await renderHealth()

    await waitFor(() => {
      expect(screen.getAllByText("Ready").length).toBeGreaterThan(0)
    })
    expect(screen.getAllByText("/api/v1/health").length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Raw response from/).length).toBeGreaterThan(0)
  })

  it("renders Degraded when only part of the health surface is failing", async () => {
    apiSendMock.mockImplementation(({ path }: { path: string }) =>
      Promise.resolve(
        path === "/api/v1/rag/health"
          ? { ok: false, status: 503, error: { detail: "RAG offline" } }
          : { ok: true, status: 200, data: { status: "ok" } }
      )
    )

    await renderHealth()

    await waitFor(() => {
      expect(screen.getAllByText("Degraded").length).toBeGreaterThan(0)
    })
    expect(screen.getByText(/RAG offline/)).toBeInTheDocument()
  })

  it("renders Unavailable for unreachable connection callouts", async () => {
    connectionUxMock.uxState = "error_unreachable"
    connectionUxMock.errorKind = "unreachable"
    connectionStateMock.lastStatusCode = 0
    connectionStateMock.lastError = "Network error"

    await renderHealth()

    expect(await screen.findByText("Unavailable")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Edit server URL" })).toBeInTheDocument()
  })

  it("renders Sign in required for auth callouts", async () => {
    connectionUxMock.uxState = "error_auth"
    connectionUxMock.errorKind = "auth"
    connectionStateMock.lastStatusCode = 401
    connectionStateMock.lastError = "Unauthorized"

    await renderHealth()

    expect(await screen.findByText("Sign in required")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Fix API key" })).toBeInTheDocument()
  })
})
