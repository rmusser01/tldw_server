import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@web/lib/i18n-web", () => ({}))

vi.mock("wxt/browser", () => ({
  browser: {
    storage: {
      local: {
        get: vi.fn(async () => ({}))
      }
    }
  }
}))

vi.mock("@web/extension/shims/runtime-bootstrap", () => ({
  runtimeBootstrapReady: Promise.resolve()
}))

const mockRouter = {
  pathname: "/media",
  asPath: "/media",
  push: vi.fn(),
  replace: vi.fn(),
  prefetch: vi.fn(() => Promise.resolve(true))
}

vi.mock("next/router", () => ({
  useRouter: () => mockRouter
}))

vi.mock("next/dynamic", () => ({
  default: (loader: () => unknown) => {
    const isBuddy = loader.toString().includes("IndependentBuddyHost")
    return function DynamicStub(props: { children?: React.ReactNode }) {
      return isBuddy ? (
        <div data-testid="persistent-buddy" />
      ) : (
        <div data-testid="option-layout">{props.children}</div>
      )
    }
  }
}))

vi.mock("@web/components/AppProviders", () => ({
  AppProviders: ({
    children
  }: {
    children: React.ReactNode
  }) => <div data-testid="app-providers">{children}</div>
}))

vi.mock("@/components/Common/PageAssistLoader", () => ({
  PageAssistLoader: ({ label }: { label?: string }) => (
    <div role="status" aria-busy="true">
      {label || "Loading…"}
    </div>
  )
}))

vi.mock("@/components/PersonaGarden/FirstRunGate", () => ({
  FirstRunGate: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="first-run-gate">{children}</div>
  )
}))

const mockGetConfig = vi.fn()
const mockGetCurrentUser = vi.fn()
const mockLogout = vi.fn()
let currentConfig: Record<string, unknown> | null = null

vi.mock("@web/lib/configured-auth-state", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@web/lib/configured-auth-state")
  >()),
  loadConfiguredAuthConfig: async () => mockGetConfig(),
  loadTldwAuth: async () => ({
    getCurrentUser: (...args: unknown[]) => mockGetCurrentUser(...args),
    logout: (...args: unknown[]) => mockLogout(...args)
  })
}))

// NOTE: ServerReadinessGate is intentionally NOT mocked here — these tests
// cover the real _app ⇄ readiness-gate integration (perf remediation W1).
import App from "@web/pages/_app"

const DummyPage = () => <div data-testid="page-content">Page</div>

const renderApp = (pathname = "/media") => {
  mockRouter.pathname = pathname
  mockRouter.asPath = pathname
  return render(<App Component={DummyPage} pageProps={{}} />)
}

const okHealth = () =>
  ({ ok: true, status: 200, json: async () => ({ status: "ok" }) }) as Response
const unavailableHealth = () =>
  ({ ok: false, status: 503, json: async () => ({ status: "unavailable" }) }) as
    Response

const originalEnvApiKey = process.env.NEXT_PUBLIC_X_API_KEY
const originalEnvBearer = process.env.NEXT_PUBLIC_API_BEARER
const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
const originalApiUrl = process.env.NEXT_PUBLIC_API_URL

beforeEach(() => {
  localStorage.clear()
  sessionStorage.clear()
  mockRouter.push.mockClear()
  mockGetConfig.mockReset()
  mockGetCurrentUser.mockReset()
  mockLogout.mockReset()
  currentConfig = null
  mockGetConfig.mockImplementation(async () => currentConfig)
  mockGetCurrentUser.mockResolvedValue({ username: "test-user" })
  mockLogout.mockResolvedValue(undefined)
  delete process.env.NEXT_PUBLIC_X_API_KEY
  delete process.env.NEXT_PUBLIC_API_BEARER
  delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  delete process.env.NEXT_PUBLIC_API_URL
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
  vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

afterAll(() => {
  process.env.NEXT_PUBLIC_X_API_KEY = originalEnvApiKey
  process.env.NEXT_PUBLIC_API_BEARER = originalEnvBearer
  process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
  process.env.NEXT_PUBLIC_API_URL = originalApiUrl
})

describe("App auth + readiness gate parallelism (W1)", () => {
  it("runs the health probe in parallel with auth validation and renders children without a duplicate probe", async () => {
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      accessToken: "stored-token"
    }
    let resolveAuth!: (value: unknown) => void
    mockGetCurrentUser.mockReturnValue(
      new Promise((resolve) => {
        resolveAuth = resolve
      })
    )
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(okHealth())

    renderApp()

    // Health fired while auth/me is still pending → parallel, not serial.
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/health",
        expect.objectContaining({ method: "GET" })
      )
    })
    // No page content while auth is unresolved.
    expect(screen.queryByTestId("page-content")).toBeNull()
    expect(screen.getByRole("status")).toHaveTextContent("Loading")

    await act(async () => {
      resolveAuth({ id: 1, username: "alice" })
      await Promise.resolve()
    })

    await waitFor(() => {
      expect(screen.getByTestId("page-content")).toBeInTheDocument()
    })
    // The gate reused the warmed probe: exactly one health request overall.
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("renders children behind a reconnect banner when health fails for an established session", async () => {
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      accessToken: "stored-token"
    }
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(unavailableHealth())

    renderApp()

    // Established session: content renders even though health is failing.
    await waitFor(() => {
      expect(screen.getByTestId("page-content")).toBeInTheDocument()
    })
    await waitFor(() => {
      expect(screen.getByTestId("server-reconnect-banner")).toBeInTheDocument()
    })
    // Non-blocking: no full-screen readiness spinner or recovery panel.
    expect(screen.queryByText(/Checking server readiness/i)).toBeNull()
    expect(screen.queryByText(/Backend readiness check failed/i)).toBeNull()
    expect(fetchMock).toHaveBeenCalled()
  })

  it("keeps children mounted behind the banner after the retry deadline instead of the recovery panel", async () => {
    vi.useFakeTimers()
    try {
      currentConfig = {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "multi-user",
        accessToken: "stored-token"
      }
      vi.spyOn(globalThis, "fetch").mockResolvedValue(unavailableHealth())

      renderApp()

      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })
      expect(screen.getByTestId("page-content")).toBeInTheDocument()
      expect(screen.getByTestId("server-reconnect-banner")).toBeInTheDocument()

      await act(async () => {
        await vi.advanceTimersByTimeAsync(16_000)
      })

      expect(screen.getByTestId("page-content")).toBeInTheDocument()
      expect(screen.getByTestId("server-reconnect-banner")).toHaveTextContent(
        /could not reach the tldw server/i
      )
      expect(
        screen.queryByText(/Backend readiness check failed/i)
      ).toBeNull()
    } finally {
      vi.useRealTimers()
    }
  })

  it("preserves the fully blocking readiness screen on cold starts without stored auth", async () => {
    vi.useFakeTimers()
    try {
      currentConfig = {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: ""
      }
      const fetchMock = vi
        .spyOn(globalThis, "fetch")
        .mockResolvedValue(unavailableHealth())

      renderApp()

      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })

      // No session → children stay hidden behind the blocking spinner.
      expect(screen.queryByTestId("page-content")).toBeNull()
      expect(screen.getByRole("status")).toHaveTextContent(
        /Checking server readiness|Retrying server readiness/
      )
      expect(screen.queryByTestId("server-reconnect-banner")).toBeNull()

      await act(async () => {
        await vi.advanceTimersByTimeAsync(16_000)
      })

      // Server down at startup → recovery panel, children never rendered.
      expect(screen.queryByTestId("page-content")).toBeNull()
      expect(
        screen.getByRole("heading", { name: /Backend readiness check failed/i })
      ).toBeInTheDocument()
      expect(fetchMock).toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })

  it("reuses the warmed probe for established single-user sessions", async () => {
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "stored-key"
    }
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(okHealth())

    renderApp()

    await waitFor(() => {
      expect(screen.getByTestId("page-content")).toBeInTheDocument()
    })
    // Warm (during auth bootstrap) + gate reuse → one health request total.
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/health",
      expect.objectContaining({ method: "GET" })
    )
  })
})
