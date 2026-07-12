import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterAll, beforeEach, describe, expect, it, vi } from "vitest"

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

let resolveRuntimeBootstrap: (() => void) | null = null
let rejectRuntimeBootstrap: ((reason?: unknown) => void) | null = null
let runtimeBootstrapReady: Promise<void> = Promise.resolve()

const resetRuntimeBootstrap = (deferred = false) => {
  if (!deferred) {
    runtimeBootstrapReady = Promise.resolve()
    resolveRuntimeBootstrap = null
    rejectRuntimeBootstrap = null
    return
  }

  runtimeBootstrapReady = new Promise<void>((resolve, reject) => {
    resolveRuntimeBootstrap = resolve
    rejectRuntimeBootstrap = reject
  })
}

vi.mock("@web/extension/shims/runtime-bootstrap", () => ({
  get runtimeBootstrapReady() {
    return runtimeBootstrapReady
  }
}))

let mockRuntimeApiKey: string | null = null

vi.mock("@web/lib/authStorage", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@web/lib/authStorage")>()
  return {
    ...actual,
    getRuntimeApiKey: () => mockRuntimeApiKey
  }
})

import App from "@web/pages/_app"
import { TldwApiClient } from "@/services/tldw/TldwApiClient"

const mockRouter = {
  pathname: "/media",
  asPath: "/media",
  push: vi.fn(),
  replace: vi.fn(),
  prefetch: vi.fn(() => Promise.resolve(true))
}

const mockGetConfig = vi.fn()
const mockGetCurrentUser = vi.fn()
const mockLogout = vi.fn()
let mockLogoutAvailable = true
let currentConfig: Record<string, unknown> | null = null

vi.mock("next/router", () => ({
  useRouter: () => mockRouter
}))

vi.mock("next/dynamic", () => ({
  default: () =>
    ({
      children,
      hideHeader,
      hideSidebar
    }: {
      children: React.ReactNode
      hideHeader?: boolean
      hideSidebar?: boolean
    }) => (
      <div
        data-testid="option-layout"
        data-hide-header={String(Boolean(hideHeader))}
        data-hide-sidebar={String(Boolean(hideSidebar))}>
        {children}
      </div>
    )
}))

vi.mock("@web/components/AppProviders", () => ({
  AppProviders: ({
    children,
    enableNotifications
  }: {
    children: React.ReactNode
    enableNotifications?: boolean
  }) => (
    <div
      data-testid="app-providers"
      data-enable-notifications={String(Boolean(enableNotifications))}
    >
      {children}
    </div>
  )
}))

vi.mock("@/components/Common/PageAssistLoader", () => ({
  PageAssistLoader: ({ label }: { label?: string }) => (
    <div role="status" aria-busy="true">
      {label || "Loading…"}
    </div>
  )
}))

vi.mock("@web/components/networking/ServerReadinessGate", () => ({
  ServerReadinessGate: ({
    children,
    allowDegraded
  }: {
    children: React.ReactNode
    allowDegraded?: boolean
  }) => (
    <div
      data-testid="server-readiness-gate"
      data-allow-degraded={String(Boolean(allowDegraded))}
    >
      {children}
    </div>
  )
}))

vi.mock("@/components/PersonaGarden/FirstRunGate", () => ({
  FirstRunGate: ({
    children,
    bypass,
    onStartSetup
  }: {
    children: React.ReactNode
    bypass?: boolean
    onStartSetup: () => void
  }) => (
    <div data-testid="first-run-gate" data-bypass={String(Boolean(bypass))}>
      <button
        type="button"
        data-testid="first-run-gate-start"
        onClick={onStartSetup}>
        Start setup
      </button>
      {children}
    </div>
  )
}))

vi.mock("@web/lib/configured-auth-state", () => ({
  loadTldwClient: async () => ({
    getConfig: (...args: unknown[]) => mockGetConfig(...args)
  }),
  loadTldwAuth: async () => ({
    getCurrentUser: (...args: unknown[]) => mockGetCurrentUser(...args),
    ...(mockLogoutAvailable
      ? { logout: (...args: unknown[]) => mockLogout(...args) }
      : {})
  })
}))

const DummyPage = () => <div data-testid="page-content">Page</div>

const renderApp = (pathname: string, asPath = pathname) => {
  mockRouter.pathname = pathname
  mockRouter.asPath = asPath
  return render(<App Component={DummyPage} pageProps={{}} />)
}

const makeStatusError = (
  message: string,
  status: number
): Error & { status: number } => Object.assign(new Error(message), { status })

const originalEnvApiKey = process.env.NEXT_PUBLIC_X_API_KEY
const originalEnvBearer = process.env.NEXT_PUBLIC_API_BEARER
const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE

beforeEach(() => {
  localStorage.clear()
  sessionStorage.clear()
  mockRouter.push.mockClear()
  mockRouter.replace.mockClear()
  mockRouter.prefetch.mockClear()
  mockGetConfig.mockReset()
  mockGetCurrentUser.mockReset()
  mockLogout.mockReset()
  mockLogoutAvailable = true
  mockGetCurrentUser.mockResolvedValue({ username: "test-user" })
  mockLogout.mockResolvedValue(undefined)
  currentConfig = null
  mockGetConfig.mockImplementation(async () => currentConfig)
  delete process.env.NEXT_PUBLIC_X_API_KEY
  delete process.env.NEXT_PUBLIC_API_BEARER
  delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  mockRuntimeApiKey = null
  resetRuntimeBootstrap()
})

afterAll(() => {
  process.env.NEXT_PUBLIC_X_API_KEY = originalEnvApiKey
  process.env.NEXT_PUBLIC_API_BEARER = originalEnvBearer
  process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
})

describe("App layout routing", () => {
  it("wraps non-login routes with OptionLayout", async () => {
    renderApp("/media")
    expect(
      await screen.findByTestId("server-readiness-gate")
    ).toBeInTheDocument()
    expect(screen.getByTestId("server-readiness-gate")).toHaveAttribute(
      "data-allow-degraded",
      "false"
    )
    expect(screen.getByTestId("first-run-gate")).toBeInTheDocument()
    expect(screen.getByTestId("first-run-gate")).toHaveAttribute(
      "data-bypass",
      "false"
    )
    expect(await screen.findByTestId("option-layout")).toBeInTheDocument()
    expect(screen.getByTestId("page-content")).toBeInTheDocument()
  })

  it("allows degraded server readiness on chat and research workspace routes", async () => {
    const { rerender } = renderApp("/chat")
    await screen.findByTestId("server-readiness-gate")
    expect(screen.getByTestId("server-readiness-gate")).toHaveAttribute(
      "data-allow-degraded",
      "true"
    )

    mockRouter.pathname = "/research-workspace"
    mockRouter.asPath = "/research-workspace"
    rerender(<App Component={DummyPage} pageProps={{}} />)

    expect(screen.getByTestId("server-readiness-gate")).toHaveAttribute(
      "data-allow-degraded",
      "true"
    )

    mockRouter.pathname = "/media"
    mockRouter.asPath = "/media"
    rerender(<App Component={DummyPage} pageProps={{}} />)

    expect(screen.getByTestId("server-readiness-gate")).toHaveAttribute(
      "data-allow-degraded",
      "false"
    )
  })

  it("routes first-time chat setup through the unified setup shell", async () => {
    renderApp("/chat")
    await screen.findByTestId("server-readiness-gate")

    expect(screen.getByTestId("server-readiness-gate")).toHaveAttribute(
      "data-allow-degraded",
      "true"
    )
    expect(screen.getByTestId("first-run-gate")).toHaveAttribute(
      "data-bypass",
      "false"
    )

    fireEvent.click(screen.getByTestId("first-run-gate-start"))

    expect(mockRouter.push).toHaveBeenCalledWith("/")
  })

  it("routes first-time media setup through the unified setup shell", async () => {
    renderApp("/media")
    await screen.findByTestId("first-run-gate")

    expect(screen.getByTestId("first-run-gate")).toHaveAttribute(
      "data-bypass",
      "false"
    )

    fireEvent.click(screen.getByTestId("first-run-gate-start"))

    expect(mockRouter.push).toHaveBeenCalledWith("/")
  })

  it("lets the unified setup host route bypass the generic first-run overlay", async () => {
    renderApp("/")
    await screen.findByTestId("first-run-gate")

    expect(screen.getByTestId("first-run-gate")).toHaveAttribute(
      "data-bypass",
      "true"
    )
  })

  it("bypasses the generic first-run splash for character-chat route intent", async () => {
    renderApp("/characters")
    await screen.findByTestId("server-readiness-gate")

    expect(screen.getByTestId("server-readiness-gate")).toBeInTheDocument()
    expect(screen.getByTestId("first-run-gate")).toHaveAttribute(
      "data-bypass",
      "true"
    )

    fireEvent.click(screen.getByTestId("first-run-gate-start"))

    expect(mockRouter.push).toHaveBeenCalledWith(
      "/?intent=character-chat&returnTo=%2Fcharacters"
    )
  })

  it("bypasses the generic first-run splash for Research Workspace direct entry", async () => {
    renderApp("/research-workspace")
    await screen.findByTestId("server-readiness-gate")

    expect(screen.getByTestId("server-readiness-gate")).toBeInTheDocument()
    expect(screen.getByTestId("first-run-gate")).toHaveAttribute(
      "data-bypass",
      "true"
    )
  })

  it("preserves explicit character-chat onboarding routes through first-run setup", async () => {
    renderApp(
      "/",
      "/?intent=character-chat&returnTo=%2Fcharacters%3Ffrom%3Dheader-select%26create%3Dtrue"
    )
    await screen.findByTestId("first-run-gate")

    expect(screen.getByTestId("first-run-gate")).toHaveAttribute(
      "data-bypass",
      "true"
    )

    fireEvent.click(screen.getByTestId("first-run-gate-start"))

    expect(mockRouter.push).toHaveBeenCalledWith(
      "/?intent=character-chat&returnTo=%2Fcharacters%3Ffrom%3Dheader-select%26create%3Dtrue"
    )
  })

  it("skips OptionLayout for /login but keeps ServerReadinessGate mounted", async () => {
    renderApp("/login")
    await screen.findByTestId("server-readiness-gate")
    expect(screen.getByTestId("server-readiness-gate")).toBeInTheDocument()
    expect(screen.queryByTestId("first-run-gate")).toBeNull()
    expect(screen.queryByTestId("option-layout")).toBeNull()
    expect(screen.getByTestId("page-content")).toBeInTheDocument()
  })

  it("hides header and sidebar while unauthenticated", async () => {
    renderApp("/media")
    const layout = await screen.findByTestId("option-layout")
    await waitFor(() => {
      expect(layout).toHaveAttribute("data-hide-header", "true")
    })
    expect(layout).toHaveAttribute("data-hide-sidebar", "true")
  })

  it("waits for runtime bootstrap before reading configured auth state", async () => {
    resetRuntimeBootstrap(true)
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "runtime-key"
    }

    renderApp("/media")

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(mockGetConfig).not.toHaveBeenCalled()

    await act(async () => {
      resolveRuntimeBootstrap?.()
      await runtimeBootstrapReady
    })

    await waitFor(() => {
      expect(mockGetConfig).toHaveBeenCalled()
    })
  })

  it("does not mount config consumers until deferred bootstrap activates the real cookie client", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        serverUrl: "https://remote.example.test",
        authMode: "single-user",
        apiKey: "manual-device-key",
        credentialSource: "manual",
        apiKeyPersistence: "device",
        apiKeyServerOrigin: "https://remote.example.test"
      })
    )
    const client = new TldwApiClient()
    mockGetConfig.mockImplementation(() => client.getConfig())
    resetRuntimeBootstrap(true)

    const ConfigConsumerPage = () => {
      const [source, setSource] = React.useState("pending")
      React.useEffect(() => {
        void client.getConfig().then((config) => {
          setSource(config?.authSource || "manual")
        })
      }, [])
      return <div data-testid="client-auth-source">{source}</div>
    }

    render(<App Component={ConfigConsumerPage} pageProps={{}} />)

    expect(screen.getByRole("status")).toHaveTextContent("Loading")
    expect(screen.queryByTestId("app-providers")).toBeNull()
    expect(screen.queryByTestId("client-auth-source")).toBeNull()

    localStorage.setItem(
      "tldwCookieSessionConfig",
      JSON.stringify({
        serverUrl: window.location.origin,
        authMode: "single-user",
        authSource: "cookie-session"
      })
    )
    await act(async () => {
      resolveRuntimeBootstrap?.()
      await runtimeBootstrapReady
    })

    expect(await screen.findByTestId("client-auth-source")).toHaveTextContent(
      "cookie-session"
    )
    expect(JSON.parse(String(localStorage.getItem("tldwConfig")))).toEqual(
      expect.objectContaining({
        serverUrl: "https://remote.example.test",
        apiKey: "manual-device-key"
      })
    )
  })

  it("leaves startup loading after a rejected bootstrap and resolves auth fail closed", async () => {
    resetRuntimeBootstrap(true)
    currentConfig = null

    renderApp("/media")

    expect(screen.getByRole("status")).toHaveTextContent("Loading")
    expect(screen.queryByTestId("app-providers")).toBeNull()

    await act(async () => {
      rejectRuntimeBootstrap?.(new Error("bootstrap unavailable"))
      await runtimeBootstrapReady.catch(() => undefined)
    })

    expect(await screen.findByTestId("app-providers")).toBeInTheDocument()
    expect(screen.getByTestId("option-layout")).toHaveAttribute(
      "data-hide-header",
      "true"
    )
  })

  it("exits startup loading within the bootstrap bound and mounts preserved manual configuration fail closed", async () => {
    vi.useFakeTimers()
    try {
      const manualConfig = {
        serverUrl: "https://remote.example.test",
        authMode: "single-user",
        credentialSource: "manual"
      }
      currentConfig = manualConfig
      localStorage.setItem("tldwConfig", JSON.stringify(manualConfig))
      runtimeBootstrapReady = new Promise<void>((resolve) => {
        setTimeout(resolve, 8_000)
      })

      renderApp("/media")

      expect(screen.getByRole("status")).toHaveTextContent("Loading")
      expect(screen.queryByTestId("app-providers")).toBeNull()

      await act(async () => {
        await vi.advanceTimersByTimeAsync(7_999)
      })
      expect(screen.getByRole("status")).toHaveTextContent("Loading")

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1)
      })
      vi.useRealTimers()

      expect(await screen.findByTestId("app-providers")).toBeInTheDocument()
      expect(screen.getByTestId("option-layout")).toHaveAttribute(
        "data-hide-header",
        "true"
      )
      expect(mockGetConfig).toHaveBeenCalled()
      expect(JSON.parse(String(localStorage.getItem("tldwConfig")))).toEqual(
        manualConfig
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it("treats a probed quickstart cookie session as authenticated without an api key", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_X_API_KEY = "stale-public-key"
    currentConfig = {
      serverUrl: window.location.origin,
      authMode: "single-user",
      authSource: "cookie-session"
    }

    renderApp("/media")
    const layout = await screen.findByTestId("option-layout")

    await waitFor(() => {
      expect(layout).toHaveAttribute("data-hide-header", "false")
    })
    expect(layout).toHaveAttribute("data-hide-sidebar", "false")
  })

  it("does not authenticate missing quickstart config with a stale public key", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_X_API_KEY = "stale-public-key"
    currentConfig = null

    renderApp("/media")
    const layout = await screen.findByTestId("option-layout")

    await waitFor(() => {
      expect(layout).toHaveAttribute("data-hide-header", "true")
    })
    expect(layout).toHaveAttribute("data-hide-sidebar", "true")
  })

  it("keeps sidebar hidden on settings routes even when authenticated", async () => {
    process.env.NEXT_PUBLIC_X_API_KEY = "env-api-key"

    renderApp("/settings/tldw")
    const layout = await screen.findByTestId("option-layout")
    expect(screen.getByTestId("server-readiness-gate")).toBeInTheDocument()
    expect(screen.queryByTestId("first-run-gate")).toBeNull()
    await waitFor(() => {
      expect(layout).toHaveAttribute("data-hide-header", "false")
    })
    expect(layout).toHaveAttribute("data-hide-sidebar", "true")
  })

  it("treats runtime single-user API key overrides as authenticated shell state", async () => {
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user"
    }
    mockRuntimeApiKey = "runtime-api-key"

    renderApp("/media")

    const layout = await screen.findByTestId("option-layout")
    await waitFor(() => {
      expect(layout).toHaveAttribute("data-hide-header", "false")
    })
    expect(layout).toHaveAttribute("data-hide-sidebar", "false")
  })

  it("keeps setup in a setup-only shell even when authenticated", async () => {
    process.env.NEXT_PUBLIC_X_API_KEY = "env-api-key"

    renderApp("/setup")
    const layout = await screen.findByTestId("option-layout")
    expect(screen.getByTestId("server-readiness-gate")).toBeInTheDocument()
    expect(screen.queryByTestId("first-run-gate")).toBeNull()
    await waitFor(() => {
      expect(layout).toHaveAttribute("data-hide-header", "true")
    })
    expect(layout).toHaveAttribute("data-hide-sidebar", "true")
    expect(screen.getByTestId("app-providers")).toHaveAttribute(
      "data-enable-notifications",
      "false"
    )
  })

  it("enables notification startup after auth resolves on app routes", async () => {
    process.env.NEXT_PUBLIC_X_API_KEY = "env-api-key"

    renderApp("/media")

    await waitFor(() => {
      expect(screen.getByTestId("app-providers")).toHaveAttribute(
        "data-enable-notifications",
        "true"
      )
    })
  })

  it("refreshes nav visibility when auth config updates", async () => {
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    }

    renderApp("/media")
    const layout = await screen.findByTestId("option-layout")

    await waitFor(() => {
      expect(layout).toHaveAttribute("data-hide-header", "true")
    })
    await waitFor(() => {
      expect(mockGetConfig).toHaveBeenCalled()
    })

    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-api-key"
    }

    act(() => {
      window.dispatchEvent(new CustomEvent("tldw:config-updated"))
    })

    await waitFor(() => {
      expect(layout).toHaveAttribute("data-hide-header", "false")
    })
    expect(layout).toHaveAttribute("data-hide-sidebar", "false")
  })

  it("counts runtime-override credentials as authenticated for shell chrome", async () => {
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    }
    mockRuntimeApiKey = "runtime-override-key"

    renderApp("/media")
    const layout = await screen.findByTestId("option-layout")

    await waitFor(() => {
      expect(mockGetConfig).toHaveBeenCalled()
    })
    await waitFor(() => {
      expect(layout).toHaveAttribute("data-hide-header", "false")
    })
    expect(layout).toHaveAttribute("data-hide-sidebar", "false")
  })

  it("redirects protected routes to login when multi-user token validation fails", async () => {
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      accessToken: "stale-token"
    }
    mockGetCurrentUser.mockRejectedValueOnce(makeStatusError("Unauthorized", 401))

    renderApp("/media")
    const layout = await screen.findByTestId("option-layout")

    await waitFor(() => {
      expect(mockGetCurrentUser).toHaveBeenCalled()
    })
    await waitFor(() => {
      expect(mockRouter.push).toHaveBeenCalledWith("/login")
    })
    expect(layout).toHaveAttribute("data-hide-header", "true")
    expect(layout).toHaveAttribute("data-hide-sidebar", "true")
    expect(mockLogout).toHaveBeenCalled()
  })

  it("redirects stale sessions when the auth provider has no logout method", async () => {
    mockLogoutAvailable = false
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      accessToken: "stale-token"
    }
    mockGetCurrentUser.mockRejectedValueOnce(makeStatusError("Unauthorized", 401))

    renderApp("/media")

    await waitFor(() => {
      expect(mockRouter.push).toHaveBeenCalledWith("/login")
    })
    expect(mockLogout).not.toHaveBeenCalled()
  })

  it("logs logout failures while still redirecting stale sessions", async () => {
    const logoutError = new Error("Storage unavailable")
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      accessToken: "stale-token"
    }
    mockGetCurrentUser.mockRejectedValueOnce(makeStatusError("Unauthorized", 401))
    mockLogout.mockRejectedValueOnce(logoutError)

    try {
      renderApp("/media")

      await waitFor(() => {
        expect(mockRouter.push).toHaveBeenCalledWith("/login")
      })
      expect(warn).toHaveBeenCalledWith(
        "Failed to clear stale tldw auth session:",
        logoutError
      )
    } finally {
      warn.mockRestore()
    }
  })

  it("redirects when auth validation returns a plain unauthenticated error", async () => {
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      accessToken: "stale-token"
    }
    mockGetCurrentUser.mockRejectedValueOnce({ message: "Not authenticated" })

    renderApp("/media")

    await waitFor(() => {
      expect(mockRouter.push).toHaveBeenCalledWith("/login")
    })
    expect(mockLogout).toHaveBeenCalled()
  })

  it("keeps persisted multi-user auth when validation fails with a non-auth status", async () => {
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      accessToken: "still-valid-token"
    }
    mockGetCurrentUser.mockRejectedValueOnce(
      makeStatusError("Server unavailable", 500)
    )

    renderApp("/media")
    const layout = await screen.findByTestId("option-layout")

    await waitFor(() => {
      expect(mockGetCurrentUser).toHaveBeenCalled()
    })
    await waitFor(() => {
      expect(layout).toHaveAttribute("data-hide-header", "false")
    })
    expect(layout).toHaveAttribute("data-hide-sidebar", "false")
    expect(mockLogout).not.toHaveBeenCalled()
    expect(mockRouter.push).not.toHaveBeenCalledWith("/login")
  })

  it("validates hosted multi-user sessions without a persisted access token", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "hosted"
    currentConfig = {
      serverUrl: "",
      authMode: "multi-user"
    }

    renderApp("/chat")
    const layout = await screen.findByTestId("option-layout")

    await waitFor(() => {
      expect(mockGetCurrentUser).toHaveBeenCalled()
    })
    await waitFor(() => {
      expect(layout).toHaveAttribute("data-hide-header", "false")
    })
    expect(layout).toHaveAttribute("data-hide-sidebar", "false")
    expect(mockLogout).not.toHaveBeenCalled()
    expect(mockRouter.push).not.toHaveBeenCalledWith("/login")
  })

  it("keeps hosted tokenless auth on non-auth validation failures", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "hosted"
    currentConfig = {
      serverUrl: "",
      authMode: "multi-user"
    }
    mockGetCurrentUser.mockRejectedValueOnce(
      makeStatusError("Server unavailable", 500)
    )

    renderApp("/chat")
    const layout = await screen.findByTestId("option-layout")

    await waitFor(() => {
      expect(mockGetCurrentUser).toHaveBeenCalled()
    })
    await waitFor(() => {
      expect(layout).toHaveAttribute("data-hide-header", "false")
    })
    expect(layout).toHaveAttribute("data-hide-sidebar", "false")
    expect(mockLogout).not.toHaveBeenCalled()
    expect(mockRouter.push).not.toHaveBeenCalledWith("/login")
  })

  it("warms primary navigation routes after auth resolves", async () => {
    process.env.NEXT_PUBLIC_X_API_KEY = "env-api-key"

    const originalRequestIdleCallback = (
      window as Window & {
        requestIdleCallback?: (callback: () => void) => number
      }
    ).requestIdleCallback
    const originalCancelIdleCallback = (
      window as Window & {
        cancelIdleCallback?: (handle: number) => void
      }
    ).cancelIdleCallback

    ;(
      window as Window & {
        requestIdleCallback?: (callback: () => void) => number
      }
    ).requestIdleCallback = (callback: () => void) => {
      callback()
      return 1
    }
    ;(
      window as Window & {
        cancelIdleCallback?: (handle: number) => void
      }
    ).cancelIdleCallback = vi.fn()

    try {
      renderApp("/media")
      await waitFor(() => {
        expect(mockRouter.prefetch).toHaveBeenCalledWith("/chat")
      })
    } finally {
      ;(
        window as Window & {
          requestIdleCallback?: (callback: () => void) => number
        }
      ).requestIdleCallback = originalRequestIdleCallback
      ;(
        window as Window & {
          cancelIdleCallback?: (handle: number) => void
        }
      ).cancelIdleCallback = originalCancelIdleCallback
    }
  })
})
