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

import App from "@web/pages/_app"

const mockRouter = {
  pathname: "/media",
  asPath: "/media",
  push: vi.fn(),
  replace: vi.fn(),
  prefetch: vi.fn(() => Promise.resolve(true))
}

const mockGetConfig = vi.fn()
const mockGetCurrentUser = vi.fn()
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
  AppProviders: ({ children }: { children: React.ReactNode }) => <>{children}</>
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
    getCurrentUser: (...args: unknown[]) => mockGetCurrentUser(...args)
  })
}))

const DummyPage = () => <div data-testid="page-content">Page</div>

const renderApp = (pathname: string, asPath = pathname) => {
  mockRouter.pathname = pathname
  mockRouter.asPath = asPath
  return render(<App Component={DummyPage} pageProps={{}} />)
}

const originalEnvApiKey = process.env.NEXT_PUBLIC_X_API_KEY
const originalEnvBearer = process.env.NEXT_PUBLIC_API_BEARER
const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE

beforeEach(() => {
  mockRouter.push.mockClear()
  mockRouter.replace.mockClear()
  mockRouter.prefetch.mockClear()
  mockGetConfig.mockReset()
  mockGetCurrentUser.mockReset()
  mockGetCurrentUser.mockResolvedValue({ username: "test-user" })
  currentConfig = null
  mockGetConfig.mockImplementation(async () => currentConfig)
  delete process.env.NEXT_PUBLIC_X_API_KEY
  delete process.env.NEXT_PUBLIC_API_BEARER
  delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
})

afterAll(() => {
  process.env.NEXT_PUBLIC_X_API_KEY = originalEnvApiKey
  process.env.NEXT_PUBLIC_API_BEARER = originalEnvBearer
  process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
})

describe("App layout routing", () => {
  it("wraps non-login routes with OptionLayout", async () => {
    renderApp("/media")
    expect(screen.getByTestId("server-readiness-gate")).toBeInTheDocument()
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

  it("allows degraded server readiness on chat and research workspace routes", () => {
    const { rerender } = renderApp("/chat")
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

  it("routes first-time chat setup through the unified setup shell", () => {
    renderApp("/chat")

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

  it("routes first-time media setup through the unified setup shell", () => {
    renderApp("/media")

    expect(screen.getByTestId("first-run-gate")).toHaveAttribute(
      "data-bypass",
      "false"
    )

    fireEvent.click(screen.getByTestId("first-run-gate-start"))

    expect(mockRouter.push).toHaveBeenCalledWith("/")
  })

  it("lets the unified setup host route bypass the generic first-run overlay", () => {
    renderApp("/")

    expect(screen.getByTestId("first-run-gate")).toHaveAttribute(
      "data-bypass",
      "true"
    )
  })

  it("bypasses the generic first-run splash for character-chat route intent", async () => {
    renderApp("/characters")

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

  it("preserves explicit character-chat onboarding routes through first-run setup", async () => {
    renderApp(
      "/",
      "/?intent=character-chat&returnTo=%2Fcharacters%3Ffrom%3Dheader-select%26create%3Dtrue"
    )

    expect(screen.getByTestId("first-run-gate")).toHaveAttribute(
      "data-bypass",
      "true"
    )

    fireEvent.click(screen.getByTestId("first-run-gate-start"))

    expect(mockRouter.push).toHaveBeenCalledWith(
      "/?intent=character-chat&returnTo=%2Fcharacters%3Ffrom%3Dheader-select%26create%3Dtrue"
    )
  })

  it("skips OptionLayout for /login but keeps ServerReadinessGate mounted", () => {
    renderApp("/login")
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

  it("keeps header and sidebar hidden when multi-user token validation fails", async () => {
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      accessToken: "stale-token"
    }
    mockGetCurrentUser.mockRejectedValueOnce(new Error("Unauthorized"))

    renderApp("/media")
    const layout = await screen.findByTestId("option-layout")

    await waitFor(() => {
      expect(mockGetCurrentUser).toHaveBeenCalled()
    })
    expect(layout).toHaveAttribute("data-hide-header", "true")
    expect(layout).toHaveAttribute("data-hide-sidebar", "true")
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
