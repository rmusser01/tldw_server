import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterAll, beforeEach, describe, expect, it, vi } from "vitest"
import { Storage } from "@plasmohq/storage"
import {
  invalidateRefreshSessionIfCurrent,
  storeRefreshRotationIfCurrent
} from "@/services/tldw/single-user-credential"
import { COOKIE_SESSION_CONFIG_KEY } from "@/services/tldw/browser-networking"

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
const firstRun = vi.hoisted(() => ({ realGate: false, profiles: vi.fn() }))
let mockLogoutAvailable = true
let currentConfig: Record<string, unknown> | null = null
let useCanonicalAuthStorage = false

vi.mock("next/router", () => ({
  useRouter: () => mockRouter
}))

const buddyLifetime = vi.hoisted(() => ({
  mounted: vi.fn(),
  unmounted: vi.fn()
}))
vi.mock("next/dynamic", () => ({
  default: (loader: () => unknown) => {
    if (loader.toString().includes("IndependentBuddyHost"))
      return function BuddyHostProbe() {
        const [draft, setDraft] = React.useState("")
        const [readAloud, setReadAloud] = React.useState(false)
        React.useEffect(() => {
          buddyLifetime.mounted()
          return () => {
            buddyLifetime.unmounted()
          }
        }, [])
        return (
          <div data-testid="persistent-buddy">
            <input
              aria-label="Buddy draft"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
            />
            <input
              type="checkbox"
              aria-label="Read Buddy aloud"
              checked={readAloud}
              onChange={(event) => setReadAloud(event.target.checked)}
            />
          </div>
        )
      }
    return ({
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
  }
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

vi.mock("@/services/tldw/request-core", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/tldw/request-core")>()
  return { ...actual, tldwRequest: (...args: Parameters<typeof actual.tldwRequest>) =>
    firstRun.realGate ? firstRun.profiles(args[0]) : actual.tldwRequest(...args) }
})

vi.mock("@/components/PersonaGarden/FirstRunGate", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/PersonaGarden/FirstRunGate")>()
  return {
  FirstRunGate: ({
    children,
    bypass,
    allowCompletedSetup,
    onStartSetup
  }: {
    children: React.ReactNode
    bypass?: boolean
    allowCompletedSetup?: boolean
    onStartSetup: () => void
  }) => firstRun.realGate ? (
    <actual.FirstRunGate bypass={bypass} allowCompletedSetup={allowCompletedSetup} onStartSetup={onStartSetup}>
      {children}
    </actual.FirstRunGate>
  ) : (
    <div data-testid="first-run-gate" data-bypass={String(Boolean(bypass))}
      data-allow-completed-setup={String(Boolean(allowCompletedSetup))}>
      <button
        type="button"
        data-testid="first-run-gate-start"
        onClick={onStartSetup}>
        Start setup
      </button>
      {children}
    </div>
  )
  }
})

vi.mock("@web/lib/configured-auth-state", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@web/lib/configured-auth-state")>()),
  loadConfiguredAuthConfig: async () =>
    useCanonicalAuthStorage
      ? (
          await importOriginal<
            typeof import("@web/lib/configured-auth-state")
          >()
        ).loadConfiguredAuthConfig()
      : mockGetConfig(),
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
  firstRun.realGate = false
  firstRun.profiles.mockReset().mockResolvedValue({ ok: true, data: [{}] })
  buddyLifetime.mounted.mockClear()
  buddyLifetime.unmounted.mockClear()
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
  useCanonicalAuthStorage = false
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
  describe("UAT159 actual first-run request boundary", () => {
    beforeEach(() => { firstRun.realGate = true })

    it.each(["/", "/media"])("does not request private profiles from blank-key %s", async route => {
      currentConfig = { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", apiKey: "  " }
      renderApp(route)
      expect(await screen.findByTestId("page-content")).toBeInTheDocument()
      expect(firstRun.profiles).not.toHaveBeenCalled()
    })

    it("waits for auth bootstrap before requesting profiles", async () => {
      resetRuntimeBootstrap(true)
      currentConfig = { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", apiKey: "valid-key" }
      renderApp("/media")
      await act(async () => { await Promise.resolve() })
      expect(firstRun.profiles).not.toHaveBeenCalled()
      await act(async () => { resolveRuntimeBootstrap?.(); await runtimeBootstrapReady })
      await waitFor(() => expect(firstRun.profiles).toHaveBeenCalledTimes(1))
    })

    it.each(["manual key", "runtime key", "quickstart cookie", "hosted cookie"])("keeps the authenticated %s profile check", async mode => {
      currentConfig = { serverUrl: "http://127.0.0.1:8000", authMode: "single-user" }
      if (mode === "manual key") currentConfig.apiKey = "valid-key"
      if (mode === "runtime key") mockRuntimeApiKey = "runtime-key"
      if (mode === "quickstart cookie") {
        process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
        currentConfig = { serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" }
      }
      if (mode === "hosted cookie") {
        process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "hosted"
        currentConfig = { serverUrl: "", authMode: "multi-user" }
      }
      renderApp("/media")
      await waitFor(() => expect(firstRun.profiles).toHaveBeenCalledTimes(1))
      expect(firstRun.profiles).toHaveBeenCalledWith({ path: "/api/v1/persona/profiles", method: "GET" })
      expect(mockGetCurrentUser).toHaveBeenCalledTimes(mode === "hosted cookie" ? 1 : 0)
    })

    it.each(["/", "/research-workspace", "/login", "/setup", "/settings/tldw"])("does not request unused profiles from bypassed %s", async route => {
      currentConfig = { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", apiKey: "valid-key" }
      renderApp(route)
      expect(await screen.findByTestId("page-content")).toBeInTheDocument()
      expect(firstRun.profiles).not.toHaveBeenCalled()
    })

    it("does not request profiles after hosted cookie validation rejects the session", async () => {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "hosted"
      currentConfig = { serverUrl: "", authMode: "multi-user" }
      mockGetCurrentUser.mockRejectedValue(makeStatusError("Unauthorized", 401))
      renderApp("/media")
      await screen.findByRole("heading", { name: "Signed out" })
      expect(firstRun.profiles).not.toHaveBeenCalled()
    })

    it("keeps one profile check through stable authenticated rerenders and auth refreshes", async () => {
      currentConfig = { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", apiKey: "valid-key" }
      const view = renderApp("/media")
      await waitFor(() => expect(firstRun.profiles).toHaveBeenCalledTimes(1))
      view.rerender(<App Component={DummyPage} pageProps={{ revision: 1 }} />)
      view.rerender(<App Component={DummyPage} pageProps={{ revision: 2 }} />)
      act(() => { window.dispatchEvent(new Event("tldw:config-updated")) })
      await waitFor(() => expect(mockGetConfig).toHaveBeenCalledTimes(2))
      expect(firstRun.profiles).toHaveBeenCalledTimes(1)
    })

    it("discards the old profile response after logout and checks again on authenticated re-entry", async () => {
      currentConfig = { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", apiKey: "first-key" }
      let resolveOld!: (value: unknown) => void
      const oldResponse = new Promise(resolve => { resolveOld = resolve })
      firstRun.profiles.mockReturnValueOnce(oldResponse)
      renderApp("/media")
      await waitFor(() => expect(firstRun.profiles).toHaveBeenCalledTimes(1))
      currentConfig = { ...currentConfig, apiKey: "" }
      act(() => { window.dispatchEvent(new Event("tldw:config-updated")) })
      await waitFor(() => expect(screen.getByTestId("option-layout")).toHaveAttribute("data-hide-header", "true"))
      currentConfig = { ...currentConfig, apiKey: "second-key" }
      act(() => { window.dispatchEvent(new Event("tldw:config-updated")) })
      await waitFor(() => expect(firstRun.profiles).toHaveBeenCalledTimes(2))
      await act(async () => { resolveOld({ ok: true, data: [] }); await oldResponse })
      expect(screen.queryByTestId("first-run-gate-overlay")).toBeNull()
      expect(screen.getByTestId("page-content")).toBeInTheDocument()
    })
  })

  describe("canonical browser authentication on the existing app owner", () => {
    const signedIn = {
      serverUrl: "https://shell.example.test",
      authMode: "multi-user" as const,
      authSource: "manual" as const,
      accessToken: "bob-access",
      refreshToken: "bob-refresh"
    }
    const signedOut = {
      ...signedIn,
      accessToken: undefined,
      refreshToken: undefined
    }
    const nativeConfigWrite = (config: typeof signedOut | typeof signedIn) => {
      const newValue = JSON.stringify(config)
      localStorage.setItem("tldwConfig", newValue)
      window.dispatchEvent(
        new StorageEvent("storage", { key: "tldwConfig", newValue })
      )
    }
    const expectHeader = (hidden: boolean) =>
      waitFor(() =>
        expect(screen.getByTestId("option-layout")).toHaveAttribute(
          "data-hide-header",
          String(hidden)
        )
      )
    beforeEach(() => {
      useCanonicalAuthStorage = true
    })

    it.each(["same-tab", "cross-tab"] as const)(
      "restores the Settings shell after %s login despite a stale client and preserves the draft",
      async (channel) => {
        const storage = new Storage({ area: "local" })
        await storage.set("tldwConfig", signedOut)
        const cachedClient = new TldwApiClient()
        await cachedClient.initialize()
        mockGetConfig.mockImplementation(() => cachedClient.getConfig())
        function SettingsDraft() {
          const [draft, setDraft] = React.useState("")
          return (
            <input
              aria-label="Settings draft"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
            />
          )
        }
        mockRouter.pathname = "/settings/tldw"
        mockRouter.asPath = "/settings/tldw"
        render(<App Component={SettingsDraft} pageProps={{}} />)
        await expectHeader(true)
        fireEvent.change(screen.getByLabelText("Settings draft"), {
          target: { value: "unsaved timeout and server settings" }
        })
        await act(async () => {
          if (channel === "same-tab") await storage.set("tldwConfig", signedIn)
          else nativeConfigWrite(signedIn)
        })
        expect((await cachedClient.getConfig())?.accessToken).toBeUndefined()
        await expectHeader(false)
        expect(screen.getByTestId("app-providers")).toHaveAttribute(
          "data-enable-notifications",
          "true"
        )
        expect(screen.getByLabelText("Settings draft")).toHaveValue(
          "unsaved timeout and server settings"
        )
      }
    )

    it("hides private shell consumers for an invalidated rotated pair and restores only the new login", async () => {
      const storage = new Storage({ area: "local" })
      await storage.set("tldwConfig", signedIn)
      currentConfig = signedIn
      renderApp("/settings/tldw")
      await expectHeader(false)
      const rotated = {
        accessToken: "rotated-access",
        refreshToken: "rotated-refresh"
      }
      await act(async () => {
        await storeRefreshRotationIfCurrent(
          storage,
          signedIn,
          signedIn.refreshToken,
          rotated
        )
        await invalidateRefreshSessionIfCurrent(storage, {
          ...signedIn,
          ...rotated
        })
      })
      await expectHeader(true)
      expect(screen.queryByTestId("persistent-buddy")).toBeNull()
      expect(screen.getByTestId("app-providers")).toHaveAttribute(
        "data-enable-notifications",
        "false"
      )
      await act(async () =>
        nativeConfigWrite({
          ...signedIn,
          accessToken: "new-login",
          refreshToken: "new-refresh"
        })
      )
      await expectHeader(false)
    })

    it.each(["success", "unauthorized"] as const)(
      "ignores a delayed %s validation after canonical logout",
      async (outcome) => {
        const storage = new Storage({ area: "local" })
        await storage.set("tldwConfig", signedIn)
        currentConfig = signedIn
        let finish!: () => void
        mockGetCurrentUser.mockReturnValueOnce(
          new Promise((resolve, reject) => {
            finish = () =>
              outcome === "success"
                ? resolve({ id: 2 })
                : reject(makeStatusError("Expired", 401))
          })
        )
        renderApp("/settings/tldw")
        await waitFor(() => expect(mockGetCurrentUser).toHaveBeenCalledTimes(1))
        await act(async () => nativeConfigWrite(signedOut))
        await expectHeader(true)
        await act(async () => finish())
        await expectHeader(true)
        expect(mockLogout).not.toHaveBeenCalled()
        expect(screen.getByTestId("app-providers")).toHaveAttribute(
          "data-enable-notifications",
          "false"
        )
      }
    )

    it("keeps the new A login after A → B → A despite the old A validation failing late", async () => {
      const storage = new Storage({ area: "local" })
      await storage.set("tldwConfig", signedIn)
      let rejectOldLogin!: () => void
      mockGetCurrentUser.mockReturnValueOnce(
        new Promise((_, reject) => {
          rejectOldLogin = () => reject(makeStatusError("Expired", 401))
        })
      )
      renderApp("/settings/tldw")
      await waitFor(() => expect(mockGetCurrentUser).toHaveBeenCalledTimes(1))

      await act(async () =>
        nativeConfigWrite({
          ...signedIn,
          accessToken: "alice-access",
          refreshToken: "alice-refresh"
        })
      )
      await expectHeader(false)
      const validationsBeforeReturn = mockGetCurrentUser.mock.calls.length
      await act(async () =>
        nativeConfigWrite({
          ...signedIn,
          accessToken: "bob-new-access",
          refreshToken: "bob-new-refresh"
        })
      )
      await waitFor(() =>
        expect(mockGetCurrentUser.mock.calls.length).toBeGreaterThan(
          validationsBeforeReturn
        )
      )
      await expectHeader(false)
      await act(async () => rejectOldLogin())

      await expectHeader(false)
      expect(mockLogout).not.toHaveBeenCalled()
      expect(screen.getByTestId("app-providers")).toHaveAttribute(
        "data-enable-notifications",
        "true"
      )
      expect(
        (await storage.get<typeof signedIn>("tldwConfig"))?.accessToken
      ).toBe("bob-new-access")
    })

    it("ignores a delayed canonical configuration read after another account signs out", async () => {
      const storage = new Storage({ area: "local" })
      await storage.set("tldwConfig", signedIn)
      const originalGet = Storage.prototype.get
      let completeOldRead!: () => void
      let readStarted = false
      const getSpy = vi
        .spyOn(Storage.prototype, "get")
        .mockImplementation(function (key) {
          if (key === "tldwConfig" && !readStarted) {
            readStarted = true
            const oldRead = originalGet.call(this, key)
            return new Promise((resolve) => {
              completeOldRead = () => {
                void oldRead.then(resolve)
              }
            })
          }
          return originalGet.call(this, key)
        })
      try {
        renderApp("/settings/tldw")
        await waitFor(() => expect(readStarted).toBe(true))
        await act(async () =>
          nativeConfigWrite({
            ...signedOut,
            serverUrl: "https://other-server.example.test"
          })
        )
        await expectHeader(true)
        await act(async () => completeOldRead())

        await expectHeader(true)
        expect(mockGetCurrentUser).not.toHaveBeenCalled()
        expect(screen.getByTestId("app-providers")).toHaveAttribute(
          "data-enable-notifications",
          "false"
        )
      } finally {
        getSpy.mockRestore()
      }
    })

    it("follows real quickstart cookie activation and removal without a cached client", async () => {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
      const storage = new Storage({ area: "local" })
      renderApp("/settings/tldw")
      await expectHeader(true)

      await act(async () =>
        storage.set(COOKIE_SESSION_CONFIG_KEY, {
          serverUrl: window.location.origin,
          authMode: "single-user",
          authSource: "cookie-session"
        })
      )
      await expectHeader(false)
      await act(async () => storage.remove(COOKIE_SESSION_CONFIG_KEY))
      await expectHeader(true)
      expect(mockGetCurrentUser).not.toHaveBeenCalled()
      expect(screen.getByTestId("app-providers")).toHaveAttribute(
        "data-enable-notifications",
        "false"
      )
    })
  })

  it("wraps non-login routes with OptionLayout", async () => {
    process.env.NEXT_PUBLIC_X_API_KEY = "env-api-key"
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
    process.env.NEXT_PUBLIC_X_API_KEY = "env-api-key"
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
    process.env.NEXT_PUBLIC_X_API_KEY = "env-api-key"
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

  it.each(["/knowledge", "/chat"])("allows completed unified setup on source destination %s", async (path) => {
    process.env.NEXT_PUBLIC_X_API_KEY = "env-api-key"
    renderApp(path)
    const gate = await screen.findByTestId("first-run-gate")
    expect(gate).toHaveAttribute("data-allow-completed-setup", "true")
    expect(gate).toHaveAttribute("data-bypass", "false")
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

    await waitFor(() => {
      expect(mockGetCurrentUser).toHaveBeenCalled()
    })
    await waitFor(() => {
      expect(mockRouter.push).toHaveBeenCalledWith("/login")
    })
    expect(screen.getByRole("heading", { name: "Signed out" })).toBeInTheDocument()
    expect(screen.queryByTestId("page-content")).toBeNull()
    expect(mockLogout).toHaveBeenCalled()
  })

  it.each(["storage", "tldw:config-updated", "expired-session"])(
    "replaces private content after an offline auth boundary via %s and preserves queued drafts",
    async (eventName) => {
      const online = vi.spyOn(navigator, "onLine", "get").mockReturnValue(true)
      const queueKey = "tldw:notesOfflineDraftQueue:v1:alice"
      const queuedDraft = JSON.stringify({ draft: { content: "Alice queued note" } })
      currentConfig = {
        serverUrl: "http://server.invalid",
        authMode: "multi-user",
        accessToken: "alice-token"
      }
      try {
        renderApp("/notes")
        await screen.findByTestId("page-content")
        localStorage.setItem(queueKey, queuedDraft)
        const oldValue = JSON.stringify(currentConfig)
        currentConfig = { ...currentConfig, accessToken: undefined }
        online.mockReturnValue(false)
        act(() => {
          window.dispatchEvent(eventName === "expired-session"
            ? new StorageEvent("storage", {
                key: "tldwInvalidRefreshSession:synthetic-session-digest", newValue: "true"
              })
            : eventName === "storage"
            ? new StorageEvent("storage", {
                key: "tldwConfig", oldValue, newValue: JSON.stringify(currentConfig)
              })
            : new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
        })

        await screen.findByRole("heading", { name: "Signed out" })
        expect(screen.getByText(/reconnect to sign in/i)).toBeInTheDocument()
        expect(screen.queryByTestId("page-content")).toBeNull()
        expect(screen.queryByTestId("persistent-buddy")).toBeNull()
        expect(screen.queryByTestId("server-readiness-gate")).toBeNull()
        expect(screen.queryByTestId("first-run-gate")).toBeNull()
        expect(mockRouter.push).not.toHaveBeenCalled()
        expect(localStorage.getItem(queueKey)).toBe(queuedDraft)
      } finally {
        online.mockRestore()
      }
    }
  )

  it("resumes the normal login redirect when a signed-out tab comes online", async () => {
    const online = vi.spyOn(navigator, "onLine", "get").mockReturnValue(false)
    currentConfig = { serverUrl: "http://server.invalid", authMode: "multi-user" }
    try {
      renderApp("/notes")
      await screen.findByRole("heading", { name: "Signed out" })
      expect(mockRouter.push).not.toHaveBeenCalled()

      online.mockReturnValue(true)
      act(() => { window.dispatchEvent(new Event("online")) })
      await waitFor(() => expect(mockRouter.push).toHaveBeenCalledWith("/login"))
      expect(screen.queryByTestId("page-content")).toBeNull()
    } finally {
      online.mockRestore()
    }
  })

  it.each(["success", "network-error", "unauthorized"])(
    "ignores a late %s auth result after a newer offline logout",
    async (outcome) => {
      const online = vi.spyOn(navigator, "onLine", "get").mockReturnValue(false)
      currentConfig = {
        serverUrl: "http://server.invalid", authMode: "multi-user", accessToken: "alice-token"
      }
      let resolveValidation!: (value: unknown) => void
      let rejectValidation!: (reason: unknown) => void
      const validation = new Promise((resolve, reject) => {
        resolveValidation = resolve
        rejectValidation = reject
      })
      try {
        renderApp("/notes")
        await screen.findByTestId("page-content")
        mockGetCurrentUser.mockReturnValueOnce(validation)
        act(() => { window.dispatchEvent(new Event("focus")) })
        await waitFor(() => expect(mockGetCurrentUser).toHaveBeenCalledTimes(2))

        currentConfig = { ...currentConfig, accessToken: undefined }
        act(() => {
          window.dispatchEvent(new StorageEvent("storage", {
            key: "tldwConfig", newValue: JSON.stringify(currentConfig)
          }))
        })
        await waitFor(() => expect(mockGetConfig).toHaveBeenCalledTimes(3))
        await act(async () => {
          if (outcome === "success") resolveValidation({ id: 2, username: "alice" })
          else rejectValidation(makeStatusError(outcome, outcome === "unauthorized" ? 401 : 0))
        })

        expect(screen.getByRole("heading", { name: "Signed out" })).toBeInTheDocument()
        expect(screen.queryByTestId("page-content")).toBeNull()
        expect(mockLogout).not.toHaveBeenCalled()
        expect(mockRouter.push).not.toHaveBeenCalled()
      } finally {
        online.mockRestore()
      }
    }
  )

  it.each(["/settings/tldw", "/login", "/setup", "/__debug__/sidepanel-chat"])(
    "keeps the signed-out %s route available offline",
    async (pathname) => {
      const online = vi.spyOn(navigator, "onLine", "get").mockReturnValue(false)
      currentConfig = { serverUrl: "http://server.invalid", authMode: "multi-user" }
      try {
        renderApp(pathname)
        await screen.findByTestId("page-content")
        expect(screen.queryByRole("heading", { name: "Signed out" })).toBeNull()
        expect(mockRouter.push).not.toHaveBeenCalled()
      } finally {
        online.mockRestore()
      }
    }
  )

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

describe("persistent authenticated Buddy placement", () => {
  it("keeps one Buddy lifetime through page and gate changes and clears it on logout", async () => {
    currentConfig = {
      serverUrl: "http://server.invalid",
      authMode: "single-user",
      apiKey: "test-key"
    }
    const { rerender } = renderApp("/persona")
    const draft = await screen.findByLabelText("Buddy draft")
    fireEvent.change(draft, { target: { value: "Private unsent draft" } })
    fireEvent.click(screen.getByLabelText("Read Buddy aloud"))
    for (const path of ["/notes", "/settings", "/persona"]) {
      mockRouter.pathname = path
      mockRouter.asPath = path
      rerender(<App Component={DummyPage} pageProps={{}} />)
      expect(screen.getByLabelText("Buddy draft")).toBe(draft)
      expect(screen.getByLabelText("Buddy draft")).toHaveValue(
        "Private unsent draft"
      )
      expect(screen.getByLabelText("Read Buddy aloud")).toBeChecked()
    }
    expect(buddyLifetime.mounted).toHaveBeenCalledTimes(1)
    currentConfig = null
    fireEvent(window, new Event("tldw:config-updated"))
    await waitFor(() =>
      expect(screen.queryByTestId("persistent-buddy")).not.toBeInTheDocument()
    )
    expect(buddyLifetime.unmounted).toHaveBeenCalledTimes(1)
  })
  it.each(["/login", "/setup", "/__debug__/sidepanel-chat"])(
    "does not mount the web Buddy on %s",
    async (path) => {
      currentConfig = {
        serverUrl: "http://server.invalid",
        authMode: "single-user",
        apiKey: "test-key"
      }
      renderApp(path)
      await screen.findByTestId("app-providers")
      expect(screen.queryByTestId("persistent-buddy")).not.toBeInTheDocument()
    }
  )
})
