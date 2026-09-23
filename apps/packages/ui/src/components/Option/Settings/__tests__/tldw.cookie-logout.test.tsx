// @vitest-environment jsdom

import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { Storage } from "@plasmohq/storage"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import {
  REFRESH_ROTATION_KEY,
  invalidateRefreshSessionIfCurrent,
  refreshSessionInvalidationKey,
  storeRefreshRotationIfCurrent
} from "@/services/tldw/single-user-credential"
import { COOKIE_SESSION_CONFIG_KEY } from "@/services/tldw/browser-networking"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// Match the WebUI storage boundary, including same-tab and cross-tab watches.
vi.mock("@plasmohq/storage", async () =>
  import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage")
)

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  logout: vi.fn(),
  probeServerHealth: vi.fn(),
  apiSend: vi.fn(),
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
  validateFields: vi.fn(async (fields?: string[]) => Object.fromEntries(Object.entries(formValues).filter(([key]) => !fields || fields.includes(key))))
}

const configuredClient = (config: Partial<TldwConfig>) => {
  localStorage.setItem("tldwConfig", JSON.stringify(config))
  mocks.getConfig.mockReset().mockResolvedValue(config)
}

vi.mock("antd", () => {
  const Form = Object.assign(
    ({ children, onValuesChange, onFinish, component }: {
      component?: false; children?: React.ReactNode; onValuesChange?: (changed: Record<string, unknown>) => void; onFinish?: (values: Record<string, unknown>) => void
    }) => component === false ? <div onChange={() => onValuesChange?.(form.getFieldsValue())}>{children}</div> : <form onChange={() => onValuesChange?.(form.getFieldsValue())} onSubmit={(event) => {
      event.preventDefault()
      onFinish?.(form.getFieldsValue())
    }}>{children}</form>,
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

vi.mock("@/services/tldw/TldwApiClient", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/tldw/TldwApiClient")>(),
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
vi.mock("@/services/api-send", () => ({ apiSend: mocks.apiSend }))
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
  TldwTimeoutSettings: ({ requestTimeoutSec, setRequestTimeoutSec }: {
    requestTimeoutSec: number; setRequestTimeoutSec: (value: number) => void
  }) => <input aria-label="Request timeout" type="number" value={requestTimeoutSec}
    onChange={(event) => setRequestTimeoutSec(Number(event.target.value))} />
}))
vi.mock("../TldwBillingSettings", () => ({
  TldwBillingSettings: () => <div>Billing controls</div>
}))
vi.mock("../TldwConnectionSettings", () => ({
  TldwConnectionSettings: (props: {
    configuredServerUrl: string
    authSource?: string
    rememberApiKey: boolean
    authMode: string
    loginMethod: string
    connectionStatus: string | null
    connectionDetail: string
    isLoggedIn: boolean
    coreStatus: string
    ragStatus: string
    onTestConnection: () => void
    onLogout: () => void
  }) => (
    <div>
      <span data-testid="server-url">{props.configuredServerUrl}</span>
      <span data-testid="auth-source">{props.authSource ?? "manual"}</span>
      <span data-testid="auth-mode">{props.authMode}</span>
      <span data-testid="login-method">{props.loginMethod}</span>
      <span data-testid="remember-key">{String(props.rememberApiKey)}</span>
      <span data-testid="connection-status">{props.connectionStatus ?? "none"}</span>
      <span data-testid="connection-detail">{props.connectionDetail}</span>
      <span data-testid="core-status">{props.coreStatus}</span>
      <span data-testid="rag-status">{props.ragStatus}</span>
      <span data-testid="login-status">{props.isLoggedIn ? "Logged In" : "Login Required"}</span>
      <input aria-label="Edit server" onChange={(event) => { formValues.serverUrl = event.target.value }} />
      {props.authSource !== "cookie-session" && <span>Manual key controls</span>}
      <button type="button" onClick={props.onTestConnection}>Test connection</button>
      <button type="button" onClick={props.onLogout}>Logout</button>
    </div>
  )
}))

import { TldwSettings } from "../tldw"

describe("TldwSettings cookie logout", () => {
  afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); vi.unstubAllEnvs(); vi.restoreAllMocks() })
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    sessionStorage.clear()
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
    mocks.apiSend.mockResolvedValue({ ok: true, status: 200, data: {} })
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

  it("tests an unverified authenticated session without profile permissions and defaults to password login", async () => {
    configuredClient({ serverUrl: "http://127.0.0.1:8000", authMode: "multi-user", accessToken: "alice-token" })
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ paths: {} }), { status: 200, headers: { "Content-Type": "application/json" } })))
    mocks.apiSend.mockImplementation(async ({ path }) => path === "/api/v1/auth/sessions"
      ? { ok: true, status: 200, data: [] }
      : { ok: false, status: 403, error: "Email not verified" })
    render(<TldwSettings />)
    expect(await screen.findByTestId("login-method")).toHaveTextContent("password")
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }))
    await waitFor(() => expect(mocks.apiSend).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/auth/sessions", method: "GET" })))
    expect(mocks.probeServerHealth).not.toHaveBeenCalled()
    expect(screen.getByTestId("connection-status")).toHaveTextContent("success")
    vi.unstubAllGlobals()
  })

  it("does not load or render billing when the server does not advertise it", async () => {
    configuredClient({ serverUrl: "http://127.0.0.1:8000", authMode: "multi-user", accessToken: "alice-token" })
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ paths: {} }), { status: 200, headers: { "Content-Type": "application/json" } }))
    vi.stubGlobal("fetch", fetchMock)
    render(<TldwSettings />)
    await screen.findByTestId("auth-mode")
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(mocks.apiSend).not.toHaveBeenCalledWith(expect.objectContaining({ path: expect.stringContaining("/billing/") }))
    expect(screen.queryByText("Billing controls")).not.toBeInTheDocument()
    vi.unstubAllGlobals()
  })

  it("does not send a saved login to an edited foreign server URL", async () => {
    configuredClient({ serverUrl: "http://127.0.0.1:8000", authMode: "multi-user", accessToken: "alice-token" })
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ paths: {} }), { status: 200, headers: { "Content-Type": "application/json" } })))
    render(<TldwSettings />)
    await screen.findByTestId("auth-mode")
    formValues.serverUrl = "https://different.example"
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }))
    await waitFor(() => expect(mocks.probeServerHealth).toHaveBeenCalledWith(expect.objectContaining({ serverUrl: "https://different.example" })))
    expect(mocks.apiSend).not.toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/auth/sessions" }))
    vi.unstubAllGlobals()
  })

  it.each(["quickstart", "advanced"])("loads direct-backend billing only after all routes are advertised in %s mode", async mode => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", mode)
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    configuredClient({ serverUrl: "http://127.0.0.1:8000", authMode: "multi-user", accessToken: "alice-token" })
    const paths = Object.fromEntries(['plans', 'subscription', 'usage', 'invoices'].map(route => [`/api/v1/billing/${route}`, { get: {} }]))
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ paths }), { status: 200, headers: { "Content-Type": "application/json" } })))
    render(<TldwSettings />)
    expect(await screen.findByText("Billing controls")).toBeInTheDocument()
    await waitFor(() => expect(mocks.apiSend).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/billing/invoices?limit=20" })))
  })

  describe("billing OpenAPI discovery routing", () => {
    const billingSpec = () => new Response(JSON.stringify({
      paths: Object.fromEntries(["plans", "subscription", "usage", "invoices"].map(route => [`/api/v1/billing/${route}`, { get: {} }]))
    }), { status: 200, headers: { "Content-Type": "application/json" } })
    const noBilling = () => {
      expect(screen.queryByText("Billing controls")).not.toBeInTheDocument()
      expect(mocks.apiSend).not.toHaveBeenCalledWith(expect.objectContaining({ path: expect.stringContaining("/billing/") }))
    }

    it.each(["", "/"])("skips unsupported same-origin quickstart discovery with suffix '%s'", async suffix => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
      configuredClient({ serverUrl: `${window.location.origin}${suffix}`, authMode: "multi-user", accessToken: "synthetic-access" })
      const probe = vi.fn().mockResolvedValue(new Response("Not Found", { status: 404 }))
      vi.stubGlobal("fetch", probe)
      render(<TldwSettings />)
      await waitFor(() => expect(screen.getByTestId("login-status")).toHaveTextContent("Logged In"))
      fireEvent.click(screen.getByRole("button", { name: "Test connection", exact: true }))
      await waitFor(() => expect(screen.getByTestId("connection-status")).toHaveTextContent("success"))
      expect(probe).not.toHaveBeenCalled()
      noBilling()
    })

    it("keeps same-origin advanced-mode discovery when the backend advertises Billing", async () => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
      vi.stubEnv("NEXT_PUBLIC_API_URL", window.location.origin)
      configuredClient({ serverUrl: window.location.origin, authMode: "multi-user", accessToken: "synthetic-access" })
      const probe = vi.fn().mockImplementation(async () => billingSpec())
      vi.stubGlobal("fetch", probe)
      render(<TldwSettings />)
      expect(await screen.findByText("Billing controls")).toBeInTheDocument()
      expect(probe).toHaveBeenCalledWith(`${window.location.origin}/openapi.json`, expect.objectContaining({ signal: expect.any(AbortSignal) }))
      await waitFor(() => expect(mocks.apiSend).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/billing/invoices?limit=20" })))
    })

    it("treats a direct-backend 404 as absent optional Billing without repeated probes", async () => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
      configuredClient({ serverUrl: "https://backend.example.test", authMode: "multi-user", accessToken: "synthetic-access" })
      const probe = vi.fn().mockResolvedValue(new Response("Not Found", { status: 404 }))
      vi.stubGlobal("fetch", probe)
      render(<TldwSettings />)
      await waitFor(() => expect(probe).toHaveBeenCalledTimes(1))
      await act(async () => { await Promise.resolve() })
      noBilling()
      expect(screen.getByTestId("login-status")).toHaveTextContent("Logged In")
    })

    it("aborts a pending capability request on logout and ignores its late advertised result", async () => {
      configuredClient({ serverUrl: "https://backend.example.test", authMode: "multi-user", accessToken: "synthetic-access" })
      let release!: (response: Response) => void
      let signal!: AbortSignal
      vi.stubGlobal("fetch", vi.fn((_url, init) => {
        signal = init.signal
        return new Promise<Response>(resolve => { release = resolve })
      }))
      render(<TldwSettings />)
      await waitFor(() => expect(signal).toBeDefined())
      mocks.getConfig.mockResolvedValue(null)
      fireEvent.click(screen.getByRole("button", { name: "Logout", exact: true }))
      await waitFor(() => expect(signal.aborted).toBe(true))
      await act(async () => { release(billingSpec()) })
      noBilling()
      expect(screen.getByTestId("login-status")).toHaveTextContent("Login Required")
    })

    it("aborts a pending capability request when Settings unmounts", async () => {
      configuredClient({ serverUrl: "https://backend.example.test", authMode: "multi-user", accessToken: "synthetic-access" })
      let signal!: AbortSignal
      vi.stubGlobal("fetch", vi.fn((_url, init) => {
        signal = init.signal
        return new Promise<Response>(() => {})
      }))
      const mounted = render(<TldwSettings />)
      await waitFor(() => expect(signal).toBeDefined())
      mounted.unmount()
      expect(signal.aborted).toBe(true)
      expect(mocks.apiSend).not.toHaveBeenCalled()
    })

    it("aborts old-target discovery on save and ignores its late advertised result", async () => {
      const first = { serverUrl: "https://first.example.test", authMode: "multi-user" as const, accessToken: "synthetic-first" }
      const second = { serverUrl: "https://second.example.test", authMode: "multi-user" as const, accessToken: "synthetic-second" }
      configuredClient(first)
      let release!: (response: Response) => void
      let oldSignal!: AbortSignal
      const probe = vi.fn((url, init) => {
        if (url === `${first.serverUrl}/openapi.json`) {
          oldSignal = init.signal
          return new Promise<Response>(resolve => { release = resolve })
        }
        return Promise.resolve(new Response("Not Found", { status: 404 }))
      })
      vi.stubGlobal("fetch", probe)
      const mounted = render(<TldwSettings />)
      await waitFor(() => expect(oldSignal).toBeDefined())
      configuredClient(second)
      Object.assign(formValues, second)
      fireEvent.submit(mounted.container.querySelector("form")!)
      await waitFor(() => expect(screen.getByTestId("server-url")).toHaveTextContent(second.serverUrl))
      await waitFor(() => expect(probe).toHaveBeenCalledWith(`${second.serverUrl}/openapi.json`, expect.anything()))
      expect(oldSignal.aborted).toBe(true)
      await act(async () => { release(billingSpec()) })
      noBilling()
    })

    it("aborts an unresponsive capability request after five seconds", async () => {
      configuredClient({ serverUrl: "https://backend.example.test", authMode: "multi-user", accessToken: "synthetic-access" })
      vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] })
      let signal!: AbortSignal
      vi.stubGlobal("fetch", vi.fn((_url, init) => {
        signal = init.signal
        return new Promise<Response>((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")))
        })
      }))
      await act(async () => { render(<TldwSettings />) })
      expect(signal.aborted).toBe(false)
      await act(async () => { await vi.advanceTimersByTimeAsync(4999) })
      expect(signal.aborted).toBe(false)
      await act(async () => { await vi.advanceTimersByTimeAsync(1) })
      expect(signal.aborted).toBe(true)
      noBilling()
    })
  })

  it("describes rejected multi-user sessions without calling them invalid API keys", async () => {
    configuredClient({ serverUrl: "http://127.0.0.1:8000", authMode: "multi-user", accessToken: "expired-token" })
    mocks.apiSend.mockResolvedValue({ ok: false, status: 401, error: "Token expired" })
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ paths: {} }), { status: 200, headers: { "Content-Type": "application/json" } })))
    render(<TldwSettings />)
    await screen.findByTestId("auth-mode")
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }))
    await waitFor(() => expect(screen.getByTestId("connection-detail")).toHaveTextContent("Sign in again"))
    expect(screen.getByTestId("connection-detail")).not.toHaveTextContent("Invalid API key")
  })

  describe("effective login status on the mounted Settings owner", () => {
    const target: TldwConfig = {
      serverUrl: "https://settings.example.test", authMode: "multi-user", authSource: "manual", orgId: 7
    }
    const signedIn = { ...target, accessToken: "alice-access", refreshToken: "alice-refresh" }
    let storage: Storage
    beforeEach(() => {
      storage = new Storage({ area: "local" })
      configuredClient(target)
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ paths: {} }), { status: 200, headers: { "Content-Type": "application/json" } })))
    })

    const status = (expected: string) => waitFor(() =>
      expect(screen.getByTestId("login-status")).toHaveTextContent(expected)
    )
    const nativeWrite = (key: string, value: unknown) => {
      const newValue = value === undefined ? null : JSON.stringify(value)
      if (newValue === null) localStorage.removeItem(key)
      else localStorage.setItem(key, newValue)
      window.dispatchEvent(new StorageEvent("storage", { key, newValue }))
    }
    const writeConfig = async (config: TldwConfig, channel: "same-tab" | "cross-tab" | "config-event") => {
      if (channel === "same-tab") await storage.set("tldwConfig", config)
      else if (channel === "cross-tab") nativeWrite("tldwConfig", config)
      else {
        localStorage.setItem("tldwConfig", JSON.stringify(config))
        window.dispatchEvent(new CustomEvent("tldw:config-updated"))
      }
    }

    it.each(["same-tab", "cross-tab", "config-event"] as const)("tracks ordinary %s login and logout without replacing unsaved fields", async (channel) => {
      render(<TldwSettings />)
      await status("Login Required")
      formValues.apiKey = "unsaved-key"
      fireEvent.change(screen.getByLabelText("Request timeout"), { target: { value: "47" } })
      const formWrites = mocks.setFieldsValue.mock.calls.length
      await act(async () => writeConfig(signedIn, channel))
      await status("Logged In")
      await act(async () => writeConfig(target, channel))
      await status("Login Required")
      expect(formValues.apiKey).toBe("unsaved-key")
      expect(formValues.serverUrl).toBe(target.serverUrl)
      expect(screen.getByLabelText("Request timeout")).toHaveValue(47)
      expect(mocks.setFieldsValue).toHaveBeenCalledTimes(formWrites)
      expect(mocks.getConfig).toHaveBeenCalledTimes(1)
    })

    it("uses actual storage rather than a stale cached signed-in client config", async () => {
      mocks.getConfig.mockResolvedValue(signedIn)
      render(<TldwSettings />)
      await status("Login Required")
    })

    it.each(["same-tab", "cross-tab"] as const)("recovers after exact-pair %s rotation while the raw pair stays invalidated", async (channel) => {
      await storage.set("tldwConfig", signedIn)
      await invalidateRefreshSessionIfCurrent(storage, signedIn)
      render(<TldwSettings />)
      await status("Login Required")
      const tokens = { accessToken: "alice-rotated", refreshToken: "alice-rotated-refresh" }
      await act(async () => {
        if (channel === "same-tab") {
          await storeRefreshRotationIfCurrent(storage, signedIn, signedIn.refreshToken, tokens)
        } else {
          nativeWrite(REFRESH_ROTATION_KEY, {
            version: 1, ...signedIn, sourceAccessToken: signedIn.accessToken,
            sourceRefreshToken: signedIn.refreshToken, ...tokens
          })
        }
      })
      await status("Logged In")
      expect(formValues.serverUrl).toBe(target.serverUrl)
    })

    it.each(["same-tab", "cross-tab", "storage-only"] as const)("observes %s exact-pair invalidation", async (channel) => {
      await storage.set("tldwConfig", signedIn)
      render(<TldwSettings />)
      await status("Logged In")
      await act(async () => {
        if (channel === "same-tab") await invalidateRefreshSessionIfCurrent(storage, signedIn)
        else if (channel === "cross-tab") nativeWrite(refreshSessionInvalidationKey(signedIn)!, true)
        else await storage.set(refreshSessionInvalidationKey(signedIn)!, true)
      })
      await status("Login Required")
    })

    it("does not treat a foreign server login as authentication for the displayed server", async () => {
      render(<TldwSettings />)
      await status("Login Required")
      await act(async () => storage.set("tldwConfig", { ...signedIn, serverUrl: "https://foreign.example.test" }))
      await status("Login Required")
      expect(formValues.serverUrl).toBe(target.serverUrl)
      await act(async () => storage.set("tldwConfig", signedIn))
      await status("Logged In")
    })

    it("keeps valid offline authentication", async () => {
      await storage.set("tldwConfig", signedIn)
      vi.mocked(fetch).mockRejectedValue(new TypeError("Failed to fetch"))
      render(<TldwSettings />)
      await status("Logged In")
    })

    it.each(["account", "server"] as const)("ignores a delayed signed-in read across A-to-B-to-A %s changes", async (boundary) => {
      await storage.set("tldwConfig", signedIn)
      render(<TldwSettings />)
      await status("Logged In")
      let release!: () => void
      const blocked = new Promise<void>((resolve) => { release = resolve })
      const originalGet = Storage.prototype.get
      let capture = true
      const reads = vi.spyOn(Storage.prototype, "get").mockImplementation(async function (key) {
        const value = await originalGet.call(this, key)
        if (key === "tldwConfig" && capture) { capture = false; await blocked }
        return value
      })
      act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated")))
      await waitFor(() => expect(reads).toHaveBeenCalledWith("tldwConfig"))
      await act(async () => {
        await storage.set("tldwConfig", {
          ...signedIn, accessToken: "bob",
          serverUrl: boundary === "server" ? "https://foreign.example.test" : signedIn.serverUrl
        })
        await storage.set("tldwConfig", target)
      })
      await status("Login Required")
      await act(async () => release())
      await status("Login Required")
      expect(formValues.serverUrl).toBe(target.serverUrl)
    })

    it("revalidates an edited target and ignores an earlier pending auth read", async () => {
      await storage.set("tldwConfig", signedIn)
      render(<TldwSettings />)
      await status("Logged In")
      let release!: () => void
      const blocked = new Promise<void>((resolve) => { release = resolve })
      const originalGet = Storage.prototype.get
      let capture = true
      const reads = vi.spyOn(Storage.prototype, "get").mockImplementation(async function (key) {
        const value = await originalGet.call(this, key)
        if (key === "tldwConfig" && capture) { capture = false; await blocked }
        return value
      })
      act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated")))
      await waitFor(() => expect(reads).toHaveBeenCalledWith("tldwConfig"))
      fireEvent.change(screen.getByLabelText("Edit server"), { target: { value: "https://unsaved.example.test" } })
      await status("Login Required")
      await act(async () => release())
      await status("Login Required")
      expect(formValues.serverUrl).toBe("https://unsaved.example.test")
    })

    it("revalidates cookie ownership without reloading the form", async () => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
      await storage.set("tldwConfig", signedIn)
      render(<TldwSettings />)
      await status("Logged In")
      await act(async () => storage.set(COOKIE_SESSION_CONFIG_KEY, {
        serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session"
      }))
      await status("Login Required")
      await act(async () => storage.remove(COOKIE_SESSION_CONFIG_KEY))
      await status("Logged In")
      expect(formValues.serverUrl).toBe(target.serverUrl)
    })

    it("cleans up storage and window subscriptions under StrictMode", async () => {
      await storage.set("tldwConfig", signedIn)
      const { unmount } = render(<React.StrictMode><TldwSettings /></React.StrictMode>)
      await status("Logged In")
      unmount()
      const reads = vi.spyOn(Storage.prototype, "get")
      await act(async () => {
        await storage.set("tldwConfig", target)
        await storage.set(refreshSessionInvalidationKey(signedIn)!, true)
        nativeWrite(REFRESH_ROTATION_KEY, null)
        window.dispatchEvent(new CustomEvent("tldw:config-updated"))
      })
      expect(reads.mock.contexts.every((context) => context === storage)).toBe(true)
    })

    it("does not let an older initial load replace fields edited after StrictMode setup", async () => {
      let finishOld!: (config: TldwConfig) => void
      mocks.getConfig.mockReset()
        .mockImplementationOnce(() => new Promise<TldwConfig>((resolve) => { finishOld = resolve }))
        .mockResolvedValue(target)
      render(<React.StrictMode><TldwSettings /></React.StrictMode>)
      await status("Login Required")
      formValues.apiKey = "unsaved-key"
      fireEvent.change(screen.getByLabelText("Edit server"), { target: { value: "https://unsaved.example.test" } })
      fireEvent.change(screen.getByLabelText("Request timeout"), { target: { value: "47" } })
      const formWrites = mocks.setFieldsValue.mock.calls.length
      await act(async () => finishOld({ ...signedIn, serverUrl: "https://stale.example.test", requestTimeoutMs: 1000 }))
      expect(formValues.serverUrl).toBe("https://unsaved.example.test")
      expect(formValues.apiKey).toBe("unsaved-key")
      expect(screen.getByLabelText("Request timeout")).toHaveValue(47)
      expect(mocks.setFieldsValue).toHaveBeenCalledTimes(formWrites)
      await status("Login Required")
    })

    it("supports extension-style boolean watch results and removes obsolete marker subscriptions", async () => {
      const originalWatch = Storage.prototype.watch
      const watched: Array<{ storage: Storage; callbacks: Parameters<Storage["watch"]>[0] }> = []
      vi.spyOn(Storage.prototype, "watch").mockImplementation(function (callbacks) {
        watched.push({ storage: this, callbacks })
        originalWatch.call(this, callbacks)
        return true as unknown as ReturnType<Storage["watch"]>
      })
      await storage.set("tldwConfig", signedIn)
      const { unmount } = render(<TldwSettings />)
      try {
        await status("Logged In")
        const rotated = { accessToken: "alice-next", refreshToken: "alice-next-refresh" }
        await act(async () => storeRefreshRotationIfCurrent(storage, signedIn, signedIn.refreshToken, rotated))
        await status("Logged In")
        const marker = refreshSessionInvalidationKey({ ...signedIn, ...rotated })!
        await act(async () => storage.set(marker, true))
        await status("Login Required")
        const reads = vi.spyOn(Storage.prototype, "get")
        await act(async () => storage.set(marker, true))
        expect(reads.mock.contexts.every((context) => context === storage)).toBe(true)
        expect(() => unmount()).not.toThrow()
      } finally {
        watched.forEach(({ storage: watchedStorage, callbacks }) => watchedStorage.unwatch(callbacks))
      }
    })
  })
})
