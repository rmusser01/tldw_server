import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { App, ConfigProvider } from "antd"
import { Storage } from "@plasmohq/storage"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

vi.mock("@plasmohq/storage", async () =>
  import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage")
)
const mocks = vi.hoisted(() => ({ bgRequest: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.bgRequest }))
vi.mock("@/services/tldw-server", () => ({ DEFAULT_TLDW_API_KEY: "default-key" }))
vi.mock("@/services/api-send", () => ({ apiSend: vi.fn() }))
vi.mock("@/store/connection", () => ({ useConnectionStore: { getState: () => ({ checkOnce: vi.fn() }) } }))
vi.mock("@/components/Common/ServerOverviewHint", () => ({ ServerOverviewHint: () => null }))
vi.mock("../server-health-probe", () => ({ probeServerHealth: vi.fn() }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({
  t: (key: string, fallback: unknown) => typeof fallback === "string" ? fallback : key
}) }))
vi.mock("react-router-dom", () => ({
  Link: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useNavigate: () => vi.fn()
}))

import { tldwClient } from "@/services/tldw/TldwApiClient"
import { TldwSettings } from "../tldw"
import { COOKIE_SESSION_CONFIG_KEY } from "@/services/tldw/browser-networking"
import { activateCookieSessionConfig } from "@/services/tldw/runtime-auth-override"
import { MANUAL_SESSION_KEY } from "@/services/tldw/single-user-credential"

beforeEach(async () => {
  localStorage.clear()
  sessionStorage.clear()
  mocks.bgRequest.mockReset().mockResolvedValue({ authenticated: false })
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "extension")
  vi.stubEnv("NEXT_PUBLIC_X_API_KEY", "")
  let tail = Promise.resolve()
  const locks = { request: (_name: string, operation: () => unknown) => {
      const next = tail.then(operation)
      tail = next.then(() => undefined, () => undefined)
      return next
    } }
  vi.stubGlobal("navigator", new Proxy(window.navigator, {
    get: (target, key) => key === "locks" ? locks : Reflect.get(target, key, target)
  }))
  await tldwClient.initialize()
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

it.each(["device", "session"] as const)("settles the real %s credential Disconnect while retaining the server", async (persistence) => {
  const serverUrl = "https://disconnect.example.test"
  await tldwClient.saveManualSingleUserCredential({ serverUrl, apiKey: "synthetic-disconnect-key", persistence })
  const observed: unknown[] = []
  const storage = new Storage({ area: "local" })
  const callbacks: Parameters<Storage["watch"]>[0] = { tldwConfig: (change) => observed.push(change.newValue) }
  storage.watch(callbacks)
  try {
    render(<React.StrictMode><ConfigProvider theme={{ token: { motion: false } }}><App><TldwSettings /></App></ConfigProvider></React.StrictMode>)
    const disconnect = await screen.findByRole("button", { name: "Disconnect" })
    await waitFor(() => expect(screen.getByLabelText("API Key")).toHaveValue("synthetic-disconnect-key"))
    fireEvent.click(disconnect)
    await waitFor(() => expect(screen.getByLabelText("API Key")).toHaveValue(""))
    await waitFor(() => expect(disconnect).not.toHaveClass("ant-btn-loading"))
    expect(disconnect.querySelector('[aria-label="loading"]')).toBeNull()
    expect(await screen.findByText("Logged out successfully")).toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Server URL" })).toHaveValue(serverUrl)
    expect((await tldwClient.getConfig())?.apiKey).toBeUndefined()
    expect(await new Storage({ area: "session" }).get(MANUAL_SESSION_KEY)).toBeUndefined()
    expect(observed).toContainEqual(expect.objectContaining({ serverUrl }))
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    await act(async () => tldwClient.saveManualSingleUserCredential({ serverUrl, apiKey: "synthetic-reconnect-key", persistence }))
    expect((await tldwClient.getConfig())?.apiKey).toBe("synthetic-reconnect-key")
  } finally {
    storage.unwatch(callbacks)
  }
})


it.each([false, true])("settles cookie logout with remote failure=%s", async (fail) => {
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
  activateCookieSessionConfig()
  const storage = new Storage({ area: "local" })
  await storage.set(COOKIE_SESSION_CONFIG_KEY, {
    serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session"
  })
  await tldwClient.initialize()
  let release!: () => void
  mocks.bgRequest.mockImplementation(() => new Promise((resolve, reject) => {
    release = () => fail ? reject(new Error("remote logout unavailable")) : resolve({ authenticated: false })
  }))
  const errors = vi.spyOn(console, "error").mockImplementation(() => {})
  render(<ConfigProvider theme={{ token: { motion: false } }}><App><TldwSettings /></App></ConfigProvider>)
  const logout = await screen.findByRole("button", { name: "Logout" })
  fireEvent.click(logout)
  await waitFor(() => expect(logout).toHaveAttribute("aria-busy", "true"))
  expect(await storage.get(COOKIE_SESSION_CONFIG_KEY)).toBeDefined()
  await act(async () => release())
  await screen.findByText(fail ? "Logout failed" : "Logged out successfully", { exact: true })
  if (fail) {
    expect(logout).toHaveAttribute("aria-busy", "false")
    expect(logout.querySelector('[aria-label="loading"]')).toBeNull()
    expect((await tldwClient.getConfig())?.authSource).toBe("cookie-session")
    expect(await storage.get(COOKIE_SESSION_CONFIG_KEY)).toBeDefined()
  } else {
    expect(screen.queryByRole("button", { name: /Logout/ })).not.toBeInTheDocument()
    expect(await storage.get(COOKIE_SESSION_CONFIG_KEY)).toBeUndefined()
  }
  expect(errors.mock.calls.length).toBe(fail ? 1 : 0)
})
