import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { App, ConfigProvider } from "antd"
import { Storage } from "@plasmohq/storage"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

// Match the WebUI storage boundary, including same-tab and cross-tab watches.
vi.mock("@plasmohq/storage", async () =>
  import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage")
)

const mocks = vi.hoisted(() => ({ getConfig: vi.fn(), logout: vi.fn(), bgRequest: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.bgRequest }))
vi.mock("@/services/tldw/TldwApiClient", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/tldw/TldwApiClient")>(),
  tldwClient: {
  getConfig: mocks.getConfig,
  initialize: vi.fn(), updateConfig: vi.fn(), ragHealth: vi.fn()
} }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { logout: mocks.logout } }))
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

import { TldwSettings } from "../tldw"

const target: TldwConfig = { serverUrl: "https://settings.example.test", authMode: "multi-user", authSource: "manual" }
const signedIn: TldwConfig = { ...target, accessToken: "alice-access", refreshToken: "alice-refresh" }
let storage: Storage
beforeEach(async () => {
  localStorage.clear()
  sessionStorage.clear()
  storage = new Storage({ area: "local" })
  await storage.set("tldwConfig", signedIn)
  mocks.getConfig.mockReset().mockResolvedValue(signedIn)
  mocks.bgRequest.mockReset().mockResolvedValue({ paths: {} })
  mocks.logout.mockReset().mockImplementation(async () => {
    await storage.set("tldwConfig", target)
    mocks.getConfig.mockResolvedValue(target)
    window.dispatchEvent(new CustomEvent("tldw:config-updated"))
  })
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ paths: {} }), { status: 200 })))
})
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })
const mount = () => render(<ConfigProvider theme={{ token: { motion: false } }}><App><TldwSettings /></App></ConfigProvider>)
const formErrors = (errors: ReturnType<typeof vi.spyOn>) => errors.mock.calls.flat().map(String).filter(line => line.includes("not connected to any Form"))

it("discovers optional billing through the shared transport at the server root", async () => {
  mount()
  await screen.findByText("Logged In", { exact: true })
  await waitFor(() => expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({
    path: `${target.serverUrl}/openapi.json`, method: "GET", noAuth: true, timeoutMs: 5000,
    abortSignal: expect.any(AbortSignal)
  })))
  expect(fetch).not.toHaveBeenCalled()
  expect(screen.queryByRole("tab", { name: "Billing" })).not.toBeInTheDocument()
})

it("ignores a delayed billing advertisement after disconnecting the server", async () => {
  let release!: (value: unknown) => void
  mocks.bgRequest.mockReturnValue(new Promise(resolve => { release = resolve }))
  mount()
  await screen.findByText("Logged In", { exact: true })
  await waitFor(() => expect(mocks.bgRequest).toHaveBeenCalled())
  fireEvent.click(await screen.findByRole("button", { name: "Logout", exact: true }))
  await screen.findByText("Login Required", { exact: true })
  await act(async () => release({ paths: Object.fromEntries(
    ["plans", "subscription", "usage", "invoices"].map(route => [`/api/v1/billing/${route}`, { get: {} }])
  ) }))
  expect(screen.queryByRole("tab", { name: "Billing" })).not.toBeInTheDocument()
})

it("loads the actual form before applying async configuration values", async () => {
  const errors = vi.spyOn(console, "error").mockImplementation(() => {})
  let resolve!: (config: TldwConfig) => void
  mocks.getConfig.mockReturnValueOnce(new Promise<TldwConfig>(done => { resolve = done }))
  mount()
  await act(async () => {
    resolve(signedIn)
    await Promise.resolve()
    // A pending concurrent commit must not require a timer to beat the Form mount.
    await new Promise(done => setTimeout(done, 0))
  })
  await screen.findByText("Logged In", { exact: true })
  expect(screen.getByRole("textbox", { name: "Server URL" })).toHaveValue(target.serverUrl)
  expect(formErrors(errors)).toEqual([])
})

it("normal logout reaches the actual login form without a disconnected-form warning", async () => {
  const errors = vi.spyOn(console, "error").mockImplementation(() => {})
  mount()
  await screen.findByText("Logged In", { exact: true })
  errors.mockClear()
  fireEvent.click(screen.getByRole("button", { name: "Logout", exact: true }))
  await screen.findByText("Login Required", { exact: true })
  await waitFor(() => expect(screen.getByRole("textbox", { name: "Server URL" })).toHaveValue(target.serverUrl))
  expect(formErrors(errors)).toEqual([])
})

it.each([
  { name: "cookie session", config: { serverUrl: target.serverUrl, authMode: "single-user", authSource: "cookie-session" } as TldwConfig, action: "Logout" },
  { name: "manual API key", config: { serverUrl: target.serverUrl, authMode: "single-user", authSource: "manual", apiKey: "synthetic-manual-key" } as TldwConfig, action: "Disconnect" }
])("clears the actual $name form after a pending disconnect", async ({ config, action }) => {
  const errors = vi.spyOn(console, "error").mockImplementation(() => {})
  await storage.set("tldwConfig", config)
  mocks.getConfig.mockResolvedValue(config)
  let release!: () => void
  const pending = new Promise<void>(resolve => { release = resolve })
  mocks.logout.mockImplementation(async () => {
    await pending
    await storage.remove("tldwConfig")
    mocks.getConfig.mockResolvedValue(null)
    window.dispatchEvent(new CustomEvent("tldw:config-updated"))
  })
  mount()
  fireEvent.click(await screen.findByRole("button", { name: action, exact: true }))
  expect(screen.getByRole("textbox", { name: "Server URL" })).toHaveValue(config.serverUrl)
  await act(async () => { release() })
  await waitFor(() => expect(screen.getByRole("textbox", { name: "Server URL" })).toHaveValue(""))
  expect(screen.getByLabelText("API Key")).toHaveValue("")
  expect(screen.queryByRole("button", { name: action, exact: true })).not.toBeInTheDocument()
  expect(formErrors(errors)).toEqual([])
})

it("preserves the mounted server draft across storage account transitions", async () => {
  mount()
  await screen.findByText("Logged In", { exact: true })
  const input = screen.getByRole("textbox", { name: "Server URL" })
  fireEvent.change(input, { target: { value: "https://unsaved.example.test" } })
  await screen.findByText("Login Required", { exact: true })
  await act(async () => {
    await storage.set("tldwConfig", { ...signedIn, accessToken: "bob-access", refreshToken: "bob-refresh" })
    await storage.set("tldwConfig", signedIn)
    window.dispatchEvent(new CustomEvent("tldw:config-updated"))
  })
  expect(input).toHaveValue("https://unsaved.example.test")
  expect(screen.getByText("Login Required", { exact: true })).toBeInTheDocument()
})

it("rejects an old account's delayed initial load after the Settings owner is replaced", async () => {
  const errors = vi.spyOn(console, "error").mockImplementation(() => {})
  let resolveOld!: (config: TldwConfig) => void
  mocks.getConfig.mockReturnValueOnce(new Promise<TldwConfig>(resolve => { resolveOld = resolve }))
  const oldOwner = mount()
  oldOwner.unmount()
  const next = { ...signedIn, serverUrl: "https://bob.example.test", accessToken: "bob-access", refreshToken: "bob-refresh" }
  await storage.set("tldwConfig", next)
  mocks.getConfig.mockResolvedValue(next)
  mount()
  await screen.findByText("Logged In", { exact: true })
  const input = screen.getByRole("textbox", { name: "Server URL" })
  fireEvent.change(input, { target: { value: "https://bob-draft.example.test" } })
  await act(async () => {
    resolveOld(signedIn)
    await Promise.resolve()
    await new Promise(resolve => setTimeout(resolve, 0))
  })
  expect(input).toHaveValue("https://bob-draft.example.test")
  expect(formErrors(errors)).toEqual([])
})

it("keeps the current form available when logout fails before credentials are cleared", async () => {
  const errors = vi.spyOn(console, "error").mockImplementation(() => {})
  mocks.logout.mockRejectedValue(new Error("Local credential storage unavailable"))
  mount()
  await screen.findByText("Logged In", { exact: true })
  fireEvent.click(screen.getByRole("button", { name: "Logout", exact: true }))
  await screen.findByText("Logout failed", { exact: true })
  expect(screen.getByText("Logged In", { exact: true })).toBeInTheDocument()
  expect(screen.getByRole("textbox", { name: "Server URL" })).toHaveValue(target.serverUrl)
  expect(formErrors(errors)).toEqual([])
})
