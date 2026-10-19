import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { App, ConfigProvider } from "antd"
import { Storage } from "@plasmohq/storage"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import { tldwClient } from '@/services/tldw/TldwApiClient'
import { createMemoryRouter, RouterProvider } from "react-router-dom"
import { requestSettingsNavigation } from '@/utils/settings-return'
import { apiSend } from "@/services/api-send"

// Match the WebUI storage boundary, including same-tab and cross-tab watches.
vi.mock("@plasmohq/storage", async () =>
  import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage")
)

const mocks = vi.hoisted(() => ({ getConfig: vi.fn(), logout: vi.fn(), bgRequest: vi.fn(), testApiKey: vi.fn(), saveManualSingleUserCredential: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.bgRequest }))
vi.mock("@/services/tldw/TldwApiClient", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/tldw/TldwApiClient")>(),
  tldwClient: {
  getConfig: mocks.getConfig,
  initialize: vi.fn(), updateConfig: vi.fn(), ragHealth: vi.fn(), saveManualSingleUserCredential: mocks.saveManualSingleUserCredential
} }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { logout: mocks.logout, testApiKey: mocks.testApiKey } }))
vi.mock("@/services/tldw-server", () => ({ DEFAULT_TLDW_API_KEY: "default-key" }))
vi.mock("@/services/api-send", () => ({ apiSend: vi.fn() }))
vi.mock("@/store/connection", () => ({ useConnectionStore: { getState: () => ({ checkOnce: vi.fn() }) } }))
vi.mock("@/components/Common/ServerOverviewHint", () => ({ ServerOverviewHint: () => null }))
vi.mock("../server-health-probe", () => ({ probeServerHealth: vi.fn() }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({
  t: (key: string, fallback: unknown) => typeof fallback === "string" ? fallback : key
}) }))


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
  vi.mocked(apiSend).mockReset()
  mocks.bgRequest.mockReset().mockResolvedValue({ paths: {} })
  mocks.testApiKey.mockReset().mockResolvedValue(true)
  mocks.saveManualSingleUserCredential.mockReset().mockResolvedValue("device")
  mocks.logout.mockReset().mockImplementation(async () => {
    await storage.set("tldwConfig", target)
    mocks.getConfig.mockResolvedValue(target)
    window.dispatchEvent(new CustomEvent("tldw:config-updated"))
  })
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ paths: {} }), { status: 200 })))
})
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })
let router: ReturnType<typeof createMemoryRouter>
const mount = (props: React.ComponentProps<typeof TldwSettings> = {}) => {
  router = createMemoryRouter([
    { path: "/", element: <TldwSettings {...props} /> },
    { path: "/settings/*", element: <div>Settings destination</div> }
  ], { initialEntries: ["/settings/prior", "/"], initialIndex: 1 })
  return render(<ConfigProvider theme={{ token: { motion: false } }}><App><RouterProvider router={router} /></App></ConfigProvider>)
}
const navigateSettings = async (destination: string) => {
  if (requestSettingsNavigation(destination)) await act(async () => { await router.navigate(destination) })
}
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
  fireEvent.click(await screen.findByRole("button", { name: "Logout" }))
  await screen.findByText("Login Required", { exact: true })
  await act(async () => release({ paths: Object.fromEntries(
    ["plans", "subscription", "usage", "invoices"].map(route => [`/api/v1/billing/${route}`, { get: {} }])
  ) }))
  expect(screen.queryByRole("tab", { name: "Billing" })).not.toBeInTheDocument()
})

it("prefills a fresh URL hint while keeping credentials absent", async () => {
  await storage.remove("tldwConfig")
  await storage.set("tldwServerUrl", "https://initial-api.example.test")
  mocks.getConfig.mockResolvedValue(null)
  mount()
  const field = await screen.findByRole("textbox", { name: "Server URL" })
  await waitFor(() => expect(field).toHaveValue("https://initial-api.example.test"))
  expect(localStorage.getItem("tldwConfig")).toBeNull()
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
  fireEvent.click(screen.getByRole("button", { name: "Logout" }))
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
  fireEvent.click(await screen.findByRole("button", { name: action }))
  expect(screen.getByRole("textbox", { name: "Server URL" })).toHaveValue(config.serverUrl)
  await act(async () => { release() })
  await waitFor(() => expect(screen.getByRole("textbox", { name: "Server URL" })).toHaveValue(""))
  expect(screen.getByLabelText("API Key")).toHaveValue("")
  expect(screen.queryByRole("button", { name: action })).not.toBeInTheDocument()
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

it('warns and keeps an unsaved Server URL when leaving Settings is cancelled', async () => {
  const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
  mount()
  await screen.findByText('Logged In', { exact: true })
  const input = screen.getByRole('textbox', { name: 'Server URL' })
  fireEvent.change(input, { target: { value: 'https://unsaved.example.test' } })
  expect(screen.getByText(/unsaved connection settings/i)).toBeInTheDocument()
  await navigateSettings('/settings/preferences')
  await waitFor(() => expect(confirm).toHaveBeenCalled())
  expect(input).toHaveValue('https://unsaved.example.test')
  expect(await storage.get('tldwConfig')).toEqual(signedIn)
})

it('warns on unload only while the connection settings contain unsaved changes', async () => {
  mount()
  await screen.findByText('Logged In', { exact: true })
  const input = screen.getByRole('textbox', { name: 'Server URL' })
  fireEvent.change(input, { target: { value: 'https://unsaved.example.test' } })
  const pending = new Event('beforeunload', { cancelable: true })
  window.dispatchEvent(pending)
  expect(pending.defaultPrevented).toBe(true)
  fireEvent.change(input, { target: { value: target.serverUrl } })
  const clean = new Event('beforeunload', { cancelable: true })
  window.dispatchEvent(clean)
  expect(clean.defaultPrevented).toBe(false)
  expect(requestSettingsNavigation('/settings/preferences')).toBe(true)
})

it('keeps an unsaved connection draft when an ordinary settings link is cancelled', async () => {
  vi.spyOn(window, 'confirm').mockReturnValue(false)
  mount()
  await screen.findByText('Logged In', { exact: true })
  fireEvent.change(screen.getByRole('textbox', { name: 'Server URL' }), { target: { value: 'https://unsaved.example.test' } })
  fireEvent.click(screen.getByRole('link', { name: 'Health' }))
  await waitFor(() => expect(window.confirm).toHaveBeenCalled())
  expect(screen.getByRole('textbox', { name: 'Server URL' })).toHaveValue('https://unsaved.example.test')
})

it('clears the unsaved navigation guard after connection settings save succeeds', async () => {
  mount()
  await screen.findByText('Logged In', { exact: true })
  fireEvent.change(screen.getByRole('textbox', { name: 'Server URL' }), { target: { value: 'https://saved.example.test' } })
  await screen.findByText('Login Required', { exact: true })
  fireEvent.click(screen.getByRole('button', { name: 'common:save' }))
  await waitFor(() => expect(screen.queryByText(/unsaved connection settings/i)).not.toBeInTheDocument())
  expect(requestSettingsNavigation('/settings/preferences')).toBe(true)
})

it('retains the draft and its navigation guard after connection save fails', async () => {
  vi.spyOn(console, 'error').mockImplementation(() => {})
  vi.spyOn(window, 'confirm').mockReturnValue(false)
  vi.mocked(tldwClient.updateConfig).mockRejectedValueOnce(new Error('Storage unavailable'))
  mount()
  await screen.findByText('Logged In', { exact: true })
  fireEvent.change(screen.getByRole('textbox', { name: 'Server URL' }), { target: { value: 'https://unsaved.example.test' } })
  fireEvent.click(screen.getByRole('button', { name: 'common:save' }))
  await screen.findByText('settings:saveFailed')
  expect(screen.getByText(/unsaved connection settings/i)).toBeInTheDocument()
  await navigateSettings('/settings/preferences')
  await waitFor(() => expect(window.confirm).toHaveBeenCalled())
  expect(screen.getByRole('textbox', { name: 'Server URL' })).toHaveValue('https://unsaved.example.test')
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
  fireEvent.click(screen.getByRole("button", { name: "Logout" }))
  await screen.findByText("Logout failed", { exact: true })
  expect(screen.getByText("Logged In", { exact: true })).toBeInTheDocument()
  expect(screen.getByRole("textbox", { name: "Server URL" })).toHaveValue(target.serverUrl)
  expect(formErrors(errors)).toEqual([])
})


it.each([-1, '/settings/health'])('preserves an unsaved connection draft when navigation %s is cancelled', async (destination) => {
  vi.spyOn(window, 'confirm').mockReturnValue(false)
  mount()
  await screen.findByText('Logged In', { exact: true })
  fireEvent.change(screen.getByRole('textbox', { name: 'Server URL' }), { target: { value: 'https://unsaved.example.test' } })
  await act(async () => { if (typeof destination === "number") await router.navigate(destination)
    else await router.navigate(destination) })
  await waitFor(() => expect(window.confirm).toHaveBeenCalled())
  expect(router.state.location.pathname).toBe('/')
  expect(screen.getByRole('textbox', { name: 'Server URL' })).toHaveValue('https://unsaved.example.test')
})


it.each([
  ["", "Please enter the server URL"],
  ["not a URL", "Please enter a valid URL"]
])("does not leave connection progress active for invalid URL %s", async (url, validation) => {
  mount()
  await screen.findByText("Logged In", { exact: true })
  vi.mocked(apiSend).mockResolvedValue({ ok: true, status: 200 })
  fireEvent.click(screen.getByRole("button", { name: "Test Connection", exact: true }))
  await screen.findByText("Core: reachable", { exact: true })
  vi.mocked(apiSend).mockClear()
  vi.mocked(tldwClient.updateConfig).mockClear()
  fireEvent.change(screen.getByLabelText("Server URL", { exact: true }), { target: { value: url } })
  fireEvent.click(screen.getByRole("button", { name: "Test Connection", exact: true }))
  await screen.findByText(validation, { exact: true })
  await waitFor(() => expect(screen.queryByText("Checking connection and RAG health…")).not.toBeInTheDocument())
  expect(screen.queryByText("Core: checking…")).not.toBeInTheDocument()
  expect(screen.getByText("Core: not checked yet")).toBeInTheDocument()
  expect(apiSend).not.toHaveBeenCalled()
  expect(tldwClient.updateConfig).not.toHaveBeenCalled()
  })

it("keeps both diagnostics actions within an embedded settings destination", async () => {
  const diagnostics = vi.fn()
  mount({ onOpenDiagnostics: diagnostics })
  await screen.findByText("Logged In", { exact: true })
  fireEvent.click(screen.getByRole("button", { name: "Recheck", exact: true }))
  fireEvent.click(await screen.findByRole("button", { name: "Health & diagnostics", exact: true }))
  expect(diagnostics).toHaveBeenCalledOnce()
  fireEvent.click(screen.getByRole("button", { name: "Health", exact: true }))
  expect(diagnostics).toHaveBeenCalledTimes(2)
  expect(router.state.location.pathname).toBe("/")
  expect(screen.getByRole("textbox", { name: "Server URL" })).toHaveValue(target.serverUrl)
})

it.each(["unmount", "foreign account ABA", "edited key"])("does not commit a visible Save after %s during held validation", async (boundary) => {
  const config: TldwConfig = { serverUrl: target.serverUrl, authMode: "single-user", apiKey: "synthetic-old-key", authSource: "manual" }
  await storage.set("tldwConfig", config)
  mocks.getConfig.mockResolvedValue(config)
  let release!: (valid: boolean) => void
  mocks.testApiKey.mockReturnValueOnce(new Promise<boolean>(resolve => { release = resolve }))
  const saving = vi.fn()
  const view = mount({ onSavingChange: saving })
  const key = await screen.findByLabelText("API Key")
  fireEvent.change(key, { target: { value: "synthetic-new-key" } })
  fireEvent.click(screen.getByRole("button", { name: "common:save", exact: true }))
  await waitFor(() => expect(mocks.testApiKey).toHaveBeenCalled())
  if (boundary === "unmount") view.unmount()
  else if (boundary === "edited key") fireEvent.change(key, { target: { value: "newer-edited-key" } })
  else await act(async () => {
    const foreign = { ...config, apiKey: "foreign-key" }
    await storage.set("tldwConfig", foreign)
    window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: JSON.stringify(config), newValue: JSON.stringify(foreign) }))
    await storage.set("tldwConfig", config)
    window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: JSON.stringify(foreign), newValue: JSON.stringify(config) }))
  })
  await act(async () => { release(true) })
  expect(mocks.saveManualSingleUserCredential).not.toHaveBeenCalled()
  await waitFor(() => expect(saving).toHaveBeenLastCalledWith(false))
})
