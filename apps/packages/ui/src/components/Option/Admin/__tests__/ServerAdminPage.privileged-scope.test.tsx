// @vitest-environment jsdom
import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { App, ConfigProvider } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import ServerAdminPage from "../ServerAdminPage"
import { adminMethods } from "@/services/tldw/domains/admin"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

const mocks = vi.hoisted(() => ({ canonical: vi.fn(), getConfig: vi.fn(), users: vi.fn(), get: vi.fn(), fetch: vi.fn() }))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({ useCanonicalConnectionConfig: mocks.canonical }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: mocks.get, set: vi.fn(), remove: vi.fn(), watch: vi.fn(), unwatch: vi.fn() }),
  safeStorageSerde: { serialize: (value: unknown) => value, deserialize: (value: unknown) => value }
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  getConfig: mocks.getConfig, listAdminUsers: mocks.users,
  getSystemStats: async () => ({ users: {}, storage: {}, sessions: {} }), listAdminRoles: async () => [],
  getMediaIngestionBudgetDiagnostics: async () => ({ status: "ok", entity: "user:22", limits: {}, usage: {} }),
  updateAdminUser: (...args: Parameters<typeof adminMethods.updateAdminUser>) => adminMethods.updateAdminUser(...args)
} }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))
vi.mock("@/components/Common/PageShell", () => ({ PageShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))
vi.mock("../AdminAudioInstallerCard", () => ({ AdminAudioInstallerCard: () => null }))
const config = (user = 1, serverUrl = "https://admin-a.example.test", revision = 1): TldwConfig => ({
  serverUrl, authMode: "multi-user", accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user), rev: revision }))}.signature`
})
let current: TldwConfig
let loading: boolean
let target: { id: number; username: string }
const tree = () => <ConfigProvider theme={{ token: { motion: false } }}><App><ServerAdminPage /></App></ConfigProvider>
const open = async (username: string) => {
  const row = (await screen.findByText(username)).closest("tr")!
  fireEvent.click(within(row).getByRole("switch"))
  const dialog = await screen.findByRole("dialog")
  await waitFor(() => expect(dialog).toBeVisible())
  return dialog
}
const fill = (dialog: HTMLElement, password: string) => {
  fireEvent.change(within(dialog).getByLabelText("Audit reason"), { target: { value: "Explicit owned account change" } })
  fireEvent.change(within(dialog).getByLabelText("Your current admin password"), { target: { value: password } })
}
const submit = (dialog: HTMLElement) => fireEvent.click(within(dialog).getByRole("button", { name: "Confirm change" }))
beforeEach(() => {
  vi.resetAllMocks()
  current = config(); loading = false; target = { id: 22, username: "bob" }
  mocks.canonical.mockImplementation(() => ({ config: current, authorityLoading: loading }))
  mocks.getConfig.mockImplementation(async () => current)
  mocks.get.mockImplementation(async (key: string) => key === "tldwConfig" ? current : null)
  mocks.users.mockImplementation(async () => ({ users: [{ ...target, email: `${target.username}@example.test`, role: "user", is_active: true }], total: 1, page: 1, limit: 20, pages: 1 }))
  mocks.fetch.mockImplementation(async () => new Response('{"message":"User updated"}', { status: 200, headers: { "Content-Type": "application/json" } }))
  vi.stubGlobal("fetch", mocks.fetch)
})
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

describe("Admin privileged credentials lifetime", () => {
  it.each(["server", "account"])("clears a held password after %s change and only dispatches a newly confirmed successor", async (change) => {
    const view = render(tree())
    const oldDialog = await open("bob")
    fill(oldDialog, "Synthetic-old-admin-742!")
    current = change === "server" ? config(1, "https://admin-b.example.test") : config(2)
    target = { id: 33, username: "carol" }
    view.rerender(tree())
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(mocks.fetch).not.toHaveBeenCalled()
    expect(screen.queryByDisplayValue("Synthetic-old-admin-742!")).not.toBeInTheDocument()
    const fresh = await open("carol")
    expect(within(fresh).getByLabelText("Your current admin password")).toHaveValue("")
    fill(fresh, "Synthetic-successor-admin-742!")
    submit(fresh)
    await waitFor(() => expect(mocks.fetch).toHaveBeenCalledOnce())
    const [url, init] = mocks.fetch.mock.calls[0]
    expect(String(url)).toBe(`${current.serverUrl}/api/v1/admin/users/33`)
    expect(new Headers(init.headers).get("Authorization")).toBe(`Bearer ${current.accessToken}`)
    expect(JSON.parse(init.body)).toEqual({ is_active: false, reason: "Explicit owned account change", admin_password: "Synthetic-successor-admin-742!" })
  })

  it("aborts an old request on unresolved authority and ignores a transport that settles late", async () => {
    let finish!: (response: Response) => void
    mocks.fetch.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve }))
    const view = render(tree())
    const dialog = await open("bob")
    fill(dialog, "Synthetic-old-admin-742!")
    submit(dialog)
    await waitFor(() => expect(mocks.fetch).toHaveBeenCalledOnce())
    const signal = mocks.fetch.mock.calls[0][1].signal as AbortSignal
    loading = true
    view.rerender(tree())
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(signal.aborted).toBe(true)
    current = config(2); target = { id: 33, username: "carol" }; loading = false
    view.rerender(tree())
    await screen.findByText("carol")
    await act(async () => finish(new Response('{"message":"User updated"}', { status: 200 })))
    expect(screen.queryByText("User updated")).not.toBeInTheDocument()
    expect(screen.queryByText("bob")).not.toBeInTheDocument()
    expect(mocks.users).toHaveBeenCalledTimes(2)
    expect(mocks.fetch).toHaveBeenCalledOnce()
  })

  it("preserves a held confirmation across a same-owner credential refresh", async () => {
    const view = render(tree())
    const dialog = await open("bob")
    fill(dialog, "Synthetic-same-admin-742!")
    current = config(1, current.serverUrl, 2)
    view.rerender(tree())
    expect(within(dialog).getByLabelText("Your current admin password")).toHaveValue("Synthetic-same-admin-742!")
    submit(dialog)
    await waitFor(() => expect(mocks.fetch).toHaveBeenCalledOnce())
    const [url, init] = mocks.fetch.mock.calls[0]
    expect(String(url)).toBe(`${current.serverUrl}/api/v1/admin/users/22`)
    expect(new Headers(init.headers).get("Authorization")).toBe(`Bearer ${current.accessToken}`)
    expect(JSON.parse(init.body).admin_password).toBe("Synthetic-same-admin-742!")
  })
})
