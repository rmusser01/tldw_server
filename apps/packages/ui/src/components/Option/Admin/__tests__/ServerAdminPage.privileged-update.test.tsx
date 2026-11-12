import { QueryClientProvider, type QueryClient } from "@tanstack/react-query"
import { createAdminQueryClient } from "../AdminQueryProvider"
// @vitest-environment jsdom
import React from "react"
import { act, cleanup, fireEvent, render as renderWithQueries, screen, waitFor, within } from "@testing-library/react"
import { App, ConfigProvider } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import ServerAdminPage from "../ServerAdminPage"
import { adminMethods } from "@/services/tldw/domains/admin"


const adminQueryClients: QueryClient[] = []
const render = (ui: React.ReactElement) => {
  const client = createAdminQueryClient()
  adminQueryClients.push(client)
  return renderWithQueries(ui, {
    wrapper: ({ children }: React.PropsWithChildren) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  })
}
afterEach(() => adminQueryClients.splice(0).forEach(client => client.clear()))

const mocks = vi.hoisted(() => ({
  canonical: vi.fn(),
  getConfig: vi.fn(), getSystemStats: vi.fn(), listAdminUsers: vi.fn(),
  listAdminRoles: vi.fn(), getMediaIngestionBudgetDiagnostics: vi.fn(), request: vi.fn()
}))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({ useCanonicalConnectionConfig: mocks.canonical }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.request }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  ...mocks,
  updateAdminUser: (...args: Parameters<typeof adminMethods.updateAdminUser>) => adminMethods.updateAdminUser(...args)
} }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))
vi.mock("@/components/Common/PageShell", () => ({ PageShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))
vi.mock("../AdminAudioInstallerCard", () => ({ AdminAudioInstallerCard: () => null }))

const bob = { id: 22, username: "bob", email: "bob@example.com", role: "user", is_active: true, is_verified: true }
const userPage = (user = bob) => ({ users: [user], total: 1, page: 1, pages: 1, limit: 20 })
const password = "Synthetic-acting-admin-742!"
const reason = "Reviewed the account access request"
const row = async () => (await screen.findByText("bob")).closest("tr")!
const openActive = async () => {
  fireEvent.click(within(await row()).getByRole("switch"))
  const dialog = await screen.findByRole("dialog")
  await waitFor(() => expect(dialog).toBeVisible())
  return dialog
}
const fill = (dialog: HTMLElement, text = reason, secret = password) => {
  fireEvent.change(within(dialog).getByLabelText("Audit reason"), { target: { value: text } })
  fireEvent.change(within(dialog).getByLabelText("Your current admin password"), { target: { value: secret } })
}
const submit = (dialog: HTMLElement) => fireEvent.click(within(dialog).getByRole("button", { name: "Confirm change" }))

beforeEach(() => {
  vi.resetAllMocks()
  mocks.canonical.mockReturnValue({ config: { serverUrl: "http://localhost:8000", authMode: "multi-user" }, authorityLoading: false })
  mocks.getConfig.mockResolvedValue({ serverUrl: "http://localhost:8000", authMode: "multi-user" })
  mocks.getSystemStats.mockResolvedValue({ users: {}, storage: {}, sessions: {} })
  mocks.listAdminUsers.mockResolvedValue(userPage())
  mocks.listAdminRoles.mockResolvedValue([{ id: 1, name: "user" }, { id: 2, name: "admin" }, { id: 3, name: "service" }])
  mocks.getMediaIngestionBudgetDiagnostics.mockResolvedValue({ status: "ok", entity: "user:22", limits: {}, usage: {} })
  mocks.request.mockResolvedValue({ message: "User updated" })
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe("Admin privileged user updates", () => {
  it("waits for explicit confirmation and sends the entered audit and acting-admin credentials", async () => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><App><ServerAdminPage /></App></ConfigProvider>)
    fireEvent.click(within(await row()).getByRole("switch"))
    expect(mocks.request).not.toHaveBeenCalled()
    const dialog = await screen.findByRole("dialog")
    expect(dialog).toHaveTextContent("bob")
    expect(dialog).toHaveTextContent("Deactivate")
    expect(within(await row()).getByRole("switch")).toBeChecked()
    fill(dialog, `  ${reason}  `)
    mocks.listAdminUsers.mockResolvedValue(userPage({ ...bob, is_active: false }))
    submit(dialog)
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith({
      path: "/api/v1/admin/users/22", method: "PUT", headers: { "Content-Type": "application/json" },
      body: { is_active: false, reason, admin_password: password },
      configSnapshot: expect.objectContaining({ serverUrl: "http://localhost:8000", authMode: "multi-user" }),
      abortSignal: expect.any(AbortSignal)
    }))
    await waitFor(() => expect(within(screen.getByText("bob").closest("tr")!).getByRole("switch")).not.toBeChecked())
    expect(mocks.listAdminUsers).toHaveBeenCalledTimes(2)
    expect(await screen.findByText("User updated")).toBeVisible()
  })

  it("Cancel leaves the user unchanged and clears sensitive fields before reopening", async () => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><App><ServerAdminPage /></App></ConfigProvider>)
    const dialog = await openActive()
    fill(dialog)
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(mocks.request).not.toHaveBeenCalled()
    expect(within(await row()).getByRole("switch")).toBeChecked()
    const reopened = await openActive()
    expect(within(reopened).getByLabelText("Audit reason")).toHaveValue("")
    expect(within(reopened).getByLabelText("Your current admin password")).toHaveValue("")
  })

  it.each([
    ["       ", password], ["1234567", password], [reason, ""], [reason, "short"], [reason, "        "]
  ])("does not dispatch invalid reason/password input %#", async (text, secret) => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><App><ServerAdminPage /></App></ConfigProvider>)
    const dialog = await openActive()
    fill(dialog, text, secret)
    submit(dialog)
    await waitFor(() => expect(dialog.querySelector(".ant-form-item-explain-error")).not.toBeNull())
    expect(mocks.request).not.toHaveBeenCalled()
    expect(within(await row()).getByRole("switch")).toBeChecked()
  })

  it("keeps a failed update visible and retries only after explicit confirmation", async () => {
    mocks.request.mockRejectedValueOnce(Object.assign(new Error("Admin reauthentication failed"), { status: 403 }))
    render(<ConfigProvider theme={{ token: { motion: false } }}><App><ServerAdminPage /></App></ConfigProvider>)
    const dialog = await openActive()
    fill(dialog)
    submit(dialog)
    expect(await within(dialog).findByText("Admin reauthentication failed")).toBeVisible()
    expect(within(dialog).getByLabelText("Audit reason")).toHaveValue(reason)
    expect(within(await row()).getByRole("switch")).toBeChecked()
    expect(mocks.listAdminUsers).toHaveBeenCalledTimes(1)
    expect(mocks.request).toHaveBeenCalledTimes(1)
    submit(dialog)
    await waitFor(() => expect(mocks.request).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
  })

  it("does not submit twice or dismiss while the update is pending", async () => {
    let finish!: (value: unknown) => void
    mocks.request.mockImplementation(() => new Promise((resolve) => { finish = resolve }))
    render(<ConfigProvider theme={{ token: { motion: false } }}><App><ServerAdminPage /></App></ConfigProvider>)
    const dialog = await openActive()
    fill(dialog)
    submit(dialog); submit(dialog)
    await waitFor(() => expect(mocks.request).toHaveBeenCalledTimes(1))
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
    expect(dialog).toBeVisible()
    expect(within(await row()).getByRole("switch")).toBeChecked()
    await act(async () => finish({ message: "User updated" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
  })

  it("confirms a canonical role change without offering unsupported custom or service roles", async () => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><App><ServerAdminPage /></App></ConfigProvider>)
    fireEvent.mouseDown(within(await row()).getByRole("combobox"))
    expect(await screen.findByRole("option", { name: "admin" })).toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "service" })).not.toBeInTheDocument()
    fireEvent.click(screen.getByText("admin", { selector: ".ant-select-item-option-content" }))
    expect(mocks.request).not.toHaveBeenCalled()
    const dialog = await screen.findByRole("dialog")
    expect(dialog).toHaveTextContent("admin")
    fill(dialog)
    submit(dialog)
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({
      path: "/api/v1/admin/users/22", body: { role: "admin", reason, admin_password: password }
    })))
  })

  it("supports single-user audit confirmation without inventing a password", async () => {
    mocks.getConfig.mockResolvedValue({ serverUrl: "http://localhost:8000", authMode: "single-user" })
    mocks.canonical.mockReturnValue({ config: { serverUrl: "http://localhost:8000", authMode: "single-user" }, authorityLoading: false })
    render(<ConfigProvider theme={{ token: { motion: false } }}><App><ServerAdminPage /></App></ConfigProvider>)
    const dialog = await openActive()
    fill(dialog, reason, "")
    submit(dialog)
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({
      body: { is_active: false, reason }
    })))
  })
})
