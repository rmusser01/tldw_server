import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { ConfigProvider, message } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import DataOpsPage from "../DataOpsPage"

const mocks = vi.hoisted(() => ({ request: vi.fn() }))
// Real component/domain methods; only external background transport is replaced.
vi.mock("@/services/background-proxy", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/background-proxy")>(), bgRequest: mocks.request
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback ?? key })
}))

const BACKUP = { id: "chacha backup#1.db", dataset: "chacha", user_id: 1, status: "ready", size_bytes: 32768, created_at: "2026-10-09T04:30:00Z" }
const PATH = "/api/v1/admin/backups/chacha%20backup%231.db/restore"
type Request = { path: string, method: string, headers?: Record<string, string>, body?: Record<string, unknown> }
let backup: Record<string, unknown>
const restoreRequest = vi.fn<() => Promise<unknown>>()
const posted = (): Request[] => mocks.request.mock.calls.map(([request]) => request as Request).filter((request) => request.method === "POST")
const mount = async () => {
  render(<ConfigProvider theme={{ token: { motion: false } }}><DataOpsPage /></ConfigProvider>)
  await waitFor(() => expect(mocks.request).toHaveBeenCalledWith({ path: "/api/v1/admin/backups", method: "GET" }))
  await waitFor(() => expect(screen.getByText(String(backup.dataset ?? "unknown"), { selector: ".ant-tag" })).toBeVisible())
  return screen.getByText(String(backup.dataset ?? "unknown"), { selector: ".ant-tag" }).closest("tr")!
}
const openConfirmation = async (row: HTMLElement) => {
  fireEvent.click(within(row).getByRole("button", { name: /Restore$/ }))
  return await waitFor(() => {
    const button = screen.getAllByRole("button", { name: /Restore$/ }).find((candidate) => candidate.closest(".ant-popover"))
    expect(button).toBeVisible()
    const popover = button!.closest(".ant-popover")
    if (!(popover instanceof HTMLElement)) throw new Error("Missing restore confirmation")
    return popover
  }, { timeout: 3000 })
}
const confirm = async (row: HTMLElement) => {
  const popover = await openConfirmation(row)
  fireEvent.click(within(popover).getByRole("button", { name: /Restore$/ }))
  await waitFor(() => expect(posted()).toHaveLength(1))
  return posted()[0]
}

beforeEach(() => {
  vi.clearAllMocks()
  restoreRequest.mockReset().mockResolvedValue({ status: "restored", message: "Database restored" })
  backup = { ...BACKUP }
  vi.spyOn(message, "success").mockImplementation(() => undefined as never)
  vi.spyOn(message, "error").mockImplementation(() => undefined as never)
  mocks.request.mockImplementation(async ({ path, method }: Request) => {
    if (path === "/api/v1/admin/backups" && method === "GET") return { items: [backup], total: 1 }
    if (path === "/api/v1/admin/backup-schedules" && method === "GET") return { items: [] }
    if (path.endsWith("/restore") && method === "POST") return restoreRequest()
    throw new Error(`Unexpected external request: ${method} ${path}`)
  })
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe("Backup restore contract", () => {
  it("displays the exact canonical artifact identity and byte size", async () => {
    const row = await mount()
    expect(within(row).getByText(BACKUP.id, { selector: "code" })).toBeVisible()
    expect(within(row).getByText("32768")).toBeVisible()
    expect(posted()).toHaveLength(0)
  }, 30000)

  it("shows artifact, dataset and user overwrite context and cancellation never posts", async () => {
    const row = await mount()
    const popover = await openConfirmation(row)
    expect(within(popover).getByText(/This will overwrite current data/)).toBeVisible()
    expect(within(popover).getByText(BACKUP.id, { selector: "code" })).toBeVisible()
    expect(within(popover).getByText("chacha", { selector: "strong" })).toBeVisible()
    expect(within(popover).getByText("1", { selector: "strong" })).toBeVisible()
    expect(posted()).toHaveLength(0)
    fireEvent.click(within(popover).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(popover).toHaveClass("ant-popover-hidden"), { timeout: 3000 })
    expect(posted()).toHaveLength(0)
  }, 30000)

  it("posts the encoded unchanged artifact and exact row scope only after explicit confirmation", async () => {
    const row = await mount()
    expect(posted()).toHaveLength(0)
    const request = await confirm(row)
    expect(request).toEqual({ path: PATH, method: "POST", headers: { "Content-Type": "application/json" }, body: { dataset: "chacha", user_id: 1, confirm: true } })
    await waitFor(() => expect(message.success).toHaveBeenCalledWith("Backup restored successfully"))
    expect(message.error).not.toHaveBeenCalled()
    await waitFor(() => expect(mocks.request.mock.calls.filter(([r]) => r.path === "/api/v1/admin/backups")).toHaveLength(2))
  }, 30000)

  it.each([null, undefined])("confirms server-wide authnz restoration with %s owner without inventing a user", async (userId) => {
    backup = { ...BACKUP, id: "auth users.dump", dataset: "authnz", user_id: userId }
    const row = await mount()
    const popover = await openConfirmation(row)
    expect(within(popover).getByText("Server-wide users and authentication data")).toBeVisible()
    expect(within(popover).getByText("auth users.dump", { selector: "code" })).toBeVisible()
    fireEvent.click(within(popover).getByRole("button", { name: /Restore$/ }))
    await waitFor(() => expect(posted()).toHaveLength(1))
    expect(posted()[0]).toEqual({ path: "/api/v1/admin/backups/auth%20users.dump/restore", method: "POST", headers: { "Content-Type": "application/json" }, body: { dataset: "authnz", user_id: undefined, confirm: true } })
  }, 30000)

  it.each<[string, Record<string, unknown>]>([
    ["missing artifact", { id: undefined }], ["empty artifact", { id: "" }], ["blank artifact", { id: "  " }], ["numeric artifact", { id: 1 }],
    ["unknown dataset", { dataset: "unknown" }], ["missing dataset", { dataset: undefined }],
    ["missing per-user owner", { user_id: undefined }], ["null per-user owner", { user_id: null }], ["zero owner", { user_id: 0 }],
    ["negative owner", { user_id: -1 }], ["fractional owner", { user_id: 1.5 }], ["string owner", { user_id: "1" }],
    ["unsafe owner", { user_id: Number.MAX_SAFE_INTEGER + 1 }], ["invalid supplied authnz owner", { dataset: "authnz", user_id: -1 }]
  ])("disables restoration for %s metadata without guessing scope", async (_name, fields) => {
    backup = { ...BACKUP, ...fields }
    const row = await mount()
    const button = within(row).getByRole("button", { name: /Restore$/ })
    expect(button).toBeDisabled()
    fireEvent.click(button)
    expect(screen.queryByText(/This will overwrite current data/)).not.toBeInTheDocument()
    expect(posted()).toHaveLength(0)
  }, 30000)

  it("retains legacy artifact and size aliases with the same explicit scoped request", async () => {
    backup = { ...BACKUP, id: undefined, backup_id: "legacy.sqlib", size_bytes: undefined, size: 0 }
    const row = await mount()
    expect(within(row).getByText("legacy.sqlib", { selector: "code" })).toBeVisible()
    expect(within(row).getByText("0")).toBeVisible()
    const request = await confirm(row)
    expect(request).toEqual({ path: "/api/v1/admin/backups/legacy.sqlib/restore", method: "POST", headers: { "Content-Type": "application/json" }, body: { dataset: "chacha", user_id: 1, confirm: true } })
  }, 30000)

  it("reports backend refusal without claiming success, silently retrying or reloading the list", async () => {
    restoreRequest.mockRejectedValueOnce(new Error("Restore refused (POST /api/v1/admin/backups/artifact/restore)"))
    const row = await mount()
    await confirm(row)
    await waitFor(() => expect(message.error).toHaveBeenCalledWith("Restore refused"))
    expect(message.success).not.toHaveBeenCalled()
    expect(posted()).toHaveLength(1)
    expect(mocks.request.mock.calls.filter(([r]) => r.path === "/api/v1/admin/backups")).toHaveLength(1)
    await waitFor(() => expect(within(row).getByRole("button", { name: /Restore$/ })).not.toHaveClass("ant-btn-loading"))
  }, 30000)

  it("retains the forbidden admin guard without enabling restore", async () => {
    mocks.request.mockImplementation(async ({ path }: Request) => {
      if (path === "/api/v1/admin/backups") throw { status: 403 }
      return []
    })
    render(<ConfigProvider theme={{ token: { motion: false } }}><DataOpsPage /></ConfigProvider>)
    expect(await screen.findByRole("alert")).toHaveTextContent("Access Denied")
    expect(screen.queryByRole("button", { name: /Restore$/ })).not.toBeInTheDocument()
    expect(posted()).toHaveLength(0)
  }, 30000)
})
