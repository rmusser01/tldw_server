import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { ConfigProvider, message } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import DataOpsPage from "../DataOpsPage"

const mocks = vi.hoisted(() => ({ request: vi.fn() }))
// Only the external transport is replaced: actual client domain methods and Antd UI remain mounted.
vi.mock("@/services/background-proxy", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/background-proxy")>(),
  bgRequest: mocks.request
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback ?? key })
}))

const BASE = "/api/v1/admin/data-subject-requests"
const CATEGORIES = ["media_records", "chat_messages", "notes", "audit_events", "embeddings"]
const PREVIEW = {
  requester_identifier: "admin@example.test", resolved_user_id: 1, request_type: "export",
  selected_categories: CATEGORIES,
  counts: { media_records: 1, chat_messages: 13, notes: 3, audit_events: 26, embeddings: 0 },
  summary: [], coverage_metadata: {}
}
const ITEM = {
  id: 41, client_request_id: "existing-preview-id", requester_identifier: "admin@example.test",
  resolved_user_id: 1, request_type: "export", status: "recorded", selected_categories: CATEGORIES,
  preview_summary: [], coverage_metadata: {}, requested_by_user_id: 1,
  requested_at: "2026-10-09T04:12:00Z", notes: "Intake review only"
}
type Request = { path: string, method: string, body?: Record<string, unknown>, headers?: Record<string, string> }
const previewRequest = vi.fn<() => Promise<unknown>>()
const recordRequest = vi.fn<(body: Record<string, unknown>) => Promise<unknown>>()
let history: unknown
const requests = (path: string, method = "POST"): Request[] =>
  mocks.request.mock.calls.map(([request]) => request as Request).filter((request) => request.path === path && request.method === method)
const select = async (label: string, option: string) => {
  fireEvent.mouseDown(screen.getByRole("combobox", { name: label }))
  fireEvent.click(await screen.findByText(option, { selector: ".ant-select-item-option-content" }))
}
const mountDsr = async () => {
  render(<ConfigProvider theme={{ token: { motion: false } }}><DataOpsPage /></ConfigProvider>)
  fireEvent.click(screen.getByRole("tab", { name: "Data Subject Requests" }))
  await screen.findByRole("textbox", { name: "Requester Identifier" })
  await waitFor(() => expect(requests(`${BASE}?limit=100`, "GET")).toHaveLength(1))
}
const preview = async (type = "Data Export") => {
  fireEvent.change(screen.getByRole("textbox", { name: "Requester Identifier" }), { target: { value: "1" } })
  await select("Request Type", type)
  fireEvent.click(screen.getByRole("button", { name: /Preview$/ }))
  return await screen.findByRole("dialog", { name: "DSR Preview" })
}
const record = async () => {
  fireEvent.click(within(screen.getByRole("dialog", { name: "DSR Preview" })).getByRole("button", { name: "Record DSR" }))
  await waitFor(() => expect(requests(BASE)).not.toHaveLength(0))
  return requests(BASE).at(-1)!
}

beforeEach(() => {
  vi.clearAllMocks()
  previewRequest.mockReset()
  recordRequest.mockReset()
  history = { items: [], total: 0, limit: 100, offset: 0 }
  previewRequest.mockResolvedValue(PREVIEW)
  recordRequest.mockImplementation(async (body) => {
    const item = { ...ITEM, ...body }
    history = { items: [item], total: 1, limit: 100, offset: 0 }
    return { item }
  })
  vi.spyOn(message, "success").mockImplementation(() => undefined as never)
  vi.spyOn(message, "error").mockImplementation(() => undefined as never)
  mocks.request.mockImplementation(async (request: Request) => {
    if (request.path === "/api/v1/admin/backups" || request.path === "/api/v1/admin/backup-schedules") return []
    if (request.path === `${BASE}?limit=100` && request.method === "GET") return history
    if (request.path === `${BASE}/preview` && request.method === "POST") return previewRequest()
    if (request.path === BASE && request.method === "POST") return recordRequest(request.body ?? {})
    if (request.path === `${BASE}/42/execute` && request.method === "POST") return { status: "completed" }
    throw new Error(`Unexpected external request: ${request.method} ${request.path}`)
  })
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe("DSR intake contract", () => {
  it("records a successful export preview with a bounded UUID and displays canonical returned history", async () => {
    await mountDsr()
    fireEvent.change(screen.getByRole("textbox", { name: "Notes (optional)" }), { target: { value: "Intake review only" } })
    const dialog = await preview()
    await waitFor(() => {
      expect(within(dialog).getByText("13 record(s)")).toBeVisible()
      expect(within(dialog).getByText("0 record(s)")).toBeVisible()
    })
    const request = await record()
    expect(request).toEqual({ path: BASE, method: "POST", headers: { "Content-Type": "application/json" }, body: {
      requester_identifier: "1", request_type: "export", categories: undefined,
      client_request_id: expect.stringMatching(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i),
      notes: "Intake review only"
    } })
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "DSR Preview" })).not.toBeInTheDocument())
    expect(await screen.findByText("41", { selector: "td" })).toBeVisible()
    expect(requests(`${BASE}/42/execute`)).toHaveLength(0)
  }, 30000)

  it("reuses exactly the preview identity and fields after a record transport failure", async () => {
    recordRequest.mockRejectedValueOnce(new Error("Temporary intake failure"))
    await mountDsr()
    await preview()
    const first = await record()
    await waitFor(() => expect(message.error).toHaveBeenCalledWith("Temporary intake failure"))
    expect(screen.getByRole("dialog", { name: "DSR Preview" })).toBeVisible()
    fireEvent.change(screen.getByRole("textbox", { name: "Requester Identifier" }), { target: { value: "2" } })
    fireEvent.click(screen.getByRole("button", { name: "Record DSR" }))
    await waitFor(() => expect(requests(BASE)).toHaveLength(2))
    expect(first.body?.client_request_id).toEqual(expect.any(String))
    expect(requests(BASE)[1].body).toEqual(first.body)
    expect(requests(BASE)[1].body?.requester_identifier).toBe("1")
  }, 30000)

  it("assigns a new identity after canceling a preview and completing a new one", async () => {
    recordRequest.mockRejectedValueOnce(new Error("Retry later"))
    await mountDsr()
    await preview()
    const first = await record()
    await waitFor(() => expect(message.error).toHaveBeenCalledWith("Retry later"))
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "DSR Preview" })).not.toBeInTheDocument())
    await preview("Access Request")
    fireEvent.click(screen.getByRole("button", { name: "Record DSR" }))
    await waitFor(() => expect(requests(BASE)).toHaveLength(2))
    const second = requests(BASE)[1]
    expect(second.body?.client_request_id).toEqual(expect.any(String))
    expect(second.body?.client_request_id).not.toBe(first.body?.client_request_id)
    expect(second.body?.request_type).toBe("access")
  }, 30000)

  it("offers only the three supported request types", async () => {
    await mountDsr()
    fireEvent.mouseDown(screen.getByRole("combobox", { name: "Request Type" }))
    for (const label of ["Erasure (Right to be Forgotten)", "Data Export", "Access Request"]) {
      await waitFor(() => expect(screen.getByText(label, { selector: ".ant-select-item-option-content" })).toBeVisible())
    }
    expect(screen.queryByText("Rectification")).not.toBeInTheDocument()
  }, 30000)

  it("sends authoritative selected category keys unchanged through preview and record", async () => {
    await mountDsr()
    fireEvent.change(screen.getByRole("textbox", { name: "Requester Identifier" }), { target: { value: "1" } })
    await select("Request Type", "Data Export")
    for (const label of ["Media", "Chats", "Notes", "Audit log", "Embeddings"]) await select("Categories (optional)", label)
    fireEvent.click(screen.getByRole("button", { name: /Preview$/ }))
    await screen.findByRole("dialog", { name: "DSR Preview" })
    expect(requests(`${BASE}/preview`)[0].body).toEqual({ requester_identifier: "1", request_type: "export", categories: ["media_records", "chat_messages", "notes", "audit_events", "embeddings"] })
    const request = await record()
    expect(request.body?.categories).toEqual(["media_records", "chat_messages", "notes", "audit_events", "embeddings"])
    expect(request.body?.client_request_id).toEqual(expect.any(String))
    expect(screen.queryByText("Profile", { selector: ".ant-select-item-option-content" })).not.toBeInTheDocument()
  }, 30000)

  it("renders items history and the authoritative requested_at timestamp", async () => {
    history = { items: [ITEM], total: 1, limit: 100, offset: 0 }
    await mountDsr()
    const cell = await screen.findByText("41", { selector: "td" })
    expect(cell.closest("tr")).toHaveTextContent("admin@example.test")
    expect(cell.closest("tr")).toHaveTextContent(new Date(ITEM.requested_at).toLocaleString())
    expect(within(cell.closest("tr")!).queryByRole("button", { name: /Execute$/ })).not.toBeInTheDocument()
  }, 30000)

  it.each(["array", "data", "requests"])("retains the legacy %s history envelope and created_at fallback", async (envelope) => {
    const legacy = { ...ITEM, requested_at: undefined, created_at: ITEM.requested_at }
    history = envelope === "array" ? [legacy] : { [envelope]: [legacy] }
    await mountDsr()
    const cell = await screen.findByText("41", { selector: "td" })
    expect(cell.closest("tr")).toHaveTextContent(new Date(ITEM.requested_at).toLocaleString())
  }, 30000)

  it("keeps erasure execution behind explicit confirmation and excludes completed/export rows", async () => {
    history = { requests: [ITEM, { ...ITEM, id: 42, request_type: "erasure" }, { ...ITEM, id: 43, request_type: "erasure", status: "completed" }] }
    await mountDsr()
    const erasureCell = await screen.findByText("42", { selector: "td" })
    const erasureRow = erasureCell.closest("tr")!
    expect(screen.getAllByRole("button", { name: /Execute$/ })).toHaveLength(1)
    fireEvent.click(within(erasureRow).getByRole("button", { name: /Execute$/ }))
    const confirmation = await screen.findByText("Execute this DSR? This action cannot be undone.")
    expect(requests(`${BASE}/42/execute`)).toHaveLength(0)
    const popover = confirmation.closest(".ant-popover")
    if (!popover) throw new Error("Missing erasure confirmation")
    fireEvent.click(within(popover as HTMLElement).getByRole("button", { name: "Cancel" }))
    expect(requests(`${BASE}/42/execute`)).toHaveLength(0)
    fireEvent.click(within(erasureRow).getByRole("button", { name: /Execute$/ }))
    const secondConfirmation = await screen.findByText("Execute this DSR? This action cannot be undone.")
    const secondPopover = secondConfirmation.closest(".ant-popover")
    if (!secondPopover) throw new Error("Missing repeated erasure confirmation")
    fireEvent.click(within(secondPopover as HTMLElement).getByRole("button", { name: /Execute$/ }))
    await waitFor(() => expect(requests(`${BASE}/42/execute`)).toEqual([{ path: `${BASE}/42/execute`, method: "POST", headers: { "Content-Type": "application/json" }, body: {} }]))
  }, 30000)

  it("does not record invalid or failed previews and retains sanitized preview feedback", async () => {
    await mountDsr()
    fireEvent.click(screen.getByRole("button", { name: /Preview$/ }))
    await screen.findByText("Requester identifier is required")
    expect(requests(`${BASE}/preview`)).toHaveLength(0)
    previewRequest.mockRejectedValueOnce(new Error("Unavailable (POST /api/v1/admin/data-subject-requests/preview)"))
    fireEvent.change(screen.getByRole("textbox", { name: "Requester Identifier" }), { target: { value: "1" } })
    await select("Request Type", "Data Export")
    fireEvent.click(screen.getByRole("button", { name: /Preview$/ }))
    await waitFor(() => expect(message.error).toHaveBeenCalledWith("Unavailable"))
    expect(screen.queryByRole("button", { name: "Record DSR" })).not.toBeInTheDocument()
    expect(requests(BASE)).toHaveLength(0)
  }, 30000)

  it("preserves the owner permission guard on a forbidden history response", async () => {
    mocks.request.mockImplementation(async ({ path }: Request) => {
      if (path === `${BASE}?limit=100`) throw { status: 403 }
      return []
    })
    await mountDsr()
    expect(await screen.findByRole("alert")).toHaveTextContent("Access Denied")
    expect(screen.queryByRole("button", { name: "Record DSR" })).not.toBeInTheDocument()
    expect(requests(BASE)).toHaveLength(0)
  }, 30000)
})
