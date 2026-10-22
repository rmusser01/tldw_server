import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { ConfigProvider, message } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import MaintenancePage from "../MaintenancePage"
import { tldwClient } from "@/services/tldw/TldwApiClient"

const mocks = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock("@/services/background-proxy", async (original) => ({ ...await original<typeof import("@/services/background-proxy")>(), bgRequest: mocks.request }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, fallback?: string | { defaultValue: string }) => typeof fallback === "object" ? fallback.defaultValue : fallback ?? key }) }))
type Request = { path: string; method: string; headers?: Record<string, string>; body?: unknown }
const FLAG = { key: "c09-owned-flag", scope: "org", org_id: 9, enabled: false, description: "Owned organization flag", target_user_ids: [7, 8], rollout_percent: 25, variant_value: "control" }
const INCIDENT = { id: "incident-c09", title: "C09 owned incident", status: "mitigating", severity: "low", summary: "Preserved incident context", created_at: "2026-10-19T23:00:00Z" }
const RUN = { id: "rotation-c09", mode: "dry_run", status: "failed", scope_summary: "domain=chat; queue=c09-owned; fields=payload,result", affected_count: 0, error_message: "Owned preview failure", started_at: "2026-10-19T23:00:00Z" }
let flags: unknown[] = []
let incident = INCIDENT
const mutations = () => mocks.request.mock.calls.map(([request]) => request as Request).filter((request) => request.method !== "GET")
const mount = () => render(<ConfigProvider theme={{ token: { motion: false } }}><MaintenancePage /></ConfigProvider>)
const openRuns = async () => {
  const header = screen.getByText("Rotation Runs", { exact: true }).closest('[role="button"]')
  expect(header).not.toBeNull()
  fireEvent.click(header!)
  await waitFor(() => expect(header).toHaveAttribute("aria-expanded", "true"))
}

beforeEach(() => {
  vi.clearAllMocks()
  flags = [FLAG]
  incident = INCIDENT
  vi.spyOn(message, "success").mockImplementation(() => undefined as never)
  vi.spyOn(message, "error").mockImplementation(() => undefined as never)
  mocks.request.mockImplementation(async ({ path, method, body }: Request) => {
    if (method === "GET" && path === "/api/v1/admin/maintenance") return { enabled: false, message: "Owned maintenance notice", allowlist_user_ids: [7], allowlist_emails: ["alice.c09@example.com"] }
    if (method === "GET" && path === "/api/v1/admin/feature-flags") return { items: flags, total: flags.length }
    if (method === "GET" && path === "/api/v1/admin/incidents") return { items: [incident], total: 1, limit: 50, offset: 0 }
    if (method === "GET" && path === "/api/v1/admin/maintenance/rotation-runs") return { items: [RUN], total: 1, limit: 50, offset: 0 }
    if (method === "PUT" && path === "/api/v1/admin/maintenance") return { enabled: false }
    if (method === "POST" && path === "/api/v1/admin/maintenance/rotation-runs") return { item: RUN }
    if (method === "PUT" && path === "/api/v1/admin/feature-flags/c09-owned-flag") return FLAG
    if (method === "DELETE" && [
      "/api/v1/admin/feature-flags/c09-owned-flag?scope=org&org_id=9",
      "/api/v1/admin/feature-flags/c09-owned-flag?scope=global&org_id=9&user_id=7",
      "/api/v1/admin/feature-flags/c09-owned-flag?scope=org&org_id=9&user_id=7",
      "/api/v1/admin/feature-flags/c09-owned-flag?scope=user&org_id=9&user_id=7"
    ].includes(path)) return { message: "deleted" }
    if (method === "POST" && path === "/api/v1/admin/incidents") return INCIDENT
    if (method === "PATCH" && path === "/api/v1/admin/incidents/incident-c09" && JSON.stringify(body) === JSON.stringify({ status: "resolved" })) {
      incident = { ...INCIDENT, status: "resolved" }
      return incident
    }
    throw new Error(`Unexpected external request: ${method} ${path}`)
  })
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe("Maintenance API contracts", () => {
  it.each([
    ["flags", () => tldwClient.listFeatureFlags(), FLAG],
    ["incidents", () => tldwClient.listIncidents(), INCIDENT],
    ["rotation history", () => tldwClient.listRotationRuns(), RUN]
  ] as const)("unwraps canonical %s items in the shared domain", async (_label, load, item) => {
    expect(await load()).toEqual([item])
    expect(mutations()).toHaveLength(0)
  })

  it("retains the real maintenance ID/email allowlist on save", async () => {
    mount()
    expect(await screen.findByDisplayValue("7, alice.c09@example.com")).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Save Changes" }))
    await waitFor(() => expect(mutations()).toEqual([{ path: "/api/v1/admin/maintenance", method: "PUT", headers: { "Content-Type": "application/json" }, body: { enabled: false, message: "Owned maintenance notice", allowlist_user_ids: [7], allowlist_emails: ["alice.c09@example.com"] } }]))
  }, 30000)

  it.each(["0", "9007199254740992", "192.168.1.1", "admin"])("rejects unsupported allowlist target %s without saving", async (value) => {
    mount()
    const input = await screen.findByDisplayValue("7, alice.c09@example.com")
    fireEvent.change(input, { target: { value } })
    fireEvent.click(screen.getByRole("button", { name: "Save Changes" }))
    await waitFor(() => expect(message.error).toHaveBeenCalledWith("Enter positive user IDs or valid email addresses"))
    expect(mutations()).toHaveLength(0)
  }, 30000)

  it("requests an explicit safe preview and never an unconfirmed execute", async () => {
    mount()
    await openRuns()
    fireEvent.click(screen.getByRole("button", { name: "Preview rotation" }))
    await waitFor(() => expect(mutations()).toEqual([{ path: "/api/v1/admin/maintenance/rotation-runs", method: "POST", headers: { "Content-Type": "application/json" }, body: { mode: "dry_run" } }]))
  }, 30000)

  it("shows the real operation mode, scope and failure without inferring success", async () => {
    mount()
    await openRuns()
    const id = await screen.findByText(RUN.id, { exact: true })
    const row = within(id.closest("tr")!)
    expect(row.getByText(RUN.mode, { exact: true })).toBeVisible()
    expect(row.getByText(RUN.scope_summary, { exact: true })).toBeVisible()
    expect(row.getByText(RUN.status, { exact: true })).toBeVisible()
    expect(row.getByText(RUN.error_message, { exact: true })).toBeVisible()
    expect(mutations()).toHaveLength(0)
  }, 30000)

  it("sends the existing org target when toggling or deleting its flag", async () => {
    mount()
    const key = await screen.findByText(FLAG.key, { selector: "code" })
    const row = within(key.closest("tr")!)
    expect(row.getByText("Organization #9", { exact: true })).toBeVisible()
    fireEvent.click(row.getByRole("switch"))
    await waitFor(() => expect(mutations()).toHaveLength(1))
    expect(mutations()[0]).toEqual({ path: "/api/v1/admin/feature-flags/c09-owned-flag", method: "PUT", headers: { "Content-Type": "application/json" }, body: { enabled: true, scope: "org", org_id: 9, target_user_ids: [7, 8], rollout_percent: 25, variant_value: "control" } })
    fireEvent.click(row.getByRole("button", { name: "Delete" }))
    fireEvent.click(await screen.findByRole("button", { name: "OK" }))
    await waitFor(() => expect(mutations()).toHaveLength(2))
    await waitFor(() => expect(message.success).toHaveBeenCalledWith(expect.stringContaining("deleted")))
    expect(message.error).not.toHaveBeenCalled()
    expect(mutations()[1]).toEqual({ path: "/api/v1/admin/feature-flags/c09-owned-flag?scope=org&org_id=9", method: "DELETE" })
  }, 30000)

  it.each(["global", "org", "user"])("preserves both identity IDs for a valid %s flag", async (scope) => {
    flags = [{ ...FLAG, scope, org_id: 9, user_id: 7 }]
    mount()
    const key = await screen.findByText(FLAG.key, { selector: "code" })
    const row = within(key.closest("tr")!)
    expect(row.getByText(/#9/)).toHaveTextContent(/#7/)
    fireEvent.click(row.getByRole("switch"))
    await waitFor(() => expect(mutations()).toHaveLength(1))
    expect(mutations()[0].body).toEqual({ enabled: true, scope, org_id: 9, user_id: 7, target_user_ids: [7, 8], rollout_percent: 25, variant_value: "control" })
    fireEvent.click(row.getByRole("button", { name: "Delete" }))
    fireEvent.click(await screen.findByRole("button", { name: "OK" }))
    await waitFor(() => expect(mutations()).toHaveLength(2))
    await waitFor(() => expect(message.success).toHaveBeenCalledWith(expect.stringContaining("deleted")))
    expect(message.error).not.toHaveBeenCalled()
    expect(mutations()[1]).toEqual({ path: `/api/v1/admin/feature-flags/c09-owned-flag?scope=${scope}&org_id=9&user_id=7`, method: "DELETE" })
  }, 30000)

  it.each([
    ["unknown scope", { scope: "unknown" }],
    ["missing active ID", { org_id: undefined }],
    ["zero active ID", { org_id: 0 }],
    ["invalid other ID", { user_id: "7" }]
  ] as const)("refuses flag mutation for %s without guessing a target", async (_label, fields) => {
    flags = [{ ...FLAG, ...fields }]
    mount()
    const key = await screen.findByText(FLAG.key, { selector: "code" })
    const row = within(key.closest("tr")!)
    const toggle = row.getByRole("switch"), remove = row.getByRole("button", { name: "Delete" })
    expect(toggle).toBeDisabled()
    expect(remove).toBeDisabled()
    fireEvent.click(toggle); fireEvent.click(remove)
    expect(mutations()).toHaveLength(0)
  }, 30000)

  it("offers only supported incident states and completes the exact status update", async () => {
    mount()
    const title = await screen.findByText(INCIDENT.title, { exact: true })
    fireEvent.mouseDown(within(title.closest("tr")!).getByRole("combobox"))
    const resolved = await screen.findByText("Resolved", { selector: ".ant-select-item-option-content" })
    const options = Array.from(document.querySelectorAll(".ant-select-dropdown:not(.ant-select-dropdown-hidden) .ant-select-item-option-content"), (item) => item.textContent).sort()
    expect(options).toEqual(["Investigating", "Mitigating", "Open", "Resolved"])
    fireEvent.click(resolved)
    await waitFor(() => expect(mutations()).toEqual([{ path: "/api/v1/admin/incidents/incident-c09", method: "PATCH", headers: { "Content-Type": "application/json" }, body: { status: "resolved" } }]))
    await waitFor(() => expect(message.success).toHaveBeenCalledWith("Incident updated"))
    await waitFor(() => expect(within(screen.getByText(INCIDENT.title, { exact: true }).closest("tr")!).getByText("resolved", { selector: ".ant-tag" })).toBeVisible())
    expect(message.error).not.toHaveBeenCalled()
  }, 30000)

  it("sends incident text as the supported summary field", async () => {
    mount()
    fireEvent.change(screen.getByPlaceholderText("Incident title"), { target: { value: "Owned new incident" } })
    fireEvent.change(screen.getByPlaceholderText("Description (optional)"), { target: { value: "Owned summary" } })
    fireEvent.click(screen.getByRole("button", { name: "Create Incident" }))
    await waitFor(() => expect(mutations()).toEqual([{ path: "/api/v1/admin/incidents", method: "POST", headers: { "Content-Type": "application/json" }, body: { title: "Owned new incident", severity: undefined, summary: "Owned summary" } }]))
  }, 30000)
})
