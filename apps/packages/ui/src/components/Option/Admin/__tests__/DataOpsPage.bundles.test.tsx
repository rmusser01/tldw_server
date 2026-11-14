import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { ConfigProvider, message } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import DataOpsPage from "../DataOpsPage"

const mocks = vi.hoisted(() => ({ request: vi.fn() }))
const authority = vi.hoisted(() => ({ config: { serverUrl: "https://c09.example", authMode: "single-user" as const, apiKey: "c09-fixture-only" }, authorityLoading: false }))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({ useCanonicalConnectionConfig: () => authority }))
const scope = { configSnapshot: authority.config, abortSignal: expect.any(AbortSignal) }
vi.mock("@/services/background-proxy", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/background-proxy")>(),
  bgRequest: mocks.request
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback ?? key })
}))

type Request = { path: string; method: string; headers?: Record<string, string>; body?: unknown }
const BUNDLE = {
  bundle_id: "bundle-c09-owned.zip", user_id: 7, datasets: ["chacha", "media"],
  created_at: "2026-10-19T22:00:00Z", size_bytes: 32768, manifest_version: 1
}
const posted = (): Request[] => mocks.request.mock.calls.map(([request]) => request as Request).filter((request) => request.method === "POST")
const mountBundles = async () => {
  render(<ConfigProvider theme={{ token: { motion: false } }}><DataOpsPage /></ConfigProvider>)
  fireEvent.click(screen.getByRole("tab", { name: "Bundles" }))
  await waitFor(() => expect(mocks.request).toHaveBeenCalledWith({ path: "/api/v1/admin/backups/bundles", method: "GET", ...scope }))
  return screen.getByRole("tabpanel", { name: "Bundles" })
}
const selectDataset = async (panel: HTMLElement, label: string) => {
  const selectedLabels = () => Array.from(panel.querySelectorAll(".ant-select-selection-item-content"), (item) => item.textContent)
  const wasSelected = selectedLabels().includes(label)
  const combo = within(panel).getByRole("combobox")
  fireEvent.mouseDown(combo)
  fireEvent.click(await screen.findByText(label, { selector: ".ant-select-item-option-content" }))
  const diagnostics = () => new Error(`Bundle form state: ${JSON.stringify({ selected: selectedLabels(), expanded: combo.getAttribute("aria-expanded"), inputs: Array.from(panel.querySelectorAll("input"), (input) => ({ id: input.id, role: input.getAttribute("role") })), labels: Array.from(panel.querySelectorAll("label"), (item) => ({ text: item.textContent, htmlFor: item.htmlFor })) })}`)
  await waitFor(() => expect(selectedLabels().includes(label)).toBe(!wasSelected), { onTimeout: diagnostics })
  fireEvent.keyDown(combo, { key: "Escape", keyCode: 27, which: 27 })
  await waitFor(() => expect(combo).toHaveAttribute("aria-expanded", "false"), { onTimeout: diagnostics })
  await waitFor(() => {
    const target = within(panel).queryByRole("spinbutton", { name: "Target user ID" })
    if (selectedLabels().includes("Chats & Notes")) expect(target).toBeVisible()
    else expect(target).toBeNull()
  }, { onTimeout: diagnostics })
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.spyOn(message, "success").mockImplementation(() => undefined as never)
  vi.spyOn(message, "error").mockImplementation(() => undefined as never)
  mocks.request.mockImplementation(async ({ path, method }: Request) => {
    if (method === "GET" && path === "/api/v1/admin/backups/bundles") return { items: [BUNDLE], total: 1, limit: 100, offset: 0 }
    if (method === "GET" && ["/api/v1/admin/backups", "/api/v1/admin/backup-schedules"].includes(path)) return { items: [] }
    if (method === "POST" && path === "/api/v1/admin/backups/bundles") return { item: BUNDLE, status: "created" }
    throw new Error(`Unexpected external request: ${method} ${path}`)
  })
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe("Admin bundle contract", () => {
  it("displays the canonical list envelope and exact artifact identity", async () => {
    const panel = await mountBundles()
    expect(await within(panel).findByText(BUNDLE.bundle_id, { selector: "code" })).toBeVisible()
    expect(posted()).toHaveLength(0)
  })

  it("requires an explicit per-user target before posting canonical datasets", async () => {
    const panel = await mountBundles()
    await selectDataset(panel, "Chats & Notes")
    fireEvent.click(within(panel).getByRole("button", { name: /Create Bundle$/ }))
    await waitFor(() => expect(within(panel).getByText("Enter a target user ID")).toBeVisible())
    expect(posted()).toHaveLength(0)
    await waitFor(() => expect(within(panel).getByRole("button", { name: /Create Bundle$/ })).toBeEnabled())
    fireEvent.change(within(panel).getByRole("spinbutton", { name: "Target user ID" }), { target: { value: "7" } })
    fireEvent.click(within(panel).getByRole("button", { name: /Create Bundle$/ }))
    await waitFor(() => expect(posted()).toEqual([{ path: "/api/v1/admin/backups/bundles", method: "POST", headers: { "Content-Type": "application/json" }, body: { datasets: ["chacha"], user_id: 7 }, ...scope }]))
    await waitFor(() => expect(message.success).toHaveBeenCalledWith("Bundle created"))
  }, 30000)

  it.each(["0", "1.5", "9007199254740992"])("rejects invalid target ID %s without a bundle write", async (value) => {
    const panel = await mountBundles()
    await selectDataset(panel, "Chats & Notes")
    fireEvent.change(within(panel).getByRole("spinbutton", { name: "Target user ID" }), { target: { value } })
    fireEvent.click(within(panel).getByRole("button", { name: /Create Bundle$/ }))
    await waitFor(() => expect(within(panel).getByText("Enter a valid target user ID")).toBeVisible())
    expect(posted()).toHaveLength(0)
  }, 30000)

  it("retires a per-user target when only the server-wide auth dataset remains", async () => {
    const panel = await mountBundles()
    await selectDataset(panel, "Chats & Notes")
    fireEvent.change(within(panel).getByRole("spinbutton", { name: "Target user ID" }), { target: { value: "7" } })
    await selectDataset(panel, "Chats & Notes")
    await selectDataset(panel, "Users & auth (server-wide)")
    fireEvent.click(within(panel).getByRole("button", { name: /Create Bundle$/ }))
    await waitFor(() => expect(posted()).toEqual([{ path: "/api/v1/admin/backups/bundles", method: "POST", headers: { "Content-Type": "application/json" }, body: { datasets: ["authnz"], user_id: undefined }, ...scope }]))
  }, 30000)
  it("exports an explicit exclusive PostgreSQL database scope without a user target", async () => {
    const panel = await mountBundles()
    await selectDataset(panel, "PostgreSQL database (all users and Jobs)")
    expect(within(panel).queryByRole("spinbutton", { name: "Target user ID" })).toBeNull()
    expect(within(panel).getByText(/Platform administrator only/)).toBeVisible()
    fireEvent.click(within(panel).getByRole("button", { name: /Create Bundle$/ }))
    await waitFor(() => expect(posted()).toEqual([{ path: "/api/v1/admin/backups/bundles", method: "POST", headers: { "Content-Type": "application/json" }, body: { datasets: ["postgresql"], user_id: undefined }, ...scope }]))
  }, 30000)

  it("requires clearing database scope before adding a legacy dataset", async () => {
    const panel = await mountBundles()
    await selectDataset(panel, "PostgreSQL database (all users and Jobs)")
    fireEvent.mouseDown(within(panel).getByRole("combobox"))
    const media = await screen.findByText("Media", { selector: ".ant-select-item-option-content" })
    expect(media.closest(".ant-select-item-option")).toHaveClass("ant-select-item-option-disabled")
    expect(posted()).toHaveLength(0)
  }, 30000)

  it("keeps legacy backup controls usable after a database-only permission denial", async () => {
    const panel = await mountBundles()
    await selectDataset(panel, "PostgreSQL database (all users and Jobs)")
    mocks.request.mockRejectedValueOnce(Object.assign(new Error("Forbidden"), { status: 403 }))
    fireEvent.click(within(panel).getByRole("button", { name: /Create Bundle$/ }))
    await waitFor(() => expect(message.error).toHaveBeenCalledWith("Platform administrator access is required for whole-database backups."))
    expect(within(panel).getByRole("button", { name: /Create Bundle$/ })).toBeVisible()
    expect(within(panel).getByRole("combobox")).toBeEnabled()
  }, 30000)

})
