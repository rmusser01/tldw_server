import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { ConfigProvider } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { McpHubExternalServer, McpHubServerReadiness } from "@/services/tldw/mcp-hub"
import { ExternalServersTab } from "../ExternalServersTab"

const mocks = vi.hoisted(() => ({ request: vi.fn() }))
// Real component and service; replace only the external HTTP request boundary.
vi.mock("@/services/background-proxy", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/background-proxy")>(), bgRequestClient: mocks.request
}))
type Request = { path: string; method: string; body?: Record<string, unknown> }
const BASE = "/api/v1/mcp/hub/external-servers"
const ID = "uat-c11-saved"
const NAME = "UAT C11 Saved Fixture"
let rows: McpHubExternalServer[]
let refreshResult: McpHubServerReadiness | Error
let createFailure: Error | undefined

const readiness = (state: McpHubServerReadiness["display_state"]): McpHubServerReadiness => ({
  server_id: ID, server_name: NAME, display_state: state, credential_state: "not_required",
  tool_count: state === "ready" ? 1 : 0,
  reason_codes: state === "needs_attention" ? ["discovery_failed"] : state === "no_tools" ? ["no_tools_returned"] : [],
  primary_reason_code: state === "needs_attention" ? "discovery_failed" : state === "no_tools" ? "no_tools_returned" : null,
  allowed_actions: state === "ready" ? ["open_tool_catalog"] : ["refresh_discovery"],
  message: state === "ready" ? "Tool catalog is ready." : state === "no_tools" ? "Discovery returned no tools." : "Discovery refresh failed.",
  current_operation: null, last_validation_at: null, last_discovery_at: null,
  last_successful_discovery_at: null, last_error_category: state === "needs_attention" ? "discovery_failed" : null,
  last_error_message: state === "needs_attention" ? "Discovery refresh failed." : null,
  refresh_result: { refreshed_servers: state === "needs_attention" ? 0 : 1, total_servers: 1,
    virtual_tools: state === "ready" ? 1 : 0, errors: state === "needs_attention" ? { [ID]: "Discovery refresh failed." } : {} }
})
const posts = () => mocks.request.mock.calls.map(([request]) => request as Request).filter(({ method }) => method === "POST")
const createRequests = () => posts().filter(({ path }) => path === BASE)
const savedPanel = async () => (await screen.findByRole("heading", { name: `${NAME} saved` })).closest("section")!
const mountAndFill = async () => {
  render(<ConfigProvider theme={{ token: { motion: false } }}><ExternalServersTab /></ConfigProvider>)
  await waitFor(() => expect(mocks.request).toHaveBeenCalledWith({ path: BASE, method: "GET" }))
  fireEvent.click(screen.getByRole("button", { name: "New Managed Server" }))
  fireEvent.click(screen.getByRole("button", { name: "Local stdio" }))
  for (const [label, value] of [["Server ID", ID], ["Name", NAME], ["Command", "/usr/bin/python3"]]) {
    fireEvent.change(screen.getByLabelText(label), { target: { value } })
  }
}
const clickSave = (discover = true) => fireEvent.click(screen.getByRole("button", {
  name: discover ? "Save and discover tools" : "Save without discovery"
}))

beforeEach(() => {
  rows = []; refreshResult = new Error("External discovery refresh tool is unavailable"); createFailure = undefined
  mocks.request.mockReset().mockImplementation(async ({ path, method, body }: Request) => {
    if (path === BASE && method === "GET") return rows
    if (path === "/api/v1/mcp/hub/tool-registry/summary") return { entries: [], modules: [] }
    if (path === "/api/v1/mcp/hub/readiness") return { servers: [] }
    if (path === BASE && method === "POST") {
      if (createFailure) throw createFailure
      const row: McpHubExternalServer = {
        id: String(body!.server_id), name: String(body!.name), transport: String(body!.transport),
        config: body!.config as Record<string, unknown>, enabled: body!.enabled === true,
        owner_scope_type: "global", owner_scope_id: null, server_source: "managed", secret_configured: false,
        binding_count: 0, runtime_executable: true, auth_template_present: false, auth_template_valid: false,
        auth_template_blocked_reason: "no_auth_template", credential_slots: []
      }
      rows = [row]; return row
    }
    if (path === `${BASE}/${ID}/refresh-discovery` && method === "POST") {
      if (refreshResult instanceof Error) throw refreshResult
      return refreshResult
    }
    if (path === `${BASE}/${ID}/auth-template` && method === "GET") return { mode: "none", mappings: [] }
    throw new Error(`Unexpected request: ${method} ${path}`)
  })
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe("MCP acknowledged save and optional discovery", () => {
  it("retains the committed server and closes the draft when discovery rejects", async () => {
    await mountAndFill(); clickSave()
    expect(await screen.findByRole("alert")).toHaveTextContent("saved, but tool discovery failed")
    expect(await savedPanel()).toBeVisible()
    expect(screen.queryByLabelText("Command")).not.toBeInTheDocument()
    expect(screen.getByLabelText("Server")).toHaveValue(ID)
    expect(createRequests()).toHaveLength(1)
    expect(posts().map(({ path }) => path)).toEqual([BASE, `${BASE}/${ID}/refresh-discovery`])
    expect(screen.queryByText(/Failed to create external server/)).not.toBeInTheDocument()
  }, 30000)

  it("retries discovery explicitly for the saved identity without creating again", async () => {
    await mountAndFill(); clickSave()
    const panel = await savedPanel()
    refreshResult = readiness("ready")
    fireEvent.click(within(panel).getByRole("button", { name: "Refresh discovery" }))
    await waitFor(() => expect(within(panel).getByText("Tool catalog is ready.")).toBeVisible())
    expect(within(panel).getByText("Ready")).toBeVisible()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(createRequests()).toHaveLength(1)
    expect(posts().map(({ path }) => path)).toEqual([BASE, `${BASE}/${ID}/refresh-discovery`, `${BASE}/${ID}/refresh-discovery`])
  }, 30000)

  it("retains the saved result and explicit retry when a subsequent discovery also fails", async () => {
    await mountAndFill(); clickSave()
    const panel = await savedPanel()
    refreshResult = new Error("Discovery still unavailable")
    fireEvent.click(within(panel).getByRole("button", { name: "Refresh discovery" }))
    expect(await screen.findByRole("alert")).toHaveTextContent("Failed to refresh tool discovery: Discovery still unavailable")
    expect(panel).toBeVisible()
    expect(within(panel).getByRole("button", { name: "Refresh discovery" })).toBeEnabled()
    expect(screen.queryByLabelText("Command")).not.toBeInTheDocument()
    expect(createRequests()).toHaveLength(1)
    expect(posts()).toHaveLength(3)
  }, 30000)

  it.each(["needs_attention", "no_tools"] as const)("shows returned %s readiness without a false Ready badge", async (state) => {
    refreshResult = readiness(state)
    await mountAndFill(); clickSave()
    const panel = await savedPanel()
    expect(within(panel).getByText(refreshResult.message)).toBeVisible()
    expect(within(panel).queryByText("Ready")).not.toBeInTheDocument()
    expect(within(panel).getByRole("button", { name: "Refresh discovery" })).toBeEnabled()
    expect(screen.queryByLabelText("Command")).not.toBeInTheDocument()
    expect(posts()).toHaveLength(2)
  }, 30000)

  it("shows returned ready status after a successful save and discovery", async () => {
    refreshResult = readiness("ready")
    await mountAndFill(); clickSave()
    const panel = await savedPanel()
    expect(within(panel).getByText("Ready")).toBeVisible()
    expect(within(panel).getByText("Tool catalog is ready.")).toBeVisible()
    expect(within(panel).queryByRole("button", { name: "Refresh discovery" })).not.toBeInTheDocument()
    expect(posts()).toHaveLength(2)
  }, 30000)

  it("retains saved without discovery as setup-required and never dispatches discovery automatically", async () => {
    await mountAndFill(); clickSave(false)
    const panel = await savedPanel()
    expect(within(panel).getByText("Tool discovery has not run for this server yet.")).toBeVisible()
    expect(within(panel).getByRole("button", { name: "Refresh discovery" })).toBeEnabled()
    expect(screen.queryByLabelText("Command")).not.toBeInTheDocument()
    expect(posts()).toHaveLength(1)
  }, 30000)

  it("retains an uncommitted draft on create refusal and does not attempt discovery", async () => {
    createFailure = new Error("Create refused")
    await mountAndFill(); clickSave()
    expect(await screen.findByRole("alert")).toHaveTextContent("Failed to create external server: Create refused")
    expect(screen.getByLabelText("Command")).toHaveValue("/usr/bin/python3")
    expect(screen.queryByRole("heading", { name: `${NAME} saved` })).not.toBeInTheDocument()
    expect(rows).toEqual([])
    expect(posts()).toHaveLength(1)
  }, 30000)
})
