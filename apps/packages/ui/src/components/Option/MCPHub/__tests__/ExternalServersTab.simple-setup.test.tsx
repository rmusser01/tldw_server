import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { ConfigProvider } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ExternalServersTab } from "../ExternalServersTab"

const mocks = vi.hoisted(() => ({ request: vi.fn() }))
// Actual component and managed service; replace only external request transport.
vi.mock("@/services/background-proxy", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/background-proxy")>(), bgRequestClient: mocks.request
}))
type Request = { path: string; method: string; body?: Record<string, unknown> }
const BASE = "/api/v1/mcp/hub/external-servers"
const COMMAND = "/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv-uat-fixall-py312-20261007/bin/python"
const SCRIPT = "/Users/macbook-dev/.codex/worktrees/uat-native-regenerate-20261008/tldw_server2/tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py"
const posts = (): Request[] => mocks.request.mock.calls.map(([request]) => request as Request).filter((request) => request.method === "POST")
const created = () => posts().filter((request) => request.path === BASE)
const fill = (label: string, value: string) => fireEvent.change(screen.getByLabelText(label), { target: { value } })
const mount = async (mode: "Local stdio" | "Advanced/manual" | "Import config") => {
  render(<ConfigProvider theme={{ token: { motion: false } }}><ExternalServersTab /></ConfigProvider>)
  await waitFor(() => expect(mocks.request).toHaveBeenCalledWith({ path: BASE, method: "GET" }))
  fireEvent.click(screen.getByRole("button", { name: "New Managed Server" }))
  fireEvent.click(screen.getByRole("button", { name: mode }))
}
const fillIdentity = () => { fill("Server ID", "uat-smoke"); fill("Name", "UAT Smoke") }
const save = async (discover = false) => {
  fireEvent.click(screen.getByRole("button", { name: discover ? "Save and discover tools" : "Save without discovery" }))
  await waitFor(() => expect(created()).toHaveLength(1))
  return created()[0]
}
const expected = (config: Record<string, unknown>, extra = {}) => ({
  path: BASE, method: "POST", body: { server_id: "uat-smoke", name: "UAT Smoke", transport: "stdio", config, owner_scope_type: "global", enabled: true, ...extra }
})

beforeEach(() => {
  mocks.request.mockReset().mockImplementation(async ({ path, method, body }: Request) => {
    if (path === BASE && method === "GET") return []
    if (path === "/api/v1/mcp/hub/tool-registry/summary") return { entries: [], modules: [] }
    if (path === "/api/v1/mcp/hub/readiness") return { servers: [] }
    if (path === BASE && method === "POST") return { id: body!.server_id, name: body!.name, transport: body!.transport, config: body!.config, enabled: body!.enabled, owner_scope_type: body!.owner_scope_type, server_source: "managed" }
    if (path === `${BASE}/uat-smoke/refresh-discovery` && method === "POST") return { server_id: "uat-smoke", server_name: "UAT Smoke", display_state: "no_tools", credential_state: "not_required", tool_count: 0, reason_codes: ["no_tools"], allowed_actions: [], message: "No tools discovered", current_operation: null }
    throw new Error(`Unexpected external request: ${method} ${path}`)
  })
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe("MCP simple setup canonical request contract", () => {
  it("saves all stdio fields nested with the selected owner and disabled state without discovery", async () => {
    await mount("Local stdio"); fillIdentity()
    fill("Command", ` ${COMMAND} `); fill("Args", ` -u   ${SCRIPT} `)
    fill("Working Directory", " /tmp/uat-smoke "); fill("Env vars", "READ_MODE=readonly\nMODE=smoke")
    fill("Owner Scope", "team"); fireEvent.click(screen.getByRole("checkbox", { name: "Enabled" }))
    expect(await save()).toEqual(expected({ stdio: { command: COMMAND, args: ["-u", SCRIPT], cwd: "/tmp/uat-smoke", env: { READ_MODE: "readonly", MODE: "smoke" } } }, { owner_scope_type: "team", enabled: false }))
    expect(posts()).toHaveLength(1)
    expect(await screen.findByText("UAT Smoke saved")).toBeVisible()
  }, 30000)

  it("preserves default owner/enabled and omitted optional stdio fields for Save and discovery", async () => {
    await mount("Local stdio"); fillIdentity(); fill("Command", COMMAND)
    expect(await save(true)).toEqual(expected({ stdio: { command: COMMAND, args: [] } }))
    await waitFor(() => expect(posts()).toEqual([expected({ stdio: { command: COMMAND, args: [] } }), { path: `${BASE}/uat-smoke/refresh-discovery`, method: "POST" }]))
    expect(await screen.findByText("UAT Smoke saved")).toBeVisible()
  }, 30000)

  it.each(["Server ID", "Name", "Command"])("refuses missing %s before create or discovery", async (field) => {
    await mount("Local stdio"); fillIdentity(); fill("Command", COMMAND); fill(field, " ")
    fireEvent.click(screen.getByRole("button", { name: "Save and discover tools" }))
    expect(await screen.findByRole("alert")).toHaveTextContent(field === "Command" ? "Command is required" : "Server id and name are required")
    expect(posts()).toHaveLength(0)
  }, 30000)

  it("retains malformed env validation before create/discovery", async () => {
    await mount("Local stdio"); fillIdentity(); fill("Command", COMMAND); fill("Env vars", "BROKEN_LINE")
    fireEvent.click(screen.getByRole("button", { name: "Save and discover tools" }))
    expect(await screen.findByRole("alert")).toHaveTextContent("Env vars must use KEY=value lines")
    expect(posts()).toHaveLength(0)
  }, 30000)

  it("marks unsupported HTTP/SSE unavailable with supported alternatives and never creates a server", async () => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><ExternalServersTab /></ConfigProvider>)
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith({ path: BASE, method: "GET" }))
    fireEvent.click(screen.getByRole("button", { name: "New Managed Server" }))
    const choice = screen.getByRole("button", { name: "HTTP/SSE" })
    expect(choice).toBeDisabled(); fireEvent.click(choice)
    expect(screen.getByText(/HTTP\/SSE is unavailable.*stdio.*WebSocket/i)).toBeVisible()
    expect(screen.queryByLabelText("URL")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Advanced/manual" })).toBeEnabled()
    expect(posts()).toHaveLength(0)
  }, 30000)

  it("advertises only the actual stdio/WebSocket manual transport choices", async () => {
    await mount("Advanced/manual")
    expect(Array.from((screen.getByLabelText("Transport") as HTMLSelectElement).options).map((option) => option.value)).toEqual(["stdio", "websocket"])
  }, 30000)

  it.each(["stdio", "websocket"])("retains valid nested %s manual configuration without automatic discovery", async (transport) => {
    await mount("Advanced/manual"); fillIdentity(); fill("Transport", transport)
    const config = transport === "stdio" ? { stdio: { command: COMMAND, args: ["-u", SCRIPT] }, policy: { allow_writes: false } } : { websocket: { url: "wss://example.test/mcp", headers: { "X-Read-Mode": "readonly" } }, policy: { allow_writes: false } }
    fill("Config JSON", JSON.stringify(config))
    expect(await save()).toEqual(expected(config, { transport }))
    expect(posts()).toHaveLength(1)
  }, 30000)

  it("preserves nested import configuration/owner/enabled and advertises a supported nested example", async () => {
    await mount("Import config")
    const editor = screen.getByLabelText("Managed Config JSON") as HTMLTextAreaElement
    const example = JSON.parse(editor.placeholder)
    expect(example.transport).toBe("websocket")
    expect(example.config.websocket.url).toMatch(/^wss:\/\//)
    const config = { stdio: { command: COMMAND, args: ["-u", SCRIPT] }, policy: { allow_writes: false } }
    fill("Managed Config JSON", JSON.stringify({ server_id: "uat-smoke", name: "UAT Smoke", transport: "stdio", config, owner_scope_type: "user", enabled: false }))
    expect(await save()).toEqual(expected(config, { owner_scope_type: "user", enabled: false }))
    expect(posts()).toHaveLength(1)
  }, 30000)

  it("retains JSON object validation in Advanced/manual before any request", async () => {
    await mount("Advanced/manual"); fillIdentity(); fill("Config JSON", "[]")
    fireEvent.click(screen.getByRole("button", { name: "Save without discovery" }))
    expect(await screen.findByRole("alert")).toHaveTextContent("Server config JSON must decode to an object")
    expect(posts()).toHaveLength(0)
  }, 30000)

  it("retains create failure feedback and draft without discovery or false success", async () => {
    const original = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((request: Request) => request.path === BASE && request.method === "POST" ? Promise.reject(new Error("Create refused")) : original(request))
    await mount("Local stdio"); fillIdentity(); fill("Command", COMMAND); await save(true)
    expect(await screen.findByRole("alert")).toHaveTextContent("Failed to create external server: Create refused")
    expect(screen.getByLabelText("Command")).toHaveValue(COMMAND)
    expect(screen.queryByText("UAT Smoke saved")).not.toBeInTheDocument()
    expect(posts()).toHaveLength(1)
  }, 30000)

  it("retains saved identity and discovery refusal after one create without automatic retry", async () => {
    const original = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((request: Request) => request.path.endsWith("/refresh-discovery") ? Promise.reject(new Error("Discovery refused")) : original(request))
    await mount("Local stdio"); fillIdentity(); fill("Command", COMMAND); await save(true)
    expect(await screen.findByRole("alert")).toHaveTextContent("UAT Smoke saved, but tool discovery failed: Discovery refused")
    expect(created()).toHaveLength(1); expect(posts()).toHaveLength(2)
    expect(await screen.findByText("UAT Smoke saved")).toBeVisible()
    expect(screen.queryByLabelText("Command")).not.toBeInTheDocument()
  }, 30000)
})
