import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ConfigProvider, notification } from "antd"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ACPPlayground } from "../index"
import { useACPSessionsStore } from "@/store/acp-sessions"
import { useWorkspaceStore } from "@/store/workspace"
import type { ACPAgentInfo, ACPAgentListResponse } from "@/services/acp/types"

const boundaries = vi.hoisted(() => ({ getConfig: vi.fn(), useStorage: vi.fn() }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({
  t: (key: string, fallback?: string | { defaultValue?: string }) =>
    typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
}) }))
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: (...args: unknown[]) => boundaries.useStorage(...args) }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { getConfig: (...args: unknown[]) => boundaries.getConfig(...args) } }))
vi.mock("@/hooks/useMediaQuery", () => ({ useMobile: () => false, useDesktop: () => true }))
// External WebSocket and unrelated panels are outside the creation boundary.
vi.mock("@/hooks/useACPSession", () => ({ useACPSession: () => ({
  state: "disconnected", isConnected: false, error: null, connect: vi.fn(), sendPrompt: vi.fn(),
  cancel: vi.fn(), approvePermission: vi.fn(), denyPermission: vi.fn()
}) }))
vi.mock("../ACPChatPanel", () => ({ ACPChatPanel: () => null }))
vi.mock("../ACPToolsPanel", () => ({ ACPToolsPanel: () => null }))
vi.mock("../ACPWorkspacePanel", () => ({ ACPWorkspacePanel: () => null }))
vi.mock("../ACPPermissionModal", () => ({ ACPPermissionModal: () => null }))

const readyAgents: ACPAgentListResponse = {
  default_agent: "goose",
  agents: ["goose", "opencode"].map((type): ACPAgentInfo => ({
    type, name: type === "goose" ? "Goose" : "OpenCode", description: `${type} native ACP entrypoint`,
    is_configured: true, requires_api_key: null, support_state: "supported_with_caveats",
    verification_level: "live_e2e_tested", compatibility_notes: "Native ACP entrypoint is ready to probe.",
    compatibility_docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md",
    entrypoint: {
      profile_key: type, entrypoint_strategy: "native_acp", probe_state: "ready_to_probe",
      acp_command: type, acp_args: ["acp"], primary_blocker: null, blockers: [],
      status_message: "Native ACP entrypoint is ready to probe.",
      docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md", display_command: `${type} acp`,
      display_binary_found: true, credential_state: "ready"
    }
  }))
}
const clients: QueryClient[] = []
const originalWorkspaceId = useWorkspaceStore.getState().workspaceId
let agentResponse: ACPAgentListResponse
let sessionStatus: number
let fetchMock: ReturnType<typeof vi.fn>
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { "Content-Type": "application/json" }
})
const sessionPosts = () => fetchMock.mock.calls.filter(([input]) => String(input).endsWith("/acp/sessions/new"))
const postedBody = () => JSON.parse(sessionPosts()[0][1].body as string)
const mount = (url = "/acp-playground?agent=opencode") => {
  window.history.pushState({}, "", url)
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  clients.push(client)
  render(<ConfigProvider theme={{ token: { motion: false } }}>
    <QueryClientProvider client={client}><ACPPlayground /></QueryClientProvider>
  </ConfigProvider>)
  return client
}
const open = async () => {
  await screen.findByTestId("acp-health-indicator")
  await waitFor(() => expect(screen.queryAllByTestId("acp-panel-loading")).toHaveLength(0))
  await userEvent.click(screen.getByRole("button", { name: "Create Session" }))
  const dialog = await screen.findByRole("dialog", { name: "Create ACP Session" })
  await within(dialog).findByText("Goose")
  await waitFor(() => expect(dialog).toBeVisible())
  return dialog
}
const card = (dialog: HTMLElement, name: string) => within(dialog).getByText(name).closest(".border-2")!
const fill = (dialog: HTMLElement, cwd = "/tmp/agent-workspace") => {
  fireEvent.change(within(dialog).getByLabelText("Working Directory"), { target: { value: cwd } })
  fireEvent.change(within(dialog).getByLabelText("Session Name"), { target: { value: "Query-bound session" } })
}
const submit = async (dialog: HTMLElement) => {
  fill(dialog)
  await userEvent.click(within(dialog).getByRole("button", { name: "Create Session" }))
  await waitFor(() => expect(sessionPosts()).toHaveLength(1))
  if (sessionStatus === 200) {
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
  }
}
beforeEach(() => {
  vi.clearAllMocks()
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
  useACPSessionsStore.getState().reset()
  useWorkspaceStore.setState({ workspaceId: "workspace-owned-74" })
  boundaries.useStorage.mockImplementation((_key: string, fallback: unknown) => [fallback, vi.fn(), { isLoading: false }])
  boundaries.getConfig.mockResolvedValue({ serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" })
  vi.spyOn(notification, "success").mockImplementation(() => undefined as never)
  vi.spyOn(notification, "error").mockImplementation(() => undefined as never)
  agentResponse = structuredClone(readyAgents)
  sessionStatus = 200
  fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(input), window.location.origin).pathname
    if (path.endsWith("/acp/health")) return json({ overall: "ok", runner: { status: "ok" }, agents: [] })
    if (path.endsWith("/acp/agents")) return json(agentResponse)
    if (path.endsWith("/acp/sessions")) return json({ sessions: [], total: 0, limit: 200, offset: 0 })
    if (path.endsWith("/acp/sessions/new")) {
      if (sessionStatus !== 200) return json({ detail: "Session creation is not permitted" }, sessionStatus)
      const request = JSON.parse(init!.body as string)
      return json({ session_id: "session-server-74", agent_type: request.agent_type, name: request.name, sandbox_session_id: null, sandbox_run_id: null, ssh_ws_url: null, ssh_user: null, persona_id: null, workspace_id: request.workspace_id, workspace_group_id: null, scope_snapshot_id: null })
    }
    if (path.endsWith("/detail") || path.endsWith("/usage")) {
      if (!path.includes("/session-server-74/")) return json({ detail: "Session not found" }, 404)
      const request = postedBody()
      const session = {
        session_id: "session-server-74", user_id: 74, agent_type: request.agent_type,
        name: request.name, cwd: request.cwd, workspace_id: request.workspace_id,
        status: "active", created_at: "2026-10-09T04:00:00Z", message_count: 0,
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }, tags: [], has_websocket: false,
        workspace_context: { mcp_server_count: 0, mcp_server_names: [] }, messages: [], fork_lineage: []
      }
      return json(session)
    }
    throw new Error(`Unexpected ACP fixture request: ${path}`)
  })
  vi.stubGlobal("fetch", fetchMock)
})
afterEach(() => {
  cleanup()
  clients.splice(0).forEach((client) => client.clear())
  useACPSessionsStore.getState().reset()
  useWorkspaceStore.setState({ workspaceId: originalWorkspaceId })
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe("ACP Registry launch agent selection", () => {
  it("submits the launched OpenCode instead of the first ready Goose and retains workspace scope", async () => {
    mount()
    const dialog = await open()
    await submit(dialog)
    expect(postedBody()).toEqual({ cwd: "/tmp/agent-workspace", name: "Query-bound session", agent_type: "opencode", workspace_id: "workspace-owned-74" })
  }, 30000)

  it("uses the extension hash query and shows the launched selection before any POST", async () => {
    mount("/options.html#/acp-playground?agent=%20opencode%20")
    const dialog = await open()
    expect(card(dialog, "OpenCode")).toHaveClass("border-primary")
    expect(card(dialog, "Goose")).not.toHaveClass("border-primary")
    expect(sessionPosts()).toHaveLength(0)
  }, 30000)

  it("keeps a ready manual override across catalog refresh", async () => {
    const client = mount()
    const dialog = await open()
    await userEvent.click(within(dialog).getByText("Goose"))
    agentResponse = { ...agentResponse, default_agent: "opencode", agents: [...agentResponse.agents].reverse() }
    await act(async () => { await client.invalidateQueries({ queryKey: ["acp", "agents"] }) })
    expect(card(dialog, "Goose")).toHaveClass("border-primary")
    await submit(dialog)
    expect(postedBody().agent_type).toBe("goose")
  }, 30000)

  it.each(["unknown-agent", "OpenCode", "<script>unknown</script>"])("refuses unavailable query %s without creating a different agent", async (agent) => {
    mount(`/acp-playground?agent=${encodeURIComponent(agent)}`)
    const dialog = await open()
    expect(within(dialog).getByText("Requested agent is unavailable")).toBeVisible()
    expect(within(dialog).getByRole("button", { name: "Create Session" })).toBeDisabled()
    fill(dialog)
    await userEvent.type(within(dialog).getByLabelText("Working Directory"), "{Enter}")
    expect(sessionPosts()).toHaveLength(0)
    expect(useACPSessionsStore.getState().sessions).toEqual({})
  }, 30000)

  it("keeps an explicit manual agent refused when refreshed catalog blocks it", async () => {
    const client = mount()
    const dialog = await open()
    await userEvent.click(within(dialog).getByText("Goose"))
    const chosen = agentResponse.agents[0]
    chosen.is_configured = false
    chosen.entrypoint = { ...chosen.entrypoint!, probe_state: "blocked", primary_blocker: "binary_missing", blockers: ["binary_missing"] }
    await act(async () => { await client.invalidateQueries({ queryKey: ["acp", "agents"] }) })
    await within(card(dialog, "Goose") as HTMLElement).findByText("Install agent binary")
    expect(card(dialog, "Goose")).toHaveClass("border-primary")
    expect(within(dialog).getByText("Requested agent is unavailable")).toBeVisible()
    expect(within(dialog).getByRole("button", { name: "Create Session" })).toBeDisabled()
    expect(sessionPosts()).toHaveLength(0)
  }, 30000)

  it("retains a blocked requested agent until an explicit ready manual choice", async () => {
    const requested = agentResponse.agents[1]
    requested.is_configured = false
    requested.entrypoint = { ...requested.entrypoint!, probe_state: "blocked", primary_blocker: "binary_missing", blockers: ["binary_missing"] }
    mount()
    const dialog = await open()
    expect(within(dialog).getByText("Requested agent is unavailable")).toBeVisible()
    expect(card(dialog, "OpenCode")).toHaveClass("border-primary")
    expect(within(dialog).getByRole("button", { name: "Create Session" })).toBeDisabled()
    expect(sessionPosts()).toHaveLength(0)
    await userEvent.click(within(dialog).getByText("Goose"))
    await submit(dialog)
    expect(postedBody().agent_type).toBe("goose")
  }, 30000)

  it.each(["", "?agent=%20%20"])("retains the existing first-ready default for an absent or blank query %s", async (query) => {
    mount(`/acp-playground${query}`)
    const dialog = await open()
    expect(card(dialog, "Goose")).toHaveClass("border-primary")
    await submit(dialog)
    expect(postedBody().agent_type).toBe("goose")
  }, 30000)

  it("applies the current URL only on the next open without overwriting a manually edited form", async () => {
    mount()
    const dialog = await open()
    await userEvent.click(within(dialog).getByText("Goose"))
    window.history.replaceState({}, "", "/acp-playground?agent=unknown-next")
    expect(card(dialog, "Goose")).toHaveClass("border-primary")
    expect(within(dialog).queryByText("Requested agent is unavailable")).not.toBeInTheDocument()
    await userEvent.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    const reopened = await open()
    expect(within(reopened).getByText("Requested agent is unavailable")).toBeVisible()
    expect(within(reopened).getByRole("button", { name: "Create Session" })).toBeDisabled()
    expect(sessionPosts()).toHaveLength(0)
  }, 30000)

  it("keeps relative cwd validation before dispatch for the selected launch agent", async () => {
    mount()
    const dialog = await open()
    fill(dialog, "relative/workspace")
    await userEvent.click(within(dialog).getByRole("button", { name: "Create Session" }))
    expect(await within(dialog).findByText("Please enter an absolute path")).toBeVisible()
    expect(sessionPosts()).toHaveLength(0)
  }, 30000)

  it("surfaces actual transport permission refusal and removes the pending local session", async () => {
    sessionStatus = 403
    mount()
    const dialog = await open()
    await submit(dialog)
    expect(await within(dialog).findByText("Session creation is not permitted")).toBeVisible()
    expect(useACPSessionsStore.getState().sessions).toEqual({})
    expect(notification.success).not.toHaveBeenCalled()
    expect(sessionPosts()).toHaveLength(1)
  }, 30000)
})
