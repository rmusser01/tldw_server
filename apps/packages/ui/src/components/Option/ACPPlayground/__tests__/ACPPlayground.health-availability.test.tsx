import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ACPPlayground } from "../index"
import { useACPSessionsStore } from "@/store/acp-sessions"
import type { ACPAgentListResponse } from "@/services/acp/types"

const boundaries = vi.hoisted(() => ({
  getConfig: vi.fn(),
  useStorage: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (...args: unknown[]) => boundaries.useStorage(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getConfig: (...args: unknown[]) => boundaries.getConfig(...args) }
}))

vi.mock("@/hooks/useMediaQuery", () => ({ useMobile: () => false, useDesktop: () => true }))

// The external WebSocket is unrelated to health and modal admission.
vi.mock("@/hooks/useACPSession", () => ({
  useACPSession: () => ({
    state: "disconnected",
    isConnected: false,
    error: null,
    connect: vi.fn(),
    sendPrompt: vi.fn(),
    cancel: vi.fn(),
    approvePermission: vi.fn(),
    denyPermission: vi.fn()
  })
}))

// Keep the actual health header, session panel, creation modal and REST transport.
vi.mock("../ACPChatPanel", () => ({ ACPChatPanel: () => null }))
vi.mock("../ACPToolsPanel", () => ({ ACPToolsPanel: () => null }))
vi.mock("../ACPWorkspacePanel", () => ({ ACPWorkspacePanel: () => null }))
vi.mock("../ACPPermissionModal", () => ({ ACPPermissionModal: () => null }))

const readyAgents: ACPAgentListResponse = {
  default_agent: "opencode",
  agents: [{
    type: "opencode",
    name: "OpenCode",
    description: "Native ACP coding agent",
    is_configured: true,
    requires_api_key: null,
    support_state: "supported_with_caveats",
    verification_level: "live_e2e_tested",
    compatibility_notes: "The selected native ACP entrypoint is ready to probe.",
    compatibility_docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md",
    entrypoint: {
      profile_key: "opencode",
      entrypoint_strategy: "native_acp",
      probe_state: "ready_to_probe",
      acp_command: "opencode",
      acp_args: ["acp"],
      primary_blocker: null,
      blockers: [],
      status_message: "Native ACP entrypoint is ready to probe.",
      docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md",
      display_command: "opencode acp",
      display_binary_found: true,
      credential_state: "ready"
    }
  }]
}

// Mirrors ACPHealthResponse and the canonical endpoint's successful fields.
const canonicalHealth = {
  timestamp: "2026-10-09T02:45:00Z",
  runner: { status: "ok", path: "/test/bin/tldw-agent-acp", source: "binary_path" },
  agents: [{
    agent_type: "opencode",
    name: "OpenCode",
    status: "available",
    binary_found: true,
    api_key_set: true,
    is_configured: true,
    support_state: "supported_with_caveats",
    verification_level: "live_e2e_tested",
    compatibility_notes: "Native ACP entrypoint is ready to probe.",
    compatibility_docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md",
    entrypoint: readyAgents.agents[0].entrypoint
  }],
  runner_probe: { status: "ok", detail: "Runner process is alive" },
  routes: { stable_only: true, acp_enabled: true, note: null },
  overall: "ok",
  message: null
}

const jsonResponse = (payload: unknown, status = 200) =>
  new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" }
  })

describe("ACP Playground canonical health availability", () => {
  const clients: QueryClient[] = []
  let healthResponse: () => Promise<Response>
  let agentResponse: ACPAgentListResponse
  let fetchMock: ReturnType<typeof vi.fn>

  const mountPlayground = () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })
    clients.push(client)
    return render(
      <QueryClientProvider client={client}>
        <ACPPlayground />
      </QueryClientProvider>
    )
  }

  const waitForHealth = async () => {
    await screen.findByTestId("acp-health-indicator")
    await waitFor(() => expect(screen.queryAllByTestId("acp-panel-loading")).toHaveLength(0))
  }

  const openCreation = async () => {
    await userEvent.click(screen.getByRole("button", { name: "Create Session" }))
    const dialog = await screen.findByRole("dialog", { name: "Create ACP Session" })
    await within(dialog).findByText("OpenCode")
    return dialog
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    window.history.pushState({}, "", "/acp-playground?agent=opencode")
    useACPSessionsStore.getState().reset()
    boundaries.useStorage.mockImplementation((_key: string, fallback: unknown) =>
      [fallback, vi.fn(), { isLoading: false }]
    )
    boundaries.getConfig.mockResolvedValue({
      serverUrl: window.location.origin,
      authMode: "single-user",
      authSource: "cookie-session"
    })
    agentResponse = structuredClone(readyAgents)
    healthResponse = async () => jsonResponse(canonicalHealth)
    fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const path = new URL(String(input), window.location.origin).pathname
      if (path.endsWith("/acp/health")) return healthResponse()
      if (path.endsWith("/acp/agents")) return jsonResponse(agentResponse)
      if (path.endsWith("/acp/sessions")) {
        return jsonResponse({ sessions: [], total: 0, limit: 200, offset: 0 })
      }
      throw new Error("Unexpected ACP fixture request")
    })
    vi.stubGlobal("fetch", fetchMock)
  })

  afterEach(() => {
    cleanup()
    clients.splice(0).forEach((client) => client.clear())
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
  })

  it("exposes prominent session creation for the canonical successful backend health shape", async () => {
    mountPlayground()
    await waitForHealth()

    expect(screen.getByRole("button", { name: "Create Session" })).toBeInTheDocument()
    expect(screen.getByTestId("acp-health-indicator")).toHaveAccessibleName("ACP backend is healthy")
    const dialog = await openCreation()
    await waitFor(() =>
      expect(within(dialog).getByRole("button", { name: "Create Session" })).toBeEnabled()
    )
  })

  it.each(["healthy", "degraded"])("preserves the existing %s availability state", async (overall) => {
    healthResponse = async () => jsonResponse({ ...canonicalHealth, overall })
    mountPlayground()
    await waitForHealth()

    expect(screen.getByRole("button", { name: "Create Session" })).toBeInTheDocument()
    expect(screen.getByTestId("acp-health-indicator")).toHaveAccessibleName("ACP backend is healthy")
  })

  it.each(["unknown", "future_status"])("does not admit unrecognized %s health as healthy", async (overall) => {
    healthResponse = async () => jsonResponse({ ...canonicalHealth, overall })
    mountPlayground()
    await waitForHealth()

    expect(screen.queryByRole("button", { name: "Create Session" })).toBeNull()
    expect(screen.getByTestId("acp-health-indicator")).toHaveAccessibleName(
      "ACP backend is not configured or unreachable"
    )
  })

  it("does not promote a malformed successful response to healthy", async () => {
    healthResponse = async () => jsonResponse({ ...canonicalHealth, overall: 1 })
    mountPlayground()
    await waitForHealth()

    expect(screen.queryByRole("button", { name: "Create Session" })).toBeNull()
    expect(screen.getByTestId("acp-health-indicator")).toHaveAccessibleName(
      "ACP backend is not configured or unreachable"
    )
  })

  it("retains unavailable capability recovery", async () => {
    healthResponse = async () => jsonResponse({ ...canonicalHealth, overall: "unavailable" })
    mountPlayground()
    await waitForHealth()

    expect(screen.getByTestId("acp-playground-capability-recovery")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Create Session" })).toBeNull()
  })

  it.each([401, 403, 503])("does not let an HTTP %s response advertise successful health", async (status) => {
    healthResponse = async () => jsonResponse({ ...canonicalHealth, detail: "ACP request denied" }, status)
    mountPlayground()
    await waitForHealth()

    expect(screen.getByTestId("acp-playground-capability-recovery")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Create Session" })).toBeNull()
    expect(screen.getByTestId("acp-health-indicator")).toHaveAccessibleName(
      "ACP backend is not configured or unreachable"
    )
  })

  it("retains unavailable recovery when the health transport fails", async () => {
    healthResponse = async () => { throw new TypeError("ACP connection failed") }
    mountPlayground()
    await waitForHealth()

    expect(screen.getByTestId("acp-playground-capability-recovery")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Create Session" })).toBeNull()
  })

  it("keeps the indicator unresolved until the successful health read completes", async () => {
    let resolveHealth!: (response: Response) => void
    healthResponse = () => new Promise((resolve) => { resolveHealth = resolve })
    mountPlayground()
    await waitFor(() => expect(resolveHealth).toBeTypeOf("function"))
    expect(screen.queryByTestId("acp-health-indicator")).toBeNull()

    await act(async () => resolveHealth(jsonResponse(canonicalHealth)))
    await waitForHealth()
    expect(screen.getByRole("button", { name: "Create Session" })).toBeInTheDocument()
  })

  it("keeps a blocked selected agent refused after canonical health exposes creation", async () => {
    agentResponse.agents[0].is_configured = false
    agentResponse.agents[0].entrypoint = {
      ...readyAgents.agents[0].entrypoint!,
      probe_state: "blocked",
      primary_blocker: "binary_missing",
      blockers: ["binary_missing"],
      status_message: "Install the configured native ACP entrypoint."
    }
    mountPlayground()
    await waitForHealth()
    const dialog = await openCreation()

    expect(within(dialog).getByRole("button", { name: "Create Session" })).toBeDisabled()
    expect(useACPSessionsStore.getState().sessions).toEqual({})
  })

  it("keeps missing and relative working directories from dispatching a session", async () => {
    mountPlayground()
    await waitForHealth()
    const dialog = await openCreation()
    const submit = within(dialog).getByRole("button", { name: "Create Session" })
    await waitFor(() => expect(submit).toBeEnabled())

    await userEvent.click(submit)
    await within(dialog).findByText("Please enter a working directory")
    fireEvent.change(within(dialog).getByLabelText("Working Directory"), {
      target: { value: "relative/workspace" }
    })
    await userEvent.click(submit)
    await within(dialog).findByText("Please enter an absolute path")
    expect(useACPSessionsStore.getState().sessions).toEqual({})
    expect(fetchMock.mock.calls.some(([input]) =>
      String(input).includes("/acp/sessions/new")
    )).toBe(false)
  })
})
