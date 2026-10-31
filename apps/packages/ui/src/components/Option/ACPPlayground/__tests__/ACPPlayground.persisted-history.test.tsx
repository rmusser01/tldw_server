import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { ConfigProvider } from "antd"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ACPPlayground } from "../index"
import { ACPChatPanel } from "../ACPChatPanel"
import { useACPSessionsStore } from "@/store/acp-sessions"
import type { ACPSessionDetailResponse } from "@/services/acp/types"

const boundaries = vi.hoisted(() => ({ getConfig: vi.fn(), useStorage: vi.fn() }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({
  t: (key: string, fallback?: string | { defaultValue?: string }) =>
    typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
}) }))
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: (...args: unknown[]) => boundaries.useStorage(...args) }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { getConfig: (...args: unknown[]) => boundaries.getConfig(...args) } }))
vi.mock("@/hooks/useMediaQuery", () => ({ useMobile: () => false, useDesktop: () => true }))
vi.mock("../ACPToolsPanel", () => ({ ACPToolsPanel: () => null }))
vi.mock("../ACPWorkspacePanel", () => ({ ACPWorkspacePanel: () => null }))
vi.mock("../ACPPermissionModal", () => ({ ACPPermissionModal: () => null }))

const saved = {
  session_id: "saved-owned-session", user_id: 1, agent_type: "opencode", name: "Saved session",
  status: "active", created_at: "2026-10-09T04:00:00Z", message_count: 2,
  usage: { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 }, tags: [],
  has_websocket: false, runtime_available: false, cwd: "/tmp/owned-project",
  messages: [{ role: "user", content: "Original saved question" }, { role: "assistant", content: "Original saved answer" }],
  fork_lineage: []
} satisfies ACPSessionDetailResponse & { runtime_available: boolean }
const clients: QueryClient[] = []
const sockets: string[] = []
let fetchMock: ReturnType<typeof vi.fn>
class BoundaryWebSocket {
  static OPEN = 1
  readyState = 0
  onopen: (() => void) | null = null
  onclose: ((event: { code: number }) => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  constructor(url: string) { sockets.push(url) }
  close() { this.readyState = 3; this.onclose?.({ code: 1000 }) }
  send() { throw new Error("Unavailable persisted session must never send") }
}
const json = (body: unknown) => new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } })
beforeEach(() => {
  vi.clearAllMocks()
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
  useACPSessionsStore.getState().reset()
  sockets.length = 0
  window.history.pushState({}, "", "/acp-playground?session=saved-owned-session")
  boundaries.useStorage.mockImplementation((_key: string, fallback: unknown) => [fallback, vi.fn(), { isLoading: false }])
  boundaries.getConfig.mockResolvedValue({ serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" })
  fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const path = new URL(String(input), window.location.origin).pathname
    if (path.endsWith("/acp/health")) return json({ overall: "ok", runner: { status: "ok" }, agents: [] })
    if (path.endsWith("/acp/sessions")) return json({ sessions: [saved], total: 1 })
    if (path.endsWith("/detail") || path.endsWith("/usage")) return json(saved)
    throw new Error(`Unexpected stored-history request: ${path}`)
  })
  vi.stubGlobal("fetch", fetchMock)
  vi.stubGlobal("WebSocket", BoundaryWebSocket)
  Element.prototype.scrollIntoView = vi.fn()
})
afterEach(() => {
  cleanup()
  clients.splice(0).forEach((client) => client.clear())
  useACPSessionsStore.getState().reset()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe("ACP persisted history runtime availability", () => {
  it("hydrates owner detail and usage without connecting the lost runtime", async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    clients.push(queryClient)
    render(<ConfigProvider theme={{ token: { motion: false } }}><QueryClientProvider client={queryClient}>
      <ACPPlayground />
    </QueryClientProvider></ConfigProvider>)
    await waitFor(() => expect(useACPSessionsStore.getState().getSession(saved.session_id)?.usage?.total_tokens).toBe(8))
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/detail"))).toBe(true))
    expect(await screen.findByText("Session runtime unavailable. Stored history remains available; create a new session to continue.")).toBeVisible()
    expect(screen.queryByRole("button", { name: "Reconnect" })).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled()
    expect(sockets).toEqual([])
    expect(fetchMock.mock.calls.every(([, init]) => (init?.method ?? "GET") === "GET")).toBe(true)
  })

  it("does not retain stale connected or pending-permission state from list or detail", () => {
    const state = useACPSessionsStore.getState()
    const local = state.createSession({ cwd: "/tmp/owned-project" })
    state.replaceSessionId(local, saved.session_id)
    state.updateSessionState(saved.session_id, "waiting_permission")
    state.upsertSessionsFromServerList([saved])
    expect(useACPSessionsStore.getState().getSession(saved.session_id)?.state).toBe("disconnected")
    state.updateSessionState(saved.session_id, "connected")
    state.applySessionDetail(saved)
    expect(useACPSessionsStore.getState().getSession(saved.session_id)?.state).toBe("disconnected")
  })

  it("unavailable runtime cannot reconnect, send or stop even with stale live props", () => {
    const state = useACPSessionsStore.getState()
    state.upsertSessionsFromServerList([saved])
    state.setActiveSession(saved.session_id)
    const sendPrompt = vi.fn(), cancel = vi.fn(), connect = vi.fn()
    render(<ACPChatPanel state="waiting_permission" isConnected={true} updates={[]} sendPrompt={sendPrompt} cancel={cancel} connect={connect} />)
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument()
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Do not send" } })
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" })
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled()
    expect(screen.queryByRole("button", { name: "Reconnect" })).not.toBeInTheDocument()
    expect(sendPrompt).not.toHaveBeenCalled()
    expect(cancel).not.toHaveBeenCalled()
    expect(connect).not.toHaveBeenCalled()
  })

  it.each([true, undefined])("retains live or legacy reconnect behavior for availability %s", (available) => {
    const detail = { ...saved, runtime_available: available }
    const state = useACPSessionsStore.getState()
    state.upsertSessionsFromServerList([detail])
    state.setActiveSession(saved.session_id)
    const connect = vi.fn()
    render(<ACPChatPanel state="disconnected" isConnected={false} updates={[]} sendPrompt={vi.fn()} cancel={vi.fn()} connect={connect} />)
    fireEvent.click(screen.getByRole("button", { name: "Reconnect" }))
    expect(connect).toHaveBeenCalledOnce()
  })
})
