import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { ConfigProvider } from "antd"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ACPChatPanel } from "../ACPChatPanel"
import { ACPPlayground } from "../index"
import { getSessionMessageCount, getSessionTokenUsage } from "../sessionMetrics"
import { useACPSessionsStore } from "@/store/acp-sessions"
import type { ACPUpdate, ACPSessionDetailResponse } from "@/services/acp/types"

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

const answerParts = ['The README’s', ' title is UAT Repository and', ' its exact marker is "ORBIT-', '742 repository', ' fixture."']
const answer = answerParts.join("")
const usage = { prompt_tokens: 201, completion_tokens: 27, total_tokens: 7930 }
const id = "owned-native-session"
const native = (type: string, fields: Record<string, unknown>): ACPUpdate => ({
  type, timestamp: new Date("2026-10-09T12:00:00Z"),
  data: { sessionId: id, update: { sessionUpdate: type, ...fields } }
})
const chunks = () => answerParts.map((text) => native("agent_message_chunk", { content: { type: "text", text } }))
const completed = (): ACPUpdate => ({ type: "prompt_complete", timestamp: new Date(), data: { usage, raw_result: { stopReason: "end_turn" } } })
const detail = (messages: ACPSessionDetailResponse["messages"] = []): ACPSessionDetailResponse => ({
  session_id: id, user_id: 1, agent_type: "opencode", name: "Native session", status: "active",
  created_at: "2026-10-09T12:00:00Z", message_count: messages.length, usage: messages.length ? usage : { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  tags: [], has_websocket: true, runtime_available: true, cwd: "/tmp/owned-project",
  fork_lineage: [], messages
})
const messages = [{ role: "user", content: "Read the README" }, { role: "assistant", content: answer }]
const clients: QueryClient[] = []
const sockets: BoundaryWebSocket[] = []
let scrollDescriptor: PropertyDescriptor | undefined
class BoundaryWebSocket {
  static OPEN = 1
  readyState = 0
  onopen: (() => void) | null = null
  onclose: ((event: { code: number }) => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  constructor(_url: string) { sockets.push(this) }
  close() { this.readyState = 3; this.onclose?.({ code: 1000 }) }
  send(_payload: string) {}
  emit(message: Record<string, unknown>) { this.onmessage?.({ data: JSON.stringify(message) }) }
}
beforeEach(() => {
  vi.clearAllMocks()
  useACPSessionsStore.getState().reset()
  sockets.length = 0
  window.history.pushState({}, "", "/acp-playground")
  scrollDescriptor = Object.getOwnPropertyDescriptor(Element.prototype, "scrollIntoView")
  Object.defineProperty(Element.prototype, "scrollIntoView", { configurable: true, value: vi.fn() })
})
afterEach(() => {
  cleanup()
  clients.splice(0).forEach((client) => client.clear())
  useACPSessionsStore.getState().reset()
  if (scrollDescriptor) Object.defineProperty(Element.prototype, "scrollIntoView", scrollDescriptor)
  else Reflect.deleteProperty(Element.prototype, "scrollIntoView")
  vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.unstubAllEnvs()
})
function owned() {
  const state = useACPSessionsStore.getState()
  const local = state.createSession({ cwd: "/tmp/owned-project" })
  state.replaceSessionId(local, id); state.setActiveSession(id)
  return useACPSessionsStore.getState()
}
function panel(updates: ACPUpdate[]) {
  owned()
  return render(<ConfigProvider theme={{ token: { motion: false } }}>
    <ACPChatPanel state="connected" isConnected updates={updates} connect={vi.fn()} sendPrompt={vi.fn()} cancel={vi.fn()} />
  </ConfigProvider>)
}

describe("native ACP event consumers", () => {
  it("cancelled native completion remains visible without invented answer or usage", () => {
    panel([{ type: "user_text", timestamp: new Date(), data: { text: "Controlled permission request" } },
      { type: "prompt_complete", timestamp: new Date(), data: { raw_result: { stopReason: "cancelled", usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } } } }])
    expect(screen.getByText("Controlled permission request")).toBeVisible()
    expect(screen.getByText("Operation cancelled")).toBeVisible()
    expect(screen.queryByText("Agent")).not.toBeInTheDocument()
    expect(screen.queryByText(/stopReason/)).not.toBeInTheDocument()
  })
  it("cancelled native completion survives canonical history restoration exactly once", () => {
    const state = owned()
    const saved = detail([{ role: "user", content: "Controlled permission request" },
      { role: "assistant", content: null, raw_result: { stopReason: "cancelled", usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } } }])
    state.applySessionDetail(saved); state.applySessionDetail(saved)
    panel(useACPSessionsStore.getState().getSession(id)!.updates)
    expect(screen.getAllByText("Operation cancelled")).toHaveLength(1)
    expect(screen.getByText("Controlled permission request")).toBeVisible()
    expect(screen.queryByText(/stopReason/)).not.toBeInTheDocument()
  })
  it("cancelled native completion does not duplicate explicit Stop feedback", () => {
    panel([{ type: "cancelled", timestamp: new Date(), data: { message: "Operation cancelled" } },
      native("usage_update", { used: 0, size: 200000 }),
      { type: "prompt_complete", timestamp: new Date(), data: { raw_result: { stopReason: "cancelled" } } }])
    expect(screen.getAllByText("Operation cancelled")).toHaveLength(1)
  })

  it("cancelled native completion does not duplicate a later Stop acknowledgement", () => {
    panel([
      { type: "prompt_complete", timestamp: new Date(), data: { raw_result: { stopReason: "cancelled" } } },
      { type: "cancelled", timestamp: new Date(), data: { message: "Operation cancelled" } }
    ])
    expect(screen.getAllByText("Operation cancelled")).toHaveLength(1)
  })
  it("cancelled native completion separates outputless turns at actual prompt starts", () => {
    panel([
      { type: "prompt_start", timestamp: new Date(), data: {} },
      { type: "prompt_complete", timestamp: new Date(), data: { raw_result: { stopReason: "cancelled" } } },
      { type: "cancelled", timestamp: new Date(), data: {} },
      { type: "prompt_start", timestamp: new Date(), data: {} },
      { type: "prompt_complete", timestamp: new Date(), data: { raw_result: { stopReason: "cancelled" } } }
    ])
    expect(screen.getAllByText("Operation cancelled")).toHaveLength(2)
    expect(screen.queryByText("{}")).not.toBeInTheDocument()
  })

  it("cancelled native completion preserves separate turns and raw results", () => {
    const events: ACPUpdate[] = [
      { type: "user_text", timestamp: new Date(), data: { text: "First controlled turn" } },
      { type: "cancelled", timestamp: new Date(), data: {} },
      { type: "prompt_complete", timestamp: new Date(), data: { raw_result: { stopReason: "cancelled" } } },
      { type: "user_text", timestamp: new Date(), data: { text: "Second controlled turn" } },
      { type: "prompt_complete", timestamp: new Date(), data: { raw_result: { stopReason: "cancelled" } } }
    ]
    const before = JSON.stringify(events)
    panel(events)
    expect(screen.getAllByText("Operation cancelled")).toHaveLength(2)
    expect(JSON.stringify(events)).toBe(before)
  })
  it("cancelled native completion hydrates partial text and separate cancellations without stale duplicates", () => {
    const state = owned()
    state.addUpdate(id, { type: "cancelled", data: {} })
    state.addUpdate(id, { type: "prompt_complete", data: { raw_result: { stopReason: "cancelled" } } })
    const saved = detail([
      { role: "user", content: "First controlled turn" },
      { role: "assistant", content: "Actual partial answer", raw_result: { stopReason: "cancelled" } },
      { role: "user", content: "Second controlled turn" },
      { role: "assistant", content: null, raw_result: { stopReason: "cancelled" } }
    ])
    state.applySessionDetail(saved); state.applySessionDetail(saved)
    panel(useACPSessionsStore.getState().getSession(id)!.updates)
    expect(screen.getByText("Actual partial answer")).toBeVisible()
    expect(screen.getAllByText("Operation cancelled")).toHaveLength(2)
    expect(screen.queryByText(/stopReason/)).not.toBeInTheDocument()
  })

  it("renders the five adjacent native fragments as one exact assistant answer", () => {
    panel(chunks())
    expect(screen.getAllByText("Agent")).toHaveLength(1)
    expect(screen.getByText(answer)).toBeVisible()
  })
  it("does not join answer streams across explicit completion boundaries", () => {
    panel([...chunks(), completed(), native("agent_message_chunk", { content: { type: "text", text: "Second turn" } })])
    expect(screen.getAllByText("Agent")).toHaveLength(2)
    expect(screen.getByText(answer)).toBeVisible()
    expect(screen.getByText("Second turn")).toBeVisible()
    expect(screen.queryByText(/stopReason/)).not.toBeInTheDocument()
  })
  it("merges native tool identity, status, input and output by the real toolCallId", () => {
    panel([
      native("tool_call", { toolCallId: "read-1", title: "Read README", status: "pending" }),
      native("tool_call_update", { toolCallId: "read-1", title: "read", status: "in_progress", rawInput: { filePath: "/tmp/owned-project/README.md" } }),
      native("tool_call_update", { toolCallId: "read-1", title: "README.md", status: "completed",
        content: [{ type: "content", content: { type: "text", text: "# UAT Repository\nORBIT-742 repository fixture." } }],
        rawOutput: { output: "ORBIT-742 repository fixture." } })
    ])
    expect(screen.getAllByText(/Tool:/)).toHaveLength(1)
    expect(screen.getByText("Tool: README.md")).toBeVisible()
    expect(screen.getByText("completed")).toBeVisible()
    expect(screen.getByText("/tmp/owned-project/README.md")).toBeVisible()
    expect(screen.getByText("ORBIT-742 repository fixture.")).toBeVisible()
    expect(screen.queryByText(/unknown/)).not.toBeInTheDocument()
    expect(screen.queryByText(/sessionUpdate/)).not.toBeInTheDocument()
  })
  it("renders native thought text separately from the answer and context budget without raw JSON", () => {
    panel([
      native("agent_thought_chunk", { messageId: "thought-1", content: { type: "text", text: "Reading the fixture" } }),
      native("usage_update", { used: 7881, size: 200000, cost: { amount: 0, currency: "USD" } }),
      ...chunks()
    ])
    expect(screen.getByText(/Reading the fixture/)).toBeVisible()
    expect(screen.getByText(/7881.*200000/)).toBeVisible()
    expect(screen.getAllByText("Agent")).toHaveLength(1)
    expect(screen.queryByText(/sessionUpdate/)).not.toBeInTheDocument()
    expect(getSessionTokenUsage({ updates: [native("usage_update", { used: 7881, size: 200000 })], usage: null })).toBeNull()
  })
  it("keeps legacy messages and malformed native diagnostics intact", () => {
    panel([{ type: "assistant_text", data: { text: "Legacy final" }, timestamp: new Date() },
      native("agent_message_chunk", { content: { type: "image", data: "opaque" } })])
    expect(screen.getByText("Legacy final")).toBeVisible()
    expect(screen.getByText(/opaque/)).toBeVisible()
  })
  it("counts one streamed message without counting thoughts, tools or context as messages", () => {
    const state = owned()
    for (const update of [...chunks(), native("agent_thought_chunk", { content: { type: "text", text: "thinking" } }), native("usage_update", { used: 0, size: 200000 })])
      state.addUpdate(id, update)
    const session = useACPSessionsStore.getState().getSession(id)!
    expect(session.messageCount).toBe(1)
    expect(getSessionMessageCount(session)).toBe(1)
    expect(getSessionTokenUsage(session)).toBeNull()
  })
  it("hydrates canonical completed owner history instead of persisted local answer fragments", () => {
    const state = owned()
    chunks().forEach((update) => state.addUpdate(id, update))
    state.applySessionDetail(detail(messages))
    const session = useACPSessionsStore.getState().getSession(id)!
    panel(session.updates)
    expect(screen.getAllByText("Agent")).toHaveLength(1)
    expect(screen.getByText("Read the README")).toBeVisible()
    expect(screen.getByText(answer)).toBeVisible()
    expect(getSessionMessageCount(session)).toBe(2)
  })
  it.each(["running", "waiting_permission"] as const)("does not erase an active %s turn with a late detail response", (stateName) => {
    const state = owned()
    state.updateSessionState(id, stateName)
    state.addUpdate(id, chunks()[0])
    const before = useACPSessionsStore.getState().getSession(id)!.updates
    state.applySessionDetail(detail(messages))
    expect(useACPSessionsStore.getState().getSession(id)!.updates).toEqual(before)
    expect(useACPSessionsStore.getState().getSession(id)!.state).toBe(stateName)
  })
  it("retains tool event details when canonical completed text replaces chunks", () => {
    const state = owned()
    chunks().forEach((update) => state.addUpdate(id, update))
    state.addUpdate(id, native("tool_call_update", { toolCallId: "read-1", title: "Read README", status: "completed", rawOutput: { output: "Exact tool output" } }))
    state.applySessionDetail(detail(messages))
    panel(useACPSessionsStore.getState().getSession(id)!.updates)
    expect(screen.getByText("Exact tool output")).toBeVisible()
    expect(screen.getByText(answer)).toBeVisible()
    expect(screen.getAllByText("Agent")).toHaveLength(1)
  })
  it("does not mutate raw tool payloads or merge the same tool id across turns", () => {
    const events = [
      native("tool_call_update", { toolCallId: "same-id", title: "First tool", status: "completed" }),
      completed(),
      native("tool_call_update", { toolCallId: "same-id", title: "Second tool", status: "pending" })
    ]
    const before = JSON.stringify(events)
    panel(events)
    expect(screen.getByText("Tool: First tool")).toBeVisible()
    expect(screen.getByText("Tool: Second tool")).toBeVisible()
    expect(JSON.stringify(events)).toBe(before)
  })
  it("does not invent tool arguments or tokens from malformed native values", () => {
    panel([native("tool_call_update", { toolCallId: "read-1", title: "Read", rawInput: ["invalid"] })])
    expect(screen.getByText("Tool: Read")).toBeVisible()
    expect(screen.getByText("{}")).toBeVisible()
    expect(getSessionTokenUsage({ updates: [native("usage_update", { used: "7930", size: 200000 })], usage: null })).toBeNull()
  })
  it("does not duplicate nontext canonical diagnostics on repeated detail hydration", () => {
    const state = owned()
    const saved = detail([{ role: "assistant", content: null, raw_result: { kind: "nontext" } }])
    state.applySessionDetail(saved); state.applySessionDetail(saved)
    expect(useACPSessionsStore.getState().getSession(id)!.updates).toHaveLength(1)
  })
  it("keeps reused tool ids in separate completed turns after canonical history hydration", () => {
    const state = owned()
    state.addUpdate(id, native("tool_call_update", { toolCallId: "same-id", title: "First tool", status: "completed" }))
    state.addUpdate(id, { type: "prompt_complete", data: {} })
    state.addUpdate(id, native("tool_call_update", { toolCallId: "same-id", title: "Second tool", status: "completed" }))
    state.applySessionDetail(detail(messages))
    panel(useACPSessionsStore.getState().getSession(id)!.updates)
    expect(screen.getByText("Tool: First tool")).toBeVisible()
    expect(screen.getByText("Tool: Second tool")).toBeVisible()
  })
  it("cancelled native completion keeps retained tools in separate restored turns", () => {
    const state = owned()
    state.addUpdate(id, native("tool_call_update", { toolCallId: "same-id", title: "First cancelled tool", status: "completed" }))
    state.addUpdate(id, { type: "prompt_complete", data: { raw_result: { stopReason: "cancelled" } } })
    state.addUpdate(id, native("tool_call_update", { toolCallId: "same-id", title: "Second tool", status: "completed" }))
    const before = JSON.stringify(useACPSessionsStore.getState().getSession(id)!.updates)
    state.applySessionDetail(detail([
      { role: "user", content: "First controlled turn" },
      { role: "assistant", content: null, raw_result: { stopReason: "cancelled" } }
    ]))
    panel(useACPSessionsStore.getState().getSession(id)!.updates)
    expect(screen.getByText("Tool: First cancelled tool")).toBeVisible()
    expect(screen.getByText("Tool: Second tool")).toBeVisible()
    expect(screen.getAllByText("Operation cancelled")).toHaveLength(1)
    expect(before).toContain("First cancelled tool")
  })

  it("actual parent prompt sends establish a non-rendering turn boundary", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    boundaries.useStorage.mockImplementation((_key: string, fallback: unknown) => [fallback, vi.fn(), { isLoading: false }])
    boundaries.getConfig.mockResolvedValue({ serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" })
    window.history.pushState({}, "", "/acp-playground?session=" + id)
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const path = new URL(String(input), window.location.origin).pathname
      const body = path.endsWith("/acp/health") ? { overall: "ok", runner: { status: "ok" }, agents: [] }
        : path.endsWith("/acp/sessions") ? { sessions: [detail()], total: 1 } : detail()
      return new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } })
    })
    vi.stubGlobal("fetch", fetchMock); vi.stubGlobal("WebSocket", BoundaryWebSocket)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } }); clients.push(queryClient)
    render(<ConfigProvider theme={{ token: { motion: false } }}><QueryClientProvider client={queryClient}><ACPPlayground /></QueryClientProvider></ConfigProvider>)
    await waitFor(() => expect(sockets).toHaveLength(1))
    act(() => { sockets[0].readyState = BoundaryWebSocket.OPEN; sockets[0].onopen?.(); sockets[0].emit({ type: "connected", session_id: id }) })
    const send = vi.spyOn(sockets[0], "send")
    fireEvent.change(screen.getByPlaceholderText("Type your message..."), { target: { value: "Controlled new turn" } })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    expect(send).toHaveBeenCalledOnce()
    expect(JSON.parse(send.mock.calls[0][0]).type).toBe("prompt")
    expect(useACPSessionsStore.getState().getSession(id)!.updates.filter((update) => update.type === "prompt_start")).toHaveLength(1)
    expect(getSessionMessageCount(useACPSessionsStore.getState().getSession(id)!)).toBe(0)
    expect(screen.queryByText("{}")).not.toBeInTheDocument()
  })

  it("does not recount legacy update usage on actual parent completion", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    boundaries.useStorage.mockImplementation((_key: string, fallback: unknown) => [fallback, vi.fn(), { isLoading: false }])
    boundaries.getConfig.mockResolvedValue({ serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" })
    window.history.pushState({}, "", "/acp-playground?session=" + id)
    let finished = false, detailReads = 0
    const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
      const path = new URL(String(input), window.location.origin).pathname
      let body: unknown
      if (path.endsWith("/acp/health")) body = { overall: "ok", runner: { status: "ok" }, agents: [] }
      else if (path.endsWith("/acp/sessions")) body = { sessions: [detail()], total: 1 }
      else if (path.endsWith("/detail")) { detailReads++; body = detail(finished ? messages : []) }
      else if (path.endsWith("/usage")) body = detail(finished ? messages : [])
      else throw new Error("Unexpected request " + path)
      return new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } })
    })
    vi.stubGlobal("fetch", fetchMock); vi.stubGlobal("WebSocket", BoundaryWebSocket)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } }); clients.push(queryClient)
    render(<ConfigProvider theme={{ token: { motion: false } }}><QueryClientProvider client={queryClient}><ACPPlayground /></QueryClientProvider></ConfigProvider>)
    await waitFor(() => expect(sockets).toHaveLength(1))
    await waitFor(() => expect(detailReads).toBe(1))
    act(() => {
      const socket = sockets[0]; socket.readyState = BoundaryWebSocket.OPEN; socket.onopen?.()
      socket.emit({ type: "connected", session_id: id })
      socket.emit({ type: "update", session_id: id, update_type: "assistant_text", data: { text: answer, usage } })
    })
    expect(useACPSessionsStore.getState().getSession(id)!.usage).toEqual(usage)
    finished = true
    act(() => sockets[0].emit({ type: "prompt_complete", session_id: id, usage, raw_result: { stopReason: "end_turn" } }))
    await waitFor(() => expect(detailReads).toBe(2))
    await waitFor(() => expect(useACPSessionsStore.getState().getSession(id)!.usage).toEqual(usage))
    expect(getSessionTokenUsage(useACPSessionsStore.getState().getSession(id)!)).toBe(7930)
    expect(screen.getAllByText("Agent")).toHaveLength(1)
    expect(screen.getByText(answer)).toBeVisible()
    expect(fetchMock.mock.calls.every(([, init]) => !init || (init.method ?? "GET") === "GET")).toBe(true)
  })
  it("refreshes canonical detail on actual parent prompt completion and preserves reported usage", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    boundaries.useStorage.mockImplementation((_key: string, fallback: unknown) => [fallback, vi.fn(), { isLoading: false }])
    boundaries.getConfig.mockResolvedValue({ serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" })
    window.history.pushState({}, "", "/acp-playground?session=" + id)
    let finished = false, detailReads = 0
    const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
      const path = new URL(String(input), window.location.origin).pathname
      let body: unknown
      if (path.endsWith("/acp/health")) body = { overall: "ok", runner: { status: "ok" }, agents: [] }
      else if (path.endsWith("/acp/sessions")) body = { sessions: [detail()], total: 1 }
      else if (path.endsWith("/detail")) { detailReads++; body = detail(finished ? messages : []) }
      else if (path.endsWith("/usage")) body = detail(finished ? messages : [])
      else throw new Error("Unexpected request " + path)
      return new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } })
    })
    vi.stubGlobal("fetch", fetchMock); vi.stubGlobal("WebSocket", BoundaryWebSocket)
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } }); clients.push(queryClient)
    render(<ConfigProvider theme={{ token: { motion: false } }}><QueryClientProvider client={queryClient}><ACPPlayground /></QueryClientProvider></ConfigProvider>)
    await waitFor(() => expect(sockets).toHaveLength(1))
    await waitFor(() => expect(detailReads).toBe(1))
    act(() => {
      const socket = sockets[0]; socket.readyState = BoundaryWebSocket.OPEN; socket.onopen?.()
      socket.emit({ type: "connected", session_id: id })
      for (const update of chunks()) socket.emit({ type: "update", session_id: id, update_type: update.type, data: update.data })
    })
    finished = true
    act(() => sockets[0].emit({ type: "prompt_complete", session_id: id, usage, raw_result: { stopReason: "end_turn" } }))
    await waitFor(() => expect(detailReads).toBe(2))
    expect(await screen.findByText("Read the README")).toBeVisible()
    expect(screen.getAllByText("Agent")).toHaveLength(1)
    expect(screen.getByText(answer)).toBeVisible()
    expect(useACPSessionsStore.getState().getSession(id)!.usage).toEqual(usage)
    expect(fetchMock.mock.calls.every(([, init]) => !init || (init.method ?? "GET") === "GET")).toBe(true)
  })
  it("keeps cumulative legacy message count beyond the retained event cap", () => {
    const state = owned()
    for (let index = 0; index < 1000; index++)
      state.addUpdate(id, { type: "assistant_text", data: { text: "Message " + index } })
    const before = useACPSessionsStore.getState().getSession(id)!
    expect(before.updates).toHaveLength(1000)
    expect(before.messageCount).toBe(1000)
    state.addUpdate(id, { type: "assistant_text", data: { text: "Message 1000" } })
    const after = useACPSessionsStore.getState().getSession(id)!
    expect(after.updates).toHaveLength(1000)
    expect(after.updates[0].data.text).toBe("Message 1")
    expect(after.updates[999].data.text).toBe("Message 1000")
    expect(after.messageCount).toBe(1001)
    expect(getSessionMessageCount(after)).toBe(1001)
  })
  it("does not recount a native chunk continuation at the retained event cap", () => {
    const state = owned()
    for (let index = 0; index < 999; index++)
      state.addUpdate(id, { type: "assistant_text", data: { text: "Message " + index } })
    state.addUpdate(id, native("agent_message_chunk", { messageId: "at-cap", content: { type: "text", text: "Native " } }))
    const before = useACPSessionsStore.getState().getSession(id)!
    expect(before.updates).toHaveLength(1000)
    expect(before.messageCount).toBe(1000)
    state.addUpdate(id, native("agent_message_chunk", { messageId: "at-cap", content: { type: "text", text: "continuation" } }))
    const after = useACPSessionsStore.getState().getSession(id)!
    expect(after.updates).toHaveLength(1000)
    expect(after.updates[0].data.text).toBe("Message 1")
    expect(after.messageCount).toBe(1000)
    expect(getSessionMessageCount(after)).toBe(1000)
    panel(after.updates)
    expect(screen.getByText("Native continuation")).toBeVisible()
  })
})
