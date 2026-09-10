import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import {
  act,
  cleanup,
  fireEvent,
  render as rtlRender,
  screen,
  waitFor
} from "@testing-library/react"
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi
} from "vitest"
import {
  BuddyShellHost,
  BuddyShellRenderContextProvider,
  useBuddyShellRenderContext
} from "@/components/Common/PersonaBuddy"
import { usePersonaVisualRuntimeStore } from "@/store/persona-visual-runtime"

const mocks = vi.hoisted(() => ({
  voiceState: "idle",
  toolName: "",
  isOnline: true,
  uxState: "connected_ok" as
    | "connected_ok"
    | "configuring_url"
    | "configuring_auth"
    | "error_auth"
    | "error_unreachable"
    | "unconfigured",
  hasCompletedFirstRun: true,
  capabilitiesState: {
    capabilities: { hasPersona: true, hasPersonalization: true },
    loading: false
  } as {
    capabilities: {
      hasPersona: boolean
      hasPersonalization?: boolean
      hasPersonaLiveControl?: boolean
    } | null
    loading: boolean
  },
  navigate: vi.fn(),
  location: {
    pathname: "/persona",
    search: "",
    hash: "",
    state: null,
    key: "persona-route"
  },
  useBlocker: vi.fn(),
  blocker: {
    state: "unblocked" as "unblocked" | "blocked" | "proceeding",
    proceed: vi.fn(),
    reset: vi.fn()
  },
  getConfig: vi.fn(),
  fetchWithAuth: vi.fn(),
  buildPersonaWebSocketUrl: vi.fn(() => ({
    url: "ws://persona.test/api/v1/persona/stream",
    protocols: ["bearer", "test"]
  })),
  fetchCompanionConversationPrompts: vi.fn(),
  buddyShellContextSnapshots: [] as Array<Record<string, unknown> | null>
}))

vi.mock("@/hooks/usePersonaLiveVoiceController", async () => {
  const actual = await vi.importActual<
    typeof import("@/hooks/usePersonaLiveVoiceController")
  >("@/hooks/usePersonaLiveVoiceController")
  return {
    ...actual,
    usePersonaLiveVoiceController: (
      ...args: Parameters<typeof actual.usePersonaLiveVoiceController>
    ) => ({
      ...actual.usePersonaLiveVoiceController(...args),
      state: mocks.voiceState,
      isListening: mocks.voiceState === "listening",
      activeToolName: mocks.toolName,
      activeToolStatus: mocks.toolName ? "Running search" : ""
    })
  }
})

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, vi.fn(), { isLoading: false }]
}))
vi.mock("@/hooks/useSetting", () => ({
  useSetting: (setting: { defaultValue: unknown }) => [
    setting.defaultValue,
    vi.fn(),
    { isLoading: false }
  ]
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => mocks.isOnline
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState: mocks.uxState,
    hasCompletedFirstRun: mocks.hasCompletedFirstRun
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => mocks.capabilitiesState
}))

vi.mock("react-router-dom", async () => {
  const actual =
    await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return {
    ...actual,
    UNSAFE_DataRouterContext: React.createContext({ router: {} }),
    useNavigate: () => mocks.navigate,
    useLocation: () => mocks.location,
    useBlocker: (...args: unknown[]) =>
      (mocks.useBlocker as (...args: unknown[]) => unknown)(...args)
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) =>
      (mocks.getConfig as (...args: unknown[]) => unknown)(...args),
    fetchWithAuth: (...args: unknown[]) =>
      (mocks.fetchWithAuth as (...args: unknown[]) => unknown)(...args)
  }
}))

vi.mock("@/services/persona-stream", () => ({
  buildPersonaWebSocketUrl: (...args: unknown[]) =>
    (mocks.buildPersonaWebSocketUrl as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/companion", () => ({
  isCompanionConsentRequiredResponse: (
    response:
      | {
          status?: number
          error?: string | null
        }
      | null
      | undefined
  ) =>
    response?.status === 409 &&
    String(response?.error || "").includes(
      "Enable personalization before using companion."
    ),
  fetchCompanionConversationPrompts: (...args: unknown[]) =>
    (
      mocks.fetchCompanionConversationPrompts as (...args: unknown[]) => unknown
    )(...args)
}))

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  default: ({
    title,
    description,
    primaryActionLabel,
    onPrimaryAction
  }: {
    title: string
    description?: string
    primaryActionLabel?: string
    onPrimaryAction?: () => void
  }) => (
    <div data-testid="feature-empty-state">
      <div>{title}</div>
      {description ? <div>{description}</div> : null}
      {primaryActionLabel ? (
        <button type="button" onClick={onPrimaryAction}>
          {primaryActionLabel}
        </button>
      ) : null}
    </div>
  )
}))

vi.mock("@/components/Option/MCPHub", () => ({
  PersonaPolicySummary: ({ personaId }: { personaId?: string | null }) => (
    <div data-testid="persona-policy-summary">{personaId || "none"}</div>
  )
}))

vi.mock("~/components/Sidepanel/Chat/SidepanelHeaderSimple", () => ({
  SidepanelHeaderSimple: ({ activeTitle }: { activeTitle?: string }) => (
    <div data-testid="sidepanel-header">{activeTitle || "header"}</div>
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string")
        return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue)
        return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  const Input = {
    ...actual.Input,
    TextArea: ({
      autoSize: _autoSize,
      onPressEnter,
      onKeyDown,
      value,
      onChange,
      ...rest
    }: React.ComponentProps<"textarea"> & {
      autoSize?: unknown
      onPressEnter?: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void
    }) => (
      <textarea
        {...rest}
        value={value ?? ""}
        onChange={(event) => onChange?.(event)}
        onKeyDown={(event) => {
          onKeyDown?.(event)
          if (event.key === "Enter") {
            onPressEnter?.(event)
          }
        }}
      />
    )
  }
  return {
    ...actual,
    Input
  }
})

import SidepanelPersona from "../sidepanel-persona"

// Real route, context, host and live-control lifecycle; transport and voice
// device signals are controlled. This does not claim physical voice qualification.
vi.setConfig({ testTimeout: 30_000 })

const BuddyShellContextProbe = () => {
  const context = useBuddyShellRenderContext()

  React.useEffect(() => {
    mocks.buddyShellContextSnapshots.push(
      context ? JSON.parse(JSON.stringify(context)) : null
    )
  }, [context])

  return <pre data-testid="buddy-shell-context">{JSON.stringify(context)}</pre>
}

const queryClients: QueryClient[] = []
const portalRoots: HTMLElement[] = []

const render = (ui: React.ReactNode) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  queryClients.push(queryClient)
  const wrappedUi = (child: React.ReactNode) => (
    <QueryClientProvider client={queryClient}>
      <BuddyShellRenderContextProvider>
        {child}
        <BuddyShellContextProbe />
        <BuddyShellHost root="sidepanel" />
      </BuddyShellRenderContextProvider>
    </QueryClientProvider>
  )

  const view = rtlRender(wrappedUi(ui))
  return {
    ...view,
    rerender: (nextUi: React.ReactNode) => view.rerender(wrappedUi(nextUi))
  }
}

class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  readyState = MockWebSocket.CONNECTING
  static instances: MockWebSocket[] = []

  url: string
  binaryType = "blob"
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  send = vi.fn<(payload: string) => void>()
  close = vi.fn<() => void>()

  constructor(
    url: string,
    readonly protocols?: string | string[]
  ) {
    this.url = new URL(url).toString()
    MockWebSocket.instances.push(this)
    this.send.mockImplementation(() => {
      if (this.readyState !== MockWebSocket.OPEN)
        throw new Error("WebSocket is not open")
    })
    this.close.mockImplementation(() => {
      this.readyState = MockWebSocket.CLOSED
      this.onclose?.({} as CloseEvent)
    })
  }

  emitOpen() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.(new Event("open"))
  }

  emitMessage(data: string | ArrayBuffer) {
    this.onmessage?.({ data } as MessageEvent)
  }
}

describe("SidepanelPersona", () => {
  const originalWebSocket = globalThis.WebSocket
  const originalResizeObserver = globalThis.ResizeObserver

  beforeAll(() => {
    globalThis.WebSocket = MockWebSocket as unknown as typeof WebSocket
    class MockResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    globalThis.ResizeObserver =
      MockResizeObserver as unknown as typeof ResizeObserver
  })

  afterEach(() => {
    cleanup()
    for (const client of queryClients.splice(0)) client.clear()
    for (const portal of portalRoots.splice(0)) portal.remove()
  })

  afterAll(() => {
    globalThis.WebSocket = originalWebSocket
    if (originalResizeObserver) {
      globalThis.ResizeObserver = originalResizeObserver
    } else {
      delete (globalThis as { ResizeObserver?: typeof ResizeObserver })
        .ResizeObserver
    }
  })

  beforeEach(() => {
    MockWebSocket.instances = []
    window.localStorage.clear()
    mocks.isOnline = true
    mocks.uxState = "connected_ok"
    mocks.hasCompletedFirstRun = true
    mocks.capabilitiesState.capabilities = {
      hasPersona: true,
      hasPersonalization: true
    }
    mocks.capabilitiesState.loading = false
    mocks.navigate.mockReset()
    mocks.location.pathname = "/persona"
    mocks.location.search = ""
    mocks.location.hash = ""
    mocks.location.state = null
    mocks.location.key = "persona-route"
    mocks.useBlocker.mockReset()
    mocks.blocker.state = "unblocked"
    mocks.blocker.proceed.mockReset()
    mocks.blocker.reset.mockReset()
    mocks.useBlocker.mockImplementation(() => mocks.blocker)
    mocks.getConfig.mockReset()
    mocks.fetchWithAuth.mockReset()
    mocks.buildPersonaWebSocketUrl.mockReset()
    mocks.fetchCompanionConversationPrompts.mockReset()
    mocks.buddyShellContextSnapshots = []
    usePersonaVisualRuntimeStore.setState({
      override: null,
      runtimeDiagnostics: null
    })
    mocks.buildPersonaWebSocketUrl.mockReturnValue({
      url: "ws://persona.test/api/v1/persona/stream",
      protocols: ["bearer", "test"]
    })
    mocks.fetchCompanionConversationPrompts.mockResolvedValue({
      prompt_source_kind: "reflection",
      prompt_source_id: "reflection-1",
      prompts: [
        {
          prompt_id: "prompt-1",
          label: "Next concrete step",
          prompt_text: "What is the next concrete step for project alpha?",
          prompt_type: "clarify_priority",
          source_reflection_id: "reflection-1",
          source_evidence_ids: ["activity-1"]
        }
      ]
    })
  })

  it("TASK13211 integrated route retains bounded Buddy loads during live updates", async () => {
    const portal = document.createElement("div")
    portal.id = "tldw-portal-root"
    document.body.appendChild(portal)
    portalRoots.push(portal)
    const personaId = "research_assistant"
    const buddy = {
      has_buddy: true,
      persona_name: "Research Buddy",
      role_summary: "Research",
      visual: { species_id: "owl", silhouette_id: "perch", palette_id: "dawn" }
    }
    const pack = {
      id: "pack-1",
      persona_id: personaId,
      title: "Lifecycle buddy",
      renderer_type: "sprite_frames",
      status: "active",
      manifest: {
        manifest_version: 1,
        renderer_type: "sprite_frames",
        states: {
          idle: { animation_id: "idle" },
          thinking: { animation_id: "idle" },
          tool_running: { animation_id: "idle" },
          error: { animation_id: "idle" }
        },
        animations: {
          idle: { frames: [{ asset_id: "frame-1", duration_ms: 100 }] }
        }
      },
      assets_by_id: {
        "frame-1": {
          id: "frame-1",
          url: "/assets/lifecycle.png",
          mime_type: "image/png",
          asset_role: "frame",
          width: 24,
          height: 24
        }
      }
    }
    const calls: string[] = []
    mocks.location.search = "?persona_id=research_assistant&tab=live"
    mocks.capabilitiesState.capabilities = {
      hasPersona: true,
      hasPersonalization: true,
      hasPersonaLiveControl: true
    }
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test"
    })
    mocks.fetchWithAuth.mockImplementation(
      async (path: string, options: { method?: string; body?: unknown }) => {
        calls.push(path)
        let data: unknown = []
        if (path.endsWith("/visual-packs"))
          data = {
            packs: [{ ...pack, assets_by_id: undefined }],
            active_pack: { ...pack, assets_by_id: undefined }
          }
        else if (path.endsWith("/visual-packs/pack-1")) data = pack
        else if (path.includes("/live/sessions"))
          data = { sessions: [], focused_session_id: null }
        else if (path.includes("/catalog"))
          data = [
            { id: personaId, name: "Research Assistant", buddy_summary: buddy }
          ]
        else if (path.includes("/profiles/"))
          data = {
            id: personaId,
            version: 1,
            buddy_summary: buddy,
            voice_defaults: {},
            setup: {
              status: "completed",
              version: 1,
              current_step: "test",
              completed_steps: [
                "persona",
                "voice",
                "commands",
                "safety",
                "test"
              ]
            }
          }
        else if (path.endsWith("/session") && options?.method === "POST")
          data = { session_id: "session-research", persona_id: personaId }
        else if (path.includes("/sessions/session-research"))
          data = {
            session_id: "session-research",
            persona_id: personaId,
            turns: []
          }
        return { ok: true, status: 200, json: async () => data }
      }
    )
    const view = render(<SidepanelPersona shell="options" />)
    await screen.findByTestId("persona-memory-toggle")
    await screen.findByTestId("persona-resume-session-select")
    await screen.findByTestId("persona-buddy-drag-handle")
    await waitFor(() =>
      expect(
        usePersonaVisualRuntimeStore.getState().runtimeDiagnostics
          ?.packLoadStatus
      ).toBe("loaded")
    )
    const count = () => ({
      list: calls.filter((p) => p.endsWith("/visual-packs")).length,
      detail: calls.filter((p) => p.endsWith("/visual-packs/pack-1")).length,
      sessions: calls.filter((p) => p.includes("/live/sessions")).length
    })
    const initial = count()
    expect(initial.list).toBe(1)
    expect(initial.detail).toBe(1)
    expect(initial.sessions).toBe(1)
    fireEvent.click(screen.getByRole("button", { name: "Connect" }))
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1))
    const ws = MockWebSocket.instances[0]
    expect(ws.url).toBe("ws://persona.test/api/v1/persona/stream")
    expect(ws.protocols).toEqual(["bearer", "test"])
    expect(ws.readyState).toBe(MockWebSocket.CONNECTING)
    act(() => ws.emitOpen())
    expect(ws.readyState).toBe(MockWebSocket.OPEN)
    await screen.findByRole("button", { name: /Disconnect/ })
    const voiceStates = [
      "idle",
      "listening",
      "thinking",
      "speaking",
      "error",
      "idle"
    ]
    for (let i = 0; i < 24; i++) {
      mocks.voiceState = voiceStates[i % voiceStates.length]
      mocks.toolName = i % 2 ? "knowledge.search" : ""
      act(() =>
        ws.emitMessage(
          JSON.stringify({
            event: "assistant_delta",
            text_delta: `Answer ${i}. `
          })
        )
      )
      act(() =>
        usePersonaVisualRuntimeStore.getState().setOverride({
          personaId,
          sessionId: "session-research",
          state: i % 2 ? "thinking" : "error",
          reason: "lifecycle-diagnostic",
          expiresAt: Date.now() + 5000
        })
      )
      view.rerender(<SidepanelPersona shell="options" />)
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 250))
      })
    }
    const publishedVoiceStates = new Set(
      mocks.buddyShellContextSnapshots
        .filter(Boolean)
        .map((c) => c?.live_voice_state)
    )
    expect([...publishedVoiceStates].sort()).toEqual(
      [...new Set(voiceStates)].sort()
    )
    expect(
      mocks.buddyShellContextSnapshots.some(
        (c) => c?.active_tool_name === "knowledge.search"
      )
    ).toBe(true)
    expect(count()).toEqual(initial)
    expect(screen.getByTestId("persona-buddy-drag-handle")).toBeInTheDocument()
    expect(
      usePersonaVisualRuntimeStore.getState().runtimeDiagnostics?.packLoadStatus
    ).toBe("loaded")
    // A real route availability change is the positive control: it must unmount
    // the host and then load exactly one replacement when connectivity returns.
    mocks.isOnline = false
    view.rerender(<SidepanelPersona shell="options" />)
    await waitFor(() =>
      expect(
        screen.queryByTestId("persona-buddy-drag-handle")
      ).not.toBeInTheDocument()
    )
    expect(count()).toEqual(initial)
    mocks.isOnline = true
    view.rerender(<SidepanelPersona shell="options" />)
    await screen.findByTestId("persona-buddy-drag-handle")
    await waitFor(() =>
      expect(
        usePersonaVisualRuntimeStore.getState().runtimeDiagnostics
          ?.packLoadStatus
      ).toBe("loaded")
    )
    expect(count()).toEqual({ list: 2, detail: 2, sessions: 2 })
    view.unmount()
    portal.remove()
  })
})
